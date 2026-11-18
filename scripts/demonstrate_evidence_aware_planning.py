#!/usr/bin/env python3
"""Compare three controllers on the admitted generated post-access fixtures."""
from __future__ import annotations

import argparse
from dataclasses import asdict
from itertools import combinations
import json
from pathlib import Path
import resource
import time

import numpy as np

from demonstrate_approach_procedure import _write_json
from run_approach_procedure_baselines import TIERS, _file_sha, _summary, conditional_controller
from resectionlab.core import semantic_digest
from resectionlab.geometry import GeometryScene, capsule_voxel_indices_batch
from resectionlab.imaging import load_case, read_case_artifacts, save_case
from resectionlab.native_backend import backend_identity
from resectionlab.surgical_engine import SurgicalEngine
from resectionlab.surgical_procedure_policy import evaluate_procedure_sequence
from resectionlab.surgical_recorded_sequential import display_case
from resectionlab.surgical_scene import from_case
from resectionlab.surgical_sequential_benchmark import ReplayVerificationError, run_episode, verify_replay
from resectionlab.surgical_sequential_task import generated_task, training_admission


ROOT = Path(__file__).resolve().parents[1]
CASES = (('gated-work', 200), ('critical-residual', 201), ('permitted-overlap', 202))
METHODS = ('conditional', 'harm-aware', 'search')
TIER = TIERS[1]
CAMPAIGN_SECONDS = 420.
REPLAY_SECONDS = 30.
STRUCTURES = ('brainstem', 'cerebellum', 'corticospinal', 'motor-cortex')
NEURAL = 'neuralRemovedMm3'
STRUCTURE = {name: f'structure:{name}:removedMm3' for name in STRUCTURES}
REQUIRED_HARMS = (NEURAL, *STRUCTURE.values())
PREFERENCE_HARMS = (STRUCTURE['corticospinal'],)
FROZEN_REPORT_SHA256 = '67accb0b99d3c93dbb90f2592f8cc1ccff7d851dbc90983c0f05cdbdd2c3c4d0'
FROZEN_PROTOCOL_ID = 'sha256:84732c04c3048761ea6972f7b9f82ffccc9d2af46a0408351d42bb33f7678f1a'


class SourceContactController:
    """Screen candidate groups using current nominal contacts and named limits."""

    def __init__(self):
        self.decisions = []

    def __call__(self, observed, skills, mask):
        from resectionlab.surgical_treatment_intent import decide_treatment_intent

        labels = observed['source_labels']
        grid = GeometryScene(labels, observed['affine'])
        voxel_mm3 = abs(float(np.linalg.det(observed['affine'][:3, :3])))
        named = observed['memberships']
        membership = {NEURAL: named['neural'], **{
            metric: named['structure:' + name] for name, metric in STRUCTURE.items()
        }}
        limits = observed['declared_limits']
        cap = {NEURAL: limits['maximumNeuralRemovalMm3'], **{
            metric: limits['maximumStructureRemovalMm3'][name] for name, metric in STRUCTURE.items()
        }}
        target = set(map(int, np.flatnonzero(labels == 1)))
        removed = set(observed['native_state']['removed'])
        remaining = target - removed
        current = observed['native_state']['slots']['right']['parameters'].get('radiusMm', 1.5)
        preparation = {s.radius: i for i, s in enumerate(skills) if mask[i] and s.kind == 'prepare'}
        work = [i for i, s in enumerate(skills) if mask[i] and s.kind == 'work']
        stop = next(i for i, s in enumerate(skills) if s.kind == 'stop')
        critical = {name: named['structure:' + name] for name in ('brainstem', 'cerebellum')}

        def harms(cells):
            return {metric: len(cells & members) * voxel_mm3 for metric, members in membership.items()}

        alternatives = []
        programs = {}
        evidence = {}
        remaining_skills = observed['remaining']['skills']
        queries = [(radius, index) for radius in (current, *sorted(preparation)) for index in work]
        contacts = capsule_voxel_indices_batch(grid, [(*skills[i].route, radius) for radius, i in queries])
        estimates = {}
        for (radius, index), indices in zip(queries, contacts):
            cells = {int(np.ravel_multi_index(ijk, labels.shape)) for ijk in indices}
            occupied = {c for c in cells - removed if labels.flat[c] != 0}
            blocking = {name for name, members in critical.items() if cells & members}
            totals = harms(removed | occupied)
            blocking.update(name for name, value in totals.items() if value > cap[name] + 1e-9)
            projected = occupied & remaining
            if projected:
                estimates[radius, index] = (occupied, blocking)

        # A source cell is explained by a single-route bound only when every
        # observed radius/route that would touch it has an applicable bound.
        for cell in sorted(remaining):
            touch = [blockers for cells, blockers in estimates.values() if cell in cells]
            if touch and all(touch):
                for name in set().union(*touch):
                    evidence.setdefault(name, set()).add(cell)

        for radius in (current, *sorted(preparation)):
            prep = () if radius == current else (preparation[radius],)
            viable_indices = [index for index in work if (radius, index) in estimates]
            for length in range(1, min(len(viable_indices), remaining_skills - len(prep) - 1) + 1):
                for group in combinations(viable_indices, length):
                    program = (*prep, *group)
                    future = set().union(*(estimates[radius, i][0] for i in group))
                    buckets = harms(removed | future)
                    blockers = set().union(*(estimates[radius, i][1] for i in group))
                    blockers.update(name for name, value in buckets.items() if value > cap[name] + 1e-9)
                    key = '+'.join(skills[i].id for i in program)
                    alternatives.append(dict(id=key, treatmentMm3=len(remaining & future) * voxel_mm3,
                        harms=buckets, admissible=not blockers, blockingBoundaries=sorted(blockers)))
                    programs[key] = program

        incumbent = dict(id='checked-stop', treatmentMm3=0.,
                         harms=harms(removed), admissible=True, blockingBoundaries=[])
        allocated = set()
        boundaries = []
        for name, cells in sorted(evidence.items(), key=lambda item: (item[0] not in critical, item[0])):
            explained = cells - allocated
            boundaries.append(dict(id=name, label=name, residualMm3=len(explained) * voxel_mm3))
            allocated.update(explained)
        uncovered = remaining - allocated
        residual = dict(residualMm3=len(remaining) * voxel_mm3,
            eligibleResidualMm3=len(uncovered) * voxel_mm3,
            unexplainedResidualMm3=len(uncovered) * voxel_mm3, boundaries=boundaries)
        coverage_complete = named['intracranial'] <= named['observed']
        decision = decide_treatment_intent(alternatives, incumbent=incumbent,
            required_harms=REQUIRED_HARMS, preference_harms=PREFERENCE_HARMS,
            residual=residual, coverage_complete=coverage_complete)
        index = stop
        if decision['status'] == 'continue':
            index = programs[decision['selectedId']][0]
        self.decisions.append(dict(decision=decision, selectedSkillId=skills[index].id,
            availableSkillIds=[s.id for i, s in enumerate(skills) if mask[i]],
            selectionIndex=len(self.decisions), alternatives=alternatives, incumbent=incumbent,
            residual=residual, coverageComplete=coverage_complete, capsuleQueries=len(queries)))
        return index


def frozen_baseline(path):
    path = path.resolve()
    protocol = json.loads((path / 'protocol.json').read_text())
    report = json.loads((path / 'report.json').read_text())
    if (_file_sha(path / 'report.json') != FROZEN_REPORT_SHA256 or
            report['protocolIdentity'] != FROZEN_PROTOCOL_ID or protocol['identity'] != FROZEN_PROTOCOL_ID or
            len(report['rows']) != 40 or report['missingRows'] or report['optimizerUpdates'] != 0):
        raise ValueError('Historical baseline receipt changed')
    return dict(identity=FROZEN_PROTOCOL_ID, reportSha256=FROZEN_REPORT_SHA256,
                protocolSha256=_file_sha(path / 'protocol.json'), completedRows=len(report['rows']))


def freeze(out, baseline):
    source_paths = sorted(path for path in (ROOT / 'src/resectionlab').glob('surgical_*.py')
                          if path.name != 'surgical_planning_evidence.py')
    source_paths.extend(ROOT / 'src/resectionlab' / f'{name}.py'
                        for name in ('core', 'geometry', 'imaging', 'native_backend'))
    source_paths.extend(ROOT / 'scripts' / f'{name}.py' for name in (
        'demonstrate_evidence_aware_planning', 'demonstrate_approach_procedure',
        'run_approach_procedure_baselines'))
    identity = backend_identity()
    cases = []
    for label, parent in CASES:
        task = generated_task(parent)
        cases.append(dict(id=label, role='DEVELOPMENT', source=task.audit, admission=training_admission(task),
            inventoryHash=semantic_digest([asdict(skill) for skill in task.skills]),
            policyHash=task.initial_engine().snapshot()['procedurePolicy']['policyHash']))
    protocol = dict(schema='resectionlab.evidence-aware-generated-planning.v1',
        cases=cases, methods=METHODS, tier=asdict(TIER), seed=0, campaignSeconds=CAMPAIGN_SECONDS,
        replaySecondsPerMethod=REPLAY_SECONDS, scenarios=['nominal'],
        historicalBaseline=baseline, sourceHashes={str(path.relative_to(ROOT)): _file_sha(path)
                                                 for path in source_paths},
        nativeBackend={key: identity[key] for key in ('selected', 'binarySha256', 'nativeThreads')},
        nomination='Current observed source contacts and named memberships with zero transition previews.',
        scope='Generated supplied post-access engineering controls. Earlier access costs and clinical outcomes are unknown.',
        negativeCoveragePolicy='Abstentions remain distinct from an explained treatment stop.',
        requiredHarmMetrics=REQUIRED_HARMS, preferenceHarmMetrics=PREFERENCE_HARMS,
        optimizerUpdates=0, clinicalValidation=False)
    protocol['identity'] = semantic_digest(protocol)
    _write_json(out / 'protocol.json', protocol)
    return protocol


def execute(case, task, method, deadline):
    selector = SourceContactController() if method == 'harm-aware' else conditional_controller
    row = run_episode(task, 'conditional' if method == 'harm-aware' else method,
        TIER, retain_engine=True, selector=selector if method != 'search' else None, deadline=deadline)
    engine = row.pop('_engine')
    row.update(method=method, case=case, voxelMm3=engine.scene.voxel_mm3,
               sequence=evaluate_procedure_sequence(engine)['outcome'])
    if method == 'harm-aware':
        state_before = task.initial_engine().state_id
        for index, decision in enumerate(selector.decisions):
            decision['sourceStateId'] = state_before
            if index == len(row['skillHistory']):
                decision['executionStatus'] = 'unexecuted'
                break
            attempt = row['skillHistory'][index]
            if decision['selectedSkillId'] != attempt['skillId']:
                raise ValueError('Intent selection and physical request diverged')
            decision['finalStateId'] = attempt['finalStateId']
            decision['checkedStopAttempted'] = attempt['checkedStopAttempted']
            decision['executionStatus'] = attempt['status']
            state_before = attempt['finalStateId']
        row['intentDecisions'] = selector.decisions
    return row, engine


def archive(out, case_id, task, rows, engines, deadline):
    from resectionlab import surgical_planning_evidence

    case = display_case(task)
    scene = task.scene
    initial = task.initial_engine()
    branches = {'baseline': initial, **engines}
    preferred = 'harm-aware'
    record = dict(mapping=scene.regions, branches={name: engine.snapshot() for name, engine in branches.items()},
        branchId=preferred, position=len(engines[preferred].actions),
        parents={name: None if name == 'baseline' else {'branchId': 'baseline', 'position': 0}
                 for name in branches},
        timeline={name: engine.history for name, engine in branches.items()})
    record['planningEvidence'] = surgical_planning_evidence.planning_evidence(
        scene, record, rows, baseline_branch='baseline')
    path = out / (case_id + '.ressectionlab')
    save_case(case, path, artifacts={'surgical_toolbox': record})
    loaded = load_case(path)
    saved = read_case_artifacts(path)['surgical_toolbox']
    cold_scene = from_case(loaded, saved['mapping'])
    if cold_scene.identity != scene.identity or loaded.semantic_hash != case.semantic_hash:
        raise ValueError('Portable source identity changed')
    cold_branches = {}
    def check():
        if time.monotonic() >= deadline:
            raise TimeoutError('Campaign deadline during archive cold replay')

    for name, engine in branches.items():
        check()
        cold = SurgicalEngine.restore(cold_scene, saved['branches'][name], use_cache=False,
                                      force_verify=True, check=check)
        if cold.snapshot() != engine.snapshot() or cold.history != engine.history:
            raise ValueError('Portable accepted replay changed for ' + name)
        cold_branches[name] = dict(stateId=cold.state_id, nativeActions=len(cold.actions),
            snapshotHash=semantic_digest(cold.snapshot()), historyHash=semantic_digest(cold.history))
    return dict(case=case_id, package=path.name, packageSha256=_file_sha(path),
        caseHash=case.semantic_hash, sourceSceneId=scene.identity,
        policyHash=initial.snapshot()['procedurePolicy']['policyHash'], branches=cold_branches,
        exportEvidenceImplementationSha256=_file_sha(Path(surgical_planning_evidence.__file__)))


def check_result(report):
    rows = {(row['case'], row['method']): row for row in report['rows']}
    if len(rows) != len(CASES) * len(METHODS):
        raise ValueError('Matched result incomplete')
    overlap = rows['permitted-overlap', 'harm-aware']
    legacy = rows['permitted-overlap', 'conditional']
    equal_fields = ('originalMm3', 'treatmentMm3', 'residualMm3', 'wholeToolExit', 'checkedStop')
    harm_metric = 'structureRemovedMm3:corticospinal'
    result = dict(
        overlapMatchedTreatment=all(overlap[k] == legacy[k] for k in equal_fields)
            and overlap['treatmentMm3'] > 0,
        overlapLessCst=overlap['harm'][harm_metric] < legacy['harm'][harm_metric]
            and all(value == legacy['harm'][key] for key, value in overlap['harm'].items() if key != harm_metric),
        continuedGatedWork=rows['gated-work', 'harm-aware']['residualMm3'] == 0.,
        justifiedResidual=rows['critical-residual', 'harm-aware']['residualMm3'] > 0
            and rows['critical-residual', 'harm-aware']['checkedStop']
            and rows['critical-residual', 'harm-aware']['wholeToolExit']
            and rows['critical-residual', 'harm-aware']['intentExecution'] == 'justified_residual_stop',
        allPhysicalReplaysVerified=all(row['replayVerified'] for row in rows.values()),
        allArchiveColdReplaysVerified=len(report['packages']) == len(CASES),
        missingCoverageAbstained=report['decisionControls']['missingCoverage']['status'] == 'reassess',
        tradeoffAbstained=report['decisionControls']['unresolvedTradeoff']['status'] == 'reassess',
    )
    report['checks'] = result
    return all(result.values())


def decision_controls():
    from resectionlab.surgical_treatment_intent import decide_treatment_intent

    empty = dict.fromkeys(REQUIRED_HARMS, 0.)
    incumbent = dict(id='checked-stop', treatmentMm3=0., harms=empty, admissible=True)
    cst = STRUCTURE['corticospinal']
    unknown = dict(id='missing-cst-coverage', treatmentMm3=27., harms={**empty, cst: None},
                   admissible=None)
    lesser = dict(id='lesser-treatment', treatmentMm3=27., harms=empty, admissible=True)
    greater = dict(id='greater-treatment', treatmentMm3=54., harms={**empty, cst: 27.},
                   admissible=True)
    return {
        'missingCoverage': decide_treatment_intent([unknown], incumbent=incumbent,
            required_harms=REQUIRED_HARMS, preference_harms=PREFERENCE_HARMS),
        'unresolvedTradeoff': decide_treatment_intent([lesser, greater], incumbent=incumbent,
            required_harms=REQUIRED_HARMS, preference_harms=PREFERENCE_HARMS, coverage_complete=True),
    }


def demonstrate(out, historical):
    from resectionlab.surgical_treatment_intent import assess_treatment_execution

    started = time.monotonic()
    deadline = started + CAMPAIGN_SECONDS
    frozen = frozen_baseline(historical)
    protocol = freeze(out, frozen)
    report = dict(schema=protocol['schema'], protocolIdentity=protocol['identity'],
        protocolSha256=_file_sha(out / 'protocol.json'), historicalBaseline=frozen,
        rows=[], packages=[], decisionControls=decision_controls(), failures=[], passed=False)
    try:
        archive_inputs = []
        for case_id, parent in CASES:
            task = generated_task(parent)
            rows = {}
            engines = {}
            for method in METHODS:
                if time.monotonic() >= deadline:
                    raise TimeoutError('Campaign deadline before matched row')
                print(f'{case_id} {method}', flush=True)
                row, engines[method] = execute(case_id, task, method, deadline)
                rows[method] = row
                path = out / f'{case_id}-{method}.json'
                _write_json(path, row)
                try:
                    row['verification'] = verify_replay(task, row, deadline=min(deadline, time.monotonic() + REPLAY_SECONDS))
                except ReplayVerificationError as error:
                    row['verification'] = error.receipt
                decisions = row.get('intentDecisions', [])
                if decisions:
                    replay = row['verification']
                    row['intentExecution'] = assess_treatment_execution(decisions[-1]['decision'],
                        stop_outcome=replay.get('adapterStopOutcome') if replay['verified'] else None,
                        terminated=row['endpoint']['terminated'], truncated=row['endpoint']['truncated'],
                        termination_reason=replay.get('adapterTerminationReason'))
                _write_json(path, row)
                summary = _summary(row)
                summary.update(receipt=path.name, receiptSha256=_file_sha(path),
                    intent=decisions[-1]['decision']['status'] if decisions else None,
                    intentExecution=row.get('intentExecution', {}).get('executionStatus'),
                    abstentions=sum(item['decision']['status'] == 'reassess' for item in decisions),
                    truncated=row['endpoint']['truncated'], candidateGenerations=row['computation']['candidateGenerations'],
                    capsuleQueries=sum(item['capsuleQueries'] for item in decisions))
                report['rows'].append(summary)
                _write_json(out / 'report.json', report)
            archive_inputs.append((case_id, task, rows, engines))
        for archived in archive_inputs:
            report['packages'].append(archive(out, *archived, deadline))
            _write_json(out / 'report.json', report)
        if frozen_baseline(historical) != frozen:
            raise ValueError('Historical baseline changed during evaluation')
        report['passed'] = check_result(report)
    except Exception as error:
        report['failures'].append(dict(type=type(error).__name__, message=str(error)))
        raise
    finally:
        report['missingRows'] = [dict(case=case, method=method) for case, _ in CASES for method in METHODS
            if not any(row['case'] == case and row['method'] == method for row in report['rows'])]
        report['wallSeconds'] = time.monotonic() - started
        report['peakRssBytes'] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        _write_json(out / 'report.json', report)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--frozen-baseline', type=Path,
        default=ROOT / 'testcases/approach-comparison-goal-v1/baseline-v1')
    args = parser.parse_args()
    out = args.output.resolve()
    if not out.is_relative_to(ROOT / 'testcases') or out == ROOT / 'testcases':
        parser.error('Keep generated evidence in a fresh ignored testcases/ directory')
    out.mkdir(parents=True, exist_ok=False)
    try:
        report = demonstrate(out, args.frozen_baseline)
    except Exception as error:
        if not (out / 'report.json').exists():
            _write_json(out / 'failure.json', dict(type=type(error).__name__, message=str(error)))
        raise
    print(json.dumps(dict(passed=report['passed'], report=str(out / 'report.json'))))
    return 0 if report['passed'] else 1


if __name__ == '__main__':
    raise SystemExit(main())
