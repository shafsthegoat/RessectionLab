"""Independent audit of frozen native command-translation stress studies.

No strategy planner or sensitivity predictor is imported. Native transitions are
replayed; literal CUSA/Frazier capsules and independent orthogonal-cell boxes
rebuild encounters, clearances, failures, progress and finite-world choices.
"""
from __future__ import annotations

import argparse
import copy
import itertools
import json
from pathlib import Path
import sys

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from verify_anatomical_encounter import (ROOT, IndependentFootprint, CaptureEngine,
    literal_capsules, axial_capsules, certify_trace, load_source_context, physical_hash,
    file_hash, semantic_digest, SurgicalEngine, load_environment, load_workspace)
from resectionlab.surgical_planning_observation import project_observation
from resectionlab.surgical_task_admission import admission

SCHEMA = 'resectionlab.two-tool-execution-sensitivity.v1'
PUBLIC_SCHEMA = 'resectionlab.public-two-tool-execution-sensitivity.v1'
SCOPE = ('Finite coherent command-offset stress against fixed observed anatomy. '
    'Both tools and withdrawal are replayed. Not measured patient uncertainty, '
    'anatomical registration, segmentation, deformation, clinical risk or a guarantee '
    'between tested offsets. Outside the supplied source grid remains unassessed. '
    'Selection covers only the frozen comparison programs; no new route search.')


def require(ok, message):
    if not ok:
        raise ValueError('INDEPENDENT_EXECUTION_SENSITIVITY: ' + message)


def same(actual, expected, message):
    require(semantic_digest(actual) == semantic_digest(expected), message)


def seal(value, field):
    require(type(value) is dict and value.get(field) == semantic_digest({k: v for k, v in value.items() if k != field}), field + ' differs')


def worlds(offset):
    require(type(offset) in (int, float) and np.isfinite(offset) and 0. <= offset <= 2., 'invalid finite offset')
    result = [('nominal', [0., 0., 0.])]
    if offset:
        for dimension, name in enumerate('RAS'):
            for sign in (-1, 1):
                delta = [0., 0., 0.]; delta[dimension] = sign * float(offset)
                result.append((name + ('-' if sign < 0 else '+'), delta))
    return result


def translate(action, delta):
    require(type(action) is dict and not action.get('path'), 'only canonical single-pose actions are supported')
    shifted = copy.deepcopy(action)
    if 'pose' in shifted and any(delta):
        shifted['pose']['position'] = (np.asarray(action['pose']['position'], float) + delta).tolist()
    return shifted


def summary(engine, before):
    labels = engine.scene.labels.ravel()
    targets = {r['id'] for r in engine.scene.regions if r.get('target') and r['material'] == 'tumor'}
    new = sorted(set(engine.state['removed']) - set(before['removed']))
    target_count = sum(int(labels[c]) in targets for c in new)
    retained = np.isin(labels, list(targets)); retained[engine.state['removed']] = False
    cusa, old = engine.state['cusaV3'], before['cusaV3']
    volume = engine.scene.voxel_mm3
    return {'additionalTargetMm3': target_count * volume, 'additionalNonTargetMm3': (len(new) - target_count) * volume,
        'removedSourceCells': new, 'residualTargetMm3': int(retained.sum()) * volume,
        'elapsedSeconds': engine.state['elapsedSeconds'] - before['elapsedSeconds'],
        'cusaActivationSeconds': cusa['effectiveActivationSeconds'] - old['effectiveActivationSeconds'],
        'cusaInputCapacityMm3': cusa['inputCapacityMm3'] - old['inputCapacityMm3'],
        'cusaPendingMm3': sum(cusa['progressMm3'].values()),
        'fluidRemovedMm3': engine.state['fluidRemovedMm3'] - before['fluidRemovedMm3']}


def burden_reason(engine, before, controls):
    value = summary(engine, before)
    if value['additionalNonTargetMm3'] > 0.: return 'TWO_TOOL_TASK: new non-target removal exceeds task'
    if value['additionalTargetMm3'] > 8.: return 'TWO_TOOL_TASK: target manipulation exceeds task'
    if value['cusaActivationSeconds'] > 3. + 1e-10: return 'TWO_TOOL_TASK: CUSA activation exceeds task'
    new_work = [int(k) for k, v in engine.state['cusaV3']['progressMm3'].items() if v > before['cusaV3']['progressMm3'].get(k, 0.)]
    labels = engine.scene.labels.ravel()
    targets = {r['id'] for r in engine.scene.regions if r.get('target') and r['material'] == 'tumor'}
    if any(int(labels[c]) not in targets for c in new_work): return 'TWO_TOOL_TASK: CUSA work reached non-target source tissue'
    selected = sorted(set(value['removedSourceCells']) | set(new_work))
    if selected:
        voxels = np.asarray(np.unravel_index(selected, engine.scene.shape)).T
        points = voxels @ engine.scene.affine[:3, :3].T + engine.scene.affine[:3, 3]
        region = controls['geometry']['target']
        if np.any(np.linalg.norm(points - region['centerMm'], axis=1) > region['radiusMm'] + 1e-9):
            return 'TWO_TOOL_TASK: new removal escapes requested target region'
    return None


def covers(action, before, after, *, conservative=False, spacing=1.):
    for hand, slot in after.items():
        moving = hand == action['slot'] and action['operation'] in ('move', 'apply', 'withdraw')
        for a, b, radius, part in (axial_capsules(before[hand], slot) if moving else literal_capsules(slot)):
            yield a, b, radius + (min(.4, spacing / 3.) / 2. if conservative and moving else 0.), part


def clearance_reason(engine, action, before, controls, extra, *, fixed, checker=None, shaft_checker=None):
    """Independent box clearance; protected masks retain conservative spheres."""
    if not extra and fixed: return None
    margin = controls['geometry']['nonTargetClearanceMm'] + extra
    if not fixed and not margin: return None
    if checker is None:
        targets = [r['id'] for r in engine.scene.regions if r.get('target') and r['material'] == 'tumor']
        mask = (engine.scene.labels != 0) & ~np.isin(engine.scene.labels, targets)
        checker = IndependentFootprint(mask, engine.scene.affine, [], {'marginMm': 0., 'maxNearMm3': 1000.})
    if fixed and shaft_checker is None:
        shaft_checker = IndependentFootprint(engine.scene.labels != 0, engine.scene.affine, [],
            {'marginMm': 0., 'maxNearMm3': 1000.})
    checker.seen = {'shaft': set(), 'working': set()}
    spacing = float(np.linalg.norm(engine.scene.affine[:3, :3], axis=0).min())
    # The ordinary task's non-target clearance retains its conservative half-
    # microstep inflation. The fixed comparator explicitly uses nominal union.
    for a, b, radius, part in covers(action, before, engine.state['slots'], conservative=not fixed, spacing=spacing):
        if fixed and part == 'shaft' and shaft_checker._cells(a, b, radius + extra) - set(engine.state['removed']):
            return 'STRATEGY_SENSITIVITY: FIXED_MARGIN_SHAFT_CLEARANCE'
        hits = checker._cells(a, b, radius + margin) - set(engine.state['removed'])
        if hits:
            return ('STRATEGY_SENSITIVITY: FIXED_MARGIN_NON_TARGET_CLEARANCE' if fixed else
                    'TWO_TOOL_TASK: mapped non-target clearance violated during the complete sequence')
        if fixed and engine.protection is not None:
            cells = np.asarray(engine.protection.cells, dtype=np.intp)
            if len(cells):
                voxels = np.asarray(np.unravel_index(cells, engine.scene.shape)).T
                points = voxels @ engine.scene.affine[:3, :3].T + engine.scene.affine[:3, 3]
                direction = b - a; squared = float(direction @ direction)
                t = np.clip((points - a) @ direction / squared, 0., 1.) if squared else np.zeros(len(points))
                corners = np.asarray(list(itertools.product((-.5, .5), repeat=3)))
                cell_radius = float(np.linalg.norm(corners @ engine.scene.affine[:3, :3].T, axis=1).max())
                bound = radius + extra + engine.protection.clearance_mm + cell_radius + 1e-9
                if np.any(np.linalg.norm(points - a - t[:, None] * direction, axis=1) <= bound):
                    return 'STRATEGY_SENSITIVITY: FIXED_MARGIN_PROTECTED_CLEARANCE'
    return None


def replay(baseline, row, controls, mask, affine, scenario_id, delta, *, fixed_extra=None, check=lambda: None):
    before = copy.deepcopy(baseline.state)
    engine = baseline.clone(); engine.__class__ = CaptureEngine; engine._audit_trace = []
    tracker = IndependentFootprint(mask, affine, before['removed'], controls['encounter'])
    expanded = IndependentFootprint(mask, affine, before['removed'], {**controls['encounter'],
        'marginMm': controls['encounter']['marginMm'] + fixed_extra}) if fixed_extra is not None else None
    clearance_checker = shaft_checker = None
    if controls['geometry']['nonTargetClearanceMm'] or fixed_extra:
        targets = [r['id'] for r in baseline.scene.regions if r.get('target') and r['material'] == 'tumor']
        non_target = (baseline.scene.labels != 0) & ~np.isin(baseline.scene.labels, targets)
        clearance_checker = IndependentFootprint(non_target, affine, [], {'marginMm': 0., 'maxNearMm3': 1000.})
        if fixed_extra:
            shaft_checker = IndependentFootprint(baseline.scene.labels != 0, affine, [],
                {'marginMm': 0., 'maxNearMm3': 1000.})
    actions = [translate(a, delta) for a in row['actionProgram']]
    require(len(actions) <= 24 and sum(a['duration'] for a in actions) <= 10. + 1e-10, 'unbounded frozen candidate')
    receipts = []; physical = []; context = []; phases = []; failure = fixed_failure = probe = None
    reassess = next((p for p in row['phases'] if p['id'] == 'reassess'), None)
    for index, action in enumerate(actions):
        check()
        previous = copy.deepcopy(engine.state['slots']); old_snapshot = engine.snapshot(); engine._audit_trace = []
        result = engine.step(copy.deepcopy(action), check=check)
        if result['status'] == 'rejected':
            require(not engine._audit_trace and engine.snapshot() == old_snapshot, 'rejected action leaked a committed prefix')
        else:
            certify_trace(action, previous, engine.state['slots'], engine._audit_trace)
        receipts.append({k: result.get(k) for k in ('status', 'reason', 'completedFraction', 'elapsedSeconds')})
        physical.append(physical_hash(engine))
        tracker.observe(action, previous, engine.state['slots'], engine.state['removed'])
        context.append(semantic_digest(tracker.receipt(engine.state['removed'])))
        if expanded: expanded.observe(action, previous, engine.state['slots'], engine.state['removed'])
        finite = action['slot'] == 'left' and action['operation'] == 'apply' and result['status'] == 'partial' and result.get('reasonCode') == 'TISSUE_ASPIRATION_IN_PROGRESS' and abs(result['elapsedSeconds'] - action['duration']) <= 1e-8
        reason = ('STRATEGY_SENSITIVITY: execution refused: ' + str(result.get('reason'))
                  if result['status'] != 'accepted' and not finite else burden_reason(engine, before, controls))
        if reason is None:
            reason = clearance_reason(engine, action, previous, controls, 0., fixed=False, checker=clearance_checker)
        if reason:
            failure = {'actionIndex': index, 'operation': action['operation'],
                'phase': next((p['id'] for p in row['phases'] if p['startAction'] <= index < p['endAction']), None), 'reason': reason}
            break
        if expanded and fixed_failure is None:
            reason = clearance_reason(engine, action, previous, controls, fixed_extra, fixed=True,
                checker=clearance_checker, shaft_checker=shaft_checker)
            if reason: fixed_failure = {'actionIndex': index, 'reason': reason}
        if reassess and index + 1 == reassess['startAction'] + 2:
            probe_action = translate(row['reassessment']['probeAction'], delta)
            trial = engine.clone(); result = trial.step(copy.deepcopy(probe_action), check=check)
            probe = {'action': probe_action, 'status': result['status'], 'reason': result.get('reason')}
        for phase in row['phases']:
            if phase['endAction'] == index + 1:
                phases.append({'id': phase['id'], 'endAction': index + 1, 'physicalStateHash': physical[-1], 'contextHash': context[-1]})
    complete = failure is None and len(receipts) == len(actions)
    withdrawn = all(not s['active'] and not s['inserted'] and not s['grip'] for s in engine.state['slots'].values())
    value = summary(engine, before); encounter = tracker.receipt(engine.state['removed'])
    progress = value['additionalTargetMm3'] >= controls['minimumTargetMm3']
    violations = ([] if complete else ['INCOMPLETE_EXECUTION']) + ([] if withdrawn else ['INCOMPLETE_WITHDRAWAL'])
    if not encounter['withinLimit']: violations.append('ANATOMICAL_CONTEXT_LIMIT')
    if not progress: violations.append('PROGRESS_THRESHOLD')
    outcome = {'scenarioId': scenario_id, 'translationRasMm': list(delta), 'actionProgram': actions,
        'actionReceipts': receipts, 'physicalStateHashes': physical, 'contextPrefixHashes': context, 'phases': phases,
        'failure': failure, 'completeExecution': complete, 'withdrawalComplete': withdrawn, 'summary': value,
        'anatomicalEncounter': encounter, 'reassessment': probe, 'taskProgressMet': progress,
        'violations': violations, 'meetsTask': not violations}
    if expanded: outcome.update(fixedMarginEncounter=expanded.receipt(engine.state['removed']), fixedMarginFailure=fixed_failure)
    require(SurgicalEngine.restore(engine.scene, engine.snapshot()).snapshot() == engine.snapshot(), 'failed/successful native prefix does not cold-replay')
    return outcome


def choose(candidates, index=0, *, fixed=False, every=False):
    eligible = []
    for candidate in candidates:
        outcome = candidate['outcomes'][index]
        qualifies = all(o['meetsTask'] for o in candidate['outcomes']) if every else outcome['meetsTask']
        if fixed: qualifies = qualifies and outcome['fixedMarginEncounter']['withinLimit'] and outcome['fixedMarginFailure'] is None
        if qualifies:
            eligible.append((outcome['summary']['elapsedSeconds'], -outcome['summary']['additionalTargetMm3'], candidate['id']))
    return min(eligible)[2] if eligible else 'stop'


def verify(study, comparison_report, source_baseline_engine, *, check=lambda: None):
    baseline = source_baseline_engine; original_snapshot = baseline.snapshot()
    require(type(baseline) is SurgicalEngine, 'authoritative native baseline engine required')
    report = comparison_report; seal(report, 'reportSeal'); plan = report['publicPlan']; seal(plan, 'publicPlanSeal')
    require(report['schema'] == 'resectionlab.two-tool-comparison.v2', 'source-context comparison required')
    require(baseline.scene.identity == report['sourceSceneId'] and baseline.state_id == report['startStateId'] and
        semantic_digest(original_snapshot) == report['baselineSnapshotHash'] and semantic_digest(baseline.actions) == report['baselineActionHash'] and
        semantic_digest(baseline.history) == report['baselineHistoryHash'] and len(baseline.actions) == report['prefixActions'], 'native baseline binding differs')
    require(set(baseline.scene.properties) <= {'tissuePhysicsV2', 'cusaV3'} and 'tissuePhysicsV2' in baseline.scene.properties and
        baseline.state.get('toolMechanicsRevision') == 2 and baseline.state.get('physicalClockRevision') == 1, 'unsupported native model/profile')
    require(not baseline.state['holds'] and not baseline.state.get('deformation') and not baseline.state.get('detachedFragments') and
        'patientPosition' not in baseline.state and all(not s['active'] and not s['inserted'] and not s['grip'] for s in baseline.state['slots'].values()), 'inactive released undeformed baseline required')
    require(not any(baseline.state.get('deformationModel', {}).get(k) for k in ('active', 'anchors', 'velocities')), 'pending mechanics unsupported')
    mask, affine = load_source_context(baseline, report)
    public = project_observation(baseline, admission(baseline)['task'], report['sourceEvidence']['availableEvidence']).manifest()
    body = {'schema': 'resectionlab.two-tool-observation.v2', 'baseline': public,
        'taskHash': semantic_digest(plan['task']), 'anatomicalContext': report['contextBinding']['publicObservation']}
    manifest = {**body, 'inputHash': semantic_digest(body), 'evaluationReferencesExposed': False}
    same(plan['publicObservation'], manifest, 'comparison public observation differs')
    seal(study, 'studySeal')
    require(set(study) == {'schema', 'comparisonReportSeal', 'contextBindingSeal', 'assessment', 'nativeReplayVerified', 'studySeal'} and
        study['schema'] == SCHEMA and study['comparisonReportSeal'] == report['reportSeal'] and
        study['contextBindingSeal'] == report['contextBinding']['bindingSeal'] and study['nativeReplayVerified'] is True, 'study source bindings or native proof declaration differ')
    assessment = study['assessment']; seal(assessment, 'assessmentSeal')
    require(type(assessment.get('controls')) is dict and set(assessment['controls']) == {'offsetMm'}, 'exact finite stress controls required')
    offset = assessment['controls']['offsetMm']; scenarios = worlds(offset); controls = plan['constraints']
    require(controls['encounter']['marginMm'] + offset <= 5., 'expanded context margin exceeds audited bound')
    config = {'family': 'nominal-and-six-RAS-axis-command-translations-v1', 'offsetMm': float(offset), 'frame': 'RAS+', 'units': 'mm',
        'appliedTo': 'all commanded instrument poses; fixed observed anatomy', 'parameterBasis': 'declared-uncalibrated-execution-stress',
        'distribution': None, 'patientCalibration': None}
    same(assessment.get('scenarioConfiguration'), config, 'scenario configuration differs')
    require(type(assessment.get('scenarioCount')) is int and assessment['scenarioCount'] == len(scenarios), 'scenario count differs')
    require(type(plan['alternatives']) is list and 1 <= len(plan['alternatives']) <= 3 and
        len({r['id'] for r in plan['alternatives']}) == len(plan['alternatives']), 'invalid frozen candidate family')
    model = {'case_hash': manifest['inputHash'], 'geometry': manifest, 'objectives': controls, 'tools': plan['task'],
        'world_generator': config, 'action_primitives': {r['id']: r['actionProgram'] for r in plan['alternatives']}, 'planning_as_of': None}
    same(assessment.get('frozenDecisionModel'), {'fingerprint': semantic_digest(model), **model}, 'frozen decision model differs')
    require(type(assessment.get('candidates')) is list and [r['id'] for r in assessment['candidates']] ==
        [r['id'] for r in plan['alternatives']], 'candidate family omitted, reordered or duplicated')
    expected = []
    for row, observed in zip(plan['alternatives'], assessment['candidates']):
        require(set(observed) == {'id', 'outcomes'} and type(observed['outcomes']) is list and len(observed['outcomes']) == len(scenarios), 'scenario matrix omitted or duplicated')
        outcomes = []
        for (scenario_id, delta), recorded in zip(scenarios, observed['outcomes']):
            check()
            same(recorded.get('actionProgram'), [translate(a, delta) for a in row['actionProgram']], 'held program or coherent translation differs')
            value = replay(baseline, row, controls, mask, affine, scenario_id, delta,
                fixed_extra=float(offset) if scenario_id == 'nominal' else None, check=check)
            same(recorded, value, 'independent native/context outcome differs: ' + row['id'] + '/' + scenario_id)
            if scenario_id == 'nominal':
                require(value['completeExecution'], 'nominal held program failed')
                same(value['actionReceipts'], row.get('actionReceipts', []), 'zero-offset action receipts differ')
                same(value['physicalStateHashes'], row.get('physicalStateHashes', []), 'zero-offset physical states differ')
                same(value['summary'], row['summary'], 'zero-offset progress differs')
                same(value['anatomicalEncounter'], row['anatomicalEncounter'], 'zero-offset encounter differs')
            outcomes.append(value)
        expected.append({'id': row['id'], 'outcomes': outcomes})
    nominal = choose(expected); fixed = choose(expected, fixed=True); all_worlds = choose(expected, every=True)
    require(nominal == plan['decision']['selectedId'], 'source comparison nominal decision differs')
    preferences = [{'scenarioId': sid, 'selectedId': choose(expected, i)} for i, (sid, _) in enumerate(scenarios)]
    expected_assessment = {'schema': PUBLIC_SCHEMA, 'publicObservation': manifest, 'comparisonPublicPlanSeal': plan['publicPlanSeal'],
        'controls': {'offsetMm': float(offset)}, 'frozenDecisionModel': {'fingerprint': semantic_digest(model), **model},
        'scenarioConfiguration': config, 'scenarioCount': len(scenarios), 'candidates': expected,
        'policies': {'nominal': nominal, 'fixedMargin': fixed, 'allDeclaredScenarios': all_worlds},
        'fixedMarginDefinition': {'nonTargetClearanceMm': controls['geometry']['nonTargetClearanceMm'] + offset,
            'contextMarginMm': controls['encounter']['marginMm'] + offset, 'protectedExtraMm': float(offset),
            'geometry': 'complete nominal axial tool union; all retained shaft cells, mapped non-target working cells and supplied protected/context cells'},
        'perScenarioPreference': preferences, 'preferenceChanges': [p for p in preferences if p['selectedId'] != nominal],
        'calibratedUncertainty': {'supported': False, 'reason': 'No patient-specific execution-error calibration is supplied'}, 'scope': SCOPE}
    expected_assessment['assessmentSeal'] = semantic_digest(expected_assessment)
    same(assessment, expected_assessment, 'independent policy, metadata or assessment seal differs')
    require(baseline.snapshot() == original_snapshot, 'audit mutated authoritative baseline')
    return {'schema': 'resectionlab.independent-execution-sensitivity-audit.v1', 'passed': True,
        'studySeal': study['studySeal'], 'comparisonReportSeal': report['reportSeal'], 'sourceContextBindingVerified': True,
        'baselineUnchanged': True, 'scenarioCount': len(scenarios), 'policies': expected_assessment['policies'],
        'candidates': [{'id': row['id'], 'outcomes': [{'scenarioId': o['scenarioId'], 'outcomeHash': semantic_digest(o),
            'completeExecution': o['completeExecution'], 'withdrawalComplete': o['withdrawalComplete'], 'meetsTask': o['meetsTask'],
            'failure': o['failure'], 'summary': o['summary'], 'anatomicalEncounter': o['anatomicalEncounter'],
            **({'fixedMarginFailure': o['fixedMarginFailure'], 'fixedMarginEncounter': o['fixedMarginEncounter']} if 'fixedMarginFailure' in o else {})}
            for o in row['outcomes']]} for row in expected],
        'scope': 'Independent frozen-command transforms, native transitions and failed prefixes, literal complete-tool source-box encounters, task burdens, finite-world policies. Nominal tissue mechanics only; no patient calibration, clinical risk, search optimality or guarantee between offsets.'}


def verify_workspace(workspace, *, source=None, revision=1, check=lambda: None):
    path = Path(workspace).resolve(); source = Path(source).resolve() if source else path
    require(path.is_relative_to(ROOT / 'testcases') and source.is_relative_to(ROOT / 'testcases'), 'inputs must stay in ignored testcases')
    hashes = {str(p): file_hash(p) for p in (path, source)}
    _, artifacts, _ = load_workspace(path); record = artifacts['surgical_toolbox']
    studies = record.get('strategySensitivityStudies', [])
    require(type(revision) is int and 1 <= revision <= len(studies), 'choose a saved sensitivity revision')
    item = studies[revision - 1]; seal(item, 'workspaceSeal')
    require(set(item) == {'schema', 'revision', 'comparisonRevision', 'study', 'workspaceSeal'} and
        item['schema'] == 'resectionlab.strategy-sensitivity-workspace.v1' and type(item['revision']) is int and item['revision'] == revision and
        type(item['comparisonRevision']) is int, 'saved sensitivity linkage schema differs')
    comparisons = [r for r in record.get('strategyComparisons', []) if r['revision'] == item['comparisonRevision']]
    require(len(comparisons) == 1, 'linked source comparison missing or duplicated')
    comparison = comparisons[0]; seal(comparison, 'workspaceSeal')
    snapshot = record['branches'][comparison['sourceBranchId']]
    require(semantic_digest(snapshot) == comparison['sourceSnapshotHash'], 'saved comparison baseline snapshot differs')
    baseline = SurgicalEngine.restore(load_environment(source), snapshot, check=check)
    result = verify(item['study'], comparison['report'], baseline, check=check)
    require(all(file_hash(p) == value for p, value in hashes.items()), 'native input bytes changed during audit')
    return {**result, 'workspace': str(path), 'sensitivityRevision': revision, 'comparisonRevision': item['comparisonRevision'],
        'inputSha256': hashes, 'inputsUnchanged': True}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('workspace', type=Path)
    parser.add_argument('--source', type=Path)
    parser.add_argument('--revision', type=int, default=1)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    require(args.output.resolve().is_relative_to(ROOT / 'testcases') and not args.output.exists(), 'preserve evidence and keep output in testcases')
    result = verify_workspace(args.workspace, source=args.source, revision=args.revision)
    args.output.parent.mkdir(parents=True, exist_ok=True); args.output.write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps({'passed': result['passed'], 'scenarioCount': result['scenarioCount'], 'policies': result['policies']}))
