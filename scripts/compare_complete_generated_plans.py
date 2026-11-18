#!/usr/bin/env python3
"""Compare a frozen generated-control selector and one-ply observed search at two starting stages.

Use PYTHONPATH=src and the project RL environment. Each arm records a start marker
and its full result. A new directory under testcases/ is required.
"""
from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import platform
import subprocess
import time

import torch

from compare_observed_skill_policies import CHECKPOINT_SHA, Selector, TIER
from generated_shell_policy_evaluation import closed_shell
from resectionlab.functional_evidence import functional_consequence_evidence
from resectionlab.native_backend import backend_identity
from resectionlab.surgical_procedure_policy import evaluate_procedure_sequence
from resectionlab.surgical_sequential_benchmark import FEATURE_NAMES, ReplayVerificationError, run_episode, verify_replay
from resectionlab.surgical_sequential_task import COMMON_CONFIG, generated_task

ROOT = Path(__file__).resolve().parents[1]
PROTOCOL = ROOT / 'manifests/experiments/continuous-generated-holdout-v1.json'
PROTOCOL_SHA = '1e1538defc0674bfd7541e8533278869a6162c4c66125b284165215c6fc0b142'
BUNDLE = ROOT / 'testcases/overnight-sequential-v1/benchmark-task-v2'
CHECKPOINT = BUNDLE / 'checkpoints/17/selected.pt'


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def save(path, value):
    tmp = path.with_suffix('.tmp')
    tmp.write_text(json.dumps(value, allow_nan=False, sort_keys=True) + '\n')
    tmp.replace(path)


def utc():
    return datetime.now(timezone.utc).isoformat()


def continuation(case, method, model, profile):
    task = generated_task(case['parent'])
    if task.audit != case:
        raise ValueError('Frozen post-access source or geometry differs')
    result = run_episode(task, method, TIER, selector=model.select if method == 'learned' else None,
                         seed=17, record_decision_inputs=True, retain_engine=True)
    engine = result.pop('_engine')
    result.update(initialStateId=task.initial_engine().state_id, voxelMm3=task.scene.voxel_mm3,
                  sourceStartingStage='post-access', acceptedActions=len(engine.actions),
                  sequence=evaluate_procedure_sequence(engine)['outcome'],
                  functional=functional_consequence_evidence(engine, task.initial_engine()),
                  wholeToolExit=not any(p.endswith((':withdrawal', ':not-exterior')) for p in result['endpoint']['pending']))
    try:
        result['replay'] = verify_replay(task, result, deadline=time.monotonic() + profile['maxReplaySeconds'])
    except ReplayVerificationError as error:
        result['replay'] = error.receipt
    return result


def summary(row):
    if 'failure' in row:
        return {k: row[k] for k in ('case', 'method', 'failure')}
    endpoint = row['endpoint']
    dims = row['sequence']['dimensions']
    physical = row['computation']
    closed = row['sourceStartingStage'] == 'post-scalp'
    if closed:
        stop = row['replay'].get('checkedStop', {}).get('status') == 'accepted'
        refusals = sum(call.get('status') == 'rejected' for call in physical['trace'])
    else:
        stop = any(r['checkedStopAttempted'] and r['status'] == 'accepted' for r in row['skillHistory'])
        refusals = row['rejections']
    return dict(case=row['case'], method=row['method'], stage=row['sourceStartingStage'],
                selected=row['selected'] if closed else row['selectedSkillIds'],
                targetMm3=endpoint['progress']['originalTargetTreatmentMm3'],
                residualMm3=endpoint['progress']['residualOriginalTargetCells'] * row['voxelMm3'],
                boneRemovedMm3=dims['boneRemovedMm3']['cumulative'],
                nonbonePreparationRemovedMm3=dims['nonbonePreparationRemovedMm3']['cumulative'],
                neuralRemovedMm3=dims['neuralRemovedMm3']['cumulative'], debrisMm3=endpoint['debrisMm3'],
                cleanupRemovedMm3=sum(p['debrisBeforeMm3'] - p['debrisAfterMm3']
                                     for p in row.get('phases', []) if p['phase'] == 'cleanup'),
                fluidMm3=endpoint['fluidMm3'], classification=endpoint['classification'],
                acceptedStop=stop, onlineStopAttempts=(1 if closed else sum(
                    int(r['checkedStopAttempted']) for r in row['skillHistory'])),
                wholeToolExit=row['wholeToolExit'], refusals=refusals,
                physicalRequests=physical['physicalCalls'], verificationRequests=row['replay']['physicalCalls'],
                acceptedActions=row['acceptedActions'], wallSeconds=physical['wallSeconds'],
                modeledSeconds=dims['elapsedSeconds']['cumulative'], verified=row['replay']['verified'],
                structureRemovalMm3={k: v['cumulative'] for k, v in dims['structureRemovedMm3'].items()},
                structureCutSourceMm3={k: v['cumulative'] for k, v in dims['structureCutSourceMm3'].items()},
                bloodGeneratedMm3=dims['bloodGeneratedMm3']['cumulative'],
                vascularSupplyDeficitSeconds=dims.get('supplyDeficitSeconds', {}).get('cumulative'),
                vascularDrainageDeficitSeconds=dims.get('drainageDeficitSeconds', {}).get('cumulative'),
                observedFunctionalOutcome=row['functional']['evidence']['observedClinicalOutcome']['status'],
                missingEvidence=row['sequence']['missingEvidence'])


def load_protocol():
    if sha(PROTOCOL) != PROTOCOL_SHA:
        raise ValueError('Prospective protocol differs')
    fixed = json.loads(PROTOCOL.read_text())
    pin = fixed['checkpoint']
    if (sha(BUNDLE / 'freeze.json') != pin['trainingFreezeSha256'] or sha(CHECKPOINT) != pin['sha256']
            or pin['sha256'] != CHECKPOINT_SHA or list(FEATURE_NAMES) != fixed['featureNames']
            or sha(ROOT / 'docs/sequential-milestone-protocol.json') != pin['trainingProtocolSha256']
            or sha(ROOT / 'docs/sequential-milestone-manifest.json') != pin['trainingReportSha256']):
        raise ValueError('Training freeze, geometry registry, feature names or checkpoint differ')
    p = fixed['continuation']
    if (TIER.max_wall_seconds, TIER.max_physical_calls, COMMON_CONFIG.max_skills) != (
            p['maxOnlineSeconds'], p['maxPhysicalCalls'], p['maxSkills']):
        raise ValueError('Finite-skill compute settings differ')
    prior = set(fixed['overlap']['priorGeometryHashes'])
    seen = set()
    for case in p['cases']:
        if case['geometryHash'] in prior | seen:
            raise ValueError('Previously used or duplicate geometry in frozen cohort')
        seen.add(case['geometryHash'])
    return fixed


def implementation_hashes():
    package = ROOT / 'src/resectionlab'
    sources = [path for pattern in ('surgical_*.py', 'functional_evidence.py', 'core.py', 'geometry.py',
                                    'native_backend.py') for path in package.glob(pattern)]
    sources += [Path(__file__).resolve(), *[ROOT / 'scripts' / name for name in
                ('generated_shell_policy_evaluation.py', 'generated_cranial_demo_case.py',
                 'compare_observed_skill_policies.py', 'run_sequential_milestone.py')]]
    return {str(path.relative_to(ROOT)): sha(path) for path in sources}


def campaign(output):
    fixed = load_protocol()
    if not output.absolute().is_relative_to(ROOT / 'testcases'):
        raise ValueError('Use a new directory under ignored testcases/')
    output.mkdir(exist_ok=False, parents=True)
    torch.set_num_threads(1)
    model = Selector()
    model.load_state_dict(torch.load(CHECKPOINT, map_location='cpu', weights_only=True))
    model.eval()
    identity = implementation_hashes()
    save(output / 'implementation.json', dict(sourceHashes=identity, protocolSha256=PROTOCOL_SHA,
         head=subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip(),
         torch=torch.__version__, threads=torch.get_num_threads(), platform=platform.platform(),
         python=platform.python_version(), native=backend_identity(), checkpointSha256=sha(CHECKPOINT)))
    result = dict(protocolSha256=PROTOCOL_SHA, expectedArms=12, optimizerUpdates=0, rows=[], pairs=[])
    for name, executor in [('continuation', continuation), ('closedShell', closed_shell)]:
        profile = fixed[name]
        for case in profile['cases']:
            pair = []
            case_id = case['id'] if name == 'closedShell' else f"continuation-{case['parent']}"
            for method in fixed['methods']:
                path = output / f'{case_id}-{method}.json'
                row = dict(case=case_id, method=method, startedUtc=utc(), status='attempt-started')
                save(path, row)
                try:
                    with torch.inference_mode():
                        row.update(executor(case, method, model, profile))
                    row['status'] = 'evaluated'
                except Exception as error:
                    row.update(status='failed', failure=dict(type=type(error).__name__, message=str(error)))
                row['finishedUtc'] = utc()
                save(path, row)
                compact = summary(row)
                compact['artifactSha256'] = sha(path)
                result['rows'].append(compact)
                pair.append(row)
                save(output / 'report.json', result)
                print(case_id, method, row['status'], compact.get('classification', compact.get('failure')), flush=True)
            matched = (all(row.get('decisionInputs') for row in pair) and
                       pair[0]['decisionInputs'][0] == pair[1]['decisionInputs'][0] and
                       pair[0]['initialStateId'] == pair[1]['initialStateId'] and
                       pair[0]['sourceSceneId'] == pair[1]['sourceSceneId'])
            result['pairs'].append(dict(case=case_id, equalInitialObservationMaskAndSource=bool(matched)))
    result['classifications'] = dict(Counter(r.get('classification', 'failed') for r in result['rows']))
    result['changedFilesDuringRun'] = [path for path, digest in identity.items() if sha(ROOT / path) != digest]
    result['allVerified'] = (len(result['rows']) == result['expectedArms'] and
                             all(row.get('verified') for row in result['rows']) and
                             all(pair['equalInitialObservationMaskAndSource'] for pair in result['pairs'])
                             and not result['changedFilesDuringRun'])
    save(output / 'report.json', result)
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('output', type=Path)
    args = parser.parse_args()
    raise SystemExit(0 if campaign(args.output)['allVerified'] else 1)
