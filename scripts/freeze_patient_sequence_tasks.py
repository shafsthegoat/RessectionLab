#!/usr/bin/env python3
"""Freeze source-sized tasks on an existing qualified-case inventory.

This read-only preparation does not admit autonomous execution or repair anatomy.
Each case runs in a separate worker for an interpretable process memory peak.
Native verification/configuration calls, wall time and source hashes are retained.
All local inventories/results must stay under ignored testcases/.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import resource
import subprocess
import sys
import time

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO/'src'))


def sha(path):
    with path.open('rb') as stream: return hashlib.file_digest(stream, 'sha256').hexdigest()


def write(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False)+'\n')


def worker(source, output, start, role_root):
    from resectionlab.surgical_engine import SurgicalEngine
    from resectionlab.surgical_environment import load_environment
    from resectionlab.surgical_patient_task import freeze_patient_task
    from resectionlab.surgical_rl_protection import restore_protection_contract
    from resectionlab.surgical_sequence_outcomes import sequence_outcome
    from resectionlab.workspace_bundle import load_workspace

    started = time.monotonic(); before = sha(source)
    case, artifacts, _ = load_workspace(source)
    scene = load_environment(source)
    record = artifacts['surgical_toolbox']
    snapshot = record['branches'][record['branchId']]
    calls = {'preview': 0, 'step': 0}
    def count(frame, event, arg):
        if event == 'call':
            if frame.f_code is SurgicalEngine.preview.__code__: calls['preview'] += 1
            if frame.f_code is SurgicalEngine.step.__code__: calls['step'] += 1
    sys.setprofile(count)
    try:
        if start == 'checkpoint':
            engine = SurgicalEngine.restore(scene, snapshot)
        else:
            protection, guard = restore_protection_contract(scene, snapshot['protectionContract'])
            engine = SurgicalEngine(scene, protection=protection, guard_config=guard)
        if engine.state.get('toolMechanicsRevision') != 2 or engine.state.get('physicalClockRevision') != 1:
            outcome = engine.step({'slot': 'right', 'operation': 'configure',
                'mechanicsRevision': 2, 'clockRevision': 1, 'duration': .1})
            if outcome['status'] != 'accepted': raise ValueError('Native setup refused: '+str(outcome['reason']))
    finally:
        sys.setprofile(None)
    task = freeze_patient_task(case, engine, role_root=role_root)
    ledger = sequence_outcome(engine)
    write(output/'task.json', task); write(output/'sequence-outcome.json', ledger)
    after = sha(source)
    if before != after: raise ValueError('Input source changed')
    result = {'schema': 'resectionlab.frozen-patient-task-readout.v1',
        'source': str(source), 'sourceSha256': before, 'sourcePreserved': True,
        'family': task['sourceFamily'], 'role': task['sourceRole'], 'start': start,
        'taskSeal': task['taskSeal'], 'sourceStateId': task['startingStateId'],
        'originalTargetMm3': task['originalTargetMm3'],
        'minimumTargetMm3': task['minimumWholeTargetMm3'],
        'minimumPerZoneMm3': task['minimumTargetPerZoneMm3'], 'policy': task['policy'],
        'eligibilityBlockers': task['eligibilityBlockers'],
        'classification': 'missing_or_unsupported_evidence' if task['eligibilityBlockers'] else 'eligible_for_separately_checked_search',
        'taskCompleted': False, 'newSearchExecuted': False,
        'cost': {'wallSeconds': time.monotonic()-started, 'previewCalls': calls['preview'],
            'directNativeSteps': calls['step'], 'physicalCalls': sum(calls.values()),
            'peakProcessRssBytes': resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*(1 if sys.platform == 'darwin' else 1024)},
        'implementation': {name: sha(REPO/name) for name in [
            'src/resectionlab/surgical_patient_task.py', 'src/resectionlab/surgical_sequence_outcomes.py',
            'src/resectionlab/surgical_engine.py', 'src/resectionlab/surgical_scene.py']},
        'clinicalValidation': False}
    write(output/'result.json', result)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest', type=Path)
    parser.add_argument('--output', required=True, type=Path)
    parser.add_argument('--start', choices=['intact', 'checkpoint'], default='intact')
    parser.add_argument('--worker-source', type=Path, help=argparse.SUPPRESS)
    parser.add_argument('--role-root', type=Path, help='Trusted source catalog root; never taken from case metadata')
    args = parser.parse_args(); output = args.output.resolve()
    if not output.is_relative_to(REPO/'testcases'): raise ValueError('Use an ignored testcases output')
    if output.exists(): raise FileExistsError('Preserve earlier evidence; choose a fresh output')
    output.mkdir(parents=True)
    if args.worker_source:
        worker(args.worker_source.resolve(), output, args.start, args.role_root); return
    if args.manifest is None: raise ValueError('An existing five-case qualification manifest is required')
    manifest = json.loads(args.manifest.read_text())
    role_root = args.role_root or Path(manifest['data_root'])
    rows = manifest['cases']
    families = [row['family_id'] for row in rows]
    if len(rows) != 5 or len(set(families)) != 5:
        raise ValueError('Retain all five distinct source families')
    results = []
    for index, row in enumerate(rows):
        reference = row['references']['bounded_native_replay']; source = Path(reference['path'])
        if sha(source) != reference['sha256']: raise ValueError('Qualified source hash differs')
        directory = output/f'case-{index+1}'
        run = subprocess.run([sys.executable, str(Path(__file__).resolve()), '--worker-source', str(source),
            '--output', str(directory), '--start', args.start, '--role-root', str(role_root)], capture_output=True, text=True)
        (output/f'case-{index+1}.log').write_text(run.stdout+run.stderr)
        if run.returncode: raise RuntimeError(f'Case worker failed: {families[index]}; see retained log')
        result = json.loads((directory/'result.json').read_text())
        if result['family'] != row['family_id']: raise ValueError('Source family differs from qualification')
        results.append(result)
    write(output/'results.json', {'schema': 'resectionlab.five-frozen-patient-tasks.v1',
        'manifestSha256': sha(args.manifest), 'cases': results, 'allFiveRetained': True,
        'goalComplete': False, 'meaning': 'Prospective tasks and exact eligibility, not completed treatment or a new-planner comparison.'})
    print(json.dumps([{'family': row['family'], 'targetMm3': row['originalTargetMm3'],
        'goalMm3': row['minimumTargetMm3'], 'blockers': row['eligibilityBlockers']} for row in results]))


if __name__ == '__main__': main()
