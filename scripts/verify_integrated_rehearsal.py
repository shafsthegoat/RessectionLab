#!/usr/bin/env python3
"""Execute pinned acquired-case recipes through the shared instrument engine.

This is a research software demonstration. The protocol declares its source
files, development roles and scripted access paths; it supplies no anatomy.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))

from resectionlab.core import array_digest
from resectionlab.imaging import save_case
from resectionlab.surgical_engine import SurgicalEngine
from resectionlab.surgical_environment import load_environment
from resectionlab.surgical_rl_protection import GuardConfig, load_protection
from resectionlab.workspace_bundle import load_workspace, save_workspace

ROOT = Path(__file__).resolve().parents[1]


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')


def check_saved(path):
    case, artifacts, workspace = load_workspace(path)
    scene = load_environment(path)
    record = artifacts['surgical_toolbox']
    engine = SurgicalEngine.restore(scene, record['branches'][record['branchId']])
    return {'caseId': case.case_id, 'sourceCaseHash': case.semantic_hash,
            'sceneId': scene.identity, 'stateId': engine.state_id,
            'actions': len(engine.actions), 'metrics': engine.metrics(),
            'protection': engine.guard_report(), 'heldHands': sorted(engine.state['holds']),
            'heldSourceCells': {hand: len(hold['cells']) for hand, hold in engine.state['holds'].items()},
            'originalFiles': len(workspace['originals']) if workspace else 0,
            'auxiliaryScans': len(workspace['sources']) if workspace else 0}


def execute(recipe, output):
    source = (ROOT / recipe['case']).resolve()
    if hashlib.sha256(source.read_bytes()).hexdigest() != recipe['sha256']:
        raise ValueError('Pinned source file changed: ' + str(source))
    if recipe['role'] != 'development':
        raise ValueError('This demonstration only admits declared development cases')
    case, artifacts, workspace = load_workspace(source)
    scene = load_environment(source)
    source_hash = array_digest(scene.labels)
    image_hash = array_digest(case.mri)
    protection = load_protection(scene, ROOT / recipe['protection'], strict=True)
    engine = SurgicalEngine(scene, protection=protection, guard_config=GuardConfig())
    rows, snapshots = [], {}
    for index, action in enumerate(recipe['actions']):
        before = engine.state_id
        if index in recipe.get('previewAt', []):
            preview = engine.preview(action)
            if engine.state_id != before:
                raise AssertionError('Preview changed committed state')
        else:
            preview = None
        started = time.perf_counter()
        result = engine.step(action)
        wall = time.perf_counter() - started
        expected = recipe.get('expectedOutcomes', {}).get(str(index))
        if expected is not None and (result['status'] != expected['status'] or
                                    not str(result.get('reason')).startswith(expected['reasonPrefix'])):
            raise AssertionError('Expected blocked outcome changed: ' + str(result))
        if preview is not None and preview['stateId'] != result['stateId']:
            raise AssertionError('Preview and commit differ')
        rows.append({'action': action, 'status': result['status'], 'reason': result.get('reason'),
                     'stateId': engine.state_id, 'metrics': result['metrics'], 'wallSeconds': wall,
                     'previewEqualsCommit': None if preview is None else True})
        if str(index + 1) in recipe.get('milestones', {}):
            snapshots[recipe['milestones'][str(index + 1)]] = engine.snapshot()
        print(recipe['id'], index + 1, action['operation'], result['status'],
              result.get('reason'), 'target', result['metrics']['targetRemovedMm3'], flush=True)
        if result['status'] == 'rejected' and index not in recipe.get('expectedRejected', []):
            raise AssertionError('Unexpected rejection: ' + str(result.get('reason')))
    metrics = engine.metrics()
    if metrics['targetRemovedMm3'] <= recipe.get('minimumTargetRemovedMm3', 0):
        raise AssertionError('No required target progress')
    if recipe.get('requireHeldHand') and not engine.state['holds'].get(recipe['requireHeldHand'], {}).get('cells'):
        raise AssertionError('Required concurrent hold has no retained source tissue')
    if recipe.get('requireBoneRemoval') and metrics['drillRemovedVolumeMm3'] <= 0:
        raise AssertionError('No supplied bone was drilled')
    if array_digest(scene.labels) != source_hash or array_digest(case.mri) != image_hash:
        raise AssertionError('Source arrays changed')
    final = engine.snapshot()
    snapshots['main'] = final
    record = copy.deepcopy(artifacts.get('surgical_toolbox') or {})
    record.update(mapping=scene.regions, branches=snapshots, branchId='main',
                  position=len(engine.actions), parents={name: None for name in snapshots},
                  timeline={'main': engine.history})
    artifacts['surgical_toolbox'] = record
    destination = output / (recipe['id'] + '.ressectionlab')
    if workspace is None:
        save_case(case, destination, artifacts=artifacts)
    else:
        save_workspace(case, destination, sources=workspace['sources'],
                       imaging=workspace['imagingState'], episode=workspace['episode'],
                       selection=workspace['selection'], originals=workspace['originals'], artifacts=artifacts)
    write_json(output / (recipe['id'] + '-snapshot.json'), final)
    write_json(output / (recipe['id'] + '-attempted-actions.json'), recipe['actions'])
    write_json(output / (recipe['id'] + '-actions.json'), rows)
    child = subprocess.run([sys.executable, str(Path(__file__).resolve()), '--check-saved', str(destination)],
                           cwd=ROOT, text=True, capture_output=True, check=True)
    reopened = json.loads(child.stdout)
    if reopened['stateId'] != engine.state_id or reopened['sourceCaseHash'] != case.semantic_hash:
        raise AssertionError('Fresh-process saved episode differs')
    receipt = {'schema': 'resectionlab.integrated-rehearsal-case.v1', 'source': str(source),
               'sourceSha256': recipe['sha256'], 'sourceCaseHash': case.semantic_hash,
               'labelArrayHash': source_hash, 'intensityArrayHash': image_hash,
               'savedCase': str(destination), 'sourceArraysUnchanged': True,
               'scriptedDevelopmentOnly': True, 'clinicalValidation': 'none',
               'actions': len(recipe['actions']), 'metrics': metrics,
               'freshProcess': reopened, 'limitations': recipe['limitations']}
    write_json(output / (recipe['id'] + '-verification.json'), receipt)
    return receipt


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--protocol', type=Path, default=ROOT / 'artifacts/integrated-rehearsal-v1/protocol.json')
    parser.add_argument('--output', type=Path, default=ROOT / 'outputs/integrated-rehearsal-v1')
    parser.add_argument('--check-saved', type=Path)
    args = parser.parse_args()
    if args.check_saved:
        print(json.dumps(check_saved(args.check_saved), allow_nan=False))
        return
    args.output.mkdir(parents=True, exist_ok=True)
    protocol = json.loads(args.protocol.read_text())
    receipts = [execute(recipe, args.output) for recipe in protocol['cases']]
    write_json(args.output / 'verification.json', {'protocol': str(args.protocol.resolve()), 'cases': receipts})


if __name__ == '__main__':
    main()
