#!/usr/bin/env python3
"""Run/resume one source-bound full-case research episode; never train a policy."""
from __future__ import annotations
import argparse
import copy
import hashlib
import json
from pathlib import Path
import sys
import time
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from resectionlab.core import array_digest
from resectionlab.surgical_environment import load_environment
from resectionlab.surgical_engine import SurgicalEngine
from resectionlab.full_case_rehearsal import bind_case_protection, new_director, run_slice, accounting, resume_director
from resectionlab.workspace_bundle import load_workspace, save_workspace

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CASE = ROOT / 'testcases/cases/meningioma-sub-06-standard-v2.ressectionlab'
DEFAULT_OUTPUT = ROOT / 'testcases/outputs/full-case-rehearsal-20261020/primary'


def persist(case, artifacts, workspace, engine, director, destination, branch_id, parent):
    record = artifacts['surgical_toolbox']
    record['branches'][branch_id] = engine.snapshot()
    record.update(branchId=branch_id, position=len(engine.actions))
    record.setdefault('parents', {})[branch_id] = parent
    record.setdefault('fullCase', {})[branch_id] = copy.deepcopy(director)
    # This derived cache is independently rebuilt by native admission/replay.
    record.pop('timeline', None)
    save_workspace(case, destination, sources=workspace['sources'], imaging=workspace['imagingState'],
                   episode=workspace['episode'], selection=workspace['selection'], originals=workspace['originals'], artifacts=artifacts)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--case', type=Path, default=DEFAULT_CASE)
    parser.add_argument('--output', type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument('--scenario', choices=['primary', 'vascular'], default='primary')
    parser.add_argument('--resume', type=Path)
    parser.add_argument('--max-slices', type=int, default=1000)
    parser.add_argument('--slice-actions', type=int, default=2)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    destination = args.output / (args.scenario + '.ressectionlab')
    if destination.resolve() == args.case.resolve() or (destination.exists() and destination.samefile(args.case)):
        raise ValueError('Episode destination must preserve the original case in a different file')
    source = args.resume or args.case
    source_sha = hashlib.sha256(args.case.read_bytes()).hexdigest()
    case, artifacts, workspace = load_workspace(source)
    if workspace is None: raise ValueError('Full-case runner requires a portable source workspace')
    scene = load_environment(source)
    if args.resume:
        original_case, _, _ = load_workspace(args.case)
        original_scene = load_environment(args.case)
        if original_scene.identity != scene.identity or original_case.semantic_hash != case.semantic_hash or array_digest(original_case.mri) != array_digest(case.mri):
            raise ValueError('Resume source differs from the declared original case')
    record = artifacts['surgical_toolbox']
    branch_id = record['branchId']
    initial_snapshot = record['branches'][branch_id]
    engine = SurgicalEngine.restore(scene, initial_snapshot, use_cache=True)
    existing = record.get('fullCase', {}).get(branch_id)
    parent = record.get('parents', {}).get(branch_id)
    if existing and existing['scenario'] == args.scenario:
        if existing.get('lastStateId', existing.get('startStateId')) != engine.state_id:
            raise ValueError('Saved director state differs from the selected action boundary')
        director = resume_director(existing)
    else:
        parent = {'branchId': branch_id, 'position': len(engine.actions)}
        engine = bind_case_protection(engine)
        names = {r['name'] for r in scene.regions if r.get('target')}
        annotation = sum(int(case.compartments[name].sum()) for name in names if name in case.compartments) * scene.voxel_mm3
        director = new_director(engine, args.scenario, full_annotation_target_mm3=annotation)
        branch_id = 'full-' + args.scenario
        if branch_id in record['branches']:
            number = 2
            while branch_id + '-' + str(number) in record['branches']: number += 1
            branch_id += '-' + str(number)
    label_hash, image_hash = array_digest(scene.labels), array_digest(case.mri)
    started = time.perf_counter()
    persist(case, artifacts, workspace, engine, director, destination, branch_id, parent)
    trace = args.output / 'execution.jsonl'
    for index in range(args.max_slices):
        before = len(engine.actions)
        before_schedule = (director['stage'], director['accessIndex'], director.get('vascularIndex'))
        engine, director = run_slice(engine, director, max_actions=args.slice_actions)
        row = {'slice': index, 'position': len(engine.actions), 'stateId': engine.state_id,
               'stage': director['stage'], 'phase': director['currentPhaseId'], 'status': director['status'],
               'metrics': engine.metrics(), 'wallSeconds': time.perf_counter() - started,
               'lastOutcome': director.get('lastOutcome')}
        with trace.open('a') as stream: stream.write(json.dumps(row, allow_nan=False) + '\n')
        print(json.dumps(row, allow_nan=False), flush=True)
        # Persist the common prefix exactly when first supported removal occurs.
        prefix = director.get('commonPrefixPosition')
        if prefix is not None and 'exposure-prefix' not in record['branches']:
            prefix_engine = engine if prefix == len(engine.actions) else engine.branch(prefix)
            record['branches']['exposure-prefix'] = prefix_engine.snapshot()
            record.setdefault('parents', {})['exposure-prefix'] = parent
        persist(case, artifacts, workspace, engine, director, destination, branch_id, parent)
        if director['status'] in ('complete', 'stopped'): break
        if len(engine.actions) == before:
            after_schedule = (director['stage'], director['accessIndex'], director.get('vascularIndex'))
            if after_schedule == before_schedule:
                raise ValueError('Director made no physical or scheduling progress while still running')
    if array_digest(scene.labels) != label_hash or array_digest(case.mri) != image_hash or hashlib.sha256(args.case.read_bytes()).hexdigest() != source_sha:
        raise AssertionError('Original source arrays or workspace bytes changed')
    receipt = {'schema': 'resectionlab.full-case-execution.v1', 'originalCase': str(args.case.resolve()),
               'originalSha256': source_sha, 'episode': str(destination.resolve()), 'sceneId': scene.identity,
               'labelArrayHash': label_hash, 'imageHash': image_hash, 'branchId': branch_id,
               'stateId': engine.state_id, 'actions': len(engine.actions), 'metrics': engine.metrics(),
               'accounting': accounting(engine), 'director': director, 'wallSeconds': time.perf_counter() - started,
               'sourceUnchanged': True}
    (args.output / 'execution-receipt.json').write_text(json.dumps(receipt, indent=2, allow_nan=False) + '\n')
    print('Saved', destination, flush=True)


if __name__ == '__main__': main()
