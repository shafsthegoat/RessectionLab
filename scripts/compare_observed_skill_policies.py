#!/usr/bin/env python3
"""Bounded observation-matched replay of a frozen, generated post-access control."""
from __future__ import annotations

import argparse
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import time

import torch

from run_sequential_milestone import Selector
from resectionlab.surgical_sequential_benchmark import (ComputeTier, FEATURE_NAMES,
    run_episode, verify_replay)
from resectionlab.surgical_sequential_task import generated_task

ROOT = Path(__file__).resolve().parents[1]
CHECKPOINT_SHA = 'bfbffdac0cd2631542bf2de71647eab1604e72ac81f15f7424d7650768d9b79f'
TIER = ComputeTier('low', 56, 25., 2, 2, 8)


def compare(output):
    output.mkdir(exist_ok=False, parents=True)
    bundle = ROOT/'testcases/overnight-sequential-v1/benchmark-task-v2'
    manifest = json.loads((ROOT/'docs/sequential-milestone-manifest.json').read_text())
    checkpoint = bundle/'checkpoints/17/selected.pt'
    if (hashlib.sha256((bundle/'freeze.json').read_bytes()).hexdigest() != manifest['freezeSha256']
            or hashlib.sha256(checkpoint.read_bytes()).hexdigest() != CHECKPOINT_SHA):
        raise ValueError('Frozen generated-control checkpoint or freeze record differs')
    torch.set_num_threads(1)
    model = Selector()
    model.load_state_dict(torch.load(checkpoint, map_location='cpu', weights_only=True))
    model.eval()
    rows = []
    for method, selector in [('learned', model.select), ('observed_greedy_search', None)]:
        task = generated_task(200)
        row = {'method': method, 'sourceSceneId': task.scene.identity}
        try:
            with torch.no_grad():
                row = run_episode(task, method, TIER, selector=selector, seed=17,
                                  record_decision_inputs=True)
            row['checkpointSha256'] = CHECKPOINT_SHA if method == 'learned' else None
            row['replay'] = verify_replay(task, row, deadline=time.monotonic()+90.)
        except Exception as error:
            row['failure'] = {'type': type(error).__name__, 'reason': str(error)}
        (output/(method+'.json')).write_text(json.dumps(row, allow_nan=False)+'\n')
        rows.append(row)
    if any('failure' in row for row in rows):
        raise RuntimeError('Incomplete comparison. Each attempted arm is retained in the output directory.')
    if (rows[0]['sourceSceneId'] != rows[1]['sourceSceneId'] or not rows[0]['decisionInputs']
            or rows[0]['decisionInputs'][0] != rows[1]['decisionInputs'][0]):
        raise ValueError('Matched public inputs required')
    summary = dict(scope='generated finite-skill post-access control, previously evaluated parent 200',
        initialStateId=task.initial_engine().state_id, sourceSceneId=task.scene.identity,
        featureNames=list(FEATURE_NAMES), initialDecisionInputs=rows[0]['decisionInputs'][0],
        budgetPerArm=asdict(TIER), optimizerUpdates=0, oraclePreviews=0,
        search='One-step greedy enumeration of available routes using the frozen actor feature matrix and mask.',
        methods=[{key: row[key] for key in ('method', 'checkpointSha256', 'selectedSkillIds',
            'progress','endpoint','reward','attempts','rejections','neuralRemovedMm3',
            'structureRemovedMm3','missingEvidence','computation','replay','finalStateId')} for row in rows])
    (output/'summary.json').write_text(json.dumps(summary, indent=2, allow_nan=False)+'\n')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('output', type=Path)
    compare(parser.parse_args().output)
