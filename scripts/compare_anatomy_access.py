#!/usr/bin/env python3
"""Compare bounded source directions and save exact native branches under testcases/.

The optional modeled-brain-corridor flag authorizes an uncalibrated harmful
research action, never clinical access or an inferred skull opening.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))

from resectionlab.core import semantic_digest
from resectionlab.imaging import save_case
from resectionlab.surgical_access_planning import compare_access_approaches
from resectionlab.surgical_engine import SurgicalEngine
from resectionlab.surgical_environment import load_environment
from resectionlab.surgical_planning import validate_saved_comparison
from resectionlab.workspace_bundle import load_workspace, save_workspace


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--case', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--branch', type=str)
    parser.add_argument('--allow-modeled-brain-corridor', action='store_true')
    parser.add_argument('--max-actions', type=int, default=48)
    parser.add_argument('--max-wall-seconds', type=float, default=120.)
    args = parser.parse_args()
    if ('testcases' not in args.output.resolve().parts or args.output.exists() or
            not 0 < args.max_wall_seconds <= 120.):
        parser.error('Use a new output directory under testcases/ and a wall budget <=120 seconds')
    args.output.mkdir(parents=True)
    started = time.perf_counter()
    source_sha = hashlib.sha256(args.case.read_bytes()).hexdigest()
    receipt = {'schema': 'resectionlab.anatomy-access-run.v1',
        'source': str(args.case.resolve()), 'sourceSha256': source_sha,
        'allowModeledBrainCorridor': args.allow_modeled_brain_corridor,
        'requestedLimits': {'maxActionsPerApproach': args.max_actions,
                            'maxWallSeconds': args.max_wall_seconds}}
    try:
        case, artifacts, workspace = load_workspace(args.case)
        scene = load_environment(args.case)
        record = artifacts.get('surgical_toolbox')
        if record is None:
            raise ValueError('A prepared surgical environment is required; no source anatomy is invented')
        branch_id = args.branch or record['branchId']
        if branch_id not in record['branches']:
            raise ValueError('Selected initial branch is unavailable')
        engine = SurgicalEngine.restore(scene, record['branches'][branch_id])
        position = len(engine.actions)
        def progress(fraction, message):
            print(json.dumps({'progress': fraction, 'message': message}), flush=True)
        branches, comparison = compare_access_approaches(engine, acknowledge_unknowns=True,
            allow_modeled_brain_corridor=args.allow_modeled_brain_corridor,
            max_actions_per_approach=args.max_actions,
            max_wall_seconds=args.max_wall_seconds, progress=progress)
        for candidate, plan in branches:
            identity = 'access-' + plan['method']
            if identity in record['branches']:
                raise ValueError('Saved branch identity already exists: ' + identity)
            plan['branchId'] = identity
            record['branches'][identity] = candidate.snapshot()
            record.setdefault('parents', {})[identity] = {'branchId': branch_id, 'position': position}
        comparison['comparisonSeal'] = semantic_digest(comparison)
        validate_saved_comparison(scene, record, comparison)
        record['planning'] = comparison
        record['branchId'] = branch_id
        record['position'] = position
        record.pop('timeline', None)
        destination = args.output / 'access-comparison.ressectionlab'
        if workspace is None:
            save_case(case, destination, artifacts=artifacts)
        else:
            save_workspace(case, destination, sources=workspace['sources'], imaging=workspace['imagingState'],
                episode=workspace['episode'], selection=workspace['selection'], originals=workspace['originals'],
                artifacts=artifacts)
        reopened_case, reopened_artifacts, _ = load_workspace(destination)
        reopened = reopened_artifacts['surgical_toolbox']
        validate_saved_comparison(scene, reopened, reopened['planning'])
        if (reopened_case.semantic_hash != case.semantic_hash or
                hashlib.sha256(args.case.read_bytes()).hexdigest() != source_sha):
            raise ValueError('Saved source or original input changed')
        receipt.update(status='saved', sceneId=scene.identity, sourceCaseHash=case.semantic_hash,
            comparisonStatus=comparison['status'], alternativeCount=len(branches),
            directionFailures=comparison['directionFailures'],
            savedWorkspace=str(destination.resolve()),
            savedSha256=hashlib.sha256(destination.read_bytes()).hexdigest(),
            sourcePreserved=True, comparison=comparison)
    except Exception as error:
        receipt.update(status='failed', errorType=type(error).__name__, error=str(error),
                       sourcePreserved=hashlib.sha256(args.case.read_bytes()).hexdigest() == source_sha)
        raise
    finally:
        receipt['wallSeconds'] = time.perf_counter()-started
        (args.output / 'receipt.json').write_text(json.dumps(receipt, indent=2, allow_nan=False) + '\n')
        print(json.dumps({'status': receipt['status'], 'wallSeconds': receipt['wallSeconds'],
                          'alternativeCount': receipt.get('alternativeCount'),
                          'directionFailures': receipt.get('directionFailures')}), flush=True)


if __name__ == '__main__':
    main()
