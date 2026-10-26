#!/usr/bin/env python3
"""Save checked CUSA sector order/hand options beside a verified access comparison.

All generated evidence belongs under ignored testcases/. The source workspace is
read-only; its anatomy-access planning remains intact for frozen CUSA replay.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'src'))

from resectionlab.core import semantic_digest
from resectionlab.imaging import save_case
from resectionlab.surgical_engine import SurgicalEngine
from resectionlab.surgical_environment import load_environment
from resectionlab.surgical_planning import validate_saved_comparison
from resectionlab.surgical_sector_planning import (
    compare_sector_strategies, validate_saved_sector_comparison)
from resectionlab.workspace_bundle import load_workspace, save_workspace


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--case', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--max-actions', type=int, default=48)
    parser.add_argument('--max-wall-seconds', type=float, default=120.)
    args = parser.parse_args()
    if (args.output.exists() or 'testcases' not in args.output.resolve().parts or
            not 1 <= args.max_actions <= 48 or not 0 < args.max_wall_seconds <= 120.):
        parser.error('Use a new testcases/ directory and the declared 48-action/120-second ceilings')
    args.output.mkdir(parents=True)
    started = time.perf_counter()
    source_sha = hashlib.sha256(args.case.read_bytes()).hexdigest()
    receipt = {'schema': 'resectionlab.anatomy-strategy-run.v1',
               'source': str(args.case.resolve()), 'sourceSha256': source_sha,
               'requestedLimits': {'maxNativeActionsPerStrategy': args.max_actions,
                                   'maxAggregateWallSeconds': args.max_wall_seconds}}
    try:
        case, artifacts, workspace = load_workspace(args.case)
        scene = load_environment(args.case)
        record = artifacts.get('surgical_toolbox')
        if not record or record.get('planning', {}).get('task') != 'anatomy-access':
            raise ValueError('A saved verified anatomy-access comparison is required')
        validate_saved_comparison(scene, record, record['planning'])
        if 'sectorPlanning' in record:
            raise ValueError('Use an unmodified access source; sector planning already exists')
        base_id = record['branchId']
        base = SurgicalEngine.restore(scene, record['branches'][base_id])
        if base.actions:
            raise ValueError('Selected source branch must be the common intact initial state')
        source_branch_ids = set(record['branches'])
        branches, comparison = compare_sector_strategies(
            base, record['planning'], record['branches'], acknowledge_unknowns=True,
            max_actions_per_strategy=args.max_actions,
            max_wall_seconds=args.max_wall_seconds,
            progress=lambda fraction, message: print(json.dumps({'progress': fraction,
                'message': message}), flush=True))
        for candidate, plan in branches:
            identity = 'sector-'+plan['method']
            if identity in record['branches']:
                raise ValueError('Generated branch identity already exists: '+identity)
            plan['branchId'] = identity
            record['branches'][identity] = candidate.snapshot()
            record.setdefault('parents', {})[identity] = {'branchId': base_id,
                                                            'position': len(base.actions)}
        if source_branch_ids != set(record['branches'])-{
                plan['branchId'] for _, plan in branches}:
            raise ValueError('Original access branches changed')
        comparison['comparisonSeal'] = semantic_digest(comparison)
        validate_saved_sector_comparison(scene, record, comparison)
        record['sectorPlanning'] = comparison
        record['branchId'] = base_id
        record['position'] = len(base.actions)
        record.pop('timeline', None)
        destination = args.output/'sector-comparison.ressectionlab'
        if workspace is None:
            save_case(case, destination, artifacts=artifacts)
        else:
            save_workspace(case, destination, sources=workspace['sources'],
                imaging=workspace['imagingState'], episode=workspace['episode'],
                selection=workspace['selection'], originals=workspace['originals'], artifacts=artifacts)
        reopened_case, reopened_artifacts, _ = load_workspace(destination)
        reopened = reopened_artifacts['surgical_toolbox']
        validate_saved_comparison(scene, reopened, reopened['planning'])
        validate_saved_sector_comparison(scene, reopened, reopened['sectorPlanning'], replay=True)
        if (reopened_case.semantic_hash != case.semantic_hash or
                hashlib.sha256(args.case.read_bytes()).hexdigest() != source_sha):
            raise ValueError('Original source or source case changed after save')
        receipt.update(status='saved', sceneId=scene.identity,
                       originalAccessComparisonSeal=record['planning']['comparisonSeal'],
                       sourceCaseHash=case.semantic_hash,
                       candidateCount=len(branches), passedCount=comparison['passedCount'],
                       optionFailures=comparison['optionFailures'],
                       savedWorkspace=str(destination.resolve()),
                       savedSha256=hashlib.sha256(destination.read_bytes()).hexdigest(),
                       sourcePreserved=True, sectorPlanning=comparison)
    except Exception as error:
        receipt.update(status='failed', errorType=type(error).__name__, error=str(error),
                       sourcePreserved=hashlib.sha256(args.case.read_bytes()).hexdigest() == source_sha)
        raise
    finally:
        receipt['wallSeconds'] = time.perf_counter()-started
        (args.output/'receipt.json').write_text(json.dumps(receipt, indent=2,
                                                            allow_nan=False)+'\n')
        print(json.dumps({'status': receipt['status'],
                          'wallSeconds': receipt['wallSeconds'],
                          'candidateCount': receipt.get('candidateCount'),
                          'passedCount': receipt.get('passedCount')}), flush=True)


if __name__ == '__main__':
    main()
