#!/usr/bin/env python3
"""Prepare authenticated observations for selected actions in a saved demo.

This is disposable local playback preparation. It verifies the complete native
archive first, preserves the workspace, and grants no policy or RL authority.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from resectionlab.surgical_engine import SurgicalEngine
from resectionlab.surgical_environment import load_environment
from resectionlab.surgical_frame_replay import SurgicalFrameCache
from resectionlab.surgical_replay_cache import SurgicalReplayCache
from resectionlab.imaging import read_case_artifacts


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--case', required=True, type=Path)
    parser.add_argument('--branch', required=True)
    parser.add_argument('--actions', required=True,
                        help='Zero-based half-open ranges, e.g. 16:23,47:73')
    parser.add_argument('--report', required=True, type=Path)
    args = parser.parse_args()
    if not args.report.resolve().is_relative_to(ROOT / 'testcases') or args.report.exists():
        parser.error('Use a new report path under testcases/')
    positions = set()
    for span in args.actions.split(','):
        start, stop = map(int, span.split(':'))
        if start < 0 or stop <= start:
            parser.error('Action ranges must be increasing and nonnegative')
        positions.update(range(start, stop))
    started = time.perf_counter()
    def check():
        if time.perf_counter() - started > 600:
            raise TimeoutError('Playback preparation exceeded 600 seconds')
    source_hash = hashlib.sha256(args.case.read_bytes()).hexdigest()
    snapshot = read_case_artifacts(args.case)['surgical_toolbox']['branches'][args.branch]
    if not positions or max(positions) >= len(snapshot['actions']):
        parser.error('Requested action is outside this branch')
    scene = load_environment(args.case)
    print(json.dumps({'stage': 'canonical_archive', 'actions': len(snapshot['actions'])}), flush=True)
    full = SurgicalEngine.restore(scene, snapshot, use_cache=True, check=check)
    canonical_seconds = time.perf_counter() - started
    cache = SurgicalReplayCache(verified_cache=True)
    frames = SurgicalFrameCache()
    admission = frames.admit(full, snapshot, cache, verified_history=full.history, check=check)
    rows = []
    for position in sorted(positions):
        before = time.perf_counter()
        _, _, playback = frames.seek(full, admission.snapshot, position, 0, cache,
            admission=admission, verified_history=admission.history, check=check, display_only=True)
        item = frames.store.load(admission, position)
        if item is None:
            raise ValueError('Prepared action was not retained in the authenticated store')
        flow = [op for frame in item['frames'] for op in frame.get('flowDisplay', {}).get('operations', [])]
        row = {'actionIndex': position, 'frames': playback['frameCount'],
               'seconds': time.perf_counter() - before, 'flowOperations': len(flow),
               'woundDeposits': sum(op['kind'] == 'deposit' and op['provenance'] == 'wound' for op in flow)}
        rows.append(row)
        print(json.dumps(row), flush=True)
    if hashlib.sha256(args.case.read_bytes()).hexdigest() != source_hash:
        raise ValueError('Playback preparation changed the source workspace')
    report = {'schema': 'resectionlab.operative-playback-preparation.v1',
              'case': str(args.case.resolve()), 'sourceSha256': source_hash,
              'sourcePreserved': True, 'branch': args.branch, 'stateId': full.state_id,
              'canonicalAdmissionSeconds': canonical_seconds,
              'wallSeconds': time.perf_counter() - started, 'actions': rows,
              'scope': 'Local cache preparation; uncontrolled background load, not a throughput benchmark.'}
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2, allow_nan=False) + '\n')
    print(json.dumps({'stage': 'prepared', 'report': str(args.report), 'wallSeconds': report['wallSeconds']}), flush=True)


if __name__ == '__main__':
    main()
