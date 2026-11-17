#!/usr/bin/env python3
"""Retry interrupted infrastructure attempts once, inside the frozen campaign."""
from __future__ import annotations

import argparse
import fcntl
import sys
from pathlib import Path

import run_approach_procedure_baselines as campaign


def recover(out, profile):
    protocol = campaign._read(out / 'protocol.json')
    protocol.pop('sourceCommitAtFreeze')
    if protocol != campaign.freeze(profile):
        raise ValueError('The frozen campaign identity changed')
    progress_file = out / 'progress.json'
    progress = campaign._read(progress_file)
    if 'active' in progress:
        raise ValueError('Resume the original campaign first to settle the active reservation')

    expected = {
        f'{tier.name}-{case[0]}-{method}'
        for tier in campaign.TIERS
        for case in campaign.CASES
        for method in campaign.METHODS
    }
    interrupted = progress['interruptions']
    remaining = expected - progress['completed'].keys()
    if not remaining <= interrupted.keys():
        raise ValueError('Run the original campaign through all unattempted rows first')
    retryable = [
        key for key in sorted(remaining)
        if interrupted[key]['reason'] == 'process interrupted before row committed'
        and f'{key}#initial' not in interrupted
    ]
    if not retryable or progress['spentSeconds'] >= campaign.CAMPAIGN_SECONDS:
        return campaign._read(out / 'report.json')
    for key in retryable:
        first = interrupted.pop(key)
        interrupted[f'{key}#initial'] = {
            **first, 'attempt': 1, 'recovery': 'one retry under the original caps',
        }
    campaign._save(progress_file, progress)
    # The scientific workload and accounting stay in the exact frozen runner.
    # Reserved time for earlier attempts stays charged to the same campaign.
    return campaign.run(out, profile, None)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--profile', type=Path, required=True)
    args = parser.parse_args()
    out = args.output.resolve()
    if not out.is_relative_to(campaign.ROOT / 'testcases') or not (out / 'protocol.json').is_file():
        parser.error('Supply an existing frozen campaign under testcases/')
    with (out / '.lock').open('w') as guard:
        fcntl.flock(guard, fcntl.LOCK_EX | fcntl.LOCK_NB)
        result = recover(out, args.profile)
    return int(result['missingRows'] > 0 or any(not row['replayVerified'] for row in result['rows']))


if __name__ == '__main__':
    sys.exit(main())
