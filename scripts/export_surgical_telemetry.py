#!/usr/bin/env python3
"""Replay a saved surgical episode and export descriptive bimanual telemetry."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))

from resectionlab.surgical_environment import load_environment
from resectionlab.surgical_telemetry import LABELS, capture_episode, export_telemetry


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--case', required=True, type=Path, help='Patient/engineering case bundle used by the saved episode')
    parser.add_argument('--actions', type=Path, help='Full attempted action array, or object with attemptedActions/actions')
    parser.add_argument('--snapshot', type=Path, help='Final SurgicalEngine snapshot, including its protection contract')
    parser.add_argument('--label', choices=LABELS, default='scripted')
    parser.add_argument('--episode-id', default='episode')
    parser.add_argument('--interval', type=float, default=.2)
    parser.add_argument('--expected-state-id')
    parser.add_argument('--output', required=True, type=Path)
    args = parser.parse_args()
    if args.actions is None and args.snapshot is None:
        parser.error('Provide --actions and/or --snapshot')
    actions = None
    if args.actions is not None:
        record = json.loads(args.actions.read_text())
        actions = record.get('attemptedActions', record.get('actions')) if isinstance(record, dict) else record
        if not isinstance(actions, list):
            parser.error('--actions must contain an action array')
        actions = [item['action'] if isinstance(item, dict) and 'action' in item and 'operation' not in item else item for item in actions]
    snapshot = json.loads(args.snapshot.read_text()) if args.snapshot else None
    episode = capture_episode(load_environment(args.case), actions, label=args.label, episode_id=args.episode_id,
                              interval_seconds=args.interval, expected_state_id=args.expected_state_id, snapshot=snapshot)
    print(json.dumps(export_telemetry([episode], args.output), indent=2))


if __name__ == '__main__':
    main()
