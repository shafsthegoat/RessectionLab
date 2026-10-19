#!/usr/bin/env python3
"""Replay a fixed existing pilot task through the same desktop bridge A/B IPC.

No training or source edits. The JSON recipe can be sent unchanged via Electron
preload compareSurgicalAnatomy after opening the two prepared patient cases.
"""
import argparse
import copy
import json
from pathlib import Path

from resectionlab.desktop_bridge import BridgeSession, _Request


def pilot_prefix(root):
    def actions(relative):
        return [copy.deepcopy(r['action']) for r in json.loads((root / relative).read_text())['actions']]
    drill = actions('outputs/surgical-patient-v2/drill-verification.json')
    cusa = actions('outputs/surgical-patient-sub06/verification.json')
    return [drill[0]] + [copy.deepcopy(drill[1]) for _ in range(11)] + cusa[2:]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--before', type=Path, default=Path('outputs/anatomy-integration-v3/surgical-baseline/patient.ressectionlab'))
    parser.add_argument('--after', type=Path, default=Path('outputs/anatomy-integration-v3/surgical-refined/patient.ressectionlab'))
    parser.add_argument('--output', type=Path, default=Path('outputs/anatomy-integration-v3'))
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    args.output.mkdir(parents=True, exist_ok=True)
    session = BridgeSession(args.output / 'comparison-transfers')
    def call(operation, payload):
        return session.execute(operation, payload, _Request(operation), lambda *_: None)
    try:
        before = call('loadCase', {'path': str(args.before.resolve())})
        after = call('loadCase', {'path': str(args.after.resolve())})
        recipe = {'beforeCaseHash': before['caseHash'], 'afterCaseHash': after['caseHash'],
                  'changedLayers': ['vessels'], 'prefix': pilot_prefix(root), 'decisions': 1}
        (args.output / 'comparison-recipe.json').write_text(json.dumps(recipe, indent=2) + '\n')
        report = call('compareSurgicalAnatomy', recipe)
        (args.output / 'rl-comparison.json').write_text(json.dumps(report, indent=2, allow_nan=False) + '\n')
        print(json.dumps({'before': before['caseHash'], 'after': after['caseHash'], 'delta': report['delta'],
                          'prefixStatuses': [r['status'] for r in report['before']['prefixOutcomes']],
                          'policyActions': [[r['selected'] for r in report[k]['policy']] for k in ('before', 'after')]}))
    finally:
        session.close()


if __name__ == '__main__':
    main()
