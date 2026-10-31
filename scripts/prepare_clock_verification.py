#!/usr/bin/env python3
"""Reuse or save the generated native engine control for hidden clock UI QA."""
import argparse
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from resectionlab.desktop_bridge import BridgeSession, _Request
from resectionlab.workspace_bundle import load_workspace


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=ROOT / 'testcases/shared-clock-v1/engineering.ressectionlab')
    args = parser.parse_args()
    destination = args.output.resolve()
    if not destination.is_relative_to(ROOT / 'testcases'):
        raise ValueError('Keep generated testing data under testcases/')
    destination.parent.mkdir(parents=True, exist_ok=True)
    session = BridgeSession(destination.parent / 'transfers')
    def call(operation, **values):
        return session.execute(operation, values, _Request('clock-generated-control'), lambda *_: None)
    try:
        initial = call('createSurgicalFixture')
        if destination.exists():
            case, artifacts, _ = load_workspace(destination)
            if case.semantic_hash != initial['caseHash'] or artifacts['surgical_toolbox']['branches']['main']['actions']:
                raise ValueError('Existing control differs; preserve it and choose another output')
            reused = True
        else:
            call('surgicalRehearsal', caseHash=initial['caseHash'], command='create')
            call('saveCase', caseHash=initial['caseHash'], path=str(destination), workspace={}, overwrite=False)
            reused = False
        print(json.dumps({'path': str(destination), 'caseHash': initial['caseHash'], 'reused': reused,
                          'sha256': hashlib.sha256(destination.read_bytes()).hexdigest(),
                          'scope': 'Generated software scheduling control; no acquired patient or physical validation'}))
    finally:
        session.close()


if __name__ == '__main__': main()
