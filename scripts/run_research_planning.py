#!/usr/bin/env python3
"""Reproduce the same opted-in development comparison offered by Electron.

Uses existing acquired source bytes and an existing checked public-action prefix.
All new cases and testing records stay in Git-ignored testcases/.
"""
import argparse
import json
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from resectionlab.desktop_bridge import BridgeSession, _Request
from resectionlab.core import freeze_json, thaw_json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, default=ROOT / 'testcases/cases/UCSF-PDGM-0004-standard-v2.ressectionlab')
    parser.add_argument('--prefix', type=Path, default=ROOT / 'outputs/surgical-pilot-v1/ucsf0004-prefix.json')
    parser.add_argument('--protection', type=Path, default=ROOT / 'testcases/environments/UCSF-PDGM-0004/rl-protection.json')
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--opponent', choices=('stop', 'checkpoint'), default='checkpoint')
    args = parser.parse_args()
    output = args.output.resolve()
    if not output.is_relative_to(ROOT / 'testcases'):
        parser.error('New app-testing cases and records must be stored under testcases/')
    if output.exists():
        parser.error('Choose a new output directory; previous results are preserved')
    # Fail before creating output if the required existing inputs are absent.
    if not args.source.is_file() or not args.prefix.is_file() or not args.protection.is_file():
        parser.error('The existing source bundle, source exclusion contract or checked prefix is unavailable')
    prefix = json.loads(args.prefix.read_text())
    output.mkdir(parents=True)
    started = time.perf_counter()
    session = BridgeSession(output / 'transfers')
    def call(operation, payload):
        return session.execute(operation, payload, _Request(operation), lambda fraction, message, *_: print(message, flush=True))
    try:
        source = call('loadCase', {'path': str(args.source.resolve())})
        binding = {'caseHash': source['caseHash']}
        state = call('surgicalRehearsal', {**binding, 'command': 'inspect'})
        if state is None or state['position'] != 0:
            raise ValueError('Use an intact prepared source environment; existing rehearsals are never erased')
        # Older intact bundles kept their exclusions beside the file. Bind those
        # exact supplied source masks to a NEW empty engine in this disposable
        # session; never change the input bundle or carry an unconfigured engine
        # into planning. Every subsequent pose/tissue change is a public action.
        from resectionlab.surgical_environment import load_environment
        from resectionlab.surgical_engine import SurgicalEngine
        from resectionlab.surgical_rl_protection import GuardConfig, load_protection
        from resectionlab.surgical_planning import SOURCE_CONTRACTS
        scene = load_environment(args.source)
        engine = SurgicalEngine(scene, protection=load_protection(scene, args.protection, strict=True), guard_config=GuardConfig())
        if engine.guard_report()['contractId'] != SOURCE_CONTRACTS.get(scene.identity):
            raise ValueError('Source/protection pair has no admission for this comparison')
        entry = session._get_case(source['caseHash'])
        artifacts = thaw_json(entry.artifacts)
        record = artifacts['surgical_toolbox']
        if any(branch['actions'] for branch in record['branches'].values()):
            raise ValueError('The source contains existing action branches; use its preserved intact source version')
        record.update(branches={'main': engine.snapshot()}, branchId='main', position=0,
                      parents={'main': None}, timeline={'main': []})
        entry.artifacts = freeze_json(artifacts)
        state = call('surgicalRehearsal', {**binding, 'command': 'inspect'})
        for index, row in enumerate(prefix['attempts']):
            state = call('surgicalRehearsal', {**binding, 'command': 'step', 'expectedStateId': state['view']['stateId'], 'action': row['action']})
            if state['outcome']['status'] != row['status'] or state['view']['stateId'] != row['stateId']:
                raise ValueError('The checked source prefix no longer matches: decision ' + str(index))
        if state['view']['stateId'] != prefix['stateId']:
            raise ValueError('Final source prefix identity differs')
        state = call('surgicalRehearsal', {**binding, 'command': 'plan', 'expectedStateId': state['view']['stateId'],
                    'acknowledgeResearchUnknowns': True, 'opponent': args.opponent})
        case_path = output / 'comparison.ressectionlab'
        call('saveCase', {**binding, 'path': str(case_path), 'overwrite': False})
        report = {'sourcePath': str(args.source.resolve()), 'prefixPath': str(args.prefix.resolve()),
                  'savedCase': str(case_path), 'caseHash': source['caseHash'], 'planning': state['planning'],
                  'totalWallSecondsIncludingSourceLoadAndPrefix': time.perf_counter() - started}
        (output / 'result.json').write_text(json.dumps(report, indent=2, allow_nan=False) + '\n')
        print(json.dumps({'savedCase': str(case_path), 'totalWallSeconds': report['totalWallSecondsIncludingSourceLoadAndPrefix'],
              'alternatives': [{k: plan[k] for k in ('method', 'stopReason', 'incrementalEndpoints', 'invalidActions')}
                               for plan in state['planning']['alternatives']]}, indent=2))
    except Exception as error:
        (output / 'failure.json').write_text(json.dumps({'error': str(error), 'type': type(error).__name__,
            'wallSeconds': time.perf_counter() - started, 'optimizerUpdates': 0}, indent=2) + '\n')
        raise
    finally:
        session.close()


if __name__ == '__main__':
    main()
