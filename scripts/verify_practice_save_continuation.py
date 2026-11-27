"""Compare the first action after a real workspace save to its previously recorded live receipt."""

import argparse
import hashlib
import json
from pathlib import Path
import sys
from tempfile import TemporaryDirectory
from time import perf_counter_ns
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from resectionlab.desktop_bridge import BridgeSession, _Request
from resectionlab.surgical_engine import SurgicalEngine


def digest(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def probe(episode, live_receipt, output):
    expected = json.loads(live_receipt.read_text())
    receipt = json.loads(episode.with_suffix('.json').read_text())
    index = receipt['position'] + 1
    before = next(row for row in expected['actions'] if row['index'] == index)
    archive_hash = digest(episode)
    if archive_hash != receipt['sha256'] or expected['branchId'] != receipt['branchId']:
        raise ValueError('Inputs do not identify a matching preserved practice branch')
    with TemporaryDirectory() as folder:
        temp = Path(folder)
        session = BridgeSession(temp / 'transfers')

        def call(operation, **args):
            return session.execute(operation, args, _Request(operation), lambda *_: None)

        try:
            started = perf_counter_ns()
            opened = call('loadCase', path=str(episode.resolve(strict=True)))
            key = opened['caseHash']

            def command(op, **args):
                return call('surgicalRehearsal', command=op, caseHash=key, **args)

            at = command('inspect')
            cold_ms = (perf_counter_ns() - started) * 1e-6
            if (
                at['position'] != index - 1
                or at['view']['stateId'] != before['stateBefore']
                or at['branchId'] != expected['branchId']
            ):
                raise ValueError('Cold-opened source state differs from the measured live action start')
            entry = session._get_case(key)
            toolbox = entry.artifacts['surgical_toolbox']
            owner = session.surgical_sessions[key][1]
            old = entry.artifacts
            cursor = list(old.get('workspace', {}).get('cursor', (0.0, 0.0, 0.0)))
            cursor[0] += 0.125  # A real changed presentation property with identical anatomical physics.
            started = perf_counter_ns()
            call('saveCase', caseHash=key, path=str(temp / 'layout.ressectionlab'), workspace={'cursor': cursor})
            save_ms = (perf_counter_ns() - started) * 1e-6
            retained = (
                session.surgical_sessions[key][1] is owner and session.surgical_sessions[key][3] is entry.artifacts
            )
            if not retained or entry.artifacts is old or toolbox is not entry.artifacts['surgical_toolbox']:
                raise ValueError('Workspace-only save lost the exact admitted engine or immutable toolbox')
            original_restore = SurgicalEngine.restore.__func__
            restored = []

            def trace(cls, *args, **kwargs):
                restored.append(args[0].identity)
                return original_restore(cls, *args, **kwargs)

            with patch.object(SurgicalEngine, 'restore', classmethod(trace)):
                started = perf_counter_ns()
                after = command(
                    'step',
                    expectedStateId=before['stateBefore'],
                    expectedEpisodeId=at['branchId'],
                    action=before['action'],
                )
                step_ms = (perf_counter_ns() - started) * 1e-6
            if (
                restored
                or after['view']['stateId'] != before['stateAfter']
                or after['position'] != index
                or any(after['outcome'][k] != before[k] for k in ('status', 'events', 'completedFraction'))
            ):
                raise ValueError('Post-save live continuation differs from the recorded outcome or replays history')
        finally:
            session.close()
    if digest(episode) != archive_hash:
        raise ValueError('Probe changed input archive')
    result = {
        'sourceArchiveSHA256': archive_hash,
        'liveReceiptSHA256': digest(live_receipt),
        'branch': receipt['branchId'],
        'priorPosition': index - 1,
        'action': before['action'],
        'before': before['stateBefore'],
        'after': before['stateAfter'],
        'identicalStatusEventsAndProgress': True,
        'restoresDuringStep': len(restored),
        'coldLoadAndVerifyMs': cold_ms,
        'saveMs': save_ms,
        'liveBeforeFixMs': before['wallMs'],
        'liveAfterFixMs': step_ms,
    }
    output.write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('episode', 'live_receipt', 'output'):
        parser.add_argument(name, type=Path)
    args = parser.parse_args()
    probe(args.episode, args.live_receipt, args.output)
