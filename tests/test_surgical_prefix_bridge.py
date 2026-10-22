"""Read-only prefix IPC cannot promote imported state into editable authority."""
import copy
from contextlib import contextmanager
import sys

import pytest

from resectionlab.core import freeze_json, thaw_json
from resectionlab.desktop_bridge import BridgeError, BridgeSession, _Request
from resectionlab.surgical_engine import SurgicalEngine


def episode(tmp_path):
    session = BridgeSession(tmp_path / 'transfers')
    case = session.execute('createSurgicalFixture', {}, _Request('fixture'), lambda *_: None)
    key = case['caseHash']
    def call(command, **body):
        return session.execute('surgicalRehearsal', {'caseHash': key, 'command': command, **body},
                               _Request('prefix-test'), lambda *_: None)
    result = call('create')
    for _ in range(8):
        result = call('step', expectedStateId=result['view']['stateId'],
                      action={'slot': 'left', 'operation': 'inspect', 'duration': .1})
    reference = session.surgical_sessions[key][1]
    del session.surgical_sessions
    del session.surgical_replay_cache
    return session, key, call, reference


@contextmanager
def step_entries():
    if not hasattr(sys, 'monitoring'):
        pytest.skip('Python 3.12 monitoring required')
    monitor = sys.monitoring
    tool_id = next(i for i in range(5, -1, -1) if monitor.get_tool(i) is None)
    calls = []
    code = SurgicalEngine.step.__code__
    def enter(current, offset):
        if current is code:
            calls.append(len(sys._getframe(1).f_locals['self'].actions))
        else:
            return monitor.DISABLE
    monitor.use_tool_id(tool_id, 'prefix-read-only-public-steps')
    monitor.register_callback(tool_id, monitor.events.PY_START, enter)
    monitor.set_events(tool_id, monitor.events.PY_START)
    try:
        yield calls
    finally:
        monitor.set_events(tool_id, 0)
        monitor.register_callback(tool_id, monitor.events.PY_START, None)
        monitor.free_tool_id(tool_id)


def test_prefix_frames_execute_only_saved_demanded_actions_and_keep_authority_separate(tmp_path):
    session, key, call, full = episode(tmp_path)
    origin = session.cases[key].artifacts
    original = thaw_json(origin)
    branch = original['surgical_toolbox']['branchId']
    expected = full.branch(1)
    try:
        with step_entries() as calls:
            result = call('seekPrefix', branchId=branch, position=0)
            assert result['verification']['verifiedThrough'] == 0
            assert calls == []
            frame = call('seekPrefixFrame', branchId=branch, position=0, frameIndex=0)
            final_index = frame['playback']['frameCount'] - 1
            final = call('seekPrefixFrame', branchId=branch, position=0, frameIndex=final_index)
            repeated = call('seekPrefixFrame', branchId=branch, position=0, frameIndex=final_index)
            assert calls == [0], 'frame cache hits must not repeat public physics'
        assert final['view']['stateId'] == repeated['view']['stateId'] == expected.state_id
        assert final['playback']['branchId'] == branch
        assert final['branches'] == [{'id': branch, 'steps': 8}]
        assert final['verification']['readOnly'] is True
        assert final['verification']['status'] == 'reconstructed-prefix'
        assert final['verification']['verifiedThrough'] == 1
        assert session.surgical_prefix_session[3].statistics['maxFrameBytes'] == 64 * 1024**2
        assert not final['verification']['archiveEndpointVerified']
        assert final['fullCase'] is None
        assert not hasattr(session, 'surgical_sessions')
        assert not hasattr(session, 'surgical_replay_cache')
        assert not hasattr(session, 'surgical_frame_cache')
        assert session.cases[key].artifacts is origin and thaw_json(origin) == original
        final['view']['slots']['left']['position'][0] = 999
        assert call('seekPrefixFrame', branchId=branch, position=0, frameIndex=final_index)['view']['stateId'] == expected.state_id
    finally:
        session.close()
    assert not hasattr(session, 'surgical_prefix_session')


def test_far_prefix_seek_is_bounded_and_replacing_origin_restarts_the_owner(tmp_path):
    session, key, call, _ = episode(tmp_path)
    try:
        branch = session.cases[key].artifacts['surgical_toolbox']['branchId']
        first = call('seekPrefix', branchId=branch, position=8)
        assert first['position'] == 4 and first['requestedPosition'] == 8 and first['budgetStopped']
        assert first['verification']['verifiedThrough'] == 4
        old = session.surgical_prefix_session[3]
        session.cases[key].artifacts = freeze_json(thaw_json(session.cases[key].artifacts))
        fresh = call('seekPrefix', branchId=branch, position=0)
        assert fresh['verification']['verifiedThrough'] == 0
        assert session.surgical_prefix_session[3] is not old
        artifact = thaw_json(session.cases[key].artifacts)
        artifact['surgical_toolbox']['branches']['other'] = copy.deepcopy(artifact['surgical_toolbox']['branches'][branch])
        session.cases[key].artifacts = freeze_json(artifact)
        other = call('seekPrefix', branchId='other', position=0)
        assert other['branchId'] == 'other' and other['verification']['verifiedThrough'] == 0
    finally:
        session.close()


def test_prefix_requires_explicit_canonical_inspection_before_legacy_computation(tmp_path):
    session, _, call, _ = episode(tmp_path)
    try:
        prefix = call('seekPrefix', position=0)
        with step_entries() as calls, pytest.raises(BridgeError) as rejected:
            call('step', expectedStateId=prefix['view']['stateId'],
                 action={'slot': 'left', 'operation': 'inspect', 'duration': .1})
        assert rejected.value.code == 'SURGICAL_PREFIX_READ_ONLY' and calls == []
        assert not hasattr(session, 'surgical_sessions')
        canonical = call('inspect')
        accepted = call('step', expectedStateId=canonical['view']['stateId'],
                        action={'slot': 'left', 'operation': 'inspect', 'duration': .1})
        assert accepted['outcome']['status'] == 'accepted'
    finally:
        session.close()


@pytest.mark.parametrize('extra', [dict(action={'operation': 'inspect'}), dict(mapping=[]),
    dict(expectedStateId='client-state'), dict(timelineToken='old-canonical-generation')])
def test_prefix_requests_reject_authority_and_unimplemented_stream_fields(tmp_path, extra):
    session, _, call, _ = episode(tmp_path)
    try:
        with pytest.raises(BridgeError, match='Unsupported argument'):
            call('seekPrefix', position=0, **extra)
        assert not hasattr(session, 'surgical_prefix_session')
    finally:
        session.close()
