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


def test_prefix_bridge_uses_committed_display_rows_and_complete_receipt_accessor(tmp_path, monkeypatch):
    from resectionlab.surgical_bridge import display_timeline
    session, key, call, _ = episode(tmp_path)
    try:
        first = call('seekPrefixFrame', position=0, frameIndex=0)
        owner = session.surgical_prefix_session[3]
        assert owner.statistics['historyReceiptsCompressed']
        assert owner.statistics['displayRowsCompressed']
        complete = owner._receipt(0)
        assert first['timeline'] == display_timeline((complete,))
        assert complete['action'] is owner._snapshot['actions'][0]
        assert set(owner._history[0]) == {'status', 'reason'}
        def forbidden(*_args, **_kwargs):
            raise AssertionError('Acknowledged bridge frame decoded receipt history')
        monkeypatch.setattr(owner, '_receipt', forbidden)
        monkeypatch.setattr(owner, '_unpack_receipt', forbidden)
        monkeypatch.setattr(owner, '_materialize_history', forbidden)
        warm = call('seekPrefixFrame', position=0, frameIndex=0,
                    prefixToken=first['timelineToken'], prefixFieldTokens=first['viewFieldTokens'])
        assert warm['timelineOmitted'] and 'timeline' not in warm
        assert warm['verification'] == first['verification']
        assert warm['fullCase'] is None and not warm['changed']
    finally:
        session.close()


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
    dict(expectedStateId='client-state'), dict(timelineToken='old-canonical-generation'),
    dict(prefixToken='old-prefix-generation'), dict(prefixFieldTokens={})])
def test_prefix_requests_reject_authority_and_unimplemented_stream_fields(tmp_path, extra):
    session, _, call, _ = episode(tmp_path)
    try:
        with pytest.raises(BridgeError, match='Unsupported argument'):
            call('seekPrefix', position=0, **extra)
        assert not hasattr(session, 'surgical_prefix_session')
    finally:
        session.close()


def test_prefix_stream_omits_only_acknowledged_current_generation_display_data(tmp_path):
    session, key, call, _ = episode(tmp_path)
    original = session.cases[key].artifacts
    try:
        with step_entries() as steps:
            first = call('seekPrefixFrame', position=0, frameIndex=0)
            token, fields = first['timelineToken'], first['viewFieldTokens']
            assert token.startswith('prefix-replay.v1/')
            assert set(fields) == {'flapComponents'}
            compact = call('seekPrefixFrame', position=0, frameIndex=0,
                           prefixToken=token, prefixFieldTokens=fields)
            assert steps == [0]
        assert compact['timelineOmitted'] is True and 'timeline' not in compact
        assert compact['viewFieldsOmitted'] == ['flapComponents']
        assert not {'scene', 'catalog', 'protection', 'flapComponents'}.intersection(compact['view'])
        assert compact['view']['stateId'] == first['view']['stateId']
        restored = {**compact['view'], **{name: first['view'][name] for name in
                    ('scene', 'catalog', 'protection', 'flapComponents')}}
        assert restored == first['view']
        assert compact['verification'] == first['verification']
        assert compact['fullCase'] is None and compact['verification']['readOnly'] is True

        unbound = call('seekPrefixFrame', position=0, frameIndex=0, prefixFieldTokens=fields)
        assert unbound['timelineOmitted'] is False and unbound['viewFieldsOmitted'] == []
        assert unbound['view'] == first['view']
        changed_field = call('seekPrefixFrame', position=0, frameIndex=0,
                             prefixToken=token, prefixFieldTokens={'flapComponents': '0' * 64})
        assert changed_field['timelineOmitted'] is True and changed_field['viewFieldsOmitted'] == []
        assert changed_field['view']['flapComponents'] == first['view']['flapComponents']

        # A stale generation never suppresses the field or the complete context.
        stale = call('seekPrefixFrame', position=0, frameIndex=0,
                     prefixToken='prefix-replay.v1/stale', prefixFieldTokens=fields)
        assert stale['timelineOmitted'] is False and stale['viewFieldsOmitted'] == []
        assert stale['view'] == first['view'] and stale['timeline'] == first['timeline']
        changed = call('seekPrefixFrame', position=1, frameIndex=0,
                       prefixToken=token, prefixFieldTokens=fields)
        assert changed['timelineToken'] != token and changed['timelineOmitted'] is False
        assert len(changed['timeline']) == changed['verification']['verifiedThrough'] == 2
        assert changed['viewFieldsOmitted'] == []
        call('seekPrefix', position=0)
        restarted = call('seekPrefixFrame', position=0, frameIndex=0,
                         prefixToken=token, prefixFieldTokens=fields)
        assert restarted['timelineToken'] != token and restarted['timelineOmitted'] is False
        assert restarted['view'] == first['view']
        assert session.cases[key].artifacts is original
        for attribute in ('surgical_sessions', 'surgical_replay_cache', 'surgical_frame_cache'):
            assert not hasattr(session, attribute)
    finally:
        session.close()


@pytest.mark.parametrize('extra', [dict(prefixToken=7), dict(prefixToken=''),
    dict(prefixToken='x' * 1025), dict(prefixFieldTokens=[]),
    dict(prefixFieldTokens={'fluid': '0' * 64}),
    dict(prefixFieldTokens={'flapComponents': 'A' * 64})])
def test_invalid_prefix_stream_acknowledgements_are_rejected_before_owner_creation(tmp_path, extra):
    session, _, call, _ = episode(tmp_path)
    try:
        with pytest.raises(ValueError, match='Invalid read-only prefix'):
            call('seekPrefixFrame', position=0, frameIndex=0, **extra)
        assert not hasattr(session, 'surgical_prefix_session')
    finally:
        session.close()


@pytest.mark.parametrize('command', ['inspect', 'inspectArchive', 'seek', 'seekFrame', 'step'])
def test_prefix_display_acknowledgements_cannot_cross_into_canonical_commands(tmp_path, command):
    session, _, call, _ = episode(tmp_path)
    try:
        with pytest.raises(BridgeError, match='Unsupported argument'):
            call(command, prefixToken='prefix-replay.v1/forged', prefixFieldTokens={})
        assert not hasattr(session, 'surgical_prefix_session')
        assert not hasattr(session, 'surgical_sessions')
    finally:
        session.close()
