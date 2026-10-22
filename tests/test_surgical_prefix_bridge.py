"""Read-only prefix IPC cannot promote imported state into editable authority."""
import copy
from contextlib import contextmanager
import sys

import pytest

from resectionlab.core import freeze_json, thaw_json
from resectionlab.desktop_bridge import BridgeError, BridgeSession, _Request
from resectionlab.surgical_engine import SurgicalEngine


@pytest.fixture(autouse=True)
def isolated_prepared_prefix_store(tmp_path, monkeypatch):
    from resectionlab import surgical_prefix_cache
    monkeypatch.setattr(surgical_prefix_cache, 'DEFAULT_DIRECTORY', tmp_path / 'prepared-prefix')


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
        call('seekPrefix', position=0)
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
        call('seekPrefix', branchId=branch, position=0)
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
        call('seekPrefix', position=0)
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


@pytest.mark.parametrize('value', [None, 0, 1, 'true', [], {}])
def test_prepared_only_flag_must_be_an_exact_boolean_before_owner_creation(tmp_path, value):
    session, _, call, _ = episode(tmp_path)
    try:
        with step_entries() as steps, pytest.raises(ValueError, match='boolean cacheOnly'):
            call('seekPrefix', position=0, cacheOnly=value)
        assert steps == [] and not hasattr(session, 'surgical_prefix_session')
    finally:
        session.close()


@pytest.mark.parametrize('command,extra', [('seekPrefix', {'position': 1}),
    ('seekPrefixFrame', {'position': 0, 'frameIndex': 0}), ('inspect', {}),
    ('inspectArchive', {}), ('seekFrame', {'position': 0, 'frameIndex': 0})])
@pytest.mark.parametrize('value', [True, False])
def test_prepared_only_flag_is_rejected_except_at_explicit_prefix_restart(tmp_path, command, extra, value):
    session, _, call, _ = episode(tmp_path)
    try:
        with step_entries() as steps, pytest.raises(BridgeError, match='Unsupported argument'):
            call(command, **extra, cacheOnly=value)
        assert steps == [] and not hasattr(session, 'surgical_prefix_session')
    finally:
        session.close()


def test_prepared_only_reopen_keeps_read_only_authority_and_never_changes_to_fresh_on_frame_requests(tmp_path):
    session, key, call, _ = episode(tmp_path)
    original = session.cases[key].artifacts
    try:
        call('seekPrefix', position=0)
        fresh = call('seekPrefixFrame', position=0, frameIndex=0)
        final_index = fresh['playback']['frameCount'] - 1
        expected = call('seekPrefixFrame', position=0, frameIndex=final_index)
        assert fresh['verification']['preparedCache']['mode'] == 'prepare'
        assert fresh['verification']['preparedCache']['publishedActions'] == 1
        old = session.surgical_prefix_session[3]
        with step_entries() as steps:
            start = call('seekPrefix', position=0, cacheOnly=True)
            cached = call('seekPrefixFrame', position=0, frameIndex=final_index)
            repeated = call('seekPrefixFrame', position=0, frameIndex=final_index)
        assert steps == []
        assert session.surgical_prefix_session[3] is not old
        assert start['verification']['preparedCache']['mode'] == 'only'
        assert cached['verification']['preparedCache']['mode'] == repeated['verification']['preparedCache']['mode'] == 'only'
        assert cached['verification']['preparedCache']['loadedActions'] == 1
        assert cached['view'] == expected['view'] and cached['timeline'] == expected['timeline']
        assert cached['playback'] == expected['playback']
        assert cached['fullCase'] is None and cached['changed'] is False
        assert cached['verification']['readOnly'] is True
        for attribute in ('surgical_sessions', 'surgical_replay_cache', 'surgical_frame_cache'):
            assert not hasattr(session, attribute)
        assert session.cases[key].artifacts is original
        # Explicit fresh restart must perform a fresh transition despite disk data.
        with step_entries() as steps:
            fresh_start = call('seekPrefix', position=0, cacheOnly=False)
            fresh_again = call('seekPrefixFrame', position=0, frameIndex=0)
        assert steps == [0]
        assert fresh_start['verification']['preparedCache']['mode'] == 'prepare'
        assert fresh_again['verification']['preparedCache']['loadedActions'] == 0
    finally:
        session.close()


def test_prepared_only_miss_reports_actual_boundary_without_hidden_reconstruction(tmp_path):
    session, key, call, _ = episode(tmp_path)
    try:
        progress = []
        with step_entries() as steps:
            session.execute('surgicalRehearsal', {'caseHash': key, 'command': 'seekPrefix',
                            'position': 0, 'cacheOnly': True}, _Request('prepared-start'),
                            lambda _fraction, message: progress.append(message))
            missing = call('seekPrefix', position=1)
        assert steps == []
        assert any('Loading prepared observations' in message for message in progress)
        assert missing['position'] == missing['verification']['verifiedThrough'] == 0
        assert missing['requestedPosition'] == 1 and missing['budgetStopped']
        assert missing['verification']['failure']
        assert missing['verification']['preparedCache']['mode'] == 'only'
        assert missing['verification']['preparedCache']['misses'] >= 1
        assert missing['fullCase'] is None and not hasattr(session, 'surgical_sessions')
    finally:
        session.close()


def test_prepared_only_owner_replacement_cannot_silently_create_a_fresh_reconstruction(tmp_path):
    session, key, call, _ = episode(tmp_path)
    try:
        call('seekPrefix', position=0, cacheOnly=True)
        session.cases[key].artifacts = freeze_json(thaw_json(session.cases[key].artifacts))
        with step_entries() as steps, pytest.raises(BridgeError) as rejected:
            call('seekPrefixFrame', position=0, frameIndex=0)
        assert steps == [] and rejected.value.code == 'SURGICAL_PREFIX_OWNER_REQUIRED'
        assert session.surgical_prefix_session[3]._cache_only
        # A deliberate fresh restart remains available on the replacement source.
        restarted = call('seekPrefix', position=0)
        assert restarted['verification']['preparedCache']['mode'] == 'prepare'
    finally:
        session.close()


@pytest.mark.parametrize('command,extra', [('seekPrefixFrame', {'position': 0, 'frameIndex': 0}),
    ('seekPrefix', {'position': 1})])
@pytest.mark.parametrize('mode', [True, False])
def test_missing_prefix_owner_after_sidecar_restart_never_reconstructs_implicitly(tmp_path, command, extra, mode):
    session, _, call, _ = episode(tmp_path)
    try:
        call('seekPrefix', position=0, cacheOnly=mode)
        del session.surgical_prefix_session
        with step_entries() as steps, pytest.raises(BridgeError) as rejected:
            call(command, **extra)
        assert rejected.value.code == 'SURGICAL_PREFIX_OWNER_REQUIRED'
        assert steps == [] and not hasattr(session, 'surgical_prefix_session')
        restarted = call('seekPrefix', position=0, cacheOnly=mode)
        assert restarted['verification']['preparedCache']['mode'] == ('only' if mode else 'prepare')
    finally:
        session.close()


@pytest.mark.parametrize('mode', [True, False])
@pytest.mark.parametrize('replacement', ['entry', 'artifact', 'branch'])
def test_nonstart_prefix_requests_require_exact_owner_binding_in_every_mode(tmp_path, mode, replacement):
    session, key, call, _ = episode(tmp_path)
    try:
        call('seekPrefix', position=0, cacheOnly=mode)
        old = session.surgical_prefix_session
        extra = {'position': 0, 'frameIndex': 0}
        if replacement == 'entry':
            session.cases[key] = copy.copy(session.cases[key])
        elif replacement == 'artifact':
            session.cases[key].artifacts = freeze_json(thaw_json(session.cases[key].artifacts))
        else:
            artifact = thaw_json(session.cases[key].artifacts)
            artifact['surgical_toolbox']['branches']['other'] = copy.deepcopy(next(iter(artifact['surgical_toolbox']['branches'].values())))
            session.cases[key].artifacts = freeze_json(artifact)
            extra['branchId'] = 'other'
        with step_entries() as steps, pytest.raises(BridgeError) as rejected:
            call('seekPrefixFrame', **extra)
        assert rejected.value.code == 'SURGICAL_PREFIX_OWNER_REQUIRED' and steps == []
        assert session.surgical_prefix_session is old
    finally:
        session.close()
