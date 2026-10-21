"""Cancellation stays a transaction boundary, including ValueError callbacks."""
import json

import numpy as np
import pytest

from resectionlab.desktop_bridge import BridgeError, BridgeRuntime, BridgeSession, _Request
from resectionlab.surgical_engine import SurgicalEngine
from resectionlab.surgical_scene import SurgicalScene


def engine():
    return SurgicalEngine(SurgicalScene(np.zeros((8, 8, 8), np.uint16), np.eye(4), [], {}, {}))


def action():
    return dict(slot='left', operation='inspect', duration=.6)


def encoded(e):
    return json.dumps(e.snapshot(), sort_keys=True, separators=(',', ':'))


def interrupt_after_substeps(monkeypatch, request, *, domain_failure=False):
    original = SurgicalEngine._substep
    calls = []
    def wrapped(self, *args):
        result = original(self, *args)
        calls.append(self.state['elapsedSeconds'])
        if len(calls) == 2:
            request.cancel()
            if domain_failure: raise ValueError('COLLISION_TEST_ONLY')
        return result
    monkeypatch.setattr(SurgicalEngine, '_substep', wrapped)
    return calls


@pytest.mark.parametrize('method', ['step', 'preview'])
@pytest.mark.parametrize('domain_failure', [False, True])
def test_mid_action_request_cancellation_propagates_without_partial_commit(monkeypatch, method, domain_failure):
    e = engine(); before = encoded(e); request = _Request('cancel-generated-action')
    calls = interrupt_after_substeps(monkeypatch, request, domain_failure=domain_failure)
    with pytest.raises(BridgeError) as exc:
        getattr(e, method)(action(), check=request.check)
    assert exc.value.code == 'CANCELLED'
    assert len(calls) == 2
    assert encoded(e) == before


def test_once_only_value_error_from_check_is_never_a_surgical_rejection():
    e = engine(); before = encoded(e); failure = ValueError('one-shot callback cancellation'); calls = 0
    def check():
        nonlocal calls
        calls += 1
        if calls == 3: raise failure
    with pytest.raises(ValueError) as exc:
        e.step(action(), check=check)
    assert exc.value is failure and encoded(e) == before


def test_cancellation_after_finalization_still_withholds_authority(monkeypatch):
    import resectionlab.surgical_effects as effects
    e = engine(); before = encoded(e); request = _Request('cancel-finalize')
    original = effects.finalize_action
    def finalize(candidate):
        original(candidate); request.cancel()
    monkeypatch.setattr(effects, 'finalize_action', finalize)
    with pytest.raises(BridgeError) as exc: e.step(action(), check=request.check)
    assert exc.value.code == 'CANCELLED' and encoded(e) == before


@pytest.mark.parametrize('method', ['restore', 'branch'])
def test_archive_reconstruction_checks_inside_each_action(monkeypatch, method):
    full = engine(); assert full.step(action())['status'] == 'accepted'
    before = encoded(full); request = _Request('cancel-archive')
    calls = interrupt_after_substeps(monkeypatch, request)
    with pytest.raises(BridgeError) as exc:
        if method == 'restore': SurgicalEngine.restore(full.scene, full.snapshot(), check=request.check)
        else: full.branch(1, check=request.check)
    assert exc.value.code == 'CANCELLED' and len(calls) == 2 and encoded(full) == before


@pytest.mark.parametrize('command', ['step', 'preview'])
def test_cancelled_bridge_action_publishes_no_artifact_or_committed_engine(tmp_path, monkeypatch, command):
    session = BridgeSession(tmp_path / 'transfers')
    def call(operation, body, request=None):
        return session.execute(operation, body, request or _Request(operation), lambda *_: None)
    try:
        case = call('createSurgicalFixture', {})
        initial = call('surgicalRehearsal', dict(caseHash=case['caseHash'], command='create'))
        current = session.surgical_sessions[case['caseHash']][1]
        before = encoded(current); artifacts = session.cases[case['caseHash']].artifacts
        request = _Request('cancel-bridge-action')
        calls = interrupt_after_substeps(monkeypatch, request)
        with pytest.raises(BridgeError) as exc:
            call('surgicalRehearsal', dict(caseHash=case['caseHash'], command=command,
                expectedStateId=initial['view']['stateId'], action=action()), request)
        assert exc.value.code == 'CANCELLED' and not request.committing
        assert len(calls) == 2 and session.cases[case['caseHash']].artifacts is artifacts
        assert session.surgical_sessions[case['caseHash']][1] is current and encoded(current) == before
    finally:
        session.close()


def test_runtime_cancel_is_acknowledged_and_never_emits_a_later_action_result(tmp_path, monkeypatch):
    events = []; runtime = BridgeRuntime(tmp_path / 'transfers', events.append)
    def prepare(operation, body):
        return runtime.session.execute(operation, body, _Request(operation), lambda *_: None)
    try:
        case = prepare('createSurgicalFixture', {})
        initial = prepare('surgicalRehearsal', dict(caseHash=case['caseHash'], command='create'))
        current = runtime.session.surgical_sessions[case['caseHash']][1]
        before = encoded(current); original = SurgicalEngine._substep; calls = 0
        def substep(self, *args):
            nonlocal calls
            result = original(self, *args); calls += 1
            if calls == 2:
                runtime.submit(dict(id='cancel-action', op='cancel', args=dict(requestId='long-action')))
            return result
        monkeypatch.setattr(SurgicalEngine, '_substep', substep)
        runtime.submit(dict(id='long-action', op='surgicalRehearsal', args=dict(caseHash=case['caseHash'],
            command='step', expectedStateId=initial['view']['stateId'], action=action())))
        assert runtime.wait_idle(5)
        assert [event['event'] for event in events if event['id'] == 'long-action'] == ['started', 'cancelled']
        assert next(event for event in events if event['id'] == 'cancel-action')['result']['cancelled']
        assert calls == 2 and encoded(current) == before
    finally:
        runtime.close()


def test_rl_action_wrapper_preserves_cancellation_and_all_attempt_ledgers(monkeypatch):
    pytest.importorskip('gymnasium')
    from resectionlab.surgical_rl_env import SurgicalRLEnv, RLConfig
    labels = np.zeros((12, 12, 12), np.uint16); labels[3:6, 3:6, 3:6] = 1
    source = SurgicalScene(labels, np.eye(4), [dict(id=1, name='Generated target', material='tumor', target=True)], {}, {})
    env = SurgicalRLEnv(source, config=RLConfig(require_critical_coverage=False)); env.reset(seed=17)
    before = encoded(env.engine); measurements = env.measurements.copy(); request = _Request('cancel-rl-wrapper')
    interrupt_after_substeps(monkeypatch, request)
    with pytest.raises(BridgeError) as exc: env.step_action(action(), check=request.check)
    assert exc.value.code == 'CANCELLED' and encoded(env.engine) == before
    assert env.attempts == [] and env.total_reward == 0 and env.measurements == measurements
