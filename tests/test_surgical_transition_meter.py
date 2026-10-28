"""Exact public native-call accounting without altering the engine class."""
import json
import sys

import numpy as np
import pytest

from resectionlab.surgical_engine import SurgicalEngine
from resectionlab.core import semantic_digest
from resectionlab.surgical_modeled_outcomes import summarize_engine_outcomes
from resectionlab.surgical_scene import SurgicalScene
from resectionlab.surgical_transition_meter import (
    MeteredEngine, TransitionBudgetExceeded, TransitionMeter,
    TransitionReserveReached)
from resectionlab.surgical_verified_cache import implementation_fingerprint


def _engine():
    labels = np.zeros((3, 3, 3), np.uint16)
    labels[1, 1, 1] = 1
    scene = SurgicalScene(labels, np.eye(4),
        [{'id': 1, 'name': 'source target', 'material': 'tumor', 'target': True}],
        {'caseHash': 'generated transition meter test'}, {})
    return SurgicalEngine(scene)


INSPECT = {'slot': 'left', 'operation': 'inspect', 'duration': .2}
INVALID = {'slot': 'left', 'operation': 'unsupported', 'duration': .2}


def test_meter_matches_original_public_code_objects_across_clones_and_rejections():
    original = _engine()
    untouched = original.snapshot()
    fingerprint = implementation_fingerprint()
    assert fingerprint is not None
    meter = TransitionMeter(10)
    wrapped = meter.wrap(original)
    assert not isinstance(wrapped, SurgicalEngine)
    counts = {'previewCalls': 0, 'previewInternalStepCalls': 0,
              'directStepCalls': 0}
    preview_code, step_code = SurgicalEngine.preview.__code__, SurgicalEngine.step.__code__

    def profile(frame, event, arg):
        if event != 'call':
            return
        if frame.f_code is preview_code:
            counts['previewCalls'] += 1
        elif frame.f_code is step_code:
            key = ('previewInternalStepCalls' if frame.f_back is not None and
                   frame.f_back.f_code is preview_code else 'directStepCalls')
            counts[key] += 1

    sys.setprofile(profile)
    try:
        with meter.phase('access'):
            access = wrapped.clone()
            assert access.preview(INSPECT, include_view=False)['status'] == 'accepted'
            assert access.step(INSPECT)['status'] == 'accepted'
            assert access.step(INVALID)['status'] == 'rejected'
        with meter.phase('replay'):
            replay = access.fresh()
            assert replay.step(INSPECT)['status'] == 'accepted'
    finally:
        sys.setprofile(None)
    report = meter.report()
    assert counts == {'previewCalls': 1, 'previewInternalStepCalls': 1,
                      'directStepCalls': 3}
    assert report['physicalCalls'] == 4 == counts['previewCalls'] + counts['directStepCalls']
    assert report['byPhase']['access']['physicalCalls'] == 3
    assert report['byPhase']['replay']['physicalCalls'] == 1
    assert report['branches'][str(access.branch_id)]['parentBranchId'] == wrapped.branch_id
    assert report['branches'][str(replay.branch_id)]['origin'] == 'fresh'
    assert semantic_digest(report) == semantic_digest(json.loads(json.dumps(report)))
    assert report['trace'][2]['status'] == 'rejected'
    assert report['trace'][2]['action'] == INVALID
    assert original.snapshot() == untouched
    assert implementation_fingerprint() == fingerprint


def test_hard_limit_reserve_and_cancellation_retain_only_checked_commits():
    meter = TransitionMeter(2)
    base = meter.wrap(_engine())
    route = base.clone()
    with meter.phase('optional'):
        with meter.reserve(1):
            assert route.step(INSPECT)['status'] == 'accepted'
            with pytest.raises(TransitionReserveReached):
                route.preview(INSPECT, include_view=False)
        assert meter.remaining_calls == 1
    with meter.phase('cleanup'):
        assert route.step(INSPECT)['status'] == 'accepted'
        with pytest.raises(TransitionBudgetExceeded):
            route.step(INSPECT)
    saved = meter.last_committed_engine_for('cleanup')
    assert saved is not None and len(saved.actions) == 2
    assert saved.state_id == route.state_id
    assert base.actions == []
    assert meter.report()['physicalCalls'] == 2
    assert len(meter.report()['trace']) == 2

    def cancelled():
        raise KeyboardInterrupt('user cancelled')

    interrupted = TransitionMeter(2, check=cancelled)
    source = interrupted.wrap(_engine())
    before = source.state_id
    with pytest.raises(KeyboardInterrupt, match='user cancelled'):
        source.step(INSPECT)
    assert interrupted.calls == 0 and source.state_id == before

    counter = {'checks': 0}

    def interrupt_in_flight():
        counter['checks'] += 1
        if counter['checks'] == 2:
            raise KeyboardInterrupt('during native call')

    interrupted = TransitionMeter(2, check=interrupt_in_flight)
    source = interrupted.wrap(_engine())
    before = source.state_id
    with pytest.raises(KeyboardInterrupt, match='during native call'):
        source.step(INSPECT)
    assert interrupted.calls == 1 and source.state_id == before
    assert interrupted.report()['trace'][0]['exceptionType'] == 'KeyboardInterrupt'
    assert interrupted.last_committed_engine_for('unlabeled') is None


def test_read_only_outcome_summary_unwraps_only_the_known_proxy():
    meter = TransitionMeter(4)
    source = meter.wrap(_engine())
    continuation = source.clone()
    assert continuation.step(INSPECT)['status'] == 'accepted'
    before = meter.calls
    expected = summarize_engine_outcomes(continuation.unwrap(),
                                          baseline=source.unwrap())
    actual = summarize_engine_outcomes(continuation, baseline=source)
    assert actual == expected and meter.calls == before
    assert type(continuation) is MeteredEngine

    class Fake:
        def unwrap(self):
            return continuation.unwrap()

    with pytest.raises(TypeError, match='authoritative SurgicalEngine'):
        summarize_engine_outcomes(Fake(), baseline=source)
    with pytest.raises(ValueError, match='baseline must share'):
        summarize_engine_outcomes(continuation, baseline=Fake())
    with pytest.raises(AttributeError, match='metered public'):
        continuation.replay([INSPECT])
