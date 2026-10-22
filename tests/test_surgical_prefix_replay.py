"""Generated read-only prefix evidence; no patient replay or learning."""
import copy
from collections import OrderedDict
import json
import zlib

import numpy as np
import pytest

from resectionlab.core import freeze_json, semantic_digest
from resectionlab.surgical_cusa_v3 import default_profile
from resectionlab.surgical_engine import SurgicalEngine
from resectionlab.surgical_frame_replay import _ObservedFrames
from resectionlab.surgical_prefix_replay import PrefixBudgetExceeded, SurgicalPrefixReplay
from resectionlab.surgical_rl_protection import GuardConfig, ProtectionAnnotation, ProtectionSet
from resectionlab.surgical_scene import SurgicalScene
import resectionlab.surgical_prefix_replay as prefix


def episode():
    labels = np.zeros((20, 20, 20), np.uint16)
    labels[9, 9, 9] = 1
    scene = SurgicalScene(labels, np.eye(4),
        [{'id': 1, 'name': 'Generated coupon', 'material': 'tumor', 'target': True}],
        {'kind': 'generated prefix replay test; not patient anatomy'}, {'cusaV3': default_profile()})
    mask = np.zeros(scene.shape, bool)
    mask[1, 1, 1] = True
    protection = ProtectionSet(scene, [ProtectionAnnotation.from_mask(scene, 'brainstem', mask,
        provenance='Generated distant exclusion coupon; not clinical anatomy')], clearance_mm=0)
    engine = SurgicalEngine(scene, protection=protection, guard_config=GuardConfig())
    boundaries = [engine.view(include_history=False)]
    actions = [
        {'slot': 'left', 'operation': 'exchange', 'instrument': 'cusa', 'duration': .1},
        {'slot': 'left', 'operation': 'place', 'duration': .1,
         'pose': {'position': [9., 9., 3.], 'quaternion': [0., 0., 0., 1.]}},
        {'slot': 'left', 'operation': 'apply', 'duration': .05,
         'pose': {'position': [9., 9., 11.], 'quaternion': [0., 0., 0., 1.]}},
        {'slot': 'left', 'operation': 'apply', 'duration': .6,
         'pose': {'position': [9., 9., 11.], 'quaternion': [0., 0., 0., 1.]}},
        {'slot': 'right', 'operation': 'inspect', 'duration': .3},
    ]
    for action in actions:
        assert engine.step(action)['status'] in ('accepted', 'partial')
        boundaries.append(engine.view(include_history=False))
    assert engine.history[2]['status'] == 'partial'
    assert engine.state['removed']
    return engine, boundaries


def test_prefix_and_causal_frames_match_public_execution_once_and_detach(monkeypatch):
    engine, boundaries = episode()
    saved = engine.snapshot()
    original = copy.deepcopy(saved)
    calls = {'step': 0, 'fresh': 0, 'view': 0}
    step, fresh, view = prefix._PrefixEngine.step, SurgicalPrefixReplay._fresh, prefix._PrefixEngine.view
    def counted_step(self, *args, **kwargs):
        calls['step'] += 1
        return step(self, *args, **kwargs)
    def counted_fresh(self):
        calls['fresh'] += 1
        return fresh(self)
    def counted_view(self, *args, **kwargs):
        calls['view'] += 1
        return view(self, *args, **kwargs)
    monkeypatch.setattr(prefix._PrefixEngine, 'step', counted_step)
    monkeypatch.setattr(SurgicalPrefixReplay, '_fresh', counted_fresh)
    monkeypatch.setattr(prefix._PrefixEngine, 'view', counted_view)
    replay = SurgicalPrefixReplay(engine.scene, saved)
    # Detach the imported record at admission, including nested checkpoint data.
    saved['checkpoint']['state']['travelMm'] = -999
    saved['actions'][0]['duration'] = 9
    result = replay.seek(3)
    assert result['view'] == boundaries[3]
    assert result['timeline'] == engine.history[:3]
    assert result['verification']['archiveEndpointVerified'] is False
    assert result['verification']['originSnapshotHash'] == semantic_digest(original)
    assert result['verification']['actionLogHash'] == semantic_digest(original['actions'])
    assert result['verification']['protectionContractHash'] == semantic_digest(original['protectionContract'])
    assert calls['step'] == 3 and calls['fresh'] == 1
    # Independently obtain actual successful public microsteps for action 3,
    # starting with pending CUSA work and finishing after causal cell removal.
    observed = _ObservedFrames(engine.scene, protection=engine.protection, guard_config=engine.guard_config)
    observed._frames = []
    observed._next_frame_time = .2
    for action in original['actions'][:3]:
        observed.step(action)
    observed._frames = []
    observed._next_frame_time = observed.state['elapsedSeconds'] + .2
    observed._capture('action_start', {key: bool(slot['active']) for key, slot in observed.state['slots'].items()})
    observed._last_effective_activation = None
    observed.step(original['actions'][3])
    observed._capture('action_boundary', observed._last_effective_activation or
                      {key: bool(slot['active']) for key, slot in observed.state['slots'].items()})
    first = replay.seek_frame(3, 0)
    assert first['playback']['frameCount'] == len(observed._frames)
    assert len(observed._frames) > 2
    for index, frame in enumerate(observed._frames):
        before_views = calls['view']
        result = replay.seek_frame(3, index)
        state = json.loads(zlib.decompress(frame['state']))
        observed.state = state
        assert result['playback']['frameStateId'] == observed.state_id
        assert result['playback']['observedTimeSeconds'] == frame['observedTimeSeconds']
        assert result['playback']['effectiveActivation'] == frame['effectiveActivation']
        assert result['view']['cusaV3'] == state['cusaV3']
        assert calls['view'] == before_views + 1
        assert calls['step'] == 4 and calls['fresh'] == 1
        assert all('state' not in row for row in result['playback']['frames'])
    result['view']['cusaV3']['progressExact'].clear()
    result['timeline'][0]['action']['duration'] = 99
    result['playback']['frames'][0]['effectiveActivation']['left'] = 'caller edit'
    assert replay.seek(3)['view'] == boundaries[3]
    final = replay.seek(5)
    assert final['view'] == boundaries[5]
    assert final['timeline'] == engine.history
    assert final['verification']['archiveEndpointVerified'] is True
    assert calls['step'] == 5 and calls['fresh'] == 1
    assert engine.snapshot() == original
    json.dumps(final, allow_nan=False)


def test_action_budget_stops_at_real_boundary_and_poisoned_future_endpoint_never_promotes():
    engine, boundaries = episode()
    saved = engine.snapshot()
    saved['checkpoint']['state']['travelMm'] += 123
    saved['checkpoint']['stateId'] = saved['stateId'] = 'sha256:poisoned-future'
    replay = SurgicalPrefixReplay(engine.scene, saved, max_new_actions=2)
    for expected in (2, 4, 5):
        result = replay.seek(5)
        assert result['position'] == expected and result['requestedPosition'] == 5
        assert result['budgetStopped'] is (expected < 5)
        assert result['view'] == boundaries[expected]
        assert result['verification']['verifiedThrough'] == expected
        assert result['verification']['archiveEndpointVerified'] is False
        if expected < 5:
            assert result['verification']['endpointValidationError'] is None
    assert 'differs' in result['verification']['endpointValidationError']
    assert result['readOnly'] and result['verification']['totalActions'] == 5


def test_cancellation_keeps_prior_committed_prefix_and_retries_only_interrupted_action():
    engine, boundaries = episode()
    replay = SurgicalPrefixReplay(engine.scene, engine.snapshot(), max_new_actions=1)
    replay.seek(1)
    checks = 0
    def cancel():
        nonlocal checks
        checks += 1
        if checks == 6:
            raise ValueError('generated cancellation')
    with pytest.raises(ValueError, match='generated cancellation'):
        replay.seek(3, check=cancel)
    result = replay.seek(1)
    assert result['view'] == boundaries[1]
    assert result['verification']['failure']['actionIndex'] == 1
    assert replay.statistics['frameActions'] == [0]
    resumed = replay.seek(2)
    assert resumed['view'] == boundaries[2]
    assert resumed['verification']['failure'] is None


def test_future_rejection_reports_original_index_without_publishing_failed_frames():
    engine, boundaries = episode()
    saved = engine.snapshot()
    saved['actions'][2]['duration'] = 0
    replay = SurgicalPrefixReplay(engine.scene, saved)
    result = replay.seek(5)
    assert result['position'] == 2 and result['view'] == boundaries[2]
    assert result['verification']['failure']['actionIndex'] == 2
    assert 'Saved action rejected' in result['verification']['failure']['reason']
    assert 2 not in replay.statistics['frameActions']
    assert replay.seek(5)['position'] == 2
    assert engine.snapshot()['actions'][2]['duration'] == .05


def test_protection_source_and_loaded_prefix_implementation_invalidation(monkeypatch):
    engine, _ = episode()
    saved = engine.snapshot()
    invalid = copy.deepcopy(saved)
    invalid['protectionContract']['config']['require_critical_coverage'] = False
    with pytest.raises(ValueError, match='Protection contract integrity'):
        SurgicalPrefixReplay(engine.scene, invalid)
    replay = SurgicalPrefixReplay(engine.scene, saved)
    replay.seek(1)
    original = prefix._FrameStage.capture
    def replacement(self, *args, **kwargs):
        return original(self, *args, **kwargs)
    monkeypatch.setattr(prefix._FrameStage, 'capture', replacement)
    with pytest.raises(ValueError, match='implementation changed'):
        replay.seek_frame(0, 0)
    monkeypatch.undo()
    replay = SurgicalPrefixReplay(engine.scene, saved)
    replay.seek(1)
    engine.scene.source['changed'] = True
    with pytest.raises(ValueError, match='Source scene changed'):
        replay.seek_frame(0, 0)


def test_base_fingerprint_unavailable_and_changed_are_refused(monkeypatch):
    engine, _ = episode()
    monkeypatch.setattr(prefix, '_fingerprint', lambda: None)
    with pytest.raises(ValueError, match='fingerprint unavailable'):
        SurgicalPrefixReplay(engine.scene, engine.snapshot())
    # Keep the same loaded helper while changing its external fingerprint result.
    fingerprint = ['implementation-a']
    monkeypatch.setattr(prefix, '_fingerprint', lambda: fingerprint[0])
    replay = SurgicalPrefixReplay(engine.scene, engine.snapshot())
    fingerprint[0] = 'implementation-b'
    with pytest.raises(ValueError, match='implementation changed'):
        replay.seek(1)
    assert replay.statistics['verifiedThrough'] == 0


def test_memory_limits_stop_during_capture_and_never_publish_or_repeat_failed_action():
    engine, boundaries = episode()
    # Inspect produces many observed states without modifying patient physics.
    saved = engine.snapshot()
    saved['actions'][2] = {'slot': 'right', 'operation': 'inspect', 'duration': 10.}
    replay = SurgicalPrefixReplay(engine.scene, saved, max_new_actions=1, max_frame_bytes=8000)
    replay.seek(1)
    replay.seek(2)
    result = replay.seek(3)
    assert result['position'] == 2 and result['view'] == boundaries[2]
    assert result['verification']['failure']['type'] == 'PrefixBudgetExceeded'
    assert result['verification']['failure']['actionIndex'] == 2
    assert 2 not in replay.statistics['frameActions']
    for key in ('frame', 'checkpoint', 'history'):
        assert replay.statistics[key + 'Bytes'] <= replay.statistics['max' + key.title() + 'Bytes']
    before = copy.deepcopy(replay.statistics)
    assert replay.seek(3)['position'] == 2
    assert replay.statistics == before
    # Bounded receipts independently refuse progress after one valid action.
    full = SurgicalPrefixReplay(engine.scene, engine.snapshot(), max_new_actions=1)
    full.seek(1)
    one_row = full.statistics['historyBytes']
    small = SurgicalPrefixReplay(engine.scene, engine.snapshot(), max_history_bytes=one_row + 32)
    result = small.seek(4)
    assert result['position'] == 1
    assert 'history retention' in result['verification']['failure']['reason']
    assert small.statistics['historyBytes'] <= one_row + 32


def test_evicted_frames_decline_without_second_execution_and_input_caps_are_explicit():
    engine, _ = episode()
    replay = SurgicalPrefixReplay(engine.scene, engine.snapshot(), max_frame_bytes=4500)
    replay.seek(2)
    assert replay.statistics['verifiedThrough'] == 2
    assert 0 not in replay.statistics['frameActions']
    # A frame miss cannot secretly reconstruct an earlier action again.
    with pytest.raises(ValueError, match='frames evicted'):
        replay.seek_frame(0, 0)
    assert replay.statistics['verifiedThrough'] == 2
    with pytest.raises(PrefixBudgetExceeded, match='input budget'):
        SurgicalPrefixReplay(engine.scene, engine.snapshot(), max_input_bytes=1024)
    with pytest.raises(PrefixBudgetExceeded, match='Checkpoint|Compressed'):
        SurgicalPrefixReplay(engine.scene, engine.snapshot(), max_checkpoint_bytes=128)
    for keyword, value in [('max_new_actions', 33), ('max_frame_bytes', 64 * 1024**2 + 1),
                           ('max_history_bytes', True), ('max_json_bytes', 0)]:
        with pytest.raises(ValueError, match='hard ceilings'):
            SurgicalPrefixReplay(engine.scene, engine.snapshot(), **{keyword: value})


def test_checkpoint_eviction_and_decoded_state_caps_are_separate():
    engine, boundaries = episode()
    full = SurgicalPrefixReplay(engine.scene, engine.snapshot())
    full.seek(5)
    full.seek(5)
    largest = max(prefix._owned_bytes(OrderedDict([(index, item)]))
                  for index, item in full._checkpoints.items())
    replay = SurgicalPrefixReplay(engine.scene, engine.snapshot(), max_checkpoint_bytes=largest + 64)
    replay.seek(5)
    result = replay.seek(5)
    assert result['view'] == boundaries[5]
    assert replay.statistics['checkpointBytes'] <= largest + 64
    assert 0 not in replay.statistics['checkpointPositions']
    with pytest.raises(ValueError, match='boundary evicted'):
        replay.seek(0)
    # The compressed-state cap cannot conceal an oversized decoded state.
    with pytest.raises(PrefixBudgetExceeded, match='Decoded JSON'):
        SurgicalPrefixReplay(engine.scene, engine.snapshot(), max_json_bytes=32)


def test_private_borrowed_input_reuses_exact_frozen_branch_without_widening_authority():
    engine, boundaries = episode()
    frozen = freeze_json(engine.snapshot())
    replay = SurgicalPrefixReplay._borrow_frozen_snapshot(engine.scene, frozen,
        max_input_bytes=128 * 1024**2)
    assert replay._snapshot is frozen
    assert replay.statistics['immutableInputBorrowed']
    assert replay.statistics['ownedImmutableInputBytes'] == 0
    assert replay.statistics['sharedImmutableInputBytes'] == prefix._owned_bytes(frozen)
    result = replay.seek(3)
    assert replay._history[0]['action'] is frozen['actions'][0]
    assert result['view'] == boundaries[3]
    assert result['readOnly'] and not result['verification']['archiveEndpointVerified']
    assert result['verification']['originSnapshotHash'] == semantic_digest(frozen)
    ordinary = SurgicalPrefixReplay(engine.scene, frozen)
    assert ordinary._snapshot is not frozen
    with pytest.raises(ValueError, match='recursively frozen'):
        SurgicalPrefixReplay._borrow_frozen_snapshot(engine.scene, engine.snapshot())
    with pytest.raises(ValueError, match='authority differs'):
        SurgicalPrefixReplay(engine.scene, frozen, _frozen_input_authority=object())


def test_private_timeline_projection_sees_immutable_full_receipts_before_json_copy():
    engine, boundaries = episode()
    replay = SurgicalPrefixReplay(engine.scene, engine.snapshot())
    def display(rows):
        assert rows[1]['action']['pose']['position'] == (9., 9., 3.)
        with pytest.raises(TypeError):
            rows[0]['action']['duration'] = 0
        return [{'status': row['status'], 'stateAfter': row['stateAfter']} for row in rows]
    result = replay.seek(3, _timeline_projection=display)
    assert result['view'] == boundaries[3]
    assert result['timeline'] == [{'status': row['status'], 'stateAfter': row['stateAfter']}
                                 for row in engine.history[:3]]
    frame = replay.seek_frame(2, 0, _timeline_projection=display)
    assert frame['timeline'] == result['timeline']
    assert replay.seek(3)['timeline'] == engine.history[:3]
