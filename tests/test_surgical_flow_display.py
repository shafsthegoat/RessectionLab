"""Bounded historical display samples; no patient data or learned execution."""
import copy
import hashlib
import hmac
import json
import sys
from types import SimpleNamespace

import numpy as np
import pytest

from resectionlab import surgical_flow_display as flow
from resectionlab import surgical_prefix_replay as prefix
from resectionlab import surgical_prefix_cache as prepared
from resectionlab.surgical_effects import apply_effect, tick_fields
from resectionlab.surgical_engine import SurgicalEngine
from resectionlab.surgical_frame_replay import SurgicalFrameCache, _ObservedFrames
from resectionlab.surgical_replay_cache import SurgicalReplayCache
from resectionlab.surgical_scene import SurgicalScene
from resectionlab.surgical_telemetry import capture_episode
from resectionlab.core import thaw_json


def digest(text):
    return 'sha256:' + hashlib.sha256(text.encode()).hexdigest()


def scene():
    return SurgicalScene(np.zeros((8, 8, 8), np.uint16), np.eye(4), [],
        {'kind': 'generated-flow-display'}, {'bleedingSources': [
            {'id': 'declared', 'cell': 219, 'rateMm3PerSecond': 20., 'coagulationExposure': 2.}]})


def episode():
    engine = SurgicalEngine(scene())
    assert not hasattr(engine, '_flow_display_trace')
    for duration in (.6, .4, .1):
        assert engine.step({'slot': 'left', 'operation': 'inspect', 'duration': duration})['status'] == 'accepted'
    return engine


def validate(block, shape=(8, 8, 8)):
    return flow.validate_flow_display(block, scene_id=block['sceneId'], action_index=block['actionIndex'],
        frame_index=block['frameIndex'], before_state_id=block['beforeStateId'], state_id=block['stateId'],
        start_seconds=block['startSeconds'], end_seconds=block['endSeconds'], shape=shape)


def sampled():
    trace = flow.FlowDisplayTrace(digest('scene'), 0, 0., (8, 8, 8))
    initial = trace.capture(digest('before'), 0.)
    trace.record('transfer', 'fluid', 'mixed', 219, 220, .25)
    trace.commit_substep(.1)
    return initial, trace.capture(digest('after'), .1)


def counted(call):
    calls = []
    code = SurgicalEngine.step.__code__
    def profile(frame, event, arg):
        if event == 'call' and frame.f_code is code: calls.append(1)
    sys.setprofile(profile)
    try: value = call()
    finally: sys.setprofile(None)
    return value, len(calls)


@pytest.mark.parametrize('persistent', [False, True])
def test_actual_canonical_frames_preserve_complete_state_history_metrics_and_seek(tmp_path, persistent):
    engine = episode(); saved = copy.deepcopy(engine.snapshot()); history = copy.deepcopy(engine.history)
    cache = SurgicalFrameCache(directory=tmp_path); replay = SurgicalReplayCache()
    token = cache.admit(engine, saved, replay, verified_history=history) if persistent else None
    if token:
        report = capture_episode(engine, frame_collector=cache.collector(token))
        assert report['expectedStateVerified'] and cache.store.publications == 3
    def seek(index):
        return cache.seek(engine, token.snapshot if token else saved, 0, index, replay,
            verified_history=token.history if token else history, admission=token)
    _, _, first = seek(0)
    assert first['flowDisplay']['operations'] == []
    assert first['flowDisplay']['startSeconds'] == first['flowDisplay']['endSeconds']
    operations = []
    previous = first['flowDisplay']
    for index in range(1, first['frameCount']):
        observed, _, info = seek(index); block = info['flowDisplay']; validate(block)
        assert block['beforeStateId'] == previous['stateId']
        assert block['startSeconds'] == previous['endSeconds']
        assert block['stateId'] == observed.state_id
        assert all('flowDisplay' not in row for row in info['frames'])
        operations.extend(block['operations']); previous = block
    reference = SurgicalEngine(engine.scene)
    outcome = reference.step(engine.actions[0])
    assert observed.state == reference.state and observed.history == reference.history
    assert observed.metrics() == reference.metrics() and outcome['stateId'] == observed.state_id
    assert any(op['kind'] == 'deposit' and op['provenance'] == 'declared-source' for op in operations)
    assert any(op['kind'] == 'transfer' and op['provenance'] == 'mixed' for op in operations)
    (_, _, again), steps = counted(lambda: seek(1))
    assert steps == 0 and again['flowDisplay'] == seek(1)[2]['flowDisplay']
    again['flowDisplay']['operations'].clear()
    assert seek(1)[2]['flowDisplay']['operations']
    assert engine.snapshot() == saved and engine.history == history


def test_fresh_prefix_prepared_hit_and_legacy_absence_do_not_change_public_receipts(tmp_path):
    engine = episode(); store = prepared.PreparedPrefixStore(tmp_path / 'prepared')
    owner = prefix.SurgicalPrefixReplay(engine.scene, engine.snapshot(), max_new_actions=32, _prepared_store=store)
    owner.seek(3)
    assert len(owner._history) == 3
    assert owner._engine_at(3).state == engine.state
    assert [thaw_json(owner._receipt(i)) for i in range(3)] == engine.history
    fresh = prefix.SurgicalPrefixReplay(engine.scene, engine.snapshot(), max_new_actions=32,
                                       _prepared_store=store, _cache_only=True)
    (result, steps) = counted(lambda: fresh.seek_frame(0, 1))
    assert steps == 0
    block = result['playback']['flowDisplay']; validate(block)
    assert block == owner.seek_frame(0, 1)['playback']['flowDisplay']
    assert all('flowDisplay' not in row for row in result['playback']['frames'])
    assert fresh.seek_frame(0, 0)['playback']['flowDisplay']['operations'] == []
    assert fresh.seek_frame(0, 1)['playback']['flowDisplay'] == block
    value = store.load_action(owner._prepared_binding, 0)
    legacy = copy.deepcopy(value)
    for frame in legacy['frames']: frame.pop('flowDisplay', None)
    legacy_store = prepared.PreparedPrefixStore(tmp_path / 'legacy')
    assert legacy_store.publish_action(owner._prepared_binding, 0, legacy, _authority=prepared._PUBLISH_AUTHORITY)
    old = prefix.SurgicalPrefixReplay(engine.scene, engine.snapshot(), _prepared_store=legacy_store, _cache_only=True)
    assert 'flowDisplay' not in old.seek_frame(0, 1)['playback']
    # Even an authenticated wrong context must miss before exposing the block.
    directory = store._entry(owner._prepared_binding, 0)
    document = json.loads((directory / 'manifest.json').read_text())
    document['payload']['frames'][1]['flowDisplay']['beforeStateId'] = digest('poison')
    raw = json.dumps(document, sort_keys=True, separators=(',', ':')).encode('ascii')
    (directory / 'manifest.json').write_bytes(raw)
    key = (store._namespace / 'key').read_bytes()
    (directory / 'manifest.mac').write_bytes(hmac.digest(key, prepared._DOMAIN + raw, 'sha256').hex().encode('ascii'))
    assert store.load_action(owner._prepared_binding, 0) is None


def test_positive_tiny_duration_with_real_overflow_keeps_state_receipt_and_zero_time_frames(tmp_path):
    engine = SurgicalEngine(scene())
    for action in (
        {'slot': 'left', 'operation': 'exchange', 'instrument': 'frazier_suction', 'duration': .1},
        {'slot': 'left', 'operation': 'place', 'pose': {'position': [2., 3., 3.],
            'quaternion': [0., 0., 0., 1.]}, 'parameters': {'radiusMm': .4, 'flowMm3PerSecond': 1000.}, 'duration': .1},
        {'slot': 'left', 'operation': 'inspect', 'duration': 1.},
    ):
        assert engine.step(action)['status'] == 'accepted'
    start = engine.state['elapsedSeconds']; central = engine.state['fluid']['219']
    assert central > 1. and engine.state['fluid']['155'] == 1.
    assert start + 1e-17 == start
    outcome = engine.step({'slot': 'left', 'operation': 'apply', 'duration': 1e-17})
    assert outcome['status'] == 'accepted' and outcome['elapsedSeconds'] == 0.
    assert any(event['kind'] == 'aspiration' and event['fluid'] > 0 for event in outcome['events'])
    # Suction opens local capacity; the existing overcapacity actually refills
    # it even though the representable clock has not advanced.
    assert engine.state['elapsedSeconds'] == start and engine.state['fluid']['219'] < central
    saved = copy.deepcopy(engine.snapshot()); action_index = len(engine.actions) - 1
    owner = prefix.SurgicalPrefixReplay(engine.scene, saved, max_new_actions=32)
    owner.seek(len(engine.actions))
    assert owner._engine_at(len(engine.actions)).state == engine.state
    assert thaw_json(owner._receipt(action_index)) == engine.history[action_index]
    selected = owner.seek_frame(action_index, 1)['playback']['flowDisplay']
    validate(selected)
    assert selected['startSeconds'] == selected['endSeconds'] == start
    assert selected['truncated'] and selected['operations'] == []
    cache = SurgicalFrameCache(directory=tmp_path); replay = SurgicalReplayCache()
    admission = cache.admit(engine, saved, replay, verified_history=engine.history)
    report = capture_episode(engine, frame_collector=cache.collector(admission))
    assert report['expectedStateVerified'] and cache.store.publications == len(engine.actions)
    observed, _, info = cache.seek(engine, admission.snapshot, action_index, 1, replay,
        admission=admission, verified_history=admission.history)
    assert observed.snapshot() == saved and observed.history == engine.history
    assert info['flowDisplay'] == selected
    assert engine.snapshot() == saved


def test_actual_field_writes_keep_local_uptake_deposition_sediment_and_unknown_paths():
    plain, traced = SurgicalEngine(scene()), SurgicalEngine(scene())
    for engine in (plain, traced):
        engine.state['fluid'] = {'219': 3.}; engine.state['debris'] = {'219': .6}
        engine.state['slots']['left']['instrument'] = 'frazier_suction'
        engine.state['slots']['left']['parameters'] = {'radiusMm': 1., 'flowMm3PerSecond': 5.}
    trace = traced._flow_display_trace = flow.FlowDisplayTrace(traced.scene.identity, 0, 0., traced.scene.shape)
    trace.capture(traced.state_id, 0.)
    for engine in (plain, traced):
        apply_effect(engine, 'left', 'apply', [], [3., 3., 3.], [0., 0., 0., 1.], .1)
        engine.state['elapsedSeconds'] = .1
        tick_fields(engine, .1)
    assert plain.state == traced.state and plain.state_id == traced.state_id and plain.metrics() == traced.metrics()
    trace.commit_substep(.1); block = trace.capture(traced.state_id, .1); validate(block)
    assert any(op['kind'] == 'uptake' and op['fromCell'] == 219 and op['toCell'] is None for op in block['operations'])
    assert any(op['field'] == 'debris' and op['kind'] == 'transfer' for op in block['operations'])
    for op in block['operations']:
        if op['kind'] == 'transfer':
            assert traced.occupancy()[op['fromCell']] == traced.occupancy()[op['toCell']] == 0
    for engine in (plain, traced):
        engine.state['slots']['left']['instrument'] = 'irrigation'
        engine.state['slots']['left']['parameters'] = {'radiusMm': 1., 'depthMm': 1., 'flowMm3PerSecond': 5.}
        apply_effect(engine, 'left', 'apply', [], [3., 3., 3.], [0., 0., 0., 1.], .1)
        apply_effect(engine, 'left', 'apply', [], [30., 30., 30.], [0., 0., 0., 1.], .1)
        engine.state['elapsedSeconds'] = .2
    assert plain.state == traced.state
    trace.commit_substep(.2); block = trace.capture(traced.state_id, .2); validate(block)
    assert any(op['kind'] == 'deposit' and op['provenance'] == 'irrigation' and op['fromCell'] is None for op in block['operations'])
    assert any(op['kind'] == 'escape' and op['fromCell'] is None and op['toCell'] is None for op in block['operations'])


def test_failed_post_write_substep_discards_trace_with_original_rollback(monkeypatch):
    engine = _ObservedFrames(scene()); engine._frames = []; engine._next_frame_time = .2
    engine._last_effective_activation = None
    engine._capture('action_start', {'left': False, 'right': False})
    before = copy.deepcopy(engine.snapshot())
    original = SurgicalEngine._physical_substep
    def fail_after_writes(self, *args):
        original(self, *args)
        raise ValueError('PROTECTED_STRUCTURE: generated post-write rollback')
    monkeypatch.setattr(SurgicalEngine, '_physical_substep', fail_after_writes)
    outcome = engine.step({'slot': 'left', 'operation': 'inspect', 'duration': .1})
    assert outcome['status'] == 'rejected' and engine.snapshot() == before
    engine._capture('action_boundary', {'left': False, 'right': False})
    assert engine._frames[-1]['flowDisplay']['operations'] == []
    assert engine._flow_display_trace.pending == []


def test_sampling_caps_and_failed_pending_omission_do_not_leak_or_count_unboundedly():
    trace = flow.FlowDisplayTrace(digest('scene'), 0, 0., (8, 8, 8))
    trace.capture(digest('before'), 0.)
    for _ in range(1000): trace.record('uptake', 'fluid', 'mixed', 219, None, 1.)
    assert len(trace.pending) == 256 and trace.pending_truncated
    trace.discard_substep()
    assert trace.pending == [] and not trace.pending_truncated
    for _ in range(1000): trace.record('uptake', 'fluid', 'mixed', 219, None, 2**1000)
    trace.commit_substep(.1)
    block = trace.capture(digest('after'), .1); validate(block)
    assert block['truncated'] and 0 < len(block['operations']) < 256
    assert len(json.dumps(block, separators=(',', ':')).encode('ascii')) <= 64*1024
    trace.close(); trace.record('uptake', 'fluid', 'mixed', 219, None, 1.)
    assert trace.pending == []


@pytest.mark.parametrize('damage', ['schema', 'state', 'ordinal', 'extra', 'bool-cell', 'outside', 'wrap',
                                  'provenance', 'nonfinite', 'early-time', 'zero-time', 'count'])
def test_strict_flow_context_geometry_and_caps_reject_tampering(damage):
    _, block = sampled()
    if damage == 'schema': block['schema'] = 'other'
    if damage == 'state': block['stateId'] = 'bad'
    if damage == 'ordinal': block['actionIndex'] = True
    if damage == 'extra': block['operations'][0]['route'] = []
    if damage == 'bool-cell': block['operations'][0]['fromCell'] = True
    if damage == 'outside': block['operations'][0]['toCell'] = 512
    if damage == 'wrap': block['operations'][0].update(fromCell=7, toCell=8)
    if damage == 'provenance': block['operations'][0]['provenance'] = 'wound'
    if damage == 'nonfinite': block['operations'][0]['volumeMm3'] = float('inf')
    if damage == 'early-time': block['operations'][0]['startSeconds'] = -.1
    if damage == 'zero-time': block['endSeconds'] = 0.
    if damage == 'count': block['operations'] *= 257
    with pytest.raises(ValueError, match='field-transfer'): validate(block)


@pytest.mark.parametrize('omit', [False, True])
def test_optional_frame_budget_drops_samples_then_envelope_without_reexecuting_writes(omit):
    def owner(limit):
        value = SimpleNamespace(_max_frame_bytes=limit, _stream_metadata_bytes=0, _max_json_bytes=65536)
        def room(size):
            if size > limit: raise prefix.PrefixBudgetExceeded('Generated frame budget')
        value._frame_room = room
        return value
    def capture(limit):
        engine = SimpleNamespace(scene=SimpleNamespace(identity=digest('scene'), shape=(8, 8, 8)),
                                 state={'elapsedSeconds': 0., 'slots': {'left': {'active': False}}},
                                 actions=[], state_id=digest('before'))
        stage = prefix._FrameStage(owner(limit)); stage.begin(engine)
        for _ in range(256): stage.flow.record('uptake', 'fluid', 'mixed', 219, None, 1.)
        engine.state['elapsedSeconds'] = .2; engine.state_id = digest('after')
        stage.flow.commit_substep(.2); stage.observe(engine, 'left', False)
        return stage.frames
    full = capture(1024**2)
    minimal = [{**row, 'flowDisplay': copy.deepcopy(row['flowDisplay'])} for row in full]
    flow.trim_frame_flows(minimal, omit=omit)
    limit = prefix._owned_bytes(minimal) + (600 if omit else 2048)
    small = capture(limit)
    assert [row['state'] for row in small] == [row['state'] for row in full]
    assert prefix._owned_bytes(small) <= limit
    if omit:
        assert all('flowDisplay' not in row for row in small)
    else:
        assert small[-1]['flowDisplay']['truncated'] and not small[-1]['flowDisplay']['operations']
