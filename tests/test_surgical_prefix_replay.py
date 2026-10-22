"""Generated read-only prefix evidence; no patient replay or learning."""
import copy
from collections import OrderedDict
import json
import zlib
from types import ModuleType, SimpleNamespace

import numpy as np
import pytest

from resectionlab.core import freeze_json, thaw_json, semantic_digest
from resectionlab.surgical_cusa_v3 import default_profile
from resectionlab.surgical_engine import SurgicalEngine
from resectionlab.surgical_frame_replay import _ObservedFrames
from resectionlab.surgical_prefix_replay import PrefixBudgetExceeded, SurgicalPrefixReplay
from resectionlab.surgical_rl_protection import GuardConfig, ProtectionAnnotation, ProtectionSet
from resectionlab.surgical_scene import SurgicalScene
import resectionlab.surgical_prefix_replay as prefix


def hash_shell():
    """Unadmitted hash input only; no scene/engine constructor or action."""
    engine = object.__new__(prefix._PrefixEngine)
    engine.scene = SimpleNamespace(identity='sha256:' + '1' * 64)
    engine._protection_contract = {'contentHash': 'sha256:' + '2' * 64}
    engine.state = {'z': [-0.0, 1.25], 'é': {'b': 2, 'a': [True, None]}}
    return engine


def test_private_native_hash_recomputes_complete_scene_state_and_protection(monkeypatch):
    engine = hash_shell()
    calls = []
    def encode(value, limit):
        calls.append(copy.deepcopy(value))
        assert limit == 16 * 1024**2
        return json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False)
    monkeypatch.setattr(prefix._native_json, 'try_encode_canonical', encode)
    assert prefix._stock_prefix_hash_path(engine)
    original = engine.state_id
    assert prefix._prefix_state_id(engine) == original
    assert calls[0] == {'scene': engine.scene.identity, 'state': engine.state,
                        'protectionContract': engine._protection_contract['contentHash']}
    engine.state['é']['a'].append(17)
    assert prefix._prefix_state_id(engine) == engine.state_id != original
    assert len(calls) == 2


def test_private_native_hash_decline_retains_legacy_and_errors_propagate(monkeypatch):
    engine = hash_shell()
    calls = []
    monkeypatch.setattr(prefix._native_json, 'try_encode_canonical',
                        lambda value, limit: calls.append(limit))
    assert prefix._prefix_state_id(engine) == engine.state_id
    assert calls == [16 * 1024**2]
    def unavailable(*_):
        raise ValueError('configured native identity unavailable')
    monkeypatch.setattr(prefix._native_json, 'try_encode_canonical', unavailable)
    with pytest.raises(ValueError, match='configured native identity unavailable'):
        prefix._prefix_state_id(engine)


@pytest.mark.parametrize('override', ['own-property', 'base-property', 'lookup', 'digest-binding',
    'digest-code', 'property-code', 'json-function', 'encoder-method', 'json-keywords', 'builtin-shadow'])
def test_private_native_hash_retains_overridden_property_semantics(monkeypatch, override):
    engine = hash_shell()
    def forbidden(*_):
        raise AssertionError('custom hash semantics entered native writer')
    monkeypatch.setattr(prefix._native_json, 'try_encode_canonical', forbidden)
    if override == 'own-property':
        monkeypatch.setattr(prefix._PrefixEngine, 'state_id', property(lambda self: 'custom own property'))
    elif override == 'base-property':
        monkeypatch.setattr(SurgicalEngine, 'state_id', property(lambda self: 'custom base property'))
    elif override == 'lookup':
        def lookup(self, name):
            return 'custom attribute lookup' if name == 'state_id' else object.__getattribute__(self, name)
        monkeypatch.setattr(prefix._PrefixEngine, '__getattribute__', lookup)
    elif override == 'digest-binding':
        monkeypatch.setattr(prefix._engine_module, '_owned_digest', lambda value: 'custom digest binding')
    elif override == 'digest-code':
        def changed(value):
            return 'custom digest code'
        monkeypatch.setattr(prefix._engine_module._owned_digest, '__code__', changed.__code__)
    elif override == 'property-code':
        def changed(self):
            return 'custom property code'
        monkeypatch.setattr(prefix._hash_property.fget, '__code__', changed.__code__)
    elif override == 'json-function':
        monkeypatch.setattr(json, 'dumps', lambda *args, **kwargs: '{"custom":true}')
    elif override == 'encoder-method':
        monkeypatch.setattr(json.JSONEncoder, 'iterencode',
                            lambda self, value, _one_shot=False: iter(['{"custom":true}']))
    elif override == 'json-keywords':
        monkeypatch.setitem(json.dumps.__kwdefaults__, 'ensure_ascii', False)
    else:
        monkeypatch.setattr(prefix._engine_module, 'dict', object, raising=False)
    expected = engine.state_id
    assert not prefix._stock_prefix_hash_path(engine)
    assert prefix._prefix_state_id(engine) == expected
    assert prefix._prefix_state_id(SimpleNamespace(state_id='external engine')) == 'external engine'


def test_private_native_hash_declines_encoder_staticmethod_binding_change(monkeypatch):
    engine = hash_shell()
    def forbidden(*_):
        raise AssertionError('custom descriptor binding entered native writer')
    monkeypatch.setattr(prefix._native_json, 'try_encode_canonical', forbidden)
    monkeypatch.setattr(json.JSONEncoder, 'iterencode', staticmethod(json.JSONEncoder.iterencode))
    assert not prefix._stock_prefix_hash_path(engine)
    with pytest.raises(TypeError) as legacy:
        _ = engine.state_id
    with pytest.raises(TypeError) as actual:
        prefix._prefix_state_id(engine)
    assert str(actual.value) == str(legacy.value)


def test_private_native_hash_declines_static_getattribute_binding(monkeypatch):
    engine = hash_shell()
    def forbidden(*_):
        raise AssertionError('custom attribute binding entered native writer')
    monkeypatch.setattr(prefix._native_json, 'try_encode_canonical', forbidden)
    monkeypatch.setattr(prefix._PrefixEngine, '__getattribute__', staticmethod(object.__getattribute__))
    assert not prefix._stock_prefix_hash_path(engine)
    with pytest.raises(TypeError) as legacy:
        _ = engine.state_id
    with pytest.raises(TypeError) as actual:
        prefix._prefix_state_id(engine)
    assert str(actual.value) == str(legacy.value)


@pytest.mark.parametrize('raises', [False, True])
def test_private_native_hash_preserves_encoder_default_descriptor_effects(monkeypatch, raises):
    engine = hash_shell()
    calls = []
    def changed_default(self):
        calls.append('default lookup')
        if raises:
            raise RuntimeError('custom default lookup failure')
        engine.state['fromDefaultLookup'] = len(calls)
        return lambda value: None
    def forbidden(*_):
        raise AssertionError('custom default descriptor entered native writer')
    monkeypatch.setattr(prefix._native_json, 'try_encode_canonical', forbidden)
    monkeypatch.setattr(json.JSONEncoder, 'default', property(changed_default))
    assert not prefix._stock_prefix_hash_path(engine)
    assert calls == []
    if raises:
        with pytest.raises(RuntimeError, match='custom default lookup failure'):
            prefix._prefix_state_id(engine)
        assert calls == ['default lookup']
    else:
        expected = engine.state_id
        calls.clear()
        engine.state.pop('fromDefaultLookup')
        assert prefix._prefix_state_id(engine) == expected
        assert calls == ['default lookup']


@pytest.mark.parametrize('module', [json, prefix.hashlib, prefix._hash_encoder_module],
                         ids=['json', 'hashlib', 'json-encoder'])
def test_private_native_hash_declines_custom_module_lookup_without_guard_side_effects(monkeypatch, module):
    engine = hash_shell()
    calls = []
    class CustomModule(ModuleType):
        def __getattribute__(self, name):
            if name in ('dumps', 'JSONEncoder', 'sha256', 'c_make_encoder'):
                calls.append(name)
            return ModuleType.__getattribute__(self, name)
    def forbidden(*_):
        raise AssertionError('custom module dispatch entered native writer')
    monkeypatch.setattr(prefix._native_json, 'try_encode_canonical', forbidden)
    monkeypatch.setattr(module, '__class__', CustomModule)
    assert not prefix._stock_prefix_hash_path(engine)
    assert calls == []
    expected = engine.state_id
    legacy_calls = list(calls)
    calls.clear()
    assert prefix._prefix_state_id(engine) == expected
    assert calls == legacy_calls


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


def bone_episode():
    """Public declared rigid-cell drill law with a retained remote component."""
    labels = np.zeros((20, 20, 20), np.uint16)
    labels[9, 9, 9] = 1
    labels[1:3, 15:17, 15:17] = 1
    scene = SurgicalScene(labels, np.eye(4),
        [{'id': 1, 'name': 'Generated bone coupon', 'material': 'bone'}],
        {'kind': 'generated display-stream test; no patient anatomy'},
        {'tissuePhysics': {'schema': 'resectionlab.tissue-physics.v1',
            'provenance': {'kind': 'assumed-research', 'description': 'Generated display-stream coupon'},
            'bone': {'model': 'rigid-cell-removal-v1'}}})
    engine = SurgicalEngine(scene)
    for action in [
        {'slot': 'left', 'operation': 'place', 'duration': .1,
         'pose': {'position': [9., 9., 3.], 'quaternion': [0., 0., 0., 1.]},
         'parameters': {'radiusMm': 1.}},
        {'slot': 'left', 'operation': 'apply', 'duration': .6,
         'pose': {'position': [9., 9., 11.], 'quaternion': [0., 0., 0., 1.]}},
        {'slot': 'right', 'operation': 'inspect', 'duration': .3},
    ]:
        assert engine.step(action)['status'] == 'accepted'
    assert len(engine.state['removed']) == 1
    assert engine.state['flapComponents']
    return engine


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
    assert replay._receipt(0)['action'] is frozen['actions'][0]
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


def test_compressed_complete_receipts_share_actions_and_metric_rows_without_losing_content():
    engine, boundaries = episode()
    replay = SurgicalPrefixReplay(engine.scene, engine.snapshot())
    replay.seek(4)
    assert replay._history[0] is replay._history[1]
    assert set(replay._history[0]) == {'status', 'reason'}
    assert replay.statistics['historyBytes'] == replay._history_store_bytes(replay._history_store)
    for index, receipt in enumerate(engine.history[:4]):
        full = replay._receipt(index)
        assert full['action'] is replay._snapshot['actions'][index]
        assert thaw_json(full) == receipt
        assert 'action' not in json.loads(zlib.decompress(replay._receipts[index][0]))
        with pytest.raises(TypeError):
            full['status'] = 'invented'
        assert replay.seek(index + 1)['view']['metrics'] == boundaries[index + 1]['metrics']
    # The full API still returns complete canonical rows rather than compact
    # metrics/display rows, and returned JSON cannot mutate compressed storage.
    result = replay.seek(4)
    result['timeline'][2]['events'].clear()
    assert thaw_json(replay._receipt(2)) == engine.history[2]


def test_row_local_display_is_compressed_once_and_warm_frames_decode_no_history(monkeypatch):
    from resectionlab.surgical_bridge import display_receipt, display_timeline
    engine, _ = episode()
    calls = []
    def project(row):
        calls.append(row['stateAfter'])
        with pytest.raises(TypeError):
            row['action']['duration'] = 0
        return display_receipt(row)
    replay = SurgicalPrefixReplay(engine.scene, engine.snapshot(), _display_row_projection=project)
    result = replay.seek(3)
    assert len(calls) == 3
    assert result['timeline'] == display_timeline(engine.history[:3])
    assert len(replay._display_receipts) == 3
    assert thaw_json(replay._receipt(2)) == engine.history[2]
    first = replay.seek_frame(2, 0)
    def forbidden(*_args, **_kwargs):
        raise AssertionError('Warm compact replay decoded or rebuilt receipt history')
    monkeypatch.setattr(replay, '_unpack_receipt', forbidden)
    monkeypatch.setattr(replay, '_materialize_history', forbidden)
    monkeypatch.setattr(replay, '_receipt', forbidden)
    warm = replay.seek_frame(2, 0, prefix_token=first['timelineToken'])
    assert warm['timelineOmitted'] and 'timeline' not in warm
    restored = {**warm['view'], **{name: first['view'][name] for name in ('scene', 'catalog', 'protection')}}
    assert restored == first['view'] and len(calls) == 3


@pytest.mark.parametrize('change', ['compressed', 'digest', 'length', 'trailing', 'ordinal'])
def test_compressed_receipt_corruption_is_refused(change):
    engine, _ = episode()
    replay = SurgicalPrefixReplay(engine.scene, engine.snapshot())
    replay.seek(2)
    blob, length, digest = replay._receipts[0]
    if change == 'compressed': blob = b'invalid zlib'
    elif change == 'digest': digest = b'0' * 32
    elif change == 'length': length -= 1
    elif change == 'trailing': blob += b'trailing'
    else: blob, length, digest = replay._receipts[1]
    replay._receipts[0] = (blob, length, digest)
    with pytest.raises(ValueError, match='receipt'):
        replay._receipt(0)


def test_receipt_codec_preserves_exact_float_sign_and_bounds_decoded_data():
    import math
    engine, _ = episode()
    replay = SurgicalPrefixReplay(engine.scene, engine.snapshot())
    value = {'negativeZero': -0.0, 'number': 1, 'flag': True, 'events': [{'values': [0.0, -0.0]}]}
    entry = replay._pack_receipt(value)
    restored = replay._unpack_receipt(entry)
    assert type(restored['number']) is int and type(restored['flag']) is bool
    assert math.copysign(1, restored['negativeZero']) == -1
    assert math.copysign(1, restored['events'][0]['values'][1]) == -1
    forged = (entry[0], replay._max_json_bytes + 1, entry[2])
    with pytest.raises(ValueError, match='descriptor'):
        replay._unpack_receipt(forged)


def test_projection_cost_is_in_same_history_quota_and_failed_row_never_publishes():
    engine, _ = episode()
    def large_projection(row):
        return {'uncompressed': list(range(400)), 'status': row['status']}
    full = SurgicalPrefixReplay(engine.scene, engine.snapshot(), _display_row_projection=large_projection)
    full.seek(1)
    cap = full.statistics['historyBytes'] + 32
    replay = SurgicalPrefixReplay(engine.scene, engine.snapshot(), max_history_bytes=cap,
                                 _display_row_projection=large_projection)
    stopped = replay.seek(4)
    assert stopped['position'] == 1 and stopped['budgetStopped']
    assert len(replay._history) == len(replay._receipts) == len(replay._display_receipts) == 1
    assert replay.statistics['historyBytes'] <= cap
    assert 1 not in replay.statistics['frameActions']
    assert 2 not in replay.statistics['checkpointPositions']
    before = replay.statistics
    assert replay.seek(4)['position'] == 1 and replay.statistics == before


def test_materialized_history_has_separate_explicit_bound_without_losing_stored_receipts():
    engine, _ = episode()
    replay = SurgicalPrefixReplay(engine.scene, engine.snapshot())
    replay.seek(3)
    original = replay._receipt(0)
    replay._max_json_bytes = sum(entry[1] for entry in replay._receipts) - 1
    with pytest.raises(PrefixBudgetExceeded, match='Transient complete history JSON'):
        replay._materialize_history()
    assert replay._receipt(0) == original


def test_failed_frame_proposal_does_not_publish_or_evict_committed_checkpoint(monkeypatch):
    engine, boundaries = episode()
    replay = SurgicalPrefixReplay(engine.scene, engine.snapshot())
    replay.seek(1)
    checkpoints, store = replay._checkpoints, replay._history_store
    prepared = []
    propose_checkpoint = replay._propose_checkpoint
    def one_checkpoint_proposal(position, current, item):
        # Exercise a proposal that really would evict the current cursor. The
        # injected failure must leave the entire original map unpublished.
        budget = replay._max_checkpoint_bytes
        replay._max_checkpoint_bytes = prefix._owned_bytes(OrderedDict([(position, item)]))
        try:
            proposal = propose_checkpoint(position, current, item)
        finally:
            replay._max_checkpoint_bytes = budget
        assert list(proposal) == [2]
        prepared.append(proposal)
        return proposal
    def failed_frames(*_):
        assert prepared
        raise MemoryError('Injected frame proposal allocation failure')
    monkeypatch.setattr(replay, '_propose_checkpoint', one_checkpoint_proposal)
    monkeypatch.setattr(replay, '_propose_frames', failed_frames)
    with pytest.raises(MemoryError, match='Injected frame proposal'):
        replay.seek(2)
    assert replay._checkpoints is checkpoints and replay._history_store is store
    assert 1 in replay._checkpoints and 2 not in replay._checkpoints
    assert len(replay._history) == len(replay._receipts) == 1
    assert replay.seek(1)['view'] == boundaries[1]
    assert replay.statistics['maxMaterializedHistoryGraphBytes'] == replay._max_json_bytes
    assert 'may coexist' in replay.statistics['materializedHistoryLimitMeaning']


def test_light_implementation_exception_permanently_invalidates_prefix_receipts(monkeypatch):
    from resectionlab.surgical_observed_cache import ImplementationChanged
    engine, _ = episode()
    replay = SurgicalPrefixReplay(engine.scene, engine.snapshot())
    replay.seek(1)
    original = prefix._light_seal
    def changed():
        raise ImplementationChanged('Native source changed after load')
    monkeypatch.setattr(prefix, '_light_seal', changed)
    with pytest.raises(prefix.PrefixInvalidated, match='implementation changed'):
        replay._receipt(0)
    monkeypatch.setattr(prefix, '_light_seal', original)
    with pytest.raises(prefix.PrefixInvalidated, match='implementation changed'):
        replay.seek_frame(0, 0)


def test_codec_identity_drift_permanently_invalidates_warm_prefix(monkeypatch):
    engine, _ = episode()
    replay = SurgicalPrefixReplay(engine.scene, engine.snapshot())
    replay.seek(1)
    original = prefix._native_json.strict_live_identity
    def changed(*, full=False):
        raise ValueError('Native JSON build changed')
    monkeypatch.setattr(prefix._native_json, 'strict_live_identity', changed)
    with pytest.raises(prefix.PrefixInvalidated, match='implementation changed'):
        replay.seek_frame(0, 0)
    monkeypatch.setattr(prefix._native_json, 'strict_live_identity', original)
    with pytest.raises(prefix.PrefixInvalidated, match='implementation changed'):
        replay.seek(1)


def test_codec_seal_checks_full_identity_before_action_publication(monkeypatch):
    calls = []
    original = prefix._native_json.strict_live_identity
    def checked(*, full=False):
        calls.append(full)
        return original(full=full)
    monkeypatch.setattr(prefix._native_json, 'strict_live_identity', checked)
    assert prefix._prefix_seal(full=True) == prefix._prefix_seal()
    calls.clear()
    engine, _ = episode()
    replay = SurgicalPrefixReplay(engine.scene, engine.snapshot())
    assert calls.count(True) >= 1
    calls.clear()
    replay.seek(1)
    assert True in calls
    calls.clear()
    replay.seek_frame(0, 0)
    assert calls and not any(calls)


def test_prefix_seal_rejects_imported_alias_with_forged_globals(monkeypatch):
    from types import FunctionType
    engine, _ = episode()
    replay = SurgicalPrefixReplay(engine.scene, engine.snapshot())
    replay.seek(1)
    original = prefix.freeze_json
    forged = FunctionType(original.__code__, dict(original.__globals__),
                          original.__name__, original.__defaults__, original.__closure__)
    forged.__module__ = original.__module__
    monkeypatch.setattr(prefix, 'freeze_json', forged)
    with pytest.raises(prefix.PrefixInvalidated, match='implementation changed'):
        replay.seek_frame(0, 0)


def test_prefix_frame_acknowledgements_hydrate_exact_full_view_and_hash_only_once(monkeypatch):
    engine = bone_episode()
    counts = {'identities': 0, 'omitted': []}
    identity_getter = SurgicalEngine.state_id.fget
    view = prefix._PrefixEngine.view
    def counted_identity(self):
        counts['identities'] += 1
        return identity_getter(self)
    def checked_view(self, *args, **kwargs):
        assert kwargs.get('_state_id') is not None
        counts['omitted'].append(kwargs.get('_omit_fields'))
        return view(self, *args, **kwargs)
    monkeypatch.setattr(prefix._PrefixEngine, 'state_id', property(counted_identity))
    monkeypatch.setattr(prefix._PrefixEngine, 'view', checked_view)
    replay = SurgicalPrefixReplay(engine.scene, engine.snapshot())
    replay.seek(2)
    last_frame = replay.seek_frame(1, 0)['playback']['frameCount'] - 1
    before = counts['identities']
    full = replay.seek_frame(1, last_frame)
    assert counts['identities'] == before + 1
    assert full['view']['flapComponents']
    assert full['timelineToken'].startswith('prefix-replay.v1/')
    assert len(full['timelineToken']) <= 1024
    assert not full['timelineOmitted'] and full['viewFieldsOmitted'] == []
    digest = semantic_digest(full['view']['flapComponents'])[7:]
    assert full['viewFieldTokens'] == {'flapComponents': digest}
    def no_projection(_):
        raise AssertionError('A compact hit copied/projected the resident timeline')
    before = counts['identities']
    compact = replay.seek_frame(1, last_frame, prefix_token=full['timelineToken'],
        prefix_field_tokens=full['viewFieldTokens'], _timeline_projection=no_projection)
    assert counts['identities'] == before + 1
    assert counts['omitted'][-1] == ['flapComponents']
    assert compact['timelineOmitted'] and 'timeline' not in compact
    assert compact['viewFieldsOmitted'] == ['flapComponents']
    for name in ('scene', 'catalog', 'protection', 'flapComponents'):
        assert name not in compact['view']
    # The prefix-specific client preserves history=[] instead of synthesizing
    # canonical playback history. Everything else hydrates by exact token.
    hydrated = copy.deepcopy(compact)
    hydrated['timeline'] = full['timeline']
    for name in ('scene', 'catalog', 'protection', 'flapComponents'):
        hydrated['view'][name] = full['view'][name]
    assert hydrated['view'] == full['view']
    for name in ('verification', 'playback', 'position', 'requestedPosition', 'readOnly', 'budgetStopped'):
        assert compact[name] == full[name]
    before = counts['identities']
    replay.seek(2)
    assert counts['identities'] == before + 1
    assert not compact['verification']['archiveEndpointVerified']


def test_prefix_stream_generation_rotates_on_new_actions_and_different_owners():
    engine = bone_episode()
    replay = SurgicalPrefixReplay(engine.scene, engine.snapshot())
    last_frame = replay.seek_frame(1, 0)['playback']['frameCount'] - 1
    full = replay.seek_frame(1, last_frame)
    token, fields = full['timelineToken'], full['viewFieldTokens']
    for stale in (None, 'canonical-timeline-token', 'prefix-replay.v1/' + '0' * 64):
        response = replay.seek_frame(1, last_frame, prefix_token=stale, prefix_field_tokens=fields)
        assert not response['timelineOmitted'] and response['timeline'] == full['timeline']
        assert response['view'] == full['view'] and response['viewFieldsOmitted'] == []
    # A correct prefix generation can suppress context/timeline, while a stale
    # field digest still requires sending the actual complete field.
    wrong_field = replay.seek_frame(1, last_frame, prefix_token=token,
        prefix_field_tokens={'flapComponents': '0' * 64})
    assert wrong_field['timelineOmitted'] and wrong_field['viewFieldsOmitted'] == []
    assert wrong_field['view']['flapComponents'] == full['view']['flapComponents']
    changed = replay.seek_frame(1, 0, prefix_token=token, prefix_field_tokens=fields)
    assert changed['timelineOmitted'] and changed['viewFieldsOmitted'] == []
    assert changed['viewFieldTokens'] != fields and changed['view']['flapComponents'] == []
    replay.seek(3)
    advanced = replay.seek_frame(1, last_frame, prefix_token=token, prefix_field_tokens=fields)
    assert advanced['timelineToken'] != token and not advanced['timelineOmitted']
    assert advanced['verification']['verifiedThrough'] == 3 and len(advanced['timeline']) == 3
    assert advanced['view']['flapComponents'] == full['view']['flapComponents']
    other = SurgicalPrefixReplay(engine.scene, engine.snapshot())
    response = other.seek_frame(1, last_frame, prefix_token=token, prefix_field_tokens=fields)
    assert response['timelineToken'] != token and not response['timelineOmitted']
    assert response['verification']['originSnapshotHash'] == full['verification']['originSnapshotHash']
    # Tokens never bypass source invalidation or grant endpoint authority.
    engine.scene.source['edited'] = True
    with pytest.raises(ValueError, match='Source scene changed'):
        replay.seek_frame(1, last_frame, prefix_token=advanced['timelineToken'], prefix_field_tokens=fields)


@pytest.mark.parametrize('values', [
    {'prefix_token': ''}, {'prefix_token': 42}, {'prefix_token': 'x' * 1025},
    {'prefix_field_tokens': []}, {'prefix_field_tokens': {'removed': 'a' * 64}},
    {'prefix_field_tokens': {'flapComponents': 'A' * 64}},
    {'prefix_field_tokens': {'flapComponents': 'a' * 63}},
    {'prefix_field_tokens': {'flapComponents': 'sha256:' + 'a' * 64}},
    {'prefix_field_tokens': {'flapComponents': None}},
])
def test_malformed_prefix_display_acknowledgements_fail_before_reconstruction(values):
    engine, _ = episode()
    replay = SurgicalPrefixReplay(engine.scene, engine.snapshot())
    with pytest.raises(ValueError, match='Invalid prefix'):
        replay.seek_frame(0, 0, **values)
    assert replay.statistics['verifiedThrough'] == 0
    assert replay.statistics['frameActions'] == []


def test_stream_metadata_is_immutable_detached_and_charged_to_frame_cap():
    engine = bone_episode()
    replay = SurgicalPrefixReplay(engine.scene, engine.snapshot())
    full = replay.seek_frame(1, 0)
    frame = replay._frames[1][0]
    with pytest.raises(TypeError):
        frame['fieldTokens']['flapComponents'] = '0' * 64
    assert all('fieldTokens' not in row for row in full['playback']['frames'])
    assert replay.statistics['frameBytes'] == (prefix._owned_bytes(replay._frames)
                                               + replay.statistics['streamMetadataBytes'])
    full['viewFieldTokens']['flapComponents'] = '0' * 64
    full['playback']['frames'][0]['effectiveActivation']['left'] = 'caller edit'
    fresh = replay.seek_frame(1, 0)
    assert fresh['viewFieldTokens'] == dict(frame['fieldTokens'])
    assert fresh['playback']['frames'][0]['effectiveActivation']['left'] is False
    # The bound includes token maps and the nonce, even on a tiny action window.
    window = max(prefix._owned_bytes(OrderedDict([(index, frames)]))
                 for index, frames in replay._frames.items())
    cap = window + replay.statistics['streamMetadataBytes'] + 1024
    limited = SurgicalPrefixReplay(engine.scene, engine.snapshot(), max_frame_bytes=cap)
    limited.seek(2)
    assert limited.statistics['verifiedThrough'] == 2
    assert limited.statistics['frameBytes'] <= cap
    assert 0 not in limited.statistics['frameActions']


def assert_frame_accounting(replay):
    exact = prefix._owned_bytes(replay._frames)
    assert replay._retained_frame_bytes == exact
    assert replay._stream_metadata_bytes == prefix._owned_bytes(replay._stream_nonce)
    assert replay._frame_bytes() == exact + replay._stream_metadata_bytes
    assert replay.statistics['frameBytes'] == replay._frame_bytes()
    assert replay._frame_bytes() <= replay.statistics['maxFrameBytes']


def test_committed_frame_counter_is_exact_and_unchanged_room_reads_do_not_rescan(monkeypatch):
    engine = bone_episode()
    replay = SurgicalPrefixReplay(engine.scene, engine.snapshot())
    assert_frame_accounting(replay)
    for index in (1, 2):
        replay.seek(index)
        assert_frame_accounting(replay)
    before = replay.statistics['frameBytes']
    scans = []
    original = prefix._owned_bytes
    def observed(value, seen=None):
        if value is replay._frames:
            scans.append(value)
        return original(value, seen)
    # No guarded engine read occurs under this instrumentation: only quota
    # callbacks and statistics over already committed immutable frame graphs.
    monkeypatch.setattr(prefix, '_owned_bytes', observed)
    for _ in range(100):
        replay._frame_room(1024)
        assert replay.statistics['frameBytes'] == before
    assert scans == []


def test_exact_counter_survives_eviction_cancellation_retry_and_failed_capture():
    engine = bone_episode()
    reference = SurgicalPrefixReplay(engine.scene, engine.snapshot())
    reference.seek(2)
    window = max(prefix._owned_bytes(OrderedDict([(index, frames)]))
                 for index, frames in reference._frames.items())
    cap = window + reference.statistics['streamMetadataBytes'] + 1024
    replay = SurgicalPrefixReplay(engine.scene, engine.snapshot(), max_frame_bytes=cap)
    replay.seek(2)
    assert_frame_accounting(replay)
    assert replay.statistics['frameActions'] == [1]
    def cancel_after_eviction():
        if not replay.statistics['frameActions']:
            raise RuntimeError('cancel after private committed-frame eviction')
    with pytest.raises(RuntimeError, match='cancel after private'):
        replay.seek(3, check=cancel_after_eviction)
    assert replay.statistics['verifiedThrough'] == 2
    assert replay.statistics['frameActions'] == []
    assert_frame_accounting(replay)
    result = replay.seek(3)
    assert result['verification']['archiveEndpointVerified']
    assert_frame_accounting(replay)
    saved = engine.snapshot()
    saved['actions'][2] = {'slot': 'right', 'operation': 'inspect', 'duration': 10.}
    failed = SurgicalPrefixReplay(engine.scene, saved, max_frame_bytes=cap)
    result = failed.seek(3)
    assert result['position'] == 2
    assert result['verification']['failure']['type'] == 'PrefixBudgetExceeded'
    assert 2 not in failed.statistics['frameActions']
    assert_frame_accounting(failed)


def test_exact_publication_counter_includes_map_growth_aliases_and_atomic_overflow():
    # Generated private quota data only: no engine/action execution or readable
    # frame authority is involved in testing mapping-capacity accounting.
    owner = object.__new__(SurgicalPrefixReplay)
    owner._frames = OrderedDict()
    owner._retained_frame_bytes = prefix._owned_bytes(owner._frames)
    owner._stream_nonce = 'private counter test nonce'
    owner._stream_metadata_bytes = prefix._owned_bytes(owner._stream_nonce)
    owner._max_frame_bytes = 4096
    shared = {'flapComponents': 'a' * 64}
    evicted = False
    for index in range(30):
        frame = {'state': bytes([index]) * 150, 'fieldTokens': shared,
                 'effectiveActivation': {'left': True, 'right': False}}
        owner._publish_frames(index, [frame])
        assert owner._retained_frame_bytes == prefix._owned_bytes(owner._frames)
        assert owner._frame_bytes() <= owner._max_frame_bytes
        evicted |= next(iter(owner._frames)) > 0
    assert evicted
    original = owner._frames
    before = owner._frame_bytes()
    with pytest.raises(PrefixBudgetExceeded, match='publication'):
        owner._publish_frames(30, [{'state': b'x' * owner._max_frame_bytes}])
    assert owner._frames is original and owner._frame_bytes() == before
    with pytest.raises(PrefixBudgetExceeded, match='during capture'):
        owner._frame_room(owner._max_frame_bytes)
    assert owner._frames == {}
    assert owner._retained_frame_bytes == prefix._owned_bytes(owner._frames)
    assert owner._frame_bytes() <= owner._max_frame_bytes


def prepared_episode(tmp_path, monkeypatch, *, projection=None):
    """One generated public script, then one fresh prefix preparation only."""
    engine, boundaries = episode()
    allowed = [True]
    calls = []
    step = prefix._PrefixEngine.step
    def guarded(self, action, **kwargs):
        assert allowed[0], 'Prepared-only playback attempted physics'
        calls.append(action)
        return step(self, action, **kwargs)
    monkeypatch.setattr(prefix._PrefixEngine, 'step', guarded)
    store = prefix._prepared_cache.PreparedPrefixStore(tmp_path / 'prepared')
    producer = SurgicalPrefixReplay(engine.scene, engine.snapshot(), max_new_actions=32,
        _prepared_store=store, _display_row_projection=projection)
    result = producer.seek(len(engine.actions))
    assert result['verification']['archiveEndpointVerified']
    assert producer.statistics['preparedCache']['publishedActions'] == len(engine.actions)
    assert len(calls) == len(engine.actions)
    allowed[0] = False
    return engine, boundaries, store, producer, calls


def test_prepared_only_reopens_exact_frames_receipts_and_activation_without_steps(tmp_path, monkeypatch):
    engine, boundaries, store, producer, calls = prepared_episode(tmp_path, monkeypatch)
    reopened_store = prefix._prepared_cache.PreparedPrefixStore(tmp_path / 'prepared')
    cached = SurgicalPrefixReplay(engine.scene, engine.snapshot(), max_new_actions=32,
        _prepared_store=reopened_store, _cache_only=True)
    final = cached.seek(len(engine.actions))
    assert final['view'] == boundaries[-1]
    assert final['verification']['archiveEndpointVerified']
    assert final['verification']['preparedCache'] == {
        'mode': 'only', 'loadedActions': 5, 'publishedActions': 0, 'misses': 0, 'lastSkip': None}
    assert len(calls) == 5
    for index in range(5):
        assert cached._receipt(index) == producer._receipt(index)
        for ordinal in range(len(producer._frames[index])):
            left = producer.seek_frame(index, ordinal)
            right = cached.seek_frame(index, ordinal)
            assert left['view'] == right['view']
            assert left['playback'] == right['playback']
            assert left['timeline'] == right['timeline']
            assert left['timelineToken'] != right['timelineToken']
    assert len(calls) == 5
    assert cached.statistics['historyBytes'] <= cached.statistics['maxHistoryBytes']


def test_prepared_only_miss_is_explicit_and_never_steps(tmp_path, monkeypatch):
    engine, _ = episode()
    def forbidden(*args, **kwargs):
        raise AssertionError('Cache miss must not execute physics')
    monkeypatch.setattr(prefix._PrefixEngine, 'step', forbidden)
    store = prefix._prepared_cache.PreparedPrefixStore(tmp_path / 'empty')
    replay = SurgicalPrefixReplay(engine.scene, engine.snapshot(), _prepared_store=store, _cache_only=True)
    result = replay.seek(1)
    assert result['position'] == 0 and result['budgetStopped']
    assert 'Prepared observations unavailable' in result['verification']['failure']['reason']
    assert result['verification']['preparedCache']['misses'] == 1
    with pytest.raises(ValueError, match='unavailable'):
        replay.seek_frame(0, 0)
    assert not replay.statistics['frameActions']
    assert replay.statistics['verifiedThrough'] == 0


def test_prepared_old_actions_page_within_original_quotas_and_no_repeated_steps(tmp_path, monkeypatch):
    engine, boundaries, store, producer, calls = prepared_episode(tmp_path, monkeypatch)
    frame_limit = max(prefix._owned_bytes(OrderedDict([(index, rows)]))
                      for index, rows in producer._frames.items()) + producer._stream_metadata_bytes + 1024
    checkpoint_limit = max(prefix._owned_bytes(OrderedDict([(index, row)]))
                           for index, row in producer._checkpoints.items()) + 64
    replay = SurgicalPrefixReplay(engine.scene, engine.snapshot(), max_new_actions=32,
        max_frame_bytes=frame_limit, max_checkpoint_bytes=checkpoint_limit,
        _prepared_store=store, _cache_only=True)
    assert replay.seek(5)['view'] == boundaries[-1]
    assert 0 not in replay.statistics['frameActions']
    assert 0 not in replay.statistics['checkpointPositions']
    count = replay.statistics['preparedCache']['loadedActions']
    assert replay.seek_frame(0, 0)['view'] == producer.seek_frame(0, 0)['view']
    assert replay.seek(0)['view'] == boundaries[0]
    assert replay.seek(5)['view'] == boundaries[-1]
    assert replay.statistics['preparedCache']['loadedActions'] == count
    assert len(calls) == 5
    for key in ('frame', 'checkpoint', 'history'):
        assert replay.statistics[key + 'Bytes'] <= replay.statistics['max' + key.title() + 'Bytes']


def test_prepared_receipts_regenerate_current_display_rows_and_compact_reads(tmp_path, monkeypatch):
    engine, _, store, producer, calls = prepared_episode(tmp_path, monkeypatch)
    def display(receipt):
        return {'status': receipt['status'], 'stateAfter': receipt['stateAfter']}
    cached = SurgicalPrefixReplay(engine.scene, engine.snapshot(), max_new_actions=32,
        _prepared_store=store, _cache_only=True, _display_row_projection=display)
    cached.seek(5)
    first = cached.seek_frame(0, 0)
    assert first['timeline'] == [{'status': row['status'], 'stateAfter': row['stateAfter']}
                                 for row in engine.history]
    assert cached._receipt(0) == producer._receipt(0)
    def no_old_history(*args, **kwargs):
        raise AssertionError('Compact warm read decoded old complete receipts')
    monkeypatch.setattr(cached, '_materialize_history', no_old_history)
    monkeypatch.setattr(cached, '_unpack_receipt', no_old_history)
    compact = cached.seek_frame(0, 1, prefix_token=first['timelineToken'])
    assert compact['timelineOmitted'] and 'timeline' not in compact
    assert cached._display_receipts and all(type(row[0]) is bytes for row in cached._display_receipts)
    assert len(calls) == 5


def test_prepared_source_log_changes_miss_and_live_store_code_drift_invalidates(tmp_path, monkeypatch):
    engine, _, store, producer, calls = prepared_episode(tmp_path, monkeypatch)
    changed = engine.snapshot()
    changed['actions'][0]['duration'] += .1
    different = SurgicalPrefixReplay(engine.scene, changed, _prepared_store=store, _cache_only=True)
    result = different.seek(1)
    assert result['position'] == 0 and result['verification']['preparedCache']['misses'] == 1
    cached = SurgicalPrefixReplay(engine.scene, engine.snapshot(), _prepared_store=store, _cache_only=True)
    original = prefix._prepared_cache.PreparedPrefixStore.load_action
    def changed_load(self, *args, **kwargs):
        return original(self, *args, **kwargs)
    monkeypatch.setattr(prefix._prepared_cache.PreparedPrefixStore, 'load_action', changed_load)
    with pytest.raises(ValueError, match='implementation changed'):
        cached.seek(1)
    assert len(calls) == 5


def test_failed_prepared_publication_keeps_committed_action_and_fresh_mode_never_loads(tmp_path, monkeypatch):
    engine, boundaries = episode()
    def forbidden(*args, **kwargs):
        raise AssertionError('Fresh preparation read a prepared observation')
    monkeypatch.setattr(prefix._prepared_cache.PreparedPrefixStore, 'load_action', forbidden)
    store = prefix._prepared_cache.PreparedPrefixStore(tmp_path / 'tiny', max_disk_bytes=32)
    replay = SurgicalPrefixReplay(engine.scene, engine.snapshot(), _prepared_store=store)
    result = replay.seek(1)
    assert result['position'] == 1 and result['view'] == boundaries[1]
    assert result['verification']['failure'] is None
    assert result['verification']['preparedCache']['mode'] == 'prepare'
    assert result['verification']['preparedCache']['publishedActions'] == 0
    assert result['verification']['preparedCache']['lastSkip']
    assert replay._receipt(0)['stateAfter'] == boundaries[1]['stateId']


@pytest.mark.parametrize('options', [{'_cache_only': True}, {'_cache_only': 1},
    {'_prepared_store': object()}, {'_cache_only': False, '_prepared_store': object()}])
def test_prepared_mode_requires_exact_private_store_and_boolean(options):
    engine, _ = episode()
    with pytest.raises(ValueError, match='exact store'):
        SurgicalPrefixReplay(engine.scene, engine.snapshot(), **options)


def test_optional_cache_allocation_failure_cannot_reject_a_committed_action(tmp_path, monkeypatch):
    engine, boundaries = episode()
    def out_of_memory(*args, **kwargs):
        raise MemoryError('Injected optional publication allocation failure')
    monkeypatch.setattr(prefix._prepared_cache.PreparedPrefixStore, 'publish_action', out_of_memory)
    store = prefix._prepared_cache.PreparedPrefixStore(tmp_path / 'allocation-failure')
    replay = SurgicalPrefixReplay(engine.scene, engine.snapshot(), _prepared_store=store)
    first = replay.seek(1)
    assert first['position'] == 1 and first['view'] == boundaries[1]
    assert first['verification']['failure'] is None
    assert 'allocation failure' in first['verification']['preparedCache']['lastSkip']
    assert replay.seek(1)['view'] == first['view']
    assert len(replay._history) == 1
