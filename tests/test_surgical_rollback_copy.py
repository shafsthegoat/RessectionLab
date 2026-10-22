"""Rollback data-graph ownership and public transactions versus 3e0bd07c.

The supported graph is JSON data plus ordinary deepcopy of unusual leaves.
Custom __deepcopy__ methods that inspect memo insertion timing are explicitly
outside that contract; flat guards do not prove arbitrary Python-object parity.
"""
import copy
import json
import math
import struct

import numpy as np
import pytest

import resectionlab.surgical_engine as module
from resectionlab.core import array_digest
from resectionlab.surgical_engine import SurgicalEngine, _microstep_checkpoint
from resectionlab.surgical_physics_v2 import default_profile
from resectionlab.surgical_rl_protection import GuardConfig, ProtectionAnnotation, ProtectionSet
from resectionlab.surgical_scene import SurgicalScene
from resectionlab.surgical_telemetry import capture_episode


def legacy_checkpoint(state):
    # Literally frozen 3e0bd07c _execute checkpoint semantics, not an alias to
    # the current helper. The existing component exemption is intentional.
    components = state.get('flapComponents')
    return copy.deepcopy(state, {id(components): components} if components is not None else None)


def encoded(value):
    return json.dumps(value, separators=(',', ':'), allow_nan=False)


def test_flat_fields_own_their_containers_preserve_aliases_order_and_float_bits():
    removed = [4, 1, 4]
    debris = {'9': -0., '2': math.nextafter(0., 1.), '1': 3}
    components = [{'cells': [8, 9], 'bounds': [[1, 2, 3], [4, 5, 6]]}]
    state = {'aliases': [removed, debris], 'removed': removed, 'debris': debris,
             'flapComponents': components, 'nested': {'mutable': [[1, 2]]}}
    saved = _microstep_checkpoint(state)
    assert encoded(saved) == encoded(legacy_checkpoint(state))
    assert saved['removed'] is not removed and saved['debris'] is not debris
    assert saved['aliases'][0] is saved['removed'] and saved['aliases'][1] is saved['debris']
    assert saved['flapComponents'] is components
    assert list(saved['debris']) == list(debris)
    assert [float(v).hex() for v in saved['debris'].values()] == [float(v).hex() for v in debris.values()]
    removed.append(20); debris['9'] = 1.; state['nested']['mutable'][0].append(3)
    assert saved['removed'] == [4, 1, 4] and saved['debris']['9'].hex() == '-0x0.0p+0'
    assert saved['nested']['mutable'] == [[1, 2]]
    saved['removed'].clear(); saved['debris'].pop('2'); saved['nested']['mutable'][0].clear()
    assert removed == [4, 1, 4, 20] and '2' in debris and state['nested']['mutable'] == [[1, 2, 3]]


def test_existing_component_memo_precedence_and_cross_field_aliases():
    shared = [1, 2]
    state = {'removed': shared, 'debris': {}, 'flapComponents': shared}
    saved = _microstep_checkpoint(state)
    # Preserve the preexisting component exemption even for this unusual alias.
    assert saved['removed'] is saved['flapComponents'] is shared
    assert legacy_checkpoint(state)['removed'] is shared
    state = {'removed': shared, 'debris': shared}
    saved = _microstep_checkpoint(state)
    assert saved['removed'] is saved['debris'] and saved['removed'] is not shared


def test_large_removed_native_admission_preserves_legacy_copy_aliases_and_rechecks_mutation(monkeypatch):
    from resectionlab import surgical_idle_native as adapter
    from itertools import repeat
    from operator import is_
    calls = []
    def predicate(value):
        calls.append(value)
        return True if all(map(is_, map(type, value), repeat(int))) else None
    monkeypatch.setattr(adapter, 'try_exact_int_list', predicate)
    removed = [0, -2**300, 0, 2**300] * 8192
    state = {'removed': removed, 'aliases': [removed], 'debris': {}}
    for item in (1, True, IntWithData(1), -2**300):
        removed[16385] = item
        actual, reference = _microstep_checkpoint(state), legacy_checkpoint(state)
        assert encoded(actual) == encoded(reference)
        assert type(actual['removed'][16385]) is type(reference['removed'][16385])
        assert actual['removed'] is actual['aliases'][0] and actual['removed'] is not removed
        if isinstance(item, IntWithData):
            assert actual['removed'][16385].data == [1]
            assert actual['removed'][16385].data is not item.data
        actual['removed'].append(19)
        assert len(removed) == 32768
    assert len(calls) == 4 and all(value is removed for value in calls)


class IntWithData(int):
    def __new__(cls, value):
        instance = super().__new__(cls, value); instance.data = [1]
        return instance


class FloatWithData(float):
    def __new__(cls, value):
        instance = super().__new__(cls, value); instance.data = [1]
        return instance


class ListWithData(list):
    pass


class DictWithData(dict):
    pass


@pytest.mark.parametrize('removed,debris', [
    ([IntWithData(2)], {'1': FloatWithData(.5)}),
    ([np.int64(2)], {'1': np.float64(.5)}),
    ([2], {'1': [1, 2]}),
    (ListWithData([2]), DictWithData({'1': .5})),
    (np.array([2]), {'1': np.array([.5])}),
])
def test_unusual_or_nested_values_use_ordinary_deepcopy(removed, debris):
    state = {'removed': removed, 'debris': debris, 'nested': {'mutable': [1]}}
    saved, reference = _microstep_checkpoint(state), legacy_checkpoint(state)
    assert type(saved['removed']) is type(reference['removed'])
    assert type(saved['debris']) is type(reference['debris'])
    assert saved['removed'] is not removed and saved['debris'] is not debris
    assert type(saved['removed'][0]) is type(reference['removed'][0])
    assert type(saved['debris']['1']) is type(reference['debris']['1'])
    if hasattr(removed[0], 'data') and isinstance(removed[0], IntWithData):
        assert saved['removed'][0].data == [1] and saved['removed'][0].data is not removed[0].data
        assert saved['debris']['1'].data is not debris['1'].data
    if isinstance(debris['1'], (list, np.ndarray)):
        assert saved['debris']['1'] is not debris['1']
        np.testing.assert_equal(saved['debris']['1'], reference['debris']['1'])
    saved['nested']['mutable'].append(2)
    assert state['nested']['mutable'] == [1]


def test_fresh_guards_detect_public_in_place_changes():
    state = {'removed': [1], 'debris': {'1': 1.}}
    first = _microstep_checkpoint(state)
    state['removed'].append(IntWithData(2)); state['debris']['1'] = [1.]
    second = _microstep_checkpoint(state)
    assert second['removed'][1].data is not state['removed'][1].data
    assert second['debris']['1'] is not state['debris']['1']
    assert first == {'removed': [1], 'debris': {'1': 1.}}


@pytest.mark.parametrize('value,expected_flat', [(1, True), (True, False), (1., False),
                                              (IntWithData(1), False), (np.int64(1), False)])
def test_removed_admission_is_strict_and_rechecks_mutation_before_copy(monkeypatch, value, expected_flat):
    removed = [0, 1, 2]
    state = {'removed': removed}
    observed = []
    def inspect_copy(actual, memo):
        assert actual is state
        cached = memo.get(id(removed))
        observed.append(cached)
        return cached
    monkeypatch.setattr(module.copy, 'deepcopy', inspect_copy)
    assert _microstep_checkpoint(state) == [0, 1, 2]
    removed[1] = value
    result = _microstep_checkpoint(state)
    assert (result is not None) is expected_flat
    if expected_flat:
        assert result == removed and result is not removed
    removed[1] = 3
    assert _microstep_checkpoint(state) == [0, 3, 2]
    assert len(observed) == 3


def test_removed_int_subclass_admission_does_not_hash_or_compare_its_metaclass(monkeypatch):
    class Meta(type):
        def __eq__(cls, other): raise AssertionError('Unexpected metaclass equality')
        def __hash__(cls): raise AssertionError('Unexpected metaclass hashing')
    class Integer(int, metaclass=Meta): pass
    removed = [0, Integer(1), 2]
    def inspect_copy(state, memo):
        assert id(removed) not in memo
        return 'ordinary-deepcopy-fallback'
    monkeypatch.setattr(module.copy, 'deepcopy', inspect_copy)
    assert _microstep_checkpoint({'removed': removed}) == 'ordinary-deepcopy-fallback'


def test_removed_list_subclass_keeps_original_deepcopy_iteration_and_errors():
    def run(function, fail_at):
        events = []
        class Removed(list):
            def __iter__(self):
                events.append('iter')
                for index in range(len(self)):
                    events.append(index)
                    if index == fail_at:
                        raise RuntimeError('custom deepcopy iteration failure')
                    yield self[index]
        removed = Removed([0, 1, 2])
        try:
            result = function({'removed': removed})
            return result['removed'], events, None
        except RuntimeError as exc:
            return None, events, str(exc)
    for fail_at in (None, 1):
        actual, reference = run(_microstep_checkpoint, fail_at), run(legacy_checkpoint, fail_at)
        assert actual == reference


def graph_signature(value):
    """Compare exact types, order, float bits and mutable alias topology."""
    seen = {}; result = []
    def visit(item, path):
        result.append((path, type(item)))
        if isinstance(item, (dict, list)):
            if id(item) in seen:
                result.append((path, 'alias', seen[id(item)])); return
            seen[id(item)] = path
            if isinstance(item, dict):
                for key, child in item.items():
                    result.append((path, 'key', type(key), key))
                    visit(child, path + (key,))
            else:
                for index, child in enumerate(item): visit(child, path + (index,))
        elif type(item) is float: result.append((path, struct.pack('!d', item)))
        else: result.append((path, repr(item)))
    visit(value, ())
    return result


def test_drilling_maps_copy_order_bits_aliases_and_ownership():
    work = {'9': -0., '2': math.nextafter(0., 1.), '1': 3}
    temperature = {'2': math.nextafter(37., 38.), '1': 38.}
    state = {'aliases': [work, temperature], 'removed': [2], 'debris': work,
             'flapComponents': [], 'drillingV2': {'workJ': work, 'temperatureC': temperature,
                 'thermalDoseCSeconds': temperature, 'other': {'mutable': [1]}}}
    before = encoded(state)
    saved = _microstep_checkpoint(state)
    assert encoded(saved) == before == encoded(legacy_checkpoint(state))
    assert graph_signature(saved) == graph_signature(legacy_checkpoint(state))
    drilling = saved['drillingV2']
    assert drilling is not state['drillingV2']
    assert drilling['workJ'] is saved['debris'] is saved['aliases'][0]
    assert drilling['temperatureC'] is drilling['thermalDoseCSeconds'] is saved['aliases'][1]
    assert drilling['workJ'] is not work and drilling['temperatureC'] is not temperature
    saved['drillingV2']['workJ']['9'] = 2.
    saved['drillingV2']['temperatureC'].clear()
    saved['drillingV2']['other']['mutable'].append(2)
    assert encoded(state) == before
    work['new'] = 4.; temperature['new'] = 40.
    assert 'new' not in drilling['workJ'] and 'new' not in drilling['temperatureC']


def test_drilling_map_component_memo_precedence_and_removed_alias():
    shared = {'1': -0.}; removed = [1, 2]
    state = {'removed': removed, 'debris': {}, 'flapComponents': shared,
             'drillingV2': {'workJ': removed, 'temperatureC': shared, 'thermalDoseCSeconds': shared}}
    saved = _microstep_checkpoint(state)
    assert graph_signature(saved) == graph_signature(legacy_checkpoint(state))
    assert saved['drillingV2']['temperatureC'] is saved['drillingV2']['thermalDoseCSeconds'] is shared
    assert saved['drillingV2']['workJ'] is saved['removed'] and saved['removed'] is not removed
    # A publicly replaced removed field may itself alias a recognized map.
    # Its list fast path declines, while the shared dict still copies once.
    state = {'removed': shared, 'debris': {}, 'flapComponents': [],
             'drillingV2': {'workJ': shared, 'temperatureC': {}, 'thermalDoseCSeconds': {}}}
    saved = _microstep_checkpoint(state)
    assert graph_signature(saved) == graph_signature(legacy_checkpoint(state))
    assert saved['removed'] is saved['drillingV2']['workJ'] and saved['removed'] is not shared


@pytest.mark.parametrize('value', [
    {'1': [1.]}, {'1': np.float64(.5)}, {'1': FloatWithData(.5)},
    {'1': IntWithData(2)}, DictWithData({'1': .5}), {'1': True}, {1: .5},
])
def test_drilling_map_unusual_values_fall_back_to_deepcopy(value):
    if type(value) is DictWithData: value.data = [2]
    state = {'removed': [], 'debris': {}, 'drillingV2': {'workJ': {},
             'temperatureC': value, 'thermalDoseCSeconds': {}}}
    saved, reference = _microstep_checkpoint(state), legacy_checkpoint(state)
    assert graph_signature(saved) == graph_signature(reference)
    actual = saved['drillingV2']['temperatureC']
    assert actual is not value
    original_leaf = next(iter(value.values())); actual_leaf = next(iter(actual.values()))
    if isinstance(original_leaf, list): assert actual_leaf is not original_leaf
    if hasattr(original_leaf, 'data'): assert actual_leaf.data is not original_leaf.data
    if hasattr(value, 'data'): assert actual.data == value.data and actual.data is not value.data


def test_outer_drilling_subclass_and_same_size_public_mutation_use_fresh_guards():
    outer = DictWithData({'workJ': {'1': 1.}, 'temperatureC': {}, 'thermalDoseCSeconds': {}})
    outer.data = [1]
    state = {'removed': [], 'debris': {}, 'drillingV2': outer}
    saved = _microstep_checkpoint(state)
    assert type(saved['drillingV2']) is DictWithData
    assert saved['drillingV2'].data is not outer.data
    state['drillingV2'] = dict(outer)
    first = _microstep_checkpoint(state)
    state['drillingV2']['workJ']['1'] = FloatWithData(.5)
    second = _microstep_checkpoint(state)
    assert graph_signature(second) == graph_signature(legacy_checkpoint(state))
    assert second['drillingV2']['workJ']['1'].data is not state['drillingV2']['workJ']['1'].data
    state['drillingV2']['workJ']['1'] = [1.]
    third = _microstep_checkpoint(state)
    assert third['drillingV2']['workJ']['1'] is not state['drillingV2']['workJ']['1']
    assert first['drillingV2']['workJ'] == {'1': 1.}


def source(v2=False):
    labels = np.zeros((24, 24, 24), np.uint16); labels[8, 8, 8] = 1
    return SurgicalScene(labels, np.eye(4), [{'id': 1, 'name': 'bone', 'material': 'bone'}],
        {'kind': 'rollback-copy-engineering-fixture'}, {'tissuePhysicsV2': default_profile()} if v2 else {})


def public_sequence():
    pose = lambda z: {'position': [8., 8., z], 'quaternion': [0., 0., 0., 1.]}
    return [{'slot': 'left', 'duration': .1, **action} for action in (
        {'operation': 'place', 'pose': pose(4.), 'parameters': {'radiusMm': .6}},
        {'operation': 'move', 'pose': pose(9.), 'duration': 1.},
        {'operation': 'move', 'pose': pose(9.), 'duration': .01},
        {'operation': 'inspect', 'duration': 0.},
        {'operation': 'apply', 'pose': pose(9.), 'duration': 2.},
        {'operation': 'apply', 'pose': pose(9.), 'duration': 2.},
        {'operation': 'inspect', 'duration': .2},
    )]


def test_actual_partial_drilling_work_checkpoint_is_independent_and_matches_legacy(monkeypatch):
    current = SurgicalEngine(source(True)); reference = SurgicalEngine(current.scene)
    actions = [
        {'slot': 'left', 'operation': 'place', 'duration': .2,
         'pose': {'position': [8., 8., 4.], 'quaternion': [0., 0., 0., 1.]},
         'parameters': {'radiusMm': 1.5}},
        {'slot': 'left', 'operation': 'apply', 'duration': .3,
         'pose': {'position': [8., 8., 10.], 'quaternion': [0., 0., 0., 1.]},
         'parameters': {'rpm': 1000., 'forceN': .1, 'duty': 1.}},
    ]
    for action in actions:
        preview = current.preview(action, include_view=False)
        actual = current.step(action)
        with monkeypatch.context() as patch:
            patch.setattr(module, '_microstep_checkpoint', legacy_checkpoint)
            expected_preview = reference.preview(action, include_view=False)
            expected = reference.step(action)
        assert encoded(preview) == encoded(expected_preview)
        assert encoded(actual) == encoded(expected)
        assert encoded(current.snapshot()) == encoded(reference.snapshot())
    assert actual['status'] == 'partial' and actual['reason'] == 'BONE_CUTTING_IN_PROGRESS'
    assert current.state['drillingV2']['workJ'] and not current.state['removed']
    before = encoded(current.snapshot())
    saved = _microstep_checkpoint(current.state)
    assert graph_signature(saved) == graph_signature(legacy_checkpoint(current.state))
    saved['drillingV2']['workJ'].clear()
    saved['drillingV2']['temperatureC'].clear()
    saved['drillingV2']['thermalDoseCSeconds'].clear()
    assert encoded(current.snapshot()) == before


@pytest.mark.parametrize('v2,guarded', [(False, False), (True, False), (True, True)])
def test_public_preview_partial_rejection_and_continuation_match_legacy(monkeypatch, v2, guarded):
    scene = source(v2); kwargs = {}
    if guarded:
        mask = np.zeros(scene.shape, bool); mask[22, 22, 22] = True
        annotation = ProtectionAnnotation.from_mask(scene, 'brainstem', mask, provenance='explicit engineering fixture')
        kwargs = {'protection': ProtectionSet(scene, [annotation], clearance_mm=0.), 'guard_config': GuardConfig()}
    current = SurgicalEngine(scene, **kwargs); reference = SurgicalEngine(scene, **kwargs)
    labels = array_digest(scene.labels); statuses = []
    actions = public_sequence()
    if guarded:
        actions.append({'slot': 'left', 'operation': 'move', 'duration': .4,
            'pose': {'position': [22., 22., 22.], 'quaternion': [0., 0., 0., 1.]}})
    for action in actions:
        before = encoded(current.snapshot())
        preview = current.preview(action, include_view=False)
        assert encoded(current.snapshot()) == before
        actual = current.step(action)
        with monkeypatch.context() as patch:
            patch.setattr(module, '_microstep_checkpoint', legacy_checkpoint)
            prior = reference.preview(action, include_view=False)
            expected = reference.step(action)
        assert encoded(preview) == encoded(prior)
        assert encoded(actual) == encoded(expected)
        assert encoded(current.snapshot()) == encoded(reference.snapshot())
        assert actual['stateId'] == preview['stateId']
        statuses.append(actual['status'])
    assert {'accepted', 'partial', 'rejected'} <= set(statuses)
    if guarded: assert actual['reasonCode'] == 'PROTECTED_STRUCTURE'
    assert array_digest(scene.labels) == labels


class LateDomainFailure(SurgicalEngine):
    fail_at = None
    fail_removal = False

    def _physical_substep(self, *args):
        before = len(self.state['removed'])
        events = super()._physical_substep(*args)
        if (self.fail_at is not None and self.state['elapsedSeconds'] >= self.fail_at
                or self.fail_removal and len(self.state['removed']) > before):
            self.failed_evidence.append({'removed': self.state['removed'][:],
                'debris': self.state['debris'].copy(), 'debrisRemovedMm3': self.state.get('debrisRemovedMm3', 0.),
                'drillingV2': copy.deepcopy(self.state['drillingV2'])})
            raise ValueError('TEST_LATE_DOMAIN_FAILURE')
        return events


def active_field_fixture():
    labels = np.zeros((32, 24, 24), np.uint16)
    labels[17:20, 7:10, 8] = 1; labels[8, 8, 8] = 1
    scene = SurgicalScene(labels, np.eye(4), [{'id': 1, 'name': 'bone', 'material': 'bone'}],
        {'kind': 'rollback-shared-clock-engineering-fixture'}, {'tissuePhysicsV2': default_profile()})
    engine = LateDomainFailure(scene); engine.failed_evidence = []
    pose = lambda x, z: {'position': [x, 8., z], 'quaternion': [0., 0., 0., 1.]}
    def step(operation, slot='left', **values):
        result = engine.step({'slot': slot, 'operation': operation, 'duration': .1, **values})
        assert result['status'] in ('accepted', 'partial'), result
        return result
    step('place', pose=pose(18., 4.), parameters={'radiusMm': 1.5, 'rpm': 80000., 'forceN': 20.})
    for _ in range(3):
        result = step('apply', pose=pose(18., 10.), duration=3.)
        if result['status'] == 'accepted': break
    assert len(engine.state['removed']) == 9 and engine.state['debrisMm3'] == 9.
    step('withdraw')
    step('place', pose=pose(8., 4.), parameters={'radiusMm': .6})
    step('exchange', slot='right', instrument='frazier_suction')
    step('place', slot='right', pose=pose(18., 4.), parameters={'radiusMm': .5, 'forceN': 0., 'flowMm3PerSecond': 1.})
    step('move', slot='right', pose=pose(18., 8.), duration=1.)
    step('activate', slot='right', parameters={'radiusMm': 2.}, duration=.01)
    assert engine.state['slots']['right']['active'] and engine.state['debrisMm3'] > 8.
    return engine


@pytest.mark.parametrize('failure', ['first', 'later', 'removal'])
def test_failed_microsteps_restore_real_shared_clock_fields_and_removal(monkeypatch, failure):
    current = active_field_fixture(); reference = current.clone()
    before = current.snapshot(); before_removed = current.state['debrisRemovedMm3']
    for engine in (current, reference):
        engine.failed_evidence = []
        if failure == 'removal': engine.fail_removal = True
        else: engine.fail_at = engine.state['elapsedSeconds'] + (.05 if failure == 'first' else .15)
    action = {'slot': 'left', 'operation': 'inspect', 'duration': .3}
    if failure == 'removal':
        action.update(operation='apply', duration=2., pose={'position': [8., 8., 9.], 'quaternion': [0., 0., 0., 1.]})
    actual = current.step(action)
    with monkeypatch.context() as patch:
        patch.setattr(module, '_microstep_checkpoint', legacy_checkpoint)
        expected = reference.step(action)
    assert encoded(actual) == encoded(expected)
    assert encoded(current.snapshot()) == encoded(reference.snapshot())
    assert current.failed_evidence and current.failed_evidence == reference.failed_evidence
    failed_drilling = current.failed_evidence[-1]['drillingV2']
    assert failed_drilling['coolingLossJ'] > current.state['drillingV2']['coolingLossJ']
    assert failed_drilling['temperatureC'] != current.state['drillingV2']['temperatureC']
    if failure == 'first':
        assert actual['status'] == 'rejected' and current.snapshot() == before
    else:
        assert actual['status'] == 'partial'
        assert current.state['debrisRemovedMm3'] > before_removed
        assert current.failed_evidence[-1]['debrisRemovedMm3'] > current.state['debrisRemovedMm3']
    if failure == 'removal':
        assert len(current.failed_evidence[-1]['removed']) > len(current.state['removed']) == len(before['checkpoint']['state']['removed'])
    # Actual ordinary continuation after the rejected fraction remains editable.
    current.fail_at = reference.fail_at = None; current.fail_removal = reference.fail_removal = False
    actual = current.step(action)
    with monkeypatch.context() as patch:
        patch.setattr(module, '_microstep_checkpoint', legacy_checkpoint)
        expected = reference.step(action)
    assert actual == expected and current.snapshot() == reference.snapshot()
    state = current.state
    assert state['drillingV2']['removedVolumeMm3'] == pytest.approx(state['debrisMm3'] + state['debrisRemovedMm3'])


@pytest.mark.parametrize('method', ['step', 'preview'])
def test_cancellation_after_real_active_field_ticks_preserves_original(monkeypatch, method):
    current = active_field_fixture(); reference = current.clone()
    before = encoded(current.snapshot())
    for engine, legacy in ((current, False), (reference, True)):
        calls = 0
        def check():
            nonlocal calls
            calls += 1
            if calls == 4: raise ValueError('engineering cancellation')
        with monkeypatch.context() as patch:
            if legacy: patch.setattr(module, '_microstep_checkpoint', legacy_checkpoint)
            with pytest.raises(ValueError, match='engineering cancellation'):
                getattr(engine, method)({'slot': 'left', 'operation': 'inspect', 'duration': .6}, check=check)
        assert calls == 4 and encoded(engine.snapshot()) == before


def test_observed_engine_transaction_samples_match_legacy(monkeypatch):
    scene = source(True); actions = public_sequence()
    actual = capture_episode(scene, actions)
    with monkeypatch.context() as patch:
        patch.setattr(module, '_microstep_checkpoint', legacy_checkpoint)
        expected = capture_episode(scene, actions)
    assert encoded(actual) == encoded(expected)
    assert {'accepted', 'partial', 'rejected'} <= {row['status'] for row in actual['attempts']}
