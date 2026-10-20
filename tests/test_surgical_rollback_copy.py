"""Rollback data-graph ownership and public transactions versus 3e0bd07c.

The supported graph is JSON data plus ordinary deepcopy of unusual leaves.
Custom __deepcopy__ methods that inspect memo insertion timing are explicitly
outside that contract; flat guards do not prove arbitrary Python-object parity.
"""
import copy
import json
import math

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
                'debris': self.state['debris'].copy(), 'debrisRemovedMm3': self.state.get('debrisRemovedMm3', 0.)})
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
