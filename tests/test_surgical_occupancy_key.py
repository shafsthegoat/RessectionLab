"""Exact occupancy-key invalidation and public replay against legacy keys."""
import copy
import numpy as np
import pytest
from resectionlab.core import semantic_digest
from resectionlab.surgical_engine import SurgicalEngine
from resectionlab.surgical_scene import SurgicalScene
from resectionlab.surgical_physics_v2 import default_profile


def legacy_key(state):
    # Frozen original cache identity; occupancy construction itself is outside
    # this optimization and remains the shared unchanged implementation.
    return semantic_digest({'removed': state['removed'], 'holds': state['holds'],
        **({'deformation': state['deformation']} if 'deformation' in state else {})})


class LegacyKeyEngine(SurgicalEngine):
    def occupancy(self):
        key = legacy_key(self.state)
        previous = getattr(self, '_legacy_key_cache', None)
        if previous is not None and previous[0] == key:
            return previous[1]
        self._occupancy_cache = None
        owner = super().occupancy()
        self._legacy_key_cache = (key, owner)
        return owner


def source(v2=False):
    labels = np.zeros((24, 24, 24), np.uint16)
    labels[8:13, 8:13, 8:15] = 2
    return SurgicalScene(labels, np.eye(4),
        [{'id': 2, 'name': 'target', 'material': 'tumor', 'target': True}],
        {'kind': 'explicit-key-engineering-fixture'}, {'tissuePhysicsV2': default_profile()} if v2 else {})


def assert_ownership(actual, reference):
    np.testing.assert_array_equal(np.asarray(actual.occupancy()), np.asarray(reference.occupancy()))
    for ids in ([0, 1, 2], [int(np.ravel_multi_index((9, 9, 9), actual.scene.shape))]):
        assert actual.occupancy().material_ids(ids) == reference.occupancy().material_ids(ids)


def test_integer_key_snapshots_append_reorder_and_in_place_replacement():
    scene = source()
    actual, reference = SurgicalEngine(scene), LegacyKeyEngine(scene)
    cells = [int(np.ravel_multi_index(ijk, scene.shape)) for ijk in [(8, 8, 8), (9, 9, 9), (10, 10, 10)]]
    previous, old_key = actual.occupancy(), legacy_key(actual.state)
    assert actual.occupancy() is previous
    for mutate in [lambda ids: ids.append(cells[0]), lambda ids: ids.append(cells[1]),
                   lambda ids: ids.reverse(), lambda ids: ids.__setitem__(0, cells[2])]:
        mutate(actual.state['removed']); mutate(reference.state['removed'])
        assert legacy_key(actual.state) != old_key
        assert actual.occupancy() is not previous
        assert_ownership(actual, reference)
        previous, old_key = actual.occupancy(), legacy_key(actual.state)
        assert actual.occupancy() is previous
    assert actual._occupancy_cache[0][0] == tuple(actual.state['removed'])


def test_large_native_removed_admission_keeps_complete_ordered_snapshot_and_invalidation(monkeypatch):
    from resectionlab import surgical_idle_native as adapter
    calls = []
    def predicate(value):
        calls.append(value)
        assert type(value) is list and all(type(item) is int for item in value)
        return True
    monkeypatch.setattr(adapter, 'try_exact_int_list', predicate)
    scene = source()
    actual, reference = SurgicalEngine(scene), LegacyKeyEngine(scene)
    cells = [int(np.ravel_multi_index(ijk, scene.shape))
             for ijk in [(8, 8, 8), (9, 9, 9), (10, 10, 10)]]
    actual.state['removed'] = [cells[0], cells[1]] * 16384
    reference.state['removed'] = actual.state['removed'].copy()
    previous = None
    for mutate in (lambda ids: None, lambda ids: ids.reverse(),
                   lambda ids: ids.__setitem__(513, cells[2]), lambda ids: ids.append(cells[0])):
        mutate(actual.state['removed']); mutate(reference.state['removed'])
        owner = actual.occupancy()
        assert owner is not previous
        snapshot = actual._occupancy_cache[0][0]
        assert type(snapshot) is tuple and snapshot == tuple(actual.state['removed'])
        assert actual.occupancy() is owner
        assert_ownership(actual, reference)
        previous = owner
    assert calls and any(value is actual.state['removed'] for value in calls)


def test_numpy_integer_and_nonlist_representations_use_the_frozen_legacy_key():
    engine = SurgicalEngine(source())
    cells = [int(np.ravel_multi_index(ijk, engine.scene.shape)) for ijk in [(8, 8, 8), (9, 9, 9)]]
    for removed in ([np.int64(cells[0])], [np.int32(cells[1])], tuple(cells)):
        engine.state['removed'] = removed
        owner = engine.occupancy()
        assert engine._occupancy_cache[0] == legacy_key(engine.state)
        assert isinstance(engine._occupancy_cache[0], str)
        assert engine.occupancy() is owner
        assert all(owner[cell] == 0 for cell in removed)
    engine.state['removed'] = [np.int64(cells[0])]
    previous = engine.occupancy()
    engine.state['removed'][0] = np.int64(cells[1])
    assert engine.occupancy() is not previous
    assert engine.occupancy()[cells[0]] != 0 and engine.occupancy()[cells[1]] == 0
    engine.state['removed'] = [float('nan')]
    with pytest.raises(ValueError, match='nonfinite'):
        engine.occupancy()


@pytest.mark.parametrize('value', [True, False, 1., np.int64(1)])
def test_strict_integer_admission_never_treats_equivalent_nonint_as_builtin_int(value):
    actual = SurgicalEngine(source())
    actual.state['removed'] = [0, value, 2]
    actual.occupancy()
    assert actual._occupancy_cache[0] == legacy_key(actual.state)
    assert type(actual._occupancy_cache[0]) is str
    actual.state['removed'][1] = 1
    actual.occupancy()
    assert actual._occupancy_cache[0][0] == (0, 1, 2)
    actual.state['removed'][1] = value
    actual.occupancy()
    assert type(actual._occupancy_cache[0]) is str


def test_integer_subclass_reaches_legacy_fallback_without_predicate_hash_or_equality(monkeypatch):
    from resectionlab import surgical_engine as module
    calls = []
    class Meta(type):
        def __eq__(cls, other):
            calls.append('equality')
            raise AssertionError('Class equality is not strict integer identity')
        def __hash__(cls):
            calls.append('hash')
            raise AssertionError('Class hashing is not strict integer identity')
    class Integer(int, metaclass=Meta): pass
    actual = SurgicalEngine(source())
    actual.state['removed'] = [0, Integer(1), 2]
    original_digest = module.semantic_digest
    def digest(value):
        if isinstance(value, dict) and 'removed' in value:
            raise RuntimeError('legacy fallback reached')
        return original_digest(value)
    # The unchanged semantic-digest fallback may itself inspect custom classes;
    # stop at that boundary to isolate the strict predicate's no-hook behavior.
    monkeypatch.setattr(module, 'semantic_digest', digest)
    with pytest.raises(RuntimeError, match='legacy fallback reached'):
        actual.occupancy()
    assert calls == []


@pytest.mark.parametrize('values,fail_at,expected_events,expected_error', [
    ([False, 1], 1, ['iter', 0], 'fallback'),
    ([0, False, 2], 2, ['iter', 0, 1], 'fallback'),
    ([0, 1, 2], 1, ['iter', 0, 1], 'custom iteration failure'),
])
def test_removed_list_subclass_retains_generator_short_circuit_and_error_order(
        monkeypatch, values, fail_at, expected_events, expected_error):
    from resectionlab import surgical_engine as module
    events = []
    class Removed(list):
        def __iter__(self):
            events.append('iter')
            for index in range(len(self)):
                events.append(index)
                if index == fail_at:
                    raise RuntimeError('custom iteration failure')
                yield self[index]
    actual = SurgicalEngine(source())
    actual.state['removed'] = Removed(values)
    original_digest = module.semantic_digest
    def digest(value):
        if isinstance(value, dict) and 'removed' in value:
            raise RuntimeError('fallback')
        return original_digest(value)
    monkeypatch.setattr(module, 'semantic_digest', digest)
    with pytest.raises(RuntimeError, match=expected_error):
        actual.occupancy()
    assert events == expected_events


def test_public_standard_library_replacements_cannot_change_bound_integer_predicate(monkeypatch):
    import itertools
    import operator
    from resectionlab import surgical_engine as module
    actual = SurgicalEngine(source())
    first = actual.occupancy()
    original_is, original_repeat = operator.is_, itertools.repeat
    def forbidden(*args, **kwargs):
        raise AssertionError('Public module attribute must not replace bound C predicate')
    with monkeypatch.context() as patch:
        patch.setattr(operator, 'is_', forbidden)
        patch.setattr(itertools, 'repeat', forbidden)
        actual.state['removed'].append(1)
        changed = actual.occupancy()
        saved = module._microstep_checkpoint({'removed': [1, 2], 'debris': {}})
    assert module._strict_int_is is original_is and module._strict_int_repeat is original_repeat
    assert changed is not first and actual._occupancy_cache[0][0] == (1,)
    assert saved == {'removed': [1, 2], 'debris': {}}


def test_hold_and_deformation_mutations_retain_canonical_invalidation():
    labels = np.zeros((10, 10, 10), np.uint16)
    labels[3, 3, 3] = labels[6, 3, 3] = 2
    scene = SurgicalScene(labels, np.eye(4), [{'id': 2, 'name': 'target', 'material': 'tumor', 'target': True}], {}, {})
    actual, reference = SurgicalEngine(scene), LegacyKeyEngine(scene)
    first, second = [int(np.ravel_multi_index(ijk, scene.shape)) for ijk in [(3, 3, 3), (6, 3, 3)]]
    def compare_changed():
        before = getattr(compare_changed, 'owner', None)
        assert_ownership(actual, reference)
        owner = actual.occupancy()
        assert owner is not before
        assert actual.occupancy() is owner
        compare_changed.owner = owner
    compare_changed()
    hold = {'cells': [first], 'origin': [3., 3., 3.], 'offset': [0, 0, 0]}
    actual.state['holds']['left'] = copy.deepcopy(hold)
    reference.state['holds']['left'] = copy.deepcopy(hold)
    compare_changed()
    for engine in (actual, reference): engine.state['holds']['left']['offset'][0] = 1
    compare_changed()
    for engine in (actual, reference): engine.state['holds']['left']['cells'][0] = second
    compare_changed()
    for engine in (actual, reference): engine.state['holds'].clear()
    compare_changed()
    for engine in (actual, reference): engine.state['deformation'] = {}
    compare_changed()
    for engine in (actual, reference): engine.state['deformation'][str(first)] = [1., 0., 0.]
    compare_changed()
    for engine in (actual, reference): engine.state['deformation'][str(first)][0] = 2.
    compare_changed()


@pytest.mark.parametrize('v2', [False, True])
def test_public_action_records_and_states_match_frozen_legacy_key_exactly(v2):
    scene = source(v2)
    actual, reference = SurgicalEngine(scene), LegacyKeyEngine(scene)
    pose = lambda z: {'position': [10., 10., z], 'quaternion': [0., 0., 0., 1.]}
    actions = [
        {'operation': 'exchange', 'instrument': 'cusa'},
        {'operation': 'place', 'pose': pose(4.), 'parameters': {'radiusMm': .6, **({'forceN': 0.} if v2 else {})}},
        {'operation': 'apply', 'pose': pose(14.), 'duration': 1.},
        {'operation': 'apply', 'pose': pose(12.), 'parameters': {'radiusMm': 1.5}, 'duration': .4},
        {'operation': 'inspect', 'parameters': {'radiusMm': .5}},
        {'operation': 'withdraw', 'pose': pose(4.)},
        {'operation': 'exchange', 'instrument': 'brain_retractor'},
        {'operation': 'place', 'pose': pose(4.)},
        {'operation': 'hold', 'pose': pose(7.)},
        {'operation': 'move', 'pose': {'position': [10.2, 10., 7.], 'quaternion': [0., 0., 0., 1.]}},
        {'operation': 'release'},
    ]
    for row in actions:
        action = {'slot': 'left', 'duration': .2, **row}
        assert actual.preview(action, include_view=False) == reference.preview(action, include_view=False)
        assert actual.step(action) == reference.step(action)
        assert actual.snapshot() == reference.snapshot()
        assert actual.history == reference.history
        assert_ownership(actual, reference)
    assert actual.metrics()['targetRemovedMm3'] > 0
