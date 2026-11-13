"""Sparse ownership lookup matches independent exact set membership."""
import numpy as np
import pytest

from resectionlab.surgical_occupancy import MaterialOccupancy
from resectionlab.surgical_engine import SurgicalEngine
from resectionlab.surgical_scene import SurgicalScene


def test_sorted_cleared_lookup_handles_arbitrary_shape_duplicates_and_replaced_sets(monkeypatch):
    labels = np.ones(100, np.uint16)
    labels[13] = 0
    owner = MaterialOccupancy(labels, [3, 7, 11, 83], {7: 42})
    ids = np.array([[99, 3, 7, 3, 83], [0, 11, 13, 7, 11]])
    def expected():
        return np.array([[owner.moved.get(int(i), 0 if i in owner.cleared or not labels[i] else i + 1)
                          for i in row] for row in ids])
    np.testing.assert_array_equal(owner[ids], expected())
    # Native lookup needs no sorted ledger; the reference ledger stays immutable.
    if owner._sorted_integer_cleared is not None:
        with pytest.raises(ValueError): owner._sorted_integer_cleared.setflags(write=True)
    saved = owner._sorted_integer_cleared
    np.testing.assert_array_equal(owner[ids], expected())
    assert owner._sorted_integer_cleared is saved
    owner.cleared = frozenset([0, 99])
    np.testing.assert_array_equal(owner[ids], expected())
    # A lower storage cap changes only lookup strategy.
    monkeypatch.setattr('resectionlab.surgical_occupancy._MAX_SORTED_CLEARED_BYTES', 0)
    fallback = MaterialOccupancy(labels, [3, 7, 11, 83], {7: 42})
    np.testing.assert_array_equal(fallback[ids], MaterialOccupancy(labels, [3, 7, 11, 83], {7: 42})[ids])
    assert fallback._sorted_integer_cleared is None


def test_sorted_lookup_preserves_shared_material_memberships_and_zero_length_queries():
    labels = np.ones(30, np.uint16)
    owner = MaterialOccupancy(labels, [3, 7], {7: 12}, {7: [11, 19]})
    query = [9, 3, 7, 9, 7, 2]
    assert owner.material_ids(query) == [2, 9, 11, 19]
    assert owner.material_ids([]) == []
    np.testing.assert_array_equal(owner[np.array([], np.int64)], [])


def test_nonbuiltin_cleared_values_keep_legacy_lookup_path():
    labels = np.ones(10, np.uint16)
    owner = MaterialOccupancy(labels, [np.int64(3)], {})
    np.testing.assert_array_equal(owner[[1, 3, 5, 7]], [2, 0, 6, 8])
    assert owner._sorted_integer_cleared is None


def test_engine_nonbuiltin_removed_equality_keeps_numpy_update_operands():
    shape = (5, 5, 4)
    static, held, target = [int(np.ravel_multi_index((x, 2, 2), shape)) for x in range(3)]
    events = []
    class Removed(int):
        def __hash__(self): return hash(held)
        def __eq__(self, other):
            events.append(type(other))
            return isinstance(other, np.integer)
    labels = np.zeros(shape, np.uint16)
    labels.ravel()[[static, held]] = 1
    engine = SurgicalEngine(SurgicalScene(labels, np.eye(4),
        [{'id': 1, 'name': 'Generated fragment', 'material': 'bone'}],
        {'kind': 'generated occupancy callback control'}, {}))
    engine.state['removed'] = [Removed(static)]
    engine.state['holds']['left'] = {'cells': [held], 'offset': [1, 0, 0]}
    removed = set(engine.state['removed'])
    cleared = set(removed)
    retained = np.array([cell for cell in [held] if cell not in removed], dtype=np.int64)
    cleared.update(retained)
    expected = MaterialOccupancy(labels, cleared, {target: held + 1})
    expected_events = events[:]
    events.clear()
    actual = engine.occupancy()
    assert events == expected_events == [int, np.int64]
    assert [type(cell) for cell in actual.cleared] == [Removed]
    assert actual._integer_cleared is None
    query = np.array([held, target], dtype=np.int64)
    np.testing.assert_array_equal(actual[query], expected[query])
    np.testing.assert_array_equal(actual[query], [held + 1, held + 1])
    np.testing.assert_array_equal(np.asarray(actual), np.asarray(expected))
    assert actual[held] == expected[held] == held + 1
    assert actual.material_ids(query) == expected.material_ids(query)


def test_engine_held_and_released_cells_use_integer_lookup_with_unchanged_ownership():
    shape = (5, 5, 4)
    cells = [int(np.ravel_multi_index((x, 2, 2), shape)) for x in range(4)]
    static, first, second, removed = cells
    shifted = int(np.ravel_multi_index((2, 3, 2), shape))
    labels = np.zeros(shape, np.uint16)
    labels.ravel()[cells] = 1
    scene = SurgicalScene(labels, np.eye(4), [{'id': 1, 'name': 'Generated fragment',
        'material': 'bone'}], {'kind': 'generated occupancy lookup control'}, {})
    engine = SurgicalEngine(scene)
    engine.state['removed'] = [removed]
    engine.state['holds']['left'] = {'cells': [first, second, removed], 'offset': [1, 0, 0]}
    query = np.array([[static, first, second, removed, shifted, 0],
                      [removed, second, first, static, 0, shifted]], dtype=np.int64)[:, ::-1]

    def check(retained, destinations):
        # Literal pre-change constructor inputs retain NumPy integers.
        cleared = set(engine.state['removed'])
        cleared.update(np.asarray(retained, dtype=np.int64))
        expected = MaterialOccupancy(scene.labels, cleared,
            {destination: source + 1 for source, destination in zip(retained, destinations)})
        actual = engine.occupancy()
        assert actual._integer_cleared is actual.cleared
        if retained:
            assert expected._integer_cleared is None
        np.testing.assert_array_equal(actual[query], expected[query])
        np.testing.assert_array_equal(np.asarray(actual), np.asarray(expected))
        assert actual.material_ids(query) == expected.material_ids(query)
        for cell in query.flat:
            assert actual[int(cell)] == expected[int(cell)]
        return actual, expected

    held, _ = check([first, second], [second, removed])
    assert held[second] == first + 1 and held[removed] == second + 1
    engine.state['removed'].append(first)
    check([second], [removed])
    engine.state['holds']['left']['offset'][:] = [0, 1, 0]
    check([second], [shifted])
    engine.state.setdefault('detachedFragments', []).append(engine.state['holds'].pop('left'))
    actual, expected = check([second], [shifted])

    # Public owner edits remain visible after the integer/sorted path has run.
    for owner in (actual, expected):
        owner.cleared = frozenset([static])
        owner.moved[static] = removed + 1
        owner.memberships[static] = [first, second]
    np.testing.assert_array_equal(actual[query], expected[query])
    np.testing.assert_array_equal(np.asarray(actual), np.asarray(expected))
    assert actual.material_ids(query) == expected.material_ids(query)
    assert actual[static] == removed + 1
    engine.state['removed'].append(second)
    final, _ = check([], [])
    assert final.material_ids(query) == [static]
