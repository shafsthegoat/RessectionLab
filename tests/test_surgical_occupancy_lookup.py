"""Sparse ownership lookup matches independent exact set membership."""
import numpy as np
import pytest

from resectionlab.surgical_occupancy import MaterialOccupancy


def test_sorted_cleared_lookup_handles_arbitrary_shape_duplicates_and_replaced_sets(monkeypatch):
    labels = np.ones(100, np.uint16)
    labels[13] = 0
    owner = MaterialOccupancy(labels, [3, 7, 11, 83], {7: 42})
    ids = np.array([[99, 3, 7, 3, 83], [0, 11, 13, 7, 11]])
    def expected():
        return np.array([[owner.moved.get(int(i), 0 if i in owner.cleared or not labels[i] else i + 1)
                          for i in row] for row in ids])
    np.testing.assert_array_equal(owner[ids], expected())
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
