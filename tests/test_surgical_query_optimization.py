"""Exact query equivalence for the broad-opening performance paths."""
import numpy as np
import pytest

from resectionlab.geometry import GeometryScene, _segment_cell_distances
from resectionlab.surgical_occupancy import MaterialOccupancy


def _original_values(owner, indices):
    ids = np.asarray(indices, dtype=np.int64)
    values = np.where(owner.labels[ids] != 0, ids + 1, 0)
    for at, cell in np.ndenumerate(ids):
        if int(cell) in owner.moved:
            values[at] = owner.moved[int(cell)]
        elif int(cell) in owner.cleared:
            values[at] = 0
    return values


@pytest.mark.parametrize('shape', [(0,), (100,), (5, 20), ()])
def test_sparse_ownership_matches_original_with_duplicate_negative_and_moved_indices(shape):
    rng = np.random.default_rng(20)
    labels = rng.integers(0, 4, size=(10, 10, 10), dtype=np.uint16)
    cleared = set(map(int, rng.choice(labels.size, 100, replace=False)))
    moved = {int(i): int(rng.integers(0, labels.size))
             for i in rng.choice(labels.size, 100, replace=False)}
    owner = MaterialOccupancy(labels, cleared, moved)
    indices = rng.integers(-labels.size, labels.size, size=shape)
    np.testing.assert_array_equal(owner[indices], _original_values(owner, indices))
    owner.moved[0] = 51
    indices = np.zeros(shape, dtype=np.int64)
    np.testing.assert_array_equal(owner[indices], _original_values(owner, indices))


def test_material_memberships_match_original_including_empty_memberships_and_overlaps():
    rng = np.random.default_rng(21)
    labels = rng.integers(0, 4, size=(10, 10, 10), dtype=np.uint16)
    cleared = set(map(int, rng.choice(labels.size, 100, replace=False)))
    moved = {int(i): int(rng.integers(0, labels.size))
             for i in rng.choice(labels.size, 100, replace=False)}
    memberships = {int(i): [int(i), int(rng.integers(labels.size))]
                   for i in rng.choice(labels.size, 100, replace=False)}
    memberships[0] = []
    owner = MaterialOccupancy(labels, cleared, moved, memberships)
    for indices in ([], [0, 0, 1], rng.integers(-labels.size, labels.size, size=3000)):
        expected = set()
        for cell, value in zip(indices, _original_values(owner, indices)):
            if int(cell) in memberships:
                expected.update(memberships[int(cell)])
            elif value:
                expected.add(int(value) - 1)
        assert owner.material_ids(indices) == sorted(expected)


def test_small_queries_into_long_cleared_history_preserve_moved_and_reassigned_ownership():
    rng = np.random.default_rng(20261020)
    labels = rng.integers(0, 4, size=(40, 40, 40), dtype=np.uint16)
    cleared = set(map(int, rng.choice(labels.size, 20000, replace=False)))
    owner = MaterialOccupancy(labels, cleared, {17: 91, 21: 0})
    for shape in ((0,), (), (16,), (8, 32), (1024,), (30000,)):
        ids = rng.integers(-labels.size, labels.size, size=shape)
        np.testing.assert_array_equal(owner[ids], _original_values(owner, ids))
    ids = np.array([[17, 17, 21], [0, 1, -1]])
    np.testing.assert_array_equal(owner[ids], _original_values(owner, ids))
    # A caller replacing the public cleared collection must not use a stale
    # construction-time set, even when the replacement is also immutable.
    owner.cleared = frozenset({0, 1, 17, 21})
    np.testing.assert_array_equal(owner[ids], _original_values(owner, ids))
    owner.cleared = {0, 1}
    np.testing.assert_array_equal(owner[ids], _original_values(owner, ids))
    owner.cleared.add(17)
    np.testing.assert_array_equal(owner[ids], _original_values(owner, ids))


def test_noninteger_cleared_representations_retain_legacy_vector_conversion():
    owner = MaterialOccupancy(np.ones(20, dtype=np.uint16), ['1', 2.7, True, np.int64(3)], {})
    ids = np.array([0, 1, 2, 3, 4, 4])
    expected = ids + 1
    expected[np.isin(ids, np.fromiter(owner.cleared, dtype=np.int64))] = 0
    np.testing.assert_array_equal(owner[ids], expected)


@pytest.mark.parametrize('cell', [-(1 << 80), 1 << 80])
def test_out_of_int64_cleared_identity_preserves_legacy_conversion_error(cell):
    owner = MaterialOccupancy(np.ones(20, dtype=np.uint16), [cell], {})
    with pytest.raises(OverflowError):
        owner[np.array([0])]


def _original_point_intervals(scene, indices, point):
    # Frozen preoptimization segment calculation at exactly zero delta. This
    # retains the original seven intervals and arithmetic for bitwise comparison.
    spacing = scene._orthogonal_spacing
    origin = (scene.world_to_voxel(point) - indices) * spacing
    delta = (scene.world_to_voxel(point) - scene.world_to_voxel(point)) * spacing
    half = spacing / 2
    count = len(indices)
    times = np.sort(np.concatenate([np.zeros((count, 1)), np.zeros((count, 6)),
                                    np.ones((count, 1))], axis=1), axis=1)
    left, right = times[:, :-1], times[:, 1:]
    midpoint = (left + right) / 2
    positions = origin[:, None, :] + midpoint[:, :, None] * delta
    outside = np.abs(positions) > half
    offsets = origin[:, None, :] - np.where(positions > half, half, -half)
    numerator = np.sum(np.where(outside, offsets * delta, 0), axis=2)
    denominator = np.sum(np.where(outside, delta**2, 0), axis=2)
    optimum = np.divide(-numerator, denominator, out=midpoint.copy(), where=denominator > 1e-18)
    optimum = np.clip(optimum, left, right)
    positions = origin[:, None, :] + optimum[:, :, None] * delta
    excess = np.maximum(np.abs(positions) - half, 0)
    return np.sqrt(np.min(np.sum(excess**2, axis=2), axis=1))


@pytest.mark.parametrize('basis', [np.diag([1., 1., 1.]), np.diag([-1., 2., .4]),
                                  np.array([[0., -2., 0.], [.5, 0., 0.], [0., 0., 3.]])])
def test_point_box_distance_preserves_exact_original_floats(basis):
    affine = np.eye(4)
    affine[:3, :3] = basis
    affine[:3, 3] = [-10.464996337890625, 30.441162109375, -421.25]
    scene = GeometryScene(np.zeros((20, 20, 20), bool), affine)
    rng = np.random.default_rng(22)
    indices = rng.integers(0, 20, size=(500, 3))
    # Include centers/faces/edges/corners as well as arbitrary outside points.
    for voxel in [np.zeros(3), np.ones(3) / 2, *rng.uniform(-2, 22, size=(20, 3))]:
        point = scene.voxel_to_world(voxel)
        np.testing.assert_array_equal(_segment_cell_distances(scene, indices, point, point),
                                      _original_point_intervals(scene, indices, point))


def _reference_segment_cell_distances(scene, indices, start, end):
    """Distances to orthogonal voxel boxes; a lower bound for sheared cells.

    In a box-aligned physical frame the squared segment-to-box distance is a
    convex piecewise quadratic in segment fraction. Crossing the six box faces
    partitions [0,1]; minimize each interval's quadratic and keep the minimum.
    This uses full cells and handles segments, points, and parallel faces alike.
    """
    spacing = scene._orthogonal_spacing
    if spacing is None:
        return point_segment_distances(scene.voxel_to_world(indices), start, end) - scene._cell_radius_mm
    count = len(indices)
    if not count:
        return np.empty(0)
    origin = (scene.world_to_voxel(start) - indices) * spacing
    delta = (scene.world_to_voxel(end) - scene.world_to_voxel(start)) * spacing
    half = spacing / 2
    if not np.any(delta):
        # A point has the same distance in every segment interval. Keep the
        # existing origin/excess/square/sum arithmetic, without constructing
        # seven identical copies per cell. Exact zero only; no new tolerance.
        excess = np.maximum(np.abs(origin) - half, 0)
        return np.sqrt(np.sum(excess**2, axis=1))
    crossings = np.zeros((count, 6))
    for axis in range(3):
        if abs(delta[axis]) > 1e-9:
            crossings[:, 2 * axis] = (-half[axis] - origin[:, axis]) / delta[axis]
            crossings[:, 2 * axis + 1] = (half[axis] - origin[:, axis]) / delta[axis]
    times = np.sort(np.concatenate([np.zeros((count, 1)), np.clip(crossings, 0, 1), np.ones((count, 1))], axis=1), axis=1)
    left, right = times[:, :-1], times[:, 1:]
    midpoint = (left + right) / 2
    positions = origin[:, None, :] + midpoint[:, :, None] * delta
    outside = np.abs(positions) > half
    offsets = origin[:, None, :] - np.where(positions > half, half, -half)
    numerator = np.sum(np.where(outside, offsets * delta, 0), axis=2)
    denominator = np.sum(np.where(outside, delta**2, 0), axis=2)
    optimum = np.divide(-numerator, denominator, out=midpoint.copy(), where=denominator > 1e-9**2)
    optimum = np.clip(optimum, left, right)
    positions = origin[:, None, :] + optimum[:, :, None] * delta
    excess = np.maximum(np.abs(positions) - half, 0)
    return np.sqrt(np.min(np.sum(excess**2, axis=2), axis=1))


@pytest.mark.parametrize('axis', [0,1,2])
@pytest.mark.parametrize('basis', [np.diag([1.,1.,1.]),np.diag([-1.,2.,.4])])
def test_axial_capsule_distances_preserve_original_minimizer(axis,basis):
    affine=np.eye(4); affine[:3,:3]=basis; affine[:3,3]=[-10.464996337890625,30.441162109375,-421.25]
    scene=GeometryScene(np.zeros((20,20,20),bool),affine)
    rng=np.random.default_rng(180+axis)
    indices=rng.integers(0,20,size=(1000,3))
    for _ in range(100):
        start=rng.uniform(-3,23,size=3); end=start.copy();end[axis]+=rng.uniform(-40,40)
        a,b=scene.voxel_to_world(np.array([start,end]))
        actual=_segment_cell_distances(scene,indices,a,b)
        expected=_reference_segment_cell_distances(scene,indices,a,b)
        np.testing.assert_array_equal(actual,expected)
