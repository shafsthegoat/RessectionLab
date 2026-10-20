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
