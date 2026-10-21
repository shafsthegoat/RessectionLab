"""Exact cell covers remain independent of mutable surgical consequences."""
import copy
import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from resectionlab import geometry
from resectionlab.evaluation import segment_box_distance_sq
from resectionlab.surgical_engine import SurgicalEngine
from resectionlab.surgical_geometry_cache import SurgicalCellCoverCache
from resectionlab.surgical_scene import SurgicalScene


def engine(affine=None):
    labels = np.zeros((7, 7, 7), np.uint16)
    labels[3, 3, 3] = 1
    return SurgicalEngine(SurgicalScene(labels, np.eye(4) if affine is None else affine,
        [dict(id=1, name='Generated target', material='tumor', target=True)], {}, {}))


@pytest.mark.parametrize('kind', ['identity', 'oblique', 'shear'])
def test_cached_full_cell_membership_matches_reference_at_contacts_and_degenerate_segments(kind):
    affine = np.eye(4)
    if kind == 'oblique':
        affine[:3, :3] = Rotation.from_euler('xyz', [17, 23, -31], degrees=True).as_matrix() @ np.diag([.6, 1.3, 2.])
        affine[:3, 3] = [100., -70., 32.]
    if kind == 'shear': affine[:3, :3] = [[1, .6, 0], [0, 1, .3], [0, 0, 1]]
    e = engine(affine)
    for first, second, radius in [([3, 3, 3], [3, 3, 3], 0.), ([3, 3, 1], [3, 3, 5], .5),
        ([1.1, 2.3, 3.5], [5.2, 4.1, 2.9], .2), ([-8, -8, -8], [-7, -7, -7], .01)]:
        a, b = e.grid.voxel_to_world(np.array([first, second]))
        expected = geometry.capsule_voxel_indices(e.grid, a, b, radius)
        expected = np.ravel_multi_index(expected.T, e.scene.shape) if len(expected) else np.empty(0, np.int64)
        for _ in range(2):
            actual = e.cells(a, b, radius)
            np.testing.assert_array_equal(actual, expected)
            with pytest.raises(ValueError): actual.setflags(write=True)
    assert e._cell_cover_cache.stats()['hits'] == 4


def test_axis_aligned_grazing_contacts_match_independent_scalar_boxes():
    e = engine()
    all_indices = np.indices(e.scene.shape).reshape(3, -1).T
    for a, b, radius in [(np.array([2.5, 3., 1.]), np.array([2.5, 3., 5.]), 0.),
                         (np.array([3., 3., 3.]), np.array([3., 3., 3.]), .5)]:
        expected = [i for i, cell in enumerate(all_indices)
            if segment_box_distance_sq(a, b, cell - .5, cell + .5) <= (radius + geometry._EPS)**2]
        np.testing.assert_array_equal(e.cells(a, b, radius), expected)


def test_clones_share_only_pure_covers_and_changed_tissue_still_changes_contact():
    e = engine(); clone = e.clone(); a = np.array([3., 3., 3.])
    before = e.state_id
    assert e.touched(a, a, .1)
    clone.state['removed'] = [int(np.ravel_multi_index((3, 3, 3), e.scene.shape))]
    assert not clone.touched(a, a, .1)
    assert e.touched(a, a, .1) and e.state_id == before
    assert clone._cell_cover_cache is e._cell_cover_cache
    fresh = e.fresh()
    assert fresh._cell_cover_cache is not e._cell_cover_cache
    assert fresh._cell_cover_cache.stats()['entries'] == 0
    another = engine()
    assert another._cell_cover_cache is not e._cell_cover_cache
    with pytest.raises(ValueError, match='source frame'):
        e._cell_cover_cache.query(another.grid, a, a, .1, kernel=geometry.capsule_voxel_indices)


@pytest.mark.parametrize('limits', [dict(max_entries=1, max_bytes=64), dict(max_entries=0), dict(max_bytes=0)])
def test_bounded_empty_and_oversized_entries_preserve_results(limits):
    e = engine(); cache = SurgicalCellCoverCache(e.grid, e.scene.identity, **limits)
    for offset in range(4):
        cache.query(e.grid, [30 + offset, 30, 30], [31 + offset, 30, 30], .1, kernel=geometry.capsule_voxel_indices)
    cache.query(e.grid, [0, 0, 0], [6, 6, 6], 2., kernel=geometry.capsule_voxel_indices)
    stats = cache.stats()
    assert stats['entries'] <= stats['maxEntries'] and stats['payloadBytes'] <= stats['maxPayloadBytes']


def test_exact_parameter_bits_and_replaced_kernel_do_not_reuse_old_entries(monkeypatch):
    e = engine(); a = np.array([3., 3., 3.]); original = geometry.capsule_voxel_indices
    for radius in (.5, np.nextafter(.5, np.inf), np.float32(.5)):
        e.cells(a, a, radius)
    assert e._cell_cover_cache.stats()['misses'] == 3
    monkeypatch.setattr('resectionlab.surgical_engine.capsule_voxel_indices', lambda *args: np.empty((0, 3), np.int64))
    assert not len(e.cells(a, a, .5))
    monkeypatch.setattr('resectionlab.surgical_engine.capsule_voxel_indices', original)
    assert len(e.cells(a, a, .5))
    for bad in (float('nan'), -1.):
        with pytest.raises(ValueError): e.cells(a, a, bad)


def test_cached_contact_does_not_admit_missing_critical_coverage():
    e = engine(); a = np.array([3., 3., 3.]); e.cells(a, a, .1); e.cells(a, a, .1)
    assert e.guard_report()['status'] == 'unknown-coverage'
    assert e.guard_report()['missingRequired'] == ['brainstem']


def test_generic_engine_deepcopy_owns_an_empty_compatible_cache():
    e = engine(); a = np.array([3., 3., 3.]); e.cells(a, a, .1)
    other = copy.deepcopy(e)
    assert other._cell_cover_cache is not e._cell_cover_cache
    assert other._cell_cover_cache.stats()['entries'] == 0
    np.testing.assert_array_equal(other.cells(a, a, .1), e.cells(a, a, .1))
