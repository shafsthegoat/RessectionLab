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


@pytest.mark.parametrize('kind', ['identity', 'oblique', 'mirrored', 'shear'])
def test_distinct_capsules_reuse_only_exact_candidate_slabs(kind):
    affine = np.eye(4)
    if kind == 'oblique':
        affine[:3, :3] = Rotation.from_euler('xyz', [17, 23, -31], degrees=True).as_matrix() @ np.diag([.6, 1.3, 2.])
        affine[:3, 3] = [100., -70., 32.]
    if kind == 'shear': affine[:3, :3] = [[1, .6, 0], [0, 1, .3], [0, 0, 1]]
    if kind == 'mirrored': affine[:3, :3] = np.diag([-1., .7, 1.5])
    e = engine(affine)
    for offset in (0., .001, .002, .003):
        a, b = e.grid.voxel_to_world(np.array([[3. + offset, 3., 3.], [3. + offset, 3., 4.]]))
        expected = geometry.capsule_voxel_indices(e.grid, a, b, .5)
        np.testing.assert_array_equal(e.cells(a, b, .5), np.ravel_multi_index(expected.T, e.scene.shape))
    stats = e._cell_cover_cache.stats()
    assert stats['hits'] == 0 and stats['candidateHits'] >= 2
    assert stats['candidateEntries'] <= stats['maxCandidateEntries']
    for grid, world in e._cell_cover_cache._candidate_entries.values():
        for array in (grid, world):
            with pytest.raises(ValueError): array.setflags(write=True)
            with pytest.raises(ValueError): array.shape = (array.size,)


@pytest.mark.parametrize('limits', [dict(max_candidate_entries=1), dict(max_candidate_bytes=1),
    dict(max_candidate_entries=0), dict(max_candidate_bytes=0)])
def test_candidate_storage_limits_and_eviction_never_change_membership(limits):
    e = engine(); cache = SurgicalCellCoverCache(e.grid, e.scene.identity, **limits)
    for position in (1., 3., 5., 1.001):
        a = np.array([position, 3., 3.])
        expected = geometry.capsule_voxel_indices(e.grid, a, a, .5)
        actual = cache.query(e.grid, a, a, .5, kernel=geometry.capsule_voxel_indices)
        np.testing.assert_array_equal(actual, np.ravel_multi_index(expected.T, e.scene.shape))
    stats = cache.stats()
    assert stats['candidateEntries'] <= stats['maxCandidateEntries']
    assert stats['candidatePayloadBytes'] <= stats['maxCandidatePayloadBytes']
    assert stats['candidateEvictions'] + stats['candidateBypasses'] > 0


def test_candidate_helpers_invalidate_both_caches_and_custom_kernel_keeps_four_argument_contract(monkeypatch):
    e = engine(); a = np.array([3., 3., 3.]); e.cells(a, a, .1)
    previous = dict(e._cell_cover_cache._candidate_entries)
    original = geometry._capsule_candidate_grid
    calls = []
    def replacement(*args):
        calls.append(args[1:])
        return original(*args)
    monkeypatch.setattr(geometry, '_capsule_candidate_grid', replacement)
    e.cells(a, a, .1)
    assert calls and all(value is not previous.get(key) for key, value in e._cell_cover_cache._candidate_entries.items())
    def four_arguments(grid, start, end, radius):
        return geometry.capsule_voxel_indices(grid, start, end, radius)
    e._cell_cover_cache.query(e.grid, a, a, .1, kernel=four_arguments)
    assert e._cell_cover_cache.stats()['candidateEntries'] == 0


def test_long_capsules_reuse_exact_slab_partition_and_order():
    labels = np.zeros((40, 80, 80), np.uint16)
    scene = SurgicalScene(labels, np.eye(4), [], {}, {})
    e = SurgicalEngine(scene)
    e._cell_cover_cache = SurgicalCellCoverCache(e.grid, scene.identity, max_candidate_bytes=32 * 1024**2)
    for offset in (0., .001):
        a, b = np.array([.5 + offset, 40., 40.]), np.array([38.5 + offset, 40., 40.])
        expected = geometry.capsule_voxel_indices(e.grid, a, b, 39.)
        np.testing.assert_array_equal(e.cells(a, b, 39.), np.ravel_multi_index(expected.T, scene.shape))
    stats = e._cell_cover_cache.stats()
    assert stats['candidateHits'] >= 2 and stats['candidateEntries'] >= 2


def test_candidate_grazing_contacts_and_transform_replacement_match_uncached(monkeypatch):
    e = engine()
    for point in (np.array([2.5, 3., 3.]), np.array([np.nextafter(2.5, np.inf), 3., 3.]),
                  np.array([np.nextafter(2.5, -np.inf), 3., 3.])):
        for radius in (0., np.float32(.5), np.nextafter(.5, np.inf)):
            expected = geometry.capsule_voxel_indices(e.grid, point, point, radius)
            np.testing.assert_array_equal(e.cells(point, point, radius), np.ravel_multi_index(expected.T, e.scene.shape))
    old_world = geometry.GeometryScene.voxel_to_world
    calls = []
    def replacement(self, values):
        calls.append(np.shape(values))
        return old_world(self, values)
    monkeypatch.setattr(geometry.GeometryScene, 'voxel_to_world', replacement)
    a = np.array([2.5, 3., 3.]); e.cells(a, a, 0.)
    assert calls and e._cell_cover_cache.stats()['candidateEntries'] == 1


def test_public_drilling_receipts_match_with_candidate_reuse_disabled():
    from resectionlab.surgical_physics_v2 import default_profile
    labels = np.zeros((16, 16, 16), np.uint16)
    labels[4:7, 4:7, 8] = 1
    scene = SurgicalScene(labels, np.eye(4), [{'id': 1, 'name': 'bone', 'material': 'bone'}],
        {'kind': 'candidate-grid-drilling-fixture'}, {'tissuePhysicsV2': default_profile()})
    pose = lambda z: {'position': [5., 5., z], 'quaternion': [1., 0., 0., 0.]}
    actions = [
        {'slot': 'left', 'operation': 'place', 'pose': pose(15.), 'duration': .2,
         'parameters': {'radiusMm': 1., 'forceN': 0., 'rpm': 0.}},
        {'slot': 'left', 'operation': 'apply', 'pose': pose(7.), 'duration': .1},
        {'slot': 'left', 'operation': 'apply', 'pose': pose(7.), 'duration': 5.,
         'parameters': {'forceN': 20., 'rpm': 80000.}},
        {'slot': 'left', 'operation': 'withdraw', 'pose': pose(15.), 'duration': .7},
    ]
    expected, actual = SurgicalEngine(scene), SurgicalEngine(scene)
    expected._cell_cover_cache.max_candidate_bytes = 0
    for action in actions:
        assert actual.step(action) == expected.step(action)
        assert actual.history == expected.history and actual.snapshot() == expected.snapshot()
    assert actual.metrics()['drillRemovedVolumeMm3'] > 0.
    assert actual._cell_cover_cache.stats()['candidateHits'] > 0
    assert expected._cell_cover_cache.stats()['candidateEntries'] == 0
