"""Exact membership and cancellation at the integrated native geometry boundary."""
import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from resectionlab import geometry, native_backend


@pytest.mark.parametrize('kind', ['identity', 'anisotropic', 'oblique', 'shear', 'mirrored'])
def test_compiled_covers_equal_retained_reference_in_original_order(kind):
    if not native_backend.cpp_enabled():
        pytest.skip('compiled extension unavailable or reference selected')
    affine = np.eye(4)
    if kind == 'anisotropic':
        affine[:3, :3] = np.diag([.6, 1.3, 2.])
    if kind == 'oblique':
        affine[:3, :3] = Rotation.from_euler('xyz', [17, 23, -31], degrees=True).as_matrix() @ np.diag([.6, 1.3, 2.])
        affine[:3, 3] = [100., -70., 32.]
    if kind == 'shear':
        affine[:3, :3] = [[1., .6, 0.], [0., 1., .3], [0., 0., 1.]]
    if kind == 'mirrored':
        affine[:3, :3] = np.diag([-1., .7, 1.5])
    scene = geometry.GeometryScene(np.zeros((11, 13, 15), bool), affine)
    rng = np.random.default_rng(7561)
    pairs = rng.uniform(-3, 17, (28, 2, 3))
    pairs[:4] = [[[3, 3, 3], [3, 3, 3]], [[2.5, 3, 1], [2.5, 3, 12]],
                 [[2.5, 3, 3], [2.5, 3, 3]], [[-9, -9, -9], [-8, -8, -8]]]
    queries = []
    for index, pair in enumerate(pairs):
        start, end = scene.voxel_to_world(pair)
        radius = (0., .5, np.float32(.5), np.nextafter(.5, np.inf))[index % 4]
        queries.append((start, end, radius))
    actual = geometry.capsule_voxel_indices_batch(scene, queries)
    for query, result in zip(queries, actual):
        expected = geometry._capsule_voxel_indices_reference(scene, *query)
        np.testing.assert_array_equal(result, expected)
        with pytest.raises(ValueError):
            result.setflags(write=True)
    assert native_backend.backend_identity()['counters']['maxBatchItems'] > 1


def test_large_query_chunks_and_cancellation_publish_no_results():
    if not native_backend.cpp_enabled():
        pytest.skip('compiled extension unavailable or reference selected')
    scene = geometry.GeometryScene(np.zeros((96, 96, 128), bool), np.eye(4))
    query = ([0., 0., 0.], [95., 95., 127.], 40.)
    native_backend.reset_counters()
    calls = []
    class Cancelled(Exception):
        pass
    def check():
        calls.append(1)
        if len(calls) == 4:
            raise Cancelled
    with pytest.raises(Cancelled):
        geometry.capsule_voxel_indices_batch(scene, [query, query], check=check)
    stats = native_backend.backend_identity()['counters']
    assert stats['bindingCalls'] == 1
    assert stats['maxPrimitiveCandidates'] <= native_backend.MAX_PRIMITIVES
    assert stats['maxBatchItems'] == 2


def test_invalid_batch_items_and_coordinates_reject_before_native_work():
    scene = geometry.GeometryScene(np.zeros((3, 3, 3), bool), np.eye(4))
    with pytest.raises(ValueError):
        geometry.capsule_voxel_indices_batch(scene, [([0, 0, 0], [0, 0, 0], 0.)] * 65)
    with pytest.raises(ValueError):
        geometry.capsule_voxel_indices_batch(scene, iter(()))
    for query in [([np.nan, 0., 0.], [0., 0., 0.], 0.),
                  ([0., 0., 0.], [0., 0., 0.], -1.)]:
        with pytest.raises(ValueError):
            geometry.capsule_voxel_indices_batch(scene, [query])


@pytest.mark.parametrize('kind', ['points', 'point-segment', 'skew', 'parallel', 'degenerate'])
def test_native_segment_boolean_matches_original_threshold_predicate(kind):
    if not native_backend.cpp_enabled():
        pytest.skip('compiled extension unavailable or reference selected')
    rng = np.random.default_rng(9761)
    for _ in range(100):
        points = rng.uniform(-50., 50., (4, 3))
        if kind == 'points':
            points[1] = points[0]; points[3] = points[2]
        elif kind == 'point-segment':
            points[1] = points[0]
        elif kind == 'parallel':
            points[3] = points[2] + points[1] - points[0]
        elif kind == 'degenerate':
            points[1] = points[0] + rng.uniform(-1e-9, 1e-9, 3)
        distance = geometry.segment_segment_distance(*points)
        for threshold in (distance, np.nextafter(distance, 0.),
                          np.nextafter(distance, np.inf), max(0., distance - .1), distance + .1):
            assert geometry.segments_within_distance(*points, threshold) == (distance <= threshold)
    counters = native_backend.backend_identity()['counters']
    assert counters['segmentPredicateCalls'] > 0
    assert counters['segmentPredicateRefinements'] > 0


def test_segment_predicate_keeps_replaced_reference_hook(monkeypatch):
    monkeypatch.setattr(geometry, 'segment_segment_distance', lambda *args: 5.)
    assert not geometry.segments_within_distance(*np.zeros((4, 3)), 4.)
    assert geometry.segments_within_distance(*np.zeros((4, 3)), 5.)


@pytest.mark.parametrize('axis', [0, 1, 2])
def test_axis_parallel_predicate_matches_reference_at_grazing_and_nextafter(axis):
    rng = np.random.default_rng(1923)
    for _ in range(50):
        points = rng.uniform(-50., 50., (4, 3))
        points[1] = points[0]; points[3] = points[2]
        points[1, axis] += rng.uniform(-30., 30.)
        points[3, axis] += rng.uniform(-30., 30.)
        distance = geometry.segment_segment_distance(*points)
        for threshold in (distance, np.nextafter(distance, 0.),
                          np.nextafter(distance, np.inf), max(0., distance - .1), distance + .1):
            assert geometry.segments_within_distance(*points, threshold) == (distance <= threshold)


def test_tiny_segment_interior_crossing_keeps_original_determinant_criterion():
    points = np.array([[-4e-10, 0., 0.], [4e-10, 0., 0.], [0., -10., 0.], [0., 10., 0.]])
    assert geometry.segment_segment_distance(*points) == 0.
    assert geometry.segments_within_distance(*points, 0.)


def test_float32_threshold_retains_numpy_scalar_comparison():
    points = np.array([[0., 0., 0.], [0., 0., 0.], [5.00000001, 0., 0.], [5.00000001, 0., 0.]])
    threshold = np.float32(5.)
    assert geometry.segments_within_distance(*points, threshold) == (geometry.segment_segment_distance(*points) <= threshold)
