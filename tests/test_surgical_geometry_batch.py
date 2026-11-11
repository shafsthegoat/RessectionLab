"""Pure multi-cover batching leaves physical transactions and public hooks intact."""
import numpy as np
import pytest

from resectionlab import geometry, native_backend
from resectionlab.surgical_engine import SurgicalEngine
from resectionlab.surgical_geometry_cache import SurgicalCellCoverCache
from resectionlab.surgical_scene import SurgicalScene


def source(cls=SurgicalEngine):
    labels = np.zeros((7, 7, 7), np.uint16)
    labels[3, 3, 3] = 1
    return cls(SurgicalScene(labels, np.eye(4),
        [dict(id=1, name='Generated target', material='tumor', target=True)], {}, {}))


def test_batch_matches_scalar_order_and_lru_admission_with_duplicate_misses():
    engine = source()
    scalar = SurgicalCellCoverCache(engine.grid, engine.scene.identity, max_entries=1)
    batch = SurgicalCellCoverCache(engine.grid, engine.scene.identity, max_entries=1)
    first, second = ([3., 3., 3.], [3., 3., 3.], .1), ([1., 2., 3.], [4., 5., 6.], .2)
    queries = (first, second, first)
    expected = tuple(scalar.query(engine.grid, *q, kernel=geometry.capsule_voxel_indices) for q in queries)
    actual = batch.query_many(engine.grid, queries, kernel=geometry.capsule_voxel_indices)
    for a, b in zip(actual, expected):
        np.testing.assert_array_equal(a, b)
        with pytest.raises(ValueError): a.setflags(write=True)
    for name in ('calls', 'hits', 'misses', 'entries', 'payloadBytes', 'evictions'):
        assert scalar.stats()[name] == batch.stats()[name]
    assert list(scalar._entries) == list(batch._entries)


def test_batch_cancel_after_computation_does_not_publish_partial_covers(monkeypatch):
    engine = source()
    cache = engine._cell_cover_cache
    original = geometry.capsule_voxel_indices_batch
    computed = False
    def batch(*args, **kwargs):
        nonlocal computed
        value = original(*args, **kwargs)
        computed = True
        return value
    def check():
        if computed:
            raise InterruptedError('cancel before cover publication')
    monkeypatch.setattr(geometry, 'capsule_voxel_indices_batch', batch)
    before = engine.snapshot()
    with pytest.raises(InterruptedError, match='publication'):
        cache.query_many(engine.grid, (([3., 3., 3.], [3., 3., 3.], .1),
            ([1., 1., 1.], [2., 2., 2.], .1)), kernel=geometry.capsule_voxel_indices, check=check)
    assert cache.stats()['entries'] == 0
    assert engine.snapshot() == before


def test_touched_batch_keeps_replaced_public_hook_order_and_mutable_occupancy():
    engine = source()
    calls = []
    original = engine.touched
    def touched(*query):
        calls.append(query)
        result = original(*query)
        engine.state['removed'] = [int(np.ravel_multi_index((3, 3, 3), engine.scene.shape))]
        return result
    engine.touched = touched
    query = ([3., 3., 3.], [3., 3., 3.], .1)
    first, second = engine.touched_many((query, query))
    assert first and not second and calls == [query, query]


def test_touched_batch_keeps_replaced_occupancy_callback_between_geometry_queries():
    engine = source()
    query = ([3., 3., 3.], [3., 3., 3.], .1)
    assert engine.touched(*query)
    owner = engine._occupancy_cache[1]
    original = owner.material_ids
    def material_ids(cells):
        engine.cells = lambda *args: np.empty(0, dtype=np.int64)
        return original(cells)
    owner.material_ids = material_ids
    first, second = engine.touched_many((query, query))
    assert first and not second


def test_cache_identity_includes_selected_native_build(monkeypatch):
    engine = source()
    query = ([3., 3., 3.], [3., 3., 3.], .1)
    engine.cells(*query)
    previous = native_backend.implementation_identity()
    monkeypatch.setattr(native_backend, 'implementation_identity', lambda: (*previous, 'different-build'))
    engine.cells(*query)
    assert engine._cell_cover_cache.stats()['misses'] == 2
    assert engine._cell_cover_cache.stats()['hits'] == 0


def test_cache_identity_includes_reference_fallback_producer(monkeypatch):
    engine = source()
    query = ([3., 3., 3.], [3., 3., 3.], .1)
    assert len(engine.cells(*query))
    monkeypatch.setattr(native_backend, 'cpp_enabled', lambda: False)
    monkeypatch.setattr(geometry, '_capsule_voxel_indices_reference',
        lambda *args, **kwargs: np.empty((0, 3), dtype=np.int64))
    assert not len(engine.cells(*query))
    assert engine._cell_cover_cache.stats()['misses'] == 2


def test_cache_in_place_geometry_code_change_invalidates_and_restores_membership(monkeypatch):
    engine = source()
    query = ([3., 3., 3.], [3., 3., 3.], .1)
    original = engine.cells(*query)
    def no_contacts(scene, indices, start, end):
        return np.full(len(indices), np.inf)
    with monkeypatch.context() as patch:
        patch.setattr(geometry._segment_cell_distances, '__code__', no_contacts.__code__)
        assert not len(engine.cells(*query))
    np.testing.assert_array_equal(engine.cells(*query), original)
    assert engine._cell_cover_cache.stats()['misses'] == 3
    assert engine._cell_cover_cache.stats()['hits'] == 0


@pytest.mark.parametrize('kind', ['defaults', 'keywords', 'mutable-default'])
def test_cache_in_place_kernel_default_change_invalidates_without_changing_query(kind):
    engine = source()
    query = ([3., 3., 3.], [3., 3., 3.], .1)
    def kernel(grid, start, end, radius, enabled=True, *, included=True):
        return (geometry.capsule_voxel_indices(grid, start, end, radius)
            if enabled and included else np.empty((0, 3), dtype=np.int64))
    if kind == 'mutable-default':
        def kernel(grid, start, end, radius, enabled=[True]):
            return (geometry.capsule_voxel_indices(grid, start, end, radius)
                if enabled[0] else np.empty((0, 3), dtype=np.int64))
    cache = engine._cell_cover_cache
    assert len(cache.query(engine.grid, *query, kernel=kernel))
    if kind == 'defaults':
        kernel.__defaults__ = (False,)
    elif kind == 'keywords':
        kernel.__kwdefaults__['included'] = False
    else:
        kernel.__defaults__[0][0] = False
    assert not len(cache.query(engine.grid, *query, kernel=kernel))
    assert cache.stats()['misses'] == 2 and cache.stats()['hits'] == 0


def test_batch_limits_and_bad_vectors_fail_without_cover_publication():
    engine = source()
    query = ([3., 3., 3.], [3., 3., 3.], .1)
    for queries in ((query,) * 65, (query, ([float('nan'), 0., 0.], [0., 0., 0.], .1))):
        with pytest.raises(ValueError):
            engine.cells_many(queries)
        assert engine._cell_cover_cache.stats()['entries'] == 0


def test_public_transaction_submits_multiple_cold_covers_and_matches_sequential():
    if not native_backend.cpp_enabled():
        pytest.skip('Compiled backend required for binding execution evidence')
    class Sequential(SurgicalEngine):
        def touched_many(self, queries):
            return tuple(self.touched(*q) for q in queries)
    actual, sequential = source(), source(Sequential)
    action = {'slot': 'left', 'operation': 'place', 'duration': .1,
        'pose': {'position': [0., 0., 3.], 'quaternion': [0., 0., 0., 1.]}}
    native_backend.reset_counters()
    result = actual.step(action)
    usage = native_backend.backend_identity()['counters']
    assert result['status'] == 'accepted'
    assert usage['maxBatchItems'] >= 2 and usage['batchItems'] > 1
    assert usage['maxPrimitiveCandidates'] <= native_backend.MAX_PRIMITIVES
    assert result == sequential.step(action)
    assert actual.snapshot() == sequential.snapshot()
