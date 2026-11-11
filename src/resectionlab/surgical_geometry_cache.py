"""Bounded pure capsule covers for one surgical engine's immutable grid.

Transactional engine clones share this cache, never mutable tissue, occupancy,
clearance or acceptance decisions. Fresh engines own an empty cache. Qualified
numerical backends produce the same ordered full-cell covers, including shear.
"""
from collections import OrderedDict
from threading import RLock
from types import FunctionType

import numpy as np

from . import geometry
from .core import immutable_array

_REFERENCE_CAPSULE_KERNEL = geometry.capsule_voxel_indices
_DEFAULT_ATOMS = (type(None), bool, int, float, str, bytes)


def _function_identity(function):
    if type(function) is not FunctionType:
        return id(function), function, object()
    code, defaults, keywords = function.__code__, function.__defaults__, function.__kwdefaults__
    if defaults is None and keywords is None:
        return id(function), function, id(code), code, None
    items = tuple(dict.items(keywords)) if type(keywords) is dict else ()
    scalar_defaults = ((defaults is None or type(defaults) is tuple)
        and (keywords is None or type(keywords) is dict)
        and all(any(type(value) is atom for atom in _DEFAULT_ATOMS)
            for value in (() if defaults is None else defaults) + tuple(v for _, v in items)))
    # Retain identities without invoking user equality; mutable defaults cannot authorize reuse.
    return (id(function), function, id(code), code, id(defaults), defaults, id(keywords), keywords,
        tuple((id(k), k, id(v), v) for k, v in items), None if scalar_defaults else object())


class SurgicalCellCoverCache:
    VERSION = 'surgical-exact-cell-cover-cache-v3'

    def __init__(self, grid, source_identity, *, max_bytes=8 * 1024**2, max_entries=4096,
                 max_candidate_bytes=8 * 1024**2, max_candidate_entries=256):
        if any(type(v) is not int or v < 0 for v in
               (max_bytes, max_entries, max_candidate_bytes, max_candidate_entries)):
            raise ValueError('Cell cache limits must be nonnegative integers')
        self.grid = grid
        self.source_identity = source_identity
        self.max_bytes, self.max_entries = max_bytes, max_entries
        self.max_candidate_bytes, self.max_candidate_entries = max_candidate_bytes, max_candidate_entries
        self._frame = self._frame_identity()
        self._implementation = None
        self._entries = OrderedDict()
        self._bytes = 0
        self._candidate_entries = OrderedDict()
        self._candidate_bytes = 0
        self._candidate_hits = self._candidate_misses = self._candidate_evictions = self._candidate_bypasses = 0
        self._calls = self._hits = self._misses = self._evictions = self._bypasses = 0
        self._lock = RLock()

    def _frame_identity(self):
        grid = self.grid
        return (grid.shape, grid.affine.tobytes(), grid._inverse.tobytes(),
                grid._cell_radius_mm, None if grid._orthogonal_spacing is None else grid._orthogonal_spacing.tobytes())

    def __deepcopy__(self, memo):
        # Generic callers may still deep-copy an engine. Locks are not
        # pickleable; the copied engine receives its own empty cache/grid.
        import copy
        result = type(self)(copy.deepcopy(self.grid, memo), self.source_identity,
            max_bytes=self.max_bytes, max_entries=self.max_entries,
            max_candidate_bytes=self.max_candidate_bytes, max_candidate_entries=self.max_candidate_entries)
        memo[id(self)] = result
        return result

    def _candidate_grid(self, grid, lo, hi):
        # Called only inside query's lock by the known reference kernel. The
        # frame and all numerical helper identities have already been checked.
        # Bounds describe an entire original slab, not just occupied tissue or
        # the cells selected by a previous capsule's distance predicates.
        key = tuple(int(v) for v in (*lo, *hi))
        value = self._candidate_entries.pop(key, None)
        if value is not None:
            self._candidate_entries[key] = value
            self._candidate_hits += 1
            return value
        self._candidate_misses += 1
        cells, world = geometry._capsule_candidate_grid(grid, lo, hi)
        size = cells.nbytes + world.nbytes
        if not self.max_candidate_entries or not self.max_candidate_bytes or size > self.max_candidate_bytes:
            self._candidate_bypasses += 1
            return cells, world
        while self._candidate_entries and (len(self._candidate_entries) >= self.max_candidate_entries or
                                          self._candidate_bytes + size > self.max_candidate_bytes):
            _, previous = self._candidate_entries.popitem(last=False)
            self._candidate_bytes -= sum(array.nbytes for array in previous)
            self._candidate_evictions += 1
        value = geometry._immutable(cells), geometry._immutable(world)
        self._candidate_entries[key] = value
        self._candidate_bytes += size
        return value

    def _query_values(self, start, end, radius):
        # Validate hits with the same vector/radius rule as the reference. Keep
        # the radius dtype too: NumPy scalar promotion can affect boundary bits.
        start, end = (np.array(value, dtype=float, copy=True) for value in (start, end))
        for name, value in (('start_mm', start), ('end_mm', end)):
            if value.shape != (3,) or not np.all(np.isfinite(value)):
                raise ValueError(f'{name} must be a finite three-vector')
        if not np.isfinite(radius) or radius < 0:
            raise ValueError('radius_mm must be finite and nonnegative')
        radius_value = np.asarray(radius)
        key = (start.tobytes(), end.tobytes(), radius_value.dtype.str, radius_value.shape, radius_value.tobytes())
        return (start, end, radius), key

    def _implementation_key(self, kernel):
        from .native_backend import implementation_identity
        functions = (kernel,
            geometry._vector, geometry._immutable, geometry.point_segment_distances,
            geometry._segment_cell_distances, geometry._capsule_candidate_grid, geometry.GeometryScene.world_to_voxel,
            geometry.GeometryScene.voxel_to_world, geometry.capsule_voxel_indices_batch,
            geometry._capsule_voxel_indices_reference, geometry._native_capsule_supported,
            type(self)._query_values, type(self)._remember, _function_identity)
        return (self.VERSION, geometry.GEOMETRY_VERSION, geometry._EPS, implementation_identity(),
            tuple(_function_identity(function) for function in functions))

    def _validate_frame(self, grid, implementation):
        if grid is not self.grid or self._frame_identity() != self._frame:
            raise ValueError('Cell cache grid differs from its immutable source frame')
        if implementation != self._implementation:
            self._entries.clear()
            self._bytes = 0
            self._candidate_entries.clear()
            self._candidate_bytes = 0
            self._implementation = implementation

    def _remember(self, key, value):
        if not self.max_entries or not self.max_bytes or value.nbytes > self.max_bytes:
            self._bypasses += 1
            return value
        while self._entries and (len(self._entries) >= self.max_entries or self._bytes + value.nbytes > self.max_bytes):
            _, previous = self._entries.popitem(last=False)
            self._bytes -= previous.nbytes
            self._evictions += 1
        self._entries[key] = value
        self._bytes += value.nbytes
        return value

    def query(self, grid, start, end, radius, *, kernel, check=None):
        (start, end, radius), key = self._query_values(start, end, radius)
        implementation = self._implementation_key(kernel)
        with self._lock:
            self._validate_frame(grid, implementation)
            self._calls += 1
            if check is not None: check()
            value = self._entries.pop(key, None)
            if value is not None:
                self._entries[key] = value
                self._hits += 1
                return value
            self._misses += 1
            indices = (kernel(grid, start, end, radius, _candidate_grid=self._candidate_grid, check=check)
                       if kernel is _REFERENCE_CAPSULE_KERNEL else kernel(grid, start, end, radius))
            value = immutable_array(np.ravel_multi_index(indices.T, grid.shape) if len(indices) else [], np.int64)
            if check is not None: check()
            return self._remember(key, value)

    def query_many(self, grid, queries, *, kernel, check=None):
        """Group pure misses; publish covers only after the complete batch check.

        Replay cache admission in query order, preserving scalar LRU behavior
        even when an earlier oversized cover evicts a later requested hit.
        """
        if not isinstance(queries, (list, tuple)) or len(queries) > 64:
            raise ValueError('Cell cover batch requires at most 64 queries')
        if kernel is not _REFERENCE_CAPSULE_KERNEL:
            return tuple(self.query(grid, *query, kernel=kernel, check=check) for query in queries)
        prepared = [self._query_values(*query) for query in queries]
        implementation = self._implementation_key(kernel)
        with self._lock:
            self._validate_frame(grid, implementation)
            if check is not None: check()
            ready = {key: self._entries[key] for _, key in prepared if key in self._entries}
            missing = {}
            for query, key in prepared:
                if key not in ready:
                    missing.setdefault(key, query)
            if missing:
                covers = geometry.capsule_voxel_indices_batch(grid, tuple(missing.values()),
                    check=check, _candidate_grid=self._candidate_grid)
                if len(covers) != len(missing):
                    raise ValueError('Cell cover batch result count differs')
                for key, indices in zip(missing, covers):
                    if check is not None: check()
                    ready[key] = immutable_array(np.ravel_multi_index(indices.T, grid.shape)
                        if len(indices) else [], np.int64)
            if check is not None: check()
            values = []
            for _, key in prepared:
                self._calls += 1
                value = self._entries.pop(key, None)
                if value is not None:
                    self._entries[key] = value
                    self._hits += 1
                else:
                    self._misses += 1
                    value = self._remember(key, ready[key])
                values.append(value)
            return tuple(values)

    def stats(self):
        with self._lock:
            return dict(version=self.VERSION, sourceIdentity=self.source_identity,
                calls=self._calls, hits=self._hits, misses=self._misses, evictions=self._evictions,
                bypasses=self._bypasses, entries=len(self._entries), payloadBytes=self._bytes,
                maxEntries=self.max_entries, maxPayloadBytes=self.max_bytes,
                candidateHits=self._candidate_hits, candidateMisses=self._candidate_misses,
                candidateEvictions=self._candidate_evictions, candidateBypasses=self._candidate_bypasses,
                candidateEntries=len(self._candidate_entries), candidatePayloadBytes=self._candidate_bytes,
                maxCandidateEntries=self.max_candidate_entries, maxCandidatePayloadBytes=self.max_candidate_bytes,
                feasibilityCached=False, occupancyCached=False)
