"""Bounded pure capsule covers for one surgical engine's immutable grid.

Transactional engine clones share this cache, never mutable tissue, occupancy,
clearance or acceptance decisions. Fresh engines own an empty cache. The exact
reference kernel remains the only producer of cell membership, including shear.
"""
from collections import OrderedDict
from threading import RLock

import numpy as np

from . import geometry
from .core import immutable_array

_REFERENCE_CAPSULE_KERNEL = geometry.capsule_voxel_indices


class SurgicalCellCoverCache:
    VERSION = 'surgical-exact-cell-cover-cache-v2'

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

    def query(self, grid, start, end, radius, *, kernel):
        # Validate hits with the same vector/radius rule as the reference. Keep
        # the radius dtype too: NumPy scalar promotion can affect boundary bits.
        start, end = (np.array(value, dtype=float, copy=True) for value in (start, end))
        for name, value in (('start_mm', start), ('end_mm', end)):
            if value.shape != (3,) or not np.all(np.isfinite(value)):
                raise ValueError(f'{name} must be a finite three-vector')
        if not np.isfinite(radius) or radius < 0:
            raise ValueError('radius_mm must be finite and nonnegative')
        radius_value = np.asarray(radius)
        implementation = (self.VERSION, geometry.GEOMETRY_VERSION, geometry._EPS, kernel,
            geometry._vector, geometry._immutable, geometry.point_segment_distances,
            geometry._segment_cell_distances, geometry._capsule_candidate_grid, geometry.GeometryScene.world_to_voxel,
            geometry.GeometryScene.voxel_to_world)
        key = (start.tobytes(), end.tobytes(), radius_value.dtype.str, radius_value.shape, radius_value.tobytes())
        with self._lock:
            if grid is not self.grid or self._frame_identity() != self._frame:
                raise ValueError('Cell cache grid differs from its immutable source frame')
            self._calls += 1
            if implementation != self._implementation:
                self._entries.clear()
                self._bytes = 0
                self._candidate_entries.clear()
                self._candidate_bytes = 0
                self._implementation = implementation
            value = self._entries.pop(key, None)
            if value is not None:
                self._entries[key] = value
                self._hits += 1
                return value
            self._misses += 1
            indices = (kernel(grid, start, end, radius, _candidate_grid=self._candidate_grid)
                       if kernel is _REFERENCE_CAPSULE_KERNEL else kernel(grid, start, end, radius))
            value = immutable_array(np.ravel_multi_index(indices.T, grid.shape) if len(indices) else [], np.int64)
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
