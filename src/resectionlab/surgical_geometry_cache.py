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


class SurgicalCellCoverCache:
    VERSION = 'surgical-exact-cell-cover-cache-v1'

    def __init__(self, grid, source_identity, *, max_bytes=8 * 1024**2, max_entries=4096):
        if any(type(v) is not int or v < 0 for v in (max_bytes, max_entries)):
            raise ValueError('Cell cache limits must be nonnegative integers')
        self.grid = grid
        self.source_identity = source_identity
        self.max_bytes, self.max_entries = max_bytes, max_entries
        self._frame = self._frame_identity()
        self._implementation = None
        self._entries = OrderedDict()
        self._bytes = 0
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
            max_bytes=self.max_bytes, max_entries=self.max_entries)
        memo[id(self)] = result
        return result

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
            geometry._segment_cell_distances, geometry.GeometryScene.world_to_voxel,
            geometry.GeometryScene.voxel_to_world)
        key = (start.tobytes(), end.tobytes(), radius_value.dtype.str, radius_value.shape, radius_value.tobytes())
        with self._lock:
            if grid is not self.grid or self._frame_identity() != self._frame:
                raise ValueError('Cell cache grid differs from its immutable source frame')
            self._calls += 1
            if implementation != self._implementation:
                self._entries.clear()
                self._bytes = 0
                self._implementation = implementation
            value = self._entries.pop(key, None)
            if value is not None:
                self._entries[key] = value
                self._hits += 1
                return value
            self._misses += 1
            indices = kernel(grid, start, end, radius)
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
                feasibilityCached=False, occupancyCached=False)
