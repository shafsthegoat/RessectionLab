"""Opt-in, process-local capsule-cover prototype; never caches clearance.

Production engines and adapters do not import this module. The explicit context
manager temporarily replaces only the native engine's pure cell-cover query.
Complete hard/access, prior-cavity shaft, containment and frontier checks remain
in their original code. One cache and injection belong to one worker thread.
"""
from __future__ import annotations

from collections import OrderedDict
from contextlib import contextmanager
from dataclasses import replace
from hashlib import sha256
import json
from pathlib import Path
import threading
from time import perf_counter
from typing import Any

import numpy as np

from . import geometry, native_proposals, native_resection
from .native_proposals import _array_identity, _source_identity

CAPSULE_CACHE_VERSION = "experimental-exact-capsule-cover-cache-v1"
_ORIGINAL_COVER = geometry.capsule_voxel_indices
_INJECTION_LOCK = threading.Lock()
_GEOMETRY_DEPENDENCIES = (
    (geometry, "_vector", geometry._vector),
    (geometry, "_immutable", geometry._immutable),
    (geometry, "point_segment_distances", geometry.point_segment_distances),
    (geometry, "_segment_cell_distances", geometry._segment_cell_distances),
    (geometry.GeometryScene, "world_to_voxel", geometry.GeometryScene.world_to_voxel),
    (geometry.GeometryScene, "voxel_to_world", geometry.GeometryScene.voxel_to_world),
)


class ExactCapsuleCoverCache:
    """A bounded LRU of immutable coverage arrays for one immutable source.

    Payload and entry limits are independent: empty arrays still occupy an
    entry. Keys retain exact float64 bits. Inputs and derived scene descriptors
    are validated on hits as well as misses; there is no cached integrity result.
    """

    def __init__(self, native_config: native_resection.NativeResectionConfig, *,
                 max_payload_bytes: int = 32 * 1024**2, max_entries: int = 16384):
        for name, value in (("max_payload_bytes", max_payload_bytes), ("max_entries", max_entries)):
            if type(value) is not int or value < 0:
                raise ValueError(f"{name} must be a nonnegative integer")
        if not isinstance(native_config, native_resection.NativeResectionConfig):
            raise TypeError("A validated native configuration is required")
        if replace(native_config).fingerprint != native_config.fingerprint:
            raise ValueError("Native source contents differ from their cached fingerprint")
        self._config = native_config
        self._source_identity = _source_identity(native_config)
        self._owner_thread = threading.get_ident()
        self._max_payload_bytes, self._max_entries = max_payload_bytes, max_entries
        self._geometry_version = geometry.GEOMETRY_VERSION
        self._native_version = native_resection.NATIVE_RESECTION_VERSION
        self._geometry_epsilon = geometry._EPS
        reference = geometry.GeometryScene(np.zeros((1, 1, 1), bool), native_config.affine)
        self._frame = (native_config.tissue_mask.shape, native_config.affine.tobytes(),
                       reference._inverse.tobytes(), reference._cell_radius_mm,
                       reference._orthogonal_spacing.tobytes())
        self._namespace_record = {
            "cache_version": CAPSULE_CACHE_VERSION,
            "geometry_version": self._geometry_version, "native_version": self._native_version,
            "geometry_source_sha256": sha256(Path(geometry.__file__).read_bytes()).hexdigest(),
            "source_guard_sha256": sha256(Path(native_proposals.__file__).read_bytes()).hexdigest(),
            "mode": "exact_orthogonal_cells", "native_config_hash": native_config.fingerprint,
            "source_hash": native_config.source_hash, "geometry_epsilon": self._geometry_epsilon,
            "guarded_direct_geometry_callables": tuple(name for _, name, _ in _GEOMETRY_DEPENDENCIES),
            "max_payload_bytes": max_payload_bytes, "max_entries": max_entries,
        }
        self._namespace = "sha256:" + sha256(json.dumps(self._namespace_record, sort_keys=True).encode()).hexdigest()
        self._cache: OrderedDict[tuple, tuple[np.ndarray, tuple, int]] = OrderedDict()
        self._payload = 0
        self._query_active = False
        self._calls = self._hits = self._misses = self._evictions = self._bypasses = 0
        self._query_seconds = self._compute_seconds = 0.

    @property
    def namespace(self) -> str:
        return self._namespace

    def _assert_source(self) -> None:
        if threading.get_ident() != self._owner_thread:
            raise RuntimeError("Capsule cache belongs to its creating worker thread")
        if (_source_identity(self._config) != self._source_identity
                or geometry.GEOMETRY_VERSION != self._geometry_version
                or native_resection.NATIVE_RESECTION_VERSION != self._native_version
                or geometry._EPS != self._geometry_epsilon
                or geometry.capsule_voxel_indices is not _ORIGINAL_COVER
                or any(getattr(owner, name) is not expected for owner, name, expected in _GEOMETRY_DEPENDENCIES)):
            raise RuntimeError("Frozen capsule cache source or geometry implementation changed")

    def _assert_scene(self, scene: geometry.GeometryScene) -> None:
        if not isinstance(scene, geometry.GeometryScene):
            raise TypeError("Capsule coverage requires a GeometryScene")
        for owner, name, expected in _GEOMETRY_DEPENDENCIES:
            if owner is geometry.GeometryScene:
                method = getattr(scene, name)
                if getattr(method, "__self__", None) is not scene or getattr(method, "__func__", None) is not expected:
                    raise RuntimeError("Capsule scene coordinate conversion changed")
        spacing = scene._orthogonal_spacing
        if spacing is None:
            raise RuntimeError("Capsule cache mode differs from validated orthogonal source")
        for array in (scene.affine, scene._inverse, spacing, scene.forbidden_mask):
            _array_identity(array)
        if (scene.affine.shape != (4, 4) or scene.affine.dtype != np.float64
                or scene._inverse.shape != (4, 4) or scene._inverse.dtype != np.float64
                or spacing.shape != (3,) or spacing.dtype != np.float64
                or (scene.shape, scene.affine.tobytes(), scene._inverse.tobytes(),
                    scene._cell_radius_mm, spacing.tobytes()) != self._frame):
            raise RuntimeError("Capsule scene frame or derived geometry differs from source")

    def query(self, scene: geometry.GeometryScene, start_mm: Any, end_mm: Any,
              radius_mm: float) -> np.ndarray:
        if threading.get_ident() != self._owner_thread:
            raise RuntimeError("Capsule cache belongs to its creating worker thread")
        if self._query_active:
            raise RuntimeError("Capsule cache queries cannot be reentered")
        self._query_active = True
        started = perf_counter()
        self._calls += 1
        try:
            start = geometry._vector(start_mm, "start_mm")
            end = geometry._vector(end_mm, "end_mm")
            radius = np.asarray(radius_mm)
            if radius.shape != () or radius.dtype.kind not in "iuf" or not np.isfinite(radius) or radius < 0:
                raise ValueError("radius_mm must be a finite nonnegative numeric scalar")
            # Native tool dimensions are normalized Python floats. Other scalar
            # precision modes are outside this narrowly declared prototype.
            if radius.dtype.kind == "f" and radius.dtype != np.float64:
                raise ValueError("Prototype radius precision must be float64")
            radius = float(radius)
            # Array conversion can execute caller code. Check the authoritative
            # source/frame after normalization and before a warm lookup.
            self._assert_source()
            self._assert_scene(scene)
            key = (self.namespace, self._frame, start.tobytes(), end.tobytes(),
                   np.asarray(radius, dtype=np.float64).tobytes())
            if key in self._cache:
                result, descriptor, _ = self._cache[key]
                if _array_identity(result) != descriptor:
                    raise RuntimeError("Cached capsule coverage descriptor changed")
                self._hits += 1
                self._cache.move_to_end(key)
                return result
            self._misses += 1
            computed = perf_counter()
            try:
                result = _ORIGINAL_COVER(scene, start, end, radius)
            finally:
                self._compute_seconds += perf_counter() - computed
            if (self._max_entries == 0 or self._max_payload_bytes == 0
                    or result.nbytes > self._max_payload_bytes):
                self._bypasses += 1
                return result
            while self._cache and (self._payload + result.nbytes > self._max_payload_bytes
                                   or len(self._cache) >= self._max_entries):
                self._payload -= self._cache.popitem(last=False)[1][2]
                self._evictions += 1
            self._cache[key] = (result, _array_identity(result), result.nbytes)
            self._payload += result.nbytes
            return result
        finally:
            self._query_seconds += perf_counter() - started
            self._query_active = False

    def stats(self) -> dict[str, Any]:
        return {**self._namespace_record, "namespace": self.namespace,
                "calls": self._calls, "hits": self._hits, "misses": self._misses,
                "evictions": self._evictions, "bypasses": self._bypasses,
                "entries": len(self._cache), "retained_payload_bytes": self._payload,
                "query_seconds": self._query_seconds, "compute_seconds": self._compute_seconds,
                "payload_scope": "array buffers only; keys/bookkeeping and process RSS separate",
                "feasibility_cached": False, "cavity_cached": False, "integrity_cached": False}


@contextmanager
def inject_native_capsule_cache(cache: ExactCapsuleCoverCache):
    """Explicitly inject and restore one thread's synthetic/prospective probe.

    Reject nesting or a pre-existing replacement. The process-global native
    symbol is restored even if the body raises. Other threads cannot use the
    replacement; this prototype must not coexist with other native workloads.
    """
    if not isinstance(cache, ExactCapsuleCoverCache):
        raise TypeError("An ExactCapsuleCoverCache is required")
    cache._assert_source()
    if not _INJECTION_LOCK.acquire(blocking=False):
        raise RuntimeError("A native capsule cache injection is already active")
    replacement = cache.query
    installed = False
    body_failed = False
    try:
        if native_resection.capsule_voxel_indices is not _ORIGINAL_COVER:
            raise RuntimeError("Native capsule query already has a foreign replacement")
        native_resection.capsule_voxel_indices = replacement
        installed = True
        try:
            yield cache
        except BaseException:
            body_failed = True
            raise
    finally:
        changed = installed and native_resection.capsule_voxel_indices is not replacement
        if installed:
            native_resection.capsule_voxel_indices = _ORIGINAL_COVER
        _INJECTION_LOCK.release()
        if changed and not body_failed:
            raise RuntimeError("Native capsule query changed during injection; original restored")
