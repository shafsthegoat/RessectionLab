"""Opt-in C++ kernels for stroke connectivity and the surface audit.

``stroke`` (exact): ``native_resection._connected_surface_cells`` and
``_extend_connected_free`` as compiled flood fills. Same accepted cells, same
LIFO mutation order (including the partial update before an unknown-domain
refusal) and the same error message.

``surface-audit`` (exact): ``surface_fairing._intersection_vertices`` with a
uniform-grid broad phase in place of the k-d tree, the same distance bound,
filters and separating-axis test. The kernel reproduces NumPy's own
``einsum`` summation order, which this module measures once per process; if no
supported order matches, or a pair sits on the k-d tree's rounding boundary,
the reference audit runs instead.

Every function returns ``None`` (or ``False``) when the extension is missing
or an input is outside what the kernels accept; callers then run their
reference code unchanged.
"""
from __future__ import annotations

import numpy as np

from . import _accel_loader

_ORDERS = (0, 1, 2)
_order = None


def _native():
    return _accel_loader.module()


def _probe_triangles(rng, count):
    """Near-contact, coplanar and widely scaled pairs for the SAT self-check."""
    first = rng.standard_normal((count, 3, 3))
    shift = rng.standard_normal((count, 1, 3)) * rng.choice([1e-9, 1e-3, .5, 2.], (count, 1, 1))
    second = first[:, rng.permutation(3)] + shift + rng.standard_normal((count, 3, 3)) * .3
    flat = rng.random(count) < .25
    first[flat, :, 2] = 0.
    second[flat, :, 2] = 0.
    scale = 10. ** rng.integers(-3, 4, (count, 1, 1))
    return first * scale, second * scale


def _measure():
    native = _native()
    if native is None:
        return False
    from .surface_fairing import intersecting
    rng = np.random.default_rng(20261126)
    for count in (1, 3, 8, 515, 4096):
        points = rng.standard_normal((count, 3, 3)) * 10. ** rng.integers(-8, 9, (count, 3, 3))
        axes = rng.standard_normal((count, 3)) * 10. ** rng.integers(-8, 9, (count, 3))
        if not np.array_equal(native.norms(axes), np.linalg.norm(axes, axis=1)):
            return False
    expected = np.einsum('nki,ni->nk', points, axes)
    matches = [order for order in _ORDERS if np.array_equal(native.projections(points, axes, order), expected)]
    if len(matches) != 1:
        return False
    order = matches[0]
    for count in (1, 2, 7, 64, 4096):
        sample = np.ascontiguousarray(rng.standard_normal((count, 3, 3)))
        axis = np.cross(sample[:, 1] - sample[:, 0], sample[:, 2] - sample[:, 0])
        if not np.array_equal(native.projections(sample, axis, order), np.einsum('nki,ni->nk', sample, axis)):
            return False
        first, second = _probe_triangles(rng, count)
        if not np.array_equal(native.intersecting(first, second, order), intersecting(first, second)):
            return False
    return order


def projection_order():
    """The kernel summation order matching this NumPy, or None to use NumPy."""
    global _order
    if _order is None:
        _order = _measure()
    return None if _order is False else _order


# --- stroke ------------------------------------------------------------------

def _cells(cells, shape):
    cells = np.asarray(cells)
    if cells.ndim != 2 or cells.shape[1:] != (3,) or cells.dtype.kind not in 'iu':
        return None
    if len(cells) and (np.any(cells < 0) or np.any(cells >= np.asarray(shape))):
        return None
    return np.ascontiguousarray(cells, dtype=np.int64)


def _mask(mask, shape=None):
    if not isinstance(mask, np.ndarray) or mask.dtype != np.bool_ or mask.ndim != 3:
        return None
    if shape is not None and mask.shape != shape:
        return None
    return np.ascontiguousarray(mask)


def connected_surface_cells(candidates, connected_free):
    """Sorted accepted cells as an int64 array, or None for the reference."""
    native = _native()
    if native is None:
        return None
    free = _mask(connected_free)
    cells = None if free is None else _cells(candidates, free.shape)
    if cells is None or not len(cells):
        return None
    return native.connected_surface_cells(cells, free)


def extend_connected_free(new_cells, remaining, connected_free, interaction_domain=None):
    """Flood ``connected_free`` in place; False means the reference must run."""
    native = _native()
    if (native is None or not isinstance(connected_free, np.ndarray) or connected_free.dtype != np.bool_
            or connected_free.ndim != 3 or not connected_free.flags.c_contiguous
            or not connected_free.flags.writeable):
        return False
    shape = connected_free.shape
    rest = _mask(remaining, shape)
    domain = None if interaction_domain is None else _mask(interaction_domain, shape)
    cells = _cells(new_cells, shape)
    if rest is None or cells is None or (interaction_domain is not None and domain is None):
        return False
    native.extend_connected_free(cells, rest, connected_free, domain)
    return True


# --- surface-audit -------------------------------------------------------------

def intersection_vertices(source, candidate, faces, cancelled, changed_points=None):
    """``surface_fairing._intersection_vertices`` in C++, or None for the reference."""
    native = _native()
    if native is None or projection_order() is None:
        return None
    source, candidate, faces = np.asarray(source), np.asarray(candidate), np.asarray(faces)
    if (source.dtype != np.float64 or candidate.dtype != np.float64 or faces.dtype.kind not in 'iu'
            or source.ndim != 2 or source.shape[1:] != (3,) or candidate.shape != source.shape
            or faces.ndim != 2 or faces.shape[1:] != (3,) or not len(faces)
            or np.any(faces < 0) or np.any(faces >= len(source))
            or not np.isfinite(source).all() or not np.isfinite(candidate).all()):
        return None
    # The reference's own masks and early exits, verbatim.
    moved = np.any(source != candidate, axis=1)[faces].any(axis=1)
    if not moved.any():
        return np.empty(0, int)
    dirty = np.ones(len(faces), bool) if changed_points is None else changed_points[faces].any(axis=1)
    if not dirty.any():
        return np.empty(0, int)
    try:
        found, ambiguous = native.intersection_vertices(
            np.ascontiguousarray(source), np.ascontiguousarray(candidate),
            np.ascontiguousarray(faces, dtype=np.int64), np.ascontiguousarray(moved),
            np.ascontiguousarray(dirty, dtype=bool), cancelled, projection_order())
    except OverflowError:
        return None
    return None if ambiguous else found.astype(int, copy=False)
