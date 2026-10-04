"""Unwired, bounded batch prototype for the independent segment/box checker.

This evaluates one finite segment against ordered physical axis-aligned boxes.
It does not transform grids, inspect tissue, certify a tool, or cache anything.
The caller retains responsibility for source transforms and causal occupancy.
No planning geometry implementation is imported.
"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np


VERSION = "independent-piecewise-segment-box-batch-prototype-v1"
DEFAULT_BATCH_SIZE = 256
MAX_BATCH_SIZE = 4096
Cancel = Callable[[], bool] | None


class IndependentBatchCancelled(RuntimeError):
    """No completed distance/contact result is available after cancellation."""


def _check_cancelled(cancelled: Cancel) -> None:
    if cancelled is not None and cancelled():
        raise IndependentBatchCancelled("Independent geometry batch cancelled")


def _prepare(start, end, lower, upper, batch_size, cancelled):
    if isinstance(batch_size, (bool, np.bool_)) or not isinstance(batch_size, (int, np.integer)) or not 1 <= batch_size <= MAX_BATCH_SIZE:
        raise ValueError(f"batch_size must be an integer in [1, {MAX_BATCH_SIZE}]")
    _check_cancelled(cancelled)
    start, end = (np.array(value, dtype=np.float64, copy=True) for value in (start, end))
    if any(value.shape != (3,) or not np.isfinite(value).all() for value in (start, end)):
        raise ValueError("Segment requires finite xyz coordinates")
    # Borrow typed arrays: coercing an arbitrary list or float32 volume here
    # would allocate an unbounded full-input copy before any chunk checks.
    if any(not isinstance(value, np.ndarray) or value.dtype != np.dtype(np.float64)
           or value.ndim != 2 or value.shape[1] != 3 for value in (lower, upper)) or lower.shape != upper.shape:
        raise ValueError("Boxes require matching float64 numpy arrays shaped (N, 3)")
    batch_size = int(batch_size)
    for offset in range(0, len(lower), batch_size):
        _check_cancelled(cancelled)
        lo, hi = lower[offset:offset + batch_size], upper[offset:offset + batch_size]
        if not np.isfinite(lo).all() or not np.isfinite(hi).all() or np.any(lo > hi):
            raise ValueError("Boxes require finite ordered xyz coordinates")
    _check_cancelled(cancelled)
    return start, end, lower, upper, batch_size


def _distance_at(start, direction, lower, upper, t):
    point = start + t[:, None] * direction
    delta = np.maximum(np.maximum(lower - point, point - upper), 0.0)
    # vecdot uses the same inner-product operation as the scalar oracle's @.
    # sum/einsum can change the three-term rounding order on the tested runtime.
    return np.vecdot(delta, delta)


def _distances_chunk(start, end, lower, upper):
    """At most MAX_BATCH_SIZE boxes; constant eight face/end break slots."""
    try:
        with np.errstate(over="raise", invalid="raise", divide="raise"):
            direction = end - start
            breaks = np.zeros((len(lower), 8), dtype=np.float64)
            breaks[:, 1] = 1.0
            for axis in range(3):
                if abs(direction[axis]) > 1e-15:  # scalar oracle's exact cutoff
                    for face_index, faces in enumerate((lower, upper)):
                        t = (faces[:, axis] - start[axis]) / direction[axis]
                        breaks[:, 2 + 2 * axis + face_index] = np.where((t > 0) & (t < 1), t, 0.0)
            breaks.sort(axis=1)
            result = np.full(len(lower), np.inf)
            for column in range(8):
                result = np.minimum(result, _distance_at(start, direction, lower, upper, breaks[:, column]))
            # Repeated breaks are harmless zero-width intervals. Skip them;
            # scalar sorted(set(...)) makes the same nonempty intervals.
            for column in range(7):
                left, right = breaks[:, column], breaks[:, column + 1]
                mid = start + ((left + right) / 2)[:, None] * direction
                active = (mid < lower) | (mid > upper)
                bound = np.where(mid < lower, lower, upper)
                patterns = active[:, 0].astype(np.uint8) + 2 * active[:, 1] + 4 * active[:, 2]
                for pattern in range(1, 8):
                    rows = np.flatnonzero((patterns == pattern) & (right > left))
                    if not len(rows):
                        continue
                    axes = np.array([bool(pattern & (1 << axis)) for axis in range(3)])
                    active_direction = direction[axes]
                    a = float(active_direction @ active_direction)
                    if a > 0:
                        delta = start[axes] - bound[rows][:, axes]
                        b = np.vecdot(active_direction, delta)
                        optimum = np.clip(-b / a, left[rows], right[rows])
                        distance = _distance_at(start, direction, lower[rows], upper[rows], optimum)
                        result[rows] = np.minimum(result[rows], distance)
            if not np.isfinite(result).all():
                raise ValueError("Independent geometry arithmetic is not finite")
            return result
    except FloatingPointError as error:
        raise ValueError("Independent geometry arithmetic overflow or invalid value") from error


def segment_box_distances_sq(start, end, lower, upper, *, batch_size=DEFAULT_BATCH_SIZE,
                             cancelled: Cancel = None) -> np.ndarray:
    """Squared distances in input order; O(batch_size) work plus O(N) output.

    Inputs are borrowed synchronously and must not be changed during the call.
    Cancellation is checked around each bounded chunk, including the last one;
    no partial array is returned. Extreme finite inputs whose arithmetic
    overflows are refused. Distances are numerical, not geometry certificates.
    """
    start, end, lower, upper, batch_size = _prepare(start, end, lower, upper, batch_size, cancelled)
    result = np.empty(len(lower), dtype=np.float64)
    for offset in range(0, len(lower), batch_size):
        _check_cancelled(cancelled)
        result[offset:offset + batch_size] = _distances_chunk(start, end, lower[offset:offset + batch_size], upper[offset:offset + batch_size])
    _check_cancelled(cancelled)
    return result


def _threshold(radius_mm, tolerance_sq):
    if any(isinstance(value, (bool, np.bool_)) or not np.isscalar(value)
           or not np.isfinite(value) or value < 0 for value in (radius_mm, tolerance_sq)):
        raise ValueError("Radius and squared-distance tolerance must be finite nonnegative scalars")
    radius_mm, tolerance_sq = float(radius_mm), float(tolerance_sq)
    threshold = radius_mm * radius_mm + tolerance_sq
    if not np.isfinite(threshold):
        raise ValueError("Squared contact threshold overflows")
    return threshold


def _contact_flags(start, end, lower, upper, threshold):
    distances = _distances_chunk(start, end, lower, upper)
    # Contact decisions near the threshold remain with the preserved scalar
    # reference, including tangency and large-coordinate cancellation cases.
    # This guard is a conservative proximity rule, not a formal error proof.
    scale = np.maximum(np.maximum(np.max(np.abs(lower), axis=1), np.max(np.abs(upper), axis=1)),
                       max(float(np.max(np.abs(start))), float(np.max(np.abs(end))), 1.0))
    with np.errstate(over="ignore"):
        guard = 64 * np.finfo(np.float64).eps * scale**2
    near = np.flatnonzero(np.abs(distances - threshold) <= guard)
    if len(near):
        from resectionlab.evaluation import segment_box_distance_sq
        for row in near:
            distances[row] = segment_box_distance_sq(start, end, lower[row], upper[row])
    return distances <= threshold


def segment_box_contact_indices(start, end, lower, upper, radius_mm, *, tolerance_sq=1e-10,
                                batch_size=DEFAULT_BATCH_SIZE, cancelled: Cancel = None) -> np.ndarray:
    """Ordered contacting row indices; caller must specify its original tolerance.

    Work storage is bounded; returned indices occupy O(number of contacts).
    No occupancy or feasibility is cached, and no partial result is returned.
    """
    threshold = _threshold(radius_mm, tolerance_sq)
    start, end, lower, upper, batch_size = _prepare(start, end, lower, upper, batch_size, cancelled)
    matches = []
    for offset in range(0, len(lower), batch_size):
        _check_cancelled(cancelled)
        flags = _contact_flags(start, end, lower[offset:offset + batch_size], upper[offset:offset + batch_size], threshold)
        indices = np.flatnonzero(flags)
        if len(indices):
            matches.append(indices + offset)
    _check_cancelled(cancelled)
    return np.concatenate(matches) if matches else np.empty(0, dtype=np.int64)


def first_segment_box_contact(start, end, lower, upper, radius_mm, *, tolerance_sq=1e-10,
                               batch_size=DEFAULT_BATCH_SIZE, cancelled: Cancel = None) -> int | None:
    """First contacting input row, preserving e.g. np.argwhere voxel order.

    All input boxes are validated even if the first row collides. The function
    returns an index, not a source coordinate or a complete-tool certificate.
    """
    threshold = _threshold(radius_mm, tolerance_sq)
    start, end, lower, upper, batch_size = _prepare(start, end, lower, upper, batch_size, cancelled)
    for offset in range(0, len(lower), batch_size):
        _check_cancelled(cancelled)
        flags = _contact_flags(start, end, lower[offset:offset + batch_size], upper[offset:offset + batch_size], threshold)
        _check_cancelled(cancelled)
        rows = np.flatnonzero(flags)
        if len(rows):
            return offset + int(rows[0])
    _check_cancelled(cancelled)
    return None
