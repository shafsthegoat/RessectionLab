"""Source-only FOV/mask bits and unknown semantics; never admits a planner."""

from __future__ import annotations

import numpy as np


T1_FOV = np.uint8(1)
FLAIR_FOV = np.uint8(2)
ESTIMATED_MASK = np.uint8(4)
BOTH_SOURCE_FOV = np.uint8(3)
UNREVIEWED_DIAGNOSTIC_INPUT = np.uint8(7)
UNKNOWN = np.int8(-1)


def _binary3(name: str, value) -> np.ndarray:
    array = np.asarray(value)
    if (array.ndim != 3 or array.dtype.kind not in "biuf"
            or not np.isfinite(array).all() or not np.isin(array, (0, 1)).all()):
        raise ValueError(f"{name} must be a finite 3-D binary array")
    return array.astype(bool, copy=True)


def encode_support_bits(t1_fov, flair_fov, estimated_mask) -> np.ndarray:
    """Keep original scan FOV, estimated anatomy and candidate input distinct.

    The estimated mask is never expanded to source FOV. A source/mask mismatch
    fails instead of silently clipping the mask. No tumor or route input exists.
    """
    t1 = _binary3("t1_fov", t1_fov)
    flair = _binary3("flair_fov", flair_fov)
    mask = _binary3("estimated_mask", estimated_mask)
    if t1.shape != flair.shape or t1.shape != mask.shape:
        raise ValueError("source FOV and estimated mask require one exact grid")
    if np.any(mask & ~(t1 & flair)):
        raise ValueError("estimated mask extends outside at least one original scan FOV")
    bits = t1.astype(np.uint8) | (flair.astype(np.uint8) << 1) | (mask.astype(np.uint8) << 2)
    return bits


def unreviewed_input_domain(bits) -> np.ndarray:
    """Input-availability candidate, explicitly not qualified anatomy."""
    array = np.asarray(bits)
    if (array.ndim != 3 or array.dtype.kind not in "iu"
            or not np.isin(array, (0, 1, 2, 3, 7)).all()):
        raise ValueError("support bits require 3-D integer codes 0,1,2,3,7")
    return array == UNREVIEWED_DIAGNOSTIC_INPUT


def diagnostic_display_state(candidate_binary, bits, prediction_coverage) -> tuple[np.ndarray, int]:
    """Future candidate display: -1 unknown outside input domain, 0/1 inside.

    This display encoding cannot be used as ``ScanEstimate.mask``. A later
    reviewed estimate must carry separate binary ``coverage`` and abstain until
    its ROI and anatomy are qualified. The model result never chooses the ROI.
    """
    candidate = _binary3("candidate_binary", candidate_binary)
    predicted = _binary3("prediction_coverage", prediction_coverage)
    domain = unreviewed_input_domain(bits)
    if candidate.shape != domain.shape or predicted.shape != domain.shape:
        raise ValueError("candidate and support map require one exact grid")
    if np.any(predicted & ~domain):
        raise ValueError("prediction coverage extends outside unreviewed input domain")
    state = np.full(candidate.shape, UNKNOWN, dtype=np.int8)
    state[predicted] = candidate[predicted].astype(np.int8)
    return state, int(np.count_nonzero(candidate & ~predicted))
