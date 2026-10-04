"""Synthetic feature proposal, deliberately outside the production package.

No policy, training, patient loader or registered input-profile integration.
Coordinates use an explicitly declared access/source basis; changing source-axis
ordering changes that basis. Volume and centroid summaries are lossy.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json

import numpy as np


VERSION = "synthetic-access-coordinate-centroid-probe-v1"
ACTION_NAMES = ("entry_u_mm", "entry_v_mm", "entry_inward_mm",
                "tip_u_mm", "tip_v_mm", "tip_inward_mm")
STATE_NAMES = ("cavity_volume_mm3", "cavity_centroid_u_mm", "cavity_centroid_v_mm",
               "cavity_centroid_inward_mm", "residual_target_volume_mm3",
               "residual_target_centroid_u_mm", "residual_target_centroid_v_mm",
               "residual_target_centroid_inward_mm")


def _array(value, shape=None):
    result = np.asarray(value, dtype=np.float64)
    if (shape is not None and result.shape != shape) or not np.isfinite(result).all():
        raise ValueError("Finite physical coordinates with the declared shape required")
    return result


def _affine(value):
    result = _array(value, (4, 4))
    if not np.allclose(result[3], (0, 0, 0, 1), rtol=0, atol=1e-12):
        raise ValueError("Expected homogeneous physical affine")
    if abs(np.linalg.det(result[:3, :3])) <= 1e-12:
        raise ValueError("Source affine must be invertible")
    return result


@dataclass(frozen=True)
class AccessSourceFrame:
    """Right-handed basis: inward normal and first usable declared source axis."""

    origin_mm: tuple[float, float, float]
    basis_columns: tuple[tuple[float, float, float], ...]
    source_tangent_axis: int

    def project(self, points):
        array = _array(points)
        if array.shape[-1:] != (3,):
            raise ValueError("Points must end in three physical coordinates")
        return (array - self.origin_mm) @ np.asarray(self.basis_columns)

    def to_dict(self):
        return {"version": VERSION, "origin_mm": list(self.origin_mm),
                "basis_columns": [list(row) for row in self.basis_columns],
                "source_tangent_axis": self.source_tangent_axis,
                "orientation": "right_handed", "units": "millimetres"}


def access_source_frame(affine, center_mm, normal_inward) -> AccessSourceFrame:
    affine = _affine(affine)
    center = _array(center_mm, (3,))
    inward = _array(normal_inward, (3,))
    norm = np.linalg.norm(inward)
    if norm <= 1e-12:
        raise ValueError("Inward normal must be nonzero")
    inward = inward / norm
    for axis in range(3):
        source = affine[:3, axis] / np.linalg.norm(affine[:3, axis])
        tangent = source - inward * np.dot(source, inward)
        if np.linalg.norm(tangent) > 1e-8:
            tangent = tangent / np.linalg.norm(tangent)
            second = np.cross(inward, tangent)
            basis = np.column_stack((tangent, second, inward))
            return AccessSourceFrame(tuple(center), tuple(tuple(row) for row in basis), axis)
    raise ValueError("No usable source tangent axis")


def candidate_coordinates(action_ids, entries_mm, tips_mm, frame: AccessSourceFrame):
    """Return six physical coordinates in supplied row order; STOP has zero row.

    This is a pure descriptor. It neither certifies an action nor changes a mask.
    Existing STOP/tool/reward features are not replaced by these six coordinates.
    """
    ids = tuple(action_ids)
    if len(set(ids)) != len(ids) or not ids:
        raise ValueError("Distinct nonempty action IDs required")
    entries, tips = _array(entries_mm, (len(ids), 3)), _array(tips_mm, (len(ids), 3))
    result = np.column_stack((frame.project(entries), frame.project(tips)))
    for index, action in enumerate(ids):
        if action == "STOP":
            result[index] = 0
    return result


def cavity_residual_summary(tissue, target, removed, affine, frame: AccessSourceFrame):
    """Eight observable source-cell volume/centroid values; no hidden anatomy.

    Empty regions use centroid zero and volume zero. These are physical units,
    not normalized inputs or a claim of sufficient state for a value function.
    """
    tissue, target, removed = (np.asarray(x) for x in (tissue, target, removed))
    if (tissue.ndim != 3 or target.shape != tissue.shape or removed.shape != tissue.shape
            or any(x.dtype != np.bool_ for x in (tissue, target, removed))):
        raise ValueError("Matching three-dimensional Boolean masks required")
    if np.any(target & ~tissue) or np.any(removed & ~tissue):
        raise ValueError("Target and actual removal must lie in declared tissue")
    affine = _affine(affine)
    voxel_volume = float(abs(np.linalg.det(affine[:3, :3])))
    parts = []
    for mask in (removed, target & ~removed):
        indices = np.argwhere(mask)
        volume = len(indices) * voxel_volume
        centroid = (frame.project(indices.mean(axis=0) @ affine[:3, :3].T + affine[:3, 3])
                    if len(indices) else np.zeros(3))
        parts.extend((volume, *centroid))
    return np.asarray(parts, dtype=np.float64)


def descriptor_contract():
    payload = {"version": VERSION, "action_names": ACTION_NAMES, "state_names": STATE_NAMES,
        "frame": "access center; inward normal; first declared source axis with tangent norm>1e-8; right-handed cross product",
        "voxel_location": "source voxel centers via supplied affine",
        "normalization": "none; physical mm and mm3", "empty_centroid": [0, 0, 0],
        "stop_coordinates": [0, 0, 0, 0, 0, 0],
        "production_input_profiles_changed": False, "markov_completeness_claimed": False}
    payload["descriptor_hash"] = "sha256:" + hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    return payload
