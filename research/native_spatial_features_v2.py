"""Explicit-reference research frame; no production profile or patient policy.

The caller declares a physical tangent reference and its source/frame identity.
This module never selects a source axis. Poor conditioning fails closed; the
engineering rejection band is not a proof of universal floating-point stability.
The unchanged volume/centroid summaries remain deliberately incomplete.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json

import numpy as np

from research.native_spatial_features import (
    ACTION_NAMES, STATE_NAMES,
    candidate_coordinates as _v1_candidate_coordinates,
    cavity_residual_summary as _v1_cavity_residual_summary,
)


VERSION = "synthetic-declared-tangent-frame-v2.1"
MIN_TANGENT_SINE = 0.1
ROUNDOFF_REJECTION_HALF_WIDTH = 64 * np.finfo(np.float64).eps


def _hash(value):
    return "sha256:" + hashlib.sha256(json.dumps(value, sort_keys=True,
        separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def _real_array(value, shape=None):
    raw = np.asarray(value)
    if np.iscomplexobj(raw) or (raw.dtype.kind == "O" and
            any(isinstance(item, (complex, np.complexfloating)) for item in raw.flat)):
        raise ValueError("Physical coordinates must be real, not complex")
    result = np.asarray(raw, dtype=np.float64)
    if (shape is not None and result.shape != shape) or not np.isfinite(result).all():
        raise ValueError("Finite real coordinates with the declared shape required")
    return result


def _vector(value):
    return _real_array(value, (3,))


def _unit(value):
    vector = _vector(value)
    scale = float(np.max(np.abs(vector)))
    if scale == 0:
        raise ValueError("A declared direction cannot be zero")
    # Scaling first avoids underflow/overflow for finite, nonzero directions.
    scaled = vector / scale
    return scaled / np.linalg.norm(scaled)


@dataclass(frozen=True)
class DeclaredTangent:
    """Explicit physical direction plus caller-supplied provenance binding.

    Metadata equality is checked, not the external declaration's authenticity.
    Coordinate transformations must also transform this physical vector.
    """

    vector: tuple[float, float, float]
    source_hash: str
    coordinate_frame: str
    declaration_id: str

    def __post_init__(self):
        vector = _vector(self.vector)
        _unit(vector)
        object.__setattr__(self, "vector", tuple(float(x) for x in vector))
        for name in ("source_hash", "coordinate_frame", "declaration_id"):
            if not isinstance(getattr(self, name), str) or not getattr(self, name).strip():
                raise ValueError("Explicit tangent source, frame and declaration identity are required")

    def to_dict(self):
        return {"vector": list(self.vector), "source_hash": self.source_hash,
                "coordinate_frame": self.coordinate_frame, "declaration_id": self.declaration_id}


class FrameConditioningError(ValueError):
    def __init__(self, receipt):
        super().__init__(receipt["rejection_reason"])
        self.receipt = receipt


@dataclass(frozen=True)
class DeclaredAccessFrame:
    origin_mm: tuple[float, float, float]
    basis_columns: tuple[tuple[float, float, float], ...]
    normal_inward: tuple[float, float, float]
    reference: DeclaredTangent
    tangent_sine: float

    def __post_init__(self):
        if not isinstance(self.reference, DeclaredTangent):
            raise TypeError("An explicit DeclaredTangent is required")
        center = _vector(self.origin_mm)
        normal, expected, sine = _conditioned_basis(center, self.normal_inward, self.reference)
        supplied = _real_array(self.basis_columns, (3, 3))
        if not np.allclose(supplied, expected, rtol=0, atol=1e-12):
            raise ValueError("Frame basis differs from its declared reference and normal")
        if (isinstance(self.tangent_sine, (bool, np.bool_))
                or not np.isfinite(self.tangent_sine)
                or abs(float(self.tangent_sine)-sine) > ROUNDOFF_REJECTION_HALF_WIDTH):
            raise ValueError("Frame tangent sine differs from its declared reference")
        # Store computed immutable tuples, never caller-owned arrays or an
        # approximately matching supplied matrix with altered conditioning.
        object.__setattr__(self, "origin_mm", tuple(float(x) for x in center))
        object.__setattr__(self, "normal_inward", tuple(float(x) for x in normal))
        object.__setattr__(self, "basis_columns", tuple(tuple(float(x) for x in row) for row in expected))
        object.__setattr__(self, "tangent_sine", sine)

    def project(self, points):
        values = _real_array(points)
        if values.shape[-1:] != (3,):
            raise ValueError("Finite three-component physical points required")
        with np.errstate(over="ignore", invalid="ignore"):
            projected = (values - self.origin_mm) @ np.asarray(self.basis_columns)
        if not np.isfinite(projected).all():
            raise ValueError("Physical coordinate projection produced nonfinite output")
        return projected

    def to_dict(self):
        return {"version": VERSION, "origin_mm": list(self.origin_mm),
            "normal_inward": list(self.normal_inward),
            "basis_columns": [list(row) for row in self.basis_columns],
            "orientation": "right_handed", "units": "millimetres",
            "reference": self.reference.to_dict(),
            "reference_hash": _hash(self.reference.to_dict()),
            "conditioning": {"status": "accepted", "tangent_sine": self.tangent_sine,
                "tangent_normalization_factor": 1 / self.tangent_sine,
                "minimum_sine": MIN_TANGENT_SINE,
                "roundoff_rejection_half_width": float(ROUNDOFF_REJECTION_HALF_WIDTH),
                "automatic_axis_selection": False, "engineering_condition_only": True}}


def _conditioned_basis(center, normal_inward, reference):
    normal, tangent = _unit(normal_inward), _unit(reference.vector)
    cross = np.cross(normal, tangent)
    sine = float(np.linalg.norm(cross))
    receipt = {"version": VERSION, "status": "rejected", "reference": reference.to_dict(),
        "reference_hash": _hash(reference.to_dict()), "origin_mm": center.tolist(),
        "normal_inward": normal.tolist(), "tangent_sine": sine,
        "minimum_sine": MIN_TANGENT_SINE,
        "roundoff_rejection_half_width": float(ROUNDOFF_REJECTION_HALF_WIDTH),
        "automatic_axis_selection": False, "engineering_condition_only": True}
    if sine < MIN_TANGENT_SINE - ROUNDOFF_REJECTION_HALF_WIDTH:
        receipt["rejection_reason"] = "declared_tangent_below_engineering_condition_limit"
        raise FrameConditioningError(receipt)
    if sine <= MIN_TANGENT_SINE + ROUNDOFF_REJECTION_HALF_WIDTH:
        receipt["rejection_reason"] = "declared_tangent_in_roundoff_rejection_band"
        raise FrameConditioningError(receipt)
    second = cross / sine
    first = _unit(np.cross(second, normal))
    basis = np.column_stack((first, second, normal))
    return normal, basis, sine


def declared_access_frame(center_mm, normal_inward, *, reference: DeclaredTangent,
                          expected_source_hash: str, coordinate_frame: str):
    """Construct one basis or reject; never infer/fallback to another reference.

    The caller must supply a reference and the same source/frame binding as the
    access geometry. The cross-product sine must exceed 0.1 + 64*float64 epsilon.
    The rejection band is an explicit conservative engineering margin, not a
    rigorous error interval or a clinical angle constraint.
    """
    if not isinstance(reference, DeclaredTangent):
        raise TypeError("An explicit DeclaredTangent is required")
    if reference.source_hash != expected_source_hash or reference.coordinate_frame != coordinate_frame:
        raise ValueError("Declared tangent source/frame differs from access geometry")
    center = _vector(center_mm)
    normal, basis, sine = _conditioned_basis(center, normal_inward, reference)
    return DeclaredAccessFrame(tuple(float(x) for x in center),
        tuple(tuple(float(x) for x in row) for row in basis),
        tuple(float(x) for x in _vector(normal_inward)), reference, sine)


def candidate_coordinates(action_ids, entries_mm, tips_mm, frame: DeclaredAccessFrame):
    if not isinstance(frame, DeclaredAccessFrame):
        raise TypeError("V2 coordinates require a validated declared frame")
    ids = tuple(action_ids)
    entries = _real_array(entries_mm, (len(ids), 3))
    tips = _real_array(tips_mm, (len(ids), 3))
    result = _v1_candidate_coordinates(ids, entries, tips, frame)
    if not np.isfinite(result).all():
        raise ValueError("Candidate coordinates produced nonfinite output")
    return result


def cavity_residual_summary(tissue, target, removed, affine, frame: DeclaredAccessFrame):
    if not isinstance(frame, DeclaredAccessFrame):
        raise TypeError("V2 summaries require a validated declared frame")
    affine = _real_array(affine, (4, 4))
    with np.errstate(over="ignore", invalid="ignore", under="ignore"):
        voxel_volume = float(abs(np.linalg.det(affine[:3, :3])))
    if not np.isfinite(voxel_volume) or voxel_volume <= 0:
        raise ValueError("Source voxel volume must be finite and positive")
    with np.errstate(over="ignore", invalid="ignore", under="ignore"):
        result = _v1_cavity_residual_summary(tissue, target, removed, affine, frame)
    if not np.isfinite(result).all():
        raise ValueError("Physical state summary produced nonfinite output")
    return result


def descriptor_contract():
    payload = {"version": VERSION, "action_names": ACTION_NAMES, "state_names": STATE_NAMES,
        "reference": "externally declared physical vector, source hash, coordinate frame and declaration ID",
        "frame": "access center; inward normal; explicit reference cross products; no source-axis selection",
        "minimum_sine": MIN_TANGENT_SINE,
        "roundoff_rejection_half_width": float(ROUNDOFF_REJECTION_HALF_WIDTH),
        "conditioning_interpretation": "engineering rejection margin, not a global floating-point bound or clinical threshold",
        "normalization": "none; candidate mm and state mm3/mm",
        "shared_summary_version": "synthetic-access-coordinate-centroid-probe-v1",
        "validation": "immutable reference-consistent frame; real finite inputs and outputs; finite source-cell volume",
        "omitted_state": ["partial_contact_history", "current_tool", "remaining_action_budget",
                          "full_geometry", "topology", "candidate_interactions"],
        "global_cross_patient_tangent_convention_claimed": False,
        "production_input_profiles_changed": False, "markov_completeness_claimed": False}
    payload["descriptor_hash"] = _hash(payload)
    return payload
