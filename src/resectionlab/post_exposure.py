"""One explicit axis0 post-exposure simulation start; source coverage is unchanged.

This models an unassessed external workspace and a declared accessible zero
component, not measured air or surgical removal. The strict default uses none
of this module's state. No patient loading, route ranking or fallback occurs.
"""
from dataclasses import dataclass, field
from typing import Mapping
import numpy as np
from scipy.ndimage import binary_propagation, generate_binary_structure
from .core import array_digest, freeze_json, immutable_array, semantic_digest
from .geometry import AccessWindow

VERSION = "public-union-post-exposure-axis0-v1"
EPS_MM = 1e-8
_PREPARED = object()


def _layout(value):
    root = value
    while isinstance(root, np.ndarray) and root.base is not None:
        root = root.base
    if value.flags.writeable or not isinstance(root, bytes):
        raise ValueError("Post-exposure masks require immutable owned bytes")
    return (id(value), id(root), value.shape, value.dtype.str, value.strides)


def access_record(access):
    return {"center_mm": np.asarray(access.center_mm).tolist(),
            "normal_inward": np.asarray(access.normal_inward).tolist(),
            "radius_mm": access.radius_mm, "window_id": access.window_id}


def preentry_tip(tool, entry, axis, normal):
    """Actual two-capsule forward extent; no cell standoff or safety layer."""
    axis, normal = np.asarray(axis, float), np.asarray(normal, float)
    c = float(axis @ normal)
    if not np.isfinite(c) or c <= 0 or not np.isclose(np.linalg.norm(axis), 1., atol=1e-9):
        raise ValueError("Post-exposure insertion requires a finite forward unit axis")
    g = max(tool.tip_radius_mm, tool.shaft_radius_mm-tool.tip_length_mm*c)
    return np.asarray(entry, float)-((g+EPS_MM)/c)*axis


@dataclass(frozen=True)
class PostExposureStart:
    record: Mapping
    external_workspace: np.ndarray = field(repr=False)
    seed: np.ndarray = field(repr=False)
    initial_free: np.ndarray = field(repr=False)
    _capability: object = field(repr=False)
    _identity: tuple = field(init=False, repr=False)

    def __post_init__(self):
        if self._capability is not _PREPARED:
            raise ValueError("Prepare the exact public post-exposure condition")
        object.__setattr__(self, "record", freeze_json(self.record))
        for name in ("external_workspace", "seed", "initial_free"):
            object.__setattr__(self, name, immutable_array(getattr(self, name), bool))
        object.__setattr__(self, "_identity", self.identity())

    def identity(self):
        return (semantic_digest(self.record), *(_layout(getattr(self, k)) for k in
                    ("external_workspace", "seed", "initial_free")))

    def assert_intact(self):
        if self.identity() != self._identity:
            raise RuntimeError("Post-exposure condition was replaced")

    @property
    def fingerprint(self):
        self.assert_intact()
        return semantic_digest(self.record)

    def require_bound(self, occupancy, domain, affine, access):
        self.assert_intact()
        if (domain is None or self.record["occupancy_hash"] != array_digest(np.asarray(occupancy, bool))
                or self.record["interaction_domain_hash"] != array_digest(np.asarray(domain, bool))
                or self.record["native_affine_hash"] != array_digest(np.asarray(affine, float))
                or self.record["access"] != freeze_json(access_record(access))):
            raise ValueError("Post-exposure condition differs from immutable native world")


def prepare_post_exposure(*, support, target, support_domain, affine, previous_access):
    """Deterministic local full-cell union boundary; return access and start.

    The historical side and transverse location are retained. A failed seed or
    out-of-image boundary is final for this rule, never a search for another one.
    """
    S, T, Ds = (np.asarray(v) for v in (support, target, support_domain))
    if (S.ndim != 3 or min(S.shape) < 3 or T.shape != S.shape or Ds.shape != S.shape
            or any(v.dtype.kind not in 'biuf' or not np.isfinite(v).all()
                   or not np.isin(v, (0, 1)).all() for v in (S, T, Ds))):
        raise ValueError("Exact binary S/T/Ds source grids are required")
    S, T, Ds = (v.astype(bool, copy=False) for v in (S, T, Ds))
    if not S.any() or not T.any() or np.any(S & ~Ds):
        raise ValueError("Post-exposure needs nonempty unchanged source S/T and S coverage")
    O, D = S | T, Ds | T
    affine = np.asarray(affine, float)
    if affine.shape != (4, 4) or not np.isfinite(affine).all():
        raise ValueError("Finite native affine required")
    basis = affine[:3, :3]
    spacing = np.linalg.norm(basis, axis=0)
    unit = basis/spacing
    if not np.allclose(unit.T@unit, np.eye(3), rtol=0, atol=1e-8):
        raise ValueError("Post-exposure requires the reconciled orthogonal native grid")
    orientation = unit.T@previous_access.normal_inward
    if (not np.isclose(abs(orientation[0]), 1., rtol=0, atol=1e-8)
            or np.any(np.abs(orientation[1:]) > 1e-8)):
        raise ValueError("Post-exposure rule is fixed to source-axis0")
    sign = 1 if orientation[0] > 0 else -1
    origin = np.linalg.solve(basis, previous_access.center_mm-affine[:3, 3])
    transverse = np.rint(origin[1:])
    if not np.allclose(origin[1:], transverse, rtol=0, atol=1e-7):
        raise ValueError("Retained transverse aperture location must be on source columns")
    if previous_access.radius_mm != 6.:
        raise ValueError("This prospective condition retains the fixed6mm aperture")
    yy, zz = np.ogrid[:O.shape[1], :O.shape[2]]
    dy = np.abs((yy-transverse[0])*spacing[1])
    dz = np.abs((zz-transverse[1])*spacing[2])
    near2 = np.maximum(dy-spacing[1]/2, 0.)**2 + np.maximum(dz-spacing[2]/2, 0.)**2
    footprint = near2 <= (previous_access.radius_mm+EPS_MM)**2
    occupied_columns = np.flatnonzero(np.any(O & footprint[None, :, :], axis=(1, 2)))
    if not len(occupied_columns):
        raise ValueError("POST_EXPOSURE_EMPTY_FIXED_OCCUPANCY_DISC")
    face = float(occupied_columns[0]-.5 if sign > 0 else occupied_columns[-1]+.5)
    new_origin = np.array((face, *transverse), float)
    if np.any(new_origin < -.5) or np.any(new_origin > np.array(O.shape)-.5):
        raise ValueError("POST_EXPOSURE_APERTURE_OUTSIDE_IMAGE")
    access = AccessWindow(basis@new_origin+affine[:3, 3], sign*unit[:, 0],
                          previous_access.radius_mm, VERSION)
    depth = sign*(np.arange(O.shape[0])-face)*spacing[0]
    lo, hi = depth-spacing[0]/2, depth+spacing[0]/2
    E = (~D) & (hi[:, None, None] < -EPS_MM)
    far2 = (dy+spacing[1]/2)**2+(dz+spacing[2]/2)**2
    inside_disc = far2 < (access.radius_mm-EPS_MM)**2
    proximal_face = (lo < -EPS_MM) & (np.abs(hi) <= EPS_MM)
    K = Ds & ~O & proximal_face[:, None, None] & inside_disc[None, :, :]
    if not K.any():
        raise ValueError("POST_EXPOSURE_EMPTY_FIXED_K")
    free_mask = Ds & ~O
    F0 = binary_propagation(K, structure=generate_binary_structure(3, 1), mask=free_mask)
    assert not np.any(F0 & (O | ~Ds | E))
    record = {
        "version": VERSION, "original_access": access_record(previous_access), "access": access_record(access),
        "selection": "same_side_and_transverse_centroid; outer_full_O_cell_face_across_closed_disc; no fallback",
        "source_S_hash": array_digest(S), "full_T_hash": array_digest(T), "source_Ds_hash": array_digest(Ds),
        "occupancy_hash": array_digest(O), "interaction_domain_hash": array_digest(D),
        "native_affine_hash": array_digest(affine), "source_shape": list(O.shape),
        "external_workspace_hash": array_digest(E), "seed_hash": array_digest(K),
        "external_encounter_hash_rule": "semantic_digest(sorted_unique_native_index_lists)",
        "initial_free_hash": array_digest(F0), "external_unknown_cells": int(E.sum()),
        "seed_cells": int(K.sum()), "initial_free_cells": int(F0.sum()),
        "fixed_disc_occupied_cells": int(np.count_nonzero(O & footprint[None, :, :])),
        "cell_face_source_index": face, "classification_tolerance_mm": EPS_MM,
        "external_workspace_rule": "E=U intersect full_cell_hi_less_than_minus1e-8mm; collision_exception_only",
        "seed_rule": "Ds&~O; proximal distal_face_on_plane; entire footprint strictly_inside_disc",
        "initial_free_rule": "six_connected_closure_of_K_in_Ds_and_not_O; zero_removal_or_reward",
        "preentry_rule": "per_insertion_p0=entry-((max(rtip,rshaft-tip_length*cos_angle)+1e-8)/cos_angle)*axis",
        "trajectory": "checked_axial_insertion_and_reverse; transfer_between_withdrawn_poses_unassessed",
        "source_domains_extended": False, "source_material_deleted": False, "source_target_deleted": False,
        "physical_air_or_operative_opening_observed": False,
        "outside_image_geometry": "existing_unassessed_extent; tip_outside_image_still_refused",
        "scope": "public_simulated_post_exposure_search_qualification_only",
    }
    return access, PostExposureStart(record, E, K, F0, _PREPARED)
