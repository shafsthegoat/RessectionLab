"""Local scan-conditioned task over the unchanged native complete-tool engine.

Source-grid geometry stays authoritative while the actor sees a fixed crop
chosen from the scan frame and declared access. No MRI intensity threshold is a
target estimator. Contact is reported, with zero history-dependent cost because
the frozen six-channel policy does not yet observe prior contact history.
"""
from __future__ import annotations

import copy
import math
import time
from itertools import product
from dataclasses import asdict, dataclass, field
from types import SimpleNamespace
from typing import Callable
from collections.abc import Mapping

import numpy as np

from .core import array_digest, immutable_array, semantic_digest, freeze_json, thaw_json
from .geometry import AccessWindow, ToolGeometry
from .native_resection import NativeResectionConfig, NativeResectionEngine
from .native_proposals import (NominalCavityProposalConfig, PreparedNominalCavityProposer,
    NOMINAL_CAVITY_PROPOSAL_VERSION, NOMINAL_CAVITY_FAMILIES)
from .simulation import InvalidActionError, RewardSpec
from .spatial_observations import (ObservedChannel, ObservedProcedureState,
    SpatialAction, SpatialInputs, build_spatial_observation)

NATIVE_SPATIAL_VERSION = "native-spatial-observed-openings-v1"
MAX_PRIMITIVES = 96  # Below the unchanged engine's 128-certificate capacity.
GRID_ROUNDOFF_MAX_DISPLACEMENT_MM = 1e-6
GRID_ROUNDOFF_MAX_GRAM_ERROR = 1e-8
SYNTHETIC_TARGET_THRESHOLD = .5
DEFAULT_NATIVE_SPATIAL_REWARD = RewardSpec(target_per_mm3=1., normal_per_mm3=.2,
    motor_per_mm3=0., language_per_mm3=0., action_cost=.03,
    motion_per_mm=.001, tool_change_cost=.03, graph_edge_cost=0.)
OPENING_TOOLS = (
    ToolGeometry("short-wide-opener", 2.25, .45, 2.2, 35., .75),
    ToolGeometry("long-narrow-cutter", .9, 1.1, 12., 35., 3.),
)


def _access_record(access):
    return {"center_mm": np.asarray(access.center_mm).tolist(),
            "normal_inward": np.asarray(access.normal_inward).tolist(),
            "radius_mm": access.radius_mm, "window_id": access.window_id}


def _binary(value, shape, name):
    array = np.asarray(value)
    if (array.shape != shape or array.dtype.kind not in "biuf" or not np.isfinite(array).all()
            or not np.isin(array, (0, 1)).all()):
        raise ValueError(name + " must be a source-aligned binary grid")
    return immutable_array(array, bool)


def _fraction(value, shape, name):
    array = np.asarray(value)
    if (array.shape != shape or array.dtype.kind not in "biuf" or not np.isfinite(array).all()
            or np.any(array < 0) or np.any(array > 1)):
        raise ValueError(name + " must be source-aligned finite target membership in [0,1]")
    return immutable_array(array, np.float32)


def _array_identity(array):
    root = array
    while isinstance(root, np.ndarray) and root.base is not None:
        root = root.base
    if array.flags.writeable or not isinstance(root, bytes):
        raise ValueError("Native spatial source arrays require immutable owned bytes")
    return (id(array), id(root), array.shape, array.strides, array.dtype.str)


def _native_identity(config):
    return (id(config), tuple(_array_identity(getattr(config, key)) for key in
            ("tissue_mask", "target_labels", "affine", "hard_exclusion")),
        semantic_digest({"access": _access_record(config.access), "tools": [asdict(t) for t in config.tools],
            "source_hash": config.source_hash, "case_id": config.case_id, "step": config.max_tip_step_mm,
            "max_microsteps": config.max_microsteps, "support": config.tissue_support_provenance,
            "fingerprint": config.fingerprint}))


def reconcile_native_grid_roundoff(affine, shape):
    """Return a declared orthogonal grid only within a fixed roundoff budget.

    No sampling occurs. The eight full source-cell extent corners bound the
    affine displacement over the complete volume, including voxel boundaries.
    """
    original = np.asarray(affine)
    if (original.shape != (4, 4) or original.dtype.kind not in "iuf"
            or not np.isfinite(original).all() or not np.array_equal(original[3], [0, 0, 0, 1])):
        raise ValueError("GRID_ROUNDOFF_INVALID_AFFINE: a real finite canonical homogeneous matrix is required")
    shape = tuple(shape)
    if len(shape) != 3 or any(type(n) is not int or n < 1 for n in shape):
        raise ValueError("GRID_ROUNDOFF_INVALID_SHAPE: three positive integer dimensions are required")
    original = np.array(original, dtype=np.float64, copy=True)
    spacing = np.linalg.norm(original[:3, :3], axis=0)
    if not np.isfinite(spacing).all() or np.any(spacing <= 0):
        raise ValueError("GRID_ROUNDOFF_INVALID_SCALE: positive finite source-axis lengths are required")
    directions = original[:3, :3] / spacing
    gram_error = float(np.abs(directions.T @ directions - np.eye(3)).max())
    if gram_error > GRID_ROUNDOFF_MAX_GRAM_ERROR:
        raise ValueError("GRID_ROUNDOFF_MEANINGFUL_SHEAR: normalized source axes exceed the fixed roundoff bound")
    u, _, vt = np.linalg.svd(directions)
    derived = original.copy()
    derived[:3, :3] = (u @ vt) * spacing
    derived_spacing = np.linalg.norm(derived[:3, :3], axis=0)
    if (np.sign(np.linalg.det(derived[:3, :3])) != np.sign(np.linalg.det(original[:3, :3]))
            or not np.allclose(derived_spacing, spacing, rtol=64 * np.finfo(float).eps, atol=0)
            or not np.array_equal(derived[:3, 3], original[:3, 3])):
        raise ValueError("GRID_ROUNDOFF_FRAME_CHANGE: handedness, axis lengths and source origin must be preserved")
    corners = np.array(list(product(*[(-.5, n - .5) for n in shape])), dtype=np.float64)
    displacements = corners @ (derived[:3, :3] - original[:3, :3]).T
    maximum = float(np.linalg.norm(displacements, axis=1).max())
    if not np.isfinite(maximum) or maximum > GRID_ROUNDOFF_MAX_DISPLACEMENT_MM:
        raise ValueError("GRID_ROUNDOFF_DISPLACEMENT_EXCEEDED: full source cells move more than 1e-6 mm")
    record = {"method": "orthogonal_roundoff_1e-6mm", "shape": list(shape),
        "original_affine_ras_mm": original.tolist(), "derived_affine_ras_mm": derived.tolist(),
        "original_affine_hash": array_digest(original), "derived_affine_hash": array_digest(derived),
        "original_spacing_mm": spacing.tolist(), "derived_spacing_mm": derived_spacing.tolist(),
        "origin_preserved": True, "handedness_preserved": True,
        "source_axis_gram_max_error": gram_error, "maximum_allowed_gram_error": GRID_ROUNDOFF_MAX_GRAM_ERROR,
        "corner_domain": "full_source_cell_extent_minus_half_to_shape_minus_half",
        "maximum_corner_displacement_mm": maximum,
        "maximum_allowed_corner_displacement_mm": GRID_ROUNDOFF_MAX_DISPLACEMENT_MM,
        "resampled": False, "proposal_and_crop_indices_basis": "original_source_affine",
        "native_and_actor_physical_grid": "derived_affine_ras_mm"}
    return immutable_array(derived, np.float64), freeze_json(record)


@dataclass(frozen=True)
class NativeSpatialCase:
    """Explicit observed geometry and separate evaluator labels on one source grid.

    The CaseData adapter below applies existing anatomical support gates. Direct
    construction requires an explicit provenance/track declaration; it does not
    confer expert review or cortical access approval on a supplied mask.
    """
    structural_intensity: np.ndarray
    observed_support: np.ndarray
    reference_target: np.ndarray
    affine_ras_mm: np.ndarray
    access: AccessWindow
    tools: tuple[ToolGeometry, ...]
    track: str = "annotation_assisted"
    support_source_kind: str = "supplied_annotation"
    support_derivation: str = "explicitly supplied source-grid tissue envelope; cortical access unverified"
    nominal_target: np.ndarray | None = None
    target_source_kind: str = "supplied_annotation"
    target_derivation: str = ""
    crop_shape: tuple[int, int, int] = (32, 32, 32)
    support_provenance: Mapping = field(default_factory=dict)
    intensity_normalization: str = "raw"
    native_grid_reconciliation: str = "none"
    proposal_mode: str = "fixed_lattice"
    proposal_config: NominalCavityProposalConfig | None = None
    _nominal_proposer: PreparedNominalCavityProposer | None = field(init=False, repr=False, default=None)
    _normalization_record: Mapping = field(init=False, repr=False)
    _native_affine_ras_mm: np.ndarray = field(init=False, repr=False)
    _grid_record: Mapping = field(init=False, repr=False)
    _native_config: NativeResectionConfig = field(init=False, repr=False)
    _native_identity: tuple = field(init=False, repr=False)
    _source_hash: str = field(init=False, repr=False)
    _reference_hash: str = field(init=False, repr=False)
    _identity: tuple = field(init=False, repr=False)
    _crop_origin: tuple[int, int, int] = field(init=False, repr=False)
    _crop_shape: tuple[int, int, int] = field(init=False, repr=False)
    _candidate_voxels: tuple[tuple[int, int, int], ...] = field(init=False, repr=False)
    _candidate_scope: str = field(init=False, repr=False)

    def __post_init__(self):
        image = np.asarray(self.structural_intensity)
        if (image.ndim != 3 or min(image.shape) < 3 or image.size > 32_000_000
                or image.dtype.kind not in "iuf" or not np.isfinite(image).all()):
            raise ValueError("A finite native structural scan within the local memory bound is required")
        image = immutable_array(image, np.float32)
        if not np.isfinite(image).all():
            raise ValueError("Structural values exceed finite float32 representation")
        support = _binary(self.observed_support, image.shape, "observed_support")
        target = _fraction(self.reference_target, image.shape, "reference_target")
        nominal = None if self.nominal_target is None else _fraction(self.nominal_target, image.shape, "nominal_target")
        if not support.any():
            raise ValueError("Explicit tissue support is essential; full-head signal is not a brain envelope")
        if nominal is not None and np.any((nominal > 0) & ~support):
            raise ValueError("Permitted target estimates conflict with supplied tissue support")
        if self.proposal_mode == "nominal_cavity_v1":
            if nominal is None:
                raise ValueError("ESSENTIAL_EVIDENCE_MISSING: nominal/cavity proposals need explicit permitted target evidence")
            rule = self.proposal_config
            if isinstance(rule, Mapping):
                rule = NominalCavityProposalConfig(**dict(rule))
            rule = rule or NominalCavityProposalConfig()
            if not isinstance(rule, NominalCavityProposalConfig):
                raise ValueError("Nominal/cavity proposal configuration must be typed or an explicit object")
            object.__setattr__(self, "proposal_config", rule)
        elif self.proposal_mode != "fixed_lattice" or self.proposal_config is not None:
            raise ValueError("Unknown proposal mode or a configuration supplied to the unchanged fixed lattice")
        affine = np.asarray(self.affine_ras_mm)
        if affine.dtype.kind not in "iuf" or not np.isfinite(affine).all():
            raise ValueError("A real finite RAS affine is required")
        if self.native_grid_reconciliation == "orthogonal_roundoff_1e-6mm":
            native_affine, grid_record = reconcile_native_grid_roundoff(affine, image.shape)
            grid_record = freeze_json({**grid_record, "source_image_hash": array_digest(image),
                "support_hash": array_digest(support)})
        elif self.native_grid_reconciliation == "none":
            native_affine, grid_record = immutable_array(affine, np.float64), freeze_json({"method": "none"})
        else:
            raise ValueError("Unknown explicit native grid reconciliation")
        object.__setattr__(self, "_native_affine_ras_mm", native_affine)
        object.__setattr__(self, "_grid_record", grid_record)
        tools = tuple(self.tools)
        if (not 1 <= len(tools) <= 4 or any(not isinstance(t, ToolGeometry) for t in tools)
                or len({t.tool_id for t in tools}) != len(tools)):
            raise ValueError("One to four distinct explicit tool configurations are required")
        if not isinstance(self.access, AccessWindow):
            raise ValueError("One explicit hypothetical access window is required")
        for value in (self.support_derivation, self.target_derivation):
            if not isinstance(value, str) or len(value) > 2048:
                raise ValueError("Source derivations must be bounded text")
        crop_shape = tuple(self.crop_shape)
        if len(crop_shape) != 3 or any(type(v) is not int or not 3 <= v <= 64 for v in crop_shape):
            raise ValueError("The fixed source/access crop requires three dimensions in [3,64]")
        if not isinstance(self.support_provenance, Mapping):
            raise ValueError("Support provenance must be an immutable JSON object")
        object.__setattr__(self, "support_provenance", freeze_json(self.support_provenance))
        if self.intensity_normalization not in {"raw", "support_percentile_1_99"}:
            raise ValueError("Unknown native spatial intensity normalization")
        normalization = {"method": "raw"}
        if self.intensity_normalization == "support_percentile_1_99":
            lower, upper = np.percentile(image[support], [1., 99.], method="linear")
            if not np.isfinite([lower, upper]).all() or upper <= lower:
                raise ValueError("DEGENERATE_SCAN_INTENSITY_RANGE: support percentiles need distinct finite bounds")
            normalization = {"method": self.intensity_normalization,
                "percentiles": [1., 99.], "percentile_method": "linear",
                "lower": float(lower), "upper": float(upper), "clip": [0., 1.],
                "statistics_voxels": int(support.sum()), "statistics_scope": "entire_permitted_support",
                "source_image_hash": array_digest(image), "support_hash": array_digest(support),
                "reference_labels_used": False, "crop_used_for_statistics": False,
                "raw_source_preserved": True}
        object.__setattr__(self, "_normalization_record", freeze_json(normalization))
        for name, value in (("structural_intensity", image), ("observed_support", support),
                            ("reference_target", target), ("affine_ras_mm", immutable_array(affine, np.float64)),
                            ("tools", tools), ("nominal_target", nominal), ("crop_shape", crop_shape)):
            object.__setattr__(self, name, value)
        # Source-grid-normal rays are the entire declared primitive family.
        directions = self.affine_ras_mm[:3, :3] / np.linalg.norm(self.affine_ras_mm[:3, :3], axis=0)
        alignment = np.abs(directions.T @ self.access.normal_inward)
        if not np.any(np.isclose(alignment, 1., rtol=0, atol=1e-10)):
            raise ValueError("This bounded primitive family requires a source-axis-normal aperture")
        axis = int(np.argmax(alignment))
        sign = 1 if directions[:, axis] @ self.access.normal_inward > 0 else -1
        access_voxel = np.linalg.solve(self.affine_ras_mm[:3, :3], self.access.center_mm - self.affine_ras_mm[:3, 3])
        actual_shape = np.minimum(image.shape, crop_shape)
        center = access_voxel.copy()
        center[axis] += sign * actual_shape[axis] / 4
        origin = np.clip(np.floor(center - actual_shape / 2).astype(int), 0, np.asarray(image.shape) - actual_shape)
        object.__setattr__(self, "_crop_origin", tuple(int(v) for v in origin))
        object.__setattr__(self, "_crop_shape", tuple(int(v) for v in actual_shape))
        if self.proposal_mode == "nominal_cavity_v1":
            voxels, scope = (), "permitted_nominal_and_observed_cavity_columns_without_crop_clipping"
        elif int(support.sum()) * len(tools) <= MAX_PRIMITIVES:
            voxels = tuple(tuple(int(v) for v in cell) for cell in np.argwhere(support))
            scope = "all_observed_support_cells_in_small_source"
        else:
            # Fixed 3-column, 8-depth lattice. No target, intensity ranking or
            # hidden field determines this explicitly bounded proposal scope.
            transverse = next(dim for dim in range(3) if dim != axis)
            start = np.rint(access_voxel).astype(int)
            start[axis] = int(np.floor(access_voxel[axis]) + 1) if sign > 0 else int(np.ceil(access_voxel[axis]) - 1)
            points = []
            for offset in (-1, 0, 1):
                for depth in (0, 1, 2, 4, 8, 16, 24, 31):
                    point = start.copy()
                    point[transverse] += offset
                    point[axis] += sign * depth
                    if np.all(point >= origin) and np.all(point < origin + actual_shape):
                        points.append(tuple(int(v) for v in point))
            voxels, scope = tuple(points), "fixed_access_grid_3columns_8depths_within_actor_crop"
        object.__setattr__(self, "_candidate_voxels", voxels)
        object.__setattr__(self, "_candidate_scope", scope)
        object.__setattr__(self, "_identity", self._identity_record())
        source_hash = semantic_digest({"version": NATIVE_SPATIAL_VERSION,
            "scan": array_digest(image), "support": array_digest(support), "affine": array_digest(self.affine_ras_mm),
            "nominal_target": None if nominal is None else array_digest(nominal),
            "access": _access_record(self.access), "tools": [asdict(tool) for tool in tools],
            "track": self.track, "crop_origin": self._crop_origin, "crop_shape": self._crop_shape,
            "proposal_scope": scope, "candidate_voxels": voxels,
            "provenance": [self.support_source_kind, self.support_derivation, self.target_source_kind, self.target_derivation],
            **({"support_provenance": self.support_provenance} if self.support_provenance else {}),
            **({"intensity_normalization": self._normalization_record} if self.intensity_normalization != "raw" else {}),
            **({"native_grid_reconciliation": self._grid_record} if self.native_grid_reconciliation != "none" else {}),
            **({"proposal_mode": self.proposal_mode, "proposal_rule": self.proposal_config.fingerprint}
               if self.proposal_mode != "fixed_lattice" else {})})
        object.__setattr__(self, "_source_hash", source_hash)
        object.__setattr__(self, "_reference_hash", semantic_digest({"source": source_hash, "target": array_digest(target)}))
        config = NativeResectionConfig(support, np.zeros(image.shape, np.int16), self._native_affine_ras_mm,
            self.access, tools, self.source_hash,
            self.support_derivation + "; hypothetical aperture; cortical access unverified",
            case_id="native-spatial", max_tip_step_mm=min(.25, float(np.linalg.norm(self._native_affine_ras_mm[:3, :3], axis=0).min()) / 2))
        object.__setattr__(self, "_native_config", config)
        object.__setattr__(self, "_native_identity", _native_identity(config))
        if self.proposal_mode == "nominal_cavity_v1":
            provenance = {"source_hash": self._source_hash, "nominal_target_hash": array_digest(nominal),
                "source_kind": self.target_source_kind, "derivation": self.target_derivation,
                "source_image_hash": array_digest(image), "original_frame_hash": array_digest(self.affine_ras_mm)}
            proposer = PreparedNominalCavityProposer(config, nominal, nominal_provenance=provenance,
                config=self.proposal_config, index_affine=self.affine_ras_mm, index_frame_record=self._grid_record)
            object.__setattr__(self, "_nominal_proposer", proposer)
            object.__setattr__(self, "_identity", self._identity_record())
        # Enforce the same observed-frame contract as the policy boundary now.
        self.spatial_inputs(np.zeros(image.shape, bool))

    @property
    def source_hash(self):
        self.assert_intact()
        return self._source_hash

    @property
    def reference_hash(self):
        self.assert_intact()
        return self._reference_hash

    def _identity_record(self):
        arrays = tuple(_array_identity(getattr(self, name)) for name in
            ("structural_intensity", "observed_support", "reference_target", "affine_ras_mm", "_native_affine_ras_mm"))
        return (arrays, None if self.nominal_target is None else _array_identity(self.nominal_target),
            semantic_digest({"access": _access_record(self.access), "tools": [asdict(t) for t in self.tools]}),
            self.track, self.support_source_kind, self.support_derivation, self.target_source_kind,
            self.target_derivation, self.crop_shape, self._crop_origin, self._crop_shape,
            self._candidate_voxels, self._candidate_scope, semantic_digest(self.support_provenance),
            self.intensity_normalization, semantic_digest(self._normalization_record),
            self.native_grid_reconciliation, semantic_digest(self._grid_record),
            self.proposal_mode, None if self.proposal_config is None else self.proposal_config.fingerprint,
            None if self._nominal_proposer is None else (id(self._nominal_proposer), self._nominal_proposer.model_hash))

    def assert_intact(self):
        if (self._identity_record() != self._identity or (hasattr(self, "_native_identity")
                and _native_identity(self._native_config) != self._native_identity)):
            raise RuntimeError("Native spatial source content or interpretation was replaced")

    def spatial_inputs(self, cavity):
        region = tuple(slice(origin, origin + size) for origin, size in zip(self._crop_origin, self._crop_shape))
        affine = np.array(self._native_affine_ras_mm, copy=True)
        affine[:3, 3] += affine[:3, :3] @ self._crop_origin
        intensity = self.structural_intensity[region]
        intensity_derivation = ""
        if self.intensity_normalization != "raw":
            lower, upper = self._normalization_record["lower"], self._normalization_record["upper"]
            # Normalize only the small actor crop; native source bytes and
            # geometry stay untouched. Float64 subtraction avoids overflow.
            intensity = np.clip((intensity.astype(np.float64) - lower) / (upper - lower), 0., 1.).astype(np.float32)
            intensity_derivation = (f"support_percentile_1_99 linear; lower={lower!r}; upper={upper!r}; "
                f"clip=[0,1]; record={semantic_digest(self._normalization_record)}")
        channels = {
            "structural_intensity": ObservedChannel(intensity,
                source_kind="synthetic_scan" if self.track == "synthetic_scan" else "observed_scan",
                derivation=intensity_derivation),
            "nominal_tissue": ObservedChannel(self.observed_support[region], source_kind=self.support_source_kind,
                derivation=self.support_derivation,
                derived_from=("structural_intensity",) if self.support_source_kind == "derived_from_scan" else ()),
            "observed_cavity": ObservedChannel(np.asarray(cavity)[region], source_kind="observed_procedure_state",
                derivation="committed fully contained connected native cells")}
        if self.nominal_target is not None:
            channels["nominal_target"] = ObservedChannel(self.nominal_target[region], source_kind=self.target_source_kind,
                derivation=self.target_derivation,
                derived_from=("structural_intensity",) if self.target_source_kind == "derived_from_scan" else ())
        return SpatialInputs(channels, affine, self.track, self.source_hash)


@dataclass(frozen=True)
class NativeSpatialStep:
    observation: object | None
    reward: float
    terminated: bool
    info: dict

    @property
    def done(self):
        return self.terminated


class NativeSpatialTask:
    """Canonical native execution, geometric objective, evaluator-only labels."""
    def __init__(self, case: NativeSpatialCase, *, max_steps: int = 3,
                 reward: RewardSpec = DEFAULT_NATIVE_SPATIAL_REWARD,
                 cancelled: Callable[[], bool] | None = None, _planning: bool = False):
        if not isinstance(case, NativeSpatialCase) or type(max_steps) is not int or not 1 <= max_steps <= 6:
            raise ValueError("A typed native source and horizon1–6 are required")
        if not isinstance(reward, RewardSpec) or reward.motor_per_mm3 or reward.language_per_mm3 or reward.graph_edge_cost:
            raise ValueError("Functional and graph costs must be disabled: function remains unassessed")
        if _planning and (case.nominal_target is None or not np.array_equal(case.reference_target, case.nominal_target)):
            raise ValueError("Planning must use the explicitly supplied nominal target field")
        self.case, self.max_steps, self.reward_spec = case, max_steps, reward
        self._cancelled, self._planning = cancelled, bool(_planning)
        self._source_hash, self._reference_hash = case.source_hash, case.reference_hash
        self._config = case._native_config
        self._contract = self._contract_record()
        self.decision_model_hash = semantic_digest(self._contract)
        self._engine = NativeResectionEngine(self._config)
        self.reset()

    def _contract_record(self):
        return {"source": self.case.source_hash, "max_steps": self.max_steps, "reward": asdict(self.reward_spec),
            "native_config": self._config.fingerprint, "partial_contact_weight": 0.,
            "proposal_rule": self.case._candidate_scope + "; source-normal entry projection",
            "target_model": self.case.target_derivation or "unavailable_no_search_objective"}

    def _state_record(self):
        return {"removed": array_digest(self._engine.removed_mask), "remaining": array_digest(self._engine.remaining_mask),
            "contact": array_digest(self._engine.contact_mask), "connected_free": array_digest(self._engine.connected_free_mask),
            "engine_history": self._engine.history, "engine_state": self._engine.state_hash,
            "history": self._history, "steps": self._steps, "current_tool": self._current_tool,
            "terminated": self._terminated, "total_reward": self._total_reward, "planning": self._planning}

    def _seal(self):
        self._state_seal = semantic_digest(self._state_record())

    def _assert_frozen(self):
        if (self._contract_record() != self._contract or self.case.reference_hash != self._reference_hash
                or self._config is not self.case._native_config or self._engine.config is not self._config
                or semantic_digest(self._state_record()) != self._state_seal):
            raise RuntimeError("Native source, objective or committed procedure state changed outside a transition")

    def _check_cancelled(self):
        if self._cancelled is not None and self._cancelled():
            raise InterruptedError("Native spatial operation cancelled")

    @property
    def terminated(self):
        return self._terminated

    def reset(self, seed=0):
        if type(seed) is not int or seed != 0:
            raise ValueError("This adapter has one deterministic geometric world; nonzero scenario seeds are unsupported")
        if hasattr(self, "_state_seal"):
            self._assert_frozen()
        self._check_cancelled()
        self._engine.reset()
        self._steps, self._total_reward, self._current_tool = 0, 0., None
        self._terminated, self._history = False, []
        self._inventory = None
        self._ledger = ()
        self._proposal_batch = None
        self._seal()
        return self.observation()

    def _prepare_inventory(self):
        self._assert_frozen()
        self._check_cancelled()
        if self._inventory is not None:
            return self._inventory
        inventory, ledger = {}, []
        batch = None
        if not self._terminated and self.case._nominal_proposer is not None:
            batch = self.case._nominal_proposer.propose(self._engine, cancelled=self._cancelled)
            outcomes = {}
            origin, last = np.array(self.case._crop_origin), np.array(self.case._crop_origin)+self.case._crop_shape
            for ray in batch.proposals:
                self._check_cancelled()
                result = self._engine.preview_stroke(ray.tool_id, ray.tip_mm, entry_mm=ray.entry_mm)
                inside = bool(np.all(np.asarray(ray.voxel) >= origin) and np.all(np.asarray(ray.voxel) < last))
                outcomes[ray.proposal_id] = {"entry_mm": list(ray.entry_mm), "tip_mm": list(ray.tip_mm),
                    "feasible": bool(result.feasible), "reason": result.reason,
                    "endpoint_center_in_actor_crop": inside}
                if result.feasible:
                    inventory[ray.proposal_id] = result
            for slot in batch.ledger:
                row = {**asdict(slot), "offset_source_voxels": list(slot.offset_source_voxels),
                    "voxel": None if slot.voxel is None else list(slot.voxel),
                    "action_id": slot.proposal_id, "proposal_reason": slot.reason, "feasible": False}
                if slot.reason == "PROPOSED_UNCERTIFIED":
                    row.update(outcomes[slot.proposal_id])
                ledger.append(row)
        elif not self._terminated:
            cavity = array_digest(self._engine.removed_mask)
            for voxel in self.case._candidate_voxels:
                tip = self.case._native_affine_ras_mm[:3, :3] @ voxel + self.case._native_affine_ras_mm[:3, 3]
                depth = float((tip - self.case.access.center_mm) @ self.case.access.normal_inward)
                entry = tip - depth * self.case.access.normal_inward
                for tool in self.case.tools:
                    self._check_cancelled()
                    identity = semantic_digest({"source": self._source_hash, "cavity": cavity,
                        "voxel": list(voxel), "tool_id": tool.tool_id})
                    action_id = "NATIVE-SPATIAL:" + identity.split(":")[1][:24]
                    result = None if depth <= 0 else self._engine.preview_stroke(tool.tool_id, tip, entry_mm=entry)
                    feasible = result is not None and result.feasible
                    ledger.append({"voxel": list(voxel), "tool_id": tool.tool_id, "entry_mm": entry.tolist(),
                        "tip_mm": tip.tolist(), "action_id": action_id, "feasible": bool(feasible),
                        "reason": "OUTSIDE_DECLARED_INWARD_WORKSPACE" if result is None else result.reason})
                    if feasible:
                        inventory[action_id] = result
        self._check_cancelled()
        self._inventory, self._ledger, self._proposal_batch = inventory, tuple(ledger), batch
        return inventory

    def observation(self):
        inventory = self._prepare_inventory()
        tools = {tool.tool_id: tool for tool in self.case.tools}
        actions = [SpatialAction("STOP")]
        actions.extend(SpatialAction(identifier, result.entry_mm, result.tip_mm, tools[result.tool_id])
                       for identifier, result in inventory.items())
        state = ObservedProcedureState(self.case.access, self._steps, self.max_steps, self._current_tool)
        return build_spatial_observation(self.case.spatial_inputs(self._engine.removed_mask), actions, state)

    def _score_record(self, geometry, prior_tool):
        if geometry.get("action_id") == "STOP":
            return {"action_id": "STOP", "reward": 0., "target_removed_mm3": 0., "normal_removed_mm3": 0.,
                    "insertion_distance_mm": 0., "complete_tool_path_length_mm": 0.}
        indices = np.asarray(geometry["removed_indices_native"], dtype=int)
        target = float(self.case.reference_target[tuple(indices.T)].sum() * self._config.voxel_volume_mm3)
        total = len(indices) * self._config.voxel_volume_mm3
        distance = float(np.linalg.norm(np.subtract(geometry["tip_mm"], geometry["entry_mm"])))
        weights = self.reward_spec
        reward = (weights.target_per_mm3 * target - weights.normal_per_mm3 * (total - target)
            - weights.action_cost - 2 * weights.motion_per_mm * distance
            - weights.tool_change_cost * (prior_tool is not None and prior_tool != geometry["tool_id"]))
        return {**geometry, "reward": float(reward), "target_removed_mm3": target,
            "normal_removed_mm3": total - target, "insertion_distance_mm": distance,
            "complete_tool_path_length_mm": 2 * distance, "partial_contact_weight": 0.,
            "function_unassessed_removed_volume_mm3": total, "motor_surrogate": None,
            "language_surrogate": None, "clinical_deficit_probability": None,
            "outcome_scope": "permitted_nominal_model" if self._planning else "separate_evaluator_reference"}

    def step(self, action: str | int):
        """Commit a transition and eagerly produce the successor observation."""
        return self._transition(action, observe_successor=True)

    def advance_planning(self, action: str | int):
        """Commit a nominal planning transition without preparing its successor.

        Only planning clones may use this path. Current-action certification,
        scoring and committed history are identical to ``step``. A later
        observation or inventory request prepares the successor on demand.
        """
        self._assert_frozen()
        if not self._planning:
            raise ValueError("LAZY_PLANNING_ONLY: advance_planning requires a nominal planning clone")
        return self._transition(action, observe_successor=False)

    def _transition(self, action: str | int, *, observe_successor: bool):
        if self._terminated:
            raise InvalidActionError("Native spatial episode has terminated")
        inventory = self._prepare_inventory()
        ids = ("STOP", *inventory)
        if isinstance(action, (int, np.integer)) and not isinstance(action, (bool, np.bool_)):
            action = ids[int(action)] if 0 <= action < len(ids) else None
        if not isinstance(action, str) or action not in ids:
            raise InvalidActionError("Unknown, stale or infeasible native spatial action")
        self._check_cancelled()
        if action == "STOP":
            record = self._score_record({"action_id": "STOP"}, self._current_tool)
        else:
            result = inventory[action]
            record = self._score_record({**result.to_history_record(), "action_id": action}, self._current_tool)
            self._engine.commit_preview(result)
            self._current_tool = result.tool_id
        self._steps += 1
        self._total_reward += record["reward"]
        self._terminated = action == "STOP" or self._steps >= self.max_steps
        self._history.append(copy.deepcopy(record))
        self._inventory, self._ledger, self._proposal_batch = None, (), None
        self._seal()
        try:
            if observe_successor:
                observed = self.observation()
            else:
                # A cancellation during commit must still report the durable
                # transition even though successor previews are deferred.
                self._check_cancelled()
                observed = None
        except InterruptedError as error:
            from .native_axis_simulation import CommittedTransitionInterrupted
            raise CommittedTransitionInterrupted(record, record["reward"]) from error
        return NativeSpatialStep(observed, record["reward"], self._terminated, copy.deepcopy(record))

    def clone(self):
        """Branch copies mutable engine/state only; immutable source and previews share safely."""
        self._assert_frozen()
        result = copy.copy(self)
        result._engine = self._engine.clone()
        result._history = copy.deepcopy(self._history)
        result._inventory = None if self._inventory is None else self._inventory.copy()
        result._ledger = copy.deepcopy(self._ledger)
        return result

    def planning_clone(self):
        """Replace private outcomes without re-executing any geometry or previews."""
        if self.case.nominal_target is None:
            raise ValueError("ESSENTIAL_EVIDENCE_MISSING: explicit nominal target required for SEARCH; no MRI threshold fallback")
        result = self.clone()
        # Arrays are immutable, so a teacher branch can share permitted source
        # and substitute the already validated nominal field without rescanning
        # or copying the patient's full MRI/support/configuration.
        result.case = copy.copy(self.case)
        object.__setattr__(result.case, "reference_target", self.case.nominal_target)
        object.__setattr__(result.case, "_identity", result.case._identity_record())
        object.__setattr__(result.case, "_reference_hash", semantic_digest({"source": self._source_hash,
            "target": array_digest(self.case.nominal_target)}))
        result._reference_hash, result._planning = result.case.reference_hash, True
        prior_tool, records = None, []
        for record in self._history:
            records.append(result._score_record(record, prior_tool))
            if record["action_id"] != "STOP":
                prior_tool = record["tool_id"]
        result._history = records
        result._total_reward = sum(record["reward"] for record in records)
        result._seal()
        return result

    def fresh(self):
        self._assert_frozen()
        return type(self)(self.case, max_steps=self.max_steps, reward=self.reward_spec,
                          cancelled=self._cancelled, _planning=self._planning)

    def observed_one_step_search(self, *, seconds: float = 60.):
        """Score every current certified nominal action, then stop after the best.

        This is an exact immediate comparison, not a whole-horizon optimum.
        Source preparation performed before this call must be charged separately.
        No change is made to this task; the returned sequence requires replay.
        """
        return self._observed_greedy_search(seconds=seconds, one_step=True)

    def observed_greedy_search(self, *, seconds: float = 60.):
        """Repeat complete immediate scoring through the same remaining horizon.

        Current native certificates supply hypothetical contained-cell removals;
        only the permitted nominal target scores them. STOP wins ties at zero.
        Negative preparatory moves are not explored, which is a stated greedy
        limitation. No successor branch is committed merely to score its action.
        """
        return self._observed_greedy_search(seconds=seconds, one_step=False)

    def _observed_greedy_search(self, *, seconds: float, one_step: bool):
        from .observed_search import ObservedSearchLimit

        if isinstance(seconds, bool) or not isinstance(seconds, (int, float)) or not math.isfinite(seconds) or seconds <= 0:
            raise ValueError("Observed greedy search needs a finite positive time budget")
        started = time.perf_counter()
        sequence, decisions = [], []
        evaluated = commits = 0
        initial_steps = self._steps
        initial_value = 0.
        model = None

        def accounting(complete=False):
            return {"method": "observed_one_step" if one_step else "observed_greedy",
                "objective_source": "permitted_nominal_target_and_frozen_geometric_costs",
                "complete": complete, "planning_seconds": time.perf_counter() - started,
                "time_budget_seconds": seconds, "evaluated_nonstop_actions": evaluated,
                "model_transition_calls": commits, "decisions": copy.deepcopy(decisions),
                "initial_steps": initial_steps, "max_steps": self.max_steps,
                "estimated_incremental_return": 0. if model is None else model._total_reward - initial_value,
                "initial_source_preparation": "outside_this_call; caller_must_report_and_charge",
                "within_call_costs": "nominal clone, certificate checks, all candidate scoring, selected commits and needed inventories",
                "global_optimality_proven": False, "native_replay_required": True}

        def check():
            self._check_cancelled()
            if time.perf_counter() - started > seconds:
                raise ObservedSearchLimit("Observed greedy search exceeded its declared time budget",
                    accounting=accounting(), best_sequence=tuple(sequence))

        check()
        model = self.planning_clone()
        initial_value = model._total_reward
        while not model.terminated:
            check()
            inventory = model._prepare_inventory()
            scores = [{"action_id": "STOP", "reward": 0., "target_removed_mm3": 0., "normal_removed_mm3": 0.,
                       "insertion_distance_mm": 0., "complete_tool_path_length_mm": 0.}]
            for identifier, preview in inventory.items():
                check()
                # Mirror the engine's non-mutating certificate admission checks;
                # hypothetical scoring must not trust a forged cache entry.
                engine = model._engine
                if (not preview.feasible or engine._preview_records.get(id(preview)) is not preview
                        or preview.source_state_hash != engine.state_hash
                        or preview.decision_model_hash != engine.config.fingerprint
                        or engine._preview_digests.get(id(preview)) != engine._result_digest(preview)):
                    raise ValueError("Observed search encountered a stale, foreign or changed native certificate")
                scored = model._score_record({**preview.to_history_record(), "action_id": identifier}, model._current_tool)
                scores.append({key: scored[key] for key in scores[0]})
                evaluated += 1
            check()
            selected = max(scores, key=lambda row: row["reward"])
            decisions.append({"step": model._steps, "source_state_hash": model._engine.state_hash,
                "legal_nonstop_actions": len(inventory), "scored_nonstop_actions": len(scores) - 1,
                "all_current_legal_actions_scored": True, "scores": scores,
                "selected_action_id": selected["action_id"]})
            model.advance_planning(selected["action_id"])
            sequence.append(selected["action_id"])
            commits += 1
            check()
            if one_step and not model.terminated:
                # STOP has no geometric effect; avoid preparing a successor
                # inventory merely to encode its known zero-cost continuation.
                sequence.append("STOP")
                break
        check()
        self._assert_frozen()
        model._assert_frozen()
        return tuple(sequence), accounting(complete=True)

    def candidate_inventory(self):
        self._prepare_inventory()
        if self.case._nominal_proposer is not None:
            config = self.case.proposal_config
            slots = len(config.offsets_source_voxels) * len(self.case.tools) * len(NOMINAL_CAVITY_FAMILIES)
            emitted = [copy.deepcopy(row) for row in self._ledger if row["proposal_reason"] == "PROPOSED_UNCERTIFIED"]
            counts = {}
            for row in self._ledger:
                counts[row["proposal_reason"]] = counts.get(row["proposal_reason"], 0)+1
            omitted, duplicate = counts.get("CANDIDATE_CAP",0), counts.get("DUPLICATE_GEOMETRY",0)
            return {"basis": self.case._candidate_scope, "provider_version": NOMINAL_CAVITY_PROPOSAL_VERSION,
                "source_hash": self._source_hash, "decision_model_hash": self.decision_model_hash,
                "cavity_state_hash": self._engine.state_hash,
                "provider_model_hash": self.case._nominal_proposer.model_hash,
                "nominal_target_hash": array_digest(self.case.nominal_target),
                "nominal_provenance_hash": None if self._proposal_batch is None else self._proposal_batch.nominal_provenance_hash,
                "declared_slots": slots, "evaluated_slots": len(emitted), "emitted_count": len(emitted),
                "accepted_count": len(self._inventory), "rejected_count": len(emitted)-len(self._inventory),
                "omitted_count": omitted, "duplicate_count": duplicate,
                "unavailable_count": len(self._ledger)-len(emitted)-omitted-duplicate,
                "complete": omitted == 0, "ledger_complete": True, "terminal": self._terminated,
                "steps_taken": self._steps, "max_steps": self.max_steps, "remaining_steps": self.max_steps-self._steps,
                "terminated_slots": slots if self._terminated else 0, "candidate_cap": config.max_candidates,
                "crop_clipping": False, "endpoint_centers_in_actor_crop": sum(row["endpoint_center_in_actor_crop"] for row in emitted),
                "all_support_cell_tool_pairs": int(self.case.observed_support.sum())*len(self.case.tools),
                "scope": "complete_dispositions_for_declared_columns_and_families_not_all_surgical_paths",
                "disposition_counts": counts, "emitted": emitted, "ledger": copy.deepcopy(list(self._ledger))}
        slots = len(self.case._candidate_voxels) * len(self.case.tools)
        emitted = [copy.deepcopy(row) for row in self._ledger if row["reason"] != "OUTSIDE_DECLARED_INWARD_WORKSPACE"]
        return {"basis": self.case._candidate_scope, "declared_slots": slots,
            "source_hash": self._source_hash, "decision_model_hash": self.decision_model_hash,
            "cavity_state_hash": self._engine.state_hash,
            "evaluated_slots": len(self._ledger), "accepted_count": len(self._inventory),
            "rejected_count": len(self._ledger) - len(self._inventory), "omitted_count": 0,
            "terminated_slots": slots if self._terminated else 0, "complete": True,
            "scope": "complete_declared_source_normal_primitives_not_all_surgical_paths",
            "terminal": self._terminated, "steps_taken": self._steps, "max_steps": self.max_steps,
            "remaining_steps": self.max_steps-self._steps, "emitted_count": len(emitted), "emitted": emitted,
            "all_support_cell_tool_pairs": int(self.case.observed_support.sum()) * len(self.case.tools),
            "ledger": copy.deepcopy(list(self._ledger))}

    def metrics(self):
        self._assert_frozen()
        voxel = self._config.voxel_volume_mm3
        removed, contact = self._engine.removed_mask, self._engine.contact_mask
        return {"task_version": NATIVE_SPATIAL_VERSION, "source_hash": self._source_hash,
            "reference_hash": self._reference_hash, "decision_model_hash": self.decision_model_hash,
            "steps": self._steps, "terminated": self._terminated, "total_reward": self._total_reward,
            "target_removed_mm3": float(self.case.reference_target[removed].sum() * voxel),
            "normal_removed_mm3": float((1 - self.case.reference_target[removed]).sum() * voxel),
            "reference_target_outside_observed_support_mm3": float(self.case.reference_target[~self.case.observed_support].sum() * voxel),
            "simulated_removed_volume_mm3": float(removed.sum() * voxel),
            "cumulative_contacted_tissue_upper_bound_mm3": float(contact.sum() * voxel),
            "currently_retained_contacted_tissue_upper_bound_mm3": float((contact & ~removed).sum() * voxel),
            "partial_contact_weight": 0., "functional_evidence_available": {"motor": False, "language": False},
            "function_unassessed_removed_volume_mm3": float(removed.sum() * voxel),
            "motor_surrogate": None, "language_surrogate": None, "clinical_deficit_probability": None,
            "clinical_probability_reason": "no_validated_clinical_outcome_model",
            "unknowns": ["motor_evidence_unavailable", "language_evidence_unavailable", "vascular_coverage_unassessed"],
            "planning_estimator_only": self._planning, "history": copy.deepcopy(self._history),
            "observation_track": self.case.track,
            "proposal_mode": self.case.proposal_mode,
            "proposal_rule_hash": None if self.case.proposal_config is None else self.case.proposal_config.fingerprint,
            "support_provenance": thaw_json(self.case.support_provenance),
            "intensity_normalization": thaw_json(self.case._normalization_record),
            "native_grid_reconciliation": thaw_json(self.case._grid_record),
            "crop": {"origin_voxels": self.case._crop_origin, "shape": self.case._crop_shape,
                     "basis": "scan_frame_and_declared_access_only", "native_geometry_resampled": False},
            "assumptions": ["rigid_fully_contained_native_cell_removal",
                "hypothetical_intracranial_aperture", "no_deformation_or_force_model",
                "geometric_objective_does_not_optimize_function_or_partial_contact"]}

    def independent_geometry_check(self):
        self._assert_frozen()
        from .evaluation import independent_check_native_history
        source = SimpleNamespace(mri=self.case.structural_intensity, affine=self.case._native_affine_ras_mm,
                                 frame="RAS+", semantic_hash=self._source_hash)
        return independent_check_native_history(source, self.case.tools, self._history,
            tissue_mask=self.case.observed_support, access=self.case.access, geometry_frame="RAS+")


def make_native_opening_task(*, tools=OPENING_TOOLS, max_steps=2, cancelled=None):
    """Six tissue cells inside a 9×9×7 workspace; exact tool-complementarity fixture.

    The offset short opener clears a shaft obstacle. The long narrow tool can
    then reach two distal target cells. This is not an anatomical brain model.
    """
    support = np.zeros((9, 9, 7), bool)
    support[4, 4, 1:6], support[5, 5, 1] = True, True
    target = np.zeros(support.shape, bool)
    target[4, 4, 4:6] = True
    scan = np.where(target, .8, np.where(support, .2, 0.)).astype(np.float32)
    case = NativeSpatialCase(scan, support, target, np.eye(4),
        AccessWindow((3.9, 4., .5), (0., 0., 1.), 2.4, "analytic-offset-opening"), tuple(tools),
        track="synthetic_scan", support_source_kind="derived_from_scan",
        support_derivation="unit-fixture forward model: nonzero structural signal identifies support",
        nominal_target=scan >= SYNTHETIC_TARGET_THRESHOLD, target_source_kind="derived_from_scan",
        target_derivation="unit-fixture .2/.8 signal midpoint .5; never an MRI estimator")
    return NativeSpatialTask(case, max_steps=max_steps, cancelled=cancelled)


def _provisional_proposal_support(case, acknowledgment):
    """Opt in to one exact research proposal without changing its review state."""
    from .structural_evidence import validate_support_assumption
    if not isinstance(acknowledgment, Mapping):
        raise ValueError("PROVISIONAL_SUPPORT_ACKNOWLEDGMENT_REQUIRED: expected a bound research declaration")
    acknowledgment = thaw_json(freeze_json(acknowledgment))
    fields = {"schema_version", "scope", "purpose", "case_hash", "planning_hash", "evidence_id",
        "evidence_hash", "source_image_hash", "source_frame_hash", "mask_hash", "model_sha256",
        "run_sha256", "declared_by", "declared_at", "rationale", "acknowledge_unreviewed",
        "cortical_access_permitted", "clinical_use_permitted"}
    if set(acknowledgment) != fields:
        raise ValueError("PROVISIONAL_SUPPORT_ACKNOWLEDGMENT_FIELDS: exact declaration fields are required")
    if (type(acknowledgment["schema_version"]) is not int or acknowledgment["schema_version"] != 1
            or acknowledgment["purpose"] != "native_spatial_provisional_research"
            or acknowledgment["acknowledge_unreviewed"] is not True
            or acknowledgment["cortical_access_permitted"] is not False
            or acknowledgment["clinical_use_permitted"] is not False):
        raise ValueError("PROVISIONAL_SUPPORT_RESEARCH_ONLY: unreviewed research use must be explicit")
    if any(not isinstance(acknowledgment[key], str) or not acknowledgment[key].strip()
           or len(acknowledgment[key]) > 1024 for key in fields - {
               "schema_version", "acknowledge_unreviewed", "cortical_access_permitted", "clinical_use_permitted"}):
        raise ValueError("PROVISIONAL_SUPPORT_ACKNOWLEDGMENT_TEXT: expected bounded nonempty strings")
    item = case.structural_evidence.get(acknowledgment["evidence_id"])
    if item is None:
        raise ValueError("PROVISIONAL_SUPPORT_UNKNOWN_PROPOSAL: select an existing source-bound proposal")
    item.assert_matches(case)
    if item.review_status != "review_required" or item.provenance != "estimated" or item.model_sha256 is None:
        raise ValueError("PROVISIONAL_SUPPORT_INELIGIBLE: only an unreviewed model proposal can use this pathway")
    expected = {"case_hash": case.semantic_hash, "planning_hash": case.planning_hash,
        "evidence_hash": item.evidence_hash, "source_image_hash": item.source_image_hash,
        "source_frame_hash": item.source_frame_hash, "mask_hash": item.mask_hash,
        "model_sha256": item.model_sha256, "run_sha256": item.run_sha256}
    if any(acknowledgment[key] != value for key, value in expected.items()):
        raise ValueError("PROVISIONAL_SUPPORT_STALE: case, source, proposal, model or run identity changed")
    # Reuse exact mask/image/frame binding and aware time validation. This is a
    # research declaration, deliberately separate from BrainEnvelopeReview.
    validate_support_assumption(case, item.mask, acknowledgment)
    return item.mask, {"method": "explicitly_acknowledged_unreviewed_model_support",
        "evidence_type": item.provenance, "evidence_hash": item.evidence_hash,
        "review_status": item.review_status, "research_use": "provisional",
        "cortical_access_permitted": False, "clinical_use_permitted": False,
        "clinical_deficit_probability": None, "acknowledgment": acknowledgment}


def native_spatial_task_from_case(case, *, access, tools, max_steps=3,
                                  track="annotation_assisted", nominal_target=None,
                                  nominal_target_derivation="", crop_shape=(32, 32, 32), cancelled=None,
                                  research_support_acknowledgment=None, intensity_normalization="raw",
                                  native_grid_reconciliation="none", proposal_mode="fixed_lattice", proposal_config=None):
    """Load a real source case with existing support gates and explicit target role.

    ``access`` is canonical RAS+. Annotation-assisted defaults to the supplied
    active source compartments. Inference-only never substitutes those labels
    for a missing estimate; planning_clone then refuses a missing objective.
    An explicit source-bound provisional research acknowledgment can select one
    unreviewed model proposal in annotation-assisted mode only. It never edits
    working anatomy or grants cortical/clinical approval on the source case.
    """
    from .structural_evidence import planning_brain_support
    if track not in {"annotation_assisted", "inference_only"}:
        raise ValueError("Real cases require annotation_assisted or inference_only track")
    if research_support_acknowledgment is not None:
        if track != "annotation_assisted":
            raise ValueError("PROVISIONAL_SUPPORT_TRACK: this explicit pathway requires annotation_assisted mode")
        support, record = _provisional_proposal_support(case, research_support_acknowledgment)
    else:
        support, record = planning_brain_support(case)
    if support is None:
        raise ValueError("ESSENTIAL_EVIDENCE_MISSING: explicit source-bound brain support; full-head intensity is not support")
    if not case.compartments:
        raise ValueError("This development adapter needs evaluator reference compartments, separately from actor inputs")
    reference = np.logical_or.reduce(tuple(case.compartments.values()))
    if nominal_target is None and track == "annotation_assisted":
        nominal_target = reference
        nominal_target_derivation = "supplied preoperative source compartments; annotation-assisted target model"
    if nominal_target is not None and not nominal_target_derivation.strip():
        raise ValueError("An explicit target estimate needs recorded derivation/checkpoint provenance")
    # "Estimated" alone does not establish a scan-derived model: an explicit
    # research assumption may describe an arbitrary supplied mask. Only an
    # exact source-bound model proposal (or simulated unit source) supports that
    # narrower provenance claim. Other working masks stay annotation-assisted.
    model_support = any(item.evidence_hash == record.get("evidence_hash")
                        and item.provenance == "estimated" and item.model_sha256
                        for item in getattr(case, "structural_evidence", {}).values())
    simulated_support = record.get("evidence_type") == "simulated"
    support_kind = "derived_from_scan" if model_support or simulated_support else "supplied_annotation"
    affine = np.asarray(case.affine)
    if case.frame == "LPS+":
        affine = np.diag([-1., -1., 1., 1.]) @ affine
    elif case.frame != "RAS+":
        raise ValueError("The source case must declare RAS+ or LPS+")
    source = NativeSpatialCase(case.mri, support, reference, affine, access, tuple(tools), track=track,
        support_source_kind=support_kind, support_derivation=record["method"] + "; cortical access unverified",
        nominal_target=nominal_target, target_source_kind="supplied_annotation" if track == "annotation_assisted" else "derived_from_scan",
        target_derivation=nominal_target_derivation, crop_shape=crop_shape,
        support_provenance=record if research_support_acknowledgment is not None else {},
        intensity_normalization=intensity_normalization, native_grid_reconciliation=native_grid_reconciliation,
        proposal_mode=proposal_mode, proposal_config=proposal_config)
    return NativeSpatialTask(source, max_steps=max_steps, cancelled=cancelled)
