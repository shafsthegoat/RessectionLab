"""Experimental, source-bound native axis-column proposals; no clearance claims.

The provider only reads the current cavity. Every emitted primary or fallback
ray still needs a fresh native-engine preview. This is a new action model and
is deliberately not integrated into selected-route refinement or learning.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field, replace
from hashlib import sha256
import json
from typing import Any
from collections.abc import Mapping
from itertools import product

import numpy as np
from scipy.ndimage import binary_fill_holes

from .native_resection import NATIVE_RESECTION_VERSION, NativeResectionConfig, NativeResectionEngine
from .core import array_digest, immutable_array, freeze_json, thaw_json, semantic_digest
from .geometry import point_segment_distances


AXIS_PROPOSAL_VERSION = "experimental-residual-axis-columns-v1"
TARGET_WITHIN_SUPPORT = "target_within_estimated_support"
SUPPLIED_GOAL_REGION = "supplied_goal_region_may_exceed_estimated_support"
DEFAULT_COLUMN_OFFSETS = ((0, 0), (-2, 0), (2, 0), (0, -2), (0, 2),
                          (-2, -2), (-2, 2), (2, -2), (2, 2),
                          (-3, 0), (3, 0), (0, -3), (0, 3))
_DIRECTION_ATOL = 1e-10
_GRID_ATOL = 1e-8
_HISTORY_ENGINE_VERSION = "contained-native-cell-connected-suction-v2"


def _digest(value: Any) -> str:
    return "sha256:" + sha256(json.dumps(value, sort_keys=True, allow_nan=False).encode()).hexdigest()


def _point(value) -> tuple[float, float, float]:
    return tuple(float(item) for item in value)


@dataclass(frozen=True)
class AxisColumnProposalConfig:
    """Frozen proposal inventory; the cap counts primary rays, not previews."""

    offsets_source_voxels: tuple[tuple[int, int], ...] = DEFAULT_COLUMN_OFFSETS
    max_primary_rays: int = 26
    version: str = field(default=AXIS_PROPOSAL_VERSION, init=False)

    def __post_init__(self):
        offsets = tuple(tuple(row) for row in self.offsets_source_voxels)
        if not offsets or len(offsets) > 128 or any(len(row) != 2 for row in offsets):
            raise ValueError("Offsets must contain between one and 128 source-grid pairs")
        if any(isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer))
               or abs(int(value)) > 2**30 for row in offsets for value in row):
            raise ValueError("Source-grid offsets must be bounded integers")
        offsets = tuple(tuple(int(value) for value in row) for row in offsets)
        if len(set(offsets)) != len(offsets):
            raise ValueError("Source-grid offsets must be unique")
        if type(self.max_primary_rays) is not int or self.max_primary_rays < 1:
            raise ValueError("max_primary_rays must be a positive integer")
        object.__setattr__(self, "offsets_source_voxels", offsets)

    @property
    def fingerprint(self) -> str:
        return _digest({**asdict(self), "direction_atol": _DIRECTION_ATOL,
                        "grid_atol": _GRID_ATOL, "relative_tolerance": 0.,
                        "primary": "distal_remaining_target_in_column",
                        "fallback": "proximal_remaining_target_only_after_primary_rejection",
                        "order": "declared_column_order_then_source_tool_order"})


@dataclass(frozen=True)
class NativeRayProposal:
    proposal_id: str
    column_index: int
    offset_source_voxels: tuple[int, int]
    tool_id: str
    entry_mm: tuple[float, float, float]
    primary_target_mm: tuple[float, float, float]
    fallback_target_mm: tuple[float, float, float] | None
    fallback_condition: str = "primary_preview_rejected"


@dataclass(frozen=True)
class ProposalSlot:
    """Exactly one disposition for each declared column × tool combination."""

    column_index: int
    offset_source_voxels: tuple[int, int]
    tool_id: str
    reason: str
    proposal_id: str | None = None


@dataclass(frozen=True)
class AxisProposalBatch:
    source_hash: str
    engine_model_hash: str
    rule_hash: str
    proposal_model_hash: str
    cavity_state_hash: str
    proposals: tuple[NativeRayProposal, ...]
    ledger: tuple[ProposalSlot, ...]
    unsupported_reason: str | None = None

    @property
    def counts(self) -> dict[str, int]:
        counts: dict[str, int] = {}
        for row in self.ledger:
            counts[row.reason] = counts.get(row.reason, 0) + 1
        return counts

    def to_dict(self) -> dict[str, Any]:
        return {**asdict(self), "counts": self.counts, "slot_count": len(self.ledger),
                "geometry_certified": False, "removal_authorized": False}


def _array_identity(array: np.ndarray) -> tuple:
    root = array
    while isinstance(root, np.ndarray) and root.base is not None:
        root = root.base
    if array.flags.writeable or not isinstance(root, bytes):
        raise ValueError("Proposal source arrays require immutable bytes-backed native configuration")
    return (id(array), id(root), array.shape, array.dtype.str, array.strides,
            array.__array_interface__["data"][0])


def _source_identity(config: NativeResectionConfig) -> tuple:
    return (config.fingerprint, config.source_hash, config.case_id, config.tissue_support_provenance,
            config.max_tip_step_mm, config.max_microsteps,
            tuple(_array_identity(getattr(config, name)) for name in
                  ("affine", "tissue_mask", "target_labels", "hard_exclusion")),
            *((_array_identity(config.interaction_domain),) if config.interaction_domain is not None else ()),
            _array_identity(config.access.center_mm), _array_identity(config.access.normal_inward),
            config.access.radius_mm, config.access.window_id,
            tuple(tuple(asdict(tool).items()) for tool in config.tools))


def _history_cells(value: Any, shape: tuple[int, ...]) -> np.ndarray:
    array = np.asarray(value)
    if array.size == 0:
        return np.empty((0, 3), dtype=np.int64)
    if (array.ndim != 2 or array.shape[1] != 3 or array.dtype.kind not in "iu"
            or np.any(array < 0) or np.any(array >= np.asarray(shape))):
        raise RuntimeError("Native history contains invalid source-cell coordinates")
    return np.asarray(array, dtype=np.int64)


def _verify_cavity(engine: NativeResectionEngine) -> str:
    """Verify V2 ancestry and all mutable masks, without recertifying geometry.

    The engine stores a digest of each full history record plus its native array
    descriptors. Reconstruct that same digest, then derive the live masks from
    the authenticated history. Cached ancestry alone cannot detect direct edits
    to the engine's mutable masks. No successful verification is cached here.
    """
    config = engine.config
    shape = config.tissue_mask.shape
    expected_removed = np.zeros(shape, bool)
    expected_contact = np.zeros(shape, bool)
    chain = "sha256:" + sha256((config.fingerprint + ":initial").encode()).hexdigest()
    if type(engine.revision) is not int or engine.revision != len(engine.history):
        raise RuntimeError("Native cavity revision differs from its committed history")
    try:
        for record in engine.history:
            if (record["source_state_hash"] != chain or record["source_hash"] != config.source_hash
                    or record["decision_model_hash"] != config.fingerprint
                    or tuple(record["source_shape"]) != shape
                    or not np.array_equal(record["native_affine"], config.affine)):
                raise RuntimeError("Native committed history does not bind this source and ancestry")
            removed = _history_cells(record["removed_indices_native"], shape)
            contacted = _history_cells(record["contact_indices_native"], shape)
            micro_removed = [_history_cells(step["removed_indices_native"], shape) for step in record["microsteps"]]
            micro_contact = [_history_cells(step["contact_indices_native"], shape) for step in record["microsteps"]]
            # This is exactly NativeResectionEngine._result_digest's V2 schema:
            # macro removed/contact, float64 affine, then micro removed/contact.
            descriptors = [(str(removed.dtype), removed.shape), (str(contacted.dtype), contacted.shape),
                           (str(config.affine.dtype), config.affine.shape)]
            descriptors.extend((str(array.dtype), array.shape) for array in (*micro_removed, *micro_contact))
            history_digest = sha256(json.dumps({**record, "array_descriptors": descriptors},
                sort_keys=True, allow_nan=False).encode()).hexdigest()
            chain = "sha256:" + sha256((chain + ":committed:" + history_digest).encode()).hexdigest()
            if np.any(expected_removed[tuple(removed.T)]) or np.any(~config.tissue_mask[tuple(removed.T)]):
                raise RuntimeError("Native committed removal is duplicated or outside source tissue")
            expected_removed[tuple(removed.T)] = True
            expected_contact[tuple(contacted.T)] = True
    except (KeyError, TypeError, ValueError, IndexError) as error:
        raise RuntimeError("Native committed history is malformed or changed") from error
    if chain != engine.state_hash:
        raise RuntimeError("Native history no longer matches the cached cavity ancestry")
    expected_remaining = config.tissue_mask & ~expected_removed
    blocked = (expected_remaining if config.interaction_domain is None else
               expected_remaining | ~config.interaction_domain)
    expected_free = ~binary_fill_holes(blocked)
    for name, expected in (("remaining_mask", expected_remaining), ("removed_mask", expected_removed),
                           ("contact_mask", expected_contact), ("connected_free_mask", expected_free)):
        actual = getattr(engine, name)
        if (not isinstance(actual, np.ndarray) or actual.dtype != np.bool_
                or actual.shape != shape or not np.array_equal(actual, expected)):
            raise RuntimeError(f"Native {name} differs from its legitimate committed history")
    return chain


@dataclass(frozen=True)
class _Column:
    offset: tuple[int, int]
    transverse_indices: tuple[int, int] | None
    entry_mm: tuple[float, float, float] | None
    aperture_fits: tuple[bool, ...]


class PreparedAxisColumnProposer:
    """Cache only immutable source transforms, entries and aperture filters.

    A provider belongs to one native configuration object; engine clones sharing
    that frozen configuration may use it. No cavity or preview result is cached.
    Labels and remaining tissue are read again on every call to ``propose``.
    """

    def __init__(self, native_config: NativeResectionConfig,
                 config: AxisColumnProposalConfig | None = None):
        if not isinstance(native_config, NativeResectionConfig):
            raise TypeError("A validated NativeResectionConfig is required")
        if NATIVE_RESECTION_VERSION != _HISTORY_ENGINE_VERSION:
            raise ValueError("Proposal ancestry validation needs a reviewed native-engine history schema")
        # A frozen dataclass can still have been forcibly replaced before this
        # provider existed. Verify content once rather than trusting its cached
        # fingerprint. The temporary validated copy is discarded immediately.
        if replace(native_config).fingerprint != native_config.fingerprint:
            raise ValueError("Native source contents no longer match their cached configuration fingerprint")
        self._native = native_config
        self._config = config or AxisColumnProposalConfig()
        if not isinstance(self._config, AxisColumnProposalConfig):
            raise TypeError("An AxisColumnProposalConfig is required")
        self._rule_hash = self._config.fingerprint
        self._source_identity = _source_identity(native_config)
        self._model_hash = _digest({"source_hash": native_config.source_hash,
            "engine_model_hash": native_config.fingerprint, "rule_hash": self._rule_hash})
        affine = np.asarray(native_config.affine)
        if (affine.shape != (4, 4) or not np.isfinite(affine).all()
                or not np.allclose(affine[3], (0, 0, 0, 1), rtol=0, atol=1e-12)):
            raise ValueError("Proposal affine must be finite, homogeneous and 4 × 4")
        try:
            inverse = np.linalg.inv(affine)
        except np.linalg.LinAlgError as error:
            raise ValueError("Proposal affine must be invertible") from error
        if not np.isfinite(inverse).all():
            raise ValueError("Proposal affine inverse must be finite")
        basis = affine[:3, :3]
        spacing = np.linalg.norm(basis, axis=0)
        if not np.isfinite(spacing).all() or np.any(spacing <= 0):
            raise ValueError("Proposal source spacing must be finite and positive")
        self._basis = tuple(_point(row) for row in basis)
        self._translation = _point(affine[:3, 3])
        unit = basis / spacing
        access = native_config.access
        center, normal = np.asarray(access.center_mm), np.asarray(access.normal_inward)
        if (center.shape != (3,) or normal.shape != (3,) or not np.isfinite(center).all()
                or not np.isfinite(normal).all()
                or not np.isclose(np.linalg.norm(normal), 1., rtol=0, atol=_DIRECTION_ATOL)):
            raise ValueError("Proposal access must have a finite center and unit inward normal")
        origin = inverse[:3, :3] @ center + inverse[:3, 3]
        if not np.isfinite(origin).all():
            raise ValueError("Proposal source-coordinate origin must be finite")
        orientation = unit.T @ normal
        axis = int(np.argmax(np.abs(orientation)))
        transverse = tuple(index for index in range(3) if index != axis)
        self._axis, self._transverse = axis, transverse
        self._sign = 1 if orientation[axis] >= 0 else -1
        self._origin_axis = float(origin[axis])
        self._unsupported = None
        if not np.allclose(unit.T @ unit, np.eye(3), rtol=0, atol=_DIRECTION_ATOL):
            self._unsupported = "UNSUPPORTED_SHEARED_GRID"
        elif (np.any(np.abs(orientation[list(transverse)]) > _DIRECTION_ATOL)
              or not np.isclose(abs(orientation[axis]), 1., rtol=0, atol=_DIRECTION_ATOL)):
            self._unsupported = "UNSUPPORTED_NONAXIAL_ACCESS"
        elif not np.allclose(origin[list(transverse)], np.rint(origin[list(transverse)]), rtol=0, atol=_GRID_ATOL):
            self._unsupported = "UNSUPPORTED_FRACTIONAL_TRANSVERSE_ORIGIN"
        columns = []
        for offset in self._config.offsets_source_voxels:
            indices = tuple(int(np.rint(origin[dim])) + offset[index] for index, dim in enumerate(transverse))
            if any(index < 0 or index >= native_config.tissue_mask.shape[dim] for index, dim in zip(indices, transverse)):
                columns.append(_Column(offset, None, None, ()))
                continue
            position = origin.copy()
            position[list(transverse)] = indices
            entry = basis @ position + affine[:3, 3]
            radial = float(np.linalg.norm(entry - center))
            fits = tuple(radial + tool.envelope_radius_mm <= access.radius_mm + 1e-9 for tool in native_config.tools)
            columns.append(_Column(offset, indices, _point(entry), fits))
        self._columns = tuple(columns)

    @property
    def model_hash(self) -> str:
        return self._model_hash

    @property
    def rule_hash(self) -> str:
        return self._rule_hash

    def propose(self, engine: NativeResectionEngine) -> AxisProposalBatch:
        if not isinstance(engine, NativeResectionEngine) or engine.config is not self._native:
            raise ValueError("Proposer is bound to its exact source native configuration")
        if _source_identity(self._native) != self._source_identity or self._config.fingerprint != self._rule_hash:
            raise RuntimeError("Frozen proposal source or rule changed after preparation")
        state_hash = _verify_cavity(engine)
        proposals, ledger = [], []
        basis, translation = np.asarray(self._basis), np.asarray(self._translation)
        for column_index, column in enumerate(self._columns):
            cells = np.empty(0, dtype=np.int64)
            if self._unsupported is None and column.transverse_indices is not None:
                selector: list[Any] = [slice(None)] * 3
                for dim, value in zip(self._transverse, column.transverse_indices):
                    selector[dim] = value
                labels = self._native.target_labels[tuple(selector)]
                remaining = engine.remaining_mask[tuple(selector)]
                cells = np.flatnonzero((labels > 0) & remaining)
                cells = cells[(cells - self._origin_axis) * self._sign > 0]
                cells = cells[np.argsort((cells - self._origin_axis) * self._sign)]
            for tool_index, tool in enumerate(self._native.tools):
                reason = self._unsupported
                if reason is None:
                    if column.transverse_indices is None:
                        reason = "COLUMN_OUT_OF_IMAGE"
                    elif not len(cells):
                        reason = "NO_REMAINING_TARGET_IN_COLUMN"
                    elif not column.aperture_fits[tool_index]:
                        reason = "FULL_TOOL_APERTURE_PREFILTER"
                    elif len(proposals) >= self._config.max_primary_rays:
                        reason = "PRIMARY_CANDIDATE_CAP"
                proposal_id = None
                if reason is None:
                    endpoints = []
                    for cell in dict.fromkeys((int(cells[-1]), int(cells[0]))):
                        position = np.empty(3, dtype=float)
                        position[self._axis] = cell
                        position[list(self._transverse)] = column.transverse_indices
                        endpoints.append(_point(basis @ position + translation))
                    primary = endpoints[0]
                    fallback = endpoints[1] if len(endpoints) > 1 else None
                    proposal_id = "axis-column-" + _digest({"model": self._model_hash,
                        "cavity": state_hash, "column": column_index, "tool_id": tool.tool_id,
                        "entry_mm": column.entry_mm, "primary_target_mm": primary,
                        "fallback_target_mm": fallback}).split(":", 1)[1][:24]
                    proposals.append(NativeRayProposal(proposal_id, column_index, column.offset,
                        tool.tool_id, column.entry_mm, primary, fallback))
                    reason = "PROPOSED_UNCERTIFIED"
                ledger.append(ProposalSlot(column_index, column.offset, tool.tool_id, reason, proposal_id))
        return AxisProposalBatch(self._native.source_hash, self._native.fingerprint, self._rule_hash,
            self._model_hash, state_hash, tuple(proposals), tuple(ledger), self._unsupported)

    def validate_batch(self, batch: AxisProposalBatch, engine: NativeResectionEngine) -> None:
        """Reject a stale or altered ledger; this still grants no tool clearance."""
        if not isinstance(batch, AxisProposalBatch) or batch != self.propose(engine):
            raise ValueError("Proposal batch is stale, altered or bound to a different model")


NOMINAL_CAVITY_PROPOSAL_VERSION = "permitted-nominal-cavity-columns-v1"
NOMINAL_CAVITY_FAMILIES = ("exposed_opening", "proximal_nominal", "distal_nominal")
INTERMEDIATE_OPENING_VERSION = "permitted-nominal-cavity-intermediate-opening-v1"
INTERMEDIATE_OPENING_FAMILY = "intermediate_opening"
TOOL_FOOTPRINT_OPENING_VERSION = "permitted-nominal-cavity-tool-footprint-opening-v1"
TOOL_FOOTPRINT_OPENING_FAMILY = "tool_footprint_opening"
OBSTRUCTION_OPENING_VERSION = "permitted-first-obstruction-opening-v1"
OBSTRUCTION_OPENING_FAMILY = "obstruction_opening"
OBSTRUCTION_CELLS_PER_PREVIEW = 16
OBSTRUCTION_SELECTED_CELLS = 16


@dataclass(frozen=True)
class NominalCavityProposalConfig:
    """A bounded family ledger, never all possible tool paths.

    Optional requested 1 mm lookahead is rounded to a positive source-cell
    count, not an exact physical advance. Coarse cells can exceed the request;
    every extra slot records its anchor, endpoint and actual physical advance.
    Footprint-enabled callers may explicitly raise the shared cap to 120;
    the default and all other configurations retain the 96-candidate limit.
    """
    offsets_source_voxels: tuple[tuple[int, int], ...] = DEFAULT_COLUMN_OFFSETS
    max_candidates: int = 96
    nominal_min_membership: float = 0.
    intermediate_opening_mm: float | None = None
    tool_footprint_opening: bool = False
    obstruction_opening: bool = False

    def __post_init__(self):
        offsets = AxisColumnProposalConfig(self.offsets_source_voxels).offsets_source_voxels
        if type(self.tool_footprint_opening) is not bool:
            raise ValueError("Tool-footprint opening must be an explicit bool")
        if type(self.obstruction_opening) is not bool:
            raise ValueError("Obstruction opening must be an explicit bool")
        limit = 120 if self.tool_footprint_opening else 96
        if type(self.max_candidates) is not int or not 1 <= self.max_candidates <= limit:
            raise ValueError(f"Nominal/cavity candidate cap must be between one and{limit}")
        value = self.nominal_min_membership
        if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, float, np.integer, np.floating)) or not np.isfinite(value) or not 0 <= value < 1:
            raise ValueError("Nominal membership threshold must be a declared finite value in [0,1)")
        object.__setattr__(self, "offsets_source_voxels", offsets)
        object.__setattr__(self, "nominal_min_membership", float(value))
        advance = self.intermediate_opening_mm
        if advance is not None:
            if isinstance(advance, (bool, np.bool_)) or not isinstance(advance, (int, float, np.integer, np.floating)) or float(advance) != 1.:
                raise ValueError("Intermediate opening is an explicit fixed 1 mm option")
            object.__setattr__(self, "intermediate_opening_mm", 1.)

    @property
    def families(self):
        return (NOMINAL_CAVITY_FAMILIES
            + (() if self.intermediate_opening_mm is None else (INTERMEDIATE_OPENING_FAMILY,))
            + ((TOOL_FOOTPRINT_OPENING_FAMILY,) if self.tool_footprint_opening else ())
            + ((OBSTRUCTION_OPENING_FAMILY,) if self.obstruction_opening else ()))

    @property
    def version(self):
        if self.obstruction_opening:
            return OBSTRUCTION_OPENING_VERSION
        if self.tool_footprint_opening:
            return TOOL_FOOTPRINT_OPENING_VERSION
        return NOMINAL_CAVITY_PROPOSAL_VERSION if self.intermediate_opening_mm is None else INTERMEDIATE_OPENING_VERSION

    @property
    def fingerprint(self):
        record = asdict(self)
        record.pop("intermediate_opening_mm")
        record.pop("tool_footprint_opening")
        record.pop("obstruction_opening")
        legacy = {"version": NOMINAL_CAVITY_PROPOSAL_VERSION, **record,
            "families": NOMINAL_CAVITY_FAMILIES, "order": "column_tool_family",
            "deduplication": "identical_tool_entry_tip", "crop_clipping": False,
            "opening": "nearest_remaining_tissue_with_face_adjacent_to_connected_free"}
        if self.intermediate_opening_mm is not None:
            legacy.update(version=self.version, families=self.families,
                order="all_original_column_tool_families_then_intermediate_column_tool",
                intermediate_opening_mm=self.intermediate_opening_mm,
                advance_rule="nearest_positive_source_axis_voxel_count; numpy_rint_ties_to_even; minimum_one",
                endpoint_rule="actual_source_voxel_center; physical_increment_recorded; native_preview_required")
        if self.tool_footprint_opening:
            legacy.update(version=self.version, families=self.families,
                tool_footprint_opening=True,
                order="all_existing_families_then_tool_footprint_column_tool",
                footprint_rule="first_positive_source_center_endpoint_whose_active_sweep_fully_contains_a_prior_face_exposed_remaining_cell",
                footprint_scope="positive_depth_public_tissue_and_authenticated_cavity; all_eight_native_corners; no_preview_or_target_ranking",
                footprint_containment_tolerance_mm=1e-10,
                footprint_budget="one_slot_per_column_tool; shared_max_candidates; explicit_capped_dispositions")
        if self.obstruction_opening:
            legacy.update(version=self.version, families=self.families, obstruction_opening=True,
                order="entire_base_prefix_then_public_nominal_first_native_cell_order_then_tool_order",
                obstruction_evidence="ordinary_base_previews_first_rejected_shaft_interval_only; no_recursive_previews",
                obstruction_cells_per_preview=OBSTRUCTION_CELLS_PER_PREVIEW,
                obstruction_selected_cells=OBSTRUCTION_SELECTED_CELLS,
                obstruction_priority="permitted_public_nominal_membership_above_declared_threshold_then_native_index",
                obstruction_rule="uncovered_eight_corner_radial_bound_across_declared_columns_and_existing_tips; cell_center_endpoint",
                obstruction_budget="shared_max_candidates; bounded_evidence_is_incomplete_discovery; explicit_all_retained_cell_tool_dispositions")
        return semantic_digest(legacy)

    def to_record(self):
        """Preserve historical protocol records; only an enabled rule adds a key."""
        record = asdict(self)
        if not self.obstruction_opening:
            record.pop("obstruction_opening")
        return record


@dataclass(frozen=True)
class NominalCavityRay:
    proposal_id: str
    family: str
    column_index: int
    offset_source_voxels: tuple[int, int]
    tool_id: str
    voxel: tuple[int, int, int]
    entry_mm: tuple[float, float, float]
    tip_mm: tuple[float, float, float]


@dataclass(frozen=True)
class NominalCavitySlot:
    column_index: int
    offset_source_voxels: tuple[int, int]
    tool_id: str
    family: str
    reason: str
    proposal_id: str | None = None
    voxel: tuple[int, int, int] | None = None


@dataclass(frozen=True)
class IntermediateOpeningSlot(NominalCavitySlot):
    opening_anchor_voxel: tuple[int, int, int] | None = None
    requested_opening_advance_mm: float = 1.
    advance_source_axis_voxels: int = 0
    actual_opening_advance_mm: float = 0.


@dataclass(frozen=True)
class ToolFootprintOpeningSlot(NominalCavitySlot):
    footprint_anchor_voxel: tuple[int, int, int] | None = None
    footprint_exposed_cells: int = 0
    footprint_radially_eligible_cells: int = 0
    footprint_endpoint_tests: int = 0


@dataclass(frozen=True)
class ObstructionOpeningSlot(NominalCavitySlot):
    blocker_voxel: tuple[int, int, int] | None = None
    public_nominal_priority: bool = False
    original_axis_min_corner_bound_mm: float | None = None
    evidence_hashes: tuple[str, ...] = ()


@dataclass(frozen=True)
class NominalCavityBatch:
    model_hash: str
    cavity_state_hash: str
    nominal_target_hash: str
    nominal_provenance_hash: str
    proposals: tuple[NominalCavityRay, ...]
    ledger: tuple[NominalCavitySlot, ...]

    @property
    def obstruction_accounting(self):
        return getattr(self, "_obstruction_accounting", None)

    def to_dict(self):
        counts = {}
        for row in self.ledger:
            counts[row.reason] = counts.get(row.reason, 0) + 1
        return {**asdict(self), "counts": counts, "slot_count": len(self.ledger),
            **({"obstruction_accounting": thaw_json(self.obstruction_accounting)}
               if self.obstruction_accounting is not None else {}),
            "emitted_count": len(self.proposals), "geometry_certified": False,
            "scope": ("base_columns_plus_bounded_first_obstruction_axes_not_all_paths"
                      if self.obstruction_accounting is not None else
                      "declared_columns_and_tool_footprint_endpoint_family_not_all_paths"
                      if any(row.family == TOOL_FOOTPRINT_OPENING_FAMILY for row in self.ledger)
                      else "declared_columns_and_original_plus_intermediate_endpoint_families_not_all_paths"
                      if any(row.family == INTERMEDIATE_OPENING_FAMILY for row in self.ledger)
                      else "declared_columns_and_three_endpoint_families_not_all_paths")}


class PreparedNominalCavityProposer:
    """Propose from explicit permitted evidence and authenticated observed state.

    Engine target labels must stay zero. Neither private references nor preview
    outcomes select endpoints. Every emitted ray still needs a native preview.
    """
    def __init__(self, native_config: NativeResectionConfig, nominal_target: np.ndarray, *,
                 nominal_provenance: Mapping, config: NominalCavityProposalConfig | None = None,
                 index_affine=None, index_frame_record: Mapping | None = None,
                 target_semantics: str = TARGET_WITHIN_SUPPORT):
        if not isinstance(native_config, NativeResectionConfig):
            raise TypeError("A validated native configuration is required")
        if NATIVE_RESECTION_VERSION != _HISTORY_ENGINE_VERSION:
            raise ValueError("Nominal/cavity proposals require the audited native history schema")
        if replace(native_config).fingerprint != native_config.fingerprint:
            raise ValueError("Native source changed before proposal preparation")
        if np.any(native_config.target_labels):
            raise ValueError("Nominal/cavity proposals require zero engine target labels")
        if target_semantics not in {TARGET_WITHIN_SUPPORT, SUPPLIED_GOAL_REGION}:
            raise ValueError("Unknown explicit target semantics")
        nominal = np.asarray(nominal_target)
        if (nominal.shape != native_config.tissue_mask.shape or nominal.dtype.kind not in "biuf"
                or not np.isfinite(nominal).all() or np.any(nominal < 0) or np.any(nominal > 1)
                or (target_semantics == TARGET_WITHIN_SUPPORT
                    and np.any((nominal > 0) & ~native_config.tissue_mask))):
            raise ValueError("Permitted nominal target must be a finite source-aligned membership grid inside observed support")
        if not isinstance(nominal_provenance, Mapping):
            raise ValueError("Explicit nominal source/derivation provenance is required")
        nominal = immutable_array(nominal, np.float32)
        provenance = freeze_json(nominal_provenance)
        if (provenance.get("source_hash") != native_config.source_hash
                or provenance.get("nominal_target_hash") != array_digest(nominal)
                or provenance.get("source_kind") not in {"supplied_annotation", "derived_from_scan"}
                or not isinstance(provenance.get("derivation"), str) or not provenance["derivation"].strip()):
            raise ValueError("Nominal provenance must bind the exact permitted target and source")
        if target_semantics == SUPPLIED_GOAL_REGION and (
                provenance.get("target_semantics") != SUPPLIED_GOAL_REGION
                or provenance.get("source_kind") != "supplied_annotation"):
            raise ValueError("Supplied goal-region semantics require explicit public annotation provenance")
        rule = NominalCavityProposalConfig() if config is None else config
        if not isinstance(rule, NominalCavityProposalConfig):
            raise TypeError("A typed nominal/cavity rule is required")
        original = native_config.affine if index_affine is None else np.asarray(index_affine)
        if (original.shape != (4,4) or original.dtype.kind not in "iuf" or not np.isfinite(original).all()
                or not np.array_equal(original[3], [0,0,0,1])):
            raise ValueError("Original index-selection affine must be finite and homogeneous")
        original = immutable_array(original, np.float64)
        if index_frame_record is not None and not isinstance(index_frame_record, Mapping):
            raise ValueError("Index-frame reconciliation record must be an explicit object")
        frame_record = freeze_json({} if index_frame_record is None else index_frame_record)
        if not np.array_equal(original, native_config.affine):
            self._validate_index_frame(original, native_config, frame_record)
        self._native, self._config = native_config, rule
        self._nominal, self._provenance, self._index_affine, self._frame_record = nominal, provenance, original, frame_record
        self._source_identity = _source_identity(native_config)
        self._nominal_identity, self._index_identity = _array_identity(nominal), _array_identity(original)
        self._rule_hash, self._nominal_hash = rule.fingerprint, array_digest(nominal)
        self._provenance_hash, self._frame_hash = semantic_digest(provenance), semantic_digest(frame_record)
        self._model_hash = semantic_digest({"source_hash": native_config.source_hash,
            "engine_model_hash": native_config.fingerprint, "rule_hash": self._rule_hash,
            "nominal_hash": self._nominal_hash, "nominal_provenance_hash": self._provenance_hash,
            "index_affine_hash": array_digest(original), "index_frame_record_hash": self._frame_hash})
        basis = original[:3,:3]
        spacing = np.linalg.norm(basis, axis=0)
        if not np.isfinite(spacing).all() or np.any(spacing <= 0):
            raise ValueError("Original index frame requires positive finite spacing")
        unit = basis / spacing
        normal = native_config.access.normal_inward
        axis = int(np.argmax(np.abs(unit.T @ normal)))
        sign = 1 if unit[:,axis] @ normal >= 0 else -1
        if not np.allclose(normal, sign * unit[:,axis], rtol=0, atol=_DIRECTION_ATOL):
            raise ValueError("Nominal/cavity columns require an original source-axis-normal access")
        origin = np.linalg.solve(basis, native_config.access.center_mm - original[:3,3])
        self._axis, self._sign = axis, sign
        self._transverse = tuple(dim for dim in range(3) if dim != axis)
        self._columns = tuple(tuple(int(np.rint(origin[dim])) + offset[i]
            for i,dim in enumerate(self._transverse)) for offset in rule.offsets_source_voxels)
        self._geometry_identity = (self._axis, self._sign, self._transverse, self._columns)

    @staticmethod
    def _validate_index_frame(original, native, record):
        if (record.get("method") != "orthogonal_roundoff_1e-6mm"
                or record.get("original_affine_hash") != array_digest(original)
                or record.get("derived_affine_hash") != array_digest(native.affine)
                or tuple(record.get("shape", ())) != native.tissue_mask.shape
                or not np.array_equal(record.get("original_affine_ras_mm"), original)
                or not np.array_equal(record.get("derived_affine_ras_mm"), native.affine)
                or record.get("resampled") is not False):
            raise ValueError("Original index frame needs the exact source-bound roundoff record")
        spacing = np.linalg.norm(original[:3,:3], axis=0)
        native_spacing = np.linalg.norm(native.affine[:3,:3], axis=0)
        corners = np.array(list(product(*[(-.5,n-.5) for n in native.tissue_mask.shape])))
        displacement = np.linalg.norm(corners @ (native.affine[:3,:3] - original[:3,:3]).T, axis=1).max()
        if (not np.array_equal(original[:3,3], native.affine[:3,3])
                or np.sign(np.linalg.det(original[:3,:3])) != np.sign(np.linalg.det(native.affine[:3,:3]))
                or not np.allclose(spacing, native_spacing, rtol=64*np.finfo(float).eps, atol=0)
                or not np.isfinite(displacement) or displacement > 1e-6
                or np.abs((original[:3,:3]/spacing).T @ (original[:3,:3]/spacing)-np.eye(3)).max() > 1e-8):
            raise ValueError("Original index frame exceeds declared roundoff-only geometry")

    @property
    def model_hash(self):
        return self._model_hash

    @property
    def rule_hash(self):
        return self._rule_hash

    def _tool_footprint_endpoint(self, engine, column, tool, *, cancelled=None):
        """Select geometry from public exposed tissue; this is NOT certification.

        The active capsule swept by a straight insertion spans entry-tip_length
        to endpoint. Full native-cell containment uses the same eight-corner
        distance rule as native_resection.contained_capsule_cells. Shaft order,
        aperture, connectivity during cutting and actual removal remain solely
        the native preview's responsibility. No preview or reward is queried.
        """
        shape, affine = self._native.tissue_mask.shape, self._native.affine
        access, normal = self._native.access, self._native.access.normal_inward
        endpoints = np.zeros((shape[self._axis], 3), dtype=np.int64)
        endpoints[:, self._axis] = np.arange(shape[self._axis])
        endpoints[:, self._transverse] = column
        tips = endpoints @ affine[:3, :3].T + affine[:3, 3]
        depths = (tips - access.center_mm) @ normal
        order = np.argsort(depths, kind="stable")
        order = order[depths[order] > 0]
        if not len(order):
            return None, None, 0, 0, 0
        endpoints, tips, depths = endpoints[order], tips[order], depths[order]
        entries = tips - depths[:, None] * normal
        # Bound scratch work to the transverse footprint of every possible ray.
        # Using both ends also covers the tiny original/native-frame roundoff.
        inverse = np.linalg.inv(affine[:3, :3])
        ends = np.concatenate((entries[[0, -1]] - tool.tip_length_mm * normal,
                               tips[[0, -1]]))
        index_ends = (ends - affine[:3, 3]) @ inverse.T
        radial_index = tool.tip_radius_mm * np.linalg.norm(inverse, axis=1)
        lower, upper = np.zeros(3, int), np.asarray(shape).copy()
        for dim in self._transverse:
            lower[dim] = max(0, int(np.floor(index_ends[:, dim].min() - radial_index[dim] - .5)))
            upper[dim] = min(shape[dim], int(np.ceil(index_ends[:, dim].max() + radial_index[dim] + .5)) + 1)
        region = tuple(slice(lo, hi) for lo, hi in zip(lower, upper))
        cells = np.argwhere(engine.remaining_mask[region]) + lower
        centers = cells @ affine[:3, :3].T + affine[:3, 3]
        cells = cells[(centers - access.center_mm) @ normal > 0]
        exposed = np.zeros(len(cells), bool)
        for dim in range(3):
            for sign in (-1, 1):
                neighbor = cells.copy(); neighbor[:, dim] += sign
                inside = np.all((neighbor >= 0) & (neighbor < shape), axis=1)
                exposed |= ~inside
                exposed[inside] |= engine.connected_free_mask[tuple(neighbor[inside].T)]
        cells = cells[exposed]
        exposed_count = len(cells)
        if not exposed_count:
            return None, None, 0, 0, 0
        corners = ((cells @ affine[:3, :3].T + affine[:3, 3])[:, None, :]
            + np.asarray(tuple(product((-.5, .5), repeat=3))) @ affine[:3, :3].T)
        radius = tool.tip_radius_mm - 1e-10
        # Necessary-only radial test: each corner must fit around at least one
        # candidate entry. It cannot award containment or preview feasibility.
        projected = corners - (((corners - access.center_mm) @ normal)[..., None] * normal)
        possible = np.all(point_segment_distances(projected, entries[0], entries[-1]) <= radius, axis=1)
        cells, corners = cells[possible], corners[possible]
        eligible_count = len(cells)
        if not eligible_count:
            return None, None, exposed_count, 0, 0
        for count, (endpoint, tip, entry) in enumerate(zip(endpoints, tips, entries), 1):
            if cancelled is not None and cancelled():
                raise InterruptedError("Tool-footprint opening preparation cancelled")
            contained = np.all(point_segment_distances(corners,
                entry - tool.tip_length_mm * normal, tip) <= radius, axis=1)
            matches = np.flatnonzero(contained)
            if len(matches):
                return (tuple(int(v) for v in endpoint),
                    tuple(int(v) for v in cells[matches[0]]), exposed_count, eligible_count, count)
        return None, None, exposed_count, eligible_count, len(endpoints)

    def propose(self, engine: NativeResectionEngine, *, cancelled=None) -> NominalCavityBatch:
        if not isinstance(engine, NativeResectionEngine) or engine.config is not self._native:
            raise ValueError("Nominal/cavity provider is bound to its exact native source")
        if (_source_identity(self._native) != self._source_identity
                or _array_identity(self._nominal) != self._nominal_identity
                or _array_identity(self._index_affine) != self._index_identity
                or self._config.fingerprint != self._rule_hash
                or semantic_digest(self._provenance) != self._provenance_hash
                or semantic_digest(self._frame_record) != self._frame_hash
                or (self._axis,self._sign,self._transverse,self._columns) != self._geometry_identity):
            raise RuntimeError("Frozen nominal/cavity source, evidence or proposal geometry changed")
        cavity_hash = _verify_cavity(engine)
        shape, affine = self._native.tissue_mask.shape, self._native.affine
        access = self._native.access
        rays, ledger, seen = [], [], {}
        opening_anchors = []
        for column_index, (offset, column) in enumerate(zip(self._config.offsets_source_voxels, self._columns)):
            if cancelled is not None and cancelled():
                raise InterruptedError("Nominal/cavity proposal preparation cancelled")
            candidates = {}
            outside = any(value < 0 or value >= shape[dim] for dim,value in zip(self._transverse,column))
            if not outside:
                selector = [slice(None)]*3
                for dim,value in zip(self._transverse,column): selector[dim] = value
                remaining = engine.remaining_mask[tuple(selector)]
                cells = np.flatnonzero(remaining)
                points = np.zeros((len(cells),3), dtype=np.int64)
                points[:,self._axis] = cells
                points[:,self._transverse] = column
                depths = (points @ affine[:3,:3].T + affine[:3,3] - access.center_mm) @ access.normal_inward
                order = np.argsort(depths, kind="stable")
                points = points[order[depths[order] > 0]]
                exposed = np.zeros(len(points), bool)
                for dim in range(3):
                    for sign in (-1,1):
                        neighbor = points.copy(); neighbor[:,dim] += sign
                        inside = np.all((neighbor >= 0) & (neighbor < shape), axis=1)
                        exposed |= ~inside
                        exposed[inside] |= engine.connected_free_mask[tuple(neighbor[inside].T)]
                exposed_indices = np.flatnonzero(exposed)
                if len(exposed_indices):
                    candidates["exposed_opening"] = tuple(int(v) for v in points[exposed_indices[0]])
                nominal = points[self._nominal[tuple(points.T)] > self._config.nominal_min_membership]
                if len(nominal):
                    candidates["proximal_nominal"] = tuple(int(v) for v in nominal[0])
                    candidates["distal_nominal"] = tuple(int(v) for v in nominal[-1])
            if self._config.intermediate_opening_mm is not None:
                opening_anchors.append((column_index, offset, candidates.get("exposed_opening"), outside))
            for tool in self._native.tools:
                for family in NOMINAL_CAVITY_FAMILIES:
                    voxel = candidates.get(family)
                    identifier = None
                    reason = "COLUMN_OUT_OF_IMAGE" if outside else ("NO_EXPOSED_REMAINING_TISSUE" if family == "exposed_opening" else "NO_REMAINING_NOMINAL_TARGET")
                    if voxel is not None:
                        tip = affine[:3,:3] @ voxel + affine[:3,3]
                        depth = float((tip-access.center_mm) @ access.normal_inward)
                        entry = tip-depth*access.normal_inward
                        key = (tool.tool_id, _point(entry), _point(tip))
                        if key in seen:
                            identifier, reason = seen[key], "DUPLICATE_GEOMETRY"
                        elif len(rays) >= self._config.max_candidates:
                            reason = "CANDIDATE_CAP"
                        else:
                            identifier = "nominal-cavity-" + semantic_digest({"model": self._model_hash,
                                "cavity": cavity_hash, "geometry": key}).split(":",1)[1][:24]
                            rays.append(NominalCavityRay(identifier, family, column_index, offset,
                                tool.tool_id, voxel, key[1], key[2]))
                            seen[key], reason = identifier, "PROPOSED_UNCERTIFIED"
                    ledger.append(NominalCavitySlot(column_index, offset, tool.tool_id, family, reason, identifier, voxel))
        # Append only after every original slot: the extra family can never
        # displace a legacy candidate at the shared declared candidate cap.
        if self._config.intermediate_opening_mm is not None:
            spacing = float(np.linalg.norm(affine[:3, self._axis]))
            count = max(1, int(np.rint(self._config.intermediate_opening_mm / spacing)))
            advance_mm = count * spacing
            for column_index, offset, anchor, outside in opening_anchors:
                if cancelled is not None and cancelled():
                    raise InterruptedError("Intermediate opening preparation cancelled")
                endpoint = None
                if anchor is not None:
                    shifted = list(anchor); shifted[self._axis] += self._sign * count
                    endpoint = tuple(shifted)
                for tool in self._native.tools:
                    identifier = None
                    reason = "COLUMN_OUT_OF_IMAGE" if outside else "NO_EXPOSED_REMAINING_TISSUE"
                    if endpoint is not None:
                        if not all(0 <= v < n for v, n in zip(endpoint, shape)):
                            reason = "ENDPOINT_OUT_OF_IMAGE"
                        else:
                            tip = affine[:3, :3] @ endpoint + affine[:3, 3]
                            depth = float((tip-access.center_mm) @ access.normal_inward)
                            entry = tip-depth*access.normal_inward
                            key = (tool.tool_id, _point(entry), _point(tip))
                            if key in seen:
                                identifier, reason = seen[key], "DUPLICATE_GEOMETRY"
                            elif len(rays) >= self._config.max_candidates:
                                reason = "CANDIDATE_CAP"
                            else:
                                identifier = "nominal-cavity-" + semantic_digest({"model": self._model_hash,
                                    "cavity": cavity_hash, "geometry": key}).split(":", 1)[1][:24]
                                rays.append(NominalCavityRay(identifier, INTERMEDIATE_OPENING_FAMILY,
                                    column_index, offset, tool.tool_id, endpoint, key[1], key[2]))
                                seen[key], reason = identifier, "PROPOSED_UNCERTIFIED"
                    ledger.append(IntermediateOpeningSlot(column_index, offset, tool.tool_id,
                        INTERMEDIATE_OPENING_FAMILY, reason, identifier, endpoint, anchor,
                        self._config.intermediate_opening_mm, count, advance_mm))
        if self._config.tool_footprint_opening:
            for column_index, (offset, column) in enumerate(zip(self._config.offsets_source_voxels, self._columns)):
                outside = any(v < 0 or v >= shape[dim] for dim, v in zip(self._transverse, column))
                for tool in self._native.tools:
                    if cancelled is not None and cancelled():
                        raise InterruptedError("Tool-footprint opening preparation cancelled")
                    endpoint, anchor, exposed, eligible, tests = ((None, None, 0, 0, 0) if outside else
                        self._tool_footprint_endpoint(engine, column, tool, cancelled=cancelled))
                    identifier = None
                    reason = "COLUMN_OUT_OF_IMAGE" if outside else "NO_FULLY_CONTAINABLE_EXPOSED_FOOTPRINT_CELL"
                    if endpoint is not None:
                        tip = affine[:3, :3] @ endpoint + affine[:3, 3]
                        depth = float((tip-access.center_mm) @ access.normal_inward)
                        entry = tip-depth*access.normal_inward
                        key = (tool.tool_id, _point(entry), _point(tip))
                        if key in seen:
                            identifier, reason = seen[key], "DUPLICATE_GEOMETRY"
                        elif len(rays) >= self._config.max_candidates:
                            reason = "CANDIDATE_CAP"
                        else:
                            identifier = "nominal-cavity-" + semantic_digest({"model": self._model_hash,
                                "cavity": cavity_hash, "geometry": key}).split(":", 1)[1][:24]
                            rays.append(NominalCavityRay(identifier, TOOL_FOOTPRINT_OPENING_FAMILY,
                                column_index, offset, tool.tool_id, endpoint, key[1], key[2]))
                            seen[key], reason = identifier, "PROPOSED_UNCERTIFIED"
                    ledger.append(ToolFootprintOpeningSlot(column_index, offset, tool.tool_id,
                        TOOL_FOOTPRINT_OPENING_FAMILY, reason, identifier, endpoint, anchor,
                        exposed, eligible, tests))
        return NominalCavityBatch(self._model_hash, cavity_hash, self._nominal_hash,
            self._provenance_hash, tuple(rays), tuple(ledger))

    def append_obstruction_openings(self, batch, engine, results, *, cancelled=None):
        """Append public blocker-centered rays using the already paid base previews.

        No preview, reward or reference label is queried here. A blocker is an
        actual prior-remaining cell from the first rejected shaft interval, not
        a complete corridor. Bounded retained prefixes imply incomplete discovery.
        Every appended ray still requires its own unchanged native certification.
        """
        if not self._config.obstruction_opening:
            raise ValueError("Obstruction opening requires its explicit proposal rule")
        if (engine.config is not self._native or _source_identity(self._native) != self._source_identity
                or _array_identity(self._nominal) != self._nominal_identity
                or _array_identity(self._index_affine) != self._index_identity
                or self._config.fingerprint != self._rule_hash
                or semantic_digest(self._provenance) != self._provenance_hash
                or semantic_digest(self._frame_record) != self._frame_hash
                or (self._axis, self._sign, self._transverse, self._columns) != self._geometry_identity):
            raise ValueError("Obstruction proposals require the exact frozen public source")
        cavity = _verify_cavity(engine)
        if (type(batch) is not NominalCavityBatch or batch.obstruction_accounting is not None
                or batch.model_hash != self._model_hash or batch.cavity_state_hash != cavity
                or batch.nominal_target_hash != self._nominal_hash
                or batch.nominal_provenance_hash != self._provenance_hash
                or len(results) != len(batch.proposals)
                or len(results) > self._config.max_candidates
                or any(r.family == OBSTRUCTION_OPENING_FAMILY for r in batch.proposals)):
            raise ValueError("One exact base batch and its ordered preview results are required")
        affine, shape = self._native.affine, self._native.tissue_mask.shape
        access, normal = self._native.access, self._native.access.normal_inward
        frame_hash = array_digest(affine)
        tools = {t.tool_id: t for t in self._native.tools}
        cells, evidence = {}, []
        for ray, sidecar in zip(batch.proposals, results):
            if cancelled is not None and cancelled():
                raise InterruptedError("Obstruction evidence processing cancelled")
            if sidecar is None:
                continue
            record = thaw_json(sidecar)
            fingerprint = record.pop("fingerprint")
            retained = np.asarray(record["blocked_indices_native"])
            total, count = record["blocked_cell_count"], record["retained_cell_count"]
            if (semantic_digest(record) != fingerprint
                    or record["version"] != "native-first-shaft-obstruction-v1"
                    or record["reason"] != "SHAFT_BLOCKED_BY_REMAINING_NATIVE_TISSUE"
                    or record["source_hash"] != self._native.source_hash
                    or record["source_state_hash"] != cavity
                    or record["decision_model_hash"] != self._native.fingerprint
                    or record["native_affine_hash"] != frame_hash
                    or tuple(record["source_shape"]) != shape
                    or record["tool"] != asdict(tools[ray.tool_id])
                    or tuple(record["entry_mm"]) != ray.entry_mm
                    or tuple(record["requested_tip_mm"]) != ray.tip_mm
                    or record["interaction_mode"] != "aspirate"
                    or record["cell_limit"] != OBSTRUCTION_CELLS_PER_PREVIEW
                    or type(total) is not int or type(count) is not int
                    or not 1 <= count <= OBSTRUCTION_CELLS_PER_PREVIEW or total < count
                    or count != min(total, OBSTRUCTION_CELLS_PER_PREVIEW)
                    or record["truncated"] is not (total > count)
                    or record["complete_first_failure_set"] is not (total == count)
                    or retained.shape != (count, 3) or retained.dtype.kind not in "iu"
                    or np.any(retained < 0) or np.any(retained >= np.asarray(shape))
                    or not np.all(engine.remaining_mask[tuple(retained.T)])):
                raise ValueError("Malformed or stale bounded obstruction sidecar")
            if total == count and array_digest(np.asarray(retained, np.int64)) != record["blocked_indices_hash"]:
                raise ValueError("Complete obstruction cell digest differs")
            evidence.append({"base_proposal_id": ray.proposal_id, "diagnostic_hash": fingerprint,
                "blocked_cell_count": total, "retained_cell_count": count, "truncated": total > count})
            for cell in retained:
                cells.setdefault(tuple(int(v) for v in cell), set()).add(fingerprint)

        # Original column axes may differ by tiny source/native reconciliation.
        # The segment between their two extreme projected entries bounds that
        # variation; failure of this necessary test proves radial noncontainment
        # for any endpoint on that column, without pretending shaft clearance.
        corners_offset = np.asarray(tuple(product((-.5, .5), repeat=3))) @ affine[:3, :3].T
        entries = []
        for column in self._columns:
            if any(v < 0 or v >= shape[d] for d, v in zip(self._transverse, column)):
                continue
            ends = np.zeros((2, 3)); ends[:, self._axis] = (0, shape[self._axis]-1)
            ends[:, self._transverse] = column
            world = ends @ affine[:3, :3].T + affine[:3, 3]
            entries.append(world - ((world-access.center_mm) @ normal)[:, None] * normal)
        max_radius = max(t.tip_radius_mm for t in self._native.tools) - 1e-10
        origin = np.linalg.solve(self._index_affine[:3, :3], access.center_mm-self._index_affine[:3, 3])
        rays, ledger = list(batch.proposals), list(batch.ledger)
        seen = {(r.tool_id, r.entry_mm, r.tip_mm): r.proposal_id for r in rays}
        ordered = sorted(cells, key=lambda c: (not bool(self._nominal[c] > self._config.nominal_min_membership), c))
        selected = selection_omitted = 0
        for ordinal, cell in enumerate(ordered):
            if cancelled is not None and cancelled():
                raise InterruptedError("Obstruction opening preparation cancelled")
            tip = affine[:3, :3] @ cell + affine[:3, 3]
            depth = float((tip-access.center_mm) @ normal)
            entry = tip-depth*normal
            corners = tip + corners_offset
            projected = corners - (((corners-access.center_mm) @ normal)[:, None] * normal)
            bound = min((float(point_segment_distances(projected, pair[0], pair[1]).max())
                         for pair in entries), default=None)
            reason = ("BLOCKER_NOT_INWARD" if depth <= 0 else
                      "ORIGINAL_AXIS_RADIAL_NONCONTAINMENT_NOT_ESTABLISHED" if bound is not None and bound <= max_radius else
                      "OBSTRUCTION_SELECTION_CAP" if selected >= OBSTRUCTION_SELECTED_CELLS else None)
            if reason is None:
                selected += 1
            elif reason == "OBSTRUCTION_SELECTION_CAP":
                selection_omitted += 1
            offset = tuple(cell[d]-int(np.rint(origin[d])) for d in self._transverse)
            priority = bool(self._nominal[cell] > self._config.nominal_min_membership)
            for tool in self._native.tools:
                identifier, disposition = None, reason
                key = (tool.tool_id, _point(entry), _point(tip))
                if disposition is None:
                    if key in seen:
                        identifier, disposition = seen[key], "DUPLICATE_GEOMETRY"
                    elif len(rays) >= self._config.max_candidates:
                        disposition = "CANDIDATE_CAP"
                    else:
                        identifier = "nominal-cavity-" + semantic_digest({"model": self._model_hash,
                            "cavity": cavity, "geometry": key}).split(":", 1)[1][:24]
                        rays.append(NominalCavityRay(identifier, OBSTRUCTION_OPENING_FAMILY,
                            len(self._columns)+ordinal, offset, tool.tool_id, cell, key[1], key[2]))
                        seen[key], disposition = identifier, "PROPOSED_UNCERTIFIED"
                ledger.append(ObstructionOpeningSlot(len(self._columns)+ordinal, offset, tool.tool_id,
                    OBSTRUCTION_OPENING_FAMILY, disposition, identifier, cell, cell, priority, bound,
                    tuple(sorted(cells[cell]))))
        accounting = {"base_preview_calls": len(results), "added_preview_calls": len(rays)-len(batch.proposals),
            "repeated_preview_calls": 0, "recursive_expansion": False,
            "cells_per_preview_cap": OBSTRUCTION_CELLS_PER_PREVIEW, "selected_cells_cap": OBSTRUCTION_SELECTED_CELLS,
            "first_failure_evidence": evidence, "retained_unique_cells": len(cells),
            "selected_cells": selected, "selection_omitted_cells": selection_omitted,
            "unretained_blocker_mentions": sum(r["blocked_cell_count"]-r["retained_cell_count"] for r in evidence),
            "evidence_truncated": any(r["truncated"] for r in evidence),
            "discovery_complete": False,
            "scope": "bounded first rejected intervals of emitted base rays only; not all blockers or paths",
            "priority": "permitted_public_nominal_membership_then_native_index; no_reference_or_reward_queries"}
        result = NominalCavityBatch(batch.model_hash, cavity, batch.nominal_target_hash,
            batch.nominal_provenance_hash, tuple(rays), tuple(ledger))
        object.__setattr__(result, "_obstruction_accounting", freeze_json(accounting))
        return result

    def validate_batch(self, batch: NominalCavityBatch, engine: NativeResectionEngine, *, obstruction_results=None) -> None:
        """Validate current evidence/cavity dispositions, never certify geometry."""
        if not isinstance(batch, NominalCavityBatch) or batch.obstruction_accounting is None:
            if not isinstance(batch, NominalCavityBatch) or batch != self.propose(engine):
                raise ValueError("Nominal/cavity proposal batch is stale, altered or source-mismatched")
            return
        if obstruction_results is None:
            raise ValueError("Extended batch validation requires its bound original preview sidecars")
        expected = self.append_obstruction_openings(self.propose(engine), engine, obstruction_results)
        if batch != expected or batch.obstruction_accounting != expected.obstruction_accounting:
            raise ValueError("Nominal/cavity proposal batch is stale, altered or source-mismatched")
