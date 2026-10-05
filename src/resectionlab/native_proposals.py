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


AXIS_PROPOSAL_VERSION = "experimental-residual-axis-columns-v1"
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
    expected_free = ~binary_fill_holes(expected_remaining)
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


@dataclass(frozen=True)
class NominalCavityProposalConfig:
    """A complete ledger over a bounded family, never all possible tool paths."""
    offsets_source_voxels: tuple[tuple[int, int], ...] = DEFAULT_COLUMN_OFFSETS
    max_candidates: int = 96
    nominal_min_membership: float = 0.

    def __post_init__(self):
        offsets = AxisColumnProposalConfig(self.offsets_source_voxels).offsets_source_voxels
        if type(self.max_candidates) is not int or not 1 <= self.max_candidates <= 96:
            raise ValueError("Nominal/cavity candidate cap must be between one and96")
        value = self.nominal_min_membership
        if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, float, np.integer, np.floating)) or not np.isfinite(value) or not 0 <= value < 1:
            raise ValueError("Nominal membership threshold must be a declared finite value in [0,1)")
        object.__setattr__(self, "offsets_source_voxels", offsets)
        object.__setattr__(self, "nominal_min_membership", float(value))

    @property
    def fingerprint(self):
        return semantic_digest({"version": NOMINAL_CAVITY_PROPOSAL_VERSION, **asdict(self),
            "families": NOMINAL_CAVITY_FAMILIES, "order": "column_tool_family",
            "deduplication": "identical_tool_entry_tip", "crop_clipping": False,
            "opening": "nearest_remaining_tissue_with_face_adjacent_to_connected_free"})


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
class NominalCavityBatch:
    model_hash: str
    cavity_state_hash: str
    nominal_target_hash: str
    nominal_provenance_hash: str
    proposals: tuple[NominalCavityRay, ...]
    ledger: tuple[NominalCavitySlot, ...]

    def to_dict(self):
        counts = {}
        for row in self.ledger:
            counts[row.reason] = counts.get(row.reason, 0) + 1
        return {**asdict(self), "counts": counts, "slot_count": len(self.ledger),
            "emitted_count": len(self.proposals), "geometry_certified": False,
            "scope": "declared_columns_and_three_endpoint_families_not_all_paths"}


class PreparedNominalCavityProposer:
    """Propose from explicit permitted evidence and authenticated observed state.

    Engine target labels must stay zero. Neither private references nor preview
    outcomes select endpoints. Every emitted ray still needs a native preview.
    """
    def __init__(self, native_config: NativeResectionConfig, nominal_target: np.ndarray, *,
                 nominal_provenance: Mapping, config: NominalCavityProposalConfig | None = None,
                 index_affine=None, index_frame_record: Mapping | None = None):
        if not isinstance(native_config, NativeResectionConfig):
            raise TypeError("A validated native configuration is required")
        if NATIVE_RESECTION_VERSION != _HISTORY_ENGINE_VERSION:
            raise ValueError("Nominal/cavity proposals require the audited native history schema")
        if replace(native_config).fingerprint != native_config.fingerprint:
            raise ValueError("Native source changed before proposal preparation")
        if np.any(native_config.target_labels):
            raise ValueError("Nominal/cavity proposals require zero engine target labels")
        nominal = np.asarray(nominal_target)
        if (nominal.shape != native_config.tissue_mask.shape or nominal.dtype.kind not in "biuf"
                or not np.isfinite(nominal).all() or np.any(nominal < 0) or np.any(nominal > 1)
                or np.any((nominal > 0) & ~native_config.tissue_mask)):
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
        return NominalCavityBatch(self._model_hash, cavity_hash, self._nominal_hash,
            self._provenance_hash, tuple(rays), tuple(ledger))

    def validate_batch(self, batch: NominalCavityBatch, engine: NativeResectionEngine) -> None:
        """Validate current evidence/cavity dispositions, never certify geometry."""
        if not isinstance(batch, NominalCavityBatch) or batch != self.propose(engine):
            raise ValueError("Nominal/cavity proposal batch is stale, altered or source-mismatched")
