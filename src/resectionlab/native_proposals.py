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

import numpy as np
from scipy.ndimage import binary_fill_holes

from .native_resection import NATIVE_RESECTION_VERSION, NativeResectionConfig, NativeResectionEngine


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
