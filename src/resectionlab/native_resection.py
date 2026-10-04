"""Native-cell, connected surface aspiration with explicit partial contact.

The active distal capsule is a declared suction/contact abstraction. Only cells
wholly contained in its continuous swept capsule and connected to outside/cavity
are removed. Partially contacted cells remain occupied and are recorded as
exposure. The complete swept shaft must clear all remaining occupied cells.
Thus touching one coarse cell cannot erase its whole volume or clear a corridor.

This is a geometric research primitive, not tissue mechanics or clinical safety.
All arrays retain the source image grid; no interpolation or coarse erosion is
performed. Full native grids are supported because geometry queries local cells.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from hashlib import sha256
from itertools import product
import copy
import json
from typing import Any, Iterable

import numpy as np
from scipy.ndimage import binary_fill_holes

from .geometry import (
    AccessWindow, GeometryScene, ToolGeometry, ToolPose, capsule_voxel_indices,
    check_motion, point_segment_distances, _immutable,
)

NATIVE_RESECTION_VERSION = "contained-native-cell-connected-suction-v2"
_NEIGHBORS = np.array([[1, 0, 0], [-1, 0, 0], [0, 1, 0], [0, -1, 0], [0, 0, 1], [0, 0, -1]])
_EMPTY = _immutable(np.empty((0, 3), dtype=np.int64))

# These are explicit alternative research configurations, not silently changed
# versions of the older .8-mm-tip / 1.2-mm-shaft coarse simulation instruments.
NATIVE_GENERIC_TOOLS = (
    ToolGeometry("native-fine-aspiration", 1.25, 0.45, 120, 35, 2.0),
    ToolGeometry("native-wide-aspiration", 2.25, 1.1, 120, 35, 3.0),
)


def _frozen(value: Any, dtype: Any = None) -> np.ndarray:
    array = np.ascontiguousarray(value, dtype=dtype)
    return _immutable(array)


def _hash_array(array: np.ndarray) -> str:
    return sha256(str(array.shape).encode() + str(array.dtype).encode() + array.tobytes()).hexdigest()


def _binary_mask(value: Any, name: str) -> np.ndarray:
    array = np.asarray(value)
    if array.dtype.kind not in "biuf" or not np.isfinite(array).all() or np.any((array != 0) & (array != 1)):
        raise ValueError(f"{name} must contain finite binary values")
    return _frozen(array, bool)


def _unique(arrays: Iterable[np.ndarray]) -> np.ndarray:
    available = [array for array in arrays if len(array)]
    return _frozen(np.unique(np.concatenate(available), axis=0), np.int64) if available else _EMPTY


@dataclass(frozen=True)
class NativeResectionConfig:
    tissue_mask: np.ndarray
    target_labels: np.ndarray
    affine: np.ndarray
    access: AccessWindow
    tools: tuple[ToolGeometry, ...]
    source_hash: str
    tissue_support_provenance: str
    case_id: str = "native"
    hard_exclusion: np.ndarray | None = None
    max_tip_step_mm: float = 0.25
    max_microsteps: int = 4096
    _fingerprint: str = field(init=False, repr=False)

    def __post_init__(self) -> None:
        tissue = _binary_mask(self.tissue_mask, "tissue_mask")
        labels = np.asarray(self.target_labels)
        if (labels.dtype.kind not in "biuf" or not np.isfinite(labels).all()
                or np.any(labels < 0) or np.any(labels > np.iinfo(np.int16).max)
                or np.any(labels != np.floor(labels))):
            raise ValueError("Native target labels must be finite nonnegative int16-representable integers")
        labels = _frozen(labels, np.int16)
        affine = _frozen(self.affine, float)
        if tissue.ndim != 3 or min(tissue.shape) < 1 or labels.shape != tissue.shape:
            raise ValueError("Native tissue and labels must be matching nonempty 3-D grids")
        if np.any(labels < 0) or np.any((labels > 0) & ~tissue):
            raise ValueError("Native target labels must be nonnegative and inside tissue")
        hard = _binary_mask(np.zeros(tissue.shape, bool) if self.hard_exclusion is None else self.hard_exclusion, "hard_exclusion")
        if hard.shape != tissue.shape:
            raise ValueError("Native hard exclusions must match the source grid")
        # Reuse geometry's physical-frame validation without installing tissue as
        # a hard exclusion: the distal active region is allowed declared contact.
        scene = GeometryScene(np.zeros((1, 1, 1), bool), affine)
        if scene._orthogonal_spacing is None:
            raise ValueError("Native cell removal currently requires an orthogonal affine, including oblique rotations")
        if (not isinstance(self.source_hash, str) or not self.source_hash
                or not isinstance(self.tissue_support_provenance, str) or not self.tissue_support_provenance):
            raise ValueError("Native source hash and tissue support provenance are mandatory")
        if not isinstance(self.case_id, str) or not self.case_id:
            raise ValueError("case_id must be a nonempty string")
        if not self.tools or len({tool.tool_id for tool in self.tools}) != len(self.tools):
            raise ValueError("Native tool catalog must have unique nonempty configurations")
        spacing = np.linalg.norm(affine[:3, :3], axis=0)
        step = np.asarray(self.max_tip_step_mm)
        if step.shape != () or step.dtype.kind not in "iuf" or not np.isfinite(step) or not 0 < step <= spacing.min() / 2 + 1e-12:
            raise ValueError("Microstep must be positive and no larger than half the smallest native spacing")
        object.__setattr__(self, "max_tip_step_mm", float(step))
        if not isinstance(self.max_microsteps, int) or isinstance(self.max_microsteps, bool) or self.max_microsteps < 1:
            raise ValueError("max_microsteps must be a positive integer")
        for name, value in (("tissue_mask", tissue), ("target_labels", labels), ("affine", affine), ("hard_exclusion", hard)):
            object.__setattr__(self, name, value)
        object.__setattr__(self, "tools", tuple(self.tools))
        record = {
            "version": NATIVE_RESECTION_VERSION, "case_id": self.case_id,
            "source_hash": self.source_hash, "tissue_support_provenance": self.tissue_support_provenance,
            "arrays": {name: _hash_array(getattr(self, name)) for name in ("tissue_mask", "target_labels", "affine", "hard_exclusion")},
            "access": asdict(self.access), "tools": [asdict(tool) for tool in self.tools],
            "max_tip_step_mm": self.max_tip_step_mm, "max_microsteps": self.max_microsteps,
            "partial_cell_policy": "record_contact_keep_occupied_credit_no_removal",
            "temporal_rule": "shaft_sweep_must_clear_prior_cavity_before_endpoint_cut_credit",
        }
        digest = sha256(json.dumps(record, sort_keys=True, default=lambda x: np.asarray(x).tolist()).encode()).hexdigest()
        object.__setattr__(self, "_fingerprint", "sha256:" + digest)

    @property
    def fingerprint(self) -> str:
        return self._fingerprint

    @property
    def decision_model_hash(self) -> str:
        return self._fingerprint

    @property
    def voxel_volume_mm3(self) -> float:
        return abs(float(np.linalg.det(self.affine[:3, :3])))


@dataclass(frozen=True)
class NativeMicrostep:
    tip_start_mm: tuple[float, float, float]
    tip_end_mm: tuple[float, float, float]
    active_stroke_start_mm: tuple[float, float, float]
    active_stroke_end_mm: tuple[float, float, float]
    active_radius_mm: float
    removed_indices_native: np.ndarray
    contact_indices_native: np.ndarray

    def to_dict(self) -> dict[str, Any]:
        return {
            "tip_start_mm": self.tip_start_mm, "tip_end_mm": self.tip_end_mm,
            "active_stroke_start_mm": self.active_stroke_start_mm,
            "active_stroke_end_mm": self.active_stroke_end_mm,
            "active_radius_mm": self.active_radius_mm,
            "removed_indices_native": self.removed_indices_native.tolist(),
            "contact_indices_native": self.contact_indices_native.tolist(),
        }


@dataclass(frozen=True)
class NativeStrokeResult:
    feasible: bool
    reason: str
    tool_id: str
    tip_mm: tuple[float, float, float]
    axis_unit: tuple[float, float, float]
    entry_mm: tuple[float, float, float]
    removed_indices_native: np.ndarray
    contact_indices_native: np.ndarray
    microsteps: tuple[NativeMicrostep, ...]
    source_hash: str
    source_state_hash: str
    decision_model_hash: str
    native_affine: np.ndarray
    source_shape: tuple[int, int, int]
    tissue_support_provenance: str
    voxel_volume_mm3: float
    failure_tip_mm: tuple[float, float, float] | None = None
    geometry_unknowns: tuple[str, ...] = ()
    native_footprint: str = "fully_contained_connected_cells_v1"

    @property
    def removed_volume_mm3(self) -> float:
        return len(self.removed_indices_native) * self.voxel_volume_mm3

    def to_history_record(self) -> dict[str, Any]:
        if not self.feasible:
            raise ValueError("A rejected preview is not an executed removal history")
        return {
            "tool_id": self.tool_id, "tip_mm": self.tip_mm, "axis_unit": self.axis_unit,
            "entry_mm": self.entry_mm,
            "removed_indices_native": self.removed_indices_native.tolist(),
            "contact_indices_native": self.contact_indices_native.tolist(),
            "microsteps": [step.to_dict() for step in self.microsteps],
            "source_hash": self.source_hash, "source_state_hash": self.source_state_hash,
            "decision_model_hash": self.decision_model_hash, "native_affine": self.native_affine.tolist(),
            "source_shape": self.source_shape, "tissue_support_provenance": self.tissue_support_provenance,
            "native_footprint": self.native_footprint, "removed_volume_mm3": self.removed_volume_mm3,
            "geometry_unknowns": self.geometry_unknowns,
            "partial_contact_policy": "exposure_recorded_cells_remain_occupied_no_partial_volume_credit",
            "temporal_rule": "shaft_sweep_must_clear_prior_cavity_before_endpoint_cut_credit",
            "retraction": "reverse_identical_insertion_path_after_removal_no_in_brain_reorientation",
            "clinical_deficit_probability": None,
        }


def contained_capsule_cells(scene: GeometryScene, candidates: np.ndarray,
                            start_mm: np.ndarray, end_mm: np.ndarray, radius_mm: float) -> np.ndarray:
    """Sufficient full-cell containment: all 8 affine corners lie in a capsule.

    Capsules are convex, so containing every cell corner contains the entire
    affine cell. This does not credit boundary slivers or voxel centers alone.
    """
    if not len(candidates):
        return _EMPTY
    offsets = np.array(list(product((-0.5, 0.5), repeat=3))) @ scene.affine[:3, :3].T
    centers = scene.voxel_to_world(candidates)
    corners = centers[:, None, :] + offsets
    distances = point_segment_distances(corners, start_mm, end_mm)
    # A tiny inward tolerance favors retaining a boundary cell over overclaiming
    # its physical removal when floating-point transforms are nearly tangent.
    return _frozen(candidates[np.all(distances <= radius_mm - 1e-10, axis=1)], np.int64)


def _connected_surface_cells(candidates: np.ndarray, connected_free: np.ndarray) -> np.ndarray:
    """Remove only candidate components with a face exposed to existing free space."""
    if not len(candidates):
        return _EMPTY
    pending = {tuple(int(v) for v in index) for index in candidates}
    accepted: set[tuple[int, int, int]] = set()
    shape = np.array(connected_free.shape)
    queue = []
    for cell in pending:
        adjacent = np.array(cell) + _NEIGHBORS
        inside = np.all((adjacent >= 0) & (adjacent < shape), axis=1)
        if not inside.all() or np.any(connected_free[tuple(adjacent[inside].T)]):
            queue.append(cell)
    while queue:
        cell = queue.pop()
        if cell in accepted:
            continue
        accepted.add(cell)
        for adjacent in np.array(cell) + _NEIGHBORS:
            neighbor = tuple(int(v) for v in adjacent)
            if neighbor in pending and neighbor not in accepted:
                queue.append(neighbor)
    return _frozen(sorted(accepted), np.int64).reshape(-1, 3) if accepted else _EMPTY


def _extend_connected_free(new_cells: np.ndarray, remaining: np.ndarray,
                            connected_free: np.ndarray) -> None:
    """Open only actually connected empty cells, including a newly opened pocket."""
    queue = [tuple(int(v) for v in index) for index in new_cells]
    shape = np.array(remaining.shape)
    while queue:
        cell = queue.pop()
        if connected_free[cell]:
            continue
        connected_free[cell] = True
        for neighbor in np.array(cell) + _NEIGHBORS:
            if np.all((neighbor >= 0) & (neighbor < shape)):
                index = tuple(int(v) for v in neighbor)
                if not remaining[index] and not connected_free[index]:
                    queue.append(index)


class NativeResectionEngine:
    """Transactional native-grid geometric rollout, with pure preview and replay.

    Preview computes a complete insertion and reverse-path retraction before any
    state changes. A failed candidate therefore changes neither cavity nor cost.
    Successful previews may be committed once to their exact originating state;
    a serialized or fabricated result cannot bypass the transition checker.
    """

    def __init__(self, config: NativeResectionConfig):
        self.config = config
        self._tools = {tool.tool_id: tool for tool in config.tools}
        self._scene = GeometryScene(config.hard_exclusion, config.affine)
        self._cell_scene = GeometryScene(np.zeros(config.tissue_mask.shape, bool), config.affine)
        # Enclosed anatomical voids are not new access entrances. They join free
        # space only if a verified removal opens a face-connected route to them.
        self._initial_connected_free = _frozen(~binary_fill_holes(config.tissue_mask), bool)
        self.reset()

    @property
    def state_hash(self) -> str:
        return self._state_hash

    @property
    def decision_model_hash(self) -> str:
        return self.config.fingerprint

    def reset(self) -> None:
        self.remaining_mask = self.config.tissue_mask.copy()
        self.removed_mask = np.zeros(self.config.tissue_mask.shape, bool)
        self.contact_mask = np.zeros(self.config.tissue_mask.shape, bool)
        self.connected_free_mask = self._initial_connected_free.copy()
        self.history: list[dict[str, Any]] = []
        self.revision = 0
        self._state_hash = "sha256:" + sha256((self.config.fingerprint + ":initial").encode()).hexdigest()
        self._preview_records: dict[int, NativeStrokeResult] = {}
        self._preview_digests: dict[int, str] = {}

    def clone(self) -> NativeResectionEngine:
        result = copy.copy(self)
        result.remaining_mask = self.remaining_mask.copy()
        result.removed_mask = self.removed_mask.copy()
        result.contact_mask = self.contact_mask.copy()
        result.connected_free_mask = self.connected_free_mask.copy()
        result.history = copy.deepcopy(self.history)
        result._preview_records = self._preview_records.copy()
        result._preview_digests = self._preview_digests.copy()
        return result

    def preview_stroke(self, tool_id: str, tip_mm: Any, *, entry_mm: Any | None = None) -> NativeStrokeResult:
        if tool_id not in self._tools:
            raise ValueError("Unknown native instrument configuration")
        tip = np.asarray(tip_mm, dtype=float)
        if tip.shape != (3,) or not np.isfinite(tip).all():
            raise ValueError("Native target tip must be a finite physical three-vector")
        tool = self._tools[tool_id]
        entry = self.config.access.center_mm if entry_mm is None else np.asarray(entry_mm, dtype=float)
        if entry.shape != (3,) or not np.isfinite(entry).all():
            raise ValueError("Native entry must be a finite physical three-vector")
        plane_distance = float((entry - self.config.access.center_mm) @ self.config.access.normal_inward)
        if abs(plane_distance) > 1e-8:
            raise ValueError("Native entry must lie on the declared access-window plane")
        displacement = tip - entry
        length = float(np.linalg.norm(displacement))
        if length < 1e-9:
            raise ValueError("Native stroke endpoint must lie inward from its access")
        axis = displacement / length
        number = int(np.ceil(length / self.config.max_tip_step_mm))
        records: list[NativeMicrostep] = []
        contacts: list[np.ndarray] = []
        removed: list[np.ndarray] = []
        unknowns: tuple[str, ...] = ()

        def finish(feasible: bool, reason: str, failure_tip: np.ndarray | None = None) -> NativeStrokeResult:
            result = NativeStrokeResult(
                feasible, reason, tool_id, tuple(tip), tuple(axis), tuple(entry),
                _unique(removed) if feasible else _EMPTY, _unique(contacts) if feasible else _EMPTY,
                tuple(records), self.config.source_hash, self.state_hash, self.config.fingerprint,
                self.config.affine, self.config.tissue_mask.shape, self.config.tissue_support_provenance,
                self.config.voxel_volume_mm3, None if failure_tip is None else tuple(failure_tip), unknowns,
            )
            if feasible:
                # Bound retained proposal certificates; committed histories are
                # ordinary JSON and must be independently replayed when loaded.
                if len(self._preview_records) >= 128:
                    oldest = next(iter(self._preview_records))
                    self._preview_records.pop(oldest)
                    self._preview_digests.pop(oldest)
                self._preview_records[id(result)] = result
                self._preview_digests[id(result)] = self._result_digest(result)
            return result

        if number + 1 > self.config.max_microsteps:
            return finish(False, "MICROSTEP_BUDGET")
        geometry = check_motion(tool, ToolPose(entry, axis), ToolPose(tip, axis), self._scene, self.config.access)
        unknowns = geometry.unknowns
        if not geometry.feasible:
            failure = geometry.failures[0]
            return finish(False, "HARD_GEOMETRY:" + failure.reason, np.asarray(failure.position_mm))
        remaining = self.remaining_mask.copy()
        connected_free = self.connected_free_mask.copy()
        previous = np.asarray(entry)
        for step in range(number + 1):
            current = entry + displacement * (step / number)
            active_start = previous - tool.tip_length_mm * axis
            active_end = current
            touched = capsule_voxel_indices(self._cell_scene, active_start, active_end, tool.tip_radius_mm)
            touched = touched[self.config.tissue_mask[tuple(touched.T)]]
            occupied = touched[remaining[tuple(touched.T)]]
            fully_inside = contained_capsule_cells(self._cell_scene, occupied, active_start, active_end, tool.tip_radius_mm)
            # A straight axial shaft's exact continuous sweep is one capsule.
            # Check against the PRIOR cavity: cells first fully cut at the end
            # of this interval cannot clear an earlier shaft collision. This
            # strict ordering prevents a coarse microstep borrowing future cuts.
            shaft_start = previous - tool.working_length_mm * axis
            shaft_end = current - tool.tip_length_mm * axis
            shaft = capsule_voxel_indices(self._cell_scene, shaft_start, shaft_end, tool.shaft_radius_mm)
            blocked = shaft[remaining[tuple(shaft.T)]]
            if len(blocked):
                return finish(False, "SHAFT_BLOCKED_BY_REMAINING_NATIVE_TISSUE", current)
            eligible = _connected_surface_cells(fully_inside, connected_free)
            if len(eligible):
                remaining[tuple(eligible.T)] = False
                _extend_connected_free(eligible, remaining, connected_free)
            contacts.append(_frozen(touched, np.int64))
            removed.append(eligible)
            records.append(NativeMicrostep(tuple(previous), tuple(current), tuple(active_start), tuple(active_end),
                                          tool.tip_radius_mm, eligible, _frozen(touched, np.int64)))
            previous = current
        if not any(len(indices) for indices in removed):
            return finish(False, "NO_NEW_FULLY_CONTAINED_SURFACE_CELLS", tip)
        return finish(True, "NATIVE_CONNECTED_STROKE")

    def commit_preview(self, result: NativeStrokeResult) -> NativeStrokeResult:
        if (not result.feasible or self._preview_records.get(id(result)) is not result
                or result.source_state_hash != self.state_hash
                or result.decision_model_hash != self.config.fingerprint):
            raise ValueError("Preview is rejected, stale, foreign, or not generated by this engine state")
        if self._preview_digests.get(id(result)) != self._result_digest(result):
            raise ValueError("Preview certificate changed after geometry validation")
        indices = result.removed_indices_native
        if (indices.shape != (len(indices), 3) or indices.dtype != np.int64
                or not np.array_equal(indices, _unique(step.removed_indices_native for step in result.microsteps))):
            raise ValueError("Preview removal accounting differs from its unique native microstep cells")
        if not np.all(self.remaining_mask[tuple(indices.T)]):
            raise ValueError("Preview contains already removed tissue")
        self.remaining_mask[tuple(indices.T)] = False
        self.removed_mask[tuple(indices.T)] = True
        _extend_connected_free(indices, self.remaining_mask, self.connected_free_mask)
        self.contact_mask[tuple(result.contact_indices_native.T)] = True
        self.history.append(result.to_history_record())
        self.revision += 1
        # The ancestry includes the full certified history/contact record, not
        # only its cavity: different exposures cannot share a state identity.
        self._state_hash = "sha256:" + sha256((self.state_hash + ":committed:" + self._preview_digests[id(result)]).encode()).hexdigest()
        self._preview_records.clear()
        self._preview_digests.clear()
        return result

    @staticmethod
    def _result_digest(result: NativeStrokeResult) -> str:
        record = result.to_history_record()
        record["array_descriptors"] = [
            (str(array.dtype), array.shape) for array in
            (result.removed_indices_native, result.contact_indices_native, result.native_affine,
             *(step.removed_indices_native for step in result.microsteps),
             *(step.contact_indices_native for step in result.microsteps))]
        return sha256(json.dumps(record, sort_keys=True).encode()).hexdigest()

    def execute_stroke(self, tool_id: str, tip_mm: Any, *, entry_mm: Any | None = None) -> NativeStrokeResult:
        result = self.preview_stroke(tool_id, tip_mm, entry_mm=entry_mm)
        return self.commit_preview(result) if result.feasible else result

    def metrics(self) -> dict[str, Any]:
        voxel = self.config.voxel_volume_mm3
        target = self.config.target_labels > 0
        removed_target = self.removed_mask & target
        return {
            "native_resection_version": NATIVE_RESECTION_VERSION,
            "decision_model_hash": self.decision_model_hash, "source_hash": self.config.source_hash,
            "native_grid_shape": self.config.tissue_mask.shape,
            "native_voxel_volume_mm3": voxel,
            "simulated_removed_target_mm3": float(np.count_nonzero(removed_target) * voxel),
            "simulated_removed_normal_mm3": float(np.count_nonzero(self.removed_mask & ~target) * voxel),
            "modeled_residual_target_mm3": float(np.count_nonzero(target & ~self.removed_mask) * voxel),
            "contacted_tissue_upper_bound_mm3": float(np.count_nonzero(self.contact_mask) * voxel),
            "contacted_but_not_removed_upper_bound_mm3": float(np.count_nonzero(self.contact_mask & ~self.removed_mask) * voxel),
            "removed_by_label_mm3": {int(label): float(np.count_nonzero(self.removed_mask & (self.config.target_labels == label)) * voxel)
                                     for label in np.unique(self.config.target_labels) if label > 0},
            "tissue_support_provenance": self.config.tissue_support_provenance,
            "removal_definition": "fully_contained_connected_native_cells; boundary partial volumes uncredited",
            "completed_strokes": self.revision, "clinical_deficit_probability": None,
        }


def native_config_from_case(case: Any, *, access: AccessWindow,
                            tools: tuple[ToolGeometry, ...] = NATIVE_GENERIC_TOOLS,
                            max_tip_step_mm: float = 0.25,
                            hard_exclusion: np.ndarray | None = None) -> NativeResectionConfig:
    """Keep the actual source grid and explicit skull-stripped support provenance."""
    if not case.compartments:
        raise ValueError("Native case requires explicit source target compartments")
    labels = np.zeros(case.mri.shape, np.int16)
    for label, (name, mask) in enumerate(sorted(case.compartments.items()), start=1):
        if np.any(labels[np.asarray(mask, bool)]):
            raise ValueError("Native target compartments must be nonoverlapping")
        labels[np.asarray(mask, bool)] = label
    if case.brain_mask is None:
        collection = case.metadata.get("source_collection", {})
        if case.metadata.get("skull_stripped") is not True and collection.get("name") != "UCSF-PDGM":
            raise ValueError("Reviewed brain support is required for a case without declared skull stripping")
        tissue = binary_fill_holes(np.asarray(case.mri) != 0)
        provenance = "hole_filled_nonzero_native_MRI_support; unreviewed skull_strip_assumption; source targets retained"
    else:
        tissue = np.asarray(case.brain_mask, bool)
        provenance = "source_case_brain_mask; source targets retained"
    tissue = tissue | (labels > 0)
    return NativeResectionConfig(tissue, labels, case.affine, access, tools, case.semantic_hash,
                                 provenance, case_id=case.case_id, hard_exclusion=hard_exclusion,
                                 max_tip_step_mm=max_tip_step_mm)
