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

from .core import array_digest, freeze_json, semantic_digest

from .geometry import (
    AccessWindow, GeometryScene, ToolGeometry, ToolPose, capsule_voxel_indices,
    check_motion, point_segment_distances, _immutable, _segment_cell_distances,
)

NATIVE_RESECTION_VERSION = "contained-native-cell-connected-suction-v2"
MAX_OBSTRUCTION_DIAGNOSTIC_CELLS = 4096
_NEIGHBORS = np.array([[1, 0, 0], [-1, 0, 0], [0, 1, 0], [0, -1, 0], [0, 0, 1], [0, 0, -1]])
_EMPTY = _immutable(np.empty((0, 3), dtype=np.int64))
_STATE_MASKS = ("remaining_mask", "removed_mask", "contact_mask", "probe_contact_mask", "connected_free_mask")


def _snapshot_identity(array):
    root = array
    while isinstance(root, np.ndarray) and root.base is not None:
        root = root.base
    if not isinstance(array, np.ndarray) or array.flags.writeable or not isinstance(root, bytes):
        raise RuntimeError("Committed native masks require immutable owned bytes")
    return (id(array), id(root), array.shape, array.strides, array.dtype.str)


@dataclass(frozen=True)
class _CommittedMaskSnapshot:
    array: np.ndarray = field(repr=False)
    identity: tuple
    digest: str

    @classmethod
    def capture(cls, value):
        array = _frozen(value, bool)
        return cls(array, _snapshot_identity(array), array_digest(array))

    def verify(self, actual):
        if actual is not self.array or _snapshot_identity(actual) != self.identity:
            raise RuntimeError("Committed native mask content or interpretation was replaced")

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
    # Optional public simulation domain. Its complement is unavailable, not air.
    interaction_domain: np.ndarray | None = None
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
        if self.interaction_domain is not None:
            domain = _binary_mask(self.interaction_domain, "interaction_domain")
            if domain.shape != tissue.shape or np.any(tissue & ~domain):
                raise ValueError("Native interaction domain must contain all modeled tissue on the source grid")
            object.__setattr__(self, "interaction_domain", domain)
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
            **({"interaction_domain": _hash_array(self.interaction_domain),
                "unknown_domain_rule": "within_grid_no_tool_encounter_or_free_space; exterior_extent_unassessed"}
               if self.interaction_domain is not None else {}),
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
    interaction_mode: str = "aspirate"

    @property
    def obstruction_diagnostic(self):
        """Optional first-shaft-failure evidence; never executed removal credit.

        A sidecar rather than a dataclass field preserves every existing result
        serialization and certificate. Its immutable JSON fingerprint binds the
        exact requested ray/state and bounded cells; it is not a full-corridor
        clearance proof or an authorization to execute a rejected stroke.
        """
        return getattr(self, "_obstruction_diagnostic", None)

    @property
    def removed_volume_mm3(self) -> float:
        return len(self.removed_indices_native) * self.voxel_volume_mm3

    def to_history_record(self) -> dict[str, Any]:
        if not self.feasible:
            raise ValueError("A rejected preview is not an executed removal history")
        return {
            **({"interaction_mode": self.interaction_mode} if self.interaction_mode != "aspirate" else {}),
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
                            connected_free: np.ndarray,
                            interaction_domain: np.ndarray | None = None) -> None:
    """Open only actually connected empty cells, including a newly opened pocket."""
    queue = [tuple(int(v) for v in index) for index in new_cells]
    shape = np.array(remaining.shape)
    while queue:
        cell = queue.pop()
        if interaction_domain is not None and not interaction_domain[cell]:
            raise ValueError("Unknown domain cannot become connected free space")
        if connected_free[cell]:
            continue
        connected_free[cell] = True
        for neighbor in np.array(cell) + _NEIGHBORS:
            if np.all((neighbor >= 0) & (neighbor < shape)):
                index = tuple(int(v) for v in neighbor)
                if (not remaining[index] and not connected_free[index]
                        and (interaction_domain is None or interaction_domain[index])):
                    queue.append(index)


class NativeResectionEngine:
    """Transactional native-grid geometric rollout, with pure preview and replay.

    Preview computes a complete insertion and reverse-path retraction before any
    state changes. A failed candidate therefore changes neither cavity nor cost.
    Successful previews may be committed once to their exact originating state;
    a serialized or fabricated result cannot bypass the transition checker.
    Instances belong to one worker; concurrent search branches use ``clone``.
    """

    def __setattr__(self, name, value):
        if name in {"_immutable_state", "_committed_snapshots"} and hasattr(self, name):
            raise AttributeError("Native state storage mode and snapshot binding are internally owned")
        object.__setattr__(self, name, value)

    def __init__(self, config: NativeResectionConfig, *, immutable_state: bool = False):
        if type(immutable_state) is not bool:
            raise TypeError("immutable_state must be an explicit bool")
        self._immutable_state = immutable_state
        self.config = config
        self._tools = {tool.tool_id: tool for tool in config.tools}
        self._scene = GeometryScene(config.hard_exclusion, config.affine)
        self._domain_scene = (None if config.interaction_domain is None else
            GeometryScene(~config.interaction_domain, config.affine))
        self._cell_scene = GeometryScene(np.zeros(config.tissue_mask.shape, bool), config.affine)
        # Enclosed anatomical voids are not new access entrances. They join free
        # space only if a verified removal opens a face-connected route to them.
        blocked = (config.tissue_mask if config.interaction_domain is None else
                   config.tissue_mask | ~config.interaction_domain)
        self._initial_connected_free = _frozen(~binary_fill_holes(blocked), bool)
        self.reset()

    @property
    def state_hash(self) -> str:
        return self._state_hash

    @property
    def decision_model_hash(self) -> str:
        return self.config.fingerprint

    def reset(self) -> None:
        if self._immutable_state:
            # Allocate/capture privately. A failed reset must not discard a
            # live committed state or leave writable arrays with stale seals.
            snapshots = tuple(_CommittedMaskSnapshot.capture(
                self.config.tissue_mask if name == "remaining_mask" else
                self._initial_connected_free if name == "connected_free_mask" else
                np.zeros(self.config.tissue_mask.shape, bool)) for name in _STATE_MASKS)
            history, records, digests = [], {}, {}
            state_hash = "sha256:" + sha256((self.config.fingerprint + ":initial").encode()).hexdigest()
            self._install_snapshots(snapshots)
            self.history, self.revision, self._state_hash = history, 0, state_hash
            self._preview_records, self._preview_digests = records, digests
            return
        self.remaining_mask = self.config.tissue_mask.copy()
        self.removed_mask = np.zeros(self.config.tissue_mask.shape, bool)
        self.contact_mask = np.zeros(self.config.tissue_mask.shape, bool)
        self.probe_contact_mask = np.zeros(self.config.tissue_mask.shape, bool)
        self.connected_free_mask = self._initial_connected_free.copy()
        self.history: list[dict[str, Any]] = []
        self.revision = 0
        self._state_hash = "sha256:" + sha256((self.config.fingerprint + ":initial").encode()).hexdigest()
        self._preview_records: dict[int, NativeStrokeResult] = {}
        self._preview_digests: dict[int, str] = {}

    def _install_snapshots(self, snapshots):
        for name, snapshot in zip(_STATE_MASKS, snapshots):
            setattr(self, name, snapshot.array)
        object.__setattr__(self, "_committed_snapshots", snapshots)

    def committed_snapshot_identity(self):
        """Bind cache interpretation as well as mask storage at a task seal."""
        self.committed_mask_digests()
        return (id(self._committed_snapshots), tuple((id(snapshot), snapshot.identity, snapshot.digest)
                for snapshot in self._committed_snapshots))

    def committed_mask_digests(self):
        """Exact old digest bytes, reusable only for verified immutable masks."""
        if self._immutable_state is not True:
            raise RuntimeError("Immutable native state mode changed")
        snapshots = self._committed_snapshots
        if type(snapshots) is not tuple or len(snapshots) != len(_STATE_MASKS):
            raise RuntimeError("Committed native snapshot collection changed")
        for name, snapshot in zip(_STATE_MASKS, snapshots):
            if type(snapshot) is not _CommittedMaskSnapshot:
                raise RuntimeError("Committed native snapshot type changed")
            snapshot.verify(getattr(self, name))
        return {name: snapshot.digest for name, snapshot in zip(_STATE_MASKS, snapshots)}

    def clone(self) -> NativeResectionEngine:
        if self._immutable_state:
            self.committed_mask_digests()
        result = copy.copy(self)
        if not self._immutable_state:
            result.remaining_mask = self.remaining_mask.copy()
            result.removed_mask = self.removed_mask.copy()
            result.contact_mask = self.contact_mask.copy()
            result.probe_contact_mask = self.probe_contact_mask.copy()
            result.connected_free_mask = self.connected_free_mask.copy()
        result.history = copy.deepcopy(self.history)
        result._preview_records = self._preview_records.copy()
        result._preview_digests = self._preview_digests.copy()
        return result

    def preview_stroke(self, tool_id: str, tip_mm: Any, *, entry_mm: Any | None = None,
                       interaction_mode: str = "aspirate", obstruction_diagnostics: bool = False,
                       obstruction_cell_limit: int = MAX_OBSTRUCTION_DIAGNOSTIC_CELLS) -> NativeStrokeResult:
        """Certify aspiration or a non-removing tangential exposed-tip probe.

        Probe contact is geometric occupancy only, with no sensor, force or
        deformation model. A 1e-8 mm numerical tangency tolerance permits no
        material penetration; all remaining contacted cells must be exposed.
        Optional obstruction evidence copies only the already computed first
        blocked set. Truncation is explicit; subsequent blockers are unknown.
        """
        if type(obstruction_diagnostics) is not bool:
            raise TypeError("obstruction_diagnostics must be an explicit bool")
        if (type(obstruction_cell_limit) is not int
                or not 1 <= obstruction_cell_limit <= MAX_OBSTRUCTION_DIAGNOSTIC_CELLS
                or (not obstruction_diagnostics and obstruction_cell_limit != MAX_OBSTRUCTION_DIAGNOSTIC_CELLS)):
            raise ValueError("Explicit obstruction diagnostics require a bounded positive cell limit")
        if self._immutable_state:
            self.committed_mask_digests()
        if interaction_mode not in {"aspirate", "probe"}:
            raise ValueError("Unsupported native instrument interaction")
        if tool_id not in self._tools:
            raise ValueError("Unknown native instrument configuration")
        source_state_hash = self.state_hash
        tip = np.array(tip_mm, dtype=float, copy=True)
        if tip.shape != (3,) or not np.isfinite(tip).all():
            raise ValueError("Native target tip must be a finite physical three-vector")
        tool = self._tools[tool_id]
        entry = self.config.access.center_mm if entry_mm is None else np.array(entry_mm, dtype=float, copy=True)
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
        probe_contacts: list[np.ndarray] = []
        unknowns: tuple[str, ...] = ()

        def finish(feasible: bool, reason: str, failure_tip: np.ndarray | None = None,
                   *, obstruction=None) -> NativeStrokeResult:
            result = NativeStrokeResult(
                feasible, reason, tool_id, tuple(tip), tuple(axis), tuple(entry),
                _unique(removed) if feasible else _EMPTY, _unique(contacts) if feasible else _EMPTY,
                tuple(records), self.config.source_hash, source_state_hash, self.config.fingerprint,
                self.config.affine, self.config.tissue_mask.shape, self.config.tissue_support_provenance,
                self.config.voxel_volume_mm3, None if failure_tip is None else tuple(failure_tip), unknowns,
                interaction_mode=interaction_mode,
            )
            if obstruction is not None:
                object.__setattr__(result, "_obstruction_diagnostic", obstruction)
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
        if self._domain_scene is not None:
            domain_geometry = check_motion(tool, ToolPose(entry, axis), ToolPose(tip, axis),
                                           self._domain_scene, self.config.access)
            unknowns = tuple(sorted(set(unknowns) | set(domain_geometry.unknowns)))
            if not domain_geometry.feasible:
                failure = domain_geometry.failures[0]
                return finish(False, "UNKNOWN_DOMAIN:" + failure.reason, np.asarray(failure.position_mm))
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
                diagnostic = None
                if obstruction_diagnostics:
                    evidence = {
                        "version": "native-first-shaft-obstruction-v1",
                        "reason": "SHAFT_BLOCKED_BY_REMAINING_NATIVE_TISSUE",
                        "source_hash": self.config.source_hash, "source_state_hash": source_state_hash,
                        "decision_model_hash": self.config.fingerprint, "tool": asdict(tool),
                        "interaction_mode": interaction_mode,
                        "requested_tip_mm": tuple(float(v) for v in tip),
                        "entry_mm": tuple(float(v) for v in entry), "axis_unit": tuple(float(v) for v in axis),
                        "failure_tip_mm": tuple(float(v) for v in current),
                        "failure_tip_meaning": "endpoint of first rejected microstep, not exact first-contact time",
                        "previous_tip_mm": tuple(float(v) for v in previous),
                        "shaft_sweep_start_mm": tuple(float(v) for v in shaft_start),
                        "shaft_sweep_end_mm": tuple(float(v) for v in shaft_end),
                        "failure_interval_index": step, "planned_microsteps": number + 1,
                        "native_affine_hash": array_digest(self.config.affine),
                        "source_shape": tuple(int(v) for v in self.config.tissue_mask.shape),
                        "blocked_cell_count": len(blocked), "blocked_indices_hash": array_digest(blocked),
                        "blocked_indices_native": blocked[:obstruction_cell_limit].tolist(),
                        "retained_cell_count": min(len(blocked), obstruction_cell_limit),
                        "cell_limit": obstruction_cell_limit, "complete_first_failure_set": len(blocked) <= obstruction_cell_limit,
                        "truncated": len(blocked) > obstruction_cell_limit,
                        "prior_temporary_removed_count": sum(len(indices) for indices in removed),
                        "prior_temporary_removals_hash": semantic_digest([_hash_array(indices) for indices in removed]),
                        "prior_temporary_removal_hash_scope": "ordered completed microstep native shape/dtype/bytes SHA256 digests; cells are disjoint by native remaining-mask updates",
                        "prior_temporary_removals_committed": False,
                        "temporal_rule": "shaft_sweep_must_clear_prior_cavity_before_endpoint_cut_credit",
                        "scope": "all remaining cells in first rejected shaft sweep, not minimal blockers; subsequent corridor unknown; re-preview after preparation",
                    }
                    diagnostic = freeze_json({**evidence, "fingerprint": semantic_digest(evidence)})
                return finish(False, "SHAFT_BLOCKED_BY_REMAINING_NATIVE_TISSUE", current,
                              obstruction=diagnostic)
            if interaction_mode == "probe":
                # Unlike aspiration the active region may not enter occupied
                # source-cell interiors, even though contact is recorded.
                distances = _segment_cell_distances(self._cell_scene, occupied, active_start, active_end)
                if len(occupied) and np.any(distances < tool.tip_radius_mm - 1e-8):
                    return finish(False, "PROBE_ACTIVE_REGION_PENETRATES_REMAINING_TISSUE", current)
                for cell in occupied:
                    neighbors = cell + _NEIGHBORS
                    inside = np.all((neighbors >= 0) & (neighbors < remaining.shape), axis=1)
                    if inside.all() and not connected_free[tuple(neighbors[inside].T)].any():
                        return finish(False, "PROBE_CONTACT_NOT_EXPOSED", current)
                probe_contacts.append(_frozen(occupied, np.int64))
                eligible = _EMPTY
            else:
                eligible = _connected_surface_cells(fully_inside, connected_free)
            if len(eligible):
                remaining[tuple(eligible.T)] = False
                _extend_connected_free(eligible, remaining, connected_free, self.config.interaction_domain)
            contacts.append(_frozen(touched, np.int64))
            removed.append(eligible)
            records.append(NativeMicrostep(tuple(previous), tuple(current), tuple(active_start), tuple(active_end),
                                          tool.tip_radius_mm, eligible, _frozen(touched, np.int64)))
            previous = current
        if interaction_mode == "probe":
            if not any(len(indices) for indices in probe_contacts):
                return finish(False, "NO_EXPOSED_PROBE_CONTACT", tip)
            return finish(True, "NATIVE_NONREMOVING_TANGENTIAL_PROBE")
        if not any(len(indices) for indices in removed):
            return finish(False, "NO_NEW_FULLY_CONTAINED_SURFACE_CELLS", tip)
        return finish(True, "NATIVE_CONNECTED_STROKE")

    def commit_preview(self, result: NativeStrokeResult) -> NativeStrokeResult:
        if self._immutable_state:
            self.committed_mask_digests()
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
        if result.interaction_mode == "probe" and len(indices):
            raise ValueError("A probe certificate cannot remove tissue")
        if self._immutable_state:
            return self._commit_immutable_preview(result, indices)
        self.remaining_mask[tuple(indices.T)] = False
        self.removed_mask[tuple(indices.T)] = True
        _extend_connected_free(indices, self.remaining_mask, self.connected_free_mask, self.config.interaction_domain)
        self.contact_mask[tuple(result.contact_indices_native.T)] = True
        if result.interaction_mode == "probe":
            contacted = result.contact_indices_native
            occupied = contacted[self.remaining_mask[tuple(contacted.T)]]
            self.probe_contact_mask[tuple(occupied.T)] = True
        self.history.append(result.to_history_record())
        self.revision += 1
        # The ancestry includes the full certified history/contact record, not
        # only its cavity: different exposures cannot share a state identity.
        self._state_hash = "sha256:" + sha256((self.state_hash + ":committed:" + self._preview_digests[id(result)]).encode()).hexdigest()
        self._preview_records.clear()
        self._preview_digests.clear()
        return result

    def _commit_immutable_preview(self, result, indices):
        """Prepare changed storage privately; keep geometry and ancestry exact."""
        changed = set()
        if len(indices):
            changed.update(("remaining_mask", "removed_mask", "connected_free_mask"))
        if len(result.contact_indices_native):
            changed.add("contact_mask")
            if result.interaction_mode == "probe":
                changed.add("probe_contact_mask")
        masks = {name: getattr(self, name).copy() if name in changed else getattr(self, name)
                 for name in _STATE_MASKS}
        if len(indices):
            masks["remaining_mask"][tuple(indices.T)] = False
            masks["removed_mask"][tuple(indices.T)] = True
            _extend_connected_free(indices, masks["remaining_mask"], masks["connected_free_mask"], self.config.interaction_domain)
        if len(result.contact_indices_native):
            masks["contact_mask"][tuple(result.contact_indices_native.T)] = True
            if result.interaction_mode == "probe":
                contacted = result.contact_indices_native
                occupied = contacted[masks["remaining_mask"][tuple(contacted.T)]]
                masks["probe_contact_mask"][tuple(occupied.T)] = True
        snapshots = tuple(_CommittedMaskSnapshot.capture(masks[name]) if name in changed else old
                          for name, old in zip(_STATE_MASKS, self._committed_snapshots))
        history = [*self.history, result.to_history_record()]
        state_hash = "sha256:" + sha256((self.state_hash + ":committed:" + self._preview_digests[id(result)]).encode()).hexdigest()
        self._install_snapshots(snapshots)
        self.history = history
        self.revision += 1
        self._state_hash = state_hash
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

    def execute_stroke(self, tool_id: str, tip_mm: Any, *, entry_mm: Any | None = None,
                       interaction_mode: str = "aspirate") -> NativeStrokeResult:
        result = self.preview_stroke(tool_id, tip_mm, entry_mm=entry_mm, interaction_mode=interaction_mode)
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
    """Keep source cells and explicit support; normalize physical coordinates to RAS+.

    The supplied access is interpreted in ``case.frame``. LPS-to-RAS conversion
    changes only physical coordinate convention, never image samples or cells.
    """
    from .critical_evidence import canonical_hard_exclusion
    hard_exclusion, critical = canonical_hard_exclusion(case, hard_exclusion)
    if not case.compartments:
        raise ValueError("Native case requires explicit source target compartments")
    labels = np.zeros(case.mri.shape, np.int16)
    for label, (name, mask) in enumerate(sorted(case.compartments.items()), start=1):
        if np.any(labels[np.asarray(mask, bool)]):
            raise ValueError("Native target compartments must be nonoverlapping")
        labels[np.asarray(mask, bool)] = label
    if case.brain_mask is None:
        from .structural_evidence import declared_mri_support_allowed
        if not declared_mri_support_allowed(case):
            raise ValueError("Reviewed brain support is required for a case without declared skull stripping")
        tissue = binary_fill_holes(np.asarray(case.mri) != 0)
        provenance = "hole_filled_nonzero_native_MRI_support; unreviewed skull_strip_assumption; source targets retained"
    else:
        from .structural_evidence import planning_brain_support
        tissue, support_record = planning_brain_support(case)
        tissue = np.asarray(tissue, bool)
        if np.any((labels > 0) & ~tissue):
            raise ValueError("Source target lies outside the supplied brain mask; review the conflicting anatomy")
        provenance = support_record["method"] + "; source targets retained; cortical_access_unverified"
    tissue = tissue | (labels > 0)
    affine = np.asarray(case.affine)
    frame = getattr(case, "frame", "RAS+")
    if frame not in {"RAS+", "LPS+"}:
        raise ValueError("Source coordinate frame must be explicit RAS+ or LPS+")
    if frame == "LPS+":
        conversion = np.diag([-1., -1., 1., 1.])
        affine = conversion @ affine
        access = AccessWindow(conversion[:3, :3] @ access.center_mm,
                              conversion[:3, :3] @ access.normal_inward, access.radius_mm, access.window_id)
    provenance += f"; source_frame={frame}; simulation_frame=RAS+"
    if critical.planning_binding is not None:
        provenance += f"; critical_evidence={critical.fingerprint}; annotation_domain_only"
    return NativeResectionConfig(tissue, labels, affine, access, tools, case.semantic_hash,
                                 provenance, case_id=case.case_id, hard_exclusion=hard_exclusion,
                                 max_tip_step_mm=max_tip_step_mm)
