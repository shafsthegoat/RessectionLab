"""Learning/search adapter for the native contained-cell cutting engine.

This backend is deliberately distinct from the coarse contact-cell pilot. It
counts only full source cells certified inside the active cutting brush as
removed. Partial boundary contacts remain separate, with explicit surrogate
costs. Tool dimensions, observation features, and engine hashes identify the
change of model; old coarse results cannot be relabeled as native results.
"""
from __future__ import annotations

import copy
from collections.abc import Mapping
from dataclasses import fields, is_dataclass
import hashlib
import json
import time
from typing import Any, Callable, Iterable

import numpy as np
from scipy.ndimage import binary_fill_holes

from .geometry import AccessWindow, ToolGeometry
from .simulation import (ACTION_FEATURE_NAMES, InvalidActionError, MacroAction,
                         RewardSpec, SearchResult, SequentialSimulator,
                         SimulationConfig, SimulationObservation, StepResult, _readonly)
from .worlds import WorldGeneratorConfig

NATIVE_ADAPTER_VERSION = "native-contained-brush-policy-v2"
NATIVE_ACTION_FEATURE_NAMES = ACTION_FEATURE_NAMES + (
    "partial_normal_contact_volume", "partial_motor_contact_surrogate", "partial_language_contact_surrogate",
)


def _immutable_model_descriptor(value: Any) -> Any:
    """Verify frozen bytes once by identity and preserve every array interpretation.

    All configuration arrays own immutable bytes snapshots. Their contents cannot
    change, so repeatedly hashing hundreds of MB adds no protection. Identity,
    pointer, shape, dtype, strides, and immutable root buffer detect replacement
    or reinterpretation; scalar/mapping/dataclass fields are compared directly.
    Mutable or unsupported buffers fail closed instead of using this fast path.
    """
    if isinstance(value, np.ndarray):
        root = value
        while isinstance(root, np.ndarray) and root.base is not None:
            root = root.base
        if value.flags.writeable or not isinstance(root, bytes):
            raise RuntimeError("Native model arrays must retain immutable bytes backing")
        return ("array", id(value), id(root), value.shape, value.dtype.str, value.strides,
                value.__array_interface__["data"][0], value.nbytes)
    if is_dataclass(value):
        return (type(value).__qualname__, tuple((field.name, _immutable_model_descriptor(getattr(value, field.name))) for field in fields(value)))
    if isinstance(value, Mapping):
        return (type(value).__qualname__, tuple((repr(key), _immutable_model_descriptor(item)) for key, item in sorted(value.items(), key=lambda pair: repr(pair[0]))))
    if isinstance(value, (tuple, list)):
        return (type(value).__qualname__, tuple(_immutable_model_descriptor(item) for item in value))
    if isinstance(value, np.generic):
        return (value.dtype.str, value.item())
    if value is None or isinstance(value, (str, int, float, bool)):
        return (type(value).__qualname__, value)
    raise TypeError(f"Unsupported mutable model value: {type(value).__qualname__}")


class NativeSequentialSimulator(SequentialSimulator):
    """Episode API shared by masked learning and native beam/greedy search."""
    def __init__(self, native_config: Any, candidate_tips_mm: Iterable[Iterable[float]], *,
                 nominal_motor: np.ndarray | None = None, nominal_language: np.ndarray | None = None,
                 candidate_entries_mm: Iterable[Iterable[float]] | None = None,
                 reward: RewardSpec = RewardSpec(), world_generator: WorldGeneratorConfig | None = None,
                 max_steps: int = 8, max_actions: int = 8,
                 partial_contact_weight: float = .05,
                 compartment_names: dict[int, str] | None = None,
                 cancelled: Callable[[], bool] | None = None):
        from .native_resection import NativeResectionEngine
        self._cancelled = cancelled
        self._check_cancelled()
        if not np.isfinite(partial_contact_weight) or partial_contact_weight < 0:
            raise ValueError("Partial-contact surrogate weight must be finite and nonnegative")
        tips = np.asarray(tuple(tuple(v) for v in candidate_tips_mm), dtype=float)
        if tips.ndim != 2 or tips.shape[1:] != (3,) or not len(tips) or not np.isfinite(tips).all():
            raise ValueError("At least one finite native physical target point is required")
        entries = np.repeat(np.asarray(native_config.access.center_mm)[None, :], len(tips), axis=0) if candidate_entries_mm is None else np.asarray(tuple(tuple(v) for v in candidate_entries_mm), dtype=float)
        if entries.shape != tips.shape or not np.isfinite(entries).all():
            raise ValueError("Native entry proposals must align with the physical target points")
        self.candidate_entries_mm = _readonly(entries)
        self.native_config = native_config
        self.engine = NativeResectionEngine(native_config)
        self.candidate_tips_mm = _readonly(tips)
        self.partial_contact_weight = float(partial_contact_weight)
        self._adapter_constants_hash = self._adapter_signature(tips, self.partial_contact_weight, entries)
        self._native_config_hash = native_config.fingerprint
        self._previews: dict[str, Any] = {}
        self._proposal_failures: list[dict[str, Any]] = []
        config = SimulationConfig(
            native_config.tissue_mask, native_config.target_labels, native_config.affine,
            native_config.access, native_config.tools, nominal_motor=nominal_motor,
            nominal_language=nominal_language, hard_exclusion=native_config.hard_exclusion,
            reward=reward, world_generator=world_generator, max_steps=max_steps,
            max_actions=max_actions, case_id=native_config.case_id,
            source_hash=native_config.source_hash,
            evidence_available=(nominal_motor is not None, nominal_language is not None),
            compartment_names=compartment_names or {1: "radiological_target"},
            derivation={"track": "annotation_assisted_native_contained_cell_simulation",
                        "tissue_support_provenance": native_config.tissue_support_provenance,
                        "removal_primitive": "full_affine_cells_contained_by_continuous_active_brush",
                        "partial_contact_policy": "retained_tissue_exposure_not_removed",
                        "partial_contact_weight": self.partial_contact_weight,
                        "candidate_sampling": "fixed_physical_tip_and_entry_points_shared_by_all_methods"})
        self._native_decision_hash = "sha256:" + hashlib.sha256(json.dumps({
            "adapter": NATIVE_ADAPTER_VERSION, "native_config": self._native_config_hash,
            "simulation_config": config.decision_model_hash, "tips_mm": tips.tolist(), "entries_mm": entries.tolist(),
            "partial_contact_weight": self.partial_contact_weight,
        }, sort_keys=True).encode()).hexdigest()
        super().__init__(config)
        self._frozen_model_descriptor = _immutable_model_descriptor((self.config, self.native_config))

    def _check_cancelled(self) -> None:
        if self._cancelled is not None and self._cancelled():
            raise InterruptedError("Native planning cancelled")

    @staticmethod
    def _adapter_signature(tips: np.ndarray, weight: float, entries: np.ndarray) -> str:
        array = np.asarray(tips)
        entry = np.asarray(entries)
        metadata = repr((array.shape, array.dtype.str, weight, entry.shape, entry.dtype.str)).encode()
        return hashlib.sha256(metadata + array.tobytes() + entry.tobytes()).hexdigest()

    @property
    def decision_model_hash(self) -> str:
        return self._native_decision_hash

    def assert_model_frozen(self) -> None:
        if hasattr(self, "_frozen_model_descriptor"):
            if _immutable_model_descriptor((self.config, self.native_config)) != self._frozen_model_descriptor:
                raise RuntimeError("Decision model changed during native optimization")
        else:
            super().assert_model_frozen()
        if self.native_config.fingerprint != self._native_config_hash:
            raise RuntimeError("Native cutting model changed during optimization")
        current = self._adapter_signature(self.candidate_tips_mm, self.partial_contact_weight, self.candidate_entries_mm)
        if current != self._adapter_constants_hash:
            raise RuntimeError("Native proposal or partial-contact objective changed during optimization")

    def _bind_engine_masks(self) -> None:
        self.remaining_mask = self.engine.remaining_mask
        self.removed_mask = self.engine.removed_mask
        self.exposed_mask = self.engine.contact_mask

    def reset(self, seed: int = 0) -> SimulationObservation:
        self._check_cancelled()
        self.engine.reset()
        self._previews = {}
        self._proposal_failures = []
        self._partial_contact_mask = np.zeros(self.config.tissue_mask.shape, bool)
        # Parent reset owns fixed-world sampling and the nominal actor contract.
        super().reset(seed)
        self._bind_engine_masks()
        self._hidden_world_hash = "sha256:" + hashlib.sha256((self._hidden_world_hash + self.decision_model_hash).encode()).hexdigest()
        return self.observation()

    def clone(self) -> NativeSequentialSimulator:
        result = super().clone()
        result.engine = self.engine.clone()
        result._bind_engine_masks()
        result._partial_contact_mask = self._partial_contact_mask.copy()
        result._previews = self._previews.copy()
        result._proposal_failures = copy.deepcopy(self._proposal_failures)
        return result

    def proposed_actions(self) -> tuple[MacroAction, ...]:
        if self._proposals is not None:
            return self._proposals
        empty = _readonly(np.empty((0, 3)), int)
        actions = [MacroAction("STOP", None, None, None, None, empty, empty)]
        self._previews = {}
        self._proposal_failures = []
        if not self.terminated:
            inverse = np.linalg.inv(self.config.affine)
            for point_index, tip in enumerate(self.candidate_tips_mm):
                target_index = tuple(np.rint(inverse[:3, :3] @ tip + inverse[:3, 3]).astype(int))
                for tool in self.config.tools:
                    self._check_cancelled()
                    entry = self.candidate_entries_mm[point_index]
                    preview = self.engine.preview_stroke(tool.tool_id, tip, entry_mm=entry)
                    if not preview.feasible or not len(preview.removed_indices_native):
                        self._proposal_failures.append({"tool_id": tool.tool_id, "tip_mm": tip.tolist(),
                                                        "entry_mm": entry.tolist(), "reason": preview.reason})
                        continue
                    action_id = f"NATIVE:{tool.tool_id}:{point_index}"
                    action = MacroAction(action_id, tool.tool_id, target_index, tuple(tip),
                                         tuple(preview.axis_unit), _readonly(preview.removed_indices_native, int),
                                         _readonly(preview.contact_indices_native, int),
                                         float(np.linalg.norm(tip - entry)))
                    actions.append(action)
                    self._previews[action_id] = preview
                    if len(actions) >= self.config.max_actions:
                        break
                if len(actions) >= self.config.max_actions:
                    break
        self._proposals = tuple(actions)
        return self._proposals

    def _new_partial_indices(self, action: MacroAction) -> np.ndarray:
        contact = action.swept_indices
        if not len(contact):
            return np.empty((0, 3), int)
        eligible = self.remaining_mask[tuple(contact.T)] & ~self._partial_contact_mask[tuple(contact.T)]
        removed = {tuple(v) for v in action.removal_indices}
        return np.asarray([v for v in contact[eligible] if tuple(v) not in removed], dtype=int).reshape(-1, 3)

    def observation(self) -> SimulationObservation:
        basic = super().observation()
        extra = np.zeros((len(basic.action_ids), 3), dtype=np.float32)
        for row, action in enumerate(self.proposed_actions()[1:], 1):
            indices = self._new_partial_indices(action)
            if not len(indices):
                continue
            coords = tuple(indices.T)
            extra[row] = (self._normal_fraction[coords].sum() * self.voxel_volume_mm3,
                          self.config.nominal_motor[coords].sum() * self.voxel_volume_mm3,
                          self.config.nominal_language[coords].sum() * self.voxel_volume_mm3)
        return SimulationObservation(basic.action_ids, _readonly(np.concatenate((basic.action_features, extra), axis=1)),
                                     basic.action_mask, basic.state_features)

    def _action_value(self, action: MacroAction, *, hidden: bool) -> tuple[float, dict[str, float], np.ndarray]:
        if action.action_id == "STOP":
            return 0.0, {}, np.empty((0, 3), int)
        coords = tuple(action.removal_indices.T)
        partial = self._new_partial_indices(action)
        contact_coords = tuple(partial.T)
        motor = self._hidden_motor if hidden else self.config.nominal_motor
        language = self._hidden_language if hidden else self.config.nominal_language
        volume = self.voxel_volume_mm3
        costs = {"target_removed_mm3": float(self._target_fraction[coords].sum() * volume),
                 "normal_removed_mm3": float(self._normal_fraction[coords].sum() * volume),
                 "motor_surrogate_delta": float(motor[coords].sum() * volume),
                 "language_surrogate_delta": float(language[coords].sum() * volume),
                 "partial_normal_contact_mm3": float(self._normal_fraction[contact_coords].sum() * volume),
                 "partial_motor_contact_surrogate": float(motor[contact_coords].sum() * volume),
                 "partial_language_contact_surrogate": float(language[contact_coords].sum() * volume)}
        w = self.config.reward
        change = self._current_tool is not None and self._current_tool != action.tool_id
        value = (w.target_per_mm3 * costs["target_removed_mm3"] - w.normal_per_mm3 * costs["normal_removed_mm3"]
                 - w.motor_per_mm3 * costs["motor_surrogate_delta"] - w.language_per_mm3 * costs["language_surrogate_delta"]
                 - self.partial_contact_weight * (w.normal_per_mm3 * costs["partial_normal_contact_mm3"]
                    + w.motor_per_mm3 * costs["partial_motor_contact_surrogate"]
                    + w.language_per_mm3 * costs["partial_language_contact_surrogate"])
                 - w.action_cost - w.motion_per_mm * 2 * action.insertion_distance_mm - w.tool_change_cost * change)
        return float(value), costs, partial

    def nominal_action_value(self, action: MacroAction) -> float:
        return self._action_value(action, hidden=False)[0]

    def step(self, action: str | int) -> StepResult:
        self.assert_model_frozen()
        options = self.proposed_actions()
        if isinstance(action, (int, np.integer)):
            if action < 0 or action >= len(options):
                raise InvalidActionError("Native action index outside current proposal set")
            selected = options[int(action)]
        else:
            selected = next((value for value in options if value.action_id == action), None)
            if selected is None:
                raise InvalidActionError("Unknown or stale native action")
        if selected.action_id == "STOP":
            return super().step("STOP")
        if self.terminated:
            raise InvalidActionError("Native episode terminated")
        value, costs, partial = self._action_value(selected, hidden=True)
        preview = self._previews[selected.action_id]
        self.engine.commit_preview(preview)
        self._bind_engine_masks()
        if len(partial):
            self._partial_contact_mask[tuple(partial.T)] = True
        self._current_tool = selected.tool_id
        self._actions.append(selected.action_id)
        self.total_reward += value
        if len(self._actions) >= self.config.max_steps:
            self.terminated = True
            self.termination_reason = "step_budget"
        info = {**preview.to_history_record(), **costs, "action_id": selected.action_id,
                "termination_reason": self.termination_reason,
                "clinical_deficit_probability": None}
        self._history.append(info)
        self._proposals = None
        self._previews = {}
        return StepResult(self.observation(), value, self.terminated, copy.deepcopy(info))

    def metrics(self) -> dict[str, Any]:
        result = super().metrics()
        result.update({"simulation_version": NATIVE_ADAPTER_VERSION,
                       "native_engine_config_hash": self._native_config_hash,
                       "native_engine_metrics": self.engine.metrics(),
                       "partial_contact_weight": self.partial_contact_weight,
                       "cumulative_partial_normal_contact_mm3": float(self._normal_fraction[self._partial_contact_mask].sum() * self.voxel_volume_mm3),
                       "proposal_failures": copy.deepcopy(self._proposal_failures),
                       "assumptions": ["continuous_active_brush_contact", "contained_native_cells_only_removed",
                                       "partial_boundary_contact_retains_tissue", "full_retraction_before_reorientation",
                                       "rigid_static_geometry", "hypothetical_intracranial_access"]})
        return result


def native_beam_search(simulator: NativeSequentialSimulator, *, beam_width: int = 8,
                       max_expansions: int = 128, max_wall_seconds: float = 60.) -> SearchResult:
    """Nominal-evidence search including the exact same partial-contact costs."""
    if beam_width < 1 or max_expansions < 1 or not np.isfinite(max_wall_seconds) or max_wall_seconds <= 0:
        raise ValueError("Native search budgets must be finite and positive")
    started = time.perf_counter()
    beam = [(0., (), simulator.clone())]
    best_score, best_sequence = 0., ()
    expansions = 0
    reason = "search_exhausted"
    while beam:
        children = []
        for score, sequence, state in beam:
            if state.terminated:
                continue
            for action in state.proposed_actions()[1:]:
                if expansions >= max_expansions or time.perf_counter() - started >= max_wall_seconds:
                    reason = "expansion_budget" if expansions >= max_expansions else "wall_budget"
                    return SearchResult(best_sequence + ("STOP",), best_score, expansions, time.perf_counter() - started, reason)
                value = score + state.nominal_action_value(action)
                child = state.clone()
                child.step(action.action_id)
                expansions += 1
                candidate = sequence + (action.action_id,)
                if value > best_score + 1e-10:
                    best_score, best_sequence = value, candidate
                children.append((value, candidate, child))
        children.sort(key=lambda item: (-item[0], item[1]))
        beam = children[:beam_width]
    return SearchResult(best_sequence + ("STOP",), best_score, expansions, time.perf_counter() - started, reason)


def native_greedy_search(simulator: NativeSequentialSimulator, *, max_wall_seconds: float = 60.) -> SearchResult:
    started = time.perf_counter()
    state = simulator.clone()
    score, sequence = 0., []
    while not state.terminated and time.perf_counter() - started < max_wall_seconds:
        options = state.proposed_actions()
        index = int(np.argmax([state.nominal_action_value(action) for action in options]))
        if index == 0:
            break
        value = state.nominal_action_value(options[index])
        state.step(options[index].action_id)
        sequence.append(options[index].action_id)
        score += value
    return SearchResult(tuple(sequence) + ("STOP",), score, len(sequence), time.perf_counter() - started, "greedy_stop")


def make_native_patient_simulator(case: Any, *, access: AccessWindow | None = None,
                                  nominal_motor: np.ndarray | None = None,
                                  nominal_language: np.ndarray | None = None,
                                  world_generator: WorldGeneratorConfig | None = None,
                                  max_steps: int = 6, max_actions: int = 8,
                                  candidate_count: int = 6,
                                  tools: tuple[ToolGeometry, ...] | None = None,
                                  access_frame: str = "RAS+",
                                  entry_mode: str = "parallel",
                                  cancelled: Callable[[], bool] | None = None) -> NativeSequentialSimulator:
    """Preserve the exact source grid for a restricted annotation-assisted run.

    Geometry and returned points use RAS+ millimeters. LPS+ source affines are
    converted without resampling the image; an explicit access uses access_frame.
    MRI support may be used as an estimated envelope only for a declared
    skull-stripped source. Full-head BTC or an unknown upload needs a reviewed
    brain mask. Functional evidence remains unavailable unless explicitly given.
    """
    from .native_resection import NATIVE_GENERIC_TOOLS, NativeResectionConfig
    if cancelled is not None and cancelled():
        raise InterruptedError("Native planning cancelled")
    if access_frame not in {"RAS+", "LPS+"}:
        raise ValueError("Native access frame must be RAS+ or LPS+")
    if entry_mode not in {"parallel", "central"}:
        raise ValueError("Entry mode must be parallel or central")
    if candidate_count < 1:
        raise ValueError("At least one target candidate is required")
    names = tuple(sorted(case.compartments))
    if not names:
        raise ValueError("Native planning requires source target compartments")
    labels = np.zeros(case.mri.shape, dtype=np.int16)
    for index, name in enumerate(names, 1):
        mask = np.asarray(case.compartments[name], bool)
        if np.any(mask & (labels > 0)):
            raise ValueError("Native planning requires nonoverlapping source target compartments")
        labels[mask] = index
    union = labels > 0
    if not union.any():
        raise ValueError("Native planning requires nonempty source target compartments")
    if case.brain_mask is None:
        collection = case.metadata.get("source_collection", {})
        declared = case.metadata.get("skull_stripped")
        if declared is False or (declared is not True and collection.get("name") != "UCSF-PDGM"):
            raise ValueError("A reviewed brain mask is required for full-head or unknown source imaging")
        tissue = binary_fill_holes(np.asarray(case.mri) != 0) | union
        provenance = "estimated_hole_filled_nonzero_UCSF_or_declared_skull_stripped_MRI_support; brain_surface_unreviewed"
    else:
        tissue = np.asarray(case.brain_mask, bool)
        if np.any(union & ~tissue):
            raise ValueError("Target annotation lies outside the supplied brain mask; review the conflicting anatomy before planning")
        provenance = "supplied_case_brain_mask"
    if cancelled is not None and cancelled():
        raise InterruptedError("Native planning cancelled")
    frame_conversion = np.diag([-1., -1., 1., 1.])
    affine = np.asarray(case.affine) if case.frame == "RAS+" else frame_conversion @ np.asarray(case.affine)
    provenance += f"; source_frame={case.frame}; simulation_frame=RAS+"
    if access is not None and access_frame == "LPS+":
        access = AccessWindow(frame_conversion[:3, :3] @ access.center_mm,
                              frame_conversion[:3, :3] @ access.normal_inward, access.radius_mm, access.window_id)
    spacing = np.linalg.norm(affine[:3, :3], axis=0)
    indices = np.argwhere(union)
    centroid = np.rint(indices.mean(axis=0)).astype(int)
    if access is None:
        options = []
        for axis in range(3):
            selector = list(centroid)
            selector[axis] = slice(None)
            occupied = np.flatnonzero(tissue[tuple(selector)])
            for side in (-1, 1):
                coordinate = centroid.astype(float)
                coordinate[axis] = occupied[0] - .5 if side == -1 else occupied[-1] + .5
                depth = abs(coordinate[axis] - centroid[axis]) * spacing[axis]
                center = affine[:3, :3] @ coordinate + affine[:3, 3]
                direction = affine[:3, axis] / spacing[axis] * (-side)
                options.append((depth, axis, side, center, direction))
        _, _, _, center, direction = min(options, key=lambda item: item[:3])
        access = AccessWindow(center, direction, 6., "native_hypothetical_nearest_envelope_axis")
    entry = np.asarray(access.center_mm)
    center_point = affine[:3, :3] @ centroid + affine[:3, 3]
    points: list[tuple[float, float, float]] = [tuple(float(v) for v in center_point)]
    # Include an axis-aligned paid opening before oblique alternatives. The
    # entrance and all tool dimensions remain in the versioned native model.
    # A few fixed physical target samples define the common action proposal set.
    # Prioritize every radiological compartment before adding depth alternatives.
    alternatives = []
    for label in range(1, len(names) + 1):
        cells = np.argwhere(labels == label)
        if not len(cells):
            continue
        center = cells.mean(axis=0)
        representative = cells[np.argmin(np.sum(((cells - center) * spacing) ** 2, axis=1))]
        physical = cells @ affine[:3, :3].T + affine[:3, 3]
        representative_mm = affine[:3, :3] @ representative + affine[:3, 3]
        points.append(tuple(float(v) for v in representative_mm))
        distance = np.linalg.norm(physical - entry, axis=1)
        alternatives.extend(tuple(float(v) for v in physical[i]) for i in (np.argmin(distance), np.argmax(distance)))
    for point in alternatives:
        if point not in points:
            points.append(point)
    points = points[:candidate_count]
    normal = np.asarray(access.normal_inward)
    entries = [np.asarray(point) - normal * np.dot(np.asarray(point) - entry, normal) for point in points] if entry_mode == "parallel" else [entry for _ in points]
    native = NativeResectionConfig(tissue, labels, affine, access, tools or NATIVE_GENERIC_TOOLS,
                                   source_hash=case.semantic_hash, tissue_support_provenance=provenance,
                                   case_id=case.case_id, max_tip_step_mm=min(.25, float(spacing.min()) / 4))
    return NativeSequentialSimulator(native, points, candidate_entries_mm=entries, nominal_motor=nominal_motor,
                                     nominal_language=nominal_language, world_generator=world_generator,
                                     max_steps=max_steps, max_actions=max_actions,
                                     compartment_names={i + 1: name for i, name in enumerate(names)},
                                     cancelled=cancelled)
