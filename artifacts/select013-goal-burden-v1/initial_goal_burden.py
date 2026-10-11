"""Initial public suction-goal ordering by estimated nominal preparation cells.

This is a geometry heuristic, not feasibility, tissue risk, or an action-count
lower bound. It keeps the existing base catalog. The full shaft supercover is
reduced by optimistic whole-cell containment in the full active-tip sweep;
exposure, prior-cavity ordering, aperture, hard structures and future removal
sequences still require ordinary native certification. Sheared-grid supercovers
are conservative, and uncertain inward coverage is never ranked as cheap space.
"""
from dataclasses import asdict, dataclass
import math
import time

import numpy as np

from resectionlab.core import freeze_json, semantic_digest
from resectionlab.focused_paged_task import FocusedPagedObstructionTask
from resectionlab.geometry import capsule_voxel_indices
from resectionlab.native_proposals import _residual_aperture_test
from resectionlab.native_resection import contained_capsule_cells, _hash_array
from resectionlab.paged_obstruction import PagedObstructionTask
from resectionlab.post_exposure import preentry_tip

VERSION = "initial_public_suction_nominal_preparation_v1"
TOOL_ID = "generic_suction"


@dataclass(frozen=True)
class Limits:
    max_goals: int = 120
    max_bbox_cells_per_query: int = 262144
    max_total_bbox_cells: int = 4000000
    seconds: float = 30.0

    def __post_init__(self):
        for name, ceiling in (("max_goals", 120), ("max_bbox_cells_per_query", 262144),
                              ("max_total_bbox_cells", 4000000)):
            if type(getattr(self, name)) is not int or not 1 <= getattr(self, name) <= ceiling:
                raise ValueError("Invalid explicit goal-burden bound: " + name)
        if not math.isfinite(self.seconds) or not 0 < self.seconds <= 60:
            raise ValueError("Goal-burden time bound must be in (0,60]")


class BurdenLimit(RuntimeError):
    def __init__(self, reason, accounting):
        self.accounting = freeze_json(accounting)
        super().__init__(reason)


def _key(ray):
    return (ray.tool_id, tuple(float(x) for x in ray.entry_mm), tuple(float(x) for x in ray.tip_mm))


def _capsules(engine, ray):
    """Same represented axis/preentry calculation as preview_stroke, no preview."""
    tool = engine._tools[ray.tool_id]
    entry, tip = np.asarray(ray.entry_mm, float), np.asarray(ray.tip_mm, float)
    length = float(np.linalg.norm(tip-entry))
    if not np.isfinite(length) or length < 1e-9:
        raise ValueError("Native-degenerate goal ray")
    axis = (tip-entry)/length
    if float(axis @ engine.config.access.normal_inward) <= 0:
        raise ValueError("Only inward existing goal rays are supported")
    start = entry
    if engine.config.post_exposure is not None:
        start = preentry_tip(tool, entry, axis, engine.config.access.normal_inward)
    # Union of every fixed-axis shaft interval, including the first physical pose.
    shaft = (start-tool.working_length_mm*axis, tip-tool.tip_length_mm*axis, tool.shaft_radius_mm)
    # Optimistic union of active-tip intervals; never substitute the shaft envelope.
    active_tip = (start-tool.tip_length_mm*axis, tip, tool.tip_radius_mm)
    return tool, axis, start, shaft, active_tip


def _bbox_work(scene, start, end, radius):
    """Exact query work-box formula from capsule_voxel_indices; not collision math."""
    extent = np.linalg.norm(scene._inverse[:3, :3], axis=1)*(radius+scene._cell_radius_mm)
    endpoints = scene.world_to_voxel(np.stack((start, end)))
    lo = np.maximum(0, np.ceil(endpoints.min(axis=0)-extent-1e-9))
    hi = np.minimum(np.asarray(scene.shape)-1, np.floor(endpoints.max(axis=0)+extent+1e-9))
    return int(np.prod(np.maximum(0, hi-lo+1), dtype=np.float64))


def _coverage(scene, entry, axis, capsules):
    """Separate allowed/unassessed proximal escape from unsafe inward clipping.

    This narrow ordering requires an orthogonal native grid and a source-axis
    ray. A rotated/reflected physical frame is fine. Other frames keep scalar
    estimates but are not eligible for a proposed ordering.
    """
    native_axis = scene._inverse[:3, :3] @ axis
    main = int(np.argmax(np.abs(native_axis)))
    rest = np.delete(native_axis, main)
    aligned = bool(np.max(np.abs(rest), initial=0.) <= 1e-10*max(1., abs(native_axis[main])))
    orthogonal = scene._orthogonal_spacing is not None
    entry_native = scene.world_to_voxel(np.asarray(entry))
    entry_inside = bool(np.all(entry_native >= -.5-1e-9)
                        and np.all(entry_native <= np.asarray(scene.shape)-.5+1e-9))
    escaped = []
    for label, (start, end, radius) in zip(("shaft", "active_tip"), capsules):
        endpoints = scene.world_to_voxel(np.stack((start, end)))
        radial_extent = np.linalg.norm(scene._inverse[:3, :3], axis=1)*radius
        lo, hi = endpoints.min(axis=0)-radial_extent, endpoints.max(axis=0)+radial_extent
        for dim in range(3):
            for side, crosses in ((-1, lo[dim] < -.5-1e-9),
                                  (1, hi[dim] > scene.shape[dim]-.5+1e-9)):
                if crosses:
                    proximal = bool(orthogonal and aligned and entry_inside and dim == main
                                    and side == (-1 if native_axis[main] > 0 else 1))
                    escaped.append({"part": label, "native_axis": dim, "side": side,
                                    "proximal_only_unassessed": proximal})
    return {"orthogonal_native_grid": orthogonal, "source_axis_aligned": aligned,
            "entry_in_native_grid": entry_inside, "outside_native_faces": escaped,
            "inward_or_lateral_crop_uncertain": bool(not orthogonal or not aligned or not entry_inside
                or any(not row["proximal_only_unassessed"] for row in escaped)),
            "outside_native_geometry_is_unassessed": bool(escaped)}


def rank_initial_goals(task, *, limits=Limits(), cancelled=None):
    """Return immutable ranked metadata; never activate pages or execute actions.

    Caller must separately charge construction. Base provider enumeration may
    itself do geometry work; its measured wall is reported, while the explicit
    query counts below cover ONLY this helper's cover/containment calls. A caller
    seeking total native geometry cost must instrument provider calls as well.
    """
    if (type(task) not in (PagedObstructionTask, FocusedPagedObstructionTask)
            or task._steps != 0 or task.terminated or task._current_tool is not None
            or task._base_inventory is not None or task._active_page is not None
            or getattr(task, "_goal_focus", None) is not None or type(limits) is not Limits):
        raise ValueError("Unqueried initial public paged task and explicit limits required")
    task._assert_frozen()
    engine, case = task._engine, task.case
    if TOOL_ID not in engine._tools:
        raise ValueError("Existing generic_suction tool required; no replacement device")
    started = time.monotonic()
    state, history = engine.state_hash, tuple(engine.history)
    registry = tuple((key, id(value)) for key, value in engine._preview_records.items())
    report = {"schema": VERSION, "complete": False, "tool_id": TOOL_ID,
        "source_hash": case.source_hash, "task_decision_model_hash": task.decision_model_hash,
        "native_decision_model_hash": engine.config.fingerprint, "source_state_hash": state,
        "limits": asdict(limits), "base_enumeration_calls": 0, "base_enumeration_wall_seconds": 0.,
        "helper_capsule_cover_calls": 0, "helper_containment_calls": 0, "bbox_cells_charged": 0,
        "native_preview_calls": 0, "native_transitions": 0, "rows": [],
        "ranked_physical_goals": [], "private_reference_used": False,
        "score_scope": "known native nominal occupied shaft cells minus optimistic full active-tip containment",
        "not_claimed": ["feasibility", "action_count_lower_bound", "complete_reachability", "clinical_risk"],
        "geometry_count_scope": "helper only; base enumeration wall separate; caller charges construction and provider geometry",
        "actor_crop_used_to_clip_score": False}

    def guard():
        task._check_cancelled()
        if cancelled is not None and cancelled():
            raise InterruptedError("Initial goal-burden cancelled")
        if time.monotonic()-started > limits.seconds:
            raise BurdenLimit("Initial goal-burden time cap", report)

    def cover(start, end, radius):
        guard()
        cells = _bbox_work(engine._cell_scene, start, end, radius)
        if (cells > limits.max_bbox_cells_per_query
                or report["bbox_cells_charged"]+cells > limits.max_total_bbox_cells):
            raise BurdenLimit("Initial goal-burden cell-work cap", report)
        report["bbox_cells_charged"] += cells
        report["helper_capsule_cover_calls"] += 1
        return capsule_voxel_indices(engine._cell_scene, start, end, radius)

    try:
        guard(); before = time.monotonic(); report["base_enumeration_calls"] += 1
        batch = case._nominal_proposer.propose(engine, cancelled=cancelled)
        report["base_enumeration_wall_seconds"] = time.monotonic()-before
        report["base_catalog_digest"] = semantic_digest([asdict(ray) for ray in batch.proposals])
        report["base_emitted_count"] = len(batch.proposals)
        report["base_omitted_pairs"] = sum(row.reason == "CANDIDATE_CAP" for row in batch.ledger)
        unique = {}; all_goal_count = 0
        for order, ray in enumerate(batch.proposals):
            if ray.tool_id != TOOL_ID or float(case.nominal_target[ray.voxel]) <= case.proposal_config.nominal_min_membership:
                continue
            all_goal_count += 1; key = _key(ray)
            if key not in unique: unique[key] = [order, ray, []]
            unique[key][2].append({"action_id": ray.proposal_id, "family": ray.family, "raw_order": order})
        report["same_tool_goal_rows"] = all_goal_count
        report["unique_goal_count"] = len(unique)
        if len(unique) > limits.max_goals:
            raise BurdenLimit("Initial goal-burden goal cap; no partial ordering", report)
        for key, (order, ray, aliases) in unique.items():
            guard(); tool, axis, start, shaft, tip_sweep = _capsules(engine, ray)
            shaft_cells = cover(*shaft); tip_cells = cover(*tip_sweep)
            occupied = shaft_cells[engine.remaining_mask[tuple(shaft_cells.T)]]
            report["helper_containment_calls"] += 1
            optimistic = contained_capsule_cells(engine._cell_scene, occupied, *tip_sweep)
            contained_set = {tuple(int(v) for v in cell) for cell in optimistic}
            preparation = np.asarray([cell for cell in occupied if tuple(cell) not in contained_set],
                                     dtype=np.int64).reshape(-1, 3)
            full_cells = np.unique(np.concatenate((shaft_cells, tip_cells)), axis=0)
            domain = engine.config.interaction_domain
            unknown = np.zeros(len(full_cells), bool) if domain is None else ~domain[tuple(full_cells.T)]
            external = (np.zeros(len(full_cells), bool) if engine.config.post_exposure is None
                        else engine.config.post_exposure.external_workspace[tuple(full_cells.T)])
            unresolved_unknown = unknown & ~external
            hard_count = int(np.count_nonzero(engine.config.hard_exclusion[tuple(full_cells.T)]))
            origin, end = np.asarray(case._crop_origin), np.asarray(case._crop_origin)+case._crop_shape
            outside_actor = np.any((full_cells < origin) | (full_cells >= end), axis=1)
            coverage = _coverage(engine._cell_scene, ray.entry_mm, axis, (shaft, tip_sweep))
            aperture = _residual_aperture_test(engine.config.access, tool, ray.entry_mm, ray.tip_mm)
            flags = []
            if coverage["inward_or_lateral_crop_uncertain"]: flags.append("uncertain_inward_native_coverage")
            if np.any(unresolved_unknown): flags.append("unknown_interaction_domain")
            if hard_count: flags.append("full_tool_supercover_intersects_hard_exclusion")
            if aperture["excluded"]: flags.append("static_aperture_excluded")
            elif aperture["clearance_mm"] is None or aperture["clearance_mm"] <= 1e-6:
                flags.append("static_aperture_boundary_uncertain")
            row = {"raw_order": order, "tool_id": key[0], "entry_mm": list(key[1]), "tip_mm": list(key[2]),
                "endpoint_voxel": list(ray.voxel), "aliases": aliases, "rankable": not flags,
                "unrankable_reasons": flags, "physical_start_mm": start.tolist(),
                "shaft_sweep": {"start_mm": shaft[0].tolist(), "end_mm": shaft[1].tolist(), "radius_mm": shaft[2]},
                "optimistic_active_tip_sweep": {"start_mm": tip_sweep[0].tolist(), "end_mm": tip_sweep[1].tolist(), "radius_mm": tip_sweep[2]},
                "shaft_supercover_cells": len(shaft_cells), "occupied_shaft_cells": len(occupied),
                "optimistically_tip_contained_cells": len(optimistic),
                "estimated_nominal_preparation_cells": len(preparation),
                "estimated_nominal_preparation_volume_mm3": len(preparation)*engine.config.voxel_volume_mm3,
                "preparation_indices_digest": _hash_array(preparation),
                "whole_cell_rule": "all_eight_affine_corners_at_radius_minus_1e-10",
                "prior_cavity_microstep_and_exposure_constraints_ignored_optimistically": True,
                "native_coverage": coverage, "interaction_domain_declared": domain is not None,
                "unknown_interaction_cells": int(np.count_nonzero(unresolved_unknown)),
                "external_workspace_unassessed_cells": int(np.count_nonzero(external)),
                "hard_exclusion_supercover_cells": hard_count,
                "full_tool_cells_outside_actor_crop": int(np.count_nonzero(outside_actor)),
                "static_aperture": aperture}
            report["rows"].append(row)
        ranking = sorted((row for row in report["rows"] if row["rankable"]),
                         key=lambda row: (row["estimated_nominal_preparation_cells"], row["raw_order"]))
        report["ranked_physical_goals"] = [{k: row[k] for k in ("tool_id", "entry_mm", "tip_mm")} for row in ranking]
        report["ranking_raw_orders"] = [row["raw_order"] for row in ranking]
        report["unrankable_goal_count"] = len(unique)-len(ranking)
        report["original_first_same_tool_goal"] = None if not unique else {
            "tool_id": next(iter(unique))[0], "entry_mm": list(next(iter(unique))[1]), "tip_mm": list(next(iter(unique))[2])}
        report["top_changed_from_original_first_same_tool"] = bool(ranking and
            report["ranked_physical_goals"][0] != report["original_first_same_tool_goal"])
        report["minimum_score_tie_count"] = sum(row["estimated_nominal_preparation_cells"] ==
            ranking[0]["estimated_nominal_preparation_cells"] for row in ranking) if ranking else 0
        guard()
        report["complete"] = True
        report["wall_seconds"] = time.monotonic()-started
        return freeze_json({**report, "seal": semantic_digest(report)})
    finally:
        task._assert_frozen()
        if (engine.state_hash != state or tuple(engine.history) != history
                or tuple((key, id(value)) for key, value in engine._preview_records.items()) != registry
                or task._base_inventory is not None or task._active_page is not None):
            raise RuntimeError("Burden ranking changed source state, certificates or query context")
