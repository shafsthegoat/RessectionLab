"""Bounded beam search over the same observed, certified actions as the actor.

No patient loading, reference-label access, optimization, or geometry shortcuts
live here. The task owns its planning clone, frozen objective and native action
certificates. A caller-supplied policy changes expansion order only; its value
head is never used to score a route. Shared actions do not imply identical
representations: a planner may read a full permitted nominal field while the
actor sees a crop. Callers must report that limitation in comparisons.
"""
from __future__ import annotations

from bisect import insort
import math
import time

import numpy as np


class ObservedSearchLimit(RuntimeError):
    """Time exhaustion is a failure with partial accounting, not a STOP result."""

    def __init__(self, message, *, accounting, best_sequence):
        super().__init__(message)
        self.accounting = accounting
        self.best_sequence = best_sequence


class ObservedPolicyChanged(RuntimeError):
    """A guidance forward mutated parameters or registered model buffers."""

    def __init__(self, accounting):
        super().__init__("Guided search changed the frozen policy state_dict")
        self.accounting = accounting


def _legal_actions(observation):
    ids = tuple(observation.action_ids)
    mask = np.asarray(observation.action_mask)
    if (not ids or ids[0] != "STOP" or len(set(ids)) != len(ids)
            or mask.shape != (len(ids),) or mask.dtype != np.bool_ or not mask[0]):
        raise ValueError("Observed search requires unique action IDs and a legal leading STOP")
    return [(index, action) for index, action in enumerate(ids) if index and mask[index]]


def _prefix_progress(parent, path, total, outcome):
    """Copy only public scalar progress from the transition already performed.

    No task, array, observation, history or geometry callback is consulted.
    Missing or differently scoped records remain unknown, never zero progress.
    """
    row = {"actions": list(path), "estimated_incremental_return": total,
           "target_removed_mm3": None, "outside_supplied_target_removed_mm3": None,
           "max_insertion_distance_mm": None, "progress_status": "nominal_record_unavailable",
           "legal_nonstop_actions": None, "legal_target_family_actions": None,
           "successor_inventory_status": "not_observed_by_search"}
    record = getattr(outcome, "info", None)
    if (not isinstance(record, dict) or record.get("outcome_scope") != "permitted_nominal_model"
            or record.get("action_id") != path[-1]
            or parent["progress_status"] != "available"):
        return row
    values = [record.get(key) for key in
              ("target_removed_mm3", "normal_removed_mm3", "insertion_distance_mm")]
    if not all(type(value) in (int, float) and math.isfinite(value) and value >= 0
               for value in values):
        return row
    target = parent["target_removed_mm3"] + values[0]
    outside = parent["outside_supplied_target_removed_mm3"] + values[1]
    distance = max(parent["max_insertion_distance_mm"], values[2])
    if not all(math.isfinite(value) for value in (target, outside, distance)):
        return row
    row.update(target_removed_mm3=target, outside_supplied_target_removed_mm3=outside,
               max_insertion_distance_mm=distance, progress_status="available")
    return row


def _public_removed_opening_depth(observation, outcome, action, *, with_volume=False):
    """Depth of newly removed, actor-covered cell centres; never tip travel.

    Reads only the existing public observation and exact nominal transition
    record. No task/geometry callback, reference label, or successor observation
    is consulted. None means no eligible new forward opening, not measured zero.
    Optional volume counts only these distinct, newly committed native cells
    at positive inward depth, using the native affine's physical cell volume.
    It never credits accessible tissue, predicted cuts, tip travel or old cavity.
    This is a retention proxy, not target reach, information gain or a bound.
    """
    record = getattr(outcome, "info", None)
    if (not isinstance(record, dict) or record.get("outcome_scope") != "permitted_nominal_model"
            or record.get("action_id") != action
            or record.get("source_hash") != getattr(observation, "source_id", None)):
        raise ValueError("Opening retention requires an exact public nominal transition record")
    observation.assert_intact()
    affine = np.asarray(observation.affine_ras_mm)
    coverage = np.asarray(observation.coverage)
    images = np.asarray(observation.image_channels)
    state = np.asarray(observation.state_features)
    native = np.asarray(record.get("native_affine"))
    shape = np.asarray(record.get("source_shape"))
    cells = np.asarray(record.get("removed_indices_native"))
    if (affine.shape != (4, 4) or native.shape != (4, 4)
            or affine.dtype.kind not in "fiu" or native.dtype.kind not in "fiu"
            or not np.isfinite(affine).all() or not np.isfinite(native).all()
            or not np.array_equal(affine[3], [0., 0., 0., 1.])
            or not np.array_equal(native[3], [0., 0., 0., 1.])
            or images.ndim != 4 or images.shape[0] != 6 or coverage.shape != images.shape
            or coverage.dtype != np.bool_ or state.shape != (10,) or not np.isfinite(state).all()
            or shape.shape != (3,) or shape.dtype.kind not in "iu" or np.any(shape <= 0)
            or not observation.channel_available[1] or not observation.channel_available[3]):
        raise ValueError("Opening retention requires public frame, access and cavity coverage")
    normal = state[6:9].astype(float)
    if with_volume:
        voxel_volume = abs(float(np.linalg.det(native[:3, :3])))
        if not math.isfinite(voxel_volume) or voxel_volume <= 0:
            raise ValueError("Opening volume requires a nondegenerate native physical frame")
    norm = float(np.linalg.norm(normal))
    if not math.isclose(norm, 1., abs_tol=1e-5, rel_tol=0.):
        raise ValueError("Opening retention requires a public unit inward normal")
    normal /= norm
    # A valid empty removal is no progress. Missing fields are never empty.
    if cells.shape == (0,):
        return (None, 0.) if with_volume else None
    if (cells.ndim != 2 or cells.shape[1] != 3 or cells.dtype.kind not in "iu"
            or np.any(cells < 0) or np.any(cells >= shape)
            or len(np.unique(cells, axis=0)) != len(cells)):
        raise ValueError("Opening retention requires distinct native removed-cell indices")
    if not len(cells):
        return (None, 0.) if with_volume else None
    world = cells @ native[:3, :3].T + native[:3, 3]
    crop = np.linalg.solve(affine[:3, :3], (world-affine[:3, 3]).T).T
    inside = np.all((crop >= -.5) & (crop < np.asarray(images.shape[1:])-.5), axis=1)
    if not inside.any():
        return (None, 0.) if with_volume else None
    world, crop = world[inside], np.floor(crop[inside]+.5).astype(np.int64)
    index = tuple(crop.T)
    covered = coverage[1][index] & coverage[3][index]
    if np.any(covered & (images[3][index] != 0)):
        raise ValueError("Opening retention cannot credit previously observed cavity cells")
    eligible = covered & (images[1][index] > 0)
    if not eligible.any():
        return (None, 0.) if with_volume else None
    depths = (world[eligible]-state[3:6]) @ normal
    depth = float(np.max(depths))
    if not math.isfinite(depth):
        raise ValueError("Opening retention produced nonfinite public depth")
    if with_volume:
        volume = int(np.count_nonzero(depths > 0)) * voxel_volume
        if not math.isfinite(volume):
            raise ValueError("Opening volume produced nonfinite physical removal")
        return (depth if depth > 0 else None), volume
    return depth if depth > 0 else None


def _opening_frontier(children, opening, width):
    """Best return first, distinct opening second, remaining slots by return."""
    if not children:
        return []
    selected = [children[0]]
    if opening is not None and opening[1] != selected[0][1]:
        selected.append(opening)
    selected.extend(row for row in children if row[1] not in {item[1] for item in selected})
    return selected[:width]


def observed_beam_search(task, *, max_calls: int, beam_width: int, seconds: float = 2.,
                         policy=None, objective_source: str = "observed_scan_estimator_only",
                         transition_mode: str = "eager", retained_prefix_diagnostics: bool = False,
                         retention_mode: str = "return_only"):
    """Run observed beam search with optional, state-checked actor ordering.

The before/after state_dict hashes cover parameters and registered buffers,
including exceptional exits. Hashing and checks count toward the wall budget.
Python-side counters are not model state, and are outside this check.
Optional diagnostics retain public nominal transition scalars for the beam and
best terminal prefixes. They do not request extra observations or previews;
their bookkeeping time remains inside the same wall budget.
Opt-in return_plus_opening_depth_v1 reserves one lane for actual public opening
depth; terminal return/STOP selection is unchanged. It requires beam width >= 2
and exact nominal removal records. Opt-in return_plus_opening_depth_volume_v1
breaks equal-depth ties by cumulative actual actor-covered removed volume,
then return/path. This deliberately favors wider removal only for retention;
final objective/STOP are unchanged. Old modes add no new output fields.
"""
    started = time.perf_counter()
    integrity = {"policy_state_checks": 0, "policy_state_check_seconds": 0.,
                 "initial_policy_state_hash": None, "final_policy_state_hash": None}
    arguments = dict(max_calls=max_calls, beam_width=beam_width, seconds=seconds, policy=policy,
                     objective_source=objective_source, transition_mode=transition_mode,
                     retained_prefix_diagnostics=retained_prefix_diagnostics,
                     retention_mode=retention_mode,
                     started=started, integrity=integrity)
    if policy is None:
        return _observed_beam_search(task, **arguments)

    def snapshot():
        check_started = time.perf_counter()
        from .spatial_policy import parameter_hash
        result = parameter_hash(policy)
        integrity["policy_state_checks"] += 1
        integrity["policy_state_check_seconds"] += time.perf_counter() - check_started
        return result

    integrity["initial_policy_state_hash"] = snapshot()

    def verify(accounting):
        integrity["final_policy_state_hash"] = snapshot()
        accounting.update(integrity)
        accounting["planning_seconds"] = time.perf_counter() - started
        if integrity["initial_policy_state_hash"] != integrity["final_policy_state_hash"]:
            raise ObservedPolicyChanged(accounting)

    try:
        result = _observed_beam_search(task, **arguments)
    except BaseException as exc:
        verify(getattr(exc, "accounting", {}))
        raise
    verify(result[1])
    if result[1]["planning_seconds"] > seconds:
        result[1]["time_cap_reached"] = True
        raise ObservedSearchLimit("Observed search exceeded its declared episode time",
                                  accounting=result[1], best_sequence=result[0])
    return result


def _observed_beam_search(task, *, max_calls, beam_width, seconds, policy,
                          objective_source, transition_mode, retained_prefix_diagnostics, retention_mode, started, integrity):
    """Return the best evaluated prefix and accounting within declared bounds.

Every prefix has a known zero-increment STOP option. Negative opening prefixes
remain eligible for the next beam layer. A call cap can interrupt a layer;
time exhaustion raises with the partial trace. Neither condition proves STOP
optimal. Optional actor guidance orders *all* legal non-STOP actions, retaining
the same physical reward, beam ranking, candidate inventory and call cap.

The policy must be a frozen spatial policy returning (logits, value). One
no-grad forward is counted per guided expanded node; value is ignored. Task
cloning, inventory preparation and policy forwards are inside the wall budget.
The explicit lazy_planning mode requires advance_planning on the nominal-only
planning clone. It defers successor observations, not current-action geometry.
"""
    if (type(max_calls) is not int or max_calls < 0 or type(beam_width) is not int
            or beam_width < 1 or isinstance(seconds, bool) or not math.isfinite(seconds)
            or seconds < 0 or not isinstance(objective_source, str) or not objective_source):
        raise ValueError("Invalid observed-search budget or objective description")
    if transition_mode not in {"eager", "lazy_planning"}:
        raise ValueError("Choose explicit eager or lazy_planning transition mode")
    if type(retained_prefix_diagnostics) is not bool:
        raise ValueError("retained_prefix_diagnostics must be an explicit bool")
    if retention_mode not in {"return_only", "return_plus_opening_depth_v1", "return_plus_opening_depth_volume_v1"}:
        raise ValueError("Unknown observed-search retention mode")
    volume_lane = retention_mode == "return_plus_opening_depth_volume_v1"
    opening_lane = retention_mode == "return_plus_opening_depth_v1" or volume_lane
    if opening_lane and beam_width < 2:
        raise ValueError("Opening retention requires beam_width >= 2")
    observed = task.planning_clone()
    metrics = observed.metrics()
    if not metrics["planning_estimator_only"]:
        raise ValueError("Search requires the observed-only planning model")
    if transition_mode == "lazy_planning" and not callable(getattr(observed, "advance_planning", None)):
        raise ValueError("Requested lazy_planning requires advance_planning on the planning clone")
    initial_steps = metrics["steps"]
    incumbent, sequence = 0., ()
    incumbent_terminated = observed.terminated
    frontier = [(0., (), observed)]
    del observed  # Do not keep the initial full native state alive after its layer.
    calls = evaluated_prefixes = completed_layers = actor_calls = expanded_nodes = 0
    eager_calls = lazy_calls = observation_requests = 0
    negative_evaluated = negative_retained = negative_pruned = beam_pruned = 0
    peak_next_layer_states = 0
    truncated, root_inventory = False, 0
    layers = []
    if opening_lane:
        frontier_opening = {(): None}
    if volume_lane:
        frontier_volume = {(): 0.}
    if retained_prefix_diagnostics:
        frontier_progress = {(): {"progress_status": "available", "target_removed_mm3": 0.,
            "outside_supplied_target_removed_mm3": 0., "max_insertion_distance_mm": 0.}}

    def executable():
        remaining = task.max_steps - initial_steps
        return sequence if incumbent_terminated or len(sequence) == remaining else (*sequence, "STOP")

    def accounting(*, time_exhausted=False):
        return {"model_transition_calls": calls, "planning_seconds": time.perf_counter() - started,
            "completed_layers": completed_layers, "call_cap_reached": truncated,
            "time_cap_reached": time_exhausted, "beam_width": beam_width, "max_calls": max_calls,
            "estimated_incremental_return": incumbent, "stop_baseline_return": 0.,
            "implicit_stop_evaluations": "all evaluated prefixes; known zero incremental reward",
            "implicit_stop_prefix_count": evaluated_prefixes + 1,
            "evaluated_transition_prefixes": evaluated_prefixes, "objective_source": objective_source,
            "root_legal_nonstop_actions": root_inventory, "expanded_nodes": expanded_nodes,
            "transition_mode": transition_mode, "eager_transition_calls": eager_calls,
            "lazy_planning_transition_calls": lazy_calls,
            "observation_requests": observation_requests,
            "actor_forward_calls": actor_calls,
            "guidance": "actor_expansion_order_only" if policy is not None else "none",
            "negative_prefixes_evaluated": negative_evaluated,
            "negative_prefixes_retained": negative_retained,
            "negative_prefixes_pruned_by_beam": negative_pruned,
            "beam_pruned_prefixes": beam_pruned,
            "peak_next_layer_states": peak_next_layer_states,
            "layers": [dict(row) for row in layers], **integrity,
            **({"opening_depth_retention": {
                "version": retention_mode, "terminal_objective_changed": False,
                "scope": "newly_removed_actor_covered_cell_centres_from_public_aperture_plane",
                "missing_progress": "null; malformed_or_non_nominal_record_refused",
                "ranking": ("best_return_then_distinct_max_depth_tied_by_actual_removed_volume_then_return_and_path"
                            if volume_lane else "best_return_then_distinct_max_depth_tied_by_return_and_path"),
                **({"volume_scope": "cumulative_unique_newly_committed_positive_depth_actor_support_and_cavity_covered_native_cells_mm3",
                    "depth_comparison": "exact_float_order_no_tolerance; volume_only_breaks_equal_depth",
                    "volume_warning": "deliberately_aggressive_retention_proxy_not_clearance_safety_or_target_utility"}
                   if volume_lane else {}),
                "extra_observations_or_previews": 0,
                "memory": "at_most_beam_width_plus_one_retained_child_states_plus_current_child",
                "costs": "progress_arithmetic_and_retention_included_in_wall_and_parent_RSS"}}
               if opening_lane else {}),
            **({"retained_prefix_diagnostics": {
                "schema": "observed-search-retained-prefixes-v1",
                "scope": "public_nominal_incremental_progress_from_this_search_root",
                "outside_removal_semantics": "outside_supplied_target; not verified normal tissue",
                "distance_semantics": "maximum recorded entry-to-tip insertion distance; not cumulative path length",
                "family_count_status": "unavailable_in_shared_observation; null_is_not_zero",
                "terminal_selection": "best at most beam_width by the existing return/path ordering",
                "extra_observations_or_previews": 0,
                "wall_accounting": "diagnostic bookkeeping included in existing wall budget"}}
               if retained_prefix_diagnostics else {})}

    def check_time():
        if time.perf_counter() - started > seconds:
            raise ObservedSearchLimit("Observed search exceeded its declared episode time",
                accounting=accounting(time_exhausted=True), best_sequence=executable())

    try:
        while frontier:
            children = []
            if opening_lane:
                opening_child, child_opening = None, {}
            if volume_lane:
                child_volume = {}
            if retained_prefix_diagnostics:
                child_progress, terminal_progress = {}, []
            child_count = negative_children = 0
            layer = {"depth": len(layers) + 1, "completed": False, "expanded_nodes": 0,
                     "model_transition_calls": 0, "negative_prefixes_evaluated": 0,
                     "negative_prefixes_retained": 0, "negative_prefixes_pruned_by_beam": 0,
                     "negative_prefixes_pending_at_cap": 0}
            layers.append(layer)
            if opening_lane:
                layer["opening_depth_retention"] = []
            if retained_prefix_diagnostics:
                layer["retained_prefix_diagnostics"] = {
                    "selection_status": "partial_layer_not_advanced",
                    "retained_nonterminal_prefixes": [], "best_terminal_prefixes": []}
            for value, prefix, node in frontier:
                if node.terminated:
                    continue
                observation_requests += 1
                observation = node.observation()
                actions = _legal_actions(observation)
                if retained_prefix_diagnostics and prefix:
                    frontier_progress[prefix].update(legal_nonstop_actions=len(actions),
                        successor_inventory_status="observed_during_normal_expansion")
                if not prefix:
                    root_inventory = len(actions)
                expanded_nodes += 1
                layer["expanded_nodes"] += 1
                if policy is not None and actions and calls < max_calls:
                    check_time()
                    import torch
                    actor_calls += 1
                    with torch.no_grad():
                        logits, _ = policy(observation)
                    scores = logits.detach().cpu().numpy()
                    if scores.shape != (len(observation.action_ids),) or not np.isfinite(
                            scores[[index for index, _ in actions]]).all():
                        raise ValueError("Actor ordering requires finite scores for every legal action")
                    # Stable ties retain the task's original inventory order.
                    actions.sort(key=lambda row: -float(scores[row[0]]))
                for _, action in actions:
                    if calls >= max_calls:
                        truncated = True
                        break
                    check_time()
                    child = node.clone()
                    extra = int(opening_lane and opening_child is not None
                                and all(row[1] != opening_child[1] for row in children))
                    peak_next_layer_states = max(peak_next_layer_states, len(children) + extra + 1)
                    calls += 1
                    layer["model_transition_calls"] += 1
                    if transition_mode == "lazy_planning":
                        lazy_calls += 1
                        outcome = child.advance_planning(action)
                    else:
                        eager_calls += 1
                        outcome = child.step(action)
                    if not math.isfinite(outcome.reward):
                        raise FloatingPointError("Nonfinite observed-model transition reward")
                    total, path = value + outcome.reward, (*prefix, action)
                    if not math.isfinite(total):
                        raise FloatingPointError("Nonfinite observed-model prefix return")
                    evaluated_prefixes += 1
                    if total < 0:
                        negative_evaluated += 1
                        layer["negative_prefixes_evaluated"] += 1
                    if total > incumbent:
                        incumbent, sequence = total, path
                        incumbent_terminated = child.terminated
                    if opening_lane:
                        if volume_lane:
                            new_depth, new_volume = _public_removed_opening_depth(
                                observation, outcome, action, with_volume=True)
                            opening_volume = frontier_volume[prefix] + new_volume
                            if not math.isfinite(opening_volume):
                                raise FloatingPointError("Nonfinite accumulated opening removal volume")
                        else:
                            new_depth = _public_removed_opening_depth(observation, outcome, action)
                        depths = [value for value in (frontier_opening[prefix], new_depth) if value is not None]
                        opening_depth = max(depths) if depths else None
                    if retained_prefix_diagnostics:
                        progress = _prefix_progress(frontier_progress[prefix], path, total, outcome)
                        if child.terminated:
                            progress["successor_inventory_status"] = "terminal_not_observed"
                            insort(terminal_progress, (total, path, progress), key=lambda row: (-row[0], row[1]))
                            if len(terminal_progress) > beam_width:
                                terminal_progress.pop()
                            layer["retained_prefix_diagnostics"]["best_terminal_prefixes"] = [
                                row[2] for row in terminal_progress]
                    if not child.terminated:
                        child_count += 1
                        negative_children += int(total < 0)
                        if opening_lane:
                            child_opening[path] = opening_depth
                            if volume_lane:
                                child_volume[path] = opening_volume
                            if opening_depth is not None and (opening_child is None or
                                    (-opening_depth, *((-opening_volume,) if volume_lane else ()), -total, path) <
                                    (-child_opening[opening_child[1]], *((-child_volume[opening_child[1]],)
                                      if volume_lane else ()), -opening_child[0], opening_child[1])):
                                opening_child = (total, path, child)
                        # Retain the exact best beam incrementally. All actions
                        # still execute, but full native child states do not
                        # accumulate until the end of a potentially wide layer.
                        insort(children, (total, path, child), key=lambda row: (-row[0], row[1]))
                        if retained_prefix_diagnostics:
                            child_progress[path] = progress
                        if len(children) > beam_width:
                            discarded = children.pop()
                            if retained_prefix_diagnostics and not opening_lane:
                                child_progress.pop(discarded[1])
                            del discarded
                        if opening_lane:
                            live_paths = {row[1] for row in children}
                            if opening_child is not None:
                                live_paths.add(opening_child[1])
                            child_opening = {key: value for key, value in child_opening.items() if key in live_paths}
                            if volume_lane:
                                child_volume = {key: value for key, value in child_volume.items() if key in live_paths}
                            if retained_prefix_diagnostics:
                                child_progress = {key: value for key, value in child_progress.items() if key in live_paths}
                            selected = _opening_frontier(children, opening_child, beam_width)
                            layer["opening_depth_retention"] = [
                                {"actions": list(row[1]), "estimated_incremental_return": row[0],
                                 "opening_depth_mm": child_opening[row[1]],
                                 **({"opening_removed_volume_mm3": child_volume[row[1]]} if volume_lane else {}),
                                 "reason": "best_return_and_opening" if row is children[0] and opening_child is not None and row[1] == opening_child[1]
                                     else "best_return" if row is children[0]
                                     else "opening" if opening_child is not None and row[1] == opening_child[1]
                                     else "return_fill"}
                                for row in selected]
                        if retained_prefix_diagnostics:
                            layer["retained_prefix_diagnostics"]["retained_nonterminal_prefixes"] = [
                                child_progress[row[1]] for row in (selected if opening_lane else children)]
                        if opening_lane:
                            del selected  # Do not retain obsolete child states through the next action.
                    del child
                if truncated:
                    break
            if truncated:
                # This logical count includes evaluated candidates no longer
                # retained in memory; the incomplete layer was never advanced.
                layer["negative_prefixes_pending_at_cap"] = negative_children
                break
            completed_layers += 1
            layer["completed"] = True
            if retained_prefix_diagnostics:
                layer["retained_prefix_diagnostics"]["selection_status"] = "completed_layer"
                frontier_progress = child_progress
            frontier = _opening_frontier(children, opening_child, beam_width) if opening_lane else children
            if opening_lane:
                frontier_opening = {row[1]: child_opening[row[1]] for row in frontier}
            if volume_lane:
                frontier_volume = {row[1]: child_volume[row[1]] for row in frontier}
            kept = sum(value < 0 for value, _, _ in frontier)
            pruned = negative_children - kept
            negative_retained += kept
            negative_pruned += pruned
            beam_pruned += child_count - len(children)
            layer["negative_prefixes_retained"] = kept
            layer["negative_prefixes_pruned_by_beam"] = pruned
        check_time()
        return executable(), accounting()
    except BaseException as exc:
        if not isinstance(exc, ObservedSearchLimit):
            exc.accounting = accounting()
        raise
