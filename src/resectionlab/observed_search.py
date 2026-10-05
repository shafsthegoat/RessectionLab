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


def observed_beam_search(task, *, max_calls: int, beam_width: int, seconds: float = 2.,
                         policy=None, objective_source: str = "observed_scan_estimator_only",
                         transition_mode: str = "eager"):
    """Run observed beam search with optional, state-checked actor ordering.

The before/after state_dict hashes cover parameters and registered buffers,
including exceptional exits. Hashing and checks count toward the wall budget.
Python-side counters are not model state, and are outside this check.
"""
    started = time.perf_counter()
    integrity = {"policy_state_checks": 0, "policy_state_check_seconds": 0.,
                 "initial_policy_state_hash": None, "final_policy_state_hash": None}
    arguments = dict(max_calls=max_calls, beam_width=beam_width, seconds=seconds, policy=policy,
                     objective_source=objective_source, transition_mode=transition_mode,
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
                          objective_source, transition_mode, started, integrity):
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
            "layers": [dict(row) for row in layers], **integrity}

    def check_time():
        if time.perf_counter() - started > seconds:
            raise ObservedSearchLimit("Observed search exceeded its declared episode time",
                accounting=accounting(time_exhausted=True), best_sequence=executable())

    try:
        while frontier:
            children = []
            child_count = negative_children = 0
            layer = {"depth": len(layers) + 1, "completed": False, "expanded_nodes": 0,
                     "model_transition_calls": 0, "negative_prefixes_evaluated": 0,
                     "negative_prefixes_retained": 0, "negative_prefixes_pruned_by_beam": 0,
                     "negative_prefixes_pending_at_cap": 0}
            layers.append(layer)
            for value, prefix, node in frontier:
                if node.terminated:
                    continue
                observation_requests += 1
                observation = node.observation()
                actions = _legal_actions(observation)
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
                    peak_next_layer_states = max(peak_next_layer_states, len(children) + 1)
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
                    if not child.terminated:
                        child_count += 1
                        negative_children += int(total < 0)
                        # Retain the exact best beam incrementally. All actions
                        # still execute, but full native child states do not
                        # accumulate until the end of a potentially wide layer.
                        insort(children, (total, path, child), key=lambda row: (-row[0], row[1]))
                        if len(children) > beam_width:
                            children.pop()
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
            frontier = children
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
