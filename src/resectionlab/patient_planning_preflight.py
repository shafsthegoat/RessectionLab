"""Owned-worker TRAIN anatomy preflight through existing native learning/replay.

The parent must hold the existing worker lease and hard wall/tree-RSS supervisor
before importing this module. This callable has no patient or reference loader,
no CLI, no checkpoint load and no independent-reference callback. Its factory is
owned by the admitted source/QC loader, never provided by the desktop renderer.
"""
from __future__ import annotations

from contextlib import contextmanager
from dataclasses import asdict
from functools import wraps
import json
from pathlib import Path
import time

import torch

from .core import freeze_json, semantic_digest, thaw_json
from .development_episode import replay_frames
from .native_resection import NativeResectionEngine
from .native_spatial_evaluation import evaluate_native_spatial_episode
from .native_spatial_task import NativeSpatialTask
from .observed_search import observed_beam_search
from .patient_planning_admission import PatientPlanningContext
from .patient_planning_learning import (PREFLIGHT_PROTOCOL, PatientImitationSample,
    PatientTrainSession, admit_collected_trace, common_patient_policies,
    patient_gradient_step, patient_imitation_loss, patient_reinforce_loss)
from .planning_budget import PlanningBudget
from .spatial_policy import SpatialPolicy, SpatialTransition, parameter_hash


VERSION = "patient-native-preflight-sealed-plans-v1"


def _write(path, value):
    with Path(path).open("x") as stream:
        json.dump(thaw_json(freeze_json(value)), stream, indent=2, allow_nan=False)
        stream.write("\n")


def _history_identity(history):
    # Both tasks use the same public target. Only the nominal/execution label differs.
    return semantic_digest([{key: value for key, value in row.items() if key != "outcome_scope"}
                            for row in history])


class _CallCosts:
    """Small instrumentation companion to existing PlanningBudget, no new guard."""
    def __init__(self, budget, forward_cap):
        self.budget = budget; self.forward_cap = forward_cap; self.phase = "setup"
        self.rows = {}; self.originals = []; self.forwards = 0

    def __enter__(self):
        for cls, name, label in ((SpatialPolicy, "forward", "policy_forward"),
                (NativeSpatialTask, "_prepare_inventory", "inventory_request"),
                (NativeSpatialTask, "_transition", "native_transition"),
                (NativeSpatialTask, "clone", "task_clone")):
            original = getattr(cls, name); self.originals.append((cls, name, original))
            def factory(original, label):
                @wraps(original)
                def measured(instance, *args, **kwargs):
                    self.budget.check()
                    if label == "policy_forward":
                        if self.forwards >= self.forward_cap:
                            raise InterruptedError("Declared total policy forward cap")
                        self.forwards += 1
                    row = self.rows.setdefault(self.phase, {})
                    row[label+"_calls"] = row.get(label+"_calls", 0)+1
                    if label == "inventory_request" and instance._inventory is None:
                        row["inventory_build_or_terminal_empty_calls"] = row.get("inventory_build_or_terminal_empty_calls", 0)+1
                    started = time.perf_counter()
                    try:
                        result = original(instance, *args, **kwargs)
                        row[label+"_returned"] = row.get(label+"_returned", 0)+1
                        return result
                    except BaseException:
                        row[label+"_raised"] = row.get(label+"_raised", 0)+1
                        raise
                    finally:
                        row[label+"_inclusive_seconds"] = row.get(label+"_inclusive_seconds", 0.)+time.perf_counter()-started
                return measured
            setattr(cls, name, factory(original, label))
        return self

    def __exit__(self, *unused):
        for cls, name, original in reversed(self.originals): setattr(cls, name, original)

    @contextmanager
    def scope(self, phase):
        previous = self.phase; self.phase = phase; started = time.perf_counter()
        before = self.budget.snapshot()["native_preview_entries"]
        try: yield
        finally:
            row = self.rows.setdefault(phase, {})
            row["complete_wall_seconds"] = row.get("complete_wall_seconds", 0.)+time.perf_counter()-started
            after = self.budget.snapshot()["native_preview_entries"]
            row["native_preview_entries"] = None if before is None or after is None else after-before
            self.phase = previous


def _collect(base, context, *, policy=None, generator=None, actions=None, output, guard):
    worker = base.planning_clone(); context.require_task(worker)
    rows = []; selected = []; decisions = []
    before = None if policy is None else parameter_hash(policy)
    while not worker.terminated:
        guard(); context.require_task(worker)
        observation = worker.observation(); context.require_observations((observation,))
        if actions is not None:
            if len(rows) >= len(actions): raise ValueError("Incomplete supplied teacher")
            action = actions[len(rows)]
        else:
            with torch.no_grad():
                logits, _ = policy(observation)
                index = int(logits.argmax()) if generator is None else int(torch.multinomial(logits.softmax(-1), 1, generator=generator))
            action = observation.action_ids[index]
        decision = {"step": len(rows), "observation_hash": observation.fingerprint,
            "action_ids": list(observation.action_ids), "action_mask": observation.action_mask.tolist(),
            "action_id": action, "behavior_parameter_hash": before}
        _write(output / ("attempt-%02d.json" % len(rows)), decision)
        try:
            outcome = worker.step(action)
        except BaseException as error:
            _write(output / ("transition-failure-%02d.json" % len(rows)), {
                "error_type": type(error).__name__, "message": str(error),
                "committed": bool(getattr(error, "committed", False)),
                "committed_info": getattr(error, "info", None), "retained_metrics": worker.metrics()})
            raise
        selected.append(action); rows.append(SpatialTransition(observation, action, outcome.reward, outcome.terminated))
        decisions.append({**decision, "reward": outcome.reward, "terminated": outcome.terminated})
        _write(output / ("returned-%02d.json" % (len(rows)-1)), decisions[-1])
    if actions is not None and len(rows) != len(actions): raise ValueError("Teacher continues after termination")
    if policy is not None and parameter_hash(policy) != before: raise ValueError("Collection changed policy")
    trace = admit_collected_trace(context, rows, worker, behavior_parameter_hash=before)
    _write(output / "complete-trace.json", {"trace_seal": trace.seal_hash,
        "decisions": decisions, "metrics": worker.metrics(), "context_hash": context.fingerprint})
    return trace


def _seal_and_replay(base, context, trace, *, method, policy, updates, output, guard):
    trace.require(); guard()
    plan = freeze_json({"context_hash": context.fingerprint, "source_hash": base.case.source_hash,
        "decision_model_hash": base.decision_model_hash,
        "initial_observation_hash": trace.transitions[0].observation.fingerprint,
        "max_steps": base.max_steps, "actions": [t.action_id for t in trace.transitions],
        "history": trace.history, "terminal_reason": "STOP" if trace.transitions[-1].action_id == "STOP" else "HORIZON",
        "parameter_hash": None if policy is None else parameter_hash(policy),
        "architecture_hash": None if policy is None else policy.architecture_hash,
        "learning_updates": updates})
    seal = semantic_digest(plan)
    _write(output / "plan.json", {"plan": plan, "plan_seal": seal})
    replay = base.fresh(); context.require_task(replay)
    for index, action in enumerate(plan["actions"]):
        guard(); _write(output / ("replay-attempt-%02d.json" % index), {"action_id": action})
        try: replay.step(action)
        except BaseException as error:
            _write(output / "replay-failure.json", {"message": str(error), "metrics": replay.metrics(),
                "committed": bool(getattr(error, "committed", False)), "info": getattr(error, "info", None)})
            raise
    if not replay.terminated or _history_identity(replay.metrics()["history"]) != _history_identity(plan["history"]):
        raise ValueError("Complete native replay differs from sealed public plan")
    audit = evaluate_native_spatial_episode(replay, cancelled=lambda: (guard() or False))
    # A rejected route is an experimental result too. Persist its certificate
    # before failing the export so the offending action and reason survive.
    _write(output / "native-replay.json", {"metrics": replay.metrics(), "independent_geometry": audit})
    if audit["accepted"] is not True:
        reasons = audit.get("geometry", {}).get("failures", ())
        raise ValueError("Independent native geometry rejected replay: " + ", ".join(reasons))
    # Legacy aspiration STOP records omit interaction_mode. Normalize only the
    # presentation copy; authoritative histories, hashes and poses stay exact.
    display_history = []
    for row in replay.metrics()["history"]:
        mode = "stop" if row["action_id"] == "STOP" else "aspirate"
        if row.get("interaction_mode", mode) != mode:
            raise ValueError("Aspiration-only export received another native mode")
        display_history.append({**row, "interaction_mode": mode})
    frames = replay_frames(replay, display_history)
    _write(output / "frames.json", {"schema": "patient-native-replay-frames-v1", "plan_seal": seal, "frames": frames})
    return {"method": method, "plan": plan, "plan_seal": seal,
        "replayed_history": replay.metrics()["history"], "independent_geometry": audit}


def run_patient_preflight(public_task_factory, *, protocol, output):
    """One fixed TRAIN source; no private loader or population/adaptation execution.

The owning worker binds factory/QC/release bytes before entry. Use the existing
owned parent for hard time and process-tree RSS; PlanningBudget is cooperative.
"""
    output = Path(output); output.mkdir(parents=True, exist_ok=False)
    protocol = freeze_json(protocol); limits = protocol
    if (limits["max_optimizer_updates"] != 2 or limits["threads"] != 1
            or limits["learning_protocol_hash"] != semantic_digest(PREFLIGHT_PROTOCOL)):
        raise ValueError("Preflight requires its exact one-update-per-method protocol")
    torch.set_num_threads(1); torch.set_num_interop_threads(1); torch.use_deterministic_algorithms(True)
    budget = PlanningBudget(NativeResectionEngine, max_native_previews=limits["max_native_previews"],
        seconds=limits["worker_seconds"])
    costs = _CallCosts(budget, limits["max_policy_forwards"])
    result = {"version": VERSION, "status": "started", "protocol": protocol,
        "learning_protocol": PREFLIGHT_PROTOCOL, "methods": {}, "plans": [],
        "private_reference_reads": 0, "checkpoint_loads": 0, "optimizer_updates": 0,
        "population_training": False, "patient_adaptation": False,
        "ventricular_avoidance_tested": False, "clinical_claim": False}
    started = time.perf_counter()
    try:
        with budget, costs:
            with costs.scope("public_preparation"):
                base, context = public_task_factory()
                if type(context) is not PatientPlanningContext: raise TypeError("Exact patient context required")
                context.require_training(); context.require_task(base)
                if context.protocol_sha256 != semantic_digest(protocol): raise ValueError("Factory used another protocol")
                if base.terminated or base.observation().state_features[0] != 0 or base.metrics()["planning_estimator_only"] is not False: raise ValueError("Fresh task required")
                result["patient_context"] = context.record(); _write(output / "context.json", context.record())
                il, rl = common_patient_policies((context,), PREFLIGHT_PROTOCOL)
                initial_hash = parameter_hash(il); result["initial_parameter_hash"] = initial_hash
                sessions = {name: PatientTrainSession((context,), name, policy, initial_parameter_hash=initial_hash,
                    protocol=PREFLIGHT_PROTOCOL) for name, policy in (("IL", il), ("RL", rl))}
            with costs.scope("offline_teacher_search"):
                try:
                    actions, stats = observed_beam_search(base, **dict(limits["search"]),
                        objective_source="supplied_public_whole_tumor_and_frozen_geometric_costs",
                        transition_mode="lazy_planning")
                    _write(output / "teacher-search.json", {"actions": actions, "accounting": stats})
                except BaseException as error:
                    _write(output / "teacher-search-failure.json", {"type": type(error).__name__,
                        "message": str(error), "accounting": getattr(error, "accounting", None)})
                    raise
                if stats["call_cap_reached"] or stats["time_cap_reached"]:
                    raise InterruptedError("Teacher unresolved at declared cap; no label or replacement")
            directory = output / "SEARCH"; directory.mkdir()
            with costs.scope("offline_teacher_trace_replay"):
                teacher = _collect(base, context, actions=actions, output=directory, guard=budget.check)
                result["plans"].append(_seal_and_replay(base, context, teacher, method="SEARCH", policy=None,
                    updates=0, output=directory, guard=budget.check))
            for method, policy in (("IL", il), ("RL", rl)):
                directory = output / method; directory.mkdir()
                session = sessions[method]
                if method == "RL":
                    collection = directory / "collection"; collection.mkdir()
                    generator = torch.Generator(device="cpu").manual_seed(PREFLIGHT_PROTOCOL["seed"]+1)
                    with costs.scope("offline_RL_collection"):
                        trace = _collect(base, context, policy=policy, generator=generator,
                            output=collection, guard=budget.check)
                        _seal_and_replay(base, context, trace, method="RL_COLLECTION", policy=policy,
                            updates=0, output=collection, guard=budget.check)
                with costs.scope("offline_"+method+"_update"):
                    loss, loss_record = (patient_imitation_loss(session,
                        [PatientImitationSample(teacher, i) for i in range(len(teacher.transitions))])
                        if method == "IL" else patient_reinforce_loss(session, (trace,)))
                    _write(directory / "loss-before-update.json", loss_record)
                    budget.check(); update = patient_gradient_step(session, loss)
                    result["optimizer_updates"] += update["optimizer_updates"]
                    _write(directory / "update.json", update)
                    result["methods"][method] = {"loss": loss_record, "update": update}
                inference = directory / "greedy"; inference.mkdir()
                with costs.scope("deployment_"+method+"_greedy_and_replay"):
                    greedy = _collect(base, context, policy=policy, output=inference, guard=budget.check)
                    result["plans"].append(_seal_and_replay(base, context, greedy, method=method,
                        policy=policy, updates=session.updates, output=inference, guard=budget.check))
            record = {"version": VERSION, "status": "complete", "scope": "target_only_geometry_preflight",
                "patient_context": context.record(), "source_hash": base.case.source_hash,
                "decision_model_hash": base.decision_model_hash,
                "initial_observation_hash": base.observation().fingerprint,
                "source_grid": {"shape": list(base.case.observed_support.shape),
                    "affine_ras_mm": base.case._native_affine_ras_mm.tolist()},
                "tools": [asdict(tool) for tool in base.case.tools],
                "expected_methods": ["SEARCH", "IL", "RL"], "plans": result["plans"]}
            _write(output / "sealed-complete-plans.json", record)
            result["sealed_plans_identity"] = semantic_digest(record)
            budget.complete(history_complete=True)
        result["status"] = "complete_preflight_not_population_evidence"
    except BaseException as error:
        result.update(status="failed_or_unresolved", failure={"type": type(error).__name__, "message": str(error)})
        raise
    finally:
        result["complete_wall_seconds"] = time.perf_counter()-started
        result["costs"] = costs.rows; result["native_budget"] = budget.snapshot()
        result["total_policy_forward_calls"] = costs.forwards
        result["cost_scope"] = "includes_public_factory_and_all_training_planning_replay;phase_subcalls_inclusive_not_additive"
        result["resource_authority"] = "existing_owned_parent_receipt_required_for_hard_wall_and_tree_RSS"
        _write(output / "result.json", result)
    return result
