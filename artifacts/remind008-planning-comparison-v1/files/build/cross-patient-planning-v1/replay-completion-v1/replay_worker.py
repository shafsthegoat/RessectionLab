"""Finish one already-sealed RL native replay; no planning or learning."""
import argparse
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import signal
import sys
import time

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT/"src"))
sys.path.insert(0, str(Path(__file__).parent))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--release", type=Path, required=True)
    parser.add_argument("--release-sha256", required=True)
    args = parser.parse_args()
    from resectionlab.legacy_transfer_worker import _require_parent_lease
    _require_parent_lease()
    raw = args.release.read_bytes()
    if hashlib.sha256(raw).hexdigest() != args.release_sha256: raise ValueError("Replay release changed")
    release = json.loads(raw); output = ROOT/release["output"]; output.mkdir(exist_ok=False)
    started = time.perf_counter(); budget = None; task = None; originals = []
    result = {"status": "started", "scope": "saved_RL_plan_replay_completion_only",
        "optimizer_updates": 0, "optimizer_construction_attempts": 0, "policy_forward_calls": 0, "policy_constructions": 0,
        "search_calls": 0, "checkpoint_loads": 0, "private_array_reads": 0,
        "replay_transition_calls": 0, "replay_restart_from_initial_state": True,
        "original_attempt_status_preserved": "failed_or_unresolved"}

    def write(name, value):
        with (output/name).open("x") as stream:
            json.dump(value, stream, indent=2, allow_nan=False); stream.write("\n")
    def progress(phase, **details):
        row = {"phase": phase, "seconds": time.perf_counter()-started, **details}
        temporary = output/"progress.tmp"; temporary.write_text(json.dumps(row)+"\n")
        temporary.replace(output/"progress.json")
    def deadline(signum, frame): raise TimeoutError("Fixed50s replay-worker deadline")
    previous_handler = signal.signal(signal.SIGALRM, deadline); signal.setitimer(signal.ITIMER_REAL, 50.)
    try:
        index_record = release["input_index"]; raw_index = (ROOT/index_record["path"]).read_bytes()
        if hashlib.sha256(raw_index).hexdigest() != index_record["sha256"]: raise ValueError("Replay input index changed")
        index = json.loads(raw_index)
        def read(name):
            row = index["files"][name]; raw = (ROOT/row["path"]).read_bytes()
            if hashlib.sha256(raw).hexdigest() != row["sha256"]: raise ValueError("Saved input changed: "+name)
            return json.loads(raw)
        progress("validate_original_seals_and_receipts")
        from resectionlab.core import semantic_digest
        context_record = read("context"); bindings = read("bindings")
        prior_release = read("original_release"); prior_parent = read("original_parent")
        prior_result = read("original_result"); prior_preflight = read("original_preflight")
        if (prior_parent["worker_termination_confirmed"] is not True or prior_parent["final_owned_pids"]
                or prior_parent["cleanup_errors"] or prior_result["status"] != "failed_or_unresolved"
                or prior_preflight["optimizer_updates"] != 2):
            raise ValueError("Original failed attempt is not the expected cleanly terminated v2")
        plans = {method: read(method+"_plan") for method in ("SEARCH", "IL", "RL")}
        for method, envelope in plans.items():
            plan = envelope["plan"]
            if (envelope["plan_seal"] != semantic_digest(plan)
                    or plan["context_hash"] != semantic_digest(context_record)
                    or plan["source_hash"] != context_record["source_hash"]
                    or plan["decision_model_hash"] != context_record["decision_model_hash"]
                    or plan["max_steps"] != 6
                    or plan["learning_updates"] != (0 if method == "SEARCH" else 1)):
                raise ValueError("Original sealed method identity differs")
        initial_hash = plans["RL"]["plan"]["initial_observation_hash"]
        if any(row["plan"]["initial_observation_hash"] != initial_hash for row in plans.values()):
            raise ValueError("Original methods do not share their initial public observation")

        def history_identity(rows):
            return semantic_digest([{k:v for k,v in row.items() if k != "outcome_scope"} for row in rows])
        def accepted(method, native):
            plan = plans[method]["plan"]; audit = native["independent_geometry"]; history = native["metrics"]["history"]
            if (history_identity(history) != history_identity(plan["history"])
                    or audit.get("accepted") is not True or audit.get("complete_episode") is not True
                    or audit["geometry"].get("feasible") is not True
                    or audit["geometry"].get("complete_tool_checked") is not True
                    or audit["committed_history_hash"] != semantic_digest(history)
                    or audit["source_hash"] != plan["source_hash"]
                    or audit["decision_model_hash"] != plan["decision_model_hash"]):
                raise ValueError("Saved complete native evidence differs from sealed plan")
            return {"method": method, **plans[method], "replayed_history": history,
                    "independent_geometry": audit}
        completed = [accepted(method, read(method+"_native")) for method in ("SEARCH", "IL")]
        trace = read("RL_trace"); rl_plan = plans["RL"]["plan"]
        if (trace["context_hash"] != semantic_digest(context_record)
                or semantic_digest(trace["metrics"]["history"]) != semantic_digest(rl_plan["history"])
                or len(trace["decisions"]) != 6 or len(rl_plan["actions"]) != 6):
            raise ValueError("Complete original RL decision trace differs")

        # PlanningBudget imports the existing tensor-aware diagnostics module;
        # forbid every model forward, construction, optimizer creation and search.
        import torch
        from resectionlab.spatial_policy import SpatialPolicy
        import resectionlab.observed_search as search
        def forbid(owner, name, counter):
            original = getattr(owner, name); originals.append((owner, name, original))
            def refused(*args, **kwargs):
                result[counter] += 1
                raise RuntimeError("Forbidden replay-only operation: "+counter)
            setattr(owner, name, refused)
        forbid(torch.nn.Module, "_call_impl", "policy_forward_calls")
        forbid(SpatialPolicy, "__init__", "policy_constructions")
        forbid(torch.optim.Optimizer, "__init__", "optimizer_construction_attempts")
        forbid(torch, "load", "checkpoint_loads")
        forbid(search, "observed_beam_search", "search_calls")
        from resectionlab.native_resection import NativeResectionEngine
        from resectionlab.patient_planning_admission import make_patient_planning_task
        from resectionlab.planning_budget import PlanningBudget
        from resectionlab.native_spatial_evaluation import evaluate_native_spatial_episode
        from resectionlab.development_episode import replay_frames
        from public_patient_factory import prepare_public_source
        budget = PlanningBudget(NativeResectionEngine, max_native_previews=512, seconds=50.)
        with budget:
            source, binding, qc, protocol = prepare_public_source(output, prior_release,
                index["files"]["original_release"]["sha256"], bindings["protocol"], progress)
            if binding != bindings["source_binding"] or qc != bindings["qc"]:
                raise ValueError("Original public/QC bindings differ after reconstruction")
            progress("before_replay_task_initial_inventory")
            task, context = make_patient_planning_task(source,
                cohort_bytes=(ROOT/"manifests/experiments/remind-component-cohort-v1.json").read_bytes(),
                source_binding=binding, qc_receipt=qc, protocol=protocol)
            if context.record() != context_record or task.observation().fingerprint != initial_hash:
                raise ValueError("Replay changed original context or initial observation")
            for step, (action, decision) in enumerate(zip(rl_plan["actions"], trace["decisions"])):
                budget.check()
                if action != decision["action_id"] or task.observation().fingerprint != decision["observation_hash"]:
                    raise ValueError("Saved RL action/observation no longer matches native state")
                progress("replay_step", step=step, action_id=action)
                write("replay-attempt-%02d.json" % step, {"step": step, "action_id": action})
                result["replay_transition_calls"] += 1
                outcome = task.step(action)
                if semantic_digest({k:v for k,v in outcome.info.items() if k != "outcome_scope"}) != semantic_digest(
                        {k:v for k,v in rl_plan["history"][step].items() if k != "outcome_scope"}):
                    raise ValueError("Replayed committed record differs from original seal")
                write("replay-returned-%02d.json" % step, {"reward": outcome.reward, "terminated": outcome.terminated})
            if not task.terminated: raise ValueError("Incomplete saved RL replay")
            progress("independent_full_tool_audit")
            audit = evaluate_native_spatial_episode(task, cancelled=lambda: (budget.check() or False))
            native = {"metrics": task.metrics(), "independent_geometry": audit}
            completed.append(accepted("RL", native)); write("RL-native-replay.json", native)
            display = [{**row, "interaction_mode": "stop" if row["action_id"] == "STOP" else "aspirate"}
                       for row in native["metrics"]["history"]]
            frames = replay_frames(task, display)
            write("RL-frames.json", {"schema": "patient-native-replay-frames-v1",
                  "plan_seal": plans["RL"]["plan_seal"], "frames": frames})
            record = {"version": "patient-native-preflight-sealed-plans-v1", "status": "complete",
                "scope": "target_only_geometry_preflight", "patient_context": context_record,
                "source_hash": source.source_hash, "decision_model_hash": task.decision_model_hash,
                "initial_observation_hash": initial_hash,
                "source_grid": {"shape": list(source.observed_support.shape),
                    "affine_ras_mm": source._native_affine_ras_mm.tolist()},
                "tools": [asdict(tool) for tool in source.tools],
                "expected_methods": ["SEARCH", "IL", "RL"], "plans": completed,
                "continuation": {"kind": "saved_RL_replay_restart_only", "release_sha256": args.release_sha256,
                    "original_release_sha256": index["files"]["original_release"]["sha256"],
                    "original_attempt_status": prior_result["status"],
                    "reused_methods": ["SEARCH", "IL"], "input_index_sha256": index_record["sha256"],
                    "zero_new_search_or_learning": True}}
            write("sealed-complete-plans.json", record)
            budget.complete(history_complete=True)
        result.update(status="complete_saved_plan_replay_comparison", plan_seals={k:v["plan_seal"] for k,v in plans.items()},
            sealed_bundle_sha256=hashlib.sha256((output/"sealed-complete-plans.json").read_bytes()).hexdigest(),
            reused_native_receipts={k:index["files"][k+"_native"] for k in ("SEARCH", "IL")},
            original_costs={"owned_wall_seconds": prior_parent["elapsed_seconds"],
                "sampled_peak_rss_bytes": prior_parent["sampled_peak_rss_bytes"],
                "optimizer_updates": prior_preflight["optimizer_updates"],
                "policy_forward_calls": prior_preflight["total_policy_forward_calls"],
                "native_budget": prior_preflight["native_budget"]},
            additional_cost_scope="restarted_RL_native_replay_and_audit; original failed replay cost is retained",
            private_evaluation_performed=False)
    except BaseException as error:
        result.update(status="failed_or_unresolved", failure={"type":type(error).__name__, "message":str(error)})
        if task is not None: write("partial-native-metrics.json", task.metrics())
        raise
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0.); signal.signal(signal.SIGALRM, previous_handler)
        for owner, name, original in reversed(originals): setattr(owner, name, original)
        result["additional_worker_wall_seconds"] = time.perf_counter()-started
        result["native_budget"] = None if budget is None else budget.snapshot()
        write("result.json", result)


if __name__ == "__main__": main()
