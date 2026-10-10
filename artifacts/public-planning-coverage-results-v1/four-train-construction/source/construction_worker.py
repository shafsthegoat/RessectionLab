"""One sequential four-TRAIN construction/inventory diagnostic, no actions."""
import argparse
from collections import Counter
import gc
import hashlib
import json
import os
from pathlib import Path
import resource
import signal
import sys
import time

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT/"src"))
sys.path.insert(0, str(ROOT/"build/goal-conditioned-policy-v1"))
SUBJECTS = ("ReMIND-008", "ReMIND-010", "ReMIND-020", "ReMIND-025")
BASELINE_008 = "sha256:a580cc297211c3e0dde31a90a133e53f1bc5df336fa3d10705285fcfaad44f64"
MAX_PREVIEWS, WORKER_SECONDS, MEMORY_BYTES = 512, 110, 3*1024**3


def read_bound(record):
    path = ROOT/record["path"]; raw = path.read_bytes()
    if hashlib.sha256(raw).hexdigest() != record["sha256"]: raise ValueError("Bound metadata changed: "+str(path))
    return json.loads(raw)


def write(path, value):
    with Path(path).open("x") as stream:
        json.dump(value, stream, indent=2, allow_nan=False); stream.write("\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--declaration", type=Path, required=True)
    parser.add_argument("--declaration-sha256", required=True)
    args = parser.parse_args()
    from resectionlab.legacy_transfer_worker import _require_parent_lease
    _require_parent_lease()
    declaration = read_bound({"path": str(args.declaration.resolve()), "sha256": args.declaration_sha256})
    output = ROOT/declaration["output"]; output.mkdir(exist_ok=False)
    started = time.perf_counter(); budget = None; originals = []; active_paths = set()
    result = {"status": "started", "scope": "four_frozen_TRAIN_public_source_task_initial_inventory_only",
        "cases": [{"patient_id": s, "status": "not_attempted"} for s in SUBJECTS],
        "model_calls": 0, "optimizer_calls": 0, "checkpoint_loads": 0, "search_calls": 0,
        "transition_calls": 0, "blocked_file_or_external_calls": 0, "public_array_loads": [],
        "source_arrays_written": 0, "failed_cases_replaced": False, "all_four_cases_retained": True}
    def deadline(signum, frame): raise TimeoutError("Construction worker deadline or termination")
    handlers = {s: signal.signal(s, deadline) for s in (signal.SIGALRM, signal.SIGTERM)}
    signal.setitimer(signal.ITIMER_REAL, WORKER_SECONDS)
    try:
        index = read_bound(declaration["public_index"])
        if tuple(r["patient_id"] for r in index["cases"]) != SUBJECTS: raise ValueError("Frozen TRAIN order changed")
        cohort_record = declaration["cohort"]
        cohort_bytes = (ROOT/cohort_record["path"]).read_bytes()
        if hashlib.sha256(cohort_bytes).hexdigest() != cohort_record["sha256"]: raise ValueError("Cohort changed")
        import numpy as np
        import torch
        torch.set_num_threads(1); torch.set_num_interop_threads(1)
        from darwin_fast_sampler import FastDarwinSampler
        from resectionlab.core import semantic_digest
        from resectionlab.native_resection import NativeResectionEngine
        from resectionlab.native_spatial_task import NativeSpatialTask
        from resectionlab.patient_planning_admission import make_patient_planning_task
        from resectionlab.patient_planning_learning import PREFLIGHT_PROTOCOL
        from resectionlab.public_patient_factory import prepare_public_source
        from resectionlab.planning_budget import PlanningBudget, PlanningBudgetViolation
        from resectionlab.spatial_policy_diagnostics import runtime_proposal_coverage
        import resectionlab.observed_search as search
        sampler = FastDarwinSampler()

        def forbid(owner, name, counter):
            original = getattr(owner, name); originals.append((owner, name, original))
            def refused(*args, **kwargs):
                result[counter] += 1
                raise RuntimeError("Forbidden construction-only call: "+counter)
            setattr(owner, name, refused)
        forbid(torch.nn.Module, "__init__", "model_calls")
        forbid(torch.nn.Module, "_call_impl", "model_calls")
        forbid(torch.optim.Optimizer, "__init__", "optimizer_calls")
        forbid(torch, "load", "checkpoint_loads")
        forbid(search, "observed_beam_search", "search_calls")
        for name in ("step", "advance_planning", "planning_clone"):
            forbid(NativeSpatialTask, name, "transition_calls")
        original_load = np.load; originals.append((np, "load", original_load))
        def public_load(path, *args, **kwargs):
            resolved = str(Path(path).resolve())
            if resolved not in active_paths: raise ValueError("Array outside current four public inputs")
            result["public_array_loads"].append(resolved)
            return original_load(path, *args, **kwargs)
        np.load = public_load
        def audit(event, args):
            if event == "open" and args and isinstance(args[0], (str, bytes)):
                path = Path(os.fsdecode(args[0])).resolve()
                if (path.suffix.lower() == ".npy" and str(path) not in active_paths
                        or path.suffix.lower() == ".dcm" or path.is_relative_to(ROOT/"data")):
                    result["blocked_file_or_external_calls"] += 1
                    raise PermissionError("Construction accepts only current public arrays")
            if event in ("subprocess.Popen", "os.system", "os.fork", "socket.connect", "socket.bind"):
                result["blocked_file_or_external_calls"] += 1
                raise PermissionError("No external work in construction worker")
        sys.addaudithook(audit)
        def rss(): return sampler.process_resident_bytes(os.getpid())
        def guard():
            if rss() > MEMORY_BYTES: raise MemoryError("Construction worker RSS cap")
            if time.perf_counter()-started >= WORKER_SECONDS: raise TimeoutError("Construction worker time cap")
            budget.check()
        def construct(entry):
            subject = entry["patient_id"]; case_output = output/subject; case_output.mkdir()
            case_started = time.perf_counter(); source = task = context = observation = inventory = coverage = None
            row = {"patient_id": subject, "role": "TRAIN", "status": "started", "rss_before_bytes": rss(), "phase_marks": []}
            before = budget.snapshot()["native_preview_entries"]
            def progress(phase, **details):
                guard(); row["phase_marks"].append({"phase": phase, "seconds": time.perf_counter()-case_started, "rss_bytes": rss(), **details})
            try:
                manifest = read_bound(entry)
                keys = ("image", "supplied_support", "supplied_whole_tumor", "whole_tumor_domain")
                if set(manifest["input_files"]) != set(keys): raise ValueError("Four public files required")
                active_paths.clear(); active_paths.update(str(Path(manifest["input_files"][k]["path"]).resolve()) for k in keys)
                limits = {"max_steps": 6, "max_optimizer_updates": 0, "max_native_previews": MAX_PREVIEWS,
                    "max_policy_forwards": 1, "worker_seconds": WORKER_SECONDS, "memory_bytes": MEMORY_BYTES,
                    "threads": 1, "search": {"max_calls": 1, "beam_width": 1, "seconds": 1}}
                release = {"limits": limits}
                if subject == "ReMIND-008": release["baseline_public_source_hash"] = BASELINE_008
                source, binding, qc, protocol = prepare_public_source(case_output, release, args.declaration_sha256, None, progress,
                    public_manifest_path=entry["path"], public_manifest_sha256=entry["sha256"], cohort_bytes=cohort_bytes,
                    learning_protocol_hash=semantic_digest(PREFLIGHT_PROTOCOL))
                if subject == "ReMIND-008" and source.source_hash != BASELINE_008: raise ValueError("008 baseline changed")
                progress("before_admitted_task_initial_inventory")
                task, context = make_patient_planning_task(source, cohort_bytes=cohort_bytes,
                    source_binding=binding, qc_receipt=qc, protocol=protocol)
                inventory = task.candidate_inventory(); observation = task.observation()
                if inventory["steps_taken"] != 0 or inventory["max_steps"] != 6: raise ValueError("Initial six-step task required")
                coverage = runtime_proposal_coverage(source, inventory, observation, native_affine=source._native_affine_ras_mm)
                write(case_output/"initial-inventory.json", inventory)
                write(case_output/"initial-public-proposal-coverage.json", coverage)
                write(case_output/"context.json", context.record())
                families = {}
                for item in inventory["emitted"]:
                    count = families.setdefault(item["family"], {"emitted": 0, "accepted": 0, "reasons": Counter()})
                    count["emitted"] += 1; count["accepted"] += bool(item["feasible"]); count["reasons"][item["reason"]] += 1
                row.update(status="constructed_initial_inventory_only", source_hash=source.source_hash,
                    context_hash=context.fingerprint, families=families,
                    emitted_count=inventory["emitted_count"], accepted_count=inventory["accepted_count"],
                    proposal_dispositions=inventory["disposition_counts"],
                    endpoint_centers_in_actor_crop=inventory["endpoint_centers_in_actor_crop"],
                    target_denominator=coverage["nominal_target_centers_total"],
                    target_depth_range_mm=coverage["nominal_target_center_depth_range_mm"],
                    accepted_envelope=coverage["accepted_envelope"], emitted_envelope=coverage["emitted_envelope"],
                    actor_crop_shape=list(observation.image_channels.shape[1:]),
                    initial_observation_hash=observation.fingerprint, array_outputs_written=False)
                progress("case_inventory_retained")
            except (TimeoutError, MemoryError, PlanningBudgetViolation) as error:
                row.update(status="fatal_runtime_stop", error=type(error).__name__+":"+str(error)); raise
            except Exception as error:
                row.update(status="case_failed_retained", error=type(error).__name__+":"+str(error))
            finally:
                source = task = context = observation = inventory = coverage = None
                active_paths.clear(); gc.collect()
                after = budget.snapshot()["native_preview_entries"]
                row.update(elapsed_seconds=time.perf_counter()-case_started, rss_after_cleanup_bytes=rss(),
                    cumulative_process_peak_rss_bytes=int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss),
                    phase_sampled_peak_rss_bytes=max([row["rss_before_bytes"]]+[r["rss_bytes"] for r in row["phase_marks"]]),
                    native_preview_entries=None if before is None or after is None else after-before,
                    cleanup="source/task/context/observation/inventory references released; gc collected before next case")
                result["cases"][SUBJECTS.index(subject)] = row
                write(case_output/"construction-result.json", row)
            return row
        budget = PlanningBudget(NativeResectionEngine, max_native_previews=MAX_PREVIEWS, seconds=WORKER_SECONDS)
        with budget:
            for entry in index["cases"]:
                guard(); construct(entry); guard()
            write(output/"construction-ledger.json", result["cases"])
            # Only this construction ledger is complete; no episode was run.
            budget.complete(history_complete=True)
        result["status"] = "construction_batch_complete" if all(r["status"] == "constructed_initial_inventory_only" for r in result["cases"]) else "construction_batch_complete_with_case_failures"
    except BaseException as error:
        result.update(status="fatal_runtime_or_setup_failure", error=type(error).__name__+":"+str(error))
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        for owner, name, original in reversed(originals): setattr(owner, name, original)
        result.update(elapsed_seconds=time.perf_counter()-started,
            cumulative_process_peak_rss_bytes=int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss),
            native_budget=None if budget is None else budget.snapshot(),
            budget_history_scope="construction ledger only; zero transitions or episodes", default_proposal_config=True,
            methods_executed=[], clinical_claim=False)
        write(output/"result.json", result)
        for sig, handler in handlers.items(): signal.signal(sig, handler)
    return 0 if result["status"].startswith("construction_batch_complete") else 1


if __name__ == "__main__": raise SystemExit(main())
