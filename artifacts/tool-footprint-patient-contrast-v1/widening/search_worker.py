"""One equal-depth committed-volume retention contrast over the unchanged footprint task."""
import argparse
from dataclasses import asdict, replace
import hashlib
import json
from pathlib import Path
import signal
import sys
import time

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "build/cross-patient-planning-v1/replay-completion-v1"))


def inventory_identity(actual, expected):
    """Compare exact canonical JSON values, allowing only tuple/list encoding."""
    canonical = lambda value: json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)
    actual_text, expected_text = canonical(actual), canonical(expected)
    actual, expected = json.loads(actual_text), json.loads(expected_text)
    paths = []
    def compare(a, b, path):
        if type(a) is not type(b):
            paths.append(path)
        elif isinstance(a, dict):
            for key in sorted(set(a) | set(b)):
                if key not in a or key not in b: paths.append(path+"."+key)
                else: compare(a[key], b[key], path+"."+key)
        elif isinstance(a, list):
            if len(a) != len(b): paths.append(path+".length")
            for i, (left, right) in enumerate(zip(a, b)): compare(left, right, path+"["+str(i)+"]")
        elif a != b:
            paths.append(path)
    compare(actual, expected, "inventory")
    return {"equal": actual_text == expected_text,
        "actual_canonical_sha256": hashlib.sha256(actual_text.encode()).hexdigest(),
        "expected_canonical_sha256": hashlib.sha256(expected_text.encode()).hexdigest(),
        "mismatch_field_paths": paths,
        "comparison": "canonical JSON; tuple/list share array encoding; values/order/types otherwise exact"}


def retained_progress(accounting):
    """Separate measured removal depth from the existing tip-insertion scalar."""
    layers = []
    for layer in accounting["layers"]:
        opening = {tuple(row["actions"]): row for row in layer.get("opening_depth_retention", [])}
        diagnostics = layer.get("retained_prefix_diagnostics", {})
        rows = []
        for name in ("retained_nonterminal_prefixes", "best_terminal_prefixes"):
            for row in diagnostics.get(name, []):
                depth = opening.get(tuple(row["actions"]), {})
                rows.append({**row, "selection": name,
                    "deepest_newly_removed_actor_covered_cell_center_mm": depth.get("opening_depth_mm"),
                    "retention_reason": depth.get("reason"),
                    "cumulative_actor_covered_removed_volume_mm3": depth.get("opening_removed_volume_mm3"),
                    "removal_depth_status": "recorded" if depth else "not_recorded_for_this_terminal_prefix"})
        layers.append({"depth": layer["depth"], "completed": layer["completed"], "prefixes": rows})
    return layers


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--release", type=Path, required=True)
    parser.add_argument("--release-sha256", required=True)
    args = parser.parse_args()
    from resectionlab.legacy_transfer_worker import _require_parent_lease
    _require_parent_lease()
    raw = args.release.read_bytes()
    if hashlib.sha256(raw).hexdigest() != args.release_sha256: raise ValueError("Diagnostic release changed")
    release = json.loads(raw); output = ROOT / release["output"]; output.mkdir(exist_ok=False)
    started = time.perf_counter(); budget = None; originals = []; observed_inventories = {}
    result = {"status": "started", "scope": "same_public_world_equal_depth_volume_six_step_beam",
        "optimizer_updates": 0, "optimizer_construction_attempts": 0, "policy_forward_calls": 0,
        "policy_constructions": 0, "checkpoint_loads": 0, "private_array_reads": 0,
        "search_invocations": 0, "native_replay_calls": 0, "original_attempt_unchanged": True}

    def write(name, value):
        with (output / name).open("x") as stream:
            json.dump(value, stream, indent=2, allow_nan=False); stream.write("\n")
    def progress(phase, **details):
        row = {"phase": phase, "seconds": time.perf_counter()-started, **details}
        temporary = output / "progress.tmp"; temporary.write_text(json.dumps(row)+"\n")
        temporary.replace(output / "progress.json")
    def deadline(signum, frame): raise TimeoutError("Fixed 110 s opening-volume worker deadline")
    previous_handler = signal.signal(signal.SIGALRM, deadline); signal.setitimer(signal.ITIMER_REAL, 110.)
    try:
        index_record = release["input_index"]; raw_index = (ROOT / index_record["path"]).read_bytes()
        if hashlib.sha256(raw_index).hexdigest() != index_record["sha256"]: raise ValueError("Input index changed")
        index = json.loads(raw_index)
        def read(name):
            row = index["files"][name]; raw = (ROOT / row["path"]).read_bytes()
            if hashlib.sha256(raw).hexdigest() != row["sha256"]: raise ValueError("Saved input changed: "+name)
            return json.loads(raw)
        context_record = read("context"); bindings = read("bindings")
        prior_release = read("original_release"); baseline = read("original_search")
        prior_contrast = read("prior_contrast_result"); prior_parent = read("prior_contrast_parent")
        if (prior_contrast["status"] != "complete_tool_footprint_opening_search_contrast"
                or prior_parent["status"] != "complete" or prior_parent["exit_code"] != 0
                or prior_parent["cleanup_errors"] or prior_parent["final_owned_pids"]):
            raise ValueError("Saved footprint comparison was not a clean complete result")
        result["saved_baseline_cost"] = {
            "search_seconds": prior_contrast["search_seconds"],
            "model_transition_calls": prior_contrast["model_transition_calls"],
            "native_previews": prior_contrast["native_budget"]["native_preview_entries"],
            "owned_wall_seconds": prior_parent["elapsed_seconds"],
            "sampled_peak_rss_bytes": prior_parent["sampled_peak_rss_bytes"],
            "costs_remain_separate_from_this_attempt": True}
        search_limits = bindings["protocol"]["search"]
        if search_limits != {"max_calls": 858, "beam_width": 2, "seconds": 60.}:
            raise ValueError("Original search limits differ from fixed diagnostic")
        if context_record["max_steps"] != 6:
            raise ValueError("Original six-step task changed")
        # Existing preview accounting imports tensor diagnostics; no model is created.
        import torch
        from resectionlab.spatial_policy import SpatialPolicy
        def forbid(owner, name, counter):
            original = getattr(owner, name); originals.append((owner, name, original))
            def refused(*args, **kwargs):
                result[counter] += 1
                raise RuntimeError("Forbidden search-only operation: "+counter)
            setattr(owner, name, refused)
        forbid(torch.nn.Module, "_call_impl", "policy_forward_calls")
        forbid(SpatialPolicy, "__init__", "policy_constructions")
        forbid(torch.optim.Optimizer, "__init__", "optimizer_construction_attempts")
        forbid(torch, "load", "checkpoint_loads")
        from resectionlab.native_resection import NativeResectionEngine
        from resectionlab.patient_planning_admission import make_patient_planning_task
        from resectionlab.planning_budget import PlanningBudget
        from resectionlab.observed_search import observed_beam_search
        from resectionlab.core import array_digest, semantic_digest
        from resectionlab.native_proposals import NominalCavityProposalConfig, DEFAULT_COLUMN_OFFSETS
        from resectionlab.native_spatial_task import NativeSpatialTask
        from public_patient_factory import prepare_public_source
        original_observation = NativeSpatialTask.observation
        originals.append((NativeSpatialTask, "observation", original_observation))
        def capture_prepared_inventory(task):
            observed = original_observation(task)
            # The ordinary observation already prepared and validated these rows.
            # Reading them adds no preview, branch, geometry or metric call.
            key = (task._engine.state_hash, task._steps, task._terminated)
            if key not in observed_inventories:
                ledger = list(task._ledger)
                counts = {}
                for row in ledger:
                    family = row["family"]
                    bucket = counts.setdefault(family, {"declared": 0, "emitted": 0,
                        "feasible": 0, "cap_omitted": 0, "dispositions": {}})
                    bucket["declared"] += 1
                    bucket["emitted"] += row["proposal_reason"] == "PROPOSED_UNCERTIFIED"
                    bucket["feasible"] += bool(row["feasible"])
                    bucket["cap_omitted"] += row["proposal_reason"] == "CANDIDATE_CAP"
                    reason = row["reason"] if row["proposal_reason"] == "PROPOSED_UNCERTIFIED" else row["proposal_reason"]
                    bucket["dispositions"][reason] = bucket["dispositions"].get(reason, 0)+1
                observed_inventories[key] = {"state_hash": key[0], "steps": task._steps,
                    "terminated": task._terminated, "observation_hash": observed.fingerprint,
                    "family_counts": counts, "ledger": ledger}
            return observed
        NativeSpatialTask.observation = capture_prepared_inventory
        budget = PlanningBudget(NativeResectionEngine, max_native_previews=2000, seconds=110.)
        with budget:
            progress("reconstruct_exact_original_public_source")
            baseline_output = output / "baseline-source"; baseline_output.mkdir(exist_ok=False)
            source, binding, qc, protocol = prepare_public_source(baseline_output, prior_release,
                index["files"]["original_release"]["sha256"], bindings["protocol"], progress)
            if binding != bindings["source_binding"] or qc != bindings["qc"]:
                raise ValueError("Original public/QC bindings differ after reconstruction")
            baseline_source_hash = source.source_hash
            original_tools = [asdict(tool) for tool in source.tools]
            original_access = (source.access.center_mm.tolist(), source.access.normal_inward.tolist(),
                               source.access.radius_mm, source.access.window_id)
            # Reconstruct the identical saved footprint proposal source from the
            # original public factory; only search retention will change.
            source = replace(source, proposal_config=NominalCavityProposalConfig(intermediate_opening_mm=1., tool_footprint_opening=True))
            if (source.proposal_config.offsets_source_voxels != DEFAULT_COLUMN_OFFSETS
                    or source.proposal_config.max_candidates != 96
                    or source.proposal_config.intermediate_opening_mm != 1.
                    or source.proposal_config.tool_footprint_opening is not True):
                raise ValueError("Fixed original footprint proposal configuration required")
            for field, key in (("structural_intensity", "image_array_hash"),
                               ("observed_support", "support_array_hash"),
                               ("nominal_target", "target_array_hash"),
                               ("affine_ras_mm", "affine_array_hash")):
                if array_digest(getattr(source, field)) != binding[key]:
                    raise ValueError("Footprint proposal changed a public array/frame")
            if ([asdict(tool) for tool in source.tools] != original_tools
                    or (source.access.center_mm.tolist(), source.access.normal_inward.tolist(),
                        source.access.radius_mm, source.access.window_id) != original_access
                    or source.source_hash == baseline_source_hash):
                raise ValueError("Tool/access changed or new proposal identity failed to bind")
            binding = {**binding, "source_hash": source.source_hash}
            qc = {**qc, "public_source_binding_hash": semantic_digest(binding),
                  "evidence_record_sha256": args.release_sha256}
            protocol = {**protocol, "public_source_binding_hash": semantic_digest(binding),
                "qc_receipt_hash": semantic_digest(qc), "runtime_release_sha256": args.release_sha256,
                "max_native_previews": 2000, "worker_seconds": 110, "max_optimizer_updates": 0,
                # Admission requires a positive numerical ceiling; this owned
                # search-only worker forbids every forward and constructs none.
                "max_policy_forwards": 1,
                "learning_protocol_hash": semantic_digest({"version": "opening-depth-volume-search-only-v1",
                    "optimizer_updates": 0, "policy_forwards": 0,
                    "proposal_rule_hash": source.proposal_config.fingerprint,
                    "retention_policy": release["retention_policy"]})}
            write("admitted-public-bindings.json", {"source_binding": binding, "qc": qc,
                "protocol": protocol, "baseline_source_hash": baseline_source_hash,
                "unchanged_public_arrays_frame_access_tools": True,
                "requested_lookahead_mm": 1., "tool_footprint_opening": True,
                "proposal_rule_hash": source.proposal_config.fingerprint,
                "endpoint_rule": "numpy_rint nearest positive source-cell count; minimum1; actual advance may exceed request",
                "model_initialization_performed": False,
                "execution_forward_limit": 0})
            progress("before_task_initial_inventory")
            task, context = make_patient_planning_task(source,
                cohort_bytes=(ROOT / "manifests/experiments/remind-component-cohort-v1.json").read_bytes(),
                source_binding=binding, qc_receipt=qc, protocol=protocol)
            initial_hash = task.observation().fingerprint
            if (context.record()["source_hash"] != source.source_hash or task.max_steps != 6
                    or context.record()["role"] != "TRAIN" or context.record()["subject"] != "ReMIND-008"
                    or asdict(task.reward_spec) != read("baseline_reward")):
                raise ValueError("Changed source admission or baseline reward")
            inventory = task.candidate_inventory()
            write("initial-inventory.json", inventory)
            old_inventory = read("prior_contrast_inventory")
            identity_checks = {"canonical_json_inventory": inventory_identity(inventory, old_inventory)}
            for name, actual in (("source_hash", source.source_hash),
                    ("decision_model_hash", task.decision_model_hash),
                    ("initial_observation_hash", initial_hash)):
                previous = prior_contrast[name]
                identity_checks[name] = {"equal": actual == previous, "actual": actual, "previous": previous,
                    "requirement": "identical footprint task, public observation and candidates"}
            write("identity-checks.json", identity_checks)
            mismatches = {key: row for key, row in identity_checks.items() if not row["equal"]}
            if mismatches:
                result["identity_mismatches"] = mismatches
                raise ValueError("Retention contrast identity mismatch: "+json.dumps(mismatches, sort_keys=True))
            write("context.json", context.record())
            progress("same_six_step_beam_search", max_calls=858, beam_width=2, seconds=60.)
            result["search_invocations"] += 1
            actions, accounting = observed_beam_search(task, **search_limits,
                objective_source="supplied_public_whole_tumor_and_frozen_geometric_costs",
                transition_mode="lazy_planning", retained_prefix_diagnostics=True,
                retention_mode=release["retention_policy"])
            write("search-prefixes.json", {"actions": actions, "accounting": accounting})
            write("retained-progress.json", {"new": retained_progress(accounting),
                "saved_footprint_baseline": retained_progress(baseline["accounting"]),
                "volume_meaning": "cumulative actual positive-depth actor-covered removed-cell volume; may include shallow/off-axis clearing, not measured shaft clearance",
                "depth_comparison": "exact float order; volume breaks equal depth only",
                "depth_meaning": "deepest newly removed public actor-covered cell center from the access plane, cumulative along retained prefix; not tip insertion or a fully cleared corridor",
                "missing_terminal_depth": "null is unavailable, not zero; no new replay or geometry to fill it"})
            if accounting["call_cap_reached"] or accounting["time_cap_reached"] or accounting["completed_layers"] != 6:
                raise RuntimeError("Diagnostic search did not complete all six fixed layers")
            parity = {
                "same_stop_decision": (list(actions) == ["STOP"]) == (baseline["actions"] == ["STOP"]),
                "baseline_actions": baseline["actions"], "new_actions": list(actions),
                "baseline_transition_calls": baseline["accounting"]["model_transition_calls"],
                "new_transition_calls": accounting["model_transition_calls"],
                "baseline_search_seconds": baseline["accounting"]["planning_seconds"],
                "new_search_seconds": accounting["planning_seconds"],
                "comparison_scope": "only equal-depth cumulative committed-volume tie preference changes; same source/model/observation/candidates, arrays, frame, access, tools, reward, horizon and beam",
                "source_bound_action_ids_expected_equal": True,
                "exact_initial_inventory_preserved": True,
                "depth_comparison": "exact float order; no tolerance",
                "candidate_cap": 96,
                "omitted_candidate_legality_unknown": True,
                "same_sampling_budgets_not_a_runtime_speed_claim": True}
            write("baseline-comparison.json", parity)
            if task.observation().fingerprint != initial_hash:
                raise ValueError("Search mutated the authoritative initial state")
            context.require_task(task)
            # This completion flag concerns the finished search accounting;
            # no episode is executed, replayed, sealed, or admitted for evaluation.
            budget.complete(history_complete=True)
        result.update(status="complete_opening_volume_retention_search_contrast", actions=list(actions),
            source_hash=source.source_hash, decision_model_hash=task.decision_model_hash,
            initial_observation_hash=initial_hash, original_release_sha256=index["files"]["original_release"]["sha256"],
            input_index_sha256=index_record["sha256"], baseline_comparison=parity,
            baseline_source_hash=baseline_source_hash,
            retention_policy=release["retention_policy"],
            same_public_world_original_candidates_preserved=True,
            tool_footprint_opening=True, proposal_rule_hash=source.proposal_config.fingerprint,
            observed_inventory_states=len(observed_inventories),
            initial_added_family_dispositions=next(iter(observed_inventories.values()))["family_counts"].get("tool_footprint_opening", {}),
            actual_opening_advance_mm=sorted({row["actual_opening_advance_mm"] for row in inventory["ledger"] if row["family"] == "intermediate_opening"}),
            initial_extra_candidate_dispositions={reason: sum(row["proposal_reason"] == reason for row in inventory["ledger"] if row["family"] == "intermediate_opening") for reason in sorted({row["proposal_reason"] for row in inventory["ledger"] if row["family"] == "intermediate_opening"})},
            model_transition_calls=accounting["model_transition_calls"], completed_layers=accounting["completed_layers"],
            search_seconds=accounting["planning_seconds"], policy_guidance=False,
            cost_scope="new retention-only search plus public reconstruction and initial inventory; saved footprint baseline costs remain separate",
            limitation="larger committed opening volume is an aggressive retention proxy, not shaft clearance or target utility; unchanged96 cap leaves omitted legality unknown; full-world search and64-cubed actor features have unequal effective inputs; retained removal depth differs from tip insertion",
            private_evaluation_performed=False, diagnostic_only_not_new_training_or_benchmark=True)
    except BaseException as error:
        result.update(status="failed_or_unresolved", failure={"type": type(error).__name__, "message": str(error)})
        if getattr(error, "accounting", None) is not None:
            write("partial-accounting.json", error.accounting)
        if getattr(error, "best_sequence", None) is not None:
            write("partial-incumbent-not-a-teacher.json", {"actions": error.best_sequence})
        raise
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0.); signal.signal(signal.SIGALRM, previous_handler)
        for owner, name, original in reversed(originals): setattr(owner, name, original)
        write("observed-inventories.json", list(observed_inventories.values()))
        result["worker_wall_seconds"] = time.perf_counter()-started
        result["native_budget"] = None if budget is None else budget.snapshot()
        write("result.json", result)


if __name__ == "__main__": main()
