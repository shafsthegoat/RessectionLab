"""Audit saved identities/ledgers and replay committed generated actions only."""
from pathlib import Path
from dataclasses import replace
import hashlib
import json
import math
import struct
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
BASE = ROOT / "build/matched-estimate-methods-v1"
RUN = BASE / "attempt-01"
OUT = Path(__file__).resolve().parent
sys.path[:0] = [str(BASE / "candidate"), str(ROOT / "src")]
def sha(path): return hashlib.sha256(path.read_bytes()).hexdigest()
def read(path): return json.loads(path.read_text(), parse_constant=lambda s: (_ for _ in ()).throw(ValueError(s)))
def write(path, value): path.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")
def points(rows): return {tuple(row) for row in rows}
def same(a, b): assert math.isclose(a, b, rel_tol=1e-12, abs_tol=1e-12), (a, b)

started = time.perf_counter()
declaration_sha = "24c9f9bc22ebf56b1567cd27f3c8777a55897ccfbb58606a5f4ae205d771dd60"
assert sha(RUN / "declaration-input.json") == declaration_sha
declaration = read(RUN / "declaration-input.json")
index = read(RUN / "output-sha256.json")
all_files = {str(p.relative_to(RUN)): p for p in RUN.rglob("*") if p.is_file()}
omitted_nested_inventories = {
    "source-and-metadata-snapshot/artifacts/native-opening-bc-capacity-v1/output-sha256.json",
    "source-and-metadata-snapshot/artifacts/native-opening-rl-capacity-v1/output-sha256.json",
}
assert set(all_files) - set(index) == omitted_nested_inventories | {"output-sha256.json"}
files = {name: path for name, path in all_files.items() if name in index}
assert set(files) == set(index)
assert all(sha(files[name]) == value for name, value in index.items())
for group in ("source_sha256", "metadata_sha256"):
    for name, value in declaration[group].items():
        assert sha(RUN / "source-and-metadata-snapshot" / name) == value
        assert sha(ROOT / name) == value
assert sha(BASE / "prospective-fixed-experiment.json") == declaration["proposal_sha256"]
checkpoint_identities = {}
for method, row in declaration["proposal"]["frozen_checkpoints"].items():
    path = ROOT / row["path"]
    prior = read(path.parent / "output-sha256.json")
    assert sha(path) == row["file_sha256"] == prior[path.name]
    assert row["declaration_sha256"] == prior["declaration-input.json"]
    checkpoint_identities[method] = {"path": row["path"], "file_sha256": sha(path),
        "decoded": False, "prior_declaration_sha256": row["declaration_sha256"]}

import torch
def forbidden(*args, **kwargs): raise AssertionError("Checkpoint loading, actor forwarding, and new search are forbidden in saved audit")
torch.load = forbidden
torch.nn.Module._call_impl = forbidden
import matched_estimate_methods as harness
import run_fixed_comparison as runner
from resectionlab.core import semantic_digest
from resectionlab.native_resection import NativeResectionEngine
from resectionlab.spatial_policy_diagnostics import NativePreviewProfiler
harness.observed_beam_search = forbidden
harness._rollout_actor = forbidden
runner.load_checkpoint = forbidden

supervisor = read(RUN / "supervisor.json")
result = read(RUN / "result.json")
assert supervisor["status"] == result["status"] == "complete"
assert supervisor["returncode"] == 0 and supervisor["worker_termination_confirmed"] is True
assert supervisor["unresolved_worker_pid"] is None and not supervisor["cleanup_errors"]
assert supervisor["termination_reason"] is None and supervisor["automatic_retry"] is False
assert supervisor["declaration_sha256"] == declaration_sha
assert supervisor["seconds"] < declaration["settings"]["max_wall_seconds"]
assert supervisor["sampled_peak_rss_bytes"] <= declaration["settings"]["max_rss_bytes"]
assert result["optimizer_updates"] == result["patient_files_opened"] == 0
assert (RUN / "worker.log").stat().st_size == 0

suites = {}
rows = []
microsteps = actions = 0
target = {(4, 4, 4), (4, 4, 5)}
support = {(4, 4, z) for z in range(1, 6)} | {(5, 5, 1)}
for condition in declaration["proposal"]["conditions_in_fixed_order"]:
    name = condition["name"]
    suite = read(RUN / name / "planning-suite.json")
    suites[name] = suite
    payload = dict(suite); claimed = payload.pop("seal_hash")
    assert semantic_digest(payload) == claimed == result["conditions"][name]["planning_suite_seal_hash"]
    assert suite["budget"] == declaration["method_budget"]
    assert suite["method_order"] == list(harness.METHODS) and set(suite["methods"]) == set(harness.METHODS)
    assert suite["optimizer_updates"] == suite["real_patient_count"] == 0
    assert len({r["strategy"]["plan"]["initial_observation_hash"] for r in suite["methods"].values()}) == 1
    evaluation = read(RUN / (name + "-evaluation.json"))
    assert evaluation == result["conditions"][name]["evaluation"]
    assert evaluation["planning_suite_seal_hash"] == claimed
    for method in harness.METHODS:
        row = suite["methods"][method]
        assert row == read(RUN / name / (method + "-method.json"))
        strategy = row["strategy"]
        assert strategy == read(RUN / name / (method + "-strategy.json"))
        plan = strategy["plan"]
        assert row["status"] == "complete" and row["outcomes"] is None
        assert plan["method"] == method and plan["input_hash"] == suite["input_hash"]
        assert plan["horizon"] == 2 and plan["clinical_use_permitted"] is False
        assert plan["outside_source_fov_tool_feasibility_assessed"] is False
        details, budget = row["details"], row["online_budget"]
        assert budget["failure"] is None and budget["counting_reliable"] is True
        assert budget["native_preview_entries"] <= 2048 and budget["elapsed_seconds"] < 10.
        assert budget["limits"] == {"native_preview_entries": 2048, "planning_execution_seconds": 10.}
        assert budget["blocked_preview_attempts"] == 0
        assert plan["method_accounting"] == {key: details[key] for key in ("actor_forward_calls", "model_transition_calls")}
        assert strategy["recording_accounting"]["nominal_replay_transition_calls"] == len(plan["action_ids"])
        assert row["seal_and_export_replay_transition_calls"] == 2 * len(plan["action_ids"])
        if method != "SEARCH":
            selected = declaration["proposal"]["frozen_checkpoints"]["RL" if method == "HYBRID" else method]
            assert row["parameter_hash"] == selected["parameter_hash"]
            if method == "HYBRID":
                assert details["initial_policy_state_hash"] == details["final_policy_state_hash"] == selected["parameter_hash"]
        if method in ("SEARCH", "HYBRID"):
            assert details["call_cap_reached"] is details["time_cap_reached"] is False
            assert details["beam_pruned_prefixes"] == 0 and details["completed_layers"] == 2
            assert details["max_calls"] == 256 and details["beam_width"] == 16
            assert details["model_transition_calls"] == sum(layer["model_transition_calls"] for layer in details["layers"])
            assert all(layer["completed"] for layer in details["layers"])
        else:
            assert details["actor_forward_calls"] == len(details["decisions"]) == len(plan["action_ids"])
            for decision, action in zip(details["decisions"], plan["action_ids"]):
                values = decision["legal_logits"]
                assert all(value is None or math.isfinite(value) for value in values)
                best = max(range(len(values)), key=lambda i: -math.inf if values[i] is None else values[i])
                assert decision["selected_action"] == action == decision["action_ids"][best]
                moves = [value for value in values[1:] if value is not None]
                if moves:
                    # The saved actor score subtraction occurred in float32.
                    rounded = struct.unpack("f", struct.pack("f", values[0] - max(moves)))[0]
                    same(decision["stop_minus_best_movement"], rounded)
        removed, contacts = set(), set()
        path_length = 0.; changes = nonstops = 0; previous = None
        for physical in strategy["physical_history"]:
            assert physical["action_id"] in plan["action_ids"]
            if physical["action_id"] == "STOP":
                assert not physical.get("microsteps") and not physical.get("removed_indices_native")
                continue
            nonstops += 1
            cells = points(physical["removed_indices_native"])
            contact = points(physical["contact_indices_native"])
            assert not removed & cells and cells <= support and contact <= support
            micro_cells, micro_contacts = set(), set()
            for step in physical["microsteps"]:
                new = points(step["removed_indices_native"])
                assert not micro_cells & new
                micro_cells |= new; micro_contacts |= points(step["contact_indices_native"])
                microsteps += 1
            assert micro_cells == cells and micro_contacts == contact
            removed |= cells; contacts |= contact
            distance = math.dist(physical["entry_mm"], physical["tip_mm"])
            same(physical["complete_tool_path_length_mm"], 2 * distance)
            same(physical["removed_volume_mm3"], len(cells))
            path_length += 2 * distance
            changes += int(previous is not None and previous != physical["tool_id"])
            previous = physical["tool_id"]
        terminal = strategy["terminal_state"]
        assert points(terminal["removed_indices_native"]) == removed
        assert points(terminal["contact_indices_native"]) == contacts
        assert points(terminal["retained_contact_indices_native"]) == contacts - removed
        assert terminal["terminated"] and terminal["steps_taken"] == len(plan["action_ids"])
        measured = evaluation["methods"][method]
        assert measured["evaluation_status"] == "accepted" and measured["independent_evaluation"]["accepted"]
        assert measured["independent_evaluation"]["geometry"]["feasible"]
        assert measured["physical_history_hash"] == plan["physical_history_hash"]
        outcome = measured["outcomes"]
        expected = {"target_removed_mm3": len(removed & target), "normal_removed_mm3": len(removed - target),
            "simulated_removed_volume_mm3": len(removed), "complete_tool_path_length_mm": path_length,
            "tool_changes": changes, "nonstop_actions": nonstops,
            "cumulative_contacted_tissue_upper_bound_mm3": len(contacts),
            "currently_retained_contacted_tissue_upper_bound_mm3": len(contacts - removed)}
        expected["total_reward"] = len(removed & target) - .2 * len(removed - target) - .03 * nonstops - .001 * path_length - .03 * changes
        for key, value in expected.items(): same(outcome[key], value)
        actions += len(plan["action_ids"])
        rows.append({"condition": name, "method": method, "action_ids": plan["action_ids"],
            "seal_hash": plan["seal_hash"], "physical_history_hash": plan["physical_history_hash"],
            "outcomes": expected, "actor_forwards": details["actor_forward_calls"],
            "planner_transitions": details["model_transition_calls"],
            "seal_export_replay_transitions": row["seal_and_export_replay_transition_calls"],
            "online_seconds": budget["elapsed_seconds"], "native_preview_entries": budget["native_preview_entries"],
            "search_complete": details.get("completed_layers") == 2 if method in ("SEARCH", "HYBRID") else None})

long = suites["actual_120mm_tools"]["methods"]
stop = long["RL"]["details"]["decisions"][0]
useful = long["SEARCH"]["strategy"]["plan"]["action_ids"][0]
assert long["RL"]["strategy"]["plan"]["action_ids"] == ["STOP"]
assert useful in stop["action_ids"] and stop["legal_logits"][stop["action_ids"].index(useful)] is not None
assert len(stop["action_ids"]) - 1 == 7
latest_strategy_ns = max((RUN / name / (m + "-strategy.json")).stat().st_mtime_ns for name in suites for m in harness.METHODS)
earliest_evaluation_ns = min((RUN / (name + "-evaluation.json")).stat().st_mtime_ns for name in suites)
assert latest_strategy_ns < earliest_evaluation_ns

# Fresh nominal/geometry replay follows only the fifteen committed transitions;
# search, policy forwarding and checkpoint parsing are disabled above.
replay_started = time.perf_counter()
with NativePreviewProfiler(NativeResectionEngine) as profile:
    baseline = runner.generated_spec()
    for condition in declaration["proposal"]["conditions_in_fixed_order"]:
        name = condition["name"]
        spec = replace(baseline, tools=tuple(replace(tool, working_length_mm=length)
            for tool, length in zip(baseline.tools, condition["working_lengths_mm"])))
        replay = harness.evaluate_matched_estimates(suites[name], spec,
            load_reference=lambda: baseline.target.mask, output=OUT / (name + "-replay.json"))
        saved = read(RUN / (name + "-evaluation.json"))
        for method in harness.METHODS:
            actual, expected = replay["methods"][method], saved["methods"][method]
            assert actual["evaluation_status"] == "accepted"
            assert actual["outcomes"] == expected["outcomes"]
            assert actual["physical_history_hash"] == expected["physical_history_hash"]
            a, e = dict(actual["independent_evaluation"]), dict(expected["independent_evaluation"])
            a.pop("evaluation_seconds"); e.pop("evaluation_seconds")
            assert a == e
    replay_profile = profile.snapshot()
replay_seconds = time.perf_counter() - replay_started
assert all(sha(files[name]) == value for name, value in index.items())
report = {"status": "PASS_saved_identity_arithmetic_and_committed_generated_replay",
    "declaration_sha256": declaration_sha, "output_index_sha256": sha(RUN / "output-sha256.json"),
    "original_indexed_files": len(files), "original_indexed_bytes": sum(p.stat().st_size for p in files.values()),
    "finalized_total_files": len(all_files), "finalized_total_bytes": sum(p.stat().st_size for p in all_files.values()),
    "output_index_coverage": {"excluded_root_index": "output-sha256.json",
        "excluded_nested_metadata_inventories": sorted(omitted_nested_inventories),
        "nested_inventories_validated_by_declaration_metadata_hashes": True,
        "note": "The original runner excludes every basename output-sha256.json. Its output index covers 108 files, not every finalized file."},
    "source_count": len(declaration["source_sha256"]), "metadata_count": len(declaration["metadata_sha256"]),
    "checkpoint_file_identities": checkpoint_identities, "checkpoint_decodes_by_reviewer": 0,
    "model_forwards_by_reviewer": 0, "new_search_calls_by_reviewer": 0,
    "saved_attempt_supervisor": supervisor, "saved_worker_elapsed_seconds": result["elapsed_seconds"],
    "reviewed_strategy_count": len(rows), "recorded_transition_count": actions, "recorded_microstep_count": microsteps,
    "rows": rows, "totals": {key: sum(row[key] for row in rows) for key in
        ("actor_forwards", "planner_transitions", "seal_export_replay_transitions", "native_preview_entries", "online_seconds")},
    "useful_action_available_to_stopping_RL": {"action_id": useful, "legal_nonstop_count": 7,
        "stop_minus_best_movement": stop["stop_minus_best_movement"], "search_return": 1.163,
        "same_observation_hash": long["RL"]["strategy"]["plan"]["initial_observation_hash"]},
    "chronology_evidence": {"reviewed_runner_finishes_both_suites_before_any_reference_callback": True,
        "all_saved_strategy_mtimes_precede_first_saved_evaluation_mtime": True,
        "limit": "source order and filesystem evidence, not an authenticated external event trace"},
    "additional_generated_replay": {"seconds": replay_seconds, "native_preview_profile": replay_profile,
        "committed_nominal_transitions": actions, "committed_evaluation_transitions": actions,
        "all_eight_outcomes_geometry_and_physical_hashes_match": True},
    "review_elapsed_seconds": time.perf_counter() - started,
    "claim_limit": "Full-information tiny generated tool shift only; no patient transfer, tissue mechanics, clinical validation or demonstrated deployment efficiency; outside-FOV tool feasibility unassessed."}
write(OUT / "audit.json", report)
print(json.dumps({"status": report["status"], "audit_sha256": sha(OUT / "audit.json"),
    "strategies": len(rows), "transitions": actions, "microsteps": microsteps,
    "files": report["finalized_total_files"], "bytes": report["finalized_total_bytes"],
    "totals": report["totals"], "replay_seconds": replay_seconds}, sort_keys=True))
