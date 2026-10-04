#!/usr/bin/env python3
"""Offline pilot report from final receipts; stdlib only, no policy/simulator calls.

Read completed or failed attempts only after the execution owner releases them.
Raw JSON and lossless .json.gz are supported, with exact disagreement refusal.
"""
from __future__ import annotations

import argparse
from collections import Counter
import gzip
import hashlib
import json
import math
from pathlib import Path


def digest(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def content_hash(value) -> str:
    return "sha256:" + digest(json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode())


def require(condition, message):
    if not condition:
        raise ValueError(message)


class Inputs:
    def __init__(self, root: Path):
        self.root = root
        self.hashes = {}

    def exists(self, name: str) -> bool:
        path = self.root / name
        return path.is_file() or path.with_suffix(path.suffix + ".gz").is_file()

    def raw(self, name: str) -> bytes:
        path = self.root / name
        compressed = path.with_suffix(path.suffix + ".gz")
        if path.is_file():
            data = path.read_bytes()
            if compressed.is_file():
                require(gzip.decompress(compressed.read_bytes()) == data,
                        "Raw/gzip evidence disagreement: " + name)
        else:
            data = gzip.decompress(compressed.read_bytes())
        self.hashes[name] = {"uncompressed_sha256": digest(data), "uncompressed_bytes": len(data)}
        return data

    def read(self, name: str):
        return json.loads(self.raw(name))


def _external(path: Path | None):
    if path is None:
        return None
    data = path.read_bytes()
    return {"path": str(path.resolve()), "sha256": digest(data), "record": json.loads(data)}


def extract(root: Path, baseline_path: Path, audit_path: Path | None = None):
    source = Inputs(root)
    launcher = source.read("launcher-status.json")
    require(launcher["status"] in ("completed", "failed"), "Attempt is not final; growing evidence must not be analyzed")
    baseline = _external(baseline_path)
    declaration = source.read("declaration.json")
    require(declaration["declaration_content_hash"] == content_hash({key: value for key, value in declaration.items()
        if key != "declaration_content_hash"}), "Declared protocol content hash differs")
    worker = source.read("worker-status.json") if source.exists("worker-status.json") else None
    pilot = source.read("pilot/status.json") if source.exists("pilot/status.json") else None
    preparation = source.read("preparation.json") if source.exists("preparation.json") else None
    snapshot = source.read("launch-source.json") if source.exists("launch-source.json") else None
    summary = {"schema_version": 1, "pilot_id": declaration["pilot_id"],
        "status": launcher["status"], "source_baseline": baseline,
        "independent_audit": _external(audit_path), "declaration_hash": declaration["declaration_content_hash"],
        "final_worlds_used": False, "stress_worlds_used": False,
        "clinical_deficit_probability": None, "new_simulator_calls": 0, "new_policy_forwards": 0,
        "new_gradient_steps": 0, "interpretation": declaration["interpretation"],
        "launcher": launcher, "worker": worker, "pilot": pilot, "preparation": preparation,
        "frozen_source_verified_files": 0,
        "scope": "Post-completion arithmetic and byte/JSON checks of recorded evidence only; independent geometry and actual-forward reconstruction have separate receipts."}
    if snapshot is not None:
        require(snapshot["runtime_content_hash"] == content_hash({"files": snapshot["file_sha256"],
            "versions": snapshot["runtime_versions"], "python": snapshot["python"]}), "Runtime source manifest hash differs")
        for path, expected in snapshot["file_sha256"].items():
            require(digest((root / "frozen-source" / path).read_bytes()) == expected, "Frozen runtime file changed: " + path)
        summary["frozen_source_verified_files"] = len(snapshot["file_sha256"])
        summary["runtime_content_hash"] = snapshot["runtime_content_hash"]
        summary["runtime_versions"] = snapshot["runtime_versions"]
    if launcher["status"] == "failed":
        require(launcher["eligible_candidate_count"] == 0, "Failed launcher claims eligibility")
        if source.exists("pilot/accounting.json"):
            journal = source.read("pilot/accounting.json")
            summary["retained_accounting"] = {key: journal.get(key) for key in
                ("status", "totals", "episodes", "counts_complete", "failure", "receipt_export_failure", "receipt_hash")}
        if source.exists("pilot/learner/result.json"):
            result = source.read("pilot/learner/result.json")
            summary["last_saved_learner_result"] = {key: result.get(key) for key in
                ("status", "gradient_steps", "completed_episodes", "discarded_partial_batches",
                 "optimization_environment_steps", "selection_environment_steps", "elapsed_seconds")}
            summary["last_saved_result_scope"] = "Last persisted learner metadata may precede a later failure; no completion or comparative outcome is inferred."
        return summary, source.hashes
    require(worker is not None and pilot is not None and worker["status"] == pilot["status"] == "completed",
            "Completed launcher lacks completed worker/pilot authorities")
    require(launcher["worker_returncode"] == 0 and not launcher["parent_timeout_requested"]
            and not launcher["hard_killed"], "Parent process did not complete within its declared lifecycle")
    require(launcher["eligible_candidate_count"] == worker["eligible_candidate_count"] == pilot["eligible_candidate_count"] == 1,
            "Candidate count is not one")
    require(worker["pilot_status_sha256"] == source.hashes["pilot/status.json"]["uncompressed_sha256"], "Worker/pilot authority hash mismatch")
    record = source.read("pilot/candidate-record.json")
    require(worker["candidate_record_sha256"] == pilot["candidate_record_sha256"]
            == source.hashes["pilot/candidate-record.json"]["uncompressed_sha256"], "Candidate payload byte identity differs")
    frozen = source.read("pilot/history-freeze.json")
    audits = source.read("pilot/native-audits.json")
    require(record["completion"] == frozen and record["audits"] == audits, "Frozen/candidate evidence differs")
    result = source.read("pilot/learner/result.json")
    contract = source.read("pilot/learner/contract.json")
    journal = source.read("pilot/accounting.json")
    require(journal["receipt_hash"] == frozen["accounting_receipt_hash"], "Accounting receipt identity differs")
    require(journal["receipt_hash"] == "sha256:" + digest(json.dumps(
        {key: value for key, value in journal.items() if key != "receipt_hash"},
        sort_keys=True, allow_nan=False).encode()), "Accounting journal content hash differs")
    require(journal["status"] == "learner_returned" and journal["counts_complete"]
            and journal["failure"] is None and journal["receipt_export_failure"] is None,
            "Accounting is incomplete")
    for name, expected in frozen["checkpoints"]["files"].items():
        require(digest(source.raw("pilot/learner/" + name)) == expected, "Saved checkpoint/result bytes changed: " + name)
    require(contract["config"] == declaration["training_config"], "Actual training configuration differs")
    require(result["gradient_steps"] == 1 and result["completed_episodes"] == 2
            and result["discarded_partial_batches"] == 0 and result["actor_parameters_changed"] is True,
            "One complete actor update is not established")
    require(len(frozen["episodes"]) == len(audits) == 6 and pilot["completed_episode_audit_receipts"] == 6,
            "Six complete histories and audit receipts required")
    require(not worker["final_worlds_used"] and not pilot["final_worlds_used"]
            and not worker["stress_worlds_used"] and not pilot["stress_worlds_used"], "Unexpected evaluation-world use")
    panels = result["selection_history"]
    require([panel["gradient_steps"] for panel in panels] == [0, 1]
            and all(panel["world_count"] == 2 for panel in panels), "Two complete initial/updated selection panels required")
    require(result["initial_checkpoint_hash"] == declaration["frozen_model"]["initial_policy_hash"], "Initializer differs")
    selected_index = max(range(2), key=lambda index: panels[index]["mean_return"])
    require(selected_index == frozen["selected_panel"] and panels[selected_index]["checkpoint_hash"]
            == result["selected_checkpoint_hash"], "Earliest-best checkpoint choice differs")
    events = journal["events"]
    decisions = [event for event in events if event["kind"] == "decision"]
    transitions = [event for event in events if event["kind"] == "transition"]
    require(len(decisions) == len(transitions), "Decision/transition counts differ")
    require([event["decision_id"] for event in decisions] == [event["decision_id"] for event in transitions],
            "Decision/transition order differs")
    require(len(journal["episodes"]) == 7 and frozen["shape_probe"]["episode"] == 0
            and not any(event.get("episode") == 0 for event in decisions + transitions),
            "Setup probe must remain separate and unscored")
    decision_map = {event["decision_id"]: event for event in decisions}
    episode_rows = []
    for episode, audit in zip(frozen["episodes"], audits):
        identifier = episode["episode"]
        ts = [event for event in transitions if event["episode"] == identifier]
        history = episode["native_history"]
        require([event["info"] for event in ts if event["native_commit"]] == history,
                "Authenticated native history differs from frozen history")
        require(math.isclose(sum(event["reward"] for event in ts), episode["return"], abs_tol=1e-8), "Episode reward differs")
        require(audit["episode"] == identifier and audit["history_hash"] == content_hash(history)
                == episode["history_hash"], "Audit does not cover exact episode history")
        certificate = audit["audit"]
        require(certificate["feasible"] is True and certificate["complete_tool_checked"] is True
                and certificate["frontier_checked"] is True and not certificate["failures"], "Native certificate failed")
        phase = "optimization" if episode["role"] == "optimization" else "initial_selection" if episode["update"] == 0 else "updated_selection"
        ds = [decision_map[row["decision_id"]] for row in episode["decisions"]]
        costs = {key: sum(record[key] for record in history) for key in
                 ("target_removed_mm3", "normal_removed_mm3", "partial_normal_contact_mm3")}
        prefix = ts[-1]["proposal_accounting_after"]
        episode_rows.append({"episode": identifier, "phase": phase, "world_seed": episode["seed"],
            "return": episode["return"], "transitions": len(ts), "native_cuts": len(history),
            "stop_transitions": sum(event["action_id"] == "STOP" for event in ts),
            "termination_reason": ts[-1]["info"]["termination_reason"],
            "actions": episode["actions"], "history_hash": episode["history_hash"],
            "policy_hash": episode["decisions"][0]["policy_hash"],
            "decision_inventory_sizes": [len(event["payload"]["inputs"]["action_ids"]) for event in ds],
            "actual_forward_count": sum(event["payload"]["forward_evaluated"] for event in ds),
            "proposal_accounting_prefix": prefix, **costs})
    counts = {role: {"transitions": sum(row["transitions"] for row in episode_rows if row["phase"] == role),
                    "native_cuts": sum(row["native_cuts"] for row in episode_rows if row["phase"] == role),
                    "episodes": sum(row["phase"] == role for row in episode_rows)}
              for role in ("initial_selection", "optimization", "updated_selection")}
    require(counts["optimization"]["transitions"] == result["optimization_environment_steps"]
            and counts["initial_selection"]["transitions"] + counts["updated_selection"]["transitions"]
            == result["selection_environment_steps"], "Generic/authenticated transition totals differ")
    timing = record["timing"]
    clones = timing["factory_clone_seconds"]
    require(len(clones) == 7, "Seven learner clone attempts required")
    complete_panel_seconds = sum(panel["panel_elapsed_seconds"] for panel in panels)
    training_residual = (timing["full_training_call_seconds"] - result["initialization_seconds"]
                         - result["elapsed_seconds"] - result["final_checkpoint_export_seconds"])
    require(training_residual >= -1e-5, "Nested learner costs exceed measured full call")
    prefix_totals = {name: sum(row["proposal_accounting_prefix"][name] for row in episode_rows)
        for name in ("integrity_calls", "integrity_seconds", "preview_calls", "preview_seconds")}
    unique_audits = {audit["audit_key"]: audit for audit in audits}
    require(len(unique_audits) == pilot["unique_native_audits"], "Unique audit count differs")
    summary.update({"gradient": result["optimization_history"][0], "counts": counts, "episodes": episode_rows,
        "shape_probe": {"receipt": frozen["shape_probe"], "decisions": 0, "transitions": 0,
            "scope": "One setup reset for dimensions; intentionally unscored and excluded from six complete episodes"},
        "selected_panel": selected_index, "initial_selection_return": panels[0]["mean_return"],
        "updated_selection_return": panels[1]["mean_return"],
        "selection_delta": panels[1]["mean_return"] - panels[0]["mean_return"],
        "selected_checkpoint_hash": result["selected_checkpoint_hash"],
        "initial_checkpoint_hash": result["initial_checkpoint_hash"], "latest_checkpoint_hash": result["latest_checkpoint_hash"],
        "initial_actor_hash": result["initial_actor_hash"], "latest_actor_hash": result["latest_actor_hash"],
        "actor_parameters_changed": result["actor_parameters_changed"],
        "decision_trace": {"records": len(decisions), "actual_forwards": sum(d["payload"]["forward_evaluated"] for d in decisions),
            "rule_counts": dict(Counter(d["payload"]["decision_rule"] for d in decisions)),
            "source": "Recorded same-forward detached outputs; report performs no new forward or random draw."},
        "observed_proposal_accounting_prefix_totals": prefix_totals,
        "proposal_accounting_scope": "Sum of the final recorded transition's reset-local counters across six episodes. Subsequent learner model checks/metrics, setup and clone checks may fall outside these prefixes; do not call this a global count or add it to enclosing timing.",
        "costs": {"launcher_seconds": launcher["full_launcher_seconds"],
            "worker_seconds": worker["full_worker_seconds"], "preparation_seconds": preparation["seconds"],
            "cold_axis_construction_seconds": preparation["cold_axis_construction_seconds"],
            "full_training_call_seconds": timing["full_training_call_seconds"],
            "learner_initialization_seconds": result["initialization_seconds"],
            "online_optimization_selection_seconds": result["elapsed_seconds"],
            "online_budget_seconds": declaration["training_config"]["max_wall_seconds"],
            "online_budget_overshoot_seconds": max(0., result["elapsed_seconds"] - declaration["training_config"]["max_wall_seconds"]),
            "initial_selection_panel_seconds": panels[0]["panel_elapsed_seconds"],
            "updated_selection_panel_seconds": panels[1]["panel_elapsed_seconds"],
            "selection_seconds_including_partial_panels": result["selection_seconds"],
            "partial_panel_seconds": result["selection_seconds"] - complete_panel_seconds,
            "optimization_plus_online_export_and_bookkeeping_seconds": result["elapsed_seconds"] - result["selection_seconds"],
            "initial_shape_probe_clone_seconds": clones[0], "six_online_clone_seconds": clones[1:],
            "final_checkpoint_export_seconds": result["final_checkpoint_export_seconds"],
            "trainer_call_residual_seconds": training_residual,
            "trainer_call_residual_scope": "Final result export, accounting finalization and return overhead not separately instrumented; this is arithmetic residual, not a direct timer.",
            "full_independent_validation_seconds": timing["independent_validation_seconds"],
            "unique_native_checker_seconds": sum(audit["seconds"] for audit in unique_audits.values()),
            "unique_native_audits": len(unique_audits), "audit_receipts": len(audits),
            "peak_process_rss_bytes": worker["resource"]["observed_peak_rss_bytes"],
            "cancellation_callback_calls": worker["resource"]["cancellation_callback_calls"],
            "cancellation_callback_seconds": worker["resource"]["cancellation_callback_seconds"],
            "scope": "Nested measured phases are not additive. Logging/accounting persistence was enabled inside online and initialization phases, but its isolated cost was not measured. Factory values are actual seven clones; resets are included in panels/batch and not independently timed."}})
    return summary, source.hashes


def render(summary: dict) -> str:
    if summary["status"] != "completed":
        return "# Native axis RAW update pilot: failed attempt\n\nThe launcher ended failed and no candidate is eligible. The original evidence is retained. `cost-summary.json` records the final worker/pilot failure context and any last durable accounting; partial outcomes are not a completed comparison. No retry or budget change is implied.\n"
    c = summary["costs"]
    prefix = summary["observed_proposal_accounting_prefix_totals"]
    selected = "initial" if summary["selected_panel"] == 0 else "updated"
    after = summary["updated_selection_return"]
    before = summary["initial_selection_return"]
    interpretation = ("Selection retained the initial checkpoint; the completed actor update did not improve the selected result."
        if selected == "initial" else "Selection chose the updated checkpoint on the same two declared development worlds; this one-update pilot does not establish generalization or clinical benefit.")
    rows = ["# Native axis RAW update pilot", "",
        f"The pilot completed one actual actor update from two complete optimization episodes, with complete two-world selection before and after, and six accepted native history certificates. Initial mean return was {before:.2f}; updated mean return was {after:.2f}. {interpretation}", "",
        "| Episode | Phase | Return | Transitions / cuts | Target removed (mm³) | Normal removed (mm³) | Newly charged partial normal contact (mm³) |",
        "|---|---|---:|---:|---:|---:|---:|",
        *[f"| {e['episode']} | {e['phase']} | {e['return']:.2f} | {e['transitions']} / {e['native_cuts']} | {e['target_removed_mm3']:.0f} | {e['normal_removed_mm3']:.0f} | {e['partial_normal_contact_mm3']:.0f} |" for e in summary["episodes"]], "",
        "A separate setup reset supplied dimensions and had zero decisions, transitions or score; it is excluded from the six complete episodes above. No optimization batch or selection panel was discarded. Partial contact is separately charged exposure and does not count as removed tissue. The two selection seeds are deterministic replays, not independent patients or uncertainty samples.", "",
        "| Cost | Seconds | Scope |", "|---|---:|---|",
        f"| Full launcher | {c['launcher_seconds']:.3f} | Source copy, subprocess, worker and parent lifecycle |",
        f"| Worker | {c['worker_seconds']:.3f} | Preparation, training, validation and publication through its measured receipt |",
        f"| Preparation | {c['preparation_seconds']:.3f} | Includes {c['cold_axis_construction_seconds']:.3f}s cold native setup |",
        f"| Full trainer call | {c['full_training_call_seconds']:.3f} | Initialization, online work and final learner/accounting exports |",
        f"| Learner initialization | {c['learner_initialization_seconds']:.3f} | Includes zero-transition shape probe and {c['initial_shape_probe_clone_seconds']:.3f}s clone |",
        f"| Online optimization + selection | {c['online_optimization_selection_seconds']:.3f} | Declared {c['online_budget_seconds']:.0f}s cooperative cap |",
        f"| Initial selection panel | {c['initial_selection_panel_seconds']:.3f} | Both worlds, actual clone/reset/logging costs included |",
        f"| Updated selection panel | {c['updated_selection_panel_seconds']:.3f} | Both worlds, actual clone/reset/logging costs included |",
        f"| Optimization and other online work | {c['optimization_plus_online_export_and_bookkeeping_seconds']:.3f} | Residual includes batch, Adam, intermediate exports and bookkeeping |",
        f"| Final checkpoint export | {c['final_checkpoint_export_seconds']:.3f} | Last atomic tensor write |",
        f"| Full native validation | {c['full_independent_validation_seconds']:.3f} | {c['unique_native_audits']} unique checks covering all six frozen histories, plus guards/receipt exports |", "",
        f"These phases are nested and must not be added. Peak worker RSS was {c['peak_process_rss_bytes']/1024**3:.3f}GiB. The online clock overshoot was {c['online_budget_overshoot_seconds']:.3f}s. Seven actual factory clones were recorded; their individual times are retained in the machine-readable report. Separate reset, decision serialization, journal fsync, and actor/critic compute timings were not instrumented, so their individual bottleneck contributions cannot be inferred.", "",
        f"The six final-transition counter snapshots record {prefix['preview_calls']} native previews taking {prefix['preview_seconds']:.3f}s and {prefix['integrity_calls']} integrity checks taking {prefix['integrity_seconds']:.3f}s. These are observed reset-local prefixes, not whole-process totals: setup, clone and subsequent model/metrics checks can lie outside them. The updated panel was {c['updated_selection_panel_seconds']-c['initial_selection_panel_seconds']:.3f}s slower despite the same selected actions; the available timers do not isolate why. No cache speedup or logging bottleneck is inferred from repetition alone.", "",
        f"The recorded pre-clipping total gradient norm was {summary['gradient']['gradient_norm_before_clip']:.6g}; the post-clipping actor norm was {summary['gradient']['actor_gradient_norm_after_clip']:.6g}. Initial and latest actor hashes differ. The selected {selected} checkpoint remains distinguished from the latest trained checkpoint; post-update decisions are bound to latest weights regardless of selection.", "",
        f"The report joins {summary['decision_trace']['records']} actual decision records to their authenticated outcomes. It verifies {summary['frozen_source_verified_files']} frozen executable input files, original checkpoint byte hashes, final publication authorities, and all six frozen history/certificate identities. It performs no new policy forward, simulator episode, random draw or gradient update. Independent source-cell/forward reconstruction remains documented in its separate audit receipt.", "",
        "This is one previously studied structural mirror-derived development patient. Brain support and hypothetical access remain unreviewed; motor/language injury probabilities and tissue mechanics are not validated. No final or stress world was opened. Other agent computation was paused during the timed run, while ordinary desktop/OS background load and file-cache state were uncontrolled. This pilot supports integration and cost accounting, not an efficacy claim or a new training-budget choice.", "",
        "Reproduce with `report_native_axis_pilot.py --run <attempt> --baseline <execution-baseline.json> --output <fresh-report-directory>`. The reader accepts exact `.json.gz` replacements and refuses raw/compressed disagreement. `report-source.json` records uncompressed source byte hashes and the reporting-script hash.", ""]
    return "\n".join(rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--audit", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    summary, sources = extract(args.run, args.baseline, args.audit)
    args.output.mkdir(parents=True, exist_ok=False)
    (args.output / "cost-summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True, allow_nan=False) + "\n")
    (args.output / "RESULT.md").write_text(render(summary))
    (args.output / "report-source.json").write_text(json.dumps({"input_files": sources,
        "report_script_sha256": digest(Path(__file__).read_bytes()),
        "execution_baseline": _external(args.baseline), "independent_audit": _external(args.audit),
        "new_simulator_calls": 0, "new_policy_forwards": 0, "new_gradient_steps": 0}, indent=2, sort_keys=True) + "\n")


if __name__ == "__main__":
    main()
