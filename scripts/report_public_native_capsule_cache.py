#!/usr/bin/env python3
"""Read a released, completed fixed-trace cache probe; stdlib only.

No simulation, policy, geometry checker or source experiment is imported.
Timing summaries are descriptive within this one paired case, not a benchmark
of learning throughput. The independent artifact audit remains separate.
"""
from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import math
from pathlib import Path

MODES = ("reference_before", "cached_cold", "cached_warm", "reference_after")
COUNTERS = ("calls", "hits", "misses", "evictions", "bypasses", "query_seconds", "compute_seconds")


def digest(data):
    return hashlib.sha256(data).hexdigest()


def content_hash(value):
    return "sha256:" + digest(json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode())


def require(condition, message):
    if not condition:
        raise ValueError(message)


class Inputs:
    def __init__(self, root):
        self.root, self.index = Path(root), {}

    def raw(self, name):
        path = self.root / name
        compressed = path.with_suffix(path.suffix + ".gz")
        if path.is_file():
            raw = path.read_bytes()
            if compressed.is_file():
                require(gzip.decompress(compressed.read_bytes()) == raw, "Raw/gzip mismatch: " + name)
        else:
            raw = gzip.decompress(compressed.read_bytes())
        self.index[name] = {"sha256": digest(raw), "bytes": len(raw)}
        return raw

    def read(self, name):
        return json.loads(self.raw(name))


def _seconds(value, name):
    require(isinstance(value, (int, float)) and not isinstance(value, bool)
            and math.isfinite(value) and value >= 0, "Invalid timing: " + name)
    return value


def file_digest(path):
    hasher = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            hasher.update(block)
    return hasher.hexdigest()


def verify_baseline_source(baseline):
    require(file_digest(baseline["source_archive_path"]) == baseline["source_archive_sha256"], "Source archive changed")
    manifest_raw = Path(baseline["source_manifest_path"]).read_bytes()
    require(digest(manifest_raw) == baseline["source_manifest_sha256"], "Source manifest changed")
    manifest = json.loads(manifest_raw)
    require(manifest["revision"] == baseline["source_commit"], "Archive revision differs")
    files = manifest["tracked_file_sha256"]
    require(len(files) == baseline["source_file_count"], "Archive file count differs")
    source = Path(baseline["immutable_source_root"])
    for name, expected in files.items():
        require(file_digest(source / name) == expected, "Archive source file changed: " + name)
    require(file_digest(baseline["case_bundle_path"]) == baseline["case_bundle_sha256"], "Copied case changed")
    require(file_digest(baseline["runtime_manifest_path"]) == baseline["runtime_manifest_sha256"], "Runtime receipt changed")
    actual = {str(path.relative_to(source)) for path in source.rglob("*") if path.is_file()}
    added = sorted(actual - set(files) - {baseline["case_bundle_relative_path"]})
    require(not added, "Source snapshot has unexpected added files")
    return {"tracked_files_verified": len(files), "added_files": added,
            "source_archive_sha256": baseline["source_archive_sha256"],
            "source_manifest_sha256": baseline["source_manifest_sha256"],
            "case_bundle_sha256": baseline["case_bundle_sha256"], "status": "byte_identical"}


def extract(root, baseline_path, audit_path=None):
    source = Inputs(root)
    launcher = source.read("launcher-status.json")
    require(launcher.get("status") == "completed", "Read completed attempts only; no growing or failed timing report")
    worker, result = source.read("worker-status.json"), source.read("result.json")
    baseline_raw = Path(baseline_path).read_bytes()
    baseline = json.loads(baseline_raw)
    archived_source = verify_baseline_source(baseline)
    require(worker.get("status") == result.get("status") == "completed", "Completion authorities disagree")
    require(launcher.get("worker_returncode") == 0 and not launcher.get("parent_timeout_requested")
            and not launcher.get("hard_killed"), "Launcher process did not complete normally")
    require(worker.get("result_hash") == content_hash(result), "Bound result hash differs")
    snapshot = source.read("launch-source.json")
    require(snapshot["runtime_content_hash"] == content_hash({"files": snapshot["file_sha256"],
        "versions": snapshot["runtime_versions"], "python": snapshot["python"]}), "Runtime manifest hash differs")
    require(worker["runtime_content_hash"] == result["runtime_content_hash"] == snapshot["runtime_content_hash"]
            == baseline["runtime_content_hash"], "Runtime identities differ")
    for name, expected in snapshot["file_sha256"].items():
        require(digest(source.raw("frozen-source/" + name)) == expected, "Frozen source differs: " + name)
    declaration = source.read("declaration.json")
    require(declaration["declaration_content_hash"] == content_hash({key: value for key, value in declaration.items()
            if key != "declaration_content_hash"}) == baseline["declaration_content_hash"], "Declaration identity differs")
    require(result["scientific_model_hash"] == declaration["v2_model"]["decision_model_hash"], "Scientific model differs")
    require(result["gradient_updates"] == 0 and not result["final_worlds_used"] and not result["stress_worlds_used"],
            "Fixed-trace scope differs")
    require([row["mode"] for row in result["phases"]] == list(MODES), "Phase order differs")
    phases = []
    trace_hashes, history_hashes, rewards = set(), set(), set()
    for row in result["phases"]:
        mode = row["mode"]
        require(source.read(mode + "/receipt.json") == row, "Phase/result receipt differs")
        timing = source.read(mode + "/timing.json")
        require(timing == row["timing"], "Nested phase timing differs")
        require(row["actions"] == declaration["v2_model"]["fixed_actions"] and row["exact_scientific_equality"],
                "Fixed action or recorded equality differs")
        trace_bytes = source.raw(mode + "/scientific-trace.json.gz")
        require(digest(trace_bytes) == row["scientific_trace_export"]["compressed_sha256"], "Compressed trace differs")
        trace = gzip.decompress(trace_bytes)
        require(digest(trace) == row["scientific_trace_export"]["canonical_sha256"], "Canonical trace differs")
        require("sha256:" + digest(trace) == row["scientific_trace_hash"], "Scientific trace hash differs")
        trace_hashes.add(row["scientific_trace_hash"])
        history_hashes.add(row["history_hash"])
        rewards.add(row["total_reward"])
        reset = _seconds(timing["reset_seconds"], mode + " reset")
        transitions = [_seconds(item["transition_and_next_inventory_seconds"], mode + " transition")
                       for item in timing["transitions"]]
        before_export = _seconds(timing["seconds_before_export"], mode + " before export")
        residual = before_export - reset - sum(transitions)
        require(residual >= -1e-6, "Phase timing scopes are inconsistent")
        accounting = timing["adapter_proposal_accounting"]
        require(accounting["preview_calls"] == declaration["fixed_work"]["preview_attempts_per_episode"],
                "Preview denominator differs")
        before, after = row["cache_before"], row["cache_after"]
        if after is not None:
            require(after["entries"] <= declaration["cache"]["max_entries"]
                    and after["retained_payload_bytes"] <= declaration["cache"]["max_payload_bytes"], "Cache limits exceeded")
        phases.append({"mode": mode, "reset_seconds": reset, "transition_seconds": transitions,
            "transition_total_seconds": sum(transitions), "phase_before_export_seconds": before_export,
            "capture_accounting_and_phase_overhead_seconds": max(0., residual),
            "certificate_capture_seconds": None,
            "export_seconds": _seconds(row["scientific_trace_export_seconds"], mode + " export"),
            "phase_including_comparison_and_export_seconds": _seconds(
                row["seconds_including_scientific_comparison_and_export"], mode + " whole phase"),
            "complete_proposer_verification": timing["full_proposal_verification"],
            "adapter_proposal_accounting": accounting,
            "cache_before": before, "cache_after": after,
            "cache_delta": None if before is None else {key: after[key] - before[key] for key in COUNTERS},
            "process_cumulative_peak_rss_bytes": row["process_cumulative_peak_rss_bytes"]})
    require(len(trace_hashes) == len(history_hashes) == len(rewards) == 1, "Phase scientific identities differ")
    require(next(iter(history_hashes)) == declaration["v2_model"]["greedy_history_hash"], "Historical V2 history differs")
    require(phases[0]["cache_before"] is phases[0]["cache_after"] is None, "First reference used cache")
    require(phases[1]["cache_before"]["calls"] == phases[1]["cache_before"]["entries"] == 0, "Cold cache was populated")
    require(phases[2]["cache_before"] == phases[1]["cache_after"], "Warm cache did not inherit cold state")
    require(phases[3]["cache_before"] == phases[3]["cache_after"] == phases[2]["cache_after"], "Final reference changed cache")
    audits = result["independent_audits"]
    require(set(audits) == set(MODES), "Four audits required")
    for mode, audit in audits.items():
        require(source.read(mode + "-independent-audit.json") == audit, "Independent audit receipt differs")
        require(all(audit["receipt"].get(key) is True for key in
                    ("feasible", "complete_tool_checked", "frontier_checked")), "Independent audit failed")
    refs = [phases[index]["phase_including_comparison_and_export_seconds"] for index in (0, 3)]
    comparisons = {row["mode"]: {"reference_min_over_phase": min(refs) / row["phase_including_comparison_and_export_seconds"],
                                "reference_max_over_phase": max(refs) / row["phase_including_comparison_and_export_seconds"]}
                   for row in phases[1:3] if row["phase_including_comparison_and_export_seconds"] > 0}
    summary = {"schema_version": 1, "status": "completed_recorded_probe_report",
        "runtime_content_hash": result["runtime_content_hash"], "scientific_model_hash": result["scientific_model_hash"],
        "baseline": {"path": str(Path(baseline_path).resolve()), "sha256": digest(baseline_raw)},
        "archived_source_verification": archived_source,
        "preparation": source.read("preparation.json"), "template": source.read("template.json"),
        "cache_construction": source.read("cache-construction.json"), "phases": phases,
        "independent_audits": audits, "worker_resource": source.read("worker-resource.json"), "launcher": launcher,
        "reference_bracket_seconds": refs, "descriptive_reference_over_cached_ratios": comparisons,
        "scientific_trace_hash": next(iter(trace_hashes)), "history_hash": next(iter(history_hashes)),
        "total_reward": next(iter(rewards)), "new_simulator_calls": 0, "new_audit_replays": 0,
        "new_gradient_updates": 0, "clinical_deficit_probability": None,
        "timing_scope": "Nested proposer, adapter preview/integrity and cache query/compute times overlap; do not sum. Certificate capture is unisolated within phase residual, alongside accounting, guards and progress writes.",
        "restoration_evidence": "Inference from completed source-bound runner checks, not a separately persisted or independently observed callable-restoration receipt.",
        "interpretation": "Descriptive paired fixed trace on one development case. Reference bracket is order/OS variation, not a confidence interval; no whole-learning throughput or clinical claim."}
    if audit_path is not None:
        data = Path(audit_path).read_bytes()
        summary["separate_artifact_audit"] = {"path": str(Path(audit_path).resolve()), "sha256": digest(data), "record": json.loads(data)}
    return summary, source.index


def markdown(summary):
    cached = summary["phases"][1:3]
    references = summary["reference_bracket_seconds"]
    zero_hits = all(row["cache_delta"]["hits"] == 0 for row in cached)
    slower = all(row["phase_including_comparison_and_export_seconds"] > max(references) for row in cached)
    outcome = ("Both cached phases had zero hits and took longer than both reference phases. "
               "This declared capacity did not improve the fixed trace." if zero_hits and slower else
               "The measured phases and complete cache counters are reported below; the scope remains this one fixed trace.")
    lines = ["# Fixed-trace native cache probe", "", outcome, "", summary["interpretation"], "",
        "| Phase | Reset (s) | Three transitions (s) | Other phase work (s) | Export (s) | Whole phase (s) |",
        "|---|---:|---:|---:|---:|---:|"]
    for row in summary["phases"]:
        lines.append(f"| {row['mode']} | {row['reset_seconds']:.3f} | {row['transition_total_seconds']:.3f} | "
            f"{row['capture_accounting_and_phase_overhead_seconds']:.3f} | {row['export_seconds']:.3f} | "
            f"{row['phase_including_comparison_and_export_seconds']:.3f} |")
    lines += ["", summary["timing_scope"], "",
        "| Phase | Complete proposer calls / s | Native previews / s | Cache calls | Hits | Misses | Evictions |",
        "|---|---:|---:|---:|---:|---:|---:|"]
    for row in summary["phases"]:
        proposer, preview, delta = row["complete_proposer_verification"], row["adapter_proposal_accounting"], row["cache_delta"]
        counts = ["—"] * 4 if delta is None else [str(delta[key]) for key in ("calls", "hits", "misses", "evictions")]
        lines.append(f"| {row['mode']} | {proposer['calls']} / {proposer['seconds']:.3f} | "
            f"{preview['preview_calls']} / {preview['preview_seconds']:.3f} | " + " | ".join(counts) + " |")
    final_cache = cached[-1]["cache_after"]
    lines += ["", f"Warm replay ended with {final_cache['entries']:,} entries and {final_cache['retained_payload_bytes']:,} "
        f"array-buffer bytes, within its {final_cache['max_entries']:,}-entry / {final_cache['max_payload_bytes']:,}-byte limits. "
        "Keys and bookkeeping are outside that payload counter. Eviction plus zero warm hits is consistent with the replay evicting covers before reuse; this is an interpretation of the recorded counters, not an access-distance trace.",
        "", f"Source preparation took {summary['preparation']['seconds']:.3f} s; the common uncached template took "
        f"{summary['template']['factory_seconds']:.3f} s; empty-cache construction took {summary['cache_construction']['seconds']:.3f} s. "
        f"The four passing native audits took {sum(row['seconds'] for row in summary['independent_audits'].values()):.3f} s in total. "
        f"Worker wall time was {summary['worker_resource']['elapsed_seconds']:.3f} s and launcher wall time was "
        f"{summary['launcher']['full_launcher_seconds']:.3f} s.",
        "", f"Process peak RSS was {summary['worker_resource']['observed_peak_rss_bytes']:,} bytes. "
        "Cumulative RSS includes all phase snapshots, histories and other process allocations; it does not isolate cache memory.",
        "", summary["restoration_evidence"], "", "This reader performs only artifact checks and arithmetic; it runs no simulation, geometry replay, policy or gradient update.", ""]
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--audit", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    require(not args.output.exists(), "Report output already exists; preserve previous reports")
    summary, index = extract(args.input, args.baseline, args.audit)
    args.output.mkdir(parents=True, exist_ok=False)
    for name, value in (("summary.json", summary), ("input-index.json", index)):
        (args.output / name).write_text(json.dumps(value, sort_keys=True, indent=2, allow_nan=False) + "\n")
    (args.output / "RESULT.md").write_text(markdown(summary))


if __name__ == "__main__":
    main()
