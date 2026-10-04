#!/usr/bin/env python3
"""Independent saved-query result audit; standard library, no native execution."""
from __future__ import annotations

import argparse
from array import array
from bisect import bisect_left
from collections import Counter
import gzip
import hashlib
import heapq
import json
import math
from pathlib import Path
import sys
import time

DECLARATION_HASH = "sha256:ff027ab3c2cea8b791841c89945a0839817144b80b0b5f8aaf9e6807e4f83af8"
PRODUCER_SHA256 = "3d4659cadeb33027b2d889bef068591aaa6b10491037e0795224e8d352e4c2ff"
DECLARATION_PATH = Path("manifests/experiments/native-cache-query-keys-v1.json")


def require(condition, message):
    if not condition:
        raise ValueError(message)


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()


def content_hash(value):
    return "sha256:" + hashlib.sha256(canonical(value)).hexdigest()


def sha(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def read_json(path, *, max_bytes=64 * 1024**2):
    path = Path(path)
    opener = gzip.open if path.suffix == ".gz" else Path.open
    with opener(path, "rb") as stream:
        raw = stream.read(max_bytes + 1)
    require(len(raw) <= max_bytes, "JSON byte cap exceeded")
    return json.loads(raw)


def doubles(values):
    require(all(type(value) in (int, float) and math.isfinite(value) for value in values),
            "Nonfinite or nonnumeric float64 evidence")
    result = array("d", values)
    require(result.itemsize == 8, "Platform does not provide binary64 arrays")
    if sys.byteorder != "little":
        result.byteswap()
    return result.tobytes()


def subtract_axis(point, axis, distance):
    # Two separately rounded binary64 operations, matching the pinned callsite.
    products = [float(distance) * float(coordinate) for coordinate in axis]
    return [float(coordinate) - product for coordinate, product in zip(point, products)]


def verify_arguments(trace, tools, metadata, projected):
    """Verify output in place against saved source; never invoke its producer."""
    required_binding = ("namespace", "cache_version", "native_config_hash", "source_hash",
                        "geometry_source_sha256", "source_guard_sha256", "geometry_version",
                        "native_version", "mode", "geometry_epsilon")
    binding = {key: metadata[key] for key in required_binding}
    binding.update(version="saved-native-exact-query-arguments-v1", float64_byte_order="little",
                   omitted_invariant_cache_frame_fields=["inverse_affine_bytes", "cell_radius_mm", "orthogonal_spacing_bytes"])
    tool_by_id = {tool["tool_id"]: tool for tool in tools}
    require(len(tool_by_id) == len(tools) > 0, "Duplicate/missing tool identity")
    rows = projected["queries"]
    cursor = 0
    counts, accepted, microsteps = [], [], []
    frame = None
    for inventory_index, snapshot in enumerate(trace["snapshots"]):
        inventory, action_ids = snapshot["inventory"], snapshot["action_ids"]
        require(inventory["status"] == "complete" and action_ids[0] == "STOP", "Incomplete inventory")
        require(len(action_ids) == len(set(action_ids)), "Duplicate inventory action")
        require(inventory["certified_action_ids"] == action_ids[1:], "Certified action order differs")
        require(set(snapshot["certificates"]) == set(action_ids[1:]), "Certificate membership differs")
        require(len(inventory["attempts"]) == len(action_ids) - 1, "Attempt denominator differs")
        beginning = cursor
        step_count = 0
        for action_id, attempt in zip(action_ids[1:], inventory["attempts"]):
            require(attempt["status"] == "complete" and attempt["feasible"] is True, "Incomplete/rejected preview")
            cert = json.loads(snapshot["certificates"][action_id])
            require(cert["tool_id"] == attempt["tool_id"], "Attempt tool differs")
            require(doubles(cert["tip_mm"]) == doubles(attempt["tip_mm"]), "Attempt target differs")
            require(doubles(cert["entry_mm"]) == doubles(attempt["entry_mm"]), "Attempt entry differs")
            require(cert["source_state_hash"] == snapshot["cavity_state_hash"], "Cavity identity differs")
            require(cert["source_hash"] == binding["source_hash"], "Source identity differs")
            require(cert["decision_model_hash"] == binding["native_config_hash"], "Native model identity differs")
            descriptor = {"source_shape": cert["source_shape"],
                          "native_affine_float64_le_hex": doubles([v for line in cert["native_affine"] for v in line]).hex()}
            if frame is None:
                frame = descriptor
                binding.update(frame)
            require(frame == descriptor, "Frame changed within the query stream")
            tool = tool_by_id[cert["tool_id"]]
            axis = cert["axis_unit"]
            previous = cert["entry_mm"]
            require(bool(cert["microsteps"]), "Missing completed microsteps")
            for microstep_index, step in enumerate(cert["microsteps"]):
                require(doubles(step["tip_start_mm"]) == doubles(previous), "Microstep start differs")
                end = step["tip_end_mm"]
                active_start = subtract_axis(previous, axis, tool["tip_length_mm"])
                require(doubles(step["active_stroke_start_mm"]) == doubles(active_start), "Active start formula differs")
                require(doubles(step["active_stroke_end_mm"]) == doubles(end), "Active end differs")
                require(doubles([step["active_radius_mm"]]) == doubles([tool["tip_radius_mm"]]), "Active radius differs")
                calls = [("active", step["active_stroke_start_mm"], end, tool["tip_radius_mm"]),
                         ("shaft", subtract_axis(previous, axis, tool["working_length_mm"]),
                          subtract_axis(end, axis, tool["tip_length_mm"]), tool["shaft_radius_mm"])]
                for kind, start, finish, radius in calls:
                    require(cursor < len(rows), "Output omitted a native query")
                    row = rows[cursor]
                    require(row["index"] == cursor and row["inventory_index"] == inventory_index
                            and row["action_id"] == action_id and row["tool_id"] == cert["tool_id"]
                            and row["microstep_index"] == microstep_index and row["kind"] == kind,
                            "Output query order/identity differs")
                    require(row["argument_float64_le_hex"] == doubles([*start, *finish, radius]).hex(),
                            "Output query argument bytes differ")
                    cursor += 1
                step_count += 1
                previous = end
            require(doubles(previous) == doubles(cert["tip_mm"]), "Microsteps missed the target")
        counts.append(cursor - beginning)
        accepted.append(len(action_ids) - 1)
        microsteps.append(step_count)
    require(frame is not None and cursor == len(rows), "Extraneous or absent query evidence")
    require(binding == projected["binding"], "Projection binding differs from source")
    binding_hash = content_hash(binding)
    require(binding_hash == projected["binding_hash"], "Projection binding hash differs")
    require(counts == projected["inventory_query_counts"], "Inventory query denominator differs")
    for row in rows:
        require(row["key_projection_hash"] == content_hash({"binding": binding_hash,
                "arguments": row["argument_float64_le_hex"]}), "Projected key hash differs")
    return {"queries": cursor, "accepted_attempts_by_inventory": accepted,
            "microsteps_by_inventory": microsteps, "queries_by_inventory": counts,
            "exact_argument_byte_mismatches": 0, "ordering_mismatches": 0,
            "binding_hash": binding_hash}


def independent_distances(keys):
    """Ordered last-access positions; distinct from the producer's Fenwick tree."""
    positions, previous, answer = [], {}, []
    for current, key in enumerate(keys):
        if key not in previous:
            answer.append(None)
        else:
            slot = bisect_left(positions, previous[key])
            require(positions[slot] == previous[key], "Internal last-access invariant")
            answer.append(len(positions) - slot - 1)
            positions.pop(slot)
        previous[key] = current
        positions.append(current)
    return answer


def independent_lru(keys, capacity):
    """Recency timestamps plus a lazy min-heap; no OrderedDict implementation."""
    require(type(capacity) is int and capacity >= 0, "Invalid LRU entry capacity")
    active, queue = {}, []
    hits = misses = evictions = 0
    for index, key in enumerate(keys):
        if key in active:
            hits += 1
        else:
            misses += 1
            if not capacity:
                continue
            if len(active) == capacity:
                while queue:
                    timestamp, old = heapq.heappop(queue)
                    if active.get(old) == timestamp:
                        del active[old]
                        evictions += 1
                        break
        active[key] = index
        heapq.heappush(queue, (index, key))
    return {"hits": hits, "misses": misses, "evictions": evictions, "entries": len(active)}


def verify_statistics(projected, summary, capacities):
    keys = [bytes.fromhex(row["argument_float64_le_hex"]) for row in projected["queries"]]
    require(bool(keys), "Missing query statistics")
    n = len(keys)
    frequency = Counter(keys)
    distances = independent_distances(keys + keys)
    phases = []
    for name, ds in (("first", distances[:n]), ("repeated", distances[n:])):
        finite = [d for d in ds if d is not None]
        phases.append({"phase": name, "calls": len(ds), "compulsory_misses": len(ds) - len(finite),
            "distinct_key_reuse_distance_histogram": {str(k): v for k, v in sorted(Counter(finite).items())},
            "maximum_distinct_key_reuse_distance": max(finite) if finite else None})
    expected = {"queries_per_phase": n, "distinct_argument_keys": len(frequency),
        "within_first_phase_repeat_count": n - len(frequency),
        "occurrence_frequency_histogram": {str(k): v for k, v in sorted(Counter(frequency.values()).items())},
        "phases": phases, "entry_capacity_for_all_repeat_accesses_to_hit_without_payload_limit": 1 + max(distances[n:])}
    limits = {}
    for capacity in capacities:
        first = independent_lru(keys, capacity)
        both = independent_lru(keys + keys, capacity)
        limits[str(capacity)] = {"first": first, "repeated_increment": {
            key: both[key] - first[key] for key in ("hits", "misses", "evictions")}}
    expected["entry_only_lru_unlimited_payload"] = limits
    for key, value in expected.items():
        require(canonical(summary[key]) == canonical(value), "Summary statistic differs: " + key)
    require(summary["exact_total_cover_payload_bytes"] is None and summary["payload_weighted_reuse_distances"] is None,
            "Unknown payload quantity was manufactured")
    require("no-bypass" in summary["interpretation"] and "not measured speedup" in summary["interpretation"],
            "Missing interpretation limitations")
    return expected


def verify_authority(declaration, status, summary, summary_sha256, query_sha256):
    body = dict(declaration)
    declared = body.pop("declaration_content_hash")
    require(content_hash(body) == declared == DECLARATION_HASH, "Prospective declaration differs")
    require(status["status"] == "completed" and status["summary_sha256"] == summary_sha256,
            "Diagnostic completion authority differs")
    require(summary["query_artifact_sha256"] == query_sha256, "Query artifact hash differs")
    require(summary["declaration_content_hash"] == DECLARATION_HASH
            and summary["script_sha256"] == PRODUCER_SHA256, "Executed declaration/source differs")
    require(summary["queries_per_phase"] == declaration["expected_queries_per_phase"] <= declaration["maximum_queries"],
            "Query cap or expected denominator differs")
    require(type(summary["elapsed_seconds"]) in (int, float)
            and 0 <= summary["elapsed_seconds"] <= declaration["maximum_wall_seconds"], "Reported diagnostic time exceeded cap")
    require(all(type(summary[key]) is int and summary[key] == 0
                for key in ("cover_geometry_calls", "patient_loads", "simulator_steps", "gradient_updates")),
            "Unexpected native/model work declaration")
    require(summary["final_worlds_used"] is False and summary["stress_worlds_used"] is False, "Closed worlds were used")


def verify_execution(root, output):
    folder = Path(root) / "artifacts/validation/native-cache-query-keys-v1"
    release_path, execution_path = folder / "root-release.json", folder / "execution.json"
    release, execution = read_json(release_path), read_json(execution_path)
    require(execution["status"] == "completed" and execution["returncode"] == 0
            and execution["external_timeout"] is False and execution["error_type"] is None,
            "Saved-only execution did not complete")
    require(execution["root_release_sha256"] == sha(release_path), "Execution release binding differs")
    require(release["declaration_content_hash"] == DECLARATION_HASH
            and release["script_sha256"] == PRODUCER_SHA256, "Release source/declaration differs")
    require(release["status"] == "released_before_execution" and release["prior_diagnostic_attempts"] == 0,
            "Release/attempt authority differs")
    require(all(release[key] is False for key in ("cover_geometry_authorized", "patient_simulation_authorized",
                                                 "policy_or_gradient_authorized", "patient_bundle_copied")),
            "Unexpected released workload")
    require(execution["source_changed"] == [] and execution["source_added"] == [], "Executed source changed")
    for kind in ("archive", "manifest"):
        require(execution[f"source_{kind}_sha256"] == release[f"source_{kind}_sha256"]
                == sha(release[f"source_{kind}_path"]), "Source archive/manifest hash differs")
    manifest = read_json(release["source_manifest_path"])
    require(manifest["source_commit"] == release["source_commit"] == execution["source_commit"], "Source commit differs")
    require(manifest["file_sha256"] == release["source_files"], "Source manifest differs")
    source_root = Path(release["source_root"])
    present = {str(path.relative_to(source_root)) for path in source_root.rglob("*") if path.is_file()}
    require(present == set(manifest["file_sha256"]), "Archived checkout gained or lost files")
    require(len(present) == execution["verified_source_file_count"] == release["source_file_count"] == 16,
            "Minimal source-file denominator differs")
    require(all(sha(source_root / name) == value for name, value in manifest["file_sha256"].items()),
            "Archived source bytes changed")
    require(all(sha(Path(output) / name) == value for name, value in execution["output_file_sha256"].items()),
            "Executed output bytes changed")
    require(0 <= execution["external_wall_seconds"] < release["limits"]["external_process_timeout_seconds"],
            "External wall-time cap was exceeded")
    return {"source_commit": execution["source_commit"], "verified_source_file_count": len(present),
            "source_archive_sha256": execution["source_archive_sha256"],
            "external_wall_seconds": execution["external_wall_seconds"],
            "process_peak_rss_bytes": None,
            "rss_scope": "Cooperative limit checked by completed producer; no numerical peak-RSS measurement was persisted.",
            "release_sha256": sha(release_path), "execution_sha256": sha(execution_path)}


def audit(root, output, *, producer_source=None):
    root, output = Path(root), Path(output)
    status = read_json(output / "status.json")
    require(status.get("status") == "completed", "Diagnostic is incomplete or failed")
    declaration = read_json(output / "declaration.json")
    require(declaration == read_json(root / DECLARATION_PATH), "Output declaration differs from committed declaration")
    summary = read_json(output / "summary.json")
    query_path = output / "ordered-query-arguments.json.gz"
    verify_authority(declaration, status, summary, sha(output / "summary.json"), sha(query_path))
    producer_source = Path(producer_source) if producer_source else root / "scripts/reconstruct_native_cache_queries.py"
    require(sha(producer_source) == PRODUCER_SHA256, "Producer source bytes differ")
    input_hashes = {name: sha(root / name) for name in declaration["input_sha256"]}
    require(input_hashes == declaration["input_sha256"], "A saved source input changed")
    run = root / declaration["input_run"]
    source = read_json(run / "launch-source.json")
    worker = read_json(run / "worker-status.json")
    launcher = read_json(run / "launcher-status.json")
    result = read_json(run / "result.json")
    runtime = declaration["source_runtime_content_hash"]
    require(all(item["runtime_content_hash"] == runtime for item in (source, worker, result)), "Source runtime differs")
    require(all(source["file_sha256"][name] == value for name, value in declaration["frozen_native_source_sha256"].items()),
            "Frozen callsite source differs")
    require(launcher["status"] == worker["status"] == result["status"] == "completed", "Source experiment is incomplete")
    require(launcher["worker_returncode"] == 0 and launcher["parent_timeout_requested"] is False
            and launcher["hard_killed"] is False, "Source launcher failed or timed out")
    require(worker["result_hash"] == content_hash(result), "Source result hash differs")
    cold, warm = (read_json(run / mode / "receipt.json") for mode in ("cached_cold", "cached_warm"))
    for phase in (cold, warm):
        require(phase["cache_after"]["calls"] - phase["cache_before"]["calls"] == summary["queries_per_phase"],
                "Measured source query count differs")
        require(phase["cache_after"]["bypasses"] - phase["cache_before"]["bypasses"] == 0,
                "No-bypass admission assumption is false")
    tools = read_json(run / "preparation.json")["native_configuration"]["actual"]["components"]["tools"]
    projected = read_json(query_path)
    trace = read_json(run / "reference_before/scientific-trace.json.gz", max_bytes=declaration["maximum_uncompressed_trace_bytes"])
    arguments = verify_arguments(trace, tools, cold["cache_after"], projected)
    require(arguments["queries"] == summary["queries_per_phase"] and arguments["binding_hash"] == summary["binding_hash"],
            "Summary/projected/source argument identities differ")
    require(arguments["queries_by_inventory"] == summary["inventory_query_counts"], "Summary inventory counts differ")
    statistics = verify_statistics(projected, summary, declaration["entry_capacities"])
    execution = verify_execution(root, output)
    for name, value in input_hashes.items():
        require(sha(root / name) == value, "Source changed during independent audit")
    return {"status": "independent_saved_result_audit_passed", "argument_verification": arguments,
        "independently_recomputed_statistics": statistics,
        "execution_verification": execution,
        "reported_diagnostic_elapsed_seconds": summary["elapsed_seconds"],
        "scope": "Saved JSON/byte verification and arithmetic only; no producer invocation, native geometry, patient loading, simulator, policy, gradient or RNG draws.",
        "limits": ["Full inverse/derived frame bytes were not saved; exact equality concerns variable arguments inside one source-bound invariant frame.",
                   "Full cover payload bytes and byte-weighted reuse remain unknown. Entry-only counts assume unchanged no-bypass admission and are not speed measurements or a capacity recommendation.",
                   "The producer checked wall/RSS cooperatively. Its summary records elapsed analysis time, not an independently measured peak-RSS field; external execution receipts must establish any peak memory claim.",
                   "Original native callable restoration and frame-guard execution are source-bound completed-run observations, not rerun here."],
        "declaration_content_hash": DECLARATION_HASH, "producer_sha256": PRODUCER_SHA256,
        "source_runtime_content_hash": runtime, "source_input_sha256": input_hashes,
        "output_sha256": {path.name: sha(path) for path in (output / "status.json", output / "declaration.json", output / "summary.json", query_path)},
        "no_bypass_condition_verified": True, "final_worlds_used": False, "stress_worlds_used": False}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--diagnostic", type=Path, required=True)
    parser.add_argument("--producer-source", type=Path)
    parser.add_argument("--receipt", type=Path, required=True)
    args = parser.parse_args()
    require(not args.receipt.exists(), "Independent receipt already exists")
    started = time.perf_counter()
    try:
        value = audit(args.root, args.diagnostic, producer_source=args.producer_source)
    except Exception as error:
        value = {"status": "failed", "error_type": type(error).__name__, "error": str(error)}
        raise
    finally:
        if "value" in locals():
            value["audit_seconds_not_a_performance_benchmark"] = time.perf_counter() - started
            value["auditor_sha256"] = sha(Path(__file__))
            args.receipt.parent.mkdir(parents=True, exist_ok=True)
            with args.receipt.open("x") as stream:
                json.dump(value, stream, indent=2, sort_keys=True, allow_nan=False)
                stream.write("\n")


if __name__ == "__main__":
    main()
