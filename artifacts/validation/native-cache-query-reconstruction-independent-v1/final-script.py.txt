#!/usr/bin/env python3
"""Reconstruct exact variable cache arguments from complete saved certificates.

Standard library only. Never loads a patient, imports numerical code, computes
capsule covers, or advances a simulation. --execute requires a separate release.
The constant cache frame is bound by provenance, not invented from saved JSON.
"""
from __future__ import annotations

from collections import Counter, OrderedDict
import argparse
import gzip
import hashlib
import json
import math
from pathlib import Path
import resource
import struct
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
DECLARATION = Path("manifests/experiments/native-cache-query-keys-v1.json")
VERSION = "saved-native-exact-query-arguments-v1"


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def digest(value):
    return "sha256:" + hashlib.sha256(canonical(value).encode()).hexdigest()


def file_hash(path):
    result = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            result.update(block)
    return result.hexdigest()


def number(value):
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError("Query coordinates and tool dimensions must be numeric")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError("Query coordinates and tool dimensions must be finite")
    return result


def vector(value):
    if not isinstance(value, (list, tuple)) or len(value) != 3:
        raise ValueError("Expected a three-vector")
    return tuple(number(item) for item in value)


def float_bytes(value):
    """Original probe ran little-endian float64; preserve signed zero and ULPs."""
    return struct.pack("<" + "d" * len(value), *(number(item) for item in value))


def same_vector(left, right):
    return float_bytes(vector(left)) == float_bytes(vector(right))


def subtract_scaled(point, scale, axis):
    # Match the two distinct float64 operations in the frozen NumPy expression:
    # previous - tool.working_length_mm * axis. Do not fuse or round arguments.
    return tuple(number(point[i]) - number(scale) * number(axis[i]) for i in range(3))


def reconstruct(trace, tools, cache_metadata, *, expected_calls=None, maximum_queries=100000, check_budget=None):
    """Return ordered variable-key bytes; reject incomplete preview evidence.

    The complete original key also contains one invariant namespace and scene
    frame tuple. JSON did not save the inverse/derived frame bytes. Source,
    affine and namespace binding makes this variable projection sufficient for
    equality within this fixed run, but it is not a serialized full cache key.
    """
    if type(maximum_queries) is not int or maximum_queries < 1:
        raise ValueError("maximum_queries must be positive")
    tool_map = {tool["tool_id"]: tool for tool in tools}
    if len(tool_map) != len(tools) or not tool_map:
        raise ValueError("Tool identifiers must be unique")
    required_cache = ("namespace", "cache_version", "native_config_hash", "source_hash", "geometry_source_sha256", "source_guard_sha256",
                      "geometry_version", "native_version", "mode", "geometry_epsilon")
    binding = {name: cache_metadata[name] for name in required_cache}
    binding.update({"version": VERSION, "float64_byte_order": "little",
        "omitted_invariant_cache_frame_fields": ["inverse_affine_bytes", "cell_radius_mm", "orthogonal_spacing_bytes"]})
    frame = None
    queries, inventory_counts = [], []
    for inventory_index, snapshot in enumerate(trace["snapshots"]):
        if check_budget:
            check_budget()
        inventory = snapshot["inventory"]
        action_ids = snapshot["action_ids"]
        attempts = inventory["attempts"]
        if (inventory["status"] != "complete" or not action_ids or action_ids[0] != "STOP"
                or len(set(action_ids)) != len(action_ids)
                or inventory["certified_action_ids"] != action_ids[1:]
                or set(snapshot["certificates"]) != set(action_ids[1:])
                or len(attempts) != len(action_ids) - 1):
            raise ValueError("Complete accepted-preview evidence is required")
        before = len(queries)
        for action_id, attempt in zip(action_ids[1:], attempts):
            if attempt.get("status") != "complete" or attempt.get("feasible") is not True:
                raise ValueError("Rejected or interrupted previews cannot be reconstructed completely")
            certificate = json.loads(snapshot["certificates"][action_id])
            tool = tool_map[certificate["tool_id"]]
            if (certificate["tool_id"] != attempt["tool_id"]
                    or not same_vector(certificate["tip_mm"], attempt["tip_mm"])
                    or not same_vector(certificate["entry_mm"], attempt["entry_mm"])
                    or certificate["source_state_hash"] != snapshot["cavity_state_hash"]
                    or certificate["source_hash"] != binding["source_hash"]
                    or certificate["decision_model_hash"] != binding["native_config_hash"]):
                raise ValueError("Certificate differs from its bound ordered attempt/source")
            shape = certificate["source_shape"]
            affine = certificate["native_affine"]
            if (len(shape) != 3 or any(type(item) is not int or item <= 0 for item in shape)
                    or len(affine) != 4 or any(len(row) != 4 for row in affine)):
                raise ValueError("Invalid source frame descriptor")
            descriptor = {"source_shape": shape,
                          "native_affine_float64_le_hex": float_bytes([item for row in affine for item in row]).hex()}
            if frame is None:
                frame = descriptor
                binding.update(frame)
            elif frame != descriptor:
                raise ValueError("One invariant source frame is required")
            axis = vector(certificate["axis_unit"])
            previous = vector(certificate["entry_mm"])
            steps = certificate["microsteps"]
            if not steps:
                raise ValueError("Accepted certificate lacks its completed microsteps")
            for microstep_index, step in enumerate(steps):
                start, end = vector(step["tip_start_mm"]), vector(step["tip_end_mm"])
                active_start, active_end = vector(step["active_stroke_start_mm"]), vector(step["active_stroke_end_mm"])
                if (not same_vector(start, previous) or not same_vector(active_end, end)
                        or not same_vector(active_start, subtract_scaled(start, tool["tip_length_mm"], axis))
                        or float_bytes([step["active_radius_mm"]]) != float_bytes([tool["tip_radius_mm"]])):
                    raise ValueError("Microstep continuity or frozen active-argument formula changed")
                arguments = (
                    ("active", active_start, active_end, tool["tip_radius_mm"]),
                    ("shaft", subtract_scaled(start, tool["working_length_mm"], axis),
                     subtract_scaled(end, tool["tip_length_mm"], axis), tool["shaft_radius_mm"]),
                )
                for kind, query_start, query_end, radius in arguments:
                    if check_budget and len(queries) % 1024 == 0:
                        check_budget()
                    if number(radius) < 0:
                        raise ValueError("Capsule radius must be nonnegative")
                    if len(queries) >= maximum_queries:
                        raise ValueError("Declared query reconstruction cap exceeded")
                    argument_hex = float_bytes((*query_start, *query_end, radius)).hex()
                    queries.append({"index": len(queries), "inventory_index": inventory_index,
                        "action_id": action_id, "tool_id": certificate["tool_id"],
                        "microstep_index": microstep_index, "kind": kind,
                        "argument_float64_le_hex": argument_hex})
                previous = end
            if not same_vector(previous, certificate["tip_mm"]):
                raise ValueError("Completed microsteps do not reach the certified target")
        inventory_counts.append(len(queries) - before)
    if frame is None:
        raise ValueError("No complete native query evidence")
    if expected_calls is not None and len(queries) != expected_calls:
        raise ValueError("Reconstructed query count differs from measured cache calls")
    binding_hash = digest(binding)
    for query in queries:
        query["key_projection_hash"] = digest({"binding": binding_hash,
                                                "arguments": query["argument_float64_le_hex"]})
    return {"binding": binding, "binding_hash": binding_hash,
            "inventory_query_counts": inventory_counts, "queries": queries}


def reuse_distances(keys):
    """Number of distinct other keys since previous access, or None if unseen."""
    tree = [0] * (len(keys) + 1)
    latest = {}
    def add(index, delta):
        while index < len(tree):
            tree[index] += delta
            index += index & -index
    def prefix(index):
        total = 0
        while index:
            total += tree[index]
            index -= index & -index
        return total
    distances = []
    for index, key in enumerate(keys, 1):
        previous = latest.get(key)
        distances.append(None if previous is None else len(latest) - prefix(previous))
        if previous is not None:
            add(previous, -1)
        latest[key] = index
        add(index, 1)
    return distances


def entry_only_lru(keys, capacity):
    if type(capacity) is not int or capacity < 0:
        raise ValueError("Entry capacity must be a nonnegative integer")
    cache = OrderedDict()
    hits = misses = evictions = 0
    for key in keys:
        if key in cache:
            hits += 1
            cache.move_to_end(key)
        else:
            misses += 1
            if capacity:
                if len(cache) >= capacity:
                    cache.popitem(last=False)
                    evictions += 1
                cache[key] = None
    return {"hits": hits, "misses": misses, "evictions": evictions, "entries": len(cache)}


def summarize(reconstruction, capacities=(0, 1024, 4096, 8192, 16384, 32768), *, check_budget=None):
    keys = [row["argument_float64_le_hex"] for row in reconstruction["queries"]]
    doubled = keys + keys
    distances = reuse_distances(doubled)
    frequency = Counter(keys)
    phases = []
    for name, start, end in (("first", 0, len(keys)), ("repeated", len(keys), len(doubled))):
        finite = [value for value in distances[start:end] if value is not None]
        phases.append({"phase": name, "calls": end - start,
            "compulsory_misses": sum(value is None for value in distances[start:end]),
            "distinct_key_reuse_distance_histogram": dict(sorted(Counter(finite).items())),
            "maximum_distinct_key_reuse_distance": max(finite) if finite else None})
    entry_limits = {}
    for capacity in capacities:
        if check_budget:
            check_budget()
        first, both = entry_only_lru(keys, capacity), entry_only_lru(doubled, capacity)
        entry_limits[str(capacity)] = {"first": first,
            "repeated_increment": {key: both[key] - first[key] for key in ("hits", "misses", "evictions")}}
    return {"queries_per_phase": len(keys), "distinct_argument_keys": len(frequency),
        "within_first_phase_repeat_count": len(keys) - len(frequency),
        "occurrence_frequency_histogram": dict(sorted(Counter(frequency.values()).items())),
        "phases": phases, "entry_only_lru_unlimited_payload": entry_limits,
        "entry_capacity_for_all_repeat_accesses_to_hit_without_payload_limit": 1 + max(distances[len(keys):]),
        "exact_total_cover_payload_bytes": None, "payload_weighted_reuse_distances": None,
        "payload_limitation": "Saved active contacts were filtered by tissue and shaft covers were not saved; neither full cover sizes nor byte-capacity hit rates follow from this trace.",
        "interpretation": "Exact variable-argument reuse for one bound invariant frame. Entry-only hit upper bounds require the same no-bypass admission policy; oversized-query bypass can change cache pollution. Byte-limited predictions remain unknown. This is not measured speedup or a cache-capacity recommendation."}


def write_json(path, value):
    with Path(path).open("x") as stream:
        stream.write(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()
    declaration = json.loads((ROOT / DECLARATION).read_text())
    body = dict(declaration)
    expected = body.pop("declaration_content_hash")
    if digest(body) != expected:
        raise ValueError("Prospective declaration content hash changed")
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    write_json(output / "declaration.json", declaration)
    if not args.execute:
        write_json(output / "status.json", {"status": "declared_not_executed"})
        return
    started = time.perf_counter()
    script_sha256 = file_hash(Path(__file__))
    def check_budget():
        peak = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss) * (1 if sys.platform == "darwin" else 1024)
        if time.perf_counter() - started > declaration["maximum_wall_seconds"]:
            raise InterruptedError("Diagnostic wall budget exceeded")
        if peak > declaration["maximum_process_peak_rss_bytes"]:
            raise InterruptedError("Diagnostic process RSS budget exceeded")
    try:
        check_budget()
        for name, expected in declaration["input_sha256"].items():
            if file_hash(ROOT / name) != expected:
                raise ValueError("Bound input changed: " + name)
        run = ROOT / declaration["input_run"]
        source = json.loads((run / "launch-source.json").read_text())
        expected_runtime = declaration["source_runtime_content_hash"]
        if source.get("runtime_content_hash") != expected_runtime:
            raise ValueError("Source runtime differs from declared runtime")
        if any(source["file_sha256"].get(name) != expected
               for name, expected in declaration["frozen_native_source_sha256"].items()):
            raise ValueError("Frozen native callsite or key implementation changed")
        launcher = json.loads((run / "launcher-status.json").read_text())
        worker = json.loads((run / "worker-status.json").read_text())
        result = json.loads((run / "result.json").read_text())
        if (worker.get("runtime_content_hash") != expected_runtime
                or result.get("runtime_content_hash") != expected_runtime):
            raise ValueError("Worker/result runtime differs from declared runtime")
        if (launcher["status"] != "completed" or worker["status"] != "completed"
                or result["status"] != "completed" or digest(result) != worker["result_hash"]):
            raise ValueError("Source experiment did not complete with matching authority")
        cold = json.loads((run / "cached_cold/receipt.json").read_text())
        warm = json.loads((run / "cached_warm/receipt.json").read_text())
        count = cold["cache_after"]["calls"] - cold["cache_before"]["calls"]
        if (count != declaration["expected_queries_per_phase"]
                or warm["cache_after"]["calls"] - warm["cache_before"]["calls"] != count):
            raise ValueError("Observed source query denominator changed")
        with gzip.open(run / "reference_before/scientific-trace.json.gz", "rb") as stream:
            data = stream.read(declaration["maximum_uncompressed_trace_bytes"] + 1)
        if len(data) > declaration["maximum_uncompressed_trace_bytes"]:
            raise ValueError("Saved trace exceeds the declared decompression limit")
        trace = json.loads(data)
        del data
        check_budget()
        components = json.loads((run / "preparation.json").read_text())["native_configuration"]["actual"]["components"]
        reconstruction = reconstruct(trace, components["tools"], cold["cache_after"],
            expected_calls=count, maximum_queries=declaration["maximum_queries"], check_budget=check_budget)
        summary = summarize(reconstruction, tuple(declaration["entry_capacities"]), check_budget=check_budget)
        # Deliberately retain exact arguments, not only hashes, to avoid relying
        # on hash uniqueness when reviewing distinct-query accounting.
        data = canonical(reconstruction).encode()
        with (output / "ordered-query-arguments.json.gz").open("xb") as raw:
            with gzip.GzipFile(filename="", mode="wb", fileobj=raw, mtime=0) as stream:
                stream.write(data)
        for name, expected in declaration["input_sha256"].items():
            if file_hash(ROOT / name) != expected:
                raise ValueError("Bound input changed during reconstruction: " + name)
        check_budget()
        if file_hash(Path(__file__)) != script_sha256:
            raise ValueError("Diagnostic source changed during reconstruction")
        write_json(output / "summary.json", {**summary, "binding_hash": reconstruction["binding_hash"],
            "inventory_query_counts": reconstruction["inventory_query_counts"],
            "declaration_content_hash": declaration["declaration_content_hash"],
            "script_sha256": script_sha256, "elapsed_seconds": time.perf_counter() - started,
            "query_artifact_sha256": file_hash(output / "ordered-query-arguments.json.gz"),
            "cover_geometry_calls": 0, "patient_loads": 0, "simulator_steps": 0, "gradient_updates": 0,
            "final_worlds_used": False, "stress_worlds_used": False})
        write_json(output / "status.json", {"status": "completed", "summary_sha256": file_hash(output / "summary.json")})
    except Exception as error:
        write_json(output / "status.json", {"status": "failed", "error_type": type(error).__name__, "error": str(error)})
        raise


if __name__ == "__main__":
    main()
