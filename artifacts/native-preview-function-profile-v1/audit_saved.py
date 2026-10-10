"""Saved JSON/source audit only. Never imports scientific code or opens arrays."""
import copy
import hashlib
import json
from pathlib import Path
import subprocess

ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent
BASE = "build/native-preview-function-profile-v1"
HEAD = "290bdccfe3350b89343876f899b4c3798bc40b38"
inputs = {}
checks = []

def digest(raw):
    return hashlib.sha256(raw).hexdigest()

def read(name, expected=None):
    path = ROOT / name
    assert path.suffix in (".json", ".py", ".md"), name
    raw = path.read_bytes()
    actual = digest(raw)
    if expected is not None:
        assert actual == expected, name
    inputs[name] = actual
    return raw

def load(name, expected=None):
    return json.loads(read(name, expected))

def require(condition, label):
    assert condition, label
    checks.append(label)

release = load(BASE + "/root-release.json", "5a4212cf619c75ad7037e73dada9c46e114eb935eb08840fa4d07d4344252691")
index = load(BASE + "/source-index.json", "c2a3d7e04e603329507bd243848c815e800a485ba425d06f1d50020c51b7ca02")
require(release["expected_head"] == index["head"] == HEAD, "executed HEAD bound")
source = {}
for name, expected in index["source_files"].items():
    if name.startswith("src/"):
        raw = subprocess.check_output(["git", "show", HEAD + ":" + name], cwd=ROOT)
        require(digest(raw) == expected, "executed committed source: " + name)
        inputs[HEAD + ":" + name] = expected
    else:
        raw = read(name, expected)
    source[name] = raw.decode()
for name, expected in index["metadata_files"].items():
    read(name, expected)

result = load(BASE + "/attempt-01/result.json", "0fc15a9ee576ccf9ac86a2d0cb1400c0d48a12af4317fd200e12d76c78c22ee7")
receipt = load(BASE + "/attempt-01.supervision/receipt.json")
profile = load(BASE + "/attempt-01/native-functions.json")
plan = load(BASE + "/attempt-01/greedy-plan.json")
metrics = load(BASE + "/attempt-01/episode-metrics.json")
prior_plan = load(release["inputs"]["plan"]["path"], release["inputs"]["plan"]["sha256"])
prior_metrics = load(release["inputs"]["metrics"]["path"], release["inputs"]["metrics"]["sha256"])
require(receipt["result_sha256"] == inputs[BASE + "/attempt-01/result.json"] and
        receipt["release_sha256"] == result["release_sha256"] == inputs[BASE + "/root-release.json"], "receipt joins")
require(receipt["exit_code"] == 1 and receipt["worker_termination_confirmed"] and
        not receipt["cleanup_errors"] and not receipt["remaining_owned_pids"], "failed attempt cleanly reaped")
require(result["failure"] == {"type": "ValueError", "message": "exact_full_history"}, "failure is history representation guard")
require(profile["call_returned"] and plan["accounting"]["complete"], "profiled greedy call completed")
require(plan["actions"] == prior_plan["actions"], "complete actions equal original")
def without_time(accounting):
    result = dict(accounting)
    result.pop("planning_seconds")
    return result
require(without_time(plan["accounting"]) == without_time(prior_plan["accounting"]), "all non-time planning accounting equal")
require(metrics["history"] == prior_metrics["history"], "complete serialized histories exactly equal")
require(metrics == prior_metrics, "complete serialized episode metrics exactly equal")
require(metrics["terminated"] and metrics["steps"] == result["committed_actions"] == len(plan["actions"]) == 13,
        "12 motions plus STOP committed")
require(result["native_previews"] == 2275 and result["native_budget"]["counting_reliable"] and
        result["native_budget"]["blocked_preview_attempts"] == 0, "preview accounting retained")
require(all(result[key] == 0 for key in ("model_calls", "optimizer_calls", "checkpoint_loads", "blocked_external_calls")), "no model or external calls")
require(not (ROOT / BASE / "attempt-01/independent-episode.json").exists(), "no new independent geometry audit")

worker = source[BASE + "/initial_inventory_worker.py"]
native = source["src/resectionlab/native_resection.py"]
task = source["src/resectionlab/native_spatial_task.py"]
require("metrics['history']==prior_metrics['history']" in worker, "worker compares in-memory to deserialized history")
require("feasible, reason, tool_id, tuple(tip), tuple(axis), tuple(entry)," in native and
        '"tip_mm": self.tip_mm' in native, "native result constructs and retains tuple tip")
require('return {**geometry, "reward": float(reward)' in task and
        "self._history.append(copy.deepcopy(record))" in task and
        '"history": copy.deepcopy(self._history)' in task, "task history preserves tuple values")
witness = copy.deepcopy(metrics["history"])
witness[0]["tip_mm"] = tuple(witness[0]["tip_mm"])
require(witness != prior_metrics["history"] and json.loads(json.dumps(witness)) == prior_metrics["history"],
        "stdlib witness: tuple/list inequality vanishes only at JSON normalization")

by_name = {row["function"]: row for row in profile["functions"]}
capsule = by_name["capsule_voxel_indices"]
native_edge = next(row for row in capsule["callers"] if row["caller"]["function"] == "preview_stroke")
top = [{key: row[key] for key in ("function", "total_calls", "self_seconds", "cumulative_seconds")}
       for row in sorted(profile["functions"], key=lambda x: x["self_seconds"], reverse=True)[:8]]
cache_path = "src/resectionlab/experimental_capsule_cache.py"
cache_sha = digest(read(cache_path))
require(cache_sha == "7b47cf36493c4baeb305beb7f18d4f5acf5bd0fa08b0e80738e948a20bd58869", "existing independently reviewed cache source unchanged")

audit = {
    "status": "completed_profile_window_with_overall_guard_failure",
    "scope": "saved JSON and source only; no numerical imports, arrays, native execution, or new geometry audit",
    "executed_head": HEAD, "checks_passed": len(checks), "checks": checks,
    "parent": {key: receipt[key] for key in ("exit_code", "elapsed_seconds", "sampled_peak_rss_bytes", "worker_termination_confirmed")},
    "profile": {key: profile[key] for key in ("call_returned", "profiled_wall_seconds", "total_calls", "sum_self_seconds", "function_count")},
    "top_self": top,
    "cache_reachable_caller": native_edge,
    "cache_reachable_fraction_of_profile_wall": native_edge["counts_and_times"][3] / profile["profiled_wall_seconds"],
    "native_preview_fraction_of_profile_wall": by_name["preview_stroke"]["cumulative_seconds"] / profile["profiled_wall_seconds"],
    "phases": result["native_budget"]["native_preview_profile"]["phases"],
    "outcome": {key: metrics[key] for key in ("steps", "terminated", "total_reward", "target_removed_mm3", "normal_removed_mm3")},
    "guard_cause": "NativeStrokeResult constructs tuple(tip); to_history_record, _score_record, deepcopy history and metrics retain it. prior_metrics comes from JSON with lists. Direct equality therefore fails despite identical full serialized metrics. A one-field stdlib witness confirms the representation mismatch; no numerical reconstruction was performed.",
    "future_guard_repair": "Normalize the complete history with the existing canonical JSON representation before equality. Preserve attempt-01 and its failed status; no rerun is needed to interpret the completed profile.",
    "limitations": ["Inclusive timings overlap and must not be added.", "cProfile instrumentation changes wall time; this is not a speed benchmark.", "No new independent geometry audit completed.", "Argument repetition, cache hit rate, guard cost and sustained RL throughput are unmeasured."],
    "input_sha256": inputs,
}
proposal = {
    "status": "source_only_proposal_not_executed",
    "optimization": "Reuse existing exact capsule-cover LRU for one native greedy window",
    "implementation": {"path": cache_path, "sha256": cache_sha},
    "reason": "45,354 native preview capsule queries consumed 10.020807126 cumulative seconds. Exact pure geometry reuse avoids recomputing arbitrary oblique segment/cell covers without changing arithmetic or relying on axis alignment.",
    "boundaries": {"max_retained_array_bytes": 33554432, "max_entries": 16384, "workers": 1, "threads": 1,
        "lifetime": "one fresh cache after task construction, scoped only around one greedy call; restore native query before authoritative execution and independent audit; release retained cache after writing stats",
        "payload_note": "entry keys/bookkeeping are outside array-payload cap; existing owned RSS/wall/output limits still apply"},
    "unchanged_checks": ["hard and access envelope", "unknown source domain and post-exposure physical segment", "remaining occupancy and prior-cavity shaft clearance", "full eight-corner active-tip containment", "connectivity/frontier", "preview budget", "state/certificate validation", "reward and complete plan replay"],
    "key": "existing exact float64 start/end/radius bits plus validated immutable frame/source namespace; no rounding; no cavity/feasibility/integrity caching",
    "not_intercepted": "geometry._check_envelope direct coverage calls (2.602034839 s) and its direct distance calls (2.840041546 s); do not add these inclusive times to totals",
    "why_before_arithmetic_change": "Existing source and synthetic adversaries already cover exact keys, immutable result tampering, eviction/empty arrays, restoration and prior-cavity rejection. Algebra changes would require new boundary-roundoff equivalence proof; this profile does not establish an axis-aligned query share.",
    "next_evidence_if_scheduled": "Reuse the existing owned one-case runner, same accepted TRAIN045 world and complete actions/accounting/history. Record cache construction, hit/miss/eviction/bypass and query/compute time, payload/entries, greedy wall and process RSS separately; restore before full authoritative replay and current independent audit. First check existing generated cache/post-exposure semantics against then-current code. No timing gain is claimed from the present profile.",
    "missing": ["TRAIN045 query reuse/hit rate", "current per-query guard overhead", "cache memory including Python objects", "full replay parity with post-exposure condition under cache", "uninstrumented end-to-end benefit", "any RL throughput benefit"],
    "activation": "none; no canonical source edits or experiment performed",
}
for name, value in (("audit-result.json", audit), ("optimization-proposal.json", proposal)):
    with (OUT / name).open("x") as stream:
        json.dump(value, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")
    print(name, digest((OUT / name).read_bytes()))
