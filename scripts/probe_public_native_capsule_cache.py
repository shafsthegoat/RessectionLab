#!/usr/bin/env python3
"""Prospectively bound V2 fixed-trace cache probe; --execute requires release.

No optimizer or policy is constructed. The default only writes a declaration.
One isolated child keeps the unchanged production engine/adapter and restores
all explicit process-local measurement/cache injection before publication.
"""
from __future__ import annotations

from contextlib import contextmanager, nullcontext
import argparse
import copy
from dataclasses import asdict
import gzip
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
import preflight_native_axis as pre
import numpy as np
from resectionlab import geometry, native_resection as native
from resectionlab.experimental_capsule_cache import ExactCapsuleCoverCache, inject_native_capsule_cache
from resectionlab.native_axis_simulation import AxisColumnNativeSimulator
from resectionlab.native_proposals import AxisColumnProposalConfig
from resectionlab.simulation import RewardSpec
from resectionlab.worlds import _canonical, content_hash

DECLARATION_PATH = Path("manifests/experiments/native-axis-capsule-cache-v1.json")
DECLARATION_HASH = "sha256:90b245b72f84acac1cbe32cb338d64e47222fd75b2f20673439dbb34e6e4f9d6"
V2_PROFILE = Path("artifacts/preflight/native-axis-v2/profile")
MODES = ("reference_before", "cached_cold", "cached_warm", "reference_after")


def declaration(root=ROOT):
    return pre._checked_manifest(root / DECLARATION_PATH, DECLARATION_HASH)


def source_snapshot(root=ROOT):
    base = pre.source_snapshot(root)
    files = dict(base["file_sha256"])
    extra = {str(DECLARATION_PATH), "scripts/probe_public_native_capsule_cache.py",
             "scripts/probe_native_capsule_cache.py", *declaration(root)["bound_input_file_sha256"]}
    for name in sorted(extra):
        files[name] = hashlib.sha256((root / name).read_bytes()).hexdigest()
    base["file_sha256"] = files
    base["runtime_content_hash"] = content_hash({"files": files,
        "versions": base["runtime_versions"], "python": sys.version})
    return base


def _check_bound_inputs(value, root=ROOT):
    for name, expected in value["bound_input_file_sha256"].items():
        if hashlib.sha256((root / name).read_bytes()).hexdigest() != expected:
            raise ValueError(f"Declared V2 input changed: {name}")
    path = root / "src/resectionlab/experimental_capsule_cache.py"
    if hashlib.sha256(path.read_bytes()).hexdigest() != value["cache"]["tested_helper_sha256"]:
        raise ValueError("Cache helper differs from independently tested source")


def plain(value):
    return json.loads(_canonical(value))


def scientific_inventory(receipt):
    result = copy.deepcopy(receipt)
    for attempt in result["attempts"]:
        attempt.pop("elapsed_seconds", None)
    return result


def scientific_metrics(metrics):
    """Only declared timing paths are excluded; new fields remain comparable."""
    result = plain(metrics)
    result.pop("initialization_timing", None)
    for name in ("integrity_seconds", "preview_seconds"):
        result["proposal_accounting"].pop(name, None)
    result["inventory_receipts"] = [scientific_inventory(row) for row in result["inventory_receipts"]]
    for row in result["proposal_failures"]:
        row.pop("elapsed_seconds", None)
    return result


@contextmanager
def measure_proposer_calls(proposer, *, accounting=None):
    """Charge pre-reset verification too; native adapter counters reset midway."""
    if "propose" in proposer.__dict__:
        raise RuntimeError("Proposer already has an instance-level replacement")
    original = proposer.propose
    accounting = {"calls": 0, "seconds": 0.} if accounting is None else accounting
    def measured(engine):
        started = time.perf_counter()
        try:
            return original(engine)
        finally:
            accounting["calls"] += 1
            accounting["seconds"] += time.perf_counter() - started
    proposer.propose = measured
    body_failed = False
    try:
        yield accounting
    except BaseException:
        body_failed = True
        raise
    finally:
        changed = proposer.__dict__.get("propose") is not measured
        proposer.__dict__.pop("propose", None)
        if changed and not body_failed:
            raise RuntimeError("Proposer measurement hook changed; class method restored")


def _snapshot(sim, observation):
    """Read immediately after guarded reset/step returned this observation."""
    sim.assert_model_frozen()
    sim._assert_episode()
    actions = tuple(action.action_id for action in sim._proposals)
    receipt = sim._inventory_receipts[-1]
    if (actions != observation.action_ids or receipt["status"] != "complete"
            or receipt["certified_action_ids"] != list(actions[1:])
            or receipt["batch"]["slot_count"] != len(sim.proposal_config.offsets_source_voxels) * len(sim.native_config.tools)):
        raise RuntimeError("Returned observation and complete certified inventory disagree")
    certificates = {action: _canonical(sim._previews[action].to_history_record()) for action in actions[1:]}
    masks = {name: getattr(sim.engine, name).tobytes() for name in
             ("remaining_mask", "removed_mask", "contact_mask", "connected_free_mask")}
    masks["partial_contact_mask"] = sim._partial_contact_mask.tobytes()
    return {"action_ids": actions, "action_features": observation.action_features.tolist(),
            "state_features": observation.state_features.tolist(), "action_mask": observation.action_mask.tolist(),
            "inventory": scientific_inventory(plain(receipt)), "certificates": certificates,
            "cavity_state_hash": sim.engine.state_hash,
            "mask_digests": {name: "sha256:" + hashlib.sha256(value).hexdigest() for name, value in masks.items()}}, masks


def run_fixed_trace(sim, actions, seed, guard, folder):
    """No fresh action choice; include reset and every complete postcut inventory."""
    folder.mkdir(parents=True, exist_ok=False)
    started = time.perf_counter()
    verification = {"calls": 0, "seconds": 0.}
    try:
        with measure_proposer_calls(sim._proposer, accounting=verification):
            guard.require()
            phase = time.perf_counter()
            observation = sim.reset(seed)
            reset_seconds = time.perf_counter() - phase
            snapshots, masks, transitions = [], [], []
            current, raw_masks = _snapshot(sim, observation)
            snapshots.append(current)
            masks.append(raw_masks)
            for action in actions:
                guard.require()
                if sim.terminated or action not in observation.action_ids:
                    raise RuntimeError("Fixed action trace differs from current V2 inventory")
                phase = time.perf_counter()
                result = sim.step(action)
                elapsed = time.perf_counter() - phase
                observation = result.observation
                current, raw_masks = _snapshot(sim, observation)
                snapshots.append(current)
                masks.append(raw_masks)
                transitions.append({"action_id": action, "reward": result.reward,
                    "terminated": result.terminated, "transition_and_next_inventory_seconds": elapsed})
                pre.write_json(folder / "progress.json", {"status": "running", "transitions": transitions})
            if not sim.terminated or len(sim._history) != len(actions):
                raise RuntimeError("Fixed trace must finish its complete declared episode")
            metrics = sim.metrics()
            trace = {"snapshots": snapshots, "metrics": scientific_metrics(metrics),
                "transitions": [{key: value for key, value in row.items()
                                 if key != "transition_and_next_inventory_seconds"} for row in transitions]}
            timing = {"reset_seconds": reset_seconds, "transitions": transitions,
                "full_proposal_verification": dict(verification),
                "adapter_proposal_accounting": metrics["proposal_accounting"],
                "seconds_before_export": time.perf_counter() - started,
                "process_cumulative_peak_rss_bytes": pre.peak_rss_bytes()}
            pre.write_json(folder / "timing.json", timing)
            guard.require()
            return trace, masks, timing
    except Exception as error:
        pre.write_json(folder / "failure.json", {"status": "incomplete", "error": str(error),
            "error_type": type(error).__name__, "raw_state": pre.raw_failure_state(sim),
            "committed_transition_after_interruption": bool(getattr(error, "committed", False)),
            "resource": guard.receipt(), "full_proposal_verification": verification})
        raise


def _write_gzip(path, value):
    data = _canonical(value).encode()
    with path.open("wb") as raw:
        with gzip.GzipFile(filename="", mode="wb", fileobj=raw, mtime=0, compresslevel=6) as stream:
            stream.write(data)
    return {"canonical_bytes": len(data), "canonical_sha256": hashlib.sha256(data).hexdigest(),
            "compressed_bytes": path.stat().st_size, "compressed_sha256": hashlib.sha256(path.read_bytes()).hexdigest()}


def _public_worker(output, case_bundle, value, guard):
    source = source_snapshot()
    if source["runtime_content_hash"] != json.loads((output / "launch-source.json").read_text())["runtime_content_hash"]:
        raise RuntimeError("Frozen worker differs from declared launch source")
    _check_bound_inputs(value)
    reference = pre._checked_manifest(ROOT / pre.REFERENCE_PATH, pre.REFERENCE_HASH)
    v2 = pre._checked_manifest(ROOT / pre.DECLARATION_PATH, pre.DECLARATION_HASH)
    model = json.loads((ROOT / V2_PROFILE / "model.json").read_text())
    historical = json.loads(gzip.decompress((ROOT / V2_PROFILE / "greedy/episode.json.gz").read_bytes()))
    target, expected = reference["target"], value["v2_model"]
    started = time.perf_counter()
    from resectionlab.imaging import load_case
    if hashlib.sha256(case_bundle.read_bytes()).hexdigest() != target["bundle_sha256"]:
        raise ValueError("Patient bundle differs from declared V2 source")
    case = load_case(case_bundle)
    if case.semantic_hash != target["semantic_hash"] or case.planning_hash != target["planning_hash"] or case.frame != "RAS+":
        raise ValueError("Public case identity or physical frame changed")
    cfg = native.native_config_from_case(case, access=geometry.AccessWindow(**target["access"]))
    config_receipt = pre.assert_declared_native_configuration(cfg, v2, reference)
    if cfg.fingerprint != expected["native_config_hash"]:
        raise ValueError("Public native configuration differs from the V2 action model")
    optimization = pre._partition(target["world_partitions"]["optimization"])
    if expected["optimization_seed"] != optimization.seeds[0] or optimization.role.value != "optimization":
        raise ValueError("Fixed replay seed is not the declared optimization world")
    pre.write_json(output / "preparation.json", {"seconds": time.perf_counter() - started,
        "native_configuration": config_receipt, "optimization": optimization.to_dict(),
        "case_hash": case.semantic_hash, "planning_hash": case.planning_hash})
    guard.require()
    phase = time.perf_counter()
    sim = AxisColumnNativeSimulator(cfg, proposal_config=AxisColumnProposalConfig(), max_steps=3,
        reward=RewardSpec(**target["reward"]), partial_contact_weight=target["partial_contact_weight"],
        world_generator=optimization.generator,
        compartment_names={i: name for i, name in enumerate(sorted(case.compartments), 1)}, cancelled=guard)
    template_seconds = time.perf_counter() - phase
    pre.assert_public_action_model(sim, v2, reference)
    if (sim.decision_model_hash != expected["decision_model_hash"]
            or sim._proposer.model_hash != expected["proposal_model_hash"]
            or sim.decision_model_hash != model["decision_model_hash"]):
        raise ValueError("Scientific axis model changed")
    template = sim.metrics()
    if template["proposal_accounting"]["preview_calls"] != value["fixed_work"]["common_template_preview_attempts"]:
        raise RuntimeError("Common template preview work denominator changed")
    initial = json.loads((ROOT / V2_PROFILE / "initial/inventory.json").read_text())
    if _canonical(scientific_inventory(template["inventory_receipts"][-1])) != _canonical(scientific_inventory(initial["complete_inventory"])):
        raise RuntimeError("Common template inventory differs from original V2")
    pre.write_json(output / "template.json", {"factory_seconds": template_seconds,
        "initialization": template["initialization_timing"], "proposal_accounting": template["proposal_accounting"],
        "scope": "one common uncached constructor/full initial inventory; not credited as cache warmup"})
    actions, seed = tuple(expected["fixed_actions"]), expected["optimization_seed"]
    if actions != tuple(historical["actions"]) or content_hash(historical["metrics"]["history"]) != expected["greedy_history_hash"]:
        raise ValueError("Declared fixed action/history identity differs from V2")
    baseline_text = baseline_masks = None
    cache = None
    audit_histories, reports = {}, []
    for mode in MODES:
        guard.require()
        if mode == "cached_cold":
            phase = time.perf_counter()
            cache = ExactCapsuleCoverCache(cfg, max_payload_bytes=value["cache"]["max_payload_bytes"],
                                          max_entries=value["cache"]["max_entries"])
            pre.write_json(output / "cache-construction.json", {"seconds": time.perf_counter() - phase,
                "stats": cache.stats(), "scope": "empty cache constructed after reference_before"})
            if cache.stats()["entries"] or cache.stats()["calls"]:
                raise RuntimeError("Cold cache is not empty")
        before = None if cache is None else cache.stats()
        phase = time.perf_counter()
        with inject_native_capsule_cache(cache) if mode.startswith("cached") else nullcontext():
            trace, masks, timing = run_fixed_trace(sim, actions, seed, guard, output / mode)
        if native.capsule_voxel_indices is not geometry.capsule_voxel_indices or "propose" in sim._proposer.__dict__:
            raise RuntimeError("Experimental injection or measurement was not restored")
        current_text = _canonical(trace)
        if baseline_text is None:
            baseline_text, baseline_masks = current_text, masks
        elif current_text != baseline_text or masks != baseline_masks:
            _write_gzip(output / f"{mode}-mismatch.json.gz", trace)
            raise RuntimeError("Cached/reversal scientific output differs from reference")
        if (content_hash(trace["metrics"]["history"]) != expected["greedy_history_hash"]
                or trace["metrics"]["total_reward"] != historical["total_reward"]
                or _canonical(trace["metrics"]["inventory_receipts"]) != _canonical(
                    [scientific_inventory(row) for row in historical["metrics"]["inventory_receipts"]])):
            raise RuntimeError("Fixed replay history/reward/inventories differ from original V2")
        counts = [len(row["attempts"]) for row in trace["metrics"]["inventory_receipts"]]
        if (counts != value["fixed_work"]["preview_attempts_per_inventory"]
                or timing["adapter_proposal_accounting"]["preview_calls"] != value["fixed_work"]["preview_attempts_per_episode"]):
            raise RuntimeError("Complete inventory work denominator changed")
        export_started = time.perf_counter()
        trace_receipt = _write_gzip(output / mode / "scientific-trace.json.gz", trace)
        export_seconds = time.perf_counter() - export_started
        audit_histories[mode] = trace["metrics"]["history"]
        summary = {"mode": mode, "scientific_trace_hash": content_hash(trace),
            "actions": actions, "history_hash": content_hash(audit_histories[mode]),
            "total_reward": trace["metrics"]["total_reward"], "timing": timing,
            "scientific_trace_export": trace_receipt, "scientific_trace_export_seconds": export_seconds,
            "cache_before": before, "cache_after": None if cache is None else cache.stats(),
            "seconds_including_scientific_comparison_and_export": time.perf_counter() - phase,
            "process_cumulative_peak_rss_bytes": pre.peak_rss_bytes(), "exact_scientific_equality": True}
        pre.write_json(output / mode / "receipt.json", summary)
        reports.append(summary)
        guard.require()
    audits = {}
    cache_calls_before_audit = cache.stats()["calls"]
    for mode, history in audit_histories.items():
        phase = time.perf_counter()
        audit = pre.independent_check_native_history(case, cfg.tools, history, tissue_mask=cfg.tissue_mask,
            access=cfg.access, hard_exclusion=cfg.hard_exclusion, cancelled=guard)
        audits[mode] = {"receipt": audit.to_dict(), "seconds": time.perf_counter() - phase}
        pre.write_json(output / f"{mode}-independent-audit.json", audits[mode])
        if not audit.feasible or not audit.complete_tool_checked or not audit.frontier_checked:
            raise RuntimeError("Independent native replay failed")
        guard.require()
    if cache.stats()["calls"] != cache_calls_before_audit:
        raise RuntimeError("Independent native audit used the experimental capsule cache")
    if source_snapshot()["runtime_content_hash"] != source["runtime_content_hash"]:
        raise RuntimeError("Frozen source changed during public probe")
    if hashlib.sha256(case_bundle.read_bytes()).hexdigest() != target["bundle_sha256"]:
        raise RuntimeError("Declared case bundle changed during public probe")
    guard.require()
    result = {"status": "completed", "phases": reports, "independent_audits": audits,
        "runtime_content_hash": source["runtime_content_hash"], "scientific_model_hash": sim.decision_model_hash,
        "original_v2_runtime_hash": json.loads((ROOT / V2_PROFILE / "sequence-freeze.json").read_text())["source_hash"],
        "resource": guard.receipt(), "gradient_updates": 0, "final_worlds_used": False,
        "stress_worlds_used": False, "scope": value["scope"], "clinical_deficit_probability": None}
    pre.write_json(output / "result.json", result)
    pre.write_json(output / "worker-status.json", {"status": "completed", "result_hash": content_hash(result),
        "runtime_content_hash": source["runtime_content_hash"], "scientific_model_hash": sim.decision_model_hash})


def worker(output, case_bundle):
    value = declaration()
    guard = pre.ResourceGuard(pre.PreflightBudget(**value["resource_budget"]))
    signal.signal(signal.SIGTERM, lambda *_: guard.cancel("parent_cancellation"))
    try:
        _public_worker(output, case_bundle, value, guard)
    except Exception as error:
        pre.write_json(output / "worker-status.json", {"status": "failed", "error_type": type(error).__name__,
            "error": str(error), "resource": guard.receipt(), "eligible_candidate_count": 0})
        raise
    finally:
        pre.write_json(output / "worker-resource.json", guard.receipt())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--case-bundle", type=Path)
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.worker:
        worker(args.output, args.case_bundle)
        return
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    value = declaration()
    _check_bound_inputs(value)
    pre.write_json(output / "declaration.json", value)
    pre.write_json(output / "launcher-status.json", {"status": "declared_not_executed", "gradient_updates": 0})
    if not args.execute:
        return
    if args.case_bundle is None:
        raise ValueError("Public execution requires an explicit case bundle and prior release")
    started = time.perf_counter()
    snapshot = source_snapshot()
    frozen = output / "frozen-source"
    for name, expected in snapshot["file_sha256"].items():
        data = (ROOT / name).read_bytes()
        if hashlib.sha256(data).hexdigest() != expected:
            raise RuntimeError("Source changed while preparing worker")
        path = frozen / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(data)
    if source_snapshot()["runtime_content_hash"] != snapshot["runtime_content_hash"]:
        raise RuntimeError("Source changed before worker launch")
    pre.write_json(output / "launch-source.json", snapshot)
    command = [sys.executable, str(frozen / "scripts/probe_public_native_capsule_cache.py"),
        "--worker", "--output", str(output), "--case-bundle", str(args.case_bundle.resolve())]
    env = {**os.environ, "PYTHONPATH": str(frozen / "src")}
    budget = pre.PreflightBudget(**value["resource_budget"])
    pre.write_json(output / "launcher-status.json", {"status": "running", "command": command,
        "launch_preparation_seconds": time.perf_counter() - started, "budget": asdict(budget)})
    parent_timeout = hard_killed = False
    with (output / "worker.log").open("w") as log:
        process = subprocess.Popen(command, cwd=frozen, env=env, stdout=log, stderr=subprocess.STDOUT)
        try:
            process.wait(timeout=budget.worker_wall_seconds)
        except subprocess.TimeoutExpired:
            parent_timeout = True
            process.terminate()
            try:
                process.wait(timeout=budget.hard_termination_grace_seconds)
            except subprocess.TimeoutExpired:
                hard_killed = True
                process.kill()
                process.wait()
    status_path = output / "worker-status.json"
    result_path = output / "result.json"
    receipt_error = None
    try:
        status = json.loads(status_path.read_text()) if status_path.exists() else {}
        result = json.loads(result_path.read_text()) if result_path.exists() else {}
        if not isinstance(status, dict) or not isinstance(result, dict):
            raise ValueError("Worker status and result must both be JSON objects")
    except (OSError, ValueError) as error:
        receipt_error = {"error_type": type(error).__name__, "error": str(error)}
        status, result = {}, {}
    complete = process.returncode == 0 and not parent_timeout and status.get("status") == "completed"
    complete = complete and content_hash(result) == status.get("result_hash")
    complete = complete and status.get("runtime_content_hash") == snapshot["runtime_content_hash"]
    complete = complete and result.get("runtime_content_hash") == snapshot["runtime_content_hash"]
    complete = complete and result.get("status") == "completed"
    pre.write_json(output / "launcher-status.json", {"status": "completed" if complete else "failed",
        "worker_returncode": process.returncode, "parent_timeout_requested": parent_timeout,
        "hard_killed": hard_killed, "receipt_error": receipt_error,
        "full_launcher_seconds": time.perf_counter() - started,
        "authority": "Worker and result must both be completed with the launch runtime and matching result hash; launcher must also complete"})
    if not complete:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
