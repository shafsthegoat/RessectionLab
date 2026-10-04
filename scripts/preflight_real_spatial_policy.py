#!/usr/bin/env python3
"""Profile or verify one spatial-policy update on a declared real TRAIN patient.

This is a pipeline check, not a population study or efficacy comparison. It
opens no selection/transfer patient and does not select a checkpoint by reward.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict
import hashlib
import json
import os
from pathlib import Path
import resource
import signal
import subprocess
import sys
import time
import traceback

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
VERSION = "real-patient-spatial-preflight-v1"
COHORT_PATH = "manifests/experiments/btc-spatial-development-cohort-v1.json"
COHORT_SHA256 = "962d964e1d71427f3625cdbebc0f7e4759e5d2345d8f95cb211ed45810ed2985"
TRAIN_SUBJECTS = frozenset({"sub-PAT05", "sub-PAT16", "sub-PAT20", "sub-PAT22", "sub-PAT25", "sub-PAT28"})


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path, payload):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, sort_keys=True, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def require_train_role(cohort, subject):
    """Check source patient identity before opening any case bundle."""
    groups = {row["subject"]: row for row in cohort["existing_development_records"]}
    for row in cohort["candidates"]:
        if row["subject"] in groups:
            raise ValueError("Duplicate patient in frozen cohort")
        groups[row["subject"]] = row
    expected_role = ("population_training" if subject in {"sub-PAT22", "sub-PAT25"}
                     else "previously_consulted_training_and_method_development")
    if subject not in TRAIN_SUBJECTS or subject not in groups or groups[subject]["development_role"] != expected_role:
        raise ValueError("Preflight may open only a declared population TRAIN patient")
    return "BTC:" + subject


def numerical_source_inventory():
    """Bind the runner and all headless modules, including dynamic imports."""
    paths = [ROOT / "scripts/preflight_real_spatial_policy.py", *(ROOT / "src/resectionlab").rglob("*.py")]
    return {str(path.relative_to(ROOT)): sha256(path) for path in sorted(paths)}


def read_declaration(path, expected_sha256=None):
    """Read one byte snapshot, checking the parent-bound identity before parsing."""
    payload = Path(path).read_bytes()
    actual = hashlib.sha256(payload).hexdigest()
    if expected_sha256 is not None and actual != expected_sha256:
        raise ValueError("Declaration bytes changed between supervisor and worker")
    return json.loads(payload), actual, payload


def validate_declaration(declaration):
    """Freeze checks require no imports of numerical modules or image access."""
    if declaration.get("version") != VERSION:
        raise ValueError("Unknown real-patient preflight version")
    if declaration.get("mode") not in {"profile", "one_update"}:
        raise ValueError("Choose profile or one_update in the frozen declaration")
    if declaration.get("track") != "annotation_assisted":
        raise ValueError("This first real-patient preflight explicitly requires annotation-assisted inputs")
    sources = declaration.get("source_sha256", {})
    if not sources or sources != numerical_source_inventory():
        raise ValueError("Complete numerical source hashes are absent, omitted or changed")
    if declaration.get("cohort_path") != COHORT_PATH or declaration.get("cohort_sha256") != COHORT_SHA256:
        raise ValueError("Cohort path and identity must match the original frozen patient roles")
    cohort_bytes = (ROOT / COHORT_PATH).read_bytes()
    if hashlib.sha256(cohort_bytes).hexdigest() != COHORT_SHA256:
        raise ValueError("Original frozen cohort bytes changed")
    cohort = json.loads(cohort_bytes)
    group = require_train_role(cohort, declaration["subject"])
    return cohort, group, sources


def load_inputs(declaration):
    """Source/cohort integrity and role gates precede real-image loading."""
    cohort, group, sources = validate_declaration(declaration)
    bundle = ROOT / declaration["case_bundle"]
    if sha256(bundle) != declaration["case_bundle_sha256"]:
        raise ValueError("Declared real source bundle changed")
    from resectionlab.imaging import load_case
    case = load_case(bundle)
    expected_id = "BTC-ds001226-" + declaration["subject"] + "-preop"
    source = case.metadata.get("source_collection", {})
    if (case.case_id != expected_id or case.semantic_hash != declaration["case_semantic_hash"]
            or source.get("accession") != "ds001226" or source.get("release") != "5.0.1"
            or source.get("git_commit") != cohort["source"]["git_commit"]
            or case.metadata.get("is_synthetic")
            or not any(ref.source_id == "structural" and ref.provenance == "observed" for ref in case.source_refs)):
        raise ValueError("Bundle does not establish the declared real preoperative BTC identity")
    return case, group, sources


def peak_rss_bytes():
    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return int(peak if sys.platform == "darwin" else peak * 1024)


def episode(base, policy, generator, *, stochastic, profile_actions=False, checkpoint=lambda row: None,
            diagnostics=False, profiler=None, native_affine=None):
    from resectionlab.spatial_policy import SpatialTransition, parameter_hash
    before = parameter_hash(policy)
    started = time.perf_counter()
    task = base.fresh()
    setup_seconds = time.perf_counter() - started
    transitions, decisions = [], []
    coverage, proposal_coverage = None, None
    if diagnostics:
        from resectionlab.spatial_policy_diagnostics import spatial_coverage, runtime_proposal_coverage

        def covered(observation):
            return spatial_coverage(observation, source_shape=task.case.structural_intensity.shape,
                source_affine=task.case.affine_ras_mm, nominal_target=task.case.nominal_target,
                ray_samples=policy.config.ray_samples, native_affine=native_affine)

        def proposals(observation, inventory):
            return runtime_proposal_coverage(task.case, inventory, observation,
                native_affine=native_affine, ray_samples=policy.config.ray_samples)

        initial_observation, initial_inventory = task.observation(), task.candidate_inventory()
        coverage = covered(initial_observation)
        proposal_coverage = proposals(initial_observation, initial_inventory)
        checkpoint({"status": "prepared", "initial_candidate_inventory": initial_inventory,
            "initial_observation_coverage": coverage, "initial_proposal_coverage": proposal_coverage})
    while not task.terminated:
        observation_start = time.perf_counter()
        observation = task.observation()
        observation_seconds = time.perf_counter() - observation_start
        decision_start = time.perf_counter()
        selected = policy.act(observation, stochastic=stochastic, generator=generator)
        forward_seconds = time.perf_counter() - decision_start
        # Profile exercises native steps even when an untrained policy chooses
        # STOP. This fixed rule is separately labeled and never used as a teacher.
        action = (observation.action_ids[1] if profile_actions and len(observation.action_ids) > 1 else
                  "STOP" if profile_actions else selected)
        transition_start = time.perf_counter()
        result = task.step(action)
        transitions.append(SpatialTransition(observation, action, result.reward, result.terminated))
        decisions.append({"action_id": action, "policy_selected_action_id": selected,
            "candidate_count_including_stop": len(observation.action_ids),
            "observation_fingerprint": observation.fingerprint, "observation_seconds": observation_seconds,
            "forward_seconds": forward_seconds, "native_step_seconds": time.perf_counter() - transition_start})
        if diagnostics:
            diagnostic_start = time.perf_counter()
            decisions[-1]["before_observation_coverage"] = coverage
            decisions[-1]["before_proposal_coverage"] = proposal_coverage
            coverage = covered(result.observation)
            decisions[-1]["after_observation_coverage"] = coverage
            inventory = task.candidate_inventory()
            decisions[-1]["after_candidate_inventory"] = inventory
            proposal_coverage = proposals(result.observation, inventory)
            decisions[-1]["after_proposal_coverage"] = proposal_coverage
            decisions[-1]["diagnostic_seconds"] = time.perf_counter() - diagnostic_start
        checkpoint({"status": "collecting", "decisions": decisions,
                    "committed_transition_count": len(transitions), "latest_transition_info": result.info})
    metrics = task.metrics()
    online_seconds = time.perf_counter() - started
    checkpoint({"status": "awaiting_independent_audit", "metrics": metrics, "decisions": decisions,
                "online_seconds": online_seconds})
    check_start = time.perf_counter()
    if profiler is None:
        checked = task.independent_geometry_check()
    else:
        with profiler.phase(profiler.phase_name + ":independent_audit"):
            checked = task.independent_geometry_check()
    audit = asdict(checked)
    checkpoint({"independent_geometry_check": audit})
    if not checked.feasible:
        raise RuntimeError("Independent native checker rejected the executed history")
    if parameter_hash(policy) != before:
        raise RuntimeError("Collection or replay changed the frozen rollout policy")
    return transitions, {"metrics": metrics, "decisions": decisions, "rollout_forward_calls": len(decisions),
        "task_setup_seconds": setup_seconds, "online_seconds": online_seconds,
        "independent_audit_seconds": time.perf_counter() - check_start,
        "independent_geometry_check": audit, "parameter_hash": before,
        "profile_action_rule": "first_legal_inventory_row; not a search teacher" if profile_actions else None}


def worker(declaration, output):
    validate_declaration(declaration)
    from resectionlab.native_resection import NativeResectionEngine
    from resectionlab.spatial_policy_diagnostics import NativePreviewProfiler
    with NativePreviewProfiler(NativeResectionEngine) as profiler:
        return _worker_with_profiler(declaration, output, profiler)


def _worker_with_profiler(declaration, output, profiler):
    import torch
    from resectionlab.geometry import AccessWindow, ToolGeometry
    from resectionlab.native_spatial_task import native_spatial_task_from_case
    from resectionlab.spatial_policy import (SpatialPolicy, SpatialPolicyConfig,
        gradient_step, parameter_hash, reinforce_loss)
    from resectionlab.spatial_policy_diagnostics import nominal_depth_coverage, spatial_coverage, runtime_proposal_coverage

    started = time.perf_counter()
    settings = declaration["settings"]
    case, group, sources = load_inputs(declaration)
    input_seconds = time.perf_counter() - started
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    torch.use_deterministic_algorithms(True)
    torch.manual_seed(settings["seed"])
    policy = SpatialPolicy(SpatialPolicyConfig(**declaration["policy_config"]))
    if policy.architecture_hash != declaration["expected_policy_architecture_hash"]:
        raise ValueError("Policy architecture differs from the declared spatial/candidate model")
    generator = torch.Generator().manual_seed(settings["seed"] + 100000)
    preparation_start = time.perf_counter()
    with profiler.phase("initial_task_preparation"):
        task = native_spatial_task_from_case(case, access=AccessWindow(**declaration["access"]),
            tools=tuple(ToolGeometry(**row) for row in declaration["tools"]),
            max_steps=settings["max_steps"], track=declaration["track"], **declaration.get("adapter_options", {}))
    preparation_seconds = time.perf_counter() - preparation_start
    reward_weights = asdict(task.reward_spec)
    if any(declaration["objective"].get(key) != value for key, value in reward_weights.items()):
        raise ValueError("Executed geometric objective differs from its declared physical weights")
    receipt = {"version": VERSION, "mode": declaration["mode"], "subject": declaration["subject"],
        "patient_group": group, "independent_human_patients": 1, "population_generalization_measured": False,
        "source_sha256": sources, "input_seconds": input_seconds, "preparation_seconds": preparation_seconds,
        "architecture": policy.architecture_record(), "architecture_hash": policy.architecture_hash,
        "reward_weights": reward_weights, "initial_parameter_hash": parameter_hash(policy),
        "optimizer_updates": 0, "episodes": [], "scope": "real anatomy; simulated actions and outcomes; pipeline preflight only"}
    receipt["initial_task_metrics"] = task.metrics()
    grid_record = receipt["initial_task_metrics"].get("native_grid_reconciliation", {})
    expected_grid = declaration.get("expected_native_grid_binding", {})
    if any(grid_record.get(key) != value for key, value in expected_grid.items()):
        raise ValueError("Executed native grid differs from the declared source/derived frame binding")
    native_affine = grid_record.get("derived_affine_ras_mm")
    initial_observation = task.observation()
    receipt["initial_candidate_inventory"] = task.candidate_inventory()
    if getattr(task.case, "proposal_mode", "fixed_lattice") == "fixed_lattice":
        receipt["nominal_depth_coverage"] = nominal_depth_coverage(task.case, native_affine=native_affine)
    receipt["initial_proposal_coverage"] = runtime_proposal_coverage(task.case,
        receipt["initial_candidate_inventory"], initial_observation, native_affine=native_affine,
        ray_samples=policy.config.ray_samples)
    receipt["initial_observation_coverage"] = spatial_coverage(initial_observation,
        source_shape=task.case.structural_intensity.shape, source_affine=task.case.affine_ras_mm,
        nominal_target=task.case.nominal_target, ray_samples=policy.config.ray_samples,
        native_affine=native_affine)
    write_json(output / "declaration.json", declaration)

    def preserve():
        receipt.update(elapsed_seconds=time.perf_counter() - started, peak_rss_bytes=peak_rss_bytes())
        receipt["native_preview_cost"] = profiler.snapshot()
        write_json(output / "receipt.json", receipt)
        if receipt["peak_rss_bytes"] > settings["max_rss_bytes"]:
            raise RuntimeError("Declared real preflight memory limit exceeded")
        if receipt["elapsed_seconds"] > settings["max_wall_seconds"]:
            raise RuntimeError("Declared real preflight wall limit exceeded")

    preserve()
    if declaration["mode"] == "one_update" and len(initial_observation.action_ids) <= 1:
        receipt.update(status="abstained", reason="no_initial_legal_nonstop_action", optimizer_updates=0)
        preserve()
        return

    def run_episode(phase, *, stochastic=False, profile_actions=False):
        saved = {"phase": phase, "status": "starting"}
        receipt["episodes"].append(saved)
        preserve()

        def checkpoint(row):
            saved.update(row)
            preserve()

        with profiler.phase(phase):
            transitions, row = episode(task, policy, generator, stochastic=stochastic,
                profile_actions=profile_actions, checkpoint=checkpoint, diagnostics=True, profiler=profiler,
                native_affine=native_affine)
        saved.update(row, status="complete")
        preserve()
        return transitions

    if declaration["mode"] == "profile":
        run_episode("untrained_fixed_inventory_profile", profile_actions=True)
    else:
        run_episode("initial_policy_readout_not_checkpoint_selection")
        batch = []
        for _ in range(settings["episodes_per_update"]):
            batch.append(run_episode("optimization", stochastic=True))
        update_start = time.perf_counter()
        optimizer = torch.optim.Adam(policy.parameters(), lr=settings["learning_rate"])
        loss, learning = reinforce_loss(policy, batch, gamma=settings["gamma"],
            entropy_weight=settings["entropy_weight"], value_weight=settings["value_weight"])
        learning.update(gradient_step(policy, optimizer, loss, max_norm=settings["gradient_clip"]))
        receipt.update(learning=learning, update_seconds=time.perf_counter() - update_start,
            optimizer_updates=1, updated_parameter_hash=parameter_hash(policy))
        torch.save(policy.state_dict(), output / "latest-policy.pt")
        receipt["latest_checkpoint_sha256"] = sha256(output / "latest-policy.pt")
        preserve()
        run_episode("latest_policy_readout_not_checkpoint_selection")
    if numerical_source_inventory() != sources:
        raise RuntimeError("Numerical source changed during real preflight")
    receipt["status"] = "complete"
    preserve()


def supervise_worker(command, output, settings, declaration_sha256):
    """Enforce sampled worker RSS and wall limits while native work is running."""
    output = Path(output)
    started, maximum, samples, reason = time.perf_counter(), 0, 0, None
    environment = {**os.environ, "OMP_NUM_THREADS": "1", "OPENBLAS_NUM_THREADS": "1",
                   "MKL_NUM_THREADS": "1", "VECLIB_MAXIMUM_THREADS": "1", "NUMEXPR_NUM_THREADS": "1"}

    def stop(process):
        if process.poll() is not None:
            return
        try:
            os.killpg(process.pid, signal.SIGTERM)
        except ProcessLookupError:
            return
        try:
            process.wait(timeout=2)
        except subprocess.TimeoutExpired:
            try:
                os.killpg(process.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            process.wait(timeout=2)

    with (output / "worker.log").open("w") as log:
        process = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT,
                                   env=environment, start_new_session=True)
        try:
            while process.poll() is None:
                elapsed = time.perf_counter() - started
                if elapsed >= settings["max_wall_seconds"]:
                    reason = "parent_wall_budget_exceeded"
                    break
                measurement = subprocess.run(["ps", "-o", "rss=", "-p", str(process.pid)],
                    capture_output=True, text=True, timeout=2, check=False)
                try:
                    current = int(measurement.stdout.strip()) * 1024
                except ValueError:
                    if process.poll() is None:
                        raise RuntimeError("Worker RSS unavailable while process remains running")
                    current = 0
                maximum, samples = max(maximum, current), samples + 1
                if current > settings["max_rss_bytes"]:
                    reason = "parent_sampled_rss_budget_exceeded"
                write_json(output / "supervisor-progress.json", {"pid": process.pid,
                    "elapsed_seconds": time.perf_counter() - started, "sampled_peak_rss_bytes": maximum,
                    "samples": samples, "termination_reason": reason, "declaration_sha256": declaration_sha256})
                if reason:
                    break
                time.sleep(.2)
        except BaseException as error:
            reason = "parent_supervision_error:" + type(error).__name__ + ":" + str(error)
        finally:
            stop(process)
        code = process.wait()
    result = {"status": "complete" if code == 0 and reason is None else "failed",
        "returncode": code, "timed_out": reason == "parent_wall_budget_exceeded",
        "termination_reason": reason, "seconds": time.perf_counter() - started,
        "sampled_peak_rss_bytes": maximum, "rss_samples": samples,
        "rss_scope": "worker process RSS; native work uses threads, not subprocess workers",
        "sampling_interval_seconds": .2, "measurement_timeout_seconds": 2,
        "termination_grace_seconds": 2, "sampling_limit": "transient peaks between samples can be missed",
        "automatic_retry": False, "declaration_sha256": declaration_sha256}
    write_json(output / "supervisor.json", result)
    if result["status"] != "complete":
        write_json(output / "supervisor-failure.json", result)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--declaration", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--expected-declaration-sha256", help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.worker:
        try:
            if not args.expected_declaration_sha256:
                raise ValueError("Worker requires the supervisor's frozen declaration identity")
            declaration, _, _ = read_declaration(args.declaration, args.expected_declaration_sha256)
            worker(declaration, args.output)
        except BaseException as error:
            write_json(args.output / "failure.json", {"type": type(error).__name__,
                "message": str(error), "traceback": traceback.format_exc(), "peak_rss_bytes": peak_rss_bytes()})
            raise
        return
    if args.output.exists():
        raise SystemExit("Preserve earlier attempts: choose a new output directory")
    declaration, declaration_sha256, declaration_bytes = read_declaration(args.declaration)
    validate_declaration(declaration)
    args.output.mkdir(parents=True)
    snapshot = args.output / "declaration-input.json"
    snapshot.write_bytes(declaration_bytes)
    result = supervise_worker([sys.executable, str(Path(__file__).resolve()), "--worker",
        "--declaration", str(snapshot.resolve()), "--expected-declaration-sha256", declaration_sha256,
        "--output", str(args.output.resolve())], args.output, declaration["settings"], declaration_sha256)
    write_json(args.output / "output-sha256.json", {path.name: sha256(path) for path in sorted(args.output.iterdir())
        if path.is_file() and path.name != "output-sha256.json"})
    if result["status"] != "complete":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
