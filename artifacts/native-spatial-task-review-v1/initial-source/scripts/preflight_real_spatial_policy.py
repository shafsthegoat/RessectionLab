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
from pathlib import Path
import resource
import subprocess
import sys
import time
import traceback

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
VERSION = "real-patient-spatial-preflight-v1"


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
    if subject not in groups or groups[subject]["development_role"] not in {
        "previously_consulted_training_and_method_development", "population_training"
    }:
        raise ValueError("Preflight may open only a declared population TRAIN patient")
    return "BTC:" + subject


def load_inputs(declaration):
    """Source/cohort integrity and role gates precede real-image loading."""
    if declaration.get("version") != VERSION:
        raise ValueError("Unknown real-patient preflight version")
    if declaration.get("mode") not in {"profile", "one_update"}:
        raise ValueError("Choose profile or one_update in the frozen declaration")
    if declaration.get("track") != "annotation_assisted":
        raise ValueError("This first real-patient preflight explicitly requires annotation-assisted inputs")
    sources = declaration.get("source_sha256", {})
    if not sources or any(sha256(ROOT / name) != value for name, value in sources.items()):
        raise ValueError("Prospective numerical source hashes are absent or changed")
    cohort_path = ROOT / declaration["cohort_path"]
    if sha256(cohort_path) != declaration["cohort_sha256"]:
        raise ValueError("Frozen patient roles changed")
    cohort = json.loads(cohort_path.read_text())
    group = require_train_role(cohort, declaration["subject"])
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


def episode(base, policy, generator, *, stochastic, profile_actions=False, checkpoint=lambda row: None):
    from resectionlab.spatial_policy import SpatialTransition, parameter_hash
    before = parameter_hash(policy)
    started = time.perf_counter()
    task = base.fresh()
    setup_seconds = time.perf_counter() - started
    transitions, decisions = [], []
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
        checkpoint({"status": "collecting", "decisions": decisions,
                    "committed_transition_count": len(transitions), "latest_transition_info": result.info})
    metrics = task.metrics()
    online_seconds = time.perf_counter() - started
    checkpoint({"status": "awaiting_independent_audit", "metrics": metrics, "decisions": decisions,
                "online_seconds": online_seconds})
    check_start = time.perf_counter()
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
    import torch
    from resectionlab.geometry import AccessWindow, ToolGeometry
    from resectionlab.native_spatial_task import native_spatial_task_from_case
    from resectionlab.spatial_policy import (SpatialPolicy, SpatialPolicyConfig,
        gradient_step, parameter_hash, reinforce_loss)

    started = time.perf_counter()
    settings = declaration["settings"]
    case, group, sources = load_inputs(declaration)
    input_seconds = time.perf_counter() - started
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    torch.use_deterministic_algorithms(True)
    torch.manual_seed(settings["seed"])
    policy = SpatialPolicy(SpatialPolicyConfig(**declaration["policy_config"]))
    generator = torch.Generator().manual_seed(settings["seed"] + 100000)
    preparation_start = time.perf_counter()
    task = native_spatial_task_from_case(case, access=AccessWindow(**declaration["access"]),
        tools=tuple(ToolGeometry(**row) for row in declaration["tools"]),
        max_steps=settings["max_steps"], track=declaration["track"], **declaration.get("adapter_options", {}))
    preparation_seconds = time.perf_counter() - preparation_start
    receipt = {"version": VERSION, "mode": declaration["mode"], "subject": declaration["subject"],
        "patient_group": group, "independent_human_patients": 1, "population_generalization_measured": False,
        "source_sha256": sources, "input_seconds": input_seconds, "preparation_seconds": preparation_seconds,
        "architecture": policy.architecture_record(), "initial_parameter_hash": parameter_hash(policy),
        "optimizer_updates": 0, "episodes": [], "scope": "real anatomy; simulated actions and outcomes; pipeline preflight only"}
    write_json(output / "declaration.json", declaration)

    def preserve():
        receipt.update(elapsed_seconds=time.perf_counter() - started, peak_rss_bytes=peak_rss_bytes())
        write_json(output / "receipt.json", receipt)
        if receipt["peak_rss_bytes"] > settings["max_rss_bytes"]:
            raise RuntimeError("Declared real preflight memory limit exceeded")
        if receipt["elapsed_seconds"] > settings["max_wall_seconds"]:
            raise RuntimeError("Declared real preflight wall limit exceeded")

    preserve()

    def run_episode(phase, *, stochastic=False, profile_actions=False):
        saved = {"phase": phase, "status": "starting"}
        receipt["episodes"].append(saved)
        preserve()

        def checkpoint(row):
            saved.update(row)
            preserve()

        transitions, row = episode(task, policy, generator, stochastic=stochastic,
            profile_actions=profile_actions, checkpoint=checkpoint)
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
    if {name: sha256(ROOT / name) for name in sources} != sources:
        raise RuntimeError("Numerical source changed during real preflight")
    receipt["status"] = "complete"
    preserve()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--declaration", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args()
    declaration = json.loads(args.declaration.read_text())
    if args.worker:
        try:
            worker(declaration, args.output)
        except BaseException as error:
            write_json(args.output / "failure.json", {"type": type(error).__name__,
                "message": str(error), "traceback": traceback.format_exc(), "peak_rss_bytes": peak_rss_bytes()})
            raise
        return
    if args.output.exists():
        raise SystemExit("Preserve earlier attempts: choose a new output directory")
    # These checks open cohort metadata only; worker rechecks before image load.
    cohort = json.loads((ROOT / declaration["cohort_path"]).read_text())
    require_train_role(cohort, declaration["subject"])
    args.output.mkdir(parents=True)
    started = time.perf_counter()
    timed_out, code = False, None
    with (args.output / "worker.log").open("w") as log:
        try:
            completed = subprocess.run([sys.executable, str(Path(__file__).resolve()), "--worker",
                "--declaration", str(args.declaration.resolve()), "--output", str(args.output.resolve())],
                stdout=log, stderr=subprocess.STDOUT, timeout=declaration["settings"]["max_wall_seconds"])
            code = completed.returncode
        except subprocess.TimeoutExpired:
            timed_out = True
    write_json(args.output / "supervisor.json", {"returncode": code, "timed_out": timed_out,
        "seconds": time.perf_counter() - started, "automatic_retry": False,
        "declaration_sha256": sha256(args.declaration)})
    write_json(args.output / "output-sha256.json", {path.name: sha256(path) for path in sorted(args.output.iterdir())
        if path.is_file() and path.name != "output-sha256.json"})
    if timed_out or code != 0:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
