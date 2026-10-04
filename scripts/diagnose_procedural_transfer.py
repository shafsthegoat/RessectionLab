#!/usr/bin/env python3
"""Read-only actor/feature diagnostics after a fully completed development run.

No training, search, final/stress sampling, checkpoint selection or parameter
changes. Saved SEARCH actions define common patient optimization states; the
two procedural families contribute their initial and first legal-cut states.
All policies are compared on identical observations, using the preserved code.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys
import time
from typing import Any


def read_json(path: Path) -> Any:
    return json.loads(path.read_text())


def file_hash(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def require_completed_run(run: Path) -> dict[str, Any]:
    """Reject partial/invalidated runs before loading models or patient images."""
    records = {name: read_json(run / name) for name in (
        "experiment-status.json", "summary.json", "comparison/status.json")}
    if any(record.get("status") != "completed" or record.get("final_worlds_used") is not False
           for record in records.values()):
        raise ValueError("Diagnostics require a completed run with no final-world evaluation")
    status = records["comparison/status.json"]
    planned = status.get("planned_learning_runs", [])
    if len(planned) != 6 or set(planned) != set(status.get("completed_learning_runs", [])):
        raise ValueError("All three scratch and three adapted runs must be completed")
    summary = records["summary.json"]
    if not summary.get("independent_geometry") or not all(value is True for value in summary["independent_geometry"].values()):
        raise ValueError("Diagnostic comparison requires completed independent geometry validation")
    if summary.get("validation", {}).get("rejected_candidate_ids"):
        raise ValueError("Invalidated candidates must not enter a validated comparison")
    return summary


def actor_statistics(policy, observation, feature_names: tuple[str, ...]) -> dict[str, Any]:
    """Actual masked probabilities plus first-layer scale/saturation measurements."""
    import numpy as np
    import torch

    with torch.no_grad():
        logits, value = policy(observation)
        probabilities = torch.softmax(logits, dim=0)
        mask = torch.as_tensor(np.array(observation.action_mask, copy=True), dtype=torch.bool)
        legal = probabilities[mask]
        entropy = float(-(legal * torch.log(legal.clamp_min(torch.finfo(legal.dtype).tiny))).sum())
        actions = torch.as_tensor(np.array(observation.action_features, copy=True), dtype=torch.float32)
        state = torch.as_tensor(np.array(observation.state_features, copy=True), dtype=torch.float32)
        inputs = torch.cat((actions, state.expand(len(actions), -1)), dim=1)
        first = policy.actor[0]
        preactivation = first(inputs)
        activation = torch.tanh(preactivation)
        derivative = 1. - activation.square()
        legal_rows = torch.where(mask)[0]
        rows = []
        for row in legal_rows.tolist():
            rows.append({"action_id": observation.action_ids[row], "logit": float(logits[row]),
                "probability": float(probabilities[row]), "preactivation_abs_mean": float(preactivation[row].abs().mean()),
                "preactivation_abs_max": float(preactivation[row].abs().max()),
                "tanh_abs_ge_095_fraction": float((activation[row].abs() >= .95).float().mean()),
                "tanh_abs_ge_099_fraction": float((activation[row].abs() >= .99).float().mean()),
                "mean_tanh_derivative": float(derivative[row].mean())})
        contribution = inputs[mask].abs().mean(dim=0) * first.weight.abs().mean(dim=0)
        return {"entropy_nats": entropy, "maximum_entropy_nats": float(np.log(len(legal))),
            "normalized_entropy": entropy / float(np.log(len(legal))) if len(legal) > 1 else None,
            "stop_probability": float(probabilities[0]),
            "greedy_action_id": observation.action_ids[int(torch.argmax(logits))], "value_prediction": float(value),
            "legal_action_count": int(mask.sum()), "rows": rows,
            "first_layer_mean_absolute_feature_contribution": dict(zip(feature_names, contribution.tolist()))}


def feature_statistics(observation, names: tuple[str, ...]) -> dict[str, Any]:
    import numpy as np
    actions = np.asarray(observation.action_features)
    mask = np.asarray(observation.action_mask, bool)
    nonstop = actions[mask & (np.arange(len(mask)) != 0)]
    return {"action_ids": list(observation.action_ids), "action_mask": mask.tolist(),
        "action_features": actions.tolist(), "state_features": np.asarray(observation.state_features).tolist(),
        "legal_nonstop_ranges": {name: {"min": float(nonstop[:, index].min()), "max": float(nonstop[:, index].max()),
            "mean": float(nonstop[:, index].mean())} for index, name in enumerate(names)} if len(nonstop) else {}}


def gradient_statistics(training: dict[str, Any]) -> dict[str, Any]:
    import numpy as np
    history = training.get("optimization_history", [])
    fields = ("mean_return", "loss", "gradient_norm_before_clip", "actor_gradient_norm_after_clip")
    result = {"gradient_steps": training["gradient_steps"],
        "optimization_environment_steps": training["optimization_environment_steps"],
        "selection_environment_steps": training["selection_environment_steps"],
        "elapsed_seconds": training["elapsed_seconds"], "initialization_seconds": training.get("initialization_seconds"),
        "actor_parameters_changed": training["actor_parameters_changed"],
        "selected_is_initial": training["selected_checkpoint_hash"] == training["initial_checkpoint_hash"],
        "initial_selection_return": training["initial_selection_return"],
        "selected_selection_return": training["selected_selection_return"],
        "selection_history": training.get("selection_history", []), "statistics": {}}
    for field in fields:
        values = np.asarray([row[field] for row in history], float)
        result["statistics"][field] = ({"first": float(values[0]), "last": float(values[-1]),
            "min": float(values.min()), "median": float(np.median(values)), "max": float(values.max()),
            "mean": float(values.mean()), "zero_count": int((values == 0).sum())} if len(values) else None)
    return result


def _load_frozen_runtime(run: Path) -> dict[str, Any]:
    frozen = run / "frozen-source"
    snapshot = read_json(run / "worker-source.json")
    for name, digest in snapshot["numerical_runtime_sha256"].items():
        if file_hash(frozen / name) != digest:
            raise ValueError(f"Preserved numerical source changed: {name}")
    if any(name == "resectionlab" or name.startswith("resectionlab.") for name in sys.modules):
        raise RuntimeError("Run diagnostics in a fresh process to bind preserved source imports")
    sys.path.insert(0, str(frozen / "src"))
    return snapshot


def diagnose(run: Path, output: Path) -> dict[str, Any]:
    started = time.perf_counter()
    summary = require_completed_run(run)
    snapshot = _load_frozen_runtime(run)
    import numpy as np
    import torch
    from resectionlab.imaging import load_case
    from resectionlab.learning import load_policy, policy_hash
    from resectionlab.native_simulation import NATIVE_ACTION_FEATURE_NAMES, make_native_patient_simulator
    from resectionlab.procedural_learning import (DECLARATION_HASH, TransferTarget,
        make_native_procedural_fixture, native_observation_schema, validate_procedural_world_panels)
    from resectionlab.worlds import WorldGeneratorConfig, WorldPartitionManifest, WorldRole, content_hash

    if output.exists():
        raise FileExistsError("Diagnostic output exists; preserve it and choose a new path")
    declaration = read_json(run / "frozen-source/manifests/experiments/procedural-native-to-ucsf-v1.json")
    if declaration["declaration_content_hash"] != DECLARATION_HASH or summary["declaration_hash"] != DECLARATION_HASH:
        raise ValueError("Completed run does not match preserved preregistration")
    case_path = run / "source-case.ressectionlab"
    if file_hash(case_path) != declaration["target"]["bundle_sha256"]:
        raise ValueError("Patient bundle bytes changed after completed comparison")
    case = load_case(case_path)
    base = make_native_patient_simulator(case, candidate_count=4, max_steps=3, max_actions=7)
    target_row = declaration["target"]
    if case.semantic_hash != target_row["semantic_hash"] or base.decision_model_hash != target_row["decision_model_hash"]:
        raise ValueError("Reconstructed patient source/model differs from completed comparison")
    target = TransferTarget(case.semantic_hash, case.planning_hash, target_row["group_id"],
                            tuple(target_row["aliases"]), target_row["source_kind"])
    def partition(role):
        row = target_row["world_partitions"][role]
        return WorldPartitionManifest(WorldRole(row["role"]), row["case_hash"],
            WorldGeneratorConfig(**row["generator"]), tuple(row["seeds"]), row["planning_hash"])
    optimization, selection = partition("optimization"), partition("selection")
    panels_receipt = validate_procedural_world_panels(target, optimization, selection)
    before_files: dict[str, str] = {}
    policies = {}
    policy_records = {}
    def retain_policy(label, path, expected_hash, *, selected=True):
        policy = load_policy(path, selected=selected)
        if policy_hash(policy) != expected_hash:
            raise ValueError(f"Checkpoint weight mismatch for {label}")
        policy.requires_grad_(False)
        policies[label] = policy
        before_files[str(path.relative_to(run))] = file_hash(path)
        policy_records[label] = {"checkpoint_file": str(path.relative_to(run)),
            "checkpoint_file_sha256": file_hash(path), "policy_hash": expected_hash, "selected_weights": selected}
    shared = read_json(run / "comparison/shared-provenance.json")
    shared_path = run / "comparison/procedural-source.pt"
    if file_hash(shared_path) != shared["checkpoint_file_sha256"]:
        raise ValueError("Shared checkpoint bytes differ from completed comparison")
    retain_policy("PROCEDURAL_SHARED", shared_path, shared["policy_hash"])
    training_rows = {}
    for row in summary["learning"]:
        mode, seed = row["optimizer_mode"], row["seed"]
        prefix = "scratch" if mode == "PATIENT_SCRATCH_RL" else "adapted"
        folder = run / "comparison" / f"{prefix}-{seed}"
        detailed = read_json(folder / "result.json")
        contract = read_json(folder / "contract.json")
        if detailed["decision_model_hash"] != base.decision_model_hash:
            raise ValueError("Learning record belongs to a different patient model")
        for key in ("initial_checkpoint_hash", "selected_checkpoint_hash", "latest_checkpoint_hash", "gradient_steps"):
            if detailed[key] != row[key]:
                raise ValueError("Summary differs from completed learning record")
        label = f"{prefix}:{seed}"
        training_rows[label] = gradient_statistics(detailed)
        norms = [update["gradient_norm_before_clip"] for update in detailed["optimization_history"]]
        training_rows[label]["gradient_clip_threshold"] = contract["config"]["max_gradient_norm"]
        training_rows[label]["preclip_exceeded_threshold_fraction"] = float(np.mean(np.asarray(norms) > contract["config"]["max_gradient_norm"])) if norms else None
        retain_policy(label + ":initial", folder / "initial.pt", row["initial_checkpoint_hash"])
        retain_policy(label + ":selected", folder / "checkpoint.pt", row["selected_checkpoint_hash"])
        retain_policy(label + ":latest", folder / "checkpoint.pt", row["latest_checkpoint_hash"], selected=False)
    offline = read_json(run / "pretraining/training/result.json")
    offline_diagnostics = gradient_statistics(offline)
    retain_policy("PRETRAINING_INITIAL", run / "pretraining/training/initial.pt", offline["initial_checkpoint_hash"])
    retain_policy("PRETRAINING_LATEST", run / "pretraining/training/checkpoint.pt", offline["latest_checkpoint_hash"], selected=False)

    schema = native_observation_schema(base)
    names = tuple(NATIVE_ACTION_FEATURE_NAMES)
    all_names = names + tuple("state:" + value for value in schema["state_feature_names"])
    states = []
    transitions = []
    def capture(domain, label, simulator, seed):
        observation = simulator.observation()
        metrics = simulator.metrics()
        states.append({"domain": domain, "state": label, "role": "optimization", "world_seed": seed,
            "case_hash": simulator.case_hash, "decision_model_hash": simulator.decision_model_hash,
            "nominal_immediate_rewards": {action.action_id: simulator.nominal_action_value(action)
                                          for action in simulator.proposed_actions()},
            "physical_totals": {name: metrics[name] for name in (
                "simulated_removed_target_volume_mm3", "simulated_removed_normal_volume_mm3",
                "modeled_residual_target_volume_mm3", "cumulative_partial_normal_contact_mm3")},
            "observations": feature_statistics(observation, names),
            "policies": {name: actor_statistics(policy, observation, all_names) for name, policy in policies.items()}})
    def advance(domain, simulator, action, seed):
        transition = simulator.step(action)
        transitions.append({"domain": domain, "action": action, "world_seed": seed,
            "reward": transition.reward, "terminated": transition.terminated})
    for member in make_native_procedural_fixture(target):
        simulator = member.factory()
        seed = member.optimization.seeds[0]
        simulator.reset(seed)
        capture(member.family_id, "initial", simulator, seed)
        action = simulator.observation().action_ids[1]
        advance(member.family_id, simulator, action, seed)
        capture(member.family_id, "after_first_declared_legal_action", simulator, seed)
    seed = optimization.seeds[0]
    base.reset(seed)
    capture("UCSF-PDGM-0004", "initial", base, seed)
    search = next(row for row in summary["search"] if row["method"] == "SEARCH")
    for index, action in enumerate(search["actions"]):
        if action == "STOP":
            break
        advance("UCSF-PDGM-0004", base, action, seed)
        capture("UCSF-PDGM-0004", f"after_saved_SEARCH_action_{index + 1}", base, seed)
    for path, digest in before_files.items():
        if file_hash(run / path) != digest:
            raise ValueError("Read-only diagnostics changed a checkpoint")
    if any(parameter.grad is not None for policy in policies.values() for parameter in policy.parameters()):
        raise ValueError("Read-only diagnostics unexpectedly produced gradients")
    output.mkdir(parents=True)
    script_path = Path(__file__).resolve()
    script_bytes = script_path.read_bytes()
    (output / "diagnostic-script.py").write_bytes(script_bytes)
    report = {"schema_version": 1, "status": "completed_read_only_diagnostic", "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "run": str(run), "completed_summary_sha256": file_hash(run / "summary.json"),
        "source_runtime_hash": snapshot["numerical_runtime_content_hash"],
        "diagnostic_script_sha256": hashlib.sha256(script_bytes).hexdigest(), "declaration_hash": DECLARATION_HASH,
        "source_case_sha256": file_hash(case_path), "decision_model_hash": base.decision_model_hash,
        "panel_preflight": panels_receipt, "feature_schema": schema, "policy_bindings": policy_records,
        "training": training_rows, "offline_training": offline_diagnostics,
        "saved_SEARCH": search, "frozen_summary": summary["frozen"], "states": states,
        "diagnostic_transitions": transitions, "new_gradient_steps": 0, "new_search_expansions": 0,
        "final_worlds_used": False, "stress_worlds_used": False, "checkpoints_unchanged": True,
        "seconds": time.perf_counter() - started,
        "limits": ["Post hoc descriptive development analysis; no causal improvement claim",
            "One previously studied patient-derived structural simulation, no motor/language assessment",
            "Common saved-search states do not estimate policy occupancy distributions",
            "Large physical features or tanh saturation alone do not prove transfer failure",
            "No normalization, objective, checkpoint selection or experiment setting changed"]}
    (output / "diagnostic.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = diagnose(args.run.resolve(), args.output.resolve())
    print(json.dumps({key: report[key] for key in ("status", "new_gradient_steps", "seconds")}, indent=2))


if __name__ == "__main__":
    main()
