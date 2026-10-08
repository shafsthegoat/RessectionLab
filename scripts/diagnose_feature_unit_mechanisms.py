#!/usr/bin/env python3
"""Read saved tensors/logs only, after the entire feature-unit study completes.

No simulator is constructed and no episode, backward pass, optimizer, candidate
selection or parameter search is run. Common physical observations come from
the previously completed, source-bound v2 diagnostic, not a new replay.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import inspect
import json
import math
from pathlib import Path
import sys
from types import SimpleNamespace
from typing import Any

from diagnose_procedural_transfer import file_hash, gradient_statistics, read_json


PROFILES = ("RAW", "FEATURE_UNITS")
SEEDS = (11, 23, 47)


def completed_records(run: Path) -> tuple[dict, dict]:
    """Read completion gates before any ranking, weights or partial summary."""
    launch = read_json(run / "experiment-status.json")
    status = read_json(run / "study/status.json")
    if any(record.get("status") != "completed" or record.get("final_worlds_used") is not False
           for record in (launch, status)):
        raise ValueError("Mechanism diagnostics require a fully completed study; partial rankings remain unopened")
    expected = {f"{phase}:{profile}:{seed}" for seed in SEEDS for phase in ("scratch", "adapted") for profile in PROFILES}
    if (set(status.get("planned_learning_runs", [])) != expected
            or set(status.get("completed_learning_runs", [])) != expected
            or len(status.get("completed_learning_runs", [])) != 12
            or set(status.get("completed_offline_runs", [])) != set(PROFILES)
            or set(status.get("completed_frozen_runs", [])) != set(PROFILES)):
        raise ValueError("Completed denominator does not cover all declared feature-unit arms")
    summary = read_json(run / "study/summary.json")
    if (summary.get("status") != "completed" or summary.get("final_worlds_used") is not False
            or summary.get("validation", {}).get("rejected_candidate_ids")
            or summary.get("validation", {}).get("candidate_count") != 23
            or summary.get("validation", {}).get("final_worlds_used") is not False
            or len(summary.get("learning", [])) != 12):
        raise ValueError("Completed summary is invalidated or incomplete")
    if summary["run_id"] != status["run_id"] or summary["declaration_hash"] != launch["declaration_hash"]:
        raise ValueError("Completion records refer to different attempts")
    return summary, status


def frozen_runtime(run: Path, summary: dict) -> dict:
    record = read_json(run / "study/source.json")
    if record["numerical_runtime_content_hash"] != summary["source_hash"]:
        raise ValueError("Completed summary source binding differs")
    for name, digest in record["numerical_runtime_sha256"].items():
        if file_hash(run / "frozen-source" / name) != digest:
            raise ValueError(f"Frozen numerical source changed: {name}")
    if any(name == "resectionlab" or name.startswith("resectionlab.") for name in sys.modules):
        raise RuntimeError("Start a fresh process so only the completed study's frozen runtime is imported")
    sys.path.insert(0, str(run / "frozen-source/src"))
    return record


def saved_observations(reference_path: Path, declaration: dict, physical: dict) -> tuple[dict, list[dict]]:
    """Validate historical optimization observations without resetting a world."""
    record = read_json(reference_path)
    if (record.get("status") != "completed_read_only_diagnostic"
            or record.get("final_worlds_used") is not False or record.get("stress_worlds_used") is not False
            or record.get("checkpoints_unchanged") is not True
            or record.get("source_runtime_hash") != declaration["reference_design"]["reference_runtime_content_hash"]
            or record.get("declaration_hash") != physical["declaration_content_hash"]):
        raise ValueError("Saved common observations lack the completed reference's provenance")
    target = physical["target"]
    expected = {target["semantic_hash"]: (target["decision_model_hash"], target["world_partitions"]["optimization"]["seeds"])}
    expected.update({member["source_hash"]: (member["model_hash"], member["world_partitions"]["optimization"]["seeds"])
                     for member in physical["procedural_training"]["members"]})
    states = record["states"]
    if len(states) != 7:
        raise ValueError("Expected exactly the seven archived common optimization states")
    for state in states:
        model, seeds = expected[state["case_hash"]]
        if (state["role"] != "optimization" or state["world_seed"] not in seeds
                or state["decision_model_hash"] != model):
            raise ValueError("Saved observation is from an undeclared world/model")
    initials = [state for state in states if state["state"] == "initial"]
    if len(initials) != 3 or {state["case_hash"] for state in initials} != set(expected):
        raise ValueError("Archived observations omit a source or patient initial state")
    names = record["feature_schema"]
    for profile in declaration["profiles"].values():
        if (profile["action_feature_names"] != names["action_feature_names"]
                or profile["state_feature_names"] != names["state_feature_names"]):
            raise ValueError("Saved feature order differs from the registered policy inputs")
    return record, states


def actor_statistics(policy: Any, observation: Any) -> dict:
    """Use the actual profile transform; all computations are inference-only."""
    import numpy as np
    import torch
    with torch.inference_mode():
        raw = torch.as_tensor(np.asarray(observation.action_features), dtype=torch.float32)
        state = torch.as_tensor(np.asarray(observation.state_features), dtype=torch.float32)
        actor_features = policy.actor_inputs(raw)
        inputs = torch.cat((actor_features, state.expand(len(raw), -1)), dim=1)
        preactivation = policy.actor[0](inputs)
        activation = torch.tanh(preactivation)
        logits, value = policy(observation)
        probabilities = torch.softmax(logits, dim=0)
        mask = torch.as_tensor(np.asarray(observation.action_mask), dtype=torch.bool)
        legal = probabilities[mask]
        rows = []
        for index in torch.where(mask)[0].tolist():
            rows.append({"action_id": observation.action_ids[index], "logit": float(logits[index]),
                "probability": float(probabilities[index]),
                "tanh_abs_ge_099_fraction": float((activation[index].abs() >= .99).float().mean()),
                "mean_tanh_derivative": float((1. - activation[index].square()).mean()),
                "preactivation_abs_max": float(preactivation[index].abs().max())})
        return {"greedy_action_id": observation.action_ids[int(logits.argmax())],
            "entropy_nats": float(-(legal * legal.clamp_min(torch.finfo(legal.dtype).tiny).log()).sum()),
            "maximum_entropy_nats": math.log(len(legal)), "value_prediction": float(value),
            "rows": rows, "actor_input_abs_max_by_feature": actor_features[mask].abs().max(dim=0).values.tolist(),
            "raw_input_abs_max_by_feature": raw[mask].abs().max(dim=0).values.tolist()}


def parameter_changes(initial: Any, latest: Any) -> dict:
    """Actual net tensor displacement, not the unrecorded optimizer path length."""
    import torch
    with torch.inference_mode():
        first, last = dict(initial.named_parameters()), dict(latest.named_parameters())
        return {group + "_net_parameter_displacement_l2": math.sqrt(sum(
            float((last[name] - value).double().square().sum())
            for name, value in first.items() if name.startswith(group + ".")))
            for group in ("actor", "value")}


def clipping_decomposition(training: dict, contract: dict) -> dict:
    """Infer group norms from existing logs, without computing any gradients.

    Valid only for the pinned actor/value-only model and Torch L2 clip with
    epsilon 1e-6. Float32 norm/clip rounding makes this an approximate inference.
    The actor-only coefficient is arithmetic context, not an executed optimizer.
    """
    clip = contract["config"]["max_gradient_norm"]
    result = []
    for update in training["optimization_history"]:
        total = float(update["gradient_norm_before_clip"])
        actor_after = float(update["actor_gradient_norm_after_clip"])
        if not all(math.isfinite(value) and value >= 0 for value in (total, actor_after)):
            raise ValueError("Gradient log contains an invalid norm")
        coefficient = min(1., clip / (total + 1e-6))
        actor_before = actor_after / coefficient
        residual_squared = total * total - actor_before * actor_before
        tolerance = 2e-5 * max(1., total * total)
        if residual_squared < -tolerance:
            raise ValueError("Logged actor and global norms are inconsistent with the pinned clipping rule")
        critic_before = math.sqrt(max(0., residual_squared))
        actor_only_coefficient = min(1., clip / (actor_before + 1e-6))
        result.append({"gradient_steps": update["gradient_steps"], "total_norm_before_clip": total,
            "actor_norm_after_clip_recorded": actor_after, "global_clip_coefficient_inferred": coefficient,
            "actor_norm_before_clip_inferred": actor_before, "critic_norm_before_clip_inferred": critic_before,
            "critic_fraction_of_squared_global_norm_inferred": critic_before**2 / total**2 if total else None,
            "actor_only_same_threshold_coefficient_arithmetic": actor_only_coefficient,
            "additional_actor_shrink_ratio_from_joint_norm_inferred": coefficient / actor_only_coefficient,
            "measured_mean_batch_return": update["mean_return"]})
    return {"rows": result, "threshold": clip, "new_gradients_computed": 0,
        "inference_assumptions": ["Only actor and value parameters contribute to the global L2 norm",
            "Pinned Torch rule min(1, threshold/(global_norm+1e-6))",
            "Actor norm logged immediately after clipping and before Adam",
            "Scalar gradient shrinkage is not a prediction of Adam parameter-update shrinkage"]}


sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from resectionlab.data_policy import historical_only as _historical_only


@_historical_only("GENERATED_POLICY_INELIGIBLE")
def diagnose(run: Path, observations: Path, output: Path) -> dict:
    if output.exists():
        raise FileExistsError("Preserve prior diagnostics; use a new output directory")
    summary, status = completed_records(run)
    snapshot = frozen_runtime(run, summary)
    import numpy as np
    import torch
    from resectionlab.learning import load_policy, policy_hash, trainable_parameter_hash
    from resectionlab.worlds import content_hash

    declaration_path = run / "frozen-source/manifests/experiments/procedural-native-feature-units-v1.json"
    declaration = read_json(declaration_path)
    supplied = declaration["declaration_content_hash"]
    if (content_hash({key: value for key, value in declaration.items() if key != "declaration_content_hash"}) != supplied
            or supplied != summary["declaration_hash"]):
        raise ValueError("Completed declaration bytes differ")
    physical = read_json(run / "frozen-source" / declaration["reference_design"]["path"])
    if (physical["declaration_content_hash"] != declaration["reference_design"]["declaration_content_hash"]
            or content_hash({key: value for key, value in physical.items() if key != "declaration_content_hash"}) != physical["declaration_content_hash"]):
        raise ValueError("Physical reference declaration differs")
    reference, states = saved_observations(observations, declaration, physical)
    # This exact published runtime rule justifies the algebra below; its source
    # bytes and installed Torch version are recorded rather than assumed.
    clip_function = torch.nn.utils.clip_grad._clip_grads_with_norm_
    clip_source = inspect.getsource(clip_function)
    if "clip_coef = max_norm / (total_norm + 1e-6)" not in clip_source:
        raise ValueError("Installed clipping implementation differs from the diagnostic derivation")
    policies, bindings, digests, training_rows = {}, {}, {}, {}
    def retain(label: str, path: Path, expected: str, profile: str, selected: bool = True):
        policy = load_policy(path, selected=selected, expected_input_profile=profile)
        if policy_hash(policy) != expected:
            raise ValueError(f"Checkpoint hash differs: {label}")
        if any(not name.startswith(("actor.", "value.")) for name, _ in policy.named_parameters()):
            raise ValueError("Gradient decomposition requires actor/value-only parameters")
        policy.requires_grad_(False)
        policies[label] = policy
        digests[str(path)] = file_hash(path)
        bindings[label] = {"path": str(path.relative_to(run)), "checkpoint_sha256": digests[str(path)],
            "policy_hash": expected, "trainable_parameter_hash": trainable_parameter_hash(policy),
            "selected_weights": selected, "input_profile_hash": policy.input_profile.fingerprint}
    for row in summary["learning"]:
        phase, profile, seed = row["phase"], row["input_profile"], row["seed"]
        folder = run / "study" / f"{profile.lower()}-{phase}-{seed}"
        detailed, contract = read_json(folder / "result.json"), read_json(folder / "contract.json")
        if contract["runtime"]["torch"] != str(torch.__version__):
            raise ValueError("Installed Torch version differs from the logged clipping implementation")
        if detailed["decision_model_hash"] != physical["target"]["decision_model_hash"]:
            raise ValueError("Learned checkpoint uses another physical planning model")
        for name in ("gradient_steps", "initial_checkpoint_hash", "selected_checkpoint_hash", "latest_checkpoint_hash"):
            if detailed[name] != row[name]:
                raise ValueError("Summary differs from completed per-arm logs")
        label = f"{profile}:{phase}:{seed}"
        training_rows[label] = {**gradient_statistics(detailed), "clipping": clipping_decomposition(detailed, contract),
            "training_record_sha256": file_hash(folder / "result.json"),
            "training_contract_sha256": file_hash(folder / "contract.json")}
        retain(label + ":initial", folder / "initial.pt", detailed["initial_checkpoint_hash"], profile)
        retain(label + ":selected", folder / "checkpoint.pt", detailed["selected_checkpoint_hash"], profile)
        retain(label + ":latest", folder / "checkpoint.pt", detailed["latest_checkpoint_hash"], profile, False)
        training_rows[label]["initial_to_latest"] = parameter_changes(policies[label + ":initial"], policies[label + ":latest"])
        training_rows[label]["initial_to_selected"] = parameter_changes(policies[label + ":initial"], policies[label + ":selected"])
    same_policy_domain_panels = {}
    for profile in PROFILES:
        folder = run / "study" / f"pretraining-{profile.lower()}" / "training"
        detailed, contract = read_json(folder / "result.json"), read_json(folder / "contract.json")
        label = f"{profile}:offline"
        training_rows[label] = {**gradient_statistics(detailed), "clipping": clipping_decomposition(detailed, contract),
            "training_record_sha256": file_hash(folder / "result.json"),
            "training_contract_sha256": file_hash(folder / "contract.json")}
        retain(label + ":initial", folder / "initial.pt", detailed["initial_checkpoint_hash"], profile)
        retain(label + ":latest", folder / "checkpoint.pt", detailed["latest_checkpoint_hash"], profile, False)
        training_rows[label]["initial_to_latest"] = parameter_changes(policies[label + ":initial"], policies[label + ":latest"])
        shared = summary["shared"][profile]
        shared_path = run / "study" / f"{profile.lower()}-procedural-source.pt"
        if file_hash(shared_path) != shared["checkpoint_file_sha256"]:
            raise ValueError("Shared checkpoint bytes differ from completed provenance")
        retain(profile + ":shared", shared_path, shared["policy_hash"], profile)
        frozen = summary["frozen"][profile]
        expected_source_worlds = sum(len(member["world_partitions"]["selection"]["seeds"])
                                     for member in physical["procedural_training"]["members"])
        matches = [panel for panel in detailed["selection_history"]
                   if panel["checkpoint_hash"] == shared["policy_hash"] and panel["world_count"] == expected_source_worlds]
        if (frozen["shared_checkpoint_hash"] != shared["policy_hash"]
                or frozen["selection_panel_complete"] is not True):
            raise ValueError("Frozen patient panel is not complete for the shared policy")
        same_policy_domain_panels[profile] = {"policy_hash": shared["policy_hash"],
            "source_complete_panel_return_same_checkpoint": matches[-1]["mean_return"] if matches else None,
            "source_panel_available": bool(matches), "patient_complete_panel_return_same_checkpoint": frozen["selection_return"],
            "limit": "Deterministic complete selection rollouts, not expected stochastic-training returns; absent exact-checkpoint source panels stay null"}

    evaluated = []
    for saved in states:
        obs = saved["observations"]
        observation = SimpleNamespace(action_features=np.asarray(obs["action_features"], dtype=np.float32),
            state_features=np.asarray(obs["state_features"], dtype=np.float32),
            action_ids=tuple(obs["action_ids"]), action_mask=np.asarray(obs["action_mask"], dtype=bool))
        evaluated.append({key: saved[key] for key in ("domain", "state", "case_hash", "decision_model_hash", "world_seed")}
            | {"raw_observations": obs, "nominal_immediate_rewards": saved["nominal_immediate_rewards"],
               "policies": {label: actor_statistics(policy, observation) for label, policy in policies.items()}})
    initial = [state for state in evaluated if state["state"] == "initial"]
    alias = {"initial_state_features": {state["domain"]: state["raw_observations"]["state_features"] for state in initial},
        "identical_initial_state_features": all(state["raw_observations"]["state_features"] == initial[0]["raw_observations"]["state_features"] for state in initial),
        "per_policy_initial_value_spread": {label: max(state["policies"][label]["value_prediction"] for state in initial)
            - min(state["policies"][label]["value_prediction"] for state in initial) for label in policies},
        "same_policy_saved_domain_panels": same_policy_domain_panels}
    for label, policy in policies.items():
        if any(parameter.grad is not None or parameter.requires_grad for parameter in policy.parameters()):
            raise ValueError("Inference-only diagnostic unexpectedly enabled or computed gradients")
        if policy_hash(policy) != bindings[label]["policy_hash"]:
            raise ValueError("Inference-only diagnostic changed policy state")
    if any(file_hash(Path(path)) != digest for path, digest in digests.items()):
        raise ValueError("Diagnostic changed checkpoint bytes")
    script = Path(__file__).read_bytes()
    report = {"schema_version": 1, "status": "completed_read_only_mechanism_diagnostic",
        "created_at_utc": datetime.now(timezone.utc).isoformat(), "study_run_id": status["run_id"],
        "completed_summary_sha256": file_hash(run / "study/summary.json"),
        "study_declaration_hash": supplied, "source_runtime_hash": snapshot["numerical_runtime_content_hash"],
        "saved_observation_file": str(observations), "saved_observation_file_sha256": file_hash(observations),
        "saved_observation_reference_runtime_hash": reference["source_runtime_hash"],
        "script_sha256": hashlib.sha256(script).hexdigest(),
        "reused_helper_sha256": file_hash(Path(__file__).with_name("diagnose_procedural_transfer.py")),
        "torch_version": str(torch.__version__), "clip_function_source_sha256": hashlib.sha256(clip_source.encode()).hexdigest(),
        "policies": bindings, "training": training_rows, "states": evaluated, "critic_aliasing": alias,
        "new_gradient_steps": 0, "new_backward_passes": 0, "new_simulator_resets": 0,
        "new_environment_steps": 0, "new_search_expansions": 0, "final_worlds_used": False,
        "stress_worlds_used": False, "checkpoints_unchanged": True,
        "limits": ["Post hoc descriptive analysis of one previously studied structural patient model",
            "Seven archived common states do not estimate learned policy occupancy",
            "Tanh derivative and saturation measure local numerical sensitivity, not causal learning failure",
            "Identical critic input proves aliasing capacity; matched deterministic panels only partly characterize value-target error",
            "Batch return logs do not retain per-transition stochastic return targets or actor/value gradient directions",
            "Group-gradient norms inferred with float32 tolerance; Adam parameter steps cannot be inferred from scalar clipping ratios",
            "No new model, feature, loss, clipping, action proposal, checkpoint selection or experiment setting chosen"]}
    output.mkdir(parents=True)
    (output / "diagnostic-script.py").write_bytes(script)
    (output / "diagnostic.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--saved-observations", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = diagnose(args.run.resolve(), args.saved_observations.resolve(), args.output.resolve())
    print(json.dumps({key: report[key] for key in ("status", "new_environment_steps", "new_backward_passes")}, indent=2))


if __name__ == "__main__":
    main()
