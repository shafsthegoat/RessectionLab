#!/usr/bin/env python3
"""Two fresh scratch axis profiles. Declaration-only by default; no implicit retries.

Only successful worker AND launcher receipts authorize the frozen candidate.
Decision evidence comes from the learner's actual forward calls, never a replay.
"""
from __future__ import annotations

import argparse
import copy
from dataclasses import asdict
import hashlib
import json
import math
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))
import preflight_native_axis as physical
import run_native_axis_pilot as baseline
from resectionlab.worlds import content_hash

DECLARATION_PATH = Path("manifests/experiments/native-axis-scratch-feature-units-v1.json")
DECLARATION_HASH = "sha256:9c0f5f51574ac1347749ff87a377367785e6df8bde01b654bde6616afe35695e"
write_json = physical.write_json
file_hash = baseline.file_hash
_require, _finite = baseline._require, baseline._finite
validate_audits = baseline.validate_audits


def load_declaration(root: Path = ROOT) -> dict:
    declaration = physical._checked_manifest(root / DECLARATION_PATH, DECLARATION_HASH)
    old = baseline.load_declaration(root)
    for item in declaration["references"].values():
        _require(file_hash(root / item["path"]) == item["file_sha256"], "Declared immutable reference changed")
        if "content_hash" in item:
            physical._checked_manifest(root / item["path"], item["content_hash"])
    fixed = declaration["fixed_protocol"]
    for field in ("training_config", "world_partitions", "optimization_episode_seeds"):
        _require(fixed[field] == old[field], "Paired physical/budget/world protocol drift")
    _require(fixed["resource_budget_per_arm"] == old["resource_budget"], "Paired worker budget drift")
    excluded = {"initial_policy_hash", "input_profile", "input_profile_manifest", "input_profile_hash"}
    _require(content_hash({k:v for k,v in old["frozen_model"].items() if k not in excluded}) ==
             fixed["physical_frozen_model_hash"], "Paired native physical model drift")
    from resectionlab.policy_inputs import policy_input_profile
    for name in declaration["execution_order"]:
        record = declaration["profiles"][name]
        profile = policy_input_profile(name)
        _require(record["input_profile_manifest"] == profile.to_dict()
                 and record["input_profile_hash"] == profile.fingerprint, "Paired profile registry changed")
    return declaration


def arm_declaration(declaration: dict, profile: str, root: Path = ROOT) -> dict:
    """Materialize the explicitly referenced six-episode protocol for one profile.

    This does not relabel historical results. Both arms run from fresh scratch.
    Private constructed tests may replace patient identities afterward.
    """
    _require(profile in declaration["execution_order"], "Unknown paired profile")
    original = baseline.load_declaration(root)
    arm = {key: copy.deepcopy(original[key]) for key in ("frozen_model", "training_config",
        "world_partitions", "optimization_episode_seeds", "resource_budget")}
    arm["pilot_id"] = declaration["study_id"] + ":" + profile
    arm["declaration_content_hash"] = declaration["declaration_content_hash"]
    arm["baseline_protocol_hash"] = declaration["references"]["baseline_protocol"]["content_hash"]
    arm["frozen_model"].update(declaration["profiles"][profile], input_profile=profile,
        initial_trainable_parameter_hash=declaration["paired_initialization"]["expected_trainable_parameter_hash"])
    arm["frozen_model"]["observation_encoding"] = "RAW"
    arm["frozen_model"]["cache_enabled"] = False
    arm["frozen_model"].pop("no_model_or_input_changes")
    arm["frozen_model"]["physical_model_unchanged"] = True
    arm["frozen_model"]["legacy_model_input_profile_scope"] = "RAW simulator encoding; policy transform bound separately"
    return arm


def source_snapshot(root: Path = ROOT) -> dict:
    snapshot = baseline.source_snapshot(root)
    files = snapshot["file_sha256"]
    for name in (str(DECLARATION_PATH), "scripts/run_native_axis_feature_units.py"):
        files[name] = file_hash(root / name)
    snapshot["runtime_content_hash"] = content_hash({"files": files,
        "versions": snapshot["runtime_versions"], "python": snapshot["python"]})
    return snapshot


def assert_source(snapshot: dict, root: Path = ROOT) -> None:
    _require(source_snapshot(root)["runtime_content_hash"] == snapshot["runtime_content_hash"],
             "SOURCE_CHANGED_DURING_AXIS_PAIR")


def validate_contract(contract: dict, declaration: dict) -> None:
    """Require this exact scratch learner contract, including physical units."""
    fields = ("schema_version", "algorithm", "input_profile", "input_profile_hash", "implementation_sha256",
        "simulator_implementation_sha256", "numerical_source_sha256", "runtime", "config", "partitions",
        "population_initialization", "procedural_initialization", "timing_contract", "elapsed_seconds_scope",
        "decision_model_hash", "dimensions", "axis_observation_contract")
    metadata = {"contract_hash", "initial_checkpoint_hash", "shared_checkpoint_hash", "code_sha256",
        "hardware", "clinical_deficit_probability", "final_evaluation_used_for_optimization"}
    _require(set(contract) == set(fields) | metadata, "Unknown or missing learner contract fields")
    _require(type(contract["schema_version"]) is int and contract["schema_version"] == 1
             and contract["algorithm"] == "masked_reinforce_state_value_v2"
             and contract["dimensions"] == [15, 6, 16]
             and contract["timing_contract"] == "optimization_selection_budget_v2_initialization_separate"
             and contract["elapsed_seconds_scope"] == "cumulative optimization and selection, including initial selection; initialization excluded"
             and contract["clinical_deficit_probability"] is None
             and contract["final_evaluation_used_for_optimization"] is False,
             "Unsupported scratch learner contract")
    _require(content_hash({key: contract[key] for key in fields}) == contract["contract_hash"],
             "Learner contract checksum differs")
    model = declaration["frozen_model"]
    schema = {
        "version": "native-axis-raw-15x6-observation-v1",
        "backend": "experimental-native-axis-column-policy-v1",
        "observation_encoding": "RAW",
        "legacy_model_input_profile_scope": "RAW simulator feature encoding; policy transform bound separately",
        "action_feature_names": model["input_profile_manifest"]["action_feature_names"],
        "action_feature_units": ["binary", "mm3", "mm3", "spatial_surrogate_mm3", "spatial_surrogate_mm3",
            "mm", "mm", "mm", "binary", "fraction_of_action_budget", "fraction", "fraction", "mm3",
            "spatial_surrogate_mm3", "spatial_surrogate_mm3"],
        "state_feature_names": model["input_profile_manifest"]["state_feature_names"],
        "state_feature_units": ["fraction", "fraction", "fraction", "fraction", "binary", "binary"],
        "depth_semantics": "completed nonSTOP actions / max_steps; not physical depth",
        "adjacent_target_semantics": "supplied target-label occupancy in clipped 3x3x3 endpoint neighborhood; not residual-only",
        "partial_contact_semantics": "new retained contact excluding prior charged contact and cells removed now",
        "actor_evidence": "nominal fields and actual cavity only; hidden world excluded",
        "removal_version": "contained-native-cell-connected-suction-v2",
        "decision_model_hash": model["decision_model_hash"],
        "proposal_model_hash": model["proposal_model_hash"],
        "reward": model["physical_reward"], "partial_contact_weight": model["partial_contact_weight"],
        "max_steps": model["action_model"]["max_cuts"],
        "max_actions_including_stop": model["action_model"]["max_actions_including_stop"],
        "scope": "scratch REINFORCE compatibility only; no transfer or resume authorization",
    }
    schema["contract_hash"] = content_hash(schema)
    _require(content_hash(contract["axis_observation_contract"]) == content_hash(schema),
             "Axis observation contract semantics differ")


def initial_receipt(folder: Path, declaration: dict) -> dict:
    """Authenticate the actual fresh initializer without a forward or RNG draw."""
    import torch
    from resectionlab.learning import checkpoint_input_profile, policy_hash
    from resectionlab.native_axis_policy_schema import AXIS_OBSERVATION_SCHEMA_VERSION
    expected = declaration["frozen_model"]
    initial = torch.load(folder / "initial.pt", map_location="cpu", weights_only=True)
    contract = json.loads((folder / "contract.json").read_text())
    validate_contract(contract, declaration)
    profile = checkpoint_input_profile(initial, expected_input_profile=expected["input_profile"])
    _require(profile.to_dict() == expected["input_profile_manifest"]
             and profile.fingerprint == expected["input_profile_hash"], "Initial profile differs")
    _require(initial["dimensions"] == [15, 6, 16], "Initial dimensions differ")
    weights = initial["policy"]
    parameter_names = {f"{network}.{layer}.{kind}" for network in ("actor", "value")
        for layer in (0, 2) for kind in ("weight", "bias")}
    _require(set(weights) == parameter_names | ({"action_divisors"} if profile.profile_id == "FEATURE_UNITS" else set()),
             "Initial tensor inventory differs")
    trainable_hash = policy_hash({name: weights[name] for name in parameter_names})
    _require(trainable_hash == initial["trainable_parameter_hash"] == expected["initial_trainable_parameter_hash"],
             "Paired initial trainable tensors differ")
    _require(policy_hash(weights) == initial["policy_hash"] == expected["initial_policy_hash"],
             "Fresh initialized behavior differs")
    _require(contract["config"] == declaration["training_config"]
             and contract["decision_model_hash"] == expected["decision_model_hash"]
             and contract["initial_checkpoint_hash"] == initial["policy_hash"]
             and contract["population_initialization"] is None and contract["procedural_initialization"] is None
             and contract["shared_checkpoint_hash"] is None, "Fresh scratch contract differs")
    _require(contract["input_profile"] == profile.to_dict() and contract["input_profile_hash"] == profile.fingerprint,
             "Contract profile differs")
    # Authenticate the learner's exact serialized role/case/generator/seed vector.
    from resectionlab.learning import _partition_record
    for role in ("optimization", "selection"):
        item = contract["partitions"][role]
        expected_panel = _partition_record(physical._partition(declaration["world_partitions"][role]), role)
        _require(content_hash(item) == content_hash(expected_panel),
                 "Fresh scratch world vectors differ")
    schema = contract["axis_observation_contract"]
    _require(schema["version"] == AXIS_OBSERVATION_SCHEMA_VERSION and schema["observation_encoding"] == "RAW"
             and schema["decision_model_hash"] == expected["decision_model_hash"]
             and schema["proposal_model_hash"] == expected["proposal_model_hash"]
             and schema["max_steps"] == 3 and schema["max_actions_including_stop"] <= 27
             and schema["reward"] == expected["physical_reward"]
             and schema["partial_contact_weight"] == expected["partial_contact_weight"]
             and schema["action_feature_names"] == profile.to_dict()["action_feature_names"]
             and schema["state_feature_names"] == profile.to_dict()["state_feature_names"], "Axis observation schema differs")
    _require(schema["contract_hash"] == content_hash({k:v for k,v in schema.items() if k != "contract_hash"}),
             "Axis schema fingerprint differs")
    return {"initial_policy_hash": initial["policy_hash"], "initial_trainable_parameter_hash": trainable_hash,
        "input_profile_hash": profile.fingerprint, "axis_observation_contract": schema,
        "initial_file_sha256": file_hash(folder / "initial.pt"), "contract_file_sha256": file_hash(folder / "contract.json")}


def checkpoint_receipt(folder: Path, declaration: dict) -> dict:
    """Authenticate actual saved weights/Adam state without another policy forward."""
    import torch
    from resectionlab.learning import checkpoint_input_profile, policy_hash
    initial = torch.load(folder / "initial.pt", map_location="cpu", weights_only=True)
    latest = torch.load(folder / "checkpoint.pt", map_location="cpu", weights_only=True)
    contract = json.loads((folder / "contract.json").read_text())
    result = json.loads((folder / "result.json").read_text())
    expected = declaration["frozen_model"]
    initialization = initial_receipt(folder, declaration)
    _require(contract["config"] == declaration["training_config"]
             and contract["decision_model_hash"] == expected["decision_model_hash"]
             and contract["initial_checkpoint_hash"] == expected["initial_policy_hash"]
             and contract["contract_hash"] == latest["contract_hash"]
             and contract["population_initialization"] is None
             and contract["procedural_initialization"] is None,
             "Learner contract differs from declared scratch update")
    for key, value in latest["state"].items():
        _require(content_hash(value) == content_hash(result[key]), "Saved checkpoint state/result mismatch: " + key)
    for payload in (initial, latest):
        profile = checkpoint_input_profile(payload, expected_input_profile=expected["input_profile"])
        _require(profile.to_dict() == expected["input_profile_manifest"]
                 and profile.fingerprint == expected["input_profile_hash"], "Checkpoint profile drift")
        _require(payload["dimensions"] == [15, 6, 16], "Checkpoint dimensions drift")
        _require(policy_hash(payload["policy"]) == payload["policy_hash"], "Checkpoint tensor hash differs")
        trainable = policy_hash({key: value for key, value in payload["policy"].items() if key != "action_divisors"})
        _require(trainable == payload["trainable_parameter_hash"], "Checkpoint trainable tensor hash differs")
    _require(trainable == result["latest_trainable_parameter_hash"], "Latest trainable result differs")
    _require(initial["policy_hash"] == expected["initial_policy_hash"], "Initial weights differ from declaration")
    _require(latest["initial_hash"] == initial["policy_hash"], "Latest checkpoint has another initializer")
    _require(policy_hash(latest["selected_policy"]) == latest["selected_hash"], "Selected tensor hash differs")
    actor_hash = lambda weights: policy_hash({key.removeprefix("actor."): value
        for key, value in weights.items() if key.startswith("actor.")})
    optimizer = latest["optimizer"]
    names = [f"{network}.{layer}.{kind}" for network in ("actor", "value")
             for layer in (0, 2) for kind in ("weight", "bias")]
    _require(set(optimizer) == {"state", "param_groups"} and len(optimizer["param_groups"]) == 1
             and optimizer["param_groups"][0]["params"] == list(range(len(names)))
             and set(optimizer["state"]) == set(range(len(names))), "Incomplete Adam parameter/state inventory")
    group = optimizer["param_groups"][0]
    _require(group["lr"] == declaration["training_config"]["learning_rate"]
             and tuple(group["betas"]) == (.9, .999) and group["eps"] == 1e-8
             and group["weight_decay"] == 0 and group["amsgrad"] is False
             and group["maximize"] is False, "Adam settings differ")
    steps = []
    for index, name in enumerate(names):
        state, parameter = optimizer["state"][index], latest["policy"][name]
        _require(set(state) == {"step", "exp_avg", "exp_avg_sq"}
                 and isinstance(state["step"], torch.Tensor) and state["step"].shape == ()
                 and torch.isfinite(state["step"]).all().item() and float(state["step"]) == 1.,
                 "Expected one actual Adam step for every parameter")
        for key in ("exp_avg", "exp_avg_sq"):
            moment = state[key]
            _require(isinstance(moment, torch.Tensor) and moment.shape == parameter.shape
                     and moment.dtype == parameter.dtype and torch.isfinite(moment).all().item()
                     and (key != "exp_avg_sq" or (moment >= 0).all().item()), "Invalid Adam moment tensor")
        steps.append(float(state["step"]))
    return {"initial_policy_hash": initial["policy_hash"], "latest_policy_hash": latest["policy_hash"],
        "selected_policy_hash": latest["selected_hash"], "initial_actor_hash": actor_hash(initial["policy"]),
        "latest_actor_hash": actor_hash(latest["policy"]), "adam_steps": steps,
        "files": {name: file_hash(folder / name) for name in ("initial.pt", "checkpoint.pt", "contract.json", "result.json")},
        "input_profile_hash": expected["input_profile_hash"],
        "checkpoint_state_hash": content_hash(latest["state"]),
        "initialization": initialization}


def _validate_decision(payload: dict, declaration: dict) -> None:
    """Check saved primitive evidence; do not re-run an actor or read a simulator."""
    import numpy as np
    _require(payload["version"] == "learner-decision-observer-v1"
             and type(payload["forward_evaluated"]) is bool, "Unknown decision observer or forward flag")
    inputs = payload["inputs"]
    ids = inputs["action_ids"]
    mask = np.asarray(inputs["action_mask"]["values"])
    _require(ids and ids[0] == "STOP" and len(ids) == len(set(ids)) and len(ids) <= 27,
             "Invalid certified action inventory")
    _require(mask.dtype == np.bool_ and np.dtype(inputs["action_mask"]["dtype"]) == np.dtype(bool)
             and tuple(inputs["action_mask"]["shape"]) == mask.shape == (len(ids),) and mask.all(), "Invalid axis mask")
    for key, shape in (("source_action_features", (len(ids), 15)), ("source_state_features", (6,))):
        array = inputs[key]
        values = np.asarray(array["values"])
        _require(np.dtype(array["dtype"]) == np.dtype("float32")
                 and tuple(array["shape"]) == shape and values.shape == shape and np.isfinite(values).all(),
                 "Invalid saved observation: " + key)
    selected = payload["selected_index"]
    _require(type(selected) is int and 0 <= selected < len(ids)
             and ids[selected] == payload["selected_action_id"], "Decision selected ID differs")
    if payload["forward_evaluated"]:
        logits = np.asarray([float(value) for value in payload["logits"]["values"]])
        _require(np.dtype(payload["logits"]["dtype"]) == np.dtype("float32")
                 and tuple(payload["logits"]["shape"]) == (len(ids),) and logits.shape == (len(ids),)
                 and np.isfinite(logits).all() and _finite(payload["value"]), "Nonfinite or incomplete actual forward")
        for key, source_key in (("action_features", "source_action_features"),
                                ("actor_action_features", "source_action_features"),
                                ("state_features", "source_state_features")):
            record = inputs[key]
            source = np.asarray(inputs[source_key]["values"], dtype=np.float32)
            if key == "actor_action_features":
                source = source / np.asarray(declaration["frozen_model"]["input_profile_manifest"]["action_divisors"], dtype=np.float32)
            _require(record is not None and np.dtype(record["dtype"]) == np.dtype("float32")
                     and tuple(record["shape"]) == source.shape
                     and np.array_equal(np.asarray(record["values"], dtype=np.float32), source),
                     "Profile actual policy input differs: " + key)
        if payload["role"] == "selection":
            _require(selected == int(np.argmax(logits)) and payload["decision_rule"] == "deterministic_argmax"
                     and payload["forced_reason"] is None, "Selection is not deterministic argmax")
        else:
            _require(payload["role"] == "optimization" and payload["decision_rule"] == "sampled_categorical"
                     and payload["forced_reason"] is None, "Pilot optimization must use its actual categorical sample")
    else:
        _require(payload["role"] == "selection" and payload["decision_rule"] == "forced_stop"
                 and payload["step"] == declaration["training_config"]["max_episode_steps"] - 1
                 and payload["selected_action_id"] == "STOP" and payload["forced_reason"] == "episode_step_limit"
                 and payload["logits"] is None and payload["value"] is None
                 and all(inputs[key] is None for key in ("action_features", "state_features", "actor_action_features")),
                 "Unevaluated decision must be an explicit forced STOP")


def validate_completion(declaration: dict, result: dict, accounting: dict, checkpoints: dict) -> dict:
    """Pure fail-closed gate, reusable for independent tamper tests.

    This returns diagnostic frozen histories, not publication authority. The
    public entry point separately pins the declaration and executable source.
    """
    from resectionlab.native_axis_accounting import _hash
    frozen = declaration["frozen_model"]
    _require(accounting["receipt_hash"] == _hash({k: v for k, v in accounting.items() if k != "receipt_hash"}),
             "Accounting receipt hash mismatch")
    _require(accounting["status"] == "learner_returned" and accounting["counts_complete"]
             and accounting["failure"] is None and accounting["receipt_export_failure"] is None
             and accounting["decision_recording_enabled"] and accounting["candidate_eligible"] is False,
             "Accounting did not complete")
    _require(accounting["decision_model_hash"] == frozen["decision_model_hash"]
             and accounting["case_hash"] == frozen["case_hash"]
             and accounting["input_profile"] == frozen["input_profile"]
             and accounting["input_profile_hash"] == frozen["input_profile_hash"]
             and accounting["input_profile_manifest"] == frozen["input_profile_manifest"]
             and accounting["observation_encoding"] == "RAW"
             and accounting["version"] == "experimental-native-axis-training-accounting-v2", "Accounting source/model/profile drift")
    _require(content_hash(accounting["partitions"]) == content_hash(declaration["world_partitions"]), "Accounting worlds differ")
    _require(result["status"] == "gradient_budget" and result["gradient_steps"] == 1
             and result["completed_episodes"] == 2 and result["discarded_partial_batches"] == 0,
             "One complete update and two optimization episodes required")
    for key in accounting["learner_result"]:
        _require(accounting["learner_result"][key] == result[key], "Generic/accounting result mismatch: " + key)
    _require(result["decision_model_hash"] == frozen["decision_model_hash"]
             and result["initial_checkpoint_hash"] == frozen["initial_policy_hash"], "Learner model/initializer differs")
    _require(result["input_profile_hash"] == frozen["input_profile_hash"], "Learner input profile differs")
    for result_key, checkpoint_key in (("initial_checkpoint_hash", "initial_policy_hash"),
            ("latest_checkpoint_hash", "latest_policy_hash"), ("selected_checkpoint_hash", "selected_policy_hash"),
            ("initial_actor_hash", "initial_actor_hash"), ("latest_actor_hash", "latest_actor_hash")):
        _require(result[result_key] == checkpoints[checkpoint_key], "Saved checkpoint/result mismatch")
    _require(result["actor_parameters_changed"] is True
             and checkpoints["initial_actor_hash"] != checkpoints["latest_actor_hash"], "No actual actor update")
    updates = result["optimization_history"]
    _require(len(updates) == 1 and updates[0]["gradient_steps"] == 1
             and len(updates[0]["episode_source_case_hashes"]) == 2
             and set(updates[0]["episode_source_case_hashes"]) == {frozen["case_hash"]}
             and all(_finite(updates[0][key]) for key in ("loss", "gradient_norm_before_clip", "actor_gradient_norm_after_clip"))
             and updates[0]["actor_gradient_norm_after_clip"] > 0, "No finite actor gradient from two episodes")
    panels = result["selection_history"]
    _require(len(panels) == 2 and [row["gradient_steps"] for row in panels] == [0, 1]
             and all(row["world_count"] == 2 and _finite(row["mean_return"]) for row in panels),
             "Both full selection panels required")
    _require([row["checkpoint_hash"] for row in panels] ==
             [checkpoints["initial_policy_hash"], checkpoints["latest_policy_hash"]], "Selection checkpoint mismatch")
    selected_panel = max(range(2), key=lambda index: panels[index]["mean_return"])
    _require(result["selected_checkpoint_hash"] == panels[selected_panel]["checkpoint_hash"]
             and result["selected_selection_return"] == panels[selected_panel]["mean_return"]
             and result["initial_selection_return"] == panels[0]["mean_return"], "Earliest-best selection differs")
    episodes, events = accounting["episodes"], accounting["events"]
    _require(len(episodes) == 7 and [row["episode"] for row in episodes] == list(range(7)), "Seven reset records required")
    _require(len({row["instance"] for row in episodes}) == 7
             and len([e for e in events if e["kind"] == "factory"]) == 7, "Fresh factory per reset required")
    _require(all(e["kind"] in ("factory", "decision", "transition") for e in events), "Failed or unknown accounting event")
    probe = episodes[0]
    _require(probe["role"] == "optimization" and probe["seed"] == declaration["optimization_episode_seeds"][0]
             and probe["status"] == "unfinished_at_learner_return"
             and not any(e.get("episode") == 0 for e in events), "Shape probe must have zero decisions/transitions")
    selection_seeds = declaration["world_partitions"]["selection"]["seeds"]
    expected = [("selection", seed, 0, 0, i) for i, seed in enumerate(selection_seeds)]
    expected += [("optimization", seed, 0, None, i) for i, seed in enumerate(declaration["optimization_episode_seeds"])]
    expected += [("selection", seed, 1, 1, i) for i, seed in enumerate(selection_seeds)]
    decisions = [e for e in events if e["kind"] == "decision"]
    transitions = [e for e in events if e["kind"] == "transition"]
    _require([e["decision_id"] for e in decisions] == [f"axis-decision-{i:06d}" for i in range(len(decisions))],
             "Decision IDs are not the actual monotonic sequence")
    _require(len(decisions) == len(transitions) <= 18
             and len({e["decision_id"] for e in decisions}) == len(decisions)
             and {e["decision_id"] for e in decisions} == {e["decision_id"] for e in transitions}, "Decision/outcome cardinality mismatch")
    _require(all(e["episode"] in range(1, 7) for e in decisions + transitions), "Unassigned decision or transition")
    completed = []
    for episode, (role, seed, update, panel, ordinal) in zip(episodes[1:], expected):
        identifier = episode["episode"]
        _require(episode["status"] == "complete" and episode["role"] == role and episode["seed"] == seed,
                 "Episode role/order/completion differs")
        ds = [e for e in decisions if e["episode"] == identifier]
        ts = [e for e in transitions if e["episode"] == identifier]
        _require(0 < len(ds) == len(ts) <= 3 and ts[-1]["terminated"]
                 and not any(t["terminated"] for t in ts[:-1]), "Incomplete or overlong episode")
        bound_decisions = []
        for step, (decision, transition) in enumerate(zip(ds, ts)):
            payload = decision["payload"]
            _require(decision["status"] == "step_returned" and decision["decision_id"] == transition["decision_id"]
                     and events.index(decision) < events.index(transition)
                     and transition["returned_to_learner"] and transition["next_observation_complete"]
                     and transition["executed_transition"], "Decision not followed by its returned outcome")
            for key, value in (("instance", episode["instance"]), ("episode", identifier), ("role", role), ("seed", seed)):
                _require(decision[key] == transition[key] == value, "Decision/transition episode join differs")
            _require((payload["role"], payload["seed"], payload["update"], payload["panel"], payload["episode"], payload["step"])
                     == (role, seed, update, panel, ordinal, step), "Learner phase/decision order differs")
            _require(payload["selected_action_id"] == transition["action_id"] and _finite(transition["reward"]),
                     "Chosen action/outcome differs")
            _require(decision["policy_input_profile_hash"] == frozen["input_profile_hash"]
                     and decision["observation_encoding"] == "RAW", "Decision profile join differs")
            _validate_decision(payload, declaration)
            bound_decisions.append({"decision_id": decision["decision_id"], "policy_hash":
                checkpoints["initial_policy_hash" if update == 0 else "latest_policy_hash"],
                "model_hash": frozen["decision_model_hash"], "profile_hash": frozen["input_profile_hash"],
                "phase": "optimization" if role == "optimization" else "initial_selection" if update == 0 else "updated_selection",
                "payload_hash": content_hash(payload)})
        history = [t["info"] for t in ts if t["native_commit"]]
        total = sum(t["reward"] for t in ts)
        completed.append({"episode": identifier, "role": role, "seed": seed, "update": update, "panel": panel,
            "return": total, "transition_count": len(ts), "native_history": history,
            "history_hash": content_hash(history), "decisions": bound_decisions,
            "actions": [t["action_id"] for t in ts]})
    for role, key, maximum in (("optimization", "optimization_environment_steps", 6), ("selection", "selection_environment_steps", 12)):
        totals = accounting["totals"][role]
        actual = [t for t in transitions if t["role"] == role]
        _require(totals["counts_complete"] and totals["committed_but_unreturned"] == 0
                 and totals["executed_transitions"] == totals["returned_transitions"] == result[key] == len(actual) <= maximum
                 and totals["native_commits"] == sum(t["native_commit"] for t in actual)
                 and math.isclose(totals["executed_reward"], sum(t["reward"] for t in actual), abs_tol=1e-8),
                 "Generic and authenticated transition counts disagree")
    for panel, rows in ((0, completed[:2]), (1, completed[4:])):
        _require(math.isclose(sum(row["return"] for row in rows) / 2, panels[panel]["mean_return"], abs_tol=1e-8),
                 "Full-panel return differs from actual transitions")
    _require(math.isclose(sum(row["return"] for row in completed[2:4]) / 2, updates[0]["mean_return"], abs_tol=1e-8),
             "Optimization return differs from actual transitions")
    return {"case_hash": frozen["case_hash"], "decision_model_hash": frozen["decision_model_hash"],
        "episodes": completed, "shape_probe": probe, "selected_panel": selected_panel,
        "selected_checkpoint_hash": result["selected_checkpoint_hash"], "selection_history": panels,
        "accounting_receipt_hash": accounting["receipt_hash"], "checkpoints": checkpoints,
        "candidate_eligible": False, "final_worlds_used": False, "stress_worlds_used": False}


def run_pilot(case, base, optimization, selection, declaration: dict, output: Path, guard,
              *, source_check, trainer=None, auditor=None) -> dict:
    """Private fixture-friendly orchestration; public_worker pins the declaration."""
    from resectionlab.learning import TrainingConfig
    from resectionlab.native_axis_accounting import AxisTrainingAccounting, train_axis_policy
    trainer = trainer or train_axis_policy
    auditor = auditor or physical.independent_check_native_history
    output.mkdir(parents=True, exist_ok=False)
    status = {"status": "running", "stage": "learning", "eligible_candidate_count": 0,
              "expected_complete_episodes": 6, "final_worlds_used": False, "stress_worlds_used": False}
    write_json(output / "status.json", status)
    started = time.perf_counter()
    factory_seconds = []
    accounting = None
    verified_initializer = None
    try:
        guard.require(); source_check()
        from resectionlab.native_axis_simulation import AxisColumnNativeSimulator
        from resectionlab.policy_inputs import policy_input_profile
        frozen_model = declaration["frozen_model"]
        _require(type(base) is AxisColumnNativeSimulator
                 and base.case_hash == frozen_model["case_hash"]
                 and base.decision_model_hash == frozen_model["decision_model_hash"]
                 and base._proposer.model_hash == frozen_model["proposal_model_hash"]
                 and base.native_config.fingerprint == frozen_model["native_config_hash"],
                 "Axis physical model differs before optimizer")
        profile = policy_input_profile(frozen_model["input_profile"])
        _require(profile.to_dict() == frozen_model["input_profile_manifest"]
                 and profile.fingerprint == frozen_model["input_profile_hash"],
                 "Axis profile differs before optimizer")
        _require(content_hash({"optimization": optimization.to_dict(), "selection": selection.to_dict()}) ==
                 content_hash(declaration["world_partitions"]), "Actual factory world vectors differ before optimizer")
        def factory():
            nonlocal verified_initializer
            folder = output / "learner"
            if factory_seconds and verified_initializer is None:
                verified_initializer = initial_receipt(folder, declaration)
                write_json(output / "pre-gradient-initializer.json", verified_initializer)
            before = time.perf_counter()
            instance = base.clone()
            factory_seconds.append(time.perf_counter() - before)
            return instance
        accounting = AxisTrainingAccounting(factory, optimization, selection,
            receipt_path=output / "accounting.json", expected_model_hash=base.decision_model_hash,
            input_profile=declaration["frozen_model"]["input_profile"])
        call = time.perf_counter()
        trainer(accounting, config=TrainingConfig(**declaration["training_config"]), output_dir=output / "learner",
                cancelled=guard, record_decisions=True)
        full_training_call_seconds = time.perf_counter() - call
        guard.require(); source_check()
        result = json.loads((output / "learner/result.json").read_text())
        journal = json.loads((output / "accounting.json").read_text())
        checkpoints = checkpoint_receipt(output / "learner", declaration)
        _require(verified_initializer == checkpoints["initialization"], "Initializer was not authenticated before first scored episode")
        completion = validate_completion(declaration, result, journal, checkpoints)
        import numpy as np
        completion["source_voxel_volume_mm3"] = float(abs(np.linalg.det(base.native_config.affine[:3, :3])))
        write_json(output / "history-freeze.json", completion)
        history_freeze_sha = file_hash(output / "history-freeze.json")
        status["stage"] = "independent_native_audit"
        write_json(output / "status.json", status)
        audits, unique = [], {}
        validation_started = time.perf_counter()
        for episode in completion["episodes"]:
            guard.require(); source_check()
            key = content_hash({"case": base.case_hash, "model": base.decision_model_hash, "history": episode["history_hash"]})
            reused = key in unique
            if not reused:
                audit_started = time.perf_counter()
                cfg = base.native_config
                audit = auditor(case, cfg.tools, episode["native_history"], tissue_mask=cfg.tissue_mask,
                    access=cfg.access, hard_exclusion=cfg.hard_exclusion, cancelled=guard)
                unique[key] = {"audit": audit.to_dict(), "seconds": time.perf_counter() - audit_started}
            audits.append({"episode": episode["episode"], "history_hash": episode["history_hash"],
                "audit_key": key, "reused": reused, **unique[key],
                "actual_check_seconds": 0. if reused else unique[key]["seconds"],
                "seconds_scope": "seconds describes the shared unique certificate execution; actual_check_seconds is zero on reuse"})
            write_json(output / "native-audits.json", audits)
        validate_audits(completion, audits)
        audit_seconds = time.perf_counter() - validation_started
        guard.require(); source_check()
        _require(file_hash(output / "history-freeze.json") == history_freeze_sha, "Frozen histories changed")
        _require(checkpoint_receipt(output / "learner", declaration) == checkpoints, "Checkpoints changed during validation")
        _require(json.loads((output / "accounting.json").read_text()) == journal, "Accounting changed during validation")
        record = {"completion": completion, "audits": audits, "declaration_hash": declaration["declaration_content_hash"],
            "timing": {"full_training_call_seconds": full_training_call_seconds,
                "learner_initialization_seconds": result["initialization_seconds"],
                "optimization_selection_seconds": result["elapsed_seconds"],
                "selection_seconds": result["selection_seconds"],
                "final_checkpoint_export_seconds": result["final_checkpoint_export_seconds"],
                "factory_clone_seconds": factory_seconds,
                "factory_scope": "First clone belongs to learner initialization; subsequent six clones are inside online budget. Nested timings are not additive.",
                "independent_validation_seconds": audit_seconds,
                "independent_validation_seconds_scope": "source guards, native checks and per-episode receipt exports; nested check times are not additive",
                "full_pilot_seconds_before_publication": time.perf_counter() - started},
            "consumer_authority": "This is a diagnostic payload; both worker-status and launcher-status must be completed and bind its exact bytes."}
        write_json(output / "candidate-record.json", record)
        guard.require(); source_check()
        _require(file_hash(output / "history-freeze.json") == history_freeze_sha, "Frozen histories changed during publication")
        _require(checkpoint_receipt(output / "learner", declaration) == checkpoints, "Checkpoints changed during publication")
        _require(json.loads((output / "accounting.json").read_text()) == journal, "Accounting changed during publication")
        _require(json.loads((output / "native-audits.json").read_text()) == json.loads(json.dumps(audits)),
                 "Native audit records changed during publication")
        _require(json.loads((output / "candidate-record.json").read_text()) == json.loads(json.dumps(record)),
                 "Candidate payload changed during publication")
        guard.require(); source_check()
        status.update(status="completed", stage="complete", eligible_candidate_count=1,
            completed_episode_audit_receipts=6, gradient_steps=1, unique_native_audits=len(unique),
            candidate_record_sha256=file_hash(output / "candidate-record.json"), resource=guard.receipt())
        write_json(output / "status.json", status)
        return status
    except Exception as error:
        status.update(status="failed", eligible_candidate_count=0, error_type=type(error).__name__, error=str(error),
            resource=guard.receipt(), full_pilot_seconds=time.perf_counter() - started,
            factory_clone_seconds=factory_seconds)
        write_json(output / "status.json", status)
        if accounting is not None:
            try:
                write_json(output / "failure-accounting.json", accounting.snapshot())
            except Exception as receipt_error:
                error.add_note("Could not export supplemental accounting: " + str(receipt_error))
        raise


def public_worker(output: Path, case_bundle: Path, profile: str) -> None:
    # A directly invoked worker cannot silently retry or overwrite a prior run.
    with (output / "worker-start.json").open("x") as marker:
        json.dump({"pid": os.getpid(), "resume_supported": False}, marker)
    # Establish a receipt even if a declared input fails before patient loading.
    guard = physical.ResourceGuard(physical.PreflightBudget())
    signal.signal(signal.SIGTERM, lambda *_: guard.cancel("parent_cancellation"))
    status = {"status": "running", "stage": "preparation", "eligible_candidate_count": 0,
              "final_worlds_used": False, "stress_worlds_used": False}
    started = time.perf_counter()
    try:
        paired = load_declaration()
        declaration = arm_declaration(paired, profile)
        _require(asdict(guard.budget) == declaration["resource_budget"], "Worker resource contract differs")
        snapshot = source_snapshot()
        _require(json.loads((output / "launch-source.json").read_text())["runtime_content_hash"] == snapshot["runtime_content_hash"],
                 "Frozen worker differs from launch source")
        source_check = lambda: assert_source(snapshot)
        source_check(); guard.require()
        reference = physical._checked_manifest(ROOT / physical.REFERENCE_PATH, physical.REFERENCE_HASH)
        preflight = physical._checked_manifest(ROOT / physical.DECLARATION_PATH, physical.DECLARATION_HASH)
        target = reference["target"]
        from resectionlab.imaging import load_case
        from resectionlab.geometry import AccessWindow
        from resectionlab.native_resection import native_config_from_case
        from resectionlab.native_axis_simulation import AxisColumnNativeSimulator
        from resectionlab.native_proposals import AxisColumnProposalConfig
        from resectionlab.simulation import RewardSpec
        _require(file_hash(case_bundle) == target["bundle_sha256"], "Patient bundle differs")
        case = load_case(case_bundle); guard.require()
        _require(case.semantic_hash == target["semantic_hash"] and case.planning_hash == target["planning_hash"], "Patient identities differ")
        cfg = native_config_from_case(case, access=AccessWindow(**target["access"]))
        configuration = physical.assert_declared_native_configuration(cfg, preflight, reference)
        write_json(output / "native-configuration.json", configuration)
        optimization = physical._partition(declaration["world_partitions"]["optimization"])
        selection = physical._partition(declaration["world_partitions"]["selection"])
        physical.validate_panel(case, optimization.generator, optimization, selection)
        guard.require()
        cold = time.perf_counter()
        base = AxisColumnNativeSimulator(cfg, proposal_config=AxisColumnProposalConfig(), max_steps=3,
            reward=RewardSpec(**target["reward"]), partial_contact_weight=target["partial_contact_weight"],
            world_generator=optimization.generator,
            compartment_names={i: name for i, name in enumerate(sorted(case.compartments), 1)}, cancelled=guard)
        cold_seconds = time.perf_counter() - cold
        _require(type(base) is AxisColumnNativeSimulator, "Cache/other backend forbidden")
        physical.assert_public_action_model(base, preflight, reference)
        frozen = declaration["frozen_model"]
        _require(base.decision_model_hash == frozen["decision_model_hash"]
                 and cfg.fingerprint == frozen["native_config_hash"]
                 and base._proposer.model_hash == frozen["proposal_model_hash"], "Axis physical model differs")
        # No extra initializer or policy RNG is consumed. The learner's actual
        # initial.pt is authenticated at its first scored factory call.
        write_json(output / "preparation.json", {"seconds": time.perf_counter() - started,
            "cold_axis_construction_seconds": cold_seconds, "model_hash": base.decision_model_hash,
            "input_profile": profile, "expected_initializer_hash": frozen["initial_policy_hash"],
            "expected_trainable_parameter_hash": frozen["initial_trainable_parameter_hash"],
            "cache_enabled": False, "zero_gradients": True, "zero_transitions": True,
            "resource": guard.receipt()})
        guard.require(); source_check()
        status["stage"] = "pilot"
        run_status = run_pilot(case, base, optimization, selection, declaration, output / "pilot", guard,
                              source_check=source_check)
        guard.require(); source_check()
        status.update(status="completed", stage="complete", eligible_candidate_count=1,
            pilot_status_sha256=file_hash(output / "pilot/status.json"),
            candidate_record_sha256=run_status["candidate_record_sha256"],
            preparation_sha256=file_hash(output / "preparation.json"),
            native_configuration_sha256=file_hash(output / "native-configuration.json"),
            launch_source_sha256=file_hash(output / "launch-source.json"),
            full_worker_seconds=time.perf_counter() - started, resource=guard.receipt())
        write_json(output / "worker-resource.json", guard.receipt())
        guard.require(); source_check()
        write_json(output / "worker-status.json", status)
    except Exception as error:
        status.update(status="failed", eligible_candidate_count=0, error_type=type(error).__name__, error=str(error),
                      full_worker_seconds=time.perf_counter() - started, resource=guard.receipt())
        write_json(output / "worker-status.json", status)
        raise


def launch_complete(output: Path, returncode: int, parent_timeout: bool) -> bool:
    if returncode != 0 or parent_timeout:
        return False
    try:
        worker = json.loads((output / "worker-status.json").read_text())
        pilot = json.loads((output / "pilot/status.json").read_text())
        return (worker["status"] == pilot["status"] == "completed"
            and worker["eligible_candidate_count"] == pilot["eligible_candidate_count"] == 1
            and worker["pilot_status_sha256"] == file_hash(output / "pilot/status.json")
            and worker["preparation_sha256"] == file_hash(output / "preparation.json")
            and worker["native_configuration_sha256"] == file_hash(output / "native-configuration.json")
            and worker["launch_source_sha256"] == file_hash(output / "launch-source.json")
            and worker["candidate_record_sha256"] == pilot["candidate_record_sha256"] == file_hash(output / "pilot/candidate-record.json"))
    except (OSError, KeyError, ValueError):
        return False

def pair_receipt(output: Path, declaration: dict) -> dict:
    """Authenticate both completed arms; never substitute historical RAW output."""
    records, hashes = {}, {}
    runtime_hash = json.loads((output / "launch-source.json").read_text())["runtime_content_hash"]
    for profile in declaration["execution_order"]:
        folder = output / profile
        launcher = json.loads((folder / "launcher-status.json").read_text())
        _require(launcher["status"] == "completed" and launcher["eligible_candidate_count"] == 1
                 and launch_complete(folder, launcher["worker_returncode"], launcher["parent_timeout_requested"]),
                 "Both completed arm authorities required")
        _require(launcher["worker_status_sha256"] == file_hash(folder / "worker-status.json"),
                 "Completed worker authority changed after launcher sealing")
        worker = json.loads((folder / "worker-status.json").read_text())
        _require(isinstance(worker.get("resource"), dict) and "full_worker_seconds" in worker,
                 "Missing completed worker resource receipt")
        resource = worker["resource"]
        _require({"budget", "cancellation_reason", "budget_request_to_receipt_seconds", "elapsed_seconds",
                  "observed_peak_rss_bytes"} <= set(resource), "Incomplete completed worker resource receipt")
        budget = declaration["fixed_protocol"]["resource_budget_per_arm"]
        _require(resource["budget"] == budget == launcher["resource_budget"]
                 and resource["cancellation_reason"] is None
                 and resource["budget_request_to_receipt_seconds"] is None
                 and _finite(resource["elapsed_seconds"])
                 and 0 <= resource["elapsed_seconds"] <= budget["worker_wall_seconds"]
                 and _finite(worker["full_worker_seconds"])
                 and 0 <= worker["full_worker_seconds"] <= budget["worker_wall_seconds"]
                 and type(resource["observed_peak_rss_bytes"]) is int
                 and 0 < resource["observed_peak_rss_bytes"] <= budget["process_peak_rss_bytes"]
                 and launcher["parent_timeout_requested"] is False and launcher["hard_killed"] is False,
                 "Completed worker resource receipt contradicts declared limits")
        _require(json.loads((folder / "launch-source.json").read_text())["runtime_content_hash"] == runtime_hash,
                 "Paired executable sources differ")
        arm = arm_declaration(declaration, profile)
        checkpoints = checkpoint_receipt(folder / "pilot/learner", arm)
        result = json.loads((folder / "pilot/learner/result.json").read_text())
        status = json.loads((folder / "pilot/status.json").read_text())
        journal = json.loads((folder / "pilot/accounting.json").read_text())
        frozen_history = json.loads((folder / "pilot/history-freeze.json").read_text())
        candidate = json.loads((folder / "pilot/candidate-record.json").read_text())
        audits = json.loads((folder / "pilot/native-audits.json").read_text())
        completion = validate_completion(arm, result, journal, checkpoints)
        completion["source_voxel_volume_mm3"] = frozen_history["source_voxel_volume_mm3"]
        _require(json.loads(json.dumps(completion)) == frozen_history == candidate["completion"]
                 and audits == candidate["audits"]
                 and candidate["declaration_hash"] == declaration["declaration_content_hash"],
                 "Completed arm payload differs from current frozen evidence")
        _require(json.loads((folder / "pilot/pre-gradient-initializer.json").read_text()) ==
                 checkpoints["initialization"], "Pre-gradient initializer receipt changed")
        validate_audits(completion, audits)
        _require(status["completed_episode_audit_receipts"] == 6 and status["gradient_steps"] == 1,
                 "Paired audit/update denominator differs")
        panels = result["selection_history"]
        _require(len(panels) == 2 and [p["gradient_steps"] for p in panels] == [0, 1]
                 and all(p["world_count"] == 2 and _finite(p["mean_return"]) for p in panels),
                 "Paired complete panels required")
        initial, latest = (p["mean_return"] for p in panels)
        selected_index = max(range(2), key=lambda i: panels[i]["mean_return"])
        _require(result["selected_selection_return"] == panels[selected_index]["mean_return"]
                 and result["selected_checkpoint_hash"] == panels[selected_index]["checkpoint_hash"],
                 "Paired earliest selection differs")
        records[profile] = {"initial_return": initial, "latest_return": latest,
            "selected_return": result["selected_selection_return"], "latest_minus_initial": latest - initial,
            "selected_minus_initial": result["selected_selection_return"] - initial,
            "selected_phase": "initial" if selected_index == 0 else "latest",
            "gradient_steps": result["gradient_steps"], "actor_parameters_changed": result["actor_parameters_changed"],
            "initialization": checkpoints["initialization"], "checkpoint_identities": {
                key: checkpoints[key] for key in ("initial_policy_hash", "latest_policy_hash", "selected_policy_hash")},
            "online_seconds": result["elapsed_seconds"], "initialization_seconds": result["initialization_seconds"],
            "launcher_seconds": launcher["full_launcher_seconds"],
            "training_and_audit_costs": candidate["timing"],
            "preparation": json.loads((folder / "preparation.json").read_text()),
            "worker": worker,
            "episode_outcomes": [{"episode": row["episode"], "role": row["role"], "update": row["update"],
                "seed": row["seed"], "return": row["return"], "transition_count": row["transition_count"],
                "history_hash": row["history_hash"], **{key: sum(h[key] for h in row["native_history"])
                    for key in ("target_removed_mm3", "normal_removed_mm3", "partial_normal_contact_mm3")}}
                for row in completion["episodes"]],
            "completed_episode_audit_receipts": status["completed_episode_audit_receipts"],
            "unique_native_audits": status["unique_native_audits"]}
        for name in ("launcher-status.json", "launch-source.json", "worker-status.json", "preparation.json",
                     "pilot/status.json", "pilot/candidate-record.json", "pilot/native-audits.json",
                     "pilot/accounting.json", "pilot/history-freeze.json", "pilot/pre-gradient-initializer.json",
                     "pilot/learner/initial.pt", "pilot/learner/checkpoint.pt",
                     "pilot/learner/contract.json", "pilot/learner/result.json"):
            hashes[profile + "/" + name] = file_hash(folder / name)
    raw, scaled = records["RAW"], records["FEATURE_UNITS"]
    expected = declaration["paired_initialization"]["expected_trainable_parameter_hash"]
    _require(raw["initialization"]["initial_trainable_parameter_hash"] ==
             scaled["initialization"]["initial_trainable_parameter_hash"] == expected,
             "Paired trainable initial tensors differ")
    _require(raw["initialization"]["axis_observation_contract"] == scaled["initialization"]["axis_observation_contract"],
             "Paired observation semantics differ")
    _require(raw["initialization"]["initial_policy_hash"] != scaled["initialization"]["initial_policy_hash"],
             "Profile behavior hashes must be distinct")
    return {"study_id": declaration["study_id"], "declaration_content_hash": declaration["declaration_content_hash"],
        "runtime_content_hash": runtime_hash, "arms": records,
        "FEATURE_UNITS_minus_RAW": {key: scaled[key] - raw[key] for key in
            ("initial_return", "latest_return", "selected_return", "latest_minus_initial", "selected_minus_initial")},
        "input_sha256": hashes, "completed_episode_audit_receipts": 12,
        "cache_enabled": False, "final_worlds_used": False, "stress_worlds_used": False,
        "scope": "One seed, one previously studied development patient, one update per profile; fixed order and uncontrolled OS load; no efficacy or runtime superiority inference.",
        "consumer_authority": "Both arm worker/launcher authorities and the final pair launcher must be completed and bind these exact bytes."}


def launch_arm(output: Path, frozen: Path, profile: str, case_bundle: Path, snapshot: dict,
               declaration: dict) -> dict:
    started = time.perf_counter()
    output.mkdir(parents=True, exist_ok=False)
    write_json(output / "launch-source.json", snapshot)
    write_json(output / "arm-declaration.json", arm_declaration(declaration, profile))
    budget = physical.PreflightBudget(**declaration["fixed_protocol"]["resource_budget_per_arm"])
    status = {"status": "running", "profile": profile, "eligible_candidate_count": 0,
        "final_worlds_used": False, "stress_worlds_used": False, "resource_budget": asdict(budget)}
    write_json(output / "launcher-status.json", status)
    try:
        env = {**os.environ, "PYTHONPATH": str(frozen / "src"), "GIT_CEILING_DIRECTORIES": str(frozen.parent)}
        command = [sys.executable, str(frozen / "scripts/run_native_axis_feature_units.py"), "--worker",
            "--profile", profile, "--output", str(output), "--case-bundle", str(case_bundle.resolve())]
        parent_timeout, hard_killed = False, False
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
                    process.kill(); process.wait()
        complete = launch_complete(output, process.returncode, parent_timeout)
        status.update(status="completed" if complete else "failed", worker_returncode=process.returncode,
            eligible_candidate_count=1 if complete else 0, parent_timeout_requested=parent_timeout,
            hard_killed=hard_killed, full_launcher_seconds=time.perf_counter() - started, worker_command=command)
        if (output / "worker-status.json").exists():
            status["worker_status_sha256"] = file_hash(output / "worker-status.json")
        write_json(output / "launcher-status.json", status)
        return status
    except Exception as error:
        status.update(status="failed", error_type=type(error).__name__, error=str(error),
            eligible_candidate_count=0, full_launcher_seconds=time.perf_counter() - started)
        write_json(output / "launcher-status.json", status)
        raise


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--case-bundle", type=Path)
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--profile", choices=("RAW", "FEATURE_UNITS"), help=argparse.SUPPRESS)
    args = parser.parse_args(argv)
    if args.worker:
        _require(args.profile is not None and args.case_bundle is not None, "Worker needs profile and case")
        public_worker(args.output, args.case_bundle, args.profile)
        return
    _require(args.profile is None, "Profiles run only in the fixed paired order")
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    declaration = load_declaration()
    write_json(output / "declaration.json", declaration)
    status = {"status": "declared_not_executed", "expected_arms": 2, "completed_arms": 0,
        "eligible_pair_count": 0, "final_worlds_used": False, "stress_worlds_used": False}
    write_json(output / "launcher-status.json", status)
    if not args.execute:
        return
    started = time.perf_counter()
    try:
        _require(args.case_bundle is not None, "--execute requires --case-bundle")
        snapshot = source_snapshot()
        write_json(output / "launch-source.json", snapshot)
        frozen = output / "frozen-source"
        files = dict(snapshot["file_sha256"])
        # Immutable cost/reference evidence copied separately from runtime code.
        for manifest in (declaration, baseline.load_declaration()):
            for item in manifest["references"].values():
                if isinstance(item, dict) and "path" in item:
                    files[item["path"]] = item["file_sha256"]
        for name, digest in files.items():
            data = (ROOT / name).read_bytes()
            _require(hashlib.sha256(data).hexdigest() == digest, "SOURCE_CHANGED_WHILE_COPYING")
            target = frozen / name
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(data)
        assert_source(snapshot)
        load_declaration(frozen)
        _require(source_snapshot(frozen)["runtime_content_hash"] == snapshot["runtime_content_hash"],
                 "Frozen source copy differs")
        status.update(status="running", launch_preparation_seconds=time.perf_counter() - started, arms={})
        write_json(output / "launcher-status.json", status)
        for profile in declaration["execution_order"]:
            assert_source(snapshot)
            _require(source_snapshot(frozen)["runtime_content_hash"] == snapshot["runtime_content_hash"],
                     "Frozen runtime changed between arms")
            arm_status = launch_arm(output / profile, frozen, profile, args.case_bundle, snapshot, declaration)
            status["arms"][profile] = arm_status["status"]
            status["completed_arms"] += int(arm_status["status"] == "completed")
            write_json(output / "launcher-status.json", status)
        _require(status["completed_arms"] == 2, "Paired comparison incomplete; all attempted arm receipts retained")
        receipt = pair_receipt(output, declaration)
        write_json(output / "paired-result.json", receipt)
        assert_source(snapshot)
        _require(source_snapshot(frozen)["runtime_content_hash"] == snapshot["runtime_content_hash"], "Frozen source changed")
        _require(pair_receipt(output, declaration) == receipt, "Paired evidence changed during final export")
        _require(json.loads((output / "paired-result.json").read_text()) == receipt, "Paired export differs")
        status.update(status="completed", eligible_pair_count=1,
            paired_result_sha256=file_hash(output / "paired-result.json"),
            full_launcher_seconds=time.perf_counter() - started,
            timing_scope="Full sequential pair including source freeze, both cold workers and final pairing exports; nested arm costs not additive to this total.")
        write_json(output / "launcher-status.json", status)
    except Exception as error:
        status.update(status="failed", eligible_pair_count=0, error_type=type(error).__name__, error=str(error),
            full_launcher_seconds=time.perf_counter() - started)
        write_json(output / "launcher-status.json", status)
        raise


if __name__ == "__main__":
    main()
