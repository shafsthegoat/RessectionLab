#!/usr/bin/env python3
"""Independent, read-only post-run audit of the declared RAW axis pilot.

This module does not import resectionlab, instantiate a policy, construct a
simulator, perform a gradient, or draw random numbers. It reads tensor-only
checkpoints and evaluates their two linear/tanh heads with NumPy. This is
post-run verification of already captured decisions, never online inference.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import importlib.util
from io import BytesIO
import json
import math
from pathlib import Path
import platform
from zipfile import ZipFile

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
DECLARATION_HASH = "sha256:7783126ab41fab01be6b148131c23fdd305c1e3276d27edb2971a734b644e385"
HELPER_PATH = ROOT / "artifacts/native-axis-v2-result-audit/audit.py"
HELPER_SHA256 = "8390d427f70635549554c3935fb4b77a3027baa39994cb41af2dd443c303b346"
FORWARD_ATOL = 2e-6
FORWARD_RTOL = 2e-5

# These helpers are from the previously reviewed independent source-cell
# auditor, not the simulator or the candidate-generating geometry checker.
if hashlib.sha256(HELPER_PATH.read_bytes()).hexdigest() != HELPER_SHA256:
    raise RuntimeError("Independent source-cell helper changed")
_spec = importlib.util.spec_from_file_location("independent_axis_v2_helpers", HELPER_PATH)
_helper = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_helper)
AuditError, Reader = _helper.AuditError, _helper.Reader
require, close, finite = _helper.require, _helper.close, _helper.finite
canonical_hash, file_hash = _helper.canonical_hash, _helper.file_hash
tensor_hash, tensor_array, array_hash = _helper.tensor_hash, _helper.tensor_array, _helper.array_hash
index_set, check_microstep_accounting = _helper.index_set, _helper.check_microstep_accounting


def accounting_hash(value: dict) -> str:
    return "sha256:" + hashlib.sha256(json.dumps(value, sort_keys=True, allow_nan=False).encode()).hexdigest()


def evidence_exists(path: Path) -> bool:
    return path.exists() or Path(str(path) + ".gz").exists()


def read_array(record: dict, shape: tuple, dtype: str) -> np.ndarray:
    require(isinstance(record, dict) and np.dtype(record["dtype"]) == np.dtype(dtype)
            and tuple(record["shape"]) == shape, "Recorded array metadata differs")
    raw = np.asarray(record["values"])
    require(raw.shape == shape and (raw.dtype == np.bool_ if dtype == "bool" else np.isfinite(raw).all()),
            "Recorded array values differ")
    result = np.asarray(record["values"], dtype=dtype)
    require(np.array_equal(raw, result), "Recorded values do not exactly represent their declared dtype")
    return result


def direct_raw_forward(weights: dict, actions: np.ndarray, state: np.ndarray,
                       mask: np.ndarray) -> tuple[np.ndarray, float]:
    """Separate NumPy implementation: concatenate, affine, tanh, affine."""
    expected = {f"{head}.{layer}.{parameter}" for head in ("actor", "value")
                for layer in ("0", "2") for parameter in ("weight", "bias")}
    require(set(weights) == expected, "Unexpected RAW tensor/buffer schema")
    arrays = {name: tensor_array(value) for name, value in weights.items()}
    shapes = {"actor.0.weight": (16, 21), "actor.0.bias": (16,), "actor.2.weight": (1, 16),
              "actor.2.bias": (1,), "value.0.weight": (16, 6), "value.0.bias": (16,),
              "value.2.weight": (1, 16), "value.2.bias": (1,)}
    require(all(value.dtype == np.float32 and value.shape == shapes[name] and np.isfinite(value).all()
                for name, value in arrays.items()), "Invalid checkpoint tensor dtype/shape/values")
    require(actions.dtype == state.dtype == np.float32 and actions.ndim == 2 and actions.shape[1] == 15
            and state.shape == (6,) and mask.dtype == np.bool_ and mask.shape == (len(actions),)
            and len(mask) > 0 and mask[0], "Invalid direct-forward inputs")
    combined = np.concatenate((actions, np.broadcast_to(state, (len(actions), 6))), axis=1)
    hidden = np.tanh(combined @ arrays["actor.0.weight"].T + arrays["actor.0.bias"])
    logits = (hidden @ arrays["actor.2.weight"].T + arrays["actor.2.bias"]).reshape(-1)
    logits = np.where(mask, logits, -np.inf)
    value_hidden = np.tanh(state @ arrays["value.0.weight"].T + arrays["value.0.bias"])
    value = float((value_hidden @ arrays["value.2.weight"].T + arrays["value.2.bias"])[0])
    return logits, value


def verify_forward(payload: dict, weights: dict) -> dict:
    require(payload["version"] == "learner-decision-observer-v1"
            and type(payload["forward_evaluated"]) is bool, "Unknown decision observer contract")
    inputs = payload["inputs"]
    ids = inputs["action_ids"]
    require(ids and ids[0] == "STOP" and len(ids) == len(set(ids)) <= 27, "Invalid action identity inventory")
    mask = read_array(inputs["action_mask"], (len(ids),), "bool")
    selected = payload["selected_index"]
    require(type(selected) is int and 0 <= selected < len(ids) and bool(mask[selected])
            and ids[selected] == payload["selected_action_id"], "Selected action not eligible")
    source_actions = read_array(inputs["source_action_features"], (len(ids), 15), "float32")
    source_state = read_array(inputs["source_state_features"], (6,), "float32")
    if not payload["forward_evaluated"]:
        require(payload["decision_rule"] == "forced_stop" and payload["selected_action_id"] == "STOP"
                and payload["forced_reason"] == "episode_step_limit" and payload["role"] == "selection"
                and payload["logits"] is None and payload["value"] is None
                and all(inputs[k] is None for k in ("action_features", "actor_action_features", "state_features")),
                "Skipped-forward decision has invented outputs or wrong rule")
        return {"evaluated": False, "forced_stop": True, "max_logit_abs_difference": None,
                "value_abs_difference": None, "rng_replayed": False}
    actions = read_array(inputs["action_features"], (len(ids), 15), "float32")
    actor_actions = read_array(inputs["actor_action_features"], (len(ids), 15), "float32")
    state = read_array(inputs["state_features"], (6,), "float32")
    require(np.array_equal(actions, source_actions) and np.array_equal(actor_actions, actions)
            and np.array_equal(state, source_state), "RAW feature transformation or source drift")
    saved = payload["logits"]
    require(np.dtype(saved["dtype"]) == np.dtype("float32") and saved["shape"] == [len(ids)], "Logit metadata differs")
    recorded = np.asarray([float(x) for x in saved["values"]], np.float64)
    require(recorded.shape == mask.shape and np.isfinite(recorded[mask]).all()
            and np.isneginf(recorded[~mask]).all() and finite(payload["value"]), "Malformed captured forward")
    independent, value = direct_raw_forward(weights, actor_actions, state, mask)
    differences = np.abs(independent[mask].astype(np.float64) - recorded[mask])
    allowance = FORWARD_ATOL + FORWARD_RTOL * np.abs(recorded[mask])
    value_difference = abs(value - payload["value"])
    require(np.all(differences <= allowance)
            and value_difference <= FORWARD_ATOL + FORWARD_RTOL * abs(payload["value"]),
            "Saved logits/value disagree with independent checkpoint tensor operations")
    argmax = int(np.argmax(independent))
    if payload["role"] == "selection":
        require(payload["decision_rule"] == "deterministic_argmax" and payload["forced_reason"] is None
                and selected == argmax, "Selection differs from independently recomputed earliest-row argmax")
    else:
        require(payload["role"] == "optimization" and payload["decision_rule"] == "sampled_categorical"
                and payload["forced_reason"] is None, "Stochastic decision rule changed")
    return {"evaluated": True, "max_logit_abs_difference": float(differences.max(initial=0.)),
            "value_abs_difference": value_difference, "independent_argmax_index": argmax,
            "selected_index": selected, "legal_action_count": int(mask.sum()),
            "argmax_verified": payload["role"] == "selection", "rng_replayed": False,
            "nonzero_logit_difference_count": int(np.count_nonzero(differences)),
            "logit_abs_differences": differences.tolist(), "atol": FORWARD_ATOL, "rtol": FORWARD_RTOL}


def check_authorities(read: Reader, attempt: Path) -> tuple[dict | None, dict]:
    names = ("launcher-status.json", "worker-status.json", "pilot/status.json")
    statuses = {name: read.json(attempt / name) if evidence_exists(attempt / name) else None for name in names}
    launcher, worker, pilot = (statuses[name] for name in names)
    if not all(row is not None and row.get("status") == "completed" for row in (launcher, worker, pilot)):
        # Inner completed records can survive a later worker/launcher failure;
        # they remain diagnostics and must never override the outer authority.
        if launcher is not None and launcher.get("status") != "completed":
            require(launcher.get("eligible_candidate_count") == 0, "Incomplete launcher claims a candidate")
        require(launcher is None or launcher.get("status") != "completed", "Completed launcher lacks completed worker/pilot authority")
        return None, {"audit_status": "incomplete_attempt_retained", "candidate_eligible": False,
                      "authorities": statuses, "interpretation": "No completed atomic authority chain; retained artifacts are not a candidate."}
    require(launcher["worker_returncode"] == 0 and launcher["parent_timeout_requested"] is False
            and launcher["hard_killed"] is False, "Timed-out/failed worker cannot publish")
    require(all(row["eligible_candidate_count"] == 1 for row in (launcher, worker, pilot)), "Candidate denominator differs")
    require(all(row["final_worlds_used"] is False and row["stress_worlds_used"] is False
                for row in (launcher, worker, pilot)), "Final/stress scope changed")
    require(worker["pilot_status_sha256"] == hashlib.sha256(read.bytes(attempt / "pilot/status.json")).hexdigest(),
            "Worker does not bind pilot authority")
    payload = read.bytes(attempt / "pilot/candidate-record.json")
    require(worker["candidate_record_sha256"] == pilot["candidate_record_sha256"] == hashlib.sha256(payload).hexdigest(),
            "Publication payload bytes differ")
    require(launcher["gradient_steps"] == pilot["gradient_steps"] == 1, "Published update denominator differs")
    return json.loads(payload), statuses


def score_episode(transitions: list, compartments: dict, tissue: np.ndarray,
                  affine: np.ndarray, model: dict, provenance: str) -> dict:
    """Recompute rewards from disjoint source-cell sets, including STOP."""
    target = np.logical_or.reduce(list(compartments.values()))
    volume = abs(float(np.linalg.det(affine[:3, :3])))
    removed, charged_partial, contacts = set(), set(), set()
    target_count = normal_count = normal_contacts = 0
    previous_tool = None
    rewards = []
    reward = model["physical_reward"]
    for transition in transitions:
        if not transition["native_commit"]:
            require(transition["action_id"] == "STOP" and transition["terminated"] is True,
                    "Uncommitted non-STOP transition")
            close(transition["reward"], 0., "STOP reward")
            rewards.append(0.)
            continue
        item = transition["info"]
        require(item["action_id"] == transition["action_id"] and item["source_hash"] == model["case_hash"]
                and item["decision_model_hash"] == model["native_config_hash"]
                and item["tissue_support_provenance"] == provenance
                and np.array_equal(item["native_affine"], affine), "Native history source/model differs")
        cells, touched = check_microstep_accounting(item, tissue)
        require(not cells.intersection(removed), "A source cell received repeated removal credit")
        new_partial = touched - removed - cells - charged_partial
        gained = sum(bool(target[cell]) for cell in cells)
        normal = len(cells) - gained
        partial = sum(not target[cell] for cell in new_partial)
        for field, value in (("removed_volume_mm3", len(cells) * volume), ("target_removed_mm3", gained * volume),
                             ("normal_removed_mm3", normal * volume), ("partial_normal_contact_mm3", partial * volume)):
            close(item[field], value, field)
        for field in ("motor_surrogate_delta", "language_surrogate_delta", "partial_motor_contact_surrogate", "partial_language_contact_surrogate"):
            close(item[field], 0., "Absent functional evidence has no surrogate charge")
        distance = float(np.linalg.norm(np.asarray(item["tip_mm"]) - np.asarray(item["entry_mm"])))
        changed = previous_tool is not None and previous_tool != item["tool_id"]
        score = (reward["target_per_mm3"] * gained * volume - reward["normal_per_mm3"] * normal * volume
                 - model["partial_contact_weight"] * reward["normal_per_mm3"] * partial * volume
                 - reward["action_cost"] - 2 * reward["motion_per_mm"] * distance - reward["tool_change_cost"] * changed)
        close(transition["reward"], score, "Independent per-transition source-cell reward")
        close(transition["removed_cells"], len(cells), "Authenticated removed cell count")
        require(transition["after_revision"] == transition["before_revision"] + 1
                and transition["before_state_hash"] != transition["after_state_hash"], "Native commit ancestry differs")
        rewards.append(score)
        removed.update(cells); contacts.update(touched); charged_partial.update(new_partial)
        target_count += gained; normal_count += normal; normal_contacts += partial
        previous_tool = item["tool_id"]
    return {"return": sum(rewards), "step_rewards": rewards, "target_removed_mm3": target_count * volume,
            "normal_removed_mm3": normal_count * volume, "partial_normal_contact_mm3": normal_contacts * volume,
            "removed_mm3": len(removed) * volume, "retained_contact_mm3": len(contacts - removed) * volume,
            "residual_target_mm3": (np.count_nonzero(target) - target_count) * volume,
            "native_commit_count": sum(row["native_commit"] for row in transitions),
            "removed_by_compartment_mm3": {name: sum(bool(mask[cell]) for cell in removed) * volume
                                             for name, mask in compartments.items()},
            "residual_by_compartment_mm3": {name: (np.count_nonzero(mask) - sum(bool(mask[cell]) for cell in removed)) * volume
                                              for name, mask in compartments.items()},
            "clinical_deficit_probability": None, "functional_evidence_available": {"motor": False, "language": False}}


def verify_sources(read: Reader, attempt: Path, baseline_path: Path, archive: Path, declaration: dict) -> dict:
    baseline = read.json(baseline_path)
    source = read.json(attempt / "launch-source.json")
    require(baseline["declaration_content_hash"] == DECLARATION_HASH, "Released declaration identity differs")
    body = dict(declaration); supplied = body.pop("declaration_content_hash")
    require(supplied == canonical_hash(body) == DECLARATION_HASH, "Pilot declaration changed")
    require(canonical_hash({"files": source["file_sha256"], "versions": source["runtime_versions"], "python": source["python"]})
            == source["runtime_content_hash"], "Runtime/source fingerprint differs")
    for name, digest in source["file_sha256"].items():
        require(file_hash(archive / name) == file_hash(attempt / "frozen-source" / name) == digest,
                "Frozen/archive executable bytes changed: " + name)
    require(source["file_sha256"]["scripts/run_native_axis_pilot.py"] == baseline["runner_sha256"], "Released runner changed")
    require(source["file_sha256"]["manifests/experiments/native-axis-raw-update-pilot-v1.json"] == baseline["declaration_sha256"],
            "Released declaration bytes changed")
    for item in declaration["references"].values():
        if isinstance(item, dict) and "path" in item:
            require(file_hash(archive / item["path"]) == file_hash(attempt / "frozen-source" / item["path"])
                    == item["file_sha256"], "Immutable declaration reference changed")
    if "runtime_content_hash" in baseline:
        require(baseline["runtime_content_hash"] == source["runtime_content_hash"], "Released runtime changed")
    return {"source_commit": baseline["source_commit"], "runtime_content_hash": source["runtime_content_hash"],
            "runtime_versions": source["runtime_versions"], "python": source["python"],
            "source_file_count": len(source["file_sha256"]), "file_sha256": source["file_sha256"], "baseline": baseline}


def read_source_case(read: Reader, attempt: Path, bundle: Path, archive: Path, declaration: dict, baseline: dict):
    reference = read.json(archive / declaration["references"]["physical_and_world_declaration"]["path"])
    target = reference["target"]
    preflight = read.json(archive / declaration["references"]["preflight_declaration"]["path"])
    require(file_hash(bundle) == baseline["case_bundle_sha256"] == target["bundle_sha256"], "Patient bundle changed")
    with ZipFile(bundle) as z:
        manifest = json.loads(z.read("manifest.json"))
        with np.load(BytesIO(z.read("arrays.npz")), allow_pickle=False) as arrays:
            compartments = {name: np.array(arrays[key], bool) for name, key in manifest["array_index"]["compartments"].items()}
            affine = np.array(arrays["affine"])
            from scipy.ndimage import binary_fill_holes
            tissue = binary_fill_holes(np.asarray(arrays["mri"]) != 0) | np.logical_or.reduce(list(compartments.values()))
    require(manifest["case_semantic_hash"] == target["semantic_hash"] == declaration["frozen_model"]["case_hash"]
            and manifest["frame"] == "RAS+", "Source case/frame differs")
    require(array_hash(tissue) == target["tissue_support_hash"], "Source tissue support differs")
    labels = np.zeros(tissue.shape, np.int16)
    for label, name in enumerate(sorted(compartments), 1):
        mask = compartments[name]
        require(array_hash(mask) == target["compartment_mask_hashes"][name] and not np.any(labels[mask]), "Source labels differ/overlap")
        labels[mask] = label
    configuration = read.json(attempt / "native-configuration.json")
    require(configuration["actual"] == preflight["native_configuration"]
            and configuration["actual"]["fingerprint"] == declaration["frozen_model"]["native_config_hash"]
            and configuration["physical_component_equality"] is True
            and configuration["different_components"] == ["tissue_support_provenance"], "Native physical configuration changed")
    for name, array in {"tissue_mask": tissue, "target_labels": labels, "affine": affine,
                        "hard_exclusion": np.zeros(tissue.shape, bool)}.items():
        require(configuration["actual"]["components"][name] == {"shape": list(array.shape), "dtype": str(array.dtype),
                                                                "array_digest": array_hash(array)}, "Native source array changed: " + name)
    require(declaration["frozen_model"]["physical_reward"] == target["reward"]
            and declaration["frozen_model"]["partial_contact_weight"] == target["partial_contact_weight"], "Frozen reward changed")
    return compartments, tissue, affine, configuration["actual"]["components"]["tissue_support_provenance"]


def check_checkpoints(read: Reader, folder: Path, declaration: dict, result: dict, completion: dict,
                      source_files: dict | None = None) -> dict:
    import torch
    initial = torch.load(folder / "initial.pt", map_location="cpu", weights_only=True)
    latest = torch.load(folder / "checkpoint.pt", map_location="cpu", weights_only=True)
    model = declaration["frozen_model"]
    contract = read.json(folder / "contract.json")
    expected_partitions = {role: {key: row[key] for key in ("role", "case_hash", "planning_hash", "generator", "seeds")}
                           | {"partition_hash": canonical_hash(row)} for role, row in declaration["world_partitions"].items()}
    require(contract["config"] == declaration["training_config"] and canonical_hash(contract["partitions"]) == canonical_hash(expected_partitions)
            and contract["decision_model_hash"] == model["decision_model_hash"]
            and contract["population_initialization"] is None and contract["procedural_initialization"] is None,
            "Scratch learner contract changed")
    contract_fields = ("schema_version", "algorithm", "input_profile", "input_profile_hash", "implementation_sha256",
        "simulator_implementation_sha256", "numerical_source_sha256", "runtime", "config", "partitions",
        "population_initialization", "procedural_initialization", "timing_contract", "elapsed_seconds_scope",
        "decision_model_hash", "dimensions")
    require(canonical_hash({key: contract[key] for key in contract_fields}) == contract["contract_hash"] == latest["contract_hash"],
            "Learner contract checksum differs")
    if source_files is not None:
        require(contract["implementation_sha256"] == contract["code_sha256"] == source_files["src/resectionlab/learning.py"]
                and contract["simulator_implementation_sha256"] == source_files["src/resectionlab/native_axis_accounting.py"],
                "Learner implementation differs from executed snapshot")
        expected_modules = {Path(path).name: digest for path, digest in source_files.items()
                            if path.startswith("src/resectionlab/") and len(Path(path).parts) == 3 and path.endswith(".py")}
        require(contract["numerical_source_sha256"] == expected_modules, "Learner numerical source contract differs")
    for payload in (initial, latest):
        require(payload["dimensions"] == [15, 6, 16]
                and payload["input_profile"] == model["input_profile_manifest"]
                and payload["input_profile_hash"] == canonical_hash(payload["input_profile"]) == model["input_profile_hash"]
                and tensor_hash(payload["policy"]) == payload["policy_hash"], "Checkpoint tensor/profile changed")
    require(initial["policy_hash"] == model["initial_policy_hash"] == latest["initial_hash"] == result["initial_checkpoint_hash"],
            "Initial RAW weights changed")
    require(latest["policy_hash"] == result["latest_checkpoint_hash"]
            and tensor_hash(latest["selected_policy"]) == latest["selected_hash"] == result["selected_checkpoint_hash"],
            "Latest/selected checkpoint tensor identity differs")
    actor = lambda state: tensor_hash({k.removeprefix("actor."): v for k, v in state.items() if k.startswith("actor.")})
    require(actor(initial["policy"]) == result["initial_actor_hash"] != result["latest_actor_hash"] == actor(latest["policy"])
            and result["actor_parameters_changed"] is True, "No actual actor tensor update")
    for name, digest in completion["checkpoints"]["files"].items():
        require(file_hash(folder / name) == digest, "Frozen checkpoint/result file changed")
    require(latest["optimizer"]["state"] and all(float(s["step"]) == 1. for s in latest["optimizer"]["state"].values()),
            "Actual Adam step count differs")
    require(all(np.isfinite(tensor_array(value)).all() for state in latest["optimizer"]["state"].values()
                for value in state.values()), "Nonfinite saved optimizer state")
    require(latest["shared_hash"] is None and latest["population_initialization"] is None
            and latest["procedural_initialization"] is None and result["optimizer_mode"] == "PATIENT_SCRATCH_RL"
            and result["shared_checkpoint_hash"] is None, "Unexpected transferred initialization")
    require(all(group["lr"] == declaration["training_config"]["learning_rate"] for group in latest["optimizer"]["param_groups"]),
            "Adam learning rate changed")
    for name, value in latest["state"].items():
        require(canonical_hash(result[name]) == canonical_hash(value), "Latest checkpoint/result state differs")
    receipt = completion["checkpoints"]
    for name, value in {"initial_policy_hash": initial["policy_hash"], "latest_policy_hash": latest["policy_hash"],
                       "selected_policy_hash": latest["selected_hash"], "initial_actor_hash": actor(initial["policy"]),
                       "latest_actor_hash": actor(latest["policy"]), "input_profile_hash": model["input_profile_hash"],
                       "checkpoint_state_hash": canonical_hash(latest["state"])}.items():
        require(receipt[name] == value, "Frozen checkpoint receipt differs: " + name)
    require(receipt["adam_steps"] == [float(s["step"]) for s in latest["optimizer"]["state"].values()],
            "Frozen optimizer step receipt differs")
    return {0: initial["policy"], 1: latest["policy"]}


def check_episode_graph(journal: dict, completion: dict, declaration: dict) -> list[tuple[dict, list, list]]:
    model = declaration["frozen_model"]
    require(journal["receipt_hash"] == accounting_hash({k: v for k, v in journal.items() if k != "receipt_hash"})
            == completion["accounting_receipt_hash"], "Accounting receipt changed")
    require(journal["status"] == "learner_returned" and journal["candidate_eligible"] is False
            and journal["counts_complete"] is True and journal["failure"] is None
            and journal["receipt_export_failure"] is None and journal["decision_recording_enabled"] is True,
            "Incomplete accounting cannot certify candidate")
    require(journal["case_hash"] == model["case_hash"] and journal["decision_model_hash"] == model["decision_model_hash"]
            and canonical_hash(journal["partitions"]) == canonical_hash(declaration["world_partitions"])
            and journal["input_profile"] == "RAW", "Accounting model/worlds changed")
    episodes, events = journal["episodes"], journal["events"]
    require(len(episodes) == 7 and [r["episode"] for r in episodes] == list(range(7))
            and len({r["instance"] for r in episodes}) == 7, "Reset/factory denominator differs")
    require(all(r["kind"] in ("factory", "decision", "transition") for r in events), "Failed/unknown accounting event")
    require(len([r for r in events if r["kind"] == "factory"]) == 7, "Factory denominator differs")
    require(episodes[0] == completion["shape_probe"] and episodes[0]["status"] == "unfinished_at_learner_return"
            and episodes[0]["role"] == "optimization" and episodes[0]["seed"] == declaration["optimization_episode_seeds"][0]
            and not any(r.get("episode") == 0 for r in events), "Shape probe executed decisions/transitions")
    seeds = declaration["world_partitions"]["selection"]["seeds"]
    expected = [("selection", seed, 0, 0, i) for i, seed in enumerate(seeds)]
    expected += [("optimization", seed, 0, None, i) for i, seed in enumerate(declaration["optimization_episode_seeds"])]
    expected += [("selection", seed, 1, 1, i) for i, seed in enumerate(seeds)]
    all_decisions = [r for r in events if r["kind"] == "decision"]
    all_transitions = [r for r in events if r["kind"] == "transition"]
    ids = [f"axis-decision-{i:06d}" for i in range(len(all_decisions))]
    require([r["decision_id"] for r in all_decisions] == ids
            and [r["decision_id"] for r in all_transitions] == ids
            and 0 < len(ids) <= 18 and all(1 <= r["episode"] <= 6 for r in all_decisions + all_transitions),
            "Attempt/transition monotonic identity join differs")
    rows = []
    require(len(completion["episodes"]) == 6, "Missing completed frozen history")
    for actual, frozen, (role, seed, update, panel, ordinal) in zip(episodes[1:], completion["episodes"], expected):
        require(actual["status"] == "complete" and actual["role"] == frozen["role"] == role
                and actual["seed"] == frozen["seed"] == seed and actual["episode"] == frozen["episode"]
                and frozen["update"] == update and frozen["panel"] == panel, "Episode phase/seed order differs")
        decisions = [r for r in all_decisions if r["episode"] == actual["episode"]]
        transitions = [r for r in all_transitions if r["episode"] == actual["episode"]]
        require(0 < len(decisions) == len(transitions) == len(frozen["decisions"]) == frozen["transition_count"] <= 3
                and transitions[-1]["terminated"] is True and not any(r["terminated"] for r in transitions[:-1]),
                "Incomplete/overlong executed episode")
        for index, (decision, transition, joined) in enumerate(zip(decisions, transitions, frozen["decisions"])):
            payload = decision["payload"]
            require(decision["status"] == "step_returned" and decision["decision_id"] == transition["decision_id"] == joined["decision_id"]
                    and events.index(decision) < events.index(transition)
                    and transition["executed_transition"] is True and transition["returned_to_learner"] is True
                    and transition["next_observation_complete"] is True, "Decision lacks completed joined transition")
            require((payload["role"], payload["seed"], payload["update"], payload["panel"], payload["episode"], payload["step"])
                    == (role, seed, update, panel, ordinal, index), "Actual decision context differs")
            for key, value in (("instance", actual["instance"]), ("episode", actual["episode"]), ("role", role), ("seed", seed)):
                require(decision[key] == transition[key] == value, "Accounting event/episode identity differs")
            require(all(payload["inputs"]["action_mask"]["values"]), "Axis inventory unexpectedly contains masked-out rows")
            require(payload["selected_action_id"] == transition["action_id"] and canonical_hash(payload) == joined["payload_hash"],
                    "Chosen action or frozen decision changed")
            require(joined["policy_hash"] == completion["checkpoints"]["initial_policy_hash" if update == 0 else "latest_policy_hash"]
                    and joined["model_hash"] == model["decision_model_hash"] and joined["profile_hash"] == model["input_profile_hash"],
                    "Decision checkpoint provenance joined to wrong phase")
        history = [r["info"] for r in transitions if r["native_commit"]]
        require(history == frozen["native_history"] and canonical_hash(history) == frozen["history_hash"]
                and [r["action_id"] for r in transitions] == frozen["actions"], "Frozen native history differs from actual commits")
        rows.append((frozen, decisions, transitions))
    return rows


def check_geometry(completion: dict, audits: list, scores: list, model: dict, volume: float) -> dict:
    require(len(audits) == len(scores) == len(completion["episodes"]) == 6, "Six geometry receipts required")
    seen = {}
    actual_seconds = 0.
    for frozen, report, score in zip(completion["episodes"], audits, scores):
        expected = canonical_hash({"case": model["case_hash"], "model": model["decision_model_hash"], "history": frozen["history_hash"]})
        require(report["episode"] == frozen["episode"] and report["history_hash"] == frozen["history_hash"]
                and report["audit_key"] == expected and report["reused"] is (expected in seen), "Geometry dedup provenance differs")
        audit = report["audit"]
        require(audit["checker_version"] == "independent-native-sequence-v2"
                and all(audit[key] is True for key in ("feasible", "complete_tool_checked", "frontier_checked"))
                and audit["failures"] == [] and audit["first_failed_action"] is None
                and audit["first_unsupported_source_voxel"] is None and audit["first_unsupported_position_mm"] is None
                and audit["source_case_hash"] == model["case_hash"] and audit["action_count"] == score["native_commit_count"],
                "Geometry certificate failed or source/history scope differs")
        for field, value in (("source_voxel_volume_mm3", volume), ("unsupported_source_tissue_volume_mm3", 0.),
                             ("claimed_source_tissue_volume_mm3", score["removed_mm3"]), ("contained_source_tissue_volume_mm3", score["removed_mm3"])):
            close(audit[field], value, "Geometry " + field)
        require(finite(report["seconds"]) and report["seconds"] >= 0, "Invalid original geometry duration")
        if expected in seen:
            require(report["audit"] == seen[expected]["audit"] and report["seconds"] == seen[expected]["seconds"], "Shared certificate differs")
            close(report["actual_check_seconds"], 0., "Reused certificate extra execution")
        else:
            close(report["actual_check_seconds"], report["seconds"], "Original certificate execution")
            actual_seconds += report["seconds"]
            seen[expected] = report
    return {"episode_receipts": 6, "unique_histories": len(seen), "actual_check_seconds": actual_seconds,
            "geometry_reexecuted_by_this_auditor": False}


def audit_attempt(attempt: Path, baseline_json: Path, case_bundle: Path, immutable_archive: Path,
                  reader: Reader | None = None) -> dict:
    read = reader or Reader()
    payload, authorities = check_authorities(read, attempt)
    if payload is None:
        journal_path = attempt / "pilot/accounting.json"
        if evidence_exists(journal_path):
            journal = read.json(journal_path)
            authorities["recorded_totals"] = journal.get("totals")
            authorities["recorded_failure"] = journal.get("failure")
            authorities["counts_complete"] = journal.get("counts_complete")
        return authorities
    declaration = read.json(attempt / "declaration.json")
    sources = verify_sources(read, attempt, baseline_json, immutable_archive, declaration)
    model = declaration["frozen_model"]
    compartments, tissue, affine, provenance = read_source_case(read, attempt, case_bundle, immutable_archive, declaration, sources["baseline"])
    require(set(declaration["world_partitions"]) == {"optimization", "selection"}, "Unexpected world role")
    opt, sel = (declaration["world_partitions"][key] for key in ("optimization", "selection"))
    require(not set(opt["seeds"]).intersection(sel["seeds"]) and declaration["optimization_episode_seeds"] == opt["seeds"][:2],
            "Optimization/selection isolation differs")
    require(opt["generator"] == sel["generator"] and opt["generator"]["translation_scale_mm"] == [0., 0., 0.]
            and opt["generator"]["rotation_scale_deg"] == [0., 0., 0.], "Declared deterministic world model differs")
    completion = read.json(attempt / "pilot/history-freeze.json")
    require(completion == payload["completion"] and completion["candidate_eligible"] is False
            and completion["final_worlds_used"] is False and completion["stress_worlds_used"] is False
            and payload["declaration_hash"] == DECLARATION_HASH, "Freeze/payload identity differs")
    journal = read.json(attempt / "pilot/accounting.json")
    result = read.json(attempt / "pilot/learner/result.json")
    require(result["status"] == "gradient_budget" and result["gradient_steps"] == 1
            and result["completed_episodes"] == 2 and result["discarded_partial_batches"] == 0, "Incomplete one-update learner")
    for key, value in journal["learner_result"].items():
        require(value == result[key], "Generic/accounting result differs")
    weights = check_checkpoints(read, attempt / "pilot/learner", declaration, result, completion, sources["file_sha256"])
    graph = check_episode_graph(journal, completion, declaration)
    scores, forwards = [], []
    for frozen, decisions, transitions in graph:
        score = score_episode(transitions, compartments, tissue, affine, model, provenance)
        close(frozen["return"], score["return"], "Frozen episode return")
        scores.append({"episode": frozen["episode"], "role": frozen["role"], "update": frozen["update"], **score})
        for decision in decisions:
            forwards.append({"decision_id": decision["decision_id"], "episode": frozen["episode"],
                             "update": frozen["update"], "role": frozen["role"],
                             **verify_forward(decision["payload"], weights[frozen["update"]])})
    for role, limit in (("optimization", 6), ("selection", 12)):
        transitions = [t for f, d, rows in graph if f["role"] == role for t in rows]
        totals = journal["totals"][role]
        require(totals["counts_complete"] is True and totals["committed_but_unreturned"] == 0
                and totals["executed_transitions"] == totals["returned_transitions"] == result[role + "_environment_steps"] == len(transitions) <= limit
                and totals["native_commits"] == sum(t["native_commit"] for t in transitions), "Actual role denominator differs")
        close(totals["executed_reward"], sum(t["reward"] for t in transitions), "Actual role rewards")
    panels = result["selection_history"]
    require(len(panels) == 2 and [p["gradient_steps"] for p in panels] == [0, 1]
            and all(p["world_count"] == 2 for p in panels), "Incomplete selection panel")
    means = [sum(scores[i]["return"] for i in indices) / 2 for indices in ((0, 1), (4, 5))]
    for update, mean in enumerate(means):
        close(panels[update]["mean_return"], mean, "Independent complete-panel mean")
        require(panels[update]["checkpoint_hash"] == tensor_hash(weights[update]), "Panel checkpoint differs")
    chosen = max(range(2), key=lambda i: means[i])
    require(completion["selected_panel"] == chosen and completion["selected_checkpoint_hash"]
            == result["selected_checkpoint_hash"] == tensor_hash(weights[chosen]), "Earliest-best selected checkpoint differs")
    close(result["initial_selection_return"], means[0], "Initial selection score")
    close(result["selected_selection_return"], means[chosen], "Selected score")
    updates = result["optimization_history"]
    require(len(updates) == 1 and updates[0]["gradient_steps"] == 1
            and updates[0]["episode_source_case_hashes"] == [model["case_hash"], model["case_hash"]]
            and finite(updates[0]["loss"]) and finite(updates[0]["gradient_norm_before_clip"])
            and finite(updates[0]["actor_gradient_norm_after_clip"]) and updates[0]["actor_gradient_norm_after_clip"] > 0.,
            "Invalid actor gradient/update evidence")
    close(updates[0]["mean_return"], sum(row["return"] for row in scores[2:4]) / 2, "Optimization episode mean")
    audits = read.json(attempt / "pilot/native-audits.json")
    require(audits == payload["audits"], "Saved native audits differ from frozen candidate payload")
    geometry = check_geometry(completion, audits, scores, model, abs(float(np.linalg.det(affine[:3, :3]))))
    timing = payload["timing"]
    for key, value in (("learner_initialization_seconds", result["initialization_seconds"]),
                       ("optimization_selection_seconds", result["elapsed_seconds"]), ("selection_seconds", result["selection_seconds"]),
                       ("final_checkpoint_export_seconds", result["final_checkpoint_export_seconds"])):
        close(timing[key], value, key)
        require(value >= 0, "Negative duration")
    require(len(timing["factory_clone_seconds"]) == 7 and all(finite(v) and v >= 0 for v in timing["factory_clone_seconds"]), "Factory timing denominator differs")
    require(result["elapsed_seconds"] >= result["selection_seconds"] >= sum(p["panel_elapsed_seconds"] for p in panels)
            and timing["full_training_call_seconds"] >= result["elapsed_seconds"] + result["initialization_seconds"]
            and timing["independent_validation_seconds"] >= geometry["actual_check_seconds"], "Timing scopes contradict")
    for key in ("worker-status.json", "pilot/status.json"):
        resource = authorities[key]["resource"]
        require(resource["cancellation_reason"] is None and resource["observed_peak_rss_bytes"] <= declaration["resource_budget"]["process_peak_rss_bytes"],
                "Published worker cancelled or exceeded measured RSS")
    return {"audit_status": "passed", "candidate_eligible": True, "declaration_hash": DECLARATION_HASH,
            "sources": {k: v for k, v in sources.items() if k != "baseline"}, "episode_scores": scores,
            "initial_selection_return": means[0], "updated_selection_return": means[1], "selected_panel": chosen,
            "selected_checkpoint_hash": result["selected_checkpoint_hash"], "geometry": geometry,
            "forward_verification": {"method": "Independent NumPy float32 affine/tanh/affine operations using saved checkpoint tensors",
                "atol": FORWARD_ATOL, "rtol": FORWARD_RTOL, "decisions": forwards,
                "max_logit_abs_difference": max((r["max_logit_abs_difference"] or 0. for r in forwards), default=0.),
                "max_value_abs_difference": max((r["value_abs_difference"] or 0. for r in forwards), default=0.),
                "rng_replayed": False, "stochastic_scope": "Mask/row eligibility and actual captured rule only; random draws not replayed."},
            "timing": {**timing, "online_cap_seconds": declaration["training_config"]["max_wall_seconds"],
                       "online_cap_overshoot_seconds": max(0., result["elapsed_seconds"] - declaration["training_config"]["max_wall_seconds"])},
            "final_worlds_used": False, "stress_worlds_used": False, "simulator_episodes_reexecuted": 0,
            "new_gradient_steps": 0, "clinical_deficit_probability": None,
            "interpretation": "One previously studied deterministic development case; this verifies artifacts and arithmetic, not efficacy or clinical validity."}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--attempt", type=Path, required=True)
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--case-bundle", type=Path, required=True)
    parser.add_argument("--archive", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    require(not args.output.resolve().is_relative_to(args.attempt.resolve()), "Audit output must be outside experiment evidence")
    require(not args.output.exists(), "Audit receipt already exists; retain earlier audits")
    read = Reader()
    try:
        report = audit_attempt(args.attempt, args.baseline, args.case_bundle, args.archive, read)
    except Exception as error:
        report = {"audit_status": "failed", "candidate_eligible": False, "error_type": type(error).__name__, "error": str(error)}
    report.update(created_utc=datetime.now(timezone.utc).isoformat(), auditor_sha256=file_hash(Path(__file__)),
                  independent_helper_sha256=HELPER_SHA256, input_file_sha256=read.hashes,
                  audit_runtime={"python": platform.python_version(), "platform": platform.platform(),
                                 **{name: importlib.metadata.version(name) for name in ("numpy", "scipy", "torch")}})
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x") as stream:
        json.dump(report, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")
    print(json.dumps({"audit_status": report["audit_status"], "candidate_eligible": report["candidate_eligible"], "receipt": str(args.output)}))
    return 0 if report["audit_status"] == "passed" else 2 if report["audit_status"] == "incomplete_attempt_retained" else 1


if __name__ == "__main__":
    raise SystemExit(main())
