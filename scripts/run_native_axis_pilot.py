#!/usr/bin/env python3
"""One declared RAW axis update. Default is declaration-only; no implicit retries.

Only successful worker AND launcher receipts authorize the frozen candidate.
Decision evidence comes from the learner's actual forward calls, never a replay.
"""
from __future__ import annotations

import argparse
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
from resectionlab.worlds import content_hash

DECLARATION_PATH = Path("manifests/experiments/native-axis-raw-update-pilot-v1.json")
PILOT_HASH = "sha256:7783126ab41fab01be6b148131c23fdd305c1e3276d27edb2971a734b644e385"
write_json = physical.write_json


def file_hash(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_declaration(root: Path = ROOT) -> dict:
    declaration = physical._checked_manifest(root / DECLARATION_PATH, PILOT_HASH)
    for record in declaration["references"].values():
        if isinstance(record, dict) and "path" in record:
            path = root / record["path"]
            if file_hash(path) != record["file_sha256"]:
                raise ValueError("Declared immutable reference changed: " + record["path"])
            if "content_hash" in record:
                physical._checked_manifest(path, record["content_hash"])
    return declaration


def source_snapshot(root: Path = ROOT) -> dict:
    snapshot = physical.source_snapshot(root)
    files = snapshot["file_sha256"]
    for name in (str(DECLARATION_PATH), "scripts/run_native_axis_pilot.py"):
        files[name] = file_hash(root / name)
    snapshot["runtime_content_hash"] = content_hash({"files": files,
        "versions": snapshot["runtime_versions"], "python": snapshot["python"]})
    return snapshot


def assert_source(snapshot: dict, root: Path = ROOT) -> None:
    if source_snapshot(root)["runtime_content_hash"] != snapshot["runtime_content_hash"]:
        raise RuntimeError("SOURCE_CHANGED_DURING_AXIS_PILOT")


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def _finite(value):
    return type(value) in (int, float) and math.isfinite(value)


def checkpoint_receipt(folder: Path, declaration: dict) -> dict:
    """Authenticate actual saved weights/Adam state without another policy forward."""
    import torch
    from resectionlab.learning import checkpoint_input_profile, policy_hash
    initial = torch.load(folder / "initial.pt", map_location="cpu", weights_only=True)
    latest = torch.load(folder / "checkpoint.pt", map_location="cpu", weights_only=True)
    contract = json.loads((folder / "contract.json").read_text())
    result = json.loads((folder / "result.json").read_text())
    expected = declaration["frozen_model"]
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
        profile = checkpoint_input_profile(payload, expected_input_profile="RAW")
        _require(profile.to_dict() == expected["input_profile_manifest"]
                 and profile.fingerprint == expected["input_profile_hash"], "Checkpoint RAW profile drift")
        _require(payload["dimensions"] == [15, 6, 16], "Checkpoint dimensions drift")
        _require(policy_hash(payload["policy"]) == payload["policy_hash"], "Checkpoint tensor hash differs")
    _require(initial["policy_hash"] == expected["initial_policy_hash"], "Initial weights differ from declaration")
    _require(latest["initial_hash"] == initial["policy_hash"], "Latest checkpoint has another initializer")
    _require(policy_hash(latest["selected_policy"]) == latest["selected_hash"], "Selected tensor hash differs")
    actor_hash = lambda weights: policy_hash({key.removeprefix("actor."): value
        for key, value in weights.items() if key.startswith("actor.")})
    steps = [float(value["step"]) for value in latest["optimizer"]["state"].values()]
    _require(steps and all(value == 1. for value in steps), "Expected exactly one actual Adam step")
    _require(all(group["lr"] == declaration["training_config"]["learning_rate"]
                 for group in latest["optimizer"]["param_groups"]), "Adam learning rate differs")
    return {"initial_policy_hash": initial["policy_hash"], "latest_policy_hash": latest["policy_hash"],
        "selected_policy_hash": latest["selected_hash"], "initial_actor_hash": actor_hash(initial["policy"]),
        "latest_actor_hash": actor_hash(latest["policy"]), "adam_steps": steps,
        "files": {name: file_hash(folder / name) for name in ("initial.pt", "checkpoint.pt", "contract.json", "result.json")},
        "input_profile_hash": expected["input_profile_hash"],
        "checkpoint_state_hash": content_hash(latest["state"])}


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
            _require(record is not None and np.dtype(record["dtype"]) == np.dtype("float32")
                     and tuple(record["shape"]) == source.shape
                     and np.array_equal(np.asarray(record["values"], dtype=np.float32), source),
                     "RAW actual policy input differs: " + key)
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
             and accounting["input_profile"] == "RAW", "Accounting source/model drift")
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


def validate_audits(completion: dict, audits: list[dict]) -> None:
    _require(len(audits) == 6 and [a["episode"] for a in audits] == [e["episode"] for e in completion["episodes"]],
             "Six native audit receipts required")
    for episode, record in zip(completion["episodes"], audits):
        audit = record["audit"]
        history = episode["native_history"]
        cells = [tuple(cell) for item in history for cell in item["removed_indices_native"]]
        claimed = sum(item["removed_volume_mm3"] for item in history)
        volume = audit["source_voxel_volume_mm3"]
        _require(record["history_hash"] == episode["history_hash"] and audit["feasible"] is True
                 and audit["complete_tool_checked"] is True and audit["frontier_checked"] is True
                 and audit["source_case_hash"] == completion["case_hash"]
                 and type(audit["action_count"]) is int and audit["action_count"] == len(history)
                 and not audit["failures"] and audit["first_failed_action"] is None
                 and audit["first_unsupported_source_voxel"] is None
                 and audit["first_unsupported_position_mm"] is None
                 and audit["unsupported_source_tissue_volume_mm3"] == 0
                 and _finite(volume) and volume > 0
                 and len(cells) == len(set(cells))
                 and math.isclose(len(cells) * volume, claimed, abs_tol=1e-8)
                 and math.isclose(audit["claimed_source_tissue_volume_mm3"], claimed, abs_tol=1e-8)
                 and math.isclose(audit["contained_source_tissue_volume_mm3"], claimed, abs_tol=1e-8)
                 and ("source_voxel_volume_mm3" not in completion
                      or volume == completion["source_voxel_volume_mm3"]), "Independent native audit failed")


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
    try:
        guard.require(); source_check()
        def factory():
            before = time.perf_counter()
            instance = base.clone()
            factory_seconds.append(time.perf_counter() - before)
            return instance
        accounting = AxisTrainingAccounting(factory, optimization, selection,
            receipt_path=output / "accounting.json", expected_model_hash=base.decision_model_hash)
        call = time.perf_counter()
        trainer(accounting, config=TrainingConfig(**declaration["training_config"]), output_dir=output / "learner",
                cancelled=guard, record_decisions=True)
        full_training_call_seconds = time.perf_counter() - call
        guard.require(); source_check()
        result = json.loads((output / "learner/result.json").read_text())
        journal = json.loads((output / "accounting.json").read_text())
        checkpoints = checkpoint_receipt(output / "learner", declaration)
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


def public_worker(output: Path, case_bundle: Path) -> None:
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
        declaration = load_declaration()
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
        import torch
        from resectionlab.learning import MaskedPatientPolicy, policy_hash
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
        physical.assert_public_action_model(base, preflight, reference)
        frozen = declaration["frozen_model"]
        _require(base.decision_model_hash == frozen["decision_model_hash"]
                 and cfg.fingerprint == frozen["native_config_hash"]
                 and base._proposer.model_hash == frozen["proposal_model_hash"], "Axis physical model differs")
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(11)
            policy = MaskedPatientPolicy(15, 6, 16, input_profile="RAW")
        _require(policy_hash(policy) == frozen["initial_policy_hash"]
                 and policy.input_profile.to_dict() == frozen["input_profile_manifest"]
                 and policy.input_profile.fingerprint == frozen["input_profile_hash"], "Seeded initializer/profile differs before optimizer")
        write_json(output / "preparation.json", {"seconds": time.perf_counter() - started,
            "cold_axis_construction_seconds": cold_seconds, "model_hash": base.decision_model_hash,
            "initializer_hash": policy_hash(policy), "zero_gradients": True, "zero_transitions": True,
            "resource": guard.receipt()})
        del policy
        guard.require(); source_check()
        status["stage"] = "pilot"
        run_status = run_pilot(case, base, optimization, selection, declaration, output / "pilot", guard,
                              source_check=source_check)
        guard.require(); source_check()
        status.update(status="completed", stage="complete", eligible_candidate_count=1,
            pilot_status_sha256=file_hash(output / "pilot/status.json"),
            candidate_record_sha256=run_status["candidate_record_sha256"],
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
            and worker["candidate_record_sha256"] == pilot["candidate_record_sha256"] == file_hash(output / "pilot/candidate-record.json"))
    except (OSError, KeyError, ValueError):
        return False


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--case-bundle", type=Path)
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args(argv)
    if args.worker:
        public_worker(args.output, args.case_bundle)
        return
    args.output = args.output.resolve()
    args.output.mkdir(parents=True, exist_ok=False)
    declaration = load_declaration()
    write_json(args.output / "declaration.json", declaration)
    status = {"status": "declared_not_executed", "gradient_steps": 0, "eligible_candidate_count": 0,
              "final_worlds_used": False, "stress_worlds_used": False}
    write_json(args.output / "launcher-status.json", status)
    if not args.execute:
        return
    launched = time.perf_counter()
    try:
        _require(args.case_bundle is not None, "--execute requires --case-bundle")
        snapshot = source_snapshot()
        write_json(args.output / "launch-source.json", snapshot)
        frozen = args.output / "frozen-source"
        files = dict(snapshot["file_sha256"])
        for item in declaration["references"].values():
            if isinstance(item, dict) and "path" in item:
                files[item["path"]] = item["file_sha256"]
        for name, digest in files.items():
            data = (ROOT / name).read_bytes()
            _require(hashlib.sha256(data).hexdigest() == digest, "SOURCE_CHANGED_WHILE_COPYING")
            path = frozen / name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(data)
        assert_source(snapshot)
        load_declaration(frozen)
        env = {**os.environ, "PYTHONPATH": str(frozen / "src"), "GIT_CEILING_DIRECTORIES": str(frozen.parent)}
        command = [sys.executable, str(frozen / "scripts/run_native_axis_pilot.py"), "--worker",
            "--output", str(args.output), "--case-bundle", str(args.case_bundle.resolve())]
        budget = physical.PreflightBudget(**declaration["resource_budget"])
        status.update(status="running", worker_command=command, resource_budget=asdict(budget),
                      launch_preparation_seconds=time.perf_counter() - launched)
        write_json(args.output / "launcher-status.json", status)
        parent_timeout, hard_killed = False, False
        with (args.output / "worker.log").open("w") as log:
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
        complete = launch_complete(args.output, process.returncode, parent_timeout)
        status.update(status="completed" if complete else "failed", worker_returncode=process.returncode,
            eligible_candidate_count=1 if complete else 0, gradient_steps=1 if complete else None,
            parent_timeout_requested=parent_timeout, hard_killed=hard_killed,
            full_launcher_seconds=time.perf_counter() - launched,
            consumer_authority="Require completed launcher + hash-bound worker + pilot status; payload presence alone is not eligibility.")
        write_json(args.output / "launcher-status.json", status)
        if not complete:
            raise SystemExit(1)
    except Exception as error:
        status.update(status="failed", eligible_candidate_count=0, error_type=type(error).__name__, error=str(error),
                      full_launcher_seconds=time.perf_counter() - launched)
        write_json(args.output / "launcher-status.json", status)
        raise


if __name__ == "__main__":
    main()
