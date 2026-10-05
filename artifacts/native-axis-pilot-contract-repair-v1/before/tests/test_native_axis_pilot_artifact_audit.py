"""Independent post-run audit tests; constructed data only, no public rollout."""
import copy
import gzip
import json
from pathlib import Path
import shutil
import sys

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import audit_native_axis_pilot as audit
from test_native_axis_pilot import completed


def packed(values, dtype="float32"):
    array = np.asarray(values, dtype=dtype)
    return {"dtype": array.dtype.str, "shape": list(array.shape), "values": array.tolist()}


def raw_fixture(role="selection"):
    shapes = {"actor.0.weight": (16, 21), "actor.0.bias": (16,), "actor.2.weight": (1, 16),
              "actor.2.bias": (1,), "value.0.weight": (16, 6), "value.0.bias": (16,),
              "value.2.weight": (1, 16), "value.2.bias": (1,)}
    weights = {name: np.zeros(shape, np.float32) for name, shape in shapes.items()}
    actions = np.arange(45, dtype=np.float32).reshape(3, 15) / 8
    state = np.arange(6, dtype=np.float32) / 8
    payload = {"version": "learner-decision-observer-v1", "role": role, "forward_evaluated": True,
        "selected_index": 0, "selected_action_id": "STOP", "decision_rule": "deterministic_argmax" if role == "selection" else "sampled_categorical",
        "forced_reason": None, "logits": packed([0., 0., 0.]), "value": 0.,
        "inputs": {"action_ids": ["STOP", "a", "b"], "action_mask": packed([True, True, True], "bool"),
            "source_action_features": packed(actions), "source_state_features": packed(state),
            "action_features": packed(actions), "actor_action_features": packed(actions), "state_features": packed(state)}}
    return weights, payload


def test_independent_numpy_forward_matches_explicit_scalar_formula():
    weights, payload = raw_fixture()
    weights["actor.0.weight"][0, 0] = .5
    weights["actor.0.weight"][0, 20] = -.25
    weights["actor.0.bias"][0] = .125
    weights["actor.2.weight"][0, 0] = .75
    weights["actor.2.bias"][0] = -.125
    weights["value.0.weight"][0, 1] = .5
    weights["value.2.weight"][0, 0] = .25
    actions = np.asarray(payload["inputs"]["action_features"]["values"], np.float32)
    state = np.asarray(payload["inputs"]["state_features"]["values"], np.float32)
    logits, value = audit.direct_raw_forward(weights, actions, state, np.ones(3, bool))
    expected = .75 * np.tanh(.5 * actions[:, 0] - .25 * state[5] + .125) - .125
    np.testing.assert_allclose(logits, expected, atol=1e-7, rtol=1e-6)
    assert value == pytest.approx(.25 * np.tanh(.5 * state[1]), abs=1e-7)


def test_first_maximum_tie_uses_independent_logits_not_tolerated_saved_values():
    weights, payload = raw_fixture()
    payload["logits"]["values"][1] = float(np.float32(1e-7))  # within forward tolerance
    payload["selected_index"] = 1
    payload["selected_action_id"] = "a"
    with pytest.raises(audit.AuditError, match="independently recomputed earliest-row argmax"):
        audit.verify_forward(payload, weights)


def test_every_tolerated_difference_is_reported():
    weights, payload = raw_fixture()
    payload["logits"]["values"] = [float(np.float32(1e-7)), 0., 0.]
    payload["value"] = float(np.float32(2e-7))
    result = audit.verify_forward(payload, weights)
    assert result["nonzero_logit_difference_count"] == 1
    assert result["max_logit_abs_difference"] == result["logit_abs_differences"][0] > 0
    assert result["value_abs_difference"] > 0 and result["argmax_verified"]


@pytest.mark.parametrize("field", ["logits", "value"])
def test_material_captured_forward_change_rejected(field):
    weights, payload = raw_fixture()
    if field == "logits":
        payload["logits"]["values"][0] = 1.
    else:
        payload["value"] = 1.
    with pytest.raises(audit.AuditError, match="checkpoint tensor operations"):
        audit.verify_forward(payload, weights)


def test_sampled_choice_checks_eligibility_without_claiming_rng_replay():
    weights, payload = raw_fixture("optimization")
    payload.update(selected_index=2, selected_action_id="b")
    result = audit.verify_forward(payload, weights)
    assert not result["argmax_verified"] and not result["rng_replayed"]
    payload["inputs"]["action_mask"]["values"][2] = False
    with pytest.raises(audit.AuditError, match="not eligible"):
        audit.verify_forward(payload, weights)


def test_forced_no_forward_preserves_absent_outputs():
    weights, payload = raw_fixture()
    payload.update(forward_evaluated=False, logits=None, value=None, decision_rule="forced_stop", forced_reason="episode_step_limit")
    for key in ("action_features", "actor_action_features", "state_features"):
        payload["inputs"][key] = None
    result = audit.verify_forward(payload, weights)
    assert result["evaluated"] is False and result["max_logit_abs_difference"] is None
    payload["value"] = 0.
    with pytest.raises(audit.AuditError, match="invented outputs"):
        audit.verify_forward(payload, weights)


@pytest.mark.parametrize("attack", ["transform", "dtype", "shape", "extra_tensor", "nonfinite_tensor", "truncated_ids"])
def test_raw_tensor_and_observation_semantics_are_strict(attack):
    weights, payload = raw_fixture()
    if attack == "transform":
        payload["inputs"]["actor_action_features"]["values"][0][0] = 1.
    elif attack == "dtype":
        payload["inputs"]["action_features"]["dtype"] = "float64"
    elif attack == "shape":
        payload["inputs"]["action_features"]["shape"] = [1, 45]
    elif attack == "extra_tensor":
        weights["action_divisors"] = np.ones(15, np.float32)
    elif attack == "nonfinite_tensor":
        weights["actor.0.weight"][0, 0] = np.nan
    else:
        payload["inputs"]["action_ids"].pop()
    with pytest.raises(audit.AuditError):
        audit.verify_forward(payload, weights)


def test_all_actual_tiny_captured_forwards_verify_without_rng_or_policy_class(completed, monkeypatch):
    import torch
    import resectionlab.learning as learning
    monkeypatch.setattr(learning.MaskedPatientPolicy, "forward", lambda *a, **k: pytest.fail("Independent auditor must not call policy class"))
    torch_state = torch.get_rng_state().clone()
    numpy_state = np.random.get_state()
    path = completed["output"]
    frozen = json.loads((path / "history-freeze.json").read_text())
    states = audit.check_checkpoints(audit.Reader(), path / "learner", completed["declaration"], completed["result"], frozen)
    graph = audit.check_episode_graph(completed["accounting"], frozen, completed["declaration"])
    receipts = [audit.verify_forward(decision["payload"], states[episode["update"]])
                for episode, decisions, _ in graph for decision in decisions]
    assert receipts and all(receipt["evaluated"] for receipt in receipts)
    assert max(row["max_logit_abs_difference"] for row in receipts) < 1e-4
    assert torch.equal(torch.get_rng_state(), torch_state)
    next_numpy_state = np.random.get_state()
    assert numpy_state[0] == next_numpy_state[0] and np.array_equal(numpy_state[1], next_numpy_state[1])
    assert numpy_state[2:] == next_numpy_state[2:]


def test_coherently_rehashed_nonfinite_optimizer_state_rejected(completed, tmp_path):
    import torch
    folder = tmp_path / "learner"
    shutil.copytree(completed["output"] / "learner", folder)
    frozen = json.loads((completed["output"] / "history-freeze.json").read_text())
    checkpoint = torch.load(folder / "checkpoint.pt", map_location="cpu", weights_only=True)
    state = next(iter(checkpoint["optimizer"]["state"].values()))
    state["exp_avg"].flatten()[0] = float("nan")
    torch.save(checkpoint, folder / "checkpoint.pt")
    frozen["checkpoints"]["files"]["checkpoint.pt"] = audit.file_hash(folder / "checkpoint.pt")
    with pytest.raises(audit.AuditError, match="Nonfinite saved optimizer state"):
        audit.check_checkpoints(audit.Reader(), folder, completed["declaration"], completed["result"], frozen)


@pytest.mark.parametrize("attack", ["lost_join", "wrong_update_hash", "wrong_event_instance", "wrong_world_role"])
def test_independent_episode_graph_rejects_resealed_join_changes(completed, attack):
    frozen = json.loads((completed["output"] / "history-freeze.json").read_text())
    journal = copy.deepcopy(completed["accounting"])
    if attack == "lost_join":
        frozen["episodes"][0]["decisions"].pop()
    elif attack == "wrong_update_hash":
        frozen["episodes"][-1]["decisions"][0]["policy_hash"] = frozen["checkpoints"]["initial_policy_hash"]
    else:
        row = next(row for row in journal["events"] if row["kind"] == "transition")
        row["instance" if attack == "wrong_event_instance" else "role"] = -1 if attack == "wrong_event_instance" else "final_evaluation"
        journal["receipt_hash"] = audit.accounting_hash({k: v for k, v in journal.items() if k != "receipt_hash"})
        frozen["accounting_receipt_hash"] = journal["receipt_hash"]
    with pytest.raises(audit.AuditError):
        audit.check_episode_graph(journal, frozen, completed["declaration"])


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value))


def publication_fixture(tmp_path):
    payload = {"private_test": True}
    write_json(tmp_path / "pilot/candidate-record.json", payload)
    digest = audit.file_hash(tmp_path / "pilot/candidate-record.json")
    base = {"status": "completed", "eligible_candidate_count": 1, "final_worlds_used": False, "stress_worlds_used": False}
    pilot = {**base, "gradient_steps": 1, "candidate_record_sha256": digest}
    write_json(tmp_path / "pilot/status.json", pilot)
    worker = {**base, "candidate_record_sha256": digest, "pilot_status_sha256": audit.file_hash(tmp_path / "pilot/status.json")}
    launcher = {**base, "gradient_steps": 1, "worker_returncode": 0, "parent_timeout_requested": False, "hard_killed": False}
    write_json(tmp_path / "worker-status.json", worker)
    write_json(tmp_path / "launcher-status.json", launcher)
    return payload


def test_completed_authority_requires_exact_payload_and_worker_chain(tmp_path):
    expected = publication_fixture(tmp_path)
    payload, _ = audit.check_authorities(audit.Reader(), tmp_path)
    assert payload == expected
    with (tmp_path / "pilot/candidate-record.json").open("a") as stream:
        stream.write(" ")
    with pytest.raises(audit.AuditError, match="payload bytes"):
        audit.check_authorities(audit.Reader(), tmp_path)


@pytest.mark.parametrize("attack", ["parent_timeout", "hard_kill", "worker_return", "world_role", "worker_chain"])
def test_corrupted_completed_authorities_rejected(tmp_path, attack):
    publication_fixture(tmp_path)
    name = "worker-status.json" if attack == "worker_chain" else "launcher-status.json"
    record = json.loads((tmp_path / name).read_text())
    key, value = {"parent_timeout": ("parent_timeout_requested", True), "hard_kill": ("hard_killed", True),
                  "worker_return": ("worker_returncode", 9), "world_role": ("final_worlds_used", True),
                  "worker_chain": ("pilot_status_sha256", "changed")}[attack]
    record[key] = value
    write_json(tmp_path / name, record)
    with pytest.raises(audit.AuditError):
        audit.check_authorities(audit.Reader(), tmp_path)


def test_failed_outer_authority_keeps_complete_inner_payload_ineligible(tmp_path):
    publication_fixture(tmp_path)
    write_json(tmp_path / "launcher-status.json", {"status": "failed", "eligible_candidate_count": 0})
    payload, summary = audit.check_authorities(audit.Reader(), tmp_path)
    assert payload is None and summary["candidate_eligible"] is False
    assert summary["audit_status"] == "incomplete_attempt_retained"
    assert (tmp_path / "pilot/candidate-record.json").exists()


def test_missing_outer_authority_is_incomplete_not_selected(tmp_path):
    publication_fixture(tmp_path)
    (tmp_path / "launcher-status.json").unlink()
    payload, summary = audit.check_authorities(audit.Reader(), tmp_path)
    assert payload is None and summary["candidate_eligible"] is False


def test_gzip_fallback_has_identical_bytes_and_rejects_disagreement(tmp_path):
    publication_fixture(tmp_path)
    path = tmp_path / "pilot/candidate-record.json"
    original = path.read_bytes()
    Path(str(path) + ".gz").write_bytes(gzip.compress(original, mtime=0))
    path.unlink()
    assert audit.Reader().bytes(path) == original
    payload, _ = audit.check_authorities(audit.Reader(), tmp_path)
    assert payload["private_test"]
    path.write_text("{}")
    with pytest.raises(audit.AuditError, match="Raw/gzip disagreement"):
        audit.Reader().bytes(path)


def cell_score_fixture(volume=1.):
    tissue = np.ones((3, 3, 3), bool)
    target = np.zeros_like(tissue)
    target[1, 1, 1] = True
    affine = np.diag([volume, 1., 1., 1.])
    model = {"case_hash": "source", "native_config_hash": "geometry", "partial_contact_weight": .05,
             "physical_reward": {"target_per_mm3": 1., "normal_per_mm3": .2, "action_cost": .02,
                                 "motion_per_mm": 0., "tool_change_cost": .02}}
    transitions = []
    for index, (cells, touched, tool) in enumerate((([[1, 1, 1]], [[1, 1, 1], [0, 0, 1]], "a"),
                                                  ([[0, 0, 1]], [[0, 0, 1]], "b"))):
        info = {"action_id": str(index), "source_hash": "source", "decision_model_hash": "geometry",
                "tissue_support_provenance": "synthetic", "native_affine": affine.tolist(),
                "removed_indices_native": cells, "contact_indices_native": touched,
                "microsteps": [{"removed_indices_native": cells, "contact_indices_native": touched}],
                "removed_volume_mm3": volume, "target_removed_mm3": volume if index == 0 else 0.,
                "normal_removed_mm3": volume if index else 0., "partial_normal_contact_mm3": volume if index == 0 else 0.,
                "motor_surrogate_delta": 0., "language_surrogate_delta": 0.,
                "partial_motor_contact_surrogate": 0., "partial_language_contact_surrogate": 0.,
                "tip_mm": [1., 1., 1.], "entry_mm": [1., 1., 0.], "tool_id": tool}
        reward = .99 * volume - .02 if index == 0 else -.2 * volume - .04
        transitions.append({"native_commit": True, "action_id": str(index), "info": info, "reward": reward,
                            "removed_cells": 1, "before_revision": index, "after_revision": index + 1,
                            "before_state_hash": str(index), "after_state_hash": str(index + 1), "terminated": index == 1})
    return transitions, {"target": target}, tissue, affine, model


@pytest.mark.parametrize("volume", [1., 6.])
def test_source_cell_reward_preserves_previously_charged_partial_contact(volume):
    rows, compartments, tissue, affine, model = cell_score_fixture(volume)
    score = audit.score_episode(rows, compartments, tissue, affine, model, "synthetic")
    assert score["return"] == pytest.approx(.79 * volume - .06)
    assert score["target_removed_mm3"] == score["normal_removed_mm3"] == score["partial_normal_contact_mm3"] == volume
    assert score["retained_contact_mm3"] == 0. and score["residual_target_mm3"] == 0.
    assert score["clinical_deficit_probability"] is None
    assert score["removed_by_compartment_mm3"] == {"target": volume}


@pytest.mark.parametrize("attack", ["duplicate_removal", "unsupported_tissue", "micro_union", "reward", "wrong_affine"])
def test_source_cell_accounting_rejects_fabricated_credit(attack):
    rows, compartments, tissue, affine, model = cell_score_fixture()
    if attack == "duplicate_removal":
        rows[1]["info"]["removed_indices_native"] = [[1, 1, 1]]
        rows[1]["info"]["contact_indices_native"] = [[1, 1, 1]]
        rows[1]["info"]["microsteps"] = [{"removed_indices_native": [[1, 1, 1]], "contact_indices_native": [[1, 1, 1]]}]
    elif attack == "unsupported_tissue":
        tissue[1, 1, 1] = False
    elif attack == "micro_union":
        rows[0]["info"]["microsteps"][0]["removed_indices_native"] = []
    elif attack == "reward":
        rows[0]["reward"] += 10
    else:
        rows[0]["info"]["native_affine"][0][0] = 9
    with pytest.raises(audit.AuditError):
        audit.score_episode(rows, compartments, tissue, affine, model, "synthetic")
