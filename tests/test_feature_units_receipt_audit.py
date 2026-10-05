"""Independent auditor regressions use fabricated receipts and three source cells."""
import importlib.util
import json
from pathlib import Path
import sys

import numpy as np
import pytest

ROOT = Path(__file__).parents[1]
spec = importlib.util.spec_from_file_location("independent_feature_receipt_audit", ROOT / "scripts/audit_native_feature_units.py")
audit = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = audit
spec.loader.exec_module(audit)


def test_running_study_refuses_before_opening_any_outcomes(tmp_path):
    (tmp_path / "study").mkdir()
    for path in (tmp_path / "study/status.json", tmp_path / "experiment-status.json"):
        path.write_text(json.dumps({"status": "running"}))
    inspector = audit.Audit(tmp_path, tmp_path / "unopened-source")
    with pytest.raises(audit.AuditError, match="outcomes remain unopened"):
        inspector.run()
    assert len(inspector.inputs) == 2


def test_expected_denominator_has_all_profiles_modes_seeds_and_shared_baselines():
    expected = audit.expected_candidates()
    assert len(expected) == 23 and {"STOP", "GREEDY", "SEARCH"} <= expected
    assert sum(":INITIAL:" in item for item in expected) == 6
    assert sum(":PATIENT_SCRATCH_RL:" in item for item in expected) == 6
    assert sum(":PROCEDURAL_PRETRAINED_ADAPTED:" in item for item in expected) == 6


@pytest.fixture
def record():
    return {"selected_selection_return": 0., "initial_selection_return": 0.,
        "selected_checkpoint_hash": "initial", "gradient_steps": 2,
        "optimization_environment_steps": 8, "selection_environment_steps": 12,
        "elapsed_seconds": 30.3, "initialization_seconds": 100., "selection_seconds": 23.,
        "final_checkpoint_export_seconds": .01,
        "selection_history": [
            {"world_count": 2, "checkpoint_hash": "initial", "mean_return": 0.,
             "gradient_steps": 0, "optimization_environment_steps": 0, "panel_elapsed_seconds": 12.},
            {"world_count": 2, "checkpoint_hash": "later", "mean_return": 0.,
             "gradient_steps": 2, "optimization_environment_steps": 8, "panel_elapsed_seconds": 10.}]}


def test_completed_initial_zero_is_eligible_and_ties_keep_earliest(record):
    assert audit.check_selection(record, 2)["checkpoint_hash"] == "initial"
    record["selected_checkpoint_hash"] = "later"
    with pytest.raises(audit.AuditError, match="tie rule"):
        audit.check_selection(record, 2)


@pytest.mark.parametrize("change", ["partial", "nan", "missing", "future"])
def test_selection_receipt_cannot_hide_partial_unknown_or_future_checkpoint(record, change):
    if change == "partial":
        record["selection_history"][0]["world_count"] = 1
    elif change == "nan":
        record["selection_history"][0]["mean_return"] = float("nan")
    elif change == "missing":
        record["selected_selection_return"] = None
    else:
        record["selection_history"][1]["gradient_steps"] = 3
    with pytest.raises(audit.AuditError):
        audit.check_selection(record, 2)


def arm():
    return {"learner_wall_budget_overshoot_seconds": .3, "initial_selection_seconds": 12.,
        "later_and_partial_selection_seconds": 11., "incomplete_selection_seconds": 1.,
        "trainer_call_seconds": 131., "preparation_seconds": 4.,
        "candidate_extraction_seconds": 2., "complete_arm_seconds_before_independent_audit": 138.}


def test_extra_selection_is_in_learner_clock_and_atomic_overshoot_is_disclosed(record):
    result = audit.check_timing(record, arm(), {"max_wall_seconds": 30., "max_gradient_steps": 32, "max_environment_steps": 256})
    assert result["additional_selection_seconds"] == 11.
    assert result["initialization_seconds"] == 100.
    assert result["cooperative_wall_overshoot_seconds"] == pytest.approx(.3)
    record["elapsed_seconds"] = 20.
    with pytest.raises(audit.AuditError, match="omits initial or subsequent"):
        audit.check_timing(record, arm(), {"max_wall_seconds": 30., "max_gradient_steps": 32, "max_environment_steps": 256})


def test_cap_overshoot_cannot_be_silently_clipped_to_zero(record):
    values = arm()
    values["learner_wall_budget_overshoot_seconds"] = 0.
    with pytest.raises(audit.AuditError, match="overshoot"):
        audit.check_timing(record, values, {"max_wall_seconds": 30., "max_gradient_steps": 32, "max_environment_steps": 256})


@pytest.mark.parametrize("profile_id", ["RAW", "FEATURE_UNITS"])
def test_independent_tensor_profile_checker_rejects_rehashed_buffer_tamper(profile_id):
    declaration = json.loads((ROOT / "manifests/experiments/procedural-native-feature-units-v1.json").read_text())
    profile = declaration["profiles"][profile_id]
    weights = {name: np.ones(shape, np.float32) for name, shape in {
        "actor.0.weight": (16, 21), "actor.0.bias": (16,), "actor.2.weight": (1, 16), "actor.2.bias": (1,),
        "value.0.weight": (16, 6), "value.0.bias": (16,), "value.2.weight": (1, 16), "value.2.bias": (1,)}.items()}
    if profile_id == "FEATURE_UNITS":
        weights["action_divisors"] = np.asarray(profile["action_divisors"], np.float32)
    checkpoint = {"dimensions": [15, 6, 16], "input_profile": {k: v for k, v in profile.items() if k != "profile_content_hash"},
        "input_profile_hash": profile["profile_content_hash"], "policy": weights,
        "policy_hash": audit.tensor_hash(weights)}
    audit.check_profile(checkpoint, profile)
    if profile_id == "RAW":
        weights["action_divisors"] = np.ones(15, np.float32)
    else:
        weights["action_divisors"][1] = 999
    checkpoint["policy_hash"] = audit.tensor_hash(weights)
    with pytest.raises(audit.AuditError, match="buffer|keys"):
        audit.check_profile(checkpoint, profile)


def test_world_manifest_role_or_seed_relabeling_is_detected():
    roles = ("optimization", "selection", "final_evaluation", "stress")
    panels = {role: {"role": role, "seeds": [i]} for i, role in enumerate(roles)}
    audit.check_partitions(panels)
    panels["final_evaluation"]["seeds"] = [0]
    with pytest.raises(audit.AuditError, match="overlap"):
        audit.check_partitions(panels)


def cell_metrics():
    target = np.zeros((3, 1, 1), bool)
    target[1, 0, 0] = True
    history = []
    for index, tool in enumerate(("A", "A", "B")):
        history.append({"removed_indices_native": [[index, 0, 0]],
            "contact_indices_native": [[index, 0, 0], [2, 0, 0]] if index != 2 else [[2, 0, 0]],
            "tip_mm": [0., 0., 1.], "entry_mm": [0., 0., 0.], "tool_id": tool,
            "target_removed_mm3": 2. if index == 1 else 0.,
            "normal_removed_mm3": 0. if index == 1 else 2.,
            "partial_normal_contact_mm3": 2. if index == 0 else 0.,
            "motor_surrogate_delta": 0., "language_surrogate_delta": 0.,
            "partial_motor_contact_surrogate": 0., "partial_language_contact_surrogate": 0.})
    metrics = {"history": history, "total_reward": .73,
        "simulated_removed_target_volume_mm3": 2., "simulated_removed_normal_volume_mm3": 4.,
        "cumulative_partial_normal_contact_mm3": 2., "modeled_residual_target_volume_mm3": 0.,
        "clinical_deficit_probability": None, "functional_evidence_available": {"motor": False, "language": False},
        "motor_surrogate": None, "language_surrogate": None,
        "removed_by_compartment_mm3": {"target": 2.}, "residual_by_compartment_mm3": {"target": 0.}}
    reward = {"target_per_mm3": 1., "normal_per_mm3": .2, "action_cost": .01,
        "motion_per_mm": .02, "tool_change_cost": .3}
    return metrics, {"target": target}, reward


def test_independent_score_counts_partial_contact_once_even_if_later_removed():
    metrics, compartments, reward = cell_metrics()
    result = audit.recompute_score(metrics, compartments, 2., reward, .05)
    assert result["score"] == pytest.approx(.73)
    assert result["cumulative_partial_normal_contact_mm3"] == 2.


@pytest.mark.parametrize("change", ["partial_twice", "removed_twice", "inflated_target", "clinical_zero"])
def test_recomputed_score_rejects_bookkeeping_or_unknown_evidence_relabeling(change):
    metrics, compartments, reward = cell_metrics()
    if change == "partial_twice":
        metrics["history"][1]["partial_normal_contact_mm3"] = 2.
    elif change == "removed_twice":
        metrics["history"][1]["removed_indices_native"] = [[0, 0, 0]]
    elif change == "inflated_target":
        metrics["simulated_removed_target_volume_mm3"] = 3.
    else:
        metrics["clinical_deficit_probability"] = 0.
    with pytest.raises(audit.AuditError):
        audit.recompute_score(metrics, compartments, 2., reward, .05)
