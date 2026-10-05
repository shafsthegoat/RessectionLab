"""Static-record diagnostic checks; no episodes, backward passes or updates."""
import importlib.util
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np
import pytest
import torch


@pytest.fixture(scope="module")
def diagnostic():
    directory = Path(__file__).resolve().parents[1] / "scripts"
    sys.path.insert(0, str(directory))
    spec = importlib.util.spec_from_file_location("mechanism_diagnostic_test", directory / "diagnose_feature_unit_mechanisms.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize("state", ["running", "failed", "invalidated"])
def test_incomplete_study_refused_without_opening_any_rankings(diagnostic, tmp_path, state):
    (tmp_path / "study").mkdir()
    for name in ("experiment-status.json", "study/status.json"):
        (tmp_path / name).write_text(json.dumps({"status": state, "final_worlds_used": False}))
    # No summary exists: refusal must precede opening it.
    with pytest.raises(ValueError, match="partial rankings remain unopened"):
        diagnostic.completed_records(tmp_path)


def test_clipping_algebra_separates_known_actor_and_value_norms(diagnostic):
    coefficient = 1. / (5. + 1e-6)
    record = {"optimization_history": [{"gradient_steps": 1, "gradient_norm_before_clip": 5.,
        "actor_gradient_norm_after_clip": 3. * coefficient, "mean_return": 12.}]}
    result = diagnostic.clipping_decomposition(record, {"config": {"max_gradient_norm": 1.}})
    row = result["rows"][0]
    assert row["actor_norm_before_clip_inferred"] == pytest.approx(3.)
    assert row["critic_norm_before_clip_inferred"] == pytest.approx(4.)
    assert row["critic_fraction_of_squared_global_norm_inferred"] == pytest.approx(.64)
    assert row["additional_actor_shrink_ratio_from_joint_norm_inferred"] == pytest.approx((3.+1e-6)/(5.+1e-6))
    assert result["new_gradients_computed"] == 0


def test_inconsistent_logged_norms_are_not_silently_clamped(diagnostic):
    record = {"optimization_history": [{"gradient_steps": 1, "gradient_norm_before_clip": 1.,
        "actor_gradient_norm_after_clip": 10., "mean_return": 0.}]}
    with pytest.raises(ValueError, match="inconsistent"):
        diagnostic.clipping_decomposition(record, {"config": {"max_gradient_norm": 5.}})


def test_actor_measurement_uses_actual_scaled_inputs_without_gradients(diagnostic):
    from resectionlab.learning import MaskedPatientPolicy, policy_hash
    policy = MaskedPatientPolicy(15, 6, 16, input_profile="FEATURE_UNITS")
    policy.requires_grad_(False)
    obs = SimpleNamespace(action_features=np.full((3, 15), 648., dtype=np.float32),
        state_features=np.zeros(6, dtype=np.float32), action_ids=("STOP", "CUT", "MASKED"),
        action_mask=np.array([True, True, False]))
    before = policy_hash(policy)
    result = diagnostic.actor_statistics(policy, obs)
    assert result["raw_input_abs_max_by_feature"][1] == 648.
    assert result["actor_input_abs_max_by_feature"][1] == 1.
    assert len(result["rows"]) == 2
    assert policy_hash(policy) == before
    assert all(parameter.grad is None and not parameter.requires_grad for parameter in policy.parameters())
