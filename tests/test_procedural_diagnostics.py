"""The post hoc diagnostic is read-only and refuses incomplete comparisons."""
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch


@pytest.fixture(scope="module")
def diagnostic():
    path = Path(__file__).resolve().parents[1] / "scripts/diagnose_procedural_transfer.py"
    spec = importlib.util.spec_from_file_location("procedural_diagnostic_test", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize("status", ["running", "failed", "invalidated"])
def test_incomplete_comparison_rejected_before_runtime_or_model_loading(diagnostic, tmp_path, status):
    (tmp_path / "comparison").mkdir()
    for name in ("experiment-status.json", "summary.json", "comparison/status.json"):
        (tmp_path / name).write_text(json.dumps({"status": status, "final_worlds_used": False}))
    with pytest.raises(ValueError, match="completed run"):
        diagnostic.require_completed_run(tmp_path)


def test_actual_entropy_mask_and_tanh_saturation_are_measured_without_gradients(diagnostic):
    class Policy(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.actor = torch.nn.Sequential(torch.nn.Linear(2, 1), torch.nn.Tanh(), torch.nn.Linear(1, 1))
            with torch.no_grad():
                self.actor[0].weight.fill_(1)
                self.actor[0].bias.zero_()
                self.actor[2].weight.fill_(1)
                self.actor[2].bias.zero_()
        def forward(self, observation):
            inputs = torch.tensor(np.column_stack((observation.action_features[:, 0],
                                                  np.repeat(observation.state_features[0], 3))), dtype=torch.float32)
            logits = self.actor(inputs).flatten()
            return logits.masked_fill(~torch.tensor(observation.action_mask), -torch.inf), torch.tensor(0.)
    policy = Policy()
    observation = SimpleNamespace(action_ids=("STOP", "CUT", "MASKED"), action_features=np.array([[0.], [10.], [100.]]),
                                  state_features=np.array([0.]), action_mask=np.array([True, True, False]))
    before = {name: value.detach().clone() for name, value in policy.state_dict().items()}
    result = diagnostic.actor_statistics(policy, observation, ("physical_volume", "state:fraction"))
    expected = torch.softmax(torch.tensor([0., 1.]), 0)
    assert result["stop_probability"] == pytest.approx(float(expected[0]))
    assert result["entropy_nats"] == pytest.approx(float(-(expected * expected.log()).sum()))
    assert result["rows"][0]["tanh_abs_ge_099_fraction"] == 0
    assert result["rows"][1]["tanh_abs_ge_099_fraction"] == 1
    assert [row["action_id"] for row in result["rows"]] == ["STOP", "CUT"]
    assert all(parameter.grad is None for parameter in policy.parameters())
    assert all(torch.equal(value, before[name]) for name, value in policy.state_dict().items())


def test_zero_update_statistics_remain_missing_instead_of_invented(diagnostic):
    record = {"gradient_steps": 0, "optimization_environment_steps": 0, "selection_environment_steps": 1,
        "elapsed_seconds": .1, "actor_parameters_changed": False, "selected_checkpoint_hash": "initial",
        "initial_checkpoint_hash": "initial", "initial_selection_return": 0., "selected_selection_return": 0.,
        "selection_history": [], "optimization_history": []}
    result = diagnostic.gradient_statistics(record)
    assert result["selected_is_initial"] is True
    assert all(value is None for value in result["statistics"].values())
