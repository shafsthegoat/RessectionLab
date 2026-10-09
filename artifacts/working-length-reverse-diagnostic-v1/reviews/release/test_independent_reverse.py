"""Worker-path controls with DTOs/scripted models and no native reconstruction."""
from dataclasses import dataclass
import json
from types import SimpleNamespace

import numpy as np
import pytest
import torch

import run_reverse as runner
import run_fixed_comparison as shared
import resectionlab.research_estimate_planning as planning
from resectionlab.geometry import ToolGeometry
from resectionlab.native_resection import NativeResectionEngine
from resectionlab.native_spatial_task import NativeSpatialTask
from test_reverse import observation, Scripted


@pytest.mark.parametrize("bad_identity", [False, True])
def test_worker_has_no_action_execution_and_records_failures(monkeypatch, tmp_path, observation, bad_identity):
    @dataclass(frozen=True)
    class Spec:
        tools: tuple
    spec = Spec((ToolGeometry("short-wide-opener", 2.25, .45, 2.2, 35., .75),
                 ToolGeometry("long-narrow-cutter", .9, 1.1, 12., 35., 3.)))
    class Model(Scripted):
        def _inputs(self, obs):
            return (torch.zeros(1), torch.zeros(1), torch.ones(1),
                    torch.tensor(np.array(obs.action_geometry)), torch.zeros(1),
                    torch.zeros(1), torch.ones(len(obs.action_ids), dtype=torch.bool))
    model = Model()
    calls = {"loads": 0, "reconstructions": 0}
    def forbidden(*args, **kwargs):
        raise AssertionError("Actual checkpoints, native previews and action execution are forbidden")
    monkeypatch.setattr(torch, "load", forbidden)
    monkeypatch.setattr(torch, "set_num_interop_threads", lambda value: None)
    monkeypatch.setattr(NativeResectionEngine, "preview_stroke", forbidden)
    monkeypatch.setattr(NativeSpatialTask, "step", forbidden)
    monkeypatch.setattr(NativeSpatialTask, "advance_planning", forbidden)
    monkeypatch.setattr(shared, "generated_spec", lambda: spec)
    def preflight(actual):
        calls["reconstructions"] += 1
        assert [tool.working_length_mm for tool in actual.tools] == [120., 120.]
        return SimpleNamespace(observation=lambda: observation)
    monkeypatch.setattr(planning, "_preflight_nominal_task", preflight)
    def load(method, lineage):
        assert method == "RL"
        calls["loads"] += 1
        return model
    monkeypatch.setattr(shared, "load_checkpoint", load)
    monkeypatch.setattr(runner, "ROOT", tmp_path)
    decision = {"observation_hash": "wrong" if bad_identity else observation.fingerprint,
                "action_ids": list(observation.action_ids), "legal_logits": [0.]*8,
                "stop_minus_best_movement": 0.}
    (tmp_path / "suite.json").write_text(json.dumps({"methods": {"RL": {"details": {"decisions": [decision]}}}}))
    record = {"scope": "scripted source control only", "saved_suite": "suite.json", "checkpoint_lineage": {}}
    if bad_identity:
        with pytest.raises(ValueError, match="differs from the saved"):
            runner.worker(record, tmp_path)
    else:
        runner.worker(record, tmp_path)
    result = json.loads((tmp_path / "result.json").read_text())
    assert result["executed_actions"] == result["search_calls"] == result["optimizer_updates"] == 0
    assert calls["reconstructions"] == 1
    assert calls["loads"] == model.calls == (0 if bad_identity else 1)
    assert result["forward_calls"] == {"attempted": model.calls, "completed": model.calls}
    assert result["status"] == ("failed" if bad_identity else "complete")
    assert not result["planning_efficacy_measured"]
    if not bad_identity:
        identity = json.loads((tmp_path / "input-identity.json").read_text())
        assert not identity["input_geometry_certified"] and not identity["strategy_execution_permitted"]
        with np.load(tmp_path / "generated-permitted-inputs.npz", allow_pickle=False) as arrays:
            assert np.count_nonzero(arrays["changed_action_geometry"] != arrays["original_action_geometry"]) == 7
            assert np.array_equal(arrays["original_action_geometry"], observation.action_geometry)


def test_failed_forward_counts_attempt_once_and_removes_hook(observation):
    class Failing(Scripted):
        def forward(self, observation):
            self.calls += 1
            raise RuntimeError("scripted forward failure")
    model = Failing()
    counts = {"attempted": 0, "completed": 0}
    with pytest.raises(RuntimeError, match="scripted forward failure"):
        runner.one_forward(model, observation, counts)
    assert counts == {"attempted": 1, "completed": 0} and model.calls == 1
    assert not model._forward_pre_hooks
