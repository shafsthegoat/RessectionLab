"""Independent release controls: generated inputs and fake workers only."""
from copy import deepcopy
from dataclasses import replace
import json
from pathlib import Path

import pytest
import torch

import frozen_preflight_supervisor as supervisor
import matched_estimate_methods as harness
import run_fixed_comparison as runner
from resectionlab.spatial_policy import parameter_hash


def test_both_condition_suites_precede_any_reference_access(monkeypatch, tmp_path):
    baseline = runner.generated_spec()
    models = {}
    for name in ("IL", "RL"):
        models[name] = torch.nn.Module().eval()
        models[name].register_buffer("identity", torch.tensor([float(name == "RL")]))
    record = json.loads((runner.BASE / "declaration.json").read_text())
    for name, model in models.items():
        record["proposal"]["frozen_checkpoints"][name]["parameter_hash"] = parameter_hash(model)
    events = []
    monkeypatch.setattr(runner, "load_checkpoint", lambda method, proposal: models[method])
    monkeypatch.setattr(runner, "generated_spec", lambda: baseline)
    monkeypatch.setattr(torch, "set_num_interop_threads", lambda value: None)
    def forbidden(*args, **kwargs):
        raise AssertionError("No checkpoint parser is permitted in this source control")
    monkeypatch.setattr(torch, "load", forbidden)
    def plan(spec, *, output, budget, **kwargs):
        index = len(events)
        condition = record["proposal"]["conditions_in_fixed_order"][index]
        assert tuple(tool.working_length_mm for tool in spec.tools) == tuple(condition["working_lengths_mm"])
        assert spec.fingerprint == replace(baseline, tools=spec.tools).fingerprint
        assert all(replace(tool, working_length_mm=old.working_length_mm) == old
                   for tool, old in zip(spec.tools, baseline.tools))
        assert budget == harness.MatchedBudget(**record["method_budget"])
        output.mkdir()
        suite = {"seal_hash": condition["name"], "methods": {}}
        for name in harness.METHODS:
            suite["methods"][name] = {"status": "complete"}
            (output / (name + "-strategy.json")).write_text("{}")
        (output / "planning-suite.json").write_text(json.dumps(suite))
        events.append(("plan", condition["name"]))
        return suite
    def evaluate(suite, spec, *, load_reference, output):
        assert events[:2] == [("plan", "original_tools"), ("plan", "actual_120mm_tools")]
        for condition in record["proposal"]["conditions_in_fixed_order"]:
            directory = tmp_path / condition["name"]
            assert (directory / "planning-suite.json").is_file()
            assert all((directory / (name + "-strategy.json")).is_file() for name in harness.METHODS)
        assert load_reference() is baseline.target.mask
        events.append(("evaluate", suite["seal_hash"]))
        return {"status": "fake_source_control_only"}
    monkeypatch.setattr(harness, "plan_matched_estimates", plan)
    monkeypatch.setattr(harness, "evaluate_matched_estimates", evaluate)
    runner.worker(record, tmp_path)
    result = json.loads((tmp_path / "result.json").read_text())
    assert result["status"] == "complete" and result["optimizer_updates"] == 0
    assert events == [("plan", "original_tools"), ("plan", "actual_120mm_tools"),
                      ("evaluate", "original_tools"), ("evaluate", "actual_120mm_tools")]


def test_metadata_preflight_checks_fixed_budgets_without_loading_checkpoint(monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError("Checkpoint parser must not run in metadata preflight")
    monkeypatch.setattr(torch, "load", forbidden)
    record = json.loads((runner.BASE / "declaration.json").read_text())
    runner.validate(record)
    changed = deepcopy(record)
    changed["method_budget"]["search_calls"] = 257
    with pytest.raises(ValueError, match="budgets changed"):
        runner.validate(changed)


@pytest.mark.parametrize("kwargs", [{"search_calls": True}, {"native_previews": 2.0},
                                   {"method_seconds": float("nan")}])
def test_budget_rejects_nonexact_or_nonfinite_limits(kwargs):
    with pytest.raises(ValueError, match="positive matched"):
        harness.MatchedBudget(**kwargs)


def test_missing_group_with_unconfirmed_worker_still_writes_negative(monkeypatch, tmp_path):
    class Unresolved:
        pid = 12345
        def poll(self): return None
        def terminate(self): raise AssertionError("Unexpected direct fallback")
        def kill(self): raise AssertionError("Unexpected direct fallback")
        def wait(self, **kwargs): raise AssertionError("No unbounded wait allowed")
    monkeypatch.setattr(supervisor.subprocess, "Popen", lambda *a, **k: Unresolved())
    clock = iter((0., 2., 3.))
    monkeypatch.setattr(supervisor.time, "perf_counter", lambda: next(clock))
    def missing(*args): raise ProcessLookupError("group missing")
    monkeypatch.setattr(supervisor.os, "killpg", missing)
    result = supervisor.supervise_worker(["fake"], tmp_path,
        {"max_wall_seconds": 1., "max_rss_bytes": 1024}, "source-control")
    assert result["status"] == "failed" and result["returncode"] is None
    assert result["worker_termination_confirmed"] is False
    assert result["unresolved_worker_pid"] == 12345
    assert json.loads((tmp_path / "supervisor-failure.json").read_text()) == result
