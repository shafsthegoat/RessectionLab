"""Independent orchestration attacks; tiny generated anatomy, no patient runs.

These exercise publication and resource boundaries separately from the native
geometry tests. A final completed receipt is the authority for eligibility.
"""
from dataclasses import replace
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import preflight_native_axis as runner
from resectionlab.core import CaseData, SourceRef
from resectionlab.geometry import AccessWindow
from resectionlab.native_axis_simulation import AxisColumnNativeSimulator
from resectionlab.native_proposals import AxisColumnProposalConfig
from resectionlab.native_resection import NATIVE_GENERIC_TOOLS, NativeResectionConfig
from resectionlab.worlds import WorldGeneratorConfig, generate_partitions


def setup_run(tmp_path):
    tissue = np.ones((7, 7, 8), bool)
    target = np.zeros_like(tissue)
    target[1:6, 1:6, 2:7] = True
    case = CaseData("axis-publication-attack", tissue.astype(float), {"target": target}, np.eye(4),
        (SourceRef("analytic", "simulated:publication-attack", provenance="simulated"),), brain_mask=tissue)
    config = NativeResectionConfig(tissue, target.astype(np.int16), np.eye(4),
        AccessWindow((3, 3, -.5), (0, 0, 1), 4), NATIVE_GENERIC_TOOLS,
        case.semantic_hash, "explicit generated tissue", case_id=case.case_id)
    panels = generate_partitions(case.semantic_hash, WorldGeneratorConfig(), 71,
        optimization=1, selection=2, final_evaluation=1, stress=1, planning_hash=case.planning_hash)
    source = tmp_path / "source"
    (source / "scripts").mkdir(parents=True)
    (source / "scripts/preflight_native_axis.py").write_text("# independent test snapshot\n")
    guard = runner.ResourceGuard(runner.PreflightBudget(120., 4 * 1024**3, 1.))
    factory = lambda callback: AxisColumnNativeSimulator(config,
        proposal_config=AxisColumnProposalConfig(((0, 0),), 2), max_steps=1, cancelled=callback)
    output = tmp_path / "attempt"

    def execute():
        return runner.run_preflight(case, factory, panels.optimization, panels.selection,
                                    output, guard, source_root=source)
    return execute, output, guard, source


def assert_ineligible(output):
    status = json.loads((output / "status.json").read_text())
    assert status["status"] in {"failed", "cancelled"}
    assert status["eligible_candidate_count"] == 0
    assert status["gradient_steps"] == 0
    assert not status["final_worlds_used"] and not status["stress_worlds_used"]
    assert (output / "failure-state.json").exists()


@pytest.mark.parametrize("failed_write", ["candidate-records.json", "completed-status"])
def test_export_failure_cannot_preserve_positive_eligibility(tmp_path, monkeypatch, failed_write):
    execute, output, _, _ = setup_run(tmp_path)
    original = runner.write_json
    triggered = []

    def fail_one_write(path, value):
        match = (path.name == failed_write or (failed_write == "completed-status"
                 and path.name == "status.json" and value.get("status") == "completed"))
        if match and not triggered:
            triggered.append(path.name)
            raise OSError("controlled final export failure")
        return original(path, value)

    monkeypatch.setattr(runner, "write_json", fail_one_write)
    with pytest.raises(OSError, match="final export failure"):
        execute()
    assert triggered
    assert_ineligible(output)


@pytest.mark.parametrize("reason", ["worker_wall_budget", "process_peak_rss_budget", "parent_cancellation"])
def test_cancellation_during_candidate_export_is_checked_before_completion(tmp_path, monkeypatch, reason):
    execute, output, guard, _ = setup_run(tmp_path)
    original = runner.write_json

    def export_then_cancel(path, value):
        result = original(path, value)
        if path.name == "candidate-records.json":
            guard.cancel(reason)
        return result

    monkeypatch.setattr(runner, "write_json", export_then_cancel)
    with pytest.raises(InterruptedError):
        execute()
    assert_ineligible(output)
    assert json.loads((output / "status.json").read_text())["resource"]["cancellation_reason"] == reason


def test_source_change_during_candidate_export_invalidates_receipt(tmp_path, monkeypatch):
    execute, output, _, source = setup_run(tmp_path)
    original = runner.write_json

    def export_then_mutate(path, value):
        result = original(path, value)
        if path.name == "candidate-records.json":
            (source / "scripts/preflight_native_axis.py").write_text("# changed after certification\n")
        return result

    monkeypatch.setattr(runner, "write_json", export_then_mutate)
    with pytest.raises(RuntimeError, match="SOURCE_CHANGED"):
        execute()
    assert_ineligible(output)


@pytest.mark.parametrize("invalid_field", ["feasible", "complete_tool_checked", "frontier_checked"])
def test_missing_independent_certificate_component_cannot_publish(tmp_path, monkeypatch, invalid_field):
    execute, output, _, _ = setup_run(tmp_path)
    original = runner.independent_check_native_history

    def incomplete_audit(*args, **kwargs):
        result = original(*args, **kwargs)
        return replace(result, **{invalid_field: False})

    monkeypatch.setattr(runner, "independent_check_native_history", incomplete_audit)
    with pytest.raises(RuntimeError, match="certification failed"):
        execute()
    assert_ineligible(output)
    assert not (output / "candidate-records.json").exists()


def test_frozen_sequence_and_initial_policy_are_created_before_their_panels(tmp_path, monkeypatch):
    execute, output, _, _ = setup_run(tmp_path)
    original = runner.run_episode
    seen = []

    def inspect_freeze(sim, guard, folder, **kwargs):
        if folder.name.startswith("selection-"):
            frozen = json.loads((output / "sequence-freeze.json").read_text())
            assert tuple(frozen["actions"]) == kwargs["fixed_actions"]
            assert kwargs.get("policy") is None
            assert not (output / "untrained-policy-freeze.json").exists()
        elif folder.name.startswith("untrained-selection-"):
            frozen = json.loads((output / "untrained-policy-freeze.json").read_text())
            assert frozen["seed"] == 11 and frozen["gradient_steps"] == 0
            assert frozen["optimizer_constructed"] is False
            assert kwargs.get("fixed_actions") is None
            assert kwargs["policy"] is not None
            assert all(not parameter.requires_grad for parameter in kwargs["policy"].parameters())
        seen.append(folder.name)
        return original(sim, guard, folder, **kwargs)

    monkeypatch.setattr(runner, "run_episode", inspect_freeze)
    result = execute()
    assert seen == ["greedy", "selection-1", "selection-2", "untrained-selection-1", "untrained-selection-2"]
    assert result["certified_complete_episodes"] == 5
    assert result["eligible_candidate_count"] == 2
    payload = output / "candidate-records.json"
    assert result["candidate_records_sha256"] == hashlib.sha256(payload.read_bytes()).hexdigest()
    candidates = json.loads(payload.read_text())["candidates"]
    assert set(candidates) == {"frozen_greedy_sequence", "untrained_raw_policy"}


def test_parent_deadline_cannot_publish_worker_success_during_grace(tmp_path, monkeypatch):
    output = tmp_path / "launcher-attempt"
    monkeypatch.setattr(sys, "argv", ["preflight", "--output", str(output),
        "--execute", "--case-bundle", str(tmp_path / "unused-fixture.ressectionlab")])
    monkeypatch.setattr(runner, "source_snapshot", lambda: {"file_sha256": {}, "runtime_content_hash": "fixture"})
    monkeypatch.setattr(runner, "assert_source", lambda *args: None)

    class CompletedDuringGrace:
        returncode = 0

        def __init__(self, *args, **kwargs):
            self.calls = 0
            self.terminated = False

        def wait(self, timeout=None):
            self.calls += 1
            if self.calls == 1:
                runner.write_json(output / "profile/status.json",
                    {"status": "completed", "eligible_candidate_count": 2})
                raise runner.subprocess.TimeoutExpired("controlled worker", timeout)
            return 0

        def terminate(self):
            self.terminated = True

        def kill(self):
            pytest.fail("Fixture cooperates during grace; no forced kill should occur")

    monkeypatch.setattr(runner.subprocess, "Popen", CompletedDuringGrace)
    with pytest.raises(SystemExit) as error:
        runner.main()
    assert error.value.code == 1
    receipt = json.loads((output / "launcher-status.json").read_text())
    assert receipt["status"] == "failed" and receipt["eligible_candidate_count"] == 0
    assert receipt["parent_timeout_requested"] and not receipt["hard_killed"]
    assert receipt["worker_returncode"] == 0


@pytest.mark.parametrize("reason", ["worker_wall_budget", "process_peak_rss_budget", "parent_cancellation"])
def test_preparation_cancellation_keeps_worker_resource_receipt(tmp_path, monkeypatch, reason):
    output = tmp_path / "worker-attempt"
    output.mkdir()
    source = {"runtime_content_hash": "controlled-frozen-source"}
    runner.write_json(output / "launch-source.json", source)
    monkeypatch.setattr(runner, "source_snapshot", lambda: source)
    monkeypatch.setattr(runner.signal, "signal", lambda *args: None)

    def incomplete_preparation(output, bundle, guard):
        guard.cancel(reason)
        guard.require()

    monkeypatch.setattr(runner, "_public_worker", incomplete_preparation)
    with pytest.raises(InterruptedError):
        runner.public_worker(output, tmp_path / "never-loaded.ressectionlab")
    failure = json.loads((output / "worker-failure.json").read_text())
    resource = json.loads((output / "worker-resource.json").read_text())
    assert failure["eligible_candidate_count"] == 0
    assert failure["stage"] == "case_and_native_preparation"
    assert failure["resource"]["cancellation_reason"] == resource["cancellation_reason"] == reason
    assert resource["budget_request_to_receipt_seconds"] is not None
    assert not (output / "profile").exists()


def test_changed_frozen_worker_source_fails_before_case_loading(tmp_path, monkeypatch):
    output = tmp_path / "worker-attempt"
    output.mkdir()
    runner.write_json(output / "launch-source.json", {"runtime_content_hash": "original"})
    monkeypatch.setattr(runner, "source_snapshot", lambda: {"runtime_content_hash": "changed"})
    monkeypatch.setattr(runner.signal, "signal", lambda *args: None)
    monkeypatch.setattr(runner, "_public_worker", lambda *args: pytest.fail("No case load after source mismatch"))
    with pytest.raises(RuntimeError, match="Frozen worker differs"):
        runner.public_worker(output, tmp_path / "never-loaded.ressectionlab")
    failure = json.loads((output / "worker-failure.json").read_text())
    assert failure["eligible_candidate_count"] == 0
    assert (output / "worker-resource.json").exists()


@pytest.mark.parametrize("attack", ["in_memory_weights", "checkpoint_bytes"])
def test_initial_policy_remains_identical_through_publication(tmp_path, monkeypatch, attack):
    execute, output, _, _ = setup_run(tmp_path)
    original = runner.run_episode

    def mutate_after_panel(sim, guard, folder, **kwargs):
        result = original(sim, guard, folder, **kwargs)
        if folder.name == "untrained-selection-2":
            if attack == "in_memory_weights":
                import torch
                with torch.no_grad():
                    next(kwargs["policy"].parameters()).add_(1.)
            else:
                with (output / "untrained-raw-initial.pt").open("ab") as checkpoint:
                    checkpoint.write(b"changed after initial policy freeze")
        return result

    monkeypatch.setattr(runner, "run_episode", mutate_after_panel)
    with pytest.raises(RuntimeError, match="[Pp]olicy|checkpoint"):
        execute()
    assert_ineligible(output)
    assert not (output / "candidate-records.json").exists()
