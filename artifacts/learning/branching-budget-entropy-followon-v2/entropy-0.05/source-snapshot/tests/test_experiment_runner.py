"""Experiment isolation and independent replay checks, not performance tests."""
import importlib.util
import json
from pathlib import Path
import sys

import pytest

from resectionlab.learning import TrainingConfig
from resectionlab.simulation import make_synthetic_simulator

spec = importlib.util.spec_from_file_location(
    "run_patient_learning", Path(__file__).parents[1] / "scripts" / "run_patient_learning.py")
runner = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = runner
spec.loader.exec_module(runner)


def test_source_snapshot_detects_uncommitted_file_edits(tmp_path):
    source = tmp_path / "src"
    source.mkdir()
    file = source / "uncommitted.py"
    file.write_text("answer = 1\n")
    snapshot = runner.source_snapshot(tmp_path)
    assert "src/uncommitted.py" in snapshot["file_sha256"]
    runner.preserve_source(snapshot, tmp_path / "archive", tmp_path)
    assert (tmp_path / "archive/src/uncommitted.py").read_text() == "answer = 1\n"
    file.write_text("answer = 2\n")
    with pytest.raises(ValueError, match="SOURCE_CHANGED"):
        runner.assert_source_unchanged(snapshot, tmp_path)


def test_independent_checker_certifies_connected_sequence_and_rejects_teleport():
    case = make_synthetic_simulator()
    actions = tuple(f"REMOVE:synthetic-cannula:1,1,{z}" for z in range(3)) + ("STOP",)
    candidate = runner.FrozenSequence("legal", case.case_hash, actions, "SEARCH")
    assert runner.independent_sequence_check(make_synthetic_simulator, candidate).feasible
    illegal = runner.FrozenSequence("teleport", case.case_hash, actions[2:], "SEARCH")
    assert not runner.independent_sequence_check(make_synthetic_simulator, illegal).feasible
    after_stop = runner.FrozenSequence("after-stop", case.case_hash, ("STOP",) + actions, "SEARCH")
    assert not runner.independent_sequence_check(make_synthetic_simulator, after_stop).feasible


def test_runner_persists_freeze_and_disjoint_evaluation(tmp_path, monkeypatch):
    # Concurrent development in the repository must not make this isolation
    # test flaky. Source change detection has its own test above.
    monkeypatch.setattr(runner, "assert_source_unchanged", lambda *_, **__: None)
    output = tmp_path / "run"
    result = runner.run_experiment(make_synthetic_simulator, output,
        config=TrainingConfig(max_environment_steps=16, max_gradient_steps=1,
                              max_wall_seconds=10, checkpoint_interval=1),
        seeds=(11,), counts=(1, 1, 1, 1))
    assert result["status"] == "completed"
    manifest = json.loads((output / "manifest.json").read_text())
    freeze = json.loads((output / "candidate-freeze.json").read_text())
    evaluation_dir = output / result["evaluation_run_id"]
    evaluation = json.loads((evaluation_dir / "final_evaluation.json").read_text())
    assert result["run_id"] != result["evaluation_run_id"]
    assert evaluation["candidate_freeze_hash"] == freeze["fingerprint"]
    assert evaluation["independent_patient_count"] == 0
    assert evaluation["unit_of_independent_patient_inference"] == "synthetic_fixture"
    assert manifest["source"]["file_sha256"]["src/resectionlab/learning.py"]
    panels = manifest["world_partitions"]
    assert not set(panels["optimization"]["seeds"]) & set(panels["final_evaluation"]["seeds"])
    assert len(json.loads((evaluation_dir / "ledger.json").read_text())) == 2
    assert all(c["clinical_deficit_probability"] is None for c in evaluation["candidates"])
    assert all(c["geometry"]["feasible"] for c in evaluation["candidates"])
    with pytest.raises(FileExistsError):
        runner.run_experiment(make_synthetic_simulator, output, config=TrainingConfig())


def test_development_only_never_opens_final_evaluator(tmp_path, monkeypatch):
    monkeypatch.setattr(runner, "assert_source_unchanged", lambda *_, **__: None)
    def forbidden(*args, **kwargs):
        raise AssertionError("development-only screening must not evaluate final worlds")
    monkeypatch.setattr(runner, "evaluate_frozen_candidates", forbidden)
    result = runner.run_experiment(make_synthetic_simulator, tmp_path / "dev",
        config=TrainingConfig(max_environment_steps=16, max_gradient_steps=1, max_wall_seconds=10),
        seeds=(11,), counts=(1, 1, 1, 1), evaluate=False)
    assert result["status"] == "completed"
    assert result["evaluation_status"] == "not_requested_development_selection_only"
    assert not list((tmp_path / "dev").glob("evaluation-*"))
