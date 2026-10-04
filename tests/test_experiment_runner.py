"""Experiment isolation and independent replay checks, not performance tests."""
import importlib.util
import json
from pathlib import Path
import sys

import numpy as np
import pytest

from resectionlab.core import CaseData, SourceRef
from resectionlab.learning import TrainingConfig
from resectionlab.simulation import make_patient_simulator, make_synthetic_simulator

spec = importlib.util.spec_from_file_location(
    "run_patient_learning", Path(__file__).parents[1] / "scripts" / "run_patient_learning.py")
runner = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = runner
spec.loader.exec_module(runner)


@pytest.fixture
def stable_sources(tmp_path, monkeypatch):
    """Exercise real snapshot protection on private files during orchestration tests."""
    root = tmp_path / "private-source-fixture"
    source = root / "src/resectionlab"
    source.mkdir(parents=True)
    (source / "learning.py").write_text("# private provenance fixture, not executed\n")
    snapshot = runner.source_snapshot
    preserve = runner.preserve_source
    monkeypatch.setattr(runner, "source_snapshot", lambda *_: snapshot(root))
    monkeypatch.setattr(runner, "preserve_source",
                        lambda record, destination, *_: preserve(record, destination, root))
    return root


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


def test_runner_persists_freeze_and_disjoint_evaluation(tmp_path, stable_sources):
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
    timing = json.loads((evaluation_dir / "validation-timings.json").read_text())
    assert timing["full_validation_seconds"] > 0
    assert set(timing["partition_seconds"]) == {"final_evaluation", "stress"}
    with pytest.raises(FileExistsError):
        runner.run_experiment(make_synthetic_simulator, output, config=TrainingConfig())


def test_development_only_never_opens_final_evaluator(tmp_path, monkeypatch, stable_sources):
    def forbidden(*args, **kwargs):
        raise AssertionError("development-only screening must not evaluate final worlds")
    monkeypatch.setattr(runner, "evaluate_frozen_candidates", forbidden)
    result = runner.run_experiment(make_synthetic_simulator, tmp_path / "dev",
        config=TrainingConfig(max_environment_steps=16, max_gradient_steps=1, max_wall_seconds=10),
        seeds=(11,), counts=(1, 1, 1, 1), evaluate=False)
    assert result["status"] == "completed"
    assert result["evaluation_status"] == "not_requested_development_selection_only"
    assert not list((tmp_path / "dev").glob("evaluation-*"))


def test_native_footprint_failure_invalidates_coarse_pass(tmp_path, stable_sources):
    mask = np.ones((6, 6, 6), dtype=bool)
    case = CaseData(case_id="native-audit-fixture", mri=mask.astype(np.float32),
                    compartments={"enhancing": mask}, affine=np.eye(4), brain_mask=mask,
                    source_refs=(SourceRef("fixture", "synthetic://native-audit", provenance="simulated"),))
    template = make_patient_simulator(case, block_size=3, max_steps=2, max_actions=4)
    output = tmp_path / "native-audit"
    result = runner.run_experiment(template.clone, output,
        config=TrainingConfig(max_environment_steps=16, max_gradient_steps=1, max_wall_seconds=10),
        seeds=(11,), counts=(1, 1, 1, 1), source_case=case)
    assert result["status"] == "invalidated"
    manifest = json.loads((output / "manifest.json").read_text())
    assert manifest["preoperative_planning_hash"] == case.planning_hash
    assert manifest["world_partitions"]["optimization"]["planning_hash"] == case.planning_hash
    evaluation_dir = output / result["evaluation_run_id"]
    native = json.loads((evaluation_dir / "native-removal-audit.json").read_text())
    coarse = json.loads((evaluation_dir / "coarse-sequence-audit.json").read_text())
    assert native["required"] is True
    assert native["candidates"]["STOP"]["feasible"] is True
    assert native["candidates"]["SEARCH"]["feasible"] is False
    assert native["candidates"]["SEARCH"]["unsupported_source_tissue_volume_mm3"] > 0
    assert coarse["SEARCH"]["feasible"] is True
    evaluation = json.loads((evaluation_dir / "final_evaluation.json").read_text())
    search = next(candidate for candidate in evaluation["candidates"] if candidate["plan_id"] == "SEARCH")
    assert search["status"] == "rejected_geometry"
    assert search["world_outcomes"] == []


def test_search_and_rl_share_nonstop_horizon(tmp_path, stable_sources):
    output = tmp_path / "short-horizon"
    runner.run_experiment(make_synthetic_simulator, output,
        config=TrainingConfig(max_environment_steps=16, max_gradient_steps=1,
                              max_wall_seconds=10, max_episode_steps=2),
        seeds=(11,), counts=(1, 1, 1, 1), evaluate=False)
    manifest = json.loads((output / "manifest.json").read_text())
    assert manifest["requested_simulator_nonstop_horizon"] == 4
    assert manifest["effective_shared_nonstop_horizon"] == 1
    frozen = json.loads((output / "candidate-freeze.json").read_text())
    assert all(sum(action != "STOP" for action in candidate["actions"]) <= 1
               for candidate in frozen["candidates"])
