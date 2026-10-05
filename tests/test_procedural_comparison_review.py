"""Orchestration-only incomplete-selection audit; no policy training executes."""
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from resectionlab.learning import TrainingResult
from resectionlab import procedural_learning
from resectionlab.simulation import SearchResult
from test_procedural_transfer_runner import native_fixture, runner


def test_comparison_retains_incomplete_initial_selection_without_promoting_checkpoint(tmp_path, monkeypatch):
    case, base, fixture_declaration = native_fixture()
    panels = runner.assert_declared_target(case, base, fixture_declaration)
    declaration = runner.load_declaration(runner.ROOT / runner.DECLARATION_PATH)
    monkeypatch.setattr(runner, "assert_declaration", lambda value: None)
    monkeypatch.setattr(runner, "assert_declared_sources", lambda value: None)
    monkeypatch.setattr(runner, "assert_declared_target", lambda *args: panels)
    monkeypatch.setattr(runner, "transfer_source_snapshot", lambda: {})
    monkeypatch.setattr(runner, "assert_transfer_source_unchanged", lambda value: None)
    monkeypatch.setattr(runner, "preserve_source", lambda *args: None)
    monkeypatch.setattr(runner, "preserve_declaration_inputs", lambda *args: None)
    checkpoint = tmp_path / "test-only-initialization.pt"
    checkpoint.write_bytes(b"orchestration-only-stub-never-loaded-as-torch")
    shared = {"policy_hash": "test-only-policy", "checkpoint_file_sha256": hashlib.sha256(checkpoint.read_bytes()).hexdigest()}
    # This test injects an intentionally incomplete trainer result; the real
    # patient preflight has separate identity/world tests and must be bypassed
    # only here to reach the candidate-selection failure being exercised.
    monkeypatch.setattr(procedural_learning, "validate_procedural_target_worlds", lambda *args, **kwargs: {})
    monkeypatch.setattr(procedural_learning, "validate_procedural_checkpoint", lambda *args, **kwargs: shared)
    monkeypatch.setattr(procedural_learning, "load_frozen_procedural_policy", lambda *args, **kwargs: object())
    monkeypatch.setattr(runner, "policy_hash", lambda actor: shared["policy_hash"])
    monkeypatch.setattr(runner, "load_policy", lambda *args, **kwargs: object())
    monkeypatch.setattr(runner, "native_greedy_search", lambda *args, **kwargs: SearchResult(("STOP",), 0., 0, 0., "test_only"))
    monkeypatch.setattr(runner, "native_beam_search", lambda *args, **kwargs: SearchResult(("STOP",), 0., 0, 0., "test_only"))
    monkeypatch.setattr(runner, "frozen_population_rollouts", lambda *args, **kwargs: ({"selection_panel_complete": True}, ("STOP",)))
    monkeypatch.setattr(runner, "rollout_policy", lambda *args, **kwargs: SimpleNamespace(actions=("STOP",)))
    def incomplete(*args, output_dir, **kwargs):
        folder = Path(output_dir)
        folder.mkdir()
        (folder / "contract.json").write_text(json.dumps({"algorithm": declaration["policy"]["algorithm"]}))
        (folder / "result.json").write_text(json.dumps({
            "actor_parameters_changed": False, "initial_actor_hash": "unchanged", "latest_actor_hash": "unchanged",
            "selection_history": [], "initial_selection_return": None, "selected_selection_return": None}))
        return TrainingResult("wall_time_budget", "PATIENT_SCRATCH_RL", shared["policy_hash"],
            shared["policy_hash"], shared["policy_hash"], base.decision_model_hash, 0, 0, 0, 30., None, None, str(folder))
    monkeypatch.setattr(runner, "train_patient_policy", incomplete)
    monkeypatch.setattr(procedural_learning, "train_procedural_adapted_policy", incomplete)
    monkeypatch.setattr(runner, "validate_frozen_candidates", lambda *args, **kwargs: {"rejected_candidate_ids": []})
    output = tmp_path / "comparison"
    target = procedural_learning.TransferTarget(base.case_hash, "planning-fixture",
        "procedural_test:orchestration-only", source_kind="procedural_test_only")
    with pytest.raises((ValueError, RuntimeError), match="selection|panel"):
        runner.run_target_comparison(base, case, target, checkpoint, output, declaration)
    status = json.loads((output / "status.json").read_text())
    assert status["status"] != "completed"
    assert len(status["planned_learning_runs"]) == 6
    assert not status["completed_learning_runs"]
    assert not (output / "candidate-freeze.json").exists()
    attempted = json.loads((output / "training.json").read_text())
    assert len(attempted) == 1
    assert attempted[0]["selected_selection_return"] is None
