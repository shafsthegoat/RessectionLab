"""Small synthetic receipts; no public data, performance study or clinical claim."""
import json

import numpy as np
import pytest

from resectionlab.geometry import AccessWindow
from resectionlab.learning import TrainingConfig, train_patient_policy
from resectionlab.native_axis_accounting import (AccountingReceiptError, AxisTrainingAccounting,
                                                train_axis_policy)
from resectionlab.native_axis_simulation import AxisColumnNativeSimulator, CommittedTransitionInterrupted
from resectionlab.native_proposals import AxisColumnProposalConfig
from resectionlab.native_resection import NATIVE_GENERIC_TOOLS, NativeResectionConfig
from resectionlab.procedural_learning import native_observation_schema
from resectionlab.worlds import generate_partitions


def fixture(*, cancelled=None):
    tissue = np.ones((7, 7, 8), bool)
    labels = np.zeros(tissue.shape, np.int16)
    labels[1:6, 1:6, 2:7] = 1
    config = NativeResectionConfig(tissue, labels, np.eye(4),
        AccessWindow((3, 3, -.5), (0, 0, 1), 4), NATIVE_GENERIC_TOOLS,
        "synthetic-axis-accounting-v1", "explicit synthetic support")
    return AxisColumnNativeSimulator(config,
        proposal_config=AxisColumnProposalConfig(((0, 0),), 2), max_steps=2, cancelled=cancelled)


def manager(tmp_path, raw, *, factory=None):
    panels = generate_partitions(raw.case_hash, raw.config.world_generator, 20261004,
        optimization=2, selection=1, final_evaluation=1, stress=1)
    accounting = AxisTrainingAccounting(factory or (lambda: raw), panels.optimization, panels.selection,
        receipt_path=tmp_path / "axis-accounting.json", expected_model_hash=raw.decision_model_hash)
    return accounting, panels


def settings():
    return TrainingConfig(seed=11, hidden_features=16, max_gradient_steps=1,
        max_environment_steps=12, max_wall_seconds=20., max_episode_steps=3,
        episodes_per_update=1, checkpoint_interval=1)


def test_constructor_unassigned_then_cut_and_stop_are_different_counts(tmp_path):
    raw = fixture()
    accounting, panels = manager(tmp_path, raw)
    sim = accounting.recorded_factory()
    assert accounting.snapshot()["events"][0]["role"] is None
    assert accounting.snapshot()["episodes"] == []
    assert all(value["executed_transitions"] == 0 for value in accounting.snapshot()["totals"].values())
    sim.reset(panels.optimization.seeds[0])
    transition = sim.step(1)
    sim.step(0)
    receipt = json.loads(accounting.receipt_path.read_text())
    totals = receipt["totals"]["optimization"]
    assert totals == {"counts_complete": True, "executed_transitions": 2, "returned_transitions": 2,
        "native_commits": 1, "committed_but_unreturned": 0, "executed_reward": transition.reward}
    assert receipt["episodes"][0]["status"] == "complete"
    assert receipt["events"][1]["info"]["removed_indices_native"] == raw.engine.history[0]["removed_indices_native"]
    assert not receipt["candidate_eligible"]
    detached = accounting.snapshot()
    detached["events"].clear()
    assert len(accounting.snapshot()["events"]) == 3


def test_actual_post_commit_interruption_retained_and_cannot_resume(tmp_path, monkeypatch):
    cancel = {"value": False}
    raw = fixture(cancelled=lambda: cancel["value"])
    accounting, panels = manager(tmp_path, raw)
    sim = accounting.recorded_factory()
    sim.reset(panels.selection.seeds[0])
    original = raw.engine.commit_preview
    def commit(preview):
        result = original(preview)
        cancel["value"] = True
        return result
    monkeypatch.setattr(raw.engine, "commit_preview", commit)
    with pytest.raises(CommittedTransitionInterrupted) as error:
        sim.step(1)
    receipt = json.loads(accounting.receipt_path.read_text())
    totals = receipt["totals"]["selection"]
    assert totals["executed_transitions"] == totals["native_commits"] == totals["committed_but_unreturned"] == 1
    assert totals["returned_transitions"] == 0
    assert totals["executed_reward"] == pytest.approx(raw.total_reward) == pytest.approx(error.value.reward)
    event = receipt["events"][-1]
    assert event["info"] == json.loads(json.dumps(error.value.info))
    assert event["next_observation_complete"] is False
    assert raw.engine.revision == 1 and raw.removed_mask.any()
    assert raw._proposals is None
    assert receipt["status"] == "failed" and receipt["candidate_eligible"] is False
    cancel["value"] = False
    for operation in (lambda: sim.step(0), lambda: sim.reset(panels.selection.seeds[0]), accounting.recorded_factory):
        with pytest.raises(RuntimeError, match="closed"):
            operation()
    assert raw.engine.revision == 1


def test_before_commit_cancellation_counts_no_transition(tmp_path):
    cancel = {"value": False}
    raw = fixture(cancelled=lambda: cancel["value"])
    accounting, panels = manager(tmp_path, raw)
    sim = accounting.recorded_factory()
    sim.reset(panels.optimization.seeds[0])
    cancel["value"] = True
    with pytest.raises(InterruptedError):
        sim.step(1)
    assert accounting.snapshot()["totals"]["optimization"]["executed_transitions"] == 0
    assert raw.engine.revision == 0 and not raw.removed_mask.any()


def test_export_failure_after_successful_cut_preserves_in_memory_truth(tmp_path, monkeypatch):
    raw = fixture()
    accounting, panels = manager(tmp_path, raw)
    sim = accounting.recorded_factory()
    sim.reset(panels.optimization.seeds[0])
    prior_durable = accounting.receipt_path.read_bytes()
    def fail(*args):
        raise OSError("synthetic full disk")
    monkeypatch.setattr("resectionlab.native_axis_accounting._atomic_receipt", fail)
    with pytest.raises(AccountingReceiptError) as error:
        sim.step(1)
    receipt = error.value.receipt
    assert receipt == accounting.snapshot() == error.value.axis_accounting_receipt
    assert receipt["totals"]["optimization"]["native_commits"] == 1
    assert receipt["totals"]["optimization"]["returned_transitions"] == 0
    assert receipt["receipt_export_failure"]["reason"] == "synthetic full disk"
    assert receipt["status"] == "failed" and not receipt["candidate_eligible"]
    assert raw.engine.revision == 1 and raw.total_reward > 0
    assert accounting.receipt_path.read_bytes() == prior_durable


def test_export_failure_preserves_original_committed_exception(tmp_path, monkeypatch):
    raw = fixture()
    accounting, panels = manager(tmp_path, raw)
    sim = accounting.recorded_factory()
    sim.reset(panels.selection.seeds[0])
    step = raw.step
    thrown = []
    def interrupt(action):
        result = step(action)
        error = CommittedTransitionInterrupted(result.info, result.reward)
        thrown.append(error)
        raise error
    monkeypatch.setattr(raw, "step", interrupt)
    def fail(*args):
        raise OSError("synthetic lost directory")
    monkeypatch.setattr("resectionlab.native_axis_accounting._atomic_receipt", fail)
    with pytest.raises(CommittedTransitionInterrupted) as error:
        sim.step(1)
    assert error.value is thrown[0]
    assert error.value.axis_accounting_receipt["totals"]["selection"]["committed_but_unreturned"] == 1
    assert "export failed" in error.value.__notes__[-1]


def test_actual_tiny_raw_training_keeps_generic_counts_and_gates(tmp_path):
    template = fixture()
    accounting, panels = manager(tmp_path, template, factory=template.fresh)
    result = train_axis_policy(accounting, config=settings(), output_dir=tmp_path / "learner")
    receipt = json.loads(accounting.receipt_path.read_text())
    generic = json.loads((tmp_path / "learner/result.json").read_text())
    assert result.gradient_steps == 1 and generic["actor_parameters_changed"]
    assert receipt["status"] == "learner_returned"
    assert receipt["totals"]["optimization"]["returned_transitions"] == result.optimization_environment_steps
    assert receipt["totals"]["selection"]["returned_transitions"] == result.selection_environment_steps
    assert receipt["learner_result"]["optimization_environment_steps"] == generic["optimization_environment_steps"]
    assert receipt["candidate_eligible"] is False
    other, _ = manager(tmp_path / "gate", template, factory=template.fresh)
    with pytest.raises(ValueError):
        native_observation_schema(other.recorded_factory())
    with pytest.raises(ValueError, match="native|Native|FEATURE_UNITS"):
        train_patient_policy(other.recorded_factory, panels.optimization, panels.selection,
            config=settings(), output_dir=tmp_path / "feature-units", input_profile="FEATURE_UNITS")
    assert not (tmp_path / "feature-units/checkpoint.pt").exists()


def test_initial_selection_failure_retains_real_commit_without_checkpoint(tmp_path, monkeypatch):
    from resectionlab.learning import MaskedPatientPolicy
    forward = MaskedPatientPolicy.forward
    def first_cut(policy, observation):
        logits, value = forward(policy, observation)
        # Deterministic initial selection cut while preserving the real policy interface.
        logits = logits * 0
        logits[1 if len(logits) > 1 else 0] = 10
        return logits, value
    monkeypatch.setattr(MaskedPatientPolicy, "forward", first_cut)
    template = fixture()
    instances = []
    def factory():
        raw = template.fresh()
        instances.append(raw)
        original = raw.engine.commit_preview
        def commit(preview):
            result = original(preview)
            raw._cancelled = lambda: True
            return result
        raw.engine.commit_preview = commit
        return raw
    accounting, _ = manager(tmp_path, template, factory=factory)
    with pytest.raises(CommittedTransitionInterrupted):
        train_axis_policy(accounting, config=settings(), output_dir=tmp_path / "learner")
    receipt = accounting.snapshot()
    assert receipt["status"] == "failed" and receipt["learner_result"] is None
    assert receipt["totals"]["selection"]["committed_but_unreturned"] == 1
    assert instances[-1].engine.revision == 1
    assert not (tmp_path / "learner/checkpoint.pt").exists()
    assert not (tmp_path / "learner/result.json").exists()
    generic_failure = json.loads((tmp_path / "learner/failures.jsonl").read_text())
    assert generic_failure["exception"] == "CommittedTransitionInterrupted"
    assert "history" not in generic_failure  # Shared learner remains unchanged.
