"""Generated native development controls; no patient data or clinical claims."""
from dataclasses import replace
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

from resectionlab.data_policy import DataPolicyError, GeneratedDevelopmentContext
from resectionlab.native_spatial_task import make_native_opening_task, NativeSpatialTask
from resectionlab.spatial_policy import (SpatialPolicy, SpatialPolicyConfig, SpatialTransition,
    gradient_step, imitation_loss, parameter_hash, reinforce_loss)


@pytest.fixture(autouse=True)
def one_thread():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


def context(task):
    return GeneratedDevelopmentContext("1" * 64, (task.case.source_hash,), task.decision_model_hash)


def policy():
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(11)
        return SpatialPolicy(SpatialPolicyConfig(critic_candidate_context=True))


@pytest.mark.parametrize("function", [imitation_loss, reinforce_loss, gradient_step])
def test_implicit_learning_stays_closed_before_inputs(function):
    with pytest.raises(DataPolicyError):
        function()


@pytest.mark.parametrize("change", [
    {"scope": "clinical"}, {"transition_lineage": "recorded"},
    {"real_patient_count": 1}, {"real_patient_count": False},
    {"declaration_sha256": "unbound"}, {"source_ids": ("2" * 64, "3" * 64)},
])
def test_generated_admission_does_not_relabel_patient_or_recorded_sources(change):
    original = context(make_native_opening_task())
    with pytest.raises(ValueError):
        replace(original, **change)


def test_wrong_source_and_real_track_rejected_before_forward(monkeypatch):
    task, model = make_native_opening_task(), policy()
    observation = task.observation()
    monkeypatch.setattr(model, "forward", lambda *_: pytest.fail("No forward before source admission"))
    with pytest.raises(ValueError, match="source/track"):
        imitation_loss(model, [(observation, "STOP")],
                       learning_context=replace(context(task), source_ids=("2" * 64,)))
    with pytest.raises(ValueError, match="source/track"):
        imitation_loss(model, [(replace(observation, track="annotation_assisted"), "STOP")],
                       learning_context=context(task))


def test_actual_admitted_gradient_changes_actor_and_encoder_and_rejects_stale_loss():
    task, model = make_native_opening_task(), policy()
    observation, permit = task.observation(), context(task)
    optimizer = torch.optim.Adam(model.parameters(), lr=.001)
    before_actor, before_encoder = parameter_hash(model.actor), parameter_hash(model.encoder)
    loss, report = imitation_loss(model, [(observation, observation.action_ids[1])], learning_context=permit)
    assert report["loss_forward_calls"] == 1
    update = gradient_step(model, optimizer, loss, learning_context=permit)
    assert update["parameters_changed"]
    assert parameter_hash(model.actor) != before_actor
    assert parameter_hash(model.encoder) != before_encoder
    assert update["actor_gradient_norm_before_clip"] > 0
    with pytest.raises(ValueError, match="matching admitted loss"):
        gradient_step(model, optimizer, loss, learning_context=permit)


def test_arbitrary_or_differently_bound_loss_cannot_step(monkeypatch):
    task, model = make_native_opening_task(), policy()
    permit = context(task)
    optimizer = torch.optim.Adam(model.parameters(), lr=.001)
    monkeypatch.setattr(optimizer, "step", lambda: pytest.fail("No update after loss mismatch"))
    with pytest.raises(ValueError, match="matching admitted loss"):
        gradient_step(model, optimizer, next(model.parameters()).sum(), learning_context=permit)
    loss, _ = imitation_loss(model, [(task.observation(), "STOP")], learning_context=permit)
    with pytest.raises(ValueError, match="matching admitted loss"):
        gradient_step(model, optimizer, loss,
                      learning_context=replace(permit, declaration_sha256="2" * 64))


def test_reference_truth_not_in_actor_or_teacher_and_partial_rl_batch_refused():
    task = make_native_opening_task()
    counterfactual = NativeSpatialTask(replace(task.case,
        reference_target=np.ones(task.case.observed_support.shape)), max_steps=2)
    assert task.observation().fingerprint == counterfactual.observation().fingerprint
    assert task.observation().action_ids == counterfactual.observation().action_ids
    assert task.planning_clone().metrics() == counterfactual.planning_clone().metrics()
    step = SpatialTransition(task.observation(), "STOP", 0., False)
    with pytest.raises(ValueError, match="complete episodes"):
        reinforce_loss(policy(), [(step,)], learning_context=context(task))


def test_legacy_patient_loader_and_ppo_remain_closed():
    from resectionlab.learning import load_policy
    from resectionlab.learning_ppo import train_patient_ppo
    with pytest.raises(DataPolicyError):
        load_policy()
    with pytest.raises(DataPolicyError):
        train_patient_ppo()


@pytest.fixture
def runner(monkeypatch):
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1] / "scripts"))
    import run_native_opening_learning
    return run_native_opening_learning


def test_exact_teacher_enumerates_all_states_and_requires_two_complementary_tools(runner):
    base = make_native_opening_task()
    before = base.metrics()
    samples, sequence, report = runner.exact_teacher(base, check=lambda: None)
    assert report["complete"] and report["nominal_optimum"] == pytest.approx(1.1)
    assert len(sequence) == 2 and len(samples) > 1 and report["terminal_sequences"] > 10
    assert all(action in observation.action_ids for observation, action in samples)
    assert base.metrics() == before
    other = NativeSpatialTask(replace(base.case, reference_target=np.ones(base.case.observed_support.shape)), max_steps=2)
    _, other_sequence, other_report = runner.exact_teacher(other, check=lambda: None)
    assert other_sequence == sequence and other_report == report


@pytest.mark.parametrize("mode", ["STOP", "opening"])
def test_real_episode_terminal_json_boundary_is_audited_outside_budget(runner, tmp_path, monkeypatch, mode):
    from resectionlab import native_spatial_evaluation
    from resectionlab.native_resection import NativeResectionEngine
    original_preview = NativeResectionEngine.preview_stroke
    actual_audit = native_spatial_evaluation.evaluate_native_spatial_episode
    calls = []
    def audit(task, **kwargs):
        assert NativeResectionEngine.preview_stroke is original_preview
        calls.append(task.metrics()["history"])
        return actual_audit(task, **kwargs)
    monkeypatch.setattr(native_spatial_evaluation, "evaluate_native_spatial_episode", audit)
    base, model = make_native_opening_task(), policy()
    planner = (lambda check: (("STOP",), {})) if mode == "STOP" else (
        lambda check: runner.exact_teacher(base, check=check)[1:])
    transitions, report, _ = runner.guarded_episode(base, model, tmp_path, mode, lambda: None,
        mode="sequence", planner=planner)
    assert calls and transitions[-1].terminated and report["status"] == "complete"
    assert report["independent_evaluation"]["accepted"]
    assert report["simulated_return"] == pytest.approx(0. if mode == "STOP" else 1.1)


def test_preview_exhaustion_preserves_commit_and_null_outcome(runner, tmp_path):
    from resectionlab.native_axis_simulation import CommittedTransitionInterrupted
    base, model = make_native_opening_task(), policy()
    first = next(row["action_id"] for row in base.candidate_inventory()["ledger"]
                 if row["feasible"] and row["tool_id"] == "short-wide-opener" and row["voxel"] == [4, 4, 1])
    with pytest.raises(CommittedTransitionInterrupted):
        runner.guarded_episode(base, model, tmp_path, "interrupted", lambda: None,
            mode="sequence", planner=lambda check: ((first, "STOP"), {}), previews=0)
    import json
    episode = json.loads((tmp_path / "interrupted.json").read_text())
    costs = json.loads((tmp_path / "interrupted-cost.json").read_text())
    assert episode["committed_transitions"] == 1
    assert episode["decisions"][0]["status"] == "committed_unreturned"
    assert episode["failure_metrics"]["history"] and costs["outcomes"] is None
    assert costs["online"]["failure"] is not None


@pytest.mark.parametrize("field,value", [("task", {"factory": "patient"}),
    ("policy", {}), ("lineage", {"real_patient_count": 1}), ("checkpoint_rule", "best_score")])
def test_changed_development_declaration_refused_before_task(runner, monkeypatch, field, value):
    record = {**runner.specification(), "source_sha256": {"test": "hash"}}
    monkeypatch.setattr(runner, "sources", lambda *_: {"test": "hash"})
    runner.validate(record)
    record[field] = value
    with pytest.raises(ValueError, match="fixed generated"):
        runner.validate(record)


def test_task_reward_mismatch_and_equal_weights_other_instance_refused():
    from resectionlab.simulation import RewardSpec
    task, model = make_native_opening_task(), policy()
    permit = context(task)
    wrong = NativeSpatialTask(task.case, max_steps=2, reward=replace(task.reward_spec, normal_per_mm3=.3))
    with pytest.raises(ValueError, match="source/reward/horizon"):
        permit.require_task(wrong)
    loss, _ = imitation_loss(model, [(task.observation(), "STOP")], learning_context=permit)
    other = policy()
    assert parameter_hash(model) == parameter_hash(other)
    with pytest.raises(ValueError, match="matching admitted loss"):
        gradient_step(other, torch.optim.Adam(other.parameters()), loss, learning_context=permit)
    with pytest.raises(ValueError, match="Optimizer must own exactly"):
        gradient_step(model, torch.optim.Adam(other.parameters()), loss, learning_context=permit)


@pytest.mark.parametrize("mutation", ["equal_weight_head", "normalization"])
def test_stale_architecture_or_parameter_objects_refused_before_backward(mutation):
    import copy
    task, model = make_native_opening_task(), policy()
    permit = context(task)
    loss, _ = imitation_loss(model, [(task.observation(), "STOP")], learning_context=permit)
    if mutation == "equal_weight_head": model.actor = copy.deepcopy(model.actor)
    else: model.config = replace(model.config, physical_reference_mm=20.)
    with pytest.raises(ValueError, match="matching admitted loss"):
        gradient_step(model, torch.optim.Adam(model.parameters()), loss, learning_context=permit)
    assert all(p.grad is None for p in model.parameters())


def test_source_closure_cannot_drop_numeric_dependency_or_include_desktop(runner):
    paths = runner.dependency_paths()
    assert "src/resectionlab/spatial_policy.py" in paths
    assert "src/resectionlab/native_spatial_task.py" in paths
    assert "src/resectionlab/desktop_bridge.py" not in paths
    with pytest.raises(ValueError, match="Source closure omitted"):
        runner.sources([p for p in paths if p != "src/resectionlab/spatial_policy.py"])


def test_preparation_failure_retains_every_declared_method(runner, monkeypatch, tmp_path):
    import json
    from resectionlab import native_spatial_task
    monkeypatch.setattr(runner, "validate", lambda *_: None)
    def unavailable(**kwargs): raise ValueError("intentional preparation failure")
    monkeypatch.setattr(native_spatial_task, "make_native_opening_task", unavailable)
    with pytest.raises(ValueError, match="intentional preparation"):
        runner.worker(runner.specification(), tmp_path, declaration_sha256="1" * 64)
    receipt = json.loads((tmp_path / "result.json").read_text())
    assert set(receipt["methods"]) == set(runner.METHODS)
    assert all(row["outcomes"] is None and row["status"] == "not_started"
               for row in receipt["methods"].values())
