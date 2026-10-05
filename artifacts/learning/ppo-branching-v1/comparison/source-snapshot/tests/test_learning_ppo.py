"""PPO likelihood clipping, old-policy isolation, terminals and honest budgets."""
from dataclasses import replace
import json
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from resectionlab.learning import MaskedPatientPolicy, load_policy, rollout_policy
from resectionlab import learning
from resectionlab.learning_ppo import (FrozenObservation, PPOBatch, PPOConfig,
    clipped_surrogate, generalized_advantages, ppo_minibatch_loss, train_patient_ppo)
from resectionlab.worlds import WorldGeneratorConfig, WorldPartitionManifest, WorldRole


class OneStepSimulator:
    case_hash = "ppo-test-case"
    decision_model_hash = "ppo-one-step-v1"
    world_generator_fingerprint = WorldGeneratorConfig().fingerprint

    def reset(self, seed=0):
        self.reward = 0.0
        return SimpleNamespace(action_features=np.eye(3, dtype=np.float32),
            state_features=np.array([1.0], np.float32), action_mask=np.array([True, True, False]),
            action_ids=("STOP", "BENEFIT", "ILLEGAL_REWARD"))

    def step(self, action):
        assert action in (0, 1), "PPO sampled a masked action"
        self.reward = float(action)
        return SimpleNamespace(observation=self.reset(), reward=float(action), terminated=True, info={})

    def metrics(self):
        return {"fixture": "ppo_one_step", "clinical_deficit_probability": None}


def partitions():
    world = WorldGeneratorConfig()
    return (WorldPartitionManifest(WorldRole.OPTIMIZATION, "ppo-test-case", world, (1, 2, 3)),
            WorldPartitionManifest(WorldRole.SELECTION, "ppo-test-case", world, (101, 102)))


def config(**changes):
    return replace(PPOConfig(seed=11, hidden_features=16, max_gradient_steps=32,
        max_environment_steps=256, max_wall_seconds=30, max_episode_steps=4,
        episodes_per_update=4, checkpoint_interval=8, minibatch_size=64), **changes)


def test_clipping_signs_and_gradient_stop_for_favorable_excess_ratios():
    ratios = torch.tensor([1.5, .5, 1.5, .5])
    new = ratios.log().requires_grad_()
    old = torch.zeros(4, requires_grad=True)
    advantage = torch.tensor([1., -1., -1., 1.], requires_grad=True)
    values = clipped_surrogate(new, old, advantage, .2)
    torch.testing.assert_close(values, torch.tensor([1.2, -.8, -1.5, .5]))
    values.sum().backward()
    torch.testing.assert_close(new.grad, torch.tensor([0., 0., -1.5, .5]))
    assert old.grad is None and advantage.grad is None


def test_terminal_returns_do_not_leak_across_stop_or_reset():
    advantage, returns = generalized_advantages([1, 2, 9], [.5, .25, 123], [False, True, True],
                                                gamma=.5, gae_lambda=1, bootstrap_value=999)
    np.testing.assert_allclose(returns, [2, 2, 9])
    np.testing.assert_allclose(advantage, [1.5, 1.75, -114])
    _, truncated = generalized_advantages([1], [.2], [False], gamma=.9, gae_lambda=1, bootstrap_value=2)
    np.testing.assert_allclose(truncated, [2.8])
    _, stopped = generalized_advantages([1], [.2], [True], gamma=.9, bootstrap_value=2)
    np.testing.assert_allclose(stopped, [1])


def test_old_mask_features_and_log_probabilities_are_frozen_during_epochs():
    observation = OneStepSimulator().reset()
    frozen = FrozenObservation.snapshot(observation)
    policy = MaskedPatientPolicy(3, 1, 8)
    with torch.no_grad():
        logits, _ = policy(frozen)
        old = float(torch.distributions.Categorical(logits=logits).log_prob(torch.tensor(1)))
    batch = PPOBatch((frozen,), (1,), (old,), (1.,), (1.,), (True,), "old-policy", (1.,))
    before = batch.fingerprint
    observation.action_mask[1] = False
    observation.action_mask[2] = True
    observation.action_features.fill(99)
    optimizer = torch.optim.Adam(policy.parameters(), lr=.1)
    for _ in range(4):
        loss, _ = ppo_minibatch_loss(policy, batch, [0], config())
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    assert batch.fingerprint == before
    assert batch.old_log_probs == (old,)
    assert frozen.action_mask.tolist() == [True, True, False]
    with pytest.raises(ValueError):
        frozen.action_mask[1] = False
    assert PPOBatch.unpack(batch.pack()).fingerprint == before


def test_actual_ppo_updates_and_minibatch_epoch_accounting(tmp_path):
    result = train_patient_ppo(OneStepSimulator, *partitions(), output_dir=tmp_path, config=config())
    report = json.loads((tmp_path / "result.json").read_text())
    assert result.gradient_steps == 32
    assert result.latest_checkpoint_hash != result.initial_checkpoint_hash
    assert report["actor_parameters_changed"] is True
    assert report["rollout_batches"] == 8
    assert report["optimization_sample_presentations"] == 128
    assert result.optimization_environment_steps == 32
    assert len(report["optimization_history"]) == 32
    for start in range(0, 32, 4):
        repeated = report["optimization_history"][start:start + 4]
        assert len({row["old_policy_hash"] for row in repeated}) == 1
        assert len({row["frozen_batch_hash"] for row in repeated}) == 1
        assert [row["epoch"] for row in repeated] == [0, 1, 2, 3]
    assert report["final_evaluation_used"] is False
    assert report["clinical_deficit_probability"] is None
    trajectory = rollout_policy(load_policy(tmp_path / "checkpoint.pt"), OneStepSimulator(), seed=1001)
    assert trajectory.total_reward == 1


def test_resume_inside_minibatch_epochs_reuses_exact_old_rollout(tmp_path):
    settings = config(max_gradient_steps=12, minibatch_size=2, checkpoint_interval=3)
    cancel = {"value": False}
    first = train_patient_ppo(OneStepSimulator, *partitions(), output_dir=tmp_path / "resume", config=settings,
        cancelled=lambda: cancel["value"], progress=lambda _: cancel.update(value=True))
    assert first.status == "cancelled" and first.gradient_steps == 3
    state = json.loads((tmp_path / "resume" / "result.json").read_text())
    assert state["pending_old_policy_batch"] is True
    resumed = train_patient_ppo(OneStepSimulator, *partitions(), output_dir=tmp_path / "resume", config=settings, resume=True)
    whole = train_patient_ppo(OneStepSimulator, *partitions(), output_dir=tmp_path / "whole", config=settings)
    assert resumed.latest_checkpoint_hash == whole.latest_checkpoint_hash
    assert resumed.selected_checkpoint_hash == whole.selected_checkpoint_hash
    assert resumed.optimization_environment_steps == whole.optimization_environment_steps
    with pytest.raises(ValueError, match="resume contract changed"):
        train_patient_ppo(OneStepSimulator, *partitions(), output_dir=tmp_path / "resume",
                          config=replace(settings, clip_epsilon=.1), resume=True)


@pytest.mark.parametrize("role", [WorldRole.FINAL_EVALUATION, WorldRole.STRESS])
def test_final_worlds_cannot_enter_ppo(tmp_path, role):
    opt, selection = partitions()
    with pytest.raises(ValueError, match="expected selection"):
        train_patient_ppo(OneStepSimulator, opt, replace(selection, role=role), output_dir=tmp_path)


def test_sampler_change_and_overlapping_worlds_rejected(tmp_path):
    opt, selection = partitions()
    with pytest.raises(ValueError, match="overlap"):
        train_patient_ppo(OneStepSimulator, opt, replace(selection, seeds=(1,)), output_dir=tmp_path)
    world = WorldGeneratorConfig(translation_scale_mm=(1, 1, 1))
    with pytest.raises(ValueError, match="actual simulator sampler"):
        train_patient_ppo(OneStepSimulator, replace(opt, generator=world), replace(selection, generator=world), output_dir=tmp_path)


def test_cancellation_and_actual_gradient_step_budget_are_strict(tmp_path):
    cancelled = train_patient_ppo(OneStepSimulator, *partitions(), output_dir=tmp_path / "cancel",
                                  config=config(), cancelled=lambda: True)
    assert cancelled.gradient_steps == cancelled.optimization_environment_steps == 0
    assert cancelled.selected_selection_return is None
    result = train_patient_ppo(OneStepSimulator, *partitions(), output_dir=tmp_path / "budget",
                               config=config(max_environment_steps=3, max_gradient_steps=7, minibatch_size=2))
    assert result.gradient_steps == 7
    assert result.optimization_environment_steps == 3
    assert result.status == "gradient_budget"


def test_runtime_reward_or_geometry_mutation_invalidates_ppo(tmp_path):
    class Mutating(OneStepSimulator):
        def step(self, action):
            result = super().step(action)
            self.decision_model_hash = "changed"
            return result
    with pytest.raises(ValueError, match="decision model changed"):
        train_patient_ppo(Mutating, *partitions(), output_dir=tmp_path, config=config())


@pytest.mark.parametrize("trainer,settings", [
    (train_patient_ppo, config(max_gradient_steps=1)),
    (learning.train_patient_policy, learning.TrainingConfig(max_gradient_steps=1)),
])
def test_resume_refuses_changed_imported_numerical_dependency(tmp_path, monkeypatch, trainer, settings):
    trainer(OneStepSimulator, *partitions(), output_dir=tmp_path, config=settings)
    dependency_hashes = learning.numerical_source_hashes()
    assert {"geometry.py", "native_resection.py", "simulation.py", "worlds.py"} <= dependency_hashes.keys()
    dependency_hashes["native_resection.py"] = "changed-imported-helper"
    monkeypatch.setattr(learning, "numerical_source_hashes", lambda: dependency_hashes)
    with pytest.raises(ValueError, match="resume contract changed"):
        trainer(OneStepSimulator, *partitions(), output_dir=tmp_path, config=settings, resume=True)
