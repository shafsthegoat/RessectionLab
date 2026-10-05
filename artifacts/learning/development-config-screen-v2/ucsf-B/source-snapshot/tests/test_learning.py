"""Actual gradients, held-out selection, budgets, and patient isolation."""

from dataclasses import replace
import json
from types import SimpleNamespace

import numpy as np
import pytest

from resectionlab.learning import (TrainingConfig, clone_checkpoint_policy, load_policy, policy_hash,
                                    rollout_policy, train_patient_policy)
from resectionlab.worlds import WorldGeneratorConfig, WorldPartitionManifest, WorldRole


class DelayedRewardSimulator:
    """Exact toy optimum: OPEN (-0.1), FINISH (+4), STOP = 3.9.

    TEMPTATION terminates at +0.5. This fixture tests optimization only, and
    provides no anatomical or uncertainty validation.
    """

    decision_model_hash = "sha256:fixed-delayed-reward-toy-v1"
    case_hash = "toy-case"
    world_generator_fingerprint = WorldGeneratorConfig().fingerprint

    def reset(self, seed=0):
        self.stage = 0
        self.total = 0.0
        return self.observation()

    def observation(self):
        return SimpleNamespace(action_ids=("STOP", "OPEN" if self.stage == 0 else "FINISH", "TEMPTATION"),
                               action_features=np.eye(3, dtype=np.float32),
                               state_features=np.array([self.stage, 1.0], dtype=np.float32),
                               action_mask=np.array([True, self.stage < 2, self.stage == 0]))

    def step(self, action):
        assert self.observation().action_mask[action], "invalid action reached simulator"
        if action == 0:
            reward, done = 0.0, True
        elif action == 2:
            reward, done = 0.5, True
        elif self.stage == 0:
            reward, done = -0.1, False
            self.stage = 1
        else:
            reward, done = 4.0, False
            self.stage = 2
        self.total += reward
        return SimpleNamespace(observation=self.observation(), reward=reward, terminated=done, info={})

    def metrics(self):
        return {"return_surrogate": self.total, "fixture": "deterministic_toy"}


def manifests():
    config = WorldGeneratorConfig()
    return (WorldPartitionManifest(WorldRole.OPTIMIZATION, "toy-case", config, (1, 2, 3)),
            WorldPartitionManifest(WorldRole.SELECTION, "toy-case", config, (101, 102)))


def small_config(**kwargs):
    return replace(TrainingConfig(max_environment_steps=3000, max_gradient_steps=80,
                                  max_wall_seconds=30, max_episode_steps=3,
                                  episodes_per_update=4, checkpoint_interval=8,
                                  hidden_features=16, seed=11), **kwargs)


def test_actual_updates_reach_known_toy_optimum_and_keep_initial(tmp_path):
    result = train_patient_policy(DelayedRewardSimulator, *manifests(),
                                  config=small_config(), output_dir=tmp_path)
    initial = load_policy(tmp_path / "initial.pt")
    selected = load_policy(tmp_path / "checkpoint.pt")
    assert result.gradient_steps == 80
    assert result.latest_checkpoint_hash != result.initial_checkpoint_hash
    assert policy_hash(initial) == result.initial_checkpoint_hash
    learned = rollout_policy(selected, DelayedRewardSimulator(), seed=1001, max_steps=3)
    assert learned.total_reward == pytest.approx(3.9)
    assert learned.actions == ("OPEN", "FINISH", "STOP")
    assert rollout_policy(None, DelayedRewardSimulator(), seed=1001).total_reward == 0.0
    saved = json.loads((tmp_path / "result.json").read_text())
    assert saved["clinical_deficit_probability"] is None
    assert saved["optimization_environment_steps"] > result.gradient_steps
    assert saved["selection_environment_steps"] > 0
    assert saved["actor_parameters_changed"] is True
    assert all(set(row) >= {"gradient_steps", "mean_return", "checkpoint_hash"}
               for row in saved["selection_history"])


@pytest.mark.parametrize("bad_role", [WorldRole.FINAL_EVALUATION, WorldRole.STRESS])
def test_trainer_refuses_final_or_stress_worlds(tmp_path, bad_role):
    opt, selection = manifests()
    with pytest.raises(ValueError, match="expected selection"):
        train_patient_policy(DelayedRewardSimulator, opt, replace(selection, role=bad_role),
                             output_dir=tmp_path)


def test_overlap_and_generator_changes_are_rejected(tmp_path):
    opt, selection = manifests()
    with pytest.raises(ValueError, match="overlap"):
        train_patient_policy(DelayedRewardSimulator, opt, replace(selection, seeds=(1,)), output_dir=tmp_path)
    different = replace(selection, generator=WorldGeneratorConfig(translation_scale_mm=(1, 0, 0)))
    with pytest.raises(ValueError, match="same frozen"):
        train_patient_policy(DelayedRewardSimulator, opt, different, output_dir=tmp_path)


def test_cancellation_resume_keeps_updates_rng_and_total_budget(tmp_path):
    interrupted = tmp_path / "interrupted"
    continuous = tmp_path / "continuous"
    config = small_config(max_gradient_steps=16, checkpoint_interval=4)
    cancel_state = {"cancel": False}
    def progress(snapshot):
        cancel_state["cancel"] = True
    first = train_patient_policy(DelayedRewardSimulator, *manifests(), config=config,
                                  output_dir=interrupted, cancelled=lambda: cancel_state["cancel"],
                                  progress=progress)
    assert first.status == "cancelled"
    assert first.gradient_steps == 4
    resumed = train_patient_policy(DelayedRewardSimulator, *manifests(), config=config,
                                    output_dir=interrupted, resume=True)
    whole = train_patient_policy(DelayedRewardSimulator, *manifests(), config=config,
                                 output_dir=continuous)
    assert resumed.gradient_steps == whole.gradient_steps == 16
    assert resumed.latest_checkpoint_hash == whole.latest_checkpoint_hash
    assert resumed.selected_checkpoint_hash == whole.selected_checkpoint_hash
    assert resumed.optimization_environment_steps == whole.optimization_environment_steps
    with pytest.raises(ValueError, match="contract changed"):
        train_patient_policy(DelayedRewardSimulator, *manifests(), config=replace(config, learning_rate=.01),
                             output_dir=interrupted, resume=True)


def test_budget_and_immediate_cancel_are_truthful(tmp_path):
    cancelled = train_patient_policy(DelayedRewardSimulator, *manifests(), output_dir=tmp_path / "cancel",
                                    cancelled=lambda: True)
    assert cancelled.gradient_steps == cancelled.optimization_environment_steps == 0
    assert cancelled.selected_selection_return is None
    result = train_patient_policy(DelayedRewardSimulator, *manifests(), output_dir=tmp_path / "bounded",
                                  config=small_config(max_environment_steps=7))
    assert result.optimization_environment_steps <= 7
    assert result.status == "environment_budget"


def test_checkpoint_clone_isolated_and_scratch_is_not_population_checkpoint(tmp_path):
    source = tmp_path / "source"
    train_patient_policy(DelayedRewardSimulator, *manifests(), output_dir=source,
                         config=small_config(max_gradient_steps=4))
    checkpoint = source / "checkpoint.pt"
    source_bytes = checkpoint.read_bytes()
    first = clone_checkpoint_policy(checkpoint)
    second = clone_checkpoint_policy(checkpoint)
    import torch
    with torch.no_grad():
        next(first.parameters()).add_(1)
    assert policy_hash(first) != policy_hash(second)
    assert policy_hash(second) == policy_hash(load_policy(checkpoint))
    assert checkpoint.read_bytes() == source_bytes
    with pytest.raises(ValueError, match="actual shared training checkpoint"):
        train_patient_policy(DelayedRewardSimulator, *manifests(), output_dir=tmp_path / "case-a",
                             config=small_config(max_gradient_steps=8), shared_checkpoint=checkpoint)


def test_reward_or_tool_mutation_during_rollout_is_rejected(tmp_path):
    class MutatingSimulator(DelayedRewardSimulator):
        def step(self, action):
            result = super().step(action)
            self.decision_model_hash = "changed-reward-or-tool"
            return result
    with pytest.raises(ValueError, match="decision model changed"):
        train_patient_policy(MutatingSimulator, *manifests(), output_dir=tmp_path)


def test_policy_cannot_execute_hidden_or_masked_action(tmp_path):
    class StopOnlySimulator(DelayedRewardSimulator):
        def observation(self):
            observation = super().observation()
            observation.action_mask[1:] = False
            return observation
    result = train_patient_policy(StopOnlySimulator, *manifests(), output_dir=tmp_path,
                                  config=small_config(max_gradient_steps=2))
    assert result.selected_selection_return == 0.0
    replay = rollout_policy(load_policy(tmp_path / "checkpoint.pt"), StopOnlySimulator(), seed=400)
    assert replay.actions == ("STOP",)
    saved = json.loads((tmp_path / "result.json").read_text())
    assert saved["actor_parameters_changed"] is False


def test_manifest_case_and_actual_sampler_are_bound(tmp_path):
    opt, selection = manifests()
    with pytest.raises(ValueError, match="actual simulator case"):
        train_patient_policy(DelayedRewardSimulator, replace(opt, case_hash="other"),
                             replace(selection, case_hash="other"), output_dir=tmp_path / "case")
    changed = WorldGeneratorConfig(translation_scale_mm=(1, 1, 1))
    with pytest.raises(ValueError, match="actual simulator sampler"):
        train_patient_policy(DelayedRewardSimulator, replace(opt, generator=changed),
                             replace(selection, generator=changed), output_dir=tmp_path / "world")
    failure = json.loads((tmp_path / "world" / "failures.jsonl").read_text())
    assert failure["status"] == "failed"


def test_connected_geometry_requires_paid_access_before_removal(tmp_path):
    from resectionlab.simulation import beam_search, make_synthetic_simulator
    from resectionlab.worlds import generate_partitions
    sim = make_synthetic_simulator()
    worlds = generate_partitions(sim.case_hash, sim.config.world_generator, 20261004,
                                 optimization=3, selection=2, final_evaluation=3, stress=2)
    result = train_patient_policy(make_synthetic_simulator, worlds.optimization, worlds.selection,
                                  config=small_config(max_episode_steps=4, max_gradient_steps=16),
                                  output_dir=tmp_path)
    assert result.gradient_steps == 16
    selected = load_policy(tmp_path / "checkpoint.pt")
    replay = rollout_policy(selected, make_synthetic_simulator(), seed=1001, max_steps=4)
    assert replay.total_reward == pytest.approx(beam_search(sim).nominal_score)
    assert replay.total_reward == pytest.approx(1.74)
    assert replay.metrics["simulated_removed_normal_volume_mm3"] == 1.0
    assert replay.metrics["simulated_removed_target_volume_mm3"] == 2.0
    assert replay.actions[0].endswith(":1,1,0")
