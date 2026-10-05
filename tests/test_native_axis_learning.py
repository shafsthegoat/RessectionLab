"""Tiny dynamic-inventory integration checks; no patient training or timing claims."""
import json
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from resectionlab.geometry import AccessWindow
from resectionlab.learning import (MaskedPatientPolicy, TrainingConfig, load_policy,
                                  rollout_policy, train_patient_policy)
from resectionlab.native_axis_simulation import AxisColumnNativeSimulator, CommittedTransitionInterrupted
from resectionlab.native_proposals import AxisColumnProposalConfig
from resectionlab.native_resection import NATIVE_GENERIC_TOOLS, NativeResectionConfig
from resectionlab.simulation import InvalidActionError
from resectionlab.worlds import generate_partitions


def fixture(*, primary_cap=4):
    tissue = np.ones((7, 7, 8), bool)
    labels = np.zeros(tissue.shape, np.int16)
    labels[1:6, 1:6, 2:7] = 1
    config = NativeResectionConfig(tissue, labels, np.eye(4),
        AccessWindow((3, 3, -.5), (0, 0, 1), 4), NATIVE_GENERIC_TOOLS,
        "synthetic-axis-learner-fixture-v1", "explicit synthetic support", case_id="synthetic_axis_learner")
    return AxisColumnNativeSimulator(config,
        proposal_config=AxisColumnProposalConfig(((0, 0), (1, 0)), primary_cap), max_steps=2)


def settings():
    return TrainingConfig(seed=11, hidden_features=16, max_gradient_steps=2,
        max_environment_steps=32, max_wall_seconds=10., max_episode_steps=3,
        episodes_per_update=2, checkpoint_interval=1)


def panels(sim):
    return generate_partitions(sim.case_hash, sim.config.world_generator, 20261004,
        optimization=2, selection=1, final_evaluation=1, stress=1)


def test_row_order_and_masked_padding_do_not_change_candidate_scores():
    sim = fixture()
    observation = sim.observation()
    assert observation.action_mask.all()  # The adapter emits only certified rows.
    policy = MaskedPatientPolicy(15, 6, 16)
    order = np.r_[0, np.arange(len(observation.action_ids)-1, 0, -1)]
    permuted = SimpleNamespace(action_features=observation.action_features[order],
        action_mask=observation.action_mask[order], state_features=observation.state_features,
        action_ids=tuple(observation.action_ids[index] for index in order))
    padded = SimpleNamespace(action_features=np.vstack((observation.action_features, np.full((4, 15), 1e6))),
        action_mask=np.r_[observation.action_mask, np.zeros(4, bool)], state_features=observation.state_features,
        action_ids=observation.action_ids + tuple(f"PAD:{index}" for index in range(4)))
    with torch.no_grad():
        original = policy(observation)[0]
        torch.testing.assert_close(policy(permuted)[0], original[order])
        logits = policy(padded)[0]
        torch.testing.assert_close(logits[:len(original)], original)
        assert torch.isneginf(logits[len(original):]).all()
        assert torch.softmax(logits, 0)[len(original):].sum() == 0


def test_current_state_ids_change_but_policy_row_contract_and_replay_survive():
    sim = fixture()
    initial = sim.observation()
    action = sim.proposed_actions()[1].action_id
    clone = sim.clone()
    next_state = clone.step(action).observation
    assert set(initial.action_ids[1:]).isdisjoint(next_state.action_ids[1:])
    with pytest.raises(InvalidActionError, match="stale"):
        clone.step(action)
    policy = MaskedPatientPolicy(15, 6, 16)
    assert policy(initial)[0].shape == initial.action_mask.shape
    assert policy(next_state)[0].shape == next_state.action_mask.shape
    replay = sim.fresh()
    replay.step(action)
    assert replay.observation().action_ids == next_state.action_ids
    np.testing.assert_array_equal(replay.observation().action_features, next_state.action_features)
    stopped = clone.step("STOP")
    assert stopped.terminated and stopped.observation.action_ids == ("STOP",)
    assert policy(stopped.observation)[0].shape == (1,)


def test_distinct_spatial_cuts_can_alias_actor_features_despite_different_continuations():
    sim = fixture()
    observation = sim.observation()
    # Two actual certified fine-tool cuts have different entries and action IDs.
    left, right = observation.action_ids[1], observation.action_ids[3]
    assert left != right
    assert sim._action_provenance[left]["entry_mm"] != sim._action_provenance[right]["entry_mm"]
    np.testing.assert_array_equal(observation.action_features[1], observation.action_features[3])
    policy = MaskedPatientPolicy(15, 6, 16)
    logits, _ = policy(observation)
    assert logits[1] == logits[3]
    returns = []
    first_rewards = []
    for action in (left, right):
        branch = sim.clone()
        transition = branch.step(action)
        first_rewards.append(transition.reward)
        # At most one further cut remains; STOP is present, so this is the
        # exact best nominal continuation for these deterministic tiny states.
        returns.append(transition.reward + max(branch.nominal_action_value(candidate)
                                               for candidate in branch.proposed_actions()))
    assert first_rewards == pytest.approx([4.33, 4.33])
    assert returns == pytest.approx([39.43, 39.68])
    assert returns[0] != returns[1]


def test_tiny_raw_training_updates_actor_over_dynamic_inventory(tmp_path):
    sim = fixture()
    worlds = panels(sim)
    seen = []
    def factory():
        instance = sim.clone()
        original = instance.step
        def checked(action):
            observation = instance.observation()
            assert isinstance(action, int) and observation.action_mask[action]
            result = original(action)
            seen.append((len(observation.action_ids), observation.action_ids[action], result.terminated))
            return result
        instance.step = checked
        return instance
    result = train_patient_policy(factory, worlds.optimization, worlds.selection,
        config=settings(), output_dir=tmp_path)
    saved = json.loads((tmp_path / "result.json").read_text())
    contract = json.loads((tmp_path / "contract.json").read_text())
    assert result.gradient_steps == 2 and saved["actor_parameters_changed"]
    assert result.latest_checkpoint_hash != result.initial_checkpoint_hash
    assert contract["decision_model_hash"] == sim.decision_model_hash
    assert contract["input_profile"]["profile_id"] == "RAW"
    assert contract["partitions"]["optimization"]["seeds"] == list(worlds.optimization.seeds)
    assert any(action.startswith("AXISv1:") for _, action, _ in seen)
    assert all(count <= sim.config.max_actions == 5 for count, _, _ in seen)
    policy = load_policy(tmp_path / "checkpoint.pt", expected_input_profile="RAW")
    replay = rollout_policy(policy, sim.fresh(), seed=worlds.selection.seeds[0], max_steps=3)
    assert replay.environment_steps <= 3 and replay.total_reward == pytest.approx(result.selected_selection_return)


def test_same_source_with_changed_proposal_rule_is_not_same_learner_world(tmp_path):
    first, different = fixture(), fixture(primary_cap=2)
    assert first.case_hash == different.case_hash
    assert first.decision_model_hash != different.decision_model_hash
    worlds = panels(first)
    calls = {"count": 0}
    def factory():
        calls["count"] += 1
        return (first if calls["count"] == 1 else different).clone()
    with pytest.raises(ValueError, match="decision model changed"):
        train_patient_policy(factory, worlds.optimization, worlds.selection, config=settings(), output_dir=tmp_path)
    assert not (tmp_path / "checkpoint.pt").exists()
    failure = json.loads((tmp_path / "failures.jsonl").read_text())
    assert failure["status"] == "failed"


def test_committed_interrupt_is_failure_not_a_resumable_training_result(tmp_path):
    sim = fixture()
    worlds = panels(sim)
    instances = []
    def factory():
        instance = sim.clone()
        instances.append(instance)
        original = instance.step
        def interrupted(_action):
            # Deterministic tiny regression of an exception after native commit;
            # it does not pretend the common learner can resume that transition.
            action = instance.proposed_actions()[1].action_id
            result = original(action)
            raise CommittedTransitionInterrupted(result.info, result.reward)
        instance.step = interrupted
        return instance
    with pytest.raises(CommittedTransitionInterrupted):
        train_patient_policy(factory, worlds.optimization, worlds.selection, config=settings(), output_dir=tmp_path)
    assert instances[-1].removed_mask.any() and len(instances[-1]._history) == 1
    assert not (tmp_path / "result.json").exists()
    failure = json.loads((tmp_path / "failures.jsonl").read_text())
    assert failure["exception"] == "CommittedTransitionInterrupted"
    # A generic failure receipt does not serialize this executed transition.
    assert "history" not in failure and "reward" not in failure
