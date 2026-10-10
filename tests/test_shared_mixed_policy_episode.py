"""Shared mixed-mode backend and policy integration."""
from __future__ import annotations

import pytest

from resectionlab.development_episode import make_development_task, plan_development_episode
from resectionlab.sequential_spatial_observation import SequentialSpatialObservation
from resectionlab.shared_episode import (execute_action_sequence, execute_search_episode,
                            verify_strategy_replay)


def test_search_and_declared_multistep_sequence_use_same_native_transition():
    task = make_development_task()
    first = task.observation()
    assert type(first) is SequentialSpatialObservation
    assert {"stop", "aspirate"} <= set(first.action_modes)
    assert not first.observed_probe_contact_grid.any()

    plan, _, _ = plan_development_episode(task, "scripted")
    changing = make_development_task()
    changing.step(plan["actions"][0])
    before_probe_removed = changing._engine.removed_mask.copy()
    before_probe = changing.observation()
    assert not before_probe.observed_probe_contact_grid.any()
    changing.step(plan["actions"][1])
    after_probe = changing.observation()
    assert after_probe.observed_probe_contact_grid.any()
    assert not (changing._engine.removed_mask ^ before_probe_removed).any()
    assert after_probe.fingerprint != before_probe.fingerprint
    scripted = execute_action_sequence(task, plan["actions"])
    assert scripted["method"] == "SCRIPTED_INTERFACE_TEST"
    assert [row["action_mode"] for row in scripted["decisions"]] == [
        "aspirate", "probe", "aspirate", "probe", "aspirate", "stop"]
    assert verify_strategy_replay(task, scripted)

    search = execute_search_episode(task, max_calls=24, beam_width=2, seconds=10.)
    assert search["method"] == "SEARCH"
    assert search["environment_contract_hash"] == scripted["environment_contract_hash"]
    assert search["source_hash"] == scripted["source_hash"]
    assert search["decisions"][0]["observation_before"] == scripted["decisions"][0]["observation_before"]
    assert verify_strategy_replay(task, search)
    assert task.metrics()["steps"] == 0


def test_mixed_task_refuses_legacy_spatial_policy_before_transition():
    from resectionlab.spatial_policy import SpatialPolicy, SpatialPolicyConfig
    from resectionlab.shared_episode import execute_policy_episode, identity_for_untrained_spatial_policy
    actor = SpatialPolicy(SpatialPolicyConfig(encoder_channels=(2, 2), hidden_features=4,
                                              ray_samples=2)).eval()
    identity = identity_for_untrained_spatial_policy(actor, policy_id="incompatible-v1-control")
    task = make_development_task()
    with pytest.raises(ValueError, match="not built for this task observation contract"):
        execute_policy_episode(task, actor, identity)
    assert task.metrics()["steps"] == 0
