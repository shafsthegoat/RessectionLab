"""Small generated software fixtures; no trained weights or patient records."""
from __future__ import annotations

import numpy as np
import pytest
import torch

from resectionlab.native_spatial_task import make_native_opening_task
from resectionlab.spatial_policy import SpatialPolicy, SpatialPolicyConfig

from resectionlab.sequential_spatial_observation import SequentialSpatialObservation
from resectionlab.shared_episode import (execute_policy_episode, execute_search_episode,
                            identity_for_untrained_spatial_policy, verify_strategy_replay,
                            PolicyIdentity)


def small_actor():
    torch.manual_seed(17)
    return SpatialPolicy(SpatialPolicyConfig(encoder_channels=(2, 2),
        hidden_features=4, ray_samples=2)).eval()


def test_search_and_executable_untrained_actor_share_native_transition_contract():
    torch.set_num_threads(1)
    task = make_native_opening_task()
    actor = small_actor()
    identity = identity_for_untrained_spatial_policy(actor, policy_id="generated-untrained-17")
    policy = execute_policy_episode(task, actor, identity)
    search = execute_search_episode(task, max_calls=48, beam_width=4, seconds=20.)
    assert task.metrics()["steps"] == 0  # Both arms used independent fresh tasks.
    assert search["method"] == "SEARCH"
    assert policy["method"] == "POLICY_SOFTWARE_CONTROL"
    assert search["environment_contract_hash"] == policy["environment_contract_hash"]
    assert search["source_hash"] == policy["source_hash"]
    assert search["observation_track"] == policy["observation_track"] == "synthetic_scan"
    assert search["decisions"][0]["observation_before"] == policy["decisions"][0]["observation_before"]
    for arm in (search, policy):
        assert arm["action_ids"] == [row["action_id"] for row in arm["decisions"]]
        assert all(row["physical_transition"]["action_id"] == row["action_id"] for row in arm["decisions"])
        assert all("target_removed_mm3" not in row["physical_transition"]
                   and "reward" not in row["physical_transition"] for row in arm["decisions"])
        assert arm["clinical_validation"] == "none"
    assert policy["policy_identity"]["checkpoint_sha256"] is None
    assert policy["policy_identity"]["training_status"] == "untrained_software_control"
    assert verify_strategy_replay(task, policy)
    assert verify_strategy_replay(task, search)
    altered = dict(search)
    altered["action_ids"] = list(reversed(search["action_ids"]))
    if altered["action_ids"] != search["action_ids"]:
        with pytest.raises((ValueError, RuntimeError)):
            verify_strategy_replay(task, altered)


def test_mixed_mode_observation_is_explicit_and_legacy_policy_fails_before_step():
    base = make_native_opening_task().observation()
    contact = np.zeros(base.image_channels.shape[1:], bool)
    modes = ("stop", *["probe" if index == 1 else "aspirate"
                         for index in range(1, len(base.action_ids))])
    mixed = SequentialSpatialObservation(base, modes, contact)
    assert mixed.action_ids == base.action_ids
    assert mixed.action_mask is base.action_mask
    assert mixed.fingerprint != base.fingerprint
    with pytest.raises(TypeError, match="SpatialObservation DTO"):
        small_actor().act(mixed)
    with pytest.raises(ValueError, match="Action modes"):
        SequentialSpatialObservation(base, ("stop",) * len(base.action_ids), contact)
    with pytest.raises(ValueError, match="covered boolean"):
        SequentialSpatialObservation(base, modes, np.ones_like(contact, dtype=np.float32))


def test_spatial_policy_parameter_identity_rejects_changed_model():
    actor = small_actor()
    identity = identity_for_untrained_spatial_policy(actor, policy_id="generated-untrained-17")
    with torch.no_grad():
        next(actor.parameters()).add_(1.)
    with pytest.raises(ValueError, match="parameters or architecture"):
        execute_policy_episode(make_native_opening_task(), actor, identity)


def test_unbound_trained_label_and_patient_track_fail_closed():
    actor = small_actor()
    untrained = identity_for_untrained_spatial_policy(actor, policy_id="generated-control")
    trained = PolicyIdentity("claimed-RL", untrained.architecture_hash, untrained.parameter_hash,
        untrained.observation_contract, "trained_checkpoint", "sha256:" + "a" * 64,
        "sha256:" + "b" * 64)
    with pytest.raises(ValueError, match="TRAINED_CHECKPOINT_LOADER_REQUIRED"):
        execute_policy_episode(make_native_opening_task(), actor, trained)
    task = make_native_opening_task()
    object.__setattr__(task.case, "track", "annotation_assisted")
    with pytest.raises(ValueError, match="GENERATED_ONLY"):
        execute_policy_episode(task, actor, untrained)
