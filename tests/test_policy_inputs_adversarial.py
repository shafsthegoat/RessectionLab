"""Independent feature-unit contract attacks; no registered training or final worlds.

The legacy actor below is the mathematical forward path at declaration commit
96e7a3e. Tiny cancelled checkpoints exercise resume gates without gradient work.
"""
from __future__ import annotations

import copy
import json
from pathlib import Path
import shutil
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from torch import nn

from resectionlab import learning
from resectionlab.policy_inputs import policy_input_profile
from resectionlab.procedural_learning import make_procedural_test_target
from resectionlab.worlds import content_hash, generate_partitions


DECLARATION = Path(__file__).resolve().parents[1] / "manifests/experiments/procedural-native-feature-units-v1.json"


class LegacyPolicy(nn.Module):
    def __init__(self):
        super().__init__()
        self.actor = nn.Sequential(nn.Linear(21, 16), nn.Tanh(), nn.Linear(16, 1))
        self.value = nn.Sequential(nn.Linear(6, 16), nn.Tanh(), nn.Linear(16, 1))

    def forward(self, observation):
        actions = torch.as_tensor(np.array(observation.action_features, copy=True), dtype=torch.float32)
        state = torch.as_tensor(np.array(observation.state_features, copy=True), dtype=torch.float32).flatten()
        mask = torch.as_tensor(np.array(observation.action_mask, copy=True), dtype=torch.bool)
        logits = self.actor(torch.cat((actions, state.expand(len(actions), -1)), dim=1)).squeeze(-1)
        return logits.masked_fill(~mask, -torch.inf), self.value(state).squeeze(-1)


def policy(profile="RAW", seed=11):
    with torch.random.fork_rng():
        torch.manual_seed(seed)
        return learning.MaskedPatientPolicy(15, 6, 16, input_profile=profile)


def observation():
    values = np.arange(75, dtype=np.float32).reshape(5, 15) / 10.
    values[:, [1, 2, 3, 4, 12, 13, 14]] *= 648.
    return SimpleNamespace(action_features=values,
        state_features=np.array([.9, .25, .1, .2, 0., 1.], np.float32),
        action_mask=np.array([True, True, False, True, False]),
        action_ids=("STOP", "CUT:1", "CUT:2", "CUT:3", "CUT:4"))


def checkpoint(model):
    return {"dimensions": list(model.dimensions), "policy": copy.deepcopy(model.state_dict()),
        "policy_hash": learning.policy_hash(model), **model.checkpoint_profile()}


@pytest.mark.parametrize("seed", [11, 23, 47])
def test_raw_defaults_preserve_legacy_tensor_bytes_forward_and_gradients(seed):
    with torch.random.fork_rng():
        torch.manual_seed(seed)
        legacy = LegacyPolicy()
    raw, scaled = policy(seed=seed), policy("FEATURE_UNITS", seed)
    for name, expected in legacy.state_dict().items():
        assert raw.state_dict()[name].numpy().tobytes() == expected.numpy().tobytes()
    assert set(raw.state_dict()) == set(legacy.state_dict())
    assert learning.policy_hash(raw) == learning.policy_hash(legacy)
    assert learning.trainable_parameter_hash(raw) == learning.trainable_parameter_hash(scaled)
    assert learning.policy_hash(raw) != learning.policy_hash(scaled)
    old_logits, old_value = legacy(observation())
    logits, value = raw(observation())
    assert torch.equal(logits, old_logits) and torch.equal(value, old_value)
    assert torch.equal(scaled(observation())[1], value)
    for model, outputs in ((legacy, (old_logits, old_value)), (raw, (logits, value))):
        scores, predicted = outputs
        loss = -torch.log_softmax(scores, 0)[1] + predicted.square() / 2
        loss.backward()
    for old, new in zip(legacy.parameters(), raw.parameters(), strict=True):
        assert torch.equal(old.grad, new.grad)


@pytest.mark.parametrize("profile_id", ["RAW", "FEATURE_UNITS"])
def test_declared_profile_order_units_and_hash_are_exact(profile_id):
    declared = json.loads(DECLARATION.read_text())["profiles"][profile_id]
    profile = policy_input_profile(profile_id)
    assert profile.fingerprint == declared["profile_content_hash"]
    assert profile.to_dict() == {key: value for key, value in declared.items() if key != "profile_content_hash"}
    detached = profile.to_dict()
    detached["action_divisors"][1] = 999
    assert profile.to_dict() != detached


@pytest.mark.parametrize("profile_id", ["RAW", "FEATURE_UNITS"])
def test_masked_rows_never_change_actor_or_critic_gradients(profile_id):
    original = policy(profile_id)
    altered = copy.deepcopy(original)
    normal, poisoned = observation(), observation()
    poisoned.action_features[~poisoned.action_mask] *= 10000
    before = normal.action_features.tobytes()
    for model, obs in ((original, normal), (altered, poisoned)):
        logits, value = model(obs)
        assert torch.isneginf(logits[~torch.tensor(obs.action_mask)]).all()
        (-torch.log_softmax(logits, 0)[1] + value.square()).backward()
    assert normal.action_features.tobytes() == before
    for left, right in zip(original.parameters(), altered.parameters(), strict=True):
        assert torch.equal(left.grad, right.grad)
    if profile_id == "FEATURE_UNITS":
        assert original.action_divisors.grad is None and not original.action_divisors.requires_grad


@pytest.mark.parametrize("mutation", ["divisor", "dtype", "selected_only", "missing_metadata",
    "raw_relabel", "order", "state_order", "return_units", "metadata_hash", "unknown"])
def test_coherently_rehashed_profile_or_buffer_tamper_is_rejected(tmp_path, mutation):
    saved = checkpoint(policy("FEATURE_UNITS"))
    if mutation in {"divisor", "dtype"}:
        saved["policy"]["action_divisors"] = saved["policy"]["action_divisors"].clone()
        if mutation == "divisor":
            saved["policy"]["action_divisors"][1] *= 2
        else:
            saved["policy"]["action_divisors"] = saved["policy"]["action_divisors"].double()
    elif mutation == "selected_only":
        saved["selected_policy"] = copy.deepcopy(saved["policy"])
        saved["selected_policy"]["action_divisors"][1] *= 2
        saved["selected_hash"] = learning.policy_hash(saved["selected_policy"])
    elif mutation == "missing_metadata":
        saved.pop("input_profile")
        saved.pop("input_profile_hash")
    elif mutation == "raw_relabel":
        raw = policy_input_profile("RAW")
        saved.update(input_profile=raw.to_dict(), input_profile_hash=raw.fingerprint)
    elif mutation in {"order", "state_order"}:
        key = "action_feature_names" if mutation == "order" else "state_feature_names"
        names = saved["input_profile"][key]
        names[1], names[2] = names[2], names[1]
        saved["input_profile_hash"] = content_hash(saved["input_profile"])
    elif mutation == "return_units":
        saved["input_profile"]["return_divisor"] = 648.
        saved["input_profile_hash"] = content_hash(saved["input_profile"])
    elif mutation == "metadata_hash":
        saved["input_profile_hash"] = "sha256:changed"
    else:
        saved["input_profile"]["profile_id"] = "UNDECLARED"
        saved["input_profile_hash"] = content_hash(saved["input_profile"])
    saved["policy_hash"] = learning.policy_hash(saved["policy"])
    path = tmp_path / "coherently-rehashed.pt"
    torch.save(saved, path)
    with pytest.raises(ValueError, match="(?i)(profile|divisor)"):
        learning.load_policy(path)


def test_legacy_profile_omission_only_loads_raw_without_a_buffer(tmp_path):
    saved = checkpoint(policy())
    saved.pop("input_profile")
    saved.pop("input_profile_hash")
    path = tmp_path / "legacy.pt"
    torch.save(saved, path)
    assert learning.policy_hash(learning.load_policy(path, expected_input_profile="RAW")) == saved["policy_hash"]
    with pytest.raises(ValueError, match="profile"):
        learning.load_policy(path, expected_input_profile="FEATURE_UNITS")
    saved["policy"]["action_divisors"] = torch.ones(15)
    saved["policy_hash"] = learning.policy_hash(saved["policy"])
    torch.save(saved, path)
    with pytest.raises(ValueError, match="RAW.*buffer"):
        learning.load_policy(path)


def test_raw_latest_null_buffer_cannot_hide_behind_a_valid_selected_policy(tmp_path):
    saved = checkpoint(policy())
    saved["selected_policy"] = copy.deepcopy(saved["policy"])
    saved["selected_hash"] = saved["policy_hash"]
    saved["policy"]["action_divisors"] = None
    path = tmp_path / "invalid-unused-latest.pt"
    torch.save(saved, path)
    with pytest.raises(ValueError, match="RAW.*buffer"):
        learning.load_policy(path)


@pytest.fixture(scope="module")
def cancelled_profiles(tmp_path_factory):
    target, factory = make_procedural_test_target()
    panel = generate_partitions(target.case_hash, factory().config.world_generator, 41,
        optimization=2, selection=2, final_evaluation=1, stress=1, planning_hash=target.planning_hash)
    config = learning.TrainingConfig(seed=11, max_gradient_steps=1, max_environment_steps=32,
        max_wall_seconds=10., max_episode_steps=4, episodes_per_update=2, hidden_features=16)
    root = tmp_path_factory.mktemp("cancelled-unit-profiles")
    for profile_id in ("RAW", "FEATURE_UNITS"):
        result = learning.train_patient_policy(factory, panel.optimization, panel.selection,
            config=config, output_dir=root / profile_id, input_profile=profile_id, cancelled=lambda: True)
        assert result.gradient_steps == 0 and result.selected_selection_return is None
    return factory, panel, config, root


@pytest.mark.parametrize("source_profile", ["RAW", "FEATURE_UNITS"])
def test_cross_profile_resume_fails_before_adam_and_preserves_checkpoint(cancelled_profiles,
        tmp_path, monkeypatch, source_profile):
    factory, panel, config, root = cancelled_profiles
    run = tmp_path / "private-run"
    shutil.copytree(root / source_profile, run)
    before = (run / "checkpoint.pt").read_bytes()
    def forbidden(*args, **kwargs):
        raise AssertionError("Cross-profile resume reached optimizer construction")
    monkeypatch.setattr(learning.torch.optim, "Adam", forbidden)
    other = "FEATURE_UNITS" if source_profile == "RAW" else "RAW"
    with pytest.raises(ValueError, match="profile"):
        learning.train_patient_policy(factory, panel.optimization, panel.selection,
            config=config, output_dir=run, input_profile=other, resume=True)
    assert (run / "checkpoint.pt").read_bytes() == before


def test_relabeling_all_profile_and_tensor_hashes_cannot_evade_resume_contract_before_adam(
        cancelled_profiles, tmp_path, monkeypatch):
    factory, panel, config, root = cancelled_profiles
    run = tmp_path / "relabelled-run"
    shutil.copytree(root / "RAW", run)
    path = run / "checkpoint.pt"
    state = torch.load(path, weights_only=True)
    profile = policy_input_profile("FEATURE_UNITS")
    state.update(input_profile=profile.to_dict(), input_profile_hash=profile.fingerprint)
    for key, hash_key in (("policy", "policy_hash"), ("selected_policy", "selected_hash")):
        state[key]["action_divisors"] = torch.tensor(profile.action_divisors)
        state[hash_key] = learning.policy_hash(state[key])
    torch.save(state, path)
    before = path.read_bytes()
    def forbidden(*args, **kwargs):
        raise AssertionError("Resealed profile reached Adam before frozen contract validation")
    monkeypatch.setattr(learning.torch.optim, "Adam", forbidden)
    with pytest.raises(ValueError, match="(?i)(profile|contract)"):
        learning.train_patient_policy(factory, panel.optimization, panel.selection,
            config=config, output_dir=run, input_profile="FEATURE_UNITS", resume=True)
    assert path.read_bytes() == before


@pytest.fixture(scope="module")
def tiny_shared_profiles(tmp_path_factory):
    from resectionlab import procedural_learning as transfer
    target, factory = transfer.make_procedural_test_target()
    members = transfer.make_native_procedural_fixture(target)
    root = tmp_path_factory.mktemp("tiny-shared-unit-profiles")
    settings = learning.TrainingConfig(seed=11, max_gradient_steps=1, max_environment_steps=32,
        max_wall_seconds=10., max_episode_steps=4, episodes_per_update=2, hidden_features=16)
    shared = {}
    for profile_id in ("RAW", "FEATURE_UNITS"):
        shared[profile_id] = transfer.train_procedural_native_policy(members,
            excluded_targets=(target,), config=settings, output_dir=root / profile_id,
            input_profile=profile_id, study_id="procedural-native-feature-units-v1")
        assert shared[profile_id]["status"] == "completed"
    return target, factory, settings, shared


@pytest.mark.parametrize("source_profile", ["RAW", "FEATURE_UNITS"])
def test_cross_profile_adaptation_rejects_before_adam_and_without_copying_shared_weights(
        tiny_shared_profiles, tmp_path, monkeypatch, source_profile):
    from resectionlab import procedural_learning as transfer
    target, factory, config, shared = tiny_shared_profiles
    panel = generate_partitions(target.case_hash, factory().config.world_generator, 20261004,
        optimization=3, selection=2, final_evaluation=3, stress=2, planning_hash=target.planning_hash)
    checkpoint_path = Path(shared[source_profile]["checkpoint_path"])
    original = checkpoint_path.read_bytes()
    output = tmp_path / "rejected"
    other = "FEATURE_UNITS" if source_profile == "RAW" else "RAW"
    def forbidden(*args, **kwargs):
        raise AssertionError("Cross-profile adaptation reached optimizer construction")
    monkeypatch.setattr(learning.torch.optim, "Adam", forbidden)
    with pytest.raises(ValueError, match="(?i)profile"):
        transfer.train_procedural_adapted_policy(factory, panel.optimization, panel.selection,
            checkpoint=checkpoint_path, target=target, config=config, output_dir=output,
            input_profile=other, study_id="procedural-native-feature-units-v1")
    assert checkpoint_path.read_bytes() == original
    assert not (output / "initial.pt").exists()
    assert not (output / "procedural-source.pt").exists()
