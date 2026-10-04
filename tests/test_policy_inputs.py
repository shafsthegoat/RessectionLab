"""Fixed actor units change inputs only, with auditable checkpoint semantics."""
import copy
from dataclasses import replace
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from resectionlab import learning
from resectionlab.policy_inputs import FEATURE_UNITS_DIVISORS, policy_input_profile
from resectionlab.procedural_learning import (FEATURE_STUDY_ID, make_procedural_test_target,
    make_native_procedural_fixture, train_procedural_native_policy, validate_procedural_checkpoint,
    load_frozen_procedural_policy, train_procedural_adapted_policy)
from resectionlab.worlds import generate_partitions


def seeded(profile):
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(101)
        return learning.MaskedPatientPolicy(15, 6, 16, input_profile=profile)


def observation():
    values = np.arange(60, dtype=np.float32).reshape(4, 15) + 1
    return SimpleNamespace(action_features=values, state_features=np.arange(6, dtype=np.float32) / 6,
        action_ids=("STOP", "CUT1", "CUT2", "BLOCKED"), action_mask=np.array([True, True, True, False]))


def config(**changes):
    return replace(learning.TrainingConfig(seed=11, hidden_features=16, max_gradient_steps=4,
        max_environment_steps=128, max_episode_steps=4, max_wall_seconds=20.,
        episodes_per_update=2, checkpoint_interval=1), **changes)


def native_fixture():
    target, factory = make_procedural_test_target()
    worlds = generate_partitions(target.case_hash, factory().config.world_generator, 20261004,
        optimization=3, selection=2, final_evaluation=3, stress=2, planning_hash=target.planning_hash)
    return target, factory, worlds


def test_registry_exactly_matches_preregistered_feature_units():
    declaration = json.loads((Path(__file__).parents[1] /
        "manifests/experiments/procedural-native-feature-units-v1.json").read_text())
    for name, record in declaration["profiles"].items():
        profile = policy_input_profile(name)
        assert record == {**profile.to_dict(), "profile_content_hash": profile.fingerprint}
        mutated = profile.to_dict()
        mutated["action_divisors"][1] = 9
        assert profile.to_dict() != mutated
    with pytest.raises(ValueError, match="15\\+6"):
        learning.MaskedPatientPolicy(12, 6, input_profile="FEATURE_UNITS")


def test_raw_is_exact_legacy_arithmetic_and_paired_tensors_critic_unchanged():
    raw, units, obs = seeded("RAW"), seeded("FEATURE_UNITS"), observation()
    actions = torch.tensor(obs.action_features)
    state = torch.tensor(obs.state_features)
    expected = raw.actor(torch.cat((actions, state.expand(len(actions), -1)), dim=1)).squeeze(-1)
    expected[3] = -torch.inf
    assert torch.equal(raw(obs)[0], expected)
    assert torch.equal(raw(obs)[1], units(obs)[1])
    assert learning.trainable_parameter_hash(raw) == learning.trainable_parameter_hash(units)
    assert learning.policy_hash(raw) != learning.policy_hash(units)
    assert set(units.state_dict()) - set(raw.state_dict()) == {"action_divisors"}
    assert not raw._buffers
    assert torch.equal(units.actor_inputs(actions), actions / torch.tensor(FEATURE_UNITS_DIVISORS))
    assert not torch.equal(raw(obs)[0][:3], units(obs)[0][:3])


def test_gradient_reaches_actor_with_fixed_buffer_and_mask_is_preserved():
    policy, obs = seeded("FEATURE_UNITS"), observation()
    original = copy.deepcopy(policy.state_dict())
    logits, value = policy(obs)
    distribution = torch.distributions.Categorical(logits=logits)
    assert distribution.probs[3] == 0
    loss = -distribution.log_prob(torch.tensor(1)) * (12.0 - value.detach()) + .5 * (12.0 - value).square()
    loss.backward()
    assert sum(float(parameter.grad.square().sum()) for parameter in policy.actor.parameters()) > 0
    optimizer = torch.optim.Adam(policy.parameters(), lr=.003)
    optimizer.step()
    assert learning.policy_hash(policy.actor) != learning.policy_hash({key.removeprefix("actor."): value
        for key, value in original.items() if key.startswith("actor.")})
    assert torch.equal(policy.action_divisors, original["action_divisors"])
    assert not policy.action_divisors.requires_grad and policy.action_divisors.grad is None


def test_legacy_raw_load_and_declared_scaled_checkpoint_roundtrip(tmp_path):
    for name in ("RAW", "FEATURE_UNITS"):
        policy = seeded(name)
        path = tmp_path / (name + ".pt")
        record = {"dimensions": list(policy.dimensions), "policy": policy.state_dict(),
                  "policy_hash": learning.policy_hash(policy)}
        if name != "RAW":
            record.update(policy.checkpoint_profile())
        torch.save(record, path)
        loaded = learning.load_policy(path, expected_input_profile=name)
        assert torch.equal(policy(observation())[0], loaded(observation())[0])
        with pytest.raises(ValueError, match="profile differs"):
            learning.load_policy(path, expected_input_profile="FEATURE_UNITS" if name == "RAW" else "RAW")
        if name == "FEATURE_UNITS":
            record["policy"]["action_divisors"][1] += 1
            record["policy_hash"] = learning.policy_hash(record["policy"])
            torch.save(record, path)
            with pytest.raises(ValueError, match="divisor buffer"):
                learning.load_policy(path)


def test_runtime_buffer_tampering_is_refused():
    policy = seeded("FEATURE_UNITS")
    policy.action_divisors[1] = 1
    with pytest.raises(ValueError, match="buffer changed"):
        policy(observation())


def test_feature_units_preserve_native_model_and_resume_actual_updates(tmp_path, monkeypatch):
    _target, factory, worlds = native_fixture()
    sim = factory()
    before_model, before_features = sim.decision_model_hash, sim.observation().action_features.copy()
    seeded("FEATURE_UNITS")(sim.observation())
    assert sim.decision_model_hash == before_model
    assert np.array_equal(sim.observation().action_features, before_features)
    stopped = {"value": False}
    first = learning.train_patient_policy(factory, worlds.optimization, worlds.selection, config=config(),
        output_dir=tmp_path / "resume", input_profile="FEATURE_UNITS",
        cancelled=lambda: stopped["value"], progress=lambda _: stopped.update(value=True))
    assert first.gradient_steps == 1 and first.status == "cancelled"
    def forbidden(*args, **kwargs):
        raise AssertionError("Mismatched profile created Adam")
    with monkeypatch.context() as patch:
        patch.setattr(torch.optim, "Adam", forbidden)
        with pytest.raises(ValueError, match="profile differs"):
            learning.train_patient_policy(factory, worlds.optimization, worlds.selection, config=config(),
                output_dir=tmp_path / "resume", input_profile="RAW", resume=True)
    resumed = learning.train_patient_policy(factory, worlds.optimization, worlds.selection, config=config(),
        output_dir=tmp_path / "resume", input_profile="FEATURE_UNITS", resume=True)
    uninterrupted = learning.train_patient_policy(factory, worlds.optimization, worlds.selection, config=config(),
        output_dir=tmp_path / "whole", input_profile="FEATURE_UNITS")
    assert resumed.gradient_steps == uninterrupted.gradient_steps == 4
    assert resumed.latest_checkpoint_hash == uninterrupted.latest_checkpoint_hash
    assert resumed.latest_checkpoint_hash != resumed.initial_checkpoint_hash
    record = json.loads((tmp_path / "resume/result.json").read_text())
    assert record["actor_parameters_changed"] is True
    assert record["input_profile_hash"] == policy_input_profile("FEATURE_UNITS").fingerprint


@pytest.fixture(scope="module", params=("RAW", "FEATURE_UNITS"))
def shared(request, tmp_path_factory):
    profile = request.param
    target, factory, worlds = native_fixture()
    members = make_native_procedural_fixture(target, input_profile=profile, study_id=FEATURE_STUDY_ID)
    result = train_procedural_native_policy(members, excluded_targets=(target,), config=config(),
        output_dir=tmp_path_factory.mktemp(profile) / "offline", input_profile=profile, study_id=FEATURE_STUDY_ID)
    assert result["status"] == "completed"
    return profile, target, factory, worlds, result


def test_profile_bound_pretrain_frozen_and_private_adaptation(shared, tmp_path, monkeypatch):
    profile, target, factory, worlds, result = shared
    checkpoint = result["checkpoint_path"]
    args = dict(target=target, simulator=factory(), hidden_features=16,
                input_profile=profile, study_id=FEATURE_STUDY_ID)
    validation = validate_procedural_checkpoint(checkpoint, **args)
    assert validation["study_id"] == FEATURE_STUDY_ID
    original_bytes = Path(checkpoint).read_bytes()
    with monkeypatch.context() as patch:
        patch.setattr(torch.optim, "Adam", lambda *a, **kw: pytest.fail("frozen constructed optimizer"))
        frozen = load_frozen_procedural_policy(checkpoint, **args)
        learning.rollout_policy(frozen, factory(), seed=worlds.selection.seeds[0], max_steps=4)
        assert all(not p.requires_grad and p.grad is None for p in frozen.parameters())
    states, optimizer = [], torch.optim.Adam
    def fresh(*args, **kwargs):
        item = optimizer(*args, **kwargs)
        states.append(len(item.state))
        return item
    monkeypatch.setattr(torch.optim, "Adam", fresh)
    adapted = train_procedural_adapted_policy(factory, worlds.optimization, worlds.selection,
        checkpoint=checkpoint, target=target, config=config(max_gradient_steps=2), output_dir=tmp_path,
        input_profile=profile, study_id=FEATURE_STUDY_ID)
    assert states == [0]
    assert adapted.initial_checkpoint_hash == validation["policy_hash"]
    assert adapted.gradient_steps == 2 and Path(checkpoint).read_bytes() == original_bytes
    initial = learning.load_policy(tmp_path / "initial.pt", expected_input_profile=profile)
    assert learning.trainable_parameter_hash(initial) == validation["trainable_parameter_hash"]


def test_cross_profile_and_original_study_refuse_adaptation_before_adam(shared, tmp_path, monkeypatch):
    profile, target, factory, worlds, result = shared
    other = "FEATURE_UNITS" if profile == "RAW" else "RAW"
    monkeypatch.setattr(torch.optim, "Adam", lambda *a, **kw: pytest.fail("mismatch created optimizer"))
    with pytest.raises(ValueError, match="profile"):
        train_procedural_adapted_policy(factory, worlds.optimization, worlds.selection,
            checkpoint=result["checkpoint_path"], target=target, config=config(), output_dir=tmp_path / "other",
            input_profile=other, study_id=FEATURE_STUDY_ID)
    with pytest.raises(ValueError, match="study|provenance"):
        train_procedural_adapted_policy(factory, worlds.optimization, worlds.selection,
            checkpoint=result["checkpoint_path"], target=target, config=config(), output_dir=tmp_path / "legacy",
            input_profile=profile)
