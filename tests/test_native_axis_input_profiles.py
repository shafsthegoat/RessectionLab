"""Closed axis profile prerequisites on tiny synthetic anatomy, never a patient."""
from dataclasses import replace
import json
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from resectionlab import learning
from resectionlab.geometry import AccessWindow
from resectionlab.native_axis_accounting import AxisTrainingAccounting, train_axis_policy
from resectionlab.native_axis_policy_schema import axis_observation_contract
from resectionlab.native_axis_simulation import AxisColumnNativeSimulator
from resectionlab.native_proposals import AxisColumnProposalConfig
from resectionlab.native_resection import NativeResectionConfig, NATIVE_GENERIC_TOOLS
from resectionlab.policy_inputs import FEATURE_UNITS_DIVISORS, policy_input_profile
from resectionlab.procedural_learning import native_observation_schema
from resectionlab.simulation import InvalidActionError
from resectionlab.worlds import generate_partitions


def fixture():
    tissue = np.ones((7, 7, 8), bool)
    target = np.zeros(tissue.shape, np.int16)
    target[1:6, 1:6, 2:7] = 1
    config = NativeResectionConfig(tissue, target, np.eye(4),
        AccessWindow((3, 3, -.5), (0, 0, 1), 4), NATIVE_GENERIC_TOOLS,
        "synthetic-axis-input-profiles-v1", "explicit synthetic support",
        case_id="synthetic_axis_input_profiles")
    return AxisColumnNativeSimulator(config,
        proposal_config=AxisColumnProposalConfig(((0, 0), (1, 0)), 4), max_steps=2)


def manager(path, sim, profile):
    worlds = generate_partitions(sim.case_hash, sim.config.world_generator, 20261004,
        optimization=2, selection=1, final_evaluation=1, stress=1)
    return AxisTrainingAccounting(sim.clone, worlds.optimization, worlds.selection,
        receipt_path=path / "accounting.json", expected_model_hash=sim.decision_model_hash,
        input_profile=profile), worlds


def settings():
    return learning.TrainingConfig(seed=11, hidden_features=16, max_gradient_steps=1,
        max_environment_steps=8, max_wall_seconds=20, max_episode_steps=3,
        episodes_per_update=2, checkpoint_interval=1)


def seeded(profile):
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(11)
        return learning.MaskedPatientPolicy(15, 6, 16, input_profile=profile)


def test_schema_uses_already_served_raw_observation_without_extra_inventory(monkeypatch):
    sim = fixture()
    observation = sim.observation()
    before_hash, before_array = sim.decision_model_hash, observation.action_features.tobytes()
    def forbidden():
        pytest.fail("Schema requested another observation or inventory")
    monkeypatch.setattr(sim, "observation", forbidden)
    monkeypatch.setattr(sim, "proposed_actions", forbidden)
    record = axis_observation_contract(sim, observation)
    assert record["observation_encoding"] == "RAW"
    assert record["decision_model_hash"] == before_hash
    assert record["partial_contact_weight"] == .05
    assert "not physical depth" in record["depth_semantics"]
    assert "not residual-only" in record["adjacent_target_semantics"]
    assert observation.action_features.tobytes() == before_array
    bad = SimpleNamespace(**{name: getattr(observation, name) for name in (
        "action_features", "state_features", "action_ids", "action_mask")})
    bad.action_features = bad.action_features[:, :-1]
    with pytest.raises(ValueError, match="15\\+6"):
        axis_observation_contract(sim, bad)


def test_paired_trainable_weights_variable_rows_masking_and_current_ids():
    sim = fixture()
    raw, units = seeded("RAW"), seeded("FEATURE_UNITS")
    assert learning.trainable_parameter_hash(raw) == learning.trainable_parameter_hash(units)
    assert learning.policy_hash(raw) != learning.policy_hash(units)
    initial = sim.observation()
    order = np.r_[0, np.arange(len(initial.action_ids) - 1, 0, -1)]
    permuted = SimpleNamespace(action_features=initial.action_features[order],
        action_ids=tuple(initial.action_ids[i] for i in order), action_mask=initial.action_mask[order],
        state_features=initial.state_features)
    padded = SimpleNamespace(action_features=np.vstack((initial.action_features, np.ones((2, 15)) * 1e6)),
        action_ids=initial.action_ids + ("PAD0", "PAD1"), action_mask=np.r_[initial.action_mask, False, False],
        state_features=initial.state_features)
    with torch.no_grad():
        for policy in (raw, units):
            logits, value = policy(initial)
            torch.testing.assert_close(policy(permuted)[0], logits[order])
            padded_logits = policy(padded)[0]
            torch.testing.assert_close(padded_logits[:-2], logits)
            assert torch.softmax(padded_logits, 0)[-2:].sum() == 0
            assert torch.equal(value, raw(initial)[1])  # Identical unchanged critic.
    clone = sim.clone()
    action = initial.action_ids[1]
    next_observation = clone.step(action).observation
    assert set(initial.action_ids[1:]).isdisjoint(next_observation.action_ids[1:])
    with pytest.raises(InvalidActionError, match="stale"):
        clone.step(action)
    terminal = clone.step("STOP").observation
    assert terminal.action_ids == ("STOP",)
    for observation in (initial, next_observation, terminal):
        assert units(observation)[0].shape == observation.action_mask.shape


def test_profile_changes_no_physical_transition_or_source_observation(tmp_path):
    sim = fixture()
    outcomes = []
    for profile in ("RAW", "FEATURE_UNITS"):
        accounting, worlds = manager(tmp_path / profile, sim, profile)
        wrapped = accounting.recorded_factory()
        obs = wrapped.reset(worlds.optimization.seeds[0])
        before = obs.action_features.copy()
        seeded(profile)(obs)
        np.testing.assert_array_equal(obs.action_features, before)
        cut = wrapped.step(obs.action_ids[1])
        stopped = wrapped.step("STOP")
        receipt = accounting.snapshot()
        events = [event for event in receipt["events"] if event.get("executed_transition")]
        # Separate real executions have measured runtime differences. Keep all
        # physical history, action identities, counters and accounting fields.
        for event in events:
            for field in ("preview_seconds", "integrity_seconds"):
                del event["proposal_accounting_after"][field]
        outcomes.append((cut.reward, cut.info, stopped.reward, cut.observation.action_ids,
                         cut.observation.action_features.tobytes(), receipt["totals"], events))
        assert receipt["decision_model_hash"] == sim.decision_model_hash
        assert receipt["input_profile"] == profile and receipt["observation_encoding"] == "RAW"
        assert receipt["input_profile_hash"] == policy_input_profile(profile).fingerprint
    assert outcomes[0] == outcomes[1]


@pytest.mark.parametrize("declared,requested", [("RAW", "FEATURE_UNITS"), ("FEATURE_UNITS", "RAW")])
def test_wrapper_profile_mismatch_fails_before_optimizer(tmp_path, monkeypatch, declared, requested):
    sim = fixture()
    accounting, worlds = manager(tmp_path, sim, declared)
    monkeypatch.setattr(torch.optim, "Adam", lambda *a, **k: pytest.fail("Mismatch constructed Adam"))
    with pytest.raises(ValueError, match="profile differs"):
        learning.train_patient_policy(accounting.recorded_factory, worlds.optimization, worlds.selection,
            config=settings(), output_dir=tmp_path / "learner", input_profile=requested)
    assert not (tmp_path / "learner/initial.pt").exists()


@pytest.mark.parametrize("mutation", ["identity", "resealed_divisors"])
def test_mutated_accounting_profile_refused_before_optimizer(tmp_path, monkeypatch, mutation):
    sim = fixture()
    accounting, worlds = manager(tmp_path, sim, "FEATURE_UNITS")
    if mutation == "identity":
        accounting._input_profile = policy_input_profile("RAW")
    else:
        accounting._input_profile = replace(accounting.input_profile, action_divisors=(1.,) * 15)
        accounting._input_profile_hash = accounting._input_profile.fingerprint
    monkeypatch.setattr(torch.optim, "Adam", lambda *a, **k: pytest.fail("Mutation constructed Adam"))
    with pytest.raises(ValueError, match="profile changed"):
        train_axis_policy(accounting, config=settings(), output_dir=tmp_path / "learner")


def test_closed_exact_types_and_transfer_gates(tmp_path, monkeypatch):
    sim = fixture()
    observation = sim.observation()
    class AxisSubclass(AxisColumnNativeSimulator):
        pass
    alias = sim.clone()
    alias.__class__ = AxisSubclass
    with pytest.raises(ValueError, match="exact axis"):
        axis_observation_contract(alias, observation)
    with pytest.raises(ValueError, match="actual native"):
        learning._validate_simulator_profile(alias, "FEATURE_UNITS", observation)
    accounting, worlds = manager(tmp_path, sim, "FEATURE_UNITS")
    for candidate in (sim, accounting.recorded_factory()):
        with pytest.raises(ValueError, match="exact native"):
            native_observation_schema(candidate)
    monkeypatch.setattr(torch.optim, "Adam", lambda *a, **k: pytest.fail("Transfer constructed Adam"))
    for transfer in (dict(shared_checkpoint=tmp_path / "unused.pt"),
                     dict(procedural_checkpoint=tmp_path / "unused.pt")):
        with pytest.raises(ValueError, match="fresh scratch"):
            learning.train_patient_policy(sim.clone, worlds.optimization, worlds.selection,
                config=settings(), output_dir=tmp_path / next(iter(transfer)),
                input_profile="FEATURE_UNITS", **transfer)


def test_two_tiny_actual_updates_bind_same_forward_scaled_inputs_and_fresh_adam(tmp_path, monkeypatch):
    sim = fixture()
    actual_forward = learning.MaskedPatientPolicy.forward
    forwards, initial_optimizers = [], []
    def counted(policy, observation, **kwargs):
        forwards.append(policy.input_profile.profile_id)
        return actual_forward(policy, observation, **kwargs)
    actual_adam = torch.optim.Adam
    def fresh_adam(*args, **kwargs):
        optimizer = actual_adam(*args, **kwargs)
        initial_optimizers.append(len(optimizer.state))
        return optimizer
    monkeypatch.setattr(learning.MaskedPatientPolicy, "forward", counted)
    monkeypatch.setattr(torch.optim, "Adam", fresh_adam)
    initial_hashes = []
    for profile in ("RAW", "FEATURE_UNITS"):
        accounting, _ = manager(tmp_path / profile, sim, profile)
        directory = tmp_path / profile / "learner"
        result = train_axis_policy(accounting, config=settings(), output_dir=directory, record_decisions=True)
        assert result.gradient_steps == 1
        saved = json.loads((directory / "result.json").read_text())
        assert saved["actor_parameters_changed"] and saved["optimization_history"][0]["actor_gradient_norm_after_clip"] > 0
        assert len(saved["selection_history"]) == 2
        contract = json.loads((directory / "contract.json").read_text())
        assert contract["axis_observation_contract"]["decision_model_hash"] == sim.decision_model_hash
        assert contract["input_profile_hash"] == policy_input_profile(profile).fingerprint
        initial = learning.load_policy(directory / "initial.pt", expected_input_profile=profile)
        initial_hashes.append(learning.trainable_parameter_hash(initial))
        decisions = [event for event in accounting.snapshot()["events"] if event["kind"] == "decision"]
        assert sum(event["payload"]["forward_evaluated"] for event in decisions) == forwards.count(profile)
        assert decisions
        for event in decisions:
            payload, inputs = event["payload"], event["payload"]["inputs"]
            assert event["policy_input_profile_hash"] == policy_input_profile(profile).fingerprint
            assert event["observation_encoding"] == "RAW"
            if not payload["forward_evaluated"]:
                assert inputs["actor_action_features"] is None
                continue
            raw = np.asarray(inputs["source_action_features"]["values"], dtype=np.float32)
            actual = np.asarray(inputs["actor_action_features"]["values"], dtype=np.float32)
            expected = raw if profile == "RAW" else raw / np.asarray(FEATURE_UNITS_DIVISORS, dtype=np.float32)
            np.testing.assert_array_equal(actual, expected)
            assert inputs["state_features"] == inputs["source_state_features"]
            assert payload["selected_action_id"] == inputs["action_ids"][payload["selected_index"]]
        with pytest.raises(ValueError, match="profile differs"):
            learning.load_policy(directory / "checkpoint.pt", expected_input_profile="FEATURE_UNITS" if profile == "RAW" else "RAW")
    assert initial_hashes[0] == initial_hashes[1]
    assert initial_optimizers == [0, 0]
