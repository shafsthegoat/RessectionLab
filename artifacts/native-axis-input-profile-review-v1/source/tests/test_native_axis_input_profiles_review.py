"""Independent tiny axis/profile checks; no patient data or population runs."""
import copy
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
from resectionlab.native_resection import NATIVE_GENERIC_TOOLS, NativeResectionConfig
from resectionlab.policy_inputs import policy_input_profile
from resectionlab.procedural_learning import native_observation_schema
from resectionlab.simulation import InvalidActionError
from resectionlab.worlds import generate_partitions


DIVISORS = (1., 648., 648., 648., 648., 120., 2.25, 1.10, 1., 1., 1., 1., 648., 648., 648.)


@pytest.fixture(autouse=True)
def one_cpu_thread():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        yield
    finally:
        torch.set_num_threads(previous)


def fixture():
    tissue = np.ones((7, 7, 8), bool)
    labels = np.zeros(tissue.shape, np.int16)
    labels[1:6, 1:6, 2:7] = 1
    affine = np.diag([1., 1., 1.5, 1.])
    config = NativeResectionConfig(tissue, labels, affine,
        AccessWindow((3., 3., -.75), (0., 0., 1.), 4.), NATIVE_GENERIC_TOOLS,
        "independent-synthetic-axis-profile-review-v1", "explicit constructed support")
    return AxisColumnNativeSimulator(config,
        proposal_config=AxisColumnProposalConfig(((0, 0), (-1, 0), (1, 0)), 6),
        nominal_motor=np.full(tissue.shape, .25), nominal_language=np.full(tissue.shape, .5),
        max_steps=3)


def manager(path, *, profile="FEATURE_UNITS"):
    source = fixture()
    worlds = generate_partitions(source.case_hash, source.config.world_generator, 187,
        optimization=2, selection=1, final_evaluation=1, stress=1)
    owner = AxisTrainingAccounting(source.fresh, worlds.optimization, worlds.selection,
        receipt_path=path / "accounting.json", expected_model_hash=source.decision_model_hash,
        input_profile=profile)
    return source, owner, worlds


def config():
    return learning.TrainingConfig(seed=31, hidden_features=8, max_gradient_steps=1,
        max_environment_steps=8, max_wall_seconds=20., max_episode_steps=4,
        episodes_per_update=1, checkpoint_interval=1)


def initialized(profile):
    with torch.random.fork_rng():
        torch.manual_seed(31)
        return learning.MaskedPatientPolicy(15, 6, 8, input_profile=profile)


def test_axis_contract_keeps_raw_physics_separate_from_existing_profile():
    sim = fixture()
    observation = sim.observation()
    before = sim.decision_model_hash
    raw = learning._validate_simulator_profile(sim, "RAW", observation)
    units = learning._validate_simulator_profile(sim, "FEATURE_UNITS", observation)
    assert raw == units == axis_observation_contract(sim, observation)
    assert units["decision_model_hash"] == before == sim.decision_model_hash
    assert units["observation_encoding"] == "RAW"
    assert "not physical depth" in units["depth_semantics"]
    profile = policy_input_profile("FEATURE_UNITS")
    assert profile.action_divisors == DIVISORS and profile.state_divisors == (1.,) * 6
    assert profile.to_dict()["return_divisor"] == profile.to_dict()["critic_output_divisor"] == 1.
    assert not any(profile.to_dict()[name] for name in ("clipping", "centering", "running_statistics"))


def test_physical_partial_contact_and_budget_columns_match_independent_cell_sets():
    sim = fixture()
    charged, removed = set(), set()
    labels = sim.config.target_labels
    voxel_volume = abs(np.linalg.det(sim.native_config.affine[:3, :3]))
    observed_partial = False
    for completed in (0, 1):
        observation, actions = sim.observation(), sim.proposed_actions()
        np.testing.assert_array_equal(observation.action_ids, tuple(action.action_id for action in actions))
        assert observation.state_features[1] == np.float32(completed / 3)
        for index, action in enumerate(actions[1:], 1):
            cells = {tuple(cell) for cell in action.removal_indices} - removed
            partial = {tuple(cell) for cell in action.swept_indices} - cells - removed - charged
            tool = next(tool for tool in sim.native_config.tools if tool.tool_id == action.tool_id)
            lo = np.maximum(np.array(action.target_index) - 1, 0)
            hi = np.minimum(np.array(action.target_index) + 2, labels.shape)
            neighborhood = labels[tuple(slice(a, b) for a, b in zip(lo, hi))]
            expected = [sum(labels[cell] > 0 for cell in cells) * voxel_volume,
                sum(labels[cell] == 0 for cell in cells) * voxel_volume,
                len(cells) * .25 * voxel_volume, len(cells) * .5 * voxel_volume,
                action.insertion_distance_mm, tool.tip_radius_mm, tool.shaft_radius_mm]
            np.testing.assert_array_equal(observation.action_features[index, 1:8], np.asarray(expected, np.float32))
            assert observation.action_features[index, 9] == np.float32(completed / 3)
            assert observation.action_features[index, 11] == np.float32(np.count_nonzero(neighborhood) / neighborhood.size)
            expected_partial = np.asarray([sum(labels[cell] == 0 for cell in partial) * voxel_volume,
                len(partial) * .25 * voxel_volume, len(partial) * .5 * voxel_volume], np.float32)
            np.testing.assert_array_equal(observation.action_features[index, 12:15], expected_partial)
            observed_partial |= bool(partial)
        if completed == 0:
            chosen = actions[1]
            cells = {tuple(cell) for cell in chosen.removal_indices}
            charged |= {tuple(cell) for cell in chosen.swept_indices} - cells
            removed |= cells
            sim.step(chosen.action_id)
    assert observed_partial and charged and removed


def test_paired_trainable_weights_and_critic_equal_while_captured_actor_inputs_scale():
    observation = fixture().observation()
    raw, units = initialized("RAW"), initialized("FEATURE_UNITS")
    assert learning.trainable_parameter_hash(raw) == learning.trainable_parameter_hash(units)
    assert learning.policy_hash(raw) != learning.policy_hash(units)
    for name, parameter in raw.named_parameters():
        assert torch.equal(parameter, dict(units.named_parameters())[name])
    original_actions, original_state = observation.action_features.tobytes(), observation.state_features.tobytes()
    capture = []
    with torch.no_grad():
        _, before_value = raw(observation)
        _, units_value = units(observation, capture_inputs=capture.append)
    assert torch.equal(before_value, units_value)
    actual = capture[0]
    np.testing.assert_array_equal(actual.source_action_features, observation.action_features)
    np.testing.assert_array_equal(actual.action_features, observation.action_features)
    np.testing.assert_array_equal(actual.actor_action_features,
        observation.action_features / np.asarray(DIVISORS, np.float32))
    np.testing.assert_array_equal(actual.source_state_features, actual.state_features)
    assert observation.action_features.tobytes() == original_actions and observation.state_features.tobytes() == original_state


def test_feature_units_keeps_variable_inventory_masks_permutations_and_current_ids():
    sim = fixture()
    observation = sim.observation()
    units = initialized("FEATURE_UNITS")
    count = len(observation.action_ids)
    order = np.r_[0, np.arange(count - 1, 0, -1)]
    permuted = SimpleNamespace(action_features=observation.action_features[order],
        action_mask=observation.action_mask[order], state_features=observation.state_features,
        action_ids=tuple(observation.action_ids[index] for index in order))
    padded = SimpleNamespace(action_features=np.vstack((observation.action_features, np.full((2, 15), 1e7, np.float32))),
        state_features=observation.state_features, action_mask=np.r_[observation.action_mask, False, False],
        action_ids=observation.action_ids + ("MASKED:one", "MASKED:two"))
    with torch.no_grad():
        expected, value = units(observation)
        logits, other_value = units(permuted)
        torch.testing.assert_close(logits, expected[order], rtol=1e-6, atol=1e-7)
        assert torch.equal(value, other_value)
        extra, _ = units(padded)
        torch.testing.assert_close(extra[:count], expected, rtol=1e-6, atol=1e-7)
        assert torch.isneginf(extra[count:]).all()
    stale = observation.action_ids[1]
    sim.step(stale)
    current = sim.observation()
    assert not set(observation.action_ids[1:]).intersection(current.action_ids[1:])
    units(current)
    with pytest.raises(InvalidActionError):
        sim.step(stale)


@pytest.mark.parametrize("change", ["dtype", "nonfinite", "masked", "duplicate_id", "stop"])
def test_schema_rejects_malformed_served_observation(change):
    sim = fixture()
    original = sim.observation()
    values = {name: copy.deepcopy(getattr(original, name)) for name in
              ("action_ids", "action_features", "state_features", "action_mask")}
    if change == "dtype": values["action_features"] = values["action_features"].astype(np.float64)
    elif change == "nonfinite": values["state_features"][0] = np.nan
    elif change == "masked": values["action_mask"][1] = False
    elif change == "duplicate_id": values["action_ids"] = ("STOP",) * len(values["action_ids"])
    else: values["action_features"][0, 1] = 1.
    with pytest.raises(ValueError):
        learning._validate_simulator_profile(sim, "FEATURE_UNITS", SimpleNamespace(**values))


def test_axis_subclass_and_duck_wrapper_do_not_authorize_units():
    sim = fixture()
    observation = sim.observation()
    proxy = SimpleNamespace(_raw=sim, decision_model_hash=sim.decision_model_hash)
    with pytest.raises(ValueError, match="native observation"):
        learning._validate_simulator_profile(proxy, "FEATURE_UNITS", observation)
    class AxisSubclass(AxisColumnNativeSimulator):
        pass
    sim.__class__ = AxisSubclass
    with pytest.raises(ValueError, match="native observation"):
        learning._validate_simulator_profile(sim, "FEATURE_UNITS", observation)


def test_wrapper_profile_mismatch_refused_before_optimizer(tmp_path, monkeypatch):
    _, owner, worlds = manager(tmp_path)
    monkeypatch.setattr(learning.torch.optim, "Adam", lambda *args, **kwargs: pytest.fail("Profile mismatch reached Adam"))
    with pytest.raises(ValueError, match="profile"):
        learning.train_patient_policy(owner.recorded_factory, worlds.optimization, worlds.selection,
            config=config(), output_dir=tmp_path / "learner", input_profile="RAW")
    assert not (tmp_path / "learner/initial.pt").exists()


def test_coherently_resealed_accounting_profile_refused_before_optimizer(tmp_path, monkeypatch):
    _, owner, _ = manager(tmp_path)
    changed = list(owner.input_profile.action_divisors)
    changed[1] *= 2
    owner._input_profile = replace(owner.input_profile, action_divisors=tuple(changed))
    owner._input_profile_hash = owner._input_profile.fingerprint
    monkeypatch.setattr(learning.torch.optim, "Adam", lambda *args, **kwargs: pytest.fail("Forged registry profile reached Adam"))
    with pytest.raises(ValueError, match="(?i)(profile|registry)"):
        train_axis_policy(owner, config=config(), output_dir=tmp_path / "learner", record_decisions=True)
    assert not (tmp_path / "learner/initial.pt").exists()


def test_fixed_native_procedural_gate_still_rejects_axis_and_recorded_axis(tmp_path):
    sim, owner, worlds = manager(tmp_path)
    wrapped = owner.recorded_factory()
    wrapped.reset(worlds.optimization.seeds[0])
    for value in (sim, wrapped):
        with pytest.raises(ValueError, match="exact native observation backend"):
            native_observation_schema(value)


def test_ppo_cannot_accept_new_profile_argument(tmp_path):
    from resectionlab.learning_ppo import train_patient_ppo
    with pytest.raises(TypeError, match="input_profile"):
        train_patient_ppo(lambda: pytest.fail("PPO profile extension reached factory"), None, None,
            output_dir=tmp_path, input_profile="FEATURE_UNITS")


def test_one_tiny_update_records_actual_transform_and_unchanged_raw_model(tmp_path):
    source, owner, _ = manager(tmp_path)
    result = train_axis_policy(owner, config=config(), output_dir=tmp_path / "learner", record_decisions=True)
    assert result.gradient_steps == 1
    receipt = json.loads(owner.receipt_path.read_text())
    contract = json.loads((tmp_path / "learner/contract.json").read_text())
    assert receipt["input_profile"] == contract["input_profile"]["profile_id"] == "FEATURE_UNITS"
    assert receipt["input_profile_hash"] == contract["input_profile_hash"] == policy_input_profile("FEATURE_UNITS").fingerprint
    assert receipt["observation_encoding"] == contract["axis_observation_contract"]["observation_encoding"] == "RAW"
    assert contract["decision_model_hash"] == receipt["decision_model_hash"] == source.decision_model_hash
    decisions = [event for event in receipt["events"] if event["kind"] == "decision"]
    assert decisions and any(event["payload"]["forward_evaluated"] for event in decisions)
    for event in decisions:
        assert event["policy_input_profile_hash"] == receipt["input_profile_hash"]
        payload = event["payload"]
        assert payload["selected_action_id"] == payload["inputs"]["action_ids"][payload["selected_index"]]
        if payload["forward_evaluated"]:
            inputs = payload["inputs"]
            original = np.asarray(inputs["source_action_features"]["values"], np.float32)
            np.testing.assert_array_equal(inputs["action_features"]["values"], original)
            np.testing.assert_array_equal(inputs["actor_action_features"]["values"], original / np.asarray(DIVISORS, np.float32))
            np.testing.assert_array_equal(inputs["source_state_features"]["values"], inputs["state_features"]["values"])
