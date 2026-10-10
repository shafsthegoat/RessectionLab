"""Small generated controls; no released checkpoint, patient or training call."""
import numpy as np
import pytest

from resectionlab.development_episode import make_development_task, plan_development_episode
from resectionlab.legacy_aspiration_projection import (
    AspirationOnlyNativeTask, BoundAspirationTransferPolicy, PROJECTION_HASH,
    _projection_trace, project_aspiration_observation)
from resectionlab.sequential_spatial_observation import SequentialSpatialObservation
from resectionlab.shared_episode import (
    execute_policy_episode, execute_search_episode, identity_for_untrained_spatial_policy,
    verify_strategy_replay)
from resectionlab.spatial_observations import SpatialObservation


def test_projection_keeps_exact_native_aspiration_rows_and_all_observed_images():
    task = make_development_task()
    mixed = task.observation()
    projected, evidence = project_aspiration_observation(mixed)
    assert type(projected) is SpatialObservation
    assert evidence.full_observation_hash == mixed.fingerprint
    assert evidence.projected_observation_hash == projected.fingerprint
    assert evidence.projection_hash == PROJECTION_HASH
    kept = [i for i, mode in enumerate(mixed.action_modes) if mode != "probe"]
    assert projected.action_ids == tuple(mixed.action_ids[i] for i in kept)
    assert projected.action_tool_ids == tuple(mixed.action_tool_ids[i] for i in kept)
    assert np.array_equal(projected.action_geometry, mixed.base.action_geometry[kept])
    assert np.array_equal(projected.action_mask, mixed.base.action_mask[kept])
    for field in ("image_channels", "coverage", "channel_available", "affine_ras_mm", "spacing_mm", "state_features"):
        assert np.array_equal(getattr(projected, field), getattr(mixed.base, field))
    assert evidence.omitted_probe_action_ids == tuple(mixed.action_ids[i] for i, mode in enumerate(mixed.action_modes)
                                                       if mode == "probe")
    assert not set(projected.action_ids) & set(evidence.omitted_probe_action_ids)


def test_wrapper_rejects_probe_without_mutating_native_state():
    task = make_development_task()
    view = AspirationOnlyNativeTask(task)
    first = next(action for action, mode in zip(task.observation().action_ids, task.observation().action_modes)
                 if mode == "aspirate")
    view.step(first)
    mixed = task.observation()
    probe = next(action for action, mode in zip(mixed.action_ids, mixed.action_modes) if mode == "probe")
    before = task._engine.state_hash
    with pytest.raises(ValueError, match="aspiration-only"):
        view.step(probe)
    assert task._engine.state_hash == before
    assert task.metrics()["steps"] == 1
    assert all(row["interaction_mode"] == "aspirate" for row in task.metrics()["history"])


def test_projected_actor_and_search_use_same_native_environment_and_replay():
    from resectionlab.spatial_policy import SpatialPolicy, SpatialPolicyConfig

    task = AspirationOnlyNativeTask(make_development_task())
    actor = SpatialPolicy(SpatialPolicyConfig(encoder_channels=(2, 2), hidden_features=4,
                                             ray_samples=2)).eval()
    base_identity = identity_for_untrained_spatial_policy(actor, policy_id="generated-projection-software-control")
    bound = BoundAspirationTransferPolicy(actor, base_identity, task)
    assert bound.identity.architecture_hash != base_identity.architecture_hash
    assert bound.projection_receipt["projection_hash"] == PROJECTION_HASH
    policy = execute_policy_episode(task, bound, bound.identity)
    search = execute_search_episode(task, max_calls=6, beam_width=2, seconds=10.)
    assert policy["method"] == "POLICY_SOFTWARE_CONTROL"
    assert search["method"] == "SEARCH"
    assert policy["source_hash"] == search["source_hash"]
    assert policy["environment_contract_hash"] == search["environment_contract_hash"]
    assert policy["decisions"][0]["observation_before"] == search["decisions"][0]["observation_before"]
    for record in (policy, search):
        assert verify_strategy_replay(task, record)
        assert all(row["action_mode"] in ("aspirate", "stop") for row in record["decisions"])
        assert all(row["physical_transition"]["interaction_mode"] in ("aspirate", "stop")
                   for row in record["decisions"])
    assert task.metrics()["steps"] == 0


def test_plain_legacy_actor_still_rejects_mixed_observation():
    from resectionlab.spatial_policy import SpatialPolicy, SpatialPolicyConfig
    actor = SpatialPolicy(SpatialPolicyConfig(encoder_channels=(2, 2), hidden_features=4, ray_samples=2)).eval()
    with pytest.raises(TypeError, match="SpatialObservation"):
        actor(make_development_task().observation())


def test_projection_rejects_raw_v1_observation():
    with pytest.raises(TypeError, match="mixed observation"):
        project_aspiration_observation(AspirationOnlyNativeTask(make_development_task()).observation())


def test_stop_only_projection_is_valid_when_no_aspiration_rows_exist():
    mixed = make_development_task().observation()
    synthetic_modes = SequentialSpatialObservation(mixed.base,
        ("stop",) + ("probe",) * (len(mixed.action_ids) - 1), mixed.observed_probe_contact_grid)
    projected, evidence = project_aspiration_observation(synthetic_modes)
    assert projected.action_ids == ("STOP",)
    assert projected.action_mask.tolist() == [True]
    assert len(evidence.omitted_probe_action_ids) == len(mixed.action_ids) - 1


def test_private_reference_change_does_not_change_projected_actor_inputs():
    ordinary = AspirationOnlyNativeTask(make_development_task())
    alternative = AspirationOnlyNativeTask(make_development_task(reference_target=np.zeros((13, 13, 12), bool)))
    assert ordinary.observation().fingerprint == alternative.observation().fingerprint
    assert ordinary.observation().action_ids == alternative.observation().action_ids
    first = ordinary.observation().action_ids[1]
    ordinary.step(first)
    alternative.step(first)
    assert ordinary.observation().fingerprint == alternative.observation().fingerprint
    assert ordinary._engine.state_hash == alternative._engine.state_hash


def test_prior_probe_history_and_observed_probe_contact_cannot_be_hidden():
    task = make_development_task()
    plan, _, _ = plan_development_episode(task, "scripted")
    task.step(plan["actions"][0])
    task.step(plan["actions"][1])
    assert task._engine.probe_contact_mask.any()
    with pytest.raises(ValueError, match="probe-exposed native history"):
        AspirationOnlyNativeTask(task)
    with pytest.raises(ValueError, match="probe-exposed state"):
        project_aspiration_observation(task.observation())


def test_clone_fresh_and_planning_clone_keep_the_probe_restriction():
    task = AspirationOnlyNativeTask(make_development_task())
    for branch in (task.clone(), task.fresh(), task.planning_clone()):
        mixed = branch._native.observation()
        assert type(mixed) is SequentialSpatialObservation
        assert branch.observation().action_ids == tuple(action for action, mode in
            zip(mixed.action_ids, mixed.action_modes) if mode != "probe")
        assert branch.projection_hash == PROJECTION_HASH
        assert all(tool_id != "development-probe" for tool_id in branch.observation().action_tool_ids[1:])


def test_external_native_probe_after_view_creation_is_refused_before_next_observation():
    native = make_development_task()
    view = AspirationOnlyNativeTask(native)
    plan, _, _ = plan_development_episode(native, "scripted")
    native.step(plan["actions"][0])
    native.step(plan["actions"][1])
    with pytest.raises(ValueError, match="probe-exposed native history"):
        view.observation()
    with pytest.raises(ValueError, match="probe-exposed native history"):
        view.fresh()


def test_restricted_search_records_full_inventory_hash_and_omitted_probe_ids():
    task = AspirationOnlyNativeTask(make_development_task())
    record = execute_search_episode(task, max_calls=6, beam_width=2, seconds=10.)
    trace = _projection_trace(task.planning_clone(), record)
    assert len(trace) == len(record["decisions"])
    for row, decision in zip(trace, record["decisions"]):
        assert row["chosen_action_id"] == decision["action_id"]
        assert row["projected_observation_hash"] == decision["observation_before"]
        assert row["projection_hash"] == PROJECTION_HASH
        assert row["chosen_action_id"] in row["retained_action_ids"]


@pytest.mark.parametrize("field,changed", [
    ("expected_source_hash", "sha256:" + "0" * 64),
    ("expected_decision_model_hash", "sha256:" + "1" * 64),
    ("aspirator_tool_ids", frozenset({"development-probe"})),
])
def test_bound_policy_rejects_live_binding_drift_before_forward(field, changed):
    from resectionlab.spatial_policy import SpatialPolicy, SpatialPolicyConfig
    task = AspirationOnlyNativeTask(make_development_task())
    actor = SpatialPolicy(SpatialPolicyConfig(encoder_channels=(2, 2), hidden_features=4, ray_samples=2)).eval()
    original = identity_for_untrained_spatial_policy(actor, policy_id="binding-drift-control")
    bound = BoundAspirationTransferPolicy(actor, original, task)
    setattr(bound, field, changed)
    with pytest.raises(RuntimeError, match="changed after binding"):
        bound.act(task.observation())
