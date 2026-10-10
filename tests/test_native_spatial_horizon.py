"""Declared horizon controls over generated geometry; no patient/model work."""
import inspect

import numpy as np
import pytest

from resectionlab.core import semantic_digest
from resectionlab.geometry import AccessWindow, ToolGeometry
from resectionlab.native_spatial_task import (
    MAX_NATIVE_SPATIAL_STEPS, NativeSpatialCase, NativeSpatialTask, make_native_opening_task)
from resectionlab.patient_planning_admission import make_patient_planning_task
from resectionlab.simulation import InvalidActionError
from test_patient_planning_admission import fixture_contract


def independent_cells_case():
    # 24 separated exposed cells permit 24 genuine, one-cell native removals.
    # This is a horizon mechanism control, not a corridor reachability claim.
    support = np.zeros((20, 14, 3), bool)
    for x in (2, 5, 8, 11, 14, 17):
        for y in (2, 5, 8, 11):
            support[x, y, 1] = True
    return NativeSpatialCase(support.astype(np.float32), support, support, np.eye(4),
        AccessWindow((9.5, 6.5, .5), (0., 0., 1.), 20., "generated-horizon-control"),
        (ToolGeometry("generated-cell-remover", .9, .2, 5., 30., 2.),),
        track="synthetic_scan", support_source_kind="derived_from_scan",
        support_derivation="generated isolated occupied cells",
        nominal_target=support, target_source_kind="derived_from_scan",
        target_derivation="generated occupied-cell objective", crop_shape=(20, 14, 3))


def remove_first_remaining_cell(task):
    row = next(row for row in task.candidate_inventory()["ledger"]
               if row["feasible"] and task._engine.remaining_mask[tuple(row["voxel"])])
    result = task.step(row["action_id"])
    assert result.info["removed_indices_native"] == [row["voxel"]]
    return result


@pytest.mark.parametrize("horizon", [1, 6, 24])
def test_declared_horizon_bound_and_context_propagate(horizon):
    case, args = fixture_contract()
    args["protocol"]["max_steps"] = horizon
    task, context = make_patient_planning_task(case, **args)
    context.require_training()
    assert task.max_steps == context.max_steps == horizon
    assert context.record()["max_optimizer_updates"] == 2
    assert context.record()["budgets"] == {key: args["protocol"][key] for key in
        ("search", "max_native_previews", "max_policy_forwards", "worker_seconds", "memory_bytes", "threads")}
    assert context.record()["protocol_sha256"] == semantic_digest(args["protocol"])
    for copy in (task.fresh(), task.clone(), task.planning_clone(), task.planning_clone().fresh()):
        assert copy.max_steps == horizon and copy.decision_model_hash == task.decision_model_hash
        assert copy.observation().state_features[1] == horizon
        context.require_task(copy)
        context.require_observations([copy.observation()])


@pytest.mark.parametrize("horizon", [0, 25, True, False, 24.])
def test_invalid_or_nontyped_horizon_refuses_in_native_and_admission(horizon):
    case, args = fixture_contract()
    with pytest.raises(ValueError, match="horizon"):
        NativeSpatialTask(case, max_steps=horizon)
    args["protocol"]["max_steps"] = horizon
    with pytest.raises(ValueError):
        make_patient_planning_task(case, **args)


def test_horizon_changes_existing_identity_and_cross_context_is_refused():
    case, args = fixture_contract()
    args["protocol"]["max_steps"] = 6
    six, context_six = make_patient_planning_task(case, **args)
    args["protocol"]["max_steps"] = 24
    longer, context_longer = make_patient_planning_task(case, **args)
    assert six.case.source_hash == longer.case.source_hash
    assert six.decision_model_hash != longer.decision_model_hash
    assert six.observation().fingerprint != longer.observation().fingerprint
    assert context_six.fingerprint != context_longer.fingerprint
    with pytest.raises(ValueError):
        context_six.require_task(longer)
    with pytest.raises(ValueError):
        context_six.require_observations([longer.observation()])


def test_numpy_integer_keeps_existing_boundary_specific_behavior():
    case, args = fixture_contract()
    with pytest.raises(ValueError, match="horizon"):
        NativeSpatialTask(case, max_steps=np.int64(24))
    # Admission already normalizes JSON-compatible NumPy scalars into Python
    # scalars before checking the protocol. The horizon extension preserves it.
    args["protocol"]["max_steps"] = np.int64(24)
    task, context = make_patient_planning_task(case, **args)
    assert type(task.max_steps) is int and task.max_steps == context.max_steps == 24


def test_twenty_four_actual_commits_terminate_at_declared_boundary():
    task = NativeSpatialTask(independent_cells_case(), max_steps=24)
    for count in range(1, 25):
        result = remove_first_remaining_cell(task)
        assert result.terminated == (count == 24)
        assert task.metrics()["steps"] == count
        assert task.metrics()["simulated_removed_volume_mm3"] == count
        assert result.observation.state_features[0] == count
        assert result.observation.state_features[1] == 24
        if count == 12:
            for branch in (task.clone(), task.planning_clone()):
                assert branch.metrics()["steps"] == 12 and branch.max_steps == 24
                clean = lambda rows: [{key: value for key, value in row.items() if key != "outcome_scope"}
                                      for row in rows]
                assert clean(branch.metrics()["history"]) == clean(task.metrics()["history"])
            assert task.fresh().metrics()["steps"] == 0
    assert task.candidate_inventory()["remaining_steps"] == 0
    assert task.observation().action_ids == ("STOP",)
    assert task.independent_geometry_check().feasible
    with pytest.raises(InvalidActionError, match="terminated"):
        task.step("STOP")


def test_stop_still_terminates_early_at_longer_horizon():
    task = NativeSpatialTask(independent_cells_case(), max_steps=24)
    result = task.step("STOP")
    assert result.terminated and result.reward == 0
    assert task.metrics()["steps"] == 1 and task.metrics()["simulated_removed_volume_mm3"] == 0


def test_six_real_commits_keep_exact_pre_extension_outputs():
    task = NativeSpatialTask(independent_cells_case(), max_steps=6)
    assert task.observation().fingerprint == "sha256:b7c3e78d771b57825aa5d1f85c3566c7a53645a3f6769c39b8f7049b9533a42c"
    for count in range(1, 7):
        assert remove_first_remaining_cell(task).terminated == (count == 6)
    assert semantic_digest(task.metrics()) == "sha256:9836199f69e82ce17a5c1b827657d66f7648acdaa2c36f03de1c3b95abdd644f"
    assert task.observation().fingerprint == "sha256:e69197c30b7e6b9dcdc3cd7680e57c5f87d8f5d6482a19413692fb0c99a7379e"
    assert semantic_digest(task.metrics()["history"]) == "sha256:03616401b64bf77bcb0c831937692177e61fc9712d532bf163c0a2d5a8cc88ea"
    assert semantic_digest(task.candidate_inventory()) == "sha256:f22113b86705fd1f7926a071326815a71b015a03882ae572698f14f2e8ac407d"


@pytest.mark.parametrize("horizon,model,observation,inventory,stop_metrics,stop_observation", [
    (3, "874aa851f2325566f2b9cb7eaaf332cd85c05fe25f71b488ef9d669edcf5f4c8",
     "d84c0d00ed23f74f0bfe43f5b94f9c418e7c2fae631cc0c997c0a04c40e76853",
     "f77d5596378b86537a7bb1ad392b796085c735fd1f1b46ac4814e30641638c69",
     "05c8c92f77a4dba5fad8848ba035b9be4f247e5d20b22bfe28d0dc5ce067714d",
     "2081f8a91fc13024c1021f2acc53b8ae1c7b5acd241d352f647033f96bb6072d"),
    (6, "5b29562d84aa86557895cd20813496223d5d1953de75e1f96c2cc4b20f027c57",
     "09d7cd6284c093ea84ba17fd598b7fe1e3da8242520dd5a77630549491ef4ea2",
     "14c6ea89249c6d357f56327f1d5d3263726a9ff5113f8d3ebf34dd0b1c1f7d67",
     "da7b5199b219e5f420bc7ec984a063810c4f7ada168c436b910d4defec383a73",
     "1f36c6e456cf335b79925a24eb4e9ec1473672e28b941a0c246abc8e904e1d17"),
])
def test_unchanged_default_and_six_step_contracts(horizon, model, observation, inventory,
                                                stop_metrics, stop_observation):
    # Exact baseline fingerprints captured from canonical code before this patch.
    assert MAX_NATIVE_SPATIAL_STEPS == 24
    assert inspect.signature(NativeSpatialTask).parameters["max_steps"].default == 3
    case = make_native_opening_task().case
    task = NativeSpatialTask(case) if horizon == 3 else NativeSpatialTask(case, max_steps=horizon)
    assert task.case.source_hash == "sha256:9b39c31e1e012b0822d538451a0452fa9c564acc863d22dabb311ec4cf8c5d4e"
    assert task.decision_model_hash == "sha256:" + model
    assert task.observation().fingerprint == "sha256:" + observation
    assert semantic_digest(task.candidate_inventory()) == "sha256:" + inventory
    task.step("STOP")
    assert semantic_digest(task.metrics()) == "sha256:" + stop_metrics
    assert task.observation().fingerprint == "sha256:" + stop_observation


def test_twenty_four_does_not_relax_select_gradient_or_runtime_limits():
    case, args = fixture_contract("ReMIND-013", "SELECT")
    args["protocol"]["max_steps"] = 24
    _, context = make_patient_planning_task(case, **args)
    with pytest.raises(ValueError, match="TRAIN_gradient"):
        context.require_training()
    args["protocol"]["max_optimizer_updates"] = 1
    with pytest.raises(ValueError, match="role_and_budget"):
        make_patient_planning_task(case, **args)
    args["protocol"]["max_optimizer_updates"] = 0
    args["protocol"]["max_native_previews"] = 0
    with pytest.raises(ValueError, match="runtime_budgets"):
        make_patient_planning_task(case, **args)
