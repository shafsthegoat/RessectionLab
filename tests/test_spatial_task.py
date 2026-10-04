"""Development-only sequential and observability checks; no training or held-out cases."""
from dataclasses import replace

import numpy as np
import pytest

from resectionlab.simulation import InvalidActionError
from resectionlab.spatial_task import (
    DEFAULT_REWARD, SpatialTask, TARGET_THRESHOLD, make_opening_task, make_spatial_case, make_spatial_task,
)


OPENING_SEQUENCE = (
    "REMOVE:opening-tool:1,1,0",
    "REMOVE:opening-tool:0,1,0",
    "REMOVE:opening-tool:0,1,1",
)


def assert_observations_identical(first, second):
    assert first.action_ids == second.action_ids
    assert first.action_tool_ids == second.action_tool_ids
    assert first.source_id == second.source_id
    assert first.track == second.track
    for name in ("image_channels", "coverage", "channel_available", "affine_ras_mm",
                 "spacing_mm", "action_geometry", "action_mask", "state_features"):
        np.testing.assert_array_equal(getattr(first, name), getattr(second, name))
    assert first.fingerprint == second.fingerprint


def exhaustive(task):
    """Enumerate every same-observation primitive/STOP sequence within this horizon."""
    if task.terminated:
        return task.metrics()["total_reward"], (), 1
    best, best_actions, leaves = -np.inf, (), 0
    for action in task.observation().action_ids:
        child = task.clone()
        child.step(action)
        value, suffix, count = exhaustive(child)
        leaves += count
        if value > best:
            best, best_actions = value, (action, *suffix)
    return best, best_actions, leaves


def test_exact_three_step_optimum_requires_two_costly_preparatory_actions():
    task = make_opening_task()
    assert task.observation().action_ids == ("STOP", OPENING_SEQUENCE[0])
    immediate = {}
    for action in task.observation().action_ids:
        child = task.planning_clone()
        immediate[action] = child.step(action).reward
    assert max(immediate, key=immediate.get) == "STOP"
    value, actions, leaves = exhaustive(task.planning_clone())
    expected = 1. - 2 * .2 - 3 * .03 - .002 * (.5 + np.sqrt(1.25) + np.sqrt(3.25))
    assert value == pytest.approx(expected)
    assert actions == OPENING_SEQUENCE
    assert leaves > 10  # Includes every reachable branch and early STOP.
    rewards = [task.step(action).reward for action in actions]
    assert rewards[0] < 0 and rewards[1] < 0 and rewards[2] > 0
    assert task.metrics()["target_removed_mm3"] == 1.
    assert task.independent_geometry_check().feasible


def test_hidden_reference_and_world_do_not_change_complete_actor_interface():
    original = make_opening_task()
    target = np.zeros_like(original.case.reference_target)
    target[2, 1, 1] = True
    hazard = np.ones_like(original.case.reference_motor)
    changed = SpatialTask(replace(original.case, reference_target=target,
        reference_motor=hazard, reference_language=hazard), world_seed=82, max_steps=3)
    assert original.decision_model_hash == changed.decision_model_hash
    for action in OPENING_SEQUENCE:
        assert_observations_identical(original.observation(), changed.observation())
        assert original.candidate_inventory() == changed.candidate_inventory()
        original.step(action)
        changed.step(action)
    assert_observations_identical(original.observation(), changed.observation())
    assert original.metrics()["total_reward"] != changed.metrics()["total_reward"]


def test_planning_clone_removes_private_outcomes_even_after_different_rewards():
    first = make_opening_task()
    second = SpatialTask(replace(first.case, reference_target=np.zeros((3, 3, 3), bool),
        reference_motor=np.ones((3, 3, 3), np.float32)), max_steps=3)
    for environment in (first, second):
        environment.step(OPENING_SEQUENCE[0])
    a, b = first.planning_clone(), second.planning_clone()
    assert_observations_identical(a.observation(), b.observation())
    assert a.metrics() == b.metrics()
    np.testing.assert_array_equal(a.case.reference_target, first.case.structural_intensity >= TARGET_THRESHOLD)
    assert not a.case.reference_motor.any() and not a.case.reference_language.any()
    assert a.metrics()["history"][0]["outcome_scope"] == "observed_scan_estimator"
    for action in OPENING_SEQUENCE[1:]:
        assert a.step(action).reward == b.step(action).reward


def test_primary_actor_has_scan_support_and_cavity_but_no_reference_channels():
    task = make_opening_task()
    observation = task.observation()
    np.testing.assert_array_equal(observation.channel_available, [True, True, False, True, False, False])
    assert not observation.image_channels[[2, 4, 5]].any()
    assert not observation.coverage[[2, 4, 5]].any()
    assert observation.action_geometry.shape == (2, 16)
    task.step(OPENING_SEQUENCE[0])
    assert task.observation().image_channels[3].sum() == 1.


def test_inventory_has_every_observed_frontier_tool_slot_and_no_omissions():
    task = make_spatial_task(0)
    inventory = task.candidate_inventory()
    assert inventory["complete"] and inventory["omitted_count"] == 0
    assert inventory["slot_count"] == inventory["frontier_cells"] * inventory["tool_count"]
    assert inventory["slot_count"] == inventory["accepted_count"] + inventory["rejected_geometry_or_cavity_count"]
    assert inventory["accepted_action_ids"] == list(task.observation().action_ids[1:])
    assert inventory["scope"] == "complete_local_primitive_inventory_not_global_surgical_paths"


def test_unopened_target_and_repeated_or_post_stop_actions_are_refused():
    task = make_opening_task()
    with pytest.raises(InvalidActionError):
        task.step(OPENING_SEQUENCE[-1])
    task.step(OPENING_SEQUENCE[0])
    with pytest.raises(InvalidActionError):
        task.step(OPENING_SEQUENCE[0])
    task.step("STOP")
    with pytest.raises(InvalidActionError):
        task.step("STOP")
    assert task.metrics()["steps"] == 2
    assert task.independent_geometry_check().feasible


def test_clone_reset_and_fresh_preserve_type_and_isolate_cavity():
    task = make_opening_task()
    initial = task.observation()
    clone = task.clone()
    clone.step(OPENING_SEQUENCE[0])
    assert not task.observation().image_channels[3].any()
    assert clone.observation().image_channels[3].sum() == 1
    assert_observations_identical(task.fresh().observation(), initial)
    assert_observations_identical(clone.reset(7), initial)
    assert clone.metrics()["history"] == []


def test_scan_estimator_is_imperfect_and_never_reads_reference():
    task = make_opening_task()
    dim_scan = task.case.structural_intensity.copy()
    # This deliberately falls below the declared generator's target class.
    dim_scan[task.case.reference_target] = .3
    dim = SpatialTask(replace(task.case, structural_intensity=dim_scan), max_steps=3)
    planning = dim.planning_clone()
    assert not planning.case.reference_target.any()
    for action in OPENING_SEQUENCE:
        dim.step(action)
        planning.step(action)
    assert dim.metrics()["target_removed_mm3"] == 1
    assert planning.metrics()["target_removed_mm3"] == 0
    assert dim.metrics()["total_reward"] > 0 > planning.metrics()["total_reward"]


def test_generated_source_has_declared_forward_model_and_actual_shape_identity():
    first, repeat, other = make_spatial_case(0), make_spatial_case(0), make_spatial_case(1)
    assert first.source_hash == repeat.source_hash and first.anatomy_hash == repeat.anatomy_hash
    assert first.anatomy_hash != other.anatomy_hash
    assert first.recipe["forward_model"] and first.recipe["planning_target_estimator"]
    noisy = replace(first, structural_intensity=np.where(first.structural_intensity > .05,
                                                       first.structural_intensity + .001, 0.))
    assert noisy.source_hash != first.source_hash and noisy.anatomy_hash == first.anatomy_hash
    with pytest.raises(ValueError):
        first.structural_intensity.setflags(write=True)


def test_reward_change_during_episode_is_refused_before_removal():
    task = make_opening_task()
    task.reward_spec = replace(task.reward_spec, target_per_mm3=100.)
    with pytest.raises(RuntimeError, match="reward changed"):
        task.step(OPENING_SEQUENCE[0])
    assert not task._geometry.removed_mask.any()


@pytest.mark.parametrize("field", ["motor_per_mm3", "language_per_mm3", "graph_edge_cost"])
def test_hidden_functional_reward_cannot_diverge_from_scan_only_search_objective(field):
    with pytest.raises(ValueError, match="zero functional"):
        SpatialTask(make_opening_task().case, reward=replace(DEFAULT_REWARD, **{field: 1.}))


def test_unknown_function_and_physical_effort_are_separate_from_geometric_reward():
    base = make_opening_task()
    task = SpatialTask(replace(base.case, reference_motor=np.ones((3, 3, 3))), max_steps=3)
    planner = task.planning_clone()
    for action in OPENING_SEQUENCE:
        assert task.step(action).reward == planner.step(action).reward
    metrics, planning = task.metrics(), planner.metrics()
    assert metrics["function_unassessed_removed_volume_mm3"] == 3.
    assert metrics["motor_surrogate_exposure"] == 3.
    assert planning["motor_surrogate_exposure"] is None
    assert metrics["insertion_distance_mm"] == pytest.approx(.5 + np.sqrt(1.25) + np.sqrt(3.25))
    assert metrics["withdrawal_distance_mm"] == metrics["insertion_distance_mm"]
    assert metrics["complete_tool_path_length_mm"] == 2 * metrics["insertion_distance_mm"]
    assert metrics["clinical_deficit_probability"] is None


@pytest.mark.parametrize("changes", [{"morphology": "unspecified"}, {"tool_regime": "unknown"}])
def test_undeclared_generator_regime_is_refused(changes):
    with pytest.raises(ValueError):
        make_spatial_case(0, **changes)
