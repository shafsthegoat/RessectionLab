"""Narrow certified-preview greedy controls, using analytic sources only."""
import copy
from dataclasses import replace

import numpy as np
import pytest

from resectionlab.native_resection import NativeResectionEngine
from resectionlab.native_spatial_task import NativeSpatialTask, make_native_opening_task
from resectionlab.observed_search import ObservedSearchLimit


def positive_task():
    original = make_native_opening_task()
    source = replace(original.case, nominal_target=original.case.observed_support.astype(np.float32),
                     reference_target=original.case.observed_support.astype(np.float32))
    return NativeSpatialTask(source, max_steps=2)


def test_one_step_scores_match_actual_nominal_transitions_for_every_legal_action():
    task = positive_task()
    before = task.metrics()
    sequence, report = task.observed_one_step_search()
    scores = report["decisions"][0]["scores"]
    assert tuple(row["action_id"] for row in scores) == task.observation().action_ids
    assert report["decisions"][0]["all_current_legal_actions_scored"]
    for row in scores:
        child = task.planning_clone()
        actual = child.advance_planning(row["action_id"])
        assert row["reward"] == actual.reward
        assert row["target_removed_mm3"] == actual.info["target_removed_mm3"]
        assert row["normal_removed_mm3"] == actual.info["normal_removed_mm3"]
    assert report["evaluated_nonstop_actions"] == len(scores) - 1
    assert sequence[-1] == "STOP" and len(sequence) == 2
    assert task.metrics() == before


def test_greedy_repeats_through_same_horizon_and_replays_exactly():
    task = positive_task()
    sequence, report = task.observed_greedy_search()
    assert len(sequence) == task.max_steps and "STOP" not in sequence
    assert report["complete"] and len(report["decisions"]) == 2
    assert report["model_transition_calls"] == 2
    assert report["evaluated_nonstop_actions"] == sum(row["legal_nonstop_actions"] for row in report["decisions"])
    for identifier in sequence:
        task.step(identifier)
    assert task.terminated
    assert task.metrics()["total_reward"] == pytest.approx(report["estimated_incremental_return"])
    assert task.independent_geometry_check().feasible
    assert report["global_optimality_proven"] is False


def test_greedy_declares_myopic_failure_on_paid_opening_fixture():
    sequence, report = make_native_opening_task().observed_greedy_search()
    assert sequence == ("STOP",)
    assert report["estimated_incremental_return"] == 0.
    assert all(row["reward"] < 0 for row in report["decisions"][0]["scores"][1:])
    # This fixture has a positive two-action plan in existing exhaustive tests;
    # stopping here is the expected limitation, not claimed global optimality.


def test_private_reference_swaps_cannot_change_nominal_scores_or_selected_sequence():
    first = positive_task()
    second = NativeSpatialTask(replace(first.case, reference_target=np.zeros_like(first.case.reference_target)), max_steps=2)
    a, report_a = first.observed_greedy_search()
    b, report_b = second.observed_greedy_search()
    assert a == b
    assert report_a["decisions"] == report_b["decisions"]
    assert report_a["estimated_incremental_return"] == report_b["estimated_incremental_return"]
    for identifier in a:
        first.step(identifier)
        second.step(identifier)
    assert first.metrics()["total_reward"] != second.metrics()["total_reward"]


def test_one_step_reuses_current_certificates_without_successor_previews(monkeypatch):
    task = positive_task()
    task.observation()
    def forbidden(*args, **kwargs):
        raise AssertionError("Already certified immediate choices need no new preview")
    monkeypatch.setattr(NativeResectionEngine, "preview_stroke", forbidden)
    sequence, report = task.observed_one_step_search()
    assert sequence[-1] == "STOP" and report["complete"]
    assert task.metrics()["steps"] == 0


def test_forged_cached_certificate_is_not_scored_as_a_valid_plan():
    task = positive_task()
    inventory = task._prepare_inventory()
    identifier = next(iter(inventory))
    inventory[identifier] = copy.copy(inventory[identifier])
    with pytest.raises(ValueError, match="certificate"):
        task.observed_one_step_search()
    assert not task._engine.history


@pytest.mark.parametrize("seconds", [0, -1, True, float("inf"), float("nan")])
def test_invalid_budgets_refuse_without_state_change(seconds):
    task = positive_task()
    with pytest.raises(ValueError):
        task.observed_greedy_search(seconds=seconds)
    assert not task._engine.history


def test_timeout_retains_partial_accounting_but_never_publishes_complete_result():
    task = positive_task()
    with pytest.raises(ObservedSearchLimit) as error:
        task.observed_greedy_search(seconds=1e-12)
    assert error.value.accounting["complete"] is False
    assert not task._engine.history


def test_cancellation_never_changes_actual_task():
    task = positive_task()
    task._cancelled = lambda: True
    with pytest.raises(InterruptedError):
        task.observed_greedy_search()
    assert not task._engine.history
