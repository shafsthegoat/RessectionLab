"""Analytic transition parity only; no patient execution or model training."""
from dataclasses import replace

import numpy as np
import pytest

from resectionlab.native_axis_simulation import CommittedTransitionInterrupted
from resectionlab.native_proposals import NominalCavityProposalConfig
from resectionlab.native_spatial_task import NativeSpatialTask, OPENING_TOOLS, make_native_opening_task
from resectionlab.simulation import InvalidActionError


def source_task(mode, *, max_steps=2):
    source = make_native_opening_task().case
    if mode == "nominal_cavity_v1":
        source = replace(source, proposal_mode=mode, proposal_config=NominalCavityProposalConfig(
            tuple((a, b) for a in (-1, 0, 1) for b in (-1, 0, 1))))
    return NativeSpatialTask(source, max_steps=max_steps)


def pick(task, tool, voxel):
    return next(row["action_id"] for row in task.candidate_inventory()["emitted"]
        if row["feasible"] and row["tool_id"] == tool and row["voxel"] == list(voxel))


def same_state(eager, lazy):
    assert eager._state_record() == lazy._state_record()
    assert eager._state_seal == lazy._state_seal
    assert eager.metrics() == lazy.metrics()
    assert eager._engine.state_hash == lazy._engine.state_hash
    assert eager.decision_model_hash == lazy.decision_model_hash
    assert eager._engine.history == lazy._engine.history
    for name in ("removed_mask", "remaining_mask", "contact_mask", "connected_free_mask"):
        np.testing.assert_array_equal(getattr(eager._engine, name), getattr(lazy._engine, name))
    assert not eager._config.target_labels.any() and not lazy._config.target_labels.any()
    left, right = eager.independent_geometry_check(), lazy.independent_geometry_check()
    assert left.feasible and right.feasible and left.to_dict() == right.to_dict()


def same_observation(eager, lazy):
    left, right = eager.observation(), lazy.observation()
    assert left.fingerprint == right.fingerprint
    assert left.action_ids == right.action_ids and left.action_tool_ids == right.action_tool_ids
    for name in ("image_channels", "coverage", "channel_available", "affine_ras_mm", "spacing_mm",
                 "action_geometry", "action_mask", "state_features"):
        np.testing.assert_array_equal(getattr(left, name), getattr(right, name))
    assert eager.candidate_inventory() == lazy.candidate_inventory()


@pytest.mark.parametrize("mode", ["fixed_lattice", "nominal_cavity_v1"])
def test_lazy_transition_defers_only_successor_previews_with_exact_later_parity(mode, monkeypatch):
    eager = source_task(mode).planning_clone()
    lazy = eager.clone()
    calls = []
    preview = lazy._engine.preview_stroke
    def counted(*args, **kwargs):
        calls.append((args, kwargs))
        return preview(*args, **kwargs)
    monkeypatch.setattr(lazy._engine, "preview_stroke", counted)
    for tool, voxel in ((OPENING_TOOLS[0].tool_id, (4, 4, 1)), (OPENING_TOOLS[1].tool_id, (4, 4, 5))):
        identifier = pick(eager, tool, voxel)
        before = len(calls)
        expected = eager.step(identifier)
        actual = lazy.advance_planning(identifier)
        assert actual.observation is None and expected.observation is not None
        assert (actual.reward, actual.terminated, actual.info) == (expected.reward, expected.terminated, expected.info)
        assert len(calls) == before and lazy._inventory is None and lazy._proposal_batch is None
        same_state(eager, lazy)
        same_observation(eager, lazy)
        prepared = len(calls)
        assert (prepared > before) == (not lazy.terminated)
        same_observation(eager, lazy)
        assert len(calls) == prepared
    assert lazy.terminated and lazy.metrics()["target_removed_mm3"] == 2.
    with pytest.raises(InvalidActionError, match="terminated"):
        lazy.advance_planning("STOP")


@pytest.mark.parametrize("mode", ["fixed_lattice", "nominal_cavity_v1"])
@pytest.mark.parametrize("stop_after_opening", [False, True])
def test_lazy_stop_retains_terminal_horizon_and_empty_successor_inventory(mode, stop_after_opening):
    eager = source_task(mode, max_steps=3).planning_clone()
    lazy = eager.clone()
    if stop_after_opening:
        identifier = pick(eager, OPENING_TOOLS[0].tool_id, (4, 4, 1))
        eager.step(identifier)
        lazy.advance_planning(identifier)
        same_observation(eager, lazy)
    expected, actual = eager.step(0), lazy.advance_planning(0)
    assert expected.info == actual.info and actual.observation is None and actual.terminated
    same_state(eager, lazy)
    same_observation(eager, lazy)
    inventory = lazy.candidate_inventory()
    assert inventory["terminal"] and inventory["remaining_steps"] == 2 - int(stop_after_opening)
    assert inventory["emitted"] == [] and lazy.observation().action_ids == ("STOP",)


@pytest.mark.parametrize("mode", ["fixed_lattice", "nominal_cavity_v1"])
def test_lazy_refuses_invalid_masked_and_stale_actions_without_committed_mutation(mode):
    task = source_task(mode).planning_clone()
    rejected = next(row["action_id"] for row in task.candidate_inventory()["emitted"] if not row["feasible"])
    for identifier in (rejected, "foreign", True, -1, 10000):
        before = task._state_seal
        with pytest.raises(InvalidActionError):
            task.advance_planning(identifier)
        assert task._state_seal == before and task.metrics()["steps"] == 0
    old = pick(task, OPENING_TOOLS[0].tool_id, (4, 4, 1))
    task.advance_planning(old)
    before = task._state_seal
    with pytest.raises(InvalidActionError):
        task.advance_planning(old)
    assert task._state_seal == before and task.metrics()["steps"] == 1


@pytest.mark.parametrize("mode", ["fixed_lattice", "nominal_cavity_v1"])
def test_lazy_before_and_postcommit_cancellation_preserve_eager_accounting(mode, monkeypatch):
    eager = source_task(mode).planning_clone()
    lazy = eager.clone()
    identifier = pick(eager, OPENING_TOOLS[0].tool_id, (4, 4, 1))
    for task, method in ((eager, eager.step), (lazy, lazy.advance_planning)):
        flag = [True]
        task._cancelled = lambda flag=flag: flag[0]
        before = task._state_seal
        with pytest.raises(InterruptedError) as cancelled:
            method(identifier)
        assert not isinstance(cancelled.value, CommittedTransitionInterrupted)
        assert task._state_seal == before and task.metrics()["steps"] == 0
        flag[0] = False
        commit = task._engine.commit_preview
        def commit_then_cancel(result, commit=commit, flag=flag):
            value = commit(result)
            flag[0] = True
            return value
        monkeypatch.setattr(task._engine, "commit_preview", commit_then_cancel)
        with pytest.raises(CommittedTransitionInterrupted) as interrupted:
            method(identifier)
        assert interrupted.value.committed and interrupted.value.info["action_id"] == identifier
        assert interrupted.value.reward == task.metrics()["history"][0]["reward"]
        assert task.metrics()["steps"] == 1 and task._inventory is None
        flag[0] = False
    same_state(eager, lazy)
    same_observation(eager, lazy)


def test_lazy_advance_requires_nominal_clone_and_keeps_hidden_reference_out_of_transition():
    actual = source_task("nominal_cavity_v1")
    before = actual._state_seal
    with pytest.raises(ValueError, match="LAZY_PLANNING_ONLY"):
        actual.advance_planning("STOP")
    assert actual._state_seal == before and actual.metrics()["steps"] == 0
    altered = NativeSpatialTask(replace(actual.case, reference_target=np.ones(actual.case.reference_target.shape)), max_steps=2)
    first, second = actual.planning_clone(), altered.planning_clone()
    same_observation(first, second)
    identifier = pick(first, OPENING_TOOLS[0].tool_id, (4, 4, 1))
    assert first.advance_planning(identifier) == second.advance_planning(identifier)
    same_state(first, second)
    same_observation(first, second)
    assert actual.metrics()["steps"] == altered.metrics()["steps"] == 0


def test_lazy_current_action_still_requires_an_unchanged_native_certificate():
    task = source_task("nominal_cavity_v1").planning_clone()
    identifier = pick(task, OPENING_TOOLS[0].tool_id, (4, 4, 1))
    result = task._inventory[identifier]
    object.__setattr__(result, "tip_mm", tuple(value + .125 for value in result.tip_mm))
    before = task._state_seal
    with pytest.raises(ValueError, match="certificate changed"):
        task.advance_planning(identifier)
    assert task._state_seal == before and task.metrics()["steps"] == 0
