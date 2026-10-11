"""Small generated controls; no patient data, checkpoint, or model forward."""
from __future__ import annotations

import numpy as np
import pytest

import resectionlab.persistent_passive as passive_module
from resectionlab.development_episode import make_development_task, _select, TOOLS
from resectionlab.geometry import AccessWindow, ToolGeometry, ToolPose
from resectionlab.native_spatial_task import NativeSpatialCase
from resectionlab.persistent_passive import (
    MOVE, INSPECT, WITHDRAW, WITHDRAW_STATIONARY, STOP, GeneratedInspectionInput,
    PassiveProtocol, PersistentPassiveTask, verify_passive_replay,
    export_passive_episode, verify_passive_episode,
    make_passive_development_task, run_passive_development_episode,
)
from resectionlab.native_axis_simulation import CommittedTransitionInterrupted


def make_case(*, collision=False):
    shape = (13, 13, 12)
    support = np.zeros(shape, bool)
    support[4:9, 4:9, 4:9] = True
    nominal = np.zeros(shape, bool)
    nominal[5:8, 5:8, 6:9] = True
    image = np.where(nominal, .8, np.where(support, .2, 0.)).astype(np.float32)
    probe = ToolGeometry("generated-passive-probe", .15, .1, 12., 30., .4)
    other = ToolGeometry("generated-aspirator", .15 if collision else 1.25,
                         .1 if collision else .35, 12., 30., .4 if collision else 1.5)
    return NativeSpatialCase(image, support, nominal, np.eye(4),
        AccessWindow((6., 6., 1.5), (0., 0., 1.), 2., "generated-access"),
        (other, probe), track="synthetic_scan", support_source_kind="derived_from_scan",
        support_derivation="analytic generated support from source intensity",
        nominal_target=nominal, target_source_kind="derived_from_scan",
        target_derivation="analytic generated target, not an MRI estimator")


def make_task(status="unresolved", *, collision=False, cancelled=None):
    if not collision:
        return make_passive_development_task(inspection_status=status, cancelled=cancelled)
    case = make_case(collision=collision)
    start = ToolPose((6., 6., 1.5), (0., 0., 1.))
    hold = ToolPose((6., 6., 2.2), (0., 0., 1.))
    stationary = ToolPose((6., 6., 2.2), (.45, 0., (1 - .45 ** 2) ** .5)) if collision else None
    exit_pose = ToolPose((6. - .45 * .7, 6., 2.2 - (1 - .45 ** 2) ** .5 * .7),
                         stationary.axis_unit) if collision else None
    protocol = PassiveProtocol(case.tools[1].tool_id, start, hold,
        case.tools[0].tool_id if collision else None, stationary, exit_pose)
    return PersistentPassiveTask(case, protocol,
        GeneratedInspectionInput(case.source_hash, status), cancelled=cancelled)


def test_held_pose_observation_withdraw_and_fresh_replay():
    task = make_task()
    original_masks = task._engine.committed_mask_digests()
    assert task._protocol.hold_pose.tip_mm[2] == 6.5 > -.5
    assert task._protocol.hold_pose.proximal_mm(task._moving_tool())[2] == .5
    assert np.all(~task.case.observed_support[6, 6, 0:8])
    assert task.case.observed_support[4, 6, 6] and task.case.observed_support[8, 6, 6]
    assert not task._engine.removed_mask.any() and not task._engine.contact_mask.any()
    actions = (MOVE, INSPECT, WITHDRAW, STOP)
    rows = []
    for action in actions:
        rows.append(task.step(action).info)
    assert task.terminated and task._elapsed == 3.0
    assert [row["duration_generated_seconds"] for row in rows] == [1., 1., 1., 0.]
    assert rows[0]["pose_after"] == rows[1]["pose_before"] == rows[1]["pose_after"]
    assert rows[1]["information_event"]["status"] == "unresolved"
    assert rows[-1]["stop_reason"] == "generated_registration_unresolved"
    assert task._engine.committed_mask_digests() == original_masks
    assert verify_passive_replay(task, actions, tuple(rows))


def test_generated_information_is_not_visible_before_acquisition():
    uncertain, verified = make_task("unresolved"), make_task("verified")
    assert uncertain.observation().fingerprint == verified.observation().fingerprint
    for task in (uncertain, verified):
        task.step(MOVE)
    assert uncertain.observation().fingerprint == verified.observation().fingerprint
    uncertain.step(INSPECT)
    verified.step(INSPECT)
    assert uncertain.observation().inspection_status == "unresolved"
    assert verified.observation().inspection_status == "verified"
    assert uncertain.observation().fingerprint != verified.observation().fingerprint


def test_no_silent_stop_or_native_cut_while_held_and_no_stale_commit():
    task = make_task()
    task.step(MOVE)
    with pytest.raises(ValueError, match="unsupported passive action"):
        task.step(STOP)
    with pytest.raises(ValueError, match="unsupported passive action"):
        task.step("NATIVE-SPATIAL:unreviewed")
    task._elapsed += 1.
    with pytest.raises(RuntimeError, match="changed outside a transition"):
        task.step(INSPECT)


def test_stationary_second_tool_collision_refuses_without_mutation():
    task = make_task(collision=True)
    before = task._state_seal
    with pytest.raises(ValueError, match="TOOL_COLLISION"):
        task.step(MOVE)
    assert task._state_seal == before and task._steps == 0


def test_both_retained_tools_require_explicit_separate_withdrawal():
    task = make_passive_development_task(inspection_status="verified", paired=True)
    case, protocol = task.case, task._protocol
    actions = (MOVE, INSPECT, WITHDRAW, WITHDRAW_STATIONARY, STOP)
    rows = []
    for action in actions[:-1]:
        rows.append(task.step(action).info)
        if action != WITHDRAW_STATIONARY:
            assert not task.observation().action_mask[0]
    assert task.observation().action_mask[0]
    rows.append(task.step(STOP).info)
    assert task.terminated and task._elapsed == 4.0 and task._steps == 5
    assert rows[3]["tool_id"] == case.tools[0].tool_id
    assert rows[3]["pose_before"]["tip_mm"] == list(protocol.stationary_pose.tip_mm)
    assert rows[3]["pose_after"]["tip_mm"] == list(protocol.stationary_exit_pose.tip_mm)
    assert rows[3]["operative_after"] != rows[2]["operative_after"]
    assert verify_passive_replay(task, actions, tuple(rows))
    final = task._operative_record()
    for role in ("moving_tool", "other_tool"):
        assert final[role]["pose_status"] == "access_pose_recorded"
        assert final[role]["pose"] is not None
        assert final[role]["retained_at_depth"] is False
        assert final[role]["physical_absence_claim"] is False


def test_second_withdrawal_checks_the_parked_moving_tool(monkeypatch):
    task = make_passive_development_task(inspection_status="verified", paired=True)
    case, protocol = task.case, task._protocol
    for action in (MOVE, INSPECT, WITHDRAW):
        task.step(action)
    seen = []
    original = passive_module.check_motion
    def checked(*args, **kwargs):
        seen.append(kwargs["other_tools"])
        return original(*args, **kwargs)
    monkeypatch.setattr(passive_module, "check_motion", checked)
    task.step(WITHDRAW_STATIONARY)
    assert len(seen) == 1 and len(seen[0]) == 1
    assert seen[0][0][0].tool_id == case.tools[1].tool_id
    assert np.array_equal(seen[0][0][1].tip_mm, protocol.start_pose.tip_mm)


def test_held_clone_independent_and_tampered_replay_refused():
    task = make_task()
    task.step(MOVE)
    clone = task.clone()
    assert clone._state_seal == task._state_seal
    task.step(INSPECT)
    assert clone._phase == "held" and clone._visible_status == "not_acquired"
    actions = (MOVE, INSPECT, WITHDRAW, STOP)
    records = [task._history[0], task._history[1]]
    for action in actions[2:]:
        records.append(task.step(action).info)
    tampered = [dict(row) for row in records]
    tampered[1]["duration_generated_seconds"] = 2.0
    with pytest.raises(ValueError, match="differs"):
        verify_passive_replay(task, actions, tuple(tampered))


def test_versioned_episode_frames_and_tamper_refusal():
    task = make_task()
    for action in (MOVE, INSPECT, WITHDRAW, STOP):
        task.step(action)
    episode = export_passive_episode(task)
    assert len(episode["actions"]) == 4 and len(episode["frames"]) == 5
    assert episode["frames"][0]["operative"]["inspection_status"] == "not_acquired"
    assert episode["frames"][1]["operative"]["inspection_status"] == "not_acquired"
    assert episode["frames"][2]["operative"]["inspection_status"] == "unresolved"
    assert episode["frames"][-1]["operative"]["moving_tool"]["pose_status"] == "access_pose_recorded"
    assert [frame["operative"]["elapsed_generated_seconds"] for frame in episode["frames"]] == [0., 1., 2., 3., 3.]
    assert verify_passive_episode(task, episode)
    tampered = dict(episode)
    tampered["frames"] = [dict(frame) for frame in episode["frames"]]
    tampered["frames"][1]["operative"] = {**tampered["frames"][1]["operative"], "phase": "stopped"}
    with pytest.raises(ValueError, match="differs"):
        verify_passive_episode(task, tampered)
    fixed = run_passive_development_episode()
    assert fixed == episode
    paired = run_passive_development_episode(paired=True)
    assert len(paired["actions"]) == 5 and len(paired["frames"]) == 6


def test_bounded_schedule_and_no_stop_before_withdrawal():
    task = make_task()
    assert not task.observation().action_mask[0]
    with pytest.raises(ValueError, match="at most 60 seconds"):
        PassiveProtocol(task.case.tools[1].tool_id, task._protocol.start_pose,
                        task._protocol.hold_pose, move_seconds=61.).record()


def test_cancellation_before_passive_motion_preserves_state():
    cancellation = {"value": False}
    task = make_task(cancelled=lambda: cancellation["value"])
    before = task._state_seal
    cancellation["value"] = True
    with pytest.raises(InterruptedError):
        task.step(MOVE)
    cancellation["value"] = False
    assert task._state_seal == before and task._steps == 0


def test_cancellation_during_geometry_is_precommit(monkeypatch):
    cancellation = {"value": False}
    task = make_task(cancelled=lambda: cancellation["value"])
    original = passive_module.check_motion
    def checked(*args, **kwargs):
        result = original(*args, **kwargs)
        cancellation["value"] = True
        return result
    monkeypatch.setattr(passive_module, "check_motion", checked)
    before = task._state_seal
    with pytest.raises(InterruptedError):
        task.step(MOVE)
    cancellation["value"] = False
    assert task._state_seal == before and task._steps == 0 and task._phase == "ready"


def test_cancellation_after_commit_returns_durable_transition(monkeypatch):
    task = make_task()
    original = task.observation
    observations = {"count": 0}
    def observed():
        observations["count"] += 1
        if observations["count"] == 2:
            raise InterruptedError("generated postcommit cancellation")
        return original()
    monkeypatch.setattr(task, "observation", observed)
    with pytest.raises(CommittedTransitionInterrupted) as caught:
        task.step(MOVE)
    assert caught.value.info["action_id"] == MOVE
    assert task._steps == 1 and task._phase == "held"
    assert task._history[0] == caught.value.info


def test_default_shared_task_remains_the_existing_native_stroke():
    task = make_development_task()
    assert type(task).__name__ == "NativeSpatialTask"
    action = _select(task, TOOLS[0].tool_id, (6, 6, 2))
    outcome = task.step(action)
    assert outcome.info["retraction"] == "reverse_identical_insertion_path_after_removal_no_in_brain_reorientation"
    assert not hasattr(task, "_phase") and len(task._engine.history) == 1
