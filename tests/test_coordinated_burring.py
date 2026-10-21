"""Physical necessity, replay and reward exploits on generated software geometry."""
import copy
import numpy as np
import pytest
from resectionlab.coordinated_burring import BurringTask, TaskConfig, QPolicy
from resectionlab.surgical_engine import SurgicalEngine
from resectionlab.surgical_scene import SurgicalScene
from resectionlab.surgical_rl_protection import ProtectionSet, GuardConfig
from resectionlab.surgical_physics_v2 import default_profile


@pytest.fixture
def task():
    labels = np.zeros((32, 32, 24), np.uint16)
    labels[8:17, 8:17, 10:13] = 1
    scene = SurgicalScene(labels, np.eye(4), [dict(id=1, name='generated bone', material='bone')],
                          {'provenance': 'generated software fixture'}, {'tissuePhysicsV2': default_profile()})
    engine = SurgicalEngine(scene, protection=ProtectionSet(scene),
                            guard_config=GuardConfig(require_critical_coverage=False))
    assert engine.step(dict(slot='right', operation='exchange', instrument='frazier_suction', duration=.2))['status'] == 'accepted'
    return BurringTask(engine, dict(position=[12., 12., -5.], quaternion=[0., 0., 0., 1.]),
                       dict(position=[12., 12., 8.], quaternion=[0., 0., 0., 1.]),
                       TaskConfig(minimum_bone_mm3=1., maximum_bone_mm3=150.),
                       burr_pose=dict(position=[12., 12., 13.], quaternion=[0., 0., 0., 1.]))


def test_two_instrument_success_and_exact_guarded_replay(task):
    labels = task.engine.scene.labels.copy()
    task.step(1)
    assert task.measurements['bone_removed_mm3'] > 0 and not task.success()
    task.step(2)
    assert task.done and task.success() and task.reason == 'success'
    assert task.engine.state['debrisRemovedMm3'] == task.measurements['bone_removed_mm3']
    replay = SurgicalEngine.restore(task.engine.scene, task.engine.snapshot())
    assert replay.state_id == task.engine.state_id
    assert replay.guard_report() == task.engine.guard_report()
    np.testing.assert_array_equal(labels, task.engine.scene.labels)
    assert not any(a['status'] == 'rejected' for a in task.attempts)
    with pytest.raises(RuntimeError): task.step(1)


@pytest.mark.parametrize('choices', [(1, 1, 1, 1), (2, 2, 2, 2), (2, 1, 0), (0,)])
def test_instrument_and_order_ablations_cannot_complete(task, choices):
    for choice in choices:
        if not task.done: task.step(choice)
    assert task.done and not task.success()


def test_repeated_suction_has_no_clearance_refund_and_clones_detach(task):
    before = task.engine.state_id
    fork = task.clone()
    fork.step(2)
    fork.step(2)
    assert task.engine.state_id == before and not task.decisions
    assert all(d['terms']['clearance'] == 0 and d['reward'] < 0 for d in fork.decisions)
    assert fork.engine.scene is task.engine.scene
    assert task.observe() == {'bone_mm3': 0., 'debris_mm3': 0., 'remaining': 4}


def test_bone_cap_fails_without_removing_more_than_real_engine_did(task):
    task.config = TaskConfig(minimum_bone_mm3=1., maximum_bone_mm3=1.)
    task.step(1)
    assert task.done and not task.success() and task.reason == 'guard_or_budget_failure'
    assert task.measurements['bone_removed_mm3'] > 1  # post-action cap can overshoot
    assert task.decisions[-1]['terms']['failure'] == -1.


def test_actual_td_update_changes_policy_and_does_not_fit_on_reads():
    policy = QPolicy()
    start = dict(bone_mm3=0., debris_mm3=0., remaining=4)
    drilled = dict(bone_mm3=34., debris_mm3=34., remaining=3)
    done = dict(bone_mm3=34., debris_mm3=0., remaining=2)
    assert policy.choose(start) == 0
    policy.update(drilled, 2, 1., done, True)
    policy.update(start, 1, .1, drilled, False)
    assert policy.choose(start) == 1 and policy.choose(drilled) == 2
    before = copy.deepcopy(policy.record())
    policy.choose(done)
    assert policy.record() == before and policy.updates == 2
