"""Independent endpoint, admission, prefix and observation contract controls."""
import copy
from dataclasses import replace
import json
import numpy as np
import pytest

from resectionlab.surgical_sequential_task import (COMMON_CONFIG, GeneratedTask, Skill,
    SequentialPrimitiveAdapter, action, available_mask, endpoint_assessment,
    execute_skill, generated_task, observation, training_admission)
from resectionlab.surgical_sequential_benchmark import feature_matrix


def skill(task, kind, radius=None):
    return next(s for s in task.skills if s.kind==kind and (radius is None or s.radius==radius))


def test_runtime_binds_inventory_engine_config_and_uncharged_prefix():
    task=generated_task(0);other=generated_task(1);engine=task.initial_engine()
    with pytest.raises(ValueError,match='BOUND_INVENTORY'):
        execute_skill(task,engine,Skill('unbound','prepare',(action('inspect'),)))
    with pytest.raises(ValueError,match='BOUND_INVENTORY'):
        execute_skill(task,other.initial_engine(),skill(task,'work'))
    assert engine.step(action('inspect'))['status']=='accepted'
    with pytest.raises(ValueError,match='UNCHARGED_PREFIX'):
        execute_skill(task,engine,skill(task,'work'))
    changed=copy.copy(task);changed.config=replace(COMMON_CONFIG,max_attempts=10)
    with pytest.raises(ValueError,match='TRAINING_ADMISSION_REFUSED'):
        training_admission(changed)


@pytest.mark.parametrize('limit',[1,2,3])
def test_partial_skill_attempt_limit_retains_exact_prefix_and_matches_adapter(limit):
    config=replace(COMMON_CONFIG,max_attempts=limit)
    task=generated_task(0,config=config);engine=task.initial_engine()
    # The shortest route is exposed even with the catalog tip.
    work=min((s for s in task.skills if s.kind=='work'),key=lambda s:s.route[1][2])
    receipt=execute_skill(task,engine,work)
    assert receipt['attemptsConsumed']==limit and receipt['endpoint']['truncated']
    assert len(engine.actions)==limit
    adapter=SequentialPrimitiveAdapter(task.scene,task_config=config);adapter.reset(seed=3)
    for row in receipt['receipts']:
        out=adapter.step_action(row['action'])
        assert out[4]['outcome']['status']==row['outcome']['status']
    assert out[2:4]==(False,True)
    assert adapter.engine.state_id==engine.state_id
    assert adapter.completion_assessment==receipt['endpoint']
    expected=(receipt['targetIncrementMm3']-receipt['neuralIncrementMm3'])/27.-.01*receipt['elapsedSeconds']-.01*limit
    assert receipt['reward']==pytest.approx(expected)
    saved=json.loads(json.dumps(adapter.snapshot(),allow_nan=False))
    restored=SequentialPrimitiveAdapter(task.scene,task_config=config);restored.restore(saved)
    assert restored.engine.state_id==engine.state_id and restored.truncated
    with pytest.raises(RuntimeError,match='EPISODE_FINISHED'):
        execute_skill(task,engine,skill(task,'stop'))


def test_rejection_zero_credit_prefix_and_real_work_withdraw_stop_parity():
    task=generated_task(0);engine=task.initial_engine()
    adapter=SequentialPrimitiveAdapter(task.scene);adapter.reset(seed=2)
    unsafe=next(s for s in task.skills if s.kind=='work' and s.route[1][2]>min(v.route[1][2] for v in task.skills if v.kind=='work'))
    refused=execute_skill(task,engine,unsafe)
    assert refused['status']=='rejected' and refused['targetIncrementMm3']==0
    assert len(engine.actions)==1 and refused['attemptsConsumed']==2
    for r in refused['receipts']:adapter.step_action(r['action'])
    assert adapter.engine.state_id==engine.state_id
    prep=execute_skill(task,engine,skill(task,'prepare',.4))
    for r in prep['receipts']:adapter.step_action(r['action'])
    for s in task.skills:
        if s.kind=='work':
            got=execute_skill(task,engine,s)
            for r in got['receipts']:adapter.step_action(r['action'])
    stopped=execute_skill(task,engine,skill(task,'stop'))
    final=adapter.stop('Finite generated inventory exhausted')
    assert final[2:4]==(True,False)
    assert adapter.engine.state_id==engine.state_id
    assert stopped['endpoint']==adapter.completion_assessment
    assert stopped['endpoint']['classification']=='successful-treatment'
    assert engine.state['debrisMm3']==0 and stopped['endpoint']['validWithdrawal']


def test_observation_reset_no_ids_or_future_and_candidate_order_equivariance():
    task=generated_task(0);a=task.initial_engine();b=task.initial_engine()
    observed=observation(task,a);other=observation(task,b)
    for name in ('source_labels','affine'):np.testing.assert_array_equal(observed[name],other[name])
    assert observed['native_state']==other['native_state'] and a.state_id==b.state_id
    assert set(observed)=={'source_labels','affine','memberships','native_state','accepted_history','declared_limits','remaining'}
    features=feature_matrix(observed,task.skills)
    order=np.arange(len(task.skills))[::-1]
    np.testing.assert_array_equal(feature_matrix(observed,tuple(task.skills[i] for i in order)),features[order])
    np.testing.assert_array_equal(available_mask(task,a,set()),available_mask(task,b,set()))
    observed['native_state']['removed'].append(1);observed['source_labels'].fill(0)
    assert not a.state['removed'] and task.scene.labels.any()


def test_motor_memberships_overlap_without_becoming_functional_outcomes():
    task=generated_task(2);observed=observation(task,task.initial_engine())
    assert observed['memberships']['structure:motor-cortex'] & observed['memberships']['structure:corticospinal']
    outcome=endpoint_assessment(task.initial_engine())
    assert outcome['progress']['originalTargetCells']==int((task.scene.labels==1).sum())
    assert training_admission(task)['generalPrimitiveTraining'] is False


def test_online_interruption_keeps_treatment_cost_reward_and_replayable_prefix():
    from resectionlab.surgical_sequential_benchmark import ComputeTier,run_episode,verify_replay
    task=generated_task(0)
    shortest=min((i for i,s in enumerate(task.skills) if s.kind=='work'),key=lambda i:task.skills[i].route[1][2])
    row=run_episode(task,'learned',ComputeTier('two-calls',2,20.,1,2,1),
        selector=lambda features,mask,stochastic:(shortest,None))
    assert row['computation']['physicalCalls']==2 and row['attempts']==2
    assert len(row['skillHistory'])==1 and row['skillHistory'][0]['status']=='interrupted'
    assert len(row['snapshot']['actions'])==2
    assert row['progress']['originalTargetTreatmentMm3']==27.
    assert row['reward']>0 and row['endpoint']['classification']=='truncated'
    assert not row['endpoint']['validWithdrawal']
    assert verify_replay(task,row)['verified']
    adapter=SequentialPrimitiveAdapter(task.scene);adapter.reset(seed=0)
    for receipt in row['skillHistory'][0]['receipts']:adapter.step_action(receipt['action'])
    adapter.truncate_external('Finite skill or online computation boundary')
    saved=json.loads(json.dumps(adapter.snapshot(),allow_nan=False))
    restored=SequentialPrimitiveAdapter(task.scene);restored.restore(saved)
    assert restored.completion_assessment==row['endpoint'] and restored.truncated


def test_later_work_relocates_tool_with_checked_motion():
    task=generated_task(3);engine=task.initial_engine()
    accepted=[]
    for s in task.skills:
        if s.kind=='work':
            r=execute_skill(task,engine,s)
            if r['status']=='accepted':accepted.append(r)
    assert accepted[0]['receipts'][0]['action']['operation']=='place'
    assert all(r['receipts'][0]['action']['operation']=='move' for r in accepted[1:])
