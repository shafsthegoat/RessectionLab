"""Generated interface/geometry controls; no learning or checkpoint calls."""
import copy
import json
from dataclasses import replace
import numpy as np
import pytest
from resectionlab.public_surface_contact import (PublicSurfaceGoal, SurfaceContactTask, SurfaceContactObservation,
    SurfaceContactDevelopmentContext, make_contact_task, seal_complete_strategy,
    verify_complete_strategy, OBSERVATION_VERSION)
from resectionlab.core import array_digest, semantic_digest
from resectionlab.data_policy import GeneratedDevelopmentContext
from resectionlab.development_episode import make_development_task, _select, TOOLS
from resectionlab.native_spatial_task import NativeSpatialTask
from resectionlab.observed_search import observed_beam_search


def select(task, mode, depth):
    return _select(task, TOOLS[0 if mode=='aspirate' else 1].tool_id, (6,6,depth))


def manual(task, sequence):
    ids=[]
    for mode,depth in sequence:
        action='STOP' if mode=='stop' else select(task,mode,depth)
        ids.append(action);task.step(action)
    return ids


def context(task):
    return task.development_context(semantic_digest({'purpose':'generated unit controls'}))


def test_uses_inherited_native_transitions_and_physical_actions():
    assert SurfaceContactTask.step is NativeSpatialTask.step
    assert SurfaceContactTask._transition is NativeSpatialTask._transition
    assert SurfaceContactTask.clone is NativeSpatialTask.clone
    legacy=make_development_task();goal=make_contact_task()
    a=legacy.step(select(legacy,'aspirate',2));b=goal.step(select(goal,'aspirate',2))
    for key in ('tool_id','entry_mm','tip_mm','axis_unit','removed_indices_native','contact_indices_native','microsteps','retraction'):
        assert a.info[key]==b.info[key]
    np.testing.assert_array_equal(legacy._engine.removed_mask,goal._engine.removed_mask)
    assert goal.independent_geometry_check().feasible


def test_negative_opening_then_probe_useful_without_removal():
    task=make_contact_task()
    first=task.step(select(task,'aspirate',2));before=task._engine.removed_mask.copy()
    second=task.step(select(task,'probe',2))
    assert first.reward==pytest.approx(-.231)
    assert second.reward==pytest.approx(.939)
    assert task.metrics()['total_reward']==pytest.approx(.708)
    assert task.terminated and task.metrics()['goal_contacted_and_retained']
    assert second.info['removed_indices_native']==[]
    np.testing.assert_array_equal(before,task._engine.removed_mask)
    assert task.independent_geometry_check().feasible


def test_deep_destruction_and_stop_remain_meaningfully_distinct():
    deep=make_contact_task();manual(deep,[('aspirate',6),('stop',0)])
    stop=make_contact_task();manual(stop,[('stop',0)])
    assert not deep.metrics()['goal_retained'] and not deep.metrics()['goal_contacted_and_retained']
    assert deep.metrics()['total_reward']==pytest.approx(-1.039)
    assert stop.metrics()['goal_retained'] and not stop.metrics()['goal_contacted_and_retained']
    assert stop.metrics()['total_reward']==0 and stop.metrics()['steps']==1


def test_search_retains_negative_prefix_and_replays_identical_inventory():
    task=make_contact_task()
    sequence,accounting=observed_beam_search(task,max_calls=256,beam_width=96,seconds=4.,
        objective_source='explicit_public_retained_surface_goal',transition_mode='lazy_planning')
    assert not accounting['call_cap_reached'] and not accounting['time_cap_reached']
    assert accounting['completed_layers']==2 and accounting['negative_prefixes_retained']>0
    other=task.fresh();expected=manual(other,[('aspirate',2),('probe',2)])
    assert tuple(expected)==sequence and task.metrics()['steps']==0
    package=seal_complete_strategy(task,sequence)
    assert package['metrics']['total_reward']==pytest.approx(.708)
    assert verify_complete_strategy(task,json.loads(json.dumps(package)))
    assert [r['interaction_mode'] for r in package['strategy']['history']]==['aspirate','probe']
    assert any(f['phase']=='withdrawal' for f in package['replayFrames'])


def test_replay_rejects_changed_motion_even_when_resealed():
    task=make_contact_task();ids=manual(task.clone(),[('aspirate',2),('probe',2)])
    package=seal_complete_strategy(task,ids)
    altered=copy.deepcopy(package)
    altered['strategy']['history'][0]['microsteps'][0]['tip_end_mm'][0]+=.1
    altered['strategySeal']=semantic_digest(altered['strategy'])
    with pytest.raises(ValueError,match='replay'):
        verify_complete_strategy(task,altered)


def test_replay_rejects_changed_reward_and_goal_identity():
    task=make_contact_task();ids=manual(task.clone(),[('aspirate',2),('probe',2)])
    package=seal_complete_strategy(task,ids)
    altered=copy.deepcopy(package);altered['strategy']['history'][0]['reward']=5
    altered['strategySeal']=semantic_digest(altered['strategy'])
    with pytest.raises(ValueError,match='replay'):verify_complete_strategy(task,altered)
    with pytest.raises(ValueError):verify_complete_strategy(make_contact_task(goal=(6,6,5)),package)


def test_replay_preserves_json_types_in_metrics_and_audit():
    task=make_contact_task();package=seal_complete_strategy(task,['STOP'])
    altered=copy.deepcopy(package);altered['metrics']['steps']=1.0
    with pytest.raises(ValueError,match='replay'):verify_complete_strategy(task,altered)
    altered=copy.deepcopy(package);altered['geometryAudit']['feasible']=1
    with pytest.raises(ValueError,match='replay'):verify_complete_strategy(task,altered)


def test_search_can_choose_stop_when_retained_contact_costs_more_than_reward():
    task=make_contact_task(goal=(6,6,7))
    feasible=task.clone();manual(feasible,[('aspirate',6),('probe',6)])
    assert feasible.metrics()['goal_contacted_and_retained']
    assert feasible.metrics()['total_reward']<0
    sequence,accounting=observed_beam_search(task,max_calls=256,beam_width=96,seconds=4.,
        objective_source='explicit_public_retained_surface_goal',transition_mode='lazy_planning')
    assert not accounting['call_cap_reached'] and not accounting['time_cap_reached']
    assert accounting['completed_layers']==2 and accounting['beam_pruned_prefixes']==0
    assert sequence==('STOP',)


def test_private_reference_cannot_change_observation_actions_reward_or_export():
    a=make_contact_task(reference_target=np.zeros((13,13,12),np.float32))
    b=make_contact_task(reference_target=np.ones((13,13,12),np.float32))
    assert a.case.reference_hash!=b.case.reference_hash
    assert a.observation().fingerprint==b.observation().fingerprint
    assert a.candidate_inventory()==b.candidate_inventory()
    actions=manual(a.clone(),[('aspirate',2),('probe',2)])
    assert seal_complete_strategy(a,actions)==seal_complete_strategy(b,actions)
    assert 'reference_hash' not in a.metrics() and 'target_removed_mm3' not in a.metrics()
    assert not a.planning_clone().case.reference_target.any()


def test_planning_clone_keeps_prefix_reward_and_history_without_double_credit():
    task=make_contact_task();task.step(select(task,'aspirate',2))
    planning=task.planning_clone()
    assert planning.metrics()['history']==task.metrics()['history']
    assert planning.metrics()['total_reward']==task.metrics()['total_reward']
    a=task.step(select(task,'probe',2));b=planning.advance_planning(select(planning,'probe',2))
    assert a.reward==b.reward==pytest.approx(.939)
    assert task.metrics()['history']==planning.metrics()['history']
    assert task.planning_clone().metrics()['history']==task.metrics()['history']
    assert task.planning_clone().fresh().metrics()['steps']==0


def test_public_observation_context_and_legacy_context_separation():
    task=make_contact_task();obs=task.observation();permit=context(task)
    assert type(obs) is SurfaceContactObservation
    permit.require_task(task);permit.require_observation(obs)
    assert obs.public_goal_grid.sum()==1 and not obs.public_goal_grid.flags.writeable
    assert obs.action_modes[0]=='stop'
    with pytest.raises(TypeError):permit.require_observation(obs.base)
    old=GeneratedDevelopmentContext('1'*64,(task.case.source_hash,),task.decision_model_hash)
    with pytest.raises(TypeError,match='validated spatial observation DTO'):old.require_observations([obs])
    with pytest.raises(ValueError):replace(permit,objective_hash='sha256:'+'2'*64).require_observation(obs)
    with pytest.raises(ValueError):permit.require_task(make_contact_task(goal=(6,6,5)))


@pytest.mark.parametrize('goal',[(True,6,3),(-1,6,3),(13,6,3),(0,0,0)])
def test_invalid_or_unobserved_goal_refused(goal):
    with pytest.raises(ValueError):make_contact_task(goal=goal)


def test_goal_mutation_and_stale_action_refused():
    task=make_contact_task();obs=task.observation();aid=select(task,'aspirate',2)
    task.step(aid)
    with pytest.raises(ValueError):task.step(aid)
    object.__setattr__(obs,'crop_origin_native',(1,0,0))
    with pytest.raises(ValueError):obs.assert_intact()
    object.__setattr__(task.objective,'native_index',(6,6,5))
    with pytest.raises(RuntimeError):task.observation()


def test_direct_wrong_goal_grid_rejected_before_detached_use():
    task=make_contact_task();obs=task.observation();grid=obs.public_goal_grid.copy()
    grid[:]=False;grid[6,6,4]=True
    with pytest.raises(ValueError,match='Goal grid differs'):
        replace(obs,public_goal_grid=grid)
    other_goal=PublicSurfaceGoal(task.case.source_hash,(6,6,4))
    claimed=replace(obs,objective=other_goal,public_goal_grid=grid)
    with pytest.raises(ValueError,match='differs'):context(task).require_observation(claimed)


def test_consistent_shifted_origin_and_grid_still_refused_by_task_context():
    task=make_contact_task();obs=task.observation();grid=obs.public_goal_grid.copy()
    grid[:]=False;grid[5,6,3]=True
    claimed=replace(obs,crop_origin_native=(1,0,0),public_goal_grid=grid)
    with pytest.raises(ValueError,match='differs'):context(task).require_observation(claimed)


def test_detached_frame_relabel_and_expected_grid_replacement_refused():
    task=make_contact_task();obs=task.observation();affine=obs.base.base.affine_ras_mm.copy()
    affine[0,3]+=1
    base=replace(obs.base,base=replace(obs.base.base,affine_ras_mm=affine))
    claimed=replace(obs,base=base)
    permit=context(task)
    with pytest.raises(ValueError,match='differs'):permit.require_observation(claimed)
    object.__setattr__(permit,'goal_grid_hash','sha256:'+'2'*64)
    with pytest.raises(ValueError,match='changed'):permit.require_observation(obs)
