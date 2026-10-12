"""Generated-only reward/packet/Gym integration controls; root owns execution."""
import copy,sys
import paths
from paths import ROOT,HERE
sys.path.insert(0,str(ROOT/'tests'))
import numpy as np
import pytest
from resectionlab.core import freeze_json,semantic_digest
from resectionlab.native_spatial_task import NativeSpatialTask
from resectionlab.patient_planning_admission import PatientPlanningContext,COHORT_SHA256,_public_observation_binding
from test_obstruction_opening_proposals import source
import focused_policy as reused
from local_goal import GoalSpec,local_reward,admit_local_context,LocalEpisode
import gym_adapter
from gym_adapter import LocalGoalEnv


def fixture():
    old=NativeSpatialTask(source(enabled=False,cap=120,extra=((2,9,9),)),max_steps=24)
    obs=old.observation();record=freeze_json({'scope':'patient_native_planning_experiment','role':'TRAIN',
        'cohort_sha256':COHORT_SHA256,'subject':'ReMIND-045','patient_group':'ReMIND:045',
        'evidence_domain':'generated_interface_control','source_hash':old.case.source_hash,
        'decision_model_hash':old.decision_model_hash,'max_optimizer_updates':64,'max_steps':24,
        'observation_track':obs.track,'public_observation_binding':_public_observation_binding(obs)})
    context=PatientPlanningContext(record,semantic_digest(record));action=next(iter(old._prepare_inventory()))
    expected=old.clone().step(action).info;template,_=reused.protocol.enabled_task(old);work=template.planning_clone()
    packets,frontier,_=reused.protocol.map_prefix(old,work,[action],reused.protocol.compact_history([expected]))
    base=work.case._nominal_proposer.propose(work._engine)
    ray=next(r for r in base.proposals if tuple(r.voxel)==(9,9,9) and r.tool_id=='generic_suction')
    goal=GoalSpec(tuple(ray.voxel),ray.tool_id,tuple(ray.entry_mm),tuple(ray.tip_mm))
    admitted=admit_local_context(old,context,work,packets,frontier,goal,expected_prefix_actions=1)
    return admitted,work


def test_goal_bonus_requires_new_removed_cell_not_arrival_and_never_doubles():
    goal=GoalSpec((1,2,3),'suction',(0.,0.,0.),(1.,2.,3.))
    row={'action_id':'goal-named','outcome_scope':'permitted_nominal_model','removed_indices_native':[],
        'normal_removed_mm3':2.,'complete_tool_path_length_mm':10.,'tool_id':'suction'}
    value,new,_=local_reward(goal,row,'aspirator',already_succeeded=False)
    assert value==pytest.approx(-.47) and not new
    row['removed_indices_native']=[[1,2,3]]
    assert local_reward(goal,row,'aspirator',already_succeeded=False)[0]==pytest.approx(9.53)
    assert local_reward(goal,row,'aspirator',already_succeeded=True)[0]==pytest.approx(-.47)
    assert local_reward(goal,{'action_id':'STOP'},'suction',already_succeeded=True)[0]==0.


def test_real_catalog_all_feasible_and_snapshot_Gym_boundary(monkeypatch):
    context,work=fixture()
    # This generated fixture has no full-target summary variant. Supply only
    # its explicitly missing public DTO feature seam, never native state/results.
    monkeypatch.setattr(gym_adapter,'public_target_features',lambda obs,reference_mm:(np.zeros(16),np.zeros((len(obs.action_ids),4))))
    env=LocalGoalEnv(context,{1:work});before=env._stamp(work)
    obs,info=env.reset(seed=7);assert env.observation_space.contains(obs)
    catalog=env.episode.catalog
    assert catalog.record['automatic_goal_diagnostic_previews']==0
    assert int(obs['action_mask'].sum())==len(catalog.identifiers)
    assert catalog.context.record['immediate_positive_guard'] is False
    assert 'scores' not in obs and 'reward' not in obs and 'removed_indices' not in obs
    assert np.array_equal(env.action_masks(),obs['action_mask'].astype(bool))
    assert all(not key.startswith('reference') for key in obs)
    with pytest.raises(ValueError):env.step(152)
    obs,reward,done,truncated,info=env.step(0)
    assert done and not truncated and reward==0 and not info['success']
    assert env.episode.work.terminated and env.episode.packets[-1]['action_id']=='STOP'
    env.reset();assert env._stamp(work)==before and not env.episode.work.terminated
    env.close()


def test_source_goal_mutation_refused_and_eval_start_fixed(monkeypatch):
    context,work=fixture()
    monkeypatch.setattr(gym_adapter,'public_target_features',lambda obs,reference_mm:(np.zeros(16),np.zeros((len(obs.action_ids),4))))
    env=LocalGoalEnv(context,{1:work},eval_mode=True)
    with pytest.raises(ValueError):env.reset(options={'start_step':2})
    original=work._history[0]['reward'];work._history[0]['reward']=original+1
    with pytest.raises((ValueError,RuntimeError)):context.require_task(work)
    work._history[0]['reward']=original
    env.close()
