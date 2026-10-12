"""Generated schedule, lineage, failed-ancestor and exact optimizer controls."""
from pathlib import Path
import sys,copy,importlib.util
from types import SimpleNamespace
HERE=Path(__file__).resolve().parent;STAGE=HERE.parent
sys.path.insert(0,str(HERE))
from recovery_contract import recovery_start,require_failure,complete_result
from recovery_learning import RecoveryCurriculum
from curriculum_contract import merge_experience
from gym_adapter import LocalGoalEnv
from learning import UniformCurriculum,PPO_CONFIG
from continuation import load_verified,exact_state
from checkpoints import checkpoint_bytes
from policy import LocalGoalPolicy,module_hash
import numpy as np
import pytest


def old_fixture():
    spec=importlib.util.spec_from_file_location('generated_recovery_fixtures',STAGE/'curriculum/test_curriculum.py')
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module);return module


def test_clock_starts14_and_preserves_original_windows():
    assert [recovery_start(x) for x in (0,127,128,255,256,511,512)]==[14,14,13,13,12,12,12]
    for x in (-1,513,True,0.):
        with pytest.raises(ValueError):recovery_start(x)


@pytest.mark.parametrize('boundary,next_start',[(128,13),(256,12)])
def test_live_episode_crosses_boundary_before_natural_reset(monkeypatch,boundary,next_start):
    env=object.__new__(RecoveryCurriculum);env.__dict__.update(old_fixture().bare_schedule('reverse').__dict__)
    def reset(env,*,seed=None,options=None):
        env.episode=SimpleNamespace(done=False);return {},options
    def step(env,action):
        env.counts['decisions']+=1;env.episode.done=env.counts['decisions']==boundary+1
        return {},-.03,env.episode.done,False,{'success':False,'terms':{'goal_bonus':0.}}
    monkeypatch.setattr(LocalGoalEnv,'reset',reset);monkeypatch.setattr(UniformCurriculum,'step',step)
    env.counts['decisions']=boundary-1;env.reset();previous=env._start
    env.step(0)
    assert env._start==previous and previous!=next_start and not env.episode.done
    with pytest.raises(ValueError,match='natural'):env.reset()
    env.step(0);env.reset()
    assert env._start==next_start
    assert env.reset_records[-1]=={'recovery_decisions':boundary+1,
        'original_curriculum_decisions':257+boundary,'cumulative_decisions':513+boundary,'start_step':next_start}


def test_four_generation_retirement_and_known_union_excludes_discarded_tail():
    f=old_fixture();prior=f.old_merge(f.exp(64,6,6,['a']),f.exp(192,20,6,['a','b']))
    checkpoint=merge_experience(prior,f.exp(256,37,6,['b','c']));held=copy.deepcopy(checkpoint)
    result=merge_experience(checkpoint,f.exp(512,60,2,['c','d']))
    assert result['decisions']==1024 and result['retired_unfinished_decisions_total']==18
    assert [r['ended_at_decisions'] for r in result['retired_segments']]==[64,256,512]
    assert result['completed_episode_decisions']==1004 and result['unfinished_episode_decisions']==2
    assert result['source_state_hashes']==['a','b','c','d']
    assert result['decision_partition_exact'] and result['reset_partition_exact'] and checkpoint==held
    # The320 failed postcheckpoint decisions are a paid-cost side branch, not added to this1024 lineage.


def failure_fixture():
    keys=('goal','prefix_actions','absolute_horizon','query','curriculum_starts','evaluation_start',
          'reward_policy','success','torch_threads','reverse_windows')
    cfg={k:k for k in keys};cfg.update(resume_decisions=512,additional_decisions=512,schedule_offset=256,
        arms=['reverse'],milestones=[1024],seeds=[0],caps={'worker_peak_rss_bytes':4*1024**3})
    endpoint={'complete':True,'audit':{'accepted':True},'local_reward_audit_accepted':True,'success':False}
    q={'decisions':512,'optimizer_completed':353,'optimizer_attempted':353,
        'experience':{'decisions':512,'unfinished_episode_decisions':6},'encoder_changed':False,'encoder_hash':'encoder',
        'checkpoint':{'includes_optimizer_state':True,'sha256':'18794e285ae2b7d6219fe297bcf5a6e9d060c46710b822f6c3dc2575e5fbac47'},
        'evaluation':endpoint}
    uniform={'status':'completed','shared_starting_checkpoint':{'sha256':'shared'},'ppo_config':{},
        'checkpoints':[{'decisions':1024,'evaluation':endpoint}]}
    curve={'arm':'reverse','seed':0,'status':'running','checkpoints':[q],'initialization':{'encoder_hash':'encoder'},
        'shared_starting_checkpoint':uniform['shared_starting_checkpoint'],'ppo_config':{}}
    report={'status':'running','source_index_sha256':'idx','config':cfg,'arms':{'uniform-seed0':uniform},
        'evaluations':{'uniform-seed0-eval-1024':endpoint,'reverse-seed0-eval-512':endpoint}}
    parent={'status':'incomplete_or_failed','stop_reason':'rss_cap','report_sha256':'report','source_index_sha256':'idx',
        'sampled_peak_group_rss_bytes':4*1024**3+1,'caps':{'worker_peak_rss_bytes':4*1024**3},'elapsed_seconds':10.,
        'owned_lifecycle':{'verified':True,'child_reaped':True,'exit_code':-15,'errors':[],
            'remaining_owned_pids':[],'remaining_state_unknown':False},
        'last_saved_costs':{'decisions':832,'optimizer_completed':393,'optimizer_attempted':393,
            'native_budget':{'counting_reliable':True,'native_preview_entries':9},'transitions':{'attempted':8}}}
    post={'source_index_sha256':'idx','mismatches':[],'source_files':{'x':'h'},'files_checked':1}
    return report,parent,curve,post,{'files':{'x':'h'}},cfg


def test_failed_ancestor_is_never_relabelled_complete_or_unverified():
    r,p,c,m,i,cfg=failure_fixture();pins={'report':'report','index':'idx','launcher':'parent','postmortem':'post'}
    accepted=require_failure(r,p,c,m,i,pins=pins,configuration=cfg)
    assert not accepted['original_campaign_completed'] and accepted['discarded_postcheckpoint_decisions_lower_bound']==320
    for target,key,value in ((p,'stop_reason','wall_cap'),(r,'status','completed'),(m,'mismatches',['x'])):
        old=target[key];target[key]=value
        with pytest.raises(ValueError):require_failure(r,p,c,m,i,pins=pins,configuration=cfg)
        target[key]=old
    assert not complete_result(r)


def test_real_generated_checkpoint_Adam_restore_then_unchanged_PPO():
    import gymnasium as gym
    import torch
    from sb3_contrib import MaskablePPO
    spec=importlib.util.spec_from_file_location('recovery_tensor_fixture',STAGE/'model/test_policy.py')
    fixture=importlib.util.module_from_spec(spec);spec.loader.exec_module(fixture)
    _,_,source,batch=fixture.fixture(True)
    class Tiny(gym.Env):
        observation_space=source.observation_space;action_space=source.action_space
        def reset(self,*,seed=None,options=None):
            super().reset(seed=seed);return {k:v[0].detach().numpy().copy() for k,v in batch.items()},{}
        def step(self,a):return self.reset()[0],float(a==1),True,False,{}
        def action_masks(self):return batch['action_mask'][0].bool().numpy()
    model=MaskablePPO(LocalGoalPolicy,Tiny(),seed=0,policy_kwargs={'freeze_encoder':True},**PPO_CONFIG)
    model.policy.load_state_dict(source.state_dict(),strict=True);model.learn(64)
    prior={'decisions':64,'sb3_epoch_updates':model._n_updates,'policy_costs':dict(model.policy.costs),
        'optimizer_completed':model.policy.costs['optimizer_completed'],'model_hash':module_hash(model.policy.spatial),
        'encoder_hash':module_hash(model.policy.spatial.encoder)}
    raw,pin=checkpoint_bytes(model);resumed,join=load_verified(raw,Tiny(),expected_sha256=pin['sha256'],prior=prior,frozen=True,seed=0)
    assert join['all_tensors_and_optimizer_scalars_exact'];exact_state(model.get_parameters(),resumed.get_parameters())
    resumed.learn(64,reset_num_timesteps=False)
    assert resumed.num_timesteps==128 and resumed.policy.costs['optimizer_completed']>prior['optimizer_completed']
    assert module_hash(resumed.policy.spatial.encoder)==prior['encoder_hash']


def test_compiled_campaign_captures_historical_reference_without_shadowing():
    from types import CodeType
    module=compile((HERE/'train_worker.py').read_bytes(),str(HERE/'train_worker.py'),'exec')
    execute=next(c for c in module.co_consts if isinstance(c,CodeType) and c.co_name=='execute')
    campaign=next(c for c in execute.co_consts if isinstance(c,CodeType) and c.co_name=='campaign')
    assert 'historical' in campaign.co_freevars and 'historical' not in campaign.co_varnames
    assert 'historical_experience' in campaign.co_varnames
    assert 'historical' in execute.co_cellvars


def test_exact_prior_startup_failure_costs_and_zero_PPO_admission():
    from recovery_contract import require_startup_failure
    pins={k:{'path':k,'sha256':k+'-hash'} for k in ('report','launcher','source_index')}
    report={'status':'failed_or_incomplete','source_unchanged':True,'source_index_sha256':'source_index-hash',
        'arms':{},'evaluations':{},'checkpoint_loads':1,'worker_seconds':46.578557417029515,
        'failure':{'phase':'fresh_prefix_and_genuine_curriculum_reconstruction','type':'UnboundLocalError',
            'message':"cannot access local variable 'historical'"},
        'native_budget':{'counting_reliable':True,'native_preview_entries':2484},
        'native_transition_counts':{'attempted':25,'completed':25,'raised':0}}
    parent={'status':'incomplete_or_failed','source_index_sha256':'source_index-hash','report_sha256':'report-hash',
        'elapsed_seconds':46.970555792097,'owned_lifecycle':{'verified':True,'child_reaped':True,'exit_code':1,
            'errors':[],'remaining_owned_pids':[],'remaining_state_unknown':False}}
    cost=require_startup_failure(report,parent,pins)
    assert cost['PPO_decisions']==cost['PPO_updates']==0 and cost['IL64_loads']==1
    assert cost['worker_seconds']==46.578557417029515 and cost['parent_seconds']==46.970555792097
    assert cost['native_previews']==2484 and cost['native_transitions']==25
    for target,key,value in ((report,'arms',{'reverse-seed0':{}}),(report,'source_unchanged',False),
            (parent,'status','completed'),(report,'checkpoint_loads',2)):
        before=target[key];target[key]=value
        with pytest.raises(ValueError,match='zero-PPO'):require_startup_failure(report,parent,pins)
        target[key]=before
