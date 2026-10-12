"""Non-native generated controls; no patient files or learned checkpoint inputs."""
from pathlib import Path
import sys,copy,random,importlib.util
from types import SimpleNamespace
HERE=Path(__file__).resolve().parent;STAGE=HERE.parent
sys.path.insert(0,str(HERE))
from curriculum_contract import reverse_start,merge_experience,require_campaign,EXPECTED_PPO,WINDOWS
from curriculum_learning import ScheduledCurriculum,isolated_evaluation
from learning import UniformCurriculum
from gym_adapter import LocalGoalEnv
from continuation import load_verified,exact_state
from contract import merge_experience as old_merge
from policy import LocalGoalPolicy,module_hash
from checkpoints import checkpoint_bytes
import numpy as np
import torch
import pytest
import gymnasium as gym


def bare_schedule(method):
    env=object.__new__(ScheduledCurriculum);env.method=method;env.curriculum_rng=np.random.default_rng(0)
    env.counts={'decisions':0,'resets':0};env.starts={str(i):0 for i in range(12,17)}
    env.decisions_by_start=dict(env.starts);env.successes_by_start=dict(env.starts)
    env.bonus_events=0;env.reset_records=[];env.episode=None;env._start=None
    return env


@pytest.mark.parametrize('boundary,next_start',[(128,15),(256,14),(384,13),(512,12)])
def test_natural_reset_schedule_does_not_truncate_live_episode(monkeypatch,boundary,next_start):
    def reset(env,*,seed=None,options=None):
        env.episode=SimpleNamespace(done=False)
        return {'start':options['start_step']},options
    def step(env,action):
        env.counts['decisions']+=1
        env.episode.done=env.counts['decisions']==boundary+1
        return {},-.03,env.episode.done,False,{'success':False,'terms':{'goal_bonus':0.}}
    monkeypatch.setattr(LocalGoalEnv,'reset',reset);monkeypatch.setattr(UniformCurriculum,'step',step)
    env=bare_schedule('reverse');env.counts['decisions']=boundary-1
    env.reset();previous=env._start
    assert previous!=next_start and len(env.reset_records)==1
    env.step(0)
    assert env.counts['decisions']==boundary and env._start==previous and not env.episode.done
    with pytest.raises(ValueError,match='natural'):env.reset()
    env.step(0)
    assert env.episode.done and env._start==previous and env.decisions_by_start[str(previous)]==2
    env.reset()
    assert env._start==next_start and env.reset_records[-1]['local_decisions']==boundary+1
    assert len(env.reset_records)==2


def test_uniform_seed_and_fixed_schedule_boundaries(monkeypatch):
    monkeypatch.setattr(LocalGoalEnv,'reset',lambda env,**kwargs:({},kwargs))
    env=bare_schedule('uniform');expected=np.random.default_rng(0).choice((12,13,14,15,16),20).tolist()
    actual=[]
    for i in range(20):env.reset(seed=0 if i==0 else None);actual.append(env._start)
    assert actual==expected
    assert [reverse_start(x) for x in (0,127,128,255,256,383,384,511,512,767,768)]==[16,16,15,15,14,14,13,13,12,12,12]
    with pytest.raises(ValueError):reverse_start(769)


def exp(decisions,complete,live,states):
    return {'patients':1,'resets':complete+1,'decisions':decisions,'completed_episodes':complete,
        'successful_episodes':0,'automatic_STOP_transitions':0,'start_counts':{'12':0,'13':0,'14':0,'15':0,'16':complete+1},
        'source_state_hashes':list(states),'distinct_source_states':len(states),
        'state_physical_action_hashes':list(states),'distinct_state_physical_actions':len(states),
        'unfinished_episode_decisions':live,'unfinished_executed_episodes':int(live>0),
        'episode_returns':[-.2]*complete,'completed_catalog_costs':{'catalog_calls':decisions-live},
        'live_episode_costs':{'catalog_calls':live+1}}


def test_three_generation_retired_decisions_costs_and_unique_unions():
    pilot=exp(64,6,6,['s0']);old_segment=exp(192,20,2,['s0','s1'])
    prior=old_merge(pilot,old_segment);held=copy.deepcopy(prior)
    for decisions,complete,live in ((256,26,3),(768,80,5)):
        current=exp(decisions,complete,live,['s1','s2'])
        result=merge_experience(prior,current)
        assert result['decisions']==256+decisions and result['distinct_source_states']==3
        assert [x['ended_at_decisions'] for x in result['retired_segments']]==[64,256]
        assert result['retired_unfinished_decisions_total']==8
        assert result['retired_unfinished_episodes_total']==2
        assert result['completed_episode_decisions']==248+decisions-live
        assert result['unfinished_episode_decisions']==live and result['retired_catalog_costs']['catalog_calls']==10
        assert result['decision_partition_exact'] and result['reset_partition_exact']
    assert prior==held
    wrong=copy.deepcopy(prior);wrong['completed_episode_decisions']+=1
    with pytest.raises(ValueError,match='partition'):merge_experience(wrong,exp(768,80,5,['s3']))


def test_real_checkpoint_continuation_and_intermediate_eval_rng_isolation():
    spec=importlib.util.spec_from_file_location('generated_curriculum_policy_fixture',STAGE/'model/test_policy.py')
    fixture=importlib.util.module_from_spec(spec);spec.loader.exec_module(fixture)
    _,_,source,batch=fixture.fixture(True)
    from sb3_contrib import MaskablePPO
    class Tiny(gym.Env):
        observation_space=source.observation_space;action_space=source.action_space
        def __init__(self):self.decisions=0;self.resets=0;self.episode=None
        def reset(self,*,seed=None,options=None):
            super().reset(seed=seed);self.resets+=1
            return {k:v[0].detach().numpy().copy() for k,v in batch.items()},{}
        def step(self,action):
            self.decisions+=1
            return {k:v[0].detach().numpy().copy() for k,v in batch.items()},float(action==1),True,False,{}
        def action_masks(self):return batch['action_mask'][0].bool().numpy()
        def summary(self):return {'decisions':self.decisions,'resets':self.resets}
    original=MaskablePPO(LocalGoalPolicy,Tiny(),seed=0,policy_kwargs={'freeze_encoder':True},**EXPECTED_PPO)
    original.policy.load_state_dict(source.state_dict(),strict=True);original.learn(64)
    prior={'decisions':64,'sb3_epoch_updates':original._n_updates,'policy_costs':dict(original.policy.costs),
        'optimizer_completed':original.policy.costs['optimizer_completed'],
        'model_hash':module_hash(original.policy.spatial),'encoder_hash':module_hash(original.policy.spatial.encoder)}
    raw,pin=checkpoint_bytes(original);final=[]
    for with_eval in (False,True):
        env=Tiny();model,join=load_verified(raw,env,expected_sha256=pin['sha256'],prior=prior,frozen=True,seed=0)
        assert join['all_tensors_and_optimizer_scalars_exact']
        model.learn(64,reset_num_timesteps=False)
        if with_eval:
            def evaluate(actor,label):
                random.random();np.random.random(3);torch.rand(3)
                actor.predict({k:v[0].detach().numpy().copy() for k,v in batch.items()},
                    deterministic=True,action_masks=env.action_masks())
                return {'complete':True,'label':label}
            _,isolation=isolated_evaluation(model,env,evaluate,'generated-middle')
            assert isolation['rng_restored'] and isolation['unchanged_training_state']
        model.learn(64,reset_num_timesteps=False)
        final.append(copy.deepcopy(model.get_parameters()))
        assert model.num_timesteps==192 and module_hash(model.policy.spatial.encoder)==prior['encoder_hash']
        model.env=None
    exact_state(final[0],final[1])  # Real next updates identical despite diagnostic RNG consumption.


def test_conditional_release_refuses_any_successful256_endpoint():
    config={k:k for k in ('goal','prefix_actions','absolute_horizon','query','curriculum_starts',
        'evaluation_start','reward_policy','success','torch_threads')}
    config.update(milestones=[512,1024],seeds=[0],arms=['uniform','reverse'],reverse_windows=[list(x) for x in WINDOWS])
    report={'status':'completed','source_unchanged':True,'qualified':True,'patients':1,'subject':'ReMIND-045',
        'role':'TRAIN','source_index_sha256':'index','config':{**config,'milestones':[256]},'arms':{},'evaluations':{}}
    for name in ('frozen_encoder-seed0','trainable_encoder-seed0'):
        evaluation={'complete':True,'success':False,'audit':{'accepted':True},'local_reward_audit_accepted':True}
        row={'decisions':256,'checkpoint':{'decisions':256,'includes_optimizer_state':True,'sha256':'same'},
            'optimizer_completed':158,'optimizer_attempted':158,'additional_optimizer_completed':146,
            'policy_costs':{'optimizer_completed':158},'evaluation':evaluation,'encoder_hash':'frozen','encoder_changed':False}
        report['arms'][name]={'status':'completed','seed':0,'ppo_config':dict(EXPECTED_PPO),'milestones':[256],
            'experience':{'decisions':256},'segment_experience':{'decisions':192},'checkpoints':[row],
            'optimizer_resume_join':{'all_tensors_and_optimizer_scalars_exact':True},'initialization':{'encoder_hash':'frozen'}}
        report['evaluations'][name+'-eval-256']=evaluation
    parent={'status':'completed','report_sha256':'report','source_index_sha256':'index','owned_lifecycle':{
        'verified':True,'child_reaped':True,'exit_code':0,'errors':[],'remaining_owned_pids':[],'remaining_state_unknown':False}}
    require_campaign(report,parent,report_sha='report',index_sha='index',configuration=config)
    report['arms']['trainable_encoder-seed0']['checkpoints'][0]['evaluation']['success']=True
    with pytest.raises(ValueError,match='release closed'):
        require_campaign(report,parent,report_sha='report',index_sha='index',configuration=config)
