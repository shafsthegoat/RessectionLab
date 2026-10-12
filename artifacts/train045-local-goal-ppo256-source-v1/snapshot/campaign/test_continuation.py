"""Generated resume controls. No acquired checkpoint, patient payload or native call."""
from pathlib import Path
import sys,copy,hashlib,importlib.util
HERE=Path(__file__).resolve().parent;STAGE=HERE.parent
sys.path.insert(0,str(HERE))
from continuation import load_verified,exact_state,state_hash
from contract import ARMS,EXPECTED_PPO,merge_experience,require_pilot,semantic
import pytest
import torch
import gymnasium as gym
from sb3_contrib import MaskablePPO
from policy import LocalGoalPolicy,module_hash
from checkpoints import checkpoint_bytes
spec=importlib.util.spec_from_file_location('pinned_generated_local_policy_fixture',STAGE/'model/test_policy.py')
fixture_module=importlib.util.module_from_spec(spec);spec.loader.exec_module(fixture_module)


def experience(states=('s1','s2'),actions=('a1','a2'),*,decisions=3):
    return {'patients':1,'resets':2,'decisions':decisions,'completed_episodes':1,'successful_episodes':0,
        'automatic_STOP_transitions':0,'start_counts':{'12':1,'13':0,'14':0,'15':0,'16':1},
        'source_state_hashes':list(states),'distinct_source_states':len(states),
        'state_physical_action_hashes':list(actions),'distinct_state_physical_actions':len(actions),
        'unfinished_episode_decisions':1,'unfinished_executed_episodes':1,'episode_returns':[-.2],
        'completed_catalog_costs':{'catalog_calls':2,'catalog_seconds':.3},
        'live_episode_costs':{'catalog_calls':2,'catalog_seconds':.4}}


def test_cumulative_unique_unions_and_retired_unfinished_costs():
    old=experience();new=experience(('s2','s3'),('a2','a3'),decisions=5)
    result=merge_experience(old,new)
    assert result['decisions']==8 and result['completed_episodes']==2
    assert result['distinct_source_states']==result['distinct_state_physical_actions']==3
    assert result['resets']==4 and result['unfinished_executed_episodes']==1
    assert result['retired_unfinished_episodes_at_resume']==1
    assert result['retired_unfinished_decisions_at_resume']==1
    assert result['completed_catalog_costs']['catalog_calls']==4
    assert result['retired_live_episode_costs']==old['live_episode_costs']
    assert result['live_episode_costs']==new['live_episode_costs']
    assert old==experience()  # No mutation/cost relabeling of original receipt.
    malformed=copy.deepcopy(new);malformed['source_state_hashes'].append('s2')
    with pytest.raises(ValueError,match='unique'):merge_experience(old,malformed)


def pilot_fixture():
    config={k:k for k in ('goal','prefix_actions','absolute_horizon','query','curriculum_starts',
                          'evaluation_start','reward_policy','success','torch_threads')}
    config.update(milestones=[256],seeds=[0],arms=['frozen_encoder','trainable_encoder'])
    report={'status':'completed','source_unchanged':True,'qualified':True,'patients':1,
        'subject':'ReMIND-045','role':'TRAIN','source_index_sha256':'index',
        'config':{**config,'milestones':[64]},'arms':{},'evaluations':{}}
    endpoint={'complete':True,'audit':{'accepted':True},'local_reward_audit_accepted':True}
    for name in ARMS:
        row={'decisions':64,'checkpoint':{'decisions':64,'includes_optimizer_state':True,'sha256':name},
             'optimizer_completed':2,'optimizer_attempted':2,
             'policy_costs':{'optimizer_completed':2,'optimizer_attempted':2},'evaluation':endpoint}
        report['arms'][name]={'status':'completed','seed':0,'ppo_config':dict(EXPECTED_PPO),
            'milestones':[64],'checkpoints':[row],'experience':experience(decisions=64),
            'initial_evaluation':endpoint}
        for label in ('init','64'):report['evaluations'][name+'-eval-'+label]=endpoint
    parent={'status':'completed','report_sha256':'report','source_index_sha256':'index',
        'owned_lifecycle':{'verified':True,'child_reaped':True,'exit_code':0,'errors':[],
            'remaining_owned_pids':[],'remaining_state_unknown':False}}
    return report,parent,config


@pytest.mark.parametrize('change',['incomplete','unaudited','wrong_goal','wrong_optimizer'])
def test_pilot_gate_rejects_noncompleted_or_changed_protocol(change):
    report,parent,config=pilot_fixture()
    assert len(require_pilot(report,parent,report_sha='report',index_sha='index',configuration=config)['arm_checkpoint_sha256'])==2
    if change=='incomplete':parent['status']='incomplete_or_failed'
    if change=='unaudited':report['arms'][ARMS[0]]['checkpoints'][0]['evaluation']['audit']['accepted']=False
    if change=='wrong_goal':config['goal']='other'
    if change=='wrong_optimizer':report['arms'][ARMS[0]]['checkpoints'][0]['optimizer_attempted']=3
    with pytest.raises(ValueError):require_pilot(report,parent,report_sha='report',index_sha='index',configuration=config)


@pytest.mark.parametrize('frozen',[True,False])
def test_real_checkpoint_exact_optimizer_resume_and_next_update(frozen,monkeypatch):
    _,_,source,batch=fixture_module.fixture(frozen)
    class Tiny(gym.Env):
        observation_space=source.observation_space;action_space=source.action_space
        def reset(self,*,seed=None,options=None):
            super().reset(seed=seed)
            return {k:v[0].detach().numpy().copy() for k,v in batch.items()},{}
        def step(self,action):return self.reset()[0],float(action==1),True,False,{}
        def action_masks(self):return batch['action_mask'][0].bool().numpy()
    model=MaskablePPO(LocalGoalPolicy,Tiny(),seed=0,
        policy_kwargs={'freeze_encoder':frozen},**EXPECTED_PPO)
    model.policy.load_state_dict(source.state_dict(),strict=True);model.learn(64)
    prior={'decisions':64,'sb3_epoch_updates':model._n_updates,
        'optimizer_completed':model.policy.costs['optimizer_completed'],
        'policy_costs':dict(model.policy.costs),'model_hash':module_hash(model.policy.spatial),
        'encoder_hash':module_hash(model.policy.spatial.encoder)}
    raw,binding=checkpoint_bytes(model)
    restored,join=load_verified(raw,Tiny(),expected_sha256=binding['sha256'],prior=prior,frozen=frozen,seed=0)
    assert join['all_tensors_and_optimizer_scalars_exact']
    assert join['policy_hash']==join['archive_policy_hash']
    assert join['optimizer_hash']==join['archive_optimizer_hash']
    exact_state(restored.get_parameters(),model.get_parameters())
    with pytest.raises(ValueError,match='mode/seed/update'):
        load_verified(raw,Tiny(),expected_sha256=binding['sha256'],prior=prior,frozen=not frozen,seed=0)
    before=copy.deepcopy(restored.policy.optimizer.state_dict())
    restored.learn(64,reset_num_timesteps=False)
    assert restored.num_timesteps==128 and restored.policy.costs['optimizer_completed']>prior['optimizer_completed']
    assert state_hash(before)!=state_hash(restored.policy.optimizer.state_dict())
    assert (module_hash(restored.policy.spatial.encoder)==prior['encoder_hash'])==frozen
    # A real load followed by a corrupted restored Adam moment must fail before learning.
    original=MaskablePPO.load
    def corrupted(*args,**kwargs):
        actual=original(*args,**kwargs)
        next(iter(actual.policy.optimizer.state.values()))['exp_avg'].add_(1.)
        return actual
    monkeypatch.setattr(MaskablePPO,'load',corrupted)
    with pytest.raises(ValueError,match='Checkpoint tensor mismatch'):
        load_verified(raw,Tiny(),expected_sha256=binding['sha256'],prior=prior,frozen=frozen,seed=0)
    with pytest.raises(ValueError,match='root-approved'):
        load_verified(raw,Tiny(),expected_sha256='0'*64,prior=prior,frozen=frozen,seed=0)
    model.env=None;restored.env=None
