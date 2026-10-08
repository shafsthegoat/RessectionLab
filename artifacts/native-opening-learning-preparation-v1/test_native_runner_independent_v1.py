"""Focused generated runner controls, no patient or optimization study."""
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
import json
import sys
import time
import numpy as np
import pytest
import torch

@pytest.fixture(scope='module')
def runner():
    sys.path.insert(0,str(Path.cwd()/'scripts'))
    import run_native_opening_learning as r
    torch.set_num_threads(1)
    return r

@pytest.fixture
def base():
    from resectionlab.native_spatial_task import make_native_opening_task
    return make_native_opening_task()

@pytest.fixture
def model():
    from resectionlab.spatial_policy import SpatialPolicy,SpatialPolicyConfig
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(11)
        return SpatialPolicy(SpatialPolicyConfig(critic_candidate_context=True))

def test_full_source_inventory_required_and_desktop_excluded(runner):
    declared=runner.declaration()
    assert len(declared['source_sha256'])==31
    assert 'src/resectionlab/desktop_bridge.py' not in declared['source_sha256']
    runner.validate(declared)
    del declared['source_sha256']['src/resectionlab/native_spatial_task.py']
    with pytest.raises(ValueError,match='Source closure omitted'):
        runner.validate(declared)

def test_foreign_runtime_module_origin_is_refused(runner,monkeypatch):
    declared=runner.declaration()
    monkeypatch.setitem(sys.modules,'resectionlab.unregistered_review_control',SimpleNamespace(__file__='/tmp/unregistered_review_control.py'))
    with pytest.raises(ValueError,match='Executing undeclared local source'):
        runner.imported_sources(declared)

def test_oracle_complete_tree_and_hidden_reference_invariance(runner,base):
    from resectionlab.native_spatial_task import NativeSpatialTask
    from resectionlab.spatial_policy import parameter_hash
    samples,sequence,report=runner.exact_teacher(base,check=lambda:None)
    root=base.planning_clone(); queue=[(root,(),0.)]; terminal=[]; calls=0; seen={}
    while queue:
        task,path,score=queue.pop(0)
        if task.terminated:
            terminal.append((score,path));continue
        obs=task.observation();seen[obs.fingerprint]=obs.action_ids
        for action in obs.action_ids:
            child=task.clone();step=child.step(action);calls+=1
            queue.append((child,path+(action,),score+step.reward))
    assert report['complete'] and report['model_transition_calls']==calls
    assert report['terminal_sequences']==len(terminal)
    assert report['unique_supervised_states']==len(seen)==len(samples)
    assert {r['observation_hash']:tuple(r['action_ids']) for r in report['state_rows']}==seen
    assert report['nominal_optimum']==pytest.approx(max(x[0] for x in terminal))==pytest.approx(1.1)
    other=NativeSpatialTask(replace(base.case,reference_target=np.ones(base.case.observed_support.shape)),max_steps=2)
    samples2,sequence2,report2=runner.exact_teacher(other,check=lambda:None)
    assert sequence2==sequence and report2==report
    assert [x[0].fingerprint for x in samples2]==[x[0].fingerprint for x in samples]
    a,b=base.clone(),other.clone()
    rewards_a=[a.step(x).reward for x in sequence]
    rewards_b=[b.step(x).reward for x in sequence]
    assert rewards_a!=rewards_b and rewards_a[0]<0<rewards_a[1]
    assert base.metrics()['steps']==0

def test_online_includes_planner_and_json_terminal_is_separate(runner,base,model,tmp_path):
    from resectionlab.spatial_policy import parameter_hash
    before=parameter_hash(model)
    def plan(check):
        check();time.sleep(.025);check();return ('STOP',),{'declared_dummy_planning_delay_seconds':.025}
    ep,report,_=runner.guarded_episode(base,model,tmp_path,'stop-cost',lambda:None,mode='sequence',planner=plan)
    cost=json.loads((tmp_path/'stop-cost-cost.json').read_text())
    terminal=json.loads((tmp_path/'stop-cost-terminal.json').read_text())
    assert report['guarded_online_seconds']==cost['online']['elapsed_seconds']
    assert report['guarded_online_seconds']>=report['online_seconds']+.015
    assert terminal['status']=='awaiting_independent_check'
    assert report['status']=='complete' and report['independent_evaluation']['accepted']
    assert parameter_hash(model)==before and ep[-1].terminated

def test_preview_exhaustion_retains_committed_negative_opening(runner,base,model,tmp_path):
    from resectionlab.native_axis_simulation import CommittedTransitionInterrupted
    action=next(x['action_id'] for x in base.candidate_inventory()['ledger'] if x['feasible'] and x['tool_id']=='short-wide-opener' and x['voxel']==[4,4,1])
    with pytest.raises(CommittedTransitionInterrupted):
        runner.guarded_episode(base,model,tmp_path,'cap',lambda:None,mode='sequence',planner=lambda c:((action,'STOP'),{}),previews=0)
    ep=json.loads((tmp_path/'cap.json').read_text());cost=json.loads((tmp_path/'cap-cost.json').read_text())
    assert ep['status']=='failed' and ep['committed_transitions']==1
    assert ep['decisions'][0]['status']=='committed_unreturned'
    assert ep['failure_metrics']['history'][0]['reward']<0
    assert cost['outcomes'] is None and cost['online']['failure'] is not None
    assert base.metrics()['steps']==0

def test_preparation_failure_preserves_all_nine_null_arms(runner,monkeypatch,tmp_path):
    from resectionlab import native_spatial_task
    monkeypatch.setattr(runner,'validate',lambda record:None)
    def fail(**kwargs):raise ValueError('independent preparation failure')
    monkeypatch.setattr(native_spatial_task,'make_native_opening_task',fail)
    with pytest.raises(ValueError,match='independent preparation'):
        runner.worker(runner.specification(),tmp_path,declaration_sha256='a'*64)
    report=json.loads((tmp_path/'result.json').read_text())
    assert len(report['methods'])==9 and set(report['methods'])==set(runner.METHODS)
    assert all(x['outcomes'] is None and x['status']=='not_started' for x in report['methods'].values())

def test_profile_audit_refusal_happens_before_any_gradient(runner,monkeypatch,tmp_path):
    from resectionlab import native_spatial_evaluation,spatial_policy
    monkeypatch.setattr(runner,'validate',lambda record:None)
    monkeypatch.setattr(native_spatial_evaluation,'evaluate_native_spatial_episode',lambda *args,**kwargs:{'accepted':False,'outcomes':None,'geometry':{'failures':['independent_test_refusal']}})
    attempted=[]
    def forbidden(*args,**kwargs):
        attempted.append(True)
        raise AssertionError('Failed independent audit reached gradient')
    monkeypatch.setattr(spatial_policy,'gradient_step',forbidden)
    with pytest.raises(ValueError,match='audit|evaluat|accept|geometry'):
        runner.worker(runner.specification(),tmp_path,declaration_sha256='a'*64,profile=True)
    assert not attempted
