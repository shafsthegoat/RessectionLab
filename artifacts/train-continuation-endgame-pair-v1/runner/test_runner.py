"""Compact generated/metadata controls. No patient arrays, weights or native task."""
import ast
import copy
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import pytest

HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE))
from common import ROOT,sha,read_json,TAIL_RUN,TAIL_PINS,pair_declaration,diagnose_pair
sys.path.insert(0,str(ROOT/'src'))


def load_entry(name):
    spec=importlib.util.spec_from_file_location('pair_'+name,HERE/(name+'.py'))
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module);return module


def test_saved_declaration_has_only_prefix_and_current_inventory():
    report=read_json(ROOT/TAIL_RUN/'report.json',TAIL_PINS['report.json'])
    config=json.loads((HERE/'config.json').read_text())
    prefix,common=pair_declaration(report['cases']['ReMIND-045']['TAIL'],config)
    assert len(prefix)==8 and common['step']==8
    assert set(common)=={'step','source_state_hash','observation_hash','scores','current_tool'}
    adapter=load_entry('endgame_planner')
    for action in config['forced_actions'].values():adapter.validate_treatment(action,prefix,common)
    changed=copy.deepcopy(common);changed['future_actions']=['STOP']
    with pytest.raises(ValueError):adapter.validate_treatment(config['forced_actions']['RETAINED_TAIL'],prefix,changed)
    # Stable ORIGINAL order, including the retained exact immediate-reward tie.
    positive=[r for r in common['scores'] if r['reward']>0]
    assert max(positive,key=lambda r:r['reward'])['action_id']==config['forced_actions']['IMMEDIATE_BEST']


def fake_adapter(monkeypatch,stop_at=10):
    """Exercise the actual adapted loop with tiny certified-interface stand-ins."""
    import numpy as np
    module=load_entry('endgame_planner')
    class Preview:
        feasible=True
        def __init__(self,identifier,step):
            self.identifier=identifier;self.source_state_hash='s'+str(step);self.decision_model_hash='model'
        def to_history_record(self):return {'action_id':self.identifier}
    class Engine:
        def __init__(self):
            self.state_hash='s0';self.config=SimpleNamespace(fingerprint='model')
            self._preview_records={};self._preview_digests={}
        def _result_digest(self,p):return p.identifier
        def preview_stroke(self,identifier,step):
            p=Preview(identifier,step);self._preview_records[id(p)]=p;self._preview_digests[id(p)]=p.identifier;return p
    class Observation:
        def __init__(self,ids,step):
            self.action_ids=ids;self.action_mask=np.ones(len(ids),dtype=bool);self.fingerprint='o'+str(step)
        def assert_intact(self):pass
    class Task:
        def __init__(self):
            self.max_steps=24;self._steps=0;self._history=[];self.terminated=False;self._planning=False
            self._engine=Engine();self._inventory=None;self._ledger=();self._current_tool=None;self._total_reward=0.
            self._source_hash='source';self.decision_model_hash='model'
        def _check_cancelled(self):pass
        def planning_clone(self):
            other=Task();other._planning=True;return other
        def _prepare_inventory(self):
            if self._inventory is None:
                ids=() if self._steps>=stop_at else ('a','b')
                self._inventory={a:self._engine.preview_stroke(a,self._steps) for a in ids}
                self._ledger=tuple({'tool_id':'tool','family':'generated','proposal_reason':'PROPOSED_UNCERTIFIED',
                    'reason':'accepted','feasible':True} for a in ids)
            return self._inventory
        def _score_record(self,record,current_tool):
            identifier=record['action_id']
            return {'action_id':identifier,'reward':1. if identifier=='a' else 2.,'target_removed_mm3':1.,
                'normal_removed_mm3':0.,'insertion_distance_mm':1.,'complete_tool_path_length_mm':1.,
                'outcome_scope':'permitted_nominal_model'}
        def observation(self):return Observation(('STOP',*self._prepare_inventory()),self._steps)
        def advance_planning(self,action):
            reward=0. if action=='STOP' else self._score_record({'action_id':action},self._current_tool)['reward']
            self._steps+=1;self._engine.state_hash='s'+str(self._steps)
            self.terminated=action=='STOP' or self._steps==24
            self._total_reward+=reward;self._history.append({'action_id':action,'reward':reward})
            if action!='STOP':self._current_tool='tool'
            self._inventory=None;self._ledger=()
            return SimpleNamespace(reward=reward,info={'action_id':action,'outcome_scope':'permitted_nominal_model'})
    class Context:
        fingerprint='context'
        def require_task(self,task):pass
        def require_observations(self,observations):pass
    class Scorer:
        def __init__(self):
            self.costs={'forwards':0};self.report={'base_parameter_hash':'base','fitted_head_hash':'head'}
        def require(self):pass
        def bind_context(self,context):pass
        def score_tails(self,observation,context):self.costs['forwards']+=1;return (0.,3.,0.)
    monkeypatch.setattr(module,'NativeSpatialTask',Task)
    monkeypatch.setattr(module,'PatientPlanningContext',Context)
    monkeypatch.setattr(module,'FittedContinuationTail',Scorer)
    prefix=[{'selected_action_id':'a','observation_hash':'o'+str(i),'source_state_hash':'s'+str(i),
        'result_state_hash':'s'+str(i+1),'actual_reward':1.,'terminated':False,
        'current_tool_before':None if i==0 else 'tool'} for i in range(8)]
    scores=[]
    for action,reward,tail in (('STOP',0.,0.),('a',1.,3.),('b',2.,0.)):
        scores.append({'action_id':action,'reward':reward,'target_removed_mm3':0. if action=='STOP' else 1.,
            'normal_removed_mm3':0.,'insertion_distance_mm':0. if action=='STOP' else 1.,
            'complete_tool_path_length_mm':0. if action=='STOP' else 1.,'predicted_q':reward+tail,
            'actual_positive_guard':action!='STOP','raw_predicted_tail':tail,'clipped_predicted_tail':tail,
            'combined_nominal_q':reward+tail})
    common={'step':8,'source_state_hash':'s8','observation_hash':'o8','scores':scores,'current_tool':'tool'}
    return module,Task,Context,Scorer,prefix,common


@pytest.mark.parametrize('stop_at',[10,30])
def test_actual_adapter_single_intervention_and_absolute_horizon(monkeypatch,stop_at):
    module,Task,Context,Scorer,prefix,common=fake_adapter(monkeypatch,stop_at)
    plans=[]
    for forced in ('a','b'):
        task=Task();actions,accounting=module.endgame_tail_search(task,Context(),Scorer(),
            forced_action=forced,expected_prefix=prefix,expected_common=common,seconds=10.)
        assert accounting['complete'] and task._steps==0
        assert actions[:8]==('a',)*8 and actions[8]==forced
        assert all(a=='a' for a in actions[9:] if a!='STOP')
        assert len(actions)==(11 if stop_at==10 else 24)
        assert sum(d.get('intervention',False) for d in accounting['decisions'])==1
        assert accounting['model_transition_calls']==len(actions)==accounting['committed_transitions']
        assert accounting['native_previews_started']==accounting['native_previews_completed']==2*min(stop_at,24)
        assert accounting['terminal_decision_catalog']['step']==len(actions)-1
        assert accounting['terminal_decision_catalog']['additional_queries']==0
        plans.append(accounting)
    assert plans[0]['decisions'][:8]==plans[1]['decisions'][:8]
    assert plans[1]['estimated_incremental_return']-plans[0]['estimated_incremental_return']==1.


@pytest.mark.parametrize('mutation',['prefix','inventory','budget'])
def test_adapter_failure_retains_costs_without_accepted_plan(monkeypatch,mutation):
    module,Task,Context,Scorer,prefix,common=fake_adapter(monkeypatch)
    calls=24
    if mutation=='prefix':prefix[3]['result_state_hash']='wrong'
    if mutation=='inventory':common['scores'][1]['reward']=999.
    if mutation=='budget':calls=8
    with pytest.raises(Exception) as caught:
        module.endgame_tail_search(Task(),Context(),Scorer(),forced_action='b',expected_prefix=prefix,
            expected_common=common,seconds=10.,max_calls=calls)
    accounting=caught.value.accounting
    assert not accounting['complete'] and accounting['selected_actions'] is None
    assert accounting['committed_transitions']>0 and accounting['native_previews_started']>0
    assert accounting['terminal_decision_catalog'] is None


def test_catalog_aggregation_is_output_only_and_count_checked(monkeypatch):
    module,*_=fake_adapter(monkeypatch)
    ledger=[{'tool_id':'a','family':'ray','proposal_reason':'PROPOSED_UNCERTIFIED','reason':'shaft_blocked','feasible':False},
        {'tool_id':'b','family':'ray','proposal_reason':'PROPOSED_UNCERTIFIED','reason':'accepted','feasible':True}]
    result=module.aggregate_catalog(ledger,1,state_hash='s',step=23,selected_action='x')
    assert result['ledger_slots']==result['proposed_count']==2 and result['certified_count']==1
    assert result['additional_queries']==0 and len(result['groups'])==2
    with pytest.raises(ValueError):module.aggregate_catalog(ledger,2,state_hash='s',step=23,selected_action='x')


def test_saved_pair_return_decomposition_and_negative_admissibility():
    metrics={'target_removed_mm3':1.,'normal_removed_mm3':0.,'cumulative_contacted_tissue_upper_bound_mm3':1.,
        'total_reward':11.,'complete_tool_path_length_mm':2.,'nonstop_actions':9,'tool_changes':0}
    base={'complete':True,'audit':{'accepted':True,'outcomes':metrics},'world':{'same':True},
        'actions':['p']*8+['a','STOP'],'decisions':[{'reward':1.}]*8+[{'reward':3.},{'reward':0.}]}
    branches={'RETAINED_TAIL':base,'IMMEDIATE_BEST':copy.deepcopy(base)}
    branches['IMMEDIATE_BEST']['actions'][8]='b';branches['IMMEDIATE_BEST']['decisions'][8]['reward']=4.
    branches['IMMEDIATE_BEST']['audit']['outcomes']['total_reward']=12.
    common={'scores':[{'action_id':'a','reward':3.,'clipped_predicted_tail':3.,'combined_nominal_q':6.},
                      {'action_id':'b','reward':4.,'clipped_predicted_tail':1.,'combined_nominal_q':5.}]}
    result=diagnose_pair(branches,common)
    assert result['prefix_return']==8 and result['realized_remaining_return_difference']==-1.
    assert result['predicted_remaining_Q_difference']==1. and result['prediction_difference_error']==2.
    assert result['comparison_admissible'] and not result['deployable_improvement_claim']
    branches['IMMEDIATE_BEST']['complete']=False
    assert not diagnose_pair(branches,common)['comparison_admissible']


def test_source_order_seals_both_plans_before_replay_and_keeps_old_files():
    for path in HERE.glob('*.py'):ast.parse(path.read_text(),filename=str(path))
    source=(HERE/'worker.py').read_text()
    assert source.index("sink.write(output/'plans.json'")<source.index("with at('replay.'+method)")
    assert 'ContinuationCorpusBuilder' not in source and 'save_continuation_cache' not in source
    assert source.count('write_logged_targets(sink,output,config')==1
    assert 'expected_prefix=expected_prefix' in source and 'expected_common=expected_common' in source
    cfg=json.loads((HERE/'config.json').read_text())
    assert cfg['caps']['work_seconds']==600 and cfg['caps']['parent_wall_seconds']==630
    assert cfg['caps']['native_previews']==30000 and cfg['caps']['frozen_policy_forwards']==128


def test_false_release_parent_and_worker_do_not_launch(tmp_path,monkeypatch):
    stage=tmp_path/'runner';stage.mkdir()
    (stage/'source-index.json').write_text(json.dumps({'files':{},'execution_released':False}))
    (stage/'config.json').write_text((HERE/'config.json').read_text())
    parent=load_entry('run_owned');monkeypatch.setattr(parent,'HERE',stage);monkeypatch.setattr(parent,'ROOT',tmp_path)
    monkeypatch.setattr(parent.subprocess,'Popen',lambda *a,**k:pytest.fail('No actual child'))
    args=['run_owned','--source-index-sha256',sha(stage/'source-index.json')]
    monkeypatch.setattr(sys,'argv',args);assert parent.main()==0
    monkeypatch.setattr(sys,'argv',args+['--execute'])
    with pytest.raises(ValueError,match='not released'):parent.main()
    worker=load_entry('worker');monkeypatch.setattr(worker,'HERE',stage);monkeypatch.setattr(worker,'ROOT',tmp_path)
    monkeypatch.setattr(worker,'execute',lambda *a,**k:pytest.fail('No scientific execute'))
    monkeypatch.setattr(sys,'argv',['worker','--source-index-sha256',sha(stage/'source-index.json'),'--output',str(tmp_path/'run')])
    with pytest.raises(ValueError,match='root-released'):worker.main()


def test_actual_isolated_bootstrap_under_boundary(tmp_path):
    output=tmp_path/'bootstrap';cache=tmp_path/'absent-source-cache'
    env={k:v for k,v in os.environ.items() if not k.startswith(('PYTHON','DYLD_','LD_'))}
    env.update({k:'1' for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','VECLIB_MAXIMUM_THREADS','NUMEXPR_NUM_THREADS')})
    env['TMPDIR']=str(tmp_path)
    result=subprocess.run([sys.executable,'-I','-B','-X','pycache_prefix='+str(cache),str(HERE/'bootstrap_smoke.py'),
        '--output',str(output)],capture_output=True,text=True,timeout=28,env=env)
    assert result.returncode==0,result.stderr
    receipt=json.loads((output/'receipt.json').read_text())
    assert receipt['status']=='passed' and receipt['payload_open_attempts']==[]
    assert receipt['native_calls']==receipt['policy_forwards']==receipt['optimizer_steps']==0
