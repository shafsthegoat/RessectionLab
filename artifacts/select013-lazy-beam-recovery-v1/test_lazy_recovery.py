"""Generated wiring and partial-domain native parity only; no acquired payloads."""
from contextlib import contextmanager
import copy
import importlib.util
import json
from pathlib import Path
import sys
from types import SimpleNamespace
import pytest

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[1]
sys.path.insert(0,str(ROOT/'src'));sys.path.insert(0,str(HERE))
import comparison_contract as contract
import select_worker as worker


def layers():
    return [dict(depth=i,completed=True,model_transition_calls=1,
        negative_prefixes_evaluated=1,negative_prefixes_pruned_by_beam=0,
        opening_depth_retention=[],retained_prefix_diagnostics={
            'retained_nonterminal_prefixes':[], 'best_terminal_prefixes':[]}) for i in range(1,5)]


@pytest.mark.parametrize('failure',[None,'returned_cap','time_cap','world','prior_layer'])
def test_one_beam_owned_execute_glue(tmp_path,monkeypatch,failure):
    import torch
    from resectionlab import patient_select_inference as inference
    from resectionlab import public_patient_factory as factory
    from resectionlab import patient_planning_admission as admission
    from resectionlab import patient_planning_preflight as preflight
    from resectionlab import observed_search,planning_budget
    config={'max_candidates':120,'intermediate_opening_mm':1.,'tool_footprint_opening':True,'obstruction_opening':False}
    protocol={'updates_per_method':64,'public_target_context_variant':contract.TARGET_CONTEXT,
        'cohort_execution':{'proposal_config':config,'occupancy_condition':contract.OCCUPANCY}}
    data={'IL64':{'protocol':protocol,'checkpoint':{'path':'IL64.unused','sha256':'a'*64},
        'context_hashes':{},'method':'IL','updates':64,'parameter_hash':'sha256:'+'b'*64,
        'training_release_sha256':'c'*64,'training_evidence':{'generated':True}}}
    loaded=[];prepared=[];collected=[];replayed=[];live={}
    from resectionlab.native_proposals import NominalCavityProposalConfig
    expected={'source_hash':'generated-source','decision_model_hash':'generated-model',
        'initial_observation_hash':'generated-observation','max_steps':24,
        'proposal_hash':NominalCavityProposalConfig(**config).fingerprint,
        'occupancy_condition':contract.OCCUPANCY,'post_exposure':{'version':contract.EXPOSURE},
        'source_domains':{'source_support_unknown_voxels':1},'action_ids':['STOP'],'action_mask':[True]}
    prior={'world':expected,'partial':{'accounting':{'layers':layers()}}}
    class Budget:
        def __init__(self,*args,**kwargs):self.count=0;live['budget']=self
        def __enter__(self):return self
        def __exit__(self,*args):return False
        def check(self):pass
        def complete(self,*,history_complete):assert history_complete;live['complete']=True
        def snapshot(self):return {'native_preview_entries':self.count}
    class Costs:
        def __init__(self,budget,cap):assert cap==0;self.rows={};self.forwards=0
        def __enter__(self):return self
        def __exit__(self,*args):return False
        @contextmanager
        def scope(self,name):
            try:yield
            finally:self.rows[name]={'complete_wall_seconds':0.}
    class Source:
        source_hash='generated-source'
        def __init__(self,proposal):
            self.proposal_config=proposal;self.post_exposure=SimpleNamespace(record={'version':contract.EXPOSURE})
            self._domain_record={'source_support_unknown_voxels':1}
    class Context:
        fingerprint='generated-context'
        def record(self):return {'occupancy_condition':contract.OCCUPANCY}
    class Task:
        max_steps=24;decision_model_hash='generated-model'
        def __init__(self,case):self.case=case
        def observation(self):return SimpleNamespace(fingerprint='wrong' if failure=='world' else 'generated-observation',
            action_ids=('STOP',),action_mask=SimpleNamespace(tolist=lambda:[True]))
    class Checkpoint:
        def lineage_record(self):return {'method':'IL','completed_updates':64,'parameter_hash':'sha256:'+'b'*64}
        def require_task(self,*args):return self
        def require(self,*args):return self
    def load(path,**kw):
        assert kw['expected_method']=='IL';loaded.append('IL');return Checkpoint()
    def prepare(out,release,sha,unused,progress,**kw):
        assert loaded==['IL'] and kw['expected_role']=='SELECT'
        assert kw['occupancy_inference_protocol']==protocol
        assert kw['post_exposure_condition']==contract.EXPOSURE and kw['partial_domain_inputs'] is True
        assert '037' not in str(kw['public_manifest_path'])
        prepared.append('013');live['budget'].count+=1
        return Source(kw['proposal_config']),{},{},{}
    def search(base,**kw):
        assert kw==dict(contract.LIMITS['search'],policy=None,
            objective_source='permitted_nominal_target_and_frozen_geometric_costs',
            transition_mode='lazy_planning',retained_prefix_diagnostics=True,retention_mode=contract.MODE)
        accounting={'call_cap_reached':failure=='returned_cap','time_cap_reached':failure=='time_cap',
            'transition_mode':'lazy_planning','layers':layers()}
        if failure=='prior_layer':accounting['layers'][0]['model_transition_calls']=2
        if failure=='time_cap':raise observed_search.ObservedSearchLimit('generated time cap',accounting=accounting,best_sequence=('STOP',))
        return ('STOP',),accounting
    def collect(base,context,checkpoint,*,output,guard,actions,accounting):
        guard();assert actions==('STOP',) and accounting['transition_mode']=='lazy_planning'
        collected.append(output.name);return SimpleNamespace(transitions=[SimpleNamespace(action_id='STOP',reward=0.)])
    def replay(base,context,trace,checkpoint,*,output,guard):
        guard();replayed.append(output.name)
        return {'replayed_history':[{'target_removed_mm3':0.,'normal_removed_mm3':0.}],'plan_seal':'generated'}
    def forbidden(*a,**k):raise AssertionError('No learned or greedy execution allowed')
    monkeypatch.setattr(worker,'authenticate_comparison',lambda release:data)
    monkeypatch.setattr(worker,'authenticate_prior',lambda:prior)
    monkeypatch.setattr(worker,'read_pinned',lambda *a:{})
    monkeypatch.setattr(worker,'select_public',lambda *a:{'path':'generated013.unused','sha256':'d'*64})
    monkeypatch.setattr(planning_budget,'PlanningBudget',Budget);monkeypatch.setattr(preflight,'_CallCosts',Costs)
    monkeypatch.setattr(factory,'prepare_public_source',prepare)
    monkeypatch.setattr(admission,'make_patient_planning_task',lambda source,**kw:(Task(source),Context()))
    monkeypatch.setattr(inference,'load_select_checkpoint',load)
    monkeypatch.setattr(inference,'collect_select_greedy',forbidden)
    monkeypatch.setattr(inference,'collect_select_search',collect);monkeypatch.setattr(inference,'seal_and_replay_select',replay)
    monkeypatch.setattr(observed_search,'observed_beam_search',search)
    old_optimizer=torch.optim.Optimizer.__init__;old_grad=torch.autograd.grad;old_backward=torch.autograd.backward
    release={'public_index':{},'public_preparation':{'generated':True}}
    if failure:
        with pytest.raises(ValueError):worker.execute(release,'f'*64,tmp_path/'out',lambda *a,**k:None)
        result=json.loads((tmp_path/'out/result.json').read_text())
    else:result=worker.execute(release,'f'*64,tmp_path/'out',lambda *a,**k:None)
    assert loaded==['IL'] and prepared==['013']
    assert collected==replayed==([] if failure else ['observed_beam'])
    assert contract.complete_result(result) is (failure is None)
    assert result['total_policy_forward_calls']==0 and result['checkpoint_loads']==1
    assert result['gradient_attempts']==result['optimizer_attempts']==0
    assert result['held_cases']==['ReMIND-037'] and result['EVAL_opened'] is False
    assert result['arms'].keys()=={'observed_beam'}
    assert torch.optim.Optimizer.__init__ is old_optimizer and torch.autograd.grad is old_grad and torch.autograd.backward is old_backward
    if failure in ('returned_cap','time_cap'):
        row=result['arms']['observed_beam'];assert row['status']=='search_unresolved'
        assert row['partial_actions']==['STOP'] and 'plan_seal' not in row
        assert not (tmp_path/'out/observed_beam/selection.json').exists()
    if failure:assert live.get('complete') is not True
    else:assert live['complete'] is True


def test_prior_layer_join_allows_only_later_successor_availability():
    old=layers();p={'actions':['a'],'estimated_incremental_return':-1.,'target_removed_mm3':0.,
        'legal_nonstop_actions':None,'successor_inventory_status':'not_observed'}
    old[-1]['retained_prefix_diagnostics']['retained_nonterminal_prefixes']=[p]
    new=copy.deepcopy(old);new[-1]['retained_prefix_diagnostics']['retained_nonterminal_prefixes'][0].update(
        legal_nonstop_actions=37,successor_inventory_status='observed_during_normal_expansion')
    assert contract.compare_completed_layers({'layers':new},{'accounting':{'layers':old}})['exact']
    new[-1]['retained_prefix_diagnostics']['retained_nonterminal_prefixes'][0]['target_removed_mm3']=1.
    assert not contract.compare_completed_layers({'layers':new},{'accounting':{'layers':old}})['exact']


def test_partial_domain_post_exposure_native_transition_parity(monkeypatch):
    import numpy as np
    from dataclasses import replace
    from resectionlab.native_proposals import NominalCavityProposalConfig
    from resectionlab.native_spatial_task import NativeSpatialTask
    from resectionlab.native_spatial_evaluation import evaluate_native_spatial_episode
    spec=importlib.util.spec_from_file_location('generated_post_exposure_fixture',ROOT/'tests/test_post_exposure.py')
    fixture=importlib.util.module_from_spec(spec);spec.loader.exec_module(fixture)
    S,T,Ds=fixture.arrays();assert (~Ds).any()
    case=replace(fixture.source(S,T,Ds,proposal=True), proposal_config=NominalCavityProposalConfig(
        max_candidates=120,intermediate_opening_mm=1.,tool_footprint_opening=True))
    eager=NativeSpatialTask(case,max_steps=24).planning_clone();lazy=eager.clone()
    action=next(row['action_id'] for row in eager.candidate_inventory()['ledger']
        if row.get('feasible') and row.get('voxel')==[8,8,8])
    calls=[];preview=lazy._engine.preview_stroke
    def counted(*args,**kwargs):calls.append(1);return preview(*args,**kwargs)
    monkeypatch.setattr(lazy._engine,'preview_stroke',counted)
    for identifier in (action,'STOP'):
        before=len(calls);a=eager.step(identifier);b=lazy.advance_planning(identifier)
        assert len(calls)==before and b.observation is None
        assert (a.reward,a.terminated,a.info)==(b.reward,b.terminated,b.info)
        assert eager.metrics()==lazy.metrics() and eager._state_seal==lazy._state_seal
        for name in ('remaining_mask','removed_mask','contact_mask','connected_free_mask'):
            np.testing.assert_array_equal(getattr(eager._engine,name),getattr(lazy._engine,name))
        aobs=eager.observation();bobs=lazy.observation()
        assert aobs.fingerprint==bobs.fingerprint and eager.candidate_inventory()==lazy.candidate_inventory()
        np.testing.assert_array_equal(bobs.coverage[1],Ds)
    assert eager.metrics()['target_removed_mm3']>0 and eager.terminated
    certificates=[]
    normalized=lambda history:[{k:v for k,v in row.items() if k!='outcome_scope'} for row in history]
    for planned in (eager,lazy):
        replay=NativeSpatialTask(case,max_steps=planned.max_steps)
        for row in planned.metrics()['history']:replay.step(row['action_id'])
        assert normalized(replay.metrics()['history'])==normalized(planned.metrics()['history'])
        certificate=evaluate_native_spatial_episode(replay)
        assert certificate['accepted'] is True
        certificates.append({k:v for k,v in certificate.items() if k!='evaluation_seconds'})
    assert certificates[0]==certificates[1]
