"""Generated causal geometry and actual owned-worker wiring; no acquired inputs."""
from contextlib import contextmanager
from dataclasses import replace
import importlib.util
import json
from pathlib import Path
import sys
from types import SimpleNamespace
import numpy as np
import pytest
HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[1]
sys.path.insert(0,str(ROOT/'src'));sys.path.insert(0,str(HERE))
sys.path.append(str(ROOT/'build/select013-dead-frontier-witness-v1'))
import comparison_contract as contract
import frontier_compare as helper
import select_worker as worker
from blocker_coverage import inspect_rejected
from resectionlab.core import freeze_json,semantic_digest
from resectionlab.native_proposals import NominalCavityProposalConfig,DEFAULT_COLUMN_OFFSETS,OBSTRUCTION_OPENING_FAMILY
from resectionlab.native_spatial_task import NativeSpatialTask


def test_real_admitted_dataclass_shape_keeps_authority_out_of_physical_serialization(monkeypatch):
    """No arrays or canonical capability construction: explicit generated double.

    Copy the actual canonical constructor field names, including the opaque
    authority field that the first acquired attempt reached. The double is
    admitted only by this test's local type/token injection, never by production.
    """
    from dataclasses import fields,make_dataclass
    from resectionlab.post_exposure import PostExposureStart
    names=[f.name for f in fields(PostExposureStart) if f.init]
    assert names==['record','external_workspace','seed','initial_free','_capability']
    Generated=make_dataclass('GeneratedPostExposure',[(name,object) for name in names],frozen=True)
    token=object();values=({'scope':'generated metadata'},'external','seed','free',token)
    generated=Generated(*values)
    old_spec=importlib.util.spec_from_file_location('preserved_v1_frontier',ROOT/'build/select013-obstruction-frontier-v1/frontier_compare.py')
    old=importlib.util.module_from_spec(old_spec);old_spec.loader.exec_module(old)
    with pytest.raises(TypeError,match='Unaccounted source value object'):old.plain(generated)
    monkeypatch.setattr(helper,'PostExposureStart',Generated);monkeypatch.setattr(helper,'_PREPARED',token)
    physical=helper.plain(generated)
    assert physical==dict(zip(names[:-1],values[:-1]))
    assert helper.post_exposure_authority(generated,generated)=={
        'same_prepared_object':True,'same_canonical_capability':True,'capability_serialized':False}
    assert helper.post_exposure_authority(None,None)=={'both_absent':True}
    with pytest.raises(ValueError):helper.post_exposure_authority(generated,Generated(*values))
    wrong=Generated(*values[:-1],object())
    with pytest.raises(ValueError):helper.plain(wrong)
    with pytest.raises(ValueError):helper.post_exposure_authority(generated,wrong)
    with pytest.raises(TypeError,match='Unaccounted source value object'):helper.plain(object())
    assert helper.plain(Generated(values[0],values[1],values[2],'changed free',token))!=physical


def generated(*,rotated=False):
    spec=importlib.util.spec_from_file_location('canonical_obstruction_fixture',ROOT/'tests/test_obstruction_opening_proposals.py')
    fixture=importlib.util.module_from_spec(spec);spec.loader.exec_module(fixture)
    case=fixture.source(enabled=False,cap=120,rotated=rotated,extra=((2,9,9),))
    config=replace(case.proposal_config,offsets_source_voxels=DEFAULT_COLUMN_OFFSETS,intermediate_opening_mm=1.)
    return replace(case,proposal_config=config)


@pytest.mark.parametrize('rotated',[False,True])
def test_fresh_fixed_geometry_replay_and_one_cut_restores_original_ray(rotated):
    case=generated(rotated=rotated);base=NativeSpatialTask(case,max_steps=24).planning_clone()
    enabled,change=helper.enabled_fresh(base)
    assert change['all_other_source_fields_equal'] and case.source_hash!=enabled.case.source_hash
    assert not base.case.proposal_config.obstruction_opening and enabled.case.proposal_config.obstruction_opening
    row=next(r for r in base.candidate_inventory()['emitted'] if r['feasible'] and r['voxel']==[2,9,9] and r['tool_id']=='generic_suction')
    witness=base.clone();expected=witness.step(row['action_id']).info
    writes={};write=lambda k,v:writes.__setitem__(k,v)
    mapped,old,new=helper.compare_prefix(base,enabled,[row['action_id']],[expected],write=write)
    assert mapped[0]!=row['action_id'] and helper.physical_history(old[0])==helper.physical_history(new[0])
    assert writes['prefix-01.json']['committed_masks_equal']
    inv=base.candidate_inventory();assert inv['accepted_count']==0
    result=helper.frontier_branches(enabled,inv,[[4,10,10]],write=write,inspect=inspect_rejected)
    assert result['branch_count']>0 and result['any_excluded_cell_removed'] and result['any_original_ray_feasible']
    assert result['any_preparation_target_progress'] and not result['complete_target_route_claim']
    assert base._steps==enabled._steps==1 and not enabled.terminated
    for key,value in writes.items():
        if key.endswith('/result.json'):
            assert value['closed_episode'] and value['history'][-1]['action_id']=='STOP'
            assert value['complete_negative_prefix_plus_preparation_STOP_return']==sum(r['reward'] for r in value['history'])
            assert len(value['history'])==3  # prefix + one prep + STOP; no direct ray injection.


def test_exact_matching_refuses_missing_infeasible_or_ambiguous_action():
    r={'action_id':'a','family':'exposed','tool_id':'t','entry_mm':[0,0,0],'tip_mm':[1,0,0],'feasible':True}
    assert helper.exact_emitted({'emitted':[r]},key=helper.physical_key(r))==r
    for rows in ([],[r,r],[{**r,'feasible':False}]):
        with pytest.raises(ValueError):helper.exact_emitted({'emitted':rows},key=helper.physical_key(r))


def test_explicit_cap_omission_is_preserved_and_not_previewed():
    case=generated();base=NativeSpatialTask(case,max_steps=24).planning_clone()
    row=next(r for r in base.candidate_inventory()['emitted'] if r['feasible'] and r['voxel']==[2,9,9] and r['tool_id']=='generic_suction')
    base.step(row['action_id']);count=base.candidate_inventory()['emitted_count']
    cap=count+1
    limited=replace(case,proposal_config=replace(case.proposal_config,max_candidates=cap))
    old=NativeSpatialTask(limited,max_steps=24).planning_clone();new,_=helper.enabled_fresh(old)
    oldrow=helper.exact_emitted(old.candidate_inventory(),key=helper.physical_key(row))
    expected=old.clone().step(oldrow['action_id']).info
    helper.compare_prefix(old,new,[oldrow['action_id']],[expected],write=lambda *a:None)
    writes={};calls=[]
    def inspect(t,r):calls.append((r.family,r.tool_id));return inspect_rejected(t,r)
    result=helper.frontier_branches(new,old.candidate_inventory(),[[4,10,10]],write=lambda k,v:writes.__setitem__(k,v),inspect=inspect)
    omissions=writes['added-dispositions.json']['omitted_cell_tool_pairs']
    assert omissions and result['censored_by_omitted_pairs'] and result['added_emitted']==1
    assert len(calls)==result['branch_count']*count
    assert all(family!=OBSTRUCTION_OPENING_FAMILY for family,tool in calls)
    assert old.case.proposal_config.max_candidates==new.case.proposal_config.max_candidates==cap


def test_changed_reference_cannot_change_nominal_preparation_results():
    case=generated();a=NativeSpatialTask(case).planning_clone()
    b=NativeSpatialTask(replace(case,reference_target=np.zeros_like(case.reference_target))).planning_clone()
    x,_=helper.enabled_fresh(a);y,_=helper.enabled_fresh(b)
    assert x.candidate_inventory()==y.candidate_inventory()
    first=next(r for r in x.candidate_inventory()['emitted'] if r['feasible'] and r['voxel']==[2,9,9] and r['tool_id']=='generic_suction')
    assert x.step(first['action_id']).info==y.step(first['action_id']).info
    assert x.candidate_inventory()==y.candidate_inventory()
    row=next(r for r in x.candidate_inventory()['emitted'] if r['family']==OBSTRUCTION_OPENING_FAMILY and r['feasible'])
    assert x.step(row['action_id']).info==y.step(row['action_id']).info


@pytest.mark.parametrize('failure',[None,'prefix'])
def test_actual_worker_wiring_failure_and_zero_model_cleanup(tmp_path,monkeypatch,failure):
    import torch
    from resectionlab import patient_planning_admission as admission,public_patient_factory as factory
    from resectionlab import patient_planning_preflight as pf,planning_budget
    log=[];proposal=NominalCavityProposalConfig(max_candidates=120,intermediate_opening_mm=1.,tool_footprint_opening=True)
    training={'cohort_execution':{'proposal_config':proposal.to_record()},'public_target_context_variant':'generated'}
    class Budget:
        def __init__(self,*a,**k):self.count=0
        def __enter__(self):return self
        def __exit__(self,*a):return False
        def check(self):pass
        def complete(self,*,history_complete):assert history_complete;log.append('complete')
        def snapshot(self):return {'native_preview_entries':self.count}
    class Costs:
        def __init__(self,*a):self.rows={};self.forwards=0
        def __enter__(self):return self
        def __exit__(self,*a):return False
        @contextmanager
        def scope(self,name):
            try:yield
            finally:self.rows[name]={'complete_wall_seconds':0.}
    class Context:
        def record(self):return {'occupancy_condition':contract.prior_contract.OCCUPANCY}
        def require_task(self,t):assert t._planning
    class Case:
        source_hash='generated';proposal_config=proposal
        post_exposure=SimpleNamespace(record=freeze_json({}));_domain_record=freeze_json({})
    class Task:
        max_steps=24;decision_model_hash='generated-model';case=Case();_planning=True
        def __init__(self):self._steps=0;self._inventory={'initial':1};self.terminated=False;self._total_reward=0.;self._engine=SimpleNamespace(state_hash='s0');self.history=[]
        def observation(self):return SimpleNamespace(fingerprint='generated-obs',action_ids=('STOP','initial'),action_mask=np.ones(2,bool))
        def planning_clone(self):return self
        def candidate_inventory(self):return {'saved':'inventory'}
        def step(self,a):
            self._steps+=1;self._inventory={};self.terminated=a=='STOP';self._engine.state_hash='s'+str(self._steps)
            r={'action_id':a};self.history.append(r);return SimpleNamespace(info=r,reward=0.)
        def metrics(self):return {'history':self.history}
    base=Task();context=Context();world={'source_hash':'generated','decision_model_hash':'generated-model','initial_observation_hash':'generated-obs',
        'max_steps':24,'proposal_hash':proposal.fingerprint,'occupancy_condition':contract.prior_contract.OCCUPANCY,
        'post_exposure':{},'source_domains':{},'action_ids':['STOP','initial'],'action_mask':[True,True]}
    actions=tuple('a%d'%i for i in range(11));saved={'training':training,'release':{'public_index':{}},'lineage':{},'context':context.record(),
        'world':world,'prefix':actions,'witness_history':[{'action_id':a} for a in actions],'witness_inventory':{'saved':'inventory'},
        'coverage':{'outside_every_fixed_axis_tip_envelope':[[1,2,3]]}}
    def prepare(*a,**kw):assert kw['partial_domain_inputs'] is True;return Case(),{},{},{}
    def compare(a,b,acts,h,*,write,guard,on_transition):
        for i,act in enumerate(acts):
            guard();a.step(act);on_transition();b.step(act+'-mapped');on_transition();write('prefix-%02d.json'%(i+1),{'exact':True})
            if failure=='prefix' and i==1:raise ValueError('generated prefix failure')
        return list(acts),[],[]
    def branches(*a,**k):
        k['write']('added-dispositions.json',{'omitted_cell_tool_pairs':['retained omission']})
        return {'branch_count':0,'added_emitted':0,'added_legal':0,'omitted_cell_tool_pairs':1}
    cohort=tmp_path/'cohort.json';cohort.write_text('{}')
    monkeypatch.setattr(contract.prior_contract,'COHORT',str(cohort));monkeypatch.setattr(contract.prior_contract,'select_public',lambda p:{'path':'generated','sha256':'generated'})
    monkeypatch.setattr(contract.prior_contract,'read_pinned',lambda ref:{})
    monkeypatch.setattr(worker,'historical',lambda:saved)
    monkeypatch.setattr(factory,'prepare_public_source',prepare);monkeypatch.setattr(admission,'make_patient_planning_task',lambda *a,**k:(base,context))
    monkeypatch.setattr(planning_budget,'PlanningBudget',Budget);monkeypatch.setattr(pf,'_CallCosts',Costs)
    monkeypatch.setattr(helper,'enabled_fresh',lambda a:(Task(),{'all_other_source_fields_equal':True}))
    monkeypatch.setattr(helper,'compare_prefix',compare);monkeypatch.setattr(helper,'frontier_branches',branches);monkeypatch.setattr(helper,'mask_record',lambda task:{'same':'mask'})
    original_load=torch.load;original_grad=torch.autograd.grad;out=tmp_path/'attempt'
    if failure:
        with pytest.raises(ValueError,match='generated prefix'):worker.execute({},'release',out,lambda *a,**k:None)
    else:
        result=worker.execute({},'release',out,lambda *a,**k:None)
        assert contract.complete_result(result) and result['committed_actions']==24 and result['diagnostic_STOP_calls']==2
        assert log==['complete'] and (out/'enabled-closed-negative-history.json').is_file()
    terminal=json.loads((out/'result.json').read_text());assert terminal['model_forwards']==terminal['checkpoint_loads']==terminal['updates']==0
    assert terminal['status']==('failed_or_unresolved' if failure else 'complete_fixed_obstruction_frontier_diagnostic')
    assert (out/'costs.json').is_file() and torch.load is original_load and torch.autograd.grad is original_grad
