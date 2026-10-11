"""Generated-only blocker geometry/capacity and actual worker glue; no acquired source."""
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
import blocker_coverage as helper
import comparison_contract as contract
import select_worker as worker
from resectionlab.native_resection import NativeResectionConfig,NativeResectionEngine
from resectionlab.geometry import AccessWindow,GENERIC_TOOLS
from resectionlab.native_proposals import NominalCavityProposalConfig,DEFAULT_COLUMN_OFFSETS


def generated_task(*,reflected=False):
    affine=np.eye(4)
    if reflected:affine[:3,:3]=[[0,0,-1],[1,0,0],[0,-1,0]];affine[:3,3]=[20,-5,8]
    tissue=np.zeros((12,13,13),bool);tissue[5,6,6]=True;tissue[5,7,7]=True
    normal=affine[:3,0];entry=affine[:3,:3]@[.5,6,6]+affine[:3,3]
    cfg=NativeResectionConfig(tissue,np.zeros(tissue.shape,np.int16),affine,AccessWindow(entry,normal,6.),
        GENERIC_TOOLS,'generated-fixed-axis','generated two cells, no anatomy')
    if reflected:
        from resectionlab.post_exposure import prepare_post_exposure
        target=np.zeros_like(tissue);target[5,6,6]=True
        access,start=prepare_post_exposure(support=tissue,target=target,support_domain=np.ones_like(tissue),
            affine=affine,previous_access=cfg.access)
        cfg=replace(cfg,access=access,interaction_domain=np.ones_like(tissue),post_exposure=start)
        entry=np.asarray(access.center_mm)
    engine=NativeResectionEngine(cfg,immutable_state=True)
    proposer=SimpleNamespace(_axis=0,_transverse=(1,2),_columns=tuple((6+a,6+b) for a,b in DEFAULT_COLUMN_OFFSETS))
    case=SimpleNamespace(_nominal_proposer=proposer,proposal_mode='nominal_cavity_v1',proposal_config=NominalCavityProposalConfig())
    task=SimpleNamespace(_planning=True,_config=cfg,_engine=engine,case=case,_assert_frozen=lambda:None)
    ray=SimpleNamespace(proposal_id='generated-ray',family='proximal_nominal',column_index=0,tool_id='generic_suction',
        entry_mm=tuple(entry),tip_mm=tuple(affine[:3,:3]@[9,6,6]+affine[:3,3]))
    return task,ray


@pytest.mark.parametrize('reflected',[False,True])
def test_real_first_shaft_failure_and_exact_whole_cell_capacity(reflected):
    task,ray=generated_task(reflected=reflected);before=task._engine.state_hash
    record=helper.inspect_rejected(task,ray)
    assert not record['feasible'] and record['reason']=='SHAFT_BLOCKED_BY_REMAINING_NATIVE_TISSUE'
    side=record['diagnostic'];assert [5,7,7] in side['blocked_indices_native']
    assert side['complete_first_failure_set'] and not side['truncated']
    assert side['prior_temporary_removals_committed'] is False
    result=helper.classify_cells(task,[[5,6,6],[5,7,7]])
    assert result['outside_every_fixed_axis_tip_envelope']==[[5,7,7]]
    assert result['inside_at_least_one_optimistic_envelope']==[[5,6,6]]
    assert len(result['pairs'])==26 and result['additional_native_previews']==0
    expected=7 if reflected else 11
    assert result['native_containment_kernel_calls']==2*26*expected
    assert all(row['positive_source_endpoints_checked']==expected for row in result['pairs'])
    if reflected:
        start=helper.preentry_tip(task._config.tools[0],ray.entry_mm,task._config.access.normal_inward,task._config.access.normal_inward)
        assert record['first_returned_microstep_start_mm']==start.tolist()
        assert record['first_returned_microstep_start_mm']!=list(ray.entry_mm)
    assert task._engine.state_hash==before and task._engine.history==[]
    assert not task._engine.removed_mask.any()


def test_cap_truncation_and_no_false_full_set_claim():
    task,ray=generated_task();tissue=task._config.tissue_mask.copy();tissue[5,5,7]=True
    cfg=replace(task._config,tissue_mask=tissue);task._config=cfg;task._engine=NativeResectionEngine(cfg,immutable_state=True)
    row=helper.inspect_rejected(task,ray,cell_limit=1);d=row['diagnostic']
    assert d['blocked_cell_count']>1 and d['retained_cell_count']==1
    assert d['truncated'] and not d['complete_first_failure_set']


def test_non_remaining_and_nonpublic_inputs_refused():
    task,_=generated_task()
    with pytest.raises(ValueError,match='remaining'):helper.classify_cells(task,[[1,1,1]])
    task._planning=False
    with pytest.raises(ValueError):helper.classify_cells(task,[[5,7,7]])


@pytest.mark.parametrize('failure',[None,'world','coverage'])
def test_actual_execute_persists_prefix_and_failure_without_weights(tmp_path,monkeypatch,failure):
    import torch
    from resectionlab import patient_planning_admission as admission,public_patient_factory as factory
    from resectionlab import patient_planning_preflight as pf,planning_budget
    from resectionlab.core import freeze_json
    log=[];live={};actions=tuple('action-%d'%i for i in range(11))
    proposal=NominalCavityProposalConfig(max_candidates=120,intermediate_opening_mm=1.,tool_footprint_opening=True)
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
        post_exposure=SimpleNamespace(record=freeze_json({}))
        _domain_record=freeze_json({})
    class Task:
        max_steps=24;decision_model_hash='generated-model'
        def __init__(self):
            self._planning=False;self._steps=0;self.terminated=False;self._engine=SimpleNamespace(state_hash='s0')
            self._inventory={'first':1};self._proposal_batch=SimpleNamespace(proposals=[SimpleNamespace(proposal_id='blocked')])
        def observation(self):return SimpleNamespace(fingerprint='generated-obs',action_ids=('STOP','first'),action_mask=np.ones(2,bool))
        def planning_clone(self):self._planning=True;return self
        def candidate_inventory(self):return {'generated':True}
        def step(self,a):
            log.append(a)
            if a=='STOP':self.terminated=True;return SimpleNamespace(info={'action_id':'STOP'},reward=0.)
            assert a==actions[self._steps];self._steps+=1;self._engine.state_hash='s'+str(self._steps)
            self._inventory={} if self._steps==11 else {'first':1}
            row={'action_id':a,'outcome_scope':'permitted_nominal_model','target_removed_mm3':0.,'normal_removed_mm3':1.,'insertion_distance_mm':2.}
            return SimpleNamespace(info=row,reward=-1.)
    task=Task();context=Context();world={'source_hash':'generated','decision_model_hash':'generated-model','initial_observation_hash':'generated-obs',
        'max_steps':24,'proposal_hash':proposal.fingerprint,'occupancy_condition':contract.prior_contract.OCCUPANCY,
        'post_exposure':{},'source_domains':{},'action_ids':['STOP','first'],'action_mask':[True,True]}
    saved={'training':training,'lineage':{'generated':True},'context':context.record(),'world':world,
        'release':{'public_index':{}},'prefix':actions,'prefix_checkpoints':[{'actions':list(actions[:i]),'estimated_incremental_return':float(-i),
        'target_removed_mm3':0.,'outside_supplied_target_removed_mm3':float(i),'max_insertion_distance_mm':2.,'legal_nonstop_actions':0 if i==11 else 1} for i in range(1,12)]}
    if failure=='world':saved['world']={**world,'source_hash':'wrong'}
    def prepare(*a,**kw):
        assert kw['checkpoint_lineage']==saved['lineage'] and kw['partial_domain_inputs'] is True
        log.append('factory');return Case(),{},{},{}
    def classify(*a,**kw):
        if failure=='coverage':raise ValueError('generated coverage failure')
        return {'outside_every_fixed_axis_tip_envelope':[[1,1,1]],'native_containment_kernel_calls':26}
    monkeypatch.setattr(worker,'historical',lambda:saved)
    monkeypatch.setattr(contract.prior_contract,'select_public',lambda x:{'path':'generated','sha256':'a'*64})
    monkeypatch.setattr(contract.prior_contract,'read_pinned',lambda x:{})
    monkeypatch.setattr(factory,'prepare_public_source',prepare);monkeypatch.setattr(admission,'make_patient_planning_task',lambda *a,**k:(task,context))
    monkeypatch.setattr(planning_budget,'PlanningBudget',Budget);monkeypatch.setattr(pf,'_CallCosts',Costs)
    monkeypatch.setattr(helper,'inspect_rejected',lambda *a,**k:{'feasible':False,'diagnostic':{'blocked_indices_native':[[1,1,1]],'truncated':False}})
    monkeypatch.setattr(helper,'classify_cells',classify)
    old=(torch.load,torch.optim.Optimizer.__init__,torch.autograd.grad,torch.autograd.backward)
    if failure:
        with pytest.raises(ValueError):worker.execute({},'a'*64,tmp_path/'out',lambda *a,**k:None)
        result=json.loads((tmp_path/'out/result.json').read_text())
    else:result=worker.execute({},'a'*64,tmp_path/'out',lambda *a,**k:None)
    assert old==(torch.load,torch.optim.Optimizer.__init__,torch.autograd.grad,torch.autograd.backward)
    assert contract.complete_result(result) is (failure is None)
    assert result['checkpoint_loads']==result['model_forwards']==result['optimizer_attempts']==0
    if failure=='world':assert log==['factory']
    else:
        assert log[:12]==['factory',*actions]
        assert (tmp_path/'out/dead-frontier-inventory.json').is_file()
        assert len(json.loads((tmp_path/'out/diagnostic-prefix-history.json').read_text()))==11
    if failure:assert 'complete' not in log and 'STOP' not in log
    else:assert log[-2:]==['STOP','complete'] and result['accepted_plan_claim'] is False


@pytest.mark.parametrize('anisotropic',[False,True])
def test_tangency_is_unknown_not_excluded(anisotropic):
    task,_=generated_task(reflected=True)
    affine=task._config.affine.copy()
    if anisotropic:affine[:3,1]*=.8;affine[:3,2]*=1.2
    radius=float(np.linalg.norm(1.5*affine[:3,1]+1.5*affine[:3,2]))
    tool=replace(task._config.tools[1],tip_radius_mm=radius)
    entry=affine[:3,:3]@[.5,6,6]+affine[:3,3]
    cfg=replace(task._config,affine=affine,tools=(task._config.tools[0],tool),access=AccessWindow(entry,affine[:3,0],6.),post_exposure=None,interaction_domain=None)
    task._config=cfg;task._engine=NativeResectionEngine(cfg,immutable_state=True)
    record=helper.classify_cells(task,[[5,7,7]])
    assert record['outside_every_fixed_axis_tip_envelope']==[]
    assert record['roundoff_ambiguous_cells']==[[5,7,7]]
    assert record['inside_at_least_one_optimistic_envelope']==[]


def test_out_of_numerical_scope_is_unknown():
    task,_=generated_task();affine=task._config.affine.copy();affine[0,3]=20000.
    access=AccessWindow([20000.5,6,6],[1,0,0],6.)
    cfg=replace(task._config,affine=affine,access=access)
    task._config=cfg;task._engine=NativeResectionEngine(cfg,immutable_state=True)
    record=helper.classify_cells(task,[[5,7,7]])
    assert record['outside_every_fixed_axis_tip_envelope']==[]
    assert record['roundoff_ambiguous_cells']==[[5,7,7]]
    assert record['numerical_applicability']['applicable'] is False
    assert record['native_containment_kernel_calls']==0
