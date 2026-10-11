"""One real generated execute path plus truthful cap/failure persistence controls."""
from dataclasses import replace
import importlib.util
import json
from pathlib import Path
import sys
import numpy as np
import pytest
HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[1]
sys.path.insert(0,str(ROOT/'src'));sys.path.insert(0,str(ROOT/'tests'));sys.path.insert(0,str(HERE))
import comparison_contract as contract
import select_worker as worker
from resectionlab.core import thaw_json
from resectionlab.native_resection import NativeResectionEngine
from resectionlab.native_spatial_task import NativeSpatialTask
from resectionlab.paged_obstruction import PagedObstructionTask
from resectionlab.observed_search import observed_beam_search
from test_obstruction_opening_proposals import source


@pytest.mark.parametrize('failure',[None,'second_source','capped'])
def test_real_execute_complete_or_retains_unresolved_and_restores_hooks(tmp_path,monkeypatch,failure):
    import torch
    from resectionlab import patient_planning_admission as admission,public_patient_factory as factory
    from resectionlab import native_spatial_evaluation as evaluation,observed_search as search
    # Exact native generated source, page proposer, beam, budget, output sink,
    # commits, replay and independent geometry are real. Only acquired admission
    # and immutable historical records are replaced by explicit generated data.
    case=source(enabled=False,cap=120,extra=((4,10,8),))
    target=np.zeros_like(case.nominal_target);target[9,9,9]=1
    case=replace(case,nominal_target=target,reference_target=target)
    context_record={'occupancy_condition':'generated-public-occupancy'}
    class Context:
        def record(self):return context_record
    baseline=NativeSpatialTask(case,max_steps=24)
    obs=baseline.observation()
    world={'source_hash':case.source_hash,'decision_model_hash':baseline.decision_model_hash,
        'initial_observation_hash':obs.fingerprint,'max_steps':24,'proposal_hash':case.proposal_config.fingerprint,
        'occupancy_condition':context_record['occupancy_condition'],'post_exposure':None,
        'source_domains':thaw_json(case._domain_record),'action_ids':list(obs.action_ids),'action_mask':obs.action_mask.tolist()}
    actions,accounting=observed_beam_search(baseline,max_calls=7191,beam_width=2,seconds=300,policy=None,
        objective_source='permitted_nominal_target_and_frozen_geometric_costs',transition_mode='lazy_planning',
        retention_mode=contract.prior_contract.MODE,retained_prefix_diagnostics=True)
    # Historical evidence is read from JSON in production, so normalize tuples.
    saved=json.loads(json.dumps({'world':world,'context':context_record,'release':{'public_index':{}},'lineage':{},
        'training':{'cohort_execution':{'proposal_config':case.proposal_config.to_record()},'public_target_context_variant':'generated'},
        'selection':{'actions':actions,'accounting':accounting}}))
    cohort=tmp_path/'cohort.json';cohort.write_text('{}')
    monkeypatch.setattr(contract.prior_contract,'COHORT',str(cohort))
    monkeypatch.setattr(contract.prior_contract,'select_public',lambda ref:{'path':'explicit-generated','sha256':'generated'})
    monkeypatch.setattr(contract.prior_contract,'read_pinned',lambda ref:{})
    monkeypatch.setattr(worker,'historical',lambda:saved)
    calls=[]
    def prepare(folder,*a,**kw):
        calls.append(folder)
        assert kw['partial_domain_inputs'] is True and kw['expected_role']=='SELECT'
        assert kw['proposal_config']==case.proposal_config
        if failure=='second_source' and len(calls)==2:raise ValueError('Generated second-source refusal')
        return case,{},{},{}
    monkeypatch.setattr(factory,'prepare_public_source',prepare)
    monkeypatch.setattr(admission,'make_patient_planning_task',lambda c,**kw:(NativeSpatialTask(c,max_steps=24),Context()))
    if failure=='capped':
        real=search.observed_beam_search
        def capped(task,**kw):
            kw['max_calls']=0
            return real(task,**kw)
        monkeypatch.setattr(search,'observed_beam_search',capped)
    evaluated=[];real_evaluate=evaluation.evaluate_native_spatial_episode;out=tmp_path/'attempt'
    def evaluate(task,**kw):
        assert all((out/arm/'selected-action-plan.json').is_file() for arm in contract.ARMS)
        evaluated.append(task.case.proposal_config.obstruction_page_size)
        return real_evaluate(task,**kw)
    monkeypatch.setattr(evaluation,'evaluate_native_spatial_episode',evaluate)
    hooks=(NativeResectionEngine.preview_stroke,NativeSpatialTask._transition,NativeSpatialTask._prepare_inventory,
        PagedObstructionTask.planning_action_pages,torch.load,torch.autograd.grad)
    if failure:
        with pytest.raises(ValueError,match='second-source|Unresolved search'):
            worker.execute({},'generated-release',out,lambda *a,**kw:None)
    else:
        result=worker.execute({},'generated-release',out,lambda *a,**kw:None)
        assert contract.complete_result(result)
        assert evaluated==[None,8]
        assert result['comparison']['positive_complete_return'] and result['comparison']['target_progress']
        assert result['arms'][contract.ARMS[0]]['actions']==['STOP']
        for arm,row in result['arms'].items():
            assert row['selector_accounting']['model_transition_calls']==row['native_transitions']['search']
            assert row['independent_accepted'] and row['full_history_equal']
            assert row['native_transitions']['rollout']==row['native_transitions']['replay']==len(row['actions'])
    assert hooks==(NativeResectionEngine.preview_stroke,NativeSpatialTask._transition,NativeSpatialTask._prepare_inventory,
        PagedObstructionTask.planning_action_pages,torch.load,torch.autograd.grad)
    result=json.loads((out/'result.json').read_text());cost=json.loads((out/'costs.json').read_text())
    assert result['checkpoint_loads']==result['model_forwards']==result['updates']==result['gradient_attempts']==0
    assert result['native_budget']['native_preview_entries']==sum(result['native_previews_by_arm'].values())
    assert cost['phases'] and len(result['arms'])==2
    if failure:
        assert result['status']=='failed_or_unresolved' and not evaluated and not contract.complete_result(result)
        if failure=='capped':
            assert result['arms'][contract.ARMS[0]]['status']=='selected_complete' # Root has no legal motions.
            assert result['arms'][contract.ARMS[1]]['status']=='search_unresolved'
            assert (out/contract.ARMS[1]/'search-unresolved.json').is_file()
            assert not (out/contract.ARMS[1]/'selected-action-plan.json').exists()
        else:assert result['failure']['phase']=={'arm':contract.ARMS[1],'name':'source_and_construction'}
