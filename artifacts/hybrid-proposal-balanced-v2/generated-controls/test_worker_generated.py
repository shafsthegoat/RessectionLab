"""Actual execute glue with tiny generated public manifests and native tasks.

Historical metadata and its fixed29 denominator are replaced explicitly here;
actual source factory, admission, ridge, native queries, replay and output writes
remain real. No acquired descriptors, checkpoint or optimizer is used.
"""
import importlib.util,json,sys,types
from pathlib import Path
import pytest
import numpy as np

ROOT=next(p for p in Path(__file__).resolve().parents if (p/'src/resectionlab').is_dir())
sys.path.insert(0,str(ROOT/'tests'));sys.path.insert(0,str(Path(__file__).parent))
import test_post_exposure_learning as generated
from test_source_admission import staged
from resectionlab import patient_planning_admission as admission
from resectionlab.public_patient_factory import prepare_public_source,sha
from resectionlab.native_proposals import NominalCavityProposalConfig
from resectionlab.public_target_context import VERSION as TARGET_CONTEXT
from resectionlab.core import semantic_digest,freeze_json,thaw_json,array_digest
from resectionlab.native_spatial_evaluation import evaluate_native_spatial_episode
from batch_contract import SUBJECTS,ARMS,GATES,CAPS,CACHE_LIMITS
import initial_inventory_worker as worker
from test_balanced_contrast import old_module


def test_actual_execute_four_generated_sources_and_all_three_strategies(tmp_path,monkeypatch):
    module=staged();monkeypatch.setattr(admission,'validate_patient_planning_source',module.validate_patient_planning_source,raising=False)
    monkeypatch.setattr(admission,'make_patient_planning_task',module.make_patient_planning_task)
    p=generated.protocol();config=thaw_json(p['cohort_execution']['proposal_config'])
    entries=[];inputs={}
    for subject in SUBJECTS:
        folder=tmp_path/subject;folder.mkdir();inputs[subject]=generated.fixture(folder,subject)
        entries.append({'patient_id':subject,'path':str(inputs[subject]['public_manifest_path']),'sha256':inputs[subject]['public_manifest_sha256']})
    index=tmp_path/'index.json';index.write_text(json.dumps({'cases':entries}))
    old={'subjects':SUBJECTS,'occupancy_condition':admission.PARTIAL_DOMAIN_UNION_OCCUPANCY,
        'post_exposure_condition':generated.EXPOSURE,'proposal_config':config,
        'retention_reference':'generated metadata control; unused by greedy',
        'public_index':{'path':str(index),'sha256':sha(index)}}
    qualification={'scope':'post_exposure_initial_and_nominal_greedy','subjects':tuple(SUBJECTS),
        'occupancy_condition':old['occupancy_condition'],'proposal_config':config,'max_steps':24,
        'methods_executed':['observed_greedy'],'post_exposure_condition':old['post_exposure_condition'],
        'greedy_seconds_per_case':60,'retention_reference':old['retention_reference'],'retention_applied':False}
    limits={'max_steps':24,'max_optimizer_updates':0,'max_native_previews':40000,'max_policy_forwards':0,
        'worker_seconds':300,'memory_bytes':3*1024**3,'threads':1,'search':{'max_calls':1,'beam_width':1,'seconds':60}}
    prior={'release':old,'result':{'cases':[]}};steps=[]
    historical=old_module();train_rows=[];xs=[];ys=[];state_ids=[];cases={}
    for subject in SUBJECTS:
        folder=tmp_path/subject/'baseline';folder.mkdir()
        case,binding,qc,protocol=prepare_public_source(folder,{'limits':limits},'b'*64,None,lambda *a,**k:None,
            **inputs[subject],learning_protocol_hash=semantic_digest(qualification),proposal_config=NominalCavityProposalConfig(**config),
            public_target_context_variant=TARGET_CONTEXT,occupancy_condition=old['occupancy_condition'],post_exposure_condition=old['post_exposure_condition'])
        task,ctx=module.make_patient_planning_task(case,cohort_bytes=inputs[subject]['cohort_bytes'],source_binding=binding,qc_receipt=qc,protocol=protocol)
        actions,accounting=task.observed_greedy_search(seconds=60)
        model=task.planning_clone()
        for step,action in enumerate(actions):
            inv=model._prepare_inventory()
            rows=historical.public_rows(case,model._engine.connected_free_mask,model._proposal_batch.proposals,
                steps=step,horizon=24,current_tool=model._current_tool)
            labels=[max(0.,model._score_record({**inv[i].to_history_record(),'action_id':i},model._current_tool)['reward']) if i in inv else 0. for i in rows.ids]
            xs.append(rows.features);ys.extend(labels);state_ids.extend([len(train_rows)]*len(labels))
            train_rows.append({'subject':subject,'step':step,'teacher_action':action,'public_input_hash':rows.public_binding,
                'feature_hash':array_digest(rows.features),'raw_ids':list(rows.ids),'legal':[i in inv for i in rows.ids],
                'priority_labels':labels,'feature_rows':len(labels)})
            model.advance_planning(action);task.step(action)
        audit=evaluate_native_spatial_episode(task);assert audit['accepted']
        prior['result']['cases'].append({'patient_id':subject,'source_hash':case.source_hash,'context_hash':ctx.fingerprint})
        prior['plan_'+subject]={'actions':list(actions),'accounting':thaw_json(freeze_json(accounting))}
        prior['metrics_'+subject]=thaw_json(freeze_json(task.metrics()));steps.append(len(actions))
        cases[subject]=case
    x,y=np.concatenate(xs),np.asarray(ys)
    historical_fit=historical.RidgePriority.fit(x,y,state_ids)
    old_control={'labels':{'states':train_rows,'data_hash':semantic_digest({'X':array_digest(x),'y':array_digest(y),'state_ids':state_ids})},
        'plans':{},'result':{'arms':[],'comparison':{'TRAIN_go':False,'generated_historical_control':True}}}
    class GeneratedHistoricalTask(historical.GeneratedSparseTask):
        def _require_source(self,case):
            # Test fixtures are explicitly generated annotation-style DTOs;
            # this does not authorize an acquired source in production.
            assert any(case is v for v in cases.values())
    for subject,case in cases.items():
        task=GeneratedHistoricalTask(case,mode='ridge_top8',ranker=historical_fit,max_steps=24)
        actions,accounting=task.observed_greedy_search(seconds=60)
        for action in actions:task.step(action)
        audit=evaluate_native_spatial_episode(task);assert audit['accepted']
        old_control['plans'][subject]={'actions':list(actions),'ridge_hash':historical_fit.fingerprint}
        old_control['result']['arms'].append({'method':'ridge_top8','subject':subject,'outcomes':audit['outcomes']})
    owner=types.SimpleNamespace(COHORT='manifests/experiments/remind-component-cohort-v1.json',
        COHORT_SHA=sha(ROOT/'manifests/experiments/remind-component-cohort-v1.json'),INPUTS={'release':{'sha256':'b'*64}})
    release={**old,'caps':CAPS,'cache_limits':CACHE_LIMITS,'gates':GATES,
        'fit':{'state_weights':{'old':'uniform','balanced':'balanced'}},'historical_v1':{'gates':{'generated':True}},
        'source_world_hashes':{r['patient_id']:r['source_hash'] for r in prior['result']['cases']}}
    monkeypatch.setattr(worker,'metadata',lambda:(owner,prior,{'outcomes':{'generated':True},'proposal_v1':old_control}))
    monkeypatch.setattr(worker,'OUTPUT',tmp_path/'execute')
    monkeypatch.setattr(worker,'STEPS',steps)
    monkeypatch.setattr(worker.sys,'addaudithook',lambda hook:None) # audit-hook behavior is a separate boundary; cannot unregister it in pytest.
    # PyTorch limits belong to isolated worker initialization; pytest already imported it.
    import torch
    monkeypatch.setattr(torch,'set_num_interop_threads',lambda n:None)
    result=worker.execute(release,'a'*64,{'blocked':0})
    assert result['status']=='complete_fixed_TRAIN_balanced_ridge_contrast'
    assert result['source_visits']==28 and result['teacher_decisions']==sum(steps)
    assert result['fits']==2 and len(result['arms'])==12
    assert result['old_ridge_parameter_identity'] and result['teacher_labels_identity']
    assert result['teacher_score_calls']==2*sum(steps)
    assert not result['comparison']['historical_v1_comparison']['TRAIN_go']
    assert result['all_plans_sealed_before_evaluation'] and result['cache_restored'] and result['ridge_unchanged']
    assert all(r['complete'] and r['independent_accepted'] for r in result['arms'])
    assert result['optimizer_calls']==result['model_calls']==result['checkpoint_loads']==0
    for row in result['arms']:
        if row['method']!='exhaustive':
            assert row['online_previews']<=8*24 and row['replay_previews']<=8*24
    assert len(list((tmp_path/'execute').glob('*/*/plan.json')))==12

    # A later source refusal must identify that new arm, preserve its zero-query
    # cost and leave the previous arm's completed planning record intact.
    from resectionlab import public_patient_factory
    original_prepare=public_patient_factory.prepare_public_source
    calls=[]
    def refuse_sixth(*a,**k):
        calls.append(1)
        if len(calls)==6:raise ValueError('generated second planning source refusal')
        return original_prepare(*a,**k)
    monkeypatch.setattr(public_patient_factory,'prepare_public_source',refuse_sixth)
    failure=tmp_path/'execute-refusal';monkeypatch.setattr(worker,'OUTPUT',failure)
    with pytest.raises(ValueError,match='second planning source refusal'):
        worker.execute(release,'a'*64,{'blocked':0})
    failed=json.loads((failure/'result.json').read_text())
    assert failed['status']=='failed_or_unresolved' and len(failed['arms'])==12
    assert failed['failed_operation_costs']['subject']=='ReMIND-015'
    assert failed['failed_operation_costs']['native_previews_paid']==0
    assert failed['arms'][0]['status']=='planning'
    assert failed['arms'][1]['status']=='failed_or_unresolved'
    assert failed['source_visits_detail'][-1]['subject']=='ReMIND-015'
    assert failed['source_visits_detail'][-1]['source_read_preparation_validation_seconds']>=0
    from resectionlab import native_resection,geometry
    assert native_resection.capsule_voxel_indices is geometry.capsule_voxel_indices
    assert not getattr(torch.nn.Module.__init__,'__name__','')=='refused'
