"""One actual five-source worker flow on invented7-cube public fixtures."""
import hashlib,json,sys
from pathlib import Path
import numpy as np
import pytest

HERE=Path(__file__).resolve().parents[1];ROOT=HERE.parents[1]
sys.path.insert(0,str(ROOT/'tests'))
import batch_contract as contract
import patient_regression as capability
import patient_transfer as transfer_capability
import policy_tasks
import initial_inventory_worker as worker
from hybrid_proposal import RidgePriority
from resectionlab import public_patient_factory as factory
from resectionlab.native_proposals import NominalCavityProposalConfig
from resectionlab.public_target_context import VERSION as TARGET_CONTEXT

def outcomes():
    return {'target_removed_mm3':100.,'total_reward':100.,'normal_removed_mm3':0.,
        'cumulative_contacted_tissue_upper_bound_mm3':0.,'currently_retained_contacted_tissue_upper_bound_mm3':0.,
        'complete_tool_path_length_mm':0.,'nonstop_actions':0,'tool_changes':0,
        'total_reference_target_mm3':100.,'clinical_deficit_probability':None,'partial_contact_reward_weight':0.}

def test_actual_ten_plans_before_twenty_visit_replays_and_negative_quality(tmp_path,monkeypatch):
    import torch
    import test_post_exposure_learning as generated
    from resectionlab import native_resection,geometry,native_spatial_evaluation
    false=tmp_path/'false.json';false.write_text(json.dumps({'execution_released':False}))
    with pytest.raises(ValueError,match='root_dispatch_required'):
        contract.guard(false,contract.sha(false),execute=True)
    entries=[];worlds={};transfer_inputs=None
    for subject in contract.SUBJECTS:
        folder=tmp_path/subject;folder.mkdir();inputs=generated.fixture(folder,subject)
        if subject==contract.SUBJECT:
            manifest=json.loads(inputs['public_manifest_path'].read_text())
            manifest['schema']='remind-fixed-TRAIN-partial-domain-public-inputs-v1'
            manifest['source_bindings']['intended_use_review_sha256']='e'*64
            inputs['public_manifest_path'].write_text(json.dumps(manifest))
            transfer_inputs={'public_manifest':{'path':str(inputs['public_manifest_path']),
                'sha256':factory.sha(inputs['public_manifest_path'])},'intended_use_review':{'path':'generated-review','sha256':'e'*64}}
        else:
            entries.append({'patient_id':subject,'path':str(inputs['public_manifest_path']),'sha256':inputs['public_manifest_sha256']})
            dest=folder/'preparation';dest.mkdir()
            case,*_=factory.prepare_public_source(dest,{'limits':{'max_steps':24,'max_optimizer_updates':0,
                'max_native_previews':40000,'max_policy_forwards':0,'worker_seconds':600,'memory_bytes':3*1024**3,
                'threads':1,'search':{'max_calls':1,'beam_width':1,'seconds':60}}},contract.SOURCE_RELEASE['sha256'],None,
                lambda *a,**k:None,**inputs,learning_protocol_hash='sha256:'+'d'*64,
                proposal_config=NominalCavityProposalConfig(**contract.PROPOSALS),public_target_context_variant=TARGET_CONTEXT,
                occupancy_condition=contract.OCCUPANCY,post_exposure_condition=contract.EXPOSURE)
            worlds[subject]=case.source_hash
    idx=tmp_path/'index.json';idx.write_text(json.dumps({'cases':entries}))
    model=RidgePriority(np.zeros(681),np.ones(681),np.zeros(681),0.)
    raw=json.dumps({'feature_version':model.fingerprint,'data_hash':contract.RIDGE['data_hash'],
        'mean':[0.]*681,'scale':[1.]*681,'weights':[0.]*681,'intercept':0.}).encode()
    fake={**contract.RIDGE,'sha256':hashlib.sha256(raw).hexdigest(),'fingerprint':model.fingerprint}
    for module in (contract,worker,capability,transfer_capability,policy_tasks):monkeypatch.setattr(module,'RIDGE',fake)
    monkeypatch.setattr(capability,'read_bytes',lambda path,expected,limit:raw)
    for module in (contract,worker):monkeypatch.setattr(module,'OUTPUT',tmp_path/'worker-output')
    hooks=[];monkeypatch.setattr(worker.sys,'addaudithook',hooks.append)
    monkeypatch.setattr(torch,'set_num_threads',lambda n:None);monkeypatch.setattr(torch,'set_num_interop_threads',lambda n:None)
    rows=[]
    for method in ('balanced_ridge_top8','exhaustive'):
        for subject in contract.FIT_SUBJECTS:
            key=method+'/'+subject
            rows.append({'method':method,'subject':subject,'outcomes':outcomes(),
                'files':{key+'/plan.json':'c'*64,key+'/replay/episode-metrics.json':'d'*64}})
    fifty=[]
    for method in ('balanced_ridge_switch_guard','exhaustive'):
        fifty.append({'method':method,'subject':contract.SUBJECT,'outcomes':outcomes(),
            'files':{method+'/plan.json':'c'*64,method+'/replay/episode-metrics.json':'d'*64}})
    source={'subjects':contract.FIT_SUBJECTS,'retention_reference':'return_plus_opening_depth_volume_v1'}
    monkeypatch.setattr(worker,'metadata',lambda:({'arms':rows},{},source))
    monkeypatch.setattr(worker,'transfer_metadata',lambda:({'arms':fifty},{}))
    old_small=worker.small
    def metadata(path,expected=None):
        if any(str(path).startswith(str(p)+'/') for p in (worker.TRAIN_OUTPUT,worker.FIFTY_OUTPUT)):
            if str(path).endswith('/plan.json'):return {'actions':[],'source_hash':'deliberately_wrong_history'}
            return {'decision_model_hash':'historical','history':[]}
        return old_small(path,expected)
    monkeypatch.setattr(worker,'small',metadata)
    original=native_spatial_evaluation.evaluate_native_spatial_episode;calls=[]
    keys=[m+'/'+s for s in contract.SUBJECTS for m in contract.ARMS]
    def evaluate(task,**kwargs):
        seals=json.loads((worker.OUTPUT/'sealed-plans.json').read_text())
        assert len(seals['plans'])==10 and set(seals['plans'])==set(keys)
        assert all((worker.OUTPUT/k/'plan.json').exists() for k in keys)
        assert native_resection.capsule_voxel_indices is geometry.capsule_voxel_indices
        calls.append(task.case.source_hash)
        return original(task,**kwargs)
    monkeypatch.setattr(native_spatial_evaluation,'evaluate_native_spatial_episode',evaluate)
    release={'public_index':{'path':str(idx),'sha256':contract.sha(idx)},
             'source_world_hashes':worlds,'transfer_inputs':transfer_inputs}
    result=worker.execute(release,'b'*64,{'blocked':0});contract.complete_result(result,release,'b'*64)
    assert len(calls)==10 and len(hooks)==1 and result['source_visits']==20
    assert result['all_plans_sealed_before_evaluation'] and result['ridge_unchanged'] and result['cache_restored']
    assert result['fits']==result['model_calls']==result['optimizer_calls']==result['checkpoint_loads']==0
    assert len(result['arms'])==10 and all(r['complete'] and r['independent_accepted'] for r in result['arms'])
    assert result['comparison']['exact_policy_equivalence'] is False # Invented poor historical comparator retained.
    assert all(p['exact_actions'] and p['exact_history'] and p['exact_metrics_except_outer_policy_identity']
               and p['exact_outcomes'] for p in result['comparison']['pairs'])
    assert result['status']=='complete_five_TRAIN_public_bound_comparison'
    assert result['native_transitions']==2*sum(r['action_count'] for r in result['arms'])
    assert result['native_previews']==sum(r['online_previews']+r['replay_previews'] for r in result['arms'])
