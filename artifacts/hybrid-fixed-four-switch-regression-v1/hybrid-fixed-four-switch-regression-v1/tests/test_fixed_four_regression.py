"""Generated glue and capability controls; no acquired payload or fitted weights."""
import copy,hashlib,json,sys
from pathlib import Path
import numpy as np
import pytest

HERE=Path(__file__).resolve().parents[1]
ROOT=next(p for p in HERE.parents if (p/'src/resectionlab').is_dir())
sys.path.insert(0,str(HERE));sys.path.insert(0,str(ROOT/'tests'))
import batch_contract as contract
import patient_regression as capability
import initial_inventory_worker as worker
from hybrid_proposal import RidgePriority
from resectionlab.core import semantic_digest
from resectionlab.native_proposals import NominalCavityProposalConfig
from resectionlab.public_target_context import VERSION as TARGET_CONTEXT
from resectionlab import public_patient_factory as factory


def source_options():
    return dict(learning_protocol_hash='sha256:'+'d'*64,
        proposal_config=NominalCavityProposalConfig(**contract.PROPOSALS),
        public_target_context_variant=TARGET_CONTEXT,occupancy_condition=contract.OCCUPANCY,
        post_exposure_condition=contract.EXPOSURE)


def limits():
    return {'max_steps':24,'max_optimizer_updates':0,'max_native_previews':40000,'max_policy_forwards':0,
        'worker_seconds':300,'memory_bytes':3*1024**3,'threads':1,
        'search':{'max_calls':1,'beam_width':1,'seconds':60}}


def generated_source(tmp_path,subject):
    import test_post_exposure_learning as fixture_owner
    root=tmp_path/subject;root.mkdir()
    inputs=fixture_owner.fixture(root,subject)
    out=root/'source';out.mkdir()
    case,binding,qc,protocol=factory.prepare_public_source(out,{'limits':limits()},contract.SOURCE_RELEASE['sha256'],None,
        lambda *a,**k:None,**inputs,**source_options())
    return inputs,(case,binding,qc,protocol)


def declaration(worlds):
    return {'version':contract.VERSION,'released_sha256':'b'*64,'source_world_hashes':worlds,
        'proposal_config':contract.PROPOSALS,'post_exposure_condition':contract.EXPOSURE,
        'fits':0,'optimizer_updates':0,'neural_forwards':0,'frozen_ridge':contract.RIDGE,'training_result':contract.TRAIN_RESULT}


def test_fixed_four_capability_zero_fit_and_no_transfer_bypass(tmp_path,monkeypatch):
    from resectionlab.native_resection import NativeResectionEngine
    inputs,(case,binding,qc,protocol)=generated_source(tmp_path,contract.SUBJECTS[0])
    worlds={s:case.source_hash for s in contract.SUBJECTS};d=declaration(worlds)
    monkeypatch.setattr(NativeResectionEngine,'preview_stroke',lambda *a,**k:pytest.fail('admission cannot preview'))
    args=dict(cohort_bytes=inputs['cohort_bytes'],source_binding=binding,qc_receipt=qc,protocol=protocol,pilot_declaration=d)
    admitted=capability.admit_source(case,**args);admitted.require_source(case)
    for field,value in [('fits',1),('optimizer_updates',1),('neural_forwards',1),('fits',False),('released_sha256',None)]:
        bad=copy.deepcopy(args);bad['pilot_declaration'][field]=value
        with pytest.raises(ValueError):capability.admit_source(case,**bad)
    bad=copy.deepcopy(args);bad['pilot_declaration']['source_world_hashes'][contract.SUBJECTS[0]]='sha256:'+'0'*64
    with pytest.raises(ValueError):capability.admit_source(case,**bad)
    bad=copy.deepcopy(args);bad['pilot_declaration']['source_world_hashes']={s:case.source_hash for s in reversed(contract.SUBJECTS)}
    with pytest.raises(ValueError):capability.admit_source(case,**bad)
    bad=copy.deepcopy(args);bad['protocol']['development_transfer']={}
    with pytest.raises(ValueError):capability.admit_source(case,**bad)


def scalar_outcomes():
    return {'target_removed_mm3':1.,'total_reward':.8,'normal_removed_mm3':0.,
        'cumulative_contacted_tissue_upper_bound_mm3':2.,'currently_retained_contacted_tissue_upper_bound_mm3':1.,
        'complete_tool_path_length_mm':2.,'nonstop_actions':1,'tool_changes':0,
        'total_reference_target_mm3':10.,'clinical_deficit_probability':None,'partial_contact_reward_weight':0.}


def test_comparison_normalizes_only_explicit_policy_identity():
    outcome=scalar_outcomes();reference={m:{'outcomes':outcome} for m in ('balanced_ridge_top8','exhaustive')}
    p={'actions':['A','STOP'],'source_hash':'source'};old={'decision_model_hash':'old','history':[{'reward':.8}],'unknowns':['unassessed']}
    current={**old,'decision_model_hash':'new'}
    actual=worker.compare_reference(p,current,outcome,reference,p,old)
    assert actual['exact_metrics_except_policy_identity'] and actual['no_quality_loss_vs_saved_exhaustive']
    for key,value in [('unknowns',[]),('history',[{'reward':.7}])]:
        assert not worker.compare_reference(p,{**current,key:value},outcome,reference,p,old)['exact_metrics_except_policy_identity']
    worse={**outcome,'target_removed_mm3':.5,'total_reward':.3}
    negative=worker.compare_reference(p,current,worse,reference,p,old)
    assert not negative['exact_outcomes_vs_balanced'] and not negative['no_quality_loss_vs_saved_exhaustive']


def test_false_release_refuses_before_metadata_or_payload(tmp_path,monkeypatch):
    path=tmp_path/'false.json';path.write_text(json.dumps({'execution_released':False}))
    monkeypatch.setattr(contract,'metadata',lambda:pytest.fail('false dispatch cannot load metadata'))
    with pytest.raises(ValueError,match='root_dispatch_required'):contract.guard(path,contract.sha(path),execute=True)


def test_actual_four_subject_worker_seals_all_before_uncached_replay(tmp_path,monkeypatch):
    """Actual factory/native/replay glue on four 7-cube invented cases.

    Historical references are deliberately different scalar DTOs, proving a
    measured negative regression is retained without a favorable-result gate.
    Only model bytes and historical metadata are synthetic replacements.
    """
    import torch
    from resectionlab import native_resection,geometry,native_spatial_evaluation
    prepared={s:generated_source(tmp_path,s) for s in contract.SUBJECTS}
    entries=[{'patient_id':s,'path':str(inputs['public_manifest_path']),'sha256':inputs['public_manifest_sha256']}
        for s,(inputs,_) in prepared.items()]
    idx=tmp_path/'public-index.json';idx.write_text(json.dumps({'cases':entries}))
    idx_ref={'path':str(idx),'sha256':contract.sha(idx)}
    worlds={s:built[0].source_hash for s,(_,built) in prepared.items()}
    model=RidgePriority(np.zeros(681),np.ones(681),np.zeros(681),0.)
    raw=json.dumps({'feature_version':model.fingerprint,'data_hash':contract.RIDGE['data_hash'],
        'mean':[0.]*681,'scale':[1.]*681,'weights':[0.]*681,'intercept':0.}).encode()
    fake={**contract.RIDGE,'sha256':hashlib.sha256(raw).hexdigest(),'fingerprint':model.fingerprint}
    for module in (contract,worker,capability):monkeypatch.setattr(module,'RIDGE',fake)
    monkeypatch.setattr(capability,'read_bytes',lambda path,expected,limit:raw)
    for module in (contract,worker):monkeypatch.setattr(module,'OUTPUT',tmp_path/'worker-output')
    hooks=[];monkeypatch.setattr(worker.sys,'addaudithook',hooks.append)
    monkeypatch.setattr(torch,'set_num_threads',lambda n:None);monkeypatch.setattr(torch,'set_num_interop_threads',lambda n:None)
    fake_rows=[]
    for method in ('balanced_ridge_top8','exhaustive'):
        for subject in contract.SUBJECTS:
            key=method+'/'+subject
            fake_rows.append({'method':method,'subject':subject,'outcomes':scalar_outcomes(),
                'files':{key+'/plan.json':'c'*64,key+'/replay/episode-metrics.json':'d'*64}})
    synthetic_source={'subjects':contract.SUBJECTS,'retention_reference':'return_plus_opening_depth_volume_v1'}
    monkeypatch.setattr(worker,'metadata',lambda:({'arms':fake_rows},{},synthetic_source))
    old_small=worker.small
    def synthetic_metadata(path,expected=None):
        if str(path).startswith(str(worker.TRAIN_OUTPUT)+'/'):
            if str(path).endswith('/plan.json'):return {'actions':[],'source_hash':'deliberately_different_reference'}
            return {'decision_model_hash':'historical_policy','history':[]}
        return old_small(path,expected)
    monkeypatch.setattr(worker,'small',synthetic_metadata)
    calls=[];original=native_spatial_evaluation.evaluate_native_spatial_episode
    def checked_evaluation(task,**kwargs):
        seals=json.loads((worker.OUTPUT/'sealed-plans.json').read_text())
        assert list(seals['plans'])==contract.SUBJECTS
        assert all((worker.OUTPUT/s/'plan.json').exists() for s in contract.SUBJECTS)
        assert native_resection.capsule_voxel_indices is geometry.capsule_voxel_indices
        calls.append(task.case.source_hash)
        return original(task,**kwargs)
    monkeypatch.setattr(native_spatial_evaluation,'evaluate_native_spatial_episode',checked_evaluation)
    release={'public_index':idx_ref,'source_world_hashes':worlds}
    result=worker.execute(release,'b'*64,{'blocked':0})
    contract.complete_result(result,release,'b'*64)
    assert len(calls)==4 and len(hooks)==1 and result['source_visits']==8
    assert result['all_plans_sealed_before_evaluation'] and result['ridge_unchanged'] and result['cache_restored']
    assert result['fits']==result['model_calls']==result['optimizer_calls']==result['checkpoint_loads']==0
    assert len(result['arms'])==4 and all(r['complete'] and r['independent_accepted'] for r in result['arms'])
    assert result['comparison']['regression_passed'] is False
    assert result['status']=='complete_fixed_four_switch_regression'
    assert result['native_transitions']==2*sum(r['action_count'] for r in result['arms'])
    assert result['native_previews']==sum(r['online_previews']+r['replay_previews'] for r in result['arms'])
