"""Generated-only controls. No acquired payload or saved fitted weight is read."""
import copy,hashlib,importlib.util,json,sys
from pathlib import Path
import numpy as np
import pytest
HERE=Path(__file__).resolve().parents[1]
ROOT=next(p for p in HERE.parents if (p/'src/resectionlab').is_dir())
sys.path.insert(0,str(HERE));sys.path.insert(0,str(ROOT/'tests'))
from resectionlab import patient_planning_admission as old_admission,public_patient_factory as old_factory
from resectionlab.core import semantic_digest,thaw_json
from resectionlab.native_resection import NativeResectionEngine
from resectionlab.native_spatial_task import NativeSpatialTask
from resectionlab.native_proposals import NominalCavityProposalConfig
from resectionlab.public_target_context import VERSION as TARGET_CONTEXT
from resectionlab.post_exposure import VERSION as EXPOSURE
from batch_contract import transfer_record,PROPOSALS,RIDGE,VERSION,template
import batch_contract
import patient_transfer
from hybrid_proposal import RidgePriority

@pytest.fixture
def modules(monkeypatch):
    import resectionlab
    result={}
    for name in ('patient_planning_admission','public_patient_factory'):
        full='resectionlab.'+name
        spec=importlib.util.spec_from_file_location(full,HERE/'stage/resectionlab'/(name+'.py'))
        module=importlib.util.module_from_spec(spec);monkeypatch.setitem(sys.modules,full,module)
        monkeypatch.setattr(resectionlab,name,module);spec.loader.exec_module(module);result[name]=module
    return result['patient_planning_admission'],result['public_patient_factory']

def generated_source(tmp_path,modules,monkeypatch):
    # Existing synthetic fixture creates only invented 7-cube public arrays.
    import test_post_exposure_learning as generated
    admission,factory=modules
    monkeypatch.setattr(generated,'factory',factory)
    inputs=generated.fixture(tmp_path,'ReMIND-050')
    manifest=json.loads(inputs['public_manifest_path'].read_bytes())
    manifest['schema']='remind-fixed-TRAIN-partial-domain-public-inputs-v1'
    manifest['source_bindings']['intended_use_review_sha256']='e'*64
    inputs['public_manifest_path'].write_text(json.dumps(manifest))
    inputs['public_manifest_sha256']=factory.sha(inputs['public_manifest_path'])
    d=transfer_record({'public_manifest':{'path':str(inputs['public_manifest_path']),'sha256':inputs['public_manifest_sha256']},
        'intended_use_review':{'path':'invented-review','sha256':'e'*64}})
    limits={'max_steps':24,'max_optimizer_updates':0,'max_native_previews':5000,'max_policy_forwards':0,
        'worker_seconds':20,'memory_bytes':512*1024**2,'threads':1,'search':{'max_calls':1,'beam_width':1,'seconds':1}}
    dest=tmp_path/'source';dest.mkdir()
    built=factory.prepare_public_source(dest,{'limits':limits},'b'*64,None,lambda *a,**k:None,**inputs,
        learning_protocol_hash=semantic_digest(d),proposal_config=NominalCavityProposalConfig(**PROPOSALS),
        public_target_context_variant=TARGET_CONTEXT,occupancy_condition=admission.PARTIAL_DOMAIN_UNION_OCCUPANCY,
        post_exposure_condition=EXPOSURE,development_transfer=d)
    case,binding,qc,protocol=built
    args={'cohort_bytes':inputs['cohort_bytes'],'source_binding':binding,'qc_receipt':qc,'protocol':protocol}
    return case,args,inputs

def test_pending_inputs_refuse_without_model_or_patient_open(monkeypatch):
    monkeypatch.setattr(batch_contract,'read_bytes',lambda *a,**k:pytest.fail('unexpected payload'))
    with pytest.raises(ValueError,match='pending'):transfer_record({'public_manifest':None,'intended_use_review':None})

@pytest.mark.parametrize('key,value',[('subject','ReMIND-002'),('role','SELECT'),('K',7),('max_steps',True),
    ('refits',1),('optimizer_updates',1),('neural_forwards',1),('intended_use_review_sha256',None)])
def test_contract_refusals(modules,key,value):
    admission,_=modules
    d=transfer_record({'public_manifest':{'sha256':'a'*64},'intended_use_review':{'sha256':'e'*64}});d[key]=value
    with pytest.raises(ValueError):admission.validate_development_transfer(d)

def test_generated_050_source_admitted_without_preview_and_old_default_stays_closed(tmp_path,modules,monkeypatch):
    def forbidden(*a,**k):pytest.fail('preview during source validation')
    monkeypatch.setattr(NativeResectionEngine,'preview_stroke',forbidden)
    case,args,inputs=generated_source(tmp_path,modules,monkeypatch)
    modules[0].validate_patient_planning_source(case,**args)
    with pytest.raises(ValueError):old_admission.validate_patient_planning_source(case,**args)
    with pytest.raises(ValueError):old_factory.load_public_manifest(inputs['public_manifest_path'],inputs['public_manifest_sha256'],inputs['cohort_bytes'],partial_domain=True)
    # S/T/Ds identities survive; target outside Ds remains present, not filled away.
    assert case._supplied_goal_extent['full_region_positive_voxels']==2
    assert not case.support_domain[4,3,3] and case.nominal_target[4,3,3] and case._interaction_domain[4,3,3]
    token=patient_transfer.admit_source(case,**args,release_sha256='b'*64)
    token.require_source(case)
    assert args['protocol']['max_optimizer_updates']==args['protocol']['max_policy_forwards']==0

def test_generated_frozen_loader_and_complete_STOP_replay(tmp_path,modules,monkeypatch):
    case,args,_=generated_source(tmp_path,modules,monkeypatch)
    model=RidgePriority(np.zeros(681),np.ones(681),np.zeros(681),0.)
    model_record={'feature_version':model.fingerprint,'data_hash':RIDGE['data_hash'],'mean':[0.]*681,'scale':[1.]*681,'weights':[0.]*681,'intercept':0.}
    raw=json.dumps(model_record).encode();fake={**RIDGE,'sha256':hashlib.sha256(raw).hexdigest(),'fingerprint':model.fingerprint}
    monkeypatch.setattr(patient_transfer,'RIDGE',fake)
    monkeypatch.setattr(patient_transfer,'read_bytes',lambda path,expected,limit:raw)
    monkeypatch.setattr(RidgePriority,'fit',classmethod(lambda *a,**k:pytest.fail('no fitting')))
    ranker=patient_transfer.load_frozen_ridge();assert ranker.fingerprint==model.fingerprint
    args['protocol']['development_transfer'].update(frozen_ridge_sha256=fake['sha256'],frozen_ridge_fingerprint=fake['fingerprint'])
    args['protocol']['learning_protocol_hash']=semantic_digest(args['protocol']['development_transfer'])
    token=patient_transfer.admit_source(case,**args,release_sha256='b'*64)
    task=patient_transfer.AdmittedTransferTask(case,admission=token,mode='ridge_top8',ranker=ranker)
    assert task.costs.counts['native_previews']<=8
    task.step('STOP');assert task.terminated
    from resectionlab.native_spatial_evaluation import evaluate_native_spatial_episode
    report=evaluate_native_spatial_episode(task)
    assert report['accepted'] and report['outcomes']['nonstop_actions']==0

@pytest.mark.parametrize('key,value',[('subject','ReMIND-013'),('role','SELECT'),('max_optimizer_updates',1),('max_policy_forwards',1),('private_reference_used',True)])
def test_bound_source_rejects_tampered_protocol_before_inventory(tmp_path,modules,monkeypatch,key,value):
    case,args,_=generated_source(tmp_path,modules,monkeypatch)
    args['protocol'][key]=value
    monkeypatch.setattr(NativeResectionEngine,'preview_stroke',lambda *a,**k:pytest.fail('preview'))
    with pytest.raises(ValueError):patient_transfer.admit_source(case,**args,release_sha256='b'*64)

def test_default_generated_interface_identity_unchanged(modules):
    from test_patient_planning_admission import fixture_contract
    case,args=fixture_contract()
    a,ac=old_admission.make_patient_planning_task(case,**args)
    b,bc=modules[0].make_patient_planning_task(case,**args)
    assert ac.record()==bc.record() and ac.fingerprint==bc.fingerprint
    assert a.observation().fingerprint==b.observation().fingerprint and a.candidate_inventory()==b.candidate_inventory()
    a.step('STOP');b.step('STOP');assert a.metrics()==b.metrics()

def test_generated_full_two_arm_worker_seals_then_replays(tmp_path,modules,monkeypatch):
    """Actual native generated 7-cube + actual worker, no saved model or patient."""
    import initial_inventory_worker as worker
    import torch
    import test_post_exposure_learning as generated
    _,factory=modules;monkeypatch.setattr(generated,'factory',factory)
    folder=tmp_path/'fixture';folder.mkdir();inputs=generated.fixture(folder,'ReMIND-050')
    manifest=json.loads(inputs['public_manifest_path'].read_bytes())
    manifest['schema']='remind-fixed-TRAIN-partial-domain-public-inputs-v1'
    manifest['source_bindings']['intended_use_review_sha256']='e'*64
    inputs['public_manifest_path'].write_text(json.dumps(manifest))
    mr={'path':str(inputs['public_manifest_path']),'sha256':factory.sha(inputs['public_manifest_path'])}
    entry={'public_manifest':mr,'intended_use_review':{'path':'synthetic-review','sha256':'e'*64}}
    model=RidgePriority(np.zeros(681),np.ones(681),np.zeros(681),0.)
    raw=json.dumps({'feature_version':model.fingerprint,'data_hash':RIDGE['data_hash'],
        'mean':[0.]*681,'scale':[1.]*681,'weights':[0.]*681,'intercept':0.}).encode()
    fake={**RIDGE,'sha256':hashlib.sha256(raw).hexdigest(),'fingerprint':model.fingerprint}
    for module in (batch_contract,worker,patient_transfer):monkeypatch.setattr(module,'RIDGE',fake)
    monkeypatch.setattr(patient_transfer,'read_bytes',lambda path,expected,limit:raw)
    for module in (batch_contract,worker):monkeypatch.setattr(module,'OUTPUT',tmp_path/'actual-worker-output')
    # The outer generated test owner guards acquired paths. Retain, but do not
    # install, a permanent process-wide audit hook that would outlive this test.
    hooks=[];monkeypatch.setattr(worker.sys,'addaudithook',hooks.append)
    monkeypatch.setattr(torch,'set_num_threads',lambda n:None)
    monkeypatch.setattr(torch,'set_num_interop_threads',lambda n:None)
    release=template('generated',{'path':'generated','sha256':'c'*64},entry)
    release['execution_released']=True
    result=worker.execute(release,'b'*64,{'blocked':0})
    batch_contract.complete_result(result,release,'b'*64)
    assert len(hooks)==1 and result['source_visits']==4 and result['all_plans_sealed_before_evaluation']
    assert result['ridge_loads']==1 and result['ridge_unchanged'] and result['cache_restored']
    assert result['fits']==result['model_calls']==result['optimizer_calls']==0
    assert all(r['complete'] and r['independent_accepted'] for r in result['arms'])
    assert result['arms'][1]['online_costs']['counts'].get('feature_rows',0)==0
    assert result['arms'][1]['online_costs']['counts'].get('ridge_calls',0)==0
    out=worker.OUTPUT
    seals=json.loads((out/'sealed-plans.json').read_text())
    assert set(seals['plans'])==set(batch_contract.ARMS)
    assert all((out/m/'replay/independent-episode.json').exists() for m in batch_contract.ARMS)
