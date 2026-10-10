"""Generated contract fixtures only; no real arrays or patient execution."""
from pathlib import Path
from dataclasses import replace
import copy,importlib.util,json,os
import numpy as np
import pytest
from resectionlab import public_patient_factory as factory
from resectionlab import patient_planning_admission as admission
from resectionlab.core import semantic_digest,thaw_json
from resectionlab.public_target_context import VERSION
ROOT=next(p for p in Path(__file__).resolve().parents if (p/'src/resectionlab').is_dir())
spec=importlib.util.spec_from_file_location('generated_occupancy_fixture',ROOT/'tests/test_paired_train_occupancy.py')
helper=importlib.util.module_from_spec(spec);spec.loader.exec_module(helper)
CONDITION=admission.PARTIAL_DOMAIN_UNION_OCCUPANCY

def fixture(tmp_path,subject='ReMIND-002'):
    inputs,arrays=helper.fixture(tmp_path,subject=subject)
    p=inputs['public_manifest_path'];manifest=json.loads(p.read_bytes())
    ds=np.ones(arrays[0].shape,np.uint8);ds[0,0,0]=0;ds[4,2,2]=0
    path=tmp_path/'support-domain.npy';np.save(path,ds,allow_pickle=False)
    manifest.update(public_support_domain_fully_covered=False,source_domain_condition=CONDITION,
        training_admitted=False,public_source_domain_counts={'support_domain':123,
        'target_in_known_support_positive':1,'target_in_known_support_zero':0,'target_in_unknown_support_domain':1})
    manifest['input_files']['supplied_support_domain']={'path':str(path),'sha256':factory.sha(path),'bytes':path.stat().st_size,'dtype':'uint8'}
    p.write_text(json.dumps(manifest));inputs['public_manifest_sha256']=factory.sha(p)
    return inputs,arrays,manifest

def build(tmp_path,inputs,*,updates=0,forwards=0,condition=CONDITION):
    tmp_path.mkdir()
    limits={'max_steps':2,'max_optimizer_updates':updates,'max_native_previews':256,'max_policy_forwards':forwards,
        'worker_seconds':10,'memory_bytes':512*1024**2,'threads':1,'search':{'max_calls':2,'beam_width':1,'seconds':1}}
    return factory.prepare_public_source(tmp_path,{'limits':limits},'1'*64,None,lambda *a,**k:None,
        **inputs,learning_protocol_hash='sha256:'+'2'*64,public_target_context_variant=VERSION,occupancy_condition=condition)

def admit(result,inputs):
    source,binding,qc,protocol=result
    return admission.make_patient_planning_task(source,cohort_bytes=inputs['cohort_bytes'],source_binding=binding,qc_receipt=qc,protocol=protocol)

def rebind(binding,qc,protocol):
    qc['public_source_binding_hash']=protocol['public_source_binding_hash']=semantic_digest(binding)
    protocol['qc_receipt_hash']=semantic_digest(qc)

@pytest.mark.parametrize('subject',admission.PARTIAL_DOMAIN_TRAIN_SUBJECTS)
def test_exact_new4_partial_domain_search_only_with_full_target(tmp_path,subject):
    inputs,arrays,manifest=fixture(tmp_path,subject)
    source,binding,qc,protocol=build(tmp_path/'run',inputs)
    assert not source.support_domain[4,2,2] and source._interaction_domain[4,2,2]
    assert source._domain_record['target_in_unknown_support_voxels']==1
    assert source._supplied_goal_extent['full_region_positive_voxels']==2
    np.testing.assert_array_equal(source.nominal_target,arrays[2])
    np.testing.assert_array_equal(source.occupancy_source_support,arrays[1])
    assert qc['native_domain_fully_covered'] is False and qc['partial_source_domain_preserved'] is True
    task,context=admit((source,binding,qc,protocol),inputs)
    assert context.record()['execution_kind']=='search_only_no_policy'
    assert context.record()['source_and_simulated_domains']['target_in_unknown_support_voxels']==1
    assert context.record()['policy_comparison_permitted'] is False
    with pytest.raises(ValueError,match='gradient_budget'):context.require_training()
    obs=task.observation();assert not obs.coverage[1,4,2,2] and obs.image_channels[2,4,2,2]==1

@pytest.mark.parametrize('subject',['ReMIND-008','ReMIND-013','ReMIND-067'])
def test_old_or_protected_person_cannot_enter_new_branch(tmp_path,subject):
    inputs,_,_=fixture(tmp_path,subject)
    with pytest.raises(ValueError):build(tmp_path/'run',inputs)

@pytest.mark.parametrize('change',['default_condition','updates','forwards','full_coverage','missing_Ds','partition'])
def test_explicit_branch_and_authentication_refusals(tmp_path,change):
    inputs,_,m=fixture(tmp_path);kwargs={}
    if change=='default_condition':kwargs['condition']='raw_cerebrum_baseline'
    elif change in ('updates','forwards'):kwargs[change]=1
    elif change=='full_coverage':m['public_support_domain_fully_covered']=True
    elif change=='missing_Ds':del m['input_files']['supplied_support_domain']
    else:m['public_source_domain_counts']['target_in_unknown_support_domain']=0
    if change in ('full_coverage','missing_Ds','partition'):
        inputs['public_manifest_path'].write_text(json.dumps(m));inputs['public_manifest_sha256']=factory.sha(inputs['public_manifest_path'])
    with pytest.raises(ValueError):build(tmp_path/'run',inputs,**kwargs)

@pytest.mark.parametrize('change',['qc_full','qc_preserved','domain_file','domain_hash','domain_record','condition','updates','forwards'])
def test_admission_refuses_rebound_metadata_tampering(tmp_path,change):
    inputs,_,_=fixture(tmp_path);source,binding,qc,protocol=build(tmp_path/'run',inputs)
    if change=='qc_full':qc['native_domain_fully_covered']=True
    elif change=='qc_preserved':qc['partial_source_domain_preserved']=False
    elif change=='domain_file':binding['support_domain_source_sha256']='0'*64
    elif change=='domain_hash':binding['support_domain_binary_hash']='sha256:'+'0'*64
    elif change=='domain_record':binding['source_and_simulated_domains']['source_domain_extended']=True
    elif change=='condition':protocol['occupancy_condition']='cerebrum_plus_supplied_tumor_assumption'
    elif change=='updates':protocol['max_optimizer_updates']=1
    else:protocol['max_policy_forwards']=1
    rebind(binding,qc,protocol)
    with pytest.raises(ValueError):admit((source,binding,qc,protocol),inputs)

def test_new_subject_never_uses_old_full_coverage_path(tmp_path):
    inputs,arrays=helper.fixture(tmp_path,subject='ReMIND-002')
    with pytest.raises(ValueError):build(tmp_path/'run',inputs,condition='cerebrum_plus_supplied_tumor_assumption')

@pytest.mark.parametrize('condition',[helper.RAW,helper.UNION])
def test_generated_legacy_source_protocol_observation_inventory_context_parity(tmp_path,condition,monkeypatch):
    import sys
    from resectionlab.native_spatial_task import NativeSpatialTask
    baseline=Path(os.environ.get('PARTIAL_DOMAIN_BASELINE_DIR',Path(__file__).resolve().parents[1]/'baseline/resectionlab'))
    if not baseline.is_dir():pytest.skip('historical comparison requires explicit PARTIAL_DOMAIN_BASELINE_DIR')
    old={}
    for name in ('public_patient_factory','patient_planning_admission'):
        module_name='resectionlab._partial_domain_baseline_'+name
        spec=importlib.util.spec_from_file_location(module_name,baseline/(name+'.py'))
        module=importlib.util.module_from_spec(spec);monkeypatch.setitem(sys.modules,module_name,module);spec.loader.exec_module(module);old[name]=module
    inputs,_=helper.fixture(tmp_path)
    a=helper.build(tmp_path/'current',inputs,condition=condition)
    b=helper.build(tmp_path/'old',inputs,condition=condition,module=old['public_patient_factory'])
    assert a[0].source_hash==b[0].source_hash and a[1:]==b[1:]
    x,c=admit(a,inputs)
    y,d=old['patient_planning_admission'].make_patient_planning_task(b[0],cohort_bytes=inputs['cohort_bytes'],source_binding=b[1],qc_receipt=b[2],protocol=b[3])
    assert c.record()==d.record() and c.fingerprint==d.fingerprint
    assert x.decision_model_hash==y.decision_model_hash
    assert x.observation().fingerprint==y.observation().fingerprint
    assert x.candidate_inventory()==y.candidate_inventory()
    for filename in ('public-task-derivation.json','admitted-public-bindings.json'):
        assert (tmp_path/'current'/filename).read_bytes()==(tmp_path/'old'/filename).read_bytes()
