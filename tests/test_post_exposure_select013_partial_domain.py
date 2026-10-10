"""Generated-only SELECT Ds controls; source-written, execution separately owned."""
import copy
import importlib.util
import json
import os
from pathlib import Path
import sys

import numpy as np
import pytest

from resectionlab import patient_planning_admission as admission
from resectionlab import patient_select_inference as inference
from resectionlab import public_patient_factory as factory
from resectionlab.core import array_digest
import test_post_exposure_select013 as full
import test_post_exposure_learning as training
import test_paired_train_occupancy as helper


def fixture(path,subject='ReMIND-013'):
    inputs=full.fixture(path,subject)
    m=json.loads(inputs['public_manifest_path'].read_text())
    Ds=np.ones((7,7,7),np.uint8);Ds[6,6,6]=0;Ds[4,3,3]=0
    p=path/'Ds.npy';np.save(p,Ds,allow_pickle=False)
    m['input_files']['supplied_support_domain']={'path':str(p),'sha256':factory.sha(p),
        'bytes':p.stat().st_size,'dtype':str(Ds.dtype)}
    m.update(schema='remind-fixed-SELECT-partial-domain-public-inputs-v1',
        public_support_domain_fully_covered=False,training_admitted=False,
        source_domain_condition=admission.PARTIAL_DOMAIN_UNION_OCCUPANCY,
        public_source_domain_counts={'support_domain':341,
            'target_in_known_support_positive':1,'target_in_known_support_zero':0,
            'target_in_unknown_support_domain':1})
    full.save_manifest(inputs,m)
    return inputs,Ds


@pytest.mark.parametrize('updates',[8,64])
def test_explicit_partial_Ds_SELECT013_keeps_unknown_and_zero_gradient(tmp_path,updates):
    inputs,Ds=fixture(tmp_path);p=training.protocol(updates)
    built=full.build(tmp_path/'out',inputs,p,partial_domain_inputs=True)
    case,binding,qc,protocol=built;task,context=training.admit(built,inputs)
    record=inference.require_select_context(context)
    np.testing.assert_array_equal(case.support_domain,Ds.astype(bool))
    assert not case.support_domain[4,3,3] and case._interaction_domain[4,3,3]
    assert not case.support_domain[6,6,6] and not case._interaction_domain[6,6,6]
    assert case._domain_record['target_in_unknown_support_voxels']==1
    assert case._supplied_goal_extent['full_region_positive_voxels']==2
    assert binding['support_domain_binary_hash']==array_digest(Ds.astype(bool))
    assert binding['support_domain_source_sha256']==factory.sha(tmp_path/'Ds.npy')
    assert 'support_domain_derivation' not in binding
    assert 'support_domain_derivation' not in case.support_provenance
    assert qc['scope']==admission.PARTIAL_DOMAIN_SELECT_QC_SCOPE
    assert qc['partial_source_domain_preserved'] is True
    assert qc['native_domain_fully_covered'] is record['source_domain_fully_covered'] is False
    assert record['source_domain_input_kind']==admission.PARTIAL_DOMAIN_SELECT_INPUT_KIND
    assert record['execution_kind']==admission.POST_EXPOSURE_SELECT_EXECUTION
    assert record['max_optimizer_updates']==0 and protocol['role']=='SELECT'
    context.require_task(task.fresh());context.require_task(task.planning_clone())
    context.require_observations([task.observation()])
    with pytest.raises(ValueError,match='TRAIN_gradient'):context.require_training()
    assert case.post_exposure.record['seed_cells']>0 and not task._engine.removed_mask.any()
    assert task.step('STOP').reward==0


@pytest.mark.parametrize('change',['not_opted_in','missing_Ds','false_full_claim','wrong_schema','private_extra','037','067','wrong_role'])
def test_input_role_and_opt_in_refusals_before_payload(tmp_path,monkeypatch,change):
    subject={'037':'ReMIND-037','067':'ReMIND-067'}.get(change,'ReMIND-013')
    inputs,_=fixture(tmp_path,subject);m=json.loads(inputs['public_manifest_path'].read_text())
    kwargs={'partial_domain_inputs':True}
    if change=='not_opted_in':kwargs.clear()
    elif change=='missing_Ds':del m['input_files']['supplied_support_domain']
    elif change=='false_full_claim':m['public_support_domain_fully_covered']=True
    elif change=='wrong_schema':m['schema']='remind-fixed-TRAIN-partial-domain-public-inputs-v1'
    elif change=='private_extra':m['input_files']['private_ventricle']={'path':'never-open'}
    elif change=='wrong_role':kwargs['expected_role']='TRAIN'
    full.save_manifest(inputs,m)
    monkeypatch.setattr(factory.np,'load',lambda *a,**k:pytest.fail('must refuse before payload'))
    with pytest.raises(ValueError):full.build(tmp_path/'out',inputs,training.protocol(),**kwargs)


@pytest.mark.parametrize('change',['file_hash','binary_hash','domains','full_QC','scope','updates'])
def test_direct_admission_rebinding_is_not_a_domain_or_gradient_bypass(tmp_path,monkeypatch,change):
    inputs,_=fixture(tmp_path);built=full.build(tmp_path/'out',inputs,training.protocol(),partial_domain_inputs=True)
    case,args=helper.contract(built,inputs);args=copy.deepcopy(args)
    if change=='file_hash':args['source_binding']['support_domain_source_sha256']='0'*64
    elif change=='binary_hash':args['source_binding']['support_domain_binary_hash']='sha256:'+'0'*64
    elif change=='domains':args['source_binding']['source_and_simulated_domains']['source_domain_extended']=True
    elif change=='full_QC':args['qc_receipt']['native_domain_fully_covered']=True
    elif change=='scope':args['qc_receipt']['scope']=admission.PARTIAL_DOMAIN_QC_SCOPE
    else:args['protocol']['max_optimizer_updates']=1
    helper.rebind(args)
    monkeypatch.setattr(admission,'NativeSpatialTask',lambda *a,**k:pytest.fail('must refuse before inventory'))
    with pytest.raises(ValueError):admission.make_patient_planning_task(case,**args)


def test_four_array_default_exact_baseline_parity(tmp_path,monkeypatch):
    path=os.environ.get('PARTIAL_SELECT_BASELINE_DIR')
    if path is None:pytest.skip('explicit pinned historical source comparison unavailable')
    old={}
    for name in ('public_patient_factory','patient_planning_admission'):
        spec=importlib.util.spec_from_file_location('resectionlab._before_partial_SELECT_'+name,Path(path)/(name+'.py'))
        module=importlib.util.module_from_spec(spec);monkeypatch.setitem(sys.modules,spec.name,module)
        spec.loader.exec_module(module);old[name]=module
    inputs=full.fixture(tmp_path);p=training.protocol()
    current=full.build(tmp_path/'current',inputs,p)
    with monkeypatch.context() as m:
        m.setattr(full,'factory',old['public_patient_factory'])
        previous=full.build(tmp_path/'previous',inputs,p)
    assert current[0].source_hash==previous[0].source_hash and current[1:]==previous[1:]
    task,context=training.admit(current,inputs)
    case,args=helper.contract(previous,inputs)
    oldtask,oldcontext=old['patient_planning_admission'].make_patient_planning_task(case,**args)
    assert context.record()==oldcontext.record()
    assert task.observation().fingerprint==oldtask.observation().fingerprint
    assert task.candidate_inventory()==oldtask.candidate_inventory()
    assert {p.name:p.read_bytes() for p in (tmp_path/'current').iterdir()}=={p.name:p.read_bytes() for p in (tmp_path/'previous').iterdir()}
