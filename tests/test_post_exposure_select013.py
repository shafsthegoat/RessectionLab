"""Generated public geometry/lineage only; no acquired payload or trained model.

Checkpoint endpoint metadata is fabricated for interface coverage, not evidence
of completed learning. Test execution requires the owner's separate compute slot.
"""
import copy
from dataclasses import replace
import importlib.util
import json
import os
from pathlib import Path
import sys

import numpy as np
import pytest

from resectionlab.core import array_digest, semantic_digest, thaw_json
from resectionlab import patient_planning_admission as admission
from resectionlab import patient_select_inference as inference
from resectionlab import public_patient_factory as factory
from resectionlab.native_proposals import NominalCavityProposalConfig
from resectionlab.patient_planning_cohort_spec import POST_EXPOSURE_TRAIN
from resectionlab.post_exposure import VERSION as EXPOSURE
from resectionlab.public_target_context import VERSION as TARGET_CONTEXT
import test_paired_train_occupancy as helper
import test_post_exposure_learning as training
import test_obstruction_opening_select013 as prior


def fixture(path, subject='ReMIND-013'):
    inputs, _ = helper.fixture(path, subject=subject, role='SELECT')
    m = json.loads(inputs['public_manifest_path'].read_text())
    image = np.arange(343, dtype=np.float32).reshape(7,7,7)
    S = np.zeros(image.shape, np.uint8); S[1:4,2:5,2:5] = 1
    T = np.zeros_like(S); T[3:5,3,3] = 1
    Dt = np.ones_like(S); Dt[0,0,0] = 0
    for key, value in zip(factory.ARRAY_KEYS, (image,S,T,Dt)):
        p = path/(key+'.npy'); np.save(p,value,allow_pickle=False)
        m['input_files'][key] = {'path':str(p),'sha256':factory.sha(p),
            'bytes':p.stat().st_size,'dtype':str(value.dtype)}
    m.update(shape_xyz=list(S.shape), public_grid_selection={
        'all_output_cell_corners_inside_public_source_domain':True},
        public_label_resampling={'cerebrum':{'saved_volume_summary':{'saved_domain_voxels':S.size}}})
    save_manifest(inputs,m)
    return inputs


def save_manifest(inputs, record):
    inputs['public_manifest_path'].write_text(json.dumps(record))
    inputs['public_manifest_sha256'] = factory.sha(inputs['public_manifest_path'])


def lineage(p, method=None):
    record = prior.lineage(p,method or ('IL' if p['updates_per_method']==64 else 'RL'))
    record['training_context_hashes'] = {'ReMIND:'+s[-3:]:'sha256:'+'f'*64 for s in POST_EXPOSURE_TRAIN}
    return record


def build(path, inputs, p, *, checkpoint_lineage=None, **changes):
    path.mkdir()
    limits = {'max_steps':24,'max_optimizer_updates':0,'max_native_previews':10000,
        'max_policy_forwards':48,'worker_seconds':300,'memory_bytes':3*1024**3,'threads':1,
        'search':thaw_json(p['cohort_execution']['search'])}
    options = dict(learning_protocol_hash=semantic_digest(p),
        proposal_config=NominalCavityProposalConfig(**thaw_json(p['cohort_execution']['proposal_config'])),
        public_target_context_variant=TARGET_CONTEXT,
        occupancy_condition=admission.PARTIAL_DOMAIN_UNION_OCCUPANCY,
        occupancy_inference_protocol=p, post_exposure_condition=EXPOSURE,
        expected_role='SELECT', checkpoint_lineage=checkpoint_lineage or lineage(p))
    options.update(changes)
    return factory.prepare_public_source(path,{'limits':limits},'b'*64,None,
        lambda *a,**k:None,**inputs,**options)


@pytest.mark.parametrize('updates',[8,64])
def test_exact_post_exposure_SELECT013_full_domain_and_zero_training(tmp_path,updates):
    inputs=fixture(tmp_path);p=training.protocol(updates)
    built=build(tmp_path/'out',inputs,p);case,binding,qc,plan=built
    task,context=training.admit(built,inputs)
    record=inference.require_select_context(context)
    assert record['execution_kind']==admission.POST_EXPOSURE_SELECT_EXECUTION
    assert record['source_domain_fully_covered'] is qc['native_domain_fully_covered'] is True
    assert set(record['checkpoint_lineage']['training_context_hashes'])=={'ReMIND:'+s[-3:] for s in POST_EXPOSURE_TRAIN}
    assert record['max_optimizer_updates']==0 and qc['derived_full_source_domain_preserved'] is True
    assert np.all(case.support_domain) and np.all(case._interaction_domain)
    assert not case.public_target_domain[0,0,0]
    assert case.post_exposure.record['external_unknown_cells']==0
    assert case.post_exposure.record['seed_cells']>0 and not case.post_exposure.external_workspace.any()
    assert not task._engine.removed_mask.any() and task.metrics()['target_removed_mm3']==0
    assert case._supplied_goal_extent['full_region_positive_voxels']==2
    assert case._occupancy_derivation['added_region_positive_voxels']==1
    assert binding['support_domain_derivation']['acquired_domain_file'] is False
    assert 'support_domain_source_sha256' not in binding and 'support_domain_file_sha256' not in case.support_provenance
    assert binding['support_domain_derivation']['public_manifest_sha256']==inputs['public_manifest_sha256']
    context.require_task(task.fresh());context.require_task(task.planning_clone())
    context.require_observations([task.observation()])
    with pytest.raises(ValueError,match='TRAIN_gradient'):context.require_training()
    # All initial cavity is a declared simulated start; no removal or reward.
    assert task.step('STOP').reward==0
    again=build(tmp_path/'again',inputs,p)
    assert case.source_hash==again[0].source_hash and built[1:]==again[1:]


@pytest.mark.parametrize('subject',['ReMIND-037','ReMIND-067',*POST_EXPOSURE_TRAIN,'ReMIND-008'])
def test_no_other_person_before_payload(tmp_path,monkeypatch,subject):
    inputs=fixture(tmp_path,subject)
    monkeypatch.setattr(factory.np,'load',lambda *a,**k:pytest.fail('payload must remain closed'))
    with pytest.raises(ValueError):build(tmp_path/'out',inputs,training.protocol())


@pytest.mark.parametrize('change',['full_flag','corners','known_count','private','fifth_file'])
def test_coverage_and_input_boundary_before_payload(tmp_path,monkeypatch,change):
    inputs=fixture(tmp_path);m=json.loads(inputs['public_manifest_path'].read_text())
    if change=='full_flag':m['public_support_domain_fully_covered']=False
    elif change=='corners':m['public_grid_selection']['all_output_cell_corners_inside_public_source_domain']=False
    elif change=='known_count':m['public_label_resampling']['cerebrum']['saved_volume_summary']['saved_domain_voxels']-=1
    elif change=='private':m['private_evaluation_files_included']=True
    else:m['input_files']['supplied_support_domain']={'path':'never-open'}
    save_manifest(inputs,m)
    monkeypatch.setattr(factory.np,'load',lambda *a,**k:pytest.fail('payload must remain closed'))
    with pytest.raises(ValueError):build(tmp_path/'out',inputs,training.protocol())


@pytest.mark.parametrize('change',['old_groups','IL8','RL64','updates','role','no_exposure','old_world','obstruction','horizon','private_protocol'])
def test_wrong_endpoint_or_world_refused_before_payload(tmp_path,monkeypatch,change):
    inputs=fixture(tmp_path);p=training.protocol(64 if change=='RL64' else 8);options={}
    if change=='old_groups':options['checkpoint_lineage']=prior.lineage(p,'RL')
    elif change=='IL8':options['checkpoint_lineage']=lineage(p,'IL')
    elif change=='RL64':options['checkpoint_lineage']=lineage(p,'RL')
    elif change=='updates':
        value=lineage(p);value['optimizer_updates_on_SELECT']=1;options['checkpoint_lineage']=value
    elif change=='role':options['expected_role']='TRAIN'
    elif change=='no_exposure':options['post_exposure_condition']=None
    elif change=='old_world':options['occupancy_condition']=helper.UNION
    elif change=='obstruction':options['proposal_config']=NominalCavityProposalConfig(max_candidates=120,
        intermediate_opening_mm=1.,tool_footprint_opening=True,obstruction_opening=True)
    else:
        p=thaw_json(p)
        if change=='horizon':p['cohort_execution']['max_steps']=6
        else:p['private_training_reward']=True
    monkeypatch.setattr(factory.np,'load',lambda *a,**k:pytest.fail('payload must remain closed'))
    with pytest.raises(ValueError):build(tmp_path/'out',inputs,p,**options)


@pytest.mark.parametrize('change',['coverage','derivation','fake_file','private_reference','source_domain','condition_hash','forwards','update'])
def test_direct_admission_refuses_rebound_bad_claims(tmp_path,monkeypatch,change):
    inputs=fixture(tmp_path);built=build(tmp_path/'out',inputs,training.protocol())
    case,args=helper.contract(built,inputs);args=copy.deepcopy(args)
    if change=='coverage':args['qc_receipt']['native_domain_fully_covered']=False
    elif change=='derivation':args['source_binding']['support_domain_derivation']['source_domain_extended']=True
    elif change=='fake_file':args['source_binding']['support_domain_source_sha256']='a'*64
    elif change=='private_reference':case=replace(case,reference_target=np.zeros_like(case.reference_target))
    elif change=='source_domain':args['source_binding']['support_domain_binary_hash']='sha256:'+'0'*64
    elif change=='condition_hash':args['protocol']['post_exposure_condition_hash']='sha256:'+'0'*64
    elif change=='forwards':args['protocol']['max_policy_forwards']=0
    else:args['protocol']['max_optimizer_updates']=1
    helper.rebind(args)
    monkeypatch.setattr(admission,'NativeSpatialTask',lambda *a,**k:pytest.fail('must refuse before inventory'))
    with pytest.raises(ValueError):admission.make_patient_planning_task(case,**args)


def test_empty_fixed_seed_refuses_no_access_fallback(tmp_path,monkeypatch):
    inputs=fixture(tmp_path);m=json.loads(inputs['public_manifest_path'].read_text())
    support=np.zeros((7,7,7),np.uint8);support[:,2:5,2:5]=1
    p=Path(m['input_files']['supplied_support']['path']);np.save(p,support,allow_pickle=False)
    m['input_files']['supplied_support']['sha256']=factory.sha(p)
    m['public_target_support_consistency']['whole_tumor_positive_outside_supplied_support']=0
    save_manifest(inputs,m)
    from resectionlab import post_exposure
    original=post_exposure.prepare_post_exposure;calls=[]
    def once(**kwargs):
        calls.append(kwargs['previous_access'])
        return original(**kwargs)
    monkeypatch.setattr(post_exposure,'prepare_post_exposure',once)
    with pytest.raises(ValueError,match='POST_EXPOSURE_EMPTY_FIXED_K'):
        build(tmp_path/'out',inputs,training.protocol())
    assert len(calls)==1
    assert not (tmp_path/'out'/'public-task-derivation.json').exists()


@pytest.mark.parametrize('updates',[8,64])
def test_old_union_SELECT_records_exact_baseline_parity(tmp_path,monkeypatch,updates):
    baseline_path=os.environ.get('POST_SELECT_BASELINE_DIR')
    if baseline_path is None:
        pytest.skip('separately pinned historical source comparison unavailable')
    baseline=Path(baseline_path)
    old={}
    for name in ('public_patient_factory','patient_planning_admission'):
        spec=importlib.util.spec_from_file_location('resectionlab._pre_post_select_'+name,baseline/(name+'.py'))
        module=importlib.util.module_from_spec(spec);monkeypatch.setitem(sys.modules,spec.name,module)
        spec.loader.exec_module(module);old[name]=module
    inputs,_=helper.fixture(tmp_path,subject='ReMIND-013',role='SELECT')
    p=prior.training.protocol(updates)
    current=prior.build(tmp_path/'current',inputs,p)
    with monkeypatch.context() as m:
        m.setattr(prior,'factory',old['public_patient_factory'])
        previous=prior.build(tmp_path/'previous',inputs,p)
    task,context=prior.training.admit(current,inputs)
    case,args=helper.contract(previous,inputs);oldtask,oldcontext=old['patient_planning_admission'].make_patient_planning_task(case,**args)
    assert current[0].source_hash==previous[0].source_hash and current[1:]==previous[1:]
    assert context.record()==oldcontext.record()
    assert task.observation().fingerprint==oldtask.observation().fingerprint
    assert task.candidate_inventory()==oldtask.candidate_inventory()
    assert {p.name:p.read_bytes() for p in (tmp_path/'current').iterdir()}=={p.name:p.read_bytes() for p in (tmp_path/'previous').iterdir()}


@pytest.mark.parametrize('updates',[8,64])
def test_generated_new4_checkpoint_reload_SELECT_greedy_and_SEARCH_replay(tmp_path,monkeypatch,updates):
    import torch
    from resectionlab.patient_planning_learning import common_patient_policies
    from resectionlab.patient_planning_cohort_io import OutputBudget,_save_checkpoint
    from resectionlab.spatial_policy import parameter_hash
    torch.set_num_threads(1);p=training.protocol(updates);method='IL' if updates==64 else 'RL'
    contexts=[];train_hashes=[]
    for subject in POST_EXPOSURE_TRAIN:
        path=tmp_path/subject;path.mkdir();inputs=training.fixture(path,subject)
        task,ctx=training.admit(training.build(path/'source',inputs,p),inputs)
        contexts.append(ctx);train_hashes.append(task.case.source_hash)
    il,rl=common_patient_policies(contexts,p);policy=il if method=='IL' else rl;initial=parameter_hash(policy)
    with torch.no_grad():
        for parameter in policy.parameters():parameter.zero_()
        policy.stop[-1].bias.fill_(5.)
    saved=_save_checkpoint(policy,method=method,contexts=contexts,protocol=p,updates=updates,
        initial_hash=initial,output=OutputBudget(tmp_path,16*1024**2),limits={'checkpoint_bytes':8*1024**2})
    checkpoint=inference.load_select_checkpoint(tmp_path/saved['path'],expected_sha256=saved['sha256'],
        expected_learning_protocol=p,expected_context_hashes={c.patient_group:c.fingerprint for c in contexts},
        expected_method=method,training_release_sha256='b'*64)
    folder=tmp_path/'public013';folder.mkdir();inputs=fixture(folder)
    task,context=training.admit(build(folder/'source',inputs,p,checkpoint_lineage=checkpoint.lineage_record()),inputs)
    assert task.case.source_hash not in train_hashes  # Person/domain hashes need not match.
    checkpoint.require_task(task,context);before=parameter_hash(checkpoint.policy)
    with pytest.raises(ValueError):context.require_training()
    with pytest.raises(ValueError):common_patient_policies([context]*4,p)
    monkeypatch.setattr(torch.optim,'Adam',lambda *a,**k:pytest.fail('no SELECT optimizer'))
    out=tmp_path/'greedy';out.mkdir()
    trace=inference.collect_select_greedy(task,context,checkpoint,output=out,guard=lambda:None)
    result=inference.seal_and_replay_select(task,context,trace,checkpoint,output=out,guard=lambda:None)
    assert [t.action_id for t in trace.transitions]==['STOP']
    assert result['independent_geometry']['accepted'] is True
    assert result['select_inference']['optimizer_updates_on_SELECT']==0
    assert parameter_hash(checkpoint.policy)==before
    assert all(not param.requires_grad and param.grad is None for param in checkpoint.policy.parameters())
    monkeypatch.setattr(checkpoint.policy,'forward',lambda *a,**k:pytest.fail('SEARCH cannot use model'))
    actions,accounting=task.observed_greedy_search(seconds=10)
    out=tmp_path/'search';out.mkdir()
    trace=inference.collect_select_search(task,context,checkpoint,actions=actions,accounting=accounting,output=out,guard=lambda:None)
    result=inference.seal_and_replay_select(task,context,trace,checkpoint,output=out,guard=lambda:None)
    assert trace.method=='SEARCH' and trace.behavior_parameter_hash is None
    assert result['plan']['learning_updates']==0 and result['plan']['parameter_hash'] is None
    assert result['independent_geometry']['accepted'] is True
    assert task.metrics()['steps']==0 and parameter_hash(checkpoint.policy)==before
    with torch.no_grad():next(checkpoint.policy.parameters()).add_(.01)
    with pytest.raises(ValueError,match='changed'):checkpoint.require_task(task,context)
