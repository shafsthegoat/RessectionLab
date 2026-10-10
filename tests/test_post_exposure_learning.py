"""Generated admission/lineage controls; no acquired arrays or fitted weights.

Fabricated annotation-style manifests exercise interfaces only, not source QC.
DTO-only teacher receipts below do not establish native replay; the owned runner
must authenticate/replay every pinned saved teacher before admitting its cache.
"""
from dataclasses import replace
import copy
import importlib.util
import json
import os
from pathlib import Path
import sys

import numpy as np
import pytest
import torch

from resectionlab.core import freeze_json, semantic_digest, thaw_json
from resectionlab import patient_planning_admission as admission
from resectionlab import public_patient_factory as factory
from resectionlab import patient_planning_learning as learning
from resectionlab.native_proposals import NominalCavityProposalConfig, DEFAULT_COLUMN_OFFSETS
from resectionlab.patient_planning_cohort_spec import (
    TRAIN, POST_EXPOSURE_TRAIN, POST_EXPOSURE_LEARNING_VERSION, POST_EXPOSURE_TEACHER_STEPS,
    CACHED_TEACHERS, BALANCED_TEACHER_CE, sequential_learning_protocol,
    sequential_limits, protocol_train_subjects, validate_sequential_protocol, validate_factories)
from resectionlab.post_exposure import VERSION as EXPOSURE
from resectionlab.public_target_context import VERSION as TARGET_CONTEXT
import test_paired_train_occupancy as helper


def protocol(updates=8, **changes):
    options = dict(updates=updates, max_steps=24,
        search={'max_calls':5640, 'beam_width':2, 'seconds':300},
        proposal_config=NominalCavityProposalConfig(max_candidates=120,
            intermediate_opening_mm=1., tool_footprint_opening=True),
        occupancy_condition=admission.PARTIAL_DOMAIN_UNION_OCCUPANCY,
        post_exposure_condition=EXPOSURE, retention_mode='return_plus_opening_depth_volume_v1')
    if updates == 64:
        options.update(il_teacher_weighting=BALANCED_TEACHER_CE,
            teacher_observations=CACHED_TEACHERS, teacher_cache_payload_bytes=256*1024**2)
    options.update(changes)
    return sequential_learning_protocol(**options)


def limits(p):
    return thaw_json(sequential_limits(p, worker_seconds=3540, memory_bytes=3*1024**3,
                                      output_bytes=128*1024**2))


def fixture(tmp_path, subject='ReMIND-002'):
    inputs, _ = helper.fixture(tmp_path, subject=subject)
    m = json.loads(inputs['public_manifest_path'].read_bytes())
    image = np.arange(343, dtype=np.float32).reshape(7,7,7)
    S = np.zeros(image.shape, np.uint8); S[1:4,2:5,2:5] = 1
    T = np.zeros_like(S); T[3:5,3,3] = 1
    Dt = np.ones_like(S); Dt[0,0,0] = 0
    Ds = Dt.copy(); Ds[4,3,3] = 0
    for key, values in zip(factory.PARTIAL_DOMAIN_ARRAY_KEYS, (image,S,T,Dt,Ds)):
        path = tmp_path/(key+'.npy'); np.save(path,values,allow_pickle=False)
        m['input_files'][key] = {'path':str(path),'sha256':factory.sha(path),
            'bytes':path.stat().st_size,'dtype':str(values.dtype)}
    m.update(shape_xyz=list(S.shape), public_support_domain_fully_covered=False,
        source_domain_condition=admission.PARTIAL_DOMAIN_UNION_OCCUPANCY, training_admitted=False,
        public_source_domain_counts={'support_domain':int(Ds.sum()),'target_in_known_support_positive':1,
            'target_in_known_support_zero':0,'target_in_unknown_support_domain':1})
    inputs['public_manifest_path'].write_text(json.dumps(m))
    inputs['public_manifest_sha256'] = factory.sha(inputs['public_manifest_path'])
    return inputs


def build(path, inputs, p, *, search_only=False, **changes):
    path.mkdir()
    bound = {k:v for k,v in limits(p).items() if k not in ('output_bytes','checkpoint_bytes')}
    if search_only: bound.update(max_optimizer_updates=0,max_policy_forwards=0)
    options = dict(learning_protocol_hash=semantic_digest(p),
        proposal_config=NominalCavityProposalConfig(**thaw_json(p['cohort_execution']['proposal_config'])),
        public_target_context_variant=TARGET_CONTEXT,
        occupancy_condition=admission.PARTIAL_DOMAIN_UNION_OCCUPANCY, post_exposure_condition=EXPOSURE)
    if not search_only: options['occupancy_learning_protocol'] = p
    options.update(changes)
    return factory.prepare_public_source(path,{'limits':bound},'b'*64,None,
        lambda *a,**k:None,**inputs,**options)


def admit(built, inputs):
    case,args = helper.contract(built,inputs)
    return admission.make_patient_planning_task(case,**args)


@pytest.mark.parametrize('updates',[8,64])
@pytest.mark.parametrize('subject',POST_EXPOSURE_TRAIN)
def test_new4_admission_changes_authority_not_tested_world(tmp_path, subject, updates):
    inputs = fixture(tmp_path,subject); p=protocol(updates)
    learned = build(tmp_path/'learned',inputs,p)
    searched = build(tmp_path/'search',inputs,p,search_only=True)
    assert learned[0].source_hash == searched[0].source_hash and learned[1:3] == searched[1:3]
    task,context=admit(learned,inputs); other,closed=admit(searched,inputs)
    context.require_training(); context.require_task(task.planning_clone()); context.require_task(task.fresh())
    assert task.decision_model_hash == other.decision_model_hash
    assert task.observation().fingerprint == other.observation().fingerprint
    assert task.candidate_inventory() == other.candidate_inventory()
    assert context.fingerprint != closed.fingerprint
    with pytest.raises(ValueError): closed.require_training()
    record=context.record()
    assert record['execution_kind']=='fixed_four_TRAIN_post_exposure_learning_v1'
    assert record['post_exposure']==thaw_json(task.case.post_exposure.record)
    assert record['derived_occupancy_anatomically_validated'] is False
    assert record['source_domain_fully_covered'] is False
    assert task.case._supplied_goal_extent['full_region_positive_voxels']==2
    assert not task.case.support_domain[4,3,3] and task.case._interaction_domain[4,3,3]
    assert task.step('STOP').terminated


@pytest.mark.parametrize('change',['no_exposure','old_occupancy','axis','obstruction','cap','horizon',
    'updates','cache_bound','teacher_steps','teacher_source','order','private','initialization','version'])
def test_protocol_refuses_unmatched_world_cohort_or_endpoint(change):
    p=thaw_json(protocol(64)); e=p['cohort_execution']
    if change=='no_exposure':e.pop('post_exposure_condition')
    elif change=='old_occupancy':e['occupancy_condition']='cerebrum_plus_supplied_tumor_assumption'
    elif change=='axis':e['proposal_config']['offsets_source_voxels'].append([-1,4])
    elif change=='obstruction':e['proposal_config']['obstruction_opening']=True
    elif change=='cap':e['proposal_config']['max_candidates']=96
    elif change=='horizon':e['max_steps']=6
    elif change=='updates':p['updates_per_method']=32
    elif change=='cache_bound':e['teacher_cache_payload_bytes']=64*1024**2
    elif change=='teacher_steps':e['teacher_steps']=[1,1,1,1]
    elif change=='teacher_source':e['teacher_source']='new_beam_search'
    elif change=='order':e['patient_order'][-1]='ReMIND-013'
    elif change=='private':p['private_training_reward']=True
    elif change=='initialization':p['initialization']='old_IL_checkpoint'
    else:p['version']='fixed-four-TRAIN-sequential-cohort-v2'
    with pytest.raises(ValueError):validate_sequential_protocol(p)


@pytest.mark.parametrize('subject',['ReMIND-008','ReMIND-025','ReMIND-013','ReMIND-067'])
def test_old_or_protected_subject_rejected_before_array_read(tmp_path,monkeypatch,subject):
    inputs=fixture(tmp_path,subject)
    monkeypatch.setattr(factory.np,'load',lambda *a,**k:pytest.fail('no source array read'))
    with pytest.raises(ValueError):build(tmp_path/'out',inputs,protocol())


@pytest.mark.parametrize('change',['exposure','protocol','initializer','update_budget','forward_budget','role','private','domain'])
def test_admission_rejects_rebound_mismatch_before_native_inventory(tmp_path,monkeypatch,change):
    inputs=fixture(tmp_path); built=build(tmp_path/'source',inputs,protocol())
    case,args=helper.contract(built,inputs);args=copy.deepcopy(args)
    if change=='exposure':args['protocol']['post_exposure_condition_hash']='sha256:'+'0'*64
    elif change=='protocol':args['protocol']['learning_protocol_hash']='sha256:'+'0'*64
    elif change=='initializer':args['protocol']['initialization']='checkpoint_warmstart'
    elif change=='update_budget':args['protocol']['max_optimizer_updates']=0
    elif change=='forward_budget':args['protocol']['max_policy_forwards']=0
    elif change=='role':args['protocol']['role']='SELECT'
    elif change=='private':args['protocol']['private_reference_used']=True
    else:args['source_binding']['source_and_simulated_domains']['source_domain_extended']=True
    helper.rebind(args)
    monkeypatch.setattr(admission,'NativeSpatialTask',lambda *a,**k:pytest.fail('no native inventory'))
    with pytest.raises(ValueError):admission.make_patient_planning_task(case,**args)


def generated_dtos(p, monkeypatch):
    import test_patient_cohort_sequential as generated
    monkeypatch.setattr(generated,'TRAIN',POST_EXPOSURE_TRAIN)
    contexts,observations=generated.generated_contexts(p)
    result=[]
    for context in contexts:
        r=context.record();r.update(occupancy_condition=admission.PARTIAL_DOMAIN_UNION_OCCUPANCY,
            post_exposure={'version':EXPOSURE},decision_model_hash=semantic_digest({'generated':context.patient_group}))
        result.append(admission.PatientPlanningContext(freeze_json(r),semantic_digest(r)))
    return tuple(result),observations


def test_exact29_cache_fresh_common_policy_and_readable_checkpoint(tmp_path,monkeypatch):
    from resectionlab.patient_teacher_trace_cache import PatientTeacherTraceCache
    from resectionlab.patient_planning_cohort_io import _save_checkpoint,load_cohort_checkpoint,OutputBudget
    from resectionlab.spatial_policy import parameter_hash
    from test_patient_cohort_sequential import synthetic_trace
    from test_patient_teacher_trace_cache import simulated_receipt
    torch.set_num_threads(1)
    p=protocol(64);contexts,observations=generated_dtos(p,monkeypatch)
    il,rl=learning.common_patient_policies(contexts,p)
    assert parameter_hash(il)==parameter_hash(rl)
    session=learning.PatientTrainSession(contexts,'IL',il,initial_parameter_hash=parameter_hash(il),protocol=p)
    assert not session._optimizer.state
    cache=PatientTeacherTraceCache(p)
    for context,obs,n in zip(contexts,observations,POST_EXPOSURE_TEACHER_STEPS):
        trace=synthetic_trace(context,obs,n)
        cache.add_replayed(trace,simulated_receipt(trace))
    record=cache.record()
    assert record['complete'] and sum(r['steps'] for r in record['traces'])==29
    assert sum(r['stop_steps'] for r in record['traces'])==4
    assert record['array_bytes']+record['metadata_json_bytes']<=256*1024**2
    for subject in POST_EXPOSURE_TRAIN:cache.trace_for_il(session,subject).require()
    other=learning.PatientTrainSession(contexts,'RL',rl,initial_parameter_hash=parameter_hash(rl),protocol=p)
    with pytest.raises(ValueError):cache.trace_for_il(other,POST_EXPOSURE_TRAIN[0])
    # Generated endpoint-format control only; no optimizer execution is claimed.
    saved=_save_checkpoint(il,method='IL',contexts=contexts,protocol=p,updates=64,
        initial_hash=parameter_hash(il),output=OutputBudget(tmp_path,16*1024**2),limits={'checkpoint_bytes':8*1024**2})
    assert saved['metadata']['TRAIN_subjects']==list(POST_EXPOSURE_TRAIN)
    restored,meta=load_cohort_checkpoint(tmp_path/saved['path'],expected_sha256=saved['sha256'],
        expected_learning_protocol=p,expected_context_hashes={c.patient_group:c.fingerprint for c in contexts},expected_method='IL')
    assert parameter_hash(restored)==parameter_hash(il)
    assert thaw_json(meta)==saved['metadata']
    with pytest.raises(ValueError):load_cohort_checkpoint(tmp_path/saved['path'],expected_sha256=saved['sha256'],
        expected_learning_protocol=p,expected_context_hashes={'ReMIND:'+s[-3:]:'sha256:'+'0'*64 for s in TRAIN},expected_method='IL')


@pytest.mark.parametrize('change',['order','missing_context','exposure','occupancy'])
def test_common_initialization_rejects_wrong_condition_before_model(monkeypatch,change):
    p=protocol();contexts,obs=generated_dtos(p,monkeypatch)
    if change=='order':contexts=tuple(reversed(contexts))
    elif change=='missing_context':contexts=contexts[:-1]
    else:
        r=contexts[0].record();r.pop('post_exposure' if change=='exposure' else 'occupancy_condition')
        contexts=(admission.PatientPlanningContext(freeze_json(r),semantic_digest(r)),*contexts[1:])
    monkeypatch.setattr(learning,'SpatialPolicy',lambda *a,**k:pytest.fail('no model allocation'))
    with pytest.raises(ValueError):learning.common_patient_policies(contexts,p)


def test_cache_refuses_incomplete_teacher_and_old_subject(monkeypatch):
    from resectionlab.patient_teacher_trace_cache import PatientTeacherTraceCache
    from test_patient_cohort_sequential import synthetic_trace
    from test_patient_teacher_trace_cache import simulated_receipt
    p=protocol(64);contexts,observations=generated_dtos(p,monkeypatch)
    cache=PatientTeacherTraceCache(p)
    first=synthetic_trace(contexts[0],observations[0],1);cache.add_replayed(first,simulated_receipt(first))
    wrong=synthetic_trace(contexts[1],observations[1],1)
    with pytest.raises(ValueError,match='decision counts'):cache.add_replayed(wrong,simulated_receipt(wrong))
    r=contexts[1].record();r.update(subject='ReMIND-010',patient_group='ReMIND:010')
    c=admission.PatientPlanningContext(freeze_json(r),semantic_digest(r))
    wrong=synthetic_trace(c,observations[1],14)
    with pytest.raises(ValueError,match='Ordered'):cache.add_replayed(wrong,simulated_receipt(wrong))
    assert not cache.record()['complete'] and len(cache.record()['traces'])==1


def test_existing_SELECT_endpoint_does_not_accept_new_protocol():
    # Rejection precedes checkpoint or lineage I/O, not a future SELECT admission.
    p=protocol()
    with pytest.raises(ValueError):admission.validate_union_select013_inference(p,
        checkpoint_lineage={},learning_protocol_hash=semantic_digest(p),
        proposal_config=NominalCavityProposalConfig(**thaw_json(p['cohort_execution']['proposal_config'])),
        max_steps=24,search=p['cohort_execution']['search'],public_target_context_variant=TARGET_CONTEXT)


def test_visits_bind_new4_and_forward_condition(tmp_path,monkeypatch):
    from resectionlab.patient_planning_cohort_visits import make_train_visit_factories
    p=protocol(); bound=limits(p);inputs=fixture(tmp_path)
    rows=[{'patient_id':s,'role':'TRAIN','path':str(tmp_path/(s+'.json')),'sha256':'a'*64} for s in POST_EXPOSURE_TRAIN]
    path=tmp_path/'index.json';path.write_text(json.dumps({'cohort_sha256':admission.COHORT_SHA256,'cases':rows}))
    release={'limits':{k:v for k,v in bound.items() if k not in ('output_bytes','checkpoint_bytes')},'learning_protocol_hash':semantic_digest(p)}
    calls=[]
    def stub(*a,**kw):calls.append(kw);return object(),{},{},{}
    monkeypatch.setattr(factory,'prepare_public_source',stub)
    monkeypatch.setattr(admission,'make_patient_planning_task',lambda *a,**kw:('task','context'))
    factories=make_train_visit_factories(manifest_index_path=path,manifest_index_sha256=factory.sha(path),
        cohort_bytes=inputs['cohort_bytes'],learning_protocol=p,limits=bound,released_record=release,
        released_sha256='a'*64,progress=lambda *a:None)
    assert tuple(factories)==protocol_train_subjects(p)==POST_EXPOSURE_TRAIN
    with pytest.raises(ValueError):validate_factories(factories) # old beam runner remains closed
    for subject in POST_EXPOSURE_TRAIN:
        out=tmp_path/subject;out.mkdir();assert factories[subject](output=out)==('task','context')
    assert all(c['post_exposure_condition']==EXPOSURE and semantic_digest(c['occupancy_learning_protocol'])==semantic_digest(p) for c in calls)


def test_old_protocol_records_and_checkpoint_lineage_are_unchanged(monkeypatch):
    baseline=os.environ.get('POST_EXPOSURE_LEARNING_BASELINE')
    if not baseline:pytest.skip('historical source parity requires explicit saved baseline')
    path=Path(baseline)/'patient_planning_cohort_spec.py'
    spec=importlib.util.spec_from_file_location('resectionlab._old_post_learning_spec',path)
    old=importlib.util.module_from_spec(spec);monkeypatch.setitem(sys.modules,spec.name,old);spec.loader.exec_module(old)
    for updates in (1,8,64):
        options=dict(updates=updates,max_steps=24,search={'max_calls':5640,'beam_width':2,'seconds':300},
            proposal_config=NominalCavityProposalConfig(max_candidates=120,intermediate_opening_mm=1.,tool_footprint_opening=True,obstruction_opening=True),
            occupancy_condition='cerebrum_plus_supplied_tumor_assumption')
        if updates==64:options.update(il_teacher_weighting=BALANCED_TEACHER_CE,teacher_observations=CACHED_TEACHERS,teacher_cache_payload_bytes=64*1024**2)
        a=old.sequential_learning_protocol(**options); b=sequential_learning_protocol(**options)
        assert semantic_digest(a)==semantic_digest(b) and protocol_train_subjects(b)==TRAIN
