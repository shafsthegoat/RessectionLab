"""Generated admission/metadata controls. No models, forwards or updates.

The tiny fabricated annotation-style arrays test adapter contracts only; they
are not acquired QC. Existing saved protocol reads are JSON metadata only.
"""
import copy
from dataclasses import replace
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import sys

import pytest
from resectionlab.core import semantic_digest, thaw_json
from resectionlab import patient_planning_admission as admission
from resectionlab import public_patient_factory as factory
from resectionlab import patient_planning_learning as learning
from resectionlab.native_proposals import NominalCavityProposalConfig
from resectionlab.native_spatial_task import SUPPLIED_TUMOR_UNION_OCCUPANCY as UNION
from resectionlab.patient_planning_cohort_spec import (TRAIN, CACHED_TEACHERS,
    BALANCED_TEACHER_CE, sequential_learning_protocol, sequential_limits,
    validate_sequential_protocol, validate_union_obstruction_learning)
from resectionlab.public_target_context import VERSION as TARGET_CONTEXT
import test_paired_train_occupancy as helper


def protocol(updates=8, *, condition=UNION, **config_changes):
    options=dict(max_candidates=120, intermediate_opening_mm=1.,
        tool_footprint_opening=True, obstruction_opening=True)
    options.update(config_changes)
    return sequential_learning_protocol(updates=updates, max_steps=24,
        search={'max_calls':5640, 'beam_width':2, 'seconds':300},
        proposal_config=NominalCavityProposalConfig(**options),
        retention_mode='return_plus_opening_depth_volume_v1',
        **({'il_teacher_weighting':BALANCED_TEACHER_CE,
            'teacher_observations':CACHED_TEACHERS,'teacher_cache_payload_bytes':64*1024**2}
           if updates==64 else {}), occupancy_condition=condition)


def limits(p):
    return thaw_json(sequential_limits(p,worker_seconds=3540,memory_bytes=3*1024**3,
        output_bytes=128*1024**2))


def build(path, inputs, p, **changes):
    path.mkdir()
    config=NominalCavityProposalConfig(**thaw_json(p['cohort_execution']['proposal_config']))
    bound={k:v for k,v in limits(p).items() if k not in ('output_bytes','checkpoint_bytes')}
    options=dict(learning_protocol_hash=semantic_digest(p),proposal_config=config,
        public_target_context_variant=TARGET_CONTEXT,occupancy_condition=UNION,
        occupancy_learning_protocol=p)
    options.update(changes)
    return factory.prepare_public_source(path,{'limits':bound},'b'*64,None,
        lambda *a,**k:None,**inputs,**options)


def admit(built,inputs):
    source,binding,qc,plan=built
    return admission.make_patient_planning_task(source,cohort_bytes=inputs['cohort_bytes'],
        source_binding=binding,qc_receipt=qc,protocol=plan)


@pytest.mark.parametrize('subject',TRAIN)
@pytest.mark.parametrize('updates',[8,64])
def test_exact_old_four_union_learning_context_and_reconstruction(tmp_path,subject,updates):
    inputs,_=helper.fixture(tmp_path,subject=subject);p=protocol(updates)
    built=build(tmp_path/'first',inputs,p)
    task,context=admit(built,inputs)
    context.require_training();context.require_task(task.planning_clone());context.require_task(task.fresh())
    context.require_observations([task.observation()])
    record=context.record()
    assert record['execution_kind']=='fixed_four_TRAIN_union_obstruction_learning_v1'
    assert record['policy_comparison_permitted'] is True
    assert record['comparison_scope']=='same_declared_union_world_fixed_four_TRAIN_only'
    assert record['derived_occupancy_anatomically_validated'] is False
    assert record['max_optimizer_updates']==2*updates
    assert record['learning_protocol_hash']==semantic_digest(p)
    assert built[2]['native_domain_fully_covered'] is True
    assert built[2]['derived_occupancy_anatomically_validated'] is False
    again=build(tmp_path/'again',inputs,p)
    assert built[1:]==again[1:] and built[0].source_hash==again[0].source_hash
    assert task.step('STOP').terminated


@pytest.mark.parametrize('change',['condition','obstruction','footprint','cap','horizon','private','order'])
def test_protocol_rejects_condition_config_objective_and_role_changes(change):
    p=thaw_json(protocol())
    if change=='condition':p['cohort_execution']['occupancy_condition']=admission.PARTIAL_DOMAIN_UNION_OCCUPANCY
    elif change in ('obstruction','footprint'):
        p['cohort_execution']['proposal_config']['obstruction_opening' if change=='obstruction' else 'tool_footprint_opening']=False
    elif change=='cap':p['cohort_execution']['proposal_config']['max_candidates']=96
    elif change=='horizon':p['cohort_execution']['max_steps']=6
    elif change=='private':p['private_training_reward']=True
    else:p['cohort_execution']['patient_order'][-1]='ReMIND-013'
    with pytest.raises(ValueError):validate_sequential_protocol(p)


@pytest.mark.parametrize('change',['hash','config','horizon','initializer','budget','zero_updates','zero_forwards','variant','condition','private','QC'])
def test_admission_rejects_rebound_learning_mismatches_before_inventory(tmp_path,monkeypatch,change):
    inputs,_=helper.fixture(tmp_path);built=build(tmp_path/'source',inputs,protocol())
    case,args=helper.contract(built,inputs);args=copy.deepcopy(args)
    if change=='hash':args['protocol']['learning_protocol_hash']='sha256:'+'0'*64
    elif change=='config':
        altered=thaw_json(args['protocol']['occupancy_learning_protocol'])
        altered['cohort_execution']['proposal_config']['obstruction_opening']=False
        args['protocol']['occupancy_learning_protocol']=altered
    elif change=='horizon':args['protocol']['max_steps']=6
    elif change=='initializer':args['protocol']['initialization']='checkpoint_warmstart'
    elif change=='budget':args['protocol']['max_optimizer_updates']=1
    elif change=='zero_updates':args['protocol']['max_optimizer_updates']=0
    elif change=='zero_forwards':args['protocol']['max_policy_forwards']=0
    elif change=='variant':case=replace(case,public_target_context_variant=None,public_target_domain=None)
    elif change=='condition':args['protocol']['occupancy_condition']='raw_cerebrum_baseline'
    elif change=='private':args['protocol']['private_reference_used']=True
    else:args['qc_receipt']['derived_occupancy_anatomically_validated']=True
    helper.rebind(args)
    monkeypatch.setattr(admission,'NativeSpatialTask',lambda *a,**k:pytest.fail('refuse before inventory'))
    with pytest.raises(ValueError):admission.make_patient_planning_task(case,**args)


@pytest.mark.parametrize('subject',('ReMIND-002','ReMIND-015','ReMIND-018','ReMIND-045','ReMIND-013','ReMIND-037','ReMIND-067'))
def test_other_people_cannot_enter_full_coverage_learning_branch(tmp_path,monkeypatch,subject):
    inputs,_=helper.fixture(tmp_path,subject=subject)
    monkeypatch.setattr(factory.np,'load',lambda *a,**k:pytest.fail('no array loading'))
    with pytest.raises(ValueError):build(tmp_path/'out',inputs,protocol())


def test_partial_domain_and_raw_learning_flags_refuse_before_manifest(tmp_path,monkeypatch):
    p=protocol();monkeypatch.setattr(factory,'load_public_manifest',lambda *a,**k:pytest.fail('no manifest read'))
    for condition in (admission.PARTIAL_DOMAIN_UNION_OCCUPANCY,'raw_cerebrum_baseline'):
        with pytest.raises(ValueError):build(tmp_path/condition,dict(public_manifest_path=tmp_path/'absent',
            public_manifest_sha256='a'*64,cohort_bytes=b'unused'),p,occupancy_condition=condition)


def test_context_condition_join_refuses_before_model_construction(tmp_path,monkeypatch):
    inputs,_=helper.fixture(tmp_path);p=protocol();_,context=admit(build(tmp_path/'source',inputs,p),inputs)
    record=context.record();record.pop('occupancy_condition')
    wrong=admission.PatientPlanningContext(admission.freeze_json(record),semantic_digest(record))
    monkeypatch.setattr(learning,'SpatialPolicy',lambda *a,**k:pytest.fail('no model construction'))
    with pytest.raises(ValueError,match='occupancy condition'):learning.common_patient_policies([wrong],p)


@pytest.mark.parametrize('enabled',[False,True])
def test_visit_factory_threads_only_explicit_condition(tmp_path,monkeypatch,enabled):
    from resectionlab.patient_planning_cohort_visits import make_train_visit_factories
    p=protocol(condition=UNION if enabled else None);bound=limits(p)
    repo=next(q for q in Path(__file__).resolve().parents if (q/'src/resectionlab').is_dir())
    cohort=(repo/'manifests/experiments/remind-component-cohort-v1.json').read_bytes()
    index={'cohort_sha256':admission.COHORT_SHA256,'cases':[
        {'patient_id':s,'role':'TRAIN','path':str(tmp_path/(s+'.json')),'sha256':'a'*64} for s in TRAIN]}
    path=tmp_path/'index.json';raw=json.dumps(index).encode();path.write_bytes(raw)
    release={'limits':{k:v for k,v in bound.items() if k not in ('output_bytes','checkpoint_bytes')},
        'learning_protocol_hash':semantic_digest(p)}
    calls=[]
    def stub(*a,**kw):calls.append(kw);return object(),{}, {},{}
    monkeypatch.setattr(factory,'prepare_public_source',stub)
    monkeypatch.setattr(admission,'make_patient_planning_task',lambda *a,**kw:('task','context'))
    factories=make_train_visit_factories(manifest_index_path=path,manifest_index_sha256=hashlib.sha256(raw).hexdigest(),
        cohort_bytes=cohort,learning_protocol=p,limits=bound,released_record=release,released_sha256='b'*64,progress=lambda *a:None)
    for subject in TRAIN:
        out=tmp_path/subject;out.mkdir();assert factories[subject](output=out)==('task','context')
    assert len(calls)==4
    for kw in calls:
        assert ('occupancy_condition' in kw) is enabled
        assert ('occupancy_learning_protocol' in kw) is enabled
        if enabled:
            assert kw['occupancy_condition']==UNION
            assert semantic_digest(kw['occupancy_learning_protocol'])==semantic_digest(p)


def test_saved_prior_protocols_keep_exact_round_trip():
    from test_obstruction_opening_proposals import test_historical_saved_protocols_and_new_enabled_record_round_trip
    test_historical_saved_protocols_and_new_enabled_record_round_trip()


@pytest.mark.parametrize('condition',[helper.RAW,helper.UNION])
def test_old_raw_and_search_only_protocol_context_and_inputs_unchanged(tmp_path,monkeypatch,condition):
    # Root pins these pre-change source copies in the source index. No patient data.
    repo=next(q for q in Path(__file__).resolve().parents if (q/'src/resectionlab').is_dir())
    baseline=Path(os.environ.get('UNION_LEARNING_BASELINE_DIR',
        repo/'build/obstruction-opening-learning-admission-v1/baseline/resectionlab'))
    if not baseline.is_dir():
        pytest.skip('historical source parity requires the explicitly pinned pre-change source directory')
    old={}
    for name in ('public_patient_factory','patient_planning_admission'):
        spec=importlib.util.spec_from_file_location('resectionlab._old_union_learning_'+name,baseline/(name+'.py'))
        module=importlib.util.module_from_spec(spec);monkeypatch.setitem(sys.modules,spec.name,module);spec.loader.exec_module(module);old[name]=module
    inputs,_=helper.fixture(tmp_path)
    current=helper.build(tmp_path/'current',inputs,condition=condition)
    prior=helper.build(tmp_path/'prior',inputs,condition=condition,module=old['public_patient_factory'])
    assert current[0].source_hash==prior[0].source_hash and current[1:]==prior[1:]
    task,context=admit(current,inputs)
    source,binding,qc,plan=prior
    old_task,old_context=old['patient_planning_admission'].make_patient_planning_task(source,
        cohort_bytes=inputs['cohort_bytes'],source_binding=binding,qc_receipt=qc,protocol=plan)
    assert context.record()==old_context.record()
    assert task.observation().fingerprint==old_task.observation().fingerprint
    assert task.candidate_inventory()==old_task.candidate_inventory()
    for file in ('admitted-public-bindings.json','public-task-derivation.json'):
        assert (tmp_path/'current'/file).read_bytes()==(tmp_path/'prior'/file).read_bytes()
    if condition==helper.UNION:
        assert (tmp_path/'current/derived-occupancy-assumption.json').read_bytes()==(tmp_path/'prior/derived-occupancy-assumption.json').read_bytes()
        with pytest.raises(ValueError):context.require_training()


def test_learning_changes_admission_not_the_successful_search_world(tmp_path):
    inputs,_=helper.fixture(tmp_path);p=protocol()
    learned=build(tmp_path/'learned',inputs,p)
    out=tmp_path/'search';out.mkdir()
    bound={k:v for k,v in limits(p).items() if k not in ('output_bytes','checkpoint_bytes')}
    bound.update(max_optimizer_updates=0,max_policy_forwards=0)
    searched=factory.prepare_public_source(out,{'limits':bound},'b'*64,None,lambda *a,**k:None,
        **inputs,learning_protocol_hash='sha256:'+'c'*64,
        proposal_config=learned[0].proposal_config,public_target_context_variant=TARGET_CONTEXT,
        occupancy_condition=UNION)
    assert learned[0].source_hash==searched[0].source_hash
    assert learned[1]==searched[1] and learned[2]==searched[2]
    a,ac=admit(learned,inputs);b,bc=admit(searched,inputs)
    assert a.decision_model_hash==b.decision_model_hash
    assert a.observation().fingerprint==b.observation().fingerprint
    assert a.candidate_inventory()==b.candidate_inventory()
    assert ac.fingerprint!=bc.fingerprint
    with pytest.raises(ValueError):bc.require_training()
