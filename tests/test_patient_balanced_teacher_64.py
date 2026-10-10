"""Protocol/weight-format controls only; no patient, forward or optimizer work.

Generated checkpoints declare an endpoint solely to test bounded loader metadata.
They do not certify that optimizer updates occurred.
"""
import pytest
import torch
from resectionlab.core import freeze_json, semantic_digest, thaw_json
from resectionlab.native_proposals import NominalCavityProposalConfig
from resectionlab.patient_planning_cohort_spec import (BALANCED_TEACHER_CE,
    sequential_learning_protocol, validate_sequential_protocol, sequential_limits,
    validate_limits, preview_budget_sizing)
from resectionlab.patient_planning_admission import (PatientPlanningContext,
    SELECT_INITIALIZATION, validate_select_checkpoint_lineage)
from resectionlab.patient_planning_learning import common_patient_policies, PatientTrainSession
from resectionlab.patient_planning_cohort_io import OutputBudget, _save_checkpoint
from resectionlab.patient_select_inference import load_select_checkpoint, require_select_context
from resectionlab.spatial_policy import parameter_hash
from test_patient_cohort_sequential import generated_contexts


@pytest.fixture(autouse=True)
def one_thread(): torch.set_num_threads(1)


def protocol(updates=64, *, balanced=True):
    return sequential_learning_protocol(updates=updates, max_steps=24,
        search={'max_calls':5640, 'beam_width':2, 'seconds':300},
        proposal_config=NominalCavityProposalConfig(max_candidates=120,
            intermediate_opening_mm=1., tool_footprint_opening=True),
        retention_mode='return_plus_opening_depth_volume_v1',
        il_teacher_weighting=BALANCED_TEACHER_CE if balanced else None)


def limits(p):
    return sequential_limits(p, worker_seconds=3540, memory_bytes=3*1024**3,
        output_bytes=128*1024**2)


def lineage(updates=64):
    return {'version':'frozen-TRAIN-checkpoint-lineage-v1', 'method':'IL',
        'checkpoint_sha256':'1'*64, 'training_release_sha256':'2'*64,
        'learning_protocol_hash':'sha256:'+'3'*64,
        'initial_parameter_hash':'sha256:'+'4'*64, 'parameter_hash':'sha256:'+'5'*64,
        'architecture_hash':'sha256:'+'6'*64, 'completed_updates':updates,
        'training_context_hashes':{'ReMIND:'+s:'sha256:'+'7'*64 for s in ('008','010','020','025')},
        'public_target_context_variant':None, 'optimizer_updates_on_SELECT':0}


def test_all_sixteen_prior_protocols_and_eight_limits_keep_exact_identities():
    # Golden derived from the frozen balanced8 release's full protocol, changing
    # only its update field and presence of the opt-in weighting field.
    rows={('balanced' if b else 'default')+str(u):protocol(u,balanced=b)
        for b in (False,True) for u in range(1,9)}
    assert semantic_digest(rows)=='sha256:cc7ddc40000183262e243117916069793ddff5715684fe393b7bd1e1f60602e4'
    assert semantic_digest(rows['default8'])=='sha256:713594a9b671ef4d13945ea6cce09d1ffcc0eb3c228442d3adbc74ddab91029b'
    assert semantic_digest(rows['balanced8'])=='sha256:16725acb244599efb87bc6f793b7bce652b0ebc666d9f613b7dfdbb9fcaf2973'
    for b in (False,True):
        assert thaw_json(limits(protocol(8,balanced=b)))=={
            'checkpoint_bytes':8388608, 'max_native_previews':3373440,
            'max_optimizer_updates':16, 'max_policy_forwards':2496, 'max_steps':24,
            'memory_bytes':3221225472, 'output_bytes':134217728,
            'search':{'beam_width':2,'max_calls':5640,'seconds':300},
            'threads':1,'worker_seconds':3540}


@pytest.mark.parametrize('updates', [0, True, 8., 9, 32, 63, 65, 128])
def test_only_declared_eight_or_fixed64_endpoints_are_accepted(updates):
    with pytest.raises(ValueError): protocol(updates)


def test_64_requires_balanced_flag_and_exact_protocol_reconstruction():
    p=protocol();validate_sequential_protocol(p)
    changed=thaw_json(p);changed['updates_per_method']=8
    assert semantic_digest(changed)==semantic_digest(protocol(8))
    with pytest.raises(ValueError): protocol(64,balanced=False)
    for flag in (None,'STOP_penalty'):
        changed=thaw_json(p)
        if flag is None:del changed['cohort_execution']['il_teacher_weighting']
        else:changed['cohort_execution']['il_teacher_weighting']=flag
        with pytest.raises(ValueError):validate_sequential_protocol(changed)


def test_only64_extends_derived_sizing_without_loosening_wall_rss_or_roles():
    p=protocol();bound=limits(p);sizing=preview_budget_sizing(p)
    assert sizing['visits_per_patient']==131
    assert bound['max_optimizer_updates']==128
    assert bound['max_native_previews']==7297920
    assert bound['max_policy_forwards']==18624
    # These are unchanged two-method context sizing formulas. The separately
    # owned IL-only driver must enforce its actual64/421/2347680 ceilings.
    for key in ('max_native_previews','max_policy_forwards'):
        for delta in (-1,1):
            changed=thaw_json(bound);changed[key]+=delta
            with pytest.raises(ValueError):validate_limits(p,changed)
    for key,value in (('max_optimizer_updates',64),('worker_seconds',3601),
            ('memory_bytes',3*1024**3+1),('threads',2),('output_bytes',256*1024**2+1)):
        changed=thaw_json(bound);changed[key]=value
        with pytest.raises(ValueError):validate_limits(p,changed)
    for balanced in (False,True):
        old=protocol(8,balanced=balanced)
        for key,value in (('max_native_previews',4194305),('max_policy_forwards',4097)):
            changed=thaw_json(limits(old));changed[key]=value
            with pytest.raises(ValueError):validate_limits(old,changed)
    changed=thaw_json(p);changed['cohort_execution']['patient_order'][-1]='ReMIND-013'
    with pytest.raises(ValueError):validate_sequential_protocol(changed)


@pytest.mark.parametrize('updates',[1,8,32,64])
def test_declared_lineage_endpoints_do_not_change_fields(updates):
    record=lineage(updates)
    assert thaw_json(validate_select_checkpoint_lineage(record))==record


@pytest.mark.parametrize('updates',[0,True,64.,*range(33,64),65,128])
def test_no_intermediate_or_larger_select_lineage_endpoints(updates):
    with pytest.raises(ValueError):validate_select_checkpoint_lineage(lineage(updates))


def test_64_lineage_still_refuses_updates_and_wrong_protocol_binding():
    record=lineage();record['optimizer_updates_on_SELECT']=1
    with pytest.raises(ValueError):validate_select_checkpoint_lineage(record)
    with pytest.raises(ValueError):
        validate_select_checkpoint_lineage(lineage(),learning_protocol_hash='sha256:'+'8'*64)


def test_generated64_loader_and_context_remain_inference_only(tmp_path,monkeypatch):
    p=protocol();contexts,_=generated_contexts(p)
    old=protocol(8);old_contexts,_=generated_contexts(old)
    policy,unused=common_patient_policies(contexts,p)
    prior,unused_prior=common_patient_policies(old_contexts,old)
    initial=parameter_hash(policy)
    assert initial==parameter_hash(prior)
    assert all(torch.equal(a,b) for a,b in zip(policy.parameters(),prior.parameters()))
    sink=OutputBudget(tmp_path,16*1024**2)
    saved=_save_checkpoint(policy,method='IL',contexts=contexts,protocol=p,updates=64,
        initial_hash=initial,output=sink,limits={'checkpoint_bytes':8*1024**2})
    pins={c.patient_group:c.fingerprint for c in contexts}
    kwargs=dict(expected_sha256=saved['sha256'],expected_context_hashes=pins,
        expected_method='IL',training_release_sha256='a'*64)
    loaded=load_select_checkpoint(tmp_path/saved['path'],expected_learning_protocol=p,**kwargs)
    assert loaded.lineage_record()['completed_updates']==64
    assert parameter_hash(loaded.policy)==initial
    assert all(not q.requires_grad and q.grad is None for q in loaded.policy.parameters())
    with pytest.raises(ValueError,match='lineage'):
        load_select_checkpoint(tmp_path/saved['path'],expected_learning_protocol=old,**kwargs)
    unweighted=thaw_json(p);unweighted['cohort_execution'].pop('il_teacher_weighting')
    with pytest.raises(ValueError,match='balanced-teacher'):
        load_select_checkpoint(tmp_path/saved['path'],expected_learning_protocol=unweighted,**kwargs)
    # Explicitly fabricated SELECT metadata exercises role/lineage capability,
    # not acquired source QC. No native task, forward or update occurs.
    record=contexts[0].record();record.update(subject='ReMIND-013',patient_group='ReMIND:013',
        role='SELECT',max_optimizer_updates=0,initialization=SELECT_INITIALIZATION,
        private_reference_in_task=False,checkpoint_lineage=loaded.lineage_record())
    context=PatientPlanningContext(freeze_json(record),semantic_digest(record))
    require_select_context(context);loaded.require(context)
    def no_optimizer(*args,**kwargs):raise AssertionError('SELECT must not construct an optimizer')
    monkeypatch.setattr(torch.optim,'Adam',no_optimizer)
    with pytest.raises(ValueError,match='TRAIN_gradient'):context.require_training()
    with pytest.raises(ValueError,match='TRAIN_gradient'):
        PatientTrainSession((context,),'IL',loaded.policy,initial_parameter_hash=initial,protocol=p)
    for subject,role in (('ReMIND-008','TRAIN'),('ReMIND-067','MEASUREMENT_EVAL')):
        bad=dict(record,subject=subject,patient_group='ReMIND:'+subject[-3:],role=role)
        with pytest.raises(ValueError):require_select_context(PatientPlanningContext(freeze_json(bad),semantic_digest(bad)))


def test_common_initialization_still_requires_full_declared_context_budget(monkeypatch):
    p=protocol();contexts,_=generated_contexts(p)
    bad=contexts[0].record();bad['max_optimizer_updates']=127
    contexts=(PatientPlanningContext(freeze_json(bad),semantic_digest(bad)),*contexts[1:])
    monkeypatch.setattr(torch,'manual_seed',lambda *args:(_ for _ in ()).throw(AssertionError('construction')))
    with pytest.raises(ValueError,match='aggregate admitted context budget'):
        common_patient_policies(contexts,p)
