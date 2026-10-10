"""Generated loss algebra and fixed-lineage controls; no patient input."""
import os
from pathlib import Path

import pytest
import torch
import resectionlab

if 'BALANCED_STAGE' in os.environ:
    resectionlab.__path__.insert(0, os.environ['BALANCED_STAGE'])
from resectionlab.core import semantic_digest, thaw_json
from resectionlab.native_proposals import NominalCavityProposalConfig
from resectionlab.patient_planning_cohort_spec import sequential_learning_protocol, validate_sequential_protocol, BALANCED_TEACHER_CE
from resectionlab.patient_planning_learning import (common_patient_policies, PatientImitationSample,
    patient_imitation_loss, patient_gradient_step, patient_imitation_group_weights)
from resectionlab.patient_planning_accumulation import PatientGradientAccumulator
from resectionlab import patient_planning_cohort_sequential
from resectionlab.patient_planning_cohort_io import OutputBudget, _save_checkpoint, load_cohort_checkpoint
from resectionlab.spatial_policy import parameter_hash
from test_patient_cohort_sequential import generated_contexts, synthetic_trace, session_for


@pytest.fixture(autouse=True)
def one_thread(): torch.set_num_threads(1)


def protocol(*, balanced=True, updates=1):
    return sequential_learning_protocol(updates=updates, max_steps=24,
        search={'max_calls':5640, 'beam_width':2, 'seconds':300},
        proposal_config=NominalCavityProposalConfig(max_candidates=120,
            intermediate_opening_mm=1., tool_footprint_opening=True),
        retention_mode='return_plus_opening_depth_volume_v1',
        il_teacher_weighting=BALANCED_TEACHER_CE if balanced else None)


def make_traces(contexts, observations):
    return tuple(synthetic_trace(c,o,n) for c,o,n in zip(contexts,observations,(1,1,1,2)))


def pins(traces):
    return {t.context.fingerprint:dict(steps=len(t.transitions), trace_seal=t.seal_hash,
        stop_steps=sum(r.action_id=='STOP' for r in t.transitions)) for t in traces}


def test_default_protocol_identity_and_initial_tensors_are_unchanged():
    old=protocol(balanced=False,updates=8); new=protocol(updates=8)
    assert semantic_digest(old)=='sha256:713594a9b671ef4d13945ea6cce09d1ffcc0eb3c228442d3adbc74ddab91029b'
    assert 'il_teacher_weighting' not in old['cohort_execution']
    changed=thaw_json(new);del changed['cohort_execution']['il_teacher_weighting']
    assert semantic_digest(changed)==semantic_digest(old)
    assert semantic_digest(new)!=semantic_digest(old)
    a,_=common_patient_policies(generated_contexts(old)[0],old)
    b,_=common_patient_policies(generated_contexts(new)[0],new)
    assert parameter_hash(a)==parameter_hash(b) and a.architecture_hash==b.architecture_hash
    bad=thaw_json(new);bad['cohort_execution']['il_teacher_weighting']='STOP_penalty'
    with pytest.raises(ValueError):validate_sequential_protocol(bad)


def test_exact_group_weights_and_all_legal_motion_gradients():
    weights=patient_imitation_group_weights(protocol(),stop_count=4,motion_count=1)
    assert weights=={'STOP':.125,'motion':.5}
    labels=torch.tensor([0,0,0,1,0])
    raw=torch.tensor([[.4,.1,-.8,9.]]*5,requires_grad=True)
    logits=raw.masked_fill(torch.tensor([[False,False,False,True]]*5),-torch.inf)
    per=torch.tensor([weights['STOP' if i==0 else 'motion'] for i in labels])
    loss=(-(logits.log_softmax(-1)[torch.arange(5),labels])*per).sum();loss.backward()
    expected=logits.detach().softmax(-1);expected[torch.arange(5),labels]-=1;expected*=per[:,None]
    torch.testing.assert_close(raw.grad,expected)
    assert raw.grad[3,1]<0 and raw.grad[3,2]>0 and raw.grad[:,3].abs().sum()==0
    # The rare correct movement is strengthened relative to other movements,
    # rather than changing the inference STOP threshold or action mask.
    assert weights['motion']/weights['STOP']==4
    uniform=patient_imitation_group_weights(protocol(balanced=False),stop_count=4,motion_count=1)
    assert uniform=={'STOP':.2,'motion':.2}


@pytest.mark.parametrize('stops,motions',[(0,5),(5,0),(True,1),(-1,2),(0,0)])
def test_missing_or_malformed_group_counts_refuse(stops,motions):
    with pytest.raises(ValueError):patient_imitation_group_weights(protocol(),stop_count=stops,motion_count=motions)


def test_group_refusal_precedes_any_forward_or_backward(monkeypatch):
    p=protocol();contexts,obs=generated_contexts(p)
    session=session_for(contexts,p,'IL')
    traces=tuple(synthetic_trace(c,o,1) for c,o in zip(contexts,obs))
    def forbidden(*args,**kwargs):raise AssertionError('No tensor work before both groups admitted')
    monkeypatch.setattr(session.policy,'forward',forbidden)
    with pytest.raises(ValueError,match='both STOP and motion'):
        PatientGradientAccumulator(session,teacher_pins=pins(traces))
    with pytest.raises(ValueError,match='both STOP and motion'):
        patient_imitation_loss(session,[PatientImitationSample(t,0) for t in traces])
    assert session.updates==0 and all(q.grad is None for q in session.policy.parameters())


def test_balanced_batch_and_sequential_gradients_one_update_and_checkpoint(tmp_path):
    p=protocol();contexts,obs=generated_contexts(p);traces=make_traces(contexts,obs)
    reference=session_for(contexts,p,'IL');sequential=session_for(contexts,p,'IL')
    samples=[PatientImitationSample(t,i) for t in traces for i in range(len(t.transitions))]
    loss,detail=patient_imitation_loss(reference,samples)
    manual=[]
    for t in traces:
        for row in t.transitions:
            logit,_=reference.policy(row.observation);index=row.observation.action_ids.index(row.action_id)
            manual.append(-logit.log_softmax(-1)[index]*(.125 if row.action_id=='STOP' else .5))
    torch.testing.assert_close(loss,torch.stack(manual).sum())
    expected=patient_gradient_step(reference,loss)
    with PatientGradientAccumulator(sequential,teacher_pins=pins(traces)) as transaction:
        for trace in traces:transaction.add_trace(trace)
        actual=transaction.finish()
    assert actual['loss']==pytest.approx(detail['loss'],abs=1e-6)
    assert actual['teacher_group_counts']=={'STOP':4,'motion':1}
    assert actual['teacher_group_weights']=={'STOP':.125,'motion':.5}
    assert actual['loss_forward_calls']==5 and actual['completed_updates']==1
    assert actual['IL_reduction']==BALANCED_TEACHER_CE
    for a,b in zip(reference.policy.parameters(),sequential.policy.parameters()):
        torch.testing.assert_close(a,b,atol=2e-6,rtol=2e-5)
        if a.grad is not None:torch.testing.assert_close(a.grad,b.grad,atol=1e-6,rtol=1e-5)
    assert actual['gradient_norm_before_clip']==pytest.approx(expected['gradient_norm_before_clip'],rel=1e-5)
    out=OutputBudget(tmp_path,16*1024**2)
    saved=_save_checkpoint(sequential.policy,method='IL',contexts=contexts,protocol=p,updates=1,
        initial_hash=sequential.initial_parameter_hash,output=out,limits={'checkpoint_bytes':8*1024**2})
    expected_contexts={c.patient_group:c.fingerprint for c in contexts}
    loaded,meta=load_cohort_checkpoint(tmp_path/saved['path'],expected_sha256=saved['sha256'],
        expected_learning_protocol=p,expected_context_hashes=expected_contexts,expected_method='IL')
    assert parameter_hash(loaded)==parameter_hash(sequential.policy)
    assert meta['learning_protocol_hash']==semantic_digest(p)
    with pytest.raises(ValueError,match='lineage'):
        load_cohort_checkpoint(tmp_path/saved['path'],expected_sha256=saved['sha256'],
            expected_learning_protocol=protocol(balanced=False),expected_context_hashes=expected_contexts,expected_method='IL')


def test_incorrect_pinned_trace_group_aborts_before_forward(monkeypatch):
    p=protocol();contexts,obs=generated_contexts(p);traces=make_traces(contexts,obs)
    session=session_for(contexts,p,'IL');declared=pins(traces)
    declared[traces[0].context.fingerprint]['stop_steps']=0
    transaction=PatientGradientAccumulator(session,teacher_pins=declared)
    monkeypatch.setattr(session.policy,'forward',lambda *a:(_ for _ in ()).throw(AssertionError('forward')))
    with pytest.raises(ValueError,match='STOP count'):transaction.add_trace(traces[0])
    assert session.updates==0 and all(p.grad is None for p in session.policy.parameters())
