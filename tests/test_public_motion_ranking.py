"""Small generated tensor/metadata controls; no acquired data, model weights or native steps."""
from dataclasses import replace
import copy
import importlib.util
import os
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from resectionlab.core import freeze_json, semantic_digest, thaw_json
from resectionlab.public_motion_ranking import (VERSION,SCOPE,OBJECTIVE_SOURCE,SUBJECTS,STEPS,
    PublicMotionRankingCorpus,motion_ranking_loss,supervision_record)
from resectionlab.patient_planning_accumulation import PatientGradientAccumulator
from resectionlab.patient_planning_learning import (PatientTrainingTrace,PatientTrainSession,
    common_patient_policies,patient_imitation_loss,_TRACE)
from resectionlab.patient_planning_cohort_spec import validate_sequential_protocol
from resectionlab.spatial_policy import parameter_hash
import test_post_exposure_learning as helper
from test_patient_cohort_sequential import synthetic_trace

H='sha256:'+'a'*64

@pytest.fixture(autouse=True)
def one_thread():
    torch.set_num_threads(1)


def loss(z,rewards=(0.,3.,1.,1.),teacher='a'):
    return motion_ranking_loss(z,action_ids=('STOP','a','b','c'),action_mask=(True,)*4,
                              rewards=rewards,teacher_action=teacher)


def test_reward_order_gradient_ties_and_positive_affine_invariance():
    z=torch.zeros(4,dtype=torch.float64,requires_grad=True)
    value=loss(z);value.backward();g=z.grad.clone()
    assert g[1]<g[2] and g[2]==g[3]  # high-reward score increases more; ties symmetric
    assert loss(z-.1*g)<value
    changed=loss(z,rewards=(0.,11.,5.,5.))
    assert torch.equal(changed,value)
    # Permuting tied motion identities does not create a winner label.
    assert loss(z,teacher='b')==value


def test_motion_gate_preserves_STOP_gradient_and_STOP_rows_exact_CE():
    z=torch.tensor([-.3,.2,-.4,1.1],dtype=torch.float64,requires_grad=True)
    new=loss(z);old=-z.log_softmax(-1)[1]
    gn=torch.autograd.grad(new,z,retain_graph=True)[0];go=torch.autograd.grad(old,z)[0]
    assert torch.allclose(gn[0],go[0],atol=1e-15,rtol=0)
    stop=loss(z,teacher='STOP');assert torch.equal(stop,-z.log_softmax(-1)[0])
    original_STOP=-z.log_softmax(-1)[0]
    assert torch.equal(torch.autograd.grad(stop,z)[0],torch.autograd.grad(original_STOP,z)[0])


def test_all_equal_rewards_have_gate_only_and_masked_actions_no_effect():
    z=torch.tensor([.2,-.1,.8,.3],dtype=torch.float64,requires_grad=True)
    expected=torch.logsumexp(z,0)-torch.logsumexp(z[1:],0)
    assert torch.equal(loss(z,rewards=(0.,2.,2.,2.)),expected)
    zm=torch.tensor([.2,-.1,.8,-float('inf')],dtype=torch.float64,requires_grad=True)
    value=motion_ranking_loss(zm,action_ids=('STOP','a','b','masked'),
        action_mask=(True,True,True,False),rewards=(0.,2.,1.,None),teacher_action='a')
    value.backward();assert zm.grad[-1]==0 and torch.isfinite(value)


def record():
    cases=[]
    for subject,n in zip(SUBJECTS,STEPS):
        decisions=[]
        for step in range(n):
            stop=step==n-1
            decisions.append(dict(step=step,observation_hash=semantic_digest([subject,step]),
                source_state_hash=H,action_ids=['STOP','a','b'],action_mask=[True,True,True],
                teacher_action='STOP' if stop else 'a',rewards=[0.,-1.,-2.] if stop else [0.,2.,1.]))
        cases.append(dict(subject=subject,role='TRAIN',source_hash=H,decision_model_hash=H,
            input_sha256={k:'1'*64 for k in ('scores','trace','plan','replay')},decisions=decisions))
    return dict(version=VERSION,scope=SCOPE,objective_source=OBJECTIVE_SOURCE,
                private_reference_used=False,subjects=cases)


@pytest.mark.parametrize('kind',['private','extra','role','subject','incomplete','nan','missing_action','reordered','winner'])
def test_label_schema_rejects_privileged_foreign_or_incomplete_targets(kind):
    r=record();case=r['subjects'][1];row=case['decisions'][0]
    if kind=='private':r['private_reference_used']=True
    elif kind=='extra':row['private_vessel_intersection']=3
    elif kind=='role':case['role']='SELECT'
    elif kind=='subject':case['subject']='ReMIND-008'
    elif kind=='incomplete':case['decisions'].pop()
    elif kind=='nan':row['rewards'][1]=float('nan')
    elif kind=='missing_action':row['rewards'].pop()
    elif kind=='reordered':row['action_ids'][0:2]=['a','STOP']
    else:row['teacher_action']='b'
    with pytest.raises((ValueError,TypeError)):
        PublicMotionRankingCorpus.admit(r,expected_hash=semantic_digest(r))


def test_wrong_observation_action_set_or_corpus_hash_refused():
    r=record();c=PublicMotionRankingCorpus.admit(r,expected_hash=semantic_digest(r));row=c._record['subjects'][1]['decisions'][0]
    observation=SimpleNamespace(fingerprint=row['observation_hash'],action_ids=tuple(row['action_ids']),
        action_mask=tuple(row['action_mask']),assert_intact=lambda:None)
    c.require_observation(row,observation,'a')
    for key,value in [('fingerprint',H),('action_ids',('STOP','a','foreign')),('action_mask',(True,True,False))]:
        bad=copy.copy(observation);setattr(bad,key,value)
        with pytest.raises(ValueError):c.require_observation(row,bad,'a')
    with pytest.raises(ValueError):PublicMotionRankingCorpus.admit(r,expected_hash=H)
    # Original mutable caller object cannot mutate immutable admitted labels.
    r['subjects'][1]['decisions'][0]['rewards'][1]=50.
    c.require();assert c._record['subjects'][1]['decisions'][0]['rewards'][1]==2.


def test_optin_protocol_default_and_old_checkpoint_lineage_unchanged():
    original=helper.protocol(64)
    assert original==helper.protocol(64,il_motion_supervision=None)
    ranking=helper.protocol(64,il_motion_supervision=supervision_record(H))
    plain=thaw_json(ranking);plain['cohort_execution'].pop('il_motion_supervision')
    assert semantic_digest(plain)==semantic_digest(original)
    validate_sequential_protocol(ranking)
    assert semantic_digest(ranking)!=semantic_digest(original)  # existing checkpoint loader exact hash separates them
    with pytest.raises(ValueError):helper.protocol(8,il_motion_supervision=supervision_record(H))
    with pytest.raises(ValueError):helper.protocol(64,il_motion_supervision={**supervision_record(H),'state_count':28})
    baseline=os.environ.get('MOTION_RANKING_BASELINE_SPEC')
    if baseline:
        spec=importlib.util.spec_from_file_location('resectionlab._ranking_baseline_spec',baseline)
        module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
        e=thaw_json(original)['cohort_execution']
        kwargs={k:e[k] for k in ('max_steps','search','retention_mode','il_teacher_weighting',
            'teacher_observations','teacher_cache_payload_bytes','occupancy_condition','post_exposure_condition')}
        from resectionlab.native_proposals import NominalCavityProposalConfig
        kwargs.update(updates=64,proposal_config=NominalCavityProposalConfig(**e['proposal_config']))
        assert semantic_digest(module.sequential_learning_protocol(**kwargs))==semantic_digest(original)


def test_complete_trace_join_and_actual_accumulator_seam(monkeypatch):
    # Actual admitted generated DTOs and SpatialPolicy; only 4^3 artificial
    # images. No native previews, acquired bytes or completed 64-update run.
    p=helper.protocol(64);contexts,observations=helper.generated_dtos(p,monkeypatch)
    traces=[];r=record()
    for case,c,o,n in zip(r['subjects'],contexts,observations,STEPS):
        trace=synthetic_trace(c,o,n);history=thaw_json(trace.history)
        for native in history:
            if native['action_id']!='STOP':native['source_state_hash']=H
        seal=semantic_digest({'context':c.fingerprint,'observations':[x.observation.fingerprint for x in trace.transitions],
                             'history':history,'behavior_parameter_hash':None})
        trace=replace(trace,history=freeze_json(history),seal_hash=seal);trace.require();traces.append(trace)
        case['source_hash']=c.record()['source_hash'];case['decision_model_hash']=c.record()['decision_model_hash']
        case['decisions']=[dict(step=i,observation_hash=x.observation.fingerprint,source_state_hash=H,
            action_ids=list(x.observation.action_ids),action_mask=[bool(v) for v in x.observation.action_mask],
            teacher_action=x.action_id,rewards=[0.,x.reward if x.action_id!='STOP' else -1.])
            for i,x in enumerate(trace.transitions)]
    corpus=PublicMotionRankingCorpus.admit(r,expected_hash=semantic_digest(r))
    p=helper.protocol(64,il_motion_supervision=supervision_record(corpus.fingerprint))
    rebound=[];bound_traces=[]
    for trace,c in zip(traces,contexts):
        cr=c.record();cr['learning_protocol_hash']=semantic_digest(p)
        context=type(c)(freeze_json(cr),semantic_digest(cr));rebound.append(context)
        seal=semantic_digest({'context':context.fingerprint,'observations':[x.observation.fingerprint for x in trace.transitions],
                             'history':trace.history,'behavior_parameter_hash':None})
        bound_traces.append(replace(trace,context=context,seal_hash=seal).require())
    il,rl=common_patient_policies(rebound,p)
    with pytest.raises(ValueError):PatientTrainSession(rebound,'RL',rl,initial_parameter_hash=parameter_hash(rl),protocol=p)
    session=PatientTrainSession(rebound,'IL',il,initial_parameter_hash=parameter_hash(il),protocol=p)
    pins={t.context.fingerprint:{'steps':len(t.transitions),'trace_seal':t.seal_hash,'stop_steps':1} for t in bound_traces}
    with pytest.raises(ValueError):PatientGradientAccumulator(session,teacher_pins=pins)
    with pytest.raises(ValueError):patient_imitation_loss(session,())
    bad=thaw_json(bound_traces[1].history);bad[0]['source_state_hash']='sha256:'+'b'*64
    badseal=semantic_digest({'context':bound_traces[1].context.fingerprint,
        'observations':[x.observation.fingerprint for x in bound_traces[1].transitions],
        'history':bad,'behavior_parameter_hash':None})
    with pytest.raises(ValueError):corpus.require_trace(replace(bound_traces[1],history=freeze_json(bad),seal_hash=badseal))
    with PatientGradientAccumulator(session,teacher_pins=pins,motion_ranking=corpus) as accumulation:
        for trace in bound_traces:accumulation.add_trace(trace)
        receipt=accumulation.finish()
    assert receipt['loss_forward_calls']==29 and receipt['completed_updates']==1
    assert receipt['public_motion_ranking_corpus_hash']==corpus.fingerprint
    assert receipt['module_gradient_norms_before_clip']['critic']==0.
