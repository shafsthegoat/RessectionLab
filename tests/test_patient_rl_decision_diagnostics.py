"""Generated diagnostic parity only; no patient, search or native trajectories."""
from dataclasses import replace
import json

import pytest
import torch

from resectionlab.core import freeze_json, semantic_digest
from resectionlab.patient_planning_accumulation import PatientGradientAccumulator
from resectionlab.patient_planning_learning import PatientTrainingTrace, _TRACE
from resectionlab.spatial_policy import parameter_hash
from test_patient_cohort_sequential import config, generated_contexts, session_for, synthetic_trace


@pytest.fixture(autouse=True)
def one_thread(): torch.set_num_threads(1)


def traces(contexts, observations, before):
    result=[]
    # Includes positive and negative returns, a positive immediate action with
    # negative future return, and a complete STOP-only episode.
    for context,obs,rewards in zip(contexts,observations,((0.,),(.75,0.),(.2,-.7,0.),(-.125,0.))):
        template=synthetic_trace(context,obs,len(rewards),before)
        rows=tuple(replace(row,reward=reward) for row,reward in zip(template.transitions,rewards))
        history=freeze_json([{'action_id':row.action_id,'reward':row.reward} for row in rows])
        seal=semantic_digest({'context':context.fingerprint,
            'observations':[row.observation.fingerprint for row in rows],
            'history':history,'behavior_parameter_hash':before})
        result.append(PatientTrainingTrace(context,rows,history,before,seal,_TRACE).require())
    return tuple(result)


def test_diagnostics_preserve_all_math_rng_and_forward_counts_for_two_updates():
    protocol,_=config(updates=2);contexts,observations=generated_contexts(protocol)
    plain=session_for(contexts,protocol,'RL');logged=session_for(contexts,protocol,'RL')
    counts={'plain':0,'logged':0};captured=[]
    def plain_hook(module,args,output):counts['plain']+=1
    def logged_hook(module,args,output):
        counts['logged']+=1;captured.append((output[0].detach().clone(),output[1].detach().clone()))
    handles=[plain.policy.register_forward_hook(plain_hook),logged.policy.register_forward_hook(logged_hook)]
    try:
        for update in range(2):
            before=parameter_hash(plain.policy)
            assert before==parameter_hash(logged.policy)
            episodes=traces(contexts,observations,before)
            rng=torch.random.get_rng_state().clone();plain_rows=[];logged_rows=[];captured.clear()
            with PatientGradientAccumulator(plain) as transaction:
                for trace in episodes:plain_rows.append(transaction.add_trace(trace))
                plain_result=transaction.finish()
            assert torch.equal(rng,torch.random.get_rng_state())
            with PatientGradientAccumulator(logged,rl_diagnostics=True) as transaction:
                for trace in episodes:logged_rows.append(transaction.add_trace(trace))
                logged_result=transaction.finish()
            assert torch.equal(rng,torch.random.get_rng_state())
            assert plain_result==logged_result
            assert parameter_hash(plain.policy)==parameter_hash(logged.policy)
            for a,b in zip(plain.policy.parameters(),logged.policy.parameters()):
                assert torch.equal(a,b)
                assert (a.grad is None)==(b.grad is None)
                if a.grad is not None:assert torch.equal(a.grad,b.grad)
            assert counts=={'plain':8*(update+1),'logged':8*(update+1)}
            offset=0
            for ordinary,receipt,trace in zip(plain_rows,logged_rows,episodes):
                extra=receipt.pop('rl_decision_diagnostics')
                assert receipt==ordinary
                assert extra['version']=='patient-RL-decision-scalars-v1'
                assert extra['behavior_parameter_hash']==before
                assert extra['episodes_in_update']==4
                assert len(extra['decisions'])==len(trace.transitions)
                episode_actor=episode_value=episode_entropy=0.
                for step,(decision,row) in enumerate(zip(extra['decisions'],trace.transitions)):
                    logits,value=captured[offset];offset+=1
                    target=sum(r.reward for r in trace.transitions[step:])
                    distribution=torch.distributions.Categorical(logits=logits)
                    index=row.observation.action_ids.index(row.action_id)
                    logp=distribution.log_prob(logits.new_tensor(index,dtype=torch.long))
                    advantage=value.new_tensor(target)-value
                    assert decision['step']==step and decision['observation_hash']==row.observation.fingerprint
                    assert decision['action_id']==row.action_id and decision['action_index']==index
                    assert decision['legal_action_count']==int(row.observation.action_mask.sum())
                    assert decision['reward']==row.reward and decision['terminated']==row.terminated
                    assert decision['return_to_go']==pytest.approx(target,abs=1e-15)
                    assert decision['return_to_go_model_dtype']==float(value.new_tensor(target))
                    assert decision['value']==float(value)
                    assert decision['detached_advantage']==float(advantage)
                    assert decision['chosen_log_probability']==float(logp)
                    assert decision['entropy']==float(distribution.entropy())
                    assert decision['actor_discount']==1.
                    assert decision['actor_score_term']==float(-logp*advantage)
                    assert decision['value_squared_error']==float((value-target).square())
                    episode_actor+=decision['actor_score_term']/4
                    episode_value+=decision['value_squared_error']/(4*len(trace.transitions))
                    episode_entropy+=decision['entropy']/(4*len(trace.transitions))
                assert episode_actor==ordinary['actor_loss']
                assert episode_value==ordinary['value_loss']
                assert episode_entropy==ordinary['entropy']
                assert (episode_actor+extra['value_weight']*episode_value
                    -extra['entropy_weight']*episode_entropy)==pytest.approx(ordinary['loss'],abs=1e-6)
                # Only JSON scalars/containers survive; no tensor or graph.
                json.dumps(extra,allow_nan=False)
    finally:
        for handle in handles:handle.remove()


@pytest.mark.parametrize('flag',[1,None,'yes'])
def test_malformed_option_refuses_before_any_update_ownership(flag):
    protocol,_=config();contexts,_=generated_contexts(protocol);session=session_for(contexts,protocol,'RL')
    with pytest.raises(ValueError,match='explicit boolean'):
        PatientGradientAccumulator(session,rl_diagnostics=flag)
    assert session._permit is None and not session._attempted and session.updates==0


def test_il_cannot_enable_rl_diagnostics():
    protocol,_=config();contexts,_=generated_contexts(protocol);session=session_for(contexts,protocol,'IL')
    with pytest.raises(ValueError,match='RL session'):
        PatientGradientAccumulator(session,rl_diagnostics=True)
    assert session._permit is None and not session._attempted and session.updates==0


def test_diagnostic_collection_does_not_weaken_stale_trace_refusal(monkeypatch):
    protocol,_=config();contexts,observations=generated_contexts(protocol);session=session_for(contexts,protocol,'RL')
    before=parameter_hash(session.policy);stale=traces(contexts,observations,'sha256:'+'0'*64)[0]
    monkeypatch.setattr(session.policy,'forward',lambda *args:(_ for _ in ()).throw(AssertionError('forward')))
    transaction=PatientGradientAccumulator(session,rl_diagnostics=True)
    with pytest.raises(ValueError,match='Stale on-policy'):transaction.add_trace(stale)
    assert transaction.rows==[] and session.updates==0 and parameter_hash(session.policy)==before
    assert all(p.grad is None for p in session.policy.parameters())
