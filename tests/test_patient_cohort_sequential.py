"""Generated tensor algebra/lifecycle only; no patient arrays or native steps.

Synthetic DTO traces test the loss/update seam, not native trace provenance.
"""
from contextlib import contextmanager
from dataclasses import replace
import json
from pathlib import Path
from types import SimpleNamespace
import weakref

import numpy as np
import pytest
import torch
import resectionlab

STAGE = Path(__file__).resolve().parents[1]/'src/resectionlab'
resectionlab.__path__.insert(0, str(STAGE))
from resectionlab.core import freeze_json, semantic_digest, thaw_json
from resectionlab.geometry import AccessWindow, ToolGeometry
from resectionlab.native_proposals import NominalCavityProposalConfig
from resectionlab.patient_planning_admission import PatientPlanningContext, COHORT_SHA256, _public_observation_binding
from resectionlab.patient_planning_learning import (PatientTrainingTrace, PatientTrainSession,
    PatientImitationSample, _TRACE, common_patient_policies, patient_imitation_loss,
    patient_reinforce_loss, patient_gradient_step)
from resectionlab.patient_planning_accumulation import PatientGradientAccumulator
from resectionlab.patient_planning_cohort_spec import (TRAIN, sequential_learning_protocol,
    sequential_limits, validate_sequential_protocol, validate_limits, preview_budget_sizing)
from resectionlab.patient_planning_cohort_io import (OutputBudget, _save_checkpoint,
    load_cohort_checkpoint, cohort_learning_protocol)
from resectionlab.public_target_context import prepare_public_target
from resectionlab.spatial_observations import (ObservedChannel, SpatialInputs,
    ObservedProcedureState, SpatialAction, build_spatial_observation)
from resectionlab.spatial_policy import SpatialTransition, parameter_hash
from resectionlab import patient_planning_cohort_sequential as runner


@pytest.fixture(autouse=True)
def one_thread():
    torch.set_num_threads(1)


def config(updates=1, horizon=3):
    protocol = sequential_learning_protocol(updates=updates, max_steps=horizon,
        search={'max_calls': max(858,96*(2*horizon-1)), 'beam_width':2, 'seconds':120},
        proposal_config=NominalCavityProposalConfig(intermediate_opening_mm=1.,tool_footprint_opening=True))
    return protocol, sequential_limits(protocol,worker_seconds=1800,memory_bytes=3*1024**3,output_bytes=128*1024**2)


def generated_contexts(protocol):
    contexts=[]; observations=[]
    for index,subject in enumerate(TRAIN):
        shape=(4,4,4); target=np.zeros(shape,np.float32); target[3,index,3]=1.
        source_hash=semantic_digest({'generated_fixture':'sequential-gradient-only','subject':subject})
        prepared=prepare_public_target(nominal_target=target,target_domain=np.ones(shape,bool),
            observed_support=np.ones(shape,bool),affine_ras_mm=np.eye(4),crop_origin=(0,0,0),
            crop_shape=shape,source_hash=source_hash)
        channels={'structural_intensity':ObservedChannel(np.arange(64).reshape(shape)/64.,source_kind='observed_scan'),
            'nominal_tissue':ObservedChannel(np.ones(shape),source_kind='supplied_annotation'),
            'nominal_target':ObservedChannel(target,source_kind='supplied_annotation'),
            'observed_cavity':ObservedChannel(np.zeros(shape),source_kind='observed_procedure_state')}
        inputs=SpatialInputs(channels,np.eye(4),'annotation_assisted',source_hash)
        access=AccessWindow((0.,1.,1.),(1.,0.,0.),3.)
        tool=ToolGeometry('generated_tool',.4,.2,20.,tip_length_mm=.5)
        actions=(SpatialAction('STOP'),SpatialAction('candidate',(0.,1.,1.),(2.,1.,1.),tool))
        obs=build_spatial_observation(inputs,actions,ObservedProcedureState(access,0,protocol['cohort_execution']['max_steps']))
        obs=replace(obs,public_target_context=prepared.observe(np.zeros(shape,bool)))
        record=freeze_json({'scope':'patient_native_planning_experiment','role':'TRAIN',
            'cohort_sha256':COHORT_SHA256,'subject':subject,'patient_group':'ReMIND:'+subject[-3:],
            'evidence_domain':'generated_interface_control','source_hash':source_hash,
            'max_optimizer_updates':2*protocol['updates_per_method'],
            'max_steps':protocol['cohort_execution']['max_steps'],'observation_track':obs.track,
            'public_observation_binding':_public_observation_binding(obs),
            'learning_protocol_hash':semantic_digest(protocol),
            'public_target_context_variant':protocol['public_target_context_variant']})
        contexts.append(PatientPlanningContext(record,semantic_digest(record))); observations.append(obs)
    return tuple(contexts),observations


def synthetic_trace(context,obs,length,behavior=None):
    rows=[]
    for step in range(length):
        state=obs.state_features.copy();state[0]=step
        observation=replace(obs,state_features=state)
        action='STOP' if step==length-1 else 'candidate'
        reward=0. if action=='STOP' else (step+1)*(.25+length/10)
        rows.append(SpatialTransition(observation,action,reward,step==length-1))
    history=freeze_json([{'action_id':r.action_id,'reward':r.reward} for r in rows])
    seal=semantic_digest({'context':context.fingerprint,'observations':[r.observation.fingerprint for r in rows],
        'history':history,'behavior_parameter_hash':behavior})
    return PatientTrainingTrace(context,tuple(rows),history,behavior,seal,_TRACE).require()


def session_for(contexts,protocol,method):
    il,rl=common_patient_policies(contexts,protocol);policy=il if method=='IL' else rl
    return PatientTrainSession(contexts,method,policy,initial_parameter_hash=parameter_hash(policy),protocol=protocol)


@pytest.mark.parametrize('method',['IL','RL'])
def test_unequal_traces_match_objective_gradients_and_two_Adam_updates(method):
    protocol,_=config(updates=2);contexts,observations=generated_contexts(protocol)
    reference=session_for(contexts,protocol,method);sequential=session_for(contexts,protocol,method)
    for update in range(2):
        # Each implementation gets current-policy seals; numerical weight hashes
        # can differ even when gradients/parameters agree within stated tolerance.
        def traces_for(session):
            return tuple(synthetic_trace(c,o,n,parameter_hash(session.policy) if method=='RL' else None)
                for c,o,n in zip(contexts,observations,(1,2,3,2)))
        reference_traces=traces_for(reference);sequential_traces=traces_for(sequential)
        if method=='IL':
            loss,details=patient_imitation_loss(reference,
                [PatientImitationSample(t,i) for t in reference_traces for i in range(len(t.transitions))])
        else:loss,details=patient_reinforce_loss(reference,reference_traces)
        expected=patient_gradient_step(reference,loss)
        pins={t.context.fingerprint:{'steps':len(t.transitions),'trace_seal':t.seal_hash} for t in sequential_traces}
        with PatientGradientAccumulator(sequential,teacher_pins=pins if method=='IL' else None) as transaction:
            for trace in sequential_traces:transaction.add_trace(trace)
            actual=transaction.finish()
        assert actual['loss']==pytest.approx(details['loss'],abs=1e-6,rel=1e-6)
        assert actual['gradient_norm_before_clip']==pytest.approx(expected['gradient_norm_before_clip'],abs=1e-6,rel=1e-5)
        assert actual['completed_updates']==expected['completed_updates']==update+1
        for (name,a),(other,b) in zip(reference.policy.named_parameters(),sequential.policy.named_parameters()):
            assert name==other
            torch.testing.assert_close(a,b,atol=2e-6,rtol=2e-5)
            if a.grad is None:assert b.grad is None
            else:torch.testing.assert_close(a.grad,b.grad,atol=1e-6,rtol=1e-5)
        assert all(int(s['step'])==update+1 for s in sequential._optimizer.state.values())


def test_no_trace_retained_and_partial_update_cannot_retry():
    protocol,_=config();contexts,observations=generated_contexts(protocol)
    session=session_for(contexts,protocol,'RL');before=parameter_hash(session.policy)
    trace=synthetic_trace(contexts[0],observations[0],2,before);reference=weakref.ref(trace)
    transaction=PatientGradientAccumulator(session);transaction.add_trace(trace);del trace
    assert reference() is None and transaction.rows[0]['steps']==2
    with pytest.raises(ValueError,match='Every admitted TRAIN'):transaction.finish()
    assert parameter_hash(session.policy)==before and session.updates==0
    assert all(p.grad is None for p in session.policy.parameters())
    with pytest.raises(ValueError):PatientGradientAccumulator(session)


@pytest.mark.parametrize('failure',['duplicate','stale','guard','backward'])
def test_bad_contribution_aborts_without_step(failure,monkeypatch):
    protocol,_=config();contexts,observations=generated_contexts(protocol)
    session=session_for(contexts,protocol,'RL');before=parameter_hash(session.policy)
    first=synthetic_trace(contexts[0],observations[0],2,before)
    transaction=PatientGradientAccumulator(session);transaction.add_trace(first)
    trace=first if failure=='duplicate' else synthetic_trace(contexts[1],observations[1],2,
        'sha256:'+'0'*64 if failure=='stale' else before)
    def fail(*args,**kwargs):raise InterruptedError('generated mid-patient failure')
    if failure=='backward':monkeypatch.setattr(torch.Tensor,'backward',fail)
    with pytest.raises((ValueError,InterruptedError)):
        transaction.add_trace(trace,**({'guard':fail} if failure=='guard' else {}))
    assert parameter_hash(session.policy)==before and session.updates==0
    assert all(p.grad is None for p in session.policy.parameters())


def test_explicit_configuration_horizon_and_reconstruction_budget():
    protocol,limits=config(updates=8,horizon=24);sizing=preview_budget_sizing(protocol)
    assert sizing['visits_per_patient']==19
    assert limits['max_policy_forwards']==2496 and limits['search']['max_calls']==4512
    for key,value in [('retention_mode','return_only'),('heldout_execution',True)]:
        changed=thaw_json(protocol);changed['cohort_execution'][key]=value
        with pytest.raises(ValueError):validate_sequential_protocol(changed)
    with pytest.raises(ValueError):
        sequential_learning_protocol(updates=1,max_steps=24,
            search={'max_calls':858,'beam_width':2,'seconds':120},
            proposal_config=NominalCavityProposalConfig(intermediate_opening_mm=1.,tool_footprint_opening=True))


def test_default_mode_identity_and_both_explicit_caps():
    original, old_limits = config(updates=8, horizon=24)
    explicit = sequential_learning_protocol(updates=8, max_steps=24,
        search={'max_calls':4512,'beam_width':2,'seconds':120},
        proposal_config=NominalCavityProposalConfig(intermediate_opening_mm=1.,tool_footprint_opening=True),
        retention_mode='return_plus_opening_depth_v1')
    assert semantic_digest(explicit) == semantic_digest(original)
    widened = sequential_learning_protocol(updates=8, max_steps=24,
        search={'max_calls':5640,'beam_width':2,'seconds':120},
        proposal_config=NominalCavityProposalConfig(max_candidates=120,intermediate_opening_mm=1.,tool_footprint_opening=True),
        retention_mode='return_plus_opening_depth_volume_v1')
    limits = sequential_limits(widened, worker_seconds=1800,memory_bytes=3*1024**3,output_bytes=128*1024**2)
    assert semantic_digest(widened) != semantic_digest(original)
    sizing = preview_budget_sizing(widened)
    assert sizing['candidate_inventory_ceiling'] == 120
    assert limits['max_native_previews'] == 4*120*(19*73+1+5640) == 3373440
    assert limits['max_policy_forwards'] == old_limits['max_policy_forwards'] == 2496
    with pytest.raises(ValueError): validate_limits(widened, old_limits)


@pytest.mark.parametrize('change', ['unknown_mode','one_lane','intermediate_cap','undersized_calls','missing_mode'])
def test_search_choice_schema_refuses_unsupported_or_incomplete_options(change):
    kwargs = {'updates':1,'max_steps':24,'search':{'max_calls':5640,'beam_width':2,'seconds':120},
        'proposal_config':NominalCavityProposalConfig(max_candidates=120,intermediate_opening_mm=1.,tool_footprint_opening=True),
        'retention_mode':'return_plus_opening_depth_volume_v1'}
    if change == 'unknown_mode': kwargs['retention_mode']='unreviewed_mode'
    elif change == 'one_lane': kwargs['search']['beam_width']=1
    elif change == 'intermediate_cap': kwargs['proposal_config']=NominalCavityProposalConfig(
        max_candidates=100,intermediate_opening_mm=1.,tool_footprint_opening=True)
    elif change == 'undersized_calls': kwargs['search']['max_calls']=4512
    else:
        protocol=thaw_json(sequential_learning_protocol(**kwargs))
        protocol['cohort_execution'].pop('retention_mode')
        with pytest.raises(ValueError): validate_sequential_protocol(protocol)
        return
    with pytest.raises(ValueError): sequential_learning_protocol(**kwargs)


@pytest.mark.parametrize('key,value', [('threads',2),('max_native_previews',1),('max_policy_forwards',1),
    ('worker_seconds',3601),('memory_bytes',3*1024**3+1),('output_bytes',256*1024**2+1)])
def test_runtime_limits_refuse_mismatch_or_weakened_envelope(key,value):
    protocol,limits=config();changed=thaw_json(limits);changed[key]=value
    with pytest.raises(ValueError):validate_limits(protocol,changed)


@pytest.mark.parametrize('new_protocol',[False,True])
def test_pinned_weight_roundtrip_preserves_old_and_new_protocols(tmp_path,new_protocol):
    context_protocol,limits=config();contexts,_=generated_contexts(context_protocol)
    protocol=context_protocol if new_protocol else cohort_learning_protocol(1)
    if not new_protocol:
        repaired=[]
        for c in contexts:
            record=c.record();record['learning_protocol_hash']=semantic_digest(protocol)
            record.pop('public_target_context_variant')
            repaired.append(PatientPlanningContext(freeze_json(record),semantic_digest(record)))
        contexts=tuple(repaired)
    il,_=common_patient_policies(contexts,protocol);sink=OutputBudget(tmp_path,limits['output_bytes'])
    saved=_save_checkpoint(il,method='IL',contexts=contexts,protocol=protocol,
        updates=1,initial_hash=parameter_hash(il),output=sink,limits=limits)
    loaded,metadata=load_cohort_checkpoint(tmp_path/saved['path'],expected_sha256=saved['sha256'],
        expected_learning_protocol=protocol,expected_context_hashes={c.patient_group:c.fingerprint for c in contexts},
        expected_method='IL')
    assert parameter_hash(loaded)==parameter_hash(il)==metadata['parameter_hash']
    assert all(torch.equal(a,b) for a,b in zip(il.parameters(),loaded.parameters()))


@pytest.fixture
def fake_runtime(monkeypatch):
    protocol,limits=config();events=[];live=weakref.WeakSet()
    class Case:pass
    class Context:
        def __init__(self,s):self.subject=s;self.fingerprint=s+'-context';self.patient_group=s
        def record(self):return {'subject':self.subject,'generated_only':True}
    class Base:
        def __init__(self,s):self.subject=s;self.case=Case();live.add(self.case)
    class Policy:
        def __init__(self):self.hash='generated-common'
    class Session:
        def __init__(self,contexts,method,policy,**kwargs):
            self.contexts=contexts;self.method=method;self.policy=policy;self.updates=0
    class Accumulator:
        def __init__(self,session,**kwargs):self.session=session;self.rows=[];self.before=session.policy.hash
        def __enter__(self):return self
        def __exit__(self,*args):return False
        def add_trace(self,trace,*,guard):
            guard();assert self.session.policy.hash==self.before
            self.rows.append(trace.context.subject);events.append(('contribution',self.session.method,trace.context.subject))
            return {'trace_seal':trace.seal_hash}
        def finish(self,*,guard):
            guard();assert self.rows==list(TRAIN)
            self.session.updates+=1;self.session.policy.hash+=self.session.method
            events.append(('step',self.session.method));return {'completed_updates':self.session.updates}
    class Budget:
        def __init__(self,*args,**kwargs):self.done=False
        def __enter__(self):return self
        def __exit__(self,*args):return False
        def check(self):assert not self.done
        def complete(self,**kwargs):self.done=True
        def snapshot(self):return {'generated_only':True,'native_preview_entries':0}
    class Costs:
        def __init__(self,*args):self.rows={};self.forwards=0
        def __enter__(self):return self
        def __exit__(self,*args):return False
        @contextmanager
        def scope(self,phase):self.rows.setdefault(phase,{});yield
    def source(base,context,subject,*args):assert base.subject==context.subject==subject
    def common(contexts,protocol):
        assert len(live)==0;events.append(('common',));return Policy(),Policy()
    def collect(base,context,*,actions=None,**kwargs):
        assert len(live)==1
        sequence=('candidate','STOP') if actions is None else actions
        return SimpleNamespace(context=context,seal_hash=context.subject+'-teacher',
            transitions=tuple(SimpleNamespace(reward=0. if a=='STOP' else 1.,action_id=a) for a in sequence))
    def search(base,**kwargs):
        assert kwargs['retention_mode']=='return_plus_opening_depth_v1'
        return ('candidate','STOP'),{'call_cap_reached':False,'time_cap_reached':False}
    def save(policy,**kwargs):return {'path':kwargs['method']+'.psckpt','sha256':'0'*64,'bytes':1,'parameter_hash':policy.hash}
    for name,value in [('_require_source',source),('PatientTrainSession',Session),('PatientGradientAccumulator',Accumulator),
                       ('PlanningBudget',Budget),('common_patient_policies',common),('parameter_hash',lambda p:p.hash),
                       ('_save_checkpoint',save)]:
        monkeypatch.setattr(runner,name,value)
    monkeypatch.setattr(runner.preflight,'_CallCosts',Costs)
    monkeypatch.setattr(runner.preflight,'_collect',collect)
    monkeypatch.setattr(runner.preflight,'_seal_and_replay',lambda *a,**k:{'plan_seal':'generated-plan'})
    monkeypatch.setattr(runner.preflight,'observed_beam_search',search)
    monkeypatch.setattr(runner.torch,'get_num_interop_threads',lambda:1)
    def factory(subject):
        def build(*,output):
            assert len(live)==0;events.append(('factory',subject));return Base(subject),Context(subject)
        return build
    return protocol,limits,{s:factory(s) for s in TRAIN},events,live,Budget


def test_one_source_shared_updates_complete_greedy_and_cost_split(tmp_path,fake_runtime):
    protocol,limits,factories,events,live,_=fake_runtime
    result=runner.run_train_cohort_sequential(factories,learning_protocol=protocol,limits=limits,output=tmp_path/'run')
    assert len(live)==0 and result['selection_readiness']['ready'] is True
    assert result['selection_readiness']['execution_admitted'] is False
    assert [e[1] for e in events if e[0]=='factory']==list(TRAIN)*5
    assert [e for e in events if e[0]=='step']==[('step','IL'),('step','RL')]
    costs=json.loads((tmp_path/'run/costs.json').read_text())
    assert len(costs['completed_patient_visits'])==20
    assert any(k.startswith('offline.') for k in costs['costs'])
    assert any(k.startswith('deployment.') for k in costs['costs'])


def test_late_completion_failure_clears_readiness(tmp_path,fake_runtime,monkeypatch):
    protocol,limits,factories,events,live,Budget=fake_runtime
    def fail(self,**kwargs):raise InterruptedError('generated late completion failure')
    monkeypatch.setattr(Budget,'complete',fail)
    with pytest.raises(InterruptedError):
        runner.run_train_cohort_sequential(factories,learning_protocol=protocol,limits=limits,output=tmp_path/'run')
    result=json.loads((tmp_path/'run/result.json').read_text())
    assert result['selection_readiness']=={'ready':False,'execution_admitted':False}


def test_all_STOP_gate_prevents_initialization(tmp_path,fake_runtime,monkeypatch):
    protocol,limits,factories,events,live,_=fake_runtime
    monkeypatch.setattr(runner.preflight,'observed_beam_search',lambda *a,**k:
        (('STOP',),{'call_cap_reached':False,'time_cap_reached':False}))
    with pytest.raises(InterruptedError,match='lack positive'):
        runner.run_train_cohort_sequential(factories,learning_protocol=protocol,limits=limits,output=tmp_path/'run')
    assert not any(e[0]=='common' for e in events)
    assert [e[1] for e in events if e[0]=='factory']==list(TRAIN)


def test_retained_source_blocks_next_patient(tmp_path,fake_runtime):
    protocol,limits,factories,events,live,_=fake_runtime
    original=factories[TRAIN[0]];cache=[]
    def caching(**kwargs):
        base,context=original(**kwargs);cache.append(base.case);return base,context
    factories[TRAIN[0]]=caching
    with pytest.raises(RuntimeError,match='retained after visit'):
        runner.run_train_cohort_sequential(factories,learning_protocol=protocol,limits=limits,output=tmp_path/'run')
    assert [e[1] for e in events if e[0]=='factory']==[TRAIN[0]]


def test_changed_context_blocks_first_training_contribution(tmp_path,fake_runtime):
    protocol,limits,factories,events,live,_=fake_runtime
    original=factories[TRAIN[0]];count=[0]
    def changed(**kwargs):
        base,context=original(**kwargs);count[0]+=1
        if count[0]>1:context.fingerprint+='drift'
        return base,context
    factories[TRAIN[0]]=changed
    with pytest.raises(ValueError,match='context changed'):
        runner.run_train_cohort_sequential(factories,learning_protocol=protocol,limits=limits,output=tmp_path/'run')
    assert not any(e[0]=='step' for e in events)


def test_heldout_factory_refused_before_open(tmp_path,fake_runtime):
    protocol,limits,factories,events,live,_=fake_runtime
    factories['ReMIND-013']=lambda **k:pytest.fail('heldout access')
    with pytest.raises(ValueError,match='SELECT/EVAL'):
        runner.run_train_cohort_sequential(factories,learning_protocol=protocol,limits=limits,output=tmp_path/'run')
    assert events==[] and not (tmp_path/'run').exists()
