"""Exact execute body with generated DTOs and simulated source/native boundaries.

Real policies,64 IL/8 RL shared updates, cache, checkpoint save/reload, output,
forward accounting and source-lifetime checks. No patient arrays/native steps.
Scripted learned routes test completion semantics, not greedy correctness.
"""
import importlib.util
import json
from pathlib import Path
import sys
from types import SimpleNamespace
import weakref

import pytest
import torch

HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[1]
sys.path.insert(0,str(HERE));sys.path.insert(0,str(ROOT/'src'));sys.path.insert(0,str(ROOT/'tests'))
import pilot_contract as contract
import cohort_worker as worker
from resectionlab.core import freeze_json,semantic_digest,thaw_json
from resectionlab.patient_planning_admission import PatientPlanningContext
from resectionlab.patient_planning_learning import PatientTrainingTrace,_TRACE
from resectionlab.spatial_policy import parameter_hash
from resectionlab import patient_planning_cohort_sequential as sequential
from resectionlab import patient_planning_cohort_visits as factory_module
from resectionlab import patient_planning_preflight as preflight
from test_patient_cohort_sequential import generated_contexts,synthetic_trace
from test_patient_teacher_trace_cache import simulated_receipt


def test_complete_two_method_worker_flow(tmp_path,monkeypatch):
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    monkeypatch.setattr(worker,'ROOT',tmp_path)
    monkeypatch.setattr(worker,'COHORT','cohort.json');(tmp_path/'cohort.json').write_text('{}')
    outcomes=[];live=weakref.WeakSet();events=[];collector_calls=[]
    class Case:pass
    class Base:
        def __init__(self,subject):self.subject=subject;self.case=Case();live.add(self.case)
    for method in ('IL','RL'):
        protocol,limits=contract.canonical_configuration(method)
        expected=contract.expected_configuration(method)
        assert contract.canonical((protocol,limits))==contract.canonical(expected)
        contexts,observations=generated_contexts(protocol);fixed=[]
        for context in contexts:
            record=context.record();record['occupancy_condition']=contract.OCCUPANCY;record['decision_model_hash']=semantic_digest({'generated_model':record['subject']})
            fixed.append(PatientPlanningContext(freeze_json(record),semantic_digest(record)))
        by_subject={s:(c,o) for s,c,o in zip(contract.TRAIN,fixed,observations)}
        baseline={'initial_result':{'initial_parameter_hash':contract.inputs()['initial_result']['initial_parameter_hash']}}
        for subject,(context,obs) in by_subject.items():
            trace=synthetic_trace(context,obs,4 if subject==contract.TRAIN[-1] else 1)
            prior=thaw_json(simulated_receipt(trace)['plan'])
            prior['context_hash']='generated-prior-search-context'
            for row in prior['history']:row['outcome_scope']='separate_evaluator_reference'
            baseline['plan_'+subject]={'plan':prior}
        monkeypatch.setattr(worker,'inputs',lambda:baseline)
        def require_source(base,context,subject,actual_protocol,actual_limits):
            assert base.subject==subject==context.record()['subject']
            context.require_training()
            assert semantic_digest(actual_protocol)==context.record()['learning_protocol_hash']
            assert contract.canonical(actual_limits)==contract.canonical(limits)
        monkeypatch.setattr(sequential,'_require_source',require_source)
        def factory(**kwargs):
            assert kwargs['learning_protocol']['cohort_execution']['occupancy_condition']==contract.OCCUPANCY
            def build(subject):
                def visit(*,output):
                    assert not live;events.append((method,subject));return Base(subject),by_subject[subject][0]
                return visit
            return {s:build(s) for s in contract.TRAIN}
        monkeypatch.setattr(factory_module,'make_train_visit_factories',factory)
        def collect(base,context,*,actions=None,policy=None,generator=None,output,guard):
            guard();assert len(live)==1
            obs=by_subject[base.subject][1]
            if actions is not None:
                trace=synthetic_trace(context,obs,len(actions));assert [r.action_id for r in trace.transitions]==list(actions)
            else:
                # Script valid complete STOP and different, negative, or equal-return
                # routes. Collector correctness has separate canonical tests.
                length=1 if base.subject==contract.TRAIN[0] else 2
                trace=synthetic_trace(context,obs,length,parameter_hash(policy))
                from dataclasses import replace
                rows=tuple(replace(r,reward=(0. if r.action_id=='STOP' else (-.5 if base.subject==contract.TRAIN[1] else 3.9))) for r in trace.transitions)
                history=freeze_json([{'action_id':r.action_id,'reward':r.reward} for r in rows])
                seal=semantic_digest({'context':context.fingerprint,'observations':[r.observation.fingerprint for r in rows],
                    'history':history,'behavior_parameter_hash':parameter_hash(policy)})
                trace=PatientTrainingTrace(context,rows,history,parameter_hash(policy),seal,_TRACE).require()
                with torch.no_grad():
                    for row in rows:policy(row.observation)
            collector_calls.append((method,base.subject,actions is not None,generator is not None))
            return trace
        def seal(base,context,trace,*,method,policy,updates,output,guard):
            guard();record=simulated_receipt(trace);record['method']=method
            plan=thaw_json(record['plan'])
            if policy is not None:plan.update(parameter_hash=parameter_hash(policy),architecture_hash=policy.architecture_hash,learning_updates=updates)
            record.update(plan=freeze_json(plan),plan_seal=semantic_digest(plan))
            preflight._write(output/'plan.json',{'plan':record['plan'],'plan_seal':record['plan_seal']})
            preflight._write(output/'replay.json',{'history':record['replayed_history'],'generated_simulated_native':True})
            return record
        monkeypatch.setattr(preflight,'_collect',collect);monkeypatch.setattr(preflight,'_seal_and_replay',seal)
        release={'method':method,'learning_protocol':protocol,'cohort_limits':limits}
        output=tmp_path/method;before=len(events)
        result=worker.execute(release,'generated-only',output,lambda *a,**k:None)
        assert contract.complete_result(result) and not live
        assert len(events)-before==contract.METHODS[method]['source_visits']
        assert result['TRAIN_greedy'][contract.TRAIN[0]]['actions']==['STOP']
        assert result['TRAIN_greedy'][contract.TRAIN[1]]['public_return']==-.5
        assert result['TRAIN_greedy'][contract.TRAIN[-1]]['actions']!=baseline['plan_'+contract.TRAIN[-1]]['plan']['actions']
        assert result['TRAIN_greedy'][contract.TRAIN[-1]]['public_return']==pytest.approx(3.9)
        assert all(row['checkpoint_reloaded'] for row in result['TRAIN_greedy'].values())
        assert result['checkpoint_loads']==1 and result['teacher_logit_forwards']==7
        assert result['optimizer_updates']=={'IL':64 if method=='IL' else 0,'RL':8 if method=='RL' else 0}
        costs=json.loads((output/'costs.json').read_text())
        assert costs['total_policy_forward_calls']==(448+7+7 if method=='IL' else 56*2+7+7)
        if method=='RL':
            decisions=0
            for path in (output/'RL').glob('update-*/*/gradient-contribution.json'):
                row=json.loads(path.read_text());data=row['rl_decision_diagnostics']
                assert data['behavior_parameter_hash']
                decisions+=len(data['decisions'])
            assert decisions==56
        assert json.loads((output/'result.json').read_text())==result
        outcomes.append(result)
        with pytest.raises(FileExistsError):worker.execute(release,'generated-only',output,lambda *a,**k:None)
    assert outcomes[0]['initial_parameter_hash']==outcomes[1]['initial_parameter_hash']
    assert sum(fresh for method,subject,teacher,fresh in collector_calls if method=='RL')==32
    # Same entry body refuses a physically changed teacher before initialization.
    baseline['plan_'+contract.TRAIN[0]]['plan']['history'][0]['reward']=123.
    refused=tmp_path/'bad-teacher'
    with pytest.raises(ValueError,match='teacher world/actions/history'):
        worker.execute(release,'generated-only',refused,lambda *a,**k:None)
    failed=json.loads((refused/'result.json').read_text())
    assert failed['optimizer_updates']=={'IL':0,'RL':0} and failed['selection_readiness']['ready'] is False
    assert not live
