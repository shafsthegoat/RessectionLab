"""Unrun complete runner glue on generated boundaries and real helper/autograd.

Factory, native replay, cache admission and checkpoint I/O are explicit doubles;
the actual execute function, output sink, helper, production loss and autograd run.
No patient files, checkpoint payloads, numerical native geometry or optimizer.
"""
from contextlib import contextmanager,nullcontext
import importlib.util
from pathlib import Path
from types import SimpleNamespace
import json,sys

import pytest
import torch

HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[1]
sys.path.insert(0,str(HERE))
import initial_inventory_worker as worker
import batch_contract as contract

def module(path,name):
    spec=importlib.util.spec_from_file_location(name,path)
    result=importlib.util.module_from_spec(spec);sys.modules[name]=result;spec.loader.exec_module(result)
    return result

fixture_source=module(ROOT/'build/public-motion-ranking-v1/frozen-gradient-diagnostic-v1/test_gradient_diagnostic_generated.py','gradient_fixture_owned')

@pytest.mark.parametrize('failure',[None,'second_collection','helper_readout'])
def test_actual_execute_generated_success_and_failure_cleanup(tmp_path,monkeypatch,failure):
    from resectionlab.core import thaw_json,semantic_digest
    from resectionlab import patient_planning_cohort_visits as factory_module
    from resectionlab import patient_planning_cohort_sequential as sequential
    from resectionlab import patient_planning_cohort_io as io_module
    from resectionlab import patient_planning_preflight as preflight
    from resectionlab import patient_teacher_trace_cache as cache_module
    from resectionlab import planning_budget as budget_module
    from resectionlab import native_spatial_task as native_module
    from resectionlab.spatial_policy import parameter_hash
    before_threads=torch.get_num_threads();torch.set_num_threads(1)
    f=fixture_source.build_fixture(monkeypatch)
    f.policy.requires_grad_(False)
    protocol=thaw_json(f.cache.protocol);limits={'output_bytes':16*1024**2}
    historical={};pins={};plans={}
    for subject,trace in f.traces.items():
        plan={'actions':[r.action_id for r in trace.transitions],'history':trace.history,
            'context_hash':trace.context.fingerprint,'source_hash':fixture_source.H}
        plans[subject]=plan
        pins[subject]={'trace_seal':trace.seal_hash,'context_hash':trace.context.fingerprint,'plan_seal':semantic_digest(plan)}
        historical['plan_'+subject]={'plan':plan}
    saved={'teacher_pins':pins,'release':{'historical':True},**f.readouts}
    if failure=='helper_readout':saved[f'{worker.SUBJECTS[1]}:0']['scores']['logits'][1]+=.125
    record=thaw_json(f.corpus._record)
    fake_owner=SimpleNamespace(PUBLIC_INDEX='generated-index.json',PUBLIC_SHA='generated',COHORT='generated-cohort.json',
        RANKING_HASH=f.corpus.fingerprint,LABEL_REFS={'public-score-corpus.json':{}},
        canonical_configuration=lambda method:(protocol,limits),inputs=lambda:historical,small=lambda ref:record)
    (tmp_path/'generated-cohort.json').write_text('{}')
    monkeypatch.setattr(worker,'ROOT',tmp_path);monkeypatch.setattr(worker,'OUTPUT',tmp_path/'attempt')
    monkeypatch.setattr(worker,'metadata',lambda:(fake_owner,saved,{'release':{'sha256':'2'*64}}))
    monkeypatch.setattr(worker,'source_module',lambda ref,name:f.helper)
    monkeypatch.setattr(worker,'CHECKPOINT',{**worker.CHECKPOINT,'parameter_hash':parameter_hash(f.policy)})
    def factories(**kwargs):
        assert kwargs['released_record']=={'historical':True}
        assert kwargs['released_sha256']=='2'*64
        return {s:None for s in worker.SUBJECTS}
    monkeypatch.setattr(factory_module,'make_train_visit_factories',factories)
    class Budget:
        def __init__(self,*a,**k):self.complete_called=False
        def __enter__(self):return self
        def __exit__(self,*a):pass
        def check(self):pass
        def snapshot(self):return {'native_preview_entries':0}
        def complete(self,**kwargs):self.complete_called=True
    class Costs:
        def __init__(self,*a,**k):self.rows={}
        @property
        def forwards(self):return f.policy.forward_calls
        def __enter__(self):return self
        def __exit__(self,*a):pass
        def scope(self,*a):return nullcontext()
    class Visits:
        def __init__(self,*a):self.contexts={};self.completed=[]
        def run(self,subject,dest,phase,callback):
            context=f.traces[subject].context;self.contexts[subject]=context
            result=callback(SimpleNamespace(subject=subject),context)
            self.completed.append({'subject':subject,'source_released':True});return result
    class Cache(fixture_source.GeneratedCache):
        def __init__(self,protocol):super().__init__(protocol,{})
        def add_replayed(self,trace,sealed):self.traces[trace.context.patient_group]=trace
    monkeypatch.setattr(budget_module,'PlanningBudget',Budget)
    monkeypatch.setattr(preflight,'_CallCosts',Costs)
    monkeypatch.setattr(sequential,'_SequentialVisits',Visits)
    monkeypatch.setattr(sequential,'_bounded_writes',lambda sink:nullcontext())
    monkeypatch.setattr(cache_module,'PatientTeacherTraceCache',Cache)
    def transition(self,action):return None
    monkeypatch.setattr(native_module.NativeSpatialTask,'_transition',transition)
    def collect(base,context,*,actions,**kwargs):
        if failure=='second_collection' and base.subject==worker.SUBJECTS[1]:raise RuntimeError('generated collection failure')
        for action in actions:native_module.NativeSpatialTask._transition(base,action)
        return f.traces[base.subject]
    def replay(base,context,trace,**kwargs):
        for row in trace.transitions:native_module.NativeSpatialTask._transition(base,row.action_id)
        plan=plans[base.subject]
        return {'plan':plan,'plan_seal':semantic_digest(plan)}
    monkeypatch.setattr(preflight,'_collect',collect);monkeypatch.setattr(preflight,'_seal_and_replay',replay)
    monkeypatch.setattr(preflight,'_history_identity',lambda value:worker.canonical(value))
    loads=[]
    def load(*a,**k):loads.append(1);return f.policy,f.metadata
    monkeypatch.setattr(io_module,'load_cohort_checkpoint',load)
    originals=(torch.optim.Optimizer.__init__,torch.autograd.grad,torch.autograd.backward,
        torch.Tensor.backward,torch.save,torch.load,native_module.NativeSpatialTask._transition)
    release={'learning_protocol':protocol,'cohort_limits':limits}
    try:
        if failure is None:
            result=worker.execute(release,'1'*64,lambda *a,**k:None)
            assert result['status']=='complete_frozen_gradient_diagnostic'
            assert result['completed_source_visits']==4 and result['collector_replay_steps']==58
            assert result['policy_forwards']==29 and result['autograd_grad_calls']==79
            assert len(loads)==1 and result['checkpoint_loads']==1
            assert all(result[k] for k in ('parameters_unchanged','gradient_buffers_unchanged','requires_grad_flags_restored'))
            assert json.loads((worker.OUTPUT/'gradient-alignment.json').read_text())['counts']['autograd_grad_calls']==79
        else:
            with pytest.raises((ValueError,RuntimeError),match='generated|readout'):
                worker.execute(release,'1'*64,lambda *a,**k:None)
            result=json.loads((worker.OUTPUT/'result.json').read_text())
            assert result['status']=='failed_or_unresolved'
            assert result['autograd_grad_calls']==(0 if failure=='second_collection' else 1)
            assert result['collector_replay_steps']==(2 if failure=='second_collection' else 58)
            assert not (worker.OUTPUT/'gradient-alignment.json').exists()
        assert originals==(torch.optim.Optimizer.__init__,torch.autograd.grad,torch.autograd.backward,
            torch.Tensor.backward,torch.save,torch.load,native_module.NativeSpatialTask._transition)
        assert all(p.grad is None and not p.requires_grad for p in f.policy.parameters())
        assert parameter_hash(f.policy)==f.metadata['parameter_hash']
    finally:torch.set_num_threads(before_threads)

def test_disabled_release_refuses_before_metadata_and_child(tmp_path,monkeypatch):
    path=tmp_path/'false.json';path.write_text('{"execution_released": false}')
    monkeypatch.setattr(contract,'metadata',lambda:(_ for _ in ()).throw(AssertionError('must not enter metadata')))
    with pytest.raises(ValueError,match='root_release_required'):
        contract.guard(path,contract.sha(path),execute=True)
