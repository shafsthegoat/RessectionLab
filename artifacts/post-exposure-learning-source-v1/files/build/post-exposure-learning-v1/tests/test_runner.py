"""Generated endpoint glue controls. Numerics are stubbed; no patient/weight reads.

The actual worker execute() traverses all four teachers, shared updates, cache,
checkpoint reload, readouts and final routes. Separate production controls own
real native/gradient correctness; these tests catch declaration/counter wiring.
"""
from pathlib import Path
import copy,hashlib,importlib.util,json,sys,types
from contextlib import contextmanager
import numpy as np
import pytest
import torch
ROOT=next(p for p in Path(__file__).resolve().parents if (p/'src/resectionlab').is_dir())
HERE=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(HERE))
import pilot_contract as contract
import cohort_worker as worker
import run_owned as parent


def test_declared_cost_maxima():
    assert contract.TEACHER_STATES==sum(contract.TEACHER_STEPS.values())==29
    assert contract.METHODS['IL']['loss_forward_cap']==64*29==1856
    assert contract.METHODS['IL']['policy_forward_cap']==64*29+29+4*24==1981
    assert contract.METHODS['RL']['policy_forward_cap']==2*(8*4*24)+29+4*24==1661
    for method,visits in [('IL',8),('RL',44)]:
        assert contract.METHODS[method]['source_visits']==visits
        assert contract.METHODS[method]['native_preview_cap']==visits*120*(1+3*24)
    assert contract.CACHE_BYTES==256*1024**2


def release(method,protocol,limits):
    return {'version':'matched-post-exposure-learning-v1','status':'released_one_attempt','method':method,
        'expected_head':'a'*40,'output':contract.output_for(method),'TRAIN':list(contract.TRAIN),'closed_roles':contract.CLOSED,
        'SELECT_EVAL_execution':False,'new_four_training':True,'attempts':1,'automatic_retry':False,
        'parent_seconds':contract.METHODS[method]['parent_seconds'],'supervision_bytes':contract.SUPERVISION_BYTES,
        'cohort_limits':limits,'limits':{k:v for k,v in limits.items() if k not in ('output_bytes','checkpoint_bytes')},
        'execution_limits':contract.METHODS[method],'learning_protocol':protocol,'learning_protocol_hash':contract.semantic(protocol),
        'inputs':contract.INPUTS,'source_index':{'path':'generated-index.json','sha256':'b'*64},
        'public_manifest_index':{'path':contract.PUBLIC_INDEX,'sha256':contract.PUBLIC_SHA},'completion':'generated','claim':'generated'}


def test_exact_declaration_keys_refuse_omission_and_extras(monkeypatch):
    protocol={'generated':True};limits={'worker_seconds':900}
    monkeypatch.setattr(contract,'expected_configuration',lambda method:(protocol,limits))
    valid=release('IL',protocol,limits);contract.validate_release(valid)
    for key in ('method','public_manifest_index','execution_limits','learning_protocol','cohort_limits','limits','parent_seconds'):
        bad=copy.deepcopy(valid);del bad[key]
        with pytest.raises(ValueError):contract.validate_release(bad)
    with pytest.raises(ValueError):contract.validate_release({**valid,'max_native_previews':1})
    with pytest.raises(ValueError):contract.validate_release({**valid,'status':'pending_root_release'})


def test_parent_real_declaration_before_worker_handles_all_keys(tmp_path,monkeypatch):
    value=release('IL',{'generated':True},{'worker_seconds':900});value['output']='generated-output'
    path=tmp_path/'release.json';path.write_text(json.dumps(value))
    monkeypatch.setattr(parent,'ROOT',tmp_path)
    monkeypatch.setattr(parent,'source_guard',lambda *a:{'source_files':{}})
    helper=types.ModuleType('run_contact_owned')
    helper._receipt=lambda p,v:p.write_text(json.dumps(v))
    helper.cleanup_owned=lambda *a:pytest.fail('No process was launched')
    def reject():raise RuntimeError('generated sampler refusal before process')
    helper.FastDarwinSampler=reject;monkeypatch.setitem(sys.modules,'run_contact_owned',helper)
    monkeypatch.setattr(sys,'argv',['run_owned.py','--release',str(path),'--release-sha256','a'*64])
    with pytest.raises(RuntimeError,match='generated sampler refusal'):parent._main()
    record=json.loads((tmp_path/'generated-output.supervision/declaration.json').read_text())
    assert record['method']=='IL' and record['caps']['hard_total_wall_seconds']==960
    assert record['execution_limits']['policy_forward_cap']==1981
    assert record['TRAIN']==list(contract.TRAIN)
    final=json.loads((tmp_path/'generated-output.supervision/receipt.json').read_text())
    assert final['status']=='failed_or_unresolved' and final['exit_code'] is None


@pytest.mark.parametrize('field',['source_hash','decision_model_hash','initial_observation_hash','actions','history','max_steps'])
def test_teacher_identity_refuses_changed_bound_evidence(field):
    prior={'source_hash':'s','decision_model_hash':'m','initial_observation_hash':'o','max_steps':24,'actions':['STOP'],'history':[{'action_id':'STOP','reward':0}]}
    actual={**copy.deepcopy(prior),'context_hash':'new-admission'}
    worker.require_teacher_identity(actual,prior,contract.canonical)
    actual[field]=None
    with pytest.raises(ValueError):worker.require_teacher_identity(actual,prior,contract.canonical)


@pytest.mark.parametrize('method',['IL','RL'])
def test_execute_full_four_teacher_endpoint_flow_generated(tmp_path,monkeypatch,method):
    from resectionlab import patient_planning_cohort_visits as visits_module
    from resectionlab import patient_planning_cohort_sequential as seq
    from resectionlab import patient_planning_cohort_io as io
    from resectionlab import patient_planning_accumulation as acc
    from resectionlab import patient_teacher_trace_cache as cachemod
    from resectionlab import patient_planning_learning as learning
    from resectionlab import patient_planning_preflight as preflight
    from resectionlab import planning_budget as budgetmod
    from resectionlab import spatial_policy as policy_module
    from resectionlab.core import freeze_json,semantic_digest
    protocol={'seed':5,'max_gradient_norm':1.,'cohort_execution':{'teacher_observations':contract.STORAGE}}
    limits={'output_bytes':16*1024**2,'checkpoint_bytes':8*1024**2}
    declaration={'method':method,'learning_protocol':protocol,'cohort_limits':limits}
    monkeypatch.setattr(worker,'canonical_configuration',lambda m:(protocol,limits))
    # All paths in this run are generated, including the cohort metadata read.
    monkeypatch.setattr(worker,'ROOT',tmp_path);(tmp_path/'cohort.json').write_text('{}');monkeypatch.setattr(worker,'COHORT','cohort.json')
    state={'costs':None,'cache_traces':{},'saved_policy':None,'models_created':0}
    class Policy:
        def __init__(self):self.version=0;state['models_created']+=1
        def __call__(self,observation):
            state['costs'].forwards+=1
            return torch.tensor([0.,1.]),torch.tensor(0.)
        def eval(self):return self
        def requires_grad_(self,v):return self
    monkeypatch.setattr(policy_module,'parameter_hash',lambda p:'sha256:'+str(p.version).zfill(64))
    class Context:
        def __init__(self,s):self.subject=s;self.fingerprint='context-'+s;self.patient_group='ReMIND:'+s[-3:]
        def require_observations(self,rows):assert list(rows)
    class Observation:
        action_ids=('STOP','MOVE');action_mask=np.array([True,True])
        def __init__(self,s,i):self.fingerprint=s+'-'+str(i)
    class Trace:
        def __init__(self,s,actions,context):
            self.context=context;self.transitions=tuple(types.SimpleNamespace(observation=Observation(s,i),action_id=a,reward=float(a!='STOP'),terminated=i==len(actions)-1) for i,a in enumerate(actions))
            self.seal_hash='trace-'+s;self.history=tuple({'action_id':a,'reward':float(a!='STOP')} for a in actions)
    contexts={s:Context(s) for s in contract.TRAIN}
    def plan(s,actions):return {'source_hash':'source-'+s,'decision_model_hash':'model-'+s,'initial_observation_hash':s+'-0','max_steps':24,
        'actions':actions,'history':[{'action_id':a,'reward':float(a!='STOP')} for a in actions],
        'terminal_reason':'STOP','parameter_hash':None,'architecture_hash':None,'learning_updates':0}
    teacherplans={s:plan(s,['MOVE']*(contract.TEACHER_STEPS[s]-1)+['STOP']) for s in contract.TRAIN}
    monkeypatch.setattr(worker,'inputs',lambda:{'plan_'+s:{'plan':p} for s,p in teacherplans.items()})
    monkeypatch.setattr(visits_module,'make_train_visit_factories',lambda **kw:{s:object() for s in contract.TRAIN})
    class Visits:
        def __init__(self,*a):self.contexts=contexts;self.completed=[]
        def run(self,s,dest,phase,callback):
            row=callback(types.SimpleNamespace(subject=s),contexts[s]);self.completed.append(s);return row
    monkeypatch.setattr(seq,'_SequentialVisits',Visits)
    @contextmanager
    def noop(*a):yield
    monkeypatch.setattr(seq,'_bounded_writes',noop)
    class Budget:
        def __init__(self,*a,**kw):self.status='running'
        def __enter__(self):return self
        def __exit__(self,*a):return False
        def check(self):pass
        def complete(self,**kw):assert kw=={'history_complete':True};self.status='complete'
        def snapshot(self):return {'native_preview_entries':0,'status':self.status}
    monkeypatch.setattr(budgetmod,'PlanningBudget',Budget)
    class Costs:
        def __init__(self,*a):self.forwards=0;self.rows=[];state['costs']=self
        def __enter__(self):return self
        def __exit__(self,*a):return False
        def scope(self,name):self.rows.append({'phase':name});return noop()
    monkeypatch.setattr(preflight,'_CallCosts',Costs)
    def collect(base,context,*,policy=None,generator=None,actions=None,output,guard):
        guard();actions=['STOP'] if actions is None else actions
        trace=Trace(base.subject,actions,context)
        if policy is not None:
            for row in trace.transitions:policy(row.observation)
        return trace
    def sealed(base,context,trace,*,method,policy,updates,output,guard):
        value=plan(base.subject,[r.action_id for r in trace.transitions]);value['context_hash']=context.fingerprint
        return {'plan':freeze_json(value),'plan_seal':semantic_digest(value),'replayed_history':trace.history}
    monkeypatch.setattr(preflight,'_collect',collect);monkeypatch.setattr(preflight,'_seal_and_replay',sealed)
    monkeypatch.setattr(preflight,'_history_identity',lambda h:contract.canonical(h))
    monkeypatch.setattr(learning,'common_patient_policies',lambda *a:(Policy(),Policy()))
    class Session:
        def __init__(self,contexts,method,policy,**kw):self.policy=policy;self.updates=0
    monkeypatch.setattr(learning,'PatientTrainSession',Session)
    class Cache:
        def __init__(self,p):self.traces={}
        def add_replayed(self,t,s):self.traces[t.context.subject]=t;state['cache_traces']=self.traces
        def trace_for_il(self,s,subject):return self.traces[subject]
        def record(self):return {'complete':len(self.traces)==4,'steps':sum(len(t.transitions) for t in self.traces.values())}
    monkeypatch.setattr(cachemod,'PatientTeacherTraceCache',Cache)
    class Accumulator:
        def __init__(self,session,**kw):self.session=session;self.calls=0
        def __enter__(self):return self
        def __exit__(self,*a):return False
        def add_trace(self,t,guard):
            for row in t.transitions:guard();self.session.policy(row.observation);self.calls+=1
            return {'loss_forward_calls':len(t.transitions)}
        def finish(self,guard):
            guard();before=self.session.policy.version;self.session.updates+=1;self.session.policy.version+=1
            return {'completed_updates':self.session.updates,'loss':1.,'gradient_norm_before_clip':.5,'module_gradient_norms_before_clip':{},
                'before_parameter_hash':str(before),'after_parameter_hash':str(before+1),'loss_forward_calls':self.calls}
    monkeypatch.setattr(acc,'PatientGradientAccumulator',Accumulator)
    def save(policy,*,method,contexts,protocol,updates,initial_hash,output,limits):
        state['saved_policy']=copy.copy(policy);raw=b'generated checkpoint';name=method+'-final.psckpt';output.write_bytes(output.root/name,raw)
        return {'path':name,'sha256':hashlib.sha256(raw).hexdigest(),'bytes':len(raw),'parameter_hash':policy_module.parameter_hash(policy)}
    def load(path,**kw):return copy.copy(state['saved_policy']),{'completed_updates':contract.METHODS[method]['updates']}
    monkeypatch.setattr(io,'_save_checkpoint',save);monkeypatch.setattr(io,'load_cohort_checkpoint',load)
    result=worker.execute(declaration,'d'*64,tmp_path/'run',lambda *a,**kw:None)
    assert contract.complete_result(result)
    assert result['teacher_decisions']==29 and result['teacher_steps']==contract.TEACHER_STEPS
    assert result['loss_forward_calls']==(1856 if method=='IL' else 32)
    assert result['total_policy_forward_calls']==(1889 if method=='IL' else 97)
    assert result['completed_source_visits']==(8 if method=='IL' else 44)
    assert result['optimizer_updates']=={'IL':64 if method=='IL' else 0,'RL':8 if method=='RL' else 0}
    assert result['teacher_trace_reuses']==(256 if method=='IL' else 0)
    assert state['models_created']==2 and result['fresh_common_initialization_verified']
    assert len(json.loads((tmp_path/'run/training-dynamics.json').read_text())['updates'])==contract.METHODS[method]['updates']


@pytest.mark.parametrize('method',['IL','RL'])
def test_metadata_expected_configuration_matches_actual_constructor(method):
    # Saved source/results are metadata only; no acquired arrays or weights.
    expected=contract.expected_configuration(method)
    actual=contract.canonical_configuration(method)
    assert contract.canonical(actual)==contract.canonical(expected)
    assert actual[0]['cohort_execution']['patient_order']==list(contract.TRAIN)
    assert actual[0]['cohort_execution']['post_exposure_condition']==contract.EXPOSURE
