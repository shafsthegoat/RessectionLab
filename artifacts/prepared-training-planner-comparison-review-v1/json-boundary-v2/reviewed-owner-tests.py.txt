"""Offline integration controls: no patient decode, checkpoint load or forwards."""
from contextlib import contextmanager
from copy import deepcopy
from dataclasses import dataclass
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).parents[1]
spec = importlib.util.spec_from_file_location('prepared_planners', ROOT/'scripts/compare_prepared_training_planners.py')
runner = importlib.util.module_from_spec(spec); spec.loader.exec_module(runner)


class Budget:
    """Strict protocol double; real budget semantics are independently tested."""
    def __init__(self):
        self.active = False; self.complete_flag = False; self.status = 'not_started'
        self.preview_entries = 0; self.events = []
    def __enter__(self):
        self.active = True; self.status = 'active'; self.events.append('enter'); return self
    def check(self):
        assert self.active, 'inactive online guard touched after audit'
        self.events.append('check')
    @contextmanager
    def phase(self, name):
        self.check(); self.events.append(name+'_enter')
        try: yield
        finally: self.check(); self.events.append(name+'_exit')
    def complete(self, *, history_complete):
        self.check(); assert history_complete; self.complete_flag = True; self.events.append('complete')
    def __exit__(self, kind, error, tb):
        self.active = False; self.events.append('exit')
        self.status = 'failed' if error or not self.complete_flag else 'complete_history_awaiting_independent_audit'
    def snapshot(self):
        return {'status': self.status, 'limits': {'native_preview_entries': 468, 'planning_execution_seconds': 90.},
            'counting_reliable': True, 'native_preview_entries': self.preview_entries,
            'failure':None, 'blocked_preview_attempts':0, 'time_overshoot_seconds':0., 'score':None,
            'elapsed_seconds': 2., 'history_complete_caller_attestation': self.complete_flag}


@pytest.fixture
def setup(tmp_path):
    budget = Budget(); events = []; binding = {'source_hash':'source', 'model_hash':'model',
        'initial_state_hash':'empty', 'engine_state_hash':'nativeempty', 'observation_hash':'obs',
        'action_ids':['STOP','a'], 'max_steps':3,'reward':{'target':1}}
    class Base:
        def observed_greedy_search(self, *, seconds):
            assert budget.active and seconds == 90
            events.append('search'); budget.preview_entries += 100
            return ('a','STOP'), {'evaluated_nonstop_actions':1000, 'complete':True}
    base=Base(); policy=object()
    def whole_guard(): events.append('whole_guard')
    def closure(): events.append('closure')
    def audit(task, *, metrics):
        assert not budget.active
        assert (tmp_path/(current[0]+'-terminal.json')).is_file()
        events.append('audit')
        return {'accepted':True,'outcomes':{'total_reward':0.},'target_access_success':False}
    def episode(base_, policy_, generator, *, mode, name, output, guard, audit, sequence):
        current[0]=name
        assert base_ is base and policy_ is policy and generator is None
        assert budget.active
        events.append(('episode',mode,sequence)); budget.preview_entries+=78
        guard()
        chosen='STOP' if mode=='sequence' and sequence==('STOP',) else 'a'
        metrics={'terminated':True,'decision_model_hash':'model','history':[{'action_id':chosen}]}
        task=SimpleNamespace(terminated=True,metrics=lambda:deepcopy(metrics))
        row={'status':'awaiting_independent_check','metrics':metrics,'decisions':[{'action_id':chosen,
            'decision_seconds':.1,'transition_seconds':.2}]}
        runner.write_json(output/(name+'.json'),row)
        result=audit(task,metrics=metrics)
        guard()  # Real rollout's post-audit preserve still enforces the whole worker.
        if not result['accepted']:raise RuntimeError('audit rejected')
        row.update(status='complete',independent_evaluation=result)
        runner.write_json(output/(name+'.json'),row)
        return (),row
    current=[None]
    kwargs={'whole_guard':whole_guard,'auditor':audit,'closure_check':closure,
        'budget_factory':lambda:budget,'episode_runner':episode,
        'policy_hasher':lambda _:runner.frozen.POLICY_HASH,'binding_reader':lambda _:deepcopy(binding)}
    return SimpleNamespace(out=tmp_path,budget=budget,events=events,base=base,policy=policy,
        binding=binding,kwargs=kwargs,episode=episode,current=current)


@pytest.mark.parametrize('name',runner.METHODS)
def test_shared_initial_state_terminal_export_then_budget_exit_then_audit(setup,name):
    s=setup; row=runner.run_arm(s.base,s.policy,name,s.out,**s.kwargs)
    assert runner.validate_completed_arm(row,s.binding)
    assert row['initial_binding']==row['final_initial_binding']==s.binding
    assert s.budget.events.index('complete')<s.budget.events.index('execution_exit')<s.budget.events.index('exit')
    assert s.events[-1]=='whole_guard'
    assert runner.sha256(s.out/row['terminal_history_path'])==row['terminal_history_sha256']
    if name=='greedy_search':
        assert s.budget.preview_entries==178
        assert row['search_accounting']['evaluated_nonstop_actions']==1000
        assert s.budget.events.index('planning_exit')<s.budget.events.index('execution_enter')
    if name=='STOP':assert ('episode','sequence',('STOP',)) in s.events


@pytest.mark.parametrize('failure',['audit_rejected','incomplete','saved_mismatch','post_audit_worker','closure','changed_weights','changed_initial','export'])
def test_each_failure_is_retained_with_null_outcomes(setup,monkeypatch,failure):
    s=setup
    if failure=='audit_rejected':s.kwargs['auditor']=lambda *a,**k:{'accepted':False}
    elif failure in ('incomplete','saved_mismatch'):
        original=s.kwargs['auditor']
        def bad_episode(*a,**kw):
            audit=kw['audit']
            def altered(task,*,metrics):
                if failure=='incomplete':task.terminated=False
                else:metrics['extra']='unsaved'
                return audit(task,metrics=metrics)
            kw['audit']=altered
            return s.episode(*a,**kw)
        s.kwargs['episode_runner']=bad_episode
    elif failure=='post_audit_worker':
        def worker():
            if not s.budget.active and s.budget.complete_flag:raise TimeoutError('whole worker')
        s.kwargs['whole_guard']=worker
    elif failure=='closure':
        calls=[0]
        def closure():
            calls[0]+=1
            if calls[0]>1:raise ValueError('source changed')
        s.kwargs['closure_check']=closure
    elif failure=='changed_weights':
        calls=[0]
        def hasher(_):calls[0]+=1;return runner.frozen.POLICY_HASH if calls[0]==1 else 'changed'
        s.kwargs['policy_hasher']=hasher
    elif failure=='changed_initial':
        calls=[0]
        def bind(_):calls[0]+=1;return s.binding if calls[0]==1 else {**s.binding,'initial_state_hash':'changed'}
        s.kwargs['binding_reader']=bind
    else:
        original=runner.write_json; failed=[False]
        def write(path,value):
            if value.get('status')=='complete' and path.name.endswith('-arm.json') and not failed[0]:
                failed[0]=True;raise OSError('export failed')
            original(path,value)
        monkeypatch.setattr(runner,'write_json',write)
    row=runner.run_arm(s.base,s.policy,'frozen_il',s.out,**s.kwargs)
    assert row['status']=='failed' and row['outcomes'] is None
    assert not runner.validate_completed_arm(row,s.binding)
    assert json.loads((s.out/'frozen_il-arm.json').read_text())['outcomes'] is None


def test_planning_cap_keeps_unreplayed_prefix_unassessed(setup):
    s=setup
    def fail(**kw):
        error=InterruptedError('preview limit');error.accounting={'complete':False};error.best_sequence=('a',)
        raise error
    s.base.observed_greedy_search=fail
    row=runner.run_arm(s.base,s.policy,'greedy_search',s.out,**s.kwargs)
    assert row['status']=='failed' and row['outcomes'] is None
    assert row['unreplayed_partial_sequence']==('a',)
    assert not any(isinstance(x,tuple) and x[0]=='episode' for x in s.events)


def test_unknown_method_refuses_before_budget_or_episode(setup):
    with pytest.raises(ValueError):runner.run_arm(setup.base,setup.policy,'random',setup.out,**setup.kwargs)
    assert not setup.budget.events


def test_no_patient_loader_for_historical_blocks(tmp_path):
    for subject in (*runner.BLOCKED,'sub-PAT26','sub-PAT29'):
        with pytest.raises(ValueError):runner.prepare_selected(subject,{},tmp_path,lambda:None)


def test_six_patient_denominator_preserves_two_historical_blocks_and_missing_attempts(tmp_path):
    for subject,count in [('sub-PAT16',19),('sub-PAT20',125)]:
        directory=tmp_path/subject;directory.mkdir()
        runner.write_json(directory/'receipt.json',{**runner.blank(subject),'status':'historical_support_block',
            'original_preparation':{'coverage':{'target_outside_support_source_cells':count}}})
    report=runner.summarize(tmp_path)
    assert report['patients_prescribed']==6 and report['complete_comparisons']==0
    assert [x['subject'] for x in report['patients']]==list(runner.SUBJECTS)
    assert [x['status'] for x in report['patients']].count('historical_support_block')==2
    assert all(a['outcomes'] is None for x in report['patients'] for a in x['arms'].values())


@pytest.mark.parametrize('mutation',['false_audit','too_many_previews','elapsed','unfinished','wrong_model','unreliable','weights'])
def test_completed_receipt_requires_all_budget_policy_and_model_gates(setup,mutation):
    s=setup;row=runner.run_arm(s.base,s.policy,'STOP',s.out,**s.kwargs)
    if mutation=='false_audit':row['independent_evaluation_accepted']=False
    elif mutation=='too_many_previews':row['planning_budget']['native_preview_entries']=469
    elif mutation=='elapsed':row['planning_budget']['elapsed_seconds']=90.001
    elif mutation=='unfinished':row['planning_budget']['history_complete_caller_attestation']=False
    elif mutation=='wrong_model':row['final_initial_binding']['model_hash']='changed'
    elif mutation=='unreliable':row['planning_budget']['counting_reliable']=False
    else:row['final_parameter_hash']='changed'
    assert not runner.validate_completed_arm(row,s.binding)


@pytest.fixture
def metadata(tmp_path,monkeypatch):
    import shutil
    cohort=tmp_path/runner.frozen.COHORT_PATH;cohort.parent.mkdir(parents=True)
    shutil.copy2(ROOT/runner.frozen.COHORT_PATH,cohort)
    anchor=tmp_path/'metadata-anchor';anchor.mkdir();checkpoint=anchor/'augmented_latest.pt'
    checkpoint.write_bytes(b'offline byte-binding double, never deserialized')
    runner.write_json(anchor/'output-sha256.json',{'augmented_latest.pt':runner.sha256(checkpoint)})
    monkeypatch.setattr(runner,'ROOT',tmp_path)
    monkeypatch.setattr(runner.frozen,'CHECKPOINT',checkpoint);monkeypatch.setattr(runner.frozen,'ANCHOR',anchor)
    monkeypatch.setattr(runner,'source_inventory',lambda paths=None:{'src/resectionlab/planning_budget.py':'bound'})
    monkeypatch.setattr(runner,'input_closure',lambda:{'fixed_historical_authority':True})
    return runner.declaration()


@pytest.mark.parametrize('change',['omitted_patient','select','unopened','methods','horizon','budget','checkpoint','cohort','inputs'])
def test_declaration_refuses_role_task_budget_or_model_change_before_loading(metadata,change):
    assert runner.validate(metadata)
    altered=deepcopy(metadata)
    if change=='omitted_patient':altered['subjects'].pop()
    elif change=='select':altered['subjects'][0]='sub-PAT26'
    elif change=='unopened':altered['attempted'][0]='sub-PAT29'
    elif change=='methods':altered['methods']=['frozen_il','greedy_search']
    elif change=='horizon':altered['settings']['max_steps']=4
    elif change=='budget':altered['settings']['arm_native_preview_entries']=469
    elif change=='checkpoint':altered['checkpoint']['parameter_hash']='different'
    elif change=='cohort':altered['cohort_sha256']='different'
    else:altered['input_closure']={'changed':True}
    with pytest.raises(ValueError):runner.validate(altered)


def test_isolated_frozen_source_validation_never_discovers_enclosing_git(tmp_path,monkeypatch):
    monkeypatch.setattr(runner,'ROOT',tmp_path)
    names={'scripts/'+name for name in runner.SCRIPT_CLOSURE}|{
        'src/resectionlab/'+name for name in ('planning_budget.py','native_access_preparation.py',
        'native_spatial_task.py','native_spatial_evaluation.py','spatial_policy.py')}
    for name in names:
        path=tmp_path/name;path.parent.mkdir(parents=True,exist_ok=True);path.write_text('# frozen source\n')
    monkeypatch.setattr(runner.subprocess,'check_output',lambda *a,**kw:pytest.fail('runtime Git lookup'))
    hashes=runner.source_inventory(names)
    assert set(hashes)==names
    with pytest.raises(ValueError):runner.source_inventory(names-{'src/resectionlab/planning_budget.py'})
    with pytest.raises(ValueError):runner.source_inventory(names|{'../outside.py'})


def test_unbound_actual_local_import_refused(tmp_path,monkeypatch):
    monkeypatch.setattr(runner,'ROOT',tmp_path)
    path=tmp_path/'scripts/unexpected.py';path.parent.mkdir();path.write_text('# unexpected import\n')
    monkeypatch.setattr(runner,'sys',SimpleNamespace(modules={'prepared_comparison_unexpected':SimpleNamespace(__file__=str(path))}))
    with pytest.raises(ValueError,match='absent or changed'):
        runner.assert_imported_source_closure({'source_sha256':{}})
    runner.assert_imported_source_closure({'source_sha256':{'scripts/unexpected.py':runner.sha256(path)}})


def test_source_parity_allows_only_access_change():
    import numpy as np
    @dataclass
    class Source:
        access: str
        structural_intensity: object
        tools: tuple=()
        track: str='annotation_assisted'
    first=Source('original',np.array([1.,2.]))
    second=Source('selected',np.array([1.,2.]))
    assert runner._source_except_access(first)==runner._source_except_access(second)
    second.structural_intensity[0]=3.
    assert runner._source_except_access(first)!=runner._source_except_access(second)


@pytest.mark.parametrize('name',('STOP','frozen_il'))
def test_real_native_task_real_rollout_roundtrip_reaches_independent_audit(tmp_path,monkeypatch,name):
    """Actual tuple-valued native metrics must survive the durable JSON boundary."""
    import torch
    from resectionlab.native_spatial_task import make_native_opening_task
    from resectionlab.native_resection import NativeResectionEngine
    from resectionlab.native_spatial_evaluation import evaluate_native_spatial_episode
    from resectionlab.spatial_policy import parameter_hash
    torch.set_num_threads(1)
    class FixedLegalPolicy(torch.nn.Module):
        def __init__(self):
            super().__init__();self.anchor=torch.nn.Parameter(torch.tensor(0.),requires_grad=False)
        def forward(self,observation):
            # Test-only frozen choice: first legal non-STOP when present, never an oracle score.
            logits=torch.full((len(observation.action_ids),),-1.)+self.anchor
            logits[0]=0.
            for i,allowed in enumerate(observation.action_mask):
                if i and allowed:logits[i]=1.;break
            return logits.masked_fill(~torch.tensor(observation.action_mask.copy()),-torch.inf),self.anchor
    policy=FixedLegalPolicy();monkeypatch.setattr(runner.frozen,'POLICY_HASH',parameter_hash(policy))
    task=make_native_opening_task(max_steps=3)
    live=task.metrics()
    assert isinstance(live['crop']['shape'],tuple) and isinstance(live['crop']['origin_voxels'],tuple)
    original_preview=NativeResectionEngine.preview_stroke;checks=[]
    def audit(actual,*,metrics):
        assert NativeResectionEngine.preview_stroke is original_preview
        checks.append(True)
        return evaluate_native_spatial_episode(actual,metrics=metrics,distance_backend='batch',distance_batch_size=256)
    row=runner.run_arm(task,policy,name,tmp_path,whole_guard=lambda:None,
        auditor=audit,closure_check=lambda:None)
    assert row['status']=='complete',row.get('failure')
    assert checks==[True] and runner.validate_completed_arm(row,runner.task_binding(task))
    saved=json.loads((tmp_path/(name+'.json')).read_text())
    assert saved['independent_evaluation']['accepted'] is True
    assert isinstance(saved['metrics']['crop']['shape'],list)
    if name=='frozen_il':
        assert any(x['action_id']!='STOP' for x in saved['metrics']['history'])
        assert saved['policy_forward_calls']>0
    else:assert saved['policy_forward_calls']==0 and row['actions']==['STOP']
    assert all(p.grad is None for p in policy.parameters())


def test_real_native_saved_scalar_tamper_still_refused_before_audit(tmp_path,monkeypatch):
    import torch
    from resectionlab.native_spatial_task import make_native_opening_task
    from resectionlab.spatial_policy import parameter_hash
    torch.set_num_threads(1)
    policy=torch.nn.Linear(1,1).requires_grad_(False)
    monkeypatch.setattr(runner.frozen,'POLICY_HASH',parameter_hash(policy))
    task=make_native_opening_task(max_steps=3)
    def intercepted_episode(*args,**kwargs):
        audit=kwargs['audit']
        def corrupt(actual,*,metrics):
            path=tmp_path/'STOP.json';saved=json.loads(path.read_text())
            saved['metrics']['crop']['shape'][0]+=1
            runner.write_json(path,saved)
            return audit(actual,metrics=metrics)
        kwargs['audit']=corrupt
        return runner.rollout.run_episode(*args,**kwargs)
    row=runner.run_arm(task,policy,'STOP',tmp_path,whole_guard=lambda:None,
        closure_check=lambda:None,auditor=lambda *a,**kw:pytest.fail('tampered record reached audit'),
        episode_runner=intercepted_episode)
    assert row['status']=='failed' and row['outcomes'] is None
    assert 'durably retained' in row['failure']['message']
