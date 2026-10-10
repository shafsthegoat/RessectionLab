"""Tiny metadata and fake-process controls; no original or saved patient arrays."""
from pathlib import Path
import ast,copy,importlib.util,sys
import pytest
HERE=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(HERE))
import batch_contract as contract
import run_owned as parent

def result():
    return {'status':'post_exposure_batch_complete','release_sha256':'a'*64,
       'occupancy_condition':contract.CONDITION,'post_exposure_condition':contract.EXPOSURE,'cases':[{'patient_id':s,'role':'TRAIN','status':'completed_fixed_post_exposure_route',
       'initial_steps_taken':0,'steps_taken':1,'route_complete':True,'independent_episode_accepted':True,
       'actions':['STOP'],'greedy_accounting':{'complete':True,'model_transition_calls':1,'decisions':[{}]},'model_or_optimizer_calls':0,'accepted_count':0,'emitted_count':120,'source_domain_extended':False,'full_target_preserved':True}
       for s in contract.SUBJECTS],'all_four_cases_retained':True,'failed_cases_replaced':False,
       **dict.fromkeys(contract.ZERO_COUNTERS,0),'source_arrays_written':0,'training_admitted':False,
       'greedy_calls':4,'planning_clone_calls':4,'transition_calls':4,'planning_transition_calls':4,
       'proposal_config':{},'max_steps':24,'native_budget':{'status':'complete_history_awaiting_independent_audit'}}

def test_zero_legal_emitted_motions_is_completed_fixed_access_result():
    assert contract.complete_result(result(),{'proposal_config':{}},'a'*64)

def test_failed_source_row_stays_retained_but_cannot_attest_completed_histories():
    r=result();r['status']+='_with_case_failures';r['cases'][1]={'patient_id':contract.SUBJECTS[1],'status':'case_failed_retained'}
    with pytest.raises(ValueError):contract.complete_result(r,{'proposal_config':{}},'a'*64)

@pytest.mark.parametrize('kind',['denominator','models','target','domain','steps','pending','budget','proposals','cap','greedy_count','transitions','plan_complete','audit','sequence'])
def test_terminal_guard_refuses_incomplete_or_changed_execution(kind):
    r=result()
    if kind=='denominator':r['cases'].pop()
    elif kind=='models':r['model_calls']=1
    elif kind=='target':r['cases'][0]['full_target_preserved']=False
    elif kind=='domain':r['cases'][0]['source_domain_extended']=True
    elif kind=='steps':r['cases'][0]['steps_taken']=0
    elif kind=='pending':r['cases'][0]['status']='not_attempted'
    elif kind=='budget':r['native_budget']['status']='failed'
    elif kind=='proposals':r['proposal_config']={'changed':True}
    elif kind=='cap':r['cases'][0]['emitted_count']=121
    elif kind=='greedy_count':r['greedy_calls']=3
    elif kind=='transitions':r['planning_transition_calls']=3
    elif kind=='plan_complete':r['cases'][0]['greedy_accounting']['complete']=False
    elif kind=='audit':r['cases'][0]['independent_episode_accepted']=False
    else:r['cases'][0]['actions'].append('STOP')
    with pytest.raises(ValueError):contract.complete_result(r,{'proposal_config':{}},'a'*64)

def test_cleanup_matches_previously_reviewed_retained_popen_fallback():
    a=ast.parse((HERE/'run_owned.py').read_text())
    b=ast.parse((HERE.parent/'remind-train-domain-crop-preparation-v1/run_public_qc_owned.py').read_text())
    select=lambda tree:ast.dump(next(x for x in tree.body if isinstance(x,ast.FunctionDef) and x.name=='cleanup_phase'),include_attributes=False)
    assert select(a)==select(b)

def test_sampler_cleanup_failure_still_reaps_retained_process(monkeypatch):
    class Process:
        pid=424242;code=None
        def poll(self):return self.code
        def kill(self):self.code=-9
        def wait(self,timeout):self.code=-9
    class Sampler:
        def group(self,*a):return {'pids':[]}
        def signal_if_same_process(self,*a):return {'identity_checked':True}
    process=Process();calls=[]
    monkeypatch.setattr(parent.os,'killpg',lambda *a:calls.append(a))
    def broken(*a):raise PermissionError('generated sampler error')
    errors=[];actions=[]
    assert parent.cleanup_phase(process,Sampler(),{},actions,errors,broken)==[]
    assert calls==[(424242,parent.signal.SIGKILL)] and process.poll()==-9 and errors

def test_fast_exit_log_overflow_is_refused(tmp_path,monkeypatch):
    monkeypatch.setitem(parent.CAPS,'log_bytes',10)
    output=tmp_path/'output';supervision=tmp_path/'supervision';output.mkdir();supervision.mkdir()
    (supervision/'worker.log').write_bytes(b'x'*11)
    with pytest.raises(ValueError,match='final_log_cap'):parent.final_extents(output,supervision)

def test_output_symlink_refused(tmp_path):
    (tmp_path/'link').symlink_to('/tmp')
    with pytest.raises(ValueError,match='symlink'):parent.tree_bytes(tmp_path)

def test_worker_compiles_and_preserves_preconstruction_budget():
    source=(HERE/'initial_inventory_worker.py').read_text();compile(source,'initial_inventory_worker.py','exec')
    tree=ast.parse(source)
    guarded=next(n for n in ast.walk(tree) if isinstance(n,ast.With) and any(isinstance(i.context_expr,ast.Name) and i.context_expr.id=='budget' for i in n.items))
    assert any(isinstance(n,ast.Call) and isinstance(n.func,ast.Name) and n.func.id=='construct' for n in ast.walk(guarded))
    assert "owned_refs.append(weakref.ref(source))" in source and "owned_refs.append(weakref.ref(task))" in source
    assert "'max_candidates']!=120" in source and 'DEFAULT_COLUMN_OFFSETS' in source


def test_generated_greedy_complete_route_uses_p0_and_independent_replay():
    root=next(p for p in HERE.parents if (p/'src/resectionlab').is_dir())
    sys.path.insert(0,str(root/'src'))
    fixture_path=root/'tests/test_post_exposure.py'
    spec=importlib.util.spec_from_file_location('generated_post_fixture',fixture_path)
    fixture=importlib.util.module_from_spec(spec);spec.loader.exec_module(fixture)
    from resectionlab.native_spatial_task import NativeSpatialTask
    from resectionlab.native_spatial_evaluation import evaluate_native_spatial_episode
    from initial_inventory_worker import install_native_counters
    counters=dict.fromkeys(('transition_calls','planning_transition_calls','planning_clone_calls','greedy_calls'),0)
    originals=[];install_native_counters(NativeSpatialTask,counters,originals)
    try:
        S,T,Ds=fixture.arrays();task=NativeSpatialTask(fixture.source(S,T,Ds),max_steps=3)
        sequence,accounting=task.observed_greedy_search(seconds=5)
        assert accounting['complete'] and sequence[-1]=='STOP' and len(sequence)>1
        assert accounting['model_transition_calls']==len(accounting['decisions'])==len(sequence)
        assert any(x['reward']>0 for x in accounting['decisions'][0]['scores'])
        for action in sequence:task.step(action)
        result=evaluate_native_spatial_episode(task)
        assert task.terminated and result['accepted'] and result['target_access_success']
        assert counters['greedy_calls']==counters['planning_clone_calls']==1
        assert counters['transition_calls']==counters['planning_transition_calls']==len(sequence)
        for row in task.metrics()['history']:
            if row['action_id']!='STOP':
                assert 'physical_start_mm' in row and row['complete_tool_path_length_mm']>0
    finally:
        for owner,name,original in reversed(originals):setattr(owner,name,original)


def test_worker_fails_before_complete_attestation_if_any_row_unresolved():
    source=(HERE/'initial_inventory_worker.py').read_text()
    assert source.index("if not all(r['status']=='completed_fixed_post_exposure_route'")<source.index('budget.complete(history_complete=True)')
    assert '"seconds": GREEDY_SECONDS' in source
    assert 'post_exposure_condition=EXPOSURE' in source
