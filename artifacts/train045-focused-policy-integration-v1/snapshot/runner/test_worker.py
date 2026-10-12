"""Actual runner failure stack plus scalar gates; only generated fixtures."""
import copy,importlib.util,json,os,sys,time
from pathlib import Path
import pytest
HERE=Path(__file__).resolve().parent
ROOT=next(p for p in HERE.parents if (p/'src/resectionlab').is_dir())
sys.path[:0]=[str(HERE),str(HERE.parent/'model'),str(ROOT/'src')]
spec=importlib.util.spec_from_file_location('focused_policy_worker',HERE/'worker.py')
worker=importlib.util.module_from_spec(spec);spec.loader.exec_module(worker)
from comparison import matched_gate,validate_plan
from common import semantic
from resectionlab.native_resection import NativeResectionEngine
from resectionlab.native_spatial_task import NativeSpatialTask,make_native_opening_task
import resectionlab.patient_planning_cohort_visits as visits


@pytest.mark.parametrize('corrupt_wrapper',[False,True])
def test_actual_execute_counts_native_work_and_restores_after_failure(monkeypatch,corrupt_wrapper):
    output=Path(os.environ['TMPDIR']).resolve()/('focused-stack-'+('tamper' if corrupt_wrapper else 'failure'))
    output.mkdir(exist_ok=False)
    original_preview=NativeResectionEngine.preview_stroke;original_transition=NativeSpatialTask._transition
    entered=[];calls=[]
    def observed_preview(engine,*args,**kwargs):
        calls.append(1);return original_preview(engine,*args,**kwargs)
    monkeypatch.setattr(NativeResectionEngine,'preview_stroke',observed_preview)
    config=json.loads((HERE/'config.json').read_text());config['caps']['work_seconds']=20
    metadata={worker.REPORT:{'pair_plus_recovery_costs':{'generated':True}},worker.PLANS:{},
        worker.IL+'/configuration.json':{'learning_protocol':{},'admission_limits':{}},
        worker.PRIOR_REPORT:{},worker.PRIOR_PARENT:{},worker.TAIL_REPORT:{}}
    index={'input_metadata':{name:'generated' for name in metadata},'files':{}}
    monkeypatch.setattr(worker,'read_bytes',lambda *a,**k:b'generated-only')
    monkeypatch.setattr(worker,'read_json',lambda p,*a:metadata[str(Path(p).relative_to(ROOT))])
    monkeypatch.setattr(worker,'retained_prefix',lambda *a:({}, {'plan_seal':'generated'}))
    monkeypatch.setattr(worker,'require_reference',lambda *a:{'generated_fixture':True})
    monkeypatch.setattr(worker,'require_tail_metadata',lambda *a:({},{}))
    monkeypatch.setattr(worker,'guard',lambda *a:None)
    def generated_factory(**kwargs):
        entered.append(True)
        task=make_native_opening_task(max_steps=3)
        action=next(iter(task._prepare_inventory()));task.step(action)
        assert task._steps==1 and calls
        if corrupt_wrapper:
            NativeResectionEngine.preview_stroke=lambda *a,**k:None
            kwargs['progress']('deliberate_guard_violation')
        raise RuntimeError('generated refusal after paid native transition')
    monkeypatch.setattr(visits,'make_train_visit_factories',generated_factory)
    report=worker.execute(index,'generated-source',config,output,time.perf_counter())
    assert entered==[True], 'Stack must reach actual source factory before generated refusal'
    assert report['status']=='failed_or_incomplete'
    assert report['native_transition_counts']=={'attempted':1,'completed':1,'raised':0}
    assert report['checkpoint_loads']==report['total_frozen_policy_forwards']==0
    budget=report['native_budget']
    assert budget['observed_admissions_before_accounting_failure']==len(calls)>0
    assert sum(p['started'] for p in budget['native_preview_profile']['phases'].values())==len(calls)
    if corrupt_wrapper:
        assert report['failure']['type']=='PlanningBudgetViolation'
        assert report['failure']['message']=='preview_wrapper_replaced'
        assert budget['counting_reliable'] is False and budget['native_preview_entries'] is None
    else:
        assert report['failure']['message']=='generated refusal after paid native transition'
        assert budget['counting_reliable'] is True and budget['native_preview_entries']==len(calls)
    assert NativeResectionEngine.preview_stroke is observed_preview
    assert NativeSpatialTask._transition is original_transition
    assert (output/'report.json').is_file() and (output/'first-failure-sidecars.json.gz').is_file()


def branches():
    outcome={'positive_target_source_cells_removed':5,'target_removed_mm3':5.,'total_reward':4.,
        'normal_removed_mm3':0.,'complete_tool_path_length_mm':10.,'tool_changes':1,'nonstop_actions':2,
        'cumulative_contacted_tissue_upper_bound_mm3':8.,'currently_retained_contacted_tissue_upper_bound_mm3':3.}
    row={'complete':True,'planned_physical_history_matches':True,
        'audit':{'accepted':True,'complete_episode':True,'geometry':{'feasible':True},'outcomes':outcome}}
    return {'GREEDY':copy.deepcopy(row),'TAIL':copy.deepcopy(row)}


def test_matched_gate_quality_not_old_capability_reference():
    rows=branches();assert matched_gate(rows)['comparison_admissible'] and not matched_gate(rows)['useful_gain']
    b=rows['TAIL']['audit']['outcomes'];b['target_removed_mm3']+=1;b['positive_target_source_cells_removed']+=1;b['total_reward']+=.5
    assert matched_gate(rows)['useful_gain']
    b['normal_removed_mm3']=.1;assert not matched_gate(rows)['useful_gain']
    b['normal_removed_mm3']=0;b['total_reward']=3.;assert not matched_gate(rows)['useful_gain']
    b['total_reward']=float('nan');assert not matched_gate(rows)['comparison_admissible']
    rows=branches();rows['TAIL']['complete']=False;assert not matched_gate(rows)['comparison_admissible']
    rows=branches();rows['TAIL']['audit']['geometry']['feasible']=False
    assert not matched_gate(rows)['comparison_admissible']
    assert not matched_gate({'ORIGINAL_STOP':{},'FOCUSED_PREPARATION':{}})['comparison_admissible']


def test_sealed_complete_positive_suffix_gate_refuses_censored_or_negative():
    packets=[]
    for i in range(3):
        p={'steps_taken':i,'action_id':'STOP' if i==2 else 'action'+str(i),
            'source_hash':'source','decision_model_hash':'model'}
        p['seal']=semantic(p);packets.append(p)
    prefix=packets[:1];frontier={'steps':1,'source_hash':'source','decision_model_hash':'model'}
    account={'complete':True,'model_transition_calls':2,'explored_action_records':packets,
        'explored_history':[{'action_id':p['action_id'],'reward':0. if i==2 else 1.} for i,p in enumerate(packets)]}
    assert validate_plan(packets,account,prefix,frontier,24)
    for changes in ({'complete':False},{'time_cap_reached':True},{'model_transition_calls':1}):
        with pytest.raises(ValueError):validate_plan(packets,{**account,**changes},prefix,frontier,24)
    bad=copy.deepcopy(account);bad['explored_history'][1]['reward']=-1
    with pytest.raises(ValueError):validate_plan(packets,bad,prefix,frontier,24)
    bad=copy.deepcopy(packets);bad[1]['steps_taken']=0
    with pytest.raises(ValueError):validate_plan(bad,account,prefix,frontier,24)
