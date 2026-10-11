"""Generated dependency selection, exact current evidence, and unchanged native replay."""
from dataclasses import replace
import hashlib
import importlib.util
from pathlib import Path
from types import SimpleNamespace
import sys

import numpy as np
import pytest

ROOT=next(p for p in Path(__file__).resolve().parents if (p/'src/resectionlab').is_dir())
HERE=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'));sys.path.insert(0,str(ROOT/'tests'));sys.path.insert(0,str(HERE))
from test_obstruction_opening_proposals import source
from resectionlab.core import semantic_digest
from resectionlab.paged_obstruction import PagedObstructionTask
from resectionlab.native_spatial_evaluation import evaluate_native_spatial_episode
from resectionlab.observed_search import ObservedSearchLimit
from recursive_corridor_selector import (select_corridor,physical_key,goal_record,
    _current_prerequisites,check_exploration_geometry)


def old_selector():
    path=ROOT/'build/select013-complete-corridor-v1/corridor_selector.py'
    assert hashlib.sha256(path.read_bytes()).hexdigest()=='80c83cbce23083519762ce443c8eb908dde43917a44d5e2f0e49263d8893b41e'
    spec=importlib.util.spec_from_file_location('previous_corridor_selector',path)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    return module


def make_task(*,recursive=False,private=False):
    case=source(cap=120)
    support=np.zeros(case.observed_support.shape,bool)
    goal=(9,9,9)
    # C blocks the goal first. After C is removed, A is the goal blocker,
    # but the centered full-shaft approach to A requires side cell B first.
    cells=((2,10,9),(2,11,11),(4,10,10),goal) if recursive else ((4,10,10),goal)
    for c in cells:support[c]=True
    nominal=np.zeros(support.shape,np.float32);nominal[goal]=1
    case=replace(case,observed_support=support,nominal_target=nominal,
        reference_target=np.zeros_like(nominal) if private else nominal,
        proposal_config=replace(case.proposal_config,obstruction_page_size=8,
            obstruction_selection_rule='target_ray_coverage_v3'))
    return PagedObstructionTask(case,max_steps=5)


def test_no_stall_preserves_previous_physical_sequence_cost_and_zero_dependency_queries():
    before,after=make_task(),make_task()
    p,a=old_selector().select_corridor(before,seconds=15)
    q,b=select_corridor(after,seconds=15)
    assert p==q and a['explored_history']==b['explored_history']
    assert a['explored_total_return']==b['explored_total_return']
    assert b['dependency_preview_calls']==0
    assert a['diagnostic_preview_calls']==b['goal_diagnostic_preview_calls']==b['diagnostic_preview_calls']
    assert before._steps==after._steps==0


def test_real_native_depth2_side_cell_then_blocker_then_goal_complete_replay():
    task=make_task(recursive=True)
    actions,report=select_corridor(task,seconds=15)
    assert report['complete'] and report['estimated_incremental_return']>0
    dependencies=[s for s in report['states'] if s.get('dependency_diagnostics',{}).get('selected_effect')]
    assert dependencies, 'Generated geometry must exercise actual depth2 selection, not a no-trigger success'
    dep=dependencies[0];effect=dep['dependency_diagnostics']['selected_effect']
    assert dep['selected']['progress_kind']=='prerequisite'
    assert dep['selected']['current_blockers_removed']==0
    assert effect['sampled_prerequisite_cells_removed']==[[2,11,11]]
    assert [4,10,10] in dep['dependency_diagnostics']['active_retained_blockers']
    assert effect['prep_ray_links']
    for probe in dep['dependency_diagnostics']['probes']:
        if probe['sidecar']:
            assert probe['sidecar']['source_state_hash']==dep['source_state_hash']
            assert probe['sidecar']['requested_tip_mm']==probe['tip_mm']
    next_state=report['states'][dep['depth']]
    assert next_state['selected']['goal']==dep['selected']['goal']
    assert next_state['source_state_hash']==dep['selected']['result_state_hash']
    assert next_state['selected']['current_blockers_removed']>0
    assert [4,10,10] in next_state['selected']['actual_removed_cells']
    assert report['dependency_preview_calls']==sum(len(s.get('dependency_diagnostics',{}).get('probes',[])) for s in report['states'])
    assert report['diagnostic_preview_calls']==report['goal_diagnostic_preview_calls']+report['dependency_preview_calls']
    assert all(len(s.get('dependency_diagnostics',{}).get('probes',[]))<=32 for s in report['states'])
    assert all(s['complete_page_enumeration'] for s in report['states'])
    assert report['model_transition_calls']==len(report['explored_history'])<=5
    assert check_exploration_geometry(task,report['explored_history'])['feasible']
    a,b=task.fresh(),task.fresh()
    for identifier in actions:a.step(identifier);b.step(identifier)
    assert a.terminated and b.terminated and a.metrics()==b.metrics()
    assert evaluate_native_spatial_episode(b)['accepted']
    assert a.metrics()['target_removed_mm3']==pytest.approx(task.case._native_config.voxel_volume_mm3)
    assert task._steps==0


def test_private_reference_permutation_preserves_depth2_path_and_nominal_credit():
    a,b=make_task(recursive=True),make_task(recursive=True,private=True)
    p,ar=select_corridor(a,seconds=15);q,br=select_corridor(b,seconds=15)
    for r in (ar,br):r.pop('wall_seconds')
    assert p==q and ar==br and ar['dependency_preview_calls']>0
    assert a._steps==b._steps==0


# Metadata doubles isolate the exact diagnostic contract; they never claim
# native physics or a real patient/source authority.
def helper_fixture(*,bad=None,count=1):
    blocker=(4,10,10);prerequisite=(2,11,11);calls=[]
    rays=[SimpleNamespace(tool_id='generated_tool',entry_mm=(0.,0.,0.),tip_mm=(float(i+1),0.,0.),
        voxel=blocker,proposal_id='generated'+str(i)) for i in range(count)]
    if count==1:rays += [rays[0],SimpleNamespace(tool_id='generated_tool',entry_mm=(0.,0.,0.),
        tip_mm=(99.,0.,0.),voxel=(99,99,99),proposal_id='unrelated')]
    class Engine:
        state_hash='generated_state'
        def clone(self):return Engine()
        def preview_stroke(self,tool,tip,*,entry_mm,**kwargs):
            calls.append((tool,tip,entry_mm,kwargs))
            evidence={'source_state_hash':'generated_state' if bad!='stale' else 'other_state',
                'source_hash':'generated_source','decision_model_hash':'generated_model',
                'entry_mm':list(entry_mm),'requested_tip_mm':list(tip),'tool':{'tool_id':tool},
                'reason':'SHAFT_BLOCKED_BY_REMAINING_NATIVE_TISSUE',
                'blocked_indices_native':[list(prerequisite)],'complete_first_failure_set':False,
                'blocked_cell_count':2,'retained_cell_count':1,'truncated':True}
            evidence['fingerprint']=semantic_digest(evidence)
            if bad=='fingerprint':evidence['fingerprint']='tampered'
            return SimpleNamespace(feasible=False,reason=evidence['reason'],
                obstruction_diagnostic=None if bad=='missing' else evidence)
    work=SimpleNamespace(_engine=Engine(),_page_catalog=SimpleNamespace(proposals=rays),
        _config=SimpleNamespace(source_hash='generated_source',fingerprint='generated_model'))
    row={'_active_key':('generated_tool',(0.,0.,0.),(10.,0.,0.))}
    report={'diagnostic_preview_calls':0,'dependency_preview_calls':0,'call_cap_reached':False}
    return work,{blocker},row,report,calls


def test_only_current_emitted_centered_physical_pairs_and_censored_evidence():
    work,blockers,row,report,calls=helper_fixture()
    cells,evidence=_current_prerequisites(work,blockers,'generated_state',row,report,lambda:None)
    assert cells=={(2,11,11)} and len(calls)==1
    assert evidence['emitted_centered_pair_count']==1 and len(evidence['probes'])==1
    assert evidence['probes'][0]['sidecar']['truncated']
    assert report['dependency_preview_calls']==report['diagnostic_preview_calls']==1
    assert work._engine.state_hash=='generated_state'


@pytest.mark.parametrize('bad',['stale','fingerprint','missing'])
def test_sidecar_binding_refusal_never_becomes_a_legal_action(bad):
    work,blockers,row,report,calls=helper_fixture(bad=bad)
    with pytest.raises(ValueError):_current_prerequisites(work,blockers,'generated_state',row,report,lambda:None)
    assert len(calls)==report['dependency_preview_calls']==1
    assert row['dependency_diagnostics']['selected_effect'] is None


def test_dependency_cap_refuses_before_any_query_and_cancellation_is_not_stop():
    work,blockers,row,report,calls=helper_fixture(count=33)
    with pytest.raises(ValueError,match='bound'):_current_prerequisites(work,blockers,'generated_state',row,report,lambda:None)
    assert report['call_cap_reached'] and not calls and row['dependency_diagnostics']['pair_cap_reached']
    work,blockers,row,report,calls=helper_fixture()
    def cancelled():raise InterruptedError('generated cancellation')
    with pytest.raises(InterruptedError):_current_prerequisites(work,blockers,'generated_state',row,report,cancelled)
    assert not calls and report['dependency_preview_calls']==0
    with pytest.raises(ObservedSearchLimit) as caught:select_corridor(make_task(),seconds=1e-12)
    assert caught.value.accounting['time_cap_reached'] and not caught.value.accounting['complete']
