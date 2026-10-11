"""Generated public focus, catalog-owned packets and complete native replay."""
from dataclasses import asdict, replace
import hashlib, importlib.util, os, sys
from pathlib import Path
import numpy as np
import pytest
from resectionlab.core import freeze_json, thaw_json, semantic_digest
from resectionlab.native_proposals import _target_ray_group_order, _focused_goal_order
from resectionlab.native_resection import NativeResectionEngine
from resectionlab.paged_obstruction import PagedObstructionTask
from resectionlab.focused_paged_task import FocusedPagedObstructionTask
from resectionlab.native_spatial_evaluation import evaluate_native_spatial_episode
from resectionlab.observed_search import ObservedSearchLimit
from resectionlab.focused_corridor_selector import select_corridor, physical_key, check_exploration_geometry
from test_obstruction_opening_proposals import source, prepared

ROOT=next(p for p in Path(__file__).resolve().parents if (p/'src/resectionlab').is_dir())
HERE=Path(__file__).resolve().parents[1]


def make_task(*, recursive=False, private=False, horizon=5, cls=FocusedPagedObstructionTask):
    case=source(cap=120)
    support=np.zeros(case.observed_support.shape,bool);goal=(9,9,9)
    cells=((2,10,9),(2,11,11),(4,10,10),goal) if recursive else ((4,10,10),goal)
    for c in cells:support[c]=True
    nominal=np.zeros(support.shape,np.float32);nominal[goal]=1
    case=replace(case,observed_support=support,nominal_target=nominal,
        reference_target=np.zeros_like(nominal) if private else nominal,
        proposal_config=replace(case.proposal_config,obstruction_page_size=8,
            obstruction_selection_rule='target_ray_coverage_v3'))
    return cls(case,max_steps=horizon)

def goal(task):
    batch=task.case._nominal_proposer.propose(task._engine)
    return next(physical_key(r) for r in batch.proposals if task.case.nominal_target[r.voxel]>0)

def page(task, focus=None):
    if focus is not None:task.set_focus(focus)
    task._activate(0)
    return task

def same_physics_and_reward(authoritative, nominal):
    assert len(authoritative)==len(nominal)
    for a,b in zip(authoritative,nominal):
        if a['action_id']=='STOP':
            assert 'outcome_scope' not in a and 'outcome_scope' not in b
        else:
            assert a['outcome_scope']=='separate_evaluator_reference'
            assert b['outcome_scope']=='permitted_nominal_model'
        assert {k:v for k,v in a.items() if k!='outcome_scope'}=={k:v for k,v in b.items() if k!='outcome_scope'}

def resign(row):
    row={k:v for k,v in row.items() if k!='seal'}
    return {**row,'seal':semantic_digest(row)}

def test_none_preserves_historical_provider_and_page_outputs_exact(monkeypatch):
    path=Path(os.environ.get('FOCUSED_BASELINE_PROVIDER',str(ROOT/'artifacts/select013-target-ray-priority-v1/stage/resectionlab/native_proposals.py')))
    assert path.is_file(), "Required pinned historical provider fixture is absent"
    assert hashlib.sha256(path.read_bytes()).hexdigest()=='fc135cb1bf8a3cbacfeb097eeabee5cd9fafc5dee7a89eebd2b9b43ab63c8d12'
    spec=importlib.util.spec_from_file_location('resectionlab._before_focus',path)
    old=importlib.util.module_from_spec(spec);sys.modules[spec.name]=old;spec.loader.exec_module(old)
    task=make_task();engine,new,batch,evidence=prepared(task.case)
    prior=old.PreparedNominalCavityProposer(task._config,new._nominal,nominal_provenance=new._provenance,
        config=old.NominalCavityProposalConfig(**task.case.proposal_config.to_record()),
        index_affine=new._index_affine,index_frame_record=new._frame_record)
    old_batch=prior.propose(engine)
    def forbidden(*a,**k):raise AssertionError('Allocation must not call native geometry')
    with monkeypatch.context() as m:
        m.setattr(NativeResectionEngine,'preview_stroke',forbidden)
        a=prior.append_obstruction_openings(old_batch,engine,evidence)
        b=new.append_obstruction_openings(batch,engine,evidence,focus=None)
    assert asdict(a)==asdict(b) and a.obstruction_accounting==b.obstruction_accounting
    before=make_task(cls=PagedObstructionTask).planning_clone();after=task.planning_clone()
    aa=list(before.planning_action_pages());bb=list(after.planning_action_pages())
    assert [(ids,t.candidate_inventory()) for t,ids in aa]==[(ids,t.candidate_inventory()) for t,ids in bb]
    assert before.decision_model_hash==after.decision_model_hash and before.case.source_hash==after.case.source_hash


def test_crowded_focus_prioritizes_exact_set_then_v3_fill_and_keeps_co_blockers():
    cells=[(i,0,0) for i in range(26)];eligible=set(cells[1:]);focus=set(cells[18:26])
    order,groups=_target_ray_group_order(cells,[((0,0),cells[:18]),((1,0),cells[18:])],eligible,set(),16)
    result,rows,selected=_focused_goal_order(order,focus,eligible,groups,16)
    assert result[:8]==cells[18:26] and focus<=selected and len(selected)==16
    assert result[8:]==[c for c in order if c not in focus]
    assert len(set(result))==len(cells) and cells[0] not in selected
    assert rows[0]['unselected_co_blockers'] and rows[1]['all_retained_co_blockers_selected']
    assert _focused_goal_order(order,set(),eligible,groups,16)[:2]==(order,groups)
    with pytest.raises(ValueError):_focused_goal_order(order,{(999,0,0)},eligible,groups,16)


@pytest.mark.parametrize('field,value',[('source_state_hash','stale'),('source_hash','wrong'),
    ('proposal_model_hash','wrong'),('native_decision_model_hash','wrong'),('tool_id','unknown'),
    ('tip_mm',[0.,0.,0.])])
def test_stale_or_unknown_focus_refused_before_preview(field,value,monkeypatch):
    t=make_task();f=thaw_json(t.focus_record(goal(t)));f[field]=value
    calls=[]
    def forbidden(*a,**k):calls.append(1);raise AssertionError('No preview on bad focus')
    monkeypatch.setattr(NativeResectionEngine,'preview_stroke',forbidden)
    with pytest.raises(ValueError):t.set_focus(f)
    assert t._goal_focus is None and not calls and t._steps==0


def test_focus_world_unchanged_but_packet_catalog_distinct_and_no_bare_execution():
    a=make_task();b=a.clone();f=a.focus_record(goal(a));b.set_focus(f)
    source_state=a._engine.state_hash;source=a.case.source_hash;model=a.decision_model_hash
    page(a);page(b)
    left,right=a.export_action('STOP'),b.export_action('STOP')
    assert left['action_id']==right['action_id']=='STOP' and left['physical_action'] is None
    assert left['query_catalog_digest']!=right['query_catalog_digest'] and left['seal']!=right['seal']
    assert a.case.source_hash==b.case.source_hash==source and a.decision_model_hash==b.decision_model_hash==model
    assert a._engine.state_hash==b._engine.state_hash==source_state
    with pytest.raises(ValueError,match='different'):a.step_record(right)
    with pytest.raises(ValueError,match='before'):a.set_focus(f)
    with pytest.raises(ValueError,match='packet'):b.step('STOP')
    with pytest.raises(ValueError,match='packet'):b.planning_clone().advance_planning('STOP')
    with pytest.raises(TypeError):right['action_id']='tampered'
    b.step_record(right)
    assert b.terminated and b._goal_focus is None and b._base_inventory is None
    assert not a.terminated and a._goal_focus is None
    fresh=b.fresh();fresh.step_record(right)
    assert fresh.metrics()==b.metrics() and evaluate_native_spatial_episode(fresh)['accepted']


def test_clone_reset_stale_packet_and_query_tampering():
    task=make_task();f=task.focus_record(goal(task));task.set_focus(f);branch=task.clone()
    page(branch);packet=branch.export_action('STOP')
    assert task._active_page is None and task._base_inventory is None
    branch.step_record(packet);branch.reset()
    assert branch._goal_focus is None and branch._base_inventory is None and branch._steps==0
    assert task._goal_focus==f and not task.terminated
    page(task);bad=thaw_json(task.export_action('STOP'));bad['query_catalog_digest']='wrong'
    with pytest.raises(ValueError,match='catalog'):task.step_record(resign(bad))
    bad=thaw_json(task.export_action('STOP'));bad['physical_action']={'tool_id':'invented'}
    with pytest.raises(ValueError,match='physical'):task.step_record(resign(bad))
    task.step_record(task.export_action('STOP'))
    with pytest.raises(ValueError):task.step_record(packet)
    rogue=make_task();rogue._goal_focus=f
    with pytest.raises(RuntimeError,match='query changed'):rogue.clone()
    class Unapproved(FocusedPagedObstructionTask):pass
    with pytest.raises(ValueError,match='page-aware'):make_task(cls=Unapproved)


def test_actual_native_focused_recursive_chain_packets_and_complete_independent_replay():
    task=make_task(recursive=True);packets,r=select_corridor(task,seconds=15)
    assert r['complete'] and r['estimated_incremental_return']>0 and r['dependency_preview_calls']>0
    assert any(p['focus'] is not None for p in packets)
    assert r['selected_action_records']==[thaw_json(p) for p in packets]
    assert r['diagnostic_preview_calls']==r['goal_diagnostic_preview_calls']+r['dependency_preview_calls']
    for row in r['states']:
        assert row['complete_page_enumeration']
        acct=row['obstruction_accounting']
        assert acct['selected_cells']<=16 and acct['added_candidates_to_verify']<=32
        if row['goal_focus'] is not None:
            focused=acct['goal_focus']
            assert focused['record']==row['goal_focus']
            assert set(map(tuple,focused['aperture_eligible_cells']))<=set(map(tuple,focused['selected_cells']))
    assert check_exploration_geometry(task,r['explored_history'])['feasible']
    a,b=task.fresh(),task.fresh()
    for packet in packets:a.step_record(packet);b.step_record(packet)
    assert a.terminated and a.metrics()==b.metrics()
    same_physics_and_reward(a.metrics()['history'],r['explored_history'])
    assert evaluate_native_spatial_episode(b)['accepted']
    assert a.metrics()['target_removed_mm3']==pytest.approx(task._config.voxel_volume_mm3)
    assert task._steps==0 and task._base_inventory is None


def test_best_zero_prefix_exports_real_stop_and_private_reference_does_not_rank():
    a=make_task(recursive=True,horizon=1);b=make_task(recursive=True,private=True,horizon=1)
    p,ar=select_corridor(a,seconds=15);q,br=select_corridor(b,seconds=15)
    assert [x['action_id'] for x in p]==['STOP'] and p==q
    assert ar['explored_total_return']<0 and ar['selected_prefix_length']==0
    assert ar['explored_history']==br['explored_history'] and ar['estimated_incremental_return']==0
    fresh=a.fresh();fresh.step_record(p[0])
    assert fresh.terminated and fresh.metrics()['history'][-1]['action_id']=='STOP'
    assert evaluate_native_spatial_episode(fresh)['accepted']
    with pytest.raises(ObservedSearchLimit) as caught:select_corridor(make_task(),seconds=1e-12)
    assert not caught.value.accounting['complete'] and caught.value.best_sequence==()


def test_positive_best_prefix_stop_packet_survives_negative_suffix():
    original=make_task(recursive=True,horizon=2)
    nominal=np.array(original.case.nominal_target);nominal[2,10,9]=1
    task=FocusedPagedObstructionTask(replace(original.case,nominal_target=nominal,reference_target=nominal),max_steps=2)
    packets,r=select_corridor(task,seconds=15)
    assert len(r['explored_history'])==2
    assert r['explored_history'][0]['reward']>0 and r['explored_history'][1]['reward']<0
    assert r['selected_prefix_length']==1 and len(packets)==2 and packets[-1]['action_id']=='STOP'
    assert packets[-1]['steps_taken']==1 and packets[-1]['focus'] is not None
    assert r['estimated_incremental_return']>r['explored_total_return']
    a,b=task.fresh(),task.fresh()
    for packet in packets:a.step_record(packet);b.step_record(packet)
    assert a.metrics()==b.metrics() and evaluate_native_spatial_episode(b)['accepted']
    same_physics_and_reward(a.metrics()['history'][:1],r['explored_history'][:1])
    assert a.metrics()['history'][1]['action_id']=='STOP'
