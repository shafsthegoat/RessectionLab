from dataclasses import replace
from pathlib import Path
import sys
import numpy as np
import pytest
ROOT=next(p for p in Path(__file__).resolve().parents if (p/'src/resectionlab').is_dir())
sys.path.insert(0,str(ROOT/'tests'));sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from test_obstruction_opening_proposals import source
from resectionlab.paged_obstruction import PagedObstructionTask
from resectionlab.native_spatial_evaluation import evaluate_native_spatial_episode
from resectionlab.observed_search import ObservedSearchLimit
from corridor_selector import select_corridor,complete_prefix,check_exploration_geometry

def task(private=False):
    case=source(cap=120,extra=((4,10,8),))
    nominal=np.zeros(case.nominal_target.shape,np.float32);nominal[9,9,9]=1
    case=replace(case,nominal_target=nominal,reference_target=np.zeros_like(nominal) if private else nominal,
        proposal_config=replace(case.proposal_config,obstruction_page_size=8,obstruction_selection_rule='target_ray_coverage_v3'))
    return PagedObstructionTask(case,max_steps=4)

def test_real_two_blockers_persistent_refresh_complete_positive_and_exact_replay():
    initial=task();path,report=select_corridor(initial,seconds=15)
    assert report['complete'] and report['estimated_incremental_return']>0
    history=report['explored_history']
    assert sum(r['reward']<0 for r in history)>=2
    assert sum(r['target_removed_mm3'] for r in history)==pytest.approx(initial.case._native_config.voxel_volume_mm3)
    assert report['model_transition_calls']==len(history)<=4
    assert report['diagnostic_preview_calls']==sum(len(s['goals']) for s in report['states'])
    assert all(s['complete_page_enumeration'] for s in report['states'])
    moves=[s for s in report['states'] if s['selected']['action_id']!='STOP']
    assert len(set(s['source_state_hash'] for s in moves))==len(moves)
    assert moves[0]['selected']['goal']==moves[1]['selected']['goal']
    a=initial.fresh();b=initial.fresh()
    for x in path:a.step(x)
    for x in path:b.step(x)
    assert a.terminated and b.terminated and a.metrics()==b.metrics()
    assert evaluate_native_spatial_episode(b)['accepted']
    assert check_exploration_geometry(initial,history)['feasible']
    assert initial._steps==0

def test_private_reference_permutation_and_immutable_original():
    a,b=task(),task(True)
    p,ar=select_corridor(a,seconds=15);q,br=select_corridor(b,seconds=15)
    for r in (ar,br):r.pop('wall_seconds')
    assert p==q and ar==br and a._steps==b._steps==0
    assert not np.array_equal(a.case.reference_target,b.case.reference_target)

def test_cap_is_unresolved_not_a_completed_stop():
    with pytest.raises(ObservedSearchLimit) as caught:select_corridor(task(),seconds=1e-12)
    assert caught.value.accounting['time_cap_reached'] and not caught.value.accounting['complete']
    assert caught.value.accounting['model_transition_calls']==0

def test_prefix_horizon_and_stop_ties():
    assert complete_prefix(['a','b'],0,4)==('STOP',)
    assert complete_prefix(['a','b'],1,4)==('a','STOP')
    assert complete_prefix(['a','b'],2,2)==('a','b')
