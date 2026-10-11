"""Generated fixed-prefix intervention checks, current certification and replay."""
from dataclasses import replace
from pathlib import Path
import copy
import sys
import numpy as np
import pytest
ROOT=next(p for p in Path(__file__).resolve().parents if (p/'src/resectionlab').is_dir())
HERE=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'src'),str(ROOT/'tests'),str(HERE)]
from resectionlab.core import thaw_json,semantic_digest
from resectionlab.focused_paged_task import FocusedPagedObstructionTask
from resectionlab.focused_corridor_selector import select_corridor as baseline
from resectionlab.native_spatial_evaluation import evaluate_native_spatial_episode
from resectionlab.observed_search import ObservedSearchLimit
from test_obstruction_opening_proposals import source
from preparation_choice_selector import (select_corridor,seal_protocol,_current_choice,
    PROTOCOL_SCHEMA,BASELINE_SELECTOR_SHA256,INTERPRETATION)


def make_task(positive=False):
    case=source(cap=120)
    support=np.zeros(case.observed_support.shape,bool)
    support[2:9,8:11,8:11]=True;support[12,9,9]=True
    nominal=np.zeros(support.shape,np.float32);nominal[12,9,9]=1
    if positive:nominal[2,8:11,8:11]=1
    case=replace(case,observed_support=support,nominal_target=nominal,reference_target=nominal,
        proposal_config=replace(case.proposal_config,obstruction_page_size=8,
            obstruction_selection_rule='target_ray_coverage_v3'))
    return FocusedPagedObstructionTask(case,max_steps=8)


def physical(row):
    return {"tool_id":row["tool_id"],"entry_mm":[float(v) for v in row["entry_mm"]],
        "tip_mm":[float(v) for v in row["tip_mm"]]}


def protocol_from(task,report):
    states=report['states'];row=states[4]
    assert row['goal_focus'] is not None and row['selected']['current_blockers_removed']>0, 'Generated decision5 must be a direct current-goal preparation'
    return seal_protocol({'schema':PROTOCOL_SCHEMA,'baseline_selector_sha256':BASELINE_SELECTOR_SHA256,
        'source_hash':task.case.source_hash,'decision_model_hash':task.decision_model_hash,
        'native_decision_model_hash':task._config.fingerprint,'max_steps':task.max_steps,
        'prefix':[{'physical_action':physical(r['selected']['action_geometry']),'source_state_hash':r['source_state_hash'],
                   'result_state_hash':r['selected']['result_state_hash']} for r in states[:4]],
        'decision':{'index':5,'source_state_hash':row['source_state_hash'],
            'current_tool':report['explored_action_records'][4]['current_tool'],
            'active_goal':row['selected']['goal'],'physical_action':physical(row['selected']['action_geometry'])},
        'interpretation':INTERPRETATION})


@pytest.fixture(scope='module')
def negative():
    task=make_task();packets,report=baseline(task,seconds=20)
    return task,packets,report,protocol_from(task,report)


def changed(protocol,change):
    record=thaw_json(protocol);record.pop('seal');change(record)
    return seal_protocol(record)


def test_same_choice_native_baseline_full_history_packets_counts_and_stop_replay(negative):
    task,packets,prior,protocol=negative
    got,r=select_corridor(task,seconds=20,intervention_protocol=protocol)
    assert r['complete'] and r['prefix_verified_count']==4 and r['intervention_applied']
    assert r['explored_history']==prior['explored_history']
    assert r['explored_action_records']==prior['explored_action_records'] and got==packets
    assert r['model_transition_calls']==prior['model_transition_calls']==len(r['explored_history'])==8
    assert r['diagnostic_preview_calls']==prior['diagnostic_preview_calls']
    assert r['dependency_preview_calls']==prior['dependency_preview_calls']
    assert r['states'][4]['choice_intervention']['interpretation']==INTERPRETATION
    assert r['states'][4]['choice_intervention']['actual_removed_current_blockers']
    a,b=task.fresh(),task.fresh()
    for packet in got:a.step_record(packet);b.step_record(packet)
    assert a.metrics()==b.metrics() and evaluate_native_spatial_episode(b)['accepted']
    assert [p['action_id'] for p in got]==['STOP'] and r['explored_total_return']<0


def test_prefix_mismatch_refuses_without_forcing_action_and_charges_paid_prefix(negative):
    task,_,prior,protocol=negative
    def alter(r):r['prefix'][1]['physical_action']['tip_mm'][0]+=0.25
    bad=changed(protocol,alter)
    with pytest.raises(ObservedSearchLimit) as caught:select_corridor(task,seconds=20,intervention_protocol=bad)
    r=caught.value.accounting
    assert not r['complete'] and not r['intervention_applied'] and r['prefix_verified_count']==1
    assert r['model_transition_calls']==len(r['explored_history'])==1
    assert r['explored_history']==prior['explored_history'][:1]
    assert not caught.value.best_sequence and task._steps==0


def test_unavailable_choice_refuses_after_exact_four_paid_decisions(negative):
    task,_,prior,protocol=negative
    def alter(r):r['decision']['physical_action']['tip_mm'][0]+=0.25
    with pytest.raises(ObservedSearchLimit) as caught:
        select_corridor(task,seconds=20,intervention_protocol=changed(protocol,alter))
    r=caught.value.accounting
    assert not r['complete'] and r['prefix_verified_count']==4 and not r['intervention_applied']
    assert r['model_transition_calls']==4 and r['explored_history']==prior['explored_history'][:4]
    assert 'unavailable' in r['failure']['message'] and task._steps==0


def test_current_blocked_goal_cannot_be_injected_as_certified_preparation(negative):
    task,_,prior,protocol=negative
    def alter(r):r['decision']['physical_action']=copy.deepcopy(r['decision']['active_goal'])
    with pytest.raises(ObservedSearchLimit) as caught:
        select_corridor(task,seconds=20,intervention_protocol=changed(protocol,alter))
    r=caught.value.accounting
    assert not r['complete'] and r['model_transition_calls']==4 and not r['intervention_applied']
    assert r['explored_history']==prior['explored_history'][:4] and task._steps==0


def test_protocol_seal_context_and_continuity_reject_before_any_transition(negative):
    task,_,_,protocol=negative
    bad=thaw_json(protocol);bad['seal']='tampered'
    with pytest.raises(ValueError,match='seal'):select_corridor(task,seconds=20,intervention_protocol=bad)
    bad=thaw_json(protocol);bad.pop('seal');bad['prefix'][2]['source_state_hash']='unjoined'
    with pytest.raises(ValueError,match='continuous'):seal_protocol(bad)
    def alter(r):r['source_hash']='different'
    with pytest.raises(ValueError,match='differs'):select_corridor(task,seconds=20,intervention_protocol=changed(protocol,alter))
    assert task._steps==0 and task._base_inventory is None


def test_positive_best_prefix_survives_negative_continuation_with_packet_replay():
    task=make_task(positive=True);expected,prior=baseline(task,seconds=20)
    protocol=protocol_from(task,prior)
    packets,r=select_corridor(task,seconds=20,intervention_protocol=protocol)
    assert r['complete'] and r['intervention_applied'] and r['prefix_verified_count']==4
    assert r['explored_history']==prior['explored_history'] and packets==expected
    assert 0<r['selected_prefix_length']<r['model_transition_calls']==8
    assert r['estimated_incremental_return']>0 and r['estimated_incremental_return']>r['explored_total_return']
    assert packets[-1]['action_id']=='STOP'
    a,b=task.fresh(),task.fresh()
    for packet in packets:a.step_record(packet);b.step_record(packet)
    assert a.metrics()==b.metrics() and evaluate_native_spatial_episode(b)['accepted']
    assert a.metrics()['target_removed_mm3']>0


def test_alternate_current_certified_choice_changes_only_fifth_decision_then_native_replays(negative):
    task,_,prior,protocol=negative
    current=task.planning_clone()
    for packet in prior['explored_action_records'][:4]:current.advance_record(packet)
    next_packet=prior['explored_action_records'][4]
    current.set_focus(next_packet['focus'])
    blockers={tuple(c) for c in prior['states'][4]['goals'][0]['retained_blocker_cells']}
    ordinary=protocol['decision']['physical_action'];alternate=None
    for page,ids in current.planning_action_pages():
        for identifier in ids:
            record=page._inventory[identifier].to_history_record();geometry=physical(record)
            if geometry!=thaw_json(ordinary) and blockers & {tuple(c) for c in record['removed_indices_native']}:
                if alternate is None:alternate=geometry
    assert alternate is not None,'Generated state must contain a genuine alternative certified blocker-removing preparation'
    def alter(r):r['decision']['physical_action']=alternate
    _,r=select_corridor(task,seconds=20,intervention_protocol=changed(protocol,alter))
    assert r['complete'] and r['intervention_applied'] and r['model_transition_calls']==8
    assert r['explored_history'][:4]==prior['explored_history'][:4]
    assert physical(r['explored_history'][4])==alternate
    assert physical(r['explored_history'][4])!=physical(prior['explored_history'][4])
    # This explicitly replays the diagnostic negative exploration too; it is
    # not relabeled as the objective-selected positive plan.
    replay=task.fresh()
    for packet in r['explored_action_records']:replay.step_record(packet)
    assert replay.terminated and evaluate_native_spatial_episode(replay)['accepted']
    for actual,nominal in zip(replay.metrics()['history'],r['explored_history']):
        assert actual['outcome_scope']=='separate_evaluator_reference'
        assert nominal['outcome_scope']=='permitted_nominal_model'
        assert {k:v for k,v in actual.items() if k!='outcome_scope'}=={k:v for k,v in nominal.items() if k!='outcome_scope'}
