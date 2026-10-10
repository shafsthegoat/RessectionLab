"""Source-written scalar glue controls; no patient/model imports or execution."""
import math
from pathlib import Path
import sys

HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE))
import cohort_worker as worker
import pilot_contract as contract


def test_ordering_regret_and_ties():
    label={'action_ids':['STOP','good','bad','tied'],'action_mask':[True]*4,
        'teacher_action':'good','rewards':[0.,2.,1.,2.]}
    score={'logits':[0.,2.,1.,2.],'greedy_action':'good','teacher_CE':.8}
    row=worker.state_metrics(score,label)
    assert row['public_nominal_regret']==row['conditional_motion_regret']==0.
    assert row['strict_motion_pairs']==row['correct_pairs']==2
    assert row['reversed_pairs']==row['logit_tied_pairs']==0
    wrong={**score,'logits':[0.,1.,2.,1.],'greedy_action':'bad'}
    bad=worker.state_metrics(wrong,label)
    assert bad['public_nominal_regret']==1 and bad['reversed_pairs']==2
    assert bad['normalized_pair_loss']>row['normalized_pair_loss']


def test_STOP_uses_original_CE_and_no_training_pairs():
    label={'action_ids':['STOP','move'],'action_mask':[True,True],
        'teacher_action':'STOP','rewards':[0.,-1.]}
    row=worker.state_metrics({'logits':[2.,1.],'greedy_action':'STOP','teacher_CE':.123},label)
    assert row['ranking_objective']==.123 and row['strict_motion_pairs']==0


def test_single_motion_has_gate_without_pair_term():
    label={'action_ids':['STOP','move'],'action_mask':[True,True],
        'teacher_action':'move','rewards':[0.,1.]}
    row=worker.state_metrics({'logits':[0.,1.],'greedy_action':'move','teacher_CE':.5},label)
    assert row['strict_motion_pairs']==0 and row['normalized_pair_loss']==0.
    assert math.isclose(row['ranking_objective'],math.log1p(math.exp(-1.)),abs_tol=1e-14)


def test_only_fixed_IL64_caps_and_complete_route_costs():
    assert set(contract.METHODS)=={'IL'}
    cap=contract.METHODS['IL']
    assert cap['updates']==64 and cap['loss_forward_cap']==29*64
    assert cap['policy_forward_cap']==29*64+29+4*24
    assert cap['ranking_pair_terms_total']==4970*64
    rows=[{'action_id':'move','tool_id':'a','complete_tool_path_length_mm':2.},
          {'action_id':'other','tool_id':'b','complete_tool_path_length_mm':3.},
          {'action_id':'STOP'}]
    assert worker.route_costs(rows)=={'motion_count':2,'complete_tool_path_length_mm':5.,'tool_changes':1,'terminal_reason':'STOP'}
