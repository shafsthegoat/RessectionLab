"""Generated scalar contracts plus one prescribed coarse row; root dispatches."""
from copy import deepcopy
import hashlib
import importlib.util
import json
from pathlib import Path
import sys

import numpy as np
import pytest

HERE=Path(__file__).resolve().parent
ROOT=next(p for p in HERE.parents if (p/'src/resectionlab').is_dir())
sys.path.insert(0,str(ROOT/'src'))
spec=importlib.util.spec_from_file_location('fixed18_exposure_test_adapter',HERE/'probe_exposure.py')
P=importlib.util.module_from_spec(spec);spec.loader.exec_module(P)
EVAL_SHA='21df1470e7a65c62eaae4857c8b7c435b1ae448783934465d3cf3bdef2a82522'


def setup():
    P.install({'files':{str(P.EVALUATOR_PATH.relative_to(ROOT)):EVAL_SHA}})
    prior=P.bound_json(P.config()['retained_baseline_report'])
    P.BASELINE_BY_ID={row['row_id']:row for row in prior['rows']}
    P.UNION.RUN_COUNTS.update(native_strokes_started=0,native_strokes_returned=0,native_seconds=0.)
    P.UNION.COVERAGE_WORK.update({name:0 for name in P.UNION.COVERAGE_WORK});P.PENDING.clear()
    P.AUDIT_WORK.update({name:0. if name.endswith('_seconds') else 0 for name in P.AUDIT_WORK})
    return prior


def test_serialized_numpy_refusal_coordinates_match_without_discarding_changes():
    row={'native':{'removed_cells':0,'committed_history_hash':'x'},
         'actions':[{'status':'rejected','failure_tip_mm':tuple(np.array([0.,1.8,.9375]))}]}
    saved=json.loads(json.dumps(row))
    assert not P.equivalent(P.projection(saved),P.projection(row))
    assert P.equivalent(P.normalized_projection(saved),P.normalized_projection(row))
    row['actions'][0]['failure_tip_mm']=(0.,1.8,1.)
    assert not P.equivalent(P.normalized_projection(saved),P.normalized_projection(row))


def fixture():
    rows=[]
    for i in range(18):
        rows.append({'row_id':str(i),'spacing_mm':.25 if i>=12 else 1.,'actions':[{}]*10,
            'native':{'complete_sequence':i<13,'accepted_actions':1},'coverage':{'budget_refusals':0},
            'independent_audit':{'status':'completed','certificate':{'feasible':True}},
            'action_result_projection_equal':False,
            'baseline_comparison':{'physical_target_difference_mm3':1. if i==12 else 0.},
            'partial_evacuation':{'status':'independently_checked_and_clipped','lower_bound_gain_mm3':1.,
                'gap_after_mm3':2.,'physical_target_inside_source_interval':True,
                'physical_target_within_source_interval':True,'within_analytic_target_cap':True,
                'parent_target_mass_enclosure_pass':True}})
    return {'rows':deepcopy(rows)},{'status':'completed','source_unchanged':True,'rows':rows}


def test_new_model_requires_gain_and_preserved_completions_not_action_equality():
    prior,report=fixture();result=P.criterion(prior,report)
    assert result['meets_predeclared_useful_criterion'] and not result['action_result_equality_required']
    assert result['strict_fine_target_gain_rows']==['12']
    assert result['baseline_completion_count']==13


@pytest.mark.parametrize('failure',['missing','completion','audit','target_regression','no_gain','mass','interval','cap','budget'])
def test_each_predeclared_negative_remains_negative(failure):
    prior,report=fixture()
    if failure=='missing':report['rows'].pop()
    elif failure=='completion':report['rows'][0]['native']['complete_sequence']=False
    elif failure=='audit':report['rows'][0]['independent_audit']['certificate']['feasible']=False
    elif failure=='target_regression':report['rows'][13]['baseline_comparison']['physical_target_difference_mm3']=-2e-8
    elif failure=='no_gain':report['rows'][12]['baseline_comparison']['physical_target_difference_mm3']=1e-8
    elif failure=='mass':report['rows'][0]['partial_evacuation']['parent_target_mass_enclosure_pass']=False
    elif failure=='interval':report['rows'][0]['partial_evacuation']['physical_target_within_source_interval']=False
    elif failure=='cap':report['rows'][0]['partial_evacuation']['within_analytic_target_cap']=False
    elif failure=='budget':report['rows'][0]['coverage']['budget_refusals']=1
    assert not P.criterion(prior,report)['meets_predeclared_useful_criterion']


def test_refused_target_preview_has_no_committed_analytic_target_cap():
    row={'actions':[{'index':10,'status':'accepted'}],'exact_continuous_target_cap_mm3':4.}
    assert P.accepted_target_cap(row)==4.
    for status in ('rejected','unexecuted_prior_refusal'):
        row['actions'][0]['status']=status
        assert P.accepted_target_cap(row)==0.


def test_explicit_model_preserves_old_depth_caps_geometry_tools_and_reward_rule():
    setup();decl=P.declaration();source,cfg=P.grid_case(1.,decl['grids'][0],decl)
    original_source,original_cfg=P.UNION.GENERIC.grid_case(1.,decl['grids'][0],decl)
    old=P.UNION.cut_spec(decl).to_record();new=cfg.cut_union.to_record()
    assert new['exposure_mode']=='same_step_parent_closure'
    assert new['mode']=='cumulative_affine_parent_closure_cut_union_v3'
    assert {key:value for key,value in new.items() if key not in ('mode','exposure_mode')}=={key:value for key,value in old.items() if key!='mode'}
    assert source.semantic_hash==original_source.semantic_hash
    assert np.array_equal(cfg.affine,original_cfg.affine) and np.array_equal(cfg.tissue_mask,original_cfg.tissue_mask)
    assert cfg.tools==original_cfg.tools and cfg.max_tip_step_mm==original_cfg.max_tip_step_mm
    assert cfg.evacuated_volume.to_record()==P.BASELINE_BY_ID['aligned-suction-1mm']['public_parent_mass_spec']


def test_one_coarse_row_uses_new_history_checker_and_compact_saved_geometry(tmp_path):
    from resectionlab import evaluation
    setup();P.OUTPUT=tmp_path;(tmp_path/'geometry').mkdir();decl=P.declaration()
    row,case,cfg,history=P.numerical_row(1.,decl['grids'][0],decl)
    assert len(row['actions'])==10 and history
    assert row['exposure_model']['exposure_mode']=='same_step_parent_closure'
    audit=P.checked_history(evaluation.independent_check_native_history,case,cfg.tools,history,
        tissue_mask=cfg.tissue_mask,access=cfg.access,cancelled=lambda:False)
    assert audit.feasible and audit.certified_evacuated_geometry is not None
    metrics=row['partial_evacuation']
    assert metrics['parent_target_mass_enclosure_pass'] and metrics['physical_target_within_source_interval']
    assert metrics['within_analytic_target_cap'] and metrics['gap_after_mm3']>=-1e-8
    assert metrics['lower_bound_gain_mm3']>=-1e-8
    raw=(tmp_path/metrics['certified_geometry']['path']).read_bytes()
    assert hashlib.sha256(raw).hexdigest()==metrics['certified_geometry']['sha256']
    sparse=json.loads(raw)
    assert sparse['geometry']['geometry_digest']==metrics['checker_geometry_digest']
    assert sparse['geometry']['cut_union_spec']==cfg.cut_union.to_record()
    assert 'history' not in sparse and len(raw)<2*1024**2
    assert len(json.dumps(row,allow_nan=False))<192*1024
    baseline=P.BASELINE_BY_ID[row['row_id']]['partial_evacuation']
    assert row['baseline_comparison']['physical_target_difference_mm3']==metrics['physical_target_evacuated_mm3']-baseline['physical_target_evacuated_mm3']
