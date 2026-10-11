"""Focused pure bookkeeping + one prescribed coarse generated row, root-run only."""
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
_spec=importlib.util.spec_from_file_location('partial_fixed18_test_adapter',HERE/'probe_partial.py')
P=importlib.util.module_from_spec(_spec);_spec.loader.exec_module(P)
EVAL_SHA='21df1470e7a65c62eaae4857c8b7c435b1ae448783934465d3cf3bdef2a82522'


def setup():
    P.install({'files':{str(P.EVALUATOR_PATH.relative_to(ROOT)):EVAL_SHA}})
    prior=P.bound_json(P.config()['retained_union_report']);P.UNION_BY_ID={r['row_id']:r for r in prior['rows']}
    P.UNION.RUN_COUNTS.update(native_strokes_started=0,native_strokes_returned=0,native_seconds=0.)
    P.UNION.COVERAGE_WORK.update({name:0 for name in P.UNION.COVERAGE_WORK});P.PENDING.clear()
    return prior


def test_rounding_bin_bounds_use_only_public_float32_and_match_independent_helper():
    setup()
    f=np.array([0.,1.,.1,.5,np.nextafter(np.float32(0),np.float32(1))],np.float32).reshape(1,1,-1)
    lo,hi=P.rounding_bounds(f,np.ones(f.shape,bool))
    for index in np.ndindex(f.shape):
        expected=P.EVALUATOR.float32_fraction_enclosure(float(f[index]))
        assert lo[index]==expected[0] and hi[index]==expected[1]
    mask=np.ones(f.shape,bool);mask[0,0,1]=False
    lo,hi=P.rounding_bounds(f,mask);assert lo[0,0,1]==hi[0,0,1]==0
    with pytest.raises(ValueError):P.rounding_bounds(f.astype(float),mask)


def test_projection_ignores_only_new_fields_state_hash_and_timing():
    a={'native':{'removed_cells':1,'frozen_reward_total':-.1,'committed_history_hash':'old'},
       'actions':[{'status':'accepted','source_state_hash':'old','result_state_hash':'old2','native_seconds':1.,'frozen_reward':-.1}]}
    b=deepcopy(a);b['native']['committed_history_hash']='new';b['actions'][0].update(source_state_hash='new',result_state_hash='new2',native_seconds=2.,partial_evacuation={'volume':1.})
    assert P.projection(a)==P.projection(b)
    b['actions'][0]['frozen_reward']=.1;assert not P.equivalent(P.projection(a),P.projection(b))


def test_predeclared_criterion_preserves_18_rows_bounds_and_old_completion():
    rows=[]
    for i in range(18):
        rows.append({'row_id':str(i),'spacing_mm':.25 if i>=12 else 1.,'actions':[{}]*10,
          'native':{'complete_sequence':i<13,'accepted_actions':1},'coverage':{'budget_refusals':0},
          'independent_audit':{'status':'completed','certificate':{'feasible':True}},'legacy_projection_equal_within_tolerance':True,
          'partial_evacuation':{'status':'independently_checked_and_clipped','lower_bound_gain_mm3':1.,'gap_after_mm3':2.,
            'physical_target_inside_source_interval':True,'within_analytic_target_cap':True,'parent_target_mass_enclosure_pass':True}})
    prior={'rows':deepcopy(rows)};default=deepcopy(prior)
    for r in default['rows'][12:]:r['native']['complete_sequence']=False
    report={'status':'completed','source_unchanged':True,'rows':rows}
    assert P.criterion(default,prior,report)['meets_predeclared_useful_criterion']
    for key in ('audit','parentmass','finegain','lostcompletion','reward','missing'):
        bad=deepcopy(report)
        if key=='audit':bad['rows'][0]['independent_audit']['status']='cancelled_cap_reached'
        elif key=='parentmass':bad['rows'][0]['partial_evacuation']['parent_target_mass_enclosure_pass']=False
        elif key=='finegain':bad['rows'][12]['partial_evacuation']['lower_bound_gain_mm3']=0.
        elif key=='lostcompletion':bad['rows'][0]['native']['complete_sequence']=False
        elif key=='reward':bad['rows'][0]['legacy_projection_equal_within_tolerance']=False
        else:bad['rows'].pop()
        assert not P.criterion(default,prior,bad)['meets_predeclared_useful_criterion']


def test_one_coarse_fixed_row_checker_geometry_and_compact_json(tmp_path):
    from resectionlab import evaluation
    setup();P.OUTPUT=tmp_path;(tmp_path/'geometry').mkdir();spec=P.declaration()
    row,case,cfg,history=P.numerical_row(1.,spec['grids'][0],spec)
    assert len(row['actions'])==10 and history and row['legacy_projection_equal_within_tolerance']
    audit=P.checked_history(evaluation.independent_check_native_history,case,cfg.tools,history,
        tissue_mask=cfg.tissue_mask,access=cfg.access,cancelled=lambda:False)
    assert audit.feasible and audit.certified_evacuated_geometry is not None
    readout=row['partial_evacuation'];assert readout['status']=='independently_checked_and_clipped'
    assert readout['physical_target_inside_source_interval'] and readout['parent_target_mass_enclosure_pass']
    assert readout['lower_bound_gain_mm3']>=-1e-8 and readout['gap_after_mm3']>=-1e-8
    raw=(tmp_path/readout['certified_geometry']['path']).read_bytes()
    assert hashlib.sha256(raw).hexdigest()==readout['certified_geometry']['sha256']
    sparse=json.loads(raw);assert 'history' not in sparse and sparse['geometry']['geometry_digest']==readout['checker_geometry_digest']
    assert len(raw)<2*1024**2
    assert len(json.dumps(row,allow_nan=False))<128*1024
    assert abs(sum(h['evacuated_volume']['delta_evacuated_source_volume_mm3'] for h in history)-readout['source_evacuated_mm3'])<1e-8
