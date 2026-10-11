"""Generated native complementary cuts and independent proof rejection."""
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
import copy, hashlib, importlib.util, sys

import numpy as np
import pytest

from resectionlab.core import freeze_json, thaw_json
from resectionlab.geometry import AccessWindow, ToolGeometry, GeometryScene
from resectionlab.native_cut_union import CutUnionSpec, PreviewCutCoverage, MODE
from resectionlab.native_resection import NativeResectionConfig, NativeResectionEngine
from resectionlab.evaluation import independent_check_native_history

HERE = Path(__file__).resolve().parents[1]


def config(*, enabled=True, spec=None, affine=None):
    material = np.zeros((5,5,7),bool); material[2,2,3] = True
    matrix = np.eye(4) if affine is None else affine
    point = lambda p: matrix[:3,:3] @ p + matrix[:3,3]
    axis = matrix[:3,2]/np.linalg.norm(matrix[:3,2])
    return NativeResectionConfig(material, material.astype(np.int16), matrix,
        AccessWindow(point([2,2,1.5]),axis,3.),
        (ToolGeometry('generated-cut',.65,.05,5.,30.,2.),), 'generated-union',
        'explicit one-cell generated support', cut_union=(spec or CutUnionSpec()) if enabled else None)


def stroke(engine, x, z=3.6, mode='aspirate', preview=False):
    matrix=engine.config.affine
    point=lambda p: matrix[:3,:3] @ p + matrix[:3,3]
    call=engine.preview_stroke if preview else engine.execute_stroke
    return call('generated-cut',point([x,2,z]),entry_mm=point([x,2,1.5]),interaction_mode=mode)


def audit(engine, *, history=None, declared=True):
    cfg=engine.config
    case=SimpleNamespace(mri=np.zeros(cfg.tissue_mask.shape), affine=cfg.affine,
        frame='RAS+',semantic_hash=cfg.source_hash)
    return independent_check_native_history(case,cfg.tools,engine.history if history is None else history,
        tissue_mask=cfg.tissue_mask,access=cfg.access,hard_exclusion=cfg.hard_exclusion,
        cut_union_spec=cfg.cut_union.to_record() if declared and cfg.cut_union is not None else None)


@pytest.mark.parametrize('immutable',[False,True])
def test_complementary_executed_sweeps_remove_only_after_complete_certificate(immutable):
    baseline=NativeResectionEngine(config(enabled=False),immutable_state=immutable)
    assert not stroke(baseline,1.65).feasible and not baseline.history
    engine=NativeResectionEngine(config(),immutable_state=immutable)
    initial=engine.state_hash; first=stroke(engine,1.65)
    assert first.feasible and first.reason=='NATIVE_CUT_COVERAGE_PROGRESS'
    assert first.removed_volume_mm3==0 and engine.remaining_mask[2,2,3]
    assert engine.state_hash!=initial and 0<engine._cut_union_ledger[(2,2,3)]<engine.config.cut_union.full_mask
    assert first.to_history_record()['cut_union']['coverage_only_progress']
    assert audit(engine).feasible
    assert not audit(engine,declared=False).feasible
    second=stroke(engine,2.35)
    assert second.feasible and second.removed_volume_mm3==1.
    assert engine.removed_mask[2,2,3] and not engine._cut_union_ledger
    assert second.native_footprint==MODE and audit(engine).feasible
    # All whole-cell gain is real certificate union, not either capsule alone.
    assert first.to_history_record()['cut_union']['new_leaf_count']<512
    assert second.to_history_record()['cut_union']['new_leaf_count']<512
    assert sum(row['cut_union']['new_leaf_count'] for row in engine.history)==512
    replay=NativeResectionEngine(config(),immutable_state=immutable)
    stroke(replay,1.65);stroke(replay,2.35)
    assert replay.history==engine.history and replay.state_hash==engine.state_hash


def test_preview_clone_reset_probe_stale_and_forged_proofs_preserve_ownership():
    engine=NativeResectionEngine(config(),immutable_state=True)
    initial=engine.state_hash; candidate=stroke(engine,1.65,preview=True)
    assert candidate.feasible and engine.state_hash==initial and not engine._cut_union_ledger
    clone=engine.clone();clone.commit_preview(candidate)
    assert clone._cut_union_ledger and not engine._cut_union_ledger
    engine.commit_preview(candidate); prior=engine._cut_union_digest
    with pytest.raises(ValueError):engine.commit_preview(candidate)
    probe=stroke(engine,2.,z=1.85,mode='probe')
    assert probe.feasible and engine._cut_union_digest==prior and audit(engine).feasible
    assert all(not m['cut_union']['cells'] for m in engine.history[-1]['microsteps'])
    child=engine.clone();second=stroke(child,2.35,preview=True)
    record=thaw_json(second._cut_union_record);record['new_leaf_count']+=1
    object.__setattr__(second,'_cut_union_record',freeze_json(record))
    before=child.state_hash
    with pytest.raises(ValueError,match='changed'):child.commit_preview(second)
    assert child.state_hash==before and child._cut_union_digest==prior
    with pytest.raises(AttributeError):engine._cut_union_ledger={}
    with pytest.raises(TypeError):engine._cut_union_ledger[(2,2,3)]=0
    clone.reset();assert clone.state_hash==initial and not clone._cut_union_ledger
    assert engine._cut_union_digest==prior


def test_rejected_future_and_cap_exhaustion_discard_all_temporary_coverage():
    engine=NativeResectionEngine(config(),immutable_state=True);before=engine.state_hash
    failed=stroke(engine,1.65,z=5.)
    assert not failed.feasible and failed.reason=='SHAFT_BLOCKED_BY_REMAINING_NATIVE_TISSUE'
    assert failed.microsteps and any(m.to_dict()['cut_union']['new_leaf_count'] for m in failed.microsteps)
    assert engine.state_hash==before and not engine._cut_union_ledger and not engine.history
    assert failed.cut_union_attempt['new_leaf_count']>0 and not failed.cut_union_attempt['feasible_for_commit']
    bounded=NativeResectionEngine(config(spec=CutUnionSpec(max_leaf_tests_per_preview=1)),immutable_state=True)
    before=bounded.state_hash;r=stroke(bounded,1.65)
    assert not r.feasible and r.reason=='CUMULATIVE_CUT_COVERAGE_BUDGET'
    assert bounded.state_hash==before and not bounded._cut_union_ledger and not bounded.history
    assert r.cut_union_attempt['budget_refused'] and r.cut_union_attempt['leaf_tests']==0


def test_eight_native_corners_in_different_capsules_do_not_certify_interior():
    ledger=PreviewCutCoverage(CutUnionSpec(),{},np.eye(4));cell=np.asarray([[2,2,2]])
    free=np.ones((5,5,5),bool);free[2,2,2]=False
    corners=np.asarray([[x,y,z] for x in (1.5,2.5) for y in (1.5,2.5) for z in (1.5,2.5)])
    assert all(any(np.linalg.norm(c-q)<.26 for q in corners) for c in corners)
    for corner in corners:
        rec=ledger.cover(cell,np.empty((0,3),int),free,corner,corner,.26)
        ledger.finish_step(rec,np.empty((0,3),int))
    assert ledger.new_leaf_count>0 and ledger.cells[(2,2,2)]!=ledger.spec.full_mask
    assert not len(ledger.complete(cell))
    # Accessibility alone cannot add a leaf; no capsule covers the central gap.
    with pytest.raises(ValueError):ledger.finish_step(ledger.empty_step(),cell)


def test_buried_partial_contact_never_banks_cutting_and_invalid_inputs_refuse():
    ledger=PreviewCutCoverage(CutUnionSpec(),{},np.eye(4));cell=np.asarray([[2,2,2]])
    record=ledger.cover(cell,np.empty((0,3),int),np.zeros((5,5,5),bool),
        np.array([1.65,2,1]),np.array([1.65,2,3]),.65)
    assert record['new_leaf_count']==0 and not ledger.cells
    for kwargs in ({'max_depth':True},{'max_depth':4},{'max_tracked_cells':0},
                   {'max_leaf_tests_per_preview':1.5}):
        with pytest.raises(ValueError):CutUnionSpec(**kwargs)
    with pytest.raises(ValueError):ledger.cover(cell,cell,np.ones((5,5,5),bool),
        np.array([np.nan,0,0]),np.ones(3),1.)
    with pytest.raises(TypeError):replace(config(),cut_union={})


def test_independent_checker_rejects_forged_full_leaf_mask():
    engine=NativeResectionEngine(config());stroke(engine,1.65);stroke(engine,2.35)
    assert audit(engine).feasible
    history=copy.deepcopy(engine.history)
    micro=next(m for m in history[0]['microsteps'] if m['cut_union']['cells'])
    cell=micro['cut_union']['cells'][0]
    # The first partial capsule cannot contain every subcell. Merely asserting
    # a full bitmask must fail independent physical containment before digest use.
    cell['added_bits_hex']=format(engine.config.cut_union.full_mask,'x')
    micro['cut_union']['new_leaf_count']=512
    micro['cut_union']['leaf_tests']=512
    assert not audit(engine,history=history).feasible
    mismatch=engine.config.cut_union.to_record();mismatch['max_depth']=2
    cfg=engine.config;case=SimpleNamespace(mri=np.zeros(cfg.tissue_mask.shape),affine=cfg.affine,
        frame='RAS+',semantic_hash=cfg.source_hash)
    assert not independent_check_native_history(case,cfg.tools,engine.history,tissue_mask=cfg.tissue_mask,
        access=cfg.access,cut_union_spec=mismatch).feasible


def test_rotated_reflected_affine_subcells_retain_same_union_removal():
    theta=.37;matrix=np.eye(4)
    matrix[:3,:3]=np.array([[np.cos(theta),-np.sin(theta),0.],
        [np.sin(theta),np.cos(theta),0.],[0.,0.,-1.]])
    matrix[:3,3]=[2.1,-1.3,9.]
    engine=NativeResectionEngine(config(affine=matrix),immutable_state=True)
    a=stroke(engine,1.65);b=stroke(engine,2.35)
    assert a.feasible and a.removed_volume_mm3==0 and b.feasible
    assert len(b.removed_indices_native)==1 and audit(engine).feasible


def test_prior_microstep_shaft_cannot_borrow_union_from_its_own_endpoint():
    tissue=np.zeros((5,5,10),bool);tissue[2,2,5]=True
    tool=ToolGeometry('short-active',.9,.1,10.,20.,.1)
    for step in (.5,.25,.1):
        cfg=NativeResectionConfig(tissue,tissue.astype(int),np.eye(4),
            AccessWindow([2,2,4.49],[0,0,1],2.),(tool,),'temporal','generated',
            max_tip_step_mm=step,cut_union=CutUnionSpec())
        e=NativeResectionEngine(cfg);state=e.state_hash
        result=e.execute_stroke(tool.tool_id,[2,2,5.49])
        assert not result.feasible and result.reason=='SHAFT_BLOCKED_BY_REMAINING_NATIVE_TISSUE'
        assert e.state_hash==state and not e._cut_union_ledger


@pytest.mark.parametrize('immutable',[False,True])
def test_disabled_default_fingerprint_preview_history_and_metrics_match_pinned_baseline(immutable):
    path=HERE/'artifacts/native-cut-union-integration-v1/baseline/native_resection.py'
    assert hashlib.sha256(path.read_bytes()).hexdigest()=='79363aa6156917514d4aa57e328adcd16babff51518c0c4587e6dcaf94a5723d'
    spec=importlib.util.spec_from_file_location('resectionlab._before_cut_union',path)
    old=importlib.util.module_from_spec(spec);sys.modules[spec.name]=old;spec.loader.exec_module(old)
    cfg=config(enabled=False)
    cfg=replace(cfg,tools=(replace(cfg.tools[0],tip_radius_mm=.9),))
    legacy=old.NativeResectionConfig(cfg.tissue_mask,cfg.target_labels,cfg.affine,cfg.access,cfg.tools,
        cfg.source_hash,cfg.tissue_support_provenance)
    assert legacy.fingerprint==cfg.fingerprint
    a=NativeResectionEngine(cfg,immutable_state=immutable)
    b=old.NativeResectionEngine(legacy,immutable_state=immutable)
    for x in (1.65,2.,2.35):
        aa=stroke(a,x);bb=stroke(b,x)
        assert aa.feasible==bb.feasible and aa.reason==bb.reason
        assert a.history==b.history and a.state_hash==b.state_hash and a.metrics()==b.metrics()
    assert not hasattr(a,'_cut_union_ledger')
