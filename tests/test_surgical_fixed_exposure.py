"""Frozen-ray v2 controls and exact pre-refactor v1 execution parity."""
import copy
import json
from pathlib import Path
import subprocess
import sys
import types

import numpy as np
import pytest
from scipy import ndimage as ndi

from resectionlab import surgical_exposure as exposure
from resectionlab.core import semantic_digest
from resectionlab.imaging import save_case
from resectionlab.surgical_engine import SurgicalEngine
from resectionlab.surgical_planning_observation import physical_state_hash
from test_surgical_exposure import intact, controls, completed, Cancelled
from test_source_exposure_audit import evidence, reseal

LEGACY_COMMIT='45eadded570bca2d71790b7bd5611628b9b1bded'


@pytest.fixture(scope='module')
def legacy():
    path='src/resectionlab/surgical_exposure.py'
    code=subprocess.check_output(['git','show',LEGACY_COMMIT+':'+path])
    assert b'FIXED_SCHEMA' not in code
    name='resectionlab._fixed_exposure_legacy_45eadded'
    module=types.ModuleType(name); module.__package__='resectionlab'
    module.__file__=str(Path(__file__).resolve().parents[1]/path)
    sys.modules[name]=module
    try:
        exec(compile(code,module.__file__,'exec'),module.__dict__)
        yield module
    finally: sys.modules.pop(name,None)


@pytest.fixture(scope='module')
def fixed(intact,controls,completed):
    engine,native=intact
    chosen=completed[1]['publicPlan']['attempts'][0]['route']
    ray={key:copy.deepcopy(chosen[key]) for key in ('surfaceMm','sourceTargetMm')}
    common={**controls,'maxNonTargetMm3':200.,'maxActions':128,'maxSeconds':180.,'maxCandidates':1}
    rows=[]
    for width in (0.,.3,1.):
        raw={**common,'lateralSweepMm':width}; before=engine.snapshot()
        selected,proof=exposure.prepare(engine,raw,native_collection=native.root,source_ray=ray)
        assert engine.snapshot()==before
        rows.append((raw,selected,proof))
    return ray,rows


def saved_candidate(evidence,selected,name):
    artifacts=copy.deepcopy(evidence['artifacts']); record=artifacts['surgical_toolbox']
    if selected is None: record.update(branchId='source',position=len(evidence['baseline'].actions))
    else:
        record['branches']['prepared']=selected.snapshot(); record['timeline']['prepared']=copy.deepcopy(selected.history)
        record.update(branchId='prepared',position=len(selected.actions))
    path=evidence['root']/name; save_case(evidence['case'],path,artifacts=artifacts)
    return path


@pytest.mark.parametrize('width',[0.,.3,1.])
def test_default_v1_public_plan_receipts_hashes_equal_frozen_pre_refactor(intact,controls,legacy,width):
    observation=exposure.project_exposure(intact[0])
    assert legacy.PlanningObservation is exposure.PlanningObservation
    raw={**controls,'maxNonTargetMm3':200.,'maxActions':128,'maxSeconds':180.,'lateralSweepMm':width}
    before=legacy.plan_observed(legacy.ExposureObservation(observation.baseline),raw)
    after=exposure.plan_observed(observation,raw)
    assert semantic_digest(before)==semantic_digest(after)
    assert json.dumps(before,sort_keys=True,separators=(',',':'))==json.dumps(after,sort_keys=True,separators=(',',':'))
    assert after['schema']==exposure.PUBLIC_SCHEMA and 'sourceRay' not in after


def test_default_v1_entire_native_proof_and_endpoint_equal_pre_refactor(intact,controls,legacy):
    engine,native=intact; before=engine.snapshot()
    old_selected,old=legacy.prepare(engine,controls,native_collection=native.root)
    selected,new=exposure.prepare(engine,controls,native_collection=native.root)
    assert old_selected.snapshot()==selected.snapshot()
    assert semantic_digest(old)==semantic_digest(new) and new['schema']==exposure.SCHEMA
    assert engine.snapshot()==before


def test_fixed_source_points_axis_roll_and_central_path_are_exact_across_widths(intact,fixed):
    ray,rows=fixed; reference=rows[0][2]['publicPlan']; central=reference['attempts'][0]['route']
    for raw,selected,proof in rows:
        public=proof['publicPlan']; assert proof['schema']==exposure.FIXED_SCHEMA
        assert public['schema']==exposure.FIXED_PUBLIC_SCHEMA and public['sourceRay']==ray
        assert public['publicObservation']==reference['publicObservation']
        assert proof['sourceSnapshotHash']==rows[0][2]['sourceSnapshotHash']
        assert len(public['attempts'])==1
        attempt=public['attempts'][0]; route=attempt['route']
        for key in ('surfaceMm','sourceTargetMm','axisRasUnit','quaternion','outsideMm','endpointMm'):
            assert route[key]==central[key]
        assert route['paths'][0]==central['paths'][0]
        assert len(route['paths'])==(1 if raw['lateralSweepMm']==0. else 5)
        model=intact[0].clone()
        for action,receipt,state_hash in zip(attempt['actionProgram'],attempt['actionReceipts'],attempt['physicalStateHashes'],strict=True):
            result=model.step(copy.deepcopy(action))
            assert exposure._receipt(result)==receipt and physical_state_hash(model)==state_hash
            assert action['parameters']=={'radiusMm':1.5,'power':1.,'duty':1.}
        assert attempt['summary']['additionalTargetMm3']==0.
        assert attempt['summary']['additionalNonTargetMm3']<=raw['maxNonTargetMm3']
        if selected is not None: assert model.snapshot()==selected.snapshot()
        else: assert public['decision']['operation']=='STOP'


def test_independent_auditor_accepts_every_frozen_width_native_result(evidence,fixed):
    for index,(_,selected,proof) in enumerate(fixed[1]):
        path=saved_candidate(evidence,selected,f'fixed-width-{index}.ressectionlab')
        result=evidence['run'](proof,source=path)
        assert result['passed'] and result['sourceInputsUnchanged']


@pytest.mark.parametrize('change',['non-center','background','out-of-bounds','interior-brain','interior-target',
    'wrong-owner','missing','extra','tuple','boolean','nonfinite'])
def test_invalid_fixed_source_ray_refuses_instead_of_substituting(intact,fixed,controls,monkeypatch,change):
    engine,_=intact; ray=copy.deepcopy(fixed[0])
    if change=='non-center': ray['surfaceMm'][0]+=.1
    elif change=='background': ray['surfaceMm']=[0.,0.,0.]
    elif change=='out-of-bounds': ray['surfaceMm']=[1.e6,0.,0.]
    elif change=='interior-brain':
        axis=fixed[1][0][2]['publicPlan']['attempts'][0]['route']['axisRasUnit']
        ray['surfaceMm']=(np.asarray(ray['surfaceMm'])+axis).tolist()
    elif change=='interior-target':
        ids=[r['id'] for r in engine.scene.regions if r.get('target')]
        cells=np.flatnonzero(ndi.binary_erosion(np.isin(engine.scene.labels,ids)).ravel())
        ray['sourceTargetMm']=engine.scene.points(cells[:1])[0].tolist()
    elif change=='wrong-owner': ray['surfaceMm']=ray['sourceTargetMm'][:]
    elif change=='missing': ray.pop('sourceTargetMm')
    elif change=='extra': ray['quaternion']=[0.,0.,0.,1.]
    elif change=='tuple': ray['surfaceMm']=tuple(ray['surfaceMm'])
    elif change=='boolean': ray['surfaceMm'][0]=True
    else: ray['sourceTargetMm'][0]=float('nan')
    def forbidden(*args,**kwargs): pytest.fail('Fixed ray attempted a replacement shortlist')
    monkeypatch.setattr(exposure,'_shortlist',forbidden)
    with pytest.raises(ValueError):
        exposure.plan_observed(exposure.project_exposure(engine),controls,source_ray=ray)


def test_previously_removed_surface_cannot_be_reused_as_retained_fixed_entry(completed,fixed,controls):
    engine=completed[0]; before=engine.snapshot()
    with pytest.raises(ValueError,match='retained exterior-connected'):
        exposure.plan_observed(exposure.project_exposure(engine),controls,source_ray=fixed[0])
    assert engine.snapshot()==before


def test_empty_width_budget_cell_remains_stop_without_ray_replacement(evidence,fixed,monkeypatch):
    raw={**fixed[1][0][0],'maxNonTargetMm3':.1,'lateralSweepMm':1.}
    def forbidden(*args,**kwargs): pytest.fail('Empty fixed cell silently replaced its ray')
    monkeypatch.setattr(exposure,'_shortlist',forbidden)
    selected,proof=exposure.prepare(evidence['baseline'],raw,native_collection=evidence['native'].root,source_ray=fixed[0])
    assert selected is None and proof['publicPlan']['attempts']==[]
    assert proof['publicPlan']['sourceRay']==fixed[0] and proof['publicPlan']['decision']['operation']=='STOP'
    path=saved_candidate(evidence,None,'empty-fixed-source.ressectionlab')
    assert evidence['run'](proof,source=path)['passed']
    bad=copy.deepcopy(proof); bad['publicPlan']['sourceRay']['surfaceMm']=[0.,0.,0.]
    with pytest.raises(ValueError): evidence['run'](reseal(bad),source=path)


@pytest.mark.parametrize('change',['public-ray','route-surface','route-target','v1-downgrade','duplicate-attempt'])
def test_native_wrapper_rejects_resealed_fixed_ray_producer_forgery(evidence,fixed,monkeypatch,change):
    ray,rows=fixed; raw,_,proof=rows[0]; public=copy.deepcopy(proof['publicPlan'])
    if change=='public-ray': public['sourceRay']['surfaceMm'][0]+=1.
    elif change=='route-surface': public['attempts'][0]['route']['surfaceMm'][0]+=1.
    elif change=='route-target': public['attempts'][0]['route']['sourceTargetMm'][0]+=1.
    elif change=='v1-downgrade': public['schema']=exposure.PUBLIC_SCHEMA; public.pop('sourceRay')
    else: public['attempts'].append(copy.deepcopy(public['attempts'][0]))
    public['publicSeal']=semantic_digest({k:v for k,v in public.items() if k!='publicSeal'})
    monkeypatch.setattr(exposure,'plan_observed',lambda *args,**kwargs:copy.deepcopy(public))
    with pytest.raises(ValueError):
        exposure.prepare(evidence['baseline'],raw,native_collection=evidence['native'].root,source_ray=ray)


@pytest.mark.parametrize('change',['ray','route','mixed-schema','v1-downgrade','missing-ray','selection','duplicate-attempt'])
def test_independent_auditor_rejects_resealed_fixed_binding_forgeries(evidence,fixed,change):
    _,rows=fixed; _,selected,proof=rows[0]; bad=copy.deepcopy(proof); public=bad['publicPlan']
    if change=='ray': public['sourceRay']['surfaceMm'][0]+=1.
    elif change=='route': public['attempts'][0]['route']['sourceTargetMm'][0]+=1.
    elif change=='mixed-schema': bad['schema']=exposure.SCHEMA
    elif change=='v1-downgrade': bad['schema']=exposure.SCHEMA; public['schema']=exposure.PUBLIC_SCHEMA
    elif change=='missing-ray': public.pop('sourceRay')
    elif change=='selection': public['decision']['selectedIndex']=None
    else: public['attempts'].append(copy.deepcopy(public['attempts'][0]))
    path=saved_candidate(evidence,selected,'fixed-forgery-native.ressectionlab')
    with pytest.raises(ValueError): evidence['run'](reseal(bad),source=path)


def test_callback_and_private_fields_cannot_change_frozen_source_ray(evidence,fixed):
    ray,rows=fixed; raw,selected,proof=rows[0]
    engine=evidence['baseline'].clone(); engine._private_evaluation_truth=np.ones(engine.scene.shape,bool)
    engine._private_evaluator=lambda:pytest.fail('Private reference affected fixed proposal')
    mutable_ray=copy.deepcopy(ray); mutable_controls=copy.deepcopy(raw)
    def callback():
        mutable_ray['surfaceMm'][:]=[0.,0.,0.]; mutable_ray['sourceTargetMm'][:]=[0.,0.,0.]
        mutable_controls['lateralSweepMm']=1.; mutable_controls['maxActions']=0
    actual,value=exposure.prepare(engine,mutable_controls,native_collection=evidence['native'].root,
                                  source_ray=mutable_ray,check=callback)
    assert value==proof and actual.snapshot()==selected.snapshot()


def test_cancellation_of_fixed_native_prefix_cannot_publish_an_endpoint(evidence,fixed,monkeypatch):
    engine=evidence['baseline']; before=engine.snapshot(); original=SurgicalEngine.step
    def interrupted(model,action,**kwargs):
        if type(model) is SurgicalEngine and action.get('operation')=='apply': raise Cancelled('fixed native cancelled')
        return original(model,action,**kwargs)
    monkeypatch.setattr(SurgicalEngine,'step',interrupted)
    with pytest.raises(Cancelled,match='fixed native cancelled'):
        exposure.prepare(engine,fixed[1][0][0],native_collection=evidence['native'].root,source_ray=fixed[0])
    assert engine.snapshot()==before
