"""Actual public engine transitions for the versioned unilateral boundary."""
import copy

import numpy as np
import pytest

from resectionlab.surgical_scene import SurgicalScene
from resectionlab.surgical_engine import SurgicalEngine
from resectionlab.surgical_physics_v2 import default_profile
from resectionlab.surgical_continuum import research_profile,PROPERTY as CONTINUUM
from resectionlab.surgical_continuum_solve import profile as solve_profile
from resectionlab.surgical_retractor_runtime import profile,PROPERTY
from resectionlab import surgical_mechanical_exposure as exposure

H=1/120


def source(mode='nonadhesive-frictionless-face', *, track_exposure=False, two_regions=False):
    labels=np.zeros((15,15,8),np.uint16);labels[4:11,4:11,1:4]=1
    physics=default_profile();physics['brain']['radiusMm']=2.
    regions = [{'id': 1, 'name': 'Generated supported slab', 'material': 'soft_tissue'}]
    if two_regions:
        labels[7:11, 4:11, 1:4] = 2
        regions.append({'id': 2, 'name': 'Second slab region', 'material': 'soft_tissue'})
    props = {'tissuePhysicsV2': physics}
    if track_exposure:
        props[exposure.PROPERTY] = exposure.profile()
    if mode != 'lattice':
        material = {'shearPa': 1000., 'poissonRatio': .3, 'relaxationShearPa': [500.],
                    'source': 'Generated engineering control; not a human fit'}
        props.update({
            CONTINUUM: research_profile({str(r['id']): copy.deepcopy(material) for r in regions},
                relaxation_seconds=[.5]),
            'continuumLinearSolveV1': solve_profile(tensor_contraction='blas-contiguous'),
            PROPERTY: profile(mode),
        })
    return SurgicalScene(labels,np.eye(4),regions,{'kind':'generated numerical contact control; not patient anatomy'}, props)


def action(operation,position=None,**extra):
    return {'slot':'left','operation':operation,'duration':H,
        **({'pose':{'position':position,'quaternion':[0.,0.,0.,1.]}} if position is not None else {}),**extra}


def accepted(e,a):
    result=e.step(a)
    assert result['status']=='accepted',result
    return result


def prepare(mode='nonadhesive-frictionless-face', *, track_exposure=False, two_regions=False):
    e=SurgicalEngine(source(mode, track_exposure=track_exposure, two_regions=two_regions))
    for a in [action('configure',mechanicsRevision=2),action('exchange',instrument='brain_retractor'),
        action('place',[7.,2.,1.9],parameters={'radiusMm':1.,'depthMm':2.}),
        action('hold',[7.,3.1,1.9],duration=4*H)]:accepted(e,a)
    return e


def test_native_nonadhesive_push_hold_separate_slide_recontact_release_and_cold_replay():
    e=prepare();original=e.scene.labels.copy();accepted(e,action('move',[7.,3.16,1.9]))
    pushed=copy.deepcopy(e.state[PROPERTY]['last'])
    assert pushed['summedNormalForceN']>0 and pushed['forceOnTissueN'][1]>0
    assert pushed['forceOnTissueN'][0]==pushed['forceOnTissueN'][2]==0
    accepted(e,action('inspect',duration=6*H));assert 0<e.state[PROPERTY]['last']['summedNormalForceN']<pushed['summedNormalForceN']
    accepted(e,action('move',[7.,2.9,1.9]))
    separated=copy.deepcopy(e.state['contact'])
    assert e.state[PROPERTY]['last']['summedNormalForceN']==0
    assert e.state[PROPERTY]['last']['contactSourceIds']==[]
    for x in (7.3,7.6,7.9):accepted(e,action('move',[x,2.9,1.9]))
    assert e.state['contact']==separated
    assert e.state[PROPERTY]['last']['summedNormalForceN']==0
    accepted(e,action('move',[7.9,3.1,1.9]))
    accepted(e,action('move',[7.9,3.14,1.9]))
    assert e.state[PROPERTY]['last']['summedNormalForceN']>0
    assert e.step(action('withdraw'))['reasonCode']=='RELEASE_BEFORE_WITHDRAWAL'
    accepted(e,action('release'));accepted(e,action('withdraw',duration=.1))
    accepted(e,action('inspect',duration=.1))
    assert e.state[PROPERTY]['last']['summedNormalForceN']==0
    assert np.linalg.norm(list(e.state[CONTINUUM]['memoryDisplacementMm'][0].values()))>0
    assert e.state[PROPERTY]['maxCumulativeEnergyResidualNmm']<1e-10
    assert e.state[CONTINUUM]['maxFreeResidualN']<1e-10
    assert e.state['removed']==[] and np.array_equal(e.scene.labels,original)
    record=e.snapshot();restored=SurgicalEngine.restore(e.scene,record)
    assert restored.state_id==e.state_id and restored.state==e.state


def test_native_fractional_pose_clock_owns_queue_and_replays_it():
    e=prepare();accepted(e,action('move',[7.,3.12,1.9],duration=H/2))
    pending=e.snapshot();assert e.state[PROPERTY]['queue']
    assert SurgicalEngine.restore(e.scene,pending).state_id==e.state_id
    accepted(e,action('move',[7.,3.14,1.9],duration=H/2))
    direct=prepare();accepted(direct,action('move',[7.,3.14,1.9]))
    np.testing.assert_allclose(list(e.state[CONTINUUM]['cornerDisplacementMm'].values()),list(direct.state[CONTINUUM]['cornerDisplacementMm'].values()),rtol=0,atol=1e-12)
    assert not e.state[PROPERTY]['queue'] and e.state['deformationModel']['remainderSeconds']==0
    assert e.state[PROPERTY]['toolWorkNmm']==pytest.approx(direct.state[PROPERTY]['toolWorkNmm'],abs=1e-14)


def test_unsupported_loaded_rotation_refuses_atomically_preview_and_step():
    e=prepare();accepted(e,action('move',[7.,3.14,1.9]));before=e.snapshot()
    a=action('move',[7.,3.14,1.9]);a['pose']['quaternion']=[0.,0.,.01,np.sqrt(1-.01**2)]
    preview=e.preview(a,include_view=False)
    assert preview['status']=='rejected' and preview['reasonCode']=='RETRACTOR_CONTACT_ROTATION_UNQUALIFIED'
    assert e.snapshot()==before
    assert e.step(a)['status']=='rejected' and e.snapshot()==before


def test_bonded_control_preserves_default_semantics_with_same_finite_geometry():
    e=prepare('bonded-centroid-control');accepted(e,action('move',[7.,3.14,1.9]));accepted(e,action('move',[7.,2.9,1.9]))
    assert PROPERTY not in e.state
    assert e.state[CONTINUUM]['handleReactionsN']['left'][1]<0


def test_profile_cannot_silently_fall_back_to_legacy():
    s=source();p=copy.deepcopy(s.properties);p.pop(CONTINUUM)
    with pytest.raises(ValueError,match='RETRACTOR_SURFACE_REQUIRES_PROPORTIONAL_CONTINUUM'):
        SurgicalScene(s.labels,s.affine,s.regions,s.source,p)
    p=copy.deepcopy(s.properties);p[PROPERTY]['halfThicknessMm']=.3
    with pytest.raises(ValueError,match='COMPLETE_RETRACTOR_SURFACE_PROFILE_REQUIRED'):
        SurgicalScene(s.labels,s.affine,s.regions,s.source,p)


def test_sampled_touch_precedes_load_and_separation_stops_force_exposure():
    engine = prepare(track_exposure=True)
    row = lambda: exposure.summarize(engine)['regions']['1']
    unloaded = row()
    assert unloaded['retractorTouchCellSeconds'] > 0
    assert unloaded['retractorLoadedCellSeconds'] == unloaded['retractorForceTimeNs'] == 0
    accepted(engine, action('move', [7., 3.16, 1.9]))
    pushed = row()
    assert pushed['retractorForcePeakN'] == pytest.approx(engine.state[PROPERTY]['last']['summedNormalForceN'])
    accepted(engine, action('inspect', duration=6 * H))
    sustained = row()
    for key in ('retractorTouchCellSeconds', 'retractorLoadedCellSeconds', 'retractorForceTimeNs',
                'strainSquaredVolumeTimeMm3Seconds', 'stressSquaredVolumeTimePa2Mm3Seconds'):
        assert sustained[key] > pushed[key]
    assert sustained['peakStressNormPa'] == pushed['peakStressNormPa'] > 0
    accepted(engine, action('move', [7., 2.9, 1.9]))
    separated = row()
    accepted(engine, action('inspect', duration=6 * H))
    assert row()['retractorForceTimeNs'] == separated['retractorForceTimeNs']
    assert row()['retractorLoadedCellSeconds'] == separated['retractorLoadedCellSeconds']
    assert row()['contactCellSeconds'] == separated['contactCellSeconds']
    assert engine.observe()['physics']['mechanicalExposure'] == exposure.summarize(engine)
    restored = SurgicalEngine.restore(engine.scene, engine.snapshot())
    assert exposure.summarize(restored) == exposure.summarize(engine)


@pytest.mark.parametrize('mode,integral,peak,unavailable', [
    ('bonded-centroid-control', 'strainSquaredVolumeTimeMm3Seconds', 'peakStrainNorm', 'peakNeighborDisplacementRatio'),
    ('lattice', 'neighborDisplacementRatioSquaredBondSeconds', 'peakNeighborDisplacementRatio', 'peakStressNormPa'),
])
def test_bonded_load_duration_accumulates_separate_model_specific_exposure(mode, integral, peak, unavailable):
    engine = prepare(mode, track_exposure=True)
    accepted(engine, action('move', [7., 3.16, 1.9]))
    pushed = exposure.summarize(engine)['regions']['1']
    accepted(engine, action('inspect', duration=6 * H))
    sustained = exposure.summarize(engine)['regions']['1']
    assert sustained[integral] > pushed[integral] > 0
    assert sustained[peak] >= pushed[peak] > 0
    assert sustained['retractorForceTimeNs'] > pushed['retractorForceTimeNs'] > 0
    assert sustained[unavailable] is None


def test_boundary_spanning_contact_retains_exposure_without_attributing_the_resultant():
    engine = prepare(track_exposure=True, two_regions=True)
    accepted(engine, action('move', [7., 3.16, 1.9]))
    assert engine.state[PROPERTY]['last']['summedNormalForceN'] > 0
    rows = exposure.summarize(engine)['regions']
    for row in rows.values():
        assert row['retractorTouchCellSeconds'] > 0
        assert row['retractorLoadedCellSeconds'] > 0
        assert row['retractorForcePeakN'] is None
        assert row['retractorForceTimeNs'] is None
        assert row['strainSquaredVolumeTimeMm3Seconds'] > 0
