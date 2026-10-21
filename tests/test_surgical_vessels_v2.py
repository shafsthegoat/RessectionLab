"""Conservative engineering surrogate checks, not physiological validation."""
from types import SimpleNamespace
import copy
import numpy as np
import pytest
from resectionlab import surgical_vessels_v2 as vessels
from resectionlab.surgical_effects import tick_fields


def profile(**changes):
    p = {'model': vessels.MODEL, 'pressurePa': 1000., 'reservoirMm3': 100.,
         'refillConductanceMm3PerPaS': .001, 'leakConductanceMmPerPaS': .0001,
         'mechanicalDoseNs': 2., 'thermalDoseCs': 200., 'coagulationDose': 2., 'clotRatePerSecond': .02}
    return dict(p, **changes)


def engine(n=1, p=None, basis=None):
    # Fixed physical cross-section, one or many voxels per wound face.
    labels = np.zeros((4, n+2, n+2), np.uint16); labels[1:3, 1:-1, 1:-1] = 1
    affine = np.eye(4); affine[:3,:3] = np.diag([1., 1/n, 1/n]) if basis is None else basis
    scene = SimpleNamespace(labels=labels, affine=affine, shape=labels.shape,
        voxel_mm3=abs(np.linalg.det(affine[:3,:3])), properties={'tissuePhysicsV2': {'vessel': p or profile()}})
    state = {'sources': [], 'fluid': {}, 'fluidAddedMm3': 0., 'fluidEscapedMm3': 0., 'debris': {}}
    e = SimpleNamespace(scene=scene, state=state, by_label={1:{'material':'vessel'}}, occupancy=lambda:labels.ravel())
    vessels.initialize(e)
    return e


def edges(e):
    n=e.scene.shape[1]-2
    return [(int(np.ravel_multi_index((1,y,z),e.scene.shape)), int(np.ravel_multi_index((2,y,z),e.scene.shape)))
            for y in range(1,n+1) for z in range(1,n+1)]


def advance(e, dt):
    vessels.tick(e, dt); tick_fields(e, dt)


def test_no_spontaneous_bleeding_and_repeated_faces_are_idempotent():
    e=engine(); advance(e, 5.)
    assert not e.state['sources'] and e.state['fluidAddedMm3']==0.
    vessels.damage(e,edges(e)); vessels.damage(e,edges(e)); advance(e,.2)
    assert len(e.state['sources'])==1
    assert vessels.metrics(e)['woundAreaMm2']==pytest.approx(1.)
    assert vessels.metrics(e)['generatedMm3']>0.
    assert not next(iter(e.state['vesselsV2']['faces'].values()))['patent']


def test_affine_face_area_uses_cross_product_and_is_rotation_invariant():
    basis=np.array([[2.,.5,0.],[0.,3.,.4],[0.,0.,4.]])
    e=engine(basis=basis); vessels.damage(e,edges(e))
    expected=np.linalg.norm(np.cross(basis[:,1],basis[:,2]))
    assert vessels.metrics(e)['woundAreaMm2']==pytest.approx(expected)
    rotation=np.array([[0.,-1.,0.],[1.,0.,0.],[0.,0.,1.]])
    r=engine(basis=rotation@basis); vessels.damage(r,edges(r))
    assert vessels.metrics(r)['woundAreaMm2']==pytest.approx(expected)


def test_grid_refinement_preserves_area_flow_and_one_cluster():
    coarse=engine(1); fine=engine(4)
    for e in (coarse,fine): vessels.damage(e,edges(e)); advance(e,3.)
    for e in (coarse,fine):
        assert vessels.metrics(e)['woundAreaMm2']==pytest.approx(1.)
        assert vessels.metrics(e)['woundCount']==1
    assert vessels.metrics(coarse)['generatedMm3']==pytest.approx(vessels.metrics(fine)['generatedMm3'],rel=1e-8)


def test_pressure_area_and_progressive_dose_sensitivity():
    values=[]
    for pressure,scale in ((500.,1.),(1000.,1.),(1000.,2.)):
        e=engine(p=profile(pressurePa=pressure),basis=np.diag([1.,scale,1.]))
        vessels.damage(e,edges(e)); advance(e,.1); values.append(vessels.metrics(e)['generatedMm3'])
    assert values[0]<values[1]<values[2]
    e=engine(); cells=[edges(e)[0][0]]
    vessels.contact(e,cells,dt=.5,force_n=2.); advance(e,.1)
    assert not e.state['sources']
    vessels.contact(e,cells,dt=1.,force_n=2.); advance(e,.1)
    partial=vessels.metrics(e)['generatedMm3']; assert partial>0
    assert next(iter(e.state['vesselsV2']['faces'].values()))['damage']==pytest.approx(.5)
    vessels.contact(e,cells,dt=.5,force_n=2.); advance(e,.1)
    assert vessels.metrics(e)['generatedMm3']-partial > partial


def test_thermal_dose_requires_supplied_temperature_and_accumulates():
    e=engine(); cells=[edges(e)[0][0]]
    vessels.contact(e,cells,1.,0.); assert not e.state['sources']
    vessels.contact(e,cells,1.,0.,temperature=43.); assert not e.state['sources']
    vessels.contact(e,cells,1.,0.,temperature=243.); assert not e.state['sources']
    vessels.contact(e,cells,.5,0.,temperature=243.); assert e.state['sources']


def test_time_partition_and_conservation_with_compliant_refill():
    one=engine(); many=engine()
    for e in (one,many): vessels.damage(e,edges(e))
    advance(one,7.)
    for _ in range(70): advance(many,.1)
    for key in ('generatedMm3','vascularRemainingMm3','refilledMm3','pressurePa'):
        assert vessels.metrics(one)[key]==pytest.approx(vessels.metrics(many)[key],rel=2e-8,abs=2e-9)
    for e in (one,many):
        m=vessels.metrics(e)
        assert m['conservationResidualMm3']==pytest.approx(0.,abs=1e-10)
        assert m['generatedMm3']==pytest.approx(m['emittedMm3'])
        assert m['emittedMm3']==pytest.approx(sum(e.state['fluid'].values())+e.state['fluidEscapedMm3']+m['blockedOutletReservoirMm3'])


def test_finite_reservoir_and_blocked_outlet_are_conservative():
    e=engine(p=profile(reservoirMm3=.01,refillConductanceMm3PerPaS=0.,clotRatePerSecond=0.))
    vessels.damage(e,edges(e)); e.occupancy=lambda: np.ones(e.scene.labels.size,np.uint16)
    advance(e,100.)
    m=vessels.metrics(e)
    assert 0.<=m['generatedMm3']<=.01
    assert m['generatedMm3']==pytest.approx(m['blockedOutletReservoirMm3'])
    assert m['vascularRemainingMm3']==pytest.approx(0.,abs=1e-8)
    assert m['conservationResidualMm3']==pytest.approx(0.,abs=1e-12)


def test_coagulation_progressive_persistent_seal_does_not_restore_patency():
    e=engine(p=profile(clotRatePerSecond=0.)); vessels.damage(e,edges(e))
    advance(e,.1); before=vessels.metrics(e)['generatedMm3']; cells=edges(e)[0]
    vessels.coagulate(e,cells,power=1.,dt=1.); advance(e,.1)
    after=vessels.metrics(e)['generatedMm3']; assert 0<after-before<before
    vessels.coagulate(e,cells,power=1.,dt=1.); advance(e,5.)
    assert vessels.metrics(e)['generatedMm3']==after
    vessels.damage(e,edges(e)); advance(e,5.)
    assert vessels.metrics(e)['generatedMm3']==after
    face=next(iter(e.state['vesselsV2']['faces'].values()))
    assert face['damage']==1. and not face['patent'] and face['clot']==1.


def test_new_adjacent_wound_merges_outlet_without_losing_existing_blood():
    e=engine(3); es=edges(e)
    vessels.damage(e,[es[0],es[-1]]); advance(e,1.)
    assert vessels.metrics(e)['woundCount']==2
    before=vessels.metrics(e)['emittedMm3']
    vessels.damage(e,es); assert vessels.metrics(e)['woundCount']==1
    assert vessels.metrics(e)['emittedMm3']==pytest.approx(before)
    advance(e,1.)
    assert vessels.metrics(e)['generatedMm3']==pytest.approx(vessels.metrics(e)['emittedMm3'])


def test_missing_or_duplicate_field_handoff_detected():
    e=engine(); vessels.damage(e,edges(e)); vessels.tick(e,.2)
    with pytest.raises(ValueError,match='MATCHING_FIELD'): vessels.tick(e,.2)
    tick_fields(e,.2); tick_fields(e,.2)
    with pytest.raises(ValueError,match='MATCHING_FIELD'): vessels.tick(e,.2)


@pytest.mark.parametrize('change',[{'pressurePa':float('nan')},{'pressurePa':True},{'reservoirMm3':0.},
    {'clotRatePerSecond':-1.},{'mechanicalDoseNs':0.},{'unexpected':1.},{'model':'guess'}])
def test_malformed_profile_rejected(change):
    with pytest.raises(ValueError): vessels.validate(profile(**change))


def test_profile_requires_every_assumption_and_no_profile_is_noop():
    p=profile(); del p['pressurePa']
    with pytest.raises(ValueError): vessels.validate(p)
    e=engine(); del e.scene.properties['tissuePhysicsV2']; del e.state['vesselsV2']
    assert vessels.initialize(e) is None
    vessels.tick(e,1.); assert not e.state['sources']
