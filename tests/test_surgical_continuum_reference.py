"""Exact fixed-topology reduction, without patient-response or clinical claims."""
import copy

import numpy as np
import pytest

from resectionlab import surgical_continuum as c, surgical_deformation_v2 as b
from resectionlab import surgical_continuum_reference as r
from resectionlab.surgical_scene import SurgicalScene
from resectionlab.surgical_engine import SurgicalEngine
from resectionlab.surgical_physics_v2 import default_profile


def specimen(affine=None, *, heterogeneous=False, times=(.2,.7)):
    labels=np.ones((7,7,7),np.uint16)
    materials={'1':{'shearPa':1000.,'poissonRatio':.3,'relaxationShearPa':[500.,200.][:len(times)],'source':'Generated analytical coefficient'}}
    regions=[{'id':1,'name':'Generated block','material':'soft_tissue'}]
    if heterogeneous:
        labels[4:]=2;materials['2']={**materials['1'],'shearPa':2000.,'poissonRatio':.4,
                                  'relaxationShearPa':[1000.,400.][:len(times)]}
        regions.append({'id':2,'name':'Generated inclusion','material':'tumor'})
    properties={'tissuePhysicsV2':default_profile(),c.PROPERTY:c.research_profile(materials,relaxation_seconds=list(times))}
    properties['tissuePhysicsV2']['brain']['radiusMm']=2.
    scene=SurgicalScene(labels,np.eye(4) if affine is None else affine,regions,{'kind':'Generated numerical coupon'},properties)
    e=SurgicalEngine(scene); cells=[int(np.ravel_multi_index(p,scene.shape)) for p in [(3,3,3),(4,3,3)]]
    b.begin_hold(e,'left',cells,scene.points(cells).mean(axis=0))
    return e,cells


def reference(e):
    cells=r.source_domain(e,2.,max_cells=1000)
    return r.build_reference(e,cells,max_dofs=10000)


@pytest.mark.parametrize('affine',[np.eye(4),np.array([[-1.,.2,0.,4.],[0.,1.1,.1,-3.],[0.,0.,.9,2.],[0.,0.,0.,1.]])])
def test_chunked_equilibrium_matches_canonical_element_assembly(affine):
    e,_=specimen(affine,heterogeneous=True); ref=reference(e)
    expected=c.assemble(e.scene,ref.mesh,e.scene.properties[c.PROPERTY]['materials'])[0][0]
    difference=ref.stiffness-expected
    assert np.max(np.abs(difference.data),initial=0.)<1e-14


@pytest.mark.parametrize('times',[(),(.2,),(.2,.7)])
def test_every_reconstructed_history_matches_full_fem_for_heterogeneous_materials(times):
    e,_=specimen(heterogeneous=True,times=times); before=copy.deepcopy(e.state); ref=reference(e)
    rng=np.random.default_rng(271); target=rng.normal(size=(14,len(ref.spring)))*.003
    reduced=r.integrate(ref,target,.025); full=r.integrate_full(ref,target,.025)
    np.testing.assert_allclose(reduced['coordinates']@ref.basis.T,full['displacement'],atol=1e-12,rtol=1e-10)
    recovered=np.einsum('tbm,nm->tbn',reduced['memoryCoordinates'],ref.basis)
    np.testing.assert_allclose(recovered,full['memories'],atol=1e-12,rtol=1e-10)
    assert reduced['maxEnergyBalanceResidualNmm']<1e-12
    assert reduced['dissipatedEnergyNmm']>=0 and reduced['numericalDissipationNmm']>=0
    assert e.state==before
    e.scene.require_cache_source()


def test_nonproportional_relaxation_cannot_be_silently_reduced():
    with pytest.raises(ValueError,match='NONPROPORTIONAL'):
        r.proportional_spectrum({'a':{'shearPa':1000.,'relaxationShearPa':[500.]},
                                 'b':{'shearPa':2000.,'relaxationShearPa':[900.]}},[.5])


def test_reference_requires_declared_support_and_does_not_invent_clamps():
    e,_=specimen()
    with pytest.raises(ValueError,match='UNSUPPORTED_RIGID_COMPONENT'):
        r.build_reference(e,list(range(e.scene.labels.size)),max_dofs=10000)


def test_nonzero_initial_memory_or_displacement_and_body_load_refuse():
    e,_=specimen(); cells=r.source_domain(e,2.,max_cells=1000)
    e.state[c.PROPERTY]={'cornerDisplacementMm':{},'memoryDisplacementMm':[{'0':[[.01,0.,0.]]}]}
    with pytest.raises(ValueError,match='ZERO_INITIAL_FIELDS'):r.build_reference(e,cells,max_dofs=10000)
    e,_=specimen(); e.state['patientPosition']={}
    with pytest.raises(ValueError,match='BODY_LOAD_UNSUPPORTED'):reference(e)


def test_reference_source_cap_includes_seeds_and_leaves_input_unchanged():
    e,_=specimen(); before=copy.deepcopy(e.state)
    for limit in [1,2,3]:
        with pytest.raises(ValueError,match='SOURCE_CELL_CAP'):r.source_domain(e,2.,max_cells=limit)
    assert e.state==before


def test_removed_cells_and_changed_source_refuse_reference():
    e,cells=specimen(); e.state['removed'].append(cells[0])
    with pytest.raises(ValueError,match='retained source tissue'):r.build_reference(e,cells,max_dofs=10000)
    e,_=specimen(); e.scene.properties[c.PROPERTY]['handleStiffnessNPerMm']*=2
    with pytest.raises(ValueError,match='Source scene changed'):reference(e)


@pytest.mark.parametrize('target,h',[(np.ones((2,6),bool),.1),(np.zeros((2,3)),.1),(np.full((2,6),np.nan),.1),(np.zeros((2,6)),0.)])
def test_invalid_target_histories_refuse(target,h):
    e,_=specimen(); ref=reference(e)
    with pytest.raises(ValueError,match='contact-target'):r.integrate(ref,target,h)


def test_detailed_control_budget_remains_separate_from_reference_budget():
    e,_=specimen(); ref=reference(e); ref.free=np.ones(8001,bool)
    with pytest.raises(ValueError,match='DETAILED_CONTROL_CAP'):
        r.integrate_full(ref,np.zeros((1,len(ref.spring))),.1)


def test_prepared_detailed_factor_cannot_cross_references_or_physical_clocks():
    e,_=specimen(); ref=reference(e); prepared=r.prepare_full(ref,.1)
    target=np.zeros((1,len(ref.spring)))
    with pytest.raises(ValueError,match='another reference or physical step'):
        r.integrate_full(ref,target,.2,prepared=prepared)
    other=reference(e)
    with pytest.raises(ValueError,match='another reference or physical step'):
        r.integrate_full(other,target,.1,prepared=prepared)
