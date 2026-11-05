"""Physical surface ownership and independent compressive-contact controls."""
import numpy as np
import pytest
from scipy import sparse
from scipy.spatial.transform import Rotation

from resectionlab.surgical_scene import SurgicalScene
from resectionlab.surgical_continuum import voxel_mesh,assemble
from resectionlab.surgical_retractor_surface import exposed_quadrature,PlanarBlade
from resectionlab.surgical_contact_complementarity import solve_unilateral
from resectionlab.surgical_retractor_contact import step as surface_step


def source(cells=((2,2,2),),affine=None,shape=(5,5,5),extra=None):
    labels=np.zeros(shape,np.uint16)
    for p in cells:labels[p]=1
    if extra:
        for p,label in extra.items():labels[p]=label
    scene=SurgicalScene(labels,np.eye(4) if affine is None else affine,
        [{'id':1,'material':'soft_tissue'},{'id':2,'material':'bone'}],{'kind':'generated contact control'}, {})
    ids=[int(np.ravel_multi_index(p,shape)) for p in cells];return scene,voxel_mesh(scene,ids)


def test_surface_uses_exposed_q1_faces_and_excludes_artificial_patch_and_solid_interfaces():
    scene,mesh=source(extra={(3,2,2):1});surface=exposed_quadrature(scene,mesh)
    assert len(surface.source_ids)==20 and not np.any((surface.axes==0)&(surface.signs==1))
    other=int(np.ravel_multi_index((3,2,2),scene.shape));cell=mesh.cells[0]
    assert len(exposed_quadrature(scene,mesh,[other]).source_ids)==24
    assert len(exposed_quadrature(scene,mesh,cut_faces=[(cell,other)]).source_ids)==24
    scene,mesh=source(extra={(3,2,2):2});assert len(exposed_quadrature(scene,mesh).source_ids)==20


def test_retained_neighbor_faces_are_not_contact_surfaces_and_removed_mesh_owner_refuses():
    scene,mesh=source(((2,2,2),(3,2,2)));surface=exposed_quadrature(scene,mesh)
    assert len(surface.source_ids)==40
    with pytest.raises(ValueError,match='REMOVED_OWNERS'):exposed_quadrature(scene,mesh,[mesh.cells[0]])


@pytest.mark.parametrize('reflected',[False,True])
def test_face_quadrature_area_outward_orientation_and_nodal_interpolation(reflected):
    affine=np.eye(4);affine[:3,:3]=Rotation.from_euler('xyz',[17.,-31.,49.],degrees=True).as_matrix()@np.diag([-.8 if reflected else .8,1.1,1.3]);affine[:3,3]=[7.,-4.,11.]
    scene,mesh=source(affine=affine);surface=exposed_quadrature(scene,mesh);points,normals=surface.geometry(mesh.nodes,np.zeros_like(mesh.nodes))
    np.testing.assert_allclose(surface.weights.sum(axis=1),1.,atol=1e-15)
    A=affine[:3,:3]
    for axis in range(3):
        for sign in (-1,1):
            chosen=(surface.axes==axis)&(surface.signs==sign);others=[a for a in range(3) if a!=axis]
            assert surface.reference_area_mm2[chosen].sum()==pytest.approx(np.linalg.norm(np.cross(A[:,others[0]],A[:,others[1]])),abs=1e-14)
            n=np.linalg.inv(A).T[:,axis]*sign;n/=np.linalg.norm(n)
            np.testing.assert_allclose(normals[chosen],np.repeat(n[None],4,axis=0),atol=1e-14)
    u=np.arange(mesh.nodes.size,dtype=float).reshape(-1,3)*1e-4;indices=np.arange(len(surface.source_ids));c=surface.rows(normals,indices)
    displaced,_=surface.geometry(mesh.nodes,u)
    np.testing.assert_allclose(c@u.ravel(),np.sum(normals*(displaced-points),axis=1),atol=3e-15)
    with pytest.raises(ValueError):surface.weights[0,0]=2.


def test_image_edge_is_unknown_instead_of_a_declared_free_contact_surface():
    scene,mesh=source(((0,0,0),));surface=exposed_quadrature(scene,mesh)
    assert len(surface.source_ids)==12 and len(surface.unknown_fov_faces)==3
    assert set(surface.signs)=={1}


def test_finite_blade_uses_surface_points_and_loses_footprint_on_reposition():
    scene,mesh=source();surface=exposed_quadrature(scene,mesh);u=np.zeros_like(mesh.nodes)
    blade=PlanarBlade.create([2.,1.1,2.5],[0,0,0,1],.4,1.)
    hit=blade.candidates(surface,mesh.nodes,u)
    assert len(hit['pointIndices'])==4 and np.all(surface.axes[hit['pointIndices']]==1) and np.all(surface.signs[hit['pointIndices']]==-1)
    np.testing.assert_allclose(hit['gapMm'],0.,atol=3e-16)
    assert np.all(hit['positionsMm'][:,1]==1.5) and not np.allclose(hit['positionsMm'].mean(axis=0),[2,2,2])
    clear=PlanarBlade.create([4.,1.1,2.5],[0,0,0,1],.4,1.).candidates(surface,mesh.nodes,u)
    assert len(clear['pointIndices'])==0
    separated=PlanarBlade.create([2.,.8,2.5],[0,0,0,1],.4,1.).candidates(surface,mesh.nodes,u)
    np.testing.assert_allclose(separated['gapMm'],.3,atol=3e-16)


def test_surface_and_blade_are_covariant_in_a_rigid_world_frame():
    scene,mesh=source();surface=exposed_quadrature(scene,mesh);u=np.zeros_like(mesh.nodes);p=np.array([2.,1.1,2.5]);blade=PlanarBlade.create(p,[0,0,0,1],.4,1.);hit=blade.candidates(surface,mesh.nodes,u)
    rotation=Rotation.from_euler('xyz',[23.,-42.,61.],degrees=True);affine=np.eye(4);affine[:3,:3]=rotation.as_matrix();affine[:3,3]=[13.,7.,-2.]
    other,other_mesh=source(affine=affine);other_surface=exposed_quadrature(other,other_mesh)
    changed=PlanarBlade.create(rotation.apply(p)+affine[:3,3],rotation.as_quat(),.4,1.).candidates(other_surface,other_mesh.nodes,np.zeros_like(other_mesh.nodes))
    assert np.array_equal(hit['pointIndices'],changed['pointIndices']);np.testing.assert_allclose(hit['gapMm'],changed['gapMm'],atol=4e-15)
    np.testing.assert_allclose(rotation.apply(hit['normal']),changed['normal'],atol=64*np.finfo(float).eps)


def solve(k,f,c,b,check=lambda:None):
    matrix=np.asarray(k,float);return solve_unilateral(sparse.csr_matrix(matrix),np.asarray(f,float),sparse.csr_matrix(c),np.asarray(b,float),lambda x:np.linalg.solve(matrix,x),check=check)


def test_analytical_compression_separation_and_zero_tangential_force():
    contact=solve([[.002,0],[0,.003]],[0,0],[[1,0]],[.1])
    np.testing.assert_allclose(contact['displacement'],[.1,0],atol=1e-13);np.testing.assert_allclose(contact['normalForcesN'],[.0002],atol=1e-15)
    clear=solve([[.002,0],[0,.003]],[0,0],[[1,0]],[-.1])
    assert np.array_equal(clear['displacement'],[0,0]) and np.array_equal(clear['normalForcesN'],[0])
    assert clear['gapMm'][0]==pytest.approx(.1)


def test_oblique_frictionless_reaction_has_no_tangential_component():
    n=np.array([.6,.8]);contact=solve(np.eye(2)*.002,[0,0],[n],[.1]);force=contact['normalForcesN'][0]*n
    np.testing.assert_allclose(contact['displacement'],.1*n,atol=1e-13);assert abs(force@[-.8,.6])<1e-18


@pytest.mark.parametrize('targets',[[.1,.1],[.1,0.]])
def test_redundant_surface_constraints_keep_true_gaps_without_projecting_targets(targets):
    contact=solve([[.002,0],[0,.003]],[0,0],[[1,0],[1,0]],targets)
    np.testing.assert_allclose(contact['displacement'],[.1,0],atol=1e-12)
    assert contact['normalForcesN'].sum()==pytest.approx(.0002,abs=1e-14)
    assert np.min(contact['gapMm'])>=-1e-9 and contact['maxComplementarityNmm']<=1e-10


def test_fixed_penetrating_constraint_and_incompatible_contacts_refuse():
    with pytest.raises(ValueError,match='INFEASIBLE_FIXED_POINT'):solve([[.002]],[0],[[0]],[.1])
    with pytest.raises(ValueError,match='COMPLEMENTARITY_OR_EQUILIBRIUM'):solve([[.002]],[0],[[1],[-1]],[.1,.1])


def test_contact_maxwell_memory_and_moving_tool_energy_identity():
    k0=.002;k1=.001;tau=.5;h=.1;r=tau/(tau+h);effective=k0+r*k1
    for old,z,target,old_target in ((.1,.03,.12,.1),(.1,.03,-.1,.1),(0.,0.,.05,-.1)):
        rhs=r*k1*z
        contact=solve([[effective]],[rhs],[[1]],[target]);u=contact['displacement'][0];force=contact['normalForcesN'][0];memory=(tau*z+h*u)/(tau+h)
        E=lambda x,v:.5*k0*x*x+.5*k1*(x-v)**2
        loss=tau/h*k1*(memory-z)**2;num=.5*k0*(u-old)**2+.5*k1*((u-memory)-(old-z))**2
        tool_work=force*(target-old_target);contact_num=force*(old-old_target)
        assert abs(tool_work-(E(u,memory)-E(old,z))-loss-num-contact_num)<1e-18
        assert memory!=0 and loss>=0 and num>=0 and contact_num>=0
        if target<0:assert force==0 and u>0


def test_cancellation_and_invalid_inverse_do_not_modify_caller_inputs():
    k=np.array([[.002]]);rhs=np.array([0.]);c=np.array([[1.]]);target=np.array([.1]);before=[x.copy() for x in (k,rhs,c,target)]
    def cancel():raise InterruptedError('Generated contact cancellation')
    with pytest.raises(InterruptedError):solve(k,rhs,c,target,cancel)
    assert all(np.array_equal(a,b) for a,b in zip((k,rhs,c,target),before))
    with pytest.raises(ValueError,match='RESPONSE_RESIDUAL'):
        solve_unilateral(k,rhs,c,target,lambda x:np.zeros_like(x))


def coupon():
    scene,mesh=source();material={'1':{'shearPa':1000.,'poissonRatio':.3,'relaxationShearPa':[500.],'source':'Generated mechanical control'}}
    matrices,_=assemble(scene,mesh,material);free=np.repeat(~np.isclose(mesh.nodes[:,1],2.5),3)
    surface=exposed_quadrature(scene,mesh);blade=PlanarBlade.create([2.,1.1,2.7],[0,0,0,1],1.,1.4)
    return mesh,matrices,free,surface,blade


def test_actual_q1_surface_push_separation_and_departure_preserve_history_and_work():
    mesh,matrices,free,surface,old=coupon();zero=np.zeros(mesh.nodes.size);memory=np.zeros((1,len(zero)));h=1/120
    pushed=PlanarBlade.create([2.,1.2,2.7],[0,0,0,1],1.,1.4)
    first=surface_step(matrices,free,zero,memory,[.5],h,zero,mesh,surface,old,pushed)
    front=np.isclose(mesh.nodes[:,1],1.5)
    np.testing.assert_allclose(first['displacement'].reshape(-1,3)[front,1],.1,atol=1e-10)
    assert first['summedNormalForceN']>0 and first['forceOnTissueN'][1]>0
    assert abs(first['forceOnTissueN'][0])+abs(first['forceOnTissueN'][2])<1e-15
    assert first['contactSourceIds']==list(mesh.cells) and first['sampledTouchAreaMm2']==pytest.approx(1.)
    assert abs(first['energyBalanceResidualNmm'])<1e-18 and first['maxFreeResidualN']<1e-10 and np.any(first['memories']!=0)
    for position in ([2.,.9,2.7],[5.,1.2,2.7]):
        clear=PlanarBlade.create(position,[0,0,0,1],1.,1.4)
        result=surface_step(matrices,free,first['displacement'],first['memories'],[.5],h,zero,mesh,surface,pushed,clear)
        assert result['summedNormalForceN']==0 and result['contactSourceIds']==[] and result['loadedSourceIds']==[]
        assert np.any(result['memories']!=0) and result['toolWorkNmm']==0 and abs(result['energyBalanceResidualNmm'])<1e-18
        assert result['edgeContactValidated'] is False and result['patientForceValidated'] is False


def test_initial_penetration_loaded_edge_entry_and_unqualified_rotation_refuse():
    mesh,matrices,free,surface,old=coupon();zero=np.zeros(mesh.nodes.size);memory=np.zeros((1,len(zero)))
    beside=PlanarBlade.create([5.,1.3,2.7],[0,0,0,1],1.,1.4);inside=PlanarBlade.create([2.,1.3,2.7],[0,0,0,1],1.,1.4)
    with pytest.raises(ValueError,match='LOADED_EDGE_ENTRY'):
        surface_step(matrices,free,zero,memory,[.5],1/120,zero,mesh,surface,beside,inside)
    turned=PlanarBlade.create(old.position,Rotation.from_euler('z',10,degrees=True).as_quat(),1.,1.4)
    with pytest.raises(ValueError,match='ROTATION_UNQUALIFIED'):
        surface_step(matrices,free,zero,memory,[.5],1/120,zero,mesh,surface,old,turned)
    assert np.all(zero==0) and np.all(memory==0)
