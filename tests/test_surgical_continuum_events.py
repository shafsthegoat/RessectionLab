"""Exact source-owner inheritance and separate instantaneous energy changes."""
import copy

import numpy as np
import pytest

from resectionlab.surgical_scene import SurgicalScene
from resectionlab.surgical_continuum import voxel_mesh
from resectionlab.surgical_continuum_events import SourceHistoryLedger,stored_energy_by_source,source_operator_identity,contact_binding_jump


def geometry(cells=range(27),cut=False):
    affine=np.array([[-1.,.2,0.,4.],[0.,1.1,.1,-3.],[0.,0.,.9,2.],[0.,0.,0.,1.]])
    scene=SurgicalScene(np.ones((3,3,3),np.uint16),affine,
        [{'id':1,'name':'Generated tissue','material':'soft_tissue'}],{'kind':'generated event control'}, {})
    blocked=[(int(np.ravel_multi_index((1,y,z),scene.shape)),int(np.ravel_multi_index((2,y,z),scene.shape))) for y in range(3) for z in range(3)] if cut else []
    mesh=voxel_mesh(scene,cells,blocked)
    return dict(cells=np.asarray(mesh.cells),shape=np.asarray(scene.shape),nodes=mesh.nodes,
        elementNodes=mesh.element_nodes,affine=affine,freeDofs=np.ones(mesh.nodes.size,bool),
        muPa=np.full(len(mesh.cells),1000.),nu=np.full(len(mesh.cells),.3))


def fields(g):
    a=np.array([[.01,.002,0.],[0.,-.003,.001],[0.,0.,.004]])
    u=(g['nodes']@a.T).ravel()
    return u,np.asarray([.25*u,.5*u]),np.array([.5,.2]),a


def test_cut_inherits_every_corner_and_memory_without_energy_change_then_equal_join():
    old=geometry();new=geometry(cut=True);u,z,alpha,_=fields(old);ledger=SourceHistoryLedger()
    result=ledger.transfer(old,new,u,z,alpha)
    assert result['event']['splitParents']>0 and result['event']['removedElementCount']==0
    assert abs(result['event']['tissueEventBalanceResidualNmm'])<1e-15
    np.testing.assert_array_equal(result['displacement'].reshape((-1,3))[new['elementNodes']],u.reshape((-1,3))[old['elementNodes']])
    joined=ledger.transfer(new,old,result['displacement'],result['memories'],alpha)
    assert joined['event']['joinedNodes']>0
    np.testing.assert_array_equal(joined['displacement'],u);np.testing.assert_array_equal(joined['memories'],z)


@pytest.mark.parametrize('branch',[None,0,1])
def test_even_one_ulp_unequal_join_refuses_transactionally(branch):
    g=geometry();cut=geometry(cut=True);u,z,alpha,_=fields(g);l=SourceHistoryLedger()
    first=l.transfer(g,cut,u,z,alpha);du=first['displacement'].copy();dz=first['memories'].copy()
    # Locate coincident sectors which will become one node, without using the
    # transfer implementation's owner rule.
    _,inverse,counts=np.unique(cut['nodes'],axis=0,return_inverse=True,return_counts=True)
    nodes=np.flatnonzero(inverse==np.flatnonzero(counts>1)[0]);index=3*nodes[-1]
    value=du if branch is None else dz[branch];value[index]=np.nextafter(value[index],np.inf)
    count=len(l.events)
    with pytest.raises(ValueError,match='UNEQUAL_JOINED'):l.transfer(cut,g,du,dz,alpha)
    assert len(l.events)==count and not l.removed_source_ids


def test_removal_energy_matches_analytic_affine_field_and_exports_once():
    old=geometry();new=geometry([i for i in range(27) if i!=13]);u,z,alpha,a=fields(old);ledger=SourceHistoryLedger()
    result=ledger.transfer(old,new,u,z,alpha);eps=(a+a.T)/2;mu=1000e-6;lam=2*mu*.3/(1-2*.3)
    expected=(mu*np.sum(eps**2)+.5*lam*np.trace(eps)**2)*abs(np.linalg.det(old['affine'][:3,:3]))*(1+.5*.75**2+.2*.5**2)
    assert result['event']['removedSourceIds']==[13]
    assert abs(result['event']['removedStoredEnergyNmm']-expected)<1e-18
    with pytest.raises(ValueError,match='DUPLICATE_REMOVAL'):ledger.transfer(old,new,u,z,alpha)
    assert ledger.removed_source_ids=={13} and len(ledger.events)==1
    with pytest.raises(ValueError,match='RESURRECTION'):ledger.transfer(old,old,u,z,alpha)
    assert ledger.removed_source_ids=={13} and len(ledger.events)==1


def test_join_does_not_discard_signed_zero_history_bits():
    g=geometry();cut=geometry(cut=True);u,z,alpha,_=fields(cut)
    _,inverse,counts=np.unique(cut['nodes'],axis=0,return_inverse=True,return_counts=True)
    nodes=np.flatnonzero(inverse==np.flatnonzero(counts>1)[0])
    for node in nodes:u[3*node]=0.
    u[3*nodes[-1]]=-0.
    with pytest.raises(ValueError,match='UNEQUAL_JOINED'):
        SourceHistoryLedger().transfer(cut,g,u,z,alpha)


def test_new_support_nonzero_displacement_or_memory_and_material_change_refuse():
    old=geometry();u,z,alpha,_=fields(old);ledger=SourceHistoryLedger();new=copy.deepcopy(old)
    new['freeDofs'][0]=False
    with pytest.raises(ValueError,match='NEW_SUPPORT'):ledger.transfer(old,new,u,z,alpha)
    u[0]=0.
    with pytest.raises(ValueError,match='NEW_SUPPORT'):ledger.transfer(old,new,u,z,alpha)
    new=copy.deepcopy(old);new['muPa'][0]*=2
    with pytest.raises(ValueError,match='MATERIAL_CHANGE'):ledger.transfer(old,new,u,z,alpha)
    assert ledger.events==[]


def test_complete_operator_identity_binds_exterior_removal_support_cuts_and_spectrum():
    g=geometry(range(8));kw={'core_free_dofs':[0,1,2]};base=source_operator_identity(g,[.5],[.4],**kw)
    assert base!=source_operator_identity(g,[.5],[.4],removed_source_ids=[26],**kw)
    assert base!=source_operator_identity(g,[.5],[.4],cut_faces=[(7,8)],**kw)
    assert base!=source_operator_identity(g,[.4],[.4],**kw)
    other=copy.deepcopy(g);other['freeDofs'][0]=False
    assert base!=source_operator_identity(other,[.5],[.4],**kw)
    assert base!=source_operator_identity(g,[.5],[.4],core_free_dofs=[2,1,0])
    assert source_operator_identity(g,[.5],[.4],removed_source_ids=[25,26],**kw)==source_operator_identity(g,[.5],[.4],removed_source_ids=[26,25,25],**kw)


def test_contact_release_jump_is_not_tissue_removal_or_be_dissipation():
    from resectionlab.surgical_continuum_exterior import ExteriorOperator,ExteriorHistorySolve
    k=np.array([[2.,-.3],[-.3,3.]])
    held=ExteriorHistorySolve(ExteriorOperator(k,[0]),[[1.,0.]],[.2],[.5],[.4])
    released=held.rebind(np.empty((0,2)),[])
    jump=contact_binding_jump(held,released,np.array([.01,.02]),np.array([.01,.02]),[.1],[])
    assert jump['contactModelEnergyJumpNmm']==pytest.approx(-.5*.2*.09**2)
    assert jump['newContactEnergyNmm']==0.
