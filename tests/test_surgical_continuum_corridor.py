"""Native raster ownership controls for a conservative algebraic selection."""
import copy

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from resectionlab import surgical_continuum as continuum
from resectionlab.geometry import GeometryScene
from resectionlab.surgical_scene import SurgicalScene
from resectionlab.surgical_source_contacts import SourceContactGeometry
from resectionlab.surgical_continuum_corridor import corridor_partition


def geometry(labels,affine):
    scene=SurgicalScene(labels,affine,[{'id':1,'material':'soft_tissue'}],{'kind':'generated corridor control'}, {})
    mesh=continuum.voxel_mesh(scene,np.flatnonzero(labels))
    g={'cells':np.asarray(mesh.cells),'shape':np.asarray(scene.shape),'nodes':mesh.nodes,'elementNodes':mesh.element_nodes,
        'affine':affine,'freeDofs':np.ones(mesh.nodes.size,bool),'muPa':np.full(len(mesh.cells),1000.),'nu':np.full(len(mesh.cells),.3)}
    return scene,g


def assert_core_owns(g,arrays,source_ids):
    free=np.flatnonzero(g['freeDofs']);core=set(free[arrays['coreFreeDofs']]);lookup={int(c):i for i,c in enumerate(g['cells'])}
    for c in source_ids:
        dofs=(3*g['elementNodes'][lookup[c],:,None]+np.arange(3)).ravel()
        assert set(dofs[g['freeDofs'][dofs]])<=core


def test_two_half_diagonal_margins_cover_real_rounding_ownership_and_remote_spring():
    labels=np.zeros((20,20,20),np.uint16)
    for p in ((3,3,3),(9,9,9),(15,15,15)):labels[p]=1
    scene,g=geometry(labels,np.eye(4));lookup={int(c):i for i,c in enumerate(g['cells'])}
    owner=int(np.ravel_multi_index((3,3,3),scene.shape));held=int(np.ravel_multi_index((9,9,9),scene.shape))
    contacts=[{'sourceCell':held}];u=np.zeros((3,8,3));u[lookup[owner]]=-1.51
    native=SourceContactGeometry(scene,GeometryScene(np.zeros(scene.shape,bool),scene.affine),[],g['cells'],u,holds={},detached_fragments=[])
    assert native.cusa_order([0.,0.,0.])==[owner] and np.linalg.norm(u[lookup[owner]].mean(axis=0))<2.62
    summary,arrays=corridor_partition(g,contacts,[0.,0.,0.],[0.,0.,0.],1.5,2.62)
    # One half-diagonal alone would omit this actual legacy raster owner.
    assert np.linalg.norm([3.,3.,3.])>1.5+2.62+summary['sourceHalfDiagonalMm']
    assert owner in arrays['corridorSourceIds'] and held not in arrays['corridorSourceIds']
    assert_core_owns(g,arrays,[owner,held])
    assert not summary['factorRetentionCertified'] and not summary['physicalValidationPassed']


@pytest.mark.parametrize('rotated',[False,True])
def test_bounded_displaced_native_contacts_along_segment_never_escape_selected_owners(rotated):
    affine=np.eye(4);affine[:3,:3]=Rotation.from_euler('xyz',[19.,-31.,47.],degrees=True).as_matrix()@np.diag([.8,1.1,1.3]) if rotated else np.eye(3)
    affine[:3,3]=[17.,-24.,8.];labels=np.zeros((14,14,14),np.uint16);labels[2:12,2:12,2:12]=1
    scene,g=geometry(labels,affine);before={k:v.copy() for k,v in g.items()};rng=np.random.default_rng(703)
    a=np.array([1.,6.,6.])@affine[:3,:3].T+affine[:3,3];b=np.array([3.,6.,6.])@affine[:3,:3].T+affine[:3,3]
    contacts=[{'sourceCell':int(np.ravel_multi_index((10,10,10),scene.shape))}]
    summary,arrays=corridor_partition(g,contacts,a,b,1.5,.75);chosen=set(arrays['corridorSourceIds']);seen=set()
    displacement=rng.normal(size=(len(g['cells']),3));displacement*=rng.uniform(0,.75,(len(g['cells']),1))/np.linalg.norm(displacement,axis=1)[:,None]
    native=SourceContactGeometry(scene,GeometryScene(np.zeros(scene.shape,bool),scene.affine),[],g['cells'],np.repeat(displacement[:,None,:],8,axis=1),holds={},detached_fragments=[])
    for t in np.r_[0.,1.,rng.random(18)]:
        actual=native.cusa_order(a+t*(b-a));assert set(actual)<=chosen;assert_core_owns(g,arrays,actual);seen.update(actual)
    assert seen and summary['physicalTotalDofs']==g['nodes'].size and summary['physicalFreeDofs']==int(g['freeDofs'].sum())
    assert all(np.array_equal(v,before[k]) for k,v in g.items())
    with pytest.raises(ValueError):arrays['coreFreeDofs'][0]=-1


@pytest.mark.parametrize('bad',[-1.,np.nan,True,np.inf])
def test_invalid_displacement_bound_refuses(bad):
    labels=np.zeros((8,8,8),np.uint16);labels[1,1,1]=labels[6,6,6]=1;_,g=geometry(labels,np.eye(4))
    with pytest.raises(ValueError,match='DISTANCE_BOUND'):corridor_partition(g,[],[0,0,0],[0,0,0],1.5,bad)


def test_shear_source_corner_mismatch_and_missing_spring_owner_refuse():
    labels=np.zeros((8,8,8),np.uint16);labels[1,1,1]=labels[6,6,6]=1;_,g=geometry(labels,np.eye(4))
    wrong=copy.deepcopy(g);wrong['affine'][0,1]=.1
    with pytest.raises(ValueError,match='ORTHOGONAL'):corridor_partition(wrong,[],[0,0,0],[0,0,0],1.5,0.)
    wrong=copy.deepcopy(g);wrong['elementNodes'][0]=wrong['elementNodes'][0,::-1]
    with pytest.raises(ValueError,match='CORNER_OWNERSHIP'):corridor_partition(wrong,[],[0,0,0],[0,0,0],1.5,0.)
    with pytest.raises(ValueError,match='CONTACT_OUTSIDE'):corridor_partition(g,[{'sourceCell':0}],[0,0,0],[0,0,0],1.5,0.)
