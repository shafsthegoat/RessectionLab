"""Computation revision2: exact owner choice and bounded tensor disagreement."""
import copy

import numpy as np
import pytest

from resectionlab import surgical_continuum as c
from resectionlab.surgical_continuum_solve import PROPERTY,profile,VECTOR_SCHEMA
from resectionlab.surgical_scene import SurgicalScene
from resectionlab.surgical_physics_v2 import default_profile
from resectionlab.surgical_engine import _copy_engine_data,_owned_digest
from resectionlab.core import semantic_digest


def scene(affine=np.eye(4),method='detailed',vector=True):
    p={'tissuePhysicsV2':default_profile(),c.PROPERTY:c.research_profile({'1':{
        'shearPa':1000.,'poissonRatio':.3,'relaxationShearPa':[500.],'source':'Generated kernel control'}},relaxation_seconds=[.5]),
        PROPERTY:profile(method,tensor_contraction='blas-contiguous' if vector else None)}
    return SurgicalScene(np.ones((3,3,3),np.uint16),affine,[{'id':1,'name':'Generated block','material':'soft_tissue'}],{'kind':'generated control'},p)


@pytest.mark.parametrize('affine',[np.eye(4),np.diag([-1.,2.,.7,1.]),np.array([[1.,.2,0.,4.],[0.,1.1,.3,-2.],[.1,0.,.8,7.],[0.,0.,0.,1.]])])
def test_tensor_revision_matches_source_quadrature_with_original_contiguous_layout(affine):
    s=scene(affine);g=c.element_operators(affine)[2];rng=np.random.default_rng(311)
    for cells in [1,7,127]:
        for scale in [1e-9,.1,1e6]:
            local=rng.normal(size=(cells,8,3))*scale
            expected=np.einsum('eai,gaj->egij',local,g)
            actual=c._gradient(s,local,g)
            assert actual.flags.c_contiguous
            np.testing.assert_allclose(actual,expected,rtol=1e-13,atol=1e-15*max(1.,scale))
    old=scene(affine,vector=False)
    np.testing.assert_array_equal(c._gradient(old,local,g),expected)


@pytest.mark.parametrize('cut',[False,True])
def test_vector_unpack_preserves_first_available_source_owner_and_split_corners(cut):
    s=scene();cells=list(range(s.labels.size));blocked=[]
    if cut:
        blocked=[(int(np.ravel_multi_index((0,y,z),s.shape)),int(np.ravel_multi_index((1,y,z),s.shape))) for y in range(3) for z in range(3)]
    mesh=c.voxel_mesh(s,cells,blocked);rng=np.random.default_rng(14);u=rng.normal(size=(len(mesh.nodes),3))*.01
    field=c._pack(mesh,u.ravel())
    for present in [cells,cells[::2],[],cells[3:9]]:
        partial={str(cell):field[str(cell)] for cell in present}
        np.testing.assert_array_equal(c._unpack_vectorized(mesh,partial),c._unpack(mesh,partial))
    shared=next(owners for owners in mesh.node_owners if len(owners)>1)
    bad=copy.deepcopy(field);cell,local=shared[1];bad[str(cell)][local][0]+=1e-5
    for unpack in [c._unpack,c._unpack_vectorized]:
        with pytest.raises(ValueError,match='INCONSISTENT'):unpack(mesh,bad)


def test_corner_copy_fast_path_preserves_memo_aliases_detachment_and_cycles():
    row=[1.,2.,3.];corners=[row]*8
    value={'row':row,'corners':corners,'again':corners};result=_copy_engine_data(value)
    assert result==copy.deepcopy(value)
    assert result['corners'] is result['again'] and result['corners'][0] is result['row']
    assert all(r is result['row'] for r in result['corners'])
    result['row'][0]=100.;assert row[0]==1.
    cycle=[];cycle.append(cycle);out=_copy_engine_data({'corners':corners,'cycle':cycle})
    assert out['cycle'][0] is out['cycle']


def test_occupancy_owned_hash_keeps_canonical_bytes_and_unusual_values_fallback():
    changes={'holds':{},'deformation':{str(i):[i*.1,0.,-0.] for i in range(100)}}
    assert _owned_digest(changes)==semantic_digest(changes)
    changes['deformation']['3'][0]=.123
    assert _owned_digest(changes)==semantic_digest(changes)
    changes['deformation']['other']=[np.float64(1.),2.,3.]
    assert _owned_digest(changes)==semantic_digest(changes)
    changes['deformation']['other']=np.array([1.,2.,3.])
    for digest in [_owned_digest,semantic_digest]:
        with pytest.raises(TypeError,match='JSON-compatible'):digest(changes)


def test_revision_one_stays_declared_and_revision_two_is_explicit():
    old=scene(vector=False);new=scene()
    assert old.properties[PROPERTY]['schema']!=VECTOR_SCHEMA
    assert new.properties[PROPERTY]['schema']==VECTOR_SCHEMA and old.identity!=new.identity
