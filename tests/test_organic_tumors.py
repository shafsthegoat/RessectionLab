import numpy as np
import pytest
from resectionlab.organic_tumors import organic_geometry,signed_domain,shape_metrics


def test_lobulated_shapes_are_connected_contained_reproducible_and_not_cuboids():
    axes=np.indices((65,65,65));domain=np.sum(((axes-32)/29)**2,axis=0)<1
    a=np.diag([1.2,1.1,1.,1.]);a[:3,3]=[-40,-30,-20]
    signed=signed_domain(domain,a)
    recipes=[]
    for seed in [8,29,47]:
        masks,r=organic_geometry(domain,a,center_voxel=[42,40,32],seed=seed,radii_mm=[7,10,12],signed=signed)
        again,repeat=organic_geometry(domain,a,center_voxel=[42,40,32],seed=seed,radii_mm=[7,10,12],signed=signed)
        assert r==repeat and all(np.array_equal(masks[k],again[k])for k in masks)
        union=np.logical_or.reduce(list(masks.values()))
        assert not np.any(union&~domain) and np.max(sum(m.astype(np.uint8)for m in masks.values()))==1
        assert r['number_of_shape_lobes']>=3 and not r['axis_aligned_box_used_as_shape']
        assert not r['metrics']['is_cuboid'] and r['metrics']['bounding_box_fill']<.75
        assert r['metrics']['surface_solidity']<.98 and r['metrics']['components']==1
        assert np.allclose(r['center_ras_mm'],a[:3,:3]@[42,40,32]+a[:3,3])
        recipes.append(r['metrics']['voxel_count'])
    assert len(set(recipes))==3


def test_shape_checker_identifies_cubes_and_generation_rejects_invalid_seeds():
    cube=np.zeros((20,20,20),bool);cube[5:15,5:15,5:15]=True
    assert shape_metrics(cube,np.eye(4))['is_cuboid']
    assert shape_metrics(cube,np.eye(4))['bounding_box_fill']==1
    with pytest.raises(ValueError,match='Center must'):
        organic_geometry(cube,np.eye(4),center_voxel=[2,2,2],seed=3)
