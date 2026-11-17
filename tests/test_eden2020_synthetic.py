"""Generated source controls for the new seeded placement option."""
import numpy as np
import pytest
from resectionlab.data_synthesis import geometry, appearance


def test_seeded_random_lesions_respect_oblique_physical_support_and_preserve_baseline():
    grid=np.indices((80,80,80));brain=np.sum(((grid-39.5)/37.)**2,axis=0)<1
    angle=.4
    affine=np.eye(4);affine[:3,:3]=np.array([[-np.cos(angle),-np.sin(angle),0],[-np.sin(angle),np.cos(angle),0],[0,0,1]])
    affine[:3,3]=[70.,-50.,10.]
    support=brain.copy();support[:10]=False
    baseline=(100+grid[0]+grid[1]*.4+grid[2]*.2).astype(np.float32);unchanged=baseline.copy()
    centers=[]
    for seed in (7,11,19):
        masks,recipe=geometry(brain,affine,seed=seed,variant=seed%3,placement_support=support,random_placement=True)
        again,repeat=geometry(brain,affine,seed=seed,variant=seed%3,placement_support=support,random_placement=True)
        assert recipe==repeat and all(np.array_equal(masks[k],again[k]) for k in masks)
        assert np.allclose(recipe['center_ras_mm'],affine[:3,:3]@recipe['center_voxel']+affine[:3,3])
        total=sum(m.astype(np.uint8) for m in masks.values())
        assert total.max()==1 and not np.any((total>0)&~support)
        image,_=appearance(baseline,brain,masks,seed=seed)
        assert np.array_equal(baseline,unchanged)
        assert np.isfinite(image).all() and not np.array_equal(image,baseline)
        assert np.array_equal(image[total==0],baseline[total==0])
        centers.append(tuple(recipe['center_voxel']))
    assert len(set(centers))==3


def test_random_placement_does_not_relax_minimum_support_or_change_default():
    brain=np.zeros((40,40,40),bool);brain[15:25,15:25,15:25]=True
    with pytest.raises(ValueError,match='minimum engineering lesion'):
        geometry(brain,np.eye(4),seed=7,random_placement=True)
    brain=np.zeros((80,80,80),bool);brain[3:-3,3:-3,3:-3]=True
    first,r1=geometry(brain,np.eye(4),seed=8)
    second,r2=geometry(brain,np.eye(4),seed=8,random_placement=False)
    assert r1==r2 and all(np.array_equal(first[k],second[k]) for k in first)
