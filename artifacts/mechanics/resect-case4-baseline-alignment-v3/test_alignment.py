"""Small transform/sampling controls; no patient files or study data."""
import importlib.util
from pathlib import Path
import numpy as np
import pytest

spec=importlib.util.spec_from_file_location('baseline_alignment',Path(__file__).with_name('align.py'))
module=importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)

POINTS=np.array([[0,0,0],[1,0,0],[0,2,0],[0,0,3],[1,2,3],[2,-1,1.]],dtype=float)


def test_proper_fit_direction_and_leave_one_out_exact_rigid_geometry():
    matrix=np.array([[0,-1,0,3],[1,0,0,-2],[0,0,1,4],[0,0,0,1.]],dtype=float)
    targets=module.apply(POINTS,matrix)
    fitted,raw,residual,loo=module.errors(POINTS,targets)
    np.testing.assert_allclose(fitted,matrix,atol=1e-14)
    assert raw.max()>1
    assert residual.max()<1e-13 and loo.max()<1e-13


def test_reflection_is_not_silently_accepted_as_proper_rotation():
    targets=POINTS*np.array([-1,1,1])
    fitted=module.rigid(POINTS,targets)
    assert np.linalg.det(fitted[:3,:3])==pytest.approx(1)
    assert np.linalg.norm(module.apply(POINTS,fitted)-targets)>1


def test_collinear_points_fail_without_fallback():
    points=np.column_stack((np.arange(4),np.zeros(4),np.zeros(4)))
    with pytest.raises(ValueError,match='degenerate'):
        module.rigid(points,points+1)


def test_loo_target_is_excluded_from_its_fit():
    targets=POINTS+np.array([1,2,3.])
    targets[0]+=np.array([10,0,0.])
    _,_,_,loo=module.errors(POINTS,targets)
    assert loo[0]==pytest.approx(10,abs=1e-12)


def test_oblique_reflected_native_plane_and_physical_map_direction():
    indices=np.indices((8,9,10),dtype=float)
    array=indices[0]+2*indices[1]+3*indices[2]
    affine=np.array([[-2,.2,0,10],[.1,3,.3,-20],[0,.4,4,7],[0,0,0,1.]])
    shift=np.eye(4); shift[2,3]=1
    world_map=affine@shift@np.linalg.inv(affine)
    sampled=module.sample_plane(array,affine,affine,array.shape,2,2,world_map)
    np.testing.assert_allclose(sampled[1:-1,1:-1],array[:,:,3][1:-1,1:-1],atol=1e-12)
    outside=module.sample_plane(array,affine,affine,array.shape,2,99)
    assert np.isnan(outside).all()
