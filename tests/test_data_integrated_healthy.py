import importlib.util
from pathlib import Path
import numpy as np
import pytest

sitk=pytest.importorskip("SimpleITK")

spec=importlib.util.spec_from_file_location('data_integrated_healthy',Path(__file__).parents[1]/'scripts/data_integrated_healthy.py')
module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)


def test_nonzero_center_lps_pull_inverse_maps_ras_landmark_impulse_to_fixed_grid():
    fixed_affine=np.diag([2.,3.,4.,1.]);fixed_affine[:3,3]=[30.,-50.,100.]
    moving_affine=np.diag([-1.5,2.,2.5,1.]);moving_affine[:3,3]=[80.,-60.,70.]
    source_ijk=np.array([6,7,8]);target_ijk=np.array([4,5,3])
    source_ras=(moving_affine@np.r_[source_ijk,1])[:3]
    target_ras=(fixed_affine@np.r_[target_ijk,1])[:3]
    forward=np.eye(4);forward[:3,:3]=[[0,-1,0],[1,0,0],[0,0,1]]
    forward[:3,3]=target_ras-forward[:3,:3]@source_ras
    pull_lps=module.FRAME_FLIP@np.linalg.inv(forward)@module.FRAME_FLIP
    transform=sitk.AffineTransform(3);center=np.array([10.,-20.,5.]);transform.SetCenter(tuple(center))
    transform.SetMatrix(tuple(pull_lps[:3,:3].ravel()))
    transform.SetTranslation(tuple(pull_lps[:3,3]-center+pull_lps[:3,:3]@center))
    assert np.allclose(module.forward_ras_from_pull(transform),forward,atol=1e-10)
    assert np.allclose(module.linear_transform_matrix(transform),pull_lps,atol=1e-10)
    moving=np.zeros((16,18,20),np.uint8);moving[tuple(source_ijk)]=1
    fixed=module.image_from_ras(np.zeros((12,14,10),np.float32),fixed_affine)
    sampled=module.warp_native(moving,moving_affine,fixed,transform,sitk.sitkNearestNeighbor)
    assert sampled.shape==(12,14,10)
    assert sampled[tuple(target_ijk)]==1 and sampled.sum()==1
    scalar=module.warp_native(moving.astype(np.float32)*123.5,moving_affine,fixed,transform,sitk.sitkLinear)
    coverage=module.warp_native(np.ones(moving.shape,np.uint8),moving_affine,fixed,transform,sitk.sitkNearestNeighbor)
    assert scalar.dtype==np.float32 and scalar[tuple(target_ijk)]==123.5
    assert coverage[tuple(target_ijk)]==1 and np.any(coverage==0)
    assert np.allclose(forward@np.r_[source_ras,1],np.r_[target_ras,1])
    # An accidental forward/pull swap must put the landmark elsewhere or outside.
    wrong=sitk.AffineTransform(transform.GetInverse())
    wrong_sampled=module.warp_native(moving,moving_affine,fixed,wrong,sitk.sitkNearestNeighbor)
    assert wrong_sampled[tuple(target_ijk)]==0


def test_oblique_image_conversion_preserves_physical_lps_locations():
    angle=.37;rotation=np.array([[np.cos(angle),-np.sin(angle),0],[np.sin(angle),np.cos(angle),0],[0,0,1]])
    affine=np.eye(4);affine[:3,:3]=rotation@np.diag([.7,1.3,2.1]);affine[:3,3]=[71.,-33.,104.]
    image=module.image_from_ras(np.zeros((8,9,10),np.float32),affine)
    for voxel in [(0,0,0),(3,4,5),(7,8,9)]:
        expected=module.FRAME_FLIP@affine@np.r_[voxel,1]
        assert np.allclose(image.TransformIndexToPhysicalPoint(voxel),expected[:3],atol=1e-10)


def test_review_axis_permutation_preserves_boolean_landmark_physical_location():
    affine=np.array([[0,0,1.2,-50],[-.9,0,0,80],[0,1.1,0,-20],[0,0,0,1.]])
    data=np.zeros((4,5,6),bool);data[1,2,3]=True
    result,new_affine=module.canonical_array(data,affine)
    coordinate=np.argwhere(result>0)
    assert result.sum()==1 and len(coordinate)==1
    assert np.allclose(new_affine@np.r_[coordinate[0],1],affine@np.array([1,2,3,1.]))
