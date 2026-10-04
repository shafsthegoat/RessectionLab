"""Small physical-geometry checks; no model training or patient registration."""
import importlib.util
from pathlib import Path
import nibabel as nib
import numpy as np
import pytest
import SimpleITK as sitk

path=Path(__file__).resolve().parents[1]/'scripts/assess_remind_rigid_feasibility.py'
spec=importlib.util.spec_from_file_location('remind_rigid_feasibility',path)
profile=importlib.util.module_from_spec(spec);spec.loader.exec_module(profile)


def test_ras_matrix_preserves_centered_euler_lps_transform_and_inverse():
    transform=sitk.Euler3DTransform();transform.SetCenter((4.,-8.,12.));transform.SetRotation(.13,-.21,.17);transform.SetTranslation((3.,-2.,7.))
    matrix=profile.matrix(transform);inverse=np.linalg.inv(matrix)
    for point in ((0.,0.,0.),(14.,-5.,19.),(-23.,41.,7.)):
        ras=np.asarray(point);lps=ras*np.array([-1.,-1.,1.])
        expected=np.asarray(transform.TransformPoint(tuple(lps)))*[-1.,-1.,1.]
        np.testing.assert_allclose((matrix@np.r_[ras,1])[:3],expected,atol=1e-13)
        np.testing.assert_allclose((inverse@np.r_[expected,1])[:3],ras,atol=1e-13)


def test_qform_conversion_tracks_ras_lps_origin_orientation_and_coverage(tmp_path):
    affine=np.array([[0.,-2.,0.,17.],[-1.5,0.,0.,-11.],[0.,0.,3.,9.],[0.,0.,0.,1.]])
    source=nib.Nifti1Image(np.arange(8*10*12,dtype=np.float32).reshape(8,10,12),affine)
    source.set_qform(affine,1);source.set_sform(affine,1);source.header.set_xyzt_units('mm')
    path=tmp_path/'geometry.nii.gz';nib.save(source,path)
    image,coverage,record=profile.image_on_qform(path)
    np.testing.assert_allclose(image.GetOrigin(),[-17.,11.,9.],atol=1e-6)
    np.testing.assert_allclose(image.TransformIndexToPhysicalPoint((1,0,0)),[-17.,14.,9.],atol=1e-6)
    assert image.GetSize()==coverage.GetSize() and record['normalized_using_this_image_only']
    assert record['coarse_spacing_mm']==[3.,3.,3.]


def test_qform_material_physical_disagreement_rejected(tmp_path):
    data=np.arange(8**3,dtype=np.float32).reshape(8,8,8);image=nib.Nifti1Image(data,np.eye(4))
    image.set_qform(np.eye(4),1); shifted=np.eye(4);shifted[0,3]=.01;image.set_sform(shifted,1)
    path=tmp_path/'disagree.nii.gz';nib.save(image,path)
    with pytest.raises(ValueError,match='differs materially'):
        profile.image_on_qform(path)
