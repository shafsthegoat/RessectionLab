"""Physical-frame controls for the measured SynthSeg memory remedy."""
import importlib.util
from pathlib import Path
import nibabel as nib
import numpy as np
import pytest

spec=importlib.util.spec_from_file_location('poc_synthseg_inference',Path(__file__).resolve().parents[1]/'scripts/poc_synthseg_inference.py')
module=importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


def test_oblique_brain_roi_preserves_source_cells_and_physical_landmarks():
    angle=.3
    rotation=np.array([[np.cos(angle),-np.sin(angle),0],[np.sin(angle),np.cos(angle),0],[0,0,1.]])
    affine=np.eye(4);affine[:3,:3]=rotation@np.diag([-2.,3.,4.]);affine[:3,3]=[80.,-60.,12.]
    values=np.arange(16*18*14,dtype=np.float32).reshape(16,18,14)
    source=nib.Nifti1Image(values,affine);source.set_qform(affine,1);source.set_sform(affine,1)
    before=values.copy();mask=np.zeros(values.shape,bool);mask[5:8,8:10,4:7]=True
    cropped,coverage,record=module.brain_roi(source,mask)
    assert record['bounds_native_exclusive']==[(2,11),(6,12),(2,9)]
    assert cropped.shape==(9,6,7)
    assert np.array_equal(np.asarray(cropped.dataobj),values[2:11,6:12,2:9])
    assert np.array_equal(values,before) and np.all(coverage[mask]==1)
    # Two independent original-grid landmarks retain both intensity and world location.
    for original_index in [(5,8,4),(7,9,6)]:
        local=np.asarray(original_index)-[2,6,2]
        assert np.asarray(cropped.dataobj)[tuple(local)]==values[original_index]
        expected=affine[:3,:3]@original_index+affine[:3,3]
        actual=cropped.affine[:3,:3]@local+cropped.affine[:3,3]
        assert np.allclose(actual,expected,atol=1e-6)


def test_empty_or_unmatched_roi_mask_refuses_before_crop():
    source=nib.Nifti1Image(np.zeros((8,9,10),np.float32),np.eye(4))
    for invalid in [np.zeros(source.shape,bool),np.ones((8,9,11),bool),np.ones(source.shape,np.uint8)]:
        with pytest.raises(ValueError,match='binary brain mask'):
            module.brain_roi(source,invalid)
