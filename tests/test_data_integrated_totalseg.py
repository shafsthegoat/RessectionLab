"""Test the installed documented array convention without any model or download."""
import importlib.util
from pathlib import Path
import nibabel as nib
import numpy as np
import pytest
spec=importlib.util.spec_from_file_location('single_array_adapter',Path(__file__).parents[1]/'scripts/data_integrated_totalseg.py')
module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)

def test_single_array_axis_spacing_and_return_grid(tmp_path):
    source=tmp_path/'input';destination=tmp_path/'output';source.mkdir();destination.mkdir()
    affine=np.diag([-2.,3.,4.,1.]);affine[:3,3]=[31.,-9.,12.]
    values=np.arange(60,dtype=np.float32).reshape(3,4,5)
    image=nib.Nifti1Image(values,affine);image.header.set_xyzt_units('mm');nib.save(image,source/'s01_0000.nii.gz')
    class Predictor:
        def predict_single_npy_array(self,data,props,*args):
            assert np.array_equal(data[0],values.transpose(2,1,0))
            assert props=={'spacing':(4.,3.,2.)} and args==(None,None,False)
            return (data[0]%3).astype(np.uint8)
    module.predict_one_file(Predictor(),source,destination)
    result=nib.load(destination/'s01.nii.gz')
    assert np.array_equal(result.affine,affine) and np.array_equal(np.asanyarray(result.dataobj),values%3)
    assert result.header.get_xyzt_units()[0]=='mm'

def test_cascade_and_probabilities_refuse_before_source_access(tmp_path):
    for kwargs in [{'save_probabilities':True},{'folder_with_segs_from_prev_stage':'unknown'},{'num_parts':2}]:
        with pytest.raises(ValueError,match='noncascade'):module.predict_one_file(None,tmp_path,tmp_path,**kwargs)

def test_wrong_array_shape_is_not_hidden_by_copying_an_affine(tmp_path):
    source=tmp_path/'source';source.mkdir()
    nib.save(nib.Nifti1Image(np.zeros((3,4,5),np.float32),np.eye(4)),source/'s01_0000.nii.gz')
    class WrongPredictor:
        def predict_single_npy_array(self,*args):return np.zeros((3,4,5),np.uint8)
    with pytest.raises(ValueError,match='exact input grid'):module.predict_one_file(WrongPredictor(),source,tmp_path/'out')
    assert not (tmp_path/'out/s01.nii.gz').exists()
