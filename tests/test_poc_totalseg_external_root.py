import sys
import types
import nibabel as nib
import numpy as np
import pytest
from resectionlab import poc_models


@pytest.mark.parametrize("setting", ["external", "unset", "empty"])
def test_totalseg_worker_uses_configured_root_and_native_label_semantics(monkeypatch,tmp_path,setting):
    root=tmp_path/'external-models' if setting=="external" else poc_models.ROOT/"data/models"
    output=tmp_path/'output';output.mkdir()
    source=tmp_path/'ct.nii.gz';affine=np.diag([.5,.5,1.,1.])
    nib.save(nib.Nifti1Image(np.zeros((4,4,4),np.float32),affine),source)
    labels=np.zeros((4,4,4),np.uint8);labels[1,1,1]=90;labels[2,2,2]=91
    def infer(*args,**kwargs):
        import os
        assert os.environ['TOTALSEG_HOME_DIR']==str(root/'poc-totalsegmentator/config')
        assert os.environ['TOTALSEG_WEIGHTS_PATH']==str(root/'poc-totalsegmentator/weights')
        return nib.Nifti1Image(labels,affine)
    if setting=="unset": monkeypatch.delenv('RESSECTIONLAB_MODEL_ROOT',raising=False)
    else: monkeypatch.setenv('RESSECTIONLAB_MODEL_ROOT',str(root) if setting=="external" else "")
    monkeypatch.setenv('TOTALSEG_HOME_DIR','before-worker')
    monkeypatch.setenv('TOTALSEG_WEIGHTS_PATH','before-worker')
    monkeypatch.setitem(sys.modules,'torch',types.SimpleNamespace(set_num_threads=lambda n:None))
    monkeypatch.setitem(sys.modules,'totalsegmentator.config',types.SimpleNamespace(setup_totalseg=lambda:None,set_config_key=lambda *a:None))
    monkeypatch.setitem(sys.modules,'totalsegmentator.python_api',types.SimpleNamespace(totalsegmentator=infer))
    poc_models.worker({'device':'cpu','image':str(source),'output':str(output),'model_id':'totalsegmentator-skull'})
    brain=nib.load(output/'brain_mask.nii.gz');skull=nib.load(output/'skull_mask.nii.gz')
    assert brain.get_fdata()[1,1,1]==1 and brain.get_fdata().sum()==1
    assert skull.get_fdata()[2,2,2]==1 and skull.get_fdata().sum()==1
    assert np.array_equal(brain.affine,affine)
