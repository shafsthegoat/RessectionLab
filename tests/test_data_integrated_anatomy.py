import importlib.util
from pathlib import Path
import numpy as np
import pytest

spec = importlib.util.spec_from_file_location('integrated_anatomy', Path(__file__).parents[1]/'scripts/data_integrated_anatomy.py')
module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)


def test_native_enclosure_excludes_dense_vessel_and_unseen_open_slices():
    x,y=np.ogrid[:65,:65];radius=np.sqrt((x-32)**2+(y-32)**2)
    ct=np.full((65,65,20),-1000.,np.float32)
    ct[radius<27,:]=35
    ct[(radius>=24)&(radius<=27),:]=1000
    vessels=np.zeros_like(ct,np.uint8);vessels[31:34,31:34,:]=1;ct[vessels>0]=800
    # A truly open lower slice cannot be closed by the 2 mm kernel.
    ct[32:,:10,0:3]=-1000
    result=module.estimate(ct,vessels,(1.,1.,1.))
    assert result['brain_estimate'][20,32,10]
    assert not result['observed_skull'][vessels>0].any()
    assert not result['tumor_placement_support'][vessels>0].any()
    assert np.all(result['tumor_placement_support'] <= result['brain_estimate'])
    assert not result['enclosed_coverage'][:,:,0:3].any()
    assert result['observed_skull'][8,32,10]
    assert all(mask.shape==ct.shape for mask in result.values())


def test_physical_bounds_use_voxel_faces_and_real_affine():
    affine=np.diag([-2.,3.,4.,1.]);affine[:3,3]=[10,20,30]
    bounds=module.physical_bounds((5,6,7),affine)
    assert bounds['minimum_ras_mm']==[1.,18.5,28.]
    assert bounds['maximum_ras_mm']==[11.,36.5,56.]


def test_no_enclosed_anatomy_fails_without_fabrication():
    with pytest.raises(ValueError,match='No enclosed'):
        module.estimate(np.zeros((10,10,10)),np.zeros((10,10,10)),(1,1,1))


def test_existing_default_version_refuses_before_source_decode_or_estimation(monkeypatch,tmp_path):
    destination=tmp_path/'integrated-v1/anatomy/TopCoW-006/enclosure-v1';destination.mkdir(parents=True)
    monkeypatch.setattr(module,'checked_root',lambda *a,**k:tmp_path)
    monkeypatch.setattr(module,'estimate',lambda *a,**k:pytest.fail('Existing immutable output must not be estimated again'))
    with pytest.raises(FileExistsError,match='Immutable output exists'):
        module.prepare(tmp_path,'006')


@pytest.mark.parametrize('version',['../escape','bad/name','has space',''])
def test_unsafe_version_is_rejected_before_external_root_access(monkeypatch,version):
    monkeypatch.setattr(module,'checked_root',lambda *a,**k:pytest.fail('Invalid version must not access data root'))
    with pytest.raises(ValueError,match='safe immutable version'):
        module.prepare('/unused','006',version=version)
