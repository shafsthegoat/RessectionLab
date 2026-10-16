"""Typed-mask source controls; injected doubles do not perform native fitting."""
import importlib.util
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
def load(name,path):
    spec=importlib.util.spec_from_file_location(name,ROOT/path)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module);return module

v3=load("typed_mask_v3","scripts/poc_ixi265_atropos_typed_mask.py")
controls=load("frozen_v1_controls","tests/test_poc_ixi265_atropos.py")
v2controls=load("frozen_v2_controls","tests/test_poc_ixi265_atropos_positive_domain.py")


class TypedBackend(controls.GeneratedBackend):
    def from_numpy(self,data,**geometry):
        image=super().from_numpy(data,**geometry)
        image.pixeltype="float" if data.dtype==np.float32 else "unsigned char"
        return image
    def n4_bias_field_correction(self,image,*,mask,**parameters):
        assert mask.pixeltype=="float" and mask.numpy().dtype==np.float32
        return super().n4_bias_field_correction(image,mask=mask,**parameters)


def test_typed_mask_preserves_values_frames_source_and_exact_v2_unknown_partition():
    values,brain,_,affine=v2controls.generated();old=v3.previous()
    domain,unknown,_=old.positive_domain(values,brain,expected_unknown=v2controls.UNKNOWN)
    before=[a.copy() for a in (values,brain,domain,unknown,affine)]
    backend=TypedBackend();stages=[]
    corrected,labels,posteriors,audit=v3.fit_arrays(values,domain,affine,backend,
        stage_callback=lambda name,details:stages.append(name))
    old.require_unknown_partition(brain,domain,unknown,labels,posteriors)
    assert stages==["N4_start","N4_complete","Atropos_start","Atropos_complete"]
    assert audit["N4_mask_boundary"]["native_pixeltype"]=="float"
    assert audit["N4_mask_boundary"]["support_voxels"]==int(domain.sum())
    for actual,expected in zip((values,brain,domain,unknown,affine),before):assert np.array_equal(actual,expected)
    for code,p in enumerate(posteriors,1):
        assert np.array_equal(p,np.where(domain,np.where(labels==code,.8,.1),0).astype(np.float32))


@pytest.mark.parametrize("fault",["native_type","dtype","binary_value","origin","shape"])
def test_bad_mask_cannot_cross_N4_boundary(fault):
    values,mask,_,affine=controls.generated()
    class BadBackend(TypedBackend):
        def from_numpy(self,data,**geometry):
            image=super().from_numpy(data,**geometry)
            if np.isin(data,[0,1]).all():
                if fault=="native_type":image.pixeltype="unsigned char"
                if fault=="dtype":image.data=image.data.astype(np.float64)
                if fault=="binary_value":image.data[0,1,0]=0
                if fault=="origin":image.origin=(0,0,0)
                if fault=="shape":image.shape=(8,8,7)
            return image
    backend=BadBackend()
    with pytest.raises(ValueError):v3.fit_arrays(values,mask,affine,backend)
    assert backend.calls==[]


def test_unmodified_raw_probability_guard_rejects_malformed_backend():
    values,mask,_,affine=controls.generated()
    class BadPosterior(TypedBackend):
        def atropos(self,**kwargs):
            out=super().atropos(**kwargs);out['probabilityimages'][0].data[0,1,0]=1.1;return out
    with pytest.raises(ValueError,match="posterior"):v3.fit_arrays(values,mask,affine,BadPosterior())


def test_new_protocol_preserves_parameters_source_support_and_unreleased_status():
    method=v3.method();old=v3.previous().method()
    assert method['parameters']==old['parameters']
    assert method['source_hashes']==old['source_hashes']
    assert method['expected_unknown_indices_native']==old['expected_unknown_indices_native']
    assert method['resources']==old['resources']
    assert not method['patient_fitting_released']
    assert method['new_kernel_native_preflight_required_before_any_patient_release']
    assert method['single_attempt_marker']!=old['single_attempt_marker']
    assert not method['anatomy_accepted'] and not method['growth_use_accepted']
