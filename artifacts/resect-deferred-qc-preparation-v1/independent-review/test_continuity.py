"""Exception routing only: all source/header readers replaced; no patient I/O."""
import importlib
from pathlib import Path
import socket
import sys
import time
import pytest
ROOT=Path(__file__).resolve().parents[2]
sys.path[:0]=[str(ROOT/'scripts'),str(ROOT/'src')]

@pytest.fixture
def seams(monkeypatch):
    q=importlib.import_module('resect_deferred_qc')
    imaging=importlib.import_module('resectionlab.imaging')
    monkeypatch.setattr(socket,'create_connection',lambda *a,**k:pytest.fail('network forbidden'))
    monkeypatch.setattr(q,'check_prefix',lambda *a:{'control_only':True})
    monkeypatch.setattr(q.streaming,'inspect_original',lambda *a:{'header_qc':{'status':'failed'}})
    monkeypatch.setattr(q.streaming.annotations,'inspect_mask',lambda *a:{'status':'passed','positive_voxels':3,'background_voxels':5})
    row={'image':{'path':'control-image-never-opened'},'mask':{'path':'control-mask-never-opened'},
         'image_sha256':'a'*64,'mask_sha256':'b'*64}
    return q,imaging,row

def test_late_mask_geometry_error_retains_returned_content(seams,monkeypatch):
    from nibabel.spatialimages import HeaderDataError
    q,imaging,row=seams
    def fail(*a):raise HeaderDataError('control unresolved transform')
    monkeypatch.setattr(imaging,'inspect_nifti',fail)
    r=q.inspect_pair(row,{},time.monotonic()+10)
    assert r['status']=='review_failed'
    assert r['mask_qc']=={'status':'passed','positive_voxels':3,'background_voxels':5}
    assert r['mask_geometry']['classification']=='header_interpretation_unresolved'
    assert r['mask_geometry']['source_anatomical_validity']=='not_assessed'
    assert r['mask_geometry']['operation_stage']=='mask_geometry_review'
    assert r['pair_geometry']['status']=='not_run'

def test_early_unknown_header_error_does_not_assert_source_invalid(seams,monkeypatch):
    from nibabel.spatialimages import HeaderDataError
    q,imaging,row=seams
    def fail(*a):raise HeaderDataError('control parser failure')
    monkeypatch.setattr(q,'check_prefix',fail)
    r=q.inspect_pair(row,{},time.monotonic()+10)
    assert r['status']=='review_failed'
    for kind in ('image','mask'):
        r0=r[kind+'_qc']
        assert r0['classification']=='header_interpretation_unresolved'
        assert r0['source_anatomical_validity']=='not_assessed'
        assert r0['operation_stage']==r0['stage']=='header_prefix'
        assert all(r0[k]['status']=='unknown_not_returned' for k in ('header_qc','scalar_qc','geometry_qc'))

def test_deadline_after_header_limitation_still_aborts(seams,monkeypatch):
    from nibabel.spatialimages import HeaderDataError
    q,imaging,row=seams
    def fail(*a):raise HeaderDataError('control parser failure')
    def expired(*a):raise TimeoutError('control expired')
    monkeypatch.setattr(q,'check_prefix',fail)
    monkeypatch.setattr(q,'check_deadline',expired)
    with pytest.raises(TimeoutError,match='control expired'):
        q.inspect_pair(row,{},time.monotonic()+10)

def test_unexpected_implementation_error_still_aborts(seams,monkeypatch):
    q,imaging,row=seams
    def fail(*a):raise RuntimeError('control implementation failure')
    monkeypatch.setattr(q.streaming,'inspect_original',fail)
    with pytest.raises(RuntimeError,match='control implementation failure'):
        q.inspect_pair(row,{},time.monotonic()+10)
