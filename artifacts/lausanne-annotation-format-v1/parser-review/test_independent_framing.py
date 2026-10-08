"""Framing controls only. These bytes are not images, labels or patient fixtures."""
import hashlib
import importlib
from pathlib import Path
import struct
import sys
import pytest
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'scripts'))
intake=importlib.import_module('lausanne_annotation_intake')

@pytest.fixture(autouse=True)
def no_scientific_access(monkeypatch):
    def forbidden(*args,**kwargs):pytest.fail('No scientific payload or network in framing controls')
    monkeypatch.setattr(intake,'open_without_redirect',forbidden)
    monkeypatch.setattr(intake.gzip,'open',forbidden)
    original=Path.open
    def guarded(path,*args,**kwargs):
        if str(path).endswith(('.nii','.nii.gz','.nii.gz.partial')):forbidden()
        return original(path,*args,**kwargs)
    monkeypatch.setattr(Path,'open',guarded)

def parse(region,endian='<',maximum=65536):
    return intake.inspect_extensions(region,data_offset=len(region)+348,endian=endian,maximum_offset=maximum)

def frame(size,code,endian='<',payload=None):
    return struct.pack(endian+'ii',size,code)+(bytes(size-8) if payload is None else payload)

@pytest.mark.parametrize('endian',['<','>'])
@pytest.mark.parametrize('code',[0,1,2,2147483647])
def test_signed_nonnegative_code_and_opaque_binary_payload(endian,code):
    private=b'\xff\xfe\0<xml>\0' + bytes(15)
    assert len(private)==24
    block=frame(32,code,endian,payload=private)
    result=parse(b'\xff\0\0\0'+block,endian)
    record=result['records'][0]
    assert record['ecode']==code and record['payload_bytes']==len(private)
    assert record['payload_sha256']==hashlib.sha256(private).hexdigest()
    assert record['block_sha256']==hashlib.sha256(block).hexdigest()
    assert record['offset']==352 and record['end_exclusive']==384
    assert set(record)=={'offset','end_exclusive','esize','ecode','block_sha256','payload_bytes','payload_sha256'}
    assert not result['payloads_interpreted'] and not result['used_as_annotation_or_coordinate_evidence']

@pytest.mark.parametrize('endian',['<','>'])
@pytest.mark.parametrize('size,code',[(0,0),(-16,0),(-2147483648,0),(2147483632,0),(2147483647,0),(16,-2147483648),(16,-1)])
def test_signed_boundary_fields_refuse_without_allocation(endian,size,code):
    region=b'\1\0\0\0'+struct.pack(endian+'ii',size,code)+bytes(8)
    with pytest.raises(intake.AcquisitionError):parse(region,endian)

@pytest.mark.parametrize('tail',[bytes(1),bytes(7),bytes(8),bytes(15),bytes(16),b'content'])
def test_completed_chain_rejects_every_unframed_trailer(tail):
    with pytest.raises(intake.AcquisitionError):parse(b'\1\0\0\0'+frame(16,0)+tail)

@pytest.mark.parametrize('endian',['<','>'])
def test_wrong_endian_and_corrupt_second_record_refused(endian):
    opposite='>' if endian=='<' else '<'
    with pytest.raises(intake.AcquisitionError):parse(b'\1\0\0\0'+frame(16,0,endian),opposite)
    with pytest.raises(intake.AcquisitionError):parse(b'\1\0\0\0'+frame(16,0,endian)+struct.pack(endian+'ii',16,-1)+bytes(8),endian)


def test_maximum_legal_chain_has_bounded_record_and_receipt_size():
    count=(65536-352)//16
    region=b'\1\0\0\0'+frame(16,0)*count
    assert 348+len(region)==65536 and count==4074
    result=parse(region)
    assert result['extension_count']==count and result['records'][-1]['end_exclusive']==65536
    assert len(intake.encoded({'content_qc':{'extensions':result}}))<2*1024**2
    with pytest.raises(intake.AcquisitionError):parse(region+frame(16,0))


def test_single_maximum_record_and_absent_padding_bounds():
    region=b'\1\0\0\0'+frame(65536-352,2147483647)
    result=parse(region)
    assert result['extension_count']==1 and result['records'][0]['payload_bytes']==65176
    absent=parse(bytes(65536-348))
    assert absent['status']=='absent_zero_padding' and absent['extension_count']==0
    # Offset alignment for extension-absent zero padding is unchanged. The spec
    # recommends 16-byte data offsets; it requires 16-byte extension record sizes.
    assert parse(bytes(5))['status']=='absent_zero_padding'
    with pytest.raises(intake.AcquisitionError):parse(bytes(65536-347))
