"""Constructed format/receipt fixtures only; no patient file/network reads."""
import gzip
import hashlib
import importlib.util
import io
import json
from pathlib import Path
import struct

import nibabel as nib
import numpy as np
import pytest

PATH = Path(__file__).resolve().parents[1] / 'scripts/inspect_rhuh_single_image.py'
SPEC = importlib.util.spec_from_file_location('rhuh_inspector_owner', PATH)
mod = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(mod)


def image_bytes(version=1, dtype=np.int16, shape=(3, 4, 5), *, endian='<', extension=None):
    cls = nib.Nifti1Header if version == 1 else nib.Nifti2Header
    h = cls(endianness=endian)
    h.set_data_shape(shape)
    h.set_data_dtype(dtype)
    h.set_xyzt_units('mm')
    h.set_qform(np.diag([1., 2., 3., 1.]), code=1)
    h.set_sform(np.diag([1., 2., 3., 1.]), code=1)
    extra = b'' if extension is None else extension
    h['vox_offset'] = h.sizeof_hdr + 4 + len(extra)
    h['descrip'] = b'PRIVATE HEADER TEXT MUST NOT ESCAPE'
    a = np.arange(np.prod(shape), dtype=dtype).reshape(shape, order='F')
    payload = a.astype(h.get_data_dtype()).tobytes(order='F')
    return h, h.binaryblock + (b'\x00'*4 if extension is None else b'\x01\x00\x00\x00') + extra + payload


@pytest.mark.parametrize('version,endian',[(1,'<'),(1,'>'),(2,'<'),(2,'>')])
def test_valid_3d_formats_have_exact_storage_and_safe_summary(version,endian):
    _, raw = image_bytes(version=version,endian=endian)
    result = mod.inspect_stream(io.BytesIO(gzip.compress(raw)))
    assert result['shape'] == [3,4,5]
    assert result['nifti_version'] == version
    assert result['intensity']['scaled_finite_min'] == 0
    assert result['intensity']['scaled_finite_max'] == 59
    assert result['intensity']['scaled_finite_mean'] == 29.5
    assert result['geometry']['issues'] == []
    assert result['scientific_use_released'] is False
    assert 'PRIVATE' not in json.dumps(result)


def test_scaling_and_nonfinite_are_reported_after_valid_stream():
    h, raw = image_bytes(dtype=np.float32)
    h['scl_slope'],h['scl_inter'] = 2.,10.
    a=np.arange(60,dtype='<f4'); a[0]=np.nan;a[1]=np.inf
    result=mod.inspect_stream(io.BytesIO(gzip.compress(h.binaryblock+b'\0'*4+a.tobytes())))
    assert result['intensity']['scaled_nonfinite_voxels']==2
    assert result['intensity']['scaled_finite_min']==14.
    assert result['binary_structure_valid'] is True


def test_unknown_units_and_absent_world_frame_not_binary_corruption():
    h,raw=image_bytes();h.set_xyzt_units('unknown');h['qform_code']=h['sform_code']=0
    result=mod.inspect_stream(io.BytesIO(gzip.compress(h.binaryblock+raw[348:])))
    assert result['binary_structure_valid']
    assert 'physical_units_unresolved' in result['geometry']['issues']
    assert 'world_transform_absent_no_fallback_assumed' in result['geometry']['issues']
    assert result['geometry']['qform']['affine_in_source_units'] is None


def test_conflicting_frames_keep_numeric_and_frame_difference_distinct():
    h,raw=image_bytes(); a=h.get_sform();a[0,3]=5;h.set_sform(a,code=2)
    result=mod.inspect_stream(io.BytesIO(gzip.compress(h.binaryblock+raw[348:])))
    g=result['geometry'];assert g['max_full_cell_corner_difference_mm']==pytest.approx(5)
    assert 'qform_sform_frame_codes_differ' in g['issues']
    assert 'qform_sform_numeric_disagreement' in g['issues']
    assert not g['anatomical_registration_accepted']


@pytest.mark.parametrize('edit,reason',[
    (lambda h:h.__setitem__('dim',[4,3,4,5,2,1,1,1]),'shape'),
    (lambda h:h.__setitem__('dim',[3,513,4,5,1,1,1,1]),'shape'),
    (lambda h:h.__setitem__('dim',[3,512,512,512,1,1,1,1]),'voxel_count_cap'),
    (lambda h:h.__setitem__('datatype',32),'datatype'),
    (lambda h:h.__setitem__('bitpix',8),'bitpix'),
    (lambda h:h.__setitem__('vox_offset',352.5),'offset'),
    (lambda h:h.__setitem__('vox_offset',1048577),'offset'),
    (lambda h:h.__setitem__('vox_offset',348),'offset'),
])
def test_malformed_header_refused_before_voxel_summary(edit,reason,monkeypatch):
    h,raw=image_bytes();edit(h)
    monkeypatch.setattr(mod,'_intensity',lambda *args:pytest.fail('voxel interpretation before validation'))
    with pytest.raises(mod.Rejected,match=reason):mod.inspect_stream(io.BytesIO(gzip.compress(h.binaryblock+raw[348:])))


@pytest.mark.parametrize('alter,reason',[
    (lambda b:b[:-1],'truncated'),
    (lambda b:b+gzip.compress(b''),'trailing_or_multimember'),
    (lambda b:b+b'junk','trailing_or_multimember'),
    (lambda b:b[:-8]+bytes([b[-8]^1])+b[-7:],'crc'),
])
def test_invalid_gzip_never_publishes_voxel_summary(alter,reason,monkeypatch):
    _,raw=image_bytes()
    monkeypatch.setattr(mod,'_intensity',lambda *args:pytest.fail('footer was not validated'))
    with pytest.raises(mod.Rejected,match=reason):mod.inspect_stream(io.BytesIO(alter(gzip.compress(raw))))


@pytest.mark.parametrize('extra',[-2,2])
def test_exact_uncompressed_length_required(extra):
    _,raw=image_bytes();raw=raw[:extra] if extra<0 else raw+b'\0'*extra
    with pytest.raises(mod.Rejected):mod.inspect_stream(io.BytesIO(gzip.compress(raw)))


def test_extension_bytes_are_opaque_and_bounded():
    extension=struct.pack('<ii',32,6)+b'PRIVATE EXTENSION TEXT!\0'
    assert len(extension)==32
    _,raw=image_bytes(extension=extension)
    r=mod.inspect_stream(io.BytesIO(gzip.compress(raw)))
    assert r['extension_count']==1 and 'PRIVATE' not in json.dumps(r)


def bind(root, path, content):
    p=root/path;p.parent.mkdir(parents=True,exist_ok=True)
    b=content if isinstance(content,bytes) else json.dumps(content,sort_keys=True).encode()
    p.write_bytes(b)
    return {'path':str(path),'sha256':hashlib.sha256(b).hexdigest()}


def request_fixture(root,monkeypatch):
    _,raw=image_bytes();compressed=gzip.compress(raw,mtime=0)
    md5=hashlib.md5(compressed).hexdigest();monkeypatch.setattr(mod,'PUBLISHED_TOKEN',md5)
    index=bind(root,'artifacts/fixture-index.sums',f'{md5} {mod.SOURCE[1:]}\n'.encode())
    monkeypatch.setattr(mod,'INDEX_SHA256',index['sha256'])
    proposal=bind(root,'artifacts/fixture-proposal.json',{'selection':{'public_source_path':mod.SOURCE,'patient':'RHUH-0001','visit':0,'modality_source_label':'T1','published_checksum_token':md5,'checksum_index_sha256':index['sha256']}})
    monkeypatch.setattr(mod,'PROPOSAL_SHA256',proposal['sha256'])
    payload={'path':'outputs/rhuh-single-image-v2/quarantine/run-fixture/RHUH-0001_0_t1.nii.gz',
             'measured_compressed_bytes':len(compressed),'expected_compressed_bytes':None,
             'prior_length_match_claim':False,'sha256':hashlib.sha256(compressed).hexdigest(),
             'candidate_md5_of_original_compressed_bytes':md5,'published_checksum_token':md5,
             'candidate_matches_published_token':True,'publisher_algorithm_confirmed':False,
             'acceptance_scope':'format_byte_domain_compatibility_only','decoded':False,'scientific_use_released':False}
    worker=bind(root,'outputs/rhuh-single-image-v2/quarantine/run-fixture-receipt.json',{
        'status':mod.SUCCESS,'source':mod.SOURCE,'proposal_sha256':proposal['sha256'],
        'checksum_index_sha256':index['sha256'],'resolved_source_sha256':mod.RESOLVED_SOURCE_SHA256,
        'client_exit_code':0,'transfer_started':True,'decoded':False,'scientific_use_released':False,'payload':payload})
    parent=bind(root,'artifacts/parent-summary.json',{'accepted':True,'status':mod.SUCCESS,'receipt':worker['path'],'final_payload':payload})
    sources={k:bind(root,'artifacts/'+k+'.py',b'# constructed inert source') for k in ('image_helper','public_transfer','checksum_helper')}
    release=bind(root,'artifacts/transfer-release.json',{'fixture':'constructed only'})
    execution=bind(root,'artifacts/parent-execution.json',{'exit_code':0,'parent_summary_sha256':parent['sha256'],'parent_summary':json.loads((root/parent['path']).read_bytes()),'release_sha256':release['sha256'],'source_closure_unchanged':True})
    review={'schema':'resectionlab.rhuh-compressed-byte-reconciliation.v1','accepted':True,'source':mod.SOURCE,
        'proposal_sha256':proposal['sha256'],'checksum_index_sha256':index['sha256'],
        'resolved_source_sha256':mod.RESOLVED_SOURCE_SHA256,'parent_summary':parent,'parent_execution':execution,
        'worker_receipt':worker,'payload':payload,'immutable_original':True,'publisher_algorithm_confirmed':False,
        'prior_length_match_claim':False,'reviewer_source':bind(root,'artifacts/reviewer.py',b'# analytical fixture'),
        'acquisition_sources':sources,'transfer_release':release,'decoded':False,'scientific_use_released':False}
    review_binding=bind(root,'artifacts/review.json',review)
    request={'schema':'resectionlab.rhuh-image-inspection-request.v1','source':mod.SOURCE,'proposal':proposal,
             'checksum_index':index,'parent_summary':parent,'parent_execution':execution,'worker_receipt':worker,
             'reconciliation':review_binding,'payload':payload,'acquisition_sources':sources,'transfer_release':release}
    return request, compressed


def test_preflight_opens_only_metadata_without_payload_existing(tmp_path,monkeypatch):
    request,_=request_fixture(tmp_path,monkeypatch)
    assert not (tmp_path/request['payload']['path']).exists()
    result=mod.preflight(tmp_path,request)
    assert result['patient_bytes_opened'] is False


@pytest.mark.parametrize('which,field,value',[
    ('parent_summary','accepted',False),('worker_receipt','resolved_source_sha256','0'*64),
    ('reconciliation','immutable_original',False),('reconciliation','publisher_algorithm_confirmed',True),
    ('parent_execution','exit_code',1),
])
def test_rehashed_foreign_or_unaccepted_receipt_fails(tmp_path,monkeypatch,which,field,value):
    request,_=request_fixture(tmp_path,monkeypatch)
    obj=json.loads((tmp_path/request[which]['path']).read_bytes());obj[field]=value
    request[which]=bind(tmp_path,request[which]['path'],obj)
    with pytest.raises(mod.Rejected):mod.preflight(tmp_path,request)


def test_release_then_identity_are_checked_before_analytical_decode(tmp_path,monkeypatch):
    request,compressed=request_fixture(tmp_path,monkeypatch)
    req=bind(tmp_path,'artifacts/request.json',request)
    release={'schema':'resectionlab.rhuh-image-inspection-release.v1','released':False,
        'action':'bounded_header_and_voxel_inspection_once','request':req,'source':mod.SOURCE,
        'inspector_source_sha256':hashlib.sha256(PATH.read_bytes()).hexdigest()}
    rel=bind(tmp_path,'artifacts/release.json',release)
    with pytest.raises(mod.Rejected,match='release'):mod.inspect_authorized(tmp_path,req,rel)
    bind(tmp_path,request['payload']['path'],compressed)
    release['released']=True;rel=bind(tmp_path,'artifacts/release.json',release)
    r=mod.inspect_authorized(tmp_path,req,rel)
    assert r['binary_structure_valid'] and r['bindings']['request']==req
    (tmp_path/request['payload']['path']).write_bytes(b'x'*len(compressed))
    with pytest.raises(mod.Rejected,match='identity'):mod.inspect_authorized(tmp_path,req,rel)


def test_bound_symlink_is_rejected(tmp_path,monkeypatch):
    request,_=request_fixture(tmp_path,monkeypatch)
    p=tmp_path/request['parent_summary']['path'];original=p.read_bytes();p.unlink()
    q=tmp_path/'outside';q.write_bytes(original);p.symlink_to(q)
    with pytest.raises(mod.Rejected,match='symlink'):mod.preflight(tmp_path,request)


@pytest.mark.parametrize('cap,value,reason',[
    ('MAX_VOXEL_BYTES',100,'voxel_storage_cap'),('MAX_UNCOMPRESSED',400,'total_uncompressed_cap'),
    ('MAX_COMPRESSED',10,'compressed_cap'),
])
def test_storage_caps_fail_without_voxel_interpretation(monkeypatch,cap,value,reason):
    _,raw=image_bytes();monkeypatch.setattr(mod,cap,value)
    monkeypatch.setattr(mod,'_intensity',lambda *args:pytest.fail('premature voxel access'))
    with pytest.raises(mod.Rejected,match=reason):mod.inspect_stream(io.BytesIO(gzip.compress(raw)))


def test_first_header_peek_stays_within_544_bytes(monkeypatch):
    h,raw=image_bytes(version=2);h['dim']=[3,513,2,2,1,1,1,1]
    calls=[];original=mod._GzipReader.exact
    def tracked(self,count):calls.append(count);return original(self,count)
    monkeypatch.setattr(mod._GzipReader,'exact',tracked)
    with pytest.raises(mod.Rejected,match='shape'):mod.inspect_stream(io.BytesIO(gzip.compress(h.binaryblock+raw[540:])))
    assert calls==[4,536,4] and sum(calls)==544


def test_complex_color_and_pair_formats_refused():
    h,raw=image_bytes()
    for code in (32,128,1792,2048,2304):
        h['datatype']=code
        with pytest.raises(mod.Rejected,match='datatype'):mod.inspect_stream(io.BytesIO(gzip.compress(h.binaryblock+raw[348:])))
    h['datatype']=4;h['magic']=b'ni1\0'
    with pytest.raises(mod.Rejected,match='magic'):mod.inspect_stream(io.BytesIO(gzip.compress(h.binaryblock+raw[348:])))


def test_extension_size_zero_or_unaligned_refused_without_payload_text():
    for size in (0,8,17,64):
        extension=struct.pack('<ii',size,6)+b'x'*24
        _,raw=image_bytes(extension=extension)
        with pytest.raises(mod.Rejected,match='extension_length'):mod.inspect_stream(io.BytesIO(gzip.compress(raw)))


def test_preserved_worker_copy_matches_original_and_documentary_extras(tmp_path,monkeypatch):
    request,_=request_fixture(tmp_path,monkeypatch)
    original=request['worker_receipt'].copy()
    request['worker_receipt']=bind(tmp_path,'artifacts/worker-receipt.json',(tmp_path/original['path']).read_bytes())
    review=json.loads((tmp_path/request['reconciliation']['path']).read_bytes())
    review['worker_receipt']=request['worker_receipt'];review['immutable_original_meaning']='preserved hash-bound bytes'
    request['reconciliation']=bind(tmp_path,request['reconciliation']['path'],review)
    assert mod.preflight(tmp_path,request)['patient_bytes_opened'] is False
    (tmp_path/original['path']).write_bytes(b'changed original receipt')
    with pytest.raises(mod.Rejected,match='metadata_hash'):mod.preflight(tmp_path,request)


def test_unknown_review_extra_is_not_an_authority_extension(tmp_path,monkeypatch):
    request,_=request_fixture(tmp_path,monkeypatch)
    review=json.loads((tmp_path/request['reconciliation']['path']).read_bytes())
    review['auto_approve_anatomy']=True
    request['reconciliation']=bind(tmp_path,request['reconciliation']['path'],review)
    with pytest.raises(mod.Rejected,match='reconciliation'):mod.preflight(tmp_path,request)
