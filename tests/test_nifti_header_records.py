"""Pure 348-byte header controls; no image or voxel payload is opened."""
import base64
import hashlib
import json
from pathlib import Path

import nibabel as nib
import pytest

from resectionlab.critical_evidence import nifti1_header_record
from resectionlab.nifti_header_records import SCHEMA, nifti1_header_record_coded_v2 as record

ROOT = Path(__file__).resolve().parents[1]


def header_control(qcode=0, qfac=0):
    h = nib.Nifti1Header()
    h.set_data_shape((2, 2, 2))
    h.set_xyzt_units('mm')
    h.set_sform([[1., 0., 0., 0.], [0., 1., 0., 0.], [0., 0., 1., 0.], [0., 0., 0., 1.]], code=1)
    h['qform_code'] = qcode
    h['pixdim'][0] = qfac
    h['vox_offset'] = 352
    return h


def test_inactive_qfac_is_retained_without_transform_or_repair():
    raw = header_control().binaryblock
    with pytest.raises(nib.spatialimages.HeaderDataError): nifti1_header_record(raw, 'a'*64)
    result = record(raw, 'a'*64)
    assert result['schema'] == SCHEMA
    assert base64.b64decode(result['header_base64']) == raw
    assert result['header_sha256'] == hashlib.sha256(raw).hexdigest()
    assert result['source_file_sha256'] == 'sha256:'+'a'*64
    assert result['raw_grid']['pixdim'][0] == 0
    assert result['raw_grid']['qform_numeric'] is None
    assert result['raw_grid']['qform_status'] == 'inactive_not_decoded'
    assert result['raw_grid']['sform_code'] == 1


def test_saved_actual_case2_header_only():
    path = ROOT/'build/resect-deferred-qc-header-diagnostic-v1/Case2-during-image-header348.bin'
    if not path.exists(): pytest.skip('Retained original header unavailable; no patient acquisition')
    raw = path.read_bytes()
    diagnostic = json.loads(path.with_name('diagnosis.json').read_bytes())
    assert len(raw) == 348 and hashlib.sha256(raw).hexdigest() == diagnostic['raw_header_sha256']
    result = record(raw, diagnostic['original_file_sha256_from_retained_fixity_receipt'])
    assert result['raw_grid']['shape'] == [295, 334, 304]
    assert result['raw_grid']['pixdim'][0] == 0
    assert result['raw_grid']['qform_status'] == 'inactive_not_decoded'
    assert base64.b64decode(result['header_base64']) == raw


@pytest.mark.parametrize('qfac', [0, 2, -2])
def test_active_qform_invalid_qfac_still_refused(qfac):
    with pytest.raises(nib.spatialimages.HeaderDataError): record(header_control(1, qfac).binaryblock, 'a'*64)


@pytest.mark.parametrize('qfac', [1, -1])
def test_active_qform_valid_encoding_retained(qfac):
    h = header_control(1, qfac)
    result = record(h.binaryblock, 'a'*64)
    assert result['raw_grid']['qform_status'] == 'active_decoded'
    assert result['raw_grid']['qform_numeric'] == h.get_qform().tolist()


def test_inactive_nonfinite_encoding_is_uninterpreted_and_json_safe():
    h = header_control()
    h['quatern_b'] = float('nan')
    raw = h.binaryblock
    result = record(raw, 'a'*64)
    assert result['raw_grid']['quaternion_encoding']['quatern_b'] == 'nan'
    assert base64.b64decode(result['header_base64']) == raw
    json.dumps(result, allow_nan=False)


@pytest.mark.parametrize('bad', [b'', b'0'*347, b'0'*349, bytearray(b'0'*348)])
def test_exact_bytes_required(bad):
    with pytest.raises(ValueError): record(bad, 'a'*64)


def test_invalid_source_hash_and_magic_refused():
    h = header_control()
    with pytest.raises(ValueError): record(h.binaryblock, 'bad')
    h['magic'] = b'ni1\0'
    with pytest.raises(ValueError): record(h.binaryblock, 'a'*64)
