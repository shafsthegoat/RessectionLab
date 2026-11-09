"""Metadata changes cannot grant a reserved original a development role."""
from dataclasses import replace
import hashlib
import json

import nibabel as nib
import numpy as np
import pytest

from resectionlab.core import CaseData, SourceRef
from resectionlab.surgical_patient_role import patient_role_evidence


def fixture(tmp_path, *, registered='TRAIN'):
    original = tmp_path/'sources'/'ct.nii.gz'; original.parent.mkdir()
    array = np.arange(27, dtype=np.float32).reshape(3, 3, 3)
    image = nib.Nifti1Image(array, np.eye(4)); image.header.set_xyzt_units('mm'); nib.save(image, original)
    digest = hashlib.sha256(original.read_bytes()).hexdigest()
    metadata = {'generated': True, 'is_synthetic': True,
                'data_case': {'family_id': 'TopCoW:014', 'role': 'TRAIN', 'split': 'TRAIN'}}
    case = CaseData(case_id='metadata-claim', mri=array, affine=np.eye(4), compartments={},
        metadata=metadata, source_refs=(SourceRef('observed', original.as_uri(), digest),))
    catalog = tmp_path/'catalog'/'topbrain'; catalog.mkdir(parents=True)
    member = 'source_topcow_ct_014.nii.gz'
    receipt = {'family': {'family_id': 'TopCoW:014', 'role': registered,
        'role_frozen_before_new_image_decode': True, 'images': {'ct': member}},
        'files': [{'source_member': member, 'path': 'sources/ct.nii.gz', 'sha256': digest}]}
    (catalog/'014-selective-extraction.json').write_text(json.dumps(receipt))
    return case, original


def test_actual_original_pixels_frame_and_predecode_role_are_joined(tmp_path):
    case, _ = fixture(tmp_path)
    result = patient_role_evidence(case, tmp_path)
    assert result['status'] == 'verified_train_source' and result['patientEvidence']
    assert result['family'] == 'TopCoW:014' and result['role'] == 'TRAIN'
    assert patient_role_evidence(case)['status'] == 'unavailable'


def test_reserved_family_relabelled_train_is_refused_before_decoding_registered_pixels(tmp_path, monkeypatch):
    case, _ = fixture(tmp_path, registered='EVAL')
    monkeypatch.setattr(nib, 'load', lambda *_: pytest.fail('Reserved pixels must not be decoded'))
    with pytest.raises(ValueError, match='reserved'):
        patient_role_evidence(case, tmp_path)


def test_changing_family_and_original_metadata_cannot_substitute_other_patient_pixels(tmp_path):
    case, original = fixture(tmp_path)
    other = replace(case, source_refs=(SourceRef('other', original.as_uri(), '0'*64),))
    with pytest.raises(ValueError, match='does not join'):
        patient_role_evidence(other, tmp_path)
    other = replace(case, mri=case.mri+1)
    with pytest.raises(ValueError, match='native image or physical frame'):
        patient_role_evidence(other, tmp_path)
    affine = case.affine.copy(); affine[0, 3] += 1
    other = replace(case, affine=affine)
    with pytest.raises(ValueError, match='native image or physical frame'):
        patient_role_evidence(other, tmp_path)


def test_synthetic_metadata_cannot_turn_acquired_originals_into_generated_controls(tmp_path):
    case, _ = fixture(tmp_path)
    assert case.metadata['generated'] and case.metadata['is_synthetic']
    assert patient_role_evidence(case)['status'] == 'unavailable'


def test_missing_or_edited_trusted_receipt_is_never_a_train_fallback(tmp_path):
    case, original = fixture(tmp_path)
    original.write_bytes(original.read_bytes()+b'changed')
    with pytest.raises(ValueError, match='trusted original'):
        patient_role_evidence(case, tmp_path)
    (tmp_path/'catalog'/'topbrain'/'014-selective-extraction.json').unlink()
    assert patient_role_evidence(case, tmp_path)['reason'] == 'SOURCE_ROLE_RECEIPT_UNAVAILABLE'
