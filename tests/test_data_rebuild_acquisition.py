"""Acquisition guards run offline without imaging fixtures or external storage."""
from pathlib import Path
import sys
import zipfile
import hashlib
import json

import pytest
import numpy as np
import nibabel as nib

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts'))
import data_rebuild_acquire as acquire
import data_rebuild_topbrain as topbrain


def test_archive_range_refuses_oversized_full_response(monkeypatch):
    class FullResponse:
        status = 200
        headers = {'Content-Length': '4840816640'}
        closed = False
        def close(self):
            self.closed = True
        def read(self, *_):
            pytest.fail('Full archive must not be read when byte range was requested')
    response = FullResponse()
    monkeypatch.setattr(acquire.urllib.request, 'urlopen', lambda *_a, **_kw: response)
    with pytest.raises(RuntimeError, match='full archive response refused'):
        acquire.request_range('https://official.invalid/archive.tar', 512, 1023, 4840816640)
    assert response.closed


def test_existing_same_size_source_must_match_prior_local_hash(tmp_path, monkeypatch):
    source = tmp_path / 'original.nii.gz'
    source.write_bytes(b'changed')
    monkeypatch.setattr(acquire, 'mount_check', lambda *_a, **_kw: None)
    with pytest.raises(RuntimeError, match='Existing source differs'):
        acquire.acquire_file(tmp_path, {'bytes': 7, 'expected_local_sha256': '0' * 64}, source, {})


def test_host_selection_preserves_source_person_roles_and_pairing():
    ixi = list(acquire.candidates('ixi'))
    lausanne = list(acquire.candidates('lausanne'))
    assert len(ixi) >= 48 and len(lausanne) == 89
    assert all(row['role'] == 'TRAIN' and row['diagnosis'] == 'healthy_control'
               for row in ixi + lausanne)
    assert 'IXI002' not in {row['subject'] for row in ixi}  # Frozen measurement evaluation.
    for row in ixi:
        prefixes = {entry['path'].removesuffix('-T1.nii.gz').removesuffix('-MRA.nii.gz')
                    for entry in row['files']}
        assert len(prefixes) == 1


def test_topbrain_inspection_refuses_zip_path_traversal(tmp_path, monkeypatch):
    archive = tmp_path / 'malformed.zip'
    with zipfile.ZipFile(archive, 'w') as out:
        out.writestr('../unsafe.txt', 'not a patient')
    monkeypatch.setattr(topbrain, 'archive_checked', lambda root: archive)
    monkeypatch.setattr(topbrain, 'guard', lambda *_a, **_kw: tmp_path)
    with pytest.raises(ValueError, match='Unsafe source ZIP member'):
        topbrain.inspect(tmp_path)
    assert not (tmp_path.parent / 'unsafe.txt').exists()


def test_topbrain_same_numeric_label_does_not_share_ct_mra_semantics():
    ct = {'background': 0, 'L-OA': 34, 'VoG': 35, 'StS': 36,
          'ICVs': 37, 'R-BVR': 38, 'L-BVR': 39, 'SSS': 40}
    mr = {'background': 0, 'L-OA': 34, 'R-ECA': 35, 'L-MMA': 42}
    assert topbrain.group_v1_labels(np.array([0, 34, 35, 40]), ct, 'ct').tolist() == [0, 1, 2, 2]
    assert topbrain.group_v1_labels(np.array([0, 34, 35, 42]), mr, 'mr').tolist() == [0, 1, 1, 1]
    with pytest.raises(ValueError, match='dictionary'):
        topbrain.group_v1_labels(np.array([99]), mr, 'mr')


def test_missing_mask_units_can_only_inherit_from_matching_source_grid():
    mask = nib.Nifti1Image(np.ones((2, 3, 4), dtype=np.uint8), np.eye(4))
    image = nib.Nifti1Image(np.ones((2, 3, 4), dtype=np.int16), np.eye(4))
    image.header.set_xyzt_units('mm')
    assert topbrain.qualified_mask_units(mask, image) == 'unknown'
    assert mask.header.get_xyzt_units()[0] == 'unknown'  # Never mutate the original.
    wrong_affine = np.eye(4)
    wrong_affine[0, 3] = 2
    shifted = nib.Nifti1Image(np.ones((2, 3, 4), dtype=np.uint8), wrong_affine)
    with pytest.raises(ValueError, match='corresponding image grid'):
        topbrain.qualified_mask_units(shifted, image)


def test_volume_rendering_off_changes_saved_view_but_no_array_payload(tmp_path):
    from resectionlab.imaging import load_nifti_case
    from resectionlab.workspace_bundle import save_workspace, load_workspace, view_state
    image = nib.Nifti1Image(np.arange(24, dtype=np.float32).reshape(2, 3, 4), np.eye(4))
    image.header.set_xyzt_units('mm')
    source = tmp_path / 'source.nii.gz'
    nib.save(image, source)
    case = load_nifti_case(source)
    original, revised = tmp_path / 'original.ressectionlab', tmp_path / 'display-v2.ressectionlab'
    save_workspace(case, original, sources=[], imaging=None, episode=None, selection=None, artifacts={})
    state = view_state(case, [])
    state['states']['primary']['viewer'] = {'volumeRendering': {'enabled': False, 'preset': 'mri'}}
    save_workspace(case, revised, sources=[], imaging=state, episode=None, selection=None, artifacts={})
    reopened, _, workspace = load_workspace(revised)
    assert workspace['imagingState']['states']['primary']['viewer']['volumeRendering']['enabled'] is False
    assert reopened.semantic_hash == case.semantic_hash
    assert topbrain.package_array_hashes(original) == topbrain.package_array_hashes(revised)


def test_completed_resume_requires_current_policy_and_unchanged_package(tmp_path):
    family = {'subject': '001', 'role': 'DEVELOPMENT'}
    entries = []
    contact = tmp_path / 'qc/source-contact.png'
    contact.parent.mkdir()
    contact.write_bytes(b'test contact sentinel')
    for suffix in ['CTA-v3', 'TOF-MRA-v3-display-v2']:
        path = tmp_path / f'staging/candidates/clinical/TopCoW-001-{suffix}.ressectionlab'
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b'previously validated native bytes')
        entries.append({'path': str(path.relative_to(tmp_path)), 'sha256': hashlib.sha256(path.read_bytes()).hexdigest(), 'contact_sheet': str(contact.relative_to(tmp_path))})
    receipt = tmp_path / 'catalog/topbrain/001-native-candidates.json'
    receipt.parent.mkdir(parents=True)
    result = {'family': family, 'candidates': entries}
    receipt.write_text(json.dumps(result))
    assert topbrain.completed_native(tmp_path, family) == result
    (tmp_path / entries[0]['path']).write_bytes(b'same-size changed native content')
    with pytest.raises(ValueError, match='immutable preparation receipt'):
        topbrain.completed_native(tmp_path, family)
