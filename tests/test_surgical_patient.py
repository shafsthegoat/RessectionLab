import hashlib
import json
import nibabel as nib
import numpy as np
import pytest
from resectionlab.patient_anatomy import PatientImagingCollection
from resectionlab.surgical_patient import prepare_surgical_patient, ORDER
from resectionlab.imaging import load_case
from resectionlab.surgical_scene import from_case


def fixture(tmp_path, *, status='reviewed'):
    affine = np.diag([-.5, .5, .5, 1.])
    affine[:3, 3] = [4, 2, -3]
    intensity = np.indices((12, 12, 12)).sum(axis=0).astype(np.float32)
    image = nib.Nifti1Image(intensity, affine)
    image.header.set_xyzt_units('mm')
    path = tmp_path / 'scan.nii.gz'
    nib.save(image, path)
    c = PatientImagingCollection.create(tmp_path / 'native', 'test')
    c.add_series(path, series_id='mri', modality='T1', timepoint='preoperative')
    c.add_series(path, series_id='ct', modality='CT', timepoint='preoperative')
    c.set_registration('ct', np.eye(4), status=status, method='explicit test registration')
    arrays = {name: np.zeros((12, 12, 12), np.uint8) for name in ORDER}
    arrays['brain'][2:10, 2:10, 2:10] = 1
    arrays['tumor'][4:8, 4:8, 4:8] = 1
    arrays['bone'][5, 4:8, 4:8] = 1
    arrays['bone'][0, :, :] = 1  # Thin sheet not sampled by target cell centers.
    arrays['vessels'][2, 2, :] = 1
    for name, array in arrays.items():
        c.add_layer(array, affine, layer_id=name, source_series='ct' if name == 'bone' else 'mri', kind=name, role='context', color='#aaaaaa')
    c.save()
    return c


def test_export_preserves_sources_overlaps_coordinates_and_thin_structures(tmp_path):
    c = fixture(tmp_path)
    before = {str(p.relative_to(c.root)): hashlib.sha256(p.read_bytes()).hexdigest() for p in c.root.rglob('*') if p.is_file()}
    report = prepare_surgical_patient(c.root, tmp_path / 'output', layer_ids={k: k for k in ORDER})
    case = load_case(report['casePath'])
    assert case.mri.shape == (6, 6, 6)
    expected_kinds = {'brain': 'brain', 'bone': 'skull', 'tumor': 'tumor', 'vessels': 'vessels'}
    for name, layer in case.metadata['anatomy_layers'].items():
        assert layer['kind'] == expected_kinds[name]
        assert layer['provenance'] == 'estimated'
        assert layer['review_status'] == 'unreviewed'
        assert layer['model']['name']
        assert layer['coverage']
    np.testing.assert_allclose(case.affine[:3, 3], [3.75, 2.25, -2.75])
    assert case.mri[2, 2, 2] == pytest.approx(13.5)
    assert all(case.compartments[k].any() for k in ORDER)
    assert np.any(case.compartments['brain'] & case.compartments['tumor'])
    assert report['overlapCells']['brain:tumor'] > 0
    assert report['sourceLayers']['bone']['sampling'].startswith('any-hit')
    assert all(report['ownershipCells'].values())
    adapted = from_case(case)
    for r in adapted.regions:
        assert np.count_nonzero(adapted.labels == r['id']) == report['ownershipCells'][r['name']]
    after = {str(p.relative_to(c.root)): hashlib.sha256(p.read_bytes()).hexdigest() for p in c.root.rglob('*') if p.is_file()}
    assert before == after
    assert json.loads((tmp_path / 'output/report.json').read_text())['caseHash'] == case.semantic_hash


def test_unreviewed_transform_opt_in_preserves_status(tmp_path):
    c = fixture(tmp_path, status='unreviewed')
    with pytest.raises(ValueError, match='opt-in'):
        prepare_surgical_patient(c.root, tmp_path / 'rejected', layer_ids={k: k for k in ORDER})
    report = prepare_surgical_patient(c.root, tmp_path / 'output', layer_ids={k: k for k in ORDER}, allow_unreviewed_registration=True)
    assert report['sourceManifest']['transforms']['ct']['status'] == 'unreviewed'
    assert PatientImagingCollection(c.root).document['transforms']['ct']['status'] == 'unreviewed'
    assert any('unreviewed registration' in warning for warning in report['warnings'])


def test_reject_missing_or_empty_selected_anatomy(tmp_path):
    c = fixture(tmp_path)
    with pytest.raises(ValueError, match='four distinct'):
        prepare_surgical_patient(c.root, tmp_path / 'output', layer_ids={'brain': 'brain'})
    row = c.layer('vessels')
    row['label_value'] = 2
    c.save()
    with pytest.raises(ValueError, match='no occupied'):
        prepare_surgical_patient(c.root, tmp_path / 'output', layer_ids={k: k for k in ORDER})


@pytest.mark.parametrize('name,provenance,expected', [
    ('brain', {'inference': {'model': {'name': 'SynthStrip'}}}, 'SynthStrip'),
    ('vessels', {'inference': {'model': {'name': 'vesselFM'}}}, 'vesselFM'),
    ('tumor', {'model': 'Raidionics meningioma proposal'}, 'Raidionics meningioma proposal'),
    ('bone', {'method': 'native CT 300 HU boundary proposal'}, 'native CT 300 HU boundary proposal'),
])
def test_anatomy_model_attribution_comes_from_native_provenance(name, provenance, expected):
    from resectionlab.surgical_patient import _anatomy_metadata
    layer = _anatomy_metadata(name, {'provenance': provenance})
    assert layer['model']['name'] == expected
    assert layer['provenance'] == 'estimated'
