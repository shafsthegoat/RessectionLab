"""Adversarial coverage/accounting controls, with no acquired patient payloads."""
import copy
from types import SimpleNamespace

import numpy as np
import pytest

from resectionlab.simulator_qualification import grid_check, native_coverage, removed_accounting, removal_phases, validate_removed_ids


def manifest():
    return {'patient_id': 'A', 'series': [{'id': 't1', 'patient_id': 'A',
        'modality': 'T1', 'affine': np.eye(4).tolist(), 'units_status': 'mm'}],
        'layers': [{'id': 'tumor', 'patient_id': 'A', 'kind': 'tumor',
            'affine': np.eye(4).tolist(), 'units_status': 'mm'}]}


def scene():
    labels = np.array([[[1, 2, 1, 2, 0]]], dtype=np.uint16)
    return SimpleNamespace(labels=labels, affine=np.diag([2., 3., 4., 1.]),
        regions=[{'id': 1, 'name': 'Brain', 'target': False}, {'id': 2, 'name': 'Tumor', 'target': True}])


def test_absent_vessels_and_function_are_unknown_and_never_safe():
    result = native_coverage(manifest())
    assert not result['coverage']['vascularImaging']
    assert not result['coverage']['vesselLabels']
    assert result['coverage']['motorMapping'] == result['coverage']['languageMapping'] == 'unknown'
    assert result['gates']['patientSpecificVascularEvaluation'].startswith('unsupported')
    assert result['gates']['largeScaleRL'] == 'hold'


def test_estimated_vessels_without_angiography_do_not_qualify_vascular_evaluation():
    data = manifest(); data['layers'][0]['kind'] = 'vessels'
    data['layers'][0]['review_status'] = 'unreviewed'
    result = native_coverage(data)
    assert result['coverage']['vesselLabels']
    assert not result['coverage']['vascularImaging']
    assert result['gates']['patientSpecificVascularEvaluation'].startswith('unsupported')


def test_cross_patient_anatomy_fails_even_on_identical_grid():
    data = manifest(); data['layers'][0]['patient_id'] = 'B'
    result = native_coverage(data)
    assert result['crossPatientRows'] == ['tumor']
    assert not result['usableForMetadataComparison']


@pytest.mark.parametrize('modality', ['US', 'PET', 'SEG', 'CT', 'CTA', 'UNKNOWN'])
def test_non_mri_modalities_are_not_mri(modality):
    data = manifest(); data['series'][0]['modality'] = modality
    assert not native_coverage(data)['coverage']['MRI']


@pytest.mark.parametrize('units', ['unknown', 'm', None])
def test_units_must_be_explicit_mm(units):
    assert not grid_check(np.eye(4), units)['millimeterGridDeclared']


@pytest.mark.parametrize('a', [None, np.zeros((4, 4)), [[1]], np.full((4, 4), np.nan)])
def test_invalid_affine_is_reported(a):
    assert not grid_check(a, 'mm')['validAffine']


def test_source_registration_declaration_is_not_independent_acceptance():
    data = manifest(); data['transforms'] = {'t1': {'status': 'source-provided'}}
    report = native_coverage(data)
    assert report['series'][0]['registration']['status'] == 'source-provided'
    assert not report['series'][0]['grid']['registrationAccepted']


def test_independent_accounting_uses_determinant_and_keeps_access_cost():
    source = scene()
    states = {0: {'removed': []}, 1: {'removed': [0]}, 2: {'removed': [0, 1]},
        3: {'removed': [0, 1, 2, 3]}, 4: {'removed': [0, 1, 2, 3]}}
    before = copy.deepcopy(states)
    report = removal_phases(source, states, exposure_end=1, work_start=2, work_end=3)
    assert report['phases'][0]['nonTargetRemovedMm3'] == pytest.approx(24)
    assert report['fullEpisode']['targetRemovedMm3'] == pytest.approx(48)
    assert report['fullEpisode']['nonTargetRemovedMm3'] == pytest.approx(48)
    assert report['fullEpisode']['residualTargetMm3'] == pytest.approx(0)
    assert states == before


@pytest.mark.parametrize('ids', [[1, 1], [-1], [5], [4], [1.0], [True]])
def test_invalid_owned_removal_fails(ids):
    with pytest.raises(ValueError): removed_accounting(scene(), {'removed': ids})


def test_removal_regeneration_and_invalid_phases_fail():
    with pytest.raises(ValueError, match='REGENERATED'):
        removal_phases(scene(), {0: {'removed': []}, 1: {'removed': [0]}, 2: {'removed': []}},
            exposure_end=1, work_start=1, work_end=2)
    with pytest.raises(ValueError, match='BOUNDARIES'):
        removal_phases(scene(), {0: {'removed': []}, 1: {'removed': []}},
            exposure_end=2, work_start=1, work_end=1)


def test_per_action_check_catches_regeneration_hidden_between_phase_boundaries():
    previous = validate_removed_ids(scene(), [0])
    with pytest.raises(ValueError, match='REGENERATED'):
        validate_removed_ids(scene(), [1], previous=previous)


def test_baseline_refuses_unpinned_archive_before_any_decode(tmp_path, monkeypatch):
    import importlib.util
    from pathlib import Path
    from resectionlab import workspace_bundle
    script = Path(__file__).resolve().parents[1] / 'scripts/qualify_surgical_simulator.py'
    spec = importlib.util.spec_from_file_location('qualification_cli', script)
    module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
    def forbidden(*args, **kwargs):
        raise AssertionError('Payload loader must not run')
    monkeypatch.setattr(workspace_bundle, 'load_workspace', forbidden)
    path = tmp_path / 'unadmitted.ressectionlab'; path.write_bytes(b'not the frozen archive')
    with pytest.raises(ValueError, match='before payload access'):
        module.baseline(path, 'operative-cusa', tmp_path)


def test_inventory_deduplicates_paths_but_never_fuses_patient_versions(tmp_path):
    import importlib.util
    import json
    from pathlib import Path
    script = Path(__file__).resolve().parents[1] / 'scripts/qualify_surgical_simulator.py'
    spec = importlib.util.spec_from_file_location('qualification_inventory_cli', script)
    module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
    for version, kind in [('one', 'tumor'), ('two', 'vessels')]:
        data = manifest(); data['patient_id'] = 'UCSF-PDGM-0007'
        for row in [*data['series'], *data['layers']]: row['patient_id'] = data['patient_id']
        data['layers'][0]['kind'] = kind
        folder = tmp_path / version; folder.mkdir()
        (folder / 'patient-anatomy.json').write_text(json.dumps(data))
        # An unreadable stand-in for image data must never be decoded.
        (folder / 'arrays.npz').write_bytes(b'not an array')
    report = module.inventory([tmp_path, tmp_path])
    assert report['uniquePatients'] == 1 and len(report['versions']) == 2
    assert not report['pixelArraysRead']
    assert all(not (r['coverage']['tumorLabels'] and r['coverage']['vesselLabels']) for r in report['versions'])
