"""Versioned batch entry points for native and explicit simulator plans."""
import importlib.util
import json
from pathlib import Path
import sys

import nibabel as nib
import numpy as np
import pytest

from resectionlab import anatomy_preparation as preparation
from resectionlab.anatomy_planning import native_target_options
from resectionlab.patient_anatomy import PatientImagingCollection, file_hash


@pytest.fixture
def batch():
    path = Path(__file__).resolve().parents[1] / 'scripts' / 'standardize_case_library.py'
    spec = importlib.util.spec_from_file_location('standardize_case_library', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def source(tmp_path, name, modality, number):
    path = tmp_path / f'{name}.nii.gz'
    image = nib.Nifti1Image(np.full((8, 8, 8), number, dtype=np.float32), np.eye(4))
    image.header.set_xyzt_units('mm')
    nib.save(image, path)
    return {'id': name, 'path': str(path), 'modality': modality}


def run_plan(batch, tmp_path, monkeypatch, schema, name, spec, *flags):
    plan = tmp_path / f'{name}.json'
    plan.write_text(json.dumps({'schema': schema, 'cases': [spec]}))
    monkeypatch.setattr(sys, 'argv', ['standardize_case_library.py', str(plan), *flags])
    assert batch.main() == 0
    result = plan.with_suffix('.readiness.json') if schema == batch.NATIVE_PLAN else tmp_path / 'readiness.json'
    return json.loads(result.read_text())['cases']['controlled']


def test_native_batch_adds_new_scans_and_resumes_without_an_implicit_target(batch, tmp_path, monkeypatch):
    ct = source(tmp_path, 'ct', 'CT', 1)
    mri = source(tmp_path, 't1', 'T1', 2)
    originals = {item['path']: file_hash(item['path']) for item in (ct, mri)}
    collection = tmp_path / 'study'
    initial = {
        'id': 'controlled', 'patient_id': 'fixture', 'collection': str(collection),
        'series': [ct], 'missing': ['Patient-specific function is unavailable.'],
    }
    runs = []

    def fixture_jobs(c, options):
        jobs = [
            {'id': 'brain-estimate', 'model': 'fixture', 'series_id': 'ct', 'status': 'ready'},
            {'id': 'optional-model', 'model': 'fixture', 'series_id': 'ct', 'status': 'ready'},
            {'id': 'tumor', 'status': 'missing_input', 'reason': 'No lesion annotation or supported tumor indication'},
        ]
        if len(c.document['series']) == 2:
            jobs.append({'id': 'new-source', 'model': 'fixture', 'series_id': 't1', 'status': 'ready'})
        return jobs

    def fixture_stage(c, stage, cancelled, progress):
        runs.append(stage['id'])
        if stage['id'] == 'optional-model':
            raise FileNotFoundError('Fixture weights unavailable')
        scan = c.series(stage['series_id'])
        layer = c.add_layer(
            np.ones(scan['data']['shape'], dtype=np.uint8), scan['affine'], layer_id=stage['id'],
            source_series=scan['id'], kind='brain' if scan['id'] == 'ct' else 'context',
            role='context', color='#abcdef', provenance={'model': 'fixture'},
        )
        return 'cached', 'Fixture estimate', [{'id': layer['id'], 'sha256': layer['data']['sha256']}]

    monkeypatch.setattr(preparation, '_jobs', fixture_jobs)
    monkeypatch.setattr(preparation, '_run_stage', fixture_stage)
    stages = run_plan(batch, tmp_path, monkeypatch, batch.NATIVE_PLAN, 'intake', initial)['stages']
    assert stages['brain-estimate']['status'] == 'complete'
    assert stages['optional-model']['status'] == 'failed'
    assert 'Fixture weights unavailable' in stages['optional-model']['reason']
    c = PatientImagingCollection(collection)
    assert c.document['study_timepoint'] == 'source-session-unknown'
    assert c.series('ct')['timepoint'] == 'source-session-unknown'
    assert c.document['reference_series'] == 'ct'
    assert native_target_options(c) == []
    assert runs == ['brain-estimate', 'optional-model']

    enriched = {**initial, 'series': [ct, mri, mri]}
    result = run_plan(batch, tmp_path, monkeypatch, batch.NATIVE_PLAN, 'enriched', enriched)
    snapshot = result['snapshot_id']
    c = PatientImagingCollection(collection)
    assert c.identity() == snapshot
    assert [row['id'] for row in c.document['series']] == ['ct', 't1']
    assert [row['id'] for row in c.document['layers']] == ['brain-estimate', 'new-source']
    assert result['reference_series'] == 'ct'
    assert result['manifest'] == str(collection / 'patient-anatomy.json')
    assert result['status'] == 'standardized_with_limitations'
    assert result['missing'] == initial['missing']
    assert result['stages']['tumor']['status'] == 'missing_input'
    assert result['stages']['optional-model']['status'] == 'failed'
    assert native_target_options(c) == []
    assert runs == ['brain-estimate', 'optional-model', 'new-source']

    assert run_plan(batch, tmp_path, monkeypatch, batch.NATIVE_PLAN, 'enriched', enriched)['snapshot_id'] == snapshot
    assert runs == ['brain-estimate', 'optional-model', 'new-source']
    run_plan(batch, tmp_path, monkeypatch, batch.NATIVE_PLAN, 'enriched', enriched, '--retry-failed')
    assert runs == ['brain-estimate', 'optional-model', 'new-source', 'optional-model']
    assert PatientImagingCollection(collection).identity() == snapshot
    assert {path: file_hash(path) for path in originals} == originals
    assert not list(collection.rglob('*.ressectionlab'))


def test_v1_keeps_explicit_export_and_legacy_timepoint_default(batch, tmp_path, monkeypatch):
    spec = {'id': 'controlled', 'patient_id': 'fixture', 'collection': str(tmp_path / 'legacy'),
            'series': [source(tmp_path, 't1', 'T1', 1)], 'output': str(tmp_path / 'export')}

    def legacy_process(spec, receipt, checkpoint):
        c = batch.initialize(spec)
        receipt.update(output=spec['output'], timepoint=c.series('t1')['timepoint'])
        checkpoint()

    monkeypatch.setattr(batch, 'process', legacy_process)
    result = run_plan(batch, tmp_path, monkeypatch, batch.LEGACY_PLAN, 'legacy-plan', spec)
    assert result == {'output': spec['output'], 'timepoint': 'preoperative'}
