#!/usr/bin/env python3
"""Execute an explicit local case plan, retaining estimates and failed stages.

Original collections/images and frozen corpus catalogs are read-only. New native
collections, cached inference and standardized derivatives have separate paths.
No cohort expansion, fabricated anatomy, role changes or policy training occurs.
Run with PYTHONPATH=src .venv/bin/python scripts/standardize_case_library.py PLAN.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import traceback

import nibabel as nib
import numpy as np

from resectionlab.patient_anatomy import MAX_ARRAY, ROLES, PatientImagingCollection, affine, file_hash
from resectionlab.anatomy_import import import_dicom, import_nifti_series, select_intake_reference
from resectionlab.anatomy_models import reconstruct, import_proposal
from resectionlab.surgical_patient import prepare_surgical_environment
from resectionlab.surgical_environment import load_environment
from resectionlab.surgical_rl_protection import (
    annotation_from_collection, ProtectionSet, save_protection,
)
from resectionlab.surgical_rl_env import SurgicalRLEnv


CRITICAL = {'brainstem': 16, 'left_thalamus': 10, 'right_thalamus': 49,
            'left_caudate': 11, 'right_caudate': 50, 'left_putamen': 12,
            'right_putamen': 51, 'left_pallidum': 13, 'right_pallidum': 52,
            'left_hippocampus': 17, 'right_hippocampus': 53}


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + '.tmp')
    temp.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')
    temp.replace(path)


def initialize(spec):
    path = Path(spec['collection'])
    exists = (path / 'patient-anatomy.json').exists()
    if exists:
        c = PatientImagingCollection(path)
        if c.document['patient_id'] != spec['patient_id']:
            raise ValueError('Existing collection belongs to another patient')
        sources = c.source_records()
        existing_timepoint = c.document.get('study_timepoint', sources[0]['timepoint'] if sources else 'preoperative')
        timepoint = spec.get('timepoint', existing_timepoint)
    else:
        timepoint = spec.get('timepoint', 'preoperative')
        c = PatientImagingCollection.create(path, spec['patient_id'], timepoint=timepoint)
    before = c.identity()
    c.verify_source_association(timepoint, spec['patient_id'])
    for source in spec.get('series', []):
        kind = source.get('kind', 'nifti')
        association = dict(timepoint=timepoint, source_patient_id=source.get('source_patient_id'),
                           visit_id=source.get('visit_id', spec.get('visit_id')))
        if kind == 'nifti':
            row = import_nifti_series(c, source['path'], series_id=source['id'],
                                      modality=source.get('modality'), patient_id=spec['patient_id'],
                                      sidecar=source.get('sidecar'), **association)
            if row['id'] != source['id']:
                raise ValueError(f"Identical source already uses series ID {row['id']}; reuse that ID")
        elif kind == 'dicom':
            import_dicom(c, source['path'], modality=source.get('modality'),
                         selected_series=source.get('selected_series'),
                         sequence_by_series=source.get('sequence_by_series'), **association)
        else:
            raise ValueError(f"Unsupported batch imaging input: {kind}")
    select_intake_reference(c)
    annotations = {}
    for source in spec.get('layers', []):
        import_supplied_layer(c, source, annotations)
    for source in spec.get('registrations', []):
        if source['series'] in c.document['transforms']:
            if np.allclose(c.registration_matrix(source['series']), source['matrix'], atol=1e-6):
                continue
            raise ValueError(f"Conflicting registration for {source['series']}; review the existing transform")
        c.set_registration(source['series'], source['matrix'], status=source['status'],
                           method=source['method'])
    for field, key in (('missing', 'missing'), ('batch_source_notes', 'notes')):
        if key in spec:
            c.document[field] = spec[key]
    if not exists or c.identity() != before:
        c.save('Explicit same-person source intake')
    return c


def import_supplied_layer(collection, source, annotations):
    prior = next((row for row in collection.document['layers'] if row['id'] == source['id']), None)
    path = Path(source['path']).resolve()
    threshold = source.get('probability_threshold')
    label_value = source.get('value', 1)
    if type(label_value) is not int or not 1 <= label_value <= 65535:
        raise ValueError('Supplied annotation requires a valid declared label value')
    if threshold is not None and (type(threshold) not in {int, float} or not 0 < threshold <= 1 or label_value != 1):
        raise ValueError('Supplied fractional mask requires an explicit threshold in (0,1] and label value 1')
    key = (path, threshold)
    if prior:
        hashes = annotations.setdefault('source_hashes', {})
        if path not in hashes:
            hashes[path] = file_hash(path)
        current_hash = hashes[path]
        provenance = prior.get('provenance', {})
        if (prior['source_series'] != source['series'] or prior['kind'] != source['kind']
                or provenance.get('source', {}).get('sha256') != current_hash
                or provenance.get('threshold') != threshold
                or prior.get('original_label_value', prior['label_value']) != label_value):
            raise ValueError(f"Layer {source['id']} conflicts with existing annotation; use a new layer ID")
        return prior
    collection.verify_new_source_id(source['id'])
    scan = collection.series(source['series'])
    role = source.get('role', 'context')
    if role not in ROLES:
        raise ValueError('Unsupported annotation role')
    if key not in annotations:
        original = collection.original(path)
        image = nib.load(collection.path(original['path']))
        units = image.header.get_xyzt_units()[0]
        if units not in {'mm', 'unknown'} or len(image.shape) != 3 or np.prod(image.shape) * 4 > MAX_ARRAY:
            raise ValueError('Supplied annotation requires bounded 3D millimeter geometry')
        matrix = affine(image.affine)
        data = np.asanyarray(image.dataobj)
        if not np.isfinite(data).all():
            raise ValueError('Supplied annotation must be finite')
        probability = None
        if threshold is not None:
            if np.any((data < 0) | (data > 1)):
                raise ValueError('Supplied fractional annotation requires values in [0,1]')
            scores = data.astype(np.float32)
            probability = collection.put_array(scores)
            data = (scores >= threshold).astype(np.uint8)
        elif np.any((data < 0) | (data > 65535)) or not np.array_equal(data, np.rint(data)):
            raise ValueError('Categorical source mask requires integer labels')
        present = {int(value) for value in np.unique(data)}
        annotations[key] = dict(original=original, units=units, affine=matrix.tolist(),
                                spacing=nib.affines.voxel_sizes(matrix).tolist(),
                                data=collection.put_array(data.astype(np.uint16 if max(present) > 255 else np.uint8)),
                                present=present, probability=probability)
    annotation = annotations[key]
    provenance = dict(source.get('provenance', {}), source=annotation['original'])
    provenance.setdefault('kind', 'observed_source_annotation')
    provenance.setdefault('label_ontology', {'kind': source['kind'], 'source_label_value': label_value})
    row = {
        'id': source['id'], 'patient_id': scan['patient_id'], 'source_series': scan['id'],
        'kind': source['kind'], 'role': role, 'color': source.get('color', '#92a7b8'),
        'label': source.get('label', source['id']), 'label_value': label_value,
        'affine': annotation['affine'], 'units_status': annotation['units'],
        'data': annotation['data'], 'revision': 0, 'review_status': 'unreviewed imported annotation',
        'coverage': source.get('coverage', 'supplied annotation field; negative voxels unassessed'),
        'evidence_status': 'available' if label_value in annotation['present'] else 'empty-output',
        'source_grid_spacing_mm': scan['grid_spacing_mm'], 'output_grid_spacing_mm': annotation['spacing'],
        'provenance': provenance,
    }
    if annotation['probability'] is not None:
        row['probability'] = annotation['probability']
        row['score_meaning'] = 'supplied fractional annotation; no probability calibration asserted'
        provenance['threshold'] = threshold
    collection.document['layers'].append(row)
    return row


def process(spec, receipt, checkpoint):
    receipt.update(patient_id=spec['patient_id'], status='running', stages=[])
    checkpoint()
    c = initialize(spec)
    mapping = dict(spec.get('mapping', {}))
    for job in spec.get('models', []):
        stage = {'model': job['model'], 'series': job['series'], 'status': 'running'}
        receipt['stages'].append(stage)
        checkpoint()
        print(f"{spec['id']}: {job['model']} on {job['series']}", flush=True)
        try:
            record = reconstruct(c, job['series'], job['model'],
                                 synthseg_version=job.get('version', 1),
                                 synthseg_crop=job.get('crop'))
            before = c.identity()
            layer = import_proposal(c, job['series'], record)
            if c.identity() != before:
                c.save('Retained unreviewed model proposal: ' + job['model'])
            stage.update(status='succeeded', output=record['path'],
                         output_sha256=record['output_sha256'], layer=layer)
            if job.get('role'):
                # Empty localization remains evidence, never a selectable target.
                if not isinstance(layer, str) or not c.mask(c.layer(layer)).any():
                    raise ValueError('Model returned an empty role mask; not mapped')
                mapping[job['role']] = layer
        except Exception as exc:
            stage.update(status='failed', error=str(exc))
            print(traceback.format_exc(), flush=True)
        checkpoint()
    output = Path(spec['output'])
    report = prepare_surgical_environment(c.root, output, layer_ids=mapping,
        resolution_mm=1., allow_unreviewed_registration=spec.get('allow_unreviewed_registration', False))
    receipt.update(case_path=report['casePath'], environment_path=report['environmentPath'],
                   scene_id=report['environment']['sceneId'], anatomy=report['environment']['anatomy'],
                   target_mm3=report['environment']['target']['initialVolumeMm3'],
                   warnings=report['warnings'], mapping=mapping)
    checkpoint()
    scene = load_environment(report['casePath'])
    # Explicit numeric SynthSeg ontology; names never establish protection.
    annotations = []
    for structure, value in CRITICAL.items():
        candidates = [r for r in c.document['layers'] if r.get('kind') == 'brain-region'
                      and r.get('label_value') == value
                      and (r['id'].startswith('synthseg-') or r.get('provenance', {}).get('model') == 'synthseg')]
        if candidates:
            annotations.append(annotation_from_collection(scene, c.root, candidates[-1]['id'],
                                structure, allow_unreviewed=True))
    protection = ProtectionSet(scene, annotations, required=tuple(CRITICAL), strict=False)
    protection_path = output / 'rl-protection.json'
    save_protection(protection, protection_path)
    receipt['protection'] = protection.report()
    receipt['protection_path'] = str(protection_path.resolve())
    try:
        env = SurgicalRLEnv(scene, protection=protection)
        observation, info = env.reset(seed=0)
        if not env.observation_space.contains(observation):
            raise ValueError('Reset observation is outside the declared RL space')
        # Exercise the real protected transition; this is not a training run.
        after, reward, terminated, truncated, step_info = env.step_action(
            {'slot': 'left', 'operation': 'inspect', 'duration': .01})
        if not env.observation_space.contains(after):
            raise ValueError('Step observation is outside the declared RL space')
        snapshot = env.snapshot()
        restored, _ = env.restore(snapshot)
        if not all(np.array_equal(after[key], restored[key]) for key in after):
            raise ValueError('Restored RL observation differs')
        write_json(output / 'rl-smoke-snapshot.json', snapshot)
        receipt['rl_runtime'] = {'status': 'passed_reset_and_inspect', 'reward': float(reward),
                                  'terminated': terminated, 'truncated': truncated,
                                  'replay_observation_equal': True, 'critical_guards_relaxed': False}
        env.close()
    except Exception as exc:
        receipt['rl_runtime'] = {'status': 'blocked', 'error': str(exc)}
    runtime_ready = receipt['rl_runtime']['status'] == 'passed_reset_and_inspect'
    receipt['rl_research_runtime_ready'] = runtime_ready
    receipt['full_material_coverage'] = all(value == 'supplied' for value in receipt['anatomy'].values())
    receipt['supported_research_scope'] = (
        'Four-material instrument research; unreviewed anatomy and assumed mechanics'
        if runtime_ready and receipt['full_material_coverage'] else
        'Partial intracranial research; missing roles cannot support their instrument tasks'
        if runtime_ready else 'RL blocked; see runtime error')
    receipt['training_ready'] = False
    receipt['training_readiness_meaning'] = ('Runtime smoke-test only. Segmentation, registration, vascular coverage, '
        'function and mechanics remain unreviewed; no clinical or cohort training admission established.')
    receipt['status'] = 'standardized_with_limitations'
    checkpoint()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('plan', type=Path)
    parser.add_argument('--only', help='Run one declared case ID')
    args = parser.parse_args()
    plan = json.loads(args.plan.read_text())
    if plan.get('schema') != 'resectionlab.local-standardization-plan.v1':
        parser.error('Unknown explicit plan schema')
    ids = [s['id'] for s in plan['cases']]
    if len(ids) != len(set(ids)):
        parser.error('Duplicate plan IDs')
    result_path = args.plan.with_name('readiness.json')
    results = json.loads(result_path.read_text()) if result_path.exists() else {
        'schema': 'resectionlab.local-standardization-results.v1', 'cases': {},
        'plan_sha256': file_hash(args.plan), 'unavailable_cases': plan.get('unavailable_cases', [])}
    if results['plan_sha256'] != file_hash(args.plan):
        parser.error('Plan changed; use a separate result directory')
    checkpoint = lambda: write_json(result_path, results)
    for spec in plan['cases']:
        if args.only and spec['id'] != args.only:
            continue
        receipt = results['cases'].setdefault(spec['id'], {})
        try:
            process(spec, receipt, checkpoint)
        except Exception as exc:
            receipt.update(status='failed', error=str(exc))
            checkpoint()
            print(traceback.format_exc(), flush=True)
    print(str(result_path.resolve()), flush=True)


if __name__ == '__main__':
    main()
