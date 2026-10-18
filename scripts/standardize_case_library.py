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

from resectionlab.patient_anatomy import PatientImagingCollection, file_hash
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
    if (path / 'patient-anatomy.json').exists():
        c = PatientImagingCollection(path)
        if c.document['patient_id'] != spec['patient_id']:
            raise ValueError('Existing collection belongs to another patient')
        return c
    c = PatientImagingCollection.create(path, spec['patient_id'], timepoint='preoperative')
    for source in spec['series']:
        c.add_series(source['path'], series_id=source['id'], modality=source['modality'],
                     patient_id=spec['patient_id'])
    for source in spec.get('layers', []):
        image = nib.load(source['path'])
        data = np.asanyarray(image.dataobj)
        if not np.isfinite(data).all() or not np.array_equal(data, np.rint(data)):
            raise ValueError('Categorical source mask required')
        provenance = {**source.get('provenance', {}),
                      'source': c.original(source['path'])}
        row = c.add_layer(data.astype(np.uint16), image.affine,
                         layer_id=source['id'], source_series=source['series'],
                         kind=source['kind'], role=source.get('role', 'context'),
                         color=source.get('color', '#92a7b8'),
                         label=source.get('label', source['id']),
                         label_value=source.get('value', 1), provenance=provenance)
        if source.get('coverage'):
            row['coverage'] = source['coverage']
    for source in spec.get('registrations', []):
        c.set_registration(source['series'], source['matrix'], status=source['status'],
                           method=source['method'])
    c.document['missing'] = spec.get('missing', [])
    c.document['batch_source_notes'] = spec.get('notes', [])
    c.save('Explicit same-person source intake for standard environment')
    return c


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
