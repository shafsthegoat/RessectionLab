#!/usr/bin/env python3
"""Create an opt-in physics-v2 derivative while preserving every patient array.

PYTHONPATH=src .venv/bin/python scripts/prepare_patient_physics_v2.py
Only declared research simulation metadata changes. This neither resegments the
patient nor claims calibrated material or hemodynamic parameters.
"""
from dataclasses import replace
from hashlib import sha256
from pathlib import Path
import argparse
import json
from resectionlab.core import array_digest, thaw_json
from resectionlab.imaging import load_case, save_case
from resectionlab.surgical_scene import from_case
from resectionlab.surgical_physics_v2 import default_profile


def file_hash(path):
    digest = sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''): digest.update(chunk)
    return digest.hexdigest()


def array_hashes(case):
    result = {'mri': array_digest(case.mri), 'affine': array_digest(case.affine)}
    for group in ('compartments', 'source_compartments'):
        result[group] = {name: array_digest(data) for name, data in getattr(case, group).items()}
    if case.brain_mask is not None: result['brainMask'] = array_digest(case.brain_mask)
    return result


def prepare(source='outputs/surgical-patient-sub06/patient.ressectionlab', output='outputs/surgical-patient-v2'):
    source = Path(source).resolve(); output = Path(output).resolve(); target = output / 'patient.ressectionlab'
    if source == target: raise ValueError('Derivative must not overwrite the source patient')
    source_file_hash = file_hash(source); case = load_case(source); hashes = array_hashes(case)
    source_labels = array_digest(from_case(case).labels)
    metadata = thaw_json(case.metadata); properties = metadata.setdefault('surgical_properties', {})
    previous = properties.pop('tissuePhysics', None)
    properties['tissuePhysicsV2'] = default_profile()
    derived = replace(case, metadata=metadata)
    output.mkdir(parents=True, exist_ok=True)
    saved = save_case(derived, target); loaded = load_case(saved)
    if array_hashes(loaded) != hashes: raise AssertionError('Patient numeric arrays changed')
    if loaded.source_refs != case.source_refs: raise AssertionError('Patient SourceRefs changed')
    if array_digest(from_case(loaded).labels) != source_labels: raise AssertionError('Material ownership changed')
    if file_hash(source) != source_file_hash: raise AssertionError('Source patient file changed')
    report = {'schema': 'resectionlab.patient-physics-export.v2', 'status': 'passed',
        'sourceCase': str(source), 'sourceFileSha256': source_file_hash, 'sourceCaseHash': case.semantic_hash,
        'casePath': str(saved), 'caseFileSha256': file_hash(saved), 'caseHash': loaded.semantic_hash,
        'arrayHashes': hashes, 'labelArrayHash': source_labels, 'shape': list(case.mri.shape),
        'sourceUnchanged': True, 'arraysUnchanged': True, 'sourceRefsUnchanged': True,
        'replacedPhysicsV1': previous, 'tissuePhysicsV2': properties['tissuePhysicsV2'],
        'limitations': ['Declared uncalibrated engineering parameters, not measured patient physiology',
                        'Original annotation and registration uncertainty remains unchanged']}
    (output / 'preparation.json').write_text(json.dumps(report, indent=2, allow_nan=False) + '\n')
    return report


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('source', nargs='?', default='outputs/surgical-patient-sub06/patient.ressectionlab')
    parser.add_argument('output', nargs='?', default='outputs/surgical-patient-v2')
    args = parser.parse_args(); report = prepare(args.source, args.output)
    print(json.dumps({k: report[k] for k in ('status', 'casePath', 'sourceUnchanged', 'arraysUnchanged')}, indent=2))
