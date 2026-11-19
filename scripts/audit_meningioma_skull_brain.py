#!/usr/bin/env python3
"""Localize CT dense-signal conflicts with retained sub-06 MRI anatomy estimates."""
from __future__ import annotations

import argparse
from pathlib import Path

import nibabel as nib
import numpy as np
from scipy.ndimage import distance_transform_edt, map_coordinates

from data_integrated_hosts import UUID, binding
from resectionlab.data_root import atomic_json, checked_root, safe_path, sha256_file
from resectionlab.patient_anatomy import PatientImagingCollection, reconstruction_lock

STUDY = 'prepared/adaptive-anatomy-migration-20261117-v1/Meningioma-sub-06'
OUTPUT = 'qc/meningioma-sub06-skull-brain-conflict-v1.json'
TISSUE_CODES = (2, 3, 7, 8, 10, 11, 12, 13, 16, 17, 18, 26, 28,
                41, 42, 46, 47, 49, 50, 51, 52, 53, 54, 58, 60)
THRESHOLDS = (300, 700)


def source_grid(root, study, key):
    row = study.series(key)
    original = study.path(row['original']['path'])
    source = binding(root, original)
    image = nib.load(original)
    data = study.array(row['data'])
    if (source['sha256'] != row['original']['sha256'] or image.shape != data.shape
            or image.header.get_xyzt_units()[0] != 'mm'
            or not np.array_equal(image.affine, row['affine'])):
        raise ValueError(f'{key} original binding or physical frame changed')
    return source, data, image.affine


def sample_ct(data, affine, transform, world):
    source = nib.affines.apply_affine(np.linalg.inv(transform @ affine), world)
    return map_coordinates(data, source.T, order=1, mode='constant', cval=np.nan)


def metrics(values, region):
    selected = values[region]
    finite = selected[np.isfinite(selected)]
    return {'samples': int(len(selected)), 'within_ct_grid': int(len(finite)),
            'median_scalar': float(np.median(finite)) if len(finite) else None,
            'dense_counts': {str(t): int(np.count_nonzero(finite >= t)) for t in THRESHOLDS}}


def audit(root, study):
    transforms = study.document['transforms']
    if (study.document['revision'] != 8 or study.document['reference_series'] != 't1ce'
            or any(transforms[k]['status'] != 'unreviewed' for k in ('ct-bone', 'ct-brain'))):
        raise ValueError('Expected the preserved sub-06 revision with unreviewed CT alignment')
    mr_source, _, mr_affine = source_grid(root, study, 't1ce')
    brain_layer = study.layer('brain')
    if not np.array_equal(brain_layer['affine'], mr_affine):
        raise ValueError('Unexpected MRI extraction grid')
    stride = 2
    brain = study.mask(brain_layer)[::stride, ::stride, ::stride] > 0
    sampled_affine = mr_affine @ np.diag([stride, stride, stride, 1.])
    spacing = np.linalg.norm(sampled_affine[:3, :3], axis=0)
    positions = np.argwhere(brain)
    world = nib.affines.apply_affine(sampled_affine, positions)
    distance = distance_transform_edt(brain, sampling=spacing)[brain]

    cortex = study.layer('synthseg-3')
    tissue_layers = [study.layer(f'synthseg-{code}') for code in TISSUE_CODES]
    if any(row['data'] != cortex['data'] or row['affine'] != cortex['affine']
           or row['source_series'] != 't1ce' or row['label_value'] != code
           for row, code in zip(tissue_layers, TISSUE_CODES)):
        raise ValueError('Expected one shared MRI tissue label volume')
    tissue = np.isin(study.array(cortex['data']), TISSUE_CODES)
    tissue_world = nib.affines.apply_affine(cortex['affine'], np.argwhere(tissue))

    regions = {'all_synthstrip': np.ones(len(world), dtype=bool),
               'border_within_4mm': distance <= 4,
               'interior_over_4mm': distance > 4}
    lower = float(np.floor(world[:, 2].min() / 20) * 20)
    boundaries = np.arange(lower, world[:, 2].max() + 20, 20)
    for lo, hi in zip(boundaries[:-1], boundaries[1:]):
        slab = (world[:, 2] >= lo) & (world[:, 2] < hi)
        if slab.any():
            regions[f'ras_z_{lo:g}_to_{hi:g}_mm'] = slab

    report = {'schema': 'meningioma-sub06-skull-brain-conflict-v1',
              'study': binding(root, study.root / 'patient-anatomy.json'), 'revision': study.document['revision'],
              'mri_source': mr_source, 'brain_estimate': brain_layer['data'],
              'synthseg_data': cortex['data'], 'synthseg_included_tissue_codes': list(TISSUE_CODES),
              'sample_spacing_mm': spacing.tolist(), 'sources': {},
              'limits': ['CT scalar threshold is a dense-signal check, with no bone reference labels.',
                         'The two MRI masks are model estimates with different tissue definitions.',
                         'Border distances are to sampled MRI mask background, with no claim of anatomy error location.',
                         'The audit estimates neither registration error nor a reviewed CT-to-MRI transform.']}
    for key in ('ct-brain', 'ct-bone'):
        source, data, affine = source_grid(root, study, key)
        row = transforms[key]
        if row['source_sha256'] != source['sha256'] or row['reference_sha256'] != mr_source['sha256']:
            raise ValueError(f'{key} alignment belongs to different source images')
        identity = np.eye(4)
        current = np.asarray(row['matrix'])
        candidates = {'shared_header': identity}
        if not np.array_equal(current, identity):
            candidates['retained_ct_to_mri'] = current
        result = {'source': source, 'retained_transform': row['transform_digest'],
                  'transform_status': row['status'], 'candidates': {}}
        for name, matrix in candidates.items():
            values = sample_ct(data, affine, matrix, world)
            ct_tissue = sample_ct(data, affine, matrix, tissue_world)
            result['candidates'][name] = {'matrix': matrix.tolist(),
                                          'synthstrip_by_region': {label: metrics(values, region)
                                                                   for label, region in regions.items()},
                                          'synthseg_tissue': metrics(ct_tissue, np.ones(len(tissue_world), dtype=bool))}
        report['sources'][key] = result
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path)
    parser.add_argument('--output-name', default=Path(OUTPUT).name)
    args = parser.parse_args()
    if Path(args.output_name).name != args.output_name or not args.output_name.endswith('.json'):
        parser.error('--output-name must be a new JSON filename under qc')
    root = checked_root(args.root, expected_uuid=UUID)
    output = safe_path(root, f'qc/{args.output_name}')
    with reconstruction_lock():
        if output.exists():
            raise FileExistsError(output)
        report = audit(root, PatientImagingCollection(safe_path(root, STUDY)))
        if sha256_file(safe_path(root, f'{STUDY}/patient-anatomy.json')) != report['study']['sha256']:
            raise ValueError('The study revision changed during the audit')
        atomic_json(output, report)
    print(output)


if __name__ == '__main__':
    main()
