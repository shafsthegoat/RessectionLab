#!/usr/bin/env python3
"""Expose retained CADS 465 structures on the full acquired CT lattice."""
from __future__ import annotations

import argparse
from copy import deepcopy
from pathlib import Path
from shutil import copyfile
from tempfile import TemporaryDirectory

import nibabel as nib
import numpy as np

from build_cta_native_showcases import register
from build_cta_organic_gliomas import read_json
from data_integrated_hosts import UUID, binding, checked_file
from prepare_cads_brainct import CASE as SOURCE_CASE, DESTINATION as SOURCE, SERIES, source_inputs
from resectionlab.anatomy_package import referenced_files
from resectionlab.data_root import atomic_json, checked_root, safe_path, sha256_file
from resectionlab.patient_anatomy import PatientImagingCollection, reconstruction_lock, world

CASE = 'CADS-BRAINCT-465-SKULL-BASE-DETAIL-v1'
DESTINATION = Path('prepared/cads465-skull-base-detail-v1') / CASE
LAYERS = (
    ('compact-head-bone', 'gray-matter', 6, 'skull', 'Compact head bone · CADS estimate', '#e4bd91'),
    ('spongy-head-bone', 'gray-matter', 7, 'skull', 'Spongy head bone · CADS estimate', '#d4985d'),
    ('left-cochlea', 'brainstem', 8, 'brain-region', 'Left cochlea · CADS estimate', '#53cfda'),
    ('right-cochlea', 'brainstem', 9, 'brain-region', 'Right cochlea · CADS estimate', '#59a8e2'),
    ('optic-chiasm', 'brainstem', 24, 'brain-region', 'Optic chiasm · CADS estimate', '#b19ce6'),
    ('left-optic-nerve', 'brainstem', 25, 'brain-region', 'Left optic nerve · CADS estimate', '#dd9be0'),
    ('right-optic-nerve', 'brainstem', 26, 'brain-region', 'Right optic nerve · CADS estimate', '#b884db'),
    ('pituitary', 'brainstem', 29, 'brain-region', 'Pituitary · CADS estimate', '#fca4a9'),
)


def class_row(study, key, template, code, kind, name, color):
    source = study.layer(template)
    row = deepcopy(source)
    row.update(id=key, label=name, kind=kind, color=color, label_value=code,
               coverage='Exact nonempty native publisher model class with unverified anatomical sensitivity')
    row['provenance']['source_label_values'] = [code]
    return row


def measure_classes(study, images):
    scan = study.series(SERIES)
    affine = np.asarray(scan['affine'])
    raw = study.array(scan['data'])
    dims = raw.shape
    categories = {}
    for template, part in (('gray-matter', 'part_557'), ('brainstem', 'part_558')):
        labels = study.array(study.layer(template)['data'])
        if not np.array_equal(labels, np.asanyarray(images[part].dataobj)):
            raise ValueError(f'Copied {part} labels differ from the native publisher file')
        categories[template] = labels
    result = {}
    for key, template, value, *_ in LAYERS:
        coordinates = np.argwhere(categories[template] == value)
        if not len(coordinates):
            raise ValueError(f'Expected the retained nonempty class {key}')
        support = raw[tuple(coordinates.T)]
        center = coordinates.mean(axis=0)
        result[key] = {'publisher_value': value, 'voxels': len(coordinates),
                       'volume_mm3': len(coordinates)*abs(np.linalg.det(affine[:3,:3])),
                       'centroid_ras_mm': world(affine, center).tolist(),
                       'bounds_ijk': [coordinates.min(0).tolist(), coordinates.max(0).tolist()],
                       'source_scalar_p10_p50_p90': np.percentile(support, [10,50,90]).tolist(),
                       'boundary_face_voxels': [int(np.count_nonzero((coordinates[:,i]==0) |
                                                   (coordinates[:,i]==dims[i]-1))) for i in range(3)]}
    left, right = (result[key]['centroid_ras_mm'][0] for key in ('left-cochlea', 'right-cochlea'))
    if left >= right:
        raise ValueError('Retained cochlea labels do not follow the acquired left/right ordering')
    return result


def build(root):
    destination = safe_path(root, DESTINATION)
    manifest = destination / 'patient-anatomy.json'
    if destination.exists():
        row = read_json(destination / 'receipt.json')
        study = PatientImagingCollection(checked_file(root, row['manifest']).parent)
        if study.root != destination or study.identity() != row['snapshot_id']:
            raise ValueError('Existing detail study does not match its verified receipt')
        for name, digest in set(referenced_files(study.document)):
            if sha256_file(study.path(name)) != digest:
                raise ValueError(f'Existing study member differs from its publisher-derived copy: {name}')
        return row

    base_path = safe_path(root, Path(SOURCE) / 'patient-anatomy.json')
    base_receipt = read_json(base_path.with_name('receipt.json'))
    if checked_file(root, base_receipt['manifest']) != base_path:
        raise ValueError('Expected the live verified CADS source study')
    source = PatientImagingCollection(base_path.parent)
    if source.document['patient_id'] != SOURCE_CASE or source.identity() != base_receipt['snapshot_id']:
        raise ValueError('Unexpected source patient or snapshot')
    _, images, acquisition, _ = source_inputs(root)
    scan = source.series(SERIES)
    if (not np.array_equal(scan['affine'], images['ncct'].affine)
            or not np.array_equal(source.array(scan['data']), images['ncct'].get_fdata(dtype=np.float32))):
        raise ValueError('Native CT must equal the retained acquired volume')
    measurements = measure_classes(source, images)
    document = deepcopy(source.document)
    document.pop('snapshot_id')
    document.update(patient_id=CASE, revision=0, history=[])
    for row in document['series'] + document['layers']:
        row['patient_id'] = CASE
    next(row for row in document['series'] if row['id'] == SERIES).pop('preview_pyramid', None)
    document['layers'].extend({**class_row(source, *layer), 'patient_id': CASE} for layer in LAYERS)
    document['display']['initial_layers'].extend(['left-cochlea', 'right-cochlea'])
    document['data_case']['limits'].append(
        'Additional native model categories are source-linked inspection evidence with unreviewed anatomical accuracy.'
    )
    destination.parent.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(dir=destination.parent, prefix='.cads465-detail-') as scratch:
        stage = Path(scratch) / 'study'
        stage.mkdir()
        study = PatientImagingCollection(stage, document)
        for name, digest in set(referenced_files(document)):
            original, output = source.path(name), study.path(name)
            output.parent.mkdir(parents=True, exist_ok=True)
            copyfile(original, output)
            if sha256_file(output) != digest:
                raise ValueError(f'Study copy differs from the retained source member: {name}')
        study.save('Expose exact retained skull-base model classes and full-resolution CT inspection')
        reopened = PatientImagingCollection(stage)
        if reopened.identity() != study.identity() or reopened.series(SERIES)['data'] != scan['data']:
            raise ValueError('Prepared study did not preserve the acquired native scan')
        for key, template, code, *_ in LAYERS:
            if reopened.layer(key)['data'] != source.layer(template)['data'] or reopened.layer(key)['label_value'] != code:
                raise ValueError(f'Reopened class changed publisher membership: {key}')
        report = {'schema':'cads465-skull-base-detail-v1', 'case_id':CASE, 'new_source_people':0,
                  'new_inference_runs':0, 'source_manifest':binding(root, base_path),
                  'source_acquisition':acquisition, 'snapshot_id':reopened.identity(),
                  'manifest':{'path':str(manifest.relative_to(root)), 'sha256':sha256_file(stage / manifest.name)},
                  'native_shape':scan['data']['shape'], 'spacing_mm':scan['grid_spacing_mm'],
                  'exact_class_measurements':measurements,
                  'scope':'Unreviewed publisher model categories on their exact source lattice; inspection only.'}
        atomic_json(stage / 'receipt.json', report)
        checked_root(root, expected_uuid=UUID)
        stage.rename(destination)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path)
    parser.add_argument('--register', type=Path)
    args = parser.parse_args()
    root = checked_root(args.root, expected_uuid=UUID, required_bytes=2*1024**3)
    with reconstruction_lock():
        report = build(root)
        if args.register:
            register(root, args.register, report)
    print(safe_path(root, report['manifest']['path']))


if __name__ == '__main__':
    main()
