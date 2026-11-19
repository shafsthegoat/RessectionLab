#!/usr/bin/env python3
"""Prepare source-linked CADS 465 cranial CT and inspection layers."""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from tempfile import TemporaryDirectory

import nibabel as nib
import numpy as np
from PIL import Image, ImageDraw
from scipy.ndimage import binary_erosion

from acquire_cads_brainct import ASSETS, BASENAME, FAMILY, LICENSE, REVISION, SCHEMA, SOURCE, checked_member
from build_cta_native_showcases import register
from data_integrated_hosts import UUID, binding, checked_file
from data_rebuild_proxy_units import write_nifti1_mm_copy
from resectionlab.anatomy_geometry import mesh
from resectionlab.data_root import atomic_json, checked_root, safe_path, sha256_file
from resectionlab.patient_anatomy import PatientImagingCollection, nifti_geometry, reconstruction_lock

CASE = 'CADS-BRAINCT-465-NATIVE-v1'
DESTINATION = f'prepared/cads-brainct-1mm-v1/{CASE}'
TIMEPOINT = 'source-session-unknown'
SERIES = 'ncct'
INITIAL = ('brain', 'head-bone', 'brainstem')
LAYERS = (
    ('brain', 'part_553', (9,), 'brain', 'CADS brain estimate', '#d98ca7'),
    ('head-bone', 'part_557', (6, 7), 'skull', 'CADS compact and spongy head bone', '#e8cdae'),
    ('brainstem', 'part_558', (5,), 'brain-region', 'CADS brainstem estimate', '#a0d9d3'),
    ('gray-matter', 'part_557', (2,), 'brain-region', 'CADS gray matter estimate', '#e088af'),
    ('white-matter', 'part_557', (1,), 'brain-region', 'CADS white matter estimate', '#b5aacd'),
    ('csf', 'part_557', (3,), 'brain-region', 'CADS CSF estimate', '#7bb5ed'),
    ('carotids', 'part_558', (1, 2), 'artery', 'CADS carotid estimates', '#f77666'),
)
COVERAGE = {
    'head-bone': 'Head-bone union may include upper cervical, facial and dental bone.',
    'csf': 'The CSF class does not distinguish ventricular and extra-axial CSF.',
    'carotids': 'Limited carotid estimates on noncontrast CT.',
}
LIMITS = [
    'All labels are publisher model outputs. Source ground truth and segmentation accuracy are unavailable.',
    *COVERAGE.values(),
    'Dedicated skull-base, cerebellar and branch-specific intracranial vessel labels are unavailable.',
    'The generic vascular estimate is retained only in the original categorical file.',
    'The study supports source and surface inspection. No operative task or multimodal registration was reviewed.',
]


def source_inputs(root):
    path = safe_path(root, f'{SOURCE}/acquisition.json')
    receipt = json.loads(path.read_text())
    if (receipt['schema'] != SCHEMA or receipt['source_family'] != FAMILY or receipt['revision'] != REVISION
            or receipt['role'] != 'DEVELOPMENT' or receipt['license'] != LICENSE
            or set(receipt['members']) != set(ASSETS)):
        raise ValueError('Expected pinned CADS Development receipt')
    paths = {}
    for key, (url, size, digest) in ASSETS.items():
        row = receipt['members'][key]
        if (row['url'], row['bytes'], row['sha256']) != (url, size, digest):
            raise ValueError(f'Different publisher receipt for {key}')
        original = safe_path(root, row['path'])
        checked_member(original, size, digest)
        paths[key] = original
    with paths['image_index'].open(newline='') as stream:
        matches = [r for r in csv.DictReader(stream) if r['Image ID'] == f'{BASENAME}.nii.gz']
    if len(matches) != 1:
        raise ValueError('Expected one exact image index row for case 465')
    index = matches[0]
    shape = tuple(int(index[f'Size {axis}']) for axis in 'XYZ')
    spacing = tuple(float(index[f'Spacing {axis}']) for axis in 'XYZ')
    images = {key: nib.load(path) for key, path in paths.items() if path.name.endswith('.nii.gz')}
    ct = images['ncct']
    for key, image in images.items():
        geometry = nifti_geometry(image)
        if (image.shape != shape or not np.array_equal(nib.affines.voxel_sizes(image.affine), spacing)
                or not np.array_equal(image.affine, ct.affine) or geometry['frame_status'] != 'coded'
                or geometry['spatial_units'] != 'unknown'):
            raise ValueError(f'Native geometry does not match the exact source lattice: {key}')
    return paths, images, binding(root, path), index


def establish(stage, paths, images, receipt, index):
    collection = PatientImagingCollection.create(stage, CASE, timepoint=TIMEPOINT)
    raw = collection.original(paths['ncct'])
    form = {'source_spatial_unit': 'undeclared', 'display_spatial_unit': 'mm',
            'basis': 'Pinned publisher image index and BrainCT-1mm protocol',
            'image_index': collection.original(paths['image_index']), 'exact_index_row': index,
            'affine_and_voxel_payload_unchanged': True}
    with TemporaryDirectory(prefix='.display-header-', dir=stage) as tmp:
        derived = Path(tmp) / 'CT-465-source-mm-display.nii.gz'
        write_nifti1_mm_copy(paths['ncct'], derived)
        scan = collection.add_series(derived, series_id=SERIES, modality='CT', timepoint=TIMEPOINT, patient_id=CASE)
    scan.update(label='CADS · acquired noncontrast CT', publisher_source=raw, unit_interpretation=form,
                publisher_dataset='https://huggingface.co/datasets/huggingface/CADS-dataset',
                source_license=LICENSE)
    if not np.array_equal(scan['affine'], images['ncct'].affine):
        raise ValueError('Display unit declaration must preserve the exact source affine')
    collection.require_native_geometry(scan)
    collection.preview_pyramid(SERIES, levels=(2,))
    for key, maximum_code in [('part_553', 18), ('part_557', 9), ('part_558', 29)]:
        original = collection.original(paths[key])
        values = np.asanyarray(images[key].dataobj)
        if values.dtype != np.uint8 or values.max() > maximum_code:
            raise ValueError(f'Invalid categorical values in {key}')
        for layer_id, part, codes, kind, label, color in LAYERS:
            if part != key:
                continue
            data = values if len(codes) == 1 else np.isin(values, codes).astype(np.uint8)
            row = collection.add_layer(data, scan['affine'], layer_id=layer_id, source_series=SERIES,
                                       kind=kind, role='context', color=color, label=label,
                                       label_value=codes[0] if len(codes) == 1 else 1,
                                       provenance={'kind': 'supplied_annotation', 'source': original,
                                                   'source_label_values': list(codes),
                                                   'source_method': 'publisher segmentation model estimate',
                                                   'evidence_use': 'native-lattice inspection only'})
            row.update(units_status='unknown', review_status='publisher model estimate',
                       coverage=COVERAGE.get(layer_id, LIMITS[0]))
    collection.document['display'] = {'initial_layers': list(INITIAL),
                                      'cursor_ras_mm': nib.affines.apply_affine(scan['affine'], [247, 227, 98]).tolist()}
    collection.document['data_case'] = {'source_family_id': FAMILY, 'role': 'DEVELOPMENT',
                                        'case_type': 'display_companion', 'clinical_validation': False,
                                        'source_receipt': receipt, 'limits': LIMITS}
    collection.document['publisher_metadata'] = {key: collection.original(paths[key])
                                                 for key in ('readme', 'label_map')}
    collection.document['missing'] = LIMITS[1:]
    collection.save('Prepare CADS case 465 from supplied CT and published model outputs')


def report_study(stage, images):
    collection = PatientImagingCollection(stage)
    scan = collection.series(SERIES)
    source = images['ncct']
    ct = collection.array(scan['data'])
    if (collection.identity() != collection.document['snapshot_id']
            or not np.array_equal(ct, source.get_fdata(dtype=np.float32))
            or not np.isfinite(ct).all()):
        raise ValueError('Reopened study must preserve the exact acquired source values')
    raw = scan['publisher_source']
    if sha256_file(collection.path(raw['path'])) != raw['sha256']:
        raise ValueError('Retained original CT changed')
    layer_arrays = {part: np.asanyarray(images[part].dataobj)
                    for part in sorted({part for _, part, *_ in LAYERS})}
    layers = {}
    for layer_id, part, codes, _, _, _ in LAYERS:
        row = collection.layer(layer_id)
        mask = collection.mask(row)
        original = row['provenance']['source']
        if (row['role'] != 'context' or row['source_series'] != SERIES
                or not np.array_equal(row['affine'], scan['affine']) or not mask.any()
                or not np.array_equal(mask, np.isin(layer_arrays[part], codes))
                or sha256_file(collection.path(original['path'])) != original['sha256']):
            raise ValueError(f'{layer_id} did not restore the exact publisher memberships')
        layer = {'voxels': int(mask.sum()), 'source_codes': list(codes),
                 'face_foreground_voxels': {f'{axis}_{side}': int(np.take(mask, index, axis=dimension).sum())
                                            for dimension, axis in enumerate('ijk')
                                            for side, index in [('first', 0), ('last', -1)]}}
        if layer_id in INITIAL:
            surface = mesh(collection, row)
            layer['surface'] = {'triangles': len(surface['indices']),
                                'max_displacement_mm': float(surface['max_displacement_mm']),
                                'policy': str(surface['policy_version'])}
            del surface
        layers[layer_id] = layer
        print(f'Checked {layer_id} ({layer["voxels"]:,} voxels)', flush=True)
    paint_source_planes(collection, stage / 'source-planes.png')
    brain = collection.mask(collection.layer('brain'))
    bone = collection.mask(collection.layer('head-bone'))
    return {'snapshot_id': collection.identity(), 'native_shape': list(ct.shape),
            'native_spacing_mm': scan['grid_spacing_mm'], 'affine_ras_mm': scan['affine'],
            'layers': layers, 'brain_bone_overlap_voxels': int(np.count_nonzero(brain & bone)),
            'bone_fraction_at_or_above_250_ct_units': float(np.mean(ct[bone] >= 250)),
            'new_inference_runs': 0, 'limits': LIMITS}


def paint_source_planes(collection, destination):
    row = collection.series(SERIES)
    image = collection.array(row['data'])[::-1]
    layers = [(collection.mask(collection.layer(key))[::-1], color) for key, color in
              [('brain', (255, 138, 185)), ('head-bone', (255, 201, 107)), ('brainstem', (91, 224, 213))]]
    image_out = Image.new('RGB', (1280, 740), '#111820')
    draw = ImageDraw.Draw(image_out)
    draw.text((14, 12), 'CADS 465  |  Acquired noncontrast CT with published model contours', fill='white')
    planes = [(2, 72, 'Axial skull base'), (2, 125, 'Axial supratentorial'),
              (1, 225, 'Coronal'), (0, 255, 'Sagittal')]
    for column, (axis, index, title) in enumerate(planes):
        ct = np.take(image, index, axis=axis)
        horizontal, vertical = np.delete(row['grid_spacing_mm'], axis)
        x = column * 320 + 14
        scale = min(294 / (ct.shape[0] * horizontal), 300 / (ct.shape[1] * vertical))
        size = (round(ct.shape[0] * horizontal * scale), round(ct.shape[1] * vertical * scale))
        for line, (name, lo, hi) in enumerate([('Bone window', -100., 1300.), ('Soft tissue window', 0., 90.)]):
            y = 42 + 341 * line
            gray = np.clip((ct.T[::-1] - lo) * 255 / (hi - lo), 0, 255).astype(np.uint8)
            rgb = np.repeat(gray[..., None], 3, axis=-1)
            for mask, color in layers:
                section = np.take(mask, index, axis=axis)
                rgb[(section & ~binary_erosion(section)).T[::-1]] = color
            draw.text((x, y), f'{title}  |  {name}', fill='white')
            image_out.paste(Image.fromarray(rgb).resize(size, Image.Resampling.BILINEAR), (x, y + 19))
    image_out.save(destination)


def build(root):
    paths, images, acquisition, index = source_inputs(root)
    final = safe_path(root, DESTINATION)
    manifest = final / 'patient-anatomy.json'
    if final.exists():
        report = json.loads((final / 'receipt.json').read_text())
        if checked_file(root, report['manifest']) != manifest or report['acquisition'] != acquisition:
            raise ValueError('Existing native study differs from the pinned acquisition')
        checked_file(root, report['source_planes'])
        return report
    final.parent.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(prefix='.cads-native-', dir=final.parent) as temporary:
        stage = Path(temporary) / CASE
        establish(stage, paths, images, acquisition, index)
        report = report_study(stage, images)
        report.update(schema='cads-brainct-native-v1', case_id=CASE, source_family=FAMILY,
                      acquisition=acquisition, role='DEVELOPMENT')
        for key, name in [('manifest', manifest.name), ('source_planes', 'source-planes.png')]:
            report[key] = {'path': str((final / name).relative_to(root)), 'sha256': sha256_file(stage / name)}
        atomic_json(stage / 'receipt.json', report)
        stage.rename(final)
    return report


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path)
    parser.add_argument('--register', type=Path)
    args = parser.parse_args()
    root = checked_root(args.root, expected_uuid=UUID, required_bytes=2 * 1024**3)
    with reconstruction_lock():
        report = build(root)
    if args.register:
        register(root, args.register, report)
    print(safe_path(root, DESTINATION) / 'receipt.json')
