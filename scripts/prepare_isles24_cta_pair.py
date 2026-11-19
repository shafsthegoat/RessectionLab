#!/usr/bin/env python3
"""Prepare the selectively acquired ISLES NCCT, registered CTA and CoW study."""
from __future__ import annotations

import argparse
import fcntl
import json
from pathlib import Path

import nibabel as nib
import numpy as np
from scipy import ndimage as ndi

from acquire_isles24_cta_pair import DESTINATION, LICENSE, PUBLISHER, SUBJECT, UUID
from cta_ct_estimates import estimate_anatomy, inspect_surfaces
from resectionlab.anatomy_model_registry import ct_intensities
from resectionlab.data_root import atomic_json, checked_root, safe_path, sha256_file
from resectionlab.patient_anatomy import MAX_ARRAY, PatientImagingCollection, nifti_geometry, nifti_geometry_issue

TIMEPOINT = 'source-session-unknown'
FAMILY = f'ISLES24-v7:{SUBJECT}'
FILES = ('acquired_ncct', 'publisher_registered_cta', 'publisher_automated_cow')


def source_inputs(root):
    receipt = safe_path(root, DESTINATION / 'acquisition.json')
    acquisition = json.loads(receipt.read_text())
    if (acquisition.get('schema') != 'ressectionlab-isles24-v7-acute-source-pair-v1'
            or acquisition.get('source_person_family') != FAMILY or acquisition.get('role') != 'DEVELOPMENT'
            or acquisition.get('publisher_release') != PUBLISHER or acquisition.get('license') != LICENSE
            or acquisition.get('source_timepoint') != 'acute admission ses-01' or not acquisition.get('same_grid')):
        raise ValueError('Expected the verified acute publisher NCCT/CTA pair')
    rows = {row['role']: row for row in acquisition['files']}
    if len(acquisition['files']) != len(FILES) or set(rows) != set(FILES):
        raise ValueError('Unexpected source members for the acute pilot')
    paths = {role: safe_path(root, rows[role]['path']) for role in FILES}
    for role, path in paths.items():
        if not path.is_relative_to(root / DESTINATION) or sha256_file(path) != rows[role]['sha256']:
            raise ValueError(f'Publisher source changed: {path}')
    images = {role: nib.load(path) for role, path in paths.items()}
    reference = images[FILES[0]]
    for role, image in images.items():
        issue = nifti_geometry_issue(nifti_geometry(image))
        if issue or image.shape != reference.shape or not np.allclose(image.affine, reference.affine, atol=1e-5, rtol=0):
            raise ValueError(f'{role} cannot be displayed in the checked NCCT frame: {issue or "grid difference"}')
    return paths, images, acquisition, sha256_file(receipt)


def open_study(directory, paths, images, acquisition, receipt_sha):
    if (directory / 'patient-anatomy.json').exists():
        c = PatientImagingCollection(directory)
        if (c.document['patient_id'] != FAMILY
                or c.document.get('publisher_evidence', {}).get('receipt_sha256') != receipt_sha):
            raise ValueError('Existing native study is not bound to this acquisition')
        return c
    c = PatientImagingCollection.create(directory, FAMILY, timepoint=TIMEPOINT)
    ncct = c.add_series(paths[FILES[0]], series_id='ncct', modality='CT', timepoint=TIMEPOINT)
    cta = c.add_series(paths[FILES[1]], series_id='cta-ncct-space', modality='CTA', timepoint=TIMEPOINT)
    ncct.update(label='Acquired NCCT · acute admission', origin='acquired_source')
    cta.update(label='Publisher-registered CTA · acute admission', origin='publisher_registered_acquired_derivative')
    c.set_registration(cta['id'], np.eye(4), status='source-provided',
                       method='Publisher CTA is sampled in the acquired NCCT frame; registration landmarks unavailable')
    labels_image = images[FILES[2]]
    labels = np.asarray(labels_image.dataobj)
    if not np.isin(labels, np.arange(14)).all() or not np.any(labels):
        raise ValueError('Unexpected or empty publisher automated CoW values')
    values = np.unique(labels[labels != 0]).astype(int).tolist()
    layer = c.add_layer((labels > 0).astype(np.uint8), labels_image.affine,
                        layer_id='publisher-estimated-cow', source_series=cta['id'], kind='artery', role='context',
                        label='Circle of Willis · publisher automated', color='#ef6b66', provenance={
                            'kind': 'image_derived_estimate', 'source': c.original(paths[FILES[2]]),
                            'basis': 'Publisher automated categorical mask from CTA, preserved on the NCCT grid',
                            'display': 'Union of positive publisher values with original categories retained',
                            'publisher_values_present': values, 'evidence_use': 'context_only',
                        })
    layer.update(units_status='mm', coverage='Partial arterial Circle of Willis; sensitivity unassessed')
    ncct['source_license'] = cta['source_license'] = LICENSE
    c.document['publisher_evidence'] = {'release': PUBLISHER, 'receipt_sha256': receipt_sha, 'license': LICENSE,
                                        'role': acquisition['role'], 'session': acquisition['source_timepoint']}
    c.document['missing'] = ['Original native CTA grid', 'Supplied brain and skull segmentation',
                             'Complete cerebral arteries and veins', 'Independent registration landmarks',
                             'Structural MRI, diffusion and functional mapping', 'Surgical timeline and route authority']
    c.save('Prepare source-bound acute NCCT, registered CTA and publisher CoW display')
    return PatientImagingCollection(directory)


def retain_native_cta(root, collection, receipt_sha):
    path = safe_path(root, DESTINATION / 'native-cta-extension.json')
    if not path.exists():
        return {'status': 'publisher native CTA not in the verified subset'}
    extension = json.loads(path.read_text())
    source = extension.get('native_cta', {})
    if (extension.get('schema') != 'ressectionlab-isles24-v7-native-cta-extension-v1'
            or extension.get('source_person_family') != FAMILY or extension.get('base_acquisition_sha256') != receipt_sha
            or source.get('role') != 'acquired_native_cta'):
        raise ValueError('Native CTA extension differs from the existing patient or source pair')
    original = safe_path(root, source['path'])
    if (not original.is_relative_to(root / DESTINATION / 'train' / 'raw_data' / SUBJECT / 'ses-01')
            or original.stat().st_size != source['bytes'] or sha256_file(original) != source['sha256']):
        raise ValueError('Original native CTA differs from the verified extension')
    image = nib.load(original)
    issue = nifti_geometry_issue(nifti_geometry(image))
    if len(image.shape) != 3 or np.prod(image.shape) * 4 > MAX_ARRAY:
        issue = 'Original native grid exceeds the application scalar array budget'
    extension_sha = sha256_file(path)
    if issue:
        return {'status': 'original preserved; display intake withheld', 'reason': issue, 'receipt_sha256': extension_sha}
    prior = next((row for row in collection.document['series'] if row['id'] == 'native-cta'), None)
    if prior is not None:
        if prior['original']['sha256'] != source['sha256'] or prior.get('extension_sha256') != extension_sha:
            raise ValueError('Existing original CTA differs from the verified extension')
    else:
        row = collection.add_series(original, series_id='native-cta', modality='CTA', timepoint=TIMEPOINT)
        row.update(label='Acquired CTA · native grid', origin='acquired_source', source_license=LICENSE,
                   extension_sha256=extension_sha)
        collection.document['missing'].remove('Original native CTA grid')
        collection.save('Retain same-patient CTA in its independent native grid')
    return {'status': 'native acquired CTA view available', 'receipt_sha256': extension_sha,
            'original_sha256': source['sha256'], 'original_shape': list(image.shape),
            'fusion': 'Publisher transform to NCCT unavailable for the original native grid'}


def review_pair(c, images, out):
    references = [c.series('ncct'), c.series('cta-ncct-space')]
    if not np.array_equal(c.registration_matrix(references[1]['id'], require_review=True, require_geometry=True), np.eye(4)):
        raise ValueError('The publisher grid unexpectedly acquired a local transform')
    arrays = [c.array(row['data']) for row in references]
    layer = c.layer('publisher-estimated-cow')
    c.require_native_geometry(layer)
    mask = c.mask(layer)
    if not np.array_equal(mask, np.asarray(images[FILES[2]].dataobj) > 0):
        raise ValueError('Native overlay differs from the original publisher categories')
    for role, array in zip(FILES, arrays):
        for index in (0, array.shape[2] // 2, array.shape[2] - 1):
            original = np.asarray(images[role].dataobj[:, :, index], dtype=np.float32)
            if not np.array_equal(array[:, :, index], original):
                raise ValueError(f'Native source pixels differ from {role}')
    labeled, components = ndi.label(mask, structure=np.ones((3, 3, 3), dtype=bool))
    sizes = np.bincount(labeled.ravel())[1:]
    count = int(mask.sum())
    del labeled
    qc = {'same_publisher_grid': True, 'selected_slices_match_source_values': True,
          'registration': 'Publisher sampling on NCCT grid; independent landmark error unavailable',
          'publisher_foreground_voxels': count, 'cow_components_26': components,
          'largest_cow_component_fraction': float(sizes.max() / count)}
    subtract = all(ct_intensities(c, row) for row in references)
    qc['paired_subtraction'] = subtraction_metrics(*arrays, mask) if subtract else {
        'status': 'refused_due_to_unqualified_intensity_scale',
    }
    render_review(arrays, mask, references[0]['affine'], out, subtract)
    return qc


def subtraction_metrics(ncct, cta, mask):
    dense_ncct, dense_cta = ncct >= 350, cta >= 350
    overlap = dense_ncct & dense_cta
    dense = int(dense_ncct.sum() + dense_cta.sum())
    return {
        'status': 'Exploratory CTA minus NCCT on publisher sampling, display only',
        'scale_basis': 'CT-family HU-like signature; source calibration unconfirmed',
        'median_difference_in_cow_voxels': float(np.median(cta[mask] - ncct[mask])),
        'median_difference_where_both_dense': float(np.median(cta[overlap] - ncct[overlap])) if overlap.any() else None,
        'dense_350_overlap_dice': 2 * int(overlap.sum()) / dense if dense else None,
        'cow_in_ncct_dense_fraction': int(np.count_nonzero(mask & dense_ncct)) / int(mask.sum()),
        'scope': 'Dense signal includes bone, calcification and sampling effects. Independent registration error unmeasured.',
    }


def render_review(arrays, mask, matrix, out, subtract):
    from PIL import Image, ImageDraw

    matrix = np.asarray(matrix)
    orientation = nib.orientations.io_orientation(matrix)
    canonical = [nib.orientations.apply_orientation(array, orientation) for array in [*arrays, mask]]
    spacing = nib.affines.voxel_sizes(matrix @ nib.orientations.inv_ornt_aff(orientation, mask.shape))
    centers = np.rint(np.median(np.argwhere(canonical[2]), axis=0)).astype(int)
    roles = ['Acquired NCCT', 'Publisher-registered CTA', 'CTA minus NCCT · exploratory']
    names = ['Sagittal index', 'Coronal index', 'Axial index']
    view = Image.new('RGB', (990, 840), '#121820')
    draw = ImageDraw.Draw(view)
    draw.text((12, 9), f'{FAMILY}  |  acute admission  |  Red: publisher automated CoW', fill='white')
    for axis, index in enumerate(centers):
        a, b, m = [np.take(array, index, axis=axis) for array in canonical]
        edge = m & ~ndi.binary_erosion(m)
        horizontal, vertical = np.delete(spacing, axis)
        for column, (name, signal) in enumerate(zip(roles, (a, b, b - a))):
            x, y = column * 330 + 12, axis * 270 + 33
            draw.text((x, y), f'{name}  |  {names[axis]} {index}', fill='white')
            if column == 2 and not subtract:
                draw.text((x, y + 20), 'Unqualified intensity scale', fill='#ffb870')
                continue
            lo, hi = (-70, 180) if column == 2 else (-100, 450)
            gray = np.clip((signal.T[::-1] - lo) * (255 / (hi - lo)), 0, 255).astype(np.uint8)
            pixels = np.repeat(gray[..., None], 3, axis=-1)
            pixels[edge.T[::-1]] = (255, 80, 76)
            scale = min(310 / (gray.shape[1] * horizontal), 240 / (gray.shape[0] * vertical))
            size = (round(gray.shape[1] * horizontal * scale), round(gray.shape[0] * vertical * scale))
            view.paste(Image.fromarray(pixels).resize(size, Image.Resampling.BILINEAR), (x, y + 18))
    view.save(out)


def prepare(root, estimate=False, recovery=None):
    root = checked_root(root, required_bytes=1024**3, expected_uuid=UUID)
    paths, images, acquisition, receipt_sha = source_inputs(root)
    directory = safe_path(root, f'prepared/isles24-v7-native/{SUBJECT}')
    review = safe_path(root, f'qc/isles24-v7-native/{SUBJECT}')
    directory.mkdir(parents=True, exist_ok=True)
    review.mkdir(parents=True, exist_ok=True)
    with (directory / '.intake.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        c = open_study(directory, paths, images, acquisition, receipt_sha)
        manifest = directory / 'patient-anatomy.json'
        for role, series_id in zip(FILES, ('ncct', 'cta-ncct-space')):
            row = c.series(series_id)
            if sha256_file(paths[role]) != row['original']['sha256']:
                raise ValueError('Existing series differs from the publisher acquisition')
        native_cta = retain_native_cta(root, c, receipt_sha)
        if estimate or recovery:
            estimate_anatomy(root, c, 'ncct', recovery)
        planes = review / 'ncct-cta-cow-overlays.png'
        qc = review_pair(c, images, planes)
        result = {'schema': 'ressectionlab-isles24-v7-native-pair-qc-v1', 'family': FAMILY,
                  'role': 'DEVELOPMENT', 'license': LICENSE, 'source_acquisition_sha256': receipt_sha,
                  'native_manifest': str(manifest.relative_to(root)), 'native_manifest_sha256': sha256_file(manifest),
                  'overlay': str(planes.relative_to(root)), 'overlay_sha256': sha256_file(planes),
                  'native_cta': native_cta, 'missing': c.document['missing'],
                  'ncct_estimated_surfaces': inspect_surfaces(c, 'ncct'), **qc}
        checked_root(root, expected_uuid=UUID)
        atomic_json(review / 'verification.json', result)
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-root', type=Path, help='Defaults to RESSECTIONLAB_DATA_ROOT.')
    model = parser.add_mutually_exclusive_group()
    model.add_argument('--estimate-skull-brain', action='store_true', help='Use the installed CT model on the NCCT.')
    model.add_argument('--recover-ct-worker', type=Path, help='Validate retained masks from a zero-exit CT worker.')
    args = parser.parse_args()
    print(json.dumps(prepare(args.data_root, args.estimate_skull_brain, args.recover_ct_worker), indent=2))
