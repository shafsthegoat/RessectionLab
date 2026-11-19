#!/usr/bin/env python3
"""Reproduce the bounded TAIR 00273 vessel-estimation quality check."""
from __future__ import annotations

import argparse
import fcntl
import json
from pathlib import Path

import nibabel as nib
import numpy as np
from scipy import ndimage as ndi

from resectionlab.data_root import atomic_json, checked_root, safe_path, sha256_file
from resectionlab.patient_anatomy import PatientImagingCollection
from resectionlab.poc_models import run

PILOT = 'TAIR:00273'
STUDY = 'prepared/tair-v1-cta-native/00273'
SERIES = 'tair-cta-display'
METHOD = 'TAIR-00273-vesselfm-cropped-v1'


def source_crop(c, source, brain):
    mask = c.mask(brain)
    spacing = np.asarray(source['grid_spacing_mm'])
    margin = np.ceil(12 / spacing).astype(int)
    bounds = [np.flatnonzero(mask.any(axis=tuple(i for i in range(3) if i != axis))) for axis in range(3)]
    lo = np.maximum(0, np.asarray([v[0] for v in bounds]) - margin)
    hi = np.minimum(mask.shape, np.asarray([v[-1] for v in bounds]) + margin + 1)
    scale = 0.75 / spacing
    shape = np.floor((hi - lo - 1) / scale).astype(int) + 1
    if int(np.prod(shape)) > 26_000_000:
        raise ValueError('Scan-derived ROI exceeds the pilot model cell budget')
    relation = np.diag(np.r_[scale, 1.])
    relation[:3, 3] = lo
    matrix = np.asarray(source['affine']) @ relation
    directory = c.root / 'reconstructions' / 'vesselfm-crop-v1'
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / 'tair-00273-cta-model-input.nii.gz'
    receipt = directory / 'model-input.json'
    identity = {'source_sha256': source['original']['sha256'], 'source_array_sha256': source['data']['sha256'],
                'brain_mask_sha256': brain['data']['sha256'], 'native_roi_start': lo.tolist(),
                'native_roi_stop': hi.tolist(), 'resampled_affine': matrix.tolist(), 'shape': shape.tolist(),
                'spacing_mm': .75, 'margin_mm': 12., 'interpolation': 'linear'}
    if receipt.exists():
        prior = json.loads(receipt.read_text())
        if any(prior.get(key) != value for key, value in identity.items()) or prior['model_input_sha256'] != sha256_file(path):
            raise ValueError('Retained CTA vessel input differs from the verified crop recipe')
    else:
        values = ndi.affine_transform(c.array(source['data']), relation[:3, :3], lo, tuple(shape), order=1, prefilter=False)
        im = nib.Nifti1Image(values, matrix)
        im.header.set_xyzt_units('mm')
        im.set_qform(matrix, code=1)
        im.set_sform(matrix, code=1)
        nib.save(im, path)
        atomic_json(receipt, {**identity, 'model_input_sha256': sha256_file(path),
                              'scope': 'Research-only CTA vessel estimation, no anatomical or route admission'})
    return path, receipt, relation


def read_outputs(record, path):
    if (record['status'] != 'succeeded' or record['model_id'] != 'vesselfm-base'
            or record['input']['sha256'] != sha256_file(path)):
        raise ValueError('Expected a completed source-bound vessel estimate')
    input_image = nib.load(path)
    binary = next(entry for entry in record['outputs'] if entry['name'] == 'vessels_mask.nii.gz')
    scores = record['probability']
    images = []
    for item in (binary, scores):
        if sha256_file(item['path']) != item['sha256']:
            raise ValueError('Retained vessel output differs from its model receipt')
        im = nib.load(item['path'])
        if im.shape != input_image.shape or not np.allclose(im.affine, input_image.affine, atol=1e-5, rtol=0):
            raise ValueError('Vessel model output changes its physical ROI grid')
        images.append(im)
    mask, score = [np.asanyarray(im.dataobj) for im in images]
    if (not np.isin(mask, (0, 1)).all() or not np.isfinite(score).all() or score.min() < 0 or score.max() > 1
            or not np.array_equal(score >= .5, mask.astype(bool)) or not mask.any()):
        raise ValueError('Vessel model outputs fail the binary or equivalent 0.5 score boundary check')
    return input_image, mask, binary


def review(c, brain, skull, mask, matrix, image, out):
    from PIL import Image, ImageDraw

    positives = mask.astype(bool)
    size = int(positives.sum())
    labels, count = ndi.label(positives, structure=np.ones((3, 3, 3), dtype=bool))
    sizes = np.bincount(labels.ravel())[1:]
    membership = {}
    for name, row in [('brain_estimate', brain), ('dense_skull_estimate', skull)]:
        local = ndi.affine_transform(c.mask(row), matrix[:3, :3], matrix[:3, 3], mask.shape, order=0, prefilter=False)
        membership[name] = int(np.count_nonzero(positives & local))
    native = nib.as_closest_canonical(image)
    orientation = nib.orientations.io_orientation(image.affine)
    vessels = nib.orientations.apply_orientation(positives, orientation)
    source = np.asanyarray(native.dataobj)
    result = Image.new('RGB', (1080, 730), '#121820')
    draw = ImageDraw.Draw(result)
    draw.text((16, 10), 'TAIR 00273  |  acquired CTA sampled at 0.75 mm  |  magenta: unclassified vessel estimate', fill='white')
    for axis, label in enumerate(('Sagittal', 'Coronal', 'Axial')):
        index = int(vessels.sum(axis=tuple(i for i in range(3) if i != axis)).argmax())
        plane = np.take(source, index, axis=axis).T[::-1]
        overlay = np.take(vessels, index, axis=axis).T[::-1]
        gray = np.clip((plane - 10) * 255 / 420, 0, 255).astype(np.uint8)
        clean = np.repeat(gray[..., None], 3, axis=-1)
        merged = clean.copy()
        edges = overlay & ~ndi.binary_erosion(overlay)
        merged[edges] = (247, 104, 193)
        for row, (mode, pixels) in enumerate([('Source CTA', clean), ('Model outline', merged)]):
            x, y = 16 + axis * 360, 35 + row * 342
            draw.text((x, y), f'{mode}  |  {label} {index}', fill='white')
            scale = min(330 / pixels.shape[1], 314 / pixels.shape[0])
            extent = round(pixels.shape[1] * scale), round(pixels.shape[0] * scale)
            result.paste(Image.fromarray(pixels).resize(extent, Image.Resampling.BILINEAR), (x, y + 18))
    result.save(out)
    return {'positive_voxels': size, 'volume_mm3': size * 0.75**3, 'components_26': int(count),
            'largest_component_voxels': int(sizes.max()), 'largest_component_fraction': float(sizes.max() / size),
            'intersection_voxels': membership, 'overlay_sha256': sha256_file(out),
            'scope': 'Three source-linked planes selected for the highest positive count per axis; no reference annotation'}


def prepare(root):
    root = checked_root(root, required_bytes=3 * 1024**3)
    directory = safe_path(root, STUDY)
    qc = safe_path(root, 'qc/tair-v1-cta-native/00273/vesselfm-crop-v1')
    qc.mkdir(parents=True, exist_ok=True)
    with (directory / '.intake.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        c = PatientImagingCollection(directory)
        if c.document['patient_id'] != PILOT:
            raise ValueError('Wrong acquired study for this bounded pilot')
        source = c.series(SERIES)
        brain, skull = [next(row for row in c.document['layers'] if row['kind'] == kind and row['source_series'] == SERIES)
                        for kind in ('brain', 'skull')]
        path, recipe, matrix = source_crop(c, source, brain)
        record = run(path, METHOD, 'vesselfm-base', 'CTA', timeout=420, blend='gaussian')
        image, mask, binary = read_outputs(record, path)
        metrics = review(c, brain, skull, mask, matrix, image, qc / 'source-vessel-overlay.png')
        inference = Path(binary['path']).with_name('inference.json')
        result = {'schema': 'ressectionlab-tair-vesselfm-crop-review-v1', 'family': PILOT, 'role': 'DEVELOPMENT',
                  'manifest': f'{STUDY}/patient-anatomy.json', 'manifest_sha256': sha256_file(directory / 'patient-anatomy.json'),
                  'model_receipt': str(inference.relative_to(root)), 'model_receipt_sha256': sha256_file(inference),
                  'mask_sha256': binary['sha256'], 'score_contour_matches_saved_binary': True,
                  'model_input_recipe': str(recipe.relative_to(root)), 'status': 'diagnostic_only',
                  'arterial_venous_identity': 'unclassified', 'clinical_or_instrument_authority': False, **metrics}
        checked_root(root)
        atomic_json(qc / 'verification.json', result)
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path)
    args = parser.parse_args()
    print(json.dumps(prepare(args.root), indent=2))
