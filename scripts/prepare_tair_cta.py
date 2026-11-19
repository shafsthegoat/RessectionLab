#!/usr/bin/env python3
"""Prepare an original TAIR CTA and a bounded source-linked native study."""
import argparse
import fcntl
import json
from pathlib import Path
from tempfile import TemporaryDirectory

import nibabel as nib
import numpy as np

from acquire_tair_cta import FILENAME, PILOTS, RELEASE
from cta_ct_estimates import estimate_anatomy, inspect_surfaces
from resectionlab.anatomy_import import import_nifti_series
from resectionlab.anatomy_model_registry import ct_intensities
from resectionlab.data_root import atomic_json, checked_root, safe_path, sha256_file
from resectionlab.patient_anatomy import MAX_ARRAY, PatientImagingCollection, nifti_geometry, nifti_geometry_issue

TIMEPOINT = 'source-session-unknown'
SERIES = 'tair-cta-display'


def display_image(image):
    # Keep sampled intensities at their exact physical locations.
    plane_bytes = image.shape[0] * image.shape[1] * 4
    if plane_bytes > MAX_ARRAY:
        raise ValueError('Even one CTA slice exceeds the application array budget')
    limit = MAX_ARRAY // plane_bytes
    stride = (image.shape[2] + limit - 1) // limit
    stride_map = np.diag([1, 1, stride, 1])
    values = np.asanyarray(image.dataobj)[..., ::stride]
    if not np.isfinite(values).all():
        raise ValueError('CTA has nonfinite sampled intensities')
    header = image.header.copy()
    header.set_data_dtype(values.dtype)
    volume = nib.Nifti1Image(values, image.affine @ stride_map, header=header)
    for form in ('qform', 'sform'):
        matrix, code = getattr(image, 'get_' + form)(coded=True)
        if code:
            getattr(volume, 'set_' + form)(matrix @ stride_map, code=int(code))
    return volume, stride


def open_study(directory, family, source, acquisition, image):
    manifest = directory / 'patient-anatomy.json'
    if manifest.exists():
        collection = PatientImagingCollection(directory)
        source_link = collection.series(SERIES)['publisher_source']
        if (collection.document['patient_id'] != family
                or source_link['sha256'] != acquisition['sha256']
                or sha256_file(collection.path(source_link['path'])) != source_link['sha256']):
            raise ValueError('Existing native study and publisher source differ')
        return collection
    collection = PatientImagingCollection.create(directory, family, timepoint=TIMEPOINT)
    original = collection.original(source)
    with TemporaryDirectory(prefix='.tair-display-', dir=directory) as temp:
        volume, stride = display_image(image)
        display = Path(temp) / 'CTA-native-sample-display.nii.gz'
        nib.save(volume, display)
        del volume
        sidecar = display.with_name('CTA-native-sample-display.json')
        sidecar.write_text(json.dumps({'SubjectID': family, 'Modality': 'CT',
                                       'IdentityBasis': 'Publisher S3 subject prefix',
                                       'SourceURL': acquisition['source_url']}))
        row = import_nifti_series(collection, display, series_id=SERIES, modality='CTA',
                                  timepoint=TIMEPOINT, source_patient_id=family, sidecar=sidecar)
    row.update(label='TAIR CTA · sampled source display', publisher_source=original,
               publisher_source_grid=nifti_geometry(image), display_sampling_ijk=[1, 1, stride],
               publisher_dataset=RELEASE, source_license='CC-BY-4.0')
    collection.document['missing'] = [
        'Native noncontrast CT and structural MRI', 'Supplied skull and brain segmentations',
        'Qualifying arterial and venous annotations', 'Ventricular and regional anatomy',
        'Patient-specific white-matter and functional information', 'Surgical timepoint and route authority',
    ]
    collection.save('Import TAIR CTA and retain unmodified publisher source')
    return collection


def source_review(collection, out):
    from PIL import Image, ImageDraw
    from scipy.ndimage import binary_erosion

    row = collection.series(SERIES)
    data = collection.array(row['data'])
    matrix = np.asarray(row['affine'])
    volume = nib.as_closest_canonical(nib.Nifti1Image(data, matrix))
    ras = np.asanyarray(volume.dataobj)
    sampling = row['display_sampling_ijk']
    centers = [ras.shape[0] // 2, ras.shape[1] // 2, round(ras.shape[2] * 0.65)]
    planes = [ras[centers[0], :, :], ras[:, centers[1], :], ras[:, :, centers[2]]]
    contours = []
    for layer in collection.document['layers']:
        if layer['kind'] not in {'brain', 'skull'}:
            continue
        mask = collection.mask(layer)
        if mask.shape != data.shape or not np.allclose(layer['affine'], matrix, atol=1e-5):
            raise ValueError('CTA estimate does not use the display grid')
        oriented = nib.as_closest_canonical(nib.Nifti1Image(mask.astype(np.uint8), matrix)).dataobj
        edges = []
        for axis, index in enumerate(centers):
            section = np.take(oriented, index, axis=axis).astype(bool)
            edges.append((section & ~binary_erosion(section)).T[::-1])
        color = (84, 218, 159) if layer['kind'] == 'brain' else (88, 175, 241)
        contours.append((color, edges))
    labels = ['Sagittal · midline', 'Coronal · center', 'Axial · superior sample']
    output = Image.new('RGB', (960, 630), '#10151b')
    draw = ImageDraw.Draw(output)
    draw.text((12, 10), collection.document['patient_id'] + ' | Original grid sampling ' + str(sampling), fill='white')
    for win, (name, low, high) in enumerate([('Soft tissue', 0, 100), ('Dense structure', -100, 1200)]):
        for axis, (plane, label) in enumerate(zip(planes, labels)):
            x, y = axis * 320 + 12, win * 300 + 36
            draw.text((x, y), name + ' | ' + label, fill='white')
            pixels = np.clip((plane.T[::-1] - low) * (255 / (high - low)), 0, 255).astype(np.uint8)
            spacing = nib.affines.voxel_sizes(volume.affine)
            horizontal, vertical = np.delete(spacing, axis)
            scale = min(296 / (pixels.shape[1] * horizontal), 268 / (pixels.shape[0] * vertical))
            rgb = np.repeat(pixels[..., None], 3, axis=-1)
            for color, edges in contours:
                rgb[edges[axis]] = color
            rendered = Image.fromarray(rgb).resize(
                (round(pixels.shape[1] * horizontal * scale), round(pixels.shape[0] * vertical * scale)),
                Image.Resampling.BILINEAR,
            )
            output.paste(rendered, (x, y + 18))
    output.save(out)
    faces = {
        f'{axis}{side}': int(np.count_nonzero(np.take(ras, index, axis=dimension) >= 250))
        for dimension, axis in enumerate('RAS') for side, index in [('-', 0), ('+', -1)]
    }
    return {'orthogonal_review_png': out.name, 'ras_slice_indices': centers,
            'face_sample_counts_at_or_above_250_scaled_units': faces,
            'intensity_signature_supports_ct_hu': bool(ct_intensities(collection, row))}


def prepare(root, subject, estimates=False, recover=None):
    family = f'TAIR:{subject}'
    source_dir = Path('sources/tair-v1-cta-selective') / subject
    acquisition_path = safe_path(root, source_dir / 'acquisition.json')
    acquisition = json.loads(acquisition_path.read_text())
    source = safe_path(root, source_dir / FILENAME)
    if (acquisition['schema'] != 'ressectionlab-tair-cta-source-v1'
            or acquisition['source_person_family'] != family
            or acquisition['path'] != str(source.relative_to(root))
            or source.stat().st_size != acquisition['bytes']
            or sha256_file(source) != acquisition['sha256']):
        raise ValueError(f'Unverified acquisition: {family}')
    image = nib.load(source)
    geometry = nifti_geometry(image)
    issue = nifti_geometry_issue(geometry)
    if len(image.shape) != 3 or issue:
        raise ValueError(f'Unsupported TAIR geometry: {issue or image.shape}')
    checked_root(root, required_bytes=2 * MAX_ARRAY + 2 * acquisition['bytes'])
    directory = safe_path(root, f'prepared/tair-v1-cta-native/{subject}')
    directory.mkdir(parents=True, exist_ok=True)
    with (directory / '.intake.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        collection = open_study(directory, family, source, acquisition, image)
        collection = PatientImagingCollection(directory)
        row = collection.series(SERIES)
        stride = row['display_sampling_ijk'][2]
        display = collection.array(row['data'])
        if (display.shape != (*image.shape[:2], len(range(0, image.shape[2], stride)))
                or not np.allclose(row['affine'], image.affine @ np.diag([1, 1, stride, 1]), atol=1e-5)):
            raise ValueError('Native display grid differs from the exact source sampling recipe')
        checked_slices = [0, display.shape[2] // 2, display.shape[2] - 1]
        for index in checked_slices:
            original = np.asarray(image.dataobj[:, :, index * stride], dtype=np.float32)
            if not np.array_equal(original, display[:, :, index]):
                raise ValueError('Native display intensities differ from sampled publisher values')
        del display
        if estimates or recover:
            estimate_anatomy(root, collection, SERIES, recover)
        review_dir = safe_path(root, f'qc/tair-v1-cta-native/{subject}')
        review_dir.mkdir(parents=True, exist_ok=True)
        review = source_review(collection, review_dir / 'source-planes.png')
        manifest = directory / 'patient-anatomy.json'
        result = {
            'schema': 'ressectionlab-tair-cta-native-review-v1',
            'family': family, 'role': 'DEVELOPMENT', 'source_sha256': acquisition['sha256'],
            'source_receipt_sha256': sha256_file(acquisition_path), 'source_shape': list(image.shape),
            'source_affine': image.affine.tolist(), 'display_shape': row['data']['shape'],
            'display_spacing_mm': row['grid_spacing_mm'], 'display_sampling_ijk': row['display_sampling_ijk'],
            'exact_source_match_display_slice_indices': checked_slices,
            'native_manifest': str(manifest.relative_to(root)), 'native_manifest_sha256': sha256_file(manifest),
            'source_review': review, 'capabilities': ['guarded_native_scalar_reader', 'source_grid_inspection'],
            'missing': collection.document['missing'], 'catalog_admission': 'unpublished development study',
        }
        if collection.document['layers']:
            result['estimated_surfaces'] = inspect_surfaces(collection, SERIES)
        checked_root(root)
        atomic_json(review_dir / 'verification.json', result)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, help='Verified external root. Defaults to RESSECTIONLAB_DATA_ROOT.')
    model = parser.add_mutually_exclusive_group()
    model.add_argument('--estimate-skull-brain', action='store_true', help='Use the installed CT model for one pilot.')
    model.add_argument('--recover-ct-worker', type=Path, help='Validate masks from an exited CT worker with an inherited log pipe.')
    parser.add_argument('subjects', nargs='*', help='Pinned pilot IDs. Omit for all three.')
    args = parser.parse_args()
    subjects = args.subjects or PILOTS
    if set(subjects) - PILOTS.keys() or len(subjects) != len(set(subjects)):
        parser.error('Choose distinct pilot IDs from ' + ', '.join(PILOTS))
    if (args.estimate_skull_brain or args.recover_ct_worker) and len(subjects) != 1:
        parser.error('Choose one pilot for model inference')
    root = checked_root(args.root)
    for subject in subjects:
        result = prepare(root, subject, args.estimate_skull_brain, args.recover_ct_worker)
        print(json.dumps({key: result[key] for key in ('family', 'source_shape', 'display_shape', 'native_manifest')}), flush=True)


if __name__ == '__main__':
    main()
