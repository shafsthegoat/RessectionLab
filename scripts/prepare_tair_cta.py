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


def recover_ct_worker(root, collection, row, receipt):
    """Validate finished masks from a zero-exit model attempt with a leaked log pipe."""
    from resectionlab import poc_models
    from resectionlab.anatomy_model_identity import poc_worker_identity
    from resectionlab.poc_worker_supervision import MEMORY_LIMIT_BYTES

    model = 'totalsegmentator-skull'
    receipt = receipt.resolve()
    output = receipt.parent
    if (receipt.name != 'inference.json' or not output.is_relative_to(root / 'cache' / 'poc-model-results')
            or receipt.stat().st_size > 1024**2):
        raise ValueError('Choose an existing supervised model receipt on the guarded cache')
    receipt_sha = sha256_file(receipt)
    record = json.loads(receipt.read_text())
    request = json.loads((output / 'request.json').read_text())
    worker_path = output / 'worker.json'
    worker = json.loads(worker_path.read_text())
    source = collection.path(row['original']['path'])
    source_sha = sha256_file(source)
    launcher_sha = sha256_file(Path(poc_models.__file__))
    if (record.get('schema') != 'resectionlab-poc-inference-v1' or record.get('status') != 'failed'
            or record.get('failure') != 'worker_output_not_closed' or record.get('exit_code') != 0
            or record.get('model_id') != model or request.get('model_id') != model
            or request.get('modality') != 'CTA' or request.get('image') != str(source)
            or request.get('case') != record.get('case') or request.get('device') != record.get('device')
            or request.get('output') != str(output) or record['input'].get('path') != str(source)
            or record['input'].get('modality') != 'CTA'
            or source_sha != record['input'].get('sha256') or source_sha != row['original']['sha256']
            or record.get('launcher_sha256') != launcher_sha
            or record['identity'].get('runner_sha256') != poc_worker_identity(launcher_sha)
            or record.get('model') != poc_models.MODEL_CONFIG[model]
            or request.get('parameters') != poc_models.MODEL_CONFIG[model]['parameters']
            or record.get('worker') != worker or not 0 < worker.get('compute_seconds', 0) <= record['elapsed_seconds']
            or not 0 < record.get('peak_process_family_rss_bytes_sampled', 0) <= MEMORY_LIMIT_BYTES):
        raise ValueError('Model completion, source binding or runtime evidence does not match')
    weights = record['weights']
    if len(weights) != 1 or weights != request.get('weights'):
        raise ValueError('Worker must identify the same single pinned checkpoint')
    weight_path = Path(weights[0]['path']).resolve()
    if (not weight_path.is_relative_to(root / 'cache' / 'models')
            or weight_path.stat().st_size != weights[0]['bytes']
            or sha256_file(weight_path) != weights[0]['sha256']):
        raise ValueError('Retained model checkpoint differs from the worker evidence')
    records = []
    for name in ('brain_mask.nii.gz', 'skull_mask.nii.gz'):
        path = output / name
        if not path.is_file() or path.is_symlink():
            raise ValueError(f'Missing completed worker artifact {name}')
        image = nib.load(path)
        mask = np.asarray(image.dataobj)
        voxels = int(np.count_nonzero(mask))
        if (list(image.shape) != row['data']['shape'] or not np.allclose(image.affine, row['affine'], atol=1e-5)
                or not voxels or not np.isin(mask, (0, 1)).all()):
            raise ValueError(f'Retained worker artifact failed spatial or binary QC: {name}')
        records.append({'path': str(path), 'name': name, 'sha256': sha256_file(path),
                        'shape': list(image.shape), 'affine_ras_mm': image.affine.tolist(),
                        'foreground_voxels': voxels, 'provenance': 'model_inference_unreviewed'})
        del mask
    if sha256_file(receipt) != receipt_sha:
        raise ValueError('Prior worker receipt changed during validation')
    return {**record, 'outputs': records,
            'recovery': {'status': 'validated_after_log_pipe_cleanup_failure',
                         'original_receipt_sha256': receipt_sha, 'original_receipt': str(receipt.relative_to(root)),
                         'worker_completion_sha256': sha256_file(worker_path), 'new_model_runs': 0}}


def estimate_anatomy(root, collection, recover=None):
    from resectionlab import anatomy_models

    model = 'totalsegmentator-skull'
    kinds = {row['kind'] for row in collection.document['layers']
             if row['source_series'] == SERIES and row.get('provenance', {}).get('model') == model}
    if not {'brain', 'skull'} <= kinds:
        if recover:
            row = collection.series(SERIES)
            inference = recover_ct_worker(root, collection, row, recover)
            if not ct_intensities(collection, row):
                raise ValueError('CT-derived reconstruction requires a supported intensity scale')
            for output in inference['outputs']:
                kind = output['name'].removesuffix('_mask.nii.gz')
                if kind == 'skull':
                    output = anatomy_models.restrict_ct_skull(collection, row, output)
                if not output['foreground_voxels']:
                    raise ValueError(f'The retained {kind} estimate has no supported voxels')
                record = {
                    'model': model, 'kind': kind, 'source_series': SERIES, 'source_modality': 'CTA',
                    'source': row['original']['sha256'], 'inference': inference,
                    'path': output['path'], 'output_sha256': output['sha256'],
                    'version': inference['model']['version'], 'inference_spacing_mm': [3, 3, 3],
                    'parameters': {**inference['model']['parameters'], 'positive_ct_bone_gate_hu': 250},
                    'method': 'image_derived_estimate', 'evidence_use': 'context_only', 'clinical_validation': False,
                    'label_ontology': {'value': 1, 'source_label': {'brain': 90, 'skull': 91}[kind],
                                       'meaning': f'Estimated {kind}'},
                    'coverage_limitation': 'Estimated anatomy in the defaced CTA field. Coverage and sensitivity remain unassessed.',
                }
                anatomy_models.import_proposal(collection, SERIES, record)
        else:
            record = anatomy_models.reconstruct(collection, SERIES, model)
            anatomy_models.import_proposal(collection, SERIES, record)
        collection.save('Retain source-bound CTA skull and brain estimates')


def inspect_surfaces(collection):
    from resectionlab.anatomy_geometry import mesh

    summary = []
    for row in collection.document['layers']:
        if row['source_series'] != SERIES or row['kind'] not in {'brain', 'skull'}:
            continue
        mask = collection.mask(row)
        geometry = mesh(collection, row)
        counts = int(np.count_nonzero(mask))
        summary.append({
            'id': row['id'], 'kind': row['kind'], 'voxels': counts,
            'volume_mm3': counts * abs(float(np.linalg.det(np.asarray(row['affine'])[:3, :3]))),
            'vertices': len(geometry['positions']), 'triangles': len(geometry['indices']),
            'max_display_displacement_mm': float(geometry['max_displacement_mm']),
            'surface_policy': str(geometry['policy_version']),
        })
    return summary


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
            estimate_anatomy(root, collection, recover)
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
            result['estimated_surfaces'] = inspect_surfaces(collection)
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
