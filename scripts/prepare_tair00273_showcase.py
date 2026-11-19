#!/usr/bin/env python3
"""Prepare the TAIR 00273 showcase and a source-bound partial vessel layer."""
from __future__ import annotations

import argparse
from copy import deepcopy
import json
from pathlib import Path
from shutil import copyfile
from tempfile import TemporaryDirectory

import nibabel as nib
import numpy as np
from PIL import Image, ImageDraw
from scipy import ndimage as ndi

from build_cta_native_showcases import register
from data_integrated_hosts import UUID, binding, checked_file
from resectionlab.anatomy_package import referenced_files
from resectionlab.data_root import atomic_json, checked_root, safe_path, sha256_file
from resectionlab.patient_anatomy import PatientImagingCollection, reconstruction_lock


CASE = 'TAIR-00273-CTA-ANATOMY-v1'
FAMILY = 'TAIR:00273'
SOURCE = 'prepared/tair-v1-cta-native/00273'
ACQUISITION = 'sources/tair-v1-cta-selective/00273/acquisition.json'
MANIFEST_SHA = 'e45dffe90fc95a95e9b2d9489b1c9be593789f0e9cc2de21390e14d798cf9f21'
ACQUISITION_SHA = '3d0d54975c21a58f27386ffcaf687457027a66f00893a2b1667a728eef856126'
DESTINATION = f'prepared/tair00273-cta-showcase-v1/{CASE}'
SERIES = 'tair-cta-display'
VESSEL = 'cta-enhancement-segment'
SCHEMA = 'tair00273-cta-showcase-v1'
ROI_START = (205, 144, 167)
ROI_STOP = (320, 297, 301)
SEED = (260, 207, 234)
STRIDE = (1, 1, 2)
THRESHOLD = 120
STOCK_MESH_FILES = (
    '0c5bb21d41691846426808ec54d7c8e2901d68c6d021df8b1a0b046eeebfcf0a.npz',
    '1495a38ab824ce795c167c4f927ae780a56873b946ff6383e02c7ef8c7d1a8ad.npz',
)
LIMITS = [
    'Acquired defaced CTA. The display retains exact every-second-slice samples in their physical positions.',
    'Brain envelope and dense skull are existing CT-derived estimates. Cortical anatomy and skull-base coverage remain unqualified.',
    'One partial, unclassified segment follows a visible enhancing tube. Its endpoints depend on the threshold.',
    'Vessel identity, complete vascular coverage, functional anatomy, and procedural authority remain unqualified.',
]


def roi(start, stop):
    return tuple(slice(int(a), int(b)) for a, b in zip(start, stop))


def component(values, seed, threshold=THRESHOLD, neighborhood=3):
    labels, _ = ndi.label(values >= threshold, ndi.generate_binary_structure(3, neighborhood))
    label = labels[tuple(seed)]
    if not label:
        raise ValueError('The pinned CTA enhancement seed is below the threshold')
    return labels == label


def load_source(root):
    manifest = checked_file(root, {'path': f'{SOURCE}/patient-anatomy.json', 'sha256': MANIFEST_SHA})
    acquisition_path = checked_file(root, {'path': ACQUISITION, 'sha256': ACQUISITION_SHA})
    acquisition = json.loads(acquisition_path.read_text())
    study = PatientImagingCollection(manifest.parent)
    scan = study.series(SERIES)
    layers = {row['kind']: row for row in study.document['layers']}
    if (study.document['patient_id'] != FAMILY or acquisition['source_person_family'] != FAMILY
            or acquisition['license'] != scan['source_license'] or acquisition['license'] != 'CC-BY-4.0'
            or acquisition['sha256'] != scan['publisher_source']['sha256'] or acquisition['defaced'] is not True
            or acquisition['supplied_segmentations'] or scan['display_sampling_ijk'] != list(STRIDE)
            or scan['data']['shape'] != [512, 512, 377] or set(layers) != {'brain', 'skull'}
            or any(row['source_series'] != SERIES or row['affine'] != scan['affine']
                   or row['data']['shape'] != scan['data']['shape'] or row['role'] != 'context'
                   for row in layers.values())):
        raise ValueError('Expected the frozen TAIR 00273 source and exact cached estimates')
    source = checked_file(root, {'path': f'{SOURCE}/{scan["publisher_source"]["path"]}',
                                 'sha256': acquisition['sha256']})
    return study, scan, layers, nib.load(source), acquisition


def segment(study, scan, layers, original):
    start, stop, stride = map(np.asarray, (ROI_START, ROI_STOP, STRIDE))
    cropped = roi(start, stop)
    seed = np.asarray(SEED) - start
    display_affine = np.asarray(scan['affine'])
    if (original.shape != (512, 512, 754) or original.header.get_xyzt_units()[0] != 'mm'
            or not np.array_equal(original.affine @ np.diag(np.r_[stride, 1]), display_affine)):
        raise ValueError('The display affines no longer match their original sample positions')
    signal = study.array(scan['data'])[cropped]
    source_signal = np.asarray(original.dataobj[roi(start * stride, stop * stride)])
    original_samples = source_signal[::stride[0], ::stride[1], ::stride[2]]
    if not np.array_equal(signal, original_samples):
        raise ValueError('Display scalar values in the reviewed region differ from the original CTA')

    display = component(signal, seed)
    native = component(source_signal, seed * stride)
    brain = study.array(layers['brain']['data'])[cropped] == layers['brain']['label_value']
    skull = study.array(layers['skull']['data'])[cropped] == layers['skull']['label_value']
    native_samples = native[::stride[0], ::stride[1], ::stride[2]]
    if (np.count_nonzero(display) != 146 or np.count_nonzero(native) != 451
            or np.any(display & ~native_samples) or np.any(display & (~brain | skull))):
        raise ValueError('The source enhancement or its relationship to the retained anatomy changed')

    spacing = nib.affines.voxel_sizes(display_affine)
    display_points = np.argwhere(display)
    native_points = np.argwhere(native)
    minimum_face_mm = float(np.min(np.minimum(display_points, stop - start - 1 - display_points) * spacing))
    distance_mm = ndi.distance_transform_edt(~skull, sampling=spacing)
    minimum_bone_mm = float(distance_mm[display].min())
    clearance_mm = min(minimum_face_mm, minimum_bone_mm)
    if clearance_mm < 20.:
        raise ValueError('The retained source component no longer has sufficient clearance for visual review')

    shell = ndi.binary_dilation(display, iterations=4) & ~ndi.binary_dilation(display, iterations=1)
    boundary = display & ~ndi.binary_erosion(display)
    transverse_areas = display.sum(axis=(0, 2)) * spacing[0] * spacing[2]
    equivalent_widths = 2 * np.sqrt(transverse_areas[transverse_areas > 0] / np.pi)
    native_spacing = nib.affines.voxel_sizes(original.affine)
    native_start = start * stride
    native_stop = stop * stride
    audit = {
        'recipe': {'roi_start_display_ijk_inclusive': list(ROI_START), 'roi_stop_display_ijk_exclusive': list(ROI_STOP),
                   'seed_display_ijk': list(SEED), 'seed_source_ijk': (np.asarray(SEED) * stride).tolist(),
                   'seed_ras_mm': nib.affines.apply_affine(display_affine, SEED).tolist(),
                   'threshold_scaled_ct_units': THRESHOLD, 'component_connectivity': 26,
                   'interpolation': 'none', 'vessel_identity': 'unclassified', 'model_runs': 0},
        'original_cta_shape': list(original.shape), 'original_cta_affine_ras_mm': original.affine.tolist(),
        'sampled_cta_shape': scan['data']['shape'], 'sampled_cta_affine_ras_mm': scan['affine'],
        'exact_original_roi_samples_checked': int(signal.size),
        'original_component': {
            'voxel_count': int(native_points.shape[0]),
            'volume_mm3': float(native_points.shape[0] * np.prod(native_spacing)),
            'bounds_source_ijk_inclusive': [(native_points.min(0) + native_start).tolist(),
                                            (native_points.max(0) + native_start).tolist()],
            'minimum_roi_face_mm': float(np.min(np.minimum(native_points, native_stop - native_start - 1
                                                           - native_points) * native_spacing)),
            'even_slice_samples_outside_display_component': int(np.count_nonzero(native_samples & ~display)),
        },
        'display_component': {
            'voxel_count': int(display_points.shape[0]), 'volume_mm3': float(display_points.shape[0] * np.prod(spacing)),
            'bounds_display_ijk_inclusive': [(display_points.min(0) + start).tolist(),
                                             (display_points.max(0) + start).tolist()],
            'minimum_roi_face_mm': minimum_face_mm, 'minimum_retained_bone_mm_inside_roi': minimum_bone_mm,
            'minimum_retained_bone_mm_conservative': clearance_mm,
            'signal_min_median_max': np.percentile(signal[display], (0, 50, 100)).tolist(),
            'shell_signal_median': float(np.median(signal[shell])),
            'source_boundary_voxels': int(np.count_nonzero(boundary)),
            'source_boundary_min_signal': float(signal[boundary].min()),
            'source_boundary_voxels_below_threshold': int(np.count_nonzero(signal[boundary] < THRESHOLD)),
            'display_voxels_by_connectivity': {str(k): int(np.count_nonzero(component(signal, seed, neighborhood=n)))
                                              for k, n in ((6, 1), (18, 2), (26, 3))},
            'threshold_voxels': {str(t): int(np.count_nonzero(component(signal, seed, threshold=t)))
                                for t in (120, 125, 130, 135)},
            'native_axis_j_segment_span_mm': float((np.ptp(display_points[:, 1]) + 1) * spacing[1]),
            'j_normal_area_equivalent_width_mm_min_median_max': np.percentile(equivalent_widths, (0, 50, 100)).tolist(),
            'width_scope': 'Area-equivalent width in native J-normal planes. Obliquity and threshold affect this measure.',
        },
    }
    return signal, display, native, audit


def source_overlay(signal, mask, spacing, destination):
    canvas = Image.new('RGB', (956, 572), '#11171d')
    draw = ImageDraw.Draw(canvas)
    draw.text((14, 12), 'TAIR 00273  |  Acquired CTA  |  Partial unclassified enhancement segment', fill='white')
    for axis, label, center, half_width in ((0, 'Sagittal maximum I 258–272', 265, 7),
                                            (1, 'Coronal slice J 207', 207, 0),
                                            (2, 'Axial maximum K 232–244', 238, 6)):
        local_center = center - ROI_START[axis]
        indices = np.arange(local_center - half_width, local_center + half_width + 1)
        raw = np.take(signal, indices, axis=axis).max(axis=axis).T[::-1]
        selected = np.take(mask, indices, axis=axis).any(axis=axis).T[::-1]
        gray = np.clip((raw - 20.) * (255. / 210.), 0, 255).astype(np.uint8)
        unmarked = np.repeat(gray[..., None], 3, axis=-1)
        marked = unmarked.copy()
        marked[selected & ~ndi.binary_erosion(selected)] = (87, 245, 139)
        sx, sy = np.delete(spacing, axis)
        scale = min(302. / (gray.shape[1] * sx), 242. / (gray.shape[0] * sy))
        size = (round(gray.shape[1] * sx * scale), round(gray.shape[0] * sy * scale))
        for row, (title, pixels) in enumerate(((label, unmarked), ('Source with component outline', marked))):
            x, y = 14 + axis * 315, 39 + row * 264
            draw.text((x, y), title, fill='white')
            canvas.paste(Image.fromarray(pixels).resize(size, Image.Resampling.BILINEAR), (x, y + 17))
    canvas.save(destination, format='PNG')


def write_original_membership(study, mask, original):
    points = np.argwhere(mask)
    start = points.min(0) - 1
    stop = points.max(0) + 2
    source_start = start + np.asarray(ROI_START) * STRIDE
    native_affine = original.affine.copy()
    native_affine[:3, 3] = nib.affines.apply_affine(original.affine, source_start)
    membership = mask[roi(start, stop)].astype(np.uint8)
    image = nib.Nifti1Image(membership, native_affine)
    image.header.set_xyzt_units('mm')
    image.set_qform(native_affine, code=1)
    image.set_sform(native_affine, code=1)
    name = 'source-native-enhancement-membership.nii.gz'
    path = study.path(name)
    nib.save(image, path)
    restored = nib.load(path)
    if (restored.header.get_xyzt_units()[0] != 'mm'
            or not np.allclose(restored.affine, native_affine, atol=1e-5, rtol=0.)
            or not np.array_equal(np.asarray(restored.dataobj), membership)):
        raise ValueError('Saved source-grid enhancement membership changed')
    return {'path': name, 'sha256': sha256_file(path), 'source_crop_start_ijk': source_start.tolist()}


def reuse_source(base, destination):
    document = deepcopy(base.document)
    document.pop('snapshot_id')
    document.update(patient_id=CASE, revision=0, history=[])
    for row in document['series'] + document['layers']:
        row['patient_id'] = CASE
    study = PatientImagingCollection(destination, document=document)
    for name, digest in set(referenced_files(base.document)):
        original = base.path(name)
        target = study.path(name)
        target.parent.mkdir(parents=True, exist_ok=True)
        copyfile(original, target)
        if sha256_file(target) != digest:
            raise ValueError(f'A frozen study member changed during reuse: {name}')
    for name in STOCK_MESH_FILES:
        cache = base.path(f'meshes/{name}')
        if cache.is_file():
            target = study.path(f'meshes/{name}')
            target.parent.mkdir(exist_ok=True)
            copyfile(cache, target)
    return study


def build(root):
    base, scan, layers, original, acquisition = load_source(root)
    destination = safe_path(root, DESTINATION)
    manifest = destination / 'patient-anatomy.json'
    if destination.exists():
        result = json.loads((destination / 'receipt.json').read_text())
        reopened = PatientImagingCollection(destination)
        if (result['schema'] != SCHEMA or result['snapshot_id'] != reopened.identity()
                or checked_file(root, result['manifest']) != manifest):
            raise ValueError('The prepared showcase differs from its sealed receipt')
        for name, digest in set(referenced_files(reopened.document)):
            if sha256_file(reopened.path(name)) != digest:
                raise ValueError(f'The saved showcase member changed: {name}')
        return result

    signal, display, native, audit = segment(base, scan, layers, original)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(prefix='.tair00273-showcase-', dir=destination.parent) as temporary:
        staging = Path(temporary) / CASE
        study = reuse_source(base, staging)
        native_membership = write_original_membership(study, native, original)
        overlay_name = 'source-segment-overlays.png'
        overlay = study.path(overlay_name)
        source_overlay(signal, display, np.asarray(scan['grid_spacing_mm']), overlay)
        overlay_sha = sha256_file(overlay)
        mask = np.zeros(scan['data']['shape'], dtype=np.uint8)
        mask[roi(ROI_START, ROI_STOP)] = display
        vessel = study.add_layer(mask, scan['affine'], layer_id=VESSEL, source_series=SERIES, kind='vessels',
                                 role='context', label='Partial CTA vessel estimate · unclassified', color='#ea876f',
                                 provenance={'kind': 'image_derived_estimate', 'source': scan['publisher_source'],
                                             'basis': 'Connected thresholded enhancement in the acquired CTA',
                                             'recipe': audit['recipe'], 'evidence_use': 'context_only',
                                             'native_threshold_membership': native_membership,
                                             'source_overlay': {'path': overlay_name, 'sha256': overlay_sha}})
        vessel.update(units_status='mm', coverage=LIMITS[2])
        study.document['display'] = {
            'initial_layers': [layers[k]['id'] for k in ('brain', 'skull')] + [VESSEL],
            'cursor_ras_mm': audit['recipe']['seed_ras_mm'],
        }
        study.document['data_case'] = {
            'source_family_id': FAMILY, 'role': 'DEVELOPMENT', 'case_type': 'display_companion',
            'clinical_validation': False, 'limitations': LIMITS,
        }
        study.save('Prepare source-bound TAIR brain, skull and partial CTA vessel display')
        reopened = PatientImagingCollection(staging)
        reopened_vessel = reopened.layer(VESSEL)
        saved_mask = reopened.array(reopened_vessel['data'])
        restored_scan = {**scan, 'patient_id': CASE}
        if (reopened.series(SERIES) != restored_scan
                or not np.array_equal(saved_mask[roi(ROI_START, ROI_STOP)], display)
                or np.count_nonzero(saved_mask) != audit['display_component']['voxel_count']
                or reopened_vessel['role'] != 'context' or reopened_vessel['affine'] != scan['affine']):
            raise ValueError('The reopened vessel membership, role or CTA grid changed')
        for kind, row in layers.items():
            saved = reopened.layer(row['id'])
            if saved['data'] != row['data'] or saved['affine'] != row['affine']:
                raise ValueError(f'The reopened native {kind} estimate changed')
        result = {
            'schema': SCHEMA, 'case_id': CASE, 'source_person_id': FAMILY, 'role': 'DEVELOPMENT',
            'new_source_people': 0, 'new_synthetic_scenarios': 0, 'limits': LIMITS,
            'source': {'frozen_study': {'path': f'{SOURCE}/patient-anatomy.json', 'sha256': MANIFEST_SHA},
                       'acquisition': {'path': ACQUISITION, 'sha256': ACQUISITION_SHA},
                       'release': acquisition['publisher_release'], 'license': acquisition['license'],
                       'original_cta_sha256': scan['publisher_source']['sha256'],
                       'display_cta_sha256': scan['original']['sha256'],
                       'reused_brain_skull_arrays': {kind: row['data']['sha256'] for kind, row in layers.items()}},
            'manifest': {'path': str(manifest.relative_to(root)), 'sha256': sha256_file(staging / manifest.name)},
            'snapshot_id': reopened.identity(), 'vessel_layer': VESSEL, 'skull_display_proof': None,
            'source_overlay': {'path': str((destination / overlay_name).relative_to(root)), 'sha256': overlay_sha},
            'original_membership': {**native_membership,
                                    'path': str((destination / native_membership['path']).relative_to(root))},
            'segment': audit,
        }
        atomic_json(staging / 'receipt.json', result)
        checked_root(root, expected_uuid=UUID)
        staging.rename(destination)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path)
    parser.add_argument('--register', type=Path, help='Optional local Cases directory for the native study alias')
    args = parser.parse_args()
    root = checked_root(args.root, expected_uuid=UUID, required_bytes=1024**3)
    with reconstruction_lock():
        report = build(root)
        if args.register is not None:
            register(root, args.register, report)
        print(report['manifest']['path'], flush=True)


if __name__ == '__main__':
    main()
