#!/usr/bin/env python3
"""Capture and measure the four vascular hero cases at fixed cameras.

Every phase of the vascular rendering work reruns this harness so that
captures, frame timing, geometry and reference agreement stay comparable.
Evidence is written to the verified external data root, never to the checkout.
Source scans and labels are only read. Like the desktop app, the wall reader
keeps its derived contour cache in each study's meshes folder.
"""
from __future__ import annotations

import argparse
import json
import os
import re
from pathlib import Path
import shutil
import signal
import subprocess

import nibabel as nib
import numpy as np
from PIL import Image
from scipy import ndimage as ndi

from resectionlab.anatomy_bridge import open_anatomy_manifest
from resectionlab.anatomy_geometry import mesh, vascular_cleanup_source
from resectionlab.data_root import atomic_json, checked_root, safe_path, sha256_file
from resectionlab.patient_anatomy import affine, world
from resectionlab.vascular_parity import (
    betti, betti_error, cl_dice, dice, floating_components, image_sharpness, source_deviation, surface_quality,
)

HDD_ROOT = Path('/Volumes/shafrir/old coding projects/RessectionLabData')
HDD_UUID = '9CC3623C-E38B-3EEE-B6BB-B9A06C193700'
EVIDENCE = Path('qc/medivis-parity-20261125')
TOPBRAIN = 'sources/topbrain/selective/{case}/TopBrain_Data_Release_Batches1n2nTA36_081726/labelsTr_topbrain_v1_ct/topcow_ct_{case}.nii.gz'
# TopBrain v1 CT values: 1-34 arteries, 35-40 veins and sinuses. Roots are the
# basilar (1), internal carotids (4, 6), vertebrals (23, 24), the vein of Galen
# (35) and the straight and superior sagittal sinuses (36, 40).
ROOT_LABELS = (1, 4, 6, 23, 24, 35, 36, 40)

HERO = {
    'CTA-A': {
        'title': 'TopCoW-014 CTA with skull', 'held_out': False,
        'manifest': 'prepared/cta-native-showcases-v1/TopCoW-014-CTA-ORGANIC-01-v1-NATIVE-SHOWCASE/patient-anatomy.json',
        'series': 'ct', 'vessels': ['supplied_arterial_annotation_union', 'supplied_venous_sinus_annotation_union'],
        'anchor': [4.07, 210.81, -67.40], 'reference': TOPBRAIN.format(case='014'),
    },
    'CTA-B': {
        'title': 'TopCoW-018 CTA, held out', 'held_out': True,
        'manifest': 'prepared/annotated-cta-vascular-comparison-v1/TopCoW-018-CTA-VESSEL-COMPARISON-v1/patient-anatomy.json',
        'series': 'cta', 'vessels': ['publisher-arteries', 'publisher-veins'],
        'anchor': [-5.07, 208.28, 78.50], 'reference': TOPBRAIN.format(case='018'),
        'model_probability': 'cache/poc-model-results/TopCoW-018-annotated-cta-benchmark-v1/vesselfm-base',
    },
    'TOF-A': {
        'title': 'EDEN PTNT_4509 TOF', 'held_out': False,
        'manifest': 'qc/eden2020-glioma-5714812/vascular-reconstruction-20261123-v1/studies/PTNT_4509/patient-anatomy.json',
        'series': 'tof', 'vessels': ['vesselfm-cleanup'], 'reference': None, 'cohort_view': True,
        'anchor': [3.2779, 5.3626, -13.3764],
    },
    'TOF-B': {
        'title': 'EDEN PTNT_9937 TOF, held out', 'held_out': True,
        'manifest': 'prepared/Eden 2020 ResectionLab Projects (restricted)/PTNT_9937/vascular-review.anatomy.json',
        'series': 'tof', 'vessels': ['vesselfm-cleanup'], 'reference': None, 'cohort_view': True,
        'anchor': [1.3713, 6.0276, -22.1647],
    },
}
# View direction is from the camera toward the anchor, in patient RAS.
PRESETS = {
    'anterior': {'from': [0, 1, 0.12], 'up': [0, 0, 1], 'distance': 1.75},
    'left-lateral': {'from': [-1, 0.05, 0.1], 'up': [0, 0, 1], 'distance': 1.75},
    'superior': {'from': [0, -0.02, 1], 'up': [0, 1, 0], 'distance': 1.75},
    'oblique': {'from': [-0.62, 0.62, 0.48], 'up': [0, 0, 1], 'distance': 1.75},
    'skull-base': {'from': [0, -0.25, 1], 'up': [0, 1, 0], 'distance': 1.05, 'clip': ('z', 22.0, True)},
    'axial-clip': {'from': [-0.35, 0.75, 0.56], 'up': [0, 0, 1], 'distance': 1.55, 'clip': ('z', 6.0, True)},
}
INTERACTION_PRESET = 'anterior'


def evidence(root, label):
    if not re.fullmatch(r'phase\d-[a-z0-9-]{1,48}', label):
        raise ValueError('Evidence labels look like phase0-baseline')
    return safe_path(root, EVIDENCE / label)


def scan_geometry(collection, case):
    scan = collection.series(case['series'])
    frame = affine(scan['affine'])
    shape = np.asarray(scan['data']['shape'])
    corners = world(frame, np.array([[i, j, k] for i in (0, shape[0] - 1) for j in (0, shape[1] - 1) for k in (0, shape[2] - 1)]))
    return scan, frame, float(np.ptp(corners, axis=0).max())


def vessel_mask(collection, case):
    mask = None
    for layer_id in case['vessels']:
        layer = collection.layer(layer_id)
        current = collection.mask(layer)
        mask = current if mask is None else mask | current
    return mask


def anchor(case):
    """Frozen circle-of-Willis level. Cameras must not move when a reconstruction changes."""
    return np.asarray(case['anchor'], float)


def displayed_surface(collection, layer_id):
    """The wall the desktop opens with. Restricted TOF companions show the unconditioned score contour."""
    layer = collection.layer(layer_id)
    companion = (collection.document.get('vascular_review') or {}).get('schema') == 'eden2020-restricted-vascular-companion-v1'
    return layer, mesh(collection, layer, display_source=vascular_cleanup_source(collection, layer), conditioned=not companion)


def recipes(collection, case, default_view, default_layers):
    scan, _, extent = scan_geometry(collection, case)
    target = anchor(case)
    settings = {key: value for key, value in default_view['settings'].items() if key not in {'camera', 'clipping'}}
    rows = []
    for name, preset in PRESETS.items():
        direction = np.asarray(preset['from'], float)
        direction /= np.linalg.norm(direction)
        view = {**settings, 'camera': {
            'position': (target + direction * preset['distance'] * extent).round(4).tolist(),
            'target': target.round(4).tolist(), 'up': preset['up'],
        }}
        row = {'name': name, 'seriesId': scan['id'], 'layers': default_layers, 'cursor': target.round(4).tolist(), 'settings': view}
        if 'clip' in preset:
            axis, above, flipped = preset['clip']
            clipping = {'axis': axis, 'value': round(float(target['xyz'.index(axis)] + above), 4), 'enabled': True, 'flipped': flipped}
            view['clipping'] = clipping
            row['patientCutaway'] = {'clipping': clipping}
        rows.append(row)
    timed = next(row for row in rows if row['name'] == INTERACTION_PRESET)
    rows.append({**timed, 'name': 'interaction', 'interaction': {'frames': 180}})
    rows.append({**timed, 'name': 'orbit', 'rotationFrames': 48})
    return rows


def run_capture(desktop, manifest, views, output, log, *, width, height):
    runtime = desktop / 'node_modules' / 'electron'
    electron = runtime / 'dist' / (runtime / 'path.txt').read_text().strip()
    command = [str(electron), 'electron/capture-vascular-surfaces.cjs', '--anatomy', str(manifest), '--views', str(views),
               '--output', str(output), '--width', str(width), '--height', str(height)]
    with log.open('w') as stream, subprocess.Popen(
            command, cwd=desktop, stdout=stream, stderr=subprocess.STDOUT, start_new_session=True) as process:
        try:
            code = process.wait(timeout=1500)
        except (subprocess.TimeoutExpired, KeyboardInterrupt):
            os.killpg(process.pid, signal.SIGTERM)
            try:
                process.wait(timeout=20)
            except subprocess.TimeoutExpired:
                os.killpg(process.pid, signal.SIGKILL)
                process.wait()
            raise
    report = json.loads((output / 'report.json').read_text()) if (output / 'report.json').is_file() else {}
    if code or not report.get('passed') or report.get('windowEverVisible'):
        raise RuntimeError(f'Hidden capture failed. See {log}')
    return report


def pane_image(view):
    pane = view['pane']
    box = (pane['x'], pane['y'], pane['x'] + pane['width'], pane['y'] + pane['height'])
    image = Image.open(view['filename']).convert('RGB').crop(box)
    # macOS sometimes returns a Retina backing bitmap for a hidden window.
    # Measure every frame at the logical display scale.
    scale = view['captureScale']
    if scale != 1:
        image = image.resize((round(image.width / scale), round(image.height / scale)), Image.LANCZOS)
    return image


def capture(root, name, case, label, *, width, height):
    """Capture today's default appearance at every fixed preset and time one continuous orbit."""
    desktop = Path(__file__).resolve().parents[1] / 'desktop'
    manifest = safe_path(root, case['manifest'])
    collection = open_anatomy_manifest(manifest)
    scan = collection.series(case['series'])
    output = evidence(root, label) / 'captures' / name
    if output.exists():
        raise FileExistsError(f'{output} already holds a capture. Choose a new label or remove it deliberately.')
    output.mkdir(parents=True)
    initial = collection.document['display']['initial_layers']
    # A slow first contour read would outlast the bounded interactive request.
    for layer_id in case['vessels']:
        displayed_surface(collection, layer_id)
    if case.get('cohort_view'):
        # Restricted TOF companions open through the cohort review view, as in the earlier EDEN captures.
        default_view, initial = {'settings': {}}, case['vessels']
    else:
        default_recipe = [{'name': 'default', 'seriesId': scan['id'], 'layers': initial, 'useDefaultState': True}]
        atomic_json(output / 'default-view.json', default_recipe)
        default = run_capture(desktop, manifest, output / 'default-view.json', output / 'default', output / 'default.log',
                              width=width, height=height)
        default_view = default['views'][0]['view']
    rows = recipes(collection, case, default_view, initial)
    atomic_json(output / 'views.json', rows)
    report = run_capture(desktop, manifest, output / 'views.json', output / 'presets', output / 'presets.log',
                         width=width, height=height)
    metrics = {'case': name, 'title': case['title'], 'held_out': case['held_out'], 'label': label,
               'manifest_sha256': sha256_file(manifest), 'snapshot_id': report['snapshotId'],
               'window': report['windowSize'], 'device_scale': report['deviceScale'], 'presets': {}}
    for view in report['views']:
        if view['name'] in PRESETS:
            image = pane_image(view)
            image.save(output / f"{view['name']}.png")
            quality = view['frame'].get('scanQuality') or {}
            settled = [row['gpu']['milliseconds'] for row in view['settledSamples'] if row.get('gpu', {}).get('completed')]
            metrics['presets'][view['name']] = {
                'image': f"{view['name']}.png", 'pane': [image.width, image.height], 'camera': view['frame']['cameraState'],
                'volume': view['frame'].get('volume'), 'settled_samples': quality.get('samples'),
                'settled_resolution': [quality.get('width'), quality.get('height')],
                'settled_sample_gpu_ms': settled,
                # The app labels an active cut along the bottom of the pane.
                'sharpness': image_sharpness(np.asarray(image)[:-56]),
            }
        elif view['name'] == 'interaction':
            metrics['interaction'] = {key: value for key, value in view['interaction'].items() if key != 'samples'}
        elif view['name'] == 'orbit':
            response = [row['toFencePollMs'] for row in view['rotationResponse']['samples']]
            metrics['orbit'] = {'frames': view['rotationFrames'], 'input_to_fence_poll_ms_p50_p95': np.percentile(response, [50, 95]).tolist()}
            frames = output / 'presets' / 'orbit-rotation'
            movie = output / 'orbit.mp4'
            pane, scale = view['pane'], view['captureScale']
            crop = (f"crop={pane['width'] // 2 * 2}:{pane['height'] // 2 * 2}:{pane['x']}:{pane['y']},"
                    f"scale=trunc(iw/{scale}/2)*2:-2")
            subprocess.run(['ffmpeg', '-v', 'error', '-y', '-framerate', '12', '-i', str(frames / '%04d.png'), '-vf', crop,
                            '-pix_fmt', 'yuv420p', '-crf', '18', str(movie)], check=True)
            # The frames exist only to encode the film. exFAT sidecar files
            # disappear with their owners, so tolerate entries that vanish.
            shutil.rmtree(frames, ignore_errors=True)
            if frames.exists():
                shutil.rmtree(frames)
            metrics['orbit']['movie'] = movie.name
    atomic_json(output / 'metrics.json', metrics)
    return metrics


def reference_labels(root, case, shape, frame):
    if not case.get('reference'):
        return None
    image = nib.load(safe_path(root, case['reference']))
    if image.shape != tuple(shape) or not np.allclose(image.affine, frame, atol=1e-3):
        raise ValueError('Reference annotation and study grids differ')
    return np.asarray(image.dataobj)


def annotated_region(reference, spacing, margin_mm=5.0):
    """Bounding region of the annotation, padded. Unlabelled anatomy outside it is not scored."""
    found = np.argwhere(reference)
    pad = np.ceil(margin_mm / np.asarray(spacing)).astype(int)
    lower = np.maximum(found.min(axis=0) - pad, 0)
    upper = np.minimum(found.max(axis=0) + pad + 1, reference.shape)
    region = np.zeros(reference.shape, bool)
    region[tuple(slice(a, b) for a, b in zip(lower, upper))] = True
    return region


def agreement(predicted, reference, spacing):
    """Overlap inside the annotated region; topology on whole components that enter it."""
    region = annotated_region(reference, spacing)
    inside = predicted & region
    labels, _ = ndi.label(predicted, np.ones((3, 3, 3), bool))
    entering = np.isin(labels, np.setdiff1d(np.unique(labels[region]), [0]))
    return {
        'scored_region': 'annotation bounding box padded by 5 mm',
        'dice': dice(inside, reference), 'cl_dice': cl_dice(inside, reference),
        'betti': {**betti_error(entering, reference), 'scope': 'whole predicted components that enter the scored region'},
        'whole_grid_dice': dice(predicted, reference),
    }


def geometry(root, name, case, label):
    """Measure the displayed vessel geometry, and reference agreement where an annotation exists."""
    collection = open_anatomy_manifest(safe_path(root, case['manifest']))
    scan, frame, _ = scan_geometry(collection, case)
    spacing = np.linalg.norm(frame[:3, :3], axis=0)
    signal = collection.array(scan['data'])
    mask = vessel_mask(collection, case)
    labels = reference_labels(root, case, mask.shape, frame)
    roots = np.isin(labels, ROOT_LABELS) if labels is not None else None
    result = {'case': name, 'label': label, 'spacing_mm': spacing.tolist(), 'displayed_layers': case['vessels'],
              'displayed_voxels': int(mask.sum()),
              'floating': {**floating_components(mask, roots, spacing),
                           'roots': 'named ICA, vertebral, basilar and sinus labels' if roots is not None else 'largest component'},
              'betti': betti(mask), 'surfaces': {}}
    for layer_id in case['vessels']:
        layer, surface = displayed_surface(collection, layer_id)
        points, faces = np.asarray(surface['positions'], float), np.asarray(surface['indices']).reshape(-1, 3)
        physical = world(affine(layer['affine']), points)
        a, b, c = physical[faces].transpose(1, 0, 2)
        face_normals = np.cross(b - a, c - a)
        # A left-handed scan affine reverses the winding. Enclosed volume fixes which side is outside.
        if np.einsum('ij,ij->', a, np.cross(b, c)) < 0:
            face_normals = -face_normals
        vertex_normals = np.zeros_like(physical)
        for corner in range(3):
            np.add.at(vertex_normals, faces[:, corner], face_normals)
        vertex_normals /= np.maximum(np.linalg.norm(vertex_normals, axis=1), 1e-20)[:, None]
        result['surfaces'][layer_id] = {
            'surface_field': str(surface.get('surface_field', 'binary-mask')), 'vertices': int(len(points)),
            **surface_quality(physical, faces),
            'source_deviation': source_deviation(physical, vertex_normals, signal, frame),
        }
    if labels is not None:
        supplied = bool((mask == (labels > 0)).all())
        result['reference'] = {'file': case['reference'], 'displayed': {
            **agreement(mask, labels > 0, spacing),
            'independent': not supplied,
            **({'note': 'The displayed layers are the annotation itself, so this row is self-agreement'} if supplied else {}),
        }}
        unrooted = floating_components(labels > 0, roots, spacing)
        result['reference']['annotation_floating'] = unrooted
        if case.get('model_probability'):
            runs = sorted(safe_path(root, case['model_probability']).glob('*/vessels_probability.nii.gz'))
            for run in runs:
                image = nib.load(run)
                if not np.allclose(image.affine, frame, atol=1e-3):
                    raise ValueError('Model output and study grids differ')
                probability = np.asarray(image.dataobj, np.float32)
                result['reference'][f'vesselfm-{run.parent.name[:8]}'] = {
                    'file': str(run.relative_to(root)), 'threshold': 0.5, **agreement(probability >= 0.5, labels > 0, spacing),
                    'floating': floating_components(probability >= 0.5, roots, spacing),
                }
    else:
        result['reference'] = {'status': 'unavailable', 'reason': 'No independent vessel annotation exists for this study'}
    output = evidence(root, label) / 'geometry'
    output.mkdir(parents=True, exist_ok=True)
    atomic_json(output / f'{name}.json', result)
    return result


def sheet(root, label, references, compare=None):
    """One row per preset: reference frame, each compared label, then this label."""
    labels = [*(compare or []), label]
    output = evidence(root, label) / 'sheets'
    output.mkdir(parents=True, exist_ok=True)
    cell = (560, 420)
    for name in HERO:
        rows = []
        for preset in PRESETS:
            folders = [evidence(root, item) / 'captures' / name for item in labels]
            # A capture without its metrics stopped part-way and is not evidence.
            if not all((folder / 'metrics.json').is_file() for folder in folders):
                break
            images = [references.get(preset)] + [folder / f'{preset}.png' for folder in folders]
            rows.append([Image.open(path).convert('RGB') if path and Path(path).is_file() else None for path in images])
        if not rows:
            continue
        canvas = Image.new('RGB', (cell[0] * len(rows[0]), cell[1] * len(rows)), (8, 8, 8))
        for y, row in enumerate(rows):
            for x, image in enumerate(row):
                if image is None:
                    continue
                image.thumbnail(cell, Image.LANCZOS)
                canvas.paste(image, (x * cell[0] + (cell[0] - image.width) // 2, y * cell[1] + (cell[1] - image.height) // 2))
        canvas.save(output / f'{name}.jpg', quality=90)
        atomic_json(output / f'{name}.json', {'columns': ['reference', *labels], 'rows': list(PRESETS)[:len(rows)]})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=['capture', 'geometry', 'sheet'])
    parser.add_argument('--label', required=True, help='Evidence folder under the parity root, such as phase0-baseline')
    parser.add_argument('--case', action='append', choices=list(HERO))
    parser.add_argument('--compare', action='append', help='Earlier labels to place before this one on a sheet')
    parser.add_argument('--references', type=Path, help='Folder of reference frames named after each preset')
    parser.add_argument('--width', type=int, default=1440)
    parser.add_argument('--height', type=int, default=960)
    args = parser.parse_args()
    root = checked_root(HDD_ROOT, expected_uuid=HDD_UUID)
    cases = args.case or list(HERO)
    if args.command == 'sheet':
        references = {preset: args.references / f'{preset}.png' for preset in PRESETS} if args.references else {}
        sheet(root, args.label, references, args.compare)
        return 0
    for name in cases:
        if args.command == 'capture':
            metrics = capture(root, name, HERO[name], args.label, width=args.width, height=args.height)
            timing = metrics.get('interaction', {})
            print(name, 'frame cost p95', timing.get('frameCostMs', {}).get('p95'), 'ms; settle',
                  (timing.get('settle') or {}).get('milliseconds'), 'ms', flush=True)
        else:
            result = geometry(root, name, HERO[name], args.label)
            print(name, 'floating', result['floating']['floating'], 'betti', result['betti'], flush=True)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
