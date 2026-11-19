#!/usr/bin/env python3
"""Audit the retained sub-13 CT-to-T1 candidates on anatomical edge landmarks."""
from __future__ import annotations

import argparse
from copy import deepcopy
import json
from pathlib import Path
from shutil import copyfile
from tempfile import TemporaryDirectory

import numpy as np
from scipy.ndimage import gaussian_filter1d, generate_binary_structure, label

from build_cta_native_showcases import register
from data_integrated_hosts import UUID, binding, checked_file
from resectionlab.anatomy_package import referenced_files
from resectionlab.data_root import atomic_json, checked_root, safe_path, sha256_file
from resectionlab.patient_anatomy import PatientImagingCollection, world

SOURCE = 'prepared/adaptive-anatomy-migration-20261117-v1/Meningioma-sub-13/patient-anatomy.json'
CASE = 'Meningioma-sub-13-CT-T1-EVIDENCE-v2'
OUTPUT = f'prepared/meningioma-sub13-anatomical-review-v2/{CASE}'
KERNELS = ('ct-bone', 'ct-brain')
# Each row records an anatomical edge, sampled native IJK transect, and
# independently read inclusive windows and gradient signs for sharp CT, soft
# CT, and contrast T1. None marks a soft-tissue edge invisible on sharp CT.
LANDMARKS = (
    ('Sphenoid sinus posterior wall', 'skull base', 1, (212, 175, 75), (175, 183, 1), (175, 183, 1), (177, 184, 1)),
    ('Sphenoid sinus roof', 'skull base', 2, (206, 172, 96), (83, 88, 1), (82, 88, 1), (87, 94, 1)),
    ('Right sphenoid lateral wall', 'skull base', 0, (185, 152, 75), (183, 190, -1), (183, 190, -1), (176, 187, -1)),
    ('Left sphenoid lateral wall', 'skull base', 0, (228, 152, 75), (228, 234, 1), (225, 233, 1), (229, 237, 1)),
    ('Right mastoid roof', 'skull base', 2, (115, 233, 90), (85, 96, 1), (85, 95, 1), (90, 98, 1)),
    ('Left mastoid roof', 'skull base', 2, (310, 224, 90), (84, 91, 1), (84, 91, 1), (89, 96, 1)),
    ('Right lateral inner vault', 'vault', 0, (87, 218, 150), (87, 94, -1), (87, 94, -1), (85, 91, 1)),
    ('Left lateral inner vault', 'vault', 0, (346, 216, 150), (337, 346, 1), (336, 345, 1), (334, 343, -1)),
    ('Anterior frontal outer vault', 'vault', 1, (151, 60, 175), (58, 66, 1), (58, 66, 1), (55, 65, -1)),
    ('Superior frontal inner vault', 'vault', 2, (155, 142, 263), (245, 255, -1), (245, 255, -1), (245, 254, 1)),
    ('Superior parietal outer vault', 'vault', 2, (267, 215, 269), (266, 274, -1), (266, 274, -1), (267, 275, 1)),
    ('Right lateral ventricle medial edge', 'ventricles', 0, (182, 238, 150), None, (183, 192, 1), (182, 191, 1)),
    ('Left lateral ventricle lateral edge', 'ventricles', 0, (270, 250, 150), None, (265, 275, 1), (264, 275, 1)),
    ('Left frontal horn anterior edge', 'ventricles', 1, (226, 122, 150), None, (119, 130, -1), (119, 129, -1)),
)


def sample_edge(signal, axis, center, window):
    low, high, sign = window
    index = list(center)
    index[axis] = slice(low - 3, high + 4)
    profile = signal[tuple(index)]
    derivative = sign * np.gradient(gaussian_filter1d(profile, .68))
    peak = int(derivative[3:-3].argmax()) + 3
    if not np.isfinite(derivative).all() or derivative[peak] <= 0:
        raise ValueError('Anatomical profile has no finite signed contrast edge')
    denominator = derivative[peak - 1] - 2 * derivative[peak] + derivative[peak + 1]
    delta = 0.5 * (derivative[peak - 1] - derivative[peak + 1]) / denominator if denominator < 0 else 0
    result = np.asarray(center, dtype=float)
    result[axis] = low - 3 + peak + np.clip(delta, -.5, .5)
    return result, {
        'window_ijk_inclusive': [low, high], 'derivative_sign': sign, 'ijk': result.tolist(),
        'profile_start_ijk': low - 3, 'profile': profile.tolist(), 'edge_at_search_boundary': peak in (3, len(profile)-4),
    }


def summaries(rows, name):
    subsets = {'all': rows, **{group: [row for row in rows if row['region'] == group]
                             for group in ('skull base', 'vault', 'ventricles')}}
    out = {}
    for group, selection in subsets.items():
        values = [row['errors_mm'][name] for row in selection if name in row['errors_mm']]
        if values:
            values = np.abs(values)
            out[group] = {'count': len(values), 'median_mm': float(np.median(values)),
                          'rms_mm': float(np.sqrt(np.mean(values**2))), 'max_mm': float(values.max())}
    return out


def alignment_review(study):
    ids = (*KERNELS, 't1ce')
    scans = {sid: study.series(sid) for sid in ids}
    for scan in scans.values():
        study.require_native_geometry(scan)
    if study.document['reference_series'] != 't1ce' or not all(
            np.array_equal(scan['affine'], scans['t1ce']['affine']) and
            scan['data']['shape'] == [414, 414, 302] for scan in scans.values()):
        raise ValueError('Expected the locked sub-13 publisher grid')
    if any(study.document['transforms'][sid]['status'] != 'unreviewed' for sid in KERNELS):
        raise ValueError('Expected the frozen unreviewed CT-to-T1 candidates')
    images = {sid: study.array(scan['data']) for sid, scan in scans.items()}
    affine = np.asarray(scans['t1ce']['affine'])
    matrices = {sid: study.registration_matrix(sid, require_geometry=True) for sid in KERNELS}
    observations = []
    for name, region, axis, center, *windows in LANDMARKS:
        locations, rases = {}, {}
        for sid, window in zip(ids, windows):
            if window is not None:
                location, info = sample_edge(images[sid], axis, center, window)
                rases[sid] = world(affine, location)
                locations[sid] = {**info, 'ras_mm': rases[sid].tolist()}
        tangent = affine[:3, axis] / np.linalg.norm(affine[:3, axis])
        errors = {}
        for ct, matrix in matrices.items():
            if ct in rases:
                errors[ct + ' identity'] = float((rases[ct] - rases['t1ce']) @ tangent)
                errors[ct + ' retained'] = float((world(matrix, rases[ct]) - rases['t1ce']) @ tangent)
        if 'ct-bone' in rases and 'ct-brain' in rases:
            errors['sharp-soft CT'] = float((rases['ct-bone'] - rases['ct-brain']) @ tangent)
        observations.append({'structure': name, 'region': region, 'axis': axis, 'transect_ijk': center,
                             'locations': locations, 'errors_mm': errors})
    metrics = {key: summaries(observations, key) for key in (
        'ct-bone identity', 'ct-bone retained', 'ct-brain identity', 'ct-brain retained', 'sharp-soft CT')}
    # The existing automated registration uses 2 mm for disagreement. The
    # observed base discrepancies alone exceed that bound. Even otherwise
    # favorable edge results supply one axis per point and no independent TRE.
    report = {'schema': 'sub13-directional-anatomical-landmarks-v1',
              'source_scans': {k: v['original']['sha256'] for k, v in scans.items()},
              'source_affine': affine.tolist(), 'retained_matrices': {k: v.tolist() for k, v in matrices.items()},
              'resolution_scope': 'Publisher-sampled 0.5 mm grid. Acquisition point spread is unmeasured.',
              'method': 'Directional anatomical edge differences. Gaussian sigma 0.68 sample; signed peak with quadratic sub-sample estimate.',
              'metrics': metrics, 'landmarks': observations,
              'registration_disposition': 'CT and T1 independent; held-out directional review does not qualify fusion',
              'disagreement_screen_mm': 2.0}
    return report


def anatomy_measurements(study):
    categories = {}
    brain = next(row for row in study.document['layers'] if row['kind'] == 'brain')
    envelope = study.mask(brain)
    for row in study.document['layers']:
        if row['kind'] not in {'brain', 'bone', 'tumor', 'vessels'} and 'cerebral cortex' not in row['label']:
            continue
        mask = study.mask(row)
        hits = [sum(int(np.count_nonzero(np.take(mask, index, axis=axis))) for index in (0, -1))
                for axis in range(3)]
        result = {
            'name': row['label'], 'native_grid_spacing_mm': np.linalg.norm(np.asarray(row['affine'])[:3,:3], axis=0).tolist(),
            'positive_voxels': int(np.count_nonzero(mask)), 'positive_voxels_on_paired_faces': hits,
            'role': row['role'], 'source': row['source_series'], 'data_sha256': row['data']['sha256'],
        }
        if row['kind'] == 'bone':
            sharp = study.series(row['source_series'])
            if not np.array_equal(row['affine'], sharp['affine']) or list(mask.shape) != sharp['data']['shape']:
                raise ValueError('Sharp CT and density-derived mask grids differ')
            ct = study.array(sharp['data'])
            components, _ = label(mask, structure=generate_binary_structure(3, 3))
            counts = np.bincount(components.ravel())[1:]
            result.update(voxels_different_from_ct_ge_300=int(np.count_nonzero(mask != (ct >= 300))),
                          connected_components_26=len(counts),
                          largest_component_fraction=float(counts.max() / counts.sum()))
        if 'cerebral cortex' in row['label']:
            transform = np.linalg.inv(np.asarray(brain['affine'])) @ np.asarray(row['affine'])
            points = np.rint(world(transform, np.argwhere(mask))).astype(int)
            on_grid = ((points >= 0) & (points < envelope.shape)).all(axis=1)
            result['fraction_inside_native_brain_estimate'] = (
                int(np.count_nonzero(envelope[tuple(points[on_grid].T)])) / len(points))
        categories[row['id']] = result
    return categories


def build(root):
    source = safe_path(root, SOURCE)
    study = PatientImagingCollection(source.parent)
    if study.document['patient_id'] != 'ZENODO-17486320-meningioma-sub-13':
        raise ValueError('Wrong person for the selected anatomical landmarks')
    destination = safe_path(root, OUTPUT)
    manifest = destination / 'patient-anatomy.json'
    if destination.exists():
        receipt = json.loads((destination / 'receipt.json').read_text())
        if checked_file(root, receipt['manifest']) != manifest or receipt['source_manifest'] != binding(root, source):
            raise ValueError('Existing view differs from previously verified source')
        checked_file(root, receipt['landmark_review'])
        return receipt
    report = alignment_review(study)
    report['anatomy_measurements'] = anatomy_measurements(study)
    document = deepcopy(study.document)
    document.pop('snapshot_id')
    document.update(patient_id=CASE, revision=0, history=[])
    for row in document['series'] + document['layers']:
        row['patient_id'] = CASE
    # Negative source outputs are kept in the parent and numeric review.
    # Expose only the reference anatomy appropriate to a focused viewer.
    document['layers'] = [row for row in document['layers'] if row['kind'] not in {'tumor','vessels'}]
    document['display'] = {'initial_layers': ['bone', 'synthstrip-476c14209ba1']}
    # The parent and the landmark receipt retain both reviewed candidates.
    # Withholding them in this companion also prevents preview-only fusion.
    document['transforms'] = {'t1ce': document['transforms']['t1ce']}
    document['missing'] = ['Additional cross-modal landmark review for registration',
                           'Independent cortical labels and microscopic skull-base validation',
                           'Acquired angiography and qualified vascular annotation',
                           'Nonempty verified tumor segmentation and dural annotation']
    document['data_case'] = {'source_family_id':'ZENODO-17486320-meningioma-sub-13','role':'DEVELOPMENT',
                             'clinical_validation':False,'case_type':'independent_source_inspection'}
    destination.parent.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(dir=destination.parent, prefix='.sub13-review-') as scratch:
        stage = Path(scratch) / CASE
        stage.mkdir()
        companion = PatientImagingCollection(stage, document)
        for name, digest in set(referenced_files(document)):
            out = companion.path(name)
            out.parent.mkdir(parents=True, exist_ok=True)
            copyfile(study.path(name), out)
            if sha256_file(out) != digest:
                raise ValueError(f'Retained source member changed: {name}')
        companion.save('Inspect each modality on its original grid with bounded registration evidence')
        if companion.document['transforms'] != {'t1ce': study.document['transforms']['t1ce']}:
            raise ValueError('Independent-source companion unexpectedly enables a CT transform')
        review = stage / 'landmark-review.json'
        atomic_json(review, report)
        receipt = {'case_id': CASE, 'source_manifest': binding(root, source), 'snapshot_id': companion.identity(),
                   'manifest': {'path': str(manifest.relative_to(root)), 'sha256': sha256_file(stage / manifest.name)},
                   'landmark_review': {'path': str(manifest.with_name(review.name).relative_to(root)),
                                       'sha256': sha256_file(review)}}
        atomic_json(stage / 'receipt.json', receipt)
        checked_root(root, expected_uuid=UUID)
        stage.rename(destination)
    return receipt


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root', type=Path, required=True)
    p.add_argument('--register', type=Path)
    args = p.parse_args()
    root = checked_root(args.root, expected_uuid=UUID, required_bytes=4 * 1024**3)
    receipt = build(root)
    if args.register:
        register(root, args.register, receipt)
    print(safe_path(root, receipt['manifest']['path']))


if __name__ == '__main__':
    main()
