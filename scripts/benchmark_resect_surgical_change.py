#!/usr/bin/env python3
"""Reproduce the two saved RESECT comparisons and verify Case3 cavity transport."""
from __future__ import annotations

import argparse
from dataclasses import replace
from hashlib import sha256
from io import BytesIO
import json
import math
from pathlib import Path
from statistics import median
from time import monotonic
from zipfile import ZipFile

from acquire_resect_cavity import ROOT, require_manifest, verify_source_file
from mechanics_patient_comparison import BaselineField, sample_baselines

import nibabel as nib
import numpy as np

from resectionlab.data_root import checked_root
from resectionlab.imaging import file_sha256, inspect_nifti, load_case, _maximum_corner_displacement
from resectionlab.workspace_bundle import OriginalFile, extension_layout, load_workspace, save_workspace, view_state
from resectionlab.workspace_imaging import DisplaySource


ARTIFACTS = ROOT / 'artifacts'
CASE3_REGISTRATION = ARTIFACTS / 'poc-resect-case3-image-registration-v1'
CASE4 = ARTIFACTS / 'resect-case4-sparse-update-v1'
CASE3_WORKSPACE = Path('cases/real/RESECT-Case3-source-stages-v3.ressectionlab')
CASE3_WORKSPACE_SHA = 'd88a16b6a41c08cedbb4c077a3ae26440614d72765381e5cc028425009d44309'
CASE3_CAVITY = Path('sources/RESECT-SEG/v1-20240215/Case3/Case3-US-during-resection.nii.gz')
CAVITY_LABEL = 'Observed visible cavity'


def read_verified(path, expected_sha):
    payload = path.read_bytes()
    if sha256(payload).hexdigest() != expected_sha:
        raise ValueError(f'Source checksum differs for {path.name}')
    return payload


def metrics(errors):
    if not errors or any(not math.isfinite(value) or value < 0 for value in errors):
        raise ValueError('Expected nonnegative finite errors for every landmark')
    return {'count': len(errors), 'rms_mm': math.sqrt(math.fsum(e * e for e in errors) / len(errors)),
            'median_mm': median(errors), 'maximum_mm': max(errors)}


def case3_registration(root):
    audit_path = ARTIFACTS / 'resect-case3-evaluation-independent-v1/independent-score-qa.json'
    audit_sha = 'e59808143c01bfea2b492ec2f4153dee67faea9044771d93662199b46040d661'
    audit = json.loads(read_verified(audit_path, audit_sha))
    summary_path = CASE3_REGISTRATION / 'scalar-results.json'
    summary_sha = audit['pins']['outputs/poc-resect-case3-image-registration-evaluation-v1/summary-no-coordinates.json']
    summary = json.loads(read_verified(summary_path, summary_sha))
    declaration = require_manifest()
    role = declaration['role']
    if audit['status'] != 'pass' or summary['case_id'] != declaration['patient_group'] or summary['role'] != role:
        raise ValueError('Expected the independently reviewed Case3 TRAIN record')
    sources = {}
    for original, digest in audit['compressed_source_pins'].items():
        relative = Path('sources/RESECT/v1/Case3') / Path(original).name
        if file_sha256(root / relative) != digest:
            raise ValueError(f'Case3 registration source has changed: {relative.name}')
        sources[str(relative)] = digest
    rows = summary['rows']
    if (len(rows) != 17 or [row['point_index'] for row in rows] != list(range(1, 18))
            or not all(row['retained_in_all_statistics'] for row in rows)):
        raise ValueError('Expected all 17 previously scored Case3 landmarks in source order')
    methods = {}
    for name in ('identity', 'rigid'):
        errors = [row[f'{name}_TRE_mm'] for row in rows]
        audited = [row[f'{name}_TRE_mm'] for row in audit['all_point_numerical_recount']['rows']]
        np.testing.assert_allclose(errors, audited, rtol=0, atol=1e-9)
        methods[name] = metrics(errors)
    return {'patient': summary['case_id'], 'role': role, 'methods': methods,
            'rigid_minus_identity_rms_mm': methods['rigid']['rms_mm'] - methods['identity']['rms_mm'],
            'observed_landmark_coordinates_reopened': False, 'refit': False,
            'ultrasound_sources': sources,
            'saved_summary': {'path': str(summary_path.relative_to(ROOT)), 'sha256': summary_sha},
            'independent_audit': {'path': str(audit_path.relative_to(ROOT)), 'sha256': audit_sha}}


def case4_observed_update():
    audit_path = CASE4 / 'evaluation-review/verification.json'
    audit_sha = '25103782b6762a507b758a4e21f7d7c446e61335d5ba352ddbf63b00b1ac943f'
    audit = json.loads(read_verified(audit_path, audit_sha))
    evaluation_path = CASE4 / 'fit-freeze/comparison/evaluation.json'
    evaluation_sha = audit['all_bound_sha256'][str(evaluation_path.relative_to(ROOT))]
    report = json.loads(read_verified(evaluation_path, evaluation_sha))
    freeze = json.loads(read_verified(ROOT / report['freeze']['path'], report['freeze']['sha256']))
    prediction = freeze['prediction_field']
    saved = json.loads(read_verified(ROOT / prediction['path'], prediction['sha256']))
    baseline = saved['baseline']
    field = BaselineField((json.dumps(baseline, sort_keys=True, indent=2, allow_nan=False) + '\n').encode())
    if (audit['status'] != 'PASS_saved_fixed_sparse_comparison_verified'
            or audit['case'] != 'RESECT:Case4' or audit['role'] != 'DEVELOPMENT'
            or saved['external_field'] is not None or report['landmark_freeze_hash'] != freeze['freeze_hash']
            or baseline['observations']['boundary_ids'] != audit['B_ids']
            or len(audit['B_ids']) != 6 or len(audit['V_ids']) != 13):
        raise ValueError('Expected the original Case4 DEVELOPMENT freeze and partition')
    points = [row['predicted_ras_mm'] for row in report['methods']['no_shift']['per_landmark']]
    predictions = sample_baselines(field, points)
    methods = {}
    for name, predicted in predictions.items():
        rows = report['methods'][name]['per_landmark']
        if [row['row_id'] for row in rows] != audit['V_ids'] or any(row['status'] != 'supported' for row in rows):
            raise ValueError('Case4 replay must retain every previously held-out point')
        np.testing.assert_allclose(predicted['predicted_ras_mm'],
                                   [row['predicted_ras_mm'] for row in rows], rtol=0, atol=1e-9)
        errors = [row['error_mm'] for row in rows]
        np.testing.assert_allclose(errors, [row[name] for row in audit['V_per_landmark']], rtol=0, atol=1e-9)
        methods[name] = metrics(errors)
    return {'patient': audit['case'], 'role': audit['role'], 'conditioning_count': len(audit['B_ids']),
            'previously_held_out_count': len(audit['V_ids']), 'methods': methods,
            'frozen_predictions_replayed': True, 'observed_landmark_coordinates_reopened': False,
            'refit': False, 'prediction_field': prediction,
            'saved_evaluation': {'path': str(evaluation_path.relative_to(ROOT)), 'sha256': evaluation_sha},
            'independent_audit': {'path': str(audit_path.relative_to(ROOT)), 'sha256': audit_sha}}


def plot_cavity(image, mask, spacing, destination):
    import matplotlib
    matplotlib.use('Agg')
    from matplotlib import pyplot as plt

    fig, axes = plt.subplots(2, 3, figsize=(9, 6), layout='constrained')
    points = np.argwhere(mask)
    limits = np.percentile(image[tuple(slice(lo, hi + 1) for lo, hi in zip(points.min(0), points.max(0)))], [2, 98])
    for axis in range(3):
        index = int(np.argmax(np.bincount(points[:, axis])))
        plane_axes = [i for i in range(3) if i != axis]
        slices = tuple(slice(max(0, points[:, i].min() - 16), min(mask.shape[i], points[:, i].max() + 17))
                       for i in plane_axes)
        plane = np.take(image, index, axis=axis)[slices].T
        label = np.take(mask, index, axis=axis)[slices].T
        for row in range(2):
            ax = axes[row, axis]
            ax.imshow(plane, cmap='gray', origin='lower', vmin=limits[0], vmax=limits[1],
                      aspect=spacing[plane_axes[1]] / spacing[plane_axes[0]])
            ax.axis('off')
        axes[0, axis].set_title(f'Native axis {axis}, slice {index}')
        axes[1, axis].contour(label, levels=[0.5], colors=['#22e2da'], linewidths=1, origin='lower')
    fig.suptitle('RESECT Case3 during-resection ultrasound\nNative planes and supplied visible cavity contour')
    fig.savefig(destination, dpi=140)
    plt.close(fig)


def case3_cavity(root, output):
    declaration = require_manifest()
    rights, ultrasound, annotation = declaration['files']
    rights_path = ARTIFACTS / 'resect-component-admission-v1/rights-review-02/README.txt'
    verify_source_file(rights_path, rights, deadline=monotonic() + 5)
    annotation_path = root / CASE3_CAVITY
    verify_source_file(annotation_path, annotation, deadline=monotonic() + 5)
    original_path = root / 'sources/RESECT/v1/Case3' / Path(ultrasound['path']).name
    ultrasound_sha = verify_source_file(original_path, ultrasound, deadline=monotonic() + 15)
    workspace_path = root / CASE3_WORKSPACE
    if file_sha256(workspace_path) != CASE3_WORKSPACE_SHA:
        raise ValueError('Existing Case3 source-stage workspace differs from its release')
    with ZipFile(workspace_path) as archive:
        metadata = extension_layout(archive)
        attachments = metadata['images']
        during_row = attachments[2]
        during = load_case(BytesIO(archive.read(during_row['member'])))
    primary = load_case(workspace_path)
    if (primary.metadata['data_case']['family_id'] != declaration['patient_group']
            or primary.metadata['data_case']['role'] != declaration['role'] or primary.compartments
            or during.semantic_hash != during_row['source']['sourceCaseHash']
            or during.metadata['acquisition_stage'] != 'intraoperative-during'
            or during.source_refs[0].sha256 != ultrasound_sha or during.compartments):
        raise ValueError('Expected the same unlabelled MRI reference and during-US source')
    source_header, mask_header = (inspect_nifti(path) for path in (original_path, annotation_path))
    source_affine = np.array(source_header['affine_ras_mm'])
    mask_affine = np.array(mask_header['affine_ras_mm'])
    if source_header['shape'] != mask_header['shape'] or tuple(source_header['shape']) != during.mri.shape:
        raise ValueError('Ultrasound and annotation grid shapes differ')
    corner = _maximum_corner_displacement(source_affine, mask_affine, during.mri.shape)
    if (corner > declaration['maximum_grid_corner_difference_mm'] or not np.array_equal(source_affine, during.affine)
            or not np.array_equal(nib.load(original_path).get_fdata(dtype=np.float32), during.mri)):
        raise ValueError('Original ultrasound, annotation and saved display grid are incompatible')

    mask_values = np.asarray(nib.load(annotation_path).dataobj)
    if not np.array_equal(np.unique(mask_values), [0, 1]):
        raise ValueError('Expected the supplied binary cavity labels')
    mask = mask_values == 1
    del mask_values
    ref = replace(during.source_refs[0], source_id='supplied_target_annotation', uri=annotation['source_url'],
                  sha256=annotation['sha256'], license=annotation['license'])
    during = replace(during, source_refs=(*during.source_refs, ref), compartments={CAVITY_LABEL: mask},
                     source_compartments={CAVITY_LABEL: mask}, metadata={**during.metadata,
                         'annotationTimeRole': 'intraoperative_observed_evaluator_only',
                         'annotationPublisherDOI': declaration['annotation_release_doi'],
                         'annotationRightsSHA256': rights['sha256'], 'annotationNativeGrid': mask_header,
                         'display_label': 'US during resection · supplied visible cavity'})
    display = DisplaySource(during, 'other-3D-scalar', 'source-provided')
    state = view_state(primary, [display])
    identity = display.identity(primary)
    state['selectedSeriesId'] = identity
    state['states'][identity]['viewer'] = {'volumeRendering': {'enabled': False, 'preset': 'mri'}}
    state['states'][identity]['cursor'] = nib.affines.apply_affine(during.affine, np.argwhere(mask).mean(0)).tolist()
    path = output / 'RESECT-Case3-observed-cavity.ressectionlab'
    save_workspace(primary, path, sources=[display], imaging=state, episode=None, selection=None, artifacts={},
                   originals=[OriginalFile.from_path(annotation_path, role='annotation')])
    reopened, artifacts, workspace = load_workspace(path)
    annotation_source, = workspace['sources']
    saved = annotation_source.case.compartments[CAVITY_LABEL]
    if (reopened.semantic_hash != primary.semantic_hash or artifacts or workspace['episode'] is not None
            or annotation_source.manifest(reopened)['scope'] != 'display-only-native-grid'
            or annotation_source.case.semantic_hash != during.semantic_hash
            or not np.array_equal(saved, mask)):
        raise ValueError('Cold reopen did not preserve the observed display-only annotation')
    overlay = output / 'case3-native-cavity-overlay.png'
    plot_cavity(during.mri, saved, source_header['spacing_mm'], overlay)
    count = int(mask.sum())
    return {'patient': declaration['patient_group'], 'role': declaration['role'], 'shape': list(mask.shape),
            'spacing_mm': source_header['spacing_mm'], 'source_mask_to_us_corner_max_mm': corner,
            'positive_voxels': count, 'visible_cavity_volume_mm3': count * mask_header['voxel_volume_mm3'],
            'original_vs_reopened_dice': 2 * int(np.count_nonzero(mask & saved)) / (count + int(saved.sum())),
            'observed_mask_sha256': annotation['sha256'], 'ultrasound_sha256': ultrasound_sha,
            'source_workspace': str(CASE3_WORKSPACE), 'source_workspace_sha256': CASE3_WORKSPACE_SHA,
            'reopened_workspace': path.name, 'reopened_workspace_sha256': file_sha256(path),
            'overlay': overlay.name, 'overlay_sha256': file_sha256(overlay),
            'cavity_observation_available_to_preoperative_planning': False, 'resampling': False,
            'scope': 'Native-grid transport fidelity for the supplied during-resection visible cavity'}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-name', required=True, help='Fresh child directory under qc/surgical-change')
    args = parser.parse_args()
    if not args.output_name or Path(args.output_name).name != args.output_name or args.output_name in {'.', '..'}:
        parser.error('--output-name must be a single directory name')
    root = checked_root(required_bytes=64 * 1024**2)
    registration = case3_registration(root)
    observed_update = case4_observed_update()
    output = root / 'qc/surgical-change' / args.output_name
    output.mkdir()
    report = {'schema': 'resectionlab.resect-surgical-change.v1', 'case3_registration': registration,
              'case4_observed_update': observed_update, 'case3_cavity': case3_cavity(root, output),
              'simulation_cavity_comparison': 'unavailable_no_same_case_registered_simulation_cavity',
              'coverage': 'Retrospective sparse landmark evidence and observed cavity transport on two exposed people',
              'source_doi': '10.11582/2017.00004', 'annotation_doi': '10.17605/OSF.IO/JV8BK',
              'annotation_methods_doi': '10.1002/mp.17317'}
    (output / 'benchmark.json').write_text(json.dumps(report, indent=2, allow_nan=False) + '\n')
    print(output / 'benchmark.json')


if __name__ == '__main__':
    main()
