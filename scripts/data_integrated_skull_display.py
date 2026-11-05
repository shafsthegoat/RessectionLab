#!/usr/bin/env python3
"""Prepare the CTA006 mineralized skull estimate as a separate display-only series."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import re
import tempfile
import time

import nibabel as nib
import numpy as np

from data_integrated_hosts import UUID, ORDER, binding, checked_file
from resectionlab.core import CaseData, SourceRef, array_digest
from resectionlab.data_root import atomic_json, checked_root, safe_path, sha256_file
from resectionlab.workspace_bundle import load_workspace, save_workspace, view_state
from resectionlab.workspace_imaging import DisplaySource

HOSTS = 'integrated-v1/anatomy-hosts-v1'
HOST_ID = 'TopCoW-006-CTA-ANATOMY-v1'
REVIEW = 'integrated-v1/qc/independent/CTA-006-hybrid-preparation-candidate-decision-v1.json'
PILOT = 'integrated-v1/anatomy/TopCoW-006/single-array-pilot-v1'
SOURCE_SHEET = f'{PILOT}/qc/hybrid-density-three-plane-review.png'
MASK_NAME = 'broader_mineralized_skull_estimate'
LIMITS = (
    'The broader mask estimates mineralized tissue using acquired CTA density inside a model-derived skull prior.',
    'Possible calcification, artifacts and partial-volume effects remain. Lower-density bone and diploe may be missing.',
    'The estimate contacts crop faces. Unowned or truncated anatomy remains unknown.',
    'Estimated display series only. The primary baseline and its skull, brain and supplied vessel material memberships are unchanged.',
)


def binary_input(path, case):
    image = nib.load(path)
    if image.shape != case.mri.shape or not np.array_equal(image.affine, case.affine):
        raise ValueError('Exact source native grid required')
    values = np.asanyarray(image.dataobj)
    if not np.isin(values, (0, 1)).all():
        raise ValueError('Binary mask required')
    return values.astype(bool)


def inspect_overlay(case, broader, path):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    planes = [(0, 145, 'Sagittal'), (1, 201, 'Coronal'),
              (2, 58, 'Inferior axial'), (2, 162, 'Superior axial')]
    rows = (('Acquired CTA, bone window', (-250, 1250), {}),
            ('Acquired CTA, skull comparison', (-250, 1250),
             {'old local dense skull': ('#6dff82', case.compartments['estimated_skull']),
              'broader mineralized estimate': ('#56e4fa', broader)}),
            ('Acquired CTA, tissue and vascular context', (-10, 110),
             {'broader mineralized estimate': ('#56e4fa', broader),
              'brain estimate': ('#efb7f7', case.compartments['brain']),
              'supplied arteries': ('#ff6756', case.compartments[ORDER[2]]),
              'supplied veins/sinuses': ('#6c98ff', case.compartments[ORDER[3]])}))
    figure, axes = plt.subplots(3, 4, figsize=(14, 10), layout='constrained')
    legend = {}
    for column, (axis, index, title) in enumerate(planes):
        section = np.take(case.mri, index, axis=axis).T
        spacing = np.delete(case.spacing_mm, axis)
        x, y = np.arange(section.shape[1]) * spacing[0], np.arange(section.shape[0]) * spacing[1]
        extent = [-spacing[0] / 2, x[-1] + spacing[0] / 2, -spacing[1] / 2, y[-1] + spacing[1] / 2]
        for row, (heading, (low, high), masks) in enumerate(rows):
            plot = axes[row, column]
            plot.imshow(section, cmap='gray', origin='lower', vmin=low, vmax=high, extent=extent)
            plot.set_title(f'{title} | native index {index}' if row == 0 else heading, fontsize=9)
            plot.set_xticks([])
            plot.set_yticks([])
            for name, (color, mask) in masks.items():
                layer = np.take(mask, index, axis=axis).T
                if layer.any() and not layer.all():
                    plot.contour(x, y, layer, levels=[.5], colors=[color], linewidths=.7,
                                 linestyles='dotted' if name == 'old local dense skull' else 'solid')
                legend.setdefault(name, Line2D([], [], color=color, label=name,
                                  linestyle='dotted' if name == 'old local dense skull' else 'solid'))
    figure.suptitle('CTA006 | Broader mineralized skull estimate for display only')
    figure.legend(handles=legend.values(), loc='outside lower center', ncols=5, frameon=False)
    figure.savefig(path, dpi=140)
    plt.close(figure)
    return [{'axis': axis, 'index': index, 'name': title} for axis, index, title in planes]


def build(root, version):
    if not re.fullmatch(r'[A-Za-z0-9_-]+', version):
        raise ValueError('Use a plain immutable version token')
    started = time.perf_counter()
    root = checked_root(root, expected_uuid=UUID, required_bytes=2 * 1024**3)
    directory = safe_path(root, f'integrated-v1/skull-display-{version}')
    if directory.exists():
        raise FileExistsError(f'Immutable candidate exists: {directory}')
    index_path = safe_path(root, f'{HOSTS}/candidate-index.json')
    host_row = next(row for row in json.loads(index_path.read_text())['cases'] if row['case_id'] == HOST_ID)
    host_report_record = {'path': host_row['report_path'], 'sha256': host_row['report_sha256']}
    host_report = json.loads(checked_file(root, host_report_record).read_text())
    if host_report['sha256'] != host_row['sha256']:
        raise ValueError('Host receipt points to another primary package')
    host, artifacts, workspace = load_workspace(checked_file(root, host_row))
    if (host.case_id != HOST_ID or set(host.compartments) != set(ORDER) or workspace is None
            or artifacts or workspace['sources'] or workspace['episode'] is not None):
        raise ValueError('Exact target-free CTA006 core host required')

    proposal_record = host_report['source']['anatomy_proposal']
    proposal = json.loads(checked_file(root, proposal_record).read_text())
    review_path = safe_path(root, REVIEW)
    review = json.loads(review_path.read_text())
    historical_sheet = {'path': SOURCE_SHEET, 'sha256': review['source_review_sheet_sha256']}
    checked_file(root, historical_sheet)
    source = proposal['sources'][0]
    inputs = proposal['inputs']
    if (review['schema'] != 'independent006-additive-density-spatial-prior-preparation-review-v1'
            or review['decision'] != 'accepted_as_separate_additive_research_geometry_candidate_pending_actual_native_QC'
            or source['sha256'] != review['source_sha256']
            or inputs['model_brain']['sha256'] != review['brain_mask_sha256']
            or inputs['model_skull']['sha256'] != review['model_skull_sha256']):
        raise ValueError('CTA006 spatial prior and source do not match the preparation review')
    ct = nib.load(checked_file(root, source))
    if (ct.header.get_xyzt_units()[0] != 'mm' or not np.array_equal(ct.affine, host.affine)
            or not np.array_equal(ct.get_fdata(dtype=np.float32), host.mri)):
        raise ValueError('Display source must match exact acquired CTA baseline')
    old = host.compartments['estimated_skull']
    if not np.array_equal(old, binary_input(checked_file(root, proposal['outputs']['observed_skull']), host)):
        raise ValueError('Primary skull estimate differs from reviewed old bone')
    model = binary_input(checked_file(root, inputs['model_skull']), host)
    candidate = old | (model & (host.mri >= 500))
    added = candidate & ~old
    if (np.count_nonzero(candidate) != review['hybrid_voxels_independently_reproduced']
            or any(host.compartments[name][added].any() for name in ORDER if name != 'estimated_skull')):
        raise ValueError('Broader skull support differs from reviewed count or overlaps retained brain or vessels')
    low_density_excluded = model & ~candidate
    covered_faces = {f'{axis}:{index}': int(np.count_nonzero(np.take(candidate, index, axis=axis)))
                     for axis in range(3) for index in (0, -1)}
    mask_name = 'broader-mineralized-skull-estimate.nii.gz'
    project_name = f'{HOST_ID}-broader-skull-display-{version}.ressectionlab'
    checked_root(root, expected_uuid=UUID, required_bytes=2 * 1024**3)
    with tempfile.TemporaryDirectory(prefix='.skull-display-', dir=directory.parent) as temp:
        staging = Path(temp) / directory.name
        staging.mkdir()
        mask = staging / mask_name
        header = ct.header.copy()
        header.set_data_dtype(np.uint8)
        nib.save(nib.Nifti1Image(candidate.astype(np.uint8), host.affine, header), mask)
        if not np.array_equal(binary_input(mask, host), candidate):
            raise ValueError('Exported inspection mask changed')
        estimate_binding = {'path': str((directory / mask_name).relative_to(root)), 'sha256': sha256_file(mask)}
        structural = next(ref for ref in host.source_refs if ref.source_id == 'structural')
        estimate = SourceRef('supplied_target_annotation', (directory / mask_name).as_uri(),
                             estimate_binding['sha256'], native_frame='RAS+', provenance='estimated')
        display = CaseData(case_id=f'{HOST_ID}-broader-skull-display', mri=host.mri,
                           affine=host.affine, frame=host.frame, compartments={MASK_NAME: candidate},
                           source_refs=(structural, estimate), metadata={
                               'display_label': 'Acquired CTA + broader mineralized skull estimate',
                               'anatomy_layers': {MASK_NAME: {
                                   'kind': 'skull', 'label': 'Broader mineralized skull estimate',
                                   'provenance': 'estimated', 'review_status': 'pending_exact_display_review',
                                   'coverage': LIMITS[0], 'color': '#56e4fa', 'uncertainty': list(LIMITS),
                               }},
                           })
        series = DisplaySource(display, 'CTA', 'estimated')
        native = staging / project_name
        state = view_state(host, [series])
        state['selectedSeriesId'] = series.identity(host)
        for image_state in state['states'].values():
            image_state.setdefault('viewer', {})['volumeRendering'] = {'enabled': False}
        save_workspace(host, native, sources=[series], imaging=state, episode=None,
                       selection=None, artifacts={}, originals=workspace['originals'])
        reopened, saved_artifacts, saved = load_workspace(native)
        if (reopened.semantic_hash != host.semantic_hash or saved_artifacts or saved['episode'] is not None
                or saved['originals'] != workspace['originals'] or len(saved['sources']) != 1
                or saved['sources'][0].case.semantic_hash != display.semantic_hash
                or saved['imagingState']['selectedSeriesId'] != series.identity(host)):
            raise ValueError('Native display export changed primary anatomy, originals, or auxiliary evidence')
        overlay = staging / 'source-skull-comparison.png'
        planes = inspect_overlay(reopened, saved['sources'][0].case.compartments[MASK_NAME], overlay)
        report = {
            'schema': 'ressectionlab-CTA006-broader-skull-display-candidate-v1',
            'path': str((directory / project_name).relative_to(root)), 'sha256': sha256_file(native),
            'bytes': native.stat().st_size, 'case_hash': reopened.semantic_hash, 'family_id': 'TopCoW:006',
            'primary_host': host_row, 'display_series': series.manifest(host), 'accepted': False,
            'primary_material_memberships_unchanged': True, 'instrument_task_authorized': False,
            'clinical_validation': False, 'new_source_people': 0, 'new_synthetic_variants': 0,
            'formula': review['formula'], 'source_ct': {key: source[key] for key in ('path', 'sha256')},
            'source_proposal': proposal_record, 'model_skull': inputs['model_skull'],
            'preparation_review': binding(root, review_path), 'historical_overlay': historical_sheet,
            'estimated_mask': estimate_binding, 'estimated_mask_array_hash': array_digest(candidate),
            'voxels': int(candidate.sum()), 'added_voxels': int(added.sum()),
            'added_minimum_scaled_ct': float(host.mri[added].min()),
            'excluded_low_density_model_voxels': int(low_density_excluded.sum()),
            'boundary_voxels_per_face': covered_faces,
            'overlay': {'path': str((directory / overlay.name).relative_to(root)),
                        'sha256': sha256_file(overlay), 'planes': planes},
            'production_workspace_roundtrip': 'passed', 'limitations': list(LIMITS),
            'elapsed_seconds': time.perf_counter() - started, 'builder_sha256': sha256_file(__file__),
        }
        atomic_json(staging / 'report.json', report)
        checked_root(root, expected_uuid=UUID)
        staging.rename(directory)
    return report


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root')
    parser.add_argument('--version', default='v1')
    args = parser.parse_args()
    print(json.dumps(build(args.root, args.version), indent=2))
