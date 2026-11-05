#!/usr/bin/env python3
"""Build target-free CTA anatomy hosts from the exact independently reviewed inputs."""
from __future__ import annotations

import argparse
from dataclasses import replace
import json
import os
from pathlib import Path
import re
import resource
import sys
import tempfile
import time

import nibabel as nib
import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / 'src'))
from data_rebuild_acquire import UUID
from resectionlab.core import array_digest, thaw_json
from resectionlab.data_root import atomic_json, checked_root, safe_path, sha256_file
from resectionlab.surgical_scene import from_case
from resectionlab.workspace_bundle import load_workspace, save_workspace, view_state

SUBJECTS = ('004', '006', '012', '014', '018')
DECISIONS = 'integrated-v1/qc/independent/model-brain-case-decisions-v1.json'
INDEX = 'integrated-v1/model-brain-candidates-v1/candidate-index.json'
ORDER = ('brain', 'estimated_skull', 'supplied_arterial_annotation_union',
         'supplied_venous_sinus_annotation_union')
TARGET = 'synthetic_target'
LIMITS = (
    'Baseline acquired clinical CTA. Tumor status unestablished; no modeled target.',
    'Coarse additive brain/intracranial estimate may include CSF. Source crop and model boundaries are uncertain.',
    'Local dense skull estimate from acquired CT. Skull base and other omitted or truncated bone remain unknown.',
    'Supplied landmark arteries and selected veins/sinuses are partial. Unlabelled tissue is not vessel-free.',
    'Unowned cells and space beyond the native field of view are unknown. The unchanged superior enclosure is not a free-space or task domain.',
    'No fused MRA, established scalp/dural coverage, functional or tract mapping, clinical validation, or instrument task authorization.',
)
ADMISSION = 'Baseline anatomy inspection and scenario input; no reviewed instrument task'


def checked_file(root, record):
    path = safe_path(root, record['path'])
    if sha256_file(path) != record['sha256']:
        raise ValueError(f'Input checksum differs: {record["path"]}')
    return path


def binding(root, path):
    return {'path': str(path.relative_to(root)), 'sha256': sha256_file(path)}


def make_baseline(parent, case_id, evidence):
    """Restore ownership from intact memberships and start a fresh authority record."""
    metadata = thaw_json(parent.metadata)
    prior_order = tuple(metadata['surgical_adapter']['ownershipPrecedence'])
    if (prior_order != (ORDER[0], TARGET, *ORDER[1:])
            or set(parent.compartments) != set(prior_order)
            or set(parent.source_compartments) != set(prior_order)
            or parent.brain_mask is not None
            or any(not np.array_equal(parent.compartments[k], parent.source_compartments[k])
                   for k in prior_order)):
        raise ValueError('Reviewed, unmodified parent memberships and explicit precedence required')
    masks = {name: parent.source_compartments[name] for name in ORDER}
    previous_target = parent.compartments[TARGET]
    if (not previous_target.any() or not masks['brain'][previous_target].all()
            or any(masks[name][previous_target].any() for name in ORDER[1:])):
        raise ValueError('Prior target must be fully backed by brain and disjoint from retained obstacles')

    layers = metadata['anatomy_layers']
    del layers[TARGET]
    layers['brain']['coverage'] = LIMITS[1]
    layers['estimated_skull']['coverage'] = LIMITS[2]
    for name in ORDER[:2]:
        layers[name].update(uncertainty=list(LIMITS), review_status='pending_exact_host_review')
    for name, label, color in zip(ORDER[2:], ('Supplied landmark arteries', 'Supplied veins and sinuses'),
                                  ('#e7645c', '#5a9de6')):
        layers[name].update(coverage=f'{label}, partial annotation.', color=color,
                            provenance='supplied_annotation', review_status='pending_exact_host_review')

    data = metadata['data_case']
    source_fields = ('schema', 'family_id', 'original_patient', 'role', 'split', 'source', 'source_doi',
                     'source_version', 'source_files', 'source_url', 'source_license', 'source_integrity',
                     'source_frame', 'source_registration', 'whole_archive_md5_verified',
                     'annotation_grouping', 'population', 'diagnosis', 'clinical_use_status', 'missing_anatomy')
    components = {name: value for name, value in data['components'].items() if name != 'tumor'}
    components['brain']['coverage'] = LIMITS[1]
    components['skull']['coverage'] = LIMITS[2]
    metadata['data_case'] = {
        **{key: data[key] for key in source_fields if key in data},
        'case_id': case_id, 'case_type': 'real', 'classification': 'real_acquired_CTA_anatomy_host',
        'accepted': False, 'status': 'pending_independent_host_review', 'components': components,
        'observations': {'primary_image': 'unchanged_acquired_CTA_baseline', 'target': 'no_modeled_target',
                         'tumor_status': 'unestablished'},
        'capabilities': ['source_CTA_and_partial_vascular_annotation_inspection',
                         'coarse_brain_and_local_dense_skull_overlay_inspection', 'baseline_scenario_input'],
        'new_real_patient_count': 0, 'synthetic_variant_count': 0, 'clinical_validation': False,
        'limitations': list(LIMITS), 'simulation_admission': ADMISSION, 'host_provenance': evidence,
    }
    for key in ('integrated_rehearsal', 'surgical_properties', 'working_grid_refinement'):
        metadata.pop(key, None)
    metadata.update(is_synthetic=False, selected_modality='CTA (ACQUIRED BASELINE)',
                    display_label='Acquired CTA anatomy host',
                    annotation_role='supplied vascular references and estimated brain and skull',
                    simulation_admission=ADMISSION,
                    surgical_adapter={'ownershipPrecedence': list(ORDER), 'coverageLimitations': list(LIMITS)})
    refs = tuple(ref for ref in parent.source_refs
                 if ref.provenance != 'simulated' and ref.source_id != 'estimate-tumor_placement_support')
    return replace(parent, case_id=case_id, compartments=masks, source_compartments=masks,
                   source_refs=refs, metadata=metadata, unknowns=(*parent.unknowns, *LIMITS))


def verify_inputs(root, parent, workspace, proposal):
    entries = proposal['sources']
    source_members = ('/imagesTr_topbrain/topcow_ct_', '/labelsTr_topbrain_v1_ct/')
    if len(entries) != len(source_members) or any(
            part not in row['source_member'] for part, row in zip(source_members, entries)):
        raise ValueError('Expected original CTA and its source V1 vascular labels')
    ct, labels = [nib.load(checked_file(root, row)) for row in entries]
    if (parent.frame != 'RAS+' or ct.header.get_xyzt_units()[0] != 'mm'
            or any(obj.shape != parent.mri.shape or not np.array_equal(obj.affine, parent.affine)
                   for obj in (ct, labels)) or not np.array_equal(ct.get_fdata(dtype=np.float32), parent.mri)):
        raise ValueError('Native package must preserve the acquired CTA values and original physical frame')
    grouped = np.asanyarray(labels.dataobj)
    if not np.isin(grouped, np.arange(41)).all() or any(
            not np.array_equal(mask, parent.compartments[name])
            for name, mask in zip(ORDER[2:], ((grouped >= 1) & (grouped <= 34), grouped >= 35))):
        raise ValueError('Supplied native vessel groups differ from the original V1 CT dictionary')
    files = thaw_json(parent.metadata)['data_case']['source_files']
    if len(files) != 6 or sorted(row['sha256'] for row in files) != sorted(
            item.sha256 for item in workspace['originals']):
        raise ValueError('Six exact acquired original files required')
    for compartment, output in zip(ORDER[:2], ('brain_estimate', 'observed_skull')):
        image = nib.load(checked_file(root, proposal['outputs'][output]))
        if (image.shape != parent.mri.shape or not np.array_equal(image.affine, parent.affine)
                or not np.array_equal(np.asanyarray(image.dataobj), parent.compartments[compartment])):
            raise ValueError(f'Native anatomy differs from exact source proposal: {compartment}')
    for output in ('enclosed_coverage', 'tumor_placement_support'):
        checked_file(root, proposal['outputs'][output])


def write_overlays(case, path):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    voxel = np.asarray(case.mri.shape) // 2
    planes = [(0, voxel[0], 'Sagittal'), (1, voxel[1], 'Coronal'),
              (2, voxel[2], 'Axial'), (2, int(case.mri.shape[2] * .72), 'Superior axial')]
    colors = ('#65ed85', '#63dce8', '#ff6756', '#6c98ff')
    layers = ('brain estimate', 'local dense skull estimate', 'supplied arteries', 'supplied veins/sinuses')
    figure, axes = plt.subplots(2, len(planes), figsize=(15, 7), layout='constrained')
    for column, (axis, index, title) in enumerate(planes):
        section = np.take(case.mri, index, axis=axis).T
        spacing = np.delete(case.spacing_mm, axis)
        # Pixel centers and contour positions both use the native millimeter grid.
        x, y = np.arange(section.shape[1]) * spacing[0], np.arange(section.shape[0]) * spacing[1]
        extent = [-spacing[0] / 2, x[-1] + spacing[0] / 2, -spacing[1] / 2, y[-1] + spacing[1] / 2]
        for row in range(2):
            plot = axes[row, column]
            plot.imshow(section, cmap='gray', origin='lower', vmin=-10, vmax=110, extent=extent)
            plot.set_title(f'{title} | native index {index}' if row == 0 else 'Aligned memberships')
            plot.set_xticks([])
            plot.set_yticks([])
        for name, color in zip(ORDER, colors):
            mask = np.take(case.compartments[name], index, axis=axis).T
            if mask.any() and not mask.all():
                axes[1, column].contour(x, y, mask, levels=[.5], colors=[color], linewidths=.65)
    figure.suptitle(f'{case.case_id} | Acquired CTA baseline and partial anatomy')
    figure.legend(handles=[plt.Line2D([], [], color=color, label=layer) for color, layer in zip(colors, layers)],
                  loc='outside lower center', ncols=4, frameon=False)
    figure.savefig(path, dpi=140)
    plt.close(figure)
    return [{'axis': axis, 'index': int(index), 'plane': name} for axis, index, name in planes]


def build(root, subject, version):
    if subject not in SUBJECTS or not re.fullmatch(r'[A-Za-z0-9_-]+', version):
        raise ValueError('Select a pinned subject and a plain immutable version token')
    start = time.perf_counter()
    root = checked_root(root, expected_uuid=UUID, required_bytes=2 * 1024**3)
    family = f'TopCoW:{subject}'
    case_id = f'TopCoW-{subject}-CTA-ANATOMY-{version}'
    directory = safe_path(root, f'integrated-v1/anatomy-hosts-{version}/{case_id}')
    if directory.exists():
        raise FileExistsError(f'Immutable host exists: {directory}')
    decisions_path, index_path = safe_path(root, DECISIONS), safe_path(root, INDEX)
    decisions, index = json.loads(decisions_path.read_text()), json.loads(index_path.read_text())
    decision = next(row for row in decisions['cases'] if row['family_id'] == family)
    prior = next(row for row in index['people'] if row['subject'] == subject)
    if (decisions['schema'] != 'ressectionlab-independent-release-decisions-v1'
            or decision['decision'] != 'accepted_with_explicit_limitations' or decision['role'] != 'TRAIN'
            or decision['sha256'] != prior['candidate_sha256'] or decision['path'] != prior['candidate_path']):
        raise ValueError('Independent exact-byte anatomy review and frozen TRAIN identity required')
    parent_path = checked_file(root, decision)
    for review in decision['evidence']:
        checked_file(root, review)
    parent, artifacts, workspace = load_workspace(parent_path)
    if (workspace is None or workspace['episode'] is not None or workspace['sources'] or artifacts
            or parent.case_id != decision['case_id'] or parent.metadata['data_case']['family_id'] != family):
        raise ValueError('Expected reviewed anatomy source without history or fused images')

    recipe = parent.metadata['data_case']['recipe']
    proposal_record = {'path': recipe['proposal_path'], 'sha256': prior['proposal_sha256']}
    proposal = json.loads(checked_file(root, proposal_record).read_text())
    if (proposal['family_id'] != family or proposal['role'] != 'TRAIN'
            or proposal['schema'] != 'ressectionlab-additive-model-brain-proposal-v1'
            or recipe['proposal_sha256'] != prior['proposal_sha256']):
        raise ValueError('Expected exact reviewed additive brain proposal for this family')
    verify_inputs(root, parent, workspace, proposal)
    evidence = {
        'reviewed_anatomy_parent': {'path': decision['path'], 'sha256': decision['sha256'],
                                    'case_id': parent.case_id, 'case_hash': parent.semantic_hash},
        'parent_review': binding(root, decisions_path), 'parent_index': binding(root, index_path),
        'anatomy_proposal': proposal_record,
        'anatomy_masks': {name: {k: proposal['outputs'][name][k] for k in ('path', 'sha256')}
                          for name in ('brain_estimate', 'observed_skull', 'enclosed_coverage',
                                       'tumor_placement_support')},
        'source_CTA_and_v1_labels': [{k: row[k] for k in ('path', 'sha256')} for row in proposal['sources']],
    }
    candidate = make_baseline(parent, case_id, evidence)
    target = parent.compartments[TARGET]
    previous_scene = from_case(parent)
    candidate_scene = from_case(candidate)
    # Region IDs move when the target is removed. Compare ownership by name.
    before = {row['name']: row['id'] for row in previous_scene.regions}
    after = {row['name']: row['id'] for row in candidate_scene.regions}
    for name in ORDER:
        expected = previous_scene.labels == before[name]
        if name == 'brain':
            expected |= target
        if not np.array_equal(candidate_scene.labels == after[name], expected):
            raise ValueError(f'Host changed ownership beyond restoring former target brain: {name}')
    if not np.array_equal(candidate_scene.labels == 0, previous_scene.labels == 0):
        raise ValueError('Host changed unknown-space ownership')
    counts = {name: int(np.count_nonzero(candidate_scene.labels == value)) for name, value in after.items()}
    counts['unknown'] = int(np.count_nonzero(candidate_scene.labels == 0))
    del previous_scene, candidate_scene, expected

    checked_root(root, expected_uuid=UUID, required_bytes=2 * 1024**3)
    directory.parent.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault('MPLCONFIGDIR', str(safe_path(root, 'integrated-v1/cache/matplotlib')))
    destination = directory / f'{case_id}.ressectionlab'
    with tempfile.TemporaryDirectory(prefix=f'.{subject}-', dir=directory.parent) as temp:
        staging = Path(temp) / case_id
        staging.mkdir()
        native = staging / destination.name
        state = view_state(candidate, [])
        state['states']['primary'].setdefault('viewer', {})['volumeRendering'] = {'enabled': False}
        save_start = time.perf_counter()
        save_workspace(candidate, native, sources=[], imaging=state, episode=None, selection=None,
                       artifacts={}, originals=workspace['originals'])
        save_seconds = time.perf_counter() - save_start
        reopen_start = time.perf_counter()
        reopened, saved_artifacts, saved = load_workspace(native)
        if (reopened.semantic_hash != candidate.semantic_hash or saved is None or saved['sources']
                or saved['episode'] is not None or saved['selection'] is not None or saved_artifacts
                or saved['originals'] != workspace['originals']):
            raise ValueError('Native reopen changed anatomy or original files, or introduced state')
        reopen_seconds = time.perf_counter() - reopen_start
        overlay = staging / 'source-three-plane-review.png'
        planes = write_overlays(reopened, overlay)
        report = {
            'schema': 'ressectionlab-acquired-CTA-anatomy-host-v1', 'case_id': case_id,
            'family_id': family, 'role': 'TRAIN', 'path': str(destination.relative_to(root)),
            'sha256': sha256_file(native), 'case_hash': reopened.semantic_hash, 'bytes': native.stat().st_size,
            'accepted': False, 'new_real_people': 0, 'new_synthetic_variants': 0,
            'source': evidence, 'shape': list(reopened.mri.shape), 'affine_ras_mm': reopened.affine.tolist(),
            'primary_array_hash': array_digest(reopened.mri),
            'compartment_hashes': {name: array_digest(mask) for name, mask in reopened.compartments.items()},
            'ownership_precedence': list(ORDER), 'owned_voxel_counts': counts,
            'former_target_voxels_owned_by_brain': int(np.count_nonzero(target)),
            'embedded_originals': [item.manifest(i) for i, item in enumerate(saved['originals'])],
            'overlay': {'path': str((directory / overlay.name).relative_to(root)),
                        'sha256': sha256_file(overlay), 'planes': planes, 'CTA_window': [-10, 110]},
            'production_workspace_roundtrip': 'passed', 'clinical_validation': False,
            'instrument_task_authorized': False, 'physical_history_carried': False,
            'builder_sha256': sha256_file(__file__), 'elapsed_seconds': time.perf_counter() - start,
            'save_seconds': save_seconds, 'reopen_seconds': reopen_seconds,
            'process_peak_rss_bytes': resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
            'limitations': list(LIMITS),
        }
        atomic_json(staging / 'report.json', report)
        checked_root(root, expected_uuid=UUID)
        staging.rename(directory)
    return report


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root')
    parser.add_argument('--subject', choices=SUBJECTS, required=True)
    parser.add_argument('--version', default='v1')
    args = parser.parse_args()
    print(json.dumps(build(args.root, args.subject, args.version), indent=2))
