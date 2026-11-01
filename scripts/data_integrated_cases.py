#!/usr/bin/env python3
"""Versioned native CTA geometry scenarios; estimates never become measured truth."""
from __future__ import annotations

import argparse
from dataclasses import replace
import json
from pathlib import Path
import re
import sys
import time

import nibabel as nib
import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(REPO / 'src')]
from resectionlab.core import SourceRef, array_digest, thaw_json, semantic_digest
from resectionlab.data_root import checked_root, atomic_json, safe_path, sha256_file
from resectionlab.surgical_scene import from_case
from resectionlab.workspace_bundle import load_workspace, save_workspace, view_state


def ellipsoid(shape, affine, center, radii):
    center, radii = np.asarray(center, float), np.asarray(radii, float)
    if center.shape != (3,) or radii.shape != (3,) or not np.isfinite(center).all() or not np.isfinite(radii).all() or np.any(radii <= 0):
        raise ValueError('Finite physical center and positive three-axis RAS radii required')
    mask = np.zeros(shape, bool)
    x, y = np.meshgrid(np.arange(shape[0]), np.arange(shape[1]), indexing='ij')
    for z in range(shape[2]):
        coordinates = np.stack((x, y, np.full_like(x, z)), axis=-1)
        world = coordinates @ np.asarray(affine)[:3, :3].T + np.asarray(affine)[:3, 3]
        mask[:, :, z] = np.sum(((world - center) / radii)**2, axis=-1) <= 1.
    if not mask.any(): raise ValueError('Target is empty in source physical grid')
    # Reject truncated ellipsoids rather than silently clipping to an image.
    inverse = np.linalg.inv(affine)
    voxel_center = inverse[:3, :3] @ center + inverse[:3, 3]
    radius_voxel = np.sqrt(np.sum((inverse[:3, :3] * radii)**2, axis=1))
    if np.any(voxel_center - radius_voxel < 0) or np.any(voxel_center + radius_voxel > np.asarray(shape) - 1):
        raise ValueError('Synthetic target extends beyond source coverage')
    return mask


def target_contact_sheet(case, center, destination):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    voxel = np.rint(case.world_to_voxel(center)).astype(int)
    colors = {'brain': 'lime', 'synthetic_target': 'yellow', 'estimated_skull': 'cyan'}
    colors.update({name: 'red' for name in case.compartments if name not in colors})
    fig, axes = plt.subplots(3, 3, figsize=(14, 13))
    for axis, name in enumerate(['sagittal/native-axis0', 'coronal/native-axis1', 'axial/native-axis2']):
        kept = [i for i in range(3) if i != axis]
        plane = np.take(case.mri, voxel[axis], axis=axis).T
        spacing = np.asarray(case.spacing_mm)[kept]
        extent = [0, plane.shape[1] * spacing[0], 0, plane.shape[0] * spacing[1]]
        for row in range(3):
            ax = axes[row, axis]; ax.imshow(plane, cmap='gray', origin='lower', vmin=-10, vmax=100, extent=extent)
            if row in (0, 2):
                for layer, mask in case.compartments.items():
                    section = np.take(mask, voxel[axis], axis=axis).T
                    if section.any() and not section.all(): ax.contour(section, levels=[.5], colors=[colors[layer]], linewidths=.7, extent=extent)
            if row > 0:
                xy = voxel[kept] * spacing
                ax.set_xlim(xy[0]-25, xy[0]+25); ax.set_ylim(xy[1]-25, xy[1]+25)
            ax.set_title(f'{name} index={voxel[axis]} | ' + ['full source context', 'unmodified source zoom', 'source + declared geometry zoom'][row], fontsize=9)
            ax.set_xlabel('mm along native source axis'); ax.set_ylabel('mm along native source axis')
    fig.suptitle(case.case_id + '\nUnmodified acquired baseline CTA; yellow synthetic target; lime estimated envelope; cyan estimated dense skull; red supplied partial vessels\nNative source axis directions retained; no tumor appearance or clinical validation', fontsize=10)
    fig.tight_layout(rect=(0, 0, 1, .95)); fig.savefig(destination, dpi=130); plt.close(fig)
    return {'path': str(destination), 'sha256': sha256_file(destination), 'center_native_ijk': voxel.tolist(),
        'center_ras_mm': list(map(float, center)), 'source_window': [-10, 100], 'zoom_field_mm': 50.,
        'rendering': 'Actual native source volume and aligned masks, three orthogonal native planes; no resampling'}


def build(source, proposal, center, radii, case_version, *, root=None, seed=0, target_review=None,
          stage='intact-baseline', reviewed_task_domain=None, task_metadata_copy_of=None):
    root = checked_root(root, required_bytes=2*1024**3)
    if stage not in ('intact-baseline', 'conditional-post-scalp-exposure'): raise ValueError('Unknown explicit anatomy stage')
    source, proposal = Path(source).resolve(), Path(proposal).resolve()
    if not source.is_relative_to(root) or not proposal.is_relative_to(root): raise ValueError('All source inputs must be on guarded external root')
    if not re.fullmatch(r'[A-Za-z0-9_-]+', case_version): raise ValueError('Use a plain immutable version token')
    started = time.perf_counter(); original_sha = sha256_file(source)
    loaded_at = time.perf_counter(); host, _, workspace = load_workspace(source)
    source_load_seconds = time.perf_counter() - loaded_at
    if workspace is None: raise ValueError('Source must retain original-file workspace bindings')
    data = thaw_json(host.metadata).get('data_case', {})
    if data.get('role') not in ('TRAIN', 'DEVELOPMENT'): raise ValueError('Protected or unassigned source family')
    provenance_path = proposal / 'provenance.json'; provenance = json.loads(provenance_path.read_text())
    review_limitations = None
    if (proposal / 'review-limitations.json').is_file():
        review_limitations = json.loads((proposal / 'review-limitations.json').read_text())
        if review_limitations.get('provenance_sha256') != sha256_file(provenance_path):
            raise ValueError('Anatomy review limitations bind another proposal')
    if provenance['family_id'] != data['family_id']: raise ValueError('Proposal belongs to another source person')
    source_image_hash = provenance.get('source_image_sha256') or provenance.get('sources', [{}])[0].get('sha256')
    if source_image_hash not in {ref.sha256 for ref in host.source_refs}: raise ValueError('Proposal source image hash is not bound to native host')
    masks = {}; bindings = {}
    for name in ('brain_estimate', 'observed_skull', 'tumor_placement_support'):
        row = provenance['outputs'][name]; path = safe_path(root, row['path'])
        if sha256_file(path) != row['sha256']: raise ValueError('Estimate mask checksum differs: ' + name)
        image = nib.load(path)
        if image.shape != host.mri.shape or not np.array_equal(image.affine, host.affine):
            raise ValueError('Estimate physical frame differs; no resizing or affine substitution allowed')
        raw = np.asanyarray(image.dataobj)
        if not np.isin(raw, [0, 1]).all(): raise ValueError('Expected binary estimate')
        masks[name] = raw.astype(bool); bindings[name] = {'path': str(path.relative_to(root)), 'sha256': row['sha256']}
    target = ellipsoid(host.mri.shape, host.affine, center, radii)
    support_fraction = float(np.count_nonzero(target & masks['tumor_placement_support']) / target.sum())
    brain_fraction = float(np.count_nonzero(target & masks['brain_estimate']) / target.sum())
    review_binding = None
    if target_review:
        review_path = Path(target_review).resolve()
        if not review_path.is_relative_to(root): raise ValueError('Review outside guarded root')
        review = json.loads(review_path.read_text())
        expected = {'source_sha256': original_sha, 'proposal_sha256': sha256_file(provenance_path),
                    'center_ras_mm': list(map(float, center)), 'radii_mm': list(map(float, radii))}
        if any(review.get(k) != v for k, v in expected.items()) or review.get('decision') != 'accepted_for_geometry_target':
            raise ValueError('Exact source/proposal/physical-target review binding missing')
        if not review.get('reviewer') or not review.get('three_plane_evidence'):
            raise ValueError('Independent target reviewer and actual three-plane evidence required')
        review_binding = {'path': str(review_path.relative_to(root)), 'sha256': sha256_file(review_path), 'decision': review['decision']}
    if (support_fraction < 1. or brain_fraction < 1.) and review_binding is None:
        raise ValueError(f'Target requires actual independent tissue review: support={support_fraction:.6f}, brain={brain_fraction:.6f}')
    if np.any(target & masks['observed_skull']) or any(np.any(target & mask) for mask in host.compartments.values()):
        raise ValueError('Synthetic target intersects retained observed skull or supplied vascular anatomy')
    compartments = {'brain': masks['brain_estimate'], 'synthetic_target': target, 'estimated_skull': masks['observed_skull'], **host.compartments}
    if len(compartments) != len(host.compartments) + 3: raise ValueError('Source annotation name collision')
    case_id = data['family_id'].replace(':', '-') + '-CTA-INTEGRATED-' + case_version
    directory = root / 'integrated-v1/cases' / case_id
    if directory.exists(): raise ValueError('Immutable version exists; choose a new --case-version')
    metadata = thaw_json(host.metadata); layers = metadata.get('anatomy_layers', {})
    restrictions = list(provenance.get('limitations', [])) + [
        'Source CT intensities remain baseline background; no synthetic CT appearance or tumor-present acquisition.',
        'Brain estimate may include CSF; dense source-supported skull segmentation remains an estimate.',
        'Source crop faces and artificial estimate boundaries remain unknown; no implied opening or empty air.',
        'Supplied arterial and sinus labels are incomplete; unannotated background is not vessel-free.',
        'No cross-modality fusion, mass effect, clinical functional safety or clinical certification.']
    for name, kind, origin, coverage, color in [
        ('brain', 'brain', 'estimated', 'Estimated intracranial envelope; partial FOV and CSF uncertainty retained.', '#c799b0'),
        ('estimated_skull', 'skull', 'estimated', 'Source-CT dense skull estimate with context/model prior; artificial cutoff and crop contacts remain unknown.', '#e5d6b8'),
        ('synthetic_target', 'tumor', 'simulated', 'Geometry-only generated ellipsoid in source physical frame; not observed disease.', '#f78db3')]:
        layers[name] = {'kind': kind, 'provenance': origin, 'review_status': 'unreviewed', 'coverage': coverage, 'color': color,
            'uncertainty': restrictions, 'source_proposal_sha256': sha256_file(provenance_path)}
    for name in host.compartments:
        layers.setdefault(name, {})['kind'] = 'vessels'
    recipe = {'method': 'physical RAS-axis ellipsoid; no observed intensity modification', 'seed': int(seed),
              'center_ras_mm': list(map(float, center)), 'radii_mm': list(map(float, radii)),
              'target_mask_hash': array_digest(target), 'support_fraction': support_fraction,
              'brain_estimate_fraction': brain_fraction, 'independent_target_review': review_binding,
              'source_native_package_sha256': original_sha, 'proposal_sha256': sha256_file(provenance_path),
              'proposal_path': str(provenance_path.relative_to(root)), 'mask_bindings': bindings, 'clinical_validation': False}
    components = dict(data.get('components', {})); components.update(
        CTA={'provenance': 'acquired_image', 'original_intensities_preserved': True},
        brain={'provenance': 'estimated', 'coverage': layers['brain']['coverage']},
        skull={'provenance': 'estimated_from_CT', 'measured_CT_geometry_source': True, 'segmentation_ground_truth': False},
        tumor={'provenance': 'synthetic_geometry', 'appearance_synthesized': False})
    integrated = {**data, 'case_id': case_id, 'classification': 'synthetic_scenario_on_acquired_host',
        'case_type': 'synthetic', 'host_case_id': host.case_id, 'host_case_hash': host.semantic_hash,
        'accepted': False, 'status': 'pending_independent_review', 'components': components,
        'observations': {'primary_image': 'baseline_acquired_background_geometry_only'},
        'source_state': 'unmodified', 'new_real_patient_count': 0, 'synthetic_variant_count': 1,
        'recipe': recipe, 'appearance': None, 'deformation': None, 'mass_effect': False,
        'prior_inspection_package_capabilities': data.get('capabilities', []),
        'capabilities': ['source_CTA_inspection', 'supplied_partial_vascular_annotation_inspection', 'integrated_geometry_candidate_pending_review'],
        'simulation_admission': 'pending source-bound task and independent review',
        'missing_anatomy': ['complete_brain_coverage', 'complete_skull_coverage', 'complete_vasculature', 'functional_mapping', 'validated_tractography'],
        'limitations': restrictions, 'clinical_validation': False}
    metadata.update(data_case=integrated, anatomy_layers=layers, is_synthetic=True,
        selected_modality='CTA (BASELINE + SYNTHETIC GEOMETRY TARGET)',
        display_label='Acquired baseline CTA + synthetic geometry target', annotation_role='supplied_vascular_reference_and_explicit_synthetic_target',
        surgical_adapter={'ownershipPrecedence': list(compartments), 'sourceCaseHash': host.semantic_hash,
                          'sourcePackageSha256': original_sha, 'anatomyProposal': recipe,
                          'anatomyProposalRecord': provenance, 'independentAnatomyReviewLimitations': review_limitations,
                          'coverageLimitations': restrictions})
    if stage == 'conditional-post-scalp-exposure':
        assumptions = ['Superficial scalp exposure is assumed, not simulated or observed',
                       'Dura/complete venous coverage unavailable',
                       'Dura mechanics/opening are not modeled; any post-calvarial passage is conditional on prior/unmodeled dural opening']
        metadata['integrated_rehearsal'] = {'stage': 'conditional_post_scalp_exposure',
            'assumptions': assumptions, 'sourceImageRole': 'acquired_baseline_before_assumed_exposure',
            'intact_head_access_supported': False}
        metadata['selected_modality'] = 'CTA (BASELINE REFERENCE)'
        metadata['display_label'] = 'Baseline CTA before assumed scalp exposure; conditional mask scene'
        integrated['limits'] = restrictions + assumptions + ['Conditional post-scalp-exposure mask scene; intact-head access unsupported']
        integrated['limitations'] = integrated['limits']
        integrated['capabilities'] = ['baseline_CTA_inspection', 'conditional_post_scalp_exposure_geometry_candidate_pending_review']
        integrated['observations']['source_stage'] = 'acquired_baseline_before_assumed_exposure'
        integrated['simulation_admission'] = 'conditional engineering candidate; assumed superficial exposure, not intact-head operative sequence'
    domain_binding = None
    if reviewed_task_domain is not None:
        domain_path = Path(reviewed_task_domain).resolve()
        if not domain_path.is_relative_to(root): raise ValueError('Reviewed task-domain input outside guarded data root')
        domain = json.loads(domain_path.read_text())
        metadata['surgical_adapter']['reviewedTaskDomainV1'] = domain
        domain_binding = {'path': str(domain_path.relative_to(root)), 'sha256': sha256_file(domain_path)}
    copy_case = None; copy_binding = None
    if task_metadata_copy_of is not None:
        copy_path = Path(task_metadata_copy_of).resolve()
        if not copy_path.is_relative_to(root): raise ValueError('Metadata-copy source outside guarded root')
        copy_case, _, _ = load_workspace(copy_path)
        copy_binding = {'path': str(copy_path.relative_to(root)), 'sha256': sha256_file(copy_path),
                        'case_id': copy_case.case_id, 'case_hash': copy_case.semantic_hash}
        integrated.update(task_metadata_copy=True, task_copy_of=copy_binding,
                          synthetic_variant_count=0, new_synthetic_variant_count=0)
    refs = list(host.source_refs)
    for name, binding in bindings.items():
        refs.append(SourceRef('estimate-' + name, safe_path(root, binding['path']).as_uri(), binding['sha256'], native_frame='RAS+', provenance='estimated'))
    refs.append(SourceRef('synthetic-target', 'urn:ressectionlab:' + case_id, array_digest(target), native_frame='RAS+', provenance='simulated'))
    candidate = replace(host, case_id=case_id, compartments=compartments, source_compartments=compartments,
                        brain_mask=None, metadata=metadata, source_refs=tuple(refs))
    if copy_case is not None and (not np.array_equal(candidate.mri, copy_case.mri)
            or not np.array_equal(candidate.affine, copy_case.affine)
            or set(candidate.compartments) != set(copy_case.compartments)
            or any(not np.array_equal(mask, copy_case.compartments[name]) for name, mask in candidate.compartments.items())):
        raise ValueError('Task-metadata copy must preserve every source and target array exactly')
    if domain_binding is not None:
        from resectionlab.surgical_task_domain import reviewed_domain
        reviewed_domain(from_case(candidate))
    checked_root(root, required_bytes=1024**3); directory.mkdir(parents=True)
    destination = directory / (case_id + '.ressectionlab'); state = view_state(candidate, [])
    state['states']['primary']['cursor'] = list(map(float, center))
    state['states']['primary'].setdefault('viewer', {})['volumeRendering'] = {'enabled': False}
    save_at = time.perf_counter()
    save_workspace(candidate, destination, sources=[], imaging=state, episode=None, selection=None, artifacts={}, originals=workspace['originals'])
    save_seconds = time.perf_counter() - save_at; reopen_at = time.perf_counter()
    reopened, _, reopened_workspace = load_workspace(destination); scene = from_case(reopened)
    reopen_seconds = time.perf_counter() - reopen_at
    if reopened.semantic_hash != candidate.semantic_hash or not np.array_equal(reopened.mri, host.mri): raise ValueError('Production native roundtrip changed primary/source data')
    if [(x.name, x.sha256, x.role) for x in reopened_workspace['originals']] != [(x.name, x.sha256, x.role) for x in workspace['originals']]: raise ValueError('Embedded originals changed')
    if sha256_file(source) != original_sha: raise ValueError('Source native package changed')
    target_contact = target_contact_sheet(reopened, center, directory / 'target-three-plane-source-review.png')
    target_contact['path'] = str(Path(target_contact['path']).relative_to(root))
    atomic_json(directory / 'target-contact-metadata.json', target_contact)
    report = {'schema': 'integrated-native-geometry-case-v1', 'path': str(destination.relative_to(root)),
        'sha256': sha256_file(destination), 'case_hash': candidate.semantic_hash, 'case_id': case_id,
        'family_id': data['family_id'], 'role': data['role'], 'accepted': False, 'recipe': recipe,
        'source_case_id': host.case_id, 'source_native_sha256': original_sha, 'source_primary_array_hash': array_digest(host.mri),
        'primary_array_unchanged': True, 'native_frame_unchanged': True, 'original_files_preserved': len(workspace['originals']),
        'source_vascular_arrays_preserved': {name: array_digest(reopened.compartments[name]) for name in host.compartments},
        'ownership_precedence': list(compartments), 'simulator_regions': scene.regions,
        'overlaps': {name: int(np.count_nonzero(target & mask)) for name, mask in compartments.items() if name != 'synthetic_target'},
        'production_save_reopen': 'passed', 'prepare_seconds': time.perf_counter() - started,
        'source_load_seconds': source_load_seconds, 'save_seconds': save_seconds,
        'production_load_and_scene_seconds': reopen_seconds, 'bytes': destination.stat().st_size,
        'target_contact': target_contact, 'independent_anatomy_review_limitations': review_limitations,
        'generator_sha256': sha256_file(__file__), 'independent_review_required': True}
    report['stage'] = stage
    report['reviewed_task_domain_input'] = domain_binding
    report['task_metadata_copy_of'] = copy_binding
    report['new_synthetic_variant_count'] = 0 if copy_binding else 1
    atomic_json(directory / 'report.json', report); return report


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', required=True); parser.add_argument('--proposal', required=True)
    parser.add_argument('--center', type=float, nargs=3, required=True); parser.add_argument('--radii', type=float, nargs=3, required=True)
    parser.add_argument('--case-version', required=True); parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--root'); parser.add_argument('--target-review')
    parser.add_argument('--stage', choices=['intact-baseline', 'conditional-post-scalp-exposure'], default='intact-baseline')
    parser.add_argument('--reviewed-task-domain')
    parser.add_argument('--task-metadata-copy-of')
    args = parser.parse_args(); print(json.dumps(build(args.source, args.proposal, args.center, args.radii,
        args.case_version, root=args.root, seed=args.seed, target_review=args.target_review, stage=args.stage,
        reviewed_task_domain=args.reviewed_task_domain, task_metadata_copy_of=args.task_metadata_copy_of), indent=2))
