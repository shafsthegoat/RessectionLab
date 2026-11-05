#!/usr/bin/env python3
"""Known-source SSS corridor constraint and separate native refusal evidence.

The production encounter endpoint is geometric, never injury or clinical safety.
No source material is changed, and a native material refusal is not attributed
uniquely to vessels when other retained anatomy also intersects the instrument.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import re
import sys

import nibabel as nib
import numpy as np
from scipy.spatial.transform import Rotation

REPO = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(REPO/'src'), str(REPO/'scripts')]
from data_integrated_probe import envelope_admission, encode_segments
from resectionlab.core import array_digest
from resectionlab.data_root import checked_root, atomic_json, sha256_file, safe_path
from resectionlab.desktop_bridge import BridgeSession, _Request
from resectionlab.geometry import GeometryScene, capsule_voxel_indices
from resectionlab.surgical_motion import envelopes
from resectionlab.surgical_tools import TOOLS
from resectionlab.surgical_scene import from_case
from resectionlab.workspace_bundle import load_workspace


def output_path(root, name):
    if not isinstance(name, str) or not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_-]{0,127}', name):
        raise ValueError('Output name must be a plain immutable token')
    return safe_path(root, 'integrated-v1/probes/'+name)


def checked_input(root, value):
    root = Path(root).resolve(); path = Path(value)
    if path.is_absolute():
        path = path.resolve()
        if not path.is_relative_to(root): raise ValueError('Input outside guarded root')
        path = path.relative_to(root)
    return safe_path(root, path)


def code_snapshot(root):
    files = [Path(__file__), REPO/'scripts/data_integrated_probe.py',
        REPO/'src/resectionlab/surgical_anatomical_encounter.py', REPO/'src/resectionlab/surgical_motion.py',
        REPO/'src/resectionlab/surgical_task_domain.py', REPO/'src/resectionlab/surgical_bridge.py']
    bindings = []
    for source in files:
        source = source.resolve()
        digest = sha256_file(source)
        destination = safe_path(root, 'integrated-v1/recipes/'+digest+'-'+source.name)
        destination.parent.mkdir(parents=True, exist_ok=True)
        if not destination.exists():
            with destination.open('xb') as stream: stream.write(source.read_bytes())
        if sha256_file(destination) != digest: raise ValueError('Existing recipe snapshot differs')
        bindings.append({'source': str(source.relative_to(REPO)), 'sha256': digest,
                         'snapshot': str(destination.relative_to(root))})
    return bindings


def setup_invariants(before, after):
    # A mechanics setup adds a contactGeometry display descriptor. Compare the
    # actual physical fields, rather than mistaking that descriptor for motion.
    pose_keys = ('position', 'quaternion')
    state_keys = ('instrument', 'active', 'inserted', 'grip', 'parameters')
    same = lambda keys: (set(before['slots']) == set(after['slots']) and all(
        all(before['slots'][hand].get(key) == after['slots'][hand].get(key) for key in keys)
        for hand in before['slots']))
    return {'poses_unchanged': same(pose_keys), 'instrument_state_unchanged': same(state_keys),
        'removal_unchanged': before['removed'] == after['removed'],
        'deformation_unchanged': before.get('deformation') == after.get('deformation'),
        'physical_elapsed_time_unchanged': before['metrics']['elapsedSeconds'] == after['metrics']['elapsedSeconds'],
        'travel_unchanged': before['metrics']['travelMm'] == after['metrics']['travelMm']}


def prepare(source, *, root=None, name='006-SSS-corridor-contract-v1'):
    root = checked_root(root, required_bytes=256*1024**2)
    directory = output_path(root, name); source = checked_input(root, source)
    case, _, _ = load_workspace(source)
    if case.metadata.get('integrated_rehearsal', {}).get('stage') != 'conditional_post_scalp_exposure':
        raise ValueError('Explicit conditional stage required')
    originals = case.metadata['data_case']['source_files']
    row = next(x for x in originals if '/labelsTr_topbrain_v1_ct/' in x.get('source_member', ''))
    fine_path = safe_path(root, row['path'])
    if sha256_file(fine_path) != row['sha256']: raise ValueError('Fine annotation bytes changed')
    image = nib.load(fine_path)
    if image.shape != case.mri.shape or not np.array_equal(image.affine, case.affine): raise ValueError('Fine annotation frame differs')
    fine = np.asanyarray(image.dataobj); ijk = np.array([166, 301, 171])
    if case.metadata['data_case']['family_id'] != 'TopCoW:006' or fine[tuple(ijk)] != 40:
        raise ValueError('Exact independently identified source SSS point differs')
    point = case.voxel_to_world(ijk); d = np.array([1., 0., 0.])
    quaternion = Rotation.align_vectors([-d], [[0., 0., 1.]])[0].as_quat().tolist()
    params = {'radiusMm': 1.5, 'depthMm': 2., 'contactGeometryRevision': 2}
    cover = envelopes(TOOLS['cusa'], params, point, quaternion)
    if directory.exists(): raise ValueError('Preserve existing proposal')
    directory.mkdir(parents=True)
    context = directory/'supplied-SSS-class40-context.nii.gz'
    header = image.header.copy(); header.set_data_dtype(np.uint8); header.set_xyzt_units('mm')
    nib.save(nib.Nifti1Image((fine == 40).astype(np.uint8), image.affine, header), context)
    # Freeze the normative engineering exclusion before any encounter query.
    task = {'schema': 'source-SSS-zero-crossing-task-v1', 'source_base_path': str(source.relative_to(root)),
        'source_base_sha256': sha256_file(source), 'source_case_hash': case.semantic_hash,
        'known_source_label_value': 40, 'known_source_label_name': 'SSS', 'known_source_ijk': ijk.tolist(),
        'known_source_ras_mm': point.tolist(), 'source_fine_annotation': {'path': row['path'], 'sha256': row['sha256']},
        'context': {'path': str(context.relative_to(root)), 'sha256': sha256_file(context), 'mask_hash': array_digest(fine == 40)},
        'corridor_limit': {'marginMm': 0., 'maxNearMm3': 0.},
        'production_endpoint': 'resectionlab.surgical_anatomical_encounter.EncounterTracker',
        'candidate_action': {'slot': 'right', 'operation': 'place', 'duration': .1,
            'pose': {'position': point.tolist(), 'quaternion': quaternion}},
        'full_tool_envelopes': encode_segments(cover),
        'scope': 'Negative geometric corridor test through a supplied SSS annotation. Zero crossing is a declared engineering constraint, not clinical certification. No approach, placement, or reachability success is claimed.',
        'native_refusal_interpretation': 'Report all retained shaft and working materials. Do not attribute a generic native refusal uniquely to vessels.',
        'source_masks_modified': False, 'status': 'pending_independent_source_scope_review'}
    atomic_json(directory/'normative-task-proposal.json', task)
    scene = from_case(case); grid = GeometryScene(np.zeros(case.mri.shape, bool), case.affine)
    diagnostics = []
    for aa, bb, radius, part in cover:
        cells = capsule_voxel_indices(grid, aa, bb, radius); flat = np.ravel_multi_index(cells.T, case.mri.shape)
        diagnostics.append({'part': part, 'source_material_contacts': {region['name']:
            int(np.count_nonzero(scene.labels.ravel()[flat] == region['id'])) for region in scene.regions},
            'SSS_class40_contact_cells': int(np.count_nonzero((fine == 40).ravel()[flat]))})
    original = dict(case.metadata['surgical_adapter']['reviewedTaskDomainV1'])
    inverse = np.linalg.inv(case.affine); lo = np.asarray(original['minimumVoxelBoundary'])+.5
    hi = np.asarray(original['maximumVoxelBoundary'])-.5; margins = np.linalg.norm(inverse[:3, :3], axis=1)
    for aa, bb, radius, part in cover:
        endpoints = np.asarray([aa, bb])@inverse[:3, :3].T+inverse[:3, 3]
        lo = np.minimum(lo, np.floor(endpoints.min(0)-radius*margins-1))
        hi = np.maximum(hi, np.ceil(endpoints.max(0)+radius*margins+1))
    lo = np.maximum(lo.astype(int), 0); hi = np.minimum(hi.astype(int), np.array(case.mri.shape)-1)
    workspace = np.zeros(case.mri.shape, np.uint8); workspace[tuple(slice(a, b+1) for a, b in zip(lo, hi))] = 1
    mask_path = directory/'combined-workspace-proposal.nii.gz'; nib.save(nib.Nifti1Image(workspace, case.affine), mask_path)
    scope = envelope_admission(cover, case.affine, np.array(case.mri.shape), workspace)
    domain = {**original, 'minimumVoxelBoundary': (lo-.5).tolist(), 'maximumVoxelBoundary': (hi+.5).tolist(),
        'maskSha256': sha256_file(mask_path), 'reviewEvidenceSha256': '0'*64}
    atomic_json(directory/'domain-proposal.json', domain)
    _contact(case, fine == 40, point, d, directory/'source-SSS-full-tool-review.png')
    result = {'normative_task_sha256': sha256_file(directory/'normative-task-proposal.json'),
        'source_sha256': sha256_file(source), 'full_tool_FOV_and_scope': scope, 'per_part_contacts': diagnostics,
        'workspace': {'path': str(mask_path.relative_to(root)), 'sha256': sha256_file(mask_path)},
        'domain_proposal': domain, 'source_review_image': 'source-SSS-full-tool-review.png',
        'actions_executed': 0, 'encounter_metric_queried': False,
        'workspace_meaning': 'Proposed combined bounded workspace for two exact geometric tests. This box does not establish absent scalp, CSF, small vessels, or other unmodeled tissue.'}
    atomic_json(directory/'proposal-report.json', result); return result


def _contact(case, sss, point, direction, path):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from scipy.ndimage import map_coordinates
    along, across = np.linspace(-12, 52, 513), np.linspace(-24, 24, 385)
    aa, bb = np.meshgrid(along, across); side = np.cross(direction, [0., 0., 1.])
    world = point+aa[:, :, None]*direction+bb[:, :, None]*side
    coords = np.moveaxis(case.world_to_voxel(world), -1, 0)
    image = map_coordinates(case.mri, coords, order=1, mode='constant', cval=np.nan)
    extent = [along[0], along[-1], across[0], across[-1]]; fig, axes = plt.subplots(3, 1, figsize=(12, 12))
    for ax, limits in zip(axes, [(-1024, 1200), (-10, 100), (-10, 100)]):
        ax.imshow(image, cmap='gray', origin='lower', extent=extent, vmin=limits[0], vmax=limits[1])
        ax.add_patch(plt.Rectangle((0, -.5), 40, 1, fill=False, edgecolor='deepskyblue'))
        ax.add_patch(plt.Circle((0, 0), 1.5, fill=False, edgecolor='deepskyblue'))
        ax.set_xlabel('mm along the source-supported lateral candidate; supplied SSS point at 0'); ax.set_ylabel('mm across native axial source plane')
    for label, mask, color in [('brain', case.compartments['brain'], 'lime'), ('skull', case.compartments['estimated_skull'], 'cyan'), ('SSS40', sss, 'red')]:
        plane = map_coordinates(mask.astype(np.uint8), coords, order=0, mode='constant', cval=0)
        if plane.any() and not plane.all(): axes[2].contour(plane, levels=[.5], colors=[color], linewidths=.8, extent=extent)
    fig.suptitle('CTA006: negative source-SSS corridor proposal, not an admitted placement\nActual baseline CTA; blue unchanged CUSA 40 mm shaft / 0.5 mm shaft radius / 1.5 mm tip\nRed supplied fine-label-40 SSS; green estimated cap; cyan retained skull. Other anatomy also obstructs this candidate.', fontsize=10)
    fig.tight_layout(rect=(0, 0, 1, .93)); fig.savefig(path, dpi=120); plt.close(fig)


def execute(source, task_path, decision_path, *, root=None, name='006-SSS-corridor-execution-v1', setup_addendum=None):
    from resectionlab.surgical_anatomical_encounter import AnatomicalObservation, EncounterTracker, axial_envelope
    from resectionlab.surgical_engine import SurgicalEngine
    from resectionlab.surgical_environment import standardize
    root = checked_root(root, required_bytes=1024**3)
    directory = output_path(root, name)
    source, task_path, decision_path = [checked_input(root, value) for value in (source, task_path, decision_path)]
    task = json.loads(task_path.read_text()); decision = json.loads(decision_path.read_text())
    addendum_binding = None
    if setup_addendum is not None:
        addendum_path = checked_input(root, setup_addendum)
        json.loads(addendum_path.read_text())
        addendum_binding = {'path': str(addendum_path.relative_to(root)), 'sha256': sha256_file(addendum_path)}
    if decision.get('normative_task_sha256') != sha256_file(task_path): raise ValueError('Review binds another normative task')
    if decision.get('execution_source_sha256') != sha256_file(source): raise ValueError('Review binds another execution source')
    if decision.get('decision') != 'accepted_for_negative_SSS_corridor_execution': raise ValueError('Negative corridor execution not independently approved')
    case, source_artifacts, source_workspace = load_workspace(source)
    base_path = safe_path(root, task['source_base_path'])
    if sha256_file(base_path) != task['source_base_sha256']: raise ValueError('Frozen base source bytes differ')
    base, _, _ = load_workspace(base_path)
    if (not np.array_equal(case.mri, base.mri) or not np.array_equal(case.affine, base.affine)
            or set(case.compartments) != set(base.compartments)
            or any(not np.array_equal(mask, base.compartments[name]) for name, mask in case.compartments.items())):
        raise ValueError('Execution case differs from the frozen source anatomical arrays')
    context_path = safe_path(root, task['context']['path'])
    if sha256_file(context_path) != task['context']['sha256']: raise ValueError('Source context changed')
    image = nib.load(context_path)
    if image.shape != case.mri.shape or not np.array_equal(image.affine, case.affine): raise ValueError('Context source frame differs')
    if directory.exists(): raise ValueError('Preserve previous execution')
    bindings = code_snapshot(root)
    directory.mkdir(parents=True); session = BridgeSession(directory/'transfers', run_dir=directory/'runs')
    def call(operation, args): return session.execute(operation, args, _Request(operation), lambda *_: None)
    try:
        identity = call('loadCase', {'path': str(source)})['caseHash']
        has_history = 'surgical_toolbox' in source_artifacts
        current = call('surgicalRehearsal', {'caseHash': identity, 'command': 'inspect' if has_history else 'create'})
        if has_history:
            # Preserve the complete primary branch and its burdens. This new
            # negative test starts on a separate native baseline branch.
            current = call('surgicalRehearsal', {'caseHash': identity, 'command': 'branch',
                'expectedStateId': current['view']['stateId'], 'position': 0})
        setup_before = current['view']
        setup_action = {'slot': 'right', 'operation': 'configure', 'mechanicsRevision': 2, 'clockRevision': 1, 'duration': .1}
        current = call('surgicalRehearsal', {'caseHash': identity, 'command': 'step', 'expectedStateId': setup_before['stateId'],
            'action': setup_action})
        setup_after = current['view']
        setup_checks = setup_invariants(setup_before, setup_after)
        setup_record = {'kind': 'nonphysical_model_setup', 'action': setup_action, 'outcome': current.get('outcome'),
            'state_before': setup_before['stateId'], 'state_after': setup_after['stateId'],
            'checks': setup_checks, 'review_addendum': addendum_binding}
        atomic_json(directory/'model-setup.json', setup_record)
        if not all(setup_checks.values()):
            call('saveCase', {'caseHash': identity, 'path': str(directory/'setup-only-failure.ressectionlab')})
            raise ValueError('Model setup changed physical state; negative query withheld')
        before = current['view']['stateId']
        preview = call('surgicalRehearsal', {'caseHash': identity, 'command': 'preview', 'expectedStateId': before, 'action': task['candidate_action']})
        saved = directory/'negative-preview.ressectionlab'; call('saveCase', {'caseHash': identity, 'path': str(saved)})
        reopened, artifacts, saved_workspace = load_workspace(saved); record = artifacts['surgical_toolbox']
        engine = SurgicalEngine.restore(standardize(from_case(reopened), physics_profile=record['physicsProfile']), record['branches'][record['branchId']], use_cache=False)
        observation = AnatomicalObservation(np.asanyarray(image.dataobj)>0, image.affine, 'reviewed')
        tracker = EncounterTracker(observation, engine, task['corridor_limit'])
        pose = task['candidate_action']['pose']
        cover = axial_envelope(pose, pose, 'cusa', engine.parameters(engine.state['slots']['right']))
        tracker.add_cover(engine, cover); receipt = tracker.receipt(engine)
        result = {'schema': 'production-known-SSS-corridor-negative-v1', 'source_sha256': sha256_file(source),
            'normative_task_sha256': sha256_file(task_path), 'independent_decision_sha256': sha256_file(decision_path),
            'native_public_preview': preview, 'preview_preserved_state': engine.state_id == before,
            'source_case_preserved': reopened.semantic_hash == case.semantic_hash,
            'source_labels_preserved': array_digest(from_case(case).labels) == array_digest(from_case(reopened).labels),
            'nonphysical_model_setup': setup_record,
            'original_files_preserved': [(x.sha256, x.role) for x in source_workspace['originals']] ==
                [(x.sha256, x.role) for x in saved_workspace['originals']],
            'existing_primary_branches_preserved': (not has_history or
                all(key in record['branches'] and record['branches'][key] == value
                    for key, value in source_artifacts['surgical_toolbox']['branches'].items())),
            'saved': str(saved), 'saved_sha256': sha256_file(saved),
            'production_endpoint': 'resectionlab.surgical_anatomical_encounter.EncounterTracker',
            'source_specific_vascular_corridor_receipt': receipt,
            'execution_code_bindings': bindings,
            'interpretation': 'The frozen zero-crossing source-SSS corridor limit is evaluated independently of the native material refusal. Additional retained brain/skull contacts are not hidden; no uniquely vascular first-refusal claim.',
            'source_masks_modified': False, 'target_reached': False, 'clinical_validation': False}
        atomic_json(directory/'report.json', result); return result
    finally:
        session.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__); parser.add_argument('--source', required=True)
    parser.add_argument('--root'); parser.add_argument('--name', required=True)
    parser.add_argument('--task'); parser.add_argument('--decision')
    parser.add_argument('--setup-addendum')
    args = parser.parse_args()
    if args.task or args.decision:
        if not args.task or not args.decision: parser.error('--task and --decision must be paired')
        result = execute(args.source, args.task, args.decision, root=args.root, name=args.name, setup_addendum=args.setup_addendum)
    else: result = prepare(args.source, root=args.root, name=args.name)
    print(json.dumps(result, indent=2))
