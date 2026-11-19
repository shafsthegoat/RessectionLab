#!/usr/bin/env python3
"""Build one route-conditioned IXI077 organic scenario and screen its stock route."""
from __future__ import annotations

import argparse
from dataclasses import replace
import json
from pathlib import Path
from tempfile import TemporaryDirectory

import nibabel as nib
import numpy as np
from scipy.ndimage import gaussian_filter
from scipy.spatial.transform import Rotation

from build_cta_organic_gliomas import OBSERVATION, TRUTH, checked_mask, register, write_truth
from data_integrated_healthy import capsule_native_fov_margin, fixed_translation_route
from data_integrated_hosts import UUID, binding, checked_file
from data_integrated_ixi077_trial import BINDINGS, NATIVE, PROPOSAL, SWEEP
from data_integrated_ixi_preflight import minimum_tool_gap
from probe_cta_organic_rehearsal import cells, tool_segments
from resectionlab.core import SourceRef, array_digest, thaw_json
from resectionlab.data_root import atomic_json, checked_root, safe_path, sha256_file
from resectionlab.organic_tumors import organic_geometry
from resectionlab.surgical_engine import SurgicalEngine
from resectionlab.surgical_environment import default_target_names, standardize
from resectionlab.surgical_motion import envelopes, swept_envelopes
from resectionlab.surgical_planning_admission import planning_admission
from resectionlab.surgical_rl_protection import GuardConfig
from resectionlab.surgical_scene import from_case
from resectionlab.surgical_tools import TOOLS
from resectionlab.workspace_bundle import load_workspace, save_workspace, view_state
from resectionlab.workspace_imaging import DisplaySource, source_registration

CASE_ID = 'IXI-077-T1-MRA-ORGANIC-ROUTE-01-v1'
COLLECTION = 'prepared/ixi077-organic-rehearsal-v1'
VASCULAR = 'registered_partial_vessels'
RIGID = 'integrated-v1/healthy/IXI077/rigid-v1'
SEED = 2026908501
RADII = (3.2, 3.6, 4.1)
SIGMA_MM = .8
CONTOUR_THRESHOLD = .54
LIMITS = (
    'One route-conditioned DEVELOPMENT fixture on unchanged acquired IXI077 T1 and MRA.',
    'Scalp exposure, a local skull opening, and dural opening are assumed. Bone, scalp, and dura mechanics are unavailable.',
    'The registered partial vessel reference has unresolved alignment uncertainty. Unlabelled vasculature remains unknown.',
    'The displayed preoperative contour is simulated. Exact generated core, rim, and edema are separate research files.',
    'The fixed stock line is a read-only engineering preflight. Fresh execution and clinical validation are unavailable.',
)


def read_json(path):
    return json.loads(path.read_text())


def load_inputs(root):
    records = {path: {'path': str(path), 'sha256': BINDINGS[path]} for path in (PROPOSAL, NATIVE, SWEEP)}
    old_proposal, old_native, fixed_route = (checked_file(root, records[path]) for path in (PROPOSAL, NATIVE, SWEEP))
    proposal, route = read_json(old_proposal), read_json(fixed_route)
    if (proposal['person_id'] != 'IXI:077' or proposal['role'] != 'TRAIN'
            or route['status'] != 'necessary_conditional_label_geometry_only_NOT_STATIC_OR_EXECUTION_ADMISSION'
            or route['actual_tool_sessions_or_actions'] != 0):
        raise ValueError('Expected frozen same-person source and read-only route geometry')
    host_path = checked_file(root, {'path': proposal['source_native_path'], 'sha256': proposal['source_native_sha256']})
    host, artifacts, host_workspace = load_workspace(host_path)
    anatomy, native_artifacts, anatomical_workspace = load_workspace(old_native)
    if (host.frame != 'RAS+' or host.case_id != 'IXI-077-control-native-v5' or host.brain_mask is None
            or host.metadata['data_case']['person_id'] != 'IXI:077'
            or host.metadata['data_case']['role'] != 'TRAIN' or artifacts or native_artifacts
            or host_workspace is None or anatomical_workspace is None
            or host_workspace['episode'] is not None or anatomical_workspace['episode'] is not None
            or not np.array_equal(host.mri, anatomy.mri) or not np.array_equal(host.affine, anatomy.affine)
            or not np.array_equal(host.brain_mask, anatomy.compartments['brain'])):
        raise ValueError('Exact target-free IXI077 source and unchanged reference anatomy required')
    registration = thaw_json(anatomy.metadata['workspace_registration'])
    registration_methods = {'path': RIGID + '/registration.json', 'sha256': proposal['registration_sha256']}
    methods = read_json(checked_file(root, registration_methods))
    checked_file(root, {'path': registration['evidence_path'], 'sha256': registration['evidence_sha256']})
    t1 = nib.load(checked_file(root, methods['fixed_T1']))
    checked_file(root, methods['moving_MRA'])
    if not np.array_equal(t1.affine, host.affine) or not np.array_equal(t1.get_fdata(dtype=np.float32), host.mri):
        raise ValueError('Original T1 disagrees with the preserved native source')
    mask_bindings = {
        'support': {'path': proposal['support_path'], 'sha256': proposal['support_sha256']},
        'mra_coverage': {'path': RIGID + '/MRA-observed-coverage.nii.gz', 'sha256': proposal['coverage_sha256']},
        'vessel': {'path': RIGID + '/registered-vascular-reference.nii.gz', 'sha256': proposal['registered_vessel_sha256']},
    }
    masks = {key: checked_mask(root, row, host) for key, row in mask_bindings.items()}
    if (not np.array_equal(masks['vessel'], anatomy.compartments[VASCULAR])
            or np.any(masks['support'] & (~host.brain_mask | ~masks['mra_coverage'] | masks['vessel']))):
        raise ValueError('Verified tissue support, brain or vessel memberships differ')
    # The dossier of anatomy-only acquisition and registration records may be
    # reused. The old synthetic proposal, replay, and task authority stay out.
    expected = {item.sha256 for item in host_workspace['originals']} | {
        registration['original_annotation_sha256'], registration['source_sha256_by_modality']['MRA'],
        mask_bindings['vessel']['sha256'], mask_bindings['mra_coverage']['sha256'], registration_methods['sha256'],
    }
    originals = [item for item in anatomical_workspace['originals'] if item.sha256 in expected]
    if {item.sha256 for item in originals} != expected:
        raise ValueError('Exact acquisition, estimate and registration originals required')
    evidence = {'host': binding(root, host_path), 'reference_anatomy': records[NATIVE],
                'strict_support_origin': records[PROPOSAL], 'fixed_line_geometry': records[SWEEP],
                'registration': registration_methods, 'registration_review': {
                    'path': registration['evidence_path'], 'sha256': registration['evidence_sha256']},
                'mask_bindings': mask_bindings}
    return host, anatomy, anatomical_workspace['sources'], originals, masks, evidence, registration, proposal, route


def generate(host, support, proposal):
    # The source-reviewed fixed route endpoint chooses the center, not a saved
    # removal, exact old sphere, clinical label or outcome-selected search.
    endpoint = np.asarray(proposal['prospective_readonly_route_phase']['fixed_route']['T_ras_mm'])
    center = np.rint(host.world_to_voxel(endpoint)).astype(int)
    if not np.allclose(host.voxel_to_world(center), endpoint, rtol=0., atol=1e-8):
        raise ValueError('Declared generation center must equal the fixed geometry endpoint')
    lo, hi = center - 16, center + 17
    if np.any(lo < 0) or np.any(hi > host.mri.shape):
        raise ValueError('Native placement crop touches unsupported image bounds')
    region = tuple(slice(int(a), int(b)) for a, b in zip(lo, hi))
    affine = host.affine.copy()
    affine[:3, 3] = host.voxel_to_world(lo)
    domain = support[region]
    masks, recipe = organic_geometry(domain, affine, center_voxel=center - lo, seed=SEED,
                                     radii_mm=RADII, profile='lobulated')
    truth = masks['synthetic_core'] | masks['synthetic_rim']
    observation_local = (gaussian_filter(truth.astype(np.float64), SIGMA_MM / np.asarray(host.spacing_mm))
                         >= CONTOUR_THRESHOLD) & domain
    if (not observation_local.any() or np.array_equal(truth, observation_local)
            or any(np.any(mask & ~domain) for mask in masks.values())):
        raise ValueError('Generated exact compartments or simulated observation failed isolation or support')
    observation = np.zeros(host.mri.shape, bool)
    observation[region] = observation_local
    recipe.update(center_full_native_voxel=center.tolist(), local_crop_origin_native_voxel=lo.tolist(),
                  conditioning='fixed source corridor endpoint in immutable strict IXI077 support',
                  observation_sigma_mm=SIGMA_MM, observation_threshold=CONTOUR_THRESHOLD,
                  mass_effect=False, generated_image_signal=False)
    return masks, region, observation, recipe


def make_case(host, anatomy, sources, vessels, observation, registration, evidence):
    metadata = thaw_json(host.metadata)
    data = metadata['data_case']
    data.update(case_id=CASE_ID, person_id='IXI:077', family_id='IXI:077', role='DEVELOPMENT',
                source_family_role=data['role'], case_type='synthetic', accepted=False,
                classification='synthetic_scenario_on_acquired_healthy_host',
                native_source=evidence['host'], new_real_patient_count=0, synthetic_variant_count=1,
                capabilities=['native_source_inspection', 'registration_overlay_inspection', 'simulated_contour_inspection'],
                developer_evidence='route-conditioned geometry and fresh static cover',
                observations={'primary_image': 'baseline_acquired_background_geometry_only',
                              'T1': 'acquired_unmodified_baseline', 'MRA': 'native_and_reviewed_approximate_rigid_derivative',
                              'target': 'simulated_partial_volume_contour'},
                allowed_planner_observations=['structural_T1', 'estimated_brain_mask', 'registered_partial_vessels', OBSERVATION],
                private_reference_fields=[], simulation_admission='geometry inspection; instrument execution unreviewed',
                missing_anatomy=['measured_skull', 'complete_vasculature', 'refined_tissue_or_CSF_labels',
                                 'tractography', 'functional_mapping'],
                clinical_validation=False, limitations=list(LIMITS), deformation=None)
    data['components'].update(vessels='reviewed_approximate_rigid_overlay_of_partial_model_assisted_source_reference',
                              tumor='approximate_simulated_preoperative_contour')
    metadata.update(is_synthetic=True, selected_modality='T1', split='development', training_admitted=False,
                    display_label='IXI077 acquired T1 and MRA with simulated organic tumor contour',
                    observation_role='baseline_reference_before_synthetic_contour',
                    simulation_admission=data['simulation_admission'])
    registration = {**registration, 'patient_id': CASE_ID}
    metadata['workspace_registration'] = registration
    metadata['integrated_rehearsal'] = {'stage': 'conditional_exposed_brain_geometry',
        'assumptions': ['Scalp exposure, a local skull opening and dural opening are assumed and unsimulated.'],
        'sourceImageRole': 'acquired_baseline_before_assumed_exposure', 'intact_head_access_supported': False}
    layer_metadata = thaw_json(anatomy.metadata['anatomy_layers'])
    metadata['anatomy_layers'] = {name: layer_metadata[name] for name in ('brain', VASCULAR)}
    metadata['anatomy_layers'][OBSERVATION] = {
        'kind': 'tumor', 'label': 'Simulated organic tumor contour', 'provenance': 'simulated',
        'color': '#f18fbd', 'coverage': LIMITS[3], 'review_status': 'engineering inspection only'}
    masks = {'brain': host.brain_mask, VASCULAR: vessels, OBSERVATION: observation}
    metadata['surgical_adapter'] = {'ownershipPrecedence': ['brain', OBSERVATION, VASCULAR],
        'coverageLimitations': list(LIMITS), 'sourceCaseHash': host.semantic_hash}
    vessel_ref = next(ref for ref in anatomy.source_refs if ref.source_id == 'registered-partial-vessels')
    contour_ref = SourceRef(OBSERVATION, f'synthetic://{CASE_ID}/contour', array_digest(observation),
                            native_frame=host.frame, provenance='simulated')
    candidate = replace(host, case_id=CASE_ID, compartments=masks, source_compartments=masks, brain_mask=None,
                        metadata=metadata, source_refs=(*host.source_refs, vessel_ref, contour_ref),
                        unknowns=(*host.unknowns, *LIMITS))
    display = []
    statuses = []
    for source in sources:
        item = thaw_json(source.case.metadata)
        # Keep image identity and only reconstruct association metadata. Exact
        # old target geometry, case role, and authorization are never copied.
        identity = {'person_id': 'IXI:077', 'deformation': None,
                    'classification': item['data_case']['classification'],
                    'source': 'ixi', 'license': 'CC BY-SA 3.0'}
        auxiliary_metadata = {'data_case': identity, 'is_synthetic': False,
                              'selected_modality': 'MRA', 'display_label': item['display_label']}
        if 'workspace_registration' in item:
            auxiliary_metadata['workspace_registration'] = registration
        auxiliary = replace(source.case, case_id=CASE_ID, metadata=auxiliary_metadata)
        display.append(DisplaySource(auxiliary, source.modality, source.annotation_kind))
        association, reviewed = source_registration(candidate, auxiliary)
        statuses.append((association, reviewed))
    if [status['overlayPermitted'] for _, status in statuses] != [True, False]:
        raise ValueError('Reviewed derived MRA and original native MRA must retain distinct registration status')
    return candidate, display


def preflight(case, route, proposal, coverage):
    """Re-query unchanged stock geometry on this unacted, exact native case."""
    scene = standardize(from_case(case), default_target_names(case))
    engine = SurgicalEngine(scene, guard_config=GuardConfig(require_critical_coverage=False))
    initial = engine.state_id
    phases = proposal['prospective_readonly_route_phase']['fixed_route']
    start, goal = (np.asarray(phases[key]) for key in ('A_ras_mm', 'T_ras_mm'))
    quaternion = phases['orientation_xyzw']
    inward = Rotation.from_quat(quaternion).apply([0., 0., 1.])
    points = fixed_translation_route(start, goal)
    if (not np.array_equal(points, route['fixed_points'])
            or not np.allclose(inward, (goal-start)/np.linalg.norm(goal-start), rtol=0., atol=1e-12)):
        raise ValueError('Fixed source line, interval size, or original stock attitude differs')
    cusa = TOOLS['cusa']
    slot = engine.state['slots']['left']
    stationary = envelopes(TOOLS[slot['instrument']], engine.parameters(slot), slot['position'], slot['quaternion'])
    parameters = {**engine.parameters(engine.state['slots']['right']), 'contactGeometryRevision': 2}
    if (parameters['radiusMm'] != 1.5 or cusa['shaftRadiusMm'] != .5 or parameters['depthMm'] != 2.
            or scene.source['caseHash'] != case.semantic_hash or engine.actions):
        raise ValueError('Expected a fresh stock-CUSA configuration')
    grid = engine.grid
    layers = {name: mask.ravel() for name, mask in case.compartments.items()}
    brain, vessels, target = (layers[name] for name in ('brain', VASCULAR, OBSERVATION))
    coverage = coverage.ravel()
    widened = (start - 40.*inward, start, parameters['radiusMm'], 'stock-widened-placement')
    pose = lambda point: {'position': list(map(float, point)), 'quaternion': quaternion}
    placed = cells(grid, [widened])
    fov = lambda part: capsule_native_fov_margin(grid.shape, grid.affine, *part[:3])
    placement_margin = fov(widened)
    place_overlap = int(np.count_nonzero(brain[placed] | vessels[placed] | target[placed]))
    tool_gap = minimum_tool_gap(tool_segments('cusa', start, quaternion), stationary)
    working_total, shaft_total, early_target = set(), set(), set()
    vessel_ids, missing_slab = set(), set()
    intervals = []
    min_margin = float('inf')
    for index, (a, b) in enumerate(zip(points, points[1:])):
        envelope = swept_envelopes(cusa, parameters, pose(a), pose(b))
        margin = min(map(fov, envelope))
        min_margin = min(min_margin, margin)
        tool_gap = min(tool_gap, minimum_tool_gap(envelope, stationary))
        query = {part: cells(grid, envelope, part) for part in ('shaft', 'working')}
        work = set(map(int, query['working']))
        shaft = set(map(int, query['shaft']))
        contact = np.union1d(query['working'], query['shaft'])
        shaft_target = set(map(int, query['shaft'][target[query['shaft']]]))
        early_target.update(shaft_target - working_total - work)
        working_total.update(work)
        shaft_total.update(shaft)
        vessel_ids.update(map(int, contact[vessels[contact]]))
        missing_slab.update(map(int, contact[(brain[contact] | target[contact]) & ~coverage[contact]]))
        intervals.append({'index': index, 'start_ras_mm': a.tolist(), 'end_ras_mm': b.tolist(),
                          'native_fov_minimum_margin_mm': margin,
                          'working_target_ids': [v for v in sorted(work) if target[v]],
                          'working_brain_ids': [v for v in sorted(work) if brain[v] and not target[v]],
                          'shaft_retained_brain_or_target_ids': [v for v in sorted(shaft) if brain[v] or target[v]]})
    observed = set(map(int, np.flatnonzero(target)))
    visible_contact = len(observed & working_total)
    checks = {'widened_placement_inside_native_fov': placement_margin > 0.,
              'widened_placement_disjoint_from_modeled_anatomy': place_overlap == 0,
              'full_moving_envelope_inside_native_fov': min_margin > 0.,
              'complete_parked_tool_separation': tool_gap > 0.,
              'zero_known_supplied_or_registered_vessel_contacts': not vessel_ids,
              'modeled_intracranial_contacts_inside_mra_coverage': not missing_slab,
              'positive_geometric_observable_contour_contact': visible_contact > 0,
              'no_forecast_contour_shaft_contacts_before_working_cover': not early_target}
    checks = {name: bool(value) for name, value in checks.items()}
    if (engine.state_id != initial or engine.actions or engine.history or engine.state['removed']
            or engine.state['elapsedSeconds'] != 0.):
        raise ValueError('Read-only preflight changed modeled state')
    return {'schema': 'resectionlab.ixi077-organic-fixed-route-static-v1',
            'case_id': case.case_id, 'source_case_hash': case.semantic_hash,
            'source_scene_id': scene.identity, 'zero_action_state_id': initial,
            'tool_mechanics_revision_proposed': 2, 'stock_parameters': parameters,
            'status': 'necessary_static_checks_pass_pending_fresh_execution_review' if all(checks.values())
                      else 'static_preflight_refused',
            'checks': checks, 'source_native_affine_ras_mm': case.affine.tolist(),
            'route': {'start_ras_mm': start.tolist(), 'goal_ras_mm': goal.tolist(), 'orientation_xyzw': quaternion,
                      'fixed_line_length_mm': float(np.linalg.norm(goal-start)), 'interval_count': len(intervals),
                      'reverse_policy': 'same analytical line; committed poses and safe return remain unverified'},
            'widened_placement_fov_margin_mm': placement_margin, 'moving_minimum_fov_margin_mm': min_margin,
            'stationary_other_tool_minimum_gap_mm': tool_gap, 'placement_occupied_voxels': place_overlap,
            'vessel_contact_cell_ids': sorted(vessel_ids), 'intracranial_outside_mra_cell_ids': sorted(missing_slab),
            'visible_contour_total_cells': len(observed), 'visible_contour_working_contact_cells': visible_contact,
            'visible_contour_shaft_contact_cells': len(observed & shaft_total),
            'visible_contour_cells_outside_single_working_cover': len(observed - working_total),
            'visible_contour_cells_with_possible_shaft_before_working': sorted(early_target),
            'intervals': intervals, 'instrument_admitted': False, 'actions_executed': 0,
            'automatic_planning': planning_admission(scene, engine.protection), 'clinical_validation': False}


def build(root, libraries):
    root = checked_root(root, expected_uuid=UUID, required_bytes=1024**3)
    directory = safe_path(root, COLLECTION)
    final = directory / CASE_ID
    if final.exists():
        receipt = read_json(final / 'receipt.json')
        for key in ('native_case', 'research_truth', 'static_preflight'):
            checked_file(root, receipt[key])
        register(root, [receipt], libraries)
        return receipt
    host, anatomical, display, originals, masks, evidence, registration, proposal, route = load_inputs(root)
    truth, region, observation, recipe = generate(host, masks['support'], proposal)
    if np.array_equal(observation, anatomical.compartments['synthetic_target']):
        raise ValueError('Fresh observable contour unexpectedly duplicates the old sphere')
    if np.any(observation & (masks['vessel'] | ~host.brain_mask | ~masks['mra_coverage'])):
        raise ValueError('Visible contour overlaps unsupported or protected memberships')
    case, sources = make_case(host, anatomical, display, masks['vessel'], observation, registration, evidence)
    directory.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(prefix='.ixi077-organic-', dir=directory) as temporary:
        stage = Path(temporary) / CASE_ID
        stage.mkdir()
        native = stage / (CASE_ID + '.ressectionlab')
        truth_path = stage / 'research-truth.nii.gz'
        proof = stage / 'stock-route-preflight.json'
        state = view_state(case, sources)
        state['states']['primary']['cursor'] = recipe['center_ras_mm']
        state['states']['primary'].setdefault('viewer', {})['volumeRendering'] = {'enabled': False}
        session_hash = save_workspace(case, native, sources=sources, imaging=state, episode=None, selection=None,
                                      artifacts={}, originals=originals)
        reopened, artifacts, workspace = load_workspace(native)
        if (reopened.semantic_hash != case.semantic_hash or artifacts or workspace is None
                or workspace['episode'] is not None or workspace['originals'] != originals
                or set(reopened.compartments) != {'brain', OBSERVATION, VASCULAR}
                or set(reopened.source_compartments) != {'brain', OBSERVATION, VASCULAR}
                or not np.array_equal(reopened.mri, host.mri) or not np.array_equal(reopened.affine, host.affine)
                or not np.array_equal(reopened.compartments['brain'], host.brain_mask)
                or not np.array_equal(reopened.compartments[VASCULAR], masks['vessel'])
                or [item.case.semantic_hash for item in workspace['sources']] != [item.case.semantic_hash for item in sources]):
            raise ValueError('Native reopen failed acquisition, anatomy, registration or observation isolation')
        write_truth(truth_path, truth, region, host)
        route_report = preflight(reopened, route, proposal, masks['mra_coverage'])
        atomic_json(proof, route_report)

        def staged_record(path):
            return {'path': str((final / path.name).relative_to(root)),
                    'sha256': sha256_file(path), 'bytes': path.stat().st_size}

        visible = int(observation.sum())
        receipt = {'schema': 'resectionlab.ixi077-organic-route-scenario-v1', 'case_id': case.case_id,
                   'case_hash': case.semantic_hash, 'session_hash': session_hash,
                   'person_id': 'IXI:077', 'source_family_role': 'TRAIN', 'role': 'DEVELOPMENT',
                   'new_real_patient_count': 0, 'cta_cohort_membership': False, 'source': evidence,
                   'native_case': staged_record(native), 'static_preflight': staged_record(proof),
                   'research_truth': {**staged_record(truth_path), 'label_codes': TRUTH, 'planning_eligible': False},
                   'source_volume_hash': array_digest(host.mri),
                   'source_brain_hash': array_digest(host.brain_mask), 'source_vessel_hash': array_digest(masks['vessel']),
                   'imaging_series': ['T1 acquired native', 'MRA reviewed derivative', 'MRA acquired native'],
                   'source_originals': [item.manifest(index) for index, item in enumerate(originals)],
                   'case_reopen': True, 'original_imaging_and_host_masks_unchanged': True,
                   'observation': {'name': OBSERVATION, 'cells': visible, 'volume_mm3': visible*host.voxel_volume_mm3,
                                   'sha256': array_digest(observation), 'generated_image_signal': False,
                                   'native_observation_differs_from_old_sphere_voxels':
                                       int(np.count_nonzero(observation ^ anatomical.compartments['synthetic_target']))},
                   'recipe': recipe, 'instrument_admitted': False, 'actions_executed': 0,
                   'static_status': route_report['status'], 'code_sha256': sha256_file(__file__),
                   'limits': list(LIMITS)}
        atomic_json(stage / 'receipt.json', receipt)
        stage.rename(final)
    register(root, [receipt], libraries)
    return receipt


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path)
    parser.add_argument('--register', action='append', type=Path, default=[])
    args = parser.parse_args()
    result = build(args.root, args.register)
    print(json.dumps({key: result[key] for key in ('case_id', 'case_hash', 'native_case', 'static_preflight',
                                                   'research_truth', 'observation', 'static_status')}, indent=2))


if __name__ == '__main__':
    main()
