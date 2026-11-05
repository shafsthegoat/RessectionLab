#!/usr/bin/env python3
"""Check the frozen IXI077 route against the production zero-action bridge."""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import tempfile
import time

import nibabel as nib
import numpy as np
from scipy.spatial.transform import Rotation

from data_integrated_healthy import capsule_native_fov_margin
from data_integrated_hosts import UUID, binding, checked_file
from resectionlab.data_root import atomic_json, checked_root, safe_path, sha256_file
from resectionlab.desktop_bridge import BridgeSession, _Request
from resectionlab.geometry import segment_segment_distance
from resectionlab.surgical_cusa_v3 import default_profile, profile
from resectionlab.surgical_motion import envelopes, pair_conflict, swept_envelopes
from resectionlab.surgical_task_domain import reviewed_domain
from resectionlab.surgical_tools import TOOLS

REPO = Path(__file__).resolve().parents[1]
POSTREVIEW = 'integrated-v1/qc/independent/IXI077-source-reviewed-superficial-native-app-postreview-v1.json'
PROPOSAL = 'integrated-v1/healthy/IXI077/source-reviewed-superficial-v1/proposal.json'
OUTPUT = 'integrated-v1/probes/IXI077-production-fixed-preflight-v1'


def load_reviewed_inputs(root):
    post_path = safe_path(root, POSTREVIEW)
    post = json.loads(post_path.read_text())
    if (post['decision'] != 'accepted_with_explicit_limits_as_exact_geometry_and_overlay_candidate_pending_release_authority'
            or post['family'] != 'IXI:077' or post['physical_execution_approved'] is not False):
        raise ValueError('Independent zero-action geometry/app decision required')
    native = checked_file(root, post['native_geometry'])
    authority = json.loads(checked_file(root, post['source_export_authority']).read_text())
    proposal = json.loads(checked_file(root, {'path': PROPOSAL, 'sha256': authority['proposal_sha256']}).read_text())
    route = json.loads(checked_file(root, authority['route_report']).read_text())
    if (route['status'] != 'necessary_conditional_label_geometry_only_NOT_STATIC_OR_EXECUTION_ADMISSION'
            or not route['reverse_same_curve_covered_by_identical_fixed_orientation_interval_envelopes']):
        raise ValueError('Expected exact prior single-route geometry screen')
    for file, field in [('geometry.py', 'geometry_sha256'), ('surgical_motion.py', 'motion_sha256'),
                        ('surgical_tools.py', 'tools_sha256')]:
        if sha256_file(REPO / 'src/resectionlab' / file) != route['bindings'][field]:
            raise ValueError(f'Prior source cover has a different production implementation: {file}')
    return post, native, authority, proposal, route


def minimum_tool_gap(first, second):
    return min(segment_segment_distance(a, b, c, d) - ar - br
               for a, b, ar, _ in first for c, d, br, _ in second)


def inspect(root):
    root = checked_root(root, expected_uuid=UUID, required_bytes=2 * 1024**3)
    destination = safe_path(root, OUTPUT)
    if destination.exists():
        raise FileExistsError(f'Preserve existing preflight: {destination}')
    post, source, authority, proposal, route = load_reviewed_inputs(root)
    destination.mkdir()
    started = time.monotonic()
    phase = proposal['prospective_readonly_route_phase']['fixed_route']
    orientation = phase['orientation_xyzw']
    points = np.asarray(route['fixed_points'], dtype=float)
    a, target = points[[0, -1]]
    if (len(points) != 122 or not np.array_equal(a, phase['A_ras_mm'])
            or not np.array_equal(target, phase['T_ras_mm'])):
        raise ValueError('Fixed route points differ')
    axis = Rotation.from_quat(orientation).apply([0., 0., 1.])
    if not np.allclose(axis, phase['inward_source_axis0_ras'], atol=1e-12, rtol=0):
        raise ValueError('Fixed tool orientation differs')
    def pose(point):
        return {'position': point.tolist(), 'quaternion': orientation}

    calls = []
    with tempfile.TemporaryDirectory(prefix='bridge-', dir=destination) as workspace:
        session = BridgeSession(Path(workspace) / 'transfers', run_dir=Path(workspace) / 'runs')
        def call(operation, args):
            calls.append({'operation': operation, 'command': args.get('command')})
            return session.execute(operation, args, _Request(operation), lambda *_: None)
        try:
            identity = call('loadCase', {'path': str(source)})['caseHash']
            created = call('surgicalRehearsal', {'caseHash': identity, 'command': 'create'})
            inspected = call('surgicalRehearsal', {'caseHash': identity, 'command': 'inspect'})
            engine = session.surgical_sessions[identity][1]
            scene = engine.scene
            initial = created['view']
            if (initial['stateId'] != inspected['view']['stateId'] or initial['stateId'] != engine.state_id
                    or engine.actions or engine.state['elapsedSeconds'] != 0 or inspected['timeline']
                    or reviewed_domain(scene) is not None or profile(engine) is not None):
                raise ValueError('Expected a zero-action production state with no task-domain or CUSA revision admitted')
            slots = engine.state['slots']
            if (slots['right']['instrument'] != 'cusa' or slots['left']['instrument'] != 'cranial_drill'
                    or any(slot['active'] or slot['inserted'] or slot['grip'] or slot['parameters']
                           for slot in slots.values())):
                raise ValueError('Unexpected initial stock tool setup')
            cusa, params = TOOLS['cusa'], engine.parameters(slots['right'])
            assumed = {**params, 'contactGeometryRevision': 2}
            rate_profile = default_profile()
            if (params['radiusMm'] != 1.5 or cusa['shaftRadiusMm'] != .5
                    or params['depthMm'] != 2. or params['power'] != 1. or params['duty'] != 1.
                    or rate_profile['capacityMm3PerSecond'] != 2.):
                raise ValueError('Stock geometry or assumed capacity differs')
            parked = {key: envelopes(TOOLS[slot['instrument']], engine.parameters(slot),
                                     slot['position'], slot['quaternion'])
                      for key, slot in slots.items()}
            widened = (a - axis * 40., a, max(params['radiusMm'], cusa['shaftRadiusMm']), 'place')
            # This is the only new source-cell query. The place operation tests
            # the max radius over the entire 40 mm, not its two moving capsules.
            ids = engine.cells(*widened[:3])
            occupied = engine.occupancy().material_ids(ids)
            mask_ids = np.asarray(ids, dtype=np.int64)
            case = session._get_case(identity).case
            brain = case.compartments['brain'].ravel()
            lesion = case.compartments['synthetic_target'].ravel()
            vessels = case.compartments['registered_partial_vessels'].ravel()
            coverage_path = safe_path(root, 'integrated-v1/healthy/IXI077/rigid-v1/MRA-observed-coverage.nii.gz')
            if sha256_file(coverage_path) != proposal['coverage_sha256']:
                raise ValueError('Native MRA observed coverage differs')
            coverage_image = nib.load(coverage_path)
            if coverage_image.shape != scene.shape or not np.array_equal(coverage_image.affine, scene.affine):
                raise ValueError('Native MRA observed coverage grid differs')
            coverage = (np.asanyarray(coverage_image.dataobj) > 0).ravel()

            fov = lambda capsule: capsule_native_fov_margin(scene.shape, scene.affine, *capsule[:3])
            place_env = envelopes(cusa, assumed, a, orientation)
            placement_pair_gap = minimum_tool_gap(place_env, parked['left'])
            # Reconstruct only the analytic tool-vs-tool covers. The frozen
            # tissue/slab source IDs are reused without geometry re-query.
            left_conflicts, min_gap = [], math.inf
            unique = {part: set() for part in ('shaft', 'working')}
            min_route_fov = math.inf
            first_working_target = None
            leaves = route['records']['production_cover_leaves']
            if len(leaves) != len(points) - 1:
                raise ValueError('Frozen route interval count differs')
            for i, leaf in enumerate(leaves):
                first, second = points[i:i + 2]
                if (leaf['depth'] != 0 or not np.array_equal(first, leaf['start'])
                        or not np.array_equal(second, leaf['end'])):
                    raise ValueError('Frozen route partition differs')
                swept = swept_envelopes(cusa, assumed, pose(first), pose(second))
                min_gap = min(min_gap, minimum_tool_gap(swept, parked['left']))
                if pair_conflict(swept, parked['left']):
                    left_conflicts.append(i)
                min_route_fov = min(min_route_fov, leaf['result']['FOV_minimum_margin_mm'])
                for part, row in leaf['result']['parts'].items():
                    values = np.asarray(row['source_ids'], dtype=np.int64)
                    if (np.any(values < 0) or np.any(values >= brain.size)
                            or row['known_vessel_ids'] != values[vessels[values]].tolist()
                            or row['intracranial_outside_MRA_ids'] != values[(brain[values] | lesion[values]) & ~coverage[values]].tolist()):
                        raise ValueError('Frozen route membership differs from native package')
                    if row['known_vessel_ids'] or row['intracranial_outside_MRA_ids']:
                        raise ValueError('Unexpected route vessel or MRA slab conflict')
                    unique[part].update(map(int, values))
                    if part == 'working' and row['target_ids'] and first_working_target is None:
                        first_working_target = i

            voxel = scene.voxel_mm3
            target_cells = int(lesion.sum())
            route_ids = {part: np.asarray(sorted(values), dtype=np.int64) for part, values in unique.items()}
            result = {
                'schema': 'ressectionlab-IXI077-production-fixed-preflight-v1',
                'geometry_candidate': post['native_geometry'], 'independent_postreview': binding(root, safe_path(root, POSTREVIEW)),
                'prior_readonly_route': authority['route_report'], 'source_context': authority['source_context'],
                'source_bindings': {name: sha256_file(REPO / 'src/resectionlab' / name)
                                    for name in ('desktop_bridge.py', 'surgical_bridge.py', 'surgical_engine.py',
                                                 'surgical_cusa_v3.py', 'surgical_motion.py', 'surgical_tools.py',
                                                 'surgical_task_domain.py', 'geometry.py')},
                'public_calls': calls, 'case_hash': identity, 'scene_id': scene.identity, 'initial_state_id': initial['stateId'],
                'same_state_after_inspect': True, 'actions_committed': 0, 'action_previews': 0,
                'physical_clock_seconds': engine.state['elapsedSeconds'], 'slots': slots,
                'full_case_readiness': inspected['fullCaseReadiness'],
                'protection': initial['protection'], 'reviewed_task_domain_present': False,
                'startup': {'catalog_CUSA_parameters': params, 'mechanics_revision_at_create': None,
                            'prospective_revision': 2, 'prospective_CUSA_profile': rate_profile,
                            'initial_pair_conflict': pair_conflict(parked['left'], parked['right']),
                            'parked_min_native_FOV_margin_mm': {key: min(map(fov, capsules)) for key, capsules in parked.items()},
                            'parked_slots_are_inactive_uninserted': True},
                'placement': {'exact_production_capsule': {'a': widened[0].tolist(), 'b': a.tolist(),
                                                            'radius_mm': widened[2], 'length_mm': 40.},
                              'native_FOV_margin_mm': fov(widened), 'source_ids': ids.tolist(),
                              'occupied_source_ids': list(map(int, occupied)), 'unowned_cells': int(np.count_nonzero(scene.labels.ravel()[mask_ids] == 0)),
                              'known_vessel_ids': mask_ids[vessels[mask_ids]].tolist(),
                              'intracranial_outside_MRA_ids': mask_ids[(brain[mask_ids] | lesion[mask_ids]) & ~coverage[mask_ids]].tolist(),
                              'stock_CUSA_vs_parked_drill_gap_mm': placement_pair_gap},
                'route': {'A_RAS_mm': a.tolist(), 'T_RAS_mm': target.tolist(), 'orientation_xyzw': orientation,
                          'forward_length_mm': float(np.linalg.norm(target - a)), 'interval_count': len(leaves),
                          'reverse_only_same_curve_and_orientation': True,
                          'minimum_frozen_cover_FOV_margin_mm': min_route_fov,
                          'minimum_CUSA_vs_parked_drill_cover_gap_mm': min_gap,
                          'tool_pair_conflict_interval_indices': left_conflicts,
                          'first_possible_working_target_cover_interval_index': first_working_target,
                          'source_cells_by_part': {part: {
                              'known_vessel': int(vessels[values].sum()),
                              'intracranial_outside_MRA': int(((brain[values] | lesion[values]) & ~coverage[values]).sum()),
                              'unowned': int(np.count_nonzero(scene.labels.ravel()[values] == 0)),
                              'target': int(lesion[values].sum()),
                              'retained_non_target_brain': int((brain[values] & ~lesion[values]).sum()),
                          } for part, values in route_ids.items()}},
                'capacity': {'fixed_rate_mm3_per_second': 2., 'duration_guard_seconds': 30.,
                             'maximum_capacity_per_action_mm3': 60.,
                             'maximum_whole_voxels_one_action_if_all_useful': math.floor(60. / voxel),
                             'voxel_mm3': voxel, 'total_target_cells': target_cells,
                             'ideal_all_target_active_seconds_lower_bound': target_cells * voxel / 2.,
                             'ideal_all_target_action_count_lower_bound_from_rate_only': math.ceil(target_cells * voxel / 60.),
                             'full_target_removal_on_fixed_curve_possible': False,
                             'fixed_route_working_target_cover_voxels_upper_bound': int(lesion[route_ids['working']].sum()),
                             'time_to_first_actual_target_contact_or_removal_known': False},
                'scope': ['Startup create and inspect performed. Configure, place, application and withdrawal were not performed.',
                          'One new source capsule query at exact placement A. Existing route source IDs were reused.',
                          'Unowned and outside native FOV cells remain unknown. Neither an observed opening nor a reviewed task domain is available.',
                          'Full target removal is impossible on this fixed curve because its working upper cover includes fewer target cells than the target itself.',
                          'Conservative route cover does not prove which cells a future action would remove, or certify reach.'],
            }
            if (not 0 < result['placement']['native_FOV_margin_mm'] or len(occupied)
                    or first_working_target is None or left_conflicts or placement_pair_gap <= 0
                    or result['capacity']['fixed_route_working_target_cover_voxels_upper_bound'] >= target_cells):
                result['preflight_status'] = 'fixed_geometry_requires_restriction_or_rejection'
            else:
                result['preflight_status'] = 'geometry_prerequisites_only_physical_admission_withheld'
            if engine.state_id != initial['stateId'] or sha256_file(source) != post['native_geometry']['sha256']:
                raise ValueError('Read-only check changed production startup or source bytes')
            result.update(source_unchanged=True, elapsed_seconds=time.monotonic() - started,
                          generator_sha256=sha256_file(__file__))
        finally:
            session.close()
    checked_root(root, expected_uuid=UUID)
    atomic_json(destination / 'report.json', result)
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root')
    args = parser.parse_args()
    print(json.dumps(inspect(args.root), indent=2))
