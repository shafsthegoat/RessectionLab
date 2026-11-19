#!/usr/bin/env python3
"""Bounded, read-only stock craniotome coverage on pristine CTA014 organic 01."""
from __future__ import annotations

import argparse
from collections import Counter
from itertools import combinations
import json
from pathlib import Path
import time

import numpy as np
from scipy.spatial.transform import Rotation

from probe_cta_organic_rehearsal import (
    COLLECTION, DIRECTIONS, DISCLOSURES, OBSERVATION, ORDER, RAY_MM, UUID,
    cells, checked_mask, cover, envelope_admission, fibonacci_directions, screen_direction,
    target_points, tool_segments,
)
from resectionlab.core import array_digest
from resectionlab.data_root import atomic_json, checked_root, safe_path, sha256_file
from resectionlab.surgical_engine import SurgicalEngine
from resectionlab.surgical_environment import standardize
from resectionlab.surgical_motion import envelopes, pair_conflict, swept_envelopes
from resectionlab.surgical_opening_skills import stock_kerf_options
from resectionlab.surgical_scene import from_case
from resectionlab.surgical_tools import TOOLS
from resectionlab.workspace_bundle import load_workspace

CASE_ID = 'TopCoW-014-CTA-ORGANIC-01-v1'
PREVIOUS = f'qc/{CASE_ID}-static-preflight/receipt.json'
OUTPUT = 'cta014-organic-01-multilane-v1'
FINAL_GATE = 'prospective_stock_drill_cannot_cover_cusa_bone_with_source_restrictions'
ROUTE_LIMIT = 3
PREFIX_LIMIT = 3
LANE_LIMIT = 5
KERF = TOOLS['craniotome']
PARAMETERS = {'radiusMm': KERF['radiusMm'], 'depthMm': 2., 'contactGeometryRevision': 2}


def route_geometry(grid, origin, direction, bone, target):
    """Reconstruct the exact ray after the straight screen reaches its final gate."""
    indices = np.rint(grid.world_to_voxel(origin + RAY_MM[:, None] * direction)).astype(np.intp)
    valid = np.all((indices >= 0) & (indices < np.array(grid.shape)), axis=1)
    end = len(valid) if valid.all() else int(np.flatnonzero(~valid)[0])
    ray = np.ravel_multi_index(indices[:end].T, grid.shape)
    crossings = np.flatnonzero(bone[ray])
    contour = np.flatnonzero(target[ray[:crossings[0]]])
    tip = origin + RAY_MM[contour[-1]] * direction
    start = origin + (RAY_MM[crossings[-1]] + 4.) * direction
    quaternion = Rotation.align_vectors([-direction], [[0., 0., 1.]])[0].as_quat()
    depth = float(np.linalg.norm(start - tip))
    return start, tip, quaternion, min(24., depth)


def motion_reason(segment, grid, stationary, support_floor):
    if not envelope_admission(segment, grid.affine, np.asarray(grid.shape))['admitted']:
        return 'whole_tool_exits_native_grid'
    if any(min(a[2], b[2]) - radius < support_floor for a, b, radius, _ in segment):
        return 'whole_tool_crosses_inferior_enclosure_cutoff'
    if pair_conflict(segment, stationary):
        return 'whole_tool_intersects_parked_instrument'
    return None


def lane_cover(grid, option, bone, occupied, vessels, blockers, stationary, support_floor):
    """Forecast kerf-eligible source cells and a reverse without new contact."""
    start = option['outsidePose']
    path = [start, *option['path']]
    initial = tool_segments('craniotome', start['position'], start['quaternion'])
    refusal = motion_reason(initial, grid, stationary, support_floor)
    if refusal is not None:
        return refusal, None
    if occupied[cells(grid, initial)].any():
        return 'external_placement_intersects_modeled_anatomy', None

    removed = set()
    worthwhile = []
    intervals = []
    half_width = .5 * float(np.linalg.norm(grid.affine[:3, :3], axis=0).min())
    inverse_orientation = Rotation.from_quat(start['quaternion']).inv()
    stop = 'reached_lane_budget'
    for index, (before, after) in enumerate(zip(path, path[1:]), 1):
        geometry = swept_envelopes(KERF, PARAMETERS, before, after)
        refusal = motion_reason(geometry, grid, stationary, support_floor)
        if refusal is not None:
            stop = refusal
            break
        shaft = set(cells(grid, geometry, 'shaft'))
        work = set(cells(grid, geometry, 'working'))
        contacts = np.asarray(sorted((shaft | work) - removed), dtype=np.intp)
        if vessels[contacts].any():
            stop = 'swept_tool_contacts_supplied_vessel_membership'
            break
        if any(occupied[value] for value in shaft - removed):
            stop = 'forward_shaft_intersects_retained_anatomy'
            break
        if np.any(occupied[contacts] & ~bone[contacts]):
            stop = 'working_envelope_contacts_nonbone'
            break
        intervals.append(shaft)
        candidate = np.asarray(sorted(work - removed), dtype=np.intp)
        candidate = candidate[bone[candidate]]
        if len(candidate):
            world = grid.voxel_to_world(np.column_stack(np.unravel_index(candidate, grid.shape)))
            xyz = inverse_orientation.apply(world - np.asarray(after['position']))
            # Same local kerf plane and depth predicates as the catalog
            # effect. These cell IDs are forecasts; the observation is static.
            new = set(candidate[(np.abs(xyz[:, 1]) <= half_width)
                                & (np.abs(xyz[:, 2]) <= PARAMETERS['depthMm'])])
            if new:
                removed.update(new)
                if blockers & removed:
                    worthwhile.append((index, set(removed)))
                    worthwhile = worthwhile[-PREFIX_LIMIT:]
    if not worthwhile:
        return stop, None

    def retained_work(pose, predicted):
        working = cells(grid, tool_segments('craniotome', pose['position'], pose['quaternion']), 'working')
        return {value for value in working if occupied[value] and value not in predicted}

    for index, predicted in reversed(worthwhile):
        previous = retained_work(path[index], predicted)
        for interval in range(index - 1, -1, -1):
            if any(occupied[value] for value in intervals[interval] - predicted):
                break
            current = retained_work(path[interval], predicted)
            if current - previous:
                break
            previous = current
        else:
            return 'reversible_bone_only_coverage_forecast', {
                'lane_id': option['id'], 'outside_pose': start, 'interval_count': index,
                'stock_interval_mm': option['microIntervalMm'],
                'forecast_bone_cells': sorted(map(int, predicted)),
                'forecast_blocker_cells': sorted(map(int, blockers & predicted)),
                'forward_stop': stop,
            }
    return 'predicted_reverse_contacts_retained_anatomy', None


def minimum_cover(blockers, lanes):
    """Match the historical set cover purpose with a finite exact small search."""
    footprints = [set(row['forecast_blocker_cells']) for row in lanes]
    union = set().union(*footprints)
    if blockers - union:
        return 'uncovered_bone_blockers', blockers - union, []
    for count in range(1, min(LANE_LIMIT, len(lanes)) + 1):
        for selected in combinations(range(len(lanes)), count):
            if blockers <= set().union(*(footprints[index] for index in selected)):
                return 'static_bone_cover_pending_review', set(), [lanes[index]['lane_id'] for index in selected]
    return 'cover_exceeds_declared_lane_limit', set(), []


def evaluate_route(engine, start, tip, quaternion, depth, bone, occupied, vessels, target, support_floor):
    grid = engine.grid
    slot = engine.state['slots']['left']
    stationary = envelopes(TOOLS[slot['instrument']], engine.parameters(slot), slot['position'], slot['quaternion'])
    cusa = cover('cusa', start, tip, quaternion)
    cusa_cells = cells(grid, cusa)
    blockers = set(cusa_cells[bone[cusa_cells]])
    target_work = cells(grid, cusa, 'working')
    reasons, lanes = Counter(), []
    options = stock_kerf_options(engine.scene.identity, {'A': start.tolist(), 'quaternion': quaternion.tolist()},
                                 max_depth_mm=depth)
    route_refusal = motion_reason(cusa, grid, stationary, support_floor)
    if route_refusal is None:
        for option in options:
            reason, evidence = lane_cover(grid, option, bone, occupied, vessels, blockers, stationary, support_floor)
            reasons[reason] += 1
            if evidence is not None:
                lanes.append(evidence)
    status, residual, selected = minimum_cover(blockers, lanes)
    return {'entry_ras_mm': start.tolist(), 'contour_endpoint_ras_mm': tip.tolist(),
            'orientation_xyzw': quaternion.tolist(), 'stock_cusa_full_sweep_refusal': route_refusal,
            'static_status': route_refusal or status, 'lane_count_generated': len(options),
            'lane_count_screened': sum(reasons.values()), 'lane_status_counts': dict(reasons),
            'stock_cusa_bone_blockers': sorted(map(int, blockers)),
            'residual_bone_blockers_after_forecast_union': sorted(map(int, residual)),
            'selected_lane_ids': selected, 'reversible_lane_forecasts': lanes,
            'stock_cusa_contour_contact_cells': int(np.count_nonzero(target[target_work])),
            'effect_or_admission_established': False}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path)
    parser.add_argument('--output-name', default=OUTPUT)
    args = parser.parse_args()
    if args.output_name in {'.', '..'} or Path(args.output_name).name != args.output_name:
        parser.error('Choose a fresh directory name under qc/')
    started = time.monotonic()
    root = checked_root(args.root, expected_uuid=UUID)
    destination = safe_path(root, Path('qc') / args.output_name)
    if destination.exists():
        raise FileExistsError('Preserve the existing experiment receipt')
    previous_path = safe_path(root, PREVIOUS)
    previous = json.loads(previous_path.read_text())
    source = previous['source']
    path = safe_path(root, source['path'])
    original_hash = sha256_file(path)
    preceding_driver = Path(__file__).with_name('probe_cta_organic_rehearsal.py')
    if (previous['case_id'] != CASE_ID or previous['driver_sha256'] != sha256_file(preceding_driver)
            or source['sha256'] != original_hash or previous['first_stop_counts'][FINAL_GATE] != 40):
        raise ValueError('Inputs differ from the frozen straight-screen facts')
    case, artifacts, workspace = load_workspace(path)
    metadata = case.metadata
    adapter = metadata['surgical_adapter']
    if (case.case_id != CASE_ID or case.frame != 'RAS+' or metadata['data_case']['role'] != 'DEVELOPMENT'
            or case.semantic_hash != previous['case_hash'] or artifacts or workspace is None
            or workspace['episode'] is not None or set(case.compartments) != {*ORDER, OBSERVATION}
            or 'reviewedTaskDomainV1' in adapter or 'integrated_rehearsal' in metadata
            or set(case.source_compartments) != set(case.compartments)
            or any(not np.array_equal(case.source_compartments[k], m) for k, m in case.compartments.items())):
        raise ValueError('Expected exact unacted anatomy, with no task admission or saved opening')
    contour = case.compartments[OBSERVATION]
    if array_digest(contour) != previous['source_contour']['hash']:
        raise ValueError('Visible simulated contour differs from the preceding screen')
    enclosure = checked_mask(root, previous['enclosure'], case)
    lower = int(np.flatnonzero(enclosure.any(axis=(0, 1)))[0])
    engine = SurgicalEngine(standardize(from_case(case)))
    initial_state = engine.state_id
    grid = engine.grid
    support_floor = float(grid.voxel_to_world([0., 0., lower - .5])[2])
    bone, brain = (case.compartments[name].ravel() for name in (ORDER[1], ORDER[0]))
    vessels = (case.compartments[ORDER[2]] | case.compartments[ORDER[3]]).ravel()
    target = contour.ravel()
    occupied = bone | brain | vessels | target
    layers = bone, brain, vessels, target
    points = target_points(grid, contour)
    directions = fibonacci_directions(DIRECTIONS)
    routes = []
    rerouted = 0
    for point in points:
        for direction in directions:
            reason, _ = screen_direction(case, grid, point, direction, layers, enclosure.ravel(), lower)
            rerouted += 1
            if reason != FINAL_GATE:
                continue
            start, tip, quaternion, depth = route_geometry(grid, point, direction, bone, target)
            routes.append(evaluate_route(engine, start, tip, quaternion, depth, bone, occupied,
                                         vessels, target, support_floor))
            if len(routes) == ROUTE_LIMIT:
                break
        if len(routes) == ROUTE_LIMIT:
            break
    if len(routes) != ROUTE_LIMIT:
        raise ValueError('Frozen straight candidate subset did not reproduce')
    if (engine.state_id != initial_state or engine.actions or engine.history or engine.state['removed']
            or sha256_file(path) != original_hash):
        raise ValueError('Unacted source or modeled state changed during geometry screening')
    successes = sum(route['static_status'] == 'static_bone_cover_pending_review' for route in routes)
    report = {
        'schema': 'resectionlab.cta-organic-stock-multilane-static-v1', 'case_id': CASE_ID,
        'source': source, 'source_case_hash': case.semantic_hash, 'source_scene_id': engine.scene.identity,
        'source_contour': previous['source_contour'], 'source_contour_cells': int(target.sum()),
        'native_voxel_mm3': float(grid.voxel_volume_mm3), 'enclosure': previous['enclosure'],
        'straight_screen': {'path': PREVIOUS, 'sha256': sha256_file(previous_path)},
        'driver_sha256': sha256_file(__file__), 'initial_and_final_state': initial_state,
        'straight_routes_rechecked_to_select_subset': rerouted, 'first_final_gate_routes_sampled': ROUTE_LIMIT,
        'stock_options_per_route': 20, 'maximum_reversible_prefixes_checked_per_lane': PREFIX_LIMIT,
        'maximum_lanes_per_cover': LANE_LIMIT, 'static_full_bone_cover_candidates': successes,
        'status': 'static_cover_pending_fresh_review' if successes else 'no_stock_bone_cover_in_bounded_family',
        'routes': routes, 'production_actions_executed': 0, 'instrument_admitted': False,
        'case_unchanged': True, 'scope': [*DISCLOSURES,
            'Independent lane forecasts exclude a pre-opened model and only hypothesize cuts within each lane.',
            'A full bone cover is not a full target treatment or a verified motion sequence.'],
        'wall_seconds': time.monotonic() - started,
    }
    destination.mkdir(exist_ok=False)
    atomic_json(destination / 'receipt.json', report)
    print(json.dumps({key: value for key, value in report.items() if key != 'routes'}, indent=2))


if __name__ == '__main__':
    main()
