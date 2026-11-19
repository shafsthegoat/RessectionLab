#!/usr/bin/env python3
"""Screen fixed-axis stock-tool geometry on one pristine organic CTA scenario."""
from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path
import sys
import time

import numpy as np
from scipy.spatial.transform import Rotation

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / 'src'))
from build_cta_organic_gliomas import COLLECTION, OBSERVATION, ORDER, UUID, checked_mask
from data_integrated_probe import envelope_admission
from resectionlab.core import array_digest
from resectionlab.data_root import atomic_json, checked_root, safe_path, sha256_file
from resectionlab.geometry import GeometryScene, capsule_voxel_indices
from resectionlab.surgical_motion import envelopes
from resectionlab.surgical_tools import TOOLS
from resectionlab.workspace_bundle import load_workspace

CASE_ID = 'TopCoW-014-CTA-ORGANIC-03-v1'
OUTPUT = 'qc/cta-organic-rehearsal-preflight-v1'
DIRECTIONS = 512
RAY_MM = np.arange(0., 111., .5)
DISCLOSURES = (
    'This is a static geometry screen of a fresh simulator scene on acquired CTA.',
    'Scalp exposure and dural opening remain conditional and unmodeled.',
    'Unowned voxels and areas without supplied vessels still have uncertain anatomy.',
    'Exact research truth, prior approvals, prior actions and virtual openings are excluded.',
    'Capsule intersections are geometric estimates of work coverage; production actuation and review are separate.',
)


def tool_segments(name, point, quaternion):
    tool = TOOLS[name]
    return envelopes(tool, {'radiusMm': tool['radiusMm'], 'depthMm': 2.,
                            'contactGeometryRevision': 2}, point, quaternion)


def cover(name, start, end, quaternion):
    # Tool attitude is fixed and the motion is parallel to the shaft. The
    # drill and CUSA capsules are collinear with this motion, so joining their
    # original rear endpoint to their final forward endpoint covers the sweep.
    initial = tool_segments(name, start, quaternion)
    final = tool_segments(name, end, quaternion)
    return [(a, last, radius, part) for (a, _, radius, part), (_, last, _, _) in zip(initial, final)]


def cells(grid, segments, part=None):
    hits = []
    for a, b, radius, shape in segments:
        if part is None or shape == part:
            contained = capsule_voxel_indices(grid, a, b, radius)
            hits.extend(np.ravel_multi_index(contained.T, grid.shape))
    return np.unique(hits).astype(np.intp)


def fibonacci_directions(count):
    index = np.arange(count)
    z = 1 - 2 * (index + .5) / count
    phi = index * np.pi * (3 - np.sqrt(5))
    radius = np.sqrt(1 - z * z)
    return np.column_stack([radius * np.cos(phi), radius * np.sin(phi), z])


def target_points(grid, target):
    points = grid.voxel_to_world(np.argwhere(target))
    center = points.mean(axis=0)
    requests = [center]
    for axis in range(3):
        for quantile in (.25, .75):
            point = center.copy()
            point[axis] = np.quantile(points[:, axis], quantile)
            requests.append(point)
    return np.unique([points[np.argmin(np.linalg.norm(points - point, axis=1))]
                      for point in requests], axis=0)


def screen_direction(case, grid, origin, direction, layers, enclosure, lower_limit):
    native = np.rint(grid.world_to_voxel(origin + RAY_MM[:, None] * direction)).astype(np.intp)
    valid = np.all((native >= 0) & (native < np.array(case.mri.shape)), axis=1)
    n = len(valid) if valid.all() else int(np.flatnonzero(~valid)[0])
    ray = native[:n]
    if not n:
        return 'target_outside_native_grid', None
    line = np.ravel_multi_index(ray.T, case.mri.shape)
    skull, brain, vessels, target = layers
    bone_on_ray = np.flatnonzero(skull[line])
    if not len(bone_on_ray) or RAY_MM[bone_on_ray[-1]] > 35.:
        return 'no_local_source_skull_crossing', None
    inner, outer = RAY_MM[bone_on_ray[[0, -1]]]
    if RAY_MM[n - 1] - outer < 44.:
        return 'source_fov_too_short_for_full_stock_shaft', None
    # Stop at the exterior-most observed contour voxel preceding the inner
    # skull on this ray. The reverse remains exactly the same fixed line.
    target_interval = np.flatnonzero(target[line[:bone_on_ray[0]]])
    if not len(target_interval):
        return 'no_contour_before_skull', None
    target_offset = RAY_MM[target_interval[-1]]
    t = origin + target_offset * direction
    a = origin + (outer + 4.) * direction
    quaternion = Rotation.align_vectors([-direction], [[0., 0., 1.]])[0].as_quat()
    cusa = cover('cusa', a, t, quaternion)
    if not envelope_admission(cusa, case.affine, np.asarray(case.mri.shape))['admitted']:
        return 'stock_cusa_sweep_exits_native_fov', None
    placement = tool_segments('cusa', a, quaternion) + tool_segments('cranial_drill', a, quaternion)
    if not envelope_admission(placement, case.affine, np.asarray(case.mri.shape))['admitted']:
        return 'stock_placement_exits_native_fov', None
    placed = cells(grid, placement)
    if np.any(skull[placed] | brain[placed] | vessels[placed] | target[placed]):
        return 'stock_placement_overlaps_modeled_anatomy', None
    cusa_all = cells(grid, cusa)
    if vessels[cusa_all].any():
        return 'stock_cusa_sweep_contacts_supplied_vessels', None
    cusa_work = cells(grid, cusa, 'working')
    cover_target = int(target[cusa_work].sum())
    if not cover_target:
        return 'stock_cusa_working_sweep_misses_contour', None
    cusa_bone = cusa_all[skull[cusa_all]]
    if not len(cusa_bone):
        return 'no_established_bone_crossing_in_stock_cusa_cover', None
    if np.any(np.unravel_index(cusa_all, case.mri.shape)[2] < lower_limit):
        return 'stock_cusa_sweep_below_supported_superior_enclosure', None
    if not enclosure[line[target_interval[-1]]]:
        return 'contour_endpoint_outside_superior_source_enclosure', None
    for drill_offset in (0., .5, 1., 1.5):
        drill_point = origin + (inner - drill_offset) * direction
        drill = cover('cranial_drill', a, drill_point, quaternion)
        if not envelope_admission(drill, case.affine, np.asarray(case.mri.shape))['admitted']:
            continue
        drill_all = cells(grid, drill)
        if (vessels[drill_all].any() or np.any((brain[drill_all] | target[drill_all]) & ~skull[drill_all])
                or np.any(np.unravel_index(drill_all, case.mri.shape)[2] < lower_limit)):
            continue
        drill_work = cells(grid, drill, 'working')
        bone_forecast = drill_work[skull[drill_work]]
        if np.setdiff1d(cusa_bone, bone_forecast, assume_unique=True).size:
            continue
        return 'static_stock_geometry_candidate', {
            'entry_ras_mm': a.tolist(), 'drill_endpoint_ras_mm': drill_point.tolist(),
            'contour_endpoint_ras_mm': t.tolist(), 'orientation_xyzw': quaternion.tolist(),
            'observed_contour_working_cover_cells': cover_target,
            'candidate_calvarial_working_cover_cells': int(len(bone_forecast)),
            'stock_cusa_bone_blocker_cells': int(len(cusa_bone)),
            'estimated_brain_working_cover_cells': int(np.count_nonzero(brain[cusa_work] & ~target[cusa_work] & ~skull[cusa_work])),
            'placement_cta_range': [float(case.mri.ravel()[placed].min()), float(case.mri.ravel()[placed].max())],
            'placement_cta_voxels_at_least_minus_200': int(np.count_nonzero(case.mri.ravel()[placed] >= -200.)),
            'checked_reverse_geometry': 'identical fixed-axis capsules returned through the same line',
        }
    return 'prospective_stock_drill_cannot_cover_cusa_bone_with_source_restrictions', None


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path)
    parser.add_argument('--output-name', default=Path(OUTPUT).name, help='A new QC directory name.')
    args = parser.parse_args()
    if not args.output_name or args.output_name in {'.', '..'} or Path(args.output_name).name != args.output_name:
        parser.error('Choose one directory name under the external qc root')
    started = time.monotonic()
    root = checked_root(args.root, expected_uuid=UUID)
    receipt_path = safe_path(root, f'{COLLECTION}/{CASE_ID}/receipt.json')
    receipt = json.loads(receipt_path.read_text())
    native_binding = receipt['native_case']
    path = safe_path(root, native_binding['path'])
    if sha256_file(path) != native_binding['sha256']:
        raise ValueError('Scenario checksum differs from its generation receipt')
    case, artifacts, workspace = load_workspace(path)
    data = case.metadata['data_case']
    adapter = case.metadata['surgical_adapter']
    if (case.case_id != CASE_ID or case.frame != 'RAS+' or case.semantic_hash != receipt['case_hash']
            or data['role'] != 'DEVELOPMENT' or data['family_id'] != 'TopCoW:014'
            or artifacts or workspace is None or workspace['episode'] is not None
            or set(case.compartments) != {*ORDER, OBSERVATION}
            or 'reviewedTaskDomainV1' in adapter or 'integrated_rehearsal' in case.metadata):
        raise ValueError('A pristine source-native DEVELOPMENT scenario with no task or history is required')
    target = case.compartments[OBSERVATION]
    if array_digest(target) != receipt['observation']['array_hash']:
        raise ValueError('Approximate observable contour checksum differs')
    enclosure_binding = receipt['source']['anatomy_masks']['enclosed_coverage']
    enclosure = checked_mask(root, enclosure_binding, case)
    layers = (case.compartments[ORDER[1]].ravel(), case.compartments[ORDER[0]].ravel(),
              (case.compartments[ORDER[2]] | case.compartments[ORDER[3]]).ravel(), target.ravel())
    grid = GeometryScene(np.zeros(case.mri.shape, bool), case.affine)
    lower_limit = int(np.flatnonzero(enclosure.any(axis=(0, 1)))[0])
    directions = fibonacci_directions(DIRECTIONS)
    points = target_points(grid, target)
    counts, candidates = Counter(), []
    for point in points:
        for direction in directions:
            reason, candidate = screen_direction(case, grid, point, direction, layers, enclosure.ravel(), lower_limit)
            counts[reason] += 1
            if candidate is not None:
                candidates.append(candidate)
    candidates.sort(key=lambda row: (row['placement_cta_voxels_at_least_minus_200'],
                                     row['estimated_brain_working_cover_cells'],
                                     row['candidate_calvarial_working_cover_cells']))
    if sha256_file(path) != native_binding['sha256']:
        raise ValueError('Pristine case changed during static screening')
    status = 'static_candidate_pending_fresh_review' if candidates else 'no_static_candidate_in_bounded_screen'
    report = {'schema': 'resectionlab.cta-organic-stock-static-preflight.v1', 'case_id': CASE_ID, 'status': status,
              'source': native_binding, 'case_hash': case.semantic_hash, 'role': data['role'],
              'source_contour': {'layer': OBSERVATION, 'hash': array_digest(target)},
              'enclosure': enclosure_binding, 'tool_names': ['cranial_drill', 'cusa'],
              'driver_sha256': sha256_file(__file__), 'target_seed_count': len(points),
              'directions_per_seed': DIRECTIONS, 'routes_screened': sum(counts.values()),
              'first_stop_counts': dict(counts), 'static_candidate_count': len(candidates),
              'ranked_static_candidates': candidates[:6], 'production_actions_executed': 0,
              'instrument_admitted': False, 'case_unchanged': True, 'scope': list(DISCLOSURES),
              'wall_seconds': time.monotonic() - started}
    destination = safe_path(root, Path(OUTPUT).with_name(args.output_name))
    destination.mkdir(exist_ok=False)
    atomic_json(destination / 'receipt.json', report)
    print(json.dumps({key: value for key, value in report.items() if key != 'ranked_static_candidates'}, indent=2))


if __name__ == '__main__':
    main()
