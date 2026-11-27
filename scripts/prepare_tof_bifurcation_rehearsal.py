#!/usr/bin/env python3
"""Prepare a restricted native TOF bifurcation and a declared local exposure."""
from __future__ import annotations

import argparse
from collections import defaultdict
from hashlib import file_digest
from itertools import permutations
import json
from pathlib import Path
import re

import numpy as np
from scipy.ndimage import distance_transform_edt, gaussian_filter, map_coordinates

from resectionlab.core import CaseData, SourceRef
from resectionlab.data_root import checked_root
from resectionlab.imaging import save_case
from resectionlab.surgical_glioma_materials import research_profile, RETAINED_VESSEL_SEGMENTS
from resectionlab.surgical_physics_v2 import default_profile
from resectionlab.surgical_scene import from_case
from resectionlab.surgical_wall_sweep import POLICY as WALL_SWEEP_POLICY, PROPERTY as WALL_SWEEP_PROPERTY
from resectionlab.surgical_vessel_vertex_motion import POLICY as VERTEX_MOTION_POLICY, PROPERTY as VERTEX_MOTION_PROPERTY
from resectionlab.workspace_bundle import save_workspace

HDD = Path('/Volumes/shafrir/old coding projects/RessectionLabData')
UUID = '9CC3623C-E38B-3EEE-B6BB-B9A06C193700'
CAMPAIGN = Path('qc/eden2020-glioma-5714812')
PRIOR = CAMPAIGN / 'vascular-reconstruction-20261123-v1'
CURRENT = CAMPAIGN / 'vascular-reconstruction-continuity-20261123-v2'
VESSEL = 'Native TOF vascular estimate'
BRAIN = 'Modeled brain exposure'
ROD_MODEL = 'distributed-source-rods-v1'
CROP_MARGIN = np.array([45, 48, 27])
GRAPH_CROP_HALF_MM = 7.5
GRAPH_LEG_MIN_MM = 3.8
GRAPH_LEG_TARGET_MM = 5.5
ASSUMPTIONS = (
    'Native TOF score geometry and original source branch identities. The exposed brain '
    'banks are a generated research geometry on the native TOF grid with low-frequency '
    'source texture. Attachments, vessel mechanical constants and hydraulic terms are '
    'assigned. Vessel class, patient mechanics and operative registration are unverified.'
)


def sha256(path):
    with path.open('rb') as stream:
        return file_digest(stream, 'sha256').hexdigest()


def read(path):
    return json.loads(path.read_text())


def write(path, data):
    path.write_text(json.dumps(data, indent=2, allow_nan=False) + '\n')


def load_source(root, subject, junction, *, source_graph_fork=False):
    prior = root / PRIOR
    source = root / 'prepared/Eden 2020 ResectionLab Projects (restricted)' / subject
    manifest_file = source / 'patient-anatomy.json'
    manifest = read(manifest_file)
    review_path = prior / 'source-review' / subject / 'rois.json'
    mark = None if source_graph_fork else next(row for row in read(review_path) if row['id'] == 'bifurcation')
    case_files = prior / 'cases' / subject
    report = read(case_files / 'report.json')
    graph_receipt = read(case_files / 'graph-receipt.json')
    graph_file = case_files / 'source-branches.json'
    graph = read(graph_file)
    scan = next(row for row in manifest['series'] if row['id'] == 'tof')
    vessels = next(row for row in manifest['layers']
                   if row['source_series'] == 'tof' and row.get('probability')
                   and row.get('provenance', {}).get('model') in {'vesselfm', 'vesselfm-base'})
    native_affine = np.asarray(scan['affine'], dtype=float)
    full_shape = tuple(scan['data']['shape'])
    original_sha = scan['original']['sha256']
    if (graph['source_scan_sha256'] != original_sha or tuple(graph['source_shape']) != full_shape
            or not np.allclose(graph['source_affine_ras_mm'], native_affine, rtol=0, atol=1e-7)):
        raise ValueError('The saved branch graph differs from the native TOF grid')
    if (graph_receipt['partition_sha256'] != report['partition_sha256']
            or graph_receipt['source_manifest_sha256'] != report['source_manifest_sha256']):
        raise ValueError('The saved branch graph differs from the reviewed partition')
    if np.any(junction < 0) or np.any(junction >= full_shape):
        raise ValueError('The selected junction lies outside acquired TOF coverage')
    if source_graph_fork:
        spacing = np.linalg.norm(native_affine[:3, :3], axis=0)
        margin = np.ceil(GRAPH_CROP_HALF_MM / spacing).astype(int) + 1
        start = np.maximum(0, junction - margin)
        stop = np.minimum(full_shape, junction + margin + 1)
    else:
        crop_center = np.asarray(mark['voxel_ijk'])
        if np.linalg.norm(native_affine[:3, :3] @ (junction - crop_center)) > mark['radius_mm']:
            raise ValueError('The selected junction lies outside the reviewed native source region')
        start = crop_center - CROP_MARGIN
        stop = crop_center + CROP_MARGIN + 1
        if np.any(start < 0) or np.any(stop > full_shape):
            raise ValueError('The reviewed fork crop exceeds acquired coverage')
    slab = tuple(slice(a, b) for a, b in zip(start, stop))
    paths = {
        'sourceManifest': (manifest_file, report['source_manifest_sha256']),
        'nativeSignal': (source / scan['data']['path'], scan['data']['sha256']),
        'nativeScores': (source / vessels['probability']['path'], vessels['probability']['sha256']),
        'visibilityPartition': (case_files / 'partition.npy', report['partition_sha256']),
        'sourceGraph': (graph_file, graph_receipt['graph_sha256']),
    }
    if mark is not None:
        paths['sourceReviewMarks'] = (review_path, sha256(review_path))
    for file, expected in paths.values():
        if sha256(file) != expected:
            raise ValueError(f'Source bytes differ from signed references: {file}')
    image, score, classes = (
        np.array(np.load(paths[key][0], mmap_mode='r', allow_pickle=False)[slab], copy=True)
        for key in ('nativeSignal', 'nativeScores', 'visibilityPartition')
    )
    if image.dtype != np.float32 or score.dtype != np.float32 or classes.dtype != np.uint8:
        raise ValueError('Unexpected native array data types')
    if np.any(classes > 3):
        raise ValueError('The reviewed partition contains unknown vessel source classes')
    # Visibility decisions never remove vessel occupancy from the rehearsal.
    vessel = classes > 0
    if np.any(score[vessel] < .5):
        raise ValueError('The selected native vessel mask differs from the review score level')
    affine = native_affine.copy()
    affine[:3, 3] += affine[:3, :3] @ start
    receipt = {
        'subject': subject,
        'nativeSourceSha256': original_sha,
        'sourceSnapshotId': manifest['snapshot_id'],
        'readOnlyInputSha256': {key: digest for key, (_, digest) in paths.items()},
        'nativeBranchJunctionIJK': junction.tolist(),
        'nativeCropStartIJK': start.tolist(),
        'nativeCropStopIJKExclusive': stop.tolist(),
        'nativeShape': list(full_shape),
        'nativeAffineRAS': native_affine.tolist(),
        'cropAffineRAS': affine.tolist(),
        'vesselMaskClasses': ['retained', 'uncertain_retained'] + (['hidden_candidate'] if np.any(classes == 3) else []),
        'registeredStructuralTissueUsed': False,
        'patientSpecificSimulationAdmitted': False,
    }
    if source_graph_fork:
        receipt['siteSelection'] = 'frozen-native-source-graph'
    else:
        receipt['sourceReviewMark'] = mark
    return image, score, vessel, graph, affine, start, receipt


def bridge_steps(points, vessel, clearance, forbidden, *, preserve_anchors=False):
    """Realize local diagonal skeleton links using only face-connected source lumen."""
    path = [tuple(points[0])]
    remaining = set(map(tuple, points[1:])) if preserve_anchors else set()
    for target in points[1:]:
        remaining.discard(tuple(target))
        origin = np.asarray(path[-1])
        delta = target - origin
        axes = np.flatnonzero(delta)
        if np.any(np.abs(delta) > 1):
            raise ValueError('A reviewed source link exceeded one native-cell neighborhood')
        candidates = []
        for ordered in permutations(axes):
            current = origin.copy()
            cells = []
            for axis in ordered:
                current[axis] += delta[axis]
                cells.append(tuple(current))
            if all(vessel[p] and p not in path and p not in forbidden and p not in remaining for p in cells):
                candidates.append(cells)
        if not candidates:
            raise ValueError(f'A reviewed skeleton link has no exclusive in-lumen face path: {origin}, {target}')
        best = max(candidates, key=lambda cells: (sum(clearance[p] for p in cells), cells))
        path.extend(best)
    return np.asarray(path, dtype=np.int64)


def graph_leg_prefix(points, supported, vessel, spacing):
    inside = ((points >= 1) & (points < np.asarray(vessel.shape) - 1)).all(axis=1)
    in_source = np.zeros(len(points), dtype=bool)
    in_source[inside] = vessel[tuple(points[inside].T)]
    valid_edges = supported & in_source[:-1] & in_source[1:]
    unusable = np.flatnonzero(~valid_edges)
    length = int(unusable[0]) + 1 if len(unusable) else len(points)
    points = points[:length]
    distances = np.linalg.norm(np.diff(points, axis=0) * spacing, axis=1).cumsum()
    samples = min(len(points), int(np.searchsorted(distances, GRAPH_LEG_TARGET_MM)) + 2)
    if samples < 3 or distances[samples - 2] < GRAPH_LEG_MIN_MM:
        raise ValueError('A native graph leg lacks a continuous source-supported prefix')
    return points[:samples]


def graph_fork_routes(branches, vessel, start, spacing, native_shape, fork):
    anchors = {}
    for row in branches:
        ids = row['source_cell_ids']
        supported = np.asarray(row['edge_local_face_supported'], dtype=bool)
        if row['closed'] or row['endpoint_degrees'][0 if ids[0] == fork else 1] != 3:
            raise ValueError('The selected source anchor is not a native degree-three junction')
        if ids[0] != fork:
            ids, supported = ids[::-1], supported[::-1]
        points = np.column_stack(np.unravel_index(ids, native_shape)) - start
        anchors[row['id']] = graph_leg_prefix(points, supported, vessel, spacing)
    clearance = distance_transform_edt(vessel, sampling=spacing)
    reserved = {key: set(map(tuple, points[1:])) for key, points in anchors.items()}
    if sum(map(len, reserved.values())) != len(set().union(*reserved.values())):
        raise ValueError('Native graph legs share source cells outside the junction')
    for order in permutations(anchors):
        paths = {}
        occupied = set()
        try:
            for key in order:
                forbidden = occupied.union(*(cells for other, cells in reserved.items() if other != key))
                path = bridge_steps(anchors[key], vessel, clearance, forbidden, preserve_anchors=True)
                occupied.update(map(tuple, path[1:]))
                paths[key] = path
        except ValueError:
            continue
        return {key: paths[key] for key in anchors}
    raise ValueError('The native graph fork has no exclusive in-lumen face paths for all three legs')


def source_routes(graph, vessel, start, spacing, junction_native, *, source_graph_fork=False):
    native_shape = tuple(graph['source_shape'])
    fork = int(np.ravel_multi_index(junction_native, native_shape))
    branches = sorted((row for row in graph['branches']
                       if len(row['source_cell_ids']) > 1
                       and fork in (row['source_cell_ids'][0], row['source_cell_ids'][-1])),
                      key=lambda branch: branch['id'])
    if source_graph_fork:
        if len(branches) != 3:
            raise ValueError('The selected source anchor needs exactly three native graph legs')
        return graph_fork_routes(branches, vessel, start, spacing, native_shape, fork)
    if len(branches) != 3 or not all(all(row['edge_local_face_supported']) for row in branches):
        raise ValueError('The reviewed native junction needs three source-supported legs')
    paths = {}
    radii = distance_transform_edt(vessel, sampling=spacing)
    occupied = set()
    junction = tuple(junction_native - start)
    for row in branches:
        ids = row['source_cell_ids']
        original = np.column_stack(np.unravel_index(ids, native_shape))
        if ids[0] != fork:
            original = original[::-1]
        local = original - start
        if np.any(local < 0) or np.any(local >= vessel.shape):
            raise ValueError('A reviewed source link exceeds the native fork crop')
        path = bridge_steps(local, vessel, radii, occupied - {junction})
        occupied.update(map(tuple, path))
        paths[row['id']] = path
    return paths


def site_geometry(paths, vessel, score, affine):
    basis = affine[:3, :3]
    spacing = np.linalg.norm(basis, axis=0)
    score = np.where(vessel | (score < .5), score, 1. - score)
    sites = defaultdict(list)
    angles = np.linspace(0, 2*np.pi, 24, endpoint=False)
    distances = np.linspace(0., 3.25, 66)
    for route in paths.values():
        smooth = gaussian_filter(route.astype(float), (2., 0.))
        center = route + np.clip(smooth - route, -.35, .35)
        tangent = np.gradient(smooth @ basis.T, axis=0)
        tangent /= np.linalg.norm(tangent, axis=1, keepdims=True)
        for ijk, c, t in zip(route, center, tangent):
            orth = np.eye(3)[np.argmin(abs(t))]
            u = np.cross(t, orth)
            u /= np.linalg.norm(u)
            v = np.cross(t, u)
            directions = (np.cos(angles)[:, None] * u + np.sin(angles)[:, None] * v) @ np.linalg.inv(basis).T
            samples = c[:, None, None] + directions.T[:, :, None] * distances
            values = map_coordinates(score, samples, order=1, mode='constant')
            after = np.argmax(values < .5, axis=1)
            keep = (after > 0) & (values[:, 0] >= .5)
            r = np.median(distances[after[keep]]) if np.count_nonzero(keep) >= 12 else min(spacing)
            sites[tuple(ijk)].append([*(basis @ c + affine[:3, 3]), float(np.clip(r, .16, 2.4))])
    return {ijk: np.mean(values, axis=0) for ijk, values in sites.items()}


def modeled_exposure(image, vessel, paths, spacing):
    """Render a broad, curved teaching exposure around the TOF-supported fork."""
    i, j, k = np.ogrid[:image.shape[0], :image.shape[1], :image.shape[2]]
    fork = next(iter(paths.values()))[0]
    x, y = (i[:, :, 0] - fork[0])*spacing[0], (j[:, :, 0] - fork[1])*spacing[1]
    slab = slice(max(0, fork[2]-8), fork[2]+3)
    texture = gaussian_filter(np.median(image[:, :, slab], axis=2), sigma=2)
    lo, hi = np.percentile(texture, [10, 90])
    texture = np.clip((texture - lo) / max(hi-lo, 1e-6), 0, 1)
    # A curved recess divides the tissue banks while leaving broad parenchymal support.
    cleft = y - .36*x - .65*np.sin(x*.28)
    bank = 3.7 * (1.-np.exp(-(.22*cleft)**2))
    height = (fork[2] + (-1.9 - .28*x + bank + 1.5*(texture-.5)
              + .75*np.sin(x*.45 + y*.24) + .4*np.cos(y*.71-x*.4))/spacing[2])
    # Bring the modeled bank to the inferior wall along the exposed source traces.
    traced = np.concatenate(list(paths.values()))
    column_min = np.full(image.shape[:2], np.inf)
    for a, b, c in traced:
        floor = c
        while floor > 0 and vessel[a, b, floor-1]:
            floor -= 1
        column_min[a, b] = min(floor - 1, column_min[a, b])
    columns = np.isfinite(column_min)
    d, nearest = distance_transform_edt(~columns, sampling=spacing[:2], return_indices=True)
    infer_wall = gaussian_filter(column_min[tuple(nearest)], sigma=1.8)
    wall_weight = np.exp(-(d/3.7)**2)
    height = np.minimum(height + np.maximum(0, 1. - .23*d) * 5.,
                        (1.-wall_weight)*height + wall_weight*infer_wall)
    brain = (k <= height[:, :, None]) & ~vessel
    return brain


def spatial_properties(paths, geometry, brain, vessel, affine, motion, subject):
    spacing = np.linalg.norm(affine[:3, :3], axis=0)
    shape = vessel.shape
    def cell(ijk):
        return int(np.ravel_multi_index(ijk, shape))

    source_geometry = {str(cell(ijk)): row.tolist() for ijk, row in geometry.items()}
    supports = {}
    directions = np.vstack([np.eye(3, dtype=int), -np.eye(3, dtype=int)])
    distance, nearest = distance_transform_edt(~brain, sampling=spacing, return_indices=True)
    for ik in geometry:
        local = np.asarray(ik)
        if motion == 'baseline':
            positions = local + directions
            supported = [tuple(x) for x in positions if brain[tuple(x)]]
            if supported:
                supports[cell(ik)] = cell(min(supported, key=lambda x: x[2]))
        elif distance[ik] <= geometry[ik][-1] + .7*np.linalg.norm(spacing):
            support = tuple(nearest[(slice(None), *ik)])
            supports[cell(ik)] = cell(support)

    profile = research_profile()
    profile.update(
        vesselMotionModel=ROD_MODEL if motion == 'rods' else RETAINED_VESSEL_SEGMENTS,
        tractionFailureStrain=.85, assignment=ASSUMPTIONS,
        vesselAttachments=sorted([[a, b] for a, b in supports.items()]),
    )
    if motion == 'rods':
        profile['vesselRodGeometry'] = {
            'source': '24 radial interpolated native vesselFM score .5 crossings per source site.',
            'sites': source_geometry,
        }
    profile['axialStiffnessN']['uncertain'] = .04
    profile['effectiveLumenAreaMm2']['uncertain'] = 1.
    edges = []
    node_ids = set()
    for branch_id, path in paths.items():
        cells = [cell(point) for point in path]
        a, b = f'native-fork:{cells[0]}', f'native-end:{cells[-1]}'
        node_ids.update((a, b))
        edges.append({'id': branch_id, 'kind': 'uncertain', 'from': a, 'to': b,
                      'sourceCells': cells, 'sourceFace': cells[:2], 'conductanceMm3PerPaS': .0002,
                      'woundConductanceMm3PerPaS': [.00035, .00035], 'source': ASSUMPTIONS})
    graph = {
        'schema': 'resectionlab.vascular-graph.v1',
        'coverage': {'status': 'partial-acquired', 'sourceCaseId': subject,
                     'missing': ['The remainder of the vascular network', 'Vessel classification and wall',
                                 'Patient physiology and actual tissue support']},
        'provenance': {'kind': 'assumed-research', 'description': ASSUMPTIONS},
        'fieldPressurePa': 0.,
        'nodes': [{'id': name, 'complianceMm3PerPa': .002, 'initialStorageMm3': 0.,
                   'source': ASSUMPTIONS} for name in sorted(node_ids)],
        'edges': edges, 'boundaries': [], 'regions': [], 'compressionForceN': .04,
        'coagulationDose': 2., 'clotRatePerSecond': .02,
    }
    physics = default_profile()
    physics['brain'].update(radiusMm=1.6, dragNsPerMm=.02, holdStiffnessNPerMm=.20)
    return {'gliomaMaterialsV1': profile, 'tissuePhysicsV2': physics, 'vascularGraphV1': graph}


def graph_fork_view(junction, affine):
    basis = affine[:3, :3]
    target = (basis @ junction + affine[:3, 3]).tolist()
    # The modeled banks fill the low-K side of the native crop. Orient the
    # microscope toward the exposed side even on oblique or reflected grids.
    outward = np.cross(basis[:, 0], basis[:, 1])
    outward *= np.sign(outward @ basis[:, 2]) / np.linalg.norm(outward)
    return {'selectedSeriesId': None, 'states': {'primary': {
        'cursor': target, 'visibleLayers': {BRAIN: True, VESSEL: True},
        'viewer': {'microscope': {'target': target, 'outward': outward.tolist(), 'angleDeg': 18,
                                 'azimuthDeg': 0, 'workingDistanceMm': 160, 'fieldMm': 14,
                                 'focusOffsetMm': 0, 'light': 2.35}},
    }}}


def stage(root, output, motion, subject, junction, *, guard_native_wall=False, vertex_wall_transport=False,
          source_graph_fork=False):
    guard_native_wall = guard_native_wall or vertex_wall_transport
    if guard_native_wall and motion != 'rods':
        raise ValueError('A native wall sweep requires the score-bound rod model')
    image, score, vessel, graph, affine, start, receipt = load_source(
        root, subject, junction, source_graph_fork=source_graph_fork)
    spacing = np.linalg.norm(affine[:3, :3], axis=0)
    paths = source_routes(graph, vessel, start, spacing, junction, source_graph_fork=source_graph_fork)
    geometry = site_geometry(paths, vessel, score, affine)
    brain = modeled_exposure(image, vessel, paths, spacing)
    properties = spatial_properties(paths, geometry, brain, vessel, affine, motion, subject)
    if guard_native_wall:
        properties['gliomaMaterialsV1'][WALL_SWEEP_PROPERTY] = WALL_SWEEP_POLICY
    if vertex_wall_transport:
        properties['gliomaMaterialsV1'][VERTEX_MOTION_PROPERTY] = VERTEX_MOTION_POLICY
        receipt[VERTEX_MOTION_PROPERTY] = VERTEX_MOTION_POLICY
    receipt.update(
        sourceBranches={key: (route + start).tolist() for key, route in paths.items()},
        vesselSourceCells=int(vessel.sum()), modeledBrainCells=int(brain.sum()),
        assumptions=ASSUMPTIONS, vesselMotionModel=properties['gliomaMaterialsV1']['vesselMotionModel'],
        vesselSupportPairs=len(properties['gliomaMaterialsV1']['vesselAttachments']),
    )
    metadata = {
        'is_synthetic': True,
        'data_case': {'classification': 'synthetic_scenario_on_acquired_host',
                      'observations': {'primary_image': 'baseline_acquired_background_geometry_only'}},
        'anatomy_layers': {
            BRAIN: {'kind': 'brain', 'color': '#c79a88', 'provenance': 'simulated',
                    'coverage': 'Generated operative exposure banks with native TOF-derived texture'},
            VESSEL: {'kind': 'vessel', 'color': '#c65864', 'provenance': 'model_estimate',
                     'coverage': ('All source-predicted vessel cells including display exclusions'
                                  if 'hidden_candidate' in receipt['vesselMaskClasses'] else
                                  'Retained and uncertain source cells, with native retained vesselFM scores')},
        },
        'surgical_adapter': {'type': 'tof-local-engineering-vascular-crop',
                             'ownershipPrecedence': [BRAIN, VESSEL],
                             'materialByRegion': {BRAIN: 'soft_tissue', VESSEL: 'vessel'}},
        'surgical_properties': properties,
        'vascular_motion_source': receipt,
    }
    probability = {'vascular_probabilities': {VESSEL: score}} if motion == 'rods' else {}
    case = CaseData(case_id=subject + '-TOF-bifurcation-research-v2', mri=image,
                    compartments={BRAIN: brain, VESSEL: vessel}, affine=affine,
                    source_refs=(SourceRef(subject + '-TOF', 'https://zenodo.org/records/5714812',
                                           receipt['nativeSourceSha256'], native_frame='RAS+'),),
                    unknowns=(ASSUMPTIONS,), metadata=metadata, **probability)
    scene = from_case(case)
    if guard_native_wall:
        from resectionlab.surgical_engine import SurgicalEngine

        SurgicalEngine(scene)
    output.mkdir(exist_ok=False)
    path = output / 'native-tof-bifurcation.ressectionlab'
    if source_graph_fork:
        save_workspace(case, path, sources=[], imaging=graph_fork_view(junction - start, affine),
                       episode=None, selection=None, artifacts={})
    else:
        save_case(case, path)
    np.savez_compressed(output / 'research-layout.npz', brain=brain, vessel=vessel, affine=affine,
                        native_crop_start=start, **{key: route for key, route in paths.items()})
    write(output / 'scene-source.json', {**receipt, 'caseHash': case.semantic_hash, 'sceneId': scene.identity,
                                        'caseFileSha256': sha256(path)})
    return path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('output', type=Path)
    parser.add_argument('--root', type=Path, default=HDD)
    parser.add_argument('--motion', choices=['baseline', 'rods'], default='rods')
    parser.add_argument('--guard-native-wall', action='store_true', help='Certify score-wall clearance during rod motion')
    parser.add_argument('--vertex-wall-transport', action='store_true',
                        help='Transport the guarded score wall at individual vertices')
    parser.add_argument('--source-graph-fork', action='store_true',
                        help='Stage a research fork from the verified native graph without a reviewed region')
    parser.add_argument('--subject', required=True)
    parser.add_argument('--junction-native', type=int, nargs=3, metavar=('I', 'J', 'K'), required=True)
    args = parser.parse_args()
    if not re.fullmatch(r'PTNT_\d{4}', args.subject):
        parser.error('Use a restricted EDEN2020 patient ID')
    root = checked_root(args.root, expected_uuid=UUID)
    output = args.output.resolve()
    if not output.is_relative_to(root / CURRENT / 'operative-scene'):
        parser.error('Write restricted derivatives below the verified campaign operative-scene folder')
    print(stage(root, output, args.motion, args.subject, np.asarray(args.junction_native),
                guard_native_wall=args.guard_native_wall, vertex_wall_transport=args.vertex_wall_transport,
                source_graph_fork=args.source_graph_fork), flush=True)


if __name__ == '__main__':
    main()
