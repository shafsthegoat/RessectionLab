"""Bounded intact-source fans for the existing staged controller.

Caller supplies routes extracted from sealed comparisons in the CURRENT record.
This module performs no I/O, source discovery, patient-role selection, native
engine construction or transition. Contact ledgers are proposals, never removal.
"""
from __future__ import annotations

import copy
import math

import numpy as np
from scipy.spatial.transform import Rotation

from .core import array_digest, semantic_digest
from .geometry import GeometryScene, capsule_voxel_indices

SCHEMA = 'resectionlab.source-initial-fan-inventory.v1'
RADII_MM = (.6, 1., 1.2, 1.5)
MAX_ROUTES = 12
MAX_RAY_CONFIGURATIONS = 60000
MAX_STAGES = 12


def _geometry(route):
    # Same seal and candidate-ID basis as the frozen source fan inventory.
    return semantic_digest({k: route[k] for k in
        ('surfacePose', 'entryPose', 'inwardRasUnit')})


def _physical_geometry(route):
    # CUSA roll and quaternion sign are not different entry geometries.
    return semantic_digest({
        'surface': route['surfacePose']['position'],
        'entry': route['entryPose']['position'],
        'inward': route['inwardRasUnit']})


def _pose(pose):
    p = np.asarray(pose['position'], float)
    q = np.asarray(pose['quaternion'], float)
    if (p.shape != (3,) or q.shape != (4,) or
            not np.isfinite(p).all() or not np.isfinite(q).all() or
            abs(np.linalg.norm(q)-1.) > 1e-5):
        raise ValueError('INVALID_FINITE_POSE')
    return p, q


def _valid_route(scene, route):
    if route.get('sourceSceneId') != scene.identity:
        raise ValueError('SOURCE_SCENE_MISMATCH')
    surface, sq = _pose(route['surfacePose'])
    entry, eq = _pose(route['entryPose'])
    outside, oq = _pose(route['outsidePose'])
    axis = np.asarray(route['inwardRasUnit'], float)
    length = route['pathLengthMm']
    if (axis.shape != (3,) or not np.isfinite(axis).all() or
            abs(np.linalg.norm(axis)-1.) > 1e-6 or
            not isinstance(length, (int, float)) or not np.isfinite(length) or
            not 0 < length <= 38. or
            np.linalg.norm(entry-surface-length*axis) > 1e-5 or
            np.dot(outside-surface, axis) >= 0 or
            any(np.linalg.norm(Rotation.from_quat(q).apply([0., 0., 1.])-axis)
                > 1e-5 for q in (sq, eq, oq))):
        raise ValueError('INVALID_SOURCE_RAY')
    if type(route.get('nativeAxis')) is not int or route['nativeAxis'] not in (0, 1, 2):
        raise ValueError('INVALID_NATIVE_AXIS')
    return surface, entry, outside, axis


def _ray_spec(surface, endpoint, radius, *, outside=None, quaternion=None):
    axis = endpoint-surface
    length = float(np.linalg.norm(axis))
    axis = axis/length
    if outside is None:
        outside = surface-8.*axis
    if quaternion is None:
        quaternion = Rotation.align_vectors([axis], [[0., 0., 1.]])[0].as_quat().tolist()
    distance = float(np.linalg.norm(endpoint-outside))
    count = max(1, math.ceil(distance/.5))
    path = [{'position': (outside+(endpoint-outside)*k/count).tolist(),
             'quaternion': list(quaternion)} for k in range(1, count+1)]
    return {'radiusMm': radius, 'depthMm': 2.,
        'outsidePose': {'position': outside.tolist(), 'quaternion': list(quaternion)},
        'endpointPose': {'position': endpoint.tolist(), 'quaternion': list(quaternion)},
        'inwardRasUnit': axis.tolist(), 'entryPath': path,
        'entryDurationSeconds': distance/2., 'dwellDurationSeconds': .6,
        'proximalEndPoseRasMm': (endpoint-40.*axis).tolist(),
        'externalDepthMm': length, 'configurationChangesOnlyOutside': True}


def _fan_rank(fan):
    return (-fan['cumulativeTargetMm3'], fan['cumulativeBrainMm3'],
            fan['routeGeometrySeal'], fan['initialRadiusMm'], fan['fullSpecSeal'])


def _summary(fan):
    keys = ('id', 'routeId', 'routeGeometrySeal', 'routePhysicalGeometrySeal',
        'sourceRouteSeal', 'originalSourceRouteSeal', 'sourceContactSeal', 'initialRadiusMm', 'initialTargetMm3',
        'initialBrainMm3', 'cumulativeTargetMm3', 'cumulativeBrainMm3',
        'additionalBrainFromInitialMm3', 'orderedRayCount', 'fullSpecSeal',
        'candidateBankComplete')
    return {key: copy.deepcopy(fan[key]) for key in keys}


def propose_source_fans(scene, routes, *, check=lambda: None, max_stages=12):
    """Propose at most three distinct source-bound initial-access/reentry fans.

    `routes` are raw route dictionaries, supplied ONLY by the caller's sealed
    CURRENT-record comparisons. The caller owns comparison-seal validation;
    sourceSceneId, finite rigid geometry and original entry binding are checked
    here. There is no saved-state or patient discovery in this pure function.

    `fans` holds at most three full selected programs. `fanSummaries` contains
    all viable original-radius variants in the same cumulative source ranking;
    it deliberately omits their cell ledgers and paths. Source contacts never
    certify native shaft clearance, guards, execution, withdrawal or task gates.
    """
    if not isinstance(routes, (tuple, list)) or len(routes) > MAX_ROUTES:
        raise ValueError('At most12 caller-supplied sealed source routes')
    if type(max_stages) is not int or not 1 <= max_stages <= MAX_STAGES:
        raise ValueError('max_stages must be in1..12')
    if not callable(check):
        raise TypeError('check must be callable')
    check()
    labels = scene.labels.ravel()
    target_ids = {int(r['id']) for r in scene.regions
        if r.get('target') and r['material'] == 'tumor'}
    brain_ids = {int(r['id']) for r in scene.regions
        if r['material'] == 'soft_tissue' and 'brain' in r['name'].lower()}
    allowed = target_ids | brain_ids
    source_indices = np.argwhere(np.isin(scene.labels, list(target_ids)))
    source_flat = np.ravel_multi_index(source_indices.T, scene.shape)
    points = source_indices@scene.affine[:3, :3].T+scene.affine[:3, 3]
    grid = GeometryScene(np.zeros(scene.shape, bool), scene.affine)
    counts = {'inputRoutes': len(routes), 'distinctRouteGeometries': 0,
        'rayConfigurationsExamined': 0, 'initialConfigurationsExamined': 0,
        'reentryConfigurationsExamined': 0, 'sourceEligibleReentries': 0,
        'storedUsefulReentries': 0, 'viableFans': 0, 'selectedFans': 0}
    output = {'schema': SCHEMA, 'sourceSceneId': scene.identity,
        'sourceLabelsHash': array_digest(scene.labels), 'nativeSelectionCalls': 0,
        'nativePreviewCalls': 0, 'nativeStepCalls': 0,
        'sourceOnlySelection': True, 'maxInputRoutes': MAX_ROUTES,
        'maxRayConfigurations': MAX_RAY_CONFIGURATIONS, 'maxStages': max_stages,
        'workingRadiiMm': list(RADII_MM), 'workingDepthMm': 2.,
        'shaftRadiusMm': .5, 'shaftLengthMm': 40.,
        'externalShaftWorkingReachMm': 38., 'counts': counts,
        'routeSummaries': [], 'fanSummaries': [], 'fans': [],
        'inputRouteFailures': [], 'selectionRejections': [], 'sourceRoutesSeals': [],
        'candidateBudgetExhausted': False,
        'ranking': 'cumulative whole source target descending,whole source brain ascending,route geometry seal,initial radius,full spec seal',
        'greedyRanking': 'marginal target/max(marginal brain,voxel volume),target,brain,content ID',
        'scope': 'Intact-source virtual tip-contact unions only. Original initial-tip brain IDs fix additional20/whole200 allowance. Full shaft source labels and outside placement are tested; native swept clearance and guards remain authoritative.'}

    def touched(start, end, radius):
        check()
        cover = capsule_voxel_indices(grid, start, end, radius)
        flat = np.ravel_multi_index(cover.T, scene.shape)
        return {int(i) for i in flat if labels[i]}

    def split(cells):
        return ({i for i in cells if int(labels[i]) in target_ids},
                {i for i in cells if int(labels[i]) in brain_ids})

    def charge(initial=False):
        check()
        if counts['rayConfigurationsExamined'] >= MAX_RAY_CONFIGURATIONS:
            output['candidateBudgetExhausted'] = True
            return False
        counts['rayConfigurationsExamined'] += 1
        key = 'initialConfigurationsExamined' if initial else 'reentryConfigurationsExamined'
        counts[key] += 1
        return True

    if 'cusaV3' in scene.properties or not len(source_indices):
        output['status'] = 'unsupported_fixed_cusa_v3' if 'cusaV3' in scene.properties else 'no_source_target'
        output['inputRouteFailures'].append({'routeId': None, 'reason': output['status'].upper()})
        output['summary'] = {'selectedRouteCount': 0, 'initialRadiusFanCount': 0, 'rayConfigurationsExamined': 0, 'candidateBudgetExhausted': False, 'selectedContactOutcomes': []}
        output['inventorySeal'] = semantic_digest(output)
        return output

    # Canonical input ordering makes permutation and duplicate IDs irrelevant.
    unique = {}
    raw_routes = []
    for route in routes:
        if not isinstance(route, dict):
            output['inputRouteFailures'].append({'routeId': None, 'reason': 'INVALID_ROUTE_DICTIONARY'})
        else:
            raw_routes.append(route)
    for route in sorted(raw_routes, key=semantic_digest):
        check()
        try:
            _valid_route(scene, route)
            physical = _physical_geometry(route)
            if physical in unique:
                output['inputRouteFailures'].append({'routeId': route.get('id'),
                    'reason': 'DUPLICATE_PHYSICAL_ROUTE', 'routePhysicalGeometrySeal': physical})
                continue
            unique[physical] = copy.deepcopy(route)
        except (ValueError, TypeError, KeyError) as error:
            output['inputRouteFailures'].append({'routeId': route.get('id')
                if isinstance(route, dict) else None, 'reason': str(error)})
    counts['distinctRouteGeometries'] = len(unique)
    best_per_route = []
    for physical, route in sorted(unique.items(), key=lambda item: (_geometry(item[1]), semantic_digest(item[1]))):
        check()
        surface, entry, outside, inward = _valid_route(scene, route)
        geometry_seal = _geometry(route)
        route_seal = semantic_digest(route)
        output['sourceRoutesSeals'].append({'routeId': route.get('id'), 'routeGeometrySeal': geometry_seal, 'sourceRouteSeal': route_seal})
        summary = {'routeId': route.get('id'), 'routeGeometrySeal': geometry_seal,
            'routePhysicalGeometrySeal': physical, 'sourceRouteSeal': route_seal,
            'candidateLocations': 0, 'candidateBankComplete': True,
            'initialConfigurationFailures': [],
            'rejections': {'externalAnchor': 0, 'outsideOccupied': 0,
                'unsupportedShaftOrTipLabels': 0, 'noInitialBudgetCouldUseRay': 0}}
        output['routeSummaries'].append(summary)
        initial = []
        for radius in RADII_MM:
            if not charge(initial=True):
                summary['candidateBankComplete'] = False
                break
            tip = touched(outside, entry, radius)
            shaft = touched(outside-40.*inward, entry-2.*inward, .5)
            targets, brains = split(tip)
            reason = None
            if touched(outside-40.*inward, outside, max(radius, .5)):
                reason = 'INITIAL_OUTSIDE_OCCUPIED'
            elif (entry-40.*inward-surface)@inward >= 0:
                reason = 'INITIAL_NOT_EXTERNALLY_ANCHORED'
            elif any(int(labels[i]) not in allowed for i in shaft | tip):
                reason = 'INITIAL_UNSUPPORTED_LABEL'
            elif len(brains)*scene.voxel_mm3 > 200.+1e-9:
                reason = 'INITIAL_BRAIN_OVER200'
            if reason:
                summary['initialConfigurationFailures'].append({'radiusMm': radius,
                    'reason': reason, 'sourceTipBrainMm3': len(brains)*scene.voxel_mm3,
                    'sourceTipTargetMm3': len(targets)*scene.voxel_mm3})
            else:
                initial.append({'radius': radius, 'tip': tip, 'targets': targets,
                                'brains': brains})
        if not initial:
            summary['status'] = 'NO_SOURCE_ELIGIBLE_INITIAL_CONFIGURATION'
            continue
        relative = points-entry
        depth = relative@inward
        radial = np.linalg.norm(relative-depth[:, None]*inward, axis=1)
        keep = np.linalg.norm(points-surface, axis=1) <= 38.+1e-9
        ps, fs, ds, rs = points[keep], source_flat[keep], depth[keep], radial[keep]
        locations = {}
        representatives = {}
        if len(ps):
            for i, bucket in enumerate(np.floor((ps-ps.min(axis=0))/3.).astype(int)):
                check()
                key = tuple(map(int, bucket))
                rank = (float(rs[i]), float(ds[i]), tuple(ps[i]))
                if key not in representatives or rank < representatives[key][0]:
                    representatives[key] = (rank, i)
            locations.update({tuple(ps[i]): {'kind': 'source3mmLattice',
                'sourceCellId': int(fs[i])} for _, i in representatives.values()})
        for offset in np.arange(0., max(0., 38.-route['pathLengthMm'])+1e-9, 1.):
            check()
            point = entry+inward*offset
            idx = np.rint(scene.inverse[:3, :3]@(point-scene.affine[:3, 3])).astype(int)
            if np.all(idx >= 0) and np.all(idx < scene.shape):
                flat = int(np.ravel_multi_index(idx, scene.shape))
                if int(labels[flat]) in target_ids and np.linalg.norm(point-surface) <= 38.+1e-9:
                    locations[tuple(point)] = {'kind': 'originalRay1mm', 'sourceCellId': flat}
        summary['candidateLocations'] = len(locations)
        candidates = []
        for coords, location in sorted(locations.items()):
            check()
            endpoint = np.array(coords)
            axis = endpoint-surface
            length = float(np.linalg.norm(axis))
            if not 0 < length <= 38.+1e-9:
                continue
            axis /= length
            ray_outside = surface-8.*axis
            anchored = (endpoint-40.*axis-surface)@inward < -1e-9
            shaft = touched(ray_outside-40.*axis, endpoint-2.*axis, .5) if anchored else set()
            for radius in RADII_MM:
                if not charge():
                    summary['candidateBankComplete'] = False
                    break
                if not anchored:
                    summary['rejections']['externalAnchor'] += 1
                    continue
                if touched(ray_outside-40.*axis, ray_outside, max(radius, .5)):
                    summary['rejections']['outsideOccupied'] += 1
                    continue
                tip = touched(ray_outside, endpoint, radius)
                if any(int(labels[i]) not in allowed for i in shaft | tip):
                    summary['rejections']['unsupportedShaftOrTipLabels'] += 1
                    continue
                counts['sourceEligibleReentries'] += 1
                targets, brains = split(tip)
                # Safe storage pruning: no shared prior ledger can undo these
                # ORIGINAL initial-cut constraints. Greedy outcome is unchanged.
                if not targets or not any(len(base['brains'] | brains)*scene.voxel_mm3 <= 200.+1e-9
                        and len(brains-base['brains'])*scene.voxel_mm3 <= 20.+1e-9 for base in initial):
                    summary['rejections']['noInitialBudgetCouldUseRay'] += 1
                    continue
                candidates.append({'id': 'fan-ray-'+semantic_digest({'scene': scene.identity,
                    'routeGeometry': geometry_seal, 'endpoint': list(coords), 'radius': radius}),
                    'radius': radius, 'location': location, 'endpoint': endpoint,
                    'tip': tip, 'shaft': shaft, 'targets': targets, 'brains': brains})
                counts['storedUsefulReentries'] += 1
            if output['candidateBudgetExhausted']:
                break
        route_fans = []
        for base in initial:
            check()
            original_brain = base['brains']
            virtual = set(base['tip'])
            selected_ids = set()
            ordered = []
            for stage in range(max_stages):
                vt, vb = split(virtual)
                winner = None
                for candidate in candidates:
                    check()
                    if candidate['id'] in selected_ids:
                        continue
                    mt = candidate['targets']-vt
                    mb = candidate['brains']-vb
                    if not mt or len(vb | candidate['brains'])*scene.voxel_mm3 > 200.+1e-9 or len((vb | candidate['brains'])-original_brain)*scene.voxel_mm3 > 20.+1e-9:
                        continue
                    rank = (-len(mt)/max(len(mb), 1), -len(mt), len(mb), candidate['id'])
                    if winner is None or rank < winner[0]:
                        winner = (rank, candidate, mt, mb)
                if winner is None:
                    break
                _, candidate, mt, mb = winner
                selected_ids.add(candidate['id'])
                virtual |= candidate['tip']
                vt, vb = split(virtual)
                ordered.append({'id': candidate['id'], 'stage': stage+1,
                    'raySpec': _ray_spec(surface, candidate['endpoint'], candidate['radius']),
                    'location': copy.deepcopy(candidate['location']),
                    'marginalTargetMm3': len(mt)*scene.voxel_mm3,
                    'marginalBrainMm3': len(mb)*scene.voxel_mm3,
                    'cumulativeTargetMm3': len(vt)*scene.voxel_mm3,
                    'cumulativeBrainMm3': len(vb)*scene.voxel_mm3,
                    'additionalBrainFromInitialMm3': len(vb-original_brain)*scene.voxel_mm3,
                    'tipSourceCellIds': sorted(candidate['tip']),
                    'tipCellsSeal': semantic_digest(sorted(candidate['tip'])),
                    'marginalTargetSourceCellIds': sorted(mt),
                    'shaftSourceCellIds': sorted(candidate['shaft']),
                    'shaftCellsSeal': semantic_digest(sorted(candidate['shaft']))})
            vt, vb = split(virtual)
            proposed_route = copy.deepcopy(route)
            proposed_route['corridorRadiusMm'] = base['radius']
            fan = {'routeId': route.get('id'), 'routeGeometrySeal': geometry_seal,
                'routePhysicalGeometrySeal': physical, 'sourceRouteSeal': semantic_digest(proposed_route),
                'originalSourceRouteSeal': route_seal,
                'sourceSceneId': scene.identity, 'initialRadiusMm': base['radius'],
                'initialTargetMm3': len(base['targets'])*scene.voxel_mm3,
                'initialBrainMm3': len(original_brain)*scene.voxel_mm3,
                'initialTipSourceCellIds': sorted(base['tip']),
                'initialTipCellsSeal': semantic_digest(sorted(base['tip'])),
                'initialBrainSourceCellIds': sorted(original_brain),
                'initialRaySpec': _ray_spec(surface, entry, base['radius'], outside=outside,
                    quaternion=route['entryPose']['quaternion']),
                'route': proposed_route, 'originalRoute': copy.deepcopy(route),
                'orderedRays': ordered, 'orderedRayCount': len(ordered),
                'cumulativeTargetMm3': len(vt)*scene.voxel_mm3,
                'cumulativeBrainMm3': len(vb)*scene.voxel_mm3,
                'additionalBrainFromInitialMm3': len(vb-original_brain)*scene.voxel_mm3,
                'virtualRemovedSourceCellIds': sorted(virtual),
                'virtualLedgerSeal': semantic_digest(sorted(virtual)),
                'candidateBankComplete': summary['candidateBankComplete']}
            fan['sourceContactSeal'] = semantic_digest({'sourceSceneId': scene.identity, 'sourceRouteSeal': fan['sourceRouteSeal'], 'initialTipCellsSeal': fan['initialTipCellsSeal'], 'orderedTipCellsSeals': [r['tipCellsSeal'] for r in ordered], 'virtualLedgerSeal': fan['virtualLedgerSeal']})
            fan['fullSpecSeal'] = semantic_digest(fan)
            fan['id'] = 'source-initial-fan-'+fan['fullSpecSeal']
            route_fans.append(fan)
            output['fanSummaries'].append(_summary(fan))
        counts['viableFans'] += len(route_fans)
        summary['status'] = 'SOURCE_FANS_PROPOSED'
        if route_fans:
            best_per_route.append(min(route_fans, key=_fan_rank))
        # No unselected full-radius candidate banks survive this route iteration.
    output['fanSummaries'].sort(key=_fan_rank)
    for fan in sorted(best_per_route, key=_fan_rank):
        check()
        route = fan['route']
        reason = None
        comparisons = []
        for previous in output['fans']:
            prior = previous['route']
            separation = float(np.linalg.norm(np.subtract(route['surfacePose']['position'], prior['surfacePose']['position'])))
            angle = float(np.degrees(np.arccos(np.clip(np.dot(route['inwardRasUnit'], prior['inwardRasUnit']), -1., 1.))))
            comparisons.append({'routeId': prior.get('id'), 'entrySeparationMm': separation, 'inwardAngleDeg': angle})
            if route['nativeAxis'] == prior['nativeAxis']:
                reason = 'NATIVE_AXIS_ALREADY_SELECTED'
            elif separation+1e-9 < 12. or angle+1e-9 < 30.:
                reason = 'SOURCE_GEOMETRY_NOVELTY'
        if reason or len(output['fans']) >= 3:
            output['selectionRejections'].append({'fanId': fan['id'],
                'routeGeometrySeal': fan['routeGeometrySeal'],
                'reason': reason or 'THREE_ROUTE_LIMIT', 'comparisons': comparisons})
            continue
        fan['noveltyComparisons'] = comparisons
        output['fans'].append(fan)
    selected_ids = {fan['id'] for fan in output['fans']}
    for summary in output['fanSummaries']:
        summary['selected'] = summary['id'] in selected_ids
    counts['selectedFans'] = len(output['fans'])
    output['status'] = 'source_fans_proposed' if output['fans'] else 'no_source_fans'
    output['summary'] = {'selectedRouteCount': counts['selectedFans'], 'initialRadiusFanCount': counts['viableFans'], 'rayConfigurationsExamined': counts['rayConfigurationsExamined'], 'candidateBudgetExhausted': output['candidateBudgetExhausted'], 'selectedContactOutcomes': [{'fanId': f['id'], 'cumulativeTargetMm3': f['cumulativeTargetMm3'], 'cumulativeBrainMm3': f['cumulativeBrainMm3'], 'additionalBrainFromInitialMm3': f['additionalBrainFromInitialMm3']} for f in output['fans']]}
    output['inventorySeal'] = semantic_digest(output)
    return output
