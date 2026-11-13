"""Bounded, current-cavity stock contact proposals; never opening admission.

Only original affine geometry, native removal and current tagged debris are used. The caller
supplies an instrument quaternion and exterior approach plane in this Scene;
entry/source normals are constraints, never inferred instrument attitudes.
The shortlist is deliberately conservative: shafts must clear the CURRENT
cavity, while suction working contact is conditional on its preceding native
drill removing the complete proposed footprint. Native checks, finite work,
debris collection, exit and the existing shared meter remain mandatory.
Measured wall diagnostics are excluded from deterministic proposal/order hashes.
"""
from __future__ import annotations

import copy
import math
import time
from itertools import islice, product

import numpy as np
from scipy.spatial.transform import Rotation, Slerp

from .core import semantic_digest
from .geometry import GEOMETRY_VERSION, _segment_cell_distances
from .surgical_craniotomy import _original_targets
from .surgical_cranial_guards import _inside_entry
from .surgical_engine import SurgicalEngine
from .surgical_milling import is_milling, milling_readiness, validate_source_scope
from .surgical_motion import envelopes, pair_conflict, swept_envelopes
from .surgical_tools import TOOLS
from .surgical_transition_meter import MeteredEngine

SCHEMA = 'resectionlab.milling-contact-proposals.v1'
APPROACH_SCHEMA = 'resectionlab.milling-contact-approach.v1'
STAGING_SCHEMA = 'resectionlab.milling-staging-plan.v1'
_OVERLAP_MM = .10
_POINT_SCHEMES = ('center', 'center_and_quarter_cell_corners')
_CELL_SCHEDULES = ('surface_first', 'axial_diverse_current_a')
_CONTACT_POSE_FAMILIES = ('leading_touch', 'cavity_carried_rays')
_CAVITY_RAY_ALGORITHM = {'name': 'current-verified-a-cavity-carried-rays-v1',
    'carrierDistances': ['0', 'stockRadius', 'stockRadius+cellCircumsphere'],
    'carrierFractions': [-.375, .375], 'tieRoundingDecimals': 10,
    'reservedProbesPerTarget': 36, 'maximumTargets': 56,
    'activeCommandMm': .20, 'overlapMm': .10,
    'passiveMoveSeconds': 'explicit caller staging_seconds; exact native one-leg subdivision',
    'dwell': 'separate stationary apply at the same caller-declared drill_seconds'}
_AXIAL_ALGORITHM = {'name': 'source-maximin-current-a-axial-v2', 'columns': 56,
    'levelsMm': [0., 1., 2., 3.], 'chunkCells': 1024, 'tieRoundingDecimals': 9,
    'scope': 'bone-only; existing inverse and current-cavity guards unchanged'}
_ALGORITHM = {'name': 'native-orthogonal-cell-ray-inverse-v1',
    'kernel': 'geometry._segment_cell_distances', 'iterations': 64,
    'bracket': '[0, stockRadius+cellCircumsphere+anchorOffsetNorm+1e-6] mm',
    'pose': 'declared interior cell point-axis*(nativeLeadingContactDistance-0.10mm)',
    'pointSchemes': list(_POINT_SCHEMES),
    'quarterCellCorners': 'center followed by all eight affine quarter-cell corners; fixed source geometry, not tool shrinkage',
    'candidateOrder': 'all-A outward-axis depth and source ID, computed once at explicit approach binding',
    'suction': 'independent stock-radius inversion; no universal .5mm offset',
    'shearedAffine': 'unsupported; native kernel is only a distance lower bound'}
_DEBRIS_ALGORITHM = {'name': 'current-tagged-milling-debris-contact-v1',
    'contact': _ALGORITHM['name'], 'selection': 'actual tagged spatial debris volume, then source cell ID',
    'nativeRecipe': 'approach apply then stationary apply, each at the caller-declared suction duration; exterior withdrawal',
    'clearance': 'current native cavity only; no prospective removal permitted'}
_ORDERING_COST_FIELDS = frozenset(('orderingWallSeconds', 'bindingWallSeconds'))


def _ordering_hash(ordering):
    return semantic_digest({key: value for key, value in ordering.items() if key not in _ORDERING_COST_FIELDS})


def proposal_hash(result):
    """Hash source/state-bound proposal data, excluding measured wall diagnostics."""
    payload = copy.deepcopy(result)
    payload.pop('contentHash', None)
    payload.pop('sourceGeometryCost', None)
    for name in _ORDERING_COST_FIELDS:
        payload['approach']['sourceOrdering'].pop(name, None)
    return semantic_digest(payload)


def _finite_positive(value):
    return type(value) in (int, float) and math.isfinite(value) and value > 0


def _approach_declaration(approach):
    value = {key: copy.deepcopy(approach[key]) for key in (
        'sceneId', 'contractHash', 'quaternion', 'outsidePlanePointRasMm', 'maximumApproachMm')}
    if 'cellSchedule' in approach:
        value['cellSchedule'] = approach['cellSchedule']
    if 'contactPoseFamily' in approach:
        value['contactPoseFamily'] = approach['contactPoseFamily']
    return value


def _geometry_check(check):
    if check is not None:
        check()


def _axial_source_binding(engine, contract, approach):
    value = {'sceneId': engine.scene.identity, 'contractHash': contract.content_hash,
        'originalAHash': semantic_digest(list(contract.aperture_cells)),
        'affineHash': semantic_digest(engine.scene.affine.tolist()),
        'quaternion': list(approach['quaternion']), 'pointScheme': approach.get('pointScheme', 'center'),
        'cellSchedule': approach['cellSchedule'], 'algorithmHash': semantic_digest(_AXIAL_ALGORITHM),
        'geometryVersion': GEOMETRY_VERSION,
        'stockCatalogHash': semantic_digest({name: TOOLS[name] for name in ('cranial_drill', 'frazier_suction')})}
    if 'contactPoseFamily' in approach:
        value.update(contactPoseFamily=approach['contactPoseFamily'],
                     contactPoseAlgorithmHash=semantic_digest(_CAVITY_RAY_ALGORITHM))
    return value


def _axial_columns(cells, coordinates, check):
    """Fixed source maximin/Voronoi work; temporary distances are <=1024x56."""
    count, chunk = len(cells), _AXIAL_ALGORITHM['chunkCells']
    ids = np.asarray(cells, dtype=np.int64)
    uv = coordinates[:, :2]
    center = (uv.min(0)+uv.max(0))/2
    radial = np.sum((uv-center)**2, axis=1)
    current = min(range(count), key=lambda j: (round(float(radial[j]), 9), int(ids[j])))
    nearest = np.full(count, np.inf)
    seeds, pairs = [], 0
    for _ in range(min(56, count)):
        seeds.append(current)
        for start in range(0, count, chunk):
            _geometry_check(check)
            stop = min(start+chunk, count)
            nearest[start:stop] = np.minimum(nearest[start:stop],
                np.sum((uv[start:stop]-uv[current])**2, axis=1))
            pairs += stop-start
        rounded = np.round(nearest, 9)
        maximum = float(rounded.max())
        if maximum <= 0 or len(seeds) == 56:
            break
        current = min(np.flatnonzero(rounded == maximum), key=lambda j: int(ids[j]))
    seed_uv = uv[seeds]
    rank = np.argsort(ids[seeds])  # Equal-distance columns use source-ID ties.
    assignments = np.empty(count, dtype=np.int64)
    for start in range(0, count, chunk):
        _geometry_check(check)
        stop = min(start+chunk, count)
        squared = np.sum((uv[start:stop, None]-seed_uv[None, rank])**2, axis=2)
        assignments[start:stop] = rank[np.round(squared, 9).argmin(axis=1)]
    return [int(ids[j]) for j in seeds], assignments.tolist(), {
        'initialSeedDistancePairs': count, 'maximinDistancePairs': pairs,
        'voronoiDistancePairs': count*len(seeds), 'maximumDistanceChunkCells': min(chunk, count)}


def _build_axial_cache(engine, contract, approach, cells, points, check):
    if engine.grid._orthogonal_spacing is None:
        raise ValueError('MILLING_CONTACT_SHEARED_AFFINE_CONTACT_UNSUPPORTED')
    basis = Rotation.from_quat(approach['quaternion']).as_matrix()
    coordinates = points@basis
    seeds, assignments, cost = _axial_columns(cells, coordinates, check)
    return {'binding': _axial_source_binding(engine, contract, approach),
        'projectedCoordinates': coordinates.tolist(), 'seedCells': seeds,
        'columnAssignments': assignments, 'preparationCost': {
            'sourceCoordinates': len(cells), **cost}}


def _validate_axial_cache(engine, contract, approach, check):
    ordering = approach['sourceOrdering']
    cache = ordering['axialCache']
    fields = {'binding', 'projectedCoordinates', 'seedCells', 'columnAssignments', 'preparationCost'}
    cells = ordering['orderedCells']
    source_a = set(contract.aperture_cells)
    if (type(cache) is not dict or set(cache) != fields or
            cache['binding'] != _axial_source_binding(engine, contract, approach) or
            type(cache['seedCells']) is not list or not 1 <= len(cache['seedCells']) <= 56 or
            any(type(c) is not int or c not in source_a for c in cache['seedCells']) or
            len(set(cache['seedCells'])) != len(cache['seedCells']) or
            type(cache['projectedCoordinates']) is not list or len(cache['projectedCoordinates']) != len(cells) or
            type(cache['columnAssignments']) is not list or len(cache['columnAssignments']) != len(cells) or
            any(type(c) is not int or not 0 <= c < len(cache['seedCells']) for c in cache['columnAssignments'])):
        raise ValueError('MILLING_CONTACT_AXIAL_SOURCE_CACHE_BINDING_REQUIRED')
    try:
        coordinates = np.asarray(cache['projectedCoordinates'], dtype=float)
    except (ValueError, TypeError):
        raise ValueError('MILLING_CONTACT_AXIAL_SOURCE_COORDINATES_DIFFER') from None
    if coordinates.shape != (len(cells), 3) or not np.isfinite(coordinates).all():
        raise ValueError('MILLING_CONTACT_AXIAL_SOURCE_COORDINATES_DIFFER')
    basis = Rotation.from_quat(approach['quaternion']).as_matrix()
    for start in range(0, len(cells), _AXIAL_ALGORITHM['chunkCells']):
        _geometry_check(check)
        actual = engine.scene.points(cells[start:start+1024])@basis
        if not np.allclose(actual, coordinates[start:start+1024], atol=1e-9, rtol=0):
            raise ValueError('MILLING_CONTACT_AXIAL_SOURCE_COORDINATES_DIFFER')
    return cache, coordinates


def _axial_current_cells(engine, contract, approach, remaining, max_cells, max_probes, check,
                         *, probe_slots=None):
    cache, coordinates = _validate_axial_cache(engine, contract, approach, check)
    groups = [[] for _ in cache['seedCells']]
    cells = approach['sourceOrdering']['orderedCells']
    for start in range(0, len(cells), 1024):
        _geometry_check(check)
        for j in range(start, min(start+1024, len(cells))):
            if cells[j] in remaining:
                groups[cache['columnAssignments'][j]].append(j)
    lookup = {cell: j for j, cell in enumerate(cells)}
    axis = Rotation.from_quat(approach['quaternion']).apply([0., 0., 1.])
    support = .5*float(np.abs(engine.scene.affine[:3, :3].T@axis).sum())
    slots, used = [], set()
    for column, members in enumerate(groups):
        _geometry_check(check)
        # A fully cleared CURRENT column has no candidate to reserve. Keep
        # missing/duplicate depth slots inside active columns bounded below.
        if not members:
            continue
        outer = min(float(coordinates[j, 2]) for j in members)
        seed = coordinates[lookup[cache['seedCells'][column]], :2]
        for level in _AXIAL_ALGORITHM['levelsMm']:
            _geometry_check(check)
            eligible = [j for j in members if abs(float(coordinates[j, 2])-outer-level) <= support+1e-9]
            chosen = min(eligible, key=lambda j: (
                round(abs(float(coordinates[j, 2])-outer-level), 9),
                round(float(np.sum((coordinates[j, :2]-seed)**2)), 9), cells[j])) if eligible else None
            cell = cells[chosen] if chosen is not None else None
            reason = 'MISSING_CURRENT_A_LEVEL' if cell is None else (
                'DUPLICATE_CURRENT_A_CELL' if cell in used else None)
            slots.append({'column': column, 'levelMm': level, 'cell': cell, 'reason': reason})
            if cell is not None:
                used.add(cell)
    anchors = probe_slots or (9 if approach.get('pointScheme', 'center') != 'center' else 1)
    if probe_slots is not None:
        max_cells = min(max_cells, 56)
    considered = slots[:min(max_cells, math.ceil(max_probes/anchors))]
    selected, reserved = [], 0
    for slot in considered:
        allowance = min(anchors, max_probes-reserved)
        reserved += allowance
        if slot['reason'] is None:
            selected.append((slot['cell'], allowance))
    return selected, {'algorithm': copy.deepcopy(_AXIAL_ALGORITHM), 'scope': 'bone_only',
        'columnCount': len(groups), 'axialCellHalfSupportMm': support,
        'activeCurrentColumnCount': sum(bool(group) for group in groups),
        'clearedCurrentColumns': [j for j, group in enumerate(groups) if not group],
        'currentColumnActivationVisits': len(groups),
        'scheduledCellSlots': slots, 'consideredCellSlots': len(considered),
        'reservedContactProbeSlots': reserved,
        'missingCellSlots': [s for s in considered if s['reason'] == 'MISSING_CURRENT_A_LEVEL'],
        'duplicateCellSlots': [s for s in considered if s['reason'] == 'DUPLICATE_CURRENT_A_CELL'],
        'uniqueScheduledCurrentACells': len(selected),
        'currentMembershipVisits': len(cells), 'currentColumnAssignmentsVisited': len(cells),
        'currentFrontierDepthVisits': sum(map(len, groups)),
        'currentDepthBandVisits': 4*sum(map(len, groups)),
        'sourceCoordinateValidationCells': len(cells),
        'cachedSourcePreparation': copy.deepcopy(cache['preparationCost'])}


def _same_pose(first, second):
    return (np.linalg.norm(np.subtract(first['position'], second['position'])) <= 1e-9 and
            (Rotation.from_quat(first['quaternion'])*Rotation.from_quat(
                second['quaternion']).inv()).magnitude() <= 1e-9)


def _staging_approach_hash(approach):
    return semantic_digest({'approach': _approach_declaration(approach),
                            'pointScheme': approach.get('pointScheme', 'center')})


def bind_milling_staging_plan(engine, *, approach, global_park, inbound, outbound):
    """Freeze bounded native staging proposals, never exterior/clearance evidence.

    Inbound vertices follow globalPark and end at the local working attitude.
    Outbound starts at that same local vertex and ends at globalPark. Posed
    callers must actually start at globalPark; no implicit alignment is added.
    """
    contract = validate_source_scope(engine)
    _approach(engine, contract, approach)
    plan = {'schema': STAGING_SCHEMA, 'sceneId': engine.scene.identity,
        'contractHash': contract.content_hash, 'sourceStateId': engine.state_id,
        'approachHash': _staging_approach_hash(approach),
        'globalPark': copy.deepcopy(global_park), 'inbound': copy.deepcopy(inbound),
        'outbound': copy.deepcopy(outbound)}
    plan['contentHash'] = semantic_digest(plan)
    return _staging_plan(engine, contract, approach, plan, require_source_state=True)


def _staging_plan(engine, contract, approach, plan, *, require_source_state=False):
    fields = {'schema', 'sceneId', 'contractHash', 'sourceStateId', 'approachHash',
              'globalPark', 'inbound', 'outbound', 'contentHash'}
    if (type(plan) is not dict or set(plan) != fields or plan['schema'] != STAGING_SCHEMA or
            plan['sceneId'] != engine.scene.identity or plan['contractHash'] != contract.content_hash or
            plan['approachHash'] != _staging_approach_hash(approach) or
            plan['contentHash'] != semantic_digest({k: v for k, v in plan.items() if k != 'contentHash'}) or
            require_source_state and plan['sourceStateId'] != engine.state_id):
        raise ValueError('MILLING_STAGING_SOURCE_BINDING_REQUIRED')
    if any(type(plan[name]) is not list or not 1 <= len(plan[name]) <= 8
           for name in ('inbound', 'outbound')):
        raise ValueError('MILLING_STAGING_EXPLICIT_PATH_BOUNDS_REQUIRED')
    poses = [plan['globalPark'], *plan['inbound'], *plan['outbound']]
    for pose in poses:
        if type(pose) is not dict or set(pose) != {'position', 'quaternion'}:
            raise ValueError('MILLING_STAGING_FINITE_NATIVE_POSES_REQUIRED')
        p, q = pose['position'], pose['quaternion']
        if (type(p) is not list or type(q) is not list or len(p) != 3 or len(q) != 4 or
                any(type(x) not in (int, float) or not math.isfinite(x) for x in [*p, *q]) or
                abs(np.linalg.norm(q)-1.) > 1e-9):
            raise ValueError('MILLING_STAGING_FINITE_NATIVE_POSES_REQUIRED')
    local = plan['inbound'][-1]
    if (not _same_pose(local, plan['outbound'][0]) or
            not _same_pose(plan['globalPark'], plan['outbound'][-1]) or
            (Rotation.from_quat(local['quaternion'])*Rotation.from_quat(
                approach['quaternion']).inv()).magnitude() > 1e-9 or
            abs(float((np.asarray(local['position'])-approach['outsidePlanePointRasMm'])@
                Rotation.from_quat(approach['quaternion']).apply([0., 0., 1.]))) > 1e-9):
        raise ValueError('MILLING_STAGING_PATH_ENDPOINTS_DIFFER')
    for path in ([plan['globalPark'], *plan['inbound']], plan['outbound']):
        if any(np.linalg.norm(np.subtract(b['position'], a['position'])) > 250
               for a, b in zip(path, path[1:])):
            raise ValueError('MILLING_STAGING_NATIVE_LEG_BOUND_EXCEEDED')
    for name in ('cranial_drill', 'frazier_suction'):
        if not _exterior(engine.scene, contract, _segments(name, plan['globalPark'])):
            raise ValueError('MILLING_STAGING_GLOBAL_PARK_NOT_EXTERIOR')
    return copy.deepcopy(plan)


def _local_clearance(engine, contract, segments, removed, protected, slot, *, require_outside_entry=True,
                     geometry_cost=None):
    """Conservative source geometry refusal only; native guards remain mandatory."""
    from .surgical_task_domain import _outside, _posed, reviewed_domain
    if geometry_cost is not None:
        geometry_cost['wholeBodyChecks'] = geometry_cost.get('wholeBodyChecks', 0)+1
    if not _within_workspace(engine.scene, contract, segments):
        return 'MILLING_STAGING_OUTSIDE_WORKSPACE'
    domain = reviewed_domain(engine.scene)
    if domain is not None and _outside(engine.scene, domain, segments):
        return 'MILLING_STAGING_OUTSIDE_REVIEWED_DOMAIN'
    try:
        if _inside_entry(contract, segments) and require_outside_entry:
            return 'MILLING_STAGING_BODY_NOT_OUTSIDE_ENTRY'
    except ValueError:
        return 'MILLING_STAGING_BODY_NOT_OUTSIDE_ENTRY'
    other = 'right' if slot == 'left' else 'left'
    if _posed(engine, other):
        current = engine.state['slots'][other]
        if pair_conflict(segments, envelopes(TOOLS[current['instrument']], engine.parameters(current),
                current['position'], current['quaternion'])):
            return 'MILLING_STAGING_OTHER_HAND_CONTACT'
    for a, b, radius, _ in segments:
        cells = set(map(int, engine.cells(a, b, radius)))
        if geometry_cost is not None:
            geometry_cost['bodyCapsuleQueries'] += 1
            geometry_cost['bodyCoveredCellVisits'] += len(cells)
            geometry_cost['protectionCapsuleQueries'] += engine.protection is not None
        if cells & protected or (engine.protection is not None and
                len(engine.protection.capsule_hits(a, b, radius))):
            return 'MILLING_STAGING_ORIGINAL_TARGET_OR_PROTECTED_CONTACT'
        if any(engine.scene.labels.ravel()[c] != 0 for c in cells-removed):
            return 'MILLING_STAGING_CURRENT_MATERIAL_CONTACT'
    return None


def _local_path_clearance(engine, contract, name, poses, slot, *, check=lambda: None,
                          allow_entry_retreat=False, allow_cleared_cavity_lead=False, geometry_cost=None,
                          native_duration=None):
    """Bounded source preflight with native swept envelopes, not native admission.

    Legacy paths retain <=.25mm/1degree subdivisions. When native_duration is
    declared, use the exact native one-leg distance/angle/duration rule instead.
    The separate 20,000-interval bound includes every proposed staging leg.
    """
    protected = set(engine._protected_ids) | _original_targets(engine.scene)
    if native_duration is not None and not _finite_positive(native_duration):
        raise ValueError('MILLING_STAGING_EXPLICIT_NATIVE_DURATION_REQUIRED')
    if allow_cleared_cavity_lead and native_duration is None:
        raise ValueError('MILLING_STAGING_EXPLICIT_NATIVE_DURATION_REQUIRED')
    if allow_cleared_cavity_lead and allow_entry_retreat:
        raise ValueError('MILLING_STAGING_DISTINCT_LEAD_AND_RETREAT_REQUIRED')
    if allow_cleared_cavity_lead and (engine.state['holds'] or engine.state.get('detachedFragments') or
            any(hand['active'] or hand['grip'] for hand in engine.state['slots'].values()) or
            any(any(offset) for offset in engine.state.get('deformation', {}).values())):
        return 'MILLING_STAGING_INACTIVE_UNMOVED_CAVITY_REQUIRED', 0
    if allow_cleared_cavity_lead:
        for row in engine.scene.source.get('protectedAnatomyV1', []):
            protected.update(row['cells'])
    removed = set(engine.state['removed'])
    intervals = 0
    reason = _local_clearance(engine, contract, _segments(name, poses[0]), removed, protected, slot,
                              require_outside_entry=not allow_entry_retreat, geometry_cost=geometry_cost)
    if reason:
        return reason, intervals
    previous = poses[0]
    for end in poses[1:]:
        rotations = Rotation.from_quat([previous['quaternion'], end['quaternion']])
        angle = float((rotations[1]*rotations[0].inv()).magnitude())
        count = max(1, math.ceil(np.linalg.norm(np.subtract(end['position'], previous['position']))/.25),
                    math.ceil(angle/np.deg2rad(1.)))
        if native_duration is not None:
            # Match the actual one-leg native program, including its duration.
            # Finer unrelated preflight subdivisions can hide native inflation.
            count = max(1, math.ceil((np.linalg.norm(np.subtract(end['position'], previous['position']))+
                40.*angle)/min(.4, float(engine.scene.spacing.min())/3.)), math.ceil(native_duration/.1))
        intervals += count
        if intervals > 20_000:
            return 'MILLING_STAGING_GEOMETRY_INTERVAL_BOUND', intervals
        start = previous
        interpolation = Slerp([0., 1.], rotations)
        for fraction in np.linspace(0., 1., count+1)[1:]:
            check()
            pose = _pose((1-fraction)*np.asarray(start['position'])+fraction*np.asarray(end['position']),
                         interpolation(fraction).as_quat())
            segments = swept_envelopes(TOOLS[name], {'radiusMm': TOOLS[name]['radiusMm'], 'depthMm': 2.}, previous, pose)
            reason = _local_clearance(engine, contract, segments, removed, protected, slot,
                                      require_outside_entry=not (allow_entry_retreat or allow_cleared_cavity_lead),
                                      geometry_cost=geometry_cost)
            if reason:
                return reason, intervals
            previous = pose
    return _local_clearance(engine, contract, _segments(name, poses[-1]), removed, protected, slot,
        require_outside_entry=not allow_cleared_cavity_lead, geometry_cost=geometry_cost), intervals


def bind_milling_approach(engine, *, quaternion, outside_plane_point_ras_mm,
                          maximum_approach_mm, point_scheme='center',
                          cell_schedule='surface_first', check=None, contact_pose_family='leading_touch'):
    """Bind explicitly supplied fixed attitude and exterior plane to the Scene.

    This records a proposal route, not a reviewed exterior/clearance proof.
    Per-contact exterior/body/workspace tests run during generation; native
    verification must still check staging repositioning and every motion.
    The opt-in axial schedule binds <=56 source columns and four actual-current
    A depth levels; all source-wide preparation remains separately reported.
    Both point schemes are supported (<=224 center or2016 quarter probes).
    The separate opt-in cavity-carried family requires quarter points and the
    axial schedule; generation also requires an explicit staging plan/duration.
    """
    binding_started = time.monotonic()
    if type(cell_schedule) is not str or cell_schedule not in _CELL_SCHEDULES or check is not None and not callable(check):
        raise ValueError('MILLING_CONTACT_DECLARED_CELL_SCHEDULE_REQUIRED')
    if (type(contact_pose_family) is not str or contact_pose_family not in _CONTACT_POSE_FAMILIES or
            contact_pose_family != 'leading_touch' and (point_scheme != _POINT_SCHEMES[1] or
                                                       cell_schedule != _CELL_SCHEDULES[1])):
        raise ValueError('MILLING_CONTACT_DECLARED_POSE_FAMILY_REQUIRED')
    _geometry_check(check)
    contract = validate_source_scope(engine)
    if not is_milling(contract):
        raise ValueError('MILLING_CONTACT_MILLING_CONTRACT_REQUIRED')
    raw = {'schema': APPROACH_SCHEMA, 'sceneId': engine.scene.identity,
           'contractHash': contract.content_hash, 'quaternion': list(quaternion),
           'outsidePlanePointRasMm': list(outside_plane_point_ras_mm),
           'maximumApproachMm': maximum_approach_mm}
    if point_scheme not in _POINT_SCHEMES:
        raise ValueError('MILLING_CONTACT_DECLARED_POINT_SCHEME_REQUIRED')
    if point_scheme != 'center':
        raw['pointScheme'] = point_scheme
    if cell_schedule != 'surface_first':
        raw['cellSchedule'] = cell_schedule
    if contact_pose_family != 'leading_touch':
        raw['contactPoseFamily'] = contact_pose_family
    _, _, axis = _approach(engine, contract, raw, require_order=False)
    started = time.monotonic()
    cells = list(contract.aperture_cells)
    if cell_schedule == 'surface_first':
        points = engine.scene.points(cells)
    else:
        chunks = []
        for start in range(0, len(cells), 1024):
            _geometry_check(check)
            chunks.append(engine.scene.points(cells[start:start+1024]))
        points = np.concatenate(chunks)
    depths = dict(zip(cells, points@axis))
    ordering = {'sceneId': engine.scene.identity, 'contractHash': contract.content_hash,
        'quaternion': list(quaternion), 'orderedCells': sorted(cells,
            key=lambda c: (round(float(depths[c]), 9), c)),
        'orderingCells': len(cells), 'orderingWallSeconds': time.monotonic()-started,
        'bindingWallSeconds': time.monotonic()-binding_started}
    if cell_schedule != 'surface_first':
        lookup = {cell: j for j, cell in enumerate(cells)}
        ordered_points = points[[lookup[cell] for cell in ordering['orderedCells']]]
        ordering['axialCache'] = _build_axial_cache(engine, contract, raw,
            ordering['orderedCells'], ordered_points, check)
        ordering['orderingWallSeconds'] = time.monotonic()-started
        ordering['bindingWallSeconds'] = time.monotonic()-binding_started
    raw['sourceOrdering'] = ordering
    raw['sourceOrderingHash'] = _ordering_hash(ordering)
    _geometry_check(check)
    return raw


def _approach(engine, contract, raw, *, require_order=True):
    if engine._cranial_access != contract:
        raise ValueError('MILLING_CONTACT_NATIVE_CONTRACT_BINDING_DIFFERS')
    fields = {'schema', 'sceneId', 'contractHash', 'quaternion',
              'outsidePlanePointRasMm', 'maximumApproachMm'}
    if isinstance(raw, dict) and 'pointScheme' in raw:
        fields.add('pointScheme')
    if isinstance(raw, dict) and 'cellSchedule' in raw:
        fields.add('cellSchedule')
    if isinstance(raw, dict) and 'contactPoseFamily' in raw:
        fields.add('contactPoseFamily')
    if require_order:
        fields.update(('sourceOrdering', 'sourceOrderingHash'))
    if (not isinstance(raw, dict) or set(raw) != fields or
            raw['schema'] != APPROACH_SCHEMA or raw['sceneId'] != engine.scene.identity or
            raw['contractHash'] != contract.content_hash):
        raise ValueError('MILLING_CONTACT_APPROACH_SOURCE_BINDING_REQUIRED')
    if raw.get('pointScheme', 'center') not in _POINT_SCHEMES:
        raise ValueError('MILLING_CONTACT_DECLARED_POINT_SCHEME_REQUIRED')
    schedule = raw.get('cellSchedule', 'surface_first')
    if type(schedule) is not str or schedule not in _CELL_SCHEDULES or 'cellSchedule' in raw and schedule == 'surface_first':
        raise ValueError('MILLING_CONTACT_DECLARED_CELL_SCHEDULE_REQUIRED')
    if ('contactPoseFamily' in raw and (raw['contactPoseFamily'] != 'cavity_carried_rays' or
            schedule != _CELL_SCHEDULES[1] or raw.get('pointScheme') != _POINT_SCHEMES[1])):
        raise ValueError('MILLING_CONTACT_DECLARED_POSE_FAMILY_REQUIRED')
    q, point = np.asarray(raw['quaternion'], float), np.asarray(raw['outsidePlanePointRasMm'], float)
    if (q.shape != (4,) or point.shape != (3,) or not np.isfinite(q).all() or
            not np.isfinite(point).all() or abs(np.linalg.norm(q)-1.) > 1e-9 or
            not _finite_positive(raw['maximumApproachMm'])):
        raise ValueError('MILLING_CONTACT_EXPLICIT_FINITE_APPROACH_REQUIRED')
    axis = Rotation.from_quat(q).apply([0., 0., 1.])
    if axis@np.asarray(contract.entry.normal_ras_unit) < math.cos(math.radians(
            contract.entry.maximum_angle_degrees))-1e-9:
        raise ValueError('MILLING_CONTACT_ATTITUDE_OUTSIDE_ENTRY_CONE')
    if require_order:
        ordering = raw['sourceOrdering']
        ordering_fields = {'sceneId', 'contractHash', 'quaternion',
                'orderedCells', 'orderingCells', 'orderingWallSeconds', 'bindingWallSeconds'}
        if schedule != 'surface_first':
            ordering_fields.add('axialCache')
        if (not isinstance(ordering, dict) or set(ordering) != ordering_fields or
                raw['sourceOrderingHash'] != _ordering_hash(ordering) or
                ordering['sceneId'] != engine.scene.identity or ordering['contractHash'] != contract.content_hash or
                ordering['quaternion'] != raw['quaternion'] or
                ordering['orderingCells'] != len(contract.aperture_cells) or
                sorted(ordering['orderedCells']) != list(contract.aperture_cells) or
                not _finite_positive(ordering['orderingWallSeconds']) or
                not _finite_positive(ordering['bindingWallSeconds'])):
            raise ValueError('MILLING_CONTACT_SOURCE_ORDERING_BINDING_REQUIRED')
        if schedule != 'surface_first' and (type(ordering['axialCache']) is not dict or
                ordering['axialCache'].get('binding') != _axial_source_binding(engine, contract, raw)):
            raise ValueError('MILLING_CONTACT_AXIAL_SOURCE_CACHE_BINDING_REQUIRED')
    return q, point, axis


def _pose(position, quaternion):
    return {'position': np.asarray(position).tolist(), 'quaternion': np.asarray(quaternion).tolist()}


def _contact(engine, cell, point, axis, radius, rho, q, *, check=None):
    """Invert the existing exact orthogonal-cell distance on an outward ray."""
    index = np.asarray([np.unravel_index(cell, engine.scene.shape)], dtype=np.int64)
    center = engine.scene.affine[:3, :3]@index[0]+engine.scene.affine[:3, 3]
    lo, hi = 0., radius+rho+float(np.linalg.norm(point-center))+1e-6
    for _ in range(_ALGORITHM['iterations']):
        _geometry_check(check)
        middle = (lo+hi)/2
        tip = point-axis*middle
        if float(_segment_cell_distances(engine.grid, index, tip, tip)[0]) <= radius:
            lo = middle
        else:
            hi = middle
    threshold = (lo+hi)/2
    pose = _pose(point-axis*(threshold-_OVERLAP_MM), q)
    return pose, threshold


def _contact_points(engine, cell, approach):
    """Fixed interior source points; every inversion counts against the probe cap."""
    center = engine.scene.points([cell])[0]
    yield 0, center
    if approach.get('pointScheme', 'center') == 'center_and_quarter_cell_corners':
        for index, offset in enumerate(product((-.25, .25), repeat=3), 1):
            yield index, center+engine.scene.affine[:3, :3]@np.asarray(offset)


def _segments(name, pose):
    return envelopes(TOOLS[name], {'radiusMm': TOOLS[name]['radiusMm'], 'depthMm': 2.},
                     pose['position'], pose['quaternion'])


def _route(name, outside, contact):
    # Constant attitude, collinear forward translation: this is the exact
    # union of each stock capsule, not an endpoint-only shaft test.
    shift = np.subtract(outside['position'], contact['position'])
    return [(a+shift, b, radius, part) for a, b, radius, part in _segments(name, contact)]


def _within_workspace(scene, contract, segments):
    inverse = scene.inverse
    lo, hi = np.asarray(contract.exit.minimum_voxel_boundary), np.asarray(contract.exit.maximum_voxel_boundary)
    scale = np.linalg.norm(inverse[:3, :3], axis=1)
    for a, b, radius, _ in segments:
        points = np.asarray([a, b])@inverse[:3, :3].T+inverse[:3, 3]
        if np.any(points.min(axis=0)-radius*scale < lo-1e-9) or np.any(points.max(axis=0)+radius*scale > hi+1e-9):
            return False
    return True


def _exterior(scene, contract, segments):
    axis, side, plane = contract.exit.axis, contract.exit.side, contract.exit.plane_voxel_boundary
    row = scene.inverse[axis]
    for a, b, radius, _ in segments:
        coordinates = np.asarray([a, b])@row[:3]+row[3]
        support = radius*np.linalg.norm(row[:3])
        if side == -1 and coordinates.max()+support >= plane-1e-9 or side == 1 and coordinates.min()-support <= plane+1e-9:
            return False
    return True


def _program(drill, suction, outside, slot, drill_seconds, suction_seconds):
    def action(operation, **kwargs):
        return {'slot': slot, 'operation': operation, 'duration': .1, **kwargs}
    result = [action('configure', mechanicsRevision=2, clockRevision=1)]
    contacts = [('frazier_suction', suction, suction_seconds)]
    if drill is not None:
        contacts.insert(0, ('cranial_drill', drill, drill_seconds))
    for name, contact, duration in contacts:
        result.extend((action('exchange', instrument=name), action('place', pose=copy.deepcopy(outside)),
            action('apply', pose=copy.deepcopy(contact), duration=duration)))
        if drill is None:
            # A moving outlet may contact debris only near the approach end.
            # Charge a separate stationary native dwell; never infer uptake
            # from the nominal endpoint footprint or refund approach time.
            result.append(action('apply', pose=copy.deepcopy(contact), duration=duration))
        result.append(action('withdraw', pose=copy.deepcopy(outside)))
    return result


def _context(engine, approach, slot, max_examined_cells, max_contacts, max_contact_probes,
             *durations, staging_plan=None):
    if not isinstance(engine, (SurgicalEngine, MeteredEngine)):
        raise TypeError('MILLING_CONTACT_NATIVE_ENGINE_REQUIRED')
    if (type(max_examined_cells) is not int or not 1 <= max_examined_cells <= 2048 or
            type(max_contacts) is not int or not 1 <= max_contacts <= 32 or
            type(max_contact_probes) is not int or not 1 <= max_contact_probes <= 2048):
        raise ValueError('MILLING_CONTACT_EXPLICIT_BOUNDS_REQUIRED')
    if any(not _finite_positive(duration) for duration in durations):
        raise ValueError('MILLING_CONTACT_EXPLICIT_DURATIONS_REQUIRED')
    if any(TOOLS[name]['radiusMm'] != radius or TOOLS[name]['shaftRadiusMm'] != .5
            for name, radius in (('cranial_drill', 2.), ('frazier_suction', 1.5))):
        raise ValueError('MILLING_CONTACT_STOCK_CATALOG_REQUIRED')
    contract = validate_source_scope(engine)
    if not is_milling(contract):
        raise ValueError('MILLING_CONTACT_MILLING_CONTRACT_REQUIRED')
    q, plane, axis = _approach(engine, contract, approach)
    if approach.get('contactPoseFamily') == 'cavity_carried_rays' and staging_plan is None:
        raise ValueError('MILLING_CONTACT_CAVITY_RAYS_REQUIRE_STAGING_PLAN')
    if staging_plan is not None:
        _staging_plan(engine, contract, approach, staging_plan)
    if engine.grid._orthogonal_spacing is None:
        raise ValueError('MILLING_CONTACT_SHEARED_AFFINE_CONTACT_UNSUPPORTED')
    if slot not in engine.state['slots'] or any(s['active'] or s['grip'] or
            s['inserted'] and (staging_plan is None or hand != slot)
            for hand, s in engine.state['slots'].items()):
        raise ValueError('MILLING_CONTACT_PARKED_INACTIVE_HANDS_REQUIRED')
    if (engine.state['holds'] or engine.state.get('detachedFragments') or
            any(any(offset) for offset in engine.state.get('deformation', {}).values())):
        raise ValueError('MILLING_CONTACT_UNMOVED_SOURCE_CAVITY_REQUIRED')
    before = engine.state_id
    readiness = milling_readiness(engine)
    if not readiness['historyVerified'] or not readiness['debrisConserved']:
        raise ValueError('MILLING_CONTACT_NATIVE_CAVITY_HISTORY_REQUIRED')
    removed = set(engine.state['removed'])
    remaining = set(contract.aperture_cells)-removed
    protected = set(engine._protected_ids) | _original_targets(engine.scene)
    for row in engine.scene.source.get('protectedAnatomyV1', []):
        protected.update(row['cells'])
    protected.update(map(int, np.flatnonzero(np.isin(engine.scene.labels, [r['id'] for r in
        engine.scene.regions if r.get('protected') is True or r.get('critical') is True]))))
    if staging_plan is not None:
        from .surgical_task_domain import _posed
        current = engine.state['slots'][slot]
        if _posed(engine, slot):
            if (current['instrument'] not in ('cranial_drill', 'frazier_suction') or
                    engine.parameters(current)['radiusMm'] != TOOLS[current['instrument']]['radiusMm'] or
                    engine.parameters(current)['depthMm'] != 2.):
                raise ValueError('MILLING_STAGING_STOCK_TOOL_REQUIRED')
            reason = _local_clearance(engine, contract, envelopes(TOOLS[current['instrument']],
                engine.parameters(current), current['position'], current['quaternion']), removed, protected, slot)
            if reason:
                raise ValueError(reason)
    # Maximum corner norm is valid for oblique AND sheared affine cells.
    corners = np.asarray(list(product((-.5, .5), repeat=3)))@engine.scene.affine[:3, :3].T
    rho = float(np.linalg.norm(corners, axis=1).max())
    return contract, q, plane, axis, before, readiness, removed, remaining, protected, rho


def _geometry_checker(engine, contract, slot, removed, protected, *, local=False):
    def check_geometry(name, contact, outside, *, allowed_work):
        route = _route(name, outside, contact)
        if local:
            reason = _local_clearance(engine, contract, _segments(name, outside), removed, protected, slot)
            if reason:
                return reason, None
        elif not _exterior(engine.scene, contract, _segments(name, outside)):
            return 'OUTSIDE_POSE_NOT_AT_ORIGINAL_EXIT', None
        if not _within_workspace(engine.scene, contract, route):
            return 'OUTSIDE_REVIEWED_WORKSPACE', None
        if local:
            from .surgical_task_domain import _outside, reviewed_domain
            domain = reviewed_domain(engine.scene)
            if domain is not None and _outside(engine.scene, domain, route):
                return 'MILLING_STAGING_OUTSIDE_REVIEWED_DOMAIN', None
        try:
            _inside_entry(contract, route)
        except ValueError as error:
            return str(error), None
        other = 'right' if slot == 'left' else 'left'
        from .surgical_task_domain import _posed
        if _posed(engine, other):
            other_slot = engine.state['slots'][other]
            if pair_conflict(route, envelopes(TOOLS[other_slot['instrument']],
                    engine.parameters(other_slot), other_slot['position'], other_slot['quaternion'])):
                return 'OTHER_HAND_CONTACT', None
        hits = set()
        for a, b, radius, part in route:
            touched = set(map(int, engine.cells(a, b, radius)))
            if touched & protected:
                return 'ORIGINAL_TARGET_OR_PROTECTED_CONTACT', min(touched & protected)
            if local and engine.protection is not None:
                guarded = engine.protection.capsule_hits(a, b, radius)
                if len(guarded):
                    return 'ORIGINAL_TARGET_OR_PROTECTED_CONTACT', int(guarded[0])
            occupied = {c for c in touched-removed if engine.scene.labels.ravel()[c] != 0}
            if part == 'shaft' and occupied:
                return 'CURRENT_CAVITY_SHAFT_OCCUPIED', min(occupied)
            if part != 'shaft':
                if occupied-allowed_work:
                    return 'WORKING_CONTACT_OUTSIDE_ALLOWED_CURRENT_A', min(occupied-allowed_work)
                hits.update(occupied)
        return None, hits

    return check_geometry


def _local_contact_program(contact, outside, slot, duration, *, suction=False):
    """Same-tool local retreat is move, never a false completed withdrawal."""
    actions = [{'slot': slot, 'operation': 'move', 'pose': copy.deepcopy(outside), 'duration': .1},
               {'slot': slot, 'operation': 'apply', 'pose': copy.deepcopy(contact), 'duration': duration}]
    if suction:
        actions.append({'slot': slot, 'operation': 'apply', 'pose': copy.deepcopy(contact), 'duration': duration})
    actions.extend(({'slot': slot, 'operation': 'deactivate', 'duration': .1},
        {'slot': slot, 'operation': 'move', 'pose': copy.deepcopy(outside), 'duration': .1}))
    return actions


def _cavity_ray_anchors(engine, cell, cavity, cavity_points, axis, rho, cost, check):
    """36 RESERVED slots; absent/duplicate carriers and projected rays stay slots."""
    center = engine.scene.points([cell])[0]
    for j, point in _contact_points(engine, cell, {'pointScheme': _POINT_SCHEMES[1]}):
        yield j, point, None
    cost['cavityCentroidVisits'] += len(cavity)
    direction = cavity_points.mean(0)-center if len(cavity) else np.zeros(3)
    direction -= axis*float(direction@axis)
    length = float(np.linalg.norm(direction))
    used = set()
    for k, distance in enumerate((0., 2., 2.+rho)):
        _geometry_check(check)
        carrier = None
        if length > 1e-12:
            query = center+distance*direction/length
            best = None
            for start in range(0, len(cavity), 1024):
                _geometry_check(check)
                points = cavity_points[start:start+1024]
                delta = points-query
                delta -= np.outer(delta@axis, axis)
                squared = np.round(np.sum(delta**2, axis=1), 10)
                axial = np.round(np.abs((points-center)@axis), 10)
                cost['carrierDistancePairs'] += len(points)
                for j in range(len(points)):
                    candidate = (float(squared[j]), float(axial[j]), cavity[start+j], start+j)
                    if best is None or candidate < best:
                        best = candidate
            carrier = best[3]
            if carrier in used:
                carrier = None
            else:
                used.add(carrier)
        offsets = [np.zeros(3), *(np.asarray(o) for o in product((-.375, .375), repeat=3))]
        for j, offset in enumerate(offsets):
            point = None if carrier is None else cavity_points[carrier]+engine.scene.affine[:3, :3]@offset
            anchor = None if point is None else point+axis*float((center-point)@axis)
            yield 9+k*9+j, anchor, None if carrier is None else cavity[carrier]


def _cavity_body_check(engine, contract, segments, removed, remaining, protected, slot, cost, check):
    """Only working material in CURRENT A may contact the full active enclosure."""
    from .surgical_task_domain import _outside, _posed, reviewed_domain
    cost['wholeBodyChecks'] += 1
    if not _within_workspace(engine.scene, contract, segments):
        return 'OUTSIDE_REVIEWED_WORKSPACE', set()
    domain = reviewed_domain(engine.scene)
    if domain is not None and _outside(engine.scene, domain, segments):
        return 'MILLING_STAGING_OUTSIDE_REVIEWED_DOMAIN', set()
    try:
        _inside_entry(contract, segments)
    except ValueError as error:
        return str(error), set()
    other = 'right' if slot == 'left' else 'left'
    if _posed(engine, other):
        hand = engine.state['slots'][other]
        if pair_conflict(segments, envelopes(TOOLS[hand['instrument']], engine.parameters(hand),
                hand['position'], hand['quaternion'])):
            return 'OTHER_HAND_CONTACT', set()
    hits = set()
    for a, b, radius, part in segments:
        _geometry_check(check)
        touched = set(map(int, engine.cells(a, b, radius)))
        cost['bodyCapsuleQueries'] += 1
        cost['bodyCoveredCellVisits'] += len(touched)
        if touched & protected:
            return 'ORIGINAL_TARGET_OR_PROTECTED_CONTACT', set()
        if engine.protection is not None:
            cost['protectionCapsuleQueries'] += 1
            if len(engine.protection.capsule_hits(a, b, radius)):
                return 'ORIGINAL_TARGET_OR_PROTECTED_CONTACT', set()
        occupied = {c for c in touched-removed if engine.scene.labels.ravel()[c] != 0}
        if part == 'shaft' and occupied:
            return 'CURRENT_CAVITY_SHAFT_OCCUPIED', set()
        if occupied-remaining:
            return 'WORKING_CONTACT_OUTSIDE_ALLOWED_CURRENT_A', set()
        hits.update(occupied)
    return None, hits


def _generate_cavity_ray_contacts(engine, approach, context, max_cells, max_contacts, max_probes,
                                   drill_seconds, staging_seconds, slot, staging_plan, check, started):
    contract, q, plane, axis, before, readiness, removed, remaining, protected, rho = context
    cost = dict(originalACurrentMembershipVisits=0, cavityCoordinates=0, cavityCentroidVisits=0,
        carrierDistancePairs=0, inverseEligibilityQueries=0, inverseKernelQueries=0,
        bodyCapsuleQueries=0, bodyCoveredCellVisits=0, protectionCapsuleQueries=0,
        wholeBodyChecks=0, passiveLeadIntervals=0, tipEffectSupportQueries=0)
    source = list(contract.aperture_cells)
    cavity = []
    for start in range(0, len(source), 1024):
        _geometry_check(check)
        cavity.extend(c for c in source[start:start+1024] if c in removed)
        cost['originalACurrentMembershipVisits'] += len(source[start:start+1024])
    chunks = []
    for start in range(0, len(cavity), 1024):
        _geometry_check(check)
        chunks.append(engine.scene.points(cavity[start:start+1024]))
    cavity_points = np.concatenate(chunks) if chunks else np.empty((0, 3))
    cost['cavityCoordinates'] = len(cavity)
    scheduled, schedule = _axial_current_cells(engine, contract, approach, remaining,
        max_cells, max_probes, check, probe_slots=36)
    candidates, rejections, examined, incomplete = [], [], [], []
    probes = 0
    for cell, allowance in scheduled:
        _geometry_check(check)
        examined.append(cell)
        seen = set()
        if allowance < 36:
            incomplete.append(cell)
        for point_index, point, carrier in islice(_cavity_ray_anchors(engine, cell, cavity, cavity_points,
                                                                    axis, rho, cost, check), allowance):
            _geometry_check(check)
            probes += 1
            witness = {'cell': cell, 'pointIndex': point_index, 'carrierSourceCell': carrier}
            if point is None:
                rejections.append({**witness, 'reason': 'MISSING_CURRENT_CAVITY_CARRIER'})
                continue
            key = tuple(np.round(engine.scene.inverse[:3, :3]@point+engine.scene.inverse[:3, 3], 10))
            if key in seen:
                rejections.append({**witness, 'reason': 'DUPLICATE_RESERVED_CONTACT_RAY'})
                continue
            seen.add(key)
            index = np.asarray([np.unravel_index(cell, engine.scene.shape)], dtype=np.int64)
            cost['inverseEligibilityQueries'] += 1
            if float(_segment_cell_distances(engine.grid, index, point, point)[0]) > 2.:
                rejections.append({**witness, 'reason': 'RAY_ANCHOR_OUTSIDE_TARGET_CONTACT_INTERVAL'})
                continue
            drill, threshold = _contact(engine, cell, point, axis, 2., rho, q, check=check)
            cost['inverseKernelQueries'] += _ALGORITHM['iterations']
            tip = np.asarray(drill['position'])
            cost['inverseEligibilityQueries'] += 1
            if float(_segment_cell_distances(engine.grid, index, tip, tip)[0]) > 2.+1e-9:
                rejections.append({**witness, 'reason': 'FIXED_OVERLAP_EXITS_TARGET_CONTACT_INTERVAL'})
                continue
            distance = float((tip-plane)@axis)
            lead = _pose(tip-.20*axis, q)
            outside = _pose(tip-distance*axis, q)
            if not .20 < distance <= approach['maximumApproachMm']:
                rejections.append({**witness, 'reason': 'APPROACH_LENGTH_OUTSIDE_BOUND'})
                continue
            # Check the whole current-body passive route, including an entry endpoint.
            refusal, intervals = _local_path_clearance(engine, contract, 'cranial_drill',
                [outside, lead], slot, check=check or (lambda: None),
                allow_cleared_cavity_lead=True, geometry_cost=cost, native_duration=staging_seconds)
            cost['passiveLeadIntervals'] += intervals
            if refusal is None:
                enclosure = swept_envelopes(TOOLS['cranial_drill'], {'radiusMm': 2., 'depthMm': 2.},
                                            lead, drill)
                refusal, _ = _cavity_body_check(engine, contract, enclosure, removed, remaining,
                                                protected, slot, cost, check)
            if refusal:
                rejections.append({**witness, 'reason': refusal})
                continue
            hits = set(map(int, engine.cells(lead['position'], drill['position'], 2.))) & remaining
            cost['tipEffectSupportQueries'] += 1
            if cell not in hits:
                rejections.append({**witness, 'reason': 'NO_CURRENT_A_CONTACT'})
                continue
            candidates.append({**witness, 'contactAnchorRasMm': point.tolist(),
                'drillPose': drill, 'leadPose': lead, 'outsidePose': outside,
                'nativeLeadingDrillContactDistanceMm': threshold, 'hits': hits})
    uncovered, selected, poses = set(remaining), [], set()
    while candidates and len(selected) < max_contacts:
        _geometry_check(check)
        best = min(candidates, key=lambda row: (-len(row['hits'] & uncovered), row['cell'], row['pointIndex']))
        candidates.remove(best)
        key = tuple(np.round(best['drillPose']['position'], 10))
        if not best['hits'] & uncovered or key in poses:
            continue
        poses.add(key)
        actions = [
            {'slot': slot, 'operation': 'move', 'pose': copy.deepcopy(best['outsidePose']), 'duration': staging_seconds},
            {'slot': slot, 'operation': 'move', 'pose': copy.deepcopy(best['leadPose']), 'duration': staging_seconds},
            {'slot': slot, 'operation': 'apply', 'pose': copy.deepcopy(best['drillPose']), 'duration': drill_seconds},
            {'slot': slot, 'operation': 'apply', 'pose': copy.deepcopy(best['drillPose']), 'duration': drill_seconds},
            {'slot': slot, 'operation': 'deactivate', 'duration': .1},
            {'slot': slot, 'operation': 'move', 'pose': copy.deepcopy(best['outsidePose']), 'duration': staging_seconds}]
        selected.append({**{k: v for k, v in best.items() if k != 'hits'},
            'nominalCurrentAContactCells': sorted(best['hits']), 'actions': actions})
        uncovered.difference_update(best['hits'])
    _geometry_check(check)
    result = {'schema': SCHEMA, 'sceneId': engine.scene.identity, 'sourceStateId': before,
        'sourceActionsSeal': semantic_digest(engine.actions), 'contractHash': contract.content_hash,
        'sourceBinding': contract.source_binding, 'guardSeal': semantic_digest(engine.guard_report()),
        'protectedSourceCellsSeal': semantic_digest(sorted(protected)), 'approach': copy.deepcopy(approach),
        'algorithm': copy.deepcopy(_CAVITY_RAY_ALGORITHM), 'algorithmHash': semantic_digest(_CAVITY_RAY_ALGORITHM),
        'inverseAlgorithmHash': semantic_digest(_ALGORITHM), 'contactPoseFamily': 'cavity_carried_rays',
        'commandDurations': {'drillSeconds': drill_seconds, 'stationaryDwellSeconds': drill_seconds,
                             'passiveMoveSeconds': staging_seconds},
        'proposalKind': 'staged_current_cavity_carried_ray_drill', 'requiredInstrument': 'cranial_drill',
        'stagingPlanHash': staging_plan['contentHash'], 'localRetreatIsCompletedWithdrawal': False,
        'cellSchedule': 'axial_diverse_current_a', 'cellScheduleScope': 'bone_only', 'axialSchedule': schedule,
        'bounds': {'maxExaminedCells': max_cells, 'maxContacts': max_contacts, 'maxContactProbes': max_probes},
        'examinedCells': examined, 'examinedContactCells': len(examined),
        'matchingSuctionChecks': 0,
        'testedDrillContactProbes': probes, 'reservedContactProbeSlots': schedule['reservedContactProbeSlots'],
        'incompletelyExaminedCurrentACells': incomplete, 'unexaminedCurrentACells': sorted(remaining-set(examined)),
        'currentVerifiedRemovedACarriers': cavity, 'remainingOriginalACells': sorted(remaining),
        'contacts': selected, 'actions': [a for row in selected for a in row['actions']], 'rejections': rejections,
        'nominalUncoveredCurrentACells': sorted(uncovered), 'nominalWholeAContactCoverage': not uncovered,
        'controllerDeficit': bool(uncovered) or not readiness['openingComplete'],
        'nativeOpening': readiness, 'openingComplete': readiness['openingComplete'],
        'currentDebrisMm3': readiness['debrisPresentMm3'],
        'currentDebrisReplanRequired': readiness['debrisPresentMm3'] > 0,
        'actualDebrisReplanRequiredAfterWork': True,
        'stockGeometry': {'drillRadiusMm': 2., 'suctionRadiusMm': 1.5, 'shaftRadiusMm': .5,
            'shaftLengthMm': 40., 'workingDepthMm': 2., 'cellCircumsphereMm': rho, 'nominalOverlapMm': .10},
        'admittedOpening': False, 'nativeVerificationRequired': True, 'nativeCalls': 0,
        'scope': 'current verified cavity geometry; no nominal removal, debris uptake or exit credit',
        'nativeObligations': ['actual endpoint and fresh full-body retreat after work',
            'full original A, actual debris conservation and full global exit under original checked meter']}
    result['contentHash'] = proposal_hash(result)
    result['sourceGeometryCost'] = {**cost, 'generationWallSeconds': time.monotonic()-started,
        'contextInputCardinalities': {'sourceLabelCells': engine.scene.labels.size,
            'originalACells': len(source), 'removedSourceIDs': len(removed), 'historyActions': len(engine.actions)},
        'axialSourcePreparation': copy.deepcopy(schedule['cachedSourcePreparation']),
        'axialCurrentScheduling': {k: v for k, v in schedule.items() if k.endswith('Visits') or k.endswith('Cells')},
        'bindingWallSeconds': approach['sourceOrdering']['bindingWallSeconds'],
        'orderingWallSeconds': approach['sourceOrdering']['orderingWallSeconds']}
    _geometry_check(check)
    if engine.state_id != before:
        raise ValueError('MILLING_CONTACT_SOURCE_STATE_CHANGED')
    return result


def generate_milling_contact_proposals(engine, *, approach, max_examined_cells=2048,
        max_contacts=32, max_contact_probes=2048, drill_seconds, suction_seconds, slot='left',
        staging_plan=None, check=None, staging_seconds=None):
    """Return JSON paired contacts and one native program from the current state.

    Bounds count source cells, individual drill inversions and selected matching
    suction checks (at most 2048 + 32 inversions); binding/readiness scans are not native
    calls. No evaluator trajectories or nominal/deepest-column stops enter.
    Full-A residual IDs are retained even when the candidate bound truncates.
    Use ``actions`` with existing checked policies, whose completion gates and
    shared prospective budgets are unchanged. Any refused suction pair is
    retained; there is no extra reselection loop beyond the selected bound.
    With an explicit staging_plan, propose drill-only same-tool contacts from
    a geometrically checked local outside-entry plane; the staged controller
    must supply checked global entry/exit and later actual-debris collection.
    cavity_carried_rays additionally requires staging_seconds: every passive
    move uses it, and native interval inflation is checked at that exact duration.
    Its separate active command and stationary dwell each charge drill_seconds.
    """
    started = time.monotonic()
    if check is not None and not callable(check):
        raise ValueError('MILLING_CONTACT_CHECK_CALLBACK_REQUIRED')
    _geometry_check(check)
    if isinstance(approach, dict) and approach.get('contactPoseFamily') == 'cavity_carried_rays' and not _finite_positive(staging_seconds):
        raise ValueError('MILLING_CONTACT_CAVITY_RAYS_EXPLICIT_STAGING_DURATION_REQUIRED')
    context = _context(
        engine, approach, slot, max_examined_cells, max_contacts, max_contact_probes,
        drill_seconds, suction_seconds, staging_plan=staging_plan)
    if approach.get('contactPoseFamily') == 'cavity_carried_rays':
        return _generate_cavity_ray_contacts(engine, approach, context, max_examined_cells, max_contacts,
            max_contact_probes, drill_seconds, staging_seconds, slot, staging_plan, check, started)
    contract, q, plane, axis, before, readiness, removed, remaining, protected, rho = context
    axial = approach.get('cellSchedule', 'surface_first') != 'surface_first'
    schedule, point_allowance = None, {}
    if axial:
        scheduled, schedule = _axial_current_cells(engine, contract, approach, remaining,
            max_examined_cells, max_contact_probes, check)
        cells = [cell for cell, _ in scheduled]
        point_allowance = dict(scheduled)
    else:
        cells = [cell for cell in approach['sourceOrdering']['orderedCells'] if cell in remaining]
    examined, incomplete_cells, probes = [], [], 0
    rejections, candidates, diagnostics = [], [], []
    effect_support_queries = 0

    check_geometry = _geometry_checker(engine, contract, slot, removed, protected, local=staging_plan is not None)

    for cell in cells[:max_examined_cells]:
        _geometry_check(check)
        if probes >= max_contact_probes:
            break
        examined.append(cell)
        for point_index, point in _contact_points(engine, cell, approach):
            _geometry_check(check)
            if probes >= max_contact_probes or axial and point_index >= point_allowance[cell]:
                incomplete_cells.append(cell)
                break
            probes += 1
            witness = {'cell': cell, 'pointIndex': point_index}
            drill, threshold = _contact(engine, cell, point, axis, TOOLS['cranial_drill']['radiusMm'], rho, q)
            prior = _pose(point-axis*(TOOLS['cranial_drill']['radiusMm']+rho-_OVERLAP_MM), q)
            index = np.asarray([np.unravel_index(cell, engine.scene.shape)], dtype=np.int64)
            prior_distance = float(_segment_cell_distances(engine.grid, index,
                np.asarray(prior['position']), np.asarray(prior['position']))[0])
            diagnostics.append({**witness, 'circumsphereOnlyDrillPose': prior,
                'circumsphereOnlyNativeCellHit': prior_distance <= TOOLS['cranial_drill']['radiusMm']+1e-9,
                'nativeLeadingDrillContactDistanceMm': threshold})
            distance = float((np.asarray(drill['position'])-plane)@axis)
            outside = _pose(np.asarray(drill['position'])-distance*axis, q)
            if not 0 < distance <= approach['maximumApproachMm']:
                rejections.append({**witness, 'reason': 'APPROACH_LENGTH_OUTSIDE_BOUND'})
                continue
            reason, hits = check_geometry('cranial_drill', drill, outside, allowed_work=remaining)
            if reason:
                rejections.append({**witness, 'reason': reason, 'witnessCell': hits})
                continue
            if axial:
                hits = set(map(int, engine.cells(outside['position'], drill['position'],
                                                TOOLS['cranial_drill']['radiusMm']))) & remaining
                effect_support_queries += 1
            if cell not in hits:
                rejections.append({**witness, 'reason': 'NO_CURRENT_A_CONTACT'})
                continue
            candidates.append({**witness, 'contactAnchorRasMm': point.tolist(),
                'drillPose': drill, 'outsidePose': outside, 'hits': hits,
                'nativeLeadingDrillContactDistanceMm': threshold})
    uncovered, selected, attempted_pairs = set(remaining), [], 0
    selected_attempts = 0
    while candidates and selected_attempts < max_contacts:
        best = min(candidates, key=lambda row: (-len(row['hits'] & uncovered), row['cell'], row['pointIndex']))
        candidates.remove(best)
        if not best['hits'] & uncovered:
            break
        selected_attempts += 1
        if staging_plan is not None:
            record = {key: value for key, value in best.items() if key != 'hits'}
            record.update(nominalCurrentAContactCells=sorted(best['hits']),
                actions=_local_contact_program(best['drillPose'], best['outsidePose'], slot, drill_seconds))
            selected.append(record)
            uncovered.difference_update(best['hits'])
            continue
        attempted_pairs += 1
        suction, threshold = _contact(engine, best['cell'], np.asarray(best['contactAnchorRasMm']),
            axis, TOOLS['frazier_suction']['radiusMm'], rho, q)
        distance = float((np.asarray(suction['position'])-plane)@axis)
        if not 0 < distance <= approach['maximumApproachMm']:
            rejections.append({'cell': best['cell'], 'tool': 'frazier_suction',
                               'reason': 'APPROACH_LENGTH_OUTSIDE_BOUND'})
            continue
        reason, witness = check_geometry('frazier_suction', suction, best['outsidePose'], allowed_work=best['hits'])
        if reason:
            rejections.append({'cell': best['cell'], 'tool': 'frazier_suction', 'reason': reason, 'witnessCell': witness})
            continue
        record = {key: value for key, value in best.items() if key != 'hits'}
        record.update(suctionPose=suction, nominalCurrentAContactCells=sorted(best['hits']),
            nativeLeadingSuctionContactDistanceMm=threshold,
            nativeRemovalRequiredBeforeSuction=sorted(witness),
            actions=_program(best['drillPose'], suction, best['outsidePose'], slot, drill_seconds, suction_seconds))
        selected.append(record)
        uncovered.difference_update(best['hits'])
    result = {'schema': SCHEMA, 'sceneId': engine.scene.identity, 'sourceStateId': before,
        'algorithm': copy.deepcopy(_ALGORITHM), 'algorithmHash': semantic_digest(_ALGORITHM),
        'sourceActionsSeal': semantic_digest(engine.actions), 'sourceBinding': contract.source_binding,
        'guardSeal': semantic_digest(engine.guard_report()),
        'protectedSourceCellsSeal': semantic_digest(sorted(protected)),
        'toolCatalogHash': semantic_digest({name: TOOLS[name] for name in ('cranial_drill', 'frazier_suction')}),
        'contractHash': contract.content_hash, 'approach': copy.deepcopy(approach),
        'bounds': {'maxExaminedCells': max_examined_cells, 'maxContacts': max_contacts,
                   'maxContactProbes': max_contact_probes},
        'examinedCells': examined, 'examinedContactCells': len(examined),
        'testedDrillContactProbes': probes, 'incompletelyExaminedCurrentACells': incomplete_cells,
        'orderingCells': approach['sourceOrdering']['orderingCells'], 'orderingCoordinatesComputedThisCall': 0,
        'matchingSuctionChecks': attempted_pairs,
        'unexaminedCurrentACells': (sorted(remaining-set(examined)) if axial else cells[len(examined):]),
        'stockGeometry': {'drillRadiusMm': 2., 'suctionRadiusMm': 1.5, 'shaftRadiusMm': .5,
                          'shaftLengthMm': 40., 'workingDepthMm': 2., 'cellCircumsphereMm': rho,
                          'nominalOverlapMm': _OVERLAP_MM},
        'contacts': selected, 'actions': [a for row in selected for a in row['actions']],
        'rejections': rejections, 'remainingOriginalACells': sorted(remaining),
        'inverseContactDiagnostics': diagnostics,
        'currentDebrisCells': sorted(int(c) for c, amount in engine.state['debris'].items() if amount > 0),
        'currentDebrisMm3': readiness['debrisPresentMm3'],
        'currentDebrisReplanRequired': readiness['debrisPresentMm3'] > 0,
        'nominalUncoveredCurrentACells': sorted(uncovered),
        'nominalWholeAContactCoverage': not uncovered,
        'controllerDeficit': bool(uncovered) or not readiness['openingComplete'],
        'nativeOpening': readiness, 'openingComplete': readiness['openingComplete'],
        'admittedOpening': False, 'nativeVerificationRequired': True, 'nativeCalls': 0,
        'scope': 'current-source geometry proposals; no removal, debris, exit or admission proof',
        'nativeObligations': ['checked staging reposition and complete approach/withdrawal sweeps',
            'finite drill work and full endpoint before suction', 'actual debris collection and complete original A',
            'recompute suction contacts from actual current debris; paired cell contact is not footprint collection',
            'unchanged shared policy/meter budgets and completed exterior exit']}
    if staging_plan is not None:
        result.update(proposalKind='staged_same_tool_current_cavity_drill',
            stagingPlanHash=staging_plan['contentHash'], requiredInstrument='cranial_drill',
            localRetreatIsCompletedWithdrawal=False,
            nativeObligations=['checked global staging and complete current-body local transitions',
                'actual full original A removal before global withdrawal and instrument exchange',
                'recompute suction from actual tagged debris; all native guards and shared costs retained'])
    if axial:
        schedule.update(actualContactProbeTests=probes, potentialTipEffectSupportQueries=effect_support_queries)
        result.update(cellSchedule='axial_diverse_current_a', cellScheduleScope='bone_only',
            axialSchedule=schedule, orderingCoordinatesComputedThisCall=len(approach['sourceOrdering']['orderedCells']),
            effectSupport='potential native spherical tip sweep; not actual removal')
    result['contentHash'] = proposal_hash(result)
    result['sourceGeometryCost'] = {'orderingCells': approach['sourceOrdering']['orderingCells'],
        'orderingWallSeconds': approach['sourceOrdering']['orderingWallSeconds'],
        'bindingWallSeconds': approach['sourceOrdering']['bindingWallSeconds'],
        'orderingReused': True, 'generationWallSeconds': time.monotonic()-started}
    if axial:
        result['sourceGeometryCost']['axialSourcePreparation'] = schedule['cachedSourcePreparation']
        result['sourceGeometryCost']['axialCurrentScheduling'] = {key: schedule[key] for key in (
            'currentMembershipVisits', 'currentColumnAssignmentsVisited', 'currentColumnActivationVisits',
            'currentFrontierDepthVisits', 'currentDepthBandVisits',
            'sourceCoordinateValidationCells', 'potentialTipEffectSupportQueries')}
    if engine.state_id != before:
        raise ValueError('MILLING_CONTACT_SOURCE_STATE_CHANGED')
    return result


def generate_milling_debris_proposals(engine, *, approach, max_examined_cells=2048,
        max_contacts=32, max_contact_probes=2048, suction_seconds, slot='left', staging_plan=None,
        check=None):
    """Propose stock suction against actual current tagged debris, not future A.

    Every candidate body/route must clear the current native cavity without
    assuming any further bone removal. Actual tags/volumes determine selection;
    native flow, other fluids, dwell and accepted exit still determine uptake.
    All current tag coordinates are explicitly charged as ordering work, then
    at most max_examined_cells receive contact/body tests. Source-wide A order
    from the bound approach is reused, never recomputed implicitly.
    suction_seconds is charged separately for approach and stationary dwell.
    """
    started = time.monotonic()
    if check is not None and not callable(check):
        raise ValueError('MILLING_CONTACT_CHECK_CALLBACK_REQUIRED')
    _geometry_check(check)
    contract, q, plane, axis, before, readiness, removed, remaining, protected, rho = _context(
        engine, approach, slot, max_examined_cells, max_contacts, max_contact_probes,
        suction_seconds, staging_plan=staging_plan)
    debris = {int(c): float(amount) for c, amount in engine.state['debris'].items() if amount > 0}
    if any(not 0 <= c < engine.scene.labels.size for c in debris):
        raise ValueError('MILLING_CONTACT_TAGGED_DEBRIS_SOURCE_CELL_INVALID')
    ordering_started = time.monotonic()
    cells = sorted(debris)
    points = engine.scene.points(cells) if cells else np.empty((0, 3))
    depths = dict(zip(cells, points@axis))
    cells = sorted(cells, key=lambda c: (round(float(depths[c]), 9), c))
    ordering_wall = time.monotonic()-ordering_started
    examined, incomplete_cells, probes, rejections, candidates = [], [], 0, [], []
    check_geometry = _geometry_checker(engine, contract, slot, removed, protected, local=staging_plan is not None)
    for cell in cells[:max_examined_cells]:
        _geometry_check(check)
        if probes >= max_contact_probes:
            break
        examined.append(cell)
        for point_index, point in _contact_points(engine, cell, approach):
            _geometry_check(check)
            if probes >= max_contact_probes:
                incomplete_cells.append(cell)
                break
            probes += 1
            record = {'cell': cell, 'pointIndex': point_index}
            contact, threshold = _contact(engine, cell, point,
                axis, TOOLS['frazier_suction']['radiusMm'], rho, q)
            distance = float((np.asarray(contact['position'])-plane)@axis)
            outside = _pose(np.asarray(contact['position'])-distance*axis, q)
            if not 0 < distance <= approach['maximumApproachMm']:
                rejections.append({**record, 'reason': 'APPROACH_LENGTH_OUTSIDE_BOUND'})
                continue
            reason, witness = check_geometry('frazier_suction', contact, outside, allowed_work=set())
            if reason:
                rejections.append({**record, 'reason': reason, 'witnessCell': witness})
                continue
            # Native uptake is the spatial spherical outlet, not the whole capsule.
            hits = set(map(int, engine.cells(contact['position'], contact['position'],
                            TOOLS['frazier_suction']['radiusMm']))) & set(debris)
            if cell not in hits:
                rejections.append({**record, 'reason': 'NO_CURRENT_TAGGED_DEBRIS_CONTACT'})
                continue
            candidates.append({**record, 'contactAnchorRasMm': point.tolist(),
                'suctionPose': contact, 'outsidePose': outside,
                'nativeLeadingSuctionContactDistanceMm': threshold, 'hits': hits})
    uncovered, selected = set(debris), []
    while candidates and len(selected) < max_contacts:
        best = min(candidates, key=lambda row: (-sum(debris[c] for c in row['hits'] & uncovered),
                                               row['cell'], row['pointIndex']))
        candidates.remove(best)
        if not best['hits'] & uncovered:
            break
        record = {key: value for key, value in best.items() if key != 'hits'}
        record.update(taggedDebrisContactCells=sorted(best['hits']),
            taggedDebrisContactVolumeMm3=sum(debris[c] for c in best['hits']),
            actions=(_program(None, best['suctionPose'], best['outsidePose'], slot, None, suction_seconds)
                if staging_plan is None else _local_contact_program(
                    best['suctionPose'], best['outsidePose'], slot, suction_seconds, suction=True)))
        selected.append(record)
        uncovered.difference_update(best['hits'])
    result = {'schema': SCHEMA, 'proposalKind': 'current_tagged_debris_suction',
        'algorithm': copy.deepcopy(_DEBRIS_ALGORITHM), 'algorithmHash': semantic_digest(_DEBRIS_ALGORITHM),
        'sceneId': engine.scene.identity, 'sourceStateId': before,
        'sourceActionsSeal': semantic_digest(engine.actions), 'sourceBinding': contract.source_binding,
        'guardSeal': semantic_digest(engine.guard_report()), 'protectedSourceCellsSeal': semantic_digest(sorted(protected)),
        'toolCatalogHash': semantic_digest({name: TOOLS[name] for name in ('cranial_drill', 'frazier_suction')}),
        'contractHash': contract.content_hash, 'approach': copy.deepcopy(approach),
        'bounds': {'maxExaminedCells': max_examined_cells, 'maxContacts': max_contacts,
                   'maxContactProbes': max_contact_probes},
        'orderingCells': len(debris), 'orderingCoordinatesComputedThisCall': len(debris),
        'stockGeometry': {'suctionRadiusMm': 1.5, 'shaftRadiusMm': .5, 'shaftLengthMm': 40.,
                          'workingDepthMm': 2., 'cellCircumsphereMm': rho, 'nominalOverlapMm': _OVERLAP_MM},
        'examinedCells': examined, 'examinedContactCells': len(examined),
        'testedSuctionContactProbes': probes, 'incompletelyExaminedTaggedDebrisCells': incomplete_cells,
        'unexaminedTaggedDebrisCells': cells[len(examined):],
        'currentTaggedDebris': [{'cell': c, 'volumeMm3': debris[c]} for c in sorted(debris)],
        'currentDebrisSeal': semantic_digest(engine.state['debris']), 'currentDebrisMm3': sum(debris.values()),
        'contacts': selected, 'actions': [a for row in selected for a in row['actions']],
        'rejections': rejections, 'nominalUncoveredTaggedDebrisCells': sorted(uncovered),
        'nominalWholeTaggedDebrisContactCoverage': not uncovered,
        'remainingOriginalACells': sorted(remaining), 'nativeOpening': readiness,
        'controllerDeficit': bool(remaining or debris) or not readiness['openingComplete'],
        'openingComplete': readiness['openingComplete'], 'admittedOpening': False,
        'nativeVerificationRequired': True, 'nativeCalls': 0,
        'scope': 'current tagged debris geometry proposals; no assumed collection or opening admission',
        'nativeObligations': ['checked current-cavity approach, uptake and complete exterior exit',
            'retain all shared-meter costs and actual fluid/debris mass accounting',
            'recompute from actual remaining tags after native execution; full original A still required']}
    if staging_plan is not None:
        result.update(stagingPlanHash=staging_plan['contentHash'], requiredInstrument='frazier_suction',
                      localRetreatIsCompletedWithdrawal=False)
    if approach.get('cellSchedule', 'surface_first') != 'surface_first':
        result.update(cellSchedule='axial_diverse_current_a', cellScheduleScope='bone_only',
                      appliedDebrisOrdering='actual_current_tagged_debris_outward_depth_then_source_id')
    result['contentHash'] = proposal_hash(result)
    result['sourceGeometryCost'] = {'orderingCells': len(debris), 'orderingWallSeconds': ordering_wall,
        'sourceApproachBindingWallSeconds': approach['sourceOrdering']['bindingWallSeconds'],
        'orderingReused': False, 'generationWallSeconds': time.monotonic()-started}
    if engine.state_id != before:
        raise ValueError('MILLING_CONTACT_SOURCE_STATE_CHANGED')
    return result
