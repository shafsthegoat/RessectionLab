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
from itertools import product

import numpy as np
from scipy.spatial.transform import Rotation

from .core import semantic_digest
from .geometry import _segment_cell_distances
from .surgical_craniotomy import _original_targets
from .surgical_cranial_guards import _inside_entry
from .surgical_engine import SurgicalEngine
from .surgical_milling import is_milling, milling_readiness, validate_source_scope
from .surgical_motion import envelopes, pair_conflict
from .surgical_tools import TOOLS
from .surgical_transition_meter import MeteredEngine

SCHEMA = 'resectionlab.milling-contact-proposals.v1'
APPROACH_SCHEMA = 'resectionlab.milling-contact-approach.v1'
_OVERLAP_MM = .10
_ALGORITHM = {'name': 'native-orthogonal-cell-ray-inverse-v1',
    'kernel': 'geometry._segment_cell_distances', 'iterations': 64,
    'bracket': '[0, stockRadius+cellCircumsphere+1e-6] mm',
    'pose': 'cellCenter-axis*(nativeLeadingContactDistance-0.10mm)',
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


def bind_milling_approach(engine, *, quaternion, outside_plane_point_ras_mm,
                          maximum_approach_mm):
    """Bind explicitly supplied fixed attitude and exterior plane to the Scene.

    This records a proposal route, not a reviewed exterior/clearance proof.
    Per-contact exterior/body/workspace tests run during generation; native
    verification must still check staging repositioning and every motion.
    """
    binding_started = time.monotonic()
    contract = validate_source_scope(engine)
    if not is_milling(contract):
        raise ValueError('MILLING_CONTACT_MILLING_CONTRACT_REQUIRED')
    raw = {'schema': APPROACH_SCHEMA, 'sceneId': engine.scene.identity,
           'contractHash': contract.content_hash, 'quaternion': list(quaternion),
           'outsidePlanePointRasMm': list(outside_plane_point_ras_mm),
           'maximumApproachMm': maximum_approach_mm}
    _, _, axis = _approach(engine, contract, raw, require_order=False)
    started = time.monotonic()
    cells = list(contract.aperture_cells)
    points = engine.scene.points(cells)
    depths = dict(zip(cells, points@axis))
    ordering = {'sceneId': engine.scene.identity, 'contractHash': contract.content_hash,
        'quaternion': list(quaternion), 'orderedCells': sorted(cells,
            key=lambda c: (round(float(depths[c]), 9), c)),
        'orderingCells': len(cells), 'orderingWallSeconds': time.monotonic()-started,
        'bindingWallSeconds': time.monotonic()-binding_started}
    raw['sourceOrdering'] = ordering
    raw['sourceOrderingHash'] = _ordering_hash(ordering)
    return raw


def _approach(engine, contract, raw, *, require_order=True):
    if engine._cranial_access != contract:
        raise ValueError('MILLING_CONTACT_NATIVE_CONTRACT_BINDING_DIFFERS')
    fields = {'schema', 'sceneId', 'contractHash', 'quaternion',
              'outsidePlanePointRasMm', 'maximumApproachMm'}
    if require_order:
        fields.update(('sourceOrdering', 'sourceOrderingHash'))
    if (not isinstance(raw, dict) or set(raw) != fields or
            raw['schema'] != APPROACH_SCHEMA or raw['sceneId'] != engine.scene.identity or
            raw['contractHash'] != contract.content_hash):
        raise ValueError('MILLING_CONTACT_APPROACH_SOURCE_BINDING_REQUIRED')
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
        if (not isinstance(ordering, dict) or set(ordering) != {'sceneId', 'contractHash', 'quaternion',
                'orderedCells', 'orderingCells', 'orderingWallSeconds', 'bindingWallSeconds'} or
                raw['sourceOrderingHash'] != _ordering_hash(ordering) or
                ordering['sceneId'] != engine.scene.identity or ordering['contractHash'] != contract.content_hash or
                ordering['quaternion'] != raw['quaternion'] or
                ordering['orderingCells'] != len(contract.aperture_cells) or
                sorted(ordering['orderedCells']) != list(contract.aperture_cells) or
                not _finite_positive(ordering['orderingWallSeconds']) or
                not _finite_positive(ordering['bindingWallSeconds'])):
            raise ValueError('MILLING_CONTACT_SOURCE_ORDERING_BINDING_REQUIRED')
    return q, point, axis


def _pose(position, quaternion):
    return {'position': np.asarray(position).tolist(), 'quaternion': np.asarray(quaternion).tolist()}


def _contact(engine, cell, point, axis, radius, rho, q):
    """Invert the existing exact orthogonal-cell distance on an outward ray."""
    index = np.asarray([np.unravel_index(cell, engine.scene.shape)], dtype=np.int64)
    lo, hi = 0., radius+rho+1e-6
    for _ in range(_ALGORITHM['iterations']):
        middle = (lo+hi)/2
        tip = point-axis*middle
        if float(_segment_cell_distances(engine.grid, index, tip, tip)[0]) <= radius:
            lo = middle
        else:
            hi = middle
    threshold = (lo+hi)/2
    pose = _pose(point-axis*(threshold-_OVERLAP_MM), q)
    return pose, threshold


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


def _context(engine, approach, slot, max_examined_cells, max_contacts, *durations):
    if not isinstance(engine, (SurgicalEngine, MeteredEngine)):
        raise TypeError('MILLING_CONTACT_NATIVE_ENGINE_REQUIRED')
    if (type(max_examined_cells) is not int or not 1 <= max_examined_cells <= 2048 or
            type(max_contacts) is not int or not 1 <= max_contacts <= 32):
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
    if engine.grid._orthogonal_spacing is None:
        raise ValueError('MILLING_CONTACT_SHEARED_AFFINE_CONTACT_UNSUPPORTED')
    if slot not in engine.state['slots'] or any(s['inserted'] or s['active'] or s['grip']
            for s in engine.state['slots'].values()):
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
    # Maximum corner norm is valid for oblique AND sheared affine cells.
    corners = np.asarray(list(product((-.5, .5), repeat=3)))@engine.scene.affine[:3, :3].T
    rho = float(np.linalg.norm(corners, axis=1).max())
    return contract, q, plane, axis, before, readiness, removed, remaining, protected, rho


def _geometry_checker(engine, contract, slot, removed, protected):
    def check_geometry(name, contact, outside, *, allowed_work):
        route = _route(name, outside, contact)
        if not _exterior(engine.scene, contract, _segments(name, outside)):
            return 'OUTSIDE_POSE_NOT_AT_ORIGINAL_EXIT', None
        if not _within_workspace(engine.scene, contract, route):
            return 'OUTSIDE_REVIEWED_WORKSPACE', None
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
            occupied = {c for c in touched-removed if engine.scene.labels.ravel()[c] != 0}
            if part == 'shaft' and occupied:
                return 'CURRENT_CAVITY_SHAFT_OCCUPIED', min(occupied)
            if part != 'shaft':
                if occupied-allowed_work:
                    return 'WORKING_CONTACT_OUTSIDE_ALLOWED_CURRENT_A', min(occupied-allowed_work)
                hits.update(occupied)
        return None, hits

    return check_geometry


def generate_milling_contact_proposals(engine, *, approach, max_examined_cells=2048,
        max_contacts=32, drill_seconds, suction_seconds, slot='left'):
    """Return JSON paired contacts and one native program from the current state.

    Bounds count cell-centred drill candidates and selected matching suction
    checks (at most 2048 + 32); source binding/readiness scans are not native
    calls. No evaluator trajectories or nominal/deepest-column stops enter.
    Full-A residual IDs are retained even when the candidate bound truncates.
    Use ``actions`` with existing checked policies, whose completion gates and
    shared prospective budgets are unchanged. Any refused suction pair is
    retained; there is no extra reselection loop beyond the selected bound.
    """
    started = time.monotonic()
    contract, q, plane, axis, before, readiness, removed, remaining, protected, rho = _context(
        engine, approach, slot, max_examined_cells, max_contacts, drill_seconds, suction_seconds)
    cells = [cell for cell in approach['sourceOrdering']['orderedCells'] if cell in remaining]
    examined = cells[:max_examined_cells]
    rejections, candidates, diagnostics = [], [], []

    check_geometry = _geometry_checker(engine, contract, slot, removed, protected)

    for cell in examined:
        point = engine.scene.points([cell])[0]
        drill, threshold = _contact(engine, cell, point, axis, TOOLS['cranial_drill']['radiusMm'], rho, q)
        prior = _pose(point-axis*(TOOLS['cranial_drill']['radiusMm']+rho-_OVERLAP_MM), q)
        index = np.asarray([np.unravel_index(cell, engine.scene.shape)], dtype=np.int64)
        prior_distance = float(_segment_cell_distances(engine.grid, index,
            np.asarray(prior['position']), np.asarray(prior['position']))[0])
        diagnostics.append({'cell': cell, 'circumsphereOnlyDrillPose': prior,
            'circumsphereOnlyNativeCellHit': prior_distance <= TOOLS['cranial_drill']['radiusMm']+1e-9,
            'nativeLeadingDrillContactDistanceMm': threshold})
        distance = float((np.asarray(drill['position'])-plane)@axis)
        outside = _pose(np.asarray(drill['position'])-distance*axis, q)
        if not 0 < distance <= approach['maximumApproachMm']:
            rejections.append({'cell': cell, 'reason': 'APPROACH_LENGTH_OUTSIDE_BOUND'})
            continue
        reason, hits = check_geometry('cranial_drill', drill, outside, allowed_work=remaining)
        if reason:
            rejections.append({'cell': cell, 'reason': reason, 'witnessCell': hits})
            continue
        if cell not in hits:
            rejections.append({'cell': cell, 'reason': 'NO_CURRENT_A_CONTACT'})
            continue
        candidates.append({'cell': cell, 'drillPose': drill, 'outsidePose': outside, 'hits': hits,
                           'nativeLeadingDrillContactDistanceMm': threshold})
    uncovered, selected, attempted_pairs = set(remaining), [], 0
    while candidates and attempted_pairs < max_contacts:
        best = min(candidates, key=lambda row: (-len(row['hits'] & uncovered), row['cell']))
        candidates.remove(best)
        if not best['hits'] & uncovered:
            break
        attempted_pairs += 1
        suction, threshold = _contact(engine, best['cell'], engine.scene.points([best['cell']])[0],
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
        'bounds': {'maxExaminedCells': max_examined_cells, 'maxContacts': max_contacts},
        'examinedCells': examined, 'examinedContactCells': len(examined),
        'orderingCells': approach['sourceOrdering']['orderingCells'], 'orderingCoordinatesComputedThisCall': 0,
        'matchingSuctionChecks': attempted_pairs,
        'unexaminedCurrentACells': cells[max_examined_cells:],
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
    result['contentHash'] = proposal_hash(result)
    result['sourceGeometryCost'] = {'orderingCells': approach['sourceOrdering']['orderingCells'],
        'orderingWallSeconds': approach['sourceOrdering']['orderingWallSeconds'],
        'bindingWallSeconds': approach['sourceOrdering']['bindingWallSeconds'],
        'orderingReused': True, 'generationWallSeconds': time.monotonic()-started}
    if engine.state_id != before:
        raise ValueError('MILLING_CONTACT_SOURCE_STATE_CHANGED')
    return result


def generate_milling_debris_proposals(engine, *, approach, max_examined_cells=2048,
        max_contacts=32, suction_seconds, slot='left'):
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
    contract, q, plane, axis, before, readiness, removed, remaining, protected, rho = _context(
        engine, approach, slot, max_examined_cells, max_contacts, suction_seconds)
    debris = {int(c): float(amount) for c, amount in engine.state['debris'].items() if amount > 0}
    if any(not 0 <= c < engine.scene.labels.size for c in debris):
        raise ValueError('MILLING_CONTACT_TAGGED_DEBRIS_SOURCE_CELL_INVALID')
    ordering_started = time.monotonic()
    cells = sorted(debris)
    points = engine.scene.points(cells) if cells else np.empty((0, 3))
    depths = dict(zip(cells, points@axis))
    cells = sorted(cells, key=lambda c: (round(float(depths[c]), 9), c))
    ordering_wall = time.monotonic()-ordering_started
    examined, rejections, candidates = cells[:max_examined_cells], [], []
    check_geometry = _geometry_checker(engine, contract, slot, removed, protected)
    for cell in examined:
        contact, threshold = _contact(engine, cell, engine.scene.points([cell])[0],
            axis, TOOLS['frazier_suction']['radiusMm'], rho, q)
        distance = float((np.asarray(contact['position'])-plane)@axis)
        outside = _pose(np.asarray(contact['position'])-distance*axis, q)
        if not 0 < distance <= approach['maximumApproachMm']:
            rejections.append({'cell': cell, 'reason': 'APPROACH_LENGTH_OUTSIDE_BOUND'})
            continue
        reason, witness = check_geometry('frazier_suction', contact, outside, allowed_work=set())
        if reason:
            rejections.append({'cell': cell, 'reason': reason, 'witnessCell': witness})
            continue
        # Native suction uptake uses the spherical spatial outlet, not the
        # entire working capsule. Existing native kernel supplies its cells.
        hits = set(map(int, engine.cells(contact['position'], contact['position'],
                        TOOLS['frazier_suction']['radiusMm']))) & set(debris)
        if cell not in hits:
            rejections.append({'cell': cell, 'reason': 'NO_CURRENT_TAGGED_DEBRIS_CONTACT'})
            continue
        candidates.append({'cell': cell, 'suctionPose': contact, 'outsidePose': outside,
                           'nativeLeadingSuctionContactDistanceMm': threshold, 'hits': hits})
    uncovered, selected = set(debris), []
    while candidates and len(selected) < max_contacts:
        best = min(candidates, key=lambda row: (-sum(debris[c] for c in row['hits'] & uncovered), row['cell']))
        candidates.remove(best)
        if not best['hits'] & uncovered:
            break
        record = {key: value for key, value in best.items() if key != 'hits'}
        record.update(taggedDebrisContactCells=sorted(best['hits']),
            taggedDebrisContactVolumeMm3=sum(debris[c] for c in best['hits']),
            actions=_program(None, best['suctionPose'], best['outsidePose'], slot, None, suction_seconds))
        selected.append(record)
        uncovered.difference_update(best['hits'])
    result = {'schema': SCHEMA, 'proposalKind': 'current_tagged_debris_suction',
        'algorithm': copy.deepcopy(_DEBRIS_ALGORITHM), 'algorithmHash': semantic_digest(_DEBRIS_ALGORITHM),
        'sceneId': engine.scene.identity, 'sourceStateId': before,
        'sourceActionsSeal': semantic_digest(engine.actions), 'sourceBinding': contract.source_binding,
        'guardSeal': semantic_digest(engine.guard_report()), 'protectedSourceCellsSeal': semantic_digest(sorted(protected)),
        'toolCatalogHash': semantic_digest({name: TOOLS[name] for name in ('cranial_drill', 'frazier_suction')}),
        'contractHash': contract.content_hash, 'approach': copy.deepcopy(approach),
        'bounds': {'maxExaminedCells': max_examined_cells, 'maxContacts': max_contacts},
        'orderingCells': len(debris), 'orderingCoordinatesComputedThisCall': len(debris),
        'stockGeometry': {'suctionRadiusMm': 1.5, 'shaftRadiusMm': .5, 'shaftLengthMm': 40.,
                          'workingDepthMm': 2., 'cellCircumsphereMm': rho, 'nominalOverlapMm': _OVERLAP_MM},
        'examinedCells': examined, 'examinedContactCells': len(examined),
        'unexaminedTaggedDebrisCells': cells[max_examined_cells:],
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
    result['contentHash'] = proposal_hash(result)
    result['sourceGeometryCost'] = {'orderingCells': len(debris), 'orderingWallSeconds': ordering_wall,
        'sourceApproachBindingWallSeconds': approach['sourceOrdering']['bindingWallSeconds'],
        'orderingReused': False, 'generationWallSeconds': time.monotonic()-started}
    if engine.state_id != before:
        raise ValueError('MILLING_CONTACT_SOURCE_STATE_CHANGED')
    return result
