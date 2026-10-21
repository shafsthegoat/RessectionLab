"""Bounded field-cleanup proposals on an already established assistant ray.

These helpers only read current anatomy/fields. They neither execute transitions
nor clear material or ledgers. The director must preview and commit every
approach, parameter change, stationary effect and withdrawal in the shared
engine. Reachable field quantities are uncalibrated research-model volumes.
"""
from __future__ import annotations

import copy
import math
import numpy as np
from scipy.spatial.transform import Rotation


def _parameters(engine, declared):
    return engine.parameters({'instrument': 'frazier_suction', 'parameters': declared})


def _field_mass(engine, position, radius):
    cells = engine.cells(position, position, radius)
    fluid = sum(float(engine.state['fluid'].get(str(int(cell)), 0.)) for cell in cells)
    debris = sum(float(engine.state.get('debris', {}).get(str(int(cell)), 0.)) for cell in cells)
    return fluid, debris


def _same_orientation(first, second):
    return (Rotation.from_quat(first) * Rotation.from_quat(second).inv()).magnitude() <= 1e-8


def cleanup_stations(engine, field, *, max_stations=3):
    """Propose at most three positive fields along the established outward ray.

    At radius 3 mm, fresh offsets are 3/7/11 mm from the recorded field; other
    radii scale these offsets proportionally. The original field is reused if
    it still has material. Positions stop at the recorded outside plane. Their
    field spheres can overlap: quantities must be remeasured before each dwell.
    A free conservative physical tip/shaft ray is a candidate filter, never
    public trajectory or protection admission.
    """
    if type(max_stations) is not int or max_stations < 0:
        raise ValueError('max_stations must be a nonnegative integer')
    limit = min(3, max_stations)
    if not limit:
        return []
    if field.get('slot', 'right') != 'right' or field.get('instrument', 'frazier_suction') != 'frazier_suction':
        raise ValueError('Cleanup requires the established right Frazier field')
    if field.get('sourceSceneId', engine.scene.identity) != engine.scene.identity:
        raise ValueError('Cleanup field source differs from current anatomy')
    original = field.get('fieldPose', field.get('pose'))
    outside = field['outsidePose']
    start = np.asarray(original['position'], dtype=float)
    anchor = np.asarray(outside['position'], dtype=float)
    if start.shape != (3,) or anchor.shape != (3,) or not np.isfinite([start, anchor]).all():
        raise ValueError('Cleanup ray requires finite source poses')
    q = list(original['quaternion'])
    if not _same_orientation(q, outside['quaternion']):
        raise ValueError('Cleanup ray must retain its recorded orientation')
    outward = -Rotation.from_quat(q).apply([0., 0., 1.])
    delta = anchor - start
    distance = float(delta @ outward)
    if distance < -1e-8 or np.linalg.norm(delta - outward * distance) > 1e-6:
        raise ValueError('Cleanup outside anchor differs from established outward ray')
    distance = max(0., distance)
    declared = copy.deepcopy(field.get('parameters', {}))
    parameters = _parameters(engine, declared)
    radius = float(parameters['radiusMm'])
    if not math.isfinite(radius) or radius <= 0:
        raise ValueError('Cleanup field radius must be finite and positive')
    # Match the existing half-millimeter physical suction tip/shaft and the
    # ordinary engine's bounded translation microsteps, preserving a margin.
    clearance_radius = max(.67, .5 + min(.4, float(engine.scene.spacing.min()) / 3.) / 2.)
    proximal = anchor + outward * 40.
    owner = engine.occupancy()
    stations, seen = [], set()
    for offset in (0., radius, radius * 7. / 3., radius * 11. / 3.):
        offset = min(offset, distance)
        if offset in seen:
            continue
        seen.add(offset)
        point = start + outward * offset
        physical_cover = engine.cells(point, proximal, clearance_radius)
        if owner.material_ids(physical_cover):
            continue
        fluid, debris = _field_mass(engine, point, radius)
        if fluid + debris <= 1e-9:
            continue
        stations.append({'slot': 'right', 'instrument': 'frazier_suction',
            'pose': {'position': point.tolist(), 'quaternion': q[:]},
            'outsidePose': copy.deepcopy(outside), 'parameters': copy.deepcopy(declared),
            'fieldFluidMm3': fluid, 'fieldDebrisMm3': debris, 'fieldMassMm3': fluid + debris,
            'outwardOffsetMm': offset, 'sourceSceneId': engine.scene.identity,
            'checkedPhysicalRayRadiusMm': clearance_radius, 'requiresPublicPreview': True,
            'assumptions': ['Existing outward ray only; no new opening or trajectory admission.',
                'Field spheres overlap; freshly remeasure before each stationary dwell.',
                'Generic research suction field and flow; no clinical force or hemostasis claim.']})
        if len(stations) == limit:
            break
    return stations


def cleanup_dwell_action(engine, station, *, max_duration=10.):
    """Freshly measure one actual stationary Frazier pose and propose its dwell.

    The right hand must already be inserted at this pose with equivalent
    orientation; this function cannot exchange, move or teleport it. The action
    applies the station's declared field parameters through the ordinary API.
    One flow budget serves fluid first, then debris, exactly as the existing
    suction law. Reported masses are measured inputs, not promised effects.
    """
    if type(max_duration) not in (int, float) or not math.isfinite(max_duration) or max_duration <= 0:
        raise ValueError('Cleanup duration must be finite and positive')
    if station.get('sourceSceneId', engine.scene.identity) != engine.scene.identity:
        raise ValueError('Cleanup station source differs from current anatomy')
    if station.get('slot', 'right') != 'right' or station.get('instrument', 'frazier_suction') != 'frazier_suction':
        return None
    state = engine.state['slots']['right']
    pose = station['pose']
    if (state['instrument'] != 'frazier_suction' or not state['inserted'] or state['grip']
            or 'right' in engine.state['holds']
            or not np.allclose(state['position'], pose['position'], rtol=0., atol=1e-8)
            or not _same_orientation(state['quaternion'], pose['quaternion'])):
        return None
    declared = copy.deepcopy(station.get('parameters', {}))
    parameters = _parameters(engine, declared)
    radius, flow = float(parameters['radiusMm']), float(parameters['flowMm3PerSecond'])
    if not math.isfinite(flow) or flow <= 0:
        return None
    if not math.isfinite(radius) or radius <= 0:
        raise ValueError('Cleanup field radius must be finite and positive')
    fluid, debris = _field_mass(engine, state['position'], radius)
    mass = fluid + debris
    if mass <= 1e-9:
        return None
    duration = min(10., float(max_duration), mass / flow + .2)
    return {'action': {'slot': 'right', 'operation': 'apply', 'duration': duration,
                       'parameters': declared},
            'fieldFluidMm3': fluid, 'fieldDebrisMm3': debris, 'fieldMassMm3': mass,
            'plannedFlowBudgetMm3': flow * duration, 'maxDurationSeconds': min(10., float(max_duration)),
            'flowBudgetOrder': ['fluid', 'debris'], 'requiresPublicPreview': True,
            'effectQualification': 'Fresh overlapping field measurement; actual preview/step determines aspiration and residual.'}


def recorded_drill_cleanup_stations(engine, *, max_stations=48, radius_mm=3., flow_mm3_per_second=20.):
    """Rank current fields on straight, receipt-established drill access rays.

    This is a read-only proposal, with no transition/effect admission. Each ray
    starts at an accepted recorded outside placement. Accepted straight drill
    strokes establish attained depth; a partial stroke contributes only when
    BONE_CUTTING_IN_PROGRESS reports the actual stopped drill position. Generic
    collision contactPosition is the failed microstep and is never used as an
    attained endpoint. Matching complete public action history is required.

    Sample at most 128 recorded rays, 512 intervals per ray (normally <=1 mm),
    and 16384 positive candidates. Return at most 64 fields, greedily measuring
    unique current spatial fluid/debris membership. Field spheres can overlap;
    these quantities are not promised aspiration and every dwell must remeasure.
    Different rays require real withdrawal and checked exterior relocation.
    """
    from .surgical_motion import envelopes, pair_conflict
    from .surgical_tools import TOOLS

    if type(max_stations) is not int or max_stations < 0:
        raise ValueError('max_stations must be a nonnegative integer')
    for name, value, low, high in (
            ('radius_mm', radius_mm, .1, 12.),
            ('flow_mm3_per_second', flow_mm3_per_second, 0., 1000.)):
        if type(value) not in (int, float) or not math.isfinite(value) or not low <= value <= high:
            raise ValueError(name + ' is outside public suction parameter limits')
    limit = min(64, max_stations)
    if not limit or not flow_mm3_per_second:
        return []
    engine.scene.require_cache_source()
    actions, history = engine.actions, engine.history
    if len(actions) != len(history) or any(
            row.get('action') != action or row.get('status') not in ('accepted', 'partial')
            or (i and row.get('stateBefore') != history[i - 1].get('stateAfter'))
            for i, (action, row) in enumerate(zip(actions, history))):
        raise ValueError('Recorded cleanup requires complete matching public action receipts')
    if history and history[-1].get('stateAfter') != engine.state_id:
        raise ValueError('Recorded cleanup history differs from current state')

    def read_pose(value):
        if not isinstance(value, dict): return None
        try:
            point = np.asarray(value['position'], float)
            q = np.asarray(value['quaternion'], float)
            if (point.shape != (3,) or q.shape != (4,) or not np.isfinite(point).all()
                    or not np.isfinite(q).all() or abs(float(np.linalg.norm(q)) - 1.) > 1e-5):
                return None
            return point, q
        except (KeyError, TypeError, ValueError):
            return None

    tools = {'left': 'cranial_drill', 'right': 'cusa'}
    anchors, rays = {}, {}
    for index, (action, receipt) in enumerate(zip(actions, history), 1):
        slot, operation = action['slot'], action['operation']
        accepted = receipt['status'] == 'accepted'
        if operation == 'exchange':
            tools[slot] = action['instrument'] if accepted else None
            anchors.pop(slot, None)
        if tools.get(slot) != 'cranial_drill': continue
        if operation == 'place':
            parsed = read_pose(action.get('pose')) if accepted else None
            if parsed is None:
                anchors.pop(slot, None)
            else:
                point, q = parsed
                anchors[slot] = {'outsidePose': {'position': point.tolist(), 'quaternion': q.tolist()}, 'origin': point,
                    'q': q, 'axis': Rotation.from_quat(q).apply([0., 0., 1.]),
                    'placement': index}
            continue
        if operation == 'withdraw':
            anchors.pop(slot, None)
            continue
        anchor = anchors.get(slot)
        if anchor is None: continue
        points = [action['pose']] if 'pose' in action else action.get('path', [])
        if not points: continue
        parsed = [read_pose(pose) for pose in points]
        if any(value is None for value in parsed):
            anchors.pop(slot, None); continue
        origin, axis = anchor['origin'], anchor['axis']

        def depth_on_ray(point, q):
            delta = point - origin; depth = float(delta @ axis)
            if (depth < -1e-8 or np.linalg.norm(delta - depth * axis) > 1e-6
                    or not _same_orientation(q, anchor['q'])):
                return None
            return max(0., depth)

        depths = [depth_on_ray(point, q) for point, q in parsed]
        if any(depth is None for depth in depths):
            anchors.pop(slot, None); continue
        if not accepted:
            # For other contact stops, contactPosition is not the attained tip.
            if receipt.get('reasonCode', receipt.get('reason')) != 'BONE_CUTTING_IN_PROGRESS':
                anchors.pop(slot, None); continue
            stopped = read_pose({'position': receipt.get('contactPosition'), 'quaternion': anchor['q']})
            stopped_depth = depth_on_ray(*stopped) if stopped is not None else None
            if stopped_depth is None or stopped_depth > max(depths) + 1e-6:
                anchors.pop(slot, None); continue
            depths = [stopped_depth]
        if operation != 'apply' or max(depths) <= 1e-9: continue
        key = tuple(origin) + tuple(axis)
        ray = rays.get(key)
        if ray is None:
            if len(rays) >= 128: continue
            ray = {'outsidePose': anchor['outsidePose'], 'axis': axis,
                   'maximumReceiptAttainedDepthMm': 0., 'drillPlacementActions1Based': [],
                   'drillApplyActions1Based': [], 'partialDepthEvidence': []}
            rays[key] = ray
        ray['maximumReceiptAttainedDepthMm'] = max(ray['maximumReceiptAttainedDepthMm'], max(depths))
        if anchor['placement'] not in ray['drillPlacementActions1Based']:
            ray['drillPlacementActions1Based'].append(anchor['placement'])
        ray['drillApplyActions1Based'].append(index)
        if not accepted:
            ray['partialDepthEvidence'].append({'actionPosition1Based': index,
                'reason': 'BONE_CUTTING_IN_PROGRESS', 'attainedDepthMm': depths[0],
                'requestedDepthMm': max(depth_on_ray(point, q) for point, q in parsed)})

    field_values = {}
    for name in ('fluid', 'debris'):
        for cell, amount in engine.state.get(name, {}).items():
            if amount > 0.: field_values[int(cell)] = field_values.get(int(cell), 0.) + float(amount)
    if not field_values:
        return []
    owner = engine.occupancy()
    parameters = {'radiusMm': float(radius_mm), 'depthMm': 2., 'forceN': 0.,
                  'flowMm3PerSecond': float(flow_mm3_per_second)}
    # Public collision geometry includes the working field radius even though
    # the static material corridor is the physical half-mm suction tip/shaft.
    margin = min(.4, float(engine.scene.spacing.min()) / 3.) / 2.
    physical_radius = max(.67, .5 + margin)
    other = engine.state['slots']['left']
    other_envelopes = envelopes(TOOLS[other['instrument']], engine.parameters(other),
                                other['position'], other['quaternion'])
    candidates, seen = [], set()
    for ray in rays.values():
        outside = ray['outsidePose']; origin = np.asarray(outside['position'], float)
        axis = ray['axis']; proximal = origin - 40. * axis
        depth = ray['maximumReceiptAttainedDepthMm']
        intervals = max(1, min(512, math.ceil(depth)))
        for distance in np.linspace(0., depth, intervals + 1):
            point = origin + axis * distance
            key = tuple(point) + tuple(axis)
            if key in seen: continue
            seen.add(key)
            spatial = engine.cells(point, point, radius_mm)
            ids = set(map(int, spatial)).intersection(field_values)
            if not ids: continue
            if owner.material_ids(engine.cells(point, proximal, physical_radius)): continue
            # This encloses every constant-orientation insertion microstep.
            if pair_conflict([(point, proximal, max(radius_mm + margin, physical_radius), 'working')], other_envelopes):
                continue
            fluid, debris = _field_mass(engine, point, radius_mm)
            if fluid + debris <= 1e-9: continue
            candidates.append({'_ids': ids, '_order': len(candidates), 'slot': 'right',
                'instrument': 'frazier_suction', 'pose': {'position': point.tolist(),
                    'quaternion': copy.deepcopy(outside['quaternion'])},
                'outsidePose': copy.deepcopy(outside), 'parameters': parameters.copy(),
                'sourceSceneId': engine.scene.identity, 'sceneId': engine.scene.identity,
                'fieldFluidMm3': fluid, 'fieldDebrisMm3': debris, 'fieldMassMm3': fluid + debris,
                'depthFromRecordedOutsideMm': float(distance), 'sampleSpacingMm': depth / intervals,
                'maximumReceiptAttainedDepthMm': depth,
                'drillPlacementActions1Based': ray['drillPlacementActions1Based'][:],
                'drillApplyActions1Based': ray['drillApplyActions1Based'][:],
                'partialDepthEvidence': copy.deepcopy(ray['partialDepthEvidence']),
                'checkedPhysicalRayRadiusMm': physical_radius, 'requiresPublicPreview': True,
                'assumptions': ['Matching saved public drill receipts; no partial requested endpoint inferred.',
                    'Current physical material corridor and other-hand envelope filter only; no protection or trajectory admission.',
                    'Withdraw and use checked exterior transit before changing rays; never travel directly between intracranial fields.',
                    'Overlapping generic research fields; fresh dwell measurement and actual public effects determine residual.',
                    'No new opening, calibrated force, complete clearance or hemostasis claim.']})
            if len(candidates) >= 16384: break
        if len(candidates) >= 16384: break
    covered, ranked = set(), []
    while candidates and len(ranked) < limit:
        def gain(candidate):
            return sum(field_values[cell] for cell in sorted(candidate['_ids'] - covered))
        best = max(candidates, key=lambda candidate: (gain(candidate), candidate['fieldMassMm3'], -candidate['_order']))
        amount = gain(best)
        if amount <= 1e-9: break
        candidates.remove(best)
        covered.update(best.pop('_ids')); best.pop('_order')
        best['newUniqueFieldMassMm3'] = amount
        best['rank'] = len(ranked) + 1
        ranked.append(best)
    engine.scene.require_cache_source()
    return ranked
