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
