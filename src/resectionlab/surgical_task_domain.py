"""Opt-in reviewed task boxes for desktop instrument commands.

The box limits where an entire tool may be manipulated. It is not anatomy,
material, clinical coverage, or a retrospective certificate for saved archives.
Production capsule geometry is read unchanged; conservative covers can refuse
otherwise valid paths. Scenes without the declaration retain existing behavior.
"""
from __future__ import annotations

import copy
from dataclasses import dataclass
import math
import re
import numpy as np
from scipy.spatial.transform import Rotation, Slerp

from .surgical_motion import envelopes, swept_envelopes
from .surgical_tools import TOOLS

PROPERTY = 'reviewedTaskDomainV1'
FIELDS = {'mode', 'shape', 'minimumVoxelBoundary', 'maximumVoxelBoundary',
          'reviewEvidenceSha256', 'maskSha256'}
BRIDGE_COMMANDS = frozenset({'create', 'inspect', 'inspectArchive', 'preview', 'step',
    'undo', 'redo', 'seek', 'selectBranch', 'branch', 'reset', 'seekFrame',
    'seekPrefix', 'seekPrefixFrame', 'selectScene'})
POSE_OPERATIONS = frozenset({'place', 'move', 'apply', 'withdraw', 'hold', 'grasp', 'bladeManeuver'})
MAX_INTERVALS = 20_000


@dataclass(frozen=True)
class TaskDomain:
    minimum: tuple[float, float, float]
    maximum: tuple[float, float, float]
    review_sha256: str
    mask_sha256: str


def reviewed_domain(scene):
    """Validate current native-grid binding; never infer/rebind refined indices."""
    adapter = scene.source.get('adapter')
    if not isinstance(adapter, dict) or PROPERTY not in adapter:
        return None
    raw = adapter[PROPERTY]
    prefix = 'REVIEWED_TASK_DOMAIN_INVALID: '
    if not isinstance(raw, dict) or set(raw) != FIELDS or raw.get('mode') != 'restrict-unknown':
        raise ValueError(prefix + 'exact restrict-unknown declaration required')
    shape = raw['shape']
    if (not isinstance(shape, list) or len(shape) != 3 or any(type(n) is not int or n <= 0 for n in shape)
            or tuple(shape) != tuple(scene.shape)):
        raise ValueError(prefix + 'shape differs from the current saved native grid')
    bounds = []
    for name in ('minimumVoxelBoundary', 'maximumVoxelBoundary'):
        value = raw[name]
        if (not isinstance(value, list) or len(value) != 3
                or any(type(x) not in (int, float) or not math.isfinite(x) for x in value)):
            raise ValueError(prefix + 'finite three-axis voxel-face bounds required')
        bounds.append(np.asarray(value, float))
    low, high = bounds
    if np.any(low < -.5) or np.any(high > np.asarray(shape) - .5) or np.any(low >= high):
        raise ValueError(prefix + 'box must lie inside observed native voxel faces')
    for name in ('reviewEvidenceSha256', 'maskSha256'):
        if not isinstance(raw[name], str) or not re.fullmatch('[a-f0-9]{64}', raw[name]):
            raise ValueError(prefix + 'raw SHA256 evidence bindings required')
    return TaskDomain(tuple(low), tuple(high), raw['reviewEvidenceSha256'], raw['maskSha256'])


def _rejection(engine, reason, code='REVIEWED_TASK_DOMAIN_OUTSIDE'):
    return {'status': 'rejected', 'reason': code + ': ' + reason, 'reasonCode': code,
            'completedFraction': 0., 'elapsedSeconds': 0., 'stateId': engine.state_id,
            'events': [], 'deltas': {}, 'metrics': engine.metrics()}


def _outside(scene, domain, segments):
    inverse = np.asarray(scene.inverse)
    margins = np.linalg.norm(inverse[:3, :3], axis=1)
    for a, b, radius, part in segments:
        points = np.asarray([a, b]) @ inverse[:3, :3].T + inverse[:3, 3]
        low = points.min(0) - float(radius) * margins
        high = points.max(0) + float(radius) * margins
        if np.any(low < np.asarray(domain.minimum) - 1e-9) or np.any(high > np.asarray(domain.maximum) + 1e-9):
            return str(part) + ' capsule extends outside the reviewed task box; outside is unknown'
    return None


def _posed(engine, key):
    slot = engine.state['slots'][key]
    return bool(slot['active'] or slot['inserted'] or slot['grip'] or key in engine.state['holds']
                or any(a.get('slot') == key and a.get('operation') in POSE_OPERATIONS for a in engine.actions))


def action_rejection(engine, action, *, domain=None, check=lambda: None):
    """Return ordinary atomic refusal, or None. No engine/action/catalog mutation.

    Unplaced inactive initial slots have no acted-on physical pose. Once posed,
    both hands are checked even while parked. Placement checks its endpoint only,
    matching the engine's deliberate free-space placement semantics.
    """
    domain = reviewed_domain(engine.scene) if domain is None else domain
    if domain is None:
        return None
    check()
    try:
        _, points = engine._validate(action)  # Existing validation is read-only.
    except (ValueError, KeyError, TypeError, OverflowError):
        return None  # Preserve the engine's existing invalid-action response.
    key, operation = action['slot'], action['operation']
    if operation == 'bladeManeuver':
        return _rejection(engine, 'internally generated planar paths are not admitted in V1',
                          'REVIEWED_TASK_DOMAIN_UNSUPPORTED_OPERATION')
    current = engine.state['slots'][key]
    proposed = copy.deepcopy(current)
    if operation == 'exchange':
        if action.get('instrument') not in TOOLS:
            return None
        proposed['instrument'] = action['instrument']; proposed['parameters'] = {}
    proposed['parameters'].update(action.get('parameters', {}))
    parameters = engine.parameters(proposed)
    # Jaw articulation is monotonic in the engine; maximal opening covers every
    # intermediate jaw, without changing requested/catalog tool dimensions.
    parameters['jawOpeningMm'] = max(parameters['jawOpeningMm'], engine.parameters(current)['jawOpeningMm'])
    tool = TOOLS[proposed['instrument']]
    stationary = []
    for other, slot in engine.state['slots'].items():
        if other != key and _posed(engine, other):
            stationary.extend(envelopes(TOOLS[slot['instrument']], engine.parameters(slot), slot['position'], slot['quaternion']))
    if _posed(engine, key):
        stationary.extend(envelopes(TOOLS[current['instrument']], engine.parameters(current), current['position'], current['quaternion']))
    reason = _outside(engine.scene, domain, stationary)
    if reason:
        return _rejection(engine, reason)
    if operation in {'configure', 'position', 'exchange'} and not _posed(engine, key):
        return None
    poses = copy.deepcopy(points)
    if operation == 'place':
        if len(poses) != 1:
            return None
        return (_rejection(engine, reason) if (reason := _outside(engine.scene, domain,
                envelopes(tool, parameters, poses[0]['position'], poses[0]['quaternion']))) else None)
    if operation == 'withdraw' and not poses:
        axis = Rotation.from_quat(current['quaternion']).apply([0, 0, 1])
        poses = [{'position': (np.asarray(current['position']) - 50 * axis).tolist(), 'quaternion': current['quaternion']}]
    if not poses:
        poses = [{'position': current['position'], 'quaternion': current['quaternion']}]
    previous = current; intervals = 0
    for end in poses:
        a, b = np.asarray(previous['position']), np.asarray(end['position'])
        rotations = Rotation.from_quat([previous['quaternion'], end['quaternion']])
        angle = float((rotations[1] * rotations[0].inv()).magnitude())
        count = max(1, int(math.ceil(np.linalg.norm(b - a) / .25)), int(math.ceil(angle / np.deg2rad(1.))))
        intervals += count
        if intervals > MAX_INTERVALS:
            return _rejection(engine, 'split the path into shorter reviewed actions', 'REVIEWED_TASK_DOMAIN_PATH_BUDGET')
        interpolate = Slerp([0., 1.], rotations)
        for fraction in np.linspace(0., 1., count + 1)[1:]:
            check()
            pose = {'position': ((1 - fraction) * a + fraction * b).tolist(), 'quaternion': interpolate(fraction).as_quat().tolist()}
            reason = _outside(engine.scene, domain, swept_envelopes(tool, parameters, previous, pose))
            if reason:
                return _rejection(engine, reason)
            previous = pose
    return None
