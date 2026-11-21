"""Declared external hand positions for the generated operative exposure model."""
from __future__ import annotations

import numpy as np
from scipy.spatial.transform import Rotation

from .surgical_motion import envelopes


def plane(scene):
    exterior = scene.properties['operativeExposureV1'].get('exterior') or {}
    point = np.asarray(exterior.get('pointRasMm'), dtype=float)
    normal = np.asarray(exterior.get('normalRas'), dtype=float)
    if (point.shape != (3,) or normal.shape != (3,) or not np.isfinite(point).all()
            or not np.isfinite(normal).all() or not np.isclose(np.linalg.norm(normal), 1)):
        raise ValueError('OPERATIVE_EXTERIOR_PLANE_REQUIRED')
    return point, normal


def parked_slots(engine):
    """Start both uninserted instruments fully outside the declared operative surface."""
    point, normal = plane(engine.scene)
    direction = -normal
    lateral = np.array([1., 0., 0.])
    if abs(lateral @ normal) > .9:
        lateral = np.array([0., 1., 0.])
    lateral -= normal * (lateral @ normal)
    lateral /= np.linalg.norm(lateral)
    orientation = Rotation.from_matrix(np.column_stack((lateral, np.cross(direction, lateral), direction)))
    for hand, sign in [('left', -1), ('right', 1)]:
        slot = engine.state['slots'][hand]
        slot.update(position=(point + normal * 15. + sign * lateral * 32.).tolist(),
                    quaternion=orientation.as_quat().tolist())


def require_exterior(engine, key, pose):
    point, normal = plane(engine.scene)
    slot = engine.state['slots'][key]
    tool = engine.tools[slot['instrument']]
    for a, b, radius, _ in envelopes(tool, engine.parameters(slot), pose['position'], pose['quaternion']):
        if min((a - point) @ normal, (b - point) @ normal) < radius:
            raise ValueError('OPERATIVE_EXTERIOR_REQUIRED: clear the full instrument above the entry surface')
        if engine.touched(a, b, radius):
            raise ValueError('OPERATIVE_EXTERIOR_OCCUPIED')
