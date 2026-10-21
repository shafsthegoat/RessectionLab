"""Whole-instrument capsule envelopes including roll-dependent jaws/blades.

Inflated midpoint capsules conservatively enclose translation and rotation over
each deterministic substep. Working geometry never shrinks the physical shaft.
"""
import numpy as np
from scipy.spatial.transform import Rotation, Slerp
from .geometry import segment_segment_distance


def envelopes(tool, parameters, position, quaternion, inflation=0.):
    p = np.asarray(position, float); r = Rotation.from_quat(quaternion)
    radius = parameters['radiusMm']; depth = parameters.get('depthMm', 2.)
    shape = tool['geometry']
    def capsule(a, b, width, part):
        return p + r.apply(a), p + r.apply(b), width + inflation, part
    result = [capsule([0, 0, -40], [0, 0, -max(1., min(depth, 5.))], tool['shaftRadiusMm'], 'shaft')]
    if shape in ('scissors', 'forceps'):
        half = parameters.get('jawOpeningMm', 2.) / 2
        result.extend(capsule([sign * half, 0, -depth], [sign * half, 0, 0], .35, 'working') for sign in (-1, 1))
    elif shape in ('blade', 'elevator', 'kerf'):
        result.extend(capsule([-radius, 0, -z], [radius, 0, -z], .4, 'working') for z in (0., min(depth, 5.)))
    else:
        result.append(capsule([0, 0, -min(depth, 3.)], [0, 0, 0], radius, 'working'))
    return result


def swept_envelopes(tool, parameters, start, end):
    a, b = np.asarray(start['position']), np.asarray(end['position'])
    rotations = Rotation.from_quat([start['quaternion'], end['quaternion']])
    angle = (rotations[1] * rotations[0].inv()).magnitude()
    # Every centerline lies within 42 mm of the tip; shortest quaternion arc.
    inflation = np.linalg.norm(b - a) / 2 + 84 * np.sin(angle / 4)
    q = Slerp([0., 1.], rotations)(.5).as_quat()
    return envelopes(tool, parameters, (a + b) / 2, q, inflation)


def pair_conflict(first, second):
    return any(segment_segment_distance(a, b, c, d) <= ar + br + 1e-9
               for a, b, ar, _ in first for c, d, br, _ in second)
