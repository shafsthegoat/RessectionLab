"""Fixed three-point support for the generated bone flap at its external parking site."""
from __future__ import annotations

import numpy as np

from .surgical_motion import pair_conflict

SCHEMA = 'resectionlab.bone-parking-rest.v1'


def rest_declaration(shape, affine, body):
    """Fit three stops to the lower support envelope of the intact rigid source cells."""
    basis = np.asarray(affine, dtype=float)[:3, :3]
    cell_radius = max(np.linalg.norm(basis @ np.array([x, y, z]))
                      for x in (-.5, .5) for y in (-.5, .5) for z in (-.5, .5))
    normal = np.asarray(body['outward'], dtype=float)
    lateral = np.asarray(body['hingeAxis'], dtype=float)
    front = np.cross(normal, lateral)
    height = np.asarray(body['parkCenterMm'], dtype=float)
    cells = np.array(np.unravel_index(body['cells'], shape)).T @ basis.T + np.asarray(affine)[:3, 3]
    positions = cells + height - np.asarray(body['restCenterMm'])
    # Three non-collinear support points lie inside the rim; the authored
    # pad tips are tangent to the conservative voxel bounds in this orientation.
    extent = .27 * min(np.ptp(cells @ front), np.ptp(cells @ lateral))
    pad_radius = .85
    pads = []
    for x, y in ((-.85, .50), (.85, .50), (0., -1.)):
        location = height + extent * (x * lateral + y * front)
        delta = positions - location
        axial = delta @ normal
        horizontal_sq = np.sum((delta - axial[:, None] * normal)**2, axis=1)
        combined = cell_radius + pad_radius
        within = horizontal_sq < combined**2
        if not within.any():
            raise ValueError('OPERATIVE_REST_REQUIRES_INFERIOR_BONE_SUPPORT')
        tangent = np.min(axial[within] - np.sqrt(combined**2 - horizontal_sq[within]))
        top = location + tangent * normal
        pads.append({'aMm':(top - normal*2.5).tolist(), 'bMm':top.tolist(), 'radiusMm':pad_radius})
    return {'schema':SCHEMA, 'cellRadiusMm':float(cell_radius), 'contactToleranceMm':.18, 'pads':pads}


def initialize(engine):
    cfg = engine.scene.properties['operativeExposureV1']
    rest = cfg.get('boneRest')
    if rest is None:
        return
    expected = rest_declaration(engine.scene.shape, engine.scene.affine, cfg['bodies']['bone'])
    if (not isinstance(rest, dict) or set(rest) != set(expected) or rest['schema'] != SCHEMA
            or rest['contactToleranceMm'] != expected['contactToleranceMm']
            or not np.isclose(rest['cellRadiusMm'], expected['cellRadiusMm'], rtol=0., atol=1e-8)
            or len(rest['pads']) != 3):
        raise ValueError('OPERATIVE_REST_BINDING_MISMATCH')
    for actual, authored in zip(rest['pads'], expected['pads']):
        if (set(actual) != set(authored) or actual['radiusMm'] != authored['radiusMm']
                or any(np.shape(actual[key]) != (3,) or not np.allclose(actual[key], authored[key], rtol=0., atol=1e-8)
                       for key in ('aMm', 'bMm'))):
            raise ValueError('OPERATIVE_REST_BINDING_MISMATCH')
    engine._bone_rest = tuple((np.asarray(row['aMm']), np.asarray(row['bMm']), row['radiusMm'], 'shaft')
                              for row in rest['pads'])


def check_tool(engine, swept):
    # The engine's exact bounding-box screen: True only when every capsule pair
    # is provably farther apart than pair_conflict's threshold.
    from .surgical_engine import _capsules_separated
    if _capsules_separated(swept, engine._bone_rest, pair_conflict):
        return
    if pair_conflict(swept, engine._bone_rest):
        raise ValueError('OPERATIVE_BONE_REST_TOOL_COLLISION')


def _gaps(engine, centers):
    config = engine.scene.properties['operativeExposureV1']['boneRest']
    for a, b, radius, _ in engine._bone_rest:
        combined = radius + config['cellRadiusMm']
        delta = centers - a
        ab = b - a
        fraction = np.clip((delta @ ab) / (ab @ ab), 0., 1.)
        # A positive gap is free, zero is supporting contact. The source cell
        # circumsphere is conservative for every rigid flap orientation.
        yield np.min(np.linalg.norm(delta - fraction[:, None] * ab, axis=1)) - combined


def check_body(engine, name, centers):
    if any(gap < -1e-6 for gap in _gaps(engine, centers)):
        raise ValueError('OPERATIVE_BONE_REST_BODY_COLLISION')


def supported(engine):
    from .surgical_exposure_geometry import centers
    config = engine.scene.properties['operativeExposureV1']['boneRest']
    return all(-1e-6 <= gap <= config['contactToleranceMm'] for gap in _gaps(engine, centers(engine, 'bone')))


def public_view(engine):
    config = engine.scene.properties['operativeExposureV1']['boneRest']
    return {'pads':[{'aMm':row['aMm'][:], 'bMm':row['bMm'][:], 'radiusMm':row['radiusMm']}
                    for row in config['pads']],
            'engaged':bool(supported(engine))}
