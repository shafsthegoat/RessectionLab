"""Source-cell geometry for the generated, attached-sheet cranial teaching task."""

import math

import numpy as np
from scipy.spatial.transform import Rotation

PROPERTY = 'operativeExposureV1'
SCHEMA = 'resectionlab.operative-exposure.v1'
BODIES = ('scalp', 'bone', 'dura')
MAX_ANGLE = math.radians(115)


def exposure_declaration(
    shape,
    affine,
    *,
    scalp,
    bone,
    dura,
    bone_material,
    park_center,
    cottonoid_site,
    hemostat_site,
    hinge_axis=(1, 0, 0),
    outward=(0, 0, 1),
):
    """Derive true face-neighbor attachments and immutable gripping/closure landmarks."""
    shape = tuple(shape)
    affine = np.asarray(affine, float)
    axis = np.asarray(hinge_axis, float)
    normal = np.asarray(outward, float)
    axis /= np.linalg.norm(axis)
    normal -= np.dot(normal, axis) * axis
    normal /= np.linalg.norm(normal)
    forward = np.cross(axis, normal)

    def points(ids):
        return np.stack(np.unravel_index(ids, shape), axis=1) @ affine[:3, :3].T + affine[:3, 3]

    bodies = {}
    for name, source in zip(BODIES, (scalp, bone, dura)):
        ids = sorted(set(map(int, source['cells'])))
        if len(ids) < 4 or len(ids) > 20000:
            raise ValueError('EXPOSURE_BODY_SIZE')
        layer = set(map(int, bone_material if name == 'bone' else source['layerCells']))
        owned = set(ids)
        if not owned <= layer:
            raise ValueError('EXPOSURE_BODY_MEMBERSHIP')
        xyz = points(ids)
        mean = xyz.mean(0)
        longitudinal = (xyz - mean) @ forward
        origin = mean + longitudinal.min() * forward
        cross = (xyz - mean) @ axis
        axial = (xyz - mean) @ normal
        # Grasp the external distal rim, using a material point attached throughout.
        grip = ids[int(np.argmax(longitudinal - 0.65 * abs(cross) + 0.25 * axial))]
        edges = []
        anchors = []
        for i in ids:
            ijk = np.unravel_index(i, shape)
            for dim in range(3):
                for sign in [-1, 1]:
                    n = list(ijk)
                    n[dim] += sign
                    if not 0 <= n[dim] < shape[dim]:
                        continue
                    j = int(np.ravel_multi_index(n, shape))
                    if j in owned or j not in layer:
                        continue
                    bridge = points([j])[0] - points([i])[0]
                    if (
                        name != 'bone'
                        and np.dot(bridge, forward) < -0.5
                        and np.dot(points([i])[0] - origin, forward) < 1.1
                    ):
                        anchors.append([i, j])
                    else:
                        edges.append([i, j])
        if not edges or (name != 'bone' and not anchors):
            raise ValueError('EXPOSURE_HINGE_OR_SEAM_MISSING_' + name)
        sites = sorted({i for i, _ in edges})
        sp = points(sites)
        anchor_ids = []
        for sign in [-1, 1]:
            rank = (
                sign * ((sp - mean) @ axis) + 0.45 * ((sp - mean) @ forward) + 0.45 * ((sp - mean) @ normal)
            )
            anchor_ids.append(sites[int(np.argmax(rank))])
        body = {
            'cells': ids,
            'seamEdges': edges,
            'hingeEdges': anchors,
            'hingeOriginMm': origin.tolist(),
            'hingeAxis': axis.tolist(),
            'outward': normal.tolist(),
            'gripCell': grip,
            'anchorCells': anchor_ids,
            'bendLengthMm': min(5.0, float(np.ptp(longitudinal)) * 0.3),
            'restCenterMm': mean.tolist(),
            'maxAngleRad': MAX_ANGLE if name != 'bone' else 0.0,
        }
        if name == 'bone':
            kerf = sorted({j for _, j in edges})
            kp = points(kerf)
            pilot_id = kerf[int(np.argmax((kp - mean) @ forward - abs((kp - mean) @ axis)))]
            pilot = points([pilot_id])[0] + 1.25 * forward
            bone_ids = sorted(layer - owned)
            bp = points(bone_ids)
            lateral = bp - pilot - np.outer((bp - pilot) @ normal, normal)
            port_ids = np.asarray(bone_ids)[
                (np.linalg.norm(lateral, axis=1) <= 2.25) & (abs((bp - pilot) @ normal) <= 5)
            ]
            body.update(
                kerfCells=kerf,
                portCells=sorted(map(int, port_ids)),
                pilotCenterMm=pilot.tolist(),
                parkCenterMm=np.asarray(park_center, float).tolist(),
            )
        bodies[name] = body
    return {
        'schema': SCHEMA,
        'bodies': bodies,
        'cottonoidSiteMm': list(map(float, cottonoid_site)),
        'hemostatSiteMm': list(map(float, hemostat_site)),
        'siteRadiusMm': 2.0,
        'materials': {
            'model': 'isometric-strip-bend-v1',
            'scalp': {
                'bendingStiffnessNmmPerRad': 1.8,
                'torsionalDampingNmmsPerRad': 0.15,
                'source': 'https://pubmed.ncbi.nlm.nih.gov/29793156/',
            },
            'dura': {
                'bendingStiffnessNmmPerRad': 0.45,
                'torsionalDampingNmmsPerRad': 0.04,
                'source': 'https://pubmed.ncbi.nlm.nih.gov/37598794/',
            },
            'scope': 'Assigned reduced hinge and membrane parameters for a generated model. Human tensile data motivate tissue-specific behavior. Fold stiffness and damping are uncalibrated assumptions.',
        },
    }


def rest(engine, name):
    config = engine.scene.properties[PROPERTY]['bodies'][name]
    cache = getattr(engine.scene, '_operative_strip_geometry', None)
    if cache is None:
        cache = engine.scene._operative_strip_geometry = {}
    if name not in cache:
        ids = np.asarray(config['cells'], dtype=np.int64)
        pos = engine.scene.points(ids)
        axis = np.asarray(config['hingeAxis'])
        normal = np.asarray(config['outward'])
        forward = np.cross(axis, normal)
        origin = np.asarray(config['hingeOriginMm'])
        local = pos - origin
        along = local @ forward
        across = local @ axis
        heights = local @ normal
        if name == 'bone':
            exterior = np.zeros(len(ids))
        else:
            # Measured outer face of each column in the curved source sheet.
            # A single horizontal neutral surface incorrectly rolls head
            # curvature into the layer thickness and drives galea into skull.
            columns = np.column_stack((np.round(across, 7), np.round(along, 7)))
            _, inverse = np.unique(columns, axis=0, return_inverse=True)
            levels = np.full(int(inverse.max()) + 1, -np.inf)
            np.maximum.at(levels, inverse, heights)
            exterior = levels[inverse]
        lookup = {int(i): k for k, i in enumerate(ids)}
        hinge_band = max((along[lookup[c]] for c, _ in config['hingeEdges']), default=0.0)
        cache[name] = (
            ids,
            pos,
            lookup,
            axis,
            normal,
            forward,
            origin,
            across,
            along,
            heights - exterior,
            exterior,
            hinge_band,
        )
    return cache[name]


def centers(engine, name, state=None, indices=None):
    cfg = engine.scene.properties[PROPERTY]['bodies'][name]
    body = state if state is not None else engine.state[PROPERTY]['bodies'][name]
    ids, points, lookup, axis, n, f, origin, x, s, z, exterior, hinge_band = rest(engine, name)
    if indices is not None:
        points = points[indices]
        x, s, z, exterior = x[indices], s[indices], z[indices], exterior[indices]
    if name == 'bone':
        center = np.asarray(cfg['restCenterMm'])
        return Rotation.from_quat(body['quaternion']).apply(points - center) + center + body['translationMm']
    a = body['angleRad']
    if abs(a) < 1e-12:
        return points
    length = cfg['bendLengthMm']
    free = s - hinge_band
    u = np.clip(free, 0, length)
    tail = free - u
    theta = a * u / length
    sn = np.sin(theta)
    cs = np.cos(theta)
    # Integrate the unit tangent through a finite bend, continuing straight
    # beyond it. The neutral strip preserves arclength and hinge attachment.
    # Through-thickness layers may shear in the short transition to the
    # attached crease; the external neutral strip still preserves arclength.
    # The smooth ramp keeps the anchored layered edge clear of the curved skull.
    fraction = u / length
    layer_theta = theta * fraction**2 * (3.0 - 2.0 * fraction)
    radial = hinge_band + length / a * sn + tail * cs - z * np.sin(layer_theta)
    height = length / a * (1 - cs) + tail * sn + z * np.cos(layer_theta) + exterior
    return origin + np.outer(x, axis) + np.outer(radial, f) + np.outer(height, n)


def transform_key(body):
    return (body['angleRad'], tuple(body['translationMm']), tuple(body['quaternion']))


def source_position(engine, name, cell, state=None):
    return centers(engine, name, state, [rest(engine, name)[2][cell]])[0]


def site_trajectory(scene, name, cell, angle_rad):
    """Exact geometric handle landmark for authoring public RAS pose paths."""
    from types import SimpleNamespace

    engine = SimpleNamespace(scene=scene)
    return source_position(engine, name, cell, {'angleRad': float(angle_rad)}).tolist()


def grip_trajectory(scene, name, angle_rad):
    return site_trajectory(scene, name, scene.properties[PROPERTY]['bodies'][name]['gripCell'], angle_rad)


def solve_angle(engine, name, requested):
    from scipy.optimize import minimize_scalar

    cfg = engine.scene.properties[PROPERTY]['bodies'][name]

    def error(a):
        diff = source_position(engine, name, cfg['gripCell'], {'angleRad': a}) - requested
        return float(diff @ diff)

    end = cfg['maxAngleRad']
    fit = minimize_scalar(error, bounds=(0, end), method='bounded', options={'xatol': 1e-7})
    residual, angle = min((error(0), 0.0), (error(end), end), (fit.fun, fit.x))
    if residual > 1.0**2:
        raise ValueError('EXPOSURE_ATTACHED_HINGE_LIMIT')
    return float(angle)


def geometry_key(engine):
    return tuple(transform_key(engine.state[PROPERTY]['bodies'][name]) for name in BODIES)


def moved(engine):
    config = engine.scene.properties[PROPERTY]['bodies']
    out = []
    removed = set(engine.state['removed'])
    for name in BODIES:
        row = engine.state[PROPERTY]['bodies'][name]
        if (
            row['angleRad'] == 0
            and row['translationMm'] == [0.0, 0.0, 0.0]
            and row['quaternion'] == [0.0, 0.0, 0.0, 1.0]
        ):
            continue
        loc = centers(engine, name)
        ijk = np.rint(loc @ engine.scene.inverse[:3, :3].T + engine.scene.inverse[:3, 3]).astype(int)
        to = np.ravel_multi_index(ijk.T, engine.scene.shape)
        out.extend((i, int(j)) for i, j in zip(config[name]['cells'], to) if i != j and i not in removed)
    return out
