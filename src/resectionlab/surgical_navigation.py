"""Opt-in neuronavigation scenario: a registered pointer that cannot see brain shift.

A scene that declares ``navigationV1`` adds a ``navigation_pointer`` instrument.
Inspecting with it records what a navigation screen would show: the pointer
position mapped into the preoperative image through a rigid registration that
carries a fixed error, the image label at that position, and the distance to
the planned (preoperative) tumour. Navigation knows nothing about brain shift,
so after dural opening it reports the preoperative coordinate under the tip
while the tissue now under the tip came from somewhere else.

The registration error is a rigid transform drawn once from a declared seed
and scale with a stable hash-based normal generator, so every replay draws
the same error on every platform. It is latent: ``observe()`` holds only the
screen readout. The truth for scoring (tip, registration error, shifted tissue,
tumour-edge discrepancy) is kept under ``state['navigationV1']`` and exposed in
``view()['navigationV1']['privilegedTruth']``, which is audit data.

Every constant is assumed research unless a source is named.
"""
from __future__ import annotations

import copy
import hashlib
import math

import numpy as np
from scipy.spatial.transform import Rotation

PROPERTY = 'navigationV1'
SCHEMA = 'resectionlab.navigation-scenario.v1'
STATE_SCHEMA = 'resectionlab.navigation-readings.v1'
TARGET_CENTROID = 'target-centroid'
SOURCES = [
    {'id': 'fitzpatrick-1998', 'citation': 'Fitzpatrick JM, West JB, Maurer CR Jr. Predicting error in rigid-body '
     'point-based registration. IEEE Trans Med Imaging. 1998.',
     'scope': 'Target registration error of a rigid point-based registration; the error model form only.'},
]
LIMITS = [
    'One rigid registration error per scene; no tracking jitter, camera line-of-sight, reference-frame slip or re-registration.',
    'Navigation shows the preoperative image only: no intraoperative imaging update or shift compensation.',
    'Scenario values for teaching, not the accuracy of any navigation product.',
]


def default_property(*, seed=0, rms_mm=2.0, rotation_sd_deg=0.5):
    return {'schema': SCHEMA,
            'registration': {
                'seed': seed,
                # assumed-research: order of a clinical fiducial or surface
                # registration target error; the model form follows SOURCES.
                'targetRegistrationErrorRmsMm': rms_mm,
                # assumed-research: rotational component about the centre.
                'rotationSdDeg': rotation_sd_deg,
                'centerRasMm': TARGET_CENTROID},
            'provenance': {'evidenceClass': 'assumed-research', 'calibration': 'none',
                           'statement': 'Registration error is a declared scenario draw, not a measured system accuracy.',
                           'sources': copy.deepcopy(SOURCES)},
            'limits': list(LIMITS)}


def _number(value, name, low, high):
    if type(value) not in (int, float) or not math.isfinite(value) or not low <= value <= high:
        raise ValueError('NAVIGATION_INVALID: ' + name)
    return float(value)


def declaration(scene):
    if PROPERTY not in scene.properties:
        return None
    declared = scene.properties[PROPERTY]
    if (not isinstance(declared, dict) or set(declared) != {'schema', 'registration', 'provenance', 'limits'}
            or declared['schema'] != SCHEMA):
        raise ValueError('NAVIGATION_INVALID: complete ' + SCHEMA + ' declaration required')
    registration = declared['registration']
    if not isinstance(registration, dict) or set(registration) != {'seed', 'targetRegistrationErrorRmsMm',
                                                                   'rotationSdDeg', 'centerRasMm'}:
        raise ValueError('NAVIGATION_INVALID: registration')
    if type(registration['seed']) is not int or not 0 <= registration['seed'] < 2 ** 63:
        raise ValueError('NAVIGATION_INVALID: seed')
    center = registration['centerRasMm']
    if center != TARGET_CENTROID:
        if not isinstance(center, (list, tuple)) or len(center) != 3:
            raise ValueError('NAVIGATION_INVALID: centerRasMm')
        center = [_number(v, 'centerRasMm', -1e4, 1e4) for v in center]
    provenance = declared['provenance']
    if (not isinstance(provenance, dict) or provenance.get('evidenceClass') != 'assumed-research'
            or not isinstance(provenance.get('statement'), str) or not provenance['statement'].strip()):
        raise ValueError('NAVIGATION_INVALID: assumed-research provenance required')
    limits = declared['limits']
    if type(limits) is not list or not limits or any(not isinstance(item, str) or not item.strip() for item in limits):
        raise ValueError('NAVIGATION_INVALID: limits')
    return {'schema': SCHEMA, 'registration': {
                'seed': registration['seed'],
                'targetRegistrationErrorRmsMm': _number(registration['targetRegistrationErrorRmsMm'], 'rms', 0., 20.),
                'rotationSdDeg': _number(registration['rotationSdDeg'], 'rotationSdDeg', 0., 10.),
                'centerRasMm': center},
            'provenance': copy.deepcopy(provenance), 'limits': list(limits)}


def _normals(seed, count):
    """Platform-stable standard normals from SHA-256 (Box-Muller)."""
    values = []
    for index in range(count):
        digest = hashlib.sha256(f'{SCHEMA}:{seed}:{index}'.encode()).digest()
        u1 = (int.from_bytes(digest[:8], 'big') + .5) / 2 ** 64
        u2 = (int.from_bytes(digest[8:16], 'big') + .5) / 2 ** 64
        values.append(math.sqrt(-2. * math.log(u1)) * math.cos(2. * math.pi * u2))
    return np.asarray(values)


def _target_cells(scene):
    targets = [r['id'] for r in scene.regions if r.get('target')]
    return np.flatnonzero(np.isin(scene.labels, targets).ravel()) if targets else np.zeros(0, dtype=np.int64)


def _scene_cache(scene):
    cache = getattr(scene, '_navigation_cache', None)
    if cache is None:
        from scipy.spatial import cKDTree
        cells = _target_cells(scene)
        points = scene.points(cells) if len(cells) else np.zeros((0, 3))
        cache = scene._navigation_cache = {'targetCells': cells, 'targetPoints': points,
                                           'tree': cKDTree(points) if len(cells) else None}
    return cache


class Registration:
    """Navigation's belief: image = R (x - c) + c + t, with the drawn error (R, t)."""

    def __init__(self, scene, decl):
        reg = decl['registration']
        draw = _normals(reg['seed'], 6)
        self.translation = draw[:3] * reg['targetRegistrationErrorRmsMm'] / math.sqrt(3.)
        self.rotation = Rotation.from_rotvec(draw[3:] * math.radians(reg['rotationSdDeg']))
        if reg['centerRasMm'] == TARGET_CENTROID:
            points = _scene_cache(scene)['targetPoints']
            corners = np.array(scene.shape, float) - 1
            self.center = points.mean(0) if len(points) else scene.points(
                [int(np.ravel_multi_index(tuple((corners / 2).astype(int)), scene.shape))])[0]
        else:
            self.center = np.asarray(reg['centerRasMm'], float)

    def to_image(self, x):
        return self.rotation.apply(np.asarray(x, float) - self.center) + self.center + self.translation

    def to_room(self, y):
        return self.rotation.inv().apply(np.asarray(y, float) - self.center - self.translation) + self.center


def registration(engine):
    return Registration(engine.scene, engine._navigation)


def _region(scene, point):
    ijk = np.rint(scene.inverse[:3, :3] @ point + scene.inverse[:3, 3]).astype(int)
    if np.any(ijk < 0) or np.any(ijk >= scene.shape):
        return None
    label = int(scene.labels[tuple(ijk)])
    if not label:
        return {'regionId': 0, 'name': 'unlabelled', 'material': None}
    region = next(r for r in scene.regions if r['id'] == label)
    return {'regionId': label, 'name': region['name'], 'material': region['material']}


def _shift_mm(engine, cells):
    if engine.__dict__.get('_brain_shift') is None:
        return np.zeros((len(cells), 3))
    from .surgical_brain_shift import active_map
    shift = active_map(engine)
    return np.zeros((len(cells), 3)) if shift is None else shift.displacement_mm(cells)


def read_pointer(engine, key, p, q):
    """Record one navigation reading; the screen to observations, the truth to state."""
    scene = engine.scene
    tip = np.asarray(p, float)
    nav = registration(engine)
    reported = nav.to_image(tip)
    cache = _scene_cache(scene)
    radius = engine.parameters(engine.state['slots'][key])['radiusMm']
    # Tissue now under the tip: the nearest source within the pointer's tip reach.
    touching = engine.touched(tip, tip, radius + .5 * float(scene.spacing.max()))
    tissue = None
    if touching:
        cells = np.asarray(touching, dtype=np.int64)
        current = scene.points(cells) + _shift_mm(engine, cells)
        nearest = int(np.argmin(np.linalg.norm(current - tip, axis=1)))
        cell = int(cells[nearest])
        region = engine.by_label[int(scene.labels.flat[cell])]
        tissue = {'sourceCell': cell, 'regionId': region['id'], 'name': region['name'], 'material': region['material'],
                  'preoperativeRasMm': scene.points([cell])[0].tolist(),
                  'shiftMm': _shift_mm(engine, [cell])[0].tolist()}
    screen = {'kind': 'navigation', 'probe': 'navigation_pointer', 'reportedImageRasMm': reported.tolist(),
              'imageRegion': _region(scene, reported),
              'registration': {'declaredTargetErrorRmsMm': engine._navigation['registration']['targetRegistrationErrorRmsMm'],
                               'evidenceClass': 'assumed-research'},
              'shiftCompensation': 'none: preoperative image only'}
    truth = {'tipRasMm': tip.tolist(), 'registrationErrorMm': (reported - tip).tolist(), 'tissueUnderTip': tissue}
    true_preop = np.asarray(tissue['preoperativeRasMm']) if tissue else tip - _shift_mm_at(engine, tip)
    truth['truePreoperativeRasMm'] = true_preop.tolist()
    truth['navigationErrorMm'] = float(np.linalg.norm(reported - true_preop))
    if cache['tree'] is not None:
        distance, index = cache['tree'].query(reported)
        edge_cell = int(cache['targetCells'][index])
        edge_image = cache['targetPoints'][index]
        screen['distanceToPlannedTumourMm'] = float(distance)
        screen['nearestPlannedTumourImageRasMm'] = edge_image.tolist()
        believed = nav.to_room(edge_image)
        actual = edge_image + _shift_mm(engine, [edge_cell])[0]
        removed = set(engine.state['removed'])
        alive = np.asarray([c not in removed for c in cache['targetCells'].tolist()])
        current = cache['targetPoints'][alive] + _shift_mm(engine, cache['targetCells'][alive])
        truth['nearestTumourEdge'] = {'sourceCell': edge_cell, 'navigationRoomRasMm': believed.tolist(),
                                      'actualRoomRasMm': actual.tolist(),
                                      'discrepancyMm': float(np.linalg.norm(actual - believed))}
        truth['trueDistanceToRemainingTumourMm'] = (float(np.min(np.linalg.norm(current - tip, axis=1)))
                                                     if len(current) else None)
    observations = engine.state['observations']
    record = engine.state.setdefault(PROPERTY, {'schema': STATE_SCHEMA, 'readings': []})
    if not observations or observations[-1] != screen or not record['readings'] or record['readings'][-1] != truth:
        observations.append(screen)
        record['readings'].append(truth)
    return [screen]


def _shift_mm_at(engine, point):
    """Scenario displacement of the tissue around a free-space point (zero without a shift)."""
    if engine.__dict__.get('_brain_shift') is None:
        return np.zeros(3)
    from .surgical_brain_shift import active_map
    shift = active_map(engine)
    if shift is None:
        return np.zeros(3)
    ijk = np.rint(engine.scene.inverse[:3, :3] @ point + engine.scene.inverse[:3, 3]).astype(np.int64)
    amplitude = shift.geometry.evaluate(ijk[None, :])[0]
    return amplitude * shift.scale * shift.direction


def metrics(engine):
    record = engine.state.get(PROPERTY)
    return {'navigationReadings': 0 if record is None else len(record['readings'])}


def view(engine):
    from .surgical_tools import NAVIGATION_POINTER
    record = engine.state.get(PROPERTY)
    return {'schema': SCHEMA, 'evidenceClass': 'assumed-research scenario',
            'instrument': dict(NAVIGATION_POINTER, id='navigation_pointer', workingLengthMm=40.,
                               approximation='generic research geometry; opt-in navigationV1 only'),
            'declaration': copy.deepcopy(engine._navigation),
            'privilegedTruth': {'note': 'Scoring and audit only; never a policy input.',
                                'readings': copy.deepcopy(record['readings']) if record else []}}
