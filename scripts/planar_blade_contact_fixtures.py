"""Frozen generated planar-contact coupons; no acquired or patient data."""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / 'manifests/experiments/planar-blade-contact-v2.json'
MANIFEST_SHA256 = 'b1d8e7c18ddbc1288cc50585b2f280e4c57c3eed20caf524ccfe2b61e0525b15'
PREDECESSOR_SHA256 = '5e0536b0c58a1eca38e1fb1096e99f0c8d4509e9580d4c3259b0648d98195dd4'


def frozen_protocol():
    raw = MANIFEST.read_bytes()
    if hashlib.sha256(raw).hexdigest() != MANIFEST_SHA256:
        raise ValueError('FROZEN_PLANAR_PROTOCOL_HASH_MISMATCH')
    return json.loads(raw)


def rigid_rotation():
    angles = np.deg2rad(frozen_protocol()['fixtures']['rotationDegreesXYZ'])
    x, y, z = angles
    rx = np.array([[1., 0., 0.], [0., np.cos(x), -np.sin(x)], [0., np.sin(x), np.cos(x)]])
    ry = np.array([[np.cos(y), 0., np.sin(y)], [0., 1., 0.], [-np.sin(y), 0., np.cos(y)]])
    rz = np.array([[np.cos(z), -np.sin(z), 0.], [np.sin(z), np.cos(z), 0.], [0., 0., 1.]])
    return rz @ ry @ rx


@dataclass
class GeneratedFixture:
    scene: object
    cells: tuple
    contact_faces: tuple
    support_faces: tuple
    blade: object
    rotation: np.ndarray
    translation_mm: np.ndarray
    name: str
    n: int
    partial: bool
    maxwell: bool


def generated_fixture(n=2, *, partial=False, maxwell=False, rotated=False):
    from resectionlab.surgical_scene import SurgicalScene
    from resectionlab.surgical_physics_v2 import default_profile
    from resectionlab import surgical_continuum as continuum
    from resectionlab.surgical_planar_contact import PlanarBlade
    f = frozen_protocol()['fixtures']
    if n not in f['meshCellsPerAxis']:
        raise ValueError('fixture mesh is outside frozen protocol')
    spacing = f['blockMm'][0] / n
    labels = np.zeros((n + 2,) * 3, np.uint16)
    labels[1:-1, 1:-1, 1:-1] = 1
    cells = tuple(int(c) for c in np.flatnonzero(labels.ravel()))
    rotation = rigid_rotation() if rotated else np.eye(3)
    translation = np.array(f['rotationTranslationMm'] if rotated else [0., 0., 0.])
    affine = np.eye(4)
    affine[:3, :3] = rotation * spacing
    affine[:3, 3] = rotation @ np.full(3, -.5 * spacing) + translation
    material = {'shearPa': f['equilibriumShearPa'],
                'poissonRatio': f['partialPoissonRatio'] if partial else f['uniformPoissonRatio'],
                'relaxationShearPa': [f['maxwellBranchShearPa']] if maxwell else [],
                'source': 'Frozen generated planar-blade numerical coupon; not patient measurement'}
    scene = SurgicalScene(labels, affine,
        [{'id': 1, 'name': 'Generated planar contact coupon', 'material': 'soft_tissue'}],
        {'kind': 'generated-numerical-verification-only'},
        {'tissuePhysicsV2': default_profile(), continuum.PROPERTY: continuum.research_profile(
            {'1': material}, relaxation_seconds=[f['maxwellRelaxationSeconds']] if maxwell else [])})
    top = tuple((c, 2, 1) for c in cells if np.unravel_index(c, labels.shape)[2] == n)
    bottom = tuple((c, 2, -1) for c in cells if np.unravel_index(c, labels.shape)[2] == 1)
    blade = PlanarBlade.create(origin_mm=rotation @ np.array(f['footprintCenterMm']) + translation,
        axis_u=rotation @ [1., 0., 0.], axis_v=rotation @ [0., 1., 0.],
        inward_normal=rotation @ [0., 0., -1.],
        size_mm=f['partialFootprintMm'] if partial else f['uniformFootprintMm'])
    name = f"{'partial' if partial else 'uniform'}-{'maxwell' if maxwell else 'elastic'}-n{n}" + ('-rotated' if rotated else '')
    return GeneratedFixture(scene, cells, top, bottom, blade, rotation, translation, name, n, partial, maxwell)


def prescribed_schedule(dt_s, *, partial=False):
    """End-of-step normal offsets; exactly ramp, hold, withdrawal, no duplicates."""
    f = frozen_protocol()['fixtures']
    if dt_s not in f['timeStepsSeconds']:
        raise ValueError('fixture timestep is outside frozen protocol')
    s = f['schedule']
    peak = f['partialIndentationMm'] if partial else f['uniformIndentationMm']
    rows = []
    elapsed = 0.
    for phase, duration in [('ramp', s['rampSeconds']), ('hold', s['holdSeconds']), ('withdraw', s['withdrawSeconds'])]:
        count = round(duration / dt_s)
        for step in range(1, count + 1):
            elapsed += dt_s
            fraction = step / count
            offset = peak * fraction if phase == 'ramp' else peak if phase == 'hold' else peak + fraction * (s['finalOffsetMm'] - peak)
            rows.append({'phase': phase, 'timeS': round(elapsed, 12), 'dtS': dt_s, 'offsetMm': offset})
    return rows
