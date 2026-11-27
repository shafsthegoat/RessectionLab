"""Operating-table frame for a host rehearsal driven through the desktop's public actions.

The engine and the desktop work in patient RAS millimetres. A host rehearsal's
working grid has +k along the outward normal at the entry, so an operator's
motions are simplest in a table frame whose origin is the scalp entry, +z is
up, +y points to the flap hinge and +x completes a right hand. This adapter
converts every submitted pose to RAS and every returned pose to the table
frame. It changes no physics and writes no engine state.
"""
from __future__ import annotations

import copy
from types import SimpleNamespace

import numpy as np
from scipy.spatial.transform import Rotation

POINTS = {'position', 'positionMm', 'gripMm', 'hingeOriginMm', 'restCenterMm', 'parkCenterMm', 'pilotCenterMm',
          'cottonoidSiteMm', 'hemostatSiteMm', 'pointRasMm', 'aMm', 'bMm', 'sourceContactMm', 'fixedPositionMm'}
POINT_LISTS = {'centersMm', 'anchorsMm', 'endpointsMm'}
VECTORS = {'gripOffsetMm', 'hingeAxis', 'outward', 'normalRas', 'exteriorNormalRas', 'incisionDirectionRas',
           'normal', 'translationMm', 'startTranslationMm'}


def _triple(value):
    return (isinstance(value, (list, tuple)) and len(value) == 3
            and all(isinstance(v, (int, float)) and not isinstance(v, bool) for v in value))


class TableFrame:
    def __init__(self, scene):
        affine = np.asarray(scene.affine, dtype=float)
        spacing = np.linalg.norm(affine[:3, :3], axis=0)
        if not np.allclose(spacing, spacing[0]):
            raise ValueError('The working grid must be isotropic')
        self.spacing = float(spacing[0])
        self.rotation = affine[:3, :3] / self.spacing
        if not np.allclose(self.rotation.T @ self.rotation, np.eye(3), atol=1e-9):
            raise ValueError('The working grid axes must be orthonormal')
        entry = scene.properties['hostRehearsalV1']['entry']['entryRasMm']
        self.origin = np.asarray(entry, dtype=float)
        self._to_table = Rotation.from_matrix(self.rotation.T)
        self._to_ras = Rotation.from_matrix(self.rotation)

    def point(self, ras):
        return (self.rotation.T @ (np.asarray(ras, dtype=float) - self.origin)).tolist()

    def vector(self, ras):
        return (self.rotation.T @ np.asarray(ras, dtype=float)).tolist()

    def quaternion(self, ras):
        return (self._to_table * Rotation.from_quat(ras)).as_quat().tolist()

    def ras_point(self, table):
        return (self.origin + self.rotation @ np.asarray(table, dtype=float)).tolist()

    def ras_quaternion(self, table):
        return (self._to_ras * Rotation.from_quat(table)).as_quat().tolist()

    def ras_pose(self, pose):
        return {**pose, 'position': self.ras_point(pose['position']),
                'quaternion': self.ras_quaternion(pose['quaternion'])}

    def ras_action(self, action):
        result = dict(action)
        if 'pose' in result:
            result['pose'] = self.ras_pose(result['pose'])
        if 'path' in result:
            result['path'] = [self.ras_pose(pose) for pose in result['path']]
        return result

    def table(self, value, key=None):
        """Table-frame copy of a reply. Large numeric lists pass through untouched."""
        if isinstance(value, dict):
            pose = _triple(value.get('position')) and isinstance(value.get('quaternion'), (list, tuple)) \
                and len(value['quaternion']) == 4
            result = {}
            for name, item in value.items():
                if name == 'quaternion' and pose:
                    result[name] = self.quaternion(item)
                else:
                    result[name] = self.table(item, name)
            return result
        if key in POINTS and _triple(value):
            return self.point(value)
        if key in VECTORS and _triple(value):
            return self.vector(value)
        if key in POINT_LISTS and isinstance(value, list) and value and _triple(value[0]):
            return (np.asarray(value, dtype=float) - self.origin) @ self.rotation
        if isinstance(value, list):
            if value and isinstance(value[0], (dict, list)):
                return [self.table(item, key) for item in value]
            return value
        return value


class TableScene:
    """The working grid seen from the table: an axis-aligned grid with the entry at the origin."""

    def __init__(self, scene, frame):
        self.labels, self.shape, self.regions = scene.labels, scene.shape, scene.regions
        self.identity, self.spacing, self.voxel_mm3 = scene.identity, scene.spacing, scene.voxel_mm3
        inverse = np.asarray(scene.inverse, dtype=float)
        centre = inverse[:3, :3] @ frame.origin + inverse[:3, 3]
        self.affine = np.eye(4)
        self.affine[:3, :3] *= frame.spacing
        self.affine[:3, 3] = -centre * frame.spacing
        self.inverse = np.linalg.inv(self.affine)
        self.properties = frame.table(copy.deepcopy(scene.properties))
        for body in self.properties['operativeExposureV1']['bodies'].values():
            for name, item in list(body.items()):
                if isinstance(item, np.ndarray):
                    body[name] = item.tolist()

    def points(self, ids):
        indices = np.column_stack(np.unravel_index(np.asarray(ids, dtype=np.int64), self.shape))
        return indices @ self.affine[:3, :3].T + self.affine[:3, 3]


class TableEpisode:
    """A desktop practice episode whose poses are read and written in the table frame."""

    def __init__(self, live):
        self.live = live
        self.frame = TableFrame(live.scene)
        self.scene = TableScene(live.scene, self.frame)
        self._seen = None
        self._view = None

    def __getattr__(self, name):
        return getattr(self.live, name)

    @property
    def phase(self):
        return self.live.phase

    @phase.setter
    def phase(self, value):
        self.live.phase = value

    @property
    def view(self):
        current = self.live.view
        if self._seen is not current:
            self._seen, self._view = current, _lists(self.frame.table(current))
        return self._view

    @property
    def last(self):
        return {**self.live.last, 'view': self.view}

    def step(self, label, action, *, allowed_partial=()):
        outcome = self.live.step(label, self.frame.ras_action(action), allowed_partial=allowed_partial)
        return _lists(self.frame.table(outcome))

    def command(self, name, **arguments):
        if 'action' in arguments:
            arguments = {**arguments, 'action': self.frame.ras_action(arguments['action'])}
        return _lists(self.frame.table(self.live.command(name, **arguments)))

    def save(self, name):
        self.live.save(name)


def _lists(value):
    if isinstance(value, dict):
        return {name: _lists(item) for name, item in value.items()}
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, list) and value and isinstance(value[0], (dict, np.ndarray)):
        return [_lists(item) for item in value]
    return value


def table_landmarks(landmarks, *, reach_mm=70.):
    """The source landmarks the shared operative motions expect, in the table frame."""
    return SimpleNamespace(**{
        'entry': [0., 0., 0.], 'outward': [0., 0., 1.],
        'outside': {'left': [-reach_mm, -20., 26.], 'right': [reach_mm, -20., 26.]},
        'boneRadiusMm': landmarks['boneRadiusMm'],
    })
