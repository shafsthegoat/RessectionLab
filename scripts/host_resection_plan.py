"""CUSA strokes for a host resection, planned against a prediction of what the aspirator clears.

All coordinates are in the operating-table frame (host_rehearsal_frame.py). The
aspirator works through one corticotomy over the tumour surface under the
thinnest cortex of the dural opening:

1. corticotomy: vertical strokes, layer by layer, through the cortex within
   the planned disc: the transcortical corridor;
2. debulking: 2 mm horizontal layers of the tumour from its top down, each tip
   position reached along a line through the corticotomy (a fulcrum grid on
   its surface) that the prediction says is already clear; the tip ball holds
   only tumour, cleared space or the non-target roof within ROOF_MM over the
   tumour, which a straight instrument must open to undercut a tumour that
   widens below its top;
3. margin: a 1 mm raster at 0.4 power over every position still touching
   tumour, never touching non-target tissue.

The prediction follows the engine's CUSA V3 contact law: the tip ball (radius
1.5 mm, whole-cell cover) clears every soft or tumour cell it covers before the
tip advances, and nothing it does not cover. The shaft (0.5 mm, 80 mm with its
shank) must lie in cleared space and the handpiece (10 mm) in free space, with
a margin for the engine's swept cover. Vessel, membrane, bone and scalp cells
are never in a tip ball. Each stroke enters and leaves on its own line from
above the grid. Nothing here reads engine state beyond the public view's field;
every stroke is still checked by the engine when it is executed.
"""
from __future__ import annotations

import math

import numpy as np
from scipy.spatial.transform import Rotation, Slerp

from resectionlab.geometry import GeometryScene, capsule_voxel_indices

TIP_MM = 1.5
SHAFT_MM = .5
SWEEP_MARGIN_MM = .45
FORBIDDEN_MARGIN_MM = .2
SHANK_MM = 80.
# Steepest line of approach through the corticotomy: cos of its angle from vertical.
MAX_TILT_COS = .35
# Weight of distance from the tumour axis against cortical depth in choosing the corticotomy.
CENTRALITY = .25
# Table height of each stroke's first and last pose: above the exterior plane (4 mm above the
# entry), the rolled scalp, the parked flap and every cell of the grid, so moves between strokes are free.
EXTERIOR_Z = 60.
# Height of the non-target roof over the tumour that the debulking may open, mm. Through one
# corticotomy a straight aspirator cannot undercut a tumour that widens below its top without
# opening the white matter over its shoulders; 7 mm is the smallest roof that reaches 85% here.
ROOF_MM = 7.
HANDPIECE = (80., 184., 10.)
# Margin on the handpiece: the engine bounds each rotating substep's sweep by
# 2*reach*sin(angle/4), about 0.77 mm at the handpiece end (184 mm).
HANDPIECE_SWEEP_MM = 1.
# Largest change of direction within one stroke; a larger turn ends the stroke.
MAX_HOP_TURN_RAD = math.radians(15.)


def tool_quaternion(direction):
    """Orientation whose local +z (into tissue) is ``direction``; downward tools keep DOWN's roll."""
    d = np.asarray(direction, dtype=float)
    d /= np.linalg.norm(d)
    down = Rotation.from_quat([1., 0., 0., 0.])
    current = down.apply([0., 0., 1.])
    axis = np.cross(current, d)
    if np.linalg.norm(axis) < 1e-9:
        return down.as_quat().tolist() if current @ d > 0 else [0., 0., 0., 1.]
    turn = Rotation.from_rotvec(axis / np.linalg.norm(axis) * math.atan2(np.linalg.norm(axis), current @ d))
    return (turn * down).as_quat().tolist()


class Carving:
    """Predicted occupancy while the aspirator works, from the current public field."""

    def __init__(self, scene, labels, obstacles=()):
        self.obstacles = list(obstacles)
        self.scene = scene
        self.shape = scene.shape
        self.grid = GeometryScene(np.zeros(scene.shape, dtype=bool), scene.affine)
        regions = {r['id']: r for r in scene.regions}
        bodies = scene.properties['operativeExposureV1']['bodies']
        cranial = {int(scene.labels.flat[c]) for body in bodies.values() for c in body['cells'][::7]}
        target = [i for i, r in regions.items() if r.get('target') and r['material'] == 'tumor']
        soft = [i for i, r in regions.items() if r['material'] in ('soft_tissue', 'tumor')]
        kind = scene.properties.get('hostRehearsalV1', {}).get('layers', [])
        extracranial = {r['id'] for r in scene.regions
                        if any(row['name'] == r['name'] and row['role'] in ('extracranial', 'skin', 'subcutaneous')
                               for row in kind)}
        flat = labels.ravel()
        self.labels = flat
        self.target = np.isin(flat, target)
        self.soft = np.isin(flat, soft) & ~np.isin(flat, list(cranial | extracranial))
        self.forbidden = (flat > 0) & ~self.soft
        self.cleared = flat == 0
        # Non-target tissue lying within ROOF_MM above tumour in its own column:
        # the roof the corridor may open to undercut a widening tumour.
        target_grid = self.target.reshape(scene.shape)
        below = np.zeros(scene.shape, dtype=bool)
        for step in range(1, int(round(ROOF_MM)) + 1):
            below[:, :, step:] |= target_grid[:, :, :-step]
        self.roof = (below.ravel() & self.soft & ~self.target)
        self.removed = {'target': 0, 'nontarget': 0}
        self.by_label = {}

    def cover(self, a, b, radius):
        cells = capsule_voxel_indices(self.grid, a, b, radius)
        if not len(cells):
            return np.empty(0, dtype=np.int64)
        return np.ravel_multi_index(cells.T, self.shape)

    def ball_ok(self, a, b, *, allow_nontarget):
        cells = self.cover(a, b, TIP_MM + FORBIDDEN_MARGIN_MM)
        if self.forbidden[cells].any():
            return False
        if allow_nontarget is True:
            return True
        tight = self.cover(a, b, TIP_MM)
        blocked = self.soft[tight] & ~self.target[tight] & ~self.cleared[tight]
        if allow_nontarget == 'roof':
            blocked &= ~self.roof[tight]
        return not blocked.any()

    def shaft_ok(self, tip, direction):
        tip = np.asarray(tip, dtype=float)
        d = np.asarray(direction, dtype=float) / np.linalg.norm(direction)
        near = set(self.cover(tip, tip, TIP_MM).tolist())
        cells = self.cover(tip, tip - SHANK_MM * d, SHAFT_MM + SWEEP_MARGIN_MM)
        blocked = [c for c in cells if not self.cleared[c] and c not in near]
        if blocked:
            return False
        start, end, radius = HANDPIECE
        handle = self.cover(tip - start * d, tip - end * d, radius + HANDPIECE_SWEEP_MM)
        if (~self.cleared[handle]).any():
            return False
        if self.obstacles:
            from resectionlab.surgical_motion import pair_conflict
            tool = [(tip, tip - SHANK_MM * d, SHAFT_MM + 1., 'shaft'),
                    (tip - start * d, tip - end * d, radius + 2., 'shaft')]
            if pair_conflict(tool, self.obstacles):
                return False
        return True

    def carve(self, a, b):
        cells = self.cover(a, b, TIP_MM)
        new = cells[self.soft[cells] & ~self.cleared[cells]]
        self.cleared[new] = True
        hit = self.target[new]
        self.removed['target'] += int(hit.sum())
        self.removed['nontarget'] += int((~hit).sum())
        for label in self.labels[new]:
            self.by_label[int(label)] = self.by_label.get(int(label), 0) + 1
        return len(new)


def pivot(scene, labels, *, opening):
    """Tumour-surface pivot under the thinnest cortex inside the dural opening ``opening(x, y)``."""
    target = np.isin(labels, [r['id'] for r in scene.regions if r.get('target')])
    columns = np.argwhere(target.any(axis=2))
    best = None
    tumour = scene.points(np.flatnonzero(target.ravel()))
    centroid = tumour.mean(axis=0)
    for i, j in columns:
        top = int(np.flatnonzero(target[i, j])[-1])
        occupied = np.flatnonzero(labels[i, j])
        surface = int(occupied[-1])
        point = scene.points([np.ravel_multi_index((i, j, top), scene.shape)])[0]
        if not opening(point[0], point[1]):
            continue
        score = (surface - top) + CENTRALITY * np.linalg.norm(point[:2] - centroid[:2])
        if best is None or score < best[0]:
            best = (score, point + [0., 0., .5])
    if best is None:
        raise ValueError('No tumour surface lies under the dural opening')
    return best[1]


def corticotomy_centre(scene, labels, *, opening):
    """Centre of the corticotomy: above the tumour surface under the thinnest cortex of the opening."""
    p = pivot(scene, labels, opening=opening)
    i, j, _ = np.rint(scene.inverse[:3, :3] @ [p[0], p[1], 0.] + scene.inverse[:3, 3]).astype(int)
    surface = scene.points([np.ravel_multi_index((i, j, int(np.flatnonzero(labels[i, j])[-1])), scene.shape)])[0]
    return p, surface + [0., 0., .5]


def swept_directions(d0, d1, count):
    """Tool axes the engine passes through between two poses (slerp of their orientations)."""
    if np.allclose(d0, d1):
        return [np.asarray(d1, dtype=float)] * count
    rotations = Rotation.from_quat([tool_quaternion(d0), tool_quaternion(d1)])
    return list(Slerp([0., 1.], rotations)(np.linspace(0., 1., count + 1)[1:]).apply([0., 0., 1.]))


def _serpentine(points, spacing):
    rows = {}
    for point in points:
        rows.setdefault(int(round(point[1] / spacing)), []).append(point)
    ordered = []
    for k, key in enumerate(sorted(rows)):
        row = sorted(rows[key], key=lambda q: q[0])
        ordered.extend(row if k % 2 == 0 else row[::-1])
    return ordered


def plan_layers(scene, labels, *, opening, corticotomy_mm=8., layer_mm=2., spacing_mm=2., outside=45.,
                margin_spacing_mm=1., margin_power=.4, fulcrum_spacing_mm=2., obstacles=()):
    """Corticotomy, then layer-by-layer debulking through it, then a low-power margin pass.

    Returns (centre, strokes, carving); each stroke is {'kind', 'poses': [(tip, direction)], 'power'}
    and starts and ends outside the head on its entry line.
    """
    carving = Carving(scene, labels, obstacles)
    p, surface = corticotomy_centre(scene, labels, opening=opening)
    down = np.array([0., 0., -1.])
    strokes = []
    target_cells = np.flatnonzero(carving.target)
    tumour = scene.points(target_cells)
    lo, hi = tumour.min(axis=0) - 1., tumour.max(axis=0) + 1.
    # 1. Corticotomy: vertical strokes through the cortex within the planned disc, layer by layer.
    disc = [np.array([surface[0] + x, surface[1] + y]) for x in np.arange(-corticotomy_mm, corticotomy_mm + .01, spacing_mm)
            for y in np.arange(-corticotomy_mm, corticotomy_mm + .01, spacing_mm)
            if math.hypot(x, y) <= corticotomy_mm]
    tops = []
    for xy in disc:
        i, j, _ = np.rint(scene.inverse[:3, :3] @ [xy[0], xy[1], 0.] + scene.inverse[:3, 3]).astype(int)
        soft = np.flatnonzero(carving.soft.reshape(scene.shape)[i, j])
        if len(soft):
            tops.append(scene.points([np.ravel_multi_index((i, j, int(soft[-1])), scene.shape)])[0][2])
    surface = np.array([surface[0], surface[1], max(tops) + .5])
    state = {'strokes': strokes, 'run': None}
    for z in np.arange(surface[2] - 1., p[2] - 1., -layer_mm):
        points = [np.array([*xy, z]) for xy in disc]
        _runs(carving, points, spacing_mm, None, surface[2], outside, power=1.,
              allow_nontarget=True, kind='corticotomy', state=state)
    fulcrums = [np.array([surface[0] + x, surface[1] + y])
                for x in np.arange(-corticotomy_mm, corticotomy_mm + .01, fulcrum_spacing_mm)
                for y in np.arange(-corticotomy_mm, corticotomy_mm + .01, fulcrum_spacing_mm)
                if math.hypot(x, y) <= corticotomy_mm - 1.5]
    # 2. Debulking: horizontal layers of the tumour from its top down, through the corticotomy.
    for z in np.arange(p[2] - 1., lo[2], -layer_mm):
        points = [np.array([x, y, z]) for x in np.arange(lo[0], hi[0] + .01, spacing_mm)
                  for y in np.arange(lo[1], hi[1] + .01, spacing_mm)]
        _runs(carving, points, spacing_mm, fulcrums, surface[2], outside, power=1.,
              allow_nontarget='roof' if ROOF_MM > 0 else False, kind='debulk', state=state)
    # 3. Margin: a finer, low-power pass over every position still touching tumour.
    for z in np.arange(p[2] - 1.5, lo[2], -layer_mm):
        points = [np.array([x, y, z]) for x in np.arange(lo[0], hi[0] + .01, margin_spacing_mm)
                  for y in np.arange(lo[1], hi[1] + .01, margin_spacing_mm)]
        _runs(carving, points, margin_spacing_mm, fulcrums, surface[2], outside,
              power=margin_power, allow_nontarget=False, kind='margin', state=state)
    _close(state, outside, carving)
    return p, strokes, carving


def _runs(carving, points, spacing, fulcrums, fulcrum_z, outside, *, power, allow_nontarget, kind,
          state=None, hop_mm=16.):
    """Extend the strokes in ``state`` over the feasible tip positions of one layer.

    The tip hops through cleared space when it can (up to ``hop_mm`` and a turn
    of MAX_HOP_TURN_RAD, with the shaft checked every millimetre along the
    orientations the engine interpolates), otherwise the current stroke ends
    outside and a new one enters along a line through the corticotomy, or
    vertically when ``fulcrums`` is None.
    """
    state = state if state is not None else {'strokes': [], 'run': None}
    down = np.array([0., 0., -1.])
    useful = []
    for x in points:
        cells = carving.cover(x, x, TIP_MM)
        if not (carving.soft[cells] & ~carving.cleared[cells]).any():
            continue
        if allow_nontarget is not True and not (carving.target[cells] & ~carving.cleared[cells]).any():
            continue
        if not carving.ball_ok(x, x, allow_nontarget=allow_nontarget):
            continue
        useful.append(x)

    def trial(a, b, checks):
        cells = carving.cover(a, b, TIP_MM)
        new = cells[carving.soft[cells] & ~carving.cleared[cells]]
        carving.cleared[new] = True
        ok = all(carving.shaft_ok(*check) for check in checks)
        carving.cleared[new] = False
        return ok

    for x in _serpentine(useful, spacing):
        run = state['run']
        if run is not None and (run['kind'] != kind or run['power'] != power):
            _close(state, outside, carving)
            run = None
        if fulcrums is None:
            lines = [down]
        else:
            lines = []
            for c in fulcrums:
                d = x - np.array([*c, fulcrum_z])
                d = d / np.linalg.norm(d) if np.linalg.norm(d) > 1e-6 else down
                if d[2] <= -MAX_TILT_COS:
                    lines.append((np.linalg.norm(c - x[:2]), d))
            lines = [d for _, d in sorted(lines, key=lambda row: row[0])]
        chosen = None
        # Continue the stroke with the smallest turn that clears, else enter afresh.
        if run is not None and np.linalg.norm(run['poses'][-1][0] - x) <= hop_mm:
            last, last_d = run['poses'][-1]
            turns = sorted((math.acos(min(1., float(d @ last_d))), k) for k, d in enumerate(lines))
            if carving.ball_ok(last, x, allow_nontarget=allow_nontarget):
                count = max(1, int(math.ceil(np.linalg.norm(x - last))))
                for turn, k in turns:
                    if turn > MAX_HOP_TURN_RAD:
                        break
                    d = lines[k]
                    axes = swept_directions(last_d, d, count)
                    checks = [((1 - t) * last + t * x, axis)
                              for t, axis in zip(np.linspace(0., 1., count + 1)[1:], axes)]
                    if trial(last, x, checks):
                        work = carving.carve(last, x)
                        chosen = ('hop', d)
                        break
        if chosen is None:
            for d in lines:
                # Enter from above the exterior plane on the line itself.
                entry = x - (EXTERIOR_Z - x[2]) / -d[2] * d
                if carving.ball_ok(entry, x, allow_nontarget=allow_nontarget) and trial(entry, x, [(x, d)]):
                    work = carving.carve(entry, x)
                    chosen = ('entry', d)
                    break
        if chosen is None:
            continue
        how, d = chosen
        if how == 'hop':
            run['poses'].append((x, d))
            run['work'].append(work)
        else:
            _close(state, outside, carving)
            state['run'] = {'kind': kind, 'power': power, 'poses': [(entry, d), (x, d)],
                            'work': [0, work]}
    return state


def _close(state, outside, carving):
    """End the current stroke. The executor leaves it by retracing its own poses back out along its entry line."""
    run = state['run']
    if run is not None:
        state['strokes'].append(run)
    state['run'] = None
