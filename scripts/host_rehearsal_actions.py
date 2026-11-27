"""Operative motions on a host head, in the operating-table frame.

The table frame (see host_rehearsal_frame.py) is the working grid with the
scalp entry at the origin, +z outward and +y toward the flap hinge, so a
column of cells is a vertical line. Every motion is a public action checked by
the engine. Landmarks are read from the source declaration and the current
public view; no tissue label decides what an instrument removes or cuts.
"""
from __future__ import annotations

import math

import numpy as np

from glioma_walkthrough_actions import DOWN, ScalpClearance, body, edge_quaternion, pose


def column_top(scene, x, y, labels=None):
    """Table height of the outer face of the highest material cell in the column at (x, y).

    ``labels`` is the current material field (Field.labels); the source labels by default.
    """
    labels = scene.labels if labels is None else labels
    i, j, _ = np.rint(scene.inverse[:3, :3] @ [x, y, 0.] + scene.inverse[:3, 3]).astype(int)
    occupied = np.flatnonzero(labels[i, j])
    return float(scene.points([np.ravel_multi_index((i, j, occupied[-1]), scene.shape)])[0][2] + .5)


class Field:
    """The current material field from the public view: source labels less removed cells, moved bodies moved."""

    def __init__(self, live):
        scene = live.scene
        view = live.view
        labels = scene.labels.ravel().copy()
        moved = np.asarray(view['moved'], dtype=np.int64).reshape(-1, 2)
        source = labels[moved[:, 0]].copy()
        labels[np.asarray(view['removed'], dtype=np.int64)] = 0
        labels[moved[:, 0]] = 0
        labels[moved[:, 1]] = source
        self.labels = labels.reshape(scene.shape)
        self.scene = scene
        self.material = {r['id']: r['material'] for r in scene.regions}

    def mask(self, *materials, exclude=False):
        ids = [i for i, m in self.material.items() if (m in materials) != exclude]
        return np.isin(self.labels, ids)


class Clearance(ScalpClearance):
    """Working-part clearance of the mounted instrument against a chosen material mask."""

    def __init__(self, hand, blocked):
        from resectionlab.geometry import GeometryScene
        scene = hand.live.scene
        self.grid = GeometryScene(np.zeros(scene.shape, dtype=bool), scene.affine)
        self.bone = blocked
        self.edges = [c for c in hand.slot['contactGeometry'] if c['part'] == 'working']


def enter_scalp(hand, course, instrument='scalpel'):
    """Stage the blade over the first seam column, clear of the skin surface."""
    angle = edge_quaternion(*course[:2])
    p = course[0]
    hand.prepare(instrument, p, orientation=angle)
    above = column_top(hand.live.scene, p[0], p[1]) + 2.
    hand.action('Approach the scalp over the planned incision', 'move', .6, at=[p[0], p[1], above],
                orientation=angle)
    return angle


def chained(course):
    """The seam points as one walk: from the first point, always to the nearest unvisited point."""
    points = [list(p) for p in course]
    remaining = points[1:]
    walk = [points[0]]
    while remaining:
        here = np.asarray(walk[-1][:2])
        index = int(np.argmin([math.dist(here, p[:2]) for p in remaining]))
        walk.append(remaining.pop(index))
    return walk


def shallow(clearance, point, q, *, limit=4.):
    """Lowest bone-clear blade depth over a seam point, or None within ``limit`` mm."""
    for dz in np.arange(0., limit + 1e-9, .1):
        proposal = np.asarray(point) + [0., 0., dz]
        if clearance.free(proposal, q):
            return proposal
    return None


def contour(hand, clearance, course, first_angle, labels=None, label='Incise scalp around the calvarial contour'):
    """Draw the blade along the seam at the deepest bone-clear depth of each point.

    As ScalpClearance.contour, but a point with no clearance within 4 mm is
    skipped and left to the column strokes, and a corner the blade cannot
    cross at depth is crossed above the skin rather than 1.5 mm higher.
    """
    scene = hand.live.scene
    previous = np.asarray(hand.slot['position'])
    old = first_angle
    path = []
    for a, b in zip(course, course[1:]):
        q = edge_quaternion(a, b) if math.dist(a[:2], b[:2]) <= 1.6 else old
        end = shallow(clearance, b, q)
        if end is None:
            continue
        if math.dist(previous[:2], end[:2]) > 1.6 or any(
                not clearance.free((1 - t) * previous + t * end, q) for t in np.linspace(0., 1., 9)):
            height = max(previous[2], end[2]) + 1.5
            if any(not clearance.free([*((1 - t) * previous[:2] + t * end[:2]), height], q)
                   for t in np.linspace(0., 1., 9)):
                height = max(column_top(scene, *previous[:2], labels), column_top(scene, *end[:2], labels)) + 1.
            path.extend(vertical(clearance, previous, [*previous[:2], height], old))
            path.append(pose([*end[:2], height], q))
            path.extend(vertical(clearance, [*end[:2], height], end, q))
        else:
            path.append(pose(end, q))
        previous, old = end, q
    previous = hand.slot['position']
    for i in range(0, len(path), 8):
        points = path[i:i + 8]
        positions = [previous] + [p['position'] for p in points]
        distance = sum(math.dist(x, y) for x, y in zip(positions, positions[1:]))
        hand.action(label, 'apply', max(.2, distance / 6.), path=points)
        previous = positions[-1]


def vertical(clearance, start, end, q):
    """Poses from ``start`` to ``end`` along a vertical line, each step no longer than the bone clearance allows.

    A substep's swept cover is inflated by half its length, so a step of
    length s needs a margin of s/2 at both of its ends.
    """
    start, end = np.asarray(start, dtype=float), np.asarray(end, dtype=float)
    poses, here = [], start.copy()
    direction = np.sign(end[2] - start[2])
    while abs(end[2] - here[2]) > 1e-9:
        for step in (.33, .2, .1, .04, .02):
            step = min(step, abs(end[2] - here[2]))
            there = here + [0., 0., direction * step]
            middle = (here + there) / 2
            if clearance.free(there, q, margin=step / 2 + .01) and clearance.free(middle, q, margin=step / 2 + .01):
                break
        else:
            raise ValueError('No bone-clear vertical blade passage')
        poses.append(pose(there, q))
        here = there
    return poses


def remaining_columns(hand, clearance, name='scalp', heights=(.33, .38, .45, .2, .55), labels=None,
                      tool='scalpel'):
    """Divide each seam column the contour left attached with one vertical stroke.

    The blade rises out of the previous column, crosses above the skin, turns
    its cutting plane across the remaining attachment and descends to the
    deepest bone-clear depth in fine steps, so its swept cover never reaches
    the calvarium.
    """
    live = hand.live
    scene = live.scene
    seam = scene.properties['operativeExposureV1']['bodies'][name]['seamEdges']
    remaining = [edge for edge in seam if edge not in body(live, name)['seamEdges']]
    columns = {}
    for edge in remaining:
        a, b = scene.points(edge)
        key = (round(float((a[0] + b[0]) / 2), 3), round(float((a[1] + b[1]) / 2), 3))
        row = columns.setdefault(key, {'deepest': np.inf, 'normal': b - a})
        row['deepest'] = min(row['deepest'], float((a[2] + b[2]) / 2))
    centre = np.asarray(scene.properties['operativeExposureV1']['bodies'][name]['restCenterMm'])
    order = sorted(columns, key=lambda key: math.atan2(key[1] - centre[1], key[0] - centre[0]))
    inserted = False
    for key in order:
        row = columns[key]
        # Either face of the blade divides the attachment; the belly may face
        # whichever side has the lower bone.
        choices = [(np.array([*key, row['deepest'] + h]), edge_quaternion([0., 0.], [sign * -row['normal'][1],
                                                                              sign * row['normal'][0]]))
                   for h in heights for sign in (1., -1.)]
        at, q = next(((point, q) for point, q in choices if clearance.free(point, q, margin=.06)), (None, None))
        if at is None:
            raise ValueError(f'No clear {name} cut in seam column {key}')
        top = column_top(scene, *key, labels) + 2.
        if not inserted:
            hand.prepare(tool, at, orientation=q, parameters={'jawOpeningMm': 0.} if tool == 'dural_scissors' else None)
            hand.action('Bring the cutting edge over a remaining ' + name + ' attachment', 'move', .5,
                        at=[*key, top], orientation=q)
            inserted = True
        tip = np.asarray(hand.slot['position'])
        lift = max(top, column_top(scene, tip[0], tip[1], labels) + 2.)
        here = hand.slot['quaternion']
        # Beside a bone wall a coarse step's swept cover is wider than the
        # clearance; move there in fine steps, above it in ordinary ones.
        path = vertical(clearance, tip, [tip[0], tip[1], lift], here) if inserted > 1 else []
        path += [pose([tip[0], tip[1], lift], here), pose([*key, lift], q), pose([*key, top], q)]
        path += vertical(clearance, [*key, top], at, q)
        travel = sum(math.dist(u['position'], v['position']) for u, v in zip(path, path[1:]))
        hand.action('Divide a remaining ' + name + ' attachment', 'apply', max(.6, travel / 6.),
                    path=path)
        inserted = 2
        if any(edge not in body(live, name)['seamEdges'] for edge in remaining
               if np.allclose(scene.points(edge).mean(axis=0)[:2], key)):
            raise ValueError(f'The vertical stroke did not divide seam column {key}')
    if inserted:
        tip = np.asarray(hand.slot['position'])
        hand.action('Raise the cutting edge clear of the incision', 'apply', .5,
                    path=vertical(clearance, tip, [tip[0], tip[1], column_top(live.scene, *tip[:2], labels) + 2.],
                                  hand.slot['quaternion']))
        hand.withdraw('Withdraw from the completed ' + name + ' incision')


def incise_scalp(hand, course, angle):
    """Cut the full scalp thickness along the seam, then divide any remaining attachment column."""
    live = hand.live
    clearance = ScalpClearance(hand)
    # The deepest seam point can sit beside an oblique outer table. Seat the
    # blade at the first depth clear of bone, as along the rest of the course.
    first = shallow(clearance, course[0], angle)
    if first is None:
        raise ValueError('The first scalp seam column has no bone-clear depth')
    tip = np.asarray(hand.slot['position'])
    path = vertical(clearance, [*first[:2], tip[2]], first, angle)
    hand.action('Incise through the scalp to the calvarium', 'apply', 1.2, path=path)
    live.save('first-engagement')
    contour(hand, clearance, course, angle)
    hand.withdraw('Lift the blade out of the completed scalp incision')
    remaining_columns(hand, clearance)
    if body(live, 'scalp')['status'] != 'incised':
        raise ValueError('Cutting left scalp attachments intact')


def hands(live):
    from glioma_walkthrough_actions import Hand
    from host_rehearsal_frame import table_landmarks
    marks = table_landmarks({'boneRadiusMm': 0.})
    return tuple(Hand(live, side, marks.outside[side]) for side in ('left', 'right'))


def _approach(live, landmarks, left, right):
    from glioma_walkthrough_actions import seam_course as source_course

    def seam_course(scene, name):
        return chained(source_course(scene, name))
    right.action('Inspect the positioned head and the planned approach', 'inspect')
    # The assistant's idle instrument starts beside the entry; park it clear of the incision.
    left.withdraw('Park the idle left instrument clear of the planned incision')
    right.withdraw('Park the idle right instrument at the scrub side')
    live.scalp_course = seam_course(live.scene, 'scalp')
    live.scalp_angle = enter_scalp(right, live.scalp_course)


def _incise_scalp(live, landmarks, left, right):
    from glioma_walkthrough_actions import seam_course
    course = getattr(live, 'scalp_course', None) or chained(seam_course(live.scene, 'scalp'))
    # A resumed approach checkpoint has the blade staged over the first column.
    angle = getattr(live, 'scalp_angle', None) or edge_quaternion(*course[:2])
    incise_scalp(right, course, angle)


def _reflect_scalp(live, landmarks, left, right):
    # No vascular graph is declared for the host scalp, so no scalp-edge source
    # exists for a hemostatic Raney clip; scalp-edge bleeding is not simulated.
    reflect_sheet(live, 'scalp', right, left)


CHECKPOINTS = [
    ('approach', 'approach', _approach),
    ('scalp-incised', 'scalp', _incise_scalp),
    ('scalp-open', 'scalp', _reflect_scalp),
]


def turned(angle, cap=math.radians(70.)):
    """A downward instrument tipped about the hinge axis (+x) to follow a flap reflected by ``angle``."""
    return _turned(angle, cap)


def _turned(angle, cap):
    from scipy.spatial.transform import Rotation
    return (Rotation.from_euler('x', -min(angle, cap)) * Rotation.from_quat(DOWN)).as_quat().tolist()


def normal(angle):
    """Outer-face normal, in the table frame, of a flap reflected by ``angle`` about +x toward +y."""
    return np.array([0., math.sin(angle), math.cos(angle)])


def back_off(hand, label, direction, distance=6.):
    """Withdraw along ``direction`` off a surface, then up and out to the parked position."""
    tip = np.asarray(hand.slot['position'])
    q = hand.slot['quaternion']
    clear = tip + distance * np.asarray(direction)
    height = max(hand.clearance, clear[2] + 5.)
    route = [pose(clear, q), pose([clear[0], clear[1], height], q), pose([*hand.parking[:2], height], DOWN),
             pose(hand.parking, DOWN)]
    hand.action(label, 'withdraw', 1.2, path=route)


def reflect_sheet(live, name, holding, assistant, *, inward=0., opened_deg=100., cap_deg=70.):
    """Grasp a sheet just inside its free rim, fold it about its hinge and place the retaining stay.

    As sheet_handling in the phantom walkthrough, with the jaws seated
    ``inward`` mm toward the hinge (+y in the table frame) so that the closed
    jaws stay clear of the retained tissue across the incision.
    """
    from resectionlab.surgical_exposure_geometry import grip_trajectory, site_trajectory
    from scipy.spatial.transform import Rotation
    from glioma_bimanual_routes import retained_site
    cfg = live.scene.properties['operativeExposureV1']['bodies'][name]
    opened = math.radians(opened_deg)
    site = np.asarray(grip_trajectory(live.scene, name, 0.)) + [0., inward, 0.]
    holding.prepare('membrane_forceps', site + [0., 0., 8.], parameters={'jawOpeningMm': 0., 'forceN': 1.2})
    # Close the jaws at first contact with the sheet's outer face, so the
    # registered grip is not pushed into the attached sheet.
    grip = first_contact(holding, site, DOWN, np.asarray(body(live, name)['centersMm']))
    holding.contact('Grasp ' + name + ' material', grip, operation='grasp')
    if body(live, name)['heldBy'] != holding.side:
        raise ValueError('No measured ' + name + ' grip')
    offset = np.asarray(body(live, name)['gripOffsetMm'])
    support = site_trajectory(live.scene, name, cfg['anchorCells'][0], opened)
    oblique = (Rotation.from_euler('y', -22., degrees=True)
               * Rotation.from_quat(edge_quaternion([0, 0], [0, 1]))).as_quat().tolist()
    angles = np.linspace(0., opened, 21)[1:]
    # The forceps turns with the flap's outer face (capped at 70 degrees), so
    # the rim beyond the jaws never folds over its shaft; the jaws stay on the
    # grip, which is all the registered grip checks.
    path = [pose(np.asarray(grip_trajectory(live.scene, name, a)) + offset, turned(a, math.radians(cap_deg)))
            for a in angles]
    for first in range(0, len(path), 4):
        holding.action('Reflect ' + name + ' about its attachments', 'move', .6, path=path[first:first + 4])
    live.save(name + '-held-reflection')
    retained_site(assistant, 'raney_clip_applier', 'Secure the reflected ' + name, support,
                  parameters={'jawOpeningMm': 0.}, orientation=oblique)
    holding.action('Release supported ' + name, 'release')
    back_off(holding, 'Withdraw ' + name + ' forceps', normal(opened) if cap_deg else np.array([0., 0., 1.]))


BURR = {'rpm': 60000., 'forceN': 7., 'duty': .8}
# The router runs at the engine's bone-model limits (80 000 rpm, 20 N, full
# duty) so the side cutter clears the 4-8 mm host skull ahead of its axis.
ROUTER = {'rpm': 80000., 'forceN': 20., 'duty': 1.}


def perforate(live, hand):
    """Burr the pilot hole through the full local skull thickness in contact-limited stations."""
    scene = live.scene
    cfg = scene.properties['operativeExposureV1']['bodies']['bone']
    pilot = np.asarray(cfg['pilotCenterMm'])
    port = scene.points(cfg['portCells'])
    column = port[np.hypot(*(port[:, :2] - pilot[:2]).T) < .6]
    top, bottom = float(column[:, 2].max()), float(column[:, 2].min())
    hand.prepare('cranial_drill', [pilot[0], pilot[1], top + 8.], parameters=BURR)
    stations = [*np.arange(top + 1.8, bottom + 1.0, -1.5), bottom + 1.0]
    for z in stations:
        for _ in range(40):
            outcome = hand.action('Perforate outer table, diploe and inner table', 'apply', 1.,
                                  at=[pilot[0], pilot[1], z], allowed_partial=('BONE_CUTTING_IN_PROGRESS',))
            if outcome['status'] == 'accepted':
                break
        else:
            raise ValueError('The burr did not clear the pilot contact with bounded work')
    hand.withdraw('Disengage and withdraw the burr')


def _foot_window(scene, p, back, bone_ids, dura_ids):
    """Tip heights at axis point ``p`` that keep the foot under every bone cell and above the dura."""
    foot = (p + 1.55 * back, p - .55 * back)
    i0, j0, _ = np.rint(scene.inverse[:3, :3] @ [*p, 0.] + scene.inverse[:3, 3]).astype(int)
    base = scene.points([np.ravel_multi_index((i0, j0, 0), scene.shape)])[0]
    bone_low, dura_high = np.inf, -np.inf
    for di in range(-3, 4):
        for dj in range(-3, 4):
            centre = base[:2] + np.array([di, dj]) * scene.spacing[:2]
            a, b = foot
            t = np.clip((centre - a) @ (b - a) / ((b - a) @ (b - a)), 0., 1.)
            near_foot = np.linalg.norm(centre - (a + t * (b - a))) <= .36 + .71
            near_axis = np.linalg.norm(centre - p) <= .75 + .71
            if not (near_foot or near_axis):
                continue
            column = scene.labels[i0 + di, j0 + dj]
            zs = base[2] + np.arange(len(column)) * scene.spacing[2]
            upper = zs > -40.
            bone = zs[np.isin(column, list(bone_ids)) & upper]
            if near_foot and len(bone):
                bone_low = min(bone_low, float(bone[bone > bone.max() - 15.].min()))
            dura = zs[np.isin(column, list(dura_ids)) & upper]
            if len(bone):
                dura = dura[dura < bone.max()]
            if len(dura):
                dura_high = max(dura_high, float(dura.max()) + (1.56 if near_foot else 1.25))
    return dura_high, bone_low - .16


def kerf_course(scene, *, radius=23.2, spacing=.3, lead=math.radians(20.), sample=.075):
    """Cutter axis points around the flap, from the burr hole, with the foot under the inner table.

    The guard foot runs .7 mm below the cutter tip, 1.55 mm behind and .55 mm
    ahead of its axis. Each tip height leaves the foot clear of every bone cell
    above it and of the dura below it, over every position within one course
    spacing, so the footplate follows the inner table all the way round.
    """
    cfg = scene.properties['operativeExposureV1']['bodies']['bone']
    pilot = np.asarray(cfg['pilotCenterMm'])
    start = math.atan2(pilot[1], pilot[0])
    sweep = lead + 2 * math.pi

    def axis(travel):
        blend = np.clip(travel / lead, 0., 1.)
        r = radius + (np.hypot(*pilot[:2]) - radius) * (1 - blend) ** 2
        return np.column_stack((r * np.cos(start + travel), r * np.sin(start + travel)))
    fine = np.linspace(0., sweep, int(math.ceil(sweep * radius / sample)))
    dense = axis(fine)
    tangent = np.gradient(dense, axis=0)
    back = -tangent / np.linalg.norm(tangent, axis=1)[:, None]
    bone_ids = {r['id'] for r in scene.regions if r['material'] == 'bone'}
    dura_ids = {r['id'] for r in scene.regions if r['material'] == 'membrane'}
    windows = np.array([_foot_window(scene, p, b, bone_ids, dura_ids) for p, b in zip(dense, back)])
    travel = np.linspace(0., sweep, int(math.ceil(sweep * radius / spacing)))
    course = []
    for t, p in zip(travel, axis(travel)):
        near = np.abs(fine - t) * radius <= spacing
        low, high = windows[near, 0].max(), windows[near, 1].min()
        if not low < high:
            raise ValueError(f'No footplate height between the inner table and the dura at {np.round(p, 2)}')
        course.append([float(p[0]), float(p[1]), float((low + high) / 2.)])
    # The tip moves linearly between course points; every sample between them
    # must stay inside its own window.
    heights = np.interp(fine, travel, [c[2] for c in course])
    bad = (heights <= windows[:, 0]) | (heights >= windows[:, 1])
    if bad.any():
        raise ValueError(f'Footplate course leaves its window at {np.round(dense[np.argmax(bad)], 2)}')
    # The cutter axis and its trailing guard post (both within .34 mm of the
    # path, over the 14 mm flute) pass only through bone the cutter may remove.
    stock = set(cfg['bladeCorridorCells']) | set(cfg['portCells'])
    checked, swath, axial = set(), set(), set()
    for p, h in zip(dense, heights):
        i0, j0, k0 = np.rint(scene.inverse[:3, :3] @ [*p, h] + scene.inverse[:3, 3]).astype(int)
        for di in (-2, -1, 0, 1, 2):
            for dj in (-2, -1, 0, 1, 2):
                for dk in range(-2, 17):
                    cell = (i0 + di, j0 + dj, k0 + dk)
                    index = int(np.ravel_multi_index(cell, scene.shape))
                    if index not in stock:
                        continue
                    centre = scene.points([index])[0]
                    gap = np.maximum(np.abs(centre - [*p, np.clip(centre[2], h, h + 14.)]) - .5, 0.)
                    if np.linalg.norm(gap) <= .75:
                        swath.add(index)
                        if np.abs(centre[:2] - p).max() <= .5 and h - .5 <= centre[2] <= h + 14.5:
                            axial.add(index)
    # The cutter removes stock only next to stock it has already removed,
    # starting from the burr hole: every cell on its axis must be reachable so.
    pilot_axis = {int(c) for c in cfg['portCells']
                  if np.abs(scene.points([c])[0][:2] - pilot[:2]).max() <= .5}
    reached, frontier = set(pilot_axis), list(pilot_axis)
    offsets = [(a, b, c) for a in (-1, 0, 1) for b in (-1, 0, 1) for c in (-1, 0, 1) if a or b or c]
    while frontier:
        cell = np.unravel_index(frontier.pop(), scene.shape)
        for a, b, c in offsets:
            index = int(np.ravel_multi_index((cell[0] + a, cell[1] + b, cell[2] + c), scene.shape))
            if index in swath and index not in reached:
                reached.add(index)
                frontier.append(index)
    stranded = axial - reached
    if stranded:
        raise ValueError(f'{len(stranded)} bone cells on the cutter axis are cut off from the burr hole, e.g. '
                         f'{np.round(scene.points([min(stranded)])[0], 1)}')
    for p, h in zip(dense, heights):
        i0, j0, k0 = np.rint(scene.inverse[:3, :3] @ [*p, h] + scene.inverse[:3, 3]).astype(int)
        for di in (-1, 0, 1):
            for dj in (-1, 0, 1):
                for dk in range(-2, 17):
                    cell = (i0 + di, j0 + dj, k0 + dk)
                    if cell in checked:
                        continue
                    checked.add(cell)
                    centre = scene.points([np.ravel_multi_index(cell, scene.shape)])[0]
                    if (np.abs(centre[:2] - p).max() > .5 + .34 or not h - .5 <= centre[2] <= h + 14.5
                            or int(scene.labels[cell]) not in bone_ids):
                        continue
                    index = int(np.ravel_multi_index(cell, scene.shape))
                    if index not in stock:
                        raise ValueError('The cutter path crosses bone outside the declared kerf stock at '
                                         f'{np.round(centre, 1)}')
    return course


def cut_kerf(live, hand, *, speed=.3, chunk=10):
    """Seat the guarded router in the burr hole and cut the full circumference with the foot under the table."""
    course = kerf_course(live.scene)

    def trailing(a, b):
        return edge_quaternion(b, a)
    first = trailing(course[0], course[1])
    pilot = live.scene.properties['operativeExposureV1']['bodies']['bone']['pilotCenterMm']
    hand.prepare('craniotome', [pilot[0], pilot[1], course[0][2] + 20.], orientation=first, parameters=ROUTER)
    hand.contact('Seat the guarded foot in the burr hole', course[0], orientation=first, duration=1.)
    dwell = 'OPERATIVE_KERF_FEED_REQUIRES_CUTTING_DWELL'
    index = 1
    while index < len(course):
        points = course[index - 1:index + chunk]
        travel = sum(math.dist(a, b) for a, b in zip(points, points[1:]))
        frames = [pose(b, trailing(a, b)) for a, b in zip(points, points[1:])]
        outcome = hand.action('Cut the guarded circumferential kerf', 'apply', max(.2, travel / speed), path=frames,
                              allowed_partial=(dwell,))
        if outcome['status'] == 'accepted':
            index += chunk
            continue
        # Thick bone ahead: hold the rotating cutter where it stopped until the
        # stock in front of its axis is cut, then go on from the next point.
        tip = np.asarray(hand.slot['position'])
        hand.action('Dwell while the cutter clears thick bone ahead', 'apply', 1.5,
                    at=tip, orientation=hand.slot['quaternion'])
        reached = int(np.argmin([math.dist(tip, c) for c in course[index - 1:index + chunk]])) + index - 1
        course[reached] = tip.tolist()
        index = reached + 1
    if body(hand.live, 'bone')['status'] != 'free':
        raise ValueError('Residual bony connections prevent flap mobilisation')
    hand.withdraw('Return the guarded cutter through the completed kerf')


def _burr(live, landmarks, left, right):
    perforate(live, right)


def _kerf(live, landmarks, left, right):
    cut_kerf(live, right)


def first_contact(hand, site, q, centres, *, above=8.):
    """Tip position over ``site`` where the mounted tool's working parts first touch a cell at ``centres``.

    The jaws close there, so the registered grip does not push the body.
    """
    from resectionlab.geometry import point_segment_distances
    from scipy.spatial.transform import Rotation
    parts = [(np.asarray(c['a']), np.asarray(c['b']), c['radiusMm']) for c in hand.slot['contactGeometry']
             if c['part'] != 'shaft']
    rotation = Rotation.from_quat(q)
    # A moved body occupies the grid cells its centres round to (table cells sit on integers).
    centres = np.rint(np.asarray(centres, dtype=float))
    near = centres[np.linalg.norm(centres[:, :2] - site[:2], axis=1) <= 6.]

    def touches(z):
        tip = np.array([site[0], site[1], z])
        for a, b, r in parts:
            a, b = tip + rotation.apply(a), tip + rotation.apply(b)
            # Distance to each cell's cube: shrink the point set by the half cell.
            d = point_segment_distances(near, a, b)
            if np.any(d <= r + .5):
                closest = near[d <= r + .87]
                for centre in closest:
                    t = np.clip((centre - a) @ (b - a) / max((b - a) @ (b - a), 1e-12), 0., 1.)
                    point = a + t * (b - a)
                    gap = np.maximum(np.abs(point - centre) - .5, 0.)
                    if np.linalg.norm(gap) <= r:
                        return True
        return False
    high = float(site[2] + above)
    low = float(site[2])
    if touches(high):
        raise ValueError('The grip site is covered from above')
    for _ in range(40):
        middle = (high + low) / 2
        if touches(middle):
            low = middle
        else:
            high = middle
    return np.array([site[0], site[1], high - .005])


def move_bone(live, hand, *, closing=False):
    """Grasp the bone flap at first cup contact, carry it to its rest or seat, release and withdraw.

    As the phantom's move_bone, but the cups close at first contact with the
    outer face of the grip column, since the dura lies directly under the flap
    where it is not stripped and a pushed flap would be driven into it.
    """
    from scipy.spatial.transform import Rotation
    cfg = live.scene.properties['operativeExposureV1']['bodies']['bone']
    other = 'right' if hand.side == 'left' else 'left'
    parked = live.view['slots'][other]
    transfer_park = np.asarray(cfg['parkCenterMm']) + [42., -27., 21.]
    live.step('Retire the opposite instrument clear of the bone transfer', {
        'slot': other, 'operation': 'withdraw', 'duration': .55,
        'pose': {'position': transfer_park.tolist(), 'quaternion': parked['quaternion']},
    })
    flap = body(live, 'bone')
    site = np.asarray(flap['gripMm'])
    hand.prepare('bone_flap_holder', site + [0., 0., 8.], parameters={'jawOpeningMm': 0., 'forceN': 1.5})
    contact = first_contact(hand, site, DOWN, np.asarray(flap['centersMm']))
    hand.contact('Grasp ' + ('parked' if closing else 'freed') + ' bone flap', contact, operation='grasp')
    if body(live, 'bone')['heldBy'] != hand.side:
        raise ValueError('Bone holder has no measured material grip')
    rest = np.asarray(cfg['restCenterMm'])
    park = np.asarray(cfg['parkCenterMm'])
    start, finish = (park, rest) if closing else (rest, park)
    held_center = np.mean(body(live, 'bone')['centersMm'], axis=0)
    offset = np.asarray(hand.slot['position']) - held_center
    lift_z = max(rest[2], park[2]) + 4.
    middle = (start + finish) / 2
    positions = [[start[0], start[1], lift_z], [middle[0], middle[1], lift_z], [finish[0], finish[1], lift_z], finish]
    for point, tilt in zip(positions, [0., 12., 0., 0.], strict=True):
        rotation = Rotation.from_euler('y', tilt, degrees=True)
        q = (rotation * Rotation.from_quat(DOWN)).as_quat().tolist()
        hand.action('Carry held flap to ' + ('its original seat' if closing else 'its exterior support'),
                    'move', .7, at=np.asarray(point) + rotation.apply(offset), orientation=q)
    hand.action('Release ' + ('replaced' if closing else 'parked') + ' bone flap', 'release')
    hand.withdraw('Withdraw bone holder')


def _park_bone(live, landmarks, left, right):
    move_bone(live, left)


CHECKPOINTS += [
    ('bone-pilot', 'bone', _burr),
    ('bone-kerf', 'bone', _kerf),
    ('bone-parked', 'bone', _park_bone),
]


def _incise_dura(live, landmarks, left, right):
    """Scissors along the dural seam at the depth of its deepest cells, then any remaining column."""
    from glioma_walkthrough_actions import seam_course
    course = [[x, y, z - .30] for x, y, z in chained(seam_course(live.scene, 'dura'))]
    angle = edge_quaternion(*course[:2])
    right.prepare('dural_scissors', course[0], orientation=angle, parameters={'jawOpeningMm': 0.})
    field = Field(live)
    # The closed scissors may touch only the dura itself.
    clearance = Clearance(right, (field.labels > 0) & ~field.mask('membrane'))
    first = shallow(clearance, course[0], angle, limit=1.)
    if first is None:
        raise ValueError('The first dural seam column has no clear depth')
    above = column_top(live.scene, first[0], first[1], field.labels) + 2.
    right.action('Approach the dura over the planned opening', 'move', .6, at=[first[0], first[1], above],
                 orientation=angle)
    right.action('Engage both dural cutting edges', 'apply', 1.,
                 path=vertical(clearance, [first[0], first[1], above], first, angle))
    contour(right, clearance, course, angle, field.labels, label='Divide the dural attachment with closed scissors')
    right.withdraw('Disengage the dural scissors from the divided course')
    remaining_columns(right, clearance, 'dura', heights=(.03, .1, .18, .25), labels=field.labels,
                      tool='dural_scissors')
    if body(live, 'dura')['status'] != 'incised':
        raise ValueError('Dural attachments remain intact')


def _reflect_dura(live, landmarks, left, right):
    # A vertical forceps keeps its handle clear of the scalp rolled back over the hinge.
    reflect_sheet(live, 'dura', right, left, cap_deg=0.)


CHECKPOINTS += [
    ('dura-incised', 'dura', _incise_dura),
    ('dura-open', 'dura', _reflect_dura),
]


CUSA_RATE = 40.


def _dural_opening(scene):
    """Inside of the reflected dural horseshoe, shrunk by the corticotomy and tip margins."""
    cells = scene.points(scene.properties['operativeExposureV1']['bodies']['dura']['cells'])
    radius = float(np.hypot(cells[:, 0], cells[:, 1]).max()) - 1.
    hinge = float(cells[:, 1].max()) - 1.
    margin = 8. + 2.5

    def inside(x, y):
        r = radius - margin
        return (y <= 0. and math.hypot(x, y) <= r) or (0. < y <= hinge - margin and abs(x) <= r)
    return inside


def _touching(hand, tip, q, centres, roles):
    """Indices of ``centres`` whose cells the mounted tool's parts of ``roles`` touch with its tip at ``tip``."""
    from scipy.spatial.transform import Rotation
    rotation = Rotation.from_quat(q)
    hits = set()
    for c in hand.slot['contactGeometry']:
        if c['part'] not in roles:
            continue
        a, b = tip + rotation.apply(c['a']), tip + rotation.apply(c['b'])
        t = np.clip((centres - a) @ (b - a) / max((b - a) @ (b - a), 1e-12), 0., 1.)
        gap = np.maximum(np.abs(a + t[:, None] * (b - a) - centres) - .5, 0.)
        hits.update(np.flatnonzero(np.linalg.norm(gap, axis=1) <= c['radiusMm']).tolist())
    return hits


def _map_cortex(live, hand, centre):
    """Stimulate exposed cortex beside the planned corticotomy; with no functional evidence it reads unknown.

    The site is the nearest point within 4 mm whose first contact, from above,
    is cortex under the electrodes and nothing under the insulated arms.
    """
    field = Field(live)
    scene = live.scene
    hand.prepare('cortical_probe', centre + [0., 0., 25.])
    index = np.rint(scene.inverse[:3, :3] @ centre + scene.inverse[:3, 3]).astype(int)
    lo = np.maximum(index - [8, 8, 10], 0)
    block = field.labels[lo[0]:index[0] + 9, lo[1]:index[1] + 9, lo[2]:index[2] + 26]
    cells = np.argwhere(block > 0) + lo
    flat = np.ravel_multi_index(cells.T, scene.shape)
    centres = scene.points(flat)
    soft = np.isin(field.labels.ravel()[flat], [r['id'] for r in scene.regions if r['material'] == 'soft_tissue'])
    offsets = sorted(((x, y) for x in np.arange(-4., 4.01, 1.) for y in np.arange(-4., 4.01, 1.)),
                     key=lambda o: math.hypot(*o))
    for dx, dy in offsets:
        site = centre + [dx, dy, 0.]
        high, low = site[2] + 22., site[2] - 8.
        if _touching(hand, np.array([site[0], site[1], high]), DOWN, centres, ('working', 'shaft')):
            continue
        for _ in range(40):
            middle = (high + low) / 2
            if _touching(hand, np.array([site[0], site[1], middle]), DOWN, centres, ('working', 'shaft')):
                low = middle
            else:
                high = middle
        tip = np.array([site[0], site[1], high - .02])
        if _touching(hand, tip + [0., 0., .01], DOWN, centres, ('shaft',)):
            continue
        under = _touching(hand, tip, DOWN, centres, ('working',))
        if under and all(soft[k] for k in under):
            break
    else:
        raise ValueError('No exposed cortex beside the corticotomy for stimulation')
    hand.contact('Stimulate the cortex beside the planned corticotomy', tip, operation='apply',
                 parameters={'amplitudeMa': 1., 'frequencyHz': 50., 'pulseWidthUs': 200.})
    rows = [o for o in live.view['observations'] if o.get('probe') == 'cortical' and o.get('cells')]
    if not rows or rows[-1]['result'] != 'unknown':
        raise ValueError('Mapping did not contact cortex, or reported evidence that does not exist')
    hand.withdraw('Withdraw the cortical stimulation probe')
    return rows[-1]


def _cusa_strokes(live, hand, strokes, *, max_seconds=25., max_poses=150):
    """Execute planned CUSA strokes; a stalled tip dwells and the stroke continues from where it stopped."""
    from host_resection_plan import tool_quaternion
    progress = 'TISSUE_ASPIRATION_IN_PROGRESS'
    for number, stroke in enumerate(strokes, 1):
        forward = [pose(t, tool_quaternion(d)) for t, d in stroke['poses']]
        # The stroke returns along its own poses with the aspirator still
        # running, so anything that moved into the corridor is taken on the way
        # out instead of blocking a passive withdrawal.
        poses = forward + forward[-2::-1]
        work = list(stroke['work']) + [0] * (len(forward) - 1)
        params = {'power': stroke['power']}
        hand.action(f'Align the aspirator on corridor line {number}', 'move', .6, path=[poses[0]])
        index = 1
        stalled = False
        while index < len(poses):
            end = index
            seconds = 0.
            while end < len(poses) and end - index < max_poses:
                travel = math.dist(poses[end - 1]['position'], poses[end]['position'])
                cost = work[end] / (CUSA_RATE * stroke['power']) * 1.25 + travel / 25.
                if end > index and seconds + cost > max_seconds:
                    break
                seconds += cost
                end += 1
            label = {'corticotomy': 'Open the corticotomy and transcortical corridor',
                     'debulk': 'Debulk the tumour core and rim', 'margin': 'Work the cavity wall at low power'}[stroke['kind']]
            # The engine shares an action's time evenly between its segments and
            # stops at the first segment whose tip is still in contact, so each
            # segment gets the time of the slowest one; after a stall the tip
            # dwells on one segment at a time.
            costs = [work[k] / (CUSA_RATE * stroke['power']) * 1.5 + .15 for k in range(index, end)]
            if stalled:
                end = index + 1
                seconds = max(3., 3. * costs[0])
            else:
                seconds = max(costs) * len(costs)
                while seconds > max_seconds and end - index > 1:
                    end -= 1
                    costs.pop()
                    seconds = max(costs) * len(costs)
            start = np.asarray(hand.slot['position'])
            outcome = hand.action(label, 'apply', min(30., max(.3, seconds)), path=poses[index:end],
                                  parameters=params, allowed_partial=(progress,))
            if outcome['status'] == 'accepted':
                index = end
                stalled = False
                continue
            stalled = True
            # The reported fraction is of path length: find the segment that stalled.
            points = [start] + [np.asarray(poses[k]['position']) for k in range(index, end)]
            lengths = np.array([np.linalg.norm(b - a) for a, b in zip(points, points[1:])])
            walked = outcome['completedFraction'] * lengths.sum()
            reached = int(np.searchsorted(np.cumsum(lengths), walked + 1e-9, side='right'))
            index = index + min(reached, end - index - 1)
        # Out above the grid on the entry line: a passive withdrawal ends the stroke.
        top = np.asarray(poses[-1]['position']) + [0., 0., 2.]
        hand.action('Withdraw the aspirator above the field', 'withdraw', .3, path=[pose(top, poses[-1]['quaternion'])])
    hand.action('Stop ultrasonic aspiration', 'deactivate')


def _irrigate_and_suction(live, hand, centre):
    hand.prepare('irrigation', centre + [0., 0., 10.])
    nozzle = centre + [0., 0., 2.]
    hand.action('Bring the irrigation nozzle into the cavity', 'move', .6, at=nozzle)
    hand.contact('Irrigate the resection cavity', nozzle, duration=1.,
                 parameters={'flowMm3PerSecond': 20., 'depthMm': 2.})
    hand.withdraw('Withdraw the irrigation nozzle')
    removed = live.view['metrics']['fluidRemovedMm3']
    wet = [(int(c), v) for c, v in live.view['fluid'].items() if v > 1e-9]
    if not wet:
        raise ValueError('Irrigation left no retained fluid in the cavity')
    points = live.scene.points([c for c, _ in wet])
    reach = [(np.linalg.norm(q[:2] - centre[:2]) + abs(q[2] - centre[2]) * .2, q) for q in points
             if np.linalg.norm(q[:2] - centre[:2]) <= 3. and centre[2] - 8. <= q[2] <= centre[2] + 4.]
    if not reach:
        raise ValueError('No retained irrigation fluid within reach of the corridor')
    pool = min(reach, key=lambda r: r[0])[1]
    hand.prepare('frazier_suction', pool + [0., 0., 10.])
    hand.contact('Aspirate the irrigation pool', pool + [0., 0., 1.65], duration=.65,
                 parameters={'flowMm3PerSecond': 12.})
    hand.action('Hold the suction above the pool', 'apply', .8)
    if live.view['metrics']['fluidRemovedMm3'] <= removed:
        raise ValueError('Suction removed no field fluid')
    hand.withdraw('Withdraw the suction from the cleared cavity')


def _treat(live, landmarks, left, right):
    from host_resection_plan import plan_layers
    right.action('Inspect the exposed cortex and plan the corticotomy', 'inspect')
    # The assistant's idle instrument moves well clear of the aspirator's lines and handpiece.
    tip = np.asarray(left.slot['position'])
    # Beyond the reach of the aspirator's 184 mm handpiece on any line through the corticotomy.
    far = [0., -260., 140.]
    home = list(left.parking)
    left.action('Park the idle assistant instrument clear of the corridor', 'withdraw', 1.5,
                path=[pose([tip[0], tip[1], max(tip[2], 140.)], left.slot['quaternion']),
                      pose([tip[0], -260., 140.], DOWN), pose(far, DOWN)])
    left.parking = far
    from resectionlab.surgical_motion import envelopes
    from resectionlab.surgical_tools import operative_tool_parameters
    slot = live.view['slots']['left']
    row = next(t for t in live.view['catalog'] if t['id'] == slot['instrument'])
    obstacles = envelopes(row, operative_tool_parameters(row['id'], slot['parameters'], 3),
                          slot['position'], slot['quaternion'])
    field = Field(live)
    pivot, strokes, carving = plan_layers(live.scene, field.labels, opening=_dural_opening(live.scene),
                                          obstacles=obstacles)
    live.resection = {'pivot': pivot, 'strokes': len(strokes), 'predicted': dict(carving.removed)}
    print(f'resection plan: {len(strokes)} strokes, predicted {carving.removed}', flush=True)
    site = np.array([pivot[0], pivot[1], column_top(live.scene, pivot[0], pivot[1], field.labels)])
    _map_cortex(live, right, site)
    right.prepare('cusa', strokes[0]['poses'][0][0], parameters={'power': 1.})
    _cusa_strokes(live, right, strokes)
    right.withdraw('Withdraw the aspirator through the corridor')
    live.save('after-cusa')
    _irrigate_and_suction(live, right, pivot)
    cfg = live.scene.properties['operativeExposureV1']
    from glioma_walkthrough_actions import put, retrieve
    site = np.asarray(cfg['cottonoidSiteMm'])
    put(right, 'dressing_forceps', 'Place a counted cottonoid on the cortex beside the corticotomy',
        site + [0., 0., .25], parameters={'jawOpeningMm': 0.})
    cottonoid = next(o for o in live.view['operativeExposureV1']['objects'] if o['kind'] == 'cottonoid')
    retrieve(right, 'dressing_forceps', 'Retrieve the counted cottonoid',
             np.asarray(cottonoid['positionMm']) + [0., 0., cottonoid['radiusMm'] + .15])
    # No vascular graph is declared for this host, so no wound port exists for bipolar sealing.
    right.action('Review the cavity, the field and the material count', 'inspect')
    left.action('Bring the assistant instrument back beside the field', 'withdraw', 1.5,
                path=[pose([home[0], -260., 140.], DOWN), pose([home[0], home[1], 140.], DOWN), pose(home, DOWN)])
    left.parking = home


CHECKPOINTS += [
    ('after-treatment', 'treatment', _treat),
]


def contact_along(hand, site, q, direction, centres, *, start=10.):
    """Tip position on the line ``site + s * direction`` where the working parts first touch a cell at ``centres``."""
    from scipy.spatial.transform import Rotation
    parts = [(np.asarray(c['a']), np.asarray(c['b']), c['radiusMm']) for c in hand.slot['contactGeometry']
             if c['part'] != 'shaft']
    rotation = Rotation.from_quat(q)
    # A moved body occupies the grid cells its centres round to (table cells sit on integers).
    centres = np.rint(np.asarray(centres, dtype=float))
    near = centres[np.linalg.norm(centres - site, axis=1) <= 8.]
    direction = np.asarray(direction, dtype=float) / np.linalg.norm(direction)

    def touches(s):
        tip = site + s * direction
        for a, b, r in parts:
            a, b = tip + rotation.apply(a), tip + rotation.apply(b)
            for centre in near:
                t = np.clip((centre - a) @ (b - a) / max((b - a) @ (b - a), 1e-12), 0., 1.)
                gap = np.maximum(np.abs(a + t * (b - a) - centre) - .5, 0.)
                if np.linalg.norm(gap) <= r:
                    return True
        return False
    high, low = start, -2.
    if touches(high):
        raise ValueError('The regrasp line starts inside the sheet')
    for _ in range(40):
        middle = (high + low) / 2
        if touches(middle):
            low = middle
        else:
            high = middle
    return site + (high - .005) * direction


def return_sheet(live, name, holding, assistant, *, cap_deg=70.):
    """Regrasp a reflected sheet, remove the stay, turn it back and release it seated.

    The forceps approaches along the outer normal tipped at most ``cap_deg``
    from vertical (vertical for the dura, clear of the rolled scalp).
    """
    from resectionlab.surgical_exposure_geometry import grip_trajectory
    from scipy.spatial.transform import Rotation
    from glioma_bimanual_routes import retained_site
    sheet = body(live, name)
    angle = sheet['angleRad']
    site = np.asarray(sheet['gripMm'])
    cap = math.radians(cap_deg)
    outward = normal(min(angle, cap))
    q = turned(angle, cap)
    holding.prepare('membrane_forceps', site + 10. * outward, orientation=q,
                    parameters={'jawOpeningMm': 0., 'forceN': 1.2})
    holding.action('Bring the forceps to the reflected ' + name, 'move', .6, at=site + 10. * outward, orientation=q)
    contact = contact_along(holding, site, q, outward, np.asarray(sheet['centersMm']))
    grasp = holding.contact('Regrasp the reflected ' + name, contact, operation='grasp', orientation=q)
    if not any(e['kind'] == 'exposure_body_grasped' and e['body'] == name for e in grasp['events']):
        raise ValueError('The forceps did not register the ' + name + ' grip')
    offset = np.asarray(body(live, name)['gripOffsetMm'])
    stay = next(o for o in live.view['operativeExposureV1']['objects']
                if o['kind'] == 'retaining-stay' and o['body'] == name and o['status'] == 'placed')
    oblique = (Rotation.from_euler('y', -22., degrees=True)
               * Rotation.from_quat(edge_quaternion([0, 0], [0, 1]))).as_quat().tolist()
    retrieve_stay(live, name, assistant, orientation=oblique)
    angles = np.linspace(angle, 0., 21)[1:]
    path = [pose(np.asarray(grip_trajectory(live.scene, name, a)) + offset, turned(a, cap)) for a in angles]
    for index in range(0, len(path), 4):
        operation = 'grasp' if index + 4 >= len(path) else 'move'
        holding.action('Return ' + name + ' about its attachments', operation, .6, path=path[index:index + 4])
    holding.action('Release the reapproximated ' + name, 'release')
    back_off(holding, 'Withdraw ' + name + ' forceps', normal(0.))
    if body(live, name)['status'] != 'seated':
        raise ValueError(f'The {name} did not seat: {body(live, name)["status"]}')


def _close_dura(live, landmarks, left, right):
    from glioma_walkthrough_actions import close_sites
    return_sheet(live, 'dura', right, left, cap_deg=0.)
    close_sites(live, 'dura', right)


def _replace_bone(live, landmarks, left, right):
    from glioma_walkthrough_actions import close_sites
    move_bone(live, left, closing=True)
    close_sites(live, 'bone', right)


def _close_scalp(live, landmarks, left, right):
    from glioma_walkthrough_actions import close_sites
    # No Raney clip was placed on the scalp edge, so none is retrieved; no
    # scalp vascular port exists to seal.
    return_sheet(live, 'scalp', right, left)
    close_sites(live, 'scalp', right)
    right.action('Review fixation, scalp approximation and the final object count', 'inspect')
    states = {name: body(live, name)['status'] for name in ('dura', 'bone', 'scalp')}
    if states != {'dura': 'closed', 'bone': 'fixed', 'scalp': 'closed'}:
        raise ValueError(f'Closure remains incomplete: {states}')
    if not live.view['operativeExposureV1']['complete']:
        raise ValueError('The endpoint still has an object in the field or an instrument inside')


CHECKPOINTS += [
    ('dura-closed', 'closure', _close_dura),
    ('bone-fixed', 'closure', _replace_bone),
    ('scalp-closed', 'closure', _close_scalp),
]


def retrieve_stay(live, name, hand, *, orientation):
    """Take the retaining stay off a regrasped sheet, carry it to the exterior count and withdraw.

    As retained_site(retrieve=True), but the stay leaves along whichever of
    its surface normal, straight up or their bisector the engine previews as
    clear, since on a curved, thick or folded sheet the normal can point into
    neighbouring tissue.
    """
    from glioma_bimanual_routes import external_route
    stay = next(o for o in live.view['operativeExposureV1']['objects']
                if o['kind'] == 'retaining-stay' and o['body'] == name and o['status'] == 'placed')
    tool = 'raney_clip_applier'
    if hand.slot['instrument'] != tool:
        live.step('Select ' + tool, dict(slot=hand.side, instrument=tool, operation='exchange', duration=.1))
    hand.action('Mount ' + tool + ' at the parked hand', 'place', at=hand.slot['position'],
                orientation=hand.slot['quaternion'], parameters={'jawOpeningMm': 0.})
    origin = pose(hand.slot['position'], hand.slot['quaternion'])
    path, contact = external_route(hand, stay['positionMm'], orientation, operation='grasp',
                                   parameters={'jawOpeningMm': 0.}, expected_event='exposure_object_grasped',
                                   expected_object=stay['id'], surface_normal=stay['normal'],
                                   object_radius=stay['radiusMm'])
    hand.action('Approach beside the supporting hand', 'move', .7, path=path)
    result = hand.action('Retrieve the ' + name + ' retaining stay', 'grasp', .35, at=contact['position'],
                         orientation=contact['quaternion'])
    if not any(e.get('kind') == 'exposure_object_grasped' and e.get('objectId') == stay['id'] for e in result['events']):
        raise ValueError('The applier did not grasp the ' + name + ' stay')
    up = np.array([0., 0., 1.])
    normal_ = np.asarray(stay['normal'])
    tip = np.asarray(contact['position'])
    lift, reasons = None, set()
    # The normal, straight up and their bisector first, then a spread of directions.
    golden = math.pi * (3. - math.sqrt(5.))
    spread = [np.array([math.cos(golden * k) * math.sqrt(1 - z * z), math.sin(golden * k) * math.sqrt(1 - z * z), z])
              for k, z in enumerate(np.linspace(.95, -.3, 24))]
    for direction in (normal_, (normal_ + up) / np.linalg.norm(normal_ + up), up, *spread):
        for distance in (3.5, 2.):
            free = tip + distance * direction
            # The closed jaws keep their grip on the stay while it is carried.
            action = dict(slot=hand.side, operation='grasp', duration=.35,
                          pose={'position': free.tolist(), 'quaternion': contact['quaternion']})
            preview = live.command('preview', expectedStateId=live.view['stateId'], expectedEpisodeId=live.episode,
                                   action=action)
            if preview['status'] == 'accepted':
                lift = free
                break
            reasons.add(str(preview.get('reason')))
        if lift is not None:
            break
    if lift is None:
        raise ValueError('No clear direction to lift the ' + name + ' stay off the sheet: ' + ', '.join(sorted(reasons)))
    hand.action('Lift the stay away from the retained surface', 'grasp', .35, at=lift.tolist(),
                orientation=contact['quaternion'])
    outside = lift.copy()
    outside[2] = max(hand.clearance, lift[2] + 5.)
    hand.action('Carry the stay to the exterior count', 'grasp', .8, at=outside.tolist(),
                orientation=contact['quaternion'])
    hand.action('Release the retrieved stay at the exterior', 'release')
    if not any(o['id'] == stay['id'] and o['status'] == 'retrieved' for o in live.view['operativeExposureV1']['objects']):
        raise ValueError('The carried stay did not reach the exterior count')
    hand.withdraw('Withdraw ' + tool + ' from the field')
