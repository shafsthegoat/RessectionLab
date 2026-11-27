"""Bounded stock-suction finish for measured debris in a generated planar kerf."""
from __future__ import annotations

import numpy as np
from scipy.spatial.transform import Rotation

from .native_backend import MAX_BATCH_ITEMS
from .surgical_joint_cranial_exit import _commands, _last_place, joint_exit_complete
from .surgical_staged_exit import _duration
from .surgical_tools import TOOLS


def kerf_cleanup_actions(engine, hand, *, check=lambda: None):
    """Propose actual suction poses; native previews, commits and STOP decide the endpoint.

    A station may approach only along the source entry normal into removed kerf.
    Each dwell uses the full stock tip footprint and the field mass it could collect.
    An exterior withdrawal separates stations whose direct transit crosses material.
    Any source class with a planar kerf qualifies (owner decision 2026-11-27).
    """
    access = engine._cranial_access
    debris = engine.state['debris']
    slot = engine.state['slots'][hand]
    if access is None or not debris or len(debris) > 512 or slot['instrument'] != 'cusa':
        return []
    cells = {int(key) for key in debris}
    if not cells <= set(access.kerf_cells).intersection(engine.state['removed']):
        return []
    outside = _last_place(engine, hand)
    if outside is None or engine.state['holds'] or engine.state.get('detachedFragments'):
        return []
    other = 'right' if hand == 'left' else 'left'
    other_slot = engine.state['slots'][other]
    if other_slot['inserted'] or other_slot['active'] or other_slot['grip']:
        return []
    normal = np.asarray(access.entry.normal_ras_unit)
    attitude = outside['quaternion']
    if not np.allclose(Rotation.from_quat(attitude).apply([0., 0., 1.]), normal):
        return []
    points = engine.scene.points(sorted(cells))
    if np.ptp(points @ normal) > 1e-5:
        return []
    exterior_depth = np.dot(outside['position'], normal)
    if exterior_depth >= points[0] @ normal:
        return []
    radius = TOOLS['frazier_suction']['radiusMm']
    # Near-plane sites cover the center of the cut. Stand off along the entry
    # normal at its rim to collect debris without contacting retained material.
    sites = np.concatenate([points - offset * normal for offset in (0., radius, radius + .5)])
    outside_sites = sites + (exterior_depth - sites @ normal)[:, None] * normal
    candidates = []
    for offset in range(0, len(sites), MAX_BATCH_ITEMS):
        check()
        group = sites[offset:offset + MAX_BATCH_ITEMS]
        exteriors = outside_sites[offset:offset + MAX_BATCH_ITEMS]
        covers = engine.cells_many([(p, p, radius) for p in group])
        traversals = engine.touched_many([(a, b, radius) for a, b in zip(exteriors, group)])
        candidates.extend((p, exterior, set(map(int, cover)) & cells)
                          for p, exterior, cover, material in zip(group, exteriors, covers, traversals)
                          if not len(material))
    remaining = set(cells)
    stations = []
    start = np.asarray(outside['position'])
    while remaining and candidates and len(stations) < 48:
        check()
        best = max(range(len(candidates)), key=lambda i: (
            len(candidates[i][2] & remaining), -np.linalg.norm(candidates[i][0] - start)))
        point, projected, covered = candidates.pop(best)
        gain = covered & remaining
        if not gain:
            break
        stations.append((point, projected, covered))
        remaining -= gain
        start = point
    if remaining:
        return []

    def command(op, position=None, duration=.2, **fields):
        item = dict(slot=hand, operation=op, duration=duration, **fields)
        if position is not None:
            item['pose'] = dict(position=position.tolist(), quaternion=attitude)
        return item

    actions = [] if joint_exit_complete(engine) else _commands(engine, (hand,), {hand: outside})
    actions.append(command('exchange', instrument='frazier_suction'))
    params = engine.parameters({'instrument': 'frazier_suction', 'parameters': {}})
    flow = params['flowMm3PerSecond']
    previous = prior_outside = None
    for point, projected, covered in stations:
        if previous is None or len(engine.touched(previous, point, radius)):
            if previous is None:
                actions.append(command('place', projected))
            else:
                actions.append(command('withdraw', prior_outside))
                transit = np.linalg.norm(projected - prior_outside)
                actions.append(command('move', projected, duration=_duration(transit, 20.)))
        actions.append(command('apply', point))
        # Fluid consumes the same finite uptake capacity before debris.
        mass = sum(debris[str(c)] for c in covered)
        mass += sum(engine.state['fluid'].get(str(c), 0.) for c in engine.cells(point, point, radius))
        actions.append(command('apply', point, duration=min(30., max(.2, mass / flow + .1))))
        previous, prior_outside = point, projected
    actions.append(command('withdraw', prior_outside))
    return actions
