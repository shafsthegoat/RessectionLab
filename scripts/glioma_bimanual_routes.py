"""Exterior route proposals for a mounted instrument beside a stationary holder.

Route selection sees the public live hand geometry. The native preview checks
the complete proposed approach against the held sheet, objects, tissue,
protection and the other hand. Only public actions perform the selected motion.
"""

from math import ceil
import numpy as np
from scipy.spatial.transform import Rotation, Slerp

from resectionlab.surgical_motion import pair_conflict, swept_envelopes
from resectionlab.surgical_tools import operative_tool_parameters


def _pose(xyz, q):
    return {'position': list(map(float, xyz)), 'quaternion': list(map(float, q))}


def _hand_geometry(view, hand):
    slot = view['slots'][hand]
    row = next(t for t in view['catalog'] if t['id'] == slot['instrument'])
    from resectionlab.surgical_motion import envelopes

    params = operative_tool_parameters(row['id'], slot['parameters'], row['physicalProfile']['revision'])
    static = envelopes(row, params, slot['position'], slot['quaternion'])
    return row, params, static


def _clears_other(current, path, tool, params, other):
    for end in path:
        a, b = np.asarray(current['position']), np.asarray(end['position'])
        rotations = Rotation.from_quat([current['quaternion'], end['quaternion']])
        angle = (rotations[0].inv() * rotations[1]).magnitude()
        count = max(1, ceil(np.linalg.norm(b - a) / 3), ceil(angle / 0.04))
        samples = np.arange(1, count + 1) / count
        quats = (
            Slerp([0.0, 1.0], rotations)(samples).as_quat() if angle > 1.0e-9 else [end['quaternion']] * count
        )
        start = current
        for t, q in zip(samples, quats):
            at = _pose(a + (b - a) * t, q)
            if pair_conflict(swept_envelopes(tool, params, start, at), other):
                return False
            start = at
        current = end
    return True


def _preview(hand, action):
    live = hand.live
    receipt = live.command(
        'preview', expectedStateId=live.view['stateId'], expectedEpisodeId=live.episode, action=action
    )
    return receipt


def _first_grip(tool, params, q, center, normal, radius):
    """Seat the working metal at first contact without driving an acquired object into its bed."""
    from resectionlab.geometry import point_segment_distances
    from resectionlab.surgical_motion import envelopes

    parts = [(a, b, r) for a, b, r, role in envelopes(tool, params, [0.0, 0.0, 0.0], q) if role == 'working']

    def touches(distance):
        source = (-normal * distance)[None, :]
        return any(point_segment_distances(source, a, b)[0] <= radius + r for a, b, r in parts)

    outside = 6.0
    while touches(outside) and outside < 24.0:
        outside += 2.0
    # The closed clip-applicator jaws meet at the tip. Select the outermost
    # contact along the registered surface normal, with micron-sized overlap.
    inside = 0.0
    for _ in range(24):
        middle = (inside + outside) / 2
        if touches(middle):
            inside = middle
        else:
            outside = middle
    return center + normal * max(0.0, inside - 0.001)


def external_route(
    hand,
    target,
    preferred,
    *,
    operation='apply',
    parameters=None,
    expected_event=None,
    expected_object=None,
    surface_normal=None,
    object_radius=None,
):
    """From the mounted hand return checked exterior waypoints and contact pose.

    Try the supplied attitude first, then cant the handle away from the other
    tip. Go around the stationary tool at the elevated plane if the direct
    approach would pass even one proximal handle through the other. The return
    uses the same path in reverse with unchanged attitude near the held sheet.
    """
    view = hand.live.view
    mate = next(k for k in view['slots'] if k != hand.side)
    tool, params, _ = _hand_geometry(view, hand.side)
    _, _, other = _hand_geometry(view, mate)
    current = _pose(hand.slot['position'], hand.slot['quaternion'])
    hand_xy = np.asarray(view['slots'][mate]['position'][:2])
    target = np.asarray(target, dtype=float)
    difference = target[:2] - hand_xy
    side = difference / max(np.linalg.norm(difference), 0.01)
    preferred_rotation = Rotation.from_quat(preferred)
    # Preserve commanded roll; rotate its outward axis into the proposed cant.
    attitudes = [list(preferred)]
    for angle in (26, 14, 36, 0):
        rad = np.deg2rad(angle)
        desired = np.r_[side * np.sin(rad), np.cos(rad)]
        present = preferred_rotation.apply([0.0, 0.0, -1.0])
        axis = np.cross(present, desired)
        if np.linalg.norm(axis) < 1.0e-7:
            continue
        delta = Rotation.from_rotvec(
            axis / np.linalg.norm(axis) * np.arctan2(np.linalg.norm(axis), present @ desired)
        )
        attitudes.append((delta * preferred_rotation).as_quat().tolist())
    refusals = set()
    for q in attitudes:
        outward = Rotation.from_quat(q).apply([0.0, 0.0, -1.0])
        if outward[2] < 0.4:
            continue
        z = max(hand.clearance, target[2] + 8.0, current['position'][2])
        contact_point = (
            target
            if surface_normal is None
            else _first_grip(tool, params, q, target, np.asarray(surface_normal), object_radius)
        )
        free = target if surface_normal is None else contact_point + 3.5 * np.asarray(surface_normal)
        staging = free + outward * ((z - free[2]) / outward[2])
        entry = _pose(staging, q)
        contact = _pose(contact_point, q)
        # Choose the nearest side of the holder first. Keep vertical transit
        # on its exterior side before turning to the final tool attitude.
        start = np.asarray(current['position'])
        offsets = [
            None,
            (0.0, -28.0),
            (0.0, 28.0),
            (-28.0, 0.0),
            (28.0, 0.0),
            (0.0, -45.0),
            (0.0, 45.0),
            (-45.0, 0.0),
            (45.0, 0.0),
        ]
        for bend in offsets:
            path = []
            if bend is not None:
                shift = np.asarray(bend)
                axis = 0 if shift[0] != 0 else 1
                via0 = start.copy()
                via1 = staging.copy()
                via0[axis] = via1[axis] = hand_xy[axis] + shift[axis]
                via0[2] = via1[2] = z
                path.extend([_pose(via0, current['quaternion']), _pose(via1, q)])
            path.append(entry)
            if surface_normal is not None:
                path.append(_pose(free, q))
            if not _clears_other(current, [*path, contact], tool, params, other):
                continue
            # The approach must be safe without an active working tip. A
            # second full-path preview qualifies contact and its potential effects.
            dry = dict(slot=hand.side, operation='move', path=path, duration=0.7)
            work = {**dry, 'operation': operation, 'path': [*path, contact], 'duration': 1.2}
            if parameters:
                work['parameters'] = parameters
            dry_result = _preview(hand, dry)
            if dry_result['status'] != 'accepted':
                refusals.add(str(dry_result.get('reason')))
                continue
            outcome = _preview(hand, work)
            if outcome['status'] == 'accepted' and (
                expected_event is None
                or any(
                    item.get('kind') == expected_event
                    and (
                        item.get('objectId') == expected_object
                        if expected_object
                        else item.get('objectKind') == 'retaining-stay'
                    )
                    for item in outcome['events']
                )
            ):
                return path, contact
            refusals.add(str(outcome.get('reason') or 'intended effect absent'))
    raise ValueError('No native-accepted bimanual route to this site: ' + ', '.join(sorted(refusals)))


def retained_site(hand, tool, label, target, *, orientation, parameters=None, retrieve=False):
    """Place or retrieve the retaining stay with both hands explicitly present."""
    objects = hand.live.view['operativeExposureV1']['objects']
    held = next(
        (
            name
            for name, body in hand.live.view['operativeExposureV1']['bodies'].items()
            if body.get('heldBy') and body['heldBy'] != hand.side
        ),
        None,
    )
    if held not in ('scalp', 'dura'):
        raise ValueError('A retaining stay needs the other hand to support the registered sheet')
    stay = next(
        (
            row
            for row in objects
            if row['kind'] == 'retaining-stay' and row['body'] == held and row['status'] == 'placed'
        ),
        None,
    )
    if retrieve:
        if not stay:
            raise ValueError('The supported sheet has no placed retaining stay')
        target = stay['positionMm']
    if hand.slot['instrument'] != tool:
        hand.live.step(
            'Select ' + tool, dict(slot=hand.side, instrument=tool, operation='exchange', duration=0.1)
        )
    hand.action(
        'Mount ' + tool + ' at the parked hand',
        'place',
        at=hand.slot['position'],
        orientation=hand.slot['quaternion'],
        parameters=parameters,
    )
    origin = _pose(hand.slot['position'], hand.slot['quaternion'])
    op = 'grasp' if retrieve else 'apply'
    expected = 'exposure_object_grasped' if retrieve else 'exposure_object_placed'
    path, contact = external_route(
        hand,
        target,
        orientation,
        operation=op,
        parameters=parameters,
        expected_event=expected,
        expected_object=stay['id'] if retrieve else None,
        surface_normal=stay['normal'] if retrieve else None,
        object_radius=stay['radiusMm'] if retrieve else None,
    )
    hand.action('Approach beside the supporting hand', 'move', 0.7, path=path)
    result = hand.action(label, op, 0.35, at=contact['position'], orientation=contact['quaternion'])
    if not any(
        e.get('kind') == expected
        and e.get('body') == held
        and (e.get('objectId') == stay['id'] if retrieve else e.get('objectKind') == 'retaining-stay')
        for e in result['events']
    ):
        raise ValueError('Accepted contact did not change the intended sheet retention')

    if retrieve:
        normal = np.asarray(stay['normal'])
        free = np.asarray(contact['position']) + 3.5 * normal
        hand.action(
            'Lift the stay away from the retained surface',
            'move',
            0.35,
            at=free.tolist(),
            orientation=contact['quaternion'],
        )
        outside = free.copy()
        outside[2] = max(hand.clearance, free[2] + 5.0)
        hand.action(
            'Carry the stay to the exterior count',
            'move',
            0.8,
            at=outside.tolist(),
            orientation=contact['quaternion'],
        )
        hand.action('Release retrieved stay at the exterior', 'release')
        ledger = hand.live.view['operativeExposureV1']['objects']
        if not any(o['id'] == stay['id'] and o['status'] == 'retrieved' for o in ledger):
            raise ValueError('The carried stay did not reach the exterior count')
    reverse = [*reversed(path), origin]
    # A resumed partial approach can start between the far parking site and
    # the incision. Clear the opposing holder's future closure arc as well.
    park = _pose(hand.parking, [1.0, 0.0, 0.0, 0.0])
    if not np.allclose(origin['position'], park['position']) or not np.allclose(
        origin['quaternion'], park['quaternion']
    ):
        reverse.extend([_pose(hand.parking, origin['quaternion']), park])
    hand.action('Withdraw ' + tool + ' along the exterior approach', 'withdraw', 1.0, path=reverse)
