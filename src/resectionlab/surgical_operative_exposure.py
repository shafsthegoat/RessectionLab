"""Generated cranial exposure. All edits are source-bound and driven by physical contact."""

from copy import deepcopy
import math

import numpy as np
from scipy.spatial.transform import Rotation

from .surgical_exposure_geometry import (
    BODIES,
    PROPERTY,
    SCHEMA,
    centers,
    exposure_declaration,
    geometry_key,
    grip_trajectory,
    moved,
    rest,
    solve_angle,
    source_position,
)
from .surgical_motion import envelopes, pair_conflict, swept_envelopes


def initialize(engine):
    config = engine.scene.properties[PROPERTY]
    if config.get('schema') != SCHEMA or set(config.get('bodies', {})) != set(BODIES):
        raise ValueError('OPERATIVE_EXPOSURE_INVALID_DECLARATION')
    seen = set()
    bodies = {}
    for name in BODIES:
        declaration = config['bodies'][name]
        ids = declaration['cells']
        owned = set(ids)
        if (
            len(owned) != len(ids)
            or seen & owned
            or not owned
            or min(owned) < 0
            or max(owned) >= engine.scene.labels.size
        ):
            raise ValueError('OPERATIVE_EXPOSURE_INVALID_OWNERSHIP')
        seen.update(owned)
        if (
            not np.all(engine.scene.labels.ravel()[ids])
            or declaration['gripCell'] not in owned
            or not set(declaration['anchorCells']) <= owned
        ):
            raise ValueError('OPERATIVE_EXPOSURE_LANDMARK_WITHOUT_MATERIAL')
        expected = {'scalp': 'soft_tissue', 'bone': 'bone', 'dura': 'membrane'}[name]
        for label in np.unique(engine.scene.labels.ravel()[ids]):
            if engine.by_label[int(label)]['material'] != expected:
                raise ValueError('OPERATIVE_EXPOSURE_INCOMPATIBLE_MATERIAL')
        if (
            len(set(declaration['anchorCells'])) < 2
            or not declaration['seamEdges']
            or (name != 'bone' and not declaration['hingeEdges'])
        ):
            raise ValueError('OPERATIVE_EXPOSURE_INCOMPLETE_BOUNDARY')
        if name == 'bone' and (
            not {j for _, j in declaration['seamEdges']} <= set(declaration['kerfCells'])
            or owned.intersection(declaration['kerfCells'])
            or not declaration['portCells']
            or owned.intersection(declaration['portCells'])
        ):
            raise ValueError('OPERATIVE_EXPOSURE_INVALID_BONE_CUT')
        if name == 'bone':
            from scipy.spatial import cKDTree

            ring = {j for _, j in declaration['seamEdges']}
            corridor = set(declaration.get('bladeCorridorCells', declaration['kerfCells']))
            stock = corridor | set(declaration['portCells'])
            if (
                not ring <= corridor
                or stock & owned
                or min(stock) < 0
                or max(stock) >= engine.scene.labels.size
            ):
                raise ValueError('OPERATIVE_EXPOSURE_INVALID_BONE_CORRIDOR')
            for label in np.unique(engine.scene.labels.ravel()[sorted(stock)]):
                if int(label) not in engine.by_label or engine.by_label[int(label)]['material'] != 'bone':
                    raise ValueError('OPERATIVE_EXPOSURE_INVALID_BONE_CORRIDOR')
            distance = cKDTree(engine.scene.points(sorted(ring))).query(
                engine.scene.points(sorted(corridor))
            )[0]
            if distance.max() > 2.55:
                raise ValueError('OPERATIVE_EXPOSURE_OFF_PATH_BONE_CORRIDOR')
        for left, right in declaration['seamEdges'] + declaration['hingeEdges']:
            if (
                left not in owned
                or right in owned
                or not 0 <= right < engine.scene.labels.size
                or not engine.scene.labels.flat[right]
            ):
                raise ValueError('OPERATIVE_EXPOSURE_INVALID_ATTACHMENT')
            if (
                sum(
                    abs(a - b)
                    for a, b in zip(
                        np.unravel_index(left, engine.scene.shape),
                        np.unravel_index(right, engine.scene.shape),
                    )
                )
                != 1
            ):
                raise ValueError('OPERATIVE_EXPOSURE_NONLOCAL_ATTACHMENT')
        bodies[name] = {
            'status': 'attached',
            'angleRad': 0.0,
            'translationMm': [0.0, 0.0, 0.0],
            'quaternion': [0.0, 0.0, 0.0, 1.0],
            'heldBy': None,
            'closedAnchors': [],
            'cut': [],
            'grip': None,
            'momentNmm': 0.0,
        }
    engine.state[PROPERTY] = {'bodies': bodies, 'objects': []}


def _body(engine, name):
    return engine.state[PROPERTY]['bodies'][name]


def _source(engine, name):
    return engine.scene.properties[PROPERTY]['bodies'][name]


def _held(engine, key):
    return next((name for name in BODIES if _body(engine, name)['heldBy'] == key), None)


def holding(engine, key):
    return _held(engine, key) is not None or _portable(engine, key) is not None


def held_cells(engine, key):
    name = _held(engine, key)
    return set(_body(engine, name)['grip']['jawNeighborhood']) if name else set()


def _shape(engine, name):
    source = _source(engine, name)
    cache = getattr(engine.scene, '_operative_contact_geometry', None)
    if cache is None:
        cache = engine.scene._operative_contact_geometry = {}
    if name not in cache:
        edges = np.asarray(source['seamEdges'], dtype=int)
        lookup = {}
        for k, (i, j) in enumerate(edges):
            lookup.setdefault(int(i), []).append(k)
            lookup.setdefault(int(j), []).append(k)
        cache[name] = (set(source['cells']), lookup)
    return cache[name]


def permits_contact(engine, key, op, source_ids):
    tool = engine.state['slots'][key]['instrument']
    names = {
        'membrane_forceps': ('scalp', 'dura'),
        'raney_clip_applier': ('scalp', 'dura'),
        'needle_driver': ('scalp', 'dura'),
        'fixation_driver': ('bone',),
        'bone_flap_holder': ('bone',),
        'scalpel': ('scalp',),
        'dural_scissors': ('dura',),
    }
    result = set()
    if tool == 'scalpel' and op in ('apply', 'move'):
        bound = engine.scene.properties.get('gliomaMaterialsV1', {}).get('scalpSeamCell')
        if bound is not None:
            cfg = _source(engine, 'scalp')
            seam = _shape(engine, 'scalp')[1]
            local_edges = (cfg['seamEdges'][i] for i in seam.get(bound, ()))
            if any(source_ids.intersection(edge) for edge in local_edges):
                graph = engine.scene.properties.get('vascularGraphV1', {})
                face = next((e['sourceFace'] for e in graph.get('edges', []) if e['id'] == 'scalp'), ())
                result.update(source_ids.intersection(face))
    if tool == 'raney_clip_applier':
        stump = engine.state.get('vascularGraphV1', {}).get('ports', {}).get('vascular:scalp:0', {})
        if stump.get('cell') in source_ids:
            result.add(stump['cell'])
    for name in names.get(tool, ()):
        owned, seam = _shape(engine, name)
        result.update(i for i in source_ids if i in owned or i in seam)
    return result


def _available(engine, name, kind):
    return next(
        (
            o
            for o in engine.state[PROPERTY]['objects']
            if o['body'] == name and o['kind'] == kind and o['status'] not in ('retrieved', 'carried')
        ),
        None,
    )


def _outstanding(engine, kind):
    return any(o['kind'] == kind and o['status'] != 'retrieved' for o in engine.state[PROPERTY]['objects'])


def _access(engine):
    return (
        _body(engine, 'scalp')['angleRad'] >= math.pi / 2
        and _body(engine, 'dura')['angleRad'] >= math.pi / 2
        and _body(engine, 'bone')['status'] == 'parked'
    )


def _event(kind, name, **data):
    return {'kind': kind, 'body': name, **data}


def _incise(engine, name, contacts, key, p, q):
    body = _body(engine, name)
    if body['status'] != 'attached' or not contacts:
        return []
    if name == 'dura' and _body(engine, 'bone')['status'] != 'parked':
        raise ValueError('OPERATIVE_DURA_COVERED')
    source = _source(engine, name)
    _, lookup = _shape(engine, name)
    new = {index for cell in contacts for index in lookup.get(cell, ())} - set(body['cut'])
    if new:
        from .surgical_tool_profiles import profile_segments

        params = engine.parameters(engine.state['slots'][key])
        if name == 'dura' and params['jawOpeningMm'] > 0.5:
            return []
        candidates = sorted(new)
        endpoints = np.array([source['seamEdges'][i] for i in candidates])
        orientation = Rotation.from_quat(q)
        coordinates = (engine.scene.points(endpoints.ravel()) - p) @ orientation.as_matrix()
        pairs = coordinates.reshape(-1, 2, 3)
        side = pairs[:, :, 1]
        spans = (side.min(1) <= 1e-8) & (side.max(1) >= -1e-8) & (np.ptp(side, axis=1) > 1e-8)
        ends = pairs[spans]
        if not len(ends):
            return []
        t = -ends[:, 0, 1] / (ends[:, 1, 1] - ends[:, 0, 1])
        crossing = ends[:, 0] + t[:, None] * (ends[:, 1] - ends[:, 0])
        cuts = np.zeros(len(ends), dtype=bool)
        for a, b, radius, role, form in profile_segments(
            params['operativeToolProfileV1'], params['jawOpeningMm']
        ):
            if role == 'working' and form != 'blunt-toe':
                cuts |= _distance(crossing, np.asarray(a), np.asarray(b)) <= radius
        new = set(np.array(candidates)[spans][cuts].tolist())
    if not new:
        return []
    from .surgical_effects import _record_edges

    _record_edges(engine, 'cutEdges', [tuple(sorted(source['seamEdges'][i])) for i in sorted(new)])
    body['cut'] = sorted(set(body['cut']) | new)
    if len(body['cut']) == len(source['seamEdges']):
        body['status'] = 'incised'
    return [
        _event(
            'exposure_seam_cut',
            name,
            cutEdges=len(new),
            remainingEdges=len(source['seamEdges']) - len(body['cut']),
        )
    ]


def _cut_bone(engine, tool, contacts, dt, params):
    if not contacts:
        return []
    source = _source(engine, 'bone')
    if _body(engine, 'scalp')['angleRad'] < math.pi / 2:
        raise ValueError('OPERATIVE_SKULL_COVERED')
    port = set(source['portCells'])
    kerf = set(source.get('bladeCorridorCells', source['kerfCells']))
    removed = set(engine.state['removed'])
    if tool == 'cranial_drill':
        work = contacts & port
    else:
        if min(params['rpm'], params['forceN'], params['duty']) <= 0:
            raise ValueError('OPERATIVE_KERF_DRIVE_REQUIRED')
        # Only start in the true perforation, then extend a locally connected trough.
        columns = getattr(engine.scene, '_operative_port_columns', None)
        if columns is None:
            axis = np.asarray(source['outward'])
            port_ids = np.asarray(sorted(port))
            xyz = engine.scene.points(port_ids)
            lateral = np.round(xyz - np.outer(xyz @ axis, axis), 6)
            columns = [
                set(port_ids[np.all(lateral == point, axis=1)].tolist())
                for point in np.unique(lateral, axis=0)
            ]
            engine.scene._operative_port_columns = columns
        if not any(column <= removed for column in columns):
            raise ValueError('OPERATIVE_PILOT_REQUIRED')
        work = contacts & kerf
        if work:
            cut = removed & (port | kerf)
            candidates = sorted(work)
            ijk = np.array(np.unravel_index(candidates, engine.scene.shape)).T
            prior = np.array(np.unravel_index(list(cut), engine.scene.shape)).T
            near = np.max(abs(ijk[:, None, :] - prior[None, :, :]), axis=2).min(axis=1) <= 1
            work = set(np.array(candidates)[near].tolist())
    if not work:
        if contacts & (set(source['cells']) | kerf | port):
            return []
        raise ValueError('OPERATIVE_BONE_OUTSIDE_AUTHORED_CUT')
    from .surgical_drilling_v2 import apply

    events, _, _ = apply(engine, sorted(work), dt, params, cool=False)
    body = _body(engine, 'bone')
    now = set(engine.state['removed'])
    if now != removed:
        body['cut'] = [i for i, (_, right) in enumerate(source['seamEdges']) if right in now]
        if body['status'] in ('attached', 'free') and len(body['cut']) == len(source['seamEdges']):
            body['status'] = 'free'
            events.append(_event('exposure_bone_freed', 'bone', kerfCells=len(kerf)))
    return events


def _grasp(engine, key, name, p, q, contacts):
    cfg = _source(engine, name)
    body = _body(engine, name)
    grip = source_position(engine, name, cfg['gripCell'])
    ids, _, lookup, *_ = rest(engine, name)
    xyz = centers(engine, name)
    local = [i for i in contacts if i in lookup and np.linalg.norm(xyz[lookup[i]] - grip) <= 2.5]
    if not local:
        return []
    if body['heldBy']:
        raise ValueError('OPERATIVE_BODY_ALREADY_HELD')
    if set(cfg['cells']) & getattr(engine, '_protected_ids', set()):
        raise ValueError('PROTECTED_STRUCTURE: grasp displaces protected exposure body')
    if body['closedAnchors']:
        raise ValueError('OPERATIVE_BODY_HAS_CLOSURE_ANCHORS')
    if body['status'] not in ('incised', 'supported', 'free', 'parked', 'seated'):
        raise ValueError('OPERATIVE_BODY_STILL_ATTACHED')
    if name == 'scalp' and body['status'] == 'supported' and _body(engine, 'bone')['status'] != 'fixed':
        raise ValueError('OPERATIVE_REPLACE_BONE_BEFORE_SCALP')
    if name == 'bone' and body['status'] == 'parked' and _body(engine, 'dura')['status'] != 'closed':
        raise ValueError('OPERATIVE_CLOSE_DURA_BEFORE_REPLACEMENT')
    if name == 'dura' and body['status'] == 'supported' and _outstanding(engine, 'cottonoid'):
        raise ValueError('OPERATIVE_REMOVE_COUNTED_COTTONOID')
    # A fixed jaw keeps contacting a small changing neighborhood as the sheet
    # rolls across the voxel raster. Never exempt a shaft, the rest of the flap,
    # retained neighbors or any source beyond the physical distal jaw length.
    neighborhood = ids[np.linalg.norm(xyz - grip, axis=1) <= 3.25].tolist()
    body['grip'] = {
        'cells': sorted(local),
        'jawNeighborhood': neighborhood,
        'toolPosition': p.tolist(),
        'toolQuaternion': q.tolist(),
        'startQuaternion': body['quaternion'][:],
        'startTranslationMm': body['translationMm'][:],
        'offset': (p - grip).tolist(),
    }
    body['heldBy'] = key
    body['status'] = 'held'
    return [_event('exposure_body_grasped', name, sourceCells=sorted(local), slot=key)]


def _distance(xyz, a, b):
    d = np.asarray(b) - a
    length = float(d @ d)
    t = np.clip((xyz - a) @ d / length, 0, 1) if length > 1e-12 else np.zeros(len(xyz))
    return np.linalg.norm(xyz - np.asarray(a) - t[:, None] * d, axis=1)


def _surface_normal(engine, name, cell, state=None):
    body = state if state is not None else _body(engine, name)
    _, _, lookup, axis, normal, forward, _, _, s, _, _, hinge_band = rest(engine, name)
    if name == 'bone':
        return Rotation.from_quat(body['quaternion']).apply(normal)
    length = _source(engine, name)['bendLengthMm']
    theta = body['angleRad'] * np.clip((s[lookup[cell]] - hinge_band) / length, 0, 1)
    return np.cos(theta) * normal - np.sin(theta) * forward


def _object_position(engine, o, state=None):
    if o['kind'] == 'raney-clip' and o['status'] == 'placed':
        from .surgical_operative_clip import rest as clip_rest

        return clip_rest(engine, o)[0]
    if o.get('body') == 'field' and 'sourceCell' in o and o['status'] not in ('carried', 'retrieved'):
        cell = o['sourceCell']
        shift = engine.state.get('deformation', {}).get(str(cell), [0.0, 0.0, 0.0])
        return engine.scene.points([cell])[0] + shift
    if (
        'sourceCell' in o
        and o['status'] not in ('carried', 'retrieved')
        and o['sourceCell'] in _shape(engine, o['body'])[0]
    ):
        point = source_position(engine, o['body'], o['sourceCell'], state)
        if 'surfaceOffsetMm' in o:
            point = point + _surface_normal(engine, o['body'], o['sourceCell'], state) * o['surfaceOffsetMm']
        return point
    return np.asarray(o['positionMm'])


def _object_envelope(engine, obj):
    point = _object_position(engine, obj)
    if obj.get('endpointsMm'):
        a, b = obj['endpointsMm']
        return np.asarray(a), np.asarray(b), obj['radiusMm'], 'fixture'
    return point, point, obj['radiusMm'], 'fixture'


def object_envelopes(engine, obj):
    if obj['kind'] == 'raney-clip':
        from .surgical_operative_clip import segments

        return segments(engine, obj)
    return [_object_envelope(engine, obj)]


def _clear_body(engine, name, key, previous, proposed, p, q):
    a = centers(engine, name, previous)
    b = centers(engine, name, proposed)
    step = float(np.min(engine.scene.spacing)) * 0.32
    slot = engine.state['slots'][key]
    start_tip = np.asarray(slot['position'])
    tool_rotation = Rotation.from_quat(q) * Rotation.from_quat(slot['quaternion']).inv()
    if name == 'bone':
        extent = np.linalg.norm(a - start_tip, axis=1).max()
        travel = np.linalg.norm(p - start_tip) + extent * tool_rotation.magnitude()
    else:
        extent = np.linalg.norm(a - rest(engine, name)[6], axis=1).max()
        travel = extent * abs(proposed['angleRad'] - previous['angleRad'])
    count = max(1, math.ceil(travel / step))
    owned, _ = _shape(engine, name)
    if getattr(engine, '_protected_ids', None):
        from .surgical_rl_protection import cell_radius

        clearance_radius = cell_radius(engine.scene.affine)
        protected = engine._positions()
    previous_sample = a
    occ = engine.occupancy()
    other_tools = []
    own_working = []
    neighborhood = held_cells(engine, key)
    beyond_grip = ~np.isin(rest(engine, name)[0], list(neighborhood))
    for hand, slot in engine.state['slots'].items():
        tool = engine.tools[slot['instrument']]
        parts = (
            swept_envelopes(
                tool, engine._sweep_parameters(hand, slot), slot, {'position': p, 'quaternion': q}
            )
            if hand == key
            else envelopes(tool, engine.parameters(slot), slot['position'], slot['quaternion'])
        )
        for x, y, r, part in parts:
            if hand != key or part == 'shaft':
                other_tools.append((x, y, r, hand))
            else:
                own_working.append((x, y, r))
    objects = [
        part
        for o in engine.state[PROPERTY]['objects']
        if o['status'] != 'retrieved'
        and (o['status'] == 'carried' or o['body'] != name or o.get('sourceCell') not in owned)
        for part in object_envelopes(engine, o)
    ]
    bone_rest = None
    if hasattr(engine, '_bone_rest'):
        from . import surgical_bone_rest as bone_rest
    for t in range(1, count + 1):
        fraction = t / count
        if t == count:
            pos = b
        elif name == 'bone':
            partial = Rotation.from_rotvec(tool_rotation.as_rotvec() * fraction)
            pos = start_tip + fraction * (p - start_tip) + partial.apply(a - start_tip)
        else:
            angle = previous['angleRad'] + fraction * (proposed['angleRad'] - previous['angleRad'])
            pos = centers(engine, name, dict(previous, angleRad=angle))
        if getattr(engine, '_protected_ids', None):
            for start, end in zip(previous_sample, pos):
                if len(engine.protection.capsule_hits(start, end, clearance_radius, positions=protected)):
                    raise ValueError('PROTECTED_STRUCTURE: carried exposure body enters declared clearance')
        previous_sample = pos
        if bone_rest is not None:
            bone_rest.check_body(engine, name, pos)
        ijk = np.rint(pos @ engine.scene.inverse[:3, :3].T + engine.scene.inverse[:3, 3]).astype(int)
        if np.any(ijk < 0) or np.any(ijk >= engine.scene.shape):
            raise ValueError('OPERATIVE_CARRIED_BODY_OUTSIDE_GRID')
        targets = np.ravel_multi_index(ijk.T, engine.scene.shape)
        blocked = set(occ.material_ids(targets)) - owned
        if blocked:
            raise ValueError('OPERATIVE_CARRIED_BODY_COLLISION')
        for x, y, r in own_working:
            if np.any(_distance(pos[beyond_grip], x, y) < r + 0.3):
                raise ValueError('OPERATIVE_CARRIED_BODY_OTHER_JAW_CONTACT')
        for x, y, r, hand in other_tools:
            if np.any(_distance(pos, x, y) < r + 0.3):
                raise ValueError('OPERATIVE_CARRIED_BODY_TOOL_COLLISION:' + hand)
        for x, y, r, _ in objects:
            if np.any(_distance(pos, x, y) < r + 0.25):
                raise ValueError('OPERATIVE_CARRIED_BODY_OBJECT_COLLISION')


def _carry(engine, key, name, p, q, dt):
    body = _body(engine, name)
    grip = body['grip']
    cfg = _source(engine, name)
    oldpose = engine.state['slots'][key]
    if np.allclose(p, oldpose['position'], rtol=0, atol=1e-10) and np.allclose(
        q, oldpose['quaternion'], rtol=0, atol=1e-10
    ):
        if name != 'bone':
            body['momentNmm'] = (
                engine.scene.properties[PROPERTY]['materials'][name]['bendingStiffnessNmmPerRad']
                * body['angleRad']
            )
        return []
    proposed = body.copy()
    if name == 'bone':
        relative = Rotation.from_quat(q) * Rotation.from_quat(grip['toolQuaternion']).inv()
        old_center = np.asarray(cfg['restCenterMm']) + grip['startTranslationMm']
        new_center = p + relative.apply(old_center - grip['toolPosition'])
        proposed['translationMm'] = (new_center - cfg['restCenterMm']).tolist()
        proposed['quaternion'] = (relative * Rotation.from_quat(grip['startQuaternion'])).as_quat().tolist()
    else:
        angle = solve_angle(engine, name, p - grip['offset'])
        if abs(angle - body['angleRad']) < 1e-9:
            return []
        if _available(engine, name, 'retaining-stay') and abs(angle - body['angleRad']) > 1e-4:
            raise ValueError('OPERATIVE_REMOVE_STAY_WHILE_HELD')
        props = engine.scene.properties[PROPERTY]['materials'][name]
        moment = props['bendingStiffnessNmmPerRad'] * angle + props['torsionalDampingNmmsPerRad'] * (
            angle - body['angleRad']
        ) / max(dt, 1e-8)
        force = float(engine.parameters(oldpose)['forceN'])
        lever = max(
            1.0, np.linalg.norm(source_position(engine, name, cfg['gripCell']) - cfg['hingeOriginMm'])
        )
        if abs(moment) > force * lever:
            raise ValueError('OPERATIVE_REFLECTION_TRACTION_LIMIT')
        proposed.update(angleRad=angle, momentNmm=moment)
    witness = centers(engine, name, proposed, [rest(engine, name)[2][i] for i in grip['cells']])
    from .surgical_rl_protection import cell_radius

    tolerance = cell_radius(engine.scene.affine)
    working = envelopes(engine.tools[oldpose['instrument']], engine.parameters(oldpose), p, q)
    if not any(
        np.any(_distance(witness, a, b) <= r + tolerance) for a, b, r, part in working if part != 'shaft'
    ):
        raise ValueError('OPERATIVE_REGISTERED_GRIP_CONTACT_LOST')
    _clear_body(engine, name, key, body, proposed, p, q)
    body.update(proposed)
    return []


def release(engine, key):
    obj = _portable(engine, key)
    if obj:
        from .surgical_exterior import plane

        point, normal = plane(engine.scene)
        if any(
            min((a - point) @ normal, (b - point) @ normal) < radius
            for a, b, radius, _ in object_envelopes(engine, obj)
        ):
            raise ValueError('OPERATIVE_RETRIEVAL_REQUIRES_EXTERIOR')
        obj.update(status='retrieved', heldBy=None)
        obj.pop('offset', None)
        return [_event('exposure_object_retrieved', obj['body'], objectId=obj['id'])]
    name = _held(engine, key)
    if not name:
        return None
    body = _body(engine, name)
    cfg = _source(engine, name)
    if name == 'bone':
        center = np.asarray(cfg['restCenterMm']) + body['translationMm']
        if (
            np.linalg.norm(center - cfg['parkCenterMm']) <= 1.2
            and Rotation.from_quat(body['quaternion']).magnitude() <= 0.12
        ):
            if hasattr(engine, '_bone_rest'):
                from .surgical_bone_rest import supported

                if not supported(engine):
                    raise ValueError('OPERATIVE_BONE_NEEDS_PHYSICAL_REST_CONTACT')
            status = 'parked'
        elif (
            np.linalg.norm(body['translationMm']) <= 0.6
            and Rotation.from_quat(body['quaternion']).magnitude() <= 0.035
        ):
            status = 'seated'
        else:
            raise ValueError('OPERATIVE_BONE_NEEDS_SUPPORTED_SEAT_OR_PARK')
    elif body['angleRad'] <= 0.035:
        status = 'seated'
    elif stay := _available(engine, name, 'retaining-stay'):
        if np.linalg.norm(_object_position(engine, stay) - stay['fixedPositionMm']) > 0.1:
            raise ValueError('OPERATIVE_RETAINING_STAY_LOST_CONTACT')
        stiffness = engine.scene.properties[PROPERTY]['materials'][name]['bendingStiffnessNmmPerRad']
        body['momentNmm'] = stiffness * body['angleRad']
        stay['reactionMomentNmm'] = -body['momentNmm']
        status = 'supported'
    else:
        raise ValueError('OPERATIVE_REFLECTED_SHEET_REQUIRES_STAY')
    body.update(status=status, heldBy=None, grip=None)
    return [_event('exposure_body_released', name, status=status)]


def _portable(engine, key):
    return next((o for o in engine.state[PROPERTY]['objects'] if o.get('heldBy') == key), None)


def _place_object(engine, name, kind, position, *, cell=None, intended=False):
    objects = engine.state[PROPERTY]['objects']
    item = {
        'id': f'{kind}-{len(objects)+1}',
        'body': name,
        'kind': kind,
        'positionMm': np.asarray(position).tolist(),
        'status': 'intended-retained' if intended else 'placed',
        'heldBy': None,
        'radiusMm': {
            'cottonoid': 1.45,
            'hemostat': 0.95,
            'raney-clip': 0.58,
            'retaining-stay': 0.9,
            'bone-anchor': 0.20,
            'suture': 0.45,
        }[kind],
    }
    if cell is not None:
        item['sourceCell'] = int(cell)
        if kind in ('retaining-stay', 'suture', 'bone-anchor'):
            normal = np.asarray(_source(engine, name)['outward'])
            item['surfaceOffsetMm'] = 0.5 * np.abs(engine.scene.affine[:3, :3].T @ normal).sum()
            item['positionMm'] = _object_position(engine, item).tolist()
    if kind == 'retaining-stay':
        item['fixedPositionMm'] = item['positionMm'][:]
        item['reactionMomentNmm'] = 0.0
    objects.append(item)
    return [_event('exposure_object_placed', name, objectId=item['id'], objectKind=kind)]


def _pick(engine, key, tool, p, q):
    allowed = ('cottonoid',) if tool == 'dressing_forceps' else ('raney-clip', 'retaining-stay')
    slot = engine.state['slots'][key]
    working = [
        (a, b, r)
        for a, b, r, part in envelopes(engine.tools[tool], engine.parameters(slot), p, q)
        if part != 'shaft'
    ]
    for obj in engine.state[PROPERTY]['objects']:
        if obj['kind'] not in allowed or obj['status'] != 'placed':
            continue
        xyz = _object_position(engine, obj)
        if not any(_distance(xyz[None, :], a, b)[0] <= r + obj['radiusMm'] for a, b, r in working):
            continue
        if obj['kind'] == 'retaining-stay' and _body(engine, obj['body'])['heldBy'] is None:
            raise ValueError('OPERATIVE_REGRASP_BEFORE_STAY_REMOVAL')
        if obj['kind'] == 'raney-clip':
            if _body(engine, 'scalp')['angleRad'] < math.pi / 2:
                raise ValueError('OPERATIVE_SCALP_CLIP_NEEDS_OPEN_MARGIN')
            obj['pickupQuaternion'] = q.tolist()
            obj['carriedRotation'] = [0.0, 0.0, 0.0, 1.0]
        if obj['kind'] == 'retaining-stay':
            obj['reactionMomentNmm'] = 0.0
        obj.update(
            positionMm=xyz.tolist(),
            offset=Rotation.from_quat(q).inv().apply(xyz - p).tolist(),
            heldBy=key,
            status='carried',
        )
        return [_event('exposure_object_grasped', obj['body'], objectId=obj['id'])]
    return []


def _anchors(engine, tool, name, key, p, contacts):
    body = _body(engine, name)
    cfg = _source(engine, name)
    if tool == 'raney_clip_applier':
        if body['angleRad'] >= math.pi / 2 and body['heldBy'] and body['heldBy'] != key:
            cell = cfg['anchorCells'][0]
            if cell in contacts and not _available(engine, name, 'retaining-stay'):
                return _place_object(
                    engine, name, 'retaining-stay', source_position(engine, name, cell), cell=cell
                )
        elif (
            name == 'scalp'
            and body['status'] in ('attached', 'incised')
            and body['cut']
            and not _available(engine, name, 'raney-clip')
        ):
            stump = engine.state.get('vascularGraphV1', {}).get('ports', {}).get('vascular:scalp:0')
            if stump and stump['cell'] in contacts:
                cell = stump['cell']
                return _place_object(engine, name, 'raney-clip', engine.scene.points([cell])[0], cell=cell)
        return []
    if body['status'] != 'seated':
        return []
    if name == 'scalp' and _outstanding(engine, 'raney-clip'):
        raise ValueError('OPERATIVE_REMOVE_SCALP_CLIP_BEFORE_CLOSURE')
    for index, cell in enumerate(cfg['anchorCells']):
        if index in body['closedAnchors'] or cell not in contacts:
            continue
        # A closure fixture must bridge the actual incised mobile margin and
        # stationary material. At a bone kerf the first neighbor was eroded.
        seam = next((edge for edge in cfg['seamEdges'] if edge[0] == cell), None)
        if not seam:
            raise ValueError('OPERATIVE_CLOSURE_REQUIRES_SEAM')
        original, stationary = engine.scene.points(seam)
        direction = stationary - original
        if name == 'bone':
            for distance in (2.0, 3.0, 4.0):
                ijk = np.rint(
                    engine.scene.inverse[:3, :3] @ (original + direction * distance)
                    + engine.scene.inverse[:3, 3]
                ).astype(int)
                if np.any(ijk < 0) or np.any(ijk >= engine.scene.shape):
                    raise ValueError('OPERATIVE_FIXATION_REQUIRES_RETAINED_BONE')
                candidate = int(np.ravel_multi_index(ijk, engine.scene.shape))
                if (
                    candidate not in _shape(engine, name)[0]
                    and candidate not in engine.state['removed']
                    and engine.by_label.get(int(engine.scene.labels.flat[candidate]), {}).get('material')
                    == 'bone'
                ):
                    stationary = engine.scene.points([candidate])[0]
                    break
            else:
                raise ValueError('OPERATIVE_FIXATION_REQUIRES_RETAINED_BONE')
        mobile = source_position(engine, name, cell)
        if np.linalg.norm(mobile - original) > 0.8:
            raise ValueError('OPERATIVE_CLOSURE_MARGIN_GAP')
        body['closedAnchors'].append(index)
        if len(body['closedAnchors']) == len(cfg['anchorCells']):
            body['status'] = 'fixed' if name == 'bone' else 'closed'
        events = _place_object(
            engine, name, 'bone-anchor' if name == 'bone' else 'suture', mobile, cell=cell, intended=True
        )
        obj = engine.state[PROPERTY]['objects'][-1]
        offset = np.asarray(cfg['outward']) * obj['surfaceOffsetMm']
        obj['endpointsMm'] = [_object_position(engine, obj).tolist(), (stationary + offset).tolist()]
        obj['marginErrorMm'] = float(np.linalg.norm(mobile - original))
        return events
    return []


def _dress(engine, p, contacts):
    if not contacts:
        return []
    if not _access(engine):
        raise ValueError('OPERATIVE_INTRACRANIAL_ACCESS_CLOSED')
    cfg = engine.scene.properties[PROPERTY]
    ports = engine.state.get('vascularGraphV1', {}).get('ports', {})
    outlet = ports.get('vascular:biopsy-bed:0')
    if outlet and outlet['cell'] in contacts and not _available(engine, 'field', 'hemostat'):
        cell = outlet['cell']
        position = engine.scene.points([cell])[0] + engine.state.get('deformation', {}).get(
            str(cell), [0.0, 0.0, 0.0]
        )
        return _place_object(engine, 'field', 'hemostat', position, cell=cell, intended=True)
    for label, kind in [('cottonoidSiteMm', 'cottonoid'), ('hemostatSiteMm', 'hemostat')]:
        site = np.asarray(cfg[label])
        if np.linalg.norm(p - site) > cfg['siteRadiusMm'] or _available(engine, 'field', kind):
            continue
        ids = np.asarray(sorted(contacts), dtype=int)
        xyz = engine.scene.points(ids)
        if np.linalg.norm(xyz - site, axis=1).min() > cfg['siteRadiusMm']:
            continue
        return _place_object(engine, 'field', kind, site, intended=kind == 'hemostat')
    return []


def _tool_may_touch_object(engine, key, op, obj, p):
    tool = engine.state['slots'][key]['instrument']
    if obj.get('heldBy') == key:
        return True
    if tool == 'raney_clip_applier' and obj['kind'] in ('retaining-stay', 'raney-clip'):
        return op in ('apply', 'grasp', 'release', 'deactivate', 'inspect', 'withdraw')
    if tool == 'dressing_forceps' and obj['kind'] in ('cottonoid', 'hemostat'):
        return op in ('apply', 'grasp', 'withdraw', 'release', 'deactivate', 'inspect')
    if tool in ('needle_driver', 'fixation_driver') and obj['kind'] in ('suture', 'bone-anchor'):
        return op in ('apply', 'deactivate', 'release', 'withdraw', 'inspect')
    if tool == 'membrane_forceps' and obj['kind'] == 'raney-clip' and obj['body'] in ('scalp', 'dura'):
        name = obj['body']
        grip = source_position(engine, name, _source(engine, name)['gripCell'])
        return np.linalg.norm(p - grip) < 4.0 and (
            op in ('grasp', 'release', 'deactivate', 'withdraw', 'inspect') or _held(engine, key) == name
        )
    return False


def _guard_side_cutter_feed(engine, slot, p, q):
    """A rotating tooth can work at its edge; its core requires cleared stock.

    Contact pressure and finite energy are applied at the current reachable
    rim. On a fast stroke the operator can dwell there, then retry the feed.
    """
    oldp = np.asarray(slot['position'])
    oldq = np.asarray(slot['quaternion'])
    if (
        np.linalg.norm(oldp - p) < 1e-10
        and (Rotation.from_quat(q) * Rotation.from_quat(oldq).inv()).magnitude() < 1e-10
    ):
        return
    from .surgical_tool_profiles import profile_segments

    params = engine.parameters(slot)
    a, b = next(
        (a, b)
        for a, b, _, role, form in profile_segments(params['operativeToolProfileV1'], params['jawOpeningMm'])
        if role == 'working' and form == 'side-cutter'
    )
    endpoints = np.asarray([a, b])
    previous = Rotation.from_quat(oldq).apply(endpoints) + oldp
    current = Rotation.from_quat(q).apply(endpoints) + p
    covers = engine.touched_many(((current[0], current[1], 0.0), (previous[0], previous[1], 0.0)))
    entered = set(covers[0]) - set(covers[1])
    for cell in entered:
        if engine.by_label[int(engine.scene.labels.flat[cell])]['material'] == 'bone':
            raise ValueError('OPERATIVE_KERF_FEED_REQUIRES_CUTTING_DWELL')


def check_motion(engine, key, op, p, q):
    p = np.asarray(p)
    slot = engine.state['slots'][key]
    tool = slot['instrument']
    if tool == 'craniotome' and (op == 'apply' or op == 'move' and slot['active']):
        _guard_side_cutter_feed(engine, slot, p, q)
    if tool in (
        'cusa',
        'cortical_probe',
        'brain_retractor',
        'bipolar_forceps',
        'frazier_suction',
        'irrigation',
        'tumor_forceps',
        'micro_scissors',
        'subcortical_probe',
        'dissector',
        'scalpel',
    ):
        dura = _source(engine, 'dura')
        depth = (p - dura['restCenterMm']) @ np.asarray(dura['outward'])
        if depth < -0.5 and not _access(engine):
            raise ValueError('OPERATIVE_INTRACRANIAL_ACCESS_CLOSED')
    objs = [o for o in engine.state[PROPERTY]['objects'] if o['status'] != 'retrieved']
    if not objs:
        return
    end = {'position': p, 'quaternion': q}
    swept = swept_envelopes(engine.tools[tool], engine._sweep_parameters(key, slot), slot, end)
    for obj in objs:
        obstacles = object_envelopes(engine, obj)
        allowed = _tool_may_touch_object(engine, key, op, obj, p)
        for a, b, r, part in swept:
            if part != 'shaft' and allowed:
                continue
            if pair_conflict([(a, b, r, part)], obstacles):
                raise ValueError('OPERATIVE_PLACED_OBJECT_COLLISION:' + obj['kind'])


def interact(engine, key, op, p, q, active, dt, cells):
    tool = engine.state['slots'][key]['instrument']
    p = np.asarray(p, dtype=float)
    q = np.asarray(q, dtype=float)
    carried = _portable(engine, key)
    name = _held(engine, key)
    if (carried or name) and engine.parameters(engine.state['slots'][key])['jawOpeningMm'] > 0.5:
        raise ValueError('OPERATIVE_SUPPORT_OR_RETRIEVE_BEFORE_OPENING_JAWS')
    if carried:
        old = np.asarray(carried['positionMm'])
        new = p + Rotation.from_quat(q).apply(carried['offset'])
        clip_rotation = None
        if carried['kind'] == 'raney-clip':
            clip_rotation = (
                (Rotation.from_quat(q) * Rotation.from_quat(carried['pickupQuaternion']).inv())
                .as_quat()
                .tolist()
            )
        if (
            np.linalg.norm(new - old) > 1e-8
            or clip_rotation is not None
            and not np.allclose(clip_rotation, carried['carriedRotation'], rtol=0, atol=1e-10)
        ):
            if clip_rotation is not None:
                from .surgical_operative_clip import check_retrieval_sweep, segments

                cargo_sweep = check_retrieval_sweep(
                    engine, segments(engine, carried), segments(engine, carried, new, clip_rotation)
                )
            else:
                blocked = set(engine.touched(old, new, carried['radiusMm'])) - set(
                    engine.touched(old, old, carried['radiusMm'])
                )
                if blocked:
                    raise ValueError('OPERATIVE_CARRIED_OBJECT_COLLISION')
                cargo_sweep = [(old, new, carried['radiusMm'], 'fixture')]
            if getattr(engine, '_protected_ids', None) and any(
                len(engine.protection.capsule_hits(a, b, r, positions=engine._positions()))
                for a, b, r, _ in cargo_sweep
            ):
                raise ValueError('PROTECTED_STRUCTURE: carried material enters declared clearance')
            if hasattr(engine, '_bone_rest'):
                from .surgical_bone_rest import check_tool

                check_tool(engine, cargo_sweep)
            for hand, slot in engine.state['slots'].items():
                handle = engine.tools[slot['instrument']]
                params = engine.parameters(slot)
                if hand == key:
                    full = swept_envelopes(
                        handle, engine._sweep_parameters(hand, slot), slot, {'position': p, 'quaternion': q}
                    )
                    parts = [part for part in full if part[3] == 'shaft']
                else:
                    parts = envelopes(handle, params, slot['position'], slot['quaternion'])
                if pair_conflict(cargo_sweep, parts):
                    raise ValueError('OPERATIVE_CARRIED_OBJECT_TOOL_COLLISION')
            other = [
                part
                for o in engine.state[PROPERTY]['objects']
                if o is not carried and o['status'] != 'retrieved'
                for part in object_envelopes(engine, o)
            ]
            if pair_conflict(cargo_sweep, other):
                raise ValueError('OPERATIVE_CARRIED_OBJECT_FIXTURE_COLLISION')
        if clip_rotation is not None:
            carried['carriedRotation'] = clip_rotation
        carried['positionMm'] = new.tolist()
        return []
    if name:
        return _carry(engine, key, name, p, q, dt)
    owned_tools = (
        'cranial_drill',
        'craniotome',
        'dural_scissors',
        'membrane_forceps',
        'bone_flap_holder',
        'raney_clip_applier',
        'dressing_forceps',
        'fixation_driver',
        'needle_driver',
    )
    if not active:
        return [] if tool in owned_tools else None
    contacts = set(cells)
    intracranial = (
        'cusa',
        'tumor_forceps',
        'brain_retractor',
        'micro_scissors',
        'dissector',
        'cortical_probe',
        'subcortical_probe',
    )
    if contacts and tool in intracranial:
        layers = getattr(engine.scene, '_operative_layer_labels', None)
        if layers is None:
            source = engine.scene.properties[PROPERTY]['bodies']
            layers = {
                int(label)
                for name in BODIES
                for label in np.unique(engine.scene.labels.ravel()[source[name]['cells']])
            }
            engine.scene._operative_layer_labels = layers
        if any(int(engine.scene.labels.flat[cell]) in layers for cell in contacts):
            raise ValueError('OPERATIVE_DEEP_INSTRUMENT_ON_CRANIAL_LAYER')
        if not _access(engine):
            raise ValueError('OPERATIVE_INTRACRANIAL_ACCESS_CLOSED')
    if tool == 'scalpel':
        if contacts.intersection(_shape(engine, 'scalp')[0]) or contacts.intersection(
            _shape(engine, 'scalp')[1]
        ):
            return _incise(engine, 'scalp', contacts, key, p, q)
        return None
    if tool == 'dural_scissors':
        return _incise(engine, 'dura', contacts, key, p, q)
    if tool in ('cranial_drill', 'craniotome'):
        return _cut_bone(engine, tool, contacts, dt, engine.parameters(engine.state['slots'][key]))
    if tool in ('membrane_forceps', 'bone_flap_holder'):
        if op != 'grasp' or engine.parameters(engine.state['slots'][key])['jawOpeningMm'] > 0.5:
            return []
        for name in ('bone',) if tool == 'bone_flap_holder' else ('scalp', 'dura'):
            result = _grasp(engine, key, name, p, q, contacts)
            if result:
                return result
        return []
    if (
        tool in ('raney_clip_applier', 'dressing_forceps', 'needle_driver')
        and engine.parameters(engine.state['slots'][key])['jawOpeningMm'] > 0.5
    ):
        return []
    if tool in ('raney_clip_applier', 'dressing_forceps') and op == 'grasp':
        return _pick(engine, key, tool, p, q)
    if tool == 'dressing_forceps':
        return _dress(engine, p, contacts)
    if tool in ('raney_clip_applier', 'fixation_driver', 'needle_driver') and op != 'grasp':
        for name in ('bone',) if tool == 'fixation_driver' else ('scalp', 'dura'):
            events = _anchors(engine, tool, name, key, p, contacts)
            if events:
                return events
        return []
    return None


def summary(engine):
    state = engine.state[PROPERTY]
    bodies = {}
    for name, body in state['bodies'].items():
        cfg = _source(engine, name)
        xyz = centers(engine, name)
        lookup = rest(engine, name)[2]
        cuts = [cfg['seamEdges'][i][:] for i in body['cut']]
        bodies[name] = {
            k: deepcopy(body[k]) for k in ('status', 'angleRad', 'heldBy', 'closedAnchors', 'momentNmm')
        }
        if body['grip']:
            bodies[name]['gripOffsetMm'] = body['grip']['offset'][:]
        removed = set(engine.state['removed'])
        retained = np.array([cell not in removed for cell in cfg['cells']], dtype=bool)
        bodies[name].update(
            cells=[cell for cell, keep in zip(cfg['cells'], retained) if keep],
            centersMm=xyz[retained].tolist(),
            hingeOriginMm=rest(engine, name)[6].tolist(),
            hingeAxis=cfg['hingeAxis'][:],
            cutCells=sorted({i for edge in cuts for i in edge}),
            seamEdges=cuts,
            cutCount=len(cuts),
            edgeCount=len(cfg['seamEdges']),
            gripMm=xyz[lookup[cfg['gripCell']]].tolist(),
            anchorsMm=[xyz[lookup[i]].tolist() for i in cfg['anchorCells']],
        )
    objects = []
    for obj in state['objects']:
        row = {**deepcopy(obj), 'positionMm': _object_position(engine, obj).tolist()}
        if 'surfaceOffsetMm' in obj and obj['status'] not in ('carried', 'retrieved'):
            row['normal'] = _surface_normal(engine, obj['body'], obj['sourceCell']).tolist()
        if obj['kind'] == 'raney-clip':
            from .surgical_operative_clip import rest as clip_rest

            row['clipSegments'] = [
                dict(aMm=a.tolist(), bMm=b.tolist(), radiusMm=r)
                for a, b, r, _ in object_envelopes(engine, obj)
            ]
            row['sourceContactMm'] = engine.scene.points([obj['sourceCell']])[0].tolist()
            row['incisionDirectionRas'] = clip_rest(engine, obj)[2].tolist()
        objects.append(row)
    closure_secure = (
        bodies['dura']['status'] == 'closed'
        and bodies['bone']['status'] == 'fixed'
        and bodies['scalp']['status'] == 'closed'
        and all(o['status'] in ('retrieved', 'intended-retained') for o in objects)
    )
    complete = closure_secure and not any(
        holding(engine, key) or slot['active'] or slot['inserted'] or slot['grip']
        for key, slot in engine.state['slots'].items()
    )
    return {
        'schema': SCHEMA,
        'bodies': bodies,
        'objects': objects,
        'accessOpen': _access(engine),
        'complete': complete,
        'closureSecured': closure_secure,
        'exteriorNormalRas': engine.scene.properties[PROPERTY]['exterior']['normalRas'][:],
    }
