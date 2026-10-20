"""Read-only native-axis 0/1 fallback proposals for the existing public engine.

Source labels, native flat identities, affine and protection are never replaced.
These bounded geometric proposals require public preview/step and its guards;
they are generic research-tool routes, not clinically established approaches.
"""
from __future__ import annotations

import copy
import math
import operator

import numpy as np
from scipy import ndimage as ndi
from scipy.spatial.transform import Rotation

from .surgical_motion import envelopes, pair_conflict, swept_envelopes
from .surgical_tools import TOOLS


def _frame(scene, axis):
    axis = operator.index(axis)
    if axis not in (0, 1):
        raise ValueError('Directional fallback supports native axis 0 or 1 only')
    if not np.allclose(scene.affine[:3, :3], np.diag(np.diag(scene.affine[:3, :3]))):
        raise ValueError('Directional fallback requires an axis-aligned source grid')
    outward = scene.affine[:3, axis] / scene.spacing[axis]
    inward = -outward
    cross = np.cross([0., 0., 1.], inward)
    q = Rotation.from_rotvec(cross / np.linalg.norm(cross) * np.pi / 2.).as_quat()
    return axis, [i for i in range(3) if i != axis], outward, q


def _world(scene, index):
    return scene.affine[:3, :3] @ np.asarray(index, dtype=float) + scene.affine[:3, 3]


def _native(axis, transverse_index, depth):
    result = list(transverse_index)
    result.insert(axis, depth)
    return result


def _pose(point, quaternion):
    return {'position': list(map(float, point)), 'quaternion': list(map(float, quaternion))}


def _grid(engine, axis):
    axis, transverse, outward, q = _frame(engine.scene, axis)
    scene = engine.scene
    # Only the view's indexing changes; engine queries always use original IDs.
    labels = np.moveaxis(scene.labels, axis, -1)
    target_ids = [r['id'] for r in scene.regions if r.get('target') and r['material'] == 'tumor']
    bone_ids = [r['id'] for r in scene.regions if r['material'] == 'bone']
    target = np.isin(labels, target_ids)
    allowed = target | np.isin(labels, bone_ids) | (labels == 0)
    k = np.arange(labels.shape[-1])
    bad_top = np.max(np.where(~allowed, k, -1), axis=2)
    original_eligible = target & (k[None, None, :] > bad_top[:, :, None])
    retained_native = np.ones(scene.shape, dtype=bool)
    retained_native.ravel()[np.asarray(engine.state['removed'], dtype=np.int64)] = False
    retained = original_eligible & np.moveaxis(retained_native, axis, -1)
    top = np.max(np.where(retained, k, -1), axis=2)
    window = top >= 0
    clearance = ndi.distance_transform_edt(window, sampling=scene.spacing[transverse])
    return labels, original_eligible, retained, top, clearance, transverse, outward, q


def directional_access_candidates(engine, *, axis, limit=8, max_radius_mm=3., separation_mm=3.5):
    """Propose retained target windows from increasing native axis 0 or 1.

    Entry/outside poses are world RAS; targetEntryIndex remains original IJK.
    Bone/free may overlie a target window. Real legal drilling is still needed.
    Capacity is a source-column upper bound, never physical admission evidence.
    """
    limit = operator.index(limit)
    cap, separation = float(max_radius_mm), float(separation_mm)
    if limit < 0 or not np.isfinite([cap, separation]).all() or not .1 <= cap <= 3. or separation <= 0.:
        raise ValueError('Nonnegative limit, drill radius in [.1, 3], and positive separation required')
    scene = engine.scene
    labels, initial, retained, top, clearance, transverse, outward, q = _grid(engine, axis)
    cap = min(cap, float(scene.properties.get('tissuePhysicsV2', {}).get('bone', {}).get('maxRadiusMm', 3.)))
    xx, yy = np.ogrid[:top.shape[0], :top.shape[1]]
    counts = retained.sum(axis=2)
    score = ndi.gaussian_filter(counts.astype(float), 1.) * (clearance >= 2.)
    score *= counts > 0
    records = []
    for _ in range(limit):
        ij = np.unravel_index(np.argmax(score), score.shape)
        if score[ij] <= 0.:
            break
        index = _native(axis, ij, int(top[ij]))
        entry = _world(scene, index)
        outside_index = list(index); outside_index[axis] = scene.shape[axis] - 1
        outside = _world(scene, outside_index) + outward * 8.
        radius = min(cap, max(.1, float(clearance[ij]) - 1.5))
        distance2 = ((xx-ij[0])*scene.spacing[transverse[0]])**2 + ((yy-ij[1])*scene.spacing[transverse[1]])**2
        disk = distance2 <= max(1., radius-1.)**2
        records.append({'id': f'directional-{axis}-{ij[0]}-{ij[1]}', 'nativeAxis': int(axis),
                        'outwardIndexSign': 1, 'outwardRAS': outward.tolist(), 'quaternion': q.tolist(),
                        'sourceSceneId': scene.identity, 'slot': 'left',
                        'entryPose': _pose(entry, q), 'outsidePose': _pose(outside, q),
                        'targetEntryIndex': list(map(int, index)), 'drillRadiusMm': radius,
                        'windowClearanceMm': float(clearance[ij]),
                        'geometricTargetCapacityCells': int(initial[disk].sum()),
                        'remainingGeometricCapacityCells': int(counts[disk].sum()),
                        'capacityStateId': engine.state_id, 'strategy': 'native-directional-frontier',
                        'evidence': 'supplied owned labels only; attachments/complete functional anatomy UNKNOWN'})
        score[distance2 < separation**2] = 0.
    return records


def _access_frame(engine, access):
    if access.get('sourceSceneId', engine.scene.identity) != engine.scene.identity:
        raise ValueError('Directional access source scene differs')
    if access.get('outwardIndexSign', 1) != 1:
        raise ValueError('Directional access requires increasing native index outward')
    return _frame(engine.scene, access['nativeAxis'])


def rank_directional_accesses(engine, accesses, start_index=0):
    """Copy/stably rank only the tail by retained original eligible disk cells."""
    start = operator.index(start_index)
    if not 0 <= start <= len(accesses):
        raise ValueError('start_index must identify an access-list boundary')
    result = copy.deepcopy(list(accesses))
    grids = {}
    state_id = engine.state_id
    for access in result[start:]:
        axis, transverse, _, _ = _access_frame(engine, access)
        if axis not in grids:
            retained = _grid(engine, axis)[2]
            grids[axis] = retained.sum(axis=2)
        counts = grids[axis]
        index = access.get('targetEntryIndex')
        if index is None:
            p = np.asarray(access['entryPose']['position'], dtype=float)
            index = np.rint(engine.scene.inverse[:3, :3] @ p + engine.scene.inverse[:3, 3])
        access['remainingGeometricCapacityCells'] = 0
        access['capacityStateId'] = state_id
        if not all(np.isfinite(v) and 0 <= v < n for v, n in zip(index, engine.scene.shape)):
            continue
        i, j = np.asarray(index)[transverse]
        xx, yy = np.ogrid[:counts.shape[0], :counts.shape[1]]
        radius = float(access['drillRadiusMm'])
        if not np.isfinite(radius) or not .1 <= radius <= 3.:
            raise ValueError('Directional drill radius must be in [.1, 3]')
        disk = ((xx-i)*engine.scene.spacing[transverse[0]])**2 + ((yy-j)*engine.scene.spacing[transverse[1]])**2 <= max(1., radius-1.)**2
        access['remainingGeometricCapacityCells'] = int(counts[disk].sum())
    result[start:] = sorted(result[start:], key=lambda r: -r['remainingGeometricCapacityCells'])
    return result


def directional_removal_candidates(engine, access, *, limit=6):
    """Return continuous target-only CUSA apply proposals through this opening.

    Full current occupancy, other-hand envelopes and each engine-sized swept
    working/shaft cover are filtered in order, crediting only preceding tip
    effects. Protection and all other stateful laws remain public-preview
    responsibilities. Radius applies to the entire action, including retreat.
    At most six source columns and a small subvoxel fallback are searched.
    """
    limit = operator.index(limit)
    if limit < 0:
        raise ValueError('Nonnegative candidate limit required')
    if not limit:
        return []
    axis, transverse, outward, q = _access_frame(engine, access)
    slot = access.get('slot', 'left')
    state = engine.state['slots'][slot]
    if state['instrument'] != 'cusa' or state['grip'] or slot in engine.state['holds']:
        return []
    if (Rotation.from_quat(state['quaternion']) * Rotation.from_quat(q).inv()).magnitude() > 1e-6:
        return []
    scene = engine.scene
    labels_view, _, retained, _, _, _, _, _ = _grid(engine, axis)
    owner = engine.occupancy(); source = scene.labels.ravel()
    targets = {r['id'] for r in scene.regions if r.get('target') and r['material'] == 'tumor'}
    current = np.asarray(state['position'], dtype=float)
    params = engine.parameters(state)
    radius_options = sorted({.3, .45, .6, 1., 1.3, *([float(params['radiusMm'])] if .1 <= params['radiusMm'] <= 3. else [])}, reverse=True)
    outside_index = [0, 0, 0]; outside_index[axis] = scene.shape[axis]-1
    outside_w = float(_world(scene, outside_index) @ outward + 8.)
    center = np.asarray(access['entryPose']['position'], dtype=float)
    search_radius = float(access.get('searchRadiusMm', access['drillRadiusMm'] + 2.))
    if not np.isfinite(search_radius) or not 0. < search_radius <= 10.:
        raise ValueError('Directional frontier search radius must be in (0, 10] mm')
    step = min(.05, max(.02, float(access.get('frontierStepMm', .05))))
    inflation = step / 2. + 1e-4
    removed = set(engine.state['removed'])
    other = [part for hand, value in engine.state['slots'].items() if hand != slot
             for part in envelopes(TOOLS[value['instrument']], engine.parameters(value), value['position'], value['quaternion'])]

    def at_w(point, w):
        return point + outward * (float(w)-float(point @ outward))

    def surface_w(point):
        ijk = np.rint(scene.inverse[:3, :3] @ point + scene.inverse[:3, 3]).astype(int)
        if np.any(ijk < 0) or np.any(ijk >= scene.shape):
            return float(center @ outward)
        ij = tuple(ijk[transverse]); ks = np.flatnonzero(np.isin(labels_view[ij], list(targets)))
        return float(_world(scene, _native(axis, ij, int(ks.max()))) @ outward) if len(ks) else float(center @ outward)

    def path_to(endpoint, entry_w, radius):
        path = []
        def append(point):
            if np.linalg.norm(point - (np.asarray(path[-1]['position']) if path else current)) > 1e-8:
                path.append(_pose(point, q))
        def axial(start, finish_w, bands):
            start_w = float(start @ outward)
            low, high = sorted((start_w, finish_w))
            bounds = {start_w, finish_w}
            for a, b in bands:
                bounds.update((max(low, min(high, a)), max(low, min(high, b))))
            points = sorted(bounds, reverse=finish_w < start_w)
            for a, b in zip(points, points[1:]):
                fine = any(lo-1e-9 <= (a+b)/2. <= hi+1e-9 for lo, hi in bands)
                count = max(1, int(np.ceil(abs(b-a)/step))) if fine else 1
                for w in np.linspace(a, b, count+1)[1:]: append(at_w(start, w))
        first_surface = surface_w(current)
        axial(current, outside_w, [(first_surface-4., first_surface+radius+3.)])
        above = at_w(endpoint, outside_w); append(above)
        end_w = float(endpoint @ outward)
        axial(above, end_w, [(entry_w-4., entry_w+radius+3.), (end_w, end_w+3.)])
        return path if len(path) <= 512 else []

    def checked(path, radius, duration):
        predicted = set(); previous = _pose(current, q)
        actual_params = {**params, 'radiusMm': radius}
        budget = min(.4, float(scene.spacing.min())/3.)
        total = 0
        for endpoint in path:
            a, b = np.asarray(previous['position']), np.asarray(endpoint['position'])
            distance = float(np.linalg.norm(b-a))
            if distance > 250.:
                return None
            n = max(1, math.ceil(distance/budget), math.ceil(duration/len(path)/.1))
            total += n
            if total > 20000:
                return None
            start = a.copy()
            for f in np.linspace(0., 1., n+1)[1:]:
                pose = _pose(start*(1.-f)+b*f, q)
                parts = swept_envelopes(TOOLS['cusa'], actual_params, previous, pose)
                if other and pair_conflict(parts, other): return None
                work = set(); shaft = set()
                for lo, hi, width, part in parts:
                    identities = set(owner.material_ids(engine.cells(lo, hi, width))) - predicted
                    (shaft if part == 'shaft' else work).update(identities)
                tip = set(owner.material_ids(engine.cells(previous['position'], pose['position'], radius))) - predicted
                if shaft or any(int(source[c]) not in targets for c in work | tip): return None
                predicted.update(tip)
                previous = pose
        return predicted

    columns = []
    for ij in np.argwhere(retained.any(axis=2)):
        ij = tuple(map(int, ij)); ks = np.flatnonzero(retained[ij])
        point = _world(scene, _native(axis, ij, 0))
        delta = point-center; lateral = delta-outward*float(delta @ outward)
        distance = float(np.linalg.norm(lateral))
        if distance > search_radius + .5:
            continue
        # Reject unopened source-column roofs cheaply. Full occupancy below
        # remains authoritative when material has been held or deformed.
        if not engine.state['holds'] and not engine.state.get('deformation', {}).get('nodes'):
            roof = np.flatnonzero(labels_view[ij])[np.flatnonzero(labels_view[ij]) > ks.max()]
            if any(int(np.ravel_multi_index(tuple(_native(axis, ij, int(k))), scene.shape)) not in removed for k in roof):
                continue
        columns.append((ij, int(ks.min()), len(ks), distance))
    columns.sort(key=lambda row: (-row[2], row[3], row[0]))
    columns = columns[:6]
    records = []
    for fallback in (False, True):
        specs = [(ij, bottom, (0., 0.)) for ij, bottom, _, _ in columns] if not fallback else [
            (ij, bottom, offset) for ij, bottom, _, _ in columns[:2]
            for offset in ((-.25, 0.), (.25, 0.), (0., -.25), (0., .25))]
        if fallback and records: break
        for ij, bottom, offset in specs:
            lateral_index = np.asarray(ij, dtype=float) + offset
            ray = _world(scene, _native(axis, lateral_index, bottom))
            delta = ray-center; lateral = delta-outward*float(delta @ outward)
            if np.linalg.norm(lateral) > search_radius: continue
            entry_w = surface_w(ray)
            for radius in radius_options:
                endpoint = ray + outward*(radius-.3)
                # Bound the endpoint by actual normal/bone/vessel cell caps.
                cells = engine.cells(endpoint, at_w(endpoint, outside_w+min(float(params.get('depthMm', 2.)), 3.)), radius+inflation)
                blocked = [c for c in owner.material_ids(cells) if int(source[c]) not in targets]
                if blocked:
                    if engine.state['holds'] or engine.state.get('deformation', {}).get('nodes'): continue
                    points = scene.points(blocked)
                    gaps = np.maximum(np.abs(points[:, transverse]-endpoint[transverse])-scene.spacing[transverse]/2., 0.)
                    height = np.sqrt(np.maximum((radius+inflation)**2-np.sum(gaps**2, axis=1), 0.))
                    end_w = max(float(endpoint @ outward), float(np.max(points @ outward+scene.spacing[axis]/2.+height))+1e-4)
                    if end_w >= outside_w: continue
                    endpoint = at_w(endpoint, end_w)
                path = path_to(endpoint, entry_w, radius)
                if not path: continue
                travel = sum(float(np.linalg.norm(np.subtract(b['position'], a['position']))) for a, b in zip([_pose(current, q), *path[:-1]], path))
                duration = max(.1, min(30., travel/20.))
                predicted = checked(path, radius, duration)
                if not predicted: continue
                action = {'slot': slot, 'operation': 'apply', 'path': path,
                          'parameters': {'radiusMm': float(radius)}, 'duration': duration}
                records.append((len(predicted), -float(np.linalg.norm(lateral)), action))
                break
    records.sort(key=lambda row: (-row[0], -row[1]))
    return [row[2] for row in records[:limit]]
