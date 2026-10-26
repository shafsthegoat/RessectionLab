"""Bounded, source-bound comparison of three physical access directions.

This is an open-loop engineering rehearsal. The existing director proposes
actions and the public engine alone admits their whole-tool physical effects.
It neither infers missing anatomy nor estimates clinical safety.
"""
from __future__ import annotations

import copy
import time

import numpy as np
from scipy.ndimage import binary_erosion
from scipy.spatial import cKDTree
from scipy.spatial.transform import Rotation

from .core import array_digest, semantic_digest
from .full_case_directional import directional_access_candidates, _frame
from .full_case_geometry import access_candidates
from .full_case_rehearsal import new_director, run_slice
from .surgical_planning import _protection_measurements
from .surgical_rl_rewards import RewardTracker

TASK = 'anatomy-access'
_DIRECTIONS = ((2, 'superior-access', 'Superior access'),
               (0, 'axis0-access', 'Native axis 0 access'),
               (1, 'axis1-access', 'Native axis 1 access'))
_UNKNOWN = ('near_contact_exposure_mm3_seconds', 'deformation_injury_mm3',
            'clinical_risk', 'complete_protected_structure_encounters')


class AccessNotExecutable(ValueError):
    """A declared source direction could not produce bounded target progress."""


def _brain_regions(scene):
    """Only a layer explicitly identified as brain can authorize this corridor."""
    return [r for r in scene.regions if r['material'] == 'soft_tissue' and
            (r['name'].strip().lower() in ('brain', 'whole brain') or
             r['name'].strip().lower().startswith('mapped brain'))]


def _layer_evidence(scene, name):
    source = (scene.source.get('adapter') or {}).get('sourceLayers', {}).get(name, {})
    provenance = source.get('provenance') or {}
    inference = provenance.get('inference') or {}
    return {'regionName': name, 'layerId': source.get('layerId'),
            'reviewStatus': source.get('reviewStatus', 'not supplied'),
            'provenanceKind': provenance.get('kind', 'not supplied'),
            'modelId': inference.get('model_id') or provenance.get('model'),
            'annotationHash': scene.source.get('annotationHashes', {}).get(name)}


def access_availability(engine):
    """Cheap necessary conditions; physical route viability is tested by execution."""
    scene = engine.scene
    reasons = []
    if engine.actions or engine.state['removed'] or engine.state['holds'] or any(
            s['inserted'] or s['active'] or s['grip'] for s in engine.state['slots'].values()):
        reasons.append('Compare access from an intact, withdrawn initial state.')
    if not any(r.get('target') and r['material'] == 'tumor' and
               np.any(scene.labels == r['id']) for r in scene.regions):
        reasons.append('A nonempty mapped tumor target is required.')
    skull = any(r['material'] == 'bone' and np.any(scene.labels == r['id']) for r in scene.regions)
    brain = _brain_regions(scene)
    if (not np.allclose(scene.affine[:3, :3], np.diag(np.diag(scene.affine[:3, :3])))
            or scene.affine[2, 2] <= 0):
        reasons.append('These three native-direction routes require a diagonal grid with superior-increasing index 2.')
    return {'accessAvailable': not reasons,
            'accessReason': ' '.join(reasons) if reasons else None,
            'skullAvailable': skull,
            'modeledBrainLayerAvailable': bool(brain),
            'modeledBrainLayerEvidence': [_layer_evidence(scene, r['name']) for r in brain],
            'skullStatus': 'mapped_source_bone' if skull else 'unknown_no_source_skull',
            'accessScope': ('Three bounded source-grid directions; no skull opening is claimed when skull anatomy is absent. '
                            'Execution may still refuse a route.')}


def _bone_on_source_ray(engine, access, axis):
    index = access['targetEntryIndex']
    labels = engine.scene.labels
    bone = {r['id'] for r in engine.scene.regions if r['material'] == 'bone'}
    return sum(int(labels[tuple(index[:axis] + [depth] + index[axis + 1:])] in bone)
               for depth in range(index[axis] + 1, labels.shape[axis]))


def _brain_corridor_candidate(engine, axis):
    """Choose a mapped-brain ray; source void remains explicitly unsupported."""
    scene = engine.scene
    oblique = _oblique_brain_corridor_candidate(engine, axis)
    if oblique is not None:
        return oblique
    view = np.moveaxis(scene.labels, axis, -1)
    targets = {r['id'] for r in scene.regions if r.get('target') and r['material'] == 'tumor'}
    brain = {r['id'] for r in _brain_regions(scene)}
    if not brain:
        return None
    target = np.isin(view, list(targets))
    mapped_brain = np.isin(view, list(brain))
    depth = np.arange(view.shape[2])
    top = np.max(np.where(target, depth, -1), axis=2)
    outward = depth[None, None, :] > top[:, :, None]
    brain_count = np.sum(mapped_brain & outward, axis=2)
    blocked = np.any((view != 0) & ~target & ~mapped_brain & outward, axis=2)
    surface = np.max(np.where(mapped_brain & outward, depth, -1), axis=2)
    # A missing segment between target and mapped brain is not treated as
    # observed free tissue. Outside the outer brain surface remains unknown.
    gaps = np.any((view == 0) & outward &
                  (depth[None, None, :] <= surface[:, :, None]), axis=2)
    eligible = (top >= 0) & (brain_count > 0) & ~blocked & ~gaps
    if not np.any(eligible):
        return None
    target_count = np.sum(target, axis=2)
    candidates = np.argwhere(eligible)
    i, j = min((tuple(map(int, ij)) for ij in candidates),
               key=lambda ij: (int(brain_count[ij]), -int(target_count[ij]), ij))
    index = [i, j]
    index.insert(axis, int(top[i, j]))
    if axis == 2:
        outward_ras = (scene.affine[:3, 2] / scene.spacing[2]).tolist()
        quaternion = [1., 0., 0., 0.]
    else:
        _, _, direction, quaternion = _frame(scene, axis)
        outward_ras = direction.tolist()
        quaternion = list(map(float, quaternion))
    entry = (scene.affine[:3, :3] @ index + scene.affine[:3, 3]).tolist()
    outside_index = list(index)
    outside_index[axis] = scene.shape[axis]-1
    outside = (scene.affine[:3, :3] @ outside_index + scene.affine[:3, 3]
               + np.asarray(outward_ras)*8.).tolist()
    return {'id': f'mapped-brain-corridor-{axis}-{i}-{j}', 'nativeAxis': axis,
        'entryPose': {'position': entry, 'quaternion': quaternion},
        'outsidePose': {'position': outside, 'quaternion': quaternion},
        'targetEntryIndex': index, 'sourceBrainSurfaceIndex': int(surface[i, j]),
        'sourceBrainCellsOnRay': int(brain_count[i, j]),
        'sourceBrainThicknessMm': float(brain_count[i, j]*scene.spacing[axis]),
        'unmappedExterior': True, 'corridorRadiusMm': .6, 'drillRadiusMm': None,
        'geometricTargetCapacityCells': int(target_count[i, j]),
        'sourceSceneId': scene.identity,
        'sourceBrainLayers': [_layer_evidence(scene, r['name']) for r in _brain_regions(scene)],
        'sourceTargetLayers': [_layer_evidence(scene, r['name']) for r in scene.regions
                               if r.get('target') and r['material'] == 'tumor'],
        'accessCompleteness': 'unsupported_skull_access_brain_corridor',
        'evidence': 'explicitly mapped/estimated brain layer and supplied target labels; exterior/skull/dura and complete vascular/functional anatomy UNKNOWN'}


def _oblique_brain_corridor_candidate(engine, axis):
    """Shortest source-labeled tissue ray from an exterior-facing native face.

    The entry face defines the approach direction; the working shaft may be
    oblique. Every sampled cell before the target must be mapped brain.
    """
    scene = engine.scene
    labels = scene.labels
    brain_regions = _brain_regions(scene)
    brain_ids = [r['id'] for r in brain_regions]
    target_ids = [r['id'] for r in scene.regions if r.get('target') and r['material'] == 'tumor']
    if not brain_ids or not target_ids:
        return None
    brain = np.isin(labels, brain_ids)
    target = np.isin(labels, target_ids)
    face = np.zeros_like(brain)
    inner = [slice(None)]*3
    outer = [slice(None)]*3
    inner[axis] = slice(0, -1)
    outer[axis] = slice(1, None)
    face[tuple(inner)] = brain[tuple(inner)] & (labels[tuple(outer)] == 0)
    surface = np.argwhere(face)
    boundary = np.argwhere(target & ~binary_erosion(target))
    if not len(surface) or not len(boundary):
        return None
    tree = cKDTree(boundary*scene.spacing)
    distances, neighbors = tree.query(surface*scene.spacing, k=min(4, len(boundary)))
    distances = distances.reshape(-1, 1) if distances.ndim == 1 else distances
    neighbors = neighbors.reshape(-1, 1) if neighbors.ndim == 1 else neighbors
    rank = np.argsort(distances, axis=None)
    preflight = engine.clone()
    preflight.step({'slot': 'left', 'operation': 'exchange', 'instrument': 'cusa', 'duration': .2})
    tested = 0
    for flat in rank[:min(2000, len(rank))]:
        row, column = np.unravel_index(flat, distances.shape)
        length = float(distances[row, column])
        # At least one distinct brain cell is necessary. A long straight
        # corridor already exceeds the caller's hard healthy-removal budget.
        if length < 1. or length > 45.:
            continue
        entry = surface[row].astype(float)
        goal = boundary[int(neighbors[row, column])].astype(float)
        direction_index = goal-entry
        if direction_index[axis] >= -1.:
            continue
        count = max(2, int(np.ceil(length/.4)))
        sampled = np.rint(entry[None, :] +
            np.linspace(0., 1., count)[:, None]*direction_index[None, :]).astype(int)
        cells = labels[tuple(sampled.T)]
        if not (np.isin(cells[:-1], brain_ids+target_ids).all() and
                np.isin(cells, brain_ids).any() and np.isin(cells, target_ids).any()):
            continue
        entry_world = scene.affine[:3, :3] @ entry + scene.affine[:3, 3]
        goal_world = scene.affine[:3, :3] @ goal + scene.affine[:3, 3]
        inward = (goal_world-entry_world)/length
        outside = entry_world-inward*8.
        exterior = np.linspace(entry_world-inward*.8, outside, 20)
        exterior_index = np.rint((scene.inverse[:3, :3] @
                                  (exterior-scene.affine[:3, 3]).T).T).astype(int)
        valid = np.all((exterior_index >= 0) &
                       (exterior_index < np.asarray(scene.shape)), axis=1)
        if np.any(valid) and np.any(labels[tuple(exterior_index[valid].T)] != 0):
            continue
        quaternion = Rotation.align_vectors([inward], [[0., 0., 1.]])[0].as_quat().tolist()
        pose = {'position': goal_world.tolist(), 'quaternion': quaternion}
        radius = 1. if length <= 20. else .6
        outside_pose = {'position': outside.tolist(), 'quaternion': quaternion}
        tested += 1
        place = preflight.preview({'slot': 'left', 'operation': 'place',
            'pose': outside_pose, 'parameters': {'radiusMm': radius, 'forceN': 0.},
            'duration': .2}, include_view=False)
        if place['status'] != 'accepted':
            continue
        return {'id': f'mapped-brain-oblique-{axis}-{row}-{column}',
            'nativeAxis': axis, 'pathType': 'oblique_mapped_brain',
            'entryPose': pose, 'outsidePose': outside_pose,
            'surfacePose': {'position': entry_world.tolist(), 'quaternion': quaternion},
            'inwardRasUnit': inward.tolist(), 'pathLengthMm': length,
            'targetEntryIndex': goal.astype(int).tolist(),
            'sourceBrainSurfaceIndex': int(entry[axis]),
            'sourceBrainCellsOnRay': int(np.count_nonzero(np.isin(cells, brain_ids))),
            'sourceBrainThicknessMm': length, 'unmappedExterior': True,
            'corridorRadiusMm': radius, 'drillRadiusMm': None,
            'routePreflightCandidates': tested,
            'geometricTargetCapacityCells': int(np.count_nonzero(np.isin(cells, target_ids))),
            'sourceSceneId': scene.identity,
            'sourceBrainLayers': [_layer_evidence(scene, r['name']) for r in brain_regions],
            'sourceTargetLayers': [_layer_evidence(scene, r['name']) for r in scene.regions
                                   if r.get('target') and r['material'] == 'tumor'],
            'accessCompleteness': 'unsupported_skull_access_brain_corridor',
            'evidence': 'sampled mapped/estimated brain layer and supplied target ray from a source brain surface; exterior/skull/dura and complete vascular/functional anatomy UNKNOWN'}
    return None


def _routes(engine, *, skull_available, allow_modeled_brain_corridor=False):
    routes, failures = [], []
    for axis, method, label in _DIRECTIONS:
        candidates = (access_candidates(engine, limit=12, max_radius_mm=3.) if axis == 2 else
                      directional_access_candidates(engine, axis=axis, limit=12, max_radius_mm=3.))
        chosen = next((r for r in candidates if r['geometricTargetCapacityCells'] > 0
                       and (not skull_available or _bone_on_source_ray(engine, r, axis) > 0)), None)
        if chosen is None and not skull_available and allow_modeled_brain_corridor:
            chosen = _brain_corridor_candidate(engine, axis)
        if chosen is None:
            failures.append({'nativeAxis': axis, 'method': method,
                'reason': 'No source-skull and target-bearing opening exists' if skull_available else
                          ('No executable mapped source-target corridor exists; missing anatomy cannot be treated as free'
                           if allow_modeled_brain_corridor else
                           'No exposed source-target window exists; missing anatomy cannot be treated as a free route')})
            continue
        route = copy.deepcopy(chosen)
        if 'accessCompleteness' not in route:
            route.update(nativeAxis=axis, sourceSceneId=engine.scene.identity,
                         sourceBoneCellsOnRay=_bone_on_source_ray(engine, chosen, axis),
                         accessCompleteness='source_skull_and_target' if skull_available else
                                            'unsupported_skull_access_source_tissue_only')
        routes.append((method, label, route))
    return routes, failures


def _residual(engine):
    scene = engine.scene
    target = np.isin(scene.labels, [r['id'] for r in scene.regions
                                        if r.get('target') and r['material'] == 'tumor'])
    target.ravel()[np.asarray(engine.state['removed'], dtype=np.int64)] = False
    indices = np.argwhere(target)
    points = indices @ scene.affine[:3, :3].T + scene.affine[:3, 3]
    return {'cells': int(len(indices)), 'volumeMm3': float(len(indices) * scene.voxel_mm3),
            'centroidRasMm': points.mean(axis=0).tolist() if len(points) else None,
            'boundsRasMm': [points.min(axis=0).tolist(), points.max(axis=0).tolist()] if len(points) else None,
            'sourceCellHash': array_digest(target)}


def _measurements(engine):
    return (_protection_measurements(engine) if engine.protection is not None else
            {'protected_damage_mm3': 0.})


def _execute_brain_corridor(base, route, *, max_actions, healthy_cap, check):
    """Progress a real CUSA tip through mapped brain, bounded by previewed effects."""
    candidate = base.clone()
    axis = route['nativeAxis']
    source = base.scene
    index = list(route['targetEntryIndex'])
    quaternion = route['entryPose']['quaternion']
    brain = {r['id'] for r in _brain_regions(source)}
    tumor = {r['id'] for r in source.regions if r.get('target') and r['material'] == 'tumor'}
    director = {'status': 'running', 'stage': 'mapped_brain_corridor', 'stopReason': None,
                'rejections': [], 'attemptedActions': 0,
                'phases': [{'id': 'mapped-brain-corridor',
                            'label': 'Modeled CUSA passage through mapped/estimated brain tissue; skull access unsupported',
                            'startPosition': 0, 'endPosition': 0}]}

    def pose(depth):
        if route.get('pathType') == 'oblique_mapped_brain':
            p = (np.asarray(route['surfacePose']['position']) +
                 np.asarray(route['inwardRasUnit'])*float(depth))
            return {'position': p.tolist(), 'quaternion': list(quaternion)}
        point = list(index)
        point[axis] = float(depth)
        p = source.affine[:3, :3] @ point + source.affine[:3, 3]
        return {'position': p.tolist(), 'quaternion': list(quaternion)}

    def attempt(action):
        check()
        director['attemptedActions'] += 1
        preview = candidate.preview(action, check=check, include_view=False)
        reason = preview.get('reason')
        if preview['status'] == 'rejected' or (preview['status'] == 'partial' and
                                               reason != 'TISSUE_ASPIRATION_IN_PROGRESS'):
            director['rejections'].append({'action': action, 'reason': reason or preview['status']})
            return False
        changed = preview.get('deltas', {}).get('removedCells', [])
        if any(int(source.labels.flat[cell]) not in brain | tumor for cell in changed):
            director['rejections'].append({'action': action,
                'reason': 'NON_BRAIN_OR_NON_TARGET_REMOVAL_REFUSED'})
            return False
        if (preview['metrics']['nonTargetRemovedMm3']-base.metrics()['nonTargetRemovedMm3']
                > healthy_cap + 1e-9):
            director['rejections'].append({'action': action, 'reason': 'HEALTHY_REMOVAL_CAP'})
            return False
        committed = candidate.step(action, check=check)
        if committed['stateId'] != preview['stateId'] or committed['status'] != preview['status']:
            raise ValueError('Brain corridor preview and committed transition differ')
        director['phases'][-1]['endPosition'] = len(candidate.actions)
        return True

    setup = [
        {'slot': 'left', 'operation': 'exchange', 'instrument': 'cusa', 'duration': .2},
        {'slot': 'left', 'operation': 'place', 'pose': copy.deepcopy(route['outsidePose']),
         'parameters': {'radiusMm': route['corridorRadiusMm'], 'forceN': 0.}, 'duration': .2},
        # The mapped surface may rise beside this center ray. Enter with the
        # active, previewed tip so adjacent real tissue is removed and counted.
        {'slot': 'left', 'operation': 'apply',
         'pose': pose(-1.5 if route.get('pathType') == 'oblique_mapped_brain' else
                      route['sourceBrainSurfaceIndex']+1.5),
         'parameters': {'radiusMm': route['corridorRadiusMm']}, 'duration': 1.},
    ]
    for action in setup:
        if len(candidate.actions) >= max_actions or not attempt(action):
            director.update(status='stopped', stopReason=(director['rejections'][-1]['reason']
                if director['rejections'] else 'setup_action_budget'))
            return candidate, director
    oblique = route.get('pathType') == 'oblique_mapped_brain'
    if not oblique:
        surface = route['sourceBrainSurfaceIndex']
        target = index[axis]
        step = 3./float(source.spacing[axis])
        depths = [surface + .5]
        while depths[-1] > target-.5+1e-9:
            depths.append(max(target-.5, depths[-1]-step))
    else:
        depths = [-.5]
    position = depths[0]
    first_stroke = True
    while True:
        if len(candidate.actions) >= max_actions:
            break
        if oblique:
            goal = route['pathLengthMm']+.5
            options = ([position] if first_stroke else []) + [
                min(goal, position+step) for step in (3., 2., 1., .5)
                if position < goal-1e-9 and min(goal, position+step) > position+1e-9]
        else:
            options = depths[:1]
            depths = depths[1:]
        if not options:
            break
        advanced = False
        for depth in dict.fromkeys(options):
            action = {'slot': 'left', 'operation': 'apply', 'pose': pose(depth),
                      'parameters': {'radiusMm': route['corridorRadiusMm']},
                      'duration': .6}
            if attempt(action):
                position = depth
                advanced = True
                first_stroke = False
                break
        if not advanced:
            director.update(status='stopped', stopReason=director['rejections'][-1]['reason'])
            break
        if candidate.metrics()['targetRemovedMm3']-base.metrics()['targetRemovedMm3'] >= 5.:
            director.update(status='stopped', stopReason='bounded_target_progress')
            director['phases'].append({'id': 'corridor-withdrawal',
                'label': 'Checked deactivation and withdrawal along the modeled corridor',
                'startPosition': len(candidate.actions), 'endPosition': len(candidate.actions)})
            distance = float(np.linalg.norm(np.subtract(
                route['outsidePose']['position'], candidate.state['slots']['left']['position'])))
            cleanup = [
                {'slot': 'left', 'operation': 'deactivate', 'duration': .2},
                {'slot': 'left', 'operation': 'withdraw', 'pose': copy.deepcopy(route['outsidePose']),
                 'duration': max(.2, min(30., distance/20.))},
            ]
            for action in cleanup:
                if len(candidate.actions) >= max_actions:
                    director['stopReason'] = 'target_progress_cleanup_action_budget'
                    break
                if not attempt(action):
                    if (action['operation'] == 'withdraw' and
                            director['rejections'][-1]['reason'] == 'SHAFT_COLLISION'):
                        # Same checked line and orientation, with smaller
                        # microsteps for the conservative swept shaft capsule.
                        slow = copy.deepcopy(action)
                        slow['duration'] = min(30., max(action['duration'], distance/2.))
                        director['withdrawalRecovery'] = {
                            'fastPreviewRejection': 'SHAFT_COLLISION',
                            'fastDurationSeconds': action['duration'],
                            'slowDurationSeconds': slow['duration'],
                            'samePoseAndQuaternion': True,
                            'slowStatus': 'not_attempted'}
                        if slow['duration'] > action['duration'] + 1e-9:
                            success = attempt(slow)
                            director['withdrawalRecovery']['slowStatus'] = (
                                'accepted' if success else 'refused')
                            if success:
                                director['stopReason'] = 'bounded_target_progress_slow_checked_withdrawal'
                                continue
                    director['stopReason'] = 'target_progress_cleanup_refused_'+str(director['rejections'][-1]['reason'])
                    break
            break
    return candidate, director


def _execute(base, template, method, label, route, *, max_actions, skull_available,
             healthy_cap, check):
    candidate = base.clone()
    director = copy.deepcopy(template)
    started = time.perf_counter()
    if route['accessCompleteness'] == 'unsupported_skull_access_brain_corridor':
        candidate, director = _execute_brain_corridor(base, route,
            max_actions=max_actions, healthy_cap=healthy_cap, check=check)
    else:
        director.update(accesses=[route], accessAxis=route['nativeAxis'], accessIndex=0,
                        passIndex=2, stage='inspect' if skull_available else 'cusa_setup')
        while len(candidate.actions) < max_actions and director['status'] not in ('complete', 'stopped'):
            check()
            if not skull_available and (candidate.metrics()['targetRemovedMm3'] > 0 or
                                        director['stage'] in ('access_setup', 'drill')):
                break
            before = (len(candidate.actions), director['stage'], director['accessIndex'])
            candidate, director = run_slice(candidate, director,
                max_actions=min(8 if skull_available else 1, max_actions-len(candidate.actions)), check=check)
            after = (len(candidate.actions), director['stage'], director['accessIndex'])
            if before == after and director['status'] not in ('complete', 'stopped'):
                raise AccessNotExecutable(f'{label} made no action or scheduling progress')
    planning_seconds = time.perf_counter()-started
    metrics = candidate.metrics()
    removed = metrics['removedByLabelMm3']
    bone = sum(float(removed.get(r['name'], 0.)) for r in candidate.scene.regions if r['material'] == 'bone')
    if (skull_available and bone <= 0) or metrics['targetRemovedMm3'] <= 0:
        raise AccessNotExecutable(f'{label} did not execute '+
            ('both source-bone opening and target removal' if skull_available else 'source-target removal')+
            f' within {max_actions} actions; stop={director.get("stopReason")}')
    if not skull_available:
        current_tool = base.state['slots']['left']['instrument']
        for action in candidate.actions:
            if action['slot'] == 'left' and action['operation'] == 'exchange':
                current_tool = action['instrument']
            if action['slot'] == 'left' and action['operation'] == 'apply' and current_tool == 'cranial_drill':
                raise ValueError('Skull-absent source-tissue comparison attempted drilling')
    rows = [{'index': i, 'actionId': action.get('instrument', action['operation']),
             'action': copy.deepcopy(action), 'scores': [],
             'observationHash': history['stateBefore'], 'status': history['status'],
             'reason': history.get('reason'), 'stateId': history['stateAfter'], 'reward': 0.}
            for i, (action, history) in enumerate(zip(candidate.actions, candidate.history))]
    if len(rows) != len(candidate.actions):
        raise ValueError('Candidate native action history is incomplete')
    replay_started = time.perf_counter()
    replay = base.fresh()
    for row in rows:
        check()
        result = replay.step(row['action'], check=check)
        if (result['status'] != row['status'] or result.get('reason') != row['reason']
                or replay.state_id != row['stateId']):
            raise ValueError('Independent native access replay differs from sealed actions')
    if replay.state != candidate.state or replay.state_id != candidate.state_id:
        raise ValueError('Independent native access endpoint differs')
    tracker = RewardTracker(candidate, candidate._protected_ids)
    endpoint = tracker.measure(candidate, _measurements(candidate))
    replay_endpoint = RewardTracker(replay, replay._protected_ids).measure(replay, _measurements(replay))
    if endpoint != replay_endpoint:
        raise ValueError('Independent native access endpoint vector differs')
    initial = RewardTracker(base, base._protected_ids).measure(base, _measurements(base))
    unknown = [key for key in _UNKNOWN if key not in endpoint or
               (key == 'near_contact_exposure_mm3_seconds' and not hasattr(candidate, 'rl_near_exposure')) or
               (key == 'deformation_injury_mm3' and not hasattr(candidate, 'rl_strain_peaks'))]
    total_target = sum(int(np.count_nonzero(base.scene.labels == r['id']))*base.scene.voxel_mm3
                       for r in base.scene.regions if r.get('target') and r['material'] == 'tumor')
    incomplete = not skull_available or director['status'] not in ('complete', 'stopped')
    summary = {key: metrics.get(key) for key in ('targetRemovedMm3', 'nonTargetRemovedMm3',
        'activeVesselWounds', 'modeledVesselFluidGeneratedMm3', 'fluidPresentMm3', 'fluidRemovedMm3',
        'thermalExposure', 'drillHeatGeneratedJ', 'drillRemovedVolumeMm3', 'elapsedSeconds')}
    summary.update(residualTargetMm3=total_target-metrics['targetRemovedMm3'],
                   boneRemovedMm3=bone if skull_available else None,
                   healthyRemovedMm3=endpoint['healthy_removed_mm3'],
                   vesselDamageMm2=endpoint['vessel_damage_mm2'],
                   bleedingGeneratedMm3=endpoint['bleeding_generated_mm3'])
    return candidate, {'method': method, 'label': label, 'route': route,
        'checkpointSha256': None, 'startStateId': base.state_id, 'finalStateId': candidate.state_id,
        'prefixActions': 0, 'decisions': rows,
        'stopReason': (('unsupported_skull_access_brain_corridor_'+str(director.get('stopReason') or 'action_budget'))
                       if route['accessCompleteness'] == 'unsupported_skull_access_brain_corridor' else
                       'unsupported_skull_access_bounded_source_tissue_progress' if not skull_available else
                       'action_budget' if incomplete else director.get('stopReason') or director['status']),
        'truncated': incomplete, 'earlyStop': False, 'localReward': 0.,
        'cumulativeEndpoints': {key: None if key in unknown else value for key, value in endpoint.items()},
        'incrementalEndpoints': {key: None if key in unknown else value-initial[key] for key, value in endpoint.items()},
        'unassessedEndpoints': list(_UNKNOWN),
        'fullTargetFractionRemoved': endpoint['tumor_removed_mm3']/total_target,
        'reachableTargetFractionRemoved': None, 'neurologicalHarm': 'unknown; no calibrated outcome model',
        'invalidActions': 0, 'previewRejections': len(director['rejections']),
        'travelMm': candidate.state['travelMm']-base.state['travelMm'],
        'instrumentChanges': sum(a['operation'] == 'exchange' for a in candidate.actions),
        'planSeal': semantic_digest({'sourceSceneId': base.scene.identity,
                                     'startStateId': base.state_id, 'decisions': rows}),
        'summary': summary, 'residualTarget': _residual(candidate),
        'accessCompleteness': route['accessCompleteness'],
        'directorStatus': director['status'], 'directorPhases': director['phases'],
        'withdrawalRecovery': copy.deepcopy(director.get('withdrawalRecovery')),
        'cost': {'planningSeconds': planning_seconds,
                 'freshReplaySeconds': time.perf_counter()-replay_started,
                 'executionCalls': len(rows), 'freshReplayCalls': len(rows),
                 'previewCalls': director['attemptedActions'], 'optimizerUpdates': 0},
        'verification': {'exactStateMatch': True, 'endpointVectorMatch': True,
            'scope': 'Separate exact native replay from the same intact source state; not physical validation.'}}


def compare_access_approaches(engine, *, acknowledge_unknowns=False,
                              allow_modeled_brain_corridor=False,
                              max_actions_per_approach=48, max_healthy_removed_mm3=200.,
                              max_wall_seconds=120., check=lambda: None, progress=lambda *_: None):
    """Compare independent source directions and retain completed branches on timeout."""
    if acknowledge_unknowns is not True:
        raise ValueError('Explicitly acknowledge missing anatomy and uncalibrated simulation')
    if type(max_actions_per_approach) is not int or not 1 <= max_actions_per_approach <= 64:
        raise ValueError('Access comparison requires 1..64 actions per approach')
    if type(allow_modeled_brain_corridor) is not bool:
        raise ValueError('Brain corridor opt-in must be boolean')
    if (type(max_healthy_removed_mm3) not in (int, float) or
            not np.isfinite(max_healthy_removed_mm3) or not 0 < max_healthy_removed_mm3 <= 200.):
        raise ValueError('Modeled healthy-removal cap must be in (0, 200] mm3')
    if type(max_wall_seconds) not in (int, float) or not np.isfinite(max_wall_seconds) or not 0 < max_wall_seconds <= 120:
        raise ValueError('Access comparison requires a finite wall budget in (0, 120] seconds')
    readiness = access_availability(engine)
    if not readiness['accessAvailable']:
        raise ValueError('Access comparison refused: ' + readiness['accessReason'])
    if engine.state_id != engine.fresh().state_id:
        raise ValueError('Access comparison refused: initial state differs from a fresh engine state')
    started = time.perf_counter()
    source_hash, state_id = array_digest(engine.scene.labels), engine.state_id
    original = engine.snapshot()
    def bounded_check():
        check()
        if time.perf_counter()-started > max_wall_seconds:
            raise TimeoutError('Access comparison exceeded its aggregate wall budget')
    routes, failures = _routes(engine, skull_available=readiness['skullAvailable'],
                               allow_modeled_brain_corridor=allow_modeled_brain_corridor)
    template = new_director(engine)
    branches = []
    for index, (method, label, route) in enumerate(routes):
        try:
            bounded_check()
            progress(.1+.28*index, 'Executing and checking '+label)
            branches.append(_execute(engine, template, method, label, route,
                max_actions=max_actions_per_approach, skull_available=readiness['skullAvailable'],
                healthy_cap=max_healthy_removed_mm3, check=bounded_check))
        except AccessNotExecutable as error:
            failures.append({'nativeAxis': route['nativeAxis'], 'method': method, 'reason': str(error)})
        except TimeoutError as error:
            failures.append({'nativeAxis': route['nativeAxis'], 'method': method,
                'reason': 'Aggregate wall budget ended during this route: '+str(error)})
            failures.extend({'nativeAxis': other_route['nativeAxis'], 'method': other_method,
                'reason': 'Not run after aggregate wall budget'}
                for other_method, _, other_route in routes[index+1:])
            break
    if engine.snapshot() != original or engine.state_id != state_id or array_digest(engine.scene.labels) != source_hash:
        raise ValueError('Access comparison modified the common initial source')
    report = {'schema': 'resectionlab.surgical-planning.v1', 'task': TASK,
        'status': 'three_executable_source_approaches' if len(branches) == 3 and readiness['skullAvailable'] else
                  'three_source_tissue_routes_skull_unsupported' if len(branches) == 3 else
                  'insufficient_distinct_routes',
        'requiredAlternatives': 3, 'directionFailures': failures,
        'sourceSceneId': engine.scene.identity, 'startStateId': state_id,
        'sourceRole': 'Source-bound three-direction access · research simulation',
        'sourceLabelsHash': source_hash, 'guardContractId': engine.guard_report()['contractId'],
        'availability': readiness, 'totalWallSeconds': time.perf_counter()-started,
        'optimizerUpdates': 0, 'allowModeledBrainCorridor': allow_modeled_brain_corridor,
        'limits': {'maxActionsPerApproach': max_actions_per_approach,
                   'maxWallSeconds': float(max_wall_seconds),
                   'maxAdditionalHealthyRemovedMm3': float(max_healthy_removed_mm3),
                   'routeCount': 3,
                   'directions': [2, 0, 1]},
        'limitations': ['Each route is an open-loop bounded engineering rehearsal, not a complete operation or clinical recommendation.',
            'All successful routes begin at one intact state; missing skull anatomy is not synthesized.',
            'When source skull is absent, no drilling is attempted and source-tissue progress does not establish a complete entry or craniotomy.',
            'Opted-in modeled brain corridors can remove mapped non-target brain; their actual healthy removal is reported and capped before each public action.',
            'The three directions require a diagonal source grid and increasing native-index entry; arbitrary oblique routes are unsupported.',
            'Whole-tool geometry, contact and source exclusions are checked by the shared public engine.',
            'Modeled removal, wound fluid, drilling heat and elapsed time are uncalibrated effects, not clinical risk.',
            'Unlabeled vascular/functional anatomy and neurologic outcome remain unknown; residual target is not proof of unresectability.',
            'A bounded action endpoint can leave tools in the field; truncated means incomplete.'],
        'alternatives': [row for _, row in branches]}
    return branches, report
