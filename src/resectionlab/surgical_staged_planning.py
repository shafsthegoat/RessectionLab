"""Anatomy-aware staged control using existing native CUSA and access plans.

This extends the checked skill inventory, not the engine or a learned actor.
Source geometry proposes work; native consequences and a checked exit admit it.
The original two-sector gate is reported separately from the larger task.
"""
from __future__ import annotations

import copy
import hashlib
import math
import time
import json
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation

from .core import array_digest, semantic_digest
from .geometry import GeometryScene, capsule_voxel_indices
from .surgical_access_planning import _brain_route_novelty, _execute, AccessNotExecutable
from .surgical_engine import SurgicalEngine
from .surgical_modeled_outcomes import summarize_engine_outcomes
from .surgical_sector_planning import _prefix_actions, _sector_cells
from .surgical_staged_skills import (
    SkillAction, SkillLimits, checked_actions, make_skill_checkpoint,
    restore_skill_checkpoint, propose_cusa_maneuvers, checked_cusa_motion)
from .surgical_transition_meter import TransitionMeter, TransitionBudgetExceeded

SCHEMA = 'resectionlab.staged-anatomy-planning.v1'
INVENTORY_SCHEMA = 'resectionlab.source-work-inventory.v1'
MAX_ACTIONS = 240
MAX_CALLS = 4000
MAX_SECONDS = 240.
MAX_TILES = 48
MAX_STAGES = 48
MAX_PROBES_PER_STAGE = 4
RADII_MM = (.6, 1., 1.2, 2.2, 2.8)
EXPORT_RESERVE = 3*MAX_ACTIONS


def require_staged_development(case, declarations_root=None):
    """Enforce pre-existing source roles before staged patient execution."""
    root = (Path(declarations_root) if declarations_root is not None else
            Path(__file__).resolve().parents[2]/'testcases/downloaded-datasets')
    adapter = case.metadata.get('surgical_adapter', {})
    patient = adapter.get('sourcePatient')
    manifest = adapter.get('sourceManifest', {})
    series = manifest.get('series', [])
    revisions = {row.get('source_mirror_revision') for row in series}
    if (not isinstance(patient, str) or manifest.get('patient_id') != patient or
            not series or len(revisions) != 1 or None in revisions or
            any(row.get('patient_id') != patient for row in series)):
        raise ValueError('Staged execution requires a source-bound declared DEVELOPMENT patient')
    matches = []
    for path in root.glob('*/development-declaration.json'):
        declaration = json.loads(path.read_text())
        if patient not in declaration.get('cases', []):
            continue
        if declaration.get('role') != 'DEVELOPMENT' or \
                declaration.get('mirror_revision') != next(iter(revisions)):
            raise ValueError('Staged patient role or revision conflicts with its declaration')
        matches.append(path)
    if len(matches) != 1:
        raise ValueError('Staged execution excludes frozen or undeclared assessment anatomy')
    return {'sourceRole': 'DEVELOPMENT', 'patientId': patient,
            'declarationSha256': hashlib.sha256(matches[0].read_bytes()).hexdigest()}


def source_zones(scene):
    """Eight disjoint RAS centroid octants; source membership is audit only."""
    target_ids = [row['id'] for row in scene.regions
                  if row.get('target') and row['material'] == 'tumor']
    indices = np.argwhere(np.isin(scene.labels, target_ids))
    if not len(indices):
        raise ValueError('A source tumor target is required')
    points = indices@scene.affine[:3, :3].T+scene.affine[:3, 3]
    center = points.mean(axis=0)
    codes = ((points >= center)*np.array([1, 2, 4])).sum(axis=1)
    flat = np.ravel_multi_index(indices.T, scene.shape)
    zones = {str(code): frozenset(map(int, flat[codes == code])) for code in range(8)}
    evidence = {'definition': 'disjoint RAS centroid octants; equality belongs to positive side',
        'centerRasMm': center.tolist(), 'sourceTargetMm3': len(flat)*scene.voxel_mm3,
        'zones': {key: {'sourceVolumeMm3': len(cells)*scene.voxel_mm3,
            'sourceCellSha256': hashlib.sha256(
                np.asarray(sorted(cells), dtype='<i8').tobytes()).hexdigest()}
            for key, cells in zones.items()}}
    return zones, evidence


def larger_task(scene):
    _, zones = source_zones(scene)
    volume = zones['sourceTargetMm3']
    task = {'schema': 'resectionlab.staged-resection-task.v1',
        'sourceSceneId': scene.identity, 'sourceTargetMm3': volume,
        'minimumWholeTargetMm3': max(500., .1*volume),
        'minimumContributingZones': 3,
        'minimumTargetPerZoneMm3': max(50., .01*volume),
        'minimumContributingConfigurations': 2,
        'minimumTargetPerConfigurationMm3': max(100., .02*volume),
        'maximumWholeNonTargetMm3': 200.,
        'maximumContinuationNonTargetMm3': 20.,
        'maximumWholeNonTargetToTargetRatio': .2,
        'maxNativeActions': MAX_ACTIONS,
        'checkedFinalWithdrawal': True, 'exactReplayRequired': True,
        'zones': zones, 'clinicalInterpretation': 'none; engineering comparison bounds'}
    task['taskSeal'] = semantic_digest(task)
    return task


def _source_work(scene, route, max_tiles):
    target_ids = {int(row['id']) for row in scene.regions if row.get('target')}
    brain_ids = {int(row['id']) for row in scene.regions
                 if row['material'] == 'soft_tissue' and 'brain' in row['name'].lower()}
    target = np.argwhere(np.isin(scene.labels, list(target_ids)))
    points = target@scene.affine[:3, :3].T+scene.affine[:3, 3]
    origin = np.asarray(route['entryPose']['position'], float)
    inward = np.asarray(route['inwardRasUnit'], float)
    rel = points-origin
    depth = rel@inward
    radial = np.linalg.norm(rel-depth[:, None]*inward, axis=1)
    # Keep the fixed catalog shaft externally anchored. This is a library
    # scope bound, not a claim about reach with other real instruments.
    max_depth = min(24., 38.-float(route['pathLengthMm']))
    keep = (depth >= 1.) & (depth <= max_depth) & (radial <= 6.)
    points, depth, radial = points[keep], depth[keep], radial[keep]
    target = target[keep]
    if not len(points):
        return []
    # Deterministic 3 mm source lattice representatives, chosen before physics.
    bins = np.floor((points-points.min(axis=0))/3.).astype(int)
    representatives = {}
    for index, (bucket, point) in enumerate(zip(bins, points)):
        key = tuple(map(int, bucket))
        rank = (float(radial[index]), float(depth[index]), tuple(map(float, point)))
        if key not in representatives or rank < representatives[key][0]:
            representatives[key] = (rank, index)
    grid = GeometryScene(np.zeros(scene.shape, bool), scene.affine)
    tiles = []
    for _, index in sorted(representatives.values()):
        point = points[index]
        # Both straight orientation and a surface-anchored pivot are proposed;
        # native full-tool sweep checks alone establish maneuverability.
        for mode in ('parallel', 'surface-pivot'):
            axis = (inward if mode == 'parallel' else
                    (point-np.asarray(route['surfacePose']['position'], float)))
            axis = axis/np.linalg.norm(axis)
            quaternion = Rotation.align_vectors([axis], [[0., 0., 1.]])[0].as_quat()
            measurements = {}
            for radius in RADII_MM:
                footprint = capsule_voxel_indices(grid, point-axis*2., point, radius)
                cells = np.ravel_multi_index(footprint.T, scene.shape)
                values = scene.labels.ravel()[cells]
                other = (values != 0) & ~np.isin(values, list(target_ids | brain_ids))
                measurements[str(radius)] = {
                    'sourceTargetMm3': float(np.count_nonzero(np.isin(values, list(target_ids)))*scene.voxel_mm3),
                    'sourceBrainMm3': float(np.count_nonzero(np.isin(values, list(brain_ids)))*scene.voxel_mm3),
                    'sourceOtherLabelContact': bool(np.any(other))}
            if all(row['sourceOtherLabelContact'] for row in measurements.values()):
                continue
            tile = {'id': f'{mode}-{int(np.ravel_multi_index(target[index], scene.shape))}',
                'pose': {'position': point.tolist(), 'quaternion': quaternion.tolist()},
                'orientationMode': mode, 'depthFromTargetEntryMm': float(depth[index]),
                'radialFromOriginalRayMm': float(radial[index]),
                'sourceFootprints': measurements}
            tiles.append(tile)
    # Preserve the exact accepted access line first. Snapping a desired work
    # center to a nearby source voxel translates the entire shaft and is not
    # equivalent to extending an already cleared oblique corridor.
    for offset in np.arange(1., max_depth+1e-9, 1.):
        point = origin+inward*float(offset)
        index = np.rint(scene.inverse[:3, :3]@(point-scene.affine[:3, 3])).astype(int)
        if np.any(index < 0) or np.any(index >= np.asarray(scene.shape)) or \
                int(scene.labels[tuple(index)]) not in target_ids:
            continue
        measurements = {}
        for radius in RADII_MM:
            footprint = capsule_voxel_indices(grid, point-inward*2., point, radius)
            values = scene.labels[tuple(footprint.T)]
            measurements[str(radius)] = {
                'sourceTargetMm3': float(np.count_nonzero(np.isin(values, list(target_ids)))*scene.voxel_mm3),
                'sourceBrainMm3': float(np.count_nonzero(np.isin(values, list(brain_ids)))*scene.voxel_mm3),
                'sourceOtherLabelContact': bool(np.any((values != 0) &
                    ~np.isin(values, list(target_ids | brain_ids))))}
        tiles.append({'id': f'axial-{offset:g}',
            'pose': {'position': point.tolist(),
                     'quaternion': list(route['entryPose']['quaternion'])},
            'orientationMode': 'source-ray', 'depthFromTargetEntryMm': float(offset),
            'radialFromOriginalRayMm': 0., 'sourceFootprints': measurements})
    # Prioritize efficient target work, but retain axial and lateral alternatives.
    tiles.sort(key=lambda row: (
        row['orientationMode'] != 'source-ray',
        row['sourceFootprints']['2.2']['sourceBrainMm3'],
        row['radialFromOriginalRayMm'], row['depthFromTargetEntryMm'],
        -row['sourceFootprints']['2.2']['sourceTargetMm3'], row['id']))
    return tiles[:max_tiles]


def build_inventory(scene, record, *, max_routes=3, max_tiles=MAX_TILES,
                    fresh_routes=None, fresh_geometry=None, check=lambda: None):
    """Reuse sealed joint/classic routes; rank extended work from source anatomy."""
    if type(max_routes) is not int or not 1 <= max_routes <= 3 or \
            type(max_tiles) is not int or not 1 <= max_tiles <= MAX_TILES:
        raise ValueError('Keep the declared source route/tile bounds')
    historical = [row.get('sourceAccessComparison') for row in
                  record.get('jointBaselineHistory', record.get('sectorPlanningHistory', []))]
    comparisons = ([record.get('planning')]+historical if fresh_routes is None else historical[:1])
    candidates = []
    seen = set()
    for comparison in comparisons:
        if not isinstance(comparison, dict) or comparison.get('task') != 'anatomy-access':
            continue
        if comparison.get('comparisonSeal') != semantic_digest({
                key: value for key, value in comparison.items() if key != 'comparisonSeal'}):
            raise ValueError('Source access comparison seal differs')
        for access in comparison.get('alternatives', []):
            check()
            route = access.get('route', {})
            if (route.get('sourceSceneId') != scene.identity or
                    route.get('pathType') != 'oblique_mapped_brain'):
                continue
            signature = semantic_digest(route['entryPose'])
            if signature in seen:
                continue
            seen.add(signature)
            if len(candidates) >= 6:
                break
            if access['planSeal'] != semantic_digest({
                    'sourceSceneId': scene.identity, 'startStateId': access['startStateId'],
                    'decisions': access['decisions']}):
                raise ValueError('Source access plan seal differs')
            tiles = _source_work(scene, route, max_tiles)
            prefix = _prefix_actions(access)
            estimate = sum(row['sourceFootprints']['2.2']['sourceTargetMm3'] for row in tiles)
            snapshot = record.get('branches', {}).get(access.get('branchId'), {})
            state = snapshot.get('checkpoint', {}).get('state', {})
            slots = state.get('slots', {})
            checked_cleanup = bool(slots) and not state.get('holds') and all(
                not row['inserted'] and not row['active'] for row in slots.values())
            observed_brain = access.get('summary', {}).get('nonTargetRemovedMm3')
            if type(observed_brain) not in (int, float) or not math.isfinite(observed_brain):
                observed_brain = 200.
            candidates.append({'route': copy.deepcopy(route),
                'sourceMethod': access['method'], 'sourcePlanSeal': access['planSeal'],
                'sourceComparisonSeal': comparison['comparisonSeal'],
                'approachActions': copy.deepcopy(prefix), 'tiles': tiles,
                'sourceWorkContactSumMm3': estimate,
                'sealedAccessNonTargetMm3': observed_brain,
                'sealedAccessCleanupComplete': checked_cleanup,
                'rankKey': [not checked_cleanup, observed_brain,
                            -estimate, len(prefix), route['pathLengthMm'], signature]})
    for route in fresh_routes or []:
        check()
        if len(candidates) >= 6 or route['sourceSceneId'] != scene.identity:
            continue
        signature = semantic_digest(route['entryPose'])
        if signature in seen:
            continue
        seen.add(signature)
        tiles = _source_work(scene, route, max_tiles)
        estimate = sum(row['sourceFootprints']['2.2']['sourceTargetMm3'] for row in tiles)
        method = {2: 'superior-access', 0: 'axis0-access', 1: 'axis1-access'}[route['nativeAxis']]
        candidates.append({'route': copy.deepcopy(route), 'sourceMethod': method,
            'sourcePlanSeal': None, 'sourceComparisonSeal': fresh_geometry['selectionSeal'],
            'approachActions': None, 'tiles': tiles, 'sourceWorkContactSumMm3': estimate,
            'rankKey': [0, -estimate, route['pathLengthMm'], signature],
            'approachController': 'existing_native_access_from_source_route'})
    candidates.sort(key=lambda row: tuple(row['rankKey']))
    selected = []
    for candidate in candidates:
        route = candidate['route']
        novelty = _brain_route_novelty(np.asarray(route['surfacePose']['position']),
            np.asarray(route['inwardRasUnit']), [row['route'] for row in selected])
        if novelty is None:
            continue
        candidate['noveltyComparisons'] = novelty
        selected.append(candidate)
        if len(selected) == max_routes:
            break
    inventory = {'schema': INVENTORY_SCHEMA, 'sourceSceneId': scene.identity,
        'sourceLabelsSha256': array_digest(scene.labels),
        'candidateRoutesExamined': len(candidates), 'maxRoutes': max_routes,
        'maxTilesPerRoute': max_tiles, 'workingRadiiMm': list(RADII_MM),
        'shaftRadiusMm': .5, 'shaftLengthMm': 40., 'workingDepthMm': 2.,
        'sourceOnlySelection': False, 'nativeSelectionCalls': 0,
        'localManeuverRule': {'angleStepDeg': .5, 'advanceStepMm': .5,
            'maxPathSteps': 4, 'maxGeometryOptionsPerStage': 15,
            'maxNativeWorkProbesPerStage': MAX_PROBES_PER_STAGE,
            'fixedCurrentToolGeometry': True,
            'revisitPolicy': 'A failed proposal may be retried only after a newly admitted native state'},
        'maximumAdmittedStages': MAX_STAGES,
        'externalShaftWorkingReachMm': 38.,
        'motionBoundaryPolicy': 'A partial shaft-collision probe is never committed. Its observed successful pose may propose a shortened path, which requires a new complete native preview/commit; dwell uses the actual accepted pose.',
        'freshSourceGeometry': fresh_geometry,
        'ranking': 'retain sealed native access baselines; checked cleanup, lower observed non-target, then extended source work contact; no new native calls during selection',
        'commonPreparation': 'Existing access/joint generation is common prepared input; its stored native costs remain in source comparisons and are not relabeled as free source inference',
        'routes': selected,
        'scope': 'source geometry proposals; native swept tool, other hand, caps and exit remain authoritative'}
    inventory['inventorySeal'] = semantic_digest(inventory)
    return inventory


def _cleanup(engine):
    return not engine.state['holds'] and not any(
        row['inserted'] or row['active'] or row['grip']
        for row in engine.state['slots'].values())


def _zone_progress(engine, zones):
    removed = set(map(int, engine.state['removed']))
    return {key: len(cells & removed)*engine.scene.voxel_mm3 for key, cells in zones.items()}


def assess_task(engine, task, zones, configuration_target, *, exact_replay, baseline=None,
                admitted_strokes=0, legacy_sectors=None):
    values = engine.metrics()
    target, other = values['targetRemovedMm3'], values['nonTargetRemovedMm3']
    progress = _zone_progress(engine, zones)
    failures = []
    checks = {'wholeTarget': target >= task['minimumWholeTargetMm3'],
        'zones': sum(value >= task['minimumTargetPerZoneMm3'] for value in progress.values()) >= 3,
        'configurations': sum(value >= task['minimumTargetPerConfigurationMm3'] for value in configuration_target.values()) >= 2,
        'wholeNonTarget': other <= 200.,
        'continuationNonTarget': baseline is not None and other-baseline['nonTargetRemovedMm3'] <= 20.,
        'ratio': target > 0 and other/target <= .2,
        'actions': len(engine.actions) <= MAX_ACTIONS,
        'cleanup': _cleanup(engine), 'exactReplay': exact_replay is True}
    failures = [key for key, passed in checks.items() if not passed]
    legacy = None
    if baseline is not None and legacy_sectors is not None:
        removed = set(map(int, engine.state['removed']))-set(baseline['removedCells'])
        sector = {key: len(removed & cells)*engine.scene.voxel_mm3
                  for key, cells in legacy_sectors.items()}
        first_progress = []
        simultaneous_first = False
        for history in engine.history[baseline.get('nativeActions', len(engine.history)):]:
            delta = set(map(int, history.get('deltas', {}).get('removedCells', ())))
            newly_touched = [key for key in ('far', 'near')
                             if delta & legacy_sectors[key] and key not in first_progress]
            if len(newly_touched) > 1:
                simultaneous_first = True
            first_progress.extend(newly_touched)
        order_passed = first_progress == ['far', 'near'] and not simultaneous_first
        legacy = {'additionalTargetMm3': target-baseline['targetRemovedMm3'],
            'additionalNonTargetMm3': other-baseline['nonTargetRemovedMm3'],
            'perSectorRemovedMm3': sector,
            'requestedSectorOrder': ['far', 'near'],
            'observedFirstProgressOrder': first_progress, 'orderPassed': order_passed,
            'firstProgressSimultaneous': simultaneous_first,
            'taskGateMet': (target-baseline['targetRemovedMm3'] >= 30. and
                min(sector.values()) >= 10. and admitted_strokes >= 2 and
                other-baseline['nonTargetRemovedMm3'] <= 20. and other <= 200. and
                len(engine.actions) <= 48 and _cleanup(engine) and exact_replay and order_passed)}
    return {'taskGateMet': not failures, 'checks': checks, 'failedGates': failures,
        'wholeTargetRemovedMm3': target, 'wholeNonTargetRemovedMm3': other,
        'residualTargetMm3': task['sourceTargetMm3']-target,
        'sourceTargetFractionRemoved': target/task['sourceTargetMm3'],
        'wholeNonTargetToTargetRatio': other/target if target else None,
        'zoneTargetRemovedMm3': progress, 'configurationTargetRemovedMm3': configuration_target,
        'cleanupComplete': _cleanup(engine), 'legacyTask': legacy}


def _work_actions(engine, tile, radius, hand):
    """Slow short translation/rotation plus stationary aspiration; fixed shaft."""
    if tile.get('proposalStateId', engine.state_id) != engine.state_id:
        raise ValueError('Local maneuver proposal no longer matches its native state')
    start = engine.state['slots'][hand]
    path = tile.get('path', [tile['pose']])
    previous = start
    travel = 0.
    for pose in path:
        distance = float(np.linalg.norm(np.subtract(pose['position'], previous['position'])))
        angle = float((Rotation.from_quat(pose['quaternion'])*
                       Rotation.from_quat(previous['quaternion']).inv()).magnitude())
        travel += distance+42.*angle
        previous = pose
    duration = min(30., max(1., travel))
    parameters = {'radiusMm': radius, 'depthMm': float(engine.parameters(start)['depthMm']), 'forceN': 0.,
                  'power': 1., 'duty': 1., 'flowMm3PerSecond': 0.}
    motion = {'path': copy.deepcopy(path)} if 'path' in tile else {'pose': copy.deepcopy(tile['pose'])}
    return [SkillAction({'slot': hand, 'operation': 'apply',
        **motion, 'parameters': parameters,
        'duration': duration}, tuple(value for value in (2., 4.) if value > duration),
        require_endpoint=True),
        SkillAction({'slot': hand, 'operation': 'apply',
        'pose': copy.deepcopy(tile['pose']), 'parameters': parameters, 'duration': .6})]


def _current_work_contacts(engine, tile, radius):
    """Current tip-path contact proxy, without predicting a native transition.

    Source tile footprints remain provenance. They cannot tell whether a later
    action will encounter already removed tissue, nor account for its transit.
    Full-tool clearance, real removal and checked exit still require execution.
    """
    previous = np.asarray(engine.state['slots']['left']['position'])
    cells = set()
    travel = 0.
    previous_q = Rotation.from_quat(engine.state['slots']['left']['quaternion'])
    for pose in tile.get('path',[tile['pose']]):
        point = np.asarray(pose['position'])
        cells.update(map(int,engine.touched(previous,point,radius)))
        orientation = Rotation.from_quat(pose['quaternion'])
        travel += float(np.linalg.norm(point-previous))+42.*float((orientation*previous_q.inv()).magnitude())
        previous,previous_q = point,orientation
    targets = {int(row['id']) for row in engine.scene.regions if row.get('target')}
    brain = {int(row['id']) for row in engine.scene.regions
             if row['material']=='soft_tissue' and 'brain' in row['name'].lower()}
    labels = engine.scene.labels.ravel()
    return {'currentTargetContactMm3':sum(int(labels[cell]) in targets for cell in cells)*engine.scene.voxel_mm3,
        'currentBrainContactMm3':sum(int(labels[cell]) in brain for cell in cells)*engine.scene.voxel_mm3,
        'otherRetainedLabelContact':any(int(labels[cell]) not in targets|brain for cell in cells),
        'fullEnvelopeTravelProxyMm':travel,
        'scope':'retained tip-path source contact only; not native removal or swept shaft clearance'}


def _choose_work(engine, route_inventory, attempted, *, radius):
    current = np.asarray(engine.state['slots']['left']['position'])
    state = engine.state_id
    local = propose_cusa_maneuvers(engine, route_inventory['route'], max_options=15)
    # Retry a previously blocked region after accepted work changed anatomy;
    # repeated probes of an unchanged native state remain disallowed.
    candidates = []
    for tile in [*local,*route_inventory['tiles']]:
        if (tile['id'],radius,state) in attempted:
            continue
        contact = _current_work_contacts(engine,tile,radius)
        if contact['currentTargetContactMm3']<=0 or contact['otherRetainedLabelContact']:
            continue
        candidate = copy.deepcopy(tile)
        candidate.update(currentContacts=contact,contactStateId=state)
        candidates.append(candidate)
    candidates.sort(key=lambda tile: (
        tile['currentContacts']['currentBrainContactMm3'],
        -tile['currentContacts']['currentTargetContactMm3'],
        tile['currentContacts']['fullEnvelopeTravelProxyMm'],
        'proposalStateId' not in tile,
        float(np.linalg.norm(np.asarray(tile['pose']['position'])-current)), tile['id']))
    return candidates[:MAX_PROBES_PER_STAGE]


def _reconfigure_and_enter(engine, cached_exit, approach_actions, route, radius,
                           hand, limits, meter, check):
    from .surgical_staged_skills import external_cusa_reconfiguration_actions
    clean = cached_exit
    if clean is None or clean.actions[:len(engine.actions)] != engine.actions:
        from .surgical_staged_exit import find_checked_exit
        clean, exit_receipt = find_checked_exit(engine, route, hand=hand,
            limits=limits, meter=meter, check=check)
        if exit_receipt['status'] != 'accepted':
            return engine.clone(), {'status': 'refused',
                'reason': 'RECONFIGURATION_EXIT_REFUSED', 'exit': exit_receipt}
    try:
        actions = list(external_cusa_reconfiguration_actions(
            clean, hand, route['outsidePose'], radius_mm=radius))
    except ValueError as error:
        if not str(error).startswith('CHECKED_CLEANUP_REQUIRED_BEFORE_RECONFIGURATION:'):
            raise
        return engine.clone(), {'status': 'refused', 'reason': str(error),
            'priorCheckedExitStateId': clean.state_id, 'configurationRadiusMm': radius}
    for raw in approach_actions:
        if raw['operation'] == 'apply':
            action = copy.deepcopy(raw)
            action.setdefault('parameters', {})['radiusMm'] = radius
            actions.append(SkillAction(action))
    reentered, receipt = checked_actions(clean, actions, limits=limits,
        meter=meter, check=check, phase='staged.external_configuration_reentry')
    receipt['priorCheckedExitStateId'] = clean.state_id
    receipt['configurationRadiusMm'] = radius
    return reentered, receipt


def compare_staged_strategies(engine, record, *, acknowledge_unknowns=False,
                              max_physical_calls=MAX_CALLS, max_wall_seconds=MAX_SECONDS,
                              check=lambda: None, progress=lambda *_: None,
                              exercise_resume=True, explore_fresh_corridors=False,
                              initial_access_radius_mm=None,
                              include_opposite_faces=False,
                              cumulative_reentry=False):
    """Execute source-defined work with bounded lookahead and checked exit."""
    from .surgical_staged_exit import find_checked_exit
    from .surgical_joint_geometry import select_joint_corridors
    if acknowledge_unknowns is not True:
        raise ValueError('Explicit acknowledgment of unsupported anatomy is required')
    if type(cumulative_reentry) is not bool:
        raise ValueError('Cumulative re-entry mode must be boolean')
    if cumulative_reentry:
        if initial_access_radius_mm is not None or explore_fresh_corridors or include_opposite_faces:
            raise ValueError('Cumulative re-entry selects its initial configuration from saved source routes')
        from .surgical_staged_fan import compare_cumulative_reentries
        return compare_cumulative_reentries(engine, record,
            max_physical_calls=max_physical_calls, max_wall_seconds=max_wall_seconds,
            check=check, progress=progress, exercise_resume=exercise_resume)
    if type(include_opposite_faces) is not bool or include_opposite_faces and not explore_fresh_corridors:
        raise ValueError('Opposite source faces require explicit fresh corridor exploration')
    if initial_access_radius_mm is not None and (type(initial_access_radius_mm) not in (int,float) or
            initial_access_radius_mm != .6):
        raise ValueError('The optional narrow initial access preset is fixed at 0.6 mm')
    if initial_access_radius_mm is not None and 'cusaV3' in engine.scene.properties:
        raise ValueError('CUSA v3 requires its unchanged fixed catalog geometry')
    if (type(max_physical_calls) is not int or not 1 <= max_physical_calls <= MAX_CALLS or
            type(max_wall_seconds) not in (int, float) or not math.isfinite(max_wall_seconds) or
            not 0 < max_wall_seconds <= MAX_SECONDS or max_physical_calls < EXPORT_RESERVE+240):
        raise ValueError('Keep the declared staged computation budget')
    if engine.actions or engine.state_id != engine.fresh().state_id:
        raise ValueError('Staged strategies start from the common intact state')
    started = time.monotonic()
    def bounded():
        check()
        if time.monotonic()-started > max_wall_seconds:
            raise TimeoutError('Staged controller aggregate wall budget exhausted')
    if explore_fresh_corridors:
        routes, geometry = select_joint_corridors(engine.scene, goal='staged',
            include_opposite_faces=include_opposite_faces, check=bounded)
        inventory = build_inventory(engine.scene, record, fresh_routes=routes,
            fresh_geometry=geometry, check=bounded)
    else:
        # Keep the stronger measured access baseline. The optional source-only
        # staged selector has not earned replacement on the development cases.
        inventory = build_inventory(engine.scene, record, check=bounded)
    if initial_access_radius_mm is not None:
        # The existing native access executor generates a newly measured plan
        # when initial outside geometry changes. No inserted radius is edited,
        # and prior source comparisons/trajectories remain unchanged.
        inventory = copy.deepcopy(inventory)
        inventory['initialAccessRadiusMm'] = initial_access_radius_mm
        for route_inventory in inventory['routes']:
            prior = route_inventory['route']['corridorRadiusMm']
            if prior != initial_access_radius_mm:
                route_inventory['route']['corridorRadiusMm'] = initial_access_radius_mm
                route_inventory['priorSelectedGeometryRadiusMm'] = prior
                route_inventory['approachActions'] = None
                route_inventory['sourceMethod'] += '-narrow'
                route_inventory['sourcePlanSeal'] = None
                route_inventory['approachController'] = 'existing_native_access_with_initial_external_configuration'
        inventory['inventorySeal'] = semantic_digest({k:v for k,v in inventory.items() if k!='inventorySeal'})
    task = larger_task(engine.scene)
    zones, _ = source_zones(engine.scene)
    meter = TransitionMeter(max_physical_calls, check=bounded)
    limits = SkillLimits(MAX_ACTIONS, max_physical_calls)
    branches, alternatives, access_plans = [], [], []
    route_quota = (max_physical_calls-EXPORT_RESERVE)//max(1, len(inventory['routes']))
    interruption = None
    stopped_executions = []
    try:
        for route_index, route_inventory in enumerate(inventory['routes']):
            hand = 'left'
            route = route_inventory['route']
            base = meter.wrap(engine).clone()
            route_start_calls = meter.calls
            attempts, stages, configuration_target = [], [], {}
            access_plan, access_error = None, None
            approach_actions = route_inventory['approachActions']
            if approach_actions is None:
                try:
                    with meter.phase(f'staged.{route_index}.source_access', reserve_calls=EXPORT_RESERVE):
                        _, access_plan = _execute(base, {}, route_inventory['sourceMethod'],
                            'Staged source access', route, max_actions=48,
                            skull_available=False, healthy_cap=200., check=bounded)
                    approach_actions = _prefix_actions(access_plan)
                except AccessNotExecutable as error:
                    access_error = str(error)
                    approach_actions = []
            if access_error is None:
                with meter.reserve(EXPORT_RESERVE):
                    candidate, approach = checked_actions(base, approach_actions,
                        limits=limits, meter=meter, check=bounded, phase=f'staged.{route_index}.approach')
            else:
                stopped = meter.last_committed_engine_for(f'staged.{route_index}.source_access')
                candidate = base.clone() if stopped is None else meter.wrap(stopped)
                approach = {'status': 'refused', 'reason': access_error, 'attempts': []}
            plan_seal = (access_plan['planSeal'] if access_plan is not None else
                route_inventory['sourcePlanSeal'] or semantic_digest({
                    'sourceSceneId': engine.scene.identity, 'route': route,
                    'failedAccessState': candidate.state_id}))
            access_plans.append({'method': route_inventory['sourceMethod'], 'planSeal': plan_seal,
                'startStateId': engine.state_id, 'sourceSceneId': engine.scene.identity,
                'routeSeal': semantic_digest(route), 'nativeAccessPlan': access_plan})
            baseline = {**candidate.metrics(), 'removedCells': list(candidate.state['removed']),
                        'nativeActions': len(candidate.actions)}
            work_limits = SkillLimits(MAX_ACTIONS, max_physical_calls,
                max_additional_non_target_mm3=20.,
                baseline_non_target_mm3=baseline['nonTargetRemovedMm3'])
            stop_reason = approach['reason'] or 'bounded_work_inventory_exhausted'
            cached_exit = None
            admitted = 0
            attempted = set()
            configuration_attempts = set()
            resume_evidence = None
            no_admission = 0
            if approach['status'] == 'accepted':
                for stage_index in range(MAX_STAGES):
                    if (meter.remaining <= EXPORT_RESERVE+MAX_ACTIONS+120 or
                            meter.calls-route_start_calls >= route_quota-MAX_ACTIONS or
                            len(candidate.actions) >= MAX_ACTIONS-30):
                        stop_reason = 'RESERVED_FINAL_WITHDRAWAL_AND_REPLAY'
                        break
                    # Give each useful configuration a substantive contribution
                    # before switching, instead of counting cosmetic changes.
                    current_radius = float(candidate.parameters(candidate.state['slots'][hand])['radiusMm'])
                    radius = current_radius
                    if (configuration_target.get(f'radius-{current_radius:g}', 0.) >=
                            task['minimumTargetPerConfigurationMm3'] or no_admission >= 2):
                        larger = [value for value in RADII_MM if value > current_radius+1e-9]
                        radius = larger[0] if larger else current_radius
                    prepared_work = candidate
                    reconfiguration = None
                    configuration_key = (candidate.state_id, radius)
                    if abs(radius-current_radius) > 1e-9 and configuration_key not in configuration_attempts:
                        configuration_attempts.add(configuration_key)
                        with meter.reserve(EXPORT_RESERVE):
                            prepared_work, reconfiguration = _reconfigure_and_enter(candidate,
                                cached_exit, approach_actions, route, radius, hand,
                                work_limits, meter, bounded)
                        if reconfiguration['status'] != 'accepted':
                            attempts.append({'status': 'external_configuration_refused',
                                'radiusMm': radius, 'configuration': reconfiguration})
                            # Failed larger-tip setup does not suppress work
                            # still possible with the current instrument.
                            prepared_work, radius, reconfiguration = candidate, current_radius, None
                    elif abs(radius-current_radius) > 1e-9:
                        radius = current_radius
                    proposals = _choose_work(prepared_work, route_inventory, attempted, radius=radius)
                    if not proposals:
                        no_admission += 1
                        continue
                    admitted_candidates = []
                    for tile in proposals:
                        if meter.calls-route_start_calls >= route_quota-MAX_ACTIONS:
                            break
                        attempted.add((tile['id'], radius, prepared_work.state_id))
                        before = prepared_work.metrics()
                        work_start = len(prepared_work.actions)
                        with meter.reserve(EXPORT_RESERVE):
                            work, receipt = checked_cusa_motion(prepared_work, _work_actions(prepared_work, tile, radius, hand)[0],
                                limits=work_limits, meter=meter, check=bounded,
                                phase=f'staged.{route_index}.stage{stage_index}.{tile["id"]}.work')
                        row = {'tileId': tile['id'], 'radiusMm': radius, 'work': receipt,
                               'currentContacts': copy.deepcopy(tile.get('currentContacts')),
                               'contactStateId': tile.get('contactStateId'),
                               'workStartPosition': work_start,
                               'orientationMode': tile['orientationMode'],
                               'status': 'work_refused'}
                        attempts.append(row)
                        if reconfiguration is not None:
                            row['configuration'] = reconfiguration
                        if receipt['status'] != 'accepted':
                            continue
                        delta = work.metrics()['targetRemovedMm3']-before['targetRemovedMm3']
                        if delta <= 0:
                            row.update(status='no_new_target')
                            continue
                        with meter.reserve(EXPORT_RESERVE):
                            exited, exit_receipt = find_checked_exit(work, route, hand=hand,
                                limits=work_limits, meter=meter, check=bounded)
                        row['exit'] = exit_receipt
                        if exit_receipt['status'] != 'accepted':
                            row.update(status='checked_exit_refused')
                            continue
                        brain = exited.metrics()['nonTargetRemovedMm3']-before['nonTargetRemovedMm3']
                        row.update(status='complete_work_and_exit_checked',
                                   targetIncrementMm3=delta, nonTargetIncludingExitMm3=brain)
                        admitted_candidates.append(((-delta, brain, tile['id']), work, exited, row))
                    if not admitted_candidates:
                        stop_reason = 'NO_ADMITTED_WORK_WITH_CHECKED_EXIT'
                        no_admission += 1
                        continue
                    _, work, exited, selected = min(admitted_candidates, key=lambda item: item[0])
                    candidate = work
                    cached_exit = exited
                    no_admission = 0
                    admitted += 1
                    config = f'radius-{radius:g}'
                    configuration_target[config] = configuration_target.get(config, 0.)+selected['targetIncrementMm3']
                    stages.append({'stage': stage_index, 'tileId': selected['tileId'],
                        'workStartPosition': selected['workStartPosition'],
                        'configuration': config, 'stateId': candidate.state_id,
                        'nativeActions': len(candidate.actions),
                        'targetIncrementMm3': selected['targetIncrementMm3'],
                        'executionMode': selected['work']['executionMode'],
                        'requestedPoseReached': selected['work']['requestedPoseReached'],
                        'observedEndPose': selected['work']['observedEndPose'],
                        'checkedExitStateId': exited.state_id,
                        'checkedExitActions': len(exited.actions)-len(candidate.actions)})
                    if exercise_resume and resume_evidence is None:
                        checkpoint = make_skill_checkpoint(candidate,
                            source_seal=plan_seal, task_seal=task['taskSeal'],
                            inventory_seal=inventory['inventorySeal'], limits=work_limits,
                            meter=meter, stage=f'after-stage-{stage_index}')
                        with meter.reserve(EXPORT_RESERVE):
                            restored, _, ledger = restore_skill_checkpoint(engine.scene, checkpoint,
                                source_seal=plan_seal, task_seal=task['taskSeal'],
                                inventory_seal=inventory['inventorySeal'], meter=meter, check=bounded)
                        candidate = restored
                        resume_evidence = {'checkpoint': checkpoint,
                            'restoredStateId': restored.state_id, 'ledger': ledger,
                            'exactResume': restored.state_id == work.state_id}
                    progress((route_index+(stage_index+1)/MAX_STAGES)/max(1,len(inventory['routes'])),
                        f'Checked staged work {route_index+1}, stage {stage_index+1}')
            if cached_exit is not None:
                # The full selected exit was executed by the same native engine
                # from this exact accepted work state; no action is synthesized.
                if cached_exit.actions[:len(candidate.actions)] != candidate.actions:
                    raise ValueError('Cached checked exit no longer matches work ancestry')
                candidate = cached_exit
            elif not _cleanup(candidate):
                with meter.reserve(EXPORT_RESERVE):
                    candidate, final_exit = find_checked_exit(candidate, route, hand=hand,
                        limits=work_limits, meter=meter, check=bounded)
                attempts.append({'status': 'final_source_exit', 'exit': final_exit})
            with meter.phase(f'staged.{route_index}.independent_replay', reserve_calls=EXPORT_RESERVE):
                replay = meter.wrap(engine.fresh())
                boundary_peaks = {}
                for action in candidate.actions:
                    bounded()
                    result = replay.step(action, check=bounded)
                    if result['status'] == 'rejected':
                        raise ValueError('Staged native trajectory no longer replays')
                    for name, value in result['metrics'].items():
                        if (name.startswith(('brainMax', 'brainElastic', 'brainNeighbor')) or
                                name in ('thermalExposure', 'contactCellSeconds')) and \
                                type(value) in (int, float) and math.isfinite(value):
                            boundary_peaks[name] = max(boundary_peaks.get(name, value), value)
                exact = replay.state_id == candidate.state_id and replay.state == candidate.state
            if not exact:
                raise ValueError('Staged independent replay endpoint differs')
            legacy_sectors, _ = _sector_cells(engine.scene, route)
            outcome = assess_task(candidate, task, zones, configuration_target,
                exact_replay=exact, baseline=baseline, admitted_strokes=admitted,
                legacy_sectors=legacy_sectors)
            branch_id = 'staged-'+str(route_index)+'-'+route_inventory['sourceMethod']
            report = {'branchId': branch_id, 'sourceMethod': route_inventory['sourceMethod'],
                'sourceAccessPlanSeal': plan_seal, 'route': route,
                'hand': hand, 'sourceStateId': engine.state_id, 'finalStateId': candidate.state_id,
                'status': 'passed' if outcome['taskGateMet'] else 'unpassed',
                'nativeActions': len(candidate.actions), 'approach': approach,
            'approachBaseline': baseline,
                'stages': stages, 'attemptedWork': attempts, 'stopReason': stop_reason,
                'interruptionRecovery': resume_evidence, 'outcome': outcome,
                'sampledNativeEffects': {'scope': 'accepted action-boundary samples over the entire approach; not microstep or clinical injury maxima',
                    'peaks': boundary_peaks, 'simulatedSeconds': candidate.metrics()['elapsedSeconds'],
                    'instrumentTravelMm': candidate.state['travelMm']},
                'modeledOutcomes': summarize_engine_outcomes(candidate, baseline=base),
                'exactNativeReplay': exact}
            alternatives.append(report)
            branches.append((candidate.unwrap(), report))
    except (TransitionBudgetExceeded, TimeoutError) as error:
        interruption = type(error).__name__+': '+str(error)
        phase = meter.trace[-1]['phase'] if meter.trace else None
        stopped = meter.last_committed_engine_for(phase) if phase else None
        if stopped is not None:
            stopped_executions.append({'retentionRole': 'unadmitted_stopped_native_prefix',
                'reason': interruption, 'phase': phase,
                'nativeSnapshot': stopped.snapshot(), 'exactReplayVerified': False})
    report = {'schema': SCHEMA, 'sourceSceneId': engine.scene.identity,
        'sourceStateId': engine.state_id, 'sourceRole': 'DEVELOPMENT',
        'controller': 'deterministic source work inventory and native checked-stage search',
        'optimizerUpdates': 0, 'task': task, 'inventory': inventory,
        'cusaModel': ('contact-capacity-v3' if 'cusaV3' in engine.scene.properties else
                      'legacy-instantaneous-with-v2-mechanics'),
        'sourceAccessPlans': access_plans, 'alternatives': alternatives, 'cost': meter.report(),
        'interruption': interruption, 'stoppedExecutions': stopped_executions,
        'wallSeconds': time.monotonic()-started,
        'limitations': ['Missing skull, complete vessels and functional anatomy remain unavailable',
            ('Contact-capacity CUSA v3 remains an engineering model' if
             'cusaV3' in engine.scene.properties else
             'DEV scenes use legacy instantaneous CUSA removal with v2 mechanics, not contact-capacity CUSA v3'),
            'Tool size changes require checked external withdrawal and re-entry',
            'No actor or learned-policy gain claimed', 'Engineering effects are not clinical injury predictions']}
    if include_opposite_faces:
        report['limitations'].append('Entry exploration includes both source exterior face signs under the same finite geometry budget; selected directions alone do not establish successful complete strategies')
    report['comparisonSeal'] = semantic_digest(report)
    return {'branches': branches, 'report': report, 'meter': meter}


def staged_summary(report):
    """Small UI evidence; source/probe membership never becomes actor input."""
    return {'schema': 'resectionlab.staged-anatomy-planning-summary.v1',
        **{key: copy.deepcopy(report[key]) for key in (
            'sourceSceneId', 'sourceStateId', 'sourceRole', 'comparisonSeal',
            'optimizerUpdates', 'task', 'wallSeconds')},
        'cusaModel': report.get('cusaModel', 'legacy-instantaneous-with-v2-mechanics'),
        **({'controllerMode': report['controllerMode']} if 'controllerMode' in report else {}),
        'comparisonScope': ('same-route-configurations' if 'configurationAlias' in report else
                            report.get('comparisonScope','source-routes')),
        'limitations': copy.deepcopy(report.get('limitations', [])),
        **({'exportRecovery': {key: copy.deepcopy(report['exportRecovery'].get(key)) for key in (
            'additionalAuthoritativeReplayCalls','fullSourceToRecoveredExport90SecondWallCapMet')}}
           if 'exportRecovery' in report else {}),
        'sourceAccessPlans': [{key: row[key] for key in (
            'method', 'planSeal', 'startStateId', 'sourceSceneId', 'routeSeal')}
            for row in report['sourceAccessPlans']],
        'cost': {key: report['cost'][key] for key in (
            'physicalCalls', 'maxPhysicalCalls', 'previewCalls', 'directStepCalls')},
        'alternatives': [{**{key: copy.deepcopy(row[key]) for key in (
            'branchId', 'sourceMethod', 'sourceAccessPlanSeal', 'hand', 'status',
            'finalStateId', 'nativeActions', 'stopReason', 'stages', 'outcome',
            'exactNativeReplay')},
            **({'executionMetrics': copy.deepcopy(row['executionMetrics'])} if 'executionMetrics' in row else {}),
            'resumeVerified': bool(row['interruptionRecovery'] and
                                    row['interruptionRecovery']['exactResume']),
            'workAttempts': len(row['attemptedWork']),
            'exitRefusals': sum(attempt['status'] == 'checked_exit_refused'
                                for attempt in row['attemptedWork'])}
            for row in report['alternatives']]}


def validate_staged_report(scene, record, report):
    if (not isinstance(report, dict) or report.get('schema') != SCHEMA or
            report.get('sourceSceneId') != scene.identity or
            report.get('sourceRole') != 'DEVELOPMENT' or report.get('optimizerUpdates') != 0 or
            report.get('comparisonSeal') != semantic_digest({
                k: v for k, v in report.items() if k != 'comparisonSeal'})):
        raise ValueError('Staged report source or content seal differs')
    if report['task'] != larger_task(scene):
        raise ValueError('Staged task gate differs from the predeclared objective')
    cost = report['cost']
    if not (cost['physicalCalls'] == cost['previewCalls']+cost['directStepCalls'] <=
            cost['maxPhysicalCalls'] <= MAX_CALLS) or report['wallSeconds'] > MAX_SECONDS:
        raise ValueError('Staged native resource accounting differs')
    ids = [row['branchId'] for row in report['alternatives']]
    if len(ids) > 3 or len(ids) != len(set(ids)):
        raise ValueError('Staged native branch identities differ')
    zones, _ = source_zones(scene)
    fan_inventory = None
    if report.get('controllerMode') == 'cumulative-reentry':
        from .surgical_fan_inventory import SCHEMA as FAN_SCHEMA, MAX_RAY_CONFIGURATIONS
        fan_inventory = report['inventory']
        if (fan_inventory.get('schema') != FAN_SCHEMA or
                fan_inventory.get('sourceSceneId') != scene.identity or
                fan_inventory.get('inventorySeal') != semantic_digest({
                    key: value for key, value in fan_inventory.items() if key != 'inventorySeal'}) or
                not 0 <= fan_inventory['counts']['rayConfigurationsExamined'] <= MAX_RAY_CONFIGURATIONS or
                len(fan_inventory['fans']) > 3):
            raise ValueError('Cumulative source inventory binding differs')
    for row in report['alternatives']:
        snapshot = record['branches'].get(row['branchId'])
        if (not isinstance(snapshot, dict) or snapshot['sceneId'] != scene.identity or
                snapshot['stateId'] != row['finalStateId'] or
                len(snapshot['actions']) != row['nativeActions'] or
                row['exactNativeReplay'] is not True or
                row['status'] != ('passed' if row['outcome']['taskGateMet'] else 'unpassed')):
            raise ValueError('Staged native endpoint or task binding differs')
        source = next((value for value in report['sourceAccessPlans'] if
            value['method'] == row['sourceMethod'] and
            value['planSeal'] == row['sourceAccessPlanSeal']), None)
        if (source is None or source['sourceSceneId'] != scene.identity or
                source['startStateId'] != report['sourceStateId'] or
                source['routeSeal'] != semantic_digest(row['route'])):
            raise ValueError('Staged source access geometry binding differs')
        if fan_inventory is not None:
            from .surgical_staged_fan import ray_actions
            fan = next((value for value in fan_inventory['fans']
                        if value['route'] == row['route']), None)
            if fan is None or fan['fullSpecSeal'] != semantic_digest({
                    key: value for key, value in fan.items()
                    if key not in ('fullSpecSeal', 'id', 'noveltyComparisons')}):
                raise ValueError('Cumulative selected source fan differs')
            seen_rays = set()
            for stage in row['stages']:
                ray = next((value for value in fan['orderedRays'] if value['id'] == stage['tileId']), None)
                if ray is None or ray['id'] in seen_rays:
                    raise ValueError('Cumulative native stage is not a unique selected ray')
                seen_rays.add(ray['id'])
                start = stage['workStartPosition']
                expected = [value.action for value in ray_actions(ray['raySpec'])]
                if (snapshot['actions'][start:stage['nativeActions']] != expected or
                        stage['configuration'] != f'radius-{ray["raySpec"]["radiusMm"]:g}' or
                        stage.get('requestedPoseReached') is not True):
                    raise ValueError('Cumulative native complete entry differs')
        removed = set(map(int, snapshot['checkpoint']['state']['removed']))
        if 'executionMetrics' in row and row['executionMetrics'] != {
                'simulatedSeconds': snapshot['checkpoint']['state']['elapsedSeconds'],
                'instrumentTravelMm': snapshot['checkpoint']['state']['travelMm'],
                'completedReentries': len(row['stages']),
                'workingConfigurations': len(row['outcome']['configurationTargetRemovedMm3'])}:
            raise ValueError('Staged whole-approach execution metrics differ')
        source_zones_removed = {key: len(cells & removed)*scene.voxel_mm3
                                for key, cells in zones.items()}
        if source_zones_removed != row['outcome']['zoneTargetRemovedMm3']:
            raise ValueError('Staged source-zone ledger differs from native removal')
        target = sum(source_zones_removed.values())
        other = len(removed)*scene.voxel_mm3-target
        outcome = row['outcome']
        state = snapshot['checkpoint']['state']
        clean = not state['holds'] and not any(
            slot['inserted'] or slot['active'] or slot['grip']
            for slot in state['slots'].values())
        if (target != outcome['wholeTargetRemovedMm3'] or
                other != outcome['wholeNonTargetRemovedMm3'] or
                report['task']['sourceTargetMm3']-target != outcome['residualTargetMm3'] or
                target/report['task']['sourceTargetMm3'] != outcome['sourceTargetFractionRemoved'] or
                outcome['cleanupComplete'] != clean):
            raise ValueError('Staged whole effects or cleanup differ from native checkpoint')
        history = record.get('timeline', {}).get(row['branchId'], [])
        if fan_inventory is not None:
            from .surgical_staged_fan import validate_cumulative_intervals
            validate_cumulative_intervals(scene, report, row, source, fan, snapshot, history)
        if len(row['stages']) > MAX_STAGES:
            raise ValueError('Staged sequence exceeds the declared bounded library')
        config_totals = {}
        target_labels = {int(region['id']) for region in scene.regions if region.get('target')}
        previous_end = 0
        previous_stage = -1
        for stage in row['stages']:
            end = stage['nativeActions']
            start = stage.get('workStartPosition', end-2)
            if (not previous_stage < stage['stage'] < MAX_STAGES or
                    not previous_end <= start < end <= len(history) or
                    history[end-1]['stateAfter'] != stage['stateId']):
                raise ValueError('Staged work interval differs from native history')
            previous_end = end
            previous_stage = stage['stage']
            config_radius = float(stage['configuration'].removeprefix('radius-'))
            work_cells = set()
            for position in range(start, end):
                action = snapshot['actions'][position]
                if (action['operation'] != 'apply' or
                        action.get('parameters', {}).get('radiusMm') != config_radius):
                    raise ValueError('Staged configuration differs from executed tool actions')
                work_cells.update(history[position].get('deltas', {}).get('removedCells', []))
            contribution = sum(int(scene.labels.flat[cell]) in target_labels
                               for cell in work_cells)*scene.voxel_mm3
            if contribution != stage['targetIncrementMm3']:
                raise ValueError('Staged target contribution differs from native action deltas')
            if 'executionMode' in stage:
                observed_pose = stage.get('observedEndPose')
                dwell_pose = history[end-1]['action'].get('pose')
                if (stage['executionMode'] not in ('requested_motion', 'checked_collision_prefix') or
                        type(stage.get('requestedPoseReached')) is not bool or
                        not isinstance(observed_pose, dict) or not isinstance(dwell_pose, dict) or
                        dwell_pose != snapshot['actions'][end-1].get('pose') or
                        np.linalg.norm(np.subtract(observed_pose['position'], dwell_pose['position'])) > 1e-6 or
                        (Rotation.from_quat(observed_pose['quaternion'])*
                         Rotation.from_quat(dwell_pose['quaternion']).inv()).magnitude() > 1e-6):
                    raise ValueError('Staged motion endpoint differs from native stationary dwell')
            name = stage['configuration']
            config_totals[name] = config_totals.get(name, 0.)+contribution
        if config_totals != outcome['configurationTargetRemovedMm3']:
            raise ValueError('Staged configuration ledger differs from native work')
        baseline = row['approachBaseline']
        baseline_end = baseline['nativeActions']
        if (not 0 <= baseline_end <= len(history) or row['stages'] and
                baseline_end > row['stages'][0].get('workStartPosition',
                                                   row['stages'][0]['nativeActions']-2)):
            raise ValueError('Staged approach interval differs from native history')
        baseline_cells = set()
        for event in history[:baseline_end]:
            baseline_cells.update(event.get('deltas', {}).get('removedCells', []))
        baseline_target = sum(int(scene.labels.flat[cell]) in target_labels
                              for cell in baseline_cells)*scene.voxel_mm3
        if (baseline_cells != set(baseline['removedCells']) or
                baseline_target != baseline['targetRemovedMm3'] or
                len(baseline_cells)*scene.voxel_mm3-baseline_target != baseline['nonTargetRemovedMm3']):
            raise ValueError('Staged approach effects differ from native history')
        # Read-only endpoint view: measure gates again from native source
        # identities, action intervals and history rather than saved booleans.
        from types import SimpleNamespace
        view = SimpleNamespace(scene=scene, state=state, actions=snapshot['actions'],
            history=history, metrics=lambda: {'targetRemovedMm3': target,
                                             'nonTargetRemovedMm3': other})
        sectors, _ = _sector_cells(scene, row['route'])
        expected = assess_task(view, report['task'], zones, config_totals,
            exact_replay=row['exactNativeReplay'], baseline=baseline,
            admitted_strokes=len(row['stages']), legacy_sectors=sectors)
        if expected != outcome:
            raise ValueError('Staged task gates differ from native effects and intervals')
    return {'verified': True, 'branches': len(ids)}


def install_staged_comparison(record, result, scene):
    """Append native alternatives without replacing any existing comparison."""
    report = result['report']
    if 'stagedPlanning' in record or record.get('position') != 0 or \
            record['branches'][record['branchId']]['actions']:
        raise ValueError('Open an original intact source before another staged comparison')
    original_branches = copy.deepcopy(record['branches'])
    original_parents = copy.deepcopy(record.get('parents', {}))
    for candidate, row in result['branches']:
        branch_id = row['branchId']
        if branch_id in record['branches'] or not isinstance(candidate, SurgicalEngine):
            raise ValueError('Staged branch identity collides or is not native')
        record['branches'][branch_id] = candidate.snapshot()
        record.setdefault('parents', {})[branch_id] = {
            'branchId': record['branchId'], 'position': 0}
        record.setdefault('timeline', {})[branch_id] = copy.deepcopy(candidate.history)
    record['stagedPlanning'] = copy.deepcopy(report)
    record['stagedSummary'] = staged_summary(report)
    validate_staged_report(scene, record, record['stagedPlanning'])
    if any(record['branches'][key] != row for key, row in original_branches.items()) or \
            any(record['parents'].get(key) != row for key, row in original_parents.items()):
        raise ValueError('Staged installation altered original native history')
