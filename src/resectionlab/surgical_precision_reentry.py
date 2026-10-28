"""Bounded surface-anchored re-entry in the existing checked skill hierarchy.

Current source occupancy proposes two candidates. Complete native entry, positive
work and a checked exit admit a stage. No physics, actor or task gate is changed.
"""
from __future__ import annotations

import copy
import math
import time
from types import SimpleNamespace

import numpy as np
from scipy.spatial.transform import Rotation

from .core import semantic_digest
from .surgical_engine import SurgicalEngine
from .surgical_sector_planning import _cleanup_pending, _sector_cells
from .surgical_staged_exit import find_checked_exit
from .surgical_staged_planning import (
    assess_task, larger_task, source_zones, validate_staged_report)
from .surgical_staged_skills import (
    SkillAction, SkillLimits, checked_actions, external_cusa_reconfiguration_actions)
from .surgical_transition_meter import (
    MeteredEngine, TransitionBudgetExceeded, TransitionMeter)

SCHEMA = 'resectionlab.precision-continuation.v1'
MAX_CALLS = 600
MAX_SECONDS = 90.
OPTIONAL_SECONDS = 60.
MAX_STAGES = 2
RADIUS_MM = .6
DEPTH_MM = 2.
MAX_EXTERNAL_DEPTH_MM = 38.
MAX_RADIAL_MM = 6.


def _clean_source(engine, route):
    if (not isinstance(engine, (SurgicalEngine, MeteredEngine)) or
            'cusaV3' in engine.scene.properties or not engine.scene.properties.get('tissuePhysicsV2')):
        raise ValueError('Precision re-entry requires the legacy fixed narrow CUSA catalog')
    if _cleanup_pending(engine) or engine.state['slots']['left']['instrument'] != 'cusa':
        raise ValueError('Precision re-entry requires global checked cleanup and left CUSA')
    if (not isinstance(route, dict) or route.get('sourceSceneId') != engine.scene.identity or
            route.get('pathType') != 'oblique_mapped_brain'):
        raise ValueError('Precision re-entry requires a bound source brain corridor')
    from .surgical_staged_exit import _pose
    entry = np.asarray(_pose(route.get('entryPose'))['position'])
    surface = np.asarray(_pose(route.get('surfacePose'))['position'])
    outside = np.asarray(_pose(route.get('outsidePose'))['position'])
    inward = np.asarray(route.get('inwardRasUnit'), float)
    length = route.get('pathLengthMm')
    if (inward.shape != (3,) or not np.isfinite(inward).all() or
            abs(np.linalg.norm(inward)-1.) > 1e-6 or
            type(length) not in (int, float) or not math.isfinite(length) or
            not 0 < length < MAX_EXTERNAL_DEPTH_MM or
            np.linalg.norm(entry-surface-length*inward) > 1e-5 or
            np.dot(outside-surface, inward) >= 0):
        raise ValueError('Precision source ray, surface or access length differs')
    for key in ('entryPose', 'surfacePose', 'outsidePose'):
        if np.linalg.norm(Rotation.from_quat(route[key]['quaternion']).apply([0., 0., 1.])-inward) > 1e-5:
            raise ValueError('Precision source orientation is not aligned with its ray')
    return entry, surface, inward


def _remaining(engine, limits):
    if (not isinstance(limits, SkillLimits) or limits.max_native_actions > 240 or
            limits.max_public_calls > MAX_CALLS or limits.prior_public_calls != 0 or
            limits.max_additional_non_target_mm3 != 20. or
            limits.baseline_non_target_mm3 is None):
        raise ValueError('Keep the original precision continuation budget')
    other = engine.metrics()['nonTargetRemovedMm3']
    baseline = limits.baseline_non_target_mm3
    if other < baseline or other > 200. or other-baseline > 20.:
        raise ValueError('Precision source already exceeds its original approach budget')
    return min(20.-(other-baseline), limits.max_total_non_target_mm3-other)


def propose_precision_reentries(engine, route, *, limits, check=lambda: None):
    """Read-only retained-source contacts; these are not clearance certificates."""
    entry, surface, inward = _clean_source(engine, route)
    remaining = _remaining(engine, limits)
    scene = engine.scene
    labels = scene.labels.ravel()
    target_ids = {int(r['id']) for r in scene.regions
                  if r.get('target') and r['material'] == 'tumor'}
    brain_ids = {int(r['id']) for r in scene.regions
                 if r['material'] == 'soft_tissue' and 'brain' in r['name'].lower()}
    removed = set(map(int, engine.state['removed']))
    indices = np.argwhere(np.isin(scene.labels, list(target_ids)))
    points = indices@scene.affine[:3, :3].T+scene.affine[:3, 3]
    rel = points-entry
    depth = rel@inward
    radial = np.linalg.norm(rel-depth[:, None]*inward, axis=1)
    keep = ((depth >= 1.) &
            (depth <= MAX_EXTERNAL_DEPTH_MM-float(route['pathLengthMm'])+1e-9) &
            (radial <= MAX_RADIAL_MM+1e-9) &
            (np.linalg.norm(points-surface, axis=1) <= MAX_EXTERNAL_DEPTH_MM+1e-9))
    indices, points, depth, radial = indices[keep], points[keep], depth[keep], radial[keep]
    report = {'sourceSceneId': scene.identity, 'sourceStateId': engine.state_id,
        'sourceRouteSeal': semantic_digest(route), 'candidateLocations': 0,
        'rankedEligible': 0, 'remainingOriginalBrainMm3': remaining,
        'rejected': {'sourceBrainCap': 0, 'otherLabels': 0, 'noNewTarget': 0}}
    if not len(points):
        report['reason'] = 'NO_SOURCE_TARGET_IN_DECLARED_ENVELOPE'
        return [], report
    # The source lattice origin stays fixed as source cells are removed.
    bins = np.floor((points-points.min(axis=0))/3.).astype(int)
    representatives = {}
    for index, bucket in enumerate(bins):
        flat = int(np.ravel_multi_index(indices[index], scene.shape))
        if flat in removed:
            continue
        key = tuple(map(int, bucket))
        rank = (float(radial[index]), float(depth[index]), tuple(map(float, points[index])))
        if key not in representatives or rank < representatives[key][0]:
            representatives[key] = (rank, index)
    locations = {tuple(map(float, points[i])): {
        'kind': 'source-3mm-lattice', 'sourceCellId': int(np.ravel_multi_index(indices[i], scene.shape))}
        for _, i in representatives.values()}
    for offset in np.arange(1., MAX_EXTERNAL_DEPTH_MM-float(route['pathLengthMm'])+1e-9, 1.):
        point = entry+inward*offset
        index = np.rint(scene.inverse[:3, :3]@(point-scene.affine[:3, 3])).astype(int)
        if np.any(index < 0) or np.any(index >= np.asarray(scene.shape)):
            continue
        flat = int(np.ravel_multi_index(index, scene.shape))
        if int(labels[flat]) in target_ids and flat not in removed and np.linalg.norm(point-surface) <= 38.+1e-9:
            locations[tuple(map(float, point))] = {'kind': 'source-ray', 'sourceCellId': flat}
    candidates = []
    for coordinates, location in sorted(locations.items()):
        check()
        endpoint = np.array(coordinates)
        axis = endpoint-surface
        length = float(np.linalg.norm(axis))
        axis /= length
        outside = surface-8.*axis
        quaternion = Rotation.align_vectors([axis], [[0., 0., 1.]])[0].as_quat().tolist()
        # Both finite tip and complete physical shaft sweep, with current
        # deformation occupancy and original source identities retained.
        tip = set(map(int, engine.touched(outside-2.*axis, endpoint, RADIUS_MM)))
        shaft = set(map(int, engine.touched(outside-40.*axis, endpoint-2.*axis, .5)))
        cells = tip | shaft
        brain = sum(int(labels[i]) in brain_ids for i in cells)*scene.voxel_mm3
        target = sum(int(labels[i]) in target_ids for i in cells)*scene.voxel_mm3
        if any(int(labels[i]) not in brain_ids | target_ids for i in cells):
            report['rejected']['otherLabels'] += 1
            continue
        if brain > remaining+1e-9:
            report['rejected']['sourceBrainCap'] += 1
            continue
        if target <= 0:
            report['rejected']['noNewTarget'] += 1
            continue
        def pose(point):
            return {'position': point.tolist(), 'quaternion': list(quaternion)}
        candidate = {'sourceSceneId': scene.identity, 'sourceStateId': engine.state_id,
            'sourceRouteSeal': semantic_digest(route), 'endpoint': pose(endpoint),
            'outsidePose': pose(outside), 'surfacePose': pose(surface),
            'inwardRasUnit': axis.tolist(), 'externalDepthMm': length,
            'radialFromOriginalRayMm': float(np.linalg.norm(endpoint-entry-np.dot(endpoint-entry, inward)*inward)),
            'location': location, 'sourceNewBrainMm3': float(brain),
            'sourceNewTargetMm3': float(target),
            'sourceTipBrainMm3': float(sum(int(labels[i]) in brain_ids for i in tip)*scene.voxel_mm3),
            'sourceShaftBrainMm3': float(sum(int(labels[i]) in brain_ids for i in shaft)*scene.voxel_mm3),
            'sourceContactCellSha256': semantic_digest(sorted(cells))}
        candidate['id'] = 'precision-reentry-'+semantic_digest(candidate)
        candidates.append(candidate)
    candidates.sort(key=lambda r: (-r['sourceNewTargetMm3'], r['sourceNewBrainMm3'],
                                    tuple(r['endpoint']['position']), r['id']))
    report.update(candidateLocations=len(locations), rankedEligible=len(candidates),
                  rankedCandidateIds=[row['id'] for row in candidates])
    return candidates[:2], report


def _entry_actions(candidate):
    """One fixed complete entry and dwell, shared by execution and validation."""
    start, end = np.asarray(candidate['outsidePose']['position']), np.asarray(candidate['endpoint']['position'])
    distance = float(np.linalg.norm(end-start))
    count = max(1, math.ceil(distance/.5))
    path = [{'position': (start+(end-start)*(k/count)).tolist(),
             'quaternion': list(candidate['endpoint']['quaternion'])} for k in range(1, count+1)]
    parameters = {'radiusMm': RADIUS_MM, 'depthMm': DEPTH_MM, 'forceN': 0.,
                  'power': 1., 'duty': 1., 'flowMm3PerSecond': 0.}
    duration = min(30., max(1., distance/2.))
    motion = SkillAction({'slot': 'left', 'operation': 'apply', 'path': path,
        'parameters': parameters, 'duration': duration}, require_endpoint=True)
    dwell = SkillAction({'slot': 'left', 'operation': 'apply',
        'pose': copy.deepcopy(candidate['endpoint']), 'parameters': copy.deepcopy(parameters),
        'duration': .6}, require_endpoint=True)
    return motion, dwell


def execute_precision_reentry(clean, candidate, original_route, *, limits, meter,
                              check=lambda: None, phase='precision'):
    """Recompute canonical membership before executing fixed, complete entry."""
    canonical, _ = propose_precision_reentries(clean, original_route, limits=limits, check=check)
    if candidate not in canonical:
        raise ValueError('Precision candidate is not a canonical option in this bound state')
    before = clean.metrics()
    receipt = {'candidate': copy.deepcopy(candidate), 'status': 'refused',
        'reason': None, 'sourceStateId': clean.state_id,
        'sourceNativeActions': len(clean.actions)}
    actions = external_cusa_reconfiguration_actions(clean, 'left', candidate['outsidePose'],
        radius_mm=RADIUS_MM, depth_mm=DEPTH_MM)
    prepared, setup = checked_actions(clean, list(actions), limits=limits, meter=meter,
        check=check, phase=phase+'.external_configuration')
    receipt['setup'] = setup
    if setup['status'] != 'accepted':
        receipt['reason'] = 'EXTERNAL_CONFIGURATION_REFUSED'
        return prepared, clean.clone(), receipt
    motion, dwell = _entry_actions(candidate)
    work_start = len(prepared.actions)
    work, movement = checked_actions(prepared, [motion, dwell], limits=limits, meter=meter,
        check=check, phase=phase+'.complete_entry_and_dwell')
    receipt.update(motion=movement, entryWaypointCount=len(motion.action['path']), entryDurationSeconds=motion.action['duration'],
        workStartPosition=work_start, workNativeActions=len(work.actions), workEndStateId=work.state_id)
    if movement['status'] != 'accepted':
        receipt['reason'] = 'COMPLETE_ENTRY_OR_DWELL_REFUSED'
        return work, clean.clone(), receipt
    receipt.update(entryTargetIncrementMm3=work.metrics()['targetRemovedMm3']-before['targetRemovedMm3'],
        entryBrainIncrementMm3=work.metrics()['nonTargetRemovedMm3']-before['nonTargetRemovedMm3'])
    if receipt['entryTargetIncrementMm3'] <= 0:
        receipt['reason'] = 'NO_NEW_TARGET'
        return work, clean.clone(), receipt
    exit_route = copy.deepcopy(original_route)
    exit_route.update(outsidePose=copy.deepcopy(candidate['outsidePose']),
        entryPose=copy.deepcopy(candidate['endpoint']), surfacePose=copy.deepcopy(candidate['surfacePose']),
        inwardRasUnit=list(candidate['inwardRasUnit']), corridorRadiusMm=RADIUS_MM,
        pathLengthMm=candidate['externalDepthMm'])
    exited, withdrawal = find_checked_exit(work, exit_route, hand='left', limits=limits,
        meter=meter, check=check)
    receipt['exit'] = withdrawal
    if withdrawal['status'] != 'accepted' or _cleanup_pending(exited):
        receipt['reason'] = 'CHECKED_NEW_OUTSIDE_EXIT_REFUSED'
        return work, clean.clone(), receipt
    receipt.update(status='accepted', reason=None, finalStateId=exited.state_id,
        finalNativeActions=len(exited.actions), checkedExitStateId=exited.state_id,
        wholeTargetIncrementMm3=exited.metrics()['targetRemovedMm3']-before['targetRemovedMm3'],
        wholeBrainIncrementMm3=exited.metrics()['nonTargetRemovedMm3']-before['nonTargetRemovedMm3'])
    return work, exited, receipt


def _source_row(engine, record):
    report = record.get('stagedPlanning')
    validate_staged_report(engine.scene, record, report)
    rows = [row for row in report['alternatives'] if row['branchId'] == record.get('branchId')]
    if len(rows) != 1:
        raise ValueError('Select a verified staged source endpoint before precision continuation')
    row = rows[0]
    snapshot = record['branches'][row['branchId']]
    if (record.get('position') != row['nativeActions'] or engine.snapshot() != snapshot or
            len(engine.actions) >= 240 or row['hand'] != 'left'):
        raise ValueError('Precision continuation requires the complete saved staged endpoint')
    _clean_source(engine, row['route'])
    if engine.history != record.get('timeline', {}).get(row['branchId']):
        raise ValueError('Precision source history differs from the admitted native program')
    return row


def continue_precision_reentry(engine, record, *, acknowledge_unknowns=False,
                               check=lambda: None, progress=lambda *_: None):
    """One bounded continuation; original approach baseline never resets."""
    if acknowledge_unknowns is not True:
        raise ValueError('Explicit acknowledgment of unsupported anatomy is required')
    if 'precisionPlanning' in record:
        raise ValueError('This case already contains its bounded precision continuation')
    source = _source_row(engine, record)
    started = time.monotonic()
    def bounded():
        check()
        if time.monotonic()-started > MAX_SECONDS:
            raise TimeoutError('Precision controller 90 second aggregate budget exhausted')
    def optional():
        bounded()
        if time.monotonic()-started > OPTIONAL_SECONDS:
            raise TimeoutError('Precision optional work 60 second budget exhausted')
    meter = TransitionMeter(MAX_CALLS, check=bounded)
    source_snapshot = engine.snapshot()
    limits = SkillLimits(240, MAX_CALLS, max_additional_non_target_mm3=20.,
        baseline_non_target_mm3=source['approachBaseline']['nonTargetRemovedMm3'])
    _remaining(engine, limits)
    def replay(snapshot, phase):
        candidate = meter.wrap(engine.fresh())
        with meter.phase(phase):
            for action in snapshot['actions']:
                if candidate.step(action, check=bounded)['status'] == 'rejected':
                    raise ValueError('Precision native program no longer replays')
        if candidate.snapshot() != snapshot:
            raise ValueError('Precision full native checkpoint replay differs')
        return candidate
    branch = replay(source_snapshot, 'precision.source_program_cold_replay')
    stages, attempts, stopped = [], [], []
    stop_reason = 'MAX_TWO_RETARGET_STAGES'
    configs = copy.deepcopy(source['outcome']['configurationTargetRemovedMm3'])
    try:
        for stage in range(MAX_STAGES):
            reserve = len(branch.actions)+72
            if meter.remaining <= reserve+20:
                stop_reason = 'RESERVED_FINAL_REPLAY'
                break
            optional()
            candidates, selection = propose_precision_reentries(branch, source['route'],
                limits=limits, check=optional)
            stage_attempt = {'stage': stage, 'selection': selection, 'attempts': []}
            attempts.append(stage_attempt)
            admitted = False
            for rank, candidate in enumerate(candidates):
                attempt = {'rank': rank, 'candidateId': candidate['id'], 'status': 'running'}
                stage_attempt['attempts'].append(attempt)
                with meter.reserve(reserve):
                    work, exited, receipt = execute_precision_reentry(branch, candidate, source['route'],
                        limits=limits, meter=meter, check=optional,
                        phase=f'precision.stage{stage}.candidate{rank}')
                attempt.update(status=receipt['status'], receipt=receipt)
                if receipt['status'] != 'accepted':
                    attempt['unadmittedNativePrefix'] = work.snapshot()
                    continue
                configs['radius-0.6'] = configs.get('radius-0.6', 0.)+receipt['entryTargetIncrementMm3']
                stages.append({key: copy.deepcopy(receipt[key]) for key in (
                    'candidate', 'sourceStateId', 'sourceNativeActions', 'workStartPosition',
                    'workNativeActions', 'workEndStateId', 'finalNativeActions', 'finalStateId',
                    'entryTargetIncrementMm3', 'entryBrainIncrementMm3',
                    'wholeTargetIncrementMm3', 'wholeBrainIncrementMm3', 'checkedExitStateId')})
                stages[-1].update(stage=stage, configuration='radius-0.6', nativeActions=len(work.actions))
                branch, admitted = exited, True
                progress((stage+1)/MAX_STAGES, 'Checked precision re-entry and withdrawal')
                break
            if not admitted:
                stop_reason = 'NO_ADMITTED_NEW_REENTRY'
                break
    except (TimeoutError, TransitionBudgetExceeded) as error:
        stop_reason = type(error).__name__+': '+str(error)
    except Exception as error:
        # Keep the last admitted, clean program and every speculative commit.
        # Unexpected optional controller errors cannot promote partial work.
        stop_reason = 'OPTIONAL_CONTROLLER_ERROR: '+type(error).__name__+': '+str(error)
    finally:
        # Exceptions cannot erase committed speculative prefixes or their cost.
        # Only final accepted branches become playable surgical alternatives.
        accepted_seal = semantic_digest(branch.actions)
        for key in meter.report()['branches']:
            prefix = meter.last_committed_engine(branch_id=int(key))
            if prefix is not None and prefix.actions != branch.actions[:len(prefix.actions)]:
                stopped.append({'retentionRole': 'unadmitted_speculative_native_prefix',
                    'meterBranchId': int(key), 'nativeSnapshot': prefix.snapshot(),
                    'acceptedProgramSeal': accepted_seal})
    final = replay(branch.snapshot(), 'precision.final_program_cold_replay')
    zones, _ = source_zones(engine.scene)
    sectors, _ = _sector_cells(engine.scene, source['route'])
    outcome = assess_task(final, larger_task(engine.scene), zones, configs,
        exact_replay=True, baseline=source['approachBaseline'],
        admitted_strokes=len(source['stages'])+len(stages), legacy_sectors=sectors)
    final_id = ('precision-reentry-'+semantic_digest({'source': source['branchId'],
        'comparison': record['stagedPlanning']['comparisonSeal']})[:16] if stages else source['branchId'])
    if stages and final_id in record['branches']:
        raise ValueError('Precision branch identity collides')
    report = {'schema': SCHEMA, 'sourceSceneId': engine.scene.identity,
        'sourceRole': 'DEVELOPMENT', 'sourceBranchId': source['branchId'],
        'sourceStateId': engine.state_id, 'sourceNativeActions': len(engine.actions),
        'sourceProgramSeal': semantic_digest(source_snapshot),
        'sourceComparisonSeal': record['stagedPlanning']['comparisonSeal'],
        'sourceRowSeal': semantic_digest(source), 'sourceRouteSeal': semantic_digest(source['route']),
        'originalBaselineNonTargetMm3': limits.baseline_non_target_mm3,
        'sourceOutcome': copy.deepcopy(source['outcome']), 'outcome': outcome,
        'finalBranchId': final_id, 'finalStateId': final.state_id,
        'finalNativeActions': len(final.actions), 'newStageCount': len(stages), 'stages': stages,
        'additionalTargetMm3': outcome['wholeTargetRemovedMm3']-source['outcome']['wholeTargetRemovedMm3'],
        'additionalNonTargetMm3': outcome['wholeNonTargetRemovedMm3']-source['outcome']['wholeNonTargetRemovedMm3'],
        'attemptedStages': attempts, 'stoppedExecutions': stopped, 'stopReason': stop_reason,
        'controller': 'bounded anatomy-aware deterministic surface-anchored precision re-entry',
        'optimizerUpdates': 0, 'exactNativeReplay': True,
        'cusaModel': 'legacy-instantaneous-with-v2-mechanics',
        'cost': meter.report(), 'wallSeconds': time.monotonic()-started,
        'costScope': 'Additional controller calls including source and final cold replay; prior planning, desktop admission and saved-file reopen have separate costs',
        'limitations': ['Missing skull, complete vessels and functional mapping remain unavailable',
            'Engineering removal and mechanics are not clinical injury or device predictions',
            'Two candidates per stage and two admitted stages within the original 38 mm depth / 6 mm radial source envelope; STOP is not global infeasibility',
            'Fixed 0.6 mm CUSA tip configured outside; complete entry and checked withdrawal required',
            'The original approach baseline, failed gates and earlier study qualification remain unchanged',
            'No learned-policy benefit or matched-controller superiority claimed']}
    report['comparisonSeal'] = semantic_digest(report)
    return {'engine': final.unwrap(), 'report': report, 'meter': meter}


def precision_summary(report):
    hidden = {'attemptedStages', 'stoppedExecutions', 'sourceProgramSeal', 'sourceRowSeal',
              'sourceRouteSeal', 'controller'}
    result = {k: copy.deepcopy(v) for k, v in report.items() if k not in hidden | {'cost'}}
    result['schema'] = 'resectionlab.precision-continuation-summary.v1'
    result['cost'] = {k: report['cost'][k] for k in (
        'physicalCalls', 'maxPhysicalCalls', 'previewCalls', 'directStepCalls')}
    return result


def validate_precision_report(scene, record, report):
    """Recompute effects/gates from native ancestry; no replay during display."""
    if (not isinstance(report, dict) or report.get('schema') != SCHEMA or
            report.get('sourceSceneId') != scene.identity or report.get('sourceRole') != 'DEVELOPMENT' or
            report.get('comparisonSeal') != semantic_digest({k: v for k, v in report.items() if k != 'comparisonSeal'}) or
            report.get('optimizerUpdates') != 0 or report.get('exactNativeReplay') is not True or
            'cusaV3' in scene.properties):
        raise ValueError('Precision report source, controller or content differs')
    staged = record.get('stagedPlanning')
    validate_staged_report(scene, record, staged)
    source = next((row for row in staged['alternatives'] if row['branchId'] == report['sourceBranchId']), None)
    if (source is None or report['sourceComparisonSeal'] != staged['comparisonSeal'] or
            report['sourceRowSeal'] != semantic_digest(source) or
            report['sourceRouteSeal'] != semantic_digest(source['route']) or
            report['originalBaselineNonTargetMm3'] != source['approachBaseline']['nonTargetRemovedMm3'] or
            report['sourceOutcome'] != source['outcome']):
        raise ValueError('Precision original staged baseline differs')
    original = record['branches'][source['branchId']]
    final = record['branches'].get(report['finalBranchId'])
    if (not isinstance(final, dict) or report['sourceProgramSeal'] != semantic_digest(original) or
            report['sourceStateId'] != original['stateId'] or
            report['sourceNativeActions'] != len(original['actions']) or
            final.get('sceneId') != scene.identity or final['stateId'] != report['finalStateId'] or
            len(final['actions']) != report['finalNativeActions'] or
            final['actions'][:len(original['actions'])] != original['actions']):
        raise ValueError('Precision exact native ancestry or endpoint differs')
    stages = report['stages']
    if not 0 <= len(stages) == report['newStageCount'] <= MAX_STAGES:
        raise ValueError('Precision stage bound differs')
    if (stages and record.get('parents', {}).get(report['finalBranchId']) != {
            'branchId': source['branchId'], 'position': len(original['actions'])} or
            not stages and final != original):
        raise ValueError('Precision branch parent differs')
    history = record.get('timeline', {}).get(report['finalBranchId'], [])
    if (len(history) != len(final['actions']) or
            history[:len(original['actions'])] != record['timeline'][source['branchId']]):
        raise ValueError('Precision native history ancestry differs')
    previous = staged['sourceStateId']
    removed = set()
    for action, event in zip(final['actions'], history):
        if event['action'] != action or event['stateBefore'] != previous:
            raise ValueError('Precision native action/history chain differs')
        previous = event['stateAfter']
        removed.update(map(int, event.get('deltas', {}).get('removedCells', [])))
    state = final['checkpoint']['state']
    for snapshot in (original, final):
        identity = {'scene': scene.identity, 'state': snapshot['checkpoint']['state']}
        if 'protectionContract' in snapshot:
            identity['protectionContract'] = snapshot['protectionContract']['contentHash']
        if (snapshot['checkpoint']['position'] != len(snapshot['actions']) or
                snapshot['checkpoint']['stateId'] != snapshot['stateId'] or
                semantic_digest(identity) != snapshot['stateId'] or
                snapshot.get('protectionContract') != original.get('protectionContract')):
            raise ValueError('Precision full native checkpoint identity differs')
    if previous != final['stateId'] or removed != set(state['removed']):
        raise ValueError('Precision history removal ledger differs from checkpoint')
    target_ids = {int(r['id']) for r in scene.regions if r.get('target') and r['material'] == 'tumor'}
    def target_volume(cells):
        return sum(int(scene.labels.flat[cell]) in target_ids for cell in cells)*scene.voxel_mm3
    configs = copy.deepcopy(source['outcome']['configurationTargetRemovedMm3'])
    end = len(original['actions'])
    for index, stage in enumerate(stages):
        start, work_end, exit_end = stage['workStartPosition'], stage['nativeActions'], stage['finalNativeActions']
        candidate = stage['candidate']
        endpoint = candidate['endpoint']
        axis = np.asarray(candidate['inwardRasUnit'])
        anchor = np.asarray(source['route']['surfacePose']['position'])
        point = np.asarray(endpoint['position'])
        origin = np.asarray(source['route']['entryPose']['position'])
        inward = np.asarray(source['route']['inwardRasUnit'])
        rel = point-origin
        if (stage['stage'] != index or stage['configuration'] != 'radius-0.6' or
                stage['sourceNativeActions'] != end or not end < start < work_end < exit_end <= len(history) or
                work_end-start != 2 or start-end != 2 or stage['workNativeActions'] != work_end or
                candidate['sourceSceneId'] != scene.identity or candidate['sourceRouteSeal'] != report['sourceRouteSeal'] or
                candidate['sourceStateId'] != history[end]['stateBefore'] or
                stage['sourceStateId'] != candidate['sourceStateId'] or
                np.linalg.norm(point-anchor) > 38.+1e-9 or np.dot(rel, inward) < 1.-1e-9 or
                np.dot(rel, inward) > 38.-source['route']['pathLengthMm']+1e-9 or
                np.linalg.norm(rel-np.dot(rel, inward)*inward) > 6.+1e-9 or
                np.linalg.norm(point-anchor-axis*candidate['externalDepthMm']) > 1e-6 or
                abs(np.linalg.norm(axis)-1.) > 1e-6 or
                np.linalg.norm(np.asarray(candidate['outsidePose']['position'])-(anchor-8.*axis)) > 1e-6 or
                stage['workEndStateId'] != history[work_end-1]['stateAfter'] or
                stage['finalStateId'] != history[exit_end-1]['stateAfter'] or
                stage['checkedExitStateId'] != stage['finalStateId']):
            raise ValueError('Precision stage geometry, state or native interval differs')
        if (candidate['id'] != 'precision-reentry-'+semantic_digest({k: v for k, v in candidate.items() if k != 'id'}) or
                candidate['surfacePose']['position'] != source['route']['surfacePose']['position']):
            raise ValueError('Precision candidate content seal or source surface differs')
        for pose in (candidate['endpoint'], candidate['outsidePose'], candidate['surfacePose']):
            from .surgical_staged_exit import _pose
            _pose(pose)
            if np.linalg.norm(Rotation.from_quat(pose['quaternion']).apply([0., 0., 1.])-axis) > 1e-6:
                raise ValueError('Precision candidate orientation differs from its ray')
        entry, dwell = final['actions'][start:work_end]
        expected_entry, expected_dwell = _entry_actions(candidate)
        expected_setup = [
            {'slot': 'left', 'operation': 'exchange', 'instrument': 'cusa', 'duration': .2},
            {'slot': 'left', 'operation': 'place', 'pose': candidate['outsidePose'],
             'parameters': {'radiusMm': .6, 'depthMm': 2.}, 'duration': .2}]
        if (final['actions'][end:start] != expected_setup or entry != expected_entry.action or
                dwell != expected_dwell.action or
                any(history[p].get('status') != 'accepted' for p in range(end, exit_end)) or
                final['actions'][exit_end-1].get('operation') != 'withdraw' or
                final['actions'][exit_end-1].get('pose') != candidate['outsidePose']):
            raise ValueError('Precision fixed setup, full entry or checked exit differs')
        for action in final['actions'][work_end:exit_end]:
            if (action.get('slot') != 'left' or action['operation'] not in ('apply', 'deactivate', 'move', 'withdraw') or
                    any(name in action.get('parameters', {}) and action['parameters'][name] != value
                        for name, value in (('radiusMm', .6), ('depthMm', 2.)))):
                raise ValueError('Precision withdrawal changed fixed tool geometry')
        if (entry.get('operation') != 'apply' or dwell.get('operation') != 'apply' or
                entry.get('slot') != 'left' or dwell.get('slot') != 'left' or
                entry.get('path', [None])[-1] != endpoint or dwell.get('pose') != endpoint or
                any(a.get('parameters', {}).get('radiusMm') != RADIUS_MM or
                    a.get('parameters', {}).get('depthMm') != DEPTH_MM for a in (entry, dwell)) or
                any(history[p].get('status') != 'accepted' for p in range(start, work_end))):
            raise ValueError('Precision complete native entry/configuration differs')
        cells = set().union(*(set(e.get('deltas', {}).get('removedCells', [])) for e in history[start:work_end]))
        contribution = target_volume(cells)
        if (contribution <= 0 or contribution != stage['entryTargetIncrementMm3'] or
                len(cells)*scene.voxel_mm3-contribution != stage['entryBrainIncrementMm3']):
            raise ValueError('Precision actual target work contribution differs')
        whole_cells = set().union(*(set(e.get('deltas', {}).get('removedCells', [])) for e in history[end:exit_end]))
        whole_target = target_volume(whole_cells)
        if (whole_target != stage['wholeTargetIncrementMm3'] or
                len(whole_cells)*scene.voxel_mm3-whole_target != stage['wholeBrainIncrementMm3']):
            raise ValueError('Precision whole-stage effects differ')
        configs['radius-0.6'] = configs.get('radius-0.6', 0.)+contribution
        end = exit_end
    if end != len(final['actions']):
        raise ValueError('Precision final branch has unassigned new actions')
    target = target_volume(removed)
    other = len(removed)*scene.voxel_mm3-target
    view = SimpleNamespace(scene=scene, state=state, actions=final['actions'], history=history,
        metrics=lambda: {'targetRemovedMm3': target, 'nonTargetRemovedMm3': other})
    zones, _ = source_zones(scene)
    sectors, _ = _sector_cells(scene, source['route'])
    expected = assess_task(view, larger_task(scene), zones, configs, exact_replay=True,
        baseline=source['approachBaseline'], admitted_strokes=len(source['stages'])+len(stages), legacy_sectors=sectors)
    if (expected != report['outcome'] or not expected['cleanupComplete'] or
            _cleanup_pending(view) or
            report['additionalTargetMm3'] != target-source['outcome']['wholeTargetRemovedMm3'] or
            report['additionalNonTargetMm3'] != other-source['outcome']['wholeNonTargetRemovedMm3']):
        raise ValueError('Precision task gates or whole-approach effects differ')
    cost = report['cost']
    if (not 0 <= cost['physicalCalls'] == cost['previewCalls']+cost['directStepCalls'] <= cost['maxPhysicalCalls'] == MAX_CALLS or
            not 0 <= report['wallSeconds'] <= MAX_SECONDS):
        raise ValueError('Precision native computation budget differs')
    trace = cost.get('trace', [])
    if (len(trace) != cost['physicalCalls'] or
            [row['callIndex'] for row in trace] != list(range(1, len(trace)+1)) or
            sum(row['kind'] == 'preview' for row in trace) != cost['previewCalls'] or
            sum(row['kind'] == 'step' for row in trace) != cost['directStepCalls'] or
            sum(row['phase'] == 'precision.source_program_cold_replay' and row['kind'] == 'step'
                for row in trace) != report['sourceNativeActions'] or
            sum(row['phase'] == 'precision.final_program_cold_replay' and row['kind'] == 'step'
                for row in trace) != report['finalNativeActions']):
        raise ValueError('Precision native transition trace or mandatory replay cost differs')
    return {'verified': True, 'stages': len(stages)}


def install_precision_continuation(record, result, scene):
    report, engine = result['report'], result['engine']
    if 'precisionPlanning' in record:
        raise ValueError('Precision continuation already recorded')
    branch_id = report['finalBranchId']
    if report['newStageCount']:
        if branch_id in record['branches']:
            raise ValueError('Precision native branch identity collides')
        record['branches'][branch_id] = engine.snapshot()
        record.setdefault('timeline', {})[branch_id] = copy.deepcopy(engine.history)
        record.setdefault('parents', {})[branch_id] = {
            'branchId': report['sourceBranchId'], 'position': report['sourceNativeActions']}
    record['precisionPlanning'] = copy.deepcopy(report)
    validate_precision_report(scene, record, report)
