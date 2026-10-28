"""Ordered external re-entry in the existing staged native skill hierarchy.

Source contact selects a finite sequence. It never admits removal, a collision
prefix, or a successful strategy. Every retained stage completes its full entry
and a checked exit under the original actual access baseline.
"""
from __future__ import annotations

import copy
import math
import time

import numpy as np

from .core import semantic_digest
from .surgical_access_planning import _execute_brain_corridor
from .surgical_sector_planning import _cleanup_pending, _sector_cells
from .surgical_staged_exit import find_checked_exit
from .surgical_staged_skills import (
    SkillAction, SkillLimits, checked_actions,
    external_cusa_reconfiguration_actions, make_skill_checkpoint,
    restore_skill_checkpoint)
from .surgical_transition_meter import TransitionMeter, TransitionBudgetExceeded


def source_routes(scene, record):
    """Use only sealed access comparisons already belonging to this source."""
    from .surgical_planning import validate_saved_comparison
    comparisons = [record.get('planning'), record.get('sectorSourceAccessComparison'),
                   (record.get('jointPlanning') or {}).get('accessComparison')]
    for key in ('sectorPlanningHistory', 'jointBaselineHistory'):
        comparisons.extend(row.get('sourceAccessComparison') for row in record.get(key, []))
    routes, seen_comparisons, seen_routes = [], set(), set()
    for comparison in comparisons:
        if comparison is None:
            continue
        seal = comparison.get('comparisonSeal')
        if seal in seen_comparisons:
            continue
        validate_saved_comparison(scene, record, comparison)
        seen_comparisons.add(seal)
        for plan in comparison['alternatives']:
            route = plan['route']
            if route.get('pathType') != 'oblique_mapped_brain':
                continue
            identity = semantic_digest({key: route[key] for key in
                ('surfacePose', 'entryPose', 'inwardRasUnit')})
            if identity not in seen_routes:
                routes.append(copy.deepcopy(route))
                seen_routes.add(identity)
    if not routes or len(routes) > 12:
        raise ValueError('Cumulative re-entry needs at most twelve sealed source brain routes')
    return routes


def retained_ray_gate(engine, ray, baseline):
    """Rebind a proposed ray to current occupancy and the ORIGINAL access cost."""
    spec = ray['raySpec']
    start = np.asarray(spec['outsidePose']['position'])
    end = np.asarray(spec['endpointPose']['position'])
    axis = np.asarray(spec['inwardRasUnit'])
    radius = spec['radiusMm']
    labels = engine.scene.labels.ravel()
    targets = {r['id'] for r in engine.scene.regions if r.get('target')}
    brain = {r['id'] for r in engine.scene.regions
             if r['material'] == 'soft_tissue' and 'brain' in r['name'].lower()}
    tip = set(engine.touched(start, end, radius))
    shaft = set(engine.touched(start-40.*axis, end-2.*axis, .5))
    target = sum(int(labels[i]) in targets for i in tip)*engine.scene.voxel_mm3
    non_target = sum(int(labels[i]) in brain for i in tip)*engine.scene.voxel_mm3
    remaining = min(200.-engine.metrics()['nonTargetRemovedMm3'],
        20.-(engine.metrics()['nonTargetRemovedMm3']-baseline['nonTargetRemovedMm3']))
    reason = None
    if _cleanup_pending(engine):
        reason = 'SOURCE_NOT_GLOBALLY_CLEAN'
    elif engine.touched(start-40.*axis, start, max(radius, .5)):
        reason = 'CURRENT_OUTSIDE_OCCUPIED'
    elif any(int(labels[i]) not in targets | brain for i in tip | shaft):
        reason = 'CURRENT_UNSUPPORTED_LABELS'
    elif target <= 0:
        reason = 'NO_CURRENT_RETAINED_TARGET'
    elif non_target > remaining+1e-9:
        reason = 'CURRENT_ORIGINAL_BRAIN_ALLOWANCE'
    return {'stateId': engine.state_id, 'eligible': reason is None, 'reason': reason,
        'retainedTipTargetMm3': target, 'retainedTipBrainMm3': non_target,
        'remainingOriginalBrainMm3': remaining,
        'sourceContactSeal': semantic_digest(sorted(tip | shaft)),
        'scope': 'current source proposal; native temporal shaft clearance remains unverified'}


def _baseline(scene, engine, position):
    cells = set(i for event in engine.history[:position]
                for i in event.get('deltas', {}).get('removedCells', []))
    targets = {r['id'] for r in scene.regions if r.get('target')}
    target = sum(int(scene.labels.flat[i]) in targets for i in cells)*scene.voxel_mm3
    return {'nativeActions': position, 'removedCells': sorted(cells),
            'targetRemovedMm3': target,
            'nonTargetRemovedMm3': len(cells)*scene.voxel_mm3-target}


def ray_actions(spec):
    """The same complete entry/dwell program is used for execution and admission."""
    parameters = {'radiusMm': spec['radiusMm'], 'depthMm': 2.,
        'forceN': 0., 'power': 1., 'duty': 1., 'flowMm3PerSecond': 0.}
    return (
        SkillAction({'slot': 'left', 'operation': 'apply',
            'path': copy.deepcopy(spec['entryPath']), 'parameters': parameters,
            'duration': spec['entryDurationSeconds']}, require_endpoint=True),
        SkillAction({'slot': 'left', 'operation': 'apply',
            'pose': copy.deepcopy(spec['endpointPose']), 'parameters': parameters,
            'duration': spec['dwellDurationSeconds']}, require_endpoint=True))


def validate_cumulative_intervals(scene, report, row, source, fan, snapshot, history):
    """Bind original access and every checked exit to the actual native program."""
    actions = snapshot['actions']
    if len(history) != len(actions):
        raise ValueError('Cumulative native history unavailable')
    phases = row['approach']['phases']
    initial = next((p for p in phases if p['id'] == 'mapped-brain-corridor'), None)
    if initial is None or initial['startPosition'] != 0:
        raise ValueError('Cumulative original access phase differs')
    end = initial['endPosition']
    if type(end) is not int or not 0 <= end <= len(actions):
        raise ValueError('Cumulative original access interval differs')
    prefix = actions[:end]
    if any(a['slot'] != 'left' or a['operation'] != (
            'exchange' if index == 0 else 'place' if index == 1 else 'apply')
            for index, a in enumerate(prefix)):
        raise ValueError('Cumulative original access includes later recovery or re-entry')
    if any(event['status'] != 'accepted' for event in history[:end]):
        raise ValueError('Cumulative original access was not fully admitted')
    last_apply = max((i+1 for i, a in enumerate(prefix) if a['operation'] == 'apply'), default=0)
    if row['approachBaseline']['nativeActions'] != last_apply:
        raise ValueError('Cumulative original access baseline moved')
    if len(prefix) > 1 and (prefix[1]['pose'] != row['route']['outsidePose'] or
            prefix[1].get('parameters') != {'radiusMm': fan['initialRadiusMm'], 'forceN': 0.}):
        raise ValueError('Cumulative original outside configuration differs')
    targets = {r['id'] for r in scene.regions if r.get('target')}
    removed = set()
    first_progress = None
    for position, event in enumerate(history[:end], 1):
        removed.update(event.get('deltas', {}).get('removedCells', []))
        if position >= 4 and sum(int(scene.labels.flat[i]) in targets for i in removed)*scene.voxel_mm3 >= 5.:
            first_progress = position
            break
    if first_progress is not None and first_progress != last_apply:
        raise ValueError('Cumulative original access passed its first native target stop')
    plan_seal = semantic_digest({'sourceSceneId': scene.identity,
        'startStateId': report['sourceStateId'], 'route': row['route'],
        'nativeAccessActions': actions[:last_apply],
        'nativeAccessStateId': history[last_apply-1]['stateAfter'] if last_apply else report['sourceStateId']})
    if source['planSeal'] != plan_seal:
        raise ValueError('Cumulative original access plan seal differs')
    for index, stage in enumerate(row['stages']):
        work_end = stage['nativeActions']
        count = stage['checkedExitActions']
        exit_end = work_end+count
        next_start = (row['stages'][index+1]['workStartPosition']
                      if index+1 < len(row['stages']) else len(actions))
        if (type(count) is not int or count < 2 or
                not work_end < exit_end <= next_start <= len(history) or
                history[exit_end-1]['stateAfter'] != stage['checkedExitStateId'] or
                history[exit_end-1]['status'] != 'accepted' or
                actions[exit_end-1]['operation'] != 'withdraw'):
            raise ValueError('Cumulative checked exit interval differs')
        ray = next(value for value in fan['orderedRays'] if value['id'] == stage['tileId'])
        if (actions[exit_end-1].get('pose') != ray['raySpec']['outsidePose'] or
                any(a['slot'] != 'left' for a in actions[work_end:exit_end])):
            raise ValueError('Cumulative checked exit outside pose differs')


def compare_cumulative_reentries(engine, record, *, max_physical_calls,
        max_wall_seconds, check=lambda: None, progress=lambda *_: None,
        exercise_resume=True):
    from .surgical_fan_inventory import propose_source_fans
    from .surgical_staged_planning import (
        SCHEMA, MAX_ACTIONS, MAX_CALLS, MAX_SECONDS, EXPORT_RESERVE,
        assess_task, larger_task, source_zones)
    from .surgical_modeled_outcomes import summarize_engine_outcomes
    if ('cusaV3' in engine.scene.properties or not engine.scene.properties.get('tissuePhysicsV2') or engine.actions or
            engine.state_id != engine.fresh().state_id):
        raise ValueError('Cumulative re-entry requires an intact legacy CUSA source with V2 mechanics')
    if (type(max_physical_calls) is not int or not EXPORT_RESERVE+240 <= max_physical_calls <= MAX_CALLS or
            type(max_wall_seconds) not in (int, float) or
            not math.isfinite(max_wall_seconds) or not 0 < max_wall_seconds <= MAX_SECONDS):
        raise ValueError('Keep the declared staged computation budget')
    started = time.monotonic()
    def bounded():
        check()
        if time.monotonic()-started > max_wall_seconds:
            raise TimeoutError('Cumulative staged controller wall budget exhausted')
    inventory = propose_source_fans(engine.scene, source_routes(engine.scene, record), check=bounded)
    task = larger_task(engine.scene)
    zones, _ = source_zones(engine.scene)
    meter = TransitionMeter(max_physical_calls, check=bounded)
    # Keep the caller's export allowance after one full internal replay of
    # ANY task-bounded candidate, including an unusually long checked exit.
    work_reserve = EXPORT_RESERVE+MAX_ACTIONS
    branches, alternatives, access_plans, stopped = [], [], [], []
    unexecuted = []
    quota = (max_physical_calls-EXPORT_RESERVE)//max(1, len(inventory['fans']))
    interruption = None
    try:
        for index, fan in enumerate(inventory['fans']):
            bounded()
            if meter.remaining <= work_reserve+40:
                unexecuted = [{'fanId': value['id'], 'reason': 'RESERVED_EXPORT_AND_INTERNAL_REPLAY'}
                              for value in inventory['fans'][index:]]
                break
            route = copy.deepcopy(fan['route'])
            route['corridorRadiusMm'] = fan['initialRadiusMm']
            method = f'cumulative-axis{route["nativeAxis"]}'
            initial = meter.wrap(engine).clone()
            start_calls = meter.calls
            stages, attempts, configurations = [], [], {}
            resume = None
            with meter.reserve(work_reserve):
                access, approach = _execute_brain_corridor(initial, route,
                    max_actions=80, healthy_cap=200., check=bounded)
            last_apply = max((i+1 for i, a in enumerate(access.actions)
                              if a['operation'] == 'apply'), default=0)
            baseline = _baseline(engine.scene, access, last_apply)
            limits = SkillLimits(MAX_ACTIONS, max_physical_calls,
                max_additional_non_target_mm3=20.,
                baseline_non_target_mm3=baseline['nonTargetRemovedMm3'])
            plan_seal = semantic_digest({'sourceSceneId': engine.scene.identity,
                'startStateId': engine.state_id, 'route': route,
                'nativeAccessActions': access.actions[:last_apply],
                'nativeAccessStateId': access.history[last_apply-1]['stateAfter'] if last_apply else engine.state_id})
            access_plans.append({'method': method, 'planSeal': plan_seal,
                'sourceSceneId': engine.scene.identity, 'startStateId': engine.state_id,
                'routeSeal': semantic_digest(route), 'nativeAccessPlan': None})
            kept = access
            if _cleanup_pending(kept):
                with meter.reserve(work_reserve):
                    clean, withdrawal = find_checked_exit(kept, route, hand='left',
                        limits=limits, meter=meter, check=bounded)
                attempts.append({'status': 'initial_checked_exit', 'exit': withdrawal})
                if withdrawal['status'] == 'accepted' and not _cleanup_pending(clean):
                    kept = clean
            access_complete = any(p['id'] == 'corridor-withdrawal' for p in approach['phases'])
            stop_reason = 'SOURCE_FAN_EXHAUSTED'
            if not access_complete or _cleanup_pending(kept):
                stop_reason = 'INITIAL_ACCESS_OR_CLEANUP_UNPASSED'
            else:
                for stage_index, ray in enumerate(fan['orderedRays']):
                    bounded()
                    if (len(kept.actions) >= 210 or meter.calls-start_calls >= quota or
                            meter.remaining <= work_reserve+40):
                        stop_reason = 'RESERVED_NATIVE_REPLAY_OR_ROUTE_BUDGET'
                        break
                    gate = retained_ray_gate(kept, ray, baseline)
                    attempt = {'tileId': ray['id'], 'stage': stage_index,
                        'sourceStateId': kept.state_id, 'sourceGate': gate,
                        'status': 'source_gate_refused', 'reason': gate['reason']}
                    attempts.append(attempt)
                    if not gate['eligible']:
                        continue
                    spec = ray['raySpec']
                    with meter.reserve(work_reserve):
                        setup = external_cusa_reconfiguration_actions(kept, 'left',
                            spec['outsidePose'], radius_mm=spec['radiusMm'], depth_mm=2.)
                        prepared, setup_receipt = checked_actions(kept, setup, limits=limits,
                            meter=meter, check=bounded, phase=f'fan.{index}.{stage_index}.external_configuration')
                        attempt['configuration'] = setup_receipt
                        if setup_receipt['status'] != 'accepted':
                            attempt.update(status='work_refused', reason='OUTSIDE_CONFIGURATION_REFUSED')
                            continue
                        before = prepared.metrics()
                        work_start = len(prepared.actions)
                        motion, dwell = ray_actions(spec)
                        work, receipt = checked_actions(prepared, [motion, dwell], limits=limits,
                            meter=meter, check=bounded, phase=f'fan.{index}.{stage_index}.complete_entry')
                        attempt['work'] = receipt
                        if receipt['status'] != 'accepted':
                            attempt.update(status='work_refused', reason='COMPLETE_ENTRY_WORK_REFUSED')
                            continue
                        gain = work.metrics()['targetRemovedMm3']-before['targetRemovedMm3']
                        if gain <= 0:
                            attempt.update(status='no_new_target', reason='NO_POSITIVE_NATIVE_WORK')
                            continue
                        exit_route = copy.deepcopy(route)
                        exit_route.update(outsidePose=spec['outsidePose'], entryPose=spec['endpointPose'],
                            surfacePose={'position': route['surfacePose']['position'],
                                         'quaternion': spec['endpointPose']['quaternion']},
                            inwardRasUnit=spec['inwardRasUnit'], corridorRadiusMm=spec['radiusMm'],
                            pathLengthMm=spec['externalDepthMm'])
                        exited, withdrawal = find_checked_exit(work, exit_route, hand='left',
                            limits=limits, meter=meter, check=bounded)
                        attempt['exit'] = withdrawal
                        if withdrawal['status'] != 'accepted' or _cleanup_pending(exited):
                            attempt.update(status='checked_exit_refused', reason='COMPLETE_CHECKED_EXIT_REFUSED')
                            continue
                    configuration = f'radius-{spec["radiusMm"]:g}'
                    configurations[configuration] = configurations.get(configuration, 0.)+gain
                    attempt.update(status='complete_work_and_exit_checked', reason=None,
                        targetIncrementMm3=gain,
                        nonTargetIncludingExitMm3=exited.metrics()['nonTargetRemovedMm3']-before['nonTargetRemovedMm3'])
                    pose = {key: copy.deepcopy(work.state['slots']['left'][key])
                            for key in ('position', 'quaternion')}
                    stages.append({'stage': stage_index, 'tileId': ray['id'],
                        'configuration': configuration, 'workStartPosition': work_start,
                        'stateId': work.state_id, 'nativeActions': len(work.actions),
                        'targetIncrementMm3': gain, 'observedEndPose': pose,
                        'executionMode': 'requested_motion', 'requestedPoseReached': True,
                        'checkedExitStateId': exited.state_id,
                        'checkedExitActions': len(exited.actions)-len(work.actions)})
                    kept = exited
                    if (exercise_resume and resume is None and
                            meter.remaining-work_reserve >= len(kept.actions)):
                        checkpoint = make_skill_checkpoint(kept, source_seal=plan_seal,
                            task_seal=task['taskSeal'], inventory_seal=inventory['inventorySeal'],
                            limits=limits, meter=meter, stage=f'after-fan-stage-{stage_index}')
                        with meter.reserve(work_reserve):
                            restored, _, ledger = restore_skill_checkpoint(engine.scene, checkpoint,
                                source_seal=plan_seal, task_seal=task['taskSeal'],
                                inventory_seal=inventory['inventorySeal'], meter=meter, check=bounded)
                        resume = {'checkpoint': checkpoint, 'restoredStateId': restored.state_id,
                                  'ledger': ledger, 'exactResume': restored.snapshot() == kept.snapshot()}
                        kept = restored
                    progress((index+(stage_index+1)/max(1, len(fan['orderedRays'])))/max(1, len(inventory['fans'])),
                             f'Checked cumulative re-entry {index+1}, stage {stage_index+1}')
            replay = meter.wrap(engine.fresh())
            with meter.phase(f'fan.{index}.final_full_replay', reserve_calls=EXPORT_RESERVE):
                for action in kept.actions:
                    if replay.step(action, check=bounded)['status'] == 'rejected':
                        raise ValueError('Cumulative native program no longer replays')
            if replay.snapshot() != kept.snapshot() or replay.history != kept.history:
                raise ValueError('Cumulative native checkpoint/history differs')
            sectors, _ = _sector_cells(engine.scene, route)
            outcome = assess_task(kept, task, zones, configurations, exact_replay=True,
                baseline=baseline, admitted_strokes=len(stages), legacy_sectors=sectors)
            row = {'branchId': f'staged-fan-{index}', 'sourceMethod': method,
                'sourceAccessPlanSeal': plan_seal, 'route': route, 'hand': 'left',
                'sourceStateId': engine.state_id, 'finalStateId': kept.state_id,
                'status': 'passed' if outcome['taskGateMet'] else 'unpassed',
                'nativeActions': len(kept.actions), 'approach': approach,
                'approachBaseline': baseline, 'stages': stages, 'attemptedWork': attempts,
                'stopReason': stop_reason, 'interruptionRecovery': resume,
                'outcome': outcome, 'exactNativeReplay': True,
                'executionMetrics': {'simulatedSeconds': kept.state['elapsedSeconds'],
                    'instrumentTravelMm': kept.state['travelMm'],
                    'completedReentries': len(stages), 'workingConfigurations': len(configurations)},
                'modeledOutcomes': summarize_engine_outcomes(kept, baseline=engine)}
            alternatives.append(row)
            branches.append((kept.unwrap(), row))
    except (TransitionBudgetExceeded, TimeoutError) as error:
        interruption = type(error).__name__+': '+str(error)
    admitted = [candidate.actions for candidate, _ in branches]
    for key in meter.report()['branches']:
        candidate = meter.last_committed_engine(branch_id=int(key))
        if candidate is not None and not any(candidate.actions == actions[:len(candidate.actions)] for actions in admitted):
            stopped.append({'retentionRole': 'unadmitted_stopped_native_prefix',
                'meterBranchId': int(key), 'nativeSnapshot': candidate.snapshot(),
                'nativeHistory': copy.deepcopy(candidate.history), 'exactReplayVerified': False})
    report = {'schema': SCHEMA, 'sourceSceneId': engine.scene.identity,
        'sourceStateId': engine.state_id, 'sourceRole': 'DEVELOPMENT',
        'controller': 'deterministic cumulative source-contact re-entry order and native checked skills',
        'controllerMode': 'cumulative-reentry', 'optimizerUpdates': 0,
        'task': task, 'inventory': inventory, 'cusaModel': 'legacy-instantaneous-with-v2-mechanics',
        'sourceAccessPlans': access_plans, 'alternatives': alternatives,
        'cost': meter.report(), 'interruption': interruption, 'stoppedExecutions': stopped,
        'unexecutedFans': unexecuted,
        'wallSeconds': time.monotonic()-started,
        'limitations': ['Missing skull, complete vessels and functional anatomy remain unavailable',
            'Cumulative re-entry plans complete outside-to-target strokes; each size change follows checked withdrawal',
            'Source contact selects a finite sequence, never native feasibility or clinical safety',
            'Actual original access effects remain the fixed continuation baseline',
            'Larger volume, spatial coverage, configuration and tissue-to-target ratio gates remain unchanged',
            'No actor, optimizer or learned-policy benefit claimed']}
    report['comparisonSeal'] = semantic_digest(report)
    return {'branches': branches, 'report': report, 'meter': meter}
