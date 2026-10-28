"""Bounded joint source-corridor and sector execution in the native engine.

Source geometry selects the shortlist before any native outcomes. The existing
access and sector controllers then execute independently from one intact state.
This is deterministic planning; no route ranking is described as learned.
"""
from __future__ import annotations

import copy
import time

from .core import array_digest, semantic_digest
from .surgical_access_planning import AccessNotExecutable, _execute, access_availability
from .surgical_access_planning import _brain_route_novelty
from .surgical_engine import SurgicalEngine
from .surgical_joint_geometry import select_joint_corridors
from .surgical_modeled_outcomes import summarize_engine_outcomes
from .surgical_sector_planning import (
    SCHEMA as SECTOR_SCHEMA, _cleanup_pending, _run_variant, _sector_cells)
from .surgical_transition_meter import TransitionBudgetExceeded, TransitionMeter

SCHEMA = 'resectionlab.joint-anatomy-strategy.v1'
FAILURE_SCHEMA = 'resectionlab.joint-anatomy-failure.v1'
DIRECTIONS = ((2, 'superior-access', 'Superior access'),
              (0, 'axis0-access', 'Native axis 0 access'),
              (1, 'axis1-access', 'Native axis 1 access'))
EXPORT_REPLAY_RESERVE = 6 * 48
FAILED_PREFIX_RESERVE = 64


def _checked_failed_cleanup(candidate, route, *, check):
    """Retain every refusal; never shrink the shaft or exceed 48 actions."""
    attempts = []
    hand = 'left'
    source_poses = [copy.deepcopy(row['pose']) for row in candidate.actions
                    if row['slot'] == hand and row['operation'] == 'apply'
                    and 'pose' in row]
    reverse = []
    for pose in reversed(source_poses):
        if not reverse or pose != reverse[-1]:
            reverse.append(pose)
        if len(reverse) == 2:
            break
    outside = {'slot': hand, 'operation': 'withdraw',
               'pose': copy.deepcopy(route['outsidePose']), 'duration': 30.}
    actions = ([{'slot': hand, 'operation': 'deactivate', 'duration': .2}]
               if candidate.state['slots'][hand]['active'] else [])
    if candidate.state['slots'][hand]['inserted']:
        actions += [outside]
        actions += [{'slot': hand, 'operation': 'move', 'pose': pose,
                     'parameters': {'radiusMm': .6}, 'duration': 2.}
                    for pose in reverse]
        actions += [outside]
    for action in actions:
        check()
        if not _cleanup_pending(candidate):
            break
        if len(candidate.actions) >= 48:
            attempts.append({'action': action, 'status': 'not_attempted',
                             'reason': 'ACTION_BUDGET'})
            break
        preview = candidate.preview(action, check=check,
            include_view=False, include_history=False)
        accepted = (preview['status'] == 'accepted' and
                    preview['metrics']['nonTargetRemovedMm3'] <= 200.+1e-9)
        row = {'action': copy.deepcopy(action),
               'status': preview['status'] if accepted else 'refused',
               'reason': preview.get('reason'), 'previewStateId': preview['stateId']}
        attempts.append(row)
        if accepted:
            outcome = candidate.step(action, check=check)
            if outcome['stateId'] != preview['stateId']:
                raise ValueError('Joint failed-prefix cleanup preview/commit differs')
            row['committedStateId'] = outcome['stateId']
    return attempts


def _retain_failure(meter, base, route, phase, reason, identity, *, check,
                    allow_recovery=True):
    stopped = meter.last_committed_engine_for(phase)
    if stopped is None:
        stopped = base.unwrap().clone()
    candidate = meter.wrap(stopped)
    attempts = []
    replayed = False
    interruption = None
    if allow_recovery:
        try:
            with meter.phase(phase+'-recovery', reserve_calls=EXPORT_REPLAY_RESERVE):
                attempts = _checked_failed_cleanup(candidate, route, check=check)
            with meter.phase(phase+'-failed-replay', reserve_calls=EXPORT_REPLAY_RESERVE):
                replay = base.fresh()
                for action in candidate.actions:
                    check()
                    result = replay.step(action, check=check)
                    if result['status'] == 'rejected':
                        raise ValueError('Retained failed native branch no longer replays')
                if replay.state_id != candidate.state_id or replay.state != candidate.state:
                    raise ValueError('Retained failed native endpoint differs')
                replayed = True
        except (TransitionBudgetExceeded, TimeoutError) as error:
            interruption = type(error).__name__+': '+str(error)
    raw = candidate.unwrap()
    receipt = {'schema': FAILURE_SCHEMA, 'branchId': identity,
               'status': 'unpassed', 'phase': phase, 'reason': reason,
               'retentionRole': 'stopped_native_branch_not_admitted',
               'startStateId': base.state_id, 'finalStateId': raw.state_id,
               'nativeActions': len(raw.actions),
               'cleanupComplete': not _cleanup_pending(raw),
               'cleanupPending': _cleanup_pending(raw),
               'cleanupAttempts': attempts, 'interruption': interruption,
               'verification': {'exactNativeReplay': replayed},
               'modeledOutcomes': summarize_engine_outcomes(raw, baseline=base)}
    return raw, receipt


def compare_joint_strategies(engine, *, acknowledge_unknowns=False,
                             max_native_actions=48, max_physical_calls=1500,
                             max_wall_seconds=120., check=lambda: None,
                             progress=lambda *_: None):
    """Select at most three routes, then run one fixed far-first left strategy.

    Returned meter retains 288 calls for caller-owned export replay. Every
    public physics call, including discarded probes and failed paths, shares
    this cap. Export serialization itself performs no native transitions.
    """
    if acknowledge_unknowns is not True:
        raise ValueError('Explicit acknowledgment of missing anatomy is required')
    if (max_native_actions != 48 or type(max_native_actions) is not int or
            type(max_physical_calls) is not int or not 352 < max_physical_calls <= 1500 or
            type(max_wall_seconds) not in (int, float) or not 0 < max_wall_seconds <= 120.):
        raise ValueError('Keep the declared 48-action, 1500-call, 120-second ceilings')
    readiness = access_availability(engine)
    if not readiness['accessAvailable'] or readiness['skullAvailable']:
        raise ValueError('Joint mapped-brain planning requires an intact skull-absent source')
    if engine.actions or engine.state_id != engine.fresh().state_id:
        raise ValueError('Joint comparison starts only at the common intact source')
    original = engine.snapshot()
    started = time.perf_counter()

    def bounded_check():
        check()
        if time.perf_counter()-started > max_wall_seconds:
            raise TimeoutError('Joint planning aggregate wall budget exhausted')

    routes, geometry = select_joint_corridors(engine.scene, check=bounded_check)
    meter = TransitionMeter(max_physical_calls, check=bounded_check)
    base = meter.wrap(engine)
    access_branches, sector_branches, failed_branches, trials = [], [], [], []
    source_by_axis = {route['nativeAxis']: route for route in routes}
    axis_failures = {}
    stopped = False
    for index, (axis, method, label) in enumerate(DIRECTIONS):
        route = source_by_axis.get(axis)
        if route is None:
            axis_failures[axis] = 'SOURCE_ONLY_SHORTLIST_NO_DISTINCT_ROUTE'
            continue
        phase = 'access-'+method
        trial = {'method': method, 'nativeAxis': axis, 'routeId': route['id'],
                 'status': 'not_attempted', 'failedBranchIds': []}
        trials.append(trial)
        if stopped:
            axis_failures[axis] = 'AGGREGATE_BUDGET_STOPPED'
            trial['reason'] = axis_failures[axis]
            continue
        try:
            bounded_check()
            with meter.phase(phase, reserve_calls=EXPORT_REPLAY_RESERVE+FAILED_PREFIX_RESERVE):
                meter.require_headroom(1)
                before = meter.calls
                candidate, access = _execute(base, {}, method, label+' · joint shortlist', route,
                    max_actions=max_native_actions, skull_available=False,
                    healthy_cap=200., check=bounded_check)
                actual_calls = meter.calls-before
                if actual_calls != sum(access['cost'][key] for key in
                        ('previewCalls', 'executionCalls', 'freshReplayCalls')):
                    raise ValueError('Joint access native-call receipt differs from meter')
            if _cleanup_pending(candidate):
                raise AccessNotExecutable('ACCESS_CHECKED_CLEANUP_INCOMPLETE')
            access['branchId'] = 'joint-access-'+method
            access_branches.append((candidate.unwrap(), access))
            trial.update(status='access_executed', accessBranchId=access['branchId'],
                         accessPhysicalCalls=actual_calls)
        except (AccessNotExecutable, TransitionBudgetExceeded, TimeoutError) as error:
            reason = type(error).__name__+': '+str(error)
            axis_failures[axis] = reason
            trial.update(status='unpassed_access', reason=reason)
            timed_out = isinstance(error, TimeoutError)
            failed = _retain_failure(meter, base, route, phase, reason,
                'joint-failed-access-'+method, check=bounded_check,
                allow_recovery=not timed_out)
            failed_branches.append(failed)
            trial['failedBranchIds'].append(failed[1]['branchId'])
            stopped = isinstance(error, (TransitionBudgetExceeded, TimeoutError))
        progress((index+1)/6., 'Checked joint access '+method)

    access_comparison = {
        'schema': 'resectionlab.surgical-planning.v1', 'task': 'anatomy-access',
        'status': 'three_source_tissue_routes_skull_unsupported' if len(access_branches) == 3
                  else 'insufficient_distinct_routes',
        'requiredAlternatives': 3,
        'directionFailures': [{'nativeAxis': axis, 'method': method,
                              'reason': axis_failures.get(axis, 'NOT_EXECUTED')}
                             for axis, method, _ in DIRECTIONS
                             if not any(plan['method'] == method for _, plan in access_branches)],
        'sourceSceneId': engine.scene.identity, 'startStateId': engine.state_id,
        'sourceLabelsHash': array_digest(engine.scene.labels),
        'guardContractId': engine.guard_report()['contractId'],
        'sourceRole': 'Joint source-only shortlist · deterministic DEVELOPMENT simulation',
        'allowModeledBrainCorridor': True, 'availability': readiness,
        'totalWallSeconds': time.perf_counter()-started, 'optimizerUpdates': 0,
        'limits': {'maxActionsPerApproach': 48, 'maxAdditionalHealthyRemovedMm3': 200.,
                   'minimumBrainEntrySeparationMm': 12., 'minimumBrainCorridorAngleDeg': 30.},
        'limitations': ['Missing skull opening, complete vessels and functional anatomy remain unknown.',
                        'Source occupancy ranks candidates; only native execution measures removal.',
                        'Deterministic route ranking is not a learned policy or clinical validation.'],
        'alternatives': [plan for _, plan in access_branches]}
    access_comparison['comparisonSeal'] = semantic_digest(access_comparison)
    for index, (_, access) in enumerate(access_branches):
        method = access['method']
        trial = next(row for row in trials if row['method'] == method)
        phase = 'sector-'+method
        try:
            bounded_check()
            with meter.phase(phase, reserve_calls=EXPORT_REPLAY_RESERVE+FAILED_PREFIX_RESERVE):
                meter.require_headroom(1)
                cells, evidence = _sector_cells(engine.scene, access['route'])
                before = meter.calls
                candidate, plan = _run_variant(base, access, cells, evidence,
                    'left', ('far', 'near'), max_actions=48, check=bounded_check)
                if meter.calls-before != plan['cost']['totalPhysicalCalls']:
                    raise ValueError('Joint sector native-call receipt differs from meter')
            plan['branchId'] = 'joint-sector-'+method+'-left-far-first'
            sector_branches.append((candidate.unwrap(), plan))
            trial.update(status='passed' if plan['continuationOutcome']['taskGateMet']
                         else 'unpassed_sector', sectorBranchId=plan['branchId'],
                         sectorPhysicalCalls=meter.calls-before)
        except (TransitionBudgetExceeded, TimeoutError) as error:
            reason = type(error).__name__+': '+str(error)
            trial.update(status='unpassed_sector', reason=reason)
            failed = _retain_failure(meter, base, access['route'], phase, reason,
                'joint-failed-sector-'+method, check=bounded_check,
                allow_recovery=not isinstance(error, TimeoutError))
            failed_branches.append(failed)
            trial['failedBranchIds'].append(failed[1]['branchId'])
            for _, later in access_branches[index+1:]:
                next(row for row in trials if row['method'] == later['method']).update(
                    status='sector_not_attempted', reason='AGGREGATE_BUDGET_STOPPED')
            break
        progress(.5+(index+1)/6., 'Checked joint sector '+method)
    sectors = {
        'schema': SECTOR_SCHEMA, 'task': 'anatomy-strategy',
        'status': 'strategy_options_compared' if sector_branches else 'no_executable_strategy',
        'sourceSceneId': engine.scene.identity, 'startStateId': engine.state_id,
        'sourceAccessComparisonSeal': access_comparison['comparisonSeal'],
        'sourceAccessComparisonStatus': access_comparison['status'],
        'anatomicalCompleteness': 'skull, vessel and functional coverage unknown',
        'clinicalValidation': False, 'limitations': access_comparison['limitations'],
        'declaredGate': {'additionalTargetMm3': 30., 'eachSectorMm3': 10.,
                         'additionalNonTargetMm3': 20., 'totalNonTargetMm3': 200.,
                         'minSectorStrokes': 2, 'checkedCleanup': True},
        'limits': {'maxNativeActionsPerStrategy': 48, 'maxAggregateWallSeconds': max_wall_seconds},
        'optionCount': len(routes), 'retainedCount': len(sector_branches),
        'passedCount': sum(plan['status'] == 'passed' for _, plan in sector_branches),
        'totalWallSeconds': time.perf_counter()-started,
        'optionFailures': [{'option': row['method'], 'reason': row.get('reason', row['status'])}
                           for row in trials if row['status'] != 'passed'],
        'alternatives': [plan for _, plan in sector_branches]}
    sectors['comparisonSeal'] = semantic_digest(sectors)
    if engine.snapshot() != original:
        raise ValueError('Joint planning modified the common intact source')
    report = {'schema': SCHEMA, 'status': 'joint_strategies_compared',
              'sourceSceneId': engine.scene.identity, 'startStateId': engine.state_id,
              'geometry': geometry, 'accessComparison': access_comparison,
              'sectorComparison': sectors, 'trials': trials,
              'failedExecutions': [receipt for _, receipt in failed_branches],
              'threeApproachTaskGateMet': len(access_branches) == 3 and sectors['passedCount'] == 3,
              'executionRule': 'one far-first left-CUSA sector strategy per source-selected route',
              'optimizerUpdates': 0, 'cost': meter.report(),
              'wallSeconds': time.perf_counter()-started,
              'limits': {'maxPhysicalCallsIncludingExportReplay': max_physical_calls,
                         'reservedExportReplayCalls': EXPORT_REPLAY_RESERVE,
                         'maxWallSeconds': max_wall_seconds},
              'sourcePreserved': True}
    report['jointSeal'] = semantic_digest(report)
    return {'accessBranches': access_branches, 'sectorBranches': sector_branches,
            'failedBranches': failed_branches, 'report': report, 'meter': meter}


def validate_joint_report(scene, record, report):
    """Fast source/action bindings for display; this does not perform replay."""
    from .surgical_planning import validate_saved_comparison, validate_modeled_outcome_binding
    from .surgical_sector_planning import validate_saved_sector_comparison
    bare = {key: value for key, value in report.items() if key != 'jointSeal'}
    if (report.get('schema') != SCHEMA or report.get('sourceSceneId') != scene.identity or
            report.get('jointSeal') != semantic_digest(bare)):
        raise ValueError('Joint report source or seal differs')
    access, sectors = report['accessComparison'], report['sectorComparison']
    validate_saved_comparison(scene, record, access)
    validate_saved_sector_comparison(scene, {**record, 'planning': access}, sectors)
    prior = []
    for plan in access['alternatives']:
        route = plan['route']
        if _brain_route_novelty(route['surfacePose']['position'], route['inwardRasUnit'], prior) is None:
            raise ValueError('Joint native routes fail physical entry/direction novelty')
        prior.append(route)
    expected_gate = len(access['alternatives']) == 3 and len(sectors['alternatives']) == 3 and \
        all(plan['continuationOutcome']['taskGateMet'] for plan in sectors['alternatives'])
    if report.get('threeApproachTaskGateMet') is not expected_gate:
        raise ValueError('Joint three-approach gate differs from native task receipts')
    for failure in report['failedExecutions']:
        snapshot = record['branches'].get(failure['branchId'])
        if (failure.get('schema') != FAILURE_SCHEMA or snapshot is None or
                failure.get('status') != 'unpassed' or
                failure.get('retentionRole') != 'stopped_native_branch_not_admitted' or
                failure['startStateId'] != report['startStateId'] or
                snapshot['stateId'] != failure['finalStateId'] or
                len(snapshot['actions']) != failure['nativeActions']):
            raise ValueError('Joint failed branch receipt differs from native snapshot')
        validate_modeled_outcome_binding(scene, snapshot, failure['modeledOutcomes'],
                                         report['startStateId'])
    cost = report['cost']
    if (cost['physicalCalls'] != cost['previewCalls']+cost['directStepCalls'] or
            cost['physicalCalls'] != len(cost['trace']) or
            not 0 <= cost['physicalCalls'] <= cost['maxPhysicalCalls'] <= 1500):
        raise ValueError('Joint physical-call inventory differs')
    return {'verified': True, 'accessRoutes': len(access['alternatives']),
            'sectorStrategies': len(sectors['alternatives']),
            'failedBranches': len(report['failedExecutions'])}


def install_joint_comparison(record, result, access, sectors):
    """Add versioned native branches and archive the exact classic pair."""
    if sum(len(result[key]) for key in
           ('accessBranches', 'sectorBranches', 'failedBranches')) > 6:
        raise ValueError('Joint export exceeds six retained native branches')
    source_branches = copy.deepcopy(record['branches'])
    source_parents = copy.deepcopy(record.get('parents', {}))
    source_access = copy.deepcopy(record['planning'])
    source_sectors = copy.deepcopy(record['sectorPlanning'])
    created = {}
    for kind, key in (('access', 'accessBranches'), ('sector', 'sectorBranches'),
                      ('failed', 'failedBranches')):
        rows = result[key]
        if not isinstance(rows, list):
            raise ValueError('Joint native branch list differs')
        for candidate, evidence in rows:
            if (not isinstance(candidate, SurgicalEngine) or
                    len(candidate.actions) > 48 or
                    not isinstance(evidence, dict)):
                raise ValueError('Joint native branch or receipt differs')
            branch_id = evidence.get('branchId')
            if kind == 'access':
                expected = 'joint-access-'+str(evidence.get('method'))
            elif kind == 'sector':
                expected = ('joint-sector-'+str(evidence.get('strategy', {}).get('sourceAccessMethod'))+
                            '-left-far-first')
            else:
                expected = None
            if (not isinstance(branch_id, str) or not branch_id.startswith('joint-') or
                    branch_id in record['branches'] or branch_id in created or
                    (expected is not None and branch_id != expected) or
                    (kind == 'failed' and not branch_id.startswith('joint-failed-'))):
                raise ValueError('Joint branch identity collides with classic source')
            if kind == 'failed' and (evidence.get('schema') !=
                    'resectionlab.joint-anatomy-failure.v1' or
                    evidence.get('retentionRole') != 'stopped_native_branch_not_admitted' or
                    evidence.get('nativeActions') != len(candidate.actions) or
                    evidence.get('startStateId') !=
                        record['branches'][record['branchId']]['stateId']):
                raise ValueError('Retained failed native branch receipt differs')
            snapshot = candidate.snapshot()
            if (snapshot['sceneId'] != record['branches'][record['branchId']]['sceneId'] or
                    snapshot['stateId'] != candidate.state_id or
                    snapshot['actions'] != candidate.actions or
                    (evidence.get('finalStateId') is not None and
                     evidence['finalStateId'] != candidate.state_id)):
                raise ValueError('Joint branch endpoint or native actions differ')
            record['branches'][branch_id] = snapshot
            record.setdefault('parents', {})[branch_id] = {
                'branchId': record['branchId'], 'position': 0}
            created[branch_id] = {'kind': kind, 'candidate': candidate,
                                  'stateId': candidate.state_id,
                                  'actionsHash': semantic_digest(candidate.actions)}
    if len({row['branchId'] for row in access['alternatives']}) != len(result['accessBranches']) or \
            {row['branchId'] for row in access['alternatives']} != {
                row['branchId'] for _, row in result['accessBranches']} or \
            len({row['branchId'] for row in sectors['alternatives']}) != len(result['sectorBranches']) or \
            {row['branchId'] for row in sectors['alternatives']} != {
                row['branchId'] for _, row in result['sectorBranches']}:
        raise ValueError('Joint active comparison alternatives differ from native branches')
    history = record.setdefault('sectorPlanningHistory', [])
    if not isinstance(history, list) or len(history) >= 8:
        raise ValueError('Classic comparison history capacity differs')
    history.append({'sourceAccessComparison': source_access, 'comparison': source_sectors})
    record['jointBaselineHistory'] = copy.deepcopy(history[-1:])
    record['planning'] = copy.deepcopy(access)
    record['sectorPlanning'] = copy.deepcopy(sectors)
    record['sectorSourceAccessComparison'] = copy.deepcopy(access)
    if (record['branchId'] not in source_branches or record['position'] != 0 or
            any(record['branches'][key] != value for key, value in source_branches.items()) or
            any(record['parents'].get(key) != value for key, value in source_parents.items())):
        raise ValueError('Classic native branch or parent was changed')
    return created
