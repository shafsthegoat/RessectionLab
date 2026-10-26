"""Source-bound, checked CUSA sector order and hand comparison.

This is a partial modeled-tissue research option. Source tumor cells define two
small, disjoint sectors next to an already verified access ray. No skull,
vessel, functional boundary, or clinical safety is inferred from missing masks.
Every retained branch starts at the same intact native source and uses public
preview/step transitions, including the real left/right parked-tool geometry.
"""
from __future__ import annotations

import copy
import hashlib
import math
import time

import numpy as np

from .core import semantic_digest
from .surgical_engine import SurgicalEngine


TASK = 'anatomy-strategy'
SCHEMA = 'resectionlab.anatomy-strategy.v1'
SECTOR_WINDOWS_MM = {'near': (0., 2.25), 'far': (3., 5.)}
SECTOR_RADIAL_MM = (1.7, 3.)
SECTOR_STROKE = {'near': (1., 1.6), 'far': (4., 1.8)}
TRANSIT_RADIUS_MM = 1.
MIN_SECTOR_MM3 = 10.
MIN_ADDITIONAL_TARGET_MM3 = 30.
MAX_ADDITIONAL_NON_TARGET_MM3 = 20.
MAX_TOTAL_NON_TARGET_MM3 = 200.


class StrategyUnavailable(ValueError):
    """A requested option cannot execute under the declared physical bounds."""


def _sector_cells(scene, route):
    if route.get('sourceSceneId') != scene.identity or route.get('pathType') != 'oblique_mapped_brain':
        raise StrategyUnavailable('Only a source-bound oblique mapped-brain access ray is supported')
    if route.get('accessCompleteness') != 'unsupported_skull_access_brain_corridor':
        raise StrategyUnavailable('The current sector option requires the verified mapped-brain corridor')
    origin = np.asarray(route['surfacePose']['position'], dtype=float)
    inward = np.asarray(route['inwardRasUnit'], dtype=float)
    entry = float(route['pathLengthMm'])
    if not np.isfinite(origin).all() or not np.isfinite(inward).all() or not math.isfinite(entry) or \
            abs(np.linalg.norm(inward)-1.) > 1e-6 or entry <= 0:
        raise StrategyUnavailable('Access ray geometry is incomplete')
    target_ids = [row['id'] for row in scene.regions
                  if row.get('target') and row['material'] == 'tumor']
    if not target_ids:
        raise StrategyUnavailable('No mapped source target tumor exists')
    ijk = np.argwhere(np.isin(scene.labels, target_ids))
    points = ijk @ scene.affine[:3, :3].T + scene.affine[:3, 3]
    depth = (points-origin) @ inward
    radial = np.linalg.norm(points-origin-depth[:, None]*inward, axis=1)
    flat = np.ravel_multi_index(ijk.T, scene.shape)
    sectors = {}
    evidence = {}
    for name, (low, high) in SECTOR_WINDOWS_MM.items():
        selected = (depth >= entry+low) & (depth < entry+high) & \
                   (radial >= SECTOR_RADIAL_MM[0]) & (radial < SECTOR_RADIAL_MM[1])
        cells = np.sort(flat[selected].astype(np.int64))
        sector_points = points[selected]
        sectors[name] = frozenset(map(int, cells))
        evidence[name] = {
            'id': name, 'sourceRegionIds': target_ids,
            'sourceFlatCellIds': cells.tolist(),
            'sourceCellCount': int(len(cells)),
            'sourceVolumeMm3': float(len(cells)*scene.voxel_mm3),
            'sourceCellSha256': hashlib.sha256(cells.astype('<i8').tobytes()).hexdigest(),
            'centerRasMm': sector_points.mean(axis=0).tolist() if len(cells) else None,
            'boundsRasMm': [sector_points.min(axis=0).tolist(),
                            sector_points.max(axis=0).tolist()] if len(cells) else None,
            'axialDepthFromTargetEntryMm': [low, high],
            'radialDistanceFromAccessRayMm': list(SECTOR_RADIAL_MM),
        }
    if sectors['near'] & sectors['far']:
        raise ValueError('Predeclared target sectors overlap')
    return sectors, evidence


def source_sector_binding(scene, route):
    """Privileged source membership for audit/ledger code, never actor input."""
    return _sector_cells(scene, route)


def _pose(route, offset_mm):
    point = (np.asarray(route['surfacePose']['position']) +
             (float(route['pathLengthMm'])+offset_mm)*np.asarray(route['inwardRasUnit']))
    return {'position': point.tolist(),
            'quaternion': list(route['entryPose']['quaternion'])}


def _sector_option_actions(route, sector, hand):
    if sector not in SECTOR_STROKE or hand not in ('left', 'right'):
        raise ValueError('Unknown sector or CUSA hand')
    actions = []
    if sector == 'far':
        actions.append({'slot': hand, 'operation': 'apply', 'pose': _pose(route, 4.),
                        'parameters': {'radiusMm': TRANSIT_RADIUS_MM}, 'duration': .2})
    depth, radius = SECTOR_STROKE[sector]
    actions.append({'slot': hand, 'operation': 'apply', 'pose': _pose(route, depth),
                    'parameters': {'radiusMm': radius}, 'duration': .2})
    return actions


def execute_strategy_option(engine, access_plan, option_id, *, corridor_binding=None,
                            max_actions=48, check=lambda: None):
    """Append one checked high-level option to the same observed native history.

    `corridor:<hand>` is an intact-source choice. Later `near:<hand>`,
    `far:<hand>` and `STOP` require its privileged binding and cannot change
    ancestry or hand. Binding/preview data are audit information, never actor
    input. A refused action leaves an accepted prefix intact and is reported.
    """
    if type(max_actions) is not int or not 1 <= max_actions <= 48:
        raise ValueError('Keep the declared 48 native action ceiling')
    if not isinstance(option_id, str) or option_id not in {
            'corridor:left', 'corridor:right', 'near:left', 'near:right',
            'far:left', 'far:right', 'STOP'}:
        raise ValueError('Unknown strategy option')
    if option_id == 'STOP' and corridor_binding is None:
        if engine.actions or engine.state_id != access_plan['startStateId']:
            raise StrategyUnavailable('Unbound STOP is valid only at the intact boundary')
        return engine.clone(), {
            'schema': 'resectionlab.anatomy-strategy-option.v1',
            'id': 'STOP', 'boundary': 'intact_before_access', 'hand': None,
            'startStateId': engine.state_id, 'finalStateId': engine.state_id,
            'status': 'accepted', 'reason': 'abstained_before_corridor',
            'attempts': [], 'nativeActionsAppended': 0,
            'sectorDeltaMm3': {'near': 0., 'far': 0.},
            'cleanupComplete': not _cleanup_pending(engine),
            'taskGateMet': False, 'corridorBinding': None}
    route = access_plan['route']
    sectors, evidence = _sector_cells(engine.scene, route)
    if any(row['sourceVolumeMm3'] < MIN_SECTOR_MM3 for row in evidence.values()):
        raise StrategyUnavailable('PREDECLARED_SOURCE_SECTOR_CAPACITY_BELOW_10_MM3')
    prefix = _prefix_actions(access_plan)
    candidate = engine.clone()
    attempts = []
    failure = None
    if option_id.startswith('corridor:'):
        hand = option_id.split(':')[1]
        if corridor_binding is not None or engine.actions or \
                engine.state_id != access_plan['startStateId']:
            raise StrategyUnavailable('Corridor hand is chosen only at the common intact boundary')
        actions = []
        for source_action in prefix:
            if source_action['slot'] != 'left':
                raise StrategyUnavailable('Saved access prefix has an unsupported hand')
            action = copy.deepcopy(source_action)
            action['slot'] = hand
            actions.append(action)
        prefix_metrics = None
        boundary = 'intact_before_access'
    else:
        if not isinstance(corridor_binding, dict) or \
                corridor_binding.get('sourceSceneId') != engine.scene.identity or \
                corridor_binding.get('sourceAccessPlanSeal') != access_plan['planSeal']:
            raise StrategyUnavailable('Post-corridor option lacks its observed source binding')
        hand = corridor_binding['hand']
        expected = [dict(action, slot=hand) for action in prefix]
        if (engine.actions[:len(prefix)] != expected or
                len(engine.history) < len(prefix) or
                engine.history[len(prefix)-1]['stateAfter'] != corridor_binding['prefixStateId'] or
                semantic_digest(expected) != corridor_binding['prefixActionSeal']):
            raise StrategyUnavailable('Native corridor ancestry differs; no trajectory splice allowed')
        if engine.state['slots'][hand]['instrument'] != 'cusa' or \
                not engine.state['slots'][hand]['inserted']:
            raise StrategyUnavailable('Observed CUSA hand is not in its checked corridor')
        if option_id != 'STOP' and option_id.split(':')[1] != hand:
            raise StrategyUnavailable('Post-corridor option cannot retroactively change hands')
        prefix_metrics = corridor_binding['prefixMetrics']
        actions = (_sector_option_actions(route, option_id.split(':')[0], hand)
                   if option_id != 'STOP' else
                   [{'slot': hand, 'operation': 'deactivate', 'duration': .2},
                    {'slot': hand, 'operation': 'withdraw',
                     'pose': copy.deepcopy(route['outsidePose']),
                     'parameters': {'radiusMm': TRANSIT_RADIUS_MM}, 'duration': 30.}])
        boundary = 'post_corridor_same_engine_same_hand'

    for index, action in enumerate(actions):
        check()
        at = len(candidate.actions)
        if at >= max_actions:
            failure = 'ACTION_BUDGET'
            attempts.append({'index': index, 'atActionPosition': at, 'action': action,
                             'status': 'not_attempted', 'reason': failure})
            break
        preview = candidate.preview(action, check=check, include_view=False,
                                    include_history=False)
        removed = set(map(int, preview.get('deltas', {}).get('removedCells', ())))
        sector_delta = {name: len(removed & cells)*engine.scene.voxel_mm3
                        for name, cells in sectors.items()}
        reason = preview.get('reason')
        accepted = (preview['status'] == 'accepted' or
                    preview['status'] == 'partial' and reason == 'TISSUE_ASPIRATION_IN_PROGRESS')
        if accepted and preview['metrics']['nonTargetRemovedMm3'] > MAX_TOTAL_NON_TARGET_MM3+1e-9:
            accepted, reason = False, 'TOTAL_NON_TARGET_CAP'
        if (accepted and prefix_metrics is not None and
                preview['metrics']['nonTargetRemovedMm3']-
                prefix_metrics['nonTargetRemovedMm3'] > MAX_ADDITIONAL_NON_TARGET_MM3+1e-9):
            accepted, reason = False, 'CONTINUATION_NON_TARGET_CAP'
        if accepted and not option_id.startswith('corridor:') and removed:
            allowed = {row['id'] for row in engine.scene.regions
                       if row['material'] == 'tumor' and row.get('target') or
                          row['material'] == 'soft_tissue' and row['name'].lower() == 'brain'}
            if any(int(engine.scene.labels.flat[cell]) not in allowed for cell in removed):
                accepted, reason = False, 'NON_BRAIN_OR_NON_TARGET_REMOVAL_REFUSED'
        row = {'index': index, 'atActionPosition': at, 'action': copy.deepcopy(action),
               'previewStatus': preview['status'], 'previewStateId': preview['stateId'],
               'status': preview['status'] if accepted else 'refused',
               'reason': reason, 'sectorDeltaMm3': sector_delta}
        attempts.append(row)
        if not accepted:
            failure = str(reason or preview['status'])
            break
        result = candidate.step(action, check=check)
        if result['stateId'] != preview['stateId'] or result['status'] != preview['status']:
            raise ValueError('High-level option preview and native commit differ')
        row['committedStateId'] = result['stateId']

    if option_id == 'STOP' and failure is not None and \
            candidate.state['slots'][hand]['inserted']:
        # The direct withdrawal refusal remains visible. A same-hand, inactive
        # return through the already cleared ray is separately previewed.
        for action in (
                {'slot': hand, 'operation': 'move', 'pose': _pose(route, 1.),
                 'parameters': {'radiusMm': TRANSIT_RADIUS_MM}, 'duration': 2.},
                {'slot': hand, 'operation': 'withdraw',
                 'pose': copy.deepcopy(route['outsidePose']),
                 'parameters': {'radiusMm': TRANSIT_RADIUS_MM}, 'duration': 30.}):
            check()
            at = len(candidate.actions)
            if at >= max_actions:
                failure = 'ACTION_BUDGET'
                break
            preview = candidate.preview(action, check=check, include_view=False,
                                        include_history=False)
            removed = set(map(int, preview.get('deltas', {}).get('removedCells', ())))
            acceptable = (preview['status'] == 'accepted' and
                preview['metrics']['nonTargetRemovedMm3'] <= MAX_TOTAL_NON_TARGET_MM3+1e-9 and
                preview['metrics']['nonTargetRemovedMm3']-
                prefix_metrics['nonTargetRemovedMm3'] <= MAX_ADDITIONAL_NON_TARGET_MM3+1e-9)
            row = {'index': len(attempts), 'atActionPosition': at,
                   'action': copy.deepcopy(action), 'previewStatus': preview['status'],
                   'previewStateId': preview['stateId'],
                   'status': preview['status'] if acceptable else 'refused',
                   'reason': preview.get('reason') if not acceptable else None,
                   'sectorDeltaMm3': {name: len(removed & cells)*engine.scene.voxel_mm3
                                      for name, cells in sectors.items()}}
            attempts.append(row)
            if not acceptable:
                failure = str(preview.get('reason') or 'CHECKED_CLEANUP_REFUSED')
                break
            result = candidate.step(action, check=check)
            if result['stateId'] != preview['stateId']:
                raise ValueError('High-level STOP recovery preview/commit differs')
            row['committedStateId'] = result['stateId']
        else:
            failure = None if not _cleanup_pending(candidate) else 'CHECKED_CLEANUP_INCOMPLETE'

    if option_id.startswith('corridor:') and failure is None:
        corridor_binding = {'sourceSceneId': engine.scene.identity,
                            'sourceAccessPlanSeal': access_plan['planSeal'],
                            'hand': hand, 'prefixStateId': candidate.state_id,
                            'prefixActionSeal': semantic_digest(candidate.actions),
                            'prefixMetrics': candidate.metrics(),
                            'prefixSectorRemovedMm3': {
                                name: len(set(candidate.state['removed']) & cells)*
                                      engine.scene.voxel_mm3 for name, cells in sectors.items()}}
    elif option_id.startswith('corridor:'):
        corridor_binding = None
    outcome = {'schema': 'resectionlab.anatomy-strategy-option.v1',
               'id': option_id, 'boundary': boundary, 'hand': hand,
               'startStateId': engine.state_id, 'finalStateId': candidate.state_id,
               'status': 'accepted' if failure is None else 'refused', 'reason': failure,
               'attempts': attempts, 'nativeActionsAppended': len(candidate.actions)-len(engine.actions),
               'sectorDeltaMm3': {name: len((set(candidate.state['removed'])-
                   set(engine.state['removed'])) & cells)*engine.scene.voxel_mm3
                   for name, cells in sectors.items()},
               'cleanupComplete': not _cleanup_pending(candidate),
               'corridorBinding': corridor_binding}
    return candidate, outcome


def _cleanup_pending(engine):
    pending = []
    for name, slot in engine.state['slots'].items():
        for field in ('active', 'inserted', 'grip'):
            if slot[field]:
                pending.append(name+':'+field)
    if engine.state['holds']:
        pending.append('holds')
    if any(source.get('bleedingActive', True) for source in engine.state['sources']):
        pending.append('active_bleeding')
    if sum(engine.state['fluid'].values()) + \
            sum(source.get('reservoirMm3', 0.) for source in engine.state['sources']) > 1e-9:
        pending.append('fluid')
    if engine.state['debrisMm3'] > 1e-9:
        pending.append('debris')
    return pending


def _prefix_actions(access_plan):
    actions = []
    for row in access_plan['decisions']:
        action = row['action']
        if action['operation'] in ('deactivate', 'withdraw'):
            break
        actions.append(action)
    if not actions or not any(a['operation'] == 'apply' for a in actions):
        raise StrategyUnavailable('Saved access contains no checked active CUSA prefix')
    return actions


def _plan_seal(scene_id, start_state_id, strategy, decisions, sector_evidence):
    return semantic_digest({'sourceSceneId': scene_id, 'startStateId': start_state_id,
                            'strategy': strategy, 'decisions': decisions,
                            'sectors': sector_evidence})


def intact_strategy_observation(engine, access_plan, *, max_actions=48):
    """Detached published local observation for the pre-access hand choice."""
    from scipy.spatial.transform import Rotation
    from .surgical_planning import _observer
    from .surgical_rl_rewards import RewardTracker
    if engine.actions or engine.state_id != access_plan['startStateId']:
        raise ValueError('Corridor hand choice requires the common intact boundary')
    route = access_plan['route']
    _, sector_evidence = _sector_cells(engine.scene, route)
    source_capacity = all(row['sourceVolumeMm3'] >= MIN_SECTOR_MM3
                          for row in sector_evidence.values())
    tracker = RewardTracker(engine, engine._protected_ids)
    observed, _ = _observer(engine, tracker)
    target = np.asarray(route['entryPose']['position'])
    options = []
    for hand in ('left', 'right'):
        slot = engine.state['slots'][hand]
        options.append({'id': 'corridor:'+hand, 'hand': hand,
                        'entryRelativeToToolMm': Rotation.from_quat(
                            slot['quaternion']).inv().apply(
                            target-np.asarray(slot['position'])).tolist(),
                        'structurallyAvailable': source_capacity and not slot['inserted'] and
                            not slot['active'] and not slot['grip']})
    options.append({'id': 'STOP', 'hand': None, 'entryRelativeToToolMm': None,
                    'structurallyAvailable': True})
    return {'boundary': 'intact_before_access',
            'observation': {key: np.array(value, copy=True)
                            for key, value in observed.items()},
            'options': options,
            'remainingNativeActions': max_actions-len(engine.actions)}


def strategy_observation(engine, sector_evidence, sector_cells, *, prefix_metrics,
                         prefix_sector_removed_mm3, active_hand, max_actions=48,
                         boundary='post_corridor'):
    """Policy-facing detached local observation plus bounded option ledger.

    Source cell identities/hashes, scene IDs, saved branch IDs, future previews,
    and raw engine history remain in privileged audit data only.
    """
    from .surgical_planning import _observer
    from .surgical_rl_rewards import RewardTracker
    from scipy.spatial.transform import Rotation
    if boundary != 'post_corridor' or active_hand not in ('left', 'right'):
        raise ValueError('Sector options require the post_corridor boundary and its observed hand')
    tracker = RewardTracker(engine, engine._protected_ids)
    observation, _ = _observer(engine, tracker)
    detached = {key: np.array(value, copy=True) for key, value in observation.items()}
    removed = set(engine.state['removed'])
    per_sector = {name: len(removed & sector_cells[name])*engine.scene.voxel_mm3-
                         prefix_sector_removed_mm3[name]
                  for name in ('near', 'far')}
    options = []
    for name in ('near', 'far'):
        center = sector_evidence[name]['centerRasMm']
        for hand in ('left', 'right'):
            slot = engine.state['slots'][hand]
            options.append({'id': name+':'+hand, 'sector': name, 'hand': hand,
                            'centerRelativeToToolMm':
                                (Rotation.from_quat(slot['quaternion']).inv().apply(
                                    np.subtract(center, slot['position'])).tolist()
                                 if center is not None else None),
                            'structurallyAvailable': hand == active_hand and center is not None and
                                slot['instrument'] == 'cusa' and not slot['grip']})
    options.append({'id': 'STOP', 'sector': None, 'hand': None,
                    'centerRelativeToToolMm': None, 'structurallyAvailable': True})
    return {'observation': detached,
            'boundary': boundary,
            'ledger': {'additionalTargetMm3': engine.metrics()['targetRemovedMm3']-
                       prefix_metrics['targetRemovedMm3'],
                       'additionalNonTargetMm3': engine.metrics()['nonTargetRemovedMm3']-
                       prefix_metrics['nonTargetRemovedMm3'],
                       'additionalPerSectorRemovedMm3': per_sector,
                       'remainingNativeActions': max(0, max_actions-len(engine.actions)),
                       'activeHand': active_hand,
                       'thresholds': {'additionalTargetMm3': MIN_ADDITIONAL_TARGET_MM3,
                                      'eachSectorMm3': MIN_SECTOR_MM3,
                                      'additionalNonTargetMm3': MAX_ADDITIONAL_NON_TARGET_MM3,
                                      'totalNonTargetMm3': MAX_TOTAL_NON_TARGET_MM3}},
            'options': options}


def _run_variant(base, access_plan, sectors, sector_evidence, hand, order, *,
                 max_actions, check):
    candidate = base.clone()
    route = access_plan['route']
    prefix_actions = _prefix_actions(access_plan)
    attempts = []
    failures = []
    phases = []
    initial_metrics = base.metrics()
    prefix_metrics = None
    prefix_sector = None
    prefix_state_id = None
    accepted_sector_strokes = 0
    first_progress = []
    started = time.perf_counter()
    allowed = {row['id'] for row in base.scene.regions
               if row['material'] == 'tumor' and row.get('target') or
                  row['material'] == 'soft_tissue' and row['name'].lower() == 'brain'}

    def attempt(action, phase, *, cap_from_prefix=True, required=False):
        check()
        at = len(candidate.actions)
        if at-len(base.actions) >= max_actions:
            attempts.append({'phase': phase, 'atActionPosition': at,
                             'action': copy.deepcopy(action), 'status': 'not_attempted',
                             'reason': 'ACTION_BUDGET'})
            return False, None
        preview = candidate.preview(action, check=check, include_view=False,
                                    include_history=False)
        removed = set(map(int, preview.get('deltas', {}).get('removedCells', ())))
        deltas = {name: len(removed & cells)*base.scene.voxel_mm3
                  for name, cells in sectors.items()}
        reason = preview.get('reason')
        acceptable = (preview['status'] == 'accepted' or
                      preview['status'] == 'partial' and reason == 'TISSUE_ASPIRATION_IN_PROGRESS')
        if phase.startswith('sector-') and removed and \
                any(int(base.scene.labels.flat[cell]) not in allowed for cell in removed):
            acceptable, reason = False, 'NON_BRAIN_OR_NON_TARGET_REMOVAL_REFUSED'
        if acceptable and preview['metrics']['nonTargetRemovedMm3'] - initial_metrics['nonTargetRemovedMm3'] \
                > MAX_TOTAL_NON_TARGET_MM3+1e-9:
            acceptable, reason = False, 'TOTAL_NON_TARGET_CAP'
        if acceptable and cap_from_prefix and prefix_metrics is not None and \
                preview['metrics']['nonTargetRemovedMm3']-prefix_metrics['nonTargetRemovedMm3'] \
                > MAX_ADDITIONAL_NON_TARGET_MM3+1e-9:
            acceptable, reason = False, 'CONTINUATION_NON_TARGET_CAP'
        row = {'phase': phase, 'atActionPosition': at, 'action': copy.deepcopy(action),
               'status': preview['status'] if acceptable else 'refused',
               'previewStatus': preview['status'], 'reason': reason,
               'previewStateId': preview['stateId'], 'sectorDeltaMm3': deltas,
               'previewTargetRemovedMm3': preview['metrics']['targetRemovedMm3'],
               'previewNonTargetRemovedMm3': preview['metrics']['nonTargetRemovedMm3']}
        attempts.append(row)
        if not acceptable:
            if required:
                failures.append(str(reason or preview['status']))
            return False, row
        committed = candidate.step(action, check=check)
        if committed['stateId'] != preview['stateId'] or committed['status'] != preview['status']:
            raise ValueError('Sector strategy preview and native commit differ')
        row['committedStateId'] = committed['stateId']
        row['committedActionPosition'] = len(candidate.actions)
        return True, row

    for original in prefix_actions:
        action = copy.deepcopy(original)
        if action['slot'] != 'left':
            failures.append('UNSUPPORTED_PREFIX_HAND')
            break
        action['slot'] = hand
        accepted, _ = attempt(action, 'verified-access-prefix', cap_from_prefix=False,
                              required=True)
        if not accepted:
            break
    if not failures:
        prefix_metrics = candidate.metrics()
        prefix_state_id = candidate.state_id
        prefix_sector = {name: len(set(candidate.state['removed']) & cells)*base.scene.voxel_mm3
                         for name, cells in sectors.items()}
        if prefix_metrics['targetRemovedMm3'] <= initial_metrics['targetRemovedMm3']:
            failures.append('ACCESS_PREFIX_HAS_NO_TARGET_PROGRESS')
        if any(sector_evidence[name]['sourceVolumeMm3']-prefix_sector[name] < MIN_SECTOR_MM3
               for name in ('near', 'far')):
            failures.append('SECTOR_RESIDUAL_BELOW_DECLARED_GATE')

    if not failures:
        for index, name in enumerate(order):
            start_position = len(candidate.actions)
            phase = 'sector-'+name
            wrong = order[1] if index == 0 else None
            before = {key: len(set(candidate.state['removed']) & cells)*base.scene.voxel_mm3
                      for key, cells in sectors.items()}
            if name == 'far':
                transit = {'slot': hand, 'operation': 'apply', 'pose': _pose(route, 4.),
                           'parameters': {'radiusMm': TRANSIT_RADIUS_MM}, 'duration': .2}
                accepted, row = attempt(transit, phase+'-checked-transit', required=True)
                if not accepted or wrong and row['sectorDeltaMm3'][wrong] > 0:
                    if accepted and wrong:
                        failures.append('OPPOSITE_SECTOR_REMOVED_BEFORE_FIRST_ORDERED_SECTOR')
                    break
                if wrong and row['sectorDeltaMm3'][wrong] > 0:
                    break
                if row['sectorDeltaMm3'][name] > 0 and name not in first_progress:
                    first_progress.append(name)
            depth, radius = SECTOR_STROKE[name]
            stroke = {'slot': hand, 'operation': 'apply', 'pose': _pose(route, depth),
                      'parameters': {'radiusMm': radius}, 'duration': .2}
            accepted, row = attempt(stroke, phase+'-working-stroke', required=True)
            if not accepted:
                break
            accepted_sector_strokes += 1
            if wrong and row['sectorDeltaMm3'][wrong] > 0:
                failures.append('OPPOSITE_SECTOR_REMOVED_BEFORE_FIRST_ORDERED_SECTOR')
                break
            if row['sectorDeltaMm3'][name] > 0 and name not in first_progress:
                first_progress.append(name)
            after = {key: len(set(candidate.state['removed']) & cells)*base.scene.voxel_mm3
                     for key, cells in sectors.items()}
            phases.append({'sector': name, 'hand': hand, 'startActionPosition': start_position,
                           'endActionPosition': len(candidate.actions),
                           'perSectorDeltaMm3': {key: after[key]-before[key] for key in sectors}})
            if after[name]-before[name] < MIN_SECTOR_MM3-1e-9:
                failures.append('SECTOR_PROGRESS_BELOW_DECLARED_10_MM3_GATE:'+name)
                break

    # STOP is a high-level decision; the native cleanup below is still required.
    stop = {'id': 'STOP', 'atActionPosition': len(candidate.actions),
            'reason': 'two_declared_sector_options_attempted' if len(phases) == 2 else
                      'unpassed_sector_option'}
    if candidate.state['slots'][hand]['inserted'] or candidate.state['slots'][hand]['active']:
        # A refused first option leaves the verified corridor state intact.
        # Reuse its original checked cleanup before trying another recovery.
        source_cleanup = (accepted_sector_strokes == 0 and
                          len(candidate.actions)-len(base.actions) == len(prefix_actions))
        if source_cleanup:
            for row in access_plan['decisions'][len(prefix_actions):]:
                original = copy.deepcopy(row['action'])
                original['slot'] = hand
                accepted, _ = attempt(original, 'verified-source-access-cleanup')
                if not accepted:
                    source_cleanup = False
                    break
        if source_cleanup and not candidate.state['slots'][hand]['inserted']:
            success = True
        else:
            attempt({'slot': hand, 'operation': 'deactivate', 'duration': .2},
                    'checked-cleanup-deactivate')
            success = False
        outside = {'slot': hand, 'operation': 'withdraw',
                   'pose': copy.deepcopy(route['outsidePose']),
                   'parameters': {'radiusMm': TRANSIT_RADIUS_MM}}
        distance = float(np.linalg.norm(np.subtract(candidate.state['slots'][hand]['position'],
                                                    route['outsidePose']['position'])))
        fast = {**outside, 'duration': max(.2, min(30., distance/20.))}
        if not success:
            success, _ = attempt(fast, 'checked-cleanup-direct')
        if not success:
            slow = {**outside, 'duration': 30.}
            success, _ = attempt(slow, 'checked-cleanup-slow')
        if not success:
            retreat = {'slot': hand, 'operation': 'move', 'pose': _pose(route, 1.),
                       'parameters': {'radiusMm': TRANSIT_RADIUS_MM}, 'duration': 2.}
            moved, _ = attempt(retreat, 'checked-cleanup-corridor-return')
            if moved:
                success, _ = attempt({**outside, 'duration': 30.},
                                     'checked-cleanup-return-withdraw')
        if not success:
            failures.append('CHECKED_WITHDRAWAL_INCOMPLETE')

    metrics = candidate.metrics()
    per_sector = {name: len(set(candidate.state['removed']) & cells)*base.scene.voxel_mm3-
                         (prefix_sector[name] if prefix_sector is not None else 0.)
                  for name, cells in sectors.items()}
    delta_target = metrics['targetRemovedMm3']-(prefix_metrics or initial_metrics)['targetRemovedMm3']
    delta_non_target = metrics['nonTargetRemovedMm3']-(prefix_metrics or initial_metrics)['nonTargetRemovedMm3']
    cleanup_pending = _cleanup_pending(candidate)
    if delta_target < MIN_ADDITIONAL_TARGET_MM3-1e-9:
        failures.append('ADDITIONAL_TARGET_BELOW_DECLARED_30_MM3_GATE')
    if accepted_sector_strokes < 2:
        failures.append('FEWER_THAN_TWO_ADMITTED_SECTOR_STROKES')
    if any(value < MIN_SECTOR_MM3-1e-9 for value in per_sector.values()):
        failures.append('EACH_SECTOR_10_MM3_GATE_UNPASSED')
    if delta_non_target > MAX_ADDITIONAL_NON_TARGET_MM3+1e-9 or \
            metrics['nonTargetRemovedMm3']-initial_metrics['nonTargetRemovedMm3'] > MAX_TOTAL_NON_TARGET_MM3+1e-9:
        failures.append('NON_TARGET_CAP_EXCEEDED')
    if cleanup_pending:
        failures.append('CHECKED_CLEANUP_INCOMPLETE')
    if first_progress != list(order):
        failures.append('OBSERVED_FIRST_SECTOR_PROGRESS_ORDER_DIFFERS')
    failures = list(dict.fromkeys(failures))
    decisions = [{'index': i, 'action': copy.deepcopy(action),
                  'status': history['status'], 'reason': history.get('reason'),
                  'stateId': history['stateAfter'], 'sectorDeltaMm3': {
                      name: len(set(history.get('deltas', {}).get('removedCells', ())) & cells)*
                            base.scene.voxel_mm3 for name, cells in sectors.items()}}
                 for i, (action, history) in enumerate(zip(candidate.actions, candidate.history))]
    replay = base.fresh()
    for row in decisions:
        check()
        outcome = replay.step(row['action'], check=check)
        if outcome['status'] != row['status'] or replay.state_id != row['stateId']:
            raise ValueError('Independent intact-source strategy replay differs')
    if replay.state_id != candidate.state_id or replay.state != candidate.state:
        raise ValueError('Strategy native endpoint differs after full independent replay')
    strategy = {'hand': hand, 'sectorOrder': list(order),
                'optionIds': [name+':'+hand for name in order]+['STOP'],
                'sourceAccessMethod': access_plan['method'],
                'sourceAccessBranchId': access_plan['branchId']}
    outcome = {'deltaTargetMm3': delta_target, 'deltaNonTargetMm3': delta_non_target,
               'totalNonTargetMm3': metrics['nonTargetRemovedMm3']-initial_metrics['nonTargetRemovedMm3'],
               'perSectorRemovedMm3': per_sector,
               'taskGateMet': not failures, 'cleanupComplete': not cleanup_pending,
               'cleanupPending': cleanup_pending, 'admittedSectorStrokes': accepted_sector_strokes,
               'observedFirstProgressOrder': first_progress,
               'sourceAccessPrefixTargetMm3': (prefix_metrics or initial_metrics)['targetRemovedMm3']-
                                               initial_metrics['targetRemovedMm3'],
               'sourceAccessPrefixNonTargetMm3': (prefix_metrics or initial_metrics)['nonTargetRemovedMm3']-
                                                  initial_metrics['nonTargetRemovedMm3']}
    plan = {'schema': SCHEMA, 'task': TASK, 'method': access_plan['method']+'-'+hand+'-'+order[0]+'-first',
            'label': access_plan['label']+' · '+hand+' CUSA · '+order[0]+' sector first',
            'status': 'passed' if not failures else 'unpassed', 'failureReasons': failures,
            'route': copy.deepcopy(route), 'strategy': strategy, 'sectors': sector_evidence,
            'startStateId': base.state_id, 'finalStateId': candidate.state_id,
            'prefixActions': 0, 'sourceAccessPrefixActions': len(prefix_actions),
            'handChoiceBoundary': 'intact_before_access',
            'sectorChoiceBoundary': 'post_corridor_same_engine_same_hand',
            'sourceAccessPrefixActionSeal': semantic_digest(prefix_actions),
            'executedPrefixActionSeal': semantic_digest(candidate.actions[:len(prefix_actions)]),
            'leftSourcePrefixPreserved': hand == 'left' and
                candidate.actions[:len(prefix_actions)] == prefix_actions,
            'sectorChoiceStartStateId': prefix_state_id,
            'decisions': decisions, 'attempts': attempts, 'sectorPhases': phases,
            'highLevelStop': stop, 'continuationOutcome': outcome,
            'summary': {key: metrics.get(key) for key in ('targetRemovedMm3',
                'nonTargetRemovedMm3', 'elapsedSeconds', 'activeVesselWounds',
                'modeledVesselFluidGeneratedMm3', 'fluidPresentMm3')},
            'unassessed': ['skull opening', 'complete vessel coverage',
                           'functional boundaries', 'calibrated clinical harm'],
            'cost': {'planningSeconds': time.perf_counter()-started,
                     'executionCalls': len(decisions), 'previewCalls': len(attempts),
                     'freshReplayCalls': len(decisions), 'optimizerUpdates': 0},
            'verification': {'independentIntactSourceReplay': True,
                             'exactNativeEndpoint': True,
                             'scope': 'Modeled tissue only; complete skull access unsupported.'}}
    plan['planSeal'] = _plan_seal(base.scene.identity, base.state_id, strategy,
                                  decisions, sector_evidence)
    from .surgical_modeled_outcomes import summarize_engine_outcomes
    plan['modeledOutcomes'] = summarize_engine_outcomes(candidate, baseline=base)
    return candidate, plan


def compare_sector_strategies(engine, source_comparison, saved_branches, *,
                              acknowledge_unknowns=False,
                              max_actions_per_strategy=48,
                              max_total_non_target_mm3=200.,
                              max_additional_non_target_mm3=20.,
                              max_wall_seconds=120., check=lambda: None,
                              progress=lambda *_: None):
    """Compare all route/order/hand options and preserve every attempted result."""
    if acknowledge_unknowns is not True:
        raise ValueError('Explicit acknowledgment of missing anatomy is required')
    if type(max_actions_per_strategy) is not int or not 1 <= max_actions_per_strategy <= 48 or \
            max_total_non_target_mm3 != MAX_TOTAL_NON_TARGET_MM3 or \
            max_additional_non_target_mm3 != MAX_ADDITIONAL_NON_TARGET_MM3 or \
            not 0 < max_wall_seconds <= 120.:
        raise ValueError('Keep the declared 48-action, 200/20 mm3, 120-second bounds')
    if source_comparison.get('task') != 'anatomy-access' or \
            source_comparison.get('sourceSceneId') != engine.scene.identity:
        raise ValueError('A verified matching anatomy-access comparison is required')
    if engine.actions or any(slot['inserted'] for slot in engine.state['slots'].values()):
        raise ValueError('Strategy comparison must start from the common intact source')
    original = engine.snapshot()
    started = time.perf_counter()
    def bounded_check():
        check()
        if time.perf_counter()-started > max_wall_seconds:
            raise TimeoutError('Aggregate strategy planning wall budget exhausted')
    branches = []
    failures = []
    alternatives = source_comparison['alternatives']
    total = len(alternatives)*4
    completed = 0
    for access in alternatives:
        for order in (('near', 'far'), ('far', 'near')):
            for hand in ('left', 'right'):
                identity = access['method']+'-'+hand+'-'+order[0]+'-first'
                if time.perf_counter()-started > max_wall_seconds:
                    failures.append({'option': identity, 'reason': 'AGGREGATE_WALL_BUDGET'})
                    continue
                try:
                    if access['branchId'] not in saved_branches:
                        raise StrategyUnavailable('Saved source access branch is missing')
                    saved = SurgicalEngine.restore(engine.scene, saved_branches[access['branchId']],
                                                   check=bounded_check)
                    if saved.state_id != access['finalStateId'] or \
                            [row['action'] for row in access['decisions']] != saved.actions:
                        raise StrategyUnavailable('Saved source access actions differ from comparison')
                    sectors, evidence = _sector_cells(engine.scene, access['route'])
                    if any(row['sourceVolumeMm3'] < MIN_SECTOR_MM3
                           for row in evidence.values()):
                        raise StrategyUnavailable('PREDECLARED_SOURCE_SECTOR_CAPACITY_BELOW_10_MM3')
                    candidate, plan = _run_variant(engine, access, sectors, evidence,
                                                   hand, order, max_actions=max_actions_per_strategy,
                                                   check=bounded_check)
                    branches.append((candidate, plan))
                    if plan['status'] != 'passed':
                        failures.append({'option': identity,
                                         'reason': ';'.join(plan['failureReasons'])})
                except (StrategyUnavailable, TimeoutError) as error:
                    failures.append({'option': identity, 'reason': str(error)})
                completed += 1
                progress(completed/total,
                         'Compared '+identity+' with checked native geometry')
    if engine.snapshot() != original:
        raise ValueError('Sector comparison modified the intact source')
    report = {'schema': SCHEMA, 'task': TASK,
              'status': 'strategy_options_compared' if branches else 'no_executable_strategy',
              'sourceSceneId': engine.scene.identity,
              'sourceAccessComparisonSeal': source_comparison['comparisonSeal'],
              'startStateId': engine.state_id,
              'sourceAccessComparisonStatus': source_comparison['status'],
              'anatomicalCompleteness': 'skull, vessel and functional coverage unknown',
              'clinicalValidation': False,
              'limitations': ['Skull opening unsupported without source skull anatomy',
                              'Unmapped vessels and functional boundaries remain unknown',
                              'Simulated tissue response and harm are not clinically calibrated'],
              'declaredGate': {'additionalTargetMm3': MIN_ADDITIONAL_TARGET_MM3,
                               'eachSectorMm3': MIN_SECTOR_MM3,
                               'additionalNonTargetMm3': MAX_ADDITIONAL_NON_TARGET_MM3,
                               'totalNonTargetMm3': MAX_TOTAL_NON_TARGET_MM3,
                               'minSectorStrokes': 2, 'checkedCleanup': True},
              'limits': {'maxNativeActionsPerStrategy': max_actions_per_strategy,
                         'maxAggregateWallSeconds': max_wall_seconds},
              'optionCount': total, 'retainedCount': len(branches),
              'passedCount': sum(plan['status'] == 'passed' for _, plan in branches),
              'totalWallSeconds': time.perf_counter()-started,
              'optionFailures': failures,
              'alternatives': [plan for _, plan in branches]}
    return branches, report


def validate_saved_sector_comparison(scene, record, comparison, *, replay=False):
    """Check source/sector/action bindings; optionally replay every native branch."""
    from .surgical_planning import validate_modeled_outcome_binding
    if comparison.get('schema') != SCHEMA or comparison.get('task') != TASK or \
            comparison.get('sourceSceneId') != scene.identity or \
            comparison.get('sourceAccessComparisonSeal') != record['planning']['comparisonSeal']:
        raise ValueError('Saved sector comparison or access binding differs')
    seal = comparison.get('comparisonSeal')
    bare = {key: value for key, value in comparison.items() if key != 'comparisonSeal'}
    if seal != semantic_digest(bare):
        raise ValueError('Saved sector comparison seal differs')
    access_by_method = {row['method']: row for row in record['planning']['alternatives']}
    sector_cache = {}
    for plan in comparison['alternatives']:
        strategy = plan['strategy']
        access = access_by_method[strategy['sourceAccessMethod']]
        if plan['route'] != access['route'] or \
                strategy['sourceAccessBranchId'] != access['branchId']:
            raise ValueError('Sector option source access differs')
        prefix = _prefix_actions(access)
        if plan['sourceAccessPrefixActions'] != len(prefix) or \
                plan['sourceAccessPrefixActionSeal'] != semantic_digest(prefix):
            raise ValueError('Saved source access prefix binding differs')
        if access['method'] not in sector_cache:
            sector_cache[access['method']] = _sector_cells(scene, plan['route'])
        cells, evidence = sector_cache[access['method']]
        if evidence != plan['sectors'] or \
                plan['planSeal'] != _plan_seal(scene.identity, plan['startStateId'],
                                               strategy, plan['decisions'], evidence):
            raise ValueError('Saved source sector membership or strategy seal differs')
        branch = plan['branchId']
        snapshot = record['branches'][branch]
        if snapshot['sceneId'] != scene.identity or \
                snapshot['stateId'] != plan['finalStateId'] or snapshot['actions'] != \
                [row['action'] for row in plan['decisions']]:
            raise ValueError('Saved sector branch endpoint or actions differ')
        if plan['executedPrefixActionSeal'] != semantic_digest(snapshot['actions'][:len(prefix)]) or \
                plan['leftSourcePrefixPreserved'] != (strategy['hand'] == 'left' and
                    snapshot['actions'][:len(prefix)] == prefix):
            raise ValueError('Saved strategy prefix hand/action identity differs')
        validate_modeled_outcome_binding(scene, snapshot, plan['modeledOutcomes'],
                                         plan['startStateId'])
        outcome = plan['continuationOutcome']
        measured_sector = {name: sum(row['sectorDeltaMm3'][name]
                                      for row in plan['decisions'][len(prefix):])
                           for name in ('near', 'far')}
        if any(abs(measured_sector[name]-outcome['perSectorRemovedMm3'][name]) > 1e-9
               for name in measured_sector):
            raise ValueError('Saved strategy measured sector totals differ')
        phases = plan['sectorPhases']
        ordered = (len(phases) == 2 and
                   [row['sector'] for row in phases] == strategy['sectorOrder'] and
                   outcome['observedFirstProgressOrder'] == strategy['sectorOrder'] and
                   phases[0]['perSectorDeltaMm3'][strategy['sectorOrder'][1]] == 0 and
                   all(row['perSectorDeltaMm3'][row['sector']] >= MIN_SECTOR_MM3
                       for row in phases))
        gate = (outcome['deltaTargetMm3'] >= MIN_ADDITIONAL_TARGET_MM3 and
                outcome['deltaNonTargetMm3'] <= MAX_ADDITIONAL_NON_TARGET_MM3 and
                outcome['totalNonTargetMm3'] <= MAX_TOTAL_NON_TARGET_MM3 and
                all(value >= MIN_SECTOR_MM3
                    for value in outcome['perSectorRemovedMm3'].values()) and
                outcome['admittedSectorStrokes'] >= 2 and
                outcome['cleanupComplete'] is True and not outcome['cleanupPending'] and
                not plan['failureReasons'] and ordered and len(snapshot['actions']) <= 48)
        if outcome['taskGateMet'] is not gate or (plan['status'] == 'passed') is not gate:
            raise ValueError('Saved strategy task gate differs from fixed measured bounds')
        if replay:
            native = SurgicalEngine.restore(scene, snapshot)
            if native.state_id != plan['finalStateId']:
                raise ValueError('Saved sector native endpoint replay differs')
            if plan['sectorChoiceStartStateId'] is not None and \
                    native.history[len(prefix)-1]['stateAfter'] != plan['sectorChoiceStartStateId']:
                raise ValueError('Saved sector decision boundary state differs')
            for row, history in zip(plan['decisions'], native.history):
                removed = set(history.get('deltas', {}).get('removedCells', ()))
                actual = {name: len(removed & group)*scene.voxel_mm3 for name, group in cells.items()}
                if row['stateId'] != history['stateAfter'] or row['sectorDeltaMm3'] != actual:
                    raise ValueError('Saved sector action membership delta differs')
    return {'verified': True, 'alternatives': len(comparison['alternatives'])}
