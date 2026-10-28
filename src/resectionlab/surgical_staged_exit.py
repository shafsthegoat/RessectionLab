"""Bounded native CUSA exits for staged, source-bound research rehearsals.

Every proposal runs on a clone through public preview/step via checked_actions.
Failed probes retain their receipts and action prefixes but never become the
returned branch. This module changes neither collision physics nor task gates.
"""
from __future__ import annotations

import copy
from dataclasses import replace
import math

import numpy as np
from scipy.spatial.transform import Rotation, Slerp

from .surgical_engine import SurgicalEngine, TOOLS
from .surgical_staged_skills import SkillLimits, checked_actions
from .surgical_transition_meter import MeteredEngine, TransitionMeter


SCHEMA = 'resectionlab.checked-staged-exit.v1'
MAX_PATH_POSES_PER_MOVE = 24
MAX_REVERSE_POSES = 240
MAX_NATIVE_REVERSE_POSES = 2048
MAX_EXIT_ACTIONS = 240
MAX_ACTIVE_REPEATS_PER_POSE = 8
MAX_INTERPOLATED_STEP_MM = .5
MAX_INTERPOLATED_ROTATION_DEG = 2.
FULL_TOOL_ENVELOPE_MM = 42.
POSE_TOLERANCE_MM = 1e-5


def _pose(value):
    if not isinstance(value, dict) or set(value) != {'position', 'quaternion'}:
        raise ValueError('Exit requires an exact RAS pose')
    position = np.asarray(value['position'], dtype=float)
    quaternion = np.asarray(value['quaternion'], dtype=float)
    if (position.shape != (3,) or quaternion.shape != (4,) or
            not np.isfinite(position).all() or not np.isfinite(quaternion).all() or
            abs(np.linalg.norm(quaternion)-1.) > 1e-5):
        raise ValueError('Exit pose is not finite and normalized')
    return {'position': position.tolist(), 'quaternion': quaternion.tolist()}


def _same_orientation(a, b):
    q1, q2 = np.asarray(a, float), np.asarray(b, float)
    return abs(float(np.dot(q1, q2))) >= 1.-1e-6


def _slot(engine, hand):
    return engine.state['slots'][hand]


def _point(engine, hand):
    return np.asarray(_slot(engine, hand)['position'], float)


def _matches_slot_pose(engine, hand, pose):
    slot = _slot(engine, hand)
    return (_distance(slot['position'], pose['position']) <= POSE_TOLERANCE_MM and
            _same_orientation(slot['quaternion'], pose['quaternion']))


def _distance(a, b):
    return float(np.linalg.norm(np.subtract(a, b)))


def _duration(distance, speed=2.):
    return max(.2, min(30., float(distance)/speed))


def _full_exit(engine, hand, outside):
    slots = engine.state['slots']
    return (not engine.state['holds'] and
            all(not row['inserted'] and not row['active'] and not row['grip']
                for row in slots.values()) and
            _distance(slots[hand]['position'], outside['position']) <= POSE_TOLERANCE_MM and
            _same_orientation(slots[hand]['quaternion'], outside['quaternion']))


def _reverse_accepted_poses(engine, hand):
    """Reverse observed native CUSA poses, including lateral fan poses.

    No source raster, tumor membership or future collision outcome enters this
    path. A partial multi-pose action lacks enough history to reconstruct its
    traversed waypoints and is deliberately unsupported.
    """
    tips = [_pose({'position': _slot(engine, hand)['position'],
                   'quaternion': _slot(engine, hand)['quaternion']})]
    found_place = False
    for row in reversed(engine.history):
        action = row['action']
        if action.get('slot') != hand:
            continue
        if action['operation'] == 'exchange':
            break
        if action['operation'] not in ('place', 'move', 'apply'):
            continue
        if row.get('status') == 'partial':
            # contactPosition may be the rejected next microstep, and the
            # requested quaternion is not the accepted partial orientation.
            # History lacks an observed post-action pose for older records.
            # Refuse reconstruction rather than invent a reverse waypoint.
            return [], 'PARTIAL_POSE_HISTORY_UNSUPPORTED'
        else:
            encountered = action.get('path', [action['pose']]
                if 'pose' in action else [])
        if not encountered:
            return [], 'SOURCE_POSE_HISTORY_INCOMPLETE'
        if action['operation'] == 'place':
            found_place = True
        for raw in reversed(encountered):
            pose = _pose(raw)
            if (_distance(pose['position'], tips[-1]['position']) > POSE_TOLERANCE_MM or
                    not _same_orientation(pose['quaternion'], tips[-1]['quaternion'])):
                tips.append(pose)
    if not found_place or len(tips) < 2:
        return [], 'SOURCE_PLACE_OR_WAYPOINTS_UNAVAILABLE'
    sampled = []
    for a, b in zip(tips, tips[1:]):
        distance = _distance(a['position'], b['position'])
        rotations = Rotation.from_quat([a['quaternion'], b['quaternion']])
        angle = float((rotations[1]*rotations[0].inv()).magnitude())
        count = max(1, math.ceil(distance/MAX_INTERPOLATED_STEP_MM),
                    math.ceil(math.degrees(angle)/MAX_INTERPOLATED_ROTATION_DEG))
        if len(sampled)+count > MAX_REVERSE_POSES:
            return [], 'REVERSE_POSE_BUDGET'
        start, end = np.asarray(a['position']), np.asarray(b['position'])
        interpolation = Slerp([0., 1.], rotations)
        for index in range(1, count+1):
            fraction = index/count
            point = start+(end-start)*fraction
            sampled.append({'position': point.tolist(),
                            'quaternion': interpolation(fraction).as_quat().tolist()})
    return sampled, None


def _pose_path_duration(start, poses):
    previous = start
    distance = 0.
    angle = 0.
    for pose in poses:
        distance += _distance(previous['position'], pose['position'])
        r1, r2 = Rotation.from_quat(previous['quaternion']), Rotation.from_quat(pose['quaternion'])
        angle += float((r2*r1.inv()).magnitude())
        previous = pose
    # Angular travel sweeps the full CUSA shaft, even when the tip is fixed.
    # Add that envelope travel to translation before applying the 2 mm/s
    # duration, so a zero-rotation path does not collapse to the 0.2 s floor.
    return max(.2, min(30., (distance+FULL_TOOL_ENVELOPE_MM*angle)/2.))


def _path_chunks(start, poses):
    chunks = []
    previous = start
    for offset in range(0, len(poses), MAX_PATH_POSES_PER_MOVE):
        chunk = copy.deepcopy(poses[offset:offset+MAX_PATH_POSES_PER_MOVE])
        chunks.append((chunk, _pose_path_duration(previous, chunk)))
        previous = chunk[-1]
    return chunks


def _reverse_native_steps(engine, hand, check):
    """Reconstruct accepted legacy poses, binding a latest partial to observation.

    A geometric SHAFT_COLLISION may leave a committed prefix. Its native step
    fraction identifies a finite command prefix, but does not certify it: the
    reconstructed endpoint must also match the currently observed tip pose.
    Earlier partial motion and contact-limited models remain unsupported.
    Every reverse command still requires fresh, unchanged native checks.
    """
    if 'cusaV3' in engine.scene.properties:
        return [], 'CONTACT_LIMITED_POSE_HISTORY_UNAVAILABLE'
    history = [row for row in engine.history if row['action'].get('slot') == hand]
    place = next((index for index in range(len(history)-1, -1, -1)
                  if history[index]['action']['operation'] == 'place'), None)
    if place is None:
        return [], 'SOURCE_PLACE_UNAVAILABLE'
    moving = [index for index, row in enumerate(history) if
              row['action']['operation'] in ('apply', 'move') and
              ('pose' in row['action'] or 'path' in row['action'])]
    last_motion = moving[-1] if moving else None
    current_parameters = engine.parameters(engine.state['slots'][hand])
    poses = [_pose(history[place]['action']['pose'])]
    for row_index, row in enumerate(history[place+1:], place+1):
        check()
        action = row['action']
        if action['operation'] == 'exchange':
            return [], 'SOURCE_EXCHANGE_WITHOUT_PLACE'
        if action['operation'] == 'withdraw':
            return [], 'SOURCE_WITHDRAWAL_WITHOUT_PLACE'
        if action['operation'] not in ('apply', 'move') and ('pose' in action or 'path' in action):
            return [], 'UNSUPPORTED_MOVING_POSE_HISTORY'
        if any(key in action.get('parameters', {}) and
               action['parameters'][key] != current_parameters[key]
               for key in ('radiusMm', 'depthMm')):
            return [], 'SOURCE_WORKING_GEOMETRY_CHANGED'
        if action['operation'] not in ('apply', 'move'):
            continue
        partial = row['status'] == 'partial'
        if row['status'] != 'accepted' and not (
                partial and row_index == last_motion and row.get('reason') == 'SHAFT_COLLISION'):
            return [], 'PARTIAL_POSE_HISTORY_UNSUPPORTED'
        points = action.get('path', [action['pose']] if 'pose' in action else [])
        previous = poses[-1]
        segments = []
        for endpoint in points or [previous]:
            endpoint = _pose(endpoint)
            distance = _distance(previous['position'], endpoint['position'])
            rotations = Rotation.from_quat([previous['quaternion'], endpoint['quaternion']])
            angle = float((rotations[1]*rotations[0].inv()).magnitude())
            n = max(1, math.ceil((distance+40.*angle)/min(
                .4, float(engine.scene.spacing.min())/3.)),
                math.ceil(float(action.get('duration', 1.))/max(1, len(points))/.1))
            segments.append((previous, endpoint, n, distance, angle))
            previous = endpoint
        accepted_steps = sum(segment[2] for segment in segments)
        if partial:
            fraction = row.get('completedFraction')
            if type(fraction) not in (int, float) or not math.isfinite(fraction) or not 0 <= fraction < 1:
                return [], 'PARTIAL_NATIVE_STEP_FRACTION_INVALID'
            completed = fraction*accepted_steps
            if abs(completed-round(completed)) > 1e-6:
                return [], 'PARTIAL_NATIVE_STEP_FRACTION_INVALID'
            accepted_steps = round(completed)
        for previous, endpoint, n, distance, angle in segments:
            take = min(n, accepted_steps)
            accepted_steps -= take
            if distance >= 1e-10 or angle >= 1e-10:
                if len(poses)+take > MAX_NATIVE_REVERSE_POSES:
                    return [], 'NATIVE_REVERSE_POSE_BUDGET'
                rotations = Rotation.from_quat([previous['quaternion'], endpoint['quaternion']])
                interpolation = Slerp([0., 1.], rotations)
                a, b = np.asarray(previous['position']), np.asarray(endpoint['position'])
                for index in range(1, take+1):
                    check()
                    fraction = index/n
                    poses.append({'position': ((1.-fraction)*a+fraction*b).tolist(),
                        'quaternion': interpolation(fraction).as_quat().tolist()})
            if take < n:
                break
        if partial:
            observed = engine.state['slots'][hand]
            if (np.linalg.norm(np.subtract(poses[-1]['position'], observed['position'])) > 1e-7 or
                    (Rotation.from_quat(poses[-1]['quaternion'])*
                     Rotation.from_quat(observed['quaternion']).inv()).magnitude() > 1e-7):
                return [], 'PARTIAL_OBSERVED_ENDPOINT_DIFFERS'
    if len(poses) < 2 or not _matches_slot_pose(engine, hand, poses[-1]):
        return [], 'RECONSTRUCTED_NATIVE_ENDPOINT_DIFFERS'
    return list(reversed(poses[:-1])), None


def _probe(source, actions, limits, meter, check, phase):
    before = meter.calls
    candidate, checked = checked_actions(source, actions, limits=limits,
        meter=meter, check=check, phase=phase)
    return candidate, {'phase': phase, 'checked': checked,
        'probeFinalStateId': candidate.state_id,
        'probeActions': copy.deepcopy(candidate.actions[len(source.actions):]),
        'publicCalls': meter.calls-before,
        'probeNativeActions': len(candidate.actions)}


def _brain_delta(source, candidate):
    cells = set(map(int, candidate.state['removed']))-set(map(int, source.state['removed']))
    brain = {int(row['id']) for row in source.scene.regions
             if row['material'] == 'soft_tissue' and
             (row['name'].strip().lower() in ('brain', 'whole brain') or
              row['name'].strip().lower().startswith('mapped brain'))}
    labels = source.scene.labels.ravel()
    return float(sum(int(labels[cell]) in brain for cell in cells)*source.scene.voxel_mm3)


def find_checked_exit(engine, route, *, hand, limits, meter, check):
    """Return a complete checked exit or an untouched-prefix clone and refusal.

    Stage order is fixed: three direct durations, full inactive reversed native
    pose history, then active CUSA aspiration along that same history. Active
    recovery may remove tissue only through the unchanged public native effect
    with the current tip radius, original 0.5 mm shaft and the caller's caps.
    """
    if (not isinstance(engine, (SurgicalEngine, MeteredEngine)) or
            not isinstance(route, dict) or hand not in ('left', 'right') or
            not isinstance(limits, SkillLimits) or not isinstance(meter, TransitionMeter) or
            not callable(check)):
        raise ValueError('Checked exit requires source engine, route, hand, limits, meter and check')
    if isinstance(engine, MeteredEngine) and engine.meter is not meter:
        raise ValueError('Checked exit cannot change native transition meter')
    if (limits.max_additional_non_target_mm3 is None or
            limits.max_additional_non_target_mm3 > 20.+1e-9 or
            limits.baseline_non_target_mm3 is None or
            limits.max_total_non_target_mm3 > 200.+1e-9):
        raise ValueError('Checked exit requires the original 20/200 mm3 non-target caps')
    if route.get('sourceSceneId') != engine.scene.identity or \
            route.get('pathType') != 'oblique_mapped_brain':
        raise ValueError('Exit route is not bound to the native source scene')
    outside = _pose(route.get('outsidePose'))
    current = _slot(engine, hand)
    if (current['instrument'] != 'cusa' or not current['inserted'] or
            TOOLS['cusa']['shaftRadiusMm'] != .5):
        raise ValueError('Checked exit requires the observed inserted fixed-shaft CUSA pose')
    source = (engine if isinstance(engine, MeteredEngine) else meter.wrap(engine)).clone()
    source_id, source_actions, source_metrics = (
        source.state_id, len(source.actions), source.metrics())
    if (source_actions >= min(limits.max_native_actions, MAX_EXIT_ACTIONS) or
            source_metrics['nonTargetRemovedMm3'] > limits.max_total_non_target_mm3+1e-9 or
            source_metrics['nonTargetRemovedMm3']-limits.baseline_non_target_mm3 >
            limits.max_additional_non_target_mm3+1e-9):
        raise ValueError('Observed exit prefix already exceeds action or tissue caps')
    effective = replace(limits, max_native_actions=min(limits.max_native_actions, MAX_EXIT_ACTIONS))
    first_call = meter.calls
    receipt = {'schema': SCHEMA, 'status': 'refused', 'reason': None,
        'sourceSceneId': source.scene.identity, 'sourceStateId': source_id,
        'sourceNativeActions': source_actions, 'hand': hand,
        'outsidePose': copy.deepcopy(outside), 'mode': None,
        'stages': [], 'appliedExitActions': [], 'additionalBrainMm3SincePrefix': 0.,
        'sourceTipRadiusMm': float(source.parameters(current)['radiusMm']),
        'sourceShaftRadiusMm': .5, 'maxPathPosesPerMove': MAX_PATH_POSES_PER_MOVE,
        'maxReversePoses': MAX_REVERSE_POSES, 'maxAppendedActions': MAX_EXIT_ACTIONS-source_actions}

    def accepted(candidate, stage, mode):
        if stage['checked']['status'] != 'accepted' or not _full_exit(candidate, hand, outside):
            return False
        metrics = candidate.metrics()
        if (metrics['nonTargetRemovedMm3'] > effective.max_total_non_target_mm3+1e-9 or
                metrics['nonTargetRemovedMm3']-effective.baseline_non_target_mm3 >
                effective.max_additional_non_target_mm3+1e-9):
            return False
        if candidate.actions[:source_actions] != source.actions or candidate.state_id == source_id:
            raise ValueError('Checked exit altered source ancestry or did not change native state')
        receipt.update(status='accepted', reason=None, mode=mode,
            finalStateId=candidate.state_id,
            appliedExitActions=copy.deepcopy(candidate.actions[source_actions:]),
            additionalBrainMm3SincePrefix=_brain_delta(source, candidate),
            additionalNonTargetMm3SincePrefix=float(metrics['nonTargetRemovedMm3']-
                source_metrics['nonTargetRemovedMm3']),
            finalNativeActions=len(candidate.actions),
            publicCallCost=meter.calls-first_call)
        return True

    distance = _distance(current['position'], outside['position'])
    durations = []
    for value in (_duration(distance, 20.), _duration(distance, 2.), 30.):
        if not any(abs(value-existing) < 1e-9 for existing in durations):
            durations.append(value)
    for index, duration in enumerate(durations):
        actions = [
            {'slot': hand, 'operation': 'deactivate', 'duration': .2},
            {'slot': hand, 'operation': 'withdraw', 'pose': copy.deepcopy(outside),
             'duration': duration}]
        candidate, stage = _probe(source, actions, effective, meter, check,
            f'staged.exit.direct.{index}')
        stage['withdrawDurationSeconds'] = duration
        receipt['stages'].append(stage)
        if accepted(candidate, stage, 'direct'):
            return candidate, receipt
        if stage['checked']['status'] == 'interrupted':
            receipt['reason'] = stage['checked']['reason']
            break
    else:
        native_poses, native_reason = _reverse_native_steps(source, hand, check)
        receipt['nativeStepReverse'] = {'status': 'available' if native_poses else 'unavailable',
            'reason': native_reason, 'poseCount': len(native_poses),
            'maxPoses': MAX_NATIVE_REVERSE_POSES,
            'scope': 'reconstructed accepted legacy CUSA integration poses; all commands require fresh native checks'}
        if native_poses:
            chunks = [native_poses[index:index+MAX_PATH_POSES_PER_MOVE]
                      for index in range(0, len(native_poses), MAX_PATH_POSES_PER_MOVE)]
            for active in (False, True):
                actions = [{'slot': hand, 'operation': 'deactivate', 'duration': .2}]
                actions.extend({'slot': hand, 'operation': 'apply' if active else 'move',
                    'path': copy.deepcopy(chunk), 'duration': .1*len(chunk)} for chunk in chunks)
                actions.extend([
                    {'slot': hand, 'operation': 'deactivate', 'duration': .2},
                    {'slot': hand, 'operation': 'withdraw', 'pose': copy.deepcopy(outside),
                     'duration': _duration(_distance(native_poses[-1]['position'], outside['position']))}])
                candidate, stage = _probe(source, actions, effective, meter, check,
                    'staged.exit.native_steps_'+('active' if active else 'inactive'))
                stage['nativePoseCount'] = len(native_poses)
                receipt['stages'].append(stage)
                if accepted(candidate, stage, 'native_steps_active' if active else 'native_steps_inactive'):
                    return candidate, receipt
                if stage['checked']['status'] == 'interrupted':
                    receipt['reason'] = stage['checked']['reason']
                    receipt['finalStateId'] = source_id
                    receipt['finalNativeActions'] = source_actions
                    receipt['publicCallCost'] = meter.calls-first_call
                    return source.clone(), receipt
        poses, path_reason = _reverse_accepted_poses(source, hand)
        receipt['reversePath'] = {'status': 'available' if poses else 'unavailable',
            'reason': path_reason, 'sampledPoseCount': len(poses),
            'maxTranslationPerPoseMm': MAX_INTERPOLATED_STEP_MM,
            'maxRotationPerPoseDeg': MAX_INTERPOLATED_ROTATION_DEG}
        if poses:
            chunks = _path_chunks({'position': current['position'],
                                   'quaternion': current['quaternion']}, poses)
            inactive = [{'slot': hand, 'operation': 'deactivate', 'duration': .2}]
            inactive.extend({'slot': hand, 'operation': 'move', 'path': chunk,
                             'duration': duration} for chunk, duration in chunks)
            inactive.append({'slot': hand, 'operation': 'withdraw',
                'pose': copy.deepcopy(outside),
                'duration': _duration(_distance(poses[-1]['position'],
                                               outside['position']))})
            candidate, stage = _probe(source, inactive, effective, meter, check,
                                      'staged.exit.inactive_reverse')
            stage['reverseChunkCount'] = len(chunks)
            receipt['stages'].append(stage)
            if accepted(candidate, stage, 'inactive_reverse'):
                return candidate, receipt
            if stage['checked']['status'] != 'interrupted':
                active = source
                # A stationary public CUSA effect can clear the current tip
                # before translating. Every effect is metered and cap checked.
                current_pose = {'position': current['position'],
                                'quaternion': current['quaternion']}
                candidate, stage = _probe(active, [{'slot': hand,
                    'operation': 'apply', 'pose': copy.deepcopy(current_pose),
                    'duration': .2}], effective, meter, check,
                    'staged.exit.active_dwell')
                receipt['stages'].append(stage)
                if stage['checked']['status'] == 'accepted':
                    active = candidate
                    active_failure = None
                    appended = len(active.actions)-source_actions
                    for index, waypoint in enumerate(poses):
                        for repetition in range(MAX_ACTIVE_REPEATS_PER_POSE):
                            if len(active.actions) >= effective.max_native_actions or \
                                    appended >= MAX_EXIT_ACTIONS:
                                active_failure = 'NATIVE_ACTION_BUDGET'; break
                            remaining = _distance(_point(active, hand),
                                                  waypoint['position'])
                            if _matches_slot_pose(active, hand, waypoint):
                                break
                            action = {'slot': hand, 'operation': 'apply',
                                'pose': copy.deepcopy(waypoint),
                                'duration': _pose_path_duration(
                                    {'position': _slot(active, hand)['position'],
                                     'quaternion': _slot(active, hand)['quaternion']},
                                    [waypoint])}
                            before_position = _point(active, hand)
                            before_quaternion = list(_slot(active, hand)['quaternion'])
                            before_removed = set(map(int, active.state['removed']))
                            candidate, stage = _probe(active, [action], effective,
                                meter, check, f'staged.exit.active_reverse.{index}')
                            stage['waypointIndex'] = index
                            stage['repetition'] = repetition
                            receipt['stages'].append(stage)
                            if stage['checked']['status'] != 'accepted':
                                active_failure = stage['checked']['reason'] or 'ACTIVE_REVERSE_REFUSED'
                                break
                            active = candidate
                            appended = len(active.actions)-source_actions
                            if (_distance(_point(active, hand), before_position) <=
                                    POSE_TOLERANCE_MM and
                                    _same_orientation(_slot(active, hand)['quaternion'],
                                                      before_quaternion) and
                                    set(map(int, active.state['removed'])) == before_removed):
                                active_failure = 'ACTIVE_REVERSE_STALLED'
                                break
                        else:
                            active_failure = 'ACTIVE_REVERSE_REPEAT_BUDGET'
                        if active_failure:
                            break
                    if active_failure is None:
                        tail = [
                            {'slot': hand, 'operation': 'deactivate', 'duration': .2},
                            {'slot': hand, 'operation': 'withdraw',
                             'pose': copy.deepcopy(outside),
                             'duration': _duration(_distance(_point(active, hand),
                                                            outside['position']))}]
                        candidate, stage = _probe(active, tail, effective, meter,
                            check, 'staged.exit.active_final_withdraw')
                        receipt['stages'].append(stage)
                        if accepted(candidate, stage, 'active_reverse'):
                            return candidate, receipt
                        active_failure = stage['checked']['reason'] or 'ACTIVE_FINAL_WITHDRAW_REFUSED'
                    receipt['reason'] = active_failure
                else:
                    receipt['reason'] = stage['checked']['reason'] or 'ACTIVE_DWELL_REFUSED'
            else:
                receipt['reason'] = stage['checked']['reason']
        else:
            receipt['reason'] = path_reason
    receipt['reason'] = receipt['reason'] or 'ALL_CHECKED_EXIT_STAGES_REFUSED'
    receipt['finalStateId'] = source_id
    receipt['finalNativeActions'] = source_actions
    receipt['publicCallCost'] = meter.calls-first_call
    # The returned failure candidate is the untouched starting prefix. Every
    # discarded branch is reconstructible from stage probeActions and receipts.
    return source.clone(), receipt


def recover_checked_cusa(engine, *, check=lambda: None):
    """One metered withdrawal from the admitted current state, with no new brain.

    The last actual outside placement supplies the route. No caller-provided
    endpoint, anatomy baseline, or guard override enters this operation.
    """
    import time
    from .core import semantic_digest
    from .surgical_sector_planning import _cleanup_pending
    hands = [key for key, slot in engine.state['slots'].items()
             if slot['inserted'] and slot['instrument'] == 'cusa']
    if len(hands) != 1 or 'cusaV3' in engine.scene.properties:
        raise ValueError('Checked recovery requires one inserted legacy CUSA')
    hand = hands[0]
    if engine.state['holds'] or any(key != hand and (slot['inserted'] or slot['active'] or slot['grip'])
            for key, slot in engine.state['slots'].items()):
        raise ValueError('Release held tissue and withdraw the other hand before checked CUSA recovery')
    if len(engine.actions) > 180 or engine.metrics()['nonTargetRemovedMm3'] > 200.:
        raise ValueError('Current source exceeds the bounded checked-recovery action or tissue budget')
    place = next((row['action'] for row in reversed(engine.history)
                  if row['action'].get('slot') == hand and row['action']['operation'] == 'place'), None)
    if place is None:
        raise ValueError('An observed outside placement is required for checked recovery')
    source = engine.snapshot()
    before = engine.metrics()
    started = time.monotonic()
    def bounded():
        check()
        if time.monotonic()-started > 45.:
            raise TimeoutError('Checked withdrawal 45 second wall budget exhausted')
    meter = TransitionMeter(240, check=bounded)
    wrapped = meter.wrap(engine)
    limits = SkillLimits(240, 240, max_additional_non_target_mm3=0.,
        baseline_non_target_mm3=before['nonTargetRemovedMm3'])
    route = {'sourceSceneId': engine.scene.identity, 'pathType': 'oblique_mapped_brain',
             'outsidePose': copy.deepcopy(place['pose'])}
    with meter.reserve(len(engine.actions)+50):
        final, receipt = find_checked_exit(wrapped, route, hand=hand, limits=limits,
            meter=meter, check=bounded)
    proposed_pending = _cleanup_pending(final)
    accepted = receipt['status'] == 'accepted' and not proposed_pending
    refusal_reason = receipt.get('reason')
    if receipt['status'] == 'accepted' and proposed_pending:
        refusal_reason = 'FIELD_CLEANUP_INCOMPLETE: '+','.join(proposed_pending)
    if accepted and meter.remaining < len(final.actions):
        # The reverse path can append more than the fixed 50-call cushion.
        # Never start an unaffordable mandatory replay or lose its diagnostic
        # receipt. The original source replay remains covered by the reserve.
        accepted = False
        refusal_reason = (f'FINAL_PROGRAM_REPLAY_HEADROOM: required {len(final.actions)}, '
                          f'remaining {meter.remaining}')
    if not accepted:
        # Geometric withdrawal still requires global cleanup and affordable
        # cold replay. Retain its speculative receipt and the original source.
        final = wrapped.clone()
    replay = meter.wrap(engine.fresh())
    with meter.phase('checked_recovery.independent_full_program_replay'):
        for action in final.actions:
            if replay.step(action, check=bounded)['status'] == 'rejected':
                raise ValueError('Checked recovery native program no longer replays')
    if replay.snapshot() != final.snapshot() or replay.history != final.history:
        raise ValueError('Checked recovery native checkpoint or history differs')
    pending = _cleanup_pending(final)
    report = {'schema': 'resectionlab.checked-cusa-recovery.v1',
        'sourceSceneId': engine.scene.identity, 'sourceStateId': engine.state_id,
        'sourceSnapshot': source, 'sourceProgramSeal': semantic_digest(source),
        'sourceNativeActions': len(engine.actions), 'finalStateId': final.state_id,
        'finalNativeActions': len(final.actions), 'hand': hand,
        'sourceMetrics': {key: before[key] for key in ('targetRemovedMm3', 'nonTargetRemovedMm3')},
        'finalMetrics': {key: final.metrics()[key] for key in ('targetRemovedMm3', 'nonTargetRemovedMm3')},
        'status': 'accepted' if accepted else 'refused', 'mode': receipt.get('mode') if accepted else None,
        'reason': refusal_reason, 'cleanupPending': pending,
        'finalNativeHistory': copy.deepcopy(final.history),
        'exactNativeReplay': True, 'receipt': receipt, 'cost': meter.report(),
        'wallSeconds': time.monotonic()-started,
        'scope': 'Checked instrument withdrawal, not a resection-task or clinical outcome; zero additional non-target removal'}
    report['stoppedExecutions'] = [{'meterBranchId': int(key),
        'retentionRole': 'unadmitted_speculative_native_prefix', 'nativeSnapshot': prefix.snapshot()}
        for key in meter.report()['branches']
        if (prefix := meter.last_committed_engine(branch_id=int(key))) is not None and
            prefix.actions != final.actions[:len(prefix.actions)]]
    bounded()
    return final.unwrap(), report


def validate_exit_recovery(scene, record, report):
    """Bind persisted recovery evidence to native source/final action ancestry."""
    from .core import semantic_digest
    from .surgical_sector_planning import _cleanup_pending
    from types import SimpleNamespace
    if ('cusaV3' in scene.properties or report.get('schema') != 'resectionlab.checked-cusa-recovery.v1' or
            report.get('sourceSceneId') != scene.identity or
            report.get('comparisonSeal') != semantic_digest({k: v for k, v in report.items() if k != 'comparisonSeal'}) or
            report.get('exactNativeReplay') is not True):
        raise ValueError('Checked withdrawal source or report seal differs')
    source = report['sourceSnapshot']
    parent = record['branches'][report['sourceBranchId']]
    final = (record['branches'][report['finalBranchId']] if report['status'] == 'accepted' else source)
    position = report['sourceNativeActions']
    hand = report['hand']
    source_slot = source['checkpoint']['state']['slots'][hand]
    source_parameters = source_slot['parameters']
    geometry = {'radiusMm': source_parameters.get('radiusMm', TOOLS['cusa']['radiusMm']),
                'depthMm': source_parameters.get('depthMm', 2.)}
    if source_slot['instrument'] != 'cusa' or not source_slot['inserted']:
        raise ValueError('Checked withdrawal source hand differs')
    if (source['actions'] != parent['actions'][:position] or len(source['actions']) != position or
            parent['sceneId'] != scene.identity or
            parent.get('protectionContract') != source.get('protectionContract') or
            position == len(parent['actions']) and parent['stateId'] != source['stateId'] or
            source['stateId'] != report['sourceStateId'] or
            semantic_digest(source) != report['sourceProgramSeal'] or
            final['actions'][:position] != source['actions'] or
            len(final['actions']) != report['finalNativeActions'] or final['stateId'] != report['finalStateId']):
        raise ValueError('Checked withdrawal native ancestry differs')
    targets = {int(row['id']) for row in scene.regions if row.get('target')}
    for snapshot, metrics in ((source, report['sourceMetrics']), (final, report['finalMetrics'])):
        state = snapshot['checkpoint']['state']
        identity = {'scene': scene.identity, 'state': state}
        if 'protectionContract' in snapshot:
            identity['protectionContract'] = snapshot['protectionContract']['contentHash']
        cells = set(state['removed'])
        target = sum(int(scene.labels.flat[i]) in targets for i in cells)*scene.voxel_mm3
        if (snapshot['sceneId'] != scene.identity or semantic_digest(identity) != snapshot['stateId'] or
                snapshot['checkpoint']['stateId'] != snapshot['stateId'] or
                snapshot['checkpoint']['position'] != len(snapshot['actions']) or
                snapshot.get('protectionContract') != source.get('protectionContract') or
                metrics != {'targetRemovedMm3': target, 'nonTargetRemovedMm3': len(cells)*scene.voxel_mm3-target}):
            raise ValueError('Checked withdrawal native checkpoint or tissue ledger differs')
    history = record.get('timeline', {}).get(report['finalBranchId'], report['finalNativeHistory'])
    if report['status'] != 'accepted':
        history = history[:len(final['actions'])]
    if (len(history) != len(final['actions']) or
            position and history[position-1]['stateAfter'] != source['stateId'] or
            final['actions'] and history[-1]['stateAfter'] != final['stateId'] or
            any(event['action'] != action or index and event['stateBefore'] != history[index-1]['stateAfter']
                for index, (action, event) in enumerate(zip(final['actions'], history)))):
        raise ValueError('Checked withdrawal native history binding differs')
    source_cells, final_cells = set(), set()
    for index, event in enumerate(history):
        delta = event.get('deltas', {}).get('removedCells', [])
        final_cells.update(delta)
        if index < position:
            source_cells.update(delta)
    if (source_cells != set(source['checkpoint']['state']['removed']) or
            final_cells != set(final['checkpoint']['state']['removed'])):
        raise ValueError('Checked withdrawal native history effects differ')
    pending = _cleanup_pending(SimpleNamespace(state=final['checkpoint']['state']))
    accepted = report['status'] == 'accepted'
    if (report['status'] not in ('accepted', 'refused') or report['cleanupPending'] != pending or
            accepted and pending or
            report['finalMetrics']['nonTargetRemovedMm3'] > report['sourceMetrics']['nonTargetRemovedMm3']+1e-9 or
            report['finalMetrics']['nonTargetRemovedMm3'] > 200. or
            not accepted and final != source):
        raise ValueError('Checked withdrawal cleanup status or original zero-additional cap differs')
    if accepted:
        place = next((action for action in reversed(source['actions'])
            if action.get('slot') == hand and action['operation'] == 'place'), None)
        if (record.get('parents', {}).get(report['finalBranchId']) != {
                'branchId': report['sourceBranchId'], 'position': position} or
                report['receipt']['status'] != 'accepted' or place is None or
                report['receipt']['outsidePose'] != place['pose'] or
                not _matches_slot_pose(SimpleNamespace(state=final['checkpoint']['state']), hand, place['pose']) or
                any(action.get('slot') != hand or action['operation'] not in ('deactivate', 'move', 'apply', 'withdraw') or
                    set(action.get('parameters', {}))-set(geometry) or
                    any(value != geometry[key] for key, value in action.get('parameters', {}).items())
                    for action in final['actions'][position:])):
            raise ValueError('Checked withdrawal accepted branch parent differs')
    cost = report['cost']
    if (not 0 <= cost['physicalCalls'] == cost['previewCalls']+cost['directStepCalls'] <= cost['maxPhysicalCalls'] == 240 or
            not 0 <= report['wallSeconds'] <= 45. or
            len(cost['trace']) != cost['physicalCalls'] or
            sum(row['kind'] == 'preview' for row in cost['trace']) != cost['previewCalls'] or
            sum(row['kind'] == 'step' for row in cost['trace']) != cost['directStepCalls'] or
            sum(row['phase'] == 'checked_recovery.independent_full_program_replay' for row in cost['trace']) != len(final['actions'])):
        raise ValueError('Checked withdrawal resource or replay accounting differs')
    return {'verified': True}


def exit_recovery_summary(report):
    hidden = {'sourceSnapshot', 'sourceProgramSeal', 'receipt', 'stoppedExecutions', 'cost', 'finalNativeHistory'}
    result = {k: copy.deepcopy(v) for k, v in report.items() if k not in hidden}
    result['cost'] = {key: report['cost'][key] for key in (
        'physicalCalls', 'maxPhysicalCalls', 'previewCalls', 'directStepCalls')}
    return result
