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
