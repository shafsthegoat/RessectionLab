"""Checked both-hand exits under one original joint B/P/W policy and meter.

Geometry proposes finite exits. Public native previews/steps remain authoritative;
this adapter neither enlarges legacy SkillLimits nor refunds preparation costs.
"""
from __future__ import annotations

import copy
import json
import time
from collections.abc import Mapping

from .surgical_engine import SurgicalEngine
from .surgical_tools import TOOLS
from .surgical_transition_meter import MeteredEngine, TransitionMeter
from .surgical_joint_cranial_skills import (
    JointCranialLimits, checked_joint_actions, joint_cranial_ledger, _INTERRUPTIONS)
from .surgical_staged_exit import (
    _pose, _distance, _duration, _path_chunks,
    _reverse_accepted_poses, _reverse_native_steps)

SCHEMA = 'resectionlab.checked-joint-cranial-exit.v1'
MAX_CANDIDATES = 6
MAX_ACTIONS_PER_CANDIDATE = 32


def _native(engine):
    return engine.unwrap() if isinstance(engine, MeteredEngine) else engine


def joint_exit_complete(engine):
    """Actual exterior flags and complete stock bodies, independent of P clear."""
    from .surgical_cranial_guards import tool_is_exterior
    from .surgical_task_domain import _posed
    native = _native(engine)
    if native.state['holds'] or native.state.get('detachedFragments'):
        return False
    return all(not slot['active'] and not slot['inserted'] and not slot['grip'] and
        (not _posed(native, hand) or tool_is_exterior(native, hand))
        for hand, slot in native.state['slots'].items())


def _stock_geometry(engine):
    native = _native(engine)
    contract = native._cranial_stages
    from .surgical_task_domain import _posed
    from .surgical_cranial_guards import tool_is_exterior
    for hand, slot in native.state['slots'].items():
        tool = TOOLS[slot['instrument']]
        params = native.parameters(slot)
        depths = (contract.scissors_depth_mm,) if slot['instrument'] == 'dural_scissors' else (2.,)
        if (slot['instrument'] == 'dural_scissors' and not slot['active'] and
                not slot['inserted'] and not slot['grip'] and
                (not _posed(native, hand) or tool_is_exterior(native, hand))):
            # A new generic stock scissors may remain at its default2mm depth
            # outside. The separate native entry guard still requires the
            # frozen source-span depth before any membrane incision.
            depths = (*depths, 2.)
        if (params['radiusMm'] != tool['radiusMm'] or tool['shaftRadiusMm'] != .5 or
                params['depthMm'] not in depths):
            return False
    return True


def _last_place(engine, hand):
    """Use only accepted native placement for the currently selected tool."""
    for event in reversed(engine.history):
        action = event['action']
        if action.get('slot') != hand:
            continue
        if action['operation'] == 'exchange':
            break
        if action['operation'] == 'place' and event['status'] == 'accepted':
            return _pose(action['pose'])
    return None


def _needs_withdrawal(engine, hand):
    from .surgical_task_domain import _posed
    from .surgical_cranial_guards import tool_is_exterior
    native = _native(engine)
    slot = native.state['slots'][hand]
    return _posed(native, hand) and (slot['inserted'] or slot['grip'] or
                                    not tool_is_exterior(native, hand))


def _commands(source, order, outside, *, duration_mode=0, paths=None):
    # Stop every active instrument before translating either hand. Any modeled
    # clock effect during deactivation still goes through the same native caps.
    actions = [{'slot': hand, 'operation': 'deactivate', 'duration': .2}
        for hand in order if source.state['slots'][hand]['active'] or _needs_withdrawal(source, hand)]
    for hand in order:
        if not _needs_withdrawal(source, hand):
            continue
        slot = source.state['slots'][hand]
        if paths and paths.get(hand):
            start = {key: copy.deepcopy(slot[key]) for key in ('position', 'quaternion')}
            actions.extend({'slot': hand, 'operation': 'move', 'path': chunk,
                'duration': duration} for chunk, duration in _path_chunks(start, paths[hand]))
            start_position = paths[hand][-1]['position']
        else:
            start_position = slot['position']
        distance = _distance(start_position, outside[hand]['position'])
        duration = (_duration(distance, 20.), _duration(distance, 2.), 30.)[duration_mode]
        actions.append({'slot': hand, 'operation': 'withdraw',
                        'pose': copy.deepcopy(outside[hand]), 'duration': duration})
    return actions


def _candidates(source, order, outside, check):
    seen = []
    for mode in range(3):
        actions = _commands(source, order, outside, duration_mode=mode)
        if actions not in seen:
            seen.append(actions)
            yield 'direct_'+str(mode), actions, None
    for name, generator in (('accepted_history_reverse', _reverse_accepted_poses),
                            ('native_history_reverse', _reverse_native_steps)):
        paths, unsupported = {}, []
        for hand in order:
            if not _needs_withdrawal(source, hand):
                continue
            if generator is _reverse_native_steps:
                poses, reason = generator(source, hand, check)
            else:
                check()
                poses, reason = generator(source, hand)
            if not poses:
                unsupported.append({'hand': hand, 'reason': reason})
            else:
                paths[hand] = poses
        if unsupported:
            yield name, None, unsupported
        else:
            actions = _commands(source, order, outside, paths=paths)
            if actions not in seen:
                seen.append(actions)
                yield name, actions, None
    alternate = _commands(source, tuple(reversed(order)), outside)
    if alternate not in seen:
        yield 'alternate_hand_order', alternate, None


def find_checked_joint_exit(engine, route=None, *, limits, meter=None, hand='left',
        outside_poses=None, check=lambda: None, max_candidates=6,
        phase='cranial.joint.exit'):
    """Return a checked exterior branch or the last actual charged stop prefix.

    Route/outside poses are proposals only. Every candidate starts from the
    original accepted input prefix; discarded probes remain charged globally.
    Failure returns its actual native prefix, never an invented complete exit.
    No active aspiration, tool resizing, release or extraction is synthesized.
    """
    if (not isinstance(engine, (SurgicalEngine, MeteredEngine)) or
            not isinstance(limits, JointCranialLimits) or hand not in ('left', 'right') or
            type(max_candidates) is not int or not 1 <= max_candidates <= MAX_CANDIDATES or
            not callable(check) or not isinstance(phase, str) or not phase):
        raise ValueError('Joint exit requires native source, typed limits and bounded candidates')
    meter = limits._meter if meter is None else meter
    if not isinstance(meter, TransitionMeter):
        raise TypeError('Joint exit requires its original transition meter')
    if route is not None and (not isinstance(route, Mapping) or
            route.get('sourceSceneId') != engine.scene.identity):
        raise ValueError('Joint exit route must match the native source scene')
    if outside_poses is not None and (not isinstance(outside_poses, Mapping) or
            set(outside_poses)-{'left', 'right'}):
        raise ValueError('Joint outside poses must use the two native hand slots')
    outside = {key: _pose(value) for key, value in (outside_poses or {}).items()}
    if route is not None and 'outsidePose' in route and hand not in outside:
        outside[hand] = _pose(route['outsidePose'])
    # Empty checked program authenticates registered policy, charged ancestry,
    # source/guard/partitions, original wall origin and cancellation first.
    source, validation = checked_joint_actions(engine, [], limits=limits, meter=meter,
        check=check, phase=phase+'.validate')
    first_calls, first_actions, first_state = meter.calls, len(source.actions), source.state_id
    attempts, last, latest = [], source, validation
    status, reason, mode = 'refused', None, None

    def finish():
        value = {**latest, 'schema': SCHEMA, 'status': status, 'reason': reason,
            'mode': mode, 'sourceStateId': first_state, 'finalStateId': last.state_id,
            'sourceNativeActions': first_actions, 'finalNativeActions': len(last.actions),
            'outsidePose': copy.deepcopy(outside.get(hand)), 'outsidePoses': copy.deepcopy(outside),
            'nativeActionsAppended': len(last.actions)-first_actions,
            'publicCallCost': meter.calls-first_calls, 'publicCalls': meter.calls-first_calls,
            'maxCandidates': max_candidates, 'maxActionsPerCandidate': MAX_ACTIONS_PER_CANDIDATE,
            'attempts': attempts, 'exitComplete': joint_exit_complete(last),
            'appliedExitActions': copy.deepcopy(last.actions[first_actions:]),
            'retentionRole': 'checked_exterior_prefix' if status == 'accepted' else 'stopped_native_prefix_not_admitted',
            'admittedPatientTask': False, 'clinicalValidation': 'none',
            'scope': 'Actual both-hand exterior exit; separate from membrane clearance or treatment completion',
            'unknowns': ['Prior conditional scalp exposure cost unknown', 'Clinical outcomes unassessed']}
        # Refresh after composing the receipt: serialization/assessment work is
        # part of the original meter lifetime, never a new exit deadline.
        json.dumps(value, allow_nan=False)
        ledger = joint_cranial_ledger(last, limits)
        value['ledger'] = ledger
        if value['status'] == 'accepted':
            try:
                check()
            except _INTERRUPTIONS as error:
                value.update(status='interrupted', reason=type(error).__name__+': '+str(error))
        ledger['wallSecondsSinceMeterCreation'] = time.monotonic()-limits.frozen_at
        if ledger['wallSecondsSinceMeterCreation'] >= limits.max_wall_seconds:
            ledger['withinLimits'] = False
            if 'WALL_BUDGET' not in ledger['refusalReasons']:
                ledger['refusalReasons'].append('WALL_BUDGET')
        if not ledger['withinLimits'] and value['status'] == 'accepted':
            value.update(status='interrupted', reason=ledger['refusalReasons'][0])
        value['admittedExit'] = value['status'] == 'accepted' and value['exitComplete']
        if not value['admittedExit']:
            value['retentionRole'] = 'stopped_native_prefix_not_admitted'
        return last, value

    if validation['status'] != 'accepted':
        status, reason = validation['status'], validation['reason']
        return finish()
    if not _stock_geometry(source):
        reason = 'JOINT_EXIT_STOCK_GEOMETRY_REQUIRED'
        return finish()
    if source.state['holds'] or source.state.get('detachedFragments'):
        reason = 'JOINT_EXIT_FRAGMENT_TRANSPORT_REQUIRED'
        return finish()
    if joint_exit_complete(source):
        status, mode = 'accepted', 'already_exterior'
        return finish()
    order = (hand, 'right' if hand == 'left' else 'left')
    for key in order:
        if _needs_withdrawal(source, key) and key not in outside:
            proposal = _last_place(source, key)
            if proposal is None:
                reason = 'JOINT_EXIT_NATIVE_OUTSIDE_POSE_UNAVAILABLE:'+key
                return finish()
            outside[key] = proposal
    count = 0
    try:
        for name, actions, unavailable in _candidates(source, order, outside, check):
            if unavailable:
                attempts.append({'mode': name, 'status': 'geometry_unavailable', 'details': unavailable})
                continue
            if count >= max_candidates:
                reason = 'JOINT_EXIT_CANDIDATE_BUDGET'
                break
            if len(actions) > MAX_ACTIONS_PER_CANDIDATE:
                attempts.append({'mode': name, 'status': 'geometry_unavailable', 'reason': 'JOINT_EXIT_ACTION_INVENTORY_BOUND'})
                continue
            count += 1
            candidate, receipt = checked_joint_actions(source, actions, limits=limits,
                meter=meter, check=check, phase=phase+'.candidate.'+str(count-1))
            last, latest = candidate, receipt
            attempts.append({'mode': name, 'probeActions': copy.deepcopy(actions), 'checked': receipt})
            if receipt['status'] == 'accepted' and joint_exit_complete(candidate):
                status, reason, mode = 'accepted', None, name
                return finish()
            if receipt['status'] == 'interrupted':
                status, reason = 'interrupted', receipt['reason']
                return finish()
            reason = receipt['reason'] or 'JOINT_EXIT_BOTH_HANDS_NOT_EXTERIOR'
    except _INTERRUPTIONS as error:
        status, reason = 'interrupted', type(error).__name__+': '+str(error)
    return finish()
