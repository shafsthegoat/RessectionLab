"""Bounded adaptive bone opening under the existing canonical joint ledger.

Source geometry only proposes contacts. Every work and posed staging motion
uses checked_joint_actions with the original limits/meter. This adapter creates
no source A, tool attitude, exit face, dural permission or patient qualification.
"""
from __future__ import annotations

import copy
import math
import time

from scipy.spatial.transform import Rotation

from .core import semantic_digest
from .surgical_engine import SurgicalEngine
from .surgical_cranial_guards import tool_is_exterior
from .surgical_joint_cranial_exit import joint_exit_complete
from .surgical_joint_cranial_skills import (
    JointCranialLimits, _INTERRUPTIONS, checked_joint_actions, joint_cranial_ledger)
from .surgical_milling import milling_readiness, validate_source_scope
from .surgical_milling_contact_proposals import (
    _approach, _exterior, _segments, _within_workspace, _approach_declaration,
    _local_path_clearance, _same_pose, _staging_plan, _CELL_SCHEDULES, bind_milling_approach,
    _CONTACT_POSE_FAMILIES, _cavity_body_check,
    generate_milling_contact_proposals, generate_milling_debris_proposals)
from .surgical_craniotomy import _original_targets
from .surgical_motion import swept_envelopes
from .surgical_tools import TOOLS
from .surgical_task_domain import _posed
from .surgical_transition_meter import MeteredEngine, TransitionBudgetExceeded, TransitionMeter

SCHEMA = 'resectionlab.checked-joint-milling-opening.v1'
_POINT_SCHEMES = ('center', 'center_and_quarter_cell_corners')


def _native(engine):
    return engine.unwrap() if isinstance(engine, MeteredEngine) else engine


def _positive(value):
    return type(value) in (int, float) and math.isfinite(value) and value > 0


def _initial_staging(engine, quaternion, slot):
    """No unverified rotation/teleport of an already posed physical instrument."""
    native = _native(engine)
    if native.state['holds'] or native.state.get('detachedFragments'):
        raise ValueError('JOINT_MILLING_INITIAL_STAGING_UNQUALIFIED')
    for hand, current in native.state['slots'].items():
        if current['active'] or current['inserted'] or current['grip']:
            raise ValueError('JOINT_MILLING_INITIAL_STAGING_UNQUALIFIED')
        if not _posed(native, hand):
            continue
        if not tool_is_exterior(native, hand):
            raise ValueError('JOINT_MILLING_INITIAL_STAGING_NOT_EXTERIOR')
        if hand == slot:
            angle = (Rotation.from_quat(quaternion)*Rotation.from_quat(current['quaternion']).inv()).magnitude()
            if angle > 1e-9:
                raise ValueError('JOINT_MILLING_POSED_ATTITUDE_STAGING_UNQUALIFIED')
            pose = {key: current[key] for key in ('position', 'quaternion')}
            for name in ('cranial_drill', 'frazier_suction'):
                segments = _segments(name, pose)
                if not _exterior(native.scene, native._cranial_access, segments) or not _within_workspace(
                        native.scene, native._cranial_access, segments):
                    raise ValueError('JOINT_MILLING_INITIAL_STOCK_STAGING_UNQUALIFIED')


def _checked_staging_program(engine, actions, slot, staging_seconds):
    """Initial unposed setup once; all subsequent outside translations are sweeps."""
    posed = _posed(_native(engine), slot)
    result, modes = [], []
    for original in actions:
        action = copy.deepcopy(original)
        if action.get('slot') != slot:
            raise ValueError('JOINT_MILLING_SINGLE_DECLARED_HAND_REQUIRED')
        if action['operation'] == 'place':
            if posed:
                action['operation'] = 'withdraw'
                action['duration'] = staging_seconds
                modes.append('checked_fixed_attitude_exterior_translation')
            else:
                modes.append('initial_unposed_exterior_setup')
            posed = True
        result.append(action)
    return result, modes


def _initial_multileg_staging(engine, plan, slot):
    native = _native(engine)
    _initial_staging(engine, plan['globalPark']['quaternion'], slot)
    if any(any(offset) for offset in native.state.get('deformation', {}).values()):
        raise ValueError('JOINT_MILLING_UNMOVED_SOURCE_CAVITY_REQUIRED')
    if _posed(native, slot) and not _same_pose(native.state['slots'][slot], plan['globalPark']):
        raise ValueError('JOINT_MILLING_DECLARED_GLOBAL_START_POSE_DIFFERS')


def _multileg_entry_actions(engine, plan, name, slot, duration):
    """Only a genuinely unposed hand receives endpoint-only global placement."""
    native = _native(engine)
    actions = [{'slot': slot, 'operation': 'configure', 'duration': .1,
                'mechanicsRevision': 2, 'clockRevision': 1}]
    if native.state['slots'][slot]['instrument'] != name:
        actions.append({'slot': slot, 'operation': 'exchange', 'duration': .1, 'instrument': name})
    if not _posed(native, slot):
        actions.append({'slot': slot, 'operation': 'place', 'duration': duration,
                        'pose': copy.deepcopy(plan['globalPark'])})
    actions.extend({'slot': slot, 'operation': 'move', 'duration': duration, 'pose': copy.deepcopy(pose)}
                   for pose in plan['inbound'])
    return actions


def _multileg_exit_actions(engine, plan, slot, duration):
    """Local moves precede one final whole-body global withdrawal."""
    pose = _native(engine).state['slots'][slot]
    vertices = plan['outbound']
    actions = []
    for end in vertices[:-1]:
        if not _same_pose(pose, end):
            actions.append({'slot': slot, 'operation': 'move', 'duration': duration,
                            'pose': copy.deepcopy(end)})
        pose = end
    actions.append({'slot': slot, 'operation': 'withdraw', 'duration': duration,
                    'pose': copy.deepcopy(vertices[-1])})
    return actions


def _opening_facts(engine):
    native = _native(engine)
    opening = milling_readiness(engine)
    from .surgical_cranial_stages import stage_readiness
    stage = stage_readiness(native)
    complete = bool(opening and opening['historyVerified'] and opening['debrisConserved'] and
                    opening['openingComplete'] and joint_exit_complete(engine) and
                    stage and stage['bonePrefixVerified'])
    return opening, stage, complete


def _content_hash(receipt):
    """Deterministic physical/program evidence, with wall and meter IDs separate."""
    return semantic_digest({key: receipt[key] for key in (
        'schema', 'sourceSceneId', 'sourceStateId', 'finalStateId', 'specHash',
        'status', 'reason', 'openingComplete', 'appliedActions', 'batchDecisions')})


def checked_joint_milling_opening(engine, *, approach, limits, meter=None,
        drill_seconds, suction_seconds, staging_seconds=.1, slot='left',
        max_batches=8, max_contacts=32, max_examined_cells=2048,
        max_contact_probes=2048, point_scheme='center', check=None,
        phase='cranial.joint.milling-opening', staging_plan=None, cell_schedule='surface_first',
        contact_pose_family='leading_touch'):
    """Return a checked complete bone-opening prefix or an exact charged stop.

    The adapter rebinds all-A ordering once inside the original shared wall
    scope. Each accepted batch replans from actual cavity and tagged debris.
    The default requires a fixed-attitude global exterior start. An optional
    frozen staging_plan uses checked native multileg moves, same-tool bone
    batches, a full global withdrawal before exchange, and actual-debris-only
    collection. Local retreats never clear inserted or count as global exit.
    Optional cavity_carried_rays requires the staged axial/quarter mode and
    charges staging_seconds on passive moves; both the short active command
    and its separate stationary dwell charge drill_seconds.
    """
    if (not isinstance(limits, JointCranialLimits) or type(approach) is not dict or
            type(phase) is not str or not phase or slot not in ('left', 'right') or
            type(max_batches) is not int or not 1 <= max_batches <= 8 or
            type(max_contacts) is not int or not 1 <= max_contacts <= 32 or
            type(max_examined_cells) is not int or not 1 <= max_examined_cells <= 2048 or
            type(max_contact_probes) is not int or not 1 <= max_contact_probes <= 2048 or
            type(point_scheme) is not str or point_scheme not in _POINT_SCHEMES or
            type(cell_schedule) is not str or cell_schedule not in _CELL_SCHEDULES or
            type(contact_pose_family) is not str or contact_pose_family not in _CONTACT_POSE_FAMILIES or
            contact_pose_family != 'leading_touch' and (staging_plan is None or
                cell_schedule != 'axial_diverse_current_a' or point_scheme != _POINT_SCHEMES[1]) or
            not all(_positive(value) for value in (drill_seconds, suction_seconds, staging_seconds)) or
            check is not None and not callable(check)):
        raise ValueError('JOINT_MILLING_EXPLICIT_BOUNDED_SPEC_REQUIRED')
    meter = limits._meter if meter is None else meter
    if not isinstance(engine, (SurgicalEngine, MeteredEngine)) or not isinstance(meter, TransitionMeter):
        raise TypeError('JOINT_MILLING_NATIVE_ENGINE_AND_SHARED_METER_REQUIRED')
    first_state, first_actions = engine.state_id, len(engine.actions)
    first_calls, started = meter.calls, time.monotonic()
    current, validation = checked_joint_actions(engine, [], limits=limits, meter=meter,
        check=check, phase=phase+'.validate')
    # This empty checked call authenticates limits/source/stage/guard/ancestry.
    contract = validate_source_scope(current)
    _approach(current, contract, approach)
    if approach.get('pointScheme', 'center') != point_scheme:
        raise ValueError('JOINT_MILLING_POINT_SCHEME_BINDING_DIFFERS')
    if approach.get('cellSchedule', 'surface_first') != cell_schedule:
        raise ValueError('JOINT_MILLING_CELL_SCHEDULE_BINDING_DIFFERS')
    if approach.get('contactPoseFamily', 'leading_touch') != contact_pose_family:
        raise ValueError('JOINT_MILLING_CONTACT_POSE_FAMILY_BINDING_DIFFERS')
    if set(current.state['removed'])-set(limits._bone):
        raise ValueError('JOINT_MILLING_BONE_ONLY_SOURCE_PREFIX_REQUIRED')
    initial_stage = validation['stageReadiness']
    if initial_stage and (initial_stage['cutBoundaryEdges'] or initial_stage['patchHeld'] or
            initial_stage['patchDetached'] or initial_stage['patchExtracted']):
        raise ValueError('JOINT_MILLING_LATER_STAGE_SOURCE_NOT_SUPPORTED')
    if staging_plan is None:
        _initial_staging(current, approach['quaternion'], slot)
    else:
        staging_plan = _staging_plan(current, contract, approach, staging_plan, require_source_state=True)
        _initial_multileg_staging(current, staging_plan, slot)
    declaration = _approach_declaration(approach)
    spec = {'approach': declaration, 'pointScheme': point_scheme, 'slot': slot,
        'drillSeconds': drill_seconds, 'suctionSeconds': suction_seconds,
        'stagingSeconds': staging_seconds, 'maxBatches': max_batches,
        'maxContacts': max_contacts, 'maxExaminedCells': max_examined_cells,
        'maxContactProbes': max_contact_probes}
    if staging_plan is not None:
        spec['stagingPlan'] = copy.deepcopy(staging_plan)
    if cell_schedule != 'surface_first':
        spec['cellSchedule'] = cell_schedule
    if contact_pose_family != 'leading_touch':
        spec['contactPoseFamily'] = contact_pose_family
    batches, decisions, binding = [], [], None
    staging_transitions = []
    status, reason = validation['status'], validation['reason']

    def guard():
        if check is not None:
            check()
        if time.monotonic()-limits.frozen_at >= limits.max_wall_seconds:
            raise TimeoutError('Joint milling opening wall bound since shared meter creation')
        if meter.calls >= min(meter.max_calls, limits.max_public_calls):
            raise TransitionBudgetExceeded('Joint milling opening shared public-call bound')

    def finish():
        nonlocal status, reason
        opening, stage, complete = _opening_facts(current)
        ledger = joint_cranial_ledger(current, limits)
        if status == 'accepted' and (not complete or not ledger['withinLimits']):
            status = 'refused'
            reason = ledger['refusalReasons'][0] if ledger['refusalReasons'] else 'COMPLETE_NATIVE_BONE_OPENING_REQUIRED'
        value = {'schema': SCHEMA, 'status': status, 'reason': reason,
            'sourceSceneId': current.scene.identity, 'sourceStateId': first_state,
            'finalStateId': current.state_id, 'spec': spec, 'specHash': semantic_digest(spec),
            'contractHash': contract.content_hash, 'stageContractHash': limits.stage_contract_hash,
            'guardSeal': limits.guard_seal, 'limitsHash': limits.content_hash,
            'approachBinding': copy.deepcopy(binding), 'batches': batches,
            'batchDecisions': decisions, 'batchCount': len(batches),
            'appliedActions': copy.deepcopy(current.actions[first_actions:]),
            'nativeActionsAppended': len(current.actions)-first_actions,
            'publicCalls': meter.calls-first_calls, 'sharedTransitionMeter': meter.report(),
            'ledger': ledger, 'nativeOpening': opening, 'stageReadiness': stage,
            'openingComplete': complete, 'remainingOriginalACells': sorted(
                set(contract.aperture_cells)-set(current.state['removed'])),
            'currentTaggedDebris': [{'cell': int(cell), 'volumeMm3': amount}
                for cell, amount in sorted(current.state['debris'].items()) if amount > 0],
            'wholeInstrumentExitComplete': joint_exit_complete(current),
            'admittedBoneOpening': status == 'accepted' and complete and ledger['withinLimits'],
            'admittedPatientTask': False, 'caseQualificationRequired': True,
            'retentionRole': 'checked_native_bone_opening' if status == 'accepted' and complete else
                'stopped_native_prefix_not_admitted',
            'wallSeconds': time.monotonic()-started,
            'wallTimeScope': 'original shared joint meter creation, including geometry and prior prefixes',
            'sourceAChanged': False, 'clinicalValidation': 'none'}
        value['contentHash'] = _content_hash(value)
        if staging_plan is not None:
            value['stagingTransitions'] = copy.deepcopy(staging_transitions)
        # Verification, meter report copies and hashing are part of the original
        # wall/cancellation scope. Completion at the exact call cap is valid;
        # the next native call would be refused, not this completed readout.
        final_error = None
        try:
            if check is not None:
                check()
        except _INTERRUPTIONS as error:
            final_error = type(error).__name__+': '+str(error)
        final_now = time.monotonic()
        final_wall = final_now-limits.frozen_at
        over_wall = final_wall >= limits.max_wall_seconds
        over_calls = meter.calls > min(limits.max_public_calls, meter.max_calls)
        ledger.update(spentPublicCalls=meter.calls,
                      wallSecondsSinceMeterCreation=final_wall)
        for failed, code in ((over_wall, 'WALL_BUDGET'), (over_calls, 'PUBLIC_CALL_BUDGET')):
            if failed:
                ledger['withinLimits'] = False
                if code not in ledger['refusalReasons']:
                    ledger['refusalReasons'].append(code)
        value.update(publicCalls=meter.calls-first_calls, wallSeconds=final_now-started,
            finalReadout={'cancelledOrInterrupted': final_error is not None,
                'wallOverrun': over_wall, 'publicCallOverrun': over_calls})
        if final_error or over_wall or over_calls:
            status = 'interrupted'
            reason = final_error or ('WALL_BUDGET_AT_FINAL_READOUT' if over_wall else
                                     'PUBLIC_CALL_BUDGET_AT_FINAL_READOUT')
            value.update(status=status, reason=reason, admittedBoneOpening=False,
                         retentionRole='stopped_native_prefix_not_admitted')
            value['contentHash'] = _content_hash(value)
        return current, value

    def staged_transition(name, *, enter):
        nonlocal current, status, reason
        guard()
        pose = {key: copy.deepcopy(current.state['slots'][slot][key])
                for key in ('position', 'quaternion')}
        if enter:
            if _posed(_native(current), slot) and not _same_pose(pose, staging_plan['globalPark']):
                raise ValueError('JOINT_MILLING_DECLARED_GLOBAL_START_POSE_DIFFERS')
            poses = [staging_plan['globalPark'], *staging_plan['inbound']]
            actions = _multileg_entry_actions(current, staging_plan, name, slot, staging_seconds)
        else:
            poses = [pose, *staging_plan['outbound']]
            actions = _multileg_exit_actions(current, staging_plan, slot, staging_seconds)
        path_kwargs = ({'native_duration': staging_seconds} if contact_pose_family == 'cavity_carried_rays' else {})
        refusal, intervals = _local_path_clearance(current, contract, name, poses, slot, check=guard, **path_kwargs)
        record = {'kind': 'inbound' if enter else 'outbound', 'instrument': name,
            'sourceStateId': current.state_id, 'geometryIntervals': intervals, 'geometryRefusal': refusal,
            'actions': copy.deepcopy(actions)}
        staging_transitions.append(record)
        if refusal:
            status, reason = 'refused', refusal
            return False
        current, part = checked_joint_actions(current, actions, limits=limits, meter=meter,
            check=check, phase=f'{phase}.staging{len(staging_transitions)}.{record["kind"]}')
        record['native'] = part
        if part['status'] != 'accepted':
            status, reason = part['status'], part['reason']
            return False
        return True

    if status != 'accepted':
        return finish()
    try:
        opening, stage, complete = _opening_facts(current)
        if complete:
            reason = 'ALREADY_COMPLETE_NATIVE_BONE_OPENING'
            return finish()
        guard()
        # Caller order/cost fields are not trusted. The actual source is ordered
        # once here, after the original meter/wall origin is authenticated.
        binding_kwargs = {} if cell_schedule == 'surface_first' else {
            'cell_schedule': cell_schedule, 'check': guard}
        if contact_pose_family != 'leading_touch':
            binding_kwargs['contact_pose_family'] = contact_pose_family
        binding = bind_milling_approach(current, quaternion=declaration['quaternion'],
            outside_plane_point_ras_mm=declaration['outsidePlanePointRasMm'],
            maximum_approach_mm=declaration['maximumApproachMm'], point_scheme=point_scheme,
            **binding_kwargs)
        guard()
        local_tool = None
        for batch in range(max_batches):
            before_removed = set(current.state['removed'])
            before_debris = sum(current.state['debris'].values())
            remaining = set(contract.aperture_cells)-before_removed
            # The optional controller finishes bone on one drill before any
            # exchange. Debris-first legacy behavior is deliberately retained.
            use_debris = before_debris > 0 and (staging_plan is None or not remaining)
            kind = 'current_tagged_debris' if use_debris else 'current_cavity_drill'
            if staging_plan is not None:
                name = 'frazier_suction' if use_debris else 'cranial_drill'
                if local_tool != name:
                    if local_tool is not None and not staged_transition(local_tool, enter=False):
                        return finish()
                    if not staged_transition(name, enter=True):
                        return finish()
                    local_tool = name
            kwargs = dict(approach=binding, max_examined_cells=max_examined_cells,
                max_contacts=max_contacts, max_contact_probes=max_contact_probes,
                suction_seconds=suction_seconds, slot=slot)
            if staging_plan is not None:
                kwargs['staging_plan'] = staging_plan
            if cell_schedule != 'surface_first':
                kwargs['check'] = guard
            guard()
            if use_debris:
                proposal = generate_milling_debris_proposals(current, **kwargs)
            else:
                if contact_pose_family == 'cavity_carried_rays':
                    kwargs['staging_seconds'] = staging_seconds
                proposal = generate_milling_contact_proposals(current,
                    drill_seconds=drill_seconds, **kwargs)
            guard()
            record = {'index': batch, 'kind': kind, 'sourceStateId': current.state_id,
                'proposalHash': proposal['contentHash'], 'proposal': proposal}
            batches.append(record)
            decision = {key: record[key] for key in ('index', 'kind', 'sourceStateId', 'proposalHash')}
            decisions.append(decision)
            if not proposal['actions']:
                status, reason = 'refused', 'MILLING_CONTACT_CONTROLLER_DEFICIT'
                decision.update(status=status, reason=reason, finalStateId=current.state_id)
                return finish()
            if staging_plan is None:
                actions, modes = _checked_staging_program(current, proposal['actions'], slot, staging_seconds)
            else:
                actions = copy.deepcopy(proposal['actions'])
                modes = ['checked_same_tool_local_moves_not_completed_withdrawal']
                if proposal.get('requiredInstrument') != local_tool:
                    raise ValueError('JOINT_MILLING_LOCAL_PROPOSAL_INSTRUMENT_DIFFERS')
            record['stagingModes'] = modes
            if staging_plan is None:
                current, part = checked_joint_actions(current, actions, limits=limits, meter=meter,
                    check=check, phase=f'{phase}.batch{batch+1}.{kind}')
                record['native'] = part
            else:
                # Query local transitions at their ACTUAL accepted prefix.
                # A retreat may initially remain within the entry prism, but
                # no occupied cell is assumed removed and its endpoint must
                # place the entire body outside the entry halfspace.
                record['nativeParts'] = []
                previous_operation = None
                for index, action in enumerate(actions):
                    guard()
                    if action['operation'] == 'move':
                        pose = {key: current.state['slots'][slot][key]
                                for key in ('position', 'quaternion')}
                        move_cost = (dict(bodyCapsuleQueries=0, bodyCoveredCellVisits=0, protectionCapsuleQueries=0)
                                     if contact_pose_family == 'cavity_carried_rays' else None)
                        refusal, intervals = _local_path_clearance(current, contract, local_tool,
                            [pose, action['pose']], slot, check=guard,
                            allow_entry_retreat=previous_operation == 'deactivate',
                            allow_cleared_cavity_lead=contact_pose_family == 'cavity_carried_rays' and
                                not use_debris and index % 6 == 1, geometry_cost=move_cost,
                            native_duration=action['duration'] if contact_pose_family == 'cavity_carried_rays' else None)
                        record.setdefault('localMoveGeometry', []).append(
                            {'index': index, 'intervals': intervals, 'refusal': refusal,
                             'sourceStateId': current.state_id})
                        if move_cost is not None:
                            record['localMoveGeometry'][-1]['cost'] = move_cost
                        if refusal:
                            raise ValueError(refusal)
                    if contact_pose_family == 'cavity_carried_rays' and not use_debris and action['operation'] == 'apply':
                        pose = {key: current.state['slots'][slot][key] for key in ('position', 'quaternion')}
                        if index % 6 == 2 and (not _same_pose(pose, actions[index-1]['pose']) or
                                math.dist(pose['position'], action['pose']['position']) > .20+1e-9 or
                                (Rotation.from_quat(pose['quaternion'])*Rotation.from_quat(
                                    action['pose']['quaternion']).inv()).magnitude() > 1e-9):
                            raise ValueError('JOINT_MILLING_ACTUAL_BOUNDED_LEAD_REQUIRED')
                        if index % 6 == 3 and not _same_pose(pose, action['pose']):
                            raise ValueError('JOINT_MILLING_ACTUAL_DRILL_ENDPOINT_REQUIRED')
                        segments = swept_envelopes(TOOLS['cranial_drill'], {'radiusMm': 2., 'depthMm': 2.},
                                                    pose, action['pose'])
                        geometry = dict(wholeBodyChecks=0, bodyCapsuleQueries=0, bodyCoveredCellVisits=0,
                                        protectionCapsuleQueries=0)
                        native = _native(current)
                        refusal, _ = _cavity_body_check(native, contract, segments, set(current.state['removed']),
                            set(contract.aperture_cells)-set(current.state['removed']),
                            set(native._protected_ids) | _original_targets(native.scene), slot, geometry, guard)
                        record.setdefault('activeCommandGeometry', []).append(
                            {'index': index, 'refusal': refusal, 'sourceStateId': current.state_id, 'cost': geometry})
                        if refusal:
                            raise ValueError(refusal)
                    current, part = checked_joint_actions(current, [action], limits=limits, meter=meter,
                        check=check, phase=f'{phase}.batch{batch+1}.{kind}.action{index+1}')
                    record['nativeParts'].append(part)
                    if part['status'] != 'accepted':
                        break
                    previous_operation = action['operation']
            decision.update(status=part['status'], reason=part['reason'], finalStateId=current.state_id)
            if part['status'] != 'accepted':
                status, reason = part['status'], part['reason']
                return finish()
            opening, stage, complete = _opening_facts(current)
            if staging_plan is not None and not (
                    set(contract.aperture_cells)-set(current.state['removed'])) and not any(
                    amount > 0 for amount in current.state['debris'].values()):
                if not staged_transition(local_tool, enter=False):
                    return finish()
                local_tool = None
                opening, stage, complete = _opening_facts(current)
            if complete:
                status, reason = 'accepted', None
                return finish()
            after_debris = sum(current.state['debris'].values())
            if not (set(current.state['removed']) > before_removed or after_debris < before_debris-1e-9):
                status, reason = 'refused', 'MILLING_NO_ACTUAL_BONE_OR_DEBRIS_PROGRESS'
                decision.update(status=status, reason=reason)
                return finish()
            if staging_plan is None:
                _initial_staging(current, declaration['quaternion'], slot)
            guard()
        status, reason = 'refused', 'MILLING_BATCH_BUDGET'
    except _INTERRUPTIONS as error:
        status, reason = 'interrupted', type(error).__name__+': '+str(error)
    except ValueError as error:
        status, reason = 'refused', str(error)
    return finish()
