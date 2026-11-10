"""Checked joint cranial programs with one prospective B/P/W source ledger.

This separate policy does not enlarge SkillLimits or authorize anatomy/stages.
The optional native stage contract remains authoritative. Preparation and bone
costs remain in the ordinary whole-sequence metrics, including stopped prefixes.
"""
from __future__ import annotations

import asyncio
import copy
from dataclasses import dataclass, field, fields
import math
import time
import weakref

from .core import semantic_digest
from .surgical_engine import SurgicalEngine
from .surgical_transition_meter import MeteredEngine, TransitionBudgetExceeded, TransitionMeter
from .surgical_craniotomy_skills import _endpoint_reached, _milling_capacity_partial
from .surgical_staged_skills import (
    _contact_clearance_action, _inserted_geometry_change, _remaining_contact_action)


SCHEMA = 'resectionlab.checked-joint-cranial-program.v1'
_REGISTERED = weakref.WeakKeyDictionary()
_INTERRUPTIONS = (TransitionBudgetExceeded, TimeoutError, InterruptedError,
                  KeyboardInterrupt, asyncio.CancelledError)


@dataclass(frozen=True)
class JointCranialLimits:
    max_bone_mm3: float
    max_preparation_nonbone_mm3: float
    max_work_nonbone_mm3: float
    max_native_actions: int
    max_public_calls: int
    max_wall_seconds: float
    max_bone_contact_replans: int
    max_tissue_contact_replans: int
    source_scene_id: str
    source_state_id: str
    stage_contract_hash: str
    guard_seal: str
    partitions_seal: str
    frozen_at: float
    meter_identity: int
    content_hash: str = ''
    _meter: TransitionMeter = field(default=None, repr=False, compare=False)
    _bone: frozenset = field(default=frozenset(), repr=False, compare=False)
    _preparation: frozenset = field(default=frozenset(), repr=False, compare=False)
    _work: frozenset = field(default=frozenset(), repr=False, compare=False)
    _target_labels: tuple = field(default=(), repr=False, compare=False)
    _contract: object = field(default=None, repr=False, compare=False)
    _source_engine: object = field(default=None, repr=False, compare=False)

    def __post_init__(self):
        for value in (self.max_bone_mm3, self.max_preparation_nonbone_mm3,
                      self.max_work_nonbone_mm3, self.max_wall_seconds):
            if type(value) not in (int, float) or not math.isfinite(value) or value < 0:
                raise ValueError('Joint cranial bounds must be explicit finite nonnegative numbers')
        if self.max_wall_seconds <= 0:
            raise ValueError('Joint cranial wall bound must be positive')
        if any(type(x) is not int or x < 1 for x in
               (self.max_native_actions, self.max_public_calls)):
            raise ValueError('Joint cranial action/call bounds must be positive integers')
        if any(type(x) is not int or not 0 <= x <= 8 for x in
               (self.max_bone_contact_replans, self.max_tissue_contact_replans)):
            raise ValueError('Joint contact continuation bounds must be in 0..8')

    def declaration(self):
        return {f.name: getattr(self, f.name) for f in fields(self)
                if not f.name.startswith('_') and f.name != 'content_hash'}


def _stage(engine):
    from .surgical_cranial_stages import validate_stage_contract, source_partition_sets
    native = engine.unwrap() if isinstance(engine, MeteredEngine) else engine
    contract = getattr(native, '_cranial_stages', None)
    if contract is None:
        raise ValueError('JOINT_CRANIAL_NATIVE_STAGE_CONTRACT_REQUIRED')
    contract = validate_stage_contract(native, contract)
    return contract, source_partition_sets(contract)


def freeze_joint_cranial_limits(engine, *, meter, max_bone_mm3,
        max_preparation_nonbone_mm3, max_work_nonbone_mm3,
        max_native_actions, max_public_calls, max_wall_seconds,
        max_bone_contact_replans=6, max_tissue_contact_replans=8):
    """Freeze before the first native action; later invocations cannot refund it."""
    if not isinstance(engine, (SurgicalEngine, MeteredEngine)) or not isinstance(meter, TransitionMeter):
        raise TypeError('Joint cranial limits require a native engine and shared meter')
    if isinstance(engine, MeteredEngine) and engine.meter is not meter:
        raise ValueError('JOINT_CRANIAL_SHARED_METER_REQUIRED')
    canonical = engine.fresh()
    if engine.actions or engine.history or engine.state_id != canonical.state_id:
        raise ValueError('JOINT_CRANIAL_FREEZE_BEFORE_ACTIONS_REQUIRED')
    origin = getattr(meter, '_created_monotonic', None)
    if type(origin) not in (int, float) or not math.isfinite(origin) or not 0 <= origin <= time.monotonic():
        raise ValueError('JOINT_CRANIAL_METER_ORIGIN_REQUIRED')
    contract, parts = _stage(engine)
    bone, preparation, work = (parts[k] for k in ('bone', 'preparation', 'work'))
    if bone & preparation or bone & work or preparation & work:
        raise ValueError('JOINT_CRANIAL_DISJOINT_SOURCE_PARTITIONS_REQUIRED')
    values = dict(max_bone_mm3=max_bone_mm3,
        max_preparation_nonbone_mm3=max_preparation_nonbone_mm3,
        max_work_nonbone_mm3=max_work_nonbone_mm3,
        max_native_actions=max_native_actions, max_public_calls=max_public_calls,
        max_wall_seconds=max_wall_seconds,
        max_bone_contact_replans=max_bone_contact_replans,
        max_tissue_contact_replans=max_tissue_contact_replans,
        source_scene_id=engine.scene.identity, source_state_id=engine.state_id,
        stage_contract_hash=contract.content_hash, guard_seal=semantic_digest(engine.guard_report()),
        partitions_seal=semantic_digest({k: sorted(parts[k]) for k in ('bone', 'preparation', 'work', 'original_target')}),
        frozen_at=origin, meter_identity=id(meter),
        _meter=meter, _bone=bone, _preparation=preparation, _work=work,
        _target_labels=tuple(r['id'] for r in engine.scene.regions if r.get('target')),
        _contract=contract, _source_engine=canonical)
    result = JointCranialLimits(**values)
    result = JointCranialLimits(**values, content_hash=semantic_digest(result.declaration()))
    registered = _REGISTERED.get(meter)
    if registered is not None:
        previous = registered[0]()
        if previous is None or registered[1] != result.content_hash:
            raise ValueError('JOINT_CRANIAL_BUDGET_ALREADY_FROZEN')
        return previous
    _REGISTERED[meter] = (weakref.ref(result), result.content_hash)
    return result


def _prefix(engine, limits, meter):
    if not engine.actions:
        if engine.history or engine.state_id != limits.source_state_id:
            raise ValueError('JOINT_CRANIAL_CANONICAL_SOURCE_REQUIRED')
        return
    if not isinstance(engine, MeteredEngine) or engine.meter is not meter:
        raise ValueError('JOINT_CRANIAL_CHARGED_PREFIX_REQUIRED')
    report = meter.report()
    lineage, branch = set(), engine.branch_id
    while branch is not None:
        if branch in lineage or str(branch) not in report['branches']:
            raise ValueError('JOINT_CRANIAL_PREFIX_LINEAGE_INVALID')
        lineage.add(branch)
        branch = report['branches'][str(branch)]['parentBranchId']
    if not any(report['branches'][str(b)]['initialActions'] == 0 and
               report['branches'][str(b)]['initialStateId'] == limits.source_state_id for b in lineage):
        raise ValueError('JOINT_CRANIAL_CHARGED_SOURCE_ANCESTRY_REQUIRED')
    if len(engine.actions) != len(engine.history):
        raise ValueError('JOINT_CRANIAL_PREFIX_HISTORY_INVALID')
    previous = limits.source_state_id
    charged = {}
    for row in report['trace']:
        if row['kind'] == 'step' and row['branchId'] in lineage:
            charged.setdefault((row['actionsBefore'], row['stateBefore'], row.get('stateAfter')), []).append(row)
    for index, (action, event) in enumerate(zip(engine.actions, engine.history)):
        if (event.get('action') != action or event.get('stateBefore') != previous or
                event.get('status') not in ('accepted', 'partial') or not any(
                    r.get('actionsAfter') == index+1 and r.get('action') == action and
                    r.get('status') == event['status'] and r.get('reason') == event.get('reason')
                    for r in charged.get((index, previous, event.get('stateAfter')), ()))):
            raise ValueError('JOINT_CRANIAL_CHARGED_PREFIX_REQUIRED')
        previous = event['stateAfter']
    if previous != engine.state_id:
        raise ValueError('JOINT_CRANIAL_UNLOGGED_NATIVE_STATE')


def _require(engine, limits, meter):
    if not isinstance(engine, (SurgicalEngine, MeteredEngine)):
        raise TypeError('Joint cranial execution requires a native engine')
    if not isinstance(limits, JointCranialLimits) or meter is not limits._meter or id(meter) != limits.meter_identity:
        raise ValueError('JOINT_CRANIAL_SHARED_METER_REQUIRED')
    registered = _REGISTERED.get(meter)
    if registered is None or registered[0]() is not limits:
        raise ValueError('JOINT_CRANIAL_REGISTERED_LIMIT_IDENTITY_REQUIRED')
    if limits.content_hash != registered[1] or limits.content_hash != semantic_digest(limits.declaration()):
        raise ValueError('JOINT_CRANIAL_LIMITS_CHANGED')
    if getattr(meter, '_created_monotonic', None) != limits.frozen_at:
        raise ValueError('JOINT_CRANIAL_METER_ORIGIN_CHANGED')
    baseline = limits._source_engine
    if (not isinstance(baseline, SurgicalEngine) or baseline.scene is not engine.scene or
            baseline.actions or baseline.history or baseline.state_id != limits.source_state_id or
            baseline._cranial_stages is not limits._contract):
        raise ValueError('JOINT_CRANIAL_CANONICAL_BASELINE_CHANGED')
    if isinstance(engine, MeteredEngine) and engine.meter is not meter:
        raise ValueError('JOINT_CRANIAL_SHARED_METER_REQUIRED')
    contract, parts = _stage(engine)
    if (engine.scene.identity != limits.source_scene_id or contract is not limits._contract or
            contract.content_hash != limits.stage_contract_hash or
            any(parts[k] is not expected for k, expected in
                (('bone', limits._bone), ('preparation', limits._preparation), ('work', limits._work))) or
            limits._target_labels != tuple(r['id'] for r in engine.scene.regions if r.get('target')) or
            semantic_digest(engine.guard_report()) != limits.guard_seal):
        raise ValueError('JOINT_CRANIAL_SOURCE_OR_CONTRACT_CHANGED')
    _prefix(engine, limits, meter)


def _burden(engine, limits, removed):
    if (not isinstance(removed, (list, tuple)) or
            any(type(i) is not int or not 0 <= i < engine.scene.labels.size for i in removed)):
        raise ValueError('JOINT_CRANIAL_NATIVE_REMOVED_LEDGER_INVALID')
    cells = set(removed)
    if len(cells) != len(removed):
        raise ValueError('JOINT_CRANIAL_NATIVE_REMOVED_LEDGER_INVALID')
    target = {i for i in cells if int(engine.scene.labels.flat[i]) in limits._target_labels}
    bone, prep, work = cells & limits._bone, cells & limits._preparation, cells & limits._work
    volume = engine.scene.voxel_mm3
    result = {'boneRemovedMm3': len(bone)*volume,
        'preparationNonboneRemovedMm3': len(prep)*volume,
        'workNonTargetRemovedMm3': len(work-target)*volume,
        'targetRemovedMm3': len(target)*volume,
        'wholeNonTargetRemovedMm3': len(cells-target)*volume,
        'uniqueRemovedSourceCells': len(cells),
        'unauthorizedRemovedSourceCells': len(cells-bone-prep-work)}
    reasons = []
    if result['unauthorizedRemovedSourceCells']:
        reasons.append('REMOVAL_OUTSIDE_JOINT_SOURCE_PARTITIONS')
    for key, cap, reason in (
            ('boneRemovedMm3', limits.max_bone_mm3, 'TOTAL_BONE_CAP'),
            ('preparationNonboneRemovedMm3', limits.max_preparation_nonbone_mm3, 'TOTAL_PREPARATION_NONBONE_CAP'),
            ('workNonTargetRemovedMm3', limits.max_work_nonbone_mm3, 'TOTAL_WORK_NON_TARGET_CAP')):
        if result[key] > cap+1e-9:
            reasons.append(reason)
    if abs(result['wholeNonTargetRemovedMm3']-(result['boneRemovedMm3']+
            result['preparationNonboneRemovedMm3']+result['workNonTargetRemovedMm3'])) > 1e-9:
        reasons.append('JOINT_PARTITION_LEDGER_MISMATCH')
    return result, reasons


def joint_cranial_ledger(engine, limits):
    """Whole unique source costs, never baseline-subtracted after preparation."""
    _require(engine, limits, limits._meter)
    result, reasons = _burden(engine, limits, engine.state['removed'])
    elapsed = time.monotonic()-limits.frozen_at
    if len(engine.actions) > limits.max_native_actions:
        reasons.append('NATIVE_ACTION_BUDGET')
    if limits._meter.calls > limits.max_public_calls:
        reasons.append('PUBLIC_CALL_BUDGET')
    if elapsed > limits.max_wall_seconds:
        reasons.append('WALL_BUDGET')
    return {**result, 'withinLimits': not reasons, 'refusalReasons': reasons,
        'nativeActions': len(engine.actions), 'spentPublicCalls': limits._meter.calls,
        'wallSecondsSinceMeterCreation': elapsed,
        'modeledElapsedSeconds': engine.state['elapsedSeconds'],
        'priorExposureCost': 'unknown for an assumed conditional post-scalp start',
        'clinicalValidation': 'none'}


def _joint_bone_capacity_partial(engine, action, preview):
    """Retain stock finite-work prefixes for either opening method.

    Standardized desktop physics makes flap kerfs require finite work too.
    Only the same single-pose drill command can continue under the existing
    bound. Native geometry, stage guards and preview B/P/W checks still apply.
    """
    if _milling_capacity_partial(engine, action, preview):
        return True
    from .surgical_craniotomy import CranialAccessContract
    from .surgical_drilling_v2 import profile
    fraction, elapsed = preview.get('completedFraction'), preview.get('elapsedSeconds')
    return (type(engine._cranial_access) is CranialAccessContract and
        profile(engine) is not None and action.get('operation') == 'apply' and
        'path' not in action and type(action.get('pose')) is dict and
        action.get('slot') in ('left', 'right') and
        engine.state['slots'][action['slot']]['instrument'] == 'cranial_drill' and
        preview.get('status') == 'partial' and preview.get('reason') == 'BONE_CUTTING_IN_PROGRESS' and
        type(fraction) in (int, float) and math.isfinite(fraction) and 0 <= fraction <= 1. and
        type(elapsed) in (int, float) and math.isfinite(elapsed) and elapsed > 0.)


def checked_joint_actions(engine, actions, *, limits, meter=None, check=None,
        phase='cranial.joint', _allow_tissue_partial=False):
    """Preview/commit under B/P/W; an interrupted physical prefix stays charged."""
    if not isinstance(actions, (list, tuple)) or not isinstance(phase, str) or not phase:
        raise ValueError('Joint program requires native action mappings and named phase')
    if check is not None and not callable(check):
        raise TypeError('Joint cancellation check must be callable')
    meter = limits._meter if meter is None else meter
    _require(engine, limits, meter)
    _, source_refusals = _burden(engine, limits, engine.state['removed'])
    if source_refusals:
        raise ValueError('JOINT_CRANIAL_PREFIX_'+source_refusals[0])
    candidate = meter.wrap(engine).clone()
    start_state, start_actions, start_calls = candidate.state_id, len(candidate.actions), meter.calls
    attempts, status, reason = [], 'accepted', None

    def guard():
        if check is not None:
            check()
        if time.monotonic()-limits.frozen_at >= limits.max_wall_seconds:
            raise TimeoutError('Joint cranial wall bound since meter creation')

    def require_call():
        guard()
        if meter.calls >= limits.max_public_calls:
            raise TransitionBudgetExceeded('Joint cranial shared public-call bound')

    with meter.phase(phase):
        for index, original in enumerate(copy.deepcopy(actions)):
            if type(original) is not dict:
                raise ValueError('Joint actions must be native mappings')
            for continuation in range(limits.max_bone_contact_replans+1):
                entry = {'index': index, 'action': copy.deepcopy(original),
                         'atActionPosition': len(candidate.actions), 'continuation': continuation}
                attempts.append(entry)
                try:
                    require_call()
                    if len(candidate.actions) >= limits.max_native_actions:
                        status, reason = 'interrupted', 'NATIVE_ACTION_BUDGET'
                        break
                    if _inserted_geometry_change(candidate, original):
                        status, reason = 'refused', 'WITHDRAW_BEFORE_GEOMETRY_CHANGE'
                        break
                    before = candidate.state_id
                    preview = candidate.preview(original, check=guard, include_view=True, include_history=False)
                    if candidate.state_id != before:
                        raise ValueError('Joint preview changed accepted prefix')
                    entry.update(previewStatus=preview['status'], previewReason=preview.get('reason'),
                                 previewStateId=preview['stateId'])
                    bone_partial = _joint_bone_capacity_partial(candidate, original, preview)
                    tissue_partial = (_allow_tissue_partial and original.get('operation') == 'apply' and
                        candidate.state['slots'][original['slot']]['instrument'] == 'cusa' and
                        preview['status'] == 'partial' and preview.get('reasonCode') == 'TISSUE_ASPIRATION_IN_PROGRESS' and
                        type(preview.get('elapsedSeconds')) in (int, float) and preview['elapsedSeconds'] > 0.)
                    accepted = preview['status'] == 'accepted' and preview.get('completedFraction') == 1.
                    if not (accepted or bone_partial or tissue_partial):
                        status, reason = 'refused', preview.get('reason') or 'FULL_NATIVE_ENDPOINT_REQUIRED'
                        break
                    ledger, refusals = _burden(candidate, limits, preview['view']['removed'])
                    entry['previewLedger'] = ledger
                    if not set(candidate.state['removed']) <= set(preview['view']['removed']):
                        raise ValueError('Joint native removal ledger regressed')
                    if refusals:
                        status, reason = 'refused', refusals[0]
                        break
                    if accepted and not _endpoint_reached(original, preview['view']):
                        status, reason = 'refused', 'FULL_NATIVE_ENDPOINT_REQUIRED'
                        break
                    require_call()
                    result = candidate.step(original, check=guard)
                    if (any(result.get(k) != preview.get(k) for k in
                            ('status', 'stateId', 'reason', 'completedFraction', 'elapsedSeconds')) or
                            candidate.state['removed'] != preview['view']['removed']):
                        raise ValueError('Joint preview/commit differs')
                    entry.update(status='committed', committedStateId=result['stateId'],
                                 endpointReached=accepted)
                    if tissue_partial:
                        status, reason = 'interrupted', 'PARTIAL_MOTION_REPLAN_REQUIRED'
                        break
                    if accepted:
                        break
                    if continuation == limits.max_bone_contact_replans:
                        status, reason = 'interrupted', 'BONE_CONTACT_CONTINUATION_BUDGET'
                        break
                except _INTERRUPTIONS as error:
                    status, reason = 'interrupted', type(error).__name__+': '+str(error)
                    break
            if status != 'accepted':
                break
    if attempts and attempts[-1].get('status') != 'committed':
        attempts[-1].update(status=status, reason=reason)
    if status == 'accepted':
        try:
            guard()
        except _INTERRUPTIONS as error:
            status, reason = 'interrupted', type(error).__name__+': '+str(error)
    readiness = _readiness(candidate)
    outcome = _outcome(candidate, limits)
    ledger = joint_cranial_ledger(candidate, limits)
    if status == 'accepted':
        try:
            guard()
        except _INTERRUPTIONS as error:
            status, reason = 'interrupted', type(error).__name__+': '+str(error)
    _finish_ledger_time(ledger, limits)
    return candidate, {'schema': SCHEMA, 'status': status, 'reason': reason,
        'sourceStateId': start_state, 'finalStateId': candidate.state_id,
        'admittedPatientTask': False, 'caseQualificationRequired': True,
        'retentionRole': 'checked_native_program' if status == 'accepted' else 'stopped_native_prefix_not_admitted',
        'nativeActionsAppended': len(candidate.actions)-start_actions,
        'publicCalls': meter.calls-start_calls, 'attempts': attempts, 'ledger': ledger,
        'limits': {**limits.declaration(), 'contentHash': limits.content_hash},
        'stageReadiness': readiness, 'sequenceOutcome': outcome,
        'clinicalValidation': 'none'}


def _readiness(engine):
    from .surgical_cranial_stages import stage_readiness
    return stage_readiness(engine.unwrap() if isinstance(engine, MeteredEngine) else engine)


def _finish_ledger_time(ledger, limits):
    ledger['wallSecondsSinceMeterCreation'] = time.monotonic()-limits.frozen_at
    if ledger['wallSecondsSinceMeterCreation'] >= limits.max_wall_seconds:
        if 'WALL_BUDGET' not in ledger['refusalReasons']:
            ledger['refusalReasons'].append('WALL_BUDGET')
        ledger['withinLimits'] = False


def _outcome(engine, limits):
    from .surgical_sequence_outcomes import sequence_outcome
    value = sequence_outcome(engine, baseline=limits._source_engine)
    ledger, _ = _burden(engine, limits, engine.state['removed'])
    for field, dimension in (('wholeNonTargetRemovedMm3', 'nonTargetRemovedMm3'),
                              ('targetRemovedMm3', 'targetRemovedMm3')):
        if abs(ledger[field]-value['dimensions'][dimension]['cumulative']) > 1e-9:
            raise ValueError('JOINT_CRANIAL_NATIVE_LEDGER_MISMATCH')
    return value


def checked_joint_cusa_motion(engine, motion, *, limits, meter=None, check=None,
        dwell_seconds=.6, phase='cranial.joint.cusa'):
    """Reuse the finite-contact remainder generator under the joint ledger."""
    from .surgical_cusa_v3 import profile
    if (type(motion) is not dict or motion.get('operation') != 'apply' or
            type(dwell_seconds) not in (int, float) or not 0 < dwell_seconds <= 30.):
        raise ValueError('Joint CUSA motion requires an apply and bounded dwell')
    hand = motion['slot']
    if engine.state['slots'][hand]['instrument'] != 'cusa' or profile(engine) is None:
        raise ValueError('Joint finite-contact continuation requires native CUSA v3')
    meter = limits._meter if meter is None else meter
    _require(engine, limits, meter)
    start_actions, start_calls, start_state = len(engine.actions), meter.calls, engine.state_id
    candidate, command, parts, continuations = engine, copy.deepcopy(motion), [], []
    start_pose = {k: copy.deepcopy(engine.state['slots'][hand][k]) for k in ('position', 'quaternion')}
    while True:
        candidate, part = checked_joint_actions(candidate, [command], limits=limits,
            meter=meter, check=check, phase=phase+'.motion', _allow_tissue_partial=True)
        parts.append(part)
        if part['status'] != 'interrupted' or part['reason'] != 'PARTIAL_MOTION_REPLAN_REQUIRED':
            break
        if len(continuations) >= limits.max_tissue_contact_replans:
            part = {**part, 'reason': 'TISSUE_CONTACT_CONTINUATION_BUDGET'}
            parts[-1] = part
            break
        observed = {k: copy.deepcopy(candidate.state['slots'][hand][k]) for k in ('position', 'quaternion')}
        remaining = _remaining_contact_action(command, start_pose, observed, candidate.history[-1])
        clearance = _contact_clearance_action(candidate, hand, command, dwell_seconds)
        if remaining is None or clearance is None:
            break
        continuations.append({'contactStateId': candidate.state_id, 'observedPose': observed,
                              'clearanceAction': clearance, 'remainingAction': remaining})
        candidate, cleared = checked_joint_actions(candidate, [clearance], limits=limits,
            meter=meter, check=check, phase=phase+'.clear')
        parts.append(cleared)
        if cleared['status'] != 'accepted':
            part = cleared
            break
        start_pose = {k: copy.deepcopy(candidate.state['slots'][hand][k]) for k in ('position', 'quaternion')}
        command = remaining
    readiness, outcome = _readiness(candidate), _outcome(candidate, limits)
    ledger = joint_cranial_ledger(candidate, limits)
    _finish_ledger_time(ledger, limits)
    status, reason = part['status'], part['reason']
    if not ledger['withinLimits'] and status == 'accepted':
        status, reason = 'interrupted', ledger['refusalReasons'][0]
    return candidate, {'schema': SCHEMA, 'status': status, 'reason': reason,
        'sourceStateId': start_state, 'finalStateId': candidate.state_id,
        'nativeActionsAppended': len(candidate.actions)-start_actions,
        'publicCalls': meter.calls-start_calls, 'parts': parts, 'contactContinuations': continuations,
        'ledger': ledger, 'stageReadiness': readiness, 'sequenceOutcome': outcome,
        'admittedPatientTask': False, 'caseQualificationRequired': True, 'clinicalValidation': 'none'}
