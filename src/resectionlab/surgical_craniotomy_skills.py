"""Prospectively bounded, public-native complete bone-opening sequences.

This is a separate bone-only engineering policy, not a larger tumor SkillLimits
budget or patient qualification. Callers supply the declared action sequence;
all physics, whole-flap transport and aperture guards remain native.
"""
from __future__ import annotations

import asyncio
import copy
from dataclasses import dataclass, field, fields
import math
import time

import numpy as np
from scipy.spatial.transform import Rotation

from .core import semantic_digest
from .surgical_craniotomy import require_evidence, validate_contract
from .surgical_engine import SurgicalEngine
from .surgical_transition_meter import (
    MeteredEngine, TransitionBudgetExceeded, TransitionMeter)


SCHEMA = 'resectionlab.checked-craniotomy-skill.v1'
_INTERRUPTIONS = (TransitionBudgetExceeded, TimeoutError, InterruptedError,
                  KeyboardInterrupt, asyncio.CancelledError)
_TOOLS = frozenset(('cranial_drill', 'craniotome', 'periosteal_elevator',
                    'dissector', 'bone_flap_holder'))
_OPERATIONS = frozenset(('configure', 'exchange', 'place', 'apply', 'grasp',
                         'move', 'extract', 'withdraw', 'inspect', 'release', 'deactivate'))


@dataclass(frozen=True)
class BoneOpeningLimits:
    """Explicit whole-bone budget frozen against one source and shared meter.

    Use freeze_bone_opening_limits before actions. There is no default bone
    allowance. Calls spent on discarded branches and elapsed wall time remain
    spent; no fresh meter, baseline subtraction or repeated invocation refund.
    """
    max_total_bone_mm3: float
    max_native_actions: int
    max_public_calls: int
    max_wall_seconds: float
    source_scene_id: str
    source_state_id: str
    source_actions_seal: str
    source_binding: str
    contract_hash: str
    guard_seal: str
    baseline_removed_cells: tuple[int, ...]
    flap_attachment_ids: tuple[str, ...]
    nonbone_effects_seal: str
    frozen_at: float
    meter_identity: int
    max_additional_nonbone_mm3: float = field(default=0., init=False)
    max_additional_target_mm3: float = field(default=0., init=False)
    content_hash: str = ''
    _meter: TransitionMeter = field(default=None, repr=False, compare=False)

    def __post_init__(self):
        for value in (self.max_total_bone_mm3, self.max_wall_seconds):
            if type(value) not in (int, float) or not math.isfinite(value) or value < 0:
                raise ValueError('Bone-opening volume/wall bounds must be explicit finite nonnegative numbers')
        if self.max_wall_seconds <= 0:
            raise ValueError('Bone-opening wall limit must be positive')
        if any(type(value) is not int or value < 1 for value in
               (self.max_native_actions, self.max_public_calls)):
            raise ValueError('Bone-opening action/call bounds must be positive integers')

    def declaration(self):
        return {row.name: getattr(self, row.name) for row in fields(self)
                if row.name not in ('content_hash', '_meter')}


def _contract(engine):
    engine.scene.require_cache_source()
    contract = require_evidence(validate_contract(engine.scene))
    if engine._cranial_access != contract:
        raise ValueError('Native cranial contract differs from source declaration')
    return contract


def _removed(engine):
    cells = engine.state['removed']
    if (any(type(cell) is not int or not 0 <= cell < engine.scene.labels.size for cell in cells)
            or len(set(cells)) != len(cells)):
        raise ValueError('Invalid native unique removed-cell ledger')
    return set(cells)


def _bone_cells(engine):
    ids = [row['id'] for row in engine.scene.regions if row['material'] == 'bone']
    return set(map(int, np.flatnonzero(np.isin(engine.scene.labels, ids))))


def _nonbone_effects(view, bone, flap_attachment_ids=()):
    """Keep nonbone topology, transport and tissue effects exactly unchanged."""
    data = {key: view.get(key) for key in ('deformation', 'deformationModel',
        'vesselConnections', 'vesselWounds', 'vesselsV2', 'vascularGraphV1',
        'bipolarThermalV1', 'cusaV3', 'sources', 'separatedAttachments', 'meshSeams')}
    # Mechanics revision 2 initializes an empty CUSA ledger even though the
    # inactive hand has performed no treatment. Preserve every actual effect.
    cusa = data['cusaV3']
    if isinstance(cusa, dict) and all(not value for key, value in cusa.items() if key != 'schema'):
        data['cusaV3'] = None
    data['targetRemovedMm3'] = view['metrics']['targetRemovedMm3']
    data['separatedAttachments'] = [value for value in (view.get('separatedAttachments') or [])
                                    if value not in flap_attachment_ids]
    data['removed'] = [cell for cell in view['removed'] if cell not in bone]
    data['moved'] = [row for row in view['moved'] if row[0] not in bone]
    data['thermal'] = {key: value for key, value in view['thermal'].items() if int(key) not in bone}
    for key in ('cutEdges', 'transectedEdges', 'meshCutEdges'):
        data[key] = [row for row in (view.get(key) or []) if any(cell not in bone for cell in row)]
    return semantic_digest(data)


def _require_prefix_call_evidence(engine, meter):
    """Never treat an imported native prefix's unknown execution costs as zero."""
    if not engine.actions:
        canonical = SurgicalEngine(engine.scene, protection=engine.protection,
                                   guard_config=engine.guard_config)
        if engine.history or engine.state_id != canonical.state_id:
            raise ValueError('CRANIOTOMY_CANONICAL_INTACT_SOURCE_REQUIRED')
        return
    failure = 'CRANIOTOMY_PREFIX_CALL_EVIDENCE_REQUIRED'
    if not isinstance(engine, MeteredEngine) or engine.meter is not meter:
        raise ValueError(failure)
    report = meter.report()
    lineage, branch_id = set(), engine.branch_id
    while branch_id is not None:
        if branch_id in lineage or str(branch_id) not in report['branches']:
            raise ValueError(failure)
        lineage.add(branch_id)
        branch_id = report['branches'][str(branch_id)]['parentBranchId']
    history = engine.history
    if len(history) != len(engine.actions) or not any(
            report['branches'][str(key)]['initialActions'] == 0 and
            report['branches'][str(key)]['initialStateId'] == history[0]['stateBefore']
            for key in lineage):
        raise ValueError(failure)
    for index, (action, event) in enumerate(zip(engine.actions, history)):
        if (event.get('status') != 'accepted' or event.get('completedFraction') != 1. or
                event.get('action') != action or not any(
                    row['branchId'] in lineage and row['kind'] == 'step' and
                    row.get('status') == 'accepted' and row['actionsBefore'] == index and
                    row.get('actionsAfter') == index+1 and row['stateBefore'] == event['stateBefore'] and
                    row.get('stateAfter') == event['stateAfter'] and row.get('action') == action
                    for row in report['trace'])):
            raise ValueError(failure)
    if history[-1]['stateAfter'] != engine.state_id:
        raise ValueError(failure)


def freeze_bone_opening_limits(engine, *, max_total_bone_mm3,
        max_native_actions, max_public_calls, max_wall_seconds, meter):
    """Bind a separate explicit bone-only policy before any proposed action."""
    if not isinstance(engine, (SurgicalEngine, MeteredEngine)) or not isinstance(meter, TransitionMeter):
        raise TypeError('Bone opening requires authoritative native engine and shared meter')
    if isinstance(engine, MeteredEngine) and engine.meter is not meter:
        raise ValueError('Cannot replace shared bone-opening transition meter')
    _require_prefix_call_evidence(engine, meter)
    contract = _contract(engine)
    bone, removed = _bone_cells(engine), _removed(engine)
    flap_attachment_ids = tuple(sorted(attachment['id']
        for attachment in engine.scene.properties['attachments']
        if attachment['kind'] in ('periosteal', 'soft') and
        set(attachment['cells']).intersection(contract.flap_cells)))
    if any(slot['active'] and slot['instrument'] not in _TOOLS
           for slot in engine.state['slots'].values()):
        raise ValueError('Bone-only source cannot have an active nonbone instrument')
    if len(removed & bone)*engine.scene.voxel_mm3 > max_total_bone_mm3+1e-9:
        raise ValueError('Existing bone prefix already exceeds prospective whole-bone cap')
    if len(engine.actions) > max_native_actions or meter.calls > max_public_calls:
        raise ValueError('Existing action/call costs already exceed prospective budget')
    values = dict(max_total_bone_mm3=max_total_bone_mm3,
        max_native_actions=max_native_actions, max_public_calls=max_public_calls,
        max_wall_seconds=max_wall_seconds, source_scene_id=engine.scene.identity,
        source_state_id=engine.state_id, source_actions_seal=semantic_digest(engine.actions),
        source_binding=contract.source_binding, contract_hash=contract.content_hash,
        guard_seal=semantic_digest(engine.guard_report()), baseline_removed_cells=tuple(sorted(removed)),
        flap_attachment_ids=flap_attachment_ids,
        nonbone_effects_seal=_nonbone_effects(engine.view(include_history=False), bone, flap_attachment_ids),
        frozen_at=time.monotonic(), meter_identity=id(meter), _meter=meter)
    existing = getattr(meter, '_craniotomy_opening_limits', None)
    if existing is not None:
        if (not isinstance(existing, BoneOpeningLimits) or existing._meter is not meter or
                existing.content_hash != semantic_digest(existing.declaration()) or
                any(getattr(existing, key) != value for key, value in values.items()
                    if key not in ('_meter', 'frozen_at'))):
            raise ValueError('CRANIOTOMY_BUDGET_ALREADY_FROZEN')
        return existing
    result = BoneOpeningLimits(**values)
    result = BoneOpeningLimits(**values, content_hash=semantic_digest(result.declaration()))
    meter._craniotomy_opening_limits = result
    return result


def _endpoint_reached(action, view):
    endpoint = action.get('pose')
    if endpoint is None and 'path' in action:
        endpoint = action['path'][-1]
    if endpoint is None:
        return True
    slot = view['slots'][action['slot']]
    return (np.linalg.norm(np.subtract(slot['position'], endpoint['position'])) <= 1e-6 and
        (Rotation.from_quat(slot['quaternion'])*
         Rotation.from_quat(endpoint['quaternion']).inv()).magnitude() <= 1e-6)


def checked_craniotomy_actions(engine, actions, *, limits, meter=None,
                               check=None, phase='craniotomy.opening'):
    """Return a metered clone and an admitted opening or actual stopped prefix.

    Every accepted endpoint is previewed before commit. Partial native actions
    are refused, including cutting-in-progress. A cap-violating whole-flap
    preview is charged as a call but never committed or counted as an opening.
    """
    if not isinstance(limits, BoneOpeningLimits) or not isinstance(actions, (list, tuple)):
        raise ValueError('Invalid checked craniotomy invocation')
    if not isinstance(engine, (SurgicalEngine, MeteredEngine)) or not isinstance(phase, str) or not phase:
        raise ValueError('Bone opening requires native engine and named phase')
    if check is not None and not callable(check):
        raise TypeError('Bone-opening cancellation check must be callable')
    meter = limits._meter if meter is None else meter
    if (not isinstance(meter, TransitionMeter) or meter is not limits._meter or
            id(meter) != limits.meter_identity or
            isinstance(engine, MeteredEngine) and engine.meter is not meter):
        raise ValueError('Cannot replace shared bone-opening transition meter')
    if limits is not getattr(meter, '_craniotomy_opening_limits', None):
        raise ValueError('CRANIOTOMY_FROZEN_BUDGET_IDENTITY_REQUIRED')
    if limits.content_hash != semantic_digest(limits.declaration()):
        raise ValueError('Frozen bone-opening limits were changed')
    contract = _contract(engine)
    _require_prefix_call_evidence(engine, meter)
    if (engine.scene.identity != limits.source_scene_id or engine.state_id != limits.source_state_id or
            semantic_digest(engine.actions) != limits.source_actions_seal or
            contract.content_hash != limits.contract_hash or contract.source_binding != limits.source_binding or
            semantic_digest(engine.guard_report()) != limits.guard_seal or
            tuple(sorted(_removed(engine))) != limits.baseline_removed_cells):
        raise ValueError('Frozen bone-opening source/state/contract binding differs')
    prepared = copy.deepcopy(actions)
    bone = _bone_cells(engine)
    if _nonbone_effects(engine.view(include_history=False), bone, limits.flap_attachment_ids) != limits.nonbone_effects_seal:
        raise ValueError('Frozen nonbone source effects differ')
    candidate = meter.wrap(engine).clone()
    start_actions, start_calls = len(candidate.actions), meter.calls
    attempts, status, reason = [], 'accepted', None

    def guard():
        if check is not None:
            check()
        if time.monotonic()-limits.frozen_at >= limits.max_wall_seconds:
            raise TimeoutError('Bone-opening frozen wall limit reached')

    def require_call():
        guard()
        if meter.calls >= limits.max_public_calls:
            raise TransitionBudgetExceeded('Bone-opening public native-call cap reached')

    with meter.phase(phase):
        for index, action in enumerate(prepared):
            entry = {'index': index, 'action': copy.deepcopy(action),
                     'atActionPosition': len(candidate.actions)}
            attempts.append(entry)
            try:
                require_call()
                if len(candidate.actions) >= limits.max_native_actions:
                    status, reason = 'interrupted', 'NATIVE_ACTION_BUDGET'
                    break
                if not isinstance(action, dict):
                    raise ValueError('Bone-opening actions must be native mappings')
                slot = candidate.state['slots'].get(action.get('slot'))
                if (action.get('operation') not in _OPERATIONS or slot is None or
                        action.get('instrument', slot['instrument']) not in _TOOLS):
                    status, reason = 'refused', 'BONE_ONLY_ACTION_REQUIRED'
                    break
                before = candidate.state_id
                preview = candidate.preview(action, check=guard, include_view=True, include_history=False)
                if candidate.state_id != before:
                    raise ValueError('Craniotomy preview changed accepted source prefix')
                cells = set(preview['view']['removed'])
                entry.update(previewStatus=preview['status'], previewStateId=preview['stateId'],
                    previewBoneRemovedMm3=len(cells & bone)*candidate.scene.voxel_mm3,
                    previewRemovedCellIds=sorted(set(preview.get('deltas', {}).get('removedCells', ()))))
                reason = (preview.get('reason') or 'FULL_ACCEPTED_ENDPOINT_REQUIRED'
                    if preview['status'] != 'accepted' or preview.get('completedFraction') != 1. else None)
                if reason is None and not _endpoint_reached(action, preview['view']):
                    reason = 'FULL_ACCEPTED_ENDPOINT_REQUIRED'
                if reason is None and entry['previewBoneRemovedMm3'] > limits.max_total_bone_mm3+1e-9:
                    reason = 'TOTAL_BONE_CAP'
                if reason is None and (_nonbone_effects(preview['view'], bone, limits.flap_attachment_ids) != limits.nonbone_effects_seal):
                    reason = 'NONBONE_EFFECT_NOT_ALLOWED'
                # Removal remains monotonic and confined to the declared F/K.
                if reason is None and (not _removed(candidate) <= cells or
                        (cells-_removed(candidate)) - (set(contract.flap_cells) | set(contract.kerf_cells))):
                    reason = 'REMOVAL_OUTSIDE_DECLARED_FLAP_KERF'
                if reason is not None:
                    status = 'refused'
                    break
                require_call()
                result = candidate.step(action, check=guard)
                if (result['status'] != 'accepted' or result['stateId'] != preview['stateId'] or
                        result.get('completedFraction') != 1. or _removed(candidate) != cells or
                        not _endpoint_reached(action, candidate.observe())):
                    raise ValueError('Native craniotomy preview/commit endpoint differs')
                entry.update(status='committed', committedStateId=result['stateId'], endpointReached=True)
            except _INTERRUPTIONS as error:
                status, reason = 'interrupted', type(error).__name__+': '+str(error)
                break
        else:
            try:
                guard()
            except _INTERRUPTIONS as error:
                status, reason = 'interrupted', type(error).__name__+': '+str(error)

    observation = candidate.observe()
    native = observation['craniotomy']
    cells = _removed(candidate)
    clean = (not candidate.state['holds'] and not candidate.state.get('detachedFragments') and
        all(not slot['inserted'] and not slot['active'] and not slot['grip']
            for slot in candidate.state['slots'].values()))
    complete = (native['openingComplete'] and native['apertureEvacuated'] and clean and
        set(contract.flap_cells) | set(contract.kerf_cells) <= cells and
        _nonbone_effects(candidate.view(include_history=False), bone, limits.flap_attachment_ids) == limits.nonbone_effects_seal)
    if status == 'accepted' and not complete:
        status, reason = 'refused', 'COMPLETE_NATIVE_OPENING_REQUIRED'
    if attempts and attempts[-1].get('status') != 'committed':
        attempts[-1].update(status=status, reason=reason)
    trace = copy.deepcopy(meter.trace[start_calls:])
    receipt = {'schema': SCHEMA, 'status': status, 'reason': reason,
        'admittedOpening': status == 'accepted' and complete,
        'retentionRole': ('accepted_native_bone_opening' if status == 'accepted' else
                          'stopped_native_prefix_not_admitted'),
        'clinicalValidation': 'none', 'evidenceScope': contract.evidence_scope,
        'limits': {**limits.declaration(), 'contentHash': limits.content_hash},
        'sourceStateId': limits.source_state_id, 'finalStateId': candidate.state_id,
        'sourceNativeActions': start_actions, 'nativeActionsAppended': len(candidate.actions)-start_actions,
        'committedPrefix': {'nativeActions': len(candidate.actions), 'stateId': candidate.state_id,
                            'actionPrefixSeal': semantic_digest(candidate.actions)},
        'nativeOpening': native, 'attempts': attempts,
        'ledger': {'boneRemovedMm3': len(cells & bone)*candidate.scene.voxel_mm3,
            'separatedFlapAttachmentIds': sorted(set(candidate.state['separatedAttachments']) & set(limits.flap_attachment_ids)),
            'additionalNonboneRemovedMm3': len((cells-set(limits.baseline_removed_cells))-bone)*candidate.scene.voxel_mm3,
            'targetRemovedMm3': observation['metrics']['targetRemovedMm3'],
            'nonTargetRemovedMm3': observation['metrics']['nonTargetRemovedMm3'],
            'spentPublicCalls': meter.calls, 'nativeActions': len(candidate.actions),
            'wallSecondsSinceFreeze': time.monotonic()-limits.frozen_at},
        'physicalCallCost': {'publicCalls': len(trace),
            'previewCalls': sum(row['kind'] == 'preview' for row in trace),
            'directStepCalls': sum(row['kind'] == 'step' for row in trace)},
        'transitionTrace': trace, 'sharedTransitionMeter': meter.report()}
    return candidate, receipt
