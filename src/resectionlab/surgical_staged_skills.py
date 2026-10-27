"""Checked, resumable native skill primitives for staged research rehearsals.

This module adds no physics or learned policy. A caller owns source-defined
poses, option inventory and task gate; every attempted transition remains a
public SurgicalEngine preview/step on a clone and shares a TransitionMeter.
"""
from __future__ import annotations

import asyncio
import copy
from dataclasses import asdict, dataclass, replace
import hashlib
import math

import numpy as np
from scipy.spatial.transform import Rotation

from .core import semantic_digest
from .surgical_engine import SCHEMA as ENGINE_SCHEMA, SurgicalEngine
from .surgical_tools import TOOLS
from .surgical_transition_meter import (
    MeteredEngine, TransitionBudgetExceeded, TransitionMeter)


SCHEMA = 'resectionlab.staged-native-skill.v1'
CHECKPOINT_SCHEMA = 'resectionlab.staged-native-checkpoint.v1'
_INTERRUPTIONS = (TransitionBudgetExceeded, TimeoutError, KeyboardInterrupt,
                  InterruptedError, asyncio.CancelledError)


@dataclass(frozen=True)
class SkillLimits:
    max_native_actions: int
    max_public_calls: int
    max_total_non_target_mm3: float = 200.
    max_additional_non_target_mm3: float | None = None
    baseline_non_target_mm3: float | None = None
    allowed_removed_label_ids: tuple[int, ...] | None = None
    prior_public_calls: int = 0

    def __post_init__(self):
        if (type(self.max_native_actions) is not int or self.max_native_actions < 1 or
                type(self.max_public_calls) is not int or self.max_public_calls < 1 or
                type(self.prior_public_calls) is not int or
                not 0 <= self.prior_public_calls < self.max_public_calls):
            raise ValueError('Staged native action/call limits differ')
        total = self.max_total_non_target_mm3
        if (type(total) not in (int, float) or not math.isfinite(total) or
                not 0 <= total <= 200.):
            raise ValueError('Staged total non-target cap must remain at most 200 mm3')
        additional, baseline = (self.max_additional_non_target_mm3,
                                self.baseline_non_target_mm3)
        if (additional is None) != (baseline is None):
            raise ValueError('Additional cap requires an exact native baseline')
        if additional is not None and (type(additional) not in (int, float) or
                type(baseline) not in (int, float) or
                not math.isfinite(additional) or not math.isfinite(baseline) or
                additional < 0 or baseline < 0 or additional > 200.):
            raise ValueError('Invalid staged continuation non-target cap')
        if self.allowed_removed_label_ids is not None:
            labels = self.allowed_removed_label_ids
            if (not isinstance(labels, (tuple, list)) or not labels or
                    any(type(value) is not int or value < 0 for value in labels) or
                    len(set(labels)) != len(labels)):
                raise ValueError('Allowed source removal labels differ')
            object.__setattr__(self, 'allowed_removed_label_ids', tuple(labels))


@dataclass(frozen=True)
class SkillAction:
    """One native action with collision retries changing duration only."""
    action: dict
    retry_durations_seconds: tuple[float, ...] = ()
    require_endpoint: bool = False

    def __post_init__(self):
        if not isinstance(self.action, dict):
            raise TypeError('A staged action must be a native action mapping')
        if type(self.require_endpoint) is not bool:
            raise ValueError('Endpoint requirement must be boolean')
        if self.require_endpoint:
            endpoint = self.action.get('pose') or self.action.get('path', [None])[-1]
            if endpoint is None:
                raise ValueError('Endpoint-checked action requires a native pose or path')
            _finite_pose(endpoint)
        initial = self.action.get('duration', 1.)
        retries = self.retry_durations_seconds
        if (type(initial) not in (int, float) or not math.isfinite(initial) or
                not 0 < initial <= 30. or not isinstance(retries, (tuple, list)) or
                any(type(value) not in (int, float) or not math.isfinite(value) or
                    not initial < value <= 30. for value in retries) or
                list(retries) != sorted(set(retries))):
            raise ValueError('Duration retries must be unique, increasing and at most 30 s')
        object.__setattr__(self, 'action', copy.deepcopy(self.action))
        object.__setattr__(self, 'retry_durations_seconds', tuple(retries))


def _finite_pose(pose):
    if (not isinstance(pose, dict) or set(pose) != {'position', 'quaternion'} or
            not isinstance(pose['position'], (list, tuple)) or
            not isinstance(pose['quaternion'], (list, tuple)) or
            len(pose['position']) != 3 or len(pose['quaternion']) != 4 or
            any(type(x) not in (int, float) or not math.isfinite(x)
                for x in (*pose['position'], *pose['quaternion']))):
        raise ValueError('A source-defined staged pose must be finite RAS+ position/quaternion')
    return copy.deepcopy(pose)


def _positive_radius(value):
    if type(value) not in (int, float) or not math.isfinite(value) or not .1 <= value <= 12.:
        raise ValueError('Source-defined CUSA radius must be in native range')
    return float(value)


def propose_cusa_maneuvers(engine, route, *, max_options=15,
                           angle_step_deg=.5, advance_step_mm=.5,
                           max_steps=4):
    """Rank short, current-occupancy CUSA paths without a native transition.

    The route's source surface is the pivot anchor. Contact is a read-only
    union of *currently retained* source identities along each proposed tip
    path; it is neither a future removal estimate nor a full-tool clearance.
    The caller must still execute every path through checked native actions.
    """
    if not isinstance(engine, (SurgicalEngine, MeteredEngine)) or \
            not isinstance(route, dict) or route.get('sourceSceneId') != engine.scene.identity:
        raise ValueError('CUSA maneuver requires a source-bound native route')
    if (type(max_options) is not int or not 1 <= max_options <= 15 or
            type(max_steps) is not int or not 1 <= max_steps <= 4 or
            type(angle_step_deg) not in (int, float) or
            not math.isfinite(angle_step_deg) or not 0 < angle_step_deg <= .5 or
            type(advance_step_mm) not in (int, float) or
            not math.isfinite(advance_step_mm) or not 0 < advance_step_mm <= .5):
        raise ValueError('CUSA maneuver exceeds the declared local bounds')
    anchor = np.asarray(_finite_pose(route.get('surfacePose'))['position'], float)
    entry = np.asarray(_finite_pose(route.get('entryPose'))['position'], float)
    inward = np.asarray(route.get('inwardRasUnit'), float)
    if (inward.shape != (3,) or not np.isfinite(inward).all() or
            not math.isclose(float(np.linalg.norm(inward)), 1., abs_tol=1e-6)):
        raise ValueError('CUSA maneuver requires a unit source access ray')
    hands = [key for key, slot in engine.state['slots'].items()
             if slot['instrument'] == 'cusa' and slot['inserted']]
    if len(hands) != 1:
        raise ValueError('CUSA maneuver requires one inserted CUSA hand')
    slot = engine.state['slots'][hands[0]]
    start = np.asarray(slot['position'], float)
    orientation = Rotation.from_quat(slot['quaternion'])
    axis = orientation.apply([0., 0., 1.])
    radius = float(engine.parameters(slot)['radiusMm'])
    depth = float(engine.parameters(slot)['depthMm'])
    displacement = start-anchor
    length = float(np.linalg.norm(displacement))
    if length <= 1e-9:
        raise ValueError('CUSA tip coincides with sealed source surface anchor')
    target_ids = {int(row['id']) for row in engine.scene.regions
                  if row.get('target') and row['material'] == 'tumor'}
    brain_ids = {int(row['id']) for row in engine.scene.regions
                 if row['material'] == 'soft_tissue' and
                 (row['name'].strip().lower() in ('brain', 'whole brain') or
                  row['name'].strip().lower().startswith('mapped brain'))}
    labels = engine.scene.labels.ravel()
    steps_list = tuple(value for value in (1, 2, 4) if value <= max_steps)
    basis = np.eye(3)[int(np.argmin(np.abs(displacement/length)))]
    tangent_a = np.cross(displacement/length, basis)
    tangent_a /= np.linalg.norm(tangent_a)
    tangent_b = np.cross(displacement/length, tangent_a)
    state_id = engine.state_id
    prefix = semantic_digest({'sourceSceneId': engine.scene.identity,
        'route': route, 'stateId': state_id,
        'currentPose': {'position': start.tolist(),
                        'quaternion': list(slot['quaternion'])},
        'radiusMm': radius, 'depthMm': depth})
    proposals = []

    def add(direction, count, path, mode):
        previous = start
        touched = set()
        for pose in path:
            point = np.asarray(pose['position'])
            touched.update(map(int, engine.touched(previous, point, radius)))
            previous = point
        ids = {int(labels[cell]) for cell in touched}
        target = sum(int(labels[cell]) in target_ids for cell in touched)*engine.scene.voxel_mm3
        if target <= 0:
            return
        brain = sum(int(labels[cell]) in brain_ids for cell in touched)*engine.scene.voxel_mm3
        point = np.asarray(path[-1]['position'])
        along = float(np.dot(point-entry, inward))
        lateral = float(np.linalg.norm(point-entry-along*inward))
        proposals.append({'id': f'local-{prefix}-{direction}-{count}',
            'pose': copy.deepcopy(path[-1]), 'path': copy.deepcopy(path),
            'orientationMode': mode, 'depthFromTargetEntryMm': along,
            'radialFromOriginalRayMm': lateral,
            'currentGeometry': {'radiusMm': radius, 'depthMm': depth,
                'shaftRadiusMm': float(TOOLS['cusa']['shaftRadiusMm'])},
            'sourceFootprints': {str(radius): {'sourceTargetMm3': float(target),
                'sourceBrainMm3': float(brain),
                'sourceOtherLabelContact': bool(ids-target_ids-brain_ids)}},
            'proposalStateId': state_id})

    for count in steps_list:
        path = [{'position': (start+axis*advance_step_mm*k).tolist(),
                 'quaternion': list(slot['quaternion'])}
                for k in range(1, count+1)]
        add('forward', count, path, 'local-forward')
    # A pivot is local only when the current tool axis already follows the
    # source-anchored ray. A misaligned state needs a separately checked move.
    if float(np.dot(axis, displacement/length)) >= math.cos(1e-6):
        for name, tangent in (('a', tangent_a), ('b', tangent_b)):
            for sign in (-1, 1):
                for count in steps_list:
                    path = []
                    for k in range(1, count+1):
                        angle = math.radians(sign*angle_step_deg*k)
                        turn = Rotation.from_rotvec(tangent*angle)
                        ray = turn.apply(displacement)
                        quaternion = (turn*orientation).as_quat().tolist()
                        path.append({'position': (anchor+ray).tolist(),
                                     'quaternion': quaternion})
                    if any(np.linalg.norm(np.subtract(current['position'],
                                                       previous['position'])) >
                           advance_step_mm+1e-9 for previous, current in zip(
                               [{'position': start.tolist()}, *path[:-1]], path)):
                        continue
                    add(f'{name}{"plus" if sign > 0 else "minus"}', count,
                        path, 'local-surface-pivot')
    proposals.sort(key=lambda row: (
        row['sourceFootprints'][str(radius)]['sourceOtherLabelContact'],
        row['sourceFootprints'][str(radius)]['sourceBrainMm3'],
        -row['sourceFootprints'][str(radius)]['sourceTargetMm3'], row['id']))
    return proposals[:max_options]


def cusa_work_actions(slot, work_pose, *, radius_mm, duration_seconds,
                      parameters, transit_pose=None, transit_duration_seconds=.2,
                      retry_durations_seconds=()):
    """Build fixed-geometry CUSA work; source supplies power/flow/duty config."""
    if slot not in ('left', 'right') or not isinstance(parameters, dict) or \
            not {'power', 'flowMm3PerSecond', 'duty'} <= set(parameters) or \
            'radiusMm' in parameters:
        raise ValueError('Explicit CUSA slot and power/flow/duty catalog config required')
    radius = _positive_radius(radius_mm)
    work = _finite_pose(work_pose)
    config = copy.deepcopy(parameters)
    config['radiusMm'] = radius
    actions = []
    if transit_pose is not None:
        actions.append(SkillAction({'slot': slot, 'operation': 'apply',
            'pose': _finite_pose(transit_pose), 'parameters': copy.deepcopy(config),
            'duration': transit_duration_seconds}))
    actions.append(SkillAction({'slot': slot, 'operation': 'apply',
        'pose': work, 'parameters': copy.deepcopy(config), 'duration': duration_seconds},
        tuple(retry_durations_seconds)))
    return tuple(actions)


def checked_exit_actions(slot, outside_pose, *, radius_mm,
                         withdraw_duration_seconds, waypoints=()):
    """Build one declared deactivate/move/withdraw path with unchanged radius."""
    if slot not in ('left', 'right'):
        raise ValueError('Unknown native exit slot')
    radius = _positive_radius(radius_mm)
    result = [SkillAction({'slot': slot, 'operation': 'deactivate',
                           'duration': .2})]
    for pose, duration in waypoints:
        result.append(SkillAction({'slot': slot, 'operation': 'move',
            'pose': _finite_pose(pose), 'parameters': {'radiusMm': radius},
            'duration': duration}))
    result.append(SkillAction({'slot': slot, 'operation': 'withdraw',
        'pose': _finite_pose(outside_pose), 'parameters': {'radiusMm': radius},
        'duration': withdraw_duration_seconds}))
    return tuple(result)


def external_cusa_reconfiguration_actions(engine, slot, outside_pose, *,
                                          radius_mm, depth_mm=2.,
                                          exchange_duration_seconds=.2,
                                          place_duration_seconds=.2):
    """Only from full checked cleanup: exchange/place CUSA outside the source.

    The builder performs no transition. The caller must execute its returned
    actions through checked_actions, so native shaft/placement checks and all
    public calls remain authoritative. CUSA v3 keeps its fixed tip/shaft catalog.
    """
    if not isinstance(engine, (SurgicalEngine, MeteredEngine)) or \
            slot not in ('left', 'right'):
        raise ValueError('External CUSA reconfiguration requires a native slot')
    from .surgical_sector_planning import _cleanup_pending
    pending = _cleanup_pending(engine)
    if pending:
        raise ValueError('CHECKED_CLEANUP_REQUIRED_BEFORE_RECONFIGURATION: '+
                         ','.join(pending))
    radius = _positive_radius(radius_mm)
    if type(depth_mm) not in (int, float) or not math.isfinite(depth_mm) or \
            not .1 <= depth_mm <= 20.:
        raise ValueError('Source-defined CUSA depth must be in native range')
    if 'cusaV3' in engine.scene.properties:
        from .surgical_cusa_v3 import TIP_RADIUS_MM, validate
        validate(engine.scene)
        if radius != TIP_RADIUS_MM:
            raise ValueError('CUSA_V3_FIXED_CATALOG_GEOMETRY_REQUIRED')
    return (SkillAction({'slot': slot, 'operation': 'exchange',
                         'instrument': 'cusa', 'duration': exchange_duration_seconds}),
            SkillAction({'slot': slot, 'operation': 'place',
                'pose': _finite_pose(outside_pose),
                'parameters': {'radiusMm': radius, 'depthMm': float(depth_mm)},
                'duration': place_duration_seconds}))


def _allowed_labels(engine, limits):
    if limits.allowed_removed_label_ids is not None:
        known = {int(row['id']) for row in engine.scene.regions}
        if not set(limits.allowed_removed_label_ids) <= known:
            raise ValueError('Allowed removal label absent from source scene')
        return set(limits.allowed_removed_label_ids)
    return {int(row['id']) for row in engine.scene.regions
            if row.get('target') and row['material'] == 'tumor' or
               row['material'] == 'soft_tissue' and
               (row['name'].strip().lower() in ('brain', 'whole brain') or
                row['name'].strip().lower().startswith('mapped brain'))}


def _inserted_geometry_change(engine, action):
    """Refuse implicit tool-size changes within a physically inserted hand."""
    slot_name = action.get('slot')
    if slot_name not in engine.state['slots'] or \
            not engine.state['slots'][slot_name]['inserted']:
        return False
    requested = action.get('parameters', {})
    if not isinstance(requested, dict):
        return False  # The native validator owns malformed input rejection.
    slot = engine.state['slots'][slot_name]
    current = engine.parameters(slot)
    current_shaft = TOOLS[slot['instrument']]['shaftRadiusMm']
    for key, existing in (('radiusMm', current['radiusMm']),
                          ('depthMm', current['depthMm']),
                          ('shaftRadiusMm', current_shaft)):
        if key in requested and requested[key] != existing:
            return True
    return False


def _ledger(engine, limits, meter):
    removed = sorted(set(map(int, engine.state['removed'])))
    if len(removed) != len(engine.state['removed']):
        raise ValueError('Native removed-cell ledger contains duplicates')
    if removed and (removed[0] < 0 or removed[-1] >= engine.scene.labels.size):
        raise ValueError('Native removed-cell ledger lies outside source scene')
    values = engine.metrics()
    target_ids = {int(row['id']) for row in engine.scene.regions if row.get('target')}
    labels = engine.scene.labels.ravel()
    target = sum(int(labels[cell]) in target_ids for cell in removed)*engine.scene.voxel_mm3
    non_target = len(removed)*engine.scene.voxel_mm3-target
    if (abs(target-values['targetRemovedMm3']) > 1e-9 or
            abs(non_target-values['nonTargetRemovedMm3']) > 1e-9):
        raise ValueError('Native unique-cell ledger differs from measured metrics')
    if non_target > limits.max_total_non_target_mm3+1e-9:
        raise ValueError('Existing native prefix exceeds total non-target cap')
    if limits.max_additional_non_target_mm3 is not None:
        additional = non_target-limits.baseline_non_target_mm3
        if additional < -1e-9:
            raise ValueError('Continuation baseline exceeds existing native effects')
        if additional > limits.max_additional_non_target_mm3+1e-9:
            raise ValueError('Existing native prefix exceeds additional non-target cap')
    flat = np.asarray(removed, dtype='<i8')
    spent = limits.prior_public_calls+meter.calls
    if spent > limits.max_public_calls or len(engine.actions) > limits.max_native_actions:
        raise ValueError('Staged native work already exceeds declared task budget')
    return {'removedCellCount': len(removed),
            'removedCellSha256': hashlib.sha256(flat.tobytes()).hexdigest(),
            'stateId': engine.state_id,
            'targetRemovedMm3': float(target),
            'nonTargetRemovedMm3': float(non_target),
            'nativeActions': len(engine.actions),
            'remainingNativeActions': limits.max_native_actions-len(engine.actions),
            'spentPublicCalls': spent,
            'remainingPublicCalls': limits.max_public_calls-spent}


def _metered_candidate(engine, meter, limits):
    if isinstance(engine, MeteredEngine):
        if meter is not None and meter is not engine.meter:
            raise ValueError('A staged option cannot switch transition meters')
        meter = engine.meter
        candidate = engine.clone()
    elif isinstance(engine, SurgicalEngine):
        meter = (meter if meter is not None else TransitionMeter(
            limits.max_public_calls-limits.prior_public_calls))
        candidate = meter.wrap(engine).clone()
    else:
        raise TypeError('Staged skill requires authoritative native engine')
    return candidate, meter


def checked_actions(engine, actions, *, limits: SkillLimits, check=None,
                    meter: TransitionMeter | None = None, phase='staged.work'):
    """Append only checked native actions; return stopped prefixes on refusals.

    A cancellation or call/action budget returns status ``interrupted`` with
    the last committed prefix. Unexpected data/integrity errors still raise.
    Rejected previews never become a committed state or measured progress.
    """
    if not isinstance(limits, SkillLimits) or not isinstance(actions, (tuple, list)) or \
            not isinstance(phase, str) or not phase or \
            check is not None and not callable(check):
        raise ValueError('Invalid staged skill invocation')
    prepared = tuple(row if isinstance(row, SkillAction) else SkillAction(row)
                     for row in actions)
    candidate, meter = _metered_candidate(engine, meter, limits)
    start_ledger = _ledger(candidate, limits, meter)
    labels = _allowed_labels(candidate, limits)
    start_actions, start_calls = len(candidate.actions), meter.calls
    start_state = candidate.state_id
    attempts = []
    status, reason = 'accepted', None

    def guard_budget():
        if check is not None:
            check()
        if limits.prior_public_calls+meter.calls >= limits.max_public_calls:
            raise TransitionBudgetExceeded('Staged public native-call cap reached')

    with meter.phase(phase):
        for index, row in enumerate(prepared):
            committed = False
            for duration in (row.action.get('duration', 1.),
                             *row.retry_durations_seconds):
                action = copy.deepcopy(row.action)
                action['duration'] = duration
                entry = {'index': index, 'atActionPosition': len(candidate.actions),
                         'action': copy.deepcopy(action)}
                attempts.append(entry)
                if _inserted_geometry_change(candidate, action):
                    status, reason = 'refused', 'WITHDRAW_BEFORE_GEOMETRY_CHANGE'
                    entry.update(status='refused', reason=reason,
                                 previewRemovedCellIds=[])
                    break
                if len(candidate.actions) >= limits.max_native_actions:
                    status, reason = 'interrupted', 'NATIVE_ACTION_BUDGET'
                    entry.update(status='not_attempted', reason=reason)
                    break
                try:
                    guard_budget()
                    before = candidate.state_id
                    preview = candidate.preview(action, check=check,
                        include_view=False, include_history=False)
                    if candidate.state_id != before:
                        raise ValueError('Staged preview changed accepted native prefix')
                    cells = sorted(set(map(int, preview.get('deltas', {}).get(
                        'removedCells', ()))))
                    entry.update(previewStatus=preview['status'],
                        previewReason=preview.get('reason'),
                        previewStateId=preview['stateId'],
                        previewRemovedCellIds=cells,
                        previewTargetRemovedMm3=preview['metrics']['targetRemovedMm3'],
                        previewNonTargetRemovedMm3=preview['metrics']['nonTargetRemovedMm3'])
                    permitted = (preview['status'] == 'accepted' or
                        preview['status'] == 'partial' and
                        preview.get('reason') == 'TISSUE_ASPIRATION_IN_PROGRESS')
                    if permitted and preview['metrics']['nonTargetRemovedMm3'] > \
                            limits.max_total_non_target_mm3+1e-9:
                        permitted, reason = False, 'TOTAL_NON_TARGET_CAP'
                    if permitted and limits.max_additional_non_target_mm3 is not None and \
                            preview['metrics']['nonTargetRemovedMm3']-limits.baseline_non_target_mm3 > \
                            limits.max_additional_non_target_mm3+1e-9:
                        permitted, reason = False, 'ADDITIONAL_NON_TARGET_CAP'
                    if permitted and any(int(candidate.scene.labels.flat[cell]) not in labels
                                         for cell in cells):
                        permitted, reason = False, 'REMOVAL_LABEL_NOT_ALLOWED'
                    if not permitted:
                        reason = reason or preview.get('reason') or preview['status']
                        entry.update(status='refused', reason=str(reason))
                        if reason == 'SHAFT_COLLISION' and duration != \
                                (row.retry_durations_seconds[-1] if row.retry_durations_seconds
                                 else row.action.get('duration', 1.)):
                            reason = None
                            continue
                        status = 'refused'
                        break
                    guard_budget()
                    outcome = candidate.step(action, check=check)
                    if (outcome['status'] != preview['status'] or
                            outcome['stateId'] != preview['stateId'] or
                            sorted(set(map(int, outcome.get('deltas', {}).get(
                                'removedCells', ())))) != cells):
                        raise ValueError('Staged native preview/commit endpoint differs')
                    entry.update(status='committed', reason=outcome.get('reason'),
                                 committedStateId=outcome['stateId'])
                    committed = True
                    reason = None
                    if row.require_endpoint:
                        endpoint = action.get('pose') or action.get('path', [None])[-1]
                        slot = candidate.state['slots'][action['slot']]
                        if endpoint is None:
                            raise ValueError('Endpoint-checked action requires a native pose or path')
                        position_error = np.linalg.norm(np.subtract(
                            slot['position'], endpoint['position']))
                        angle_error = (Rotation.from_quat(slot['quaternion'])*
                            Rotation.from_quat(endpoint['quaternion']).inv()).magnitude()
                        if position_error > 1e-6 or angle_error > 1e-6:
                            status, reason = 'interrupted', 'PARTIAL_MOTION_REPLAN_REQUIRED'
                            entry.update(endpointReached=False, reason=reason)
                        else:
                            entry['endpointReached'] = True
                    break
                except _INTERRUPTIONS as error:
                    status, reason = 'interrupted', type(error).__name__+': '+str(error)
                    entry.update(status='interrupted', reason=reason,
                                 committedAfterInterrupt=
                                 len(candidate.actions) > entry['atActionPosition'])
                    break
            if status != 'accepted' or not committed:
                break
    ledger = _ledger(candidate, limits, meter)
    receipt = {'schema': SCHEMA, 'status': status, 'reason': reason,
        'phase': phase, 'sourceStateId': start_state,
        'finalStateId': candidate.state_id,
        'retentionRole': ('accepted_native_skill' if status == 'accepted' else
                          'stopped_native_prefix_not_admitted'),
        'admittedSkillProgressMm3': (
            ledger['targetRemovedMm3']-start_ledger['targetRemovedMm3']
            if status == 'accepted' else 0.),
        'sourceNativeActions': start_actions,
        'nativeActionsAppended': len(candidate.actions)-start_actions,
        'attempts': attempts,
        'committedPrefix': {'nativeActions': len(candidate.actions),
            'stateId': candidate.state_id,
            'actionPrefixSeal': semantic_digest(candidate.actions)},
        'ledger': ledger,
        'transitionTrace': copy.deepcopy(meter.trace[start_calls:meter.calls]),
        'physicalCallCost': {'publicCalls': meter.calls-start_calls,
            'previewCalls': sum(row['kind'] == 'preview' for row in
                                meter.trace[start_calls:meter.calls]),
            'directStepCalls': sum(row['kind'] == 'step' for row in
                                   meter.trace[start_calls:meter.calls])}}
    return candidate, receipt


def make_skill_checkpoint(engine, *, source_seal, task_seal, inventory_seal,
                          limits: SkillLimits, meter: TransitionMeter | None = None,
                          stage='unassigned'):
    """Save actual native state and cumulative task/call ledger; no actor data."""
    if not isinstance(limits, SkillLimits) or any(not isinstance(value, str) or
            not value for value in (source_seal, task_seal, inventory_seal, stage)):
        raise ValueError('Staged checkpoint requires bound source/task/inventory/stage')
    if isinstance(engine, MeteredEngine):
        if meter is not None and meter is not engine.meter:
            raise ValueError('Checkpoint meter differs from active native branch')
        meter = engine.meter
    else:
        raise TypeError('Checkpoint requires a metered authoritative engine')
    report = meter.report()
    ledger = _ledger(engine, limits, meter)
    ledger['stage'] = stage
    snapshot = engine.snapshot()
    record = {'schema': CHECKPOINT_SCHEMA,
        'sourceSceneId': engine.scene.identity,
        'sourceSeal': source_seal, 'taskSeal': task_seal,
        'skillInventorySeal': inventory_seal, 'stage': stage,
        'limits': asdict(limits), 'nativeSnapshot': snapshot,
        'actionPrefixSeal': semantic_digest(snapshot['actions']),
        'ledger': ledger, 'meterSegment': report}
    record['checkpointSeal'] = semantic_digest(record)
    return record


def restore_skill_checkpoint(scene, checkpoint, *, source_seal, task_seal,
                             inventory_seal, check=None,
                             meter: TransitionMeter | None = None):
    """Replay every saved action through public metered steps and remeasure.

    Returns ``(engine, continuation_limits, ledger)``. A new meter receives
    only the saved remaining call capacity; an existing meter must match the
    saved segment exactly, so neither path resets consumed calls.
    """
    if not isinstance(checkpoint, dict) or checkpoint.get('schema') != CHECKPOINT_SCHEMA:
        raise ValueError('Unknown staged native checkpoint')
    bare = {key: value for key, value in checkpoint.items() if key != 'checkpointSeal'}
    if checkpoint.get('checkpointSeal') != semantic_digest(bare):
        raise ValueError('Staged checkpoint content seal differs')
    if (checkpoint['sourceSceneId'] != scene.identity or
            checkpoint['sourceSeal'] != source_seal or
            checkpoint['taskSeal'] != task_seal or
            checkpoint['skillInventorySeal'] != inventory_seal):
        raise ValueError('Staged checkpoint source/task/inventory binding differs')
    limits = SkillLimits(**checkpoint['limits'])
    snapshot = checkpoint['nativeSnapshot']
    if (snapshot['sceneId'] != scene.identity or
            snapshot['schema'] != ENGINE_SCHEMA or
            checkpoint['actionPrefixSeal'] != semantic_digest(snapshot['actions']) or
            snapshot['stateId'] != checkpoint['ledger']['stateId'] or
            snapshot['checkpoint']['position'] != len(snapshot['actions']) or
            snapshot['checkpoint']['stateId'] != snapshot['stateId']):
        raise ValueError('Staged native action prefix differs')
    segment = checkpoint['meterSegment']
    saved_spent = checkpoint['ledger']['spentPublicCalls']
    if (segment['physicalCalls'] != segment['previewCalls']+
            segment['directStepCalls'] or
            saved_spent != limits.prior_public_calls+segment['physicalCalls'] or
            checkpoint['ledger']['remainingPublicCalls'] !=
            limits.max_public_calls-saved_spent):
        raise ValueError('Staged checkpoint physical-call ledger differs')
    if meter is None:
        remaining = limits.max_public_calls-saved_spent
        if remaining <= 0:
            raise TransitionBudgetExceeded('No native calls remain for exact resume replay')
        meter = TransitionMeter(remaining, check=check)
        continuation = replace(limits, prior_public_calls=saved_spent)
    elif meter.report() == segment:
        continuation = limits
    elif meter.calls == 0 and meter.max_calls <= limits.max_public_calls-saved_spent:
        continuation = replace(limits, prior_public_calls=saved_spent)
    else:
        raise ValueError('Resume meter would reset or replace prior native calls')
    if len(snapshot['actions']) > continuation.max_public_calls-\
            continuation.prior_public_calls-meter.calls:
        raise TransitionBudgetExceeded('Insufficient native calls for exact resume replay')
    if 'protectionContract' in snapshot:
        from .surgical_rl_protection import restore_protection_contract
        protection, config = restore_protection_contract(
            scene, snapshot['protectionContract'])
        original = SurgicalEngine(scene, protection=protection, guard_config=config)
    else:
        original = SurgicalEngine(scene)
    with meter.phase('staged.resume_full_replay'):
        restored = meter.wrap(original)
        for action in snapshot['actions']:
            if check is not None:
                check()
            if _inserted_geometry_change(restored, action):
                raise ValueError('Saved staged trajectory changes inserted working geometry')
            result = restored.step(action, check=check)
            if result['status'] == 'rejected':
                raise ValueError('Saved staged native action no longer replays')
    if (restored.state_id != snapshot['stateId'] or
            semantic_digest(restored.state) !=
            semantic_digest(snapshot['checkpoint']['state']) or
            restored.actions != snapshot['actions']):
        raise ValueError('Staged native checkpoint differs from full replay')
    current_ledger = _ledger(restored, continuation, meter)
    current_ledger['stage'] = checkpoint['stage']
    for key in ('removedCellCount', 'removedCellSha256', 'targetRemovedMm3',
                'nonTargetRemovedMm3', 'nativeActions'):
        if current_ledger[key] != checkpoint['ledger'][key]:
            raise ValueError('Staged native ledger changed on exact resume')
    return restored, continuation, current_ledger
