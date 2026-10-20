"""Bounded deterministic director over the shared surgical action API.

The director proposes geometry, previews physical substeps and commits the same
public actions. It never modifies labels, occupancy, tissue or effect state.
Its resumable bookkeeping is outside the patient engine identity.
"""
from __future__ import annotations
import copy
import itertools
from pathlib import Path
import time
import numpy as np
from .core import semantic_digest
from .surgical_engine import SurgicalEngine
from .full_case_geometry import access_candidates, next_removal_candidates

SCHEMA = 'resectionlab.full-case-rehearsal.v1'
LIMITATIONS = [
    'Acquired MRI/CT; brain, meningioma, bone and vessels are unreviewed estimates.',
    'Scalp, dura, tumor attachment and complete functional/vascular anatomy UNKNOWN; no such steps are claimed.',
    'Tool geometry, drilling work/heat, deformation, bleeding and fluid are uncalibrated engineering predictions.',
    'Geometric superior access is a simulator task, not a clinical approach recommendation.',
]


def _prefix_phases(engine, count):
    tools = {'left': 'cranial_drill', 'right': 'cusa'}
    phases = []
    for index, action in enumerate(engine.actions[:count]):
        hand = action['slot']
        if action['operation'] == 'exchange': tools[hand] = action['instrument']
        tool = tools[hand]
        role, label = (('bone', 'Recorded initial skull opening') if tool in ('cranial_drill', 'craniotome') else
                       ('target', 'Recorded initial target access and removal') if tool == 'cusa' else
                       ('tools', 'Recorded initial instrument actions'))
        if not phases or phases[-1]['role'] != role:
            phases.append({'id': 'prefix-' + str(len(phases)) + '-' + role, 'role': role,
                           'label': label, 'startPosition': index, 'endPosition': index + 1})
        else: phases[-1]['endPosition'] = index + 1
    return [{k:v for k,v in phase.items() if k != 'role'} for phase in phases]


def accounting(engine):
    labels = engine.scene.labels.ravel()
    removed = np.asarray(engine.state['removed'], dtype=np.int64)
    counts = {r['material']: 0 for r in engine.scene.regions}
    for r in engine.scene.regions:
        counts[r['material']] += int(np.count_nonzero(labels[removed] == r['id'])) if len(removed) else 0
    total = sum(np.count_nonzero(labels == r['id']) for r in engine.scene.regions if r.get('target')) * engine.scene.voxel_mm3
    return {'originalTargetMm3': float(total), 'healthyRemovedMm3': counts.get('soft_tissue', 0) * engine.scene.voxel_mm3,
            'unknownRemovedMm3': counts.get('unknown', 0) * engine.scene.voxel_mm3}


def bind_case_protection(engine):
    """Bind only matching local source evidence; never relax an existing guard."""
    from .surgical_rl_protection import load_protection, GuardConfig
    root = Path(__file__).resolve().parents[2]
    path = root / 'testcases/environments/meningioma-sub-06/rl-protection.json'
    # load_protection verifies the source-grid identity and all supplied masks.
    if not path.exists():
        raise ValueError('Matching source exclusion evidence is unavailable for full rehearsal')
    protection = load_protection(engine.scene, path, strict=True)
    strict = SurgicalEngine(engine.scene, protection=protection, guard_config=GuardConfig())
    if engine._protection_contract == strict._protection_contract:
        return engine
    if engine.actions:
        raise ValueError('Bind matching strict protection at an intact initial checkpoint before full rehearsal')
    return strict


def new_director(engine, scenario='primary', *, full_annotation_target_mm3=None):
    if scenario not in ('primary', 'vascular'):
        raise ValueError('Choose primary or vascular full-case scenario')
    if not any(r.get('target') and r['material'] == 'tumor' and np.any(engine.scene.labels == r['id']) for r in engine.scene.regions):
        raise ValueError('Full rehearsal requires a supplied owned tumor target')
    return {'schema': SCHEMA, 'scenario': scenario, 'status': 'ready', 'sourceSceneId': engine.scene.identity,
            'startStateId': engine.state_id, 'startPosition': len(engine.actions),
            'phases': _prefix_phases(engine, len(engine.actions)), 'currentPhaseId': 'inspect', 'stage': 'inspect',
            'accessIndex': 0, 'accesses': access_candidates(engine, limit=160, max_radius_mm=3.),
            'passIndex': 0, 'passStartTargetMm3': engine.metrics()['targetRemovedMm3'],
            'drillAttempts': 0, 'rejections': [], 'attemptedActions': 0, 'committedActions': 0,
            'wallSeconds': 0., 'limitations': list(LIMITATIONS),
            'fullAnnotationTargetMm3': full_annotation_target_mm3,
            **accounting(engine)}


def resume_director(director):
    result = copy.deepcopy(director)
    if result.get('status') == 'stopped':
        result.setdefault('priorStops', []).append({'position': result.get('lastPosition'), 'reason': result.get('stopReason')})
        result['priorStops'] = result['priorStops'][-8:]
        result['status'] = 'ready'; result['stopReason'] = None
    return result


def configure_directional_fallback(engine, director, *, axis, limit=8):
    """Prepare an evidenced source-axis pass only at a withdrawn residual endpoint.

    This changes detached scheduling metadata only. It cannot move a tool,
    restore tissue, relax a guard or reinterpret a saved superior access.
    """
    if director.get('schema') != SCHEMA or director.get('sourceSceneId') != engine.scene.identity:
        raise ValueError('Directional fallback director source differs')
    if director.get('lastStateId') != engine.state_id:
        raise ValueError('Directional fallback action boundary differs')
    if director.get('scenario') != 'primary' or director.get('status') != 'stopped' or director.get('stage') != 'endpoint':
        raise ValueError('Directional fallback requires a stopped primary residual endpoint')
    if engine.state['holds'] or any(state['inserted'] or state['active'] or state['grip'] for state in engine.state['slots'].values()):
        raise ValueError('Withdraw and release both tools before configuring directional fallback')
    if type(axis) is not int or axis not in (0, 1) or type(limit) is not int or not 1 <= limit <= 32:
        raise ValueError('Directional fallback requires native axis 0/1 and 1..32 accesses')
    if axis in [row['axis'] for row in director.get('directionalPasses', [])]:
        raise ValueError('This bounded directional pass was already configured')
    from .full_case_directional import directional_access_candidates, rank_directional_accesses
    accesses = directional_access_candidates(engine, axis=axis, limit=limit)
    accesses = rank_directional_accesses(engine, accesses)
    accesses = [access for access in accesses if access['remainingGeometricCapacityCells'] > 0]
    if not accesses:
        raise ValueError('No retained target capacity in the bounded directional candidates')
    result = resume_director(director)
    result.setdefault('completedAccessPasses', []).append({
        'axis': result.get('accessAxis', 2), 'accesses': result['accesses'],
        'endPosition': len(engine.actions), 'endStateId': engine.state_id,
        'passIndex': result.get('passIndex', 0), 'stopReason': director.get('stopReason')})
    result.setdefault('directionalPasses', []).append({'axis': axis, 'startPosition': len(engine.actions),
        'startStateId': engine.state_id, 'candidateIds': [access['id'] for access in accesses],
        'qualification': 'Source-column candidates only; every physical action still requires public preview and step.'})
    if 'fieldCleanup' in result:
        result.setdefault('priorFieldCleanups', []).append(result.pop('fieldCleanup'))
    result.update(stage='directional_transition', accesses=accesses, accessAxis=axis, accessIndex=0,
        passIndex=0, passStartTargetMm3=engine.metrics()['targetRemovedMm3'],
        drillAttempts=0, coreSearchExhausted=False, fieldCleanupAttempted=False)
    result['directionalTransition'] = {'poses': _exterior_transition(engine, accesses[0]['outsidePose']),
        'index': 0, 'results': [], 'status': 'prepared',
        'qualification': 'Continuous public withdrawal through exterior half-spaces before changing access direction.'}
    return result


def _same_pose(state, pose):
    return (np.allclose(state['position'], pose['position'], rtol=0., atol=1e-8)
        and (np.allclose(state['quaternion'], pose['quaternion'], rtol=0., atol=1e-8)
             or np.allclose(state['quaternion'], -np.asarray(pose['quaternion']), rtol=0., atol=1e-8)))


def _exterior_transition(engine, outside):
    """Propose exterior transit; exact public sweeps remain the authority."""
    from scipy.spatial.transform import Rotation
    state = engine.state['slots']['left']
    old_q, new_q = state['quaternion'], outside['quaternion']
    old_outward = -Rotation.from_quat(old_q).apply([0., 0., 1.])
    new_outward = -Rotation.from_quat(new_q).apply([0., 0., 1.])
    corners = np.asarray(list(itertools.product(*[(0, n - 1) for n in engine.scene.shape])))
    corners = corners @ engine.scene.affine[:3, :3].T + engine.scene.affine[:3, 3]
    current = np.asarray(state['position'], float)
    old_anchor = current + old_outward * max(0., float((corners @ old_outward).max()) + 60. - current @ old_outward)
    deficit = max(0., float((corners @ new_outward).max()) + 60. - old_anchor @ new_outward)
    transverse = new_outward - old_outward * (old_outward @ new_outward)
    projection = float(transverse @ new_outward)
    if deficit and projection <= 1e-8:
        raise ValueError('Directional transition requires distinct exterior half-spaces')
    corner = old_anchor + transverse * (deficit / projection if deficit else 0.)
    poses = []
    previous = current
    for point, quaternion in ((old_anchor, old_q), (corner, old_q), (corner, new_q),
                              (np.asarray(outside['position'], float), new_q)):
        # Rotation occurs only at the remote corner. Split long translations
        # without changing their line, retaining the public 250 mm segment cap.
        count = max(1, int(np.ceil(np.linalg.norm(point - previous) / 200.)))
        poses.extend({'position': (previous + (point - previous) * (step / count)).tolist(),
                      'quaternion': list(quaternion)} for step in range(1, count + 1))
        previous = point
    return poses


def _phase(meta, name, label, position):
    if not meta['phases'] or meta['phases'][-1]['id'] != name:
        meta['phases'].append({'id': name, 'label': label, 'startPosition': position, 'endPosition': position})
    meta['currentPhaseId'] = name


def _admin(slot, operation, **fields):
    return {'slot': slot, 'operation': operation, 'duration': .2, **fields}


def _batch_frontiers(actions):
    """Join at most two equal-aperture strokes through their outside plane."""
    if not actions: return actions
    first = actions[0]
    if not first.get('path'): return actions
    for second in actions[1:]:
        if not second.get('path') or first.get('parameters') != second.get('parameters'): continue
        if np.linalg.norm(np.subtract(first['path'][-1]['position'][:2], second['path'][-1]['position'][:2])) < .75: continue
        if not np.allclose(first['path'][1]['position'][2], second['path'][1]['position'][2]): continue
        # Retrace the fine entry waypoints. A single coarse lift can collide
        # with a narrow collar even after a legal fine descent.
        reverse = list(reversed(copy.deepcopy(first['path'][1:-1])))
        returned = [first['path'][-1], *reverse]
        return_distance = sum(np.linalg.norm(np.subtract(b['position'], a['position'])) for a,b in zip(returned, returned[1:]))
        path = copy.deepcopy(first['path']) + reverse + copy.deepcopy(second['path'][1:])
        duration = first['duration'] + second['duration'] + return_distance / 20.
        if len(path) > 512 or duration > 30.: continue
        return [{**copy.deepcopy(first), 'path': path, 'duration': duration}, *actions]
    return actions


def _prepare(engine, instrument, outside, parameters, *, slot='left'):
    state = engine.state['slots'][slot]
    if state['active']:
        return _admin(slot, 'deactivate')
    if slot in engine.state['holds'] or state['grip']:
        return _admin(slot, 'release')
    if state['inserted']:
        if state['instrument'] in ('cusa', 'frazier_suction', 'irrigation') and engine.parameters(state)['radiusMm'] > .5:
            return _admin(slot, 'inspect', parameters={'radiusMm': .5})
        from .full_case_withdrawal import withdrawal_candidate
        return withdrawal_candidate(engine, slot=slot) or _admin(slot, 'withdraw')
    if state['instrument'] != instrument:
        return _admin(slot, 'exchange', instrument=instrument)
    if not (np.allclose(state['position'], outside['position']) and np.allclose(state['quaternion'], outside['quaternion']) and
            all(state['parameters'].get(k) == v for k,v in parameters.items())):
        return _admin(slot, 'place', pose=copy.deepcopy(outside), parameters=parameters)
    return None


def _record_rejection(meta, action, outcome, reason=None):
    code = str(reason or outcome.get('reason') or 'NO_ADMISSIBLE_ACTION').split(':')[0]
    counts = meta.setdefault('rejectionReasons', {})
    counts[code] = counts.get(code, 0) + 1
    physical = outcome.get('reason')
    if physical:
        physical_code = str(physical).split(':')[0]
        physical_counts = meta.setdefault('engineRejectionReasons', {})
        physical_counts[physical_code] = physical_counts.get(physical_code, 0) + 1
    meta['rejections'].append({'position': meta.get('lastPosition', 0), 'phase': meta['currentPhaseId'],
        'action': copy.deepcopy(action), 'reason': reason or outcome.get('reason'), 'engineReason': physical,
        'status': outcome.get('status')})
    meta['rejections'] = meta['rejections'][-64:]


def _rank_accesses(engine, accesses, start=0):
    axes = {access.get('nativeAxis', 2) for access in accesses}
    if len(axes) > 1:
        raise ValueError('Access passes must retain one native direction')
    if axes and next(iter(axes)) in (0, 1):
        from .full_case_directional import rank_directional_accesses
        return rank_directional_accesses(engine, accesses, start)
    from .full_case_geometry import rank_remaining_accesses
    return rank_remaining_accesses(engine, accesses, start)


def _frontier_candidates(engine, access):
    if access.get('nativeAxis', 2) in (0, 1):
        from .full_case_directional import directional_removal_candidates
        return directional_removal_candidates(engine, access, limit=6)
    return _batch_frontiers(next_removal_candidates(engine, access, limit=6))


def _advance_access(engine, meta):
    meta['accessIndex'] += 1
    # The original candidate set stays fixed. Prefer retained source target
    # over neighboring columns already cleared by a previous core pocket.
    meta['accesses'] = _rank_accesses(engine, meta['accesses'], meta['accessIndex'])
    meta['stage'] = 'access_setup'


def _finish_field_cleanup(meta, status, reason=None):
    cleanup = meta['fieldCleanup']
    cleanup.update(status=status, reason=reason)
    meta['stage'] = 'cleanup'


def run_slice(engine, director, *, max_actions=12, check=None, progress=None):
    """Return a detached engine + resumable director after bounded real actions.

    A preview is never promoted into committed state. Every admitted action is
    executed through step and must match its preview identity exactly.
    """
    if type(max_actions) is not int or not 1 <= max_actions <= 64:
        raise ValueError('Full rehearsal slice requires 1..64 actions')
    meta = copy.deepcopy(director)
    if meta.get('schema') != SCHEMA or meta.get('sourceSceneId') != engine.scene.identity:
        raise ValueError('Full rehearsal director source differs')
    if meta.get('lastStateId', meta.get('startStateId')) != engine.state_id:
        raise ValueError('Full rehearsal director action boundary differs')
    if meta['status'] in ('complete', 'stopped'):
        return engine, meta
    if meta.get('startPosition', 0) and not any(phase['startPosition'] == 0 for phase in meta['phases']):
        meta['phases'] = _prefix_phases(engine, meta['startPosition']) + meta['phases']
    candidate = engine.clone()
    meta['status'] = 'running'
    started = time.perf_counter()
    commits = 0
    proposals = 0
    while commits < max_actions and proposals < max_actions * 8:
        if check: check()
        proposals += 1
        meta['lastPosition'] = len(candidate.actions)
        stage = meta['stage']
        action = None
        checked_preview = None
        assistant_parking = False
        if stage == 'inspect':
            _phase(meta, 'inspect', 'Inspect supplied anatomy and unknowns', len(candidate.actions))
            active_hand = next((hand for hand,state in candidate.state['slots'].items() if state['active']), None)
            action = _admin(active_hand, 'deactivate') if active_hand else _admin('right', 'inspect')
            next_stage = ('inspect' if active_hand else
                          'vascular_setup' if meta['scenario'] == 'vascular' and candidate.metrics()['targetRemovedMm3'] > 0. else
                          'access_setup')
        elif stage == 'directional_transition':
            from scipy.spatial.transform import Rotation
            _phase(meta, 'directional_transition-'+str(meta['accessAxis']),
                'Checked exterior transit to the next access direction', len(candidate.actions))
            transition = meta['directionalTransition']
            if transition['index'] >= len(transition['poses']):
                transition['status'] = 'complete'; meta['stage'] = 'access_setup'; continue
            pose = transition['poses'][transition['index']]
            state = candidate.state['slots']['left']
            distance = float(np.linalg.norm(np.subtract(pose['position'], state['position'])))
            angle = (Rotation.from_quat(pose['quaternion']) * Rotation.from_quat(state['quaternion']).inv()).magnitude()
            action = _admin('left', 'withdraw', pose=copy.deepcopy(pose), duration=max(.2, (distance + 40. * angle) / 20.))
            next_stage = stage
        elif stage in ('access_setup', 'drill', 'cusa_setup', 'remove'):
            if meta['accessIndex'] >= len(meta['accesses']):
                # Neighboring openings can join into a wider real corridor.
                # Revisit it only while the previous bounded pass progressed.
                removed = candidate.metrics()['targetRemovedMm3']
                if meta['scenario'] == 'primary' and meta.get('passIndex', 0) < 2 and removed > meta.get('passStartTargetMm3', 0.):
                    meta['passIndex'] = meta.get('passIndex', 0) + 1
                    meta['passStartTargetMm3'] = removed
                    meta['accesses'] = _rank_accesses(candidate, meta['accesses'])
                    meta['accessIndex'] = 0; meta['stage'] = 'access_setup'; continue
                meta['stage'] = 'cleanup'; continue
            access = meta['accesses'][meta['accessIndex']]
            aid = access['id']
            if stage == 'access_setup':
                _phase(meta, 'access-'+aid, 'Modeled skull opening · '+aid, len(candidate.actions))
                action = _prepare(candidate, 'cranial_drill', access['outsidePose'],
                    {'radiusMm': access['drillRadiusMm'], 'forceN': 20., 'rpm': 80000., 'duty': 1.})
                if action is None:
                    meta['stage'] = 'drill'; meta['drillAttempts'] = 0
                    meta['cycleBoneRemovedCells'] = 0
                    meta['coreSearchExhausted'] = False
                    continue
                next_stage = stage
            elif stage == 'drill':
                action = {'slot': 'left', 'operation': 'apply', 'duration': 10., 'pose': copy.deepcopy(access['entryPose'])}
                next_stage = stage
            elif stage == 'cusa_setup':
                _phase(meta, 'removal-'+aid, 'Target debulking from checked frontier · '+aid, len(candidate.actions))
                action = _prepare(candidate, 'cusa', access['outsidePose'], {'radiusMm': min(6., max(.6, access['drillRadiusMm'] - 1.7)), 'forceN': 0.})
                if action is None:
                    meta['stage'] = 'remove'
                    meta['cycleStartTargetMm3'] = candidate.metrics()['targetRemovedMm3']
                    continue
                next_stage = stage
            else:
                actions = _frontier_candidates(candidate, access)
                assistant_conflict = False
                for proposed in actions:
                    if check: check()
                    preview = candidate.preview(proposed, check=check, include_view=False)
                    meta['attemptedActions'] += 1
                    if str(preview.get('reason')).startswith('TOOL_COLLISION:right') and candidate.state['slots']['right']['inserted'] and not preview.get('deltas', {}).get('removedCells'):
                        _record_rejection(meta, proposed, preview)
                        meta['assistantResumeStage'] = 'remove'; meta['stage'] = 'assistant_retract'; assistant_conflict = True; break
                    if preview['status'] == 'rejected':
                        _record_rejection(meta, proposed, preview)
                        if str(preview.get('reason')).startswith('TOOL_COLLISION:right') and candidate.state['slots']['right']['inserted']:
                            meta['assistantResumeStage'] = 'remove'; meta['stage'] = 'assistant_retract'; assistant_conflict = True; break
                        continue
                    delta = preview.get('deltas', {}).get('removedCells', [])
                    target_labels = {r['id'] for r in candidate.scene.regions if r.get('target') and r['material'] == 'tumor'}
                    if not delta or any(int(candidate.scene.labels.ravel()[i]) not in target_labels for i in delta):
                        _record_rejection(meta, proposed, preview, 'NO_TARGET_PROGRESS' if not delta else 'HEALTHY_OR_UNKNOWN_REMOVAL_REFUSED'); continue
                    action = proposed; checked_preview = preview; break
                if action is None:
                    if assistant_conflict: continue
                    if candidate.metrics()['targetRemovedMm3'] > meta.get('cycleStartTargetMm3', candidate.metrics()['targetRemovedMm3']):
                        # A broad burr can stop at a sloped target margin while
                        # bone remains above neighboring target. Clear the
                        # exposed target first, then deepen the SAME opening.
                        # Once both bounded searches exhaust without new bone
                        # access, defer this opening to a later progressing
                        # pass. Repeating identical dry drilling can otherwise
                        # delay fresh target-bearing routes for tiny gains.
                        # A capped productive core is not search exhaustion;
                        # old saved metadata first gets a measured drill cycle.
                        if meta.get('cycleBoneRemovedCells', 1) or meta.get('coreSearchExhausted') is not True:
                            meta['stage'] = 'access_setup'
                        else:
                            _advance_access(candidate, meta)
                        continue
                    _advance_access(candidate, meta); continue
                next_stage = stage
        elif stage == 'core':
            from .full_case_debulking import local_debulking_candidates
            _phase(meta, 'core-'+str(meta['accessIndex']), 'Controlled debulking within the checked target passage', len(candidate.actions))
            # Retain a productive checked passage through a bounded run of
            # local effects. Returning to access after only six effects caused
            # repeated withdrawal/exchange/drill cycles despite an open pocket.
            if meta.get('coreAttempts', 0) >= 32:
                meta['coreSearchExhausted'] = False
                meta['stage'] = 'remove'; continue
            for proposal in local_debulking_candidates(candidate, limit=4, radius_mm=6.):
                if proposal['kind'] == 'core-positioning' and meta.get('corePositioning', 0) >= 2: continue
                preview = candidate.preview(proposal['action'], check=check, include_view=False)
                meta['attemptedActions'] += 1
                if preview['status'] == 'rejected':
                    _record_rejection(meta, proposal['action'], preview); continue
                if not preview.get('deltas', {}).get('removedCells') and proposal['kind'] != 'core-positioning': continue
                action = proposal['action']; checked_preview = preview
                meta['corePositioning'] = meta.get('corePositioning', 0) + (proposal['kind'] == 'core-positioning')
                break
            if action is None:
                meta['coreSearchExhausted'] = True
                meta['stage'] = 'remove'; continue
            next_stage = stage
        elif stage == 'assistant_setup':
            _phase(meta, 'assistant', 'Second-hand aspiration of the opened field', len(candidate.actions))
            field = meta['assistantField']
            if meta.get('assistantStep', 0) == 0:
                action = _prepare(candidate, 'frazier_suction', field['outsidePose'], field['parameters'], slot='right')
                if action is None:
                    meta['assistantStep'] = 1; continue
            elif meta['assistantStep'] == 1:
                action = _admin('right', 'apply', pose=field['fieldPose'])
            else:
                action = _admin('right', 'activate')
            next_stage = stage
        elif stage == 'assistant_retract':
            _phase(meta, 'assistant_retract', 'Withdraw the assisting hand before conflicting access', len(candidate.actions))
            state = candidate.state['slots']['right']
            if state['active']: action = _admin('right', 'deactivate', parameters={'radiusMm': .5})
            elif state['inserted'] and candidate.parameters(state)['radiusMm'] > .5:
                action = _admin('right', 'inspect', parameters={'radiusMm': .5})
            elif state['inserted']:
                from .full_case_withdrawal import withdrawal_candidate
                action = withdrawal_candidate(candidate, slot='right') or _admin('right', 'withdraw')
            elif not meta.get('assistantParked'):
                from scipy.spatial.transform import Rotation
                outward = -Rotation.from_quat(state['quaternion']).apply([0., 0., 1.])
                anchor = meta.get('assistantField', {}).get('outsidePose', {}).get('position', state['position'])
                # Both generic instruments have a 40 mm shaft. Clear the
                # complete outside transit envelope as well as the tip plane.
                parked = (np.asarray(anchor) + outward * 100.).tolist()
                action = _admin('right', 'withdraw', pose={'position': parked, 'quaternion': state['quaternion']}, duration=2.5)
                assistant_parking = True
            else:
                meta['stage'] = meta.get('assistantResumeStage', 'remove'); continue
            next_stage = stage
        elif stage == 'vascular_setup':
            _phase(meta, 'vascular_setup', 'Withdraw from the saved common target-removal prefix', len(candidate.actions))
            for hand, state in candidate.state['slots'].items():
                if state['active']: action = _admin(hand, 'deactivate'); break
                if hand in candidate.state['holds'] or state['grip']: action = _admin(hand, 'release'); break
                if state['inserted'] and state['instrument'] in ('cusa', 'frazier_suction', 'irrigation') and candidate.parameters(state)['radiusMm'] > .5:
                    action = _admin(hand, 'inspect', parameters={'radiusMm': .5}); break
                if state['inserted']:
                    from .full_case_withdrawal import withdrawal_candidate
                    action = withdrawal_candidate(candidate, slot=hand) or _admin(hand, 'withdraw'); break
            if action is None:
                meta['stage'] = 'vascular'; continue
            next_stage = stage
        elif stage == 'vascular':
            from .full_case_vascular import rescue_plan
            if 'vascularPlan' not in meta:
                meta['vascularPlan'] = rescue_plan(candidate)
                meta['vascularIndex'] = 0
            plan = meta['vascularPlan']
            if meta['vascularIndex'] >= len(plan['actions']):
                meta['stage'] = 'cleanup'; continue
            row = plan['actions'][meta['vascularIndex']]
            # Plan rows retain action and a declared mechanism phase.
            if 'action' in row:
                action = row['action']; phase = row.get('phase', 'vascular')
                label = row.get('label', phase.replace('_', ' '))
            else:
                action = row; phase = 'vascular'; label = 'Vascular injury and rescue'
                offset = 0
                for planned_phase in plan['phases']:
                    offset += len(planned_phase['actions'])
                    if meta['vascularIndex'] < offset:
                        phase = planned_phase['id']; label = planned_phase['label']; break
            _phase(meta, phase, label, len(candidate.actions))
            next_stage = stage
        elif stage == 'cleanup':
            _phase(meta, 'cleanup', 'Field inspection and tool withdrawal', len(candidate.actions))
            for hand, state in candidate.state['slots'].items():
                if state['active']: action = _admin(hand, 'deactivate'); break
                if hand in candidate.state['holds'] or state['grip']: action = _admin(hand, 'release'); break
                if state['inserted'] and state['instrument'] in ('cusa', 'frazier_suction', 'irrigation') and candidate.parameters(state)['radiusMm'] > .5:
                    action = _admin(hand, 'inspect', parameters={'radiusMm': .5}); break
                if state['inserted']:
                    from .full_case_withdrawal import withdrawal_candidate
                    action = withdrawal_candidate(candidate, slot=hand) or _admin(hand, 'withdraw'); break
            if action is None:
                if meta['scenario'] == 'primary' and not meta.get('fieldCleanupAttempted'):
                    from .full_case_cleanup import cleanup_stations
                    meta['fieldCleanupAttempted'] = True
                    field = meta.get('assistantField')
                    stations = cleanup_stations(candidate, field) if field else []
                    meta['fieldCleanup'] = {'stations': stations, 'index': 0, 'mode': 'setup',
                        'results': [], 'status': 'prepared' if stations else 'no-accessible-field',
                        'qualification': 'Bounded aspiration along the established field ray; residual field material is retained.'}
                    if stations:
                        meta['stage'] = 'field_cleanup'; continue
                action = _admin('right', 'inspect'); next_stage = 'endpoint'
            else: next_stage = stage
        elif stage == 'field_cleanup':
            from .full_case_cleanup import cleanup_dwell_action
            _phase(meta, 'field_cleanup', 'Aspiration of accessible field material', len(candidate.actions))
            cleanup = meta['fieldCleanup']
            if cleanup['index'] >= len(cleanup['stations']):
                _finish_field_cleanup(meta, 'bounded-pass-finished'); continue
            station = cleanup['stations'][cleanup['index']]
            if cleanup['mode'] == 'setup':
                action = _prepare(candidate, 'frazier_suction', station['outsidePose'], station['parameters'], slot='right')
                if action is None:
                    cleanup['mode'] = 'approach'; continue
            elif cleanup['mode'] == 'approach':
                distance = float(np.linalg.norm(np.subtract(station['pose']['position'], candidate.state['slots']['right']['position'])))
                action = _admin('right', 'apply', pose=copy.deepcopy(station['pose']),
                    parameters=copy.deepcopy(station['parameters']), duration=max(.2, min(30., distance / 10.)))
            elif cleanup['mode'] == 'dwell':
                measured = cleanup_dwell_action(candidate, station)
                if measured is None:
                    cleanup['results'].append({'stationIndex': cleanup['index'], 'mode': 'dwell',
                        'position': len(candidate.actions), 'status': 'empty-field-at-actual-pose'})
                    cleanup['index'] += 1; cleanup['mode'] = 'approach'; continue
                action = measured['action']
                cleanup['freshMeasurement'] = {key: value for key, value in measured.items() if key != 'action'}
            else:
                raise ValueError('Unknown bounded field cleanup mode')
            next_stage = stage
        elif stage == 'endpoint':
            metrics = candidate.metrics()
            remaining = accounting(candidate)['originalTargetMm3'] - metrics['targetRemovedMm3']
            if meta['scenario'] == 'vascular':
                sources = [source for source in candidate.state['sources'] if source.get('kind') == 'modeled-vessel-wound']
                faces = candidate.state.get('vesselsV2', {}).get('faces', {})
                dose = candidate.scene.properties.get('tissuePhysicsV2', {}).get('vessel', {}).get('coagulationDose', float('inf'))
                controlled = (bool(sources) and bool(faces) and any(source.get('emittedMm3', 0.) > 0. for source in sources)
                              and all(source.get('bleedingActive') is False and source.get('patent') is False for source in sources)
                              and all(face.get('patent') is False and face.get('coagulationDose', 0.) + 1e-9 >= dose
                                      and face.get('clot', 0.) >= 1. - 1e-9 for face in faces.values()))
                meta['status'] = 'complete' if controlled else 'stopped'
                meta['stopReason'] = ('Modeled vessel injury and source control completed; severed faces remain nonpatent. '
                    'This alternative retains the common-prefix target residual; it does not claim target clearance.' if controlled else
                    'Vascular alternative did not establish a bleeding wound followed by persistent source control; inspect the recorded state.')
            else:
                meta['status'] = 'complete' if remaining <= 1e-9 else 'stopped'
                direction = {0: 'left', 1: 'anterior'}.get(meta.get('accessAxis'), 'superior')
                meta['stopReason'] = ('Complete modeled owned-target removal.' if remaining <= 1e-9 else
                    'Configured '+direction+' access/frontier candidates and bounded attempts are exhausted under full tool/shaft collisions, target-only removal and supplied protection. '
                    'Further access strategies remain untested; residual target is retained. Missing dura/attachment/functional evidence remains UNKNOWN.')
            _phase(meta, 'endpoint', 'Endpoint and explained residual', len(candidate.actions))
            break
        else:
            raise ValueError('Unknown full rehearsal director stage')
        preview = checked_preview if checked_preview is not None else candidate.preview(action, check=check, include_view=False)
        if checked_preview is None: meta['attemptedActions'] += 1
        if stage == 'directional_transition' and preview['status'] == 'rejected':
            _record_rejection(meta, action, preview)
            transition = meta['directionalTransition']
            transition['status'] = 'blocked'
            transition['results'].append({'index': transition['index'], 'position': len(candidate.actions),
                'status': 'rejected', 'reason': preview.get('reason'), 'arrived': False})
            meta.update(status='stopped', stopReason='Checked exterior directional transition blocked: '+str(preview.get('reason')))
            break
        if stage == 'field_cleanup' and preview['status'] == 'rejected':
            _record_rejection(meta, action, preview)
            meta['fieldCleanup']['results'].append({'stationIndex': meta['fieldCleanup']['index'],
                'mode': meta['fieldCleanup']['mode'], 'position': len(candidate.actions),
                'status': 'rejected', 'reason': preview.get('reason')})
            _finish_field_cleanup(meta, 'blocked', preview.get('reason')); continue
        if preview['status'] != 'rejected' and str(preview.get('reason')).startswith('TOOL_COLLISION:right') and candidate.state['slots']['right']['inserted'] and stage not in ('assistant_retract', 'field_cleanup') and not preview.get('deltas', {}).get('removedCells'):
            _record_rejection(meta, action, preview)
            meta['assistantResumeStage'] = stage; meta['stage'] = 'assistant_retract'; continue
        healthy_labels = {r['id'] for r in candidate.scene.regions if r['material'] in ('soft_tissue', 'unknown', 'functional_region', 'membrane') and not r.get('target')}
        if preview['status'] != 'rejected' and any(int(candidate.scene.labels.ravel()[i]) in healthy_labels for i in preview.get('deltas', {}).get('removedCells', [])):
            _record_rejection(meta, action, preview, 'HEALTHY_OR_UNKNOWN_REMOVAL_REFUSED')
            meta.update(status='stopped', stopReason='Persistent instrument exposure would remove healthy or unknown tissue; checked action refused.')
            break
        if preview['status'] == 'rejected':
            _record_rejection(meta, action, preview)
            if str(preview.get('reason')).startswith('TOOL_COLLISION:right') and candidate.state['slots']['right']['inserted'] and stage != 'assistant_retract':
                meta['assistantResumeStage'] = stage; meta['stage'] = 'assistant_retract'; continue
            if stage == 'assistant_setup':
                meta['stage'] = 'core'; continue
            if stage == 'drill':
                meta['stage'] = 'cusa_setup'; continue
            if stage in ('access_setup', 'cusa_setup'):
                # A failed withdrawal cannot authorize a placement elsewhere.
                if action['operation'] in ('withdraw', 'release', 'deactivate'):
                    meta.update(status='stopped', stopReason='Tool cleanup blocked: '+str(preview.get('reason'))); break
                _advance_access(candidate, meta); continue
            meta.update(status='stopped', stopReason='Required action blocked: '+str(preview.get('reason'))); break
        before_removed = set(candidate.state['removed'])
        before_aspirated = {key: float(candidate.state.get(key, 0.)) for key in ('fluidRemovedMm3', 'debrisRemovedMm3')}
        outcome = candidate.step(action, check=check)
        if outcome['stateId'] != preview['stateId']:
            raise ValueError('Full rehearsal preview and committed transition differ')
        commits += 1; meta['committedActions'] += 1
        meta['stage'] = next_stage
        if stage == 'directional_transition':
            transition = meta['directionalTransition']
            arrived = outcome['status'] == 'accepted' and _same_pose(candidate.state['slots']['left'], action['pose'])
            transition['results'].append({'index': transition['index'], 'position': len(candidate.actions),
                'status': outcome['status'], 'reason': outcome.get('reason'), 'arrived': arrived})
            if arrived:
                transition['index'] += 1
            else:
                transition['status'] = 'blocked'
                meta.update(status='stopped', stopReason='Checked exterior directional transition blocked: '+str(outcome.get('reason')))
        if stage == 'field_cleanup':
            cleanup = meta['fieldCleanup']
            mode = cleanup['mode']
            deltas = {key: float(candidate.state.get(key, 0.)) - before_aspirated[key] for key in before_aspirated}
            cleanup['results'].append({'stationIndex': cleanup['index'], 'mode': mode,
                'position': len(candidate.actions), 'status': outcome['status'], 'reason': outcome.get('reason'),
                **deltas, 'retainedDebrisMm3': candidate.state['debrisMm3'],
                'retainedFluidMm3': candidate.metrics()['fluidPresentMm3'],
                **({'freshMeasurement': cleanup.pop('freshMeasurement')} if 'freshMeasurement' in cleanup else {})})
            if outcome['status'] != 'accepted':
                _finish_field_cleanup(meta, 'blocked', outcome.get('reason'))
            elif mode == 'approach':
                state, pose = candidate.state['slots']['right'], cleanup['stations'][cleanup['index']]['pose']
                arrived = state['inserted'] and _same_pose(state, pose)
                if arrived: cleanup['mode'] = 'dwell'
                else: _finish_field_cleanup(meta, 'blocked', 'Checked approach did not reach the proposed station')
            elif mode == 'dwell':
                if sum(deltas.values()) <= 1e-9:
                    _finish_field_cleanup(meta, 'no-aspiration-progress')
                else:
                    cleanup['index'] += 1; cleanup['mode'] = 'approach'
        if stage == 'drill':
            if 'cycleBoneRemovedCells' in meta:
                bone_labels = {r['id'] for r in candidate.scene.regions if r['material'] == 'bone'}
                labels = candidate.scene.labels.ravel()
                meta['cycleBoneRemovedCells'] += sum(
                    int(labels[cell]) in bone_labels for cell in outcome.get('deltas', {}).get('removedCells', []))
            meta['drillAttempts'] += 1
            if outcome.get('reason') != 'BONE_CUTTING_IN_PROGRESS' or meta['drillAttempts'] >= 30:
                meta['stage'] = 'cusa_setup'
        if stage == 'vascular': meta['vascularIndex'] += 1
        if stage == 'core':
            meta['coreAttempts'] += 1
            meta['coreSearchExhausted'] = False
        if stage == 'assistant_setup':
            if meta['assistantStep'] == 1 and action['operation'] == 'apply': meta['assistantStep'] = 2
            elif action['operation'] == 'activate': meta['stage'] = 'core'; meta['assistantParked'] = False
        if assistant_parking and outcome['status'] == 'accepted' and np.allclose(candidate.state['slots']['right']['position'], action['pose']['position']):
            meta['assistantParked'] = True
        if stage == 'remove' and meta['scenario'] == 'vascular':
            meta['stage'] = 'vascular_setup'
        elif stage == 'remove' and meta['scenario'] == 'primary':
            meta['stage'] = 'core'; meta['coreAttempts'] = 0; meta['corePositioning'] = 0
            meta['coreSearchExhausted'] = False
            if not meta.get('assistantAttempted') and meta['accesses'][meta['accessIndex']].get('nativeAxis', 2) == 2:
                from .full_case_assist import assistant_field_candidate
                field = assistant_field_candidate(candidate, meta['accesses'][meta['accessIndex']])
                if field is not None:
                    meta['assistantAttempted'] = True; meta['assistantField'] = field
                    meta['assistantStep'] = 0; meta['stage'] = 'assistant_setup'
        if stage == 'remove' and 'commonPrefixPosition' not in meta:
            meta['commonPrefixPosition'] = len(candidate.actions)
        if str(outcome.get('reason')).startswith('TOOL_COLLISION:right') and stage not in ('assistant_retract', 'field_cleanup', 'directional_transition'):
            meta['assistantResumeStage'] = meta['stage']; meta['stage'] = 'assistant_retract'
        meta['phases'][-1]['endPosition'] = len(candidate.actions)
        meta['lastOutcome'] = {'position': len(candidate.actions), 'status': outcome['status'], 'reason': outcome.get('reason'),
                               'operation': action['operation'], 'removedCells': len(set(candidate.state['removed']) - before_removed)}
        if progress: progress(commits / max_actions, meta['phases'][-1]['label'])
        if meta['status'] == 'stopped': break
        if stage not in ('cleanup', 'field_cleanup', 'endpoint') and candidate.metrics()['targetRemovedMm3'] >= accounting(candidate)['originalTargetMm3'] and meta['scenario'] == 'primary':
            meta['stage'] = 'cleanup'
    meta['wallSeconds'] += time.perf_counter() - started
    meta.update(accounting(candidate))
    meta['lastStateId'] = candidate.state_id
    return candidate, meta
