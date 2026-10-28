"""Checked bimanual retraction skills using the existing native V2 mechanics.

The catalog blade never shrinks. Contact acquisition uses one integration step
per hold command, followed by separate held moves. These skills do not supply
new anatomy, mechanical parameters, a learned actor, or clinical injury limits.
"""
from __future__ import annotations

import copy
from dataclasses import replace
import math

import numpy as np
from scipy.spatial.transform import Rotation

from .core import semantic_digest
from .surgical_motion import envelopes, pair_conflict
from .surgical_staged_skills import SkillAction, checked_actions
from .surgical_tools import TOOLS

SCHEMA = 'resectionlab.checked-retraction-skill.v1'


def _outside_image_coverage(engine, pose):
    inside, total = 0, 0
    for a,b,_,_ in envelopes(TOOLS['brain_retractor'],{'radiusMm':3.,'depthMm':2.},
                             pose['position'],pose['quaternion']):
        for fraction in np.linspace(0.,1.,max(2,math.ceil(np.linalg.norm(b-a)/.5)+1)):
            point = a+(b-a)*fraction
            index = engine.scene.inverse[:3,:3]@(point-engine.scene.affine[:3,3])
            inside += bool(np.all(index>=-.5) and np.all(index<=np.asarray(engine.scene.shape)-.5))
            total += 1
    return {'sampleCount':total,'insideImageSamples':inside,'insideImageFraction':inside/total,
            'scope':'capsule centerline samples only; outside-image occupancy remains unknown'}


def shaft_clearance(engine):
    """Read current deformed occupancy; report each physical shaft's contacts."""
    contacts, parts = {}, {}
    for hand, slot in engine.state['slots'].items():
        parts[hand] = envelopes(TOOLS[slot['instrument']], engine.parameters(slot),
                               slot['position'], slot['quaternion'])
        cells = set()
        for a, b, radius, part in parts[hand]:
            if part == 'shaft':
                cells.update(map(int, engine.touched(a, b, radius)))
        cells.difference_update(engine.state['holds'].get(hand, {}).get('cells', ()))
        contacts[hand] = sorted(cells)
    conflict = pair_conflict(parts['left'], parts['right'])
    return {'clear': not conflict and not any(contacts.values()),
        'shaftSourceCellIds': contacts, 'toolPairConflict': conflict,
        'scope': 'current full shaft and tool-pair geometry in deformed source occupancy'}


def hold_binding(engine, hand):
    hold = engine.state['holds'].get(hand)
    if (not isinstance(hold, dict) or not hold.get('cells') or
            'target' not in hold or 'baseDisplacements' not in hold):
        raise ValueError('A nonempty coupled V2 retractor hold is required')
    payload = {'sceneId': engine.scene.identity, 'hand': hand,
        'cells': sorted(map(int, hold['cells'])), 'origin': copy.deepcopy(hold['origin']),
        'baseDisplacements': copy.deepcopy(hold['baseDisplacements'])}
    return {'binding': semantic_digest(payload), 'heldCells': payload['cells'],
            'origin': payload['origin']}


def checked_pair_action(engine, action, *, retractor_hand, limits, meter, check,
                        expected_hold=None, phase='retraction.action'):
    """Append one native action while preserving the other hand's held cells."""
    action = copy.deepcopy(action)
    other_hand = 'right' if retractor_hand == 'left' else 'left'
    if retractor_hand not in ('left', 'right'):
        raise ValueError('Unknown retractor hand')
    if (action['slot'] == retractor_hand and
            engine.state['slots'][other_hand]['active']):
        raise ValueError('Deactivate the other tool before a retractor operation')
    if expected_hold is not None and hold_binding(engine, retractor_hand) != expected_hold:
        raise ValueError('Retraction hold changed before the next native action')
    parked = copy.deepcopy(engine.state['slots'][retractor_hand])
    forbidden = set(limits.forbidden_removed_cell_ids)
    if action['slot'] != retractor_hand:
        forbidden.update(engine.state['holds'].get(retractor_hand, {}).get('cells', ()))
    local_limits = replace(limits, forbidden_removed_cell_ids=tuple(sorted(forbidden)))
    moving = 'pose' in action or 'path' in action
    branch, receipt = checked_actions(engine,
        [SkillAction(action, require_endpoint=moving)], limits=local_limits,
        meter=meter, check=check, phase=phase)
    if receipt['status'] == 'accepted':
        audit = shaft_clearance(branch)
        receipt['postActionClearance'] = audit
        reason = None
        if not audit['clear']:
            reason = 'POST_ACTION_BIMANUAL_CLEARANCE_REFUSED'
        if expected_hold is not None:
            try:
                if hold_binding(branch, retractor_hand) != expected_hold:
                    reason = 'RETRACTION_HOLD_SOURCE_CHANGED'
            except ValueError:
                reason = 'RETRACTION_HOLD_SOURCE_CHANGED'
        if action['slot'] != retractor_hand and branch.state['slots'][retractor_hand] != parked:
            reason = 'PARKED_RETRACTOR_CHANGED'
        if reason:
            receipt.update(status='refused', reason=reason,
                retentionRole='stopped_native_prefix_not_admitted', admittedSkillProgressMm3=0.)
    receipt['retractionMetrics'] = {key: value for key, value in branch.metrics().items()
        if key.startswith('brain') or key in ('contactCellSeconds','thermalExposure',
            'fluidPresentMm3','targetRemovedMm3','nonTargetRemovedMm3','elapsedSeconds')}
    return branch, receipt


def acquire_retractor(engine, proposal, *, hand='right', limits, meter, check):
    """Native outside placement, free approach, then first single-step contact."""
    if 'tissuePhysicsV2' not in engine.scene.properties or engine.state['holds']:
        raise ValueError('Retraction acquisition requires V2 mechanics and no existing hold')
    if proposal.get('sourceSceneId') != engine.scene.identity:
        raise ValueError('Retraction contact proposal belongs to another source')
    slot = engine.state['slots'][hand]
    if slot['inserted'] or slot['active'] or slot['grip']:
        raise ValueError('Withdraw the retractor hand before a new acquisition')
    proposal_used = copy.deepcopy(proposal)
    outside = np.asarray(proposal['outsidePose']['position'],float)
    contact = np.asarray(proposal['contactPose']['position'],float)
    direction = contact-outside
    length = float(np.linalg.norm(direction))
    if not 0 < length <= 20.:
        raise ValueError('Retraction free approach exceeds its bounded source proposal')
    direction /= length
    q = list(proposal['contactPose']['quaternion'])
    if (Rotation.from_quat(q)*Rotation.from_quat(proposal['outsidePose']['quaternion']).inv()).magnitude() > 1e-8:
        raise ValueError('Retraction acquisition must keep the declared blade roll')
    source = engine.clone()
    branch = source
    receipts = []
    source_actions = len(source.actions)
    def run(action, phase):
        nonlocal branch
        candidate, receipt = checked_pair_action(branch, action, retractor_hand=hand,
            limits=limits, meter=meter, check=check, phase=phase)
        receipts.append(receipt)
        if receipt['status'] == 'accepted':
            branch = candidate
            return True
        return False
    status, reason, binding = 'refused', None, None
    placed = False
    if run({'slot':hand,'operation':'exchange','instrument':'brain_retractor','duration':.2},
           'retraction.setup'):
        # The native placement guard uses a full working-radius capsule. If
        # that refuses the source proposal, try only two bounded outward
        # placements with identical geometry and orientation.
        for extra in (0.,4.,8.):
            if length+extra > 20.:
                continue
            candidate_outside = outside-direction*extra
            pose = {'position':candidate_outside.tolist(),'quaternion':q}
            if run({'slot':hand,'operation':'place','pose':pose,
                    'parameters':{'radiusMm':3.,'depthMm':2.,'forceN':.08},'duration':.2},
                   'retraction.setup' if extra==0 else 'retraction.outward_placement_recovery'):
                outside = candidate_outside
                length += extra
                proposal_used['outsidePose'] = copy.deepcopy(pose)
                proposal_used['outsideRecoveryOffsetMm'] = extra
                proposal_used['outsideCapsuleCenterlineImageCoverage'] = _outside_image_coverage(engine,pose)
                placed = True
                break
            reason = receipts[-1]['reason']
            if not str(reason).startswith('PLACE_OUTSIDE_MATERIAL:'):
                break
    else:
        reason = receipts[-1]['reason']
    if placed:
        reason = None
        reached_boundary = False
        for offset in np.arange(.5,length+1e-9,.5):
            pose = {'position':(outside+direction*float(offset)).tolist(),'quaternion':q}
            if not run({'slot':hand,'operation':'move','pose':pose,'duration':.25},
                       'retraction.free_approach'):
                reason = receipts[-1]['reason']
                reached_boundary = reason == 'RETAINED_TISSUE_COLLISION'
                break
        if reached_boundary:
            step = min(.25,float(engine.scene.spacing.min())/6.)
            free = np.asarray(branch.state['slots'][hand]['position'],float)
            for index in range(1,math.ceil(1./step)+1):
                # Geometry and duration each require exactly one native tool
                # integration step, so contact never turns into an implicit drag.
                pose = {'position':(free+direction*step*index).tolist(),'quaternion':q}
                if not run({'slot':hand,'operation':'hold','pose':pose,'duration':.05},
                           'retraction.single_step_contact'):
                    reason = receipts[-1]['reason']; break
                if hand not in branch.state['holds']:
                    continue
                binding = hold_binding(branch, hand)
                brain_ids = {int(row['id']) for row in branch.scene.regions
                    if row['material']=='soft_tissue' and 'brain' in row['name'].lower()}
                if any(int(branch.scene.labels.flat[cell]) not in brain_ids for cell in binding['heldCells']):
                    reason = 'RETRACTION_CONTACT_IS_NOT_MAPPED_BRAIN'; break
                origin = branch.state['holds'][hand]['origin']
                target = branch.state['holds'][hand]['target']
                if (np.linalg.norm(np.subtract(origin,pose['position'])) > 1e-6 or
                        np.linalg.norm(np.subtract(target,origin)) > 1e-6 or
                        not any(event.get('kind')=='coupled_hold_created'
                                for event in branch.history[-1]['events'])):
                    reason = 'RETRACTION_CONTACT_WAS_NOT_A_SINGLE_COUPLED_HOLD'; break
                status, reason = 'accepted', None
                break
            else:
                reason = 'NO_NATIVE_CONTACT_WITHIN_SINGLE_STEP_ACQUISITION_BOUND'
        elif reason is None:
            reason = 'NO_NATIVE_CONTACT_BOUNDARY_IN_SOURCE_PROPOSAL'
    return branch, {'schema':SCHEMA,'status':status,'reason':reason,
        'sourceStateId':source.state_id,'finalStateId':branch.state_id,
        'sourceNativeActions':source_actions,'nativeActions':len(branch.actions),
        'binding':binding,'proposalUsed':proposal_used,'attempts':receipts,
        'contactSearchMarginMm':1.,
        'observedContactPose':({'position':copy.deepcopy(branch.state['slots'][hand]['position']),
            'quaternion':copy.deepcopy(branch.state['slots'][hand]['quaternion'])} if binding else None),
        'model':'existing uncalibrated V2 coupled patch; no clinical injury inference'}


def held_move(engine, position, *, hand='right', binding, limits, meter, check):
    slot = engine.state['slots'][hand]
    if slot['instrument']!='brain_retractor' or not slot['inserted']:
        raise ValueError('Held movement requires an inserted native brain retractor')
    distance = float(np.linalg.norm(np.subtract(position,slot['position'])))
    if not 0 < distance <= .5+1e-9:
        raise ValueError('Held retractor movement is limited to 0.5 mm per action')
    profile = engine.scene.properties['tissuePhysicsV2']['brain']
    if np.linalg.norm(np.subtract(position,binding['origin'])) > profile['maxDisplacementMm']:
        raise ValueError('Held target exceeds the unchanged native displacement profile')
    return checked_pair_action(engine, {'slot':hand,'operation':'move',
        'pose':{'position':list(map(float,position)),'quaternion':list(slot['quaternion'])},
        'duration':.5}, retractor_hand=hand, expected_hold=binding,
        limits=limits,meter=meter,check=check,phase='retraction.held_move')


def release_retractor(engine, proposal, *, hand='right', binding, limits, meter, check):
    """Return, release and withdraw using acquisition receipt's proposalUsed."""
    branch = engine.clone()
    receipts = []
    point = np.asarray(branch.state['slots'][hand]['position'],float)
    origin = np.asarray(binding['origin'],float)
    count = math.ceil(float(np.linalg.norm(point-origin))/.5)
    for index in range(1,count+1):
        position = point+(origin-point)*index/count
        branch, receipt = held_move(branch,position,hand=hand,binding=binding,
            limits=limits,meter=meter,check=check)
        receipts.append(receipt)
        if receipt['status']!='accepted':
            return branch, {'status':'refused','reason':receipt['reason'],'attempts':receipts}
    for action in (
        {'slot':hand,'operation':'inspect','duration':.5},
        {'slot':hand,'operation':'release','duration':.05},
        {'slot':hand,'operation':'withdraw','pose':copy.deepcopy(proposal['outsidePose']),
         'duration':max(.2,min(30.,float(np.linalg.norm(
             np.subtract(proposal['outsidePose']['position'],origin)))/2.))}):
        branch, receipt = checked_pair_action(branch,action,retractor_hand=hand,
            expected_hold=binding if action['operation']=='inspect' else None,
            limits=limits,meter=meter,check=check,phase='retraction.release_withdraw')
        receipts.append(receipt)
        if receipt['status']!='accepted':
            return branch, {'status':'refused','reason':receipt['reason'],'attempts':receipts}
    clean = not branch.state['holds'] and not any(
        slot['inserted'] or slot['active'] or slot['grip'] for slot in branch.state['slots'].values())
    return branch, {'status':'accepted' if clean else 'refused',
        'reason':None if clean else 'OTHER_HAND_CLEANUP_REQUIRED','attempts':receipts}


def withdraw_cusa_while_held(engine, route, *, hand='left', retractor_hand='right',
                             binding, limits, meter, check):
    """Certify CUSA-only withdrawal while the other hand keeps its native hold.

    This is deliberately not global cleanup. The caller must subsequently
    return/release/withdraw the retractor and check the whole native state.
    """
    from .surgical_staged_exit import _reverse_native_steps
    outside = route['outsidePose']
    slot = engine.state['slots'][hand]
    if slot['instrument']!='cusa' or not slot['inserted'] or hand==retractor_hand:
        raise ValueError('A distinct inserted CUSA hand is required')
    source = engine.clone()
    trials = []
    def attempt(actions, mode):
        branch = source.clone()
        entries = []
        for action in actions:
            branch, receipt = checked_pair_action(branch,action,retractor_hand=retractor_hand,
                expected_hold=binding,limits=limits,meter=meter,check=check,
                phase='retraction.cusa_exit.'+mode)
            entries.append(receipt)
            if receipt['status']!='accepted':
                break
        trials.append({'mode':mode,'attempts':entries})
        end = branch.state['slots'][hand]
        clean = (entries and entries[-1]['status']=='accepted' and not end['inserted'] and
            not end['active'] and not end['grip'] and
            np.linalg.norm(np.subtract(end['position'],outside['position']))<1e-6 and
            (Rotation.from_quat(end['quaternion'])*
             Rotation.from_quat(outside['quaternion']).inv()).magnitude()<1e-6)
        return branch, bool(clean)
    distance = float(np.linalg.norm(np.subtract(slot['position'],outside['position'])))
    for duration in sorted(set((max(.2,min(30.,distance/2.)),30.))):
        branch, clean = attempt([
            {'slot':hand,'operation':'deactivate','duration':.2},
            {'slot':hand,'operation':'withdraw','pose':copy.deepcopy(outside),'duration':duration}],
            'direct')
        if clean:
            return branch, {'status':'accepted','reason':None,'scope':'CUSA hand only; coupled retractor hold retained','trials':trials}
        if trials[-1]['attempts'][-1]['status']=='interrupted':
            return source, {'status':'interrupted','reason':trials[-1]['attempts'][-1]['reason'],'trials':trials}
    poses, reason = _reverse_native_steps(source,hand,check)
    if poses:
        for active in (False,True):
            actions=[{'slot':hand,'operation':'deactivate','duration':.2}]
            for index in range(0,len(poses),24):
                chunk=poses[index:index+24]
                actions.append({'slot':hand,'operation':'apply' if active else 'move',
                    'path':chunk,'duration':.1*len(chunk)})
            actions.extend([
                {'slot':hand,'operation':'deactivate','duration':.2},
                {'slot':hand,'operation':'withdraw','pose':copy.deepcopy(outside),'duration':.2}])
            branch,clean=attempt(actions,'native_steps_active' if active else 'native_steps_inactive')
            if clean:
                return branch, {'status':'accepted','reason':None,'scope':'CUSA hand only; coupled retractor hold retained','trials':trials}
            if trials[-1]['attempts'][-1]['status']=='interrupted':
                break
    return source, {'status':'refused','reason':reason or trials[-1]['attempts'][-1]['reason'] or
        'NO_CHECKED_CUSA_EXIT_WITH_HELD_RETRACTOR','trials':trials}
