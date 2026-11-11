"""Conservative source-cell bone-flap transport, not calibrated bone mechanics.

The rigid surrogate translates whole affine cells at integer grid offsets. Cell
circumspheres enclose the entire continuous translation, including quantization
jumps; false refusals are preferable to an endpoint-only clearance claim.
"""
from __future__ import annotations

import numpy as np

from .core import semantic_digest
from .geometry import segments_within_distance
from .surgical_motion import envelopes, swept_envelopes
from .surgical_rl_protection import cell_radius
from .surgical_tools import TOOLS

HOLDER = 'bone_flap_holder'
HOLD_KIND = 'bone-flap-v1'


def _contract(engine):
    from .surgical_craniotomy import require_evidence, validate_contract
    contract = validate_contract(engine.scene)
    if contract is None:
        raise ValueError('BONE_FLAP_CONTRACT_REQUIRED')
    require_evidence(contract)
    return contract


def validate_holder(engine, key):
    slot = engine.state['slots'][key]
    if slot['instrument'] != HOLDER or engine.state.get('toolMechanicsRevision') != 2:
        raise ValueError('BONE_FLAP_REQUIRES_STOCK_HOLDER_REVISION2')
    params = engine.parameters(slot)
    if params['radiusMm'] != 1.5 or params['depthMm'] != 2. or not 0 <= params['jawOpeningMm'] <= 2.:
        raise ValueError('BONE_FLAP_HOLDER_GEOMETRY_FIXED')
    if key in engine.state['holds'] and params['jawOpeningMm'] > .5:
        raise ValueError('BONE_FLAP_RELEASE_BEFORE_OPENING_JAWS')


def validate_hold(engine, key):
    """Recheck immutable membership and every declared attachment, without release."""
    validate_holder(engine, key)
    contract = _contract(engine)
    hold = engine.state['holds'].get(key)
    if (hold is None or hold.get('kind') != HOLD_KIND or
            hold.get('contractHash') != contract.content_hash or
            tuple(hold.get('cells', ())) != contract.flap_cells or
            hold.get('componentHash') != semantic_digest(list(contract.flap_cells))):
        raise ValueError('BONE_FLAP_HOLD_BINDING_INVALID')
    if set(hold['cells']) & set(engine.state['removed']):
        raise ValueError('BONE_FLAP_HELD_MEMBERSHIP_CHANGED')
    if set(hold['cells']) & engine._protected_ids:
        raise ValueError('PROTECTED_STRUCTURE: bone flap contains protected membership')
    released = set(engine.state['separatedAttachments'])
    for attachment in engine.scene.properties['attachments']:
        if set(attachment['cells']) & set(hold['cells']) and attachment['id'] not in released:
            raise ValueError('BONE_FLAP_ATTACHMENTS_NOT_RELEASED')
    return contract, hold


def grasp(engine, key, cells, position, quaternion):
    from .surgical_craniotomy import eligible_bone_component
    validate_holder(engine, key)
    contract = _contract(engine)
    if engine.parameters(engine.state['slots'][key])['jawOpeningMm'] > .5:
        return []
    if key in engine.state['holds']:
        validate_hold(engine, key)
        return []
    contacts = set(map(int, cells)) & set(contract.flap_cells)
    if not contacts:
        return []
    fragment = list(contract.flap_cells)
    if set(fragment) & engine._protected_ids:
        raise ValueError('PROTECTED_STRUCTURE: bone flap contains protected membership')
    if any(set(fragment) & set(hold['cells']) for hold in engine.state['holds'].values()):
        raise ValueError('BONE_FLAP_MATERIAL_ALREADY_HELD')
    detached = [hold for hold in engine.state.get('detachedFragments', [])
                if set(fragment) & set(hold['cells'])]
    if detached:
        if len(detached) != 1 or detached[0].get('kind') != HOLD_KIND or detached[0].get('contractHash') != contract.content_hash:
            raise ValueError('BONE_FLAP_DETACHED_BINDING_INVALID')
    eligible_bone_component(engine, contract=contract, allow_detached=detached[0] if detached else None)
    if detached:
        hold = detached[0]
        engine.state['detachedFragments'].remove(hold)
        hold.update(origin=np.asarray(position).tolist(), baseOffset=hold['offset'][:])
    else:
        hold = {'kind': HOLD_KIND, 'contractHash': contract.content_hash,
                'componentHash': semantic_digest(fragment), 'cells': fragment,
                'origin': np.asarray(position).tolist(), 'offset': [0, 0, 0],
                'baseOffset': [0, 0, 0]}
    centers = engine.scene.points(fragment)+engine.scene.affine[:3, :3]@np.asarray(hold['offset'])
    radius = cell_radius(engine.scene.affine)
    jaws = [segment for segment in envelopes(TOOLS[HOLDER], engine.parameters(engine.state['slots'][key]),
            position, quaternion) if segment[3] != 'shaft']
    # The snapped piece can lag its physical grip by half a voxel on each
    # axis. Freeze only this local conservative jaw region, never the full flap
    # or a shaft exemption. Exact native material contacts remain separate.
    region = [source_id for source_id, point in zip(fragment, centers)
              if any(segments_within_distance(point, point, a, b, 2*radius+width+1e-9)
                     for a, b, width, _ in jaws)]
    hold.update(gripCells=region, gripContactCells=sorted(contacts),
                holderSlot=key,
                gripPose={'position': np.asarray(position).tolist(), 'quaternion': np.asarray(quaternion).tolist()})
    engine.state['holds'][key] = hold
    engine.state['slots'][key]['grip'] = fragment[:]
    return [{'kind': 'bone_flap_grasp', 'componentHash': hold['componentHash'],
             'cells': len(fragment), 'regrasp': bool(detached), 'removalMm3': 0.}]


def _inside_bounds(engine, contract, indices):
    low = np.asarray(indices).min(0)-.5
    high = np.asarray(indices).max(0)+.5
    if np.any(low < -.5) or np.any(high > np.asarray(engine.scene.shape)-.5):
        raise ValueError('BONE_FLAP_OUTSIDE_OBSERVED_GRID')
    # The contract workspace and the pre-existing reviewed box both apply.
    bounds = [(contract.exit.minimum_voxel_boundary, contract.exit.maximum_voxel_boundary)]
    from .surgical_task_domain import reviewed_domain
    domain = reviewed_domain(engine.scene)
    if domain is not None:
        bounds.append((domain.minimum, domain.maximum))
    if any(np.any(low < np.asarray(lo)-1e-9) or np.any(high > np.asarray(hi)+1e-9) for lo, hi in bounds):
        raise ValueError('BONE_FLAP_OUTSIDE_REVIEWED_WORKSPACE')


def check_sweep(engine, key, new_offset, end_pose):
    """Guard full held volume before publishing its next snapped translation."""
    contract, hold = validate_hold(engine, key)
    original = np.column_stack(np.unravel_index(hold['cells'], engine.scene.shape))
    before = original+np.asarray(hold['offset'])
    after = original+np.asarray(new_offset)
    _inside_bounds(engine, contract, np.concatenate((before, after)))
    a = before @ engine.scene.affine[:3, :3].T+engine.scene.affine[:3, 3]
    b = after @ engine.scene.affine[:3, :3].T+engine.scene.affine[:3, 3]
    radius = cell_radius(engine.scene.affine)
    own = set(hold['cells'])
    grip = set(hold['gripCells'])
    witness = set(hold['gripContactCells'])
    jaws = [segment for segment in envelopes(TOOLS[HOLDER], engine.parameters(engine.state['slots'][key]),
            end_pose['position'], end_pose['quaternion']) if segment[3] != 'shaft']
    if not witness or not witness <= grip <= own or not any(
            source_id in witness and any(segments_within_distance(point, point, u, v, 2*radius+width+1e-9)
                                        for u, v, width, _ in jaws)
            for source_id, point in zip(hold['cells'], b)):
        raise ValueError('BONE_FLAP_GRIP_CONTACT_LOST')
    owner = engine.occupancy()
    protection_positions = engine._positions()
    capsules = {}
    for hand, slot in engine.state['slots'].items():
        tool = TOOLS[slot['instrument']]
        capsules[hand] = (swept_envelopes(tool, engine.parameters(slot), slot, end_pose)
                          if hand == key else envelopes(tool, engine.parameters(slot), slot['position'], slot['quaternion']))
    for index, (source_id, start, end) in enumerate(zip(hold['cells'], a, b)):
        if index % 32 == 0 and getattr(engine, '_transition_check', None) is not None:
            engine._transition_check()
        if engine.protection is not None and len(engine.protection.capsule_hits(
                start, end, radius, positions=protection_positions)):
            raise ValueError('PROTECTED_STRUCTURE: entire bone flap sweep enters declared clearance')
        occupied = set(owner.material_ids(engine.cells(start, end, radius)))-own
        if occupied:
            raise ValueError('BONE_FLAP_SWEPT_ANATOMY_COLLISION')
        for hand, segments in capsules.items():
            for u, v, width, part in segments:
                if hand == key and part != 'shaft' and source_id in grip:
                    continue  # Only the witnessed jaw contact may hold its bone.
                if segments_within_distance(start, end, u, v, radius+width+1e-9):
                    raise ValueError('BONE_FLAP_SHAFT_COLLISION' if hand == key else 'BONE_FLAP_OTHER_TOOL_COLLISION:'+hand)


def check_tool_sweep(engine, key, end_pose, operation):
    """The moving other hand must clear whole parked and released flaps."""
    slot = engine.state['slots'][key]
    capsules = swept_envelopes(TOOLS[slot['instrument']], engine.parameters(slot), slot, end_pose)
    fragments = [(hand, hold) for hand, hold in engine.state['holds'].items()]
    fragments.extend((None, hold) for hold in engine.state.get('detachedFragments', []))
    radius = cell_radius(engine.scene.affine)
    for hand, hold in fragments:
        if hold.get('kind') != HOLD_KIND or hand == key:
            continue
        centers = engine.scene.points(hold['cells'])+engine.scene.affine[:3, :3]@np.asarray(hold['offset'])
        for index, (source_id, point) in enumerate(zip(hold['cells'], centers)):
            if index % 32 == 0 and getattr(engine, '_transition_check', None) is not None:
                engine._transition_check()
            for a, b, width, part in capsules:
                if hand is None and slot['instrument'] == HOLDER and part != 'shaft':
                    if operation == 'grasp' or (hold.get('holderSlot') == key and
                            operation in ('release', 'deactivate', 'withdraw', 'inspect') and
                            source_id in hold['gripCells']):
                        continue
                if segments_within_distance(point, point, a, b, radius+width+1e-9):
                    raise ValueError('BONE_FLAP_MOVING_TOOL_COLLISION:'+key)


def check_detached(engine):
    """Retained released flaps cannot disappear into a later coupled field."""
    fragments = [hold for hold in engine.state.get('detachedFragments', []) if hold.get('kind') == HOLD_KIND]
    if not fragments:
        return
    contract = _contract(engine)
    radius = cell_radius(engine.scene.affine)
    owner = engine.occupancy()
    positions = engine._positions()
    for hold in fragments:
        if (hold.get('contractHash') != contract.content_hash or tuple(hold['cells']) != contract.flap_cells or
                set(hold['cells']) & (set(engine.state['removed']) | engine._protected_ids)):
            raise ValueError('BONE_FLAP_DETACHED_BINDING_INVALID')
        indices = np.column_stack(np.unravel_index(hold['cells'], engine.scene.shape))+hold['offset']
        _inside_bounds(engine, contract, indices)
        points = indices@engine.scene.affine[:3, :3].T+engine.scene.affine[:3, 3]
        own = set(hold['cells'])
        for index, (source_id, point) in enumerate(zip(hold['cells'], points)):
            if index % 32 == 0 and getattr(engine, '_transition_check', None) is not None:
                engine._transition_check()
            if engine.protection is not None and len(engine.protection.capsule_hits(point, point, radius, positions=positions)):
                raise ValueError('PROTECTED_STRUCTURE: retained detached bone flap contacts declared clearance')
            if set(owner.material_ids(engine.cells(point, point, radius)))-own:
                raise ValueError('BONE_FLAP_DETACHED_ANATOMY_COLLISION')
            for key, slot in engine.state['slots'].items():
                for a, b, width, part in envelopes(TOOLS[slot['instrument']], engine.parameters(slot), slot['position'], slot['quaternion']):
                    if key == hold.get('holderSlot') and slot['instrument'] == HOLDER and part != 'shaft' and source_id in hold['gripCells']:
                        continue
                    if segments_within_distance(point, point, a, b, radius+width+1e-9):
                        raise ValueError('BONE_FLAP_DETACHED_TOOL_COLLISION:'+key)


def extract(engine, key):
    """Retire the flap once, beyond its declared common source exterior face."""
    contract, hold = validate_hold(engine, key)
    if not any(hold['offset']):
        raise ValueError('BONE_FLAP_MOVE_THROUGH_EXIT_BEFORE_EXTRACTION')
    check_sweep(engine, key, hold['offset'], engine.state['slots'][key])
    original = np.column_stack(np.unravel_index(hold['cells'], engine.scene.shape))
    destination = original+hold['offset']
    axis, side, plane = contract.exit.axis, contract.exit.side, contract.exit.plane_voxel_boundary
    occupied = np.column_stack(np.nonzero(engine.scene.labels))
    # Contract validation binds the outward face. Check complete cell extents,
    # not centers; touching the boundary is not complete exterior evacuation.
    if side == -1:
        exterior = destination[:, axis].max()+.5 < min(plane, occupied[:, axis].min()-.5)-1e-9
    else:
        exterior = destination[:, axis].min()-.5 > max(plane, occupied[:, axis].max()+.5)+1e-9
    if not exterior:
        raise ValueError('BONE_FLAP_NOT_AT_DECLARED_EXTERIOR_EXIT')
    volume = len(hold['cells'])*engine.scene.voxel_mm3
    receipt = {'kind': 'bone_flap_evacuation', 'contractHash': contract.content_hash,
               'componentHash': hold['componentHash'], 'cells': len(hold['cells']),
               'volumeMm3': volume, 'offset': hold['offset'][:],
               'exitAxis': axis, 'exitSide': side, 'checkedWholeFragmentSweep': True}
    engine.state['removed'] = sorted(set(engine.state['removed']) | set(hold['cells']))
    engine.state.setdefault('boneFlapEvacuations', []).append(receipt)
    del engine.state['holds'][key]
    engine.state['slots'][key]['grip'] = []
    engine.state['boneTopologyDirty'] = True
    return [receipt]


def aperture_evacuated(engine, contract=None):
    from .surgical_milling import is_milling, milling_readiness
    candidate = engine._cranial_access if contract is None else contract
    if is_milling(candidate):
        return milling_readiness(engine)['apertureCleared']
    contract = _contract(engine) if contract is None else contract
    return (set(contract.flap_cells) | set(contract.kerf_cells) <= set(engine.state['removed']) and
            any(event.get('contractHash') == contract.content_hash for event in engine.state.get('boneFlapEvacuations', [])))


def describe(engine):
    contract = engine._cranial_access
    from .surgical_milling import is_milling, milling_readiness
    if is_milling(contract):
        if getattr(engine, '_cranial_stages', None) is not None:
            from .surgical_cranial_stages import bone_prefix_description
            staged = bone_prefix_description(engine)
            if staged is not None:
                return staged
        status = milling_readiness(engine)
        phase = ('MILLING_WITHDRAWN' if status['openingComplete'] else
                 'MILLING_CLEARED' if status['apertureCleared'] else
                 'MILLING_DEBRIS_RETAINED' if status['apertureRemoved'] else
                 'MILLING_IN_PROGRESS' if status['removedBoneMm3'] else 'MILLING_BOUND')
        return {'contractHash': contract.content_hash, 'method': 'milling', 'phase': phase,
                'kerfRemovedMm3': 0., 'flapEvacuatedMm3': 0.,
                'milledBoneMm3': status['removedBoneMm3'],
                'declaredBoneRemovedMm3': status['removedBoneMm3'],
                'apertureEvacuated': status['apertureCleared'],
                'openingComplete': status['openingComplete'], 'milling': status,
                'transport': 'native solid-to-debris erosion and metered suction; no rigid flap',
                'clinicalValidation': 'none'}
    removed = set(engine.state['removed'])
    kerf = len(removed & set(contract.kerf_cells))
    flap = len(removed & set(contract.flap_cells))
    held = any(h.get('kind') == HOLD_KIND for h in engine.state['holds'].values())
    detached = any(h.get('kind') == HOLD_KIND for h in engine.state.get('detachedFragments', []))
    evacuated = aperture_evacuated(engine, contract)
    withdrawn = not engine.state['holds'] and not engine.state.get('detachedFragments') and all(
        not slot['inserted'] and not slot['active'] and not slot['grip'] for slot in engine.state['slots'].values())
    from .surgical_cranial_guards import tool_is_exterior
    from .surgical_task_domain import _posed
    withdrawn = withdrawn and all(not _posed(engine, key) or tool_is_exterior(engine, key) for key in engine.state['slots'])
    phase = ('HOLDER_WITHDRAWN' if evacuated and withdrawn else 'EVACUATED' if evacuated
             else 'GRASPED_OR_TRANSPORTED' if held else 'DETACHED_IN_FIELD' if detached
             else 'KERF_IN_PROGRESS' if kerf else 'BOUND')
    return {'contractHash': contract.content_hash, 'phase': phase,
            'kerfRemovedMm3': kerf*engine.scene.voxel_mm3,
            'flapEvacuatedMm3': flap*engine.scene.voxel_mm3,
            'declaredBoneRemovedMm3': (kerf+flap)*engine.scene.voxel_mm3,
            'apertureEvacuated': evacuated, 'openingComplete': evacuated and withdrawn,
            'transport': 'translation-only snapped affine cells with conservative whole-volume sweeps',
            'clinicalValidation': 'none'}
