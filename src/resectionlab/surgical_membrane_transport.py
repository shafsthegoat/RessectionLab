"""Source-bound rigid membrane surrogate: real cuts, whole-patch transport/exit.

This is a conservative categorical-cell engineering method, not calibrated dura
mechanics. Source cells stay occupied after cutting and after release. Only an
actual accepted full exterior extraction retires the exact declared patch.
"""
from __future__ import annotations
import numpy as np
from .core import semantic_digest
from .geometry import segments_within_distance
from .surgical_motion import envelopes,swept_envelopes,_held_cell_contact
from .surgical_rl_protection import cell_radius
from .surgical_tools import TOOLS
HOLDER='membrane_forceps'
HOLD_KIND='membrane-patch-v1'


def _contract(engine):
    from .surgical_cranial_stages import validate_stage_contract,require_evidence,verified_bone_prefix
    raw=getattr(engine,'_cranial_stages',None)
    if raw is None:raise ValueError('MEMBRANE_PATCH_STAGE_CONTRACT_REQUIRED')
    contract=validate_stage_contract(engine,raw);require_evidence(engine,contract)
    verified_bone_prefix(engine,contract,full='_transition_check' not in engine.__dict__)
    return contract


def _eligible_patch(engine,contract,allow_detached=None):
    from .surgical_cranial_stages import source_partition_sets
    p=source_partition_sets(contract)['preparation']
    if p&set(engine.state['removed']):raise ValueError('MEMBRANE_PATCH_MEMBERSHIP_ALREADY_REMOVED')
    cut={tuple(sorted(e)) for e in engine.state.get('cutEdges',[])}
    if not source_partition_sets(contract)['boundary']<=cut:raise ValueError('MEMBRANE_PATCH_COMPLETE_BOUNDARY_CUT_REQUIRED')
    if any(a['id'] not in engine.state['separatedAttachments'] for a in engine.scene.properties['attachments'] if a['id'] in contract.attachment_ids):
        raise ValueError('MEMBRANE_PATCH_ATTACHMENTS_NOT_RELEASED')
    if p&engine._protected_ids:raise ValueError('PROTECTED_STRUCTURE: membrane patch contains protected source cells')
    if any(p&set(h['cells']) for h in engine.state.get('detachedFragments',[]) if h is not allow_detached):
        raise ValueError('MEMBRANE_PATCH_MATERIAL_ALREADY_DETACHED')
    return contract.membrane_cells



def swept_cell_interior_intersects(start,end,stationary):
    """Exact same-affine unit-cell interiors; face/edge/corner touching is not overlap.

    A common nonsingular affine is bijective, so testing the index-space open
    box q+(-1,1)^3 covers sheared, reflected and oblique whole cells exactly.
    Inputs here are exact integer source indices/offsets, not rounded world poses.
    """
    start=np.asarray(start);end=np.asarray(end);stationary=np.asarray(stationary)
    lower,upper=0.,1.
    for a,d,q in zip(start,end-start,stationary):
        lo,hi=q-1,q+1
        if d==0:
            if a<=lo or a>=hi:return False
        else:
            t0,t1=(lo-a)/d,(hi-a)/d
            lower=max(lower,min(t0,t1));upper=min(upper,max(t0,t1))
            if lower>=upper:return False
    return lower<upper


def _lift_rule(engine,contract,before,after):
    from .surgical_cranial_stages import membrane_geometry
    axis,sign,_=membrane_geometry(contract);before=np.asarray(before);after=np.asarray(after)
    distances=np.asarray([before[axis]*sign,after[axis]*sign])
    if np.any(distances<0):raise ValueError('MEMBRANE_PATCH_INWARD_LIFT_UNSUPPORTED')
    lateral=[a for a in range(3) if a!=axis]
    if min(distances)<2 and np.any(before[lateral]!=after[lateral]):
        raise ValueError('MEMBRANE_PATCH_NORMAL_LIFT_TWO_CELLS_REQUIRED')


def _original_rim_contact_only(engine,contract,source_id,candidate,before,after,boundary_cut):
    from .surgical_cranial_stages import source_partition_sets
    sets=source_partition_sets(contract)
    if candidate not in sets['retained']:return False
    source=np.asarray(np.unravel_index(source_id,engine.scene.shape));rim=np.asarray(np.unravel_index(candidate,engine.scene.shape))
    if np.max(np.abs(source-rim))!=1:return False
    if not boundary_cut:return False
    if candidate in engine.state['removed'] or any(candidate in h['cells'] for h in [*engine.state['holds'].values(),*engine.state.get('detachedFragments',[])]):
        raise ValueError('MEMBRANE_PATCH_RETAINED_RIM_CHANGED')
    if np.linalg.norm(engine.state.get('deformation',{}).get(str(candidate),[0,0,0]))>0:
        raise ValueError('MEMBRANE_PATCH_RETAINED_RIM_CHANGED')
    return not swept_cell_interior_intersects(before,after,rim)

def validate_holder(engine, key):
    slot = engine.state['slots'][key]
    if slot['instrument'] != HOLDER or engine.state.get('toolMechanicsRevision') != 2:
        raise ValueError('MEMBRANE_PATCH_REQUIRES_STOCK_HOLDER_REVISION2')
    params = engine.parameters(slot)
    if params['radiusMm'] != 1.5 or params['depthMm'] != 2. or not 0 <= params['jawOpeningMm'] <= 2.:
        raise ValueError('MEMBRANE_PATCH_HOLDER_GEOMETRY_FIXED')
    if key in engine.state['holds'] and params['jawOpeningMm'] > .5:
        raise ValueError('MEMBRANE_PATCH_RELEASE_BEFORE_OPENING_JAWS')


def validate_hold(engine, key):
    """Recheck immutable membership and every declared attachment, without release."""
    validate_holder(engine, key)
    contract = _contract(engine)
    hold = engine.state['holds'].get(key)
    if (hold is None or hold.get('kind') != HOLD_KIND or
            hold.get('contractHash') != contract.content_hash or
            tuple(hold.get('cells', ())) != contract.membrane_cells or
            hold.get('componentHash') != semantic_digest(list(contract.membrane_cells))):
        raise ValueError('MEMBRANE_PATCH_HOLD_BINDING_INVALID')
    if set(hold['cells']) & set(engine.state['removed']):
        raise ValueError('MEMBRANE_PATCH_HELD_MEMBERSHIP_CHANGED')
    if set(hold['cells']) & engine._protected_ids:
        raise ValueError('PROTECTED_STRUCTURE: membrane patch contains protected membership')
    released = set(engine.state['separatedAttachments'])
    for attachment in engine.scene.properties['attachments']:
        if set(attachment['cells']) & set(hold['cells']) and attachment['id'] not in released:
            raise ValueError('MEMBRANE_PATCH_ATTACHMENTS_NOT_RELEASED')
    return contract, hold


def grasp(engine, key, cells, position, quaternion):
    # Eligibility derives actual cut boundary and declared attachment release.
    validate_holder(engine, key)
    contract = _contract(engine)
    if engine.parameters(engine.state['slots'][key])['jawOpeningMm'] > .5:
        return []
    if key in engine.state['holds']:
        validate_hold(engine, key)
        return []
    contacts = set(map(int, cells)) & set(contract.membrane_cells)
    if not contacts:
        return []
    fragment = list(contract.membrane_cells)
    if set(fragment) & engine._protected_ids:
        raise ValueError('PROTECTED_STRUCTURE: membrane patch contains protected membership')
    if any(set(fragment) & set(hold['cells']) for hold in engine.state['holds'].values()):
        raise ValueError('MEMBRANE_PATCH_MATERIAL_ALREADY_HELD')
    detached = [hold for hold in engine.state.get('detachedFragments', [])
                if set(fragment) & set(hold['cells'])]
    if detached:
        if len(detached) != 1 or detached[0].get('kind') != HOLD_KIND or detached[0].get('contractHash') != contract.content_hash:
            raise ValueError('MEMBRANE_PATCH_DETACHED_BINDING_INVALID')
    _eligible_patch(engine, contract, allow_detached=detached[0] if detached else None)
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
    # axis. Freeze only this local conservative jaw region, never the full patch
    # or a shaft exemption. Exact native material contacts remain separate.
    region = [source_id for source_id, point in zip(fragment, centers)
              if any(segments_within_distance(point, point, a, b, 2*radius+width+1e-9)
                     for a, b, width, _ in jaws)]
    hold.update(gripCells=region, gripContactCells=sorted(contacts),
                holderSlot=key,
                gripPose={'position': np.asarray(position).tolist(), 'quaternion': np.asarray(quaternion).tolist()})
    engine.state['holds'][key] = hold
    engine.state['slots'][key]['grip'] = fragment[:]
    return [{'kind': 'membrane_patch_grasp', 'componentHash': hold['componentHash'],
             'cells': len(fragment), 'regrasp': bool(detached), 'removalMm3': 0.}]


def _inside_bounds(engine, contract, indices):
    low = np.asarray(indices).min(0)-.5
    high = np.asarray(indices).max(0)+.5
    if np.any(low < -.5) or np.any(high > np.asarray(engine.scene.shape)-.5):
        raise ValueError('MEMBRANE_PATCH_OUTSIDE_OBSERVED_GRID')
    # The contract workspace and the pre-existing reviewed box both apply.
    bounds = [(engine._cranial_access.exit.minimum_voxel_boundary, engine._cranial_access.exit.maximum_voxel_boundary)]
    from .surgical_task_domain import reviewed_domain
    domain = reviewed_domain(engine.scene)
    if domain is not None:
        bounds.append((domain.minimum, domain.maximum))
    if any(np.any(low < np.asarray(lo)-1e-9) or np.any(high > np.asarray(hi)+1e-9) for lo, hi in bounds):
        raise ValueError('MEMBRANE_PATCH_OUTSIDE_REVIEWED_WORKSPACE')


def check_sweep(engine, key, new_offset, end_pose):
    """Guard full held volume before publishing its next snapped translation."""
    contract, hold = validate_hold(engine, key)
    original = np.column_stack(np.unravel_index(hold['cells'], engine.scene.shape))
    before = original+np.asarray(hold['offset'])
    after = original+np.asarray(new_offset)
    _lift_rule(engine,contract,hold['offset'],new_offset)
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
        raise ValueError('MEMBRANE_PATCH_GRIP_CONTACT_LOST')
    from .surgical_cranial_stages import source_partition_sets
    boundary_cut=source_partition_sets(contract)['boundary']<={tuple(sorted(e)) for e in engine.state['cutEdges']}
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
            raise ValueError('PROTECTED_STRUCTURE: entire membrane patch sweep enters declared clearance')
        occupied = set(owner.material_ids(engine.cells(start, end, radius)))-own
        if any(not _original_rim_contact_only(engine,contract,source_id,candidate,before[index],after[index],boundary_cut) for candidate in occupied):
            raise ValueError('MEMBRANE_PATCH_SWEPT_ANATOMY_COLLISION')
        contact = _held_cell_contact(start, end, radius, capsules, key, source_id, grip, segments_within_distance)
        if contact is not None:
            if contact is not False:
                raise ValueError('MEMBRANE_PATCH_SHAFT_COLLISION' if contact == key else 'MEMBRANE_PATCH_OTHER_TOOL_COLLISION:'+contact)
            continue
        for hand, segments in capsules.items():
            for u, v, width, part in segments:
                if hand == key and part != 'shaft' and source_id in grip:
                    continue  # Only the witnessed jaw contact may hold its membrane.
                if segments_within_distance(start, end, u, v, radius+width+1e-9):
                    raise ValueError('MEMBRANE_PATCH_SHAFT_COLLISION' if hand == key else 'MEMBRANE_PATCH_OTHER_TOOL_COLLISION:'+hand)


def check_tool_sweep(engine, key, end_pose, operation):
    """The moving other hand must clear whole parked and released patches."""
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
                    raise ValueError('MEMBRANE_PATCH_MOVING_TOOL_COLLISION:'+key)


def check_detached(engine):
    """Retained released patches cannot disappear into a later coupled field."""
    fragments = [hold for hold in engine.state.get('detachedFragments', []) if hold.get('kind') == HOLD_KIND]
    if not fragments:
        return
    contract = _contract(engine)
    from .surgical_cranial_stages import source_partition_sets
    boundary_cut=source_partition_sets(contract)['boundary']<={tuple(sorted(e)) for e in engine.state['cutEdges']}
    radius = cell_radius(engine.scene.affine)
    owner = engine.occupancy()
    positions = engine._positions()
    for hold in fragments:
        if (hold.get('contractHash') != contract.content_hash or tuple(hold['cells']) != contract.membrane_cells or
                set(hold['cells']) & (set(engine.state['removed']) | engine._protected_ids)):
            raise ValueError('MEMBRANE_PATCH_DETACHED_BINDING_INVALID')
        indices = np.column_stack(np.unravel_index(hold['cells'], engine.scene.shape))+hold['offset']
        _inside_bounds(engine, contract, indices)
        points = indices@engine.scene.affine[:3, :3].T+engine.scene.affine[:3, 3]
        own = set(hold['cells'])
        for index, (source_id, point) in enumerate(zip(hold['cells'], points)):
            if index % 32 == 0 and getattr(engine, '_transition_check', None) is not None:
                engine._transition_check()
            if engine.protection is not None and len(engine.protection.capsule_hits(point, point, radius, positions=positions)):
                raise ValueError('PROTECTED_STRUCTURE: retained detached membrane patch contacts declared clearance')
            candidates=set(owner.material_ids(engine.cells(point, point, radius)))-own
            if any(not _original_rim_contact_only(engine,contract,source_id,candidate,indices[index],indices[index],boundary_cut) for candidate in candidates):
                raise ValueError('MEMBRANE_PATCH_DETACHED_ANATOMY_COLLISION')
            for key, slot in engine.state['slots'].items():
                for a, b, width, part in envelopes(TOOLS[slot['instrument']], engine.parameters(slot), slot['position'], slot['quaternion']):
                    if key == hold.get('holderSlot') and slot['instrument'] == HOLDER and part != 'shaft' and source_id in hold['gripCells']:
                        continue
                    if segments_within_distance(point, point, a, b, radius+width+1e-9):
                        raise ValueError('MEMBRANE_PATCH_DETACHED_TOOL_COLLISION:'+key)


def extract(engine, key):
    """Retire the patch once, beyond its declared common source exterior face."""
    contract, hold = validate_hold(engine, key)
    if not any(hold['offset']):
        raise ValueError('MEMBRANE_PATCH_MOVE_THROUGH_EXIT_BEFORE_EXTRACTION')
    check_sweep(engine, key, hold['offset'], engine.state['slots'][key])
    original = np.column_stack(np.unravel_index(hold['cells'], engine.scene.shape))
    destination = original+hold['offset']
    axis, side, plane = engine._cranial_access.exit.axis, engine._cranial_access.exit.side, engine._cranial_access.exit.plane_voxel_boundary
    occupied = np.column_stack(np.nonzero(engine.scene.labels))
    # Contract validation binds the outward face. Check complete cell extents,
    # not centers; touching the boundary is not complete exterior evacuation.
    if side == -1:
        exterior = destination[:, axis].max()+.5 < min(plane, occupied[:, axis].min()-.5)-1e-9
    else:
        exterior = destination[:, axis].min()-.5 > max(plane, occupied[:, axis].max()+.5)+1e-9
    if not exterior:
        raise ValueError('MEMBRANE_PATCH_NOT_AT_DECLARED_EXTERIOR_EXIT')
    volume = len(hold['cells'])*engine.scene.voxel_mm3
    receipt = {'kind': 'membrane_patch_evacuation', 'contractHash': contract.content_hash,
               'componentHash': hold['componentHash'], 'cells': len(hold['cells']),
               'volumeMm3': volume, 'offset': hold['offset'][:],
               'exitAxis': axis, 'exitSide': side, 'checkedWholeFragmentSweep': True}
    engine.state['removed'] = sorted(set(engine.state['removed']) | set(hold['cells']))
    engine.state.setdefault('membranePatchEvacuations', []).append(receipt)
    del engine.state['holds'][key]
    engine.state['slots'][key]['grip'] = []
    return [receipt]



def clearance(engine,contract=None):
    if getattr(engine, '_resident', None) is not None and (contract is None or contract is engine._cranial_stages):
        from .surgical_native_engine import query
        native = query(engine, 'membrane')
        if native is not NotImplemented:return native
    """Require actual native extraction ancestry, not a removed count/flag alone."""
    contract=_contract(engine) if contract is None else contract
    from .surgical_cranial_stages import source_partition_sets
    sets=source_partition_sets(contract);p,r=sets['preparation'],sets['retained']
    removed=set(engine.state['removed'])
    fragments=[*engine.state['holds'].values(),*engine.state.get('detachedFragments',[])]
    events=[];selected={'left':'cranial_drill','right':'cusa'}
    for action,event in zip(engine.actions,engine.history):
        hand=action['slot']
        if action['operation']=='exchange':selected[hand]=action['instrument']
        for effect in event.get('events',[]):
            if effect.get('kind')!='membrane_patch_evacuation':continue
            if (event.get('status')!='accepted' or event.get('completedFraction')!=1. or
                action['operation']!='extract' or selected[hand]!=HOLDER or
                effect.get('contractHash')!=contract.content_hash or
                effect.get('componentHash')!=semantic_digest(list(contract.membrane_cells)) or
                effect.get('cells')!=len(p) or effect.get('volumeMm3')!=len(p)*engine.scene.voxel_mm3):
                raise ValueError('MEMBRANE_PATCH_NATIVE_EXTRACTION_HISTORY_REQUIRED')
            events.append(effect)
    recorded=engine.state.get('membranePatchEvacuations',[])
    valid=(len(events)==1 and recorded==events and p<=removed and not r&removed and
        not any((p|r)&set(h['cells']) for h in fragments) and
        not any(int(c) in r and np.linalg.norm(v)>0 for c,v in engine.state.get('deformation',{}).items()))
    return {'cleared':bool(valid),'patchSourceCells':len(p),'removedPatchCells':len(p&removed),
        'removedMembraneMm3':len(p&removed)*engine.scene.voxel_mm3,'retainedRimIntact':not bool(r&removed),
        'actualNativeExtractionEvents':len(events),'clinicalValidation':'none',
        'method':'source-bound categorical whole-patch cut/translate/extract surrogate; no membrane continuum or clinical calibration'}
