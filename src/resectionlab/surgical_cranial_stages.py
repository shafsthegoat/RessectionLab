"""Optional source-bound bone -> membrane -> work guard, not clinical access.

The envelope is external to scene properties: existing Milling/flap source hashes
remain unchanged. Only categorical whole-cell membrane cut/extraction is modeled;
scalp exposure is an explicit conditional start, never inferred from absent cells.
"""
from __future__ import annotations
from dataclasses import dataclass,fields
import math
import json
import re
import weakref
from types import MappingProxyType
import numpy as np
from .core import semantic_digest
from .surgical_craniotomy import EvidenceProof, _original_targets
from .surgical_milling import resolve_contract, require_coverage
from .surgical_rl_protection import grid_binding

SCHEMA='resectionlab.cranial-stages.v1'
PROOFS=('exposure','membrane','support','attachments','work','exit')
# Per-owner runtime witnesses; collection does not change implementation constants.
_prefixes=weakref.WeakValueDictionary()
_prefix_facts={}
_contracts=weakref.WeakValueDictionary()
_source_facts={}
_support={}
_initial_state={}
_partitions={}
_contract_fields={}
_geometry={}
_guard_config={}


def _fail(reason):raise ValueError('CRANIAL_STAGE_'+reason)
def _cells(value,size,*,empty=False):
    if (not isinstance(value,(list,tuple)) or (not value and not empty) or
        any(type(i) is not int or not 0<=i<size for i in value) or len(set(value))!=len(value)):
        _fail('INVALID_SOURCE_CELLS')
    return tuple(sorted(value))
def _neighbors(shape,cell):
    p=list(np.unravel_index(cell,shape))
    for a in range(3):
        for d in (-1,1):
            q=p.copy();q[a]+=d
            if 0<=q[a]<shape[a]:yield int(np.ravel_multi_index(q,shape))
def _components(shape,cells):
    todo=set(cells);result=[]
    while todo:
        stack=[todo.pop()];group=set(stack)
        while stack:
            for n in _neighbors(shape,stack.pop()):
                if n in todo:todo.remove(n);group.add(n);stack.append(n)
        result.append(group)
    return result

def _target_binding(scene):
    ids=tuple(sorted(_original_targets(scene)))
    return ids,semantic_digest({'sourceSceneId':scene.identity,'originalTargetCells':ids})
def _protection(engine):
    return None if engine._protection_contract is None else engine._protection_contract['contentHash']
def _proofs(value):
    if type(value) is not dict or set(value)!=set(PROOFS):_fail('EXACT_PROOFS_REQUIRED')
    out=[]
    for name in PROOFS:
        p=value[name]
        if (type(p) is not dict or set(p)!={'status','provenance','evidenceSha256'} or
            p['status'] not in ('unknown','unreviewed','reviewed') or
            type(p['provenance']) is not str or not p['provenance'] or
            p['evidenceSha256'] is not None and (type(p['evidenceSha256']) is not str or not re.fullmatch('[a-f0-9]{64}',p['evidenceSha256']))):
            _fail('MALFORMED_PROOF')
        out.append((name,EvidenceProof(p['status'],p['provenance'],p['evidenceSha256'])))
    return tuple(out)


@dataclass(frozen=True)
class CranialStageContract:
    source_scene_id:str
    source_binding:str
    target_binding:str
    protection_binding:str|None
    bone_contract_hash:str
    bone_cells:tuple[int,...]
    membrane_cells:tuple[int,...]
    retained_membrane_cells:tuple[int,...]
    membrane_anchor_cells:tuple[int,...]
    boundary_edges:tuple[tuple[int,int],...]
    attachment_ids:tuple[str,...]
    work_cells:tuple[int,...]
    original_target_cells:tuple[int,...]
    starting_exposure:str
    membrane_lift_axis:int
    membrane_outward_sign:int
    scissors_depth_mm:float
    evidence:tuple[tuple[str,EvidenceProof],...]
    content_hash:str

    def as_dict(self):
        return {'schema':SCHEMA,'sourceSceneId':self.source_scene_id,'sourceBinding':self.source_binding,
            'targetBinding':self.target_binding,'protectionBinding':self.protection_binding,'boneContractHash':self.bone_contract_hash,
            'boneCells':list(self.bone_cells),'membraneCells':list(self.membrane_cells),'retainedMembraneCells':list(self.retained_membrane_cells),
            'membraneAnchorCells':list(self.membrane_anchor_cells),'boundaryEdges':[list(e) for e in self.boundary_edges],
            'attachmentIds':list(self.attachment_ids),'workCells':list(self.work_cells),'originalTargetCells':list(self.original_target_cells),
            'startingExposure':self.starting_exposure,'membraneLiftAxis':self.membrane_lift_axis,
            'membraneOutwardSign':self.membrane_outward_sign,'scissorsDepthMm':self.scissors_depth_mm,'evidence':{k:{'status':p.status,'provenance':p.provenance,'evidenceSha256':p.evidence_sha256} for k,p in self.evidence},
            'contentHash':self.content_hash}


def _declaration(engine,*,membrane_cells,retained_membrane_cells,membrane_anchor_cells,work_cells,evidence,starting_exposure):
    scene=engine.scene;scene.require_cache_source();bone=require_coverage(resolve_contract(scene))
    if bone!=engine._cranial_access:_fail('BONE_CONTRACT_BINDING')
    p=_cells(membrane_cells,scene.labels.size);r=_cells(retained_membrane_cells,scene.labels.size)
    anchors=_cells(membrane_anchor_cells,scene.labels.size,empty=True);w=_cells(work_cells,scene.labels.size)
    material=lambda c:engine.by_label.get(int(scene.labels.flat[c]),{}).get('material')
    all_mem=set(map(int,np.flatnonzero(np.isin(scene.labels,[x['id'] for x in scene.regions if x['material']=='membrane']))))
    if set(p)&set(r) or set(p)|set(r)!=all_mem or not set(anchors)<=set(r):_fail('MEMBRANE_PARTITION_REQUIRED')
    if len(_components(scene.shape,p))!=1:_fail('CONNECTED_PATCH_REQUIRED')
    if any(material(c) not in ('soft_tissue','tumor') for c in w):_fail('WORK_MATERIAL_SCOPE')
    target,target_hash=_target_binding(scene)
    if set(p)&(set(target)|set(engine._protected_ids)):_fail('PATCH_TARGET_OR_PROTECTED_MEMBERSHIP')
    if set(w)&(set(p)|set(r)|set(bone.window_cells)):_fail('DISJOINT_WORK_SCOPE_REQUIRED')
    if 'attachments' not in scene.properties:_fail('EXPLICIT_ATTACHMENTS_REQUIRED')
    touched=[a for a in scene.properties['attachments'] if set(p)&set(a['cells'])]
    if len({a['id'] for a in touched})!=len(touched):_fail('ATTACHMENT_IDENTITY')
    # Mesh faces cannot silently retain a second untransported sheet owner.
    adapter=scene.source.get('adapter') or {}
    if not isinstance(adapter,dict):_fail('SOURCE_ADAPTER_DECLARATION')
    source=adapter.get('sourceGeometry') or scene.source.get('sourceGeometry') or scene.source
    if not isinstance(source,dict):_fail('SOURCE_GEOMETRY_DECLARATION')
    if source.get('topology') in ('sheet','strand') or source.get('faces'):_fail('MESH_PATCH_CORRESPONDENCE_UNSUPPORTED')
    if starting_exposure not in ('conditional_post_scalp_exposure','intact_head'):_fail('EXPOSURE_DECLARATION')
    retained=set(r)
    edges=tuple(sorted({tuple(sorted((c,n))) for c in p for n in _neighbors(scene.shape,c) if n in retained}))
    if not edges:_fail('RETAINED_RIM_BOUNDARY_REQUIRED')
    mapped=scene.inverse[:3,:3]@np.asarray(bone.entry.normal_ras_unit)
    axis=int(np.argmax(np.abs(mapped)));sign=-1 if mapped[axis]>0 else 1
    span=float(np.ptp(scene.points(p)@np.asarray(bone.entry.normal_ras_unit)))
    scissors_depth=float(max(2,math.ceil(span+2.-1e-9)))
    if scissors_depth>5.:_fail('SOURCE_SPAN_EXCEEDS_SCISSOR_FAMILY')
    return CranialStageContract(scene.identity,grid_binding(scene),target_hash,_protection(engine),bone.content_hash,
        tuple(bone.window_cells),p,r,anchors,edges,tuple(sorted(a['id'] for a in touched)),w,target,starting_exposure,axis,sign,scissors_depth,_proofs(evidence),'')


def freeze_cranial_stage_contract(engine,*,membrane_cells,retained_membrane_cells,membrane_anchor_cells,work_cells,evidence,starting_exposure='conditional_post_scalp_exposure'):
    """Freeze source sets before actions; reviewed proofs are explicit, not inferred."""
    if engine.actions or engine.history:_fail('FREEZE_BEFORE_ACTIONS_REQUIRED')
    if getattr(engine,'_cranial_stages',None) is not None:_fail('ALREADY_BOUND')
    if engine.state_id!=engine.fresh().state_id:_fail('CANONICAL_SOURCE_REQUIRED')
    result=_declaration(engine,membrane_cells=membrane_cells,retained_membrane_cells=retained_membrane_cells,
        membrane_anchor_cells=membrane_anchor_cells,work_cells=work_cells,evidence=evidence,starting_exposure=starting_exposure)
    raw=result.as_dict();raw.pop('contentHash')
    from dataclasses import replace
    return _register(engine,replace(result,content_hash=semantic_digest(raw)))


def validate_stage_contract(engine,contract):
    """Typed revalidation and exact source/protection equality, no evidence promotion."""
    if isinstance(contract,CranialStageContract) and _contracts.get(id(contract)) is contract:
        witness,evidence_witness=_contract_fields[id(contract)]
        if any((getattr(contract,k) is not v if isinstance(v,tuple) else getattr(contract,k)!=v) for k,v in witness):_fail('FROZEN_CONTRACT_CHANGED')
        if any((p.status,p.provenance,p.evidence_sha256)!=(status,provenance,evidence_hash) for p,status,provenance,evidence_hash in evidence_witness):_fail('FROZEN_EVIDENCE_CHANGED')
        scene=engine.scene;scene.require_cache_source()
        guard,guard_values=_guard_config[id(contract)]
        if engine.guard_config is not guard or (guard is not None and
            ((type(guard.require_critical_coverage),guard.require_critical_coverage),
             (type(guard.critical_max_displacement_mm),guard.critical_max_displacement_mm))!=guard_values):_fail('PROTECTION_CONFIG_CHANGED')
        original_proof,protection,guard_facts=_source_facts[id(contract)]
        if (scene.identity!=contract.source_scene_id or scene._original_target_proof is not original_proof or
            _protection(engine)!=contract.protection_binding or engine._cranial_access is None or
            engine._cranial_access.content_hash!=contract.bone_contract_hash):_fail('CONTRACT_SOURCE_CHANGED')
        if engine.protection is not protection:_fail('PROTECTION_OBJECT_CHANGED')
        if protection is not None and (protection.clearance_mm!=guard_facts[0] or protection.required!=guard_facts[1] or
            protection.annotations is not guard_facts[2] or protection.cells is not guard_facts[3]):_fail('PROTECTION_SOURCE_CHANGED')
        witness=getattr(engine,'_cranial_stage_guard_ids',None)
        if witness is None:engine._cranial_stage_guard_ids=engine._protected_ids
        elif witness is not engine._protected_ids:_fail('PROTECTION_MEMBERSHIP_CHANGED')
        return contract
    raw=contract.as_dict() if isinstance(contract,CranialStageContract) else contract
    if type(raw) is not dict or raw.get('schema')!=SCHEMA:_fail('CONTRACT_SCHEMA')
    try:
        expected=_declaration(engine,membrane_cells=raw['membraneCells'],retained_membrane_cells=raw['retainedMembraneCells'],
            membrane_anchor_cells=raw['membraneAnchorCells'],work_cells=raw['workCells'],evidence=raw['evidence'],starting_exposure=raw['startingExposure'])
        bare=expected.as_dict();bare.pop('contentHash')
        from dataclasses import replace
        expected=replace(expected,content_hash=semantic_digest(bare))
        if raw!=expected.as_dict():_fail('CONTRACT_BINDING')
    except (KeyError,TypeError) as e:raise ValueError('CRANIAL_STAGE_CONTRACT_BINDING') from e
    return _register(engine,contract if isinstance(contract,CranialStageContract) else expected)


def _register(engine,contract):
    if engine.actions or engine.history:_fail('BIND_BEFORE_ACTIONS_REQUIRED')
    from scipy.ndimage import label
    protection=engine.protection
    guard=engine.guard_config
    _guard_config[id(contract)]=(guard,None if guard is None else
        ((type(guard.require_critical_coverage),guard.require_critical_coverage),
         (type(guard.critical_max_displacement_mm),guard.critical_max_displacement_mm)))
    _contracts[id(contract)]=contract
    _contract_fields[id(contract)]=(tuple((f.name,getattr(contract,f.name)) for f in fields(contract)),
        tuple((p,p.status,p.provenance,p.evidence_sha256) for _,p in contract.evidence))
    _source_facts[id(contract)]=(engine.scene._original_target_proof,protection,None if protection is None else
        (protection.clearance_mm,protection.required,protection.annotations,protection.cells))
    mask=np.zeros(engine.scene.shape,bool);mask.ravel()[list(contract.retained_membrane_cells)]=True
    groups,count=label(mask)
    covered=set(map(int,groups.ravel()[list(contract.membrane_anchor_cells)]));covered.discard(0)
    _support[id(contract)]=len(covered)==count
    normal_axis=contract.membrane_lift_axis;bone=engine._cranial_access
    normal=np.asarray(bone.entry.normal_ras_unit);center=np.asarray(bone.entry.center_ras_mm)
    inner=float(((engine.scene.points(contract.membrane_cells)-center)@normal).max())+.5*float(np.abs(engine.scene.affine[:3,:3].T@normal).sum())
    _geometry[id(contract)]=(normal_axis,contract.membrane_outward_sign,inner)
    engine._cranial_stage_guard_ids=engine._protected_ids
    _partitions[id(contract)]=MappingProxyType({'bone':frozenset(contract.bone_cells),'preparation':frozenset(contract.membrane_cells),
        'retained':frozenset(contract.retained_membrane_cells),'work':frozenset(contract.work_cells),'original_target':frozenset(contract.original_target_cells),
        'boundary':frozenset(contract.boundary_edges)})
    def cleanup(key):
        for cache in (_source_facts,_support,_initial_state,_partitions,_contract_fields,_geometry,_guard_config):cache.pop(key,None)
    weakref.finalize(contract,cleanup,id(contract))
    _initial_state[id(contract)]=semantic_digest({'scene':engine.scene.identity,'state':engine.state,
        **({'protectionContract':engine._protection_contract['contentHash']} if engine._protection_contract else {}),
        'cranialStagesContract':contract.content_hash})
    return contract


def require_evidence(engine,contract):
    if contract.starting_exposure!='conditional_post_scalp_exposure':_fail('INTACT_SCALP_STAGE_UNSUPPORTED')
    if any(p.status!='reviewed' or p.evidence_sha256 is None for _,p in contract.evidence):_fail('REVIEWED_STAGE_EVIDENCE_REQUIRED')
    if not _support.get(id(contract),False):_fail('RETAINED_RIM_SUPPORT_REQUIRED')
    if any(a['kind']!='soft' for a in engine.scene.properties['attachments'] if a['id'] in contract.attachment_ids):_fail('ATTACHMENT_KIND_UNSUPPORTED')
    if engine.state.get('toolMechanicsRevision')!=2 or engine.state.get('physicalClockRevision')!=1:_fail('MECHANICS2_CLOCK1_REQUIRED')
    return contract


def allowed_source_partitions(contract):
    return {'bone':contract.bone_cells,'preparation':contract.membrane_cells,'work':contract.work_cells,'original_target':contract.original_target_cells}


@dataclass(frozen=True)
class NativeBonePrefix:
    stage_hash:str
    position:int
    state_id:str
    actions_hash:str
    history_hash:str
    removed_bone_cells:tuple[int,...]
    opening_summary_json:str=''


def _history_current(engine):
    if len(engine.actions)!=len(engine.history):_fail('NATIVE_HISTORY_REQUIRED')
    contract=getattr(engine,'_cranial_stages',None)
    previous=_initial_state.get(id(contract))
    if previous is None:_fail('CANONICAL_STAGE_SOURCE_REQUIRED')
    for action,event in zip(engine.actions,engine.history):
        if event.get('action')!=action or event['stateBefore']!=previous:_fail('NATIVE_HISTORY_REQUIRED')
        previous=event['stateAfter']
    if previous!=engine.state_id:_fail('NATIVE_CURRENT_HISTORY_REQUIRED')


def prepare_transition(engine):
    """Derive the true complete-bone boundary from the current native prefix.

    Cache is private, registered by object identity and discarded on fresh/restore;
    canonical replay must derive it again before dependent membrane actions.
    """
    contract=getattr(engine,'_cranial_stages',None)
    if contract is None:return
    validate_stage_contract(engine,contract);_history_current(engine)
    if getattr(engine,'_cranial_stage_bone_prefix',None) is not None:
        verified_bone_prefix(engine,contract);_prepare_permission(engine,contract);return
    if not engine.actions:return
    from .surgical_bone_transport import describe
    status=describe(engine)
    if not status or not status['openingComplete']:return
    # Prior to the boundary no membrane/work removal or movement may be hidden.
    if set(engine.state['removed'])-set(contract.bone_cells):_fail('NONBONE_BEFORE_BONE_BOUNDARY')
    if engine.state.get('holds') or engine.state.get('detachedFragments'):_fail('BONE_BOUNDARY_WITHDRAWAL_REQUIRED')
    proof=NativeBonePrefix(contract.content_hash,len(engine.actions),engine.state_id,semantic_digest(engine.actions),
        semantic_digest(engine.history),tuple(sorted(set(engine.state['removed'])&set(contract.bone_cells))),
        json.dumps(status,sort_keys=True,separators=(',',':'),allow_nan=False))
    _prefixes[id(proof)]=proof
    _prefix_facts[id(proof)]=tuple((f.name,getattr(proof,f.name)) for f in fields(proof))
    weakref.finalize(proof,_prefix_facts.pop,id(proof),None)
    engine._cranial_stage_bone_prefix=proof
    _prepare_permission(engine,contract)


def verified_bone_prefix(engine,contract=None,*,full=True):
    contract=contract or getattr(engine,'_cranial_stages',None);proof=getattr(engine,'_cranial_stage_bone_prefix',None)
    if proof is not None and _prefixes.get(id(proof)) is proof:
        if any((getattr(proof,k) is not v if isinstance(v,tuple) else getattr(proof,k)!=v) for k,v in _prefix_facts[id(proof)]):_fail('DERIVED_BONE_PREFIX_CHANGED')
    if (contract is None or proof is None or _prefixes.get(id(proof)) is not proof or proof.stage_hash!=contract.content_hash or
        not 0<proof.position<=len(engine.actions) or len(engine.history)<proof.position or
        engine.history[proof.position-1]['stateAfter']!=proof.state_id or
        full and (semantic_digest(engine.actions[:proof.position])!=proof.actions_hash or semantic_digest(engine.history[:proof.position])!=proof.history_hash) or
        not set(contract.bone_cells)<=set(proof.removed_bone_cells)):
        _fail('ACCEPTED_BONE_PREFIX_REQUIRED')
    if set(proof.removed_bone_cells)-set(engine.state['removed']):_fail('BONE_REMOVAL_REGRESSED')
    return proof



def rebind_checkpoint_runtime(engine):
    """Rebind derived runtime after an authenticated state/ancestry projection.

    A retained bone proof may witness an earlier native prefix of this state.
    An endpoint shell's future proof cannot travel backward with state-only
    checkpoints. Permissions always derive again at the current native boundary.
    This helper never changes physical state or creates a new permission/proof.
    """
    contract=getattr(engine,'_cranial_stages',None)
    if contract is None:return
    validate_stage_contract(engine,contract)
    proof=getattr(engine,'_cranial_stage_bone_prefix',None)
    if proof is not None:
        if _prefixes.get(id(proof)) is not proof:_fail('DERIVED_BONE_PREFIX_CHANGED')
        if any((getattr(proof,k) is not v if isinstance(v,tuple) else getattr(proof,k)!=v) for k,v in _prefix_facts[id(proof)]):_fail('DERIVED_BONE_PREFIX_CHANGED')
        if proof.position>len(engine.actions):engine._cranial_stage_bone_prefix=None
        else:verified_bone_prefix(engine,contract)
    engine._cranial_stage_permission=None


def stage_readiness(engine):
    if getattr(engine, '_resident', None) is not None:
        from .surgical_native_engine import query
        native = query(engine, 'cranialStages')
        if native is not NotImplemented:return native
    contract=getattr(engine,'_cranial_stages',None)
    if contract is None:return None
    result={'schema':SCHEMA,'contractHash':contract.content_hash,'stateId':engine.state_id,'sceneId':engine.scene.identity,
        'phase':'BOUND','assessmentCurrent':False,'startingExposure':contract.starting_exposure,
        'sourceUnknowns':['Uncalibrated membrane mechanics','Clinical outcomes unknown'],
        'patchCells':len(contract.membrane_cells),'requiredBoundaryEdges':len(contract.boundary_edges),'cutBoundaryEdges':0,
        'boundaryCutComplete':False,'patchHeld':False,'patchDetached':False,'patchExtracted':False,'bonePrefixVerified':False,'duraCleared':False,
        'workEntrySupported':False,'wholeExitComplete':False,'clinicalValidation':'none','reason':None}
    try:
        validate_stage_contract(engine,contract)
        cut={tuple(sorted(e)) for e in engine.state.get('cutEdges',[])}
        result['cutBoundaryEdges']=len(source_partition_sets(contract)['boundary']&cut)
        result['boundaryCutComplete']=result['cutBoundaryEdges']==result['requiredBoundaryEdges']
        result['patchHeld']=any(h.get('kind')=='membrane-patch-v1' and h.get('contractHash')==contract.content_hash for h in engine.state['holds'].values())
        result['patchDetached']=any(h.get('kind')=='membrane-patch-v1' and h.get('contractHash')==contract.content_hash for h in engine.state.get('detachedFragments',[]))
        result['sourceUnknowns'] += [name+' evidence '+p.status for name,p in contract.evidence if p.status!='reviewed']
        require_evidence(engine,contract);_history_current(engine);result['assessmentCurrent']=True
        prepare_transition(engine);verified_bone_prefix(engine,contract)
        result['bonePrefixVerified']=True
        from .surgical_membrane_transport import clearance
        membrane=clearance(engine,contract);result['membrane']=membrane;result['duraCleared']=membrane['cleared'];result['patchExtracted']=membrane['cleared']
        result['workEntrySupported']=membrane['cleared']
        from .surgical_cranial_guards import tool_is_exterior
        from .surgical_task_domain import _posed
        result['wholeExitComplete']=membrane['cleared'] and not engine.state['holds'] and not engine.state.get('detachedFragments') and all(
            not s['inserted'] and not s['active'] and not s['grip'] and (not _posed(engine,k) or tool_is_exterior(engine,k)) for k,s in engine.state['slots'].items())
        if not membrane['cleared']:result['reason']='MEMBRANE_CLEARANCE_REQUIRED'
    except ValueError as e:result['reason']=str(e)
    result['phase']=('WITHDRAWN' if result['wholeExitComplete'] else 'EXTRACTED' if result['patchExtracted'] else
        'RELEASED_IN_FIELD' if result['patchDetached'] else 'HELD' if result['patchHeld'] else
        'CUT_RETAINED' if result['boundaryCutComplete'] else 'MEMBRANE_CUTTING' if result['cutBoundaryEdges'] else
        'MEMBRANE_READY' if result['bonePrefixVerified'] else 'BONE_WORK' if engine.actions else 'BOUND')
    return result


def source_partition_sets(contract):
    """Immutable prevalidated sets; no million-cell reconstruction in microsteps."""
    if _contracts.get(id(contract)) is not contract:_fail('VALIDATED_CONTRACT_REQUIRED')
    return _partitions[id(contract)]


@dataclass(frozen=True)
class NativeStagePermission:
    stage_hash:str
    position:int
    state_id:str
    dura_cleared:bool

_permissions=weakref.WeakValueDictionary()
_permission_facts={}

def _prepare_permission(engine,contract):
    from .surgical_membrane_transport import clearance
    status=clearance(engine,contract)
    permission=NativeStagePermission(contract.content_hash,len(engine.actions),engine.state_id,status['cleared'])
    _permissions[id(permission)]=permission
    _permission_facts[id(permission)]=tuple(getattr(permission,f.name) for f in fields(permission))
    weakref.finalize(permission,_permission_facts.pop,id(permission),None)
    engine._cranial_stage_permission=permission


def enforce(engine,key,operation,position,quaternion,working_cells=()):
    """Return True only when the new typed stage guard replaces bone-only entry.

    Original standalone Milling/flap enforcement remains the default until an
    actual complete bone prefix has been derived. This never grants from flags.
    """
    contract=getattr(engine,'_cranial_stages',None)
    if contract is None:return False
    validate_stage_contract(engine,contract);require_evidence(engine,contract)
    sets=source_partition_sets(contract)
    proof=getattr(engine,'_cranial_stage_bone_prefix',None)
    from .surgical_motion import envelopes,swept_envelopes
    from .surgical_tools import TOOLS
    if proof is None:
        # Even outside the rectangular entry, bone stage has no soft-work grant.
        for hand,slot in engine.state['slots'].items():
            contacts=set(working_cells) if hand==key else set()
            if hand!=key and slot['active']:
                for a,b,radius,part in envelopes(TOOLS[slot['instrument']],engine.parameters(slot),slot['position'],slot['quaternion']):
                    if part!='shaft':contacts.update(engine.touched(a,b,radius))
            if contacts-sets['bone']:_fail('NONBONE_BEFORE_BONE_BOUNDARY')
        return False
    verified_bone_prefix(engine,contract,full=False)
    permission=getattr(engine,'_cranial_stage_permission',None)
    if (permission is None or _permissions.get(id(permission)) is not permission or permission.stage_hash!=contract.content_hash or
        permission.position!=len(engine.actions) or not engine.history or permission.state_id!=engine.history[-1]['stateAfter']):
        _fail('NATIVE_STAGE_PREPARATION_REQUIRED')
    if tuple(getattr(permission,f.name) for f in fields(permission))!=_permission_facts[id(permission)]:_fail('DERIVED_STAGE_PERMISSION_CHANGED')
    from .surgical_cranial_guards import _inside_entry
    from .surgical_task_domain import _outside,_posed
    from scipy.spatial.transform import Rotation
    from types import SimpleNamespace
    bone=engine._cranial_access
    workspace=SimpleNamespace(minimum=bone.exit.minimum_voxel_boundary,maximum=bone.exit.maximum_voxel_boundary)
    normal=np.asarray(bone.entry.normal_ras_unit);center=np.asarray(bone.entry.center_ras_mm)
    mem_inner=_geometry[id(contract)][2]
    membrane_tools={'dural_scissors','dissector','membrane_forceps','frazier_suction'}
    work_tools={'cusa','brain_retractor','frazier_suction'}
    for hand,slot in engine.state['slots'].items():
        if not _posed(engine,hand) and (hand!=key or operation in ('exchange','inspect','deactivate')):continue
        tool=TOOLS[slot['instrument']];params=engine.parameters(slot)
        expected_depth=contract.scissors_depth_mm if slot['instrument']=='dural_scissors' else 2.
        family_valid=params['radiusMm']==tool['radiusMm'] and params['depthMm']==expected_depth
        end={'position':position,'quaternion':quaternion} if hand==key else slot
        segments=swept_envelopes(tool,params,slot,end) if hand==key else envelopes(tool,params,slot['position'],slot['quaternion'])
        if _outside(engine.scene,workspace,segments):_fail('OUTSIDE_REVIEWED_WORKSPACE')
        if _inside_entry(bone,segments):
            if not family_valid:_fail('FROZEN_STOCK_PARAMETER_FAMILY')
            axis=Rotation.from_quat(end['quaternion']).apply([0.,0.,1.])
            if axis@normal<np.cos(np.deg2rad(bone.entry.maximum_angle_degrees))-1e-9:_fail('ATTITUDE_OUTSIDE')
            allowed=membrane_tools|work_tools if permission.dura_cleared else membrane_tools
            if slot['instrument'] not in allowed:_fail('MEMBRANE_CLEARANCE_REQUIRED' if not permission.dura_cleared else 'TOOL_SCOPE')
            maximum=bone.entry.maximum_inward_mm if permission.dura_cleared else mem_inner
            if any(max(float((a-center)@normal),float((b-center)@normal))+radius>maximum+1e-9 for a,b,radius,_ in segments):_fail('BEYOND_STAGE_INNER_BOUNDARY')
        contacts=set(working_cells) if hand==key else set()
        if hand!=key and slot['active']:
            for a,b,radius,part in segments:
                if part!='shaft':contacts.update(engine.touched(a,b,radius))
        if not contacts:continue
        name=slot['instrument']
        if name in ('dural_scissors','dissector','membrane_forceps'):
            if contacts-(sets['preparation']|sets['retained']):_fail('MEMBRANE_TOOL_SOURCE_SCOPE')
        elif name in ('cusa','brain_retractor'):
            if not permission.dura_cleared:_fail('MEMBRANE_CLEARANCE_REQUIRED')
            if contacts-sets['work']:_fail('WORK_SOURCE_SCOPE')
        elif name=='frazier_suction':_fail('SUCTION_REQUIRES_RETAINED_VOID')
        else:_fail('TOOL_SOURCE_SCOPE')
    return True


def validate_effects(engine):
    """Native post-coupling source partition and retained-rim invariants."""
    contract=getattr(engine,'_cranial_stages',None)
    if contract is None:return
    sets=source_partition_sets(contract);removed=set(engine.state['removed']);rim=sets['retained']
    if removed-(sets['bone']|sets['preparation']|sets['work']):_fail('REMOVAL_SOURCE_SCOPE')
    if removed&rim:_fail('RETAINED_RIM_REMOVAL')
    if any(int(c) in rim and np.linalg.norm(v)>0 for c,v in engine.state.get('deformation',{}).items()):_fail('RETAINED_RIM_DEFORMATION')
    fragments=[*engine.state['holds'].values(),*engine.state.get('detachedFragments',[])]
    if any(rim&set(h['cells']) for h in fragments):_fail('RETAINED_RIM_HOLD')
    for field in ('cutEdges','transectedEdges'):
        for edge in engine.state.get(field,[]):
            e=tuple(sorted(edge))
            if set(e)&(sets['preparation']|rim) and e not in sets['boundary']:_fail('RETAINED_MEMBRANE_TOPOLOGY')
    allowed=set(contract.attachment_ids)
    allowed.update(a['id'] for a in engine.scene.properties['attachments'] if set(a['cells'])&sets['bone'] and a['kind'] in ('soft','periosteal'))
    if set(engine.state['separatedAttachments'])-allowed:_fail('ATTACHMENT_RELEASE_SOURCE_SCOPE')


def membrane_geometry(contract):
    if _contracts.get(id(contract)) is not contract:_fail('VALIDATED_CONTRACT_REQUIRED')
    return _geometry[id(contract)]


def bone_prefix_description(engine):
    """Display verified bone-only prefix plus actual current withdrawal state.

    This does not turn standalone Milling into a treatment contract, nor claim
    its bone-only history validator accepts later membrane/work actions.
    """
    from .surgical_milling import is_milling
    if not is_milling(engine._cranial_access) or getattr(engine,'_cranial_stage_bone_prefix',None) is None:return None
    contract=validate_stage_contract(engine,engine._cranial_stages)
    in_transition='_transition_check' in engine.__dict__
    if not in_transition:_history_current(engine)
    proof=verified_bone_prefix(engine,contract,full=not in_transition)
    result=json.loads(proof.opening_summary_json)
    if result.get('method')!='milling' or not result.get('openingComplete') or not result.get('apertureEvacuated'):_fail('BONE_PREFIX_DESCRIPTION_BINDING')
    from .surgical_cranial_guards import tool_is_exterior
    from .surgical_task_domain import _posed
    withdrawn=not engine.state['holds'] and not engine.state.get('detachedFragments') and all(
        not s['inserted'] and not s['active'] and not s['grip'] and (not _posed(engine,k) or tool_is_exterior(engine,k)) for k,s in engine.state['slots'].items())
    result.update(openingComplete=withdrawn,phase='MILLING_WITHDRAWN' if withdrawn else 'MILLING_CLEARED',
        nativePrefix={'position':proof.position,'stateId':proof.state_id,'historyScope':'accepted bone-only stage prefix; subsequent membrane/work separately checked'})
    result['milling'].update(openingComplete=withdrawn,phase='WITHDRAWN' if withdrawn else 'CLEARED',
        historyVerifiedScope='accepted_bone_stage_prefix',treatmentEntrySupported=False)
    return result
