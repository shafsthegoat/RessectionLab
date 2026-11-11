"""Generated source-bound stage contracts; no patient/clinical qualification."""
import copy
from dataclasses import replace
from types import SimpleNamespace
import weakref
import numpy as np
import pytest
from resectionlab import surgical_cranial_stages as stages
from resectionlab.surgical_craniotomy import PROPERTY,binding,contract_hash
from resectionlab.surgical_engine import SurgicalEngine
from resectionlab.surgical_scene import SurgicalScene
from test_surgical_bone_transport import shell


def stage_source(*,unknown=False,attachment=False,anchors=True,obstacle=None,protected=False,curved=False):
    engine,contour,_,_=shell(obstacle=obstacle)
    membrane_id=max(x['id'] for x in engine.scene.regions)+1
    labels=engine.scene.labels.copy();labels[25:40,25:40,64]=membrane_id
    if curved:
        labels[labels==membrane_id]=0
        for x in range(25,40):
            for y in range(25,40):
                if 29<=x<36 and 29<=y<36:
                    height=64+max(0,max(abs(x-32),abs(y-32))-1)
                    labels[x,y,64:height+1]=membrane_id
                else:labels[x,y,66]=membrane_id
    p=tuple(int(np.ravel_multi_index((x,y,z),labels.shape)) for x in range(29,36) for y in range(29,36) for z in range(64,67) if labels[x,y,z]==membrane_id)
    all_mem=set(map(int,np.flatnonzero(labels==membrane_id)));r=tuple(sorted(all_mem-set(p)))
    regions=engine.scene.regions+[{'id':membrane_id,'name':'Generated membrane surrogate','material':'membrane'}]
    props=copy.deepcopy(engine.scene.properties);raw=props.pop(PROPERTY)
    if attachment:props['attachments'].append({'id':'generated-soft-membrane-attachment','kind':'soft','cells':[p[len(p)//2]]})
    base=SurgicalScene(labels,np.eye(4),regions,engine.scene.source,props)
    raw.update(binding(base));raw['contentHash']=contract_hash(raw);props[PROPERTY]=raw
    scene=SurgicalScene(labels,np.eye(4),regions,engine.scene.source,props)
    if protected:
        from resectionlab.surgical_rl_protection import ProtectionAnnotation,ProtectionSet,GuardConfig
        annotation=ProtectionAnnotation.from_mask(scene,'generated-protected-obstacle',labels==3,provenance='Generated hard exclusion outside the membrane holder path; no patient coverage')
        protection=ProtectionSet(scene,[annotation],clearance_mm=0.,required=())
        source=SurgicalEngine(scene,protection=protection,guard_config=GuardConfig(require_critical_coverage=False))
    else:source=SurgicalEngine(scene)
    proof={'status':'unknown' if unknown else 'reviewed','provenance':'Generated categorical membrane transport control, not clinical dura/scalp evidence','evidenceSha256':None if unknown else '2'*64}
    evidence={k:copy.deepcopy(proof) for k in stages.PROOFS}
    w=tuple(map(int,np.flatnonzero(labels==2)))
    contract=stages.freeze_cranial_stage_contract(source,membrane_cells=p,retained_membrane_cells=r,membrane_anchor_cells=(r[0],) if anchors else (),work_cells=w,evidence=evidence)
    return source,contract,contour,p,r


def test_external_stage_binding_preserves_bone_source_and_original_target():
    source,c,_,p,r=stage_source();before=source.snapshot()
    assert stages.validate_stage_contract(source,c) is c
    assert stages.validate_stage_contract(source,c.as_dict()).as_dict()==c.as_dict()
    assert source._cranial_access.content_hash==c.bone_contract_hash and c.source_scene_id==source.scene.identity
    sets=stages.source_partition_sets(c)
    assert sets['preparation']==set(p) and sets['retained']==set(r)
    assert sets['original_target']<=sets['work'] and not sets['preparation']&sets['original_target']
    assert stages.allowed_source_partitions(c)['original_target']==c.original_target_cells
    assert source.snapshot()==before and source.actions==[]
    with pytest.raises(TypeError):sets['work']=frozenset()


def test_cached_validation_never_rebuilds_million_member_declaration(monkeypatch):
    source,c,*_=stage_source()
    monkeypatch.setattr(stages,'_declaration',lambda *a,**k:(_ for _ in ()).throw(AssertionError('hot rebuild')))
    monkeypatch.setattr(stages,'_original_targets',lambda *a:(_ for _ in ()).throw(AssertionError('hot target scan')))
    for _ in range(4):assert stages.validate_stage_contract(source,c) is c


@pytest.mark.parametrize('field',['membrane_cells','work_cells','content_hash','source_binding','scissors_depth_mm','membrane_lift_axis'])
def test_same_identity_frozen_field_replacement_cannot_use_fast_cache(field):
    source,c,*_=stage_source();value=getattr(c,field)
    changed=value[:-1] if isinstance(value,tuple) else ('changed' if isinstance(value,str) else value+1)
    object.__setattr__(c,field,changed)
    try:
        with pytest.raises(ValueError,match='FROZEN_CONTRACT_CHANGED'):stages.validate_stage_contract(source,c)
    finally:object.__setattr__(c,field,value)


def test_nested_evidence_cannot_change_behind_registered_tuple_identity():
    source,c,*_=stage_source();proof=c.evidence[0][1];before=proof.status
    object.__setattr__(proof,'status','unknown')
    try:
        with pytest.raises(ValueError,match='FROZEN_EVIDENCE_CHANGED'):stages.validate_stage_contract(source,c)
    finally:object.__setattr__(proof,'status',before)


def test_unknown_proofs_inspectable_but_not_runtime_permission():
    source,c,*_=stage_source(unknown=True)
    assert stages.validate_stage_contract(source,c) is c
    with pytest.raises(ValueError,match='REVIEWED_STAGE_EVIDENCE_REQUIRED'):stages.require_evidence(source,c)


def test_retained_rim_anchor_unknown_does_not_become_support():
    source,c,*_=stage_source(anchors=False)
    with pytest.raises(ValueError,match='RETAINED_RIM_SUPPORT_REQUIRED'):stages.require_evidence(source,c)


def test_source_edit_invalidates_cached_contract():
    source,c,*_=stage_source();source.scene.properties['unversioned-edit']=True
    with pytest.raises(ValueError,match='Source scene changed'):stages.validate_stage_contract(source,c)


def test_tissue_or_target_cannot_be_relabelled_as_nonbone_preparation():
    source,c,*_=stage_source();raw=c.as_dict();raw['membraneCells']+=list(c.original_target_cells)
    raw.pop('contentHash');raw['contentHash']=stages.semantic_digest(raw)
    with pytest.raises(ValueError,match='MEMBRANE_PARTITION_REQUIRED'):stages.validate_stage_contract(source,raw)


def test_forged_bone_prefix_is_not_an_accepted_native_boundary():
    source,c,*_=stage_source();source._cranial_stages=c
    source._cranial_stage_bone_prefix=stages.NativeBonePrefix(c.content_hash,0,source.state_id,'x','x',c.bone_cells)
    with pytest.raises(ValueError,match='ACCEPTED_BONE_PREFIX_REQUIRED'):stages.verified_bone_prefix(source,c)


def test_curved_source_derived_scissor_family_is_part_of_frozen_contract():
    source,c,*_=stage_source(curved=True)
    assert c.scissors_depth_mm==4. and c.membrane_lift_axis==2 and c.membrane_outward_sign==-1
    assert len(c.membrane_cells)==113 and len(c.original_target_cells)==27
    raw=c.as_dict();raw['scissorsDepthMm']=2.;raw.pop('contentHash');raw['contentHash']=stages.semantic_digest(raw)
    with pytest.raises(ValueError,match='CONTRACT_BINDING'):stages.validate_stage_contract(source,raw)


def test_opt_in_native_observation_binds_stage_identity_and_unknowns_without_state_change():
    source,c,*_=stage_source(unknown=True)
    assert 'cranialStages' not in source.observe()
    stage=SurgicalEngine(source.scene,cranial_stages=c);before=stage.snapshot();v=stage.observe()['cranialStages']
    assert v['stateId']==stage.state_id and v['sceneId']==stage.scene.identity and v['phase']=='BOUND'
    assert not v['assessmentCurrent'] and not v['workEntrySupported'] and not v['patchExtracted']
    assert any('evidence unknown' in x for x in v['sourceUnknowns']) and stage.snapshot()==before


@pytest.mark.parametrize('change',['displacement','coverage-type','object'])
def test_cached_stage_binds_actual_live_guard_policy_not_only_its_prior_hash(change):
    source,c,*_=stage_source(obstacle=(34,34,62),protected=True);guard=source.guard_config
    if change=='displacement':object.__setattr__(guard,'critical_max_displacement_mm',1.)
    elif change=='coverage-type':object.__setattr__(guard,'require_critical_coverage',0)
    else:source.guard_config=replace(guard)
    try:
        with pytest.raises(ValueError,match='PROTECTION_CONFIG_CHANGED'):stages.validate_stage_contract(source,c)
    finally:
        source.guard_config=guard
        object.__setattr__(guard,'critical_max_displacement_mm',0.)
        object.__setattr__(guard,'require_critical_coverage',False)


def _effects_lifetime_case():
    partitions={'bone':frozenset({1}),'preparation':frozenset({2}),
        'retained':frozenset({3}),'work':frozenset({4}),'boundary':frozenset({(2,3)})}
    engine=SimpleNamespace(_cranial_stages=SimpleNamespace(attachment_ids=()),
        state={'removed':[],'holds':{},'cutEdges':[(2,3)],'transectedEdges':[],
            'separatedAttachments':[]},scene=SimpleNamespace(properties={'attachments':[]}))
    return engine,partitions


def test_effects_replaced_preparation_finalizes_before_remaining_validation(monkeypatch):
    engine,partitions=_effects_lifetime_case();events=[]
    class Partitions(dict):
        def __getitem__(self,key):
            if key=='boundary':
                events.append('boundary-enter')
                self['preparation']=frozenset({4})
                events.append('boundary-exit')
            return super().__getitem__(key)
    partitions=Partitions(partitions)
    def finalized():
        events.append('preparation-finalized')
        engine.state['separatedAttachments'].append('unbound')
    finalizer=weakref.finalize(partitions['preparation'],finalized)
    monkeypatch.setattr(stages,'source_partition_sets',lambda contract:partitions)
    with pytest.raises(ValueError,match='^CRANIAL_STAGE_ATTACHMENT_RELEASE_SOURCE_SCOPE$'):
        stages.validate_effects(engine)
    assert events==['boundary-enter','preparation-finalized','boundary-exit']
    assert not finalizer.alive and engine.state['separatedAttachments']==['unbound']


def test_effects_union_temporary_finalizes_before_boundary_and_remaining_validation(monkeypatch):
    engine,partitions=_effects_lifetime_case();events=[]
    def finalized():
        events.append('union-finalized')
        engine.state['separatedAttachments'].append('unbound')
    class Preparation(frozenset):
        def __or__(self,other):
            union=super().__or__(other)
            weakref.finalize(union,finalized)
            return union
    class Partitions(dict):
        def __getitem__(self,key):
            if key=='boundary':events.append('boundary-read')
            return super().__getitem__(key)
    partitions=Partitions(partitions);partitions['preparation']=Preparation({2})
    monkeypatch.setattr(stages,'source_partition_sets',lambda contract:partitions)
    with pytest.raises(ValueError,match='^CRANIAL_STAGE_ATTACHMENT_RELEASE_SOURCE_SCOPE$'):
        stages.validate_effects(engine)
    assert events==['union-finalized','boundary-read']
    assert engine.state['separatedAttachments']==['unbound']
