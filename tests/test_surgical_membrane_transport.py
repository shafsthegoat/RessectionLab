"""Actual public-native joint opening controls on declared generated geometry."""
import copy
from dataclasses import replace
import numpy as np
import pytest
from scipy.spatial.transform import Rotation
from resectionlab.surgical_engine import SurgicalEngine
from resectionlab.surgical_cranial_stages import stage_readiness,prepare_transition
from resectionlab.surgical_membrane_transport import swept_cell_interior_intersects
from test_surgical_bone_transport import accepted,pose,request
from test_surgical_cranial_stages import stage_source


def joint_source(**kwargs):
    source,contract,contour,p,r=stage_source(**kwargs)
    engine=SurgicalEngine(source.scene,protection=source.protection,guard_config=source.guard_config,cranial_stages=contract)
    return engine,contract,contour,p,r


def bone_actions(contour):
    a=[request('configure',mechanicsRevision=2,clockRevision=1),request('place',pose=pose(26,26,55))]
    a += [request('apply',pose=p) for p in contour]
    a += [request('withdraw',pose=pose(26,26,55)),request('exchange',instrument='bone_flap_holder'),
        request('place',pose=pose(),parameters={'jawOpeningMm':0.}),request('grasp',pose=pose(z=59.5)),
        request('move',pose=pose(z=54.5)),request('extract'),request('withdraw',pose=pose(z=54.5))]
    return a


def membrane_cut_actions(engine,contract):
    """Finite source-edge inventory, fixed stock tool; no outcome-tuned poses."""
    a=[request('exchange',instrument='dural_scissors')]
    for i,j in contract.boundary_edges:
        u,v=engine.scene.points([i,j]);mid=(u+v)/2
        axis=int(np.argmax(np.abs(v-u)));q=Rotation.from_euler('z',90,degrees=True).as_quat().tolist() if axis==0 else [0.,0.,0.,1.]
        outside={'position':[float(mid[0]),float(mid[1]),55.],'quaternion':q}
        cutting={'position':[float(mid[0]),float(mid[1]),float(mid[2]-.2)],'quaternion':q}
        a += [request('place',pose=outside,parameters={'jawOpeningMm':0.,'depthMm':contract.scissors_depth_mm}),request('apply',pose=cutting),request('withdraw',pose=outside)]
    return a


def membrane_grasp_actions(hand='left'):
    return [request('exchange',slot=hand,instrument='membrane_forceps'),request('place',slot=hand,pose=pose(),parameters={'jawOpeningMm':0.}),request('grasp',slot=hand,pose=pose(z=63.5))]


def run(engine,actions):
    for a in actions:
        before=engine.snapshot();p=engine.preview(a,include_view=False);assert engine.snapshot()==before
        r=engine.step(a);assert r['status']=='accepted',r
        assert p['stateId']==r['stateId']
    return engine


@pytest.fixture(scope='module')
def cut_membrane():
    e,c,contour,p,r=joint_source();run(e,bone_actions(contour));assert e.observe()['craniotomy']['openingComplete']
    run(e,membrane_cut_actions(e,c));assert set(c.boundary_edges)<={tuple(x) for x in e.state['cutEdges']}
    assert not set(p)&set(e.state['removed'])
    return e,c,p,r


def test_complete_native_bone_cut_membrane_cut_transport_real_extraction_exit(cut_membrane):
    base,c,p,r=cut_membrane;e=base.clone();before=base.snapshot();assert not stage_readiness(e)['duraCleared']
    run(e,membrane_grasp_actions());assert set(e.state['holds']['left']['cells'])==set(p)
    run(e,[request('move',pose=pose(z=54.5))]);assert not set(p)&set(e.state['removed'])
    lifted=e.snapshot();restored=SurgicalEngine.restore(e.scene,lifted,use_cache=False);assert restored.snapshot()==lifted
    run(e,[request('extract'),request('withdraw',pose=pose(z=54.5))]);status=stage_readiness(e)
    assert status['duraCleared'] and status['workEntrySupported'] and status['wholeExitComplete']
    assert set(p)<=set(e.state['removed']) and not set(r)&set(e.state['removed'])
    assert e.metrics()['nonTargetRemovedMm3']==334. and e.metrics()['targetRemovedMm3']==0.
    assert e.state['membranePatchEvacuations'][0]['cells']==49
    assert SurgicalEngine.restore(e.scene,e.snapshot(),use_cache=False).snapshot()==e.snapshot()
    assert e.replay(e.actions).state_id==e.state_id and base.snapshot()==before


def test_cuts_keep_source_occupied_and_cusa_unqualified(cut_membrane):
    e,c,p,r=cut_membrane;e=e.clone();run(e,[request('exchange',instrument='cusa'),request('place',pose=pose())]);before=e.snapshot()
    for cell in p:assert e.occupancy()[cell]==cell+1 and e.scene.labels.flat[cell]==3
    a=request('apply',pose=pose(z=67.5),duration=1.)
    result=e.step(a);assert result['status']=='rejected' and result['reasonCode']=='CRANIAL_STAGE_MEMBRANE_CLEARANCE_REQUIRED';assert e.snapshot()==before


def test_same_affine_swept_cell_open_box_distinguishes_contact_and_penetration():
    assert not swept_cell_interior_intersects([0,0,0],[0,0,-2],[1,0,0])
    assert not swept_cell_interior_intersects([0,0,0],[0,0,-2],[1,1,0])
    assert swept_cell_interior_intersects([0,0,0],[1,0,0],[1,0,0])
    assert swept_cell_interior_intersects([0,0,0],[2,0,-1],[1,0,0])


@pytest.mark.parametrize('p',[pose(33.5,32,63.5),pose(z=65.)])
def test_initial_lateral_or_inward_lift_refuses_atomically(cut_membrane,p):
    e=cut_membrane[0].clone();run(e,membrane_grasp_actions());before=e.snapshot();a=request('move',pose=p)
    preview=e.preview(a,include_view=False);result=e.step(a)
    assert result['status']=='rejected' and preview['status']=='rejected' and e.snapshot()==before


def test_shaft_never_exempts_the_whole_patch(cut_membrane):
    from resectionlab.surgical_membrane_transport import check_sweep
    e=cut_membrane[0].clone();run(e,membrane_grasp_actions());before=e.snapshot()
    with pytest.raises(ValueError,match='MEMBRANE_PATCH_SHAFT_COLLISION'):check_sweep(e,'left',[0,0,0],pose(z=66.5))
    assert e.snapshot()==before


def test_release_retains_real_piece_and_cannot_unlock_work(cut_membrane):
    e=cut_membrane[0].clone();p=cut_membrane[2];run(e,membrane_grasp_actions());run(e,[request('move',pose=pose(z=61.5)),request('release')])
    assert len(e.state['detachedFragments'])==1 and not stage_readiness(e)['duraCleared']
    assert not set(p)&set(e.state['removed'])
    run(e,[request('grasp'),request('move',pose=pose(z=54.5)),request('extract'),request('withdraw',pose=pose(z=54.5))])
    assert stage_readiness(e)['wholeExitComplete'] and not e.state['detachedFragments']


def test_actual_opening_then_bounded_cusa_work_and_full_exit(cut_membrane):
    e,c,p,r=cut_membrane;e=e.clone();run(e,membrane_grasp_actions())
    run(e,[request('move',pose=pose(z=54.5)),request('extract'),request('withdraw',pose=pose(z=54.5)),
        request('exchange',instrument='cusa'),request('place',pose=pose())])
    work=request('apply',pose=pose(z=66.5),duration=30.)
    attempts=[]
    for _ in range(7):
        before=e.snapshot();preview=e.preview(work,include_view=False);assert e.snapshot()==before
        actual=e.step(work);attempts.append(actual)
        assert actual['status']==preview['status'] and actual['stateId']==preview['stateId']
        if actual['status']=='accepted':break
        assert actual['status']=='partial' and actual['reasonCode']=='TISSUE_ASPIRATION_IN_PROGRESS'
    assert attempts[-1]['status']=='accepted' and e.metrics()['targetRemovedMm3']>0
    run(e,[request('deactivate'),request('withdraw',pose=pose())])
    assert stage_readiness(e)['wholeExitComplete'] and e.metrics()['nonTargetRemovedMm3']==334.
    assert len(c.original_target_cells)==27 and set(r).isdisjoint(e.state['removed'])
    assert SurgicalEngine.restore(e.scene,e.snapshot(),use_cache=False).snapshot()==e.snapshot()


def test_right_hand_native_patch_transport_uses_same_whole_cell_guards(cut_membrane):
    e=cut_membrane[0].clone();run(e,[request('place',pose=pose(22,22,55))]);run(e,membrane_grasp_actions('right'))
    run(e,[request('move',slot='right',pose=pose(z=54.5)),request('extract',slot='right'),request('withdraw',slot='right',pose=pose(z=54.5))])
    assert stage_readiness(e)['wholeExitComplete'] and e.metrics()['nonTargetRemovedMm3']==334.


def test_whole_patch_detects_parked_other_tool_outside_holder_trajectory(cut_membrane):
    e=cut_membrane[0].clone();run(e,membrane_grasp_actions());run(e,[request('place',slot='right',pose=pose(35,32,57))]);before=e.snapshot()
    a=request('move',pose=pose(z=54.5));preview=e.preview(a,include_view=False);result=e.step(a)
    assert result['status']=='rejected' and preview['status']=='rejected'
    assert result['reasonCode']=='MEMBRANE_PATCH_OTHER_TOOL_COLLISION' and e.snapshot()==before


def test_moving_other_hand_checks_whole_parked_patch(cut_membrane):
    e=cut_membrane[0].clone();run(e,membrane_grasp_actions());run(e,[request('move',pose=pose(z=61.5)),
        request('exchange',slot='right',instrument='membrane_forceps'),request('place',slot='right',pose=pose(26,35,55),parameters={'jawOpeningMm':0.}),request('move',slot='right',pose=pose(26,35,61.5))])
    before=e.snapshot();a=request('move',slot='right',pose=pose(40,35,61.5));preview=e.preview(a,include_view=False);result=e.step(a)
    assert result['status']=='rejected' and preview['status']=='rejected'
    assert result['reasonCode']=='MEMBRANE_PATCH_MOVING_TOOL_COLLISION' and e.snapshot()==before


@pytest.mark.parametrize('protected',[False,True])
def test_whole_patch_obstacle_or_protection_outside_holder_path_refuses_atomically(protected):
    e,c,contour,p,r=joint_source(obstacle=(34,34,62),protected=protected)
    run(e,bone_actions(contour));run(e,membrane_cut_actions(e,c));run(e,membrane_grasp_actions());before=e.snapshot()
    a=request('move',pose=pose(z=54.5));preview=e.preview(a,include_view=False);result=e.step(a)
    assert result['status']=='rejected' and preview['status']=='rejected' and e.snapshot()==before
    assert result['reasonCode']==('PROTECTED_STRUCTURE' if protected else 'MEMBRANE_PATCH_SWEPT_ANATOMY_COLLISION')


def test_curved_thick_native_patch_cut_lift_extract_then_cusa_and_exit():
    e,c,contour,p,r=joint_source(curved=True)
    assert c.scissors_depth_mm==4. and len(p)==113 and len(c.original_target_cells)==27
    run(e,bone_actions(contour));run(e,membrane_cut_actions(e,c))
    assert not set(p)&set(e.state['removed']) and not stage_readiness(e)['duraCleared']
    run(e,membrane_grasp_actions());run(e,[request('move',pose=pose(z=54.5)),request('extract'),request('withdraw',pose=pose(z=54.5)),
        request('exchange',instrument='cusa'),request('place',pose=pose())])
    command=request('apply',pose=pose(z=66.5),duration=30.)
    for _ in range(7):
        preview=e.preview(command,include_view=False);actual=e.step(command)
        assert actual['status']==preview['status'] and actual['stateId']==preview['stateId']
        if actual['status']=='accepted':break
        assert actual['status']=='partial' and actual['reasonCode']=='TISSUE_ASPIRATION_IN_PROGRESS'
    assert actual['status']=='accepted' and e.metrics()['targetRemovedMm3']>0
    run(e,[request('deactivate'),request('withdraw',pose=pose())])
    assert stage_readiness(e)['wholeExitComplete'] and e.metrics()['nonTargetRemovedMm3']==398.
    assert set(r).isdisjoint(e.state['removed'])
    assert SurgicalEngine.restore(e.scene,e.snapshot(),use_cache=False).snapshot()==e.snapshot()


def test_actual_complete_cut_still_requires_declared_soft_attachment_release():
    e,c,contour,p,r=joint_source(attachment=True)
    run(e,bone_actions(contour));run(e,membrane_cut_actions(e,c))
    run(e,membrane_grasp_actions()[:-1]);before=e.snapshot();result=e.step(membrane_grasp_actions()[-1])
    assert result['status']=='rejected' and result['reasonCode']=='MEMBRANE_PATCH_ATTACHMENTS_NOT_RELEASED' and e.snapshot()==before
    run(e,[request('exchange',instrument='dissector'),request('place',pose=pose()),request('apply',pose=pose(z=63.3)),request('withdraw',pose=pose())])
    assert e.state['separatedAttachments']==['generated-soft-membrane-attachment']
    run(e,membrane_grasp_actions());run(e,[request('move',pose=pose(z=54.5)),request('extract'),request('withdraw',pose=pose(z=54.5))])
    assert stage_readiness(e)['wholeExitComplete'] and e.metrics()['nonTargetRemovedMm3']==334.


def test_uncut_source_patch_grasp_cannot_create_an_opening():
    e,c,contour,p,r=joint_source();run(e,bone_actions(contour));run(e,membrane_grasp_actions()[:-1]);before=e.snapshot()
    a=membrane_grasp_actions()[-1];preview=e.preview(a,include_view=False);result=e.step(a)
    assert result['status']=='rejected' and preview['status']=='rejected' and result['reasonCode']=='MEMBRANE_PATCH_COMPLETE_BOUNDARY_CUT_REQUIRED'
    assert e.snapshot()==before and not stage_readiness(e)['duraCleared']


def test_unknown_stage_evidence_has_no_native_physical_permission():
    e,c,contour,p,r=joint_source(unknown=True)
    run(e,[request('configure',mechanicsRevision=2,clockRevision=1)]);before=e.snapshot()
    result=e.step(request('place',pose=pose()))
    assert result['status']=='rejected' and result['reasonCode']=='CRANIAL_STAGE_REVIEWED_STAGE_EVIDENCE_REQUIRED' and e.snapshot()==before


def test_registered_bone_prefix_fields_cannot_be_rewritten_in_place(cut_membrane):
    e=cut_membrane[0].clone();prepare_transition(e);proof=e._cranial_stage_bone_prefix;old=proof.position
    object.__setattr__(proof,'position',old+1)
    try:
        before=e.snapshot();result=e.step(request('inspect'))
        assert result['status']=='rejected' and result['reasonCode']=='CRANIAL_STAGE_DERIVED_BONE_PREFIX_CHANGED' and e.snapshot()==before
    finally:object.__setattr__(proof,'position',old)


def test_registered_stage_permission_cannot_promote_retained_membrane(cut_membrane):
    from resectionlab.surgical_cranial_stages import enforce
    e=cut_membrane[0].clone();prepare_transition(e);permission=e._cranial_stage_permission;assert not permission.dura_cleared
    object.__setattr__(permission,'dura_cleared',True)
    try:
        with pytest.raises(ValueError,match='DERIVED_STAGE_PERMISSION_CHANGED'):enforce(e,'left','inspect',e.state['slots']['left']['position'],e.state['slots']['left']['quaternion'])
    finally:object.__setattr__(permission,'dura_cleared',False)


def test_curved_external_stage_case_save_open_binds_original_targets_and_replays(tmp_path):
    from resectionlab.core import CaseData,SourceRef,thaw_json
    from resectionlab.imaging import save_case,load_case,read_case_artifacts
    from resectionlab.surgical_scene import from_case
    from resectionlab.surgical_craniotomy import PROPERTY,binding,contract_hash
    from resectionlab.surgical_cranial_stages import freeze_cranial_stage_contract
    original,c,contour,p,r=joint_source(curved=True)
    properties=copy.deepcopy(original.scene.properties);bone=properties.pop(PROPERTY);mapping=original.scene.regions
    metadata={'surgical_properties':properties,'scope':'Generated curved native stage serialization; no patient evidence',
        'anatomy_layers':{row['name']:{'kind':{'bone':'skull','tumor':'tumor','membrane':'dura'}[row['material']]} for row in mapping}}
    base=CaseData(case_id='generated-curved-membrane-stage-save-open',mri=np.zeros(original.scene.shape,np.float32),affine=np.eye(4),
        compartments={row['name']:original.scene.labels==row['id'] for row in mapping},metadata=metadata,
        source_refs=(SourceRef('generated','simulated:curved-membrane-stage',provenance='simulated'),))
    bone.update(binding(from_case(base,mapping)));bone['contentHash']=contract_hash(bone)
    metadata=thaw_json(base.metadata);metadata['surgical_properties'][PROPERTY]=bone;case=replace(base,metadata=metadata)
    scene=from_case(case,mapping);source=SurgicalEngine(scene)
    contract=freeze_cranial_stage_contract(source,membrane_cells=p,retained_membrane_cells=r,membrane_anchor_cells=c.membrane_anchor_cells,
        work_cells=c.work_cells,evidence=c.as_dict()['evidence'])
    e=SurgicalEngine(scene,cranial_stages=contract);run(e,bone_actions(contour));run(e,membrane_cut_actions(e,contract));run(e,membrane_grasp_actions())
    run(e,[request('move',pose=pose(z=54.5)),request('extract'),request('withdraw',pose=pose(z=54.5))])
    before=e.snapshot();path=save_case(case,tmp_path/'curved-stage.ressectionlab',artifacts={'stageSnapshot':before})
    reopened=load_case(path);restored=SurgicalEngine.restore(from_case(reopened,mapping),read_case_artifacts(path)['stageSnapshot'],use_cache=True)
    assert reopened.semantic_hash==case.semantic_hash and restored.snapshot()==before
    status=restored.observe()['cranialStages'];assert status['patchExtracted'] and status['wholeExitComplete'] and status['workEntrySupported']
    assert status['stateId']==restored.state_id and status['sceneId']==scene.identity
    assert contract.original_target_cells==tuple(map(int,np.flatnonzero(case.source_compartments['Generated target'])))
    assert e.snapshot()==before and len(contract.original_target_cells)==27 and case.semantic_hash==reopened.semantic_hash


def test_native_milling_membrane_descriptor_retains_verified_bone_prefix():
    from test_surgical_milling_native import milling_shell,mill_aperture,suction_aperture
    from resectionlab.surgical_milling import PROPERTY,binding,contract_hash,milling_readiness
    from resectionlab.surgical_scene import SurgicalScene
    from resectionlab.surgical_cranial_stages import freeze_cranial_stage_contract,PROOFS
    old,aperture=milling_shell();labels=old.scene.labels.copy();labels[25:40,25:40,64]=3
    p=tuple(int(np.ravel_multi_index((x,y,64),labels.shape)) for x in range(31,34) for y in range(31,34))
    r=tuple(sorted(set(map(int,np.flatnonzero(labels==3)))-set(p)))
    regions=old.scene.regions+[{'id':3,'name':'Generated milling membrane','material':'membrane'}]
    properties=copy.deepcopy(old.scene.properties);raw=properties.pop(PROPERTY)
    base=SurgicalScene(labels,np.eye(4),regions,old.scene.source,properties);raw.update(binding(base));raw['contentHash']=contract_hash(raw);properties[PROPERTY]=raw
    scene=SurgicalScene(labels,np.eye(4),regions,old.scene.source,properties);source=SurgicalEngine(scene)
    proof={'status':'reviewed','provenance':'Generated finite milling and membrane descriptor control; no clinical calibration','evidenceSha256':'3'*64}
    c=freeze_cranial_stage_contract(source,membrane_cells=p,retained_membrane_cells=r,membrane_anchor_cells=(r[0],),
        work_cells=tuple(map(int,np.flatnonzero(labels==2))),evidence={k:dict(proof) for k in PROOFS})
    e=SurgicalEngine(scene,cranial_stages=c);mill_aperture(e);suction_aperture(e)
    assert e.observe()['craniotomy']['openingComplete']
    run(e,membrane_cut_actions(e,c));run(e,membrane_grasp_actions());run(e,[request('move',pose=pose(z=54.5)),request('extract'),request('withdraw',pose=pose(z=54.5))])
    assert not milling_readiness(e)['openingComplete'],'standalone bone-only full-history validator remains strict'
    view=e.observe();bone=view['craniotomy']
    assert bone['method']=='milling' and bone['phase']=='MILLING_WITHDRAWN' and bone['milledBoneMm3']==81.
    assert bone['milling']['debrisRemovedMm3']==81. and bone['milling']['debrisPresentMm3']==0.
    assert bone['milling']['historyVerifiedScope']=='accepted_bone_stage_prefix' and bone['milling']['treatmentEntrySupported'] is False
    assert bone['nativePrefix']['position']<len(e.actions) and view['cranialStages']['workEntrySupported']
    assert e.metrics()['nonTargetRemovedMm3']==90. and e.metrics()['targetRemovedMm3']==0.
    restored=SurgicalEngine.restore(scene,e.snapshot(),use_cache=False);assert restored.snapshot()==e.snapshot()
    assert restored.observe()['craniotomy']==bone
