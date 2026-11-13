"""Generated software regressions; explicit masks are not patient validation."""
import copy
import json

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from resectionlab.core import CaseData, SourceRef, semantic_digest, array_digest
from resectionlab.desktop_bridge import BridgeSession, _Request
from resectionlab.surgical_engine import SurgicalEngine
from resectionlab.surgical_rl_protection import GuardConfig, ProtectionAnnotation, ProtectionSet
from resectionlab.surgical_scene import SurgicalScene, from_case
from resectionlab.surgical_tools import TOOLS


def source(*, flag='protected', affine=None, frame='RAS+'):
    shape = (32, 32, 32)
    critical = np.zeros(shape, bool); critical[12, 12, 12] = True
    target = critical.copy(); target[12, 12, 6] = True
    layer = {'kind': 'brain', 'structure_id': 'declared-exclusion'}
    if flag: layer[flag] = True
    return CaseData(case_id='GENERATED-declared-exclusion', mri=np.zeros(shape, np.float32),
        compartments={'critical-source': critical, 'target': target},
        affine=np.eye(4) if affine is None else affine, frame=frame,
        source_refs=(SourceRef('generated', 'synthetic://declared-exclusion', native_frame=frame,
                               provenance='simulated'),),
        metadata={'is_synthetic': True, 'anatomy_layers': {
            'critical-source': layer, 'target': {'kind': 'tumor'}}})


def pose(position, quaternion=(0., 0., 0., 1.)):
    return {'position': list(position), 'quaternion': list(quaternion)}


def action(operation, position=None, *, hand='right', quaternion=(0., 0., 0., 1.), **extra):
    return {'slot': hand, 'operation': operation, 'duration': .1,
            **({'pose': pose(position, quaternion)} if position is not None else {}), **extra}


def accepted(engine, command):
    result = engine.step(command)
    assert result['status'] == 'accepted', result
    return result


@pytest.mark.parametrize('flag', ['protected', 'critical'])
def test_explicit_membership_survives_target_ownership_and_aborts_complete_proposal(flag):
    case = source(flag=flag); scene = from_case(case); engine = SurgicalEngine(scene)
    assert scene.labels[12, 12, 12] == next(r['id'] for r in scene.regions if r['target'])
    assert int(np.ravel_multi_index((12, 12, 12), scene.shape)) in engine._protected_ids
    assert engine.guard_report()['missingRequired'] == ['brainstem']
    accepted(engine, action('place', [12., 12., -5.]))
    before = engine.snapshot(); command = action('apply', [12., 12., 12.])
    preview = engine.preview(command); result = engine.step(command)
    assert result['status'] == 'rejected' and result['reasonCode'] == 'PROTECTED_STRUCTURE'
    assert result['deltas'] == {} and result['elapsedSeconds'] == 0
    assert preview['stateId'] == result['stateId'] == before['stateId']
    assert result['metrics']['targetRemovedMm3'] == 0
    assert engine.snapshot() == before and not engine.state['removed']
    # A smaller safe proposal can be explicitly accepted; rejecting the next
    # unsafe proposal preserves that already accepted treatment and its history.
    accepted(engine, action('apply', [12., 12., 6.]))
    prefix = engine.snapshot()
    assert engine.metrics()['targetRemovedMm3'] == 1
    assert engine.step(command)['status'] == 'rejected'
    assert engine.snapshot() == prefix
    restored = SurgicalEngine.restore(scene, json.loads(json.dumps(prefix)))
    assert restored.snapshot() == prefix
    assert restored.branch(len(restored.actions)).snapshot() == prefix
    assert restored.step(command)['status'] == 'rejected'


def test_unmarked_display_name_is_not_anatomical_evidence_and_old_scene_identity_is_unchanged():
    case = source(flag=None); scene = from_case(case)
    assert 'protectedAnatomyV1' not in scene.source
    expected = semantic_digest({'labels': array_digest(scene.labels),
        'affine': scene.affine.tolist(), 'regions': scene.regions, 'source': scene.source, 'properties': scene.properties})
    assert scene.identity == expected
    engine = SurgicalEngine(scene)
    assert not engine._protected_ids and engine.guard_report()['status'] == 'unknown-coverage'
    assert engine.snapshot().get('hardConstraintPolicy') is None


def test_mapping_declarations_and_supplemental_contract_cannot_erase_masks():
    case = source(flag=None)
    mapping = [{'id': 1, 'name': 'critical-source', 'material': 'soft_tissue', 'target': False,
                'protected': True, 'structureId': 'explicit-id'},
               {'id': 2, 'name': 'target', 'material': 'tumor', 'target': True}]
    scene = from_case(case, mapping)
    empty = ProtectionSet(scene, (), clearance_mm=0., required=())
    engine = SurgicalEngine(scene, protection=empty, guard_config=GuardConfig(require_critical_coverage=False))
    assert len(engine._protected_ids) == 1
    assert engine.protection.annotations[0].structure_id == 'explicit-id'
    # Direct scene declarations also protect owned labels.
    direct = SurgicalScene(scene.labels, scene.affine,
        [{'id': 1, 'name': 'unused', 'material': 'soft_tissue'},
         {'id': 2, 'name': 'arbitrary-label', 'material': 'tumor', 'critical': True}], {}, {})
    assert len(SurgicalEngine(direct)._protected_ids) == 2


def test_drill_cannot_remove_tumor_even_without_anatomical_exclusions():
    engine = SurgicalEngine(from_case(source(flag=None)))
    accepted(engine, action('place', [12., 12., -5.], hand='left'))
    result = engine.step(action('apply', [12., 12., 12.], hand='left'))
    assert result['status'] in ('partial', 'rejected')
    assert result['reasonCode'] == 'UNSUPPORTED_MATERIAL'
    assert engine.state['removed'] == []


@pytest.mark.parametrize('frame', ['RAS+', 'LPS+'])
def test_declared_mask_uses_exact_oblique_sheared_world_transform(frame):
    affine = np.eye(4); affine[:3, :3] = [[0., -1.2, .1], [1.3, .2, 0.], [0., 0., 1.5]]
    affine[:3, 3] = [100., -80., 40.]
    scene = from_case(source(affine=affine, frame=frame)); engine = SurgicalEngine(scene)
    cell = next(iter(engine._protected_ids)); point = scene.points([cell])[0]
    assert cell in engine.protection.capsule_hits(point, point, .1)
    # A placement at an excluded cell is rejected transactionally in RAS mm.
    before = engine.snapshot()
    result = engine.step(action('place', point))
    assert result['status'] == 'rejected' and engine.snapshot() == before


def test_intermediate_rotation_shaft_other_hand_and_withdrawal_are_atomic():
    labels = np.zeros((64,64,64), np.uint16); labels[5,5,10] = 1
    scene = SurgicalScene(labels, np.eye(4), [dict(id=1, name='exclusion', material='tumor', protected=True)], {}, {})
    engine = SurgicalEngine(scene)
    tilt = Rotation.from_euler('x',90,degrees=True).as_quat()
    accepted(engine, action('place',[5.,5.,20.],hand='left',quaternion=tilt))
    end_tilt = Rotation.from_euler('x',-90,degrees=True).as_quat()
    parameters = engine.parameters(engine.state['slots']['left'])
    for q in (tilt,end_tilt):
        assert not len(engine.protection.sweep_hits(TOOLS['cranial_drill'],parameters,
            pose([5.,5.,20.],q),pose([5.,5.,20.],q)))
    before = engine.snapshot()
    for op in ('move','withdraw'):
        result = engine.step(action(op,[5.,5.,20.],hand='left',quaternion=end_tilt))
        assert result['status'] == 'rejected' and result['reasonCode'] == 'PROTECTED_STRUCTURE'
        assert engine.snapshot() == before
    engine.state['slots']['right'].update(position=[5.,5.,20.],quaternion=[0.,0.,0.,1.])
    before = engine.snapshot()
    result=engine.step(action('inspect',hand='left'))
    assert result['status'] == 'rejected' and result['reasonCode'] == 'PROTECTED_STRUCTURE'
    assert engine.snapshot() == before


def test_legacy_guard_partial_replay_is_internal_and_live_continuation_is_atomic():
    scene=from_case(source()); engine=SurgicalEngine(scene)
    accepted(engine, action('place',[12.,12.,-5.]))
    # Generate the previous implementation's causal receipt as a historical
    # fixture. Only restore/branch may replay this interpretation.
    engine._restoring_legacy_guard_partial=True
    result=engine.step(action('apply',[12.,12.,12.]))
    del engine._restoring_legacy_guard_partial
    assert result['status']=='partial' and result['reasonCode']=='PROTECTED_STRUCTURE'
    old=engine.snapshot(); old.pop('hardConstraintPolicy'); old.pop('legacyGuardPartialActions')
    restored=SurgicalEngine.restore(scene,old)
    saved=restored.snapshot()
    assert saved['legacyGuardPartialActions']==[1]
    assert SurgicalEngine.restore(scene,saved).snapshot()==saved
    assert restored.branch(2).snapshot()==saved
    from resectionlab.surgical_replay_cache import SurgicalReplayCache
    replay=SurgicalReplayCache()
    assert replay.seek(restored,saved,0).actions==[]
    assert replay.seek(restored,saved,2).snapshot()==saved
    # Force a cache reconstruction from an earlier checkpoint.
    replay.seek(restored,saved,1)
    assert replay.seek(restored,saved,2).snapshot()==saved
    before=restored.snapshot()
    assert restored.step(action('apply',[12.,12.,12.]))['status']=='rejected'
    assert restored.snapshot()==before


def test_actual_desktop_create_step_save_cold_restore_retains_declared_overlap(tmp_path):
    session=BridgeSession(tmp_path/'transfers')
    case=source()
    try:
        key=session._install_case(case,{},_Request('install'))['caseHash']
        def call(command, previous=None, **extra):
            return session.execute('surgicalRehearsal', {'caseHash':key,'command':command,
                **({'expectedStateId':previous['view']['stateId']} if previous else {}),**extra},
                _Request(command),lambda *_:None)
        initial=call('create')
        assert initial['view']['protection']['structures'][0]['sourceCells']==1
        placed=call('step',initial,action=action('place',[12.,12.,-5.]))
        refused=call('step',placed,action=action('apply',[12.,12.,12.]))
        assert refused['outcome']['status']=='rejected'
        assert refused['outcome']['reasonCode']=='PROTECTED_STRUCTURE'
        assert refused['view']['stateId']==placed['view']['stateId']
        assert refused['view']['metrics']['targetRemovedMm3']==0
        destination=tmp_path/'GENERATED-protected.ressectionlab'
        session.execute('saveCase',{'caseHash':key,'path':str(destination),'workspace':{},'overwrite':False},
            _Request('save'),lambda *_:None)
    finally: session.close()
    cold=BridgeSession(tmp_path/'cold')
    try:
        loaded=cold.execute('loadCase',{'path':str(destination)},_Request('load'),lambda *_:None)
        opened=cold.execute('surgicalRehearsal',{'caseHash':loaded['caseHash'],'command':'inspect'},
            _Request('inspect'),lambda *_:None)
        assert opened['view']['stateId']==placed['view']['stateId']
        assert opened['view']['protection']==placed['view']['protection']
    finally: cold.close()


def test_checked_planner_skill_cannot_erase_declared_protection_or_gain_rejected_credit():
    from resectionlab.surgical_staged_skills import SkillLimits, checked_actions
    engine=SurgicalEngine(from_case(source()))
    accepted(engine,action('place',[12.,12.,-5.]))
    before=engine.snapshot()
    candidate,receipt=checked_actions(engine,[action('apply',[12.,12.,12.])],
        limits=SkillLimits(max_native_actions=20,max_public_calls=10))
    assert receipt['status']=='refused'
    assert receipt['admittedSkillProgressMm3']==0
    assert receipt['nativeActionsAppended']==0
    assert candidate.state_id==before['stateId']
    assert candidate.metrics()['targetRemovedMm3']==0
    assert engine.snapshot()==before


def test_reviewed_unknown_domain_is_enforced_by_direct_engine_and_planner_skill():
    from resectionlab.surgical_staged_skills import SkillLimits, checked_actions
    labels=np.zeros((80,80,80),np.uint16);labels[75,75,75]=1
    domain={'mode':'restrict-unknown','shape':list(labels.shape),
        'minimumVoxelBoundary':[5.,5.,10.],'maximumVoxelBoundary':[74.,74.,74.],
        'reviewEvidenceSha256':'a'*64,'maskSha256':'b'*64}
    scene=SurgicalScene(labels,np.eye(4),[dict(id=1,name='target',material='tumor',target=True)],
        {'adapter':{'reviewedTaskDomainV1':domain}}, {})
    engine=SurgicalEngine(scene);before=engine.snapshot()
    bad=action('place',[40.,40.,30.]) # Tip inside; shaft crosses unknown floor.
    result=engine.step(bad)
    assert result['status']=='rejected' and result['reasonCode']=='REVIEWED_TASK_DOMAIN_OUTSIDE'
    assert engine.snapshot()==before
    preview=engine.preview(bad)
    assert preview['stateId']==before['stateId'] and preview['status']=='rejected'
    candidate,receipt=checked_actions(engine,[bad],limits=SkillLimits(max_native_actions=10,max_public_calls=10))
    assert receipt['status']=='refused' and candidate.state_id==before['stateId']
    accepted(engine,action('place',[40.,40.,60.]))


@pytest.mark.parametrize('flag', ['protected','critical'])
def test_ambiguous_protection_declarations_are_rejected(flag):
    case=source(flag=None)
    from resectionlab.core import thaw_json
    metadata=thaw_json(case.metadata);metadata['anatomy_layers']['critical-source'][flag]='true'
    case=CaseData(case_id=case.case_id,mri=case.mri,compartments=case.compartments,affine=case.affine,
        source_refs=case.source_refs,metadata=metadata)
    with pytest.raises(ValueError,match='must be boolean'):from_case(case)


def test_original_target_membership_survives_later_protected_owner_and_standardization():
    from resectionlab.core import array_digest
    from resectionlab.surgical_environment import standardize
    from resectionlab.surgical_scene import original_target_memberships
    case=source()
    scene=from_case(case,[dict(id=1,name='target',material='tumor',target=True),
        dict(id=2,name='critical-source',material='soft_tissue',target=False,protected=True)])
    target=original_target_memberships(scene)['target']
    assert target.tolist()==np.flatnonzero(case.compartments['target']).tolist()
    assert len(target)==2 and np.count_nonzero(scene.labels==1)==1
    assert scene.source['annotationHashes']['target']==array_digest(case.compartments['target'])
    with pytest.raises(ValueError):target.flags.writeable=True
    standardized=standardize(scene,target_names=['target'])
    assert original_target_memberships(standardized)['target'] is target
    assert standardize(scene,target_names=['critical-source'])._original_target_memberships is None
    # The same source narrative cannot manufacture exact masks from labels.
    categorical=SurgicalScene(scene.labels,scene.affine,scene.regions,scene.source,scene.properties)
    assert categorical.identity==scene.identity
    assert original_target_memberships(categorical) is None
    assert len(SurgicalEngine(scene)._protected_ids)==1


def test_original_target_evidence_is_source_bound_and_distinct_target_overlap_is_explicit():
    from resectionlab.surgical_scene import original_target_memberships
    from resectionlab.core import thaw_json
    case=source(flag=None);metadata=thaw_json(case.metadata)
    metadata['anatomy_layers']['second-target']={'kind':'tumor'}
    case=CaseData(case_id=case.case_id,mri=case.mri,affine=case.affine,source_refs=case.source_refs,
        compartments={**case.compartments,'second-target':case.compartments['target']},metadata=metadata)
    scene=from_case(case);memberships=original_target_memberships(scene)
    assert len(memberships)==2 and len(np.unique(np.concatenate(list(memberships.values()))))==2
    with pytest.raises(TypeError):memberships['forged']=np.array([0])
    scene._original_target_memberships=dict(memberships)
    with pytest.raises(ValueError,match='membership binding changed'):scene.require_cache_source()
