"""Exact current-prefix derived stage runtime through real replay adapters."""
import pytest
from resectionlab.surgical_replay_cache import SurgicalReplayCache
from resectionlab.surgical_frame_replay import SurgicalFrameCache
from test_surgical_membrane_transport import joint_source,bone_actions,run

@pytest.fixture(scope='module')
def bone_endpoint():
    e,c,contour,p,r=joint_source();run(e,bone_actions(contour))
    assert e.observe()['cranialStages']['bonePrefixVerified']
    return e


def test_cached_rewind_then_observed_first_action_matches_native_endpoint(bone_endpoint):
    endpoint=bone_endpoint;saved=endpoint.snapshot();replay=SurgicalReplayCache();frames=SurgicalFrameCache()
    first,history=replay.seek(endpoint,saved,0,include_history=True)
    # The second0 seek takes the actual demanded state-only checkpoint path.
    repeated,_=replay.seek(first,saved,0,include_history=True)
    initial,_,info=frames.seek(repeated,saved,0,0,replay,verified_history=history)
    boundary,_,_=frames.seek(repeated,saved,0,info['frameCount']-1,replay,verified_history=history)
    assert initial.state_id==history[0]['stateBefore']
    assert boundary.state_id==history[0]['stateAfter']
    assert boundary.history==history[:1]
    assert endpoint.snapshot()==saved


def test_persistent_frame_admission_drops_future_proof_without_granting_entry(bone_endpoint,tmp_path):
    endpoint=bone_endpoint;saved=endpoint.snapshot();replay=SurgicalReplayCache();frames=SurgicalFrameCache(directory=tmp_path)
    admission=frames.admit(endpoint,saved,replay,verified_history=endpoint.history)
    _,_,info=frames.seek(endpoint,admission.snapshot,0,0,replay,admission=admission,verified_history=admission.history)
    boundary,_=frames.boundary(admission.snapshot,1,admission=admission,verified_history=admission.history)
    assert boundary._cranial_stage_bone_prefix is None and boundary._cranial_stage_permission is None
    status=boundary.observe()['cranialStages'];assert status['assessmentCurrent'] and not status['workEntrySupported']
    expected=endpoint.history[1]
    result=boundary.step(saved['actions'][1]);assert result['status']==expected['status'] and result['stateId']==expected['stateAfter']
    assert endpoint.snapshot()==saved


def test_projection_refuses_tampered_registered_proof_instead_of_dropping_it(bone_endpoint):
    endpoint=bone_endpoint.clone();proof=endpoint._cranial_stage_bone_prefix;old=proof.position
    saved=endpoint.snapshot();cache=SurgicalReplayCache();cache.seek(endpoint,saved,0)
    try:
        object.__setattr__(proof,'position',old+1)
        with pytest.raises(ValueError,match='DERIVED_BONE_PREFIX_CHANGED'):
            cache.seek(endpoint,saved,0)
    finally:object.__setattr__(proof,'position',old)


def test_optional_bounded_prefix_refuses_stage_and_keeps_canonical_identity(bone_endpoint):
    from resectionlab.surgical_prefix_replay import SurgicalPrefixReplay,_prefix_state_id
    endpoint=bone_endpoint;saved=endpoint.snapshot()
    assert _prefix_state_id(endpoint)==endpoint.state_id
    with pytest.raises(ValueError,match='CRANIAL_STAGE_PREFIX_REPLAY_UNSUPPORTED'):
        SurgicalPrefixReplay(endpoint.scene,saved)
    assert endpoint.snapshot()==saved


def test_equal_content_archives_keep_exact_stage_bound_guard_objects():
    from resectionlab.surgical_engine import SurgicalEngine
    from test_surgical_bone_transport import request
    import numpy as np
    from resectionlab.surgical_rl_protection import GuardConfig,ProtectionAnnotation,ProtectionSet
    from resectionlab.surgical_cranial_stages import freeze_cranial_stage_contract
    original,c,contour,p,r=joint_source();mask=np.zeros(original.scene.shape,bool);mask[45,45,80]=True
    annotation=ProtectionAnnotation.from_mask(original.scene,'generated-distant-guard',mask,provenance='Generated exact object-binding test only')
    protection=ProtectionSet(original.scene,[annotation],clearance_mm=0.,required=())
    guard=GuardConfig(require_critical_coverage=False)
    source=SurgicalEngine(original.scene,protection=protection,guard_config=guard)
    bound=freeze_cranial_stage_contract(source,membrane_cells=p,retained_membrane_cells=r,membrane_anchor_cells=c.membrane_anchor_cells,work_cells=c.work_cells,evidence=c.as_dict()['evidence'])
    endpoint=SurgicalEngine(source.scene,protection=protection,guard_config=guard,cranial_stages=bound)
    run(endpoint,bone_actions(contour));endpoint.observe()
    cache=SurgicalReplayCache();cache.seek(endpoint,endpoint.snapshot(),len(endpoint.actions))
    other=SurgicalEngine.restore(endpoint.scene,endpoint.snapshot(),use_cache=False)
    assert other.guard_config is not endpoint.guard_config
    assert other.step(request('inspect'))['status']=='accepted'
    projected=cache.seek(other,other.snapshot(),len(other.actions))
    assert projected.guard_config is other.guard_config and projected.protection is other.protection
    assert projected.observe()['cranialStages']['bonePrefixVerified']
    assert not projected.observe()['cranialStages']['workEntrySupported']


def test_equal_content_distinct_adapter_scenes_keep_original_target_proofs():
    import copy
    import numpy as np
    from dataclasses import replace
    from resectionlab.core import CaseData,SourceRef,thaw_json
    from resectionlab.surgical_scene import from_case
    from resectionlab.surgical_engine import SurgicalEngine
    from resectionlab.surgical_craniotomy import PROPERTY,binding,contract_hash
    from resectionlab.surgical_cranial_stages import freeze_cranial_stage_contract
    original,c,contour,p,r=joint_source();mapping=original.scene.regions
    properties=copy.deepcopy(original.scene.properties);bone=properties.pop(PROPERTY)
    metadata={'surgical_properties':properties,'anatomy_layers':{row['name']:{'kind':{'bone':'skull','tumor':'tumor','membrane':'dura'}[row['material']]} for row in mapping}}
    base=CaseData(case_id='generated-stage-cache-scene-identity',mri=original.scene.labels.astype(np.float32),affine=original.scene.affine,
        compartments={row['name']:original.scene.labels==row['id'] for row in mapping},metadata=metadata,
        source_refs=(SourceRef('generated','simulated:stage-cache-scene-identity',provenance='simulated'),))
    bone.update(binding(from_case(base,mapping)));bone['contentHash']=contract_hash(bone)
    metadata=thaw_json(base.metadata);metadata['surgical_properties'][PROPERTY]=bone;case=replace(base,metadata=metadata)
    first_scene=from_case(case,mapping);second_scene=from_case(case,mapping)
    assert first_scene is not second_scene and first_scene.identity==second_scene.identity
    assert first_scene._original_target_proof is not second_scene._original_target_proof
    source=SurgicalEngine(first_scene)
    bound=freeze_cranial_stage_contract(source,membrane_cells=p,retained_membrane_cells=r,membrane_anchor_cells=c.membrane_anchor_cells,work_cells=c.work_cells,evidence=c.as_dict()['evidence'])
    first=SurgicalEngine(first_scene,cranial_stages=bound);second=SurgicalEngine(second_scene,cranial_stages=bound.as_dict())
    assert first.state_id==second.state_id
    cache=SurgicalReplayCache();cache.seek(first,first.snapshot(),0);result=cache.seek(second,second.snapshot(),0)
    assert result.scene is second_scene and result.observe()['cranialStages']['reason']!='CRANIAL_STAGE_CONTRACT_SOURCE_CHANGED'
    assert cache.statistics['archives']==1
