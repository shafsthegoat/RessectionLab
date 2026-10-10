"""Generated-only controls; no patient files, model or training execution."""
from pathlib import Path
import sys
ROOT=next(p for p in Path(__file__).resolve().parents if (p/'src/resectionlab').is_dir())
BASE=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
import resectionlab
resectionlab.__path__.insert(0,str(BASE/'stage/resectionlab'))
from dataclasses import replace
from types import SimpleNamespace
import copy, json, os, subprocess, tempfile
import numpy as np
import pytest
from resectionlab.geometry import AccessWindow, ToolGeometry, GeometryScene, ToolPose, tool_capsules, capsule_voxel_indices
from resectionlab.post_exposure import prepare_post_exposure, preentry_tip, VERSION
from resectionlab.native_resection import NativeResectionConfig, NativeResectionEngine
from resectionlab.native_proposals import _verify_cavity, SUPPLIED_GOAL_REGION, NominalCavityProposalConfig
from resectionlab.native_spatial_task import NativeSpatialCase, NativeSpatialTask, DERIVED_OCCUPANCY_SOURCE_KIND
from resectionlab.native_spatial_evaluation import evaluate_native_spatial_episode
from resectionlab.evaluation import independent_check_native_history
from resectionlab.public_target_context import VERSION as CONTEXT
from resectionlab.core import array_digest

TOOL=ToolGeometry('generated',1.3,.25,12.,45.,1.5)

def arrays(layers=1):
    S=np.zeros((17,17,17),bool); S[8:13,1:16,1:16]=True
    T=np.zeros_like(S); T[8,8,8]=True
    Ds=np.zeros_like(S); Ds[8-layers:16,1:16,1:16]=True
    return S,T,Ds


def prepared(S,T,Ds,tools=(TOOL,)):
    previous=AccessWindow([7.5,8,8],[1,0,0],6.)
    access,start=prepare_post_exposure(support=S,target=T,support_domain=Ds,
        affine=np.eye(4),previous_access=previous)
    config=NativeResectionConfig(S|T,T.astype(np.int16),np.eye(4),access,tools,
        'generated-source','explicit generated material',interaction_domain=Ds|T,post_exposure=start)
    return config,start


def source(S,T,Ds, *, proposal=False, tools=(TOOL,)):
    cfg,start=prepared(S,T,Ds,tools)
    return NativeSpatialCase(np.arange(S.size,dtype=np.float32).reshape(S.shape),S|T,T.astype(np.float32),
        np.eye(4),cfg.access,tools,track='annotation_assisted',support_source_kind=DERIVED_OCCUPANCY_SOURCE_KIND,
        support_derivation='generated union assumption',nominal_target=T.astype(np.float32),
        target_source_kind='supplied_annotation',target_derivation='generated supplied whole region',
        crop_shape=S.shape,occupancy_source_support=S,support_domain=Ds,post_exposure=start,
        target_semantics=SUPPLIED_GOAL_REGION,public_target_context_variant=CONTEXT,
        public_target_domain=np.ones_like(S),proposal_mode='nominal_cavity_v1' if proposal else 'fixed_lattice',
        proposal_config=NominalCavityProposalConfig() if proposal else None)


@pytest.mark.parametrize('layers',[1,2])
def test_thin_known_zero_layer_supports_seed_and_clear_preentry_without_initial_cut(layers):
    S,T,Ds=arrays(layers); cfg,start=prepared(S,T,Ds)
    assert cfg.access.center_mm[0]==7.5 and start.seed[7,8,8]
    p0=preentry_tip(TOOL,cfg.access.center_mm,np.array([1.,0,0]),cfg.access.normal_inward)
    assert p0[0]==pytest.approx(6.2-1e-8)
    engine=NativeResectionEngine(cfg)
    assert not engine.removed_mask.any() and not engine.contact_mask.any()
    assert np.array_equal(engine.connected_free_mask,start.initial_free)
    assert not np.any(engine.connected_free_mask & (~Ds | (S|T)))
    cell_scene=GeometryScene(np.zeros_like(S),np.eye(4))
    for a,b,r,_ in tool_capsules(TOOL,ToolPose(p0,[1,0,0])):
        cells=capsule_voxel_indices(cell_scene,a,b,r)
        assert not (S|T)[tuple(cells.T)].any()
    before=engine.state_hash
    result=engine.preview_stroke(TOOL.tool_id,[8,8,8])
    assert result.feasible and result.removed_volume_mm3>0
    assert engine.state_hash==before and not engine.removed_mask.any()
    assert 'declared_external_workspace_unassessed' in result.geometry_unknowns
    assert not NativeResectionEngine(replace(cfg,post_exposure=None)).preview_stroke(TOOL.tool_id,[8,8,8]).feasible


@pytest.mark.parametrize('location',[(7,8,8),(9,8,8)])
def test_touching_or_inward_U_is_never_exempt(location):
    S,T,Ds=arrays(); S[location]=False; T[location]=False; Ds[location]=False
    cfg,start=prepared(S,T,Ds)
    assert not start.external_workspace[location]
    engine=NativeResectionEngine(cfg); before=engine.state_hash
    r=engine.preview_stroke(TOOL.tool_id,[8,8,8])
    assert not r.feasible and r.reason=='UNKNOWN_DOMAIN:FORBIDDEN_COLLISION'
    assert engine.state_hash==before and not engine.removed_mask.any()


def test_lateral_initial_shaft_material_is_not_exempted_as_external():
    S,T,Ds=arrays(); S[1,1,8]=True; Ds[1,1,8]=True
    cfg,start=prepared(S,T,Ds)
    assert cfg.access.center_mm[0]==7.5 # extra cell lies wholly outside fixed disc
    assert not start.external_workspace[1,1,8]
    r=NativeResectionEngine(cfg).preview_stroke(TOOL.tool_id,[8.5,9,8])
    assert not r.feasible and r.reason=='POST_EXPOSURE_INITIAL_TOOL_OCCUPIED'


def test_K_closure_cannot_cross_unknown_into_an_enclosed_known_zero_pocket():
    S,T,Ds=arrays(); S[8:11,8,8]=False; T[8,8,8]=False; T[8,9,8]=True
    Ds[9,8,8]=False
    cfg,start=prepared(S,T,Ds)
    assert start.initial_free[8,8,8]
    assert not start.initial_free[9,8,8] and not start.initial_free[10,8,8]
    assert not np.any(start.initial_free & start.external_workspace)


def test_cut_connects_new_known_zero_without_counting_it_as_removed():
    S,T,Ds=arrays(); S[9,8,8]=False
    cfg,start=prepared(S,T,Ds)
    engine=NativeResectionEngine(cfg)
    assert not engine.connected_free_mask[9,8,8]
    result=engine.execute_stroke(TOOL.tool_id,[8,8,8])
    assert result.feasible and engine.connected_free_mask[9,8,8]
    assert not engine.removed_mask[9,8,8]
    assert not np.any(engine.removed_mask & ~cfg.tissue_mask)
    _verify_cavity(engine)
    independent=SimpleNamespace(mri=np.zeros(S.shape),affine=np.eye(4),frame='RAS+',semantic_hash=cfg.source_hash)
    audit=independent_check_native_history(independent,cfg.tools,engine.history,
        tissue_mask=cfg.tissue_mask,access=cfg.access,interaction_domain=cfg.interaction_domain,post_exposure=start)
    assert audit.feasible
    changed=copy.deepcopy(engine.history); changed[0]['physical_start_mm'][0]+=.1
    bad=independent_check_native_history(independent,cfg.tools,changed,tissue_mask=cfg.tissue_mask,
        access=cfg.access,interaction_domain=cfg.interaction_domain,post_exposure=start)
    assert not bad.feasible and 'native_post_exposure_start_mismatch' in bad.failures


def test_empty_K_fails_without_alternative_access_and_masks_remain_immutable():
    S,T,Ds=arrays(); Ds[7]=False
    with pytest.raises(ValueError,match='EMPTY_FIXED_K'):prepared(S,T,Ds)
    S,T,Ds=arrays(); cfg,start=prepared(S,T,Ds)
    for a in (start.external_workspace,start.seed,start.initial_free):
        with pytest.raises(ValueError):a.setflags(write=True)
    changed=cfg.tissue_mask.copy(); changed[7,8,8]=True
    with pytest.raises(ValueError,match='differs from immutable'):replace(cfg,tissue_mask=changed)


def test_exact_preentry_pose_and_image_tip_guard():
    S,T,Ds=arrays(); cfg,start=prepared(S,T,Ds)
    axis=np.array([1,1,0],float)/np.sqrt(2)
    p0=preentry_tip(TOOL,cfg.access.center_mm,axis,cfg.access.normal_inward)
    assert (cfg.access.center_mm-p0)@cfg.access.normal_inward==pytest.approx(TOOL.tip_radius_mm+1e-8)
    # A deep endpoint still uses unchanged tip-image rejection.
    r=NativeResectionEngine(cfg).preview_stroke(TOOL.tool_id,[18,8,8])
    assert not r.feasible and r.reason=='HARD_GEOMETRY:TIP_OUTSIDE_IMAGE'


def test_actor_observes_connected_free_with_no_initial_reward_and_same_Ds():
    S,T,Ds=arrays(); S[9,8,8]=False
    task=NativeSpatialTask(source(S,T,Ds),max_steps=2)
    before=task.observation(); before.public_target_context.require_observation(before)
    assert task.metrics()['simulated_removed_volume_mm3']==0 and task.metrics()['total_reward']==0
    assert before.public_target_context.committed_overlap_mm3==0
    np.testing.assert_array_equal(before.coverage[1],Ds)
    np.testing.assert_array_equal(before.image_channels[3].astype(bool),task._engine.connected_free_mask)
    row=next(r for r in task.candidate_inventory()['ledger'] if r['feasible'] and r.get('voxel')==[8,8,8])
    task.step(row['action_id']); task.step('STOP')
    assert task._engine.connected_free_mask[9,8,8] and not task._engine.removed_mask[9,8,8]
    after=task.observation(); after.public_target_context.require_observation(after)
    assert after.image_channels[3,9,8,8]==1
    record=task.metrics()['history'][0]
    expected=np.linalg.norm(np.asarray(record['tip_mm'])-record['physical_start_mm'])
    assert record['insertion_distance_mm']==pytest.approx(expected)
    assert record['complete_tool_path_length_mm']==pytest.approx(2*expected)
    result=evaluate_native_spatial_episode(task)
    assert result['accepted'] and result['target_access_success']
    _verify_cavity(task._engine)


def test_approach_and_reverse_are_charged_per_insertion_and_tool_switch():
    S,T,Ds=arrays(); other=replace(TOOL,tool_id='generated-switch')
    case=source(S,T,Ds,tools=(TOOL,other)); task=NativeSpatialTask(case,max_steps=3)
    # Use actual distinct generated previews, without committing invented action records.
    for tool,tip,prior in ((TOOL,[8,8,8],None),(other,[8,8,8.2],TOOL.tool_id)):
        r=task._engine.preview_stroke(tool.tool_id,tip)
        assert r.feasible
        record=r.to_history_record(); record['action_id']='generated-cost-probe'
        scored=task._score_record(record,prior)
        start=np.asarray(record['physical_start_mm'])
        distance=np.linalg.norm(np.asarray(tip)-start)
        assert scored['complete_tool_path_length_mm']==pytest.approx(2*distance)
        target=scored['target_removed_mm3']; normal=scored['normal_removed_mm3']; w=task.reward_spec
        expected=w.target_per_mm3*target-w.normal_per_mm3*normal-w.action_cost-2*w.motion_per_mm*distance
        expected-=w.tool_change_cost*(prior is not None)
        assert scored['reward']==pytest.approx(expected)


def test_nominal_proposal_verifier_accepts_explicit_initial_closure():
    S,T,Ds=arrays();task=NativeSpatialTask(source(S,T,Ds,proposal=True),max_steps=2)
    _verify_cavity(task._engine)
    assert task.candidate_inventory()['ledger']


def test_absent_option_default_source_model_observation_history_replay_parity():
    values=[]
    for branch in ('baseline','stage'):
        with tempfile.TemporaryDirectory(prefix='post-exposure-pycache-') as fresh:
            result=subprocess.run([sys.executable,'-I','-B','-X','pycache_prefix='+fresh,
                str(BASE/'tests/parity_probe.py'),branch], check=True,capture_output=True,text=True,
                env={**os.environ,'OPENBLAS_NUM_THREADS':'1','OMP_NUM_THREADS':'1'})
        values.append(json.loads(result.stdout))
        if output := os.environ.get("POST_EXPOSURE_TEST_OUTPUT"):
            (Path(output)/(branch+"-default-parity.json")).write_text(result.stdout)
    assert values[0]==values[1]


def test_boundary_uses_full_projected_cell_footprint_in_rotated_native_frame():
    S,T,Ds=arrays(); S[5,14,10]=True; Ds[:8]=True
    # Its centre radius sqrt(40)>6; its projected rectangle nevertheless meets
    # the disc, so a centre-only rule would place the boundary incorrectly.
    theta=.37
    rotation=np.array([[np.cos(theta),-np.sin(theta),0],[np.sin(theta),np.cos(theta),0],[0,0,1.]])
    affine=np.eye(4);affine[:3,:3]=rotation;affine[:3,3]=[12.,-8.,2.]
    previous=AccessWindow(rotation@np.array([7.5,8,8])+affine[:3,3],rotation[:,0],6.)
    access,start=prepare_post_exposure(support=S,target=T,support_domain=Ds,affine=affine,previous_access=previous)
    source_index=np.linalg.solve(rotation,access.center_mm-affine[:3,3])
    np.testing.assert_allclose(source_index,[4.5,8,8],rtol=0,atol=1e-12)
    assert start.seed[4,8,8] and not start.initial_free[S|T].any()


@pytest.mark.parametrize('field,value',[
    ('external_workspace_encounter_cells',0),
    ('external_workspace_encounter_hash','sha256:'+'0'*64),
    ('inter_insertion_transfer','certified')])
def test_independent_replay_rejects_changed_external_reporting(field,value):
    S,T,Ds=arrays();cfg,start=prepared(S,T,Ds);engine=NativeResectionEngine(cfg)
    result=engine.execute_stroke(TOOL.tool_id,[8,8,8]);assert result.feasible
    history=copy.deepcopy(engine.history)
    assert history[0][field]!=value
    history[0][field]=value
    source_case=SimpleNamespace(mri=np.zeros(S.shape),affine=np.eye(4),frame='RAS+',semantic_hash=cfg.source_hash)
    audit=independent_check_native_history(source_case,cfg.tools,history,tissue_mask=cfg.tissue_mask,
        access=cfg.access,interaction_domain=cfg.interaction_domain,post_exposure=start)
    assert not audit.feasible and 'native_post_exposure_external_report_mismatch' in audit.failures


def test_nonforward_input_keeps_original_geometry_refusal():
    S,T,Ds=arrays();cfg,start=prepared(S,T,Ds)
    old=NativeResectionEngine(replace(cfg,post_exposure=None)).preview_stroke(TOOL.tool_id,[6,8,8])
    new=NativeResectionEngine(cfg).preview_stroke(TOOL.tool_id,[6,8,8])
    assert not old.feasible and not new.feasible
    assert new.reason==old.reason=='HARD_GEOMETRY:ACCESS_ANGLE'


def test_empty_external_encounter_report_is_well_defined_and_replayed():
    S,T,Ds=arrays();Ds[:]=True
    cfg,start=prepared(S,T,Ds);engine=NativeResectionEngine(cfg)
    result=engine.execute_stroke(TOOL.tool_id,[8,8,8]);assert result.feasible
    assert engine.history[0]['external_workspace_encounter_cells']==0
    assert 'declared_external_workspace_unassessed' not in result.geometry_unknowns
    source_case=SimpleNamespace(mri=np.zeros(S.shape),affine=np.eye(4),frame='RAS+',semantic_hash=cfg.source_hash)
    audit=independent_check_native_history(source_case,cfg.tools,engine.history,tissue_mask=cfg.tissue_mask,
        access=cfg.access,interaction_domain=cfg.interaction_domain,post_exposure=start)
    assert audit.feasible
