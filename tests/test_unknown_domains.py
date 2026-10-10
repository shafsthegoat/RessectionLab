"""Tiny generated-only checks. No patient files, model or training calls."""
from dataclasses import replace
from types import SimpleNamespace
import numpy as np
import pytest
from resectionlab.core import array_digest
from resectionlab.geometry import AccessWindow, ToolGeometry, GeometryScene, capsule_voxel_indices
from resectionlab.native_resection import (NativeResectionConfig, NativeResectionEngine,
    _extend_connected_free, _connected_surface_cells)
from resectionlab.native_spatial_task import (NativeSpatialTask, make_native_opening_task,
    DERIVED_OCCUPANCY_SOURCE_KIND, OPENING_TOOLS)
from resectionlab.native_proposals import SUPPLIED_GOAL_REGION, NominalCavityProposalConfig, _verify_cavity
from resectionlab.public_target_context import VERSION, PublicTargetContext
from resectionlab.evaluation import _extend_independent_free_space, independent_check_native_history
from resectionlab.native_spatial_evaluation import evaluate_native_spatial_episode


def config(tissue, domain=None, tool=None):
    tool = tool or ToolGeometry('generated', 1.25, .1, 12., 35., 2.)
    return NativeResectionConfig(tissue, np.zeros(tissue.shape, np.int16), np.eye(4),
        AccessWindow([4, 4, .5], [0, 0, 1], 3), (tool,), 'generated-source',
        'generated occupancy only', interaction_domain=domain)


def case(*, domain=True, context=True, proposal=False):
    original = make_native_opening_task().case
    raw = original.observed_support.copy(); raw[4, 4, 5] = False
    ds = np.ones(raw.shape, bool); ds[4, 4, 5] = False; ds[0, 0, 0] = False
    return replace(original, track='annotation_assisted', support_source_kind=DERIVED_OCCUPANCY_SOURCE_KIND,
        support_derivation='generated S union T material assumption; label zero is not anatomy truth',
        target_source_kind='supplied_annotation', target_derivation='generated full supplied region',
        target_semantics=SUPPLIED_GOAL_REGION, occupancy_source_support=raw,
        support_domain=ds if domain else None,
        public_target_context_variant=VERSION if context else None,
        public_target_domain=np.ones(raw.shape, bool) if context else None,
        proposal_mode='nominal_cavity_v1' if proposal else 'fixed_lattice',
        proposal_config=NominalCavityProposalConfig() if proposal else None)


def action(task, tool, voxel):
    return next(row['action_id'] for row in task.candidate_inventory()['ledger']
        if row['feasible'] and row['tool_id']==tool and row['voxel']==list(voxel))


def test_unknown_tunnel_does_not_seed_or_expose_a_pocket():
    tissue = np.zeros((9,9,9), bool); tissue[1:8,1:8,1:8] = True
    tissue[1:5,4,4] = False
    ds = np.ones(tissue.shape, bool); ds[2,4,4] = False
    old = NativeResectionEngine(config(tissue))
    new = NativeResectionEngine(config(tissue, ds))
    assert old.connected_free_mask[4,4,4]
    assert not new.connected_free_mask[4,4,4]
    assert not new.connected_free_mask[2,4,4]
    candidate = np.array([[4,5,4]])
    assert len(_connected_surface_cells(candidate, old.connected_free_mask))==1
    assert len(_connected_surface_cells(candidate, new.connected_free_mask))==0
    _verify_cavity(new)


@pytest.mark.parametrize('threshold', [0, 4096])
def test_flood_cannot_cross_unknown_after_a_cut_in_either_implementation(threshold):
    remaining=np.ones((7,3,3), bool); remaining[1:6,1,1]=False
    ds=np.ones(remaining.shape,bool); ds[3,1,1]=False
    production=np.zeros(remaining.shape,bool); independent=production.copy()
    _extend_connected_free(np.array([[1,1,1]]),remaining,production,ds)
    _extend_independent_free_space(remaining,independent,{(1,1,1)},
        full_flood_threshold=threshold,interaction_domain=ds)
    np.testing.assert_array_equal(production,independent)
    assert production[2,1,1] and not production[3:6,1,1].any()


def test_unknown_seed_refused_by_both_flood_implementations():
    remaining=np.zeros((3,3,3),bool); ds=np.ones_like(remaining); ds[1,1,1]=False
    with pytest.raises(ValueError,match='Unknown domain'):
        _extend_connected_free(np.array([[1,1,1]]),remaining,np.zeros_like(ds),ds)
    with pytest.raises(ValueError,match='Unknown domain'):
        _extend_independent_free_space(remaining,np.zeros_like(ds),{(1,1,1)},interaction_domain=ds)


def test_complete_shaft_crossing_unknown_refused_before_any_state_change():
    tissue=np.zeros((9,9,9),bool); tissue[4,4,5]=True
    ds=np.ones(tissue.shape,bool); ds[4,4,1]=False
    tool=ToolGeometry('shaft-only',.9,.3,4.,35.,1.)
    cfg=NativeResectionConfig(tissue,np.zeros(tissue.shape,np.int16),np.eye(4),
        AccessWindow([4,4,4.5],[0,0,1],3),(tool,),'generated-source','generated occupancy',interaction_domain=ds)
    active=capsule_voxel_indices(GeometryScene(np.zeros_like(ds),np.eye(4)),[4,4,3.5],[4,4,5],.9)
    assert (4,4,1) not in set(map(tuple,active))
    assert NativeResectionEngine(replace(cfg,interaction_domain=None)).preview_stroke(tool.tool_id,[4,4,5]).feasible
    engine=NativeResectionEngine(cfg)
    before=engine.state_hash
    result=engine.execute_stroke(tool.tool_id,[4,4,5])
    assert not result.feasible and result.reason.startswith('UNKNOWN_DOMAIN:')
    assert engine.state_hash==before and engine.revision==0
    assert not engine.removed_mask.any() and not engine.contact_mask.any()


def test_external_shaft_flag_and_tip_guard_remain_unchanged():
    original=make_native_opening_task().case
    ds=np.ones(original.observed_support.shape,bool); ds[0,0,0]=False
    engine=NativeResectionEngine(replace(original._native_config,interaction_domain=ds))
    result=engine.preview_stroke(OPENING_TOOLS[0].tool_id,[4,4,1])
    assert result.feasible
    assert 'tool_geometry_outside_image_unassessed' in result.geometry_unknowns
    outside=engine.preview_stroke(OPENING_TOOLS[0].tool_id,[4,4,8])
    assert not outside.feasible and outside.reason=='HARD_GEOMETRY:TIP_OUTSIDE_IMAGE'


def test_observed_Ds_is_not_extended_to_assumed_target_domain():
    source=case(); task=NativeSpatialTask(source,max_steps=2)
    observation=task.observation()
    assert not observation.coverage[1,4,4,5]
    assert observation.image_channels[1,4,4,5]==0
    assert observation.coverage[2,4,4,5] and observation.image_channels[2,4,4,5]==1
    assert observation.coverage[3,4,4,5] and not observation.coverage[3,0,0,0]
    assert source._native_config.tissue_mask[4,4,5]
    assert source._native_config.interaction_domain[4,4,5]
    assert not source.support_domain[4,4,5]
    assert source._domain_record['target_in_unknown_support_voxels']==1
    assert source._supplied_goal_extent['full_region_positive_voxels']==2
    assert source._domain_record['source_domain_extended'] is False
    observation.public_target_context.require_observation(observation)
    assert observation.public_target_context.static['mass_mm3']==2
    assert 'Ds OR (T > 0)' in observation.channel_provenance['nominal_tissue']['derivation']


def test_domain_masks_and_full_target_are_immutable_and_bound():
    source=case()
    for values in (source.support_domain,source._interaction_domain,source.occupancy_source_support,source.nominal_target):
        with pytest.raises(ValueError): values.setflags(write=True)
    shifted=source.support_domain.copy(); shifted[0,0,1]=False
    other=replace(source,support_domain=shifted)
    assert source.source_hash!=other.source_hash
    np.testing.assert_array_equal(source.nominal_target,other.nominal_target)
    np.testing.assert_array_equal(source.occupancy_source_support,other.occupancy_source_support)
    object.__setattr__(source,'support_domain',shifted)
    with pytest.raises((ValueError,RuntimeError)): source.assert_intact()


@pytest.mark.parametrize('mutation',['shape','nonbinary','missing_positive'])
def test_malformed_source_domain_refused(mutation):
    source=case(); ds=source.support_domain.copy()
    if mutation=='shape':ds=ds[:-1]
    elif mutation=='nonbinary':ds=ds.astype(float);ds[0,0,0]=.5
    else:ds[4,4,1]=False
    with pytest.raises(ValueError):replace(source,support_domain=ds)


def test_cavity_unknown_positive_cannot_be_silently_zeroed():
    source=case(); cavity=np.zeros(source.observed_support.shape,bool);cavity[0,0,0]=True
    with pytest.raises(ValueError,match='unavailable interaction'):source.spatial_inputs(cavity)


def test_optional_context_refuses_forged_knownness_without_masking_target():
    observation=NativeSpatialTask(case(),max_steps=2).observation()
    values=observation.coverage.copy();values[1,4,4,5]=True
    with pytest.raises(ValueError,match='context differs'):
        replace(observation,coverage=values)
    static=dict(observation.public_target_context.static);static['local_support_domain_hash']=array_digest(np.ones((9,9,7),bool))
    forged=PublicTargetContext(static,0.,observation.public_target_context.committed_cavity_hash,
        observation.public_target_context.local_cavity_hash)
    with pytest.raises(ValueError,match='context differs'):forged.require_observation(observation)


def test_initial_and_post_cut_proposal_cavity_verification_include_domain():
    source=case(proposal=True)
    engine=NativeResectionEngine(source._native_config)
    _verify_cavity(engine)
    engine.execute_stroke(OPENING_TOOLS[0].tool_id,[4,4,1])
    assert engine.revision==1
    _verify_cavity(engine)
    task=NativeSpatialTask(source,max_steps=2)
    assert task.candidate_inventory()['ledger']


def test_full_task_to_sealed_evaluator_keeps_assumed_target_and_unknown_domain():
    source=case();task=NativeSpatialTask(source,max_steps=2)
    for tool,voxel in ((OPENING_TOOLS[0].tool_id,(4,4,1)),(OPENING_TOOLS[1].tool_id,(4,4,5))):
        task.step(action(task,tool,voxel))
    assert task._engine.removed_mask[4,4,5] and not source.support_domain[4,4,5]
    assert not task._engine.connected_free_mask[0,0,0]
    result=evaluate_native_spatial_episode(task)
    assert result['accepted'] and result['target_access_success']
    assert result['outcomes']['target_removed_mm3']==2.
    assert task.independent_geometry_check().feasible
    assert task.metrics()['source_and_simulated_domains']['target_in_unknown_support_voxels']==1
    assert task.planning_clone().case._domain_record==source._domain_record


def test_independent_history_rejects_a_previously_legal_path_crossing_new_unknown():
    task=make_native_opening_task()
    task.step(action(task,OPENING_TOOLS[0].tool_id,(4,4,1)))
    ds=np.ones(task.case.observed_support.shape,bool);ds[4,4,0]=False
    source=SimpleNamespace(mri=task.case.structural_intensity,affine=task.case._native_affine_ras_mm,
        frame='RAS+',semantic_hash=task.case.source_hash)
    result=independent_check_native_history(source,task.case.tools,task.metrics()['history'],
        tissue_mask=task.case.observed_support,access=task.case.access,interaction_domain=ds)
    assert not result.feasible and 'native_full_tool_hard_constraint_failure' in result.failures
