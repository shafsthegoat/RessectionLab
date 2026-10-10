"""Generated footprint endpoint controls; no patient arrays, models or search."""
from dataclasses import replace
from itertools import product
import numpy as np
import pytest
from resectionlab.core import array_digest
from resectionlab.geometry import AccessWindow, GENERIC_TOOLS, point_segment_distances
from resectionlab.native_proposals import (NominalCavityProposalConfig, PreparedNominalCavityProposer,
    DEFAULT_COLUMN_OFFSETS, TOOL_FOOTPRINT_OPENING_FAMILY, TOOL_FOOTPRINT_OPENING_VERSION)
from resectionlab.native_resection import NativeResectionConfig, NativeResectionEngine


def fixture(*, spacing=(.5,.5,.5), sign=1, support=None, tools=GENERIC_TOOLS,
            offsets=((0,0),), cap=96, enabled=True, intermediate=1., rotated=False):
    if support is None:
        support=np.zeros((16,23,23),bool);support[2:14,:,:]=True
    affine=np.diag([*spacing,1.])
    if rotated:
        rotation=np.array([[0.,-1.,0.],[1.,0.,0.],[0.,0.,1.]])
        affine[:3,:3]=rotation@affine[:3,:3]
    affine[:3,3]=[10.,-7.,20.]
    center=np.array([1.5 if sign>0 else support.shape[0]-2.5,
                     (support.shape[1]-1)//2,(support.shape[2]-1)//2])
    normal=sign*affine[:3,0]/spacing[0]
    access=AccessWindow(affine[:3,:3]@center+affine[:3,3],normal,6.)
    config=NativeResectionConfig(support,np.zeros(support.shape,np.int16),affine,access,
        tools,'generated-footprint','generated occupancy only',max_tip_step_mm=min(.25,min(spacing)/2))
    engine=NativeResectionEngine(config)
    target=np.zeros(support.shape,np.float32)
    rule=NominalCavityProposalConfig(offsets_source_voxels=offsets,max_candidates=cap,
        intermediate_opening_mm=intermediate,tool_footprint_opening=enabled)
    provider=PreparedNominalCavityProposer(config,target,nominal_provenance={
        'source_hash':config.source_hash,'nominal_target_hash':array_digest(target),
        'source_kind':'derived_from_scan','derivation':'generated zero public goal'},config=rule)
    return engine,provider


def new_rays(engine,provider):
    return [r for r in provider.propose(engine).proposals if r.family==TOOL_FOOTPRINT_OPENING_FAMILY]


def test_empty_centerline_recovers_real_off_axis_removal_without_proposal_preview(monkeypatch):
    support=np.zeros((16,23,23),bool);support[2,12,11]=True
    engine,provider=fixture(support=support,intermediate=None)
    original=NativeResectionEngine.preview_stroke
    def forbidden(*args,**kwargs): raise AssertionError('proposer queried native preview')
    with monkeypatch.context() as guard:
        guard.setattr(NativeResectionEngine,'preview_stroke',forbidden)
        batch=provider.propose(engine)
    assert not any(r.family!='tool_footprint_opening' for r in batch.proposals)
    ray=next(r for r in batch.proposals if r.tool_id=='generic_aspirator')
    slot=next(s for s in batch.ledger if s.proposal_id==ray.proposal_id)
    assert slot.footprint_anchor_voxel==(2,12,11) and not engine.remaining_mask[ray.voxel]
    assert slot.footprint_endpoint_tests==1 and slot.footprint_exposed_cells==1
    result=original(engine,ray.tool_id,ray.tip_mm,entry_mm=ray.entry_mm)
    assert result.feasible and result.removed_indices_native.tolist()==[[2,12,11]]
    engine.commit_preview(result)
    assert not new_rays(engine,provider)
    replay=NativeResectionEngine(engine.config)
    replay.execute_stroke(ray.tool_id,ray.tip_mm,entry_mm=ray.entry_mm)
    assert replay.history==engine.history and replay.state_hash==engine.state_hash


def test_saved_failure_mechanism_recovered_after_suction_and_old_family_saturation():
    # Same physical mechanism as the saved slab, with round .5-mm generated cells.
    engine,provider=fixture(offsets=DEFAULT_COLUMN_OFFSETS)
    legacy=PreparedNominalCavityProposer(engine.config,np.zeros(engine.config.tissue_mask.shape,np.float32),
        nominal_provenance=provider._provenance,
        config=NominalCavityProposalConfig(intermediate_opening_mm=1.))
    commits=[]
    for pass_number in range(4):
        before=len(engine.history)
        for family in ('exposed_opening','intermediate_opening'):
            for column in range(len(DEFAULT_COLUMN_OFFSETS)):
                for tool in GENERIC_TOOLS:
                    rays=[r for r in legacy.propose(engine).proposals
                          if r.family==family and r.column_index==column and r.tool_id==tool.tool_id]
                    if not rays: continue
                    r=rays[0];result=engine.preview_stroke(r.tool_id,r.tip_mm,entry_mm=r.entry_mm)
                    if result.feasible:
                        engine.commit_preview(result);commits.append(r)
        if len(engine.history)==before: break
    else: pytest.fail('generated legacy order did not saturate within fixed four passes')
    assert commits and commits[0].tool_id=='generic_suction'
    assert all(not engine.preview_stroke(r.tool_id,r.tip_mm,entry_mm=r.entry_mm).feasible
               for r in legacy.propose(engine).proposals)
    ray=next(r for r in new_rays(engine,provider)
             if r.column_index==0 and r.tool_id=='generic_aspirator')
    assert not engine.remaining_mask[ray.voxel]
    result=engine.preview_stroke(ray.tool_id,ray.tip_mm,entry_mm=ray.entry_mm)
    assert result.feasible and len(result.removed_indices_native)>0
    assert any(tuple(cell[1:])!=(11,11) for cell in result.removed_indices_native)
    prior=engine.removed_mask.copy();engine.commit_preview(result)
    assert int(engine.removed_mask.sum())>int(prior.sum())


@pytest.mark.parametrize('sign',[1,-1])
@pytest.mark.parametrize('rotated',[False,True])
def test_exact_earliest_containment_in_anisotropic_physical_frame(sign,rotated):
    support=np.zeros((16,23,23),bool)
    support[2 if sign>0 else 13,12,11]=True
    engine,provider=fixture(support=support,spacing=(.6,.4,.7),sign=sign,rotated=rotated,
                            intermediate=None,tools=(GENERIC_TOOLS[1],))
    ray=new_rays(engine,provider)[0]
    affine=engine.config.affine;tool=engine.config.tools[0]
    cell=np.argwhere(support)[0]
    corners=(cell+np.asarray(list(product((-.5,.5),repeat=3))))@affine[:3,:3].T+affine[:3,3]
    candidates=[]
    for x in range(support.shape[0]):
        tip=affine[:3,:3]@np.array([x,11,11])+affine[:3,3]
        depth=(tip-engine.config.access.center_mm)@engine.config.access.normal_inward
        if depth<=0: continue
        entry=tip-depth*engine.config.access.normal_inward
        distances=point_segment_distances(corners,entry-tool.tip_length_mm*engine.config.access.normal_inward,tip)
        if np.all(distances<=tool.tip_radius_mm-1e-10): candidates.append((depth,tip))
    expected=min(candidates,key=lambda v:v[0])[1]
    np.testing.assert_allclose(ray.tip_mm,expected,rtol=0,atol=1e-12)
    result=engine.preview_stroke(ray.tool_id,ray.tip_mm,entry_mm=ray.entry_mm)
    assert result.feasible and len(result.removed_indices_native)==1


def test_partial_contact_only_is_not_a_footprint_anchor():
    support=np.zeros((16,23,23),bool);support[2,13,11]=True
    engine,provider=fixture(support=support,tools=(GENERIC_TOOLS[0],),intermediate=None)
    batch=provider.propose(engine)
    assert not batch.proposals
    slot=batch.ledger[-1]
    assert slot.reason=='NO_FULLY_CONTAINABLE_EXPOSED_FOOTPRINT_CELL'
    assert slot.footprint_exposed_cells==1 and slot.footprint_radially_eligible_cells==0


def test_full_tool_shafts_are_still_certified_by_native_engine():
    support=np.zeros((16,23,23),bool)
    # A cell in the larger shaft annulus remains outside the active-tip footprint.
    support[2,16,11]=True
    support[10,12,11]=True
    engine,provider=fixture(support=support,tools=(GENERIC_TOOLS[1],),intermediate=None)
    ray=new_rays(engine,provider)[0]
    result=engine.preview_stroke(ray.tool_id,ray.tip_mm,entry_mm=ray.entry_mm)
    assert not result.feasible and result.reason=='SHAFT_BLOCKED_BY_REMAINING_NATIVE_TISSUE'
    assert not engine.history and not engine.removed_mask.any()


def test_preserves_prior_physical_rows_then_logs_cap_omission_and_dedup():
    engine,provider=fixture(cap=1)
    batch=provider.propose(engine)
    assert len(batch.proposals)==1
    assert len(batch.ledger)==10 and all(r.family=='tool_footprint_opening' for r in batch.ledger[-2:])
    assert batch.ledger[-2].reason=='DUPLICATE_GEOMETRY'
    assert batch.ledger[-1].reason=='CANDIDATE_CAP'
    assert batch.ledger[-1].footprint_anchor_voxel is not None
    assert batch.ledger[-1].footprint_endpoint_tests>0
    old=PreparedNominalCavityProposer(engine.config,np.zeros(engine.config.tissue_mask.shape,np.float32),
        nominal_provenance=provider._provenance,config=replace(provider._config,tool_footprint_opening=False))
    old_batch=old.propose(engine)
    physical=lambda r:(r.tool_id,r.family,r.voxel,r.entry_mm,r.tip_mm)
    assert [physical(r) for r in batch.proposals]==[physical(r) for r in old_batch.proposals]


def test_no_target_ranking_and_no_input_or_state_mutation():
    engine,provider=fixture()
    state=engine.state_hash;before=engine.remaining_mask.copy();history=list(engine.history)
    a=provider.propose(engine)
    other=np.array(engine.config.tissue_mask,dtype=np.float32)
    alt=PreparedNominalCavityProposer(engine.config,other,nominal_provenance={
        'source_hash':engine.config.source_hash,'nominal_target_hash':array_digest(other),
        'source_kind':'supplied_annotation','derivation':'different public target'},config=provider._config)
    b=alt.propose(engine)
    select=lambda batch:[(r.tool_id,r.voxel,r.footprint_anchor_voxel,r.footprint_endpoint_tests)
                         for r in batch.ledger if r.family==TOOL_FOOTPRINT_OPENING_FAMILY]
    assert select(a)==select(b)
    assert state==engine.state_hash and history==engine.history
    np.testing.assert_array_equal(before,engine.remaining_mask)


def test_cancellation_before_footprint_work():
    engine,provider=fixture()
    calls=0
    def cancel():
        nonlocal calls
        calls+=1
        return calls>=3
    with pytest.raises(InterruptedError): provider.propose(engine,cancelled=cancel)
    assert not engine.history


@pytest.mark.parametrize('value',[0,1,None,'yes'])
def test_opt_in_requires_exact_bool(value):
    with pytest.raises(ValueError,match='explicit bool'):
        NominalCavityProposalConfig(tool_footprint_opening=value)


def test_version_and_shared_cap_are_explicit():
    old=NominalCavityProposalConfig(intermediate_opening_mm=1.)
    new=replace(old,tool_footprint_opening=True)
    assert new.version==TOOL_FOOTPRINT_OPENING_VERSION
    assert new.fingerprint!=old.fingerprint and new.max_candidates==old.max_candidates==96
    assert new.families[:-1]==old.families and new.families[-1]=='tool_footprint_opening'


def test_private_reference_change_cannot_change_footprint_actions_observation_or_transition():
    from resectionlab.native_spatial_task import NativeSpatialCase, NativeSpatialTask
    support=np.zeros((16,23,23),bool);support[2,12,11]=True
    engine,provider=fixture(support=support,intermediate=None)
    config=engine.config;nominal=np.zeros(support.shape,np.float32)
    source=NativeSpatialCase(np.zeros(support.shape),support,np.zeros(support.shape,np.int16),
        config.affine,config.access,config.tools,nominal_target=nominal,
        target_derivation='generated public zero target',crop_shape=support.shape,
        proposal_mode='nominal_cavity_v1',proposal_config=provider._config)
    a=NativeSpatialTask(source,max_steps=2)
    b=NativeSpatialTask(replace(source,reference_target=np.ones(support.shape,np.int16)),max_steps=2)
    assert a.observation().fingerprint==b.observation().fingerprint
    assert a.candidate_inventory()==b.candidate_inventory()
    action=next(r['action_id'] for r in a.candidate_inventory()['emitted']
                if r['family']==TOOL_FOOTPRINT_OPENING_FAMILY and r['feasible'])
    pa,pb=a.planning_clone(),b.planning_clone()
    first,second=pa.advance_planning(action),pb.advance_planning(action)
    assert first.info==second.info and first.reward==second.reward
    assert pa.independent_geometry_check().feasible
