"""Tiny generated-only burden controls; no acquired arrays, models or training."""
from dataclasses import replace
from pathlib import Path
import sys

ROOT = next(p for p in Path(__file__).resolve().parents if (p/'src/resectionlab').is_dir())
sys.path[:0] = [str(Path(__file__).resolve().parents[1]), str(ROOT/'src'), str(ROOT/'tests')]

import numpy as np
import pytest

import initial_goal_burden as helper
from resectionlab.core import semantic_digest, thaw_json
from resectionlab.focused_paged_task import FocusedPagedObstructionTask
from resectionlab.geometry import GENERIC_TOOLS, GeometryScene, capsule_voxel_indices
from resectionlab.native_resection import NativeResectionEngine, contained_capsule_cells
from resectionlab.post_exposure import preentry_tip
from test_obstruction_opening_proposals import source
from test_post_exposure import arrays, source as post_source


def make_task(*, private=False, rotated=False, sign=1, unknown=False, crop=False, edge=False):
    base = source(cap=120, rotated=rotated, sign=sign)
    support = np.zeros(base.observed_support.shape, bool)
    x = 9 if sign > 0 else 6
    central, shifted = (x, 9, 9), (x, 12 if not edge else 18, 9)
    for c in ((4 if sign > 0 else 11, 10, 10), central, shifted): support[c] = True
    target = np.zeros(support.shape, np.float32)
    target[central] = 1.; target[shifted] = 1.
    domain = np.ones_like(support)
    if unknown: domain[3 if sign > 0 else 12, 12, 9] = False
    config = replace(base.proposal_config, offsets_source_voxels=((0, 0), (9 if edge else 3, 0)),
                     obstruction_page_size=8, obstruction_selection_rule='target_ray_coverage_v3')
    base = replace(base, observed_support=support, nominal_target=target,
                   reference_target=np.zeros_like(target) if private else target,
                   support_domain=domain, crop_shape=(3,3,3) if crop else base.crop_shape,
                   proposal_config=config)
    return FocusedPagedObstructionTask(base, max_steps=48)


def scores(report):
    return [(row['endpoint_voxel'], row['estimated_nominal_preparation_cells'], row['rankable'])
            for row in report['rows']]


def test_discriminates_existing_same_tool_goals_without_preview_or_inventory(monkeypatch):
    task = make_task(); state = task._engine.state_hash
    def forbidden(*args, **kwargs): raise AssertionError('Ranking may not certify or execute')
    with monkeypatch.context() as m:
        for name in ('observation', 'candidate_inventory', '_prepare_base', 'step_record'):
            m.setattr(task, name, forbidden)
        m.setattr(NativeResectionEngine, 'preview_stroke', forbidden)
        m.setattr(NativeResectionEngine, 'commit_preview', forbidden)
        report = thaw_json(helper.rank_initial_goals(task))
    assert report['complete'] and report['top_changed_from_original_first_same_tool']
    assert report['ranking_raw_orders'][0] != report['rows'][0]['raw_order']
    assert report['rows'][0]['estimated_nominal_preparation_cells'] == 1
    assert report['rows'][1]['estimated_nominal_preparation_cells'] == 0
    assert report['minimum_score_tie_count'] == 1
    assert report['native_preview_calls'] == report['native_transitions'] == 0
    assert report['helper_capsule_cover_calls'] == 2*report['unique_goal_count'] == 4
    assert report['helper_containment_calls'] == report['unique_goal_count']
    assert task._engine.state_hash == state and not task._engine.history
    assert task._base_inventory is None and task._goal_focus is None
    unsigned = {k:v for k,v in report.items() if k != 'seal'}
    assert semantic_digest(unsigned) == report['seal']


@pytest.mark.parametrize('rotated,sign', [(False,-1), (True,1), (True,-1)])
def test_orthogonal_rotated_reflected_anisotropic_frames_preserve_estimate(rotated,sign):
    normal = helper.rank_initial_goals(make_task())
    transformed = helper.rank_initial_goals(make_task(rotated=rotated,sign=sign))
    assert [r['estimated_nominal_preparation_cells'] for r in normal['rows']] == [
        r['estimated_nominal_preparation_cells'] for r in transformed['rows']]
    assert transformed['ranking_raw_orders'] == normal['ranking_raw_orders']
    assert all(r['native_coverage']['orthogonal_native_grid'] for r in transformed['rows'])
    assert all(r['native_coverage']['outside_native_geometry_is_unassessed'] for r in transformed['rows'])


def test_private_reference_and_actor_crop_do_not_change_native_score():
    a = helper.rank_initial_goals(make_task())
    b = helper.rank_initial_goals(make_task(private=True))
    c = helper.rank_initial_goals(make_task(crop=True))
    assert scores(a) == scores(b) == scores(c)
    assert a['ranking_raw_orders'] == b['ranking_raw_orders'] == c['ranking_raw_orders']
    assert not a['private_reference_used'] and not c['actor_crop_used_to_clip_score']
    assert any(r['full_tool_cells_outside_actor_crop'] > 0 for r in c['rows'])


def test_unknown_and_inward_native_clipping_cannot_rank_as_low_burden():
    unknown = helper.rank_initial_goals(make_task(unknown=True))
    row = next(r for r in unknown['rows'] if tuple(r['endpoint_voxel'])[1] == 12)
    assert row['estimated_nominal_preparation_cells'] == 0
    assert not row['rankable'] and row['unknown_interaction_cells'] > 0
    assert 'unknown_interaction_domain' in row['unrankable_reasons']
    assert row['raw_order'] not in unknown['ranking_raw_orders']
    clipped = helper.rank_initial_goals(make_task(edge=True))
    edge = next(r for r in clipped['rows'] if tuple(r['endpoint_voxel'])[1] == 18)
    assert not edge['rankable'] and edge['native_coverage']['inward_or_lateral_crop_uncertain']
    assert 'uncertain_inward_native_coverage' in edge['unrankable_reasons']
    # Unsupported shear never becomes optimistic clear space, even though the
    # current native engine itself refuses shear at source construction.
    affine = np.eye(4); affine[0,1] = .2
    scene = GeometryScene(np.zeros((8,8,8),bool), affine)
    flag = helper._coverage(scene, [1.,3.,3.], np.array([1.,0,0]),
        ((np.array([-5.,3.,3.]),np.array([4.,3.,3.]),1.),
         (np.array([-2.,3.,3.]),np.array([7.,3.,3.]),1.)))
    assert not flag['orthogonal_native_grid'] and flag['inward_or_lateral_crop_uncertain']


def test_real_post_exposure_endpoints_and_full_sweeps_include_native_microsteps():
    S,T,Ds = arrays(); case = post_source(S,T,Ds, proposal=True, tools=GENERIC_TOOLS)
    case = replace(case, proposal_config=replace(case.proposal_config, max_candidates=120,
        tool_footprint_opening=True, obstruction_opening=True, obstruction_page_size=8,
        obstruction_selection_rule='target_ray_coverage_v3'))
    task = FocusedPagedObstructionTask(case,max_steps=48)
    report = helper.rank_initial_goals(task)
    batch = task.case._nominal_proposer.propose(task._engine)
    ray = next(r for r in batch.proposals if r.tool_id == helper.TOOL_ID and case.nominal_target[r.voxel] > 0)
    tool, axis, physical_start, shaft, active_tip = helper._capsules(task._engine,ray)
    expected = preentry_tip(tool,np.asarray(ray.entry_mm),axis,task._config.access.normal_inward)
    assert np.array_equal(physical_start,expected) and physical_start[0] < ray.entry_mm[0]
    row = report['rows'][0]
    assert row['physical_start_mm'] == tuple(physical_start)
    assert row['external_workspace_unassessed_cells'] > 0 and row['unknown_interaction_cells'] == 0
    assert row['optimistic_active_tip_sweep']['radius_mm'] == tool.tip_radius_mm < tool.shaft_radius_mm
    whole = {tuple(v) for v in capsule_voxel_indices(task._engine._cell_scene,*shaft)}
    full_tip_cells = capsule_voxel_indices(task._engine._cell_scene,*active_tip)
    full_contained = {tuple(v) for v in contained_capsule_cells(task._engine._cell_scene,full_tip_cells,*active_tip)}
    count = int(np.ceil(np.linalg.norm(np.asarray(ray.tip_mm)-physical_start)/task._config.max_tip_step_mm))
    previous = physical_start
    for step in range(count+1):
        current = physical_start+(np.asarray(ray.tip_mm)-physical_start)*(step/count)
        segment = capsule_voxel_indices(task._engine._cell_scene,
            previous-tool.working_length_mm*axis,current-tool.tip_length_mm*axis,tool.shaft_radius_mm)
        assert {tuple(v) for v in segment} <= whole
        local = contained_capsule_cells(task._engine._cell_scene, full_tip_cells,
            previous-tool.tip_length_mm*axis,current,tool.tip_radius_mm)
        assert {tuple(v) for v in local} <= full_contained
        previous = current
    native = task._engine.clone().preview_stroke(ray.tool_id,ray.tip_mm,entry_mm=ray.entry_mm,
        obstruction_diagnostics=True)
    if native.obstruction_diagnostic is not None:
        assert {tuple(v) for v in native.obstruction_diagnostic['blocked_indices_native']} <= whole


def test_physical_aliases_do_not_spend_extra_queries_or_change_score(monkeypatch):
    task = make_task(); baseline = helper.rank_initial_goals(task)
    proposer = task.case._nominal_proposer; original = proposer.propose
    def alias(engine, **kwargs):
        batch = original(engine, **kwargs)
        ray = next(r for r in batch.proposals if r.tool_id == helper.TOOL_ID and task.case.nominal_target[r.voxel] > 0)
        return replace(batch, proposals=(*batch.proposals,replace(ray,proposal_id='generated-alias',family='generated-alias')))
    monkeypatch.setattr(proposer,'propose',alias)
    result = helper.rank_initial_goals(task)
    assert scores(result) == scores(baseline)
    assert result['ranking_raw_orders'] == baseline['ranking_raw_orders']
    assert result['same_tool_goal_rows'] == baseline['same_tool_goal_rows']+1
    assert result['helper_capsule_cover_calls'] == baseline['helper_capsule_cover_calls']
    assert len(result['rows'][0]['aliases']) == len(baseline['rows'][0]['aliases'])+1


def test_caps_cancellation_and_noninitial_query_refuse_without_partial_ordering():
    task = make_task(); state = task._engine.state_hash
    for limits in (helper.Limits(max_goals=1), helper.Limits(max_bbox_cells_per_query=1)):
        with pytest.raises(helper.BurdenLimit) as exc: helper.rank_initial_goals(task,limits=limits)
        assert not exc.value.accounting['complete'] and not exc.value.accounting['ranked_physical_goals']
    with pytest.raises(InterruptedError): helper.rank_initial_goals(task,cancelled=lambda:True)
    assert task._engine.state_hash == state and task._base_inventory is None
    with pytest.raises(TypeError): helper.rank_initial_goals(task)['complete'] = False
    goal = next(iter(task.case._nominal_proposer.propose(task._engine).proposals))
    key = helper._key(goal)
    # Select an authenticated goal query: a ranking diagnostic cannot silently
    # replace an existing policy query even before a transition.
    key = next(helper._key(r) for r in task.case._nominal_proposer.propose(task._engine).proposals
               if task.case.nominal_target[r.voxel] > 0)
    task.set_focus(task.focus_record(key))
    with pytest.raises(ValueError,match='Unqueried'): helper.rank_initial_goals(task)
