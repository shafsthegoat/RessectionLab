"""Focused generated candidate controls, no search/model or held-out task."""
from dataclasses import replace
import json

import numpy as np
import pytest

from resectionlab import public_contact_family as family
from resectionlab.core import semantic_digest
from resectionlab.development_episode import MODES, replay_frames
from resectionlab.geometry import AccessWindow
from resectionlab.native_spatial_task import NativeSpatialTask, ACCESS_CENTERLINE_PROPOSAL_VERSION
from resectionlab.native_resection import NativeResectionEngine
from resectionlab.public_surface_contact import (SurfaceContactTask, PublicSurfaceGoal,
    seal_complete_strategy, verify_complete_strategy)
from resectionlab.spatial_observations import SpatialAction, ObservedProcedureState, build_spatial_observation
from resectionlab.simulation import InvalidActionError


def case():
    return family.build_family_source('pcf-06')


def task(source=None):
    source = case() if source is None else source
    return SurfaceContactTask(source, objective=PublicSurfaceGoal(source.source_hash, (7, 6, 5)))


def opening(t):
    matches = [r for r in t.candidate_inventory()['ledger']
        if r.get('family') == 'access_centerline' and r['tip_mm'] == [5.75, 6., 2.]
        and r['interaction_mode'] == 'aspirate']
    assert len(matches) == 1 and matches[0]['feasible']
    return matches[0]


def test_default_source_model_and_rejections_are_unchanged():
    old = replace(case(), proposal_mode='fixed_lattice')
    assert old.source_hash == 'sha256:39ce8e6223a24af6c503b48f3f4b0472a5b61e21a905e4bf39883929ebf2f5a0'
    original = task(old)
    assert original.decision_model_hash == 'sha256:d9126a6fcc427e2791122dd09858b69bd8419f302ce380b322955d94a6e4e21a'
    assert original.observation().action_ids == ('STOP',)
    rows = original.candidate_inventory()['ledger']
    assert len(rows) == 30 and all('family' not in r for r in rows)
    assert {r['reason'] for r in rows if r['interaction_mode'] == 'aspirate'} == {'HARD_GEOMETRY:ACCESS_APERTURE'}
    assert {r['reason'] for r in rows if r['interaction_mode'] == 'probe'} == {'PROBE_REQUIRES_EXISTING_CAVITY'}
    assert rows[0]['action_id'] == 'NATIVE-SPATIAL:e76135c7d8c8ea1aa1aedfce'


def test_new_mode_keeps_every_legacy_pose_and_rejection():
    revised = task()
    original = task(replace(revised.case, proposal_mode='fixed_lattice'))
    old_rows = original.candidate_inventory()['ledger']
    rows = revised.candidate_inventory()['ledger']
    retained = [r for r in rows if r['family'] == 'legacy_integer_lattice']
    keys = ('voxel', 'entry_mm', 'tip_mm', 'tool_id', 'interaction_mode', 'feasible', 'reason')
    assert [{k:r[k] for k in keys} for r in retained] == [{k:r[k] for k in keys} for r in old_rows]
    assert len(rows) == revised.candidate_inventory()['declared_slots'] == 40
    assert len({r['action_id'] for r in rows}) == 40
    added = [r for r in rows if r['family'] == 'access_centerline' and r['interaction_mode'] == 'aspirate']
    assert [r['tip_mm'][2] for r in added] == [2., 3., 4., 6., 10.]
    assert all(r['reason'] == 'SHAFT_BLOCKED_BY_REMAINING_NATIVE_TISSUE' for r in added[1:])
    assert opening(revised)['voxel'] is None
    assert opening(revised)['endpoint_native'] == [5.75, 6., 2.]
    assert opening(revised)['endpoint_cell'] == [6, 6, 2]


def test_actual_float_pose_is_identical_in_actor_and_planning_clone():
    t = task()
    row = opening(t)
    obs = t.observation()
    assert len(obs.action_ids) == 2
    assert t.planning_clone().observation().fingerprint == obs.fingerprint
    expected = build_spatial_observation(t.case.spatial_inputs(t._engine.removed_mask),
        [SpatialAction('STOP'), SpatialAction(row['action_id'], tuple(row['entry_mm']),
            tuple(row['tip_mm']), t.case.tools[0])], ObservedProcedureState(t.case.access, 0, 2, None))
    assert np.array_equal(obs.base.base.action_geometry, expected.action_geometry)
    assert obs.action_modes == ('stop', 'aspirate')


def test_subvoxel_action_seal_execution_and_exact_native_replay():
    t = task()
    action = opening(t)['action_id']
    package = seal_complete_strategy(t, [action, 'STOP'])
    history = package['strategy']['history']
    assert history[0]['entry_mm'] == [5.75, 6., 1.5]
    assert history[0]['tip_mm'] == [5.75, 6., 2.]
    assert history[0]['removed_indices_native'] == [[6, 6, 2]]
    assert package['geometryAudit']['feasible']
    assert verify_complete_strategy(t, json.loads(json.dumps(package)))
    t.step(action)
    with pytest.raises(InvalidActionError):
        t.step(action)
    t.step('STOP')
    assert semantic_digest(t.metrics()['history']) == semantic_digest(history)
    assert semantic_digest(replay_frames(t, t.metrics()['history'])) == semantic_digest(package['replayFrames'])
    assert all(f['tipRasMm'][0] == 5.75 for f in package['replayFrames'] if f['tipRasMm'] is not None)
    tampered = json.loads(json.dumps(package))
    tampered['strategy']['history'][0]['tip_mm'][0] = 6.
    tampered['strategySeal'] = semantic_digest(tampered['strategy'])
    with pytest.raises(ValueError, match='replay differs'):
        verify_complete_strategy(task(), tampered)


def test_probe_uses_committed_cell_but_engine_still_refuses_penetration():
    t = task()
    def centered_probe():
        return next(r for r in t.candidate_inventory()['ledger'] if r['family'] == 'access_centerline'
            and r['tip_mm'] == [5.75, 6., 2.] and r['interaction_mode'] == 'probe')
    before = centered_probe()
    assert before['reason'] == 'PROBE_REQUIRES_EXISTING_CAVITY'
    t.step(opening(t)['action_id'])
    after = centered_probe()
    assert after['endpoint_cell'] == [6, 6, 2]
    assert t._engine.removed_mask[6, 6, 2]
    assert after['reason'] == 'PROBE_ACTIVE_REGION_PENETRATES_REMAINING_TISSUE'
    assert not after['feasible'] and before['action_id'] != after['action_id']
    assert not t._engine.probe_contact_mask.any()


def test_contexts_versioned_and_old_context_rejected():
    revised = task()
    original = task(replace(revised.case, proposal_mode='fixed_lattice'))
    old = original.development_context('sha256:'+'a'*64)
    with pytest.raises(ValueError):
        old.require_task(revised)
    context, binding = family.bind_family_context(revised, layout_id='pcf-06', goal_id='surface', experiment_hash='sha256:'+'b'*64)
    context.require_observation(revised.observation())
    assert binding['version'] == 'generated-public-contact-family-v2'
    assert binding['source_candidate_version'] == binding['proposal_mode'] == ACCESS_CENTERLINE_PROPOSAL_VERSION
    assert binding['role'] == 'TRAIN'


def test_private_reference_does_not_choose_physical_candidates_or_legality():
    original = case()
    changed = replace(original, reference_target=original.observed_support.astype(np.float32))
    assert original.source_hash == changed.source_hash
    assert original.reference_hash != changed.reference_hash
    a, b = task(original), task(changed)
    assert a.candidate_inventory() == b.candidate_inventory()
    assert a.observation().fingerprint == b.observation().fingerprint


def test_new_mode_does_not_admit_annotation_or_deployment_tracks():
    with pytest.raises(ValueError, match='explicit generated sources'):
        replace(case(), track='annotation_assisted')
    with pytest.raises(ValueError, match='explicit generated sources'):
        replace(case(), track='deployment_scan')


def test_integer_access_deduplicates_without_extra_depths_or_private_arrays():
    source = case()
    centered = replace(source, access=AccessWindow((6., 6., 1.5), (0., 0., 1.), 1.5, source.access.window_id))
    assert len(centered._physical_candidates) == len(centered._candidate_voxels) == 15
    assert all(r['family'] == 'legacy_integer_lattice' for r in centered._physical_candidates)
    keys = {(tuple(r['entry_mm']), tuple(r['tip_mm'])) for r in centered._physical_candidates}
    assert len(keys) == 15


def test_actual_endpoint_crop_refusal_does_not_snap_into_coverage():
    source = case()
    outside = replace(source, access=AccessWindow((-.75, 6., 1.5), (0., 0., 1.), 2., source.access.window_id))
    added = [r for r in outside._physical_candidates if r['family'] == 'access_centerline']
    assert len(added) == 5
    assert all(r['voxel'] is None and r['endpoint_cell'] is None and not r['endpoint_in_actor_crop'] for r in added)
    assert all(r['endpoint_native'][0] == -.75 and r['entry_mm'][0] == -.75 for r in added)


def test_new_mode_refuses_small_source_legacy_endpoint_outside_actor_crop(monkeypatch):
    source = case()
    support = np.zeros_like(source.observed_support)
    support[6, 6, 5] = True
    source = replace(source, structural_intensity=support.astype(np.float32) * .2,
        observed_support=support, crop_shape=(3, 3, 3))
    assert source._candidate_scope.startswith('all_observed_support_cells_in_small_source')
    def forbidden(*args, **kwargs):
        raise AssertionError('Outside-crop candidates must not reach native preview')
    monkeypatch.setattr(NativeResectionEngine, 'preview_stroke', forbidden)
    t = NativeSpatialTask(source, max_steps=2, tool_modes=MODES)
    assert t.observation().action_ids == ('STOP',)
    rows = t.candidate_inventory()['ledger']
    retained = [r for r in rows if r['family'] == 'legacy_integer_lattice']
    assert len(retained) == 2
    assert all(r['voxel'] == [6, 6, 5] and r['tip_mm'] == [6., 6., 5.] for r in retained)
    assert all(r['endpoint_cell'] is None and not r['endpoint_in_actor_crop'] for r in rows)
    assert all(not r['feasible'] and r['reason'] == 'ENDPOINT_OUTSIDE_ACTOR_CROP' for r in rows)


def test_all_recipes_roles_and_source_arrays_match_frozen_v1_without_task_execution(monkeypatch):
    def forbidden(*args, **kwargs): raise AssertionError('Source-only all-layout comparison')
    monkeypatch.setattr(NativeSpatialTask, '__init__', forbidden)
    record = family.family_record()
    assert semantic_digest(record['layouts']) == 'sha256:72e9fc974433a707c58872d8d6833f021a0a436ce22fe77966b2af9884ba18bd'
    manifest = family.family_manifest()
    rows = [{k:r[k] for k in ('layout_id','role','recipe_hash','support_hash','structural_intensity_hash','nominal_target_hash','affine_hash')} for r in manifest['source_bindings']]
    assert semantic_digest(rows) == 'sha256:70aeec4dfdd7d70fde343533f73b7f20d1b1f22bd09d85bb5a2435e17d3eb83f'
    assert all(r['decision_model_hash'] is None for r in manifest['source_bindings'])
