"""Native staged-skill prefix, budget and exact-resume contracts."""
from __future__ import annotations

import copy
import json
import sys

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from resectionlab.core import semantic_digest
from resectionlab.surgical_engine import SurgicalEngine
from resectionlab.surgical_cusa_v3 import default_profile, initialize
from resectionlab.surgical_scene import SurgicalScene
from resectionlab.surgical_staged_skills import (
    SkillAction, SkillLimits, checked_actions, checked_cusa_motion, checked_exit_actions,
    cusa_work_actions, external_cusa_reconfiguration_actions,
    make_skill_checkpoint, propose_cusa_maneuvers, restore_skill_checkpoint)
from resectionlab.surgical_transition_meter import TransitionBudgetExceeded, TransitionMeter
from resectionlab.surgical_verified_cache import implementation_fingerprint


INSPECT = {'slot': 'left', 'operation': 'inspect', 'duration': .2}
INVALID = {'slot': 'left', 'operation': 'unsupported', 'duration': .2}
SEALS = {'source_seal': 'sha256:source', 'task_seal': 'sha256:task',
         'inventory_seal': 'sha256:source-skill-inventory'}


def _engine():
    labels = np.zeros((3, 3, 3), np.uint16)
    labels[1, 1, 1] = 1
    scene = SurgicalScene(labels, np.eye(4),
        [{'id': 1, 'name': 'generated target', 'material': 'tumor', 'target': True}],
        {'kind': 'generated staged-skill test'}, {})
    return SurgicalEngine(scene)


def _maneuver_engine():
    labels = np.zeros((20, 20, 20), np.uint16)
    labels[10, 10, 7:9] = 1
    labels[11, 10, 8] = 2
    scene = SurgicalScene(labels, np.eye(4), [
        {'id': 1, 'name': 'generated target', 'material': 'tumor', 'target': True},
        {'id': 2, 'name': 'brain', 'material': 'soft_tissue'}],
        {'kind': 'generated current-occupancy maneuver coupon'}, {})
    engine = SurgicalEngine(scene)
    pose = lambda z: {'position': [10., 10., z],
                      'quaternion': [0., 0., 0., 1.]}
    # Seed a geometric current-state coupon. The proposer must never depend on
    # how the upstream checked corridor reached this inserted pose.
    engine.state['slots']['left'].update(instrument='cusa',
        position=[10., 10., 6.], quaternion=[0., 0., 0., 1.], inserted=True)
    route = {'sourceSceneId': scene.identity, 'surfacePose': pose(2.),
             'entryPose': pose(6.),
             'inwardRasUnit': [0., 0., 1.]}
    return engine, route


def test_local_cusa_proposals_are_bounded_deterministic_and_read_only():
    engine, route = _maneuver_engine()
    meter = TransitionMeter(20)
    wrapped = meter.wrap(engine)
    before = engine.snapshot()
    source = engine.scene.labels.copy()
    preview_code, step_code = SurgicalEngine.preview.__code__, SurgicalEngine.step.__code__
    calls = []
    def profile(frame, event, arg):
        if event == 'call' and frame.f_code in (preview_code, step_code):
            calls.append(frame.f_code.co_name)
    sys.setprofile(profile)
    try:
        proposals = propose_cusa_maneuvers(wrapped, route)
        assert proposals == propose_cusa_maneuvers(wrapped, route)
    finally:
        sys.setprofile(None)
    assert not calls and meter.calls == 0
    assert engine.snapshot() == before
    assert np.array_equal(engine.scene.labels, source)
    assert 0 < len(proposals) <= 15
    assert len({row['id'] for row in proposals}) == len(proposals)
    assert any(row['orientationMode'] == 'local-forward' for row in proposals)
    assert any(row['orientationMode'] == 'local-surface-pivot' for row in proposals)
    current = np.asarray(engine.state['slots']['left']['position'])
    current_q = Rotation.from_quat(engine.state['slots']['left']['quaternion'])
    anchor = np.asarray(route['surfacePose']['position'])
    for row in proposals:
        assert row['proposalStateId'] == before['stateId']
        assert row['pose'] == row['path'][-1]
        assert row['currentGeometry'] == {'radiusMm': 1.5,
            'depthMm': 2., 'shaftRadiusMm': .5}
        assert row['depthFromTargetEntryMm'] == pytest.approx(
            row['pose']['position'][2]-route['entryPose']['position'][2])
        footprint = row['sourceFootprints']['1.5']
        assert footprint['sourceTargetMm3'] > 0
        previous, previous_q = current, current_q
        for pose in row['path']:
            point = np.asarray(pose['position'])
            orientation = Rotation.from_quat(pose['quaternion'])
            assert np.linalg.norm(point-previous) <= .5+1e-9
            assert (orientation*previous_q.inv()).magnitude() <= np.deg2rad(.5)+1e-9
            if row['orientationMode'] == 'local-surface-pivot':
                ray = (point-anchor)/np.linalg.norm(point-anchor)
                assert np.allclose(orientation.apply([0., 0., 1.]), ray)
            previous, previous_q = point, orientation


def test_local_cusa_proposals_follow_current_removed_cells_and_route_binding():
    engine, route = _maneuver_engine()
    original = propose_cusa_maneuvers(engine, route)
    removed = engine.clone()
    removed.state['removed'] = [int(np.ravel_multi_index((10, 10, 7),
                                      removed.scene.shape))]
    updated = propose_cusa_maneuvers(removed, route)
    assert updated and all(row['proposalStateId'] == removed.state_id for row in updated)
    assert {row['id'] for row in updated}.isdisjoint(row['id'] for row in original)
    assert max(row['sourceFootprints']['1.5']['sourceTargetMm3'] for row in updated) < \
        max(row['sourceFootprints']['1.5']['sourceTargetMm3'] for row in original)
    assert all(row['sourceFootprints']['1.5']['sourceTargetMm3'] > 0 for row in updated)
    removed.state['removed'] = [int(np.ravel_multi_index((10, 10, z),
                                  removed.scene.shape)) for z in (7, 8)]
    assert propose_cusa_maneuvers(removed, route) == []
    with pytest.raises(ValueError, match='source-bound'):
        propose_cusa_maneuvers(engine, {**route, 'sourceSceneId': 'wrong'})
    with pytest.raises(ValueError, match='local bounds'):
        propose_cusa_maneuvers(engine, route, max_steps=5)
    with pytest.raises(ValueError, match='local bounds'):
        propose_cusa_maneuvers(engine, route, angle_step_deg=1.)


def test_local_pivot_preserves_current_roll_and_option_bound():
    engine, route = _maneuver_engine()
    engine.state['slots']['left']['quaternion'] = Rotation.from_euler(
        'z', 37., degrees=True).as_quat().tolist()
    proposals = propose_cusa_maneuvers(engine, route, max_options=3)
    assert len(proposals) == 3
    all_options = propose_cusa_maneuvers(engine, route)
    pivots = [row for row in all_options
              if row['orientationMode'] == 'local-surface-pivot']
    assert pivots
    start = Rotation.from_quat(engine.state['slots']['left']['quaternion'])
    for row in pivots:
        first = Rotation.from_quat(row['path'][0]['quaternion'])
        assert (first*start.inv()).magnitude() == pytest.approx(np.deg2rad(.5))


def test_local_work_keeps_the_source_inventory_external_shaft_scope():
    engine, route = _maneuver_engine()
    route['surfacePose']['position'][2] = -32.
    rows = propose_cusa_maneuvers(engine, route)
    assert all(row['orientationMode'] != 'local-forward' for row in rows)
    assert all(np.linalg.norm(np.subtract(pose['position'], route['surfacePose']['position'])) <= 38.+1e-9
               for row in rows for pose in row['path'])


def _reseal(checkpoint):
    checkpoint['checkpointSeal'] = semantic_digest({
        key: value for key, value in checkpoint.items() if key != 'checkpointSeal'})
    return checkpoint


def test_checked_prefix_refusal_keeps_original_engine_and_exact_public_call_count():
    original = _engine()
    original_snapshot = original.snapshot()
    fingerprint = implementation_fingerprint()
    meter = TransitionMeter(20)
    preview_code, step_code = SurgicalEngine.preview.__code__, SurgicalEngine.step.__code__
    prepare_code = SurgicalEngine._prepare_step.__code__
    observed = {'preview': 0, 'previewInternalStep': 0, 'directStep': 0,
                'previewPreparations': 0, 'stepPreparations': 0}

    def profile(frame, event, arg):
        if event != 'call':
            return
        if frame.f_code is preview_code:
            observed['preview'] += 1
        elif frame.f_code is step_code:
            key = ('previewInternalStep' if frame.f_back is not None and
                   frame.f_back.f_code is preview_code else 'directStep')
            observed[key] += 1
        elif frame.f_code is prepare_code:
            key = 'previewPreparations' if frame.f_back.f_code is preview_code else 'stepPreparations'
            observed[key] += 1

    sys.setprofile(profile)
    try:
        candidate, receipt = checked_actions(original, [INSPECT, INVALID],
            limits=SkillLimits(8, 20), meter=meter)
    finally:
        sys.setprofile(None)
    assert receipt['status'] == 'refused'
    assert receipt['retentionRole'] == 'stopped_native_prefix_not_admitted'
    assert receipt['admittedSkillProgressMm3'] == 0.
    assert receipt['nativeActionsAppended'] == 1
    assert receipt['attempts'][0]['status'] == 'committed'
    assert receipt['attempts'][1]['status'] == 'refused'
    assert receipt['attempts'][1]['previewRemovedCellIds'] == []
    assert candidate.actions == [INSPECT]
    assert original.snapshot() == original_snapshot
    assert receipt['committedPrefix']['actionPrefixSeal'] == semantic_digest([INSPECT])
    assert observed == {'preview': 2, 'previewInternalStep': 0, 'directStep': 1,
                        'previewPreparations': 2, 'stepPreparations': 1}
    assert receipt['physicalCallCost'] == {'publicCalls': 3,
        'previewCalls': 2, 'directStepCalls': 1}
    assert implementation_fingerprint() == fingerprint


def test_budget_and_explicit_cancellation_return_last_accepted_prefix():
    meter = TransitionMeter(2)
    base = meter.wrap(_engine())
    limits = SkillLimits(8, 2)
    stopped, receipt = checked_actions(base, [INSPECT, INSPECT],
                                        limits=limits, meter=meter)
    assert receipt['status'] == 'interrupted'
    assert receipt['nativeActionsAppended'] == 1
    assert stopped.actions == [INSPECT] and base.actions == []
    assert receipt['ledger']['remainingPublicCalls'] == 0
    assert meter.calls == 2

    meter = TransitionMeter(10)
    base = meter.wrap(_engine())
    def cancel_after_commit():
        if any(row.get('actionsAfter') == 1 for row in meter.trace):
            raise TimeoutError('interrupted after first committed action')
    stopped, receipt = checked_actions(base, [INSPECT, INSPECT],
        limits=SkillLimits(8, 10), meter=meter, check=cancel_after_commit)
    assert receipt['status'] == 'interrupted'
    assert receipt['nativeActionsAppended'] == 1
    assert stopped.actions == [INSPECT]
    assert receipt['physicalCallCost']['publicCalls'] == 2


def test_versioned_checkpoint_cold_replay_reconstructs_unique_ledger_and_budgets():
    limits = SkillLimits(8, 20)
    meter = TransitionMeter(20)
    original = meter.wrap(_engine())
    branch, receipt = checked_actions(original, [INSPECT],
                                      limits=limits, meter=meter)
    assert receipt['status'] == 'accepted'
    checkpoint = make_skill_checkpoint(branch, limits=limits, meter=meter,
                                       stage='outside', **SEALS)
    saved = json.loads(json.dumps(checkpoint))
    restored, continued, ledger = restore_skill_checkpoint(
        original.scene, saved, **SEALS)
    assert restored.state_id == branch.state_id
    assert restored.actions == branch.actions
    assert ledger['removedCellCount'] == 0
    assert continued.prior_public_calls == 2
    assert ledger['spentPublicCalls'] == 3  # Cold replay itself is a direct native call.
    assert ledger['remainingNativeActions'] == 7
    assert ledger['remainingPublicCalls'] == 17
    resumed, result = checked_actions(restored, [INSPECT], limits=continued)
    assert result['status'] == 'accepted'
    assert resumed.actions == [INSPECT, INSPECT]
    assert result['ledger']['spentPublicCalls'] == 5
    assert original.actions == []

    same_meter, same_limits, same_ledger = restore_skill_checkpoint(
        original.scene, saved, meter=meter, **SEALS)
    assert same_meter.state_id == branch.state_id
    assert same_limits.prior_public_calls == 0
    assert same_ledger['spentPublicCalls'] == 3
    assert meter.calls == 3


def test_checkpoint_rejects_corrupt_prefix_state_seals_and_exhausted_replay_budget():
    limits = SkillLimits(8, 20)
    meter = TransitionMeter(20)
    branch, _ = checked_actions(_engine(), [INSPECT], limits=limits, meter=meter)
    checkpoint = make_skill_checkpoint(branch, limits=limits, meter=meter,
                                       stage='outside', **SEALS)
    scene = branch.scene
    corrupt = copy.deepcopy(checkpoint)
    corrupt['nativeSnapshot']['actions'][0]['duration'] = .3
    with pytest.raises(ValueError, match='content seal'):
        restore_skill_checkpoint(scene, corrupt, **SEALS)
    with pytest.raises(ValueError, match='binding differs'):
        restore_skill_checkpoint(scene, checkpoint, **{**SEALS,
                                 'inventory_seal': 'changed'})
    corrupt = _reseal(copy.deepcopy(corrupt))
    with pytest.raises(ValueError, match='action prefix differs'):
        restore_skill_checkpoint(scene, corrupt, **SEALS)
    corrupt = copy.deepcopy(checkpoint)
    corrupt['nativeSnapshot']['checkpoint']['state']['slots']['left']['active'] = True
    corrupt = _reseal(corrupt)
    with pytest.raises(ValueError, match='differs from full replay'):
        restore_skill_checkpoint(scene, corrupt, **SEALS)
    with pytest.raises(ValueError, match='reset or replace'):
        restore_skill_checkpoint(scene, checkpoint,
            meter=TransitionMeter(20), **SEALS)

    tight = SkillLimits(8, 2)
    tight_meter = TransitionMeter(2)
    tight_branch, _ = checked_actions(_engine(), [INSPECT],
                                      limits=tight, meter=tight_meter)
    tight_checkpoint = make_skill_checkpoint(tight_branch, limits=tight,
        meter=tight_meter, stage='outside', **SEALS)
    with pytest.raises(TransitionBudgetExceeded, match='No native calls remain'):
        restore_skill_checkpoint(tight_branch.scene, tight_checkpoint, **SEALS)


def test_source_defined_action_builders_keep_pose_radius_and_config_fixed():
    pose = {'position': [1., 2., 3.], 'quaternion': [0., 0., 0., 1.]}
    outside = {'position': [1., 2., -2.], 'quaternion': [0., 0., 0., 1.]}
    config = {'power': 1., 'flowMm3PerSecond': 20., 'duty': .5}
    work = cusa_work_actions('left', pose, radius_mm=1.5,
        duration_seconds=.2, parameters=config, transit_pose=outside,
        retry_durations_seconds=(.6, 1., 2.))
    exit_actions = checked_exit_actions('left', outside, radius_mm=1.5,
        withdraw_duration_seconds=16., waypoints=((pose, 2.),))
    assert [row.action['operation'] for row in work] == ['apply', 'apply']
    assert [row.action['operation'] for row in exit_actions] == [
        'deactivate', 'move', 'withdraw']
    assert all(row.action.get('parameters', {}).get('radiusMm', 1.5) == 1.5
               for row in (*work, *exit_actions))
    assert work[-1].retry_durations_seconds == (.6, 1., 2.)
    assert work[-1].action['pose'] == pose
    assert work[-1].action['parameters']['flowMm3PerSecond'] == 20.
    config['power'] = 9.
    pose['position'][0] = 99.
    assert work[-1].action['parameters']['power'] == 1.
    assert work[-1].action['pose']['position'][0] == 1.
    with pytest.raises(ValueError, match='Duration retries'):
        SkillAction(INSPECT, (.1,))
    with pytest.raises(ValueError, match='both pose and path'):
        SkillAction({'slot': 'left', 'operation': 'apply', 'pose': outside,
                     'path': [pose]}, require_endpoint=True)
    with pytest.raises(ValueError, match='requires a native pose or path'):
        SkillAction({'slot': 'left', 'operation': 'apply', 'path': []}, require_endpoint=True)


@pytest.mark.parametrize('limits, reason', [
    (SkillLimits(20, 100, max_total_non_target_mm3=0.,
                 allowed_removed_label_ids=(1,)), 'TOTAL_NON_TARGET_CAP'),
    (SkillLimits(20, 100, max_additional_non_target_mm3=0.,
                 baseline_non_target_mm3=0.,
                 allowed_removed_label_ids=(1,)), 'ADDITIONAL_NON_TARGET_CAP'),
])
def test_real_native_cusa_preview_refuses_first_non_target_cell_under_zero_cap(
        limits, reason):
    labels = np.zeros((20, 20, 20), np.uint16)
    labels[9, 9, 9] = 1
    scene = SurgicalScene(labels, np.eye(4),
        [{'id': 1, 'name': 'brain', 'material': 'soft_tissue', 'target': False}],
        {'kind': 'generated staged CUSA cap coupon'},
        {'cusaV3': default_profile()})
    engine = SurgicalEngine(scene)
    initialize(engine)
    pose = lambda z: {'position': [9., 9., z],
                      'quaternion': [0., 0., 0., 1.]}
    stroke = {'slot': 'left', 'operation': 'apply', 'duration': .1,
              'pose': pose(11.)}
    actions = [
        {'slot': 'left', 'operation': 'exchange', 'instrument': 'cusa', 'duration': .1},
        {'slot': 'left', 'operation': 'place', 'duration': .1, 'pose': pose(3.)},
        *[stroke for _ in range(12)]]
    candidate, receipt = checked_actions(engine, actions, limits=limits)
    assert receipt['status'] == 'refused'
    assert receipt['reason'] == reason
    assert receipt['admittedSkillProgressMm3'] == 0.
    assert receipt['attempts'][-1]['previewRemovedCellIds']
    assert candidate.metrics()['nonTargetRemovedMm3'] == 0.
    assert candidate.state['removed'] == []
    assert engine.actions == []


def _legacy_inserted_cusa():
    labels = np.zeros((16, 16, 16), np.uint16)
    labels[6:9, 6:9, 6:9] = 1
    scene = SurgicalScene(labels, np.eye(4),
        [{'id': 1, 'name': 'target', 'material': 'tumor', 'target': True}],
        {'kind': 'generated legacy geometry guard coupon'}, {})
    engine = SurgicalEngine(scene)
    pose = lambda z: {'position': [6., 6., z],
                      'quaternion': [0., 0., 0., 1.]}
    for action in (
            {'slot': 'left', 'operation': 'exchange', 'instrument': 'cusa',
             'duration': .1},
            {'slot': 'left', 'operation': 'place', 'pose': pose(2.),
             'parameters': {'radiusMm': 1.5, 'depthMm': 2.}, 'duration': .1},
            {'slot': 'left', 'operation': 'apply', 'pose': pose(8.),
             'parameters': {'radiusMm': 1.5}, 'duration': .2}):
        assert engine.step(action)['status'] == 'accepted'
    assert engine.state['slots']['left']['inserted']
    return engine, pose


@pytest.mark.parametrize('parameters', [
    {'radiusMm': 2.2}, {'depthMm': 3.}, {'shaftRadiusMm': .8},
])
def test_inserted_hand_cannot_change_working_geometry_even_in_legacy_physics(
        parameters):
    engine, pose = _legacy_inserted_cusa()
    original = engine.snapshot()
    meter = TransitionMeter(10)
    candidate, receipt = checked_actions(engine, [
        {'slot': 'left', 'operation': 'apply', 'pose': pose(8.),
         'parameters': parameters, 'duration': .2}],
        limits=SkillLimits(20, 10), meter=meter)
    assert receipt['status'] == 'refused'
    assert receipt['reason'] == 'WITHDRAW_BEFORE_GEOMETRY_CHANGE'
    assert receipt['physicalCallCost']['publicCalls'] == 0
    assert receipt['nativeActionsAppended'] == 0
    assert receipt['admittedSkillProgressMm3'] == 0.
    assert candidate.snapshot() == original == engine.snapshot()


def test_external_reconfiguration_requires_cleanup_and_native_checked_exchange_place():
    outside = {'position': [6., 6., -5.],
               'quaternion': [0., 0., 0., 1.]}
    inserted, _ = _legacy_inserted_cusa()
    with pytest.raises(ValueError, match='CHECKED_CLEANUP_REQUIRED'):
        external_cusa_reconfiguration_actions(inserted, 'left', outside,
                                               radius_mm=2.2)
    clean = _engine()
    actions = external_cusa_reconfiguration_actions(clean, 'left', outside,
        radius_mm=2.2, depth_mm=3.)
    candidate, receipt = checked_actions(clean, actions,
        limits=SkillLimits(8, 20))
    assert receipt['status'] == 'accepted'
    assert receipt['physicalCallCost'] == {'publicCalls': 4,
        'previewCalls': 2, 'directStepCalls': 2}
    assert candidate.state['slots']['left']['instrument'] == 'cusa'
    assert candidate.state['slots']['left']['parameters'] == {
        'radiusMm': 2.2, 'depthMm': 3.}
    assert not candidate.state['slots']['left']['inserted']
    assert clean.actions == []

    labels = np.zeros((3, 3, 3), np.uint16)
    cusa3_scene = SurgicalScene(labels, np.eye(4), [],
        {'kind': 'generated fixed-catalog CUSA v3 guard'},
        {'cusaV3': default_profile()})
    cusa3 = SurgicalEngine(cusa3_scene)
    with pytest.raises(ValueError, match='CUSA_V3_FIXED_CATALOG_GEOMETRY_REQUIRED'):
        external_cusa_reconfiguration_actions(cusa3, 'left', outside,
                                               radius_mm=2.2)
    catalog_actions = external_cusa_reconfiguration_actions(cusa3, 'left', outside,
                                                              radius_mm=1.5)
    catalog, catalog_receipt = checked_actions(cusa3, catalog_actions,
        limits=SkillLimits(8, 20))
    assert catalog_receipt['status'] == 'accepted'
    assert catalog.state['slots']['left']['parameters']['radiusMm'] == 1.5


def test_resume_rejects_a_legacy_native_prefix_with_inserted_tool_resize():
    legacy, pose = _legacy_inserted_cusa()
    resized = {'slot': 'left', 'operation': 'apply', 'pose': pose(8.),
               'parameters': {'radiusMm': 2.2}, 'duration': .2}
    assert legacy.step(resized)['status'] == 'accepted'  # Older native v2 permits it.
    meter = TransitionMeter(12)
    wrapped = meter.wrap(legacy)
    limits = SkillLimits(20, 20, prior_public_calls=8)
    checkpoint = make_skill_checkpoint(wrapped, limits=limits, meter=meter,
                                       stage='working', **SEALS)
    with pytest.raises(ValueError, match='inserted working geometry'):
        restore_skill_checkpoint(legacy.scene, checkpoint, **SEALS)


def test_partial_capacity_motion_stops_before_a_dwell_can_cut_a_different_path():
    labels = np.zeros((20, 20, 20), np.uint16)
    labels[9, 9, 9] = 1
    scene = SurgicalScene(labels, np.eye(4),
        [{'id': 1, 'name': 'target', 'material': 'tumor', 'target': True}],
        {'kind': 'generated partial path contract'}, {'cusaV3': default_profile()})
    engine = SurgicalEngine(scene)
    initialize(engine)
    pose = lambda x, z: {'position': [x, 9., z], 'quaternion': [0., 0., 0., 1.]}
    for action in (
        {'slot': 'left', 'operation': 'exchange', 'instrument': 'cusa', 'duration': .1},
        {'slot': 'left', 'operation': 'place', 'pose': pose(9., 3.), 'duration': .1}):
        assert engine.step(action)['status'] == 'accepted'
    path = [pose(9., 10.), pose(10., 10.)]
    work = SkillAction({'slot': 'left', 'operation': 'apply', 'path': path,
                        'duration': .1}, require_endpoint=True)
    dwell = {'slot': 'left', 'operation': 'apply', 'pose': path[-1], 'duration': .1}
    candidate, receipt = checked_actions(engine, [work, dwell], limits=SkillLimits(20, 20))
    assert receipt['status'] == 'interrupted'
    assert receipt['reason'] == 'PARTIAL_MOTION_REPLAN_REQUIRED'
    assert receipt['nativeActionsAppended'] == 1
    assert receipt['attempts'][0]['previewStatus'] == 'partial'
    assert receipt['attempts'][0]['endpointReached'] is False
    assert receipt['admittedSkillProgressMm3'] == 0.
    assert candidate.actions[-1]['path'] == path
    assert len(engine.actions) == 2


def test_effect_caps_apply_to_existing_prefix_checkpoint_and_cold_restore():
    labels = np.zeros((20, 20, 20), np.uint16)
    labels[9, 9, 9] = 1
    scene = SurgicalScene(labels, np.eye(4),
        [{'id': 1, 'name': 'brain', 'material': 'soft_tissue'}],
        {'kind': 'generated existing prefix cap contract'}, {'cusaV3': default_profile()})
    engine = SurgicalEngine(scene)
    initialize(engine)
    pose = lambda z: {'position': [9., 9., z], 'quaternion': [0., 0., 0., 1.]}
    actions = [
        {'slot': 'left', 'operation': 'exchange', 'instrument': 'cusa', 'duration': .1},
        {'slot': 'left', 'operation': 'place', 'duration': .1, 'pose': pose(3.)},
        *[{'slot': 'left', 'operation': 'apply', 'duration': .1,
           'pose': pose(11.)} for _ in range(12)]]
    branch, receipt = checked_actions(engine, actions, limits=SkillLimits(30, 100))
    assert branch.metrics()['nonTargetRemovedMm3'] == 1.
    checkpoint = make_skill_checkpoint(branch, limits=SkillLimits(30, 100), **SEALS)
    tight = SkillLimits(30, 100, max_additional_non_target_mm3=0.,
                        baseline_non_target_mm3=0.)
    with pytest.raises(ValueError, match='additional non-target cap'):
        checked_actions(branch, [INSPECT], limits=tight)
    with pytest.raises(ValueError, match='additional non-target cap'):
        make_skill_checkpoint(branch, limits=tight, **SEALS)
    checkpoint['limits']['max_additional_non_target_mm3'] = 0.
    checkpoint['limits']['baseline_non_target_mm3'] = 0.
    with pytest.raises(ValueError, match='additional non-target cap'):
        restore_skill_checkpoint(scene, _reseal(checkpoint), **SEALS)


@pytest.mark.parametrize('rotate', [False, True])
def test_collision_prefix_is_rechecked_and_dwell_uses_actual_pose_without_shortcuts(rotate):
    labels = np.zeros((20, 20, 20), np.uint16)
    labels[8, 6, 8] = 1
    labels[10, 6, 4] = 2
    scene = SurgicalScene(labels, np.eye(4), [
        {'id': 1, 'name': 'target', 'material': 'tumor', 'target': True},
        {'id': 2, 'name': 'brain', 'material': 'soft_tissue'}],
        {'kind': 'generated guarded collision-prefix path'}, {})
    engine = SurgicalEngine(scene)
    pose = lambda x, y, z: {'position': [float(x), float(y), float(z)],
                           'quaternion': [0., 0., 0., 1.]}
    for action in (
        {'slot': 'left', 'operation': 'exchange', 'instrument': 'cusa', 'duration': .1},
        {'slot': 'left', 'operation': 'place', 'pose': pose(6,6,2),
         'parameters': {'radiusMm': .6}, 'duration': .1},
        {'slot': 'left', 'operation': 'apply', 'pose': pose(6,6,8),
         'parameters': {'radiusMm': .6}, 'duration': 1.}):
        assert engine.step(action)['status'] == 'accepted'
    before = engine.snapshot()
    requested = {'slot': 'left', 'operation': 'apply',
        'path': [pose(12,6,8), pose(12,10,8)],
        'parameters': {'radiusMm': .6}, 'duration': 2.}
    if rotate:
        for endpoint in requested['path']:
            endpoint['quaternion'] = Rotation.from_euler('z', 25., degrees=True).as_quat().tolist()
    preview = engine.preview(requested)
    assert preview['status'] == 'partial' and preview['reason'] == 'SHAFT_COLLISION'
    assert preview['deltas']['removedCells']
    meter = TransitionMeter(50)
    branch, receipt = checked_cusa_motion(engine,
        SkillAction(requested, require_endpoint=True), limits=SkillLimits(20, 50,
            max_total_non_target_mm3=0.), meter=meter)
    assert receipt['status'] == 'accepted'
    assert receipt['executionMode'] == 'checked_collision_prefix'
    assert not receipt['requestedPoseReached']
    assert receipt['admittedSkillProgressMm3'] == 1.
    assert np.allclose(receipt['observedEndPose']['position'], receipt['clippedProposal']['path'][-1]['position'])
    assert np.allclose(receipt['observedEndPose']['quaternion'], receipt['clippedProposal']['path'][-1]['quaternion'])
    assert np.allclose(branch.actions[-1]['pose']['position'], receipt['observedEndPose']['position'])
    assert np.allclose(branch.actions[-1]['pose']['quaternion'], receipt['observedEndPose']['quaternion'])
    assert len(branch.actions[-2]['path']) == 1
    assert branch.actions[-2]['path'][-1]['position'][0] < 10.
    assert branch.actions[-2]['path'][-1]['position'][1] == 6.
    assert all(x['status'] == 'accepted' for x in branch.history)
    assert branch.metrics()['nonTargetRemovedMm3'] == 0.
    assert SurgicalEngine.restore(scene, branch.snapshot()).state == branch.state
    assert engine.snapshot() == before
    assert receipt['physicalCallCost']['publicCalls'] == meter.calls
    from resectionlab.surgical_staged_skills import _clip_cusa_motion
    corrupted_contact = copy.deepcopy(preview)
    corrupted_contact['contactPosition'] = [999.,999.,999.]
    assert _clip_cusa_motion(engine, requested, corrupted_contact)['path'][-1] == {
        key: preview['view']['slots']['left'][key] for key in ('position', 'quaternion')}
    denied, failure = checked_cusa_motion(engine,
        SkillAction(requested, require_endpoint=True), limits=SkillLimits(20, 50,
            allowed_removed_label_ids=(2,)))
    assert failure['status'] == 'refused' and failure['reason'] == 'REMOVAL_LABEL_NOT_ALLOWED'
    assert failure['admittedSkillProgressMm3'] == 0.
    assert denied.snapshot() == before


def test_motion_boundary_probe_and_dwell_share_call_and_action_budgets():
    engine, pose = _legacy_inserted_cusa()
    original = engine.snapshot()
    motion = SkillAction({'slot': 'left', 'operation': 'apply', 'pose': pose(8.),
        'parameters': {'radiusMm': 1.5}, 'duration': .2}, require_endpoint=True)
    meter = TransitionMeter(1)
    stopped, receipt = checked_cusa_motion(engine, motion, limits=SkillLimits(20,1), meter=meter)
    assert receipt['status'] == 'interrupted'
    assert meter.calls == 1 and receipt['physicalCallCost']['publicCalls'] == 1
    assert receipt['admittedSkillProgressMm3'] == 0.
    assert stopped.snapshot() == original
    stopped, receipt = checked_cusa_motion(engine, motion,
        limits=SkillLimits(len(engine.actions)+1,20))
    assert receipt['status'] == 'interrupted' and receipt['reason'] == 'NATIVE_ACTION_BUDGET'
    assert len(stopped.actions) == len(engine.actions)+1
    assert stopped.actions[-1] == motion.action
    assert receipt['admittedSkillProgressMm3'] == 0.
    assert engine.snapshot() == original
