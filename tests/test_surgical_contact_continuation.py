"""Bounded finite aspiration through public native steps, including path stops."""
import copy
import sys

import numpy as np
import pytest

from resectionlab.surgical_cusa_v3 import default_profile
from resectionlab.surgical_engine import SurgicalEngine
from resectionlab.surgical_scene import SurgicalScene
from resectionlab.surgical_rl_protection import ProtectionAnnotation, ProtectionSet, GuardConfig
from resectionlab.surgical_sequence_search import checked_sequence_search
from resectionlab.core import semantic_digest
from resectionlab.surgical_staged_skills import (
    SkillAction, SkillLimits, checked_actions, checked_cusa_motion)
from resectionlab.surgical_transition_meter import TransitionMeter


def pose(x=9., z=11.):
    return {'position': [float(x), 9., float(z)], 'quaternion': [0., 0., 0., 1.]}


def source(*, material='tumor', second=False):
    labels = np.zeros((24, 24, 24), np.uint16); labels[9, 9, 9] = 1
    if second: labels[13, 9, 13] = 1
    scene = SurgicalScene(labels, np.eye(4), [{'id': 1, 'name': 'target' if
        material == 'tumor' else 'brain', 'material': material, 'target': material == 'tumor'}],
        {'kind': 'GENERATED finite-contact software control'}, {'cusaV3': default_profile()})
    engine = SurgicalEngine(scene)
    for action in ({'slot': 'left', 'operation': 'exchange', 'instrument': 'cusa', 'duration': .1},
                   {'slot': 'left', 'operation': 'place', 'pose': pose(z=3.), 'duration': .1}):
        assert engine.step(action)['status'] == 'accepted'
    return engine


def motion(**extra):
    return SkillAction({'slot': 'left', 'operation': 'apply', 'duration': .05,
                        'pose': pose(), **extra})


def run(engine, action=None, *, replans=8, calls=200, limits=None):
    meter = TransitionMeter(calls)
    candidate, receipt = checked_cusa_motion(engine, action or motion(),
        limits=limits or SkillLimits(80, calls), meter=meter, dwell_seconds=.05,
        max_contact_replans=replans)
    return candidate, receipt, meter


def test_native_contact_clearance_completes_motion_and_exact_exit_replay():
    engine = source(); before = engine.snapshot()
    old, stopped, _ = run(engine, replans=0)
    assert stopped['status'] == 'interrupted' and stopped['reason'] == 'PARTIAL_MOTION_REPLAN_REQUIRED'
    assert old.state['cusaV3']['progressExact'] and old.metrics()['targetRemovedMm3'] == 0
    observed = {'preview': 0, 'step': 0}
    def profile(frame, event, arg):
        if event == 'call':
            if frame.f_code is SurgicalEngine.preview.__code__: observed['preview'] += 1
            if frame.f_code is SurgicalEngine.step.__code__: observed['step'] += 1
    sys.setprofile(profile)
    try:
        work, receipt, meter = run(engine)
    finally:
        sys.setprofile(None)
    assert receipt['status'] == 'accepted', receipt['reason']
    assert receipt['requestedPoseReached'] and receipt['admittedSkillProgressMm3'] == 1.
    assert len(receipt['contactContinuations']) == 1
    assert work.state['slots']['left']['position'] == pose()['position']
    assert not work.state['cusaV3']['progressExact']
    assert receipt['physicalCallCost']['publicCalls'] == meter.calls
    assert observed == {'preview': meter.preview_calls, 'step': meter.direct_step_calls}
    assert meter.preview_calls == meter.direct_step_calls+1  # Safe work still previews every commit.
    assert any(row['status'] == 'partial' for row in work.history)
    exited, exit_receipt = checked_actions(work,
        [SkillAction({'slot': 'left', 'operation': 'withdraw', 'pose': pose(z=3.), 'duration': 2.},
                     require_endpoint=True)], limits=SkillLimits(80, 200), meter=meter)
    assert exit_receipt['status'] == 'accepted'
    assert not exited.state['slots']['left']['inserted']
    assert SurgicalEngine.restore(engine.scene, exited.snapshot()).snapshot() == exited.snapshot()
    assert engine.snapshot() == before


def test_contact_in_second_leg_never_repeats_or_skips_drawn_waypoints():
    engine = source(second=True)
    action = SkillAction({'slot': 'left', 'operation': 'apply', 'duration': .05,
                         'path': [pose(z=5.), pose(), pose(x=13., z=13.)]})
    work, receipt, _ = run(engine, action)
    assert receipt['status'] == 'accepted', receipt['reason']
    assert work.metrics()['targetRemovedMm3'] == 2.
    continuations = receipt['contactContinuations']
    assert len(continuations) == 2
    assert continuations[0]['remainingAction']['path'] == [pose(), pose(x=13., z=13.)]
    assert continuations[1]['remainingAction']['path'] == [pose(x=13., z=13.)]
    assert work.state['slots']['left']['position'] == [13., 9., 13.]


def test_matching_last_pose_does_not_claim_an_unfinished_loop_was_executed():
    engine = source(); first = engine.step(motion().action)
    assert first['reason'] == 'TISSUE_ASPIRATION_IN_PROGRESS'
    current = {key: copy.deepcopy(engine.state['slots']['left'][key])
               for key in ('position', 'quaternion')}
    action = SkillAction({'slot': 'left', 'operation': 'apply', 'duration': .05,
                         'path': [pose(), current]}, require_endpoint=True)
    stopped, receipt = checked_actions(engine, [action], limits=SkillLimits(80, 200))
    assert np.allclose(stopped.state['slots']['left']['position'], current['position'])
    assert receipt['status'] == 'interrupted' and receipt['reason'] == 'PARTIAL_MOTION_REPLAN_REQUIRED'
    assert receipt['admittedSkillProgressMm3'] == 0.
    finished, receipt, _ = run(engine, action)
    assert receipt['status'] == 'accepted'
    assert finished.metrics()['targetRemovedMm3'] == 1.
    assert receipt['contactContinuations'][0]['remainingAction']['path'] == [pose(), current]


def test_stop_at_completed_waypoint_does_not_replay_its_orientation():
    engine = source(); preview = engine.preview(motion().action)
    contact = {key: copy.deepcopy(preview['view']['slots']['left'][key])
               for key in ('position', 'quaternion')}
    destination = pose(); destination['quaternion'] = [0., 0., 2**-.5, 2**-.5]
    action = SkillAction({'slot': 'left', 'operation': 'apply', 'duration': .05,
                         'path': [contact, destination]})
    work, receipt, _ = run(engine, action)
    assert receipt['status'] == 'accepted', receipt['reason']
    assert receipt['contactContinuations'][0]['remainingAction']['path'] == [destination]
    assert np.allclose(work.state['slots']['left']['quaternion'], destination['quaternion'])


def test_clearance_dwell_cannot_buy_removal_beyond_whole_tissue_cap():
    engine = source(material='soft_tissue'); before = engine.snapshot()
    work, receipt, _ = run(engine, limits=SkillLimits(80, 200, max_total_non_target_mm3=0.))
    assert receipt['status'] == 'refused' and receipt['reason'] == 'TOTAL_NON_TARGET_CAP'
    assert work.state['removed'] == [] and work.state['cusaV3']['progressExact']
    assert receipt['admittedSkillProgressMm3'] == 0. and engine.snapshot() == before


@pytest.mark.parametrize('constraint,expected', [
    ('whole', 'TOTAL_NON_TARGET_CAP'), ('additional', 'ADDITIONAL_NON_TARGET_CAP'),
    ('label', 'REMOVAL_LABEL_NOT_ALLOWED'), ('forbidden', 'FORBIDDEN_SOURCE_CELL_REMOVAL')])
def test_definite_boundary_removal_refusal_reuses_only_the_unchanged_unsafe_probe(constraint, expected):
    engine = source(material='soft_tissue' if constraint in ('whole', 'additional') else 'tumor')
    settings = {}
    if constraint == 'whole': settings['max_total_non_target_mm3'] = 0.
    if constraint == 'additional':
        settings.update(max_additional_non_target_mm3=0., baseline_non_target_mm3=0.)
    if constraint == 'forbidden': settings['forbidden_removed_cell_ids'] = (int(np.flatnonzero(engine.scene.labels)[0]),)
    if constraint == 'label':
        original = engine; labels = original.scene.labels.copy(); labels[20, 20, 20] = 2
        engine = SurgicalEngine(SurgicalScene(labels, original.scene.affine,
            [*original.scene.regions, {'id': 2, 'name': 'separate bone', 'material': 'bone'}],
            {'kind': 'GENERATED source-label refusal control'}, original.scene.properties))
        for action in original.actions: assert engine.step(copy.deepcopy(action))['status'] == 'accepted'
        settings['allowed_removed_label_ids'] = (2,)
    before = engine.snapshot(); limits = SkillLimits(80, 200, **settings)
    action = motion(duration=2.)
    reference, reference_receipt = checked_actions(engine, [action], limits=limits)
    work, receipt, meter = run(engine, action, limits=limits)
    assert receipt['status'] == reference_receipt['status'] == 'refused'
    assert receipt['reason'] == reference_receipt['reason'] == expected
    assert receipt['attempts'][0]['removalConstraintRefusal'] == expected
    assert meter.calls == meter.preview_calls == 1 and meter.direct_step_calls == 0
    assert work.snapshot() == reference.snapshot() == before and engine.snapshot() == before


def test_replan_and_public_call_bounds_preserve_unadmitted_prefix_cost():
    engine = source(second=True); before = engine.snapshot()
    action = SkillAction({'slot': 'left', 'operation': 'apply', 'duration': .05,
                         'path': [pose(), pose(x=13., z=13.)]})
    work, receipt, _ = run(engine, action, replans=1)
    assert receipt['status'] == 'interrupted' and len(receipt['contactContinuations']) == 1
    assert work.metrics()['targetRemovedMm3'] == 1.
    assert receipt['admittedSkillProgressMm3'] == 0.
    assert receipt['retentionRole'] == 'stopped_native_prefix_not_admitted'
    work, receipt, meter = run(engine, calls=3)
    assert receipt['status'] == 'interrupted' and meter.calls == 3
    assert work.metrics()['targetRemovedMm3'] == 0.
    assert receipt['physicalCallCost']['publicCalls'] == 3 and engine.snapshot() == before


def test_finite_aspiration_is_used_by_actual_bounded_search_and_strong_immediate_baseline():
    engine = source(); before = engine.snapshot()
    route = {'sourceSceneId': engine.scene.identity, 'pathType': 'oblique_mapped_brain',
             'outsidePose': pose(z=3.), 'surfacePose': pose(z=0.),
             'entryPose': pose(z=3.), 'inwardRasUnit': [0., 0., 1.]}
    def search(horizon, replans):
        meter = TransitionMeter(200)
        return checked_sequence_search(engine, route, propose=lambda state: [
            {'id': 'finite-target', 'orientationMode': 'source-ray'}]
            if not state.state['removed'] else [], motion=lambda *_: motion(),
            hand='left', radius=1.5, limits=SkillLimits(80, 200,
                max_additional_non_target_mm3=20., baseline_non_target_mm3=0.), meter=meter,
            horizon=horizon, max_contact_replans=replans)
    old, _, old_evidence = search(1, 0)
    assert old is None and old_evidence['decision'] == 'STOP'
    immediate, _, evidence = search(1, 8)
    lookahead, _, _ = search(2, 8)
    assert immediate is not None and lookahead is not None
    assert evidence['maxContactReplansPerSkill'] == 8
    assert immediate.exited.snapshot() == lookahead.exited.snapshot()
    assert immediate.target_mm3 == 1. and immediate.non_target_mm3 == 0.
    assert immediate.path[0]['work']['contactContinuations']
    assert all(not slot['inserted'] and not slot['active']
               for slot in immediate.exited.state['slots'].values())
    assert engine.snapshot() == before


def test_protected_overlap_is_never_clipped_or_continued():
    original = source(); mask = original.scene.labels == 1
    annotation = ProtectionAnnotation.from_mask(original.scene, 'brainstem', mask,
        provenance='GENERATED protected target overlap', review_status='engineering-reviewed')
    engine = SurgicalEngine(original.scene, protection=ProtectionSet(original.scene, [annotation]),
        guard_config=GuardConfig(require_critical_coverage=True))
    for action in original.actions:
        assert engine.step(action)['status'] == 'accepted'
    before = engine.snapshot()
    work, receipt, _ = run(engine)
    assert receipt['status'] == 'refused' and 'PROTECTED_STRUCTURE' in str(receipt['reason'])
    assert not receipt['contactContinuations'] and receipt['admittedSkillProgressMm3'] == 0.
    assert work.snapshot() == before and engine.snapshot() == before


def test_cancellation_after_partial_capacity_never_publishes_the_prefix_as_complete():
    engine = source(); before = engine.snapshot(); meter = TransitionMeter(200)
    def cancel():
        if any(row['kind'] == 'step' and row.get('status') == 'partial' and
               row.get('reason') == 'TISSUE_ASPIRATION_IN_PROGRESS' for row in meter.trace):
            raise InterruptedError('after native finite-contact prefix')
    work, receipt = checked_cusa_motion(engine, motion(), limits=SkillLimits(80, 200),
        meter=meter, check=cancel, max_contact_replans=8)
    assert receipt['status'] == 'interrupted' and receipt['admittedSkillProgressMm3'] == 0.
    assert receipt['retentionRole'] == 'stopped_native_prefix_not_admitted'
    assert not work.state['removed'] and work.state['cusaV3']['progressExact']
    assert len(work.actions) == len(engine.actions)+1
    assert engine.snapshot() == before


@pytest.mark.parametrize('with_v2', [False, True])
def test_outer_controller_keeps_stock_geometry_and_binds_contact_policy_to_native_history(with_v2):
    from resectionlab.surgical_physics_v2 import default_profile as physics_profile
    from resectionlab.surgical_staged_planning import (
        compare_staged_strategies, install_staged_comparison, validate_staged_report)
    prior = source(); labels = prior.scene.labels.copy(); labels[8:11, 8:11, 8:11] = 1
    properties = {**prior.scene.properties, **({'tissuePhysicsV2': physics_profile()} if with_v2 else {})}
    engine = SurgicalEngine(SurgicalScene(labels, prior.scene.affine, prior.scene.regions,
        {'kind': 'GENERATED dense finite-aspiration controller control'}, properties))
    for action in [*prior.actions, {'slot': 'left', 'operation': 'apply', 'pose': pose(z=3.), 'duration': .05}]:
        assert engine.step(copy.deepcopy(action))['status'] == 'accepted'
    before = engine.snapshot()
    route = {'sourceSceneId': engine.scene.identity, 'pathType': 'oblique_mapped_brain',
        'outsidePose': pose(z=3.), 'surfacePose': pose(z=0.), 'entryPose': pose(z=3.),
        'inwardRasUnit': [0., 0., 1.], 'nativeAxis': 2, 'pathLengthMm': 8., 'corridorRadiusMm': 1.5,
        'accessCompleteness': 'unsupported_skull_access_brain_corridor'}
    plan = {'method': 'superior-access', 'route': route, 'startStateId': engine.fresh().state_id,
        'decisions': [{'action': copy.deepcopy(action), 'status': 'accepted'} for action in engine.actions]}
    plan['planSeal'] = semantic_digest({'sourceSceneId': engine.scene.identity,
        'startStateId': plan['startStateId'], 'decisions': plan['decisions']})
    comparison = {'task': 'anatomy-access', 'alternatives': [plan]}
    comparison['comparisonSeal'] = semantic_digest(comparison)
    record = {'planning': comparison, 'branchId': 'main', 'position': 0,
              'branches': {'main': engine.fresh().snapshot()}}
    old = compare_staged_strategies(engine.fresh(), record, acknowledge_unknowns=True,
        exercise_resume=False, max_physical_calls=2000, max_wall_seconds=30., max_contact_replans=0)
    result = compare_staged_strategies(engine.fresh(), record, acknowledge_unknowns=True,
        exercise_resume=False, max_physical_calls=2000, max_wall_seconds=30., max_contact_replans=8)
    assert old['report']['alternatives'][0]['outcome']['wholeTargetRemovedMm3'] < 27.
    row = result['report']['alternatives'][0]
    assert row['outcome']['wholeTargetRemovedMm3'] == 27. and not row['outcome']['taskGateMet']
    assert result['report']['inventory']['workingRadiiMm'] == [1.5]
    assert any(attempt.get('work', {}).get('contactContinuations') for attempt in row['attemptedWork'])
    saved = copy.deepcopy(record); install_staged_comparison(saved, result, engine.scene)
    assert validate_staged_report(engine.scene, saved, result['report'])['verified']
    altered = copy.deepcopy(result['report']); altered['maxContactReplansPerSkill'] = 0
    for alternative in altered['alternatives']:
        alternative['attemptedWork'] = []; alternative.pop('sequenceSearches', None)
    altered['comparisonSeal'] = semantic_digest({k: v for k, v in altered.items() if k != 'comparisonSeal'})
    with pytest.raises(ValueError, match='selected native contact recipe'):
        validate_staged_report(engine.scene, saved, altered)
    assert engine.snapshot() == before


@pytest.mark.parametrize('replans', [True, -1, 9])
def test_contact_replans_are_explicit_and_bounded(replans):
    with pytest.raises(ValueError, match='bounded dwell'):
        run(source(), replans=replans)
