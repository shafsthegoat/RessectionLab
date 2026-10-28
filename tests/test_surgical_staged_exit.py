"""Native checked-exit behavior and source-pose reversal contracts."""
from __future__ import annotations

import copy
import math

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from resectionlab.surgical_engine import SurgicalEngine
from resectionlab.surgical_scene import SurgicalScene
from resectionlab.surgical_staged_exit import (
    MAX_INTERPOLATED_ROTATION_DEG, MAX_INTERPOLATED_STEP_MM,
    _pose_path_duration, _reverse_accepted_poses, _reverse_native_steps, find_checked_exit)
from resectionlab.surgical_staged_skills import SkillLimits
from resectionlab.surgical_transition_meter import TransitionMeter


def _pose(x, y, z, quaternion=(1., 0., 0., 0.)):
    return {'position': [float(x), float(y), float(z)],
            'quaternion': list(quaternion)}


def _inserted_cusa(*, pivot=False, obstacle=False):
    labels = np.zeros((16, 16, 16), np.uint16)
    labels[15, 15, 15] = 2
    if obstacle:
        labels[8, 4, 10] = 2
    scene = SurgicalScene(labels, np.eye(4),
        [{'id': 2, 'name': 'generated target', 'material': 'tumor', 'target': True}],
        {'kind': 'generated staged exit native test'}, {})
    engine = SurgicalEngine(scene)
    outside = _pose(4, 4, 8)
    for action in (
            {'slot': 'left', 'operation': 'exchange', 'instrument': 'cusa', 'duration': .2},
            {'slot': 'left', 'operation': 'place', 'pose': outside, 'duration': .2},
            {'slot': 'left', 'operation': 'move', 'pose': _pose(4, 4, 5), 'duration': 1.},
            {'slot': 'left', 'operation': 'move', 'pose': _pose(4.3, 4, 4), 'duration': 1.}):
        assert engine.step(action)['status'] == 'accepted'
    if pivot:
        q = (Rotation.from_euler('y', 10., degrees=True)*
             Rotation.from_quat(outside['quaternion'])).as_quat()
        assert engine.step({'slot': 'left', 'operation': 'move',
            'pose': _pose(4.3, 4, 4, q), 'duration': 1.})['status'] == 'accepted'
    route = {'sourceSceneId': scene.identity, 'pathType': 'oblique_mapped_brain',
             'outsidePose': outside}
    return engine, route


def _limits(calls=300):
    return SkillLimits(max_native_actions=240, max_public_calls=calls,
        max_total_non_target_mm3=200., max_additional_non_target_mm3=20.,
        baseline_non_target_mm3=0.)


def test_reverse_path_duration_counts_translation_and_full_shaft_rotation():
    start = _pose(0, 0, 0)
    translated = _pose(12, 0, 0)
    pivot = (Rotation.from_euler('y', 10., degrees=True)*
             Rotation.from_quat(start['quaternion'])).as_quat()
    rotated = _pose(0, 0, 0, pivot)
    mixed = _pose(12, 0, 0, rotated['quaternion'])
    shaft_rotation_travel = 42.*math.radians(10.)
    assert _pose_path_duration(start, [translated]) == pytest.approx(6.)
    assert _pose_path_duration(start, [rotated]) == pytest.approx(
        shaft_rotation_travel/2.)
    assert _pose_path_duration(start, [mixed]) == pytest.approx(
        (12.+shaft_rotation_travel)/2.)
    assert _pose_path_duration(start, [mixed]) > max(
        _pose_path_duration(start, [translated]),
        _pose_path_duration(start, [rotated]))
    assert _pose_path_duration(start, [start]) == .2
    assert _pose_path_duration(start, [_pose(100, 0, 0)]) == 30.


def test_native_step_reverse_reconstructs_checked_translation_rotation_and_reads_only():
    source, route = _inserted_cusa(pivot=True)
    start = source.state['slots']['left']
    q = (Rotation.from_euler('z', 3., degrees=True)*
         Rotation.from_quat(start['quaternion'])).as_quat().tolist()
    path = [_pose(4.5,4.,3.7,start['quaternion']), _pose(4.7,4.1,3.3,q)]
    assert source.step({'slot':'left','operation':'apply','path':path,'duration':1.3})['status']=='accepted'
    assert source.step({'slot':'left','operation':'apply','pose':path[-1],'duration':.3})['status']=='accepted'
    before = source.snapshot()
    meter = TransitionMeter(300)
    poses, reason = _reverse_native_steps(meter.wrap(source), 'left', lambda: None)
    assert reason is None and poses and meter.calls == 0
    assert source.snapshot() == before
    assert poses[-1]['position'] == route['outsidePose']['position']
    # Native check of the reversed integration poses, without restoring or
    # splicing any intermediate state. Same native collision implementation.
    branch = meter.wrap(source.clone())
    assert branch.step({'slot':'left','operation':'deactivate','duration':.2})['status']=='accepted'
    for index in range(0,len(poses),24):
        chunk=poses[index:index+24]
        assert branch.step({'slot':'left','operation':'move','path':chunk,
                            'duration':.1*len(chunk)})['status']=='accepted'
    assert np.allclose(branch.state['slots']['left']['position'],route['outsidePose']['position'])
    assert source.snapshot() == before


def test_native_step_reverse_refuses_nonmoving_history_geometry_changes():
    source, _ = _inserted_cusa()
    for radius in (2.2,1.5):
        assert source.step({'slot':'left','operation':'inspect','parameters':{'radiusMm':radius},
                            'duration':.2})['status']=='accepted'
    poses, reason = _reverse_native_steps(source,'left',lambda:None)
    assert not poses and reason=='SOURCE_WORKING_GEOMETRY_CHANGED'


def test_direct_exit_uses_native_preview_commit_and_preserves_source_prefix():
    source, route = _inserted_cusa()
    before = source.snapshot()
    meter = TransitionMeter(300)
    candidate, receipt = find_checked_exit(source, route, hand='left',
        limits=_limits(), meter=meter, check=lambda: None)
    assert receipt['status'] == 'accepted' and receipt['mode'] == 'direct'
    assert receipt['sourceStateId'] == before['stateId']
    assert receipt['finalStateId'] == candidate.state_id
    assert receipt['appliedExitActions'] == candidate.actions[len(source.actions):]
    assert [row['operation'] for row in receipt['appliedExitActions']] == [
        'deactivate', 'withdraw']
    assert receipt['additionalBrainMm3SincePrefix'] == 0.
    assert receipt['additionalNonTargetMm3SincePrefix'] == 0.
    assert not candidate.state['slots']['left']['inserted']
    assert not candidate.state['slots']['left']['active']
    assert candidate.state['slots']['left']['position'] == route['outsidePose']['position']
    assert source.snapshot() == before
    assert receipt['publicCallCost'] == meter.calls
    assert meter.preview_calls == meter.direct_step_calls == 2


def test_reverse_source_history_keeps_lateral_and_pivot_poses_with_small_steps():
    engine, _ = _inserted_cusa(pivot=True)
    poses, reason = _reverse_accepted_poses(engine, 'left')
    assert reason is None and poses
    assert any(abs(p['position'][0]-4.) < 1e-8 for p in poses)
    previous = {'position': engine.state['slots']['left']['position'],
                'quaternion': engine.state['slots']['left']['quaternion']}
    for pose in poses:
        assert np.linalg.norm(np.subtract(pose['position'], previous['position'])) <= \
            MAX_INTERPOLATED_STEP_MM+1e-8
        angle = (Rotation.from_quat(pose['quaternion'])*
                 Rotation.from_quat(previous['quaternion']).inv()).magnitude()
        assert np.degrees(angle) <= MAX_INTERPOLATED_ROTATION_DEG+1e-8
        previous = pose
    assert poses[-1]['position'] == [4., 4., 8.]


def test_pivoted_cusa_can_use_checked_direct_exit_with_actual_native_rotation():
    source, route = _inserted_cusa(pivot=True)
    before = source.snapshot()
    candidate, receipt = find_checked_exit(source, route, hand='left',
        limits=_limits(), meter=TransitionMeter(300), check=lambda: None)
    assert receipt['status'] == 'accepted'
    assert candidate.state['slots']['left']['position'] == route['outsidePose']['position']
    assert source.snapshot() == before


def test_call_budget_refusal_returns_original_prefix_and_retains_failed_probe():
    source, route = _inserted_cusa()
    before = source.snapshot()
    meter = TransitionMeter(2)
    candidate, receipt = find_checked_exit(source, route, hand='left',
        limits=_limits(2), meter=meter, check=lambda: None)
    assert receipt['status'] == 'refused'
    assert receipt['finalStateId'] == before['stateId']
    assert receipt['appliedExitActions'] == []
    assert receipt['stages'][0]['checked']['status'] == 'interrupted'
    assert receipt['stages'][0]['probeActions'][0]['operation'] == 'deactivate'
    assert candidate.state_id == source.state_id and candidate.actions == source.actions
    assert source.snapshot() == before
    assert meter.calls == receipt['publicCallCost'] == 2


def test_route_binding_and_additional_cap_refuse_before_native_calls():
    source, route = _inserted_cusa()
    meter = TransitionMeter(300)
    wrong = copy.deepcopy(route); wrong['sourceSceneId'] = 'sha256:other'
    with pytest.raises(ValueError, match='bound'):
        find_checked_exit(source, wrong, hand='left', limits=_limits(),
            meter=meter, check=lambda: None)
    loose = SkillLimits(240, 300, max_additional_non_target_mm3=21.,
                        baseline_non_target_mm3=0.)
    with pytest.raises(ValueError, match='20/200'):
        find_checked_exit(source, route, hand='left', limits=loose,
            meter=meter, check=lambda: None)
    assert meter.calls == 0


def test_latest_partial_native_prefix_is_bound_to_fraction_and_observed_pose(monkeypatch):
    source, route = _inserted_cusa(obstacle=True)
    start = {key: copy.deepcopy(source.state['slots']['left'][key])
             for key in ('position', 'quaternion')}
    # Include a stationary segment: it consumes native integration time and
    # must participate in completedFraction's denominator despite zero travel.
    result = source.step({'slot': 'left', 'operation': 'move',
        'path': [start, _pose(4.5, 4, 4), _pose(12, 4, 4)], 'duration': 1.3})
    assert result['status'] == 'partial' and result['reason'] == 'SHAFT_COLLISION'
    assert 0 < result['completedFraction'] < 1
    source.history[-1]['contactPosition'] = [999., 999., 999.]
    original = source.snapshot()
    # Reconstruction is read-only and never treats a rejected contact point
    # as an accepted pose or silently restores an intermediate anatomy state.
    with monkeypatch.context() as read_only:
        read_only.setattr(SurgicalEngine, 'step', lambda *_a, **_k: pytest.fail('No reconstruction transition'))
        read_only.setattr(SurgicalEngine, 'preview', lambda *_a, **_k: pytest.fail('No reconstruction preview'))
        poses, reason = _reverse_native_steps(source, 'left', lambda: None)
    assert reason is None and poses and source.snapshot() == original
    assert poses[-1]['position'] == route['outsidePose']['position']
    branch = source.clone()
    assert branch.step({'slot': 'left', 'operation': 'deactivate', 'duration': .2})['status'] == 'accepted'
    for index in range(0, len(poses), 24):
        chunk = poses[index:index+24]
        assert branch.step({'slot': 'left', 'operation': 'move', 'path': chunk,
                            'duration': .1*len(chunk)})['status'] == 'accepted'
    assert np.allclose(branch.state['slots']['left']['position'], route['outsidePose']['position'])
    assert source.snapshot() == original
    fraction = source.history[-1]['completedFraction']
    source.history[-1]['completedFraction'] = fraction+.00001
    poses, reason = _reverse_native_steps(source, 'left', lambda: None)
    assert not poses and reason == 'PARTIAL_NATIVE_STEP_FRACTION_INVALID'
    source.history[-1]['completedFraction'] = 0.
    poses, reason = _reverse_native_steps(source, 'left', lambda: None)
    assert not poses and reason == 'PARTIAL_OBSERVED_ENDPOINT_DIFFERS'


def test_partial_history_with_later_motion_is_still_refused():
    source, _ = _inserted_cusa(obstacle=True)
    assert source.step({'slot': 'left', 'operation': 'move',
        'pose': _pose(12, 4, 4), 'duration': 1.})['status'] == 'partial'
    assert source.step({'slot': 'left', 'operation': 'move',
        'pose': _pose(4.3, 4, 4), 'duration': 1.})['status'] == 'accepted'
    poses, reason = _reverse_native_steps(source, 'left', lambda: None)
    assert not poses and reason == 'PARTIAL_POSE_HISTORY_UNSUPPORTED'


def test_metered_recovery_report_binds_native_ancestry_and_zero_additional_cap():
    from resectionlab.core import semantic_digest
    from resectionlab.surgical_staged_exit import recover_checked_cusa, validate_exit_recovery
    source, _ = _inserted_cusa()
    original = source.snapshot()
    final, report = recover_checked_cusa(source)
    assert source.snapshot() == original
    assert report['status'] == 'accepted' and report['cleanupPending'] == []
    assert report['sourceMetrics'] == report['finalMetrics']
    assert report['cost']['physicalCalls'] <= 240 and report['wallSeconds'] <= 45.
    assert report['exactNativeReplay']
    report.update(sourceBranchId='source', finalBranchId='recovered')
    report['comparisonSeal'] = semantic_digest(report)
    record = {'branches': {'source': original, 'recovered': final.snapshot()},
        'parents': {'recovered': {'branchId': 'source', 'position': len(source.actions)}},
        'timeline': {'recovered': copy.deepcopy(final.history)}}
    assert validate_exit_recovery(source.scene, record, report)['verified']
    # Full-case persistence omits its redundant top-level timeline. The bounded
    # recovery receipt retains the exact verified history needed for validation.
    del record['timeline']
    assert validate_exit_recovery(source.scene, record, report)['verified']
    for changed in ('cost', 'source', 'geometry'):
        forged = copy.deepcopy(report)
        if changed == 'cost':
            forged['cost']['physicalCalls'] = forged['cost']['previewCalls'] = forged['cost']['directStepCalls'] = 0
        elif changed == 'source':
            forged['sourceMetrics']['nonTargetRemovedMm3'] = 20.
        else:
            forged['sourceSnapshot']['checkpoint']['state']['slots']['left']['position'][0] += 1.
            forged['sourceProgramSeal'] = semantic_digest(forged['sourceSnapshot'])
        forged['comparisonSeal'] = semantic_digest({k: v for k, v in forged.items() if k != 'comparisonSeal'})
        with pytest.raises(ValueError):
            validate_exit_recovery(source.scene, record, forged)


def test_refused_recovery_at_a_history_cursor_keeps_the_original_prefix(monkeypatch):
    import resectionlab.surgical_staged_exit as exits
    from resectionlab.core import semantic_digest
    source, _ = _inserted_cusa()
    parent = source.clone()
    assert parent.step({'slot': 'left', 'operation': 'inspect', 'duration': .2})['status'] == 'accepted'
    monkeypatch.setattr(exits, 'find_checked_exit', lambda engine, *_a, **_kw: (
        engine.clone(), {'status': 'refused', 'reason': 'GENERATED_REFUSAL', 'mode': None}))
    final, report = exits.recover_checked_cusa(source)
    assert final.snapshot() == source.snapshot() and report['status'] == 'refused'
    report.update(sourceBranchId='source', finalBranchId='source')
    report['comparisonSeal'] = semantic_digest(report)
    record = {'branches': {'source': parent.snapshot()}, 'timeline': {'source': parent.history}}
    assert exits.validate_exit_recovery(source.scene, record, report)['verified']


def test_recovery_evidence_cannot_claim_inserted_tip_resize_as_checked_withdrawal():
    from resectionlab.core import semantic_digest
    from resectionlab.surgical_staged_exit import recover_checked_cusa, validate_exit_recovery
    source, _ = _inserted_cusa()
    final, report = recover_checked_cusa(source)
    snapshot = final.snapshot()
    snapshot['actions'][-1]['parameters'] = {'radiusMm': .1}
    report['finalNativeHistory'][-1]['action']['parameters'] = {'radiusMm': .1}
    report.update(sourceBranchId='source', finalBranchId='recovered')
    report['comparisonSeal'] = semantic_digest(report)
    record = {'branches': {'source': source.snapshot(), 'recovered': snapshot},
        'parents': {'recovered': {'branchId': 'source', 'position': len(source.actions)}}}
    with pytest.raises(ValueError, match='accepted branch'):
        validate_exit_recovery(source.scene, record, report)


def test_field_fluid_does_not_become_a_successful_cleanup_or_replace_source():
    from resectionlab.core import semantic_digest
    from resectionlab.surgical_staged_exit import recover_checked_cusa, validate_exit_recovery
    source, _ = _inserted_cusa()
    outside = _pose(8, 8, 12, (0., 0., 0., 1.))
    for action in (
        {'slot': 'right', 'operation': 'exchange', 'instrument': 'irrigation', 'duration': .2},
        {'slot': 'right', 'operation': 'place', 'pose': outside, 'duration': .2},
        {'slot': 'right', 'operation': 'apply', 'parameters': {'flowMm3PerSecond': 1.}, 'duration': 1.},
        {'slot': 'right', 'operation': 'deactivate', 'duration': .2},
        {'slot': 'right', 'operation': 'withdraw', 'pose': outside, 'duration': .2}):
        assert source.step(action)['status'] == 'accepted'
    assert sum(source.state['fluid'].values()) > 0
    original = source.snapshot()
    final, report = recover_checked_cusa(source)
    assert report['status'] == 'refused' and 'fluid' in report['cleanupPending']
    assert report['reason'].startswith('FIELD_CLEANUP_INCOMPLETE')
    assert final.snapshot() == source.snapshot() == original
    assert report['stoppedExecutions']
    report.update(sourceBranchId='source', finalBranchId='source')
    report['comparisonSeal'] = semantic_digest(report)
    record = {'branches': {'source': original}, 'timeline': {'source': source.history}}
    assert validate_exit_recovery(source.scene, record, report)['verified']


def test_unaffordable_final_replay_retains_candidate_and_replays_original_source(monkeypatch):
    import resectionlab.surgical_staged_exit as exits
    from resectionlab.core import semantic_digest
    source, _ = _inserted_cusa()
    original = source.snapshot()

    def long_checked_exit(engine, route, *, hand, meter, check, **_kw):
        # Exercise accounting with actual public native transitions. The
        # generated field is empty; this isolates final replay admission from
        # any patient-specific geometry or speculative outcome prediction.
        candidate = engine.clone()
        action = {'slot': hand, 'operation': 'inspect', 'duration': .01}
        with meter.phase('generated.long_checked_exit'):
            for _ in range(110):
                assert candidate.preview(action, check=check, include_view=False,
                    include_history=False)['status'] == 'accepted'
            for _ in range(65):
                assert candidate.step(action, check=check)['status'] == 'accepted'
            for action in (
                    {'slot': hand, 'operation': 'deactivate', 'duration': .2},
                    {'slot': hand, 'operation': 'withdraw', 'pose': route['outsidePose'], 'duration': 1.}):
                assert candidate.step(action, check=check)['status'] == 'accepted'
        assert not candidate.state['slots'][hand]['inserted']
        assert meter.remaining < len(candidate.actions)
        return candidate, {'status': 'accepted', 'reason': None, 'mode': 'generated_long_exit'}

    monkeypatch.setattr(exits, 'find_checked_exit', long_checked_exit)
    final, report = exits.recover_checked_cusa(source)
    assert report['status'] == 'refused' and report['mode'] is None
    assert report['reason'].startswith('FINAL_PROGRAM_REPLAY_HEADROOM')
    assert final.snapshot() == source.snapshot() == original
    assert report['receipt']['status'] == 'accepted'
    assert any(len(row['nativeSnapshot']['actions']) > len(source.actions)+50
               for row in report['stoppedExecutions'])
    replay = report['cost']['byPhase']['checked_recovery.independent_full_program_replay']
    assert replay['directStepCalls'] == len(source.actions)
    assert report['cost']['physicalCalls'] <= 240 and report['exactNativeReplay']
    report.update(sourceBranchId='source', finalBranchId='source')
    report['comparisonSeal'] = semantic_digest(report)
    record = {'branches': {'source': original}, 'timeline': {'source': source.history}}
    assert exits.validate_exit_recovery(source.scene, record, report)['verified']
