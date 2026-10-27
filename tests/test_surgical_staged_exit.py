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
    _pose_path_duration, _reverse_accepted_poses, find_checked_exit)
from resectionlab.surgical_staged_skills import SkillLimits
from resectionlab.surgical_transition_meter import TransitionMeter


def _pose(x, y, z, quaternion=(1., 0., 0., 0.)):
    return {'position': [float(x), float(y), float(z)],
            'quaternion': list(quaternion)}


def _inserted_cusa(*, pivot=False):
    labels = np.zeros((16, 16, 16), np.uint16)
    labels[15, 15, 15] = 2
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
