import csv
import json

import numpy as np
import pytest

from resectionlab.surgical_engine import SurgicalEngine
from resectionlab.surgical_scene import SurgicalScene
from resectionlab.surgical_telemetry import capture_episode, export_telemetry


def scene(with_target=False, properties=None):
    labels = np.zeros((24, 24, 24), np.uint16)
    if with_target:
        labels[8, 8, 8] = 1
        labels[8, 8, 9] = 2
    return SurgicalScene(labels, np.eye(4), [
        {'id': 1, 'name': 'target', 'material': 'tumor', 'target': True},
        {'id': 2, 'name': 'non-target', 'material': 'soft_tissue'}], {}, properties or {})


def pose(position):
    return {'position': position, 'quaternion': [0., 0., 0., 1.]}


def test_microstep_kinematics_are_actual_and_reporting_does_not_interpolate():
    s = scene()
    base = SurgicalEngine(s)
    left = base.state['slots']['left']['position']
    end = [left[0] + 1., *left[1:]]
    action = {'slot': 'left', 'operation': 'move', 'pose': pose(end), 'duration': .31}
    base.step(action)
    report = capture_episode(base, [action])
    assert report['expectedStateVerified']
    assert report['finalStateId'] == base.state_id
    assert report['summary']['pathLengthMm'] == {'left': 1., 'right': 0.}
    row = next(sample for sample in report['samples'] if sample['timeSeconds'] == .2)
    assert row['observedTimeSeconds'] == pytest.approx(.155)
    assert row['observationAgeSeconds'] == pytest.approx(.045)
    assert row['slots']['left']['positionMm'][0] == pytest.approx(left[0] + .5)
    assert row['slots']['left']['speedMmPerSecond'] == pytest.approx(1 / .31)
    assert row['slots']['right']['velocityMmPerSecond'] == [0., 0., 0.]
    assert row['slots']['left']['accelerationMmPerSecond2'] == pytest.approx([0., 0., 0.])
    assert row['slots']['left']['measuredForceN'] is None
    assert row['slots']['left']['forceAssumedN'] is None
    assert row['woundCount'] is None
    assert report['samples'][-1]['timeSeconds'] == pytest.approx(.31)
    assert report['observations'][0]['slots']['left']['velocityMmPerSecond'] is None


def test_placement_jumps_excluded_and_exchange_activation_rejection_recorded():
    s = scene()
    actions = [
        {'slot': 'left', 'operation': 'exchange', 'instrument': 'cusa', 'duration': .1},
        {'slot': 'left', 'operation': 'place', 'pose': pose([8., 8., 4.]), 'duration': .1},
        {'slot': 'left', 'operation': 'apply', 'duration': .2},
        {'slot': 'left', 'operation': 'exchange', 'instrument': 'irrigation', 'duration': .1},
    ]
    report = capture_episode(s, actions)
    assert report['summary']['invalidActionsCumulative'] == 1
    assert report['summary']['toolExchangesCumulative'] == 1
    assert report['summary']['pathLengthMm']['left'] == 0.
    placement = [sample for sample in report['observations'] if sample['sampleKind'] == 'microstep' and sample['actionIndex'] == 1][0]
    assert placement['slots']['left']['poseDiscontinuity']
    assert placement['slots']['left']['velocityMmPerSecond'] is None
    assert placement['slots']['left']['repositionDistanceMm'] > 0
    applied = [sample for sample in report['observations'] if sample['sampleKind'] == 'microstep' and sample['actionIndex'] == 2]
    assert all(sample['slots']['left']['active'] and not sample['slots']['left']['latchedActive'] for sample in applied)
    assert all(sample['actionIndex'] != 3 for sample in report['observations'] if sample['sampleKind'] == 'microstep')
    assert report['attempts'][-1]['startTimeSeconds'] == report['attempts'][-1]['endTimeSeconds']


def test_target_non_target_and_timed_removal_events_match_engine():
    s = scene(with_target=True)
    actions = [
        {'slot': 'left', 'operation': 'exchange', 'instrument': 'cusa', 'duration': .1},
        {'slot': 'left', 'operation': 'place', 'pose': pose([8., 8., 4.]), 'parameters': {'radiusMm': .6}, 'duration': .1},
        {'slot': 'left', 'operation': 'apply', 'pose': pose([8., 8., 9.]), 'duration': 1.},
    ]
    base = SurgicalEngine(s)
    for action in actions:
        base.step(action)
    report = capture_episode(base, actions)
    assert report['summary']['targetRemovedMm3'] == base.metrics()['targetRemovedMm3'] == 1.
    assert report['summary']['nonTargetRemovedMm3'] == base.metrics()['nonTargetRemovedMm3'] == 1.
    removal = [event for event in report['events'] if event['event']['kind'] == 'removal']
    assert removal and all(event['timing'] == 'committed_microstep' for event in removal)
    assert sum(event['event']['volumeMm3'] for event in removal) == 2.
    assert all(event['timeSeconds'] <= report['summary']['timeSeconds'] for event in removal)


def test_partial_contacts_keep_only_successful_samples():
    s = scene(with_target=True)
    actions = [
        {'slot': 'left', 'operation': 'place', 'pose': pose([8., 8., 4.]), 'parameters': {'radiusMm': .6}, 'duration': .1},
        {'slot': 'left', 'operation': 'move', 'pose': pose([8., 8., 9.]), 'duration': 1.},
    ]
    base = SurgicalEngine(s)
    for action in actions:
        result = base.step(action)
    assert result['status'] == 'partial'
    report = capture_episode(base, actions)
    assert report['summary']['contactStopsCumulative'] == 1
    assert report['summary']['timeSeconds'] < 1.1
    assert report['samples'][-1]['slots']['left']['positionMm'] == base.state['slots']['left']['position']
    assert report['summary']['targetRemovedMm3'] == 0


def test_zero_duration_is_rejected_without_a_motion_sample():
    s = scene()
    action = {'slot': 'left', 'operation': 'inspect', 'duration': 0}
    report = capture_episode(s, [action], expected_state_id=SurgicalEngine(s).state_id)
    assert report['summary']['timeSeconds'] == 0.
    assert report['summary']['microstepCount'] == 0
    assert report['summary']['invalidActionsCumulative'] == 1
    assert report['attempts'][0]['status'] == 'rejected'
    assert report['samples'][0]['slots']['left']['velocityMmPerSecond'] is None


def test_snapshot_replay_and_missing_attempt_history_are_explicit(tmp_path):
    base = SurgicalEngine(scene())
    base.step({'slot': 'left', 'operation': 'inspect', 'duration': .1})
    report = capture_episode(base.scene, snapshot=base.snapshot(), label='manual', episode_id='trial </script>')
    assert report['attemptHistory'] == 'accepted_history_only_rejections_unknown'
    assert report['summary']['invalidActionsCumulative'] is None
    assert report['finalStateId'] == base.state_id
    paths = export_telemetry([report], tmp_path)
    assert json.loads((tmp_path / 'telemetry.json').read_text())['episodes'][0]['finalStateId'] == base.state_id
    with open(paths['csv']) as stream:
        rows = list(csv.DictReader(stream))
    assert rows and rows[0]['left_velocityMmPerSecond_x'] == ''
    page = (tmp_path / 'telemetry.html').read_text()
    assert 'trial </script>' not in page
    assert '10.1038/s41746-022-00596-8' in page
    with pytest.raises(ValueError, match='state identity differs'):
        capture_episode(base.scene, [], expected_state_id=base.state_id)


def test_assumed_v2_force_is_not_measured_force():
    from resectionlab.surgical_physics_v2 import default_profile
    report = capture_episode(scene(properties={'tissuePhysicsV2': default_profile()}), [])
    slot = report['samples'][0]['slots']['left']
    assert slot['forceAssumedN'] == 1.
    assert slot['measuredForceN'] is None
    assert report['samples'][0]['woundCount'] == 0
    assert report['physicsProvenance']['provenance']['kind'] == 'assumed-research'


def test_bound_protection_survives_engine_and_snapshot_telemetry_replay():
    from resectionlab.surgical_rl_protection import GuardConfig, ProtectionAnnotation, ProtectionSet
    s = scene(with_target=True)
    mask = s.labels == 1
    protection = ProtectionSet(s, [ProtectionAnnotation.from_mask(s, 'brainstem', mask,
                                provenance='Synthetic test overlap, not patient anatomy')], clearance_mm=0.)
    base = SurgicalEngine(s, protection=protection, guard_config=GuardConfig())
    actions = [
        {'slot': 'left', 'operation': 'exchange', 'instrument': 'cusa', 'duration': .1},
        {'slot': 'left', 'operation': 'place', 'pose': pose([8., 8., 4.]), 'parameters': {'radiusMm': .6}, 'duration': .1},
        {'slot': 'left', 'operation': 'apply', 'pose': pose([8., 8., 9.]), 'duration': 1.},
    ]
    for action in actions:
        base.step(action)
    for report in (capture_episode(base, actions), capture_episode(s, actions, snapshot=base.snapshot())):
        assert report['finalStateId'] == base.state_id
        assert report['protection']['contractId'] == base.guard_report()['contractId']
        assert report['summary']['targetRemovedMm3'] == 0.
        assert report['attempts'][-1]['reasonCode'] == 'PROTECTED_STRUCTURE'


def test_both_active_slots_and_fluid_match_engine_metrics():
    s = scene()
    actions = [
        {'slot': 'left', 'operation': 'exchange', 'instrument': 'irrigation', 'duration': .1},
        {'slot': 'right', 'operation': 'exchange', 'instrument': 'irrigation', 'duration': .1},
        {'slot': 'left', 'operation': 'place', 'pose': pose([3., 3., 3.]), 'duration': .1},
        {'slot': 'right', 'operation': 'place', 'pose': pose([19., 19., 3.]), 'duration': .1},
        {'slot': 'left', 'operation': 'activate', 'duration': .1},
        {'slot': 'right', 'operation': 'activate', 'duration': .2},
    ]
    base = SurgicalEngine(s)
    for action in actions:
        assert base.step(action)['status'] == 'accepted'
    report = capture_episode(base, actions)
    row = report['samples'][-1]
    assert row['slots']['left']['active'] and row['slots']['right']['active']
    assert row['fluidAddedMm3'] > 0.
    for key in ('fluidAddedMm3', 'fluidPresentMm3', 'fluidRemovedMm3', 'fluidEscapedMm3'):
        assert row[key] == pytest.approx(base.metrics()[key])
    assert row['fluidAddedMm3'] == pytest.approx(row['fluidPresentMm3'] + row['fluidRemovedMm3'] + row['fluidEscapedMm3'])


@pytest.mark.parametrize('interval', [0, -1, float('nan'), float('inf')])
def test_invalid_intervals_fail(interval):
    with pytest.raises(ValueError, match='interval_seconds'):
        capture_episode(scene(), [], interval_seconds=interval)
