"""Public-action regression for remote protected fragment membership."""
import json
import numpy as np
import pytest

pytest.importorskip('gymnasium')

from resectionlab.surgical_scene import SurgicalScene
from resectionlab.surgical_rl_env import SurgicalRLEnv, RLConfig
from resectionlab.surgical_rl_protection import ProtectionSet, ProtectionAnnotation


def fragment_env(displacement_limit=0.):
    labels = np.zeros((50, 50, 50), np.uint16)
    labels[10:30, 20, 20] = 1
    scene = SurgicalScene(labels, np.eye(4), [dict(id=1, name='Engineering target',
        material='tumor', target=True)], {}, dict(attachments=[]))
    mask = np.zeros(scene.shape, bool)
    mask[29, 20, 20] = True
    protection = ProtectionSet(scene, [ProtectionAnnotation.from_mask(scene, 'brainstem',
        mask, provenance='Synthetic overlap test; no patient anatomy')], clearance_mm=0.)
    return SurgicalRLEnv(scene, protection=protection,
        config=RLConfig(critical_max_displacement_mm=displacement_limit))


def pose(position):
    return dict(position=position, quaternion=[0., 0., 0., 1.])


def grasp(env):
    env.reset(seed=17)
    for action in [dict(operation='exchange', instrument='tumor_forceps'),
                   dict(operation='place', pose=pose([10., 20., -5.]),
                        parameters=dict(radiusMm=.5, jawOpeningMm=0.)),
                   dict(operation='grasp', pose=pose([10., 20., 20.]))]:
        result = env.step_action(dict(slot='left', duration=1., **action))
        assert result[4]['outcome']['status'] == 'accepted'
    assert len(env.engine.state['holds']['left']['cells']) == 20


def test_remote_critical_overlap_cannot_move_with_forceps_and_replays():
    env = fragment_env()
    grasp(env)
    result = env.step_action(dict(slot='left', operation='move', duration=1.,
                                 pose=pose([10., 22., 20.])))
    outcome = result[4]['outcome']
    assert outcome['status'] == 'partial'
    assert outcome['reasonCode'] == 'PROTECTED_STRUCTURE'
    assert env.engine.state['holds']['left']['offset'] == [0, 0, 0]
    assert not env.engine.state['removed']
    assert env.engine.rl_near_exposure == 0.
    assert env.engine.rl_strain_peaks == {}
    assert result[4]['reward_terms']['protected_blocked'] < 0
    restored = fragment_env()
    restored.restore(json.loads(json.dumps(env.snapshot(), allow_nan=False)))
    assert restored.engine.state_id == env.engine.state_id
    assert restored.measurements == env.measurements


def test_explicit_displacement_allowance_tracks_lattice_geometry():
    env = fragment_env(displacement_limit=1.)
    grasp(env)
    result = env.step_action(dict(slot='left', operation='move', duration=1.,
                                 pose=pose([10., 22., 20.])))
    assert result[4]['outcome']['reasonCode'] == 'PROTECTED_STRUCTURE'
    assert env.engine.state['holds']['left']['offset'] == [0, 1, 0]
    np.testing.assert_array_equal(env.engine._positions(), [[29., 21., 20.]])
    assert env.measurements['deformation_injury_mm3'] == pytest.approx(0.)
    # A freely translated detached fragment has no internal strain; the
    # critical displacement cap still constrains it independently of reward.


def test_settings_are_observed_without_promoting_thermal_proxy_to_celsius():
    from resectionlab.surgical_rl_env import PARAMETERS, LOCAL_CHANNELS
    env = fragment_env()
    env.reset(seed=1)
    obs, *_ = env.step_action(dict(slot='left', operation='inspect', duration=.1,
                                  parameters=dict(power=2.5, radiusMm=1.25)))
    assert obs['tool_parameters'][0, PARAMETERS.index('power')] == 2.5
    assert obs['tool_parameters'][0, PARAMETERS.index('radiusMm')] == 1.25
    assert LOCAL_CHANNELS[4] == 'bone_temperature'
    assert env.observation_space.contains(obs)


def test_excess_duration_cannot_force_an_early_time_limit():
    source = fragment_env()
    env = SurgicalRLEnv(source.scene, protection=source.protection,
                        config=RLConfig(max_seconds=.2))
    env.reset(seed=1)
    rejected = env.step_action(dict(slot='left', operation='inspect', duration=.3))
    assert rejected[4]['outcome']['status'] == 'rejected'
    assert rejected[2:4] == (False, False)
    assert env.measurements['elapsed_seconds'] == 0.
    completed = env.step_action(dict(slot='left', operation='inspect', duration=.2))
    assert completed[2:4] == (False, True)
    assert completed[4]['termination_reason'] == 'time_limit'


def test_inactive_other_slot_accumulates_proximity_during_shared_time():
    env = fragment_env()
    env.reset(seed=1)
    for action in [dict(operation='exchange', instrument='irrigation'),
                   dict(operation='place', pose=pose([29., 23., -5.]),
                        parameters=dict(radiusMm=.5)),
                   dict(operation='move', pose=pose([29., 23., 20.]))]:
        result = env.step_action(dict(slot='right', duration=1., **action))
        assert result[4]['outcome']['status'] == 'accepted'
    assert not env.engine.state['slots']['right']['active']
    before = env.measurements['near_contact_exposure_mm3_seconds']
    result = env.step_action(dict(slot='left', operation='inspect', duration=.1))
    assert env.measurements['near_contact_exposure_mm3_seconds'] > before
    assert result[4]['reward_terms']['near_contact'] < 0
