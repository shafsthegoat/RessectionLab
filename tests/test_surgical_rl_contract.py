"""RL contract checks use declared engineering fixtures, never patient evidence."""
import json
import numpy as np
import pytest
pytest.importorskip('gymnasium')
from resectionlab.surgical_scene import SurgicalScene
from resectionlab.surgical_physics_v2 import default_profile
from resectionlab.surgical_rl_env import SurgicalRLEnv, RLConfig, INSTRUMENTS


def fixture():
    labels = np.zeros((24, 24, 20), np.uint16)
    labels[8:11, 8:11, 8:11] = 1
    return SurgicalScene(labels, np.eye(4), [{'id': 1, 'name': 'Synthetic target',
        'material': 'tumor', 'target': True}], {'provenance': 'generated test geometry'},
        {'tissuePhysicsV2': default_profile()})


def env(**kwargs):
    return SurgicalRLEnv(fixture(), config=RLConfig(require_critical_coverage=False, **kwargs))


def command(operation='inspect', **kwargs):
    return {'slot': 'left', 'operation': operation, 'duration': .1, **kwargs}


def test_reset_spaces_seed_and_all_fifteen_tools():
    e = env(); first, info = e.reset(seed=17)
    assert len(INSTRUMENTS) == 15
    assert e.observation_space.contains(first)
    second, _ = e.reset(seed=17)
    for key in first: np.testing.assert_array_equal(first[key], second[key])
    e.action_space.seed(17); a = e.action_space.sample()
    e.action_space.seed(17); b = e.action_space.sample()
    for key in a: np.testing.assert_array_equal(a[key], b[key])
    result = e.step(e.encode_action(command()))
    assert len(result) == 5 and e.observation_space.contains(result[0])
    assert np.isfinite(result[1])


def test_attempt_budget_truncates_and_requires_reset():
    e = env(max_steps=2); e.reset(seed=1)
    assert e.step_action(command())[2:4] == (False, False)
    _, reward, terminated, truncated, _ = e.step_action(command())
    assert not terminated and truncated and reward < 0
    with pytest.raises((RuntimeError, ValueError)): e.step_action(command())


def test_json_restore_branch_and_source_preservation():
    e = env(); e.reset(seed=42); source = e.engine.scene.labels.copy()
    e.step_action(command())
    record = json.loads(json.dumps(e.snapshot(), allow_nan=False))
    restored = env(); restored.restore(record)
    assert restored.engine.state_id == e.engine.state_id
    a = e.step_action(command()); b = restored.step_action(command())
    assert a[1:4] == b[1:4]
    assert restored.engine.state_id == e.engine.state_id
    branch = e.branch(1)
    before = e.engine.state_id
    branch.step_action(command(duration=.2))
    assert e.engine.state_id == before
    np.testing.assert_array_equal(e.engine.scene.labels, source)


def test_zero_quaternion_and_occupied_placement_cannot_remove_tissue():
    e = env(); e.reset(seed=1)
    for quaternion in ([0., 0., 0., 0.], [0., 0., 0., 1.]):
        _, reward, _, _, _ = e.step_action(command('place', pose={
            'position': [9., 9., 9.], 'quaternion': quaternion}))
        assert reward < 0 and not e.engine.state['removed']


def test_codec_preserves_double_precision_and_quaternion_roll():
    e = env(); e.reset(seed=1)
    a = command('place', pose={'position': [8.123456789, 8., -5.],
        'quaternion': [0., 0., np.sqrt(.5), np.sqrt(.5)]}, parameters={'radiusMm': 1.5})
    encoded = e.encode_action(a)
    assert e.action_space.contains(encoded)
    assert encoded['pose'].dtype == np.float64
    np.testing.assert_allclose(encoded['pose'], a['pose']['position'] + a['pose']['quaternion'], rtol=0, atol=0)


def test_missing_brainstem_is_unknown_and_explicit_mask_blocks_overlap():
    from resectionlab.surgical_rl_protection import ProtectionAnnotation, ProtectionSet
    source = fixture()
    with pytest.raises(ValueError): SurgicalRLEnv(source)
    mask = np.zeros(source.shape, bool); mask[9, 9, 8] = True
    annotation = ProtectionAnnotation.from_mask(source, 'brainstem', mask,
        provenance='Synthetic exclusion-mask contract test; not patient anatomy')
    protection = ProtectionSet(source, [annotation], clearance_mm=1., strict=True)
    e = SurgicalRLEnv(source, protection=protection)
    e.reset(seed=1)
    e.step_action(command('exchange', instrument='cusa'))
    e.step_action(command('place', pose={'position': [9., 9., -5.], 'quaternion': [0., 0., 0., 1.]},
        parameters={'radiusMm': .5}))
    protected_cell = int(np.flatnonzero(mask)[0])
    _, reward, _, _, _ = e.step_action(command('apply', pose={
        'position': [9., 9., 9.], 'quaternion': [0., 0., 0., 1.]}))
    assert protected_cell not in e.engine.state['removed']
    assert reward < 0


def test_gymnasium_checker():
    from gymnasium.utils.env_checker import check_env
    check_env(env(), skip_render_check=True)


def test_rejected_attempt_survives_snapshot_and_consumes_budget():
    e = env(max_steps=2); e.reset(seed=9)
    before = e.engine.state_id
    bad = command('place', pose={'position': [9., 9., 9.], 'quaternion': [0., 0., 0., 0.]})
    first = e.step_action(bad)
    assert e.engine.state_id == before and first[1] < 0
    restored = env(max_steps=2)
    restored.restore(json.loads(json.dumps(e.snapshot(), allow_nan=False)))
    a = e.step_action(command()); b = restored.step_action(command())
    assert a[1:4] == b[1:4] and a[3]


def test_target_removal_terminates_success_and_cannot_repeat_reward():
    e = env(); e.reset(seed=3)
    e.step_action(command('exchange', instrument='cusa'))
    e.step_action(command('place', pose={'position': [9., 9., -5.], 'quaternion': [0., 0., 0., 1.]},
        parameters={'radiusMm': 3.}))
    _, _, terminated, truncated, info = e.step_action(command('apply', pose={
        'position': [9., 9., 11.], 'quaternion': [0., 0., 0., 1.]}, duration=1.))
    assert terminated and not truncated and info['success']
    assert info['reward_terms']['tumor_removal'] > 0
    with pytest.raises(RuntimeError): e.step_action(command())


@pytest.mark.parametrize('slot', ['left', 'right'])
def test_all_catalog_tools_exchange_through_encoded_actions(slot):
    e = env(); e.reset(seed=3)
    for instrument in INSTRUMENTS:
        action = {'slot': slot, 'operation': 'exchange', 'instrument': instrument, 'duration': .01}
        _, _, terminated, truncated, info = e.step(e.encode_action(action))
        assert info['outcome']['status'] == 'accepted'
        assert e.engine.state['slots'][slot]['instrument'] == instrument
        assert not terminated and not truncated


def test_manifest_required_list_cannot_bypass_strict_brainstem():
    from resectionlab.surgical_rl_protection import ProtectionSet
    source = fixture()
    protection = ProtectionSet(source, required=())
    with pytest.raises(ValueError, match='CRITICAL|brainstem'):
        SurgicalRLEnv(source, protection=protection)


def test_malformed_parameter_container_is_penalized_rejection():
    e = env(); e.reset(seed=1)
    before = e.engine.state_id
    _, reward, _, _, info = e.step_action(command(parameters=[]))
    assert reward < 0 and info['outcome']['status'] == 'rejected'
    assert e.engine.state_id == before
