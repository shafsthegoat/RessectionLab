import numpy as np
import pytest
from scipy.spatial.transform import Rotation, Slerp

from resectionlab import surgical_vessel_rods as rods
from resectionlab.surgical_engine import SurgicalEngine
from resectionlab.surgical_scene import SurgicalScene, propagate_original_target_memberships
from test_surgical_vessel_rods import _action, _fixture


def clock_scene(scene, policy=rods.TOOL_CONTACT_CLOCK_POLICY, **properties):
    active_scene = SurgicalScene(scene.labels, scene.affine, scene.regions, scene.source,
                                 {**scene.properties, **properties, rods.TOOL_CONTACT_CLOCK_PROPERTY: policy})
    return propagate_original_target_memberships(scene, active_scene)


def start(scene):
    engine = SurgicalEngine(scene)
    for action in (_action('exchange', instrument='dissector', dt=.08), _action('place', z=4., dt=.16)):
        assert engine.step(action)['status'] == 'accepted'
    return engine


def uniform(action, start_z, count):
    endpoint = action['pose']
    path = [{**endpoint, 'position': [*endpoint['position'][:2], z]}
            for z in np.linspace(start_z, endpoint['position'][2], count + 1)[1:]]
    return {**{key: value for key, value in action.items() if key != 'pose'}, 'path': path}


def wall_displacement(engine):
    wall = rods.view(engine)['vertexDisplacementsByLabel']['2']
    field = np.zeros((wall['vertexCount'], 3))
    field[wall['indices']] = np.asarray(wall['offsetsMm']).reshape(-1, 3)
    return field


def test_same_passive_trajectory_reaches_close_wall_states_at_different_waypoint_counts():
    original = _fixture(guarded=True, vertex_transport=True, section_mechanics=True, conditioned=True,
                        soft_contact=True, soft_bank=True, bounded_tissue=True).scene
    fixed_tick = clock_scene(original)
    assert original.identity != fixed_tick.identity
    outcomes = []
    for scene in (original, fixed_tick):
        initial = start(scene)
        # The approach begins between rod ticks. A pause of another sub-tick
        # fraction makes the final withdrawal start at a different phase.
        period = rods.STEP_SECONDS
        assert 0. < initial.state[rods.FIELD]['remainderSeconds'] < period
        endpoints = [_action('move', z=5.05, dt=.16), _action('inspect', dt=.037), _action('withdraw', z=4., dt=.35)]
        dense = [uniform(endpoints[0], 4., 24), endpoints[1], uniform(endpoints[2], 5.05, 48)]
        boundaries = []
        for route in (endpoints, dense):
            engine = initial.clone()
            frames = []
            for action in route:
                outcome = engine.step(action)
                assert outcome['status'] == 'accepted', outcome
                assert outcome['elapsedSeconds'] == pytest.approx(action['duration'], abs=1e-12)
                frames.append(wall_displacement(engine))
            assert not rods.view(engine)['contactedCells']
            assert np.linalg.norm(frames[2], axis=1).max() < np.linalg.norm(frames[0], axis=1).max()
            boundaries.append(frames)
            restored = SurgicalEngine.restore(scene, engine.snapshot(), force_verify=True)
            assert restored.state_id == engine.state_id
        outcomes.append([np.linalg.norm(a-b, axis=1).max() for a, b in zip(*boundaries)])
    assert outcomes[0][0] > .01
    assert all(delta < 1e-4 for delta in outcomes[1][:2])
    assert outcomes[1][2] < outcomes[0][2] / 10.


def test_fixed_contact_clock_declines_unsupported_tool_motion(monkeypatch):
    original = _fixture(guarded=True, vertex_transport=True, section_mechanics=True, conditioned=True,
                        soft_contact=True, bounded_tissue=True).scene
    with pytest.raises(ValueError, match='VESSEL_TOOL_CONTACT_CLOCK_REQUIRES_SUPPORTED_NATIVE_WALL'):
        SurgicalEngine(clock_scene(original, 'unknown'))
    observed = []
    tick = rods.tick

    def capture(engine, dt, **kwargs):
        observed.append(kwargs.get('tool_clock_hand'))
        return tick(engine, dt, **kwargs)

    monkeypatch.setattr(rods, 'tick', capture)
    idle = start(clock_scene(original, rods.AXIAL_TOOL_CONTACT_CLOCK_POLICY))
    for action in (
            _action('move', z=4.09, parameters={'depthMm': .8}),
            _action('move', pose={'position': [6.82, 6., 4.09], 'quaternion': [0., 0., 0., 1.]}),
            _action('move', pose={'position': [6.8, 6., 4.09], 'quaternion': Rotation.from_rotvec([.005, 0., 0.]).as_quat().tolist()})):
        observed.clear()
        assert idle.clone().step(action)['status'] == 'accepted'
        assert observed and all(hand is None for hand in observed)

    active = start(clock_scene(original, attachments=[]))
    assert active.step(_action('activate'))['status'] == 'accepted'
    observed.clear()
    assert active.step(_action('move', z=4.09))['status'] == 'accepted'
    assert observed and all(hand is None for hand in observed)
    observed.clear()
    assert idle.step(_action('move', z=4.09))['status'] == 'accepted'
    assert observed and all(hand == 'left' for hand in observed)
    observed.clear()
    assert idle.step(_action('withdraw', z=4.))['status'] == 'accepted'
    assert observed and all(hand == 'left' for hand in observed)
    rigid = start(clock_scene(original))
    for pose in (
            {'position': [6.82, 6., 4.09], 'quaternion': [0., 0., 0., 1.]},
            {'position': [6.8, 6., 4.09], 'quaternion': Rotation.from_rotvec([.005, 0., 0.]).as_quat().tolist()}):
        observed.clear()
        assert rigid.clone().step(_action('move', pose=pose))['status'] == 'accepted'
        assert observed and all(hand == 'left' for hand in observed)
    observed.clear()
    assert rigid.step(_action('move', z=4.09, parameters={'depthMm': .8}))['status'] == 'accepted'
    assert observed and all(hand is None for hand in observed)


def test_clock_contact_broad_phase_keeps_earlier_capsule_coverage(monkeypatch):
    capsule = ('left', np.array([3., 0., 0.]), np.array([4., 0., 0.]), .1, .001)
    monkeypatch.setattr(rods, '_work_capsules', lambda _: (capsule,))
    lo, hi = np.array([0., -1., -1.]), np.ones(3)
    assert not tuple(rods._swept_work_capsules(None, lo, hi, 'right', [-3., 0., 0.]))
    assert tuple(rods._swept_work_capsules(None, lo, hi, 'left', [-3., 0., 0.])) == (capsule,)


def test_rigid_sampling_retains_rotational_candidates_and_rewinds_friction_pivot(monkeypatch):
    origin = np.array([1., 2., 0.])
    linear = np.array([2., 0., 0.])
    angular = np.array([0., 0., np.pi])
    capsule = ('left', origin + [1., 0., 0.], origin + [1., 0., 1.], .1, .001)
    stationary = ('right', np.full(3, 5.), np.full(3, 6.), .1, .001)
    monkeypatch.setattr(rods, '_work_capsules', lambda _: (capsule, stationary))
    lo, hi = np.array([-.03, .97, -.03]), np.array([.03, 1.03, .03])
    assert not tuple(rods._swept_work_capsules(None, lo, hi, 'left', -linear / 2))
    tools = tuple(rods._swept_work_capsules(None, lo, hi, 'left', -linear / 2, (origin, np.pi / 2)))
    assert len(tools)==1 and tools[0] is capsule
    sampled, motion = rods._rigid_tool_sample((capsule, stationary), 'left', .5, {'left': (origin, linear, angular)})
    np.testing.assert_allclose(sampled[0][1], [0., 1., 0.], atol=1e-12)
    np.testing.assert_allclose(sampled[0][2], [0., 1., 1.], atol=1e-12)
    np.testing.assert_allclose(motion['left'][0], [0., 2., 0.], atol=1e-12)
    np.testing.assert_array_equal(sampled[1][1], stationary[1])
    np.testing.assert_array_equal(sampled[1][2], stationary[2])


def test_loaded_lateral_rotation_converges_across_equal_spaced_requests():
    scene = _fixture(guarded=True, vertex_transport=True, section_mechanics=True, conditioned=True,
                     soft_contact=True, soft_bank=True, bounded_tissue=True).scene
    errors = []
    for policy in (rods.AXIAL_TOOL_CONTACT_CLOCK_POLICY, rods.TOOL_CONTACT_CLOCK_POLICY):
        version = clock_scene(scene, policy)
        initial = start(version)
        for action in (_action('move', z=5.08, dt=.4), _action('inspect', dt=.043)):
            assert initial.step(action)['status']=='accepted'
        slot = initial.state['slots']['left']
        origin = np.asarray(slot['position'])
        turn = Rotation.from_rotvec([.0025, 0., 0.]).as_quat()
        end = origin + [.22, 0., 0.]
        interpolation = Slerp([0., 1.], Rotation.from_quat([slot['quaternion'], turn]))
        dense = [{'position': origin + t * (end - origin), 'quaternion': interpolation(t).as_quat()}
                 for t in np.arange(1, 25) / 24]
        dense = [{key: value.tolist() for key, value in pose.items()} for pose in dense]
        states = []
        for action in (_action('move', pose=dense[-1], dt=.23),
                       {**{key: value for key, value in _action('move', dt=.23).items() if key!='pose'}, 'path':dense}):
            engine = initial.clone()
            outcome = engine.step(action)
            assert outcome['status']=='accepted', outcome
            assert rods.view(engine)['contactedCells']['left']
            states.append(wall_displacement(engine))
            assert SurgicalEngine.restore(version,engine.snapshot(),force_verify=True).state_id==engine.state_id
        errors.append(np.linalg.norm(states[0]-states[1],axis=1).max())
    assert errors[0]>1e-4 and errors[1]<errors[0]/3, errors
