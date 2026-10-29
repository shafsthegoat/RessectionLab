"""Public setup/removal/position actions, source identity and guarded replay."""
import copy
import json
import numpy as np
import pytest

from resectionlab.surgical_engine import SurgicalEngine
from resectionlab.surgical_scene import SurgicalScene
from resectionlab.surgical_physics_v2 import default_profile
from resectionlab.surgical_replay_cache import SurgicalReplayCache
from resectionlab.surgical_rl_protection import GuardConfig, ProtectionAnnotation, ProtectionSet


def scene(*, v2=True):
    labels = np.zeros((20, 20, 20), np.uint16)
    labels[6:11, 6:11, 8:12] = 1
    return SurgicalScene(labels, np.eye(4),
        [{'id': 1, 'name': 'Synthetic brain block', 'material': 'soft_tissue', 'target': False}],
        {'kind': 'generated local gravity integration coupon; no patient anatomy'},
        {'tissuePhysicsV2': default_profile()} if v2 else {})


def command(operation, *, duration=.1, point=None, **kwargs):
    return {'slot': 'left', 'operation': operation, 'duration': duration,
            **({'pose': {'position': point, 'quaternion': [0., 0., 0., 1.]}} if point is not None else {}), **kwargs}


def accepted(engine, action):
    result = engine.step(action)
    assert result['status'] == 'accepted', result
    return result


def exposure_prefix(engine, *, irrigate=False):
    """Create exposure with accepted time-limited CUSA action, never mutate state."""
    accepted(engine, command('configure'))
    accepted(engine, command('exchange', instrument='cusa'))
    accepted(engine, command('place', point=[8., 8., 4.]))
    contact = engine.step(command('apply', point=[8., 8., 7.], duration=.05))
    assert contact['status'] == 'partial' and contact['reasonCode'] == 'TISSUE_ASPIRATION_IN_PROGRESS'
    assert not engine.state['removed']
    result = accepted(engine, command('apply', duration=2.))
    removed = int(np.ravel_multi_index((8, 8, 8), engine.scene.shape))
    assert result['deltas']['removedCells'] == [removed]
    accepted(engine, command('withdraw'))
    if irrigate:
        accepted(engine, command('exchange', instrument='irrigation'))
        accepted(engine, command('place', point=[3., 10., 4.],
            parameters={'flowMm3PerSecond': 5., 'depthMm': 1.}))
        accepted(engine, command('apply', duration=.1))
        accepted(engine, command('withdraw'))
        assert sum(engine.state['fluid'].values()) == pytest.approx(.5)
    return engine


def position(preset, **kwargs):
    return command('position', duration=.3, position={'preset': preset, **kwargs})


def fluid_centroid(engine):
    total = sum(engine.state['fluid'].values())
    return sum(engine.scene.points([int(c)])[0] * v for c, v in engine.state['fluid'].items()) / total


def test_same_recorded_exposure_prefix_has_position_dependent_gravity_and_conservative_flow():
    source = scene(); labels = source.labels.copy(); properties = copy.deepcopy(source.properties)
    original = exposure_prefix(SurgicalEngine(source), irrigate=True)
    prefix = original.snapshot(); origin = fluid_centroid(original)
    outcomes = {}
    for preset in ('supine', 'prone', 'semisitting'):
        branch = original.branch(len(original.actions))
        accepted(branch, position(preset))
        assert branch.state['removed'] == original.state['removed']
        assert 0 < len(branch.state['deformation']) <= 2000
        assert not set(map(str, branch.state['removed'])) & set(branch.state['deformation'])
        displacement = np.array(list(branch.state['deformation'].values())).sum(axis=0)
        gravity = np.array(branch.view()['positionEffects']['gravityRasMPerS2'])
        assert np.linalg.norm(displacement) > .01 and displacement @ gravity > 0
        assert (fluid_centroid(branch) - origin) @ gravity > 0
        assert sum(branch.state['fluid'].values()) == pytest.approx(.5, abs=1e-12)
        assert branch.state['fluidEscapedMm3'] == 0
        assert branch.metrics()['fluidAddedMm3'] == pytest.approx(.5)
        assert branch.metrics()['fluidPresentMm3'] == pytest.approx(.5)
        outcomes[preset] = displacement
    assert outcomes['supine'][1] < 0 < outcomes['prone'][1]
    assert outcomes['semisitting'][2] < -.01
    assert original.snapshot() == prefix
    np.testing.assert_array_equal(source.labels, labels)
    assert source.properties == properties


def test_snapshot_json_restore_seek_and_branch_reconstruct_exact_posture_endpoint():
    engine = exposure_prefix(SurgicalEngine(scene()), irrigate=True)
    prefix = engine.snapshot(); prefix_length = len(engine.actions)
    accepted(engine, position('semisitting', headRotationRasDeg=[10., -5., 15.],
        siteHeightAboveHeartMm=210., heightEstimateSource='synthetic operator estimate'))
    snapshot = json.loads(json.dumps(engine.snapshot()))
    restored = SurgicalEngine.restore(engine.scene, snapshot)
    assert restored.snapshot() == engine.snapshot()
    np.testing.assert_array_equal(restored.occupancy(), engine.occupancy())
    assert restored.view()['positionEffects'] == engine.view()['positionEffects']
    assert restored.view()['positionEffects']['heightIsMeasured'] is False
    replay = SurgicalReplayCache()
    earlier = replay.seek(engine, snapshot, prefix_length)
    assert earlier.snapshot() == prefix and 'patientPosition' not in earlier.state
    endpoint = replay.seek(engine, snapshot, len(engine.actions))
    assert endpoint.snapshot() == engine.snapshot()
    branch = engine.branch(prefix_length)
    assert branch.snapshot() == prefix
    accepted(branch, position('prone'))
    assert branch.state['patientPosition']['preset'] == 'prone'
    assert engine.state['patientPosition']['preset'] == 'semisitting'


def test_indirect_gravity_guard_rolls_back_entire_setup_action_and_occupancy():
    source = scene(); mask = np.zeros(source.shape, bool); mask[8, 9, 10] = True
    annotation = ProtectionAnnotation.from_mask(source, 'brainstem', mask,
        provenance='generated protected source cell, not inferred anatomy')
    protected = ProtectionSet(source, [annotation], clearance_mm=0)
    engine = exposure_prefix(SurgicalEngine(source, protection=protected,
        guard_config=GuardConfig(critical_max_displacement_mm=0.)))
    before = engine.snapshot(); occupancy = np.array(engine.occupancy())
    trial = position('supine')
    preview = engine.preview(trial)
    assert preview['status'] == 'rejected'
    assert preview['reason'] == 'PROTECTED_STRUCTURE: indirect deformation exceeds the declared limit'
    assert engine.snapshot() == before
    result = engine.step(trial)
    assert result['status'] == 'rejected' and result['stateId'] == before['stateId']
    assert result['reasonCode'] == 'PROTECTED_STRUCTURE'
    assert engine.snapshot() == before and 'patientPosition' not in engine.state
    np.testing.assert_array_equal(engine.occupancy(), occupancy)


@pytest.mark.parametrize('setup', ['configure', 'position'])
@pytest.mark.parametrize('blocked_by', ['inserted', 'active', 'hold'])
def test_setup_requires_withdrawn_inactive_unheld_tools(setup, blocked_by):
    engine = SurgicalEngine(scene())
    if blocked_by == 'inserted':
        accepted(engine, command('move', point=[2., 2., 2.]))
        assert engine.state['slots']['left']['inserted']
    elif blocked_by == 'active':
        accepted(engine, command('activate'))
        assert engine.state['slots']['left']['active']
    else:
        accepted(engine, command('exchange', instrument='brain_retractor'))
        accepted(engine, command('place', point=[8., 8., 4.]))
        accepted(engine, command('hold', point=[8., 8., 6.5]))
        assert engine.state['holds']
    before = engine.snapshot()
    result = engine.step(position('supine') if setup == 'position' else command('configure'))
    assert result['status'] == 'rejected' and result['reasonCode'] == 'WITHDRAW_BOTH_TOOLS_BEFORE_SETUP'
    assert engine.snapshot() == before


@pytest.mark.parametrize('field', ['camera', 'cameraPosition', 'cameraQuaternion', 'zoom'])
def test_camera_like_position_payloads_are_rejected_without_state_effects(field):
    engine = exposure_prefix(SurgicalEngine(scene()))
    before = engine.snapshot(); occupancy = np.array(engine.occupancy())
    result = engine.step(position('supine', **{field: [0, 0, 1]}))
    assert result['status'] == 'rejected' and result['reasonCode'] == 'INVALID_PATIENT_POSITION'
    assert engine.snapshot() == before and 'patientPosition' not in engine.state
    np.testing.assert_array_equal(engine.occupancy(), occupancy)


def test_position_rejects_old_scene_without_silently_upgrading_episode():
    engine = SurgicalEngine(scene(v2=False)); before = engine.snapshot()
    result = engine.step(position('supine'))
    assert result['status'] == 'rejected' and result['reasonCode'] == 'POSITION_REQUIRES_V2_MECHANICS'
    assert engine.snapshot() == before and 'patientPosition' not in engine.view()
