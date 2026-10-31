"""Generated scheduling controls; no acquired anatomy or biological validation."""
import copy
import json

import numpy as np
import pytest

from resectionlab.surgical_engine import SurgicalEngine
from resectionlab.surgical_scene import SurgicalScene
from resectionlab.surgical_cusa_v3 import default_profile
from resectionlab.surgical_verified_cache import VerifiedSurgicalCache
from test_surgical_cusa_v3 import fixture
from test_surgical_bipolar_thermal import public_engine, place_bipolar, public_act


def act(e, operation, *, hand='left', point=None, duration=.1, **kwargs):
    a = {'slot': hand, 'operation': operation, 'duration': duration, **kwargs}
    if point is not None: a['pose'] = {'position': point, 'quaternion': [0., 0., 0., 1.]}
    result = e.step(a)
    assert result['status'] != 'rejected', result
    return result


def enable(e):
    act(e, 'configure', clockRevision=1)


def cusa_contact(*, shared=True):
    e, _ = fixture(rate=.01)
    if shared: enable(e)
    act(e, 'exchange', instrument='cusa')
    act(e, 'place', point=[9., 9., 3.])
    act(e, 'apply', point=[9., 9., 11.], duration=.05)
    act(e, 'activate')
    return e


@pytest.mark.parametrize('hand,expected', [
    ('left', 'sha256:3d69b68e78402f7a59ac47b8c79d44109d6c1d9af7ca03f5c5cdd6ddcf95436d'),
    ('right', 'sha256:e4891099cf63ca3e880361ca6b52e87a71d5114391a8868e56902b83bd63184c')])
def test_prechange_legacy_state_ids_and_replay_are_exact(hand, expected):
    e = cusa_contact(shared=False)
    act(e, 'inspect', hand=hand, duration=.2)
    assert e.state_id == expected
    assert 'physicalClockRevision' not in e.state
    assert SurgicalEngine.restore(e.scene, json.loads(json.dumps(e.snapshot()))).snapshot() == e.snapshot()


def test_own_and_other_hand_inspection_have_identical_complete_physical_state():
    initial = cusa_contact()
    a, b = initial.clone(), initial.clone()
    before = initial.state['cusaV3']['inputCapacityMm3']
    act(a, 'inspect', duration=.2)
    act(b, 'inspect', hand='right', duration=.2)
    assert a.state == b.state and a.state_id == b.state_id
    assert a.state['cusaV3']['inputCapacityMm3'] - before == pytest.approx(.002, abs=1e-15)
    assert a.state['elapsedSeconds'] - initial.state['elapsedSeconds'] == pytest.approx(.2)
    assert a.state['slots']['left']['active']


def test_configuration_is_replayed_without_changing_prefix_or_resetting_burdens():
    e, _ = fixture()
    act(e, 'configure', mechanicsRevision=2)
    act(e, 'inspect')
    before = copy.deepcopy(e.state)
    prefix = e.snapshot()
    enable(e)
    assert e.state == {**before, 'physicalClockRevision': 1}
    assert e.branch(len(prefix['actions'])).snapshot() == prefix
    assert e.observe()['physicalClock']['revision'] == 1
    assert e.observe()['physicalClock']['maxStepSeconds'] == .1
    assert e.state['toolMechanicsRevision'] == 2
    assert e.fresh().state == SurgicalEngine(e.scene).state
    act(e, 'configure', mechanicsRevision=1)
    assert e.state['physicalClockRevision'] == 1


@pytest.mark.parametrize('value', [True, 0, 2, '1', None])
def test_invalid_revision_is_rejected_atomically(value):
    e, _ = fixture(); before = e.snapshot()
    r = e.step({'slot': 'left', 'operation': 'configure', 'clockRevision': value, 'duration': .1})
    assert r['reasonCode'] == 'INVALID_PHYSICAL_CLOCK_REVISION'
    assert e.snapshot() == before


def test_clock_upgrade_requires_withdrawn_inactive_tools_and_cannot_be_an_action_parameter():
    e = cusa_contact(shared=False); before = e.snapshot()
    r = e.step({'slot': 'right', 'operation': 'configure', 'clockRevision': 1, 'duration': .1})
    assert r['reasonCode'] == 'WITHDRAW_BOTH_TOOLS_BEFORE_SETUP'
    assert e.snapshot() == before
    r = e.step({'slot': 'right', 'operation': 'inspect', 'clockRevision': 1, 'duration': .1})
    assert r['reasonCode'] == 'INVALID_PHYSICAL_CLOCK_REVISION'
    assert e.snapshot() == before


def test_two_active_cusas_receive_one_interval_each_with_one_elapsed_clock():
    labels = np.zeros((24, 24, 24), np.uint16)
    labels[7, 9, 9] = labels[17, 9, 9] = 1
    p = default_profile(); p['capacityMm3PerSecond'] = .01
    scene = SurgicalScene(labels, np.eye(4), [{'id': 1, 'name': 'Generated two-tip coupon', 'material': 'tumor', 'target': True}],
                          {'kind': 'generated-scheduling-control'}, {'cusaV3': p})
    e = SurgicalEngine(scene); enable(e)
    for hand, x in [('left', 7.), ('right', 17.)]:
        act(e, 'exchange', hand=hand, instrument='cusa')
        act(e, 'place', hand=hand, point=[x, 9., 3.])
        act(e, 'apply', hand=hand, point=[x, 9., 11.], duration=.05)
        act(e, 'activate', hand=hand)
    a, b = e.clone(), e.clone(); before = copy.deepcopy(e.state)
    act(a, 'inspect', duration=.2); act(b, 'inspect', hand='right', duration=.2)
    assert a.state == b.state
    assert a.state['elapsedSeconds'] - before['elapsedSeconds'] == pytest.approx(.2)
    assert a.state['cusaV3']['inputCapacityMm3'] - before['cusaV3']['inputCapacityMm3'] == pytest.approx(.004)
    for cell, capacity in before['cusaV3']['progressMm3'].items():
        assert a.state['cusaV3']['progressMm3'][cell] - capacity == pytest.approx(.002)


def test_matched_time_partition_and_zero_power_deactivation_preserve_ledgers():
    e = cusa_contact(); a, b = e.clone(), e.clone()
    act(a, 'inspect', duration=.2)
    for hand in ('left', 'right'): act(b, 'inspect', hand=hand)
    assert a.state == b.state
    initial = a.state['cusaV3']['inputCapacityMm3']
    act(a, 'inspect', parameters={'power': 0.}, duration=.2)
    act(a, 'deactivate')
    act(a, 'inspect', hand='right', duration=.2)
    assert a.state['cusaV3']['inputCapacityMm3'] == initial
    assert a.state['cusaV3']['progressExact']


def test_one_shot_and_latched_activation_have_same_physics_for_matched_intervals():
    e = cusa_contact(); act(e, 'deactivate')
    one_shot, latched = e.clone(), e.clone()
    act(one_shot, 'apply', duration=.2); act(one_shot, 'inspect')
    act(latched, 'activate'); act(latched, 'inspect'); act(latched, 'deactivate')
    assert one_shot.state == latched.state


def test_uneven_fixed_contact_time_partition_has_declared_roundoff_tolerance():
    e = cusa_contact(); a, b = e.clone(), e.clone()
    act(a, 'inspect', duration=.3)
    for dt in (.013, .077, .041, .169): act(b, 'inspect', hand='right', duration=dt)
    assert a.state['removed'] == b.state['removed'] == []
    for field in ('inputCapacityMm3', 'effectiveActivationSeconds', 'unusedCapacityMm3'):
        assert a.state['cusaV3'][field] == pytest.approx(b.state['cusaV3'][field], rel=0., abs=1e-14)
    assert a.state['cusaV3']['progressMm3'] == pytest.approx(b.state['cusaV3']['progressMm3'], rel=0., abs=1e-14)
    assert a.state['elapsedSeconds'] == pytest.approx(b.state['elapsedSeconds'], rel=0., abs=1e-14)


def test_thermal_and_mechanical_fields_share_inspection_time_and_cooling_continues():
    e = public_engine(); enable(e); place_bipolar(e)
    public_act(e, 'activate', parameters={'power': 1.}, duration=.2)
    a, b = e.clone(), e.clone(); initial = e.metrics()['bipolarHeatInputJ']
    act(a, 'inspect', duration=.2); act(b, 'inspect', hand='right', duration=.2)
    assert a.state == b.state
    assert a.metrics()['bipolarHeatInputJ'] - initial == pytest.approx(.004)
    elapsed = a.state['elapsedSeconds']
    act(a, 'deactivate'); input_j = a.metrics()['bipolarHeatInputJ']
    act(a, 'inspect', hand='right', duration=.2)
    assert a.metrics()['bipolarHeatInputJ'] == input_j
    assert a.state['bipolarThermalV1']['elapsedSeconds'] == pytest.approx(a.state['elapsedSeconds'])
    assert a.state['elapsedSeconds'] == pytest.approx(elapsed + .3)


def test_preview_rejection_and_cancellation_do_not_publish_time_or_capacity():
    e = cusa_contact(); before = e.snapshot()
    preview = e.preview({'slot': 'left', 'operation': 'inspect', 'duration': .2})
    assert preview['status'] == 'accepted'
    assert e.snapshot() == before
    bad = e.step({'slot': 'left', 'operation': 'inspect', 'parameters': {'power': 2.}, 'duration': .2})
    assert bad['status'] == 'rejected' and e.snapshot() == before
    calls = 0
    def cancel():
        nonlocal calls
        calls += 1
        if calls == 12: raise RuntimeError('cancel clock control')
    with pytest.raises(RuntimeError, match='cancel clock control'):
        e.step({'slot': 'left', 'operation': 'inspect', 'duration': 1.}, check=cancel)
    assert calls == 12 and e.snapshot() == before


def test_cold_cached_and_branched_replay_preserve_explicit_clock(tmp_path):
    e = cusa_contact(); act(e, 'inspect', hand='right', duration=.2)
    record = json.loads(json.dumps(e.snapshot()))
    cache = VerifiedSurgicalCache(tmp_path / 'cache')
    for use_cache in (False, True, True):
        actual = SurgicalEngine.restore(e.scene, record, use_cache=use_cache, verified_cache=cache)
        assert actual.snapshot() == record and actual.history == e.history
    assert cache.hits == 1
    for position in range(len(e.actions) + 1):
        branch = e.branch(position)
        assert branch.state.get('physicalClockRevision') == (1 if position else None)
        assert SurgicalEngine.restore(e.scene, branch.snapshot()).state == branch.state


def test_persistent_irrigation_and_suction_are_hand_invariant_and_conservative():
    e, _ = fixture(points=()); enable(e)
    for hand, tool, point in [('left', 'irrigation', [4., 4., 4.]),
                              ('right', 'frazier_suction', [10., 4., 4.])]:
        act(e, 'exchange', hand=hand, instrument=tool)
        act(e, 'place', hand=hand, point=point, parameters={'flowMm3PerSecond': 3.})
        act(e, 'activate', hand=hand)
    a, b = e.clone(), e.clone(); before = e.state['fluidAddedMm3']
    act(a, 'inspect', duration=.2); act(b, 'inspect', hand='right', duration=.2)
    assert a.state == b.state
    assert a.state['fluidAddedMm3'] - before == pytest.approx(.6)
    assert a.state['fluidAddedMm3'] == pytest.approx(
        sum(a.state['fluid'].values()) + a.state['fluidRemovedMm3'] + a.state['fluidEscapedMm3'])


def test_retractor_hold_deformation_and_release_persist_across_inspections():
    from test_surgical_revised_tools import coupon, prepare, accepted, pose
    points = tuple((x, y, 10) for x in range(7, 10) for y in range(7, 10))
    e = coupon('soft_tissue', points); enable(e)
    prepare(e, 'brain_retractor', radius=2.)
    accepted(e, 'hold', pose=pose(9.8))
    accepted(e, 'move', pose=pose(9.8, x=8.3), duration=.4)
    a, b = e.clone(), e.clone()
    act(a, 'inspect', duration=.2); act(b, 'inspect', hand='right', duration=.2)
    assert a.state == b.state and a.state['holds']
    assert max(np.linalg.norm(d) for d in a.state['deformation'].values()) > 0.
    act(a, 'release', duration=.2)
    assert not a.state['holds'] and a.state['deformation']
    assert SurgicalEngine.restore(a.scene, a.snapshot()).state == a.state


def test_vascular_graph_compression_flow_and_storage_are_hand_invariant():
    from test_surgical_vascular_graph import engine as vascular_engine
    e = SurgicalEngine(vascular_engine().scene); enable(e)
    act(e, 'configure', mechanicsRevision=2)
    act(e, 'exchange', instrument='bipolar_forceps')
    act(e, 'place', point=[2.5, 1., -4.], parameters={'jawOpeningMm': .5, 'forceN': 1., 'power': 0., 'depthMm': 1.})
    act(e, 'apply', point=[2.5, 1., 1.8], duration=.5)
    act(e, 'activate')
    a, b = e.clone(), e.clone()
    act(a, 'inspect', duration=.2); act(b, 'inspect', hand='right', duration=.2)
    assert a.state == b.state
    assert a.state['vascularGraphV1']['edges']['artery']['compressionFraction'] == 1.
    from resectionlab.surgical_vascular_graph import metrics as vascular_metrics
    assert vascular_metrics(a)['conservationResidualMm3'] == pytest.approx(0., abs=1e-12)


def test_failed_second_microstep_retains_only_accepted_time_and_capacity(monkeypatch):
    e = cusa_contact(); expected = e.clone(); act(expected, 'inspect', duration=.3 / 3)
    original = SurgicalEngine._physical_substep
    def stop_after_effects(self, *args):
        result = original(self, *args)
        self._clock_test_calls = getattr(self, '_clock_test_calls', 0) + 1
        if self._clock_test_calls == 2: raise ValueError('CLOCK_TEST_POST_EFFECT_STOP')
        return result
    monkeypatch.setattr(SurgicalEngine, '_physical_substep', stop_after_effects)
    result = act(e, 'inspect', duration=.3)
    assert result['status'] == 'partial' and result['elapsedSeconds'] == pytest.approx(.1)
    assert e.state == expected.state


def test_observed_frames_keep_persistent_activation_and_exact_endpoint():
    from resectionlab.surgical_frame_replay import SurgicalFrameCache
    from resectionlab.surgical_replay_cache import SurgicalReplayCache
    e = cusa_contact(); act(e, 'inspect', duration=.4)
    snapshot = e.snapshot(); position = len(e.actions) - 1
    frames, replay = SurgicalFrameCache(), SurgicalReplayCache()
    observed, _, info = frames.seek(e, snapshot, position, 1, replay)
    assert info['readOnly'] and info['effectiveActivation'] == {'left': True, 'right': False}
    assert observed.state['cusaV3']['inputCapacityMm3'] > e.branch(position).state['cusaV3']['inputCapacityMm3']
    endpoint, _, _ = frames.seek(e, snapshot, position, info['frameCount'] - 1, replay)
    assert endpoint.snapshot() == snapshot and e.snapshot() == snapshot
