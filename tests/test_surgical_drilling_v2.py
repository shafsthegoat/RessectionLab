import copy
import json
import math
from types import SimpleNamespace
import numpy as np
import pytest
from resectionlab import surgical_drilling_v2 as drill


def engine():
    return SimpleNamespace(scene=SimpleNamespace(
        properties={'tissuePhysicsV2': {'bone': copy.deepcopy(drill.DEFAULT_PROFILE)}},
        labels=np.ones((4, 1, 1), dtype=int), voxel_mm3=1., spacing=np.ones(3)),
        by_label={1: {'material': 'bone'}}, state={'removed': [], 'debris': {}})


def inputs(**updates):
    return dict(rpm=1000., forceN=1., radiusMm=1., **updates)


def test_persistent_work_and_zero_inputs():
    e = engine()
    _, new, contact = drill.apply(e, [0], 1., inputs())
    assert not new and not contact['advanceAllowed']
    pending = e.state['drillingV2']['workJ']['0']
    for p in ({'rpm': 0, 'forceN': 10, 'radiusMm': 1}, {'rpm': 10000, 'forceN': 0, 'radiusMm': 1}):
        drill.apply(e, [0], 1., p)
    assert e.state['drillingV2']['workJ']['0'] == pending
    assert e.state['removed'] == [] and e.state['debris'] == {}
    drill.apply(e, [0], 1., inputs())
    assert e.state['drillingV2']['workJ']['0'] == pytest.approx(2 * pending)


@pytest.mark.parametrize('field', ['rpm', 'forceN'])
def test_speed_force_sensitivity(field):
    a, b = engine(), engine()
    params = inputs(); params[field] *= 2
    drill.apply(a, [0], 1., inputs()); drill.apply(b, [0], 1., params)
    assert drill.metrics(b)['drillPendingWorkJ'] == pytest.approx(2 * drill.metrics(a)['drillPendingWorkJ'])


def test_conservative_contact_distribution_and_removal():
    a, b = engine(), engine()
    drill.apply(a, [0], 1., inputs()); drill.apply(b, [0, 1], 1., inputs())
    assert drill.metrics(a)['drillInputEnergyJ'] == drill.metrics(b)['drillInputEnergyJ']
    assert b.state['drillingV2']['workJ']['0'] == pytest.approx(a.state['drillingV2']['workJ']['0'] / 2)
    _, removed, contact = drill.apply(b, [0, 1], 100., inputs())
    assert removed == [0, 1] and contact['advanceAllowed']
    assert sum(b.state['debris'].values()) == 2.
    m = drill.metrics(b)
    assert m['drillCuttingWorkJ'] == pytest.approx(1.)
    assert m['drillInputEnergyJ'] == pytest.approx(m['drillCuttingWorkJ'] + m['drillUnusedEnergyJ'])
    assert m['drillHeatGeneratedJ'] == pytest.approx(.35)
    drill.apply(b, [0, 1], 100., inputs())
    assert sum(b.state['debris'].values()) == 2.


def test_temperature_cooling_stable_and_dose_independent_removal():
    e = engine(); drill.apply(e, [0], 1., inputs())
    a, b = copy.deepcopy(e), copy.deepcopy(e)
    drill.cool(a, 10.)
    for _ in range(100): drill.cool(b, .1)
    assert a.state['drillingV2']['temperatureC']['0'] == pytest.approx(b.state['drillingV2']['temperatureC']['0'])
    assert a.state['drillingV2']['thermalDoseCSeconds']['0'] == pytest.approx(b.state['drillingV2']['thermalDoseCSeconds']['0'])
    p = drill.profile(a); s = a.state['drillingV2']
    stored = sum((t - p['ambientTemperatureC']) * p['heatCapacityJPerMm3K'] for t in s['temperatureC'].values())
    assert s['heatGeneratedJ'] == pytest.approx(stored + s['coolingLossJ'])
    assert a.state['removed'] == []


def test_caps_determinism_and_readonly():
    a, b = engine(), engine()
    params = {'rpm': 1e10, 'forceN': 1e10, 'radiusMm': 1e10, 'duty': 1e10}
    result = drill.apply(a, [1, 0, 1], .001, params)
    assert result[2]['powerW'] == 30.
    assert result == drill.apply(b, [0, 1], .001, params)
    assert a.state == b.state
    before = copy.deepcopy(a.state)
    d = drill.descriptor(a); d['profile']['maxRpm'] = -1
    drill.metrics(a)
    assert a.state == before and drill.profile(a)['maxRpm'] == 80000.
    with pytest.raises(ValueError): drill.apply(a, [0], float('nan'), inputs())


def test_frontier_clamps_no_tunnel():
    e = engine()
    e.touched = lambda a, b, r: [0] if 1 <= a[0] <= 2 and 0 not in e.state['removed'] else []
    pos, cells = drill.contact_limited_pose(e, [0, 0, 0], [4, 0, 0], .1)
    assert cells == [0] and pos[0] == pytest.approx(1.)
    _, _, contact = drill.apply(e, cells, .1, inputs())
    assert not contact['advanceAllowed']
    pos2, cells2 = drill.contact_limited_pose(e, pos, [4, 0, 0], .1)
    assert np.array_equal(pos, pos2) and cells2 == [0]
    drill.apply(e, cells2, 100., inputs())
    pos3, cells3 = drill.contact_limited_pose(e, pos2, [4, 0, 0], .1)
    assert cells3 == [] and pos3.tolist() == [4., 0., 0.]


def test_unsaturated_heating_and_dwell_partition_invariant():
    a, b = engine(), engine()
    drill.apply(a, [0, 1], 2., inputs())
    for _ in range(20): drill.apply(b, [0, 1], .1, inputs())
    for field in ('workJ', 'temperatureC', 'thermalDoseCSeconds'):
        assert a.state['drillingV2'][field] == pytest.approx(b.state['drillingV2'][field])
    assert drill.metrics(a)['drillCoolingLossJ'] == pytest.approx(drill.metrics(b)['drillCoolingLossJ'])


def test_two_simultaneous_drills_order_independent_shared_thermal_step():
    a, b = engine(), engine()
    # Pre-existing heat also cools exactly once, not once per active tool.
    drill.apply(a, [2], 1., inputs())
    b.state = copy.deepcopy(a.state)
    reference = copy.deepcopy(a)
    drill.cool(reference, .5)
    for target, order in ((a, [0, 1]), (b, [1, 0])):
        drill.begin_step(target)
        before = copy.deepcopy(target.state['drillingV2']['temperatureC'])
        for cell in order:
            drill.apply(target, [cell], .5, inputs(), cool=False)
        assert target.state['drillingV2']['temperatureC'] == before
        drill.end_step(target, .5)
    for field in ('workJ', 'temperatureC', 'thermalDoseCSeconds'):
        assert a.state['drillingV2'][field] == pytest.approx(b.state['drillingV2'][field])
    assert a.state['drillingV2']['temperatureC']['2'] == reference.state['drillingV2']['temperatureC']['2']
    state = a.state['drillingV2']; p = drill.profile(a)
    stored = sum((t - p['ambientTemperatureC']) * p['heatCapacityJPerMm3K'] for t in state['temperatureC'].values())
    assert state['heatGeneratedJ'] == pytest.approx(stored + state['coolingLossJ'])
    assert a._drill_step_powers == {}


def test_shared_step_same_cell_adds_power_and_accumulator_not_aliased():
    a, b = engine(), engine()
    drill.begin_step(a)
    original = a._drill_step_powers
    drill.apply(a, [0], .1, inputs(), cool=False)
    assert original == {}
    drill.apply(a, [0], .1, inputs(), cool=False)
    drill.end_step(a, .1)
    params = inputs(); params['forceN'] *= 2
    drill.apply(b, [0], .1, params)
    for field in ('workJ', 'temperatureC', 'thermalDoseCSeconds'):
        assert a.state['drillingV2'][field] == pytest.approx(b.state['drillingV2'][field])
    drill.begin_step(a)
    assert a._drill_step_powers == {}


def _legacy_thermal(e, dt, powers):
    """Frozen pre-optimization arithmetic from 5d05ddb3 for replay parity."""
    p = drill.profile(e); state = e.state['drillingV2']
    capacity = p['heatCapacityJPerMm3K'] * e.scene.voxel_mm3
    tau = p['coolingTimeSeconds']; ambient = p['ambientTemperatureC']
    decay = math.exp(-dt / tau)
    for key in sorted(set(state['temperatureC']) | set(powers), key=int):
        old = state['temperatureC'].get(key, ambient) - ambient
        power = powers.get(key, 0.)
        equilibrium = power * tau / capacity
        new = old * decay + equilibrium * (-math.expm1(-dt / tau))
        state['temperatureC'][key] = ambient + new
        state['thermalDoseCSeconds'][key] = state['thermalDoseCSeconds'].get(key, 0.) + drill._dose_integral(old, equilibrium, p['doseThresholdC'] - ambient, tau, dt)
        state['coolingLossJ'] += max(0., power * dt - capacity * (new - old))


@pytest.mark.parametrize('dt', [0., math.ulp(0.), 1.e-20, .001, .1, 30.])
def test_thermal_shortcut_preserves_exact_state_and_float_boundary_arithmetic(dt):
    for ambient, threshold, tau in [(37., 43., 8.), (-37., -31., 8.),
                                    (0., 0., 8.), (-0., 6., 1.e100)]:
        actual = engine()
        p = actual.scene.properties['tissuePhysicsV2']['bone']
        p.update(ambientTemperatureC=ambient, doseThresholdC=threshold, coolingTimeSeconds=tau)
        drill.initialize(actual)
        thermal = actual.state['drillingV2']
        thermal['temperatureC'] = {'2': float(np.nextafter(threshold, math.inf)),
            '0': ambient, '1': threshold, '3': float(np.nextafter(threshold, -math.inf))}
        thermal['thermalDoseCSeconds'] = {'2': -0., '0': .125, '1': 0.}
        thermal['coolingLossJ'] = -0.
        expected = copy.deepcopy(actual)
        # Cover both zero-power cooling and existing/new powered fields.
        powers = {'2': .001, '4': .002}
        _legacy_thermal(expected, dt, powers)
        drill._thermal(actual, dt, powers)
        encode = lambda state: json.dumps(state, separators=(',', ':'), allow_nan=False)
        assert encode(actual.state) == encode(expected.state)
        assert list(actual.state['drillingV2']['temperatureC']) == list(expected.state['drillingV2']['temperatureC'])


def test_threshold_equality_retains_original_dose_integral(monkeypatch):
    e = engine(); drill.initialize(e)
    e.state['drillingV2']['temperatureC'] = {'0': 37., '1': 43.}
    original = drill._dose_integral
    calls = []
    def observed(*args):
        calls.append(args)
        return original(*args)
    monkeypatch.setattr(drill, '_dose_integral', observed)
    drill._thermal(e, 1.e-20, {})
    assert len(calls) == 1 and calls[0][0] == 6.


def test_public_drill_preview_partial_rejection_and_cooling_match_legacy(monkeypatch):
    from resectionlab.surgical_engine import SurgicalEngine
    from resectionlab.surgical_scene import SurgicalScene
    from resectionlab.surgical_physics_v2 import default_profile
    labels = np.zeros((16, 16, 16), np.uint16)
    labels[4:7, 4:7, 8] = 1; labels[5, 5, 6] = 2
    scene = SurgicalScene(labels, np.eye(4), [
        {'id': 1, 'name': 'bone', 'material': 'bone'},
        {'id': 2, 'name': 'target', 'material': 'tumor', 'target': True}],
        {'kind': 'explicit-thermal-parity-fixture'}, {'tissuePhysicsV2': default_profile()})
    pose = lambda z: {'position': [5., 5., z], 'quaternion': [1., 0., 0., 0.]}
    actions = [
        {'slot': 'left', 'operation': 'place', 'pose': pose(15.), 'duration': .2,
         'parameters': {'radiusMm': 1., 'forceN': 0., 'rpm': 0.}},
        {'slot': 'left', 'operation': 'apply', 'pose': pose(7.), 'duration': .1},
        {'slot': 'left', 'operation': 'apply', 'pose': pose(7.), 'duration': 5.,
         'parameters': {'forceN': 20., 'rpm': 80000.}},
        {'slot': 'left', 'operation': 'exchange', 'instrument': 'cusa', 'duration': .2},
        {'slot': 'left', 'operation': 'withdraw', 'pose': pose(15.), 'duration': .7},
        {'slot': 'right', 'operation': 'inspect', 'duration': 2.5},
    ]
    current = drill._thermal
    runs = []
    for implementation in (_legacy_thermal, current):
        monkeypatch.setattr(drill, '_thermal', implementation)
        actual = SurgicalEngine(scene)
        outcomes = []
        for action in actions:
            preview = actual.preview(action, include_view=False)
            outcome = actual.step(action)
            assert outcome['stateId'] == preview['stateId']
            outcomes.append(outcome)
        assert actual.metrics()['drillRemovedVolumeMm3'] > 0.
        assert {'accepted', 'partial', 'rejected'} <= {row['status'] for row in outcomes}
        runs.append((actual.snapshot(), outcomes, actual.history))
    assert runs[0] == runs[1]
