import copy
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
