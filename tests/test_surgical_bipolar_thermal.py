"""Independent heat balances, analytic limits and public electrode integration."""
import copy
import math
from types import SimpleNamespace

import numpy as np
import pytest

from resectionlab import surgical_bipolar_thermal as heat
from resectionlab.surgical_engine import SurgicalEngine
from resectionlab.surgical_scene import SurgicalScene
from resectionlab.surgical_physics_v2 import default_profile

CAPACITY = .00383
CONDUCTIVITY = .000553


def make_engine(labels=None, affine=None, *, cells=None, boundary='insulated', transfer=0., maximum=80.):
    if labels is None: labels = np.ones((2, 1, 1), np.uint16)
    if affine is None: affine = np.eye(4)
    if cells is None: cells = list(map(int, np.flatnonzero(labels)))
    materials = {str(int(labels.flat[c])): {'volumetricHeatCapacityJPerMm3K': CAPACITY,
        'conductivityWPerMmK': CONDUCTIVITY, 'source': 'Generated test coefficients'} for c in cells}
    profile = heat.research_profile(cells, materials, ambient_c=33., watts_per_activation=.02,
                                   maximum_c=maximum, retained_boundary=boundary, exposed_transfer=transfer)
    scene = SurgicalScene(labels, affine,
        [{'id': int(i), 'name': f'Material{i}', 'material': 'soft_tissue'} for i in np.unique(labels) if i],
        {'kind': 'generated-control'}, {'tissuePhysicsV2': default_profile(), heat.PROPERTY: profile})
    e = SimpleNamespace(scene=scene, state={'removed': [], 'cutEdges': [], 'transectedEdges': []})
    heat.initialize(e)
    return e


def seed(e, values):
    state = e.state[heat.PROPERTY]
    state['temperatureRiseK'] = {str(c): t for c, t in values.items()}
    # Manufactured initial energy: not an experimental temperature measurement.
    energy = sum(e.scene.properties[heat.PROPERTY]['materials'][str(int(e.scene.labels.flat[c]))]
                 ['volumetricHeatCapacityJPerMm3K'] * e.scene.voxel_mm3 * v for c, v in values.items())
    state['inputEnergyJ'] = energy; state['storedHeatJ'] = energy


def advance(e, dt, contacts=(), activation=0.):
    heat.begin_step(e); heat.deposit(e, contacts, activation); heat.end_step(e, dt)


def test_isolated_cell_heat_capacity_and_delivered_power_not_contact_count():
    results = []
    for n in (1, 2):
        a = np.diag([1/n, 1/n, 1/n, 1.])
        e = make_engine(np.ones((n, n, n), np.uint16), a)
        for _ in range(20): advance(e, .1, range(n**3), 1.)
        s = e.state[heat.PROPERTY]; results.append(list(s['temperatureRiseK'].values()))
        assert list(s['temperatureRiseK'].values()) == pytest.approx([.04/CAPACITY]*(n**3))
        assert s['inputEnergyJ'] == pytest.approx(.04)
        assert s['storedHeatJ'] == pytest.approx(.04)
        assert abs(s['energyResidualJ']) < 1e-15
    assert results[0][0] == pytest.approx(results[1][0])


def test_closed_pair_backward_euler_and_continuous_refinement():
    errors = []
    analytic = 5 * (1 - math.exp(-2 * CONDUCTIVITY / CAPACITY))
    for steps in (1, 2, 4, 8):
        e = make_engine(); seed(e, {0: 10.})
        for _ in range(steps): advance(e, 1/steps)
        actual = e.state[heat.PROPERTY]['temperatureRiseK']
        discrete = 5 * (1 - (1 + 2*CONDUCTIVITY/CAPACITY/steps)**(-steps))
        assert actual['1'] == pytest.approx(discrete, rel=1e-12)
        assert actual['0'] + actual['1'] == pytest.approx(10.)
        assert e.state[heat.PROPERTY]['storedHeatJ'] == pytest.approx(.0383)
        errors.append(abs(actual['1'] - analytic))
    assert all(a > b for a, b in zip(errors, errors[1:]))
    assert errors[-1] < .02


def test_declared_surface_bath_accounts_for_energy_and_cooling_after_deactivation():
    h = .0001; e = make_engine(np.ones((1, 1, 1), np.uint16), transfer=h)
    seed(e, {0: 10.}); advance(e, 1.)
    state = e.state[heat.PROPERTY]; expected = 10 / (1 + 6*h/CAPACITY)
    assert state['temperatureRiseK']['0'] == pytest.approx(expected)
    assert state['bathExportJ'] == pytest.approx(6*h*expected)
    assert state['inputEnergyJ'] == pytest.approx(state['storedHeatJ'] + state['bathExportJ'])
    assert state['excessTemperatureIntegralKSeconds']['0'] == pytest.approx(expected)


@pytest.mark.parametrize('field', ['cutEdges', 'transectedEdges'])
def test_cut_face_cannot_conduct_through_a_cavity(field):
    e = make_engine(); seed(e, {0: 10.}); e.state[field] = [[0, 1]]
    advance(e, 2.)
    assert e.state[heat.PROPERTY]['temperatureRiseK'] == {'0': pytest.approx(10.)}
    assert e.state[heat.PROPERTY]['bathExportJ'] == 0.


def test_removed_heat_is_exported_once_and_removed_cell_cannot_bridge_neighbors():
    e = make_engine(np.ones((3, 1, 1), np.uint16)); seed(e, {0: 10., 1: 3.})
    e.state['removed'] = [1]; advance(e, .5); advance(e, .5)
    state = e.state[heat.PROPERTY]
    assert state['removedHeatExportJ'] == pytest.approx(CAPACITY*3)
    assert state['temperatureRiseK'] == {'0': pytest.approx(10.)}
    assert abs(state['energyResidualJ']) < 1e-15


def test_outside_patch_removal_invalidates_retained_boundary_factor():
    e = make_engine(np.ones((3, 1, 1), np.uint16), cells=[1], boundary='ambient-at-patch-face')
    seed(e, {1: 10.}); advance(e, .1)
    before = e.state[heat.PROPERTY]['temperatureRiseK']['1']
    assert before == pytest.approx(10 / (1 + 4*CONDUCTIVITY*.1/CAPACITY))
    e.state['removed'] = [0]; advance(e, .1)
    assert e.state[heat.PROPERTY]['temperatureRiseK']['1'] == pytest.approx(before / (1 + 2*CONDUCTIVITY*.1/CAPACITY))


@pytest.mark.parametrize('basis', [np.diag([1., 2., 3.]), np.array([[0., -2., 0.], [1., 0., 0.], [0., 0., 3.]]), np.diag([-1., 2., 3.])])
def test_orthogonal_affine_physical_area_distance_and_reflection(basis):
    a = np.eye(4); a[:3, :3] = basis
    e = make_engine(affine=a); seed(e, {0: 10.}); advance(e, .4)
    # Face area6mm², center distance1mm, volume6mm³.
    expected = 5 * (1 - 1/(1 + 2*(CONDUCTIVITY*6)/(CAPACITY*6)*.4))
    assert e.state[heat.PROPERTY]['temperatureRiseK']['1'] == pytest.approx(expected)


def test_multiple_deposits_share_one_clock_and_preserve_total_input():
    e = make_engine(); heat.begin_step(e)
    heat.deposit(e, [0, 1], 1.); heat.deposit(e, [1], .5); heat.end_step(e, .1)
    assert e.state[heat.PROPERTY]['inputEnergyJ'] == pytest.approx(.003)
    assert e.state[heat.PROPERTY]['elapsedSeconds'] == .1
    with pytest.raises(ValueError, match='INTERVAL_REQUIRED'): heat.end_step(e, .1)


def test_coverage_source_mutation_and_temperature_domain_refuse_instead_of_clipping():
    e = make_engine(cells=[0]); heat.begin_step(e)
    with pytest.raises(ValueError, match='COVERAGE_UNKNOWN'): heat.deposit(e, [1], 1.)
    e.scene.properties[heat.PROPERTY]['heatWattsPerActivation'] = .03
    with pytest.raises(ValueError, match='Source scene changed'): heat.end_step(e, .1)
    e = make_engine(np.ones((1, 1, 1), np.uint16), maximum=34.)
    heat.begin_step(e); heat.deposit(e, [0], 1.)
    with pytest.raises(ValueError, match='TEMPERATURE_DOMAIN'): heat.end_step(e, 1.)


def test_sheared_grid_and_missing_parent_are_rejected():
    a = np.eye(4); a[0, 1] = .2
    with pytest.raises(ValueError, match='ORTHOGONAL_GRID'): make_engine(affine=a)
    e = make_engine(); props = copy.deepcopy(e.scene.properties); props.pop('tissuePhysicsV2')
    with pytest.raises(ValueError, match='REQUIRES_EXPLICIT_V2'):
        SurgicalScene(e.scene.labels, e.scene.affine, e.scene.regions, e.scene.source, props)
    props = copy.deepcopy(e.scene.properties); props[heat.PROPERTY] = None
    with pytest.raises(ValueError, match='COMPLETE_EXPLICIT_BIPOLAR_THERMAL_PROFILE'):
        SurgicalScene(e.scene.labels, e.scene.affine, e.scene.regions, e.scene.source, props)


def public_engine(maximum=80.):
    labels = np.zeros((8, 5, 5), np.uint16); labels[2:4, 1, 2] = 1
    initial = make_engine(labels, maximum=maximum)
    return SurgicalEngine(initial.scene)


def public_act(e, operation, point=None, **kwargs):
    action = {'slot': 'left', 'operation': operation, 'duration': .1, **kwargs}
    if point is not None: action['pose'] = {'position': point, 'quaternion': [0., 0., 0., 1.]}
    result = e.step(action)
    assert result['status'] == 'accepted', result
    return result


def place_bipolar(e):
    public_act(e, 'configure', mechanicsRevision=2)
    public_act(e, 'exchange', instrument='bipolar_forceps')
    public_act(e, 'place', [2.5, 1., -4.], parameters={'jawOpeningMm': .5, 'forceN': 0., 'power': 0., 'depthMm': 1.})
    public_act(e, 'apply', [2.5, 1., 1.8], duration=.5)


def test_public_electrode_contact_open_jaws_deactivation_and_canonical_restore():
    e = public_engine(); place_bipolar(e)
    assert e.metrics()['bipolarHeatInputJ'] == 0.
    public_act(e, 'activate', parameters={'power': 1.}, duration=.2)
    assert e.metrics()['bipolarHeatInputJ'] == pytest.approx(.004)
    own = e.clone(); public_act(own, 'inspect', duration=.1)
    result = e.step({'slot': 'right', 'operation': 'inspect', 'duration': .1})
    assert result['status'] == 'accepted'
    assert e.metrics()['bipolarHeatInputJ'] == pytest.approx(.006)
    assert heat.metrics(own) == heat.metrics(e)
    public_act(e, 'deactivate', duration=.1)
    before = e.metrics()['bipolarHeatInputJ']
    public_act(e, 'inspect', duration=1.)
    assert e.metrics()['bipolarHeatInputJ'] == before
    public_act(e, 'apply', parameters={'jawOpeningMm': 2., 'power': 1.}, duration=.2)
    assert e.metrics()['bipolarHeatInputJ'] == before
    restored = SurgicalEngine.restore(e.scene, e.snapshot())
    assert restored.state == e.state
    assert restored.view()['bipolarThermalV1'] == e.state[heat.PROPERTY]
    assert e.observe()['physics']['bipolarThermal']['model'] == heat.SCHEMA


def test_public_temperature_domain_refusal_has_exact_prefix_and_heat_budget():
    e = public_engine(maximum=33.1); place_bipolar(e); before = e.snapshot()
    result = e.step({'slot': 'left', 'operation': 'apply', 'duration': .05, 'parameters': {'power': 10.}})
    assert result['status'] == 'rejected' and 'TEMPERATURE_DOMAIN' in result['reason']
    assert e.snapshot() == before


def test_rl_refuses_heat_scene_without_device_qualification():
    from resectionlab.surgical_rl_rewards import RewardTracker
    with pytest.raises(ValueError, match='BIPOLAR_THERMAL_RL_HOLD'): RewardTracker(public_engine())


def test_export_keeps_physical_heat_separate_from_coagulation_proxy():
    from resectionlab.surgical_modeled_outcomes import summarize_engine_outcomes
    from resectionlab.surgical_telemetry import _ObservedEngine
    e = _ObservedEngine(public_engine().scene); place_bipolar(e); public_act(e, 'apply', parameters={'power': 1.}, duration=.1)
    result = summarize_engine_outcomes(e)
    assert result['thermal']['bipolarHeat']['bipolarHeatInputJ']['value'] == pytest.approx(.002)
    assert result['thermal']['bipolarHeat']['bipolarMaxTemperatureC']['unit'] == 'degC'
    assert result['thermal']['bipolarHeat']['coagulationAndTemperatureCoupled'] is False
    assert result['neurologicalRisk']['status'] == 'unknown'
    assert e._sample('action_end')['bipolarThermal']['bipolarHeatInputJ'] == pytest.approx(.002)


def test_indirect_heat_into_protected_source_is_rejected_atomically():
    from resectionlab.surgical_rl_protection import GuardConfig, ProtectionSet, ProtectionAnnotation
    labels = np.zeros((8, 5, 5), np.uint16); labels[2:4, 1:4, 2] = 1
    scene = make_engine(labels).scene; mask = np.zeros(scene.shape, bool); mask[2, 3, 2] = True
    protection = ProtectionSet(scene, [ProtectionAnnotation.from_mask(scene, 'brainstem', mask,
        provenance='Generated indirect heat guard control')], clearance_mm=0.)
    e = SurgicalEngine(scene, protection=protection, guard_config=GuardConfig()); place_bipolar(e)
    before = e.snapshot()
    result = e.step({'slot': 'left', 'operation': 'apply', 'duration': .1, 'parameters': {'power': 1.}})
    assert result['status'] == 'rejected' and result['reason'].startswith('PROTECTED_STRUCTURE: indirect thermal')
    assert e.snapshot() == before
