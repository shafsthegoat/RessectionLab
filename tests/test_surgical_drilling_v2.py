import copy
import json
import math
from types import MethodType, SimpleNamespace
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


def _legacy_contact_pose(engine, previous, requested, radius):
    """Frozen pre-reuse contact arithmetic and dynamic query dispatch."""
    previous = np.asarray(previous, dtype=float); requested = np.asarray(requested, dtype=float)
    def bone_at(point):
        labels = engine.scene.labels.ravel()
        return sorted(i for i in engine.touched(point, point, radius)
                      if labels[i] and engine.by_label[int(labels[i])]['material'] == 'bone')
    initial = bone_at(previous)
    if initial: return previous.copy(), initial
    distance = float(np.linalg.norm(requested - previous))
    steps = max(1, math.ceil(distance / (float(engine.scene.spacing.min()) / 4)))
    low = previous.copy()
    for index in range(1, steps + 1):
        high = previous + (requested - previous) * (index / steps)
        contacts = bone_at(high)
        if contacts:
            for _ in range(32):
                mid = (low + high) / 2
                if bone_at(mid): high = mid
                else: low = mid
            return high, bone_at(high)
        low = high
    return requested.copy(), []


def _contact_engine(cls=None):
    from resectionlab.surgical_engine import SurgicalEngine
    from resectionlab.surgical_scene import SurgicalScene
    labels = np.zeros((9, 3, 3), np.uint16)
    labels[4, 1, 1] = labels[6, 1, 1] = 1
    scene = SurgicalScene(labels, np.eye(4), [{'id': 1, 'name': 'bone', 'material': 'bone'}],
                          {'kind': 'generated-pure-contact-query-fixture'}, {})
    return (cls or SurgicalEngine)(scene)


def _contact_trace(implementation, actual, radius):
    """Observe original functions without replacing methods under admission."""
    import sys
    from resectionlab.surgical_engine import _DRILL_CONTACT_QUERY_ORIGINALS
    codes = {name: code for name, _, code, _, _ in _DRILL_CONTACT_QUERY_ORIGINALS[2]}
    queries, occupancy_calls = [], []
    def observe(frame, event, arg):
        if event == 'call' and frame.f_locals.get('self') is actual:
            if frame.f_code is codes['cells']:
                values = frame.f_locals
                queries.append((values['a'].tobytes(), values['b'].tobytes(),
                                np.asarray(values['radius']).dtype.str, np.asarray(values['radius']).tobytes()))
            elif frame.f_code is codes['occupancy']:
                occupancy_calls.append(None)
    previous = sys.getprofile()
    try:
        sys.setprofile(observe)
        result = implementation(actual, [0., 1., 1.], [8., 1., 1.], radius)
    finally:
        sys.setprofile(previous)
    return result, queries, len(occupancy_calls)


@pytest.mark.parametrize('radius', [.1, np.float32(.1), np.float64(.5)])
def test_stock_contact_reuses_only_local_occupancy_with_exact_query_order(radius):
    actual, reference = _contact_engine(), _contact_engine()
    for removed in ([], [40], [40, 58], []):
        # In-place edits between contact searches must invalidate occupancy.
        actual.state['removed'][:] = reference.state['removed'][:] = removed
        before = copy.deepcopy(actual.state)
        result, queries, calls = _contact_trace(drill.contact_limited_pose, actual, radius)
        expected, original_queries, original_calls = _contact_trace(_legacy_contact_pose, reference, radius)
        assert np.array_equal(result[0], expected[0]) and result[1] == expected[1]
        assert queries == original_queries and len(queries) > 1
        assert calls == 1 and original_calls == len(queries)
        assert actual.state == reference.state == before


@pytest.mark.parametrize('name', ['touched', 'cells', 'occupancy'])
@pytest.mark.parametrize('placement', ['instance', 'subclass', 'class'])
def test_contact_preserves_overrides_that_mutate_state_between_queries(monkeypatch, name, placement):
    from resectionlab.surgical_engine import SurgicalEngine
    original = getattr(SurgicalEngine, name)
    def custom(self, *args):
        self._contact_query_calls = getattr(self, '_contact_query_calls', 0) + 1
        if self._contact_query_calls == 3:
            self.state['removed'][:] = [40, 58]
        return original(self, *args)
    cls = SurgicalEngine
    if placement == 'class':
        monkeypatch.setattr(cls, name, custom)
    elif placement == 'subclass':
        cls = type('CustomContactEngine', (SurgicalEngine,), {name: custom})
    actual, reference = _contact_engine(cls), _contact_engine(cls)
    if placement == 'instance':
        for current in (actual, reference):
            setattr(current, name, MethodType(custom, current))
    assert not drill._stock_contact_queries(actual)
    result = drill.contact_limited_pose(actual, [0., 1., 1.], [8., 1., 1.], .1)
    expected = _legacy_contact_pose(reference, [0., 1., 1.], [8., 1., 1.], .1)
    assert np.array_equal(result[0], expected[0]) and result[1] == expected[1]
    assert actual._contact_query_calls == reference._contact_query_calls > 3
    assert actual.state == reference.state and result[1] == []


def test_contact_admission_does_not_invoke_replacement_descriptor(monkeypatch):
    from resectionlab.surgical_engine import SurgicalEngine
    original = SurgicalEngine.touched
    def touched(self):
        self._contact_descriptor_calls = getattr(self, '_contact_descriptor_calls', 0) + 1
        return MethodType(original, self)
    monkeypatch.setattr(SurgicalEngine, 'touched', property(touched))
    actual, reference = _contact_engine(), _contact_engine()
    assert not drill._stock_contact_queries(actual)
    assert not hasattr(actual, '_contact_descriptor_calls')
    result = drill.contact_limited_pose(actual, [0., 1., 1.], [8., 1., 1.], .1)
    expected = _legacy_contact_pose(reference, [0., 1., 1.], [8., 1., 1.], .1)
    assert np.array_equal(result[0], expected[0]) and result[1] == expected[1]
    assert actual._contact_descriptor_calls == reference._contact_descriptor_calls


def test_contact_admission_rejects_getattribute_and_function_body_patches(monkeypatch):
    from resectionlab.surgical_engine import SurgicalEngine
    actual = _contact_engine()
    with monkeypatch.context() as patch:
        patch.setattr(SurgicalEngine, '__getattribute__', lambda self, name: object.__getattribute__(self, name))
        assert not drill._stock_contact_queries(actual)
    with monkeypatch.context() as patch:
        def replacement(self, a, b, radius): return []
        patch.setattr(SurgicalEngine.touched, '__code__', replacement.__code__)
        assert not drill._stock_contact_queries(actual)
        result = drill.contact_limited_pose(actual, [0., 1., 1.], [8., 1., 1.], .1)
        assert result[0].tolist() == [8., 1., 1.] and result[1] == []
    assert drill._stock_contact_queries(actual)


def test_contact_invalid_geometry_precedes_occupancy_validation():
    actual = _contact_engine()
    actual.state['holds'] = {'invalid': {}}
    with pytest.raises(ValueError, match='radius_mm'):
        drill.contact_limited_pose(actual, [0., 1., 1.], [8., 1., 1.], -1.)
    assert actual._occupancy_cache is None


@pytest.mark.parametrize('dependency', ['kernel', 'cache-query', 'instance-query', 'candidate-helper', 'material-ids'])
def test_contact_replacement_dependencies_mutating_state_keep_dynamic_queries(monkeypatch, dependency):
    from resectionlab import geometry, surgical_engine, surgical_geometry_cache, surgical_occupancy
    active = [None, 0]
    def changed():
        active[1] += 1
        if active[1] == 3:
            active[0].state['removed'][:] = [40, 58]
    if dependency == 'kernel':
        original = surgical_engine.capsule_voxel_indices
        def replacement(*args):
            changed()
            return original(*args)
        monkeypatch.setattr(surgical_engine, 'capsule_voxel_indices', replacement)
    elif dependency == 'candidate-helper':
        original = geometry._capsule_candidate_grid
        def replacement(*args):
            changed()
            return original(*args)
        monkeypatch.setattr(geometry, '_capsule_candidate_grid', replacement)
    else:
        cls = (surgical_occupancy.MaterialOccupancy if dependency == 'material-ids'
               else surgical_geometry_cache.SurgicalCellCoverCache)
        name = 'material_ids' if dependency == 'material-ids' else 'query'
        original = getattr(cls, name)
        def replacement(self, *args, **kwargs):
            changed()
            return original(self, *args, **kwargs)
        if dependency != 'instance-query':
            monkeypatch.setattr(cls, name, replacement)
    actual, reference = _contact_engine(), _contact_engine()
    if dependency == 'instance-query':
        for current in (actual, reference):
            current._cell_cover_cache.query = MethodType(replacement, current._cell_cover_cache)
    assert not drill._stock_contact_queries(actual)
    results = []
    for current, implementation in ((actual, drill.contact_limited_pose), (reference, _legacy_contact_pose)):
        active[:] = [current, 0]
        result = implementation(current, [0., 1., 1.], [8., 1., 1.], .1)
        results.append((result, active[1]))
    assert np.array_equal(results[0][0][0], results[1][0][0])
    assert results[0][0][1] == results[1][0][1] == []
    assert results[0][1] == results[1][1] >= 3 and actual.state == reference.state


@pytest.mark.parametrize('dependency', ['kernel', 'cache-query', 'material-ids'])
def test_contact_dependency_body_changes_in_place_do_not_admit_stock_identity(monkeypatch, dependency):
    from resectionlab import surgical_engine, surgical_geometry_cache, surgical_occupancy
    actual = _contact_engine()
    function = {'kernel': surgical_engine.capsule_voxel_indices,
                'cache-query': surgical_geometry_cache.SurgicalCellCoverCache.query,
                'material-ids': surgical_occupancy.MaterialOccupancy.material_ids}[dependency]
    def replacement(*args, **kwargs): return []
    monkeypatch.setattr(function, '__code__', replacement.__code__)
    assert not drill._stock_contact_queries(actual)


def test_contact_mutated_keyword_defaults_and_custom_query_owners_fall_back(monkeypatch):
    from resectionlab import geometry
    actual = _contact_engine()
    with monkeypatch.context() as patch:
        patch.setitem(geometry.capsule_voxel_indices.__kwdefaults__, '_candidate_grid', lambda *args: None)
        assert not drill._stock_contact_queries(actual)
    with monkeypatch.context() as patch:
        patch.setattr(actual, '_cell_cover_cache', SimpleNamespace(query=actual._cell_cover_cache.query))
        assert not drill._stock_contact_queries(actual)
    with monkeypatch.context() as patch:
        patch.setitem(actual.grid.__dict__, 'world_to_voxel', lambda value: value)
        assert not drill._stock_contact_queries(actual)
    assert drill._stock_contact_queries(actual)


def test_contact_custom_occupancy_instance_keeps_per_query_revalidation(monkeypatch):
    actual, reference = _contact_engine(), _contact_engine()
    for current in (actual, reference):
        owner = current.occupancy()
        original = owner.material_ids
        def replacement(cells, *, current=current, original=original):
            current._owner_calls = getattr(current, '_owner_calls', 0) + 1
            if current._owner_calls == 3:
                current.state['removed'][:] = [40, 58]
            return original(cells)
        monkeypatch.setattr(owner, 'material_ids', replacement)
    result = drill.contact_limited_pose(actual, [0., 1., 1.], [8., 1., 1.], .1)
    expected = _legacy_contact_pose(reference, [0., 1., 1.], [8., 1., 1.], .1)
    assert np.array_equal(result[0], expected[0]) and result[1] == expected[1]
    assert actual._owner_calls == reference._owner_calls == 3
    assert actual.state == reference.state


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


def test_empty_thermal_is_a_python_noop_without_dose_or_cooling_reads(monkeypatch):
    def unused_native(*args):
        raise AssertionError('Empty thermal loop must not invoke a kernel')
    monkeypatch.setattr(drill._idle_native, 'try_idle', unused_native)
    for ledger in ({'temperatureC': {}},
                   {'temperatureC': {}, 'thermalDoseCSeconds': {'2': -0.},
                    'coolingLossJ': -0.}):
        actual = engine()
        actual.state['drillingV2'] = ledger
        expected = copy.deepcopy(actual)
        temperature_object = ledger['temperatureC']
        _legacy_thermal(expected, .1, {})
        drill._thermal(actual, .1, {})
        assert json.dumps(actual.state) == json.dumps(expected.state)
        assert ledger['temperatureC'] is temperature_object


def test_empty_thermal_keeps_scalar_errors_and_nonempty_custom_paths(monkeypatch):
    calls = []
    class NativeReached(Exception):
        pass
    def native(*args):
        calls.append(args)
        raise NativeReached
    monkeypatch.setattr(drill._idle_native, 'try_idle', native)
    actual = engine()
    actual.state['drillingV2'] = {'temperatureC': {}}
    actual.scene.properties['tissuePhysicsV2']['bone']['coolingTimeSeconds'] = 0.
    with pytest.raises(ZeroDivisionError):
        drill._thermal(actual, .1, {})
    assert calls == []
    class Mapping(dict):
        pass
    for temperatures, powers in (({'0': 38.}, {}), ({}, {'0': .1}),
                                  (Mapping(), {}), ({}, Mapping())):
        actual = engine()
        actual.state['drillingV2'] = {'temperatureC': temperatures}
        with pytest.raises(NativeReached):
            drill._thermal(actual, .1, powers)
    lookups = []
    class CustomKey:
        def __hash__(self):
            return hash('temperatureC')
        def __eq__(self, other):
            lookups.append(other)
            return other == 'temperatureC'
    actual = engine()
    actual.state['drillingV2'] = {CustomKey(): {}}
    with pytest.raises(NativeReached):
        drill._thermal(actual, .1, {})
    assert lookups == []  # Extra lookups cannot run before the original hook.
    assert len(calls) == 5


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
