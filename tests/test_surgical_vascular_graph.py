"""Independent RC oracles and adversarial source/fluid handoff controls."""
import copy
import math
from types import SimpleNamespace

import numpy as np
import pytest

from resectionlab import surgical_vascular_graph as graph
from resectionlab.surgical_effects import tick_fields
from resectionlab.surgical_scene import SurgicalScene
from resectionlab.surgical_physics_v2 import default_profile

SOURCE = 'Generated numerical coefficient; not physiological calibration'


def profile(shape=(8, 5, 5)):
    def cell(point): return int(np.ravel_multi_index(point, shape))
    def edge(name, a, b, kind, points):
        cells = [cell(point) for point in points]
        return {'id': name, 'from': a, 'to': b, 'kind': kind, 'conductanceMm3PerPaS': .0001,
            'sourceCells': cells, 'sourceFace': cells[:], 'woundConductanceMm3PerPaS': [.0001, .0001], 'source': SOURCE}
    return {'schema': graph.SCHEMA, 'provenance': {'kind': 'assumed-research', 'description': SOURCE},
        'coverage': {'sourceCaseId': 'generated-rc-control', 'status': 'generated-control',
            'missing': ['No acquired patient or neurological outcomes']}, 'fieldPressurePa': 0.,
        'compressionForceN': 1., 'coagulationDose': 2., 'clotRatePerSecond': 0.,
        'nodes': [{'id': name, 'complianceMm3PerPa': .001, 'initialStorageMm3': volume, 'source': SOURCE}
            for name, volume in [('a', 7.5), ('b', 5.), ('c', 2.5)]],
        'edges': [edge('artery', 'a', 'b', 'artery', [(2,1,2),(3,1,2)]),
                  edge('vein', 'b', 'c', 'vein', [(2,3,2),(3,3,2)])],
        'boundaries': [{'id': name, 'node': node, 'pressurePa': pressure, 'conductanceMm3PerPaS': .0001, 'source': SOURCE}
            for name, node, pressure in [('inlet','a',10000.),('outlet','c',0.)]],
        'regions': [{'id': 'declared-region', 'supply': [{'edge':'artery','direction':1}],
            'drainage': [{'edge':'vein','direction':1}], 'storageNodes': ['b'],
            'referenceSupplyMm3PerSecond': .25, 'referenceDrainageMm3PerSecond': .25, 'source': SOURCE}]}


def engine(p=None, affine=None, extra_vessel=False):
    p = profile() if p is None else p
    labels = np.zeros((8,5,5), np.uint16)
    for i, edge in enumerate(p['edges']): labels.ravel()[edge['sourceCells']] = i+1
    if extra_vessel: labels[4,1,2] = 1
    regions = [{'id': i+1, 'name': edge['kind'], 'material': 'vessel'} for i, edge in enumerate(p['edges'])]
    scene = SurgicalScene(labels, np.eye(4) if affine is None else affine, regions, {'kind': 'generated-control'},
        {'tissuePhysicsV2': default_profile(), graph.PROPERTY: p})
    state = {'sources': [], 'fluid': {}, 'fluidAddedMm3': 0., 'fluidEscapedMm3': 0., 'debris': {}}
    e = SimpleNamespace(scene=scene, state=state, by_label={r['id']:r for r in regions}, occupancy=lambda:labels.ravel())
    graph.initialize(e)
    return e


def advance(e, dt):
    graph.tick(e, dt); tick_fields(e, dt)


def cut(e, name):
    edge = next(r for r in e.scene.properties[graph.PROPERTY]['edges'] if r['id'] == name)
    graph.damage(e, [edge['sourceFace']]); return edge


def test_single_node_rc_matches_independent_storage_supply_and_wound_integrals():
    p = profile(); p['regions'] = []; p['edges'] = p['edges'][:1]; p['nodes'] = p['nodes'][:2]
    p['nodes'][0]['initialStorageMm3'] = 2.; p['nodes'][1]['initialStorageMm3'] = 0.
    p['boundaries'] = p['boundaries'][:1]; p['edges'][0]['woundConductanceMm3PerPaS'][1] = 0.
    e = engine(p); cut(e, 'artery'); t = 3.7; advance(e, t)
    c = .001; gs = gw = .0001; ps = 10000.; p0 = 2000.; pinf = gs*ps/(gs+gw); tau = c/(gs+gw)
    pend = pinf+(p0-pinf)*math.exp(-t/tau)
    pressure_integral = pinf*t+(p0-pinf)*tau*(1-math.exp(-t/tau))
    m = graph.metrics(e)
    assert e.state[graph.PROPERTY]['storageMm3']['a'] == pytest.approx(c*pend, rel=1e-9)
    assert m['generatedMm3'] == pytest.approx(gw*pressure_integral, rel=1e-9)
    assert m['boundaryInMm3'] == pytest.approx(gs*(ps*t-pressure_integral), rel=1e-9)
    assert m['boundaryOutMm3'] == 0.
    assert abs(m['conservationResidualMm3']) < 1e-10


def test_closed_two_node_exchange_pressure_difference_and_total_storage():
    p = profile(); p['nodes'] = p['nodes'][:2]; p['nodes'][0]['initialStorageMm3'] = 2.
    p['nodes'][1].update(initialStorageMm3=.5, complianceMm3PerPa=.002)
    p['edges'] = p['edges'][:1]; p['boundaries'] = []; p['regions'] = []
    e = engine(p); t = 12.; advance(e, t)
    v = e.state[graph.PROPERTY]['storageMm3']; delta = v['a']/.001-v['b']/.002
    assert sum(v.values()) == pytest.approx(2.5, abs=1e-11)
    assert delta == pytest.approx((2000.-250.)*math.exp(-.0001*(1/.001+1/.002)*t), rel=2e-9)
    assert graph.metrics(e)['generatedMm3'] == 0.


def test_boundary_reverses_within_interval_and_in_out_ledgers_match_analytic_oracle():
    p = profile(); p['regions'] = []; p['edges'] = p['edges'][:1]; p['nodes'] = p['nodes'][:2]
    p['nodes'][0]['initialStorageMm3'] = 2.; p['nodes'][1]['initialStorageMm3'] = 0.
    p['boundaries'] = p['boundaries'][:1]; p['boundaries'][0]['pressurePa'] = 1000.
    p['edges'][0]['woundConductanceMm3PerPaS'][1] = 0.
    e = engine(p); cut(e,'artery'); t = 12.; advance(e,t)
    pressure_integral = 500*t+7500*(1-math.exp(-.2*t))
    out = .5-.25*math.log(3)
    signed_in = .0001*(1000*t-pressure_integral)
    m = graph.metrics(e)
    assert m['boundaryOutMm3'] == pytest.approx(out, rel=2e-8, abs=1e-10)
    assert m['boundaryInMm3'] == pytest.approx(signed_in+out, rel=2e-8, abs=1e-10)
    assert m['generatedMm3'] == pytest.approx(.0001*pressure_integral, rel=2e-9)


def test_independent_natural_clot_oracle_for_isolated_two_stumps():
    p = profile(); p['regions'] = []; p['edges'] = p['edges'][:1]; p['nodes'] = p['nodes'][:2]
    p['boundaries'] = []; p['clotRatePerSecond'] = .2
    e = engine(p); cut(e,'artery'); t = 8.; advance(e,t)
    for node in p['nodes']:
        expected = node['initialStorageMm3']*math.exp(-.0001/(.001*.2)*(1-math.exp(-.2*t)))
        assert e.state[graph.PROPERTY]['storageMm3'][node['id']] == pytest.approx(expected, rel=2e-9)


def test_two_cut_ports_drain_separate_storage_and_repeated_cut_is_idempotent():
    p = profile(); p['nodes'] = p['nodes'][:2]; p['edges'] = p['edges'][:1]; p['boundaries'] = []; p['regions'] = []
    e = engine(p); edge = cut(e, 'artery'); graph.damage(e, [edge['sourceFace']]); advance(e, 4.)
    assert len(e.state['sources']) == 2
    for source, volume in zip(e.state['sources'], (7.5,5.)):
        assert source['emittedMm3'] == pytest.approx(volume*(1-math.exp(-.4)), rel=2e-9)
    m = graph.metrics(e)
    assert m['storageMm3']+m['generatedMm3'] == pytest.approx(12.5, abs=1e-10)


def test_intact_arterial_occlusion_has_no_bleeding_but_loses_declared_delivery():
    e = engine(); cells = e.scene.properties[graph.PROPERTY]['edges'][0]['sourceCells']
    graph.coagulate(e, cells, 1., 2.); advance(e, 20.)
    m = graph.metrics(e); region = m['regions']['declared-region']
    assert m['generatedMm3'] == 0. and m['occludedEdgeCount'] == 1 and m['woundPortCount'] == 0
    assert region['supplyDeficitMm3'] == pytest.approx(5., abs=1e-9)
    assert region['supplyDeficitSeconds'] == pytest.approx(20., abs=1e-9)
    assert region['storageChangeMm3'] < 0.


def test_venous_occlusion_raises_upstream_storage_and_separates_drainage_signal():
    e = engine(); cells = e.scene.properties[graph.PROPERTY]['edges'][1]['sourceCells']
    graph.coagulate(e, cells, 1., 2.); advance(e, 20.)
    m = graph.metrics(e); r = m['regions']['declared-region']
    assert m['generatedMm3'] == 0. and r['storageChangeMm3'] > 0
    assert r['drainageDeficitSeconds'] == pytest.approx(20., abs=1e-9)
    assert r['supplyDeficitSeconds'] < r['drainageDeficitSeconds']


def test_sealing_both_ports_stops_bleed_but_does_not_reconnect_cut_edge():
    e = engine(); edge = cut(e, 'artery'); advance(e, 1.)
    before = graph.metrics(e)['generatedMm3']; graph.coagulate(e, edge['sourceCells'], 1., 2.); advance(e, 4.)
    m = graph.metrics(e)
    assert m['generatedMm3'] == before and m['cutEdgeCount'] == 1
    assert e.state[graph.PROPERTY]['edgeFlowMm3PerSecond']['artery'] == 0.
    assert m['regions']['declared-region']['supplyDeficitSeconds'] == pytest.approx(5.)


def test_only_contacted_stump_seals_and_distal_backbleeding_is_retained():
    e = engine(); edge = cut(e, 'artery'); graph.coagulate(e, edge['sourceFace'][:1], 1., 2.); advance(e, 2.)
    assert e.state['sources'][0]['emittedMm3'] == 0.
    assert e.state['sources'][1]['emittedMm3'] > 0.


def test_clotting_partition_sensitivity_and_field_conservation():
    p = profile(); p['clotRatePerSecond'] = .2
    one = engine(p); many = engine(p)
    for e in (one, many): cut(e, 'artery')
    advance(one, 5.)
    for _ in range(50): advance(many, .1)
    for key in ('storageMm3', 'boundaryInMm3', 'boundaryOutMm3', 'generatedMm3'):
        assert graph.metrics(one)[key] == pytest.approx(graph.metrics(many)[key], rel=3e-8, abs=1e-9)
    for e in (one, many):
        m = graph.metrics(e)
        assert m['generatedMm3'] == pytest.approx(m['emittedMm3'], abs=1e-10)
        assert abs(m['conservationResidualMm3']) < 1e-9
        assert m['emittedMm3'] == pytest.approx(sum(e.state['fluid'].values())+e.state['fluidEscapedMm3']+m['blockedOutletReservoirMm3'])


def test_field_handoff_is_exactly_once_and_wrong_duration_is_atomic():
    e = engine(); cut(e, 'artery'); graph.tick(e, .2); before = copy.deepcopy(e.state)
    for operation in (lambda:graph.tick(e,.2), lambda:tick_fields(e,.1)):
        with pytest.raises(ValueError, match='MATCHING_FIELD'): operation()
        assert e.state == before
    tick_fields(e,.2); before = copy.deepcopy(e.state)
    with pytest.raises(ValueError, match='MATCHING_FIELD'): tick_fields(e,.2)
    assert e.state == before


def test_blocked_outlet_then_open_does_not_double_count_or_lose_blood():
    e = engine(); cut(e, 'artery'); e.occupancy = lambda:np.ones(e.scene.labels.size,np.uint16)
    advance(e, 1.); m = graph.metrics(e)
    assert m['blockedOutletReservoirMm3'] == m['generatedMm3'] > 0
    e.occupancy = lambda:np.zeros(e.scene.labels.size,np.uint16)
    advance(e, 1.); m = graph.metrics(e)
    assert m['blockedOutletReservoirMm3'] == 0
    assert m['emittedMm3'] == pytest.approx(sum(e.state['fluid'].values())+e.state['fluidEscapedMm3'])


def test_unknown_cut_refused_before_mutation_and_unmapped_contact_stays_unknown():
    e = engine(extra_vessel=True)
    unknown = int(np.ravel_multi_index((4,1,2),e.scene.shape))
    neighbor = int(np.ravel_multi_index((3,1,2),e.scene.shape))
    graph.contact(e,[unknown],1.,1.)
    assert graph.metrics(e)['unmappedContactCells'] == 1
    assert graph.descriptor(e)['boundSourceCells'] < graph.descriptor(e)['vesselSourceCells']
    before = copy.deepcopy(e.state)
    with pytest.raises(ValueError, match='UNMAPPED_CUT'): graph.damage(e, [[neighbor,unknown]])
    assert e.state == before


@pytest.mark.parametrize('change', [lambda p:p['boundaries'][0].update(pressurePa=-1.),
    lambda p:p['nodes'][0].update(complianceMm3PerPa=0.), lambda p:p.update(fieldPressurePa=1.),
    lambda p:p['edges'][0].update(sourceFace=[1,3]), lambda p:p['edges'][0].update(kind='capillary-measured'),
    lambda p:p['regions'][0].update(supply=[]), lambda p:p['coverage'].update(status='complete'),
    lambda p:p['nodes'][0].update(initialStorageMm3=float('nan')),
    lambda p:p['regions'][0]['supply'][0].update(direction=True)])
def test_missing_or_unqualified_assumptions_refused(change):
    p = profile(); change(p)
    with pytest.raises(ValueError): engine(p)


def test_oblique_reflected_affine_changes_no_source_identity_or_declared_hemodynamics():
    baseline = engine(); reflected = engine(affine=np.array([[-1.,.2,0.,4.],[0.,1.,.1,-3.],[0.,0.,.7,9.],[0.,0.,0.,1.]]))
    for e in (baseline, reflected): cut(e,'artery'); advance(e,1.)
    assert baseline.state[graph.PROPERTY] == reflected.state[graph.PROPERTY]
    assert [r['cell'] for r in baseline.state['sources']] == [r['cell'] for r in reflected.state['sources']]


def test_unversioned_source_edit_refused_and_zero_duration_has_no_state_effect():
    e = engine(); before = copy.deepcopy(e.state); graph.tick(e,0.); tick_fields(e,0.); assert e.state == before
    e.scene.properties[graph.PROPERTY]['nodes'][0]['initialStorageMm3'] = 12.
    with pytest.raises(ValueError, match='Source scene changed'): graph.tick(e, .1)


@pytest.mark.parametrize('mutation', [lambda rows:rows.append(copy.deepcopy(rows[0])),
    lambda rows:rows[0].update(cell=0), lambda rows:rows[1].pop('reservoirMm3'),
    lambda rows:rows[0].update(emittedMm3=.1), lambda rows:rows[0].update(reservoirMm3=float('nan')),
    lambda rows:rows[0].update(modelVersion=2), lambda rows:rows[0].update(kind='declared-source')])
def test_source_corruption_refused_before_any_field_mutation(mutation):
    e = engine(); cut(e,'artery'); graph.tick(e,.2); mutation(e.state['sources']); before = copy.deepcopy(e.state)
    with pytest.raises(ValueError): tick_fields(e,.2)
    # NaN is not equal to itself; canonical payload comparison preserves it.
    import json
    assert json.dumps(e.state, sort_keys=True) == json.dumps(before, sort_keys=True)


def test_alternate_venous_drain_carries_reversed_signed_flow_and_limits_storage_rise():
    p = profile(); p['edges'].append({'id': 'collateral', 'from': 'c', 'to': 'b', 'kind': 'assumed-link',
        'conductanceMm3PerPaS': .0001, 'sourceCells': [], 'sourceFace': [],
        'woundConductanceMm3PerPaS': [0.,0.], 'source': SOURCE})
    p['regions'][0]['drainage'].append({'edge':'collateral','direction':-1})
    collateral = engine(p); sole = engine()
    for e in (collateral, sole):
        vein = e.scene.properties[graph.PROPERTY]['edges'][1]
        graph.coagulate(e, vein['sourceCells'], 1., 2.); advance(e, 20.)
    assert collateral.state[graph.PROPERTY]['edgeFlowMm3PerSecond']['collateral'] < 0
    assert graph.metrics(collateral)['regions']['declared-region']['storageChangeMm3'] < graph.metrics(sole)['regions']['declared-region']['storageChangeMm3']
    assert graph.metrics(collateral)['regions']['declared-region']['drainageDeficitSeconds'] < 1e-7


def test_public_bipolar_geometry_compression_release_occlusion_export_and_replay():
    from resectionlab.surgical_engine import SurgicalEngine
    from resectionlab.surgical_modeled_outcomes import summarize_engine_outcomes
    from resectionlab.surgical_rl_rewards import RewardTracker
    e = SurgicalEngine(engine().scene)
    def act(operation, point=None, **extra):
        action = {'slot':'left', 'operation':operation, 'duration':.1, **extra}
        if point is not None: action['pose'] = {'position':point,'quaternion':[0.,0.,0.,1.]}
        result = e.step(action)
        assert result['status'] == 'accepted', result
    act('configure', mechanicsRevision=2); act('exchange', instrument='bipolar_forceps')
    act('place', [2.5,1.,-4.], parameters={'jawOpeningMm':.5,'forceN':1.,'power':0.,'depthMm':1.})
    act('apply', [2.5,1.,1.8], duration=.5)
    act('deactivate'); act('inspect', duration=.3)
    row = e.state[graph.PROPERTY]['edges']['artery']
    assert row['compressionFraction'] == 1. and row['occlusionFraction'] == 0.
    assert graph.metrics(e)['generatedMm3'] == 0. and sum(e.state['thermal'].values()) == 0.
    act('apply', parameters={'jawOpeningMm':2.})
    assert e.state[graph.PROPERTY]['edges']['artery']['compressionFraction'] == 0.
    act('apply', parameters={'jawOpeningMm':.5, 'forceN':0., 'power':0.})
    act('apply', parameters={'power':1.}, duration=2.1)
    assert e.state[graph.PROPERTY]['edges']['artery']['occlusionFraction'] == 1.
    act('withdraw')
    restored = SurgicalEngine.restore(e.scene,e.snapshot())
    assert restored.state == e.state
    outcome = summarize_engine_outcomes(e)
    assert outcome['localVessels']['status'] == 'declared_partial_graph_model_only'
    assert outcome['globalPerfusion']['status'] == 'unknown'
    with pytest.raises(ValueError,match='RL_HOLD'): RewardTracker(e)


def test_public_unpowered_stump_compression_controls_bleed_only_while_held():
    from resectionlab.surgical_engine import SurgicalEngine
    e = SurgicalEngine(engine().scene)
    def act(operation, point=None, quaternion=None, **extra):
        action = {'slot':'left','operation':operation,'duration':.1,**extra}
        if point is not None: action['pose'] = {'position':point,'quaternion':quaternion or [0.,0.,0.,1.]}
        result = e.step(action); assert result['status'] == 'accepted',result
    act('configure',mechanicsRevision=2); act('exchange',instrument='micro_scissors')
    q = [0.,0.,2**-.5,2**-.5]
    act('place',[2.5,1.,-4.],quaternion=q,parameters={'jawOpeningMm':0.,'depthMm':1.,'forceN':0.})
    act('apply',[2.5,1.,1.8],quaternion=q,duration=.5); act('deactivate'); act('withdraw')
    assert graph.metrics(e)['cutEdgeCount'] == 1
    act('exchange',instrument='bipolar_forceps')
    act('place',[2.5,1.,-4.],parameters={'jawOpeningMm':.5,'forceN':1.,'power':0.,'depthMm':1.})
    act('apply',[2.5,1.,1.8],duration=.5); act('deactivate')
    before = graph.metrics(e)['generatedMm3']; act('inspect',duration=1.)
    assert graph.metrics(e)['generatedMm3'] == before
    assert all(port['compressionFraction'] == 1. for port in e.state[graph.PROPERTY]['ports'].values())
    act('apply',parameters={'jawOpeningMm':2.},duration=.2)
    assert graph.metrics(e)['generatedMm3'] > before
    assert graph.metrics(e)['cutEdgeCount'] == 1
    assert all(port['sealFraction'] == 0. for port in e.state[graph.PROPERTY]['ports'].values())


def test_graph_telemetry_reports_ports_and_emitted_volume():
    from resectionlab.surgical_telemetry import _ObservedEngine
    e = _ObservedEngine(engine().scene); cut(e,'artery')
    result = e.step({'slot':'left','operation':'inspect','duration':.2})
    assert result['status'] == 'accepted'
    sample = e._sample('action_end')
    assert sample['woundCount'] == 2 and sample['activeWoundCount'] == 2
    assert sample['modeledWoundEmittedMm3'] == pytest.approx(graph.metrics(e)['generatedMm3'])


def test_indirect_occlusion_of_protected_segment_is_rejected_atomically():
    from resectionlab.surgical_engine import SurgicalEngine
    from resectionlab.surgical_rl_protection import GuardConfig,ProtectionSet,ProtectionAnnotation
    p = profile(); protected_cell = int(np.ravel_multi_index((4,1,2),(8,5,5)))
    p['edges'][0]['sourceCells'].append(protected_cell)
    scene = engine(p).scene; mask = np.zeros(scene.shape,bool); mask.flat[protected_cell] = True
    protection = ProtectionSet(scene,[ProtectionAnnotation.from_mask(scene,'brainstem',mask,
        provenance='Generated indirect conductance guard control')],clearance_mm=0.)
    e = SurgicalEngine(scene,protection=protection,guard_config=GuardConfig())
    for action in [
        {'operation':'configure','mechanicsRevision':2}, {'operation':'exchange','instrument':'bipolar_forceps'},
        {'operation':'place','pose':{'position':[2.,1.,-4.],'quaternion':[0.,0.,0.,1.]},
            'parameters':{'jawOpeningMm':0.,'forceN':0.,'power':0.,'depthMm':1.}},
        {'operation':'apply','pose':{'position':[2.,1.,1.8],'quaternion':[0.,0.,0.,1.]},'duration':.5}]:
        result = e.step({'slot':'left','duration':.1,**action}); assert result['status'] == 'accepted',result
    before = e.snapshot()
    result = e.step({'slot':'left','operation':'apply','duration':.1,'parameters':{'power':1.}})
    assert result['status'] == 'rejected' and result['reason'].startswith('PROTECTED_STRUCTURE: indirect vascular')
    assert e.snapshot() == before
