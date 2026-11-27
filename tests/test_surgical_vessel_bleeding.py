"""Cut-vessel bleeding: Murray-scaled arterial pulses, venous welling, tumour ooze and bipolar limits.

Synthetic grids only; the checks are numerical oracles for declared research laws,
not physiological validation.
"""
import copy
import math
from types import SimpleNamespace

import numpy as np
import pytest

from resectionlab import surgical_effects
from resectionlab import surgical_vessel_bleeding as vb
from resectionlab.surgical_engine import SurgicalEngine
from resectionlab.surgical_physics_v2 import default_profile
from resectionlab.surgical_scene import SurgicalScene

SHAPE = (20, 20, 16)
CUT = [8., 8., 7.]


def scene(name='Cortical artery', width=1, *, tumour=False, bleeding=True, extra=None):
    labels = np.zeros(SHAPE, np.uint16)
    lo = 8 - width // 2
    labels[lo:lo + width, 3:17, lo:lo + width] = 1
    regions = [{'id': 1, 'name': name, 'material': 'vessel'}]
    if tumour:
        labels[2:6, 4:12, 4:10] = 2
        regions.append({'id': 2, 'name': 'Tumour', 'material': 'tumor', 'target': True})
    properties = {'tissuePhysicsV2': default_profile()}
    if bleeding:
        properties[vb.PROPERTY] = vb.declare(labels, regions, np.eye(4))
    properties.update(extra or {})
    return SurgicalScene(labels, np.eye(4), regions, {}, properties)


def act(e, operation, position=None, **extra):
    action = {'slot': 'left', 'operation': operation, 'duration': extra.pop('duration', .2),
              **({'pose': {'position': position, 'quaternion': [0., 0., 0., 1.]}} if position else {}), **extra}
    result = e.step(action)
    assert result['status'] == 'accepted', result
    return result


def tool(e, name, at=(8., 8., 4.)):
    if e.state['slots']['left']['inserted']:
        act(e, 'withdraw')
    act(e, 'exchange', instrument=name)
    act(e, 'place', list(at), parameters={'radiusMm': 1.5, 'jawOpeningMm': 0.})


def position(e, declared):
    result = e.step({'slot': 'left', 'operation': 'position', 'duration': .1, 'position': declared})
    assert result['status'] == 'accepted', result


def cut(name='Cortical artery', width=1, **kwargs):
    e = SurgicalEngine(scene(name, width, **kwargs))
    tool(e, 'micro_scissors'); act(e, 'apply', CUT, duration=1.)
    return e


def wound(e):
    wounds = e.state[vb.PROPERTY]['wounds']
    assert len(wounds) == 1
    return next(iter(wounds.values()))


def murray(calibre, kind='artery', head=1.):
    # 717 mL/min total CBF x 0.21 MCA share, mL/min -> mm3/s, Q ~ d^3 from a 2.8 mm M1.
    reference = 717. * .21 * 1000. / 60.
    flow = reference * (calibre / 2.8) ** 3
    return flow if kind == 'artery' else flow * 10. / 90. * head


def test_scene_without_property_never_calls_the_model(monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError('vesselBleedingV1 ran on a scene that did not opt in')
    for name in ('tick', 'coagulate', 'initialize'):
        monkeypatch.setattr(vb, name, forbidden)
    e = SurgicalEngine(scene(bleeding=False, tumour=True)); first = e.state_id
    tool(e, 'micro_scissors'); act(e, 'apply', CUT, duration=1.)
    tool(e, 'bipolar_forceps'); act(e, 'apply', CUT, parameters={'power': 10.}, duration=1.)
    tool(e, 'cusa', (4., 8., 1.)); act(e, 'apply', [4., 8., 4.5])
    assert vb.PROPERTY not in e.state and e.state['removed']
    # The legacy compliant compartment still owns this scene's wounds.
    assert e.state['vesselsV2']['wounds'] and e.state['vesselsV2']['generatedMm3'] > 0
    again = SurgicalEngine(scene(bleeding=False, tumour=True))
    assert again.state_id == first


def test_declare_segments_kinds_runs_and_calibre_floor():
    labels = np.zeros((12, 12, 12), np.uint16)
    labels[2, 2:10, 2] = 1; labels[6:9, 2:10, 6:9] = 1; labels[2, 2:10, 9] = 2; labels[9, 2:10, 2] = 3
    regions = [{'id': 1, 'name': 'Middle cerebral artery', 'material': 'vessel'},
               {'id': 2, 'name': 'Vessel (angiography)', 'material': 'vessel'},
               {'id': 3, 'name': 'Transverse sinus', 'material': 'vessel'}]
    p = vb.declare(labels, regions, np.eye(4))
    rows = {s['id']: s for s in p['segments']}
    assert sorted(rows) == ['middle-cerebral-artery-1', 'middle-cerebral-artery-2', 'transverse-sinus-1',
                            'vessel-angiography-1']
    assert [rows[k]['kind'] for k in sorted(rows)] == ['artery', 'artery', 'sinus', 'uncertain']
    thin, thick = rows['middle-cerebral-artery-1'], rows['middle-cerebral-artery-2']
    assert thin['calibreMm'] == {'maximum': 1., 'medialMedian': 1., 'floorMm': 1., 'resolvedAtGrid': False}
    assert thick['calibreMm']['maximum'] == 3. and thick['calibreMm']['resolvedAtGrid']
    assert sorted(vb.cells_from_runs(thin['runs']).tolist()) == sorted(
        np.flatnonzero((labels == 1).ravel() & (np.indices(labels.shape)[0].ravel() == 2)).tolist())
    assert len(thick['runs']) == 24 and thick['cellCount'] == 72
    assert all(p['sources'][k] for k in vb.PARAMETERS)
    assert p['sources']['totalCerebralBloodFlowMlPerMin'].startswith('Zarrinkoob')
    assert p['sources']['referenceM1DiameterMm'] == 'assumed-research'
    with pytest.raises(ValueError, match='REQUIRES_VESSEL_CELLS'):
        vb.declare(labels, [{'id': 4, 'name': 'Artery', 'material': 'vessel'}], np.eye(4))
    wall = vb.declare_mask(labels == 2, np.eye(4), name='angiography vessel')
    assert [(s['kind'], s['regionId'], s['cellCount']) for s in wall['segments']] == [('uncertain', None, 8)]


def test_validation_requires_v2_and_vessel_cells():
    s = scene()
    props = copy.deepcopy(s.properties); props.pop('tissuePhysicsV2')
    with pytest.raises(ValueError, match='REQUIRES_EXPLICIT_V2'):
        SurgicalEngine(SurgicalScene(s.labels, s.affine, s.regions, s.source, props))
    props = copy.deepcopy(s.properties); props[vb.PROPERTY]['segments'][0]['runs'][0][0] -= 1
    with pytest.raises(ValueError, match='SEGMENT_CELL_NOT_VESSEL'):
        SurgicalEngine(SurgicalScene(s.labels, s.affine, s.regions, s.source, props))
    props = copy.deepcopy(s.properties); props[vb.PROPERTY]['parameters']['heartRatePerMin'] = -1.
    with pytest.raises(ValueError, match='INVALID_VESSEL_BLEEDING_PARAMETER'):
        SurgicalEngine(SurgicalScene(s.labels, s.affine, s.regions, s.source, props))


def test_cut_artery_bleeds_at_murray_rate_and_pulses():
    e = cut()
    w = wound(e)
    assert w['kind'] == 'artery' and w['calibreMm'] == 1. and w['calibreBasis'] == 'grid-floor'
    assert w['openingFraction'] == 1. and w['mechanisms'] == ['transection']
    assert murray(1.) == pytest.approx(114.3176, rel=1e-5)
    q = e.scene.properties[vb.PROPERTY]['parameters']
    rates, before, t0 = [], w['lossMm3'], e.state[vb.PROPERTY]['elapsedSeconds']
    for _ in range(12):
        act(e, 'inspect', duration=.1); rates.append(wound(e)['rateMm3PerSecond'])
    t1 = e.state[vb.PROPERTY]['elapsedSeconds']
    assert t1 - t0 == pytest.approx(1.2)
    expected = murray(1.) * vb.pulse_integral(q, t0, t1)
    assert wound(e)['lossMm3'] - before == pytest.approx(expected, rel=1e-9)
    # A 0.86 s cardiac period sampled at 0.1 s: the rate swings by about the pulsatility index.
    assert max(rates) / min(rates) > 1.8
    assert min(rates) < murray(1.) < max(rates)
    assert (max(rates) - min(rates)) / murray(1.) == pytest.approx(.9, abs=.15)


def test_vein_wells_steadily_and_follows_position_head():
    e = cut('Cortical vein')
    rates = []
    for _ in range(6):
        act(e, 'inspect', duration=.1); rates.append(wound(e)['rateMm3PerSecond'])
    assert rates == pytest.approx([murray(1., 'vein')] * 6, rel=1e-12)
    assert murray(1., 'vein') == pytest.approx(murray(1.) / 9)
    # Raising the site 50 mm above the heart lowers site venous pressure by rho g h.
    raised = cut('Cortical vein')
    act(raised, 'withdraw')
    position(raised, {'modelVersion': 1, 'preset': 'supine', 'siteHeightAboveHeartMm': 50.,
                      'heightEstimateSource': 'synthetic test'})
    head = raised.state[vb.PROPERTY]['venousHeadFactor']
    hydro = raised.observe()['positionEffects']['hydrostaticSiteMinusHeartPa']
    assert hydro < 0 and head == pytest.approx((10 * 133.3224 + hydro) / (10 * 133.3224))
    assert wound(raised)['rateMm3PerSecond'] == pytest.approx(murray(1., 'vein', head))


def test_uncertain_vessel_uses_venous_law_and_stays_labelled():
    e = cut('Vessel (angiography)')
    act(e, 'inspect', duration=.1)
    w = wound(e)
    assert w['kind'] == 'uncertain' and w['kindBasis'] == 'no artery, vein or sinus label'
    assert w['rateMm3PerSecond'] == pytest.approx(murray(1., 'vein'))
    m = e.metrics()['vesselPhysicsV2'][vb.PROPERTY]
    assert m['openVesselWoundsByKind'] == {'artery': 0, 'vein': 0, 'sinus': 0, 'uncertain': 1}
    assert m['bloodLossByKindMm3']['uncertain'] == pytest.approx(m['bloodLossMm3'])


def test_partial_transection_opens_a_lumen_fraction_of_a_large_artery():
    e = cut('Large artery', 3)
    w = wound(e)
    assert w['calibreMm'] == 3. and w['calibreBasis'] == 'local-thickness'
    assert w['openingFraction'] == pytest.approx(len(w['faces']) / (math.pi * 9 / 4))
    assert e.metrics()['vesselPhysicsV2'][vb.PROPERTY]['largestUncontrolledVesselCalibreMm'] == 3.


def test_mass_is_conserved_through_pooling_suction_and_overflow():
    e = cut()
    act(e, 'inspect', duration=1.)
    tool(e, 'frazier_suction'); act(e, 'apply', [8., 8., 6.], duration=.5)
    act(e, 'withdraw')
    position(e, {'modelVersion': 1, 'preset': 'left_lateral', 'heightEstimateSource': 'synthetic test'})
    act(e, 'inspect', duration=.5)
    s = e.state[vb.PROPERTY]; m = e.metrics()
    assert m['fluidRemovedMm3'] > 0
    assert s['bloodLossMm3'] == pytest.approx(s['depositedMm3'] + sum(s['reservoirMm3'].values()), rel=1e-12)
    assert e.state['fluidAddedMm3'] == pytest.approx(s['depositedMm3'])
    present = sum(e.state['fluid'].values())
    assert e.state['fluidAddedMm3'] == pytest.approx(present + m['fluidRemovedMm3'] + m['fluidEscapedMm3'], rel=1e-9)
    assert abs(m['vesselPhysicsV2'][vb.PROPERTY]['bloodConservationResidualMm3']) < 1e-9
    assert s['segmentLossMm3'] == {'cortical-artery-1': pytest.approx(s['bloodLossMm3'])}


def test_bipolar_seals_a_small_vessel():
    e = cut()
    tool(e, 'bipolar_forceps'); result = act(e, 'apply', CUT, parameters={'power': 10.}, duration=1.)
    assert any(ev['kind'] == 'vessel_coagulation' for ev in result['events'])
    w = wound(e)
    assert w['sealFraction'] == 1. and w['bipolarSealCap'] == 1.
    loss = e.state[vb.PROPERTY]['bloodLossMm3']
    act(e, 'withdraw'); act(e, 'inspect', duration=1.)
    assert e.state[vb.PROPERTY]['bloodLossMm3'] == loss
    m = e.metrics()['vesselPhysicsV2'][vb.PROPERTY]
    assert m['openVesselWoundsByKind']['artery'] == 0 and m['largestUncontrolledVesselCalibreMm'] is None


def test_bipolar_only_partly_controls_a_large_artery_and_not_a_sinus():
    big = cut('Large artery', 3)
    tool(big, 'bipolar_forceps'); act(big, 'apply', CUT, parameters={'power': 10.}, duration=3.)
    w = wound(big)
    assert w['bipolarSealCap'] == pytest.approx((1.5 / 3.) ** 2)
    assert w['sealFraction'] == pytest.approx(w['bipolarSealCap'])
    act(big, 'withdraw'); act(big, 'inspect', duration=.1)
    assert wound(big)['rateMm3PerSecond'] > 0
    assert big.metrics()['vesselPhysicsV2'][vb.PROPERTY]['largestUncontrolledVesselCalibreMm'] == 3.
    sinus = cut('Superior sagittal sinus', 3)
    tool(sinus, 'bipolar_forceps'); result = act(sinus, 'apply', CUT, parameters={'power': 10.}, duration=1.)
    assert wound(sinus)['sealFraction'] == 0.
    assert any(ev['kind'] == 'bipolar_ineffective' and ev['vesselKind'] == 'sinus' for ev in result['events'])


def contact_cells(e):
    return sorted(vb._wound_contact_cells(e, wound(e)))


def test_irrigation_raises_seal_energy_and_cleans_char():
    dry, wet = cut(), cut()
    dry.state['fluid'] = {}
    wet.state['fluid'] = {str(c): 1. for c in wet.cells(CUT, CUT, 1.)}
    for e in (dry, wet):
        vb.coagulate(e, 'left', contact_cells(e), CUT, 1., 1.)
    assert wound(dry)['sealFraction'] == pytest.approx(.5)
    assert wound(wet)['sealFraction'] == pytest.approx(.5 / 1.5)
    assert dry.state[vb.PROPERTY]['bipolar']['left']['charSticking'] == pytest.approx(.25)
    assert wet.state[vb.PROPERTY]['bipolar']['left']['charSticking'] == 0.
    dry.state['fluid'] = {str(c): 1. for c in dry.cells(CUT, CUT, 1.)}
    vb.coagulate(dry, 'left', contact_cells(dry), CUT, .2, 1.)
    assert dry.state[vb.PROPERTY]['bipolar']['left']['charSticking'] == pytest.approx(.15)


def test_pulling_a_stuck_bipolar_reopens_the_wound_once():
    e = cut()
    tool(e, 'bipolar_forceps'); act(e, 'apply', CUT, parameters={'power': .5})
    e.state['fluid'] = {}
    vb.coagulate(e, 'left', contact_cells(e), CUT, 2., 2.)
    row = e.state[vb.PROPERTY]['bipolar']['left']
    assert wound(e)['sealFraction'] == 1. and row['stuck'] and row['stuckWounds'] == [wound(e)['id']]
    act(e, 'withdraw')
    w = wound(e)
    assert w['charReopened'] and w['sealFraction'] == 0. and w['rateMm3PerSecond'] > 0
    row = e.state[vb.PROPERTY]['bipolar']['left']
    assert not row['stuck'] and row['pullOffs'] == 1 and row['lastReopenedWounds'] == [w['id']]
    tool(e, 'bipolar_forceps'); act(e, 'apply', CUT, parameters={'power': .5}); e.state['fluid'] = {}
    vb.coagulate(e, 'left', contact_cells(e), CUT, 2., 2.)
    assert e.state[vb.PROPERTY]['bipolar']['left']['stuck']
    act(e, 'withdraw')
    assert wound(e)['sealFraction'] == 1. and e.state[vb.PROPERTY]['bipolar']['left']['pullOffs'] == 2


def test_tumour_removal_oozes_and_decays():
    e = SurgicalEngine(scene(tumour=True))
    tool(e, 'cusa', (4., 8., 1.)); act(e, 'apply', [4., 8., 4.5])
    s = e.state[vb.PROPERTY]
    assert e.state['removed'] and s['oozeCells'] and s['tumourOozeMm3'] > 0
    removed = set(e.state['removed'])
    assert not removed.intersection(map(int, s['oozeCells']))
    assert all(int(e.scene.labels.flat[int(k)]) == 2 for k in s['oozeCells'])
    assert all(entry[2] in removed for entry in s['oozeCells'].values())
    act(e, 'withdraw'); act(e, 'inspect', duration=.1); first = s_rate = e.state[vb.PROPERTY]['rateMm3PerSecond']
    act(e, 'inspect', duration=20.); act(e, 'inspect', duration=.1)
    assert e.state[vb.PROPERTY]['rateMm3PerSecond'] == pytest.approx(first * math.exp(-20.1 / 1800.), rel=1e-6)
    area = sum(entry[1] for entry in e.state[vb.PROPERTY]['oozeCells'].values())
    assert e.state[vb.PROPERTY]['rateMm3PerSecond'] == pytest.approx(.004 * area * math.exp(.1 / 1800.) * 1800 /
                                                                     .1 * (1 - math.exp(-.1 / 1800.)), rel=1e-6)
    assert s_rate > 0 and wound_free(e)


def wound_free(e):
    return not e.state[vb.PROPERTY]['wounds']


def test_ooze_stops_under_bipolar_and_a_placed_dressing():
    e = SurgicalEngine(scene(tumour=True))
    tool(e, 'cusa', (4., 8., 1.)); act(e, 'apply', [4., 8., 4.5])
    ooze = e.state[vb.PROPERTY]['oozeCells']
    target = min(ooze, key=int)
    vb.coagulate(e, 'left', [int(target)], e.scene.points([int(target)])[0], 1., 1.)
    assert ooze[target][3] == 1 and ooze[target][1] == 0.
    point = e.scene.points([int(max(ooze, key=int))])[0]
    e.state['operativeExposureV1'] = {'objects': [{'id': 'hemostat-1', 'kind': 'hemostat', 'status': 'placed',
                                                   'heldBy': None, 'positionMm': point.tolist(), 'radiusMm': .95}]}
    act(e, 'withdraw'); act(e, 'inspect', duration=.1)
    stopped = [k for k, entry in e.state[vb.PROPERTY]['oozeCells'].items() if entry[3] == 2]
    assert str(int(max(ooze, key=int))) in stopped
    assert all(e.state[vb.PROPERTY]['oozeCells'][k][1] == 0. for k in stopped)


def test_ooze_rate_is_tens_of_ml_per_hour_for_a_14_cm3_cavity():
    shape = (40, 40, 40); labels = np.full(shape, 2, np.uint16)
    radius = (3 * 14000. / (4 * math.pi)) ** (1 / 3)
    inside = np.linalg.norm(np.indices(shape).reshape(3, -1).T - 19.5, axis=1) <= radius
    regions = [{'id': 1, 'name': 'Artery', 'material': 'vessel'}, {'id': 2, 'name': 'Tumour', 'material': 'tumor'}]
    labels.flat[0] = 1
    s = SurgicalScene(labels, np.eye(4), regions, {}, {'tissuePhysicsV2': default_profile(),
                                                       vb.PROPERTY: vb.declare(labels, regions, np.eye(4))})
    e = SimpleNamespace(scene=s, state={'removed': np.flatnonzero(inside).tolist()})
    vb.initialize(e); vb._scan_ooze(e, e.state[vb.PROPERTY])
    area = sum(entry[1] for entry in e.state[vb.PROPERTY]['oozeCells'].values())
    smooth = 4 * math.pi * radius ** 2
    assert inside.sum() == pytest.approx(14000, rel=.01) and 1.2 < area / smooth < 1.7
    per_hour_ml = .004 * area * 3600 / 1000
    assert 10 < .004 * smooth * 3600 / 1000 < per_hour_ml < 100


def test_replay_and_snapshot_restore_are_deterministic():
    def run():
        e = SurgicalEngine(scene(tumour=True))
        tool(e, 'micro_scissors'); act(e, 'apply', CUT, duration=1.)
        tool(e, 'cusa', (4., 8., 1.)); act(e, 'apply', [4., 8., 4.5])
        tool(e, 'bipolar_forceps'); act(e, 'apply', CUT, parameters={'power': 3.}, duration=.4)
        return e
    a, b = run(), run()
    assert a.state_id == b.state_id and a.state[vb.PROPERTY] == b.state[vb.PROPERTY]
    restored = SurgicalEngine.restore(a.scene, a.snapshot())
    assert restored.state_id == a.state_id
    for e in (a, restored):
        act(e, 'inspect', duration=.3)
    assert restored.state_id == a.state_id and restored.state[vb.PROPERTY]['bloodLossMm3'] > 0
    replayed = SurgicalEngine.restore(a.scene, {k: v for k, v in a.snapshot().items() if k != 'checkpoint'})
    assert replayed.state_id == a.state_id
    assert a.observe()['physics']['vessel'][vb.PROPERTY]['model'] == vb.SCHEMA
