"""Tissue-selective ultrasonic aspirator (cusaV4) on small generated grids.

Synthetic engineering geometry only; the checks are oracles for the declared
research law, not device or tissue validation.
"""
import copy
from fractions import Fraction
import json

import numpy as np
import pytest

from resectionlab import surgical_cusa_v4 as v4
from resectionlab import surgical_vessel_bleeding as vb
from resectionlab.surgical_cusa_v3 import default_profile as v3_profile
from resectionlab.surgical_engine import SurgicalEngine
from resectionlab.surgical_injury import PROPERTY as INJURY, default_profile as injury_profile
from resectionlab.surgical_physics_v2 import default_profile as v2_profile
from resectionlab.surgical_scene import SurgicalScene

Q = [0., 0., 0., 1.]
WET = {'irrigation': 4.}


def pose(position):
    return {'position': [float(v) for v in position], 'quaternion': Q}


def coupon(groups, *, overrides=None, extra=None, shape=(20, 20, 20)):
    """groups: [(name, material, [ijk, ...]), ...]; regions get ids in order."""
    labels = np.zeros(shape, np.uint16)
    regions = []
    for index, (name, material, points) in enumerate(groups, 1):
        for point in points:
            labels[point] = index
        regions.append({'id': index, 'name': name, 'material': material, 'target': material == 'tumor'})
    properties = {v4.PROPERTY: {'schema': v4.SCHEMA, **(overrides or {})}, **(extra or {})}
    scene = SurgicalScene(labels, np.eye(4), regions, {'kind': 'generated CUSA V4 coupon; no acquired anatomy'},
                          properties)
    engine = SurgicalEngine(scene)
    cells = {name: sorted(map(int, np.flatnonzero(labels.ravel() == index)))
             for index, (name, _, _) in enumerate(groups, 1)}
    return engine, cells


def contact(engine, seconds, position=(9., 9., 9.), active=True, **values):
    cells = engine.touched(np.asarray(position, float), np.asarray(position, float), v4.TIP_RADIUS_MM)
    return v4.apply(engine, cells, seconds, {'radiusMm': 1.5, **values},
                    position=np.asarray(position, float), active=active)


def ledger(engine):
    state = engine.state[v4.PROPERTY]
    pending = sum(state['progressMm3'].values())
    assert state['inputCapacityMm3'] == pytest.approx(
        state['fragmentedSolidMm3'] + state['selectivityCapacityMm3'] + pending +
        state['unusedCapacityMm3'] + state['discardedCapacityMm3'], abs=1e-9)
    assert state['fragmentedSolidMm3'] + state['sparedFailedMm3'] == pytest.approx(
        state['aspiratedSolidMm3'] + state['debrisLeftMm3'], abs=1e-9)
    assert set(state['progressMm3']) == set(state['progressExact'])
    return state


# Captured from HEAD d49f379d before cusaV4/suctionTractionV1 existed (reference
# backend): a cusaV3 stroke and a V2 suction-then-retractor sequence.
V3_IDS = ['sha256:cf7d819a412a73066fc2e27c9a8f11307706cdd7e08019c4397d2cd177269528',
          'sha256:3c35752cc169f048a7523ffca0858358b48edb89c025a5b30d0f76a9bbb4e56d',
          'sha256:79d5c11c5c909115479137d2fa34a8dafc6aabe0aa1870616d7bf8a2b26b0f62',
          'sha256:3343f53695037518c61005284c9e9d2cd23339f872739915be24b65b4b97cb8f']


def test_absent_property_keeps_v3_state_ids_and_ledgers():
    labels = np.zeros((20, 20, 20), np.uint16)
    labels[9, 9, 9] = labels[9, 9, 10] = 1
    scene = SurgicalScene(labels, np.eye(4), [{'id': 1, 'name': 'coupon', 'material': 'tumor', 'target': True}],
                          {'kind': 'generated'}, {'cusaV3': v3_profile()})
    engine = SurgicalEngine(scene)
    ids = [engine.step(a)['stateId'] for a in [
        {'slot': 'left', 'operation': 'exchange', 'instrument': 'cusa', 'duration': .1},
        {'slot': 'left', 'operation': 'place', 'duration': .1, 'pose': pose([9, 9, 3])},
        {'slot': 'left', 'operation': 'apply', 'duration': .5, 'pose': pose([9, 9, 12])},
        {'slot': 'left', 'operation': 'apply', 'duration': 1., 'pose': pose([9, 9, 12])}]]
    assert ids == V3_IDS
    assert v4.PROPERTY not in engine.state and not any(k.startswith('cusaV4') for k in engine.metrics())
    refused = engine.step({'slot': 'left', 'operation': 'apply', 'duration': .1, 'parameters': {'amplitude': .5}})
    assert refused['reason'] == 'Unsupported instrument parameters'


def test_profile_is_validated_and_region_kinds_must_match_material():
    assert v4.validate(v4.default_profile())['baseRateMm3PerSecond'] == 40.
    for change in ({'schema': 'other'}, {'unknown': 1}, {'baseRateMm3PerSecond': 0.},
                   {'sparingAmplitude': 1.}, {'relativeRate': {'tumour_core': 1.}},
                   {'regionKinds': {'x': 'bone'}}, {'defaultAmplitude': 2.},
                   {'provenance': {'kind': 'calibrated', 'description': 'no'}}):
        with pytest.raises(ValueError):
            v4.validate({**v4.default_profile(), **change})
    with pytest.raises(ValueError, match='REGION_KIND_MATERIAL_MISMATCH'):
        coupon([('Artery', 'vessel', [(9, 9, 9)])], overrides={'regionKinds': {'Artery': 'tumour_core'}})
    engine, _ = coupon([('Tumour core', 'tumor', [(9, 9, 9)]), ('Tumour rim', 'tumor', [(9, 9, 10)]),
                        ('Cerebral cortex', 'soft_tissue', [(9, 9, 11)]), ('White matter', 'soft_tissue', [(9, 9, 12)]),
                        ('CST (tract)', 'soft_tissue', [(9, 9, 13)]), ('Dura (assumed)', 'membrane', [(9, 9, 14)]),
                        ('Arachnoid', 'membrane', [(9, 9, 15)]), ('Vein', 'vessel', [(9, 9, 16)])])
    p = v4.profile(engine)
    assert [v4._region_kind(p, r) for r in engine.scene.regions] == [
        'tumour_core', 'tumour_rim', 'grey_matter', 'white_matter', 'tract', 'dura', 'membrane', 'vessel']


def test_rate_scales_with_amplitude_and_tissue_kind():
    core, cells = coupon([('Tumour core', 'tumor', [(9, 9, 9)])])
    contact(core, .01, amplitude=1., **WET)
    contact(core, .01, amplitude=.5, **WET)
    state = ledger(core)
    assert state['progressMm3'][str(cells['Tumour core'][0])] == pytest.approx(.4 + .2)
    white, cells = coupon([('White matter', 'soft_tissue', [(9, 9, 9)])])
    contact(white, 1 / 40, amplitude=1., **WET)
    # Tumour core would be gone after 1/40 s; white matter needs 1/0.3 of the capacity.
    assert not white.state['removed']
    assert ledger(white)['progressMm3'][str(cells['White matter'][0])] == pytest.approx(1.)
    contact(white, 1 / 12 - 1 / 40 + 1e-9, amplitude=1., **WET)
    assert white.state['removed'] == cells['White matter']
    assert ledger(white)['selectivityCapacityMm3'] == pytest.approx(1 / .3 - 1)
    for kind, rate in (('tumour_rim', .6), ('grey_matter', .5), ('tract', .3)):
        assert v4.profile(white)['relativeRate'][kind] == rate


def test_capacity_is_shared_nearest_first_and_spared_cells_take_none():
    engine, cells = coupon([('Tumour core', 'tumor', [(8, 9, 9), (10, 9, 9)]), ('Vein', 'vessel', [(9, 8, 9)])])
    events = contact(engine, .0375, amplitude=1., **WET)
    tumour = cells['Tumour core']
    assert engine.state['removed'] == [min(tumour)]
    assert engine.state[v4.PROPERTY]['progressMm3'] == {str(max(tumour)): pytest.approx(.5)}
    assert events[0]['sparedContacts'] == 1 and events[0]['inputCapacityMm3'] == pytest.approx(1.5)


def test_vessel_and_dura_are_spared_at_low_amplitude_and_fail_with_dwell_at_high():
    thin, cells = coupon([('Cortical artery', 'vessel', [(9, 9, 9)])])
    for _ in range(10):
        assert not any(e['kind'] == 'cusa_spared_failure' for e in contact(thin, 1., amplitude=.3, **WET))
    assert not thin.state['removed'] and not thin.state[v4.PROPERTY]['sparedDoseSeconds']
    contact(thin, .9, amplitude=1., **WET)
    assert not thin.state['removed']
    assert thin.state[v4.PROPERTY]['sparedDoseSeconds'][str(cells['Cortical artery'][0])] == pytest.approx(.9)
    events = contact(thin, .1, amplitude=1., **WET)
    assert thin.state['removed'] == cells['Cortical artery']
    failure = [e for e in events if e['kind'] == 'cusa_spared_failure'][0]
    assert failure['kinds'] == ['vessel'] and failure['regions'] == ['Cortical artery']
    assert thin.state[v4.PROPERTY]['sparedFailures'][0]['calibreMm'] == 1.
    # Half the excess amplitude above the sparing floor halves the dose rate.
    half, _ = coupon([('Cortical artery', 'vessel', [(9, 9, 9)])])
    contact(half, 1.9, amplitude=.65, **WET)
    assert not half.state['removed']
    contact(half, .1, amplitude=.65, **WET)
    assert half.state['removed']
    # A 3 mm vessel needs (3/1)^2 = 9 full-amplitude seconds.
    thick_points = [(i, j, k) for i in (8, 9, 10) for j in (8, 9, 10) for k in range(4, 16)]
    thick, cells = coupon([('Cortical artery', 'vessel', thick_points)])
    tip = (9., 9., 2.)
    nearest = v4._ordered_contacts(thick, cells['Cortical artery'], np.array(tip))
    assert nearest and v4.failure_seconds(thick, nearest[0])[1] == 3.
    contact(thick, 8.9, position=tip, amplitude=1., **WET)
    assert not thick.state['removed']
    contact(thick, .2, position=tip, amplitude=1., **WET)
    assert thick.state['removed']
    dura, _ = coupon([('Dura (assumed)', 'membrane', [(9, 9, 9)])])
    contact(dura, 2.75, amplitude=1., **WET)
    assert not dura.state['removed']
    contact(dura, .25, amplitude=1., **WET)
    assert dura.state['removed']


def test_aspiration_cap_leaves_debris_that_aspiration_then_clears():
    points = [(x, y, z) for x in (8, 9, 10) for y in (8, 9, 10) for z in (8, 9, 10)]
    engine, cells = coupon([('Tumour core', 'tumor', points)])
    contact(engine, .5, amplitude=1., aspiration=.02, irrigation=0., position=(9., 9., 9.))
    state = ledger(engine)
    fragmented = state['fragmentedSolidMm3']
    assert fragmented >= 3 and state['aspiratedSolidMm3'] == pytest.approx(150 * .02 * .5)
    assert state['debrisLeftMm3'] == pytest.approx(fragmented - 1.5)
    assert engine.state['debrisMm3'] == pytest.approx(fragmented - 1.5)
    assert set(engine.state['debris']) <= {str(c) for c in engine.state['removed']}
    contact(engine, .5, amplitude=0., aspiration=1., irrigation=0., position=(9., 9., 9.))
    assert engine.state['debrisMm3'] == pytest.approx(0., abs=1e-9)
    assert engine.state['debrisRemovedMm3'] == pytest.approx(fragmented - 1.5)
    assert ledger(engine)['debrisAspiratedMm3'] == pytest.approx(fragmented - 1.5)


def test_irrigation_wets_the_field_and_running_dry_heats_the_tip():
    engine, _ = coupon([('Tumour core', 'tumor', [(9, 9, 12)])])
    contact(engine, .6, position=(9., 9., 9.), amplitude=1., aspiration=0., irrigation=6.)
    assert engine.state['fluidAddedMm3'] == pytest.approx(6 * 1000 / 60 * .6)
    assert sum(engine.state['fluid'].values()) == pytest.approx(60.)
    assert engine.state[v4.PROPERTY]['tipHeat'] == 0.
    contact(engine, .6, position=(9., 9., 9.), amplitude=0., aspiration=1., irrigation=0.)
    assert engine.state['fluidRemovedMm3'] == pytest.approx(60.)
    dry, _ = coupon([('Tumour core', 'tumor', [(9, 9, 12)])])
    budgets = [contact(dry, .5, position=(9., 9., 9.), amplitude=1., irrigation=0.)[0]['inputCapacityMm3']
               for _ in range(4)]
    assert budgets[0] == 20. and all(a > b for a, b in zip(budgets, budgets[1:]))
    heat = dry.state[v4.PROPERTY]['tipHeat']
    assert heat > .5
    # Idle time cools the tip before the next application.
    dry.state['elapsedSeconds'] += 10.
    contact(dry, .01, position=(9., 9., 9.), amplitude=1., irrigation=4.)
    assert dry.state[v4.PROPERTY]['tipHeat'] < heat / 5


def test_exact_progress_never_removes_a_cell_early():
    engine, cells = coupon([('Tumour core', 'tumor', [(9, 9, 9)])], overrides={'baseRateMm3PerSecond': 1.})
    almost = float(np.nextafter(1., 0.))
    gap = 1. - almost
    contact(engine, almost, amplitude=1., **WET)
    contact(engine, gap * .75, amplitude=1., **WET)
    state = engine.state[v4.PROPERTY]
    assert state['progressMm3'][str(cells['Tumour core'][0])] == 1.
    assert Fraction(*state['progressExact'][str(cells['Tumour core'][0])]) < 1
    assert not engine.state['removed']
    contact(engine, gap * .25, amplitude=1., **WET)
    assert engine.state['removed'] == cells['Tumour core']
    ledger(engine)


def test_inactive_zero_amplitude_and_protected_contact():
    engine, cells = coupon([('Tumour core', 'tumor', [(9, 9, 9)])])
    assert contact(engine, 5., active=False) == []
    contact(engine, 5., amplitude=0., aspiration=0., irrigation=0.)
    assert not engine.state['removed'] and ledger(engine)['inputCapacityMm3'] == 0.
    engine._protected_ids = frozenset(cells['Tumour core'])
    before = copy.deepcopy(engine.state)
    with pytest.raises(ValueError, match='PROTECTED_STRUCTURE'):
        contact(engine, 1., amplitude=1.)
    assert engine.state == before
    with pytest.raises(ValueError, match='INVALID_CUSA_V4_INPUT:amplitude'):
        contact(engine, 1., amplitude=1.5)
    with pytest.raises(ValueError, match='FIXED_CATALOG_GEOMETRY'):
        v4.validate_parameters({'radiusMm': 3.})


def test_contact_limited_pose_stops_at_a_vessel_wall():
    engine, cells = coupon([('Vein', 'vessel', [(9, 9, 9)])])
    before = copy.deepcopy(engine.state)
    limited, found = v4.contact_limited_pose(engine, np.array([9., 9., 2.]), np.array([9., 9., 12.]))
    assert limited[2] < 8. and found == cells['Vein'] and engine.state == before


def stroke_scene(extra=None, *, vessel=True):
    labels = np.zeros((20, 20, 18), np.uint16)
    labels[:, :, 8:] = 1
    labels[5:13, 5:13, 8:13] = 2
    regions = [{'id': 1, 'name': 'Cerebral white matter', 'material': 'soft_tissue'},
               {'id': 2, 'name': 'Tumour core', 'material': 'tumor', 'target': True}]
    if vessel:
        labels[9, 3:17, 13] = 3
        regions.append({'id': 3, 'name': 'Cortical artery', 'material': 'vessel'})
    properties = {v4.PROPERTY: {'schema': v4.SCHEMA}, **(extra or {})}
    return SurgicalScene(labels, np.eye(4), regions, {'kind': 'generated'}, properties)


STROKE = [
    {'slot': 'left', 'operation': 'configure', 'mechanicsRevision': 2, 'duration': .1},
    {'slot': 'right', 'operation': 'place', 'duration': .1, 'pose': pose([9, 9, 3])},
    {'slot': 'right', 'operation': 'apply', 'duration': 1., 'pose': pose([9, 9, 12]),
     'parameters': {'amplitude': 1., 'aspiration': .5, 'irrigation': 4.}},
    {'slot': 'right', 'operation': 'apply', 'duration': 2., 'pose': pose([9, 9, 12])},
    {'slot': 'right', 'operation': 'apply', 'duration': 2., 'pose': pose([9, 9, 12]),
     'parameters': {'amplitude': .2}},
    {'slot': 'right', 'operation': 'apply', 'duration': 2., 'pose': pose([9, 9, 12]),
     'parameters': {'amplitude': 1.}},
]


def run(engine, actions):
    return [engine.step(action) for action in actions]


def test_engine_stroke_replays_and_restores_deterministically():
    engine = SurgicalEngine(stroke_scene())
    results = run(engine, STROKE)
    assert all(r['status'] != 'rejected' for r in results)
    assert engine.state['removed'] and engine.metrics()['cusaV4FragmentedSolidMm3'] > 0
    # Low amplitude stalls on the artery without breaching it; full amplitude breaches it.
    assert results[4]['reasonCode'] == 'TISSUE_ASPIRATION_IN_PROGRESS'
    assert not any(e['kind'] == 'cusa_spared_failure' for e in results[4]['events'])
    assert any(e['kind'] == 'cusa_spared_failure' for e in results[5]['events'])
    assert 'cusaV4' in engine.view() and 'cusaV4' in engine.observe()['physics']
    replayed = SurgicalEngine(stroke_scene()).replay(engine.actions)
    assert replayed.state_id == engine.state_id
    restored = SurgicalEngine.restore(engine.scene, json.loads(json.dumps(engine.snapshot())))
    assert restored.state_id == engine.state_id
    partial = SurgicalEngine(stroke_scene())
    run(partial, STROKE[:4])
    resumed = SurgicalEngine.restore(partial.scene, json.loads(json.dumps(partial.snapshot())))
    for action in STROKE[4:]:
        assert partial.step(action) == resumed.step(action)
    assert partial.state_id == engine.state_id


def test_cusa_v4_replaces_v3_after_mechanics_revision():
    engine = SurgicalEngine(stroke_scene(vessel=False))
    run(engine, STROKE[:3])
    assert engine.state['removed'] and engine.state['cusaV3']['inputCapacityMm3'] == 0.
    assert engine.state[v4.PROPERTY]['inputCapacityMm3'] > 0.


def test_v4_with_vessel_bleeding_and_injury_replays_identically():
    def scene():
        base = stroke_scene()
        properties = {**base.properties, 'tissuePhysicsV2': v2_profile(),
                      vb.PROPERTY: vb.declare(base.labels, base.regions, base.affine),
                      INJURY: {**injury_profile(), 'mode': 'injury'}}
        return SurgicalScene(base.labels, base.affine, base.regions, base.source, properties)
    engine = SurgicalEngine(scene())
    results = run(engine, STROKE)
    assert all(r['status'] != 'rejected' for r in results)
    failures = engine.state[v4.PROPERTY]['sparedFailures']
    assert failures and all(f['kind'] == 'vessel' for f in failures)
    failed = {f['cell'] for f in failures}
    assert failed <= set(engine.state['removed'])
    assert engine.state[vb.PROPERTY]['wounds'], 'a breached vessel opens a bleeding wound'
    record = engine.state['injuryV1']
    assert failed <= set(record['vesselInjured'])
    ultrasonic = [e for e in record['events'] if e.get('mechanism') == 'ultrasonic']
    assert {e['kind'] for e in ultrasonic} == {'laceration', 'vessel_injury'}
    assert any(e['kind'] == 'vessel_injury' for r in results for e in r['events'])
    replayed = SurgicalEngine(scene()).replay(engine.actions)
    assert replayed.state_id == engine.state_id
    restored = SurgicalEngine.restore(engine.scene, json.loads(json.dumps(engine.snapshot())))
    assert restored.state_id == engine.state_id
