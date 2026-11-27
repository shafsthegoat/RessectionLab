"""Suction as the second hand (suctionTractionV1) on a small generated grid.

Synthetic engineering geometry; checks the declared hold law, not tissue data.
"""
import json
import math

import numpy as np
import pytest

from resectionlab import surgical_suction_traction as st
from resectionlab.surgical_engine import SurgicalEngine
from resectionlab.surgical_injury import PROPERTY as INJURY, default_profile as injury_profile
from resectionlab.surgical_physics_v2 import default_profile as v2_profile
from resectionlab.surgical_scene import SurgicalScene

Q = [0., 0., 0., 1.]


def pose(position):
    return {'position': [float(v) for v in position], 'quaternion': Q}


def scene(traction=True, *, extra=None, v2=True):
    labels = np.zeros((20, 20, 16), np.uint16)
    labels[:, :, 8:] = 1
    labels[6:10, 6:10, 8:11] = 2
    regions = [{'id': 1, 'name': 'Cerebral cortex', 'material': 'soft_tissue'},
               {'id': 2, 'name': 'Tumour core', 'material': 'tumor', 'target': True}]
    properties = {'tissuePhysicsV2': v2_profile()} if v2 else {}
    if traction:
        properties[st.PROPERTY] = {'schema': st.SCHEMA}
    properties.update(extra or {})
    return SurgicalScene(labels, np.eye(4), regions, {}, properties)


def engage(aspiration=None):
    parameters = {} if aspiration is None else {'parameters': {'aspiration': aspiration}}
    return [{'slot': 'left', 'operation': 'exchange', 'instrument': 'frazier_suction', 'duration': .1},
            {'slot': 'left', 'operation': 'place', 'duration': .1, 'pose': pose([14, 14, 3])},
            {'slot': 'left', 'operation': 'activate', 'duration': .1, **parameters},
            {'slot': 'left', 'operation': 'move', 'duration': .5, 'pose': pose([14, 14, 6.6])}]


def pull(z, duration=.5):
    return {'slot': 'left', 'operation': 'move', 'duration': duration, 'pose': pose([14, 14, z])}


def run(engine, actions):
    results = [engine.step(action) for action in actions]
    assert all(r['status'] == 'accepted' for r in results), [r.get('reason') for r in results]
    return results


def kinds(result):
    return [e['kind'] for e in result['events']]


# Captured from HEAD d49f379d before suctionTractionV1 existed (reference backend).
NO_PROPERTY_IDS = ['sha256:eb816dfa1bbca33f5650472099e721ab85663fa2d9f86781fa8bc9566d05e55b',
                   'sha256:1b53caacaec630f7cc02899c90a5065c68d45366ebfd9ce975023ee2006df2f4',
                   'sha256:ed56b827370dd1cea39d47d511cdbc608d50b1f290311740515b71c8f2110dfd',
                   'sha256:f29e2007fb5105caa7a5d3ea7c35d1f21848d239c048d3af7969329306a69a81',
                   'sha256:36612662c021f76d221f22a8d46fbaa0f7376179c65ce35a0ca475cddc11e1a2',
                   'sha256:8abf91fe46e39acaa7ce4ed3459b035fd6f3c67ba6de1c879882e09ff7d4cb7c',
                   'sha256:7708de78f2b220500cb374a6a13d394e06c8b25041edd92454ea605ee5aeaf4b',
                   'sha256:ef5d6437e25d48271843ad4a2dfebcc66be54c1c667b76aa6e92cf79562d94ec',
                   'sha256:2bf831d062b7a47d84cfa313a6743142e79fcf9ad529b147e8de36628cf059d7',
                   'sha256:6b43d9bfb3df0379b5cd30efeaa84f4d0a2325576377ee6637876928ddc14050',
                   'sha256:d2dbde30a8e094374df63926cf81a3668a91285ddc9a2036fc6cb37e630aa186',
                   'sha256:bc2b8449cfa55a2778a3105570c032b6f543c4d146f8f4d5697a13f02c9e9283']


def test_absent_property_keeps_suction_and_retractor_state_ids_and_events():
    engine = SurgicalEngine(scene(traction=False))
    engine.state['fluid']['1000'] = 1.
    actions = [
        *engage(),{'slot': 'left', 'operation': 'apply', 'duration': .5}, pull(5.),
        {'slot': 'left', 'operation': 'deactivate', 'duration': .1},
        {'slot': 'right', 'operation': 'exchange', 'instrument': 'brain_retractor', 'duration': .1},
        {'slot': 'right', 'operation': 'place', 'duration': .1, 'pose': pose([4, 4, 3])},
        {'slot': 'right', 'operation': 'hold', 'duration': .2, 'pose': pose([4, 4, 6.])},
        {'slot': 'right', 'operation': 'move', 'duration': .4, 'pose': pose([4.4, 4, 6.])},
        {'slot': 'right', 'operation': 'release', 'duration': .1}]
    results = run(engine, actions)
    assert [r['stateId'] for r in results] == NO_PROPERTY_IDS
    assert not engine.state['holds'].get('left') and st.PROPERTY not in engine.state
    assert not any(k.startswith('suction') for k in engine.metrics())
    assert not any(e['kind'].startswith(('suction', 'tissue_reaction')) for r in results for e in r['events'])
    refused = engine.step({'slot': 'left', 'operation': 'activate', 'duration': .1, 'parameters': {'aspiration': .5}})
    assert refused['reason'] == 'Unsupported instrument parameters'


def test_profile_requires_schema_and_v2_mechanics():
    with pytest.raises(ValueError, match='REQUIRES_TISSUE_PHYSICS_V2'):
        SurgicalEngine(scene(v2=False))
    with pytest.raises(ValueError, match='PROFILE_INVALID'):
        SurgicalEngine(scene(extra={st.PROPERTY: {'schema': st.SCHEMA, 'maxVacuumMmHg': -1.}}))
    with pytest.raises(ValueError, match='PROFILE_INVALID'):
        SurgicalEngine(scene(extra={st.PROPERTY: {'schema': 'other'}}))
    engine = SurgicalEngine(scene())
    # 200 mmHg x pi (1 mm)^2: about 0.084 N at full vacuum.
    assert st.hold_force_n(engine, 1.) == pytest.approx(200 * 133.322 * math.pi * 1e-6)
    assert st.hold_force_n(engine, .5) == pytest.approx(st.hold_force_n(engine, 1.) / 2)


def test_suction_holds_gentle_traction_and_releases_at_the_force_limit():
    engine = SurgicalEngine(scene())
    results = run(engine, engage())
    assert 'suction_hold_created' in kinds(results[-1])
    hold = engine.state['holds']['left']
    assert 'target' in hold and hold['cells']
    row = engine.state[st.PROPERTY]['holds']['left']
    assert row['holdForceN'] == pytest.approx(st.hold_force_n(engine, .5))
    gentle = run(engine, [pull(6.2), pull(5.8)])
    assert 'left' in engine.state['holds']
    # The coupled hold drags the gripped patch toward the tip.
    moved = [np.linalg.norm(engine.state['deformation'][str(c)]) for c in hold['cells']]
    assert max(moved) > .3
    estimate = [e for e in gentle[-1]['events'] if e['kind'] == 'tissue_reaction_estimate']
    assert estimate and estimate[0]['instrument'] == 'frazier_suction'
    assert 0 < estimate[0]['magnitudeN'] < row['holdForceN']
    results = run(engine, [pull(5.4), pull(5.), pull(4.), pull(3.)])
    released = [e for r in results for e in r['events'] if e['kind'] == 'suction_hold_released']
    assert released and released[0]['reason'] == 'hold-force-exceeded'
    assert released[0]['loadN'] > released[0]['holdForceN']
    release = engine.state[st.PROPERTY]['releases'][0]
    assert release['reason'] == 'hold-force-exceeded' and engine.metrics()['suctionReleases'] >= 1
    # A torn-off patch is not re-gripped in the action that tore it.
    torn = next(r for r in results if any(e['kind'] == 'suction_hold_released' for e in r['events']))
    after = kinds(torn)[kinds(torn).index('suction_hold_released'):]
    assert 'suction_hold_created' not in after


def test_more_vacuum_holds_a_longer_pull_than_less():
    def release_depth(aspiration):
        engine = SurgicalEngine(scene())
        run(engine, [*engage(aspiration), pull(6.2), pull(5.8), pull(5.4)])
        z = 5.4
        while 'left' in engine.state['holds']:
            z = round(z - .05, 3)
            assert z > 3.
            run(engine, [pull(z, .25)])
        row = engine.state[st.PROPERTY]['releases'][0]
        assert row['reason'] == 'hold-force-exceeded' and row['loadN'] > row['holdForceN']
        return z
    assert release_depth(1.) < release_depth(.5)


def test_vacuum_off_releases_the_patch_and_withdraw_requires_release():
    engine = SurgicalEngine(scene())
    run(engine, engage())
    refused = engine.step({'slot': 'left', 'operation': 'withdraw', 'duration': .2})
    assert refused['reasonCode'] == 'RELEASE_BEFORE_WITHDRAWAL'
    result = run(engine, [{'slot': 'left', 'operation': 'deactivate', 'duration': .1}])[0]
    assert [e['reason'] for e in result['events'] if e['kind'] == 'suction_hold_released'] == ['vacuum-off']
    assert 'left' not in engine.state['holds']
    run(engine, [{'slot': 'left', 'operation': 'withdraw', 'duration': .2}])


def test_injury_mode_records_a_torn_suction_patch():
    engine = SurgicalEngine(scene(extra={INJURY: {**injury_profile(), 'mode': 'injury'}}))
    results = run(engine, [*engage(), pull(6.2), pull(5.8), pull(5.4), pull(5.), pull(4.)])
    record = engine.state['injuryV1']
    tears = [e for e in record['events'] if e['kind'] == 'avulsion' and e['code'] == 'SUCTION_HOLD_EXCEEDED']
    assert tears and tears[0]['attachments'] == ['suction-patch:left'] and tears[0]['mechanism'] == 'traction'
    assert 'suction-patch:left' in record['avulsions'] and record['contused']
    assert any(e['kind'] == 'avulsion' for r in results for e in r['events'])


def test_suction_through_a_cottonoid_takes_fluid_without_holding_tissue():
    engine = SurgicalEngine(scene())
    tip = [14., 14., 6.6]
    # A placed cottonoid as the operative exposure records it (centre, radius, status).
    engine.state['operativeExposureV1'] = {'objects': [{'id': 'cottonoid-1', 'body': 'field', 'kind': 'cottonoid',
        'positionMm': [14., 14., 7.5], 'status': 'placed', 'heldBy': None, 'radiusMm': 1.45}]}
    wet = int(np.ravel_multi_index((14, 14, 7), engine.scene.shape))
    engine.state['fluid'][str(wet)] = .8
    results = run(engine, engage())
    assert 'left' not in engine.state['holds']
    assert 'suction_through_cottonoid' in kinds(results[-1])
    assert 'suction_hold_created' not in [k for r in results for k in kinds(r)]
    assert engine.state['fluidRemovedMm3'] == pytest.approx(.8) and not engine.state['fluid']
    assert engine.metrics()['suctionThroughCottonoidSeconds'] > 0
    assert st.cottonoid_over(engine, tip)['id'] == 'cottonoid-1'


def test_traction_replays_and_restores_deterministically():
    actions = [*engage(), pull(6.2), pull(5.8), pull(5.), pull(4.),
               {'slot': 'left', 'operation': 'apply', 'duration': .3}]
    engine = SurgicalEngine(scene())
    run(engine, actions)
    replayed = SurgicalEngine(scene()).replay(engine.actions)
    assert replayed.state_id == engine.state_id
    restored = SurgicalEngine.restore(engine.scene, json.loads(json.dumps(engine.snapshot())))
    assert restored.state_id == engine.state_id
    partial = SurgicalEngine(scene())
    run(partial, actions[:5])
    resumed = SurgicalEngine.restore(partial.scene, json.loads(json.dumps(partial.snapshot())))
    for action in actions[5:]:
        assert partial.step(action) == resumed.step(action)
    assert partial.state_id == engine.state_id


def test_retractor_reports_a_read_only_reaction_estimate():
    actions = [{'slot': 'right', 'operation': 'exchange', 'instrument': 'brain_retractor', 'duration': .1},
               {'slot': 'right', 'operation': 'place', 'duration': .1, 'pose': pose([4, 4, 3])},
               {'slot': 'right', 'operation': 'hold', 'duration': .2, 'pose': pose([4, 4, 6.])},
               {'slot': 'right', 'operation': 'move', 'duration': .4, 'pose': pose([4.4, 4, 6.])}]
    with_estimate = SurgicalEngine(scene())
    without = SurgicalEngine(scene(traction=False))
    a, b = run(with_estimate, actions), run(without, actions)
    estimate = [e for e in a[-1]['events'] if e['kind'] == 'tissue_reaction_estimate']
    assert estimate and estimate[0]['instrument'] == 'brain_retractor'
    assert estimate[0]['magnitudeN'] > 0 and estimate[0]['inputForceN'] == 1.
    # The estimate reads state only: deformation is identical with and without it.
    assert with_estimate.state['deformation'] == without.state['deformation']
