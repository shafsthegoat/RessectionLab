"""Versioned operative CUSA envelopes preserve replay and contact budgets.

Generated mechanism coupons are software contracts, not calibrated surgical
geometry, tissue properties, or patient validation.
"""
import json

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from resectionlab.surgical_engine import SurgicalEngine
from resectionlab.surgical_physics_v2 import default_profile
from resectionlab.surgical_rl_protection import GuardConfig, ProtectionAnnotation, ProtectionSet
from resectionlab.surgical_scene import SurgicalScene


def coupon(*, block=False, point=None, points=(), material='tumor', protected=False,
           properties=None, obstacles=()):
    labels = np.zeros((24, 24, 32), np.uint16)
    if block:
        labels[5:12, 5:12, 10:18] = 1
    if point is not None:
        labels[point] = 1
    for location in points:
        labels[location] = 1
    for location in obstacles:
        labels[location] = 2
    regions = [{'id': 1, 'name': 'Target coupon', 'material': material, 'target': material == 'tumor'}]
    if obstacles:
        regions.append({'id': 2, 'name': 'Retained obstacle', 'material': 'bone'})
    scene = SurgicalScene(labels, np.eye(4), regions,
        {'kind': 'generated operative CUSA geometry coupon'},
        {'tissuePhysicsV2': default_profile(), 'attachments': [], **(properties or {})})
    options = {}
    if protected:
        annotation = ProtectionAnnotation.from_mask(scene, 'brainstem', labels == 1,
            provenance='generated exclusion coupon; no patient anatomy')
        options = {'protection': ProtectionSet(scene, [annotation], clearance_mm=0.),
                   'guard_config': GuardConfig()}
    return SurgicalEngine(scene, **options)


def pose(position, quaternion=(0., 0., 0., 1.)):
    return {'position': list(position), 'quaternion': list(quaternion)}


def action(operation, *, slot='left', duration=.1, **kwargs):
    return {'slot': slot, 'operation': operation, 'duration': duration, **kwargs}


def accepted(engine, operation, **kwargs):
    result = engine.step(action(operation, **kwargs))
    assert result['status'] == 'accepted', result
    return result


def prepare(engine, revision=2, position=(8., 8., 4.)):
    options = {} if revision == 1 else {'mechanicsRevision': revision}
    accepted(engine, 'configure', **options)
    accepted(engine, 'exchange', instrument='cusa')
    accepted(engine, 'place', pose=pose(position))


def test_default_revision1_replays_prechange_golden_state_and_envelopes():
    # Frozen from the unmodified revision1 engine before revision2 was added.
    engine = coupon(block=True)
    prepare(engine, revision=1)
    assert engine.scene.identity == 'sha256:f4ec66744472986e0d88f8f9caec14b3e23ccd9c090d5d0c297492ef45060a1a'
    assert engine.state['toolMechanicsRevision'] == 1
    expected = [
        'sha256:c93a139e1ac9c20f0afb8f81932b59ff1920fb87e19b2d06afa1e3b3fbc9e23e',
        'sha256:b1c3c041026649c38d4bb6e27e3fe6ac97c0e209814e7b8cd77cc9b557cd6c69',
        'sha256:9bce4ac4d8cf289a3e2115d2876e815daa657cfd517c1f21ff015229afcd7a55',
    ]
    for identity in expected:
        result = engine.step(action('apply', duration=30., pose=pose((8., 8., 20.))))
        assert result['status'] == 'partial'
        assert result['reasonCode'] == 'RETAINED_TISSUE_COLLISION'
        assert engine.state_id == identity
    assert engine.state['removed'] == [5610, 5611, 5642, 5643, 5674, 5675, 6378, 6379,
        6410, 6411, 6442, 6443, 7146, 7147, 7178, 7179, 7210, 7211]
    assert engine.state['slots']['left']['position'][2] == pytest.approx(9.177124343338498)
    assert engine.metrics()['cusaAspiratedSolidMm3'] == 18.
    assert engine.metrics()['cusaInputCapacityMm3'] == pytest.approx(37.2)
    geometry = engine.observe()['slots']['left']['contactGeometry']
    assert geometry == [
        {'a': [0., 0., -40.], 'b': [0., 0., -2.], 'radiusMm': .5, 'part': 'shaft'},
        {'a': [0., 0., -2.], 'b': [0., 0., 0.], 'radiusMm': 1.5, 'part': 'working'},
    ]
    saved = json.loads(json.dumps(engine.snapshot()))
    assert SurgicalEngine.restore(engine.scene, saved).snapshot() == saved


def test_revision2_emits_ball_and_continuous_shaft_and_replays_explicit_choice():
    engine = coupon()
    prepare(engine)
    observed = engine.observe()
    assert engine.state['toolMechanicsRevision'] == observed['toolMechanicsRevision'] == 2
    assert engine.actions[0]['mechanicsRevision'] == 2
    assert engine.parameters(engine.state['slots']['left'])['contactGeometryRevision'] == 2
    assert observed['slots']['left']['contactGeometry'] == [
        {'a': [0., 0., -40.], 'b': [0., 0., 0.], 'radiusMm': .5, 'part': 'shaft'},
        {'a': [0., 0., 0.], 'b': [0., 0., 0.], 'radiusMm': 1.5, 'part': 'working'},
    ]
    saved = json.loads(json.dumps(engine.snapshot()))
    assert SurgicalEngine.restore(engine.scene, saved).snapshot() == saved
    before = engine.snapshot()
    # The geometry discriminator belongs to the engine, not user tool settings.
    result = engine.step(action('apply', parameters={'contactGeometryRevision': 1}))
    assert result['status'] == 'rejected'
    assert engine.snapshot() == before


def test_revision2_tip_led_advance_only_aspirates_current_contacts_at_unchanged_rate(monkeypatch):
    import resectionlab.surgical_cusa_v3 as cusa

    engine = coupon(block=True)
    source = engine.scene.labels.copy()
    prepare(engine)
    original = cusa.apply
    calls = []

    def checked_apply(current, cells, dt, params, **kwargs):
        before = set(current.state['removed'])
        contacts = set(current.touched(kwargs['position'], kwargs['position'], 1.5))
        result = original(current, cells, dt, params, **kwargs)
        removed = set(current.state['removed']) - before
        assert removed <= contacts
        assert params['radiusMm'] == 1.5
        calls.append((dt, len(removed)))
        return result

    monkeypatch.setattr(cusa, 'apply', checked_apply)
    for _ in range(3):
        result = engine.step(action('apply', duration=30., pose=pose((8., 8., 16.))))
        assert result['status'] in ('accepted', 'partial'), result
        assert result.get('reasonCode') in (None, 'TISSUE_ASPIRATION_IN_PROGRESS'), result
    assert calls and engine.metrics()['targetRemovedMm3'] > 18.
    assert engine.state['slots']['left']['position'][2] > 10.
    assert engine.metrics()['nonTargetRemovedMm3'] == 0.
    ledger = engine.state['cusaV3']
    metrics = engine.metrics()
    assert cusa.profile(engine)['capacityMm3PerSecond'] == 2.
    assert ledger['inputCapacityMm3'] == pytest.approx(2. * ledger['effectiveActivationSeconds'])
    assert metrics['cusaInputCapacityMm3'] == pytest.approx(
        metrics['cusaAspiratedSolidMm3'] + metrics['cusaPendingCapacityMm3'] +
        metrics['cusaUnusedCapacityMm3'] + metrics['cusaDiscardedCapacityMm3'])
    assert metrics['cusaAspiratedSolidMm3'] <= metrics['cusaInputCapacityMm3'] + 1e-9
    np.testing.assert_array_equal(engine.scene.labels, source)


@pytest.mark.parametrize('material', ['tumor', 'bone'])
def test_revision2_free_tip_cannot_translate_shaft_through_retained_material(material):
    engine = coupon(point=(8, 8, 10), material=material)
    prepare(engine, position=(5., 8., 20.))
    accepted(engine, 'move', pose=pose((6.6, 8., 20.)))
    assert not engine.touched((8., 8., 20.), (8., 8., 20.), 1.5)
    result = engine.step(action('apply', pose=pose((8., 8., 20.))))
    assert result['status'] in ('partial', 'rejected')
    assert result['reasonCode'] == 'SHAFT_COLLISION'
    assert not engine.state['removed'] and engine.metrics()['cusaAspiratedSolidMm3'] == 0.
    assert engine.state['slots']['left']['position'][0] < 7.
    # A later action at the obstruction cannot commit its colliding microstep.
    before = engine.snapshot()
    result = engine.step(action('apply', pose=pose((8., 8., 20.))))
    assert result['status'] == 'rejected' and result['reasonCode'] == 'SHAFT_COLLISION'
    assert engine.snapshot() == before


def test_revision2_free_tip_rotation_keeps_full_shaft_collision():
    engine = coupon(point=(8, 8, 10))
    prepare(engine, position=(10.3, 8., 20.))
    quaternion = Rotation.from_euler('y', 20., degrees=True).as_quat()
    result = engine.step(action('apply', pose=pose((10.3, 8., 20.), quaternion)))
    assert result['status'] in ('partial', 'rejected')
    assert result['reasonCode'] == 'SHAFT_COLLISION'
    assert not engine.state['removed'] and engine.metrics()['cusaAspiratedSolidMm3'] == 0.


def test_revision2_protected_shaft_sweep_remains_blocked_with_free_tip():
    engine = coupon(point=(8, 8, 10), protected=True)
    prepare(engine, position=(5., 8., 20.))
    accepted(engine, 'move', pose=pose((6.6, 8., 20.)))
    before = engine.snapshot()
    result = engine.step(action('apply', pose=pose((8., 8., 20.))))
    assert result['status'] in ('partial', 'rejected')
    assert result['reasonCode'] == 'PROTECTED_STRUCTURE'
    assert not engine.state['removed']
    assert engine.metrics()['cusaAspiratedSolidMm3'] == 0.
    if result['status'] == 'rejected':
        assert engine.snapshot() == before


def test_revision2_tip_cannot_aspirate_unsupported_material():
    engine = coupon(point=(8, 8, 10), material='bone')
    prepare(engine)
    result = engine.step(action('apply', duration=2., pose=pose((8., 8., 12.))))
    assert result['status'] in ('partial', 'rejected')
    assert result['reasonCode'] == 'UNSUPPORTED_MATERIAL'
    assert not engine.state['removed'] and engine.metrics()['cusaAspiratedSolidMm3'] == 0.


def test_revision2_two_tool_collision_still_blocks_overlap_transactionally():
    engine = coupon()
    prepare(engine, position=(8., 8., 20.))
    accepted(engine, 'exchange', slot='right', instrument='frazier_suction')
    accepted(engine, 'place', slot='right', pose=pose((11.1, 8., 20.)))
    before = engine.snapshot()
    result = engine.step(action('apply', slot='right', pose=pose((8., 8., 20.))))
    assert result['status'] == 'rejected' and result['reasonCode'] == 'TOOL_COLLISION'
    assert engine.snapshot() == before


CATALOGUE_MECHANISMS = [
    ('cranial_drill', 'bone'), ('cusa', 'tumor'),
    ('brain_retractor', 'soft_tissue'), ('scalpel', 'soft_tissue'),
    ('periosteal_elevator', 'soft_tissue'), ('craniotome', 'bone'),
    ('dural_scissors', 'membrane'), ('dissector', 'soft_tissue'),
    ('frazier_suction', 'soft_tissue'), ('bipolar_forceps', 'vessel'),
    ('micro_scissors', 'vessel'), ('tumor_forceps', 'tumor'),
    ('irrigation', 'soft_tissue'), ('cortical_probe', 'soft_tissue'),
    ('subcortical_probe', 'soft_tissue'),
]


def catalogue_prepare(engine, tool, **parameters):
    accepted(engine, 'configure', mechanicsRevision=2)
    accepted(engine, 'exchange', instrument=tool)
    accepted(engine, 'place', pose=pose((8., 8., 5.)), parameters={
        'radiusMm': 1.5, 'jawOpeningMm': 0., 'forceN': 0., **parameters})


def source_cell(engine, location):
    return int(np.ravel_multi_index(location, engine.scene.shape))


@pytest.mark.parametrize('tool,material', CATALOGUE_MECHANISMS)
def test_revision2_public_catalogue_mechanisms_retain_their_effect_contract(tool, material):
    from resectionlab.surgical_tools import TOOLS

    assert {entry[0] for entry in CATALOGUE_MECHANISMS} == set(TOOLS)
    line = ((8, 7, 10), (8, 8, 10), (8, 9, 10))
    point = (8, 8, 10)
    identifier = int(np.ravel_multi_index(point, (24, 24, 32)))
    properties = {}
    if tool in ('periosteal_elevator', 'dissector'):
        kind = 'periosteal' if tool == 'periosteal_elevator' else 'soft'
        properties['attachments'] = [
            {'id': 'matching', 'kind': kind, 'cells': [identifier]},
            {'id': 'other', 'kind': 'soft' if kind == 'periosteal' else 'periosteal',
             'cells': [identifier]}]
    if tool in ('cortical_probe', 'subcortical_probe'):
        probe = 'cortical' if tool == 'cortical_probe' else 'subcortical'
        evidence = {'id': 'declared-site', 'cell': identifier, 'probe': probe,
                    'result': 'synthetic response', 'uncertainty': 'coupon only'}
        properties['functionalEvidence'] = [evidence]
    points = line if tool in ('scalpel', 'dural_scissors', 'micro_scissors', 'craniotome') else (point,)
    engine = coupon(points=points, material=material, properties=properties)
    source = engine.scene.labels.copy()
    catalogue_prepare(engine, tool)
    assert engine.state['toolMechanicsRevision'] == 2
    geometry = engine.observe()['slots']['left']['contactGeometry']
    assert geometry and all(part['radiusMm'] > 0. for part in geometry)

    if tool == 'cranial_drill':
        result = engine.step(action('apply', duration=.1, pose=pose((8., 8., 12.)),
            parameters={'rpm': 1000., 'forceN': .1}))
        assert result['status'] == 'partial' and not engine.state['removed']
        assert sum(engine.state['drillingV2']['workJ'].values()) > 0.
        accepted(engine, 'apply', duration=1., pose=pose((8., 8., 12.)),
                 parameters={'rpm': 40000., 'forceN': 1.})
        assert engine.state['removed'] == [identifier]
        assert engine.state['debrisMm3'] == engine.scene.voxel_mm3
        ledger = engine.state['drillingV2']
        assert ledger['inputEnergyJ'] == pytest.approx(ledger['cuttingWorkJ'] + ledger['unusedEnergyJ'])
    elif tool == 'cusa':
        accepted(engine, 'apply', duration=2., pose=pose((8., 8., 12.)))
        assert engine.state['removed'] == [identifier]
        assert engine.state['cusaV3']['aspiratedSolidMm3'] == engine.scene.voxel_mm3
        assert engine.state['debrisMm3'] == 0.
    elif tool in ('scalpel', 'dural_scissors'):
        accepted(engine, 'apply', pose=pose((8., 8., 9.8)))
        assert engine.state['cutEdges'] and not engine.state['removed']
    elif tool == 'micro_scissors':
        accepted(engine, 'apply', pose=pose((8., 8., 9.8)))
        assert engine.state['transectedEdges'] and engine.state['vesselsV2']['wounds']
        vessel = engine.state['vesselsV2']
        assert vessel['initialMm3'] + vessel['refilledMm3'] == pytest.approx(
            vessel['remainingMm3'] + vessel['generatedMm3'])
        assert not engine.state['removed']
    elif tool in ('periosteal_elevator', 'dissector'):
        accepted(engine, 'apply', pose=pose((8., 8., 9.8)))
        assert engine.state['separatedAttachments'] == ['matching']
        assert not engine.state['removed']
    elif tool == 'craniotome':
        accepted(engine, 'apply', pose=pose((8., 8., 9.8)))
        assert engine.state['removed'] == [identifier]
        assert engine.state['debrisMm3'] == engine.scene.voxel_mm3
        assert len(engine.state['flapComponents']) == 2
    elif tool == 'brain_retractor':
        accepted(engine, 'hold', pose=pose((8., 8., 9.8)))
        accepted(engine, 'move', duration=.4, pose=pose((8.3, 8., 9.8)))
        displacement = max(np.linalg.norm(value) for value in engine.state['deformation'].values())
        assert displacement > 0. and not engine.state['removed']
        accepted(engine, 'release', duration=.2)
        assert max(np.linalg.norm(value) for value in engine.state['deformation'].values()) < displacement
        assert not engine.state['holds']
    elif tool == 'bipolar_forceps':
        accepted(engine, 'apply', pose=pose((8., 8., 9.8)), parameters={'power': 1.})
        assert sum(engine.state['thermal'].values()) > 0.
        assert not engine.state['removed']
    elif tool == 'tumor_forceps':
        accepted(engine, 'grasp', pose=pose((8., 8., 9.8)))
        assert engine.state['holds']['left']['cells'] == [identifier]
        accepted(engine, 'release')
        assert engine.occupancy()[identifier] == identifier + 1
        assert not engine.state['removed']
    elif tool in ('frazier_suction', 'irrigation'):
        accepted(engine, 'apply', pose=pose((8., 8., 7.5)),
                 parameters={'flowMm3PerSecond': 0.})
        assert engine.state['fluidAddedMm3'] == engine.state['fluidRemovedMm3'] == 0.
        assert not engine.state['removed'] and not engine.state['debris']
    else:
        accepted(engine, 'apply', pose=pose((8., 8., 9.8)), parameters={
            'amplitudeMa': 2., 'frequencyHz': 60., 'pulseWidthUs': 100.})
        accepted(engine, 'inspect')
        assert engine.state['observations'][-1]['evidence'] == [evidence]
        assert not engine.state['thermal'] and not engine.state['removed']
    np.testing.assert_array_equal(engine.scene.labels, source)
    assert SurgicalEngine.restore(engine.scene, engine.snapshot()).state_id == engine.state_id


def test_revision2_forceps_release_regrasp_and_checked_exit_preserve_one_source_owner():
    engine = coupon(point=(8, 8, 10), obstacles=((20, 20, 5), (20, 20, 15)))
    catalogue_prepare(engine, 'tumor_forceps')
    source = source_cell(engine, (8, 8, 10))
    original = engine.scene.labels.copy()
    accepted(engine, 'grasp', pose=pose((8., 8., 9.8)))
    before = engine.snapshot()
    result = engine.step(action('extract'))
    assert result['status'] == 'rejected'
    assert result['reason'] == 'MOVE_FRAGMENT_THROUGH_EXIT_BEFORE_EXTRACTION'
    assert engine.snapshot() == before
    accepted(engine, 'move', pose=pose((8., 8., 7.8)))
    destination = source_cell(engine, (8, 8, 8))
    accepted(engine, 'release')
    accepted(engine, 'inspect', duration=.3)
    assert not engine.state['removed'] and not engine.state['holds']
    assert engine.occupancy()[source] == 0 and engine.occupancy()[destination] == source + 1
    assert np.count_nonzero(np.asarray(engine.occupancy()) == source + 1) == 1
    assert [source, destination] in engine.view()['moved']
    resumed = SurgicalEngine.restore(engine.scene, engine.snapshot())
    accepted(engine, 'grasp')
    assert engine.state['holds']['left']['cells'] == [source]
    assert not engine.state.get('detachedFragments')
    before = engine.snapshot()
    result = engine.step(action('extract'))
    assert result['status'] == 'rejected' and result['reason'] == 'FRAGMENT_NOT_AT_SOURCE_EXTERIOR_EXIT'
    assert engine.snapshot() == before
    accepted(engine, 'move', pose=pose((8., 8., 3.8)))
    assert np.count_nonzero(np.asarray(engine.occupancy()) == source + 1) == 1
    accepted(engine, 'extract')
    assert engine.state['removed'] == [source] and not engine.state['holds']
    assert engine.state['evacuatedForcepsMm3'] == engine.scene.voxel_mm3
    assert np.count_nonzero(np.asarray(engine.occupancy()) == source + 1) == 0
    before = engine.snapshot()
    assert engine.step(action('extract'))['status'] == 'rejected'
    assert engine.snapshot() == before
    for request in engine.actions[len(resumed.actions):]:
        assert resumed.step(request)['status'] == 'accepted'
    assert resumed.state_id == engine.state_id
    np.testing.assert_array_equal(engine.scene.labels, original)


@pytest.mark.parametrize('blocked', [False, True])
def test_revision2_forceps_observed_jaws_follow_actual_checked_aperture(tmp_path, blocked):
    from resectionlab.surgical_frame_replay import SurgicalFrameCache
    from resectionlab.surgical_replay_cache import SurgicalReplayCache

    engine = coupon(point=(8, 8, 10), obstacles=((11, 8, 5),) if blocked else ())
    catalogue_prepare(engine, 'tumor_forceps')
    boundary = len(engine.actions)
    result = engine.step(action('apply', duration=1.2, parameters={'jawOpeningMm': 8.}))
    actual = engine.state['slots']['left']['parameters']['jawOpeningMm']
    if blocked:
        assert result['status'] == 'partial' and result['reasonCode'] == 'UNSUPPORTED_MATERIAL'
        assert 0. < actual < 8.
    else:
        assert result['status'] == 'accepted' and actual == 8.
    saved = engine.snapshot()
    frames = SurgicalFrameCache(directory=tmp_path)
    replay = SurgicalReplayCache()
    _, _, first = frames.seek(engine, saved, boundary, 0, replay)
    apertures = []
    for index in range(first['frameCount']):
        frame, _, info = frames.seek(engine, saved, boundary, index, replay)
        observed = frame.observe()['slots']['left']
        jaw = observed['parameters']['jawOpeningMm']
        apertures.append(jaw)
        working = [part for part in observed['contactGeometry'] if part['part'] == 'working']
        assert sorted(part['a'][0] for part in working) == pytest.approx([-jaw / 2, jaw / 2])
        assert info['frameStateId'] == frame.state_id
    assert len(apertures) > 2 and apertures == sorted(apertures)
    assert apertures[0] == 0. and apertures[-1] == actual
    assert any(0. < jaw < actual for jaw in apertures)
    assert frame.snapshot() == saved and engine.snapshot() == saved


def test_revision2_drill_irrigation_and_frazier_conserve_solid_debris_and_fluid():
    engine = coupon(point=(8, 8, 10), material='bone')
    catalogue_prepare(engine, 'cranial_drill')
    contact = engine.step(action('apply', duration=.1, pose=pose((8., 8., 12.)),
        parameters={'rpm': 1000., 'forceN': .1}))
    assert contact['status'] == 'partial' and contact['reasonCode'] == 'BONE_CUTTING_IN_PROGRESS'
    accepted(engine, 'apply', duration=1., pose=pose((8., 8., 12.)),
             parameters={'rpm': 40000., 'forceN': 1.})
    assert len(engine.state['removed']) == 1 and engine.state['debrisMm3'] == 1.
    accepted(engine, 'withdraw')
    accepted(engine, 'exchange', instrument='irrigation')
    accepted(engine, 'place', pose=pose((8., 8., 5.)),
             parameters={'flowMm3PerSecond': 8., 'depthMm': 2.})
    accepted(engine, 'move', pose=pose((8., 8., 8.)))
    accepted(engine, 'apply', duration=.5)
    assert engine.state['fluidAddedMm3'] == pytest.approx(4.)
    accepted(engine, 'withdraw')
    accepted(engine, 'exchange', instrument='frazier_suction')
    accepted(engine, 'place', pose=pose((8., 8., 5.)),
             parameters={'radiusMm': 3., 'flowMm3PerSecond': 20.})
    accepted(engine, 'move', pose=pose((8., 8., 10.)))
    accepted(engine, 'apply', duration=1.)
    metrics = engine.metrics()
    assert metrics['fluidRemovedMm3'] > 0. and metrics['debrisRemovedMm3'] > 0.
    assert metrics['fluidAddedMm3'] == pytest.approx(metrics['fluidRemovedMm3'] +
        metrics['fluidPresentMm3'] + metrics['fluidEscapedMm3'])
    assert metrics['drillRemovedVolumeMm3'] == pytest.approx(metrics['debrisMm3'] +
        metrics['debrisRemovedMm3'] + metrics['debrisEscapedMm3'])
    assert len(engine.state['removed']) == 1
    assert SurgicalEngine.restore(engine.scene, engine.snapshot()).state_id == engine.state_id
