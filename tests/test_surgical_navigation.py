"""Opt-in navigation pointer: registered readout that knows nothing about brain shift."""
import json
import math

import numpy as np
import pytest

from scripts.generated_glioma_walkthrough_case import build_glioma_case
from resectionlab import surgical_brain_shift as shift
from resectionlab import surgical_navigation as navigation
from resectionlab.surgical_engine import SurgicalEngine
from resectionlab.surgical_scene import SurgicalScene, propagate_original_target_memberships
from resectionlab.surgical_tools import OPERATIVE_TOOLS, TOOLS

Q = [0., 0., 0., 1.]


def small_scene(**properties):
    labels = np.zeros((24, 24, 28), np.uint16)
    labels[:, :, 10:24] = 2
    labels[10:14, 10:14, 12:16] = 5
    regions = [dict(id=2, name='cortex', material='soft_tissue'),
               dict(id=5, name='lesion', material='tumor', target=True)]
    return SurgicalScene(labels, np.eye(4), regions, {'caseId': 'navigation-fixture',
                         'provenance': 'generated engineering geometry; not patient anatomy'}, properties)


def pointer_actions(at, slot='left'):
    pose = {'position': [float(v) for v in at], 'quaternion': Q}
    return [{'slot': slot, 'operation': 'exchange', 'instrument': 'navigation_pointer', 'duration': .1},
            {'slot': slot, 'operation': 'place', 'duration': .1, 'path': [pose]},
            {'slot': slot, 'operation': 'inspect', 'duration': .3}]


def test_pointer_is_added_only_with_the_property_and_catalogues_are_unchanged():
    assert 'navigation_pointer' not in TOOLS and 'navigation_pointer' not in OPERATIVE_TOOLS
    assert 'navigation_pointer' not in SurgicalEngine(small_scene()).tools
    engine = SurgicalEngine(small_scene(navigationV1=navigation.default_property()))
    assert 'navigation_pointer' in engine.tools
    assert engine.step(pointer_actions((5, 5, 4))[0])['status'] == 'accepted'
    plain = SurgicalEngine(small_scene())
    assert plain.step(pointer_actions((5, 5, 4))[0])['status'] == 'rejected'


def test_registration_error_has_the_declared_rms_over_many_seeds():
    scene = small_scene()
    errors = []
    for seed in range(400):
        decl = navigation.declaration(small_scene(navigationV1=navigation.default_property(seed=seed)))
        nav = navigation.Registration(scene, decl)
        errors.append(nav.to_image(nav.center) - nav.center)
        assert nav.to_room(nav.to_image([3., 4., 5.])) == pytest.approx([3., 4., 5.])
    rms = math.sqrt(np.mean(np.sum(np.square(errors), axis=1)))
    assert rms == pytest.approx(2.0, rel=0.1)
    # The draw is fixed by the seed, independent of the platform's random generator.
    a = navigation.Registration(scene, navigation.declaration(small_scene(navigationV1=navigation.default_property(seed=7))))
    b = navigation.Registration(scene, navigation.declaration(small_scene(navigationV1=navigation.default_property(seed=7))))
    assert np.array_equal(a.translation, b.translation)


def test_reading_reports_image_coordinate_and_keeps_truth_out_of_the_policy_observation():
    scene = small_scene(navigationV1=navigation.default_property(seed=3))
    engine = SurgicalEngine(scene)
    tip = np.array([11.5, 11.5, 8.0])
    results = [engine.step(action) for action in pointer_actions(tip)]
    assert [r['status'] for r in results] == ['accepted'] * 3
    screen = engine.state['observations'][-1]
    nav = navigation.registration(engine)
    assert screen['kind'] == 'navigation'
    assert screen['reportedImageRasMm'] == pytest.approx(nav.to_image(tip).tolist())
    assert screen['distanceToPlannedTumourMm'] == pytest.approx(
        np.min(np.linalg.norm(scene.points(np.flatnonzero(scene.labels.ravel() == 5)) - nav.to_image(tip), axis=1)))
    truth = engine.view()['navigationV1']['privilegedTruth']['readings'][-1]
    assert truth['registrationErrorMm'] == pytest.approx((nav.to_image(tip) - tip).tolist())
    assert truth['tissueUnderTip']['name'] == 'cortex'
    # Without a shift, navigation error is the registration error alone.
    assert truth['navigationErrorMm'] == pytest.approx(
        np.linalg.norm(nav.to_image(tip) - truth['truePreoperativeRasMm']))
    assert len(engine.state['observations']) == len(engine.state['navigationV1']['readings']) == 1
    policy = json.dumps(engine.observe())
    for latent in ('registrationErrorMm', 'privilegedTruth', 'truePreoperativeRasMm', 'tissueUnderTip', 'seed'):
        assert latent not in policy
    assert engine.metrics()['navigationReadings'] == 1
    again = SurgicalEngine(scene)
    assert [again.step(action)['stateId'] for action in pointer_actions(tip)] == [r['stateId'] for r in results]
    assert SurgicalEngine.restore(scene, engine.snapshot(), use_cache=False, force_verify=True).state_id == engine.state_id


def test_navigation_cannot_see_brain_shift():
    _, base, _ = build_glioma_case()
    properties = {**base.properties, 'navigationV1': navigation.default_property(seed=11),
                  'brainShiftV1': shift.default_property(base, settlingSeconds=1.0, settlingSteps=2)}
    scene = propagate_original_target_memberships(base, SurgicalScene(
        base.labels, base.affine, base.regions, base.source, properties))
    engine = SurgicalEngine(scene)
    bodies = engine.state['operativeExposureV1']['bodies']
    bone = scene.properties['operativeExposureV1']['bodies']['bone']
    bodies['scalp'].update(angleRad=math.pi / 2, status='supported')
    bodies['bone'].update(status='parked', translationMm=np.subtract(bone['parkCenterMm'], bone['restCenterMm']).tolist())
    bodies['dura'].update(angleRad=math.pi / 2, status='supported')
    assert engine.step({'slot': 'left', 'operation': 'inspect', 'duration': 1.5})['status'] == 'accepted'
    current = shift.active_map(engine)
    # Rest the pointer on the sunken cortex at the centre of the opening.
    top = int(np.max(np.flatnonzero(np.isin(scene.labels[62, 60, :], [7, 8, 9, 10]))))
    source = int(np.ravel_multi_index((62, 60, top), scene.shape))
    tip = scene.points([source])[0] + current.displacement_mm([source])[0] + [0., 0., 1.6]
    engine.state['slots']['right'].update(instrument='navigation_pointer', position=tip.tolist(), quaternion=Q)
    navigation.read_pointer(engine, 'right', tip, np.asarray(Q))
    screen = engine.state['observations'][-1]
    truth = engine.state['navigationV1']['readings'][-1]
    nav = navigation.registration(engine)
    # Navigation reports the unshifted image coordinate of the tip ...
    assert screen['reportedImageRasMm'] == pytest.approx(nav.to_image(tip).tolist())
    # ... while the tissue under the tip came from about the sag height above it.
    sag = np.asarray(truth['tissueUnderTip']['shiftMm'])
    assert np.linalg.norm(sag) > 4.0 and sag[2] < 0
    assert truth['navigationErrorMm'] > np.linalg.norm(truth['registrationErrorMm'])
    assert truth['nearestTumourEdge']['discrepancyMm'] > 1.0
    assert 'brain' not in json.dumps(screen)
