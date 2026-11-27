"""Public off-route specimen motions in a generated exterior clearance coupon."""

import numpy as np
import pytest

from scripts.generated_glioma_walkthrough_case import build_glioma_case
from resectionlab.surgical_engine import SurgicalEngine
from resectionlab.surgical_operative_exposure import _place_object
from resectionlab.surgical_rl_protection import GuardConfig, ProtectionAnnotation, ProtectionSet
from resectionlab.surgical_scene import SurgicalScene


def ijk(scene, xyz):
    return tuple(np.rint(scene.inverse[:3, :3] @ xyz + scene.inverse[:3, 3]).astype(int))


@pytest.fixture(scope='module')
def scene():
    original = build_glioma_case()[1]
    labels = original.labels.copy()
    # Sixteen connected cells let the remote specimen end encounter an obstacle
    # that lies clear of both operative forceps jaws and the complete shaft.
    for x in range(32, 48):
        point = ijk(original, [x, 25, 80])
        assert labels[point] == 0
        labels[point] = 9
    labels[ijk(original, [47, 26, 80])] = 3
    return SurgicalScene(
        labels,
        original.affine,
        original.regions,
        {'kind': 'generated-control', 'id': 'specimen-sweep-coupon'},
        original.properties,
    )


def action(engine, operation, xyz=None, *, hand='right', **extra):
    request = dict(operation=operation, slot=hand, duration=0.1, **extra)
    if xyz is not None:
        request['pose'] = dict(position=xyz, quaternion=[1.0, 0.0, 0.0, 0.0])
    return engine.step(request)


def grasped(scene, *, protect=False):
    options = {}
    if protect:
        mask = np.zeros(scene.shape, bool)
        mask[ijk(scene, [47, 25, 83])] = True
        annotation = ProtectionAnnotation.from_mask(
            scene, 'specimen-exclusion', mask, provenance='generated obstacle'
        )
        options = dict(
            protection=ProtectionSet(scene, [annotation], clearance_mm=0.0, required=()),
            guard_config=GuardConfig(require_critical_coverage=False),
        )
    engine = SurgicalEngine(scene, **options)
    # The independently generated exterior sample is exercised as a
    # post-exposure contact coupon. The full walkthrough verifies the opening.
    bodies = engine.state['operativeExposureV1']['bodies']
    for sheet in ('scalp', 'dura'):
        bodies[sheet].update(status='supported', angleRad=np.deg2rad(100.0).item())
    cfg = scene.properties['operativeExposureV1']['bodies']['bone']
    bodies['bone'].update(
        status='parked', translationMm=np.subtract(cfg['parkCenterMm'], cfg['restCenterMm']).tolist()
    )
    assert action(engine, 'exchange', instrument='tumor_forceps')['status'] == 'accepted'
    assert (
        action(engine, 'place', [32.0, 25.0, 85.0], parameters={'jawOpeningMm': 0.0})['status'] == 'accepted'
    )
    grasp = action(engine, 'grasp', [32.0, 25.0, 80.55])
    assert grasp['status'] == 'accepted' and len(engine.state['holds']['right']['cells']) == 16, grasp
    return engine


def test_carried_remote_specimen_end_cannot_enter_declared_clearance(scene):
    engine = grasped(scene, protect=True)
    before = engine.state_id
    out = action(engine, 'apply', [32.0, 25.0, 84.0])
    assert out['status'] == 'rejected' and 'PROTECTED_STRUCTURE: held specimen' in out['reason']
    assert engine.state_id == before


def test_specimen_follows_held_forceps_and_excludes_retained_objects(scene):
    engine = grasped(scene)
    assert action(engine, 'apply', [32.0, 25.0, 81.55])['status'] == 'accepted'
    assert engine.state['holds']['right']['offset'] == [0, 0, 1]
    _place_object(engine, 'field', 'cottonoid', [47.0, 25.0, 85.0])
    out = action(engine, 'move', [32.0, 25.0, 86.0])
    assert 'OPERATIVE_SPECIMEN_OBJECT_COLLISION:cottonoid' in out['reason']
    assert engine.state['holds']['right']['offset'][2] < 4
    out = action(engine, 'move', parameters={'jawOpeningMm': 2.0})
    assert 'OPERATIVE_RELEASE_SPECIMEN_BEFORE_OPENING_JAWS' in out['reason']


def test_opposite_working_tip_cannot_enter_held_specimen(scene):
    engine = grasped(scene)
    assert action(engine, 'exchange', hand='left', instrument='cortical_probe')['status'] == 'accepted'
    # Both whole instruments are free at this pose, while the remote electrode
    # can reach the tissue some distance from the gripping forceps.
    mount = action(engine, 'place', engine.state['slots']['left']['position'], hand='left')
    assert mount['status'] == 'accepted', mount
    for point in ([0.0, 8.0, 85.0], [45.0, 8.0, 85.0], [48.0, 25.0, 84.0]):
        moved = action(engine, 'move', point, hand='left')
        assert moved['status'] == 'accepted', moved
    out = action(engine, 'apply', [48.0, 25.0, 80.55], hand='left')
    assert out['reason'] == 'OPERATIVE_SPECIMEN_TOOL_COLLISION:left', out


def test_diagonal_translation_cannot_skip_intervening_occupied_voxel(scene):
    engine = grasped(scene)
    # Both snapped endpoints are clear at the remote tip. Its unit voxel
    # would sweep through the adjacent bone cell during the diagonal.
    out = action(engine, 'move', [32.0, 26.0, 81.55])
    assert out['reason'] == 'OPERATIVE_SPECIMEN_SWEPT_ANATOMY_COLLISION', out
    assert engine.state['holds']['right']['offset'] == [0, 0, 0]
