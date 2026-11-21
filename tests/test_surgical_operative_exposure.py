"""Focused material regressions for the generated three-layer operative scene."""

import math

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from scripts.generated_glioma_walkthrough_case import build_glioma_case
from resectionlab.surgical_engine import SurgicalEngine
from resectionlab import surgical_operative_exposure as exposure


@pytest.fixture(scope='module')
def scene():
    return build_glioma_case()[1]


def test_sheet_curvature_keeps_attached_edge_and_whole_sweep_clear(scene):
    engine = SurgicalEngine(scene)
    name = 'scalp'
    cfg = scene.properties[exposure.PROPERTY]['bodies'][name]
    state = engine.state[exposure.PROPERTY]['bodies'][name]
    ids, rest, lookup, *_ = exposure.rest(engine, name)
    distal = lookup[cfg['gripCell']]
    hinge = [lookup[c] for c, _ in cfg['hingeEdges']]
    hand = engine.state['slots']['left']
    hand.update(
        instrument='membrane_forceps',
        position=exposure.grip_trajectory(scene, name, 0.0),
        quaternion=[1.0, 0.0, 0.0, 0.0],
        parameters={'jawOpeningMm': 0.0},
    )
    state['status'] = 'incised'
    exposure._grasp(
        engine, 'left', name, np.asarray(hand['position']), np.asarray(hand['quaternion']), [cfg['gripCell']]
    )
    for angle in np.linspace(0.0, math.radians(100), 101)[1:]:
        next_state = dict(state, angleRad=float(angle))
        grip = np.asarray(exposure.grip_trajectory(scene, name, angle))
        exposure._clear_body(engine, name, 'left', state, next_state, grip, np.array([1.0, 0.0, 0.0, 0.0]))
        state.update(next_state)
        hand['position'] = grip.tolist()
    positions = exposure.centers(engine, name)
    assert np.max(np.linalg.norm(positions[hinge] - rest[hinge], axis=1)) < 1e-9
    assert np.linalg.norm(positions[distal] - rest[distal]) > 20.0
    assert exposure.geometry_key(engine)[0][0] == pytest.approx(math.radians(100))
    assert engine.occupancy().material_ids([dict(exposure.moved(engine))[int(ids[distal])]]) == [
        int(ids[distal])
    ]
    with pytest.raises(ValueError, match='HINGE_LIMIT'):
        exposure.solve_angle(engine, name, grip + [8.0, 0.0, 0.0])


def test_working_plane_and_closed_jaw_are_required_for_local_seam(scene):
    engine = SurgicalEngine(scene)
    cfg = scene.properties[exposure.PROPERTY]['bodies']['dura']
    body = engine.state[exposure.PROPERTY]['bodies']['dura']
    engine.state[exposure.PROPERTY]['bodies']['bone']['status'] = 'parked'
    slot = engine.state['slots']['right']
    slot.update(instrument='dural_scissors', parameters={'jawOpeningMm': 1.0})
    edge = cfg['seamEdges'][len(cfg['seamEdges']) // 2]
    ends = scene.points(edge)
    pos = ends.mean(axis=0)
    forward = np.array([0.0, 0.0, -1.0])
    normal = ends[1] - ends[0]
    orientation = Rotation.from_matrix(np.column_stack((np.cross(normal, forward), normal, forward)))
    assert exposure.interact(engine, 'right', 'apply', pos, orientation.as_quat(), True, 0.1, edge) == []
    slot['parameters']['jawOpeningMm'] = 0.0
    rolled = orientation * Rotation.from_rotvec([0.0, 0.0, math.pi / 2])
    exposure.interact(engine, 'right', 'apply', pos, rolled.as_quat(), True, 0.1, edge)
    assert cfg['seamEdges'].index(edge) not in body['cut']
    exposure.interact(engine, 'right', 'apply', pos, orientation.as_quat(), True, 0.1, edge)
    assert cfg['seamEdges'].index(edge) in body['cut'] and len(body['cut']) < len(cfg['seamEdges'])
    assert sorted(edge) in engine.state['cutEdges']
    assert body['status'] == 'attached'


def test_portable_counts_and_real_closure_face(scene):
    engine = SurgicalEngine(scene)
    cfg = scene.properties[exposure.PROPERTY]['bodies']['dura']
    body = engine.state[exposure.PROPERTY]['bodies']['dura']
    body['status'] = 'seated'
    anchors = cfg['anchorCells']
    for i in anchors:
        point = scene.points([i])[0]
        assert exposure._anchors(engine, 'needle_driver', 'dura', 'right', point, {i})
    assert body['status'] == 'closed'
    for obj in exposure.summary(engine)['objects']:
        assert obj['kind'] == 'suture'
        assert obj['marginErrorMm'] == 0.0
        assert np.linalg.norm(np.subtract(*obj['endpointsMm'])) >= 1.0
        assert obj['positionMm'][2] == scene.points([obj['sourceCell']])[0, 2] + 0.5
    cargo = {
        'id': 'cottonoid-3',
        'body': 'field',
        'kind': 'cottonoid',
        'heldBy': 'left',
        'radiusMm': 1.0,
        'status': 'carried',
        'positionMm': [32.0, 32.0, 60.0],
        'offset': [0.0, 0.0, 0.0],
    }
    engine.state[exposure.PROPERTY]['objects'].append(cargo)
    assert exposure.holding(engine, 'left')
    with pytest.raises(ValueError, match='REQUIRES_EXTERIOR'):
        exposure.release(engine, 'left')
    cargo['positionMm'] = [32.0, 32.0, 75.0]
    exposure.release(engine, 'left')
    assert not exposure.holding(engine, 'left')
    assert exposure.summary(engine)['objects'][-1]['positionMm'] == [32.0, 32.0, 75.0]
    assert cargo['status'] == 'retrieved'


def test_registered_traction_material_response_and_timestep(scene):
    """Isolate the post-incision material response with identical prescribed motion."""
    measurements = []
    for count in (45, 90):
        engine = SurgicalEngine(scene)
        cfg = scene.properties[exposure.PROPERTY]['bodies']['scalp']
        body = engine.state[exposure.PROPERTY]['bodies']['scalp']
        body.update(status='incised', cut=list(range(len(cfg['seamEdges']))))
        hand = engine.state['slots']['left']
        orientation = np.array([1.0, 0.0, 0.0, 0.0])
        grip = np.asarray(exposure.grip_trajectory(scene, 'scalp', 0.0))
        start = grip + [0, 0, 0.65]
        hand.update(
            instrument='membrane_forceps',
            quaternion=orientation.tolist(),
            position=start.tolist(),
            parameters={'jawOpeningMm': 0.0, 'forceN': 1.2},
        )
        assert exposure.interact(engine, 'left', 'grasp', start, orientation, True, 0.05, [cfg['gripCell']])
        for angle in np.linspace(0.0, math.radians(100.0), count + 1)[1:]:
            point = np.asarray(exposure.grip_trajectory(scene, 'scalp', angle)) + [0, 0, 0.65]
            exposure.interact(engine, 'left', 'move', point, orientation, False, 3.0 / count, [])
            hand['position'] = point.tolist()
        measurements.append(
            [body['angleRad'], body['momentNmm'], *exposure.source_position(engine, 'scalp', cfg['gripCell'])]
        )
        hand['parameters']['jawOpeningMm'] = 1.0
        with pytest.raises(ValueError, match='BEFORE_OPENING_JAWS'):
            exposure.interact(engine, 'left', 'move', point, orientation, False, 0.05, [])
        hand['parameters']['jawOpeningMm'] = 0.0
        with pytest.raises(ValueError, match='REQUIRES_STAY'):
            exposure.release(engine, 'left')
        cell = cfg['anchorCells'][0]
        near = exposure.source_position(engine, 'scalp', cell)
        assert exposure._anchors(engine, 'raney_clip_applier', 'scalp', 'right', near, {cell})
        exposure.release(engine, 'left')
        assert body['status'] == 'supported'
        assert not exposure.holding(engine, 'left')
    assert np.allclose(measurements[0], measurements[1], rtol=0, atol=1e-5)
    assert measurements[0][1] == pytest.approx(
        1.8 * math.radians(100.0) + 0.15 * math.radians(100.0) / 3, abs=1e-5
    )


def test_cutting_work_requires_local_perforation_and_releases_only_complete_bone(scene):
    engine = SurgicalEngine(scene)
    cfg = scene.properties[exposure.PROPERTY]['bodies']['bone']
    body = engine.state[exposure.PROPERTY]['bodies']['bone']
    engine.state[exposure.PROPERTY]['bodies']['scalp']['angleRad'] = math.radians(100.0)
    params = {'forceN': 7.0, 'radiusMm': 0.75, 'duty': 0.8, 'rpm': 60000.0}
    kerf = set(cfg['kerfCells'])
    with pytest.raises(ValueError, match='PILOT_REQUIRED'):
        exposure._cut_bone(engine, 'craniotome', {min(kerf)}, 0.1, params)
    port = set(cfg['portCells'])
    params['radiusMm'] = 2.0
    exposure._cut_bone(engine, 'cranial_drill', port, 0.001, params)
    assert engine.state['removed'] == []
    assert engine.state['drillingV2']['cuttingWorkJ'] > 0.0
    exposure._cut_bone(engine, 'cranial_drill', port, 10.0, params)
    assert set(engine.state['removed']) == port and body['status'] == 'attached'
    # Feed successive real source contacts from the eroded perforation along
    # retained neighbors. This measures local work and connectivity, upstream
    # guard/footplate clearance is separately verified by the public episode.
    params['radiusMm'] = 0.75
    for _ in range(100):
        frontier = kerf - set(engine.state['removed'])
        if not frontier:
            break
        before = len(engine.state['removed'])
        exposure._cut_bone(engine, 'craniotome', frontier, 10.0, params)
        assert len(engine.state['removed']) > before
        attached = {j for _, j in cfg['seamEdges']} - set(engine.state['removed'])
        if attached:
            assert body['status'] == 'attached'
    assert body['status'] == 'free'
    assert len(body['cut']) == len(cfg['seamEdges'])
    assert not set(cfg['cells']) & set(engine.state['removed'])
    assert engine.state['debrisMm3'] == pytest.approx(len(port | kerf) * scene.voxel_mm3)


def test_unpowered_and_uncleared_kerf_stop_without_moving_source(scene):
    engine = SurgicalEngine(scene)
    config = scene.properties[exposure.PROPERTY]['bodies']['bone']
    engine.state[exposure.PROPERTY]['bodies']['scalp']['angleRad'] = math.radians(100)
    params = {'rpm': 0.0, 'forceN': 7.0, 'duty': 1.0, 'radiusMm': 0.75}
    with pytest.raises(ValueError, match='KERF_DRIVE_REQUIRED'):
        exposure._cut_bone(engine, 'craniotome', {config['kerfCells'][0]}, 0.1, params)
    assert engine.state['removed'] == []
    center = np.asarray(config['restCenterMm'])
    q = np.array([1.0, 0.0, 0.0, 0.0])
    hand = engine.state['slots']['right']
    hand.update(instrument='craniotome', position=(center + [0, 0, 12]).tolist(), quaternion=q.tolist())
    with pytest.raises(ValueError, match='KERF_FEED_REQUIRES_CUTTING_DWELL'):
        exposure.check_motion(engine, 'right', 'apply', center, q)
    assert engine.state[exposure.PROPERTY]['bodies']['bone']['status'] == 'attached'
    assert engine.state['removed'] == []
