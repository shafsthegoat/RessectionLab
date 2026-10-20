"""Actual engineering drill/irrigation/suction laws, without patient fixtures."""
import copy
import numpy as np
import pytest
from scipy.spatial.transform import Rotation
from resectionlab.core import array_digest
from resectionlab.surgical_engine import SurgicalEngine
from resectionlab.surgical_scene import SurgicalScene
from resectionlab.surgical_physics_v2 import default_profile
from resectionlab.full_case_cleanup import cleanup_stations, cleanup_dwell_action, recorded_drill_cleanup_stations


def setup_field():
    labels = np.zeros((64, 40, 40), np.uint16)
    labels[41:44, 19:22, 18] = labels[41:44, 19:22, 24] = 1
    labels[42, 20, 15] = 2
    scene = SurgicalScene(labels, np.eye(4), [
        {'id': 1, 'name': 'bone', 'material': 'bone'},
        {'id': 2, 'name': 'target', 'material': 'tumor', 'target': True}],
        {'kind': 'explicit-cleanup-engineering-fixture'}, {'tissuePhysicsV2': default_profile()})
    engine = SurgicalEngine(scene)
    pose = lambda z: {'position': [42., 20., float(z)], 'quaternion': [1., 0., 0., 0.]}
    def step(operation, *, slot='left', **values):
        result = engine.step({'slot': slot, 'operation': operation, 'duration': .2, **values})
        assert result['status'] in ('accepted', 'partial'), result.get('reason')
        return result
    step('place', pose=pose(45.), parameters={'radiusMm': 1.5, 'forceN': 20., 'rpm': 80000.})
    for _ in range(3):
        step('apply', pose=pose(17.), duration=5.)
        if len(engine.state['removed']) == 18: break
    assert len(engine.state['removed']) == 18 and engine.state['debrisMm3'] == 18.
    step('withdraw')
    step('place', pose={'position': [10., 10., 45.], 'quaternion': [1., 0., 0., 0.]})
    step('exchange', slot='right', instrument='frazier_suction')
    step('place', slot='right', pose=pose(45.), parameters={'radiusMm': .5, 'flowMm3PerSecond': 20., 'forceN': 0.})
    step('move', slot='right', pose=pose(17.), duration=1.5)
    step('inspect', slot='right', parameters={'radiusMm': 3.})
    field = {'slot': 'right', 'instrument': 'frazier_suction', 'fieldPose': pose(17.), 'outsidePose': pose(45.),
             'parameters': {'radiusMm': 3., 'depthMm': 2., 'flowMm3PerSecond': 20., 'forceN': 0.},
             'sourceSceneId': scene.identity}
    return engine, field, step


def assert_ledgers(engine):
    state = engine.state
    generated = state['drillingV2']['removedVolumeMm3']
    assert generated == pytest.approx(state['debrisMm3'] + state.get('debrisRemovedMm3', 0.) + state.get('debrisEscapedMm3', 0.))
    assert state['fluidAddedMm3'] == pytest.approx(sum(state['fluid'].values())
        + sum(source.get('reservoirMm3', 0.) for source in state['sources'])
        + state['fluidRemovedMm3'] + state['fluidEscapedMm3'])


def test_stations_are_detached_bounded_and_fresh_overlap_is_not_double_counted():
    engine, field, step = setup_field()
    original, original_field = engine.snapshot(), copy.deepcopy(field)
    source_hash = array_digest(engine.scene.labels)
    stations = cleanup_stations(engine, field, max_stations=99)
    assert engine.snapshot() == original and field == original_field
    assert len(stations) == 3
    assert [station['pose']['position'][2] for station in stations] == [17., 20., 24.]
    assert [station['fieldDebrisMm3'] for station in stations] == [9., 9., 9.]
    # The first two measured fields overlap. These 27 membership mm3 are not
    # a physical removal total: the actual generated debris is only18 mm3.
    assert sum(station['fieldMassMm3'] for station in stations) > engine.state['debrisMm3']
    for station in stations:
        assert station['parameters'] == field['parameters'] and station['outsidePose'] == field['outsidePose']
        assert station['pose']['quaternion'] == field['fieldPose']['quaternion']
        assert station['requiresPublicPreview']
    first = cleanup_dwell_action(engine, stations[0])
    assert first and first['fieldDebrisMm3'] == 9. and first['fieldFluidMm3'] == 0.
    assert 'pose' not in first['action'] and 'path' not in first['action']
    assert first['action']['slot'] == 'right' and first['action']['duration'] <= 10.
    assert engine.snapshot() == original
    preview = engine.preview(first['action'], include_view=False)
    assert engine.step(first['action'])['stateId'] == preview['stateId']
    assert engine.state['debrisRemovedMm3'] == pytest.approx(9.)
    assert_ledgers(engine)
    # Actual public movement to the overlapping second field, then a fresh
    # measurement: its old positive descriptor cannot manufacture more debris.
    step('apply', slot='right', pose=stations[1]['pose'], duration=.2)
    assert cleanup_dwell_action(engine, stations[1]) is None
    step('apply', slot='right', pose=stations[2]['pose'],
         parameters={'radiusMm': .5, 'flowMm3PerSecond': 0.}, duration=.2)
    remaining = cleanup_dwell_action(engine, stations[2], max_duration=99.)
    assert remaining and remaining['fieldDebrisMm3'] == pytest.approx(9.)
    assert remaining['action']['duration'] <= 10.
    engine.step(remaining['action'])
    assert engine.state['debrisMm3'] == pytest.approx(0.)
    assert engine.state['debrisRemovedMm3'] == pytest.approx(18.)
    assert cleanup_stations(engine, field) == []
    assert_ledgers(engine)
    assert len(engine.state['removed']) == 18 and all(engine.scene.labels.flat[cell] == 1 for cell in engine.state['removed'])
    assert array_digest(engine.scene.labels) == source_hash and field == original_field


def test_dwell_uses_actual_fluid_first_budget_and_never_relocates_or_exchanges():
    engine, field, step = setup_field()
    station = cleanup_stations(engine, field)[0]
    # Deposit real public irrigation at the suction field from a separated
    # stationary outlet; no field state is seeded or changed directly.
    step('exchange', instrument='irrigation')
    step('place', pose={'position': [48., 20., 17.],
        'quaternion': [0., -2. ** -.5, 0., 2. ** -.5]},
        parameters={'radiusMm': .5, 'depthMm': 6., 'flowMm3PerSecond': 10., 'forceN': 0.})
    step('apply', duration=.2)
    assert engine.state['fluidAddedMm3'] == pytest.approx(2.)
    original = engine.snapshot()
    proposal = cleanup_dwell_action(engine, station, max_duration=.05)
    assert proposal and proposal['fieldFluidMm3'] == pytest.approx(2.)
    assert proposal['plannedFlowBudgetMm3'] == pytest.approx(1.)
    assert engine.snapshot() == original
    before = engine.state.get('debrisRemovedMm3', 0.)
    outcome = engine.step(proposal['action'])
    assert outcome['status'] == 'accepted'
    assert engine.state['fluidRemovedMm3'] == pytest.approx(1.)
    assert engine.state.get('debrisRemovedMm3', 0.) == before
    assert_ledgers(engine)
    empty_flow = copy.deepcopy(station)
    empty_flow['parameters']['flowMm3PerSecond'] = 0.
    assert cleanup_dwell_action(engine, empty_flow) is None
    moved = copy.deepcopy(station); moved['pose']['position'][2] += 1.
    assert cleanup_dwell_action(engine, moved) is None
    equivalent = copy.deepcopy(station)
    equivalent['pose']['quaternion'] = [-value for value in equivalent['pose']['quaternion']]
    assert cleanup_dwell_action(engine, equivalent) is not None
    assert cleanup_stations(engine, field, max_stations=0) == []


def recorded_access(axis=2, signed=False, *, partial=False):
    """A translated real drill opening with a deeper unremoved stop layer."""
    labels = np.zeros((70, 70, 70), np.uint16)
    center = np.array([50., 20., 20.]) if axis == 2 else np.array([20., 50., 20.])
    for depth in (8, 18, 24):
        index = center.astype(int)
        for a in (-1, 0, 1):
            for b in (-1, 0, 1):
                index[axis] = depth
                index[(axis + 1) % 3] = int(center[(axis + 1) % 3]) + a
                index[(axis + 2) % 3] = int(center[(axis + 2) % 3]) + b
                labels[tuple(index)] = 1
    affine = np.eye(4); affine[:3, 3] = [70., -40., 12.]
    scene = SurgicalScene(labels, affine, [{'id': 1, 'name': 'bone', 'material': 'bone'}],
        {'kind': 'translated-recorded-cleanup-fixture'}, {'tissuePhysicsV2': default_profile()})
    engine = SurgicalEngine(scene)
    q = ({0: Rotation.from_euler('y', -90, degrees=True),
          1: Rotation.from_euler('x', 90, degrees=True),
          2: Rotation.from_euler('x', 180, degrees=True)}[axis]).as_quat()
    if signed: q = -q
    def pose_at(depth, *, parked=False):
        point = center.copy(); point[axis] = depth
        if parked: point[(axis + 1) % 3] = 5.
        return {'position': (point + affine[:3, 3]).tolist(), 'quaternion': q.tolist()}
    def step(operation, **values):
        action = {'slot': 'left', 'operation': operation, 'duration': .2, **values}
        outcome = engine.step(action)
        assert outcome['status'] in ('accepted', 'partial'), outcome
        return outcome
    step('place', pose=pose_at(60), parameters={'radiusMm': 1.5, 'forceN': 20., 'rpm': 80000.})
    for _ in range(3):
        outcome = step('apply', pose=pose_at(17), duration=5.)
        if outcome['status'] == 'accepted': break
    assert outcome['status'] == 'accepted' and len(engine.state['removed']) == 18
    if partial:
        outcome = step('apply', pose=pose_at(1), duration=.2, parameters={'forceN': 0., 'rpm': 0.})
        assert outcome['status'] == 'partial' and outcome['reasonCode'] == 'BONE_CUTTING_IN_PROGRESS'
        partial_receipt = copy.deepcopy(engine.history[-1])
    else:
        partial_receipt = None
    step('withdraw', pose=pose_at(60), duration=2.)
    step('place', pose=pose_at(60, parked=True))
    return engine, q, partial_receipt


@pytest.mark.parametrize('axis,signed', [(0, True), (1, False), (2, True)])
def test_recorded_rays_are_source_bound_detached_and_overlap_unique(axis, signed):
    import json
    engine, q, _ = recorded_access(axis, signed)
    before = engine.snapshot(); source_hash = array_digest(engine.scene.labels)
    stations = recorded_drill_cleanup_stations(engine, max_stations=999)
    assert stations and len(stations) <= 64
    assert stations == recorded_drill_cleanup_stations(engine, max_stations=999)
    assert engine.snapshot() == before and array_digest(engine.scene.labels) == source_hash
    assert json.loads(json.dumps(stations)) == stations
    covered = set(); unique_mass = 0.
    for station in stations:
        assert station['sceneId'] == station['sourceSceneId'] == engine.scene.identity
        assert station['pose']['quaternion'] == q.tolist()
        assert station['drillPlacementActions1Based'] == [1]
        assert all(engine.history[i - 1]['action']['operation'] == 'apply'
                   for i in station['drillApplyActions1Based'])
        point = station['pose']['position']; radius = station['parameters']['radiusMm']
        ids = set(map(int, engine.cells(point, point, radius)))
        gain = sum(engine.state['debris'].get(str(cell), 0.) for cell in ids - covered)
        assert station['newUniqueFieldMassMm3'] == pytest.approx(gain)
        assert gain > 0. and station['requiresPublicPreview']
        assert not engine.occupancy().material_ids(engine.cells(point,
            np.asarray(station['outsidePose']['position']) - 40. *
            Rotation.from_quat(q).apply([0., 0., 1.]),
            station['checkedPhysicalRayRadiusMm']))
        covered.update(ids); unique_mass += gain
    assert unique_mass == pytest.approx(18.)
    assert sum(station['newUniqueFieldMassMm3'] for station in stations) == pytest.approx(engine.state['debrisMm3'])
    stations[0]['outsidePose']['position'][0] += 99.
    stations[0]['drillApplyActions1Based'].append(-1)
    assert engine.snapshot() == before


def test_recorded_partial_drill_depth_uses_actual_stop_and_dwell_is_ordinary():
    engine, q, receipt = recorded_access(partial=True)
    before = engine.snapshot()
    stations = recorded_drill_cleanup_stations(engine, radius_mm=1.5, flow_mm3_per_second=7.)
    assert stations and engine.snapshot() == before
    outside = np.asarray(stations[0]['outsidePose']['position'])
    stopped = np.asarray(receipt['contactPosition'])
    expected_depth = np.linalg.norm(stopped - outside)
    requested_depth = np.linalg.norm(np.asarray(receipt['action']['pose']['position']) - outside)
    assert expected_depth < requested_depth - 1.
    for station in stations:
        assert station['maximumReceiptAttainedDepthMm'] == pytest.approx(expected_depth)
        assert station['depthFromRecordedOutsideMm'] <= expected_depth + 1e-9
        assert station['partialDepthEvidence'][-1]['requestedDepthMm'] == pytest.approx(requested_depth)
        assert station['parameters']['radiusMm'] == 1.5 and station['parameters']['flowMm3PerSecond'] == 7.
    # Use ordinary public setup/approach/effect for the proposed right station.
    station = stations[0]
    for action in (
        {'operation': 'exchange', 'instrument': 'frazier_suction'},
        {'operation': 'place', 'pose': station['outsidePose'], 'parameters': {**station['parameters'], 'radiusMm': .5}},
        {'operation': 'move', 'pose': station['pose'], 'duration': 2.},
    ):
        result = engine.step({'slot': 'right', 'duration': .2, **action})
        assert result['status'] == 'accepted', result
    dwell = cleanup_dwell_action(engine, station)
    assert dwell and dwell['action']['parameters']['flowMm3PerSecond'] == 7.
    preview = engine.preview(dwell['action'], include_view=False)
    assert engine.step(dwell['action'])['stateId'] == preview['stateId']
    assert engine.state['debrisRemovedMm3'] > 0.
    assert_ledgers(engine)


def test_recorded_cleanup_limits_receipts_and_other_hand_filter():
    engine, _, _ = recorded_access()
    assert recorded_drill_cleanup_stations(engine, max_stations=0) == []
    assert recorded_drill_cleanup_stations(engine, flow_mm3_per_second=0.) == []
    assert len(recorded_drill_cleanup_stations(engine, max_stations=1)) == 1
    for kwargs in ({'max_stations': -1}, {'max_stations': 1.5}, {'radius_mm': 13.},
                   {'radius_mm': float('nan')}, {'flow_mm3_per_second': -1.},
                   {'flow_mm3_per_second': 1001.}):
        with pytest.raises(ValueError): recorded_drill_cleanup_stations(engine, **kwargs)
    # Public placement of a passive other hand at the recorded outside makes
    # the whole insertion ray conflicting, even when an internal field is free.
    station = recorded_drill_cleanup_stations(engine)[0]
    result = engine.step({'slot': 'left', 'operation': 'place', 'duration': .2,
                          'pose': station['outsidePose']})
    assert result['status'] == 'accepted'
    assert recorded_drill_cleanup_stations(engine) == []
    engine.history[-1]['action']['duration'] = .3
    with pytest.raises(ValueError, match='matching public action receipts'):
        recorded_drill_cleanup_stations(engine)


def test_generic_partial_failed_contact_is_not_an_attained_drill_endpoint():
    engine, _, _ = setup_field()
    partial = [row for row in engine.history if row['status'] == 'partial']
    assert partial and partial[0]['reasonCode'] == 'UNSUPPORTED_MATERIAL'
    assert partial[0]['contactPosition'] == partial[0]['action']['pose']['position']
    assert engine.state['debrisMm3'] == 18.
    # The public failed microstep is not the actual preceding slot pose. No
    # other accepted drill stroke established a usable ray in this fixture.
    assert recorded_drill_cleanup_stations(engine) == []
