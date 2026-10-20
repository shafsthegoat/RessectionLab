"""Actual engineering drill/irrigation/suction laws, without patient fixtures."""
import copy
import numpy as np
import pytest
from resectionlab.core import array_digest
from resectionlab.surgical_engine import SurgicalEngine
from resectionlab.surgical_scene import SurgicalScene
from resectionlab.surgical_physics_v2 import default_profile
from resectionlab.full_case_cleanup import cleanup_stations, cleanup_dwell_action


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
