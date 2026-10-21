"""Recorded-ray cleanup uses actual withdrawal, exterior motion and aspiration."""
import copy

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from resectionlab import full_case_rehearsal as director
from resectionlab.core import array_digest
from resectionlab.surgical_engine import SurgicalEngine
from resectionlab.surgical_physics_v2 import default_profile
from resectionlab.surgical_scene import SurgicalScene


def _two_openings(axis=2, *, wall=False, right_z=50.):
    labels = np.zeros((70, 70, 70), np.uint16)
    labels[5, 5, 5] = 2
    transverse = (axis + 1) % 3
    other = (axis + 2) % 3
    for center in (18, 42):
        for a in (-1, 0, 1):
            for b in (-1, 0, 1):
                index = [0, 0, 0]
                index[axis], index[transverse], index[other] = 18, center + a, 20 + b
                labels[tuple(index)] = 1
    if wall:
        labels[15:20, 15:20, 0] = 3
    scene = SurgicalScene(labels, np.eye(4), [
        {'id': 1, 'name': 'bone', 'material': 'bone'},
        {'id': 2, 'name': 'retained target', 'material': 'tumor', 'target': True},
        {'id': 3, 'name': 'exterior obstruction', 'material': 'soft_tissue'}],
        {'kind': 'explicit-two-recorded-cleanup-rays'}, {'tissuePhysicsV2': default_profile()})
    engine = SurgicalEngine(scene)
    q = Rotation.from_euler('x', 180, degrees=True).as_quat() if axis == 2 else Rotation.from_euler('y', -90, degrees=True).as_quat()

    def pose(center, depth):
        point = [0., 0., 0.]
        point[axis], point[transverse], point[other] = depth, center, 20.
        return {'position': point, 'quaternion': q.tolist()}

    for center in (18, 42):
        outcome = engine.step({'slot': 'left', 'operation': 'place', 'duration': .2,
            'pose': pose(center, 60.), 'parameters': {'radiusMm': 1.5, 'forceN': 20., 'rpm': 80000.}})
        assert outcome['status'] == 'accepted', outcome
        for _ in range(4):
            outcome = engine.step({'slot': 'left', 'operation': 'apply', 'duration': 5., 'pose': pose(center, 17.)})
            if outcome['status'] == 'accepted': break
        assert outcome['status'] == 'accepted', outcome
        assert engine.step({'slot': 'left', 'operation': 'withdraw', 'duration': 2., 'pose': pose(center, 60.)})['status'] == 'accepted'
    assert engine.step({'slot': 'left', 'operation': 'place', 'duration': .2, 'pose': pose(5., 60.)})['status'] == 'accepted'
    if wall:
        assert engine.step({'slot': 'right', 'operation': 'exchange', 'duration': .2, 'instrument': 'frazier_suction'})['status'] == 'accepted'
        assert engine.step({'slot': 'right', 'operation': 'place', 'duration': .2,
            'pose': {'position': [17., 17., right_z], 'quaternion': [0., 0., 0., 1.]},
            'parameters': {'radiusMm': .5}})['status'] == 'accepted'
    assert len(engine.state['removed']) == 18 and engine.state['debrisMm3'] == 18.
    meta = director.new_director(engine)
    meta.update(stage='endpoint', status='stopped', stopReason='Explicit retained-target fixture endpoint', lastStateId=engine.state_id)
    return engine, meta


@pytest.mark.parametrize('axis', [0, 2])
def test_recorded_cleanup_relocates_between_rays_and_conserves_source(axis):
    engine, meta = _two_openings(axis)
    meta['fieldCleanup'] = {'status': 'earlier-bounded-pass', 'results': []}
    before, prior = engine.snapshot(), copy.deepcopy(meta)
    source_hash = array_digest(engine.scene.labels)
    prepared = director.configure_recorded_field_cleanup(engine, meta, max_stations=8)
    assert engine.snapshot() == before and meta == prior
    assert prepared['priorFieldCleanups'][-1] == meta['fieldCleanup']
    assert prepared['priorStops'][-1]['reason'] == meta['stopReason']
    assert prepared['recordedFieldCleanupPasses'][-1]['startStateId'] == engine.state_id
    candidate, running = engine, prepared
    for _ in range(8):
        candidate, running = director.run_slice(candidate, running, max_actions=16)
        if running['status'] in ('stopped', 'complete'): break
    assert running['status'] == 'stopped' and running['stage'] == 'endpoint'
    assert running['fieldCleanup']['status'] == 'bounded-pass-finished'
    assert len(running['fieldCleanup']['transits']) >= 2
    assert all(transit['status'] == 'complete' and all(row['arrived'] for row in transit['results'])
               for transit in running['fieldCleanup']['transits'])
    assert candidate.state['debrisMm3'] == pytest.approx(0.)
    assert candidate.state['debrisRemovedMm3'] == pytest.approx(18.)
    assert candidate.state['drillingV2']['removedVolumeMm3'] == pytest.approx(18.)
    assert candidate.state['removed'] == engine.state['removed']
    assert all(not state['inserted'] and not state['active'] and not state['grip'] for state in candidate.state['slots'].values())
    assert not candidate.state['holds'] and array_digest(candidate.scene.labels) == source_hash
    replay = engine.clone()
    for action in candidate.actions[len(engine.actions):]:
        preview = replay.preview(action, include_view=False)
        assert replay.step(action)['stateId'] == preview['stateId']
    assert replay.snapshot() == candidate.snapshot()
    assert engine.snapshot() == before and meta == prior


def test_recorded_cleanup_configuration_refuses_stale_running_or_inserted_state():
    engine, meta = _two_openings()
    before = engine.snapshot()
    for update, message in [({'lastStateId': 'stale'}, 'boundary'),
                            ({'status': 'running'}, 'endpoint'),
                            ({'stage': 'core'}, 'endpoint'),
                            ({'scenario': 'vascular'}, 'endpoint'),
                            ({'sourceSceneId': 'other'}, 'source')]:
        invalid = {**meta, **update}
        with pytest.raises(ValueError, match=message):
            director.configure_recorded_field_cleanup(engine, invalid)
    assert engine.snapshot() == before
    assert engine.step({'slot': 'right', 'operation': 'apply', 'duration': .2})['status'] == 'accepted'
    meta['lastStateId'] = engine.state_id
    with pytest.raises(ValueError, match='Withdraw|release'):
        director.configure_recorded_field_cleanup(engine, meta)


def test_complete_target_endpoint_still_executes_recorded_field_cleanup():
    engine, meta = _two_openings()
    outside = {'position': [5., 5., -5.], 'quaternion': [0., 0., 0., 1.]}
    for action in (
        {'operation': 'place', 'pose': outside, 'parameters': {'radiusMm': .6, 'forceN': 0.}},
        {'operation': 'apply', 'pose': {**outside, 'position': [5., 5., 5.]}, 'duration': 2.},
        {'operation': 'withdraw', 'pose': outside, 'duration': 2.},
    ):
        outcome = engine.step({'slot': 'right', 'duration': .2, **action})
        assert outcome['status'] == 'accepted', outcome
    assert engine.metrics()['targetRemovedMm3'] == 1.
    meta.update(status='complete', lastStateId=engine.state_id)
    prepared = director.configure_recorded_field_cleanup(engine, meta)
    candidate, running = engine, prepared
    for _ in range(8):
        candidate, running = director.run_slice(candidate, running, max_actions=16)
        if running['status'] in ('stopped', 'complete'): break
    assert running['status'] == 'complete' and running['stage'] == 'endpoint'
    assert running['fieldCleanup']['status'] == 'bounded-pass-finished'
    assert candidate.state['debrisMm3'] == pytest.approx(0.)
    assert candidate.state['debrisRemovedMm3'] == pytest.approx(18.)
    assert candidate.state['removed'] == engine.state['removed']
    assert candidate.metrics()['targetRemovedMm3'] == 1.


@pytest.mark.parametrize('right_z,expected_status', [(50., 'partial'), (41.001, 'rejected')])
def test_blocked_cleanup_relocation_keeps_actual_pose_and_never_approaches(right_z, expected_status):
    engine, meta = _two_openings(wall=True, right_z=right_z)
    prepared = director.configure_recorded_field_cleanup(engine, meta, max_stations=2)
    before = engine.snapshot()
    candidate, stopped = director.run_slice(engine, prepared, max_actions=16)
    assert stopped['status'] == 'stopped' and stopped['stage'] in ('cleanup', 'endpoint')
    if stopped['stage'] == 'cleanup':
        assert 'blocked:' in stopped['stopReason'] and 'SHAFT_COLLISION' in stopped['stopReason']
    cleanup = stopped['fieldCleanup']
    assert cleanup['status'] == 'blocked'
    transit = cleanup['transits'][0]
    assert transit['status'] == 'blocked' and transit['index'] == 0
    assert transit['results'][0]['status'] == expected_status and not transit['results'][0]['arrived']
    assert not np.allclose(candidate.state['slots']['right']['position'], transit['poses'][0]['position'])
    assert all(action['operation'] != 'apply' for action in candidate.actions[len(engine.actions):])
    assert candidate.state['removed'] == engine.state['removed'] and candidate.state['debrisMm3'] == 18.
    assert engine.snapshot() == before
