"""Recorded withdrawal proposals use actual successfully traversed anatomy."""
import numpy as np
from scipy.spatial.transform import Rotation
from resectionlab.core import array_digest
from resectionlab.surgical_engine import SurgicalEngine
from resectionlab.surgical_scene import SurgicalScene
from resectionlab.full_case_withdrawal import (
    recorded_withdrawal_candidate, withdrawal_candidate, _outward_withdrawal_candidate)


def bent_entry(*, cleanup_marker=True):
    labels = np.zeros((64, 32, 64), np.uint16)
    # Two apertures permit a straight entry. The inner fan occurs below the
    # lower collar, with its 40 mm shaft ending below the upper collar. A
    # straight pullback at the final fan angle eventually hits the upper bone;
    # returning along the observed fan restores the original entry alignment.
    labels[:, :, 18] = labels[:, :, 60] = 1
    labels[29:32, 15:18, 18] = labels[29:32, 15:18, 60] = 0
    labels[30:33, 16, 14] = 2
    labels[30:34, 16, 11] = 3
    source = SurgicalScene(labels, np.eye(4), [
        {'id': 1, 'name': 'bone', 'material': 'bone'},
        {'id': 2, 'name': 'target', 'material': 'tumor', 'target': True},
        {'id': 3, 'name': 'healthy', 'material': 'soft_tissue'},
    ], {'kind': 'explicit-bent-entry-engineering-fixture'}, {})
    engine = SurgicalEngine(source)
    q = [1., 0., 0., 0.]
    pose = lambda z: {'position': [30., 16., z], 'quaternion': q}
    for action in [
        {'operation': 'exchange', 'instrument': 'cusa'},
        {'operation': 'place', 'pose': pose(72.), 'parameters': {'radiusMm': .5}},
        {'operation': 'apply', 'pose': pose(14.), 'duration': 3.},
    ]:
        result = engine.step({'slot': 'left', 'duration': .2, **action})
        assert result['status'] == 'accepted', result.get('reason')
    base_rotation = Rotation.from_quat(q)
    pivot = np.array([30., 16., 18.])
    path = []
    for lateral in np.linspace(0., 2., 101)[1:]:
        point = np.array([30. + lateral, 16., 14.])
        desired = (point - pivot) / np.linalg.norm(point - pivot)
        base_axis = base_rotation.apply([0., 0., 1.])
        cross = np.cross(base_axis, desired)
        turn = Rotation.from_rotvec(cross / np.linalg.norm(cross) * np.arctan2(np.linalg.norm(cross), base_axis @ desired))
        path.append({'position': point.tolist(), 'quaternion': (turn * base_rotation).as_quat().tolist()})
    outcome = engine.step({'slot': 'left', 'operation': 'apply', 'path': path, 'duration': .2})
    assert outcome['status'] == 'accepted', outcome.get('reason')
    if cleanup_marker:
        assert engine.step({'slot': 'left', 'operation': 'inspect', 'parameters': {'radiusMm': .5}, 'duration': .1})['status'] == 'accepted'
    return engine


def test_recorded_route_recovers_from_a_genuine_bent_entry_axis_withdrawal_block():
    engine = bent_entry()
    source_hash = array_digest(engine.scene.labels)
    outward = _outward_withdrawal_candidate(engine)
    assert outward is not None and outward['operation'] == 'move'
    assert engine.step(outward)['status'] == 'accepted'
    assert engine.state['slots']['left']['inserted']
    # The existing strategy has exhausted its admissible straight-axis prefix.
    assert _outward_withdrawal_candidate(engine) is None
    reference_position = len(engine.actions)
    original = engine.snapshot()
    first = recorded_withdrawal_candidate(engine, reference_position=reference_position)
    assert first is not None and engine.snapshot() == original
    assert len(first['path']) <= 512
    seen = set()
    committed_prefixes = 0
    for _ in range(8):
        before = engine.snapshot()
        candidate = recorded_withdrawal_candidate(engine, reference_position=reference_position)
        assert candidate is not None
        assert engine.snapshot() == before
        assert len(candidate.get('path', [])) <= 512 and 0 < candidate['duration'] <= 30.
        preview = engine.preview(candidate, include_view=False)
        assert preview['status'] == 'accepted', preview.get('reason')
        assert all(engine.by_label[int(engine.scene.labels.flat[cell])]['material'] == 'tumor'
                   for cell in preview.get('deltas', {}).get('removedCells', []))
        outcome = engine.step(candidate)
        committed_prefixes += 1
        assert outcome['stateId'] == preview['stateId']
        slot = engine.state['slots']['left']
        pose_key = tuple(np.round([*slot['position'], *slot['quaternion']], 10))
        assert pose_key not in seen, 'Repeated calls must not loop over a withdrawal pose'
        seen.add(pose_key)
        if not engine.state['slots']['left']['inserted']:
            break
    assert not engine.state['slots']['left']['inserted']
    assert committed_prefixes > 1
    assert np.allclose(engine.state['slots']['left']['position'], [30., 16., 72.])
    assert array_digest(engine.scene.labels) == source_hash
    assert not any(engine.scene.labels.flat[cell] == 3 for cell in engine.state['removed'])


def test_recorded_route_requires_inactive_released_hand_and_keeps_original_fallback():
    engine = bent_entry()
    assert engine.step({'slot': 'left', 'operation': 'activate', 'duration': .1})['status'] == 'accepted'
    assert recorded_withdrawal_candidate(engine) is None
    assert engine.step({'slot': 'left', 'operation': 'deactivate', 'duration': .1})['status'] == 'accepted'
    assert recorded_withdrawal_candidate(engine) is not None
    # A hand with no recorded placement still uses the original geometry-only
    # strategy where possible, rather than inventing a historical return path.
    fresh = SurgicalEngine(engine.scene)
    assert recorded_withdrawal_candidate(fresh) is None
    assert withdrawal_candidate(fresh) == _outward_withdrawal_candidate(fresh)


def test_inferred_cleanup_cutoff_ignores_new_recovery_apply_and_makes_strict_progress():
    engine = bent_entry()
    exterior = np.array([30., 16., 72.])
    original_q = Rotation.from_quat([1., 0., 0., 0.])
    def remaining():
        state = engine.state['slots']['left']
        angle = (Rotation.from_quat(state['quaternion']) * original_q.inv()).magnitude()
        return np.linalg.norm(np.asarray(state['position']) - exterior) + 40. * angle
    previous = remaining()
    actions = 0
    while engine.state['slots']['left']['inserted'] and actions < 8:
        proposal = recorded_withdrawal_candidate(engine)
        assert proposal is not None
        if actions == 0:
            assert len(proposal['path']) == 512
            # A legal target-only active recovery must not become the newest
            # insertion path when the next proposal infers its fixed cutoff.
            proposal['operation'] = 'apply'
        preview = engine.preview(proposal, include_view=False)
        assert preview['status'] == 'accepted', preview.get('reason')
        assert engine.step(proposal)['stateId'] == preview['stateId']
        assert remaining() < previous - 1e-6
        previous = remaining()
        actions += 1
    assert actions > 1 and not engine.state['slots']['left']['inserted']
    assert np.allclose(engine.state['slots']['left']['position'], exterior)


def test_no_cleanup_marker_requires_an_explicit_frozen_reference():
    engine = bent_entry(cleanup_marker=False)
    assert recorded_withdrawal_candidate(engine) is None
    assert withdrawal_candidate(engine) == _outward_withdrawal_candidate(engine)
    assert recorded_withdrawal_candidate(engine, reference_position=len(engine.actions)) is not None
