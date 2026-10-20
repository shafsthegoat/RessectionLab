"""Public-engine checks on explicit small directional engineering anatomy."""
import copy

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from resectionlab.full_case_directional import (
    directional_access_candidates, directional_removal_candidates, rank_directional_accesses,
)
from resectionlab.surgical_engine import SurgicalEngine
from resectionlab.surgical_scene import SurgicalScene
from resectionlab.surgical_rl_protection import (
    GuardConfig, ProtectionAnnotation, ProtectionSet, grid_binding,
)


def _engine(axis, *, barrier=None, protected=False, skull=True, spacing=(1., 1., 1.)):
    # The view is only a fixture-construction convenience; SurgicalScene owns
    # native labels throughout every real engine transition below.
    labels = np.zeros((18, 18, 18), dtype=np.uint16)
    view = np.moveaxis(labels, axis, -1)
    view[4:14, 4:14, 5:11] = 7
    if skull:
        view[4:14, 4:14, 14:16] = 9
    view[4:14, 4:14, 2] = 1
    if barrier is not None:
        view[4:14, 4:14, 16] = barrier
    affine = np.diag([-float(spacing[0]), float(spacing[1]), float(spacing[2]), 1.])
    affine[:3, 3] = [40., -10., -20.]
    scene = SurgicalScene(labels, affine, [
        {'id': 1, 'name': 'brain', 'material': 'soft_tissue'},
        {'id': 5, 'name': 'vessel', 'material': 'vessel'},
        {'id': 7, 'name': 'target', 'material': 'tumor', 'target': True},
        {'id': 9, 'name': 'bone', 'material': 'bone'},
    ], {'kind': 'explicit-directional-engineering-fixture'}, {})
    if protected:
        cells = tuple(map(int, np.flatnonzero(scene.labels == 7)))
        annotation = ProtectionAnnotation('brainstem', cells, grid_binding(scene), 'explicit fixture only')
        policy = ProtectionSet(scene, [annotation], clearance_mm=0.)
        return SurgicalEngine(scene, protection=policy, guard_config=GuardConfig())
    return SurgicalEngine(scene)


def _opened_cusa(engine, access):
    actions = [
        {'operation': 'place', 'pose': access['outsidePose'],
         'parameters': {'radiusMm': access['drillRadiusMm']}},
        {'operation': 'apply', 'pose': access['entryPose'], 'duration': 1.},
        {'operation': 'inspect', 'parameters': {'radiusMm': .5}},
        {'operation': 'withdraw', 'pose': access['outsidePose']},
        {'operation': 'exchange', 'instrument': 'cusa'},
        {'operation': 'place', 'pose': access['outsidePose'],
         'parameters': {'radiusMm': 1.3}},
    ]
    for action in actions:
        result = engine.step({'slot': 'left', 'duration': .2, **action})
        assert result['status'] in ('accepted', 'partial'), result
        if action['operation'] == 'apply':
            assert result['reason'] == 'UNSUPPORTED_MATERIAL'
    assert engine.metrics()['targetRemovedMm3'] == 0.
    assert engine.metrics()['removedByLabelMm3']['bone'] > 0.


def test_both_axes_use_continuous_public_access_and_identical_native_accounting():
    effects = []
    for axis in (0, 1):
        engine = _engine(axis)
        source = engine.scene.labels.copy(); source_id = engine.scene.identity
        original = engine.snapshot()
        access = directional_access_candidates(engine, axis=axis, limit=1)[0]
        assert engine.snapshot() == original
        assert access['nativeAxis'] == axis and access['sourceSceneId'] == source_id
        assert .1 <= access['drillRadiusMm'] <= 3.
        inward = Rotation.from_quat(access['entryPose']['quaternion']).apply([0, 0, 1])
        np.testing.assert_allclose(inward, -np.asarray(access['outwardRAS']), atol=1e-12)
        np.testing.assert_allclose(
            engine.scene.affine[:3, :3] @ access['targetEntryIndex'] + engine.scene.affine[:3, 3],
            access['entryPose']['position'])
        _opened_cusa(engine, access)
        before = engine.snapshot(); before_access = copy.deepcopy(access)
        candidates = directional_removal_candidates(engine, access, limit=2)
        assert engine.snapshot() == before and access == before_access
        assert candidates, 'A genuinely cleared directional opening should admit a frontier proposal'
        action = candidates[0]
        assert action['operation'] == 'apply' and 0 < len(action['path']) <= 512
        preview = engine.preview(action, include_view=False)
        assert engine.snapshot() == before
        assert preview['status'] == 'accepted', preview
        target_ids = preview['deltas']['removedCells']
        assert len(target_ids) > 10
        assert all(source.ravel()[i] == 7 for i in target_ids)
        committed = engine.step(action)
        assert committed['stateId'] == preview['stateId']
        assert committed['metrics'] == preview['metrics']
        effect = np.zeros(source.shape, dtype=bool); effect.ravel()[target_ids] = True
        effects.append(np.moveaxis(effect, axis, -1))
        assert engine.scene.identity == source_id
        np.testing.assert_array_equal(engine.scene.labels, source)
    np.testing.assert_array_equal(*effects)


@pytest.mark.parametrize('axis', [0, 1])
def test_unopened_roof_has_no_removal_candidate_and_source_barriers_block_windows(axis):
    engine = _engine(axis)
    access = directional_access_candidates(engine, axis=axis, limit=1)[0]
    assert engine.step({'slot': 'left', 'operation': 'exchange', 'instrument': 'cusa'})['status'] == 'accepted'
    assert engine.step({'slot': 'left', 'operation': 'place', 'pose': access['outsidePose']})['status'] == 'accepted'
    before = engine.snapshot()
    assert directional_removal_candidates(engine, access) == []
    assert engine.snapshot() == before
    for barrier in (1, 5):
        blocked = _engine(axis, barrier=barrier)
        before = blocked.snapshot()
        assert directional_access_candidates(blocked, axis=axis) == []
        assert blocked.snapshot() == before


@pytest.mark.parametrize('axis', [0, 1])
def test_retained_capacity_ranking_preserves_prefix_ties_and_native_flat_ids(axis):
    engine = _engine(axis)
    routes = directional_access_candidates(engine, axis=axis, limit=4)
    assert len(routes) >= 3
    initial = rank_directional_accesses(engine, routes)
    assert all(r['geometricTargetCapacityCells'] == r['remainingGeometricCapacityCells'] for r in initial)
    # Fixture setup uses known original identities without claiming execution.
    # The real removal/accounting trajectory is tested separately above.
    depleted = copy.deepcopy(routes[1])
    ij = np.asarray(depleted['targetEntryIndex'])
    native = np.indices(engine.scene.shape).reshape(3, -1).T
    transverse = [i for i in range(3) if i != axis]
    disk = np.sum((native[:, transverse]-ij[transverse])**2, axis=1) <= max(1., depleted['drillRadiusMm']-1.)**2
    engine.state['removed'] = np.flatnonzero(disk & (engine.scene.labels.ravel() == 7)).tolist()
    before, before_routes = engine.snapshot(), copy.deepcopy(routes)
    ranked = rank_directional_accesses(engine, [routes[0], depleted, routes[2], copy.deepcopy(routes[2])], 1)
    assert ranked[0] == routes[0]
    assert ranked[1]['id'] == ranked[2]['id'] == routes[2]['id']
    assert ranked[-1]['remainingGeometricCapacityCells'] == 0
    assert all(r['capacityStateId'] == engine.state_id for r in ranked[1:])
    assert engine.snapshot() == before and routes == before_routes


@pytest.mark.parametrize('axis', [0, 1])
def test_existing_public_protection_blocks_directional_contact_without_source_changes(axis):
    # Protection covers target here deliberately: target ownership must not
    # override source-bound exclusions. No clinical anatomy is inferred.
    engine = _engine(axis, protected=True, skull=False)
    access = directional_access_candidates(engine, axis=axis, limit=1)[0]
    assert engine.step({'slot': 'left', 'operation': 'exchange', 'instrument': 'cusa'})['status'] == 'accepted'
    assert engine.step({'slot': 'left', 'operation': 'place', 'pose': access['outsidePose']})['status'] == 'accepted'
    before = engine.snapshot(); source = engine.scene.labels.copy()
    candidates = directional_removal_candidates(engine, access, limit=1)
    assert candidates  # Target ownership alone never constitutes guard admission.
    preview = engine.preview(candidates[0], include_view=False)
    assert preview['status'] != 'accepted'
    assert preview['reason'].startswith('PROTECTED_STRUCTURE')
    assert engine.snapshot() == before and preview['metrics']['targetRemovedMm3'] == 0.
    np.testing.assert_array_equal(engine.scene.labels, source)


def test_invalid_axis_affine_source_binding_and_aperture_are_refused():
    engine = _engine(0)
    for axis in (-1, 2):
        with pytest.raises(ValueError, match='axis'):
            directional_access_candidates(engine, axis=axis)
    with pytest.raises(ValueError, match='radius'):
        directional_access_candidates(engine, axis=0, max_radius_mm=3.1)
    access = directional_access_candidates(engine, axis=0, limit=1)[0]
    access['sourceSceneId'] = 'different-source'
    with pytest.raises(ValueError, match='source'):
        directional_removal_candidates(engine, access)


@pytest.mark.parametrize('axis', [0, 1])
def test_signed_anisotropic_affine_capacity_is_in_native_transverse_millimeters(axis):
    engine = _engine(axis, spacing=(.7, 1.3, 1.8))
    before = engine.snapshot(); source = engine.scene.labels.copy()
    rows = directional_access_candidates(engine, axis=axis, limit=3)
    assert rows
    ranked = rank_directional_accesses(engine, rows)
    transverse = [i for i in range(3) if i != axis]
    native = np.indices(engine.scene.shape).reshape(3, -1).T
    for row in ranked:
        index = np.asarray(row['targetEntryIndex'])
        delta_mm = (native[:, transverse]-index[transverse])*engine.scene.spacing[transverse]
        disk = np.sum(delta_mm**2, axis=1) <= max(1., row['drillRadiusMm']-1.)**2
        expected = np.count_nonzero(disk & (source.ravel() == 7))
        assert row['remainingGeometricCapacityCells'] == row['geometricTargetCapacityCells'] == expected
        point = np.asarray(row['entryPose']['position'])
        np.testing.assert_allclose(engine.scene.inverse[:3, :3] @ point + engine.scene.inverse[:3, 3], index)
        displacement = np.subtract(row['outsidePose']['position'], point)
        assert np.dot(displacement, row['outwardRAS']) > 8.
    assert engine.snapshot() == before
    np.testing.assert_array_equal(engine.scene.labels, source)


def _nearby_frontier(axis, blocker=None):
    """Source-labeled free cavity and target rim, never a patient checkpoint."""
    labels = np.zeros((24, 24, 24), dtype=np.uint16)
    view = np.moveaxis(labels, axis, -1)
    if blocker is None:
        view[9:12, 9:12, 5] = 7
    else:
        view[10, 10, 5] = 7
        if blocker == 'material': view[8, 10, 7] = 1
        elif blocker == 'shaft': view[8, 10, 14] = 9
        elif blocker == 'subvoxel-collar': view[9, 10, 14] = 9
    affine = np.diag([-1., 1., 1., 1.]); affine[:3, 3] = [40., -10., -20.]
    if blocker == 'subvoxel-collar':
        # Integer ray clears the real .5mm shaft with .05mm fine steps, but
        # the existing coarse outside sweep inflates to ~.667mm at this collar.
        transverse = [a for a in range(3) if a != axis]
        affine[transverse[0], transverse[0]] *= 1.2
    engine = SurgicalEngine(SurgicalScene(labels, affine, [
        {'id': 1, 'name': 'brain', 'material': 'soft_tissue'},
        {'id': 7, 'name': 'target', 'material': 'tumor', 'target': True},
        {'id': 9, 'name': 'bone', 'material': 'bone'},
    ], {'kind': 'explicit-short-directional-fixture'}, {}))
    def point(depth, first=10, second=10):
        index = [first, second]; index.insert(axis, depth)
        return (affine[:3, :3] @ index + affine[:3, 3]).tolist()
    q = ([0., np.sqrt(.5), 0., np.sqrt(.5)] if axis == 0
         else [np.sqrt(.5), 0., 0., np.sqrt(.5)])
    def pose(depth, first=10):
        return {'position': point(depth, first), 'quaternion': list(q)}
    index = [10, 10]; index.insert(axis, 5)
    access = {'id': 'explicit-nearby-rim', 'nativeAxis': axis, 'outwardIndexSign': 1,
              'sourceSceneId': engine.scene.identity, 'targetEntryIndex': index,
              'entryPose': pose(5), 'outsidePose': pose(31), 'drillRadiusMm': 3.}
    first = 10.25 if blocker == 'subvoxel-collar' else (10 if blocker is None else 6)
    approach = {'operation': 'move', 'pose': pose(8, first)}
    if blocker == 'subvoxel-collar':
        start = np.asarray(point(31, first)); end = np.asarray(point(8, first))
        approach = {'operation': 'move', 'path': [
            {'position': (start*(1.-f)+end*f).tolist(), 'quaternion': list(q)}
            for f in np.linspace(0., 1., 461)[1:]]}
    for action in [
        {'operation': 'exchange', 'instrument': 'cusa'},
        {'operation': 'place', 'pose': pose(31, first), 'parameters': {'radiusMm': .3}},
        approach,
    ]:
        result = engine.step({'slot': 'left', 'duration': .2, **action})
        assert result['status'] == 'accepted', result
    if blocker == 'other-hand':
        assert axis == 0
        result = engine.step({'slot': 'right', 'operation': 'place',
                              'pose': {'position': point(14, 8), 'quaternion': [1., 0., 0., 0.]},
                              'parameters': {'radiusMm': .3}, 'duration': .2})
        assert result['status'] == 'accepted', result
    return engine, access


def _travel(engine, action):
    positions = [engine.state['slots'][action['slot']]['position'],
                 *[p['position'] for p in action['path']]]
    return float(np.linalg.norm(np.diff(positions, axis=0), axis=1).sum())


def _visits_outside(access, action):
    outward = -Rotation.from_quat(access['outsidePose']['quaternion']).apply([0., 0., 1.])
    plane = np.asarray(access['outsidePose']['position']) @ outward
    return any(np.asarray(p['position']) @ outward >= plane-1e-8 for p in action['path'])


@pytest.mark.parametrize('axis', [0, 1])
def test_short_candidate_starts_at_actual_pose_and_retains_checked_outside_alternative(axis):
    engine, access = _nearby_frontier(axis)
    before, source = engine.snapshot(), engine.scene.labels.copy()
    actions = directional_removal_candidates(engine, access, limit=2)
    assert len(actions) == 2
    assert engine.snapshot() == before
    direct, outside = actions
    assert not _visits_outside(access, direct) and _visits_outside(access, outside)
    np.testing.assert_array_equal(direct['path'][0]['position'], engine.state['slots']['left']['position'])
    assert len(direct['path']) <= 512
    points = np.asarray([p['position'] for p in direct['path']])
    assert np.max(np.linalg.norm(np.diff(points, axis=0), axis=1)) <= .05+1e-9
    assert np.linalg.norm(points[-1]-points[0]) <= 15.
    assert all(p['quaternion'] == direct['path'][0]['quaternion'] for p in direct['path'])
    assert _travel(engine, direct) < _travel(engine, outside)/10.
    short_preview = engine.preview(direct, include_view=False)
    outside_preview = engine.preview(outside, include_view=False)
    assert short_preview['status'] == outside_preview['status'] == 'accepted'
    assert len(short_preview['deltas']['removedCells']) > 1
    # Tied actual target effect is ordered by travel, with the checked outside
    # route still present if the caller's public preview refuses the first.
    assert set(short_preview['deltas']['removedCells']) == set(outside_preview['deltas']['removedCells'])
    assert engine.snapshot() == before
    result = engine.step(direct)
    assert result['stateId'] == short_preview['stateId']
    assert engine.actions[:-1] == before['actions']
    assert all(source.ravel()[cell] == 7 for cell in result['deltas']['removedCells'])
    np.testing.assert_array_equal(engine.scene.labels, source)


@pytest.mark.parametrize('blocker', ['material', 'shaft', 'other-hand'])
def test_short_blocking_material_shaft_or_other_hand_preserves_legal_outside_fallback(blocker):
    engine, access = _nearby_frontier(0, blocker)
    before = engine.snapshot(); source = engine.scene.labels.copy()
    actions = directional_removal_candidates(engine, access, limit=3)
    assert actions and all(_visits_outside(access, action) for action in actions)
    preview = engine.preview(actions[0], include_view=False)
    assert preview['status'] == 'accepted', preview
    assert preview['deltas']['removedCells']
    assert all(source.ravel()[cell] == 7 for cell in preview['deltas']['removedCells'])
    assert engine.snapshot() == before
    np.testing.assert_array_equal(engine.scene.labels, source)


def test_outside_start_does_not_create_a_short_path_that_teleports_to_the_cavity():
    engine, access = _nearby_frontier(0)
    assert engine.step({'slot': 'left', 'operation': 'withdraw', 'pose': access['outsidePose'],
                        'duration': .2})['status'] == 'accepted'
    assert engine.step({'slot': 'left', 'operation': 'place', 'pose': access['outsidePose'],
                        'parameters': {'radiusMm': .3}, 'duration': .2})['status'] == 'accepted'
    before = engine.snapshot()
    actions = directional_removal_candidates(engine, access, limit=3)
    assert actions
    np.testing.assert_array_equal(engine.state['slots']['left']['position'], access['outsidePose']['position'])
    for action in actions:
        assert _travel(engine, action) > 15.
        assert engine.preview(action, include_view=False)['status'] == 'accepted'
    assert engine.snapshot() == before


def test_integer_short_does_not_suppress_a_subvoxel_outside_fallback():
    engine, access = _nearby_frontier(0, 'subvoxel-collar')
    before = engine.snapshot(); source = engine.scene.labels.copy()
    actions = directional_removal_candidates(engine, access, limit=2)
    assert len(actions) == 2
    assert not _visits_outside(access, actions[0]) and _visits_outside(access, actions[1])
    endpoint = np.asarray(actions[1]['path'][-1]['position'])
    native = engine.scene.inverse[:3, :3] @ endpoint + engine.scene.inverse[:3, 3]
    # Integer outside rays cannot pass the collar; the saved outside alternative
    # is an actually checked quarter-voxel ray away from that bone face.
    assert native[1] == pytest.approx(10.25)
    for action in actions:
        preview = engine.preview(action, include_view=False)
        assert preview['status'] == 'accepted', preview
        assert preview['deltas']['removedCells']
    assert engine.snapshot() == before
    np.testing.assert_array_equal(engine.scene.labels, source)
