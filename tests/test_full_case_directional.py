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
