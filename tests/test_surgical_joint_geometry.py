"""Meaningful source-geometry shortlist checks; no native transitions."""
from __future__ import annotations

import numpy as np
import pytest

from resectionlab.geometry import GeometryScene, capsule_voxel_indices
from resectionlab.surgical_joint_geometry import (
    _fixed_far_first_contact, _geometry_burden, _staged_work_tube_contact,
    _working_sphere,
    select_joint_corridors,
)
from resectionlab.surgical_scene import SurgicalScene
from resectionlab.surgical_sector_planning import _sector_cells


def _two_equal_targets(*, affine=None, small_target=False):
    """Equal source targets behind a short and a long mapped-brain corridor."""
    labels = np.zeros((96, 130, 130), np.uint16)
    labels[30:52, 5:55, 30:100] = 1
    labels[25:70, 70:120, 30:100] = 1
    if small_target:
        labels[34:36, 29:31, 64:66] = 2
        labels[29:31, 94:96, 64:66] = 2
    else:
        labels[30:40, 22:38, 57:73] = 2
        labels[25:35, 87:103, 57:73] = 2
    regions = [{'id': 1, 'name': 'brain', 'material': 'soft_tissue', 'target': False},
               {'id': 2, 'name': 'tumor', 'material': 'tumor', 'target': True}]
    return SurgicalScene(labels, np.eye(4) if affine is None else affine,
                         regions, {'caseId': 'generated-two-equal-targets'}, {})


def _select(scene, **kwargs):
    return select_joint_corridors(scene, max_surface_samples=256,
                                  max_geometry_pairs=2048, **kwargs)


def _equal_working_targets_unequal_brain():
    """Same tip target density; short route has brain around working lobes."""
    labels = np.zeros((96, 130, 130), np.uint16)
    labels[30:52, 5:55, 30:100] = 1
    labels[35:70, 70:120, 30:100] = 1
    for start, stop, cy, role in ((30, 40, 30, 'short'),
                                  (25, 35, 95, 'long')):
        for y in range(cy-4, cy+5):
            for z in range(61, 70):
                dy, dz = y-cy, z-65
                inner = (dy*dy+dz*dz)**.5 <= 1.5
                short_ring = abs(dy) == 2 and abs(dz) <= 1
                long_ring = short_ring or (abs(dz) == 2 and abs(dy) <= 1)
                if inner or (short_ring if role == 'short' else long_ring):
                    labels[start:stop, y, z] = 2
    regions = [{'id': 1, 'name': 'brain', 'material': 'soft_tissue', 'target': False},
               {'id': 2, 'name': 'tumor', 'material': 'tumor', 'target': True}]
    return SurgicalScene(labels, np.eye(4), regions,
                         {'caseId': 'generated-working-exposure-contrast'}, {})


def test_shorter_corridor_wins_equal_target_coverage_without_native_steps(monkeypatch):
    from resectionlab.surgical_engine import SurgicalEngine
    def forbidden(*_args, **_kwargs):
        raise AssertionError('Source-only selection called the native engine')
    monkeypatch.setattr(SurgicalEngine, 'step', forbidden)
    monkeypatch.setattr(SurgicalEngine, 'preview', forbidden)
    scene = _two_equal_targets()
    routes, report = _select(scene, max_routes=1)
    assert len(routes) == 1
    assert routes[0]['nativeAxis'] == 0
    assert routes[0]['surfacePose']['position'][0] == 51.
    assert routes[0]['pathLengthMm'] < 20.
    assert report['selected'][0]['corridorAndShaftBrainUnionMm3'] < 200.
    assert report['counts']['pairsExamined'] <= 2048
    assert report['status'] == 'selected'
    assert report['sourceBasis']['skullSourceAvailable'] is False


def test_routes_bind_affine_scene_and_original_sector_windows():
    affine = np.array([[0., -1., 0., 28.],
                       [1., 0., 0., -43.],
                       [0., 0., 1., 7.],
                       [0., 0., 0., 1.]])
    scene = _two_equal_targets(affine=affine)
    routes, report = _select(scene, max_routes=3)
    assert len(routes) == 3
    assert {route['nativeAxis'] for route in routes} == {0, 1, 2}
    for route, selected in zip(routes, report['selected']):
        surface_index = np.asarray(selected['sourceSurfaceIndex'])
        expected = affine[:3, :3] @ surface_index + affine[:3, 3]
        np.testing.assert_allclose(route['surfacePose']['position'], expected)
        assert route['sourceSceneId'] == scene.identity
        assert route['pathType'] == 'oblique_mapped_brain'
        assert route['accessCompleteness'] == 'unsupported_skull_access_brain_corridor'
        _, evidence = _sector_cells(scene, route)
        assert all(evidence[name]['sourceVolumeMm3'] >= 10. for name in ('near', 'far'))
        assert selected['farShaftTailDepthMm'] <= 1e-9
        assert selected['mathematicalMinimumNonCorridorActions'] == 8
        assert selected['heuristicActionHeadroom'] > 0
        for novelty in selected['noveltyComparisons']:
            assert novelty['entrySeparationMm'] >= 12.-1e-9
            assert novelty['inwardAngleDeg'] >= 30.-1e-9


def test_insufficient_source_capacity_is_reported_without_inventing_routes():
    scene = _two_equal_targets(small_target=True)
    routes, report = _select(scene)
    assert routes == []
    assert report['status'] == 'insufficient_routes'
    assert report['counts']['selectedRoutes'] == 0
    assert report['rejectionHistogram'].get('SECTOR_CAPACITY', 0) > 0
    assert report['sourceBasis']['sceneId'] == scene.identity
    assert report['selectionSeal'].startswith('sha256:')


def test_selection_is_deterministic_and_check_can_cancel():
    scene = _two_equal_targets()
    first_routes, first_report = _select(scene, max_routes=3)
    second_routes, second_report = _select(scene, max_routes=3)
    assert first_routes == second_routes
    assert first_report == second_report
    assert first_report['selectionSeal'] == second_report['selectionSeal']
    calls = 0
    def cancel():
        nonlocal calls
        calls += 1
        if calls == 5:
            raise TimeoutError('caller wall budget')
    with pytest.raises(TimeoutError, match='caller wall budget'):
        _select(scene, max_routes=3, check=cancel)
    assert calls == 5


def test_working_brain_exposure_beats_equal_tip_density_and_shorter_corridor(monkeypatch):
    from resectionlab.surgical_engine import SurgicalEngine
    def forbidden(*_args, **_kwargs):
        raise AssertionError('Source-only contact score called native physics')
    monkeypatch.setattr(SurgicalEngine, 'step', forbidden)
    monkeypatch.setattr(SurgicalEngine, 'preview', forbidden)
    scene = _equal_working_targets_unequal_brain()
    grid = GeometryScene(np.zeros(scene.shape, bool), scene.affine)
    brain_flat = (scene.labels == 1).ravel()
    target_flat = (scene.labels == 2).ravel()
    candidates = {}
    for role, surface, goal in (
            ('short', np.array([51., 30., 65.]), np.array([39., 30., 65.])),
            ('long', np.array([69., 95., 65.]), np.array([34., 95., 65.]))):
        inward = np.array([-1., 0., 0.])
        length = float(np.linalg.norm(goal-surface))
        route = {'id': role, 'sourceSceneId': scene.identity,
                 'pathType': 'oblique_mapped_brain',
                 'accessCompleteness': 'unsupported_skull_access_brain_corridor',
                 'surfacePose': {'position': surface.tolist()},
                 'inwardRasUnit': inward.tolist(), 'pathLengthMm': length}
        sectors, evidence = _sector_cells(scene, route)
        assert all(evidence[name]['sourceVolumeMm3'] >= 10. for name in sectors)
        item = {'surfaceRasMm': surface.tolist(), 'inwardRasUnit': inward.tolist(),
                'pathLengthMm': length,
                '_nearSectorFlatIds': list(sectors['near']),
                '_farSectorFlatIds': list(sectors['far'])}
        burden, corridor, corridor_centers = _geometry_burden(
            scene, grid, item, [1], [2])
        macro = _fixed_far_first_contact(scene, grid, item, corridor,
                                         corridor_centers, brain_flat, target_flat)
        near_target, _ = _working_sphere(scene, goal+inward, 1.6, [1], [2])
        far_target, _ = _working_sphere(scene, goal+4.*inward, 1.8, [1], [2])
        candidates[role] = (length, (near_target, far_target), burden, macro)
    short, long = candidates['short'], candidates['long']
    assert short[1] == long[1] == (19, 27)
    assert short[0] < long[0]
    assert short[2]['corridorAndShaftBrainUnionMm3'] < long[2]['corridorAndShaftBrainUnionMm3']
    assert short[3]['farFirstAdditionalBrainSupercoverMm3'] > 20.
    assert long[3]['farFirstAdditionalBrainSupercoverMm3'] < 20.
    assert short[3]['farFirstJointSourceDeficitMm3'] > long[3]['farFirstJointSourceDeficitMm3']
    # The far transit touches source brain, but all of that portion lies in
    # the source corridor supercover and is excluded from *additional* contact.
    transit = capsule_voxel_indices(grid, np.array([39.,30.,65.]),
                                    np.array([35.,30.,65.]), 1.)
    assert np.count_nonzero(scene.labels[tuple(transit.T)] == 1) == 9
    assert short[3]['farFirstPhaseBrainBeyondCorridorMm3']['farTransit'] == 0.
    routes, report = _select(scene, max_routes=1)
    assert len(routes) == 1
    assert report['selected'][0]['sourceSurfaceIndex'][0] == 69
    assert report['selected'][0]['farFirstJointSourceDeficitMm3'] == 0.


def test_bounds_reject_invalid_requests():
    scene = _two_equal_targets()
    with pytest.raises(ValueError, match='positive integers'):
        _select(scene, max_routes=0)
    with pytest.raises(ValueError, match='three native axes'):
        _select(scene, max_routes=4)
    with pytest.raises(ValueError, match='goal must be'):
        _select(scene, goal='unbound')


def test_staged_goal_retains_physical_source_cuts_without_tiny_sector_gate(monkeypatch):
    from resectionlab.surgical_engine import SurgicalEngine
    def forbidden(*_args, **_kwargs):
        raise AssertionError('Staged source selection called native physics')
    monkeypatch.setattr(SurgicalEngine, 'step', forbidden)
    monkeypatch.setattr(SurgicalEngine, 'preview', forbidden)
    scene = _two_equal_targets(small_target=True)
    old_routes, _ = _select(scene, goal='two-sector')
    staged_routes, report = _select(scene, goal='staged')
    assert old_routes == []
    assert len(staged_routes) == 3
    assert {row['nativeAxis'] for row in staged_routes} == {0, 1, 2}
    assert report['selectionRule']['selectionGoal'] == 'staged'
    assert report['selectionRule']['minSourceSectorVolumeMm3'] is None
    assert report['selectionRule']['minWorkingTargetMm3'] is None
    assert report['selectionRule']['stagedMaxNativeActions'] == 240
    assert report['counts']['pairsExamined'] <= 2048
    assert report['rejectionHistogram'].get('SECTOR_CAPACITY', 0) == 0
    assert report['rejectionHistogram'].get('WORKING_TARGET_FOOTPRINT', 0) == 0
    assert any(min(row['sourceSectorCapacityMm3'].values()) == 0.
               for row in report['selected'])
    for route, selected in zip(staged_routes, report['selected']):
        assert route['sourceSceneId'] == scene.identity
        assert route['pathType'] == 'oblique_mapped_brain'
        assert route['accessCompleteness'] == 'unsupported_skull_access_brain_corridor'
        assert selected['farShaftTailDepthMm'] <= 1e-9
        assert selected['corridorCenterContactBrainMm3'] <= 200.
        assert selected['rankKey'][:3] == [
            selected['stagedJointSourceDeficitMm3'],
            selected['corridorAndShaftBrainUnionMm3'],
            selected['pathLengthMm']]
        assert selected['stagedJointSourceDeficitMm3'] == (
            selected['stagedContinuationBrainExcessMm3']+
            selected['stagedTargetContactDeficitMm3'])
        assert selected['stagedWorkTubeRadiusMm'] == route['corridorRadiusMm']
        assert selected['stagedWorkTubeOffsetFromTargetEntryMm'][0] == 1.
        assert selected['stagedWorkTubeOffsetFromTargetEntryMm'][1] <= 24.
        _, evidence = _sector_cells(scene, route)
        assert selected['sourceSectorCapacityMm3'] == {
            name: evidence[name]['sourceVolumeMm3'] for name in ('near', 'far')}
        for novelty in selected['noveltyComparisons']:
            assert novelty['entrySeparationMm'] >= 12.-1e-9
            assert novelty['inwardAngleDeg'] >= 30.-1e-9


def test_staged_rank_uses_extended_work_contact_before_whole_corridor_burden():
    scene = _equal_working_targets_unequal_brain()
    old_routes, old = _select(scene, max_routes=1)
    staged_routes, staged = _select(scene, max_routes=1, goal='staged')
    assert len(old_routes) == len(staged_routes) == 1
    assert staged['selected'][0]['corridorAndShaftBrainUnionMm3'] < \
        old['selected'][0]['corridorAndShaftBrainUnionMm3']
    assert staged['selected'][0]['rankKey'][0] == \
        staged['selected'][0]['stagedJointSourceDeficitMm3']
    assert staged['selected'][0]['rankKey'][1] == \
        staged['selected'][0]['corridorAndShaftBrainUnionMm3']
    reference = staged['selectionRule']['stagedSourceTargetContactReferenceMm3']
    selected = staged['selected'][0]
    assert selected['stagedTargetContactDeficitMm3'] == max(
        0., reference-selected['stagedWorkTubeAdditionalTargetSupercoverMm3'])
    repeated_routes, repeated = _select(scene, max_routes=1, goal='staged')
    assert repeated_routes == staged_routes and repeated == staged
    assert repeated['selectionSeal'].startswith('sha256:')


def test_staged_tube_counts_union_beyond_access_capsule_at_original_radius():
    scene = _two_equal_targets()
    route = _select(scene, max_routes=1, goal='staged')[0][0]
    grid = GeometryScene(np.zeros(scene.shape, bool), scene.affine)
    item = {'surfaceRasMm': route['surfacePose']['position'],
            'inwardRasUnit': route['inwardRasUnit'],
            'pathLengthMm': route['pathLengthMm']}
    _, corridor, _ = _geometry_burden(scene, grid, item, [1], [2])
    brain = (scene.labels == 1).ravel()
    target = (scene.labels == 2).ravel()
    measured = _staged_work_tube_contact(scene, grid, item, corridor,
        brain, target, 500.)
    inward = np.asarray(route['inwardRasUnit'])
    entry = np.asarray(route['entryPose']['position'])
    low, high = measured['stagedWorkTubeOffsetFromTargetEntryMm']
    ijk = capsule_voxel_indices(grid, entry+inward*low, entry+inward*high,
                                route['corridorRadiusMm'])
    full = set(map(int, np.ravel_multi_index(ijk.T, scene.shape)))
    additional = np.fromiter(full-corridor, dtype=np.int64)
    assert measured['stagedWorkTubeSourceCellCount'] == len(full)
    assert measured['stagedDeepShaftSourceLabelsAllowed'] is True
    assert measured['stagedDeepShaftTailDepthMm'] <= -2.
    assert measured['stagedWorkTubeBeyondCorridorCellCount'] == len(additional)
    assert measured['stagedWorkTubeAdditionalTargetSupercoverMm3'] == \
        np.count_nonzero(target[additional])*scene.voxel_mm3
    assert measured['stagedWorkTubeAdditionalBrainSupercoverMm3'] == \
        np.count_nonzero(brain[additional])*scene.voxel_mm3
    assert measured['stagedWorkTubeAdditionalTargetSupercoverMm3'] > 0.


def test_staged_deep_shaft_detects_newly_reached_unmapped_label():
    scene = _two_equal_targets()
    route = _select(scene, max_routes=1, goal='staged')[0][0]
    grid = GeometryScene(np.zeros(scene.shape, bool), scene.affine)
    surface = np.asarray(route['surfacePose']['position'])
    inward = np.asarray(route['inwardRasUnit'])
    length = route['pathLengthMm']
    far_offset = min(24., 38.-length)
    old_tip = surface+inward*(length+4.)
    deep_tip = surface+inward*(length+far_offset)
    old_shaft = capsule_voxel_indices(grid, old_tip-inward*40.,
                                      old_tip-inward*2., .5)
    deep_shaft = capsule_voxel_indices(grid, deep_tip-inward*40.,
                                       deep_tip-inward*2., .5)
    old_cells = set(map(tuple, old_shaft))
    newly_reached = sorted(set(map(tuple, deep_shaft))-old_cells)
    assert newly_reached
    labels = scene.labels.copy()
    labels[newly_reached[0]] = 3
    regions = [*scene.regions,
               {'id': 3, 'name': 'unmapped', 'material': 'bone', 'target': False}]
    altered = SurgicalScene(labels, scene.affine, regions,
                            {'caseId': 'generated-unmapped-deep-shaft'}, {})
    item = {'surfaceRasMm': surface.tolist(),
            'inwardRasUnit': inward.tolist(), 'pathLengthMm': length}
    _, corridor, _ = _geometry_burden(altered, grid, item, [1], [2])
    assert _staged_work_tube_contact(altered, grid, item, corridor,
        (altered.labels == 1).ravel(), (altered.labels == 2).ravel(),
        500.)['stagedDeepShaftSourceLabelsAllowed'] is False
