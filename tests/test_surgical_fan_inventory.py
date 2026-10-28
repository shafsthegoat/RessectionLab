"""Generated source geometry coupons. No native engine construction or transitions."""
import copy

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from resectionlab.core import semantic_digest
from resectionlab.geometry import GeometryScene, capsule_voxel_indices
from resectionlab.surgical_scene import SurgicalScene

import resectionlab.surgical_fan_inventory as fan


def coupon():
    labels = np.zeros((48, 48, 48), np.uint16)
    labels[18:29, 18:29, 18:29] = 2
    for axis in range(3):
        slices = [slice(22, 25)]*3
        slices[axis] = slice(14, 18)
        labels[tuple(slices)] = 1
    scene = SurgicalScene(labels, np.eye(4), [
        {'id': 1, 'name': 'brain', 'material': 'soft_tissue'},
        {'id': 2, 'name': 'tumor', 'material': 'tumor', 'target': True}],
        {'kind': 'generated source-only three-axis fan coupon'}, {})
    routes = []
    for axis in range(3):
        inward = np.eye(3)[axis]
        surface = np.array([23., 23., 23.]); surface[axis] = 14.
        entry = surface+4*inward
        quaternion = Rotation.align_vectors([inward], [[0., 0., 1.]])[0].as_quat().tolist()
        pose = lambda p: {'position': p.tolist(), 'quaternion': quaternion}
        routes.append({'id': 'generated-axis-'+str(axis), 'nativeAxis': axis,
            'sourceSceneId': scene.identity, 'surfacePose': pose(surface),
            'entryPose': pose(entry), 'outsidePose': pose(surface-8*inward),
            'inwardRasUnit': inward.tolist(), 'pathLengthMm': 4.,
            'corridorRadiusMm': .6, 'pathType': 'oblique_mapped_brain',
            'accessCompleteness': 'unsupported_skull_access_brain_corridor'})
    return scene, routes


def cover(scene, start, end, radius):
    grid = GeometryScene(np.zeros(scene.shape, bool), scene.affine)
    ix = capsule_voxel_indices(grid, start, end, radius)
    return {int(i) for i in np.ravel_multi_index(ix.T, scene.shape) if scene.labels.ravel()[i]}


def test_exact_shared_ledger_source_immutability_and_selected_route_geometry(monkeypatch):
    from resectionlab.surgical_engine import SurgicalEngine
    def forbidden(*_args, **_kwargs):
        raise AssertionError('Pure source proposer cannot construct or transition a native engine')
    for method in ('__init__', 'step', 'preview'):
        monkeypatch.setattr(SurgicalEngine, method, forbidden)
    scene, routes = coupon()
    before, labels = copy.deepcopy(routes), scene.labels.copy()
    inventory = fan.propose_source_fans(scene, routes, max_stages=4)
    assert routes == before and np.array_equal(scene.labels, labels)
    assert inventory['nativeSelectionCalls'] == 0
    assert inventory['counts']['rayConfigurationsExamined'] <= 60000
    assert len(inventory['fans']) == 3
    assert {r['route']['nativeAxis'] for r in inventory['fans']} == {0, 1, 2}
    assert len(inventory['fanSummaries']) == 12
    assert all('orderedRays' not in r and 'initialTipSourceCellIds' not in r for r in inventory['fanSummaries'])
    assert inventory['inventorySeal'] == semantic_digest({k: v for k, v in inventory.items() if k != 'inventorySeal'})
    overlap_seen = False
    for proposal in inventory['fans']:
        assert proposal['route']['corridorRadiusMm'] == proposal['initialRadiusMm']
        assert proposal['sourceRouteSeal'] == semantic_digest(proposal['route'])
        assert proposal['sourceContactSeal']
        initial = proposal['initialRaySpec']
        virtual = cover(scene, initial['outsidePose']['position'], initial['endpointPose']['position'], initial['radiusMm'])
        initial_brain = {i for i in virtual if scene.labels.ravel()[i] == 1}
        disjoint = set()
        individual_target_sum = sum(scene.labels.ravel()[i] == 2 for i in virtual)
        for ray in proposal['orderedRays']:
            s = ray['raySpec']
            actual = cover(scene, s['outsidePose']['position'], s['endpointPose']['position'], s['radiusMm'])
            assert actual == set(ray['tipSourceCellIds'])
            mt = {i for i in actual if scene.labels.ravel()[i] == 2}-virtual
            assert mt == set(ray['marginalTargetSourceCellIds']) and not disjoint & mt
            disjoint |= mt
            individual_target_sum += sum(scene.labels.ravel()[i] == 2 for i in actual)
            virtual |= actual
            brains = {i for i in virtual if scene.labels.ravel()[i] == 1}
            assert len(brains) == ray['cumulativeBrainMm3'] <= 200
            assert len(brains-initial_brain) == ray['additionalBrainFromInitialMm3'] <= 20
        assert virtual == set(proposal['virtualRemovedSourceCellIds'])
        assert individual_target_sum >= proposal['cumulativeTargetMm3']
        overlap_seen |= individual_target_sum > proposal['cumulativeTargetMm3']
        for comp in proposal['noveltyComparisons']:
            assert comp['entrySeparationMm'] >= 12-1e-9 and comp['inwardAngleDeg'] >= 30-1e-9
    assert overlap_seen, 'Coupon must distinguish cumulative ledger from independent ray sums'


def test_deterministic_permutation_deduplication_and_original_radius_variants():
    scene, routes = coupon()
    baseline = fan.propose_source_fans(scene, routes, max_stages=2)
    assert fan.propose_source_fans(scene, list(reversed(routes)), max_stages=2) == baseline
    duplicate = copy.deepcopy(routes[0]); duplicate['id'] = 'another-ID-same-physical-ray'
    duplicate['corridorRadiusMm'] = 1.5
    result = fan.propose_source_fans(scene, [*routes, duplicate], max_stages=2)
    assert result['counts']['distinctRouteGeometries'] == 3
    assert len(result['fans']) == 3
    assert any(r['reason'] == 'DUPLICATE_PHYSICAL_ROUTE' for r in result['inputRouteFailures'])


def test_hard_geometry_budget_and_cancellation_preserve_inputs(monkeypatch):
    scene, routes = coupon()
    monkeypatch.setattr(fan, 'MAX_RAY_CONFIGURATIONS', 8)
    result = fan.propose_source_fans(scene, routes, max_stages=1)
    assert result['counts']['rayConfigurationsExamined'] == 8
    assert result['candidateBudgetExhausted']
    assert any(not r['candidateBankComplete'] for r in result['routeSummaries'])
    calls = 0
    before = copy.deepcopy(routes)
    def cancel():
        nonlocal calls
        calls += 1
        if calls == 15:
            raise InterruptedError('generated cancellation')
    with pytest.raises(InterruptedError):
        fan.propose_source_fans(scene, routes, check=cancel)
    assert routes == before


def test_input_source_binding_and_bounds():
    scene, routes = coupon()
    with pytest.raises(ValueError):
        fan.propose_source_fans(scene, routes*5)
    with pytest.raises(ValueError):
        fan.propose_source_fans(scene, routes, max_stages=13)
    wrong = copy.deepcopy(routes[0]); wrong['sourceSceneId'] = 'different-source'
    output = fan.propose_source_fans(scene, [wrong])
    assert output['fans'] == []
    assert output['inputRouteFailures'][0]['reason'] == 'SOURCE_SCENE_MISMATCH'


def test_failed_static_source_geometry_is_preserved():
    source, routes = coupon()
    for marker, expected in ((1, 'INITIAL_OUTSIDE_OCCUPIED'), (3, 'INITIAL_UNSUPPORTED_LABEL')):
        labels = source.labels.copy()
        labels[(6, 23, 23) if marker == 1 else (16, 23, 23)] = marker
        regions = source.regions+([{'id': 3, 'name': 'vessel', 'material': 'vessel'}] if marker == 3 else [])
        scene = SurgicalScene(labels, source.affine, regions, {'kind': 'generated failed geometry coupon'}, {})
        route = copy.deepcopy(routes[0]); route['sourceSceneId'] = scene.identity
        result = fan.propose_source_fans(scene, [route])
        assert not result['fans']
        assert all(r['reason'] == expected for r in result['routeSummaries'][0]['initialConfigurationFailures'])


def test_empty_source_does_not_make_zero_target_strategies():
    source, routes = coupon()
    labels = np.zeros(source.shape, np.uint16)
    scene = SurgicalScene(labels, source.affine, source.regions, {'kind': 'generated empty source coupon'}, {})
    route = copy.deepcopy(routes[0]); route['sourceSceneId'] = scene.identity
    result = fan.propose_source_fans(scene, [route])
    assert result['fans'] == [] and result['status'] == 'no_source_target'
