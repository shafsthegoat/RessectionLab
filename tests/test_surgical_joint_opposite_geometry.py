"""Generated opposite-face selection tests; no native transitions/patient inputs."""
import json
import numpy as np
import pytest
from resectionlab.surgical_scene import SurgicalScene

from resectionlab.surgical_joint_geometry import select_joint_corridors as select


def scene_for_opposite(*, affine=None):
    labels = np.zeros((144, 144, 144), np.uint16)
    labels[40:100, 40:100, 40:100] = 1
    labels[44:60, 64:80, 64:80] = 2
    regions = [{'id': 1, 'name': 'brain', 'material': 'soft_tissue', 'target': False},
               {'id': 2, 'name': 'tumor', 'material': 'tumor', 'target': True}]
    return SurgicalScene(labels, np.eye(4) if affine is None else affine,
        regions, {'caseId': 'generated-opposite-shorter'}, {})


def kwargs():
    return dict(goal='staged', max_routes=3, max_surface_samples=384,
                nearest_targets=4, max_geometry_pairs=1536)


def test_opposite_short_corridor_and_native_axis_identity(monkeypatch):
    from resectionlab.surgical_engine import SurgicalEngine
    def forbidden(*args, **kw):
        raise AssertionError('source-only test invoked native engine')
    monkeypatch.setattr(SurgicalEngine, 'step', forbidden)
    monkeypatch.setattr(SurgicalEngine, 'preview', forbidden)
    scene = scene_for_opposite()
    old_routes, old = select(scene, **kwargs())
    routes, report = select(scene, include_opposite_faces=True, **kwargs())
    assert not any(r['nativeAxis'] == 0 for r in old_routes)
    axis0 = next(r for r in routes if r['nativeAxis'] == 0)
    assert axis0['sourceFaceSign'] == -1
    assert axis0['inwardRasUnit'][0] > 0
    assert axis0['pathLengthMm'] < 20
    assert '-negative-' in axis0['id']
    selected = next(r for r in report['selected'] if r['nativeAxis'] == 0)
    assert selected['sourceFaceSign'] == -1
    assert selected['sourceSurfaceIndex'][0] == 40
    assert selected['sourceTargetBoundaryIndex'][0] > 41
    assert len(routes) <= 3 and len({r['nativeAxis'] for r in routes}) == len(routes)
    assert sum(report['counts']['surfacePointsSampled'].values()) <= 384
    assert report['counts']['pairsExamined'] <= 1536
    assert set(report['counts']['surfacePointsSampled']) == {'0:+1','0:-1','1:+1','1:-1','2:+1','2:-1'}
    for r in routes:
        s = next(x for x in report['selected'] if x['routeId'] == r['id'])
        idx = np.asarray(s['sourceSurfaceIndex'])
        neighbor = idx.copy(); neighbor[r['nativeAxis']] += r['sourceFaceSign']
        assert np.all(neighbor >= 0) and np.all(neighbor < scene.shape)
        assert scene.labels[tuple(idx)] == 1 and scene.labels[tuple(neighbor)] == 0
        for n in r['noveltyComparisons']:
            assert n['entrySeparationMm'] >= 12-1e-9 and n['inwardAngleDeg'] >= 30-1e-9


def test_affine_maps_index_face_sign_to_physical_inward():
    affine = np.array([[0., -1., 0., 28.], [-1., 0., 0., -43.],
                       [0., 0., 1., 7.], [0., 0., 0., 1.]])
    scene = scene_for_opposite(affine=affine)
    routes, report = select(scene, include_opposite_faces=True, **kwargs())
    route = next(r for r in routes if r['nativeAxis'] == 0)
    assert route['sourceFaceSign'] == -1 and route['inwardRasUnit'][1] < 0
    for route, row in zip(routes, report['selected']):
        idx = np.asarray(row['sourceSurfaceIndex'])
        np.testing.assert_allclose(route['surfacePose']['position'], affine[:3,:3]@idx+affine[:3,3])
        inward = np.subtract(route['entryPose']['position'], route['surfacePose']['position'])
        inward /= np.linalg.norm(inward)
        np.testing.assert_allclose(route['inwardRasUnit'], inward)


def test_default_preserves_positive_face_contract():
    scene = scene_for_opposite()
    default = select(scene, **kwargs())
    explicit = select(scene, include_opposite_faces=False, **kwargs())
    encode = lambda value: json.dumps(value, separators=(',', ':'))
    assert encode(default) == encode(explicit)
    routes, report = default
    assert 'includeOppositeFaces' not in report['selectionRule']
    assert set(report['counts']['surfacePointsSampled']) == {'0', '1', '2'}
    assert all('sourceFaceSign' not in r and '-negative-' not in r['id'] for r in routes)


@pytest.mark.parametrize('value', [0, 1, None, 'true', np.bool_(True)])
def test_opposite_flag_is_strict_bool(value):
    with pytest.raises(ValueError, match='must be boolean'):
        select(scene_for_opposite(), include_opposite_faces=value)


def test_shared_small_budget_and_cancellation():
    scene = scene_for_opposite()
    routes, report = select(scene, include_opposite_faces=True, goal='staged',
        max_surface_samples=7, nearest_targets=8, max_geometry_pairs=17)
    assert sum(report['counts']['surfacePointsSampled'].values()) <= 2
    assert report['counts']['pairsExamined'] <= 17
    calls = 0
    def cancel():
        nonlocal calls
        calls += 1
        if calls == 5:
            raise TimeoutError('cancel generated geometry')
    with pytest.raises(TimeoutError, match='cancel generated geometry'):
        select(scene, include_opposite_faces=True, check=cancel, **kwargs())
    assert calls == 5


def test_grid_boundary_is_not_an_observed_exterior_neighbor():
    source = scene_for_opposite()
    labels = source.labels.copy()
    labels[:40, 40:100, 40:100] = 1
    scene = SurgicalScene(labels, source.affine, source.regions,
        {'caseId': 'generated-brain-touches-grid-boundary'}, {})
    _, report = select(scene, include_opposite_faces=True, goal='staged',
        max_surface_samples=6, nearest_targets=1, max_geometry_pairs=6)
    assert report['counts']['surfacePointsAvailable']['0:-1'] == 0
    assert report['counts']['surfacePointsSampled']['0:-1'] == 0
    assert report['counts']['pairsExamined'] <= 6
