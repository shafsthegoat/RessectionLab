"""Independent analytic/index controls; no saved anatomy or native API calls."""
import importlib.util
import json
from pathlib import Path
import sys
from types import ModuleType, SimpleNamespace

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]


def module(name, source):
    spec = importlib.util.spec_from_file_location(name, ROOT / source)
    result = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(result)
    return result


D = module('review_saved_mesh_diagnostic', 'scripts/mechanics_patient_mesh_diagnostic.py')
BASE = module('review_original_surface_samples', 'scripts/mechanics_patient_mesh.py')
FACES = np.array([[1, 2, 3], [0, 3, 2], [0, 1, 3], [0, 2, 1]])


def test_heterogeneous_lattice_global_ordinals_match_original_across_chunk_boundary():
    # The small-triangle group exceeds 256 faces; other sizes are interleaved
    # in source indexing but grouped later by the original lattice iterator.
    vertices = []
    for i in range(263):
        scale = 3 if i in (0, 131, 262) else 1
        vertices.extend(np.array([[0., 0., 0.], [.001, 0., 0.], [0., .001, .0002]]) * scale + [i * .02, 0., 0.])
    vertices = np.array(vertices)
    faces = np.arange(len(vertices)).reshape(-1, 3)
    original = list(BASE.triangle_samples(vertices, faces, .001, 10000))
    expected_points = np.concatenate([points for points, _ in original])
    expected_distance = expected_points[:, 2]  # Exact distance to the z=0 plane.
    result, maxima = D.directed(vertices, faces, lambda points: points[:, 2], D.Budget())
    assert result['sample_count'] == len(expected_points)
    assert np.allclose(maxima, vertices[faces, 2].max(axis=1), rtol=0, atol=1e-18)
    expected_ids = np.lexsort((np.arange(len(expected_distance)), -expected_distance))[:10]
    assert [row['ordinal'] for row in result['worst_10']] == expected_ids.tolist()
    for row, ordinal in zip(result['worst_10'], expected_ids):
        assert np.array_equal(row['point_m'], expected_points[ordinal])
        assert row['triangle'] == round(expected_points[ordinal, 0] / .02)
    expected_bound = max(float(points[:, 2].max()) + cover for points, cover in original)
    assert result['full_surface_upper_bound_m'] == expected_bound


def test_area_quantiles_and_connected_witnesses_use_face_area_not_face_count():
    unit = np.array([[0., 0., 0.], [1., 0., 0.], [0., 1., 0.], [0., 0., 1.]])
    vertices = np.vstack([unit, 2 * unit + [10., 0., 0.]])
    faces = np.vstack([FACES, FACES + 4])
    maxima = np.r_[np.full(4, .003), np.zeros(4)]
    report, _, labels = D.summaries(vertices, faces, maxima)
    assert report['faces_with_sample_over_2mm'] == 4
    assert report['area_fraction_of_faces_with_sample_over_2mm'] == pytest.approx(.2)
    assert report['area_weighted_face_maximum_quantiles_m'] == {
        '0.5': 0., '0.9': .003, '0.95': .003, '0.99': .003, '1.0': .003}
    assert report['connected_witness_components'] == 1
    group = report['largest_10_components'][0]
    assert group['triangles'] == 4 and group['first_triangle'] == 0
    assert group['area_m2'] == pytest.approx((3 + np.sqrt(3)) / 2)
    assert len(set(labels[:4])) == 1 and (labels[4:] == -1).all()


def test_fixed_normal_change_proxy_distinguishes_30_and_60_degrees():
    # A tall octagonal bipyramid has adjacent face-normal changes strictly
    # between 30 and 60 degrees at side edges and smaller equator changes.
    angles = np.arange(8) * np.pi / 4
    ring = np.c_[np.cos(angles), np.sin(angles), np.zeros(8)]
    vertices = np.vstack([ring, [0., 0., 10.], [0., 0., -10.]])
    faces = np.array([[i, (i + 1) % 8, 8] for i in range(8)] +
                     [[(i + 1) % 8, i, 9] for i in range(8)])
    report, proxy, _ = D.summaries(vertices, faces, np.full(16, .003))
    assert ((proxy > 30) & (proxy < 60)).all()
    assert report['edge_normal_change_association'] == {
        '30': {'all_face_area_fraction': 1., 'witness_face_area_fraction': 1.},
        '60': {'all_face_area_fraction': 0., 'witness_face_area_fraction': 0.}}
    assert 'not smooth curvature' in report['interpretation']


def test_post_publication_cap_failure_cannot_leave_completed_acceptance(tmp_path, monkeypatch):
    monkeypatch.setattr(D, 'specification', lambda *_: (tmp_path, {}, {}))
    runtime = SimpleNamespace(private_environment=lambda _: {}, declaration=lambda: {})

    def supervised(_runtime, _command, output, _environment, caps):
        assert caps == {'aggregate_seconds': 120, 'process_group_rss_bytes': 2 * 1024**3}
        (output / 'report.json').write_text(json.dumps({'status': 'completed_saved_geometry_diagnostic_only'}))
        return {'status': 'completed', 'exit_code': 0}, {'error': None}

    def output_usage(output):
        if len(list(output.iterdir())) > 1:
            raise ValueError('final acceptance crosses file count cap')
        return {}

    guard = SimpleNamespace(supervised=supervised, output_usage=output_usage)
    monkeypatch.setattr(D, 'load', lambda _path, name: runtime if name == 'diagnostic_supervisor' else guard)
    output = tmp_path / 'attempt'
    assert D.run(tmp_path / 'release.json', output) == 1
    record = json.loads((output / 'acceptance.json').read_text())
    assert record['status'] == 'failed_or_incomplete'
    assert 'cap' in record['final_output_error']


@pytest.mark.parametrize('selected,actual', [(False, 'TBB'), (True, 'TBB')])
def test_sequential_backend_is_enforced_before_any_saved_array_load(tmp_path, monkeypatch, selected, actual):
    source = tmp_path / 'metadata.json'
    source.write_text('{}')
    monkeypatch.setattr(D, 'specification', lambda *_: (tmp_path, {'packages': {}, 'inputs': {}}, {str(source): D.sha(source)}))
    monkeypatch.setattr(D.resource, 'setrlimit', lambda *_: None)
    native = ModuleType('vtkmodules.vtkCommonCore')
    native.vtkSMPTools = SimpleNamespace(SetBackend=lambda _: selected, GetBackend=lambda: actual)
    monkeypatch.setitem(sys.modules, 'vtkmodules', ModuleType('vtkmodules'))
    monkeypatch.setitem(sys.modules, 'vtkmodules.vtkCommonCore', native)
    monkeypatch.setattr(D.np, 'load', lambda *_args, **_kwargs: pytest.fail('Saved array load before backend rejection'))
    monkeypatch.setattr(D, 'load', lambda *_: pytest.fail('Helper import before backend rejection'))
    assert D.worker(tmp_path / 'release.json', tmp_path) == 1
    report = json.loads((tmp_path / 'report.json').read_text())
    assert report['status'] == 'failed_or_incomplete'
    assert 'Sequential' in report['error']
