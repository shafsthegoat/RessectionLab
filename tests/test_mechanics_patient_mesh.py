"""Analytic arrays and mocked APIs only; no patient access or native meshing."""
import copy
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace
import sys

import numpy as np
import pytest
from scipy.spatial import cKDTree

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('patient_mesh_controls', ROOT / 'scripts/mechanics_patient_mesh.py')
v = importlib.util.module_from_spec(spec); spec.loader.exec_module(v)
PROTOCOL = json.loads((ROOT / v.MANIFEST).read_text())


def tetrahedron():
    X = np.array([[0., 0, 0], [.01, 0, 0], [0, .01, 0], [0, 0, .01]])
    return X, v.FACES.copy()


def quadratic_tetrahedron():
    X, _ = tetrahedron()
    return np.vstack((X, X[v.EDGES].mean(axis=1))), np.arange(10)[None, :]


def test_closed_surface_volume_orientation_and_genus():
    X, F = tetrahedron(); report = v.validate_surface(X, F)
    assert report['enclosed_volume_m3'] == pytest.approx(1e-6 / 6)
    assert report['components'] == 1 and report['genus'] == 0
    with pytest.raises(ValueError, match='Outward'):
        v.validate_surface(X, F[:, ::-1])


@pytest.mark.parametrize('corruption', ['open', 'duplicate', 'wrong_orientation'])
def test_surface_topology_faults_fail(corruption):
    X, F = tetrahedron()
    if corruption == 'open': F = F[:-1]
    elif corruption == 'duplicate': F = np.vstack((F, F[0]))
    else: F[0] = F[0, ::-1]
    with pytest.raises(ValueError): v.validate_surface(X, F)


def test_pinched_vertex_is_rejected_even_when_every_edge_has_two_faces():
    X, F = tetrahedron()
    nodes = np.vstack((X, -X[1:]))
    second = np.array([0, 4, 5, 6])[F[:, ::-1]]
    with pytest.raises(ValueError, match='vertex fan'):
        v.validate_surface(nodes, np.vstack((F, second)))


def test_native_transform_and_fixed_extraction_options_are_forwarded_to_mock():
    mask = np.zeros((4, 4, 4), np.uint8); mask[1:3, 1:3, 1:3] = 1
    points, faces = tetrahedron(); points = points * 100 + 1
    calls = []
    def fake_marching(data, **kwargs):
        calls.append(kwargs)
        np.testing.assert_array_equal(data, mask.astype(np.float32))
        return points, faces, None, None
    affine = np.array([[0., -1, 0, 80], [1, 0, 0, -90], [0, 0, 1, 30], [0, 0, 0, 1]])
    X, F, report = v.extract_native_surface(mask, affine, fake_marching)
    np.testing.assert_allclose(X, (points @ affine[:3, :3].T + affine[:3, 3]) / 1000, atol=0, rtol=0)
    assert calls == [{'level': .5, 'spacing': (1., 1., 1.), 'gradient_direction': 'descent',
                      'step_size': 1, 'allow_degenerate': False, 'method': 'lewiner'}]
    assert not report['global_orientation_reversed'] and report['voxel_mask_volume_m3'] == pytest.approx(8e-9)


def test_boundary_contact_and_nonbinary_mask_fail_before_extraction():
    never = lambda *a, **k: pytest.fail('No extraction allowed')
    mask = np.zeros((3, 3, 3)); mask[0, 1, 1] = 1
    with pytest.raises(ValueError, match='boundary'): v.extract_native_surface(mask, np.eye(4), never)
    mask[:] = 0; mask[1, 1, 1] = .6
    with pytest.raises(ValueError, match='binary'): v.extract_native_surface(mask, np.eye(4), never)


def test_barycentric_lattice_covers_interiors_and_edges():
    X = np.array([[0., 0, 0], [.009, 0, 0], [.001, .006, 0]])
    chunks = list(v.triangle_samples(X, np.array([[0, 1, 2]]), .001, 1000))
    samples = np.vstack([points for points, _ in chunks]); bound = max(b for _, b in chunks)
    assert len(samples) > 3 and bound <= .001
    bary = np.array([(i/120, j/120, 1-(i+j)/120) for i in range(121) for j in range(121-i)])
    distance, _ = cKDTree(samples).query(bary @ X)
    assert distance.max() <= bound
    np.testing.assert_array_equal(samples.min(axis=0), X.min(axis=0))
    np.testing.assert_array_equal(samples.max(axis=0), X.max(axis=0))


def test_distance_gate_retains_cover_bound_and_preallocation_cap():
    X, F = tetrahedron()
    config = copy.deepcopy(PROTOCOL['surface_fidelity'])
    actual = v.directed_surface_bound(X, F, lambda q: np.full(len(q), .0012), config)
    assert actual['maximum_sample_distance_m'] == pytest.approx(.0012)
    assert actual['full_surface_upper_bound_m'] > config['maximum_distance_m']
    with pytest.raises(ValueError, match='before allocation'):
        list(v.triangle_samples(X, F, .001, 10))
    with pytest.raises(ValueError, match='Invalid'):
        v.directed_surface_bound(X, F, lambda q: np.full(len(q), np.nan), config)


def test_positive_tet10_full_geometry_and_source_order():
    X, E = quadratic_tetrahedron()
    report, boundary, faces = v.validate_tet10(X, E, PROTOCOL['levels'][0], PROTOCOL)
    assert report['minimum_reference_determinant_m3'] == pytest.approx(1e-6)
    assert report['minimum_actual_sampled_reference_determinant_m3'] == pytest.approx(1e-6)
    assert report['jacobian_witnesses_per_element'] == 19
    assert report['maximum_midpoint_error_m'] == 0
    assert report['boundary']['genus'] == 0 and len(boundary) == 4 and len(faces) == 4
    permutation = np.array([0, 1, 2, 3, 4, 5, 6, 7, 9, 8])
    np.testing.assert_array_equal(v.tet10_permutation(X[permutation] / .01), permutation)


@pytest.mark.parametrize('corruption', ['curved', 'inverted', 'duplicate', 'index', 'node_cap'])
def test_invalid_volume_fails(corruption):
    X, E = quadratic_tetrahedron(); level = copy.deepcopy(PROTOCOL['levels'][0])
    if corruption == 'curved': X[4, 0] += 1e-5
    elif corruption == 'inverted': X[:, 0] *= -1
    elif corruption == 'duplicate': E = np.vstack((E, E))
    elif corruption == 'index': E[0, 0] = len(X)
    else: level['maximum_nodes'] = 9
    with pytest.raises(ValueError): v.validate_tet10(X, E, level, PROTOCOL)


def test_tetrahedron_sat_detects_overlap_and_accepts_disjoint_aabb_overlap():
    X, _ = tetrahedron()
    cells = np.arange(8).reshape(2, 4)
    separate = np.vstack((X, X + .004))
    result = v.reject_tetrahedral_overlap(separate, cells, maximum_pairs=10)
    assert result['aabb_candidate_pairs_checked'] == 1
    overlap = np.vstack((X, X + .002))
    with pytest.raises(ValueError, match='overlap detected'):
        v.reject_tetrahedral_overlap(overlap, cells, maximum_pairs=10)
    with pytest.raises(ValueError, match='candidate-pair cap'):
        v.reject_tetrahedral_overlap(overlap, cells, maximum_pairs=0)


def test_face_contact_is_allowed_without_positive_volume_overlap():
    X, _ = tetrahedron(); nodes = np.vstack((X, [[0, 0, -.01]]))
    v.reject_tetrahedral_overlap(nodes, np.array([[0, 1, 2, 3], [0, 2, 1, 4]]), maximum_pairs=10)


def test_frozen_mesh_caps_leave_explicit_skyline_value_storage_headroom():
    assert [x['target_edge_m'] for x in PROTOCOL['levels']] == [.024, .020, .016]
    assert [x['maximum_nodes'] for x in PROTOCOL['levels']] == [2000, 3000, 4500]
    estimate = v.skyline_storage(4500)
    assert estimate['three_value_arrays_bytes'] == 2187162000
    assert PROTOCOL['caps']['process_group_rss_bytes'] - estimate['three_value_arrays_bytes'] > 1e9
    assert PROTOCOL['gmsh_options']['Mesh.SecondOrderLinear'] == 1
    assert PROTOCOL['gmsh_options']['Mesh.ElementOrder'] == 2
    assert PROTOCOL['caps']['retries'] == 0


def test_missing_release_fails_before_any_input_or_native_module(tmp_path, monkeypatch):
    release = tmp_path / 'not-released.json'; release.write_text(json.dumps({'authorized': False}))
    monkeypatch.setattr(v, 'sha', lambda *_: pytest.fail('No input digest access'))
    monkeypatch.setattr(v, 'import_file', lambda *_: pytest.fail('No native import'))
    with pytest.raises(ValueError, match='Separate root release'):
        v.prepare_release(release, tmp_path / 'attempt')


def mocked_worker(tmp_path, monkeypatch, generation_fails=False, corrupt_geometry=False):
    """Mock all I/O-heavy/native API entry points; reuse explicit analytic tet."""
    output = tmp_path / 'attempt'; output.mkdir()
    marker = tmp_path / 'bound'; marker.write_text('fixed')
    protocol = copy.deepcopy(PROTOCOL); protocol['package_versions'] = {}
    bound = {str(marker): v.sha(marker)}
    monkeypatch.setattr(v, 'prepare_release', lambda *_: (protocol, bound))
    image = SimpleNamespace(shape=tuple(protocol['mask']['shape']), affine=np.array(protocol['mask']['affine_ras_mm']),
        header=SimpleNamespace(get_xyzt_units=lambda: ('mm', 'unknown')), dataobj=np.zeros((2, 2, 2)))
    monkeypatch.setitem(sys.modules, 'nibabel', SimpleNamespace(load=lambda _: image))
    monkeypatch.setitem(sys.modules, 'skimage.measure', SimpleNamespace(marching_cubes=lambda *_: None))
    smp = SimpleNamespace(SetBackend=lambda _: True, GetBackend=lambda: 'Sequential')
    monkeypatch.setitem(sys.modules, 'vtkmodules.vtkCommonCore', SimpleNamespace(vtkSMPTools=smp))
    vertices, faces = tetrahedron()
    monkeypatch.setattr(v, 'extract_native_surface', lambda *_: (vertices, faces, v.validate_surface(vertices, faces)))
    native = SimpleNamespace(__file__=protocol['gmsh_runtime']['module_path'],
        lib=SimpleNamespace(_name=protocol['gmsh_runtime']['library_path']), __version__='4.15.2',
        initialize=lambda *a, **k: None, finalize=lambda: None,
        write=lambda path: Path(path).write_text('MOCK NATIVE OUTPUT'))
    monkeypatch.setattr(v, 'import_file', lambda *_: native)
    monkeypatch.setattr(v, 'vtk_distance_function', lambda *_: (lambda q: np.zeros(len(q))))
    calls = []
    def generate(_gmsh, _vertices, _faces, level, _protocol, before_generate):
        calls.append(level['id']); before_generate()
        if generation_fails: raise RuntimeError('Deliberate mock generation failure')
        nodes, cells = quadratic_tetrahedron()
        if corrupt_geometry: nodes[4, 0] += .001
        return nodes, cells, {'scope': 'MOCK ONLY'}
    monkeypatch.setattr(v, 'generate_gmsh_level', generate)
    monkeypatch.setattr(v.subprocess, 'run', lambda *_a, **_k: pytest.fail('No subprocess allowed'))
    return output, calls


def test_mock_worker_calls_exact_three_levels_without_solver(tmp_path, monkeypatch):
    output, calls = mocked_worker(tmp_path, monkeypatch)
    assert v.worker('mock-release', output) == 0
    assert calls == ['coarse', 'medium', 'fine']
    result = json.loads((output / 'worker.json').read_text())
    assert result['native_generation_calls'] == 3 and result['solver_calls'] == 0
    assert all(row['status'] == 'passed' for row in result['levels'])


def test_native_failure_retains_unexecuted_levels_and_never_retries(tmp_path, monkeypatch):
    output, calls = mocked_worker(tmp_path, monkeypatch, generation_fails=True)
    assert v.worker('mock-release', output) == 1 and calls == ['coarse']
    result = json.loads((output / 'worker.json').read_text())
    assert [row['status'] for row in result['levels']] == ['failed', 'not_executed', 'not_executed']
    original = (output / 'worker.json').read_bytes()
    with pytest.raises(FileExistsError): v.worker('mock-release', output)
    assert (output / 'worker.json').read_bytes() == original


def test_geometry_failure_preserves_bounded_raw_native_outputs(tmp_path, monkeypatch):
    output, calls = mocked_worker(tmp_path, monkeypatch, corrupt_geometry=True)
    assert v.worker('mock-release', output) == 1 and calls == ['coarse']
    assert (output / 'coarse.npz').is_file() and (output / 'coarse.msh').is_file()
    assert not (output / 'medium.npz').exists()


def test_fidelity_failure_preserves_completed_diagnostics(tmp_path, monkeypatch):
    output, calls = mocked_worker(tmp_path, monkeypatch)
    monkeypatch.setattr(v, 'vtk_distance_function', lambda *_: (lambda q: np.full(len(q), .003)))
    assert v.worker('mock-release', output) == 1 and calls == ['coarse']
    result = json.loads((output / 'worker.json').read_text())
    first = result['levels'][0]
    assert first['status'] == 'failed' and first['quality']['elements'] == 1
    assert first['source_to_mesh']['maximum_sample_distance_m'] == .003
    assert first['mesh_to_source']['full_surface_upper_bound_m'] > .002
    assert first['relative_volume_error'] == pytest.approx(0, abs=1e-12)
    assert result['levels'][1]['status'] == 'not_executed'


def test_launch_preflight_failure_always_has_receipt(tmp_path, monkeypatch):
    def fail(*_): raise ValueError('Unaccepted diagnostic')
    monkeypatch.setattr(v, 'prepare_release', fail)
    assert v.launch('mock-release', tmp_path / 'attempt') == 1
    record = json.loads((tmp_path / 'attempt/acceptance.json').read_text())
    assert record['status'] == 'failed_or_incomplete' and not record['worker_started']
    assert 'Unaccepted diagnostic' in record['error']['message']


def test_mocked_gmsh_adapter_uses_discrete_geometry_and_true_tet10_permutation():
    X, F = tetrahedron(); quadratic, _ = quadratic_tetrahedron()
    local_order = np.array([0, 1, 2, 3, 4, 5, 6, 7, 9, 8])
    options = {}; calls = []
    def event(name):
        return lambda *args: calls.append((name, args))
    mesh = SimpleNamespace(addNodes=event('nodes'), addElementsByType=event('triangles'),
        classifySurfaces=event('classify'), createGeometry=event('geometry'), generate=event('generate'),
        getElements=lambda dim: ([11], [[101]], [np.arange(1, 11)]),
        getElementProperties=lambda kind: ('tet10', 3, 2, 10, (quadratic[local_order] / .01).reshape(-1), 4),
        getNodes=lambda: (np.arange(10, 0, -1), quadratic[local_order][::-1].reshape(-1), []))
    geo = SimpleNamespace(addSurfaceLoop=lambda surfaces: 1, addVolume=lambda loops: 1, synchronize=event('synchronize'))
    model = SimpleNamespace(add=event('model'), addDiscreteEntity=lambda dim: 1,
        getEntities=lambda dim: [(2, 1)], mesh=mesh, geo=geo)
    gmsh = SimpleNamespace(clear=event('clear'), model=model,
        option=SimpleNamespace(setNumber=lambda key, value: options.__setitem__(key, value)))
    nodes, cells, record = v.generate_gmsh_level(gmsh, X, F, PROTOCOL['levels'][0], PROTOCOL,
                                                lambda: calls.append(('charge', ())))
    np.testing.assert_array_equal(nodes[cells[0]], quadratic)
    assert record['gmsh_to_febio_permutation'] == local_order.tolist()
    assert options['Mesh.SecondOrderLinear'] == 1 and options['Mesh.ElementOrder'] == 2
    assert options['Mesh.MeshSizeMin'] == options['Mesh.MeshSizeMax'] == .024
    names = [name for name, _ in calls]
    assert names.index('geometry') < names.index('charge') < names.index('generate')
    assert [args for name, args in calls if name == 'generate'] == [(3,)]
