"""Independent preparation controls; analytic arrays and mocks only.

No Gmsh/VTK, retained anatomy, landmarks, solver, or native generation is used.
The maximum packet fixture deliberately tests serialization, not mesh validity.
"""
import copy
import hashlib
import importlib.util
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest


ROOT = Path(__file__).resolve().parents[1]
_spec = importlib.util.spec_from_file_location(
    'independent_curvature_launcher', ROOT/'scripts/mechanics_patient_mesh_curvature_run.py')
RUN = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(RUN)
V = RUN.HELPER
C = RUN.SHARED
D = V.SHARED
CONFIG = V.declaration()


def packet_arrays(n, e):
    # Deterministic numeric byte patterns, expressly not an anatomical mesh.
    nodes = (np.arange(n*3, dtype=np.float64).reshape(n, 3)-n)/1e6
    cells = np.arange(e*10, dtype=np.int64).reshape(e, 10) % max(n, 1)
    origin = {'gmsh_node_ids': (2*np.arange(n)+11).tolist(),
              'gmsh_element_ids': (3*np.arange(e)+17).tolist(),
              'gmsh_to_febio_permutation': [0, 1, 2, 3, 4, 5, 6, 7, 9, 8]}
    return nodes, cells, origin


def test_actual_maximum_complete_packet_has_exact_bytes_ids_and_no_acceptance(tmp_path):
    nodes, cells, origin = packet_arrays(80000, 100000)
    limits = CONFIG['diagnostic_limits']
    result = D.preserve_diagnostic(tmp_path/'packet', nodes, cells, origin,
                                   {'review_fixture': 'serialization_only'}, limits)
    assert result['complete'] and not result['candidate_accepted']
    assert result['reserved_total_bytes'] == 11_368_704
    path = tmp_path/'packet'
    record = json.loads((path/'manifest.json').read_text())
    assert record['counts'] == {'nodes': 80000, 'tet10_elements': 100000}
    assert record['diagnostic_complete'] and not record['candidate_accepted']
    assert record['coordinate_units'] == 'm'
    assert record['gmsh_to_febio_permutation'] == origin['gmsh_to_febio_permutation']
    expected = {'nodes_m.npy': nodes, 'tet10_indices.npy': cells,
                'gmsh_node_ids.npy': np.array(origin['gmsh_node_ids']),
                'gmsh_element_ids.npy': np.array(origin['gmsh_element_ids'])}
    for name, array in expected.items():
        raw = (path/name).read_bytes()
        assert hashlib.sha256(raw).hexdigest() == record['files'][name]['sha256']
        assert len(raw) == record['files'][name]['bytes']
        np.testing.assert_array_equal(np.load(path/name, allow_pickle=False), array)
    assert record['files']['tet10_indices.npy']['bytes'] == 8_000_128
    usage = C.output_usage(path, spec=RUN.SPEC)
    assert usage['bytes'] == result['total_bytes'] < 16*1024**2
    assert usage['files'] == 5 and usage['largest_file_bytes'] < 8*1024**2
    with pytest.raises(ValueError, match='cap exceeded'):
        C.output_usage(path)  # Old v2 limit has not silently grown.


@pytest.mark.parametrize('n,e', [(80001, 1), (10, 100001)])
def test_over_diagnostic_count_keeps_original_count_and_omits_whole_packet(tmp_path, n, e):
    nodes, cells, origin = packet_arrays(n, e)
    result = D.preserve_diagnostic(tmp_path/'packet', nodes, cells, origin,
                                   {}, CONFIG['diagnostic_limits'])
    assert result['counts'] == {'nodes': n, 'tet10_elements': e}
    assert result['reason'] == 'diagnostic_count_cap' and not result['complete']
    assert not list(tmp_path.iterdir())


@pytest.mark.parametrize('change', ['source_hash', 'source_frame', 'evidence', 'extra_key'])
def test_exact_source_and_evidence_config_changes_block_before_output_or_generation(tmp_path, change):
    value = copy.deepcopy(CONFIG)
    if change == 'source_hash': value['source_surface_binding']['sha256'] = '0'*64
    elif change == 'source_frame': value['source_frame'] = 'different frame'
    elif change == 'evidence': value['evidence_bindings'].pop()
    else: value['unreviewed_option'] = True
    with pytest.raises(ValueError, match='exact declared'):
        V.assess_candidate(None, None, None, tmp_path/'attempt',
                           distance_factory=None, config=value)
    assert not list(tmp_path.iterdir())


def test_profile_changes_are_local_and_defaults_unchanged_in_both_orders():
    def fake():
        options = {}; fields = {}; ids = iter([7, 8])
        api = SimpleNamespace(option=SimpleNamespace(setNumber=options.__setitem__),
            model=SimpleNamespace(mesh=SimpleNamespace(field=SimpleNamespace(
                add=lambda _: next(ids), setNumbers=lambda k, n, v: fields.__setitem__((k, n), v),
                setNumber=lambda k, n, v: fields.__setitem__((k, n), v),
                setAsBackgroundMesh=lambda _: None))))
        return api, options, fields
    default_spec = C.execution_spec()
    for callback, floor in [(V.configure_profile, .003), (D.configure_profile, .012),
                            (V.configure_profile, .003), (D.configure_profile, .012)]:
        api, options, fields = fake()
        callback(api, [3, 11], CONFIG['size_profile'])
        assert options == {'Mesh.MeshSizeMin': floor, 'Mesh.MeshSizeMax': .024}
        assert fields[(7, 'SurfacesList')] == [3, 11] and fields[(7, 'Sampling')] == 100
        assert fields[(8, 'SizeMin')] == .012 and fields[(8, 'SizeMax')] == .024
        assert fields[(8, 'DistMin')] == .002 and fields[(8, 'DistMax')] == .024
        assert C.execution_spec() == default_spec
    assert C.execution_spec().file_bytes == 4*1024**2
    assert C.execution_spec().output_bytes == 8*1024**2


@pytest.mark.parametrize('edge', ['bytes', 'files'])
def test_v3_final_parent_receipt_is_included_in_exact_output_limits(tmp_path, monkeypatch, edge):
    bound_path = tmp_path/'bound'; bound_path.write_text('analytic-token')
    bound = {str(bound_path): C.sha(bound_path)}
    monkeypatch.setattr(C, 'released', lambda *a, **k: (CONFIG, None, None, bound))
    env = {key: '1' for key in ['OMP_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS',
                               'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS']}
    env['OMP_DYNAMIC'] = 'FALSE'
    monkeypatch.setattr(C, 'module', lambda *a: SimpleNamespace(
        declaration=lambda: {}, private_environment=lambda _: dict(env)))

    def supervised(_runtime, _command, output, _environment, _caps, *, spec):
        assert spec is RUN.SPEC
        marker = output/'candidate/diagnostic/manifest.json'
        marker.parent.mkdir(parents=True); marker.write_text('{}')
        record = {'status': 'completed_geometry_only',
                  'candidate_status': 'geometry_candidate_passed_no_solver_authorization',
                  'native_generation_calls': 1, 'solver_calls': 0,
                  'candidate_output_sha256': {'candidate/diagnostic/manifest.json': C.sha(marker)}}
        (output/'worker.json').write_text(json.dumps(record))
        if edge == 'bytes':
            remaining = spec.output_bytes-C.output_usage(output, spec=spec)['bytes']-16
            for i in range(4):
                count = min(spec.file_bytes, remaining)
                with (output/f'padding-{i}').open('wb') as stream: stream.truncate(count)
                remaining -= count
            assert remaining == 0
        else:
            for i in range(30): (output/f'padding-{i}').write_bytes(b'')
        C.output_usage(output, spec=spec)
        return {'status': 'completed'}, {'error': None}

    monkeypatch.setattr(C, 'supervised', supervised)
    assert C.launch('mocked-release', tmp_path/'attempt', spec=RUN.SPEC) == 1
    result = json.loads((tmp_path/'attempt/acceptance.json').read_text())
    assert result['status'] == 'failed_or_incomplete'
    assert 'cap exceeded' in result['output_error']


@pytest.mark.parametrize('solver_admitted', [False, True])
def test_v3_worker_forwards_eight_mib_limit_and_rejects_solver_claim(tmp_path, monkeypatch, solver_admitted):
    output = tmp_path/'attempt'; output.mkdir()
    source = tmp_path/'never-decoded-surface'
    bound_path = tmp_path/'bound'; bound_path.write_text('analytic-token')
    events = []
    first = {'package_versions': {}, 'gmsh_runtime': {'module_path': '/mock/gmsh.py',
             'library_path': '/mock/libgmsh', 'version': '4.15.2'}}
    monkeypatch.setattr(C, 'released', lambda *a, **k: (
        CONFIG, first, source, {str(bound_path): C.sha(bound_path)}))
    def hardlimit(kind, values):
        assert kind == C.resource.RLIMIT_FSIZE and values == (8*1024**2, 8*1024**2)
        events.append('limit')
    monkeypatch.setattr(C.resource, 'setrlimit', hardlimit)
    class Archive:
        files = ['vertices_m', 'triangles']
        def __enter__(self): return self
        def __exit__(self, *a): events.append('close')
        def __getitem__(self, key): return 'analytic-'+key
    def load(path, **kwargs):
        assert path == source and kwargs == {'allow_pickle': False} and events == ['limit']
        events.append('mock-load'); return Archive()
    monkeypatch.setattr(np, 'load', load)
    def assessed(_api, vertices, faces, directory, **kwargs):
        assert vertices == 'analytic-vertices_m' and faces == 'analytic-triangles'
        assert 'close' in events
        events.append('assess'); directory.mkdir()
        return {'status': 'geometry_candidate_passed_no_solver_authorization',
                'native_generation_calls': 1, 'solver_calls': 0, 'solver_admitted': solver_admitted}
    helper = SimpleNamespace(assess_candidate=assessed,
                             BASE=SimpleNamespace(vtk_distance_function=None))
    api = SimpleNamespace(__file__='/mock/gmsh.py', lib=SimpleNamespace(_name='/mock/libgmsh'),
        __version__='4.15.2', initialize=lambda *a, **k: events.append('mock-init'),
        finalize=lambda: events.append('mock-finalize'))
    def module(path, name):
        if name == 'released_graded_mesh':
            assert path == C.ROOT/RUN.SPEC.helper_path
            return helper
        return api
    monkeypatch.setattr(C, 'module', module)
    monkeypatch.delitem(sys.modules, 'gmsh', raising=False)
    monkeypatch.setitem(sys.modules, 'vtkmodules.vtkCommonCore', SimpleNamespace(
        vtkSMPTools=SimpleNamespace(SetBackend=lambda _: True, GetBackend=lambda: 'Sequential')))
    assert C.worker('mock-release', output, spec=RUN.SPEC) == int(solver_admitted)
    result = json.loads((output/'worker.json').read_text())
    assert events.count('assess') == 1 and events.count('mock-finalize') == 1
    assert result['per_file_hard_limit_bytes'] == 8*1024**2
    assert result['B_or_V_access'] is False and result['MRI_reextracted'] is False
    assert result['solver_calls'] == 0
    assert (result['status'] == 'completed_geometry_only') is not solver_admitted


def test_ten_file_closure_missing_new_helper_fails_before_any_input_hash(tmp_path, monkeypatch):
    archive = tmp_path/'archive'; archive.mkdir()
    repository = tmp_path/'repository'; repository.mkdir()
    monkeypatch.setattr(C, 'ROOT', archive)
    release = {'schema': RUN.SPEC.version, 'authorized': True,
        'source_directory': str(archive), 'attempt_directory': str(tmp_path/'attempt'),
        'repository_directory': str(repository), 'source_commit': '1'*40,
        'clinical_validation': False, 'anatomical_registration_accepted': False,
        'solver_authorized': False, 'reuse_saved_native_surface_only': True,
        'source_sha256': {name: '0'*64 for name in RUN.SPEC.closure
                          if name != 'scripts/mechanics_patient_mesh_curvature.py'}}
    path = tmp_path/'release.json'; path.write_text(json.dumps(release))
    monkeypatch.setattr(C, 'sha', lambda *_: pytest.fail('No anatomy/source hashing permitted'))
    with pytest.raises(ValueError, match='execution closure'):
        C.released(path, tmp_path/'attempt', spec=RUN.SPEC)

