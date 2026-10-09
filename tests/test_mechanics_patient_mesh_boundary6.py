"""Source-only candidate controls; generated fixtures, no patient/native payload."""
from __future__ import annotations

import copy
import importlib.util
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location(
    'case4_boundary6_preparation', ROOT/'scripts/mechanics_patient_mesh_boundary6_run.py')
RUN = importlib.util.module_from_spec(spec)
spec.loader.exec_module(RUN)
V = RUN.HELPER
SHARED = V.SHARED
CALLER = RUN.SHARED
CONFIG = V.declaration()


def tetra_nodes():
    corners = np.array([[0., 0, 0], [.01, 0, 0], [0, .01, 0], [0, 0, .01]])
    return np.vstack([corners, corners[V.BASE.EDGES].mean(axis=1)])


def fake_gmsh():
    events = []; options = {}; fields = {}; ids = iter([31, 32]); nodes = tetra_nodes()
    def option(name, value):
        options[name] = value; events.append(('option', name, value))
    mesh = SimpleNamespace(
        field=SimpleNamespace(add=lambda _: next(ids),
            setNumbers=lambda tag, name, value: fields.__setitem__((tag, name), value),
            setNumber=lambda tag, name, value: fields.__setitem__((tag, name), value),
            setAsBackgroundMesh=lambda tag: events.append(('background', tag))),
        addNodes=lambda *a: events.append(('nodes',)),
        addElementsByType=lambda *a: events.append(('triangles',)),
        classifySurfaces=lambda *a: events.append(('classify', a, dict(options))),
        createGeometry=lambda: events.append(('parametrize',)),
        generate=lambda dim: events.append(('generate', dim, dict(options))),
        getElements=lambda dim: ([11], [np.array([91])], [np.arange(1, 11)]),
        getElementProperties=lambda kind: ('tet10', 3, 2, 10, (nodes/.01).ravel(), 4),
        getNodes=lambda: (np.arange(1, 11), nodes.ravel(), []))
    api = SimpleNamespace(option=SimpleNamespace(setNumber=option),
        clear=lambda: events.append(('clear',)), model=SimpleNamespace(
            add=lambda name: events.append(('model', name)), mesh=mesh,
            addDiscreteEntity=lambda dim: 7, getEntities=lambda dim: [(2, 7)],
            geo=SimpleNamespace(addSurfaceLoop=lambda tags: 8,
                                addVolume=lambda loops: 9,
                                synchronize=lambda: events.append(('synchronize',)))))
    return api, events, fields


def test_one_fixed_geometry_delta_and_no_solver_release():
    old = SHARED.declaration()
    for key in ['source_case', 'source_surface_binding', 'source_frame',
                'quality', 'surface_fidelity', 'gmsh_options']:
        assert CONFIG[key] == old[key]
    assert CONFIG['size_profile']['boundary_size_m'] == .006
    assert CONFIG['size_profile']['interior_size_m'] == .024
    assert CONFIG['size_profile']['distance_min_m'] == .002
    assert CONFIG['size_profile']['distance_max_m'] == .024
    assert CONFIG['gmsh_options']['Mesh.MeshSizeFromCurvature'] == 0
    assert CONFIG['gmsh_options']['Mesh.SecondOrderLinear'] == 1
    assert CONFIG['caps']['maximum_generations'] == 1
    assert CONFIG['caps']['retries'] == 0
    assert CONFIG['caps']['aggregate_seconds'] == 180
    assert CONFIG['caps']['process_group_rss_bytes'] == 3*1024**3
    assert CONFIG['caps']['numerical_threads'] == 1
    assert not CONFIG['execution_release']['authorized']
    assert not CONFIG['solver_admission']['authorized']
    assert len(RUN.SPEC.closure) == 10
    assert RUN.SPEC.file_bytes == 8*1024**2
    assert RUN.SPEC.output_bytes == 32*1024**2
    assert RUN.SPEC.output_files == 32
    assert CONFIG['output_limits'] == {'maximum_total_bytes': RUN.SPEC.output_bytes,
        'maximum_file_bytes': RUN.SPEC.file_bytes, 'maximum_files': RUN.SPEC.output_files}


def test_exact_source_closure_and_prior_failures_are_bound():
    assert RUN.SPEC.closure == CALLER.CLOSURE | {
        'scripts/mechanics_patient_mesh_boundary6.py',
        'scripts/mechanics_patient_mesh_boundary6_run.py',
        'manifests/experiments/resect-case4-patient-mesh-boundary6-v4.json'}
    assert V.sha(ROOT/'scripts/mechanics_patient_mesh_candidate.py') == V.CANDIDATE_SHA
    assert V.sha(ROOT/'manifests/experiments/resect-case4-patient-mesh-graded-v2.json') == V.PREVIOUS_SHA
    assert {entry['path'] for entry in CONFIG['evidence_bindings']} == {
        'artifacts/mechanics/resect-case4-patient-mesh-graded-v2/summary.json',
        'artifacts/mechanics/resect-case4-patient-mesh-curvature-v3/summary.json',
        'artifacts/mechanics/resect-case4-saved-mesh-diagnostic-v1/summary.json'}
    for entry in CONFIG['evidence_bindings']:
        assert V.sha(ROOT/entry['path']) == entry['sha256']


@pytest.mark.parametrize('group, key, value', [
    ('source_surface_binding', 'sha256', '0'*64),
    ('size_profile', 'boundary_size_m', .012),
    ('gmsh_options', 'Mesh.MeshSizeFromCurvature', 24),
    ('gmsh_options', 'Mesh.SecondOrderLinear', 0),
    ('caps', 'maximum_generations', 2),
    ('caps', 'retries', 1),
    ('caps', 'aggregate_seconds', 181),
    ('surface_fidelity', 'maximum_distance_m', .003),
    ('quality', 'minimum_mean_ratio', .01),
    ('eligibility', 'maximum_nodes', 64001),
    ('diagnostic_limits', 'maximum_total_bytes', 32*1024**2),
    ('solver_admission', 'authorized', True),
    ('execution_release', 'authorized', True),
])
def test_unreviewed_change_fails_before_output_or_native_call(tmp_path, group, key, value):
    altered = copy.deepcopy(CONFIG)
    altered[group][key] = value
    with pytest.raises(ValueError, match='exact declared'):
        V.assess_candidate(None, None, None, tmp_path/'attempt',
                           distance_factory=None, config=altered)
    assert not (tmp_path/'attempt').exists()


def test_one_curvature_off_generation_uses_fixed_distance_field():
    api, events, fields = fake_gmsh()
    X, E, origin = V.generate_one(api, tetra_nodes()[:4], V.BASE.FACES,
                                  CONFIG, lambda: events.append(('charge',)))
    generated = next(event for event in events if event[0] == 'generate')
    classified = next(event for event in events if event[0] == 'classify')
    assert classified[1] == (np.pi, True, True, np.pi)
    assert generated[2]['Mesh.MeshSizeFromCurvature'] == 0
    assert generated[2]['Mesh.SecondOrderLinear'] == 1
    assert generated[2]['Mesh.MeshSizeMin'] == .006
    assert generated[2]['Mesh.MeshSizeMax'] == .024
    assert fields[(31, 'Sampling')] == 100
    assert fields[(32, 'SizeMin')] == .006
    assert fields[(32, 'SizeMax')] == .024
    assert fields[(32, 'DistMin')] == .002
    assert fields[(32, 'DistMax')] == .024
    assert events[events.index(generated)-1] == ('charge',)
    assert sum(event[0] == 'generate' for event in events) == 1
    np.testing.assert_array_equal(X, tetra_nodes())
    np.testing.assert_array_equal(E, [range(10)])
    assert origin['gmsh_element_ids'] == [91]


@pytest.mark.parametrize('distance, passes', [(0., True), (.003, False)])
def test_geometry_gates_still_run_and_no_solver_admission(tmp_path, distance, passes):
    api, _, _ = fake_gmsh()
    result = V.assess_candidate(api, tetra_nodes()[:4], V.BASE.FACES,
        tmp_path/'attempt', distance_factory=lambda *_: lambda q: np.full(len(q), distance))
    assert (result['status'] == 'geometry_candidate_passed_no_solver_authorization') is passes
    assert result['native_generation_calls'] == 1
    assert result['solver_calls'] == 0
    assert not result['solver_admitted']
    assert result['diagnostic']['complete']
    assert result['source_to_mesh']['maximum_sample_distance_m'] == distance


def test_complete_packet_upper_bound_and_output_limits_without_patient_arrays(tmp_path):
    limits = CONFIG['diagnostic_limits']
    n = limits['maximum_nodes']; e = limits['maximum_elements']
    arrays = [np.broadcast_to(np.float64(0), (n, 3)),
              np.broadcast_to(np.int64(0), (e, 10)),
              np.broadcast_to(np.int64(0), (n,)),
              np.broadcast_to(np.int64(0), (e,))]
    sizes = [len(SHARED.array_header(a)) + a.nbytes for a in arrays]
    assert sum(sizes) + limits['maximum_metadata_bytes'] == 11_368_704
    assert sum(sizes) + limits['maximum_metadata_bytes'] < 16*1024**2
    assert max(sizes) == 8_000_128 < RUN.SPEC.file_bytes
    with (tmp_path/'oversize').open('wb') as stream:
        stream.truncate(RUN.SPEC.file_bytes + 1)
    with pytest.raises(ValueError, match='cap exceeded'):
        CALLER.output_usage(tmp_path, spec=RUN.SPEC)


def test_separate_release_required_before_source_hash_or_patient_read(tmp_path, monkeypatch):
    release = tmp_path/'release.json'
    release.write_text('{"schema":"resect-case4-patient-mesh-boundary6-v4-release","authorized":false}')
    monkeypatch.setattr(CALLER, 'sha', lambda *_: pytest.fail('No source hash before release'))
    with pytest.raises(ValueError, match='Separate'):
        CALLER.released(release, tmp_path/'attempt', spec=RUN.SPEC)
