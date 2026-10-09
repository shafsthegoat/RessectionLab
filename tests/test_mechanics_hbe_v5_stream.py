"""Generated-fixture stream controls; no native FEBio or HBE response access."""
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import subprocess

import numpy as np
import pytest

from scripts import mechanics_hbe_branch_calibration_v5 as v5
from scripts import mechanics_hbe_v5_frame as frame
from scripts import mechanics_hbe_v5_stream as stream
from scripts import mechanics_hbe_v5_source_bindings as sources
from scripts.mechanics_hbe_physics import HexMesh
from test_mechanics_hbe_branch_calibration_v5 import fixture_source
from test_mechanics_hbe_halfheight_readout import geometry

ROOT = Path(__file__).resolve().parents[1]
STUDY_BYTES = (ROOT / v5.DECLARATION_PATH).read_bytes()
STUDY = json.loads(STUDY_BYTES)
PRIOR_BYTES = (ROOT / STUDY['previous_v4']['path']).read_bytes()


@pytest.fixture(autouse=True)
def no_native_execution(monkeypatch):
    monkeypatch.setattr(subprocess, 'Popen', lambda *a, **k: pytest.fail('No native execution'))


def generated_run(branch='tension', *, half=False, steps=60):
    N = (36 if branch == 'compression' and steps == 120 else 24) if half else 8
    domain = 'lower_half_reconstructed' if half else 'full_native'
    run_id = f'{branch}:N{N}:S{steps}:reference'
    old = fixture_source(branch, domain)
    _, adapted, receipt = v5.adapt_deck(STUDY, json.loads(PRIOR_BYTES), run_id, old)
    contract = frame.verified_schedule(STUDY_BYTES, PRIOR_BYTES, run_id, old, adapted, receipt)
    full, native_half, mapping = geometry()
    mesh = native_half if half else full
    X = np.asarray(mesh['rest_nodes_m'])
    H = full['geometry']['height_m']
    d = np.asarray(contract['full_coordinates_m'])
    full_model = HexMesh.from_manifest(full)
    full_X = full_model.rest_nodes_m
    energies = []
    for value in d:
        current = full_X.copy()
        current[:, 2] += full_X[:, 2] * value / H
        energies.append(full_model.deformation(current, 1000.)['energy_J'])
    force = np.gradient(energies, d)
    force[0] = 0.
    node_lines, element_lines = [], []
    top = np.asarray(mesh['boundaries']['top']['node_ids']) - 1
    bottom = np.asarray(mesh['boundaries']['bottom']['node_ids']) - 1
    for i, (time, value, applied) in enumerate(zip(contract['times'], d, force)):
        current = X.copy()
        current[:, 2] += X[:, 2] * value / H
        reactions = np.zeros_like(X)
        reactions[top, 2] = -applied / len(top)
        reactions[bottom, 2] = applied / len(bottom)
        values = np.column_stack((current, current - X, reactions))
        logged = np.tile([10., 11., 12., 13., 14., 15., 1 + value/H, 16.],
                         (len(mesh['element_ids']), 1))
        node_lines += [f'*Step = {i}\n', f'*Time = {time:.9g}\n', '*Data = mechanics_nodes_si\n']
        element_lines += [f'*Step = {i}\n', f'*Time = {time:.9g}\n', '*Data = mechanics_elements_si\n']
        node_lines += [f'{j},' + ','.join(repr(float(v)) for v in row) + '\n'
                       for j, row in enumerate(values, 1)]
        element_lines += [f'{j},' + ','.join(repr(float(v)) for v in row) + '\n'
                          for j, row in enumerate(logged, 1)]
    solver = []
    for time in contract['times'][1:]:
        solver += [f'Nonlinear solution status: time= {time:.6g}\n',
                   ' residual 1e-6 1e-16 1e-14\n']
    solver += ['N O R M A L T E R M I N A T I O N\n']
    reconstruction = (full, mapping) if half else None
    return contract, mesh, node_lines, element_lines, solver, reconstruction


@pytest.mark.parametrize('branch,half,steps', [('tension', False, 60),
                                               ('compression', True, 120)])
def test_complete_generated_stream_and_work(branch, half, steps):
    spec, mesh, nodes, elements, solver, reconstruction = generated_run(
        branch, half=half, steps=steps)
    result = stream.evaluate_stream(spec, mesh, nodes, elements, solver,
                                    reconstruction=reconstruction)
    assert result['numerical_passed']
    assert result['frame_count'] == steps + 1
    assert len(result['solver']['states']) == steps
    assert len(result['full_energy_work']['work_J']) == steps + 1
    assert len(result['probe_displacements_m']) == steps + 1
    assert result['native_energy_work'] is not None if half else result['native_energy_work'] is None
    assert result['provenance']['generated_fixture_only']
    assert result['provenance']['physical_validation_pass'] is None


@pytest.mark.parametrize('mutation', ['missing_node_frame', 'missing_element_frame',
                                      'wrong_node_time', 'wrong_element_time',
                                      'duplicate_node_id', 'missing_residual',
                                      'solver_cutback', 'wrong_force_sign',
                                      'force_energy_mismatch', 'solver_bad_residual'])
def test_stream_defects_fail_closed(mutation):
    spec, mesh, nodes, elements, solver, reconstruction = generated_run()
    nodes, elements, solver = deepcopy((nodes, elements, solver))
    if mutation == 'missing_node_frame':
        nodes = nodes[:-(3 + len(mesh['node_ids']))]
    elif mutation == 'missing_element_frame':
        elements = elements[:-(3 + len(mesh['element_ids']))]
    elif mutation == 'wrong_node_time':
        nodes[3 + len(mesh['node_ids']) + 1] = '*Time = 0.12345\n'
    elif mutation == 'wrong_element_time':
        elements[3 + len(mesh['element_ids']) + 1] = '*Time = 0.12345\n'
    elif mutation == 'duplicate_node_id':
        nodes[4] = nodes[3].replace('1,', '2,', 1)
    elif mutation == 'missing_residual':
        solver.pop(3)
    elif mutation == 'solver_cutback':
        solver[2] = solver[0]
    elif mutation == 'solver_bad_residual':
        solver[1] = ' residual 1e-6 1e-8 1e-14\n'
    elif mutation == 'wrong_force_sign':
        # Invert the top and bottom reactions at the last load state.
        start = 60 * (3 + len(mesh['node_ids'])) + 3
        for i in range(start, start + len(mesh['node_ids'])):
            fields = nodes[i].strip().split(',')
            fields[-1] = repr(-float(fields[-1]))
            nodes[i] = ','.join(fields) + '\n'
    else:
        for i, line in enumerate(nodes):
            if not line[0].isdigit():
                continue
            fields = line.strip().split(',')
            fields[-1] = repr(.5 * float(fields[-1]))
            nodes[i] = ','.join(fields) + '\n'
    if mutation in {'wrong_force_sign', 'force_energy_mismatch', 'solver_bad_residual'}:
        result = stream.evaluate_stream(spec, mesh, nodes, elements, solver)
        assert not result['numerical_passed']
        if mutation == 'force_energy_mismatch':
            assert result['criteria_max_ratio']['full_work_energy'] > 1
        if mutation == 'solver_bad_residual':
            assert result['criteria_max_ratio']['solver_residual'] > 1
    else:
        with pytest.raises(ValueError):
            stream.evaluate_stream(spec, mesh, nodes, elements, solver)


def test_exact_six_binding_inventory_required(tmp_path):
    with pytest.raises(ValueError, match='Exactly six'):
        stream.read_bound_run(tmp_path, 'tension:N8:S60:reference', {}, {})


def test_six_hash_bound_saved_files_keep_output_origin_unknown(tmp_path, monkeypatch):
    spec, mesh, nodes, elements, solver, _ = generated_run()
    source = fixture_source('tension', 'full_native')
    _, adapted, receipt = v5.adapt_deck(STUDY, json.loads(PRIOR_BYTES), spec['run_id'], source)
    study_path = tmp_path / v5.DECLARATION_PATH
    prior_path = tmp_path / STUDY['previous_v4']['path']
    study_path.parent.mkdir(parents=True)
    prior_path.parent.mkdir(parents=True, exist_ok=True)
    study_path.write_bytes(STUDY_BYTES)
    prior_path.write_bytes(PRIOR_BYTES)
    payloads = {'source_deck': source.encode(), 'adapted_deck': adapted.encode(),
                'mesh': (json.dumps(mesh) + '\n').encode(),
                'nodes': ''.join(nodes).encode(), 'elements': ''.join(elements).encode(),
                'solver': ''.join(solver).encode()}
    bindings = {}
    for name, payload in payloads.items():
        relative = (f'{name}.txt' if name in {'source_deck', 'mesh'}
                    else f'outputs/mechanics/generated-fixture/{name}.txt')
        target = tmp_path / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(payload)
        bindings[name] = {'path': relative,
                          'sha256': hashlib.sha256(payload).hexdigest()}
    manifest = {'run_source_keys': {spec['run_id']: 'tension:N8'},
                'source_decks': {'tension:N8': {**bindings['source_deck'], 'mesh': 'full:N8'}},
                'meshes': {'full:N8': bindings['mesh']}}
    manifest_raw = json.dumps(manifest).encode()
    (tmp_path / 'source-bindings.json').write_bytes(manifest_raw)
    monkeypatch.setattr(stream.sources, 'BINDING_PATH', 'source-bindings.json')
    monkeypatch.setattr(stream.sources, 'BINDING_SHA256', hashlib.sha256(manifest_raw).hexdigest())
    monkeypatch.setattr(stream.sources, 'validate_binding_manifest', lambda *a, **k: None)
    monkeypatch.setattr(stream.v5, 'validate_preparation', lambda *a: (STUDY, json.loads(PRIOR_BYTES)))
    result = stream.read_bound_run(tmp_path, spec['run_id'], bindings, receipt)
    assert result['numerical_passed']
    assert result['provenance']['source_binding_checked']
    assert result['provenance']['output_origin'] == 'unverified_saved_stream'
    assert result['provenance']['native_output_observed'] is None
    (tmp_path / bindings['nodes']['path']).write_bytes(payloads['nodes'] + b'changed')
    with pytest.raises(ValueError, match='hash changed|SHA256'):
        stream.read_bound_run(tmp_path, spec['run_id'], bindings, receipt)


def test_candidate_output_cannot_point_into_measured_data_or_symlink(tmp_path, monkeypatch):
    measured = tmp_path / 'data/mechanics/observed.csv'
    measured.parent.mkdir(parents=True)
    measured.write_text('not for this checker')
    with pytest.raises(ValueError, match='outputs/mechanics'):
        stream._candidate_output_path(tmp_path, {'path': 'data/mechanics/observed.csv'})
    safe = tmp_path / 'outputs/mechanics/candidate'
    safe.parent.mkdir(parents=True)
    safe.symlink_to(measured)
    with pytest.raises(ValueError, match='Symlinked'):
        stream._candidate_output_path(tmp_path, {'path': 'outputs/mechanics/candidate'})
    monkeypatch.chdir(tmp_path)
    with pytest.raises(ValueError, match='Symlinked'):
        stream._candidate_output_path('.', {'path': 'outputs/mechanics/candidate'})


@pytest.mark.parametrize('n', (16, 24, 32, 36))
def test_actual_hash_bound_half_reconstruction_representation_at_rest(n):
    """Existing saved anatomy-free specimen manifests, never native responses."""
    prior = json.loads(PRIOR_BYTES)
    old = prior['meshes'][str(n)]
    if not all((ROOT / old[key]['path']).is_file() for key in
               ('full_mesh', 'half_mesh', 'reconstruction')):
        pytest.skip('Local ignored historical mesh manifests unavailable')
    half = json.loads(sources._read_bound(ROOT, old['half_mesh'], maximum=sources.MAX_MESH_BYTES))
    full, mapping = stream._bound_reconstruction(ROOT, prior, {'N': n}, old['half_mesh'], half)
    assert mapping['schema'] == 'hbe-halfheight-mapping-v1'
    assert len(full['rest_nodes_m']) > len(half['rest_nodes_m'])
    assert len(full['elements_hex8']) == 2 * len(half['elements_hex8'])
