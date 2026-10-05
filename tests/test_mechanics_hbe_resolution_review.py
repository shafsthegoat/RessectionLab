"""Independent metadata and tiny analytical controls; no native computation.

The preparation fixture isolates the new prepared-output closure. Historical
runtime/source authentication is stubbed explicitly, not claimed as revalidated.
"""
from copy import deepcopy
import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

from scripts import mechanics_hbe_outputs as outputs
from scripts import mechanics_hbe_resolution as resolution

ROOT = Path(__file__).resolve().parents[1]


def save(root, name, value):
    path = root/name
    path.parent.mkdir(parents=True, exist_ok=True)
    data = value if isinstance(value, bytes) else (json.dumps(value, sort_keys=True)+'\n').encode()
    path.write_bytes(data)
    return {'path': name, 'sha256': hashlib.sha256(data).hexdigest()}


def preparation_fixture(root, monkeypatch):
    study = json.loads((ROOT/resolution.STUDY_PATH).read_text())
    study['original_protocol'] = save(root, 'protocol.json', {'mesher': {'mesh_quality': {
        'minimum_rest_scaled_jacobian_exclusive': .1, 'finest_relative_volume_error_max': .005,
        'finest_max_radial_sag_over_R': .005}}})
    study['prior_failure'] = {'state': save(root, 'old-state.json', {'runs': {}})}
    study['diagnosis'] = save(root, 'diagnosis.json', {'fixture': True})
    study['prior_runs'] = {}  # Historical bindings are outside this isolated gate.
    study_binding = save(root, 'study.json', study)
    monkeypatch.setattr(resolution, 'STUDY_SHA', study_binding['sha256'])
    source = save(root, 'source.py', b'# explicit mock mesher source\n')
    profile = save(root, 'profile.json', {'fixture': True})
    monkeypatch.setattr(resolution, 'PROFILE_SHA', profile['sha256'])
    runtime = save(root, 'runtime.json', {'fixture': True})
    interpreter = save(root, 'interpreter', b'never executed')
    monkeypatch.setattr(resolution.sys, 'executable', str(root/interpreter['path']))
    monkeypatch.setattr(resolution, '_sources', lambda *args: None)
    monkeypatch.setattr(resolution.backend, 'verify_profile', lambda *args: {
        'inputs': {}, 'runtime_identity': runtime, 'runtime': {'executable': str(root/'never-solve')}})
    gmsh_files = {name: save(root, f'gmsh/{name}', b'never imported') for name in ('module', 'library', 'license')}
    gmsh = {name: {'path': str(root/value['path']), 'sha256': value['sha256']} for name, value in gmsh_files.items()}
    gmsh_binding = save(root, 'gmsh.json', gmsh)
    monkeypatch.setattr(resolution.mesh, 'verify_runtime', lambda *args: deepcopy(gmsh))
    archive = save(root, 'source.tar', b'source closure independently tested elsewhere')
    sources = {'mesh_deck': source}
    directory = Path(study['output_root'])/'mesh-preparation'
    state = {'status': 'prepared_not_solved', 'gmsh_generation_calls': 2,
             'mesh_preparation_invocations': 2, 'solver_invocations': 0,
             'measured_data_accessed': False, 'levels': {}, 'cases': {}}
    for level in study['mesh_levels']:
        N = level['N']; base = directory/f'N{N}'/'generated'
        mesh = save(root, str(base/'mesh.json'), {
            'mesh_N': N, 'rest_nodes_m': [None]*level['expected_nodes'],
            'elements_hex8': [None]*level['nominal_hex8_cells']})
        native = save(root, str(base/'specimen.msh'), b'not a native mesh')
        receipt = {'status': 'prepared_not_solved', 'mesh_N': N, 'gmsh_generation_calls': 1,
                   'solver_calls': 0, 'curve_values_opened': False, 'runtime_unchanged': True,
                   'source_sha256': source['sha256'], 'protocol_sha256': study['original_protocol']['sha256'],
                   'resolution_declaration_sha256': study_binding['sha256'],
                   'runtime_receipt_sha256': gmsh_binding['sha256'], 'mesh_sha256': mesh['sha256'],
                   'native_mesh_sha256': native['sha256'], 'elapsed_seconds': .1,
                   'quality': {'passed': True, 'node_count': level['expected_nodes'],
                       'hex8_cell_count': level['nominal_hex8_cells'], 'connected_cell_count': level['nominal_hex8_cells'],
                       'minimum_rest_determinant_m3': 1e-12, 'minimum_rest_scaled_jacobian': .5,
                       'relative_volume_error': .001, 'maximum_radial_boundary_sag_over_R': .001,
                       'checks': {'declared_node_count': True, 'minimum_scaled_jacobian': True,
                       'declared_cell_count': True, 'positive_rest_jacobians': True,
                       'finest_volume': True, 'finest_boundary_sag': True}}, 'decks': {}}
        for branch in study['branches']:
            key = f'{branch}:N{N}:S60:reference'; name = f'{branch}-60-reference'
            original = save(root, str(base/name/'specimen.feb'),
                b'<febio_spec><Control><solver><linear_solver type="skyline" /></solver></Control></febio_spec>')
            loading = {'branch': branch, 'steps': 60, 'mu_Pa': 1000.,
                       'protocol_sha256': study['original_protocol']['sha256'],
                       'mesh_sha256': mesh['sha256'], 'deck_sha256': original['sha256'],
                       'resolution_declaration_sha256': study_binding['sha256']}
            original_loading = save(root, str(base/name/'loading.json'), loading)
            target = directory/'cases'/key.replace(':', '-')
            adapted = save(root, str(target/'specimen.feb'), resolution.backend.transform_deck(
                (root/original['path']).read_bytes()).encode())
            final_loading = save(root, str(target/'loading.json'), dict(loading, deck_sha256=adapted['sha256']))
            receipt['decks'][name] = {'deck_sha256': original['sha256'], 'loading_sha256': original_loading['sha256']}
            state['cases'][key] = {'mesh': mesh, 'deck': adapted, 'loading': final_loading,
                                   'backend_source_deck': original}
        state['levels'][str(N)] = save(root, str(base/'receipt.json'), receipt)
    state_binding = save(root, str(directory/'state.json'), state)
    baseline = save(root, str(directory/'baseline.json'), {'study_binding': study_binding,
        'phase': 'prepare', 'directory': str(root/directory),
        'source_bindings': sources, 'backend_profile': profile, 'inputs': {}})
    prepared = save(root, str(directory/'result.json'), {'phase': 'prepare', 'status': 'prepared_not_solved',
        'supervision': {'status': 'completed', 'wall_cap_seconds': 120, 'rss_cap_bytes': 3*1024**3,
                        'exit_code': 0, 'elapsed_seconds': .2, 'kill_reason': None, 'cleanup_error': None},
        'state': state_binding, 'baseline': baseline})
    release = {'schema': 'hbe-resolution-release-v1', 'authorized': True, 'phase': 'solve',
        'study': study_binding, 'source_bindings': sources, 'source_archive': archive,
        'interpreter': {'path': str(root/interpreter['path']), 'sha256': interpreter['sha256']},
        'backend_profile': profile, 'gmsh_runtime': gmsh_binding, 'preparation': prepared}
    release_binding = save(root, 'release.json', release)
    return study_binding, release_binding, state


def test_bound_preparation_level_receipt_is_checked_before_solve(tmp_path, monkeypatch):
    study, release, state = preparation_fixture(tmp_path, monkeypatch)
    resolution.preflight(tmp_path, study, release, 'solve')
    level = tmp_path/state['levels']['24']['path']
    level.write_bytes(level.read_bytes()+b' ')
    with pytest.raises(ValueError, match='hash|level|receipt'):
        resolution.preflight(tmp_path, study, release, 'solve')


def test_valid_level_receipts_join_ongoing_input_integrity_set(tmp_path, monkeypatch):
    study, release, state = preparation_fixture(tmp_path, monkeypatch)
    plan = resolution.preflight(tmp_path, study, release, 'solve')
    for binding in state['levels'].values():
        assert plan['inputs'][str((tmp_path/binding['path']).resolve())] == binding['sha256']


@pytest.mark.parametrize('fault', ['failed_level', 'wrong_source', 'relocated_level', 'volume_limit', 'wrong_phase_cap'])
def test_coherently_rehashed_bad_preparation_still_rejects(tmp_path, monkeypatch, fault):
    study, release, state = preparation_fixture(tmp_path, monkeypatch)
    release_body = json.loads((tmp_path/release['path']).read_text())
    prepared_binding = release_body['preparation']
    prepared = json.loads((tmp_path/prepared_binding['path']).read_text())
    if fault == 'wrong_phase_cap':
        prepared['supervision']['wall_cap_seconds'] = 121
    else:
        level_binding = state['levels']['24']
        level = json.loads((tmp_path/level_binding['path']).read_text())
        name = level_binding['path']
        if fault == 'failed_level':
            level['status'] = 'failed'
        elif fault == 'wrong_source':
            level['source_sha256'] = '0'*64
        elif fault == 'relocated_level':
            name = 'unrelated-level-receipt.json'
        else:
            level['quality']['relative_volume_error'] = .006
        state['levels']['24'] = save(tmp_path, name, level)
        prepared['state'] = save(tmp_path, prepared['state']['path'], state)
    release_body['preparation'] = save(tmp_path, prepared_binding['path'], prepared)
    changed_release = save(tmp_path, release['path'], release_body)
    with pytest.raises(ValueError, match='level|supervision'):
        resolution.preflight(tmp_path, study, changed_release, 'solve')


def test_resolution_and_legacy_parsers_share_exact_small_state_values():
    times = [i/60 for i in range(61)]
    text = '\n'.join(f'*Step = {i}\n*Time = {t:.9g}\n*Data = tiny\n1,{i},{-i}\n2,{i/2},{-i/2}'
                     for i, t in enumerate(times))
    kwargs = dict(expected_times=times, item_count=2, field_count=2, record_name='tiny')
    old = list(outputs.iter_data_records(text.splitlines(), **kwargs))
    new = list(outputs.iter_resolution_records(text.splitlines(), declaration_sha256=resolution.STUDY_SHA, **kwargs))
    assert len(old) == len(new) == 61
    for left, right in zip(old, new):
        assert set(left) == set(right)
        for key in left:
            if isinstance(left[key], np.ndarray):
                np.testing.assert_array_equal(left[key], right[key])
            else:
                assert left[key] == right[key]
    corrupted = text.replace('2,30.0,-30.0', '1,30.0,-30.0')
    for reader, extra in ((outputs.iter_data_records, {}), (outputs.iter_resolution_records,
                      {'declaration_sha256': resolution.STUDY_SHA})):
        with pytest.raises(ValueError):
            list(reader(corrupted.splitlines(), **kwargs, **extra))


def test_prepare_phase_release_cannot_be_used_for_solve(tmp_path, monkeypatch):
    study, release, _ = preparation_fixture(tmp_path, monkeypatch)
    body = json.loads((tmp_path/release['path']).read_text()); body['phase'] = 'prepare'
    changed = save(tmp_path, 'wrong-phase.json', body)
    monkeypatch.setattr(resolution, '_sources', lambda *args: pytest.fail('Phase must reject before source/runtime access'))
    with pytest.raises(ValueError, match='phase release'):
        resolution.preflight(tmp_path, study, changed, 'solve')


@pytest.mark.parametrize('excess,passed', [(0., True), (1e-12, False)])
def test_absolute_motion_gate_uses_maximum_probe_norm_in_both_groups(excess, passed):
    study = json.loads((ROOT/resolution.STUDY_PATH).read_text())
    runs = {}
    for branch in study['branches']:
        for N, displacement in ((8, -5e-5), (12, -4e-5), (16, 0.), (24, 8e-6+excess)):
            values = np.zeros((61, 75, 3))
            # A single last-state probe must not disappear in an average.
            values[-1, -1, 0] = displacement
            runs[f'{branch}:N{N}:S60:reference'] = {
                'branch': branch, 'mesh_N': N, 'steps': 60,
                'applied_force_N': [0.01]*61, 'probe_displacements_m': values.tolist()}
    report = resolution.comparison_report(runs, study)
    assert report['passed'] is passed
    assert len(report['groups']) == 4
    for group in report['groups'].values():
        assert group['motion']['actual'] == 8e-6+excess
        assert group['motion']['limit'] == 8e-6
    assert report['physical_validation_pass'] is None
