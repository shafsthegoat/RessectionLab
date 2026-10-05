"""Tiny source/orchestration controls: no native mesh, FEBio, or HBE curve reads."""
from copy import deepcopy
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from scripts import mechanics_hbe_resolution as resolution

ROOT = Path(__file__).resolve().parents[1]


def study():
    return json.loads((ROOT/resolution.STUDY_PATH).read_text())


def sample(branch, N, displacement):
    return {'branch': branch, 'steps': 60, 'mesh_N': N, 'applied_force_N': [0.01]*61,
            'probe_displacements_m': np.full((61, 75, 3), displacement).tolist()}


def test_both_prospective_triplets_are_mandatory():
    declaration = study()
    runs = {f'{branch}:N{N}:S60:reference': sample(branch, N, displacement)
            for branch in declaration['branches']
            for N, displacement in ((8, 0.), (12, 0.), (16, 0.), (24, 0.))}
    for row in runs.values():
        row['applied_force_N'] = [{8: .013, 12: .0119, 16: .012, 24: .0119}[row['mesh_N']]]*61
    report = resolution.comparison_report(runs, declaration)
    assert set(report['groups']) == {'compression:N12-N16-N24', 'compression:N8-N16-N24',
                                     'tension:N12-N16-N24', 'tension:N8-N16-N24'}
    # Absolute errors pass, but equal newest force increments cannot hide behind N8.
    assert report['passed'] is False
    assert report['groups']['compression:N12-N16-N24']['motion']['actual'] < 8e-6
    assert report['groups']['compression:N12-N16-N24']['reaction_trend']['actual'] == report['groups']['compression:N12-N16-N24']['reaction_trend']['limit']
    assert report['physical_validation_pass'] is None


def test_comparison_preserves_failure_and_no_measured_claim():
    declaration = study()
    runs = {f'{branch}:N{N}:S60:reference': sample(branch, N, 0.)
            for branch in declaration['branches'] for N in (8, 12, 16, 24)}
    report = resolution.comparison_report(runs, declaration)
    assert report['passed'] is True
    assert report['original_failure'] == declaration['prior_failure']
    assert report['measured_data_accessed'] is False


def case_fixture(tmp_path):
    declaration = study()
    cases = {}
    for key in declaration['ordered_runs']:
        directory = tmp_path/'prepared'/key
        directory.mkdir(parents=True)
        for name, body in [('mesh', '{}'), ('deck', 'unchanged deck'), ('loading', '{}'), ('backend_source_deck', 'source deck')]:
            path = directory/(name+'.txt'); path.write_text(body)
            cases.setdefault(key, {})[name] = resolution.old.binding(tmp_path, path)
    directory = tmp_path/'new-experiment'; directory.mkdir()
    plan = {'directory': str(directory), 'study': declaration, 'study_binding': {'path': resolution.STUDY_PATH, 'sha256': resolution.STUDY_SHA},
            'cases': cases, 'prior_runs': {}, 'inputs': {}, 'executable': str(tmp_path/'solver'),
            'backend_profile': {'path': 'profile', 'sha256': 'a'*64}, 'runtime_identity': {'path': 'runtime', 'sha256': 'b'*64}}
    (tmp_path/'solver').write_text('unexecuted mock executable')
    state = {'solver_invocations': 0, 'runs': {key: {'status': 'not_executed'} for key in declaration['ordered_runs']}}
    return plan, state, SimpleNamespace(active=None)


def test_first_solver_failure_preserves_partial_attempt_and_stops(tmp_path, monkeypatch):
    plan, state, watch = case_fixture(tmp_path)
    calls = []
    def fail(command, directory, seconds):
        calls.append((command, seconds))
        resolution.runtime.write_json(directory/'solver-execution.json', {'exit_code': 7, 'elapsed_seconds': .5})
        raise RuntimeError('declared terminal failure')
    monkeypatch.setattr(resolution.old, 'solve', fail)
    monkeypatch.setattr(resolution.readout, 'read_resolution_run', lambda *a, **k: pytest.fail('No readout after failed solver'))
    with pytest.raises(RuntimeError, match='terminal'):
        resolution.solve_cases(tmp_path, plan, state, watch, resolution.time.monotonic()+1200)
    assert len(calls) == state['solver_invocations'] == 1
    assert calls[0][1] <= 420
    rows = list(state['runs'].values())
    assert rows[0]['status'] == 'failed' and rows[0]['partial_execution']
    assert [v['status'] for v in rows[1:]] == ['not_executed']*3
    assert watch.active is None
    assert (Path(plan['directory'])/'state.json').is_file()


def test_first_individual_gate_failure_stops_before_second_call(tmp_path, monkeypatch):
    plan, state, watch = case_fixture(tmp_path)
    calls = []
    def solve(command, directory, seconds):
        calls.append(command)
        for name in ('nodes.log', 'elements.log', 'solver.log'):
            (directory/name).write_text('mock retained primitive, not solver output')
        return {'exit_code': 0, 'elapsed_seconds': .1, 'timed_out': False}
    monkeypatch.setattr(resolution.old, 'solve', solve)
    monkeypatch.setattr(resolution.readout, 'read_resolution_run', lambda *a, **k: ({'passed': False}, None))
    with pytest.raises(ValueError, match='Individual numerical'):
        resolution.solve_cases(tmp_path, plan, state, watch, resolution.time.monotonic()+1200)
    assert len(calls) == 1
    first = state['runs'][plan['study']['ordered_runs'][0]]
    assert first['readout'] and first['status'] == 'failed'
    assert all(state['runs'][key]['status'] == 'not_executed' for key in plan['study']['ordered_runs'][1:])


def test_no_call_after_exhausted_aggregate_time(tmp_path, monkeypatch):
    plan, state, watch = case_fixture(tmp_path)
    monkeypatch.setattr(resolution.old, 'solve', lambda *a, **k: pytest.fail('Budget exhausted'))
    with pytest.raises(TimeoutError):
        resolution.solve_cases(tmp_path, plan, state, watch, resolution.time.monotonic()-1)
    assert state['solver_invocations'] == 0


def test_late_input_change_stops_before_solver(tmp_path, monkeypatch):
    plan, state, watch = case_fixture(tmp_path)
    path = tmp_path/'bound-input'; path.write_text('before')
    plan['inputs'][str(path)] = resolution.runtime.sha(path)
    path.write_text('after')
    monkeypatch.setattr(resolution.old, 'solve', lambda *a, **k: pytest.fail('Changed input'))
    with pytest.raises(ValueError, match='Bound input changed'):
        resolution.solve_cases(tmp_path, plan, state, watch, resolution.time.monotonic()+1200)
    assert state['solver_invocations'] == 0


@pytest.mark.parametrize('phase, seconds', [('prepare', 120), ('solve', 1200)])
def test_separate_supervised_caps_and_exclusive_attempt(tmp_path, monkeypatch, phase, seconds):
    declaration = study(); root_dir = tmp_path/'new-root'
    plan = {'study': declaration, 'phase': phase, 'inputs': {}, 'output_root': str(root_dir),
            'directory': str(root_dir/phase)}
    monkeypatch.setattr(resolution, 'preflight', lambda *a: plan)
    calls = []
    def supervise(command, output, **kwargs):
        calls.append(kwargs)
        resolution.runtime.write_json(Path(plan['directory'])/'state.json', {'status': 'failed_or_incomplete'})
        return {'status': 'failed_or_incomplete'}
    monkeypatch.setattr(resolution.runtime, 'supervise', supervise)
    result = resolution.launch(tmp_path, {'path': 'study', 'sha256': 'a'*64}, {'path': 'release', 'sha256': 'b'*64}, phase)
    assert result['status'] == 'failed_or_incomplete'
    assert calls[0]['seconds'] == seconds and calls[0]['rss_bytes'] == 3*1024**3
    assert all(calls[0]['environment'][key] == '1' for key in resolution.THREADS)
    with pytest.raises(FileExistsError):
        resolution.launch(tmp_path, {}, {}, phase)
    assert len(calls) == 1


def test_real_declaration_pin_and_original_science_remain_unchanged():
    value = study()
    assert resolution.runtime.sha(ROOT/resolution.STUDY_PATH) == resolution.STUDY_SHA
    assert resolution.runtime.sha(ROOT/value['original_protocol']['path']) == resolution.mesh.PROTOCOL_SHA256
    assert value['co_primary_triplets'] == [[12, 16, 24], [8, 16, 24]]
    assert value['measured_data_access'] is False and value['calibration'] is False
    assert value['budgets']['each_solver_seconds'] == 420
    assert value['budgets']['aggregate_specimen_seconds'] == 1200


def test_exact_four_call_order_then_saved_negative_comparison(tmp_path, monkeypatch):
    plan, state, watch = case_fixture(tmp_path)
    calls = []
    def solve(command, directory, seconds):
        calls.append(directory.name)
        for name in ('nodes.log', 'elements.log', 'solver.log'):
            (directory/name).write_text('explicit mock primitive')
        return {'exit_code': 0, 'elapsed_seconds': .1, 'timed_out': False}
    def read(root, bindings, *, declaration_binding, expected_branch, expected_mesh_N):
        assert declaration_binding['sha256'] == resolution.STUDY_SHA
        return {'passed': True, 'branch': expected_branch, 'mesh_N': expected_mesh_N}, None
    monkeypatch.setattr(resolution.old, 'solve', solve)
    monkeypatch.setattr(resolution.readout, 'read_resolution_run', read)
    monkeypatch.setattr(resolution, 'comparison_report', lambda *args: {'passed': False, 'mock': True})
    with pytest.raises(ValueError, match='Co-primary'):
        resolution.solve_cases(tmp_path, plan, state, watch, resolution.time.monotonic()+1200)
    assert calls == [key.replace(':', '-') for key in plan['study']['ordered_runs']]
    assert state['solver_invocations'] == 4
    assert all(row['status'] == 'passed_individual_numerical_checks' for row in state['runs'].values())
    assert state['comparison'] and (Path(plan['directory'])/'comparison.json').is_file()
    assert not (Path(plan['directory'])/'access.jsonl').exists()


def test_preparation_failure_counts_only_actual_generation_calls(tmp_path, monkeypatch):
    declaration = study(); directory = tmp_path/'mesh-preparation'; directory.mkdir()
    plan = {'study': declaration, 'directory': str(directory), 'study_binding': {'path': resolution.STUDY_PATH},
            'gmsh_runtime': {'path': 'runtime', 'sha256': 'a'*64}, 'inputs': {}}
    state = {'mesh_preparation_invocations': 0, 'gmsh_generation_calls': 0, 'levels': {}, 'cases': {}}
    def fail(N, protocol, extension, extension_sha, runtime, runtime_sha, output):
        output.mkdir(parents=True)
        resolution.runtime.write_json(output/'receipt.json', {'status': 'failed', 'gmsh_generation_calls': 0})
        raise ValueError('before native import')
    monkeypatch.setattr(resolution.mesh, 'prepare_resolution_level', fail)
    with pytest.raises(ValueError, match='before native'):
        resolution.prepare(tmp_path, plan, state, SimpleNamespace(active=None))
    assert state['mesh_preparation_invocations'] == 1
    assert state['gmsh_generation_calls'] == 0
    assert set(state['levels']) == {'16'}


def test_concrete_preparation_draft_refuses_before_runtime_or_native_access(monkeypatch):
    path = ROOT/'artifacts/mechanics/hbe-resolution-preparation-v1/prepare-release-draft.json'
    monkeypatch.setattr(resolution, '_sources', lambda *a: pytest.fail('Draft must refuse before source/runtime checks'))
    with pytest.raises(ValueError, match='Separate explicit phase release'):
        resolution.preflight(ROOT, {'path': resolution.STUDY_PATH, 'sha256': resolution.STUDY_SHA},
                             resolution.old.binding(ROOT, path), 'prepare')
