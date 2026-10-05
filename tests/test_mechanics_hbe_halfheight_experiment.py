"""Orchestration controls with explicit doubles: no real meshes, tools or curves."""
from copy import deepcopy
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from scripts import mechanics_hbe_halfheight_experiment as runner

ROOT = Path(__file__).resolve().parents[1]


def study():
    return json.loads((ROOT/runner.DECLARATION_PATH).read_text())


def plan_fixture(tmp_path):
    declaration = study()
    directory = tmp_path/'study'/'preparation'; directory.mkdir(parents=True)
    full = runner.old.saved(tmp_path, tmp_path/'full.json', {'mock_full': True})
    source = runner.old.saved(tmp_path, tmp_path/'mesh-source.json', {'mock_source': True})
    plan = {'study': declaration, 'study_binding': {'path': runner.DECLARATION_PATH, 'sha256': runner.DECLARATION_SHA},
            'directory': str(directory), 'output_root': str(directory.parent), 'protocol': {'mock': True}, 'inputs': {},
            'source_bindings': {'halfheight_mesh': source}, 'backend_profile': declaration['original_backend_profile'],
            'runtime_identity': declaration['runtime_identity'],
            'full_references': {key: {'primitive_bindings': {'mesh': full}} for key in declaration['ordered_runs']}}
    state = {'extraction_invocations': 0, 'solver_invocations': 0, 'gmsh_generation_calls': 0,
             'measured_data_accessed': False, 'cases': {}, 'levels': {},
             'runs': {key: {'status': 'not_executed'} for key in declaration['ordered_runs']}}
    return plan, state, SimpleNamespace(active=None)


def prepare_double(tmp_path, monkeypatch):
    plan, state, watch = plan_fixture(tmp_path)
    extraction = {'mesh': {'mock_half': True}, 'mapping': {'mock_map': True}, 'verification': {'mock_verified': True}}
    monkeypatch.setattr(runner.half_mesh, 'extract_halfheight', lambda *a: deepcopy(extraction))
    monkeypatch.setattr(runner.half_mesh, 'halfheight_deck',
                        lambda extraction, branch, protocol: ('<febio_spec><Control><solver><linear_solver type="skyline" /></solver></Control></febio_spec>', {'branch': branch}))
    runner.prepare(tmp_path, plan, state, watch)
    plan.update(cases=state['cases'], levels=state['levels'])
    return plan, state, watch


def solve_fixture(tmp_path, monkeypatch):
    plan, state, watch = prepare_double(tmp_path, monkeypatch)
    path = tmp_path/'study'/'experiment'; path.mkdir(); plan['directory'] = str(path)
    executable = tmp_path/'mock-executable'; executable.write_text('Never loaded')
    plan['executable'] = str(executable)
    return plan, state, watch


def test_exact_pure_preparation_then_in_memory_verification(tmp_path, monkeypatch):
    plan, state, watch = prepare_double(tmp_path, monkeypatch)
    assert state['extraction_invocations'] == 2 and state['solver_invocations'] == state['gmsh_generation_calls'] == 0
    assert state['status'] == 'prepared_not_solved' and set(state['cases']) == set(study()['ordered_runs'])
    before = {p: p.read_bytes() for p in Path(plan['directory']).rglob('*') if p.is_file()}
    runner.verify_prepared_geometry(tmp_path, plan)
    assert {p: p.read_bytes() for p in before} == before
    for key, row in state['cases'].items():
        loading = runner.access.verify_binding(tmp_path, row['loading'], read_json=True)
        assert loading['halfheight_equivalence_declaration_sha256'] == runner.DECLARATION_SHA
        assert loading['reconstruction_sha256'] == row['reconstruction']['sha256']
        assert loading['full_mesh_sha256'] == plan['full_references'][key]['primitive_bindings']['mesh']['sha256']
        assert loading['deck_sha256'] == row['deck']['sha256']


@pytest.mark.parametrize('field', ['mesh', 'reconstruction', 'deck', 'loading'])
def test_coherently_rehashed_prepared_drift_rejected(tmp_path, monkeypatch, field):
    plan, state, watch = prepare_double(tmp_path, monkeypatch)
    first = plan['study']['ordered_runs'][0]
    binding = plan['cases'][first][field]
    path = runner.access.local_path(tmp_path, binding['path'])
    if field == 'deck':
        path.write_text(path.read_text() + '<changed_boundary/>')
    else:
        value = json.loads(path.read_text()); value['coherent_but_wrong'] = True
        runner.runtime.write_json(path, value)
    binding['sha256'] = runner.runtime.sha(path)
    with pytest.raises(ValueError, match='differs'):
        runner.verify_prepared_geometry(tmp_path, plan)


def test_original_extraction_failure_retained_before_any_native(tmp_path, monkeypatch):
    plan, state, watch = plan_fixture(tmp_path)
    monkeypatch.setattr(runner.half_mesh, 'extract_halfheight', lambda *a: (_ for _ in ()).throw(ValueError('straddler')))
    with pytest.raises(ValueError, match='straddler'):
        runner.prepare(tmp_path, plan, state, watch)
    assert state['extraction_invocations'] == 1 and not state['cases']
    assert state['solver_invocations'] == state['gmsh_generation_calls'] == 0


def mock_solver(monkeypatch, calls, *, fail=False):
    def solve(command, directory, seconds):
        calls.append((directory.name, seconds))
        execution = {'exit_code': 7 if fail else 0, 'elapsed_seconds': .01, 'timed_out': False}
        runner.runtime.write_json(directory/'solver-execution.json', execution)
        if fail:
            raise RuntimeError('mock terminal native failure')
        for name in ('nodes.log', 'elements.log', 'solver.log'):
            (directory/name).write_text('Explicit mocked primitive; no physics claim')
        return execution
    monkeypatch.setattr(runner.old, 'solve', solve)


def test_first_solver_failure_is_terminal_and_preserves_partial(tmp_path, monkeypatch):
    plan, state, watch = solve_fixture(tmp_path, monkeypatch); calls = []
    mock_solver(monkeypatch, calls, fail=True)
    monkeypatch.setattr(runner.half_readout, 'read_halfheight_equivalence', lambda *a, **k: pytest.fail('no failed readout'))
    with pytest.raises(RuntimeError, match='terminal'):
        runner.solve_cases(tmp_path, plan, state, watch, runner.time.monotonic()+420)
    assert state['solver_invocations'] == len(calls) == 1 and calls[0][1] <= 90
    rows = list(state['runs'].values())
    assert rows[0]['status'] == 'failed' and rows[0]['partial_execution'] and rows[0]['retained_files']
    assert [row['status'] for row in rows[1:]] == ['not_executed']*3
    assert watch.active is None


def test_equivalence_failure_stops_before_second_solve(tmp_path, monkeypatch):
    plan, state, watch = solve_fixture(tmp_path, monkeypatch); calls = []
    mock_solver(monkeypatch, calls)
    monkeypatch.setattr(runner.half_readout, 'read_halfheight_equivalence', lambda *a, **k: {'passed': False})
    with pytest.raises(ValueError, match='equivalence gate'):
        runner.solve_cases(tmp_path, plan, state, watch, runner.time.monotonic()+420)
    assert len(calls) == 1 and state['runs'][plan['study']['ordered_runs'][0]]['readout']


def test_exact_four_calls_and_reference_dispatch_without_response_api(tmp_path, monkeypatch):
    plan, state, watch = solve_fixture(tmp_path, monkeypatch); calls = []; read_calls = []
    mock_solver(monkeypatch, calls)
    def read(root, **kwargs):
        read_calls.append(kwargs)
        key = f"{kwargs['expected_branch']}:N{kwargs['expected_mesh_N']}:S60:reference"
        assert kwargs['full_bindings'] == plan['full_references'][key]['primitive_bindings']
        assert kwargs['reconstruction_binding'] == plan['cases'][key]['reconstruction']
        return {'passed': True, 'explicit_mock': True}
    monkeypatch.setattr(runner.half_readout, 'read_halfheight_equivalence', read)
    runner.solve_cases(tmp_path, plan, state, watch, runner.time.monotonic()+420)
    assert [name for name, _ in calls] == [key.replace(':', '-') for key in plan['study']['ordered_runs']]
    assert len(read_calls) == state['solver_invocations'] == 4
    assert state['status'] == 'completed_numerical_equivalence_only'
    assert not (Path(plan['directory'])/'access.jsonl').exists()


@pytest.mark.parametrize('mode', ['deadline', 'input_change'])
def test_budget_or_late_mutation_prevents_native_call(tmp_path, monkeypatch, mode):
    plan, state, watch = solve_fixture(tmp_path, monkeypatch)
    deadline = runner.time.monotonic()+420
    if mode == 'deadline':
        deadline = runner.time.monotonic()-1
    else:
        path = tmp_path/'original-bound'; path.write_text('before'); plan['inputs'][str(path)] = runner.runtime.sha(path)
        path.write_text('after')
    monkeypatch.setattr(runner.old, 'solve', lambda *a: pytest.fail('must not invoke'))
    with pytest.raises((ValueError, TimeoutError)):
        runner.solve_cases(tmp_path, plan, state, watch, deadline)
    assert state['solver_invocations'] == 0


@pytest.mark.parametrize('phase, seconds', [('prepare', 60), ('solve', 420)])
def test_phase_watchdog_limits_and_exclusive_marker(tmp_path, monkeypatch, phase, seconds):
    declaration = study(); root = tmp_path/'study'
    plan = {'study': declaration, 'phase': phase, 'inputs': {}, 'output_root': str(root), 'directory': str(root/phase)}
    monkeypatch.setattr(runner, 'preflight', lambda *a: plan)
    calls = []
    def supervise(command, output, **kwargs):
        calls.append(kwargs)
        runner.runtime.write_json(Path(plan['directory'])/'state.json', {'status': 'failed_or_incomplete'})
        return {'status': 'failed_or_incomplete'}
    monkeypatch.setattr(runner.runtime, 'supervise', supervise)
    runner.launch(tmp_path, {'path': 'study', 'sha256': 'a'*64}, {'path': 'release', 'sha256': 'b'*64}, phase)
    assert calls[0]['seconds'] == seconds and calls[0]['rss_bytes'] == 3*1024**3
    assert all(calls[0]['environment'][key] == '1' for key in runner.THREADS)
    with pytest.raises(FileExistsError):
        runner.launch(tmp_path, {}, {}, phase)
    assert len(calls) == 1


def test_frozen_contract_and_unauthorized_draft_refuse_early(tmp_path, monkeypatch):
    assert runner.runtime.sha(ROOT/runner.DECLARATION_PATH) == runner.DECLARATION_SHA
    path = tmp_path/runner.DECLARATION_PATH; path.parent.mkdir(parents=True); path.write_bytes((ROOT/runner.DECLARATION_PATH).read_bytes())
    binding = {'path': runner.DECLARATION_PATH, 'sha256': runner.DECLARATION_SHA}
    release = runner.old.saved(tmp_path, tmp_path/'draft.json', {'schema': 'hbe-halfheight-release-draft-v1', 'authorized': False,
        'phase': 'prepare', 'study': binding})
    monkeypatch.setattr(runner, 'source_inventory', lambda *a: pytest.fail('must refuse before source/runtime'))
    with pytest.raises(ValueError, match='Separate explicit'):
        runner.preflight(tmp_path, binding, release, 'prepare')
