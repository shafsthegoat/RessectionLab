"""Independent launcher tests: every subprocess/supervisor call is mocked.

No FEBio execution, patient data, or real runtime libraries are accessed.
"""
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    'constraint_launcher_review', ROOT / 'scripts/mechanics_patient_constraints_run.py'
)
v = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(v)


def bound_archive(tmp_path, monkeypatch):
    """Real source/deck hashes, mocked git reads, and tiny fake runtime bytes."""
    archive = tmp_path / 'archive'
    repository = tmp_path / 'repository'
    repository.mkdir()
    original_sources = {}
    for name in v.CLOSURE:
        raw = (ROOT / name).read_bytes()
        path = archive / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(raw)
        original_sources[name] = raw
    executable = tmp_path / 'runtime/install/bin/febio4'
    library = tmp_path / 'runtime/install/lib/example.dylib'
    omp = tmp_path / 'runtime/openmp/lib/libomp.dylib'
    for path in (executable, library, omp):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b'not an executable or shared library')
    identity_path = tmp_path / 'runtime-identity.json'
    identity_path.write_text(json.dumps({
        'executable': str(executable),
        'libraries': {'install/bin/febio4': v.sha(executable), 'install/lib/example.dylib': v.sha(library)},
        'private_openmp': {'path': str(omp), 'sha256': v.sha(omp)}, 'receipts_sha256': {},
    }))
    release = tmp_path / 'release.json'
    release.write_text(json.dumps({
        'schema': v.VERSION + '-release', 'authorized': True, 'root_release': 'MOCK TEST ONLY',
        'source_commit': 'a' * 40, 'source_directory': str(archive),
        'repository_directory': str(repository), 'runtime_identity': str(identity_path),
        'attempt_directory': str(tmp_path / 'attempt'),
    }))
    monkeypatch.setattr(v, 'ROOT', archive)
    monkeypatch.setattr(v, 'RUNTIME_IDENTITY_SHA', v.sha(identity_path))

    def fake_git(command, **kwargs):
        assert command[:4] == ['git', '-C', str(repository), 'show']
        commit, name = command[4].split(':', 1)
        assert commit == 'a' * 40
        assert kwargs == {'capture_output': True, 'check': True, 'timeout': 5}
        return SimpleNamespace(stdout=original_sources[name])

    monkeypatch.setattr(v.subprocess, 'run', fake_git)
    return archive, library, release


def test_baseline_binds_loaded_sources_executable_libraries_release_and_decks(tmp_path, monkeypatch):
    archive, library, release = bound_archive(tmp_path, monkeypatch)
    base = v.baseline(release)
    assert str(library) in base['input_hashes']
    assert base['executable'] in base['input_hashes']
    assert str(release) in base['input_hashes']
    assert all(str(archive / name) in base['input_hashes'] for name in v.CLOSURE)
    assert base['attempt_directory'] == str(tmp_path / 'attempt')
    # Both source and runtime mutations are detected independently of mock git.
    library.write_bytes(b'changed library')
    after = v.unchanged(base['input_hashes'])
    assert after[str(library)]['unchanged'] is False
    with pytest.raises(ValueError, match='Bound input changed'):
        v.baseline(release)


def test_baseline_rejects_checker_changed_from_committed_source(tmp_path, monkeypatch):
    archive, _, release = bound_archive(tmp_path, monkeypatch)
    checker = archive / 'scripts/mechanics_patient_constraints.py'
    checker.write_text(checker.read_text() + '\n# changed since mocked immutable commit\n')
    with pytest.raises(ValueError, match='Source differs from exact commit'):
        v.baseline(release)


def test_parent_forwards_fixed_caps_and_private_one_thread_environment(tmp_path, monkeypatch):
    output = tmp_path / 'attempt'
    bound = tmp_path / 'bound-input'
    bound.write_text('unchanged')
    inputs = {str(bound): v.sha(bound)}
    monkeypatch.setattr(v, 'baseline', lambda _: {'input_hashes': inputs, 'attempt_directory': str(output)})
    # Load the real environment function but replace supervision before use.
    runtime_spec = importlib.util.spec_from_file_location('reviewed_env_only', ROOT / 'scripts/febio_runtime.py')
    runtime = importlib.util.module_from_spec(runtime_spec)
    runtime_spec.loader.exec_module(runtime)
    monkeypatch.setenv('OMP_NUM_THREADS', '19')
    monkeypatch.setenv('PYTHONPATH', '/not-a-permitted-dependency-root')
    observed = []

    def supervise(command, directory, *, cwd, environment, seconds, rss_bytes):
        observed.append(command)
        assert seconds == 60 and rss_bytes == 3 * 1024 ** 3
        assert cwd == ROOT and directory == output / 'supervision'
        assert command[2:] == ['worker', '--output', str(output)]
        assert 'PYTHONPATH' not in environment
        for name, value in runtime.declaration()['caps']['thread_environment'].items():
            assert environment[name] == value
            assert value == ('FALSE' if name == 'OMP_DYNAMIC' else '1')
        v.write(output / 'results.json', {
            'status': 'completed', 'solver_invocations': 3, 'no_retry': True,
            'cases': [{'case': case, 'status': 'passed', 'checker_passed': True} for case in v.CASES],
            'inputs_after': {str(bound): {'unchanged': True}},
        })
        return {'status': 'completed', 'exit_code': 0}

    monkeypatch.setattr(runtime, 'supervise', supervise)
    monkeypatch.setattr(v, 'load', lambda *_: runtime)
    monkeypatch.setattr(v.subprocess, 'run', lambda *_, **__: pytest.fail('No subprocess allowed'))
    assert v.launch('mock-release', output) == 0
    assert len(observed) == 1


def test_worker_rejects_other_output_directory_before_solver_call(tmp_path, monkeypatch):
    output = tmp_path / 'unauthorized-output'
    output.mkdir()
    bound = tmp_path / 'bound'
    bound.write_text('fixed')
    base = {'attempt_directory': str(tmp_path / 'only-released-attempt'),
            'input_hashes': {str(bound): v.sha(bound)}, 'release_path': 'mock-release'}
    v.write(output / 'execution-baseline.json', base)
    v.write(output / 'results.json', v.initial_result())
    monkeypatch.setattr(v, 'baseline', lambda _: base)
    monkeypatch.setattr(v.subprocess, 'run', lambda *_, **__: pytest.fail('No subprocess allowed'))
    assert v.worker(output) == 1
    result = json.loads((output / 'results.json').read_text())
    assert result['solver_invocations'] == 0
    assert 'single released attempt' in result['worker_error']
    assert all(row['status'] == 'not_executed' for row in result['cases'])


def test_repeated_worker_refusal_preserves_original_completed_receipt(tmp_path, monkeypatch):
    output = tmp_path / 'existing-attempt'
    output.mkdir()
    original = {'status': 'completed', 'cases': ['original evidence must survive']}
    v.write(output / 'results.json', original)
    previous_bytes = (output / 'results.json').read_bytes()
    monkeypatch.setattr(v, 'baseline', lambda _: pytest.fail('No repeated attempt'))
    monkeypatch.setattr(v.subprocess, 'run', lambda *_, **__: pytest.fail('No subprocess allowed'))
    try:
        v.worker(output)
    except ValueError:
        pass
    assert (output / 'results.json').read_bytes() == previous_bytes
