"""Collect this one terminal resolution phase; no solver, mesher or data-reader imports.

Do not execute until the parent confirms the separately supervised run ended.
Every generated artifact is exclusive. Only the closed experiment tree is frozen.
"""
from pathlib import Path
import hashlib
import json
import math
import re
import stat
import time

ROOT = Path(__file__).resolve().parents[3]
RAW = ROOT/'outputs/mechanics/hbe-01-03-resolution-v1/experiment'
OUT = Path(__file__).resolve().parent
EXPECTED_RELEASE_SHA = 'd9fc28acaec95142711784a4ec838f9f76e05548bea97d6d1e7388e4d62d11e9'
assert (ROOT/'scripts/mechanics_hbe_resolution.py').is_file()
cache = {}


def digest(path):
    value = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024**2), b''):
            value.update(block)
    return value.hexdigest()


def sha(path):
    path = Path(path).resolve()
    if path not in cache:
        cache[path] = digest(path)
    return cache[path]


def relative(path):
    return str(Path(path).resolve().relative_to(ROOT))


def binding(path):
    return {'path': relative(path), 'sha256': sha(path)}


def load(path):
    return json.loads(Path(path).read_text())


def verify(record):
    path = ROOT/record['path']
    assert sha(path) == record['sha256'], path
    return path


def save(name, value):
    path = OUT/name
    with path.open('x') as stream:
        stream.write(json.dumps(value, indent=2, sort_keys=True, allow_nan=False)+'\n')
    return binding(path)


def old_index(path):
    index = load(path)
    base = ROOT/index['root']
    actual = {str(p.relative_to(base)) for p in base.rglob('*') if p.is_file()}
    assert actual == set(index['files'])
    for name, row in index['files'].items():
        target = base/name
        assert not target.is_symlink()
        assert target.stat().st_size == row['bytes'] and sha(target) == row['sha256'], target
    return {'index': binding(path), 'file_count': index['file_count'], 'bytes': index['bytes'],
            'all_hashes_sizes_and_inventory_unchanged': True}



def saved_log_progress(path):
    """Small closed solver log only; never infer primitive validity from banners."""
    if not path.is_file():
        return {'status': 'no_saved_solver_log'}
    assert path.stat().st_size <= 16*1024**2
    text = path.read_text()
    active = None
    started_steps = []
    converged = []
    for line in text.splitlines():
        begin = re.match(r'^===== beginning time step (\d+) : (\S+) =====', line)
        if begin:
            active = {'step': int(begin[1]), 'printed_pseudotime': begin[2]}
            started_steps.append(active)
        done = re.match(r'^------- converged at time : (\S+)', line)
        if done:
            assert active is not None and float(done[1]) == float(active['printed_pseudotime'])
            converged.append(dict(active))
    timing = {}
    for name, pattern in [('solve_seconds', r'Solve time.*?\(([^ ]+) sec\)'),
                          ('linear_solver_seconds', r'time in linear solver.*?\(([^ ]+) sec\)')]:
        matches = re.findall(pattern, text)
        if matches:
            timing[name] = float(matches[-1])
    for name, label in [('completed_steps', 'Number of time steps completed'),
                        ('equilibrium_iterations', 'Total number of equilibrium iterations'),
                        ('right_hand_evaluations', 'Total number of right hand evaluations'),
                        ('stiffness_reformations', 'Total number of stiffness reformations'),
                        ('linear_solver_calls', 'Total calls to linear solver')]:
        matches = re.findall(re.escape(label)+r'[^\n]*?:\s*(\d+)', text)
        if matches:
            timing[name] = int(matches[-1])
    for name, pattern in [('rounded_linear_solver_time', r'Time in linear solver:\s*(\d+:\d+:\d+)'),
                          ('rounded_elapsed_summary', r'^\s*Elapsed time\s*:\s*(\d+:\d+:\d+)')]:
        matches = re.findall(pattern, text, re.MULTILINE)
        if matches:
            timing[name] = matches[-1]
    return {'solver_log': binding(path), 'begun_step_count': len(started_steps),
            'converged_banner_count': len(converged),
            'last_begun_step': started_steps[-1] if started_steps else None,
            'last_log_confirmed_converged_step': converged[-1] if converged else None,
            'normal_termination_banner': 'N O R M A L   T E R M I N A T I O N' in text,
            'reported_timing': timing,
            'timing_scope': 'Native saved summary counters/timing; not an independent precise profiler split.',
            'limit': 'Saved banners only; later computation or primitive blocks may not yet have been flushed. A partial run has no completed readout or individual numerical pass.'}

def main():
    started = time.monotonic()
    # Fail before hashing large inputs or changing permissions unless terminal.
    result = load(RAW/'result.json')
    supervision = load(RAW/'supervision/supervision.json')
    state = load(RAW/'state.json')
    baseline = load(RAW/'baseline.json')
    assert result['phase'] == state['phase'] == baseline['phase'] == 'solve'
    assert supervision['status'] in ('completed', 'failed_or_incomplete')
    assert type(supervision['exit_code']) is int and supervision['status'] != 'running'
    assert result['supervision'] == supervision
    assert state['status'] in ('completed_numerical_resolution_only', 'failed_or_incomplete')
    assert baseline['directory'] == str(RAW) and baseline['root'] == str(ROOT)
    assert baseline['release_binding']['sha256'] == EXPECTED_RELEASE_SHA
    assert state['solver_invocations'] <= 4 and state['gmsh_generation_calls'] == 0
    assert state['measured_data_accessed'] is False
    for key in ('baseline', 'state'):
        if key in result:
            verify(result[key])
    release = load(verify(baseline['release_binding']))
    verify(release['source_archive'])
    preserved = {}
    for name, expected in baseline['inputs'].items():
        path = Path(name)
        assert sha(path) == expected, path
        preserved[relative(path) if path.is_relative_to(ROOT) else str(path)] = {'sha256': expected, 'unchanged': True}
    prior = old_index(ROOT/'artifacts/mechanics/hbe-accelerate-experiment-result-v1/raw-output-index.json')
    meshes = old_index(ROOT/'artifacts/mechanics/hbe-resolution-mesh-result-v1/raw-output-index.json')
    preservation = save('preservation.json', {'baseline': binding(RAW/'baseline.json'),
        'baseline_input_count': len(preserved), 'all_baseline_inputs_unchanged': True, 'baseline_inputs': preserved,
        'previous_failed_experiment': prior, 'accepted_mesh_preparation': meshes,
        'source_commit': release['source_commit'], 'source_archive': release['source_archive'],
        'source_bindings': baseline['source_bindings']})
    runs = {}
    for key in baseline['study']['ordered_runs']:
        row = state['runs'][key]
        summary = {'status': row['status'], 'execution': row.get('execution')}
        for name in ('readout', 'partial_execution'):
            if name in row:
                verify(row[name]); summary[name] = row[name]
        if 'readout' in row:
            receipt = load(ROOT/row['readout']['path'])
            execution = load(verify(receipt['execution_binding']))
            assert execution['runtime_identity'] == baseline['runtime_identity']
            assert execution['backend_profile'] == baseline['backend_profile']
            assert execution['primitive_bindings'] == receipt['primitive_bindings']
            assert execution['study'] == baseline['study_binding'] and execution['run_id'] == key
            for record in receipt['primitive_bindings'].values():
                verify(record)
            summary.update(individual_passed=receipt['passed'], criteria=receipt['criteria'],
                           frame_count=receipt['frame_count'], execution_binding=receipt['execution_binding'])
        directory = RAW/'runs'/key.replace(':', '-')
        if directory.is_dir():
            summary['saved_solver_log_progress'] = saved_log_progress(directory/'solver.log')
            if summary['execution'] is None and 'partial_execution' in summary:
                summary['partial_solver_execution'] = load(ROOT/summary['partial_execution']['path'])
        for name, expected in row.get('retained_files', {}).items():
            path = directory/name
            assert path.stat().st_size == expected['bytes'] and sha(path) == expected['sha256']
        runs[key] = summary
    run_outcomes = save('run-outcomes.json', {'solver_invocations': state['solver_invocations'], 'runs': runs})
    gates = None
    if (RAW/'comparison.json').is_file():
        comparison = load(RAW/'comparison.json')
        assert comparison['co_primary_triplets'] == [[12, 16, 24], [8, 16, 24]]
        assert comparison['measured_data_accessed'] is False and comparison['physical_validation_pass'] is None
        records = []
        for group, criteria in comparison['groups'].items():
            for name, metric in criteria.items():
                actual, limit = metric['actual'], metric['limit']
                assert math.isfinite(actual) and math.isfinite(limit)
                operator = metric.get('comparison', 'le')
                assert operator in ('lt', 'le')
                passed = actual < limit if operator == 'lt' else actual <= limit
                records.append({'group': group, 'criterion': name, **metric, 'operator': operator, 'passed': passed})
        assert len(records) == 16 and comparison['passed'] == all(v['passed'] for v in records)
        if 'comparison' in state:
            verify(state['comparison'])
        gates = save('aggregate-gates.json', {'comparison': binding(RAW/'comparison.json'),
            'arithmetic': 'Saved scalar metrics and declared comparisons only; no field/probe recomputation.',
            'criterion_count': len(records), 'passed': comparison['passed'],
            'criteria': records, 'failures': [v for v in records if not v['passed']]})
    absent = ['access.jsonl', 'fit.json', 'parameter-prediction-freeze.json', 'held-out-metrics.json']
    assert all(not (RAW/name).exists() for name in absent)
    access_record = save('access-chronology.json', {'measured_data_accessed': state['measured_data_accessed'],
        'measured_member_reads_by_collector': 0, 'calibration_in_protocol': False,
        'absent_response_fit_freeze_paths': absent, 'no_fit_or_held_out_phase_in_this_study': True})
    files = {}
    for path in sorted(RAW.rglob('*')):
        assert not path.is_symlink(), path
        if path.is_file():
            files[str(path.relative_to(RAW))] = {'bytes': path.stat().st_size, 'sha256': sha(path)}
    raw_index = save('raw-output-index.json', {'root': relative(RAW), 'file_count': len(files),
        'bytes': sum(row['bytes'] for row in files.values()), 'files': files})
    parent_mode = stat.S_IMODE(RAW.parent.stat().st_mode)
    for path in RAW.rglob('*'):
        if path.is_file(): path.chmod(0o444)
    for path in sorted([v for v in RAW.rglob('*') if v.is_dir()], key=lambda v: len(v.parts), reverse=True):
        path.chmod(0o555)
    RAW.chmod(0o555)
    assert stat.S_IMODE(RAW.parent.stat().st_mode) == parent_mode and parent_mode & 0o200
    for name, row in files.items():
        path = RAW/name
        assert digest(path) == row['sha256'] and stat.S_IMODE(path.stat().st_mode) == 0o444
    assert all(stat.S_IMODE(p.stat().st_mode) == 0o555 for p in [RAW]+[v for v in RAW.rglob('*') if v.is_dir()])
    freeze = save('readonly-freeze.json', {'raw_index': raw_index, 'file_count': len(files),
        'phase_directory_only': relative(RAW), 'file_mode': '0444', 'directory_mode': '0555',
        'study_root_mode_preserved': oct(parent_mode), 'study_root_remains_owner_writable': True,
        'all_raw_bytes_rehashed_after_permission_freeze': True, 'raw_content_edited': False,
        'compression_or_deletion': False})
    completed = [(row.get('execution') or row.get('partial_solver_execution'))['elapsed_seconds']
                 for row in runs.values() if row.get('execution') or row.get('partial_solver_execution')]
    outcome = {'schema': 'hbe-resolution-experiment-outcome-v1', 'status': result['status'],
        'worker_status': state['status'], 'error': state.get('error'), 'result': binding(RAW/'result.json'),
        'state': binding(RAW/'state.json'), 'baseline': binding(RAW/'baseline.json'),
        'release': baseline['release_binding'], 'source_commit': release['source_commit'],
        'source_archive': release['source_archive'], 'study': baseline['study_binding'],
        'runtime_identity': baseline['runtime_identity'], 'backend_profile': baseline['backend_profile'],
        'supervision': supervision, 'worker_elapsed_seconds': state.get('elapsed_seconds'),
        'sum_recorded_solver_elapsed_seconds': sum(completed), 'solver_invocations': state['solver_invocations'],
        'gmsh_generation_calls': 0, 'measured_data_accessed': False, 'physical_validation_pass': None,
        'aggregate_gates': gates, 'run_outcomes': run_outcomes, 'raw_index': raw_index,
        'preservation': preservation, 'access_chronology': access_record, 'readonly_freeze': freeze,
        'output_watch': load(RAW/'output-watch.json'), 'collector': binding(__file__),
        'collector_scope': 'Saved JSON, scalar comparisons, file hashes/sizes and permissions only; no primitive replay, solver, Gmsh or measured member reads.',
        'independent_numerical_review': 'separate_review_pending_at_collection',
        'collection_seconds': time.monotonic()-started}
    save('outcome.json', outcome)
    print(json.dumps({'outcome': binding(OUT/'outcome.json'), 'raw_index': raw_index,
        'raw_file_count': len(files), 'raw_bytes': sum(v['bytes'] for v in files.values()),
        'status': outcome['status'], 'solver_invocations': state['solver_invocations']}, indent=2))


if __name__ == '__main__':
    main()
