"""Saved-result audit; two bounded raw-readout replays, never a solver or curve reader."""
import hashlib
import json
import math
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[3]
RAW = ROOT / 'outputs/mechanics/hbe-01-03-poc-v1/experiment-accelerate-csc-v1'
FROZEN = ROOT / 'outputs/validation/hbe-accelerate-6486dbc/frozen-source'
RELEASE_SHA = 'edb91bdbaa542852ec65a87eba8e2676556d926d9c979ceea3cb8e460624ba6b'
seen = {}


def check(path, expected=None):
    path = (ROOT / path).resolve()
    assert path.suffix not in ('.csv', '.zip', '.nii')
    with path.open('rb') as stream:
        digest = hashlib.file_digest(stream, 'sha256').hexdigest()
    assert expected is None or digest == expected, str(path)
    assert str(path) not in seen or seen[str(path)] == digest, str(path)
    seen[str(path)] = digest
    return path


def read(path):
    return json.loads(check(path).read_bytes())


def bound(record, parse=False):
    path = check(record['path'], record['sha256'])
    return json.loads(path.read_bytes()) if parse else path


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False)


start = time.monotonic()
result = read(RAW / 'result.json')
state = bound(result['state'], True)
baseline = bound(result['baseline'], True)
report = read(RAW / 'reference-numerical.json')
watch = read(RAW / 'output-watch.json')
release = bound(state['release'], True)
assert state['release']['sha256'] == RELEASE_SHA
assert baseline['release_binding'] == state['release']
assert result['status'] == state['status'] == 'failed_or_incomplete'
assert state['error'] == {'type': 'ValueError', 'message': 'Numerical error gate failed'}
assert state['solver_invocations'] == state['new_solver_invocations'] == 18
assert state['charged_prior_parent_seconds'] == 0 and state['automatic_retry'] is False
assert all(state[field] is False for field in ('calibration_access_attempted', 'calibration_responses_accessed',
                                              'held_out_access_attempted', 'held_out_responses_accessed'))
assert result['physical_validation_pass'] is None
assert 'continuation' not in release['execution']
assert state['original_inputs_unchanged'] is True
for path, digest in baseline['inputs'].items():
    check(path, digest)
for binding in report['source_bindings'].values():
    bound(binding)
for name in ('access.jsonl', 'fit.json', 'predictions.json', 'parameter-prediction-freeze.json', 'held-out.json', 'numerical.json'):
    assert not (RAW / name).exists(), name
legacy_hashes = {}
for name in ('result.json', 'state.json'):
    path = ROOT / 'outputs/mechanics/hbe-01-03-poc-v1/experiment' / name
    if path.exists():
        check(path)
        legacy_hashes[str(path.relative_to(ROOT))] = seen[str(path)]
run_receipts, run_seconds, pids = {}, [], []
fitted_not_executed = []
for name, row in state['runs'].items():
    if name.endswith(':fitted'):
        assert row == {'status': 'not_executed'}
        fitted_not_executed.append(name)
        continue
    assert row['status'] == 'passed_individual_numerical_checks'
    execution = row['execution']
    assert execution['status'] == 'completed' and type(execution['exit_code']) is int and execution['exit_code'] == 0
    assert execution['timed_out'] is False and execution['cleanup']['reaped'] is True
    assert 0 < execution['elapsed_seconds'] <= execution['seconds_cap'] == 90
    pids.append(execution['pid']);run_seconds.append(execution['elapsed_seconds'])
    for filename, binding in row['retained_files'].items():
        path = check(Path(row['directory']) / filename, binding['sha256'])
        assert path.stat().st_size == binding['bytes']
    readout = bound(row['readout'], True)
    assert readout == report['runs'][name]
    assert readout['passed'] is True
    for criterion, metric in readout['criteria'].items():
        assert math.isfinite(metric['actual'])
        assert metric['actual'] > 0 if criterion == 'minimum_sampled_J' else metric['actual'] <= metric['limit']
    execution_record = bound(readout['execution_binding'], True)
    assert execution_record['run_id'] == name
    assert execution_record['execution'] == execution
    assert execution_record['runtime_identity'] == baseline['runtime_identity']
    assert execution_record['backend_profile'] == baseline['backend_profile']
    assert execution_record['primitive_bindings'] == readout['primitive_bindings'] == row['input_bindings']
    assert execution_record['backend_source_deck'] == baseline['backend_source_decks'][name]
    assert execution_record['command'] == row['command'] == execution['command']
    assert (ROOT / execution_record['cwd']).resolve() == Path(execution['cwd']).resolve()
    assert execution_record['executable']['sha256'] == row['executable_sha256']
    run_receipts[name] = readout
assert len(run_receipts) == len(set(pids)) == 18 and len(fitted_not_executed) == 2
assert len(list((RAW / 'runs').glob('*/solver-execution.json'))) == 18

# Independent array arithmetic for all mesh and load-step comparisons.
import numpy as np
recomputed = {'mesh': {}, 'step': {}}
for branch in ('compression', 'tension', 'torsion_neg', 'torsion_pos'):
    response_field = 'applied_torque_Nm' if branch.startswith('torsion') else 'applied_force_N'
    units = 'Nm' if branch.startswith('torsion') else 'N'
    scale = 1000. * .004 ** (3 if units == 'Nm' else 2)
    for kind, names in (
        ('mesh', [f'{branch}:N{n}:S60:reference' for n in (4, 8, 12)]),
        ('step', [f'{branch}:N12:S60:reference'] * 2 + [f'{branch}:N12:S120:reference']),
    ):
        sample = [(np.asarray(run_receipts[n][response_field])[::run_receipts[n]['steps']//60],
                   np.asarray(run_receipts[n]['probe_displacements_m'])[::run_receipts[n]['steps']//60]) for n in names]
        (r0, p0), (r1, p1), (r2, p2) = sample
        assert all(p.shape == (61, 75, 3) for _, p in sample)
        before_r, after_r = float(np.max(np.abs(r1-r0))), float(np.max(np.abs(r2-r1)))
        before_p, after_p = float(np.max(np.linalg.norm(p1-p0, axis=-1))), float(np.max(np.linalg.norm(p2-p1, axis=-1)))
        relative, floor, motion = (.02, 1e-5, .002) if kind == 'mesh' else (.002, 1e-6, .0002)
        group = {'reaction': {'actual': after_r, 'limit': floor*scale+relative*float(np.max(np.abs(r2))), 'units': units},
                 'motion': {'actual': after_p, 'limit': motion*.004, 'units': 'm'}}
        if kind == 'mesh':
            for key, current, previous, minimum, unit in (
                ('reaction_trend', after_r, before_r, 1e-5*scale, units),
                ('motion_trend', after_p, before_p, .002*.004, 'm')):
                group[key] = ({'actual': max(current, previous), 'limit': minimum, 'units': unit}
                              if max(current, previous) <= minimum else
                              {'actual': current, 'limit': previous, 'units': unit, 'comparison': 'lt'})
        assert group == report[kind][branch], (kind, branch)
        recomputed[kind][branch] = group
failures, metric_count = [], 0
for category in ('mesh', 'step', 'scale'):
    for branch, group in report[category].items():
        for name, metric in group.items():
            metric_count += 1
            assert math.isfinite(metric['actual']) and math.isfinite(metric['limit'])
            passed = metric['actual'] < metric['limit'] if metric.get('comparison') == 'lt' else metric['actual'] <= metric['limit']
            if not passed:
                failures.append({'metric': f'{category}.{branch}.{name}', **metric})
assert [row['metric'] for row in failures] == ['mesh.compression.motion', 'mesh.tension.motion', 'mesh.tension.reaction_trend']

# Two saved raw replays confirm the medium/fine tension fields driving failure.
sys.path.insert(0, str(FROZEN))
from scripts import mechanics_hbe_readout as reader
for name in ('mechanics_hbe_readout', 'mechanics_hbe_access', 'mechanics_hbe_outputs', 'mechanics_hbe_physics', 'mechanics_hbe_evaluation'):
    module = sys.modules['scripts.' + name]
    assert Path(module.__file__).resolve() == FROZEN / 'scripts' / (name + '.py')
replays = []
for n in (8, 12):
    name = f'tension:N{n}:S60:reference'
    expected = run_receipts[name]
    replay_start = time.monotonic()
    actual, cache = reader.read_run(ROOT, expected['primitive_bindings'], protocol_sha256=report['protocol_sha256'],
        expected_branch='tension', expected_mesh_N=n, expected_steps=60, expected_mu_Pa=1000., retain_scale_primitives=False)
    assert cache is None
    expected_without_execution = {key: value for key, value in expected.items() if key != 'execution_binding'}
    assert canonical(actual) == canonical(expected_without_execution)
    replays.append({'run_id': name, 'complete_readout_exact_match': True, 'frames': actual['frame_count'],
                    'elapsed_seconds': time.monotonic()-replay_start})
supervision = result['supervision']
assert supervision['exit_code'] == 1 and supervision['kill_reason'] is None and supervision['cleanup_error'] is None
assert supervision['elapsed_seconds'] < supervision['wall_cap_seconds'] == 900
assert supervision['sampled_peak_process_group_rss_bytes'] < supervision['rss_cap_bytes'] == 3*1024**3
assert watch['maximum_total_bytes'] < 1024**3 and watch['maximum_active_bytes'] < 256*1024**2
final_output_bytes = sum(path.stat().st_size for path in RAW.rglob('*') if path.is_file())
assert final_output_bytes < 1024**3
for path, digest in tuple(seen.items()):
    check(path, digest)
review = {
    'status': 'saved_failure_independently_reproduced',
    'release_sha256': RELEASE_SHA, 'new_solver_calls_by_auditor': 0, 'measured_curve_reads_by_auditor': 0,
    'recorded_solver_calls': 18, 'individual_run_checks_passed': 18, 'unexecuted_fitted_cases': fitted_not_executed,
    'aggregate_metrics': metric_count, 'failed_aggregate_metrics': failures,
    'mesh_and_step_metric_groups_independently_recomputed': 8,
    'scale_metrics_pass_in_saved_report': True, 'scale_raw_replay_performed': False,
    'selected_raw_replays': replays, 'numerical_reference_sha256': seen[str(RAW/'reference-numerical.json')],
    'no_calibration_or_heldout_access_attempt_recorded': True, 'access_ledger_fit_freeze_validation_absent': True,
    'physical_validation_pass': None, 'automatic_retry': False,
    'supervision': {key: supervision[key] for key in ('elapsed_seconds', 'wall_cap_seconds', 'exit_code', 'kill_reason',
                    'sampled_peak_process_group_rss_bytes', 'rss_cap_bytes', 'sampling_note')},
    'worker_elapsed_seconds': state['elapsed_seconds'], 'sum_nested_solver_seconds': sum(run_seconds),
    'maximum_single_solver_seconds': max(run_seconds), 'per_solver_cap_seconds': 90,
    'output_watch': watch, 'final_output_bytes': final_output_bytes,
    'unique_rehashed_inputs_and_outputs': len(seen), 'all_checked_files_unchanged': True,
    'original_skyline_failure_hashes_unchanged': legacy_hashes,
    'audit_source_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    'elapsed_seconds': time.monotonic()-start,
    'limits': ['Mesh-refinement rejection is numerical convergence evidence, not comparison with experimental measurements.',
               'No constitutive fit, held-out prediction or physical-validation accuracy is available.',
               'Solver elapsed times are nested within worker/parent timings and must not be added to them.',
               'RSS and output-watch peaks are sampled and may miss between-sample peaks.',
               'Only two raw run readouts were replayed; remaining individual and scale results are hash-bound saved results.'],
}
Path(__file__).with_name('review.json').write_text(json.dumps(review, indent=2) + '\n')
print(json.dumps(review, indent=2))
