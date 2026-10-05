"""Bounded saved-only terminal audit; partial N24 primitives are never parsed."""
from pathlib import Path
import hashlib
import json
import re
import sys
import time

ROOT = Path(__file__).resolve().parents[3]
OUT = Path(__file__).resolve().parent
RAW = ROOT/'outputs/mechanics/hbe-01-03-resolution-v1/experiment'
COLLECTION = ROOT/'artifacts/mechanics/hbe-resolution-experiment-result-v1'
FROZEN = ROOT/'outputs/validation/hbe-resolution-5b80e3f/frozen-source'
checked = {}


def sha(path):
    path = Path(path)
    assert path.suffix not in {'.csv', '.zip', '.nii'}
    h = hashlib.sha256()
    with path.open('rb') as stream:
        while block := stream.read(1024**2):
            h.update(block)
    return h.hexdigest()


def bound(path, expected=None):
    path = Path(path).resolve(); digest = sha(path)
    assert expected is None or digest == expected, str(path)
    checked[str(path)] = digest
    return {'path': str(path.relative_to(ROOT)) if path.is_relative_to(ROOT) else str(path), 'sha256': digest}


def read(binding):
    path = ROOT/binding['path']; bound(path, binding['sha256'])
    return json.loads(path.read_text())


def raw_index(index_path, raw, *, readonly=False):
    index = json.loads(index_path.read_text()); binding = bound(index_path)
    assert {str(p.relative_to(raw)) for p in raw.rglob('*') if p.is_file()} == set(index['files'])
    total = 0
    for name, row in index['files'].items():
        p = raw/name; assert p.stat().st_size == row['bytes']; bound(p, row['sha256']); total += row['bytes']
        if readonly:
            assert not p.stat().st_mode & 0o222
    if readonly:
        assert all(not p.stat().st_mode & 0o222 for p in [raw]+[p for p in raw.rglob('*') if p.is_dir()])
    assert index.get('bytes', total) == total
    return {'index': binding, 'files': len(index['files']), 'bytes': total, 'readonly_checked': readonly}


def main():
    started = time.monotonic()
    outcome_binding = bound(COLLECTION/'outcome.json'); outcome = read(outcome_binding)
    result = read(outcome['result']); state = read(result['state']); baseline = read(result['baseline'])
    release = read(baseline['release_binding']); study = read(baseline['study_binding'])
    assert release['phase'] == state['phase'] == result['phase'] == 'solve'
    assert release['authorized'] is True and release['schema'] == 'hbe-resolution-release-v1'
    assert release['source_commit'] == '5b80e3fee2349bc78d3f2ceb9b211efb305a1fc9'
    bound(ROOT/release['source_archive']['path'], '29f5866817a260e49275e51510a1937f432cc1bc40762009025748092e8eefdb')
    for path, digest in baseline['inputs'].items():
        bound(path, digest)
    for binding in baseline['source_bindings'].values():
        bound(ROOT/binding['path'], binding['sha256'])
    assert result['status'] == state['status'] == outcome['status'] == 'failed_or_incomplete'
    assert state['solver_invocations'] == outcome['solver_invocations'] == 2
    assert state['gmsh_generation_calls'] == 0 and state['error']['type'] == 'TimeoutExpired'
    complete_id = 'compression:N16:S60:reference'; partial_id = 'compression:N24:S60:reference'
    complete, partial = state['runs'][complete_id], state['runs'][partial_id]
    assert complete['status'] == 'passed_individual_numerical_checks' and partial['status'] == 'failed'
    assert all(state['runs'][key] == {'status': 'not_executed'} for key in study['ordered_runs'][2:])
    assert set(p.name for p in (RAW/'runs').iterdir()) == {complete_id.replace(':', '-'), partial_id.replace(':', '-')}
    assert 'readout' not in partial and not (RAW/'runs'/partial_id.replace(':','-')/'readout.json').exists()
    failed_execution = read(partial['partial_execution']); execution = complete['execution']
    assert failed_execution['timed_out'] is True and failed_execution['exit_code'] == -9
    assert failed_execution['cleanup'] == {'kill_sent': True, 'reaped': True}
    assert failed_execution['seconds_cap'] == execution['seconds_cap'] == 420
    assert failed_execution['elapsed_seconds'] >= 420 and execution['elapsed_seconds'] < 420
    assert execution['exit_code'] == 0 and execution['cleanup']['reaped'] is True
    assert execution['pid'] != failed_execution['pid']
    partial_log_path = RAW/'runs'/partial_id.replace(':','-')/'solver.log'
    # Banners only: no strict primitive or numerical readout on incomplete data.
    text = partial_log_path.read_text()
    begun = re.findall(r'beginning time step (\d+) : ([0-9.]+)', text)
    converged = re.findall(r'converged at time\s*:\s*([0-9.]+)', text)
    assert [int(v[0]) for v in begun] == list(range(1,38))
    assert len(converged) == 36 and converged[-1] == '0.6' and begun[-1] == ('37', '0.616667')
    assert 'N O R M A L   T E R M I N A T I O N' not in text
    assert outcome['aggregate_gates'] is None and outcome['physical_validation_pass'] is None
    assert 'comparison' not in state and not (RAW/'comparison.json').exists()
    assert state['measured_data_accessed'] is outcome['measured_data_accessed'] is False
    assert not any(p.name in {'access.jsonl','fit.json','predictions.json','freeze.json','heldout.json'} for p in RAW.rglob('*'))
    # One completed new case only. All reader imports originate in the sealed archive.
    sys.path.insert(0, str(FROZEN))
    from scripts import mechanics_hbe_readout as reader
    for name in ['mechanics_hbe_readout','mechanics_hbe_outputs','mechanics_hbe_physics','mechanics_hbe_access','mechanics_hbe_evaluation']:
        module = sys.modules['scripts.'+name]
        assert Path(module.__file__).resolve() == FROZEN/'scripts'/(name+'.py')
    saved = read(complete['readout']); actual_execution = read(saved['execution_binding'])
    assert actual_execution['execution'] == execution and actual_execution['primitive_bindings'] == saved['primitive_bindings']
    before = time.monotonic()
    replayed, cache = reader.read_resolution_run(ROOT, saved['primitive_bindings'],
        declaration_binding=baseline['study_binding'], expected_branch='compression', expected_mesh_N=16)
    replay_seconds = time.monotonic()-before
    expected = dict(saved); expected.pop('execution_binding')
    assert replayed == expected and cache is None and replayed['passed'] is True and replayed['frame_count'] == 61
    preservation = {
        'new_experiment': raw_index(COLLECTION/'raw-output-index.json', RAW, readonly=True),
        'prior_mesh_preparation': raw_index(ROOT/'artifacts/mechanics/hbe-resolution-mesh-result-v1/raw-output-index.json', RAW.parent/'mesh-preparation', readonly=True),
        'prior_failed_experiment': raw_index(ROOT/'artifacts/mechanics/hbe-accelerate-experiment-result-v1/raw-output-index.json',
            ROOT/'outputs/mechanics/hbe-01-03-poc-v1/experiment-accelerate-csc-v1', readonly=True)}
    assert [v['files'] for v in preservation.values()] == [22,28,169]
    supervision = result['supervision']; watch = outcome['output_watch']
    assert supervision['exit_code'] == 1 and supervision['kill_reason'] is None and supervision['cleanup_error'] is None
    assert supervision['elapsed_seconds'] < supervision['wall_cap_seconds'] == 1200
    assert supervision['sampled_peak_process_group_rss_bytes'] < supervision['rss_cap_bytes'] == 3*1024**3
    assert watch['maximum_active_bytes'] < 512*1024**2 and watch['maximum_total_bytes'] < 2*1024**3
    assert outcome['sum_recorded_solver_elapsed_seconds'] == execution['elapsed_seconds']+failed_execution['elapsed_seconds']
    for path, digest in checked.items():
        assert sha(path) == digest, path
    report = {'schema':'hbe-resolution-terminal-independent-review-v1','status':'saved_timeout_confirmed_no_retry',
        'outcome':outcome_binding,'result':outcome['result'],'baseline':result['baseline'],'state':result['state'],
        'release':baseline['release_binding'],'archive':release['source_archive'],
        'source_commit':release['source_commit'],'baseline_inputs_unchanged':len(baseline['inputs']),
        'checked_files_unchanged':len(checked),'preservation':preservation,
        'completed_case':{'id':complete_id,'individual_pass':True,'frames':61,'full_raw_replay_exact':True,
            'readout':complete['readout'],'replay_seconds':replay_seconds,'criteria':replayed['criteria']},
        'incomplete_case':{'id':partial_id,'execution':failed_execution,'last_log_confirmed_converged_step':36,
            'last_log_confirmed_pseudotime':.6,'last_log_begun_step':37,'last_log_begun_pseudotime':.616667,
            'primitive_replay_performed':False,'individual_pass':None,
            'progress_limit':'Saved banners only; later work or primitive bytes may not have flushed.'},
        'not_executed':study['ordered_runs'][2:],'aggregate_comparisons':None,'physical_validation_pass':None,
        'measured_response_access':False,'resource_evidence':{
            'parent_seconds':supervision['elapsed_seconds'],'worker_seconds':state['elapsed_seconds'],
            'completed_solver_seconds':execution['elapsed_seconds'],'timed_out_solver_seconds_including_cleanup':failed_execution['elapsed_seconds'],
            'cleanup_accounting_over_420s':failed_execution['elapsed_seconds']-420,
            'sampled_peak_group_rss_bytes':supervision['sampled_peak_process_group_rss_bytes'],
            'output_watch':watch,'scope':'Solver, worker and parent intervals are nested and must not be added. The per-case timeout stopped the case; aggregate/RSS/output caps did not trigger.'},
        'audit_seconds':time.monotonic()-started,
        'limits':['One completed N16 raw replay only; no N24 primitive parsing, solver, meshing, retry, curves or new tests.',
            'A completed individual case does not establish either co-primary mesh trend. Both comparisons remain unavailable.',
            'The original18-run failed aggregate outcome and all prior preparation raw files remain unchanged.',
            'Timeout is a resource termination; the saved evidence does not establish a numerical divergence diagnosis.',
            'Sampled memory/output peaks may miss between-sample peaks.'], 'blocking_report_mismatches':[]}
    (OUT/'review.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps({'review_sha256':sha(OUT/'review.json'),'audit_seconds':report['audit_seconds'],
                     'replay_seconds':replay_seconds,'files_checked':len(checked)}))


if __name__ == '__main__':
    main()
