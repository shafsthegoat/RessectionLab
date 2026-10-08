#!/usr/bin/env python3
"""One separately released N36 S120 diagnostic; no mesh generation or retry.

The 60-second pure deck preparation is nested in the 2400-second aggregate.
Historical readouts are authenticated accepted inputs, not replayed raw logs.
"""
from __future__ import annotations

import argparse
import math
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from scripts import mechanics_hbe_halfheight_global_n36_temporal as core
from scripts import mechanics_hbe_halfheight_global_n36_temporal_readout as readout
from scripts import mechanics_hbe_halfheight_global_n36_experiment as previous_runner

old = previous_runner.old
access, backend, runtime = old.access, old.backend, old.runtime
THREADS = previous_runner.THREADS
NEW_SOURCES = {'halfheight_global_n36_temporal': core,
               'halfheight_global_n36_temporal_readout': readout}
RUNNER_KEY = 'halfheight_global_n36_temporal_runner'


def remaining_seconds(deadline, cap, *, now=None):
    now = time.monotonic() if now is None else now
    if not all(math.isfinite(x) for x in (deadline, cap, now)) or cap <= 0:
        raise ValueError('Finite positive nested time allowance required')
    value = min(cap, deadline-now)
    if value <= 0:
        raise TimeoutError('Inclusive aggregate allowance exhausted')
    return value


def source_inventory(root, release, study):
    sources = release['source_bindings']
    if set(sources) != set(study['inherited_source_sha256']) | set(NEW_SOURCES) | {RUNNER_KEY}:
        raise ValueError('Exact twenty-three-module temporal source closure required')
    wanted = {}
    for key, record in sources.items():
        path = access.local_path(root, record['path'])
        if key in study['inherited_source_sha256']:
            baseline = study['baseline']['source_bindings'][key]
            if record['sha256'] != study['inherited_source_sha256'][key] or path.name != Path(baseline['path']).name:
                raise ValueError('Inherited source changed: '+key)
            module = sys.modules.get('scripts.'+path.stem)
            if module is None:
                raise ValueError('Inherited module missing from explicit import closure')
            actual = Path(module.__file__).resolve()
        else:
            actual = Path(__file__).resolve() if key == RUNNER_KEY else Path(NEW_SOURCES[key].__file__).resolve()
        if path != actual:
            raise ValueError('Imported temporal source origin differs: '+key)
        access.verify_binding(root, record, maximum_bytes=1024**2)
        wanted['scripts/'+path.name] = record['sha256']
    for record in list(study['original_declarations'].values())+[study['baseline']['declaration'], release['study']]:
        wanted[record['path']] = record['sha256']
    previous_runner.verify_archive(root, release['source_archive'], release['source_commit'], wanted, 30)


def accepted_baseline(root, study, context, bound):
    b = study['baseline']
    review, result, publication = (bound(b[key]) for key in ('independent_output_review', 'result', 'publication'))
    comparison, state, baseline, release, execution = (bound(b[key]) for key in
        ('comparison', 'state', 'execution_baseline', 'release', 'execution'))
    previous_runner.require_publication(publication, result_binding=b['result'], phase='solve', cap=1800,
                                       output_cap=2*1024**3)
    if (review.get('status') != 'saved_execution_and_declared_diagnostic_verified'
            or review.get('blocking_record_mismatches') != []
            or review.get('N36_complete_native_readout_canonical_equal') is not True
            or review.get('comparison_canonical_equal') is not True
            or any(review.get(key) != b[key] for key in ('result', 'publication', 'comparison', 'source_archive', 'source_commit'))
            or result.get('status') != 'completed_numerical_diagnostic_only'
            or result.get('state') != b['state'] or result.get('baseline') != b['execution_baseline']
            or result.get('measured_data_accessed') is not False
            or result.get('supervision', {}).get('status') != 'completed'
            or result['supervision'].get('exit_code') != 0
            or result['supervision'].get('kill_reason') is not None
            or result['supervision'].get('cleanup_error') is not None
            or state.get('status') != 'completed_numerical_diagnostic_only'
            or state.get('solver_invocations') != 1 or state.get('gmsh_generation_calls') != 0
            or state.get('comparison') != b['comparison']
            or set(state.get('runs', {})) != {b['run_id']}):
        raise ValueError('Independently accepted complete N36 S60 execution required')
    if (release.get('schema') != 'hbe-halfheight-global-n36-release-v1'
            or release.get('authorized') is not True or release.get('phase') != 'solve'
            or release.get('study') != b['declaration']
            or any(release.get(key) != b[key] for key in ('source_bindings', 'source_archive', 'source_commit',
                'interpreter', 'backend_profile', 'preparation', 'preparation_publication'))
            or baseline.get('release_binding') != b['release']
            or baseline.get('source_bindings') != b['source_bindings']
            or baseline.get('study_binding') != b['declaration']
            or baseline.get('runtime_identity') != context['runtime_identity']
            or context['runtime_identity'] != b['runtime_identity']):
        raise ValueError('Original S60 release/source/runtime origin differs')
    old_study = bound(b['declaration'])
    core.previous.require_study(old_study)
    wanted = {}
    if set(b['source_bindings']) != set(study['inherited_source_sha256']):
        raise ValueError('Exact twenty-module original N36 source closure required')
    for key, binding in b['source_bindings'].items():
        if binding['sha256'] != study['inherited_source_sha256'][key]:
            raise ValueError('Original accepted helper identity changed')
        bound(binding, json_value=False, maximum_bytes=1024**2)
        wanted['scripts/'+Path(binding['path']).name] = binding['sha256']
    for binding in list(study['original_declarations'].values())+[b['declaration']]:
        bound(binding)
        wanted[binding['path']] = binding['sha256']
    previous_runner.verify_archive(root, b['source_archive'], b['source_commit'], wanted, 26)
    bound(b['source_archive'], json_value=False)
    prep, prep_pub = bound(b['preparation']), bound(b['preparation_publication'])
    previous_runner.require_publication(prep_pub, result_binding=b['preparation'], phase='prepare', cap=60,
                                       output_cap=2*1024**3)
    if prep.get('status') != 'prepared_not_solved':
        raise ValueError('Accepted original N36 preparation required')
    row = state['runs'][b['run_id']]
    if (row.get('status') != 'passed_individual_numerical_checks' or row.get('readout') != b['readout']
            or row.get('input_bindings') != b['primitive_bindings']
            or execution.get('schema') != 'hbe-halfheight-global-n36-run-execution-v1'
            or execution.get('run_id') != b['run_id'] or execution.get('study') != b['declaration']
            or execution.get('primitive_bindings') != b['primitive_bindings']
            or execution.get('reconstruction') != b['reconstruction']
            or execution.get('runtime_identity') != b['runtime_identity']
            or execution.get('backend_profile') != b['backend_profile']
            or execution.get('execution') != row.get('execution')):
        raise ValueError('Exact complete S60 native primitive origin required')
    previous_runner.require_completed_native(execution['execution'], 1500)
    for key, binding in b['primitive_bindings'].items():
        bound(binding, json_value=False, maximum_bytes=previous_runner.readout.primitive_limit(key))
    bound(b['full_mesh'], json_value=False, maximum_bytes=32*1024**2)
    bound(b['reconstruction'], json_value=False, maximum_bytes=32*1024**2)
    bound(execution['backend_source_deck'], json_value=False, maximum_bytes=32*1024**2)
    backend.verify_deck(access.local_path(root, execution['backend_source_deck']['path']).read_bytes(),
                        access.local_path(root, b['primitive_bindings']['deck']['path']).read_bytes())
    rows = {int(N): bound(binding) for N, binding in b['prior_readouts'].items()}
    rows[36] = bound(b['readout'])
    if (rows[36].get('primitive_bindings') != b['primitive_bindings']
            or rows[36].get('execution_binding') != b['execution']
            or rows[36].get('reconstruction') != b['reconstruction']):
        raise ValueError('Accepted compact S60 readout lost native origin')
    # Pure comparison of previously verified compact readouts, never raw replay.
    actual = previous_runner.readout.comparison_report(
        {f'compression:N{N}:S60:reference': record for N, record in rows.items()}, old_study)
    if access.canonical_json(actual) != access.canonical_json(comparison):
        raise ValueError('Accepted S60 conditional comparison no longer reproduces')
    return rows, comparison


def preflight(root, study_binding, release_binding):
    root, inputs = Path(root).resolve(), {}
    def bound(record, *, json_value=True, maximum_bytes=16*1024**2):
        value = access.verify_binding(root, record, maximum_bytes=maximum_bytes, read_json=json_value)
        inputs[str(access.local_path(root, record['path']))] = record['sha256']
        return value
    study = core.declaration(root, study_binding)
    bound(study_binding)
    release = bound(release_binding)
    if (release.get('schema') != 'hbe-halfheight-global-n36-temporal-release-v1'
            or release.get('authorized') is not True or release.get('phase') != 'solve'
            or release.get('study') != study_binding):
        raise ValueError('Separate explicit temporal solve release required')
    source_inventory(root, release, study)
    for binding in release['source_bindings'].values():
        bound(binding, json_value=False, maximum_bytes=1024**2)
    bound(release['source_archive'], json_value=False)
    b = study['baseline']
    if release.get('interpreter') != b['interpreter'] or release.get('backend_profile') != b['backend_profile']:
        raise ValueError('Original interpreter and backend profile required')
    interpreter = Path(sys.executable).resolve()
    if interpreter != Path(b['interpreter']['path']).resolve() or runtime.sha(interpreter) != b['interpreter']['sha256']:
        raise ValueError('Executing interpreter differs from release')
    inputs[str(interpreter)] = b['interpreter']['sha256']
    context = backend.verify_profile(root, b['backend_profile'])
    inputs.update(context['inputs'])
    proposal, review = bound(study['reviewed_proposal']), bound(study['independent_design_review'])
    if (review.get('status') != 'no_blocking_numerical_design_defect_not_execution_release'
            or review.get('blocking_findings') != [] or review.get('proposal') != study['reviewed_proposal']
            or proposal.get('authorized') is not False or proposal.get('baseline') is None
            or review['checked_evidence']['comparison'] != b['comparison']
            or review['checked_evidence']['independent_output_review'] != b['independent_output_review']):
        raise ValueError('Exact independently reviewed S120 design required')
    for binding in study['preserved_failures'].values():
        bound(binding)
    accepted_baseline(root, study, context, bound)
    directory = root/study['output_root']/'experiment'
    return {'root': str(root), 'study': study, 'study_binding': study_binding, 'release_binding': release_binding,
        'source_bindings': release['source_bindings'], 'runtime_identity': context['runtime_identity'],
        'backend_profile': b['backend_profile'], 'executable': context['runtime']['executable'], 'inputs': inputs,
        'output_root': str(root/study['output_root']), 'directory': str(directory), 'phase': 'solve'}


def prepare_deck(root, plan):
    """Pure child, same outer supervised group; never import/call a mesher."""
    directory = Path(plan['directory'])/'preparation'
    protocol, extraction, _ = core.baseline_geometry(root, plan['study'])
    skyline, deck, loading = core.case_contents(root, plan['study'], protocol, extraction)
    for filename, content in (('skyline.feb', skyline), ('specimen.feb', deck)):
        with (directory/filename).open('x') as stream:
            stream.write(content)
    old.saved(root, directory/'loading.json', loading)
    case = {'deck': old.binding(root, directory/'specimen.feb'),
        'loading': old.binding(root, directory/'loading.json'),
        'backend_source_deck': old.binding(root, directory/'skyline.feb'),
        'mesh': plan['study']['baseline']['primitive_bindings']['mesh'],
        'reconstruction': plan['study']['baseline']['reconstruction']}
    old.recheck(plan['inputs'])
    previous_runner.durable_json(root, directory/'prepared.json',
        {'schema': 'hbe-n36-temporal-pure-deck-v1', 'study': plan['study_binding'],
         'run_id': core.RUN_ID, 'case': case, 'solver_calls': 0, 'gmsh_generation_calls': 0,
         'baseline_geometry_reused': True, 'measured_data_accessed': False})


def prepare_with_deadline(root, plan, baseline_binding, deadline, state, watch):
    started = time.monotonic()
    subdeadline = min(deadline, started+plan['study']['budgets']['pure_preparation_seconds'])
    directory = Path(plan['directory'])/'preparation'
    directory.mkdir(exist_ok=False)
    watch.active = directory
    command = [sys.executable, '-B', str(Path(__file__).resolve()), '--root', str(root),
        '--prepare-baseline', baseline_binding['path'], '--prepare-baseline-sha256', baseline_binding['sha256']]
    process = None
    record = {'schema': 'hbe-n36-temporal-nested-preparation-v1', 'command': command,
              'cap_seconds': 60, 'included_in_aggregate': True, 'status': 'starting',
              'solver_calls': 0, 'gmsh_generation_calls': 0}
    try:
        with (directory/'console.txt').open('x') as stream:
            # No new session: the outer 3 GiB process-group supervisor includes this child.
            process = subprocess.Popen(command, cwd=root, stdout=stream, stderr=subprocess.STDOUT)
            record['pid'] = process.pid
            code = process.wait(timeout=remaining_seconds(subdeadline, 60))
        record['exit_code'] = code
        if code != 0:
            raise RuntimeError('Pure preparation child failed')
        binding = old.binding(root, directory/'prepared.json')
        receipt = access.verify_binding(root, binding, read_json=True)
        if (receipt.get('schema') != 'hbe-n36-temporal-pure-deck-v1'
                or receipt.get('study') != plan['study_binding'] or receipt.get('run_id') != core.RUN_ID
                or receipt.get('solver_calls') != 0 or receipt.get('gmsh_generation_calls') != 0
                or receipt.get('baseline_geometry_reused') is not True
                or receipt.get('measured_data_accessed') is not False):
            raise ValueError('Complete source-bound pure-deck receipt required')
        case = receipt['case']
        if set(case) != {'deck', 'loading', 'backend_source_deck', 'mesh', 'reconstruction'}:
            raise ValueError('Exact temporal prepared case required')
        for key, filename in (('deck', 'specimen.feb'), ('loading', 'loading.json'), ('backend_source_deck', 'skyline.feb')):
            if access.local_path(root, case[key]['path']) != directory/filename:
                raise ValueError('Prepared deck outside exact new namespace')
            access.verify_binding(root, case[key], maximum_bytes=32*1024**2)
        b = plan['study']['baseline']
        if case['mesh'] != b['primitive_bindings']['mesh'] or case['reconstruction'] != b['reconstruction']:
            raise ValueError('Prepared case changed retained geometry')
        record.update(status='prepared_not_solved', receipt=binding)
        remaining_seconds(subdeadline, 60)
        state['case'] = case
    except BaseException as error:
        record.update(status='failed_or_incomplete', error={'type': type(error).__name__, 'message': str(error)})
        raise
    finally:
        if process is not None:
            if process.poll() is None:
                process.kill()
            process.wait()
        record['elapsed_seconds_before_publication'] = time.monotonic()-started
        state['preparation'] = previous_runner.durable_json(root, directory/'execution.json', record)
        watch.active = None
    # Include the preparation receipt publication in this nested allowance.
    remaining_seconds(subdeadline, 60)
    state['preparation_elapsed_seconds_including_publication'] = time.monotonic()-started
    return state['case']


def solve_one(root, plan, prepared, state, watch, deadline):
    directory = Path(plan['directory'])
    target = directory/'runs'/core.RUN_ID.replace(':', '-')
    target.mkdir(parents=True, exist_ok=False)
    watch.active = target
    row = state['run']
    try:
        for name, filename in (('deck', 'specimen.feb'), ('loading', 'loading.json')):
            with (target/filename).open('xb') as stream:
                stream.write(access.local_path(root, prepared[name]['path']).read_bytes())
            if runtime.sha(target/filename) != prepared[name]['sha256']:
                raise ValueError('Copied solver input differs')
        primitives = {'mesh': prepared['mesh'], 'deck': old.binding(root, target/'specimen.feb'),
                      'loading': old.binding(root, target/'loading.json')}
        old.recheck(plan['inputs'])
        seconds = remaining_seconds(deadline, plan['study']['budgets']['each_solver_seconds'])
        if state['solver_invocations'] != 0:
            raise ValueError('Exactly one native invocation permitted')
        command = [plan['executable'], '-noconfig', '-no_title', '-i', 'specimen.feb', '-o', 'solver.log']
        state['solver_invocations'] = 1
        row.update(status='solver_started', command=command)
        runtime.write_json(directory/'state.json', state)
        row['execution'] = old.solve(command, target, seconds)
        previous_runner.require_completed_native(row['execution'], seconds)
        for name, filename in (('nodes', 'nodes.log'), ('elements', 'elements.log'), ('solver', 'solver.log')):
            primitives[name] = old.binding(root, target/filename)
        execution = {'schema': 'hbe-n36-temporal-native-execution-v1', 'run_id': core.RUN_ID,
            'study': plan['study_binding'], 'runtime_identity': plan['runtime_identity'],
            'backend_profile': plan['backend_profile'], 'backend_source_deck': prepared['backend_source_deck'],
            'executable': old.binding(root, plan['executable']), 'primitive_bindings': primitives,
            'reconstruction': prepared['reconstruction'], 'command': command, 'execution': row['execution']}
        execution_binding = old.saved(root, target/'execution.json', execution)
        report = readout.read_temporal_run(root, half_bindings=primitives,
            reconstruction_binding=prepared['reconstruction'], declaration_binding=plan['study_binding'])
        report['execution_binding'] = execution_binding
        row['readout'] = old.saved(root, target/'readout.json', report)
        if report['passed'] is not True:
            raise ValueError('Individual native/reconstructed numerical gate failed')
        row.update(status='passed_individual_numerical_checks', execution_binding=execution_binding)
        b = plan['study']['baseline']
        baseline_rows = {int(N): access.verify_binding(root, binding, read_json=True)
                         for N, binding in b['prior_readouts'].items()}
        baseline_rows[36] = access.verify_binding(root, b['readout'], read_json=True)
        comparison = readout.comparison_report(baseline_rows, report, plan['study'],
                                               access.verify_binding(root, b['comparison'], read_json=True))
        state['comparison'] = old.saved(root, directory/'comparison.json', comparison)
        if comparison['original_temporal_checks_passed'] is not True:
            raise ValueError('Original temporal reaction/probe criterion failed; diagnostic retained')
        state['status'] = 'completed_numerical_diagnostic_only'
    except BaseException as error:
        row.update(status='failed', error={'type': type(error).__name__, 'message': str(error)})
        if (target/'solver-execution.json').is_file():
            row['partial_execution'] = old.binding(root, target/'solver-execution.json')
        raise
    finally:
        row['retained_files'] = {str(p.relative_to(target)): {'bytes': p.stat().st_size, 'sha256': runtime.sha(p)}
                                for p in sorted(target.rglob('*')) if p.is_file()}
        runtime.write_json(directory/'state.json', state)
        watch.active = None


def worker(root, plan, baseline_binding, deadline):
    directory, caps = Path(plan['directory']), plan['study']['budgets']
    remaining_seconds(deadline, caps['aggregate_specimen_seconds'])
    state = {'schema': 'hbe-n36-temporal-state-v1', 'status': 'running', 'solver_invocations': 0,
             'gmsh_generation_calls': 0, 'run_id': core.RUN_ID, 'run': {'status': 'not_executed'},
             'measured_data_accessed': False}
    runtime.write_json(directory/'state.json', state)
    try:
        with old.OutputWatch(plan['output_root'], directory/'output-watch.json', caps) as watch:
            prepared = prepare_with_deadline(root, plan, baseline_binding, deadline, state, watch)
            runtime.write_json(directory/'state.json', state)
            solve_one(root, plan, prepared, state, watch, deadline)
        old.recheck(plan['inputs'])
        remaining_seconds(deadline, caps['aggregate_specimen_seconds'])
    except BaseException as error:
        state.update(status='failed_or_incomplete', error={'type': type(error).__name__, 'message': str(error)})
        raise
    finally:
        runtime.write_json(directory/'state.json', state)


def launch(root, study_binding, release_binding):
    started = time.monotonic()
    plan = preflight(root, study_binding, release_binding)
    caps = plan['study']['budgets']
    cap = caps['aggregate_specimen_seconds']
    deadline = started+cap
    remaining_seconds(deadline, cap)
    directory, raw_root = Path(plan['directory']), Path(plan['output_root'])
    raw_root.mkdir(parents=True, exist_ok=True)
    old.saved(root, raw_root/'.solve-started.json', {'study': study_binding, 'release': release_binding, 'no_retry': True})
    directory.mkdir(exist_ok=False)
    baseline = old.saved(root, directory/'baseline.json', plan)
    command = [sys.executable, '-B', str(Path(__file__).resolve()), '--root', str(root),
        '--worker-baseline', baseline['path'], '--worker-baseline-sha256', baseline['sha256'],
        '--deadline', repr(deadline)]
    environment = runtime.private_environment({'caps': {'thread_environment': THREADS}})
    for key in list(environment):
        if key.startswith('PYTHON') or key in ('__PYVENV_LAUNCHER__', 'LD_PRELOAD', 'LD_LIBRARY_PATH'):
            environment.pop(key)
    environment.update(PYTHONNOUSERSITE='1', PYTHONDONTWRITEBYTECODE='1')
    supervision = runtime.supervise(command, directory/'supervision', cwd=root, environment=environment,
        seconds=remaining_seconds(deadline, cap), rss_bytes=caps['sampled_process_family_rss_bytes'])
    result = {'schema': 'hbe-n36-temporal-supervised-result-v1', 'phase': 'solve', 'baseline': baseline,
        'supervision': supervision, 'status': 'failed_or_incomplete', 'aggregate_cap_seconds': cap,
        'preparation_nested_in_aggregate': True, 'measured_data_accessed': False}
    try:
        old.recheck(plan['inputs'])
        result['state'] = old.binding(root, directory/'state.json')
        state = access.verify_binding(root, result['state'], read_json=True)
        if (supervision['status'] == 'completed' and state['status'] == 'completed_numerical_diagnostic_only'
                and state.get('solver_invocations') == 1 and state.get('gmsh_generation_calls') == 0
                and 0 <= state.get('preparation_elapsed_seconds_including_publication', math.inf) < 60):
            result['status'] = state['status']
    except BaseException as error:
        result['error'] = {'type': type(error).__name__, 'message': str(error)}
    return previous_runner.finalize_result(root, directory, raw_root, result, started, cap, caps['generated_output_bytes'])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=ROOT)
    parser.add_argument('--study', default=core.DECLARATION_PATH)
    parser.add_argument('--study-sha256', default=core.DECLARATION_SHA256)
    parser.add_argument('--release'); parser.add_argument('--release-sha256')
    parser.add_argument('--execute', action='store_true')
    parser.add_argument('--worker-baseline', help=argparse.SUPPRESS)
    parser.add_argument('--worker-baseline-sha256', help=argparse.SUPPRESS)
    parser.add_argument('--prepare-baseline', help=argparse.SUPPRESS)
    parser.add_argument('--prepare-baseline-sha256', help=argparse.SUPPRESS)
    parser.add_argument('--deadline', type=float, help=argparse.SUPPRESS)
    args = parser.parse_args()
    root = args.root.resolve()
    if args.worker_baseline or args.prepare_baseline:
        if args.worker_baseline and args.prepare_baseline:
            raise ValueError('Ambiguous private execution mode')
        binding = {'path': args.worker_baseline or args.prepare_baseline,
                   'sha256': args.worker_baseline_sha256 or args.prepare_baseline_sha256}
        plan = access.verify_binding(root, binding, read_json=True)
        fresh = preflight(root, plan['study_binding'], plan['release_binding'])
        if access.canonical_json(plan) != access.canonical_json(fresh):
            raise ValueError('Source/input-bound worker plan differs')
        if args.worker_baseline:
            if os.getpgrp() != os.getpid():
                raise ValueError('Worker must lead its supervised process group')
            worker(root, plan, binding, args.deadline)
        else:
            if os.getpgrp() == os.getpid():
                raise ValueError('Pure preparation must remain within outer supervised group')
            prepare_deck(root, plan)
        return
    study = {'path': args.study, 'sha256': args.study_sha256}
    release = {'path': args.release, 'sha256': args.release_sha256}
    if args.execute:
        if launch(root, study, release)['status'] == 'failed_or_incomplete':
            raise SystemExit(1)
    else:
        preflight(root, study, release)
        print('Metadata verified; no deck preparation or native solve executed.')


if __name__ == '__main__':
    main()
