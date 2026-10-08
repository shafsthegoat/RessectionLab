#!/usr/bin/env python3
"""Separately released pure preparation and three-solve boundary diagnostic.

No archive/release is generated automatically. No mesher or measured-response
access exists on this path. Existing process, output and native-call supervision
are reused; a failed attempt cannot be resumed or retried in this namespace.
"""
from __future__ import annotations

import argparse
import hashlib
import math
import os
from pathlib import Path
import sys
import subprocess
import tarfile
import time

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from scripts import mechanics_hbe_experiment as old
from scripts import mechanics_hbe_halfheight_boundary as core
from scripts import mechanics_hbe_halfheight_boundary_readout as readout

from scripts import mechanics_hbe_halfheight_spatial_experiment as prior_runner

access, backend, runtime = old.access, old.backend, old.runtime
THREADS = {key: '1' for key in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS',
                              'VECLIB_MAXIMUM_THREADS', 'NUMEXPR_NUM_THREADS')}


def source_inventory(root, release, study):
    """Fourteen unchanged modules plus three adapters; exact 22-file archive."""
    extras = {'halfheight_mesh': core.original, 'halfheight_readout': readout.inherited,
              'halfheight_spatial': core.spatial, 'halfheight_spatial_readout': prior_runner.readout,
              'halfheight_spatial_runner': prior_runner,
              'halfheight_boundary': core, 'halfheight_boundary_readout': readout}
    sources = release['source_bindings']
    if set(sources) != set(study['inherited_source_sha256']) | {
            'halfheight_boundary', 'halfheight_boundary_readout', 'halfheight_boundary_runner'}:
        raise ValueError('Exact seventeen-module source closure required')
    base = {k: v for k, v in sources.items() if k not in extras and k != 'halfheight_boundary_runner'}
    old.source_inventory(root, base, release['source_archive'], release['source_archive']['sha256'])
    for key, expected in study['inherited_source_sha256'].items():
        if sources[key]['sha256'] != expected:
            raise ValueError('Inherited scientific/runtime helper changed: '+key)
    paths = {k: Path(v.__file__).resolve() for k, v in extras.items()}
    paths['halfheight_boundary_runner'] = Path(__file__).resolve()
    for key, path in paths.items():
        if access.local_path(root, sources[key]['path']) != path:
            raise ValueError('Imported diagnostic source origin differs: '+key)
        access.verify_binding(root, sources[key], maximum_bytes=1024**2)
    wanted = {f'scripts/{Path(v["path"]).name}': v['sha256'] for v in sources.values()}
    wanted[core.DECLARATION_PATH] = core.DECLARATION_SHA256
    for name in ('original_protocol', 'original_resolution_declaration', 'original_halfheight_declaration',
                 'original_spatial_declaration'):
        wanted[study[name]['path']] = study[name]['sha256']
    verify_archive(root, release['source_archive'], release['source_commit'], wanted, 22)


def verify_archive(root, binding, commit, wanted, count):
    if len(wanted) != count:
        raise ValueError('Ambiguous committed source inventory')
    access.verify_binding(root, binding)
    seen = set()
    with tarfile.open(access.local_path(root, binding['path']), 'r:*') as archive:
        if archive.pax_headers.get('comment') != commit:
            raise ValueError('Exact committed Git archive required')
        for member in archive:
            if member.isdir():
                continue
            if member.name not in wanted or member.name in seen or not member.isfile() or member.size > 1024**2:
                raise ValueError('Unexpected/ambiguous archived source member')
            stream = archive.extractfile(member)
            if stream is None or hashlib.sha256(stream.read()).hexdigest() != wanted[member.name]:
                raise ValueError('Archived source/declaration identity differs')
            seen.add(member.name)
    if seen != set(wanted):
        raise ValueError('Incomplete committed source archive')


def require_completed_native(record, cap):
    if (record.get('status') != 'completed' or type(record.get('exit_code')) is not int
            or record['exit_code'] != 0 or record.get('timed_out') is not False
            or record.get('cleanup', {}).get('reaped') is not True
            or not math.isfinite(record.get('elapsed_seconds', math.inf))
            or not 0 <= record['elapsed_seconds'] < cap):
        raise ValueError('Complete bounded native execution required')


def prior_metadata(root, study, context, bound):
    """Authenticate P1's individual completion and retain aggregate failure."""
    prior = study['baseline_P1']
    release, result, review = (bound(prior[name]) for name in ('release', 'result', 'review'))
    bound(prior['comparison'])
    state, baseline = bound(result['state']), bound(result['baseline'])
    if (review.get('accepted_spatial_convergence') is not False
            or review.get('whole_record_mismatches') != 0 or review.get('comparison_reproduced') is not True
            or review.get('result_sha256') != prior['result']['sha256']
            or review.get('comparison_sha256') != prior['comparison']['sha256']
            or result.get('status') != 'failed_or_incomplete'
            or result.get('measured_data_accessed') is not False
            or result['supervision'].get('exit_code') != 1
            or result['supervision'].get('kill_reason') is not None
            or result['supervision'].get('cleanup_error') is not None
            or state.get('solver_invocations') != 3 or state.get('gmsh_generation_calls') != 0
            or state.get('measured_data_accessed') is not False
            or state['runs'][prior['run_id']]['status'] != 'passed_individual_numerical_checks'
            or state['runs'][prior['run_id']]['readout'] != prior['readout']):
        raise ValueError('Independently retained failed study with complete P1 required')
    if (release.get('authorized') is not True or release.get('phase') != 'solve'
            or release.get('study') != study['original_spatial_declaration']
            or release.get('source_archive') != prior['source_archive']
            or release.get('source_commit') != prior['source_commit']
            or release.get('source_bindings') != prior['source_bindings']
            or release.get('interpreter') != study['interpreter']
            or release.get('backend_profile') != study['original_backend_profile']
            or baseline.get('release_binding') != prior['release']
            or baseline.get('source_bindings') != prior['source_bindings']
            or baseline.get('runtime_identity') != context['runtime_identity']):
        raise ValueError('Original P1 source/execution release differs')
    prior_root = access.local_path(root, prior['source_root'])
    wanted = {}
    for key, record in prior['source_bindings'].items():
        if record['sha256'] != study['inherited_source_sha256'][key]:
            raise ValueError('Original source replay closure changed')
        member = f'scripts/{Path(record["path"]).name}'
        if access.local_path(root, record['path']) != prior_root/member:
            raise ValueError('P1 source is outside the accepted archived context')
        bound(record, json_value=False)
        wanted[member] = record['sha256']
    for name in ('original_protocol', 'original_resolution_declaration', 'original_halfheight_declaration',
                 'original_spatial_declaration'):
        record = study[name]
        wanted[record['path']] = record['sha256']
        archived = {'path': str((prior_root/record['path']).relative_to(root)), 'sha256': record['sha256']}
        bound(archived, json_value=False)
    bound(prior['source_archive'], json_value=False)
    verify_archive(root, prior['source_archive'], prior['source_commit'], wanted, 18)
    # This exact native record is eligible despite its parent's comparison failure.
    execution = bound(prior['execution'])
    if (execution.get('schema') != 'hbe-halfheight-spatial-run-execution-v1'
            or execution.get('run_id') != prior['run_id']
            or execution.get('study') != study['original_spatial_declaration']
            or execution.get('primitive_bindings') != prior['primitive_bindings']
            or execution.get('reconstruction') != prior['reconstruction']
            or execution.get('runtime_identity') != context['runtime_identity']
            or execution.get('backend_profile') != study['original_backend_profile']
            or execution.get('execution') != state['runs'][prior['run_id']]['execution']):
        raise ValueError('P1 native origin differs')
    require_completed_native(execution['execution'], 420)
    bound(prior['readout'], json_value=False)
    bound(prior['reconstruction'])
    if set(prior['primitive_bindings']) != readout.inherited.PRIMITIVE_KEYS:
        raise ValueError('Six complete P1 primitives required')
    for record in prior['primitive_bindings'].values():
        bound(record, json_value=False)
    loading = bound(prior['primitive_bindings']['loading'])
    if ((loading.get('branch'), loading.get('steps'), loading.get('mu_Pa')) != ('compression', 60, 1000.)
            or loading.get('protocol_sha256') != study['original_protocol']['sha256']
            or loading.get('mesh_sha256') != study['half_mesh']['sha256']
            or loading.get('deck_sha256') != prior['primitive_bindings']['deck']['sha256']):
        raise ValueError('P1 fixed loading identity differs')
    bound(execution['backend_source_deck'], json_value=False)
    backend.verify_deck(access.local_path(root, execution['backend_source_deck']['path']).read_bytes(),
                        access.local_path(root, prior['primitive_bindings']['deck']['path']).read_bytes())


def preflight(root, study_binding, release_binding, phase):
    root, inputs = Path(root).resolve(), {}
    def bound(record, *, json_value=True, maximum_bytes=256*1024**2):
        value = access.verify_binding(root, record, maximum_bytes=maximum_bytes, read_json=json_value)
        inputs[str(access.local_path(root, record['path']))] = record['sha256']
        return value
    if phase not in ('prepare', 'solve'):
        raise ValueError('Separate preparation/solve phase required')
    study = core.declaration(root, study_binding)
    bound(study_binding)
    release = bound(release_binding)
    if (release.get('schema') != 'hbe-halfheight-boundary-release-v1' or release.get('authorized') is not True
            or release.get('phase') != phase or release.get('study') != study_binding):
        raise ValueError('Separate explicit source-bound boundary phase release required')
    protocol = bound(study['original_protocol'])
    core.spatial.require_protocol(protocol)
    for name in ('original_resolution_declaration', 'original_halfheight_declaration', 'original_spatial_declaration'):
        bound(study[name])
    source_inventory(root, release, study)
    for record in release['source_bindings'].values():
        bound(record, json_value=False, maximum_bytes=1024**2)
    bound(release['source_archive'], json_value=False)
    if release.get('interpreter') != study['interpreter']:
        raise ValueError('Original bound interpreter required')
    interpreter = Path(release['interpreter']['path']).resolve()
    if interpreter != Path(sys.executable).resolve() or runtime.sha(interpreter) != release['interpreter']['sha256']:
        raise ValueError('Executing interpreter differs from release')
    inputs[str(interpreter)] = release['interpreter']['sha256']
    if release.get('backend_profile') != study['original_backend_profile']:
        raise ValueError('Original backend profile required')
    context = backend.verify_profile(root, release['backend_profile'])
    if context['runtime_identity'] != study['runtime_identity']:
        raise ValueError('Original accepted runtime identity required')
    inputs.update(context['inputs'])
    for record in study['preserved_failures'].values():
        bound(record)
    bound(study['full_mesh'], json_value=False)
    bound(study['half_mesh'], json_value=False)
    prior_metadata(root, study, context, bound)
    plan = {'study': study, 'study_binding': study_binding, 'release_binding': release_binding,
        'phase': phase, 'root': str(root), 'inputs': inputs, 'protocol': protocol,
        'source_bindings': release['source_bindings'], 'backend_profile': release['backend_profile'],
        'runtime_identity': context['runtime_identity'], 'executable': context['runtime']['executable'],
        'output_root': str(root/study['output_root']),
        'directory': str(root/study['output_root']/('preparation' if phase == 'prepare' else 'experiment'))}
    if phase == 'solve':
        accepted_preparation(root, plan, release, bound)
    return plan


def prepare(root, plan, state, watch):
    directory = Path(plan['directory'])
    for variant in core.ORDERED_VARIANTS:
        target = directory/variant
        target.mkdir(exist_ok=False)
        watch.active = target
        state['subdivision_invocations'] += 1
        runtime.write_json(directory/'state.json', state)
        extraction = core.load_variant(root, plan['study'], variant)
        level = {'full_mesh': old.saved(root, target/'full-mesh.json', extraction['full_mesh']),
                 'mesh': old.saved(root, target/'mesh.json', extraction['mesh'])}
        wrapper = core.reconstruction(plan['study_binding'], plan['source_bindings'], extraction, level, variant)
        level['reconstruction'] = old.saved(root, target/'reconstruction.json', wrapper)
        state['levels'][variant] = level
        key = f'compression:N24:{variant}:S60:reference'
        case = directory/'cases'/key.replace(':', '-')
        case.mkdir(parents=True, exist_ok=False)
        watch.active = case
        xml, adapted, loading = core.case_contents(plan['study'], plan['protocol'], extraction, variant, level)
        (case/'skyline.feb').write_text(xml)
        (case/'specimen.feb').write_text(adapted)
        state['cases'][key] = dict(level, deck=old.binding(root, case/'specimen.feb'),
            backend_source_deck=old.binding(root, case/'skyline.feb'), loading=old.saved(root, case/'loading.json', loading))
        old.recheck(plan['inputs'])
        runtime.write_json(directory/'state.json', state)
    state['status'] = 'prepared_not_solved'
    watch.active = None


def verify_prepared_geometry(root, plan):
    for key in core.ORDERED_RUNS:
        row = plan['cases'][key]
        core.verify_prepared_case(root, plan['study_binding'], row, row['reconstruction'], core.variant_of(key))
        backend.verify_deck(access.local_path(root, row['backend_source_deck']['path']).read_bytes(),
                            access.local_path(root, row['deck']['path']).read_bytes())


def accepted_preparation(root, plan, release, bound):
    directory = Path(plan['output_root'])/'preparation'
    result = bound(release['preparation'])
    state, baseline = bound(result['state']), bound(result['baseline'])
    if (access.local_path(root, release['preparation']['path']) != directory/'result.json'
            or access.local_path(root, result['state']['path']) != directory/'state.json'
            or access.local_path(root, result['baseline']['path']) != directory/'baseline.json'
            or result.get('schema') != 'hbe-halfheight-boundary-supervised-result-v1'
            or result.get('phase') != 'prepare' or result.get('status') != 'prepared_not_solved'
            or result.get('measured_data_accessed') is not False
            or baseline.get('phase') != 'prepare' or baseline.get('directory') != str(directory)
            or baseline.get('study_binding') != plan['study_binding']
            or baseline.get('source_bindings') != plan['source_bindings']
            or baseline.get('backend_profile') != plan['backend_profile']
            or baseline.get('runtime_identity') != plan['runtime_identity']
            or state.get('status') != 'prepared_not_solved' or state.get('phase') != 'prepare'
            or state.get('solver_invocations') != 0 or state.get('gmsh_generation_calls') != 0
            or state.get('subdivision_invocations') != 3 or state.get('measured_data_accessed') is not False
            or set(state.get('levels', {})) != set(core.ORDERED_VARIANTS)
            or set(state.get('cases', {})) != set(core.ORDERED_RUNS)):
        raise ValueError('Exact accepted boundary preparation required')
    fresh = preflight(root, baseline['study_binding'], baseline['release_binding'], 'prepare')
    if access.canonical_json(fresh) != access.canonical_json(baseline):
        raise ValueError('Preparation baseline differs from released source/input graph')
    supervision, caps = result['supervision'], plan['study']['budgets']
    elapsed, supervised_cap = result.get('phase_elapsed_seconds', math.inf), supervision.get('wall_cap_seconds', math.inf)
    if (supervision.get('status') != 'completed' or type(supervision.get('exit_code')) is not int
            or supervision['exit_code'] != 0 or supervision.get('kill_reason') is not None
            or supervision.get('cleanup_error') is not None
            or supervision.get('rss_cap_bytes') != caps['sampled_process_family_rss_bytes']
            or not math.isfinite(supervised_cap) or not 0 < supervised_cap <= caps['pure_preparation_seconds']
            or not math.isfinite(elapsed) or not 0 <= elapsed < caps['pure_preparation_seconds']):
        raise ValueError('Preparation exceeded aggregate supervision')
    old.recheck(baseline['inputs'])
    plan['inputs'].update(baseline['inputs'])
    for variant in core.ORDERED_VARIANTS:
        level = state['levels'][variant]
        if set(level) != {'full_mesh', 'mesh', 'reconstruction'}:
            raise ValueError('Exact prepared variant bindings required')
        for name, filename in (('full_mesh', 'full-mesh.json'), ('mesh', 'mesh.json'), ('reconstruction', 'reconstruction.json')):
            if access.local_path(root, level[name]['path']) != directory/variant/filename:
                raise ValueError('Prepared variant outside exact directory')
            bound(level[name])
    for key, row in state['cases'].items():
        variant = core.variant_of(key)
        if (set(row) != {'full_mesh', 'mesh', 'reconstruction', 'deck', 'loading', 'backend_source_deck'}
                or any(row[name] != state['levels'][variant][name] for name in ('full_mesh', 'mesh', 'reconstruction'))):
            raise ValueError('Prepared case/variant graph differs')
        for name, filename in (('deck', 'specimen.feb'), ('loading', 'loading.json'), ('backend_source_deck', 'skyline.feb')):
            if access.local_path(root, row[name]['path']) != directory/'cases'/key.replace(':', '-')/filename:
                raise ValueError('Prepared case outside exact directory')
            bound(row[name], json_value=name == 'loading')
    plan.update(levels=state['levels'], cases=state['cases'])


# The child imports the *original* source tree. Historical absolute module-origin
# checks remain intact. This is pinned runner code, not a generated response.
P1_REPLAY_CODE = '''
import sys
from pathlib import Path
sys.path.insert(0, sys.argv[1])
from scripts import mechanics_hbe_halfheight_spatial_readout as reader
a = reader.access
root = Path(sys.argv[2])
study = a.verify_binding(root, {'path':sys.argv[3], 'sha256':sys.argv[4]}, read_json=True)
prior = study['baseline_P1']
actual = reader.read_spatial_run(root, half_bindings=prior['primitive_bindings'],
    reconstruction_binding=prior['reconstruction'], declaration_binding=study['original_spatial_declaration'],
    expected_branch='compression', expected_mesh_N=24)
actual['execution_binding'] = prior['execution']
accepted = a.verify_binding(root, prior['readout'], read_json=True)
if actual['passed'] is not True or a.canonical_json(actual) != a.canonical_json(accepted):
    raise ValueError('Complete P1 replay differs from accepted receipt')
with Path(sys.argv[5]).open('xb') as stream:
    stream.write(a.canonical_json(actual))
'''


def replay_P1(root, plan, state, watch, deadline):
    directory = Path(plan['directory'])/'P1-replay'
    directory.mkdir(exist_ok=False)
    watch.active = directory
    prior = plan['study']['baseline_P1']
    command = [sys.executable, '-I', '-B', '-c', P1_REPLAY_CODE,
        str(access.local_path(root, prior['source_root'])), str(root), plan['study_binding']['path'],
        plan['study_binding']['sha256'], str(directory/'readout.json')]
    started, process = time.monotonic(), None
    receipt = {'schema': 'hbe-boundary-P1-replay-execution-v1', 'status': 'starting',
               'native_solver_calls': 0, 'original_source_commit': prior['source_commit'],
               'original_source_archive': prior['source_archive'], 'command': command,
               'elapsed_seconds': 0., 'exit_code': None, 'reaped': False}
    try:
        if deadline <= started:
            raise TimeoutError('Aggregate allowance exhausted before P1 replay')
        with (directory/'console.txt').open('x') as stream:
            process = subprocess.Popen(command, cwd=root, stdout=stream, stderr=subprocess.STDOUT)
            receipt['exit_code'] = process.wait(timeout=max(0., deadline-time.monotonic()))
        if receipt['exit_code'] != 0 or time.monotonic() >= deadline:
            raise RuntimeError('Original-context P1 replay failed or exceeded aggregate cap')
        actual = access.verify_binding(root, old.binding(root, directory/'readout.json'), read_json=True)
        accepted = access.verify_binding(root, prior['readout'], read_json=True)
        if access.canonical_json(actual) != access.canonical_json(accepted):
            raise ValueError('Returned P1 replay differs from accepted whole receipt')
        # Additional region diagnostics never mutate the accepted P1 receipt.
        actual['endpoint_regions'] = readout.baseline_endpoint(root, plan['study'])
        old.saved(root, directory/'endpoint-regions.json', actual['endpoint_regions'])
        receipt.update(status='completed_exact_original_context_replay', complete_raw_replay_exact=True)
        old.recheck(plan['inputs'])
        return actual
    except BaseException as error:
        receipt.update(status='failed_or_incomplete', error={'type': type(error).__name__, 'message': str(error)})
        raise
    finally:
        if process is not None:
            if process.poll() is None:
                process.kill()
            process.wait()
            receipt['reaped'] = True
        receipt['elapsed_seconds'] = time.monotonic()-started
        state['baseline_replay'] = old.saved(root, directory/'execution.json', receipt)
        runtime.write_json(Path(plan['directory'])/'state.json', state)
        watch.active = None


def solve_cases(root, plan, state, watch, deadline):
    directory = Path(plan['directory'])
    verify_prepared_geometry(root, plan)
    runs = {'P1': replay_P1(root, plan, state, watch, deadline)}
    for key in core.ORDERED_RUNS:
        variant, row = core.variant_of(key), state['runs'][key]
        target = directory/'runs'/key.replace(':', '-')
        target.mkdir(parents=True, exist_ok=False)
        watch.active = target
        try:
            prepared = plan['cases'][key]
            for name, filename in (('deck', 'specimen.feb'), ('loading', 'loading.json')):
                with (target/filename).open('xb') as stream:
                    stream.write(access.local_path(root, prepared[name]['path']).read_bytes())
                if runtime.sha(target/filename) != prepared[name]['sha256']:
                    raise ValueError('Copied solver input differs')
            primitives = {'mesh': prepared['mesh'], 'deck': old.binding(root, target/'specimen.feb'),
                          'loading': old.binding(root, target/'loading.json')}
            old.recheck(plan['inputs'])
            seconds = min(plan['study']['budgets']['each_solver_seconds'], deadline-time.monotonic())
            if seconds <= 0 or state['solver_invocations'] >= 3:
                raise TimeoutError('Declared native solve allocation exhausted')
            command = [plan['executable'], '-noconfig', '-no_title', '-i', 'specimen.feb', '-o', 'solver.log']
            row.update(status='solver_started', input_bindings=primitives, command=command)
            state['solver_invocations'] += 1
            runtime.write_json(directory/'state.json', state)
            row['execution'] = old.solve(command, target, seconds)
            require_completed_native(row['execution'], seconds)
            for name, filename in (('nodes', 'nodes.log'), ('elements', 'elements.log'), ('solver', 'solver.log')):
                primitives[name] = old.binding(root, target/filename)
            execution = {'schema': 'hbe-halfheight-boundary-run-execution-v1', 'run_id': key,
                'cwd': str(target.relative_to(root)), 'study': plan['study_binding'],
                'runtime_identity': plan['runtime_identity'], 'backend_profile': plan['backend_profile'],
                'backend_source_deck': prepared['backend_source_deck'], 'executable': old.binding(root, plan['executable']),
                'primitive_bindings': primitives, 'reconstruction': prepared['reconstruction'],
                'command': command, 'execution': row['execution']}
            execution_binding = old.saved(root, target/'execution.json', execution)
            report = readout.read_boundary_run(root, half_bindings=primitives,
                reconstruction_binding=prepared['reconstruction'], declaration_binding=plan['study_binding'],
                expected_variant=variant)
            report['execution_binding'] = execution_binding
            row['readout'] = old.saved(root, target/'readout.json', report)
            if report['passed'] is not True:
                raise ValueError('Individual native/reconstructed numerical gate failed')
            row['status'] = 'passed_individual_numerical_checks'
            runs[variant] = report
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
    report = readout.comparison_report(runs, plan['study'])
    state['comparison'] = old.saved(root, directory/'comparison.json', report)
    state['status'] = 'completed_numerical_diagnostic_only'


def worker(root, plan, deadline):
    directory = Path(plan['directory'])
    started = time.monotonic()
    phase, caps = plan['phase'], plan['study']['budgets']
    if not math.isfinite(deadline) or deadline <= started:
        raise TimeoutError('Supervised phase allowance exhausted during preflight')
    state = {'schema': 'hbe-halfheight-boundary-state-v1', 'phase': phase, 'status': 'running',
             'solver_invocations': 0, 'gmsh_generation_calls': 0, 'subdivision_invocations': 0,
             'cases': {}, 'levels': {}, 'reused_runs': {}, 'measured_data_accessed': False,
             'runs': {key: {'status': 'not_executed'} for key in core.ORDERED_RUNS}}
    runtime.write_json(directory/'state.json', state)
    try:
        with old.OutputWatch(plan['output_root'], directory/'output-watch.json', caps) as watch:
            if phase == 'prepare':
                prepare(root, plan, state, watch)
            else:
                solve_cases(root, plan, state, watch, deadline)
        old.recheck(plan['inputs'])
        if time.monotonic() >= deadline:
            raise TimeoutError('Phase exceeded its inclusive aggregate allowance')
    except BaseException as error:
        state.update(status='failed_or_incomplete', error={'type': type(error).__name__, 'message': str(error)})
        raise
    finally:
        state['elapsed_seconds'] = time.monotonic()-started
        runtime.write_json(directory/'state.json', state)
    return state



finalize_result = prior_runner.finalize_result


def launch(root, study_binding, release_binding, phase):
    started = time.monotonic()
    plan = preflight(root, study_binding, release_binding, phase)
    cap = plan['study']['budgets']['pure_preparation_seconds' if phase == 'prepare' else 'aggregate_specimen_seconds']
    deadline = started+cap
    if time.monotonic() >= deadline:
        raise TimeoutError('Phase allowance exhausted during launch preflight')
    directory, raw_root = Path(plan['directory']), Path(plan['output_root'])
    raw_root.mkdir(parents=True, exist_ok=True)
    old.saved(root, raw_root/f'.{phase}-started.json', {'study': study_binding, 'release': release_binding, 'no_retry': True})
    directory.mkdir(exist_ok=False)
    baseline = old.saved(root, directory/'baseline.json', plan)
    command = [sys.executable, str(Path(__file__).resolve()), '--root', str(root),
               '--worker-baseline', baseline['path'], '--worker-baseline-sha256', baseline['sha256'],
               '--phase-deadline', repr(deadline)]
    environment = runtime.private_environment({'caps': {'thread_environment': THREADS}})
    for key in list(environment):
        if key.startswith('PYTHON') or key in ('__PYVENV_LAUNCHER__', 'LD_PRELOAD', 'LD_LIBRARY_PATH'):
            environment.pop(key)
    environment.update(PYTHONNOUSERSITE='1', PYTHONDONTWRITEBYTECODE='1')
    seconds = deadline-time.monotonic()
    if seconds <= 0:
        raise TimeoutError('Phase allowance exhausted before supervision')
    supervision = runtime.supervise(command, directory/'supervision', cwd=root, environment=environment,
        seconds=seconds, rss_bytes=plan['study']['budgets']['sampled_process_family_rss_bytes'])
    result = {'schema': 'hbe-halfheight-boundary-supervised-result-v1', 'phase': phase,
              'baseline': baseline, 'supervision': supervision, 'status': 'failed_or_incomplete',
              'measured_data_accessed': False, 'aggregate_cap_seconds': cap}
    try:
        old.recheck(plan['inputs'])
        result['state'] = old.binding(root, directory/'state.json')
        state = access.verify_binding(root, result['state'], read_json=True)
        expected = 'prepared_not_solved' if phase == 'prepare' else 'completed_numerical_diagnostic_only'
        if supervision['status'] == 'completed' and state['status'] == expected:
            result['status'] = expected
    except BaseException as error:
        result['error'] = {'type': type(error).__name__, 'message': str(error)}
    return finalize_result(root, directory, raw_root, result, started, cap,
                           plan['study']['budgets']['generated_output_bytes'])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=ROOT)
    parser.add_argument('--phase', choices=('prepare', 'solve'))
    parser.add_argument('--study', default=core.DECLARATION_PATH)
    parser.add_argument('--study-sha256', default=core.DECLARATION_SHA256)
    parser.add_argument('--release'); parser.add_argument('--release-sha256')
    parser.add_argument('--execute', action='store_true')
    parser.add_argument('--worker-baseline', help=argparse.SUPPRESS)
    parser.add_argument('--worker-baseline-sha256', help=argparse.SUPPRESS)
    parser.add_argument('--phase-deadline', type=float, help=argparse.SUPPRESS)
    args = parser.parse_args()
    root = args.root.resolve()
    if args.worker_baseline:
        if os.getpgrp() != os.getpid():
            raise ValueError('Worker must lead its supervised process group')
        plan = access.verify_binding(root, {'path': args.worker_baseline, 'sha256': args.worker_baseline_sha256}, read_json=True)
        fresh = preflight(root, plan['study_binding'], plan['release_binding'], plan['phase'])
        if access.canonical_json(plan) != access.canonical_json(fresh):
            raise ValueError('Bound worker plan changed')
        worker(root, plan, args.phase_deadline)
        return
    study = {'path': args.study, 'sha256': args.study_sha256}
    release = {'path': args.release, 'sha256': args.release_sha256}
    if args.execute:
        if launch(root, study, release, args.phase)['status'] == 'failed_or_incomplete':
            raise SystemExit(1)
    else:
        preflight(root, study, release, args.phase)
        print('Metadata verified; neither preparation nor solver has executed.')


if __name__ == '__main__':
    main()
