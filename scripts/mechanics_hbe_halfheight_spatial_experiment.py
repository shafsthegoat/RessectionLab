#!/usr/bin/env python3
"""Separately released pure preparation and three-solve spatial experiment.

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
import tarfile
import time

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from scripts import mechanics_hbe_experiment as old
from scripts import mechanics_hbe_halfheight_spatial as core
from scripts import mechanics_hbe_halfheight_spatial_readout as readout

access, backend, runtime = old.access, old.backend, old.runtime
THREADS = {key: '1' for key in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS',
                              'VECLIB_MAXIMUM_THREADS', 'NUMEXPR_NUM_THREADS')}


def source_inventory(root, release, study):
    """Exact original eleven helpers plus three new modules and four specs."""
    extras = {'halfheight_mesh': Path(core.original.__file__).resolve(),
              'halfheight_readout': Path(readout.inherited.__file__).resolve(),
              'halfheight_spatial': Path(core.__file__).resolve(),
              'halfheight_spatial_readout': Path(readout.__file__).resolve(),
              'halfheight_spatial_runner': Path(__file__).resolve()}
    sources = release['source_bindings']
    inherited = {key: value for key, value in sources.items() if key not in extras}
    archive_binding = release['source_archive']
    old.source_inventory(root, inherited, archive_binding, archive_binding['sha256'])
    if set(sources) != set(study['inherited_source_sha256']) | {
            'halfheight_spatial', 'halfheight_spatial_readout', 'halfheight_spatial_runner'}:
        raise ValueError('Exact fourteen-module source closure required')
    for key, expected in study['inherited_source_sha256'].items():
        if sources[key]['sha256'] != expected:
            raise ValueError('Accepted scientific/runtime helper changed: '+key)
    for key, path in extras.items():
        if access.local_path(root, sources[key]['path']) != path:
            raise ValueError('Imported spatial/helper source origin differs: '+key)
        access.verify_binding(root, sources[key], maximum_bytes=1024**2)
    wanted = {f'scripts/{Path(value["path"]).name}': value['sha256'] for value in sources.values()}
    wanted[core.DECLARATION_PATH] = core.DECLARATION_SHA256
    for name in ('original_protocol', 'original_resolution_declaration', 'original_halfheight_declaration'):
        wanted[study[name]['path']] = study[name]['sha256']
    if len(wanted) != 18:
        raise ValueError('Ambiguous source/archive member names')
    seen = set()
    with tarfile.open(access.local_path(root, archive_binding['path']), 'r:*') as archive:
        if archive.pax_headers.get('comment') != release['source_commit']:
            raise ValueError('Exact committed Git archive required')
        for member in archive:
            if member.isdir():
                continue
            if member.name not in wanted or member.name in seen or not member.isfile() or member.size > 1024**2:
                raise ValueError('Unexpected spatial archive member')
            stream = archive.extractfile(member)
            if stream is None or hashlib.sha256(stream.read()).hexdigest() != wanted[member.name]:
                raise ValueError('Archived spatial source/declaration differs')
            seen.add(member.name)
    if seen != set(wanted):
        raise ValueError('Incomplete spatial source archive')


def prior_metadata(root, study, context, bound):
    """Hash saved responses without parsing them; bind complete execution metadata."""
    for key, source in study['prior_runs'].items():
        bound(source['readout'], json_value=False)
        execution = bound(source['execution'])
        branch, mesh_id, _, _ = key.split(':')
        N = int(mesh_id[1:])
        if (execution['run_id'] != key or execution['primitive_bindings'] != source['primitive_bindings']
                or execution['runtime_identity'] != context['runtime_identity']
                or execution['backend_profile'] != study['original_backend_profile']
                or set(source['primitive_bindings']) != readout.inherited.PRIMITIVE_KEYS):
            raise ValueError('Reused native execution identity differs')
        solved = execution['execution']
        cap = 420 if N == 16 else 90
        if (solved.get('status') != 'completed' or type(solved.get('exit_code')) is not int
                or solved['exit_code'] != 0 or solved.get('timed_out') is not False
                or solved.get('cleanup', {}).get('reaped') is not True
                or not math.isfinite(solved.get('elapsed_seconds', math.inf))
                or not 0 <= solved['elapsed_seconds'] < cap):
            raise ValueError('Only complete bounded old native runs may be reused')
        for record in source['primitive_bindings'].values():
            bound(record, json_value=False)
        loading = bound(source['primitive_bindings']['loading'])
        if ((loading.get('branch'), loading.get('steps'), loading.get('mu_Pa')) != (branch, 60, 1000.)
                or loading.get('protocol_sha256') != study['original_protocol']['sha256']
                or loading.get('mesh_sha256') != source['primitive_bindings']['mesh']['sha256']
                or loading.get('deck_sha256') != source['primitive_bindings']['deck']['sha256']):
            raise ValueError('Reused loading differs from fixed protocol/input identity')
        bound(execution['backend_source_deck'], json_value=False)
        backend.verify_deck(access.local_path(root, execution['backend_source_deck']['path']).read_bytes(),
                            access.local_path(root, source['primitive_bindings']['deck']['path']).read_bytes())


def accepted_evidence(study, bound):
    result = bound(study['accepted_halfheight']['result'])
    bound(study['accepted_halfheight']['review'], json_value=False)
    state = bound(result['state'])
    if (result.get('status') != 'completed_numerical_equivalence_only'
            or result.get('measured_data_accessed') is not False
            or result['supervision'].get('status') != 'completed'
            or result['supervision'].get('exit_code') != 0
            or state.get('status') != 'completed_numerical_equivalence_only'
            or state.get('solver_invocations') != 4 or state.get('gmsh_generation_calls') != 0
            or state.get('measured_data_accessed') is not False
            or any(row.get('status') != 'passed_numerical_equivalence' for row in state['runs'].values())):
        raise ValueError('Accepted original half-height result required')
    review = bound(study['accepted_N16_review'])
    complete = review['completed_case']
    if (complete.get('id') != 'compression:N16:S60:reference' or complete.get('frames') != 61
            or complete.get('full_raw_replay_exact') is not True or complete.get('individual_pass') is not True
            or complete.get('readout') != study['prior_runs'][complete['id']]['readout']
            or review.get('blocking_report_mismatches') != []):
        raise ValueError('Independent complete N16 replay acceptance required')
    quality = bound(study['accepted_mesh_preparation']['quality'])
    bound(study['accepted_mesh_preparation']['review'])
    for N in (16, 24):
        level = study['levels'][str(N)]
        q = quality['levels'][str(N)]
        receipt = bound(level['receipt'])
        bound(level['full_mesh'], json_value=False)
        if (q['receipt'] != level['receipt'] or q['mesh_sha256'] != level['full_mesh']['sha256']
                or q['quality']['passed'] is not True or receipt.get('status') != 'prepared_not_solved'
                or receipt.get('mesh_sha256') != level['full_mesh']['sha256']
                or receipt.get('protocol_sha256') != study['original_protocol']['sha256']
                or receipt.get('resolution_declaration_sha256') != study['original_resolution_declaration']['sha256']
                or receipt.get('quality') != q['quality']):
            raise ValueError('Saved finer geometry preparation acceptance differs')


def preflight(root, study_binding, release_binding, phase):
    """Authenticate inputs and saved backend controls; no new native solve."""
    root = Path(root).resolve()
    inputs = {}
    def bound(record, *, json_value=True, maximum_bytes=256*1024**2):
        value = access.verify_binding(root, record, maximum_bytes=maximum_bytes, read_json=json_value)
        inputs[str(access.local_path(root, record['path']))] = record['sha256']
        return value
    if phase not in ('prepare', 'solve'):
        raise ValueError('Separate preparation/solve phase required')
    study = core.declaration(root, study_binding)
    bound(study_binding)
    release = bound(release_binding)
    if (release.get('schema') != 'hbe-halfheight-spatial-release-v1' or release.get('authorized') is not True
            or release.get('phase') != phase or release.get('study') != study_binding):
        raise ValueError('Separate explicit source-bound spatial phase release required')
    protocol = bound(study['original_protocol'])
    core.require_protocol(protocol)
    for name in ('original_resolution_declaration', 'original_halfheight_declaration'):
        bound(study[name])
    source_inventory(root, release, study)
    for record in release['source_bindings'].values():
        bound(record, json_value=False, maximum_bytes=1024**2)
    bound(release['source_archive'], json_value=False)
    if release.get('interpreter') != study['interpreter']:
        raise ValueError('Original interpreter binding required')
    interpreter = Path(release['interpreter']['path']).resolve()
    if interpreter != Path(sys.executable).resolve() or runtime.sha(interpreter) != release['interpreter']['sha256']:
        raise ValueError('Executing interpreter differs from release')
    inputs[str(interpreter)] = release['interpreter']['sha256']
    if release.get('backend_profile') != study['original_backend_profile']:
        raise ValueError('Original repaired backend profile required')
    context = backend.verify_profile(root, release['backend_profile'])
    if context['runtime_identity'] != study['runtime_identity']:
        raise ValueError('Original accepted runtime identity required')
    inputs.update(context['inputs'])
    bound(study['original_halfheight_source']['archive'], json_value=False)
    for record in study['preserved_failures'].values():
        bound(record)
    accepted_evidence(study, bound)
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


def accepted_preparation(root, plan, release, bound):
    directory = Path(plan['output_root'])/'preparation'
    result = bound(release['preparation'])
    state, baseline = bound(result['state']), bound(result['baseline'])
    if (access.local_path(root, release['preparation']['path']) != directory/'result.json'
            or access.local_path(root, result['state']['path']) != directory/'state.json'
            or access.local_path(root, result['baseline']['path']) != directory/'baseline.json'
            or result.get('schema') != 'hbe-halfheight-spatial-supervised-result-v1'
            or result.get('phase') != 'prepare' or result.get('status') != 'prepared_not_solved'
            or result.get('measured_data_accessed') is not False
            or baseline.get('phase') != 'prepare' or baseline.get('directory') != str(directory)
            or baseline.get('study_binding') != plan['study_binding']
            or baseline.get('source_bindings') != plan['source_bindings']
            or baseline.get('backend_profile') != plan['backend_profile']
            or baseline.get('runtime_identity') != plan['runtime_identity']
            or state.get('status') != 'prepared_not_solved' or state.get('phase') != 'prepare'
            or state.get('solver_invocations') != 0 or state.get('gmsh_generation_calls') != 0
            or state.get('extraction_invocations') != 2 or state.get('measured_data_accessed') is not False
            or set(state.get('levels', {})) != {'16', '24'}
            or set(state.get('cases', {})) != set(core.ORDERED_RUNS)):
        raise ValueError('Exact accepted spatial preparation phase required')
    # Authenticate the actual preparation release/plan, not a self-described
    # subset of its inputs. This recursion terminates in phase=prepare.
    fresh = preflight(root, baseline['study_binding'], baseline['release_binding'], 'prepare')
    if access.canonical_json(fresh) != access.canonical_json(baseline):
        raise ValueError('Preparation baseline differs from its released source/input graph')
    supervision = result['supervision']
    caps = plan['study']['budgets']
    elapsed = result.get('phase_elapsed_seconds', math.inf)
    supervised_cap = supervision.get('wall_cap_seconds', math.inf)
    if (supervision.get('status') != 'completed' or type(supervision.get('exit_code')) is not int
            or supervision['exit_code'] != 0 or supervision.get('kill_reason') is not None
            or supervision.get('cleanup_error') is not None
            or supervision.get('rss_cap_bytes') != caps['sampled_process_family_rss_bytes']
            or not math.isfinite(supervised_cap) or not 0 < supervised_cap <= caps['pure_preparation_seconds']
            or not math.isfinite(elapsed) or not 0 <= elapsed < caps['pure_preparation_seconds']):
        raise ValueError('Preparation did not finish within its aggregate supervision')
    old.recheck(baseline['inputs'])
    plan['inputs'].update(baseline['inputs'])
    for N in (16, 24):
        level = state['levels'][str(N)]
        if set(level) != {'mesh', 'reconstruction'}:
            raise ValueError('Exact prepared level bindings required')
        for name, filename in (('mesh', 'mesh.json'), ('reconstruction', 'reconstruction.json')):
            if access.local_path(root, level[name]['path']) != directory/f'N{N}'/filename:
                raise ValueError('Prepared level outside its exact directory')
            bound(level[name])
    for key, row in state['cases'].items():
        N = int(key.split(':')[1][1:])
        if (set(row) != {'mesh', 'reconstruction', 'deck', 'loading', 'backend_source_deck'}
                or any(row[name] != state['levels'][str(N)][name] for name in ('mesh', 'reconstruction'))):
            raise ValueError('Prepared case/level binding graph differs')
        for name, filename in (('deck', 'specimen.feb'), ('loading', 'loading.json'),
                               ('backend_source_deck', 'skyline.feb')):
            if access.local_path(root, row[name]['path']) != directory/'cases'/key.replace(':', '-')/filename:
                raise ValueError('Prepared case outside its exact directory')
            bound(row[name], json_value=name == 'loading')
    plan.update(levels=state['levels'], cases=state['cases'])


def prepare(root, plan, state, watch):
    directory = Path(plan['directory'])
    for N in (16, 24):
        target = directory/f'N{N}'
        target.mkdir(exist_ok=False)
        watch.active = target
        state['extraction_invocations'] += 1
        runtime.write_json(directory/'state.json', state)
        extraction = core.load_extraction(root, plan['study'], N)
        full_binding = plan['study']['levels'][str(N)]['full_mesh']
        half_binding = old.saved(root, target/'mesh.json', extraction['mesh'])
        wrapper = core.reconstruction(plan['study_binding'], plan['source_bindings'], extraction, full_binding, half_binding)
        level = {'mesh': half_binding, 'reconstruction': old.saved(root, target/'reconstruction.json', wrapper)}
        state['levels'][str(N)] = level
        for key in core.ORDERED_RUNS:
            branch, mesh_id, _, _ = key.split(':')
            if int(mesh_id[1:]) != N:
                continue
            case = directory/'cases'/key.replace(':', '-')
            case.mkdir(parents=True, exist_ok=False)
            watch.active = case
            xml, adapted, loading = core.case_contents(plan['study'], plan['protocol'], extraction, branch, full_binding, level)
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
        branch, mesh_id, _, _ = key.split(':')
        row = plan['cases'][key]
        core.verify_prepared_case(root, plan['study_binding'], row, row['reconstruction'], branch, int(mesh_id[1:]))
        backend.verify_deck(access.local_path(root, row['backend_source_deck']['path']).read_bytes(),
                            access.local_path(root, row['deck']['path']).read_bytes())


def solve_cases(root, plan, state, watch, deadline):
    directory = Path(plan['directory'])
    verify_prepared_geometry(root, plan)
    runs = readout.replay_prior_runs(root, plan['study'])
    state['reused_runs'] = {key: {'readout': plan['study']['prior_runs'][key]['readout'],
                                 'complete_raw_replay_exact': True} for key in runs}
    runtime.write_json(directory/'state.json', state)
    for key in core.ORDERED_RUNS:
        row = state['runs'][key]
        target = directory/'runs'/key.replace(':', '-')
        target.mkdir(parents=True, exist_ok=False)
        watch.active = target
        try:
            inputs = plan['cases'][key]
            for name, filename in (('deck', 'specimen.feb'), ('loading', 'loading.json')):
                with (target/filename).open('xb') as stream:
                    stream.write(access.local_path(root, inputs[name]['path']).read_bytes())
                if runtime.sha(target/filename) != inputs[name]['sha256']:
                    raise ValueError('Copied solver input differs')
            primitives = {'mesh': inputs['mesh'], 'deck': old.binding(root, target/'specimen.feb'),
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
            for name, filename in (('nodes', 'nodes.log'), ('elements', 'elements.log'), ('solver', 'solver.log')):
                primitives[name] = old.binding(root, target/filename)
            execution = {'schema': 'hbe-halfheight-spatial-run-execution-v1', 'run_id': key,
                'cwd': str(target.relative_to(root)), 'study': plan['study_binding'],
                'runtime_identity': plan['runtime_identity'], 'backend_profile': plan['backend_profile'],
                'backend_source_deck': inputs['backend_source_deck'], 'executable': old.binding(root, plan['executable']),
                'primitive_bindings': primitives, 'reconstruction': inputs['reconstruction'],
                'command': command, 'execution': row['execution']}
            execution_binding = old.saved(root, target/'execution.json', execution)
            branch, mesh_id, _, _ = key.split(':')
            report = readout.read_spatial_run(root, half_bindings=primitives,
                reconstruction_binding=inputs['reconstruction'], declaration_binding=plan['study_binding'],
                expected_branch=branch, expected_mesh_N=int(mesh_id[1:]))
            report['execution_binding'] = execution_binding
            row['readout'] = old.saved(root, target/'readout.json', report)
            if report['passed'] is not True:
                raise ValueError('Individual native/reconstructed numerical gate failed')
            row['status'] = 'passed_individual_numerical_checks'
            runs[key] = report
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
    if report['passed'] is not True:
        raise ValueError('Both unchanged co-primary spatial comparisons must pass')
    state['status'] = 'completed_numerical_spatial_only'


def worker(root, plan, deadline):
    directory = Path(plan['directory'])
    started = time.monotonic()
    phase, caps = plan['phase'], plan['study']['budgets']
    if not math.isfinite(deadline) or deadline <= started:
        raise TimeoutError('Supervised phase allowance exhausted during preflight')
    state = {'schema': 'hbe-halfheight-spatial-state-v1', 'phase': phase, 'status': 'running',
             'solver_invocations': 0, 'gmsh_generation_calls': 0, 'extraction_invocations': 0,
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


def finalize_result(root, directory, raw_root, result, started, cap, output_cap):
    """Publish once after quiescent checks; count the receipt in retained bytes.

    Elapsed time ends at the final acceptance clock read, before durable receipt
    publication. Failed evidence is retained even when already above the cap.
    """
    result['retained_bytes_before_result'] = old.meshing.tree_bytes(raw_root)
    result['output_cap_bytes'] = output_cap
    elapsed = time.monotonic() - started
    result['phase_elapsed_seconds'] = elapsed
    result['timing_scope'] = 'launch_preflight_through_final_checks_and_tree_scan; excludes_result_publication'
    if not math.isfinite(elapsed) or not 0 <= elapsed < cap:
        result.update(status='failed_or_incomplete', final_cap_failure='inclusive_time_cap')
    if result['retained_bytes_before_result'] + len(access.canonical_json(result)) > output_cap:
        result.update(status='failed_or_incomplete', final_cap_failure='retained_output_cap_including_receipt')
    old.saved(root, directory/'result.json', result)
    return result


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
    result = {'schema': 'hbe-halfheight-spatial-supervised-result-v1', 'phase': phase,
              'baseline': baseline, 'supervision': supervision, 'status': 'failed_or_incomplete',
              'measured_data_accessed': False, 'aggregate_cap_seconds': cap}
    try:
        old.recheck(plan['inputs'])
        result['state'] = old.binding(root, directory/'state.json')
        state = access.verify_binding(root, result['state'], read_json=True)
        expected = 'prepared_not_solved' if phase == 'prepare' else 'completed_numerical_spatial_only'
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
