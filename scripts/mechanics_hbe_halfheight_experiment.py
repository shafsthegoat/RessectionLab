#!/usr/bin/env python3
"""Bounded axial halfheight/full equivalence runner; preparation is not execution.

Two separately released phases reuse existing supervision, execution and output
guards; no response-data API is used. Preparation alone cannot launch a solver.
"""
from __future__ import annotations

import argparse
import hashlib
import json
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
from scripts import mechanics_hbe_halfheight_mesh as half_mesh
from scripts import mechanics_hbe_halfheight_readout as half_readout

access, backend, runtime = old.access, old.backend, old.runtime
DECLARATION_PATH = 'manifests/experiments/hbe-01-03-halfheight-equivalence-v1.json'
DECLARATION_SHA = '0df5587ab7a70eb1ac092919ff067b4d85ce6c7457af9726875beb58af19ad1e'
THREADS = {key: '1' for key in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS',
                              'VECLIB_MAXIMUM_THREADS', 'NUMEXPR_NUM_THREADS')}


def declaration(root, binding):
    if binding.get('sha256') != DECLARATION_SHA:
        raise ValueError('Half-height extraction contract is not frozen for execution')
    return access.verify_binding(root, binding, maximum_bytes=1024**2, read_json=True)


def source_inventory(root, release, study, halfheight_mesh, halfheight_readout):
    """Exact inherited nine modules plus these three additions and two specs."""
    sources = release['source_bindings']
    extra = {'halfheight_mesh': Path(halfheight_mesh.__file__).resolve(),
             'halfheight_readout': Path(halfheight_readout.__file__).resolve(),
             'halfheight_runner': Path(__file__).resolve()}
    inherited = {key: value for key, value in sources.items() if key not in extra}
    archive_binding = release['source_archive']
    old.source_inventory(root, inherited, archive_binding, archive_binding['sha256'])
    for key, path in extra.items():
        if access.local_path(root, sources[key]['path']) != path:
            raise ValueError('Half-height helper source origin differs')
        access.verify_binding(root, sources[key], maximum_bytes=1024**2)
    wanted = {f'scripts/{Path(value["path"]).name}': value['sha256'] for value in sources.values()}
    wanted.update({DECLARATION_PATH: DECLARATION_SHA,
                   study['original_protocol']['path']: study['original_protocol']['sha256']})
    seen = set()
    with tarfile.open(access.local_path(root, archive_binding['path']), 'r:*') as archive:
        if archive.pax_headers.get('comment') != release['source_commit']:
            raise ValueError('Half-height source archive commit differs')
        for member in archive:
            if member.isdir():
                continue
            if member.name not in wanted or member.name in seen or not member.isfile() or member.size > 1024**2:
                raise ValueError('Unexpected halfheight source archive member')
            stream = archive.extractfile(member)
            if stream is None or hashlib.sha256(stream.read()).hexdigest() != wanted[member.name]:
                raise ValueError('Half-height archived source or declaration differs')
            seen.add(member.name)
    if seen != set(wanted):
        raise ValueError('Incomplete halfheight source archive')


def full_references(root, study, context, bound):
    """Bind every saved full primitive before any halfheight extraction or solve."""
    if set(study['full_references']) != set(study['ordered_runs']):
        raise ValueError('Exact four full-model references required')
    result = {}
    for key in study['ordered_runs']:
        source = study['full_references'][key]
        reference = bound(source['readout'])
        execution = bound(source['execution'])
        branch, mesh_id, _, _ = key.split(':')
        if (reference.get('passed') is not True
                or (reference['branch'], reference['mesh_N'], reference['steps'],
                    reference['mu_Pa'], reference['frame_count']) != (branch, int(mesh_id[1:]), 60, 1000., 61)
                or reference['protocol_sha256'] != study['original_protocol']['sha256']
                or reference['primitive_bindings'] != source['primitive_bindings']
                or reference['execution_binding'] != source['execution']
                or execution['run_id'] != key or execution['primitive_bindings'] != source['primitive_bindings']
                or execution['runtime_identity'] != context['runtime_identity']
                or execution['backend_profile'] != study['original_backend_profile']):
            raise ValueError('Saved full-model reference identity differs')
        solved = execution['execution']
        if (solved.get('status') != 'completed' or type(solved.get('exit_code')) is not int
                or solved['exit_code'] != 0 or solved.get('timed_out') is not False
                or solved.get('cleanup', {}).get('reaped') is not True
                or not 0 <= solved['elapsed_seconds'] < 90):
            raise ValueError('Completed bounded full-model reference execution required')
        if set(source['primitive_bindings']) != {'mesh', 'deck', 'loading', 'nodes', 'elements', 'solver'}:
            raise ValueError('All six original full primitive bindings are required')
        for record in source['primitive_bindings'].values():
            bound(record, json_value=False)
        bound(execution['backend_source_deck'], json_value=False)
        backend.verify_deck(access.local_path(root, execution['backend_source_deck']['path']).read_bytes(),
                            access.local_path(root, source['primitive_bindings']['deck']['path']).read_bytes())
        result[key] = source
    return result


def preflight(root, study_binding, release_binding, phase):
    """Authenticate metadata and hashes without extracting geometry or solving."""
    inputs = {}
    def bound(record, *, json_value=True, maximum_bytes=256*1024**2):
        value = access.verify_binding(root, record, read_json=json_value, maximum_bytes=maximum_bytes)
        inputs[str(access.local_path(root, record['path']))] = record['sha256']
        return value
    if phase not in ('prepare', 'solve'):
        raise ValueError('Separate prepare or solve phase required')
    study = declaration(root, study_binding)
    bound(study_binding)
    release = bound(release_binding)
    if (release.get('schema') != 'hbe-halfheight-release-v1' or release.get('authorized') is not True
            or release.get('phase') != phase or release.get('study') != study_binding):
        raise ValueError('Separate explicit half-height phase release required')
    protocol = bound(study['original_protocol'])
    source_inventory(root, release, study, half_mesh, half_readout)
    for record in release['source_bindings'].values():
        bound(record, json_value=False, maximum_bytes=1024**2)
    bound(release['source_archive'], json_value=False)
    interpreter = Path(release['interpreter']['path']).resolve()
    if interpreter != Path(sys.executable).resolve() or runtime.sha(interpreter) != release['interpreter']['sha256']:
        raise ValueError('Released interpreter differs')
    inputs[str(interpreter)] = release['interpreter']['sha256']
    if release['backend_profile'] != study['original_backend_profile']:
        raise ValueError('Original accepted repaired backend profile required')
    context = backend.verify_profile(root, release['backend_profile'])
    if context['runtime_identity'] != study['runtime_identity']:
        raise ValueError('Original runtime identity required')
    inputs.update(context['inputs'])
    for record in list(study['preserved_failures'].values()) + [study['saved_geometry_eligibility'],
            study['abandoned_quarter_preparation'], study['original_source_archive_receipt']]:
        bound(record)
    references = full_references(root, study, context, bound)
    plan = {'study': study, 'study_binding': study_binding, 'release_binding': release_binding,
            'phase': phase, 'root': str(root), 'inputs': inputs, 'protocol': protocol,
            'source_bindings': release['source_bindings'], 'backend_profile': release['backend_profile'],
            'runtime_identity': context['runtime_identity'], 'executable': context['runtime']['executable'],
            'full_references': references, 'output_root': str(root/study['output_root']),
            'directory': str(root/study['output_root']/('preparation' if phase == 'prepare' else 'experiment'))}
    if phase == 'solve':
        accepted_preparation(root, plan, release, bound)
    return plan


def accepted_preparation(root, plan, release, bound):
    """Authenticate the prior phase; pure regenerated geometry is checked in worker."""
    study = plan['study']; directory = Path(plan['output_root'])/'preparation'
    result = bound(release['preparation'])
    state = bound(result['state']); baseline = bound(result['baseline'])
    if (access.local_path(root, release['preparation']['path']) != directory/'result.json'
            or access.local_path(root, result['state']['path']) != directory/'state.json'
            or access.local_path(root, result['baseline']['path']) != directory/'baseline.json'
            or result.get('phase') != 'prepare' or result.get('status') != 'prepared_not_solved'
            or baseline.get('phase') != 'prepare' or baseline.get('directory') != str(directory)
            or baseline.get('study_binding') != plan['study_binding']
            or baseline.get('source_bindings') != plan['source_bindings']
            or baseline.get('backend_profile') != plan['backend_profile']
            or baseline.get('runtime_identity') != plan['runtime_identity']
            or state.get('status') != 'prepared_not_solved'
            or state.get('solver_invocations') != 0 or state.get('gmsh_generation_calls') != 0
            or state.get('extraction_invocations') != 2 or state.get('measured_data_accessed') is not False
            or set(state.get('levels', {})) != {'8', '12'}
            or set(state.get('cases', {})) != set(study['ordered_runs'])):
        raise ValueError('Accepted exact half-height preparation phase required')
    supervision = result['supervision']; elapsed = supervision.get('elapsed_seconds', math.inf)
    if (supervision.get('status') != 'completed' or type(supervision.get('exit_code')) is not int
            or supervision['exit_code'] != 0 or supervision.get('kill_reason') is not None
            or supervision.get('cleanup_error') is not None
            or supervision.get('wall_cap_seconds') != study['budgets']['pure_preparation_seconds']
            or supervision.get('rss_cap_bytes') != study['budgets']['sampled_process_family_rss_bytes']
            or not math.isfinite(elapsed) or not 0 <= elapsed < supervision['wall_cap_seconds']):
        raise ValueError('Preparation exceeded or failed declared supervision')
    old.recheck(baseline['inputs']); plan['inputs'].update(baseline['inputs'])
    for N in (8, 12):
        level = state['levels'][str(N)]
        if set(level) != {'mesh', 'reconstruction'}:
            raise ValueError('Exact prepared mesh and reconstruction bindings required')
        for name, filename in [('mesh', 'mesh.json'), ('reconstruction', 'reconstruction.json')]:
            if access.local_path(root, level[name]['path']) != directory/f'N{N}'/filename:
                raise ValueError('Prepared level belongs to a different directory')
            bound(level[name])
    for key, row in state['cases'].items():
        branch, mesh_id, _, _ = key.split(':'); N = int(mesh_id[1:])
        if set(row) != {'mesh', 'reconstruction', 'deck', 'loading', 'backend_source_deck'}:
            raise ValueError('Exact prepared case bindings required')
        if any(row[name] != state['levels'][str(N)][name] for name in ('mesh', 'reconstruction')):
            raise ValueError('Prepared case level identity differs')
        for name, filename in [('deck', 'specimen.feb'), ('loading', 'loading.json'), ('backend_source_deck', 'skyline.feb')]:
            if access.local_path(root, row[name]['path']) != directory/'cases'/key.replace(':', '-')/filename:
                raise ValueError('Prepared case belongs to a different directory')
            bound(row[name], json_value=name == 'loading')
    plan.update(cases=state['cases'], levels=state['levels'])


def reconstruction(plan, extraction, full_binding, half_binding):
    return {'schema': 'hbe-halfheight-reconstruction-v1', 'mapping': extraction['mapping'],
            'verification': extraction['verification'], 'full_mesh': full_binding,
            'half_mesh': half_binding, 'declaration': plan['study_binding'],
            'mesh_source': plan['source_bindings']['halfheight_mesh']}


def case_contents(plan, extraction, branch, full_binding, level):
    xml, loading = half_mesh.halfheight_deck(extraction, branch, plan['protocol'])
    adapted = backend.transform_deck(xml.encode())
    loading.update(protocol_sha256=plan['study']['original_protocol']['sha256'],
                   halfheight_equivalence_declaration_sha256=DECLARATION_SHA,
                   mesh_sha256=level['mesh']['sha256'], full_mesh_sha256=full_binding['sha256'],
                   reconstruction_sha256=level['reconstruction']['sha256'],
                   deck_sha256=hashlib.sha256(adapted.encode()).hexdigest())
    return xml, adapted, loading


def prepare(root, plan, state, watch):
    directory = Path(plan['directory'])
    for N in (8, 12):
        target = directory/f'N{N}'; target.mkdir(exist_ok=False); watch.active = target
        state['extraction_invocations'] += 1
        runtime.write_json(directory/'state.json', state)
        full_binding = plan['full_references'][f'compression:N{N}:S60:reference']['primitive_bindings']['mesh']
        extraction = half_mesh.extract_halfheight(access.verify_binding(root, full_binding, read_json=True), plan['protocol'])
        half_binding = old.saved(root, target/'mesh.json', extraction['mesh'])
        wrapper = reconstruction(plan, extraction, full_binding, half_binding)
        level = {'mesh': half_binding, 'reconstruction': old.saved(root, target/'reconstruction.json', wrapper)}
        state['levels'][str(N)] = level
        for branch in ('compression', 'tension'):
            key = f'{branch}:N{N}:S60:reference'; case = directory/'cases'/key.replace(':', '-')
            case.mkdir(parents=True, exist_ok=False); watch.active = case
            xml, adapted, loading = case_contents(plan, extraction, branch, full_binding, level)
            (case/'skyline.feb').write_text(xml); (case/'specimen.feb').write_text(adapted)
            state['cases'][key] = dict(level, deck=old.binding(root, case/'specimen.feb'),
                backend_source_deck=old.binding(root, case/'skyline.feb'), loading=old.saved(root, case/'loading.json', loading))
        old.recheck(plan['inputs']); runtime.write_json(directory/'state.json', state)
    state['status'] = 'prepared_not_solved'; watch.active = None


def verify_prepared_geometry(root, plan):
    """Regenerate exact pure inputs inside solve supervision before any invocation."""
    for N in (8, 12):
        full_binding = plan['full_references'][f'compression:N{N}:S60:reference']['primitive_bindings']['mesh']
        extraction = half_mesh.extract_halfheight(access.verify_binding(root, full_binding, read_json=True), plan['protocol'])
        level = plan['levels'][str(N)]
        expected = {'mesh': extraction['mesh'], 'reconstruction': reconstruction(plan, extraction, full_binding, level['mesh'])}
        for name in expected:
            actual = access.verify_binding(root, level[name], read_json=True)
            if access.canonical_json(actual) != access.canonical_json(expected[name]):
                raise ValueError('Prepared geometry or reconstruction differs from exact original extraction')
        for branch in ('compression', 'tension'):
            key = f'{branch}:N{N}:S60:reference'; row = plan['cases'][key]
            xml, adapted, loading = case_contents(plan, extraction, branch, full_binding, level)
            for name, content in [('backend_source_deck', xml.encode()), ('deck', adapted.encode())]:
                access.verify_binding(root, row[name])
                if access.local_path(root, row[name]['path']).read_bytes() != content:
                    raise ValueError('Prepared half-height deck differs from exact declared boundary/loading')
            if access.canonical_json(access.verify_binding(root, row['loading'], read_json=True)) != access.canonical_json(loading):
                raise ValueError('Prepared half-height loading differs')


def solve_cases(root, plan, state, watch, deadline):
    directory = Path(plan['directory'])
    verify_prepared_geometry(root, plan)
    for key in plan['study']['ordered_runs']:
        row = state['runs'][key]; target = directory/'runs'/key.replace(':', '-')
        target.mkdir(parents=True, exist_ok=False); watch.active = target
        try:
            inputs = plan['cases'][key]
            for name, filename in [('deck', 'specimen.feb'), ('loading', 'loading.json')]:
                (target/filename).write_bytes(access.local_path(root, inputs[name]['path']).read_bytes())
                if runtime.sha(target/filename) != inputs[name]['sha256']:
                    raise ValueError('Copied solver input differs')
            primitives = {'mesh': inputs['mesh'], 'deck': old.binding(root, target/'specimen.feb'),
                          'loading': old.binding(root, target/'loading.json')}
            old.recheck(plan['inputs'])
            seconds = min(plan['study']['budgets']['each_solver_seconds'], deadline-time.monotonic())
            if seconds <= 0 or state['solver_invocations'] >= plan['study']['budgets']['maximum_specimen_solver_calls']:
                raise TimeoutError('Declared solve allocation exhausted')
            command = [plan['executable'], '-noconfig', '-no_title', '-i', 'specimen.feb', '-o', 'solver.log']
            row.update(status='solver_started', input_bindings=primitives, command=command)
            state['solver_invocations'] += 1; runtime.write_json(directory/'state.json', state)
            row['execution'] = old.solve(command, target, seconds)
            for name, filename in [('nodes', 'nodes.log'), ('elements', 'elements.log'), ('solver', 'solver.log')]:
                primitives[name] = old.binding(root, target/filename)
            execution = {'schema': 'hbe-halfheight-run-execution-v1', 'run_id': key, 'cwd': str(target.relative_to(root)),
                'study': plan['study_binding'], 'runtime_identity': plan['runtime_identity'],
                'backend_profile': plan['backend_profile'], 'backend_source_deck': inputs['backend_source_deck'],
                'executable': old.binding(root, plan['executable']), 'primitive_bindings': primitives,
                'reconstruction': inputs['reconstruction'], 'full_reference': plan['full_references'][key],
                'command': command, 'execution': row['execution']}
            execution_binding = old.saved(root, target/'execution.json', execution)
            branch, N, _, _ = key.split(':')
            report = half_readout.read_halfheight_equivalence(root, half_bindings=primitives,
                full_bindings=plan['full_references'][key]['primitive_bindings'], reconstruction_binding=inputs['reconstruction'],
                declaration_binding=plan['study_binding'], expected_branch=branch, expected_mesh_N=int(N[1:]))
            report['execution_binding'] = execution_binding
            row['readout'] = old.saved(root, target/'readout.json', report)
            if report['passed'] is not True:
                raise ValueError('Individual or half-height equivalence gate failed')
            row['status'] = 'passed_numerical_equivalence'
        except BaseException as error:
            row.update(status='failed', error={'type': type(error).__name__, 'message': str(error)})
            if (target/'solver-execution.json').exists():
                row['partial_execution'] = old.binding(root, target/'solver-execution.json')
            raise
        finally:
            row['retained_files'] = {str(p.relative_to(target)): {'bytes': p.stat().st_size, 'sha256': runtime.sha(p)}
                                     for p in sorted(target.rglob('*')) if p.is_file()}
            runtime.write_json(directory/'state.json', state); watch.active = None
    state['status'] = 'completed_numerical_equivalence_only'


def worker(root, plan):
    directory = Path(plan['directory']); started = time.monotonic(); phase = plan['phase']; caps = plan['study']['budgets']
    seconds = caps['pure_preparation_seconds' if phase == 'prepare' else 'aggregate_specimen_seconds']
    state = {'schema': 'hbe-halfheight-state-v1', 'phase': phase, 'status': 'running', 'solver_invocations': 0,
             'gmsh_generation_calls': 0, 'extraction_invocations': 0, 'cases': {}, 'levels': {},
             'measured_data_accessed': False, 'runs': {key: {'status': 'not_executed'} for key in plan['study']['ordered_runs']}}
    runtime.write_json(directory/'state.json', state)
    try:
        with old.OutputWatch(plan['output_root'], directory/'output-watch.json', caps) as watch:
            if phase == 'prepare':
                prepare(root, plan, state, watch)
            else:
                solve_cases(root, plan, state, watch, started+seconds)
        old.recheck(plan['inputs'])
        if time.monotonic()-started >= seconds:
            raise TimeoutError('Phase exceeded its aggregate allowance')
    except BaseException as error:
        state.update(status='failed_or_incomplete', error={'type': type(error).__name__, 'message': str(error)})
        raise
    finally:
        state['elapsed_seconds'] = time.monotonic()-started
        runtime.write_json(directory/'state.json', state)
    return state


def launch(root, study_binding, release_binding, phase):
    plan = preflight(root, study_binding, release_binding, phase)
    directory = Path(plan['directory']); raw_root = Path(plan['output_root']); raw_root.mkdir(parents=True, exist_ok=True)
    old.saved(root, raw_root/f'.{phase}-started.json', {'study': study_binding, 'release': release_binding, 'no_retry': True})
    directory.mkdir(exist_ok=False)
    baseline = old.saved(root, directory/'baseline.json', plan)
    command = [sys.executable, str(Path(__file__).resolve()), '--root', str(root),
               '--worker-baseline', baseline['path'], '--worker-baseline-sha256', baseline['sha256']]
    environment = runtime.private_environment({'caps': {'thread_environment': THREADS}})
    for key in list(environment):
        if key.startswith('PYTHON') or key in ('__PYVENV_LAUNCHER__', 'LD_PRELOAD', 'LD_LIBRARY_PATH'):
            environment.pop(key)
    environment.update(PYTHONNOUSERSITE='1', PYTHONDONTWRITEBYTECODE='1')
    seconds = plan['study']['budgets']['pure_preparation_seconds' if phase == 'prepare' else 'aggregate_specimen_seconds']
    supervision = runtime.supervise(command, directory/'supervision', cwd=root, environment=environment,
        seconds=seconds, rss_bytes=plan['study']['budgets']['sampled_process_family_rss_bytes'])
    result = {'schema': 'hbe-halfheight-supervised-result-v1', 'phase': phase, 'baseline': baseline,
              'supervision': supervision, 'status': 'failed_or_incomplete', 'measured_data_accessed': False}
    try:
        old.recheck(plan['inputs']); result['state'] = old.binding(root, directory/'state.json')
        state = access.verify_binding(root, result['state'], read_json=True)
        expected = 'prepared_not_solved' if phase == 'prepare' else 'completed_numerical_equivalence_only'
        if supervision['status'] == 'completed' and state['status'] == expected:
            result['status'] = expected
    except BaseException as error:
        result['error'] = {'type': type(error).__name__, 'message': str(error)}
    old.saved(root, directory/'result.json', result)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=ROOT); parser.add_argument('--phase', choices=('prepare', 'solve'))
    parser.add_argument('--study', default=DECLARATION_PATH); parser.add_argument('--study-sha256', default=DECLARATION_SHA)
    parser.add_argument('--release'); parser.add_argument('--release-sha256'); parser.add_argument('--execute', action='store_true')
    parser.add_argument('--worker-baseline', help=argparse.SUPPRESS)
    parser.add_argument('--worker-baseline-sha256', help=argparse.SUPPRESS)
    args = parser.parse_args(); root = args.root.resolve()
    if args.worker_baseline:
        if os.getpgrp() != os.getpid():
            raise ValueError('Worker must lead its supervised process group')
        plan = access.verify_binding(root, {'path': args.worker_baseline, 'sha256': args.worker_baseline_sha256}, read_json=True)
        fresh = preflight(root, plan['study_binding'], plan['release_binding'], plan['phase'])
        if access.canonical_json(plan) != access.canonical_json(fresh):
            raise ValueError('Bound worker plan changed')
        worker(root, plan)
        return
    study = {'path': args.study, 'sha256': args.study_sha256}; release = {'path': args.release, 'sha256': args.release_sha256}
    if args.execute:
        if launch(root, study, release, args.phase)['status'] == 'failed_or_incomplete':
            raise SystemExit(1)
    else:
        preflight(root, study, release, args.phase)
        print('Metadata verified; neither phase has executed.')


if __name__ == '__main__':
    main()
