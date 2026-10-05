#!/usr/bin/env python3
"""Four fixed axial resolution runs; two separately released phases, no curve API.

Reuses the specimen generator, numerical readout, exact backend transform, and
existing process/output watchdogs. This is not a continuation or fit release.
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

access, backend, mesh, readout, runtime = old.access, old.backend, old.meshing, old.readout, old.runtime
STUDY_SHA = '23b4f5e4d8a0b5ff05fcd45fbca88463a32460793d5b0f27d1a913a5e0a2443c'
STUDY_PATH = 'manifests/experiments/hbe-01-03-axial-resolution-v1.json'
PROFILE_SHA = 'c9fafd50be2ed0c824b6ac15e8055b0f916d1331cc3b0e1e7e1d52a1ed1cbfc1'
THREADS = {key: '1' for key in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS',
                              'VECLIB_MAXIMUM_THREADS', 'NUMEXPR_NUM_THREADS')}


def _sources(root, release, study):
    sources = release['source_bindings']
    inherited = {key: value for key, value in sources.items() if key != 'resolution'}
    archive_binding = release['source_archive']
    old.source_inventory(root, inherited, archive_binding, archive_binding['sha256'])
    if access.local_path(root, sources['resolution']['path']) != Path(__file__).resolve():
        raise ValueError('Resolution source origin differs')
    wanted = {f'scripts/{Path(row["path"]).name}': row['sha256'] for row in sources.values()}
    wanted.update({STUDY_PATH: STUDY_SHA, study['original_protocol']['path']: study['original_protocol']['sha256']})
    seen = set()
    with tarfile.open(access.local_path(root, archive_binding['path']), 'r:*') as archive:
        if archive.pax_headers.get('comment') != release['source_commit']:
            raise ValueError('Exact committed Git archive required')
        for member in archive:
            if member.isdir():
                continue
            if member.name not in wanted or member.name in seen or not member.isfile() or member.size > 1024**2:
                raise ValueError('Unexpected source archive member')
            stream = archive.extractfile(member)
            if stream is None or hashlib.sha256(stream.read()).hexdigest() != wanted[member.name]:
                raise ValueError('Archived source or declaration differs')
            seen.add(member.name)
    if seen != set(wanted):
        raise ValueError('Incomplete exact source closure')


def preflight(root, study_binding, release_binding, phase):
    """Metadata/hash verification only; no meshing, equilibrium or response reads."""
    inputs = {}
    def bound(record, *, json_value=True, maximum_bytes=256*1024**2):
        value = access.verify_binding(root, record, read_json=json_value, maximum_bytes=maximum_bytes)
        inputs[str(access.local_path(root, record['path']))] = record['sha256']
        return value
    if study_binding['sha256'] != STUDY_SHA or phase not in ('prepare', 'solve'):
        raise ValueError('Only the fixed axial resolution study is supported')
    study = bound(study_binding)
    release = bound(release_binding)
    if (release.get('schema') != 'hbe-resolution-release-v1' or release.get('authorized') is not True
            or release.get('phase') != phase or release.get('study') != study_binding):
        raise ValueError('Separate explicit phase release required')
    protocol = bound(study['original_protocol'])
    _sources(root, release, study)
    for record in release['source_bindings'].values():
        bound(record, json_value=False, maximum_bytes=1024**2)
    bound(release['source_archive'], json_value=False)
    interpreter = Path(release['interpreter']['path']).resolve()
    if interpreter != Path(sys.executable).resolve() or runtime.sha(interpreter) != release['interpreter']['sha256']:
        raise ValueError('Released interpreter differs')
    inputs[str(interpreter)] = release['interpreter']['sha256']
    if release['backend_profile']['sha256'] != PROFILE_SHA:
        raise ValueError('Original accepted repaired backend profile required')
    context = backend.verify_profile(root, release['backend_profile'])
    inputs.update(context['inputs'])
    gmsh_binding = release['gmsh_runtime']
    gmsh = mesh.verify_runtime(access.local_path(root, gmsh_binding['path']), gmsh_binding['sha256'], protocol)
    bound(gmsh_binding)
    for name in ('module', 'library', 'license'):
        inputs[str(Path(gmsh[name]['path']).resolve())] = gmsh[name]['sha256']
    for record in study['prior_failure'].values():
        bound(record)
    bound(study['diagnosis'])
    prior_state = bound(study['prior_failure']['state'])
    prior = {}
    for key, records in study['prior_runs'].items():
        row, execution = bound(records['readout']), bound(records['execution'])
        branch, N, steps, role = key.split(':')
        if (prior_state['runs'][key]['readout'] != records['readout'] or row.get('passed') is not True
                or (row['branch'], row['mesh_N'], row['steps'], row['mu_Pa'], row['frame_count'])
                   != (branch, int(N[1:]), 60, 1000., 61)
                or row['protocol_sha256'] != study['original_protocol']['sha256']
                or execution['primitive_bindings'] != row['primitive_bindings']
                or execution['run_id'] != key or execution['runtime_identity'] != context['runtime_identity']
                or execution['backend_profile'] != release['backend_profile']
                or execution['execution']['exit_code'] != 0 or execution['execution']['timed_out'] is not False):
            raise ValueError('Historical comparison receipt differs from preserved run')
        for record in row['primitive_bindings'].values():
            bound(record, json_value=False)
        prior[key] = row
    plan = {'study': study, 'study_binding': study_binding, 'release_binding': release_binding,
            'phase': phase, 'root': str(root), 'inputs': inputs,
            'protocol': protocol, 'gmsh_runtime': gmsh_binding, 'source_bindings': release['source_bindings'],
            'backend_profile': release['backend_profile'], 'runtime_identity': context['runtime_identity'],
            'executable': context['runtime']['executable'], 'prior_runs': prior,
            'directory': str(root/study['output_root']/('mesh-preparation' if phase == 'prepare' else 'experiment')),
            'output_root': str(root/study['output_root'])}
    if phase == 'solve':
        prepared = bound(release['preparation'])
        if (prepared.get('phase') != 'prepare' or prepared.get('status') != 'prepared_not_solved'
                or prepared['supervision']['status'] != 'completed'):
            raise ValueError('Accepted separately supervised preparation required')
        preparation = bound(prepared['state'])
        baseline = bound(prepared['baseline'])
        if (baseline['study_binding'] != study_binding or baseline['source_bindings'] != release['source_bindings']
                or baseline['backend_profile'] != release['backend_profile']
                or preparation.get('status') != 'prepared_not_solved'
                or preparation.get('gmsh_generation_calls') != 2
                or set(preparation.get('cases', {})) != set(study['ordered_runs'])):
            raise ValueError('Preparation source, study or case inventory differs')
        prep_directory = root/study['output_root']/'mesh-preparation'
        if (access.local_path(root, release['preparation']['path']) != prep_directory/'result.json'
                or access.local_path(root, prepared['state']['path']) != prep_directory/'state.json'
                or access.local_path(root, prepared['baseline']['path']) != prep_directory/'baseline.json'
                or baseline.get('phase') != 'prepare' or baseline.get('directory') != str(prep_directory)
                or preparation.get('mesh_preparation_invocations') != 2
                or preparation.get('solver_invocations') != 0 or preparation.get('measured_data_accessed') is not False
                or set(preparation.get('levels', {})) != {'16', '24'}):
            raise ValueError('Preparation phase or level receipt inventory differs')
        supervision = prepared['supervision']
        elapsed = supervision.get('elapsed_seconds', math.inf)
        if (supervision.get('wall_cap_seconds') != study['budgets']['gmsh_aggregate_preparation_seconds']
                or supervision.get('rss_cap_bytes') != study['budgets']['sampled_process_family_rss_bytes']
                or type(supervision.get('exit_code')) is not int or supervision['exit_code'] != 0
                or supervision.get('kill_reason') is not None or supervision.get('cleanup_error') is not None
                or not math.isfinite(elapsed) or not 0 <= elapsed < supervision['wall_cap_seconds']):
            raise ValueError('Preparation did not finish within its declared supervision limits')
        old.recheck(baseline['inputs'])
        inputs.update(baseline['inputs'])
        levels = {}
        quality_checks = {'declared_cell_count', 'declared_node_count', 'positive_rest_jacobians',
                          'minimum_scaled_jacobian', 'finest_volume', 'finest_boundary_sag'}
        for expected in study['mesh_levels']:
            N = expected['N']; generated = prep_directory/f'N{N}'/'generated'
            record_binding = preparation['levels'][str(N)]
            if access.local_path(root, record_binding['path']) != generated/'receipt.json':
                raise ValueError('Prepared level receipt belongs to a different directory')
            level = bound(record_binding)
            quality = level.get('quality', {})
            limits = protocol['mesher']['mesh_quality']
            if (level.get('status') != 'prepared_not_solved' or level.get('mesh_N') != N
                    or level.get('source_sha256') != release['source_bindings']['mesh_deck']['sha256']
                    or level.get('protocol_sha256') != study['original_protocol']['sha256']
                    or level.get('resolution_declaration_sha256') != STUDY_SHA
                    or level.get('runtime_receipt_sha256') != gmsh_binding['sha256']
                    or level.get('gmsh_generation_calls') != 1 or level.get('solver_calls') != 0
                    or level.get('curve_values_opened') is not False or level.get('runtime_unchanged') is not True
                    or level.get('finalize_error') is not None or quality.get('passed') is not True
                    or set(quality.get('checks', {})) != quality_checks
                    or not all(value is True for value in quality['checks'].values())
                    or quality.get('node_count') != expected['expected_nodes']
                    or quality.get('hex8_cell_count') != expected['nominal_hex8_cells']
                    or quality.get('connected_cell_count') != expected['nominal_hex8_cells']
                    or set(level.get('decks', {})) != {'compression-60-reference', 'tension-60-reference'}):
                raise ValueError('Prepared level receipt identity or quality failed')
            for field, limit, greater in (
                ('minimum_rest_determinant_m3', 0., True),
                ('minimum_rest_scaled_jacobian', limits['minimum_rest_scaled_jacobian_exclusive'], True),
                ('relative_volume_error', limits['finest_relative_volume_error_max'], False),
                ('maximum_radial_boundary_sag_over_R', limits['finest_max_radial_sag_over_R'], False)):
                value = quality.get(field, math.nan)
                if not math.isfinite(value) or not (value > limit if greater else 0 <= value <= limit):
                    raise ValueError('Prepared level numeric quality criterion failed')
            for name, digest in (('mesh.json', level['mesh_sha256']), ('specimen.msh', level['native_mesh_sha256'])):
                bound({'path': str((generated/name).relative_to(root)), 'sha256': digest}, json_value=False)
            levels[N] = level
        for key, row in preparation['cases'].items():
            for record in row.values():
                bound(record, json_value=False)
            backend.verify_deck(access.local_path(root, row['backend_source_deck']['path']).read_bytes(),
                                access.local_path(root, row['deck']['path']).read_bytes())
            branch = key.split(':')[0]
            N = int(key.split(':')[1][1:])
            metadata = bound(row['mesh'])
            loading = bound(row['loading'])
            expected = next(v for v in study['mesh_levels'] if v['N'] == N)
            generated = prep_directory/f'N{N}'/'generated'
            original_directory = generated/f'{branch}-60-reference'
            original = levels[N]['decks'][f'{branch}-60-reference']
            source_deck = {'path': str((original_directory/'specimen.feb').relative_to(root)), 'sha256': original['deck_sha256']}
            original_loading = bound({'path': str((original_directory/'loading.json').relative_to(root)),
                                      'sha256': original['loading_sha256']})
            if (row['backend_source_deck'] != source_deck
                    or row['mesh'] != {'path': str((generated/'mesh.json').relative_to(root)), 'sha256': levels[N]['mesh_sha256']}
                    or loading != dict(original_loading, deck_sha256=row['deck']['sha256'])):
                raise ValueError('Prepared case differs from its generated level and original loading')
            if ((loading.get('branch'), loading.get('steps'), loading.get('mu_Pa')) != (branch, 60, 1000.)
                    or loading.get('protocol_sha256') != study['original_protocol']['sha256']
                    or metadata['mesh_N'] != N or len(metadata['rest_nodes_m']) != expected['expected_nodes']
                    or len(metadata['elements_hex8']) != expected['nominal_hex8_cells']
                    or loading['mesh_sha256'] != row['mesh']['sha256']
                    or loading['deck_sha256'] != row['deck']['sha256']
                    or loading['resolution_declaration_sha256'] != STUDY_SHA):
                raise ValueError('Prepared mesh/loading identity differs')
        plan['cases'] = preparation['cases']
    return plan


def prepare(root, plan, state, watch):
    directory = Path(plan['directory'])
    for level in plan['study']['mesh_levels']:
        N = level['N']; generated = directory/f'N{N}'/'generated'
        watch.active = generated.parent
        state['mesh_preparation_invocations'] += 1
        runtime.write_json(directory/'state.json', state)
        try:
            record = mesh.prepare_resolution_level(N,
                access.local_path(root, plan['study']['original_protocol']['path']),
                access.local_path(root, plan['study_binding']['path']), STUDY_SHA,
                access.local_path(root, plan['gmsh_runtime']['path']), plan['gmsh_runtime']['sha256'], generated)
        finally:
            if (generated/'receipt.json').is_file():
                actual = json.loads((generated/'receipt.json').read_text())
                state['gmsh_generation_calls'] += actual['gmsh_generation_calls']
                state['levels'][str(N)] = old.binding(root, generated/'receipt.json')
                runtime.write_json(directory/'state.json', state)
        if record['status'] != 'prepared_not_solved' or record['quality']['passed'] is not True:
            raise ValueError('Mesh preparation failed')
        for branch in plan['study']['branches']:
            key = f'{branch}:N{N}:S60:reference'
            original = generated/f'{branch}-60-reference'
            target = directory/'cases'/key.replace(':', '-')
            target.mkdir(parents=True, exist_ok=False)
            xml = backend.transform_deck((original/'specimen.feb').read_bytes())
            (target/'specimen.feb').write_text(xml)
            loading = json.loads((original/'loading.json').read_text())
            loading['deck_sha256'] = runtime.sha(target/'specimen.feb')
            old.saved(root, target/'loading.json', loading)
            state['cases'][key] = {'mesh': old.binding(root, generated/'mesh.json'),
                'deck': old.binding(root, target/'specimen.feb'), 'loading': old.binding(root, target/'loading.json'),
                'backend_source_deck': old.binding(root, original/'specimen.feb')}
        old.recheck(plan['inputs'])
        runtime.write_json(directory/'state.json', state)
    state['status'] = 'prepared_not_solved'


def comparison_report(runs, study):
    """All co-primary groups are mandatory; reuse unchanged comparison math."""
    groups = {}
    for triplet in study['co_primary_triplets']:
        for branch in study['branches']:
            rows = [runs[f'{branch}:N{N}:S60:reference'] for N in triplet]
            groups[f'{branch}:N{triplet[0]}-N{triplet[1]}-N{triplet[2]}'] = readout._refinement_group(*rows, kind='mesh')
    passed = all((v['actual'] < v['limit'] if v.get('comparison') == 'lt' else v['actual'] <= v['limit'])
                 for group in groups.values() for v in group.values())
    return {'schema': 'hbe-axial-resolution-comparison-v1', 'study_sha256': STUDY_SHA,
            'co_primary_triplets': study['co_primary_triplets'], 'groups': groups, 'passed': passed,
            'original_failure': study['prior_failure'], 'physical_validation_pass': None,
            'measured_data_accessed': False}


def solve_cases(root, plan, state, watch, deadline):
    directory = Path(plan['directory']); runs = dict(plan['prior_runs'])
    for key in plan['study']['ordered_runs']:
        row = state['runs'][key]; target = directory/'runs'/key.replace(':', '-')
        target.mkdir(parents=True, exist_ok=False); watch.active = target
        try:
            inputs = plan['cases'][key]
            for name, filename in (('deck', 'specimen.feb'), ('loading', 'loading.json')):
                (target/filename).write_bytes(access.local_path(root, inputs[name]['path']).read_bytes())
                if runtime.sha(target/filename) != inputs[name]['sha256']:
                    raise ValueError('Copied input differs')
            primitives = {name: inputs[name] for name in ('mesh',)}
            primitives.update(deck=old.binding(root, target/'specimen.feb'), loading=old.binding(root, target/'loading.json'))
            old.recheck(plan['inputs'])
            seconds = min(plan['study']['budgets']['each_solver_seconds'], deadline-time.monotonic())
            if seconds <= 0 or state['solver_invocations'] >= 4:
                raise TimeoutError('Declared solve allocation exhausted')
            command = [plan['executable'], '-noconfig', '-no_title', '-i', 'specimen.feb', '-o', 'solver.log']
            row.update(status='solver_started', input_bindings=primitives, command=command)
            state['solver_invocations'] += 1
            runtime.write_json(directory/'state.json', state)
            row['execution'] = old.solve(command, target, seconds)
            for name, filename in (('nodes', 'nodes.log'), ('elements', 'elements.log'), ('solver', 'solver.log')):
                primitives[name] = old.binding(root, target/filename)
            execution = {'schema': 'hbe-resolution-run-execution-v1', 'run_id': key, 'cwd': str(target.relative_to(root)),
                'study': plan['study_binding'], 'runtime_identity': plan['runtime_identity'],
                'backend_profile': plan['backend_profile'], 'backend_source_deck': inputs['backend_source_deck'],
                'executable': old.binding(root, plan['executable']), 'primitive_bindings': primitives,
                'command': command, 'execution': row['execution']}
            execution_binding = old.saved(root, target/'execution.json', execution)
            branch, N, _, _ = key.split(':')
            report, _ = readout.read_resolution_run(root, primitives, declaration_binding=plan['study_binding'],
                expected_branch=branch, expected_mesh_N=int(N[1:]))
            report['execution_binding'] = execution_binding
            row['readout'] = old.saved(root, target/'readout.json', report)
            if report['passed'] is not True:
                raise ValueError('Individual numerical gate failed')
            row['status'] = 'passed_individual_numerical_checks'; runs[key] = report
        except BaseException as error:
            row.update(status='failed', error={'type': type(error).__name__, 'message': str(error)})
            if (target/'solver-execution.json').exists():
                row['partial_execution'] = old.binding(root, target/'solver-execution.json')
            raise
        finally:
            row['retained_files'] = {str(p.relative_to(target)): {'bytes': p.stat().st_size, 'sha256': runtime.sha(p)}
                                     for p in sorted(target.rglob('*')) if p.is_file()}
            runtime.write_json(directory/'state.json', state); watch.active = None
    report = comparison_report(runs, plan['study'])
    state['comparison'] = old.saved(root, directory/'comparison.json', report)
    if report['passed'] is not True:
        raise ValueError('Co-primary mesh resolution gate failed')
    state['status'] = 'completed_numerical_resolution_only'


def worker(root, plan):
    directory = Path(plan['directory']); started = time.monotonic()
    caps = plan['study']['budgets']; phase = plan['phase']
    seconds = caps['gmsh_aggregate_preparation_seconds'] if phase == 'prepare' else caps['aggregate_specimen_seconds']
    state = {'schema': 'hbe-axial-resolution-state-v1', 'phase': phase, 'status': 'running',
             'solver_invocations': 0, 'gmsh_generation_calls': 0, 'mesh_preparation_invocations': 0, 'cases': {}, 'levels': {},
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
    directory = Path(plan['directory']); raw_root = Path(plan['output_root'])
    raw_root.mkdir(parents=True, exist_ok=True)
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
    cap = plan['study']['budgets']['gmsh_aggregate_preparation_seconds' if phase == 'prepare' else 'aggregate_specimen_seconds']
    supervision = runtime.supervise(command, directory/'supervision', cwd=root, environment=environment,
        seconds=cap, rss_bytes=plan['study']['budgets']['sampled_process_family_rss_bytes'])
    result = {'schema': 'hbe-resolution-supervised-result-v1', 'phase': phase, 'baseline': baseline,
              'supervision': supervision, 'status': 'failed_or_incomplete', 'measured_data_accessed': False}
    try:
        old.recheck(plan['inputs'])
        result['state'] = old.binding(root, directory/'state.json')
        state = access.verify_binding(root, result['state'], read_json=True)
        expected = 'prepared_not_solved' if phase == 'prepare' else 'completed_numerical_resolution_only'
        if supervision['status'] == 'completed' and state['status'] == expected:
            result['status'] = expected
    except BaseException as error:
        result['error'] = {'type': type(error).__name__, 'message': str(error)}
    old.saved(root, directory/'result.json', result)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=ROOT)
    parser.add_argument('--phase', choices=('prepare', 'solve'))
    parser.add_argument('--study', default=STUDY_PATH); parser.add_argument('--study-sha256', default=STUDY_SHA)
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
        result = launch(root, study, release, args.phase)
        if result['status'] == 'failed_or_incomplete':
            raise SystemExit(1)
    else:
        preflight(root, study, release, args.phase)
        print('Metadata verified; neither phase has executed.')


if __name__ == '__main__':
    main()
