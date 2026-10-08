#!/usr/bin/env python3
"""Separately released one-mesh preparation and one-solve terminal uniform N36 diagnostic.

No archive/release is generated automatically. Native meshing is confined to released preparation; measured-response
access is unavailable on this path. Existing process, output and native-call supervision
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
from scripts import mechanics_hbe_halfheight_global_n36 as core
from scripts import mechanics_hbe_halfheight_global_n36_readout as readout

from scripts import mechanics_hbe_halfheight_spatial_experiment as prior_runner
from scripts import mechanics_hbe_halfheight_global_experiment as previous_runner

access, backend, runtime = old.access, old.backend, old.runtime
THREADS = {key: '1' for key in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS',
                              'VECLIB_MAXIMUM_THREADS', 'NUMEXPR_NUM_THREADS')}


def source_inventory(root, release, study):
    """Seventeen unchanged modules plus three adapters; exact 26-file archive."""
    extras = {'halfheight_mesh': core.original, 'halfheight_readout': readout.inherited,
              'halfheight_spatial': core.spatial, 'halfheight_spatial_readout': prior_runner.readout,
              'halfheight_spatial_runner': prior_runner,
              'halfheight_global': core.previous, 'halfheight_global_readout': previous_runner.readout,
              'halfheight_global_runner': previous_runner,
              'halfheight_global_n36': core, 'halfheight_global_n36_readout': readout}
    sources = release['source_bindings']
    if set(sources) != set(study['inherited_source_sha256']) | {
            'halfheight_global_n36', 'halfheight_global_n36_readout', 'halfheight_global_n36_runner'}:
        raise ValueError('Exact twenty-module source closure required')
    base = {k: v for k, v in sources.items() if k not in extras and k != 'halfheight_global_n36_runner'}
    old.source_inventory(root, base, release['source_archive'], release['source_archive']['sha256'])
    for key, expected in study['inherited_source_sha256'].items():
        if sources[key]['sha256'] != expected:
            raise ValueError('Inherited scientific/runtime helper changed: '+key)
    paths = {k: Path(v.__file__).resolve() for k, v in extras.items()}
    paths['halfheight_global_n36_runner'] = Path(__file__).resolve()
    for key, path in paths.items():
        if access.local_path(root, sources[key]['path']) != path:
            raise ValueError('Imported diagnostic source origin differs: '+key)
        access.verify_binding(root, sources[key], maximum_bytes=1024**2)
    wanted = {f'scripts/{Path(v["path"]).name}': v['sha256'] for v in sources.values()}
    wanted[core.DECLARATION_PATH] = core.DECLARATION_SHA256
    for name in ('original_protocol', 'original_resolution_declaration', 'original_halfheight_declaration',
                 'original_spatial_declaration', 'original_global_declaration'):
        wanted[study[name]['path']] = study[name]['sha256']
    verify_archive(root, release['source_archive'], release['source_commit'], wanted, 26)


verify_archive = previous_runner.verify_archive
require_completed_native = previous_runner.require_completed_native
prior_metadata = previous_runner.prior_metadata


def N32_metadata(root, study, context, bound):
    """Authenticate complete N32 origin and its independently retained failure.

    Primitive and accepted response bytes are hash-bound without parsing any
    response during preflight. Complete raw replay occurs only in solve phase.
    """
    prior = study['baseline_N32']
    release, result, review = (bound(prior[name]) for name in ('release', 'result', 'review'))
    bound(prior['comparison'], json_value=False)
    state, baseline = bound(result['state']), bound(result['baseline'])
    if (review.get('status') != 'saved_execution_verified_negative_spatial_diagnostic'
            or review.get('blocking_record_mismatches') != []
            or review.get('complete_comparison_reproduced') is not True
            or review.get('original_inputs_and_saved_results_unchanged') is not True
            or review.get('actual_result_sha256') != prior['result']['sha256']
            or review.get('comparison_sha256') != prior['comparison']['sha256']
            or review.get('source_archive') != prior['source_archive']
            or review.get('source_commit') != prior['source_commit']
            or review.get('source_modules') != 17 or review.get('declarations') != 5
            or review.get('total_frames_replayed') != 305
            or review.get('conclusion', {}).get('spatial_convergence_accepted') is not False
            or review['conclusion'].get('calibration_released') is not False
            or review['conclusion'].get('remaining_exceedance_states') != list(range(49, 61))
            or review['conclusion'].get('individual_numerical_checks_passed') is not True
            or result.get('status') != 'completed_numerical_diagnostic_only'
            or result.get('measured_data_accessed') is not False
            or result['supervision'].get('status') != 'completed'
            or type(result['supervision'].get('exit_code')) is not int or result['supervision']['exit_code'] != 0
            or result['supervision'].get('kill_reason') is not None
            or result['supervision'].get('cleanup_error') is not None
            or not math.isfinite(result.get('phase_elapsed_seconds', math.inf))
            or not 0 <= result['phase_elapsed_seconds'] < 1200
            or state.get('solver_invocations') != 1 or state.get('gmsh_generation_calls') != 0
            or state.get('measured_data_accessed') is not False
            or set(state.get('runs', {})) != {prior['run_id']}
            or state['runs'][prior['run_id']]['status'] != 'passed_individual_numerical_checks'
            or state['runs'][prior['run_id']]['readout'] != prior['readout']):
        raise ValueError('Complete independently reviewed negative N32 diagnostic required')
    expected_replays = set(study['prior_runs']) | {study['baseline_P1']['run_id'], prior['run_id']}
    if (set(review['whole_record_replays']) != expected_replays
            or any(v.get('canonical_whole_record_equal') is not True or v.get('frame_count') != 61
                   for v in review['whole_record_replays'].values())):
        raise ValueError('Five exact accepted historical complete raw replays required')
    if (release.get('authorized') is not True or release.get('phase') != 'solve'
            or release.get('study') != study['original_global_declaration']
            or release.get('source_archive') != prior['source_archive']
            or release.get('source_commit') != prior['source_commit']
            or release.get('source_bindings') != prior['source_bindings']
            or release.get('interpreter') != study['interpreter']
            or release.get('backend_profile') != study['original_backend_profile']
            or baseline.get('release_binding', {}).get('sha256') != prior['release']['sha256']
            or baseline.get('source_bindings') != prior['source_bindings']
            or baseline.get('runtime_identity') != context['runtime_identity']):
        raise ValueError('Original N32 source/execution release differs')
    # Artifacts contains a verbatim copy; authenticate the original build-path
    # binding retained by its baseline as well as that copied release.
    bound(baseline['release_binding'])
    bound(release['independent_preparation_review'])
    source_root, wanted = access.local_path(root, prior['source_root']), {}
    if set(prior['source_bindings']) != set(study['inherited_source_sha256']):
        raise ValueError('Exact original seventeen-module N32 closure required')
    for key, record in prior['source_bindings'].items():
        member = f'scripts/{Path(record["path"]).name}'
        if (record['sha256'] != study['inherited_source_sha256'][key]
                or access.local_path(root, record['path']) != source_root/member):
            raise ValueError('Original N32 source replay context changed')
        bound(record, json_value=False, maximum_bytes=1024**2)
        wanted[member] = record['sha256']
    for name in ('original_protocol', 'original_resolution_declaration', 'original_halfheight_declaration',
                 'original_spatial_declaration', 'original_global_declaration'):
        record = study[name]
        wanted[record['path']] = record['sha256']
        bound({'path': str((source_root/record['path']).relative_to(root)), 'sha256': record['sha256']}, json_value=False)
    bound(prior['source_archive'], json_value=False)
    verify_archive(root, prior['source_archive'], prior['source_commit'], wanted, 22)
    execution = bound(prior['execution'])
    if (execution.get('schema') != 'hbe-halfheight-global-run-execution-v1'
            or execution.get('run_id') != prior['run_id']
            or execution.get('study') != study['original_global_declaration']
            or execution.get('primitive_bindings') != prior['primitive_bindings']
            or execution.get('reconstruction') != prior['reconstruction']
            or execution.get('runtime_identity') != context['runtime_identity']
            or execution.get('backend_profile') != study['original_backend_profile']
            or execution.get('execution') != state['runs'][prior['run_id']]['execution']):
        raise ValueError('Original N32 native origin differs')
    require_completed_native(execution['execution'], 750)
    bound(prior['readout'], json_value=False)
    bound(prior['reconstruction'], json_value=False, maximum_bytes=32*1024**2)
    if set(prior['primitive_bindings']) != readout.inherited.PRIMITIVE_KEYS:
        raise ValueError('Six complete original N32 primitives required')
    for name, record in prior['primitive_bindings'].items():
        bound(record, json_value=False, maximum_bytes=previous_runner.readout.primitive_limit(name))
    loading = bound(prior['primitive_bindings']['loading'])
    if ((loading.get('branch'), loading.get('steps'), loading.get('mu_Pa')) != ('compression', 60, 1000.)
            or loading.get('protocol_sha256') != study['original_protocol']['sha256']
            or loading.get('mesh_sha256') != prior['primitive_bindings']['mesh']['sha256']
            or loading.get('deck_sha256') != prior['primitive_bindings']['deck']['sha256']):
        raise ValueError('Original N32 fixed loading identity differs')
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
    if (release.get('schema') != 'hbe-halfheight-global-n36-release-v1' or release.get('authorized') is not True
            or release.get('phase') != phase or release.get('study') != study_binding):
        raise ValueError('Separate explicit source-bound N36 phase release required')
    protocol = bound(study['original_protocol'])
    core.spatial.require_protocol(protocol)
    for name in ('original_resolution_declaration', 'original_halfheight_declaration', 'original_spatial_declaration', 'original_global_declaration'):
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
    proposal, review = bound(study['reviewed_proposal']), bound(study['independent_design_review'])
    if (review.get('status') != 'one_uniform_consistency_test_is_informative_with_mandatory_scope_limits_not_a_release'
            or review.get('blocking_arithmetic_errors') != []
            or review['input_bindings']['build/hbe-n32-next-diagnostic-v1/diagnostic.json']['sha256'] != study['reviewed_proposal']['sha256']
            or review['mandatory_prospective_checks'] != study['numerical_design']['mandatory_prospective_checks']
            or proposal['recommendation']['prospective_resources_not_authorized'] != review['resource_review']['caps']):
        raise ValueError('Exact independently reviewed terminal N36 design required')
    prior_study = bound(study['original_spatial_declaration'])
    core.spatial.require_study(prior_study)
    if any(prior_study['prior_runs'].get(k) != v for k, v in study['prior_runs'].items()):
        raise ValueError('Original full-domain reference selection changed')
    prior_runner.accepted_evidence(prior_study, bound)
    prior_runner.prior_metadata(root, study, context, bound)
    bound(study['gmsh_runtime'])
    gmsh_runtime = core.BASE.verify_runtime(access.local_path(root, study['gmsh_runtime']['path']),
                                            study['gmsh_runtime']['sha256'], protocol)
    for name in ('module', 'library', 'license'):
        inputs[str(Path(gmsh_runtime[name]['path']).resolve())] = gmsh_runtime[name]['sha256']
    prior_metadata(root, study, context, bound)
    N32_metadata(root, study, context, bound)
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
    target = directory/'N36'
    target.mkdir(exist_ok=False)
    watch.active = target
    generated = target/'generated'
    try:
        extraction = core.generate_full_mesh(root, plan['study'], plan['protocol'], generated)
    finally:
        if (generated/'receipt.json').is_file():
            state['generation_receipt'] = old.binding(root, generated/'receipt.json')
            record = access.verify_binding(root, state['generation_receipt'], read_json=True)
            state['gmsh_generation_calls'] = record['gmsh_generation_calls']
            runtime.write_json(directory/'state.json', state)
    level = {'full_mesh': old.binding(root, generated/'mesh.json'),
             'mesh': old.saved(root, target/'mesh.json', extraction['mesh'])}
    wrapper = core.reconstruction(plan['study_binding'], plan['source_bindings'], extraction, level)
    level['reconstruction'] = old.saved(root, target/'reconstruction.json', wrapper)
    state['levels']['N36'] = level
    case = directory/'cases'/core.RUN_ID.replace(':', '-')
    case.mkdir(parents=True, exist_ok=False)
    watch.active = case
    xml, adapted, loading = core.case_contents(plan['study'], plan['protocol'], extraction, level)
    with (case/'skyline.feb').open('x') as stream:
        stream.write(xml)
    with (case/'specimen.feb').open('x') as stream:
        stream.write(adapted)
    state['cases'][core.RUN_ID] = dict(level, deck=old.binding(root, case/'specimen.feb'),
        backend_source_deck=old.binding(root, case/'skyline.feb'), loading=old.saved(root, case/'loading.json', loading))
    old.recheck(plan['inputs'])
    state['status'] = 'prepared_not_solved'
    watch.active = None


def verify_prepared_geometry(root, plan):
    row = plan['cases'][core.RUN_ID]
    core.verify_prepared_case(root, plan['study_binding'], row, row['reconstruction'])
    backend.verify_deck(access.local_path(root, row['backend_source_deck']['path']).read_bytes(),
                        access.local_path(root, row['deck']['path']).read_bytes())


def accepted_preparation(root, plan, release, bound):
    directory = Path(plan['output_root'])/'preparation'
    result = bound(release['preparation'])
    publication = bound(release['preparation_publication'])
    require_publication(publication, result_binding=release['preparation'], phase='prepare', cap=60,
                        output_cap=plan['study']['budgets']['generated_output_bytes'])
    if access.local_path(root, release['preparation_publication']['path']) != directory/'publication-check.json':
        raise ValueError('Exact accepted preparation publication check required')
    state, baseline = bound(result['state']), bound(result['baseline'])
    if (access.local_path(root, release['preparation']['path']) != directory/'result.json'
            or access.local_path(root, result['state']['path']) != directory/'state.json'
            or access.local_path(root, result['baseline']['path']) != directory/'baseline.json'
            or result.get('schema') != 'hbe-halfheight-global-n36-supervised-result-v1'
            or result.get('phase') != 'prepare' or result.get('status') != 'prepared_not_solved'
            or result.get('measured_data_accessed') is not False
            or baseline.get('phase') != 'prepare' or baseline.get('directory') != str(directory)
            or baseline.get('study_binding') != plan['study_binding']
            or baseline.get('source_bindings') != plan['source_bindings']
            or baseline.get('backend_profile') != plan['backend_profile']
            or baseline.get('runtime_identity') != plan['runtime_identity']
            or state.get('status') != 'prepared_not_solved' or state.get('phase') != 'prepare'
            or state.get('solver_invocations') != 0 or state.get('gmsh_generation_calls') != 1
            or state.get('measured_data_accessed') is not False
            or set(state.get('levels', {})) != {'N36'} or set(state.get('cases', {})) != {core.RUN_ID}):
        raise ValueError('Exact accepted one-mesh N36 preparation required')
    fresh = preflight(root, baseline['study_binding'], baseline['release_binding'], 'prepare')
    if access.canonical_json(fresh) != access.canonical_json(baseline):
        raise ValueError('Preparation baseline differs from released source/input graph')
    supervision, caps = result['supervision'], plan['study']['budgets']
    elapsed = publication.get('elapsed_through_result_publication_seconds', math.inf)
    supervised_cap = supervision.get('wall_cap_seconds', math.inf)
    if (supervision.get('status') != 'completed' or type(supervision.get('exit_code')) is not int
            or supervision['exit_code'] != 0 or supervision.get('kill_reason') is not None
            or supervision.get('cleanup_error') is not None
            or supervision.get('rss_cap_bytes') != caps['sampled_process_family_rss_bytes']
            or not math.isfinite(supervised_cap) or not 0 < supervised_cap <= caps['pure_preparation_seconds']
            or not math.isfinite(elapsed) or not 0 <= elapsed < caps['pure_preparation_seconds']):
        raise ValueError('Preparation exceeded aggregate supervision')
    old.recheck(baseline['inputs'])
    plan['inputs'].update(baseline['inputs'])
    level = state['levels']['N36']
    if set(level) != {'full_mesh', 'mesh', 'reconstruction'}:
        raise ValueError('Exact prepared N36 bindings required')
    for name, filename in (('full_mesh', 'generated/mesh.json'), ('mesh', 'mesh.json'),
                           ('reconstruction', 'reconstruction.json')):
        if access.local_path(root, level[name]['path']) != directory/'N36'/filename:
            raise ValueError('Prepared N32 geometry outside exact directory')
        bound(level[name], json_value=False, maximum_bytes=32*1024**2)
    row = state['cases'][core.RUN_ID]
    if (set(row) != {'full_mesh', 'mesh', 'reconstruction', 'deck', 'loading', 'backend_source_deck'}
            or any(row[name] != level[name] for name in level)):
        raise ValueError('Prepared N32 case/geometry graph differs')
    for name, filename in (('deck', 'specimen.feb'), ('loading', 'loading.json'), ('backend_source_deck', 'skyline.feb')):
        if access.local_path(root, row[name]['path']) != directory/'cases'/core.RUN_ID.replace(':', '-')/filename:
            raise ValueError('Prepared case outside exact directory')
        bound(row[name], json_value=name == 'loading', maximum_bytes=32*1024**2)
    generated = directory/'N36'/'generated'
    if access.local_path(root, state['generation_receipt']['path']) != generated/'receipt.json':
        raise ValueError('Exact native mesh preparation receipt required')
    receipt = bound(state['generation_receipt'])
    if (receipt.get('schema') != 'hbe-global-n36-mesh-receipt-v1'
            or receipt.get('status') != 'prepared_not_solved' or receipt.get('mesh_N') != 36
            or receipt.get('gmsh_generation_calls') != 1 or receipt.get('solver_calls') != 0
            or receipt.get('measured_data_accessed') is not False
            or receipt.get('declaration_sha256') != core.DECLARATION_SHA256
            or receipt.get('protocol_sha256') != plan['study']['original_protocol']['sha256']
            or receipt.get('runtime_receipt') != plan['study']['gmsh_runtime']
            or receipt.get('source_sha256') != plan['source_bindings']['halfheight_global_n36']['sha256']
            or receipt.get('generator_source_sha256') != plan['study']['inherited_source_sha256']['mesh_deck']
            or receipt.get('full_mesh_sha256') != level['full_mesh']['sha256']
            or receipt.get('runtime_unchanged') is not True or receipt.get('quality', {}).get('passed') is not True
            or 'finalize_error' in receipt or not math.isfinite(receipt.get('elapsed_seconds', math.inf))
            or not 0 <= receipt['elapsed_seconds'] < caps['pure_preparation_seconds']):
        raise ValueError('Complete source/runtime/quality-bound single Gmsh origin required')
    bound({'path': str((generated/'specimen.msh').relative_to(root)), 'sha256': receipt['native_mesh_sha256']},
          json_value=False)
    plan.update(levels=state['levels'], cases=state['cases'], generation_receipt=state['generation_receipt'])


# N24 replay remains byte-for-byte the accepted original-context pathway.
replay_P1 = previous_runner.replay_P1


# N32 is also replayed in its original source tree; its absolute-origin checks
# stay intact and no new native call or model response is manufactured.
N32_REPLAY_CODE = """
import sys
from pathlib import Path
sys.path.insert(0, sys.argv[1])
from scripts import mechanics_hbe_halfheight_global_readout as reader
a = reader.access
root = Path(sys.argv[2])
study = a.verify_binding(root, {'path':sys.argv[3], 'sha256':sys.argv[4]}, read_json=True)
prior = study['baseline_N32']
actual = reader.read_global_run(root, half_bindings=prior['primitive_bindings'],
    reconstruction_binding=prior['reconstruction'], declaration_binding=study['original_global_declaration'])
actual['execution_binding'] = prior['execution']
accepted = a.verify_binding(root, prior['readout'], read_json=True)
if actual['passed'] is not True or a.canonical_json(actual) != a.canonical_json(accepted):
    raise ValueError('Complete original-context N32 replay differs from accepted receipt')
with Path(sys.argv[5]).open('xb') as stream:
    stream.write(a.canonical_json(actual))
"""


def replay_N32(root, plan, state, watch, deadline):
    directory = Path(plan['directory'])/'N32-replay'
    directory.mkdir(exist_ok=False)
    watch.active = directory
    prior = plan['study']['baseline_N32']
    command = [sys.executable, '-I', '-B', '-c', N32_REPLAY_CODE,
        str(access.local_path(root, prior['source_root'])), str(root), plan['study_binding']['path'],
        plan['study_binding']['sha256'], str(directory/'readout.json')]
    started, process = time.monotonic(), None
    receipt = {'schema': 'hbe-n36-N32-replay-execution-v1', 'status': 'starting',
        'native_solver_calls': 0, 'original_source_commit': prior['source_commit'],
        'original_source_archive': prior['source_archive'], 'command': command,
        'elapsed_seconds': 0., 'exit_code': None, 'reaped': False}
    try:
        old.recheck(plan['inputs'])
        if deadline <= time.monotonic():
            raise TimeoutError('Aggregate allowance exhausted before original-context N32 replay')
        with (directory/'console.txt').open('x') as stream:
            process = subprocess.Popen(command, cwd=root, stdout=stream, stderr=subprocess.STDOUT)
            receipt['exit_code'] = process.wait(timeout=max(0., deadline-time.monotonic()))
        if receipt['exit_code'] != 0 or time.monotonic() >= deadline:
            raise RuntimeError('Original-context N32 replay failed or exceeded aggregate cap')
        actual = access.verify_binding(root, old.binding(root, directory/'readout.json'), read_json=True)
        accepted = access.verify_binding(root, prior['readout'], read_json=True)
        if access.canonical_json(actual) != access.canonical_json(accepted):
            raise ValueError('Returned N32 replay differs from accepted whole receipt')
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
        state['N32_replay'] = old.saved(root, directory/'execution.json', receipt)
        runtime.write_json(Path(plan['directory'])/'state.json', state)
        watch.active = None


def solve_cases(root, plan, state, watch, deadline):
    directory = Path(plan['directory'])
    verify_prepared_geometry(root, plan)
    runs = readout.replay_prior_runs(root, plan['study'])
    for key, report in runs.items():
        state['reused_runs'][key] = old.saved(root, directory/('reused-'+key.replace(':', '-')+'.json'), report)
    old.recheck(plan['inputs'])
    runs[plan['study']['baseline_P1']['run_id']] = replay_P1(root, plan, state, watch, deadline)
    runs[plan['study']['baseline_N32']['run_id']] = replay_N32(root, plan, state, watch, deadline)
    for key in core.ORDERED_RUNS:
        row = state['runs'][key]
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
            if seconds <= 0 or state['solver_invocations'] >= 1:
                raise TimeoutError('Declared native solve allocation exhausted')
            command = [plan['executable'], '-noconfig', '-no_title', '-i', 'specimen.feb', '-o', 'solver.log']
            row.update(status='solver_started', input_bindings=primitives, command=command)
            state['solver_invocations'] += 1
            runtime.write_json(directory/'state.json', state)
            row['execution'] = old.solve(command, target, seconds)
            require_completed_native(row['execution'], seconds)
            for name, filename in (('nodes', 'nodes.log'), ('elements', 'elements.log'), ('solver', 'solver.log')):
                primitives[name] = old.binding(root, target/filename)
            execution = {'schema': 'hbe-halfheight-global-n36-run-execution-v1', 'run_id': key,
                'cwd': str(target.relative_to(root)), 'study': plan['study_binding'],
                'runtime_identity': plan['runtime_identity'], 'backend_profile': plan['backend_profile'],
                'backend_source_deck': prepared['backend_source_deck'], 'executable': old.binding(root, plan['executable']),
                'primitive_bindings': primitives, 'reconstruction': prepared['reconstruction'],
                'command': command, 'execution': row['execution']}
            execution_binding = old.saved(root, target/'execution.json', execution)
            report = readout.read_global_run(root, half_bindings=primitives,
                reconstruction_binding=prepared['reconstruction'], declaration_binding=plan['study_binding'])
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
    state['status'] = 'completed_numerical_diagnostic_only'


def worker(root, plan, deadline):
    directory = Path(plan['directory'])
    started = time.monotonic()
    phase, caps = plan['phase'], plan['study']['budgets']
    if not math.isfinite(deadline) or deadline <= started:
        raise TimeoutError('Supervised phase allowance exhausted during preflight')
    state = {'schema': 'hbe-halfheight-global-n36-state-v1', 'phase': phase, 'status': 'running',
             'solver_invocations': 0, 'gmsh_generation_calls': 0,
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



# New timing contract includes durable result publication. Historical runners
# retain their explicitly different timing boundary unchanged.



def durable_json(root, path, value):
    """Exclusive JSON publication followed by file and containing-dir fsync."""
    with Path(path).open('xb') as stream:
        stream.write(access.canonical_json(value))
        stream.flush()
        os.fsync(stream.fileno())
    descriptor = os.open(str(Path(path).parent), os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)
    return old.binding(root, path)


def require_publication(record, *, result_binding, phase, cap, output_cap):
    if (record.get('schema') != 'hbe-n36-result-publication-check-v1'
            or record.get('accepted') is not True or record.get('result') != result_binding
            or record.get('phase') != phase or record.get('cap_seconds') != cap
            or record.get('output_cap_bytes') != output_cap
            or record.get('durable_result_publication_checked') is not True
            or record.get('clock_scope') != 'through_result_fsync_directory_fsync_and_retained_scan_excludes_closeout_publication'
            or record.get('own_publication_inside_clock_claim') is not False
            or not math.isfinite(record.get('elapsed_through_result_publication_seconds', math.inf))
            or not 0 <= record['elapsed_through_result_publication_seconds'] < cap
            or type(record.get('retained_bytes_including_closeout')) is not int
            or not 0 <= record['retained_bytes_including_closeout'] <= output_cap):
        raise ValueError('Complete bounded durable result publication required')


def finalize_result(root, directory, raw_root, result, started, cap, output_cap):
    """The immutable result is accepted only by a separate closing observation.

    Count source checks, execution, readout, cleanup, durable result publication
    and retained-byte scan. The tiny closeout describes that observation and
    honestly excludes its own subsequent publication; no hard realtime claim.
    """
    result['timing_scope'] = 'payload_before_publication; acceptance_requires_publication-check.json'
    result['phase_elapsed_before_result_publication_seconds'] = time.monotonic()-started
    if not 0 <= result['phase_elapsed_before_result_publication_seconds'] < cap:
        result.update(status='failed_or_incomplete', final_cap_failure='inclusive_time_cap')
    result['output_cap_bytes'] = output_cap
    result_binding = durable_json(root, directory/'result.json', result)
    retained = old.meshing.tree_bytes(raw_root)
    elapsed = time.monotonic()-started
    expected = 'prepared_not_solved' if result['phase'] == 'prepare' else 'completed_numerical_diagnostic_only'
    closeout = {'schema': 'hbe-n36-result-publication-check-v1', 'result': result_binding,
        'phase': result['phase'], 'result_status': result['status'],
        'accepted': result['status'] == expected and math.isfinite(elapsed) and 0 <= elapsed < cap,
        'elapsed_through_result_publication_seconds': elapsed, 'cap_seconds': cap,
        'output_cap_bytes': output_cap, 'retained_bytes_before_closeout': retained,
        'retained_bytes_including_closeout': 0, 'durable_result_publication_checked': True,
        'clock_scope': 'through_result_fsync_directory_fsync_and_retained_scan_excludes_closeout_publication',
        'own_publication_inside_clock_claim': False,
        'hard_real_time_or_power_loss_guarantee': False}
    for _ in range(8):
        total = retained+len(access.canonical_json(closeout))
        accepted = closeout['accepted'] and total <= output_cap
        if total == closeout['retained_bytes_including_closeout'] and accepted == closeout['accepted']:
            break
        closeout.update(retained_bytes_including_closeout=total, accepted=accepted)
    else:
        raise RuntimeError('Publication byte accounting did not settle')
    publication = durable_json(root, directory/'publication-check.json', closeout)
    return dict(result, result_binding=result_binding, publication=publication,
                status=result['status'] if closeout['accepted'] else 'failed_or_incomplete')


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
    result = {'schema': 'hbe-halfheight-global-n36-supervised-result-v1', 'phase': phase,
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
