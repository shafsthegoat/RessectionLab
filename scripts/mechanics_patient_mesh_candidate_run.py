#!/usr/bin/env python3
"""Prepared source-bound launcher for one separately released graded mesh.

Import is read-only and never loads patient arrays or a native runtime. The
worker may only reuse the exact retained v1 native surface after root release.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import importlib.util
import json
import os
from pathlib import Path
import re
import resource
import stat
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
VERSION = 'resect-case4-patient-mesh-graded-v2-release'
V1 = 'manifests/experiments/resect-case4-patient-mesh-v1.json'
V2 = 'manifests/experiments/resect-case4-patient-mesh-graded-v2.json'
CLOSURE = frozenset(['scripts/mechanics_patient_mesh_candidate_run.py',
    'scripts/mechanics_patient_mesh_candidate.py', 'scripts/mechanics_patient_mesh.py',
    V1, V2, 'scripts/febio_runtime.py', 'artifacts/febio-runtime-investigation-v1/prospective-runtime.json'])
CONTEXT_ROLES = frozenset(['mask_qc', 'baseline_diagnostic', 'root_visual_decision',
                          'prior_mesh_worker', 'independent_candidate_review'])
OUTPUT_BYTES = 8 * 1024**2
FILE_BYTES = 4 * 1024**2
OUTPUT_FILES = 32


def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def write(path, value):
    path = Path(path); temporary = path.with_suffix(path.suffix + '.tmp')
    data = (json.dumps(value, indent=2, allow_nan=False) + '\n').encode()
    if len(data) > 65536:
        raise ValueError('Launcher receipt exceeds64KiB')
    temporary.write_bytes(data); temporary.replace(path)


def module(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    value = importlib.util.module_from_spec(spec); spec.loader.exec_module(value)
    if Path(value.__file__).resolve() != Path(path).resolve():
        raise ValueError('Unexpected module origin')
    return value


def unchanged(bindings):
    result = {}
    for path, expected in bindings.items():
        try: result[path] = sha(path) == expected
        except OSError: result[path] = False
    return result


def released(release_path, output):
    path = Path(release_path).resolve(); release = json.loads(path.read_text())
    if release.get('schema') != VERSION or release.get('authorized') is not True:
        raise ValueError('Separate exact-source root release required')
    if Path(release['source_directory']).resolve() != ROOT or Path(release['attempt_directory']).resolve() != Path(output).resolve():
        raise ValueError('Released source/attempt mismatch')
    repository = Path(release['repository_directory']).resolve()
    commit = release.get('source_commit', '')
    if repository == ROOT or not re.fullmatch('[0-9a-f]{40}', commit):
        raise ValueError('Immutable exact committed source archive required')
    if any(release.get(name) is not False for name in ['clinical_validation', 'anatomical_registration_accepted', 'solver_authorized']):
        raise ValueError('Mesh release cannot authorize clinical claims or a solve')
    if release.get('reuse_saved_native_surface_only') is not True:
        raise ValueError('No MRI re-extraction or alternate anatomy permitted')
    if set(release['source_sha256']) != CLOSURE:
        raise ValueError('Exact seven-file execution closure required')
    bound = {str(path): sha(path)}
    for name, expected in release['source_sha256'].items():
        raw = subprocess.run(['git', '-C', str(repository), 'show', f'{commit}:{name}'], capture_output=True, check=True, timeout=5).stdout
        if hashlib.sha256(raw).hexdigest() != expected or sha(ROOT/name) != expected:
            raise ValueError('Committed source/archive mismatch')
        bound[str(ROOT/name)] = expected
    first = json.loads((ROOT/V1).read_text()); candidate = json.loads((ROOT/V2).read_text())
    if candidate['caps']['maximum_generations'] != 1 or candidate['caps']['retries'] != 0:
        raise ValueError('One fixed candidate required')
    if candidate['surface_fidelity'] != first['surface_fidelity'] or candidate['quality'] != first['quality']:
        raise ValueError('Independent physical gates changed')
    expected_surface = repository/candidate['source_surface_binding']['path']
    inputs = list(first['input_bindings']) + [{'path':str(expected_surface), 'sha256':candidate['source_surface_binding']['sha256']}]
    contexts = release['context_bindings']
    if len(contexts) != len(CONTEXT_ROLES) or {item['role'] for item in contexts} != CONTEXT_ROLES:
        raise ValueError('Fixed source/QC/review ancestry required')
    for item in inputs + contexts:
        item_path = Path(item['path']).resolve()
        if sha(item_path) != item['sha256']:
            raise ValueError('Released input/runtime/context changed')
        bound[str(item_path)] = item['sha256']
    prior = json.loads(Path(next(item['path'] for item in contexts if item['role']=='prior_mesh_worker')).read_text())
    if (prior.get('output_sha256', {}).get('native-surface.npz') != candidate['source_surface_binding']['sha256']
            or prior.get('status') != 'failed_or_incomplete'):
        raise ValueError('Retained failed-v1 source-surface ancestry mismatch')
    return candidate, first, expected_surface, bound


def output_usage(output):
    """Bounded metadata walk; do not follow symlinks or consume artifact bodies."""
    output = Path(output); files = 0; total = 0; largest = 0
    if not output.exists(): return {'files':0, 'bytes':0, 'largest_file_bytes':0}
    for directory, dirs, names in os.walk(output, followlinks=False):
        directory = Path(directory)
        if len(directory.relative_to(output).parts) > 4:
            raise ValueError('Output directory depth cap')
        for name in dirs + names:
            path = directory/name
            if path.is_symlink(): raise ValueError('Output symlink prohibited')
        for name in names:
            try:
                info = (directory/name).stat(follow_symlinks=False)
            except FileNotFoundError:
                # Child atomically publishes temporary files/directories. A
                # renamed entry can disappear between listing and stat; the
                # next sample and the stable final check see its new name.
                continue
            if not stat.S_ISREG(info.st_mode): raise ValueError('Only regular output files permitted')
            files += 1; total += info.st_size; largest = max(largest, info.st_size)
            if files > OUTPUT_FILES or total > OUTPUT_BYTES or largest > FILE_BYTES:
                raise ValueError('Output count/aggregate/per-file cap exceeded')
    return {'files':files, 'bytes':total, 'largest_file_bytes':largest}


def supervised(runtime, command, output, environment, caps):
    """Reuse reviewed group supervisor, extending its numeric observer with disk checks."""
    observer = runtime.process_group_rss
    audit = {'scope':'Sampled whole-attempt output plus hard child per-file limit',
        'aggregate_bytes_cap':OUTPUT_BYTES, 'per_file_bytes_cap':FILE_BYTES,
        'maximum_files':OUTPUT_FILES, 'observations':0, 'peak_sampled_bytes':0, 'error':None}
    def guarded(pgid, *, timeout_seconds):
        try:
            use = output_usage(output)
            audit['observations'] += 1
            audit['peak_sampled_bytes'] = max(audit['peak_sampled_bytes'], use['bytes'])
        except BaseException as error:
            audit['error'] = str(error)
            raise
        return observer(pgid, timeout_seconds=timeout_seconds)
    runtime.process_group_rss = guarded
    try:
        receipt = runtime.supervise(command, output/'supervision', cwd=ROOT, environment=environment,
            seconds=caps['aggregate_seconds'], rss_bytes=caps['process_group_rss_bytes'])
    finally:
        runtime.process_group_rss = observer
    try: audit['final_usage'] = output_usage(output)
    except BaseException as error: audit['error'] = str(error)
    return receipt, audit


def worker(release_path, output):
    output = Path(output)
    if (output/'worker.json').exists(): raise FileExistsError('Existing attempt preserved')
    result = {'status':'running', 'solver_calls':0, 'native_generation_calls':0,
              'source_arrays_reused':False, 'B_or_V_access':False, 'MRI_reextracted':False}
    bound = {}; gmsh = None; start = time.monotonic()
    try:
        config, first, surface_path, bound = released(release_path, output)
        # RLIMIT_FSIZE is child-local; stdout shares the supervised log descriptor.
        resource.setrlimit(resource.RLIMIT_FSIZE, (FILE_BYTES, FILE_BYTES))
        result.update(input_sha256=bound, per_file_hard_limit_bytes=FILE_BYTES)
        write(output/'worker.json', result)
        for name, expected in first['package_versions'].items():
            if importlib.metadata.version(name) != expected: raise ValueError('Package version changed: '+name)
        import numpy as np
        helper = module(ROOT/'scripts/mechanics_patient_mesh_candidate.py', 'released_graded_mesh')
        with np.load(surface_path, allow_pickle=False) as arrays:
            if set(arrays.files) != {'vertices_m', 'triangles'}: raise ValueError('Unexpected source-surface arrays')
            vertices = arrays['vertices_m']; faces = arrays['triangles']
        result['source_arrays_reused'] = True; write(output/'worker.json', result)
        if 'gmsh' in sys.modules: raise ValueError('Ambient Gmsh prohibited')
        runtime = first['gmsh_runtime']; gmsh = module(runtime['module_path'], 'gmsh')
        if Path(gmsh.lib._name).resolve() != Path(runtime['library_path']).resolve() or gmsh.__version__ != runtime['version']:
            raise ValueError('Unexpected native Gmsh runtime')
        gmsh.initialize([], readConfigFiles=False, run=False)
        from vtkmodules.vtkCommonCore import vtkSMPTools
        if not vtkSMPTools.SetBackend('Sequential') or vtkSMPTools.GetBackend() != 'Sequential':
            raise ValueError('Sequential VTK required')
        result['gmsh_runtime'] = {'module':str(gmsh.__file__), 'library':str(gmsh.lib._name), 'version':gmsh.__version__}
        result['native_generation_calls'] = None  # Candidate receipt charges before native generation; interruption can leave this unknown.
        write(output/'worker.json', result)
        assessed = helper.assess_candidate(gmsh, vertices, faces, output/'candidate',
            distance_factory=helper.BASE.vtk_distance_function)
        result['native_generation_calls'] = assessed['native_generation_calls']
        result['candidate_status'] = assessed['status']
        if (assessed['status']=='geometry_candidate_passed_no_solver_authorization'
                and assessed['native_generation_calls']==1 and assessed['solver_calls']==0
                and assessed['solver_admitted'] is False):
            result['status'] = 'completed_geometry_only'
        else: result['status'] = 'failed_or_incomplete'
    except BaseException as error:
        result.update(status='failed_or_incomplete', error={'type':type(error).__name__, 'message':str(error)[:4096]})
    finally:
        if gmsh is not None:
            try: gmsh.finalize()
            except BaseException as error: result.update(status='failed_or_incomplete', finalize_error=str(error)[:4096])
        result['elapsed_seconds'] = time.monotonic()-start
        result['inputs_after'] = unchanged(bound)
        if not bound or not all(result['inputs_after'].values()): result['status']='failed_or_incomplete'
        try:
            result['output_usage'] = output_usage(output)
            result['candidate_output_sha256'] = {str(p.relative_to(output)):sha(p)
                for p in (output/'candidate').rglob('*') if p.is_file()}
        except BaseException as error:
            result.update(status='failed_or_incomplete', output_error=str(error)[:4096])
        write(output/'worker.json',result)
    return 0 if result['status']=='completed_geometry_only' else 1


def launch(release_path, output):
    output=Path(output).resolve(); output.mkdir(parents=True,exist_ok=False)
    acceptance={'status':'failed_or_incomplete','worker_started':False,'solver_authorized':False,
                'clinical_validation':False,'anatomical_registration_accepted':False}
    bound={}
    try:
        config, _, _, bound=released(release_path,output)
        runtime=module(ROOT/'scripts/febio_runtime.py','graded_mesh_supervisor')
        env=runtime.private_environment(runtime.declaration())
        env.update(VTK_SMP_MAX_THREADS='1',VTK_SMP_IMPLEMENTATION_TYPE='Sequential')
        command=[sys.executable,'-B',str(Path(__file__).resolve()),'worker','--release',str(Path(release_path).resolve()),'--output',str(output)]
        acceptance.update(worker_started=True,command=command,thread_environment={key:env[key] for key in
            ['OMP_NUM_THREADS','OMP_DYNAMIC','VECLIB_MAXIMUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','VTK_SMP_MAX_THREADS','VTK_SMP_IMPLEMENTATION_TYPE']})
        receipt, disk=supervised(runtime,command,output,env,config['caps'])
        acceptance.update(supervision=receipt,output_guard=disk)
        record=json.loads((output/'worker.json').read_text()) if (output/'worker.json').is_file() else {}
        hashes=record.get('candidate_output_sha256',{})
        checks={}
        for name, expected in hashes.items():
            relative=Path(name)
            valid=not relative.is_absolute() and '..' not in relative.parts and relative.parts[0]=='candidate'
            checks[name]=valid and unchanged({str(output/relative):expected}).get(str(output/relative),False)
        acceptance['candidate_output_checks']=checks
        if (receipt['status']=='completed' and not disk['error'] and record.get('status')=='completed_geometry_only'
            and record.get('candidate_status')=='geometry_candidate_passed_no_solver_authorization'
            and record.get('native_generation_calls')==1 and record.get('solver_calls')==0
            and bool(checks) and all(checks.values()) and 'candidate/diagnostic/manifest.json' in checks):
            acceptance['status']='completed_geometry_only_no_solver_authorization'
    except BaseException as error:
        acceptance['error']={'type':type(error).__name__,'message':str(error)[:4096]}
    finally:
        acceptance['inputs_after']=unchanged(bound)
        if not bound or not all(acceptance['inputs_after'].values()): acceptance['status']='failed_or_incomplete'
        try: acceptance['final_output_usage']=output_usage(output)
        except BaseException as error:
            acceptance.update(status='failed_or_incomplete',output_error=str(error)[:4096])
        write(output/'acceptance.json',acceptance)
        try: output_usage(output)  # Include the final acceptance receipt itself.
        except BaseException as error:
            acceptance.update(status='failed_or_incomplete',output_error=str(error)[:4096])
            write(output/'acceptance.json',acceptance)
    return 0 if acceptance['status']=='completed_geometry_only_no_solver_authorization' else 1


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode',choices=['run','worker']);parser.add_argument('--release',required=True);parser.add_argument('--output',required=True)
    args=parser.parse_args()
    return worker(args.release,args.output) if args.mode=='worker' else launch(args.release,args.output)


if __name__=='__main__': raise SystemExit(main())
