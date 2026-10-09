#!/usr/bin/env python3
"""Prepared, source-bound saved-geometry witness; import does no patient I/O.

Only a later exact-source root release may run the worker. The rejected v4
mesh remains rejected regardless of this diagnostic's result.
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
import secrets
import subprocess
import sys
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
VERSION = 'resect-case4-boundary6-fidelity-witness-v1'
MANIFEST = f'manifests/experiments/{VERSION}.json'
DIAGNOSTIC = 'scripts/mechanics_patient_mesh_diagnostic.py'
BASE = 'scripts/mechanics_patient_mesh.py'
GUARD = 'scripts/mechanics_patient_mesh_candidate_run.py'
RUNTIME = 'scripts/febio_runtime.py'
CLOSURE = frozenset((f'scripts/mechanics_patient_mesh_boundary6_witness.py',
                     MANIFEST, DIAGNOSTIC, BASE, GUARD, RUNTIME,
                     'artifacts/febio-runtime-investigation-v1/prospective-runtime.json'))
CAPS = dict(parent_seconds=120, cooperative_seconds=110, rss_bytes=2*1024**3,
            numerical_threads=1, maximum_distance_queries=1250000,
            output_bytes=16*1024**2, per_file_bytes=8*1024**2,
            maximum_files=32, attempts=1, retries=0)
BASE_SHA = 'd8aae815c66ae017e925c1b4ba16f1e6a64c1d2ee314153ef9889c165d64b74d'
OUTCOME = 'completed_saved_geometry_diagnostic_only'
LAUNCH_TOKEN_ENV = 'RESECTIONLAB_CASE4_V4_WITNESS_LAUNCH_TOKEN'


def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def module(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    value = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(value)
    if Path(value.__file__).resolve() != Path(path).resolve():
        raise ValueError('Unexpected module origin')
    return value


def write(path, value):
    raw = (json.dumps(value, indent=2, allow_nan=False) + '\n').encode()
    if len(raw) > 1024**2:
        raise ValueError('JSON receipt cap')
    temporary = Path(path).with_suffix('.tmp')
    temporary.write_bytes(raw)
    temporary.replace(path)


def unchanged(bound):
    result = {}
    for path, digest in bound.items():
        try:
            result[path] = sha(path) == digest
        except OSError:
            result[path] = False
    return result


def require_supervised_launch(release_path, output):
    """Reject an ordinary direct worker call before any saved-array access.

    A random token is passed only by the one-shot parent launch and is bound to
    its release and output. This prevents accidental bypass of parent wall/RSS
    supervision; it is not a protection against a local operator changing code.
    """
    token = os.environ.get(LAUNCH_TOKEN_ENV, '')
    if not re.fullmatch('[0-9a-f]{64}', token):
        raise ValueError('Worker requires one-shot supervised launch token')
    context_path = Path(output).resolve()/'launch-context.json'
    context = json.loads(context_path.read_text())
    if (context.get('schema') != VERSION+'-launch-context'
            or context.get('release_sha256') != sha(release_path)
            or context.get('attempt_directory') != str(Path(output).resolve())
            or context.get('token_sha256') != hashlib.sha256(bytes.fromhex(token)).hexdigest()):
        raise ValueError('Supervised launch context mismatch')


def specification(release_path, output):
    """Hash-only preflight: no patient array/image decoding or geometry work."""
    release_path = Path(release_path).resolve()
    release = json.loads(release_path.read_text())
    config = json.loads((ROOT / MANIFEST).read_text())
    repository = Path(release['repository_directory']).resolve()
    attempt = Path(output).resolve()
    if (release.get('schema') != VERSION+'-release' or release.get('authorized') is not True
            or not release.get('root_release', '').strip()
            or Path(release['source_directory']).resolve() != ROOT
            or repository == ROOT or not ROOT.is_relative_to(repository)
            or not attempt.is_relative_to(repository)
            or Path(release['attempt_directory']).resolve() != attempt
            or not re.fullmatch('[0-9a-f]{40}', release.get('source_commit', ''))
            or set(release.get('source_sha256', {})) != CLOSURE):
        raise ValueError('Separate exact-source root release required')
    members = {str(path.relative_to(ROOT)) for path in ROOT.rglob('*') if path.is_file()}
    if members != CLOSURE or any(path.is_symlink() for path in ROOT.rglob('*')):
        raise ValueError('Exact source archive members required')
    if (config.get('schema') != VERSION or config.get('caps') != CAPS
            or config.get('status') != 'prospective_preparation_not_executed'
            or config.get('execution_authorized') is not False
            or any(release.get(k) is not False for k in
                   ('clinical_validation', 'mesh_accepted', 'solver_authorized'))):
        raise ValueError('Fixed caps/status/non-admission changed')
    bound = {str(release_path): sha(release_path)}
    for name, expected in release['source_sha256'].items():
        raw = subprocess.run(['git', '-C', str(repository), 'show',
                              release['source_commit']+':'+name],
                             capture_output=True, check=True, timeout=5).stdout
        if hashlib.sha256(raw).hexdigest() != expected or sha(ROOT/name) != expected:
            raise ValueError('Committed source/archive mismatch')
        bound[str(ROOT/name)] = expected
    if sha(ROOT/BASE) != BASE_SHA:
        raise ValueError('Original mesh helper identity changed')
    expected_keys = {'native_surface','nodes_m','tet10_indices','gmsh_node_ids',
                     'gmsh_element_ids','candidate_manifest','candidate_result',
                     'candidate_acceptance','candidate_release','candidate_summary'}
    if set(config['inputs']) != expected_keys:
        raise ValueError('Exact saved input set required')
    inputs = {}
    for key, row in config['inputs'].items():
        path = (repository / row['path']).resolve()
        if not path.is_relative_to(repository) or sha(path) != row['sha256']:
            raise ValueError('Saved input identity changed: '+key)
        inputs[key] = path
        bound[str(path)] = row['sha256']
    original = json.loads(inputs['candidate_result'].read_text())
    saved = json.loads(inputs['candidate_manifest'].read_text())
    accepted = json.loads(inputs['candidate_acceptance'].read_text())
    prior_release = json.loads(inputs['candidate_release'].read_text())
    summary = json.loads(inputs['candidate_summary'].read_text())
    if (original.get('status') != 'failed_or_incomplete'
            or original.get('diagnostic', {}).get('candidate_accepted') is not False
            or original.get('diagnostic', {}).get('complete') is not True
            or original.get('solver_calls') != 0 or original.get('solver_admitted') is not False
            or accepted.get('status') != 'failed_or_incomplete'
            or saved.get('candidate_accepted') is not False
            or prior_release.get('solver_authorized') is not False
            or summary.get('status') != 'rejected_independent_surface_fidelity'
            or summary.get('release_sha256') != sha(inputs['candidate_release'])):
        raise ValueError('Rejected v4 ancestry changed')
    for key in ('nodes_m','tet10_indices','gmsh_node_ids','gmsh_element_ids'):
        if saved['files'][key+'.npy']['sha256'] != config['inputs'][key]['sha256']:
            raise ValueError('Complete saved array manifest/hash mismatch')
    if (saved['context']['source_surface_binding']['sha256'] !=
            config['inputs']['native_surface']['sha256']):
        raise ValueError('Saved source-surface binding changed')
    fixed = config['fixed_geometry']
    if (saved['counts'] != {'nodes':fixed['mesh_nodes'],'tet10_elements':fixed['mesh_tet10']}
            or original['source_surface']['vertices'] != fixed['source_vertices']
            or original['source_surface']['triangles'] != fixed['source_triangles']
            or original['quality']['boundary']['vertices'] != fixed['exterior_vertices']
            or original['quality']['boundary']['triangles'] != fixed['exterior_triangles']):
        raise ValueError('Saved geometry counts changed')
    fidelity = config['original_fidelity']
    if fidelity['cover_m'] != .001 or fidelity['chunk_triangles'] != 256 or fidelity['limit_m'] != .002:
        raise ValueError('Original unchanged lattice/gate required')
    for direction in ('source_to_mesh','mesh_to_source'):
        for key, value in fidelity[direction].items():
            if original[direction][key] != value:
                raise ValueError('Saved fidelity result/declaration mismatch')
        if fidelity[direction]['maximum_sample_distance_m'] <= fidelity['limit_m']:
            raise ValueError('Rejected sampled distance must remain above gate')
    return repository, config, inputs, saved, original, bound


class Budget:
    def __init__(self):
        self.started = time.monotonic()
        self.queries = 0

    def check(self, count=0):
        if time.monotonic()-self.started > CAPS['cooperative_seconds']:
            raise TimeoutError('Cooperative deadline')
        if self.queries+count > CAPS['maximum_distance_queries']:
            raise ValueError('Total distance-query cap')
        self.queries += count


def directed(vertices, faces, distance, budget, numerical):
    """Replay original grouped lattice and retain per-face original-chunk cover."""
    maxima = np.empty(len(faces), dtype=float)
    covers = np.empty(len(faces), dtype=float)
    worst = []
    total = 0
    for points, chosen, count_each, cover in numerical.indexed_samples(
            vertices, faces, .001, CAPS['maximum_distance_queries'], chunk_triangles=256):
        budget.check(len(points))
        values = np.asarray(distance(points))
        if values.shape != (len(points),) or not np.isfinite(values).all() or np.any(values < 0):
            raise ValueError('Invalid distance oracle')
        maxima[chosen] = values.reshape(len(chosen), count_each).max(axis=1)
        covers[chosen] = cover
        best = np.lexsort((np.arange(len(values)), -values))[:10]
        worst.extend(dict(ordinal=total+int(i), triangle=int(chosen[i//count_each]),
                          point_m=points[i].tolist(), distance_m=float(values[i]),
                          original_chunk_cover_m=cover) for i in best)
        worst = sorted(worst, key=lambda row:(-row['distance_m'],row['ordinal']))[:10]
        total += len(points)
    contribution = maxima+covers
    return (dict(sample_count=total, maximum_sample_distance_m=float(maxima.max()),
                 full_surface_upper_bound_m=float(contribution.max()), worst_10=worst),
            maxima, contribution)


def cube_decision(directions, config):
    """Exact grid-cell summed-area cubes; face-centroid/whole-area proxy only."""
    step = config['absolute_ras_grid_m']
    side = config['cube_side_m']
    cells = round(side/step)
    if step != .005 or side != .03 or cells != 6:
        raise ValueError('Prospective cube/grid changed')
    selected = []
    for row in directions:
        xyz = np.asarray(row['centroids_m'], float)
        area = np.asarray(row['areas_m2'], float)
        if xyz.ndim != 2 or xyz.shape[1] != 3 or area.shape != (len(xyz),):
            raise ValueError('Witness centroid/area shape')
        if not np.isfinite(xyz).all() or not np.isfinite(area).all() or np.any(area <= 0):
            raise ValueError('Invalid witness centroid/area')
        selected.append((np.floor(xyz/step).astype(np.int64), area))
    if any(len(indices) == 0 for indices, _ in selected):
        raise ValueError('Both original failed directions require witnesses')
    low = np.min(np.vstack([v[0] for v in selected]), axis=0)-cells+1
    high = np.max(np.vstack([v[0] for v in selected]), axis=0)
    shape = tuple((high-low+1).tolist())
    if np.prod(shape) > 1000000:
        raise ValueError('Localization grid cap')
    captures = []
    totals = []
    for indices, areas in selected:
        histogram = np.zeros(shape, dtype=float)
        shifted = indices-low
        np.add.at(histogram, tuple(shifted.T), areas)
        prefix = np.pad(histogram, ((1,0),(1,0),(1,0))).cumsum(0).cumsum(1).cumsum(2)
        n0,n1,n2 = (dimension-cells+1 for dimension in shape)
        # Inclusion/exclusion of each half-open 6×6×6 grid-cell cube.
        x,y,z = n0,n1,n2
        sums = (prefix[cells:cells+x,cells:cells+y,cells:cells+z]
                -prefix[:x,cells:cells+y,cells:cells+z]
                -prefix[cells:cells+x,:y,cells:cells+z]
                -prefix[cells:cells+x,cells:cells+y,:z]
                +prefix[:x,:y,cells:cells+z]
                +prefix[:x,cells:cells+y,:z]
                +prefix[cells:cells+x,:y,:z]
                -prefix[:x,:y,:z])
        captures.append(np.maximum(sums, 0))
        totals.append(float(areas.sum()))
    fractions = [np.clip(capture/total,0,1) for capture,total in zip(captures,totals)]
    eligible = (captures[0]+captures[1]) > 0
    joint = np.minimum(fractions[0], fractions[1])
    top = np.max(joint[eligible])
    index = np.argwhere(eligible & (joint == top))[0]
    best = [float(v) for v in (low+index)*step]
    area_fractions = [float(row['witness_area_fraction']) for row in directions]
    captured = [float(value[tuple(index)]) for value in fractions]
    concentrated = (all(value <= config['maximum_witness_face_area_fraction_each_direction']
                        for value in area_fractions)
                    and all(value >= config['minimum_joint_cube_capture_fraction_each_direction']
                            for value in captured))
    return dict(status=('concentrated_under_diagnostic' if concentrated else
                        'local_refinement_premise_falsified_at_fixed_scale'),
                best_cube_origin_ras_m=best, best_cube_capture_fractions=dict(
                    source_to_mesh=captured[0], mesh_to_source=captured[1]),
                whole_witness_face_area_fractions=dict(
                    source_to_mesh=area_fractions[0], mesh_to_source=area_fractions[1]),
                scope='30 mm cube on absolute 5 mm RAS grid, half-open; entire-face/centroid proxy only')


def validate_arrays(inputs, saved, fixed):
    with np.load(inputs['native_surface'], allow_pickle=False) as archive:
        if set(archive.files) != {'vertices_m','triangles'}:
            raise ValueError('Unexpected native surface keys')
        source = {key: archive[key] for key in archive.files}
    arrays = {key:np.load(inputs[key],allow_pickle=False)
              for key in ('nodes_m','tet10_indices','gmsh_node_ids','gmsh_element_ids')}
    expected = dict(vertices_m=((fixed['source_vertices'],3),'float64'),
                    triangles=((fixed['source_triangles'],3),'int64'),
                    nodes_m=((fixed['mesh_nodes'],3),'float64'),
                    tet10_indices=((fixed['mesh_tet10'],10),'int64'),
                    gmsh_node_ids=((fixed['mesh_nodes'],),'int64'),
                    gmsh_element_ids=((fixed['mesh_tet10'],),'int64'))
    for key, value in {**source,**arrays}.items():
        shape, dtype = expected[key]
        if value.shape != shape or str(value.dtype) != dtype:
            raise ValueError('Saved array shape/dtype changed: '+key)
        if key in source:
            original = saved['context']['source_arrays'][key]
            if (original['shape'] != list(shape) or original['dtype'] != dtype or
                    hashlib.sha256(np.ascontiguousarray(value).tobytes()).hexdigest() != original['sha256']):
                raise ValueError('Original source array digest changed')
        else:
            metadata = saved['files'][key+'.npy']
            if metadata['shape'] != list(shape) or metadata['dtype'] != dtype:
                raise ValueError('Saved NPY metadata changed')
    X,F = source['vertices_m'],source['triangles']
    nodes,cells = arrays['nodes_m'],arrays['tet10_indices']
    if (not np.isfinite(X).all() or not np.isfinite(nodes).all()
            or F.min() < 0 or F.max() >= len(X)
            or cells.min() < 0 or cells.max() >= len(nodes)
            or np.any(F[:,0]==F[:,1]) or np.any(F[:,1]==F[:,2]) or np.any(F[:,0]==F[:,2])
            or np.any(np.diff(np.sort(cells[:,:4],axis=1),axis=1)==0)
            or any(len(np.unique(arrays[key])) != len(arrays[key]) for key in
                   ('gmsh_node_ids','gmsh_element_ids'))):
        raise ValueError('Invalid saved geometry/connectivity')
    return X,F,nodes,cells


def worker(release_path, output):
    output = Path(output)
    require_supervised_launch(release_path, output)
    if (output/'report.json').exists():
        raise FileExistsError('Existing attempt preserved')
    os.umask(0o077)
    budget = Budget()
    bound = {}
    report = dict(status='running', candidate_accepted=False, clinical_validation=False,
                  anatomical_registration_accepted=False, mesher_calls=0,
                  solver_calls=0, image_or_landmark_reads=0, retries=0)
    try:
        repository,config,inputs,saved,original,bound = specification(release_path,output)
        resource.setrlimit(resource.RLIMIT_FSIZE,(CAPS['per_file_bytes'],CAPS['per_file_bytes']))
        for name,version in config['packages'].items():
            if importlib.metadata.version(name) != version:
                raise ValueError('Package version changed: '+name)
        from vtkmodules.vtkCommonCore import vtkSMPTools
        numerical = module(ROOT/DIAGNOSTIC,'v4_frozen_diagnostic_helpers')
        numerical.configure_vtk_threads(vtkSMPTools)
        base = module(ROOT/BASE,'v4_frozen_mesh_helper')
        X,F,nodes,cells = validate_arrays(inputs,saved,config['fixed_geometry'])
        Y,G,used,middle = numerical.boundary(nodes,cells)
        fixed = config['fixed_geometry']
        if ((len(Y),len(G),len(middle)) !=
                (fixed['exterior_vertices'],fixed['exterior_triangles'],
                 fixed['exterior_distinct_midsides'])):
            raise ValueError('Exterior identity/count changed')
        distances = dict(source_to_mesh=base.vtk_distance_function(Y,G),
                         mesh_to_source=base.vtk_distance_function(X,F))
        report['counts'] = dict(source_vertices=len(X),source_triangles=len(F),
                                exterior_vertices=len(Y),exterior_triangles=len(G),
                                exterior_distinct_midsides=len(middle))
        replays = {}
        for name,A,T,B,U in (('source_to_mesh',X,F,Y,G),('mesh_to_source',Y,G,X,F)):
            result,maxima,contribution = directed(A,T,distances[name],budget,numerical)
            expected = config['original_fidelity'][name]
            tolerance = config['original_fidelity']['replay_absolute_tolerance_m']
            if (result['sample_count'] != expected['sample_count'] or
                    any(abs(result[k]-expected[k]) > tolerance for k in
                        ('maximum_sample_distance_m','full_surface_upper_bound_m'))):
                raise ValueError('Original 1 mm fidelity replay mismatch')
            replays[name] = (A,T,B,U,result,maxima,contribution)
        # Both original-gate directions must replay before localization, probes,
        # diagnostic output, or any new interpretation.
        payload = {}
        selections = []
        for name,(A,T,B,U,result,maxima,contribution) in replays.items():
            summary,angle,labels = numerical.summaries(A,T,maxima)
            nearest = numerical.witness_locator(B,U)
            for row in result['worst_10']:
                budget.check(1)
                row.update(nearest(row['point_m']))
                if abs(row['nearest_distance_m']-row['distance_m']) > 1e-10:
                    raise ValueError('Independent witness locator disagreement')
            result['summary'] = summary
            report[name] = result
            _,areas,_ = numerical.triangle_geometry(A,T)
            centroids = A[T].mean(axis=1)
            selected = maxima > config['localization_decision']['strict_face_max_threshold_m']
            selections.append(dict(centroids_m=centroids[selected],areas_m2=areas[selected],
                                   witness_area_fraction=summary['area_fraction_of_faces_with_sample_over_2mm']))
            payload[name+'_sampled_maximum_m'] = maxima
            payload[name+'_original_chunk_cover_upper_contribution_m'] = contribution
            payload[name+'_whole_face_area_m2'] = areas
            payload[name+'_maximum_adjacent_normal_angle_degrees'] = angle
            payload[name+'_exceedance_component_id'] = labels
            budget.check()
        report['localization'] = cube_decision(selections,config['localization_decision'])
        probes = dict(boundary_corners=nodes[used],boundary_straight_midsides=nodes[middle],
                      boundary_face_centroids=Y[G].mean(axis=1))
        report['mesh_to_source_probes'] = {}
        for name,points in probes.items():
            budget.check(len(points))
            values = np.asarray(distances['mesh_to_source'](points))
            if values.shape != (len(points),) or not np.isfinite(values).all() or np.any(values<0):
                raise ValueError('Invalid probe distance')
            report['mesh_to_source_probes'][name] = dict(count=len(points),maximum_m=float(values.max()),
                quantiles_m=np.quantile(values,[.5,.9,.95,.99,1.]).tolist(),
                over_2mm=int((values>.002).sum()))
        if budget.queries != 1193462:
            raise ValueError('Exact original-lattice/probe query budget changed')
        np.savez_compressed(output/'per-triangle-diagnostics.npz',**payload)
        report.update(status=OUTCOME, query_count=budget.queries,
                      original_2mm_gate='failed_unchanged',
                      interpretation='Saved-geometry localization only; no seam/cause, anatomy, mechanics, force or clinical validation')
    except BaseException as error:
        report.update(status='failed_or_incomplete',error=f'{type(error).__name__}: {error}')
    finally:
        report.update(query_count=budget.queries,elapsed_seconds=time.monotonic()-budget.started,
                      input_hashes=bound,inputs_unchanged=unchanged(bound))
        if not bound or not all(report['inputs_unchanged'].values()):
            report['status']='failed_or_incomplete'
        write(output/'report.json',report)
    return 0 if report['status']==OUTCOME else 1


def run(release_path, output):
    output = Path(output).resolve()
    _,_,_,_,_,bound = specification(release_path,output)
    os.umask(0o077)
    output.mkdir(parents=True,exist_ok=False,mode=0o700)
    acceptance = dict(status='failed_or_incomplete',candidate_accepted=False,
                      mesher_calls=0,solver_calls=0,clinical_validation=False,
                      anatomical_registration_accepted=False,
                      input_hashes_unchanged={},output_sha256={})
    guard = None
    spec = None
    passed = False
    try:
        token = secrets.token_hex(32)
        write(output/'launch-context.json',dict(schema=VERSION+'-launch-context',
              attempt_directory=str(output),release_sha256=sha(release_path),
              token_sha256=hashlib.sha256(bytes.fromhex(token)).hexdigest()))
        runtime = module(ROOT/RUNTIME,'v4_witness_supervisor')
        guard = module(ROOT/GUARD,'v4_witness_output_guard')
        spec = guard.ExecutionSpec(VERSION,MANIFEST,CLOSURE,frozenset(),BASE,
                                   'scripts/mechanics_patient_mesh_boundary6_witness.py',
                                   CAPS['output_bytes'],CAPS['per_file_bytes'],CAPS['maximum_files'])
        environment = runtime.private_environment(runtime.declaration())
        environment.update(PYTHONDONTWRITEBYTECODE='1',VTK_SMP_MAX_THREADS='1',
                           VTK_SMP_IMPLEMENTATION_TYPE='Sequential')
        environment[LAUNCH_TOKEN_ENV] = token
        command = [sys.executable,'-B',str(Path(__file__).resolve()),'worker',
                   '--release',str(Path(release_path).resolve()),'--output',str(output)]
        supervision,disk = guard.supervised(runtime,command,output,environment,
            {'aggregate_seconds':CAPS['parent_seconds'],
             'process_group_rss_bytes':CAPS['rss_bytes']},spec=spec)
        report = json.loads((output/'report.json').read_text()) if (output/'report.json').is_file() else {}
        report_hash = sha(output/'report.json') if (output/'report.json').is_file() else None
        geometry_hash = (sha(output/'per-triangle-diagnostics.npz')
                         if (output/'per-triangle-diagnostics.npz').is_file() else None)
        acceptance.update(supervision=supervision,output_guard=disk,
            output_sha256={'report.json':report_hash,
                           'per-triangle-diagnostics.npz':geometry_hash})
        passed = (supervision['status']=='completed' and supervision['exit_code']==0
                  and disk['error'] is None and report.get('status')==OUTCOME
                  and report.get('query_count')==1193462
                  and report.get('original_2mm_gate')=='failed_unchanged'
                  and report.get('candidate_accepted') is False
                  and report.get('mesher_calls')==0 and report.get('solver_calls')==0
                  and geometry_hash is not None and all(unchanged(bound).values()))
        if passed:
            acceptance['status'] = OUTCOME
    except BaseException as error:
        acceptance.update(status='failed_or_incomplete',error={
            'type':type(error).__name__,'message':str(error)[:4096]})
    finally:
        acceptance['input_hashes_unchanged']=unchanged(bound)
        if not bound or not all(acceptance['input_hashes_unchanged'].values()):
            passed=False
            acceptance['status']='failed_or_incomplete'
        try:
            if guard is not None:
                acceptance['final_output_usage']=guard.output_usage(output,spec=spec)
            for path in output.rglob('*'):
                if path.is_dir():
                    if path.stat().st_mode & 0o077:
                        raise ValueError('Private directory mode required')
                elif path.is_file() and path.stat().st_mode & 0o077:
                    raise ValueError('Private file mode required')
        except BaseException as error:
            passed=False
            acceptance.update(status='failed_or_incomplete',final_output_error=str(error)[:4096])
        write(output/'acceptance.json',acceptance)
        if guard is not None:
            try:
                acceptance['final_output_usage']=guard.output_usage(output,spec=spec)
            except BaseException as error:
                passed=False
                acceptance.update(status='failed_or_incomplete',final_output_error=str(error)[:4096])
            write(output/'acceptance.json',acceptance)
            try:
                guard.output_usage(output,spec=spec)
            except BaseException as error:
                passed=False
                acceptance.update(status='failed_or_incomplete',final_output_error=str(error)[:4096])
                write(output/'acceptance.json',acceptance)
    return 0 if passed else 1


if __name__ == '__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode',choices=('run','worker'))
    parser.add_argument('--release',required=True)
    parser.add_argument('--output',required=True)
    args=parser.parse_args()
    raise SystemExit((run if args.mode=='run' else worker)(args.release,args.output))
