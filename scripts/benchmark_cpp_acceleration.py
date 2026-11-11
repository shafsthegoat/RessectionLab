#!/usr/bin/env python3
"""Data-free fixed-work C++ comparison, in equivalent isolated fresh processes.

Profile runs are attribution only and never enter timing samples. The manifest
owns the comparison contract; outputs retain every sample, failure and exact
result. Run only one copy at a time alongside other heavyweight repository work.
"""
from pathlib import Path
import argparse
import cProfile
import hashlib
import json
import os
import platform
import pstats
import resource
import statistics
import subprocess
import sys
import sysconfig
import time

ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / 'manifests/experiments/cpp-acceleration-v1.json'
THREADS = ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS',
           'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS', 'BLIS_NUM_THREADS')


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')


MEASURED_PATHS = {('report', 'cost', 'wallSeconds'),
                  ('report', 'alternatives', '[]', 'ledger', 'wallSecondsSinceMeterCreation'),
                  ('report', 'comparisonSeal'),
                  ('report', 'alternatives', '[]', 'goalAssessment', 'goalHash')}


def canonical_result(value, path=()):
    """Exclude enumerated wall fields and process-bound goal/report seals.

    Existing goalHash derives from limits containing monotonic frozen_at and
    meter_identity=id(meter). All actual goal decisions/counts remain exact.
    Physical modeled elapsedSeconds and other identities remain exact.
    """
    if isinstance(value, list):
        return [canonical_result(v, (*path, '[]')) for v in value]
    if isinstance(value, dict):
        return {k: canonical_result(v, (*path, k)) for k, v in value.items()
                if (*path, k) not in MEASURED_PATHS}
    return value


def geometry_control(stress=False):
    import numpy as np
    from resectionlab.geometry import GeometryScene, capsule_voxel_indices
    n = (96, 96, 128) if stress else (12, 13, 14)
    affines = [np.eye(4), np.array([[.8, .2, 0., 3.], [0., 1.1, .15, -2.],
                                  [0., 0., 1.4, 5.], [0., 0., 0., 1.]])]
    if stress:
        queries = [([3.+i*.125, 5., 7.], [90., 89.-i*.125, 118.], 1.25)
                   for i in range(6)]
    else:
        queries = [([3., 3., 3.], [3., 3., 3.], 0.),
                   ([1., 1., 1.], [10., 10., 10.], .5),
                   ([1.5, 0., 0.], [1.5, 12., 0.], 0.),
                   ([-5., -5., -5.], [-2., -2., -2.], 0.),
                   ([0., 0., 0.], [11., 0., 0.], 1.)]
    results = []
    for affine in affines:
        scene = GeometryScene(np.zeros(n, bool), affine)
        for a, b, radius in queries:
            start, end = scene.voxel_to_world(np.asarray([a, b]))
            first = capsule_voxel_indices(scene, start, end, radius)
            # Exact repeated queries exercise warm use without changing work.
            again = capsule_voxel_indices(scene, start, end, radius)
            assert np.array_equal(first, again)
            results.append({'count': len(first), 'orderedIndicesHash':
                            'sha256:' + hashlib.sha256(first.tobytes(order='C')).hexdigest()})
    return {'geometry': results, 'shape': n, 'queriesPerAffine': queries,
            'affines': [a.tolist() for a in affines]}


def perform(workload):
    import numpy as np
    from resectionlab.core import semantic_digest
    from resectionlab.generated_cpp_fixture import cranial_generate_fixture, continuum_fixture
    from resectionlab.surgical_cranial_planning import compare_cranial_strategies
    started = time.perf_counter()
    diagnostics = {}
    if workload in ('generate', 'standardized-refusal'):
        physics = 'declared_legacy' if workload == 'generate' else 'standardized'
        case, base, roots, record = cranial_generate_fixture(physics=physics)
        before = semantic_digest(record)
        setup = time.perf_counter() - started
        compute = time.perf_counter()
        # Retain the actual registered goal, whose existing declaration binds a
        # monotonic origin and object address, to validate its process-bound seal.
        from resectionlab import surgical_joint_cranial_search as search
        goals = []
        root_states = {r['sourceSnapshot']['stateId'] for r in roots}
        def retain_goals(*args):
            for goal in tuple(search._GOALS.values()):
                if goal.source_state_id in root_states and not any(goal is previous for previous in goals):
                    goals.append(goal)
        result = compare_cranial_strategies(base, record, planning_horizon=3, beam_width=3,
                                            progress=retain_goals)
        elapsed = time.perf_counter() - compute
        assert len(goals) == len(roots), 'Supported progress callback did not retain each registered root goal'
        assert semantic_digest(record) == before, 'Generate mutated the accepted common source'
        report = result['report']
        assert report['comparisonSeal'] == semantic_digest(
            {k: v for k, v in report.items() if k != 'comparisonSeal'}), 'Invalid comparison seal'
        contracts = []
        raw_contracts = []
        for goal in goals:
            search._require_goal(goal, goal._limits)
            raw_goal, raw_limits = goal.declaration(), goal._limits.declaration()
            assert goal.content_hash == semantic_digest(raw_goal)
            assert goal._limits.content_hash == semantic_digest(raw_limits)
            assert raw_goal['limitsHash'] == goal._limits.content_hash
            stable_limits = {k: v for k, v in raw_limits.items() if k not in ('frozen_at', 'meter_identity')}
            stable_goal = {**raw_goal, 'limitsHash': semantic_digest(stable_limits)}
            contracts.append({'goalDeclaration': stable_goal, 'limitsDeclaration': stable_limits})
            raw_contracts.append({'goalHash': goal.content_hash, 'goalDeclaration': raw_goal,
                                  'limitsHash': goal._limits.content_hash, 'limitsDeclaration': raw_limits})
        assert all(row['goalAssessment']['goalHash'] in {g.content_hash for g in goals}
                   for row in report['alternatives']), 'Unregistered report goal hash'
        if workload == 'generate':
            assert report['decision'] == 'COMPLETED_FROZEN_TASK'
            assert report['cost']['physicalCalls'] == 370
            assert [r['nativeActions'] for r in report['alternatives']] == [112]
            assert report['alternatives'][0]['ledger']['wholeNonTargetRemovedMm3'] == 334.
        else:
            assert not result['branches'] and report['selectedBranchId'] is None
            assert any(r['message'] == 'CRANIAL_ACCESS_BEYOND_INNER_TABLE_BEFORE_EVACUATION'
                       for r in report['failures'])
        value = {'report': report, 'snapshots': [e.snapshot() for e, _ in result['branches']],
                 'caseSemanticHash': case.semantic_hash, 'goalContracts': contracts}
        diagnostics['validatedRawGoalContracts'] = raw_contracts
        diagnostics['productionCellCoverCaches'] = [e._cell_cover_cache.stats() for e, _ in result['branches']]
    elif workload == 'stress':
        setup = 0.
        from resectionlab import surgical_deformation_v2 as brain
        value = geometry_control(stress=True)
        engine, cell = continuum_fixture(13)
        states = []
        for duration in (.1, .2, .3):
            brain.tick(engine, duration)
            states.append(semantic_digest(engine.state))
        engine.state['removed'] = [cell]
        brain.tick(engine, .1)
        states.append(semantic_digest(engine.state))
        brain.release(engine, 'left')
        brain.tick(engine, .2)
        states.append(semantic_digest(engine.state))
        value.update(continuumStateHashes=states, continuumState=engine.state)
        elapsed = time.perf_counter() - started
    else:
        assert workload == 'small'
        setup = 0.
        value = geometry_control()
        elapsed = time.perf_counter() - started
    return value, {'fixtureSeconds': setup, 'computeSeconds': elapsed,
                   'completeSeconds': setup + elapsed}, diagnostics


def worker(args):
    # Explicit path prevents an editable install in a neighboring environment
    # from silently supplying the wrong checkout.
    sys.path.insert(0, str(args.source_root.resolve()))
    import resectionlab
    from resectionlab.core import semantic_digest
    profile = cProfile.Profile() if args.profile else None
    if profile:
        profile.enable()
    value, timings, diagnostics = perform(args.worker)
    if profile:
        profile.disable()
    try:
        from resectionlab.native_backend import backend_identity
        backend = backend_identity()
    except ModuleNotFoundError as exc:
        if exc.name != 'resectionlab.native_backend' or args.backend == 'cpp':
            raise
        backend = {'mode': 'reference', 'implementation': 'pre-port-baseline'}
    exact = canonical_result(value)
    row = {'schema': 'resectionlab.cpp-acceleration-sample.v1', 'workload': args.worker,
           'backendRequested': args.backend, 'backend': backend,
           'diagnostics': diagnostics,
           'timings': timings, 'exactDigest': semantic_digest(exact), 'result': value,
           'peakRssBytes': resource.getrusage(resource.RUSAGE_SELF).ru_maxrss *
               (1 if sys.platform == 'darwin' else 1024),
           'environment': {'python': sys.version, 'executable': sys.executable,
               'machine': platform.machine(), 'platform': platform.platform(),
               'abi': sysconfig.get_config_var('SOABI'),
               'modulePath': str(Path(resectionlab.__file__).resolve()),
               'threads': {k: os.environ.get(k) for k in THREADS}}}
    if profile:
        stats = pstats.Stats(profile)
        row['profile'] = {'totalSeconds': stats.total_tt, 'functions': [
            {'file': k[0], 'line': k[1], 'function': k[2], 'calls': v[1],
             'selfSeconds': v[2], 'cumulativeSeconds': v[3]}
            for k, v in sorted(stats.stats.items(), key=lambda item: item[1][3], reverse=True)]}
        profile.dump_stats(args.output.with_suffix('.prof'))
    write_json(args.output, row)
    print(json.dumps({k: row[k] for k in ('workload', 'backendRequested', 'timings', 'exactDigest', 'peakRssBytes')}))


def run_sample(args, workload, backend, name, *, profile=False):
    output = args.output / (name + '.json')
    source = args.baseline_source if backend == 'reference' and args.baseline_source else args.source_root
    command = [sys.executable, str(Path(__file__).resolve()), '--worker', workload,
               '--backend', backend, '--output', str(output), '--source-root', str(source)]
    if profile:
        command.append('--profile')
    env = dict(os.environ, RESSECTIONLAB_NUMERICAL_BACKEND=backend, PYTHONPATH=str(source.resolve()))
    env.update({name: '1' for name in THREADS})
    started = time.perf_counter()
    with output.with_suffix('.stdout').open('w') as stdout, output.with_suffix('.stderr').open('w') as stderr:
        process = subprocess.Popen(command, cwd=ROOT, env=env, stdout=stdout, stderr=stderr)
        # psutil is optional: ru_maxrss remains authoritative for this
        # single-process workload; tree samples cover unexpected child work.
        try:
            import psutil
            parent = psutil.Process(process.pid)
        except ImportError:
            parent = None
        peak = 0
        while process.poll() is None:
            if time.perf_counter() - started > 900:
                process.kill()
                process.wait()
                raise TimeoutError(f'{name}: 900 second hard process limit; outputs retained')
            if parent is not None:
                try:
                    peak = max(peak, sum(p.memory_info().rss for p in [parent, *parent.children(recursive=True)]
                                         if p.is_running()))
                    if peak > 1073741824:
                        process.kill()
                        process.wait()
                        write_json(output.with_suffix('.failure.json'),
                            {'error': 'process-tree-memory-hard-cap', 'peakRssBytes': peak,
                             'hardCapBytes': 1073741824})
                        raise MemoryError(f'{name}: 1 GiB process-tree hard cap; outputs retained')
                except psutil.Error:
                    pass
            time.sleep(.02)
    if process.returncode:
        raise RuntimeError(f'{name} failed ({process.returncode}); see {output.with_suffix(".stderr")}')
    row = json.loads(output.read_text())
    row['processElapsedSeconds'] = time.perf_counter() - started
    row['processTreePeakRssBytes'] = max(peak, row['peakRssBytes'])
    write_json(output, row)
    return row


def summarize(rows):
    times = [r['timings']['completeSeconds'] for r in rows]
    return {'medianSeconds': statistics.median(times), 'minSeconds': min(times),
            'maxSeconds': max(times), 'stdevSeconds': statistics.stdev(times) if len(times) > 1 else 0.,
            'sampleCount': len(times), 'samplesSeconds': times,
            'medianPeakRssBytes': statistics.median(r['processTreePeakRssBytes'] for r in rows),
            'peakRssBytes': max(r['processTreePeakRssBytes'] for r in rows)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--source-root', type=Path, default=ROOT/'src')
    parser.add_argument('--baseline-source', type=Path, help='Captured original src for pre-port comparison')
    parser.add_argument('--workload', nargs='+', choices=('small', 'generate', 'standardized-refusal', 'stress'),
                        default=['small', 'generate', 'standardized-refusal', 'stress'])
    parser.add_argument('--mode', choices=('compare', 'baseline', 'profile'), default='compare')
    parser.add_argument('--repeats', type=int, default=5)
    parser.add_argument('--warmups', type=int, default=1)
    parser.add_argument('--worker', choices=('small', 'generate', 'standardized-refusal', 'stress'))
    parser.add_argument('--backend', choices=('reference', 'cpp', 'auto'), default='reference')
    parser.add_argument('--profile', action='store_true')
    parser.add_argument('--export-case', type=Path, help='Save a pristine generated source for actual desktop Generate')
    args = parser.parse_args()
    if args.export_case:
        sys.path.insert(0, str(args.source_root.resolve()))
        from resectionlab.generated_cpp_fixture import cranial_generate_fixture
        from resectionlab.imaging import save_case
        case, _, _, record = cranial_generate_fixture()
        args.export_case.parent.mkdir(parents=True, exist_ok=True)
        save_case(case, args.export_case, artifacts={'surgical_toolbox': record})
        print(args.export_case.resolve())
        return
    if args.worker:
        worker(args)
        return
    if args.repeats < 5 or args.warmups < 1:
        parser.error('Frozen protocol requires at least five repetitions and one warmup')
    args.output.mkdir(parents=True, exist_ok=False)
    manifest = json.loads(MANIFEST.read_text())
    receipt = {'manifestSha256': hashlib.sha256(MANIFEST.read_bytes()).hexdigest(),
               'manifest': manifest, 'mode': args.mode, 'workloads': {}, 'passed': True}
    write_json(args.output/'summary.json', receipt)
    for workload in args.workload:
        if args.mode == 'profile':
            row = run_sample(args, workload, args.backend, workload+'-profile', profile=True)
            receipt['workloads'][workload] = {'attributionOnly': True, 'profileSeconds': row['profile']['totalSeconds']}
            continue
        backends = ['reference', 'cpp'] if args.mode == 'compare' else ['reference']
        for backend in backends:
            for index in range(args.warmups):
                run_sample(args, workload, backend, f'{workload}-warmup-{backend}-{index}')
        rows = {backend: [] for backend in backends}
        pairs = []
        for index in range(args.repeats):
            order = backends if index % 2 == 0 else list(reversed(backends))
            for backend in order:
                row = run_sample(args, workload, backend, f'{workload}-pair-{index}-{backend}')
                rows[backend].append(row)
            pairs.append({'index': index, 'order': order})
        digests = {r['exactDigest'] for group in rows.values() for r in group}
        result = {'pairs': pairs, 'backends': {b: summarize(r) for b, r in rows.items()},
                  'exactEquivalent': len(digests) == 1, 'exactDigests': sorted(digests)}
        result['memoryHardCapPassed'] = all(r['processTreePeakRssBytes'] <= 1073741824
                                            for group in rows.values() for r in group)
        if workload == 'generate':
            from_digest = manifest['correctness']['exactBaselineRepresentativeCanonicalReportDigest']
            snapshot_hashes = manifest['correctness']['exactBaselineRepresentativeSnapshotHashes']
            # Guard the original pre-port behavior, including work/call trace,
            # before comparing the two current backend modes.
            sys.path.insert(0, str(args.source_root.resolve()))
            from resectionlab.core import semantic_digest
            result['preportEquivalent'] = all(
                semantic_digest(canonical_result(r['result'])['report']) == from_digest and
                [semantic_digest(s) for s in r['result']['snapshots']] == snapshot_hashes
                for group in rows.values() for r in group)
        if 'cpp' in rows:
            ref, cpp = result['backends']['reference'], result['backends']['cpp']
            result['speedup'] = ref['medianSeconds']/cpp['medianSeconds']
            result['latencyPassed'] = result['speedup'] >= (2. if workload == 'generate' else 1/1.1)
            allowance = 16777216 if workload == 'small' else 0
            result['memoryRegressionPassed'] = cpp['medianPeakRssBytes'] <= ref['medianPeakRssBytes']*1.1 + allowance
            result['nativeExecutionPassed'] = all(
                r['backend'].get('selected') == 'cpp' and
                r['backend'].get('counters', {}).get('bindingCalls', 0) > 0
                for r in rows['cpp'])
            if workload == 'generate':
                result['productionMultiItemBatchPassed'] = all(
                    r['backend'].get('counters', {}).get('maxBatchItems', 0) > 1
                    for r in rows['cpp'])
        receipt['workloads'][workload] = result
        receipt['passed'] &= (result['exactEquivalent'] and result['memoryHardCapPassed'] and
            result.get('latencyPassed', True) and result.get('memoryRegressionPassed', True) and
            result.get('preportEquivalent', True) and result.get('nativeExecutionPassed', True) and
            result.get('productionMultiItemBatchPassed', True))
        write_json(args.output/'summary.json', receipt)
        print(workload, json.dumps({k: v for k, v in result.items() if k != 'backends'}), flush=True)
    write_json(args.output/'summary.json', receipt)
    if not receipt['passed']:
        raise SystemExit('One or more frozen gates failed; all samples retained')


if __name__ == '__main__':
    main()
