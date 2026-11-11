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
import shutil
import statistics
import subprocess
import sys
import sysconfig
import time

ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / 'manifests/experiments/cpp-acceleration-v1.json'
SUPPLEMENT_PROPOSAL = ROOT / 'manifests/experiments/cpp-acceleration-simulation-v1.proposed.json'
SUPPLEMENT_MANIFEST = ROOT / 'manifests/experiments/cpp-acceleration-simulation-v1.json'
PLANNING_WORKLOADS = ('small', 'generate', 'standardized-refusal', 'stress')
SUPPLEMENT_WORKLOADS = ('simulation-steps', 'rl-episodes')
THREADS = ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS',
           'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS', 'BLIS_NUM_THREADS')


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')


def backend_snapshot():
    try:
        from resectionlab.native_backend import backend_identity
        return backend_identity()
    except ModuleNotFoundError as exc:
        if exc.name != 'resectionlab.native_backend' or os.environ.get('RESSECTIONLAB_NUMERICAL_BACKEND') == 'cpp':
            raise
        return {'selected': 'reference', 'implementation': 'pre-port-baseline'}


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


def perform(workload, profile=None):
    import numpy as np
    from resectionlab.core import semantic_digest
    from resectionlab.generated_cpp_fixture import cranial_generate_fixture, continuum_fixture
    from resectionlab.surgical_cranial_planning import compare_cranial_strategies
    if profile:
        profile.enable()
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
        if profile:
            profile.disable()
        diagnostics['backendAfterTimedWork'] = backend_snapshot()
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
    if profile:
        profile.disable()
    return value, {'fixtureSeconds': setup, 'computeSeconds': elapsed,
                   'completeSeconds': setup + elapsed}, diagnostics


def exact_bits(value):
    """Typed comparison projection; this audit runs after the measured work."""
    import numpy as np
    if isinstance(value, np.ndarray):
        return ['ndarray', value.dtype.str, list(value.shape), value.tobytes(order='C').hex()]
    if isinstance(value, np.generic):
        return ['numpy-scalar', value.dtype.str, exact_bits(value.item())]
    if isinstance(value, float):
        return ['float', value.hex()]
    if isinstance(value, dict):
        return {key: exact_bits(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [exact_bits(item) for item in value]
    return value


def qualify_supplement_row(row):
    """Actual exercised physics, checked after timing instead of guessed passes."""
    import numpy as np
    engine = row['env'].engine if 'env' in row else row['engine']
    metrics = engine.metrics()
    family = row['family']
    if 'env' in row:
        env = row['env']
        assert env.terminated or env.truncated
        if family == 'rl-protected-attempt-episode':
            protected = int(np.ravel_multi_index((9, 9, 8), engine.scene.shape))
            assert protected not in engine.state['removed'] and env.measurements['protected_damage_mm3'] == 0
            assert env.truncated and env.termination_reason == 'attempt_limit'
            assert any(r['returned'][4]['outcome']['status'] in ('partial', 'rejected') for r in row['attempts'])
        else:
            assert env.terminated and not env.truncated and env.termination_reason == 'target_fraction_reached'
            assert row['attempts'][0]['returned'][4]['outcome']['status'] == 'rejected'
            assert any(r['returned'][4]['reward_terms']['tumor_removal'] > 0 for r in row['attempts'])
            assert row['attempts'][-1]['returned'][4]['reward_terms']['tumor_removal'] == 0
            assert env._cleanup_state()['pending'] == []
        return {'terminationReason': env.termination_reason, 'terminated': env.terminated,
                'truncated': env.truncated, 'measurements': env.measurements}
    steps = [r['step'] for r in row['attempts']]
    receipts = [r['receipt'] for r in steps if 'receipt' in r]
    statuses = [r['status'] for r in receipts]
    facts = {'receiptStatuses': statuses, 'exceptions': [r['exception'] for r in steps if 'exception' in r]}
    assert len(receipts) == len(steps), facts
    if family == 'simulation-target-episode':
        assert steps[0]['receipt']['status'] == 'rejected'
        assert len(receipts) == len(steps) and metrics['targetRemovedMm3'] > 0
    elif family == 'simulation-drill-thermal':
        state = engine.state['drillingV2']
        assert {'accepted', 'partial', 'rejected'} <= set(statuses)
        assert metrics['drillRemovedVolumeMm3'] > 0 and engine.state['debrisMm3'] > 0
        assert state['inputEnergyJ'] > 0 and state['cuttingWorkJ'] > 0
        assert max(state['temperatureC'].values()) > engine.scene.properties['tissuePhysicsV2']['bone']['ambientTemperatureC']
        facts['drilling'] = state
    elif family == 'simulation-cusa-v3':
        first = steps[2]['receipt']
        assert first['status'] == 'partial' and first['reasonCode'] == 'TISSUE_ASPIRATION_IN_PROGRESS'
        assert first['metrics']['cusaPendingCapacityMm3'] > 0
        assert metrics['cusaAspiratedSolidMm3'] == 1. and metrics['cusaPendingCells'] == 0
        residual = metrics['cusaInputCapacityMm3'] - sum(metrics[k] for k in (
            'cusaAspiratedSolidMm3', 'cusaPendingCapacityMm3', 'cusaUnusedCapacityMm3', 'cusaDiscardedCapacityMm3'))
        assert abs(residual) < 1e-12
        facts['capacityConservationResidualMm3'] = residual
    elif family in ('simulation-vascular', 'simulation-vascular-wound'):
        from resectionlab.surgical_vascular_graph import PROPERTY, metrics as vascular_metrics
        vascular = vascular_metrics(engine)
        state = engine.state[PROPERTY]
        assert statuses == ['accepted'] * len(steps)
        assert vascular['boundaryInMm3'] > 0 or vascular['boundaryOutMm3'] > 0
        assert any(abs(v) > 0 for v in state['edgeFlowMm3PerSecond'].values())
        assert abs(vascular['conservationResidualMm3']) < 1e-9
        assert any(r['metrics']['vesselPhysicsV2']['compressedEdgeCount'] > 0 for r in receipts)
        if family == 'simulation-vascular':
            assert state['edges']['artery']['occlusionFraction'] == 1. and vascular['generatedMm3'] == 0
            assert receipts[5]['metrics']['vesselPhysicsV2']['compressedEdgeCount'] > 0
            assert receipts[6]['metrics']['vesselPhysicsV2']['compressedEdgeCount'] == 0
            assert receipts[8]['metrics']['vesselPhysicsV2']['occludedEdgeCount'] > 0
        else:
            assert vascular['cutEdgeCount'] == 1 and vascular['woundPortCount'] > 0
            assert vascular['generatedMm3'] > 0 and vascular['emittedMm3'] > 0
            held = receipts[9]['metrics']['vesselPhysicsV2']['generatedMm3']
            assert receipts[10]['metrics']['vesselPhysicsV2']['generatedMm3'] == held
            assert receipts[11]['metrics']['vesselPhysicsV2']['generatedMm3'] > held
            facts['heldSuppressedBleed'] = True
        facts['vascular'] = vascular
    else:
        assert family == 'simulation-continuum'
        state = engine.state['brainContinuumV1']
        assert statuses == ['accepted'] * len(steps)
        assert len({r['stateId'] for r in receipts}) == len(steps)
        assert state['activeElements'] > 0 and state['activeDofs'] > 0 and state['maxGradientNorm'] > 0
        assert state['cornerDisplacementMm'] and state['memoryDisplacementMm'][0]
        assert state['dissipatedEnergyNmm'] > 0
        assert sum(v['strainSquaredVolumeTimeMm3Seconds'] for v in state['exposureBySourceLabel'].values()) > 0
        facts['continuum'] = {k: state[k] for k in ('activeElements', 'activeDofs', 'maxGradientNorm',
            'dissipatedEnergyNmm', 'exposureBySourceLabel')}
    return facts


def perform_supplement(workload, profile=None):
    """Whole fixed episodes; endogenous hashes stay timed, audit hashes do not."""
    from resectionlab.generated_cpp_supplement import (
        SIMULATION_FAMILIES, simulation_fixture, rl_fixture)
    raw = []
    fixture_seconds = 0.
    family_times = []
    if profile:
        profile.enable()
    started = time.perf_counter()
    if workload == 'simulation-steps':
        for family in SIMULATION_FAMILIES:
            family_start = time.perf_counter()
            engine, actions = simulation_fixture(family)
            fixture_seconds += time.perf_counter() - family_start
            attempts = []
            for action in actions:
                row = {'requestedAction': action}
                operations = [('preview', lambda: engine.preview(action, include_view=False))] if (
                    family == 'simulation-drill-thermal') else []
                operations.append(('step', lambda: engine.step(action)))
                for name, operation in operations:
                    try:
                        row[name] = {'receipt': operation()}
                    except (ValueError, TypeError) as exc:
                        row[name] = {'exception': {'type': type(exc).__name__, 'message': str(exc)}}
                attempts.append(row)
            raw.append({'family': family, 'engine': engine, 'attempts': attempts})
            family_times.append({'family': family, 'completeSeconds': time.perf_counter() - family_start})
    else:
        assert workload == 'rl-episodes'
        episodes = [('rl-success-episode', 318), ('rl-protected-attempt-episode', 318),
                    *[('rl-independent-instances', seed) for seed in (318, 319, 320, 321)]]
        for family, seed in episodes:
            family_start = time.perf_counter()
            factory_family = 'rl-success-episode' if family == 'rl-independent-instances' else family
            env, actions, seed = rl_fixture(factory_family, seed)
            initial = env.reset(seed=seed)
            fixture_seconds += time.perf_counter() - family_start
            attempts = []
            for action in actions:
                encoded = env.encode_action(action)
                attempts.append({'requestedAction': action, 'encodedAction': encoded,
                                 'returned': env.step(encoded)})
            raw.append({'family': family, 'factoryFamily': factory_family, 'seed': seed,
                        'env': env, 'initial': initial, 'attempts': attempts})
            family_times.append({'family': family, 'seed': seed,
                                 'completeSeconds': time.perf_counter() - family_start})
    elapsed = time.perf_counter() - started
    if profile:
        profile.disable()
    backend = backend_snapshot()
    timed_peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * (1 if sys.platform == 'darwin' else 1024)
    audit_started = time.perf_counter()
    projected = []
    work = {'requestedAttempts': sum(len(row['attempts']) for row in raw),
            'publicPreviewAttempts': sum('preview' in attempt for row in raw for attempt in row['attempts']),
            'episodes': len(raw) if workload == 'rl-episodes' else 0,
            'committedNativeActions': sum(len((row['engine'] if 'engine' in row else row['env'].engine).actions) for row in raw)}
    for row in raw:
        env = row.get('env')
        engine = row['engine'] if env is None else env.engine
        result = {'family': row['family'], 'attempts': row['attempts'],
            'structuralQualification': qualify_supplement_row(row),
            'source': {'descriptor': engine.scene.descriptor(), 'labels': engine.scene.labels,
                       'affine': engine.scene.affine, 'properties': engine.scene.properties},
            'snapshot': engine.snapshot(), 'history': engine.history, 'metrics': engine.metrics()}
        if env is not None:
            assert env.terminated or env.truncated, 'Literal RL trace did not complete its declared episode'
            assert all(env.observation_space.contains(attempt['returned'][0]) for attempt in row['attempts'])
            snapshot = env.snapshot()
            restored, _, _ = rl_fixture(row['factoryFamily'], row['seed'])
            restored.restore(json.loads(json.dumps(snapshot, allow_nan=False)))
            assert exact_bits(restored.snapshot()) == exact_bits(snapshot), 'RL episode restore differs'
            assert exact_bits(restored._observe()) == exact_bits(env._observe()), 'RL restored observation differs'
            result.update(seed=row['seed'], initial=row['initial'], episodeSnapshot=snapshot,
                          terminationReason=env.termination_reason, restoreVerified=True)
        elif row['family'] == 'simulation-drill-thermal':
            for attempt in row['attempts']:
                assert attempt['preview'].keys() == attempt['step'].keys(), 'Preview/step validation classes differ'
                if 'receipt' in attempt['step']:
                    assert attempt['preview']['receipt']['stateId'] == attempt['step']['receipt']['stateId']
                else:
                    assert attempt['preview']['exception'] == attempt['step']['exception']
        projected.append(exact_bits(result))
    if workload == 'rl-episodes':
        import numpy as np
        independent = [row['env'] for row in raw if row['family'] == 'rl-independent-instances']
        for index, env in enumerate(independent):
            for other in independent[index+1:]:
                assert env.engine.state is not other.engine.state
                assert env.engine.state['removed'] is not other.engine.state['removed']
                assert env.engine.state['slots'] is not other.engine.state['slots']
                assert all(env.engine.state['slots'][slot] is not other.engine.state['slots'][slot] for slot in ('left', 'right'))
                assert env.np_random is not other.np_random
                assert env.action_space.np_random is not other.action_space.np_random
                assert not np.shares_memory(env.scene.labels, other.scene.labels)
            assert exact_bits(env.engine.snapshot()) == exact_bits(raw[0]['env'].engine.snapshot())
    value = {'schema': 'resectionlab.cpp-supplement-result.v1', 'work': work, 'families': projected}
    diagnostics = {'backendAfterTimedWork': backend, 'timedWorkPeakRssBytes': timed_peak,
        'familyTimings': family_times, 'auditProjectionAndRestoreSeconds': time.perf_counter() - audit_started,
        'hashTimingContract': 'Production semantic/proof/state hashes are endogenous and included. Typed projection, equality hashing, snapshot/restore checks and file writes are audit work excluded from elapsed/profile.',
        'noPolicyTraining': True, 'pytorchExecuted': False}
    return value, {'fixtureSeconds': fixture_seconds, 'computeSeconds': elapsed - fixture_seconds,
                   'completeSeconds': elapsed}, diagnostics


def worker(args):
    # Explicit path prevents an editable install in a neighboring environment
    # from silently supplying the wrong checkout.
    sys.path.insert(0, str(args.source_root.resolve()))
    import resectionlab
    from resectionlab.core import semantic_digest
    profile = cProfile.Profile() if args.profile else None
    if args.worker in SUPPLEMENT_WORKLOADS:
        value, timings, diagnostics = perform_supplement(args.worker, profile)
    else:
        value, timings, diagnostics = perform(args.worker, profile)
    # Registered-goal/hash checks can themselves use native JSON copies.
    backend = diagnostics.get('backendAfterTimedWork') or backend_snapshot()
    audit_hash_started = time.perf_counter()
    exact = canonical_result(value)
    exact_digest = semantic_digest(exact)
    diagnostics['auditEqualityHashSeconds'] = time.perf_counter() - audit_hash_started
    row = {'schema': 'resectionlab.cpp-acceleration-sample.v1', 'workload': args.worker,
           'backendRequested': args.backend, 'backend': backend,
           'diagnostics': diagnostics,
           'timings': timings, 'exactDigest': exact_digest, 'result': value,
           'peakRssBytes': resource.getrusage(resource.RUSAGE_SELF).ru_maxrss *
               (1 if sys.platform == 'darwin' else 1024),
           'environment': {'python': sys.version, 'executable': sys.executable,
               'machine': platform.machine(), 'platform': platform.platform(),
               'abi': sysconfig.get_config_var('SOABI'),
               'modulePath': str(Path(resectionlab.__file__).resolve()),
               'threads': {k: os.environ.get(k) for k in THREADS}}}
    if profile:
        stats = pstats.Stats(profile)
        for function in (backend_snapshot, canonical_result, exact_bits, qualify_supplement_row):
            code = function.__code__
            assert (code.co_filename, code.co_firstlineno, code.co_name) not in stats.stats, 'Timed profile includes audit work'
        row['profile'] = {'scopeVersion': 2, 'scope': 'timed fixture and public computation only',
            'auditExcluded': True, 'totalSeconds': stats.total_tt, 'functions': [
            {'file': k[0], 'line': k[1], 'function': k[2], 'calls': v[1],
             'selfSeconds': v[2], 'cumulativeSeconds': v[3]}
            for k, v in sorted(stats.stats.items(), key=lambda item: item[1][3], reverse=True)]}
        if args.worker in SUPPLEMENT_WORKLOADS:
            row['profile']['attribution'] = profile_attribution(stats)
        profile.dump_stats(args.output.with_suffix('.prof'))
    write_json(args.output, row)
    print(json.dumps({k: row[k] for k in ('workload', 'backendRequested', 'timings', 'exactDigest', 'peakRssBytes')}))


def run_sample(args, workload, backend, name, *, profile=False):
    output = args.output / (name + '.json')
    source = args.baseline_source if backend == 'reference' and args.baseline_source else args.source_root
    selected_backend = getattr(args, 'baseline_backend', 'reference') if backend == 'reference' else backend
    command = [sys.executable, str(Path(__file__).resolve()), '--worker', workload,
               '--backend', selected_backend, '--output', str(output), '--source-root', str(source)]
    if profile:
        command.append('--profile')
    env = dict(os.environ, RESSECTIONLAB_NUMERICAL_BACKEND=selected_backend, PYTHONPATH=str(source.resolve()))
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
    row['comparisonRole'] = backend
    write_json(output, row)
    return row


def summarize(rows):
    times = [r['timings']['completeSeconds'] for r in rows]
    result = {'medianSeconds': statistics.median(times), 'minSeconds': min(times),
            'maxSeconds': max(times), 'stdevSeconds': statistics.stdev(times) if len(times) > 1 else 0.,
            'sampleCount': len(times), 'samplesSeconds': times,
            'medianPeakRssBytes': statistics.median(r['processTreePeakRssBytes'] for r in rows),
            'peakRssBytes': max(r['processTreePeakRssBytes'] for r in rows)}
    if all('timedWorkPeakRssBytes' in r['diagnostics'] for r in rows):
        result['medianTimedWorkPeakRssBytes'] = statistics.median(r['diagnostics']['timedWorkPeakRssBytes'] for r in rows)
    return result


def profile_attribution(stats):
    stages = {}
    for (filename, _, function), value in stats.stats.items():
        name = Path(filename).name
        if filename == '~' and 'resectionlab._geometry_cpp' in function:
            stage = 'newCppKernelEntryPoints'
        elif filename == '~' and any(library in function for library in ('numpy', 'scipy', 'torch', 'SuperLU')):
            stage = 'knownCompiledNumericEntryPoints'
        elif name in ('core.py', 'copy.py') or 'sha256' in function:
            stage = 'productionHashingAndCopying'
        elif name in ('geometry.py', 'surgical_geometry_cache.py', 'surgical_motion.py'):
            stage = 'geometryAndMotionPythonAndDispatch'
        elif name.startswith('surgical_continuum') or name == 'surgical_deformation_v2.py':
            stage = 'continuumPythonAndDispatch'
        elif name in ('surgical_drilling_v2.py', 'surgical_cusa_v3.py', 'surgical_vessels_v2.py',
                      'surgical_vascular_graph.py', 'surgical_effects.py'):
            stage = 'physicalModelsPythonAndDispatch'
        elif name in ('surgical_rl_env.py', 'surgical_rl_rewards.py'):
            stage = 'rlObservationRewardAndCodecPython'
        elif '/numpy/' in filename or '/scipy/' in filename or '/torch/' in filename:
            stage = 'compiledLibraryInterfacesIncludingPythonWrappers'
        elif filename == '~':
            stage = 'builtinOrExtensionCalls'
        else:
            stage = 'otherPythonAndDispatch'
        stages[stage] = stages.get(stage, 0.) + value[2]
    return {'exclusiveStageSeconds': stages,
        'interpretation': 'Module buckets identify allocation/control and dispatch locations. Inclusive Python caller time can include compiled NumPy/SciPy work; it is not proof of interpreted arithmetic. Raw extension/builtin entries and cumulative call paths remain available.',
        'auditExcluded': True, 'pytorchExecuted': False}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path)
    parser.add_argument('--source-root', type=Path, default=ROOT/'src')
    parser.add_argument('--baseline-source', type=Path, help='Captured original src for pre-port comparison')
    parser.add_argument('--baseline-backend', choices=('reference', 'cpp'), default='reference',
                        help='Supplement only: use captured pre-pass C++ as the baseline comparison role')
    parser.add_argument('--supplemental', action='store_true', help='Select the separate simulation/RL protocol')
    parser.add_argument('--freeze-supplement', action='store_true', help='After successful qualification, freeze a new supplement manifest')
    parser.add_argument('--capture-supplement-baseline', type=Path,
                        help='After qualification, capture current verified CPP source/binary before new ports')
    parser.add_argument('--workload', nargs='+', choices=(*PLANNING_WORKLOADS, *SUPPLEMENT_WORKLOADS))
    parser.add_argument('--mode', choices=('compare', 'baseline', 'profile', 'qualify'), default='compare')
    parser.add_argument('--repeats', type=int, default=5)
    parser.add_argument('--warmups', type=int, default=1)
    parser.add_argument('--worker', choices=(*PLANNING_WORKLOADS, *SUPPLEMENT_WORKLOADS))
    parser.add_argument('--backend', choices=('reference', 'cpp', 'auto'), default='reference')
    parser.add_argument('--profile', action='store_true')
    parser.add_argument('--export-case', type=Path, help='Save a pristine generated source for actual desktop Generate')
    args = parser.parse_args()
    if args.output is None and not (args.capture_supplement_baseline or args.export_case):
        parser.error('--output is required for benchmark, profile and qualification results')
    if args.capture_supplement_baseline:
        if not args.supplemental or not SUPPLEMENT_MANIFEST.exists():
            parser.error('Capture requires an already qualified supplementary manifest')
        sys.path.insert(0, str(args.source_root.resolve()))
        identity = backend_snapshot()
        if identity.get('selected') != 'cpp':
            parser.error('Pre-pass capture requires the working C++ backend')
        if Path(identity['binaryPath']).resolve().parent != (args.source_root/'resectionlab').resolve():
            parser.error('Loaded C++ binary is outside the requested source root')
        captured = args.capture_supplement_baseline.resolve()
        captured.mkdir(parents=True, exist_ok=False)
        shutil.copytree(args.source_root.resolve(), captured/'src',
                        ignore=shutil.ignore_patterns('__pycache__', '*.pyc', '.DS_Store'))
        shutil.copy2(SUPPLEMENT_MANIFEST, captured/SUPPLEMENT_MANIFEST.name)
        write_json(captured/'capture.json', {
            'sourceCommit': subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip(),
            'backend': identity, 'runnerSha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            'manifestSha256': hashlib.sha256(SUPPLEMENT_MANIFEST.read_bytes()).hexdigest(),
            'sourceFilesSha256': {str(path.relative_to(captured/'src')): hashlib.sha256(path.read_bytes()).hexdigest()
                                  for path in sorted((captured/'src').rglob('*')) if path.is_file()}})
        print(captured)
        return
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
    if args.baseline_backend != 'reference' and not args.supplemental:
        parser.error('--baseline-backend cpp is only valid for the separate supplementary protocol')
    if args.mode == 'qualify' and not args.supplemental:
        parser.error('Qualification is only valid for the supplementary protocol')
    if args.freeze_supplement and (args.mode != 'qualify' or SUPPLEMENT_MANIFEST.exists()):
        parser.error('--freeze-supplement requires a new supplementary qualification; never overwrite a frozen manifest')
    args.workload = args.workload or list(SUPPLEMENT_WORKLOADS if args.supplemental else PLANNING_WORKLOADS)
    if any((workload in SUPPLEMENT_WORKLOADS) != args.supplemental for workload in args.workload):
        parser.error('Supplementary workloads require --supplemental and cannot replace the original planning workload')
    if args.repeats < 5 or args.warmups < 1:
        parser.error('Frozen protocol requires at least five repetitions and one warmup')
    args.output.mkdir(parents=True, exist_ok=False)
    manifest_path = (SUPPLEMENT_MANIFEST if SUPPLEMENT_MANIFEST.exists() else SUPPLEMENT_PROPOSAL) if args.supplemental else MANIFEST
    manifest = json.loads(manifest_path.read_text())
    if args.supplemental and args.mode != 'qualify' and manifest.get('status') != 'qualified-workloads-frozen':
        parser.error('Qualify and freeze the supplementary workloads before performance/profile runs')
    receipt = {'manifestSha256': hashlib.sha256(manifest_path.read_bytes()).hexdigest(),
               'manifest': manifest, 'mode': args.mode, 'workloads': {}, 'passed': True}
    receipt['comparisonRoles'] = {
        'reference': {'meaning': 'baseline', 'selectedBackend': args.baseline_backend,
                      'sourceRoot': str((args.baseline_source or args.source_root).resolve())},
        'cpp': {'meaning': 'current', 'selectedBackend': 'cpp', 'sourceRoot': str(args.source_root.resolve())}}
    write_json(args.output/'summary.json', receipt)
    for workload in args.workload:
        if args.mode == 'qualify':
            try:
                rows = {backend: run_sample(args, workload, backend, workload+'-qualify-'+backend)
                        for backend in ('reference', 'cpp')}
            except BaseException as exc:
                receipt['passed'] = False
                receipt['workloads'][workload] = {'qualificationOnly': True, 'error': str(exc)}
                write_json(args.output/'summary.json', receipt)
                raise
            matches = rows['reference']['exactDigest'] == rows['cpp']['exactDigest']
            native = rows['cpp']['backend']
            native_passed = native.get('selected') == 'cpp' and native.get('counters', {}).get('bindingCalls', 0) > 0
            receipt['workloads'][workload] = {'qualificationOnly': True, 'exactEquivalent': matches,
                'nativeExecutionPassed': native_passed,
                'exactDigest': rows['reference']['exactDigest'], 'work': rows['reference']['result']['work'],
                'families': [{key: family[key] for key in ('family', 'structuralQualification', 'terminationReason') if key in family}
                             for family in rows['reference']['result']['families']],
                'backendIdentities': {backend: row['backend'] for backend, row in rows.items()}}
            receipt['passed'] &= matches and native_passed
            write_json(args.output/'summary.json', receipt)
            print(workload, 'qualification exact:', matches, flush=True)
            continue
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
        if args.supplemental:
            frozen = manifest['qualification']['workloads'][workload]['exactDigest']
            result['frozenExactEquivalent'] = all(r['exactDigest'] == frozen for group in rows.values() for r in group)
            result['throughput'] = {backend: {
                'attemptsPerSecond': group[0]['result']['work']['requestedAttempts']/result['backends'][backend]['medianSeconds'],
                'committedNativeActionsPerSecond': group[0]['result']['work']['committedNativeActions']/result['backends'][backend]['medianSeconds'],
                'episodesPerSecond': group[0]['result']['work']['episodes']/result['backends'][backend]['medianSeconds']}
                for backend, group in rows.items()}
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
            if args.supplemental:
                result['memoryRegressionPassed'] = cpp['medianTimedWorkPeakRssBytes'] <= ref['medianTimedWorkPeakRssBytes']*1.1
                result['memoryComparisonBasis'] = 'Pre-audit timedWorkPeakRssBytes; whole-process audit/restore peak is diagnostic and remains under the hard cap.'
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
            result.get('productionMultiItemBatchPassed', True) and result.get('frozenExactEquivalent', True))
        write_json(args.output/'summary.json', receipt)
        print(workload, json.dumps({k: v for k, v in result.items() if k != 'backends'}), flush=True)
    write_json(args.output/'summary.json', receipt)
    if args.freeze_supplement and receipt['passed']:
        assert set(args.workload) == set(SUPPLEMENT_WORKLOADS), 'Freeze requires every proposed suite'
        frozen = {**manifest, 'status': 'qualified-workloads-frozen',
            'schema': 'resectionlab.cpp-acceleration-supplement.v1',
            'qualification': {'workloads': receipt['workloads'],
                'sourceCommit': subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip(),
                'fixtureSourceSha256': hashlib.sha256((args.source_root/'resectionlab/generated_cpp_supplement.py').read_bytes()).hexdigest(),
                'receipt': str((args.output/'summary.json').resolve())}}
        write_json(SUPPLEMENT_MANIFEST, frozen)
    if not receipt['passed']:
        raise SystemExit('One or more frozen gates failed; all samples retained')


if __name__ == '__main__':
    main()
