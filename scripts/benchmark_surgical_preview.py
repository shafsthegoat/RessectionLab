#!/usr/bin/env python3
"""Paired, headless preview benchmark; source/checkpoint preparation is untimed.

Run a suite twice with distinct --source-root/--label values. Each sample cell
and each cProfile cell gets its own fresh process, including the same actual
BridgeSession admission. Nothing opens a desktop or writes an input case.
Evidence belongs under ignored testcases/, never beside acquired source data.
"""
from __future__ import annotations

import argparse
import cProfile
import hashlib
import json
import os
from pathlib import Path
import platform
import pstats
import resource
import subprocess
import sys
import tempfile
import time


def encoded(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()


def digest(value):
    return hashlib.sha256(encoded(value)).hexdigest()


def file_hash(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def evidence_path(value):
    root = next(parent for parent in Path(__file__).resolve().parents
                if (parent / 'src/resectionlab').is_dir() and (parent / 'testcases').is_dir())
    if Path(value).is_symlink():
        raise ValueError('Benchmark output destinations must not be symbolic links')
    destination = Path(value).resolve()
    if destination == root / 'testcases' or not destination.is_relative_to((root / 'testcases').resolve()):
        raise ValueError('Benchmark outputs must be under this checkout\'s ignored testcases/ directory')
    return destination


def prepare(args, transfer_dir):
    from resectionlab.desktop_bridge import BridgeSession, _Request
    from resectionlab.imaging import load_case, read_case_artifacts

    session = BridgeSession(Path(transfer_dir))
    request = _Request('preview-benchmark-preparation')
    execute = lambda operation, value: session.execute(operation, value, request, lambda *_: None)
    if args.dataset == 'numerical-fixture':
        case_hash = execute('createSurgicalFixture', {})['caseHash']
        execute('surgicalRehearsal', {'caseHash': case_hash, 'command': 'create'})
        count = 4 if args.prefix == 'short' else 122
        # This generated numerical control measures ancestry scaling. These are
        # genuine accepted inspect transitions, not a patient/effect benchmark.
        for _ in range(count):
            current = session.surgical_sessions[case_hash][1]
            result = execute('surgicalRehearsal', {'caseHash': case_hash,
                'command': 'step', 'expectedStateId': current.state_id,
                'action': {'slot': 'left', 'operation': 'inspect', 'duration': .1}})
            assert result['outcome']['status'] == 'accepted'
        source = {'kind': 'fixed generated numerical fixture; no patient claim',
                  'generator': 'actual createSurgicalFixture handler',
                  'prefixRecipe': f'{count} accepted left inspect duration=0.1 actions'}
    else:
        path = Path(args.case).resolve()
        case = load_case(path)
        artifacts = read_case_artifacts(path)
        record = artifacts['surgical_toolbox']
        branch = 'branch-7'
        checkpoint_hash = digest(record['branches'][branch])
        record['branchId'] = branch
        record['position'] = 16 if args.prefix == 'short' else 122
        case_hash = session._install_case(case, artifacts, request)['caseHash']
        # Canonical full checkpoint restore and prefix seek happen here, before
        # any measured preview. No hand-edited patient effect state is admitted.
        execute('surgicalRehearsal', {'caseHash': case_hash, 'command': 'inspect'})
        source = {'kind': 'existing acquired-source development case',
                  'path': str(path), 'fileSha256': file_hash(path),
                  'caseSemanticHash': case.semantic_hash, 'branchId': branch,
                  'originalCheckpointSha256': checkpoint_hash,
                  'originalEndpointStateId': record['branches'][branch]['stateId'],
                  'prefixRecipe': 'canonical restore of branch-7, actual prefix seek',
                  'sourceRefs': [str(ref) for ref in case.source_refs]}
    engine = session.surgical_sessions[case_hash][1]
    record = session.surgical_sessions[case_hash][2]
    assert len(engine.actions) == (4 if args.dataset == 'numerical-fixture' and args.prefix == 'short'
                                  else 16 if args.prefix == 'short' else 122)
    return session, engine, case_hash, source, record


def worker(args):
    # This precedes all project imports, so frozen-before imports cannot resolve
    # the concurrently edited primary checkout by accident.
    source_root = Path(args.source_root).resolve()
    sys.path.insert(0, str(source_root))
    import numpy as np
    from resectionlab.desktop_bridge import _Request
    from resectionlab.core import thaw_json
    import resectionlab.surgical_engine as engine_module
    import resectionlab.surgical_bridge as bridge_module

    destination = evidence_path(args.output)
    for target in (destination, destination.with_suffix('.response.json'),
                   destination.with_suffix('.pstats'), destination.with_suffix('.profile.txt')):
        if target.exists() or target.is_symlink():
            raise FileExistsError(f'Preserve prior evidence; choose a fresh output destination: {target}')
    destination.parent.mkdir(parents=True, exist_ok=True)
    preparation_start = time.perf_counter()
    with tempfile.TemporaryDirectory(prefix='preview-transfer-', dir=destination.parent) as directory:
        session, engine, case_hash, source, record = prepare(args, directory)
        preparation_seconds = time.perf_counter() - preparation_start
        action = {'slot': 'left', 'operation': 'inspect', 'duration': .1}
        expected_state = engine.state_id
        request = _Request('preview-benchmark')
        payload = {'caseHash': case_hash, 'command': 'preview',
                   'expectedStateId': expected_state, 'action': action}
        if args.mode == 'direct-engine':
            invoke = lambda: engine.preview(action)
        else:
            invoke = lambda: session.execute('surgicalRehearsal', payload, request, lambda *_: None)
        baseline = {'snapshot': digest(engine.snapshot()), 'history': digest(engine.history),
                    'artifacts': digest(thaw_json(session.cases[case_hash].artifacts)),
                    'record': digest(record)}
        response = None
        for _ in range(args.warmups):
            response = invoke()
            assert response['status'] == 'accepted'
        samples = []
        hashes = []
        profiler = cProfile.Profile() if args.profile else None
        for _ in range(args.repeats):
            if profiler:
                profiler.enable()
            start = time.perf_counter_ns()
            response = invoke()
            duration_ns = time.perf_counter_ns() - start
            if profiler:
                profiler.disable()
            samples.append(duration_ns / 1e6)
            assert response['status'] == 'accepted'
            assert response['stateId'] != expected_state
            hashes.append(digest(response))  # wire encoding is outside timer
        assert len(set(hashes)) == 1, 'Preview responses changed across samples'
        after = {'snapshot': digest(engine.snapshot()), 'history': digest(engine.history),
                 'artifacts': digest(thaw_json(session.cases[case_hash].artifacts)),
                 'record': digest(record)}
        assert baseline == after, 'Preview mutated committed engine/archive'
        assert session.surgical_sessions[case_hash][1] is engine
        if source.get('fileSha256'):
            assert file_hash(source['path']) == source['fileSha256'], 'Input case changed'
        response_bytes = encoded(response)
        response_path = destination.with_suffix('.response.json')
        response_path.write_bytes(response_bytes)
        profile_rows = []
        if profiler:
            profiler.dump_stats(str(destination.with_suffix('.pstats')))
            stats = pstats.Stats(profiler)
            for (filename, line, name), (primitive, calls, own, cumulative, callers) in stats.stats.items():
                if ('copy' in name or 'clone' in name or filename.endswith('/copy.py')):
                    profile_rows.append({'file': filename, 'line': line, 'function': name,
                        'primitiveCalls': primitive, 'calls': calls,
                        'selfSeconds': own, 'cumulativeSeconds': cumulative})
            profile_rows.sort(key=lambda row: row['cumulativeSeconds'], reverse=True)
            with destination.with_suffix('.profile.txt').open('w') as stream:
                pstats.Stats(profiler, stream=stream).sort_stats('cumulative').print_stats(60)
        metadata = {'label': args.label, 'dataset': args.dataset, 'prefix': args.prefix,
            'mode': args.mode, 'profileOnly': args.profile, 'sourceRoot': str(source_root),
            'source': source, 'python': sys.version, 'executable': sys.executable,
            'platform': platform.platform(), 'machine': platform.machine(),
            'numpyVersion': np.__version__, 'preparationSeconds': preparation_seconds,
            'action': action, 'committedStateId': expected_state,
            'committedSnapshotSha256': baseline['snapshot'],
            'actionsCount': len(engine.actions), 'historyCount': len(engine.history),
            'stateJsonBytes': len(encoded(engine.state)),
            'historyJsonBytes': len(encoded(engine.history)),
            'removedCellCount': len(engine.state['removed']),
            'artifactJsonBytes': len(encoded(thaw_json(session.cases[case_hash].artifacts))),
            'warmups': args.warmups, 'sampleCount': len(samples), 'samplesMs': samples,
            'p50Ms': float(np.percentile(samples, 50)), 'p95Ms': float(np.percentile(samples, 95)),
            'quantileMethod': 'numpy.percentile linear interpolation',
            'responseSha256': hashes[0], 'responseJsonBytes': len(response_bytes),
            'responseHistoryCount': len(response['view']['history']),
            'fullCase': bool(record.get('fullCase')), 'committedStateUnchanged': baseline == after,
            'responsePath': str(response_path.resolve()),
            'ruMaxrssBytes': resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * (1 if sys.platform == 'darwin' else 1024),
            'rssMeaning': 'fresh worker lifetime OS peak incl identical admission, preparation, preview and outside-timer response encoding; one scenario/implementation per process',
            'timingMeaning': 'actual Python call to return full response/view; preparation, warmup and JSON encoding excluded; bridge includes request/result budgets; direct includes default complete history',
            'sourceFiles': {str(Path(module.__file__).resolve()): file_hash(module.__file__)
                            for module in (engine_module, bridge_module)},
            'copyProfileRows': profile_rows}
        destination.write_text(json.dumps(metadata, indent=2) + '\n')
        print(json.dumps({key: metadata[key] for key in ('label', 'dataset', 'prefix', 'mode', 'profileOnly', 'p50Ms', 'p95Ms', 'ruMaxrssBytes')}), flush=True)


def suite(args):
    out = evidence_path(args.output)
    # A whole suite owns a fresh directory. Refuse even an empty existing one
    # rather than overwriting hardware, partial worker results or old summaries.
    out.mkdir(parents=True, exist_ok=False)
    hardware = {'platform': platform.platform(), 'python': sys.version,
                'cpuCount': os.cpu_count(), 'clock': time.get_clock_info('perf_counter').__dict__}
    if sys.platform == 'darwin':
        hardware['sysctl'] = subprocess.check_output(['sysctl', 'hw.model', 'hw.memsize', 'hw.physicalcpu', 'hw.logicalcpu', 'machdep.cpu.brand_string'], text=True)
    (out / 'hardware.json').write_text(json.dumps(hardware, indent=2) + '\n')
    results = []
    for dataset in ('numerical-fixture', 'source-backed'):
        for prefix in ('short', 'long'):
            for mode in ('direct-engine', 'bridge-handler'):
                profiles = (False, True) if args.profiles and prefix == 'long' else (False,)
                for profile in profiles:
                    name = f'{dataset}-{prefix}-{mode}' + ('-profile' if profile else '')
                    path = out / f'{name}.json'
                    command = [sys.executable, str(Path(__file__).resolve()), '--worker',
                        '--source-root', str(Path(args.source_root).resolve()), '--label', args.label,
                        '--dataset', dataset, '--prefix', prefix, '--mode', mode,
                        '--case', str(Path(args.case).resolve()), '--output', str(path.resolve()),
                        '--warmups', str(args.warmups), '--repeats', str(3 if profile else args.repeats)]
                    if profile:
                        command.append('--profile')
                    environment = dict(os.environ, OPENBLAS_NUM_THREADS='1', OMP_NUM_THREADS='1',
                                       MKL_NUM_THREADS='1', VECLIB_MAXIMUM_THREADS='1', PYTHONHASHSEED='0')
                    subprocess.run(command, env=environment, check=True)
                    results.append(json.loads(path.read_text()))
                    (out / 'summary.json').write_text(json.dumps(results, indent=2) + '\n')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--worker', action='store_true')
    parser.add_argument('--source-root', default='src')
    parser.add_argument('--label', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--case', default='testcases/cases/UCSF-PDGM-0007-operative-demo.ressectionlab')
    parser.add_argument('--dataset', choices=('numerical-fixture', 'source-backed'))
    parser.add_argument('--prefix', choices=('short', 'long'))
    parser.add_argument('--mode', choices=('direct-engine', 'bridge-handler'))
    parser.add_argument('--warmups', type=int, default=3)
    parser.add_argument('--repeats', type=int, default=25)
    parser.add_argument('--profile', action='store_true')
    parser.add_argument('--profiles', action='store_true')
    args = parser.parse_args()
    if args.warmups < 1 or args.repeats < 1:
        parser.error('Positive warmup/sample counts required')
    worker(args) if args.worker else suite(args)


if __name__ == '__main__':
    main()
