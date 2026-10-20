#!/usr/bin/env python3
"""Supplemental real bridge cursor timings and separately profiled attribution.

Run with PYTHONPATH selecting the frozen baseline or current src. This is a
backend diagnostic; the production foreground Electron benchmark remains the
end-to-end display measurement. Output and transfer files belong in testcases/.
"""
import argparse
import cProfile
from datetime import datetime, timezone
import json
from pathlib import Path
import platform
import pstats
import resource
import statistics
import sys
import time

import numpy as np

from resectionlab.core import semantic_digest, thaw_json
from resectionlab.desktop_bridge import BridgeSession, _Request
from resectionlab.surgical_engine import SurgicalEngine


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--case', required=True, type=Path)
    parser.add_argument('--output', required=True, type=Path)
    parser.add_argument('--history', type=int, choices=(12, 48), default=12)
    parser.add_argument('--repetitions', type=int, default=3)
    parser.add_argument('--label', required=True)
    args = parser.parse_args()
    if 'testcases' not in args.output.resolve().parts:
        parser.error('Place all generated benchmark data under testcases/')
    args.output.mkdir(parents=True, exist_ok=True)
    report = {'schema': 'resectionlab.rehearsal-state-profile.v1', 'label': args.label,
        'recordedAt': datetime.now(timezone.utc).isoformat(), 'casePath': str(args.case.resolve()),
        'python': sys.version, 'platform': platform.platform(), 'numpy': np.__version__,
        'implementation': str(Path(sys.modules[SurgicalEngine.__module__].__file__).resolve()),
        'historySteps': args.history, 'repetitions': args.repetitions,
        'measurement': 'Python BridgeSession handler wall time; excludes Electron, GPU and screen presentation. '
            'cProfile attribution is a separate overhead-bearing run; nested inclusive stages overlap.',
        'seedActions': 'left inspect, duration0.1 simulated seconds; generated workload on acquired anatomy, no trained policy',
        'checks': [], 'seeks': [], 'stages': []}
    session = BridgeSession(args.output / 'transfers')
    def call(operation, body):
        return session.execute(operation, body, _Request(operation), lambda *_: None)
    try:
        start = time.perf_counter()
        source = call('loadCase', {'path': str(args.case.resolve())})
        report['loadMs'] = (time.perf_counter() - start) * 1000
        identity = source['caseHash']
        report['source'] = {key: source[key] for key in ('caseHash', 'caseId', 'shape', 'affine', 'frame')}
        report['source']['enabledLayers'] = [layer['name'] for layer in source['compartments']]
        result = call('surgicalRehearsal', {'caseHash': identity, 'command': 'inspect'})
        if result is None:
            result = call('surgicalRehearsal', {'caseHash': identity, 'command': 'create'})
        if result['position'] != 0 or any(row['steps'] for row in result['branches']):
            raise ValueError('Use an intact prepared source; this benchmark will not reset saved history')
        states = {0: semantic_digest(session.surgical_sessions[identity][1].state)}
        state_ids = {0: result['view']['stateId']}
        seed_ms = []
        for position in range(1, args.history + 1):
            start = time.perf_counter()
            result = call('surgicalRehearsal', {'caseHash': identity, 'command': 'step',
                'expectedStateId': result['view']['stateId'],
                'action': {'slot': 'left', 'operation': 'inspect', 'duration': .1}})
            seed_ms.append((time.perf_counter() - start) * 1000)
            if result['outcome']['status'] != 'accepted':
                raise ValueError(result['outcome'])
            states[position] = semantic_digest(session.surgical_sessions[identity][1].state)
            state_ids[position] = result['view']['stateId']
        report['seedActionMs'] = {'median': statistics.median(seed_ms), 'p95': float(np.percentile(seed_ms, 95)), 'samples': seed_ms}
        record = thaw_json(session.cases[identity].artifacts)['surgical_toolbox']
        report['archiveBytes'] = len(json.dumps(record, separators=(',', ':'), allow_nan=False).encode())
        endpoint = session.surgical_sessions[identity][1]
        independently_restored = SurgicalEngine.restore(endpoint.scene, record['branches'][record['branchId']])
        assert independently_restored.state == endpoint.state
        report['checks'].append('complete endpoint state equals independent full action replay')
        del independently_restored
        def asset_bytes():
            return sum(p.stat().st_size for p in (args.output / 'transfers').rglob('*') if p.is_file())
        before_bytes = asset_bytes()
        positions = (0, args.history // 2, args.history, args.history // 2, 0, args.history)
        def seek(position):
            nonlocal result
            result = call('surgicalRehearsal', {'caseHash': identity, 'command': 'seek',
                'position': position, 'expectedStateId': result['view']['stateId']})
            return result
        for repetition in range(args.repetitions):
            for position in positions:
                start = time.perf_counter()
                next_result = seek(position)
                elapsed = (time.perf_counter() - start) * 1000
                assert next_result['view']['stateId'] == state_ids[position]
                assert semantic_digest(session.surgical_sessions[identity][1].state) == states[position]
                report['seeks'].append({'repetition': repetition, 'position': position, 'handlerMs': elapsed,
                    'responseJsonBytes': len(json.dumps(next_result, separators=(',', ':'), allow_nan=False).encode())})
        report['checks'].append('every forward/reverse/revisited cursor equals the original complete effect state and state ID')
        report['newBinaryAssetBytesDuringSeeks'] = asset_bytes() - before_bytes
        warm = [row['handlerMs'] for row in report['seeks'] if row['repetition'] > 0]
        report['warmSeekMs'] = {'median': statistics.median(warm), 'p95': float(np.percentile(warm, 95))}
        profiler = cProfile.Profile()
        profiler.enable()
        for position in positions:
            seek(position)
        profiler.disable()
        profiler.dump_stats(str(args.output / 'seek.prof'))
        stats = pstats.Stats(profiler)
        stages = ('restore', 'branch', 'clone', 'state_id', 'view', 'observe', 'metrics', 'dispatch',
            '_require_json_budget', 'semantic_digest', 'freeze_json', 'thaw_json', 'deepcopy', '_iterencode',
            'iterencode', 'capsule_voxel_indices', 'seek', 'require_cache_source')
        for (filename, line, function), (primitive, calls, own, cumulative, _callers) in stats.stats.items():
            if function in stages:
                report['stages'].append({'file': filename, 'line': line, 'function': function,
                    'primitiveCalls': primitive, 'calls': calls, 'selfMs': own * 1000, 'inclusiveMs': cumulative * 1000})
        report['stages'].sort(key=lambda row: row['inclusiveMs'], reverse=True)
        report['profiledHandlerTotalMs'] = stats.total_tt * 1000
        report['maxRssBytes'] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        cache = getattr(session, 'surgical_replay_cache', None)
        report['checkpointCache'] = cache.statistics if cache is not None else None
        report['passed'] = True
    finally:
        session.close()
    (args.output / 'report.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps({key: report[key] for key in ('label', 'source', 'historySteps', 'warmSeekMs', 'archiveBytes',
        'newBinaryAssetBytesDuringSeeks', 'checkpointCache', 'passed')}, indent=2))


if __name__ == '__main__':
    main()
