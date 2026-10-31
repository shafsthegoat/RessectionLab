#!/usr/bin/env python3
"""Paired exact replay benchmark; no policy optimization or approximate physics.

Retained baseline Python files are loaded as isolated modules, never as another
checkout. Only the checkpoint and continuum tick are temporarily selected in
this single-process benchmark. Every accepted state and receipt must agree.
"""
import argparse
from contextlib import contextmanager
import cProfile
import hashlib
import importlib.util
import json
from pathlib import Path
import platform
import pstats
import resource
import statistics
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from resectionlab import surgical_engine as engine_module
from resectionlab import surgical_continuum as continuum
from resectionlab.core import array_digest, semantic_digest
from resectionlab.simulator_qualification import removed_accounting
from resectionlab.surgical_environment import standardize
from resectionlab.surgical_scene import from_case
from resectionlab.workspace_bundle import load_workspace


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def retained_module(path, name):
    spec = importlib.util.spec_from_file_location('resectionlab.' + name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--case', type=Path, required=True)
    parser.add_argument('--baseline', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--branch', default='sustained-hold')
    parser.add_argument('--repetitions', type=int, default=3)
    args = parser.parse_args()
    if args.repetitions < 2:
        parser.error('At least two alternately ordered paired repetitions required')
    for path in (args.case, args.baseline, args.output):
        if not path.resolve().is_relative_to(ROOT / 'testcases'):
            parser.error('Inputs and generated evidence must be under ignored testcases/')
    args.output.mkdir(parents=True, exist_ok=False)
    frozen = json.loads((args.baseline / 'manifest.json').read_text())
    for name in ('surgical_engine.py', 'surgical_continuum.py'):
        if sha(args.baseline / name) != frozen['files']['src/resectionlab/' + name]:
            raise ValueError('Retained baseline differs from its frozen source hash')
    baseline_engine = retained_module(args.baseline / 'surgical_engine.py', '_benchmark_old_engine')
    baseline_continuum = retained_module(args.baseline / 'surgical_continuum.py', '_benchmark_old_continuum')
    current = engine_module._microstep_checkpoint, continuum.tick

    @contextmanager
    def selected(mode):
        engine_module._microstep_checkpoint, continuum.tick = (
            (baseline_engine._microstep_checkpoint, baseline_continuum.tick)
            if mode == 'baseline' else current)
        try:
            yield
        finally:
            engine_module._microstep_checkpoint, continuum.tick = current

    original = sha(args.case)
    case, artifacts, _ = load_workspace(args.case)
    record = artifacts['surgical_toolbox']
    saved = record['branches'][args.branch]
    scene = standardize(from_case(case, record['mapping']), physics_profile=record['physicsProfile'])
    if scene.identity != saved['sceneId']:
        raise ValueError('Source scene identity differs from saved episode')
    sources = array_digest(case.mri), array_digest(scene.labels)
    rows = []; expected_trace = None

    def replay(mode, repetition, *, profile=None):
        nonlocal expected_trace
        engine = engine_module.SurgicalEngine(scene)
        timings = []; trace = []; statuses = []
        with selected(mode):
            for i, action in enumerate(saved['actions']):
                started = time.perf_counter()
                if profile is not None and i == profile[0]:
                    profile[1].enable()
                receipt = engine.step(action)
                if profile is not None and i == profile[0]:
                    profile[1].disable()
                timings.append(time.perf_counter() - started)
                if receipt['status'] == 'rejected':
                    raise AssertionError((i, receipt))
                statuses.append(receipt['status'])
                trace.append(semantic_digest({'state': engine.state, 'receipt': receipt}))
            assert engine.snapshot() == saved
        if expected_trace is None:
            expected_trace = trace
        assert trace == expected_trace
        row = {'mode': mode, 'repetition': repetition, 'stepSeconds': sum(timings),
               'actionSeconds': timings, 'actionsPerSecond': len(timings) / sum(timings),
               'episodesPerSecond': 1 / sum(timings), 'exactEveryStateAndReceipt': True,
               'completeActions': statuses.count('accepted'), 'partialActions': statuses.count('partial'),
               'traceSha256': semantic_digest(trace), 'finalStateId': engine.state_id}
        if profile is None:
            rows.append(row)
            (args.output / 'runs.json').write_text(json.dumps(rows, indent=2) + '\n')
            print(json.dumps({key: value for key, value in row.items() if key != 'actionSeconds'}), flush=True)
        return engine, timings

    # One untimed admission/warmup per implementation. Timed sums exclude the
    # independent state/receipt hashes and snapshot assertions between actions.
    replay('baseline', 'warmup'); replay('current', 'warmup')
    for repetition in range(args.repetitions):
        for mode in (('baseline', 'current') if repetition % 2 == 0 else ('current', 'baseline')):
            final, times = replay(mode, repetition)
    slowest = max(range(len(times)), key=times.__getitem__)
    profiles = {}
    for mode in ('baseline', 'current'):
        profiler = cProfile.Profile(); replay(mode, 'profile', profile=(slowest, profiler))
        stats = pstats.Stats(profiler)
        profiles[mode] = [{'file': str(Path(file).relative_to(ROOT)) if Path(file).is_relative_to(ROOT) else file,
            'line': line, 'function': function, 'calls': value[1], 'selfSeconds': value[2],
            'cumulativeSeconds': value[3]} for (file, line, function), value in sorted(
                stats.stats.items(), key=lambda row: row[1][3], reverse=True)[:35]]
    assert sha(args.case) == original and sources == (array_digest(case.mri), array_digest(scene.labels))
    summary = {}
    for mode in ('baseline', 'current'):
        values = [row['stepSeconds'] for row in rows if row['mode'] == mode and type(row['repetition']) is int]
        median = statistics.median(values)
        summary[mode] = {'medianEpisodeSeconds': median, 'actionsPerSecond': len(saved['actions']) / median,
            'episodesPerSecond': 1 / median, 'minSeconds': min(values), 'maxSeconds': max(values)}
    report = {'schema': 'resectionlab.continuum-performance.v1', 'input': str(args.case.resolve()),
        'inputSha256': original, 'branch': args.branch, 'sceneId': scene.identity,
        'baseline': frozen, 'currentSources': {name: sha(ROOT / 'src/resectionlab' / name)
            for name in ('surgical_engine.py', 'surgical_continuum.py')},
        'measurement': 'Serial accepted SurgicalEngine.step wall time; source load, equality hashes, disk writes and cProfile excluded. Warmups excluded from summaries. Alternating paired order, same machine. No training.',
        'summary': summary, 'speedup': summary['baseline']['medianEpisodeSeconds'] / summary['current']['medianEpisodeSeconds'],
        'profiledActionIndex': slowest, 'profiles': profiles, 'removed': removed_accounting(scene, final.state),
        'exactEveryStateAndReceipt': True, 'sourcePreserved': True, 'platform': platform.platform(),
        'python': platform.python_version(), 'peakProcessRssBytes': resource.getrusage(resource.RUSAGE_SELF).ru_maxrss}
    (args.output / 'report.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps({'summary': summary, 'speedup': report['speedup']}), flush=True)


if __name__ == '__main__':
    main()
