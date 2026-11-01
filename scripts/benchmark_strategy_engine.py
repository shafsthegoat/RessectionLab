"""Paired cold strategy proofs in fresh processes with isolated engine sources.

Use saved DEVELOPMENT workspaces under testcases. Each arm starts a new process,
loads the same source, then times complete canonical validation and its warm
repeat. Source loading/rendering are excluded; source admission is retained.
No archive, simulator version, computation limit or historical outcome is edited.
"""
from __future__ import annotations

import argparse
from hashlib import sha256
import importlib.abc
import importlib.util
import json
from pathlib import Path
import platform
import resource
import statistics
import subprocess
import sys
import time

_PHYSICS = ('surgical_engine', 'surgical_continuum', 'surgical_motion')


def _digest(path):
    return sha256(path.read_bytes()).hexdigest()


def _worker(config_path):
    config = json.loads(config_path.read_text())
    assert _digest(Path(__file__)) == config['recipeSha256']
    root = Path(config['root']); sys.path.insert(0, str(root/'src'))
    sources = {}
    for name, item in config['sharedSources'].items():
        data = Path(item['path']).read_bytes()
        assert sha256(data).hexdigest() == item['sha256']
        sources[name] = data
    for name, path in config['physicsSources'].items():
        data = Path(path).read_bytes()
        assert sha256(data).hexdigest() == config['physicsSha256'][name]
        sources['resectionlab.'+name] = data
    class Loader(importlib.abc.Loader):
        def __init__(self, name): self.name = name
        def create_module(self, spec): return None
        def exec_module(self, module):
            # Freeze executed Python bytes, preserving normal resource paths.
            module.__file__ = str(root/'src/resectionlab'/config['sharedSources'][self.name]['relative'])
            exec(compile(sources[self.name], module.__file__, 'exec'), module.__dict__)
    class Finder(importlib.abc.MetaPathFinder):
        def find_spec(self, name, path=None, target=None):
            if name in sources:
                return importlib.util.spec_from_loader(name, Loader(name),
                    is_package=config['sharedSources'][name]['package'])
    sys.meta_path.insert(0, Finder())
    from resectionlab.core import semantic_digest
    from resectionlab.workspace_bundle import load_workspace
    from resectionlab.surgical_environment import load_environment
    from resectionlab.surgical_engine import SurgicalEngine
    from resectionlab import surgical_strategy_workspace as workspace
    from resectionlab import surgical_strategy_sensitivity as sensitivity
    from resectionlab import surgical_two_tool as planning
    case_path = Path(config['case']); before_bytes = _digest(case_path)
    case, artifacts, _ = load_workspace(case_path)
    scene = load_environment(case_path); record = artifacts['surgical_toolbox']
    original = semantic_digest(record)
    counts = {}; step, micro = SurgicalEngine.step, SurgicalEngine._physical_substep
    step_code, micro_code = step.__code__, micro.__code__
    def started(code, offset):
        if code is micro_code:
            name = 'physicalMicrostepAttempts'
        else:
            engine = sys._getframe(1).f_locals['self']
            name = 'nativeStepAttempts' if type(engine) is SurgicalEngine else 'publicStepAttempts'
        counts[name] = counts.get(name, 0)+1
    # Wrapping step changes the engine's definition-time native-preview fence.
    # Local monitoring preserves all function identities and execution paths.
    monitoring = sys.monitoring; tool_id = monitoring.PROFILER_ID
    monitoring.use_tool_id(tool_id, 'strategy-proof-counts')
    monitoring.register_callback(tool_id, monitoring.events.PY_START, started)
    for code in (step_code, micro_code):
        monitoring.set_local_events(tool_id, code, monitoring.events.PY_START)
    workspace._VALIDATED.clear(); sensitivity._VALIDATED.clear(); planning._VALIDATED_REPORTS.clear()
    def rss():
        import os
        return 1024*int(subprocess.check_output(['ps','-o','rss=','-p',str(os.getpid())], text=True).strip())
    rss_before = rss(); start = time.perf_counter()
    try:
        workspace.validate_workspace(scene, record, case.semantic_hash)
        cold = time.perf_counter()-start; cold_counts = counts.copy(); rss_after = rss()
        cold_diagnostics = dict(baselineCache=workspace.baseline_cache_stats(), prefixCaches=planning.prefix_cache_stats())
        assert semantic_digest(record) == original
        counts.clear(); start = time.perf_counter()
        workspace.validate_workspace(scene, record, case.semantic_hash)
        warm = time.perf_counter()-start
    finally:
        for code in (step_code, micro_code): monitoring.set_local_events(tool_id, code, 0)
        monitoring.free_tool_id(tool_id)
    assert (SurgicalEngine.step is step and step.__code__ is step_code and
            SurgicalEngine._physical_substep is micro and micro.__code__ is micro_code)
    assert semantic_digest(record) == original and _digest(case_path) == before_bytes
    row = dict(round=config['round'], arm=config['arm'], coldSeconds=cold, warmSeconds=warm,
        coldCounts=cold_counts, warmCounts=counts.copy(), sourceSha256=before_bytes,
        entireRecordHash=original, sourcePreserved=True, entireRecordExact=True,
        physicsSha256={name:sha256(sources['resectionlab.'+name]).hexdigest() for name in config['physicsSources']},
        sharedCodeFromSnapshot=True, recipeSha256=config['recipeSha256'],
        instrumentation='sys.monitoring local PY_START; untouched engine methods',
        engineMethodsUnchanged=True,
        rssBeforeProofBytes=rss_before, rssAfterColdProofBytes=rss_after,
        workerLifetimePeakRssBytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*(1 if sys.platform=='darwin' else 1024),
        **cold_diagnostics)
    Path(config['result']).write_text(json.dumps(row, indent=2)+'\n')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--case', type=Path)
    parser.add_argument('--output', type=Path)
    parser.add_argument('--reference-commit')
    parser.add_argument('--repeats', type=int, default=3)
    parser.add_argument('--index-physics', action='store_true', help='Current physics from Git index, excluding concurrent unstaged edits')
    parser.add_argument('--worker-config', type=Path, help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.worker_config is not None:
        _worker(args.worker_config); return
    if not args.case or not args.output or not args.reference_commit:
        parser.error('--case, --output and --reference-commit are required')
    root = Path(__file__).resolve().parents[1]; case = args.case.resolve(); output = args.output.resolve()
    if (not case.is_relative_to(root/'testcases') or not case.is_file() or
            not output.is_relative_to(root/'testcases') or output.exists()):
        parser.error('Use an existing testcases source and fresh testcases output')
    if not 1 <= args.repeats <= 10: parser.error('repeats must be 1..10')
    def git(*values): return subprocess.check_output(['git','-C',str(root),*values])
    reference = git('rev-parse','--verify',args.reference_commit+'^{commit}').decode().strip()
    output.mkdir(parents=True)
    # Snapshot the shared package too. Check that concurrent source changes did
    # not silently turn an engine-only experiment into different environments.
    package = root/'src/resectionlab'; shared = {}; shared_sources = {}
    for path in sorted(package.rglob('*.py')):
        relative = path.relative_to(package); target = output/'source/shared'/relative
        data = path.read_bytes()
        target.parent.mkdir(parents=True, exist_ok=True); target.write_bytes(data)
        shared[str(relative)] = sha256(data).hexdigest()
        parts = list(relative.with_suffix('').parts); is_package = parts[-1] == '__init__'
        if is_package: parts.pop()
        name = '.'.join(('resectionlab', *parts))
        shared_sources[name] = dict(path=str(target), relative=str(relative), package=is_package,
                                    sha256=shared[str(relative)])
    physics = {}; physics_hashes = {}
    for arm in ('before','after'):
        folder = output/'source'/arm; folder.mkdir()
        physics[arm] = {}; physics_hashes[arm] = {}
        for name in _PHYSICS:
            path = 'src/resectionlab/'+name+'.py'
            data = (git('show',reference+':'+path) if arm == 'before' else
                    git('show',':'+path) if args.index_physics else (root/path).read_bytes())
            target = folder/(name+'.py'); target.write_bytes(data); physics[arm][name] = str(target)
            physics_hashes[arm][name] = sha256(data).hexdigest()
    recipe = Path(__file__).read_bytes(); recipe_hash = sha256(recipe).hexdigest()
    result = dict(referenceCommit=reference, head=git('rev-parse','HEAD').decode().strip(),
        python=platform.python_version(), platform=platform.platform(), case=str(case), sourceSha256=_digest(case),
        recipeSha256=recipe_hash, sharedPackageSha256=shared,
        scope='Fresh process and empty canonical proof caches per arm; identical source-verification load/proof sequence; warm OS files; loading/rendering excluded; source admission included',
        memoryScope='RSS before/after cold proof; worker lifetime peak includes source loading and the warm proof; separate worker per arm',
        indexPhysics=args.index_physics, rounds=[])
    recipe_path = output/'recipe.py'; recipe_path.write_bytes(recipe)
    for repeat in range(args.repeats):
        for arm in (('before','after') if repeat%2 == 0 else ('after','before')):
            folder = output/f'round-{repeat+1}-{arm}'; folder.mkdir()
            config = dict(root=str(root), case=str(case), round=repeat+1, arm=arm,
                          physicsSources=physics[arm], physicsSha256=physics_hashes[arm],
                          sharedSources=shared_sources, recipeSha256=recipe_hash, result=str(folder/'result.json'))
            config_path = folder/'config.json'; config_path.write_text(json.dumps(config, indent=2)+'\n')
            with (folder/'worker.log').open('w') as log:
                assert _digest(recipe_path) == recipe_hash
                subprocess.run([sys.executable,str(recipe_path),'--worker-config',str(config_path)],
                    stdout=log, stderr=subprocess.STDOUT, check=True, cwd=root)
            row = json.loads((folder/'result.json').read_text()); result['rounds'].append(row)
            assert row['physicsSha256'] == physics_hashes[arm]
            assert row['recipeSha256'] == recipe_hash
            (output/'results.json').write_text(json.dumps(result, indent=2)+'\n')
            print(json.dumps({key:row[key] for key in ('round','arm','coldSeconds','warmSeconds','coldCounts','warmCounts')}), flush=True)
    rows = result['rounds']; first = rows[0]
    assert _digest(case) == result['sourceSha256']
    assert all(row['sourceSha256'] == first['sourceSha256'] and row['entireRecordHash'] == first['entireRecordHash'] and
               row['coldCounts'] == first['coldCounts'] and row['warmCounts'] == first['warmCounts'] for row in rows)
    assert {str(p.relative_to(package)):_digest(p) for p in sorted(package.rglob('*.py'))} == shared, 'Shared package changed during benchmark'
    result.update(sourcePreserved=True, entireRecordExact=True, transitionCountsExact=True, sharedPackageUnchanged=True,
        medians={arm:{key:statistics.median(row[key] for row in rows if row['arm']==arm)
                      for key in ('coldSeconds','warmSeconds')} for arm in ('before','after')})
    (output/'results.json').write_text(json.dumps(result, indent=2)+'\n')


if __name__ == '__main__':
    main()
