"""Paired exact two-tool validation benchmark; existing DEVELOPMENT inputs only.

Source admission/setup is outside the timed region. 'Cold' means an empty
canonical-validation cache, not cold filesystem/OS/source-byte caches. Both
implementations use the same loaded native baseline and physics. This does not
measure renderer or image preparation latency; use the hidden Electron verifier
for that separate application check. No physics step is skipped on materialize.
"""
from __future__ import annotations

import argparse
import importlib.abc
import importlib.util
import json
from pathlib import Path
import platform
import statistics
import subprocess
import sys
import time
import types


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--case', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--reference-commit', required=True)
    parser.add_argument('--controls-json', type=Path, help='Exact native draft controls, including optional context selection')
    parser.add_argument('--candidate', help='Materialize this admissible candidate; default is the report selection')
    parser.add_argument('--repeats', type=int, default=3)
    parser.add_argument('--isolate-index-physics', action='store_true',
        help='Read the three shared physics modules from the index without changing the checkout')
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(root/'src'))
    output = args.output.resolve()
    if not output.is_relative_to(root/'testcases') or output.exists():
        parser.error('Use a fresh output directory under testcases/ to preserve prior evidence')
    if not 1 <= args.repeats <= 10: parser.error('repeats must be 1..10')
    output.mkdir(parents=True)
    def git(*values): return subprocess.check_output(['git', '-C', str(root), *values])
    reference_commit = git('rev-parse', '--verify', args.reference_commit+'^{commit}').decode().strip()
    from hashlib import sha256
    physics_sources = {}
    if args.isolate_index_physics:
        physics_sources = {f'resectionlab.{name}': git('show', f':src/resectionlab/{name}.py')
            for name in ('surgical_engine', 'surgical_continuum', 'surgical_motion')}
        class Loader(importlib.abc.Loader):
            def __init__(self, name): self.name = name
            def create_module(self, spec): return None
            def exec_module(self, module):
                module.__file__ = str(root/'src/resectionlab'/(self.name.rsplit('.', 1)[1]+'.py'))
                exec(compile(physics_sources[self.name], module.__file__, 'exec'), module.__dict__)
        class Finder(importlib.abc.MetaPathFinder):
            def find_spec(self, name, path=None, target=None):
                if name in physics_sources: return importlib.util.spec_from_loader(name, Loader(name))
        sys.meta_path.insert(0, Finder())

    from resectionlab import surgical_two_tool as after
    from resectionlab.core import semantic_digest
    from resectionlab.workspace_bundle import load_workspace
    from resectionlab.surgical_environment import load_environment
    from resectionlab.surgical_engine import SurgicalEngine
    from resectionlab.surgical_task_admission import bind_admission
    before_source = git('show', reference_commit+':src/resectionlab/surgical_two_tool.py')
    before = types.ModuleType('resectionlab._benchmark_two_tool_reference')
    before.__package__ = 'resectionlab'; before.__file__ = '<reference surgical_two_tool.py>'
    sys.modules[before.__name__] = before
    exec(compile(before_source, before.__file__, 'exec'), before.__dict__)
    source = args.case.resolve(); source_hash = sha256(source.read_bytes()).hexdigest()
    _, artifacts, _ = load_workspace(source); record = artifacts['surgical_toolbox']
    engine = SurgicalEngine.restore(load_environment(source), record['branches'][record['branchId']])
    bind_admission(engine, record['taskAdmission'])
    controls = json.loads(args.controls_json.read_text()) if args.controls_json else after.defaults(engine)['defaults']
    initial = engine.snapshot()
    result = {'case': str(source), 'sourceSha256': source_hash, 'referenceCommit': reference_commit,
        'head': git('rev-parse', 'HEAD').decode().strip(), 'python': platform.python_version(),
        'platform': platform.platform(), 'controls': controls, 'startStateId': engine.state_id,
        'referenceModuleSha256': sha256(before_source).hexdigest(),
        'currentModuleSha256': sha256(Path(after.__file__).read_bytes()).hexdigest(),
        'indexPhysicsSha256': {name: sha256(value).hexdigest() for name, value in physics_sources.items()},
        'timingScope': 'backend only; warm source admission; cold canonical cache per round; alternating order',
        'rounds': []}
    canonical = None
    counters = {}
    native_step, physical_substep = SurgicalEngine.step, SurgicalEngine._physical_substep
    def counted_step(self, *values, **options):
        key = 'nativeStepAttempts' if type(self) is SurgicalEngine else 'publicStepAttempts'
        counters[key] = counters.get(key, 0)+1
        return native_step(self, *values, **options)
    def counted_physical(self, *values, **options):
        counters['physicalMicrostepAttempts'] = counters.get('physicalMicrostepAttempts', 0)+1
        return physical_substep(self, *values, **options)
    SurgicalEngine.step, SurgicalEngine._physical_substep = counted_step, counted_physical
    def timed(operation):
        counters.clear()
        start = time.perf_counter(); value = operation()
        return value, time.perf_counter()-start, counters.copy()
    for round_id in range(args.repeats):
        for label, module in ([('before', before), ('after', after)] if round_id % 2 == 0 else [('after', after), ('before', before)]):
            if hasattr(module, '_VALIDATED_REPORTS'): module._VALIDATED_REPORTS.clear()
            report, generation, generation_counts = timed(lambda: module.generate_comparison(engine, controls))
            candidate_id = args.candidate or report['publicPlan']['decision']['selectedId']
            _, cold, cold_counts = timed(lambda: module.validate_report(engine, report))
            _, warm, warm_counts = timed(lambda: module.validate_report(engine, report))
            def preview():
                module.validate_report(engine, report)  # current_report's live validation
                return module.materialize(engine, report, candidate_id)  # validates again, then executes
            (native, checkpoints, receipt), preview_seconds, preview_counts = timed(preview)
            (again, phases_again, receipt_again), materialize_seconds, materialize_counts = timed(lambda: module.materialize(engine, report, candidate_id))
            proof = {'reportHash': semantic_digest(report), 'endpointHash': semantic_digest(native.snapshot()),
                'receiptHash': semantic_digest(receipt),
                'phaseHashes': {key: semantic_digest(value.snapshot()) for key, value in checkpoints.items()}}
            assert engine.snapshot() == initial and native.snapshot() == again.snapshot() and receipt == receipt_again
            assert proof['phaseHashes'] == {key: semantic_digest(value.snapshot()) for key, value in phases_again.items()}
            if canonical is None:
                canonical = proof
                for name, value in (('report', report), ('endpoint', native.snapshot()), ('receipt', receipt)):
                    (output/(name+'.json')).write_text(json.dumps(value, indent=2)+'\n')
            assert canonical == proof, 'Exact canonical report/native endpoint/receipt/phase parity failed'
            row = {'round': round_id+1, 'arm': label, 'generationSeconds': generation, 'coldValidationSeconds': cold,
                'warmValidationSeconds': warm, 'previewSeconds': preview_seconds, 'materializeSeconds': materialize_seconds,
                'exactProof': proof, 'candidateId': candidate_id,
                'computationCounts': {'generation': generation_counts, 'coldValidation': cold_counts,
                    'warmValidation': warm_counts, 'preview': preview_counts, 'materialize': materialize_counts},
                'prefixScopes': module.prefix_cache_stats() if hasattr(module, 'prefix_cache_stats') else None,
                'validationCache': module.validation_cache_stats() if hasattr(module, 'validation_cache_stats') else None}
            result['rounds'].append(row)
            (output/'results.json').write_text(json.dumps(result, indent=2)+'\n')
            print(json.dumps({key: value for key, value in row.items() if key not in ('exactProof',)}), flush=True)
    result['medians'] = {label: {key: statistics.median(row[key] for row in result['rounds'] if row['arm'] == label)
        for key in ('generationSeconds', 'coldValidationSeconds', 'warmValidationSeconds', 'previewSeconds', 'materializeSeconds')}
        for label in ('before', 'after')}
    assert sha256(source.read_bytes()).hexdigest() == source_hash
    result.update(exactParity=True, sourcePreserved=True)
    (output/'results.json').write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps(result['medians']), flush=True)


if __name__ == '__main__': main()
