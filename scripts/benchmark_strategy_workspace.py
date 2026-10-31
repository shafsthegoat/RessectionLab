"""Paired read-only cold workspace proof; no generation, rendering or installation.

Both arms use the same saved DEVELOPMENT studies and physics. Cold means empty
canonical-validation caches; source/OS caches are warm. Source loading is outside
timing. Retained modules are loaded without checkout/worktree changes.
"""
from __future__ import annotations

import argparse
from hashlib import sha256
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
    parser.add_argument('--repeats', type=int, default=3)
    parser.add_argument('--isolate-index-physics', action='store_true')
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(root/'src'))
    output = args.output.resolve(); source = args.case.resolve()
    if output.exists() or not output.is_relative_to(root/'testcases') or not source.is_relative_to(root/'testcases'):
        parser.error('Use an existing testcases source and fresh testcases output')
    if not 1 <= args.repeats <= 10: parser.error('repeats must be 1..10')
    output.mkdir(parents=True)
    def git(*values): return subprocess.check_output(['git','-C',str(root),*values])
    reference = git('rev-parse','--verify',args.reference_commit+'^{commit}').decode().strip()
    physics = {}
    if args.isolate_index_physics:
        physics = {f'resectionlab.{name}': git('show',f':src/resectionlab/{name}.py')
            for name in ('surgical_engine','surgical_continuum','surgical_motion')}
        class Loader(importlib.abc.Loader):
            def __init__(self, name): self.name = name
            def create_module(self, spec): return None
            def exec_module(self, module):
                module.__file__ = str(root/'src'/'resectionlab'/(self.name.rsplit('.',1)[1]+'.py'))
                exec(compile(physics[self.name],module.__file__,'exec'),module.__dict__)
        class Finder(importlib.abc.MetaPathFinder):
            def find_spec(self, name, path=None, target=None):
                if name in physics: return importlib.util.spec_from_loader(name,Loader(name))
        sys.meta_path.insert(0,Finder())
    import resectionlab
    from resectionlab import surgical_two_tool as after_plan
    from resectionlab import surgical_strategy_sensitivity as after_sensitivity
    from resectionlab import surgical_strategy_workspace as after_workspace
    from resectionlab.core import semantic_digest
    from resectionlab.surgical_engine import SurgicalEngine
    from resectionlab.surgical_environment import load_environment
    from resectionlab.workspace_bundle import load_workspace
    retained_sources = {}
    def retained(name):
        data = git('show',reference+':src/resectionlab/'+name+'.py'); retained_sources[name] = data
        module = types.ModuleType('resectionlab._workspace_reference_'+name)
        module.__package__ = 'resectionlab'; module.__file__ = '<reference '+name+'.py>'
        sys.modules[module.__name__] = module
        exec(compile(data,module.__file__,'exec'),module.__dict__)
        return module
    before_plan = retained('surgical_two_tool')
    before_sensitivity = retained('surgical_strategy_sensitivity'); before_sensitivity.planning = before_plan
    before_workspace = retained('surgical_strategy_workspace')
    before_workspace.validate_report = before_plan.validate_report; before_workspace.materialize = before_plan.materialize
    source_sha = sha256(source.read_bytes()).hexdigest()
    case, artifacts, _ = load_workspace(source); record = artifacts['surgical_toolbox']; scene = load_environment(source)
    original = semantic_digest(record)
    expected = {name: semantic_digest(record.get(name)) for name in
        ('strategyComparisons','strategyApplications','strategySensitivityStudies','branches','parents','branchId','position')}
    counts = {}
    native_step, physical = SurgicalEngine.step, SurgicalEngine._physical_substep
    def counted_step(engine,*a,**k):
        key = 'nativeStepAttempts' if type(engine) is SurgicalEngine else 'publicStepAttempts'
        counts[key] = counts.get(key,0)+1; return native_step(engine,*a,**k)
    def counted_physical(engine,*a,**k):
        counts['physicalMicrostepAttempts'] = counts.get('physicalMicrostepAttempts',0)+1
        return physical(engine,*a,**k)
    SurgicalEngine.step, SurgicalEngine._physical_substep = counted_step, counted_physical
    result = {'source':str(source),'sourceSha256':source_sha,'referenceCommit':reference,
        'head':git('rev-parse','HEAD').decode().strip(),'python':platform.python_version(),'platform':platform.platform(),
        'referenceCodeSha256':{n:sha256(s).hexdigest() for n,s in retained_sources.items()},
        'currentCodeSha256':{n:sha256(Path(m.__file__).read_bytes()).hexdigest() for n,m in
            (('surgical_two_tool',after_plan),('surgical_strategy_sensitivity',after_sensitivity),('surgical_strategy_workspace',after_workspace))},
        'indexPhysicsSha256':{n:sha256(s).hexdigest() for n,s in physics.items()},
        'scope':'backend canonical proof; source load excluded; empty proof caches per arm; warm source/OS; alternating order',
        'exactSourceFieldHashes':expected,'rounds':[]}
    arms = {'before':(before_workspace,before_sensitivity,before_plan),'after':(after_workspace,after_sensitivity,after_plan)}
    try:
        for repeat in range(args.repeats):
            for label in (('before','after') if repeat%2 == 0 else ('after','before')):
                workspace, sensitivity, planning = arms[label]
                # Workspace validation imports this leaf lazily. Keep reference
                # and current proof caches/implementations fully separate.
                sys.modules['resectionlab.surgical_strategy_sensitivity'] = sensitivity
                resectionlab.surgical_strategy_sensitivity = sensitivity
                workspace._VALIDATED.clear(); sensitivity._VALIDATED.clear(); planning._VALIDATED_REPORTS.clear()
                counts.clear(); start = time.perf_counter()
                workspace.validate_workspace(scene,record,case.semantic_hash)
                cold = time.perf_counter()-start; cold_counts = counts.copy()
                diagnostics = {'baseline':workspace.baseline_cache_stats() if hasattr(workspace,'baseline_cache_stats') else None,
                    'publicPrefixes':planning.prefix_cache_stats()}
                assert semantic_digest(record) == original
                assert {k:semantic_digest(record.get(k)) for k in expected} == expected
                counts.clear(); start = time.perf_counter()
                workspace.validate_workspace(scene,record,case.semantic_hash)
                warm = time.perf_counter()-start
                row = {'round':repeat+1,'arm':label,'coldSeconds':cold,'warmSeconds':warm,
                    'coldCounts':cold_counts,'warmCounts':counts.copy(),'diagnostics':diagnostics,'allSourceFieldsExact':True}
                result['rounds'].append(row)
                (output/'results.json').write_text(json.dumps(result,indent=2)+'\n')
                print(json.dumps(row),flush=True)
    finally:
        sys.modules['resectionlab.surgical_strategy_sensitivity'] = after_sensitivity
        resectionlab.surgical_strategy_sensitivity = after_sensitivity
        SurgicalEngine.step, SurgicalEngine._physical_substep = native_step, physical
    assert sha256(source.read_bytes()).hexdigest() == source_sha
    result.update(sourcePreserved=True,exactParity=True,
        medians={arm:{key:statistics.median(r[key] for r in result['rounds'] if r['arm']==arm)
            for key in ('coldSeconds','warmSeconds')} for arm in arms})
    (output/'results.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result['medians']),flush=True)


if __name__ == '__main__': main()
