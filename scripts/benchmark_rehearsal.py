#!/usr/bin/env python3
"""Sequential production Electron comparisons; testing artifacts stay ignored.

Capture the baseline before edits, then run balanced baseline/optimized pairs.
Only generated desktop/dist is swapped; the source checkout is never restored,
branched or copied into a Git worktree. The current build is restored in finally.
"""
from __future__ import annotations
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess

ROOT = Path(__file__).resolve().parents[1]


def within_testcases(value):
    target = Path(value).resolve()
    if not target.is_relative_to((ROOT / 'testcases').resolve()):
        raise ValueError('Keep benchmark data/source snapshots under testcases/')
    return target


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', default=str(ROOT / 'testcases/performance/rehearsal-2026-10-19'))
    parser.add_argument('--node', default=shutil.which('node'))
    parser.add_argument('--capture-baseline', action='store_true')
    parser.add_argument('--phase', choices=['baseline', 'optimized', 'balanced'], default='balanced')
    parser.add_argument('--workloads', nargs='+', choices=['ordinary', 'dense', 'large'], default=['ordinary', 'dense', 'large'])
    parser.add_argument('--repeats', type=int, default=3)
    parser.add_argument('--large-history', type=int, default=48)
    parser.add_argument('--series', default='comparison')
    parser.add_argument('--verify', action='store_true')
    parser.add_argument('--display',choices=['primary','secondary'],default='secondary',help='Keep the disposable benchmark on the selected monitor. Secondary mode never activates the app or moves the OS pointer.')
    parser.add_argument('--trace-transport',action='store_true',help='Separate overhead-bearing worker-byte trace; exclude its latencies from primary comparisons.')
    args = parser.parse_args()
    output = within_testcases(args.output)
    output.mkdir(parents=True, exist_ok=True)
    baseline = output / 'baseline-source'
    desktop = ROOT / 'desktop'
    def build():
        for command in [['node_modules/typescript/bin/tsc', '--noEmit'], ['node_modules/vite/bin/vite.js', 'build']]:
            subprocess.run([args.node, *command], cwd=desktop, check=True)
    if args.capture_baseline:
        if baseline.exists():
            raise ValueError('Baseline already exists; preserve it or choose a distinct output')
        build()
        shutil.copytree(ROOT / 'src', baseline / 'src', ignore=shutil.ignore_patterns('__pycache__'))
        shutil.copytree(desktop / 'dist', baseline / 'dist')
        (baseline / 'identity.json').write_text(json.dumps({
            'commit': subprocess.check_output(['git','rev-parse','HEAD'], cwd=ROOT,text=True).strip(),
            'source': {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest() for p in (ROOT/'src').rglob('*.py')},
        }, indent=2))
        return
    if args.phase != 'optimized' and not baseline.exists():
        raise ValueError('Capture a baseline before editing, using --capture-baseline')
    build()
    optimized = output / f'{args.series}-optimized-build'
    if optimized.exists():
        raise ValueError('Comparison series already exists; choose another --series')
    shutil.copytree(desktop / 'dist', optimized)
    workloads = {
        'ordinary': ('UCSF-PDGM-0004-standard-v2.ressectionlab', 12),
        'dense': ('meningioma-sub-06-standard-v2.ressectionlab', 12),
        'large': ('meningioma-sub-06-standard-v2.ressectionlab', args.large_history),
    }
    order = ['baseline', 'optimized', 'optimized', 'baseline'] if args.phase == 'balanced' else [args.phase]
    results=[]
    try:
        for workload in args.workloads:
            filename, history = workloads[workload]
            for index, phase in enumerate(order):
                label=f'{args.series}-{workload}-{index+1}-{phase}'
                directory=output/label
                if directory.exists():
                    raise ValueError(f'Results already exist: {directory}')
                shutil.rmtree(desktop/'dist')
                shutil.copytree(baseline/'dist' if phase=='baseline' else optimized, desktop/'dist')
                env={**os.environ,'INTERACTION_MODE':'rehearsal','INTERACTION_LABEL':label,
                     'INTERACTION_OUTPUT':str(directory),'INTERACTION_HISTORY':str(history),'INTERACTION_REPEATS':str(args.repeats),'INTERACTION_DISPLAY':args.display}
                if phase=='baseline':env['INTERACTION_PYTHON_SOURCE']=str(baseline/'src')
                else:env.pop('INTERACTION_PYTHON_SOURCE',None)
                if args.verify and phase=='optimized':env['INTERACTION_VERIFY']='1'
                else:env.pop('INTERACTION_VERIFY',None)
                print(label, flush=True)
                entry='electron/benchmark-rehearsal-transport.cjs' if args.trace_transport else 'electron/benchmark-interactions.cjs'
                result=subprocess.run([args.node,'node_modules/electron/cli.js',entry,'--case',str(ROOT/'testcases/cases'/filename)],cwd=desktop,env=env)
                report_path=directory/'report.json'
                passed=report_path.exists() and json.loads(report_path.read_text()).get('passed') is True
                results.append({'label':label,'exitCode':result.returncode,'passed':passed,'report':str(report_path)})
                (output/f'{args.series}-runs.json').write_text(json.dumps(results,indent=2))
                if result.returncode or not passed:raise RuntimeError(f'Benchmark failed or incomplete: {label}')
    finally:
        shutil.rmtree(desktop/'dist')
        shutil.copytree(optimized,desktop/'dist')


if __name__ == '__main__':
    main()
