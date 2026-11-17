#!/usr/bin/env python3
"""Execute the predeclared two-case development-only tip feasibility check."""
import argparse,importlib.util,time,datetime
from pathlib import Path
from run_fixed_preparation_diagnostic import ROOT,DEFAULT,freeze_bundle,load_original,read,save,sha,check_original_immutable

def main():
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path,default=DEFAULT.parent/'development-variation-v1');args=p.parse_args();out=args.output.resolve()
 if (out/'evaluation.json').exists() or (out/'results').exists():raise ValueError('Preserve existing variation results; choose a new version')
 plan=read(out/'plan.json');assert plan['developmentParents']==[4000,4004]
 bundle,manifest=freeze_bundle(DEFAULT);fixed,backend=load_original(bundle)
 spec=importlib.util.spec_from_file_location('development_variation',ROOT/'src/resectionlab/surgical_preparation_variation.py');variation=importlib.util.module_from_spec(spec);spec.loader.exec_module(variation)
 import torch
 torch.set_num_threads(1);torch.set_num_interop_threads(1)
 from resectionlab.surgical_sequential_benchmark import ComputeTier,run_episode,verify_replay
 from resectionlab.surgical_sequential_task import observation
 tier=ComputeTier(**read(DEFAULT/'protocol.json')['tiers'][-1]);started=time.monotonic()
 report=dict(schema='resectionlab.state-dependent-tip-development-results.v1',planSha256=sha(out/'plan.json'),
  sourceBundleSha256=sha(bundle/'bundle.json'),nativeBackend=backend,sourceSha256=sha(ROOT/'src/resectionlab/surgical_preparation_variation.py'),
  runnerSha256=sha(Path(__file__)),startedUtc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
  rows=[],failures=[],optimizerUpdates=0,scope='Two development-only feasibility witnesses; no untouched or learning evaluation')
 for parent in plan['developmentParents']:
  for method in ('greedy','fixed-narrow-greedy','source-state-geometry-rule'):
   task=variation.variation_task(parent);row=dict(parent=parent,method=method,tier=tier.name,seed=0)
   try:
    if method=='fixed-narrow-greedy':row=fixed.run_fixed_preparation(task,tier)
    elif method=='greedy':row=run_episode(task,'greedy',tier)
    else:
     row=run_episode(task,'learned',tier,selector=variation.geometry_state_selector)
     row['method']=method;row['notLearned']=True;row['ruleInputs']='Same28source/current-state features+mask; no previews/outcome labels'
    row['developmentAudit']=task.audit
    row['developmentVariation']=task.scene.properties['sequentialSkillTaskV1']['developmentVariation']
    row['replay']=verify_replay(task,row)
   except Exception as error:
    row['failure']=dict(errorType=type(error).__name__,error=str(error))
    if hasattr(error,'receipt'):row['replay']=error.receipt
    report['failures'].append(dict(parent=parent,method=method,**row['failure']))
   report['rows'].append(row);save(out/'results'/f'{parent}-{method}.json',row)
   save(out/'progress.json',dict(completed=len(report['rows']),wallSeconds=time.monotonic()-started,failures=report['failures']))
 report['wallSeconds']=time.monotonic()-started;report['originalEvidenceUnchanged']=True;check_original_immutable(manifest)
 if report['sourceSha256']!=sha(ROOT/'src/resectionlab/surgical_preparation_variation.py'):raise ValueError('Variation changed during execution')
 report['summary']=[dict(parent=r['parent'],method=r['method'],fraction=r.get('progress',{}).get('originalTargetTreatmentFraction'),
  classification=r.get('endpoint',{}).get('classification'),calls=r.get('computation',{}).get('physicalCalls'),
  seconds=r.get('computation',{}).get('wallSeconds'),neural=r.get('neuralRemovedMm3'),failure=r.get('failure')) for r in report['rows']]
 report['limitations']=['This only challenges a constant tip; a fixed two-stage rule can still explain these two cases.','Source geometry rule is hand-coded and untrained; no policy/generalization advantage.','Require prospective richer development variation/simple controls before any later learning campaign.']
 save(out/'evaluation.json',report);print(__import__('json').dumps(dict(summary=report['summary'],failures=report['failures']),indent=2))
if __name__=='__main__':main()
