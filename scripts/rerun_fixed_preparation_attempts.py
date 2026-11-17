#!/usr/bin/env python3
"""Retain timing-contended attempts and rerun predeclared affected jobs only."""
import datetime,time,json
from run_fixed_preparation_diagnostic import ROOT,DEFAULT,ORIGINAL,freeze_bundle,load_original,check_original_immutable,read,save,sha

def main():
 out=DEFAULT/'clean-attempt-v2';plan=read(out/'plan.json')
 if (out/'evaluation.json').exists() or (out/'results').exists():raise ValueError('Preserve existing attempts; choose a new attempt version')
 original=read(DEFAULT/'evaluation.json')
 if original['status']!='complete':raise ValueError('Initial sweep must finish first')
 bundle,manifest=freeze_bundle(DEFAULT);control,backend=load_original(bundle)
 import torch
 torch.set_num_threads(1);torch.set_num_interop_threads(1)
 from resectionlab.surgical_sequential_task import generated_task
 from resectionlab.surgical_sequential_benchmark import ComputeTier,verify_replay
 tiers={t['name']:ComputeTier(**t) for t in original['tiers']};started=time.monotonic()
 report=dict(schema='resectionlab.diagnostic-timing-retry.v1',planSha256=sha(out/'plan.json'),initialEvaluationSha256=sha(DEFAULT/'evaluation.json'),sourceBundleSha256=sha(bundle/'bundle.json'),controllerSha256=sha(ROOT/'src/resectionlab/surgical_fixed_preparation_control.py'),runnerSha256=sha(__file__),rows=[],failures=[],nativeBackend=backend)
 for job in plan['jobs']:
  start=datetime.datetime.now(datetime.timezone.utc).isoformat();task=generated_task(job['parent']);row=dict(**job,method=control.METHOD,seed=0)
  try:
   row=control.run_fixed_preparation(task,tiers[job['tier']]);row['replay']=verify_replay(task,row)
   row['diagnosticProvenance']=dict(sourceCommit=original['sourceCommit'],sourceBundleManifestSha256=sha(bundle/'bundle.json'),controllerSha256=report['controllerSha256'],attemptVersion='clean-attempt-v2')
  except Exception as error:
   row['failure']=dict(errorType=type(error).__name__,error=str(error))
   if hasattr(error,'receipt'):row['replay']=error.receipt
   report['failures'].append(dict(**job,**row['failure']))
  row['attemptTiming']=dict(startedUtc=start,finishedUtc=datetime.datetime.now(datetime.timezone.utc).isoformat())
  save(out/'results'/f'{job["tier"]}-{job["parent"]}-{control.METHOD}-0.json',row);report['rows'].append(row)
  save(out/'progress.json',dict(completed=len(report['rows']),wallSeconds=time.monotonic()-started,failures=report['failures']))
 check_original_immutable(manifest);report['wallSeconds']=time.monotonic()-started
 report['originalEvidenceUnchanged']=True
 save(out/'evaluation.json',report);print(json.dumps(dict(rows=len(report['rows']),wallSeconds=report['wallSeconds'],failures=report['failures'])))
if __name__=='__main__':main()
