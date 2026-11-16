#!/usr/bin/env python3
"""Two development runs with the same real, charged initial narrow prefix."""
from pathlib import Path
import importlib.util,json,time,datetime
from run_fixed_preparation_diagnostic import ROOT,DEFAULT,freeze_bundle,load_original,read,save,sha,check_original_immutable

class PrefixRule:
 def __init__(self,adaptive):self.first=True;self.adaptive=adaptive
 def __call__(self,features,mask,stochastic=False):
  def preparation(radius):
   return next((i for i,x in enumerate(features) if mask[i] and x[0] and abs(x[4]*1.5-radius)<1e-6),None)
  stop=next(i for i,x in enumerate(features) if mask[i] and x[2])
  if self.first:
   self.first=False;return preparation(.4),{}
  work=[i for i,x in enumerate(features) if mask[i] and x[1] and x[9]>0]
  wide_routes=[i for i in work if features[i,10]*30>=3.-1e-6]
  narrow_routes=[i for i in work if i not in wide_routes]
  wide=features[0,3]>.4/1.5+1e-6
  if self.adaptive:
   ready=wide_routes if wide else narrow_routes
   other=narrow_routes if wide else wide_routes
  else:
   desired=wide_routes or narrow_routes;desired_wide=bool(wide_routes)
   ready=desired if wide==desired_wide else [];other=desired if not ready else []
  if ready:return max(ready,key=lambda i:features[i,9]),{}
  if other:
   radius=(.4 if wide else 1.5) if self.adaptive else (1.5 if wide_routes else .4)
   index=preparation(radius)
   if index is not None:return index,{}
  return stop,{}

def main():
 out=DEFAULT.parent/'development-state-intervention-v1'
 if out.exists():raise ValueError('Preserve prior intervention; declare a new version')
 out.mkdir()
 plan=dict(schema='resectionlab.development-tip-state-intervention-plan.v1',declaredUtc=datetime.datetime.now(datetime.timezone.utc).isoformat(),parent=4000,
  intervention='First selection is actual costed0.4mm exchange; same one-use inventory and8selection budget for both methods.',
  methods=['adaptive-current-tip-geometry','fixed-wide-then-narrow'],runs=2,training=False,
  hypothesis='Current narrow tip should retain gated work first, widen only for off-axis remainder. Fixed wide-first should need an unavailable repeat narrow preparation.',
  limitation='Finite one-use inventory, not physical impossibility of another exchange. Hand-coded conditional schedules remain possible; no learning/generalization admission.')
 save(out/'plan.json',plan);bundle,manifest=freeze_bundle(DEFAULT);load_original(bundle)
 sp=importlib.util.spec_from_file_location('unchanged_development_geometry',ROOT/'src/resectionlab/surgical_preparation_variation.py');vm=importlib.util.module_from_spec(sp);sp.loader.exec_module(vm)
 import torch
 torch.set_num_threads(1);torch.set_num_interop_threads(1)
 from resectionlab.surgical_sequential_benchmark import ComputeTier,run_episode,verify_replay
 tier=ComputeTier(**read(DEFAULT/'protocol.json')['tiers'][-1]);started=time.monotonic()
 report=dict(schema='resectionlab.development-tip-state-intervention-results.v1',planSha256=sha(out/'plan.json'),sourceSha256=sha(__file__),geometrySourceSha256=sha(ROOT/'src/resectionlab/surgical_preparation_variation.py'),sourceBundleSha256=sha(bundle/'bundle.json'),rows=[],failures=[],optimizerUpdates=0)
 for adaptive in (True,False):
  name=plan['methods'][0 if adaptive else 1];task=vm.variation_task(4000);row=dict(parent=4000,method=name)
  try:
   row=run_episode(task,'learned',tier,selector=PrefixRule(adaptive));row['method']=name;row['notLearned']=True
   row['replay']=verify_replay(task,row);row['developmentAudit']=task.audit
  except Exception as error:
   row['failure']=dict(errorType=type(error).__name__,error=str(error))
   if hasattr(error,'receipt'):row['replay']=error.receipt
   report['failures'].append(dict(method=name,**row['failure']))
  save(out/(name+'.json'),row);report['rows'].append(row)
 check_original_immutable(manifest);report['wallSeconds']=time.monotonic()-started;report['originalEvidenceUnchanged']=True
 report['summary']=[dict(method=r['method'],fraction=r.get('progress',{}).get('originalTargetTreatmentFraction'),classification=r.get('endpoint',{}).get('classification'),calls=r.get('computation',{}).get('physicalCalls'),seconds=r.get('computation',{}).get('wallSeconds'),neural=r.get('neuralRemovedMm3'),actualExchanges=[x['action']['parameters']['radiusMm'] for s in r.get('skillHistory',[]) for x in s['receipts'] if x['action']['operation']=='exchange'],replayRequests=r.get('replay',{}).get('physicalCalls'),replaySeconds=r.get('replay',{}).get('wallSeconds')) for r in report['rows']]
 save(out/'evaluation.json',report);print(json.dumps(dict(summary=report['summary'],failures=report['failures']),indent=2))
if __name__=='__main__':main()
