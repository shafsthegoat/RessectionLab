"""Lossless single-prefix history archive plus decisive saved scalar/sidecar evidence."""
from collections import Counter
import hashlib
import json
from pathlib import Path
HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[1];OUT=HERE/'attempt-01';POST=HERE/'postrun'
def read(p):return json.loads(p.read_text())
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def pin(p):return {'path':p.relative_to(ROOT).as_posix(),'sha256':sha(p),'bytes':p.stat().st_size}
def save(name,obj):
 p=POST/name
 with p.open('x') as f:json.dump(obj,f,separators=(',',':'),sort_keys=True);f.write('\n')
 return p

def main():
 projection=read(POST/'terminal-projection.json');result=read(OUT/'result.json')
 if sha(OUT/'result.json')!=projection['result']['sha256']:raise ValueError('Terminal result changed')
 history=read(OUT/'original-native-history.json');metrics=read(OUT/'original-negative-prefix-metrics.json')
 base_paths=sorted((OUT/'prospective').glob('base-ray-*.json'));base=[read(p) for p in base_paths]
 catalog=read(OUT/'prospective/prospective-catalog.json');old=read(OUT/'old-frontier/page00.json')
 witness=(7,92,85);witness_rays=[];branches=[]
 for i,row in enumerate(base):
  if row['sidecar'] and list(witness) in row['sidecar']['blocked_indices_native']:witness_rays.append(i)
 for j in range(result['branch_count']):
  p=OUT/('prospective/branch-%02d/result.json'%j);branch=read(p)
  if branch['full_native_history'][:-1]!=history:raise ValueError('Cannot losslessly deduplicate unequal prefix')
  tail=branch['full_native_history'][-1:]
  compact={k:v for k,v in branch.items() if k!='full_native_history'}
  compact.update(source_record=pin(p),full_native_history_encoding={'prefix':'original_native_history','append':tail})
  changed=[];reasons=Counter();target_reasons=Counter();remaining=set();critical={}
  for i,before in enumerate(base):
   q=OUT/('prospective/branch-%02d/ray-%03d.json'%(j,i));after=read(q);reasons[after['reason']]+=1
   if after['proposal']['family'] in ('proximal_nominal','distal_nominal'):target_reasons[after['reason']]+=1
   previous=[] if before['sidecar'] is None else before['sidecar']['blocked_indices_native']
   later=[] if after['sidecar'] is None else after['sidecar']['blocked_indices_native']
   remaining.update(map(tuple,later))
   if previous!=later or before['reason']!=after['reason']:
    changed.append({'ray':i,'before_source':pin(base_paths[i]),'after_source':pin(q),'before':before,'after':after})
   if i in witness_rays:critical[str(i)]={'source':pin(q),'cells':later,'unchanged':previous==later}
  compact.update(sidecar_changes=changed,after_reasons=dict(reasons),target_ray_reasons=dict(target_reasons),
      unique_current_first_blockers=len(remaining),critical_witness_rays=critical)
  branches.append(compact)
 evidence={'format':'saved-residual-diagnostic-prefix-dedup-v1','result':projection['result'],
  'original_native_history':history,'original_native_history_source':pin(OUT/'original-native-history.json'),
  'original_negative_prefix_metrics_without_duplicate_history':{k:v for k,v in metrics.items() if k!='history'},
  'original_negative_prefix_metrics_source':pin(OUT/'original-negative-prefix-metrics.json'),
  'prefix_progress_controls':read(OUT/'prefix-controls.json'),
  'old_obstruction_rows':[r for r in old['ledger'] if r['family']=='obstruction_opening'],
  'old_accounting':old['obstruction_accounting'],'old_inventory_source':pin(OUT/'old-frontier/page00.json'),
  'prospective_catalog':catalog,'prospective_catalog_source':pin(OUT/'prospective/prospective-catalog.json'),
  'base_proposals_and_first_failure_sidecars':[{k:v for k,v in r.items() if k!='history_if_feasible'} for r in base],
  'base_record_pins':[pin(p) for p in base_paths],
  'all_extra_previews':[read(p) for p in sorted((OUT/'prospective').glob('extra-*.json'))],
  'branch_selection':read(OUT/'prospective/branch-selection.json'),'branches':branches,
  'witness':{'cell':witness,'original_rays':witness_rays,
   'old_dispositions':[r for r in old['ledger'] if r.get('blocker_voxel')==list(witness)],
   'new_dispositions':[r for r in catalog['ledger'] if r.get('blocker_voxel')==list(witness)],
   'new_aperture_tests':[r for r in catalog['accounting']['static_aperture_pairs'] if r['voxel']==list(witness)]},
  'scope':'lossless original47 native history plus exact branch48th records; prospective unexecuted ray records are diagnostics, not accepted route histories'}
 payload=save('causal-evidence.json',evidence)
 manifest=save('output-json-manifest.json',{'files':[pin(p) for p in sorted(OUT.rglob('*.json'))],
  'scope':'hashes of all original emitted JSON; no source arrays or weights read; original run untouched'})
 chosen=[OUT/name for name in ('result.json','costs.json','original-world-control.json','old-authority.json','original-prefix-STOP.json')]
 chosen+=sorted((OUT/'independent').glob('*.json'))
 chosen+=[HERE/'attempt-01.supervision'/name for name in ('receipt.json','declaration.json','worker-final.json','worker.log')]
 chosen+=[HERE/name for name in ('comparison_contract.py','residual_frontier.py','select_worker.py','run_owned.py','freeze-runtime.py',
  'extract_terminal.py','compact_terminal.py','root-release.json','runtime-source-index.json','runner-pins.json','HANDOFF.txt',
  'canonical-controls-01.json','canonical-controls-01.log','helper-controls-01.json','helper-controls-01.log',
  'controls-01.json','controls-01.log','controls-02.json','controls-02.log','promotion-index.json','source.patch')]
 chosen+=[payload,manifest,POST/'terminal-projection.json',POST/'compact-scalars.json']
 chosen=[p for p in sorted(set(chosen)) if p.is_file()]
 total=sum(p.stat().st_size for p in chosen)
 archive=save('lean-archive-list.json',{'files':[pin(p) for p in chosen],'bytes':total,'file_count':len(chosen),
  'requested_ceiling_bytes':2*1024**2,'within_requested_ceiling':total<=2*1024**2,
  'history_encoding':'original47 once, each branch appends its exact native preparation record; no branch history discarded',
  'copy_policy':'only compact derived metadata written; original artifacts unchanged'})
 print(json.dumps({'payload':pin(payload),'manifest':pin(manifest),'archive':pin(archive),'bytes':total,'files':len(chosen),'within_2MiB':total<=2*1024**2}))
if __name__=='__main__':main()
