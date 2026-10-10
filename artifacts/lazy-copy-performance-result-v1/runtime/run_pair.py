"""Run two existing owned children sequentially, then enforce exact saved parity."""
import argparse
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[1]
sys.path.insert(0,str(HERE))
from pilot_contract import (PAIR_OUTPUT,IMPLEMENTATIONS,ROOT,canonical,small,sha,source_guard,
 invariant_search,invariant_replay)

def compare(root):
 arms=[]
 for name in IMPLEMENTATIONS:
  output=root/name;receipt=small(root/(name+'.supervision')/'receipt.json');result=small(output/'result.json');cost=small(output/'costs.json')
  if receipt['status']!='complete' or receipt['result_sha256']!=sha(output/'result.json') or receipt['final_owned_pids'] or receipt['cleanup_errors']:
   raise ValueError('Incomplete owned arm: '+name)
  d=output/'arm-00'
  world=small(d/'world.json');fields=('source_hash','decision_model_hash','initial_observation_hash','context_hash','common_public_world','normalization','derived_occupancy','supplied_goal_extent','proposal_config','proposal_rule_hash')
  identity={k:world[k] for k in fields}
  inventory_index=small(d/'search-inventories.json');inventories=[small(d/e['file']) for e in inventory_index]
  phases=cost['native_budget']['native_preview_profile']['phases']
  profile={key:{k:v for k,v in row.items() if k!='seconds'} for key,row in phases.items()}
  proof={'identity':identity,'initial_inventory':small(d/'initial-inventory.json'),'inventories':inventories,
   'search':invariant_search(small(d/'search.json')),'plan':small(d/'plan.json'),
   'replay':invariant_replay(small(d/'replay.json')),'final_state':result['arms'][0]['final_native_state'],
   'native_counts':result['native_counts'],'native_previews':cost['native_budget']['native_preview_entries'],
   'preview_outcome_profile':profile}
  arms.append((name,proof,receipt,result,cost))
 before,after=arms
 unequal=[key for key in before[1] if canonical(before[1][key])!=canonical(after[1][key])]
 if unequal:raise ValueError('Behavior changed: '+','.join(unequal))
 return {'status':'complete_exact_parity','compared_fields':list(before[1]),
  'arms':{name:{'receipt_sha256':sha(root/(name+'.supervision')/'receipt.json'),
   'result_sha256':sha(root/name/'result.json'),'parent_wall_seconds':rec['elapsed_seconds'],
   'sampled_peak_rss_bytes':rec['sampled_peak_rss_bytes'],'worker_wall_seconds':res['complete_wall_seconds'],
   'native_counts':res['native_counts'],'native_previews':cost['native_budget']['native_preview_entries'],
   'native_preview_seconds':sum(p['seconds'] for p in cost['native_budget']['native_preview_profile']['phases'].values()),
   'greedy_seconds':res['arms'][0]['search_accounting']['planning_seconds'],
   'temporary_mask_copies':cost['temporary_mask_copies']} for name,proof,rec,res,cost in arms},
  'interpretation':'Single sequential pair with identical instrumentation; filesystem-cache/order and sampled-RSS limits apply. No stable speedup or training-speed claim.'}

def main():
 parser=argparse.ArgumentParser();parser.add_argument('--release',type=Path,required=True);parser.add_argument('--release-sha256',required=True);args=parser.parse_args()
 release=small(args.release);source_guard(release,args.release,args.release_sha256)
 root=ROOT/PAIR_OUTPUT
 if root.exists() or root.is_symlink():raise FileExistsError('Fresh pair only; no retry')
 root.mkdir();started=time.monotonic();current=[None];result={'status':'failed_or_unresolved','completed_children':[]}
 def cancel(signum,frame):
  child=current[0]
  if child is not None and child.poll() is None:
   child.send_signal(signal.SIGTERM)
   try:child.wait(timeout=15)
   except subprocess.TimeoutExpired:raise RuntimeError('Owned supervisor did not finish cleanup')
  raise InterruptedError('Pair launcher interrupted')
 old=signal.signal(signal.SIGTERM,cancel)
 try:
  for name in IMPLEMENTATIONS:
   env={**os.environ,'RESECTIONLAB_PERFORMANCE_ARM':name}
   command=[sys.executable,'-I','-B',str(HERE/'run_owned.py'),'--release',str(args.release.resolve()),'--release-sha256',args.release_sha256]
   current[0]=subprocess.Popen(command,cwd=ROOT,env=env)
   if current[0].wait()!=0:raise RuntimeError('Owned child incomplete: '+name)
   result['completed_children'].append(name)
  result=compare(root);source_guard(release,args.release,args.release_sha256)
 except BaseException as error:
  child=current[0]
  if child is not None and child.poll() is None:
   child.send_signal(signal.SIGTERM)
   try:child.wait(timeout=15)
   except subprocess.TimeoutExpired:result['cleanup_failure']='Owned supervisor did not finish within15seconds'
  result.update(status='failed_or_unresolved',failure={'type':type(error).__name__,'message':str(error)});raise
 finally:
  signal.signal(signal.SIGTERM,old);result['pair_wall_seconds']=time.monotonic()-started;result['release_sha256']=args.release_sha256
  with (root/'pair-result.json').open('x') as stream:json.dump(result,stream,indent=2,sort_keys=True,allow_nan=False);stream.write('\n')
 print(json.dumps(result,indent=2))
if __name__=='__main__':main()
