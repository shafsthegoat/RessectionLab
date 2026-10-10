"""Generated real-process setup control; no scientific imports, arrays or models.

Actual pair/parent source runs unchanged. Only the scientific worker and its
source/endpoint guard boundary are generated stubs. Real owned process sampling,
lease, cleanup, declaration and receipts are exercised in a temporary checkout.
The dictionary uses the real frozen v1 release template's exact fields, so the
old declaration must fail instead of receiving a permissive missing-key default.
"""
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import time
HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[1]
OLD=ROOT/'build/lazy-copy-train025-pair-v1'

def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def write(path,value):path.write_text(json.dumps(value,indent=2,sort_keys=True)+'\n')

CONTRACT='''
import hashlib,json,os
from pathlib import Path
HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[1]
IMPLEMENTATIONS=('baseline','lazy_copy');IMPLEMENTATION=os.environ.get('RESECTIONLAB_PERFORMANCE_ARM','baseline')
PAIR_OUTPUT='generated-attempt';OUTPUT=PAIR_OUTPUT+'/'+IMPLEMENTATION
TRAIN=('GENERATED',);CLOSED={'all_acquired':True}
WORKER_SECONDS=3;PARENT_SECONDS=5;MEMORY_BYTES=3*1024**3;OUTPUT_BYTES=1024**2;SUPERVISION_BYTES=16*1024**2
canonical=lambda v:json.dumps(v,sort_keys=True,separators=(',',':'))
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def small(p):return json.loads(Path(p).read_text())
def source_guard(release,path,digest):
    assert sha(path)==digest
    assert 'execution_limits_per_child' in release and 'execution_limits' not in release
    return {'source_files':{'generated_boundary':digest}}
def complete_result(result):return result['status']=='complete_generated' and result['implementation']==IMPLEMENTATION
def endpoint_control(output,release):return {'generated_boundary':True,'implementation':IMPLEMENTATION}
def invariant_search(value):return value
def invariant_replay(value):return value
'''
WORKER='''
import argparse,json,os,time
from pathlib import Path
import sys
HERE=Path(__file__).resolve().parent;sys.path.insert(0,str(HERE))
from pilot_contract import *
p=argparse.ArgumentParser();p.add_argument('--release');p.add_argument('--release-sha256');args=p.parse_args()
lease=int(os.environ['RESECTIONLAB_PARENT_LEASE_FD']);os.fstat(lease)
output=ROOT/OUTPUT;output.mkdir();d=output/'arm-00';d.mkdir();sup=output.with_name(output.name+'.supervision')
def write(path,value):
    with path.open('x') as stream:json.dump(value,stream,sort_keys=True);stream.write('\\n')
world={k:'GENERATED_ONLY' for k in ('source_hash','decision_model_hash','initial_observation_hash','context_hash','common_public_world','normalization','derived_occupancy','supplied_goal_extent','proposal_config','proposal_rule_hash')}
write(d/'world.json',world);write(d/'initial-inventory.json',{'generated_inventory':True})
write(d/'search-inventories.json',[]);write(d/'search.json',{'generated_search':True})
write(d/'plan.json',{'generated_plan':True});write(d/'replay.json',{'generated_replay':True})
result={'status':'complete_generated','implementation':IMPLEMENTATION,'complete_wall_seconds':0.,'generated_no_science':True,'parent_lease_present':True,
 'native_counts':{'search':0,'rollout':0,'replay':0},'arms':[{'final_native_state':{'generated_state':True},'search_accounting':{'planning_seconds':0.}}]}
write(output/'result.json',result)
write(output/'costs.json',{'native_budget':{'native_preview_entries':0,'native_preview_profile':{'phases':{'generated':{'seconds':0.,'started':0}}}},'temporary_mask_copies':{'copy_calls':0,'copied_bytes':0}})
write(sup/'endpoint-control.json',endpoint_control(output,small(args.release)))
write(sup/'worker-final.json',{'status':'complete_owned_paired_TRAIN_obstruction','result_sha256':sha(output/'result.json'),'canonical_result_sha256':sha(output/'result.json'),'endpoint_control_sha256':sha(sup/'endpoint-control.json')})
time.sleep(.3) # Only ensures the real parent samples this generated child.
'''

def run_case(path, *, broken):
 folder=path/'build/driver';folder.mkdir(parents=True)
 for name in ('run_pair.py','run_owned.py'):
  shutil.copyfile((OLD if broken else HERE)/name,folder/name)
 (folder/'pilot_contract.py').write_text(CONTRACT);(folder/'search_worker.py').write_text(WORKER)
 helpers=path/'build/goal-conditioned-policy-v1';helpers.mkdir()
 for name in ('run_contact_owned.py','darwin_fast_sampler.py'):
  shutil.copyfile(ROOT/'build/goal-conditioned-policy-v1'/name,helpers/name)
 release=json.loads((OLD/'release-template.json').read_text());release['status']='released_one_attempt'
 release_path=folder/'generated-release.json';write(release_path,release)
 env={**os.environ,**{k:'1' for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','VECLIB_MAXIMUM_THREADS','NUMEXPR_NUM_THREADS')}}
 env.pop('RESECTIONLAB_PERFORMANCE_ARM',None)
 with (path/'console.log').open('x') as log:
  completed=subprocess.run([sys.executable,'-I','-B',str(folder/'run_pair.py'),'--release',str(release_path),'--release-sha256',sha(release_path)],cwd=path,env=env,stdout=log,stderr=subprocess.STDOUT,timeout=12)
 out=path/'generated-attempt';pair=json.loads((out/'pair-result.json').read_text());console=(path/'console.log').read_text()
 if broken:
  assert completed.returncode!=0 and pair['completed_children']==[]
  assert "KeyError: 'execution_limits'" in console
  assert not (out/'baseline.supervision/worker.log').exists() and not (out/'baseline').exists()
  return {'old_failure_reproduced_before_scientific_boundary':True,'exit_code':completed.returncode}
 assert completed.returncode==0,console
 assert pair['status']=='complete_exact_parity'
 for arm in ('baseline','lazy_copy'):
  sup=out/(arm+'.supervision');dec=json.loads((sup/'declaration.json').read_text());rec=json.loads((sup/'receipt.json').read_text());res=json.loads((out/arm/'result.json').read_text())
  assert dec['execution_limits']==release['execution_limits_per_child']
  assert rec['status']=='complete' and rec['samples']>0 and rec['exit_code']==0 and rec['worker_termination_confirmed']
  assert not rec['final_owned_pids'] and not rec['cleanup_errors']
  assert rec['result_sha256']==sha(out/arm/'result.json')
  assert res['implementation']==arm and res['parent_lease_present'] and res['generated_no_science']
 return {'both_real_generated_children_reached_and_reaped':True,'exact_parent_declaration_fields':True,'both_durable_receipts_complete':True,'actual_pair_comparison_complete':True}

def main():
 report=HERE/'setup-controls.json';log=HERE/'setup-controls.log'
 if report.exists() or log.exists():raise FileExistsError('Preserve prior generated attempt')
 started=time.monotonic();record={'status':'FAIL','scope':'actual pair+owned parent setup, generated child only; no scientific runtime/patient/model imports','threads':1,'per_case_timeout_seconds':12}
 try:
  with tempfile.TemporaryDirectory(prefix='lazy-copy-owned-setup-') as tmp:
   root=Path(tmp);a=root/'old';b=root/'fixed';a.mkdir();b.mkdir()
   record['old']=run_case(a,broken=True);record['fixed']=run_case(b,broken=False)
   log.write_text('OLD (expected failure):\n'+(a/'console.log').read_text()+'\nFIXED:\n'+(b/'console.log').read_text())
  record['status']='PASS'
 except BaseException as error:record['failure']={'type':type(error).__name__,'message':str(error)};raise
 finally:
  record['wall_seconds']=time.monotonic()-started
  record['source_pins']={f:sha(HERE/f) for f in ('pilot_contract.py','search_worker.py','run_owned.py','run_pair.py','freeze-runtime.py','test_owned_setup.py')}
  write(report,record)
 print(json.dumps(record,indent=2))
if __name__=='__main__':main()
