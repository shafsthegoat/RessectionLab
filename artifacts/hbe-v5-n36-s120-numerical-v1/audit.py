"""Independent saved-run audit. No native execution entry point is used."""
import hashlib, json, math, os, re, stat, subprocess, sys, time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]
OUT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT))
REL=ROOT/'build/hbe-v5-n36-s120-one-shot-release/release.json'
ATT=ROOT/'outputs/mechanics/hbe-v5-remaining-one-shot-v1/06-compression-N36-S120-reference/attempt-01'
REC=ATT/'receipt.json'
HEAD='0f7eee3a17b6743fe15549cf8aba396fa4e98a2c'
REL_SHA='1ea9f55f9d1db13cb214e8eac649713fb39b2492d0721c8bed9921d4cd886a58'
EXPECTED=json.loads((OUT/'expected-terminal.json').read_text())
REC_SHA=EXPECTED['receipt_sha256']
assert EXPECTED['root_terminal_notice_received'] is True
assert hashlib.sha256(Path(__file__).read_bytes()).hexdigest()==EXPECTED['audit_script_sha256']
def sha(p):
 p=Path(p); assert stat.S_ISREG(p.lstat().st_mode)
 h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(1024*1024),b''):h.update(b)
 return h.hexdigest()
def read(p):return json.loads(Path(p).read_text())
def save(name,obj):
 with (OUT/name).open('x') as f:json.dump(obj,f,sort_keys=True,indent=2,allow_nan=False);f.write('\n')
def bind(b):
 p=ROOT/b['path']; assert sha(p)==b['sha256'],str(p)
 if 'bytes' in b: assert p.stat().st_size==b['bytes'],str(p)
def sources(rel,rec):
 assert subprocess.check_output(['/usr/bin/git','rev-parse','HEAD'],cwd=ROOT,text=True).strip()==HEAD
 assert rel['source_commit']==rec['source_commit']==rec['observed_head_preflight']==rec['observed_head_after']==HEAD
 result={}
 for p,b in rel['source_bindings'].items():
  assert p==b['path'];bind(b)
  blob=subprocess.check_output(['/usr/bin/git','cat-file','blob',HEAD+':'+p],cwd=ROOT,timeout=10)
  assert hashlib.sha256(blob).hexdigest()==b['sha256'],p
  result[p]=b['sha256']
 assert len(result)==20
 assert result==rec['source_hashes_before']==rec['source_hashes_after']
 return result

def source_audit():
 started=time.monotonic()
 assert sha(REL)==REL_SHA and sha(REC)==REC_SHA
 rel,rec=read(REL),read(REC)
 sm=sources(rel,rec)
 for k in ('preparation','v5_declaration','source_map','old_source_deck','native_mesh','backend_profile','runtime_identity'):
  bind(rel[k]);assert rel[k]['sha256']==rec[k+'_sha256']
 native={'console.txt','elements.log','nodes.log','solver.log','specimen.feb'}
 names=native|{'readout-console.txt','readout-work-order.json','readout.json'}
 assert set(rec['output_bindings'])==names
 assert {p.name for p in ATT.iterdir()}==names|{'receipt.json'}
 for b in rec['output_bindings'].values():bind(b)
 assert rec['native_output_bindings']=={k:rec['output_bindings'][k] for k in native}
 assert rec['adapted_deck_sha256']==rel['adapted_deck_sha256']==sha(ATT/'specimen.feb')
 total=sum(p.stat().st_size for p in ATT.iterdir())
 assert total<rec['caps']['active_output_bytes']
 for field in ('native','readout'):
  st=rec[field+'_stage'];assert st['exit_code']==0 and st['kill_reason'] is None and st['status']=='completed_within_caps'
  assert st['elapsed_seconds']<st['wall_cap_seconds']==rec['caps'][field+'_wall_seconds']
  assert st['peak_sampled_process_group_rss_bytes']<st['sampled_process_group_rss_cap_bytes']==rec['caps'][field+'_sampled_process_group_rss_bytes']
  assert st['peak_sampled_active_output_bytes']<st['active_output_cap_bytes']==rec['caps']['active_output_bytes']
 assert rec['native_calls_attempted']==rec['readout_calls_attempted']==1 and rec['no_retry'] is True
 assert rec['status']=='passed_numerical_software_only' and rec['runtime_profile_verified_after'] is True
 assert rec['prep_elapsed_seconds']<rec['caps']['prep_wall_seconds']
 assert rec['caps']['native_wall_seconds']==2400 and rec['caps']['readout_wall_seconds']==600
 assert rec['caps']['active_output_bytes']==2*1024**3 and rec['caps']['numerical_threads']==1
 assert rec['saved_numerical_readout']['frame_count']==121
 runtime=read(ROOT/rel['runtime_identity']['path'])
 install_base=Path(runtime['executable']).parents[2]
 for p,h in runtime['libraries'].items():assert sha(install_base/p)==h,p
 bind(runtime['private_openmp'])
 logs={}
 for n in ('console.txt','solver.log'):
  t=(ATT/n).read_text(); selections=[s.strip() for s in t.splitlines() if 'selecting linear solver' in s.lower()]
  assert len(selections)==1 and 'accelerate' in selections[0].lower()
  assert not re.search(r'\b(?:fallback|fall\s+back|switching\s+(?:linear\s+)?solver)\b',t,re.I)
  assert 'N O R M A L   T E R M I N A T I O N' in t
  logs[n]={'selection':selections[0],'normal_termination':True,'no_force_warning_count':t.count('No force acting on the system')}
 order=read(ATT/'readout-work-order.json')
 assert order['source_commit']==HEAD and order['release_sha256']==REL_SHA and order['readout_token_sha256']==rec['readout_token_sha256']
 assert order['run_id']==rel['run_id']==rec['run_id'] and order['ordinal']==6
 assert order['adapter_receipt']==rel['adapter_receipt']
 expected={'source_deck':rel['old_source_deck'],'mesh':rel['native_mesh']}
 for key,name in (('adapted_deck','specimen.feb'),('nodes','nodes.log'),('elements','elements.log'),('solver','solver.log')):
  expected[key]={k:rec['output_bindings'][name][k] for k in ('path','sha256')}
 assert order['bindings']==expected
 from scripts import mechanics_hbe_v5_remaining_one_shot as runner
 from scripts import mechanics_hbe_backend as backend
 from scripts import mechanics_hbe_v5_n12_admission as admission
 assert runner.source_hashes(rel,root=ROOT)==(sm,HEAD)
 runner.validate_preparation(read(ROOT/rel['preparation']['path']))
 profile=backend.verify_profile(ROOT,rel['backend_profile'])
 assert profile['runtime_identity']==rel['runtime_identity']
 prior=runner.validate_prior_chain(rel['prior_receipts'],6,root=ROOT,expected_profile=rel['backend_profile'],expected_runtime=rel['runtime_identity'])
 fields={'native_seconds':'native_wall_seconds','readout_seconds':'readout_wall_seconds','prep_seconds':'prep_wall_seconds','output_bytes':'active_output_bytes','supplement_readout_seconds':'supplement_replay_wall_seconds','supplement_prep_seconds':'supplement_prep_wall_seconds','supplement_output_bytes':'supplement_output_bytes','supplement_replay_calls':'supplement_replay_calls','combined_wall_seconds':'combined_wall_seconds','combined_output_bytes':'combined_output_bytes'}
 for k,v in fields.items():assert prior[k]==rec['prior_'+v],k
 for k,st in (('native_wall_seconds',rec['native_stage']['elapsed_seconds']),('readout_wall_seconds',rec['readout_stage']['elapsed_seconds']),('prep_wall_seconds',rec['prep_elapsed_seconds'])):
  assert rec['aggregate_'+k]==rec['prior_'+k]+st,k
 assert rec['aggregate_native_calls']==7
 assert rec['aggregate_combined_wall_seconds']==sum(rec[k] for k in ('aggregate_native_wall_seconds','aggregate_readout_wall_seconds','aggregate_prep_wall_seconds','aggregate_supplement_replay_wall_seconds','aggregate_supplement_prep_wall_seconds'))
 accepted=admission.verify_exact(root=ROOT)
 runner.audit_imports(root=ROOT)
 result={'status':'passed_source_and_saved_file_audit','source_commit':HEAD,'release_sha256':REL_SHA,'receipt_sha256':REC_SHA,'source_hashes':sm,'verified_release_input_count':7,'verified_output_count':8,'closed_output_bytes':total,'runtime_library_count':len(runtime['libraries']),'private_openmp_verified':True,'logs':logs,'prior_ledger':prior,'n12_admission':accepted,'known_timing_gap':rec['known_timing_gap'],'elapsed_seconds':time.monotonic()-started}
 save('source-audit.json',result)
 print(json.dumps({k:v for k,v in result.items() if k not in ('source_hashes','n12_admission','prior_ledger')},indent=2))

def replay():
 assert read(OUT/'source-audit.json')['status']=='passed_source_and_saved_file_audit'
 rel,rec=read(REL),read(REC);sources(rel,rec)
 from scripts import mechanics_hbe_v5_remaining_one_shot as runner
 from scripts import mechanics_hbe_v5_stream as stream
 runner.audit_imports(root=ROOT)
 order=read(ATT/'readout-work-order.json')
 result=stream.read_bound_run(ROOT,order['run_id'],order['bindings'],order['adapter_receipt'])
 runner.audit_imports(root=ROOT)
 saved=read(ATT/'readout.json')
 assert result==saved,'Entire numerical readout JSON differs'
 assert result['frame_count']==121 and len(result['solver']['states'])==120
 assert result['numerical_passed'] is True
 sources(rel,rec)
 assert sha(REC)==REC_SHA and sha(REL)==REL_SHA
 for b in rec['output_bindings'].values():bind(b)
 save('replayed-readout.json',result)
 save('replay-comparison.json',{'entire_json_equal':True,'all_top_level_keys':sorted(result),'frame_count':result['frame_count'],'numerical_passed':result['numerical_passed'],'native_calls':0,'saved_stream_replay_calls':1,'source_after_matches':True,'saved_outputs_after_match':True,'thread_environment':{k:os.environ.get(k) for k in runner.io.THREAD_ENV}})
 print('Entire numerical readout reproduced; saved inputs and sources unchanged.')

def supervise():
 assert read(OUT/'source-audit.json')['status']=='passed_source_and_saved_file_audit'
 from scripts import mechanics_hbe_v5_remaining_one_shot as runner
 receipt={'schema':'independent-n36-s120-saved-stream-audit-supervision-v1','native_calls':0,'attempts':1,'no_retry':True,'source_commit':HEAD}
 env=runner.io.private_environment()
 command=[str(ROOT/'.venv/bin/python'),'-B',str(OUT/'audit.py'),'replay']
 result=runner.supervise_stage('readout',command,OUT,receipt,cwd=ROOT,environment=env,wall_cap=600,rss_cap=3*1024**3,output_cap=1024**3)
 print(json.dumps(result,indent=2))
 assert result['status']=='completed_within_caps'

if __name__=='__main__':
 {'source':source_audit,'replay':replay,'supervise':supervise}[sys.argv[1]]()
