"""Saved JSON/source-only audit. No project imports, tensor loads or replay."""
import hashlib, json, math, os, pathlib, resource, signal, stat, time
ROOT=pathlib.Path(__file__).resolve().parents[2]
P=ROOT/'build/cross-patient-planning-v1/fixed-four-train-pilot-v1'
R=P/'attempt-01'; S=P/'attempt-01.supervision'; OUT=pathlib.Path(__file__).parent
start=time.monotonic(); signal.signal(signal.SIGALRM,lambda *_: (_ for _ in ()).throw(TimeoutError('60s audit limit'))); signal.alarm(60)
seen={}; checks=0

def require(ok,msg):
 global checks
 checks+=1
 if not ok: raise AssertionError(msg)
def raw(p):
 p=pathlib.Path(p); require(p.is_relative_to(ROOT),'in-root read')
 require(p.suffix in ('.json','.py'),'JSON/source only')
 fd=os.open(p,os.O_RDONLY|os.O_NOFOLLOW|os.O_NONBLOCK)
 try:
  st=os.fstat(fd); require(stat.S_ISREG(st.st_mode) and st.st_size<=8*1024*1024,'bounded regular input')
  with os.fdopen(fd,'rb',closefd=False) as f:b=f.read(8*1024*1024+1)
 finally:os.close(fd)
 require(len(b)<=8*1024*1024,'read bound'); h=hashlib.sha256(b).hexdigest(); key=str(p.relative_to(ROOT))
 require(key not in seen or seen[key]['sha256']==h,'stable read')
 seen[key]={'sha256':h,'bytes':len(b)};return b

def read(p):return json.loads(raw(p),parse_constant=lambda x: (_ for _ in ()).throw(ValueError(x)))
def digest(x):return 'sha256:'+hashlib.sha256(json.dumps(x,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()
def close(a,b):return math.isclose(a,b,rel_tol=1e-12,abs_tol=1e-9)
def identity(h):return [{k:v for k,v in row.items() if k!='outcome_scope'} for row in h]
result=read(R/'result.json');receipt=read(S/'receipt.json');worker=read(S/'worker-final.json');release=read(P/'root-release.json');index=read(P/'source-index.json');config=read(R/'configuration.json');cost=read(R/'costs.json');ready=read(R/'teacher-readiness.json');freeze=read(R/'checkpoint-freeze.json')
require(seen[str((R/'result.json').relative_to(ROOT))]['sha256']=='6cbc6899cea5ea9c86c913e0ef02f194e81a15b1c451c2d0b8c72aa4e8a5949c','exact result')
require(seen[str((P/'root-release.json').relative_to(ROOT))]['sha256']=='01e428eed049432a051f8f4433b6d1da94fd5ee6d83c84efa2ebfd2e1e692aa7','exact release')
require(seen[str((P/'source-index.json').relative_to(ROOT))]['sha256']==release['source_index']['sha256'],'index binding')
for group in ['source_files','metadata_files']:
 for path,h in index[group].items():require(hashlib.sha256(raw(ROOT/path)).hexdigest()==h,'source/metadata '+path)
subjects=['ReMIND-008','ReMIND-010','ReMIND-020','ReMIND-025'];contexts={s:read(R/(s+'-context.json')) for s in subjects}
require(result['TRAIN_subjects']==subjects and set(ready)==set(subjects),'four fixed subjects')
require(result['optimizer_updates']=={'IL':1,'RL':1},'two shared updates')
require(result['SELECT_EVAL_opened'] is False and result['private_reference_reads']==0 and result['clinical_claim'] is False,'scope')
require(receipt['status']=='complete' and receipt['exit_code']==0 and receipt['worker_termination_confirmed'] and receipt['final_owned_pids']==[] and receipt['cleanup_errors']==[] and receipt['stop_reason'] is None,'clean parent')
require(worker['result_sha256']==worker['canonical_result_sha256']==seen[str((R/'result.json').relative_to(ROOT))]['sha256'],'worker result binding')
require(receipt['elapsed_seconds']<3600 and receipt['sampled_peak_rss_bytes']<3221225472 and receipt['output_bytes']<134217728,'parent caps')
require(config['learning_protocol']==release['learning_protocol'] and digest(config['learning_protocol'])==release['learning_protocol_hash'],'protocol binding')
require(config['learning_protocol']['cohort_execution']['retention_mode']=='return_plus_opening_depth_volume_v1' and config['learning_protocol']['public_target_context_variant']=='full-supplied-public-target-context-v1','declared opt-ins')
visits={};tables=[]
for phase in ['teachers','IL/update-01','RL/update-01','TRAIN-greedy/IL','TRAIN-greedy/RL']:
 for s in subjects:
  base=R/phase/s; sealed=read(base/'plan.json');plan=sealed['plan'];trace=read(base/'complete-trace.json');replay=read(base/'native-replay.json');m=replay['metrics'];a=replay['independent_geometry'];h=m['history'];dec=trace['decisions'];n=len(plan['actions'])
  require(digest(plan)==sealed['plan_seal'],'plan seal')
  require(plan['context_hash']==trace['context_hash']==digest(contexts[s]),'context identity')
  require(plan['history']==trace['metrics']['history'] and identity(plan['history'])==identity(h),'complete nominal/replay equality')
  require(digest(h)==a['committed_history_hash'] and a['accepted'] is True and a['complete_episode'] and a['geometry']['complete_tool_checked'] and a['geometry']['failures']==[],'saved full-tool audit binding')
  require(n==len(h)==len(dec)==m['steps'] and m['terminated'] and a['geometry']['action_count']==n,'full sequence counts')
  require(plan['actions']==[x['action_id'] for x in h]==[x['action_id'] for x in dec],'action identity')
  require(plan['terminal_reason']==('STOP' if plan['actions'][-1]=='STOP' else 'HORIZON') and (plan['actions'][-1]=='STOP' or n==24),'complete terminal')
  require(all(not x['terminated'] for x in dec[:-1]) and dec[-1]['terminated'],'decision termination')
  behavior=dec[0]['behavior_parameter_hash'];require(all(x['behavior_parameter_hash']==behavior for x in dec),'frozen collection parameters')
  require(digest({'context':trace['context_hash'],'observations':[x['observation_hash'] for x in dec],'history':trace['metrics']['history'],'behavior_parameter_hash':behavior})==trace['trace_seal'],'trace seal')
  require(all(x['reward']==y['reward'] for x,y in zip(dec,h)) and close(sum(x['reward'] for x in h),m['total_reward']),'reward sum')
  require(all(x['action_mask'][x['action_ids'].index(x['action_id'])] for x in dec),'chosen action legal in saved inventory')
  for field in ['target_removed_mm3','normal_removed_mm3']:
   require(close(sum(x.get(field,0) for x in h),m[field]) and close(m[field],a['outcomes'][field]),'removal sum '+field)
  target=m['supplied_goal_region'];require(target['fraction_denominator']=='entire_unchanged_supplied_region' and target['unsupported_region_removable'] is False and target['target_modified'] is False and target['occupancy_modified'] is False,'unchanged target/support')
  require(close(target['fraction_of_full_region_removed'],m['target_removed_mm3']/target['full_region_membership_mm3']),'full denominator')
  row={'phase':phase,'subject':s,'steps':n,'terminal':plan['terminal_reason'],'target_mm3':m['target_removed_mm3'],'outside_supplied_goal_mm3':m['normal_removed_mm3'],'public_return':m['total_reward'],'target_source_cells':a['outcomes']['positive_target_source_cells_removed'],'target_fraction':target['fraction_of_full_region_removed'],'plan_seal':sealed['plan_seal'],'trace_seal':trace['trace_seal'],'behavior_parameter_hash':behavior}
  visits[(phase,s)]=(row,plan,trace);tables.append(row)
  if phase=='teachers':
   search=read(base/'search.json');require(search['actions']==plan['actions'] and not search['accounting']['call_cap_reached'] and not search['accounting']['time_cap_reached'],'complete teacher search')
   require(ready[s]['status']=='complete_replayed' and ready[s]['plan_seal']==sealed['plan_seal'] and ready[s]['trace_seal']==trace['trace_seal'] and close(ready[s]['public_return'],m['total_reward']),'teacher readiness')
  if phase.startswith('TRAIN-greedy'):
   method=phase.split('/')[-1];e=result['TRAIN_greedy'][method][s]
   require(e['plan_seal']==sealed['plan_seal'] and e['actions']==plan['actions'] and e['steps']==n and e['complete'] and close(e['public_return'],m['total_reward']),'endpoint result join')
   require(plan['parameter_hash']==behavior==result['checkpoints'][method]['parameter_hash'] and plan['learning_updates']==1,'frozen endpoint')
require([s for s in subjects if ready[s]['positive_nonstop']]==['ReMIND-025'],'positive teacher gate')
updates={}
for method in ['IL','RL']:
 u=read(R/method/'update-01/update.json');contrib=[]
 for s in subjects:
  c=read(R/method/'update-01'/s/'gradient-contribution.json');row,plan,trace=visits[(method+'/update-01',s)];contrib.append(c)
  require(c['context_hash']==digest(contexts[s]) and c['trace_seal']==trace['trace_seal'] and c['plan_seal']==row['plan_seal'] and c['steps']==c['loss_forward_calls']==row['steps'],'contribution binding')
  if method=='IL':require(trace['trace_seal']==visits[('teachers',s)][2]['trace_seal'],'complete teacher recollection')
  else:require(row['behavior_parameter_hash']==result['initial_parameter_hash'],'on-policy shared initialization')
 require(u['context_hashes']==[c['context_hash'] for c in contrib] and u['trace_seals']==[c['trace_seal'] for c in contrib],'ordered four contributions')
 for field in ['loss','actor_loss','value_loss','entropy','loss_forward_calls']:require(close(u[field],sum(c[field] for c in contrib)),'sum contributions '+field)
 require(u['before_parameter_hash']==result['initial_parameter_hash'] and u['after_parameter_hash']==result['checkpoints'][method]['parameter_hash'] and u['parameters_changed'] and u['optimizer_updates']==u['completed_updates']==1 and u['patient_adaptation'] is False,'one global update lineage')
 f=freeze['checkpoints'][method];require(all(f[k]==v for k,v in result['checkpoints'][method].items()),'checkpoint saved metadata')
 require(f['metadata']['initial_parameter_hash']==result['initial_parameter_hash'] and f['metadata']['completed_updates']==1 and f['metadata']['TRAIN_subjects']==subjects and f['metadata']['parameter_hash']==u['after_parameter_hash'],'checkpoint four-case lineage')
 updates[method]={k:u[k] for k in ['loss','actor_loss','value_loss','entropy','loss_forward_calls','gradient_norm_before_clip','before_parameter_hash','after_parameter_hash']}
require(len(cost['completed_patient_visits'])==20 and all(x['source_released'] for x in cost['completed_patient_visits']),'20 released visits')
require(cost['native_budget']['native_preview_entries']==47214<=config['limits']['max_native_previews'] and cost['native_budget']['blocked_preview_attempts']==0 and cost['native_budget']['failure'] is None,'preview cap')
forwards=sum(updates[x]['loss_forward_calls'] for x in updates)+sum(visits[('RL/update-01',s)][0]['steps'] for s in subjects)+sum(visits[('TRAIN-greedy/'+method,s)][0]['steps'] for method in ['IL','RL'] for s in subjects)
require(forwards==cost['total_policy_forward_calls']==169<=config['limits']['max_policy_forwards'],'forward accounting')
# Recheck exact small sources and saved evidence; no binary checkpoint/array opens.
for path,meta in list(seen.items()):require(hashlib.sha256(raw(ROOT/path)).hexdigest()==meta['sha256'],'post-read identity')
summary={'status':'PASS_saved_metadata_consistency','checks':checks,'source_files_verified':len(index['source_files']),'metadata_files_verified':len(index['metadata_files']),'saved_records_and_sources_verified':len(seen),'endpoints':[x for x in tables if x['phase']=='teachers' or x['phase'].startswith('TRAIN-greedy')],'updates':updates,'parent':{k:receipt[k] for k in ['elapsed_seconds','sampled_peak_rss_bytes','output_bytes','final_owned_pids','cleanup_errors']},'costs':{'native_previews':47214,'policy_forwards':forwards,'completed_replayed_visits':20},'scope':'Saved JSON/source consistency and arithmetic only; no arrays, checkpoint contents, policy execution, patient reconstruction or geometry replay. Saved independent geometry results are authenticated, not independently recomputed here. Final greedy used in-memory frozen parameters, not checkpoint reload. TRAIN only; no learned improvement, generalization or clinical validation claim.','audit_elapsed_seconds':time.monotonic()-start,'audit_peak_rss_bytes':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss}
(OUT/'audit-result.json').write_text(json.dumps(summary,sort_keys=True,indent=2,allow_nan=False)+'\n')
(OUT/'input-hashes.json').write_text(json.dumps(seen,sort_keys=True,indent=2)+'\n')
print(json.dumps({k:v for k,v in summary.items() if k not in ['endpoints','updates']},sort_keys=True))
