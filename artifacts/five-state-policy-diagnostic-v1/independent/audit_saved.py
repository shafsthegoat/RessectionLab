"""Independent scalar saved-logit and metadata audit; no scientific imports."""
import hashlib,json,math,os,resource,signal,stat,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];P=ROOT/'build/five-state-policy-diagnostic-v1';R=P/'attempt-01';OUT=Path(__file__).parent
PRIOR={1:ROOT/'build/cross-patient-planning-v1/fixed-four-train-pilot-v1',8:ROOT/'build/cross-patient-planning-v1/fixed-four-train-eight-updates-v1'}
SUBJECTS=['ReMIND-008','ReMIND-010','ReMIND-020','ReMIND-025'];MODELS=['INITIAL','IL_1','RL_1','IL_8','RL_8']
start=time.monotonic();signal.signal(signal.SIGALRM,lambda *_:(_ for _ in ()).throw(TimeoutError('60s audit cap')));signal.alarm(60)
seen={};checks=0

def require(ok,msg):
 global checks
 checks+=1
 if not ok:raise AssertionError(msg)
def raw(path):
 path=Path(path);require(path.is_relative_to(ROOT) and '..' not in path.parts and path.suffix in ['.json','.py'],'saved JSON/source only')
 fd=os.open(path,os.O_RDONLY|os.O_NONBLOCK|os.O_NOFOLLOW)
 try:
  st=os.fstat(fd);require(stat.S_ISREG(st.st_mode) and st.st_size<=8*1024**2,'bounded regular input')
  with os.fdopen(fd,'rb',closefd=False) as f:b=f.read(8*1024**2+1)
 finally:os.close(fd)
 require(len(b)<=8*1024**2,'read bound');h=hashlib.sha256(b).hexdigest();key=str(path.relative_to(ROOT));require(key not in seen or seen[key]['sha256']==h,'stable saved input');seen[key]={'sha256':h,'bytes':len(b)};return b
def read(p):return json.loads(raw(p),parse_constant=lambda x:(_ for _ in ()).throw(ValueError(x)))
def sha(p):return hashlib.sha256(raw(p)).hexdigest()
def digest(x):return 'sha256:'+hashlib.sha256(json.dumps(x,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()
def near(a,b,tol=1e-9):return math.isclose(a,b,rel_tol=1e-12,abs_tol=tol)
def identity(h):return [{k:v for k,v in row.items() if k!='outcome_scope'} for row in h]
result=read(R/'result.json');receipt=read(P/'attempt-01.supervision/receipt.json');release=read(P/'root-release.json');index=read(P/'source-index.json');models=read(R/'model-identities.json')
require(sha(R/'result.json')=='9b3bfa60e67277091f6b04c35eea6697013e6cbbca211cb94a5229ec4ac73420','exact root result')
require(receipt['result_sha256']==sha(R/'result.json') and receipt['release_sha256']==sha(P/'root-release.json') and release['source_index']==receipt['source_index'] and release['source_index']['sha256']==sha(P/'source-index.json'),'release/parent joins')
require(receipt['status']=='complete' and receipt['exit_code']==0 and receipt['worker_termination_confirmed'] and not receipt['final_owned_pids'] and not receipt['cleanup_errors'] and receipt['stop_reason'] is None,'clean owned completion')
require(receipt['elapsed_seconds']<release['caps']['parent_seconds'] and receipt['sampled_peak_rss_bytes']<release['caps']['memory_bytes'] and result['wall_seconds']<release['caps']['worker_seconds'],'resource caps')
for section in ['source_files','metadata_files']:
 for path,h in index[section].items():require(sha(ROOT/path)==h,'source/input pin '+path)
prior={u:read(root/'attempt-01/result.json') for u,root in PRIOR.items()}
require(models['parameters']['INITIAL']==prior[1]['initial_parameter_hash']==prior[8]['initial_parameter_hash'],'initial identity')
for u in [1,8]:
 prot=read(PRIOR[u]/'root-release.json')['learning_protocol_hash']
 for method in ['IL','RL']:
  key=f'{method}_{u}';c=models['checkpoints'][key];old=prior[u]['checkpoints'][method]
  require(c['parameter_hash']==models['parameters'][key]==old['parameter_hash'] and c['file_sha256']==old['sha256'] and c['completed_updates']==u and c['learning_protocol_hash']==prot,'checkpoint saved identity '+key)
require((result['policy_forwards'],result['checkpoint_loads'],result['optimizer_updates'],result['teacher_states'],result['complete_greedy_replays'])==(25,4,0,5,4),'diagnostic fixed counts')
require(result['SELECT_EVAL_opened'] is False and result['private_reference_reads']==0,'scope')
rows=[];maxerr={'probability':0.,'CE':0.,'margin':0.};greedy=[];scores_total=commits_total=0
for s in SUBJECTS:
 traces={u:read(PRIOR[u]/'attempt-01/teachers'/s/'complete-trace.json') for u in [1,8]}
 for step in range(2 if s=='ReMIND-025' else 1):
  state=read(R/s/f'state-{step:02d}.json');ids=state['action_ids'];mask=state['action_mask'];teacher=state['teacher_action'];label=ids.index(teacher)
  require(state['subject']==s and state['step']==step and set(state['scores'])==set(MODELS),'five state/model grid')
  for u in [1,8]:
   expected=traces[u]['decisions'][step]
   require(expected['observation_hash']==state['inputs']['observation_hash'] and expected['action_ids']==ids and expected['action_mask']==mask and expected['action_id']==teacher,'exact historical state identity')
  for name in MODELS:
   item=state['scores'][name];logits=item['logits'];probs=item['probabilities'];legal=[i for i,ok in enumerate(mask) if ok]
   require(len(logits)==len(probs)==len(ids) and ids[0]=='STOP' and mask[label],'logit/action shape')
   require(all(math.isfinite(logits[i]) for i in legal) and all(logits[i] is None for i,ok in enumerate(mask) if not ok),'legal finite logits')
   maximum=max(logits[i] for i in legal);den=sum(math.exp(logits[i]-maximum) for i in legal)
   reconstructed=[math.exp(logits[i]-maximum)/den if mask[i] else 0. for i in range(len(ids))]
   ce=math.log(den)+maximum-logits[label];order=sorted(legal,key=lambda i:(-logits[i],i));movement=[i for i in order if i!=0]
   perr=max(abs(a-b) for a,b in zip(probs,reconstructed));cerr=abs(item['teacher_CE']-ce);margin=logits[label]-logits[0];merr=abs(item['teacher_minus_STOP']-margin)
   # Stored source computations were CPU float32; scalar reconstruction is float64.
   require(perr<=2e-7 and cerr<=2e-6 and merr<=1e-6,'float32 logit arithmetic tolerance')
   require(near(sum(probs),1.,2e-7) and item['teacher_probability']==probs[label] and item['stop_probability']==probs[0],'probability summaries')
   require(item['teacher_rank']==order.index(label)+1 and item['teacher_movement_rank']==(None if label==0 else movement.index(label)+1) and item['greedy_action']==ids[order[0]],'exact rank and tie order')
   if movement:require(near(item['best_movement_minus_STOP'],logits[movement[0]]-logits[0],1e-6),'best movement margin')
   maxerr={k:max(maxerr[k],v) for k,v in [('probability',perr),('CE',cerr),('margin',merr)]}
   rows.append({'subject':s,'step':step,'teacher_action':teacher,'model':name,**{k:item[k] for k in ['teacher_probability','stop_probability','teacher_rank','teacher_movement_rank','teacher_CE','teacher_minus_STOP','best_movement_minus_STOP','greedy_action']}})
 selection=read(R/s/'GREEDY/selection.json');acc=selection['accounting'];sealed=read(R/s/'GREEDY/plan.json');plan=sealed['plan'];trace=read(R/s/'GREEDY/complete-trace.json');replay=read(R/s/'GREEDY/native-replay.json');hist=replay['metrics']['history'];a=replay['independent_geometry'];entry=result['patients'][s]
 require(digest(plan)==sealed['plan_seal']==entry['plan_seal'] and digest(hist)==a['committed_history_hash'],'plan/replay seals')
 require(plan['history']==trace['metrics']['history'] and identity(plan['history'])==identity(hist),'exact nominal/native history')
 require(a['accepted'] and a['complete_episode'] and a['geometry']['complete_tool_checked'] and not a['geometry']['failures'],'saved full-tool replay')
 require(selection['actions']==plan['actions']==entry['actions']==entry['saved_SEARCH_actions']==[x['action_id'] for x in traces[8]['decisions']],'exact greedy/SEARCH actions')
 require(identity(hist)==identity(traces[8]['metrics']['history']),'exact greedy/SEARCH physical history')
 require(acc['complete'] and acc['global_optimality_proven'] is False and len(acc['decisions'])==entry['selection_commits']==acc['model_transition_calls'],'complete greedy selection counts')
 for i,d in enumerate(acc['decisions']):
  scores=d['scores'];require(d['all_current_legal_actions_scored'] and d['legal_nonstop_actions']==d['scored_nonstop_actions']==len(scores)-1,'all legal candidates scored')
  best=max(scores,key=lambda x:x['reward']);require(best['action_id']==d['selected_action_id']==plan['actions'][i],'exact maximum/tie selection')
  require(near(best['reward'],hist[i]['reward']),'selection/replay reward')
  scores_total+=len(scores)-1;commits_total+=1
 require(entry['candidate_scores']==acc['evaluated_nonstop_actions']==sum(len(x['scores'])-1 for x in acc['decisions']),'candidate counter')
 require(near(entry['return_value'],sum(x['reward'] for x in hist)) and near(entry['return_value'],acc['estimated_incremental_return']),'greedy return')
 greedy.append({'subject':s,'actions':plan['actions'],'return':entry['return_value'],'plan_seal':entry['plan_seal'],'candidate_scores':entry['candidate_scores'],'selection_commits':entry['selection_commits']})
require(len(rows)==25 and scores_total==129 and commits_total==5,'total scores/commits')
require(result['native_budget']['native_preview_entries']==1234 and not result['native_budget']['blocked_preview_attempts'] and result['native_budget']['failure'] is None,'preview accounting')
require(sum(v.get('policy_forward_calls',0) for v in result['costs'].values())==25,'forward cost accounting')
phase_seconds={suffix:sum(v['complete_wall_seconds'] for k,v in result['costs'].items() if k.endswith(suffix)) for suffix in ['public_reconstruction','fixed_teacher_states','native_greedy_selection','native_greedy_collection_replay']}
means={m:sum(x['teacher_CE'] for x in rows if x['model']==m)/5 for m in MODELS}
for path,meta in list(seen.items()):require(sha(ROOT/path)==meta['sha256'],'final saved/source identity')
summary={'status':'PASS_saved_scalar_and_metadata_audit','checks':checks,'input_count':len(seen),'source_files':len(index['source_files']),'metadata_files':len(index['metadata_files']),'readouts':rows,'mean_five_label_CE':means,'scalar_reconstruction_max_abs_error':maxerr,'scalar_tolerances':{'probability':2e-7,'CE':2e-6,'margin':1e-6},'native_greedy':greedy,'counts':{'policy_forwards':25,'checkpoint_loads':4,'updates':0,'teacher_states':5,'candidate_scores':129,'selection_commits':5,'native_previews':1234},'phase_seconds_inclusive':phase_seconds,'parent':{k:receipt[k] for k in ['elapsed_seconds','sampled_peak_rss_bytes','cleanup_errors','final_owned_pids']},'scope':'Saved scalar logits and metadata only; no arrays/checkpoints/model execution or independent geometry replay. Checkpoint loader evidence is saved metadata. Selection phase excludes public reconstruction/readout/collection/replay and historical training/search; no end-to-end speedup claim.','audit_seconds':time.monotonic()-start,'audit_peak_rss_bytes':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss}
for name,value in [('audit-result.json',summary),('input-hashes.json',seen)]:
 with (OUT/name).open('x') as f:json.dump(value,f,sort_keys=True,indent=2,allow_nan=False);f.write('\n')
print(json.dumps({k:v for k,v in summary.items() if k not in ['readouts','native_greedy']},sort_keys=True))
