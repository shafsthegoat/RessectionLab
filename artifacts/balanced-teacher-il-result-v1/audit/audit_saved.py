"""Prepared balanced-IL saved audit; run only after root confirms terminal.
Stdlib scalar math and saved JSON/source; no arrays, checkpoints or models.
"""
import argparse,hashlib,json,math,os,resource,signal,stat,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];P=ROOT/'build/balanced-teacher-il-v1';OUT=Path(__file__).parent
SUBJECTS=['ReMIND-008','ReMIND-010','ReMIND-020','ReMIND-025']
EXPECTED_RELEASE='63acce8556bda8a23552e8007b6851e11d80a022c4920ca3838104df6117e8f3'
seen={};checks=0

def require(ok,msg):
 global checks
 checks+=1
 if not ok:raise AssertionError(msg)
def raw(path):
 path=Path(path);require(path.is_relative_to(ROOT) and '..' not in path.parts and path.suffix in ['.json','.py'],'contained JSON/source only')
 fd=os.open(path,os.O_RDONLY|os.O_NONBLOCK|os.O_NOFOLLOW)
 try:
  st=os.fstat(fd);require(stat.S_ISREG(st.st_mode) and st.st_size<=8*1024**2,'bounded regular file')
  with os.fdopen(fd,'rb',closefd=False) as f:b=f.read(8*1024**2+1)
 finally:os.close(fd)
 require(len(b)<=8*1024**2,'read bound');h=hashlib.sha256(b).hexdigest();key=str(path.relative_to(ROOT));require(key not in seen or seen[key]['sha256']==h,'unchanged input');seen[key]={'sha256':h,'bytes':len(b)};return b
def read(p):return json.loads(raw(p),parse_constant=lambda x:(_ for _ in ()).throw(ValueError(x)))
def sha(p):return hashlib.sha256(raw(p)).hexdigest()
def digest(x):return 'sha256:'+hashlib.sha256(json.dumps(x,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()
def near(a,b,tol=1e-9):return math.isclose(a,b,rel_tol=1e-12,abs_tol=tol)
def identity(h):return [{k:v for k,v in row.items() if k!='outcome_scope'} for row in h]
def write(name,x):
 with (OUT/name).open('x') as f:json.dump(x,f,sort_keys=True,indent=2,allow_nan=False);f.write('\n')

def visit(run,phase,s,context,baseline,final):
 base=run/phase/s;sealed=read(base/'plan.json');plan=sealed['plan'];trace=read(base/'complete-trace.json');replay=read(base/'native-replay.json');m=replay['metrics'];h=m['history'];a=replay['independent_geometry'];d=trace['decisions'];actions=plan['actions']
 require(digest(plan)==sealed['plan_seal'] and digest(h)==a['committed_history_hash'],'plan and replay seal')
 require(plan['context_hash']==trace['context_hash']==digest(context),'complete context identity')
 require(plan['history']==trace['metrics']['history'] and identity(plan['history'])==identity(h),'complete nominal/replay history')
 require(a['accepted'] and a['complete_episode'] and a['geometry']['complete_tool_checked'] and not a['geometry']['failures'],'saved geometry acceptance')
 require(len(actions)==len(h)==len(d)==m['steps'] and m['terminated'],'complete steps')
 require(actions==[x['action_id'] for x in h]==[x['action_id'] for x in d],'action sequence')
 require(plan['terminal_reason']==('STOP' if actions[-1]=='STOP' else 'HORIZON') and (actions[-1]=='STOP' or len(actions)==24),'complete terminal')
 require(a['geometry']['action_count']==sum(x!='STOP' for x in actions),'full-tool action count')
 behavior=d[0]['behavior_parameter_hash'];require(all(x['behavior_parameter_hash']==behavior for x in d),'frozen behavior')
 require(digest({'context':trace['context_hash'],'observations':[x['observation_hash'] for x in d],'history':trace['metrics']['history'],'behavior_parameter_hash':behavior})==trace['trace_seal'],'trace seal')
 require(all(x['action_mask'][x['action_ids'].index(x['action_id'])] for x in d),'saved legal choices')
 require(near(sum(x['reward'] for x in h),m['total_reward']) and all(x['reward']==y['reward'] for x,y in zip(h,d)),'reward sum')
 for field in ['target_removed_mm3','normal_removed_mm3']:require(near(sum(x.get(field,0) for x in h),m[field]) and near(m[field],a['outcomes'][field]),'removal sum')
 region=m['supplied_goal_region'];require(region['fraction_denominator']=='entire_unchanged_supplied_region' and region['unsupported_region_removable'] is False and region['target_modified'] is False and region['occupancy_modified'] is False,'unchanged full target denominator')
 require(near(region['fraction_of_full_region_removed'],m['target_removed_mm3']/region['full_region_membership_mm3']),'fraction arithmetic')
 if phase!='TRAIN-greedy/IL':
  old=baseline['teacher_'+s]['plan']
  require({k:v for k,v in plan.items() if k!='context_hash'}=={k:v for k,v in old.items() if k!='context_hash'},'exact old complete teacher apart from context')
  for step,x in enumerate(d):
   state=baseline[f'state_{s}_{step}'];require(x['action_id']==state['teacher_action'] and x['observation_hash']==state['inputs']['observation_hash'] and x['action_ids']==state['action_ids'] and x['action_mask']==state['action_mask'],'exact teacher state')
 else:require(plan['parameter_hash']==behavior==final and plan['learning_updates']==8,'reloaded fixed endpoint')
 return {'subject':s,'phase':phase,'steps':len(actions),'actions':actions,'terminal':plan['terminal_reason'],'plan_seal':sealed['plan_seal'],'trace_seal':trace['trace_seal'],'context_hash':trace['context_hash'],'public_return':m['total_reward'],'target_mm3':m['target_removed_mm3'],'outside_supplied_goal_mm3':m['normal_removed_mm3'],'target_source_cells':a['outcomes']['positive_target_source_cells_removed'],'target_fraction':region['fraction_of_full_region_removed']}

def score(row,ids,mask,label):
 logits=row['logits'];legal=[i for i,x in enumerate(mask) if x];motion=[i for i in legal if i];require(len(logits)==len(ids)==len(row['probabilities']),'score shape')
 require(all(math.isfinite(logits[i]) for i in legal),'finite logits')
 top=max(logits[i] for i in legal);den=sum(math.exp(logits[i]-top) for i in legal);p=[math.exp(logits[i]-top)/den if mask[i] else 0. for i in range(len(ids))];ce=math.log(den)+top-logits[label]
 require(max(abs(a-b) for a,b in zip(p,row['probabilities']))<=2e-7 and near(row['teacher_CE'],ce,2e-6),'saved float32 probability/CE arithmetic')
 order=sorted(legal,key=lambda i:(-logits[i],i));movement=[i for i in order if i]
 require(row['teacher_rank']==order.index(label)+1 and row['teacher_movement_rank']==(None if label==0 else movement.index(label)+1) and row['greedy_action']==ids[order[0]],'exact rank/tie')
 require(near(row['teacher_probability'],p[label],2e-7) and near(row['stop_probability'],p[0],2e-7) and near(row['teacher_minus_STOP'],logits[label]-logits[0],1e-6),'probability/margin summary')
 if label:
  topm=max(logits[i] for i in motion);conditional=math.exp(logits[label]-topm)/sum(math.exp(logits[i]-topm) for i in motion)
  require(near(row['conditional_teacher_movement_probability'],conditional,2e-7),'conditional movement probability')
 else:require(row['conditional_teacher_movement_probability'] is None,'STOP has no conditional movement probability')
 return {k:row[k] for k in ['teacher_CE','teacher_probability','stop_probability','teacher_rank','teacher_movement_rank','conditional_teacher_movement_probability','teacher_minus_STOP','best_movement_minus_STOP','greedy_action']}

def main(expected_result):
 start=time.monotonic();signal.signal(signal.SIGALRM,lambda *_:(_ for _ in ()).throw(TimeoutError('60s audit cap')));signal.alarm(60)
 R=P/'attempt-01';S=P/'attempt-01.supervision'
 require(len(expected_result)==64 and all(c in '0123456789abcdef' for c in expected_result),'root terminal SHA required')
 require(sha(R/'result.json')==expected_result and sha(P/'root-release.json')==EXPECTED_RELEASE,'exact root pins')
 result=read(R/'result.json');release=read(P/'root-release.json');receipt=read(S/'receipt.json');worker=read(S/'worker-final.json');control=read(S/'endpoint-control.json')
 require(receipt['status']=='complete' and receipt['exit_code']==0 and receipt['worker_termination_confirmed'] and not receipt['final_owned_pids'] and not receipt['cleanup_errors'] and receipt['stop_reason'] is None,'clean parent completion')
 require(receipt['result_sha256']==worker['result_sha256']==worker['canonical_result_sha256']==expected_result and receipt['release_sha256']==EXPECTED_RELEASE,'result/worker/parent joins')
 require(receipt['endpoint_control_sha256']==worker['endpoint_control_sha256']==sha(S/'endpoint-control.json') and receipt['worker_final_sha256']==sha(S/'worker-final.json'),'saved final control joins')
 require(receipt['elapsed_seconds']<3600 and worker['wall_seconds']<3540 and receipt['sampled_peak_rss_bytes']<3*1024**3 and receipt['output_bytes']<128*1024**2,'resource caps')
 require(sha(ROOT/release['source_index']['path'])==release['source_index']['sha256'] and receipt['source_index']==release['source_index'],'source inventory binding')
 index=read(ROOT/release['source_index']['path'])
 for section in ['source_files','metadata_files']:
  for path,h in index[section].items():require(sha(ROOT/path)==h,'source/metadata pin '+path)
 baseline={}
 for key,ref in release['baseline'].items():require(sha(ROOT/ref['path'])==ref['sha256'],'baseline pin');baseline[key]=read(ROOT/ref['path'])
 expected=json.loads(json.dumps(baseline['release']['learning_protocol']));expected['cohort_execution']['il_teacher_weighting']='balanced_STOP_motion_CE_v1'
 require(release['learning_protocol']==expected,'only balanced objective protocol change')
 require(result['status']=='complete_balanced_IL_TRAIN_only' and result['optimizer_updates']=={'IL':8,'RL':0} and result['TRAIN']==SUBJECTS and result['search_calls']==0,'balanced IL only completion')
 require(result['initial_parameter_hash']==baseline['result']['initial_parameter_hash'] and result['SELECT_EVAL_opened'] is False and result['private_reference_reads']==0,'same initialization/public TRAIN only')
 require((result['loss_forward_calls'],result['teacher_logit_forwards'],result['checkpoint_loads'],result['completed_source_visits'])==(40,5,1,44),'fixed execution counts')
 config=read(R/'configuration.json');costs=read(R/'costs.json');pins=read(R/'teacher-pins.json');reload=read(R/'checkpoint-reload.json');endpoint=read(R/'endpoint-teacher-metrics.json')
 require(config['learning_protocol']==release['learning_protocol'] and digest(config['learning_protocol'])==release['learning_protocol_hash'] and config['execution_limits']==release['execution_limits'],'configuration binding')
 final=result['checkpoints']['IL']['parameter_hash'];require(reload['exact_parameter_match'] and reload['parameter_hash']==reload['trained_parameter_hash']==final and reload['completed_updates']==8 and reload['sha256']==result['checkpoints']['IL']['sha256'],'saved reload identity')
 contexts={s:read(R/(s+'-context.json')) for s in SUBJECTS};visits={};curve=[];before=result['initial_parameter_hash']
 for s in SUBJECTS:
  row=visit(R,'teachers',s,contexts[s],baseline,final);visits[('teachers',s)]=row
  require(pins[s]['trace_seal']==row['trace_seal'] and pins[s]['plan_seal']==row['plan_seal'] and pins[s]['steps']==row['steps'] and pins[s]['stop_steps']==row['actions'].count('STOP'),'teacher pin')
 require(sum(x['steps'] for x in pins.values())==5 and sum(x['stop_steps'] for x in pins.values())==4,'fixed four STOP plus one motion labels')
 for u in range(1,9):
  phase=f'IL/update-{u:02d}';update=read(R/phase/'update.json');contributions=[]
  for s in SUBJECTS:
   row=visit(R,phase,s,contexts[s],baseline,final);visits[(phase,s)]=row;c=read(R/phase/s/'gradient-contribution.json');contributions.append(c)
   require(c['context_hash']==digest(contexts[s]) and c['trace_seal']==row['trace_seal']==pins[s]['trace_seal'] and c['plan_seal']==row['plan_seal'] and c['steps']==c['loss_forward_calls']==row['steps'],'complete ordered contribution')
  require(update['context_hashes']==[x['context_hash'] for x in contributions] and update['trace_seals']==[x['trace_seal'] for x in contributions],'all four contexts per update')
  require(update['teacher_group_counts']=={'STOP':4,'motion':1} and update['teacher_group_weights']=={'STOP':.125,'motion':.5} and update['IL_reduction']=='balanced_STOP_motion_CE_v1','balanced group normalization')
  require(update['before_parameter_hash']==before and update['optimizer_updates']==1 and update['completed_updates']==u and update['loss_forward_calls']==5,'shared update chain')
  for field in ['loss','actor_loss','value_loss','entropy','loss_forward_calls']:require(near(update[field],sum(x[field] for x in contributions)),'sum contribution '+field)
  if u==1:
   initial_loss=sum(baseline[f'state_{s}_{step}']['scores']['INITIAL']['teacher_CE']*(.125 if action=='STOP' else .5) for s in SUBJECTS for step,action in enumerate(pins[s]['actions']))
   require(near(update['loss'],initial_loss,2e-6),'initial balanced loss matches original logits')
  before=update['after_parameter_hash'];curve.append({k:update[k] for k in ['completed_updates','loss','gradient_norm_before_clip','before_parameter_hash','after_parameter_hash','teacher_group_counts','teacher_group_weights']})
 require(before==final,'eighth-update endpoint')
 readouts=[];outcomes=[]
 for s in SUBJECTS:
  visits[('teacher-readout',s)]=visit(R,'teacher-readout',s,contexts[s],baseline,final)
  for step,action in enumerate(pins[s]['actions']):
   state=read(R/'teacher-readout'/s/f'state-{step:02d}.json');old=baseline[f'state_{s}_{step}'];ids=state['action_ids'];mask=state['action_mask'];label=ids.index(action)
   require(state['observation_hash']==old['inputs']['observation_hash'] and ids==old['action_ids'] and mask==old['action_mask'] and state['teacher_action']==action,'exact old five state observation/inventory')
   require({k:v for k,v in state['unweighted_IL_8'].items() if k!='conditional_teacher_movement_probability'}==old['scores']['IL_8'],'unchanged saved unweighted comparator')
   row={'subject':s,'step':step,'teacher_action':action}
   for name in ['balanced_IL_8','unweighted_IL_8']:row[name]=score(state[name],ids,mask,label)
   readouts.append(row)
  row=visit(R,'TRAIN-greedy/IL',s,contexts[s],baseline,final);visits[('TRAIN-greedy/IL',s)]=row;e=result['TRAIN_greedy'][s]
  require(e['complete'] and e['checkpoint_reloaded'] and e['parameter_hash']==final and e['actions']==row['actions'] and e['plan_seal']==row['plan_seal'],'reloaded four endpoint join')
  require(near(e['public_return'],row['public_return']) and near(e['target_removed_mm3'],row['target_mm3']) and near(e['outside_supplied_target_removed_mm3'],row['outside_supplied_goal_mm3']),'endpoint scalar join')
  row['unweighted_IL8']=baseline['result']['TRAIN_greedy']['IL'][s];outcomes.append(row)
 for name in ['balanced_IL_8','unweighted_IL_8']:
  stops=[r[name]['teacher_CE'] for r in readouts if r['teacher_action']=='STOP'];motion=[r[name]['teacher_CE'] for r in readouts if r['teacher_action']!='STOP']
  wanted={'STOP_count':4,'motion_count':1,'STOP_mean_NLL':sum(stops)/4,'motion_mean_NLL':motion[0],'unweighted_mean_CE':(sum(stops)+sum(motion))/5,'balanced_mean_CE':.5*sum(stops)/4+.5*motion[0]}
  require(all(near(endpoint[name][k],v) for k,v in wanted.items()),'endpoint group loss arithmetic')
 require(result['endpoint_teacher_metrics']==endpoint and len(readouts)==5,'endpoint metrics copied exactly')
 require(len(visits)==44 and len(costs['completed_patient_visits'])==44 and all(x['source_released'] for x in costs['completed_patient_visits']),'44 released complete visits')
 forwards=45+sum(row['steps'] for row in outcomes);require(costs['total_policy_forward_calls']==forwards<=141,'40 loss + 5 logits + complete greedy forwards')
 require(costs['native_budget']['native_preview_entries']<=385440 and costs['native_budget']['blocked_preview_attempts']==0 and costs['native_budget']['failure'] is None,'executed preview envelope')
 for path,meta in list(seen.items()):require(sha(ROOT/path)==meta['sha256'],'post-read stability')
 summary={'status':'PASS_saved_balanced_IL_metadata_and_scalar_audit','checks':checks,'input_count':len(seen),'source_files':len(index['source_files']),'metadata_files':len(index['metadata_files']),'updates':curve,'endpoint_group_metrics':endpoint,'readouts':readouts,'greedy_outcomes':outcomes,'counts':{'visits':44,'shared_IL_updates':8,'loss_forwards':40,'teacher_logit_forwards':5,'checkpoint_reloads':1,'policy_forwards':forwards,'native_previews':costs['native_budget']['native_preview_entries'],'RL_updates':0,'search_calls':0},'parent':{k:receipt[k] for k in ['elapsed_seconds','sampled_peak_rss_bytes','output_bytes','cleanup_errors','final_owned_pids']},'scope':'Saved scalar/metadata audit only; no weights, arrays, model or geometry execution. Training receipts do not expose per-label NLL after update1: subsequent training group contributions cannot be separately reconstructed. Endpoint group losses are independently recomputed from all five saved vectors. Same five teacher labels and all four TRAIN cases remain; no held-out/clinical claim.','audit_seconds':time.monotonic()-start,'audit_peak_rss_bytes':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss}
 write('audit-result.json',summary);write('input-hashes.json',seen)
 print(json.dumps({k:v for k,v in summary.items() if k not in ['updates','readouts','greedy_outcomes']},sort_keys=True))

if __name__=='__main__':
 parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--terminal-result-sha256',required=True);args=parser.parse_args();main(args.terminal_result_sha256)
