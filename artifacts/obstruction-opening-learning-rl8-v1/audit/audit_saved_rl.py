"""Saved JSON/scalar RL audit. No acquired arrays, tensors or native execution."""
import hashlib,json,math,resource,runpy,signal,struct,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];BASE=Path(__file__).with_name('audit_saved.py')
assert hashlib.sha256(BASE.read_bytes()).hexdigest()=='6f6d5aa74d334b918c16be1c52fd8ebff31a008ddf2ac48421dc122412696931'
A=runpy.run_path(str(BASE));H=A['H']
require=A['require'];read=A['read'];sha=A['sha'];digest=A['digest'];near=A['near'];identity=A['identity'];score=A['score'];episode=A['episode'];pinned=A['pinned']
TRAIN=A['TRAIN'];P=ROOT/'build/obstruction-opening-learning-v1/RL8';OUT=Path(__file__).parent/'RL8'
RESULT='60631a2fb4cebda2d6b29475156e487a8b51b4bd1c5cc1d6ca89dd2511e29897'
RELEASE='276106b7a23caede8a010acb99eeed968b7abbb7f4631b9b91aec33e746b17d6'
def f32(v):return struct.unpack('<f',struct.pack('<f',v))[0]
def close32(a,b):return math.isclose(a,b,abs_tol=2e-6,rel_tol=2**-20)
def write(name,data):
 with (OUT/name).open('x') as f:json.dump(data,f,sort_keys=True,indent=2,allow_nan=False);f.write('\n')
def main():
 start=time.monotonic();signal.signal(signal.SIGALRM,lambda *_:(_ for _ in ()).throw(TimeoutError('60s audit cap')));signal.alarm(60)
 OUT.mkdir(exist_ok=False);R=P/'attempt-01';S=P/'attempt-01.supervision'
 require(sha(R/'result.json')==RESULT and sha(P/'root-release.json')==RELEASE,'exact terminal pins')
 result=read(R/'result.json');release=read(P/'root-release.json');parent=read(S/'receipt.json');worker=read(S/'worker-final.json');control=read(S/'endpoint-control.json')
 require(parent['status']=='complete' and parent['exit_code']==0 and parent['worker_termination_confirmed'] and not parent['cleanup_errors'] and not parent['final_owned_pids'] and parent['stop_reason'] is None,'clean parent completion')
 require(parent['result_sha256']==worker['result_sha256']==worker['canonical_result_sha256']==RESULT and parent['release_sha256']==RELEASE and parent['worker_final_sha256']==sha(S/'worker-final.json'),'terminal joins')
 require(parent['endpoint_control_sha256']==worker['endpoint_control_sha256']==sha(S/'endpoint-control.json'),'endpoint control joins')
 index=pinned(release['source_index']);require(parent['source_index']==release['source_index'],'source index join')
 for section in ('source_files','metadata_files'):
  for path,h in index[section].items():require(sha(ROOT/path)==h,'source/metadata pin '+path)
 source_paths=set(index['source_files'])
 prior={k:pinned(ref) for k,ref in release['inputs'].items()};matched=pinned(release['matched_IL_completion'])
 require(matched['status']=='complete' and matched['exit_code']==0 and matched['worker_termination_confirmed'] and not matched['cleanup_errors'] and not matched['final_owned_pids'],'matched IL terminal prerequisite')
 il=read(ROOT/'build/obstruction-opening-learning-v1/IL64/attempt-01/result.json');require(sha(ROOT/'build/obstruction-opening-learning-v1/IL64/attempt-01/result.json')==matched['result_sha256'],'matched IL result binding')
 require(result['status']=='complete_matched_TRAIN_endpoint' and result['method']=='RL' and result['TRAIN']==TRAIN and result['optimizer_updates']=={'IL':0,'RL':8},'exact RL8 completion')
 require(result['initial_parameter_hash']==il['initial_parameter_hash']==prior['initial_result']['initial_parameter_hash'],'same scratch initialization')
 require(result['SELECT_EVAL_opened'] is False and result['private_reference_reads']==result['search_calls']==result['teacher_trace_reuses']==0 and 'teacher_cache' not in result,'no hidden heldout/cache/search work')
 require((result['loss_forward_calls'],result['teacher_logit_forwards'],result['checkpoint_loads'],result['completed_source_visits'])==(320,7,1,44),'actual endpoint counts')
 require(parent['elapsed_seconds']<3600 and worker['wall_seconds']<3540 and parent['sampled_peak_rss_bytes']<=3*1024**3 and parent['output_bytes']<=128*1024**2,'owned resource caps')
 require(worker['runtime']['threads']==worker['runtime']['interop_threads']==1 and worker['runtime']['device']=='cpu' and worker['runtime']['dtype']=='float32','recorded runtime')
 config=read(R/'configuration.json');costs=read(R/'costs.json');pins=read(R/'teacher-pins.json');reload=read(R/'checkpoint-reload.json');endpoint=read(R/'endpoint-teacher-metrics.json');contexts={s:read(R/(s+'-context.json')) for s in TRAIN}
 protocol=release['learning_protocol'];require(config['learning_protocol']==protocol and digest(protocol)==release['learning_protocol_hash'] and config['execution_limits']==release['execution_limits'] and config['admission_limits']==release['cohort_limits'],'configuration joins')
 require(protocol['gamma']==1 and protocol['value_weight']==.5 and protocol['entropy_weight']==.01 and protocol['updates_per_method']==8,'unchanged RL objective')
 final=result['checkpoints']['RL']['parameter_hash'];require(parent['checkpoints']==result['checkpoints'] and reload['exact_parameter_match'] is True and reload['parameter_hash']==final and reload['sha256']==result['checkpoints']['RL']['sha256'] and reload['completed_updates']==8,'recorded own weight reload exact identity')
 require(control['initial_parameter_hash']==result['initial_parameter_hash'] and control['TRAIN_greedy']==result['TRAIN_greedy'] and control['teacher_metrics']==endpoint==result['endpoint_teacher_metrics'],'endpoint summaries')
 worlds={x['subject']:x['common_public_world'] for x in prior['result']['arms'] if x['condition']=='obstruction_opening'}
 teachers={};readouts=[];routes=[];episodes=[];updates=[];dyn=[];before=result['initial_parameter_hash'];total_steps=0;diagnostics=[]
 for s in TRAIN:
  ctx=contexts[s];require(ctx['role']=='TRAIN' and ctx['subject']==s and ctx['learning_protocol_hash']==digest(protocol) and ctx['occupancy_condition']=='cerebrum_plus_supplied_tumor_assumption','exact TRAIN protocol/world')
  route,sealed,trace,replay=episode(R/'teachers'/s,ctx,worlds[s]['reward'],None);old=prior['plan_'+s]['plan'];actual=sealed['plan']
  require({k:v for k,v in actual.items() if k not in ('context_hash','history')}=={k:v for k,v in old.items() if k not in ('context_hash','history')} and identity(actual['history'])==identity(old['history']),'teacher exact original physical plan')
  pin=pins[s];require(pin['context_hash']==digest(ctx) and pin['trace_seal']==trace['trace_seal'] and pin['plan_seal']==sealed['plan_seal'] and pin['actions']==actual['actions'] and pin['steps']==route['steps'] and pin['stop_steps']==1,'teacher pin')
  require(pin['readout_binding_hash']==digest([[x['observation_hash'],x['action_id'],x['reward'],x['terminated']] for x in trace['decisions']]),'teacher readout binding')
  teachers[s]=(route,sealed,trace,replay)
 require(sum(p['steps'] for p in pins.values())==7 and sum(p['stop_steps'] for p in pins.values())==4,'fixed seven teacher labels')
 for u in range(1,9):
  dest=R/'RL'/f'update-{u:02d}';update=read(dest/'update.json');contributions=[]
  for s in TRAIN:
   route,sealed,trace,replay=episode(dest/s,contexts[s],worlds[s]['reward'],before);c=read(dest/s/'gradient-contribution.json');contributions.append(c);n=route['steps'];total_steps+=n
   require(c['context_hash']==digest(contexts[s]) and c['trace_seal']==trace['trace_seal'] and c['plan_seal']==sealed['plan_seal'] and c['steps']==c['loss_forward_calls']==n and c['actions']==route['actions'] and near(c['return'],route['public_return']),'actual fresh trajectory used for RL loss')
   require(sealed['plan']['learning_updates']==u-1,'collection before current update')
   diag=c['rl_decision_diagnostics'];d=diag['decisions'];td=trace['decisions']
   require(diag['scope']=='same_on_policy_loss_forwards_before_shared_update' and diag['behavior_parameter_hash']==before and diag['episodes_in_update']==4 and diag['value_weight']==.5 and diag['entropy_weight']==.01 and len(d)==n,'on-policy behavior and episode reduction')
   rtg=0.;targets=[]
   for x in reversed(td):rtg=x['reward']+protocol['gamma']*rtg;targets.append(rtg)
   targets.reverse()
   for i,(x,t,target) in enumerate(zip(d,td,targets)):
    require(all(x[k]==t[k] for k in ('observation_hash','action_id','reward','terminated')) and x['step']==i and x['action_index']==t['action_ids'].index(t['action_id']) and x['legal_action_count']==sum(t['action_mask']),'same recorded forward observation/action/legality')
    require(near(x['return_to_go'],target) and x['return_to_go_model_dtype']==f32(target) and x['actor_discount']==protocol['gamma']**i,'independent return-to-go and float32 conversion')
    advantage=f32(x['return_to_go_model_dtype']-x['value']);require(x['detached_advantage']==advantage,'detached advantage from current critic')
    actor=f32(f32(-x['actor_discount']*x['chosen_log_probability'])*advantage);value=f32(f32(x['value']-f32(target))**2)
    require(close32(x['actor_score_term'],actor) and close32(x['value_squared_error'],value),'actor/value scalar equation')
    require(math.isfinite(x['chosen_log_probability']) and x['chosen_log_probability']<=0 and -2e-6<=x['entropy']<=math.log(x['legal_action_count'])+2e-6,'finite probability/entropy range')
    diagnostics.append({'update':u,'subject':s,**x})
   actor=sum(x['actor_score_term']/4 for x in d);value=sum(x['value_squared_error']/(4*n) for x in d);entropy=sum(x['entropy']/(4*n) for x in d)
   require(near(c['actor_loss'],actor) and near(c['value_loss'],value) and near(c['entropy'],entropy) and close32(c['loss'],actor+.5*value-.01*entropy),'episode-normalized RL contribution')
   scope=costs['costs'][f'offline.RL.update-{u:02d}.action_backward.'+s];collect=costs['costs'][f'offline.RL.update-{u:02d}.collection_replay.'+s]
   require(scope['policy_forward_calls']==collect['policy_forward_calls']==n and scope['native_preview_entries']==0 and collect['native_transition_calls']==2*n,'collection/loss/replay call totals')
   episodes.append({'update':u,'subject':s,'behavior_parameter_hash':before,**route})
  require(update['context_hashes']==[c['context_hash'] for c in contributions] and update['trace_seals']==[c['trace_seal'] for c in contributions],'fixed ordered four contributions')
  require(update['before_parameter_hash']==before and update['optimizer_updates']==1 and update['completed_updates']==u and update['method']=='RL' and update['RL_reduction']=='mean_episodes_of_actor_sum_and_value_entropy_step_means','eight shared Adam steps chain')
  for f in ('loss','actor_loss','value_loss','entropy','loss_forward_calls'):require(near(update[f],sum(c[f] for c in contributions)),'summed contribution '+f)
  require(update['parameters_changed']==(update['after_parameter_hash']!=before) and math.isfinite(update['gradient_norm_before_clip']) and update['gradient_norm_before_clip']>=0 and all(math.isfinite(v) and v>=0 for v in update['module_gradient_norms_before_clip'].values()),'finite preclip diagnostics')
  dyn.append({'completed_updates':u,'pre_update_loss':update['loss'],'loss_change_from_previous':None if not dyn else update['loss']-dyn[-1]['pre_update_loss'],'gradient_norm_before_clip':update['gradient_norm_before_clip'],'module_gradient_norms_before_clip':update['module_gradient_norms_before_clip'],'clip_threshold':protocol['max_gradient_norm'],'gradient_exceeds_clip_threshold':update['gradient_norm_before_clip']>protocol['max_gradient_norm'],'before_parameter_hash':before,'after_parameter_hash':update['after_parameter_hash'],'loss_forward_calls':update['loss_forward_calls'],'update_record_sha256':sha(dest/'update.json')})
  before=update['after_parameter_hash'];updates.append(update)
 require(total_steps==320 and before==final and read(R/'training-dynamics.json')['updates']==dyn and sha(R/'training-dynamics.json')==result['training_dynamics_sha256']==control['training_dynamics_sha256'],'all training decisions and eight-update endpoint chain')
 for s in TRAIN:
  rr,ss,tt,rp=episode(R/'teacher-readout'/s,contexts[s],worlds[s]['reward'],None)
  require(ss==teachers[s][1] and tt['trace_seal']==teachers[s][2]['trace_seal'],'fresh readout teacher replay identical')
  for i,t in enumerate(tt['decisions']):
   item=read(R/'teacher-readout'/s/f'state-{i:02d}.json');ids=item['action_ids'];mask=item['action_mask'];action=t['action_id']
   require(item['subject']==s and item['step']==i and item['teacher_action']==action and item['observation_hash']==t['observation_hash'] and ids==t['action_ids'] and mask==t['action_mask'],'same seven diagnostic inputs')
   readouts.append({'subject':s,'step':i,'teacher_action':action,**score(item['scores'],ids,mask,ids.index(action))})
  route,sealed,trace,replay=episode(R/'TRAIN-greedy'/'RL'/s,contexts[s],worlds[s]['reward'],final);row=result['TRAIN_greedy'][s]
  require(row['complete'] is True and row['checkpoint_reloaded'] is True and row['parameter_hash']==final and row['plan_seal']==sealed['plan_seal'] and row['actions']==route['actions'] and row['steps']==route['steps'] and sealed['plan']['learning_updates']==8,'complete reloaded endpoint route')
  for k,v in (('public_return',route['public_return']),('target_removed_mm3',route['target_mm3']),('outside_supplied_target_removed_mm3',route['outside_supplied_goal_mm3'])):require(near(row[k],v),'endpoint route scalar '+k)
  routes.append({'subject':s,**route})
 stops=[x['teacher_CE'] for x in readouts if x['teacher_action']=='STOP'];motion=[x['teacher_CE'] for x in readouts if x['teacher_action']!='STOP']
 wanted={'STOP_count':4,'motion_count':3,'STOP_mean_NLL':sum(stops)/4,'motion_mean_NLL':sum(motion)/3,'unweighted_mean_CE':sum(stops+motion)/7,'balanced_mean_CE':.5*sum(stops)/4+.5*sum(motion)/3,'teacher_action_accuracy':sum(x['teacher_action']==x['greedy_action'] for x in readouts)/7}
 require(all(near(endpoint[k],v,2e-6) for k,v in wanted.items()),'endpoint metrics from saved logits')
 expected_visits=[('offline.fixed_teacher_replay.'+s,s) for s in TRAIN]+[(f'offline.RL.update-{u:02d}.collection_replay.'+s,s) for u in range(1,9) for s in TRAIN]+[('offline.endpoint_teacher_recollection.'+s,s) for s in TRAIN]+[('deployment.reloaded_TRAIN_greedy_replay.'+s,s) for s in TRAIN]
 require([(v['phase'],v['subject']) for v in costs['completed_patient_visits']]==expected_visits and all(v['source_released'] for v in costs['completed_patient_visits']),'44 ordered released source visits')
 require(costs['total_policy_forward_calls']==2*total_steps+7+sum(r['steps'] for r in routes)==651<=1639,'all collection/loss/diagnostic/deployment forwards')
 budget=costs['native_budget'];profile=budget['native_preview_profile']['phases']['unclassified'];require(budget['native_preview_entries']==80356<=385440 and budget['failure'] is None and budget['blocked_preview_attempts']==0 and profile['started']==profile['returned']==profile['feasible']+profile['rejected']==80356 and profile['raised']==0,'complete capped preview accounting')
 # Sources were verified before root released its source hold. Subsequent legal
 # canonical edits do not rewrite that executed snapshot or this saved result.
 for path,meta in list(H['seen'].items()):
  if path not in source_paths:require(sha(ROOT/path)==meta['sha256'],'stable saved evidence '+path)
 positive=[d for d in diagnostics if d['reward']>0];motion_d=[d for d in diagnostics if d['action_id']!='STOP'];stop_d=[d for d in diagnostics if d['action_id']=='STOP']
 credit={'positive_immediate_rewards':len(positive),'positive_rewards_negative_return_to_go':sum(d['return_to_go']<0 for d in positive),'positive_rewards_negative_advantage':sum(d['detached_advantage']<0 for d in positive),'motion_steps':len(motion_d),'STOP_steps':len(stop_d),'episodes_positive_return':sum(e['public_return']>0 for e in episodes),'episodes_zero_return':sum(e['public_return']==0 for e in episodes),'episodes_negative_return':sum(e['public_return']<0 for e in episodes),'positive_reward_rows':positive}
 summary={'status':'PASS_saved_RL8_metadata_scalar_audit','result_sha256':RESULT,'release_sha256':RELEASE,'executed_head':release['expected_head'],'source_index':release['source_index'],'source_files':len(index['source_files']),'metadata_files':len(index['metadata_files']),'checks':require.__globals__['checks'],'input_count':len(H['seen']),'updates':updates,'training_episodes':episodes,'credit_diagnostics':credit,'endpoint_metrics_recomputed':wanted,'teacher_readouts':readouts,'greedy_routes':routes,'counts':{'shared_RL_updates':8,'IL_updates':0,'fresh_training_episodes':32,'loss_forwards':320,'teacher_readout_forwards':7,'policy_forwards':651,'checkpoint_reloads':1,'cache_reuses':0,'source_visits':44,'native_previews':80356},'parent':{k:parent[k] for k in ('elapsed_seconds','sampled_peak_rss_bytes','output_bytes','cleanup_errors','final_owned_pids')},'runtime':worker['runtime'],'audit_seconds':time.monotonic()-start,'audit_peak_rss_bytes':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,'limits':['Saved JSON/history/seal/scalar audit only; no acquired arrays, checkpoint tensors, models or native geometry re-execution.','Parameter/gradient/reload identity is recorded evidence, not independent tensor recomputation. Chosen log probabilities and values are not independently model-recomputed.','Scalar float32 comparisons allow absolute2e-6/relative2^-20 rounding tolerance; returns and model-dtype casts separately verified.','Fixed TRAIN endpoint under unvalidated S union T material; no heldout/clinical, global optimum or equal-compute claim. Return-to-go associations do not isolate the cause of learned STOP.']}
 write('audit-result.json',summary);write('input-hashes.json',H['seen']);print(json.dumps({k:summary[k] for k in ('status','checks','input_count','counts','parent','audit_seconds','audit_peak_rss_bytes')},sort_keys=True))
if __name__=='__main__':main()
