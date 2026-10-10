"""Prepared fixed64 balanced-IL saved audit; run only after root confirms terminal.
Stdlib scalar math and saved JSON/source; no arrays, checkpoints or models.
"""
import argparse,hashlib,json,math,os,resource,signal,stat,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];P=ROOT/'build/balanced-teacher-il64-v1';OUT=Path(__file__).parent
SUBJECTS=['ReMIND-008','ReMIND-010','ReMIND-020','ReMIND-025']
PROBABILITY_ABSOLUTE_TOLERANCE=2**-22 # Four float32 unit roundoffs for bounded probabilities.
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
 else:require(plan['parameter_hash']==behavior==final and plan['learning_updates']==64,'reloaded fixed64 endpoint')
 return {'subject':s,'phase':phase,'steps':len(actions),'actions':actions,'terminal':plan['terminal_reason'],'plan_seal':sealed['plan_seal'],'trace_seal':trace['trace_seal'],'context_hash':trace['context_hash'],'physical_history_hash':digest(identity(h)),'public_return':m['total_reward'],'target_mm3':m['target_removed_mm3'],'outside_supplied_goal_mm3':m['normal_removed_mm3'],'target_source_cells':a['outcomes']['positive_target_source_cells_removed'],'target_fraction':region['fraction_of_full_region_removed']}

def score(row,ids,mask,label):
 logits=row['logits'];legal=[i for i,x in enumerate(mask) if x];motion=[i for i in legal if i];require(len(logits)==len(ids)==len(row['probabilities']),'score shape')
 require(ids[0]=='STOP' and mask[0] and all(logits[i] is None and row['probabilities'][i]==0 for i,x in enumerate(mask) if not x),'masked score convention')
 require(all(math.isfinite(logits[i]) for i in legal),'finite logits')
 top=max(logits[i] for i in legal);den=sum(math.exp(logits[i]-top) for i in legal);p=[math.exp(logits[i]-top)/den if mask[i] else 0. for i in range(len(ids))];ce=math.log(den)+top-logits[label]
 probability_error=max(abs(a-b) for a,b in zip(p,row['probabilities']))
 require(probability_error<=PROBABILITY_ABSOLUTE_TOLERANCE and near(row['teacher_CE'],ce,2e-6),'saved float32 probability/CE arithmetic')
 order=sorted(legal,key=lambda i:(-logits[i],i));movement=[i for i in order if i]
 require(row['teacher_rank']==order.index(label)+1 and row['teacher_movement_rank']==(None if label==0 else movement.index(label)+1) and row['greedy_action']==ids[order[0]],'exact rank/tie')
 require(near(row['teacher_probability'],p[label],PROBABILITY_ABSOLUTE_TOLERANCE) and near(row['stop_probability'],p[0],PROBABILITY_ABSOLUTE_TOLERANCE) and near(row['teacher_minus_STOP'],logits[label]-logits[0],1e-6),'probability/margin summary')
 require(near(row['best_movement_minus_STOP'],max(logits[i] for i in motion)-logits[0],1e-6) if motion else row['best_movement_minus_STOP'] is None,'best movement margin')
 if label:
  topm=max(logits[i] for i in motion);conditional=math.exp(logits[label]-topm)/sum(math.exp(logits[i]-topm) for i in motion)
  require(near(row['conditional_teacher_movement_probability'],conditional,PROBABILITY_ABSOLUTE_TOLERANCE),'conditional movement probability')
 else:require(row['conditional_teacher_movement_probability'] is None,'STOP has no conditional movement probability')
 return {**{k:row[k] for k in ['teacher_CE','teacher_probability','stop_probability','teacher_rank','teacher_movement_rank','conditional_teacher_movement_probability','teacher_minus_STOP','best_movement_minus_STOP','greedy_action']},'scalar_probability_max_absolute_error':probability_error,'scalar_CE_absolute_error':abs(row['teacher_CE']-ce)}

def reference_route(sealed,saved=None):
 plan=sealed['plan'];h=plan['history'];actions=plan['actions']
 require(digest(plan)==sealed['plan_seal'] and actions==[x['action_id'] for x in h],'historical complete plan seal')
 require(actions and (actions[-1]=='STOP' or len(actions)==plan['max_steps']),'historical complete terminal')
 row={'actions':actions,'steps':len(actions),'terminal':plan['terminal_reason'],'plan_seal':sealed['plan_seal'],'physical_history_hash':digest(identity(h)),
      'public_return':sum(x['reward'] for x in h),'target_mm3':sum(x.get('target_removed_mm3',0) for x in h),'outside_supplied_goal_mm3':sum(x.get('normal_removed_mm3',0) for x in h)}
 if saved is not None:
  require(saved['complete'] and saved['steps']==len(actions) and saved['plan_seal']==sealed['plan_seal'] and saved['actions']==actions,'historical endpoint result/plan join')
  require(near(saved['public_return'],row['public_return']),'historical endpoint return sum')
  # Original unweighted8 result rows contain no removal totals: these are
  # reconstructed from their authenticated complete plan history instead.
  for old,new in [('target_removed_mm3','target_mm3'),('outside_supplied_target_removed_mm3','outside_supplied_goal_mm3')]:
   if old in saved:require(near(saved[old],row[new]),'historical endpoint removal sum')
 return row

def route_comparison(current,prior):
 a=current['actions'];b=prior['actions'];first=next((i for i,(x,y) in enumerate(zip(a,b)) if x!=y),None)
 if first is None and len(a)!=len(b):first=min(len(a),len(b))
 return {'reference':prior,'same_actions':a==b,'same_physical_history':current['physical_history_hash']==prior['physical_history_hash'],
         'first_different_action_step_0based':first,'return_difference':current['public_return']-prior['public_return'],
         'target_mm3_difference':current['target_mm3']-prior['target_mm3'],'outside_mm3_difference':current['outside_supplied_goal_mm3']-prior['outside_supplied_goal_mm3']}

def main(expected_result,expected_release):
 start=time.monotonic();signal.signal(signal.SIGALRM,lambda *_:(_ for _ in ()).throw(TimeoutError('60s audit cap')));signal.alarm(60)
 R=P/'attempt-01';S=P/'attempt-01.supervision'
 require(len(expected_result)==64 and all(c in '0123456789abcdef' for c in expected_result),'root terminal SHA required')
 require(len(expected_release)==64 and all(c in '0123456789abcdef' for c in expected_release),'root release SHA required')
 require(sha(R/'result.json')==expected_result and sha(P/'root-release.json')==expected_release,'exact root pins')
 result=read(R/'result.json');release=read(P/'root-release.json');receipt=read(S/'receipt.json');worker=read(S/'worker-final.json');control=read(S/'endpoint-control.json')
 require(receipt['status']=='complete' and receipt['exit_code']==0 and receipt['worker_termination_confirmed'] and not receipt['final_owned_pids'] and not receipt['cleanup_errors'] and receipt['stop_reason'] is None,'clean parent completion')
 require(receipt['result_sha256']==worker['result_sha256']==worker['canonical_result_sha256']==expected_result and receipt['release_sha256']==expected_release,'result/worker/parent joins')
 require(receipt['endpoint_control_sha256']==worker['endpoint_control_sha256']==sha(S/'endpoint-control.json') and receipt['worker_final_sha256']==sha(S/'worker-final.json'),'saved final control joins')
 require(receipt['elapsed_seconds']<3600 and worker['wall_seconds']<3540 and receipt['sampled_peak_rss_bytes']<3*1024**3 and receipt['output_bytes']<128*1024**2,'resource caps')
 require(sha(ROOT/release['source_index']['path'])==release['source_index']['sha256'] and receipt['source_index']==release['source_index'],'source inventory binding')
 index=read(ROOT/release['source_index']['path'])
 for section in ['source_files','metadata_files']:
  for path,h in index[section].items():require(sha(ROOT/path)==h,'source/metadata pin '+path)
 baseline={}
 for key,ref in release['baseline'].items():require(sha(ROOT/ref['path'])==ref['sha256'],'baseline pin');baseline[key]=read(ROOT/ref['path'])
 balanced={}
 for key,ref in release['balanced_baseline'].items():require(sha(ROOT/ref['path'])==ref['sha256'],'balanced8 baseline pin');balanced[key]=read(ROOT/ref['path'])
 expected=json.loads(json.dumps(baseline['release']['learning_protocol']));expected['cohort_execution']['il_teacher_weighting']='balanced_STOP_motion_CE_v1'
 expected['updates_per_method']=64
 require(release['learning_protocol']==expected,'only balanced objective/count change against unweighted8')
 require(release['learning_protocol']=={**balanced['release']['learning_protocol'],'updates_per_method':64},'only update count changes against balanced8')
 require(result['status']=='complete_balanced_IL64_TRAIN_only' and result['optimizer_updates']=={'IL':64,'RL':0} and result['TRAIN']==SUBJECTS and result['search_calls']==0,'balanced IL64 only completion')
 require(result['selection_readiness']['ready'] is True and result['selection_readiness']['execution_admitted'] is False and result['teacher_statuses']=={s:'complete_replayed' for s in SUBJECTS},'complete fixed teacher/endpoint gate without heldout admission')
 require(result['initial_parameter_hash']==baseline['result']['initial_parameter_hash'] and result['SELECT_EVAL_opened'] is False and result['private_reference_reads']==0,'same initialization/public TRAIN only')
 require(result['initial_parameter_hash']==balanced['result']['initial_parameter_hash'],'exact balanced8 initial tensors')
 require((result['loss_forward_calls'],result['teacher_logit_forwards'],result['checkpoint_loads'],result['completed_source_visits'])==(320,5,1,268),'fixed64 execution counts')
 config=read(R/'configuration.json');costs=read(R/'costs.json');pins=read(R/'teacher-pins.json');reload=read(R/'checkpoint-reload.json');endpoint=read(R/'endpoint-teacher-metrics.json')
 require(config['learning_protocol']==release['learning_protocol'] and digest(config['learning_protocol'])==release['learning_protocol_hash'] and config['execution_limits']==release['execution_limits'] and config['admission_limits']==release['cohort_limits'],'configuration binding')
 final=result['checkpoints']['IL']['parameter_hash'];require(receipt['checkpoints']==result['checkpoints'] and reload['exact_parameter_match'] and reload['parameter_hash']==reload['trained_parameter_hash']==final and reload['completed_updates']==64 and reload['sha256']==result['checkpoints']['IL']['sha256'],'saved reload identity')
 contexts={s:read(R/(s+'-context.json')) for s in SUBJECTS};visits={};curve=[];before=result['initial_parameter_hash']
 for s in SUBJECTS:
  row=visit(R,'teachers',s,contexts[s],baseline,final);visits[('teachers',s)]=row
  require(pins[s]['trace_seal']==row['trace_seal'] and pins[s]['plan_seal']==row['plan_seal'] and pins[s]['steps']==row['steps'] and pins[s]['stop_steps']==row['actions'].count('STOP'),'teacher pin')
 require(sum(x['steps'] for x in pins.values())==5 and sum(x['stop_steps'] for x in pins.values())==4,'fixed four STOP plus one motion labels')
 parity=[];dynamics=[]
 for u in range(1,65):
  phase=f'IL/update-{u:02d}';update=read(R/phase/'update.json');contributions=[]
  for s in SUBJECTS:
   row=visit(R,phase,s,contexts[s],baseline,final);visits[(phase,s)]=row;c=read(R/phase/s/'gradient-contribution.json');contributions.append(c)
   require(c['context_hash']==digest(contexts[s]) and c['trace_seal']==row['trace_seal']==pins[s]['trace_seal'] and c['plan_seal']==row['plan_seal'] and c['steps']==c['loss_forward_calls']==row['steps'],'complete ordered contribution')
  require(update['context_hashes']==[x['context_hash'] for x in contributions] and update['trace_seals']==[x['trace_seal'] for x in contributions],'all four contexts per update')
  require(update['teacher_group_counts']=={'STOP':4,'motion':1} and update['teacher_group_weights']=={'STOP':.125,'motion':.5} and update['IL_reduction']=='balanced_STOP_motion_CE_v1','balanced group normalization')
  require(update['before_parameter_hash']==before and update['optimizer_updates']==1 and update['completed_updates']==u and update['loss_forward_calls']==5,'shared update chain')
  require(update['parameters_changed']==(update['after_parameter_hash']!=before) and math.isfinite(update['gradient_norm_before_clip']) and update['gradient_norm_before_clip']>=0 and all(math.isfinite(v) and v>=0 for v in update['module_gradient_norms_before_clip'].values()),'finite recorded gradient dynamics')
  if u<=8:
   excluded={'context_hashes','trace_seals'};old=balanced[f'update_{u:02d}']
   require({k:v for k,v in update.items() if k not in excluded}=={k:v for k,v in old.items() if k not in excluded},'exact first8 full numerical/parameter control')
   parity.append({'update':u,'actual_sha256':sha(R/phase/'update.json'),'baseline_sha256':release['balanced_baseline'][f'update_{u:02d}']['sha256'],'loss':update['loss'],'gradient_norm_before_clip':update['gradient_norm_before_clip'],'before_parameter_hash':update['before_parameter_hash'],'after_parameter_hash':update['after_parameter_hash']})
  for field in ['loss','actor_loss','value_loss','entropy','loss_forward_calls']:require(near(update[field],sum(x[field] for x in contributions)),'sum contribution '+field)
  if u==1:
   initial_loss=sum(baseline[f'state_{s}_{step}']['scores']['INITIAL']['teacher_CE']*(.125 if action=='STOP' else .5) for s in SUBJECTS for step,action in enumerate(pins[s]['actions']))
   require(near(update['loss'],initial_loss,2e-6),'initial balanced loss matches original logits')
  dynamics.append({'completed_updates':u,'pre_update_loss':update['loss'],'loss_change_from_previous':None if not dynamics else update['loss']-dynamics[-1]['pre_update_loss'],
   'gradient_norm_before_clip':update['gradient_norm_before_clip'],'module_gradient_norms_before_clip':update['module_gradient_norms_before_clip'],
   'clip_threshold':release['learning_protocol']['max_gradient_norm'],'gradient_exceeds_clip_threshold':update['gradient_norm_before_clip']>release['learning_protocol']['max_gradient_norm'],
   'before_parameter_hash':update['before_parameter_hash'],'after_parameter_hash':update['after_parameter_hash'],'loss_forward_calls':update['loss_forward_calls'],'update_record_sha256':sha(R/phase/'update.json')})
  before=update['after_parameter_hash'];curve.append({k:update[k] for k in ['completed_updates','loss','gradient_norm_before_clip','module_gradient_norms_before_clip','before_parameter_hash','after_parameter_hash','teacher_group_counts','teacher_group_weights']})
 require(before==final,'fixed64 endpoint')
 require(result['first_eight_update_control']=={'exact_match':True,'updates':parity,'excluded_metadata_fields':['context_hashes','trace_seals'],'scope':'all remaining numerical/scalar update fields and canonical full tensor hashes identical'},'saved first8 control exactly reconstructed')
 require(read(R/'first-eight-update-control.json')==control['first_eight_update_control']==result['first_eight_update_control'],'first8 persisted and parent joins')
 require(read(R/'training-dynamics.json')=={'updates':dynamics,'loss_scope':'each receipt is before its update; final teacher metrics are after update64','schedule':'all64 updates mandatory; diagnostics never select an endpoint or stop training'},'all64 dynamics exactly reconstructed')
 require(result['training_dynamics_updates']==64 and sha(R/'training-dynamics.json')==result['training_dynamics_sha256']==control['training_dynamics_sha256'],'training dynamics hash/count joins')
 require(control['balanced64_parameter_hash']==final and control['balanced8_parameter_hash']==balanced['result']['checkpoints']['IL']['parameter_hash'] and control['balanced64_TRAIN_greedy']==result['TRAIN_greedy'] and control['balanced8_TRAIN_greedy']==balanced['result']['TRAIN_greedy'],'parent endpoint comparison joins')
 readouts=[];outcomes=[]
 for s in SUBJECTS:
  visits[('teacher-readout',s)]=visit(R,'teacher-readout',s,contexts[s],baseline,final)
  for step,action in enumerate(pins[s]['actions']):
   state=read(R/'teacher-readout'/s/f'state-{step:02d}.json');old=baseline[f'state_{s}_{step}'];ids=state['action_ids'];mask=state['action_mask'];label=ids.index(action)
   require(state['observation_hash']==old['inputs']['observation_hash'] and ids==old['action_ids'] and mask==old['action_mask'] and state['teacher_action']==action,'exact old five state observation/inventory')
   require({k:v for k,v in state['unweighted_IL_8'].items() if k!='conditional_teacher_movement_probability'}==old['scores']['IL_8'],'unchanged saved unweighted comparator')
   old_balanced=balanced[f'state_{s}_{step}']
   require(state['balanced_IL_8']==old_balanced['balanced_IL_8'] and all(state[k]==old_balanced[k] for k in ['observation_hash','action_ids','action_mask','teacher_action','subject','step']),'unchanged balanced8 vector on exact same state')
   row={'subject':s,'step':step,'teacher_action':action}
   for name in ['balanced_IL_64','balanced_IL_8','unweighted_IL_8']:row[name]=score(state[name],ids,mask,label)
   readouts.append(row)
  row=visit(R,'TRAIN-greedy/IL',s,contexts[s],baseline,final);visits[('TRAIN-greedy/IL',s)]=row;e=result['TRAIN_greedy'][s]
  require(e['complete'] and e['checkpoint_reloaded'] and e['parameter_hash']==final and e['actions']==row['actions'] and e['plan_seal']==row['plan_seal'],'reloaded four endpoint join')
  require(near(e['public_return'],row['public_return']) and near(e['target_removed_mm3'],row['target_mm3']) and near(e['outside_supplied_target_removed_mm3'],row['outside_supplied_goal_mm3']),'endpoint scalar join')
  prior8=read((ROOT/release['balanced_baseline']['result']['path']).parent/'TRAIN-greedy'/'IL'/s/'plan.json')
  unweighted8=read((ROOT/release['baseline']['result']['path']).parent/'TRAIN-greedy'/'IL'/s/'plan.json')
  row['comparisons']={'balanced_IL_8':route_comparison(row,reference_route(prior8,balanced['result']['TRAIN_greedy'][s])),
   'unweighted_IL_8':route_comparison(row,reference_route(unweighted8,baseline['result']['TRAIN_greedy']['IL'][s])),
   'SEARCH':route_comparison(row,reference_route(baseline['teacher_'+s]))};outcomes.append(row)
 for name in ['balanced_IL_64','balanced_IL_8','unweighted_IL_8']:
  stops=[r[name]['teacher_CE'] for r in readouts if r['teacher_action']=='STOP'];motion=[r[name]['teacher_CE'] for r in readouts if r['teacher_action']!='STOP']
  wanted={'STOP_count':4,'motion_count':1,'STOP_mean_NLL':sum(stops)/4,'motion_mean_NLL':motion[0],'unweighted_mean_CE':(sum(stops)+sum(motion))/5,'balanced_mean_CE':.5*sum(stops)/4+.5*motion[0]}
  require(all(near(endpoint[name][k],v) for k,v in wanted.items()),'endpoint group loss arithmetic')
 require(result['endpoint_teacher_metrics']==endpoint and len(readouts)==5,'endpoint metrics copied exactly')
 require(len(visits)==268 and len(costs['completed_patient_visits'])==268 and all(x['source_released'] for x in costs['completed_patient_visits']),'268 released complete visits')
 forwards=325+sum(row['steps'] for row in outcomes);require(costs['total_policy_forward_calls']==forwards<=421,'320 loss + 5 logits + complete greedy forwards')
 require(costs['native_budget']['native_preview_entries']<=2347680 and costs['native_budget']['blocked_preview_attempts']==0 and costs['native_budget']['failure'] is None,'executed preview envelope')
 for path,meta in list(seen.items()):require(sha(ROOT/path)==meta['sha256'],'post-read stability')
 loss_values=[x['loss'] for x in curve];grad_values=[x['gradient_norm_before_clip'] for x in curve]
 dynamics_summary={'pre_update_loss_first':loss_values[0],'pre_update_loss_at8':loss_values[7],'pre_update_loss_at64':loss_values[-1],
  'pre_update_loss_min':min(loss_values),'pre_update_loss_max':max(loss_values),'strict_loss_increases':sum(b>a for a,b in zip(loss_values,loss_values[1:])),
  'gradient_norm_min':min(grad_values),'gradient_norm_max':max(grad_values),'gradient_norm_at64':grad_values[-1],
  'gradient_exceeds_clip_count':sum(x['gradient_exceeds_clip_threshold'] for x in dynamics),'clip_threshold':release['learning_protocol']['max_gradient_norm'],
  'changed_parameter_updates':sum(x['before_parameter_hash']!=x['after_parameter_hash'] for x in curve),
  'post_update64_balanced_CE':endpoint['balanced_IL_64']['balanced_mean_CE'],'scope':'Pre-update receipts and post-update64 teacher logits have distinct evaluation times; fixed64 schedule, no endpoint selection.'}
 summary={'status':'PASS_saved_balanced_IL64_metadata_and_scalar_audit','checks':checks,'input_count':len(seen),'source_files':len(index['source_files']),'metadata_files':len(index['metadata_files']),
  'result_sha256':expected_result,'release_sha256':expected_release,'score_arithmetic_tolerances':{'probability_absolute':PROBABILITY_ABSOLUTE_TOLERANCE,'CE_absolute':2e-6,'margin_absolute':1e-6,'ranks_and_actions':'exact'},'first_eight_exact_control':result['first_eight_update_control'],'updates':curve,'training_dynamics_summary':dynamics_summary,
  'endpoint_group_metrics':endpoint,'readouts':readouts,'greedy_outcomes':outcomes,'counts':{'visits':268,'shared_IL_updates':64,'loss_forwards':320,'teacher_logit_forwards':5,'checkpoint_reloads':1,'policy_forwards':forwards,'native_previews':costs['native_budget']['native_preview_entries'],'RL_updates':0,'search_calls':0},
  'parent':{k:receipt[k] for k in ['elapsed_seconds','sampled_peak_rss_bytes','output_bytes','cleanup_errors','final_owned_pids']},
  'scope':'Saved scalar/metadata audit only; no weights, arrays, model or geometry execution. Recorded checkpoint/parameter identities are authenticated metadata, not an independent tensor reload. Training receipts do not expose per-label NLL after update1: subsequent training group contributions cannot be separately reconstructed. Endpoint group losses are independently recomputed from five states and three saved endpoint vectors each (only five new forwards). Complete route comparison uses saved nominal/replayed histories, not new geometry execution. Same five teacher labels and all four TRAIN cases remain; no held-out/clinical claim.',
  'audit_seconds':time.monotonic()-start,'audit_peak_rss_bytes':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss}
 write('audit-result.json',summary);write('input-hashes.json',seen)
 print(json.dumps({k:v for k,v in summary.items() if k not in ['updates','readouts','greedy_outcomes','first_eight_exact_control']},sort_keys=True))

if __name__=='__main__':
 parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--terminal-result-sha256',required=True);parser.add_argument('--terminal-release-sha256',required=True);args=parser.parse_args();main(args.terminal_result_sha256,args.terminal_release_sha256)
