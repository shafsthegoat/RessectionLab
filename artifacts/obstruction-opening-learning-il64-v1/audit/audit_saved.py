"""Bounded saved-JSON IL endpoint audit. No array, checkpoint or model reads."""
import argparse,hashlib,json,math,resource,runpy,signal,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];OUT=Path(__file__).parent/'IL64'
BASE=ROOT/'build/balanced-teacher-il64-audit-v1/audit_saved.py'
BASE_SHA='b12157e2a3d92e959798092d22e17d6c29ebfa2070b726cbd8b5b5b932261fb1'
if hashlib.sha256(BASE.read_bytes()).hexdigest()!=BASE_SHA:raise ValueError('Audit helper changed')
H=runpy.run_path(str(BASE));require=H['require'];read=H['read'];sha=H['sha'];digest=H['digest'];near=H['near'];identity=H['identity'];score=H['score'];reference_route=H['reference_route']
TRAIN=['ReMIND-008','ReMIND-010','ReMIND-020','ReMIND-025']
P=ROOT/'build/obstruction-opening-learning-v1/IL64'
def pinned(ref):
 p=ROOT/ref['path'];require(sha(p)==ref['sha256'],'bound metadata '+ref['path']);return read(p)
def write(name,value):
 with (OUT/name).open('x') as f:json.dump(value,f,indent=2,sort_keys=True,allow_nan=False);f.write('\n')
def cells(history):return sorted({tuple(c) for row in history for c in row.get('removed_indices_native',[])})
def motions(history):
 return sorted([json.dumps({k:r[k] for k in ('tool_id','entry_mm','tip_mm','axis_unit','removed_indices_native','target_removed_mm3','normal_removed_mm3','complete_tool_path_length_mm','reward')},sort_keys=True) for r in history if r['action_id']!='STOP'])
def episode(directory,context,weights,behavior):
 sealed=read(directory/'plan.json');p=sealed['plan'];trace=read(directory/'complete-trace.json');replay=read(directory/'native-replay.json');m=replay['metrics'];a=replay['independent_geometry'];h=m['history'];d=trace['decisions'];actions=p['actions']
 route=reference_route(sealed)
 require(p['history']==trace['metrics']['history'] and identity(p['history'])==identity(h),'complete nominal and native histories')
 require(p['context_hash']==trace['context_hash']==digest(context) and all(p[k]==context[k] for k in ('source_hash','decision_model_hash','max_steps')),'admitted complete plan identity')
 require(p['parameter_hash']==behavior and len(actions)==len(d)==len(h)==m['steps'] and m['terminated'] and len(actions)<=24,'complete behavior-bound trajectory')
 require(actions==[x['action_id'] for x in d]==[x['action_id'] for x in h] and p['initial_observation_hash']==d[0]['observation_hash'],'exact actions/initial inputs')
 require(all(x['behavior_parameter_hash']==behavior and x['action_mask'][x['action_ids'].index(x['action_id'])] for x in d),'frozen behavior and saved legality')
 require(digest({'context':trace['context_hash'],'observations':[x['observation_hash'] for x in d],'history':trace['metrics']['history'],'behavior_parameter_hash':behavior})==trace['trace_seal'],'trace seal')
 require(a['accepted'] is True and a['complete_episode'] is True and a['geometry']['complete_tool_checked'] is True and not a['geometry']['failures'] and a['geometry']['action_count']==sum(x!='STOP' for x in actions),'all full-tool independent replay records accepted')
 require(a['committed_history_hash']==digest(h) and all(a[k]==context[k] for k in ('source_hash','decision_model_hash')),'independent geometry bound to exact history/world')
 removed=set();last=None
 for x,y in zip(h,d):
  require(x['reward']==y['reward'],'collection/native scalar reward')
  if x['action_id']=='STOP':require(x['reward']==0,'STOP zero');continue
  expected=weights['target_per_mm3']*x['target_removed_mm3']-weights['normal_per_mm3']*x['normal_removed_mm3']-weights['action_cost']-weights['motion_per_mm']*x['complete_tool_path_length_mm']-weights['tool_change_cost']*(last is not None and last!=x['tool_id']);last=x['tool_id']
  require(near(expected,x['reward']),'independent reward equation incl full path and tool changes')
  now={tuple(c) for c in x['removed_indices_native']};require(len(now)==len(x['removed_indices_native']) and not now&removed,'no duplicate removed cells');removed|=now
 for key,value in (('total_reward',route['public_return']),('target_removed_mm3',route['target_mm3']),('normal_removed_mm3',route['outside_supplied_goal_mm3'])):require(near(m[key],value) and near(a['outcomes'][key],value),'complete scalar outcome sums '+key)
 region=m['supplied_goal_region'];require(region['fraction_denominator']=='entire_unchanged_supplied_region' and region['target_modified'] is False and near(region['fraction_of_full_region_removed'],route['target_mm3']/region['full_region_membership_mm3']),'unchanged full target denominator')
 route.update(target_fraction=region['fraction_of_full_region_removed'],target_source_cells=a['outcomes']['positive_target_source_cells_removed'],removed_cells=[list(c) for c in sorted(removed)],complete_tool_path_mm=sum(x.get('complete_tool_path_length_mm',0) for x in h))
 return route,sealed,trace,replay
def main(expected_result,expected_release):
 start=time.monotonic();signal.signal(signal.SIGALRM,lambda *_:(_ for _ in ()).throw(TimeoutError('60s saved audit cap')));signal.alarm(60)
 OUT.mkdir(exist_ok=False);R=P/'attempt-01';S=P/'attempt-01.supervision'
 require(sha(R/'result.json')==expected_result and sha(P/'root-release.json')==expected_release,'exact root terminal pins')
 result=read(R/'result.json');release=read(P/'root-release.json');parent=read(S/'receipt.json');worker=read(S/'worker-final.json');control=read(S/'endpoint-control.json')
 require(parent['status']=='complete' and parent['exit_code']==0 and parent['worker_termination_confirmed'] and not parent['cleanup_errors'] and not parent['final_owned_pids'] and parent['stop_reason'] is None,'clean owned completion')
 require(parent['result_sha256']==worker['result_sha256']==worker['canonical_result_sha256']==expected_result and parent['release_sha256']==expected_release and parent['worker_final_sha256']==sha(S/'worker-final.json'),'exact result/parent/worker joins')
 require(parent['endpoint_control_sha256']==worker['endpoint_control_sha256']==sha(S/'endpoint-control.json'),'endpoint control hash')
 index=pinned(release['source_index']);require(parent['source_index']==release['source_index'],'source inventory join')
 for section in ('source_files','metadata_files'):
  for path,h in index[section].items():require(sha(ROOT/path)==h,'source/metadata pin '+path)
 prior={k:pinned(ref) for k,ref in release['inputs'].items()};worlds={a['subject']:a['common_public_world'] for a in prior['result']['arms'] if a['condition']=='obstruction_opening'}
 require(result['status']=='complete_matched_TRAIN_endpoint' and result['method']=='IL' and result['TRAIN']==TRAIN and result['optimizer_updates']=={'IL':64,'RL':0},'fixed IL64 completion')
 require(result['SELECT_EVAL_opened'] is False and result['private_reference_reads']==result['search_calls']==0,'public TRAIN only no new search')
 require(result['initial_parameter_hash']==prior['initial_result']['initial_parameter_hash'],'original fresh initialization preserved')
 require((result['loss_forward_calls'],result['teacher_logit_forwards'],result['checkpoint_loads'],result['completed_source_visits'],result['teacher_trace_reuses'])==(448,7,1,8,256),'actual fixed endpoint counts')
 require(parent['elapsed_seconds']<960 and worker['wall_seconds']<900 and parent['sampled_peak_rss_bytes']<=3*1024**3 and parent['output_bytes']<=128*1024**2,'declared method resource caps')
 config=read(R/'configuration.json');costs=read(R/'costs.json');pins=read(R/'teacher-pins.json');reload=read(R/'checkpoint-reload.json');endpoint=read(R/'endpoint-teacher-metrics.json');cache=read(R/'teacher-cache.json');contexts={s:read(R/(s+'-context.json')) for s in TRAIN}
 protocol=release['learning_protocol'];require(config['learning_protocol']==protocol and digest(protocol)==release['learning_protocol_hash'] and config['execution_limits']==release['execution_limits'] and config['admission_limits']==release['cohort_limits'],'declared configuration joins')
 final=result['checkpoints']['IL']['parameter_hash'];require(parent['checkpoints']==result['checkpoints'] and reload['exact_parameter_match'] is True and reload['parameter_hash']==final and reload['sha256']==result['checkpoints']['IL']['sha256'] and reload['completed_updates']==64,'saved own checkpoint reload exact identity')
 require(control['initial_parameter_hash']==result['initial_parameter_hash'] and control['TRAIN_greedy']==result['TRAIN_greedy'] and control['teacher_metrics']==endpoint==result['endpoint_teacher_metrics'],'parent endpoint joins')
 teachers={};routes=[];readouts=[];curve=[];dyn=[];before=result['initial_parameter_hash']
 for s in TRAIN:
  context=contexts[s];require(context['role']=='TRAIN' and context['subject']==s and context['learning_protocol_hash']==digest(protocol) and context['occupancy_condition']=='cerebrum_plus_supplied_tumor_assumption','exact admitted learning world')
  route,sealed,trace,replay=episode(R/'teachers'/s,context,worlds[s]['reward'],None);old=prior['plan_'+s]['plan'];actual=sealed['plan']
  require({k:v for k,v in actual.items() if k not in ('context_hash','history')}=={k:v for k,v in old.items() if k not in ('context_hash','history')} and identity(actual['history'])==identity(old['history']),'teacher physical data/actions/history identical except context and outcome scope')
  require(pins[s]['context_hash']==digest(context) and pins[s]['trace_seal']==trace['trace_seal'] and pins[s]['plan_seal']==sealed['plan_seal'] and pins[s]['actions']==actual['actions'] and pins[s]['steps']==route['steps'] and pins[s]['stop_steps']==1,'teacher pin join')
  require(pins[s]['readout_binding_hash']==digest([[x['observation_hash'],x['action_id'],x['reward'],x['terminated']] for x in trace['decisions']]),'teacher readout binding')
  teachers[s]=(route,sealed,trace,replay)
 require(sum(p['steps'] for p in pins.values())==7 and sum(p['stop_steps'] for p in pins.values())==4,'fixed7labels4STOP3motion')
 require(cache['complete'] is True and cache['RL_reuse'] is False and cache['learning_protocol_hash']==digest(protocol) and digest(cache['traces'])==cache['cache_seal'],'sealed IL-only cache')
 require(sum(r['array_bytes'] for r in cache['traces'])==cache['array_bytes'] and sum(r['metadata_json_bytes'] for r in cache['traces'])==cache['metadata_json_bytes'] and cache['array_bytes']+cache['metadata_json_bytes']<=64*1024**2,'cache reported payload bounds')
 require(result['teacher_cache']=={**cache,'trace_reuses':256,'readout_rows':7},'unchanged cache at endpoint')
 for item,s in zip(cache['traces'],TRAIN):
  route,sealed,trace,replay=teachers[s]
  require(item['subject']==s and item['trace_seal']==trace['trace_seal'] and item['plan_seal']==sealed['plan_seal'] and item['observations']==[x['observation_hash'] for x in trace['decisions']] and item['independent_replay_hash']==digest(replay['independent_geometry']),'cache authenticated replay/observations')
 for u in range(1,65):
  dest=R/'IL'/f'update-{u:02d}';update=read(dest/'update.json');contributions=[]
  for s in TRAIN:
   c=read(dest/s/'gradient-contribution.json');contributions.append(c);p=pins[s]
   require(c['context_hash']==p['context_hash'] and c['trace_seal']==p['trace_seal'] and c['plan_seal']==p['plan_seal'] and c['steps']==c['loss_forward_calls']==p['steps'] and c['observation_storage']==cache['mode'],'every cached loss contribution is pinned complete teacher')
  require(update['context_hashes']==[c['context_hash'] for c in contributions] and update['trace_seals']==[c['trace_seal'] for c in contributions],'ordered all4 contributions')
  require(update['teacher_group_counts']=={'STOP':4,'motion':3} and update['teacher_group_weights']=={'STOP':.125,'motion':1/6} and update['IL_reduction']=='balanced_STOP_motion_CE_v1','balanced normalization unchanged')
  require(update['before_parameter_hash']==before and update['optimizer_updates']==1 and update['completed_updates']==u and update['loss_forward_calls']==7 and update['method']=='IL','64 shared update identity chain')
  for f in ('loss','actor_loss','value_loss','entropy','loss_forward_calls'):require(near(update[f],sum(c[f] for c in contributions)),'summed contributions '+f)
  require(all(update[f]==0 for f in ('actor_loss','value_loss','entropy')) and update['parameters_changed']==(update['after_parameter_hash']!=before),'IL-only scalar and changed parameter claims')
  require(math.isfinite(update['gradient_norm_before_clip']) and update['gradient_norm_before_clip']>=0 and all(math.isfinite(v) and v>=0 for v in update['module_gradient_norms_before_clip'].values()),'finite gradient diagnostics')
  dyn.append({'completed_updates':u,'pre_update_loss':update['loss'],'loss_change_from_previous':None if not dyn else update['loss']-dyn[-1]['pre_update_loss'],'gradient_norm_before_clip':update['gradient_norm_before_clip'],'module_gradient_norms_before_clip':update['module_gradient_norms_before_clip'],'clip_threshold':protocol['max_gradient_norm'],'gradient_exceeds_clip_threshold':update['gradient_norm_before_clip']>protocol['max_gradient_norm'],'before_parameter_hash':before,'after_parameter_hash':update['after_parameter_hash'],'loss_forward_calls':7,'update_record_sha256':sha(dest/'update.json')})
  before=update['after_parameter_hash'];curve.append({'update':u,'loss':update['loss'],'gradient_norm_before_clip':update['gradient_norm_before_clip'],'before_parameter_hash':update['before_parameter_hash'],'after_parameter_hash':before})
 require(before==final and read(R/'training-dynamics.json')['updates']==dyn and sha(R/'training-dynamics.json')==result['training_dynamics_sha256']==control['training_dynamics_sha256'],'all64 dynamics and endpoint hash chain')
 for s in TRAIN:
  for step,old in enumerate(teachers[s][2]['decisions']):
   item=read(R/'teacher-readout'/s/f'state-{step:02d}.json');ids=item['action_ids'];mask=item['action_mask'];action=old['action_id']
   require(item['subject']==s and item['step']==step and item['teacher_action']==action and item['observation_hash']==old['observation_hash'] and ids==old['action_ids'] and mask==old['action_mask'],'exact7replayed teacher input identities')
   readouts.append({'subject':s,'step':step,'teacher_action':action,**score(item['scores'],ids,mask,ids.index(action))})
  route,sealed,trace,replay=episode(R/'TRAIN-greedy'/'IL'/s,contexts[s],worlds[s]['reward'],final);row=result['TRAIN_greedy'][s];old=teachers[s][1]['plan'];actual=sealed['plan']
  require(row['complete'] is True and row['checkpoint_reloaded'] is True and row['parameter_hash']==final and row['plan_seal']==sealed['plan_seal'] and row['actions']==actual['actions'] and row['steps']==route['steps'] and actual['learning_updates']==64,'own reloaded complete deployment route')
  for key,val in (('public_return',route['public_return']),('target_removed_mm3',route['target_mm3']),('outside_supplied_target_removed_mm3',route['outside_supplied_goal_mm3'])):require(near(row[key],val),'endpoint actual complete route '+key)
  route.update(subject=s,teacher_comparison={'same_action_sequence':actual['actions']==old['actions'],'same_ordered_physical_history':identity(actual['history'])==identity(old['history']),'same_removed_cell_set':cells(actual['history'])==cells(old['history']),'same_unordered_motions':motions(actual['history'])==motions(old['history']),'teacher_removed_cells':[list(c) for c in cells(old['history'])],'return_difference':route['public_return']-teachers[s][0]['public_return'],'target_mm3_difference':route['target_mm3']-teachers[s][0]['target_mm3'],'outside_mm3_difference':route['outside_supplied_goal_mm3']-teachers[s][0]['outside_supplied_goal_mm3']});routes.append(route)
 stops=[r['teacher_CE'] for r in readouts if r['teacher_action']=='STOP'];motion=[r['teacher_CE'] for r in readouts if r['teacher_action']!='STOP']
 wanted={'STOP_count':4,'motion_count':3,'STOP_mean_NLL':sum(stops)/4,'motion_mean_NLL':sum(motion)/3,'unweighted_mean_CE':sum(stops+motion)/7,'balanced_mean_CE':.5*sum(stops)/4+.5*sum(motion)/3,'teacher_action_accuracy':sum(r['greedy_action']==r['teacher_action'] for r in readouts)/7}
 require(all(near(endpoint[k],v,2e-6) for k,v in wanted.items()),'endpoint metrics from saved logits')
 require(len(costs['completed_patient_visits'])==8 and all(v['source_released'] for v in costs['completed_patient_visits']),'8complete released source visits')
 forwards=448+7+sum(r['steps'] for r in routes);require(costs['total_policy_forward_calls']==forwards<=551,'448backward7readout+actualdeployment forwards')
 require(costs['native_budget']['native_preview_entries']<=70080 and costs['native_budget']['failure'] is None and costs['native_budget']['blocked_preview_attempts']==0,'actual preview cap and complete accounting')
 for path,meta in list(H['seen'].items()):require(sha(ROOT/path)==meta['sha256'],'post-read stable saved evidence')
 summary={'status':'PASS_saved_IL64_metadata_scalar_audit','result_sha256':expected_result,'release_sha256':expected_release,'source_files':len(index['source_files']),'metadata_files':len(index['metadata_files']),'checks':require.__globals__['checks'],'input_count':len(H['seen']),'updates':curve,'endpoint_metrics_recomputed':wanted,'teacher_readouts':readouts,'greedy_routes':routes,'counts':{'shared_IL_updates':64,'RL_updates':0,'loss_forwards':448,'teacher_readout_forwards':7,'policy_forwards':forwards,'checkpoint_reloads':1,'cache_reuses':256,'source_visits':8,'native_previews':costs['native_budget']['native_preview_entries']},'parent':{k:parent[k] for k in ('elapsed_seconds','sampled_peak_rss_bytes','output_bytes','cleanup_errors','final_owned_pids')},'audit_seconds':time.monotonic()-start,'audit_peak_rss_bytes':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,'limits':['Saved JSON/history/seal audit; no checkpoint tensors, source arrays, model or native replay executed.','Parameter/update/reload identities are recorded evidence, not independent tensor recomputation. Training per-label NLL is unavailable; group normalization and sum of reported contributions verified.','Teacher top1 agreement and complete route outcome are distinct; alternate routes compared by exact removed cells and recorded motion geometry.','Fixed posthoc TRAIN endpoint under unvalidated S union T material assumption; no heldout/clinical or equal-compute claim.']}
 write('audit-result.json',summary);write('input-hashes.json',H['seen']);print(json.dumps({k:v for k,v in summary.items() if k not in ('updates','teacher_readouts','greedy_routes')},sort_keys=True))
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--terminal-result-sha256',required=True);p.add_argument('--terminal-release-sha256',required=True);a=p.parse_args();main(a.terminal_result_sha256,a.terminal_release_sha256)
