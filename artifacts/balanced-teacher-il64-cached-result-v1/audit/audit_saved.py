"""Prepared saved-only cached64 extension. Execute only after root terminal notice.

Reuses the completed recollected64 scalar/seal audit helpers; no project,
NumPy/Torch, array, checkpoint, native or model imports/execution.
"""
import argparse,hashlib,json,math,resource,runpy,signal,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];OUT=Path(__file__).parent
BASE=ROOT/'build/balanced-teacher-il64-audit-v1/audit_saved.py'
BASE_SHA='b12157e2a3d92e959798092d22e17d6c29ebfa2070b726cbd8b5b5b932261fb1'
if hashlib.sha256(BASE.read_bytes()).hexdigest()!=BASE_SHA:raise ValueError('Saved scalar/seal helper changed')
H=runpy.run_path(str(BASE));require=H['require'];read=H['read'];sha=H['sha'];digest=H['digest'];near=H['near']
visit=H['visit'];score=H['score'];reference_route=H['reference_route'];route_comparison=H['route_comparison'];SUBJECTS=H['SUBJECTS']
P=ROOT/'build/balanced-teacher-il64-cached-v1';STORAGE='cache_complete_replayed_TRAIN_teacher_traces_v1'
def write(name,value):
 with (OUT/name).open('x') as f:json.dump(value,f,sort_keys=True,indent=2,allow_nan=False);f.write('\n')
def pinned(ref):
 require(sha(ROOT/ref['path'])==ref['sha256'],'exact metadata binding '+ref['path']);return read(ROOT/ref['path'])
def plain(row,excluded):return {k:v for k,v in row.items() if k not in excluded}

def main(expected_result,expected_release):
 start=time.monotonic();signal.signal(signal.SIGALRM,lambda *_:(_ for _ in ()).throw(TimeoutError('60s audit cap')));signal.alarm(60)
 for s in [expected_result,expected_release]:require(len(s)==64 and all(c in '0123456789abcdef' for c in s),'root terminal/release SHA required')
 R=P/'attempt-01';S=P/'attempt-01.supervision'
 require(sha(R/'result.json')==expected_result and sha(P/'root-release.json')==expected_release,'terminal root pins')
 result=read(R/'result.json');release=read(P/'root-release.json');receipt=read(S/'receipt.json');worker=read(S/'worker-final.json');control=read(S/'endpoint-control.json')
 require(receipt['status']=='complete' and receipt['exit_code']==0 and receipt['worker_termination_confirmed'] and not receipt['final_owned_pids'] and not receipt['cleanup_errors'] and receipt['stop_reason'] is None,'clean owned terminal')
 require(receipt['result_sha256']==worker['result_sha256']==worker['canonical_result_sha256']==expected_result and receipt['release_sha256']==expected_release,'parent/worker/result joins')
 require(receipt['worker_final_sha256']==sha(S/'worker-final.json') and receipt['endpoint_control_sha256']==worker['endpoint_control_sha256']==sha(S/'endpoint-control.json'),'final control identity')
 require(receipt['elapsed_seconds']<3600 and worker['wall_seconds']<3540 and receipt['sampled_peak_rss_bytes']<3*1024**3 and receipt['output_bytes']<128*1024**2,'resource caps')
 require(receipt['source_index']==release['source_index'],'source index identity');index=pinned(release['source_index'])
 for section in ['source_files','metadata_files']:
  for path,h in index[section].items():require(sha(ROOT/path)==h,'source/metadata guard '+path)
 refs=pinned(release['fixed64_baseline'])['files'];old={k:pinned(ref) for k,ref in refs.items()}
 baseline={k:pinned(ref) for k,ref in release['baseline'].items()}
 require(old['parent']['status']=='complete' and old['parent']['result_sha256']==refs['result']['sha256'] and old['parent']['release_sha256']==refs['release']['sha256'],'completed recollected64 baseline')
 protocol=json.loads(json.dumps(release['learning_protocol']));execution=protocol['cohort_execution']
 require(execution['teacher_observations']==STORAGE and execution.pop('teacher_cache_payload_bytes')==64*1024**2,'explicit bounded cache opt-in')
 execution['teacher_observations']='recollect_complete_pinned_plan_each_IL_update'
 require(protocol==old['release']['learning_protocol'] and release['cohort_limits']==old['release']['cohort_limits'],'only cache storage and executed work change')
 config=read(R/'configuration.json');require(config['learning_protocol']==release['learning_protocol'] and digest(release['learning_protocol'])==release['learning_protocol_hash'] and config['execution_limits']==release['execution_limits'] and config['admission_limits']==release['cohort_limits'],'configuration binding')
 require(result['status']=='complete_cached_balanced_IL64_TRAIN_only' and result['TRAIN']==SUBJECTS and result['optimizer_updates']=={'IL':64,'RL':0} and result['initial_parameter_hash']==old['result']['initial_parameter_hash'],'fixed64 fresh initialization')
 require(result['SELECT_EVAL_opened'] is False and result['private_reference_reads']==0 and result['search_calls']==0,'TRAIN/public only, no search')
 require(result['selection_readiness']['ready'] is True and result['selection_readiness']['execution_admitted'] is False and result['teacher_statuses']=={s:'complete_replayed' for s in SUBJECTS},'complete endpoint without heldout admission')
 require((result['loss_forward_calls'],result['teacher_logit_forwards'],result['checkpoint_loads'],result['completed_source_visits'],result['teacher_trace_reuses'])==(320,5,1,8,256),'exact executed counts')
 final=result['checkpoints']['IL']['parameter_hash'];reload=read(R/'checkpoint-reload.json')
 require(final==old['result']['checkpoints']['IL']['parameter_hash'] and receipt['checkpoints']==result['checkpoints'] and reload['exact_parameter_match'] and reload['parameter_hash']==reload['trained_parameter_hash']==final and reload['completed_updates']==64 and reload['sha256']==result['checkpoints']['IL']['sha256'],'reloaded endpoint exact tensors, file metadata may differ')
 contexts={s:read(R/(s+'-context.json')) for s in SUBJECTS};pins=read(R/'teacher-pins.json');cache=read(R/'teacher-cache.json');costs=read(R/'costs.json');initial=[]
 require(cache['mode']==STORAGE and cache['complete'] is True and cache['RL_reuse'] is False and cache['learning_protocol_hash']==release['learning_protocol_hash'] and digest(cache['traces'])==cache['cache_seal'],'cache metadata seal')
 require([x['subject'] for x in cache['traces']]==SUBJECTS and plain(result['teacher_cache'],{'trace_reuses','readout_rows'})==cache and result['teacher_cache']['trace_reuses']==256 and result['teacher_cache']['readout_rows']==5,'exact final cache record')
 for s,entry in zip(SUBJECTS,cache['traces']):
  row=visit(R,'teachers',s,contexts[s],baseline,final);initial.append(row)
  trace=read(R/'teachers'/s/'complete-trace.json');replay=read(R/'teachers'/s/'native-replay.json')
  require(entry['context_hash']==row['context_hash'] and entry['plan_seal']==row['plan_seal'] and entry['trace_seal']==row['trace_seal'] and entry['steps']==row['steps'] and entry['stop_steps']==row['actions'].count('STOP'),'cache admitted teacher identity')
  require(entry['observations']==[d['observation_hash'] for d in trace['decisions']] and entry['independent_replay_hash']==digest(replay['independent_geometry']),'cache observation/replay seals')
  require(all(entry[k]==pins[s][k] for k in ['context_hash','plan_seal','trace_seal','steps','stop_steps']),'cache/pin identity')
  require(pins[s]['readout_binding_hash']==digest([[d['observation_hash'],d['action_id'],d['reward'],d['terminated']] for d in trace['decisions']]),'detached readout binding')
 require(sum(x['steps'] for x in cache['traces'])==5 and sum(x['stop_steps'] for x in cache['traces'])==4,'same five labels')
 for field in ['array_bytes','metadata_json_bytes']:require(cache[field]==sum(x[field] for x in cache['traces']) and all(type(x[field]) is int and x[field]>=0 for x in cache['traces']),'cache payload accounting')
 require(cache['array_bytes']+cache['metadata_json_bytes']<=64*1024**2,'declared payload cap (not RSS)')
 expected_control=[];dynamics=[];before=result['initial_parameter_hash']
 for u in range(1,65):
  phase=f'IL/update-{u:02d}';row=read(R/phase/'update.json');prior=old[f'update_{u:02d}'];contrib=[]
  require(plain(row,{'context_hashes','trace_seals'})==plain(prior,{'context_hashes','trace_seals'}),'exact all64 scalar/gradient/full tensor receipt '+str(u))
  require(row['before_parameter_hash']==before and row['completed_updates']==u and row['optimizer_updates']==1 and row['loss_forward_calls']==5,'64 shared update chain')
  for s in SUBJECTS:
   c=read(R/phase/s/'gradient-contribution.json');contrib.append(c)
   require(c['context_hash']==digest(contexts[s]) and c['trace_seal']==pins[s]['trace_seal'] and c['plan_seal']==pins[s]['plan_seal'] and c['steps']==c['loss_forward_calls']==pins[s]['steps'] and c['observation_storage']==STORAGE,'same cached complete trace contribution')
  require(row['context_hashes']==[c['context_hash'] for c in contrib] and row['trace_seals']==[c['trace_seal'] for c in contrib],'all four contexts contribute in order')
  for f in ['loss','actor_loss','value_loss','entropy','loss_forward_calls']:require(near(row[f],sum(c[f] for c in contrib)),'shared contribution sum')
  expected_control.append({'update':u,'actual_sha256':sha(R/phase/'update.json'),'baseline_sha256':refs[f'update_{u:02d}']['sha256'],'after_parameter_hash':row['after_parameter_hash'],'loss':row['loss']})
  expected=dict(old['dynamics']['updates'][u-1]);expected['update_record_sha256']=sha(R/phase/'update.json');dynamics.append(expected);before=row['after_parameter_hash']
 require(before==final,'final fixed64 parameter chain')
 def update_control(rows):return {'exact_match':True,'updates':rows,'excluded_metadata_fields':['context_hashes','trace_seals'],'scope':'all other update fields, including gradient and full parameter hashes, equal completed recollected64'}
 require(control['updates']==update_control(expected_control) and result['first_eight_update_control']==read(R/'first-eight-update-control.json')==update_control(expected_control[:8]),'all64/first8 comparison exact')
 actual_dynamics=read(R/'training-dynamics.json');require(actual_dynamics=={**old['dynamics'],'updates':dynamics} and result['training_dynamics_updates']==64 and sha(R/'training-dynamics.json')==result['training_dynamics_sha256']==control['training_dynamics_sha256'],'dynamics parity and hash')
 readouts=[];routes=[]
 for s in SUBJECTS:
  for step,action in enumerate(pins[s]['actions']):
   state=read(R/'teacher-readout'/s/f'state-{step:02d}.json');require(state==old[f'state_{s}_{step}'],'entire final teacher readout identical')
   readouts.append({'subject':s,'step':step,**score(state['balanced_IL_64'],state['action_ids'],state['action_mask'],state['action_ids'].index(action))})
  row=visit(R,'TRAIN-greedy/IL',s,contexts[s],baseline,final);sealed=read(R/'TRAIN-greedy/IL'/s/'plan.json');prior=old['greedy_'+s]
  require(plain(sealed['plan'],{'context_hash'})==plain(prior['plan'],{'context_hash'}),'entire final plan/history parity')
  saved=result['TRAIN_greedy'][s];require(saved['complete'] and saved['checkpoint_reloaded'] and saved['parameter_hash']==final and saved['plan_seal']==row['plan_seal'] and saved['actions']==row['actions'],'final endpoint/replay joins')
  for field,current in [('public_return','public_return'),('target_removed_mm3','target_mm3'),('outside_supplied_target_removed_mm3','outside_supplied_goal_mm3')]:require(near(saved[field],row[current]),'endpoint scalar sum')
  row['recollected64_comparison']=route_comparison(row,reference_route(prior,old['result']['TRAIN_greedy'][s]));routes.append(row)
 require(read(R/'endpoint-teacher-metrics.json')==result['endpoint_teacher_metrics']==old['result']['endpoint_teacher_metrics'],'exact endpoint group losses')
 visits=costs['completed_patient_visits'];require(len(visits)==8 and [x['subject'] for x in visits]==SUBJECTS*2 and all(x['source_released'] for x in visits),'eight released source visits')
 require([x['phase'] for x in visits]==['offline.fixed_teacher_replay.'+s for s in SUBJECTS]+['deployment.reloaded_TRAIN_greedy_replay.'+s for s in SUBJECTS],'no source recollection visits')
 phases=costs['costs'];reuse=[r for name,r in phases.items() if '.cached_loss_backward.' in name]
 require(len(reuse)==256 and sum(r.get('policy_forward_calls',0) for r in reuse)==320 and all(r.get(k,0)==0 for r in reuse for k in ['native_preview_entries','native_transition_calls','inventory_request_calls','task_clone_calls']),'zero cached geometry, same320 forward/backwards')
 readout_phases=[phases['offline.endpoint_teacher_logits.'+s] for s in SUBJECTS];require(sum(r['policy_forward_calls'] for r in readout_phases)==5,'five actual endpoint forwards')
 detached_phases=[phases['offline.endpoint_cached_teacher_readout.'+s] for s in SUBJECTS]
 require(all(r.get(k,0)==0 for r in detached_phases for k in ['native_preview_entries','native_transition_calls','inventory_request_calls','task_clone_calls']),'cached readout performs no new geometry queries')
 forwards=325+sum(x['steps'] for x in routes);require(costs['total_policy_forward_calls']==forwards<=421 and costs['native_budget']['native_preview_entries']<=70080 and costs['native_budget']['blocked_preview_attempts']==0 and costs['native_budget']['failure'] is None,'executed forward/preview bounds')
 require(control['status']=='exact_cached_vs_recollected64' and control['same_endpoint_parameter_hash']==final and control['same_five_readouts'] and control['same_four_complete_native_histories'],'owned parent full parity confirmation')
 oldcost=read((ROOT/refs['result']['path']).parent/'costs.json')
 cost_comparison={'cached_parent_seconds':receipt['elapsed_seconds'],'recollected_parent_seconds':old['parent']['elapsed_seconds'],
  'observed_parent_wall_ratio_recollected_over_cached':old['parent']['elapsed_seconds']/receipt['elapsed_seconds'],
  'cached_peak_rss_bytes':receipt['sampled_peak_rss_bytes'],'recollected_peak_rss_bytes':old['parent']['sampled_peak_rss_bytes'],
  'cached_previews':costs['native_budget']['native_preview_entries'],'recollected_previews':oldcost['native_budget']['native_preview_entries'],
  'cached_source_visits':8,'recollected_source_visits':268,'policy_forwards_each':forwards,
  'scope':'One sequential owned run per condition; whole-parent time is comparable scope but not a repeated or hardware-independent speedup estimate. Inclusive nested scopes are not summed.'}
 require(oldcost['total_policy_forward_calls']==forwards,'equal actual forwards')
 for path,meta in list(H['seen'].items()):require(sha(ROOT/path)==meta['sha256'],'post-read stability')
 require(sha(BASE)==BASE_SHA,'unchanged reused independent audit helper')
 summary={'status':'PASS_saved_cached64_exact_parity_audit','checks':require.__globals__['checks'],'inputs':len(H['seen']),'source_files':len(index['source_files']),'metadata_files':len(index['metadata_files']),
  'result_sha256':expected_result,'release_sha256':expected_release,'all64_update_exact_match':True,'excluded_update_fields':['context_hashes','trace_seals'],'endpoint_parameter_hash':final,
  'readouts':readouts,'complete_routes':routes,'cache':cache,'counts':{'source_visits':8,'trace_reuses':256,'shared_IL_updates':64,'loss_forwards':320,'teacher_readouts':5,'policy_forwards':forwards,'checkpoint_reloads':1,'RL_updates':0,'search_calls':0},
  'cost_comparison':cost_comparison,'audit_seconds':time.monotonic()-start,'audit_peak_rss_bytes':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
  'scope':'Saved metadata/scalar/parity audit only. No checkpoint bytes, observations, patient arrays, geometry/model execution or tests. Cache byte accounting is declared array/JSON payload, not total Python/RSS. Exact TRAIN imitation remains five labels, no new clinical or heldout evidence.'}
 write('audit-result.json',summary);write('input-hashes.json',H['seen']);print(json.dumps({k:v for k,v in summary.items() if k not in ['readouts','complete_routes','cache']},sort_keys=True))

if __name__=='__main__':
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('--terminal-result-sha256',required=True);p.add_argument('--terminal-release-sha256',required=True);a=p.parse_args();main(a.terminal_result_sha256,a.terminal_release_sha256)
