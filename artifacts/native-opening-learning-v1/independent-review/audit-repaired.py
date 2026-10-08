"""Saved bytes/records/tensors only. No simulator or policy imports/forwards."""
from pathlib import Path
from datetime import datetime, timezone
import json,hashlib,math,time,traceback
ROOT=Path.cwd(); B=ROOT/'artifacts/native-opening-learning-v1'; OUT=ROOT/'build/native-opening-learning-output-review-v1'
started=time.perf_counter();checks=0

def sha(path):
 h=hashlib.sha256()
 with Path(path).open('rb') as f:
  for block in iter(lambda:f.read(1<<20),b''):h.update(block)
 return h.hexdigest()
def read(name):return json.loads((B/name).read_text())
def require(condition,message):
 global checks
 checks+=1
 if not condition:raise AssertionError(message)
def close(a,b,label):require(math.isclose(float(a),float(b),rel_tol=1e-10,abs_tol=1e-10),f'{label}: {a!r} != {b!r}')
def canonical(x):return json.dumps(x,sort_keys=True,separators=(',',':'),allow_nan=False)
def cells(rows):
 require(all(isinstance(row,list) and len(row)==3 and all(type(x) is int for x in row) for row in rows),'bad source-cell schema')
 result={tuple(row) for row in rows};require(len(result)==len(rows),'duplicate source cells');return result

def audit():
 index=read('output-sha256.json'); index_sha=sha(B/'output-sha256.json');file_info={}
 require(len(index)==278,'unexpected frozen output inventory size')
 for name,expected in index.items():
  path=(B/name).resolve();require(path.is_relative_to(B.resolve()),'output escaped root')
  require(sha(path)==expected,'output fixity mismatch '+name)
  file_info[name]={'sha256':expected,'bytes':path.stat().st_size}
 declaration=read('declaration-input.json');raw_sha=sha(B/'declaration-input.json')
 require(raw_sha=='963238d34dce36f4a9128c782f0d3ca667640123a58a2eb2498abb01d68c5212','wrong fixed declaration')
 require(len(declaration['source_sha256'])==31,'wrong source closure count')
 require(declaration['source_sha256']['scripts/run_native_opening_learning.py']=='380ec02502840c9ab8acd30bf5c7542728ee622f819c5d12c73e135cea4bf508','wrong runner')
 for path,digest in declaration['source_sha256'].items():
  require(sha(B/'source-snapshot'/path)==digest,'source snapshot changed '+path)
  require(sha(ROOT/path)==digest,'current numerical source changed '+path)
 require(declaration['lineage']['real_patient_count']==0 and declaration['lineage']['human_roles_opened']==[] and declaration['lineage']['transitions']=='simulator_generated','lineage mismatch')
 result=read('result.json');supervisor=read('supervisor.json');prep=read('preparation.json');settings=declaration['settings']
 require(result['status']=='complete' and supervisor['status']=='complete' and supervisor['returncode']==0,'terminal status')
 require(supervisor['declaration_sha256']==raw_sha and supervisor['automatic_retry'] is False and supervisor['timed_out'] is False and supervisor['termination_reason'] is None,'supervision identity/retry')
 require(supervisor['seconds']<=settings['max_wall_seconds'] and result['elapsed_seconds']<=settings['max_wall_seconds'],'wall budget')
 require(supervisor['sampled_peak_rss_bytes']<=settings['max_rss_bytes'],'memory budget')
 require(result['real_patient_count']==0 and result['preparation']==prep,'preparation/lineage')
 require(set(result['methods'])==set(declaration['methods']) and len(result['methods'])==9,'method denominator')
 source_hash=prep['source_hash'];model_hash=prep['decision_model_hash'];reference_hash=prep['reference_hash']
 # Frozen source defines six 1 mm^3 cells and two target cells; independent set arithmetic, no simulator generation.
 support={(4,4,z) for z in range(1,6)}|{(5,5,1)};target={(4,4,4),(4,4,5)}
 teacher=read('teacher.json');teacher_states={r['observation_hash']:r for r in teacher['state_rows']}
 require(len(teacher_states)==teacher['unique_supervised_states']==5,'teacher unique state count')
 require(teacher['complete'] is True and teacher['reference_fields_used'] is False,'teacher scope')
 require(teacher['model_transition_calls']==sum(len(r['action_ids']) for r in teacher_states.values())==20,'teacher call count')
 require(teacher['terminal_sequences']==20-(5-1)==16,'teacher tree terminal count')
 for state in teacher_states.values():
  require(state['action_ids']==[s['action_id'] for s in state['scores']],'teacher score inventory')
  best=max(state['scores'],key=lambda r:r['nominal_return_to_go'])
  require(best['action_id']==state['selected_action_id'],'teacher earliest max selection')
 require(teacher==result['methods']['depth2-search']['planning'],'teacher summary binding')
 close(teacher['nominal_optimum'],1.1,'known exact tiny-task optimum')
 episode_names=list(declaration['methods'])+[f'rl-{u:02}-{e}' for u in range(16) for e in range(4)]+[f'teacher-state-{n:02}' for n in range(5)]
 require(len(episode_names)==78,'episode denominator')
 episode_results={};all_transitions=all_forwards=all_native_previews=0;online=audit_seconds=actor_seconds=native_seconds=0.
 for name in episode_names:
  ep=read(name+'.json');terminal=read(name+'-terminal.json');cost=read(name+'-cost.json');metrics=ep['metrics'];ev=ep['independent_evaluation'];out=ev['outcomes'];hist=metrics['history'];dec=ep['decisions']
  require(ep['status']=='complete' and terminal['status']=='awaiting_independent_check','episode terminal '+name)
  require(terminal['metrics']==metrics and terminal['decisions']==dec,'durable history differs '+name)
  require(ev['accepted'] is True and ev['complete_episode'] is True and ev['geometry']['feasible'] is True and ev['geometry']['failures']==[],'saved independent geometry rejection '+name)
  require(ev['geometry']['complete_tool_checked'] is True and ev['geometry']['frontier_checked'] is True,'incomplete saved audit '+name)
  require(metrics['source_hash']==source_hash and metrics['decision_model_hash']==model_hash and metrics['reference_hash']==reference_hash,'source/task/reference binding '+name)
  require(ev['decision_model_hash']==model_hash and ev['source_hash']==source_hash and ev['reference_hash']==reference_hash,'evaluator binding '+name)
  require(metrics['observation_track']=='synthetic_scan' and metrics['planning_estimator_only'] is False and metrics['terminated'] is True,'episode lineage '+name)
  require(len(hist)==len(dec)==metrics['steps']==ep['committed_transitions']==ep['attempted_actions'] and 1<=len(hist)<=2,'episode action counts '+name)
  require(ep['invalid_actions']==0 and dec[-1]['terminated'] is True,'invalid or incomplete episode '+name)
  removed=set();contact=set();normal=target_volume=reward=path_length=0.;tool=None;changes=nonstop=0
  for k,(h,d) in enumerate(zip(hist,dec)):
   require(d['status']=='returned' and d['committed_info']==h and d['action_id']==h['action_id'],'decision history '+name)
   require(d['step']==k and d['behavior_parameter_hash']==ep['behavior_parameter_hash'],'behavior binding '+name)
   observed=teacher_states[d['observation_hash']]
   require(d['action_ids']==observed['action_ids'] and all(d['action_mask']) and len(d['action_mask'])==len(d['action_ids']),'candidate evidence mismatch '+name)
   require(d['action_ids'][d['selected_index']]==d['action_id'],'selected action mismatch '+name)
   require(d['terminated']==(d['action_id']=='STOP' or k==1),'terminal flag '+name)
   if 'logits' in d:
    values=d['logits'];top=max(values);weights=[math.exp(x-top) for x in values];den=sum(weights)
    for expected,actual in zip([x/den for x in weights],d['probabilities']):require(abs(expected-actual)<2e-7,'saved softmax mismatch '+name)
    if ep['mode']=='argmax':require(d['selected_index']==values.index(top),'argmax mismatch '+name)
   if h['action_id']=='STOP':
    require(k==len(hist)-1,'actions after STOP '+name);close(h['reward'],0.,'STOP reward');close(h['complete_tool_path_length_mm'],0.,'STOP path')
   else:
    require(h['source_hash']==source_hash and h['source_shape']==[9,9,7],'native source binding '+name)
    require(h['native_affine']==[[1.,0.,0.,0.],[0.,1.,0.,0.],[0.,0.,1.,0.],[0.,0.,0.,1.]],'source physical scale '+name)
    removed_now=cells(h['removed_indices_native']);contacts_now=cells(h['contact_indices_native'])
    require(removed_now<=support and contacts_now<=support and not removed_now&removed,'out-of-support or duplicate reward '+name)
    micro_removed=set();micro_contact=set()
    for micro in h['microsteps']:
     mr=cells(micro['removed_indices_native']);mc=cells(micro['contact_indices_native'])
     require(not mr&micro_removed,'duplicate micro removal '+name);micro_removed|=mr;micro_contact|=mc
    require(micro_removed==removed_now and micro_contact==contacts_now,'microstep cell union '+name)
    tv=len(removed_now&target);nv=len(removed_now-target);length=2*math.dist(h['entry_mm'],h['tip_mm']);changed=tool is not None and tool!=h['tool_id']
    value=tv-.2*nv-.03-.001*length-.03*changed
    close(h['target_removed_mm3'],tv,'per-action target');close(h['normal_removed_mm3'],nv,'per-action normal');close(h['reward'],value,'per-action reward');close(h['complete_tool_path_length_mm'],length,'per-action path')
    require(h['outcome_scope']=='separate_evaluator_reference' and h['partial_contact_weight']==0.,'reward lineage/contact '+name)
    removed|=removed_now;contact|=contacts_now;target_volume+=tv;normal+=nv;reward+=value;path_length+=length;nonstop+=1;changes+=changed;tool=h['tool_id']
   close(d['reward'],h['reward'],'decision reward')
   actor_seconds+=d['decision_seconds'];native_seconds+=d['transition_seconds']
  sums={'target_removed_mm3':target_volume,'normal_removed_mm3':normal,'simulated_removed_volume_mm3':len(removed),'total_reward':reward,'cumulative_contacted_tissue_upper_bound_mm3':len(contact),'currently_retained_contacted_tissue_upper_bound_mm3':len(contact-removed)}
  for key,value in sums.items():close(metrics[key],value,name+' metrics '+key);close(out[key],value,name+' outcomes '+key)
  for key,value in {'complete_tool_path_length_mm':path_length,'nonstop_actions':nonstop,'tool_changes':changes,'positive_target_source_cells_removed':target_volume,'reference_target_fraction_removed':target_volume/2.,'total_reference_target_mm3':2.}.items():close(out[key],value,name+' '+key)
  for key in ['clinical_deficit_probability','motor_surrogate','language_surrogate']:require(out[key] is None,name+' unsupported clinical claim')
  close(ep['simulated_return'],reward,name+' episode return')
  budget=cost['online'];require(budget['failure'] is None and budget['counting_reliable'] is True and budget['history_complete_caller_attestation'] is True,'bad online budget '+name)
  require(budget['elapsed_seconds']<=10. and budget['native_preview_entries']<=2048 and budget['blocked_preview_attempts']==0,'per episode bounds '+name)
  phases=budget['native_preview_profile']['phases'];require(sum(x['started'] for x in phases.values())==budget['native_preview_entries'],'native preview accounting '+name)
  require(all(x['started']==x['returned'] and x['raised']==0 and x['feasible']+x['rejected']==x['returned'] for x in phases.values()),'preview phase accounting '+name)
  close(cost['independent_audit_seconds'],ep['independent_evaluation_seconds'],'audit cost '+name)
  require(budget['elapsed_seconds']>=ep['online_seconds'],'planning excluded from online cost '+name)
  if name in result['methods']:
   row=result['methods'][name];require(row['status']=='complete' and row['accepted'] is True and row['outcomes']==out,'method summary '+name)
   require(row['actions']==[d['action_id'] for d in dec] and row['parameter_hash']==ep['behavior_parameter_hash'],'method action/checkpoint '+name)
   close(row['online_seconds'],budget['elapsed_seconds'],'inclusive method seconds '+name);close(row['replay_seconds'],ep['online_seconds'],'method replay seconds '+name)
  episode_results[name]={'return':reward,'transitions':len(dec),'target_mm3':target_volume,'normal_mm3':normal,'behavior_parameter_hash':ep['behavior_parameter_hash'],'online_seconds':budget['elapsed_seconds'],'audit_seconds':ep['independent_evaluation_seconds'],'native_previews':budget['native_preview_entries'],'forwards':ep['policy_forward_calls']}
  all_transitions+=len(dec);all_forwards+=ep['policy_forward_calls'];all_native_previews+=budget['native_preview_entries'];online+=budget['elapsed_seconds'];audit_seconds+=ep['independent_evaluation_seconds']
 # Verify each supervised state was replayed from a real saved complete simulated trajectory.
 proofs=teacher['supervised_state_demonstrations'];teacher_checks=read('teacher-demonstration-checks.json')
 require(len(proofs)==len(teacher_checks)==5,'teacher proof denominator')
 for i,(proof,record) in enumerate(zip(proofs,teacher_checks)):
  ep=read(f'teacher-state-{i:02}.json');d=ep['decisions'][len(proof['prefix'])]
  require([x['action_id'] for x in ep['decisions']]==proof['complete_demonstration'],'teacher complete demonstration')
  require(d['observation_hash']==proof['observation_hash'] and d['action_id']==proof['selected_action_id'],'teacher label evidence')
  require(all(record[k]==v for k,v in proof.items()) and record['accepted'] is True and record['transitions']==len(ep['decisions']),'teacher replay receipt')
 require(result['offline_teacher_validation']['complete_episodes']==5 and result['offline_teacher_validation']['transitions']==10,'teacher validation totals')
 # Parameter tensors are loaded safely solely for byte hashing; no policy/model constructed or executed.
 import torch
 import numpy as np
 torch.set_num_threads(1)
 def load_checkpoint(path):
  extra=torch.serialization.get_unsafe_globals_in_checkpoint(path)
  require(set(extra)=={"numpy._core.multiarray.scalar","numpy.dtype"},"unexpected checkpoint global metadata")
  with torch.serialization.safe_globals([np._core.multiarray.scalar,np.dtype,np.dtypes.Float64DType]):
   return torch.load(path,map_location="cpu",weights_only=True)
 checkpoints={}
 def state_hash(state):
  h=hashlib.sha256()
  for name,tensor in sorted(state.items()):
   array=tensor.detach().cpu().contiguous().numpy();h.update(name.encode());h.update(str(array.dtype).encode());h.update(str(array.shape).encode());h.update(array.tobytes())
  return 'sha256:'+h.hexdigest()
 initial= load_checkpoint(B/'initial.pt')
 ih=state_hash(initial['policy']);require(ih==initial['parameter_hash']==result['initial_parameter_hash'],'initial tensor hash')
 require(initial['context']['declaration_sha256']==raw_sha and initial['context']['source_ids']==(source_hash,) and initial['context']['decision_model_hash']==model_hash,'checkpoint context')
 require(initial['context']['real_patient_count']==0 and initial['context']['transition_lineage']=='simulator_generated','checkpoint lineage')
 require(canonical(initial['architecture']['config'])==canonical(declaration['policy']),'checkpoint architecture')
 checkpoints['initial']={'tensor_sha256':ih,'file_sha256':sha(B/'initial.pt')}
 for arm in ['BC','scratch-RL']:
  updates=read(arm+'-updates.json');train=result['training'][arm];require(len(updates)==train['updates']==16 and train['status']=='complete','update count '+arm)
  current=ih;forwards=transitions=0
  for i,row in enumerate(updates):
   grad=row['gradient'];require(row['update']==i+1 and grad['optimizer_steps']==1,'gradient attempt index '+arm)
   require(grad['initial_parameter_hash']==current and grad['parameters_changed'] is True,'update tensor chain '+arm)
   current=grad['updated_parameter_hash'];require(current!=grad['initial_parameter_hash'],'unchanged update '+arm)
   require(grad['gradient_norm_after_clip']<=5.00001 and grad['actor_gradient_norm_before_clip']>0 and grad['encoder_gradient_norm_before_clip']>0,'gradient diagnostics '+arm)
   forwards+=row['loss']['loss_forward_calls']
   if arm=='BC':
    require(row['episodes']==[] and row['loss']['supervised_actions']==5 and row['loss']['loss_forward_calls']==5,'BC label exposure')
    close(grad['critic_gradient_norm_before_clip'],0.,'BC critic supervision')
   else:
    require(len(row['episodes'])==row['loss']['completed_episodes']==4,'complete RL batch')
    for e,saved in enumerate(row['episodes']):
     name=f'rl-{i:02}-{e}';ep=read(name+'.json');checked=episode_results[name]
     require(checked['behavior_parameter_hash']==grad['initial_parameter_hash'],'on-policy weight identity '+name)
     require(saved['decisions']==checked['transitions'] and saved['outcomes']==ep['independent_evaluation']['outcomes'],'RL batch record '+name)
     close(saved['return'],checked['return'],'RL return '+name);transitions+=checked['transitions']
  require(train['initial_parameter_hash']==ih and train['latest_parameter_hash']==current,'paired training endpoints '+arm)
  require(forwards==train['loss_forward_calls'] and transitions==train['optimization_transitions'],'training accounting '+arm)
  ck=load_checkpoint(B/(arm+'-latest.pt'));computed=state_hash(ck['policy'])
  require(computed==current==ck['parameter_hash']==result['methods'][arm]['parameter_hash'],'latest tensor hash '+arm)
  require(ck['initial_parameter_hash']==ih and ck['updates']==16 and ck['context']==initial['context'],'latest context/init '+arm)
  require(canonical(ck['architecture'])==canonical(initial['architecture']),'architecture drift '+arm)
  steps=[int(v['step'].item()) for v in ck['optimizer']['state'].values()];require(steps and set(steps)=={16},'optimizer state actual steps '+arm)
  changed={group:any(not torch.equal(t,ck['policy'][name]) for name,t in initial['policy'].items() if name.startswith(group+'.')) for group in ['encoder','actor','stop','critic']}
  require(changed['encoder'] and changed['actor'] and (not changed['critic'] if arm=='BC' else changed['critic']),'module tensor changes '+arm)
  checkpoints[arm]={'tensor_sha256':computed,'file_sha256':sha(B/(arm+'-latest.pt')),'actual_optimizer_steps':16,'changed_modules':changed}
  del ck
 require(sum(v['transitions'] for n,v in episode_results.items() if n.startswith('rl-'))==112,'RL transitions')
 require(all_transitions==137,'all recorded actual episode transitions')
 require(len([n for n in episode_results if n.startswith('rl-')])==64,'RL episode count')
 for n in ['STOP','initial','random-0','random-1','random-2','greedy','depth2-search']:
  require(episode_results[n]['behavior_parameter_hash']==ih,'frozen untrained baseline '+n)
 require(episode_results['BC']['return']==0. and read('BC.json')['decisions'][0]['action_id']=='STOP','BC negative meaning changed')
 close(episode_results['initial']['return'],-.464,'initial result');close(episode_results['scratch-RL']['return'],-.896,'RL negative result')
 require(episode_results['scratch-RL']['return']<episode_results['initial']['return'],'RL deterioration missing')
 # Recheck every execution-time indexed byte after audit; later owner report files are outside this frozen inventory.
 for name,expected in index.items():require(sha(B/name)==expected,'output changed during audit '+name)
 require(sha(B/'output-sha256.json')==index_sha,'output inventory changed')
 extra=sorted(str(p.relative_to(B)) for p in B.rglob('*') if p.is_file() and str(p.relative_to(B)) not in index and p.name!='output-sha256.json')
 return {'schema':'native-opening-saved-output-audit-v1','created_at':datetime.now(timezone.utc).isoformat(),'status':'passed','checks':checks,'declaration_sha256':raw_sha,'source_files_verified':31,'frozen_output_files_verified':len(index),'frozen_output_bytes':sum(x['bytes'] for x in file_info.values()),'output_inventory_sha256':index_sha,'result_sha256':sha(B/'result.json'),'output_files_unchanged_before_after':True,'current_numerical_sources_match':True,'methods':{name:episode_results[name] for name in declaration['methods']},'training':result['training'],'checkpoints':checkpoints,'teacher':{'states':5,'tree_transitions':20,'terminal_sequences':16,'validated_demonstrations':5,'validation_transitions':10,'offline_validation_seconds':result['offline_teacher_validation']['seconds']},'totals':{'accepted_comparison_arms':9,'accepted_saved_episodes':78,'actual_episode_transitions':all_transitions,'RL_episodes':64,'RL_transitions':112,'BC_updates':16,'RL_updates':16,'policy_rollout_forwards':all_forwards,'loss_forwards':192,'actual_episode_native_previews':all_native_previews,'shared_preparation_native_previews':12,'summed_guarded_online_seconds':online,'summed_independent_audit_seconds':audit_seconds,'summed_actor_decision_seconds':actor_seconds,'summed_native_transition_seconds':native_seconds},'bounds':{'worker_seconds':result['elapsed_seconds'],'supervisor_seconds':supervisor['seconds'],'wall_seconds_limit':300,'sampled_peak_RSS_bytes':supervisor['sampled_peak_rss_bytes'],'RSS_limit_bytes':settings['max_rss_bytes'],'per_episode_seconds_limit':10,'per_episode_native_previews_limit':2048,'automatic_retry':False},'interpretation':'Completed negative learning pilot. BC changes to STOP and improves scalar return from -0.464 to 0 without target removal; scratch RL worsens to -0.896. Neither learns the opening-to-target behavior. Exact nominal search reaches1.1; one random episode reaches1.098. One generated engineering fixture, not patient generalization, physical validation or clinical evidence.','scope':'Saved JSON/source snapshots/checkpoint tensors only. Tensor states loaded with torch weights_only=True, narrowly allowlisting NumPy float64 scalar/dtype metadata found by static pickle inspection, solely for hashing and optimizer-step metadata. No policy/model construction, forwards, optimizer updates, geometry replay, image reads, network or heldout access. Saved geometry acceptance/bindings inspected; clearances not recomputed.','unindexed_later_reporting_files':extra,'audit_elapsed_seconds':time.perf_counter()-started,'file_inventory':file_info}
try:
 report=audit();raw=(json.dumps(report,indent=2,allow_nan=False)+'\n').encode();p=OUT/'verification.json'
 with p.open('xb') as f:f.write(raw)
 print(json.dumps({k:report[k] for k in ['status','checks','source_files_verified','frozen_output_files_verified','methods','training','checkpoints','totals','bounds','audit_elapsed_seconds']},indent=2))
 print(json.dumps({'verification_path':str(p),'verification_sha256':hashlib.sha256(raw).hexdigest()},indent=2))
except BaseException as e:
 p=OUT/'failure-repaired.json'
 with p.open('x') as f:json.dump({'checks':checks,'type':type(e).__name__,'message':str(e),'traceback':traceback.format_exc(),'elapsed_seconds':time.perf_counter()-started},f,indent=2)
 raise
