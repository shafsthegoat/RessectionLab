"""Saved-only eight-arm opening-proposal audit; run only after root terminal notice.

Reuses the prior bounded stdlib scalar/seal helper. No project/scientific
imports, checkpoint/array reads, geometry execution or model calls.
"""
import argparse, hashlib, json, math, resource, runpy, signal, time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];OUT=Path(__file__).parent
BASE=ROOT/'build/balanced-teacher-il64-audit-v1/audit_saved.py'
BASE_SHA='b12157e2a3d92e959798092d22e17d6c29ebfa2070b726cbd8b5b5b932261fb1'
if hashlib.sha256(BASE.read_bytes()).hexdigest()!=BASE_SHA:raise ValueError('Bounded audit helper changed')
H=runpy.run_path(str(BASE));require=H['require'];read=H['read'];sha=H['sha'];digest=H['digest'];near=H['near']
reference_route=H['reference_route'];route_comparison=H['route_comparison']
SUBJECTS=['ReMIND-008','ReMIND-010','ReMIND-020','ReMIND-025']
CONDITIONS=['original_13_axes','obstruction_opening']
P=ROOT/'build/paired-obstruction-opening-search-v1'
def without_ids(rows):return [{k:v for k,v in r.items() if k not in ('action_id','proposal_id')} for r in rows]
def check_inventory(inv):
 counts={}
 for row in inv['ledger']:counts[row['proposal_reason']]=counts.get(row['proposal_reason'],0)+1
 cap=counts.get('CANDIDATE_CAP',0);selection=counts.get('OBSTRUCTION_SELECTION_CAP',0)
 evidence=inv.get('obstruction_accounting') or {};truncated=bool(evidence.get('evidence_truncated',False))
 emitted=[r for r in inv['ledger'] if r['proposal_reason']=='PROPOSED_UNCERTIFIED']
 legal=[r for r in inv['ledger'] if r['feasible']]
 require(inv['candidate_cap']==120 and inv['max_steps']==24 and len(emitted)<=120,'fixed catalog caps')
 require(inv['complete']==(cap+selection==0 and not truncated) and inv['ledger_complete'] is True,'catalog vs discovery completion')
 require(inv['disposition_counts']==counts and inv['emitted']==emitted and inv['emitted_count']==len(emitted) and inv['accepted_count']==len(legal) and inv['rejected_count']==len(emitted)-len(legal),'catalog count semantics')
 require(inv['omitted_count']==cap+selection,'selection and candidate omissions preserved')
 if 'obstruction_accounting' in inv:
  require(inv['candidate_cap_omitted_count']==cap and inv['obstruction_selection_omitted_count']==selection and inv['discovery_complete'] is False,'explicit bounded discovery')
 return {'declared_slots':inv['declared_slots'],'emitted':len(emitted),'legal':len(legal),'candidate_cap_omitted':cap,'selection_omitted':selection,'evidence_truncated':truncated,'complete':inv['complete'],'families':sorted({r['family'] for r in emitted})}
def write(name,value):
 with (OUT/name).open('x') as f:json.dump(value,f,sort_keys=True,indent=2,allow_nan=False);f.write('\n')
def pinned(ref):
 require(sha(ROOT/ref['path'])==ref['sha256'],'bound small metadata '+ref['path']);return read(ROOT/ref['path'])
def main(expected_result,expected_release):
 start=time.monotonic();signal.signal(signal.SIGALRM,lambda *_:(_ for _ in ()).throw(TimeoutError('60s saved audit cap')));signal.alarm(60)
 for value in (expected_result,expected_release):require(len(value)==64 and all(c in '0123456789abcdef' for c in value),'root terminal pin')
 R=P/'attempt-01';S=P/'attempt-01.supervision'
 require(sha(R/'result.json')==expected_result and sha(P/'root-release.json')==expected_release,'exact terminal/release bytes')
 result=read(R/'result.json');release=read(P/'root-release.json');receipt=read(S/'receipt.json');worker=read(S/'worker-final.json');costs=read(R/'costs.json')
 require(receipt['worker_termination_confirmed'] and not receipt['final_owned_pids'] and not receipt['cleanup_errors'],'owned child reaped and clean')
 require(receipt['result_sha256']==worker['canonical_result_sha256']==expected_result and receipt['release_sha256']==expected_release,'parent/worker result/release joins')
 require(receipt['worker_final_sha256']==sha(S/'worker-final.json'),'worker final identity')
 require(receipt['source_index']==release['source_index'],'source inventory join');index=pinned(release['source_index'])
 for section in ('source_files','metadata_files'):
  for path,h in index[section].items():require(sha(ROOT/path)==h,'source/metadata binding '+path)
 refs=pinned(release['input_index'])['files'];baseline={k:pinned(v) for k,v in refs.items()}
 require(release['TRAIN']==SUBJECTS and release['planned_arms']==[{'subject':s,'condition':c} for s in SUBJECTS for c in CONDITIONS],'fixed four TRAIN/eight arms')
 require(release['status']=='released_one_attempt' and release['attempts']==1 and release['automatic_retry'] is False,'sole root release')
 require(release['execution_limits']=={'arms':8,'max_steps':24,'candidate_cap':120,'search_seconds_per_arm':45,'max_native_previews':100000,'max_search_commits':192,'max_execution_replay_commits':384,'max_native_transition_calls':576,'policy_forwards':0,'optimizer_updates':0,'checkpoint_loads':0,'threads':1,'search_method':'observed_greedy_search'},'declared execution caps')
 require(release['SELECT_EVAL_execution'] is False and result['SELECT_EVAL_opened'] is False,'no SELECT/EVAL execution')
 require(all(type(result[k]) is int and result[k]==0 for k in ('policy_forwards','optimizer_updates','checkpoint_loads','private_reference_reads')),'actual zero learning/model/private counters')
 require(costs['prohibited_calls']=={'policy_forwards':0,'optimizer_updates':0,'checkpoint_loads':0} and result['native_counts']==costs['native_counts'],'actual counters agree')
 require(set(result['native_counts'])=={'search','rollout','replay'} and all(type(v) is int and v>=0 for v in result['native_counts'].values()),'nonnegative transition accounting')
 require(result['native_counts']['search']<=192 and result['native_counts']['rollout']+result['native_counts']['replay']<=384 and sum(result['native_counts'].values())<=576,'transition caps')
 require(costs['native_budget']['native_preview_entries']<=100000,'preview cap')
 arms=result['arms'];require([{k:r[k] for k in ('subject','condition')} for r in arms]==release['planned_arms'],'all eight denominator/order retained')
 complete=result['status']=='complete_paired_TRAIN_search'
 if complete:
  require(receipt['status']=='complete' and receipt['exit_code']==0 and receipt['stop_reason'] is None and worker['status']=='complete_owned_paired_TRAIN_obstruction','complete owned acceptance')
  require(receipt['elapsed_seconds']<660 and worker['wall_seconds']<600 and receipt['sampled_peak_rss_bytes']<=3*1024**3 and receipt['output_bytes']<=64*1024**2,'completed within declared caps')
  require(result['global_resource_failure'] is None and all(r['status']=='complete' and r['source_released'] is True for r in arms),'all arms complete/released')
  control=read(S/'endpoint-control.json');require(receipt['endpoint_control_sha256']==worker['endpoint_control_sha256']==sha(S/'endpoint-control.json'),'endpoint control identity')
 else:require(receipt['status']!='complete','failed result not reported as whole-attempt success');control=None
 summaries=[];pairs={};transitions={'search':0,'rollout':0,'replay':0};previews=0
 for ordinal,row in enumerate(arms):
  directory=R/f'arm-{ordinal:02d}';s=row['subject'];condition=row['condition']
  if row['status']!='complete':
   require(row.get('failure') or row.get('not_run_reason'),'explicit unresolved denominator reason');summaries.append(row);continue
  require(read(directory/'terminal.json')==row,'final arm/source release record')
  world=read(directory/'world.json');context=read(directory/'context.json');inventory=read(directory/'initial-inventory.json')
  for k in ('source_hash','decision_model_hash','initial_observation_hash','context_hash','common_public_world','normalization','derived_occupancy','supplied_goal_extent','proposal_config','proposal_rule_hash'):require(row[k]==world[k],'initial world binding '+k)
  require(digest(context)==row['context_hash'],'context seal');common=row['common_public_world'];extent=row['supplied_goal_extent'];derivation=row['derived_occupancy']
  require(0<=common['T_minus_S_voxels']<=common['full_target_voxels'] and common['S_intersect_T_voxels']==common['full_target_voxels']-common['T_minus_S_voxels'],'occupancy count partition')
  require(near(common['raw_occupancy_target_fraction_upper_bound'],common['S_intersect_T_voxels']/common['full_target_voxels']),'raw full-target upper bound')
  require(extent['full_region_positive_voxels']==common['full_target_voxels'] and near(extent['full_region_membership_mm3'],common['full_target_mm3']),'full target denominator')
  require(inventory['source_hash']==row['source_hash'] and inventory['decision_model_hash']==row['decision_model_hash'] and inventory['nominal_target_hash']==common['target_hash'] and inventory['candidate_cap']==120 and inventory['max_steps']==24,'initial proposal world/limits')
  check_inventory(inventory)
  sealed=read(directory/'plan.json');plan=sealed['plan'];replay=read(directory/'replay.json');m=replay['metrics'];a=replay['independent_geometry'];history=m['history'];actions=plan['actions'];search=read(directory/'search.json');account=search['accounting']
  route=reference_route(sealed)
  require(sealed['plan_seal']==row['plan_seal'] and plan['history']==history and plan['context_hash']==row['context_hash'] and plan['source_hash']==row['source_hash'] and plan['decision_model_hash']==row['decision_model_hash'] and plan['initial_observation_hash']==row['initial_observation_hash'],'own complete plan/replay/identity seal')
  require(plan['parameter_hash'] is None and plan['architecture_hash'] is None and plan['learning_updates']==0,'no fitted policy attribution')
  require(actions==search['actions'] and search['status']=='complete' and {k:v for k,v in account.items() if k!='decisions'}==row['search_accounting'] and account['complete'] is True and account['global_optimality_proven'] is False,'complete greedy selection only')
  require(m['terminated'] is True and m['steps']==len(actions)<=24 and a['accepted'] is True and a['complete_episode'] is True and a['geometry']['complete_tool_checked'] is True and not a['geometry']['failures'],'saved full tool complete replay acceptance')
  require(a['committed_history_hash']==digest(history) and a['source_hash']==row['source_hash'] and a['decision_model_hash']==row['decision_model_hash'] and a['outcomes']==row['outcomes'],'independent geometry/result joins')
  require(a['geometry']['action_count']==sum(x!='STOP' for x in actions) and row['stop_only']==(actions==['STOP']),'all actions/STOP accounting')
  require(len(account['decisions'])==account['model_transition_calls']==len(actions) and account['method']=='observed_greedy' and account['time_budget_seconds']==45 and account['planning_seconds']<=45,'complete greedy steps/time')
  saved_inventories=read(directory/'search-inventories.json');by_state={};state_summaries=[]
  for item in saved_inventories:
   path=directory/item['file'];require(path.parent==directory and path.suffix=='.json','local captured inventory path');require(sha(path)==item['sha256'],'captured inventory identity')
   inv=read(path);state=inv['cavity_state_hash'];require(state not in by_state,'one capture per native state');by_state[state]=inv
   require(inv['source_hash']==row['source_hash'] and inv['decision_model_hash']==row['decision_model_hash'] and inv['nominal_target_hash']==common['target_hash'],'captured own public world')
   state_summaries.append({'state_hash':state,**check_inventory(inv)})
  decision_summaries=read(directory/'decision-summary.json');require(len(decision_summaries)==len(account['decisions']),'all decision summaries retained')
  score_count=0;prior_tool=None;removed=set()
  for step,(decision,h) in enumerate(zip(account['decisions'],history)):
   scores=decision['scores'];ids=[x['action_id'] for x in scores];score_count+=len(scores)-1
   inv=by_state[decision['source_state_hash']];legal={r['action_id']:r for r in inv['ledger'] if r['feasible']}
   require(set(legal)==set(ids)-{'STOP'},'all and only captured legal actions scored')
   require(ids[0]=='STOP' and len(set(ids))==len(ids) and scores[0]['reward']==0 and decision['step']==step,'unique STOP-first score table')
   require(all(math.isfinite(x['reward']) for x in scores) and decision['all_current_legal_actions_scored'] is True and decision['legal_nonstop_actions']==decision['scored_nonstop_actions']==len(scores)-1,'all currently emitted legal actions scored')
   chosen=max(scores,key=lambda x:x['reward']);require(chosen['action_id']==actions[step]==decision['selected_action_id']==h['action_id'],'exact maximum reward/stable first tie')
   ds=decision_summaries[step];added=[x for x in scores if x['action_id']!='STOP' and legal[x['action_id']]['family']=='obstruction_opening']
   require(ds['step']==step and ds['state_hash']==decision['source_state_hash'] and ds['selected_action_id']==chosen['action_id'] and ds['selected_score']==chosen,'decision summary selected result')
   require(ds['selected_family']==('STOP' if chosen['action_id']=='STOP' else legal[chosen['action_id']]['family']) and ds['added_scores']==added,'exact selected and added families')
   require(ds['added_legal_count']==len(added) and ds['added_target_removing_count']==sum(x['target_removed_mm3']>0 for x in added) and ds['added_nonT_removing_count']==sum(x['normal_removed_mm3']>0 for x in added) and ds['added_positive_reward_count']==sum(x['reward']>0 for x in added),'added score totals')
   require(ds['omitted_count']==inv['omitted_count'] and ds['obstruction_accounting']==inv.get('obstruction_accounting'),'decision retains cap and discovery evidence')
   for key in ('reward','target_removed_mm3','normal_removed_mm3','insertion_distance_mm','complete_tool_path_length_mm'):require(near(chosen[key],h[key]),'selected scalar matches committed replay '+key)
   if h['action_id']!='STOP':
    weights=common['reward'];expected=weights['target_per_mm3']*h['target_removed_mm3']-weights['normal_per_mm3']*h['normal_removed_mm3']-weights['action_cost']-weights['motion_per_mm']*h['complete_tool_path_length_mm']-weights['tool_change_cost']*(prior_tool is not None and prior_tool!=h['tool_id']);prior_tool=h['tool_id']
    require(near(h['reward'],expected),'independent frozen scalar reward equation')
    cells={tuple(v) for v in h['removed_indices_native']};require(len(cells)==len(h['removed_indices_native']) and not cells&removed,'unique newly removed cells');removed|=cells
  require(account['evaluated_nonstop_actions']==score_count and near(account['estimated_incremental_return'],route['public_return']),'greedy score/return totals')
  for field,value in (('target_removed_mm3',route['target_mm3']),('normal_removed_mm3',route['outside_supplied_goal_mm3']),('total_reward',route['public_return'])):require(near(m[field],value) and near(a['outcomes'][field],value),'full saved scalar sum '+field)
  region=m['supplied_goal_region'];require(region['fraction_denominator']=='entire_unchanged_supplied_region' and region['target_modified'] is False and near(region['fraction_of_full_region_removed'],route['target_mm3']/common['full_target_mm3']),'unchanged complete target fraction')
  require(row['target_access_success']==a['target_access_success']==(a['outcomes']['positive_target_source_cells_removed']>=a['minimum_positive_target_source_cells']),'access means positive source-cell threshold only')
  require(all(a['outcomes'][k] is None for k in ('motor_surrogate','language_surrogate','clinical_deficit_probability')),'no clinical surrogate claims')
  require(row['native_counts']=={'search':len(actions),'rollout':len(actions),'replay':len(actions)},'three separate complete transition sequences')
  for k in transitions:transitions[k]+=row['native_counts'][k]
  previews+=row['native_previews']
  summary={'subject':s,'condition':condition,'status':'complete',**route,'target_fraction':region['fraction_of_full_region_removed'],'target_source_cells':a['outcomes']['positive_target_source_cells_removed'],'target_access_success':a['target_access_success'],'removed_cells':len(removed),'max_removed_cell_depth_mm':row['max_removed_cell_depth_mm'],'initial_catalog':{k:inventory[k] for k in ('declared_slots','emitted_count','accepted_count','omitted_count','complete')},'scores':score_count,'native_counts':row['native_counts'],'native_previews':row['native_previews'],'search_seconds':account['planning_seconds'],'complete_arm_seconds':row['arm_wall_seconds'],'world':common,'normalization':row['normalization']}
  require(derivation['operation']=='S OR (T > 0)' and derivation['source_support_unchanged'] is True and derivation['target_unchanged'] is True and derivation['anatomical_or_material_validation'] is False,'same explicit unvalidated union assumption')
  require(derivation['source_support_positive_voxels']==common['S_voxels'] and derivation['added_region_positive_voxels']==common['T_minus_S_voxels'] and extent['unsupported_region_positive_voxels']==0,'union count effect without target clipping')
  summary['state_catalogs']=state_summaries;summary['decision_summaries']=decision_summaries
  expected_config=baseline['release']['learning_protocol']['cohort_execution']['proposal_config']
  if condition==CONDITIONS[0]:
   old=baseline['plan_'+s];require(all(row[k]==old['plan'][k] for k in ('source_hash','decision_model_hash','initial_observation_hash')),'exact historical union world')
   require(row['proposal_config']==expected_config and inventory==baseline['initial-inventory_'+s],'historical baseline exact config and complete initial inventory')
   summary['historical_union_comparison']=route_comparison(route,reference_route(old));pairs[s]={'row':row,'summary':summary,'inventory':inventory}
  elif s in pairs:
   base=pairs[s]['row'];require(common==base['common_public_world'] and row['normalization']==base['normalization'] and derivation==base['derived_occupancy'] and extent==base['supplied_goal_extent'],'identical paired public world and normalization')
   require(row['proposal_config']=={**expected_config,'obstruction_opening':True} and all(row[k]!=base[k] for k in ('source_hash','decision_model_hash','initial_observation_hash')),'sole explicit rule change with changed identities')
   prior_rows=pairs[s]['inventory']['ledger'];require(without_ids(inventory['ledger'][:len(prior_rows)])==without_ids(prior_rows),'whole original physical prefix and dispositions preserved')
   require(row['pairing']['unchanged_public_world'] is True and row['pairing']['normalization_equal'] is True and row['pairing']['original_initial_prefix_equal_except_ids'] is True,'recorded matching claims')
   summary['paired_original_comparison']=route_comparison(route,pairs[s]['summary'])
  if control is not None:
   require(control['arms'][ordinal]=={'ordinal':ordinal,'plan_seal':row['plan_seal'],'plan_sha256':sha(directory/'plan.json'),'replay_sha256':sha(directory/'replay.json'),'search_sha256':sha(directory/'search.json'),'inventory_index_sha256':sha(directory/'search-inventories.json')},'parent endpoint exact file hashes')
  summaries.append(summary)
 if complete:
  require(transitions==result['native_counts'] and previews==costs['native_budget']['native_preview_entries'] and costs['source_visits_attempted']==8,'aggregate eight-arm work sums')
  require(costs['native_budget']['failure'] is None and costs['native_budget']['blocked_preview_attempts']==0,'no hidden preview exhaustion')
 for path,meta in list(H['seen'].items()):require(sha(ROOT/path)==meta['sha256'],'post-read stable evidence')
 require(sha(BASE)==BASE_SHA,'unchanged reused helper')
 summary={'status':'PASS_saved_metadata_audit','attempt_complete':complete,'result_status':result['status'],'result_sha256':expected_result,'release_sha256':expected_release,'checks':require.__globals__['checks'],'source_files':len(index['source_files']),'metadata_files':len(index['metadata_files']),'input_count':len(H['seen']),'arms':summaries,'costs':costs,'parent':{k:receipt[k] for k in ('status','elapsed_seconds','sampled_peak_rss_bytes','output_bytes','final_owned_pids','cleanup_errors')},'audit_seconds':time.monotonic()-start,'audit_peak_rss_bytes':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,'limitations':['Saved scalar/history/seal audit; no arrays, checkpoints, geometry or model rerun.','Only public T is scored. S union T is an explicit unvalidated simulation occupancy, not anatomical correction.','Greedy scores the currently emitted legal catalog; STOP does not prove absence of a negative-preparation route.','Equal120 capacity is not equal work: original rows retain priority; additions use remaining slots. Every captured state retains cap and discovery limits.','Complete-arm and parent costs include source preparation and replay; search selection time alone is not comparable with full learning/search runs.','Eight paired TRAIN arms do not establish transfer, patient benefit, clinical safety or validated material semantics.']}
 write('audit-result.json',summary);write('input-hashes.json',H['seen']);print(json.dumps({k:v for k,v in summary.items() if k not in ('arms','costs')},sort_keys=True))
if __name__=='__main__':
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('--terminal-result-sha256',required=True);p.add_argument('--terminal-release-sha256',required=True);a=p.parse_args();main(a.terminal_result_sha256,a.terminal_release_sha256)
