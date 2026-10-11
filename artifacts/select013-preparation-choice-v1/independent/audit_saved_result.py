"""One stdlib-only audit of sealed saved paired outputs; no scientific imports."""
from pathlib import Path
import json, hashlib, math, collections
ROOT=Path(__file__).resolve().parents[3]
STAGE=ROOT/'build/select013-preparation-choice-v1'
OUT=STAGE/'attempt-01'
PINS={
 'attempt-01/result.json':'ef64519bd1762ab0867eeb09010e26c28cf8ef01430047ca76b82261e1cd0f1b',
 'attempt-01.supervision/receipt.json':'e61277083304852b3dea127332fdfebfc4744d11f509fd0c381b971a09443b70',
 'root-release.json':'ad17e551827cbecf90ff35809e7663c5d15a2ec3d2c56d8c48fb1e245b4434a0',
 'source-index.json':'7c8c2e03bc10356999582143000ac8076919051270d585ab4da65c9173106ce8',
}
def sha(p): return hashlib.sha256(p.read_bytes()).hexdigest()
def digest(x):return 'sha256:'+hashlib.sha256(json.dumps(x,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()
def read(p):return json.loads(p.read_text())
def cells(x):return set(map(tuple,x))
def physical(x):return {k:x[k] for k in ('tool_id','entry_mm','tip_mm')}
def key(x):return (x['tool_id'],tuple(x['entry_mm']),tuple(x['tip_mm']))
def check_seal(x):assert x['seal']==digest({k:v for k,v in x.items() if k!='seal'})
def close(a,b):assert math.isclose(a,b,rel_tol=1e-12,abs_tol=1e-10),(a,b)
for name,pin in PINS.items():assert sha(STAGE/name)==pin,name
result=read(OUT/'result.json');receipt=read(STAGE/'attempt-01.supervision/receipt.json')
release=read(STAGE/'root-release.json');index=read(STAGE/'source-index.json');costs=read(OUT/'costs.json')
assert receipt['status']=='complete' and receipt['exit_code']==0 and receipt['worker_termination_confirmed']
assert not receipt['cleanup_errors'] and not receipt['final_owned_pids'] and receipt['stop_reason'] is None
assert receipt['result_sha256']==PINS['attempt-01/result.json'] and receipt['release_sha256']==PINS['root-release.json']
assert receipt['source_index']==release['source_index'] and release['source_index']['sha256']==PINS['source-index.json']
assert release['expected_head']==index['head']=='a19b72993ef3ded6d147b364f45b7afc671975a1'
assert result['status']=='complete_SELECT013_preparation_choice_comparison'
assert result['all_action_plans_sealed_before_evaluation'] and result['world_equal'] and result['source_physics_equal']
assert result['focused_plan_context_replayed'] and result['exploration_geometry_feasible']
assert result['role']=='SELECT' and result['subject']=='ReMIND-013' and result['planned_SELECT_denominator']==2
assert not result['held_inputs_opened'] and not result['EVAL_opened']
assert all(result[k]==0 for k in ('checkpoint_loads','model_forwards','updates','optimizer_attempts','gradient_attempts','private_reference_reads'))
assert result['native_budget']['counting_reliable'] and result['native_budget']['failure'] is None
assert result['native_budget']['blocked_preview_attempts']==0
assert receipt['elapsed_seconds']<release['parent_seconds'] and result['wall_seconds']<release['worker_seconds']
assert receipt['sampled_peak_rss_bytes']<receipt['caps']['sampled_owned_tree_rss_bytes']
assert receipt['output_bytes']<receipt['caps']['output_bytes'] and receipt['caps']['threads']==1
final=STAGE/'attempt-01.supervision/worker-final.json'
assert sha(final)==receipt['worker_final_sha256'];f=read(final)
assert f['status']=='complete_owned_SELECT013_preparation_choice'
assert f['result_sha256']==f['canonical_result_sha256']==PINS['attempt-01/result.json']
# Verify the relevant already-reviewed runtime/core pins; do not re-audit the old broad closure.
source_pins={}
for name in ('comparison_contract.py','select_worker.py','run_owned.py','freeze-runtime.py','preparation_choice_selector.py'):
 p=STAGE/name;rel=p.relative_to(ROOT).as_posix();assert sha(p)==index['source_files'][rel];source_pins[rel]=sha(p)
for name in ('focused_corridor_selector.py','focused_paged_task.py','native_proposals.py','native_spatial_task.py','paged_obstruction.py','native_resection.py','geometry.py'):
 rel='src/resectionlab/'+name;assert sha(ROOT/rel)==index['source_files'][rel];source_pins[rel]=sha(ROOT/rel)
witness=read(STAGE/'decision5-witness.json');wp=release['prior_pins']['decision5_witness']
assert sha(STAGE/'decision5-witness.json')==wp['sha256'];PINS['decision5-witness.json']=wp['sha256']
protocols=witness['protocols'];assert set(protocols)=={'narrow','wide'}
for p in protocols.values():check_seal(p)
a,b=(protocols[x] for x in ('narrow','wide'))
assert {k:v for k,v in a.items() if k not in ('decision','seal')}=={k:v for k,v in b.items() if k not in ('decision','seal')}
assert {k:v for k,v in a['decision'].items() if k!='physical_action'}=={k:v for k,v in b['decision'].items() if k!='physical_action'}
assert a['decision']['physical_action']!=b['decision']['physical_action']
control=read(OUT/'narrow-historical-control.json');assert control==result['narrow_historical_control']
old_ref=release['prior_pins']['focused_selection'];old_path=ROOT/old_ref['path'];assert sha(old_path)==old_ref['sha256']
old=read(old_path)['accounting']
assert all(control[k] for k in ('exact_history','exact_action_records','exact_selected_records'))
summary={'status':'PASS_COMPLETE_NEGATIVE','pins':dict(PINS),'reviewed_source_pins':source_pins,
 'head':index['head'],'scope':'Saved JSON/source fixity only; no arrays, model, native execution or replay.',
 'common_protocol_fields_equal_except_treatment':True,'arms':{},'pair':{},'costs':{},
 'limitations':['Current query digests are joined across saved packet/state records, not recomputed from absent full raw catalogs.',
 'Seal-before-evaluation ordering is supported by reviewed worker control flow and saved paired plan bindings, not an independent event timestamp system.',
 'Full negative exploration has independent geometry; only the selected STOP plans have fresh authoritative replay.',
 'Terminal known remaining blockers are retained pre-action cells not removed by the final action; no new post-horizon preview is inferred.']}
loaded={};worlds={};plans={};total_previews=total_transitions=0
for arm in ('narrow','wide'):
 folder=OUT/arm;sel=read(folder/'selection.json');acc=sel['accounting'];states=acc['states'];hist=acc['explored_history'];packets=acc['explored_action_records']
 loaded[arm]=acc;row=result['arms'][arm];p=protocols[arm];plan=read(folder/'selected-action-plan.json');plans[arm]=plan
 check_seal(plan);assert plan['seal']==row['plan_seal']==control['plans_already_sealed'][arm]
 assert plan['intervention_protocol']==acc['intervention_protocol']==p
 assert plan['actions']==sel['actions']==row['actions']==['STOP']
 assert plan['action_records']==acc['selected_action_records'] and len(plan['action_records'])==1
 check_seal(plan['action_records'][0]);assert plan['action_records'][0]['steps_taken']==0
 assert acc['complete'] and acc['prefix_verified_count']==4 and acc['intervention_applied']
 assert not acc['time_cap_reached'] and not acc['call_cap_reached'] and acc['terminal']=='HORIZON'
 assert acc['selected_prefix_length']==0 and acc['estimated_incremental_return']==0
 assert acc['model_transition_calls']==len(states)==len(hist)==len(packets)==48
 assert row['native_transitions']==dict(search=48,rollout=1,replay=1)
 assert row['status']=='complete_replayed' and row['focus_context_replayed'] and row['full_history_equal'] and row['independent_accepted']
 assert row['focus_validity']['explicitly_cleared_count']==0
 assert row['selector_accounting']=={k:v for k,v in acc.items() if k not in ('states','explored_history')}
 metrics=read(folder/'rollout-metrics.json');rb=read(folder/'replay-binding.json');episode=read(folder/'independent-episode.json');geo=read(folder/'exploration-geometry.json')
 assert metrics['terminated'] and rb['terminated'] and digest(metrics)==rb['metrics_sha256']
 assert digest(metrics['history'])==rb['history_sha256']==episode['committed_history_hash']
 assert row['complete_history_seal']==digest({'action_plan_seal':plan['seal'],'history':metrics['history'],'total_reward':metrics['total_reward']})
 assert row['outcomes']==episode['outcomes'] and episode['accepted'] and episode['complete_episode']
 assert not episode['target_access_success'] and episode['geometry']['complete_tool_checked'] and episode['geometry']['feasible']
 assert episode['geometry']['action_count']==0 and not episode['geometry']['failures']
 assert episode['outcomes']['total_reward']==episode['outcomes']['target_removed_mm3']==episode['outcomes']['normal_removed_mm3']==0
 assert sha(folder/'independent-episode.json')==row['independent_certificate_sha256']
 assert sha(folder/'exploration-geometry.json')==row['exploration_geometry_sha256']
 assert geo['feasible'] and geo['geometry']['feasible'] and geo['geometry']['complete_tool_checked'] and geo['geometry']['frontier_checked']
 assert geo['geometry']['action_count']==48 and not geo['geometry']['failures'] and geo['geometry']['unsupported_source_tissue_volume_mm3']==0
 assert geo['history_sha256']==digest(hist) and geo['source_hash']==plan['source_hash'] and geo['decision_model_hash']==plan['decision_model_hash']
 world=read(folder/'original-world-control.json');rule=read(folder/'declared-rule-change.json')
 assert world['exact'] and rule['source_physics_equal'];worlds[arm]=world
 assert rule['source_hash']==plan['source_hash'] and rule['decision_model_hash']==plan['decision_model_hash']
 removed=set();running=0.;trace=[];base=extra=pages=legal=goal_calls=dep_calls=focused_omissions=global_omissions=0
 layer_changes=[];last_layer=None;all_goals=set();changes=0;last_tool=None;full_dep_sidecars=0
 for i,(state,event,packet) in enumerate(zip(states,hist,packets)):
  chosen=state['selected'];check_seal(packet)
  assert packet['action_id']==event['action_id']==chosen['action_id']==acc['explored_actions'][i]
  assert packet['steps_taken']==i and packet['source_state_hash']==state['source_state_hash']==event['source_state_hash']
  assert packet['physical_action']==physical(event)==chosen['action_geometry']
  assert packet['source_hash']==plan['source_hash'] and packet['decision_model_hash']==plan['decision_model_hash']
  assert packet['current_tool']==last_tool
  assert state['goal_focus']==packet['focus'];assert state['focus_explicitly_cleared'] is None
  if i:assert states[i-1]['selected']['result_state_hash']==state['source_state_hash']
  if i<4:
   assert physical(event)==p['prefix'][i]['physical_action']
   assert state['source_state_hash']==p['prefix'][i]['source_state_hash'] and chosen['result_state_hash']==p['prefix'][i]['result_state_hash']
  if packet['focus'] is not None:
   focus=packet['focus'];assert focus['source_state_hash']==state['source_state_hash'] and focus['source_hash']==plan['source_hash']
   assert focus['native_decision_model_hash']==event['decision_model_hash'] and physical(focus)==physical(state['goals'][0])
   fc=state['obstruction_accounting']['goal_focus'];assert fc['record']==focus
   assert fc['retained_cells']==state['goals'][0]['retained_blocker_cells']
   assert fc['first_failure']['diagnostic_hash']==state['goals'][0]['sidecar_fingerprint']
   focused_omissions+=len(fc['unselected_cells']);assert not fc['unselected_cells']
  newly=cells(event['removed_indices_native']);assert newly==cells(chosen['actual_removed_cells']) and not newly & removed
  micro=set()
  for m in event['microsteps']:micro |= cells(m['removed_indices_native'])
  assert micro==newly;removed|=newly
  close(event['reward'],chosen['nominal_reward']);running+=event['reward'];close(running,chosen['cumulative_return']);assert event['reward']<0 and event['target_removed_mm3']==0
  all_goals.add(key(chosen['goal']));assert key(chosen['goal'])==key(p['decision']['active_goal'])
  assert state['complete_page_enumeration'] and state['goals_omitted_by_diagnostic_cap']==0
  inventory=state['obstruction_accounting'];ps=state['pages'];bc=ps[0]['base_emitted_count'];ec=ps[0]['additional_emitted_count']
  assert bc<=120 and ec<=32 and len(ps)==1+(ec+7)//8
  assert [x['page_index'] for x in ps]==list(range(len(ps)))
  assert all(x['base_emitted_count']==bc and x['additional_emitted_count']==ec for x in ps)
  assert all(x['legal_actions']<=8 for x in ps[1:])
  base+=bc;extra+=ec;pages+=len(ps);legal+=sum(x['legal_actions'] for x in ps);goal_calls+=len(state['goals'])
  global_omissions+=inventory['selection_omitted_cells']
  d=state.get('dependency_diagnostics')
  if d:
   assert d['source_state_hash']==state['source_state_hash'] and not d['pair_cap_reached']
   dep_calls+=len(d['probes']);prereqs=set()
   for probe in d['probes']:
    prereqs |= cells(probe['retained_prerequisite_cells']);side=probe.get('sidecar')
    if side is not None:
     assert side['fingerprint']==digest({k:v for k,v in side.items() if k!='fingerprint'})
     assert side['source_state_hash']==state['source_state_hash'] and side['source_hash']==plan['source_hash']
     assert side['decision_model_hash']==event['decision_model_hash']
     assert side['tool']['tool_id']==probe['tool_id'] and side['entry_mm']==probe['entry_mm'] and side['requested_tip_mm']==probe['tip_mm']
     assert cells(side['blocked_indices_native'])==cells(probe['retained_prerequisite_cells'])
     full_dep_sidecars+=1
   assert prereqs==cells(d['retained_prerequisite_union'])
   effect=d['selected_effect']
   if effect:
    assert effect['action_id']==event['action_id'] and cells(effect['actual_removed_cells'])==newly
    assert cells(effect['sampled_prerequisite_cells_removed'])==newly & prereqs
    for link in effect['prep_ray_links']:
     matches=[v for v in d['probes'] if v['current_catalog_action_id']==link['prep_action_id'] and physical(v)==link['prep_geometry']]
     assert len(matches)==1 and cells(link['removed_prerequisites'])==newly & cells(matches[0]['retained_prerequisite_cells'])
  layer=sorted(set(c[0] for c in state['goals'][0]['retained_blocker_cells']))
  if layer!=last_layer:layer_changes.append({'decision':i+1,'first_blocker_layers':layer});last_layer=layer
  if last_tool is not None and last_tool!=event['tool_id']:changes+=1
  last_tool=event['tool_id']
  trace.append({'decision':i+1,'first_blocker_layers':layer,'active_first_count':len(state['goals'][0]['retained_blocker_cells']),
   'tool':event['tool_id'],'removed_count':len(newly),'progress_kind':chosen['progress_kind'],'reward':event['reward'],'cumulative_return':running})
 close(running,acc['explored_total_return']);assert len(all_goals)==1
 assert goal_calls==acc['goal_diagnostic_preview_calls'] and dep_calls==acc['dependency_preview_calls']
 assert goal_calls+dep_calls==acc['diagnostic_preview_calls']
 assert base+extra+goal_calls+dep_calls==sel['native_previews']==costs['phases'][arm+'.search']['native_preview_entries']
 arm_previews=sum(v.get('native_preview_entries',0) for k,v in costs['phases'].items() if k.startswith(arm+'.'))
 assert arm_previews==result['native_previews_by_arm'][arm]
 for phase in ('search','rollout','replay'):
  assert costs['phases'][arm+'.'+phase]['native_transition_calls']==row['native_transitions'][phase]
 total_previews+=arm_previews;total_transitions+=sum(row['native_transitions'].values())
 assert arm_previews<=release['limits']['native_previews_per_arm'] and acc['wall_seconds']<release['limits']['search_seconds_per_arm']
 affine=hist[0]['native_affine'];entry=hist[0]['entry_mm'];axis=hist[0]['axis_unit']
 center_depths=[]
 for c in removed:
  point=[sum(affine[i][j]*c[j] for j in range(3))+affine[i][3] for i in range(3)]
  center_depths.append(sum((point[i]-entry[i])*axis[i] for i in range(3)))
 totals={k:sum(x[k] for x in hist) for k in ('reward','target_removed_mm3','normal_removed_mm3','removed_volume_mm3','complete_tool_path_length_mm')}
 close(totals['removed_volume_mm3'],geo['geometry']['contained_source_tissue_volume_mm3'])
 close(len(removed)*geo['geometry']['source_voxel_volume_mm3'],totals['removed_volume_mm3'])
 choice=states[4]['choice_intervention'];event=hist[4];packet=packets[4];current=cells(states[4]['goals'][0]['retained_blocker_cells'])
 assert choice['source_state_hash']==p['decision']['source_state_hash']==packet['source_state_hash']
 assert choice['fixed_physical_choice']==p['decision']['physical_action']==physical(event)
 assert choice['active_goal']==p['decision']['active_goal'] and choice['current_tool']==p['decision']['current_tool']
 assert choice['current_catalog_action_id']==packet['action_id'] and choice['query_catalog_digest']==packet['query_catalog_digest']
 assert choice['protocol_seal']==p['seal'] and choice['fresh_current_certificate']
 assert cells(choice['active_retained_first_blockers'])==current
 assert cells(choice['actual_removed_current_blockers'])==cells(event['removed_indices_native']) & current
 assert states[4]['selected']['current_blockers_removed']==len(choice['actual_removed_current_blockers'])==1
 close(choice['nominal_reward'],event['reward'])
 for name in ('selection.json','selected-action-plan.json','exploration-geometry.json','independent-episode.json','rollout-metrics.json','replay-binding.json','original-world-control.json','declared-rule-change.json'):
  summary['pins'][arm+'/'+name]=sha(folder/name)
 summary['arms'][arm]={'selection_complete':True,'full_geometry_accepted':True,'STOP_replay_accepted':True,
  'history_semantic_hash':digest(hist),'packets_semantic_hash':digest(packets),'plan_seal':plan['seal'],
  'selection_wall_seconds':acc['wall_seconds'],'exploration':totals,'removed_cells':len(removed),'tool_changes':changes,
  'tools':dict(collections.Counter(x['tool_id'] for x in hist)),
  'max_removed_native_layer':max(c[0] for c in removed),'max_removed_cell_center_depth_mm':max(center_depths),
  'max_tip_depth_mm':max(sum((x['tip_mm'][i]-x['entry_mm'][i])*x['axis_unit'][i] for i in range(3)) for x in hist),
  'layer_changes':layer_changes,'all_rewards_negative':True,'single_persistent_goal':p['decision']['active_goal'],
  'known_preterminal_blockers_still_unremoved':sorted(cells(states[-1]['goals'][0]['retained_blocker_cells'])-cells(hist[-1]['removed_indices_native'])),
  'decision5':{**choice,'all_removed_cells':event['removed_indices_native'],'path_length_mm':event['complete_tool_path_length_mm'],
    'removed_volume_mm3':event['removed_volume_mm3']},
  'focus_omissions':focused_omissions,'global_omitted_cell_mentions':global_omissions,
  'base_previews':base,'additional_previews':extra,'goal_diagnostic_previews':goal_calls,'dependency_previews':dep_calls,
  'dependency_sidecars_checked':full_dep_sidecars,'pages':pages,'legal_candidate_mentions':legal,
  'total_previews':arm_previews,'search_previews':sel['native_previews'],'native_transitions':row['native_transitions'],
  'progress_kinds':dict(collections.Counter(s['selected']['progress_kind'] for s in states)),'trace':trace}
assert worlds['narrow']==worlds['wide']
n,w=loaded['narrow'],loaded['wide']
assert n['explored_history'][:4]==w['explored_history'][:4] and n['explored_action_records'][:4]==w['explored_action_records'][:4]
assert n['states'][:4]==w['states'][:4]
assert {k:v for k,v in n['states'][4].items() if k not in ('choice_intervention','selected')}=={k:v for k,v in w['states'][4].items() if k not in ('choice_intervention','selected')}
assert n['explored_history']==old['explored_history'] and n['explored_action_records']==old['explored_action_records'] and n['selected_action_records']==old['selected_action_records']
assert digest(n['explored_history'])==control['current_history_sha256']==control['historical_history_sha256']
assert digest(n['explored_action_records'])==control['current_packets_sha256']==control['historical_packets_sha256']
assert plans['narrow']['action_records']==plans['wide']['action_records']
assert total_previews==19770==sum(v.get('native_preview_entries',0) for v in costs['phases'].values())==result['native_budget']['native_preview_entries']
assert total_previews<=release['limits']['native_previews_total'] and total_transitions==100<=release['limits']['native_transitions_total']
assert costs['native_budget']==result['native_budget'] and costs['native_previews_by_arm']==result['native_previews_by_arm']
profile=result['native_budget']['native_preview_profile']['phases']['unclassified']
assert profile['started']==profile['returned']==profile['feasible']+profile['rejected']==total_previews and profile['raised']==0
summary['pair']={'common_prefix_exact_full_history_packets_states':True,'current_decision5_state_catalog_goal_previews_exact':True,
 'narrow_historical_full_history_packets_selected_packets_exact':True,'both_selected_identical_STOP_packet':True,
 'wide_minus_narrow_exploration_return':w['explored_total_return']-n['explored_total_return'],
 'wide_minus_narrow_exploration_cells':summary['arms']['wide']['removed_cells']-summary['arms']['narrow']['removed_cells'],
 'wide_minus_narrow_selected_return':0,'wide_minus_narrow_selected_target':0,
 'causal_scope':'Two predeclared physical preparations at one common simulator state; tool and lateral aim both differ. No isolated tool-width, clinical or learned-policy claim.'}
summary['costs']={'native_previews':total_previews,'native_transitions':total_transitions,
 'phase_counts_and_inclusive_times':costs['phases'],'worker_seconds':result['wall_seconds'],
 'parent_seconds':receipt['elapsed_seconds'],'sampled_peak_rss_bytes':receipt['sampled_peak_rss_bytes'],
 'output_bytes':receipt['output_bytes'],'clean_exit_0':True,'all_declared_caps_respected':True,
 'preview_profile':profile,'timing_note':'Do not add nested native/clone timings to enclosing phases. Historical timing is not a matched speed comparator.'}
summary['pins']['costs.json']=sha(OUT/'costs.json');summary['pins']['narrow-historical-control.json']=sha(OUT/'narrow-historical-control.json')
p=STAGE/'independent/saved-result-audit.json';p.write_text(json.dumps(summary,indent=2,sort_keys=True)+'\n')
print(json.dumps({'status':summary['status'],'audit_bytes':p.stat().st_size,'audit_sha256':sha(p),'pair':summary['pair'],'native_previews':total_previews,'native_transitions':total_transitions},indent=2))
