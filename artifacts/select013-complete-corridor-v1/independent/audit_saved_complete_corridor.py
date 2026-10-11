"""Saved JSON only: independent terminal, history, cost and causal audit."""
import collections
import hashlib
import json
import math
from pathlib import Path

ROOT=Path(__file__).resolve().parents[2]
RUN=ROOT/'build/select013-complete-corridor-v1'
P=RUN/'attempt-01'
OUT=Path(__file__).resolve().parent
pins={}

def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def read(p, expected=None):
    p=Path(p)
    if not p.is_absolute():p=P/p
    digest=sha(p)
    if expected:assert digest==expected,str(p)
    pins[str(p.relative_to(ROOT))]=digest
    return json.loads(p.read_text())
def digest(value):
    return 'sha256:'+hashlib.sha256(json.dumps(value,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()
def key(x):return x['tool_id'],tuple(x['entry_mm']),tuple(x['tip_mm'])
def cells(rows):return set(map(tuple,rows))

r=read('result.json','6aecbed77aa0235abaa03147d2cd2769cdab712dba1b1bcbe69449b4a725cd84')
receipt=read(RUN/'attempt-01.supervision/receipt.json','e69d62cb95c92f59140f71cfcec8b9df3969dda6cacb9213cc31ab73b3876caf')
release=read(RUN/'root-release.json','447442457d29f1f74a19d1284a04b2f6cf6f1fba00999f9b5ac30f81489e7206')
index=read(RUN/'source-index.json','82994d955b6d3c4c93132ecfb874d6b8034c072d08fe44c6bd6fda7489edaf73')
assert index['head']==release['expected_head']=='75ba5344d59ed8c96cad22ba9a47f94f3dbe2f61'
assert receipt['release_sha256']==sha(RUN/'root-release.json')
assert receipt['result_sha256']==sha(P/'result.json')
assert release['source_index']['sha256']==receipt['source_index']['sha256']==sha(RUN/'source-index.json')
final=read(RUN/'attempt-01.supervision/worker-final.json',receipt['worker_final_sha256'])
assert final['result_sha256']==final['canonical_result_sha256']==receipt['result_sha256']
assert receipt['status']=='complete' and receipt['exit_code']==0 and receipt['stop_reason'] is None
assert receipt['worker_termination_confirmed'] and not receipt['cleanup_errors'] and not receipt['final_owned_pids']
assert not receipt['automatic_retry']
for value,cap in [('elapsed_seconds','hard_total_wall_seconds'),('output_bytes','output_bytes'),('sampled_peak_rss_bytes','sampled_owned_tree_rss_bytes')]:
    assert receipt[value]<receipt['caps'][cap]
assert r['status']=='complete_SELECT013_corridor_comparison'
assert r['all_action_plans_sealed_before_evaluation'] and r['world_equal'] and r['source_physics_equal']
assert r['subject']=='ReMIND-013' and r['role']=='SELECT' and r['held_cases']==['ReMIND-037']
assert r['planned_SELECT_denominator']==2
for k in ['model_forwards','checkpoint_loads','optimizer_attempts','gradient_attempts','updates','private_reference_reads']:assert r[k]==0
for k in ['EVAL_opened','held_inputs_opened']:assert not r[k]

prior={}
for name in ['world','context']:
    ref=release['prior_pins'][name];prior[name]=read(ROOT/ref['path'],ref['sha256'])
for arm,row in r['arms'].items():
    control=read(arm+'/original-world-control.json')
    assert control['exact'] and all(control[k]==prior[k] for k in prior)
    horizon=read(arm+'/horizon-change.json');assert horizon['declared_horizon']==48 and horizon['historical_source_admission_horizon']==24
    plan=read(arm+'/selected-action-plan.json')
    assert plan['seal']==digest({k:v for k,v in plan.items() if k!='seal'})==row['plan_seal']
    assert plan['actions']==row['actions']==['STOP'] and plan['search_complete']
    metrics=read(arm+'/rollout-metrics.json');replay=read(arm+'/replay-binding.json')
    audit=read(arm+'/independent-episode.json',row['independent_certificate_sha256'])
    assert metrics['terminated'] and metrics['steps']==1 and len(metrics['history'])==1
    assert metrics['history'][0]['action_id']=='STOP'
    assert metrics['total_reward']==metrics['target_removed_mm3']==0
    assert replay['terminated'] and replay['history_sha256']==digest(metrics['history'])==audit['committed_history_hash']
    assert replay['metrics_sha256']==digest(metrics)
    assert audit['accepted'] and audit['complete_episode'] and not audit['target_access_success']
    assert audit['outcomes']==row['outcomes'] and audit['geometry']['feasible'] and not audit['geometry']['failures']
    assert row['complete_history_seal']==digest({'action_plan_seal':plan['seal'],'history':metrics['history'],'total_reward':metrics['total_reward']})
    assert audit['source_hash']==plan['source_hash']==horizon['source_hash']
    assert audit['decision_model_hash']==plan['decision_model_hash']==horizon['decision_model_hash']
    assert row['status']=='complete_replayed' and row['full_history_equal'] and row['independent_accepted']
rule=read('persistent_corridor/declared-rule-change.json')
assert rule['source_physics_equal'] and rule['proposal_config']['obstruction_selection_rule']=='target_ray_coverage_v3'
assert rule['post_exposure_authority']=={'capability_serialized':False,'same_canonical_capability':True,'same_prepared_object':True}

selection=read('persistent_corridor/selection.json','316fd5f092cc61b764257d2a152b820c6d757597db11df330adcdbd53560e8de')
a=selection['accounting'];states=a['states'];history=a['explored_history']
assert a['complete'] and not a['call_cap_reached'] and not a['time_cap_reached']
assert len(states)==len(history)==len(a['explored_actions'])==a['model_transition_calls']==48
assert a['terminal']=='HORIZON' and selection['actions']==['STOP']
assert {k:v for k,v in a.items() if k not in ['states','explored_history']}==r['arms']['persistent_corridor']['selector_accounting']
reward=0.;removed=set();trace=[];switches=[];goal_ids={};segments=[];last=None
base_calls=extra_calls=diagnostic_calls=page_count=legal=0
for i,(s,h) in enumerate(zip(states,history)):
    assert s['depth']==i+1 and s['complete_page_enumeration'] and s['goals_omitted_by_diagnostic_cap']==0
    assert h['action_id']==s['selected']['action_id']==a['explored_actions'][i]
    assert h['source_state_hash']==s['source_state_hash']
    if i<47:assert s['selected']['result_state_hash']==states[i+1]['source_state_hash']
    assert h['source_hash']==rule['source_hash']
    assert h['target_removed_mm3']==0 and h['reward']<0
    fresh=cells(h['removed_indices_native']);assert not (fresh&removed);removed|=fresh
    reward+=h['reward'];assert reward==s['selected']['cumulative_return'] and h['reward']==s['selected']['nominal_reward']
    k=key(s['selected']['goal']);g=next(x for x in s['goals'] if key(x)==k)
    if k not in goal_ids:goal_ids[k]=len(goal_ids)+1
    assert not s['selected']['direct_goal'] and g['eligible_legal_preparations']>0
    assert len(fresh&cells(g['retained_blocker_cells']))==s['selected']['current_blockers_removed']>0
    assert g['current_catalog_action_id'] is not None
    assert all(x['reason']=='SHAFT_BLOCKED_BY_REMAINING_NATIVE_TISSUE' and not x['feasible'] and x['first_failure_set_complete'] is True for x in s['goals'])
    pages=s['pages'];assert [p['page_index'] for p in pages]==list(range(len(pages)))
    base,extra=pages[0]['base_emitted_count'],pages[0]['additional_emitted_count']
    assert base<=120 and extra<=32 and len(pages)==1+(extra+7)//8
    assert all(p['base_emitted_count']==base and p['additional_emitted_count']==extra for p in pages)
    assert pages[0]['legal_actions']<=120 and all(p['legal_actions']<=8 for p in pages[1:])
    base_calls+=base;extra_calls+=extra;diagnostic_calls+=len(s['goals']);page_count+=len(pages);legal+=sum(p['legal_actions'] for p in pages)
    acc=s['obstruction_accounting'];assert not acc['discovery_complete'] and not acc['evidence_truncated']
    assert acc['selected_cells']==16 and acc['selection_omitted_cells']>0
    selected={tuple(c) for x in acc['target_ray_groups'] for c in x['selected_co_blockers']}
    if k!=last:
        segments.append({'from_step':i+1,'goal_id':goal_ids[k]})
        if last is not None:
            previous=next(x for x in s['goals'] if key(x)==last);blocked=cells(previous['retained_blocker_cells'])
            assert previous['eligible_legal_preparations']==0 and not previous['direct_legal_goal_available']
            aperture=[x for x in acc['static_aperture_pairs'] if tuple(x['voxel']) in blocked]
            eligible={tuple(x['voxel']) for x in aperture if not x['excluded']}
            switches.append({'step':i+1,'previous_goal_id':goal_ids[last],'next_goal_id':goal_ids[k],
                'previous_goal':previous,'selected_current_blockers':sorted(blocked&selected),
                'eligible_but_unselected':sorted((blocked&eligible)-selected),
                'no_centered_tool_passes_aperture':sorted(blocked-eligible),'aperture_pairs':aperture})
        last=k
    trace.append({'step':i+1,'goal_id':goal_ids[k],'action_id':h['action_id'],
        'source_state_hash':s['source_state_hash'],'result_state_hash':s['selected']['result_state_hash'],
        'removed_cells':sorted(fresh),'goal_current_blockers':g['retained_blocker_cells'],
        'overlap':s['selected']['current_blockers_removed'],'reward':h['reward'],'cumulative_return':reward,
        'preparation_entry_to_tip_mm':math.dist(h['entry_mm'],h['tip_mm']),
        'goal_entry_to_tip_mm':math.dist(g['entry_mm'],g['tip_mm']),
        'current_goal_legal_preparations':g['eligible_legal_preparations'],'selection_omitted_cells':acc['selection_omitted_cells']})
assert reward==a['explored_total_return']==-16.893798571567377 and len(removed)==74
assert a['selected_prefix_length']==0 and a['estimated_incremental_return']==0
assert len(goal_ids)==4 and len(segments)==8 and len(switches)==7
assert diagnostic_calls==a['diagnostic_preview_calls']==2496
assert base_calls+extra_calls+diagnostic_calls==selection['native_previews']==9193
exploration=read('persistent_corridor/exploration-geometry.json',r['exploration_geometry_sha256'])
assert exploration['history_sha256']==digest(history) and exploration['feasible']
geometry=exploration['geometry'];assert geometry['action_count']==48 and not geometry['failures']
assert geometry['unsupported_source_tissue_volume_mm3']==0 and geometry['complete_tool_checked'] and geometry['frontier_checked']
assert math.isclose(geometry['contained_source_tissue_volume_mm3'],74*geometry['source_voxel_volume_mm3'],abs_tol=1e-12)

costs=read('costs.json');assert costs['native_budget']==r['native_budget'] and costs['native_previews_by_arm']==r['native_previews_by_arm']
assert sum(v['native_preview_entries'] for v in costs['phases'].values())==r['native_budget']['native_preview_entries']==9895
assert r['native_previews_by_arm']=={'fresh_STOP':351,'persistent_corridor':9544}
for arm in r['arms']:
    assert sum(v.get('native_preview_entries',0) for k,v in costs['phases'].items() if k.startswith(arm+'.'))==r['native_previews_by_arm'][arm]
    for phase in ['search','rollout','replay']:
        count=costs['phases'][arm+'.'+phase].get('native_transition_calls',0)
        assert count==r['arms'][arm]['native_transitions'][phase]
assert sum(sum(x['native_transitions'].values()) for x in r['arms'].values())==52
assert r['native_budget']['failure'] is None and r['native_budget']['blocked_preview_attempts']==0 and r['native_budget']['counting_reliable']
assert r['native_budget']['time_overshoot_seconds']==0
assert a['wall_seconds']<release['limits']['search_seconds_per_arm'] and r['wall_seconds']<release['worker_seconds']
assert not r['comparison']['positive_complete_return'] and not r['comparison']['target_progress']
assert r['comparison']['nominal_return_improvement']==r['comparison']['target_improvement']==0

def blocker_allocation(state, blocked):
    acc=state['obstruction_accounting']
    chosen={tuple(c) for group in acc['target_ray_groups'] for c in group['selected_co_blockers']}
    pairs=sorted((x for x in acc['static_aperture_pairs'] if tuple(x['voxel']) in blocked),key=lambda x:(x['voxel'],x['tool_id']))
    return {'selected_cells':sorted(chosen&blocked),'aperture_pairs':pairs}
indirect=[]
for step, goal_id in [(10,1),(37,3),(38,2)]:
    before,after=states[step-1:step+1]
    physical=next(k for k,v in goal_ids.items() if v==goal_id)
    b=next(g for g in before['goals'] if key(g)==physical)
    c=next(g for g in after['goals'] if key(g)==physical)
    blocked=cells(b['retained_blocker_cells']);removed_here=cells(history[step-1]['removed_indices_native'])
    assert blocked==cells(c['retained_blocker_cells']) and not (removed_here&blocked)
    assert b['eligible_legal_preparations']==0 and c['eligible_legal_preparations']>0
    allocation=blocker_allocation(before,blocked)
    assert allocation==blocker_allocation(after,blocked)
    indirect.append({'action_step':step,'previously_pursued_goal_id':goal_id,
        'before_source_state_hash':before['source_state_hash'],'after_source_state_hash':after['source_state_hash'],
        'removed_elsewhere':sorted(removed_here),'unchanged_first_blockers':sorted(blocked),
        'legal_overlap_counts':[b['eligible_legal_preparations'],c['eligible_legal_preparations']],
        'unchanged_allocation':allocation,
        'interpretation':'supports indirect preparation; changing base geometries and absent legal/rejected ray lists prevent fixed-ray causal attribution'})

summary={'status':'PASS_saved_only','selected_both':['STOP'],'selected_return':0,'selected_target':0,
    'exploration':{'steps':48,'return':reward,'distinct_outside_cells':74,'outside_mm3':geometry['contained_source_tissue_volume_mm3'],
        'unique_physical_goals':4,'contiguous_goal_runs':8,'target_removed':0,'independent_geometry_feasible':True,
        'max_preparation_entry_to_tip_mm':max(x['preparation_entry_to_tip_mm'] for x in trace),
        'chosen_goal_entry_to_tip_mm':sorted(set(round(x['goal_entry_to_tip_mm'],9) for x in trace)),
        'post_action48_goals_observed':False},
    'costs':{'previews':9895,'search_previews':9193,'base_search_previews':base_calls,'extra_search_previews':extra_calls,
        'diagnostic_previews':diagnostic_calls,'native_transitions':52,'pages':page_count,'legal_catalog_candidates':legal,
        'worker_seconds':r['wall_seconds'],'parent_seconds':receipt['elapsed_seconds'],
        'sampled_peak_rss_bytes':receipt['sampled_peak_rss_bytes'],'output_bytes':receipt['output_bytes'],'cleanup_clean':True},
    'censoring':{'diagnostic_goal_omissions':0,'truncated_first_failure_sets':0,'states_with_cell_selection_omissions':48,
        'selection_omitted_cell_mentions':sum(s['obstruction_accounting']['selection_omitted_cells'] for s in states),
        'switches_with_eligible_active_blockers_unselected':sum(bool(s['eligible_but_unselected']) for s in switches)},
    'goals':[{'goal_id':v,'tool_id':k[0],'entry_mm':k[1],'tip_mm':k[2]} for k,v in goal_ids.items()],
    'goal_segments':segments,'switch_witnesses':switches,'indirect_preparation_witnesses':indirect,
    'compact_exploration_trace':trace,'pins':pins}
(OUT/'complete-corridor-saved-audit.json').write_text(json.dumps(summary,indent=2,sort_keys=True)+'\n')
print(json.dumps({k:v for k,v in summary.items() if k not in ['goals','goal_segments','switch_witnesses','indirect_preparation_witnesses','compact_exploration_trace','pins']},indent=2))
