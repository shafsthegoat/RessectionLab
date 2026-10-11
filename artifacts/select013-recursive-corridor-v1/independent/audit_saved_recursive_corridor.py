"""Saved JSON only: independent terminal, history, cost and causal audit."""
import collections
import hashlib
import json
import math
from pathlib import Path

ROOT=Path(__file__).resolve().parents[2]
RUN=ROOT/'build/select013-recursive-corridor-v1'
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

r=read('result.json','fa418501772054f61f26cc5ab410cf1f05c90a7126be429e09787916816fc411')
receipt=read(RUN/'attempt-01.supervision/receipt.json','013a525881729be2fdf7d1713ebe3aa3bb2b5c03aafd29b39930597d48fa368b')
release=read(RUN/'root-release.json','f86d2faa22a247e320bcc2d46cd9a0737c5e7d4bb0fec6719bc51a2fd138d225')
index=read(RUN/'source-index.json','e56fc0ba54821f678291439d1718ca0eeeed66d0f022d70cc556600cf46011e1')
assert index['head']==release['expected_head']=='9286a6b094a7d23e46835d26a73bb409522322de'
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
assert r['status']=='complete_SELECT013_recursive_corridor_comparison'
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
rule=read('recursive_corridor/declared-rule-change.json')
assert rule['source_physics_equal'] and rule['proposal_config']['obstruction_selection_rule']=='target_ray_coverage_v3'
assert rule['post_exposure_authority']=={'capability_serialized':False,'same_canonical_capability':True,'same_prepared_object':True}

selection=read('recursive_corridor/selection.json','fe2b03c4d600a797451e7c9e3e380e9123fd96741e508083fb8d84a41c112ffd')
a=selection['accounting'];states=a['states'];history=a['explored_history']
assert a['complete'] and not a['call_cap_reached'] and not a['time_cap_reached']
assert len(states)==len(history)==len(a['explored_actions'])==a['model_transition_calls']==48
assert a['terminal']=='HORIZON' and selection['actions']==['STOP']
assert {k:v for k,v in a.items() if k not in ['states','explored_history']}==r['arms']['recursive_corridor']['selector_accounting']

ready=read(RUN/'runtime-source-ready.json','103af78b0867aae05cb00b559a43c764a132e0b7e41a997564bfe3979d31bda1')
for name,pin in ready['files'].items():
    assert index['source_files'][str((RUN/name).relative_to(ROOT))]==pin
    assert sha(RUN/name)==pin
assert index['source_files'][str((RUN/'recursive_corridor_selector.py').relative_to(ROOT))]=='dc670b56f57e82786de4057277c691219b1de36d7c35f6a1102b59f63b888e93'
reward=0.;removed=set();trace=[];goal_ids={};segments=[];last=None
base_calls=extra_calls=goal_calls=dep_calls=page_count=legal=0
prerequisites=[];all_probes=[];allocation_witnesses=[];switches=[]
def allocation(state,blocked):
    acc=state['obstruction_accounting']
    chosen={tuple(c) for g in acc['target_ray_groups'] for c in g['selected_co_blockers']}
    pairs=[x for x in acc['static_aperture_pairs'] if tuple(x['voxel']) in blocked]
    eligible={tuple(x['voxel']) for x in pairs if not x['excluded']}
    return {'selected':sorted(blocked&chosen),'eligible_omitted':sorted((blocked&eligible)-chosen),
            'all_tools_aperture_excluded':sorted(blocked-eligible),'pairs':pairs}
for i,(s,h) in enumerate(zip(states,history)):
    selected=s['selected'];fresh=cells(h['removed_indices_native']);k=key(selected['goal'])
    g=next(x for x in s['goals'] if key(x)==k)
    assert s['depth']==i+1 and s['complete_page_enumeration'] and s['goals_omitted_by_diagnostic_cap']==0
    assert h['action_id']==selected['action_id']==a['explored_actions'][i]
    assert h['source_state_hash']==s['source_state_hash'] and h['source_hash']==rule['source_hash']
    if i<47:assert selected['result_state_hash']==states[i+1]['source_state_hash']
    assert key(h)==key(selected['action_geometry']) and fresh==cells(selected['actual_removed_cells'])
    assert fresh==set().union(*(cells(m['removed_indices_native']) for m in h['microsteps']))
    assert not fresh&removed;removed|=fresh
    assert h['target_removed_mm3']==0 and h['reward']<0 and h['normal_removed_mm3']==h['removed_volume_mm3']
    reward+=h['reward'];assert reward==selected['cumulative_return'] and h['reward']==selected['nominal_reward']
    blocked=cells(g['retained_blocker_cells'])
    assert len(fresh&blocked)==selected['current_blockers_removed']
    assert not selected['direct_goal']
    if k not in goal_ids:goal_ids[k]=len(goal_ids)+1
    if k!=last:
        segments.append({'from_step':i+1,'goal_id':goal_ids[k]})
        if last is not None:
            prev=next(x for x in s['goals'] if key(x)==last)
            assert not prev['direct_legal_goal_available'] and prev['eligible_legal_preparations']==0
            switches.append({'step':i+1,'previous_goal_id':goal_ids[last],'new_goal_id':goal_ids[k],
                'previous_blockers':prev['retained_blocker_cells'],'allocation':allocation(s,cells(prev['retained_blocker_cells']))})
        last=k
    assert all(x['reason']=='SHAFT_BLOCKED_BY_REMAINING_NATIVE_TISSUE' and not x['feasible'] and x['first_failure_set_complete'] for x in s['goals'])
    pages=s['pages'];base=pages[0]['base_emitted_count'];extra=pages[0]['additional_emitted_count']
    assert [x['page_index'] for x in pages]==list(range(1+(extra+7)//8)) and base<=120 and extra<=32
    assert all(x['base_emitted_count']==base and x['additional_emitted_count']==extra for x in pages)
    assert pages[0]['legal_actions']<=120 and all(x['legal_actions']<=8 for x in pages[1:])
    base_calls+=base;extra_calls+=extra;goal_calls+=len(s['goals']);page_count+=len(pages);legal+=sum(x['legal_actions'] for x in pages)
    acc=s['obstruction_accounting'];assert not acc['evidence_truncated'] and not acc['discovery_complete']
    assert acc['selected_cells']==16 and acc['selection_omitted_cells']>0
    d=s.get('dependency_diagnostics')
    if selected['progress_kind']=='prerequisite':assert d is not None
    if d:
        assert d['source_state_hash']==s['source_state_hash'] and d['dependency_depth']==2
        assert d['trigger']=='established_active_goal_has_no_current_direct_legal_progress'
        assert not d['pair_cap_reached'] and d['pair_limit']==32
        assert d['emitted_centered_pair_count']==len(d['probes'])<=32
        active=next(x for x in s['goals'] if key(x)==key(d['active_goal']))
        assert key(states[i-1]['selected']['goal'])==key(d['active_goal'])
        assert active['eligible_legal_preparations']==0 and not active['direct_legal_goal_available']
        active_cells=cells(active['retained_blocker_cells']);assert active_cells==cells(d['active_retained_blockers'])
        dep_calls+=len(d['probes']);union=set();seen=set()
        for p in d['probes']:
            assert key(p) not in seen;seen.add(key(p));side=p['sidecar']
            assert tuple(p['centered_active_blocker']) in active_cells
            assert not p['feasible'] and p['reason']=='SHAFT_BLOCKED_BY_REMAINING_NATIVE_TISSUE'
            assert side['fingerprint']==digest({q:v for q,v in side.items() if q!='fingerprint'})
            assert side['source_hash']==h['source_hash'] and side['source_state_hash']==s['source_state_hash']
            assert side['decision_model_hash']==h['decision_model_hash'] and side['tool']['tool_id']==p['tool_id']
            assert side['entry_mm']==p['entry_mm'] and side['requested_tip_mm']==p['tip_mm']
            assert side['reason']==p['reason'] and side['blocked_indices_native']==p['retained_prerequisite_cells']
            assert side['blocked_cell_count']==side['retained_cell_count']==len(p['retained_prerequisite_cells'])<=16
            assert side['complete_first_failure_set'] and not side['truncated'] and not side['prior_temporary_removals_committed']
            union|=cells(p['retained_prerequisite_cells']);all_probes.append((i,p))
        assert union==cells(d['retained_prerequisite_union'])
        alloc=allocation(s,active_cells)
        allocation_witnesses.append({'step':i+1,'active_goal':d['active_goal'],'active_first_blockers':sorted(active_cells),
            'centered_pairs':len(d['probes']),'selected':alloc['selected'],'eligible_omitted':alloc['eligible_omitted'],
            'all_tools_aperture_excluded':alloc['all_tools_aperture_excluded'],
            'per_tool_aperture':[{q:z[q] for q in ['voxel','tool_id','clearance_mm','excluded']} for z in alloc['pairs']]})
        if selected['progress_kind']=='prerequisite':
            assert k==key(d['active_goal']) and not fresh&active_cells
            effect=d['selected_effect'];hit=fresh&union
            assert hit and effect['action_id']==selected['action_id']
            assert cells(effect['actual_removed_cells'])==fresh
            assert cells(effect['sampled_prerequisite_cells_removed'])==hit
            assert len(hit)==selected['sampled_prerequisites_removed']
            links=[]
            for p in d['probes']:
                overlap=fresh&cells(p['retained_prerequisite_cells'])
                if overlap:
                    links.append((p['current_catalog_action_id'],key(p),overlap))
            assert len(links)==len(effect['prep_ray_links'])
            for link,expected in zip(effect['prep_ray_links'],links):
                assert (link['prep_action_id'],key(link['prep_geometry']),cells(link['removed_prerequisites']))==expected
            prerequisites.append({'step':i+1,'source_state_hash':s['source_state_hash'],
                'active_goal':d['active_goal'],'active_first_blockers':d['active_retained_blockers'],
                'action_geometry':selected['action_geometry'],'reward':h['reward'],
                'effect':effect,'probes':[{'action_id':p['current_catalog_action_id'],'geometry':{q:p[q] for q in ['tool_id','entry_mm','tip_mm']},
                   'centered_active_blocker':p['centered_active_blocker'],'sidecar_fingerprint':p['sidecar']['fingerprint'],
                   'prerequisite_cells':p['retained_prerequisite_cells'],'temporary_discarded_count':p['sidecar']['prior_temporary_removed_count']}
                   for p in d['probes']]})
        else:assert d.get('selected_effect') is None
    if selected['progress_kind']=='direct_blocker':assert fresh&blocked and g['eligible_legal_preparations']>0
    trace.append({'step':i+1,'goal_id':goal_ids[k],'progress_kind':selected['progress_kind'],
        'removed':sorted(fresh),'reward':h['reward'],'cumulative_return':reward,
        'current_blockers_removed':selected['current_blockers_removed'],
        'sampled_prerequisites_removed':selected['sampled_prerequisites_removed'],
        'entry_to_tip_mm':math.dist(h['entry_mm'],h['tip_mm']),
        'goal_entry_to_tip_mm':math.dist(g['entry_mm'],g['tip_mm']),
        'active_layer_max':max(c[0] for c in blocked)})
assert reward==a['explored_total_return']==-17.568598513619033 and len(removed)==77
assert a['selected_prefix_length']==0 and a['estimated_incremental_return']==0
assert len(prerequisites)==14 and len(allocation_witnesses)==15 and dep_calls==36
assert goal_calls==a['goal_diagnostic_preview_calls']==2496
assert dep_calls==a['dependency_preview_calls'] and goal_calls+dep_calls==a['diagnostic_preview_calls']==2532
assert base_calls+extra_calls+goal_calls+dep_calls==selection['native_previews']==9154
exploration=read('recursive_corridor/exploration-geometry.json',r['exploration_geometry_sha256'])
assert exploration['history_sha256']==digest(history) and exploration['feasible']
geometry=exploration['geometry'];assert geometry['action_count']==48 and not geometry['failures']
assert geometry['unsupported_source_tissue_volume_mm3']==0 and geometry['complete_tool_checked'] and geometry['frontier_checked']
assert math.isclose(geometry['contained_source_tissue_volume_mm3'],77*geometry['source_voxel_volume_mm3'],abs_tol=1e-12)
costs=read('costs.json');assert costs['native_budget']==r['native_budget'] and costs['native_previews_by_arm']==r['native_previews_by_arm']
assert sum(v['native_preview_entries'] for v in costs['phases'].values())==r['native_budget']['native_preview_entries']==9856
assert r['native_previews_by_arm']=={'fresh_STOP':351,'recursive_corridor':9505}
for arm in r['arms']:
    assert sum(v.get('native_preview_entries',0) for k,v in costs['phases'].items() if k.startswith(arm+'.'))==r['native_previews_by_arm'][arm]
    for phase in ['search','rollout','replay']:
        assert costs['phases'][arm+'.'+phase].get('native_transition_calls',0)==r['arms'][arm]['native_transitions'][phase]
assert sum(sum(x['native_transitions'].values()) for x in r['arms'].values())==52
assert r['native_budget']['failure'] is None and r['native_budget']['blocked_preview_attempts']==0 and r['native_budget']['counting_reliable']
assert r['native_budget']['time_overshoot_seconds']==0
assert a['wall_seconds']<release['limits']['search_seconds_per_arm'] and r['wall_seconds']<release['worker_seconds']
assert not r['comparison']['positive_complete_return'] and not r['comparison']['target_progress']
assert r['comparison']['nominal_return_improvement']==r['comparison']['target_improvement']==0
# Exact literal physical-ray identity, not approximate direction or action-ID equality.
exact_execution=[]
for j,h in enumerate(history):
    preceding=[(i,p) for i,p in all_probes if i<j and key(p)==key(h)]
    if not preceding:continue
    linked=[x for x in prerequisites if x['step']<=j and any(key(z['prep_geometry'])==key(h) for z in x['effect']['prep_ray_links'])]
    assert linked
    exact_execution.append({'execution_step':j+1,'geometry':{q:h[q] for q in ['tool_id','entry_mm','tip_mm']},
        'removed':h['removed_indices_native'],'reward':h['reward'],'source_state_hash':h['source_state_hash'],
        'rejected_probe_steps':[i+1 for i,p in preceding],
        'linked_prerequisite_steps':[x['step'] for x in linked],
        'latest_rejected_probe':{'step':preceding[-1][0]+1,'sidecar':{q:preceding[-1][1]['sidecar'][q] for q in ['fingerprint','source_hash','source_state_hash','decision_model_hash','entry_mm','requested_tip_mm','blocked_indices_native','complete_first_failure_set','failure_interval_index','failure_tip_mm','prior_temporary_removed_count']}}})
assert [x['execution_step'] for x in exact_execution]==[11,19,23,32,40,48]
# Retain strongest repeated-layer counterexample with full original sidecars.
p8=states[7]['dependency_diagnostics']['probes'][0];p9=states[8]['dependency_diagnostics']['probes'][0]
assert key(p8)==key(p9) and cells(p8['retained_prerequisite_cells'])<=cells(history[7]['removed_indices_native'])
assert cells(p9['retained_prerequisite_cells']).isdisjoint(cells(p8['retained_prerequisite_cells']))
assert not p9['feasible']
layer_witness={'steps':[8,9],'geometry':{q:p8[q] for q in ['tool_id','entry_mm','tip_mm']},
    'before_sidecar':p8['sidecar'],'after_sidecar':p9['sidecar'],'removed_at8':history[7]['removed_indices_native'],
    'interpretation':'Clearing the complete first rejected set leaves this exact ray rejected by another layer; it commits at11 after further preparation.'}
summary={'status':'PASS_saved_only_complete_negative','selected_actions':{k:x['actions'] for k,x in r['arms'].items()},
 'exploration':{'steps':48,'outside_cells':77,'outside_mm3':geometry['contained_source_tissue_volume_mm3'],
   'target_removed':0,'return':reward,'unique_active_goal_geometries':len(goal_ids),'goal_runs':len(segments),
   'max_removed_native_axis0_layer':max(x[0] for x in removed),'max_removed_center_advance_mm':max(sum((sum(h['native_affine'][q][j]*c[j] for j in range(3))+h['native_affine'][q][3]-h['entry_mm'][q])*h['axis_unit'][q] for q in range(3)) for h in history for c in h['removed_indices_native']),'max_prep_entry_to_tip_mm':max(x['entry_to_tip_mm'] for x in trace),
   'active_goal_entry_to_tip_mm':sorted(set(round(x['goal_entry_to_tip_mm'],9) for x in trace)),
   'independent_geometry_feasible':True,'post_action48_inventory_observed':False},
 'dependency':{'stalled_states':15,'probes':36,'prerequisite_actions':14,'exact_preparation_geometries_later_executed':6,
   'all_sidecars_complete':True,'all_sidecar_state_source_tool_geometry_fingerprints_joined':True,
   'all_selected_effect_links_match_actual_removed_cells':True,
   'zero_prerequisite_fallback_step':26},
 'omissions':{'states_with_allocation_censoring':48,'total_omitted_cell_mentions':sum(x['obstruction_accounting']['selection_omitted_cells'] for x in states),
   'stalls_with_eligible_active_cells_unselected':sum(bool(x['eligible_omitted']) for x in allocation_witnesses),
   'goal_diagnostic_omissions':0,'sidecar_truncations':0,'omitted_candidates_not_verified_feasible':True},
 'costs':{'total_previews':9856,'arm_previews':r['native_previews_by_arm'],'search_previews':9154,
   'base_previews':base_calls,'added_previews':extra_calls,'goal_previews':goal_calls,'dependency_previews':dep_calls,
   'transitions':52,'page_count':page_count,'legal_candidate_mentions':legal,'worker_seconds':r['wall_seconds'],
   'parent_seconds':receipt['elapsed_seconds'],'rss_sampled_bytes':receipt['sampled_peak_rss_bytes'],
   'output_bytes':receipt['output_bytes'],'cleanup_clean':True},
 'goals':[{'goal_id':v,'tool_id':k[0],'entry_mm':k[1],'tip_mm':k[2]} for k,v in goal_ids.items()],
 'segments':segments,'trace':trace,'prerequisite_actions':prerequisites,
 'exact_preparation_execution_witnesses':exact_execution,'new_layer_witness':layer_witness,
 'stalled_active_allocation':allocation_witnesses,'pins':pins}
output=OUT/'recursive-corridor-saved-audit.json'
output.write_text(json.dumps(summary,sort_keys=True,separators=(',',':'))+'\n')
assert output.stat().st_size<100000
print(json.dumps({k:summary[k] for k in ['status','exploration','dependency','omissions','costs']},indent=2))
print('output',output.stat().st_size,sha(output))
