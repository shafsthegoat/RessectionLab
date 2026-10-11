"""Saved scalar/index/history diagnosis only. No scientific imports or payload reads."""
import collections
import hashlib
import json
import math
from pathlib import Path
ROOT=Path(__file__).resolve().parents[3];STAGE=Path(__file__).resolve().parents[1];OUT=STAGE/'attempt-01';POST=Path(__file__).resolve().parent
PINS={'result.json':'ef64519bd1762ab0867eeb09010e26c28cf8ef01430047ca76b82261e1cd0f1b',
 '../attempt-01.supervision/receipt.json':'e61277083304852b3dea127332fdfebfc4744d11f509fd0c381b971a09443b70'}
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def ref(p):return {'path':p.relative_to(ROOT).as_posix(),'sha256':sha(p),'bytes':p.stat().st_size}
def read(p):return json.loads(p.read_text())
def cells(h):return {tuple(c) for c in h['removed_indices_native']}
def key(h):return h['tool_id'],tuple(h['entry_mm']),tuple(h['tip_mm'])
def physical(h):return {k:h[k] for k in ('tool_id','entry_mm','tip_mm')}
def close(a,b):return math.isclose(a,b,rel_tol=1e-12,abs_tol=1e-10)
def segment(rows):return {'steps':[r['step'] for r in rows],'cells':sum(r['removed_count'] for r in rows),'reward':sum(r['reward'] for r in rows)}
def main():
 for name,digest in PINS.items():assert sha(OUT/name)==digest,name
 result=read(OUT/'result.json');receipt=read(OUT/'../attempt-01.supervision/receipt.json');costs=read(OUT/'costs.json')
 assert receipt['status']=='complete' and receipt['exit_code']==0 and not receipt['cleanup_errors'] and not receipt['final_owned_pids']
 assert all(result['narrow_historical_control'][k] for k in ('exact_history','exact_action_records','exact_selected_records'))
 data={a:read(OUT/a/'selection.json')['accounting'] for a in ('narrow','wide')}
 world=read(OUT/'narrow/original-world-control.json')['world'];center=world['post_exposure']['access']['center_mm'];normal=world['post_exposure']['access']['normal_inward']
 summaries={};sets={};cumulative={};allrows={}
 for arm,a in data.items():
  H=a['explored_history'];S=a['states'];U=set();contacts=set();rows=[];C=[];runs=[];run=[]
  assert len(H)==len(S)==48 and a['complete'] and not a['call_cap_reached'] and not a['time_cap_reached']
  assert a['prefix_verified_count']==4 and a['intervention_applied'] and a['focus_validity_summary']['explicitly_cleared_count']==0
  for h,s in zip(H,S):
   c=cells(h);assert not U&c;U|=c;contacts|={tuple(v) for v in h['contact_indices_native']};C.append(frozenset(U))
   focus=s['obstruction_accounting'].get('goal_focus');goal=s['goals'][0];dep=s.get('dependency_diagnostics')
   if focus:assert not focus['unselected_cells'] and focus['first_failure']['complete_first_failure_set']
   if dep:
    assert not dep['pair_cap_reached']
    assert all(p['reason']=='SHAFT_BLOCKED_BY_REMAINING_NATIVE_TISSUE' and p['sidecar']['complete_first_failure_set'] for p in dep['probes'])
   row={'step':s['depth'],'physical_action':physical(h),'kind':s['selected']['progress_kind'],
     'removed_cells':h['removed_indices_native'],'removed_count':len(c),'reward':h['reward'],
     'cumulative_return':s['selected']['cumulative_return'],'max_removed_native_axis0':max(v[0] for v in U),
     'active_first_blockers':goal['retained_blocker_cells'],'current_direct_legal_preparations':goal['eligible_legal_preparations'],
     'dependency_probes':0 if dep is None else len(dep['probes']),
     'sampled_prerequisites_removed':s['selected']['sampled_prerequisites_removed']}
   rows.append(row)
   if row['kind']=='prerequisite':run.append(row)
   elif run:runs.append(segment(run));run=[]
  if run:runs.append(segment(run))
  affine=H[0]['native_affine'];depths=[sum((sum(affine[i][j]*c[j] for j in range(3))+affine[i][3]-center[i])*normal[i] for i in range(3)) for c in U]
  switches=sum(x['tool_id']!=y['tool_id'] for x,y in zip(H,H[1:]));path=sum(h['complete_tool_path_length_mm'] for h in H)
  normal_volume=sum(h['normal_removed_mm3'] for h in H);target=sum(h['target_removed_mm3'] for h in H);reward=sum(h['reward'] for h in H)
  # Frozen task coefficients: 1 target, .2 normal, .03 each action, .001/mm, .03 switch.
  reconstructed=target-.2*normal_volume-.03*len(H)-.001*path-.03*switches
  assert close(reward,a['explored_total_return']) and close(reward,reconstructed)
  assert target==0 and all(h['reward']<0 for h in H)
  geom=read(OUT/arm/'exploration-geometry.json');episode=read(OUT/arm/'independent-episode.json')
  assert geom['feasible'] and episode['accepted'] and episode['outcomes']['total_reward']==0
  summaries[arm]={'moves':len(H),'selected_actions':read(OUT/arm/'selected-action-plan.json')['actions'],
    'target_removed_mm3':target,'normal_cells':len(U),'normal_removed_mm3':normal_volume,'exploration_return':reward,
    'maximum_removed_cell_center_depth_mm':max(depths),'maximum_removed_axis0_layer':max(c[0] for c in U),
    'depth_scope':'saved native cell centers projected through saved affine/access; no geometry rerun; not distal cell-face depth',
    'first_step_reaching_layer7':next(r['step'] for r in rows if r['max_removed_native_axis0']>=7),
    'layers':dict(sorted(collections.Counter(v[0] for v in U).items())),
    'transverse_columns':len({v[1:] for v in U}),
    'tools':dict(collections.Counter(h['tool_id'] for h in H)),
    'tool_changes':switches,'complete_tool_path_length_mm':path,
    'contact_cells_union':len(contacts),'contact_cell_mentions':sum(len(h['contact_indices_native']) for h in H),
    'direct_blocker':segment([r for r in rows if r['kind']=='direct_blocker']),
    'prerequisite':segment([r for r in rows if r['kind']=='prerequisite']),
    'prerequisite_runs':runs,'goal_diagnostic_previews':a['goal_diagnostic_preview_calls'],
    'dependency_previews':a['dependency_preview_calls'],'diagnostic_previews':a['diagnostic_preview_calls'],
    'focused_states':sum(bool(s['obstruction_accounting'].get('goal_focus')) for s in S),
    'active_first_blockers_omitted':0,'focus_clears':0,'all_probe_reasons':'SHAFT_BLOCKED_BY_REMAINING_NATIVE_TISSUE',
    'final_precommit_root_blockers':S[-1]['goals'][0]['retained_blocker_cells'],
    'last_removed_active_blockers':S[-1]['selected']['current_blockers_removed'],
    'native_previews_all_phases':result['native_previews_by_arm'][arm],
    'search_cost':costs['phases'][arm+'.search'],
    'phase_wall_seconds':{k.split('.',1)[1]:v['complete_wall_seconds'] for k,v in costs['phases'].items() if k.startswith(arm+'.')},
    'certificates':{'exploration':ref(OUT/arm/'exploration-geometry.json'),'episode':ref(OUT/arm/'independent-episode.json')}}
  sets[arm]=U;cumulative[arm]=C;allrows[arm]=rows
 n,w=(data[z]['explored_history'] for z in ('narrow','wide'))
 assert n[:4]==w[:4]
 assert data['narrow']['states'][4]['source_state_hash']==data['wide']['states'][4]['source_state_hash']
 assert data['narrow']['states'][4]['choice_intervention']['query_catalog_digest']==data['wide']['states'][4]['choice_intervention']['query_catalog_digest']
 assert key(w[4])==key(n[34])
 assert cells(w[4])==cells(n[7])|cells(n[28])|cells(n[34])
 assert cells(n[31])==cells(w[31])|cells(w[34])
 assert cumulative['wide'][46]==cumulative['narrow'][47]
 assert cumulative['wide'][36]==cumulative['narrow'][37]
 assert sets['wide']-sets['narrow']==cells(w[47]) and not sets['narrow']-sets['wide']
 for arm in data:
  sums=sum(v['native_preview_entries'] for k,v in costs['phases'].items() if k.startswith(arm+'.'))
  assert sums==result['native_previews_by_arm'][arm]
 ns,ws=(summaries[z] for z in ('narrow','wide'));delta=ws['exploration_return']-ns['exploration_return']
 terms={'normal_removal':-.2*(ws['normal_removed_mm3']-ns['normal_removed_mm3']),
  'tool_changes':-.03*(ws['tool_changes']-ns['tool_changes']),
  'motion':-.001*(ws['complete_tool_path_length_mm']-ns['complete_tool_path_length_mm']),
  'action_count':-.03*(len(w)-len(n))}
 assert close(delta,sum(terms.values()))
 out={'status':'saved_history_diagnosis_passed','scope':'saved JSON/source only; no arrays, model, native geometry or new experiment',
  'inputs':{k:ref(OUT/k) for k in PINS},
  'bulk_trace_pointers':{a:ref(OUT/a/'selection.json') for a in data},
  'complete_negative':True,'narrow_exact_historical_control':result['narrow_historical_control'],
  'common_state':{'first4_native_histories_exact':True,'state':data['narrow']['states'][4]['source_state_hash'],
   'catalog_digest':data['narrow']['states'][4]['choice_intervention']['query_catalog_digest'],
   'treatments':{a:data[a]['states'][4]['choice_intervention'] for a in data}},
  'arms':summaries,'compact_trace_rows':allrows,
  'causal_effect':{'wide5_cells_equal_narrow_steps':[8,29,35],
   'wide5_cells':sorted(cells(w[4])),'same_wide_physical_ray_was_narrow_step':35,
   'narrow32_four_cell_cut_split_into_wide_steps':[32,35],
   'same_root_blocker_set_at_step15':data['narrow']['states'][14]['goals'][0]['retained_blocker_cells']==data['wide']['states'][14]['goals'][0]['retained_blocker_cells'],
   'step15_legal_preparations':{a:data[a]['states'][14]['goals'][0]['eligible_legal_preparations'] for a in data},
   'earlier_wide_legal_rays_step15_16_equal_narrow_steps':[30,31],
   'equal_removed_set_pairs':[[i+1,j+1] for i,x in enumerate(cumulative['narrow']) for j,y in enumerate(cumulative['wide']) if x==y],
   'equal_removed_sets_are_not_equal_history_state_hashes':True,
   'wide47_vs_narrow48_return_difference':data['wide']['states'][46]['selected']['cumulative_return']-ns['exploration_return'],
   'wide48_added_cells':sorted(cells(w[47])),'wide48_reward':w[47]['reward'],
   'wide_minus_narrow_final_return':delta,'return_difference_terms':terms,
   'final_common_unremoved_observed_blockers':[[7,90,84],[7,90,85],[7,90,86]],
   'terminal_diagnostic_limitation':'No post-horizon goal preview; known retained occupied blockers persist because none were removed, but no complete final first-failure set or next-action feasibility is claimed.'},
  'costs':{'parent_seconds':receipt['elapsed_seconds'],'peak_owned_RSS_bytes':receipt['sampled_peak_rss_bytes'],
   'output_bytes':receipt['output_bytes'],'native_previews_total':result['native_budget']['native_preview_entries'],
   'native_transitions_total':sum(sum(v['native_transitions'].values()) for v in result['arms'].values()),
   'selected_scope':'two STOP rollouts plus two STOP fresh replays;96 exploratory transitions additionally checked geometrically',
   'timing_limitation':'One fixed-order pair; phase timers are inclusive, native timers nested; no repeated-run speed inference.'},
  'limitations':['One simulator-trace treatment on one annotation-assisted anatomy, no general wide-tool policy conclusion.',
   'Equal removed masks do not erase different contact/history/tool paths. Contact remains an unassessed upper bound, not injury.',
   'No target or positive prefix; both accepted selected plans are STOP0. No clinical, operative or global reachability claim.',
   'Neither policy inspected remaining geometry after action48; no claim that another step or increased horizon would reach target.']}
 p=POST/'diagnosis.json';p.write_text(json.dumps(out,separators=(',',':'),sort_keys=True)+'\n')
 print(json.dumps({'diagnosis':ref(p),'arms':{a:{k:v for k,v in r.items() if k in ('normal_cells','exploration_return','maximum_removed_cell_center_depth_mm','first_step_reaching_layer7','tool_changes','complete_tool_path_length_mm','dependency_previews')} for a,r in summaries.items()},'return_delta':delta,'return_terms':terms}))
if __name__=='__main__':main()
