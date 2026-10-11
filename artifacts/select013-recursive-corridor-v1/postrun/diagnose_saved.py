"""Saved JSON scalar diagnosis only; no acquired arrays, model or geometry execution."""
import collections,hashlib,json,math
from pathlib import Path
ROOT=Path(__file__).resolve().parents[3]
BASE=ROOT/'build/select013-recursive-corridor-v1'
OUT=BASE/'postrun'
refs={}
def load(rel,expected=None):
 p=ROOT/rel;b=p.read_bytes();h=hashlib.sha256(b).hexdigest()
 if expected:assert h==expected,(rel,h)
 refs[rel]={'sha256':h,'bytes':len(b)}
 return json.loads(b)
r=load('build/select013-recursive-corridor-v1/attempt-01/result.json','fa418501772054f61f26cc5ab410cf1f05c90a7126be429e09787916816fc411')
receipt=load('build/select013-recursive-corridor-v1/attempt-01.supervision/receipt.json','013a525881729be2fdf7d1713ebe3aa3bb2b5c03aafd29b39930597d48fa368b')
trace_rel='build/select013-recursive-corridor-v1/attempt-01/recursive_corridor/selection.json'
a=load(trace_rel)['accounting']
w=load('build/select013-recursive-corridor-v1/attempt-01/recursive_corridor/original-world-control.json')['world']
old=load('build/select013-complete-corridor-v1/postrun/diagnosis.json','9754a3b395ab23c55d9a92574c812d1eb43ea0bbdf56cde18f3623a187d48019')
key=lambda g:(g['tool_id'],tuple(g['entry_mm']),tuple(g['tip_mm']))
cells=lambda x:{tuple(c) for c in x}
def goal(s,k):return next(g for g in s['goals'] if key(g)==k)
def allocation(s,g):
 bc=cells(g['retained_blocker_cells']); tool=0 if g['tool_id']=='generic_suction' else 1
 groups=[x for x in s['obstruction_accounting']['target_ray_groups'] if x['tool_order']==tool and bc<=cells(x['retained_co_blockers'])]
 groups=[q for q in groups if q['column_index']==goal_columns[key(g)]]
 assert len(groups)==1
 q=groups[0];sel=bc&cells(q['selected_co_blockers']);elig=bc&cells(q['aperture_eligible_co_blockers'])
 return {'first_blockers':len(bc),'blocker_layers':sorted({c[0] for c in bc}),'selected_blockers':len(sel),'allocation_omitted_blockers':len(elig-sel),'aperture_eligible_blockers':len(elig),'column_index':q['column_index']}
ids={}
for s in a['states']:
 k=key(s['selected']['goal'])
 if k not in ids:ids[k]=len(ids)+1
# Resolve stable physical-goal columns across all saved states. Single-cell
# blocker subsets may be shared by several groups in an isolated state.
goal_columns={}
for k in ids:
 candidates=None
 for state in a['states']:
  g=goal(state,k);bc=cells(g['retained_blocker_cells']);tool=0 if g['tool_id']=='generic_suction' else 1
  choices={z['column_index'] for z in state['obstruction_accounting']['target_ray_groups'] if z['tool_order']==tool and bc<=cells(z['retained_co_blockers'])}
  candidates=choices if candidates is None else candidates&choices
 assert len(candidates)==1
 goal_columns[k]=next(iter(candidates))
A=a['explored_history'][0]['native_affine'];n=w['post_exposure']['access']['normal_inward'];m=math.sqrt(sum(v*v for v in n));n=[v/m for v in n];center=w['post_exposure']['access']['center_mm']
def depth(c):return sum((sum(A[i][j]*c[j] for j in range(3))+A[i][3]-center[i])*n[i] for i in range(3))
rows=[];triggers=[];switches=[];union=set();previous=None;cum=0.;bykind={}
for i,(s,h) in enumerate(zip(a['states'],a['explored_history'])):
 sel=s['selected'];k=key(sel['goal']);g=goal(s,k);rm=cells(h['removed_indices_native']);assert not (rm&union);union|=rm;cum+=h['reward']
 assert rm==cells(sel['actual_removed_cells'])
 assert len(rm&cells(g['retained_blocker_cells']))==sel['current_blockers_removed']
 kind=sel['progress_kind'];z=bykind.setdefault(kind,{'actions':0,'cells':0,'reward':0.,'current_blockers_removed':0});z['actions']+=1;z['cells']+=len(rm);z['reward']+=h['reward'];z['current_blockers_removed']+=sel['current_blockers_removed']
 row={'step':i+1,'selected_goal':ids[k],'progress_kind':kind,'tool':h['tool_id'],'removed_cells':len(rm),'removed_layers':sorted({c[0] for c in rm}),'removed_current_blockers':sel['current_blockers_removed'],'max_removed_center_depth_mm':max(map(depth,union)),'cumulative_reward':cum,'legal_direct_preparations':g['eligible_legal_preparations'],**allocation(s,g)}
 rows.append(row)
 d=s.get('dependency_diagnostics')
 if d:
  dg=goal(s,key(d['active_goal']));probes=d['probes'];effect=d['selected_effect']
  assert d['source_state_hash']==s['source_state_hash']
  for p in probes:
   assert p['sidecar']['source_state_hash']==s['source_state_hash']
   assert p['sidecar']['complete_first_failure_set'] and not p['sidecar']['truncated']
  t={'at_step':i+1,'active_goal':ids[key(d['active_goal'])],**allocation(s,dg),'emitted_centered_pairs':d['emitted_centered_pair_count'],'probes':len(probes),'pair_cap_reached':d['pair_cap_reached'],'prerequisite_union_cells':len(cells(d['retained_prerequisite_union'])),'prerequisite_selected':effect is not None,'probe_reasons':dict(collections.Counter(p['reason'] for p in probes)),'probe_rows':[{'tool':p['tool_id'],'blocker_layer':p['centered_active_blocker'][0],'prerequisite_count':len(p['retained_prerequisite_cells']),'prerequisite_layers':sorted({c[0] for c in p['retained_prerequisite_cells']}),'catalog_action_id':p['current_catalog_action_id'],'sidecar_fingerprint':p['sidecar']['fingerprint']} for p in probes]}
  if effect:
   assert cells(effect['actual_removed_cells'])==rm
   t['prerequisites_removed']=len(cells(effect['sampled_prerequisite_cells_removed']))
  triggers.append(t)
 if previous is not None and previous!=k:
  prior=goal(s,previous)
  switches.append({'at_step':i+1,'from_goal':ids[previous],'to_goal':ids[k],**allocation(s,prior),'direct_legal_preparations':prior['eligible_legal_preparations'],'dependency_selected':bool(d and d['selected_effect']),'reason':'no legal direct or sampled-prerequisite-removing action for prior active goal; switch within same transverse corridor'})
 previous=k
chains=[];i=0
while i<len(rows):
 if rows[i]['progress_kind']!='prerequisite':i+=1;continue
 start=i
 while i<len(rows) and rows[i]['progress_kind']=='prerequisite':i+=1
 chain=rows[start:i]
 after=rows[i] if i<len(rows) else None
 chains.append({'start_step':start+1,'end_step':i,'actions':len(chain),'removed_cells':sum(x['removed_cells'] for x in chain),'reward':sum(a['explored_history'][j]['reward'] for j in range(start,i)),'active_goal':chain[0]['selected_goal'],'next_step':None if after is None else after['step'],'same_goal_direct_progress_restored':bool(after and after['selected_goal']==chain[0]['selected_goal'] and after['progress_kind']=='direct_blocker'),'next_legal_direct_preparations':None if after is None else after['legal_direct_preparations'],'exact_first_blocker_set_unchanged_through_preparation':all(goal(a['states'][start],key(a['states'][start]['selected']['goal']))['retained_blocker_cells']==goal(a['states'][j],key(a['states'][start]['selected']['goal']))['retained_blocker_cells'] for j in range(start,min(i+1,len(rows))))})
ray_unlocks=[]
seen=set()
for i,state in enumerate(a['states']):
 d=state.get('dependency_diagnostics')
 if not d:continue
 for probe in d['probes']:
  pk=key(probe)
  if pk in seen:continue
  later=[j for j in range(i+1,len(a['explored_history'])) if key(a['explored_history'][j])==pk]
  if later:
   j=later[0];seen.add(pk)
   ray_unlocks.append({'first_rejected_step':i+1,'exact_same_geometry_committed_step':j+1,'tool':pk[0],'first_probe_fingerprint':probe['sidecar']['fingerprint'],'eventual_removed_cells':len(a['explored_history'][j]['removed_indices_native']),'scope':'exact tool/entry/tip match; may require multiple intervening removals'})
assert len(rows)==48 and len(union)==77 and len(triggers)==15
assert sum(t['probes'] for t in triggers)==a['dependency_preview_calls']==36
assert sum(t['prerequisite_selected'] for t in triggers)==14
assert all(h['target_removed_mm3']==0 for h in a['explored_history'])
assert all(g['reason']=='SHAFT_BLOCKED_BY_REMAINING_NATIVE_TISSUE' and g['first_failure_set_complete'] for s in a['states'] for g in s['goals'])
assert receipt['exit_code']==0 and not receipt['cleanup_errors'] and not receipt['final_owned_pids']
assert r['all_action_plans_sealed_before_evaluation'] and r['exploration_geometry_feasible']
assert all(v['actions']==['STOP'] and v['independent_accepted'] for v in r['arms'].values())
report={'schema':'saved-recursive-corridor-diagnosis-v1','inputs':refs,'scope':'saved scalar projection; no images, native geometry, models or fit executed','complete_run_negative':True,'both_selected_STOP':True,'exploration':{'actions':len(rows),'target_cells':0,'normal_cells':len(union),'return':cum,'normal_removed_mm3':sum(h['normal_removed_mm3'] for h in a['explored_history']),'max_removed_center_depth_mm':max(map(depth,union)),'max_native_layer':max(c[0] for c in union),'removed_transverse_columns':len({c[1:] for c in union}),'goal_transverse_columns':len({k[1] for k in ids}),'goal_tool_keys':len(ids),'goal_switches':len(switches),'cut_tool_counts':dict(collections.Counter(h['tool_id'] for h in a['explored_history'])),'progress_kind_totals':bykind,'goal_preview_calls':a['goal_diagnostic_preview_calls'],'dependency_preview_calls':a['dependency_preview_calls'],'total_native_preview_calls':r['native_budget']['native_preview_entries']},'allocation':{'selected_goal_states_with_omitted_aperture_eligible_blockers':sum(x['allocation_omitted_blockers']>0 for x in rows),'selected_goal_omitted_count_range':[min(x['allocation_omitted_blockers'] for x in rows),max(x['allocation_omitted_blockers'] for x in rows)],'trigger_states_with_omissions':sum(t['allocation_omitted_blockers']>0 for t in triggers),'trigger_selected_blocker_range':[min(t['selected_blockers'] for t in triggers),max(t['selected_blockers'] for t in triggers)],'trigger_omitted_blocker_range':[min(t['allocation_omitted_blockers'] for t in triggers),max(t['allocation_omitted_blockers'] for t in triggers)],'trigger_probe_count_range':[min(t['probes'] for t in triggers),max(t['probes'] for t in triggers)],'triggers_at32pair_limit':sum(t['pair_cap_reached'] for t in triggers),'static_aperture_excludes_active_first_blockers_at_triggers':sum(t['first_blockers']-t['aperture_eligible_blockers'] for t in triggers),'count_scope':'per-state incidences, not distinct omitted cells; omission does not establish feasibility'},'steps':rows,'dependency_triggers':triggers,'exact_rejected_rays_later_committed':ray_unlocks,'prerequisite_chains':chains,'switches':switches,'historical_v1':old['exploration'],'observed_comparison':{'focus_restored':True,'deeper_removed_frontier':False,'extra_normal_cells':77-old['exploration']['removed_cells'],'return_delta':cum-old['exploration']['return'],'matched_timing':False},'costs':{'parent_seconds':receipt['elapsed_seconds'],'peak_rss_bytes':receipt['sampled_peak_rss_bytes'],'output_bytes':receipt['output_bytes'],'selector_seconds':a['wall_seconds'],'fits':r['optimizer_attempts'],'model_forwards':r['model_forwards']},'next_targeted_change':'Before certifying obstruction extras, allocate unchanged bounded slots to the active physical goal current authenticated first blockers, then sampled prerequisites if room; refresh after each real commit, preserve native rules/reward/h48/32-extra budget and current catalog binding. Omitted alternative legality remains to be measured.','reader_note':'Initial scalar reader required unique blocker-subset group in every state; states4/35 shared one-cell subsets with other columns. Corrected by intersecting group-column candidates across all saved states for each identical physical goal. No experiment changed. A separate console inspection attempted to subscript legal_actions (an integer count); no output was written.', 'limitations':['This is modeled annotation-assisted geometry, not surgical or clinical safety.','Six exact rejected tool/entry/tip rays later execute; intervening removals can be multiple, so individual cuts are not uniquely causal. Other restored legal actions may include regenerated base geometry.','The omitted active blockers were not queried; no claim that active-goal allocation must succeed.','No post-action48 inventory exists; final blocker diagnostics are pre-action48.','No proof that horizon48 is sufficient or insufficient globally.','Full reproduction requires locally retained hash-pinned selection.json and historical referenced evidence; bulk trace intentionally excluded from archive.']}
(OUT/'diagnosis.json').write_text(json.dumps(report,indent=2)+'\n')
print(json.dumps({k:report[k] for k in ('exploration','allocation','prerequisite_chains','switches','observed_comparison','costs')},indent=2))
