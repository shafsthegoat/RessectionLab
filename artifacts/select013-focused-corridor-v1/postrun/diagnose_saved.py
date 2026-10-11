"""Compact saved-scalar diagnosis. No native, array, model or patient payload imports."""
import collections,hashlib,json,math
from pathlib import Path
ROOT=Path(__file__).resolve().parents[3];BASE=ROOT/'build/select013-focused-corridor-v1';OUT=BASE/'postrun';refs={}
def load(rel,expected=None):
 p=ROOT/rel;b=p.read_bytes();h=hashlib.sha256(b).hexdigest()
 if expected:assert h==expected,(rel,h)
 refs[rel]={'sha256':h,'bytes':len(b)};return json.loads(b)
r=load('build/select013-focused-corridor-v1/attempt-01/result.json','046e9da6086020475e57b7eb79a3cc793d12283c77c1414ff01a98657cd7295d')
p=load('build/select013-focused-corridor-v1/attempt-01.supervision/receipt.json','db4f92d736fc57f7b5176988f51e5928b6eac72d2ed0ce22dfa51977db1cfed9')
trace_rel='build/select013-focused-corridor-v1/attempt-01/focused_corridor/selection.json'
a=load(trace_rel)['accounting']
w=load('build/select013-focused-corridor-v1/attempt-01/focused_corridor/original-world-control.json')['world']
old=load('build/select013-recursive-corridor-v1/attempt-01/recursive_corridor/selection.json')['accounting']
prior=load('build/select013-recursive-corridor-v1/postrun/diagnosis.json','d6595218ce3859c9303d12a441ebd6dc42e0c77f59ddba0cdbc8da91946006ce')
key=lambda g:(g['tool_id'],tuple(g['entry_mm']),tuple(g['tip_mm']))
cells=lambda x:{tuple(c) for c in x}
A=a['explored_history'][0]['native_affine'];n=w['post_exposure']['access']['normal_inward'];m=math.sqrt(sum(v*v for v in n));n=[v/m for v in n];center=w['post_exposure']['access']['center_mm']
def depth(c):return sum((sum(A[i][j]*c[j] for j in range(3))+A[i][3]-center[i])*n[i] for i in range(3))
rows=[];triggers=[];union=set();bykind={};cum=0.;focus_count=0;focus_omissions=0;probe_reasons=collections.Counter();goals=set();probe_pairs=set()
for i,(s,h) in enumerate(zip(a['states'],a['explored_history'])):
 selected=s['selected'];k=key(selected['goal']);goals.add(k);g=next(g for g in s['goals'] if key(g)==k);rm=cells(h['removed_indices_native']);bc=cells(g['retained_blocker_cells']);assert not(rm&union);union|=rm;cum+=h['reward'];assert rm==cells(selected['actual_removed_cells'])
 assert len(rm&bc)==selected['current_blockers_removed']
 f=s['obstruction_accounting'].get('goal_focus');focus=None
 if f:
  focus_count+=1;focus_omissions+=len(f['unselected_cells']);assert key(f['record'])==k
  assert cells(f['retained_cells'])==bc and f['record']['source_state_hash']==s['source_state_hash']
  assert cells(f['selected_cells'])==bc==cells(f['aperture_eligible_cells'])
  assert f['first_failure']['complete_first_failure_set']
  focus={'retained':len(bc),'selected':len(f['selected_cells']),'omitted':len(f['unselected_cells']),'aperture_eligible':len(f['aperture_eligible_cells'])}
 kind=selected['progress_kind'];z=bykind.setdefault(kind,{'actions':0,'cells':0,'reward':0.,'active_first_blockers_removed':0});z['actions']+=1;z['cells']+=len(rm);z['reward']+=h['reward'];z['active_first_blockers_removed']+=selected['current_blockers_removed']
 row={'step':i+1,'kind':kind,'tool':h['tool_id'],'removed_cells':len(rm),'removed_layers':sorted({c[0] for c in rm}),'current_blockers_removed':selected['current_blockers_removed'],'first_blockers':len(bc),'first_blocker_layers':sorted({c[0] for c in bc}),'legal_direct_preparations':g['eligible_legal_preparations'],'focus':focus,'max_removed_center_depth_mm':max(map(depth,union)),'cumulative_reward':cum};rows.append(row)
 d=s.get('dependency_diagnostics')
 if d:
  assert key(d['active_goal'])==k and d['selected_effect'] is not None
  assert d['source_state_hash']==s['source_state_hash'] and cells(d['active_retained_blockers'])==bc
  assert len(d['probes'])==2*len(bc) and not d['pair_cap_reached']
  for q in d['probes']:
   probe_reasons[q['reason']]+=1;probe_pairs.add(key(q));assert q['sidecar']['complete_first_failure_set'] and not q['sidecar']['truncated']
  effect=d['selected_effect'];assert cells(effect['actual_removed_cells'])==rm
  triggers.append({'step':i+1,'active_first_blockers':len(bc),'allocated_blocker_cells':focus['selected'],'probes':len(d['probes']),'prerequisite_union_cells':len(cells(d['retained_prerequisite_union'])),'removed_prerequisites':len(cells(effect['sampled_prerequisite_cells_removed'])),'actual_removed_cells':len(rm),'direct_first_blockers_removed':len(rm&bc),'complete_first_sets':True,'pair_limit_reached':False})
chains=[];i=0
while i<len(rows):
 if rows[i]['kind']!='prerequisite':i+=1;continue
 start=i
 while i<len(rows) and rows[i]['kind']=='prerequisite':i+=1
 nxt=rows[i] if i<len(rows) else None
 chains.append({'start':start+1,'end':i,'actions':i-start,'cells':sum(x['removed_cells'] for x in rows[start:i]),'reward':sum(a['explored_history'][j]['reward'] for j in range(start,i)),'next_direct_step':None if nxt is None else nxt['step'],'same_root_first_set_through_chain':all(a['states'][j]['dependency_diagnostics']['active_retained_blockers']==a['states'][start]['dependency_diagnostics']['active_retained_blockers'] for j in range(start,i)),'next_legal_direct_preparations':None if nxt is None else nxt['legal_direct_preparations']})
ray_unlocks=[];seen=set()
for i,s in enumerate(a['states']):
 d=s.get('dependency_diagnostics')
 if not d:continue
 for q in d['probes']:
  k=key(q)
  if k in seen:continue
  later=[j for j in range(i+1,len(a['explored_history'])) if key(a['explored_history'][j])==k]
  if later:
   seen.add(k);ray_unlocks.append({'first_rejection':i+1,'later_exact_geometry_execution':later[0]+1,'tool':k[0],'first_sidecar_fingerprint':q['sidecar']['fingerprint']})
assert len(rows)==48 and len(union)==83 and focus_count==47 and focus_omissions==0
assert len(triggers)==23 and sum(t['probes'] for t in triggers)==a['dependency_preview_calls']==226
assert all(h['target_removed_mm3']==0 and h['reward']<0 for h in a['explored_history'])
assert not p['cleanup_errors'] and not p['final_owned_pids'] and p['exit_code']==0
assert r['focus_validity']['explicitly_cleared_count']==0 and r['focused_plan_context_replayed']
assert all(x['actions']==['STOP'] and x['independent_accepted'] for x in r['arms'].values())
assert all(key(a['explored_history'][j])==key(old['explored_history'][j]) and a['explored_history'][j]['removed_indices_native']==old['explored_history'][j]['removed_indices_native'] for j in range(4))
assert a['states'][4]['source_state_hash']==old['states'][4]['source_state_hash']
common={'first_different_action':5,'four_prior_physical_actions_and_removals_equal':True,'exact_native_state_hash':a['states'][4]['source_state_hash'],'focused':{k:a['states'][4]['selected'][k] for k in ('current_blockers_removed','nominal_reward')},'prior_recursive':{k:old['states'][4]['selected'][k] for k in ('current_blockers_removed','nominal_reward')}}
common['focused']['cells']=len(a['states'][4]['selected']['actual_removed_cells']);common['prior_recursive']['cells']=len(old['states'][4]['selected']['actual_removed_cells'])
oldkey=key(old['explored_history'][4]);common['old_physical_ray_executed_later_in_focused']=[j+1 for j,h in enumerate(a['explored_history']) if key(h)==oldkey]
common['focused']['physical_action']=a['states'][4]['selected']['action_geometry']
common['prior_recursive']['physical_action']=old['states'][4]['selected']['action_geometry']
common['deferred_step35']={'same_exact_physical_ray':key(a['explored_history'][34])==oldkey,'cells':len(a['explored_history'][34]['removed_indices_native']),'reward':a['explored_history'][34]['reward'],'sampled_prerequisites_removed':a['states'][34]['selected']['sampled_prerequisites_removed']}
common['intervening_steps5_through34']={'moves':30,'prerequisite_moves':sum(s['selected']['progress_kind']=='prerequisite' for s in a['states'][4:34]),'cells':sum(len(h['removed_indices_native']) for h in a['explored_history'][4:34]),'reward':sum(h['reward'] for h in a['explored_history'][4:34]),'scope':'descriptive interval, not attributable incremental cost of choosing the narrow action'}
common['scope']='same-world native state and observed feasible old choice; current full legal geometry/rank list is not saved, so no complete counterfactual-route conclusion'
report={'schema':'saved-focused-corridor-diagnosis-v1','scope':'saved JSON scalar/source reasoning only; no acquired data or numerical/native/model execution','inputs':refs,'complete_negative':True,'selected_STOP0_both':True,'exploration':{'actions':48,'target_cells':0,'normal_cells':83,'return':cum,'max_removed_center_depth_mm':max(map(depth,union)),'max_native_layer':max(c[0] for c in union),'removed_transverse_columns':len({c[1:] for c in union}),'goal_columns':len({k[1] for k in goals}),'physical_goal_keys':len(goals),'tool_counts':dict(collections.Counter(h['tool_id'] for h in a['explored_history'])),'progress_kind':bykind},'focused_coverage':{'focused_states':focus_count,'unfocused_initial_states':1,'omitted_active_blocker_mentions':focus_omissions,'active_blockers_static_aperture_excluded':0,'focus_clear_events':0,'all_current_active_first_blockers_allocated':True},'dependency':{'trigger_states':len(triggers),'probes':226,'distinct_preparation_geometries_probed':len(probe_pairs),'rejection_reasons':dict(probe_reasons),'all_current_blocker_tool_pairs_probed_at_stalls':True,'probe_limit_reached':False,'exact_previously_rejected_rays_later_executed':ray_unlocks},'steps':rows,'triggers':triggers,'chains':chains,'common_state_first_divergence':common,'prior_recursive':prior['exploration'],'costs':{'parent_seconds':p['elapsed_seconds'],'selector_seconds':a['wall_seconds'],'peak_rss_bytes':p['sampled_peak_rss_bytes'],'output_bytes':p['output_bytes'],'native_previews_total':r['native_budget']['native_preview_entries'],'goal_diagnostics':a['goal_diagnostic_preview_calls'],'dependency_diagnostics':226,'model_forwards':r['model_forwards'],'updates':r['updates']},'next_evidence_based_question':'Does bounded lookahead over already legal preparations reduce the full dependency bundle cost? Common state5 supplies an observed narrow1-cell versus wide7-cell alternative with equal current blocker overlap. Test downstream exact-ray availability and cumulative reward, not immediate overlap or total removed volume alone. This evidence does not yet prove a positive48-step route or justify a blind changed ranking.', 'limitations':['Allocation omissions are eliminated only for the active exact first-failure set; subsequent depth and other goal coverage are different questions.','All current first blockers can be allocated while their preparation rays are shaft-blocked by shallower cells; clearing a first set can reveal another.','No post-action48 refreshed inventory; final4-blocker diagnostic precedes the last1-cell cut.','Current full legal-candidate geometry/reward list was not saved; policy counterfactual completeness cannot be reconstructed here.','No clinical/surgical or global reachability claim, no matched-runtime comparison with earlier runs.','One exploratory console projection assumed goal_focus exists in initial unfocused accounting; KeyError only, no saved output/experiment mutation.']}
(OUT/'diagnosis.json').write_text(json.dumps(report,indent=2)+'\n')
print(json.dumps({k:report[k] for k in ('exploration','focused_coverage','dependency','chains','common_state_first_divergence','costs')},indent=2))
