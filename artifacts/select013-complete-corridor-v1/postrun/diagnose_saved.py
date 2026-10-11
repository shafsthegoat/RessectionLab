"""Bounded stdlib scalar projection of completed JSON; no arrays or native imports."""
import collections,hashlib,json,math
from pathlib import Path
ROOT=Path(__file__).resolve().parents[3];P=ROOT/'build/select013-complete-corridor-v1';OUT=P/'postrun'
refs={}
def load(rel,expected=None):
 p=ROOT/rel;b=p.read_bytes();h=hashlib.sha256(b).hexdigest()
 if expected is not None:assert h==expected,(rel,h)
 refs[rel]={'sha256':h,'bytes':len(b)};return json.loads(b)
r=load('build/select013-complete-corridor-v1/attempt-01/result.json','6aecbed77aa0235abaa03147d2cd2769cdab712dba1b1bcbe69449b4a725cd84')
p=load('build/select013-complete-corridor-v1/attempt-01.supervision/receipt.json','e69d62cb95c92f59140f71cfcec8b9df3969dda6cacb9213cc31ab73b3876caf')
a=load('build/select013-complete-corridor-v1/attempt-01/persistent_corridor/selection.json')['accounting']
w=load('build/select013-complete-corridor-v1/attempt-01/persistent_corridor/original-world-control.json')['world']
old=load('build/select013-paged-route-h48-v1/attempt-01/result.json','8dcbf51e12fc4a1175f52422bd5e4b745ce6d35300fd589b8b4b23404d7bdcf2')
oldfocus=load('build/select013-target-ray-priority-v1/horizon-cutting-scalars.json')
key=lambda g:(g['tool_id'],tuple(g['entry_mm']),tuple(g['tip_mm']))
def cells(x):return {tuple(c) for c in x}
def goal(state,k):return next(g for g in state['goals'] if key(g)==k)
def group(state,g):
 c=cells(g['retained_blocker_cells']);tool=0 if g['tool_id']=='generic_suction' else 1
 matching=[z for z in state['obstruction_accounting']['target_ray_groups'] if z['tool_order']==tool and c<=cells(z['retained_co_blockers'])]
 assert len(matching)==1
 return matching[0]
ids={};corridor_keys=[]
for s in a['states']:
 k=key(s['selected']['goal'])
 if k not in ids:ids[k]=len(ids)+1;corridor_keys.append(k)
A=a['explored_history'][0]['native_affine'];access=w['post_exposure']['access'];n=access['normal_inward'];norm=math.sqrt(sum(v*v for v in n));n=[v/norm for v in n];center=access['center_mm']
def depth(c):return sum((sum(A[i][j]*c[j] for j in range(3))+A[i][3]-center[i])*n[i] for i in range(3))
rows=[];switches=[];unlocks=[];union=set();previous=None
for i,(s,h) in enumerate(zip(a['states'],a['explored_history'])):
 k=key(s['selected']['goal']);g=goal(s,k);rm=cells(h['removed_indices_native']);union|=rm
 bc=cells(g['retained_blocker_cells']);assert len(rm&bc)==s['selected']['current_blockers_removed']
 rows.append({'step':i+1,'goal':ids[k],'cut_tool':h['tool_id'],'removed_cells':len(rm),
  'removed_current_blockers':len(rm&bc),'first_blockers':len(bc),'first_blocker_layers':sorted({c[0] for c in bc}),
  'whole_run_max_removed_center_depth_mm':max(map(depth,union)),'cumulative_reward':sum(x['reward'] for x in a['explored_history'][:i+1]),
  'legal_direct_blocker_removing_actions':g['eligible_legal_preparations']})
 if previous is not None and previous!=k:
  prior=goal(s,previous);gr=group(s,prior);oldbc=cells(prior['retained_blocker_cells'])
  assert prior['eligible_legal_preparations']==0 and not prior['direct_legal_goal_available']
  switches.append({'at_step':i+1,'from_goal':ids[previous],'to_goal':ids[k],'first_blocker_count':len(oldbc),
   'selected_first_blockers':len(oldbc&cells(gr['selected_co_blockers'])),
   'aperture_eligible_first_blockers':len(oldbc&cells(gr['aperture_eligible_co_blockers'])),
   'first_blocker_layers':sorted({c[0] for c in oldbc}),
   'reason':'zero current legal action removing first blockers; exact goal still shaft-blocked'})
 previous=k
 if i+1<len(a['states']):
  nxt=a['states'][i+1]
  for k2 in corridor_keys:
   before=goal(s,k2);after=goal(nxt,k2)
   if before['eligible_legal_preparations']==0 and after['eligible_legal_preparations']>0:
    gb,ga=group(s,before),group(nxt,after)
    sameallocation=all(gb[f]==ga[f] for f in ('selected_co_blockers','unselected_co_blockers','aperture_eligible_co_blockers'))
    unlocks.append({'after_action':i+1,'goal':ids[k2],'preparation_goal':ids[k],
     'removed_first_blocker_overlap':len(rm&cells(before['retained_blocker_cells'])),
     'exact_first_blocker_set_unchanged':before['retained_blocker_cells']==after['retained_blocker_cells'],
     'exact_group_selection_and_aperture_eligibility_unchanged':sameallocation,
     'legal_direct_blocker_actions_before':0,'legal_direct_blocker_actions_after':after['eligible_legal_preparations'],
     'evidence_scope':'observed side-cell preparation precedes newly available legal removal; exact rejected extra-ray sidecars were not saved'})
assert len(rows)==48 and len(union)==74 and all(h['target_removed_mm3']==0 for h in a['explored_history'])
assert all(g['reason']=='SHAFT_BLOCKED_BY_REMAINING_NATIVE_TISSUE' and g['first_failure_set_complete'] for s in a['states'] for g in s['goals'])
historical={}
for name,arm in old['arms'].items():
 openings=[z for l in arm['selector_accounting']['layers'] for z in l.get('opening_depth_retention',[])]
 peak=max(openings,key=lambda z:z['opening_depth_mm'])
 historical[name]={'max_saved_opening_depth_mm':peak['opening_depth_mm'],'prefix_length_at_first_max_depth':len(peak['actions']),
  'first_max_depth_prefix_return':peak['estimated_incremental_return'],'first_max_depth_prefix_removed_mm3':peak['opening_removed_volume_mm3'],
  'comparison_scope':'retained prefix diagnostic, not accepted positive route or matched runtime'}
report={'schema':'saved-corridor-causal-diagnosis-v1','inputs':refs,'complete_run_negative':True,
 'selected_outcomes':{k:x['outcomes'] for k,x in r['arms'].items()},
 'exploration':{'moves':48,'removed_cells':len(union),'distinct_transverse_removed_columns':len({c[1:] for c in union}),
  'axial_layers':dict(collections.Counter(c[0] for c in union)),'max_removed_center_depth_mm':max(map(depth,union)),
  'target_removed_mm3':0.,'normal_removed_mm3':sum(h['normal_removed_mm3'] for h in a['explored_history']),
  'return':sum(h['reward'] for h in a['explored_history']),
  'physical_tool_goal_keys':len(ids),'transverse_goal_columns':len({k[1] for k in ids}),
  'cut_tool_counts':dict(collections.Counter(h['tool_id'] for h in a['explored_history'])),
  'goal_diagnostics':sum(len(s['goals']) for s in a['states']),'diagnostic_truncations':0,
  'goal_diagnostic_cap_omissions':sum(s['goals_omitted_by_diagnostic_cap'] for s in a['states']),
  'global16_cell_selection_omission_range':[min(s['obstruction_accounting']['selection_omitted_cells'] for s in a['states']),max(s['obstruction_accounting']['selection_omitted_cells'] for s in a['states'])]},
 'steps':rows,'switches':switches,'observed_prerequisite_unlocks':unlocks,'historical_depth':historical,
 'next_single_change':'When active goal has no direct legal progress, perform bounded depth2 exact current emitted preparation-ray prerequisite diagnostics; choose legal action removing its current blockers, refresh after real commit. No changed catalog, reward, horizon or model. All additional queries and preparation costs charged.',
 'limitations':['No rejected extra-candidate sidecars or complete legal candidate geometry were saved; causal ray-specific blocker chain cannot be reconstructed here.','Global16-cell allocation omits active goal blockers and remains a confound.','Static necessary aperture exclusion affects a late corridor but is not the cause of the first switches.','No post-action48 refreshed inventory was generated; final observed blocker sets belong to pre-action48.','No proof that any h48 complete positive route exists or that all routes fail.'],
 'reader_note':'One exploratory console projection treated selected_cells (a scalar count) as a list and failed before output; corrected by reading target_ray_groups.selected_co_blockers. Experiment files unchanged.'}
(OUT/'diagnosis.json').write_text(json.dumps(report,indent=2)+'\n')
print(json.dumps({'exploration':report['exploration'],'switches':switches,'unlocks':unlocks,'diagnosis_sha256':hashlib.sha256((OUT/'diagnosis.json').read_bytes()).hexdigest()}))
