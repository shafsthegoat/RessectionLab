"""Read saved JSON only. No arrays, project imports, models, or geometry execution."""
import hashlib,json,math
from pathlib import Path
from collections import Counter
ROOT=Path(__file__).resolve().parents[2]; OUT=Path(__file__).resolve().parent
BASE='build/post-exposure-learning-v1/IL64/attempt-01'; pins={}
def read(path):
 raw=(ROOT/path).read_bytes();pins[path]=hashlib.sha256(raw).hexdigest();return json.loads(raw)
def history_summary(trace):
 h=trace['metrics']['history'];motions=[x for x in h if x['action_id']!='STOP']
 cells=[tuple(c) for x in motions for c in x['removed_indices_native']]
 geometry=[(x['tool_id'],tuple(x['entry_mm']),tuple(x['tip_mm'])) for x in motions]
 tools=[x['tool_id'] for x in motions]
 return {'steps':len(h),'motions':len(motions),'STOP_decisions':len(h)-len(motions),
  'positive_motion_rewards':sum(x['reward']>0 for x in motions),
  'zero_removal_motions':sum(not x['removed_indices_native'] for x in motions),
  'exact_tool_entry_tip_repeats':len(geometry)-len(set(geometry)),
  'removed_cells':len(cells),'unique_removed_cells':len(set(cells)),
  'repeated_removed_cells':len(cells)-len(set(cells)),
  'removed_cell_set_sha256':hashlib.sha256(json.dumps(sorted(set(cells)),separators=(',',':')).encode()).hexdigest(),
  'tool_counts':dict(Counter(tools)),'tool_switches':sum(a!=b for a,b in zip(tools,tools[1:])),
  'complete_tool_path_length_mm':sum(x['complete_tool_path_length_mm'] for x in h),
  'target_removed_mm3':sum(x['target_removed_mm3'] for x in h),
  'outside_target_removed_mm3':sum(x['normal_removed_mm3'] for x in h),
  'public_return':sum(x['reward'] for x in h),
  'removed_cells_per_motion':dict(sorted(Counter(len(x['removed_indices_native']) for x in motions).items()))}
result=read(BASE+'/result.json'); teacher=read(BASE+'/teachers/ReMIND-045/complete-trace.json')
actor=read(BASE+'/TRAIN-greedy/IL/ReMIND-045/complete-trace.json')
plan=read(BASE+'/TRAIN-greedy/IL/ReMIND-045/plan.json')
source_plan=read('build/remind-post-exposure-feasibility-v1/attempt-01/ReMIND-045/greedy-plan.json')
coverage=read('build/remind-post-exposure-feasibility-v1/attempt-01/ReMIND-045/initial-public-proposal-coverage.json')
dynamics=read(BASE+'/training-dynamics.json'); config=read(BASE+'/configuration.json')
T,A=history_summary(teacher),history_summary(actor)
assert T['removed_cell_set_sha256']==A['removed_cell_set_sha256']
assert A['positive_motion_rewards']==A['motions']==24
assert plan['plan']['terminal_reason']=='HORIZON'
assert plan['plan']['history']==actor['metrics']['history']
assert result['optimizer_updates']=={'IL':64,'RL':0}
readouts=[]
for i,decision in enumerate(source_plan['accounting']['decisions']):
 r=read(BASE+f'/teacher-readout/ReMIND-045/state-{i:02d}.json');s=r['scores']; ids=r['action_ids']
 assert ids==[x['action_id'] for x in decision['scores']]
 assert r['observation_hash']==teacher['decisions'][i]['observation_hash']
 assert r['teacher_action']==decision['selected_action_id']==teacher['decisions'][i]['action_id']
 values={x['action_id']:x for x in decision['scores']}; teacher_reward=values[r['teacher_action']]['reward']; top_reward=values[s['greedy_action']]['reward']
 true_rank=1+sum(x>s['logits'][ids.index(r['teacher_action'])] for x in s['logits'])
 assert true_rank==s['teacher_rank']
 readouts.append({'step':i,'legal_motions':len(ids)-1,'teacher_action':r['teacher_action'],
  'top_action':s['greedy_action'],'exact_teacher_choice':r['teacher_action']==s['greedy_action'],
  'teacher_logit_rank':s['teacher_rank'],'teacher_logit':s['logits'][ids.index(r['teacher_action'])],
  'top_logit':max(s['logits']),'teacher_minus_top_logit':s['logits'][ids.index(r['teacher_action'])]-max(s['logits']),
  'STOP_probability':s['stop_probability'],'teacher_probability':s['teacher_probability'],
  'teacher_minus_STOP':s['teacher_minus_STOP'],'teacher_nominal_reward':teacher_reward,
  'top_action_nominal_reward':top_reward,'nominal_immediate_regret':teacher_reward-top_reward})
# Native affine is orthogonal; scalar projections recover source indices without array loads.
affine=coverage['native_physical_affine_ras_mm'];cols=[[affine[r][c] for r in range(3)] for c in range(3)]
origin=actor['metrics']['crop']['origin_voxels'];shape=actor['metrics']['crop']['shape']
def inside(point):
 delta=[point[r]-affine[r][3] for r in range(3)]
 idx=[sum(delta[r]*col[r] for r in range(3))/sum(x*x for x in col) for col in cols]
 return all(origin[i]-1e-6<=v<=origin[i]+shape[i]-1+1e-6 for i,v in enumerate(idx))
steps=[]
for i,(h,dec) in enumerate(zip(actor['metrics']['history'],actor['decisions'])):
 assert dec['reward']==h['reward'] and dec['action_id']==h['action_id']
 steps.append({'step':i,'action_id':h['action_id'],'tool':h['tool_id'],
 'removed_cells':len(h['removed_indices_native']),'target_removed_mm3':h['target_removed_mm3'],
 'outside_target_removed_mm3':h['normal_removed_mm3'],'reward':h['reward'],
 'complete_tool_path_length_mm':h['complete_tool_path_length_mm'],'legal_motions':len(dec['action_ids'])-1,
 'matches_any_teacher_observation':dec['observation_hash'] in {x['observation_hash'] for x in teacher['decisions']},
 'entry_inside_actor_crop_center_bounds':inside(h['entry_mm']), 'tip_inside_actor_crop_center_bounds':inside(h['tip_mm'])})
others={}
for subject in result['TRAIN']:
 rows=[read(BASE+f'/teacher-readout/{subject}/state-{i:02d}.json') for i in range(result['teacher_steps'][subject])] if isinstance(result['teacher_steps'],dict) else [json.loads(p.read_text()) for p in []]
 # Paths below are already fixed declared saved readouts, not acquired inputs.
 if not rows:
  rows=[read(str(p.relative_to(ROOT))) for p in sorted((ROOT/BASE/'teacher-readout'/subject).glob('state-*.json'))]
 others[subject]={'teacher_states':len(rows),'exact_teacher_choices':sum(r['teacher_action']==r['scores']['greedy_action'] for r in rows),
 'STOP_readouts':[{'step':r['step'],'legal_motions':len(r['action_ids'])-1,'STOP_probability':r['scores']['stop_probability']} for r in rows if r['teacher_action']=='STOP']}
extra_actions=A['motions']-T['motions'];extra_path=A['complete_tool_path_length_mm']-T['complete_tool_path_length_mm'];extra_switch=A['tool_switches']-T['tool_switches']
decomposition={'extra_action_cost':extra_actions*.03,'extra_path_cost':extra_path*.001,'extra_tool_switch_cost':extra_switch*.03,
 'total_return_gap':T['public_return']-A['public_return']}
assert math.isclose(sum(v for k,v in decomposition.items() if k!='total_return_gap'),decomposition['total_return_gap'],abs_tol=1e-10)
out={'scope':'saved-only; 045 plan not accepted by independent geometry; no repaired result inferred',
 'run_status':result['status'],'failure':result['failure'],'all_four_retained':result['TRAIN'],
 'accepted_other_case_outcomes':result['TRAIN_greedy'],'teacher045':T,'actor045_unaccepted_trace':A,
 'cost_decomposition':decomposition,'actor_steps':steps,'teacher_readouts045':readouts,'all_teacher_readout_counts':others,
 'rank_summary':{'exact_motion_choices':sum(r['exact_teacher_choice'] for r in readouts[:-1]),'motion_states':12,
 'positive_top_rewards_all_motion_states':all(r['top_action_nominal_reward']>0 for r in readouts[:-1]),
 'sum_same_teacher_state_immediate_regret':sum(r['nominal_immediate_regret'] for r in readouts),
 'warning':'Counterfactual same-state regrets must not be added to estimate an alternate whole-route return.'},
 'input_presence':{'initial_all_emitted_five_ray_samples_inside_crop':coverage['ray_sample_coverage_counts'],
 'all24_selected_entries_and_tips_inside_crop':all(r['entry_inside_actor_crop_center_bounds'] and r['tip_inside_actor_crop_center_bounds'] for r in steps),
 'contents_not_read':'No target/cavity image arrays or learned features opened; input coverage does not prove retained encoder detail.'},
 'training':{'objective':config['objective'],'updates':len(dynamics['updates']),'first_pre_update_loss':dynamics['updates'][0]['pre_update_loss'],
 'last_pre_update_loss':dynamics['updates'][-1]['pre_update_loss'],'endpoint':result['endpoint_teacher_metrics'],
 'last8_pre_update_losses':[x['pre_update_loss'] for x in dynamics['updates'][-8:]]},'input_pins':pins}
(OUT/'scalars.json').write_text(json.dumps(out,indent=2,sort_keys=True)+'\n')
print(json.dumps({'teacher':T,'IL_unaccepted_trace':A,'costs':decomposition,'ranks':out['rank_summary'],'inputs':out['input_presence']},indent=2))
