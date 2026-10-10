"""Completed saved scalar JSON only; no numerical/project imports or payloads."""
from pathlib import Path
import hashlib,json,statistics
ROOT=Path(__file__).resolve().parents[2];HERE=Path(__file__).resolve().parent
inputs={}
def read(p):
 p=ROOT/p;raw=p.read_bytes();inputs[str(p.relative_to(ROOT))]=hashlib.sha256(raw).hexdigest();return json.loads(raw)
def span(xs):return [min(xs),max(xs)] if xs else None
base='build/post-exposure-rl8-diagnosis-v1/projection-01/'
rl=read(base+'teacher-readouts.json');eps=read(base+'episodes.json');stops=read(base+'STOP-decisions.json');positive=read(base+'positive-cuts.json');updates=read(base+'updates.json');endpoints=read(base+'endpoints.json')
il=[]
for subject,count in zip(('ReMIND-002','ReMIND-015','ReMIND-018','ReMIND-045'),(1,14,1,13)):
 for step in range(count):
  r=read(f'build/post-exposure-learning-v1/IL64/attempt-01/teacher-readout/{subject}/state-{step:02d}.json')
  il.append({'subject':subject,'step':step,'teacher_action':r['teacher_action'],'legal_actions':sum(r['action_mask']),**r['scores']})
records=[]
for label,rows in [('IL64',il),('RL8',rl)]:
 for subject in ('ReMIND-002','ReMIND-015','ReMIND-018','ReMIND-045'):
  rs=[r for r in rows if r['subject']==subject];motion=[r for r in rs if r['teacher_action']!='STOP'];stop=[r for r in rs if r['teacher_action']=='STOP'][0]
  records.append({'method':label,'subject':subject,'teacher_STOP':{k:stop[k] for k in ('step','legal_actions','stop_probability','teacher_rank','value')},'teacher_STOP_forced':stop['legal_actions']==1,
   'motion_states':len(motion),'exact_motion_rank1':sum(r['teacher_movement_rank']==1 for r in motion),'movement_ranks':[r['teacher_movement_rank'] for r in motion],'movement_counts':[r['legal_actions']-1 for r in motion],
   'conditional_teacher_vs_uniform_ratio':span([r['conditional_teacher_movement_probability']*(r['legal_actions']-1) for r in motion]),
   'teacher_value_range':span([r['value'] for r in rs])})
by_subject=[]
for subject in ('ReMIND-002','ReMIND-015','ReMIND-018','ReMIND-045'):
 es=[r for r in eps if r['subject']==subject];ss=[r for r in stops if r['subject']==subject];ps=[r for r in positive if r['subject']==subject]
 by_subject.append({'subject':subject,'episode_returns':[e['return'] for e in es],
 'root_values':[e['root']['value'] for e in es],'positive_cut_count':len(ps),'positive_cut_advantages':span([r['detached_advantage'] for r in ps]),
 'STOP_count':len(ss),'forced_STOP_count':sum(r['forced_by_legal_inventory'] for r in ss),'STOP_advantages':span([r['detached_advantage'] for r in ss]),
 'STOP_actor_score_term':span([r['actor_score_term'] for r in ss]),
 'endpoint':next({k:r[k] for k in ('saved_plan_return','terminal_reason','certificate_outcomes')} for r in endpoints if r['subject']==subject)})
result={'teacher_rows':records,'sampled_by_subject':by_subject,'positive_cut_negative_advantage_rows':[{k:r[k] for k in ('subject','update','step','reward','return_to_go','value','detached_advantage')} for r in positive if r['detached_advantage']<0],
 'gradients':[{'update':r['update'],'preclip_norm':r['gradient_norm_before_clip'],'critic_head_squared_norm_fraction':r['critic_head_squared_norm_fraction'],'value_MSE':r['value_loss'],'actor_loss':r['actor_loss']} for r in updates],
 'interpretation':'Saved scalar evidence only. Positive advantage is not a guarantee of improved action rank; module norms do not identify shared-encoder causality.'}
(HERE/'mechanism-scalars.json').write_text(json.dumps(result,indent=2,sort_keys=True)+'\n');(HERE/'mechanism-inputs.json').write_text(json.dumps(inputs,indent=2,sort_keys=True)+'\n')
print(json.dumps(result,indent=2))
