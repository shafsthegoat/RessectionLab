"""Saved scalar JSON only. No project/ML imports, payloads or checkpoint reads."""
from pathlib import Path
import hashlib,json,math

REPO=Path(__file__).resolve().parents[2]
SOURCE=REPO/'build/obstruction-opening-learning-v1/RL8/attempt-01'
OUTPUT=Path(__file__).resolve().parent
inputs={}
def read(path):
    data=path.read_bytes();inputs[str(path.relative_to(REPO))]=hashlib.sha256(data).hexdigest()
    return json.loads(data)
def write(name,value):
    (OUTPUT/name).write_text(json.dumps(value,indent=2,sort_keys=True,allow_nan=False)+'\n')
result=read(SOURCE/'result.json')
assert inputs[str((SOURCE/'result.json').relative_to(REPO))]=='60631a2fb4cebda2d6b29475156e487a8b51b4bd1c5cc1d6ca89dd2511e29897'
assert result['status']=='complete_matched_TRAIN_endpoint'
episodes=[];positive=[];stops=[];all_decisions=[]
for update in range(1,9):
    for subject in ('ReMIND-008','ReMIND-010','ReMIND-020','ReMIND-025'):
        r=read(SOURCE/'RL'/f'update-{update:02d}'/subject/'gradient-contribution.json')
        ds=r['rl_decision_diagnostics']['decisions']
        assert len(ds)==r['steps']==r['loss_forward_calls']
        assert abs(sum(d['reward'] for d in ds)-r['return'])<1e-12
        running=0.;best=0.;best_step=None
        for i,d in enumerate(ds):
            assert d['step']==i
            assert abs(sum(x['reward'] for x in ds[i:])-d['return_to_go'])<1e-12
            assert abs(d['return_to_go_model_dtype']-d['value']-d['detached_advantage'])<1e-5
            running+=d['reward']
            if running>best:best=running;best_step=i
            item={'update':update,'subject':subject,**d}
            all_decisions.append(item)
            if d['reward']>0:positive.append(item)
            if d['action_id']=='STOP':stops.append(item)
        episodes.append({'update':update,'subject':subject,'steps':len(ds),'return':r['return'],
            'positive_immediate_cuts':sum(d['reward']>0 for d in ds),
            'positive_cut_positive_RTG':sum(d['reward']>0 and d['return_to_go']>0 for d in ds),
            'positive_cut_positive_advantage':sum(d['reward']>0 and d['detached_advantage']>0 for d in ds),
            'best_observed_prefix_return_including_STOP_zero':best,'best_prefix_last_step':best_step,
            'post_best_prefix_return':r['return']-best if best_step is not None else None,
            'STOP_step':next((d['step'] for d in ds if d['action_id']=='STOP'),None),
            'STOP_advantage':next((d['detached_advantage'] for d in ds if d['action_id']=='STOP'),None),
            'root_entropy_nats':ds[0]['entropy'],'root_max_entropy_nats':math.log(ds[0]['legal_action_count']),
            'root_sampled_action_probability':math.exp(ds[0]['chosen_log_probability']),
            'root_value':ds[0]['value'],'root_legal_action_count':ds[0]['legal_action_count']})
endpoint=[]
for subject in ('ReMIND-008','ReMIND-010','ReMIND-020','ReMIND-025'):
    for step in range(4 if subject=='ReMIND-025' else 1):
        row=read(SOURCE/'teacher-readout'/subject/f'state-{step:02d}.json');s=row['scores']
        endpoint.append({'subject':subject,'step':step,'teacher_action':row['teacher_action'],
            'legal_actions':sum(row['action_mask']),**{k:v for k,v in s.items() if k not in ('logits','probabilities')}})
dynamics=read(SOURCE/'training-dynamics.json')['updates']
summary={'status':result['status'],'episodes':len(episodes),'decisions':len(all_decisions),
    'episode_returns':{'positive':sum(r['return']>0 for r in episodes),'zero':sum(r['return']==0 for r in episodes),'negative':sum(r['return']<0 for r in episodes)},
    'positive_immediate_cuts':len(positive),'positive_cut_positive_RTG':sum(d['return_to_go']>0 for d in positive),
    'positive_cut_positive_advantage':sum(d['detached_advantage']>0 for d in positive),
    'positive_cut_negative_advantage':sum(d['detached_advantage']<0 for d in positive),
    'STOP_decisions':len(stops),'STOP_positive_advantage':sum(d['detached_advantage']>0 for d in stops),
    'STOP_negative_advantage':sum(d['detached_advantage']<0 for d in stops),'root_STOP_episodes':sum(d['step']==0 for d in stops),
    'updates':len(dynamics),'clipped_updates':sum(d['gradient_exceeds_clip_threshold'] for d in dynamics),
    'gradient_norm_range':[min(d['gradient_norm_before_clip'] for d in dynamics),max(d['gradient_norm_before_clip'] for d in dynamics)],
    'checkpoint':result['checkpoints']['RL'],'endpoint_teacher_metrics':result['endpoint_teacher_metrics'],
    'all_four_reloaded_greedy_STOP':all(r['actions']==['STOP'] and r['public_return']==0 for r in result['TRAIN_greedy'].values()),
    'SELECT_EVAL_opened':result['SELECT_EVAL_opened'], 'private_reference_reads':result['private_reference_reads'],
    'source_scope':'completed saved scalar JSON only; no model/checkpoint payload or patient-array reads'}
assert summary['episodes']==32 and summary['decisions']==320 and len(endpoint)==7
write('summary.json',summary);write('episode-table.json',episodes);write('positive-cut-table.json',positive)
write('endpoint-table.json',endpoint);write('input-index.json',inputs)
print(json.dumps(summary,indent=2))
