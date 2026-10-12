"""Saved scalar ranking diagnostic; no model calls or counterfactual execution."""
import hashlib, json, math, signal, time
from pathlib import Path
signal.signal(signal.SIGALRM, lambda *args: (_ for _ in ()).throw(TimeoutError("20 s cap")))
signal.alarm(20)
t0=time.perf_counter()
root=Path(__file__).resolve().parents[3]
p=root/'build/continuation-next-iteration-v1/recovery-01/report.json'
raw=p.read_bytes()
assert hashlib.sha256(raw).hexdigest()=='c423ea87fae67fcdc5751ff0c338297429c343b912d4a7baf28dd7cb5dce3d5b'
r=json.loads(raw);result={}
for subject, methods in r['cases'].items():
 decisions=methods['CALIBRATED']['planning']['decisions'];rows=[]
 for d in decisions:
  scores=[s for s in d['scores'] if s['action_id']!='STOP' and s['reward']>0]
  if not scores:continue
  selected=next(s for s in scores if s['action_id']==d['selected_action_id'])
  greedy=max(scores,key=lambda s:s['reward'])
  row={'step':d['step'],'selected_action_id':selected['action_id'],'local_greedy_action_id':greedy['action_id'],'selected_reward':selected['reward'],'local_greedy_reward':greedy['reward'],'immediate_reward_gap':greedy['reward']-selected['reward'],'selected_Q':selected['predicted_q'],'local_greedy_Q':greedy['predicted_q'],'Q_margin':selected['predicted_q']-greedy['predicted_q'],'selected_implied_tail':selected['predicted_q']-selected['reward'],'local_greedy_implied_tail':greedy['predicted_q']-greedy['reward']}
  assert row['immediate_reward_gap']>=-1e-10 and row['Q_margin']>=-1e-6
  rows.append(row)
 result[subject]={'productive_decisions':len(rows),'locally_lower_immediate_reward_choices':sum(x['immediate_reward_gap']>1e-8 for x in rows),'sum_local_immediate_reward_gaps_not_realized_regret':math.fsum(x['immediate_reward_gap'] for x in rows),'negative_selected_implied_tail':sum(x['selected_implied_tail']<0 for x in rows),'rows':rows}
report={'source_sha256':hashlib.sha256(raw).hexdigest(),'schema':'saved-direct-calibrated-local-ranking-v1','cases':result,'seconds':time.perf_counter()-t0,'model_calls':0,'native_calls':0,'scope':'Same visited states only; summed local immediate gaps are not complete-route regret or counterfactual outcome. Q-minus-r is diagnostic, not the new fitted tail prediction.'}
output=Path(__file__).with_name('direct-ranking-01.json');data=(json.dumps(report,indent=2,sort_keys=True)+'\n').encode();assert len(data)<65536
with output.open('xb') as f:f.write(data)
print(json.dumps({k:{n:v for n,v in x.items() if n!='rows'} for k,x in result.items()}))
