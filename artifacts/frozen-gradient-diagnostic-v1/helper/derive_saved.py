"""Saved TRAIN scalar arithmetic only: no project, array or ML imports."""
import hashlib
import json
import math
from pathlib import Path
import statistics

ROOT=Path(__file__).resolve().parents[3]
OUT=Path(__file__).resolve().parent

def main():
    pins={}
    def read(path):
        raw=(ROOT/path).read_bytes();pins[path]={'sha256':hashlib.sha256(raw).hexdigest(),'bytes':len(raw)}
        return json.loads(raw)
    corpus=read('build/public-motion-ranking-v1/public-score-corpus.json')
    results={}
    for name,folder in [('original_CE','build/post-exposure-learning-v1/IL64/attempt-01'),
                        ('ranking','build/public-motion-ranking-il64-v1/IL64/attempt-01')]:
        updates=read(folder+'/training-dynamics.json')['updates']
        rows=[];ties=0;minimum_separation=math.inf
        for case in corpus['subjects']:
            for label in case['decisions']:
                d=read(folder+'/teacher-readout/'+case['subject']+'/state-%02d.json'%label['step'])
                assert d['observation_hash']==label['observation_hash'] and d['action_ids']==label['action_ids']
                assert d['action_mask']==label['action_mask'] and d['teacher_action']==label['teacher_action']
                if label['teacher_action']=='STOP':continue
                z=d['scores']['logits'];r=label['rewards'];mask=label['action_mask']
                legal=[i for i,m in enumerate(mask) if m];moves=[i for i in legal if i]
                # Float64 scalar reconstruction from saved float32 logits.
                top=max(z[i] for i in legal);exps={i:math.exp(z[i]-top) for i in legal}
                den=math.fsum(exps.values());p={i:exps[i]/den for i in legal}
                pairs=[(i,j,r[i]-r[j]) for i in moves for j in moves if r[i]>r[j]]
                gap_sum=math.fsum(g for i,j,g in pairs);gradient={i:0. for i in legal}
                for i,j,gap in pairs:
                    v=gap/gap_sum/(1+math.exp(z[i]-z[j]));gradient[i]-=v;gradient[j]+=v
                    ties+=z[i]==z[j];minimum_separation=min(minimum_separation,abs(z[i]-z[j]))
                gate={i:-p[0]*p[i]/(1-p[0]) if i else p[0] for i in legal}
                teacher=label['action_ids'].index(label['teacher_action']);chosen=max(moves,key=lambda i:z[i])
                regret=r[teacher]-r[chosen]
                rows.append({'subject':case['subject'],'step':label['step'],'motion_candidates':len(moves),
                    'strict_pairs':len(pairs),'teacher_action':label['teacher_action'],
                    'chosen_motion':label['action_ids'][chosen],'nominal_regret':regret,
                    'STOP_probability':p[0], 'motion_gate_logit_gradient_L1':p[0],
                    'ranking_motion_logit_gradient_L1':math.fsum(abs(gradient[i]) for i in moves),
                    'teacher_minus_chosen_margin_descent_derivative':gradient[chosen]+gate[chosen]-gradient[teacher]-gate[teacher],
                    'reversed_reward_gap_mass':math.fsum(g for i,j,g in pairs if z[i]<z[j])/gap_sum})
        wrong=[row for row in rows if row['nominal_regret']>0]
        results[name]={'scope':'free-logit derivatives, not parameter gradients or Adam steps',
            'motion_rows':rows,'unequal_reward_logit_ties':ties,'minimum_unequal_reward_logit_separation':minimum_separation,
            'strict_pairs':sum(x['strict_pairs'] for x in rows),'regretful_rows':len(wrong),
            'regretful_margins_improved_by_free_logit_descent':sum(x['teacher_minus_chosen_margin_descent_derivative']>0 for x in wrong),
            'weighted_motion_gate_L1':.5/25*math.fsum(x['motion_gate_logit_gradient_L1'] for x in rows),
            'weighted_motion_ranking_L1':.5/25*math.fsum(x['ranking_motion_logit_gradient_L1'] for x in rows),
            'mean_reversed_gap_mass':statistics.mean(x['reversed_reward_gap_mass'] for x in rows),
            'first_update':updates[0],'last_update':updates[-1],
            'late32_module_gradient_medians':{k:statistics.median(x['module_gradient_norms_before_clip'][k] for x in updates[32:]) for k in ['actor','encoder','stop']},
            'clipped_update_count':sum(x['gradient_exceeds_clip_threshold'] for x in updates)}
    for path in ['src/resectionlab/spatial_policy.py','src/resectionlab/public_motion_ranking.py',
                 'src/resectionlab/patient_planning_accumulation.py','build/post-exposure-representation-audit-v1/REPORT.txt',
                 'artifacts/public-motion-ranking-il64-result-v1/REPORT.txt']:
        raw=(ROOT/path).read_bytes();pins[path]={'sha256':hashlib.sha256(raw).hexdigest(),'bytes':len(raw)}
    for name,data in [('saved-scalar-derivation.json',results),('saved-input-pins.json',pins)]:
        with (OUT/name).open('x') as f:json.dump(data,f,sort_keys=True,indent=2,allow_nan=False);f.write('\n')
    print({name:{k:v for k,v in r.items() if k in ['strict_pairs','regretful_rows','regretful_margins_improved_by_free_logit_descent','weighted_motion_gate_L1','weighted_motion_ranking_L1']} for name,r in results.items()})

if __name__=='__main__':main()
