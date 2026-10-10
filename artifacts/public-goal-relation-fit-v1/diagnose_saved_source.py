"""Arithmetic on already saved TRAIN JSON only; no project/model/checkpoint imports."""
from pathlib import Path
import json
import math
import hashlib
ROOT=Path(__file__).resolve().parents[3];BASE=Path(__file__).resolve().parent
OUT=BASE/'stage/artifacts/public-goal-relation-fit-v1';OUT.mkdir(parents=True,exist_ok=True)
OLD=ROOT/'build/goal-conditioned-policy-v1/full-teacher-refit-run-v1'
NEW=ROOT/'build/goal-conditioned-policy-v1/goal-relation-fit-run-v1'
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def read(p):return json.loads(p.read_text())
def dump(name,value):(OUT/name).write_text(json.dumps(value,sort_keys=True,indent=2)+'\n')
rows=[];input_pins={}
for i in range(40):
    paths=[OLD/f'after-state-{i:02d}.json',NEW/f'after-state-{i:02d}.json']
    old,new=map(read,paths)
    for p in paths:input_pins[p.relative_to(ROOT).as_posix()]=sha(p)
    for key in ('layout_id','goal_id','step','teacher_action','teacher_action_index','observation_hash','action_ids','legal_mask','action_modes'):
        if old[key]!=new[key]:raise ValueError('Fixed state changed '+key)
    records=[]
    for row in (old,new):
        legal=[(j,z) for j,z in enumerate(row['logits']) if row['legal_mask'][j]]
        target=row['teacher_action_index'];teacher=row['logits'][target]
        competitor=max(z for j,z in legal if j!=target)
        maxz=max(z for _,z in legal);den=sum(math.exp(z-maxz) for _,z in legal)
        probs={j:math.exp(z-maxz)/den for j,z in legal}
        ce=-math.log(probs[target]);stop=probs[0]
        if abs(ce-row['cross_entropy'])>1e-10 or abs(stop-row['STOP_probability'])>1e-10:raise ValueError('Score arithmetic differs')
        records.append({'cross_entropy':ce,'correct':row['correct'],'greedy_mode':row['greedy_mode'],
            'greedy_action':row['greedy_action'],'teacher_probability':probs[target],'STOP_probability':stop,
            'STOP_minus_best_movement_margin':row['STOP_minus_best_movement_margin'],
            'teacher_minus_best_other_margin':teacher-competitor,
            'teacher_rank_among_legal_movements':row['teacher_rank_among_legal_movements'],
            'raw_CE_STOP_logit_gradient':stop-(1 if target==0 else 0),
            'raw_CE_teacher_logit_gradient':probs[target]-1})
    rows.append({'state_index':i,'layout_id':new['layout_id'],'goal_id':new['goal_id'],'step':new['step'],
        'teacher_mode':new['action_modes'][new['teacher_action_index']],
        'teacher_action':new['teacher_action'],'observation_hash':new['observation_hash'],
        'legal_actions':sum(new['legal_mask']),'movement_inventory_counts':new['movement_inventory_counts'],
        'baseline':records[0],'relation':records[1]})
groups={'STOP8':lambda r:r['teacher_mode']=='stop',
    'aspiration_root16':lambda r:r['step']==0 and r['teacher_mode']=='aspirate',
    'second16':lambda r:r['step']==1}
summary={}
for name,condition in groups.items():
    items=[r for r in rows if condition(r)];g={'n':len(items)}
    for model in ('baseline','relation'):
        g[model]={key:sum(r[model][key] for r in items)/len(items) for key in
            ('cross_entropy','teacher_probability','STOP_probability','raw_CE_STOP_logit_gradient')}
        margins=[r[model]['STOP_minus_best_movement_margin'] for r in items]
        g[model].update(STOP_margin_min=min(margins),STOP_margin_max=max(margins),
            STOP_greedy=sum(r[model]['greedy_mode']=='stop' for r in items),
            STOP_probability_above_half=sum(r[model]['STOP_probability']>=.5 for r in items),
            conditional_teacher_rank1=sum(r[model]['teacher_rank_among_legal_movements']==1 for r in items))
    g['weighted_mean_CE_change']=(g['relation']['cross_entropy']-g['baseline']['cross_entropy'])*len(items)/40
    summary[name]=g
roots=[r for r in rows if r['step']==0]
positive=[r for r in roots if r['teacher_mode']=='aspirate'];negative=[r for r in roots if r['teacher_mode']=='stop']
intervals={}
for model in ('baseline','relation'):
    # Analytical feasibility only; no bias is selected or applied.
    lower=max(r[model]['STOP_minus_best_movement_margin'] for r in positive)
    upper=min(r[model]['STOP_minus_best_movement_margin'] for r in negative)
    intervals[model]={'necessary_lower_exclusive_to_unstop_all16':lower,
        'necessary_upper_inclusive_to_preserve_all8_STOP':upper,'separable_by_single_STOP_bias':lower<upper,
        'no_bias_selected_or_applied':True}
updates=[]
for i in range(1,33):
    p=NEW/f'update-{i:02d}.json';updates.append(read(p));input_pins[p.relative_to(ROOT).as_posix()]=sha(p)
dump('saved-TRAIN-state-comparison.json',{'scope':'complete40_saved_TRAIN_score_arithmetic_only','rows':rows})
dump('STOP-diagnosis.json',{'class_decomposition':summary,'single_bias_feasibility_not_tuning':intervals,
    'all40_CE_change':sum(g['weighted_mean_CE_change'] for g in summary.values()),
    'raw_logit_gradients':'exact derivative of unweighted categorical CE at stored outputs; not parameter/backward execution',
    'source_paths':['src/resectionlab/contact_learning.py','src/resectionlab/contact_learning_contract.py',
                    'src/resectionlab/goal_mode_spatial_policy.py','src/resectionlab/goal_relation_spatial_policy.py'],
    'new_native_rollout':False,'checkpoint_reads':0,'model_forwards':0,'optimizer_updates':0,'input_pins':input_pins,
    'recorded_training_curve':[{'update':u['update'],'pre_update_CE':u['loss']['loss'],
        'pre_clip_gradient_norm':u['update_receipt']['gradient_norm_before_clip'],
        'module_gradient_norms':u['update_receipt']['module_gradient_norms_before_clip']} for u in updates]})
print(json.dumps({'curve_points':[(u['update'],u['loss']['loss']) for u in updates if u['update'] in (1,8,16,24,31,32)],
    'pcf19':[r for r in rows if r['step']==0 and r['layout_id']=='pcf-19']},indent=2))
