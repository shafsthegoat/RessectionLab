"""Stdlib-only projection of completed pilot JSON; no arrays, coefficients or models."""
import hashlib
import json
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
PILOT = ROOT / 'build/hybrid-proposal-pilot-v1'
RUN = PILOT / 'attempt-01'
OUT = Path(__file__).resolve().parent
INPUTS = {}


def load(path):
    raw = path.read_bytes()
    INPUTS[str(path.relative_to(ROOT))] = hashlib.sha256(raw).hexdigest()
    return json.loads(raw)


def compact_row(inv, identifier):
    matches = [r for r in inv['ledger'] if r.get('proposal_id') == identifier]
    main = next((r for r in matches if r['reason'] == 'PROPOSED_UNCERTIFIED'), matches[0])
    return main


def retained_summary(inv):
    rows = [compact_row(inv, x) for x in inv['shortlist']['selected_ids']]
    return {'raw_count': inv['shortlist']['raw_count'],
            'legal_count': inv['shortlist']['legal_count'],
            'family_counts': dict(Counter(r['family'] for r in rows)),
            'tool_counts': dict(Counter(r['tool_id'] for r in rows)),
            'native_reasons': dict(Counter(r['native_reason'] for r in rows)),
            'selected_rows_in_priority_order': rows}


def write(name, value):
    (OUT / name).write_text(json.dumps(value, indent=2, sort_keys=True) + '\n')


result = load(RUN / 'result.json')
receipt = load(PILOT / 'attempt-01.supervision/receipt.json')
release = load(PILOT / 'root-release.json')
source_index = load(PILOT / 'source-index.json')
assert result['status'] == 'complete_fixed_TRAIN_ridge_proposals'
assert receipt['status']=='complete' and receipt['exit_code']==0
assert receipt['cleanup_errors']==[] and receipt['remaining_owned_pids']==[] and receipt['worker_termination_confirmed']
assert receipt['result_sha256']==INPUTS[str((RUN/'result.json').relative_to(ROOT))]
assert receipt['release_sha256']==result['release_sha256']
assert len(result['arms']) == 12 and all(a['complete'] and a['independent_accepted'] for a in result['arms'])
assert result['release_sha256'] == INPUTS[str((PILOT / 'root-release.json').relative_to(ROOT))]
labels = load(RUN / 'training-labels.json')['states']
assert len(labels) == 29
methods = ['ridge_top8', 'sampled_target_top8', 'exhaustive']
subjects = ['ReMIND-002', 'ReMIND-015', 'ReMIND-018', 'ReMIND-045']
plans = {(m,s): load(RUN / m / s / 'plan.json') for m in methods for s in subjects}
arms = {(a['method'],a['subject']): a for a in result['arms']}
motion_recall = []
for state in labels:
    subject, step, teacher = state['subject'], state['step'], state['teacher_action']
    inv = load(RUN / 'ridge_top8' / subject / 'inventories' / f'state-{step:02d}.json')
    selected = inv['shortlist']['selected_ids']
    same_input = inv['shortlist']['input_binding'] == state['public_input_hash']
    best_queried = max(v['reward'] for v in plans['ridge_top8',subject]['accounting']['decisions'][step]['scores'])
    motion_recall.append({'subject': subject, 'step': step, 'teacher_action': teacher,
        'same_public_input_binding': same_input,
        'teacher_shortlisted': teacher in selected if teacher != 'STOP' else None,
        'teacher_priority_rank_if_top8': selected.index(teacher)+1 if teacher in selected else None,
        'best_priority_label':max(state['priority_labels']), 'best_queried_native_reward':best_queried,
        'best_reward_retained_when_same_input':best_queried>=max(state['priority_labels'])-1e-7 if same_input else None,
        'raw_count': len(state['raw_ids']), 'positive_label_count': sum(y>0 for y in state['priority_labels']),
        'teacher_label': state['priority_labels'][state['raw_ids'].index(teacher)] if teacher != 'STOP' else None})

ridge, exhaustive = [plans[m,'ReMIND-045'] for m in ('ridge_top8','exhaustive')]
div = next(i for i,(a,b) in enumerate(zip(ridge['actions'],exhaustive['actions'])) if a != b)
assert div == 11
ri, ei = [load(RUN / m / 'ReMIND-045/inventories' / f'state-{div:02d}.json') for m in ('ridge_top8','exhaustive')]
assert ri['source_state_hash'] == ei['source_state_hash']
assert ri['shortlist']['input_binding'] == ei['shortlist']['input_binding']
rid, eid = ridge['actions'][div], exhaustive['actions'][div]
state = next(s for s in labels if s['subject']=='ReMIND-045' and s['step']==div)
assert state['teacher_action'] == eid and state['public_input_hash'] == ri['shortlist']['input_binding']
assert eid not in ri['shortlist']['selected_ids'] and ri['shortlist']['selected_ids'][0] == rid
rd, ed = ridge['accounting']['decisions'][div], exhaustive['accounting']['decisions'][div]
scored = {v['action_id']: v for v in ed['scores']}
rh, eh = [load(RUN / m / 'ReMIND-045/replay/episode-metrics.json')['history'] for m in ('ridge_top8','exhaustive')]
assert rh[div]['removed_indices_native'] == eh[div]['removed_indices_native']
reward_gap = scored[eid]['reward'] - scored[rid]['reward']
path_gap = scored[rid]['complete_tool_path_length_mm'] - scored[eid]['complete_tool_path_length_mm']
assert abs(reward_gap - (.03 + .001*path_gap)) < 1e-12
assert abs(arms['exhaustive','ReMIND-045']['outcomes']['total_reward'] - arms['ridge_top8','ReMIND-045']['outcomes']['total_reward'] - reward_gap) < 1e-12
divergence = {'zero_based_step':div, 'one_based_decision':div+1, 'identical_prior_actions':div,
    'source_state_hash':ri['source_state_hash'], 'public_input_binding':ri['shortlist']['input_binding'],
    'raw_count':len(state['raw_ids']), 'positive_labels':sum(y>0 for y in state['priority_labels']),
    'zero_labels':sum(y==0 for y in state['priority_labels']), 'teacher_priority_rank':None,
    'teacher_priority_rank_bound':'9..91; exact rank and predicted scores were not persisted',
    'selected_priority_rank':1, 'teacher_row':compact_row(ei,eid), 'selected_row':compact_row(ri,rid),
    'selected_top8':retained_summary(ri), 'teacher_native_score':scored[eid], 'selected_native_score':scored[rid],
    'same_actual_removed_cells':rh[div]['removed_indices_native'], 'incremental_path_mm':path_gap,
    'tool_switch_cost':.03, 'extra_path_cost':.001*path_gap, 'entire_return_gap':reward_gap,
    'uniform_state_positive_weight_fraction':sum(y>0 for y in state['priority_labels'])/len(state['raw_ids']),
    'teacher_label':state['priority_labels'][state['raw_ids'].index(eid)],
    'selected_label':state['priority_labels'][state['raw_ids'].index(rid)]}

heuristic = []
for s in subjects:
    inv = load(RUN / 'sampled_target_top8' / s / 'inventories/state-00.json')
    initial = next(v for v in labels if v['subject']==s and v['step']==0)
    summary = retained_summary(inv)
    assert summary['legal_count']==0 and summary['native_reasons']=={'SHAFT_BLOCKED_BY_REMAINING_NATIVE_TISSUE':8}
    summary.update(subject=s, exhaustive_root_legal_count=sum(initial['legal']),
        exhaustive_root_positive_count=sum(y>0 for y in initial['priority_labels']),
        all_selected_training_labels_zero=all(initial['priority_labels'][initial['raw_ids'].index(i)]==0 for i in inv['shortlist']['selected_ids']),
        sample_values_coverage_bits_and_proxy_scores_persisted=False)
    heuristic.append(summary)

timings = {}
for m in methods:
    rows=[arms[m,s] for s in subjects]
    secs={k:sum(a['online_costs']['seconds'].get(k,0) for a in rows) for k in ('constructor_inclusive','inventory_inclusive','proposal_enumeration','public_features','ranking','exact_verification')}
    residual=secs['inventory_inclusive']-sum(secs[k] for k in ('proposal_enumeration','public_features','ranking','exact_verification'))
    timings[m]={'online_seconds':secs, 'inventory_unpartitioned_seconds':residual,
        'unpartitioned_includes':'ledger assembly, callback/file serialization, guards and other inventory overhead; not a logging-only measurement',
        'replay_features_seconds':sum(a['replay_costs']['seconds']['public_features'] for a in rows),
        'source_through_plan_seconds':sum(a['source_through_plan_seconds'] for a in rows)}
timings['exhaustive_unused_feature_fraction_of_online_wall']=timings['exhaustive']['online_seconds']['public_features']/timings['exhaustive']['source_through_plan_seconds']

def failures(a,b):
    x,y=a['outcomes'],b['outcomes']; tol=release['gates']['numeric_tolerance']
    return [k for k,ok in {
        'target':x['target_removed_mm3']>=y['target_removed_mm3']-tol,
        'return':x['total_reward']>=y['total_reward']-tol,
        'outside':x['normal_removed_mm3']<=y['normal_removed_mm3']+tol,
        'contact':x['cumulative_contacted_tissue_upper_bound_mm3']<=y['cumulative_contacted_tissue_upper_bound_mm3']+tol}.items() if not ok]
gates={s:{'versus_exhaustive':failures(arms['ridge_top8',s],arms['exhaustive',s]),
          'versus_heuristic':failures(arms['ridge_top8',s],arms['sampled_target_top8',s])} for s in subjects}
summary={'scope':'Saved JSON/scalar/source-only; no data arrays, feature reconstruction, coefficients, refit, native queries or model execution',
    'complete_cases':4,'complete_arms':12,'teacher_states':29,'fit_rows':result['fit_rows'],
    'teacher_motion_states':sum(v['teacher_action']!='STOP' for v in labels),
    'literal_teacher_ID_membership_not_quality_recall':sum(v['teacher_shortlisted'] is True for v in motion_recall),
    'matched_input_positive_states':sum(v['same_public_input_binding'] and v['teacher_action']!='STOP' for v in motion_recall),
    'matched_input_positive_states_retaining_best_reward':sum(v['same_public_input_binding'] and v['teacher_action']!='STOP' and v['best_reward_retained_when_same_input'] for v in motion_recall),
    'first_045_divergence':divergence,'all_teacher_states':motion_recall,
    'heuristic_roots':heuristic,'timing':timings,'literal_gate_failures':gates,
    'original_comparison':result['comparison'],
    'limitations':['No predicted scores or ranks outside top8 saved.',
        'Feature arrays and sample coverage bits are not saved; no diagnosis of feature aliasing or missing target samples is identified.',
        'Single fixed ordered TRAIN run; instrumented exhaustive is not minimum-cost native greedy.',
        'Raw contact is an upper-bound exposure measure including target tissue, not clinical harm. The original gate remains false.']}
write('diagnosis.json',summary)
for path in [PILOT/'hybrid_proposal.py',PILOT/'initial_inventory_worker.py',ROOT/'src/resectionlab/native_spatial_task.py']:
    INPUTS[str(path.relative_to(ROOT))]=hashlib.sha256(path.read_bytes()).hexdigest()
write('inputs.json',INPUTS)
print(json.dumps({'first_divergence':div, 'same_input_best_reward_recall':[summary['matched_input_positive_states_retaining_best_reward'],summary['matched_input_positive_states']], 'gate_failures':gates,'result_sha256':INPUTS[str((RUN/'result.json').relative_to(ROOT))]},indent=2))
