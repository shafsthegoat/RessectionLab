"""Saved finite-tree arithmetic only: no model, simulator, checkpoint or image imports."""
import collections
import hashlib
import json
import math
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
PILOT = ROOT/'artifacts/native-opening-learning-v1'
FIT = ROOT/'build/native-opening-learning-diagnostic-v1'
CAPACITY = ROOT/'artifacts/native-opening-bc-capacity-v1'
bindings = {}


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read(path, expected=None):
    raw = path.read_bytes(); digest = hashlib.sha256(raw).hexdigest()
    if expected is not None and digest != expected:
        raise ValueError('Saved input hash mismatch: '+str(path))
    bindings[str(path.relative_to(ROOT))] = digest
    return json.loads(raw)


index = read(PILOT/'output-sha256.json')
teacher = read(PILOT/'teacher.json', index['teacher.json'])
preparation = read(PILOT/'preparation.json', index['preparation.json'])
pilot_result = read(PILOT/'result.json', index['result.json'])
fit_index = read(FIT/'evidence-index.json')
fit_sha = next(r['sha256'] for r in fit_index['files'] if r['path']=='teacher-fit-attempt-01/readout.json')
fit = read(FIT/'teacher-fit-attempt-01/readout.json', fit_sha)
capacity_index = read(CAPACITY/'output-sha256.json')
capacity = read(CAPACITY/'result.json', capacity_index['result.json'])
assert bindings[str((CAPACITY/'result.json').relative_to(ROOT))] == 'fca1c98193f585b283a215397843b1b2a41604627ff2bcc3a6ab6884b443ea3b'
assert teacher['complete'] and teacher['unique_supervised_states']==5 and teacher['terminal_sequences']==16
assert fit['status']=='complete' and fit['forward_completed']==15 and fit['optimizer_updates']==0
assert capacity['status']=='complete' and capacity['updates']==256 and capacity['update16_exact_reproduction']
assert fit['context']['decision_model_hash']==preparation['decision_model_hash']==capacity['context']['decision_model_hash']
assert fit['context']['source_ids']==capacity['context']['source_ids']==[preparation['source_hash']]
states = {r['observation_hash']:r for r in teacher['state_rows']}
prefixes = {tuple(r['prefix']):r['observation_hash'] for r in teacher['supervised_state_demonstrations']}
root = prefixes[()]
assert root == preparation['initial_observation_hash']
root_scores = {r['action_id']:r['nominal_return_to_go'] for r in states[root]['scores']}
initial_rewards = {'STOP':root_scores['STOP']}
routes = [{'actions':['STOP'], 'nominal_return':root_scores['STOP']}]
for prefix, state in prefixes.items():
    if not prefix: continue
    assert len(prefix)==1 and prefix[0]!='STOP'
    scores = states[state]['scores']
    initial_rewards[prefix[0]] = root_scores[prefix[0]] - max(r['nominal_return_to_go'] for r in scores)
    for row in scores:
        routes.append({'actions':[*prefix,row['action_id']],
                       'nominal_return':initial_rewards[prefix[0]]+row['nominal_return_to_go']})
route_map = {tuple(r['actions']):r for r in routes}
assert len(route_map)==16
visits = collections.Counter()
largest_return_mismatch = largest_step_mismatch = 0.
for update in range(16):
    for member in range(4):
        name=f'rl-{update:02}-{member}.json'
        episode=read(PILOT/name,index[name])
        assert episode['status']=='complete' and episode['independent_evaluation']['accepted']
        assert episode['decisions'][-1]['terminated'] and episode['metrics']['terminated']
        actions=tuple(r['action_id'] for r in episode['decisions']); route=route_map[actions]
        actual=episode['independent_evaluation']['outcomes']
        mismatch=abs(route['nominal_return']-actual['total_reward'])
        largest_return_mismatch=max(largest_return_mismatch,mismatch);assert mismatch<1e-12
        first=episode['decisions'][0]
        largest_step_mismatch=max(largest_step_mismatch,abs(first['reward']-initial_rewards[actions[0]]))
        assert abs(first['reward']-initial_rewards[actions[0]])<1e-12
        if len(actions)==2:
            second=episode['decisions'][1]; state=prefixes[(actions[0],)]
            q=next(r['nominal_return_to_go'] for r in states[state]['scores'] if r['action_id']==actions[1])
            assert second['observation_hash']==state and abs(second['reward']-q)<1e-12
        assert actual['total_reference_target_mm3']==2.
        fields={key:actual[key] for key in ('target_removed_mm3','normal_removed_mm3','total_reward')}
        if 'saved_audited_outcomes' in route: assert route['saved_audited_outcomes']==fields
        else:route['saved_audited_outcomes']=fields
        visits[actions]+=1
assert set(visits)==set(route_map)
for route in routes: route['saved_audited_episode_occurrences']=visits[tuple(route['actions'])]

models={name:row['predictions'] for name,row in fit['models'].items()}
models['BC256']=capacity['readouts']['256']['states']


def distributions(predictions, mode):
    result={}
    for row in predictions:
        state=row['observation_hash']; ids=states[state]['action_ids']
        assert row['teacher_action']==states[state]['selected_action_id']
        assert len(row['probabilities'])==len(row['logits'])==len(ids)
        if 'action_ids' in row:assert row['action_ids']==ids
        assert all(math.isfinite(x) for x in row['logits'])
        assert all(math.isfinite(x) and x>=0 for x in row['probabilities'])
        assert abs(math.fsum(row['probabilities'])-1)<1e-6
        values=(row['probabilities'] if mode=='saved_probabilities' else
                [math.exp(x-max(row['logits'])) for x in row['logits']])
        total=math.fsum(values);result[state]={a:p/total for a,p in zip(ids,values)}
    assert set(result)==set(states)
    return result


def evaluate(probabilities):
    leaves=[]; grouped={a:[] for a in states[root]['action_ids']}
    for route in routes:
        a=route['actions'];mass=probabilities[root][a[0]]
        if len(a)==2:mass*=probabilities[prefixes[(a[0],)]][a[1]]
        out=route['saved_audited_outcomes'];entry={'actions':a,'probability':mass,
            'return_contribution':mass*route['nominal_return'],
            'any_target_contribution':mass*(out['target_removed_mm3']>0),
            'full_target_contribution':mass*(out['target_removed_mm3']==2),
            'optimal_route_contribution':mass*(abs(route['nominal_return']-teacher['nominal_optimum'])<1e-12),
            'expected_target_mm3_contribution':mass*out['target_removed_mm3'],
            'expected_normal_mm3_contribution':mass*out['normal_removed_mm3']}
        leaves.append(entry);grouped[a[0]].append(entry)
    assert abs(math.fsum(r['probability'] for r in leaves)-1)<1e-12
    total={k.removesuffix('_contribution'):math.fsum(r[k] for r in leaves)
           for k in leaves[0] if k.endswith('_contribution')}
    contributions=[]
    inventory={r['action_id']:r for r in preparation['inventory']['emitted']}
    for action,subset in grouped.items():
        p=probabilities[root][action];contribution=math.fsum(r['return_contribution'] for r in subset)
        conditional=0. if action=='STOP' else math.fsum(
            probabilities[prefixes[(action,)]][r['action_id']]*r['nominal_return_to_go']
            for r in states[prefixes[(action,)]]['scores'])
        assert abs(contribution-p*(initial_rewards[action]+conditional))<1e-12
        contributions.append({'action_id':action,'tool_id':None if action=='STOP' else inventory[action]['tool_id'],
            'tip_mm':None if action=='STOP' else inventory[action]['tip_mm'], 'probability':p,
            'immediate_nominal_reward':initial_rewards[action], 'conditional_future_return':conditional,
            'conditional_total_return':initial_rewards[action]+conditional, 'expected_return_contribution':contribution,
            'any_target_probability_contribution':math.fsum(r['any_target_contribution'] for r in subset)})
    return {'totals':total,'root_contributions':contributions,'terminal_probabilities':leaves}


results={}; probabilities={}
return_range=max(r['nominal_return'] for r in routes)-min(r['nominal_return'] for r in routes)
for name,predictions in models.items():
    p=distributions(predictions,'saved_probabilities');q=distributions(predictions,'float64_softmax')
    probabilities[name]=p;out=evaluate(p);alternative=evaluate(q)
    state_tv={s:.5*math.fsum(abs(p[s][a]-q[s][a]) for a in p[s]) for s in states}
    path_tv_bound=state_tv[root]+max(state_tv[s] for s in states if s!=root)
    out['rounding']={'maximum_saved_probability_sum_error':max(abs(math.fsum(r['probabilities'])-1) for r in predictions),
        'float64_softmax_alternative_totals':alternative['totals'],
        'state_total_variation_against_float64_softmax':state_tv,
        'terminal_total_variation_bound':path_tv_bound,
        'return_difference_bound_against_alternative':return_range*path_tv_bound,
        'actual_return_difference_against_alternative':abs(out['totals']['return']-alternative['totals']['return'])}
    assert out['rounding']['actual_return_difference_against_alternative']<=return_range*path_tv_bound+1e-14
    # Reproduce greedy selection from the already saved logits, without calling a model.
    by_state={r['observation_hash']:r for r in predictions}
    action=max(zip(states[root]['action_ids'],by_state[root]['logits']),key=lambda x:x[1])[0]
    actions=[action]
    if action!='STOP':
        s=prefixes[(action,)];actions.append(max(zip(states[s]['action_ids'],by_state[s]['logits']),key=lambda x:x[1])[0])
    historical=(pilot_result['methods'][name] if name!='BC256' else capacity['methods']['BC256'])
    assert actions==historical['actions']
    assert abs(route_map[tuple(actions)]['nominal_return']-historical['outcomes']['total_reward'])<1e-12
    out['saved_argmax_route_return']=route_map[tuple(actions)]['nominal_return']
    results[name]=out

initial_by={r['action_id']:r for r in results['initial']['root_contributions']}
decomposition={}
for name in ('BC','scratch-RL','BC256'):
    final_by={r['action_id']:r for r in results[name]['root_contributions']}
    root_effect=math.fsum((final_by[a]['probability']-r['probability'])*r['conditional_total_return'] for a,r in initial_by.items())
    continuation_effect=math.fsum(final_by[a]['probability']*(final_by[a]['conditional_total_return']-r['conditional_total_return']) for a,r in initial_by.items())
    delta=results[name]['totals']['return']-results['initial']['totals']['return']
    assert abs(root_effect+continuation_effect-delta)<1e-12
    decomposition[name]={'expected_return_change':delta,'root_distribution_effect_with_initial_continuations':root_effect,
                         'continuation_effect_under_final_root_distribution':continuation_effect,
                         'any_target_probability_change':results[name]['totals']['any_target']-results['initial']['totals']['any_target']}
for path,digest in bindings.items():assert sha(ROOT/path)==digest
report={'schema':'native-opening-saved-policy-expectation-v1','status':'complete','method':'Exhaustive probability products on the five-state, sixteen-leaf saved horizon-two nominal tree.',
    'probability_rule':'Normalize saved float32 probabilities in float64 with math.fsum; separately check float64 softmax of saved logits.',
    'nominal_route_rewards':'Root Q*(a) minus max child Q*(b) gives the first reward; child Q*(b) equals the terminal one-step reward. Every association checked against all64 saved independently accepted episodes.',
    'success_rule':'Any/full target indicators come from saved audited per-route target removal (greater than0 / exactly2mm3). No new geometry audit, clinical outcome or rollout.',
    'route_count':16,'saved_episode_count':64,'largest_nominal_vs_saved_actual_return_error':largest_return_mismatch,
    'largest_derived_first_reward_error':largest_step_mismatch,'routes':routes,'models':results,'initial_to_final_decomposition':decomposition,
    'uncertainty':'No Monte Carlo sampling error conditional on this complete deterministic task and frozen probabilities. Reported float32 probabilities/logits introduce rounding; the alternative and TV bound quantify only numerical representation differences. No anatomy, mechanics or generalization uncertainty is estimated.',
    'source_sha256':sha(Path(__file__)),'input_files':bindings,'all_inputs_unchanged_after_calculation':True,
    'execution_scope':{'new_forwards':0,'optimizer_updates':0,'new_simulator_transitions':0,'checkpoints_loaded':0,'patient_access':False}}
output=Path(__file__).with_name('expectation.json')
if output.exists():raise ValueError('Preserve existing arithmetic result')
output.write_text(json.dumps(report,indent=2,sort_keys=True,allow_nan=False)+'\n')
print(json.dumps({'path':str(output.relative_to(ROOT)),'sha256':sha(output),'decomposition':decomposition,
                  'totals':{name:r['totals'] for name,r in results.items()},'rounding':{name:r['rounding'] for name,r in results.items()}},indent=2))
