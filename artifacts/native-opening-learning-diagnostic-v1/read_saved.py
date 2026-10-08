"""JSON/source arithmetic only; no torch, checkpoint, environment or forward calls."""
import collections
import hashlib
import json
import math
from pathlib import Path
import statistics

ROOT = Path(__file__).resolve().parents[2]
SAVED = ROOT / 'artifacts/native-opening-learning-v1'
bindings = {}


def read(name):
    content = (SAVED / name).read_bytes()
    bindings[name] = hashlib.sha256(content).hexdigest()
    return json.loads(content)


teacher = read('teacher.json')
rows = {row['observation_hash']: row for row in teacher['state_rows']}
proofs = teacher['supervised_state_demonstrations']
root = next(row for row in proofs if not row['prefix'])['observation_hash']
best = rows[root]['selected_action_id']
tree_sequences = {('STOP',)}
for proof in proofs:
    if proof['prefix']:
        assert len(proof['prefix']) == 1
        tree_sequences.update((*proof['prefix'], action) for action in rows[proof['observation_hash']]['action_ids'])
episodes = [read(f'rl-{update:02}-{member}.json') for update in range(16) for member in range(4)]
assert all(e['status'] == 'complete' and e['independent_evaluation']['accepted'] for e in episodes)
sequences = collections.Counter(tuple(d['action_id'] for d in e['decisions']) for e in episodes)
assert set(sequences) == tree_sequences and len(tree_sequences) == teacher['terminal_sequences'] == 16
initial = read('initial.json')
evaluations = {}
for name in ('initial', 'BC', 'scratch-RL'):
    episode = initial if name == 'initial' else read(name + '.json')
    decision = episode['decisions'][0]
    probabilities = dict(zip(decision['action_ids'], decision['probabilities']))
    evaluations[name] = {'return': episode['simulated_return'], 'root_action': decision['action_id'],
        'root_probabilities': probabilities, 'root_teacher_probability': probabilities[best],
        'root_teacher_cross_entropy': -math.log(probabilities[best]), 'root_value': decision['value']}
gradients = {}
for name in ('BC', 'scratch-RL'):
    records = read(name + '-updates.json')
    gradients[name] = {'updates': len(records), 'all_parameters_changed': all(r['gradient']['parameters_changed'] for r in records),
        'gradient_ranges': {key: [min(r['gradient'][key] for r in records), max(r['gradient'][key] for r in records)]
            for key in ('actor_gradient_norm_before_clip', 'encoder_gradient_norm_before_clip',
                        'critic_gradient_norm_before_clip', 'stop_gradient_norm_before_clip', 'gradient_norm_before_clip')},
        'clipped_updates': [{'update': r['update'], 'factor': 5/(r['gradient']['gradient_norm_before_clip']+1e-6)}
            for r in records if r['gradient']['gradient_norm_before_clip'] > 5],
        'logged_preupdate_loss_first': records[0]['loss'], 'logged_preupdate_loss_last': records[-1]['loss']}
conditioned = []
for action in rows[root]['action_ids']:
    subset = [e for e in episodes if e['decisions'][0]['action_id'] == action]
    conditioned.append({'root_action': action, 'episodes': len(subset),
        'mean_sampled_return': statistics.mean(e['simulated_return'] for e in subset),
        'positive_return_episodes': sum(e['simulated_return'] > 0 for e in subset),
        'mean_recorded_advantage': statistics.mean(e['simulated_return']-e['decisions'][0]['value'] for e in subset)})
coverage = []
for state, row in rows.items():
    visits = [d for e in episodes for d in e['decisions'] if d['observation_hash'] == state]
    coverage.append({'observation_hash': state, 'teacher_action': row['selected_action_id'], 'visits': len(visits),
        'teacher_selected': sum(d['action_id'] == row['selected_action_id'] for d in visits),
        'action_counts': dict(collections.Counter(d['action_id'] for d in visits))})
source_checks = {}
declaration = read('declaration-input.json')
for name in ('scripts/run_native_opening_learning.py', 'src/resectionlab/spatial_policy.py', 'src/resectionlab/native_spatial_task.py'):
    current = hashlib.sha256((ROOT/name).read_bytes()).hexdigest()
    snapshot = hashlib.sha256((SAVED/'source-snapshot'/name).read_bytes()).hexdigest()
    assert current == snapshot == declaration['source_sha256'][name]
    source_checks[name] = current
result = {'scope': 'Read-only saved JSON and source arithmetic. No new policy/optimizer/rollout/checkpoint/holdout access.',
    'teacher': {'states': len(rows), 'stop_labels': sum(r['selected_action_id']=='STOP' for r in rows.values()),
                'tool_labels': sum(r['selected_action_id']!='STOP' for r in rows.values()),
                'state_weight': 1/len(rows), 'observation_conflicts': teacher['observation_conflicts']},
    'exploration': {'episodes': len(episodes), 'transitions': sum(len(e['decisions']) for e in episodes),
        'all_16_terminal_sequences_observed': True, 'target_episodes': sum(e['metrics']['target_removed_mm3']>0 for e in episodes),
        'two_target_episodes': sum(e['metrics']['target_removed_mm3']==2 for e in episodes),
        'exact_optimal_episodes': sum(abs(e['simulated_return']-1.1)<1e-10 for e in episodes),
        'near_optimal_1_098_episodes': sum(abs(e['simulated_return']-1.098)<1e-10 for e in episodes),
        'first_positive_return_episode': next(e['name'] for e in episodes if e['simulated_return']>0),
        'sequences': [{'actions': list(actions), 'count': count} for actions,count in sequences.items()],
        'state_coverage': coverage, 'conditional_root_samples': conditioned},
    'evaluations': evaluations, 'gradients': gradients, 'source_sha256': source_checks,
    'limitations': ['Conditional empirical returns combine changing policies; they are not optimal Q values.',
        'Logged BC loss is measured before each update, not the final post-update loss.',
        'Final BC predictions are saved only at the root because it immediately stops.',
        'Shared encoder actor/critic objective gradients and their cosine are not recorded.',
        'No claim that shared-loss interference, insufficient iterations or STOP pressure is the causal explanation.',
        'Toy-fixture training coverage is not anatomy or clinical generalization.'],
    'next_diagnostic': 'One prospective zero-update teacher-fit readout: unchanged initial and final BC checkpoints on the same five independently replayed teacher observations; ten forward calls, per-state CE/rank/STOP probability and non-STOP ranking. No new search, labels, rollout collection or optimizer step.',
    'input_json_sha256': bindings}
out = Path(__file__).with_name('diagnostic.json')
out.write_text(json.dumps(result, indent=2, sort_keys=True, allow_nan=False)+'\n')
print(json.dumps({'path': str(out.relative_to(ROOT)), 'sha256': hashlib.sha256(out.read_bytes()).hexdigest(),
                  'episodes': len(episodes), 'terminal_sequences': len(sequences), 'exploration': result['exploration']['target_episodes'],
                  'gradient_ranges': gradients}, indent=2))
