"""Independent saved-data calculations; never construct a patient trace/model.

No imports from the producer/runner/native/torch. Targets use an explicit
quadratic prefix scan (episodes have at most24 decisions), not the producer's
likely backward recurrence. A caller must first authenticate file bytes against
the frozen experiment's input index. This validates what the saved evidence says;
it does not replace live source/context/observation admission or native replay.
"""
import hashlib
import json
import math

SUBJECTS = ('ReMIND-002', 'ReMIND-015', 'ReMIND-018', 'ReMIND-045')


def need(condition, message):
    if not condition:
        raise ValueError(message)


def semantic(value):
    return 'sha256:'+hashlib.sha256(json.dumps(value, sort_keys=True,
        separators=(',', ':'), allow_nan=False).encode()).hexdigest()


def best_recorded_prefix_returns(rewards):
    """Best contiguous recorded continuation, permitting immediate STOP=0.

The result is a realized behavior-supported target, not an optimal value, a
prediction about unrecorded actions, or a continuation beyond the saved horizon.
Identical observations in different traces may legitimately have different
labels unless a separately declared equivalence/pooling rule is introduced.
"""
    need(1 <= len(rewards) <= 24, 'Complete bounded episode required')
    need(all(type(x) in (int, float) and math.isfinite(x) for x in rewards),
         'Finite recorded rewards required')
    return [max([0.]+[math.fsum(rewards[i:j]) for j in range(i+1, len(rewards)+1)])
            for i in range(len(rewards))]


def mandatory_logged_action_returns(rewards):
    """Take the recorded first action, then choose a recorded stop prefix.

Unlike a state value, current negative reward is not clipped away. Each allowed
prefix contains at least the recorded first action. This independent direct
enumeration equals r_i + max(0, best remaining recorded prefix from i+1).
It does not imply any value for unlogged actions or optimal counterfactuals.
"""
    best_recorded_prefix_returns(rewards)  # input checks only
    return [max(math.fsum(rewards[i:j]) for j in range(i+1, len(rewards)+1))
            for i in range(len(rewards))]


def strip_outcome_scope(history):
    # This is the sole planning-versus-authoritative history projection in the
    # retained canonical collector. All geometry/reward/other fields remain.
    return [{k: v for k, v in row.items() if k != 'outcome_scope'} for row in history]


def validate_saved_episode(trace, sealed_plan, replay, *, subject, family, update):
    """Join one already hash-authenticated teacher or RL episode and labels.

Returns only scalar/provenance rows, never observations, coordinates or a live
PatientTrainingTrace capability. Empty, truncated, mismatched or post-terminal
records are rejected. Horizon24 is an actual task terminal; no bootstrap follows.
"""
    need(subject in SUBJECTS, 'Only the fixed four TRAIN subjects')
    need((family == 'teacher' and update is None) or
         (family == 'RL' and type(update) is int and 1 <= update <= 8), 'Fixed corpus family/update')
    decisions, metrics = trace['decisions'], trace['metrics']
    history, plan = metrics['history'], sealed_plan['plan']
    need(plan['max_steps'] == 24 and 1 <= len(decisions) <= 24, 'Original task horizon24')
    need(len(history) == len(decisions) == metrics['steps'], 'Complete decision/history count')
    need(metrics['terminated'] is True and metrics['planning_estimator_only'] is True, 'Owned nominal complete trace')
    need(decisions[-1]['terminated'] is True and all(d['terminated'] is False for d in decisions[:-1]), 'Single final termination')
    actions = [d['action_id'] for d in decisions]
    need(actions == plan['actions'] == [h['action_id'] for h in history], 'Exact recorded actions')
    if actions[-1] == 'STOP':
        need('STOP' not in actions[:-1] and plan['terminal_reason'] == 'STOP', 'Only final STOP')
    else:
        need('STOP' not in actions and len(actions) == 24 and plan['terminal_reason'] == 'HORIZON', 'Actual horizon termination')
    need(plan['context_hash'] == trace['context_hash'], 'Same admitted context')
    need(plan['source_hash'] == metrics['source_hash'] and plan['decision_model_hash'] == metrics['decision_model_hash'], 'Same source/world')
    need(plan['history'] == history and semantic(plan) == sealed_plan['plan_seal'], 'Sealed exact plan')
    need(plan['initial_observation_hash'] == decisions[0]['observation_hash'], 'Initial observation binding')
    behavior = decisions[0]['behavior_parameter_hash']
    need((family == 'teacher' and behavior is None) or (family == 'RL' and isinstance(behavior, str) and behavior.startswith('sha256:')), 'Recorded behavior provenance')
    rewards, observations = [], []
    for i, (d, h) in enumerate(zip(decisions, history)):
        need(d['step'] == i and d['behavior_parameter_hash'] == behavior, 'Ordered fixed-behavior collection')
        ids, mask = d['action_ids'], d['action_mask']
        need(ids and ids[0] == 'STOP' and len(ids) == len(mask) and len(set(ids)) == len(ids), 'Ordered unique action catalog')
        need(all(type(x) is bool for x in mask) and mask[0], 'Public legal STOP')
        need(d['action_id'] in ids and mask[ids.index(d['action_id'])], 'Chosen recorded action legal')
        need(type(d['reward']) in (int, float) and math.isfinite(d['reward']) and d['reward'] == h['reward'], 'Recorded/history reward exact')
        need(isinstance(d['observation_hash'], str) and d['observation_hash'].startswith('sha256:'), 'Saved observation binding')
        if d['action_id'] == 'STOP':
            need(d['reward'] == 0., 'Unchanged immediate STOP reward')
        rewards.append(d['reward']); observations.append(d['observation_hash'])
    expected_seal = semantic({'context': trace['context_hash'], 'observations': observations,
                              'history': history, 'behavior_parameter_hash': behavior})
    need(trace['trace_seal'] == expected_seal, 'Trace seal authenticates full saved chain')
    need(abs(math.fsum(rewards)-metrics['total_reward']) <= 1e-10, 'Recorded reward sum')
    audit, actual = replay['independent_geometry'], replay['metrics']
    need(audit['accepted'] is True and audit['complete_episode'] is True
         and audit['geometry']['feasible'] is True and audit['geometry']['failures'] == [], 'Complete accepted independent certificate')
    need(audit['geometry']['complete_tool_checked'] and audit['geometry']['frontier_checked'], 'Full-tool/exposure scope')
    need(actual['terminated'] is True and actual['planning_estimator_only'] is False, 'Authoritative complete replay')
    need(strip_outcome_scope(actual['history']) == strip_outcome_scope(history), 'Exact physical/reward replay; only scope differs')
    need(audit['committed_history_hash'] == semantic(actual['history']), 'Independent executed-history binding')
    for name in ('source_hash', 'decision_model_hash', 'reference_hash'):
        need(audit[name] == actual[name] == metrics[name], 'Episode identity '+name)
    for name in ('total_reward', 'target_removed_mm3', 'normal_removed_mm3'):
        need(abs(audit['outcomes'][name]-metrics[name]) <= 1e-10, 'Certificate outcome '+name)
    values = best_recorded_prefix_returns(rewards)
    logged_values = mandatory_logged_action_returns(rewards)
    return {'subject': subject, 'family': family, 'update': update,
            'source_hash': metrics['source_hash'], 'decision_model_hash': metrics['decision_model_hash'],
            'context_hash': trace['context_hash'], 'trace_seal': trace['trace_seal'], 'plan_seal': sealed_plan['plan_seal'],
            'terminal_reason': plan['terminal_reason'], 'steps': len(rewards), 'recorded_return': math.fsum(rewards),
            'labels': [{'step': i, 'observation_hash': observations[i], 'action_id': actions[i],
                        'recorded_reward': rewards[i], 'best_recorded_prefix_return': values[i],
                        'mandatory_logged_action_return': logged_values[i]}
                       for i in range(len(rewards))]}


def validate_corpus(rows):
    expected = {(s, 'teacher', None) for s in SUBJECTS} | {(s, 'RL', u) for s in SUBJECTS for u in range(1, 9)}
    need(len(rows) == 36 and {(r['subject'], r['family'], r['update']) for r in rows} == expected, 'All4 teachers +32RL, no extras or exclusions')
    need(sum(r['steps'] for r in rows if r['family'] == 'teacher') == 29, 'All29 teacher states')
    need(sum(r['steps'] for r in rows if r['family'] == 'RL') == 468, 'All468 recordedRL states')
    return {'episodes': 36, 'teacher_decisions': 29, 'RL_decisions': 468, 'all_decisions': 497,
            'unique_observations': len({d['observation_hash'] for r in rows for d in r['labels']}),
            'positive_labels': sum(d['best_recorded_prefix_return'] > 0 for r in rows for d in r['labels']),
            'negative_mandatory_action_labels': sum(d['mandatory_logged_action_return'] < 0 for r in rows for d in r['labels']),
            'negative_immediate_positive_continuation': sum(d['recorded_reward'] < 0 and d['best_recorded_prefix_return'] > 0 for r in rows for d in r['labels'])}
