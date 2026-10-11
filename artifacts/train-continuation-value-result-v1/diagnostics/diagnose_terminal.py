"""Terminal-only scalar diagnosis; stdlib, no model/native/array/weight access.

This is a projection of authenticated saved evidence, not a second outcome audit.
Every case remains represented, including missing/incomplete endpoints. Do not
run until root supplies the three final hashes; never read progress.json.
"""
import argparse
from collections import Counter
import hashlib
import json
import math
from pathlib import Path
import signal
import time

HERE = Path(__file__).resolve().parent
STAGE = HERE.parent
SUBJECTS = ('ReMIND-002', 'ReMIND-015', 'ReMIND-018', 'ReMIND-045')
METHODS = ('GREEDY', 'IL64', 'RL8', 'Q_GUIDED')
READS = {}
MAX_OUTPUT = 64 * 1024
OUTCOMES = ('target_removed_mm3', 'normal_removed_mm3', 'total_reward',
            'complete_tool_path_length_mm', 'cumulative_contacted_tissue_upper_bound_mm3',
            'nonstop_actions')


def need(condition, message):
    if not condition:
        raise ValueError(message)


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()


def semantic(value):
    return 'sha256:' + hashlib.sha256(canonical(value)).hexdigest()


def read(path, digest):
    need(not path.is_symlink() and path.suffix == '.json' and path.is_file(), 'Regular pinned JSON only')
    need(0 < path.stat().st_size <= 16 * 1024**2, 'Bounded saved JSON input')
    raw = path.read_bytes()
    actual = hashlib.sha256(raw).hexdigest()
    need(actual == digest, 'Pinned terminal file changed: ' + str(path))
    READS[str(path.relative_to(STAGE))] = {'sha256': actual, 'bytes': len(raw)}
    return json.loads(raw)


def finite(value):
    need(type(value) in (int, float) and math.isfinite(value), 'Finite saved scalar required')
    return value


def physical(history):
    return {key: history[key] for key in ('action_id', 'tool_id', 'entry_mm', 'tip_mm', 'physical_start_mm')
            if key in history}


def route_summary(arm):
    if not arm or arm.get('complete') is not True:
        return {'available_complete_endpoint': False, 'saved_status': None if not arm else arm.get('status')}
    outcomes = arm['audit']['outcomes']
    history = arm['history']; decisions = arm['decisions']; actions = arm['actions']
    need(len(history) == len(decisions) == len(actions) <= 24, 'Bounded full route denominator')
    tools = [row['tool_id'] for row in history if row['action_id'] != 'STOP']
    return {'available_complete_endpoint': True, 'independent_accepted': arm['audit']['accepted'],
        'outcomes': {key: finite(outcomes[key]) for key in OUTCOMES},
        'terminal_reason': arm['terminal_reason'], 'actions': len(actions),
        'tool_changes': sum(a != b for a, b in zip(tools, tools[1:])),
        'history_hash': arm['history_hash'], 'nominal_history_hash': arm['nominal_history_hash'],
        'physical_route_hash': semantic([physical(row) for row in history]),
        'planning_previews': arm['planning_costs']['native_preview_entries'],
        'planning_transitions': arm['planning_costs'].get('native_transition_calls'),
        'planning_wall_seconds': arm['planning_costs']['complete_wall_seconds'],
        'replay_wall_seconds': arm['replay_costs']['complete_wall_seconds']}


def ranking_choices(arm):
    if not arm or arm.get('complete') is not True:
        return {'available': False, 'decisions': [], 'reason_counts': {}}
    plan = arm['planning']; rows = plan['decisions']
    need(len(rows) == len(arm['actions']) <= 24, 'Q decisions retain complete denominator')
    result = []; reasons = Counter(); totals = Counter()
    for ordinal, (row, executed) in enumerate(zip(rows, arm['decisions']), 1):
        scores = row['scores']; chosen = row['selected_action_id']; reasons[row['selection_reason']] += 1
        need(row['all_current_legal_actions_scored'] is True and scores[0]['action_id'] == 'STOP',
             'Complete current-state nominal score inventory')
        ids = [s['action_id'] for s in scores]
        need(len(ids) == len(set(ids)) and chosen in ids, 'Unique chosen candidate membership')
        chosen_index = ids.index(chosen)
        rewards = [finite(s['reward']) for s in scores]
        greedy_index = max(range(len(scores)), key=lambda i: rewards[i])
        positive = [i for i in range(1, len(scores)) if rewards[i] > 0.]
        immediate_order = sorted(range(len(scores)), key=lambda i: (-rewards[i], i))
        positive_q = []; q_order = []; blocked_optimism = 0
        if positive:
            predicted = [finite(s['predicted_q']) for s in scores]
            need(predicted[0] == 0., 'Fixed STOP0')
            positive_q = [i for i in positive if predicted[i] > 0.]
            q_order = sorted(positive, key=lambda i: (-predicted[i], i))
            blocked_optimism = sum(predicted[i] > 0. and rewards[i] <= 0. for i in range(1, len(scores)))
            expected = max(positive_q, key=lambda i: predicted[i]) if positive_q else greedy_index
            expected_reason = ('positive_q_among_actual_positive' if positive_q
                               else 'actual_greedy_fallback_no_positive_q')
        else:
            need(all(s['predicted_q'] is None for s in scores), 'No unnecessary Q batch on forced STOP')
            expected = 0; expected_reason = 'no_actual_positive_action'
        need(chosen_index == expected and row['selection_reason'] == expected_reason,
             'Saved choice must follow the released guarded rule')
        need(chosen == executed['action_id'] and rewards[chosen_index] == executed['reward'],
             'Committed choice/reward join')
        totals['actual_positive_candidates'] += len(positive)
        totals['positive_Q_actual_positive_candidates'] += len(positive_q)
        totals['positive_Q_actual_nonpositive_candidates_guarded_out'] += blocked_optimism
        totals['differs_from_same_state_greedy'] += int(chosen_index != greedy_index)
        result.append({'step': ordinal, 'reason': row['selection_reason'],
            'selected': chosen, 'same_state_greedy': ids[greedy_index],
            'selected_immediate_rank': immediate_order.index(chosen_index) + 1,
            'selected_reward': rewards[chosen_index], 'greedy_reward': rewards[greedy_index],
            'immediate_regret': rewards[greedy_index] - rewards[chosen_index],
            'selected_Q': scores[chosen_index]['predicted_q'],
            'greedy_Q': scores[greedy_index]['predicted_q'],
            'greedy_Q_rank_among_productive': (q_order.index(greedy_index) + 1) if positive else None,
            'productive': len(positive), 'positive_Q_productive': len(positive_q)})
    return {'available': True, 'reason_counts': dict(reasons), 'totals': dict(totals),
        'decision_count': len(result), 'Q_batches': plan['q_batches_completed'],
        'STOP_is_guard_forced_not_learned': True,
        'sum_local_immediate_regret': sum(r['immediate_regret'] for r in result), 'decisions': result}


def compare_routes(greedy, q):
    if not (greedy and q and greedy.get('complete') is True and q.get('complete') is True):
        return {'available': False}
    g, h = greedy['decisions'], q['decisions']
    common = 0; first = None
    for index in range(min(len(g), len(h))):
        need(g[index]['source_state_hash'] == h[index]['source_state_hash'] and
             g[index]['observation_hash'] == h[index]['observation_hash'],
             'Equal action prefix must reach exact same public state before divergence')
        if g[index]['action_id'] != h[index]['action_id']:
            first = {'step': index + 1, 'common_source_state_hash': h[index]['source_state_hash'],
                'common_observation_hash': h[index]['observation_hash'],
                'GREEDY': physical(greedy['history'][index]), 'Q_GUIDED': physical(q['history'][index]),
                'GREEDY_immediate_reward': g[index]['reward'], 'Q_GUIDED_immediate_reward': h[index]['reward']}
            break
        need(g[index]['result_state_hash'] == h[index]['result_state_hash'] and
             g[index]['reward'] == h[index]['reward'], 'Equal physical decision must match native transition')
        common += 1
    same = len(g) == len(h) == common
    if first is None and not same:
        first = {'step': common + 1, 'reason': 'one_saved_route_ends_before_other'}
    x, y = greedy['audit']['outcomes'], q['audit']['outcomes']
    gstates = {r['source_state_hash']: i + 1 for i, r in enumerate(g)}
    return {'available': True, 'same_complete_route': same, 'same_prefix_decisions': common,
        'first_divergence': first, 'outcome_delta_Q_minus_GREEDY': {k: finite(y[k]) - finite(x[k]) for k in OUTCOMES},
        'later_Q_states_also_on_GREEDY_route': [{'Q_step': i + 1, 'GREEDY_step': gstates[r['source_state_hash']]}
            for i, r in enumerate(h) if i >= common + 1 and r['source_state_hash'] in gstates],
        'causal_limit': 'After first divergence, same-state greedy scores are local alternatives, not executed counterfactual continuation returns.'}


def run(args):
    started = time.perf_counter()
    launcher = read(STAGE/'run-01/launcher.json', args.launcher_sha256)
    need(launcher.get('status') != 'running' and launcher.get('worker_exit_code') is not None,
         'An actual terminal parent receipt is required')
    report = read(STAGE/'run-01/report.json', args.report_sha256)
    index = read(STAGE/'runner/source-index.json', args.index_sha256)
    need(index['execution_released'] is True and
         launcher['source_index_sha256'] == report['source_index_sha256'] == args.index_sha256 and
         launcher['report_sha256'] == args.report_sha256, 'Exact terminal release/report binding')
    need(tuple(report['config']['TRAIN']) == SUBJECTS and tuple(report['config']['methods']) == METHODS,
         'All original four subjects and four fixed comparator arms')
    lifecycle = launcher.get('owned_lifecycle') or {}
    clean = (launcher['status'] == 'completed' and launcher['worker_exit_code'] == 0
        and launcher.get('stop_reason') is None and lifecycle.get('verified') is True
        and lifecycle.get('child_reaped') is True and not lifecycle.get('remaining_owned_pids')
        and not lifecycle.get('errors'))
    cases = {}
    for subject in SUBJECTS:
        arms = report.get('cases', {}).get(subject, {})
        before = report.get('initial_case_weighted_MSE', {}).get(subject)
        after = report.get('final_case_weighted_MSE', {}).get(subject)
        mse = {'before': before, 'after': after, 'delta_after_minus_before': None, 'fraction_reduction': None}
        if before is not None and after is not None:
            finite(before); finite(after)
            mse.update(delta_after_minus_before=after-before,
                       fraction_reduction=None if before == 0 else (before-after)/before)
        cases[subject] = {'training_case_weighted_MSE': mse,
            'endpoints': {m: route_summary(arms.get(m)) for m in METHODS},
            'guarded_Q_choices': ranking_choices(arms.get('Q_GUIDED')),
            'route_comparison': compare_routes(arms.get('GREEDY'), arms.get('Q_GUIDED'))}
    fit = report.get('head_fit')
    gate = report.get('useful_criterion')
    complete = all(all(r['available_complete_endpoint'] for r in c['endpoints'].values()) for c in cases.values())
    changed = [s for s, c in cases.items() if c['route_comparison'].get('available') and
               c['route_comparison']['same_complete_route'] is False]
    checks = {} if not gate else gate.get('checks', {})
    regressions = {s: [k for k, passed in row.items() if passed is False] for s, row in checks.items()
                   if any(v is False for v in row.values())}
    result = {'schema': 'terminal-logged-Q-causal-projection-v1',
        'scope': 'Saved scalars only; no new inference, geometry, learning, or counterfactual execution.',
        'terminal_supervision_clean': clean, 'scientific_report_status': report.get('status'),
        'all_four_by_four_endpoints_complete': complete,
        'recorded_useful_criterion_unchanged': gate, 'changed_complete_routes': changed,
        'recorded_gate_regressions': regressions, 'cases': cases,
        'fit': {'updates': report.get('head_optimizer_updates'),
            'weighted_MSE_before_first_update': None if not fit else fit['updates'][0]['weighted_mse_before_update'],
            'weighted_MSE_after_final_update': None if not fit else fit['final_weighted_mse'],
            'failed_fit_accounting': report.get('failed_head_fit_accounting'),
            'negative_immediate_positive_recorded_continuation_examples': report.get('corpus', {}).get(
                'negative_immediate_positive_recorded_continuation'),
            'MSE_scope': 'Fit corpus only, not held-out generalization or clinical performance.'},
        'full_costs': {'parent_seconds': launcher.get('elapsed_seconds'),
            'parent_peak_group_RSS_bytes': launcher.get('sampled_peak_group_rss_bytes'),
            'worker_seconds': report.get('worker_seconds'), 'native_budget': report.get('native_budget'),
            'frozen_policy_forwards': report.get('total_frozen_policy_forwards'),
            'head_inference_costs': report.get('head_inference_costs'),
            'scope': 'Upfront36-trace reconstruction and64 updates remain paid. Arm scopes do not replace end-to-end totals; do not sum nested inclusive phases.'},
        'interpretation_limits': [
            'Fitted targets are behavior-supported logged-action continuations, not optimal Q values.',
            'This corpus supplies no successful immediate-negative preparation continuation.',
            'Native positive-action guard determines stopping and prevents negative moves; no learned stopping or preparation claim.',
            'Local greedy regret is not a decomposition of final return loss after trajectories diverge.',
            'Same-TRAIN outcomes and one ordered runtime comparison do not establish transfer, clinical safety, or repeatable speedup.'],
        'input_files': READS, 'diagnostic_seconds': time.perf_counter()-started}
    return result


def main():
    parser = argparse.ArgumentParser()
    for name in ('report-sha256', 'launcher-sha256', 'index-sha256'):
        parser.add_argument('--'+name, required=True)
    parser.add_argument('--output', default='terminal-diagnosis-01.json')
    args = parser.parse_args()
    for value in (args.report_sha256, args.launcher_sha256, args.index_sha256):
        need(len(value) == 64 and all(c in '0123456789abcdef' for c in value), 'Exact SHA256 required')
    target = HERE/args.output
    need(target.parent.resolve() == HERE.resolve() and target.suffix == '.json' and not target.exists(),
         'Fresh compact output within this diagnostics directory required')
    signal.signal(signal.SIGALRM, lambda *_: (_ for _ in ()).throw(TimeoutError('30-second scalar diagnostic cap')))
    signal.alarm(30)
    value = run(args)
    raw = canonical(value) + b'\n'
    need(len(raw) <= MAX_OUTPUT, '64KiB compact diagnostic output cap')
    # Recheck all input hashes before publishing an immutable projection.
    for name, metadata in READS.items():
        need(hashlib.sha256((STAGE/name).read_bytes()).hexdigest() == metadata['sha256'], 'Input changed while read')
    with target.open('xb') as stream:
        stream.write(raw)
    signal.alarm(0)
    print(json.dumps({'output': str(target), 'bytes': len(raw), 'sha256': hashlib.sha256(raw).hexdigest(),
        'terminal_supervision_clean': value['terminal_supervision_clean'],
        'changed_complete_routes': value['changed_complete_routes']}))


if __name__ == '__main__':
    main()
