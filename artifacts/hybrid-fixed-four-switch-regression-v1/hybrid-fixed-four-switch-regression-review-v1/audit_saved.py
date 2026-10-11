"""Saved fixed-four audit. Execute only after root confirms terminal output.

Stdlib JSON/source only; no arrays, weights, production imports or replay.
Inventories are restricted to each terminal STOP and its uncached counterpart.
"""
import argparse
import hashlib
import json
import math
from pathlib import Path
import stat
import subprocess

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
BASE = ROOT / 'build/hybrid-fixed-four-switch-regression-v1'
OUT = BASE / 'attempt-01'
SUP = BASE / 'attempt-01.supervision'
HEAD = 'e8fe1bb2ee78017df1cab629dd7ad3eb87a70390'
RELEASE = 'defcdf3802c153386756b181ebb79f6e89a9024b96222d3c313e41fb7d78d6e3'
INDEX = 'bd9783ac5f0bc9dd94a889cb99c1bcfae9f6008a3190056d2bef424c0028cb32'
SUBJECTS = ['ReMIND-002', 'ReMIND-015', 'ReMIND-018', 'ReMIND-045']
METHOD = 'balanced_ridge_switch_guard'
READS = {}

def need(ok, reason):
    if not ok:
        raise ValueError(reason)

def sha(data):
    return hashlib.sha256(data).hexdigest()

def semantic(value):
    return 'sha256:' + sha(json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode())

def raw(path, expected=None):
    path = Path(path)
    if not path.is_absolute():
        path = ROOT / path
    need(path.suffix in ('.json', '.py', '.log'), 'metadata_or_source_only')
    need(path.name not in ('balanced_ridge_top8.json', 'old_ridge_top8.json'), 'no_model_parameters')
    before = path.lstat()
    need(stat.S_ISREG(before.st_mode) and before.st_size <= 16 * 1024**2, 'bounded_regular_file')
    data = path.read_bytes()
    after = path.stat()
    need((before.st_size, before.st_mtime_ns) == (after.st_size, after.st_mtime_ns) and len(data) == before.st_size, 'changing_file')
    digest = sha(data)
    need(expected is None or expected == digest, 'file_hash:' + str(path))
    READS[str(path.relative_to(ROOT))] = {'bytes': len(data), 'sha256': digest}
    return data

def read(path, expected=None):
    return json.loads(raw(path, expected), parse_constant=lambda _: (_ for _ in ()).throw(ValueError('nonfinite_JSON')))

def close(a, b):
    need(math.isclose(a, b, rel_tol=1e-10, abs_tol=1e-7), 'numeric_disagreement')

def extent(directory):
    total = 0
    for path in directory.rglob('*'):
        info = path.lstat()
        need(not stat.S_ISLNK(info.st_mode), 'output_symlink')
        if stat.S_ISREG(info.st_mode):
            total += info.st_size
        else:
            need(stat.S_ISDIR(info.st_mode), 'nonregular_output')
    return total

def quality(outcome, reference):
    signs = {'target_removed_mm3': 1, 'total_reward': 1, 'normal_removed_mm3': -1,
        'cumulative_contacted_tissue_upper_bound_mm3': -1, 'currently_retained_contacted_tissue_upper_bound_mm3': -1,
        'complete_tool_path_length_mm': -1, 'nonstop_actions': -1, 'tool_changes': -1}
    result = {k: sign * (outcome[k] - reference[k]) >= -1e-7 for k, sign in signs.items()}
    result['same_full_target_denominator'] = outcome['total_reference_target_mm3'] == reference['total_reference_target_mm3']
    result['physical_probability_remains_null'] = outcome['clinical_deficit_probability'] is reference['clinical_deficit_probability'] is None
    result['unchanged_partial_contact_reward'] = outcome['partial_contact_reward_weight'] == reference['partial_contact_reward_weight'] == 0.
    return result

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--result-sha256', required=True)
    parser.add_argument('--parent-sha256', required=True)
    args = parser.parse_args()
    release = read(BASE / 'root-release.json', RELEASE)
    index = read(BASE / 'source-index.json', INDEX)
    need(release['expected_head'] == index['head'] == HEAD and release['execution_released'], 'exact_release')
    need(release['subjects'] == SUBJECTS and release['arms'] == [METHOD] and release['horizon'] == 24 and release['K'] == 8, 'fixed_denominator')
    need(release['fits'] == release['optimizer_updates'] == release['neural_forwards'] == 0 and release['SELECT_EVAL_closed'], 'inference_only')
    prior_index = read(ROOT / 'build/hybrid-train050-switch-guard-v1/source-index.json', '7560323e7316607a1615a23f51500c5bb8af79a8beeb256c7d91f0e2a9c8412f')
    changed = {k: v for k, v in index['files'].items() if prior_index['files'].get(k) != v}
    # Only new stage source/reference bindings are rehashed; unchanged canonical
    # mapping is reused and current tracked source is compared to executed HEAD.
    for path, pin in changed.items():
        raw(path, pin)
    delta = subprocess.check_output(['git', 'diff', '--name-only', HEAD, '--', 'src/resectionlab'], cwd=ROOT, text=True)
    need(not delta.strip(), 'canonical_source_diff_from_executed_HEAD')
    captured_head = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip()
    historical = read(release['training_result']['path'], release['training_result']['sha256'])
    train_parent = read(release['training_parent']['path'], release['training_parent']['sha256'])
    train_release = read(release['training_release']['path'], release['training_release']['sha256'])
    need(train_parent['status'] == 'complete' and train_parent['exit_code'] == 0 and not train_parent['cleanup_errors'] and not train_parent['remaining_owned_pids'], 'terminal_training')
    need(train_parent['result_sha256'] == release['training_result']['sha256'] and train_parent['release_sha256'] == release['training_release']['sha256'], 'training_receipt_joins')
    need(train_release['subjects'] == SUBJECTS and historical['ridge_file_sha256']['balanced_ridge_top8'] == release['ridge']['sha256'], 'exact_fitted_cohort_and_model')
    old = {(r['method'], r['subject']): r for r in historical['arms']}
    parent = read(SUP / 'receipt.json', args.parent_sha256)
    result = read(OUT / 'result.json', args.result_sha256)
    need(parent['result_sha256'] == args.result_sha256 and parent['release_sha256'] == result['release_sha256'] == RELEASE, 'terminal_joins')
    need(parent['status'] == 'complete' and parent['exit_code'] == 0 and parent['worker_termination_confirmed'] and not parent['remaining_owned_pids'] and not parent['cleanup_errors'] and parent['stop_reason'] is None, 'clean_terminal_worker')
    need(result['status'] == 'complete_fixed_four_switch_regression' and result['subjects'] == SUBJECTS and result['role'] == 'TRAIN', 'complete_fixed_four')
    need(result['source_visits'] == 8 and result['ridge_loads'] == 1 and result['ridge_fingerprint'] == release['ridge']['fingerprint'] and result['ridge_unchanged'], 'single_frozen_model')
    need(all(result[k] == 0 for k in ('fits', 'model_calls', 'optimizer_calls', 'checkpoint_loads', 'blocked_external_calls')), 'zero_forbidden_operations')
    need(not result['SELECT_EVAL_opened'] and not result['private_reference_loaded'] and result['cache_restored'], 'information_boundary')
    caps = release['caps']
    need(parent['caps'] == caps and parent['source_index'] == release['source_index'], 'exact_caps_and_source_index')
    for k in ('native_previews', 'native_transitions', 'ridge_calls'):
        need(result[k] <= caps[k], 'cap:' + k)
    need(result['elapsed_seconds'] <= caps['worker_seconds'] and parent['elapsed_seconds'] <= caps['parent_seconds'] and parent['sampled_peak_rss_bytes'] <= caps['sampled_rss_bytes'], 'resource_caps')
    output_bytes, supervision_bytes = extent(OUT), extent(SUP)
    need(output_bytes == parent['output_bytes'] and output_bytes <= caps['output_bytes'] and supervision_bytes <= caps['supervision_bytes'] and (SUP / 'worker.log').stat().st_size <= caps['log_bytes'], 'final_output_caps')
    declaration = read(SUP / 'declaration.json')
    need(declaration['head'] == HEAD and declaration['release_sha256'] == RELEASE and declaration['source_index'] == release['source_index'], 'launch_bindings')
    need('-I' in declaration['command'] and '-B' in declaration['command'] and any(x.startswith('pycache_prefix=') for x in declaration['command']), 'isolated_startup')
    budget = result['native_budget']
    need(budget['counting_reliable'] and budget['failure'] is None and budget['native_preview_entries'] == result['native_previews'] and budget['blocked_preview_attempts'] == 0, 'preview_guard_counts')
    arms = result['arms']
    need([(r['method'], r['subject']) for r in arms] == [(METHOD, s) for s in SUBJECTS] and all(r['complete'] and r['independent_accepted'] for r in arms), 'all_four_results_retained')
    visits = result['source_visits_detail']
    need([v['phase'] for v in visits] == ['planning.' + s for s in SUBJECTS] + ['replay.' + s for s in SUBJECTS], 'all_planning_before_any_replay')
    sealed = read(OUT / 'sealed-plans.json')
    need(result['all_plans_sealed_before_evaluation'] and list(sealed['plans']) == SUBJECTS and not sealed['private_reference_loaded'], 'all_four_sealed')
    totals = {'previews': 0, 'transitions': 0, 'ridge_calls': 0, 'extra_terminal_STOP_queries_online': 0}
    rows = []
    for row in arms:
        subject = row['subject']; folder = OUT / subject
        for path, pin in row['files'].items():
            raw(OUT / path, pin)
        plan = read(folder / 'plan.json')
        need(semantic({k: v for k, v in plan.items() if k != 'seal'}) == plan['seal'] and plan == sealed['plans'][subject], 'sealed_plan')
        need(plan['subject'] == subject and plan['method'] == METHOD and plan['ridge_hash'] == release['ridge']['fingerprint'] and plan['max_steps'] == 24, 'plan_identity')
        actions = plan['actions']; accounting = plan['accounting']; decisions = accounting['decisions']
        need(accounting['complete'] and not accounting['global_optimality_proven'] and len(actions) <= 24 and row['action_count'] == len(actions), 'finite_complete_plan')
        need([d['selected_action_id'] for d in decisions] == actions, 'decisions_actions')
        for dec in decisions:
            scores = dec['scores']
            need(scores[0]['action_id'] == 'STOP' and scores[0]['reward'] == 0 and max(scores, key=lambda v: v['reward'])['action_id'] == dec['selected_action_id'], 'unchanged_greedy_reward_ties')
            need(dec['all_queried_legal_actions_scored'] and len(scores) - 1 == dec['scored_nonstop_actions'] == dec['legal_nonstop_actions'], 'all_queried_legal_scores')
        metrics = read(folder / 'replay/episode-metrics.json'); evaluation = read(folder / 'replay/independent-episode.json')
        history = metrics['history']; outcomes = evaluation['outcomes']
        need(evaluation['accepted'] and evaluation['complete_episode'] and evaluation['geometry']['feasible'] and metrics['terminated'], 'saved_geometry_and_completion')
        need([h['action_id'] for h in history] == actions and evaluation['committed_history_hash'] == semantic(history) and outcomes == row['outcomes'], 'exact_replay_history')
        need(metrics['source_hash'] == evaluation['source_hash'] == plan['source_hash'] and metrics['decision_model_hash'] == evaluation['decision_model_hash'] == plan['decision_model_hash'], 'replay_identity')
        close(math.fsum(h['reward'] for h in history), outcomes['total_reward'])
        for key in ('target_removed_mm3', 'normal_removed_mm3', 'complete_tool_path_length_mm'):
            close(math.fsum(h[key] for h in history), outcomes[key])
        balanced, exhaustive = old['balanced_ridge_top8', subject], old['exhaustive', subject]
        prefix = 'balanced_ridge_top8/' + subject
        oldplan = read(ROOT / 'build/hybrid-proposal-balanced-v2/attempt-01' / (prefix + '/plan.json'), balanced['files'][prefix + '/plan.json'])
        oldmetrics = read(ROOT / 'build/hybrid-proposal-balanced-v2/attempt-01' / (prefix + '/replay/episode-metrics.json'), balanced['files'][prefix + '/replay/episode-metrics.json'])
        normalized = {**metrics, 'decision_model_hash': oldmetrics['decision_model_hash']}
        calculated = {'exact_actions_vs_balanced': actions == oldplan['actions'], 'exact_source_vs_balanced': plan['source_hash'] == oldplan['source_hash'],
            'exact_outcomes_vs_balanced': outcomes == balanced['outcomes'], 'exact_history_vs_balanced': history == oldmetrics['history'],
            'exact_metrics_except_policy_identity': normalized == oldmetrics, 'quality_gates_vs_saved_exhaustive': quality(outcomes, exhaustive['outcomes'])}
        calculated['no_quality_loss_vs_saved_exhaustive'] = all(calculated['quality_gates_vs_saved_exhaustive'].values())
        need(all(row['regression'][k] == v for k, v in calculated.items()), 'independent_regression_flags')
        public = read(folder / 'source/admitted-public-bindings.json'); replay_public = read(folder / 'replay/source/admitted-public-bindings.json')
        need(public == replay_public, 'unchanged_world_on_reload')
        binding, qc, protocol = public['source_binding'], public['qc'], public['protocol']
        world = result['public_worlds'][subject]
        need(semantic(binding) == world['binding'] and semantic(qc) == world['qc'] and semantic(protocol) == world['protocol'] and binding['source_hash'] == plan['source_hash'] == release['source_world_hashes'][subject], 'fixed_source_world_hashes')
        need(binding['subject'] == protocol['subject'] == subject and protocol['role'] == 'TRAIN' and 'development_transfer' not in protocol and protocol['initialization'] == 'public_world_search_only', 'original_four_admission_scope')
        need(protocol['max_optimizer_updates'] == protocol['max_policy_forwards'] == 0 and not protocol['private_reference_used'] and not protocol['split_changes'], 'zero_update_roles')
        need(all(v['status'] == 'complete' and all(v[k] == x for k, x in world.items()) for v in visits if v['subject'] == subject), 'all_visits_same_world')
        terminal_info = None
        if actions[-1] == 'STOP':
            number = len(actions) - 1
            terminal = read(folder / 'inventories' / f'state-{number:02d}.json')
            replay_terminal = read(folder / 'replay/inventories' / f'state-{number:02d}.json')
            need(terminal == replay_terminal and terminal['source_state_hash'] == decisions[-1]['source_state_hash'], 'same_terminal_inventory')
            q = terminal['shortlist']; ids = q['raw_ids']; initial = q['initial_selected_ids']; selected = q['selected_ids']
            need(q['fallback_completed'] and q['fallback_triggered'] and not q['current_tool_comparison_triggered'] and not any(x['nominal_reward'] > 0 for x in q['initial_nominal_scores']), 'full_nonpositive_terminal_expansion')
            need(initial == [ids[j] for j in sorted(range(len(ids)), key=lambda j: (-q['priority_scores'][j], j))[:8]], 'same_terminal_top8')
            need(q['additional_selected_ids'] == [x for x in ids if x not in set(initial)] and selected == initial + q['additional_selected_ids'] and len(set(selected)) == q['queried_count'] == q['raw_count'] == len(ids), 'terminal_queries_once_complete_catalog')
            extra = len(ids) - len(initial)
            # This exact reconciliation makes the claimed STOP-only overhead
            # testable without rereading every nonterminal inventory.
            terminal_only = all(row[phase]['counts']['fallback_triggered_states'] == 1 and row[phase]['counts'].get('current_tool_comparison_states', 0) == 0 and row[phase]['counts']['fallback_additional_previews'] == extra for phase in ('online_costs', 'replay_costs'))
            if terminal_only:
                need(row['online_previews'] == balanced['online_previews'] + extra and row['replay_previews'] == balanced['replay_previews'] + extra, 'exact_extra_STOP_query_counts')
            terminal_info = {'step': number, 'raw_count': len(ids), 'initial_queries': len(initial), 'extra_queries': extra, 'all_reported_extra_work_at_terminal_STOP': terminal_only,
                'STOP_scope': 'Full emitted catalog at this state only; not globally exhausted routes.'}
            totals['extra_terminal_STOP_queries_online'] += extra
        for phase, count_key in (('online_costs', 'online_previews'), ('replay_costs', 'replay_previews')):
            counts = row[phase]['counts']
            need(counts['native_previews'] == row[count_key] and counts['ridge_calls'] == counts['proposal_calls'] == counts['fallback_gate_states'] == len(actions), 'per_phase_counts')
            totals['ridge_calls'] += counts['ridge_calls']
        totals['previews'] += row['online_previews'] + row['replay_previews']
        totals['transitions'] += accounting['model_transition_calls'] + len(history)
        rows.append({'subject': subject, 'actions': len(actions), 'outcomes': outcomes, 'regression': calculated, 'terminal_inventory': terminal_info,
            'online_previews': row['online_previews'], 'replay_previews': row['replay_previews'], 'historical_balanced_previews': balanced['online_previews'],
            'source_preparation_seconds': row['source_preparation_seconds'], 'source_through_plan_seconds': row['source_through_plan_seconds'],
            'replay_source_preparation_seconds': row['replay_source_preparation_seconds'], 'replay_and_audit_seconds': row['replay_and_audit_seconds'],
            'online_costs': row['online_costs'], 'replay_costs': row['replay_costs'], 'source_domains': binding.get('source_and_simulated_domains'),
            'full_target_extent': public['supplied_goal_extent']})
    need(totals['previews'] == result['native_previews'] and totals['transitions'] == result['native_transitions'] and totals['ridge_calls'] == result['ridge_calls'], 'aggregate_costs')
    regression = all(all(r['regression'][k] for k in ('exact_actions_vs_balanced', 'exact_source_vs_balanced', 'exact_outcomes_vs_balanced', 'exact_metrics_except_policy_identity', 'no_quality_loss_vs_saved_exhaustive')) for r in rows)
    need(result['comparison']['regression_passed'] == regression and not result['comparison']['favorable_outcome_required_for_completion'], 'unselected_complete_regression_result')
    report = {'status': 'PASS_SAVED_FIXED_FOUR_REGRESSION_ACCOUNTING', 'regression_passed': regression,
        'release_sha256': RELEASE, 'source_index_sha256': INDEX, 'result_sha256': args.result_sha256, 'parent_sha256': args.parent_sha256,
        'runtime_source_delta': {'executed_head': HEAD, 'captured_head': captured_head, 'canonical_diff': [], 'changed_or_new_pins': changed, 'inherited_mapping_count': len(index['files']) - len(changed)},
        'fixed_subjects': SUBJECTS, 'role': 'TRAIN', 'frozen_model_identity': release['ridge'], 'plans_before_any_replay': True, 'rows': rows, 'totals': totals,
        'costs': {'worker_seconds': result['elapsed_seconds'], 'parent_seconds': parent['elapsed_seconds'], 'sampled_owned_peak_rss_bytes': parent['sampled_peak_rss_bytes'], 'output_bytes': output_bytes, 'supervision_bytes': supervision_bytes},
        'cleanup': {'exit_code': 0, 'reaped': True, 'remaining': [], 'errors': []},
        'limitations': ['Original fitted TRAIN cohort integration regression, not held-out transfer.',
            'No historical timing ratio is calculated; past exhaustive is a saved quality reference only.',
            'Terminal STOP covers emitted actions at that state, not arbitrary routes or surgery.',
            'Only saved metadata/source reviewed; geometry acceptance is the saved independent worker evaluation, not a reviewer replay.',
            'Source uncertainty, full target denominator and S union T/post-exposure simulation assumptions are preserved.',
            'Sampled memory can miss transient peaks; no per-case memory attribution.'],
        'read_boundary': {'patient_arrays': False, 'model_parameters': False, 'original_DICOM': False, 'production_imports': False, 'native_replay': False, 'nonterminal_inventories_read': False}, 'read_pins': READS}
    destination = HERE / 'audit.json'
    with destination.open('x') as stream:
        json.dump(report, stream, indent=2, allow_nan=False); stream.write('\n')
    print(json.dumps({'path': str(destination.relative_to(ROOT)), 'sha256': sha(destination.read_bytes()), 'regression_passed': regression, 'totals': totals}, indent=2))

if __name__ == '__main__':
    main()
