"""Saved JSON arithmetic only; no project, array, model or simulator imports."""
import hashlib
import itertools
import json
import math
from pathlib import Path

BASE = Path(__file__).resolve().parents[1]
ROOT = BASE.parents[1]
OUT = Path(__file__).resolve().parent
inputs = {}


def read(relative):
    path = BASE / relative
    raw = path.read_bytes()
    inputs[str(path.relative_to(ROOT))] = hashlib.sha256(raw).hexdigest()
    return json.loads(raw)


def semantic(value):
    return 'sha256:' + hashlib.sha256(json.dumps(value, sort_keys=True,
        separators=(',', ':'), allow_nan=False).encode()).hexdigest()


def close(a, b):
    assert math.isclose(a, b, rel_tol=1e-12, abs_tol=1e-10), (a, b)


run = 'ReMIND-008-run-v2/'
replay = 'ReMIND-008-replay-completion-v1/'
original = read(run+'result.json')
preflight = read(run+'preflight/result.json')
old_parent = read('ReMIND-008-run-v2.supervision/receipt.json')
completion = read(replay+'result.json')
parent = read('ReMIND-008-replay-completion-v1.supervision/receipt.json')
bundle = read(replay+'sealed-complete-plans.json')
teacher = read(run+'preflight/teacher-search.json')
comparison = read('ReMIND-008-postseal-evaluation-v2/annotation-comparison.json')
eval_results = [read('ReMIND-008-postseal-evaluation-v%d/result.json' % i) for i in (1, 2)]
eval_parents = [read('postseal-evaluation-v%d/parent-receipt.json' % i) for i in (1, 2)]
releases = [read('postseal-evaluation-v%d/release.json' % i) for i in (1, 2)]
assert original['status'] == preflight['status'] == 'failed_or_unresolved'
assert preflight['native_budget']['failure'] == 'native_preview_entry_limit'
assert completion['original_attempt_status_preserved'] == 'failed_or_unresolved'
assert parent['status'] == 'complete' and parent['exit_code'] == 0
assert old_parent['status'] == 'failed' and old_parent['exit_code'] == 1
for receipt in (old_parent, parent):
    assert receipt['worker_termination_confirmed'] and not receipt['final_owned_pids'] and not receipt['cleanup_errors']
    assert receipt['elapsed_seconds'] < receipt['caps']['hard_total_wall_seconds']
    assert receipt['sampled_peak_rss_bytes'] < receipt['caps']['sampled_owned_tree_rss_bytes']
for receipt, path in ((old_parent, run+'result.json'), (parent, replay+'result.json')):
    assert receipt['result_sha256'] == inputs[str((BASE/path).relative_to(ROOT))]
assert completion['sealed_bundle_sha256'] == comparison['sealed_file_sha256'] == inputs[str((BASE/(replay+'sealed-complete-plans.json')).relative_to(ROOT))]
assert bundle['expected_methods'] == [p['method'] for p in bundle['plans']] == ['SEARCH', 'IL', 'RL']
assert bundle['patient_context']['role'] == 'TRAIN' and bundle['patient_context']['patient_group'] == 'ReMIND:008'
updates = {m: read(run+'preflight/'+m+'/update.json') for m in ('IL','RL')}
assert sum(u['optimizer_updates'] for u in updates.values()) == preflight['optimizer_updates'] == 2
assert {u['before_parameter_hash'] for u in updates.values()} == {preflight['initial_parameter_hash']}
for m, update in updates.items():
    assert update['method'] == m and update['completed_updates'] == 1 and update['parameters_changed']
    assert not update['population_training'] and not update['patient_adaptation']
assert completion['optimizer_updates'] == completion['policy_forward_calls'] == completion['search_calls'] == completion['checkpoint_loads'] == completion['private_array_reads'] == 0
assert sum(v.get('policy_forward_calls', 0) for v in preflight['costs'].values()) == preflight['total_policy_forward_calls'] == 15
assert sum(v['native_preview_entries'] for v in preflight['costs'].values()) == preflight['native_budget']['native_preview_entries'] == 2048
assert completion['native_budget']['native_preview_entries'] == 408 and completion['replay_transition_calls'] == 6
for key, value in completion['original_costs'].items():
    expected = {'owned_wall_seconds': old_parent['elapsed_seconds'], 'sampled_peak_rss_bytes': old_parent['sampled_peak_rss_bytes'],
        'optimizer_updates': preflight['optimizer_updates'], 'policy_forward_calls': preflight['total_policy_forward_calls'], 'native_budget': preflight['native_budget']}[key]
    assert value == expected
stats = teacher['accounting']
assert teacher['actions'] == ['STOP'] and not stats['call_cap_reached'] and not stats['time_cap_reached']
assert stats['completed_layers'] == 6 and sum(row['model_transition_calls'] for row in stats['layers']) == stats['model_transition_calls'] == 211
assert stats['negative_prefixes_evaluated'] == 211 and stats['beam_pruned_prefixes'] == 165
summaries = []
for row, score in zip(bundle['plans'], comparison['methods']):
    method, plan, history, audit = row['method'], row['plan'], row['replayed_history'], row['independent_geometry']
    relative = {'SEARCH': run+'preflight/SEARCH/', 'IL': run+'preflight/IL/greedy/', 'RL': run+'preflight/RL/greedy/'}[method]
    saved_plan = read(relative+'plan.json')
    native = read(replay+'RL-native-replay.json' if method == 'RL' else relative+'native-replay.json')
    assert saved_plan['plan'] == plan and saved_plan['plan_seal'] == row['plan_seal'] == semantic(plan)
    assert completion['plan_seals'][method] == row['plan_seal'] == score['plan_seal']
    assert score['method'] == method and native['metrics']['history'] == history and native['independent_geometry'] == audit
    assert plan['source_hash'] == bundle['source_hash'] and plan['context_hash'] == semantic(bundle['patient_context'])
    assert plan['decision_model_hash'] == bundle['decision_model_hash'] and plan['initial_observation_hash'] == bundle['initial_observation_hash']
    assert len(history) == len(plan['history']) and plan['actions'] == [r['action_id'] for r in history]
    strip = lambda rows: [{k:v for k,v in r.items() if k != 'outcome_scope'} for r in rows]
    assert strip(history) == strip(plan['history'])
    assert audit['accepted'] and audit['complete_episode'] and audit['geometry']['feasible'] and audit['geometry']['complete_tool_checked']
    assert audit['committed_history_hash'] == semantic(history)
    metrics = native['metrics']; outcomes = audit['outcomes']
    for key in ('total_reward','target_removed_mm3','normal_removed_mm3'):
        row_key = 'reward' if key == 'total_reward' else key
        close(sum(r[row_key] for r in history), metrics[key]); close(metrics[key], outcomes[key])
    close(sum(r['complete_tool_path_length_mm'] for r in history), outcomes['complete_tool_path_length_mm'])
    removed = [tuple(index) for r in history for index in r.get('removed_indices_native', [])]
    assert len(removed) == len(set(removed))
    close(len(removed)*audit['geometry']['source_voxel_volume_mm3'], metrics['simulated_removed_volume_mm3'])
    assert not audit['target_access_success'] and metrics['target_removed_mm3'] == 0
    if method != 'SEARCH':
        assert plan['learning_updates'] == 1 and plan['parameter_hash'] == updates[method]['after_parameter_hash']
        assert plan['terminal_reason'] == 'HORIZON' and len(history) == 6
    else:
        assert plan['learning_updates'] == 0 and plan['actions'] == ['STOP']
    contact = score['contacts']; whole = contact['whole_tool']
    assert contact['action_count'] == outcomes['nonstop_actions'] and contact['capsule_count'] == 2*outcomes['nonstop_actions']
    assert whole['positive_reference_cells'] == 0
    if method != 'SEARCH':
        assert whole['unknown_reference_cells'] == whole['touched_reference_cells'] > 0
        assert whole['outside_reference_fov'] and not whole['annotation_coverage_complete_for_sweep']
        assert whole['annotated_positive_encounter'] is None
    summaries.append({'method': method, 'plan_seal': row['plan_seal'], 'actions': len(history),
        'nonstop_actions': outcomes['nonstop_actions'], 'terminal': plan['terminal_reason'],
        'reward': metrics['total_reward'], 'supplied_target_removed_mm3': metrics['target_removed_mm3'],
        'outside_supplied_target_removed_mm3': metrics['normal_removed_mm3'], 'removed_cells': len(removed),
        'complete_tool_path_mm': outcomes['complete_tool_path_length_mm'], 'whole_tool_reference': whole})
assert eval_results[0]['status'] == 'failed' and eval_results[0]['error'] == 'ValueError: orthogonal_reference_grid_required'
assert eval_results[1]['status'] == 'complete' and eval_parents[1]['exit_code'] == 0
assert eval_results[1]['comparison_sha256'] == inputs[str((BASE/'ReMIND-008-postseal-evaluation-v2/annotation-comparison.json').relative_to(ROOT))]
assert all(r['private_array_reads'] == 2 for r in eval_results)
assert all(p['child_reaped'] and p['stop_reason'] is None for p in eval_parents)
assert releases[0]['reference']['mask'] == releases[1]['reference']['mask']
assert releases[0]['reference']['coverage'] == releases[1]['reference']['coverage']
assert releases[1]['reference']['affine_ras_mm'] == bundle['source_grid']['affine_ras_mm']
assert releases[1]['reference']['source_affine_ras_mm_retained'] == releases[0]['reference']['affine_ras_mm']
for release in releases:
    for row in release['required_completed_files']:
        path = ROOT / row['path']; assert path.suffix in ('.json','.py')
        digest = hashlib.sha256(path.read_bytes()).hexdigest(); inputs[row['path']] = digest
        assert digest == row['sha256']
    assert release['bundle']['sha256'] == comparison['sealed_file_sha256']
old, new = releases[0]['reference']['affine_ras_mm'], releases[1]['reference']['affine_ras_mm']
corners = itertools.product(*[(-.5, size-.5) for size in bundle['source_grid']['shape']])
delta = max(math.sqrt(sum(sum((new[i][j]-old[i][j])*point[j] for j in range(3))**2 for i in range(3))) for point in corners)
close(delta, releases[1]['reference']['grid_reconciliation']['maximum_cell_corner_displacement_mm'])
assert delta < 1e-6
result = {'status': 'PASS_saved_identity_and_arithmetic_only', 'methods': summaries,
    'learning': {'IL_updates': 1,'RL_updates': 1,'same_initial_parameter_hash': preflight['initial_parameter_hash'],
        'policy_forwards': 15, 'population_training': False, 'transfer_evaluated': False},
    'search': stats, 'cost_retention': {'failed_run_seconds': old_parent['elapsed_seconds'],
        'continuation_seconds': parent['elapsed_seconds'], 'failed_evaluation_seconds': eval_parents[0]['elapsed_seconds'],
        'completed_evaluation_seconds': eval_parents[1]['elapsed_seconds'],
        'sum_these_four_owned_seconds': old_parent['elapsed_seconds']+parent['elapsed_seconds']+sum(p['elapsed_seconds'] for p in eval_parents),
        'native_preview_entries_including_restart': 2456,'blocked_preview_attempts': 1,
        'original_native_transition_attempts': sum(v.get('native_transition_calls',0) for v in preflight['costs'].values()),
        'restarted_RL_transitions': 6,'private_array_reads_across_both_evaluations': 4,
        'sampled_peak_RSS_bytes': [old_parent['sampled_peak_rss_bytes'],parent['sampled_peak_rss_bytes']]+[p['sampled_peak_group_rss_bytes'] for p in eval_parents],
        'excluded': 'Earlier attempts, acquisition/QC/conversion/preparation outside these four parents and this saved-only audit; profiled run is not a latency benchmark'},
    'reference': {'source_kind': comparison['source_kind'], 'label_name': comparison['label_name'],
        'reference_mask_hash': comparison['reference_mask_hash'],'reference_coverage_hash': comparison['reference_coverage_hash'],
        'mask_or_coverage_file_changed_between_attempts': False,'same_presealed_native_grid': True,
        'computed_maximum_corner_displacement_mm': delta,'annotation_accuracy_established': False,
        'moving_methods_ventricular_encounters': 'UNKNOWN: all touched in-grid cells uncovered and sweeps extend outside FOV'},
    'audit_scope': 'stdlib saved JSON/source reads only; no array/DICOM/checkpoint reads, simulator/model/search/replay calls or geometry recomputation',
    'inputs_sha256': inputs}
(OUT/'audit.json').write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
print(json.dumps({'status':result['status'],'methods':[{k:v for k,v in m.items() if k!='whole_tool_reference'} for m in summaries],
    'inputs':len(inputs),'owned_seconds':result['cost_retention']['sum_these_four_owned_seconds']},indent=2))
