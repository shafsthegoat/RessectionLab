"""Stdlib-only audit of a terminal, explicitly pinned saved comparison.

No native imports, acquired arrays, reconstruction, replay, or execution.
"""
import argparse
from collections import defaultdict
import hashlib
import json
from pathlib import Path


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def semantic(value):
    return 'sha256:' + hashlib.sha256(json.dumps(value, sort_keys=True,
        separators=(',', ':'), allow_nan=False).encode()).hexdigest()


def read(path, expected=None):
    assert path.is_file() and not path.is_symlink(), path
    if expected:
        assert digest(path) == expected, path
    return json.loads(path.read_text())


def main():
    p = argparse.ArgumentParser()
    p.add_argument('result', type=Path)
    p.add_argument('result_sha256')
    p.add_argument('receipt', type=Path)
    p.add_argument('receipt_sha256')
    p.add_argument('output', type=Path)
    p.add_argument('--horizon', type=int, choices=(24, 48), default=24)
    p.add_argument('--release-sha256')
    args = p.parse_args()
    r = read(args.result, args.result_sha256)
    receipt = read(args.receipt, args.receipt_sha256)
    assert receipt['result_sha256'] == args.result_sha256
    assert receipt['worker_termination_confirmed'] is True
    assert not receipt['cleanup_errors'] and not receipt['final_owned_pids']
    assert receipt['arms'] == r['arms']
    if args.release_sha256:
        assert receipt['release_sha256'] == args.release_sha256
    out = {'result_sha256': args.result_sha256, 'receipt_sha256': args.receipt_sha256,
           'status': r['status'], 'parent_status': receipt['status'], 'arms': {}}
    root = args.result.parent
    costs = read(root / 'costs.json')
    assert costs['native_previews_by_arm'] == r['native_previews_by_arm']
    assert costs['native_budget'] == r['native_budget']
    for arm, row in r['arms'].items():
        a = row['selector_accounting']
        counts = row['native_transitions']
        assert a['model_transition_calls'] == counts['search']
        assert sum(x['model_transition_calls'] for x in a['layers']) == counts['search']
        previews = sum(v['native_preview_entries'] for k, v in costs['phases'].items()
                       if k.startswith(arm + '.'))
        assert previews == r['native_previews_by_arm'][arm]
        for phase, count in counts.items():
            assert costs['phases'].get(arm + '.' + phase, {}).get('native_transition_calls', 0) == count
        summary = {'status': row['status'], 'transitions': counts, 'previews': previews,
            'completed_layers': a['completed_layers'], 'expanded_nodes': a['expanded_nodes'],
            'planning_seconds': a['planning_seconds'], 'call_cap_reached': a['call_cap_reached'],
            'time_cap_reached': a['time_cap_reached']}
        if row['status'] == 'complete_replayed':
            assert not a['call_cap_reached'] and not a['time_cap_reached']
            assert len(a['layers']) == a['completed_layers']
            assert all(x['completed'] and x['depth'] == i + 1 for i, x in enumerate(a['layers']))
            assert a['completed_layers'] <= args.horizon
            assert a['completed_layers'] == args.horizon or not a['layers'][-1]['retained_prefix_diagnostics']['retained_nonterminal_prefixes']
            plan = read(root / arm / 'selected-action-plan.json')
            history = read(root / arm / 'complete-history.json')
            for item in (plan, history):
                assert semantic({k: v for k, v in item.items() if k != 'seal'}) == item['seal']
            assert plan['seal'] == row['plan_seal'] == history['action_plan_seal']
            assert plan['actions'] == row['actions']
            assert plan['max_steps'] == args.horizon
            assert history['seal'] == row['complete_history_seal']
            rollout = read(root / arm / 'rollout-metrics.json')
            replay = read(root / arm / 'replay-metrics.json')
            assert history['history'] == rollout['history'] == replay['history']
            assert history['total_reward'] == rollout['total_reward'] == replay['total_reward']
            assert rollout['terminated'] and replay['terminated'] and row['full_history_equal']
            assert counts['rollout'] == counts['replay'] == len(plan['actions'])
            independent = read(root / arm / 'independent-episode.json', row['independent_certificate_sha256'])
            assert independent['accepted'] and row['independent_accepted']
            assert independent['outcomes'] == row['outcomes']
            assert row['outcomes']['total_reward'] == history['total_reward']
            summary.update(actions=plan['actions'], outcomes=row['outcomes'], endpoint=row['endpoint'])
        pages = a.get('action_pages')
        if pages is not None:
            groups = defaultdict(list)
            for page in pages:
                groups[(tuple(page['prefix']), page['source_state_hash'])].append(page)
            assert len(groups) == a['expanded_nodes']
            assert sum(x['evaluated_actions'] for x in pages) == counts['search']
            if row['status'] == 'complete_replayed':
                for key, group in groups.items():
                    total = group[0]['additional_page_count'] + 1
                    assert [x['page_index'] for x in group] == list(range(total))
                    assert all(x['complete'] and x['evaluated_actions'] == x['legal_actions'] for x in group)
                    assert all(x['additional_page_count'] + 1 == total for x in group)
                    assert all(x['base_emitted_count'] <= 120 and x['additional_emitted_count'] <= 32
                               and x['additional_verified_in_page'] <= 8 for x in group)
            summary.update(page_records=len(pages), page_groups=len(groups),
                extra_pairs_verified=sum(x['additional_verified_in_page'] for x in pages))
        out['arms'][arm] = summary
    complete_status = ('complete_SELECT013_paged_route_comparison' if args.horizon == 24
                       else 'complete_SELECT013_paged_route_h48_comparison')
    if r['status'] == complete_status:
        assert receipt['status'] == 'complete' and receipt['exit_code'] == 0
        control_key = ('baseline_saved_accounting_equal' if args.horizon == 24
                       else 'both_h24_prefix_controls_equal')
        assert all(r[k] for k in ('world_equal', 'source_physics_equal',
            control_key, 'all_action_plans_sealed_before_evaluation'))
        if args.horizon == 48:
            for arm, row in r['arms'].items():
                control = read(root / arm / 'h24-prefix-control.json')
                assert row['h24_prefix_control'] == control
                assert control['exact'] and control['compared_complete_layers'] == 23
                horizon = read(root / arm / 'horizon-change.json')
                assert horizon['only_model_horizon_changed']
                assert horizon['old_horizon'] == 24 and horizon['new_horizon'] == 48
        assert not r['held_inputs_opened'] and not r['EVAL_opened']
        assert all(r[k] == 0 for k in ('checkpoint_loads', 'model_forwards', 'optimizer_attempts',
            'gradient_attempts', 'updates', 'private_reference_reads'))
        original, paged = (r['arms'][k]['outcomes'] for k in
            ('original_lazy_beam', 'paged_obstruction_lazy_beam'))
        assert r['comparison']['nominal_return_improvement'] == paged['total_reward'] - original['total_reward']
        assert r['comparison']['target_improvement'] == paged['target_removed_mm3'] - original['target_removed_mm3']
        out['comparison'] = r['comparison']
    else:
        out['failure'] = r.get('failure')
    out['checks_passed'] = True
    args.output.write_text(json.dumps(out, indent=2, sort_keys=True) + '\n')
    print(json.dumps(out, indent=2, sort_keys=True))


if __name__ == '__main__':
    main()
