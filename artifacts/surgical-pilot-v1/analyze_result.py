"""Read-only post-hoc audit and compact tables; never executes an environment or fit.

Run with the repository Python: .venv/bin/python artifacts/surgical-pilot-v1/analyze_result.py
"""
from collections import Counter
import csv
import hashlib
import json
from pathlib import Path

import numpy as np
import torch


HERE = Path(__file__).resolve().parent


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write(name, value):
    (HERE / name).write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')


def checkpoint_parameter_hash(state):
    h = hashlib.sha256()
    for key, tensor in sorted(state.items()):
        h.update(key.encode())
        h.update(tensor.detach().cpu().numpy().tobytes())
    return h.hexdigest()


def episode_row(e, settings):
    actions = e['decisions']
    ids = Counter(a['action_id'] for a in actions)
    reasons = Counter(a.get('reason') or 'none' for a in actions)
    statuses = Counter(a['status'] for a in actions)
    movement = [a for a in actions if a['action_id'] in ('x-', 'x+', 'y-', 'y+', 'z-', 'z+')]
    zero_target_movement = [a for a in movement if a['reward_terms']['tumor_removal'] <= 0]
    row = dict(case=e['case'], method=e['method'], seed=e['seed'],
               local_return=e['local_return'], cusa_return=e['cusa_return'],
               target_mm3=e['local_metrics']['tumor_removed_mm3'],
               healthy_mm3=e['local_metrics']['healthy_removed_mm3'],
               vessel_damage_mm2=e['local_metrics']['vessel_damage_mm2'],
               bleeding_mm3=e['local_metrics']['bleeding_generated_mm3'],
               protected_damage_mm3=e['local_metrics']['protected_damage_mm3'],
               decisions=len(actions), stop=ids['STOP'], inspect=ids['inspect'],
               rejected=statuses['rejected'], partial=statuses['partial'],
               collisions=sum(n for reason, n in reasons.items() if 'COLLISION' in reason),
               invalid_penalties=sum(a['reward_terms']['invalid_action'] != 0 for a in actions),
               zero_target_movement_attempts=len(zero_target_movement),
               zero_target_nominal_requested_mm=len(zero_target_movement) * settings['step_mm'],
               wall_seconds=e['wall_seconds'],
               action_wall_seconds=sum(a['wall_seconds'] for a in actions),
               simulated_seconds=e['local_metrics']['elapsed_seconds'],
               healthy_budget_exceeded=e['local_metrics']['healthy_removed_mm3'] > settings['policy_healthy_budget_mm3'],
               stop_reason=e['stop_reason'], native_success=e['native_success'],
               action_counts=dict(ids), status_counts=dict(statuses), reason_counts=dict(reasons),
               reward_terms={k: sum(a['reward_terms'][k] for a in actions) for k in actions[0]['reward_terms']})
    if 'update' in e:
        row['update'] = e['update']
    return row


def csv_table(name, rows):
    keys = [k for k in rows[0] if not isinstance(rows[0][k], dict)]
    with (HERE / name).open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=keys, extrasaction='ignore', lineterminator='\n')
        writer.writeheader()
        writer.writerows(rows)


def main():
    result = json.loads((HERE / 'result.json').read_text())
    preparation = json.loads((HERE / 'preparation.json').read_text())
    gate = json.loads((HERE / 'launch-gate.json').read_text())
    settings = result['settings']
    assert result['status'] == 'completed' and result['optimizer_updates'] == 12
    expected = dict(preparation)
    claimed = expected.pop('preparation_sha256')
    actual = hashlib.sha256(json.dumps(expected, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()).hexdigest()
    assert actual == claimed == result['preparation_sha256'] == gate['preparation_sha256']
    assert gate['authorized'] is True
    checkpoints = []
    for seed in settings['seeds']:
        updates = [u for u in result['updates'] if u['seed'] == seed]
        assert [u['update'] for u in updates] == list(range(settings['updates']))
        assert all(u['parameters_changed'] and u['before_sha256'] != u['after_sha256'] for u in updates)
        assert all(np.isfinite(u['loss']) and np.isfinite(u['gradient_norm']) for u in updates)
        assert all(a['after_sha256'] == b['before_sha256'] for a, b in zip(updates, updates[1:]))
        path = HERE / f'policy-seed-{seed}.pt'
        checkpoint = torch.load(path, map_location='cpu', weights_only=True)
        assert checkpoint['settings'] == settings and checkpoint['seed'] == seed
        assert checkpoint['preparation_sha256'] == claimed
        assert checkpoint['initial_sha256'] == updates[0]['before_sha256']
        assert checkpoint_parameter_hash(checkpoint['state_dict']) == checkpoint['final_sha256'] == updates[-1]['after_sha256']
        assert all(torch.isfinite(v).all() for v in checkpoint['state_dict'].values())
        checkpoints.append(dict(seed=seed, path=path.name, file_sha256=sha(path),
            initial_parameter_sha256=checkpoint['initial_sha256'], final_parameter_sha256=checkpoint['final_sha256'],
            optimizer_updates=len(updates), all_update_hashes_changed=True))
    episodes = [episode_row(e, settings) for e in result['episodes']]
    assessed = [e for e in episodes if e['method'] != 'training']
    training = [e for e in episodes if e['method'] == 'training']
    assert len(assessed) == 30 and len(training) == 24
    methods = ('untrained', 'random', 'scripted', 'observed_greedy_local', 'trained')
    tables = []
    for case in preparation['cases']:
        prefix = json.loads((HERE / preparation['cases'][case]['prefix_snapshot']).read_text())
        assert sha(HERE / preparation['cases'][case]['prefix_snapshot']) == preparation['cases'][case]['prefix_sha256']
        assert all(e['prefix_metrics'] == prefix['measurements'] for e in result['episodes'] if e['case'] == case)
        for method in methods:
            rows = [e for e in assessed if e['case'] == case and e['method'] == method]
            assert sorted(e['seed'] for e in rows) == settings['seeds']
            mean_keys = ('local_return', 'target_mm3', 'healthy_mm3', 'vessel_damage_mm2', 'bleeding_mm3',
                         'protected_damage_mm3', 'wall_seconds', 'action_wall_seconds', 'simulated_seconds')
            row = dict(case=case, method=method, count=len(rows),
                       **{'mean_' + key: float(np.mean([e[key] for e in rows])) for key in mean_keys})
            for key in ('decisions', 'stop', 'inspect', 'rejected', 'partial', 'collisions', 'invalid_penalties',
                        'zero_target_movement_attempts', 'zero_target_nominal_requested_mm', 'healthy_budget_exceeded'):
                row[key + '_total'] = sum(e[key] for e in rows)
            row['eligible_under_local_healthy_budget'] = row['healthy_budget_exceeded_total'] == 0
            tables.append(row)
    replay_checks = []
    assert len(result['final_replay']) == 6
    assert {(r['case'], r['seed']) for r in result['final_replay']} == {(c, s) for c in preparation['cases'] for s in settings['seeds']}
    for receipt in result['final_replay']:
        snapshot = json.loads((HERE / receipt['path']).read_text())
        prefix = json.loads((HERE / preparation['cases'][receipt['case']]['prefix_snapshot']).read_text())
        episode = next(e for e in result['episodes'] if e['case'] == receipt['case'] and e['seed'] == receipt['seed'] and e['method'] == 'trained')
        assert receipt['reward_ledger_equal'] and receipt['measurements_equal']
        assert snapshot['attempts'][:len(prefix['attempts'])] == prefix['attempts']
        assert snapshot['stateId'] == receipt['state_id'] == episode['state_id']
        assert snapshot['measurements'] == episode['metrics'] and snapshot['totalReward'] == episode['cumulative_return']
        replay_checks.append(dict(**receipt, file_sha256=sha(HERE / receipt['path']), prefix_equal=True))
    curves = []
    for update in result['updates']:
        rows = [e for e in training if e['seed'] == update['seed'] and e['update'] == update['update']]
        assert len(rows) == settings['episodes_per_update']
        assert sum(e['decisions'] for e in rows) == update['transition_count']
        curves.append(dict(**update, batch_return=sum(e['local_return'] for e in rows),
                           batch_target_mm3=sum(e['target_mm3'] for e in rows),
                           batch_healthy_mm3=sum(e['healthy_mm3'] for e in rows)))
    checks = dict(status='verified_from_recorded_outputs_without_new_transitions',
        preparation_integrity_verified=True, checkpoints=checkpoints, optimizer_update_hash_chain_verified=True,
        assessment_episodes=len(assessed), training_episodes=len(training), assessment_seeds_complete=True,
        identical_prefix_metrics_verified=True, final_physical_replay_receipts=replay_checks,
        all_trained_additional_target_zero=all(e['target_mm3'] == 0 for e in assessed if e['method'] == 'trained'),
        all_assessed_vessel_bleeding_protected_damage_zero=all(e['vessel_damage_mm2'] == e['bleeding_mm3'] == e['protected_damage_mm3'] == 0 for e in assessed),
        local_healthy_budget_failures=[dict(case=e['case'], method=e['method'], seed=e['seed'], healthy_mm3=e['healthy_mm3']) for e in episodes if e['healthy_budget_exceeded']],
        gradient_norm_is_preclip=True, losses_are_combined_policy_plus_entropy_objective=True,
        max_preclip_gradient_norm=max(u['gradient_norm'] for u in result['updates']),
        updates_clipped=sum(u['gradient_norm'] > settings['max_grad_norm'] for u in result['updates']),
        proxy_definition='A zero-target movement attempt requests an axis move but has zero tumor-removal reward. Requested mm = count x 2. This includes rejections, partial commands, reversals and healthy-only movement; actual wasted travel is not measured.',
        no_held_out_evidence=True, no_further_fit=True)
    write('verification.json', checks)
    write('summary.json', dict(status=result['status'], preparation_sha256=claimed,
        settings=settings, cost=result['cost'], setup_cost=result['setup_cost'], cumulative_cost=result['cumulative_cost'],
        case_method_table=tables, all_seed_assessments=assessed, training_diagnostics=training,
        paired_trained_minus_baseline=result['summary'], checkpoints=checkpoints,
        limitations=result['limitations'], motion_proxy_definition=checks['proxy_definition']))
    csv_table('all-seed-assessment.csv', assessed)
    csv_table('case-method-comparison.csv', tables)
    csv_table('update-curve.csv', curves)
    write('evidence-index.json', dict(schema='resectionlab.surgical-pilot-evidence.v1',
        preparation_sha256=claimed, source_hashes=preparation['source_hashes'],
        artifacts={p.name: dict(sha256=sha(p), bytes=p.stat().st_size)
                   for p in sorted(HERE.iterdir()) if p.is_file() and p.name != 'evidence-index.json'}))
    print(json.dumps(dict(status='verified', optimizer_updates=len(curves), replay_receipts=len(replay_checks),
        assessed_episodes=len(assessed), local_healthy_budget_failures=len(checks['local_healthy_budget_failures'])), indent=2))


if __name__ == '__main__':
    main()
