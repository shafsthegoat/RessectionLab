#!/usr/bin/env python3
"""Replay saved-result arithmetic only, using stdlib; no patient/native imports.

The same union/reward checks were first executed inline after this run completed.
This saved executable binds their inputs and preserves the actual check result.
It does not independently reproduce tool geometry or load model checkpoints.
"""
import hashlib
import json
import math
from pathlib import Path
import statistics


ROOT = Path(__file__).resolve().parent
EXPECTED = {'initial_policy', 'latest_policy', 'greedy_search',
            *(f'random_legal_{i}' for i in range(3)),
            *(f'optimization_{i}_{j}' for i in range(2) for j in range(2))}


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    bindings = {}

    def read(name):
        path = ROOT / name
        if path.parent != ROOT or path.suffix != '.json':
            raise ValueError('Only declared sibling JSON receipts are read')
        payload = path.read_bytes()
        bindings[name] = hashlib.sha256(payload).hexdigest()
        return json.loads(payload)

    run = read('receipt.json')
    declaration = read('declaration-input.json')
    assert run['status'] == 'complete' and run['optimizer_updates'] == 2
    assert run['patient'] == 'sub-PAT05' and run['role'] == 'TRAIN'
    assert run['other_patients_opened'] == 0 and not run['generalization_measured']
    assert set(run['episodes']) == EXPECTED
    weights = declaration['objective']
    episodes = {}
    checks = []
    for name in sorted(EXPECTED):
        episode = read(run['episodes'][name]['receipt'])
        episodes[name] = episode
        assert episode['status'] == 'complete'
        removed, contacts = set(), set()
        reward, previous_tool = 0., None
        for row in episode['metrics']['history']:
            if row['action_id'] == 'STOP':
                assert row['reward'] == 0
                continue
            cells = {tuple(x) for x in row['removed_indices_native']}
            assert not removed & cells
            micro = set().union(*({tuple(x) for x in m['removed_indices_native']} for m in row['microsteps']))
            assert micro == cells
            contact = set().union(*({tuple(x) for x in m['contact_indices_native']} for m in row['microsteps']))
            assert contact == {tuple(x) for x in row['contact_indices_native']}
            removed |= cells
            contacts |= contact
            amount = (weights['target_per_mm3'] * row['target_removed_mm3']
                      - weights['normal_per_mm3'] * row['normal_removed_mm3']
                      - weights['action_cost']
                      - weights['motion_per_mm'] * row['complete_tool_path_length_mm']
                      - weights['tool_change_cost'] * (previous_tool is not None and previous_tool != row['tool_id']))
            assert math.isclose(amount, row['reward'], rel_tol=1e-12, abs_tol=1e-12)
            reward += amount
            previous_tool = row['tool_id']
        audit = episode['independent_evaluation']
        outcome = audit['outcomes']
        volume = audit['geometry']['source_voxel_volume_mm3']
        assert audit['accepted'] and audit['geometry']['complete_tool_checked']
        assert audit['geometry']['frontier_checked'] and not audit['geometry']['failures']
        assert math.isclose(reward, outcome['total_reward'], rel_tol=1e-12, abs_tol=1e-12)
        assert len(removed) * volume == outcome['simulated_removed_volume_mm3']
        assert len(contacts - removed) * volume == outcome['currently_retained_contacted_tissue_upper_bound_mm3']
        assert episode['invalid_actions'] == 0
        assert episode['attempted_actions'] == episode['committed_transitions'] == len(episode['decisions'])
        assert [x['action_id'] for x in episode['decisions']] == [x['action_id'] for x in episode['metrics']['history']]
        assert all(x['action_id'] in x['action_ids'] for x in episode['decisions'])
        assert outcome['reference_target_fraction_removed'] == outcome['target_removed_mm3'] / outcome['total_reference_target_mm3']
        checks.append({'episode': name, 'passed': True, 'reward': reward,
                       'removed_cells': len(removed), 'retained_contacted_cells': len(contacts - removed),
                       'recorded_independent_geometry_accepted': True})
    before = run['initial_parameter_hash']
    for update in run['gradient_curve']:
        assert update['behavior_parameter_hash'] == update['initial_parameter_hash'] == before
        assert update['parameters_changed'] and update['updated_parameter_hash'] != before
        before = update['updated_parameter_hash']
    assert before == run['latest_parameter_hash']
    initial, latest = episodes['initial_policy'], episodes['latest_policy']
    logit_changes = []
    for a, b in zip(initial['decisions'], latest['decisions']):
        assert a['observation_hash'] == b['observation_hash'] and a['action_ids'] == b['action_ids']
        logit_changes.append({
            'maximum_absolute_logit_change': max(abs(x - y) for x, y in zip(a['logits'], b['logits'])),
            'KL_initial_to_latest': sum(p * math.log(p / q) for p, q in zip(a['probabilities'], b['probabilities']) if p > 0),
            'same_argmax_action': a['action_id'] == b['action_id'],
        })
    for name, digest in bindings.items():
        assert sha(ROOT / name) == digest, 'Input changed during check: ' + name
    result = {
        'status': 'passed', 'completed_episodes_checked': len(checks), 'episodes': checks,
        'input_sha256': bindings, 'checker_sha256': sha(Path(__file__).resolve()),
        'checks': ['micro_to_macro_union', 'no_repeated_removal', 'reward_arithmetic',
                   'recorded_independent_audit_acceptance', 'retained_contact_volume',
                   'selected_action_inventory_membership', 'attempt_commit_counts',
                   'full_target_fraction_denominator', 'reported_parameter_hash_chain', 'input_unchanged'],
        'initial_return': run['episodes']['initial_policy']['return'],
        'latest_return': run['episodes']['latest_policy']['return'],
        'random_three_mean_return': statistics.mean(run['episodes'][f'random_legal_{i}']['return'] for i in range(3)),
        'greedy_return': run['episodes']['greedy_search']['return'],
        'same_state_logit_changes': logit_changes,
        'native_geometry_rerun': False, 'patient_arrays_opened': False, 'checkpoint_tensors_loaded': False,
        'interpretation': 'Stored-result integrity/accounting check; recorded geometry certificates reused. No useful deterministic RL gain, physical validation or generalization established.',
    }
    (ROOT / 'independent-check.json').write_text(json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + '\n')
    print(f"{len(checks)} completed episodes passed standalone stored-result checks")


if __name__ == '__main__':
    main()
