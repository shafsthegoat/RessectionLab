#!/usr/bin/env python3
"""Standalone saved-result accounting; no native replay or patient/tensor reads."""
import hashlib
import json
import math
from pathlib import Path


ROOT = Path(__file__).resolve().parent
BC = ROOT.parent / 'pat05-real-geometric-imitation-v1'
RL = ROOT.parent / 'pat05-real-geometric-learning-v1'


def main():
    bindings = {}

    def bind(path):
        data = path.read_bytes()
        bindings[str(path.relative_to(ROOT.parent))] = hashlib.sha256(data).hexdigest()
        return data

    def read(path):
        return json.loads(bind(path))

    run = read(ROOT / 'receipt.json')
    declaration = read(ROOT / 'declaration-input.json')
    output_hashes = read(ROOT / 'output-sha256.json')
    previous = read(BC / 'receipt.json')
    prior_visited = read(BC / 'latest_policy.json')
    prior_teacher = read(RL / 'greedy_search.json')
    weights = read(RL / 'declaration-input.json')['objective']
    for name, expected in declaration['anchor_sha256'].items():
        assert '/' not in name
        assert hashlib.sha256(bind(BC / name)).hexdigest() == expected
    assert run['status'] == 'complete' and run['algorithm'] == 'controlled_visited_state_behavior_cloning'
    assert run['patient'] == 'sub-PAT05' and run['role'] == 'TRAIN'
    assert run['other_patients_opened'] == 0 and not run['generalization_measured']
    assert run['settings'] == declaration['settings']
    assert run['initial_parameter_hash'] == previous['latest_parameter_hash']
    episodes, checks = {}, []
    for name in ('original_teacher', 'visited_collection', 'control_latest', 'augmented_latest'):
        episode = read(ROOT / (name + '.json'))
        episodes[name] = episode
        assert episode['status'] == 'complete' and episode['metrics']['terminated']
        assert episode['metrics']['steps'] == 3 and episode['invalid_actions'] == 0
        assert episode['attempted_actions'] == episode['committed_transitions'] == len(episode['decisions']) == 3
        removed, contacts = set(), set()
        reward, previous_tool = 0., None
        per_action = []
        for row, decision in zip(episode['metrics']['history'], episode['decisions']):
            assert row['action_id'] != 'STOP'
            key = 'actual_action_id' if name == 'visited_collection' else 'action_id'
            assert row['action_id'] == decision[key]
            index = decision['action_ids'].index(decision[key])
            assert decision['action_mask'][index] and decision['status'] == 'returned'
            cells = {tuple(x) for x in row['removed_indices_native']}
            micro = set().union(*({tuple(x) for x in m['removed_indices_native']} for m in row['microsteps']))
            contact = set().union(*({tuple(x) for x in m['contact_indices_native']} for m in row['microsteps']))
            assert micro == cells and not removed & cells
            assert contact == {tuple(x) for x in row['contact_indices_native']}
            removed |= cells
            contacts |= contact
            amount = (weights['target_per_mm3'] * row['target_removed_mm3']
                      - weights['normal_per_mm3'] * row['normal_removed_mm3'] - weights['action_cost']
                      - weights['motion_per_mm'] * row['complete_tool_path_length_mm']
                      - weights['tool_change_cost'] * (previous_tool is not None and previous_tool != row['tool_id']))
            assert math.isclose(amount, row['reward'], rel_tol=1e-12, abs_tol=1e-12)
            reward += amount
            previous_tool = row['tool_id']
            per_action.append({'reward': amount, 'target_mm3': row['target_removed_mm3'], 'normal_mm3': row['normal_removed_mm3']})
            if name != 'original_teacher':
                legal = [i for i, allowed in enumerate(decision['action_mask']) if allowed]
                assert all(math.isfinite(decision['logits'][i]) for i in legal)
                assert index == max(legal, key=lambda i: decision['logits'][i])
        audit = episode['independent_evaluation']
        outcome = audit['outcomes']
        volume = audit['geometry']['source_voxel_volume_mm3']
        assert audit['accepted'] and audit['complete_episode']
        assert audit['geometry']['complete_tool_checked'] and audit['geometry']['frontier_checked']
        assert not audit['geometry']['failures']
        assert math.isclose(reward, outcome['total_reward'], rel_tol=1e-12, abs_tol=1e-12)
        assert reward == episode['metrics']['total_reward']
        assert len(removed) * volume == outcome['simulated_removed_volume_mm3']
        assert len(contacts - removed) * volume == outcome['currently_retained_contacted_tissue_upper_bound_mm3']
        assert outcome['reference_target_fraction_removed'] == outcome['target_removed_mm3'] / outcome['total_reference_target_mm3']
        for key in ('source_hash', 'reference_hash', 'decision_model_hash'):
            assert audit[key] == prior_teacher['independent_evaluation'][key]
        assert outcome['total_reference_target_mm3'] == prior_teacher['independent_evaluation']['outcomes']['total_reference_target_mm3']
        checks.append({'episode': name, 'passed': True, 'reward': reward, 'per_action': per_action,
                       'target_mm3': outcome['target_removed_mm3'], 'normal_mm3': outcome['normal_removed_mm3'],
                       'target_fraction': outcome['reference_target_fraction_removed'],
                       'removed_cells': len(removed), 'retained_contacted_cells': len(contacts - removed)})
    original, visited = episodes['original_teacher'], episodes['visited_collection']
    assert original['metrics']['history'] == prior_teacher['metrics']['history']
    assert visited['metrics']['history'] == prior_visited['metrics']['history']
    assert visited['policy_hash_before'] == visited['policy_hash_after'] == run['initial_parameter_hash']
    original_samples = [{'observation_hash': row['observation_hash'], 'label_action_id': row['action_id']}
                        for row in original['decisions']]
    visited_samples, labels = [], []
    for row, prior in zip(visited['decisions'], prior_visited['decisions']):
        assert row['observation_hash'] == prior['observation_hash']
        assert row['actual_action_id'] == prior['action_id']
        assert row['action_ids'] == prior['action_ids'] and row['action_mask'] == prior['action_mask']
        assert row['state_hash_before_teacher'] == row['state_hash_after_teacher']
        account = row['teacher_accounting']
        assert account['complete'] and account['method'] == 'observed_one_step'
        assert account['objective_source'] == 'permitted_nominal_target_and_frozen_geometric_costs'
        assert account['initial_steps'] == row['step'] and account['max_steps'] == 3
        assert account['model_transition_calls'] == 1 and len(account['decisions']) == 1
        scored = account['decisions'][0]
        legal = [action for action, mask in zip(row['action_ids'], row['action_mask']) if mask]
        assert [x['action_id'] for x in scored['scores']] == legal
        assert scored['step'] == row['step'] and scored['all_current_legal_actions_scored']
        assert scored['legal_nonstop_actions'] == scored['scored_nonstop_actions'] == account['evaluated_nonstop_actions'] == len(legal) - 1
        assert scored['scores'][0]['action_id'] == 'STOP' and scored['scores'][0]['reward'] == 0
        assert all(math.isfinite(x['reward']) for x in scored['scores'])
        best = max(scored['scores'], key=lambda x: x['reward'])
        assert row['label_action_id'] == scored['selected_action_id'] == best['action_id']
        visited_samples.append({'observation_hash': row['observation_hash'], 'label_action_id': row['label_action_id']})
        labels.append({'observation_hash': row['observation_hash'], 'legal_action_count': len(legal),
                       'label_action_id': row['label_action_id'], 'selected_nominal_reward': best['reward']})
    for name, samples in (('control', original_samples * 2), ('augmented', original_samples + visited_samples)):
        branch = run['branches'][name]
        assert branch['status'] == 'complete' and branch['samples'] == samples
        assert branch['example_count'] == 6
        assert branch['unique_observation_count'] == len({x['observation_hash'] for x in samples})
        assert branch['initial_parameter_hash'] == run['initial_parameter_hash']
        assert branch['initial_optimizer_hash'] == run['initial_optimizer_hash']
        assert branch['optimizer_updates'] == len(branch['curve']) == 8
        before = run['initial_parameter_hash']
        for index, update in enumerate(branch['curve'], 1):
            assert update['update'] == index and update['optimizer_steps'] == 1
            assert update['initial_parameter_hash'] == before and update['parameters_changed']
            assert update['updated_parameter_hash'] != before
            assert update['supervised_actions'] == update['loss_forward_calls'] == 6
            assert update['kind'] == 'search_action_behavior_cloning'
            assert update['critic_gradient_norm_before_clip'] == 0
            before = update['updated_parameter_hash']
        assert before == branch['latest_parameter_hash'] == episodes[name + '_latest']['behavior_parameter_hash']
        assert branch['training_loss_forward_calls'] == 48
        assert branch['diagnostic_forward_calls'] == 18 and branch['rollout_forward_calls'] == 3
        for stage in ('initial', 'latest'):
            diagnostics = branch[stage + '_diagnostics']
            assert len(diagnostics) == 6
            for sample, diagnostic in zip(samples, diagnostics):
                assert sample['observation_hash'] == diagnostic['observation_hash']
                assert sample['label_action_id'] == diagnostic['teacher_action_id']
                assert 0 < diagnostic['teacher_probability'] <= 1
            loss = -sum(math.log(x['teacher_probability']) for x in diagnostics) / 6
            reported = branch['curve'][0]['loss'] if stage == 'initial' else branch['latest_training_loss']['loss']
            assert math.isclose(loss, reported, abs_tol=5e-7)
        episode = episodes[name + '_latest']
        assert branch['latest']['independent_evaluation'] == episode['independent_evaluation']
        assert branch['latest']['return'] == episode['metrics']['total_reward']
        assert branch['gain_over_BC8'] == branch['latest']['return'] - previous['latest_policy']['return']
    delta = run['branches']['augmented']['latest']['return'] - run['branches']['control']['latest']['return']
    assert delta == run['augmented_minus_control_return']
    for name in ('receipt.json', 'declaration-input.json', *(x + '.json' for x in episodes), 'control_latest.pt', 'augmented_latest.pt'):
        assert hashlib.sha256(bind(ROOT / name)).hexdigest() == output_hashes[name]
    for name, expected in bindings.items():
        assert hashlib.sha256((ROOT.parent / name).read_bytes()).hexdigest() == expected
    result = {
        'status': 'passed', 'completed_episodes_checked': 4, 'episodes': checks,
        'input_sha256': bindings, 'checker_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        'teacher_labels': labels, 'collected_behavior_history_and_observations_match_BC8': True,
        'collection_reported_policy_and_state_hashes_unchanged': True,
        'same_reported_initial_model_and_Adam_hashes': True,
        'model_update_hash_chains_verified': True, 'nominal_legal_score_max_labels_verified': True,
        'examples_per_update': 6, 'updates_per_branch': 8, 'training_forward_calls_per_branch': 48,
        'control_unique_states': 3, 'augmented_unique_states': 5,
        'augmented_minus_control_return': delta,
        'native_geometry_rerun': False, 'patient_arrays_opened': False, 'checkpoint_tensors_loaded': False,
        'interpretation': 'One matched deterministic same-patient imitation comparison favors visited-state augmentation under the frozen reward. More target and more normal tissue are removed; no reduced clinical harm, RL improvement, mechanics validity or patient generalization is established.',
    }
    (ROOT / 'independent-check.json').write_text(json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + '\n')
    print('4 completed episodes, 3 visited labels and both matched 8-update branches passed')


if __name__ == '__main__':
    main()
