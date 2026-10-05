#!/usr/bin/env python3
"""Check saved BC8 result arithmetic with stdlib only; no native/patient imports.

Geometry certificates are checked for internal consistency, not recomputed.
Checkpoint bytes are hashed without deserializing tensors. The report distinguishes
imitation improvement on this development patient from RL or transfer evidence.
"""
import hashlib
import json
import math
from pathlib import Path


ROOT = Path(__file__).resolve().parent
ANCHOR = ROOT.parent / 'pat05-real-geometric-learning-v1'


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


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
    previous = read(ANCHOR / 'receipt.json')
    previous_declaration = read(ANCHOR / 'declaration-input.json')
    teacher_original = read(ANCHOR / 'greedy_search.json')
    initial_original = read(ANCHOR / 'initial_policy.json')
    execution_hashes = read(ROOT / 'output-sha256.json')
    for name, expected in declaration['anchor_sha256'].items():
        assert '/' not in name and hashlib.sha256(bind(ANCHOR / name)).hexdigest() == expected
    for name in ('declaration-input.json', 'receipt.json', 'teacher_replay.json', 'latest_policy.json', 'latest.pt'):
        assert hashlib.sha256(bind(ROOT / name)).hexdigest() == execution_hashes[name]
    assert run['status'] == 'complete'
    assert run['algorithm'] == 'behavior_cloning_cross_entropy_only'
    assert run['patient'] == previous['patient'] == 'sub-PAT05'
    assert run['role'] == previous['role'] == 'TRAIN'
    assert run['other_patients_opened'] == 0 and not run['generalization_measured']
    assert declaration['same_patient_development_only']
    assert run['settings'] == declaration['settings']
    assert run['optimizer_updates'] == len(run['curve']) == 8
    assert run['initial_parameter_hash'] == previous['initial_parameter_hash']
    weights = previous_declaration['objective']
    episodes = {}
    checks = []
    for name in ('teacher_replay', 'latest_policy'):
        episode = read(ROOT / (name + '.json'))
        episodes[name] = episode
        assert episode['status'] == 'complete'
        assert episode['metrics']['terminated'] and episode['metrics']['steps'] == 3
        removed, contacts = set(), set()
        reward, previous_tool = 0., None
        per_action = []
        for row in episode['metrics']['history']:
            assert row['action_id'] != 'STOP'
            cells = {tuple(x) for x in row['removed_indices_native']}
            assert not removed & cells
            micro = set().union(*({tuple(x) for x in m['removed_indices_native']} for m in row['microsteps']))
            contact = set().union(*({tuple(x) for x in m['contact_indices_native']} for m in row['microsteps']))
            assert micro == cells
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
            per_action.append({'reward': amount, 'target_mm3': row['target_removed_mm3'],
                               'normal_mm3': row['normal_removed_mm3']})
        audit = episode['independent_evaluation']
        outcome = audit['outcomes']
        volume = audit['geometry']['source_voxel_volume_mm3']
        assert audit['accepted'] and audit['complete_episode']
        assert audit['geometry']['complete_tool_checked'] and audit['geometry']['frontier_checked']
        assert not audit['geometry']['failures']
        assert math.isclose(reward, outcome['total_reward'], rel_tol=1e-12, abs_tol=1e-12)
        assert reward == episode['simulated_return'] == run[name]['return']
        assert run[name]['independent_evaluation'] == audit
        assert len(removed) * volume == outcome['simulated_removed_volume_mm3']
        assert len(contacts - removed) * volume == outcome['currently_retained_contacted_tissue_upper_bound_mm3']
        assert episode['invalid_actions'] == 0
        assert episode['attempted_actions'] == episode['committed_transitions'] == len(episode['decisions']) == 3
        assert [x['action_id'] for x in episode['decisions']] == [x['action_id'] for x in episode['metrics']['history']]
        for decision in episode['decisions']:
            assert decision['action_ids'][decision['selected_index']] == decision['action_id']
            assert decision['action_mask'][decision['selected_index']]
        assert outcome['reference_target_fraction_removed'] == outcome['target_removed_mm3'] / outcome['total_reference_target_mm3']
        for key in ('source_hash', 'reference_hash', 'decision_model_hash'):
            assert audit[key] == teacher_original['independent_evaluation'][key]
        assert outcome['total_reference_target_mm3'] == teacher_original['independent_evaluation']['outcomes']['total_reference_target_mm3']
        checks.append({'episode': name, 'passed': True, 'reward': reward, 'per_action': per_action,
                       'target_mm3': outcome['target_removed_mm3'], 'normal_mm3': outcome['normal_removed_mm3'],
                       'target_fraction': outcome['reference_target_fraction_removed'],
                       'removed_cells': len(removed), 'retained_contacted_cells': len(contacts - removed),
                       'recorded_independent_geometry_accepted': True})

    teacher = episodes['teacher_replay']
    latest = episodes['latest_policy']
    assert teacher['metrics']['history'] == teacher_original['metrics']['history']
    assert teacher['independent_evaluation']['committed_history_hash'] == teacher_original['independent_evaluation']['committed_history_hash']
    assert teacher['behavior_parameter_hash'] == run['initial_parameter_hash']
    before = run['initial_parameter_hash']
    for index, update in enumerate(run['curve'], 1):
        assert update['update'] == index and update['optimizer_steps'] == 1
        assert update['initial_parameter_hash'] == before
        assert update['parameters_changed'] and update['updated_parameter_hash'] != before
        assert update['supervised_actions'] == update['loss_forward_calls'] == 3
        assert update['kind'] == 'search_action_behavior_cloning'
        assert update['critic_gradient_norm_before_clip'] == 0
        assert update['actor_gradient_norm_before_clip'] > 0
        before = update['updated_parameter_hash']
    assert before == run['latest_parameter_hash'] == latest['behavior_parameter_hash']
    assert run['loss_forward_calls'] == 3 * (8 + 2)
    assert run['teacher_diagnostic_forward_calls'] == 6 and run['rollout_policy_forward_calls'] == 3
    for stage in ('initial', 'latest'):
        diagnostic = run[stage + '_teacher_diagnostics']
        assert len(diagnostic) == 3
        for row, decision in zip(diagnostic, teacher['decisions']):
            assert row['teacher_action_id'] == decision['action_id']
            assert row['observation_hash'] == decision['observation_hash']
            assert row['action_count'] == len(decision['action_ids'])
            assert 0 < row['teacher_probability'] <= 1
            assert 1 <= row['teacher_rank'] <= row['action_count']
        ce = -sum(math.log(row['teacher_probability']) for row in diagnostic) / 3
        assert math.isclose(ce, run[stage + '_teacher_loss']['loss'], abs_tol=5e-7)
    for row in latest['decisions']:
        assert row['behavior_parameter_hash'] == before
        assert all(math.isfinite(x) for x in row['logits'])
        legal = [i for i, allowed in enumerate(row['action_mask']) if allowed]
        assert row['selected_index'] == max(legal, key=lambda i: row['logits'][i])
        assert math.isclose(sum(row['probabilities']), 1., abs_tol=1e-6)
    initial_actions = [x['action_id'] for x in initial_original['decisions']]
    latest_actions = [x['action_id'] for x in latest['decisions']]
    assert run['behavior_changed'] == (initial_actions != latest_actions)
    assert latest_actions == run['latest_policy']['actions']
    assert run['return_change_from_initial'] == latest['simulated_return'] - initial_original['simulated_return']
    assert run['latest_minus_greedy_return'] == latest['simulated_return'] - teacher['simulated_return']
    for name, expected in bindings.items():
        assert digest(ROOT.parent / name) == expected, 'Input changed during check: ' + name
    result = {
        'status': 'passed', 'completed_episodes_checked': 2, 'episodes': checks,
        'input_sha256': bindings, 'checker_sha256': digest(Path(__file__).resolve()),
        'same_initial_checkpoint_bytes_and_reported_parameter_hash': True,
        'teacher_history_identical_to_previously_accepted_greedy_history': True,
        'optimizer_updates': 8, 'supervised_actions_total': 24,
        'reported_parameter_hash_chain_verified': True, 'teacher_CE_reconstructed_from_probabilities': True,
        'initial_return': initial_original['simulated_return'], 'latest_return': latest['simulated_return'],
        'greedy_return': teacher['simulated_return'], 'behavior_changed': run['behavior_changed'],
        'imitation_return_change': run['return_change_from_initial'],
        'latest_minus_greedy_return': run['latest_minus_greedy_return'],
        'native_geometry_rerun': False, 'patient_arrays_opened': False, 'checkpoint_tensors_loaded': False,
        'interpretation': 'Stored-result integrity/accounting check. Same-patient imitation improved the fixed latest rollout, but remains inferior to greedy search. No RL improvement, physical validation, clinical accuracy or generalization established.',
    }
    (ROOT / 'independent-check.json').write_text(json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + '\n')
    print('2 complete BC episodes and 8 updates passed standalone stored-result checks')


if __name__ == '__main__':
    main()
