#!/usr/bin/env python3
"""Check the precision-only repeat and compare original saved histories.

Accepted certificates are reused, not geometrically regenerated. Original failed
methods remain unassessed; arithmetic drift is described without promoting them.
"""
import hashlib
import json
import math
from pathlib import Path


ROOT = Path(__file__).resolve().parent
SUBJECTS = ('sub-PAT16', 'sub-PAT20', 'sub-PAT22', 'sub-PAT25', 'sub-PAT28')
METHODS = ('frozen_policy', 'greedy_search', 'random_legal')
POLICY = 'sha256:74e90d9487dff6fe9db83c92e3887f15533e0499abaac386a73bbd7d4d566b4e'


def main():
    bindings = {}

    def read(name):
        payload = (ROOT / name).read_bytes()
        bindings[name] = hashlib.sha256(payload).hexdigest()
        return json.loads(payload)

    declaration = read('declaration-input.json')
    summary = read('summary.json')
    index = read('output-sha256.json')
    registry = declaration['patient_registry']
    weights = registry['common_task']['objective']
    assert declaration['subjects'] == list(SUBJECTS) == registry['subjects']
    assert declaration['checkpoint']['parameter_hash'] == POLICY
    assert declaration['settings']['optimizer_updates'] == 0
    assert declaration['settings']['max_steps'] == 3
    assert summary['patients_prescribed'] == 5
    assert [x['subject'] for x in summary['patients']] == list(SUBJECTS)
    assert summary['optimizer_updates'] == 0 and not summary['adaptation']
    patients, accepted, failed, pairs = [], [], [], {}
    for subject, summarized in zip(SUBJECTS, summary['patients']):
        prefix = subject + '/'
        record = read(prefix + 'receipt.json')
        preparation = read(prefix + 'preparation.json')
        closure = read(prefix + 'closure-check.json')
        supervisor = read(prefix + 'supervisor.json')
        assert record['subject'] == preparation['subject'] == subject
        assert record['role'] == preparation['role'] == 'TRAIN'
        assert record['optimizer_updates'] == record['gradient_steps'] == 0
        assert not record['adaptation']
        assert record['initial_parameter_hash'] == record['final_parameter_hash'] == POLICY
        assert record['method_order'] == list(METHODS)
        assert summarized['methods'] == record['methods']
        assert summarized['supervisor'] == supervisor and summarized['closure_check'] == closure
        assert all(closure[k] for k in ('sources_unchanged', 'preparation_inputs_unchanged', 'checkpoint_bytes_unchanged'))
        assert record['preparation_sha256'] == bindings[prefix + 'preparation.json']
        binding = preparation['binding']
        calculated = hashlib.sha256(json.dumps(binding, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()).hexdigest()
        assert preparation['binding_hash'] == 'sha256:' + calculated
        assert binding['subject'] == subject and binding['patient_group'] == 'BTC:' + subject
        assert binding['role'] == 'TRAIN' and binding['common_task'] == registry['common_task']
        assert binding['cohort_sha256'] == registry['cohort_sha256']
        for key, value in registry['members'][subject].items():
            assert binding['member'][key] == value
        coverage = preparation['coverage']
        count = coverage['full_target_source_cells']
        assert count == coverage['target_inside_support_source_cells'] + coverage['target_outside_support_source_cells']
        assert not coverage['reference_or_nominal_clipped'] and not coverage['support_expanded']
        row = {'subject': subject, 'preparation_status': preparation['status'],
               'full_target_cells': count, 'outside_support_cells': coverage['target_outside_support_source_cells'],
               'methods': {name: record['methods'][name]['status'] for name in METHODS}}
        if preparation['status'] == 'blocked_support_conflict':
            assert record['status'] == 'preparation_blocked' and coverage['target_outside_support_source_cells'] > 0
            assert preparation['executed_transitions'] == 0
            assert all(m['status'] == 'not_executed' and m['outcomes'] is None for m in record['methods'].values())
            patients.append(row)
            continue
        assert preparation['status'] == 'prepared' and coverage['target_outside_support_source_cells'] == 0
        actor = coverage['actor']
        assert actor['crop_shape'] == [64, 64, 64]
        assert actor['nominal_target_positive_voxels_total'] == count
        assert count == actor['nominal_target_positive_voxels_in_crop'] + coverage['target_outside_actor_crop_source_cells']
        assert coverage['native_geometry_uses_full_source_grid']
        grid = binding['member']['expected_native_grid_binding']
        assert grid['method'] == 'orthogonal_roundoff_1e-6mm' and not grid['resampled']
        assert grid['origin_preserved'] and grid['handedness_preserved']
        assert grid['maximum_corner_displacement_mm'] <= grid['maximum_allowed_corner_displacement_mm'] == 1e-6
        assert binding['member']['access_derivation']['rule'] == registry['access_rule']
        assert not binding['member']['access_derivation']['selection_uses_reward_or_native_preview']
        row.update(outside_actor_crop_cells=coverage['target_outside_actor_crop_source_cells'],
                   accepted_initial_actions=preparation['initial_inventory']['accepted_count'])
        for method in METHODS:
            claimed = record['methods'][method]
            episode = read(prefix + method + '.json')
            assert claimed['initial_parameter_hash'] == claimed['final_parameter_hash'] == POLICY
            assert episode['behavior_parameter_hash'] == POLICY
            if claimed['status'] != 'complete':
                assert claimed['outcomes'] is None and episode['status'] == 'failed'
                assert claimed['failure'] == episode['failure']
                metrics = episode.get('metrics')
                drift = {}
                if metrics is not None:
                    for field in ('target_removed_mm3', 'normal_removed_mm3'):
                        per_action = sum(x.get(field, 0.) for x in metrics['history'])
                        drift[field] = {'aggregate': metrics[field], 'per_action_sum': per_action,
                                        'sum_minus_aggregate': per_action - metrics[field]}
                failed.append({'subject': subject, 'method': method, 'failure': claimed['failure'],
                               'outcomes': None, 'saved_arithmetic_drift_only': drift})
                continue
            assert episode['status'] == 'complete' and episode['metrics']['terminated']
            history = episode['metrics']['history']
            decisions = episode['decisions']
            assert 1 <= len(history) <= 3
            assert episode['attempted_actions'] == episode['committed_transitions'] == len(history) == len(decisions)
            assert episode['invalid_actions'] == 0
            stops = [i for i, h in enumerate(history) if h['action_id'] == 'STOP']
            assert stops == [len(history) - 1] if stops else len(history) == 3
            removed, contacts, reward, prior_tool = set(), set(), 0., None
            for h, decision in zip(history, decisions):
                assert h['action_id'] == decision['action_id'] and decision['status'] == 'returned'
                chosen = decision['action_ids'].index(decision['action_id'])
                assert decision['action_mask'][chosen]
                if method == 'frozen_policy':
                    legal = [i for i, mask in enumerate(decision['action_mask']) if mask]
                    assert chosen == max(legal, key=lambda i: decision['logits'][i])
                if h['action_id'] == 'STOP':
                    assert h['reward'] == h['target_removed_mm3'] == h['normal_removed_mm3'] == 0
                    assert not h.get('removed_indices_native') and not h.get('microsteps')
                    continue
                cells = {tuple(x) for x in h['removed_indices_native']}
                micro = set().union(*({tuple(x) for x in m['removed_indices_native']} for m in h['microsteps']))
                contact = set().union(*({tuple(x) for x in m['contact_indices_native']} for m in h['microsteps']))
                assert cells == micro and not removed & cells
                assert contact == {tuple(x) for x in h['contact_indices_native']}
                removed |= cells
                contacts |= contact
                amount = (weights['target_per_mm3'] * h['target_removed_mm3']
                          - weights['normal_per_mm3'] * h['normal_removed_mm3'] - weights['action_cost']
                          - weights['motion_per_mm'] * h['complete_tool_path_length_mm']
                          - weights['tool_change_cost'] * (prior_tool is not None and prior_tool != h['tool_id']))
                assert math.isclose(amount, h['reward'], rel_tol=1e-12, abs_tol=1e-12)
                reward += amount
                prior_tool = h['tool_id']
            audit = episode['independent_evaluation']
            outcome = audit['outcomes']
            assert audit['accepted'] and audit['complete_episode']
            assert audit['geometry']['complete_tool_checked'] and audit['geometry']['frontier_checked']
            assert not audit['geometry']['failures']
            assert audit['source_hash'] == binding['native_source_hash']
            assert audit['decision_model_hash'] == binding['decision_model_hash']
            assert claimed['outcomes'] == outcome and claimed['independent_evaluation_accepted']
            assert math.isclose(reward, outcome['total_reward'], rel_tol=1e-12, abs_tol=1e-12)
            volume = audit['geometry']['source_voxel_volume_mm3']
            assert len(removed) * volume == outcome['simulated_removed_volume_mm3']
            assert len(contacts - removed) * volume == outcome['currently_retained_contacted_tissue_upper_bound_mm3']
            assert math.isclose(count * volume, outcome['total_reference_target_mm3'], rel_tol=1e-12)
            assert outcome['reference_target_fraction_removed'] == outcome['target_removed_mm3'] / outcome['total_reference_target_mm3']
            assert audit['target_access_success'] == (outcome['positive_target_source_cells_removed'] >= 1)
            accepted.append({'subject': subject, 'method': method, 'reward': reward,
                             'target_mm3': outcome['target_removed_mm3'], 'normal_mm3': outcome['normal_removed_mm3'],
                             'target_fraction': outcome['reference_target_fraction_removed'],
                             'removed_cells': len(removed), 'target_access_success': audit['target_access_success']})
        if all(record['methods'][name]['status'] == 'complete' for name in METHODS[:2]):
            pairs[subject] = record['methods']['frozen_policy']['outcomes']['total_reward'] - record['methods']['greedy_search']['outcomes']['total_reward']
        patients.append(row)
    assert summary['complete_pairs'] == len(pairs) and summary['paired_return_difference'] == pairs
    previous_root = ROOT.parent / 'remaining-training-frozen-spatial-v1'
    previous_bindings = {}

    def read_previous(name):
        payload = (previous_root / name).read_bytes()
        previous_bindings[name] = hashlib.sha256(payload).hexdigest()
        return json.loads(payload)

    previous_index = read_previous('output-sha256.json')
    previous_declaration = read_previous('declaration-input.json')
    before_fields = {k: v for k, v in previous_declaration.items() if k != 'source_sha256'}
    revision_fields = {'revision_created_at', 'revision_reason', 'changed_numerical_sources', 'revision_of'}
    assert set(declaration) - set(previous_declaration) == revision_fields
    assert declaration['revision_of']['sha256'] == previous_bindings['declaration-input.json']
    after_fields = {k: v for k, v in declaration.items() if k != 'source_sha256' and k not in revision_fields}
    assert before_fields == after_fields
    changed_sources = [name for name, digest in declaration['source_sha256'].items()
                       if previous_declaration['source_sha256'][name] != digest]
    assert set(changed_sources) == {'src/resectionlab/native_spatial_task.py',
                                   'src/resectionlab/native_spatial_evaluation.py'}
    assert set(declaration['changed_numerical_sources']) == set(changed_sources)
    comparisons, original_failed = [], []
    for subject in SUBJECTS:
        prefix = subject + '/'
        previous_record = read_previous(prefix + 'receipt.json')
        previous_preparation = read_previous(prefix + 'preparation.json')
        current_record = read(prefix + 'receipt.json')
        current_preparation = read(prefix + 'preparation.json')
        assert previous_preparation['binding'] == current_preparation['binding']
        assert previous_preparation['coverage'] == current_preparation['coverage']
        assert previous_preparation['status'] == current_preparation['status']
        for method in METHODS:
            original_method = previous_record['methods'][method]
            current_method = current_record['methods'][method]
            if original_method['status'] == 'not_executed':
                assert current_method['status'] == 'not_executed' and current_method['outcomes'] is None
                continue
            old_episode = read_previous(prefix + method + '.json')
            new_episode = read(prefix + method + '.json')
            old_history = old_episode['metrics']['history']
            new_history = new_episode['metrics']['history']
            # Only target, normal and reward accounting may differ. Full native
            # geometry, all cell lists, microsteps, and provenance remain exact.
            accounting_fields = {'target_removed_mm3', 'normal_removed_mm3', 'reward'}
            geometry = lambda history: [{k: v for k, v in h.items() if k not in accounting_fields} for h in history]
            assert geometry(old_history) == geometry(new_history)
            for before, after in zip(old_episode['decisions'], new_episode['decisions']):
                for key in ('action_id', 'action_ids', 'action_mask', 'observation_hash', 'behavior_parameter_hash'):
                    assert before[key] == after[key]
                if method == 'frozen_policy':
                    assert before['logits'] == after['logits'] and before['probabilities'] == after['probabilities']
            assert len(old_episode['decisions']) == len(new_episode['decisions'])
            comparisons.append({'subject': subject, 'method': method,
                                'original_status': original_method['status'], 'repeat_status': current_method['status'],
                                'exact_native_geometry_and_cell_history_unchanged': True,
                                'observations_actions_and_legal_inventories_unchanged': True})
            if original_method['status'] == 'failed':
                assert original_method['outcomes'] is None
                original_failed.append({'subject': subject, 'method': method,
                                        'original_status': 'failed', 'original_outcomes': None,
                                        'original_failure': original_method['failure']})
    for name, digest in previous_bindings.items():
        if name != 'output-sha256.json':
            assert previous_index[name] == digest
        assert hashlib.sha256((previous_root / name).read_bytes()).hexdigest() == digest
    for name, digest in bindings.items():
        if name != 'output-sha256.json':
            assert index[name] == digest
        assert hashlib.sha256((ROOT / name).read_bytes()).hexdigest() == digest
    result = {'status': 'passed', 'patients_prescribed': 5, 'patients': patients,
              'accepted_episodes_checked': len(accepted), 'accepted_episodes': accepted,
              'failed_episodes_retained_unassessed': failed,
              'complete_primary_pairs': len(pairs), 'paired_return_difference': pairs,
              'original_run_input_sha256': previous_bindings, 'changed_numerical_sources_only': changed_sources,
              'added_revision_metadata_fields': sorted(revision_fields),
              'exact_history_comparisons': comparisons, 'original_failures_preserved': original_failed,
              'input_sha256': bindings, 'checker_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
              'native_geometry_rerun': False, 'patient_arrays_opened': False, 'checkpoint_tensors_loaded': False,
              'interpretation': 'Precision-only repeat: unchanged declarations except two numerical sources; original rejections remain failed/null in original records. Exact native geometry, source cells, observations and actions are compared across attempts. STOP-only remains valid zero-target; existing TRAIN development transfer is not held-out clinical generalization.'}
    (ROOT / 'independent-check.json').write_text(json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + '\n')
    print(f'5 patients retained; {len(accepted)} accepted episodes checked; {len(failed)} failed episodes remain null; {len(pairs)} primary pairs')


if __name__ == '__main__':
    main()
