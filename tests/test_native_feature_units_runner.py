"""Registered orchestration checks on explicitly nonpatient tiny geometry."""
from dataclasses import asdict
import copy
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest
import torch

ROOT = Path(__file__).parents[1]
sys.path.insert(0, str(ROOT / 'scripts'))
import run_native_feature_units as runner
from resectionlab.learning import TrainingConfig
from resectionlab.worlds import content_hash, generate_partitions


def declared():
    return runner.load_declaration(ROOT / runner.DECLARATION_PATH)


@pytest.mark.parametrize('change', ['divisor', 'budget', 'order', 'alias'])
def test_resealed_study_changes_fail_before_any_training(tmp_path, change):
    value = declared()
    if change == 'divisor':
        value['profiles']['FEATURE_UNITS']['action_divisors'][1] = 100
    elif change == 'budget':
        value['budgets']['online_scratch_and_adapted_each_seed']['max_wall_seconds'] = 60
    elif change == 'order':
        value['execution_order'][-1], value['execution_order'][-2] = value['execution_order'][-2], value['execution_order'][-1]
    else:
        value['cohort']['aliases'] = []
    value.pop('declaration_content_hash')
    value['declaration_content_hash'] = content_hash(value)
    output = tmp_path / 'refused'
    with pytest.raises(ValueError, match='committed preregistration'):
        runner.run_feature_study(None, None, None, (), output, value, {}, execute=True)
    status = json.loads((output / 'status.json').read_text())
    assert status['status'] == 'failed'
    assert len(status['unfinished_learning_runs']) == 12
    assert len(status['unfinished_offline_runs']) == len(status['unfinished_frozen_runs']) == 2
    assert status['expected_candidate_count'] == 23
    assert not (output / 'candidate-freeze.json').exists()


def test_runner_and_reused_helper_are_inside_source_guard(tmp_path):
    (tmp_path / 'scripts').mkdir()
    for name in ['run_procedural_transfer.py', 'run_native_feature_units.py']:
        (tmp_path / 'scripts' / name).write_text('# stable test source\n')
    snapshot = runner.study_source_snapshot(tmp_path)
    assert all('scripts/' + name in snapshot['numerical_runtime_sha256']
               for name in ['run_procedural_transfer.py', 'run_native_feature_units.py'])
    (tmp_path / 'scripts/run_native_feature_units.py').write_text('# changed\n')
    with pytest.raises(ValueError, match='SOURCE_CHANGED'):
        runner.assert_source_unchanged(snapshot, tmp_path)


def fixture_study(tmp_path, monkeypatch):
    from resectionlab.procedural_learning import (make_procedural_test_target,
        make_native_procedural_fixture, TEST_SCOPE)
    target, factory = make_procedural_test_target()
    base = factory()
    members = tuple(make_native_procedural_fixture(target, input_profile='RAW', study_id=runner.STUDY_ID))
    case = SimpleNamespace(mri=base.native_config.tissue_mask, affine=base.native_config.affine,
        frame='RAS+', semantic_hash=base.case_hash)
    panels = generate_partitions(base.case_hash, base.config.world_generator, 20261004,
        optimization=3, selection=2, final_evaluation=3, stress=2, planning_hash=target.planning_hash)
    declaration = declared()
    reference = runner.physical.load_declaration(ROOT / runner.physical.DECLARATION_PATH)
    settings = TrainingConfig(seed=101, hidden_features=16, max_gradient_steps=2,
        max_environment_steps=64, max_wall_seconds=15, max_episode_steps=4,
        episodes_per_update=2, checkpoint_interval=1)
    declaration['budgets']['offline_training'] = asdict(settings)
    online = asdict(settings)
    online.pop('seed')
    online.update(max_gradient_steps=1, max_environment_steps=16)
    declaration['budgets']['online_scratch_and_adapted_each_seed'] = online
    declaration['budgets']['search'] = dict(beam_width=2, max_expansions=12, max_wall_seconds=15)
    declaration['budgets']['frozen']['max_wall_seconds_for_complete_selection_panel'] = 15
    # These pytest-only substitutions never relax the actual registered API.
    monkeypatch.setattr(runner, 'SCOPE', TEST_SCOPE)
    monkeypatch.setattr(runner, 'assert_declaration', lambda value: None)
    monkeypatch.setattr(runner.physical, 'assert_declared_sources', lambda value: None)
    monkeypatch.setattr(runner.physical, 'assert_declared_target', lambda *args: panels)
    monkeypatch.setattr(runner, 'preserve_inputs', lambda *args: None)
    stable = tmp_path / 'private-source'
    (stable / 'scripts').mkdir(parents=True)
    for name in ['run_procedural_transfer.py', 'run_native_feature_units.py']:
        (stable / 'scripts' / name).write_text('# private stable test source\n')
    capture, check = runner.study_source_snapshot, runner.assert_source_unchanged
    preserve = runner.preserve_source
    monkeypatch.setattr(runner, 'preserve_source', lambda snapshot, destination: preserve(snapshot, destination, root=stable))
    monkeypatch.setattr(runner, 'study_source_snapshot', lambda root=stable: capture(root))
    monkeypatch.setattr(runner, 'assert_source_unchanged', lambda snapshot: check(snapshot, stable))
    return base, case, target, members, declaration, reference


def test_default_preflight_validates_both_profiles_all_seeds_without_optimizer(tmp_path, monkeypatch):
    base, case, target, members, declaration, reference = fixture_study(tmp_path, monkeypatch)
    monkeypatch.setattr(torch.optim, 'Adam', lambda *args, **kwargs: pytest.fail('preflight created an optimizer'))
    output = tmp_path / 'preflight'
    result = runner.run_feature_study(base, case, target, members, output, declaration, reference)
    assert result['status'] == 'preflight_passed_no_training'
    assert not result['training_executed']
    proof = json.loads((output / 'public-world-preflight.json').read_text())
    assert [(x['input_profile'], x['seed']) for x in proof['checks']] == [
        (profile, seed) for profile in runner.PROFILES for seed in runner.SEEDS]
    assert proof['before_offline_pretraining'] and proof['gradient_steps'] == 0
    assert len(proof['paired_initialization']) == 4
    for row in proof['paired_initialization']:
        assert row['policy_hashes']['RAW'] != row['policy_hashes']['FEATURE_UNITS']
        assert set(row['diagnostics']) == set(runner.PROFILES)
        assert row['diagnostics']['RAW']['critic_physical_value'] == row['diagnostics']['FEATURE_UNITS']['critic_physical_value']
    assert not any(output.glob('pretraining-*'))
    assert not (output / 'training.json').exists()
    assert base.metrics()['history'] == []


def test_public_preflight_failure_precedes_offline_optimizer(tmp_path, monkeypatch):
    import resectionlab.procedural_learning as procedural
    base, case, target, members, declaration, reference = fixture_study(tmp_path, monkeypatch)
    original = procedural.validate_procedural_target_worlds
    seen = []
    def fail_scaled(*args, **kwargs):
        seen.append(kwargs['input_profile'])
        if kwargs['input_profile'] == 'FEATURE_UNITS':
            raise ValueError('test public profile gate')
        return original(*args, **kwargs)
    monkeypatch.setattr(procedural, 'validate_procedural_target_worlds', fail_scaled)
    monkeypatch.setattr(procedural, 'train_procedural_native_policy', lambda *args, **kwargs: pytest.fail('offline reached before all preflights'))
    output = tmp_path / 'rejected-preflight'
    with pytest.raises(ValueError, match='test public profile gate'):
        runner.run_feature_study(base, case, target, members, output, declaration, reference, execute=True)
    assert seen == ['RAW'] * 3 + ['FEATURE_UNITS']
    status = json.loads((output / 'status.json').read_text())
    assert len(status['unfinished_learning_runs']) == 12
    assert status['completed_offline_runs'] == []


def test_tiny_two_profile_interleaved_study_retains_all23_candidates(tmp_path, monkeypatch):
    base, case, target, members, declaration, reference = fixture_study(tmp_path, monkeypatch)
    output = tmp_path / 'all-arms'
    result = runner.run_feature_study(base, case, target, members, output, declaration, reference, execute=True)
    assert result['status'] == 'completed'
    assert result['completed_learning_runs'] == runner.planned_learning_ids()
    assert result['completed_offline_runs'] == result['completed_frozen_runs'] == list(runner.PROFILES)
    assert result['completed_operations'] == [runner.operation_id(x) for x in declaration['execution_order']]
    training = json.loads((output / 'training.json').read_text())
    assert [row['operation_id'] for row in training] == runner.planned_learning_ids()
    assert all(row['gradient_steps'] == 1 and row['optimization_environment_steps'] <= 16 for row in training)
    assert all(row['preparation_seconds'] > 0 for row in training)
    assert all(row['initialization_seconds'] is not None for row in training)
    shared = json.loads((output / 'shared-provenance.json').read_text())
    frozen = json.loads((output / 'frozen.json').read_text())
    for profile in runner.PROFILES:
        assert frozen[profile]['gradient_steps'] == frozen[profile]['optimization_environment_steps'] == 0
        assert frozen[profile]['selection_panel_complete']
        for row in training:
            if row['phase'] == 'adapted' and row['input_profile'] == profile:
                assert row['initial_checkpoint_hash'] == shared[profile]['policy_hash']
    for seed in runner.SEEDS:
        raw, scaled = [row for row in training if row['phase'] == 'scratch' and row['seed'] == seed]
        assert raw['initial_trainable_parameter_hash'] == scaled['initial_trainable_parameter_hash']
        assert raw['initial_checkpoint_hash'] != scaled['initial_checkpoint_hash']
    freeze = json.loads((output / 'candidate-freeze.json').read_text())
    assert len(freeze['candidates']) == 23
    assert len(freeze['candidate_profiles']) == 20
    for name, metadata in freeze['candidate_profiles'].items():
        assert metadata['profile_content_hash'] == declaration['profiles'][metadata['profile_id']]['profile_content_hash']
        assert metadata['full_policy_hash']
    assert result['validation']['candidate_count'] == 23
    assert result['validation']['rejected_candidate_ids'] == []
    assert not result['final_worlds_used']
    assert not list(output.rglob('final_evaluation.json'))
    assert not list(output.rglob('stress.json'))
    assert not list(output.rglob('ledger.json'))
    with pytest.raises(FileExistsError):
        runner.run_feature_study(base, case, target, members, output, declaration, reference)


def fake_stop_result(monkeypatch, *, complete):
    """Produce a zero-update result to test orchestration, never train a policy."""
    from resectionlab.learning import MaskedPatientPolicy, TrainingResult, policy_hash, trainable_parameter_hash
    policy = MaskedPatientPolicy(15, 6, 16)
    with torch.no_grad():
        for parameter in policy.parameters():
            parameter.zero_()
    digest = policy_hash(policy)
    def fake(factory, optimization, selection, *, config, output_dir, **kwargs):
        directory = Path(output_dir)
        directory.mkdir(parents=True)
        result = TrainingResult('wall_time_budget', 'PATIENT_SCRATCH_RL', digest, digest,
            digest, factory().decision_model_hash, 0, 0, 2 if complete else 1,
            .01, 0. if complete else None, 0. if complete else None,
            str(directory), initialization_seconds=.001)
        history = [{'world_count': len(selection.seeds), 'mean_return': 0.,
            'checkpoint_hash': digest, 'panel_elapsed_seconds': .004}] if complete else []
        details = {**asdict(result), 'selection_history': history,
            'actor_parameters_changed': False, 'initial_actor_hash': policy_hash(policy.actor),
            'latest_actor_hash': policy_hash(policy.actor), 'input_profile_hash': policy.input_profile.fingerprint,
            'selection_seconds': .004, 'final_checkpoint_export_seconds': .001}
        (directory / 'result.json').write_text(json.dumps(details))
        (directory / 'contract.json').write_text(json.dumps({'algorithm': 'masked_reinforce_state_value_v2'}))
        payload = {'dimensions': policy.dimensions, 'policy': policy.state_dict(),
            'policy_hash': digest, **policy.checkpoint_profile()}
        torch.save(payload, directory / 'initial.pt')
        torch.save(payload, directory / 'checkpoint.pt')
        return result
    monkeypatch.setattr(runner, 'train_patient_policy', fake)
    return trainable_parameter_hash(policy)


def test_incomplete_selection_preserves_arm_record_and_never_extracts(tmp_path, monkeypatch):
    base, case, target, members, declaration, reference = fixture_study(tmp_path, monkeypatch)
    initial_hash = fake_stop_result(monkeypatch, complete=False)
    monkeypatch.setattr(runner, 'load_policy', lambda *args, **kwargs: pytest.fail('incomplete selection was extracted'))
    panels = generate_partitions(base.case_hash, base.config.world_generator, 20261004,
        optimization=3, selection=2, final_evaluation=3, stress=2, planning_hash=target.planning_hash)
    config = TrainingConfig(**declaration['budgets']['online_scratch_and_adapted_each_seed'])
    with pytest.raises(RuntimeError, match='no completed selection panel'):
        runner.learning_arm(base, panels, target, config,
            {'phase': 'scratch', 'profile': 'RAW', 'seed': 11}, tmp_path, {}, {11: initial_hash}, lambda: False)
    record = json.loads((tmp_path / 'raw-scratch-11/arm-record.json').read_text())
    assert record['selected_selection_return'] is None
    assert record['candidate_extraction_seconds'] is None
    assert record['gradient_steps'] == 0


def test_complete_initial_stop_is_eligible_without_learning_claim(tmp_path, monkeypatch):
    base, case, target, members, declaration, reference = fixture_study(tmp_path, monkeypatch)
    initial_hash = fake_stop_result(monkeypatch, complete=True)
    panels = generate_partitions(base.case_hash, base.config.world_generator, 20261004,
        optimization=3, selection=2, final_evaluation=3, stress=2, planning_hash=target.planning_hash)
    config = TrainingConfig(**declaration['budgets']['online_scratch_and_adapted_each_seed'])
    record, candidates = runner.learning_arm(base, panels, target, config,
        {'phase': 'scratch', 'profile': 'RAW', 'seed': 11}, tmp_path, {}, {11: initial_hash}, lambda: False)
    assert record['selected_selection_return'] == 0
    assert record['selected_is_initial']
    assert record['actor_parameters_changed'] is False
    assert record['gradient_steps'] == 0
    assert len(candidates) == 2 and all(x.actions == ('STOP',) for x in candidates)


def test_failed_first_learning_arm_retains_all12_unfinished_and_raw_attempt(tmp_path, monkeypatch):
    base, case, target, members, declaration, reference = fixture_study(tmp_path, monkeypatch)
    fake_stop_result(monkeypatch, complete=False)
    output = tmp_path / 'unfinished-study'
    with pytest.raises(RuntimeError, match='no completed selection panel'):
        runner.run_feature_study(base, case, target, members, output, declaration, reference, execute=True)
    status = json.loads((output / 'status.json').read_text())
    assert status['status'] == 'failed'
    assert status['completed_learning_runs'] == []
    assert status['unfinished_learning_runs'] == runner.planned_learning_ids()
    assert len(status['completed_offline_runs']) == len(status['completed_frozen_runs']) == 2
    attempts = json.loads((output / 'training.json').read_text())
    assert len(attempts) == 1 and attempts[0]['selected_selection_return'] is None
    assert not (output / 'candidate-freeze.json').exists()
