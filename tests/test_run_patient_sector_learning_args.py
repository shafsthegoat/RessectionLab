"""Runner argument, source-choice and bounded native-call receipt contracts."""
from __future__ import annotations

import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace
import time

import pytest


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    'run_patient_sector_learning_args_test',
    ROOT / 'scripts/run_patient_sector_learning.py')
runner = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(runner)


def _argv():
    return ['--train-source', 'train.ressectionlab',
            '--transfer-source', 'transfer.ressectionlab',
            '--output', 'testcases/new-profile', '--source-role', 'DEVELOPMENT']


def test_parser_preserves_superior_defaults_and_accepts_explicit_axis_and_profile():
    defaults = runner.build_parser().parse_args(_argv())
    assert defaults.train_access_method == 'superior-access'
    assert defaults.transfer_access_method == 'superior-access'
    assert defaults.max_suite_physical_calls == 3600
    assert defaults.ppo_source_schedule == 'train-only'
    assert not defaults.fit and not defaults.profile
    selected = runner.build_parser().parse_args(_argv() + [
        '--train-access-method', 'axis0-access',
        '--transfer-access-method', 'axis0-access',
        '--max-suite-physical-calls', '5000',
        '--ppo-source-schedule', 'paired-development', '--profile'])
    assert (selected.train_access_method, selected.transfer_access_method,
            selected.max_suite_physical_calls, selected.ppo_source_schedule,
            selected.profile) == (
                'axis0-access', 'axis0-access', 5000, 'paired-development', True)
    with pytest.raises(SystemExit):
        runner.build_parser().parse_args(_argv() + ['--fit', '--profile'])
    with pytest.raises(SystemExit):
        runner.build_parser().parse_args(_argv() + [
            '--ppo-source-schedule', 'choose-best-case'])


def test_fixed_ppo_source_plan_and_stage_specific_development_roles():
    assert [row['sourceKey'] for row in runner.ppo_episode_plan('train-only')] == [
        'train', 'train', 'train', 'train']
    paired = runner.ppo_episode_plan('paired-development')
    assert paired == [
        {'update': 0, 'episode': 0, 'sourceKey': 'train'},
        {'update': 0, 'episode': 1, 'sourceKey': 'transfer'},
        {'update': 1, 'episode': 0, 'sourceKey': 'train'},
        {'update': 1, 'episode': 1, 'sourceKey': 'transfer'}]
    assert runner.optimization_role('bc', 'transfer', 'paired-development') == \
        'DEV_EVALUATION_ONLY'
    assert runner.optimization_role('ppo', 'train', 'paired-development') == \
        'DEV_TRAINED_FOR_PPO'
    assert runner.optimization_role('ppo', 'transfer', 'paired-development') == \
        'DEV_TRAINED_FOR_PPO'
    assert runner.optimization_role('ppo', 'transfer', 'train-only') == \
        'DEV_FROZEN_TRANSFER_EVALUATION'


def test_suite_check_uses_selected_cap_and_counts_every_native_category():
    report = {'cost': {'previewCalls': 2, 'committedCalls': 3,
                       'independentReplayCalls': 4}}
    result = runner._suite_check(time.perf_counter(), [report],
        source_replays=5, exported_replays=6, max_physical_calls=20)
    assert result['physicalCalls'] == 20 and result['maxPhysicalCalls'] == 20
    with pytest.raises(TimeoutError, match='native-call budget'):
        runner._suite_check(time.perf_counter(), [report],
            source_replays=5, exported_replays=6, max_physical_calls=19)


@pytest.mark.parametrize('schedule, expected_sources', [
    ('train-only', ['train', 'train', 'train', 'train']),
    ('paired-development', ['train', 'transfer', 'train', 'transfer']),
])
def test_selected_methods_and_cap_are_receipted_before_first_rollout(
        monkeypatch, tmp_path, schedule, expected_sources):
    train = tmp_path / 'train.ressectionlab'
    transfer = tmp_path / 'transfer.ressectionlab'
    train.write_bytes(b'declared train source')
    transfer.write_bytes(b'declared transfer source')
    output = tmp_path / 'testcases' / 'profile'
    output.parent.mkdir()
    observed = []

    def prepared(source, access_method):
        observed.append(access_method)
        source = Path(source).resolve()
        return SimpleNamespace(source=source, source_sha256=runner.sha256(source),
            case_id=source.stem, scene=SimpleNamespace(identity='scene'),
            access={'planSeal': 'saved-plan'}, access_method=access_method,
            sector_evidence={'near': {'sourceCellSha256': 'near'},
                             'far': {'sourceCellSha256': 'far'}})

    monkeypatch.setattr(runner, 'development_declaration', lambda source: {'role': 'DEVELOPMENT'})
    monkeypatch.setattr(runner, 'prepare_case', prepared)
    monkeypatch.setattr(runner, '_source_validation_calls', lambda case: {
        'actualReplayCalls': 0, 'savedAccessBranchActionInventory': 0})
    monkeypatch.setattr(runner, 'producer_identity', lambda: {'runnerSourceSha256': 'test'})

    def stop_before_physics(*_args, **_kwargs):
        raise RuntimeError('controlled stop before physics')

    monkeypatch.setattr(runner, 'rollout', stop_before_physics)
    with pytest.raises(RuntimeError, match='controlled stop before physics'):
        runner.run(train, transfer, output, fit=False,
            train_access_method='axis0-access',
            transfer_access_method='axis1-access',
            max_suite_physical_calls=5000,
            ppo_source_schedule=schedule)
    receipt = json.loads((output / 'receipt.json').read_text())
    assert observed == ['axis0-access', 'axis1-access']
    assert receipt['requestedAccessMethods'] == {
        'train': 'axis0-access', 'transfer': 'axis1-access'}
    assert receipt['requestedMaxSuitePhysicalCalls'] == 5000
    assert receipt['ppoSourceSchedule'] == schedule
    assert receipt['ppoScheduleExecuted'] is False
    assert [row['sourceKey'] for row in receipt['ppoEpisodePlan']] == expected_sources
    assert receipt['plannedOptimizationRoles']['transfer']['ppo'] == (
        'DEV_TRAINED_FOR_PPO' if schedule == 'paired-development' else
        'DEV_FROZEN_TRANSFER_EVALUATION')
    assert receipt['sources']['transfer']['plannedOptimizationRoles'] == \
        receipt['plannedOptimizationRoles']['transfer']
    assert receipt['sources']['train']['accessMethod'] == 'axis0-access'
    assert receipt['sources']['transfer']['accessMethod'] == 'axis1-access'
    assert receipt['cost']['maxPhysicalCalls'] == 5000
    assert receipt['cost']['physicalCalls'] == 0


def test_upper_call_cap_rejected_before_output_creation(tmp_path):
    output = tmp_path / 'testcases' / 'not-created'
    for cap in (0, 6001, 3.5, True):
        with pytest.raises(ValueError, match='1..6000'):
            runner.run('unused', 'unused', output,
                       max_suite_physical_calls=cap)
        assert not output.exists()
    with pytest.raises(ValueError, match='Unknown PPO source schedule'):
        runner.run('unused', 'unused', output,
                   ppo_source_schedule='post-hoc-best')
    assert not output.exists()
