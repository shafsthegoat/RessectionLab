"""Constructed-cube paired integration only; no public patient experiment."""
import copy
import json
from pathlib import Path
import shutil
import sys

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts'))
import run_native_axis_feature_units as runner
from test_native_axis_pilot import fixture as original_fixture, guard


def fixture(tmp_path, profile):
    case, base, panels, old = original_fixture(tmp_path)
    paired = runner.load_declaration()
    arm = runner.arm_declaration(paired, profile)
    arm['pilot_id'] = 'private_synthetic_pair:' + profile
    arm['frozen_model'].update(case_hash=case.semantic_hash, planning_hash=case.planning_hash,
        decision_model_hash=base.decision_model_hash, native_config_hash=base.native_config.fingerprint,
        proposal_model_hash=base._proposer.model_hash)
    arm['frozen_model']['action_model']['max_actions_including_stop'] = base.config.max_actions
    arm['world_partitions'] = old['world_partitions']
    arm['optimization_episode_seeds'] = old['optimization_episode_seeds']
    return case, base, panels, arm


@pytest.fixture(scope='module')
def completed(tmp_path_factory):
    parent = tmp_path_factory.mktemp('paired-axis-constructed')
    records = {}
    for profile in ('RAW', 'FEATURE_UNITS'):
        case, base, panels, arm = fixture(parent, profile)
        output = parent / profile / 'pilot'
        status = runner.run_pilot(case, base, panels.optimization, panels.selection, arm,
            output, guard(), source_check=lambda: None)
        records[profile] = {'output': output, 'declaration': arm, 'status': status,
            'result': json.loads((output / 'learner/result.json').read_text()),
            'journal': json.loads((output / 'accounting.json').read_text()),
            'checkpoints': runner.checkpoint_receipt(output / 'learner', arm)}
    return records


def test_actual_two_fresh_profiles_one_update_each_and_twelve_audits(completed):
    for profile, record in completed.items():
        assert record['status']['status'] == 'completed'
        assert record['status']['completed_episode_audit_receipts'] == 6
        assert record['result']['gradient_steps'] == 1
        assert record['result']['actor_parameters_changed']
        assert record['journal']['input_profile'] == profile
        frozen = runner.validate_completion(record['declaration'], record['result'],
            record['journal'], record['checkpoints'])
        assert len(frozen['episodes']) == 6
        assert frozen['shape_probe']['episode'] == 0
        saved = json.loads((record['output'] / 'pre-gradient-initializer.json').read_text())
        assert saved == record['checkpoints']['initialization']
        assert saved['initial_trainable_parameter_hash'] == runner.load_declaration()['paired_initialization']['expected_trainable_parameter_hash']
    raw, scaled = (completed[p]['checkpoints']['initialization'] for p in ('RAW','FEATURE_UNITS'))
    assert raw['initial_trainable_parameter_hash'] == scaled['initial_trainable_parameter_hash']
    assert raw['initial_policy_hash'] != scaled['initial_policy_hash']
    assert raw['axis_observation_contract'] == scaled['axis_observation_contract']


def test_scaled_same_forward_journal_retains_raw_and_actual_scaled_inputs(completed):
    import numpy as np
    row = next(e for e in completed['FEATURE_UNITS']['journal']['events'] if e['kind'] == 'decision')
    inputs = row['payload']['inputs']
    source = np.asarray(inputs['source_action_features']['values'], dtype=np.float32)
    actual = np.asarray(inputs['actor_action_features']['values'], dtype=np.float32)
    divisors = np.asarray(completed['FEATURE_UNITS']['declaration']['frozen_model']['input_profile_manifest']['action_divisors'],dtype=np.float32)
    assert np.array_equal(actual, source / divisors)
    assert not np.array_equal(actual, source)
    assert row['observation_encoding'] == 'RAW'
    assert row['policy_input_profile_hash'] == completed['FEATURE_UNITS']['journal']['input_profile_hash']


@pytest.mark.parametrize('change', ['model', 'world', 'profile'])
def test_mismatch_rejects_before_trainer(tmp_path, change):
    case, base, panels, arm = fixture(tmp_path, 'RAW')
    if change == 'model':
        arm['frozen_model']['decision_model_hash'] = 'wrong'
    elif change == 'world':
        arm['world_partitions']['selection']['seeds'].reverse()
    else:
        arm['frozen_model']['input_profile_hash'] = 'wrong'
    with pytest.raises(ValueError, match='before optimizer'):
        runner.run_pilot(case, base, panels.optimization, panels.selection, arm,
            tmp_path / 'bad', guard(), source_check=lambda: None,
            trainer=lambda *a, **k: pytest.fail('trainer must not run'))
    assert json.loads((tmp_path / 'bad/status.json').read_text())['eligible_candidate_count'] == 0


def test_actual_initial_digest_rejects_before_scored_transition_or_gradient(tmp_path):
    case, base, panels, arm = fixture(tmp_path, 'FEATURE_UNITS')
    arm['frozen_model']['initial_trainable_parameter_hash'] = 'wrong'
    with pytest.raises(ValueError, match='trainable tensors'):
        runner.run_pilot(case, base, panels.optimization, panels.selection, arm,
            tmp_path / 'bad', guard(), source_check=lambda: None)
    journal = json.loads((tmp_path / 'bad/accounting.json').read_text())
    assert not any(e['kind'] in ('decision','transition') for e in journal['events'])
    assert (tmp_path / 'bad/learner/initial.pt').exists()
    assert not (tmp_path / 'bad/learner/checkpoint.pt').exists()


@pytest.mark.parametrize('change', ['drop_latest', 'wrong_profile', 'scaled_as_raw'])
def test_profile_or_incomplete_panel_fails_closed(completed, change):
    item = completed['FEATURE_UNITS']
    result, journal = copy.deepcopy(item['result']), copy.deepcopy(item['journal'])
    if change == 'drop_latest':
        result['selection_history'].pop()
    elif change == 'wrong_profile':
        journal['input_profile'] = 'RAW'
    else:
        decision = next(e for e in journal['events'] if e['kind'] == 'decision')
        decision['payload']['inputs']['actor_action_features'] = copy.deepcopy(decision['payload']['inputs']['source_action_features'])
    from resectionlab.native_axis_accounting import _hash
    journal['receipt_hash'] = _hash({k:v for k,v in journal.items() if k != 'receipt_hash'})
    with pytest.raises(ValueError):
        runner.validate_completion(item['declaration'], result, journal, item['checkpoints'])


def make_authorities(completed, tmp_path, monkeypatch):
    """Completed synthetic arm outputs under explicit test-only publication files."""
    snapshot = {'runtime_content_hash':'test-only-common-runtime'}
    budget = runner.load_declaration()['fixed_protocol']['resource_budget_per_arm']
    runner.write_json(tmp_path / 'launch-source.json', snapshot)
    for profile, record in completed.items():
        folder = tmp_path / profile
        shutil.copytree(record['output'], folder / 'pilot')
        runner.write_json(folder / 'launch-source.json', snapshot)
        runner.write_json(folder / 'preparation.json', {'test_only': True})
        runner.write_json(folder / 'native-configuration.json', {'test_only': True})
        status = json.loads((folder / 'pilot/status.json').read_text())
        runner.write_json(folder / 'worker-status.json', {'status':'completed','eligible_candidate_count':1,
            'full_worker_seconds':1., 'resource':{'budget':copy.deepcopy(budget),
                'elapsed_seconds':1., 'observed_peak_rss_bytes':1024, 'cancellation_reason':None,
                'budget_request_to_receipt_seconds':None},
            'pilot_status_sha256':runner.file_hash(folder / 'pilot/status.json'),
            'candidate_record_sha256':status['candidate_record_sha256'],
            'preparation_sha256':runner.file_hash(folder / 'preparation.json'),
            'native_configuration_sha256':runner.file_hash(folder / 'native-configuration.json'),
            'launch_source_sha256':runner.file_hash(folder / 'launch-source.json')})
        runner.write_json(folder / 'launcher-status.json', {'status':'completed','eligible_candidate_count':1,
            'worker_returncode':0,'parent_timeout_requested':False,'hard_killed':False,
            'resource_budget':copy.deepcopy(budget),'full_launcher_seconds':1.,
            'worker_status_sha256':runner.file_hash(folder / 'worker-status.json')})
    monkeypatch.setattr(runner, 'arm_declaration', lambda paired, profile: copy.deepcopy(completed[profile]['declaration']))


def test_paired_returns_keep_initial_latest_selected_and_current_audits(completed, tmp_path, monkeypatch):
    make_authorities(completed, tmp_path, monkeypatch)
    result = runner.pair_receipt(tmp_path, runner.load_declaration())
    assert result['completed_episode_audit_receipts'] == 12
    for profile, values in result['arms'].items():
        panels = completed[profile]['result']['selection_history']
        assert values['initial_return'] == panels[0]['mean_return']
        assert values['latest_return'] == panels[1]['mean_return']
        assert values['selected_return'] == max(p['mean_return'] for p in panels)
        assert values['latest_minus_initial'] == values['latest_return'] - values['initial_return']
    assert result['FEATURE_UNITS_minus_RAW']['latest_minus_initial'] == (
        result['arms']['FEATURE_UNITS']['latest_minus_initial'] - result['arms']['RAW']['latest_minus_initial'])


@pytest.mark.parametrize('name', ['native-audits.json','accounting.json','history-freeze.json','pre-gradient-initializer.json'])
def test_pair_refuses_evidence_mutated_after_completed_arm(completed, tmp_path, monkeypatch, name):
    make_authorities(completed, tmp_path, monkeypatch)
    path = tmp_path / 'FEATURE_UNITS/pilot' / name
    data = json.loads(path.read_text())
    if isinstance(data,list):
        data.pop()
    else:
        data['tampered'] = True
    runner.write_json(path, data)
    with pytest.raises(ValueError):
        runner.pair_receipt(tmp_path, runner.load_declaration())


def test_pair_refuses_one_failed_arm_despite_retained_payload(completed, tmp_path, monkeypatch):
    make_authorities(completed, tmp_path, monkeypatch)
    path = tmp_path / 'FEATURE_UNITS/launcher-status.json'
    data = json.loads(path.read_text()); data['status'] = 'failed'
    runner.write_json(path,data)
    with pytest.raises(ValueError, match='authorities'):
        runner.pair_receipt(tmp_path, runner.load_declaration())


def test_default_is_declaration_only_and_fresh_output(tmp_path, monkeypatch):
    monkeypatch.setattr(runner, 'launch_arm', lambda *a,**k: pytest.fail('not released'))
    out=tmp_path/'declaration'
    runner.main(['--output',str(out)])
    assert json.loads((out/'launcher-status.json').read_text())['status']=='declared_not_executed'
    with pytest.raises(FileExistsError):
        runner.main(['--output',str(out)])


def test_source_snapshot_includes_new_runner_declaration_and_runtime_only():
    files=runner.source_snapshot()['file_sha256']
    for name in (str(runner.DECLARATION_PATH),'scripts/run_native_axis_feature_units.py',
        'scripts/run_native_axis_pilot.py','scripts/preflight_native_axis.py',
        'src/resectionlab/native_axis_policy_schema.py','src/resectionlab/native_axis_accounting.py',
        'src/resectionlab/policy_inputs.py','src/resectionlab/learning.py'):
        assert name in files
    assert not any(name.startswith('artifacts/') for name in files)


def test_profile_relabel_buffer_and_initial_trainable_mutation_reject(completed,tmp_path):
    import torch
    item=completed['FEATURE_UNITS']
    folder=tmp_path/'learner';shutil.copytree(item['output']/'learner',folder)
    p=torch.load(folder/'initial.pt',weights_only=True)
    p['policy']['action_divisors'][1]=1.
    torch.save(p,folder/'initial.pt')
    with pytest.raises(ValueError,match='divisor buffer'):
        runner.initial_receipt(folder,item['declaration'])
