"""No simulator work: fail closed on uncertain interrupted execution evidence."""
import copy
from contextlib import closing
import importlib.util
import json
from pathlib import Path
import pytest

spec = importlib.util.spec_from_file_location('regression_recovery', Path(__file__).parents[1] / 'scripts/data_integrated_regression_recovery.py')
recovery = importlib.util.module_from_spec(spec)
spec.loader.exec_module(recovery)
spec_prefix = importlib.util.spec_from_file_location('prefix_recovery', Path(__file__).parents[1] / 'scripts/data_integrated_regression_prefix_recovery.py')
prefix_recovery = importlib.util.module_from_spec(spec_prefix)
spec_prefix.loader.exec_module(prefix_recovery)



@pytest.fixture
def evidence(tmp_path):
    directory = tmp_path / 'integrated-v1/regressions/executions/original'
    (directory / 'runs').mkdir(parents=True)
    (directory / 'runs/.bridge-integrity-key').write_bytes(b'key')
    transfer = directory / 'transfers/bridge-unique'
    transfer.mkdir(parents=True)
    blob = b'original source MRI'
    import hashlib
    mri_hash = hashlib.sha256(blob).hexdigest()
    (transfer / (mri_hash + '.bin')).write_bytes(blob)
    reproduction = {'driver_sha256': 'driver', 'decision_sha256': 'decision', 'source_sha256': 'source'}
    (directory / 'reproduction.json').write_text(json.dumps(reproduction))
    claim = tmp_path / 'integrated-v1/regressions/decision-use/decision.json'
    claim.parent.mkdir(parents=True)
    claim.write_text(json.dumps({'output': str(directory.relative_to(tmp_path)), **reproduction}))
    audit = {'schema': 'interrupted-one-pass-execution-read-only-audit-v1', 'executor_sha256': 'driver',
             'source_sha256': 'source', 'physical_commits': 0, 'public_previews': 0,
             'original_claim_remains_consumed': True, 'source_mri_transfer_sha256': mri_hash,
             'consumed_claim': {'path': str(claim.relative_to(tmp_path)), 'sha256': recovery.sha256(claim)},
             'evidence_files': recovery.inventory(tmp_path, directory)}
    kwargs = dict(original_driver_sha256='driver', original_decision_sha256='decision',
                  source_sha256='source', mri_sha256=mri_hash, mri_bytes=len(blob))
    return tmp_path, directory, audit, claim, kwargs


def check(evidence):
    root, directory, audit, claim, kwargs = evidence
    return recovery.validate_interruption(root, directory, audit, claim, **kwargs)


def test_only_complete_preload_evidence_may_continue_original_allocation(evidence):
    assert check(evidence)['zero_commits'] is True


@pytest.mark.parametrize('name', ['actions.json', 'report.json', '.actions.json.interrupted.tmp', 'transfers/bridge-unique/other.bin'])
def test_any_journal_or_additional_transfer_blocks_recovery_even_if_audit_lists_it(evidence, name):
    root, directory, audit, claim, kwargs = evidence
    (directory / name).write_text('potential consumed action')
    audit['evidence_files'] = recovery.inventory(root, directory)
    with pytest.raises(ValueError): check(evidence)


@pytest.mark.parametrize('field,value', [('physical_commits', 1), ('physical_commits', False),
    ('public_previews', 1), ('executor_sha256', 'other'), ('source_sha256', 'other'),
    ('original_claim_remains_consumed', False), ('source_mri_transfer_sha256', 'other')])
def test_any_uncertain_consumption_or_different_binding_blocks(evidence, field, value):
    evidence[2][field] = value
    with pytest.raises(ValueError): check(evidence)


def test_inventory_change_and_missing_claim_block(evidence):
    (evidence[1] / 'reproduction.json').write_text('{}')
    with pytest.raises(ValueError): check(evidence)
    evidence[3].unlink()
    with pytest.raises(ValueError): check(evidence)


def test_claim_cannot_be_replaced_with_fresh_authority(evidence):
    root, directory, audit, claim, kwargs = evidence
    value = json.loads(claim.read_text());value['output'] = 'another-trial'
    claim.write_text(json.dumps(value))
    audit['consumed_claim']['sha256'] = recovery.sha256(claim)
    with pytest.raises(ValueError): check(evidence)


def test_recovery_authority_is_exact_and_cannot_expand_budget():
    expected = {'physical_execution_approved': True, 'remaining_budgets': {'maximum_new_commits': 30},
                'original_allocation_only_no_reset': True}
    recovery.validate_authority(copy.deepcopy(expected), expected)
    changed = copy.deepcopy(expected);changed['remaining_budgets']['maximum_new_commits'] += 1
    with pytest.raises(ValueError): recovery.validate_authority(changed, expected)
    changed = copy.deepcopy(expected);changed['extra'] = 'unreviewed'
    with pytest.raises(ValueError): recovery.validate_authority(changed, expected)


def test_committed_prefix_requires_literal_complete_and_linked_receipts():
    actions = [{'operation': 'configure', 'duration': .1}, {'operation': 'apply', 'duration': 2.}]
    rows = []
    for i, action in enumerate(actions):
        view = {'stateId': str(i+1), 'metrics': {'elapsedSeconds': i}, 'slots': {}}
        delta = {'removedCells': [i]}
        rows.append({'literal_index': i, 'cleanup': False, 'committed': True, 'step_attempted': True,
                     'task_predicate_passed': True, 'task_reason': None, 'action': action,
                     'state_before': str(i), 'state_after': str(i+1), 'preview': {'stateId': str(i+1), 'deltas': delta},
                     'actual': {'view': view, 'outcome': {'status': 'accepted', 'reason': None, 'deltas': delta}},
                     'actual_metrics_after': view['metrics'], 'actual_slots_after': {}})
    result = prefix_recovery.validate_prefix(rows, actions, expected_count=2, initial_state_id='0')
    assert result['declared_seconds'] == 2.1 and result['removed_source_ids'] == [0, 1]
    for field, value in [('state_before', 'different'), ('committed', False), ('step_attempted', False),
                         ('task_predicate_passed', False), ('literal_index', 7), ('cleanup', True)]:
        changed = copy.deepcopy(rows);changed[-1][field] = value
        with pytest.raises(ValueError): prefix_recovery.validate_prefix(changed, actions, expected_count=2, initial_state_id='0')
    with pytest.raises(ValueError): prefix_recovery.validate_prefix(rows[:-1], actions, expected_count=2, initial_state_id='0')


def test_native_action_only_record_reconstructs_full_view_and_exports_own_checkpoint(tmp_path, monkeypatch):
    monkeypatch.syspath_prepend(str(Path(__file__).parents[1] / 'scripts'))
    from data_integrated_regression_resume018 import require_same_view, write_native_recipe
    from resectionlab.desktop_bridge import BridgeSession, _Request
    from resectionlab.imaging import MAX_MANIFEST_BYTES
    from resectionlab.workspace_bundle import load_workspace

    def call(session, operation, **args):
        return json.loads(json.dumps(session.execute(operation, args, _Request(operation), lambda *_: None)))

    source = tmp_path / 'source.ressectionlab'
    recipe = tmp_path / 'action-only.ressectionlab'
    checkpoint = tmp_path / 'verified.ressectionlab'
    with closing(BridgeSession(tmp_path / 'original-transfers', run_dir=tmp_path / 'original-runs')) as session:
        case_hash = call(session, 'createSurgicalFixture')['caseHash']
        call(session, 'saveCase', caseHash=case_hash, path=str(source))
        created = call(session, 'surgicalRehearsal', caseHash=case_hash, command='create')
        zero_path = tmp_path / 'fresh-zero.ressectionlab'
        call(session, 'saveCase', caseHash=case_hash, path=str(zero_path))
        _, zero_artifacts, _ = load_workspace(zero_path)
        record = zero_artifacts['surgical_toolbox']
        action = {'operation': 'configure', 'slot': 'right', 'duration': .1,
                  'mechanicsRevision': 2, 'clockRevision': 1}
        actual = call(session, 'surgicalRehearsal', caseHash=case_hash, command='step',
                      expectedStateId=created['view']['stateId'], action=action)

    snapshot = prefix_recovery.replay_snapshot(record['branches']['main'],
                                               {'committed_actions': [action], 'state_id': actual['view']['stateId']})
    assert 'checkpoint' not in snapshot
    record.update(branches={'main': snapshot}, position=1)
    record.pop('timeline')
    write_native_recipe(source, recipe, record, MAX_MANIFEST_BYTES)

    with closing(BridgeSession(tmp_path / 'recipe-transfers', run_dir=tmp_path / 'recipe-runs')) as session:
        call(session, 'loadCase', path=str(recipe))
        restored = call(session, 'surgicalRehearsal', caseHash=case_hash, command='inspect')
        require_same_view(restored, actual, branch='main')
        changed_history = copy.deepcopy(restored)
        changed_history['view']['history'][0]['status'] = 'partial'
        with pytest.raises(ValueError):
            require_same_view(changed_history, actual, branch='main')
        branch = call(session, 'surgicalRehearsal', caseHash=case_hash, command='branch',
                      expectedStateId=restored['view']['stateId'])
        require_same_view(branch, actual, branch='branch-1')
        call(session, 'saveCase', caseHash=case_hash, path=str(checkpoint))
    _, artifacts, workspace = load_workspace(checkpoint)
    assert workspace['sessionHash'] == load_workspace(source)[2]['sessionHash']
    own = artifacts['surgical_toolbox']['branches']['branch-1']
    assert own['actions'] == [action] and own['checkpoint']['position'] == 1

    with closing(BridgeSession(tmp_path / 'cold-transfers', run_dir=tmp_path / 'cold-runs')) as session:
        call(session, 'loadCase', path=str(checkpoint))
        cold = call(session, 'surgicalRehearsal', caseHash=case_hash, command='inspect')
        require_same_view(cold, actual, branch='branch-1')
