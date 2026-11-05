"""Pure authority/task validators; never construct or run a patient pipeline."""
import copy
import importlib.util
import json
from pathlib import Path

import pytest

spec = importlib.util.spec_from_file_location('integrated_regression', Path(__file__).parents[1] / 'scripts/data_integrated_regression.py')
regression = importlib.util.module_from_spec(spec)
spec.loader.exec_module(regression)


@pytest.fixture
def inputs():
    allowed = {'by_material': {'bone': [10, 11, 12], 'soft_tissue': [20, 21]},
               'source_cells': [{'cell_id': i, 'original_region': region, 'original_material': material}
                                for ids, region, material in [([10, 11, 12], 'estimated_skull', 'bone'), ([20, 21], 'brain', 'soft_tissue')]
                                for i in ids]}
    protocol = {'allowed_removals': {'brain_volume_cap_mm3_exact_historical': 2.,
                                     'bone_volume_cap_mm3_exact_historical': 3.,
                                     'counts_are_also_hard_caps': {'bone': 3, 'soft_tissue': 2}}}
    before = {'removedByLabelMm3': {}, 'targetRemovedMm3': 0., 'vesselEncounters': 0,
              'elapsedSeconds': 0., 'travelMm': 0.}
    preview = {'status': 'accepted', 'reason': None, 'deltas': {'removedCells': [10]},
               'metrics': {**before, 'removedByLabelMm3': {'estimated_skull': 1.}, 'elapsedSeconds': 1.}}
    return protocol, allowed, before, preview


def test_accepted_bone_request_is_limited_by_original_source_ids(inputs):
    protocol, allowed, before, preview = inputs
    assert regression.preview_predicate(protocol, allowed, {'operation': 'apply'}, 'cranial_drill', before, preview, set(), 1.) == (True, None)
    preview['deltas']['removedCells'] = [99]
    assert regression.preview_predicate(protocol, allowed, {'operation': 'apply'}, 'cranial_drill', before, preview, set())[1] == 'UNALLOWED_SOURCE_REMOVAL_ID'


def test_cap_is_cumulative_not_only_last_preview_delta(inputs):
    protocol, allowed, before, preview = inputs
    before['removedByLabelMm3'] = {'estimated_skull': 2.}
    preview['deltas']['removedCells'] = [12]
    preview['metrics']['removedByLabelMm3'] = {'estimated_skull': 4.}
    assert regression.preview_predicate(protocol, allowed, {'operation': 'apply'}, 'cranial_drill', before, preview, {10, 11})[1] == 'CUMULATIVE_MATERIAL_VOLUME_CAP'


def test_metric_and_original_source_cell_accounting_must_agree(inputs):
    protocol, allowed, before, preview = inputs
    preview['metrics']['removedByLabelMm3']['estimated_skull'] = 2.
    assert regression.preview_predicate(protocol, allowed, {'operation': 'apply'}, 'cranial_drill', before, preview, set(), 1.)[1] == 'REMOVAL_METRIC_SOURCE_ID_MISMATCH'


def test_allowed_cell_cannot_be_removed_by_wrong_material_tool(inputs):
    protocol, allowed, before, preview = inputs
    preview['deltas']['removedCells'] = [20]
    preview['metrics']['removedByLabelMm3'] = {'brain': 1.}
    assert regression.preview_predicate(protocol, allowed, {'operation': 'apply'}, 'cranial_drill', before, preview, set())[1] == 'WRONG_TOOL_ORIGINAL_MATERIAL_REMOVAL'


@pytest.mark.parametrize('field', ['targetRemovedMm3', 'vesselEncounters', 'activeVesselWounds', 'modeledVesselFluidGeneratedMm3'])
def test_any_target_or_vessel_effect_withholds_preview(inputs, field):
    protocol, allowed, before, preview = inputs
    preview['metrics'][field] = 1.
    assert regression.preview_predicate(protocol, allowed, {'operation': 'apply'}, 'cranial_drill', before, preview, set())[1] == 'TARGET_OR_VESSEL_EFFECT'


def test_pending_new_brain_capture_is_blocked_without_any_removal(inputs):
    _, allowed, _, _ = inputs
    assert regression.capture_predicate('cusa', [20], [], [], allowed) == (True, None)
    assert regression.capture_predicate('cusa', [99], [], [], allowed)[1] == 'UNALLOWED_BRAIN_OR_BONE_WORK_SOURCE_ID'
    assert regression.capture_predicate('cusa', [], [100], [], allowed)[1] == 'TARGET_OR_VESSEL_CONTACT'
    assert regression.capture_predicate('cranial_drill', [], [], [101], allowed)[1] == 'TARGET_OR_VESSEL_CONTACT'


def test_existing_hard_stop_receipt_binds_only_its_executed_microstep_prefix():
    receipt = {'status': 'partial', 'reason': 'UNSUPPORTED_MATERIAL', 'completedFraction': 88 / 169}
    assert regression.historical_success_count(receipt, 169) == 88
    assert not regression.allowed_status({'operation': 'apply'}, 'cusa', receipt)
    receipt['reason'] = 'TISSUE_ASPIRATION_IN_PROGRESS'
    assert regression.historical_success_count(receipt, 169) == 169
    receipt.update(reason='UNSUPPORTED_MATERIAL', completedFraction=.12345)
    with pytest.raises(ValueError): regression.historical_success_count(receipt, 169)


@pytest.mark.parametrize('instrument,reason,allowed', [('cusa', 'TISSUE_ASPIRATION_IN_PROGRESS', True),
    ('cranial_drill', 'BONE_CUTTING_IN_PROGRESS', True), ('cusa', 'BONE_CUTTING_IN_PROGRESS', False),
    ('cranial_drill', 'TISSUE_ASPIRATION_IN_PROGRESS', False), ('cusa', 'UNSUPPORTED_MATERIAL', False)])
def test_partial_status_is_not_approval_without_exact_progress_reason(instrument, reason, allowed):
    assert regression.allowed_status({'operation': 'apply'}, instrument, {'status': 'partial', 'reason': reason}) is allowed
    assert not regression.allowed_status({'operation': 'withdraw'}, instrument, {'status': 'partial', 'reason': reason})


def test_nonmaterial_request_cannot_do_hidden_tissue_work(inputs):
    protocol, allowed, before, preview = inputs
    preview['deltas']['removedCells'] = []
    preview['metrics']['removedByLabelMm3'] = {}
    preview['metrics']['drillInputEnergyJ'] = 1.
    assert regression.preview_predicate(protocol, allowed, {'operation': 'withdraw'}, 'cranial_drill', before, preview, set())[1] == 'NONMATERIAL_REQUEST_PERFORMED_WORK'


@pytest.mark.parametrize('value', [-1., float('nan'), float('inf')])
def test_invalid_or_decreasing_physical_metrics_fail_closed(inputs, value):
    protocol, allowed, before, preview = inputs
    preview['metrics']['elapsedSeconds'] = value
    assert not regression.preview_predicate(protocol, allowed, {'operation': 'apply'}, 'cranial_drill', before, preview, set())[0]


def test_missing_metrics_and_duplicate_delta_ids_are_not_assumed_zero(inputs):
    protocol, allowed, before, preview = inputs
    del preview['metrics']['targetRemovedMm3']
    assert not regression.preview_predicate(protocol, allowed, {'operation': 'apply'}, 'cranial_drill', before, preview, set())[0]
    preview['metrics']['targetRemovedMm3'] = 0.
    preview['deltas']['removedCells'] = [10, 10]
    assert regression.preview_predicate(protocol, allowed, {'operation': 'apply'}, 'cranial_drill', before, preview, set())[1] == 'INVALID_NEW_REMOVAL_IDS'


def test_literal_request_mutation_repeat_or_extra_waypoint_is_rejected():
    original = [{'slot': 'right', 'operation': 'apply', 'duration': 2., 'path': [{'position': [1., 2., 3.], 'quaternion': [0., 0., 0., 1.]}]}]
    regression.validate_requests(copy.deepcopy(original), original)
    for changed in [original + original, [{**original[0], 'duration': 1.}], [{**original[0], 'path': original[0]['path'] * 2}]]:
        with pytest.raises(ValueError): regression.validate_requests(changed, original)


def test_limits_cannot_expand_request_count_or_cleanup_time():
    actions = [{'slot': 'right', 'operation': 'configure', 'duration': .1}, {'slot': 'right', 'operation': 'withdraw', 'duration': 1.}]
    budget = {'maximum_main_requests': 2, 'maximum_cleanup_commits': 1, 'maximum_new_commits': 3,
              'maximum_public_previews': 3, 'maximum_additional_cleanup_seconds': 1., 'maximum_total_declared_action_seconds': 2.1,
              'maximum_main_declared_material_apply_seconds': 0.}
    regression.validate_limits({'budgets': budget}, actions)
    for key in ('maximum_main_requests', 'maximum_cleanup_commits', 'maximum_new_commits', 'maximum_total_declared_action_seconds'):
        expanded = {**budget, key: budget[key] + 1}
        with pytest.raises(ValueError): regression.validate_limits({'budgets': expanded}, actions)


def test_cleanup_selects_earliest_remaining_literal_for_actual_hand_only():
    actions = [{'slot': 'left', 'operation': 'apply'}, {'slot': 'right', 'operation': 'withdraw'},
               {'slot': 'left', 'operation': 'withdraw'}, {'slot': 'left', 'operation': 'withdraw'}]
    slots = {'left': {'inserted': True}, 'right': {'inserted': False}}
    assert regression.select_cleanup(actions, 0, slots, {'left'}, 'left') == (2, None)
    slots['right']['inserted'] = True
    assert regression.select_cleanup(actions, 0, slots, {'left', 'right'}, 'left')[1] == 'MULTIPLE_HANDS_REQUIRE_CLEANUP'
    assert regression.select_cleanup(actions, 3, {'left': {'inserted': True}}, {'left'}, 'left')[1] == 'NO_REMAINING_LITERAL_WITHDRAWAL'


def test_one_decision_cannot_reset_budget_under_another_output_name(tmp_path):
    path = tmp_path / 'decision.json'
    regression.claim_once(path, {'name': 'first'})
    with pytest.raises(FileExistsError): regression.claim_once(path, {'name': 'second'})
    assert json.loads(path.read_text()) == {'name': 'first'}


@pytest.mark.parametrize('name', ['../elsewhere', '/absolute', 'two/parts', '', '.hidden'])
def test_output_name_rejects_path_traversal(name):
    with pytest.raises(ValueError): regression.token(name)


def test_input_symlink_cannot_escape_guarded_root(tmp_path):
    root = tmp_path / 'root'; root.mkdir()
    outside = tmp_path / 'outside.json'; outside.write_text('{}')
    (root / 'escape.json').symlink_to(outside)
    with pytest.raises(ValueError): regression.contained(root, 'escape.json')


def authority_fixture():
    protocol = {'subject': '018', 'source': {'sha256': '1' * 64},
                'old_reference': {'history_sha256': '2' * 64},
                'fresh_initialization': {'initial_state_id': 'sha256:' + '3' * 64},
                'literal_requests': {'sha256': '4' * 64},
                'allowed_removals': {'sha256': '5' * 64, 'brain_volume_cap_mm3_exact_historical': 0., 'bone_volume_cap_mm3_exact_historical': 3.},
                'backend': {'manifest_sha256': '6' * 64},
                'budgets': {'maximum_main_requests': 31, 'maximum_new_commits': 32, 'maximum_total_declared_action_seconds': 47.7}}
    decision = {'schema': 'independent-exact-old-request-regression-decision-v1',
                'decision': 'accepted_for_one_pass_exact_old_request_regression', 'reviewer': '/root/integration_review',
                'subject': '018', 'protocol_sha256': '7' * 64, 'source_sha256': '1' * 64,
                'old_history_sha256': '2' * 64, 'initial_state_id': 'sha256:' + '3' * 64,
                'old_action_list_sha256': '4' * 64, 'allowed_removal_ids_sha256': '5' * 64,
                'backend_manifest_sha256': '6' * 64, 'executor_sha256': '8' * 64,
                'maximum_main_requests': 31, 'maximum_cleanup_commits': 1, 'maximum_new_commits': 32,
                'maximum_total_declared_action_seconds': 47.7, 'maximum_brain_removed_mm3': 0.,
                'maximum_bone_removed_mm3': 3., 'maximum_target_removed_mm3': 0.,
                'maximum_vessel_removed_mm3': 0., 'physical_execution_approved': True}
    return protocol, decision


@pytest.mark.parametrize('field,value', [('executor_sha256', '9' * 64), ('physical_execution_approved', False),
    ('maximum_brain_removed_mm3', 1.), ('source_sha256', '9' * 64), ('old_history_sha256', '9' * 64),
    ('reviewer', '/root'), ('maximum_cleanup_commits', True)])
def test_independent_authority_cannot_be_reused_for_different_driver_source_or_caps(field, value):
    protocol, decision = authority_fixture()
    regression.validate_authority(protocol, decision, protocol_hash='7' * 64, driver_hash='8' * 64)
    decision[field] = value
    with pytest.raises(ValueError): regression.validate_authority(protocol, decision, protocol_hash='7' * 64, driver_hash='8' * 64)


def test_authority_extra_fields_are_not_an_approval_extension():
    protocol, decision = authority_fixture(); decision['allow_new_offsets'] = True
    with pytest.raises(ValueError): regression.validate_authority(protocol, decision, protocol_hash='7' * 64, driver_hash='8' * 64)


def test_backend_override_and_module_drift_fail_before_production_import(tmp_path, monkeypatch):
    import platform
    root = tmp_path / 'data'; package = root / 'snapshot/resectionlab'; package.mkdir(parents=True)
    code = package / '__init__.py'; code.write_text('# frozen control\n')
    manifest_path = root / 'manifest.json'
    manifest_path.write_text(json.dumps({'files': [{'relative_path': 'resectionlab/__init__.py', 'sha256': regression.digest(code), 'bytes': code.stat().st_size}]}))
    monkeypatch.delenv('RESSECTIONLAB_IDLE_THERMAL_BUILD', raising=False)
    monkeypatch.setattr(regression.importlib.metadata, 'version', lambda name: 'test')
    protocol = {'backend': {'manifest_path': 'manifest.json', 'manifest_sha256': regression.digest(manifest_path),
                'snapshot_root': 'snapshot', 'required_optional_native_environment': {'RESSECTIONLAB_IDLE_THERMAL_BUILD': None},
                'required_runtime_versions': {'python': platform.python_version(), 'numpy': 'test', 'scipy': 'test', 'nibabel': 'test'}}}
    assert regression.verify_backend(root, protocol)[0] == package.parent
    metadata = (b'\x00\x05\x16\x07\x00\x02\x00\x00' + b'\0' * 16 + b'\x00\x01'
                + (9).to_bytes(4, 'big') + (38).to_bytes(4, 'big') + (4).to_bytes(4, 'big') + b'META')
    sidecar = package / '.___init__.py'; sidecar.write_bytes(metadata)
    assert len(regression.verify_backend(root, protocol)[1]['validated_filesystem_metadata']) == 1
    sidecar.write_text('# not AppleDouble\n')
    with pytest.raises(ValueError): regression.verify_backend(root, protocol)
    sidecar.write_bytes(metadata)
    extra = package / 'extra.py'; extra.write_text('# genuine unexpected code\n')
    with pytest.raises(ValueError): regression.verify_backend(root, protocol)
    extra.unlink()
    orphan = package / '._orphan.py'; orphan.write_bytes(metadata)
    with pytest.raises(ValueError): regression.verify_backend(root, protocol)
    orphan.unlink()
    other = root / 'other'; other.mkdir()
    with pytest.raises(ValueError): regression.verify_backend(root, protocol, other)
    code.write_text('# changed\n')
    with pytest.raises(ValueError): regression.verify_backend(root, protocol)
