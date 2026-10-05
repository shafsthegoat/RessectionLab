"""Independent saved-JSON and mocked-boundary review; no patient/native execution."""
import copy
from dataclasses import replace
import hashlib
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import diagnose_training_observation_coverage as runner
from resectionlab.core import array_digest, freeze_json
from resectionlab.simulation import RewardSpec

REPOSITORY = Path(__file__).resolve().parents[1]
ORIGINAL_PATH = REPOSITORY / "artifacts/pat05-real-geometric-learning-v1/declaration-input.json"
RECEIPT_PATH = REPOSITORY / "artifacts/pat05-real-geometric-learning-v1/receipt.json"
ORIGINAL = json.loads(ORIGINAL_PATH.read_text())
SAVED_BYTES = RECEIPT_PATH.read_bytes()
SAVED_GRID = json.loads(SAVED_BYTES)["initial_task_metrics"]["native_grid_reconciliation"]
GRID_FIELDS = tuple(sorted(SAVED_GRID))


@pytest.fixture(autouse=True)
def forbid_patient_and_native_calls(monkeypatch):
    def forbidden(*args, **kwargs):
        pytest.fail("Patient decoder/native construction was reached by metadata-only review")
    monkeypatch.setattr(runner.prep, "_decode_case", forbidden)
    monkeypatch.setattr(runner.prep, "_construct_task", forbidden)
    monkeypatch.setattr(runner.prep, "load_prepared_training_case", forbidden)


def fake_task(grid=None, objective=None, max_steps=3):
    return SimpleNamespace(case=SimpleNamespace(_grid_record=freeze_json(copy.deepcopy(SAVED_GRID if grid is None else grid))),
        reward_spec=RewardSpec(**(ORIGINAL["objective"] if objective is None else objective)), max_steps=max_steps)


def authentic_binding():
    return runner.pat05_complete_grid_binding(copy.deepcopy(ORIGINAL))


def record_with_binding(binding):
    return {"members": {"sub-PAT05": {"member": copy.deepcopy(ORIGINAL["member"]),
                                     "complete_native_grid_binding": binding}}}


def changed(value):
    if isinstance(value, bool):
        return not value
    if isinstance(value, float):
        return float(np.nextafter(value, np.inf))
    if isinstance(value, int):
        return value + 1
    if isinstance(value, str):
        return value + "-changed"
    if isinstance(value, list):
        result = copy.deepcopy(value)
        result[0] = changed(result[0])
        return result
    raise AssertionError("Uncovered grid-field type")


def test_full_saved_binding_preserves_old_eight_fields_and_pin_identities():
    binding = authentic_binding()
    assert len(binding["complete_grid"]) == 20
    assert set(binding["complete_grid"]) == runner.PAT05_COMPLETE_GRID_KEYS
    assert len(ORIGINAL["member"]["expected_native_grid_binding"]) == 8
    assert {key: binding["complete_grid"][key] for key in runner.PAT05_ORIGINAL_GRID_KEYS} == ORIGINAL["member"]["expected_native_grid_binding"]
    assert hashlib.sha256(SAVED_BYTES).hexdigest() == runner.PAT05_GRID_RECEIPT_SHA256
    assert hashlib.sha256(runner._canonical(SAVED_GRID)).hexdigest() == runner.PAT05_GRID_SHA256
    # V1's whole-dict equality was necessarily false even for this saved valid record.
    assert SAVED_GRID != ORIGINAL["member"]["expected_native_grid_binding"]
    runner.verify_pat05_task_binding(fake_task(), ORIGINAL, binding)


def test_matrix_hash_is_not_the_composite_structural_frame_hash():
    matrix = np.asarray(SAVED_GRID["original_affine_ras_mm"], dtype=np.float64)
    assert array_digest(matrix) == SAVED_GRID["original_affine_hash"]
    assert SAVED_GRID["original_affine_hash"] != ORIGINAL["member"]["research_support_acknowledgment"]["source_frame_hash"]
    assert SAVED_GRID["source_image_hash"] == ORIGINAL["member"]["research_support_acknowledgment"]["source_image_hash"]
    assert SAVED_GRID["support_hash"] == ORIGINAL["member"]["research_support_acknowledgment"]["mask_hash"]


@pytest.mark.parametrize("field", GRID_FIELDS)
def test_each_of_twenty_runtime_fields_rejects_even_one_ulp_changes(field):
    actual = copy.deepcopy(SAVED_GRID)
    actual[field] = changed(actual[field])
    with pytest.raises(runner.PAT05TaskBindingMismatch) as caught:
        runner.verify_pat05_task_binding(fake_task(actual), ORIGINAL, authentic_binding())
    details = caught.value.details
    assert details["mismatched_fields"] == ["native_grid." + field]
    assert details["changed_grid_fields"] == [field]
    assert details["missing_grid_fields"] == details["extra_grid_fields"] == []
    assert details["expected"]["native_grid"] == SAVED_GRID
    assert details["actual"]["native_grid"] == actual


@pytest.mark.parametrize("field", GRID_FIELDS)
def test_each_of_twenty_missing_runtime_fields_is_rejected(field):
    actual = copy.deepcopy(SAVED_GRID)
    del actual[field]
    with pytest.raises(runner.PAT05TaskBindingMismatch) as caught:
        runner.verify_pat05_task_binding(fake_task(actual), ORIGINAL, authentic_binding())
    assert caught.value.details["missing_grid_fields"] == [field]
    assert "native_grid." + field in caught.value.details["mismatched_fields"]


def test_extra_grid_field_and_bool_number_reinterpretation_rejected():
    for extra in ({**SAVED_GRID, "undeclared": "value"}, {**SAVED_GRID, "resampled": 0}):
        with pytest.raises(runner.PAT05TaskBindingMismatch):
            runner.verify_pat05_task_binding(fake_task(extra), ORIGINAL, authentic_binding())


@pytest.mark.parametrize("field", sorted(ORIGINAL["objective"]))
def test_each_reward_weight_drift_is_rejected_and_retained(field):
    objective = copy.deepcopy(ORIGINAL["objective"])
    objective[field] += .001
    with pytest.raises(runner.PAT05TaskBindingMismatch) as caught:
        runner.verify_pat05_task_binding(fake_task(objective=objective), ORIGINAL, authentic_binding())
    assert caught.value.details["mismatched_fields"] == ["objective"]
    assert caught.value.details["actual"]["objective"][field] == objective[field]


@pytest.mark.parametrize("horizon", [2, 4, 3.0])
def test_horizon_value_or_type_drift_is_rejected(horizon):
    with pytest.raises(runner.PAT05TaskBindingMismatch) as caught:
        runner.verify_pat05_task_binding(fake_task(max_steps=horizon), ORIGINAL, authentic_binding())
    assert caught.value.details["mismatched_fields"] == ["max_steps"]


@pytest.mark.parametrize("field", ("path", "sha256", "json_pointer"))
def test_declared_receipt_path_hash_pointer_tamper_rejected_before_decode(field):
    binding = authentic_binding()
    binding["receipt"][field] += "-changed"
    with pytest.raises(ValueError, match="authenticated historical record"):
        runner.load_initial("sub-PAT05", {"sub-PAT05": ORIGINAL}, record_with_binding(binding), lambda: None)


@pytest.mark.parametrize("tamper", ["grid_sha256", "schema", "complete_grid", "missing_field", "extra_field"])
def test_resealed_or_incomplete_declared_binding_cannot_reach_decoder(tamper):
    binding = authentic_binding()
    if tamper == "complete_grid":
        binding["complete_grid"]["maximum_corner_displacement_mm"] = 0.
        binding["grid_sha256"] = hashlib.sha256(runner._canonical(binding["complete_grid"])).hexdigest()
    elif tamper == "missing_field":
        del binding["complete_grid"]["support_hash"]
    elif tamper == "extra_field":
        binding["complete_grid"]["future_field"] = 0
    else:
        binding[tamper] += "-changed"
    with pytest.raises(ValueError, match="authenticated historical record"):
        runner.load_initial("sub-PAT05", {"sub-PAT05": ORIGINAL}, record_with_binding(binding), lambda: None)


@pytest.mark.parametrize("field", sorted(runner.PAT05_ORIGINAL_GRID_KEYS))
def test_every_old_projection_field_remains_authenticated_before_decode(field):
    original = copy.deepcopy(ORIGINAL)
    original["member"]["expected_native_grid_binding"][field] = changed(original["member"]["expected_native_grid_binding"][field])
    record = record_with_binding(authentic_binding())
    record["members"]["sub-PAT05"]["member"] = copy.deepcopy(original["member"])
    with pytest.raises(ValueError, match="eight-field projection"):
        runner.load_initial("sub-PAT05", {"sub-PAT05": original}, record, lambda: None)


@pytest.mark.parametrize("field", ["source_image_hash", "mask_hash"])
def test_acknowledgment_image_or_support_mismatch_cannot_reach_decoder(field):
    original = copy.deepcopy(ORIGINAL)
    original["member"]["research_support_acknowledgment"][field] = "sha256:" + "0" * 64
    record = record_with_binding(authentic_binding())
    record["members"]["sub-PAT05"]["member"] = copy.deepcopy(original["member"])
    with pytest.raises(ValueError, match="acknowledgment"):
        runner.load_initial("sub-PAT05", {"sub-PAT05": original}, record, lambda: None)


def test_changed_historical_receipt_bytes_rejected_before_decode(monkeypatch, tmp_path):
    binding = authentic_binding()
    path = tmp_path / runner.PAT05_GRID_RECEIPT
    path.parent.mkdir(parents=True)
    path.write_bytes(SAVED_BYTES + b"\n")
    monkeypatch.setattr(runner, "ROOT", tmp_path)
    with pytest.raises(ValueError, match="receipt bytes changed"):
        runner.load_initial("sub-PAT05", {"sub-PAT05": ORIGINAL}, record_with_binding(binding), lambda: None)


def test_failure_details_are_serializable_owned_snapshots():
    actual = copy.deepcopy(SAVED_GRID)
    actual["shape"][0] += 1
    with pytest.raises(runner.PAT05TaskBindingMismatch) as caught:
        runner.verify_pat05_task_binding(fake_task(actual), ORIGINAL, authentic_binding())
    saved = copy.deepcopy(caught.value.details)
    actual["shape"][0] += 1
    assert caught.value.details == saved
    assert json.loads(json.dumps(caught.value.details, allow_nan=False)) == saved


def test_mock_worker_durably_retains_actual_grid_failure(monkeypatch, tmp_path):
    actual = copy.deepcopy(SAVED_GRID)
    actual["derived_affine_ras_mm"][0][0] = changed(actual["derived_affine_ras_mm"][0][0])
    with pytest.raises(runner.PAT05TaskBindingMismatch) as captured:
        runner.verify_pat05_task_binding(fake_task(actual), ORIGINAL, authentic_binding())
    expected_details = captured.value.details
    originals = {subject: {"coverage": {"historical_only": True}} for subject in runner.BLOCKED}
    record = {"members": {subject: {"member": {"case_bundle": "unused", "case_bundle_sha256": "fixed"}} for subject in runner.SUBJECTS},
              "source_sha256": {"metadata-only": "fixed"}}
    monkeypatch.setattr(runner, "validate_release", lambda *a: None)
    monkeypatch.setattr(runner, "validate", lambda *a: originals)
    monkeypatch.setattr(runner, "loaded_source_paths", lambda: {})
    monkeypatch.setattr(runner, "sha256", lambda path: "fixed")
    monkeypatch.setattr(runner, "source_inventory", lambda: record["source_sha256"])
    monkeypatch.setattr(runner, "original_records", lambda: originals)
    # No native objects or profiling imports are needed for this persistence check.
    class FakeGuard:
        total = case_calls = 0
        subject = None
        def __init__(self, check): self.check = check
        def __enter__(self): return self
        def __exit__(self, *args): pass
        def begin_case(self, subject): self.check()
    monkeypatch.setattr(runner, "InitialInventoryGuard", FakeGuard)
    def load(subject, *args):
        if subject == "sub-PAT05":
            raise runner.PAT05TaskBindingMismatch(expected_details)
        return object()
    monkeypatch.setattr(runner, "load_initial", load)
    monkeypatch.setattr(runner, "inspect_task", lambda *a: {"mock_only": True})
    report = runner.worker(record, tmp_path, manifest_sha256="unused", release={})
    durable = json.loads((tmp_path / "receipt.json").read_text())
    assert report["status"] == durable["status"] == "incomplete"
    assert durable["subjects"]["sub-PAT05"]["failure"]["binding_mismatch"] == expected_details
    assert durable["new_representation_completions"] == 3 and len(durable["subjects"]) == 6
    assert durable["subjects"]["sub-PAT05"]["representation"] is None
