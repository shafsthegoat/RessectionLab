"""Offline saved-metadata/mocked-wrapper controls; never decode or preview a case."""
import copy
from dataclasses import make_dataclass
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from scripts import diagnose_training_observation_coverage as runner


ROOT = Path(__file__).resolve().parents[1]


def original():
    raw = (ROOT / runner.prep.COMMON_PATH).read_bytes()
    assert hashlib.sha256(raw).hexdigest() == runner.ANCHOR_SHA256["sub-PAT05"]
    return json.loads(raw)


def fake_task(row, binding):
    Reward = make_dataclass("AnalyticReward", [(key, float) for key in row["objective"]])
    return SimpleNamespace(case=SimpleNamespace(_grid_record=copy.deepcopy(binding["complete_grid"])),
        reward_spec=Reward(**row["objective"]), max_steps=row["settings"]["max_steps"])


def changed(value):
    if isinstance(value, bool):
        return not value
    if isinstance(value, str):
        return value + "_changed"
    if isinstance(value, (int, float)):
        return value + 1e-12
    if isinstance(value, list):
        result = copy.deepcopy(value)
        result[0] = changed(result[0])
        return result
    raise AssertionError(type(value))


def test_saved20_is_bound_without_mutating_the_historical8_or_reclassifying_v1():
    row = original()
    before = copy.deepcopy(row)
    binding = runner.pat05_complete_grid_binding(row)
    full, subset = binding["complete_grid"], row["member"]["expected_native_grid_binding"]
    assert len(full) == 20 and len(subset) == 8
    assert full != subset  # Retain the exact old failure mechanism.
    assert {k: full[k] for k in subset} == subset
    assert row == before
    runner.verify_pat05_task_binding(fake_task(row, binding), row, binding)
    assert full["source_image_hash"] == row["member"]["research_support_acknowledgment"]["source_image_hash"]
    assert full["support_hash"] == row["member"]["research_support_acknowledgment"]["mask_hash"]
    assert full["original_affine_hash"] != row["member"]["research_support_acknowledgment"]["source_frame_hash"]
    assert runner.VERSION.endswith("v2") and runner.RELEASE_VERSION.endswith("v2")
    assert runner.MANIFEST.endswith("coverage-v2.json")
    assert json.loads((ROOT / "manifests/experiments/training-observation-coverage-v1.json").read_bytes())["version"].endswith("v1")


@pytest.mark.parametrize("key", sorted(runner.PAT05_COMPLETE_GRID_KEYS))
def test_every_runtime_field_is_checked_and_actual_expected_values_retained(key):
    row = original(); binding = runner.pat05_complete_grid_binding(row)
    task = fake_task(row, binding)
    task.case._grid_record[key] = changed(task.case._grid_record[key])
    with pytest.raises(runner.PAT05TaskBindingMismatch) as caught:
        runner.verify_pat05_task_binding(task, row, binding)
    details = caught.value.details
    assert details["changed_grid_fields"] == [key]
    assert details["mismatched_fields"] == ["native_grid." + key]
    assert details["expected"]["native_grid"] == binding["complete_grid"]
    assert details["actual"]["native_grid"] == task.case._grid_record
    assert details["grid_anchor"]["receipt"]["json_pointer"] == runner.PAT05_GRID_POINTER


@pytest.mark.parametrize("mutation", ["missing", "extra", "boolean_type"])
def test_runtime_keyset_and_json_types_are_exact(mutation):
    row = original(); binding = runner.pat05_complete_grid_binding(row); task = fake_task(row, binding)
    if mutation == "missing":
        del task.case._grid_record["source_image_hash"]
    elif mutation == "extra":
        task.case._grid_record["ignored_new_field"] = "not ignored"
    else:
        task.case._grid_record["resampled"] = 0  # Python dict equality alone equates False and 0.
    with pytest.raises(runner.PAT05TaskBindingMismatch):
        runner.verify_pat05_task_binding(task, row, binding)


@pytest.mark.parametrize("field", ["objective", "max_steps"])
def test_objective_and_horizon_remain_part_of_exact_task_check(field):
    row = original(); binding = runner.pat05_complete_grid_binding(row); task = fake_task(row, binding)
    if field == "max_steps":
        task.max_steps += 1
    else:
        key = next(iter(row["objective"]))
        setattr(task.reward_spec, key, row["objective"][key] + .001)
    with pytest.raises(runner.PAT05TaskBindingMismatch) as caught:
        runner.verify_pat05_task_binding(task, row, binding)
    assert caught.value.details["mismatched_fields"] == [field]


@pytest.mark.parametrize("key", sorted(runner.PAT05_ORIGINAL_GRID_KEYS))
def test_each_original_eight_field_is_rejoined_before_decoding(key):
    row = original()
    row["member"]["expected_native_grid_binding"][key] = changed(row["member"]["expected_native_grid_binding"][key])
    with pytest.raises(ValueError, match="eight-field projection"):
        runner.pat05_complete_grid_binding(row)


@pytest.mark.parametrize("field", ["source_image_hash", "mask_hash"])
def test_original_image_and_support_lineage_cannot_be_swapped(field):
    row = original()
    row["member"]["research_support_acknowledgment"][field] = "sha256:" + "0" * 64
    with pytest.raises(ValueError, match="image or support"):
        runner.pat05_complete_grid_binding(row)


def test_receipt_hash_is_checked_before_parsing_and_read_is_bounded(tmp_path, monkeypatch):
    row = original()
    path = tmp_path / runner.PAT05_GRID_RECEIPT
    path.parent.mkdir(parents=True)
    path.write_bytes(b"not a valid original receipt")
    monkeypatch.setattr(runner, "ROOT", tmp_path)
    monkeypatch.setattr(runner.json, "loads", lambda *a, **k: pytest.fail("Changed bytes reached JSON parser"))
    with pytest.raises(ValueError, match="receipt bytes changed"):
        runner.pat05_complete_grid_binding(row)
    path.write_bytes(b"x" * (2 * 1024**2 + 2))
    with pytest.raises(ValueError, match="receipt bytes changed"):
        runner.pat05_complete_grid_binding(row)


def test_missing_receipt_pointer_is_not_silently_replaced(monkeypatch):
    row = original()
    monkeypatch.setattr(runner, "PAT05_GRID_POINTER", "/initial_task_metrics/not_a_grid")
    with pytest.raises(ValueError, match="JSON pointer"):
        runner.pat05_complete_grid_binding(row)


@pytest.mark.parametrize("field", ["path", "sha256", "json_pointer"])
def test_expected_binding_anchor_cannot_be_relabelled(field):
    row = original(); binding = runner.pat05_complete_grid_binding(row); task = fake_task(row, binding)
    binding["receipt"][field] += "_changed"
    with pytest.raises(ValueError, match="pinned historical contract"):
        runner.verify_pat05_task_binding(task, row, binding)


def test_changed_expected_value_cannot_be_used_to_admit_changed_runtime():
    row = original(); binding = runner.pat05_complete_grid_binding(row)
    binding["complete_grid"]["maximum_allowed_corner_displacement_mm"] = 1e-3
    with pytest.raises(ValueError, match="pinned historical contract"):
        runner.verify_pat05_task_binding(fake_task(row, binding), row, binding)


def mock_load_environment(monkeypatch, row, binding):
    source = {k: "fixture" for k in ("accession", "release", "git_commit")}
    case = SimpleNamespace(case_id="BTC-ds001226-sub-PAT05-preop", semantic_hash=row["member"]["case_semantic_hash"],
        planning_hash=row["member"]["research_support_acknowledgment"]["planning_hash"],
        metadata={"source_collection": source}, source_refs=[SimpleNamespace(source_id="structural", provenance="observed")])
    calls = []
    monkeypatch.setattr(runner.prep, "read_development_cohort", lambda p: {"source": source})
    monkeypatch.setattr(runner.prep, "require_development_role", lambda *a, **k: None)
    monkeypatch.setattr(runner, "sha256", lambda p: row["member"]["case_bundle_sha256"])
    monkeypatch.setattr(runner.prep, "_decode_case", lambda p: calls.append("mock_decode") or case)
    task = fake_task(row, binding)
    monkeypatch.setattr(runner.prep, "_construct_task", lambda *a: calls.append("mock_construct") or task)
    record = {"members": {"sub-PAT05": {"member": copy.deepcopy(row["member"]),
        "complete_native_grid_binding": copy.deepcopy(binding)}}, "common_task": {}}
    return record, task, calls


@pytest.mark.parametrize("mutation", ["binding_missing", "binding_changed", "member_changed"])
def test_declaration_binding_rejected_before_mock_decoder_or_constructor(monkeypatch, mutation):
    row = original(); binding = runner.pat05_complete_grid_binding(row)
    record, _, calls = mock_load_environment(monkeypatch, row, binding)
    declared = record["members"]["sub-PAT05"]
    if mutation == "binding_missing":
        del declared["complete_native_grid_binding"]
    elif mutation == "binding_changed":
        declared["complete_native_grid_binding"]["complete_grid"]["support_hash"] = "changed"
    else:
        declared["member"]["expected_native_grid_binding"]["resampled"] = True
    with pytest.raises(ValueError):
        runner.load_initial("sub-PAT05", {"sub-PAT05": row}, record, lambda: None)
    assert calls == []


def test_matching_wrapper_only_reaches_the_two_mocks(monkeypatch):
    row = original(); binding = runner.pat05_complete_grid_binding(row)
    record, task, calls = mock_load_environment(monkeypatch, row, binding)
    assert runner.load_initial("sub-PAT05", {"sub-PAT05": row}, record, lambda: None) is task
    assert calls == ["mock_decode", "mock_construct"]


def test_worker_persists_mismatch_metadata_without_task_or_image_calls(monkeypatch, tmp_path):
    row = original(); binding = runner.pat05_complete_grid_binding(row); task = fake_task(row, binding)
    task.case._grid_record["resampled"] = True
    with pytest.raises(runner.PAT05TaskBindingMismatch) as caught:
        runner.verify_pat05_task_binding(task, row, binding)
    error = caught.value
    originals = {subject: {"coverage": {"retained": True}} for subject in runner.BLOCKED}
    record = {"members": {subject: {"member": {"case_bundle": subject, "case_bundle_sha256": "constant"}}
                          for subject in runner.SUBJECTS}, "source_sha256": {}}
    monkeypatch.setattr(runner, "validate_release", lambda *a: None)
    monkeypatch.setattr(runner, "validate", lambda *a: originals)
    monkeypatch.setattr(runner, "original_records", lambda: originals)
    monkeypatch.setattr(runner, "source_inventory", lambda: {})
    monkeypatch.setattr(runner, "loaded_source_paths", lambda: {})
    monkeypatch.setattr(runner, "sha256", lambda p: "constant")
    def mocked_load(subject, *args):
        if subject == "sub-PAT05":
            raise error
        return object()
    monkeypatch.setattr(runner, "load_initial", mocked_load)
    monkeypatch.setattr(runner, "inspect_task", lambda *a: {"mocked": True})
    result = runner.worker(record, tmp_path, manifest_sha256="fixture", release={})
    saved = json.loads((tmp_path / "receipt.json").read_bytes())
    assert result["status"] == saved["status"] == "incomplete"
    failed = saved["subjects"]["sub-PAT05"]
    assert failed["representation"] is None and failed["initial_previews"] == 0
    assert failed["failure"]["binding_mismatch"] == error.details
    assert saved["total_previews"] == saved["executed_transitions"] == saved["optimizer_updates"] == 0
    assert all(saved["subjects"][s]["status"] == "historical_support_block" for s in runner.BLOCKED)
