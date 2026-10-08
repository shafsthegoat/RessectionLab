"""Source-metadata and process controls only; no generated or decoded patients."""
import base64
import copy
import json
from pathlib import Path
import sys
import time

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import lausanne_annotation_intake as intake


@pytest.fixture(autouse=True)
def prohibit_scientific_access(monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError("Control tests may not acquire or decode scientific payloads")
    monkeypatch.setattr(intake, "open_without_redirect", forbidden)
    monkeypatch.setattr(intake.gzip, "open", forbidden)
    original = Path.open

    def guarded(path, *args, **kwargs):
        if path.name.endswith((".nii", ".nii.gz", ".nii.gz.partial")):
            forbidden()
        return original(path, *args, **kwargs)
    monkeypatch.setattr(Path, "open", guarded)


@pytest.fixture(scope="module")
def metadata():
    manifest = json.loads(intake.MANIFEST.read_bytes())
    for entry in manifest["metadata_sources"].values():
        cached = intake.CACHE / "metadata" / (entry["sha256"] + ".bin")
        if not cached.exists() and not (ROOT / entry["path"]).exists():
            pytest.skip("Requires retained real Lausanne qualification metadata; no payload substitute")
    return intake.preflight()


def sub253_receipt():
    path = intake.originals.CACHE / "acquired/sub-253_ses-20110628.json"
    if not path.is_file():
        pytest.skip("Requires actual source-bound sub253 original intake receipt")
    return json.loads(path.read_bytes())


def test_frozen_full_inventory_and_semantics(metadata):
    manifest, sessions = metadata
    assert len(sessions) == 210
    assert len(manifest["records"]) == 148
    passing = [r for r in manifest["records"] if r["status"] == "metadata_qualified"]
    assert len(passing) == 144
    assert sum(r["file"]["bytes"] for r in passing) == 13977015
    assert {r["subject"] for r in manifest["records"] if r["status"] == "metadata_failed"} == {
        "sub-148", "sub-163", "sub-220"}
    assert manifest["counts"]["annotation_subtypes"] == {
        "manual_region_subtype_unresolved": 111,
        "source_voxelwise_person_date_mismatch": 31,
        "source_voxelwise_exact_crosswalk": 2}
    assert all(not manifest["semantics"][k] for k in
               ("training_admitted", "scanner_frame_admitted", "spatial_planning_admitted"))
    assert manifest["semantics"]["background"].startswith("unknown")


def record_validation_inputs(metadata):
    manifest, sessions = metadata
    source = json.loads(intake.source_metadata(manifest["metadata_sources"]["qualification"]))
    cohort = json.loads(intake.source_metadata(manifest["cohort"]))
    people = {m["subject"]: m for m in cohort["members"] if m["role"] == "TRAIN"}
    return manifest, sessions, source, people


@pytest.mark.parametrize("change", ["role", "subject", "RawSources", "version", "pointer", "subtype"])
def test_changed_source_contract_refused_before_transfer(metadata, change):
    manifest, sessions, source, people = record_validation_inputs(metadata)
    row = copy.deepcopy(manifest["records"][0])
    if change == "role":
        row["role"] = "SELECT"
    elif change == "subject":
        row["subject"] = "sub-476"
    elif change == "RawSources":
        row["sidecar"]["RawSources"] = "sub-476_ses-20140519_angio.nii.gz"
    elif change == "version":
        row["file"]["source_url"] = row["file"]["source_url"].split("?")[0]
    elif change == "pointer":
        row["pointer_base64"] = base64.b64encode(b"unqualified metadata").decode()
    else:
        row["annotation_subtype_status"] = "source_voxelwise_exact_crosswalk"
    with pytest.raises(intake.AcquisitionError):
        intake.validate_record(row, source["records"][0], sessions, people, {})


def test_failed_record_cannot_be_promoted(metadata):
    manifest, sessions, source, people = record_validation_inputs(metadata)
    i = next(i for i, r in enumerate(manifest["records"]) if r["status"] == "metadata_failed")
    row = copy.deepcopy(manifest["records"][i])
    row["status"] = "metadata_qualified"
    with pytest.raises(intake.AcquisitionError, match="qualified source record"):
        intake.validate_record(row, source["records"][i], sessions, people, {})


def test_wrong_patient_failure_source_bytes_are_retained(metadata):
    manifest, _ = metadata
    row = next(r for r in manifest["records"] if r["subject"] == "sub-220")
    payload = base64.b64decode(row["failed_metadata_responses"][1]["body_base64"])
    assert json.loads(payload)["RawSources"] == "sub-217_ses-20100812_angio.nii.gz"
    assert row["status"] == "metadata_failed" and "file" not in row


def test_one_mask_scope_retains_all_failures_and_remaining_denominator(metadata):
    manifest, _ = metadata
    path = manifest["records"][0]["path"]
    outcomes = intake.initial_outcomes(manifest, path)
    assert len(outcomes) == 148
    assert sum(r["status"] == "deferred_not_attempted" for r in outcomes) == 1
    assert sum(r["status"] == "outside_bounded_mask_scope" for r in outcomes) == 143
    assert sum(r["status"] == "metadata_failed_excluded" for r in outcomes) == 4
    with pytest.raises(intake.AcquisitionError, match="whitelist"):
        intake.initial_outcomes(manifest, "sub-459")


def test_sub253_actual_metadata_keeps_tof_pass_and_t1_failure(metadata):
    manifest, sessions = metadata
    original = sessions[("sub-253", "ses-20110628")]
    saved = sub253_receipt()
    reference = next(f for f in original["files"] if f["path"].endswith("_angio.nii.gz"))
    result = intake.original_receipt_contract(original, saved, manifest["original_index"]["sha256"], reference)
    assert result["referenced_tof_passed"] is True
    assert result["source_receipt_pair_qc"] == "failed"
    assert result["other_modality_qc"][0]["status"] == "qc_failed"
    assert "QFORM_SFORM_DISAGREEMENT" in result["other_modality_qc"][0]["reason"]
    assert result["full_pair_current_fixity_checked"] is False


def test_missing_original_receipt_explicitly_defers_reference(metadata, tmp_path, monkeypatch):
    manifest, sessions = metadata
    row = manifest["records"][0]
    monkeypatch.setattr(intake, "ROOT", tmp_path)
    monkeypatch.setattr(intake.originals, "CACHE", tmp_path / "original-cache")
    result = intake.verify_reference(row, sessions[(row["subject"], row["session"])], manifest,
                                    tmp_path / "trial", time.monotonic() + 1)
    assert result["status"] == "deferred_missing_original_receipt"
    assert "raw_grid" not in result


@pytest.mark.parametrize("change", ["role", "identity", "missing_tof", "file_hash", "claim"])
def test_original_receipt_metadata_mismatch_refused(metadata, change):
    manifest, sessions = metadata
    original = sessions[("sub-253", "ses-20110628")]
    saved = sub253_receipt()
    reference = next(f for f in original["files"] if f["path"].endswith("_angio.nii.gz"))
    if change == "role":
        saved["role"] = "SELECT"
    elif change == "identity":
        saved["session"] = "ses-20140519"
    elif change == "missing_tof":
        saved["integrity_qc"].pop()
    elif change == "file_hash":
        saved["files"][0]["sha256"] = "unverified"
    else:
        saved["spatial_planning_admitted"] = True
    with pytest.raises(intake.AcquisitionError):
        intake.original_receipt_contract(original, saved, manifest["original_index"]["sha256"], reference)


def test_budget_uses_actual_source_metadata_without_allocation(metadata):
    manifest, _ = metadata
    saved = sub253_receipt()
    shape = saved["integrity_qc"][1]["header"]["shape"]
    # Scalar item widths exercise memory arithmetic, not synthetic image data.
    for itemsize in (1, 2, 4, 8):
        budget = intake.decoding_budget(shape, itemsize, manifest["bounds"])
        assert budget["voxels"] == 36700160
        assert budget["chunk_working_bytes_bound"] <= manifest["bounds"]["max_decoded_chunk_working_bytes"]
        assert budget["chunk_voxels"] < budget["voxels"]
    with pytest.raises(intake.AcquisitionError, match="size bounds"):
        intake.decoding_budget([2**63, 1, 1], 8, manifest["bounds"])
    with pytest.raises(intake.AcquisitionError, match="dimensions/dtype"):
        intake.decoding_budget([0, 1, 1], 1, manifest["bounds"])


def actual_saved_grid_records():
    # These are retained headers from the existing real source component, not
    # generated NIfTI arrays. The original scientific files are never opened.
    saved = json.loads((ROOT / "artifacts/lausanne-sub476-annotation-v1/component.json").read_bytes())
    def find(value):
        if isinstance(value, dict):
            if "annotation_raw_grid" in value and "reference_raw_grid" in value:
                return value["annotation_raw_grid"], value["reference_raw_grid"]
            for child in value.values():
                result = find(child)
                if result:
                    return result
        if isinstance(value, list):
            for child in value:
                result = find(child)
                if result:
                    return result
        return None
    result = find(saved)
    assert result, "Actual component must retain its source header proof"
    return result


def test_actual_retained_header_proof_reuses_fixed_envelope():
    mask, reference = actual_saved_grid_records()
    proof = intake.grid_proof(mask, reference)
    assert proof["rule"] == "source_reference_serialization_equivalence"
    assert proof["metrics"]["maximum_observed_float32_steps"] == 3
    assert proof["metrics"]["maximum_allowed_float32_steps"] == 4
    assert proof["metrics"]["all_voxel_centres_keep_reference_index"] is True


def test_relabelled_raw_grid_summary_cannot_supply_proof():
    mask, reference = actual_saved_grid_records()
    mask = copy.deepcopy(mask)
    mask["raw_grid"]["shape"][0] += 1  # Tampered proof metadata, no image creation.
    with pytest.raises((intake.AcquisitionError, ValueError)):
        intake.grid_proof(mask, reference)


def test_destination_rejects_escape_and_symlink(tmp_path, monkeypatch):
    monkeypatch.setattr(intake, "ROOT", tmp_path)
    with pytest.raises(intake.AcquisitionError):
        intake.safe_path(tmp_path / ".." / "outside")
    (tmp_path / "link").symlink_to(tmp_path, target_is_directory=True)
    with pytest.raises(intake.AcquisitionError, match="Symlink"):
        intake.safe_path(tmp_path / "link" / "future-mask")


def test_expired_transfer_refused_before_downloader(metadata, monkeypatch):
    manifest, _ = metadata
    row = manifest["records"][0]
    def call_opener(entry, root, *, opener):
        from urllib.request import Request
        return opener(Request(entry["source_url"]), timeout=45)
    monkeypatch.setattr(intake, "acquire_file", call_opener)
    with pytest.raises(TimeoutError):
        intake.transfer(row, time.monotonic() - 1)


def test_wrong_transfer_url_refused_before_network(metadata, monkeypatch):
    manifest, _ = metadata
    def call_opener(entry, root, *, opener):
        from urllib.request import Request
        return opener(Request("https://example.invalid/unqualified"), timeout=45)
    monkeypatch.setattr(intake, "acquire_file", call_opener)
    with pytest.raises(intake.AcquisitionError, match="whitelist"):
        intake.transfer(manifest["records"][0], time.monotonic() + 2)


def test_second_http_attempt_refused_without_reading_any_body(metadata, monkeypatch):
    manifest, _ = metadata
    row = manifest["records"][0]
    class HeaderOnlyControl:
        status = 200
        headers = {}
        def geturl(self):
            return row["file"]["source_url"]
    monkeypatch.setattr(intake, "open_without_redirect", lambda *a, **k: HeaderOnlyControl())
    def call_opener(entry, root, *, opener):
        from urllib.request import Request
        opener(Request(entry["source_url"]), timeout=45)
        opener(Request(entry["source_url"]), timeout=45)
    monkeypatch.setattr(intake, "acquire_file", call_opener)
    with pytest.raises(intake.AcquisitionError, match="one-request"):
        intake.transfer(row, time.monotonic() + 2)


def test_execution_requires_explicit_frozen_identity():
    with pytest.raises(SystemExit) as raised:
        intake.main(["batch", "--run-id", "control-no-execute"])
    assert raised.value.code == 2


def test_metadata_cache_cannot_shadow_changed_tracked_cohort(tmp_path, monkeypatch):
    monkeypatch.setattr(intake, "ROOT", tmp_path)
    monkeypatch.setattr(intake, "CACHE", tmp_path / "cache")
    payload = b"source metadata control"
    sha = intake.digest(payload)
    entry = {"path": "cohort-control.json", "bytes": len(payload), "sha256": sha}
    (tmp_path / entry["path"]).write_bytes(b"changed metadata")
    cached = intake.CACHE / "metadata" / (sha + ".bin")
    cached.parent.mkdir(parents=True)
    cached.write_bytes(payload)
    assert intake.source_metadata(entry) == payload
    with pytest.raises(intake.AcquisitionError, match="differs"):
        intake.source_metadata(entry, current_tracked=True)


@pytest.mark.parametrize("seconds", [-1, 0, 121, float("nan"), float("inf")])
def test_worker_rejects_unbounded_allowance_before_access(seconds, tmp_path):
    with pytest.raises(intake.AcquisitionError, match="time allowance"):
        intake.worker("not-a-mask", tmp_path, tmp_path, seconds)


def test_batch_control_failure_retains_148_outcomes(metadata, tmp_path, monkeypatch):
    # Fail before source retention/worker launch; only control metadata is saved.
    manifest, sessions = metadata
    monkeypatch.setattr(intake, "ROOT", tmp_path)
    monkeypatch.setattr(intake, "DATA", tmp_path / "data")
    monkeypatch.setattr(intake, "CACHE", tmp_path / "data/cache")
    monkeypatch.setattr(intake, "preflight", lambda: (manifest, sessions))
    def refuse(entry):
        raise intake.AcquisitionError("Deliberate metadata-control failure")
    monkeypatch.setattr(intake, "source_metadata", refuse)
    result = intake.batch("metadata-control", 1, 1)
    assert result["status"] == "interrupted_or_failed"
    assert len(result["outcomes"]) == 148
    assert result["outcome_counts"] == {"deferred_not_attempted": 144, "metadata_failed_excluded": 4}
    assert result["attempted_source_bytes"] == 0
    assert (intake.CACHE / "attempts/metadata-control/batch.json").is_file()
