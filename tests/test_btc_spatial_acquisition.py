"""Offline safeguards for four prospective BTC training/selection cases."""
from copy import deepcopy
import hashlib
import importlib.util
import json
from pathlib import Path
import re
import sys

import pytest


SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
spec = importlib.util.spec_from_file_location("spatial_acquisition", SCRIPTS / "acquire_btc_case.py")
acquisition = importlib.util.module_from_spec(spec)
sys.path.insert(0, str(SCRIPTS))
try:
    spec.loader.exec_module(acquisition)
finally:
    sys.path.pop(0)


@pytest.mark.parametrize("subject", acquisition.SPATIAL_SUBJECTS)
def test_four_case_structural_contract_is_source_bound(subject, tmp_path):
    manifest = acquisition.spatial_manifest(subject)
    files = acquisition.validate_manifest(manifest, tmp_path, allow_pending_annex=True)
    assert len(files) == 7
    assert sum(f["bytes"] for f in files) == acquisition.DOWNLOAD_LIMITS[subject]
    assert sum(f["sha256"] is None for f in files) == 2
    assert all("/dwi/" not in f["path"] and "postop" not in f["path"] for f in files)
    expected_role = "population_training" if subject in acquisition.SPATIAL_SUBJECTS[:2] else "checkpoint_selection_development"
    assert manifest["development_role"] == expected_role
    assert manifest["anatomical_gates"]["working_brain_mask"] is None
    assert manifest["anatomical_gates"]["nonzero_MRI_support_allowed"] is False
    assert manifest["clinical_context"]["eligible_as_preoperative_policy_input"] is False
    with pytest.raises(acquisition.AcquisitionError, match="SHA256 is pending"):
        acquisition.validate_manifest(manifest, tmp_path)


def test_committed_pointer_lengths_hashes_and_roles_agree():
    raw = acquisition.SPATIAL_MANIFEST.read_bytes()
    assert hashlib.sha256(raw).hexdigest() == acquisition.SPATIAL_SHA256
    declaration = json.loads(raw)
    groups = [x["patient_group"] for x in declaration["existing_development_records"] + declaration["candidates"]]
    assert len(groups) == len(set(groups)) == 10
    assert declaration["first_structural_batch"]["subjects"] == list(acquisition.SPATIAL_SUBJECTS)
    assert declaration["first_structural_batch"]["forbidden_subjects"] == list(acquisition.SPATIAL_TRANSFER_SUBJECTS)
    for case in declaration["candidates"]:
        assert case["outer_role"] == "development" and case["eligible_for_external_final"] is False
        assert case["acquisition_permitted_in_first_structural_batch"] == (case["subject"] in acquisition.SPATIAL_SUBJECTS)
        for file in case["files"]:
            if "annex_pointer" in file:
                pointer = file["annex_pointer"]
                assert hashlib.sha256(pointer.encode()).hexdigest() == file["annex_pointer_sha256"]
                size, md5 = re.search(r"MD5E-s([0-9]+)--([a-f0-9]{32})", pointer).groups()
                assert int(size) == file["expected_bytes"]
                assert md5 == file["expected_annex_md5"] == file["source_object_metadata"]["etag"]
                assert file["source_object_metadata"]["image_body_requested"] is False


@pytest.mark.parametrize("subject", ("sub-PAT29", "sub-PAT31", "sub-PAT99", "../sub-PAT22"))
def test_unopened_and_arbitrary_patients_have_no_fetch_path(subject, tmp_path, monkeypatch):
    monkeypatch.setattr(acquisition, "acquire_file", lambda *a, **k: pytest.fail("Unexpected network"))
    with pytest.raises(acquisition.AcquisitionError, match="outside"):
        acquisition.spatial_manifest(subject)
    with pytest.raises(acquisition.AcquisitionError, match="outside"):
        acquisition.expected_paths(subject)
    with pytest.raises(SystemExit) as exc:
        acquisition.main(["--spatial-subject", subject, "--output-root", str(tmp_path)])
    assert exc.value.code == 2


def test_changed_source_declaration_is_rejected_before_fetch(tmp_path, monkeypatch):
    changed = tmp_path / "changed.json"
    changed.write_bytes(acquisition.SPATIAL_MANIFEST.read_bytes() + b" ")
    monkeypatch.setattr(acquisition, "SPATIAL_MANIFEST", changed)
    with pytest.raises(acquisition.AcquisitionError, match="frozen identity"):
        acquisition.spatial_manifest("sub-PAT22")


@pytest.mark.parametrize("mutation", [
    lambda m: m.update(development_role="population_training", subject="sub-PAT26"),
    lambda m: m.update(development_role="checkpoint_selection_development"),
    lambda m: m.update(patient_group="BTC:sub-PAT28"),
    lambda m: m.update(role="final"),
    lambda m: m.update(selection_cohort_sha256="0" * 64),
    lambda m: m["clinical_context"].update(eligible_as_preoperative_policy_input=True),
    lambda m: m["anatomical_gates"].update(cortical_access_permitted=True),
    lambda m: m["files"][-1].update(source_url=m["files"][-1]["source_url"].split("?")[0] + "?versionId=other"),
    lambda m: m["files"][-1].update(expected_md5="0" * 32),
    lambda m: m["files"][-1].update(bytes=m["files"][-1]["bytes"] + 1),
    lambda m: m["files"].pop(),
    lambda m: m["files"].append(deepcopy(m["files"][-1])),
])
def test_declared_role_source_or_anatomy_cannot_be_changed(tmp_path, mutation):
    manifest = acquisition.spatial_manifest("sub-PAT22")
    mutation(manifest)
    with pytest.raises(acquisition.AcquisitionError):
        acquisition.validate_manifest(manifest, tmp_path, allow_pending_annex=True)


def test_completed_sha256_does_not_weaken_source_binding(tmp_path):
    manifest = acquisition.spatial_manifest("sub-PAT22")
    for file in manifest["files"]:
        if file["sha256"] is None:
            file["sha256"] = "a" * 64
    assert len(acquisition.validate_manifest(manifest, tmp_path)) == 7
    # A syntactically valid measured SHA is not source validation: annex MD5
    # remains independently mandatory when actual content is read.
    image = next(f for f in manifest["files"] if "expected_md5" in f)
    image["expected_md5"] = "b" * 32
    with pytest.raises(acquisition.AcquisitionError, match="frozen source"):
        acquisition.validate_manifest(manifest, tmp_path)


@pytest.mark.parametrize("subject", acquisition.SPATIAL_SUBJECTS)
def test_dry_run_is_offline_and_creates_nothing(subject, tmp_path, monkeypatch, capsys):
    output = tmp_path / "unused"
    monkeypatch.setattr(acquisition, "acquire_file", lambda *a, **k: pytest.fail("Unexpected network"))
    assert acquisition.main(["--spatial-subject", subject, "--dry-run", "--output-root", str(output)]) == 0
    report = json.loads(capsys.readouterr().out)
    assert report["subject"] == subject and report["files"] == 7
    assert report["bytes"] == acquisition.DOWNLOAD_LIMITS[subject]
    assert not output.exists()


def test_existing_four_modes_and_defaults_preserved():
    assert acquisition.SUBJECT == "sub-PAT28"
    assert acquisition.QUEUED_SUBJECTS == ("sub-PAT16", "sub-PAT20")
    assert len(acquisition.expected_paths("sub-PAT28")) == 15
    assert all(len(acquisition.expected_paths(s)) == 7 for s in ("sub-PAT05", "sub-PAT16", "sub-PAT20"))


def test_metadata_selection_is_deterministic_and_excludes_ependymoma():
    ids = ["sub-PAT31", "sub-PAT05", "sub-PAT16", "sub-PAT20", "sub-PAT22", "sub-PAT25", "sub-PAT26", "sub-PAT27", "sub-PAT28", "sub-PAT29"]
    metadata = "participant_id\ttumor type & grade\n" + "".join(s + "\tAstrocytoma II\n" for s in ids)
    metadata += "sub-PAT07\tEpendymoma II\nsub-CON01\tnone\nsub-PAT01\tMeningioma I\n"
    assert acquisition.selected_spatial_subjects(metadata) == list(acquisition.SPATIAL_SUBJECTS + acquisition.SPATIAL_TRANSFER_SUBJECTS)
