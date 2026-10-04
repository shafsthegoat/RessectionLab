"""Offline integrity and interrupted-transfer checks for the real-case fetcher."""

from __future__ import annotations

import hashlib
import importlib.util
import io
import json
from pathlib import Path
import sys

import pytest


SCRIPT_DIR = Path(__file__).resolve().parents[1] / "scripts"
SPEC = importlib.util.spec_from_file_location("acquire_btc_case", SCRIPT_DIR / "acquire_btc_case.py")
acquisition = importlib.util.module_from_spec(SPEC)
sys.path.insert(0, str(SCRIPT_DIR))
try:
    SPEC.loader.exec_module(acquisition)
finally:
    sys.path.pop(0)


class Response(io.BytesIO):
    def __init__(self, body, *, status=200, headers=None):
        super().__init__(body)
        self.status = status
        self.headers = headers or {}

    def geturl(self):
        return "https://s3.amazonaws.com/openneuro.org/ds001226/test?versionId=fixed"


@pytest.fixture
def content():
    return b"Pinned diffusion bytes with fractional source annotation"


@pytest.fixture
def entry(content):
    return {"path": "subject/image.nii.gz", "bytes": len(content),
            "sha256": hashlib.sha256(content).hexdigest(),
            "expected_md5": hashlib.md5(content).hexdigest(),
            "source_url": "https://s3.amazonaws.com/openneuro.org/ds001226/test?versionId=fixed"}


@pytest.fixture
def manifest():
    return json.loads(acquisition.DEFAULT_MANIFEST.read_text())


def test_reviewed_manifest_is_closed_and_small(manifest, tmp_path):
    files = acquisition.validate_manifest(manifest, tmp_path)
    assert len(files) == 15
    assert sum(entry["bytes"] for entry in files) == 64_808_763
    assert len([entry for entry in files if "expected_md5" in entry]) == 4
    assert all("ses-postop" not in entry["path"] for entry in files)
    assert manifest["annotation_semantics"]["clinical_probability"] is False
    assert manifest["clinical_context"]["eligible_as_preoperative_policy_input"] is False


@pytest.mark.parametrize("key,value", [("release", "latest"), ("git_commit", "main"),
                                      ("license", "unknown"), ("subject", "sub-PAT29")])
def test_unreviewed_identity_rejected(manifest, tmp_path, key, value):
    manifest[key] = value
    with pytest.raises(acquisition.AcquisitionError, match="reviewed source"):
        acquisition.validate_manifest(manifest, tmp_path)


def test_missing_gradient_file_rejected_before_fetch(manifest, tmp_path):
    manifest["files"] = [entry for entry in manifest["files"] if not entry["path"].endswith("AP_dwi.bvec")]
    with pytest.raises(acquisition.AcquisitionError, match="15 reviewed"):
        acquisition.validate_manifest(manifest, tmp_path)


@pytest.mark.parametrize("mutation", ["unversioned", "mirror", "bad_md5", "length_disagrees"])
def test_unpinned_imaging_rejected(manifest, tmp_path, mutation):
    image = next(entry for entry in manifest["files"] if "expected_md5" in entry)
    if mutation == "unversioned":
        image["source_url"] = image["source_url"].split("?")[0]
    elif mutation == "mirror":
        image["source_url"] = image["source_url"].replace("s3.amazonaws.com", "example.org")
    elif mutation == "bad_md5":
        image["expected_md5"] = "not-a-source-checksum"
    else:
        image["expected_bytes"] += 1
    with pytest.raises(acquisition.AcquisitionError):
        acquisition.validate_manifest(manifest, tmp_path)


def test_valid_download_and_repeat_are_immutable(tmp_path, entry, content):
    assert acquisition.acquire_file(entry, tmp_path, opener=lambda *args, **kwargs: Response(content)) == "downloaded_and_verified"
    target = tmp_path / entry["path"]
    initial_mtime = target.stat().st_mtime_ns
    assert acquisition.acquire_file(entry, tmp_path, opener=lambda *args, **kwargs: pytest.fail("Unexpected network")) == "already_verified"
    assert target.read_bytes() == content
    assert target.stat().st_mtime_ns == initial_mtime


def test_existing_corruption_is_preserved_and_reported(tmp_path, entry, content):
    target = tmp_path / entry["path"]
    target.parent.mkdir()
    corrupt = b"x" * len(content)
    target.write_bytes(corrupt)
    with pytest.raises(acquisition.AcquisitionError, match="SHA256"):
        acquisition.acquire_file(entry, tmp_path, opener=lambda *args, **kwargs: pytest.fail("Unexpected network"))
    assert target.read_bytes() == corrupt


def test_corrupt_download_is_not_published(tmp_path, entry, content):
    with pytest.raises(acquisition.AcquisitionError, match="SHA256"):
        acquisition.acquire_file(entry, tmp_path, opener=lambda *args, **kwargs: Response(b"x" * len(content)))
    assert not (tmp_path / entry["path"]).exists()


def test_annex_md5_is_independently_checked(tmp_path, entry, content):
    target = tmp_path / entry["path"]
    target.parent.mkdir()
    target.write_bytes(content)
    entry["expected_md5"] = "0" * 32
    with pytest.raises(acquisition.AcquisitionError, match="annex MD5"):
        acquisition.verify_file(target, entry)


def test_interrupted_transfer_resumes_with_byte_range(tmp_path, entry, content):
    class Interrupted(Response):
        def read(self, size=-1):
            if self.tell():
                raise OSError("simulated dropped connection")
            return super().read(8)

    with pytest.raises(OSError, match="dropped connection"):
        acquisition.acquire_file(entry, tmp_path, opener=lambda *args, **kwargs: Interrupted(content))
    partial = tmp_path / (entry["path"] + ".partial")
    assert partial.read_bytes() == content[:8]

    def resume(request, timeout):
        assert request.headers["Range"] == "bytes=8-"
        return Response(content[8:], status=206,
                        headers={"Content-Range": f"bytes 8-{len(content)-1}/{len(content)}"})

    assert acquisition.acquire_file(entry, tmp_path, opener=resume) == "downloaded_and_verified"
    assert (tmp_path / entry["path"]).read_bytes() == content
    assert not partial.exists()


def test_wrong_resume_range_does_not_change_partial(tmp_path, entry, content):
    partial = tmp_path / (entry["path"] + ".partial")
    partial.parent.mkdir()
    partial.write_bytes(content[:8])
    with pytest.raises(acquisition.AcquisitionError, match="Content-Range"):
        acquisition.acquire_file(entry, tmp_path, opener=lambda *args, **kwargs: Response(
            content, status=206, headers={"Content-Range": f"bytes 0-{len(content)-1}/{len(content)}"}))
    assert partial.read_bytes() == content[:8]


def test_verify_only_missing_files_fails_without_fetch(tmp_path, capsys, monkeypatch):
    monkeypatch.setattr(acquisition, "acquire_file", lambda *args, **kwargs: pytest.fail("Unexpected network"))
    assert acquisition.main(["--verify-only", "--output-root", str(tmp_path)]) == 1
    assert "failed" in capsys.readouterr().err


def test_dry_run_does_not_create_output_or_fetch(tmp_path, capsys, monkeypatch):
    monkeypatch.setattr(acquisition, "acquire_file", lambda *args, **kwargs: pytest.fail("Unexpected network"))
    output = tmp_path / "unused"
    assert acquisition.main(["--dry-run", "--output-root", str(output)]) == 0
    assert not output.exists()
    assert json.loads(capsys.readouterr().out)["bytes"] == 64_808_763


def test_predeclared_structural_case_has_only_seven_files(tmp_path):
    path = SCRIPT_DIR.parent / "manifests/btc_pat05_acquisition.json"
    manifest = json.loads(path.read_text())
    files = acquisition.validate_manifest(manifest, tmp_path)
    assert len(files) == 7
    assert sum(entry["bytes"] for entry in files) == 17_862_831
    assert sum("expected_md5" in entry for entry in files) == 2
    assert all("/dwi/" not in entry["path"] and "postop" not in entry["path"] for entry in files)
    assert manifest["clinical_context"]["eligible_as_preoperative_policy_input"] is False


def test_structural_selection_is_reproducible_without_images():
    metadata = "participant_id\ttumor type & grade\nsub-PAT28\tOligodendroglioma II\nsub-PAT16\tAnaplastic astrocytoma II-III\nsub-PAT01\tMeningioma I\nsub-PAT05\tOligo-astrocytoma II\n"
    assert acquisition.select_additional_structural_subject(metadata) == "sub-PAT05"
    with pytest.raises(acquisition.AcquisitionError, match="No eligible"):
        acquisition.select_additional_structural_subject("participant_id\ttumor type & grade\nsub-PAT28\tGlioma II\n")


def test_changed_structural_selection_cannot_relabel_case(tmp_path):
    manifest = json.loads((SCRIPT_DIR.parent / "manifests/btc_pat05_acquisition.json").read_text())
    manifest["selection_manifest_sha256"] = "0" * 64
    with pytest.raises(acquisition.AcquisitionError, match="unchanged predeclared"):
        acquisition.validate_manifest(manifest, tmp_path)


def test_pat28_diffusion_cannot_be_silently_dropped(manifest, tmp_path):
    manifest["subject"] = "sub-PAT05"
    with pytest.raises(acquisition.AcquisitionError, match="7 reviewed"):
        acquisition.validate_manifest(manifest, tmp_path)
