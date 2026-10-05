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
        # Valid S3 image responses retain exact identity even after a source
        # SHA256 was measured. Explicit attack headers can override these.
        self.headers = {"Content-Length": str(len(body)), "x-amz-version-id": "fixed", **(headers or {})}

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


@pytest.mark.parametrize("subject,total", [("sub-PAT16", 18_210_576), ("sub-PAT20", 17_546_223)])
def test_queue_is_exact_structural_scope_with_uninvented_image_hashes(subject, total, tmp_path):
    manifest = acquisition.queued_manifest(subject)
    entries = acquisition.validate_manifest(manifest, tmp_path, allow_pending_annex=True)
    assert len(entries) == 7 and sum(entry["bytes"] for entry in entries) == total
    assert sum(entry["sha256"] is None for entry in entries) == 2
    assert all("/dwi/" not in entry["path"] and "postop" not in entry["path"] for entry in entries)
    assert manifest["role"] == "development" and manifest["outer_role_locked"] is True
    assert manifest["eligible_for_outer_final"] is False
    with pytest.raises(acquisition.AcquisitionError, match="pending annex"):
        acquisition.validate_manifest(manifest, tmp_path)


def test_queue_cannot_be_edited_after_freeze(tmp_path, monkeypatch):
    changed = tmp_path / "queue.json"
    changed.write_bytes(acquisition.QUEUE_MANIFEST.read_bytes() + b" ")
    monkeypatch.setattr(acquisition, "QUEUE_MANIFEST", changed)
    with pytest.raises(acquisition.AcquisitionError, match="frozen identity"):
        acquisition.queued_manifest("sub-PAT16")


@pytest.mark.parametrize("subject", ["sub-PAT05", "sub-PAT22", "sub-PAT28", "../../other"])
def test_queue_never_generalizes_unselected_subjects(subject):
    with pytest.raises(acquisition.AcquisitionError, match="outside the frozen"):
        acquisition.queued_manifest(subject)


@pytest.mark.parametrize("mutation", ["role", "queue_hash", "md5", "object_version", "metadata_sha", "diffusion"])
def test_queued_manifest_cannot_rebind_selection_or_source(tmp_path, mutation):
    manifest = acquisition.queued_manifest("sub-PAT16")
    image = next(item for item in manifest["files"] if "expected_md5" in item)
    if mutation == "role":
        manifest["role"] = "final_evaluation"
    elif mutation == "queue_hash":
        manifest["selection_queue_sha256"] = "0" * 64
    elif mutation == "md5":
        image["expected_md5"] = "0" * 32
    elif mutation == "object_version":
        image["source_url"] += "-different"
    elif mutation == "metadata_sha":
        manifest["files"][0]["sha256"] = "0" * 64
    else:
        manifest["files"].append({"path": "sub-PAT16/ses-preop/dwi/unrequested.nii.gz"})
    with pytest.raises(acquisition.AcquisitionError):
        acquisition.validate_manifest(manifest, tmp_path, allow_pending_annex=True)


def test_queue_selection_is_source_metadata_only_and_order_independent():
    rows = ["sub-PAT28\tGlioma II", "sub-PAT20\tAnaplastic astrocytoma III",
            "sub-PAT05\tOligo-astrocytoma II", "sub-PAT01\tMeningioma I",
            "sub-PAT22\tOligodendroglioma II", "sub-PAT16\tAnaplastic astrocytoma II-III"]
    for ordered in (rows, list(reversed(rows))):
        metadata = "participant_id\ttumor type & grade\n" + "\n".join(ordered)
        assert acquisition.selected_queued_subjects(metadata) == ["sub-PAT16", "sub-PAT20"]


@pytest.mark.parametrize("subject,total", [("sub-PAT16", 18_210_576), ("sub-PAT20", 17_546_223)])
def test_queued_dry_run_is_offline_and_nonmutating(subject, total, tmp_path, capsys, monkeypatch):
    monkeypatch.setattr(acquisition, "acquire_file", lambda *a, **k: pytest.fail("Unexpected fetch"))
    output = tmp_path / "unused"
    assert acquisition.main(["--queued-subject", subject, "--dry-run", "--output-root", str(output)]) == 0
    assert json.loads(capsys.readouterr().out)["bytes"] == total
    assert not output.exists()


def test_pending_annex_download_verifies_before_publish_and_reuses_existing(tmp_path, entry, content):
    entry["sha256"] = None
    headers = {"x-amz-version-id": "fixed", "Content-Length": str(len(content))}
    assert acquisition.acquire_file(entry, tmp_path, opener=lambda *a, **k: Response(content, headers=headers)) == "downloaded_and_verified"
    target = tmp_path / entry["path"]
    assert hashlib.sha256(target.read_bytes()).hexdigest() == hashlib.sha256(content).hexdigest()
    original_mtime = target.stat().st_mtime_ns
    assert acquisition.acquire_file(entry, tmp_path, opener=lambda *a, **k: pytest.fail("Unexpected fetch")) == "already_verified"
    assert target.stat().st_mtime_ns == original_mtime
    assert entry["sha256"] is None  # The caller records the measured hash explicitly.


@pytest.mark.parametrize("failure", ["checksum", "version", "length", "compressed"])
def test_pending_annex_rejects_wrong_content_before_publication(tmp_path, entry, content, failure):
    entry["sha256"] = None
    body = b"x" * len(content) if failure == "checksum" else content
    headers = {"x-amz-version-id": "other" if failure == "version" else "fixed",
               "Content-Length": str(len(content) - (failure == "length"))}
    if failure == "compressed":
        headers["Content-Encoding"] = "gzip"
    with pytest.raises(acquisition.AcquisitionError):
        acquisition.acquire_file(entry, tmp_path, opener=lambda *a, **k: Response(body, headers=headers))
    assert not (tmp_path / entry["path"]).exists()


def test_pending_annex_resume_rejects_wrong_ranges_then_completes(tmp_path, entry, content):
    entry["sha256"] = None
    partial = tmp_path / (entry["path"] + ".partial")
    partial.parent.mkdir()
    partial.write_bytes(content[:8])
    headers = {"x-amz-version-id": "fixed", "Content-Length": str(len(content) - 8),
               "Content-Range": f"bytes 0-{len(content) - 1}/{len(content)}"}
    with pytest.raises(acquisition.AcquisitionError, match="Content-Range"):
        acquisition.acquire_file(entry, tmp_path, opener=lambda *a, **k: Response(content[8:], status=206, headers=headers))
    assert partial.read_bytes() == content[:8]
    headers["Content-Range"] = f"bytes 8-{len(content) - 1}/{len(content)}"
    def resume(request, timeout):
        assert request.headers["Range"] == "bytes=8-"
        return Response(content[8:], status=206, headers=headers)
    assert acquisition.acquire_file(entry, tmp_path, opener=resume) == "downloaded_and_verified"
    assert (tmp_path / entry["path"]).read_bytes() == content
