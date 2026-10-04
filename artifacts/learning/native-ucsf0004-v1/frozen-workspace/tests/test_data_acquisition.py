"""Network-free regression tests for immutable, bounded public acquisition."""

from __future__ import annotations

import hashlib
import importlib.util
import io
from pathlib import Path

import pytest


SPEC = importlib.util.spec_from_file_location(
    "acquire_public_case", Path(__file__).resolve().parents[1] / "scripts/acquire_public_case.py"
)
acquisition = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(acquisition)


class Response(io.BytesIO):
    def __init__(self, body, status=200, headers=None):
        super().__init__(body)
        self.status = status
        self.headers = headers or {}

    def geturl(self):
        return "https://example.org/public-case"


@pytest.fixture
def entry():
    payload = b"verified public source bytes"
    return {
        "path": "case/image.nii.gz",
        "source_url": "https://example.org/public-case",
        "size_bytes": len(payload),
        "sha256": hashlib.sha256(payload).hexdigest(),
    }


def test_first_download_is_hashed_and_idempotent(tmp_path, entry):
    def opener(request, timeout):
        return Response(b"verified public source bytes")

    assert acquisition.acquire_file(entry, tmp_path, opener=opener) == "downloaded_and_verified"
    assert acquisition.acquire_file(entry, tmp_path, opener=lambda *a, **k: pytest.fail("unexpected fetch")) == "already_verified"
    assert not (tmp_path / (entry["path"] + ".partial")).exists()


def test_corrupt_existing_file_is_preserved(tmp_path, entry):
    path = tmp_path / entry["path"]
    path.parent.mkdir()
    path.write_bytes(b"someone else's file")
    with pytest.raises(acquisition.AcquisitionError, match="Size mismatch"):
        acquisition.acquire_file(entry, tmp_path, opener=lambda *a, **k: pytest.fail("unexpected fetch"))
    assert path.read_bytes() == b"someone else's file"


def test_resume_checks_exact_content_range(tmp_path, entry):
    payload = b"verified public source bytes"
    partial = tmp_path / (entry["path"] + ".partial")
    partial.parent.mkdir()
    partial.write_bytes(payload[:8])

    def opener(request, timeout):
        assert request.headers["Range"] == "bytes=8-"
        return Response(payload[8:], 206, {"Content-Range": f"bytes 8-{len(payload)-1}/{len(payload)}"})

    acquisition.acquire_file(entry, tmp_path, opener=opener)
    assert (tmp_path / entry["path"]).read_bytes() == payload


def test_server_ignoring_range_restarts_only_partial(tmp_path, entry):
    payload = b"verified public source bytes"
    partial = tmp_path / (entry["path"] + ".partial")
    partial.parent.mkdir()
    partial.write_bytes(payload[:8])
    acquisition.acquire_file(entry, tmp_path, opener=lambda *a, **k: Response(payload))
    assert (tmp_path / entry["path"]).read_bytes() == payload


def test_wrong_resume_range_is_rejected_without_altering_partial(tmp_path, entry):
    partial = tmp_path / (entry["path"] + ".partial")
    partial.parent.mkdir()
    partial.write_bytes(b"verified")
    with pytest.raises(acquisition.AcquisitionError, match="Content-Range"):
        acquisition.acquire_file(entry, tmp_path, opener=lambda *a, **k: Response(b"incorrect", 206, {"Content-Range": "bytes 0-8/9"}))
    assert partial.read_bytes() == b"verified"
    assert not (tmp_path / entry["path"]).exists()


@pytest.mark.parametrize("body", [b"x" * 28, b"x" * 29])
def test_hash_mismatch_or_oversize_never_publishes(tmp_path, entry, body):
    with pytest.raises(acquisition.AcquisitionError):
        acquisition.acquire_file(entry, tmp_path, opener=lambda *a, **k: Response(body))
    assert not (tmp_path / entry["path"]).exists()


@pytest.mark.parametrize("relative", ["../escape", "/tmp/escape", "a/../../escape"])
def test_manifest_path_escape_rejected(tmp_path, relative):
    with pytest.raises(acquisition.AcquisitionError):
        acquisition.checked_path(tmp_path, relative)


def test_symlink_destination_escape_rejected(tmp_path):
    outside = tmp_path.parent / (tmp_path.name + "-outside")
    outside.mkdir()
    (tmp_path / "link").symlink_to(outside, target_is_directory=True)
    with pytest.raises(acquisition.AcquisitionError):
        acquisition.checked_path(tmp_path, "link/image.nii.gz")


def test_partial_symlink_escape_rejected(tmp_path, entry):
    path = tmp_path / (entry["path"] + ".partial")
    path.parent.mkdir()
    path.symlink_to(tmp_path.parent / "outside.partial")
    with pytest.raises(acquisition.AcquisitionError):
        acquisition.acquire_file(entry, tmp_path)


def test_budget_and_duplicate_destinations_rejected(tmp_path, entry):
    with pytest.raises(acquisition.AcquisitionError, match="download limit"):
        acquisition.validate_manifest({"files": [entry], "storage_forecast": {"max_download_bytes": 1}}, tmp_path)
    with pytest.raises(acquisition.AcquisitionError, match="Duplicate"):
        acquisition.validate_manifest({"files": [entry, entry], "storage_forecast": {"max_download_bytes": 1000}}, tmp_path)
