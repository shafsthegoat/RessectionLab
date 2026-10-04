"""Independent cohort and bounded fake-transport review; never image network IO."""
from copy import deepcopy
import hashlib
from io import BytesIO
import json
from pathlib import Path
import socket
import sys
from urllib.parse import parse_qs, urlparse

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import acquire_btc_case as acquisition
sys.path.pop(0)


@pytest.fixture(autouse=True)
def forbid_live_network(monkeypatch):
    monkeypatch.setattr(socket, "create_connection", lambda *a, **k: pytest.fail("Review attempted live network"))


def test_frozen_roles_and_original_four_manifest_bytes_are_independent_constants():
    path = ROOT / "manifests/experiments/btc-spatial-development-cohort-v1.json"
    assert hashlib.sha256(path.read_bytes()).hexdigest() == "962d964e1d71427f3625cdbebc0f7e4759e5d2345d8f95cb211ed45810ed2985"
    declared = json.loads(path.read_text())
    roles = {row["subject"]: row["development_role"] for row in declared["candidates"]}
    assert roles == {"sub-PAT22": "population_training", "sub-PAT25": "population_training",
                     "sub-PAT26": "checkpoint_selection_development", "sub-PAT27": "checkpoint_selection_development",
                     "sub-PAT29": "unopened_frozen_development_transfer", "sub-PAT31": "unopened_frozen_development_transfer"}
    assert tuple(acquisition.SPATIAL_SUBJECTS) == ("sub-PAT22", "sub-PAT25", "sub-PAT26", "sub-PAT27")
    originals = {"btc_pat05_acquisition.json": "09df59d9ac8b1c8d8d555f0e165148c79c0a2d1a404180451d0b9e90b814badf",
                 "btc_pat16_acquisition.json": "86f251909381a22f339b05bb01fdb0297e7929515cc5486cc53a9dce516e251f",
                 "btc_pat20_acquisition.json": "d6bf19700a3e3e1108b98a5b0eb5c794d68576e2770c42d9967f450e14bb1118",
                 "btc_acquisition.json": "f88d3a01a06e4e21c29de6ba49aff4a54bd230c9fdba4787170b10e19bdd18c4"}
    for name, expected in originals.items():
        assert hashlib.sha256((ROOT / "manifests" / name).read_bytes()).hexdigest() == expected


@pytest.mark.parametrize("subject", ["sub-PAT29", "sub-PAT31"])
def test_sealed_subject_cannot_reenter_through_supplied_verified_manifest(subject, tmp_path, monkeypatch):
    supplied = acquisition.spatial_manifest("sub-PAT22")
    supplied = json.loads(json.dumps(supplied).replace("sub-PAT22", subject))
    for entry in supplied["files"]:
        if entry["sha256"] is None:
            entry["sha256"] = "a" * 64
    supplied["source_content_status"] = "matches_pinned_official_release_checksums"
    supplied["verified_at"] = "2026-10-04T00:00:00Z"
    source = tmp_path / "forged.json"
    source.write_text(json.dumps(supplied))
    destination = tmp_path / "must-not-exist"
    monkeypatch.setattr(acquisition, "acquire_file", lambda *a, **kw: pytest.fail("Sealed subject reached transport"))
    assert acquisition.main(["--manifest", str(source), "--output-root", str(destination)]) == 1
    assert not destination.exists()


def fake_image(*, known_sha=False):
    # Only the declared URL is real; these tiny bytes are constructed test data
    # and cannot satisfy the source acquisition manifest itself.
    entry = deepcopy(next(row for row in acquisition.spatial_manifest("sub-PAT22")["files"]
                          if row["path"].endswith("T1w.nii.gz")))
    payload = b"constructed independent transport body"
    entry.update(bytes=len(payload), expected_bytes=len(payload), expected_md5=hashlib.md5(payload).hexdigest(),
                 sha256=hashlib.sha256(payload).hexdigest() if known_sha else None)
    return entry, payload


class Response(BytesIO):
    def __init__(self, entry, payload, *, attack=None):
        super().__init__(payload)
        self.status = 200
        self.url = entry["source_url"]
        self.headers = {"Content-Length": str(entry["bytes"]), "Content-Encoding": "identity",
                        "x-amz-version-id": parse_qs(urlparse(self.url).query)["versionId"][0]}
        self.read_calls = 0
        if attack == "version": self.headers["x-amz-version-id"] = "wrong-version"
        elif attack == "missing_version": del self.headers["x-amz-version-id"]
        elif attack == "redirect": self.url = self.url.replace("sub-PAT22", "sub-PAT29")
        elif attack == "length": self.headers["Content-Length"] = str(entry["bytes"] + 1)

    def geturl(self):
        return self.url

    def read(self, *args):
        self.read_calls += 1
        return super().read(*args)


@pytest.mark.parametrize("known_sha", [False, True])
@pytest.mark.parametrize("attack", ["version", "missing_version", "redirect", "length"])
def test_both_image_hash_states_require_exact_transport_identity_before_body_read(tmp_path, known_sha, attack):
    entry, payload = fake_image(known_sha=known_sha)
    response = Response(entry, payload, attack=attack)
    with pytest.raises(acquisition.AcquisitionError):
        acquisition.acquire_file(entry, tmp_path, opener=lambda *a, **kw: response)
    assert response.read_calls == 0
    assert not (tmp_path / entry["path"]).exists()


@pytest.mark.parametrize("known_sha", [False, True])
def test_valid_constructed_transport_and_existing_file_are_both_immutable(tmp_path, known_sha):
    entry, payload = fake_image(known_sha=known_sha)
    seen = []
    def opener(request, **kwargs):
        seen.append(request.full_url)
        return Response(entry, payload)
    assert acquisition.acquire_file(entry, tmp_path, opener=opener) == "downloaded_and_verified"
    path = tmp_path / entry["path"]
    assert path.read_bytes() == payload and seen == [entry["source_url"]]
    assert acquisition.acquire_file(entry, tmp_path,
        opener=lambda *a, **kw: pytest.fail("Existing verified image issued GET")) == "already_verified"
    path.write_bytes(b"X" * len(payload))
    with pytest.raises(acquisition.AcquisitionError):
        acquisition.acquire_file(entry, tmp_path, opener=lambda *a, **kw: pytest.fail("Corrupt image was overwritten"))
    assert path.read_bytes() == b"X" * len(payload)


@pytest.mark.parametrize("attack", ["md5", "oversized_body", "truncated_body"])
def test_constructed_annex_failure_does_not_publish_image_or_sha(tmp_path, attack):
    entry, payload = fake_image()
    body = payload
    if attack == "md5": entry["expected_md5"] = "0" * 32
    elif attack == "oversized_body": body += b"extra"
    else: body = body[:-1]
    with pytest.raises(acquisition.AcquisitionError):
        acquisition.acquire_file(entry, tmp_path, opener=lambda *a, **kw: Response(entry, body))
    assert not (tmp_path / entry["path"]).exists()
    assert entry["sha256"] is None
