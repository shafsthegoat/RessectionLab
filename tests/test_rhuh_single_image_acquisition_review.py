"""Independent analytical review; no real images, provider, client or decoder."""
import hashlib
import importlib.util
import io
import json
from pathlib import Path
import signal
from types import SimpleNamespace

import pytest


MODULE = Path(__file__).resolve().parents[1] / "scripts/acquire_rhuh_single_image.py"
SPEC = importlib.util.spec_from_file_location("rhuh_image_independent_review", MODULE)
image = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(image)


@pytest.fixture(autouse=True)
def forbid_live_activity(monkeypatch):
    def forbidden(*args, **kwargs):
        pytest.fail("Unsimulated network or process access in analytical review")
    monkeypatch.setattr(image, "request", forbidden)
    monkeypatch.setattr(image.subprocess, "Popen", forbidden)


@pytest.fixture
def isolated(tmp_path, monkeypatch):
    monkeypatch.setattr(image, "ROOT", tmp_path)
    monkeypatch.setattr(image, "QUARANTINE_ROOT", tmp_path / "quarantine")
    monkeypatch.setattr(image.public_transfer.signal, "signal", lambda *args: None)
    monkeypatch.setattr(image.public_transfer.signal, "setitimer", lambda *args: None)
    return tmp_path


@pytest.fixture
def provenance(isolated, monkeypatch):
    """Constructed metadata only; no real image or checksum index is opened."""
    proposal = isolated / "proposal.json"
    index = isolated / "index.sums"
    rows = [[image.PUBLISHED_TOKEN, image.SOURCE[1:]]]
    rows += [["1" * 32, f"fixture/path-{i}"] for i in range(719)]
    record = {"selection": {"public_source_path": image.SOURCE,
        "published_checksum_token": image.PUBLISHED_TOKEN, "patient": "RHUH-0001",
        "visit": 0, "modality_source_label": "T1"}}
    monkeypatch.setattr(image, "PROPOSAL", proposal)
    monkeypatch.setattr(image, "INDEX", index)
    def bind():
        index.write_text("\n".join(" ".join(row) for row in rows) + "\n")
        index_hash = hashlib.sha256(index.read_bytes()).hexdigest()
        record["selection"]["checksum_index_sha256"] = index_hash
        proposal.write_text(json.dumps(record))
        monkeypatch.setattr(image, "INDEX_SHA256", index_hash)
        monkeypatch.setattr(image, "PROPOSAL_SHA256", hashlib.sha256(proposal.read_bytes()).hexdigest())
    bind()
    return record, rows, bind


@pytest.mark.parametrize("mutation", ["proposal_bytes", "index_bytes", "wrong_visit", "duplicate_selected", "wrong_token"])
def test_bad_provenance_stops_before_fresh_authorization(provenance, mutation):
    record, rows, bind = provenance
    if mutation == "proposal_bytes":
        image.PROPOSAL.write_bytes(image.PROPOSAL.read_bytes() + b" ")
    elif mutation == "index_bytes":
        image.INDEX.write_bytes(image.INDEX.read_bytes() + b" ")
    else:
        if mutation == "wrong_visit":
            record["selection"]["visit"] = 1
        elif mutation == "duplicate_selected":
            rows[1] = list(rows[0])
        elif mutation == "wrong_token":
            rows[0][0] = "0" * 32
        bind()
    with pytest.raises(image.Rejected):
        image.fresh_spec(image.time.monotonic() + 5)


def spec_fixture(monkeypatch):
    source = "/analytical-server/package" + image.SOURCE
    monkeypatch.setattr(image, "RESOLVED_SOURCE_SHA256", hashlib.sha256(source.encode()).hexdigest())
    return {"paths": [{"source": source}], "direction": "receive", "authentication": "token",
        "remote_host": image.public_transfer.HOST, "remote_user": "public-fixture",
        "ssh_port": 33001, "fasp_port": 33001, "cipher": "aes-128",
        "token": "NOT_A_REAL_TRANSFER_TOKEN", "cookie": "NOT_A_REAL_COOKIE"}


def test_valid_binding_requests_only_fixed_t1(provenance, monkeypatch):
    spec = spec_fixture(monkeypatch)
    replies = iter([
        (200, {}, b'<a href="https://faspex.cancerimagingarchive.net/public?context=fixture">x</a>'),
        (200, {}, b"client_id: 'public', redirect_uri: '/aspera/faspex/token'"),
        (302, {"Location": "/aspera/faspex/token?code=fixture"}, b""),
        (201, {}, b'{"access_token":"NOT_A_REAL_ACCESS_TOKEN"}'),
        (201, {}, json.dumps(spec).encode()),
    ])
    calls = []
    def request(url, deadline, **kwargs):
        calls.append((url, kwargs))
        return next(replies)
    monkeypatch.setattr(image, "request", request)
    received = image.fresh_spec(image.time.monotonic() + 5)
    image.validate_spec(received)
    assert len(calls) == 5
    assert json.loads(calls[-1][1]["data"]) == {"paths": [{"path": image.SOURCE}]}
    assert image.SOURCE.endswith("/RHUH-0001/0/RHUH-0001_0_t1.nii.gz")


def inert_payload(isolated, monkeypatch, raw=b"\x1f\x8bnot-a-valid-gzip-or-image"):
    directory = image.QUARANTINE_ROOT / "run-fixture"
    directory.mkdir(parents=True)
    target = directory / Path(image.SOURCE).name
    target.write_bytes(raw)
    monkeypatch.setattr(image, "PUBLISHED_TOKEN", hashlib.md5(raw, usedforsecurity=False).hexdigest())
    return directory, target, raw


def test_compatibility_does_not_decode_or_claim_known_length(isolated, monkeypatch):
    directory, _, raw = inert_payload(isolated, monkeypatch)
    result = image.verify_payload(directory)
    assert image.EXPECTED_BYTES is None
    assert result["measured_compressed_bytes"] == len(raw)
    assert result["expected_compressed_bytes"] is None and result["prior_length_match_claim"] is False
    assert result["sha256"] == hashlib.sha256(raw).hexdigest()
    assert result["candidate_md5_of_original_compressed_bytes"] == hashlib.md5(raw, usedforsecurity=False).hexdigest()
    assert result["publisher_algorithm_confirmed"] is False
    assert result["decoded"] is False and result["scientific_use_released"] is False


def test_mismatch_is_preserved_without_alternate_hash_or_decode(isolated, monkeypatch):
    directory, _, raw = inert_payload(isolated, monkeypatch)
    monkeypatch.setattr(image, "PUBLISHED_TOKEN", "0" * 32)
    with pytest.raises(image.CompatibilityMismatch) as caught:
        image.verify_payload(directory)
    observed = caught.value.observation
    assert observed["candidate_matches_published_token"] is False
    assert observed["candidate_md5_of_original_compressed_bytes"] == hashlib.md5(raw, usedforsecurity=False).hexdigest()
    assert observed["measured_compressed_bytes"] == len(raw) and observed["decoded"] is False


@pytest.mark.parametrize("kind", ["empty", "oversize", "extra", "symlink"])
def test_unaccepted_payload_shapes(isolated, monkeypatch, kind):
    directory, target, _ = inert_payload(isolated, monkeypatch)
    if kind == "empty":
        target.write_bytes(b"")
        monkeypatch.setattr(image, "PUBLISHED_TOKEN", hashlib.md5(b"", usedforsecurity=False).hexdigest())
    elif kind == "oversize":
        with target.open("wb") as stream:
            stream.truncate(image.MAX_BYTES + 1)  # Sparse local fixture, no image allocation.
    elif kind == "extra":
        (directory / "extra").write_bytes(b"inert")
    else:
        target.unlink()
        target.symlink_to(isolated / "does-not-exist")
    with pytest.raises((image.Rejected, OSError)):
        image.verify_payload(directory)


def test_actual_stream_bytes_are_capped_despite_stale_stat(isolated, monkeypatch):
    directory, _, _ = inert_payload(isolated, monkeypatch, b"x" * 32)
    original = image.os.fstat
    monkeypatch.setattr(image, "MAX_BYTES", 24)
    def stale_size(fd):
        info = original(fd)
        return SimpleNamespace(st_mode=info.st_mode, st_size=16)
    monkeypatch.setattr(image.os, "fstat", stale_size)
    with pytest.raises(image.Rejected, match="compressed_payload_cap_exceeded"):
        image.verify_payload(directory)


def test_wrong_source_never_starts_client(isolated, monkeypatch):
    spec = spec_fixture(monkeypatch)
    spec["paths"].append({"source": "/other-image.nii.gz"})
    monkeypatch.setattr(image, "fresh_spec", lambda deadline: spec)
    receipt, _ = image.execute()
    assert receipt["status"] == "failed" and receipt["transfer_started"] is False
    assert receipt["failure"] == "image_spec_source_mismatch"


def test_failed_client_does_not_promote_complete_matching_bytes(isolated, monkeypatch):
    spec = spec_fixture(monkeypatch)
    monkeypatch.setattr(image, "fresh_spec", lambda deadline: spec)
    raw = b"inert completed-looking bytes"
    monkeypatch.setattr(image, "PUBLISHED_TOKEN", hashlib.md5(raw, usedforsecurity=False).hexdigest())
    calls = []
    class Client:
        def wait(self, **kwargs):
            return 7
        def poll(self):
            return 7
    def launch(argv, **kwargs):
        calls.append((argv, kwargs))
        Path(argv[-1]).write_bytes(raw)
        return Client()
    def unexpected_hash(*args):
        pytest.fail("Failed client must not promote bytes via payload hashing")
    monkeypatch.setattr(image.subprocess, "Popen", launch)
    monkeypatch.setattr(image, "verify_payload", unexpected_hash)
    receipt, path = image.execute()
    assert len(calls) == 1 and receipt["status"] == "failed" and "payload" not in receipt
    argv, options = calls[0]
    assert options["start_new_session"] is False and options["preexec_fn"] is image.child_limits
    assert options["env"]["ASPERA_SCP_TOKEN"] == spec["token"]
    assert spec["token"] not in " ".join(argv) and spec["cookie"] not in " ".join(argv)
    assert "NOT_A_REAL" not in path.read_text()


@pytest.mark.parametrize("state", ["complete", "surviving_writer", "stale_receipt_binding"])
def test_image_supervisor_uses_image_entrypoint_and_final_verifier(isolated, monkeypatch, capsys, state):
    directory, _, _ = inert_payload(isolated, monkeypatch)
    record = {"status": image.SUCCESS, "source": image.SOURCE,
        "proposal_sha256": image.PROPOSAL_SHA256, "checksum_index_sha256": image.INDEX_SHA256,
        "supervisor_acceptance_required": True, "payload": image.verify_payload(directory)}
    if state == "stale_receipt_binding":
        record["checksum_index_sha256"] = "0" * 64
    path = directory.parent / (directory.name + "-receipt.json")
    path.write_text(json.dumps(record))
    calls, killed = [], []
    class Worker:
        pid = 918999
        returncode = 0
        stdout = io.BytesIO()
        def communicate(self, timeout):
            assert 0 < timeout <= 57
            return json.dumps({"status": image.SUCCESS, "receipt": str(path.relative_to(isolated))}).encode(), None
        def wait(self, timeout):
            assert 0 < timeout <= 1
            return 0
    def launch(argv, **kwargs):
        calls.append((argv, kwargs))
        return Worker()
    def kill(pid, sig):
        killed.append((pid, sig))
        if state != "surviving_writer":
            raise ProcessLookupError
    monkeypatch.setattr(image.subprocess, "Popen", launch)
    monkeypatch.setattr(image.os, "killpg", kill)
    result = image.supervise(image.time.monotonic())
    summary = json.loads(capsys.readouterr().out)
    assert len(calls) == 1 and calls[0][0][1] == str(MODULE)
    assert calls[0][0][-2:] == ["--execute", "--_worker"]
    assert calls[0][1]["start_new_session"] is True
    assert killed == [(Worker.pid, signal.SIGKILL)]
    assert summary["images_requested"] is True
    assert summary["accepted"] is (state == "complete")
    assert (result == 0) is (state == "complete")
