"""Inert transport/byte fixtures only: no image acquisition or decoding."""
import hashlib
import importlib.util
import json
from pathlib import Path
import subprocess
import sys

import pytest

MODULE = Path(__file__).resolve().parents[1] / "scripts/acquire_rhuh_single_image.py"
SPEC = importlib.util.spec_from_file_location("rhuh_single_image", MODULE)
acquire = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(acquire)
FIXTURE_RESOLVED = "/public-provider/package" + acquire.SOURCE
INERT = b"Transport test bytes, deliberately not a gzip or NIfTI image."


@pytest.fixture(autouse=True)
def no_live_metadata(monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError("Live provider calls are forbidden in this suite")
    monkeypatch.setattr(acquire, "request", forbidden)


def transfer_spec():
    return {
        "paths": [{"source": FIXTURE_RESOLVED}], "direction": "receive",
        "authentication": "token", "remote_host": acquire.public_transfer.HOST,
        "remote_user": "public-client", "ssh_port": 33001, "fasp_port": 33001,
        "cipher": "aes-128", "token": "INERT_TOKEN", "cookie": "INERT_COOKIE",
    }


@pytest.fixture
def payload(tmp_path, monkeypatch):
    monkeypatch.setattr(acquire, "ROOT", tmp_path)
    monkeypatch.setattr(acquire, "PUBLISHED_TOKEN", hashlib.md5(INERT).hexdigest())
    directory = tmp_path / "run-fixture"
    directory.mkdir(mode=0o700)
    (directory / Path(acquire.SOURCE).name).write_bytes(INERT)
    return directory


def test_fixed_real_source_proposal_and_index_bind_without_image_read():
    acquire.provenance_checks()
    assert acquire.EXPECTED_BYTES is None
    assert acquire.MAX_BYTES == 67_108_864
    assert acquire.SOURCE == "/RHUH-GBM_nii_v1/RHUH-0001/0/RHUH-0001_0_t1.nii.gz"
    assert acquire.PUBLISHED_TOKEN == "a5c579950d0431d04928e5b59e91ce9d"


@pytest.mark.parametrize("input_name", ["PROPOSAL", "INDEX"])
def test_mutated_provenance_rejected_before_authorization(tmp_path, monkeypatch, input_name):
    path = tmp_path / "mutated"
    path.write_bytes(b"mutated")
    monkeypatch.setattr(acquire, input_name, path)
    with pytest.raises(acquire.Rejected, match="hash_mismatch"):
        acquire.fresh_spec(100)


def test_provenance_symlink_rejected(tmp_path, monkeypatch):
    path = tmp_path / "proposal-link"
    path.symlink_to(acquire.PROPOSAL)
    monkeypatch.setattr(acquire, "PROPOSAL", path)
    with pytest.raises(acquire.Rejected, match="symlink"):
        acquire.provenance_checks()


@pytest.mark.parametrize("mutation", [
    {"paths": [{"source": FIXTURE_RESOLVED.replace("_t1.", "_adc.")}]},
    {"paths": [{"source": FIXTURE_RESOLVED}, {"source": "/other"}]},
    {"paths": [{"source": FIXTURE_RESOLVED, "destination": "/other"}]},
    {"paths": [{"source": "/unreviewed/package" + acquire.SOURCE}]},
    {"remote_host": "unreviewed.example"}, {"direction": "send"},
    {"authentication": "password"}, {"ssh_port": 22}, {"fasp_port": 22},
    {"cipher": "none"}, {"token": ""}, {"source_root": "/other"},
    {"multi_session": 4}, {"remote_user": "-argument\n"},
])
def test_unreviewed_spec_rejected_before_client(monkeypatch, mutation):
    monkeypatch.setattr(acquire, "RESOLVED_SOURCE_SHA256", hashlib.sha256(FIXTURE_RESOLVED.encode()).hexdigest())
    spec = transfer_spec()
    spec.update(mutation)
    with pytest.raises(acquire.Rejected):
        acquire.transfer_invocation(spec, Path("destination"))


def test_credentials_and_exact_destination(monkeypatch):
    monkeypatch.setattr(acquire, "RESOLVED_SOURCE_SHA256", hashlib.sha256(FIXTURE_RESOLVED.encode()).hexdigest())
    monkeypatch.setenv("ASPERA_SCP_PASS", "OTHER_SECRET")
    monkeypatch.setenv("SSH_AUTH_SOCK", "/agent/socket")
    argv, env = acquire.transfer_invocation(transfer_spec(), Path("destination"))
    assert argv[-2:] == [FIXTURE_RESOLVED, "destination"]
    assert "INERT_TOKEN" not in " ".join(argv) and "INERT_COOKIE" not in " ".join(argv)
    assert env["ASPERA_SCP_TOKEN"] == "INERT_TOKEN"
    assert not {"ASPERA_SCP_PASS", "SSH_AUTH_SOCK"} & env.keys()
    assert "-L-" in argv and "--overwrite=never" in argv


def test_unknown_length_compatible_original_bytes_remain_quarantined(payload):
    result = acquire.verify_payload(payload)
    assert result["measured_compressed_bytes"] == len(INERT)
    assert result["sha256"] == hashlib.sha256(INERT).hexdigest()
    assert result["candidate_matches_published_token"] is True
    assert result["expected_compressed_bytes"] is None
    assert result["prior_length_match_claim"] is False
    assert result["publisher_algorithm_confirmed"] is False
    assert result["decoded"] is False and result["scientific_use_released"] is False


def test_nonmatch_retains_only_unaccepted_measurements(payload, monkeypatch):
    monkeypatch.setattr(acquire, "PUBLISHED_TOKEN", "0" * 32)
    with pytest.raises(acquire.CompatibilityMismatch) as failure:
        acquire.verify_payload(payload)
    observed = failure.value.observation
    assert observed["measured_compressed_bytes"] == len(INERT)
    assert observed["candidate_matches_published_token"] is False
    assert observed["decoded"] is False


@pytest.mark.parametrize("mode", ["empty", "oversize", "extra", "symlink"])
def test_unaccepted_payload_shapes(payload, mode):
    path = payload / Path(acquire.SOURCE).name
    if mode == "empty":
        path.write_bytes(b"")
    elif mode == "oversize":
        with path.open("wb") as stream:
            stream.truncate(acquire.MAX_BYTES + 1)  # Sparse, no image bytes.
    elif mode == "extra":
        (payload / "unexpected").write_bytes(b"x")
    elif mode == "symlink":
        path.unlink()
        path.symlink_to(payload.parent / "outside")
    with pytest.raises((acquire.Rejected, OSError)):
        acquire.verify_payload(payload)


def test_actual_64mib_os_file_cap_rejects_sparse_out_of_bounds_write(tmp_path):
    target = tmp_path / "inert-sparse-limit-control"
    result = subprocess.run([
        sys.executable, "-c",
        "import sys; f=open(sys.argv[1],'wb'); f.seek(67108864); f.write(b'x'); f.flush()",
        str(target),
    ], preexec_fn=acquire.child_limits, stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL, timeout=5)
    assert result.returncode != 0
    assert target.stat().st_size <= acquire.MAX_BYTES


def test_mock_public_authorization_requests_only_one_fixed_image(monkeypatch):
    calls = []
    replies = [
        (200, {}, b'<a href="https://faspex.cancerimagingarchive.net/aspera/faspex/public?context=TEST">download</a>'),
        (200, {}, b"client_id: 'public-client', redirect_uri: '/aspera/faspex/token'"),
        (302, {"Location": "/aspera/faspex/token?code=INERT_CODE"}, b""),
        (201, {}, b'{"access_token":"INERT_ACCESS_TOKEN"}'),
        (201, {}, json.dumps(transfer_spec()).encode()),
    ]
    def fake_request(url, deadline, **kwargs):
        calls.append((url, kwargs))
        return replies.pop(0)
    monkeypatch.setattr(acquire, "request", fake_request)
    acquire.fresh_spec(100)
    assert len(calls) == 5
    assert json.loads(calls[-1][1]["data"]) == {"paths": [{"path": acquire.SOURCE}]}


def test_failures_redact_credentials_and_do_not_start_client(tmp_path, monkeypatch):
    monkeypatch.setattr(acquire, "ROOT", tmp_path)
    monkeypatch.setattr(acquire, "QUARANTINE_ROOT", tmp_path / "quarantine")
    def fail(*args, **kwargs):
        raise ValueError("DO_NOT_PERSIST_TOKEN_OR_SIGNED_URL")
    monkeypatch.setattr(acquire, "fresh_spec", fail)
    monkeypatch.setattr(acquire.subprocess, "Popen", lambda *args, **kwargs: pytest.fail("client forbidden"))
    receipt, path = acquire.execute()
    assert receipt["status"] == "failed" and receipt["transfer_started"] is False
    assert receipt["failure"] == "redacted_exception_ValueError"
    assert "DO_NOT_PERSIST" not in path.read_text()
    directory = path.parent / path.name.removesuffix("-receipt.json")
    assert directory.stat().st_mode & 0o777 == 0o700


def test_final_reconciliation_binds_payload_and_provenance(payload):
    receipt = {
        "source": acquire.SOURCE, "status": acquire.SUCCESS,
        "proposal_sha256": acquire.PROPOSAL_SHA256,
        "checksum_index_sha256": acquire.INDEX_SHA256,
        "supervisor_acceptance_required": True,
        "payload": acquire.verify_payload(payload),
    }
    path = payload.parent / (payload.name + "-receipt.json")
    path.write_text(json.dumps(receipt))
    assert acquire.reconcile_worker_receipt(path) == receipt["payload"]
    receipt["checksum_index_sha256"] = "wrong"
    path.write_text(json.dumps(receipt))
    with pytest.raises(acquire.Rejected, match="reconciliation"):
        acquire.reconcile_worker_receipt(path)


def test_supervisor_receives_new_image_entrypoint(monkeypatch):
    observed = {}
    def fake_supervisor(started, group, **kwargs):
        observed.update(kwargs)
        return 31
    monkeypatch.setattr(acquire.public_transfer, "supervise_worker", fake_supervisor)
    assert acquire._supervise(10, []) == 31
    assert Path(observed["worker_argv"][1]) == MODULE
    assert observed["worker_argv"][-2:] == ["--execute", "--_worker"]
    assert observed["reconcile"] is acquire.reconcile_worker_receipt
    assert observed["images_requested"] is True


def test_default_cli_is_local_preflight_only(monkeypatch, capsys):
    calls = []
    monkeypatch.setattr(sys, "argv", [str(MODULE)])
    monkeypatch.setattr(acquire, "local_checks", lambda: calls.append("local"))
    monkeypatch.setattr(acquire, "supervise", lambda *args: pytest.fail("transfer forbidden"))
    assert acquire.main() == 0 and calls == ["local"]
    assert json.loads(capsys.readouterr().out)["status"] == "preflight_passed_no_network_no_transfer_no_decode"
