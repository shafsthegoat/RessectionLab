"""Transport guard tests only: no patient data or live provider requests."""
import importlib.util
import hashlib
import json
from pathlib import Path
import subprocess
import sys

import pytest

MODULE = Path(__file__).resolve().parents[1] / "scripts/acquire_rhuh_checksum.py"
SPEC = importlib.util.spec_from_file_location("rhuh_checksum", MODULE)
acquire = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(acquire)
FIXTURE_RESOLVED_SOURCE = "/public-provider/package" + acquire.SOURCE


@pytest.fixture(autouse=True)
def pin_transport_fixture_path(monkeypatch):
    monkeypatch.setattr(acquire, "RESOLVED_SOURCE_SHA256", hashlib.sha256(FIXTURE_RESOLVED_SOURCE.encode()).hexdigest())


def transfer_spec():
    return {
        "paths": [{"source": FIXTURE_RESOLVED_SOURCE}], "direction": "receive",
        "authentication": "token", "remote_host": acquire.HOST,
        "remote_user": "public-client", "ssh_port": 33001, "fasp_port": 33001,
        "cipher": "aes-128", "token": "TEST_ONLY_TRANSFER_TOKEN",
        "cookie": "TEST_ONLY_COOKIE", "multi_session": 0,
    }


@pytest.mark.parametrize("mutation", [
    {"paths": [{"source": "/RHUH-GBM_nii_v1/RHUH-0001/0/RHUH-0001_0_adc.nii.gz"}]},
    {"paths": [{"source": acquire.SOURCE}, {"source": "/other"}]},
    {"paths": [{"source": acquire.SOURCE, "destination": "/other"}]},
    {"paths": [{"source": "/different-provider/package" + acquire.SOURCE}]},
    {"remote_host": "unreviewed.example"}, {"direction": "send"},
    {"authentication": "password"}, {"ssh_port": 22}, {"fasp_port": 22},
    {"cipher": "none"}, {"token": ""}, {"source_root": "/other"},
    {"multi_session": 4}, {"remote_user": "-argument-injection\n"},
])
def test_unreviewed_transfer_is_rejected(mutation):
    spec = transfer_spec()
    spec.update(mutation)
    with pytest.raises(acquire.Rejected):
        acquire.transfer_invocation(spec, Path("destination"))


def test_credentials_are_env_only_and_no_inherited_auth(monkeypatch):
    monkeypatch.setenv("ASPERA_SCP_PASS", "OTHER_SECRET")
    monkeypatch.setenv("SSH_AUTH_SOCK", "/agent/socket")
    spec = transfer_spec()
    argv, env = acquire.transfer_invocation(spec, Path("destination"))
    assert spec["token"] not in " ".join(argv)
    assert spec["cookie"] not in " ".join(argv)
    assert env["ASPERA_SCP_TOKEN"] == spec["token"]
    assert "ASPERA_SCP_PASS" not in env and "SSH_AUTH_SOCK" not in env
    assert argv[-2:] == [FIXTURE_RESOLVED_SOURCE, "destination"]
    assert "-L-" in argv and "--overwrite=never" in argv


def test_os_file_limit_actually_bounds_child_write(tmp_path):
    target = tmp_path / "oversize"
    result = subprocess.run(
        [sys.executable, "-c", "import pathlib,sys; pathlib.Path(sys.argv[1]).write_bytes(b'x'*65537)", str(target)],
        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
        preexec_fn=acquire.child_limits, timeout=5,
    )
    assert result.returncode != 0
    assert target.stat().st_size <= acquire.MAX_BYTES


@pytest.mark.parametrize("mode", ["too_small", "too_large", "extra_file", "symlink"])
def test_payload_mismatch_never_accepted(tmp_path, mode):
    target = tmp_path / acquire.SOURCE.lstrip("/")
    size = acquire.EXPECTED_BYTES
    if mode == "too_small":
        size -= 1
    if mode == "too_large":
        size += 1
    target.write_bytes(b"x" * size)
    if mode == "extra_file":
        (tmp_path / "unexpected").write_text("x")
    if mode == "symlink":
        target.rename(tmp_path / "original")
        target.symlink_to(tmp_path / "original")
    with pytest.raises(acquire.Rejected):
        acquire.verify_payload(tmp_path)


def test_fresh_auth_requests_only_checksum_and_does_not_save_tokens(monkeypatch):
    calls = []
    replies = [
        (200, {}, b'<a href="https://faspex.cancerimagingarchive.net/aspera/faspex/public?context=TEST">download</a>'),
        (200, {}, b"client_id: 'public-client', redirect_uri: '/aspera/faspex/token'"),
        (302, {"Location": "/aspera/faspex/token?code=TEST_CODE"}, b""),
        (201, {}, b'{"access_token":"TEST_ONLY_ACCESS_TOKEN"}'),
        (201, {}, json.dumps(transfer_spec()).encode()),
    ]
    def fake_request(url, deadline, **kwargs):
        calls.append((url, kwargs))
        return replies.pop(0)
    monkeypatch.setattr(acquire, "request", fake_request)
    result = acquire.fresh_spec(100)
    acquire.validate_spec(result)
    assert len(calls) == 5
    assert json.loads(calls[-1][1]["data"]) == {"paths": [{"path": acquire.SOURCE}]}
    assert all(".nii" not in url for url, _ in calls)


def test_failures_redact_exception_content_and_never_start_client(tmp_path, monkeypatch):
    monkeypatch.setattr(acquire, "ROOT", tmp_path)
    monkeypatch.setattr(acquire, "QUARANTINE_ROOT", tmp_path / "quarantine")
    def fail(deadline):
        raise ValueError("DO_NOT_PERSIST_THIS_TOKEN_OR_URL")
    monkeypatch.setattr(acquire, "fresh_spec", fail)
    receipt, path = acquire.execute()
    assert receipt["transfer_started"] is False
    assert receipt["failure"] == "redacted_exception_ValueError"
    assert "DO_NOT_PERSIST" not in path.read_text()


def test_wall_alarm_has_fixed_redacted_reason():
    with pytest.raises(acquire.Rejected, match="wall_time_limit"):
        acquire.wall_alarm(None, None)
