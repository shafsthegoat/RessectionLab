"""Source metadata and process controls; no generated patient examples."""
import csv
import gzip
import importlib
import io
import json
from pathlib import Path
import time
import sys
from urllib.error import HTTPError
from email.message import Message

import pytest

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def intake(monkeypatch):
    monkeypatch.syspath_prepend(str(ROOT / "scripts"))
    return importlib.import_module("acquire_resect_cavity")


def test_pilot_fixity_matches_actual_source_inventory(intake):
    toc = gzip.decompress((ROOT / "artifacts/mechanics/resect-case4-acquisition-metadata-v1/table-of-contents.csv.gz").read_bytes())
    rows = [{k.strip(): v.strip() if v else v for k, v in row.items()}
            for row in csv.DictReader(io.StringIO(toc.decode()), delimiter="|")]
    original = next(row for row in rows if row.get("filename") == "RESECT/NIFTI/Case3/US/Case3-US-during.nii.gz")
    assert int(original["size"]) == intake.SOURCES[1]["bytes"]
    assert original["fixity"] == intake.SOURCES[1]["expected_md5"]
    declaration = intake.require_manifest()
    assert declaration["patient_group"] == "RESECT:Case3" and declaration["role"] == "TRAIN"
    assert declaration["cohort_sha256"] == intake.sha(intake.COHORT)


def test_missing_rights_refuses_acquisition_before_network(intake, tmp_path, monkeypatch):
    attempt = tmp_path / "attempt"
    attempt.mkdir()
    (attempt / "source.json").write_bytes(intake.encode(intake.code_binding()))
    monkeypatch.setattr(intake, "DATA", tmp_path / "unacquired")
    def forbidden(*args, **kwargs):
        pytest.fail("Rights failure must precede scientific acquisition")
    monkeypatch.setattr(intake, "transfer", forbidden)
    with pytest.raises(FileNotFoundError):
        intake.worker("acquire", attempt)
    record = json.loads((attempt / "worker-result.json").read_bytes())
    assert record["status"] == "failed" and not record["files"] and not record["events"]
    assert record["optimizer_updates"] == record["recorded_rl_transitions"] == 0


def test_source_change_refuses_worker(intake, tmp_path):
    source = intake.code_binding()
    source["files"][intake.SOURCE_CODE[0]] = "0" * 64
    (tmp_path / "source.json").write_bytes(intake.encode(source))
    with pytest.raises(ValueError, match="Execution source changed"):
        intake.worker("rights", tmp_path)
    assert not (tmp_path / "worker-result.json").exists()


@pytest.mark.parametrize("url", ["http://osf.io/download/4mkfg/", "https://example.org/file",
                                     "https://osf.io:8443/download/4mkfg/", "file:///tmp/input"])
def test_unreviewed_destination_rejected_before_request(intake, url):
    with pytest.raises(ValueError, match="Unreviewed source"):
        intake.validate_url(url, intake.SOURCES[0])


@pytest.mark.parametrize("source_index", [0, 2])
def test_osf_storage_redirect_requires_exact_frozen_object(intake, source_index):
    source = intake.SOURCES[source_index]
    route = f"https://storage.googleapis.com/cos-osf-prod-files-de-1/{source['sha256']}"
    intake.validate_url(route + "?X-Goog-Signature=transport-control", source)
    for bad in (route.replace("https:", "http:"), route + "/extra",
                route.replace("cos-osf-prod-files-de-1", "unrelated-bucket"),
                route.replace(source['sha256'], "0" * 64)):
        with pytest.raises(ValueError, match="Unreviewed source"):
            intake.validate_url(bad, source)
    saved = intake.safe_route(route + "?X-Goog-Signature=transport-control")
    assert "transport-control" not in json.dumps(saved)
    assert set(saved) == {"host", "path", "full_url_sha256"}


def test_malformed_signed_query_cannot_reach_network_or_error_text(intake, tmp_path, monkeypatch):
    source = dict(intake.SOURCES[0])
    monkeypatch.setattr(intake, "DATA", tmp_path / "unacquired")
    def forbidden(*args, **kwargs):
        pytest.fail("Malformed URL must be refused before network")
    class NoNetwork:
        open = staticmethod(forbidden)
    monkeypatch.setattr(intake, "build_opener", lambda *args: NoNetwork())
    for character in (" ", "\n", "\x00", "\x7f", "é"):
        source["source_url"] = ("https://storage.googleapis.com/cos-osf-prod-files-de-1/"
                                + source["sha256"] + "?X-Goog-Signature=private" + character + "value")
        with pytest.raises(ValueError) as error:
            intake.transfer(source, tmp_path, deadline=time.monotonic() + 2, events=[])
        assert "private" not in str(error.value)
        assert "Signature" not in str(error.value)
    assert not list(tmp_path.glob("request-*.json"))


def test_rate_limit_is_not_retried_and_retains_retry_after(intake, tmp_path, monkeypatch):
    """HTTP status/header control only; no acquired or fabricated medical bytes."""
    calls = []
    headers = Message()
    headers["Retry-After"] = "120"
    class Limited:
        def open(self, request, **kwargs):
            calls.append(request.full_url)
            raise HTTPError(request.full_url, 429, "Too Many Requests", headers, None)
    monkeypatch.setattr(intake, "DATA", tmp_path / "source")
    monkeypatch.setattr(intake, "build_opener", lambda *args: Limited())
    events = []
    with pytest.raises(RuntimeError, match="HTTP 429; no automatic retry"):
        intake.transfer(intake.SOURCES[0], tmp_path, deadline=time.monotonic() + 2, events=events)
    assert calls == [intake.SOURCES[0]["source_url"]]
    assert events[0]["status"] == 429 and events[0]["retry_after"] == "120"
    assert json.loads((tmp_path / "response-01.json").read_bytes())["retry_after"] == "120"
    assert not list(tmp_path.glob("*.partial"))


def test_qc_refuses_unbound_import_before_any_image(intake, monkeypatch, tmp_path):
    import resectionlab.imaging as imaging
    monkeypatch.setattr(imaging, "__file__", str(tmp_path / "foreign-imaging.py"))
    def forbidden(*args, **kwargs):
        pytest.fail("Foreign QC implementation must be rejected before image access")
    monkeypatch.setattr(imaging, "inspect_nifti", forbidden)
    with pytest.raises(ValueError, match="reviewed checkout"):
        intake.structural_qc()


def test_setup_failure_has_supervision_receipt(intake, monkeypatch, tmp_path):
    monkeypatch.setattr(intake, "DATA", tmp_path)
    monkeypatch.setattr(intake, "ROOT", tmp_path)
    monkeypatch.setattr(sys, "argv", ["acquire_resect_cavity.py", "rights"])
    def unavailable():
        raise OSError("source snapshot unavailable")
    monkeypatch.setattr(intake, "code_binding", unavailable)
    with pytest.raises(OSError, match="snapshot unavailable"):
        intake.main()
    receipts = list(tmp_path.glob("attempts/*/supervision.json"))
    assert len(receipts) == 1
    saved = json.loads(receipts[0].read_bytes())
    assert saved["status"] == "setup_or_supervision_failed" and saved["error_type"] == "OSError"


def test_nird_client_is_exact_source_only(intake, monkeypatch, tmp_path):
    monkeypatch.setattr(intake, 'DATA', tmp_path / 'data')
    monkeypatch.setattr(intake, 'system_curl_binding', lambda **kwargs: pytest.fail('No process for changed source'))
    for field, value in (('source_url', intake.SOURCES[1]['source_url'] + '?changed=1'),
                         ('bytes', 1), ('expected_md5', '0' * 32),
                         ('path', 'originals/Case4.nii.gz')):
        source = {**intake.SOURCES[1], field: value}
        with pytest.raises(ValueError, match='exact frozen NIRD'):
            intake.transfer(source, tmp_path, deadline=time.monotonic() + 2, events=[])
    assert not list(tmp_path.iterdir())


def test_nird_requires_remaining_deadline_and_native_trust(intake, monkeypatch, tmp_path):
    monkeypatch.setattr(intake, 'DATA', tmp_path / 'data')
    monkeypatch.setattr(intake, 'system_curl_binding', lambda **kwargs: {'available': False})
    with pytest.raises(TimeoutError):
        intake.transfer(intake.SOURCES[1], tmp_path, deadline=time.monotonic() - 1, events=[])
    with pytest.raises(ValueError, match='SecureTransport is unavailable'):
        intake.transfer(intake.SOURCES[1], tmp_path, deadline=time.monotonic() + 2, events=[])
    assert not list(tmp_path.iterdir())


def test_nird_environment_has_no_ambient_ca_or_backend_overrides(intake, monkeypatch):
    overrides = ('CURL_CA_BUNDLE', 'SSL_CERT_FILE', 'SSL_CERT_DIR', 'CURL_SSL_BACKEND')
    for name in overrides:
        monkeypatch.setenv(name, 'format-control')
    environment = intake.system_curl_environment()
    assert not set(overrides) & set(environment)
    assert environment['PATH'] == intake.os.environ['PATH']


class CurlProcessControl:
    """Transport/process control: header bytes only; never an image fixture."""
    def __init__(self, *, code=0, interruption=None):
        self.code = code
        self.interruption = interruption
        self.returncode = None
        self.killed = False
        self.waits = 0

    def wait(self, timeout=None):
        self.waits += 1
        if self.waits == 1 and self.interruption is not None:
            raise self.interruption
        self.returncode = -9 if self.killed else self.code
        return self.returncode

    def poll(self):
        return self.returncode

    def kill(self):
        self.killed = True
        self.returncode = -9


def install_curl_control(intake, monkeypatch, tmp_path, *, code=0, headers=None, interruption=None):
    monkeypatch.setattr(intake, 'DATA', tmp_path / 'data')
    monkeypatch.setattr(intake, 'system_curl_binding', lambda **kwargs: {
        'available': True, 'executable': '/usr/bin/curl', 'trust': 'existing_macos_system_store'})
    if headers is None:
        headers = b'HTTP/2 200\r\nContent-Length: 9156131\r\n\r\n'
    process = CurlProcessControl(code=code, interruption=interruption)
    captured = []
    def launch(command, **kwargs):
        captured.append((command, kwargs))
        intake.os.write(kwargs['pass_fds'][0], headers)
        return process
    monkeypatch.setattr(intake.subprocess, 'Popen', launch)
    return process, captured


def test_nird_bounded_command_and_published_fixity_gate(intake, monkeypatch, tmp_path):
    process, captured = install_curl_control(intake, monkeypatch, tmp_path)
    events = []
    # An empty body passes no scientific integrity gate. This tests the actual
    # fixed size verifier after simulated header/process success, not a patient.
    with pytest.raises(ValueError, match='Source size/type mismatch'):
        intake.transfer(intake.SOURCES[1], tmp_path, deadline=time.monotonic() + 2, events=events)
    assert len(captured) == 1
    command, kwargs = captured[0]
    assert command[:3] == ['/usr/bin/curl', '-q', '--no-location']
    assert command[-1] == intake.SOURCES[1]['source_url']
    for option, value in (('--max-redirs', '0'), ('--retry', '0'), ('--proto', '=https'),
                          ('--max-filesize', '9156131'), ('--header', 'Accept-Encoding: identity')):
        assert command[command.index(option) + 1] == value
    assert 0 < float(command[command.index('--max-time') + 1]) <= 2
    assert not {'-k', '--insecure', '--cacert', '--capath', '--location'} & set(command)
    assert kwargs['stderr'] == intake.subprocess.DEVNULL
    assert not kwargs.get('start_new_session', False)
    assert process.waits == 2 and not process.killed
    assert events[0]['status'] == 200
    assert not (intake.DATA / intake.SOURCES[1]['path']).exists()
    assert (tmp_path / 'Case3-US-during.nii.gz.partial').stat().st_size == 0


@pytest.mark.parametrize('headers', [
    b'HTTP/2 302\r\nLocation: https://unreviewed.example/?secret=private\r\nContent-Length: 9156131\r\n\r\n',
    b'HTTP/2 200\r\nContent-Length: 1\r\n\r\n',
    b'HTTP/2 200\r\nContent-Length: 9156131\r\nContent-Length: 9156131\r\n\r\n',
    b'HTTP/2 200\r\nContent-Length: 9156131\r\nContent-Encoding: gzip\r\n\r\n',
])
def test_nird_response_contract_refuses_without_redirect_or_body_admission(intake, monkeypatch, tmp_path, headers):
    process, captured = install_curl_control(intake, monkeypatch, tmp_path, headers=headers)
    with pytest.raises(ValueError, match='response status, encoding or declared byte length differs') as error:
        intake.transfer(intake.SOURCES[1], tmp_path, deadline=time.monotonic() + 2, events=[])
    assert len(captured) == 1 and 'private' not in str(error.value)
    assert 'private' not in (tmp_path / 'response-01.json').read_text()
    assert not (intake.DATA / intake.SOURCES[1]['path']).exists()


def test_nird_nonzero_client_exit_is_sanitized_not_retried(intake, monkeypatch, tmp_path):
    process, captured = install_curl_control(intake, monkeypatch, tmp_path, code=60)
    with pytest.raises(RuntimeError, match='exit 60.*no automatic retry') as error:
        intake.transfer(intake.SOURCES[1], tmp_path, deadline=time.monotonic() + 2, events=[])
    assert len(captured) == 1 and 'https://' not in str(error.value)
    assert json.loads((tmp_path / 'response-01.json').read_text())['curl_exit_code'] == 60


@pytest.mark.parametrize('interrupt', ['timeout', 'interrupt'])
def test_nird_child_is_killed_and_reaped_on_timeout_or_interruption(intake, monkeypatch, tmp_path, interrupt):
    error = intake.subprocess.TimeoutExpired('/usr/bin/curl', 2) if interrupt == 'timeout' else KeyboardInterrupt('process control')
    process, captured = install_curl_control(intake, monkeypatch, tmp_path, interruption=error)
    expected = TimeoutError if interrupt == 'timeout' else KeyboardInterrupt
    with pytest.raises(expected):
        intake.transfer(intake.SOURCES[1], tmp_path, deadline=time.monotonic() + 2, events=[])
    assert process.killed and process.waits == 2
    assert (tmp_path / 'response-01.json').exists()
    assert not (intake.DATA / intake.SOURCES[1]['path']).exists()


def test_nird_partial_is_exclusive_before_process_start(intake, monkeypatch, tmp_path):
    process, captured = install_curl_control(intake, monkeypatch, tmp_path)
    partial = tmp_path / 'Case3-US-during.nii.gz.partial'
    partial.write_bytes(b'preexisting partial format control')
    with pytest.raises(FileExistsError):
        intake.transfer(intake.SOURCES[1], tmp_path, deadline=time.monotonic() + 2, events=[])
    assert not captured and partial.read_bytes() == b'preexisting partial format control'


def test_nird_headers_accept_proxy_connect_but_require_final_response(intake, tmp_path):
    path = tmp_path / 'headers'
    path.write_bytes(b'HTTP/1.1 200 Connection established\r\n\r\nHTTP/2 200\r\nContent-Length: 9156131\r\n\r\n')
    status, headers = intake.nird_response_headers(path)
    assert status == 200 and headers['Content-Length'] == '9156131'
    path.write_bytes(b'x' * 65537)
    with pytest.raises(ValueError, match='headers exceed byte limit'):
        intake.nird_response_headers(path)


def test_nird_transport_runtime_is_bound_in_source_receipt(intake):
    binding = intake.code_binding()['nird_original_transport']
    assert binding['executable'] == '/usr/bin/curl'
    assert binding['trust'] == 'existing_macos_system_store'
    if sys.platform == 'darwin':
        assert binding['available'] and '(SecureTransport)' in binding['version']
        assert binding['sha256'] == intake.sha(Path('/usr/bin/curl'))


def test_osf_mask_still_uses_original_python_transport(intake, monkeypatch, tmp_path):
    calls = []
    class Limited:
        def open(self, request, **kwargs):
            calls.append(request.full_url)
            raise HTTPError(request.full_url, 429, 'limited', Message(), None)
    monkeypatch.setattr(intake, 'DATA', tmp_path / 'data')
    monkeypatch.setattr(intake, 'build_opener', lambda *args: Limited())
    monkeypatch.setattr(intake, 'transfer_nird_original', lambda *args, **kwargs: pytest.fail('OSF must not use NIRD client'))
    with pytest.raises(RuntimeError, match='HTTP 429'):
        intake.transfer(intake.SOURCES[2], tmp_path, deadline=time.monotonic() + 2, events=[])
    assert calls == [intake.SOURCES[2]['source_url']]
