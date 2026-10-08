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
