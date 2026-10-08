"""Actual file/process controls, without generated patient fixtures."""
import importlib
import hashlib
import json
import os
from pathlib import Path
import sys
import signal
import subprocess
import time

import pytest

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def intake(monkeypatch):
    monkeypatch.syspath_prepend(str(ROOT / "scripts"))
    return importlib.import_module("real_intake_io")


def test_atomic_publication_preserves_prior_metadata(intake, tmp_path):
    payload = (ROOT / "manifests/lausanne-original-pilot-v1.json").read_bytes()
    path = tmp_path / "metadata.json"
    intake.atomic_preserve(path, payload)
    intake.atomic_preserve(path, payload)
    with pytest.raises(ValueError, match="Preserve existing"):
        intake.atomic_preserve(path, payload + b"\n")
    assert path.read_bytes() == payload
    assert not list(tmp_path.glob("*.partial"))


def test_publication_failure_leaves_no_truncated_destination(intake, tmp_path, monkeypatch):
    def interrupted(*args, **kwargs):
        raise OSError("publication interrupted")
    monkeypatch.setattr(intake.os, "link", interrupted)
    path = tmp_path / "metadata.json"
    with pytest.raises(OSError, match="publication interrupted"):
        intake.atomic_preserve(path, b"{}\n")
    assert not path.exists()
    assert not list(tmp_path.iterdir())


def test_source_verification_checks_published_hash_and_deadline(intake):
    import hashlib
    path = ROOT / "manifests/lausanne-original-pilot-v1.json"
    data = path.read_bytes()
    entry = {"bytes": len(data), "sha256": hashlib.sha256(data).hexdigest()}
    assert intake.verify_source_file(path, entry) == entry["sha256"]
    with pytest.raises(ValueError, match="Published source SHA256"):
        intake.verify_source_file(path, {**entry, "sha256": "unmatched"})
    with pytest.raises(intake.IntakeDeadline):
        intake.verify_source_file(path, entry, deadline=time.monotonic() - 1)


def test_hard_worker_timeout_reaps_actual_process(intake, tmp_path):
    pids = []
    status, code = intake.supervise([sys.executable, "-c", "import time; time.sleep(30)"],
                                    tmp_path / "worker.log", deadline=time.monotonic() + .2, on_start=pids.append)
    assert status == "timeout" and code is None and len(pids) == 1
    with pytest.raises(ProcessLookupError):
        os.kill(pids[0], 0)


def test_parent_callback_failure_reaps_worker(intake, tmp_path):
    pids = []
    def interrupt(pid):
        pids.append(pid)
        raise RuntimeError("parent interrupted")
    with pytest.raises(RuntimeError, match="parent interrupted"):
        intake.supervise([sys.executable, "-c", "import time; time.sleep(30)"],
                         tmp_path / "worker.log", deadline=time.monotonic() + 2, on_start=interrupt)
    with pytest.raises(ProcessLookupError):
        os.kill(pids[0], 0)


def test_full_actual_index_preserves_all_train_visits(intake, monkeypatch):
    module = importlib.import_module("lausanne_train_intake")
    if not (module.DATA / "source-metadata/participants.tsv").exists():
        pytest.skip("Pinned source metadata is required")
    index = module.validated_index()
    assert len(index["sessions"]) == 210
    assert sum(len(row["files"]) for row in index["sessions"]) == 840
    assert len({row["subject"] for row in index["sessions"]}) == 199
    assert sum(entry["bytes"] for row in index["sessions"] for entry in row["files"]) == 10020802851
    assert all(row["role"] == "TRAIN" for row in index["sessions"])


def test_sigterm_routes_through_actual_supervisor_cleanup(intake, tmp_path):
    script = """
import sys, time
from pathlib import Path
from real_intake_io import supervise, termination_cleanup
folder = Path(sys.argv[1])
try:
    with termination_cleanup():
        supervise([sys.executable, '-c', 'import time; time.sleep(30)'],
                  folder / 'worker.log', deadline=time.monotonic() + 20,
                  on_start=lambda pid: (folder / 'pid').write_text(str(pid)))
finally:
    (folder / 'closed').write_text('cleanup completed')
"""
    environment = dict(os.environ, PYTHONPATH=str(ROOT / "scripts"))
    supervisor = subprocess.Popen([sys.executable, "-c", script, str(tmp_path)], env=environment,
                                  stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    try:
        deadline = time.monotonic() + 5
        while not (tmp_path / "pid").exists():
            assert time.monotonic() < deadline and supervisor.poll() is None
            time.sleep(.01)
        # The small write can be observed before close; wait for complete digits.
        while not (pid_text := (tmp_path / "pid").read_text()):
            assert time.monotonic() < deadline
            time.sleep(.01)
        pid = int(pid_text)
        supervisor.send_signal(signal.SIGTERM)
        assert supervisor.wait(timeout=5) != 0
        assert (tmp_path / "closed").read_text() == "cleanup completed"
        with pytest.raises(ProcessLookupError):
            os.kill(pid, 0)
    finally:
        if supervisor.poll() is None:
            supervisor.kill()
            supervisor.wait()


def test_cached_source_requires_actual_retained_snapshot(intake, tmp_path, monkeypatch):
    module = importlib.import_module("lausanne_train_intake")
    source = module.execution_source()
    cache = tmp_path / "intake"
    run = cache / "attempts" / "control"
    digest = module.retain_execution_source(run, source, deadline=time.monotonic() + 5)
    monkeypatch.setattr(module, "ROOT", tmp_path)
    monkeypatch.setattr(module, "CACHE", cache)
    receipt = {"execution_source_sha256": digest,
               "execution_source_record": str((run / "source.json").relative_to(tmp_path))}
    # No current source files exist beneath this root. Historical snapshots suffice.
    module.validate_retained_source(receipt)
    with pytest.raises(module.AcquisitionError, match="digest/schema"):
        module.validate_retained_source({**receipt, "execution_source_sha256": "0" * 64})
    (run / "source-snapshot" / module.SOURCE_NAMES[0]).unlink()
    with pytest.raises(FileNotFoundError):
        module.validate_retained_source(receipt)


def test_existing_only_missing_original_does_not_call_downloader(intake, tmp_path, monkeypatch):
    module = importlib.import_module("lausanne_train_intake")
    if not (module.DATA / "source-metadata/participants.tsv").exists():
        pytest.skip("Pinned source metadata is required")
    index = module.validated_index()
    index_sha = hashlib.sha256(module.INDEX.read_bytes()).hexdigest()
    source = tmp_path / "source.json"
    source.write_bytes(module.json_bytes(module.execution_source()))
    monkeypatch.setattr(module, "DATA", tmp_path)
    monkeypatch.setattr(module, "validated_index", lambda: index)
    def forbidden(*args, **kwargs):
        pytest.fail("Existing-files-only worker attempted acquisition")
    monkeypatch.setattr(module, "acquire_file", forbidden)
    with pytest.raises(FileNotFoundError):
        module.acquire_session("sub-000/ses-20110101", index_sha, tmp_path / "output.json", source,
                               existing_only=True)
    assert not (tmp_path / "output.json").exists()
    assert not (tmp_path / "sub-000").exists()


def test_setup_failure_still_writes_batch_closure(intake, tmp_path, monkeypatch):
    module = importlib.import_module("lausanne_train_intake")
    monkeypatch.setattr(module, "CACHE", tmp_path / "intake")
    monkeypatch.setattr(module, "ROOT", tmp_path)
    def unavailable():
        raise OSError("source is unavailable")
    monkeypatch.setattr(module, "execution_source", unavailable)
    with pytest.raises(OSError, match="source is unavailable"):
        module.batch(5, 1, "0" * 64)
    paths = list((module.CACHE / "attempts").glob("*/batch.json"))
    assert len(paths) == 1
    saved = json.loads(paths[0].read_bytes())
    assert saved["outcomes"] == [] and saved["expected_inventory_resolved"] is False
    assert "source is unavailable" in saved["problem"]


def test_failed_closure_probe_retains_original_problem(intake, tmp_path, monkeypatch):
    module = importlib.import_module("lausanne_train_intake")
    source = module.execution_source()
    monkeypatch.setattr(module, "CACHE", tmp_path / "intake")
    monkeypatch.setattr(module, "ROOT", tmp_path)
    monkeypatch.setattr(module, "INDEX", tmp_path / "absent-index.json")
    monkeypatch.setattr(module, "train_sessions", lambda: [])
    calls = []
    def source_probe():
        calls.append(1)
        if len(calls) > 1:
            raise OSError("closure source probe failed")
        return source
    monkeypatch.setattr(module, "execution_source", source_probe)
    monkeypatch.setattr(module, "retain_execution_source", lambda *args, **kwargs: "0" * 64)
    with pytest.raises(FileNotFoundError):
        module.batch(5, 1, "0" * 64)
    saved = json.loads(next((module.CACHE / "attempts").glob("*/batch.json")).read_bytes())
    assert "absent-index.json" in saved["problem"]
    assert {error["check"] for error in saved["closure_check_errors"]} == {"execution_source", "index"}


@pytest.mark.parametrize("field,value", [("registration", "verified"), ("anatomical_coverage", "complete"),
                                         ("spatial_planning_admitted", True)])
def test_replayed_real_receipt_rejects_promoted_claims(intake, field, value):
    module = importlib.import_module("lausanne_train_intake")
    path = ROOT / "artifacts/lausanne-train-intake-v1/replay-acquisition.json"
    if not path.exists():
        pytest.skip("Requires completed real existing-pilot replay receipt")
    saved = json.loads(path.read_bytes())
    index = json.loads(module.INDEX.read_bytes())
    row = next(row for row in index["sessions"] if row["subject"] == saved["subject"]
               and row["session"] == saved["session"])
    with pytest.raises(module.AcquisitionError, match="identity/status/claim"):
        module.validate_receipt(row, {**saved, field: value}, saved["index_sha256"])
