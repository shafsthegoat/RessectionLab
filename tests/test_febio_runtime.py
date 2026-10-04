"""Tiny real-process watchdog and archive controls; no FEBio execution/download."""
from __future__ import annotations

import importlib.util
import io
import json
import os
from pathlib import Path
import subprocess
import sys
import tarfile
import time
import zipfile

import pytest

SCRIPT = Path(__file__).resolve().parents[1] / "scripts/febio_runtime.py"
spec = importlib.util.spec_from_file_location("febio_runtime", SCRIPT)
runtime = importlib.util.module_from_spec(spec)
spec.loader.exec_module(runtime)


def stopped(pid):
    result = subprocess.run(["/bin/ps", "-o", "stat=", "-p", str(pid)],
                            capture_output=True, text=True, timeout=1)
    return result.returncode != 0 or not result.stdout.strip() or result.stdout.strip().startswith("Z")


def run(tmp_path, code, *, seconds=2, rss=256*1024**2):
    output = tmp_path / "supervised"
    receipt = runtime.supervise([sys.executable, "-c", code], output, cwd=tmp_path,
        environment=dict(os.environ), seconds=seconds, rss_bytes=rss)
    assert json.loads((output / "supervision.json").read_text()) == receipt
    return receipt


def test_actual_clean_process_records_success(tmp_path):
    receipt = run(tmp_path, "print('finite control')")
    assert receipt["status"] == "completed"
    assert receipt["exit_code"] == 0
    assert receipt["kill_reason"] is None


def test_actual_deadline_stops_process_group_and_reaps_leader(tmp_path):
    code = ("import subprocess,sys,time,pathlib; "
            "p=subprocess.Popen([sys.executable,'-c','import time; time.sleep(30)']); "
            "pathlib.Path('grandchild.pid').write_text(str(p.pid)); time.sleep(30)")
    receipt = run(tmp_path, code, seconds=.45)
    assert receipt["status"] == "failed_or_incomplete"
    assert receipt["kill_reason"] == "wall_cap"
    assert receipt["exit_code"] == -9
    assert stopped(receipt["pid"])
    grandchild = int((tmp_path / "grandchild.pid").read_text())
    assert stopped(grandchild)


def test_actual_low_memory_cap_stops_group(tmp_path):
    receipt = run(tmp_path, "import time; time.sleep(30)", rss=1)
    assert receipt["kill_reason"] == "process_group_rss_cap"
    assert receipt["sampled_peak_process_group_rss_bytes"] > 1
    assert receipt["exit_code"] == -9


def test_actual_orphan_group_is_rejected_and_stopped(tmp_path):
    code = ("import subprocess,sys,pathlib; "
            "p=subprocess.Popen([sys.executable,'-c','import time; time.sleep(30)']); "
            "pathlib.Path('grandchild.pid').write_text(str(p.pid))")
    receipt = run(tmp_path, code)
    assert receipt["kill_reason"] == "descendants_outlived_parent"
    assert receipt["status"] == "failed_or_incomplete"
    assert stopped(int((tmp_path / "grandchild.pid").read_text()))


def test_observer_failure_stops_actual_child_and_preserves_error(tmp_path, monkeypatch):
    def fail(*args, **kwargs):
        raise subprocess.TimeoutExpired("ps", kwargs["timeout_seconds"])
    monkeypatch.setattr(runtime, "process_group_rss", fail)
    receipt = run(tmp_path, "import time; time.sleep(30)")
    assert receipt["kill_reason"] == "supervision_exception"
    assert receipt["error"]["type"] == "TimeoutExpired"
    assert receipt["exit_code"] == -9
    assert stopped(receipt["pid"])


def test_existing_output_is_never_overwritten(tmp_path):
    (tmp_path / "supervised").mkdir()
    with pytest.raises(FileExistsError):
        run(tmp_path, "print('must not launch')")


def tar_bytes(items):
    output = io.BytesIO()
    with tarfile.open(fileobj=output, mode="w") as archive:
        for name, value, link in items:
            info = tarfile.TarInfo(name)
            if link:
                info.type, info.linkname = tarfile.SYMTYPE, value
                archive.addfile(info)
            else:
                body = value.encode()
                info.size = len(body)
                archive.addfile(info, io.BytesIO(body))
    output.seek(0)
    return output


@pytest.mark.parametrize("name", ["../escape", "/absolute", "safe/../../escape", "a\\b"])
def test_source_tar_traversal_rejected(tmp_path, name):
    with pytest.raises(ValueError, match="Unsafe"):
        runtime.extract_tar(tar_bytes([(name, "data", False)]), tmp_path / "out", expanded_limit=100)


def test_tar_link_chain_cannot_escape(tmp_path):
    with pytest.raises(ValueError, match="link chain"):
        runtime.extract_tar(tar_bytes([("a", ".", True), ("b", "a/../escape", True)]),
                            tmp_path / "out", expanded_limit=100)


def test_tar_size_and_exact_commit_root_are_enforced(tmp_path):
    with pytest.raises(ValueError, match="expansion cap"):
        runtime.extract_tar(tar_bytes([("commit/a", "1234", False)]), tmp_path / "cap",
                            expanded_limit=3, strip_root="commit")
    with pytest.raises(ValueError, match="exact commit"):
        runtime.extract_tar(tar_bytes([("wrong/a", "x", False)]), tmp_path / "root",
                            expanded_limit=3, strip_root="commit")


def test_wheel_traversal_rejected(tmp_path):
    path = tmp_path / "bad.whl"
    with zipfile.ZipFile(path, "w") as archive:
        archive.writestr("../escape", "data")
    with pytest.raises(ValueError, match="Unsafe"):
        runtime.extract_zip(path, tmp_path / "out", expanded_limit=100)
    assert not (tmp_path / "escape").exists()


def test_download_hash_mismatch_is_retained_and_rejected(tmp_path, monkeypatch):
    class Response(io.BytesIO):
        url = "https://example.invalid/exact"
    monkeypatch.setattr(runtime.urllib.request, "urlopen", lambda *args, **kwargs: Response(b"bad"))
    with pytest.raises(ValueError, match="SHA256 mismatch"):
        runtime.download("https://example.invalid/exact", tmp_path / "payload", limit=3,
                         expected_sha="0"*64, expected_bytes=3)
    assert (tmp_path / "payload").read_bytes() == b"bad"


def test_child_environment_preserves_home_and_removes_ambient_build_paths(monkeypatch):
    monkeypatch.setenv("HOME", "/preserved/home")
    monkeypatch.setenv("CODEX_HOME", "/preserved/codex")
    monkeypatch.setenv("CPATH", "/unrelated/include")
    monkeypatch.setenv("DYLD_INSERT_LIBRARIES", "/unrelated/injected.dylib")
    result = runtime.private_environment({"caps": {"thread_environment": {"OMP_NUM_THREADS": "1"}}})
    assert result["HOME"] == os.environ["HOME"] == "/preserved/home"
    assert result["CODEX_HOME"] == os.environ["CODEX_HOME"] == "/preserved/codex"
    assert "CPATH" not in result and os.environ["CPATH"] == "/unrelated/include"
    assert not any(key.startswith("DYLD_") for key in result)
    assert result["OMP_NUM_THREADS"] == "1"


def test_committed_declaration_pin():
    assert runtime.declaration()["source"]["commit"] == "32ae206ff4881dfb54f62296cd1558e58ed9fcc6"


def test_parent_rejected_acquisition_is_not_reusable(tmp_path):
    for name in ("result", "supervision"):
        (tmp_path / (name+".json")).write_text(json.dumps({"status": "completed"}))
    (tmp_path / "acceptance.json").write_text(json.dumps({"status": "failed_or_incomplete"}))
    with pytest.raises(ValueError, match="parent acceptance"):
        runtime.verify_acquisition(tmp_path)


def test_build_cannot_start_from_cache_without_accepted_configure(tmp_path, monkeypatch):
    monkeypatch.setattr(runtime, "verify_acquisition", lambda path: None)
    monkeypatch.setattr(runtime, "PREFIX", tmp_path)
    (tmp_path / "build").mkdir()
    (tmp_path / "build/CMakeCache.txt").write_text("plausible matching cache")
    with pytest.raises(ValueError, match="Accepted configure receipt required"):
        runtime.build_stage("build", {}, tmp_path, tmp_path, None)
    assert not (tmp_path / "state").exists()


def test_accepted_stage_with_incomplete_artifact_inventory_is_rejected(tmp_path):
    row = {"status": "completed", "stage": "configure", "driver_sha256": runtime.sha(SCRIPT),
           "declaration_sha256": runtime.DECLARATION_SHA}
    for name in ("result", "supervision"):
        (tmp_path / (name+".json")).write_text(json.dumps(row))
    (tmp_path / "acceptance.json").write_text(json.dumps({**row, "artifact_sha256": {}}))
    with pytest.raises(ValueError, match="inventory incomplete"):
        runtime.verify_stage_receipt(tmp_path, "configure")
