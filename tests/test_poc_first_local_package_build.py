"""Generated worker/filesystem controls; no builds, image arrays, or models."""
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys

import psutil
import pytest

spec = importlib.util.spec_from_file_location("package_supervisor", Path(__file__).resolve().parents[1] /
                                             "scripts/poc_first_local_package_build.py")
runner = importlib.util.module_from_spec(spec)
spec.loader.exec_module(runner)


def limits(**overrides):
    return {"rss_bytes": 2 * 1024**3, "available_bytes": 0, "free_disk_bytes": 0,
            "new_bytes": 1024**2, "wall_seconds": 2, "interval_seconds": .02, **overrides}


def watch(tmp_path, script, caps=None, observer=runner.sample_processes):
    scope = tmp_path / "scope"
    scope.mkdir(exist_ok=True)
    return runner.supervise([sys.executable, "-c", script], tmp_path, {}, tmp_path / "worker.log",
                            [scope], {}, {}, caps or limits(), observer=observer)


def test_generated_worker_success_and_scalar_resource_observations(tmp_path):
    result = watch(tmp_path, "print('generated worker')")
    assert result["status"] == "completed_unverified" and result["exit_code"] == 0
    assert result["worker_reaped"] and result["peak_family_plus_supervisor_rss_bytes"] > 0
    assert result["minimum_available_memory_bytes"] > 0 and result["minimum_free_disk_bytes"] > 0
    assert result["sampled_stop_enforcement"] and not result["os_hard_caps"]


def test_real_sampled_rss_stop_reaps_generated_worker(tmp_path):
    result = watch(tmp_path, "import time; data=bytearray(8*1024**2); time.sleep(2)", limits(rss_bytes=1))
    assert result["status"] == "rss_stop" and result["worker_reaped"]
    assert result["peak_family_plus_supervisor_rss_bytes"] > 1


def test_timeout_kills_nested_child_but_not_unrelated_created_process(tmp_path):
    unrelated = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(5)"])
    child_path = tmp_path / "child.pid"
    script = ("import subprocess,sys,time; from pathlib import Path; "
              "p=subprocess.Popen([sys.executable,'-c','import time; time.sleep(5)']); "
              f"Path({str(child_path)!r}).write_text(str(p.pid)); time.sleep(5)")
    try:
        result = watch(tmp_path, script, limits(wall_seconds=.25))
        assert result["status"] == "timeout" and result["worker_reaped"]
        assert unrelated.poll() is None
        try:
            child = psutil.Process(int(child_path.read_text()))
            assert not child.is_running() or child.status() == psutil.STATUS_ZOMBIE
        except psutil.NoSuchProcess:
            pass
    finally:
        unrelated.kill()
        unrelated.wait()


def test_group_cleanup_permission_error_retains_original_stop_and_reaps(tmp_path, monkeypatch):
    def denied(*_): raise PermissionError("generated group failure")
    monkeypatch.setattr(runner.os, "killpg", denied)
    result = watch(tmp_path, "import time; time.sleep(2)", limits(wall_seconds=.15))
    assert result["status"] == "cleanup_failed" and result["original_status"] == "timeout"
    assert result["worker_reaped"] and "PermissionError" in result["cleanup_errors"]


def test_observer_failure_still_cleans_worker(tmp_path):
    def broken(*_): raise RuntimeError("generated observation error")
    result = watch(tmp_path, "import time; time.sleep(2)", observer=broken)
    assert result["status"] == "supervisor_failed" and result["worker_reaped"]


def test_inventory_counts_links_without_following_external_directory(tmp_path):
    scope, external = tmp_path / "scope", tmp_path / "external"
    scope.mkdir(); external.mkdir()
    (external / "large").write_bytes(b"x" * 1000)
    (scope / "original").write_bytes(b"old")
    (scope / "link").symlink_to(external, target_is_directory=True)
    values = runner.disk_inventory([scope])
    assert values == {str(scope / "original"): 3, str(scope / "link"): (scope / "link").lstat().st_size}
    with pytest.raises(ValueError, match="OVERLAPPING"):
        runner.disk_inventory([scope, scope / "nested"])


def test_first_stage_baseline_and_growth_survive_deletion_and_next_stage(tmp_path):
    baseline = {"old": 100, "original": 10}
    growth = {}
    assert runner.accumulate_growth(baseline, growth, {"original": 15, "sidecar": 25}) == 30
    # Persisted first-stage accounting survives removal of both old and new files.
    ledger = tmp_path / "first-stage.json"
    runner.publish(ledger, growth)
    restored = json.loads(ledger.read_text())
    assert runner.accumulate_growth(baseline, restored, {"electron": 40}) == 70
    assert runner.accumulate_growth(baseline, restored, {"sidecar": 20}) == 70
    with pytest.raises(FileExistsError): runner.publish(ledger, {})


def test_disk_growth_stop_observes_generated_output_without_old_file_credit(tmp_path):
    scope = tmp_path / "scope"
    scope.mkdir()
    old = scope / "old"
    old.write_bytes(b"x" * 100)
    baseline = runner.disk_inventory([scope])
    script = f"from pathlib import Path; Path({str(old)!r}).unlink(); Path({str(scope / 'new')!r}).write_bytes(b'12345678')"
    result = runner.supervise([sys.executable, "-c", script], tmp_path, {}, tmp_path / "worker.log", [scope],
                              baseline, {}, limits(new_bytes=5))
    assert result["status"] in {"cumulative_bytes_stop", "terminal_disk_stop"}
    assert result["cumulative_new_bytes"] == 8 and result["worker_reaped"]


def test_low_disk_reserve_refuses_launch(tmp_path, monkeypatch):
    class Usage: free = 1
    monkeypatch.setattr(runner.shutil, "disk_usage", lambda *_: Usage())
    monkeypatch.setattr(runner.subprocess, "Popen", lambda *_a, **_k: pytest.fail("worker started below reserve"))
    result = watch(tmp_path, "pass", limits(free_disk_bytes=2))
    assert result["status"] == "prelaunch_resource_stop" and result["samples"] == 0


def test_clean_environment_excludes_credentials_and_only_adds_named_archive_key(monkeypatch):
    monkeypatch.setenv("EXAMPLE_SECRET_TOKEN", "must not enter build")
    monkeypatch.setattr(runner.shutil, "which", lambda *_a, **_k: "/usr/bin/tool")
    stage = {"stage": "electron", "environment": {"PATH": "/usr/bin", "TMPDIR": "/generated/tmp"}}
    document = {"offline_build": {"existing_verified_archive": "/generated/cache/electron.zip"}}
    assert runner.effective_environment(stage, document) == {
        "PATH": "/usr/bin", "TMPDIR": "/generated/tmp", "RESECTIONLAB_ELECTRON_ZIP_DIR": "/generated/cache"}


def stage_fixture(tmp_path, monkeypatch, issuer="ROOT"):
    stage = {"stage": "sidecar", "program": sys.executable}
    source = {"commit": "generated-committed-source", "files": {}}
    monkeypatch.setattr(runner, "ROOT", tmp_path)
    monkeypatch.setattr(runner, "plan", lambda: {"build_sequence": [stage]})
    monkeypatch.setattr(runner, "source_manifest", lambda: source)
    monkeypatch.setattr(runner, "supervise", lambda *_a, **_k: pytest.fail("refused release started worker"))
    release = {"schema": runner.SCHEMA + "-root-release", "issuer": issuer, "decision": "release_one_stage",
               "stage": "sidecar", "plan_sha256": runner.PLAN_SHA, "runner_sha256": runner.sha(runner.__file__),
               "session_directory": str(tmp_path / "outputs/session"), "receipt_path": str(tmp_path / "artifacts/receipt.json"),
               "source_commit": "wrong-generated-source", "source_manifest_sha256": runner.value_sha(source),
               "program_sha256": runner.sha(sys.executable)}
    path = tmp_path / "release.json"
    path.write_text(json.dumps(release))
    return path, release


def test_run_stage_invalid_binding_seals_scalar_failure_without_worker(tmp_path, monkeypatch):
    path, release = stage_fixture(tmp_path, monkeypatch)
    assert runner.run_stage("sidecar", path) == 1
    artifact = runner.read(release["receipt_path"])
    private = runner.read(Path(release["session_directory"]) / "sidecar/receipt.json")
    assert artifact == private
    assert artifact["status"] == "stage_failed"
    assert artifact["error_code"] == "SOURCE_OR_PROGRAM_BINDING_CHANGED"
    assert artifact["error_type"] == "ValueError" and artifact["attempts"] == 1
    assert not artifact["environment_values_logged"] and not artifact["build_verified"]
    assert not (Path(release["session_directory"]) / "sidecar/worker.log").exists()


@pytest.mark.parametrize("issuer", [None, "OTHER"])
def test_release_issuer_must_be_root_before_stage_directory_or_worker(tmp_path, monkeypatch, issuer):
    path, release = stage_fixture(tmp_path, monkeypatch, issuer)
    with pytest.raises(ValueError, match="EXACT_ROOT_STAGE_RELEASE"):
        runner.run_stage("sidecar", path)
    assert not Path(release["session_directory"]).exists()
    assert not Path(release["receipt_path"]).exists()


def generated_predecessor(tmp_path, monkeypatch, subdirectory="sidecar", **overrides):
    monkeypatch.setattr(runner, "ROOT", tmp_path)
    session = tmp_path / runner.RECOVERY_SESSION
    output = session / subdirectory
    output.mkdir(parents=True)
    baseline = {"roots": [str(tmp_path / name) for name in runner.SCOPES], "files": {}}
    ledger = {str(tmp_path / "build/electron-sidecar/generated.bin"): 100}
    runner.publish(session / "baseline.json", baseline)
    runner.publish(output / "growth-ledger.json", ledger)
    receipt = {"schema": runner.SCHEMA + "-receipt", "stage": subdirectory, "plan_sha256": runner.PLAN_SHA,
               "attempts": 1, "baseline_sha256": runner.sha(session / "baseline.json"),
               "growth_ledger_sha256": runner.sha(output / "growth-ledger.json"),
               "cumulative_new_bytes": 100, "limits": {"new_bytes": 8 * 1024**3},
               "status": "worker_failed", "exit_code": 1, "worker_reaped": True,
               "source_manifest_sha256": "generated-source", **overrides}
    runner.publish(output / "receipt.json", receipt)
    binding = {"receipt_sha256": runner.sha(output / "receipt.json"),
               "growth_ledger_sha256": receipt["growth_ledger_sha256"], "baseline_sha256": receipt["baseline_sha256"]}
    monkeypatch.setattr(runner, "RECOVERY_PREDECESSOR", binding)
    return session, binding


def test_recovery_preserves_first_baseline_and_charges_archive_new_paths(tmp_path, monkeypatch):
    session, binding = generated_predecessor(tmp_path, monkeypatch)
    before = runner.sha(session / "baseline.json")
    prior, baseline, carried = runner.recovery_state(session, {"predecessor": binding})
    with pytest.raises(ValueError, match="EXACT_FIRST_FAILURE"):
        runner.recovery_state(session.with_name("new-session"), {"predecessor": binding})
    with pytest.raises(ValueError, match="CURRENT_SOURCE"):
        runner.require_successful_sidecar(prior, "generated-source")
    archive = str(tmp_path / "build/poc-first-local-package-v1/failed-first-sidecar/electron-sidecar/generated.bin")
    assert sum(carried.values()) == 100
    assert runner.accumulate_growth(baseline["files"], carried, {archive: 100}) == 200
    assert runner.sha(session / "baseline.json") == before
    assert runner.read(session / "sidecar/growth-ledger.json") != carried


@pytest.mark.parametrize("key", ["receipt_sha256", "growth_ledger_sha256", "baseline_sha256"])
def test_recovery_requires_all_exact_original_failure_pins(tmp_path, monkeypatch, key):
    session, binding = generated_predecessor(tmp_path, monkeypatch)
    altered = {**binding, key: "0" * 64}
    with pytest.raises(ValueError, match="EXACT_FIRST_FAILURE"):
        runner.recovery_state(session, {"predecessor": altered})


@pytest.mark.parametrize("overrides", [{"status": "completed_unverified", "exit_code": 0},
                                      {"worker_reaped": False}, {"attempts": 2}])
def test_recovery_rejects_success_unreaped_or_repeated_predecessor(tmp_path, monkeypatch, overrides):
    session, binding = generated_predecessor(tmp_path, monkeypatch, **overrides)
    with pytest.raises(ValueError):
        runner.recovery_state(session, {"predecessor": binding})


def test_changed_original_baseline_is_rejected_without_reset(tmp_path, monkeypatch):
    session, binding = generated_predecessor(tmp_path, monkeypatch)
    (session / "baseline.json").write_text('{"reset":true}')
    with pytest.raises(ValueError, match="ACCOUNTING_BINDING"):
        runner.recovery_state(session, {"predecessor": binding})


@pytest.mark.parametrize("subdirectory", ["../sidecar", "sidecar-recovery-02", "electron"])
def test_arbitrary_sidecar_subdirectory_is_refused(tmp_path, subdirectory):
    with pytest.raises(ValueError, match="SUBDIRECTORY_NOT_ALLOWED"):
        runner.predecessor_state(tmp_path, subdirectory, {})


def test_recovery_directory_cannot_be_silently_reused(tmp_path, monkeypatch):
    session, binding = generated_predecessor(tmp_path, monkeypatch)
    (session / runner.RECOVERY_STAGE).mkdir()
    monkeypatch.setattr(runner, "plan", lambda: {"build_sequence": [{"stage": "sidecar"}]})
    monkeypatch.setattr(runner, "supervise", lambda *_a, **_k: pytest.fail("reused recovery launched"))
    release = {"schema": runner.SCHEMA + "-root-release", "issuer": "ROOT", "decision": "release_one_stage",
               "stage": runner.RECOVERY_STAGE, "plan_sha256": runner.PLAN_SHA, "runner_sha256": runner.sha(runner.__file__),
               "session_directory": str(session), "receipt_path": str(tmp_path / "artifacts/recovery.json"),
               "predecessor": binding}
    path = tmp_path / "recovery-release.json"
    path.write_text(json.dumps(release))
    with pytest.raises(FileExistsError):
        runner.run_stage(runner.RECOVERY_STAGE, path)
    assert not Path(release["receipt_path"]).exists()


def test_failed_outputs_must_be_preserved_and_old_targets_absent(tmp_path, monkeypatch):
    monkeypatch.setattr(runner, "ROOT", tmp_path)
    archive = tmp_path / "build/poc-first-local-package-v1/failed-first-sidecar"
    with pytest.raises(ValueError, match="PRESERVED_FAILED_OUTPUTS"):
        runner.preserved_outputs_ready()
    (archive / "electron-sidecar").mkdir(parents=True)
    (archive / "ressectionlab-engine").mkdir()
    runner.preserved_outputs_ready()
    (tmp_path / "build/electron-sidecar").mkdir()
    with pytest.raises(ValueError, match="PRESERVED_FAILED_OUTPUTS"):
        runner.preserved_outputs_ready()


@pytest.mark.parametrize("subdirectory", ["sidecar", "sidecar-recovery-01"])
def test_electron_selects_explicit_successful_sidecar_with_current_source(tmp_path, monkeypatch, subdirectory):
    session, binding = generated_predecessor(tmp_path, monkeypatch, subdirectory,
                                             status="completed_unverified", exit_code=0)
    prior, _, ledger = runner.predecessor_state(session, subdirectory, binding)
    runner.require_successful_sidecar(prior, "generated-source")
    assert sum(ledger.values()) == 100
    with pytest.raises(ValueError, match="CURRENT_SOURCE"):
        runner.require_successful_sidecar(prior, "changed-source")
