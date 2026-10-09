"""Synthetic supervisor controls only; these never launch FEBio."""
from __future__ import annotations

import json
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import pytest

from scripts import mechanics_nonpatient_n5_nonuniform_supervisor as runner


def test_preparation_is_closed_and_caps_match_parent():
    prep = json.loads((runner.ROOT / runner.PREPARATION).read_text())
    runner.validate_preparation(prep)
    assert prep["release"] is None
    assert prep["caps"] == runner.CAPS
    assert runner.CAPS["wall_seconds"] == 600
    assert runner.CAPS["sampled_process_group_rss_bytes"] == 3 * 1024**3
    assert runner.CAPS["active_output_bytes"] == 512 * 1024**2
    assert prep["prior_receipt"] == {"path": runner.PRIOR, "sha256": runner.PRIOR_SHA256}
    assert prep["prior_native_wall_seconds"] == 0.5189758341293782
    assert runner.AGGREGATE_CAPS == {"maximum_native_calls": 7,
                                     "aggregate_native_wall_seconds": 1800,
                                     "prior_calls": 1, "case_index": 1}
    for changed in ({"release": {}}, {"case_id": "n9_nonuniform"},
                    {"caps": {**runner.CAPS, "wall_seconds": 601}},
                    {"aggregate_caps": {**runner.AGGREGATE_CAPS, "prior_calls": 0}},
                    {"prior_native_wall_seconds": 0.}):
        with pytest.raises(ValueError):
            runner.validate_preparation({**prep, **changed})


def test_release_is_required_before_any_output(tmp_path):
    with pytest.raises(ValueError, match="release"):
        runner.execute(tmp_path / "absent.json", root=tmp_path)
    assert not (tmp_path / runner.OUTPUT).exists()


def test_malformed_release_cannot_create_output(tmp_path):
    release = tmp_path / "release.json"
    release.write_text('{"schema":"not-a-release"}')
    with pytest.raises(ValueError, match="release"):
        runner.execute(release, root=tmp_path)
    assert not (tmp_path / runner.OUTPUT).exists()


def test_explicit_cli_execute_cannot_launch_without_release(tmp_path):
    output = runner.ROOT / runner.OUTPUT
    existed = output.exists()
    completed = subprocess.run([sys.executable, "-B", "-m", "scripts.mechanics_nonpatient_n5_nonuniform_supervisor",
                                "--execute", "--release", str(tmp_path / "missing.json")],
                               cwd=runner.ROOT, capture_output=True, text=True, timeout=10)
    assert completed.returncode != 0
    assert output.exists() is existed


def test_module_cli_without_release_fails_before_output_creation(tmp_path):
    output = runner.ROOT / runner.OUTPUT
    existed = output.exists()
    completed = subprocess.run([sys.executable, "-B", "-m",
                                "scripts.mechanics_nonpatient_n5_nonuniform_supervisor",
                                "--execute"], cwd=runner.ROOT, capture_output=True,
                               text=True, timeout=10)
    assert completed.returncode != 0 and "--release" in completed.stderr
    assert output.exists() is existed


def test_frozen_prior_receipt_and_budget_drift_fail_closed(tmp_path, monkeypatch):
    copied = tmp_path / runner.PRIOR
    copied.parent.mkdir(parents=True)
    record = {"schema": "nonpatient-n5-one-call-receipt-v1", "case_id": "n5_affine",
              "status": "passed_numerical_software_only", "native_calls_attempted": 1,
              "no_retry": True, "exit_code": 0, "kill_reason": None,
              "source_commit": runner.PRIOR_SOURCE_COMMIT,
              "release_sha256": runner.PRIOR_RELEASE_SHA256,
              "deck_sha256": runner.PRIOR_DECK_SHA256,
              "saved_output_checks": {"passed": True}, "affine_oracle": {"passed": True},
              "elapsed_seconds": 0.5189758341293782}
    original = json.dumps(record).encode()
    copied.write_bytes(original)
    digest = runner.sha_bytes(original)
    monkeypatch.setattr(runner, "PRIOR_SHA256", digest)
    binding = {"path": runner.PRIOR, "sha256": digest}
    prior = runner.validate_prior_receipt(binding, root=tmp_path)
    assert prior["elapsed_seconds"] == 0.5189758341293782
    assert prior["calls_consumed"] == 1
    assert prior["elapsed_seconds"] + runner.CAPS["wall_seconds"] < 1800
    copied.unlink()
    with pytest.raises(FileNotFoundError):
        runner.validate_prior_receipt(binding, root=tmp_path)
    copied.write_bytes(original)
    for key, value in (("status", "failed_or_incomplete"),
                       ("native_calls_attempted", 2),
                       ("elapsed_seconds", 1800.)):
        damaged = dict(record)
        damaged[key] = value
        copied.write_text(json.dumps(damaged))
        with pytest.raises(ValueError, match="changed"):
            runner.validate_prior_receipt(binding, root=tmp_path)
    copied.write_bytes(original)
    monkeypatch.setattr(runner, "AGGREGATE_CAPS", {**runner.AGGREGATE_CAPS,
                                                   "aggregate_native_wall_seconds": 600})
    with pytest.raises(ValueError, match="aggregate"):
        runner.validate_prior_receipt(binding, root=tmp_path)


def test_path_and_binding_reject_escape_symlink_tamper(tmp_path):
    with pytest.raises(ValueError):
        runner.local_path("../outside", root=tmp_path)
    with pytest.raises(ValueError):
        runner.local_path("/outside", root=tmp_path)
    path = tmp_path / "input.txt"
    path.write_bytes(b"one")
    binding = {"path": "input.txt", "sha256": runner.sha_bytes(b"one")}
    assert runner.binding_bytes(binding, "input.txt", root=tmp_path) == b"one"
    path.write_bytes(b"two")
    with pytest.raises(ValueError, match="changed"):
        runner.binding_bytes(binding, "input.txt", root=tmp_path)
    path.unlink()
    path.symlink_to(tmp_path / "elsewhere")
    with pytest.raises((ValueError, FileNotFoundError)):
        runner.binding_bytes(binding, "input.txt", root=tmp_path)


def test_output_ancestor_symlink_is_rejected(tmp_path):
    (tmp_path / "real").mkdir()
    (tmp_path / "link").symlink_to(tmp_path / "real", target_is_directory=True)
    with pytest.raises(ValueError, match="Symlink"):
        runner.local_path("link/attempt-01", root=tmp_path)


def test_release_must_be_regular_stable_and_unsymlinked(tmp_path):
    release = tmp_path / "release.json"
    release.write_bytes(b"{}")
    raw, identity = runner.read_release(release)
    assert raw == b"{}" and len(identity) == 2
    link = tmp_path / "link.json"
    link.symlink_to(release)
    with pytest.raises(ValueError, match="symlink"):
        runner.read_release(link)
    with pytest.raises(ValueError, match="regular"):
        runner.read_release(tmp_path)
    with pytest.raises(ValueError, match="traversal"):
        runner.read_release(tmp_path / "child" / ".." / "release.json")


def test_git_provenance_scrubs_ambient_config_and_object_redirects(tmp_path, monkeypatch):
    commit = "a" * 40
    relative = runner.SOURCE_PATHS[0]
    payload = b"pinned module"
    path = tmp_path / relative
    path.parent.mkdir(parents=True)
    path.write_bytes(payload)
    monkeypatch.setenv("GIT_DIR", "/wrong")
    monkeypatch.setenv("GIT_OBJECT_DIRECTORY", "/wrong/objects")
    monkeypatch.setenv("GIT_CONFIG_COUNT", "1")
    def fake_run(command, **kwargs):
        environment = kwargs["env"]
        assert "GIT_DIR" not in environment
        assert "GIT_OBJECT_DIRECTORY" not in environment
        assert "GIT_CONFIG_COUNT" not in environment
        assert environment["GIT_CONFIG_NOSYSTEM"] == "1"
        if command[1:3] == ["cat-file", "-s"]:
            return SimpleNamespace(stdout=str(len(payload)).encode())
        if command[1:3] == ["cat-file", "blob"]:
            return SimpleNamespace(stdout=payload)
        return SimpleNamespace(stdout=(commit + "\n").encode())
    monkeypatch.setattr(runner.subprocess, "run", fake_run)
    assert runner.git_blob(commit, relative, root=tmp_path) == payload


def test_full_source_closure_and_commit_are_required(tmp_path):
    bindings = {}
    for relative in runner.SOURCE_PATHS:
        path = tmp_path / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(relative.encode())
        bindings[relative] = {"path": relative, "sha256": runner.sha_bytes(relative.encode())}
    release = {"source_commit": "a" * 40, "source_bindings": bindings}
    assert runner.source_hashes(release, root=tmp_path, require_git=False) == {
        key: value["sha256"] for key, value in bindings.items()}
    with pytest.raises(ValueError, match="closure"):
        runner.source_hashes({**release, "source_bindings": {}}, root=tmp_path, require_git=False)
    first = tmp_path / runner.SOURCE_PATHS[0]
    first.write_bytes(b"altered")
    with pytest.raises(ValueError, match="changed"):
        runner.source_hashes(release, root=tmp_path, require_git=False)


def test_source_commit_drift_rejected_before_native_call(tmp_path, monkeypatch):
    release = {"source_commit": "a" * 40,
               "source_bindings": {path: {"path": path, "sha256": "b" * 64}
                                   for path in runner.SOURCE_PATHS}}
    monkeypatch.setattr(runner.subprocess, "run",
                        lambda *_args, **_kwargs: SimpleNamespace(stdout=("c" * 40 + "\n").encode()))
    with pytest.raises(ValueError, match="HEAD differs"):
        runner.source_hashes(release, root=tmp_path)


def test_release_freezes_second_case_prior_and_runtime(monkeypatch):
    from scripts import mechanics_nonpatient_sparse_deck as deck
    from scripts import mechanics_hbe_backend as backend

    root = runner.ROOT
    parent = json.loads((root / runner.PARENT).read_text())
    def binding(relative):
        data = (root / relative).read_bytes()
        return {"path": relative, "sha256": runner.sha_bytes(data)}
    xml, _ = deck.build_deck(runner.CASE)
    release = {"schema": "nonpatient-n5-nonuniform-one-call-release-v1",
               "status": "root_released_one_native_call", "case_id": runner.CASE,
               "source_commit": "a" * 40, "source_bindings": {},
               "preparation": binding(runner.PREPARATION),
               "parent_declaration": binding(runner.PARENT),
               "backend_profile": parent["runtime"]["backend_profile"],
               "runtime_identity": parent["runtime"]["runtime_identity"],
               "prior_receipt": {"path": runner.PRIOR, "sha256": runner.PRIOR_SHA256},
               "deck_sha256": runner.sha_bytes(xml.encode()),
               "output_directory": runner.OUTPUT}
    original_binding_bytes = runner.binding_bytes
    prior_document = {"parent_declaration_sha256": release["parent_declaration"]["sha256"],
                      "runtime_identity_sha256": release["runtime_identity"]["sha256"],
                      "backend_profile_sha256": release["backend_profile"]["sha256"],
                      "backend_selection": [
                          "* Selecting linear solver accelerate                                    *"]}
    def fixture_binding_bytes(value, relative, **kwargs):
        if relative == runner.PRIOR:
            return json.dumps(prior_document).encode()
        return original_binding_bytes(value, relative, **kwargs)
    monkeypatch.setattr(runner, "binding_bytes", fixture_binding_bytes)
    monkeypatch.setattr(runner, "validate_prior_receipt",
                        lambda *_args, **_kwargs: {"sha256": runner.PRIOR_SHA256,
                                                   "elapsed_seconds": 0.5189758341293782,
                                                   "calls_consumed": 1})
    monkeypatch.setattr(runner, "source_hashes", lambda *_args, **_kwargs: {})
    monkeypatch.setattr(runner, "audit_loaded_modules", lambda **_kwargs: None)
    fake_runtime = {"profile_id": "accelerate_csc_v1",
                    "runtime_identity": release["runtime_identity"],
                    "runtime": {"executable_sha256": "e" * 64,
                                "libraries": {"install/bin/febio4": "e" * 64}}}
    monkeypatch.setattr(backend, "verify_profile", lambda *_args, **_kwargs: fake_runtime)
    assert runner.validate_release(release)["prior"]["calls_consumed"] == 1
    fake_runtime["profile_id"] = "skyline"
    with pytest.raises(ValueError, match="backend"):
        runner.validate_release(release)
    fake_runtime["profile_id"] = "accelerate_csc_v1"
    with pytest.raises(ValueError, match="runtime/profile"):
        runner.validate_release({**release, "runtime_identity": {"path": "other", "sha256": "0" * 64}})
    with pytest.raises(ValueError, match="prior"):
        runner.validate_release({**release, "prior_receipt": {"path": runner.PRIOR,
                                                                "sha256": "0" * 64}})
    with pytest.raises(ValueError, match="deck"):
        runner.validate_release({**release, "deck_sha256": "0" * 64})


def test_active_output_counts_and_rejects_symlinks(tmp_path):
    (tmp_path / "a").write_bytes(b"abc")
    (tmp_path / "b").write_bytes(b"d")
    assert runner.active_bytes(tmp_path) == 4
    (tmp_path / "link").symlink_to(tmp_path / "a")
    with pytest.raises(ValueError, match="symlink"):
        runner.active_bytes(tmp_path)


def test_active_output_scan_has_entry_bound(tmp_path, monkeypatch):
    for index in range(3):
        (tmp_path / str(index)).write_bytes(b"x")
    monkeypatch.setattr(runner, "MAX_ACTIVE_ENTRIES", 2)
    with pytest.raises(ValueError, match="work bound"):
        runner.active_bytes(tmp_path)


class FakeProcess:
    pid = 424242

    def __init__(self, code=None):
        self.code = code
        self.waited = False

    def poll(self):
        return self.code

    def wait(self, timeout):
        self.waited = True
        return self.code if self.code is not None else -9


def test_observed_rss_cap_kills_and_retains_failure_receipt(tmp_path, monkeypatch):
    process = FakeProcess()
    calls = []
    def popen(command, **kwargs):
        calls.append((command, kwargs))
        return process
    killed = []
    monkeypatch.setattr(runner.os, "killpg", lambda pid, sig: killed.append((pid, sig)))
    receipt = runner.supervise("/private/febio4", tmp_path, {}, popen=popen,
        rss_observer=lambda _pid, **_kwargs: (runner.CAPS["sampled_process_group_rss_bytes"] + 1, [{"pid": 424242}]),
        sleep=lambda _: None)
    assert len(calls) == 1
    assert calls[0][0] == ["/private/febio4", "-noconfig", "-no_title", "-i", "n5_nonuniform.feb", "-o", "n5_nonuniform.log"]
    assert calls[0][1]["start_new_session"] is True
    assert calls[0][1]["env"]["OMP_NUM_THREADS"] == "1"
    assert killed == [(424242, runner.signal.SIGKILL)] and process.waited
    assert receipt["kill_reason"] == "process_group_rss_cap"
    assert receipt["status"] == "failed_or_incomplete"
    assert json.loads((tmp_path / "receipt.json").read_text())["status"] == "failed_or_incomplete"


def test_observer_failure_kills_and_reaps(tmp_path, monkeypatch):
    process = FakeProcess()
    monkeypatch.setattr(runner.os, "killpg", lambda *_: None)
    def bad_observer(*_args, **_kwargs):
        raise RuntimeError("ps failed")
    receipt = runner.supervise("/private/febio4", tmp_path, {},
        popen=lambda *_args, **_kwargs: process, rss_observer=bad_observer, sleep=lambda _: None)
    assert process.waited and receipt["kill_reason"] == "supervision_exception"
    assert receipt["supervision_error"]["message"] == "ps failed"


def test_observed_active_output_cap_kills_and_reaps(tmp_path, monkeypatch):
    process = FakeProcess()
    killed = []
    monkeypatch.setattr(runner.os, "killpg", lambda pid, sig: killed.append((pid, sig)))
    monkeypatch.setattr(runner, "active_bytes", lambda _directory: runner.CAPS["active_output_bytes"] + 1)
    receipt = runner.supervise("/private/febio4", tmp_path, {},
        popen=lambda *_args, **_kwargs: process,
        rss_observer=lambda *_args, **_kwargs: (0, [{"pid": 424242}]), sleep=lambda _: None)
    assert process.waited and killed
    assert receipt["kill_reason"] == "active_output_cap"
    assert receipt["status"] == "failed_or_incomplete"


def test_wall_cap_kills_one_native_attempt(tmp_path, monkeypatch):
    process = FakeProcess()
    killed = []
    monkeypatch.setattr(runner.os, "killpg", lambda pid, sig: killed.append((pid, sig)))
    ticks = iter((0., 601., 601.))
    monkeypatch.setattr(runner.time, "monotonic", lambda: next(ticks))
    receipt = runner.supervise("/private/febio4", tmp_path, {},
        popen=lambda *_args, **_kwargs: process,
        rss_observer=lambda *_args, **_kwargs: (0, [{"pid": process.pid}]), sleep=lambda _: None)
    assert receipt["native_calls_attempted"] == 1
    assert receipt["kill_reason"] == "wall_cap"
    assert killed == [(process.pid, runner.signal.SIGKILL)] and process.waited


def test_clean_exit_does_not_count_as_numerical_pass(tmp_path):
    process = FakeProcess(0)
    receipt = runner.supervise("/private/febio4", tmp_path, {},
        popen=lambda *_args, **_kwargs: process,
        rss_observer=lambda *_args, **_kwargs: (0, []), sleep=lambda _: None)
    assert receipt["status"] == "native_exit_zero"
    with pytest.raises(ValueError, match="inventory"):
        runner.inspect_outputs(tmp_path, receipt)


def test_backend_selection_requires_one_actual_accelerate_line(tmp_path):
    for name in (runner.CASE + ".feb", runner.CASE + ".log",
                 runner.CASE + ".nodes.log", runner.CASE + ".elements.log",
                 "console.txt", "receipt.json"):
        (tmp_path / name).write_text("placeholder\n")
    receipt = {"status": "native_exit_zero"}
    for line in ("* Selecting linear solver skyline *", "\n",
                 "* Selecting linear solver accelerate *\n* Selecting linear solver accelerate *",
                 "not selecting linear solver accelerate",
                 "* Selecting linear solver accelerate ... fallback skyline *",
                 "* Selecting linear solver accelerate *\nfallback skyline"):
        (tmp_path / "console.txt").write_text(line)
        with pytest.raises(ValueError, match="Accelerate"):
            runner.inspect_outputs(tmp_path, receipt)


def test_output_inventory_and_ascii_mutations_are_rejected(tmp_path):
    names = {runner.CASE + extension for extension in
             (".feb", ".log", ".nodes.log", ".elements.log")}
    names.update(("console.txt", "receipt.json"))
    for name in names:
        (tmp_path / name).write_text("placeholder\n")
    (tmp_path / "console.txt").write_text("* Selecting linear solver accelerate *\n")
    (tmp_path / "extra.log").write_text("unexpected\n")
    with pytest.raises(ValueError, match="inventory"):
        runner.inspect_outputs(tmp_path, {"status": "native_exit_zero"})
    (tmp_path / "extra.log").unlink()
    (tmp_path / (runner.CASE + ".nodes.log")).write_bytes(b"\xff")
    with pytest.raises(UnicodeDecodeError):
        runner.inspect_outputs(tmp_path, {"status": "native_exit_zero"})


def test_thread_environment_scrubs_ambient_overrides(monkeypatch):
    monkeypatch.setenv("DYLD_LIBRARY_PATH", "/wrong")
    monkeypatch.setenv("OMP_NUM_THREADS", "8")
    monkeypatch.setenv("PYTHONPATH", "/wrong")
    monkeypatch.setenv("MKLROOT", "/wrong")
    child = runner.private_environment()
    assert "DYLD_LIBRARY_PATH" not in child
    assert "PYTHONPATH" not in child and "MKLROOT" not in child
    assert child["OMP_NUM_THREADS"] == "1" and child["OMP_DYNAMIC"] == "FALSE"
