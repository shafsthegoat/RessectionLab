"""Source-only n9 release and adversarial controls; no FEBio invocation."""
from __future__ import annotations

import json
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import pytest

from scripts import mechanics_nonpatient_n9_affine_supervisor as runner


def _binding(relative: str) -> dict:
    return {"path": relative, "sha256": runner.core.sha_bytes((runner.ROOT / relative).read_bytes())}


def test_preparation_is_null_and_third_frozen_row():
    prep = json.loads((runner.ROOT / runner.PREPARATION).read_text())
    runner.validate_preparation(prep)
    assert prep["release"] is None and prep["case_id"] == "n9_affine"
    assert prep["prior_receipts"] == [{"path": path, "sha256": digest,
                                       "elapsed_seconds": elapsed}
                                      for _, path, digest, _, _, _, elapsed in runner.PRIOR]
    assert runner.AGGREGATE == {"maximum_native_calls": 7,
                                "aggregate_native_wall_seconds": 1800,
                                "prior_calls": 2, "case_index": 2}
    for mutation in ({"release": {}}, {"case_id": "n9_nonuniform"},
                     {"aggregate_caps": {**runner.AGGREGATE, "prior_calls": 1}},
                     {"caps": {**runner.CAPS, "wall_seconds": 601}},
                     {"prior_receipts": prep["prior_receipts"][:-1]}):
        with pytest.raises(ValueError):
            runner.validate_preparation({**prep, **mutation})


def test_cli_missing_release_fails_before_output_creation(tmp_path):
    output = runner.ROOT / runner.OUTPUT
    before = output.exists()
    result = subprocess.run([sys.executable, "-B", "-m",
                             "scripts.mechanics_nonpatient_n9_affine_supervisor", "--execute"],
                            cwd=runner.ROOT, capture_output=True, text=True, timeout=10)
    assert result.returncode != 0 and "--release" in result.stderr
    assert output.exists() is before
    with pytest.raises(ValueError, match="release"):
        runner.execute(tmp_path / "missing.json", root=tmp_path)
    assert not (tmp_path / runner.OUTPUT).exists()


def test_existing_output_directory_refuses_second_attempt(tmp_path, monkeypatch):
    destination = tmp_path / runner.OUTPUT
    destination.mkdir(parents=True)
    monkeypatch.setattr(runner.core, "read_release", lambda *_: (b"{}", (1, 1)))
    monkeypatch.setattr(runner, "validate_release", lambda *_args, **_kwargs: {})
    with pytest.raises(FileExistsError):
        runner.execute(tmp_path / "release.json", root=tmp_path)
    assert list(destination.iterdir()) == []


def _fake_prior_receipts(tmp_path: Path, monkeypatch) -> tuple[list[dict], dict]:
    parent = {"sha256": "a" * 64, "runtime_identity_sha256": "b" * 64,
              "backend_profile_sha256": "c" * 64}
    rows, bindings = [], []
    previous_digest = None
    for index, (case, path, _, commit, release_digest, deck_digest, elapsed) in enumerate(runner.PRIOR):
        names = {case + ".feb", case + ".log", case + ".nodes.log",
                 case + ".elements.log", "console.txt"}
        output_bindings = {}
        for name in names:
            file = tmp_path / Path(path).parent / name
            file.parent.mkdir(parents=True, exist_ok=True)
            payload = (case + ":" + name).encode()
            file.write_bytes(payload)
            output_bindings[name] = {"bytes": len(payload), "sha256": runner.core.sha_bytes(payload)}
        receipt = {"schema": ("nonpatient-n5-one-call-receipt-v1" if index == 0 else
                              "nonpatient-n5-nonuniform-one-call-receipt-v1"),
                   "case_id": case, "status": "passed_numerical_software_only",
                   "native_calls_attempted": 1, "no_retry": True, "exit_code": 0,
                   "kill_reason": None, "source_commit": commit,
                   "release_sha256": release_digest, "deck_sha256": deck_digest,
                   "parent_declaration_sha256": parent["sha256"],
                   "runtime_identity_sha256": parent["runtime_identity_sha256"],
                   "backend_profile_sha256": parent["backend_profile_sha256"],
                   "backend_selection": [runner.EXPECTED_SELECTION],
                   "saved_output_checks": {"passed": True},
                   ("affine_oracle" if index == 0 else "nonuniform_readout"): {"passed": True},
                   "elapsed_seconds": elapsed, "output_bindings": output_bindings}
        if index:
            receipt.update(prior_receipt_sha256=previous_digest,
                           prior_native_wall_seconds=runner.PRIOR[0][6],
                           aggregate_native_calls_attempted=2,
                           aggregate_native_wall_seconds_through_this_attempt=runner.PRIOR[0][6] + elapsed)
        payload = json.dumps(receipt, sort_keys=True).encode()
        file = tmp_path / path
        file.write_bytes(payload)
        digest = runner.core.sha_bytes(payload)
        rows.append((case, path, digest, commit, release_digest, deck_digest, elapsed))
        bindings.append({"path": path, "sha256": digest})
        previous_digest = digest
    monkeypatch.setattr(runner, "PRIOR", tuple(rows))
    return bindings, parent


def test_both_prior_receipts_and_saved_native_hashes_are_required(tmp_path, monkeypatch):
    bindings, parent = _fake_prior_receipts(tmp_path, monkeypatch)
    result = runner.validate_prior_receipts(bindings, parent, root=tmp_path)
    assert result["calls_consumed"] == 2
    assert result["elapsed_seconds"] == pytest.approx(1.0384037501644343)
    assert result["elapsed_seconds"] + runner.CAPS["wall_seconds"] < 1800
    with pytest.raises(ValueError, match="Both ordered"):
        runner.validate_prior_receipts(bindings[:-1], parent, root=tmp_path)
    with pytest.raises(ValueError, match="order"):
        runner.validate_prior_receipts(bindings[::-1], parent, root=tmp_path)
    saved = tmp_path / Path(bindings[1]["path"]).parent / "n5_nonuniform.nodes.log"
    saved.write_bytes(b"tampered")
    with pytest.raises(ValueError, match="native output changed"):
        runner.validate_prior_receipts(bindings, parent, root=tmp_path)
    saved.unlink()
    with pytest.raises(ValueError, match="inventory"):
        runner.validate_prior_receipts(bindings, parent, root=tmp_path)


def test_prior_status_and_accounting_mutation_are_detected(tmp_path, monkeypatch):
    bindings, parent = _fake_prior_receipts(tmp_path, monkeypatch)
    file = tmp_path / bindings[1]["path"]
    original = file.read_bytes()
    for key, value in (("status", "failed_or_incomplete"),
                       ("aggregate_native_calls_attempted", 1),
                       ("nonuniform_readout", {"passed": False})):
        damaged = json.loads(original)
        damaged[key] = value
        file.write_text(json.dumps(damaged))
        with pytest.raises(ValueError, match="changed"):
            runner.validate_prior_receipts(bindings, parent, root=tmp_path)
    file.write_bytes(original)
    monkeypatch.setattr(runner, "AGGREGATE", {**runner.AGGREGATE,
                                              "aggregate_native_wall_seconds": 600})
    with pytest.raises(ValueError, match="aggregate"):
        runner.validate_prior_receipts(bindings, parent, root=tmp_path)


def test_source_commit_and_working_blob_drift_rejected(tmp_path, monkeypatch):
    bindings = {}
    for relative in runner.SOURCE_PATHS:
        path = tmp_path / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        data = relative.encode()
        path.write_bytes(data)
        bindings[relative] = {"path": relative, "sha256": runner.core.sha_bytes(data)}
    release = {"source_commit": "a" * 40, "source_bindings": bindings}
    assert len(runner.source_hashes(release, root=tmp_path, require_git=False)) == len(runner.SOURCE_PATHS)
    with pytest.raises(ValueError, match="closure"):
        runner.source_hashes({**release, "source_bindings": {}}, root=tmp_path, require_git=False)
    (tmp_path / runner.SOURCE_PATHS[0]).write_bytes(b"changed")
    with pytest.raises(ValueError, match="changed"):
        runner.source_hashes(release, root=tmp_path, require_git=False)
    monkeypatch.setattr(runner.subprocess, "run",
                        lambda *_args, **_kwargs: SimpleNamespace(stdout=("c" * 40 + "\n").encode()))
    with pytest.raises(ValueError, match="HEAD differs"):
        runner.source_hashes(release, root=tmp_path)


def test_release_wrong_case_runtime_deck_and_predecessor_fail(monkeypatch):
    from scripts import mechanics_hbe_backend as backend
    from scripts import mechanics_nonpatient_sparse_deck as deck
    parent = json.loads((runner.ROOT / runner.PARENT).read_text())
    xml, _ = deck.build_deck(runner.CASE)
    release = {"schema": "nonpatient-n9-affine-one-call-release-v1",
               "status": "root_released_one_native_call", "case_id": runner.CASE,
               "source_commit": "a" * 40, "source_bindings": {},
               "preparation": _binding(runner.PREPARATION), "parent_declaration": _binding(runner.PARENT),
               "backend_profile": parent["runtime"]["backend_profile"],
               "runtime_identity": parent["runtime"]["runtime_identity"],
               "prior_receipts": [{"path": path, "sha256": digest}
                                  for _, path, digest, _, _, _, _ in runner.PRIOR],
               "deck_sha256": runner.core.sha_bytes(xml.encode()), "output_directory": runner.OUTPUT}
    monkeypatch.setattr(runner, "source_hashes", lambda *_args, **_kwargs: {})
    monkeypatch.setattr(runner, "audit_loaded_modules", lambda **_kwargs: None)
    monkeypatch.setattr(runner, "validate_prior_receipts", lambda *_args, **_kwargs:
                        {"sha256": [row[2] for row in runner.PRIOR],
                         "elapsed_seconds": 1.0384037501644343, "calls_consumed": 2})
    runtime = {"profile_id": "accelerate_csc_v1", "runtime_identity": release["runtime_identity"],
               "runtime": {"executable_sha256": "e" * 64,
                           "libraries": {"install/bin/febio4": "e" * 64}}}
    monkeypatch.setattr(backend, "verify_profile", lambda *_args, **_kwargs: runtime)
    assert runner.validate_release(release)["prior"]["calls_consumed"] == 2
    for change, match in (({"case_id": "n9_nonuniform"}, "release"),
                          ({"deck_sha256": "0" * 64}, "deck"),
                          ({"runtime_identity": {"path": "other", "sha256": "0" * 64}}, "runtime"),
                          ({"prior_receipts": release["prior_receipts"][:-1]}, "predecessor")):
        with pytest.raises(ValueError, match=match):
            runner.validate_release({**release, **change})
    runtime["profile_id"] = "skyline"
    with pytest.raises(ValueError, match="backend"):
        runner.validate_release(release)


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


@pytest.mark.parametrize("failure", ["rss", "output", "observer"])
def test_supervised_caps_and_observer_failure_kill_group(tmp_path, monkeypatch, failure):
    process = FakeProcess()
    killed = []
    monkeypatch.setattr(runner.os, "killpg", lambda pid, sig: killed.append((pid, sig)))
    if failure == "output":
        monkeypatch.setattr(runner.core, "active_bytes", lambda _directory: runner.CAPS["active_output_bytes"] + 1)
    def observer(*_args, **_kwargs):
        if failure == "observer":
            raise RuntimeError("RSS observer unavailable")
        return ((runner.CAPS["sampled_process_group_rss_bytes"] + 1 if failure == "rss" else 0),
                [{"pid": process.pid}])
    receipt = runner.supervise("/private/febio4", tmp_path, {},
        popen=lambda *_args, **_kwargs: process, rss_observer=observer, sleep=lambda _: None)
    assert receipt["native_calls_attempted"] == 1 and process.waited and killed
    assert receipt["status"] == "failed_or_incomplete"
    assert receipt["command"][-2:] == ["-o", "n9_affine.log"]


def test_output_backend_inventory_and_parser_corruption(tmp_path):
    names = {runner.CASE + suffix for suffix in (".feb", ".log", ".nodes.log", ".elements.log")}
    names.update(("console.txt", "receipt.json"))
    for name in names:
        (tmp_path / name).write_text("placeholder\n")
    for banner in ("* Selecting linear solver skyline *", "* Selecting linear solver accelerate *\nfallback skyline"):
        (tmp_path / "console.txt").write_text(banner)
        with pytest.raises(ValueError, match="Accelerate"):
            runner.inspect_outputs(tmp_path, {"status": "native_exit_zero"})
    (tmp_path / "console.txt").write_text("")
    with pytest.raises(ValueError, match="Missing or oversized"):
        runner.inspect_outputs(tmp_path, {"status": "native_exit_zero"})
    (tmp_path / "console.txt").write_text("* Selecting linear solver accelerate *\n")
    (tmp_path / "extra.log").write_text("extra\n")
    with pytest.raises(ValueError, match="inventory"):
        runner.inspect_outputs(tmp_path, {"status": "native_exit_zero"})
    (tmp_path / "extra.log").unlink()
    with pytest.raises(ValueError):
        runner.inspect_outputs(tmp_path, {"status": "native_exit_zero"})  # malformed saved parser text


def test_output_calls_affine_readout_with_exact_n9_case(tmp_path, monkeypatch):
    from scripts import mechanics_nonpatient_sparse_output as parser
    from scripts import mechanics_nonpatient_sparse_affine_readout as affine
    names = {runner.CASE + suffix for suffix in (".feb", ".log", ".nodes.log", ".elements.log")}
    names.update(("console.txt", "receipt.json"))
    for name in names:
        (tmp_path / name).write_text("placeholder\n")
    (tmp_path / "console.txt").write_text("* Selecting linear solver accelerate *\n")
    monkeypatch.setattr(runner, "audit_loaded_modules", lambda **_kwargs: None)
    monkeypatch.setattr(parser, "check_case_outputs", lambda case, *_: {"passed": case == runner.CASE})
    seen = []
    def readout(_nodes, _elements, *, case_id):
        seen.append(case_id)
        return {"passed": case_id == "n9_affine"}
    monkeypatch.setattr(affine, "check_affine_readout", readout)
    assert runner.inspect_outputs(tmp_path, {"status": "native_exit_zero"})["affine_oracle"]["passed"]
    assert seen == ["n9_affine"]
