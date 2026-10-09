"""Generated-only continuation controls; no FEBio, patient, or release."""
from __future__ import annotations

import json
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import pytest

from scripts import mechanics_nonpatient_sparse_continuation_supervisor as run
from scripts import mechanics_nonpatient_sparse_feasibility as design


def _binding(relative):
    return {"path": relative, "sha256": run.io.sha_bytes((run.ROOT / relative).read_bytes())}


def _synthetic_chain(tmp_path, monkeypatch, count):
    parent = {"cases": design.EXPECTED_CASES, "sha256": "a" * 64,
              "runtime_identity_sha256": "b" * 64,
              "backend_profile_sha256": "c" * 64}
    original_fixed = [dict(row) for row in run.FIXED_PRIOR]
    fixed, bindings, hashes = [], [], []
    elapsed_total = 0.
    for index in range(count):
        case = design.EXPECTED_CASES[index]
        case_id = case["id"]
        path = (original_fixed[index]["path"] if index < 3 else
                run.output_directory(case_id) + "/receipt.json")
        names = {case_id + suffix for suffix in (".feb", ".log", ".nodes.log", ".elements.log")}
        names.add("console.txt")
        outputs = {}
        for name in names:
            file = tmp_path / Path(path).parent / name
            file.parent.mkdir(parents=True, exist_ok=True)
            payload = (case_id + ":" + name).encode()
            file.write_bytes(payload)
            outputs[name] = {"bytes": len(payload), "sha256": run.io.sha_bytes(payload)}
        elapsed = original_fixed[index]["elapsed_seconds"] if index < 3 else 1.0
        decision = "affine_oracle" if case["load"] == "affine" else "nonuniform_readout"
        record = {"schema": ("nonpatient-n5-one-call-receipt-v1" if index == 0 else
                             "nonpatient-n5-nonuniform-one-call-receipt-v1" if index == 1 else
                             "nonpatient-n9-affine-one-call-receipt-v1" if index == 2 else
                             "nonpatient-sparse-continuation-one-call-receipt-v1"),
                  "case_id": case_id, "status": "passed_numerical_software_only",
                  "native_calls_attempted": 1, "no_retry": True, "exit_code": 0,
                  "kill_reason": None, "source_commit": (original_fixed[index]["source_commit"] if index < 3 else "d" * 40),
                  "release_sha256": (original_fixed[index]["release_sha256"] if index < 3 else "e" * 64),
                  "deck_sha256": (original_fixed[index]["deck_sha256"] if index < 3 else "f" * 64),
                  "elapsed_seconds": elapsed, "parent_declaration_sha256": parent["sha256"],
                  "runtime_identity_sha256": parent["runtime_identity_sha256"],
                  "backend_profile_sha256": parent["backend_profile_sha256"],
                  "backend_selection": [run.SELECTION],
                  "saved_output_checks": {"case_id": case_id, "passed": True},
                  decision: {"case_id": case_id, "passed": True},
                  "output_bindings": outputs}
        if index:
            record.update(prior_receipt_sha256=(hashes[0] if index == 1 else list(hashes)),
                          prior_native_wall_seconds=elapsed_total,
                          prior_native_calls_consumed=index,
                          aggregate_native_calls_attempted=index + 1,
                          aggregate_native_wall_seconds_through_this_attempt=elapsed_total + elapsed)
        if index >= 3:
            record["planned_case_index"] = index
        payload = json.dumps(record, sort_keys=True).encode()
        (tmp_path / path).write_bytes(payload)
        digest = run.io.sha_bytes(payload)
        hashes.append(digest)
        bindings.append({"case_id": case_id, "path": path, "sha256": digest})
        if index < 3:
            fixed.append({**original_fixed[index], "sha256": digest})
        elapsed_total += elapsed
    monkeypatch.setattr(run, "FIXED_PRIOR", tuple(fixed + original_fixed[len(fixed):]))
    return bindings, parent


def test_null_preparation_and_exact_four_case_order():
    prep = json.loads((run.ROOT / run.PREPARATION).read_text())
    run.validate_preparation(prep)
    assert prep["release"] is None
    assert prep["case_ids"] == list(run.REMAINING)
    assert run.AGGREGATE["maximum_native_calls"] == 7
    assert run.CAPS["wall_seconds"] == 600
    for changed in ({"release": {}}, {"case_ids": prep["case_ids"][::-1]},
                    {"caps": {**run.CAPS, "wall_seconds": 601}},
                    {"fixed_prior_receipts": prep["fixed_prior_receipts"][:-1]}):
        with pytest.raises(ValueError):
            run.validate_preparation({**prep, **changed})


@pytest.mark.parametrize("index,case", list(enumerate(run.REMAINING, start=3)))
def test_output_path_is_case_specific_and_single_attempt(index, case):
    assert run.output_directory(case) == f"{run.OUTPUT_ROOT}/{case}/attempt-01"
    assert run.canonical_receipt(index, case) == run.output_directory(case) + "/receipt.json"
    with pytest.raises(ValueError):
        run.output_directory("n5_affine")


def test_module_cli_requires_release_before_creating_output():
    existing = [(run.ROOT / run.output_directory(case)).exists() for case in run.REMAINING]
    result = subprocess.run([sys.executable, "-B", "-m",
                             "scripts.mechanics_nonpatient_sparse_continuation_supervisor", "--execute"],
                            cwd=run.ROOT, capture_output=True, text=True, timeout=10)
    assert result.returncode != 0 and "--release" in result.stderr
    assert [(run.ROOT / run.output_directory(case)).exists() for case in run.REMAINING] == existing


@pytest.mark.parametrize("count", [3, 4, 5, 6])
def test_generated_ordered_chain_checks_all_prior_rows_and_budget(tmp_path, monkeypatch, count):
    bindings, parent = _synthetic_chain(tmp_path, monkeypatch, count)
    result = run.validate_prior_chain(bindings, count, parent, root=tmp_path)
    assert result["calls_consumed"] == count and result["sha256"] == [row["sha256"] for row in bindings]
    assert result["elapsed_seconds"] + 600 < 1800
    with pytest.raises(ValueError, match="ordered"):
        run.validate_prior_chain(bindings[:-1], count, parent, root=tmp_path)
    with pytest.raises(ValueError, match="order"):
        run.validate_prior_chain(bindings[::-1], count, parent, root=tmp_path)


def test_failing_or_altered_predecessor_stops_all_later_rows(tmp_path, monkeypatch):
    bindings, parent = _synthetic_chain(tmp_path, monkeypatch, 5)
    target = tmp_path / bindings[3]["path"]
    original = target.read_bytes()
    for key, value in (("status", "failed_or_incomplete"),
                       ("aggregate_native_calls_attempted", 99),
                       ("prior_receipt_sha256", []),
                       ("nonuniform_readout", {"case_id": "n9_nonuniform", "passed": False})):
        damaged = json.loads(original)
        damaged[key] = value
        target.write_text(json.dumps(damaged))
        with pytest.raises(ValueError, match="changed"):
            run.validate_prior_chain(bindings, 5, parent, root=tmp_path)
    target.write_bytes(original)
    native = target.parent / "n9_nonuniform.nodes.log"
    native.write_bytes(b"corrupted")
    with pytest.raises(ValueError, match="native output changed"):
        run.validate_prior_chain(bindings, 5, parent, root=tmp_path)
    native.unlink()
    with pytest.raises(ValueError, match="inventory"):
        run.validate_prior_chain(bindings, 5, parent, root=tmp_path)


def test_rebound_failing_predecessor_receipt_still_stops_sequence(tmp_path, monkeypatch):
    bindings, parent = _synthetic_chain(tmp_path, monkeypatch, 4)
    target = tmp_path / bindings[3]["path"]
    original = json.loads(target.read_text())
    for change in ({"status": "failed_or_incomplete"},
                   {"aggregate_native_calls_attempted": 3},
                   {"nonuniform_readout": {"case_id": "n13_nonuniform", "passed": True}}):
        damaged = {**original, **change}
        payload = json.dumps(damaged, sort_keys=True).encode()
        target.write_bytes(payload)
        rebound = [*bindings[:-1], {**bindings[-1], "sha256": run.io.sha_bytes(payload)}]
        with pytest.raises(ValueError, match="decision|chain"):
            run.validate_prior_chain(rebound, 4, parent, root=tmp_path)


def test_full_prospective_budget_must_fit_before_call(tmp_path, monkeypatch):
    bindings, parent = _synthetic_chain(tmp_path, monkeypatch, 3)
    monkeypatch.setattr(run, "AGGREGATE", {**run.AGGREGATE,
                                          "aggregate_native_wall_seconds": 600})
    with pytest.raises(ValueError, match="aggregate"):
        run.validate_prior_chain(bindings, 3, parent, root=tmp_path)


def test_source_closure_and_commit_drift(tmp_path, monkeypatch):
    bindings = {}
    for relative in run.SOURCE_PATHS:
        path = tmp_path / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = relative.encode()
        path.write_bytes(payload)
        bindings[relative] = {"path": relative, "sha256": run.io.sha_bytes(payload)}
    release = {"source_commit": "a" * 40, "source_bindings": bindings}
    assert len(run.source_hashes(release, root=tmp_path, require_git=False)) == len(run.SOURCE_PATHS)
    with pytest.raises(ValueError, match="closure"):
        run.source_hashes({**release, "source_bindings": {}}, root=tmp_path, require_git=False)
    (tmp_path / run.SOURCE_PATHS[0]).write_bytes(b"changed")
    with pytest.raises(ValueError, match="changed"):
        run.source_hashes(release, root=tmp_path, require_git=False)
    monkeypatch.setattr(run.subprocess, "run",
                        lambda *_args, **_kwargs: SimpleNamespace(stdout=("b" * 40 + "\n").encode()))
    with pytest.raises(ValueError, match="HEAD differs"):
        run.source_hashes(release, root=tmp_path)


def test_release_exact_case_runtime_deck_and_prior_sequence(monkeypatch):
    from scripts import mechanics_hbe_backend as backend
    from scripts import mechanics_nonpatient_sparse_deck as deck
    parent = json.loads((run.ROOT / run.PARENT).read_text())
    case = "n9_nonuniform"
    xml, _ = deck.build_deck(case)
    release = {"schema": "nonpatient-sparse-continuation-one-call-release-v1",
               "status": "root_released_one_native_call", "case_index": 3, "case_id": case,
               "source_commit": "a" * 40, "source_bindings": {},
               "preparation": _binding(run.PREPARATION), "parent_declaration": _binding(run.PARENT),
               "backend_profile": parent["runtime"]["backend_profile"],
               "runtime_identity": parent["runtime"]["runtime_identity"],
               "prior_receipts": [{"case_id": row["case_id"], "path": row["path"],
                                   "sha256": row["sha256"]} for row in run.FIXED_PRIOR],
               "deck_sha256": run.io.sha_bytes(xml.encode()),
               "output_directory": run.output_directory(case)}
    monkeypatch.setattr(run, "source_hashes", lambda *_args, **_kwargs: {})
    monkeypatch.setattr(run, "audit_loaded_modules", lambda **_kwargs: None)
    def prior_check(bindings, case_index, *_args, **_kwargs):
        if len(bindings) != case_index:
            raise ValueError("Incomplete predecessor chain")
        return {"sha256": [row["sha256"] for row in run.FIXED_PRIOR],
                "elapsed_seconds": 4.687578625278547, "calls_consumed": 3}
    monkeypatch.setattr(run, "validate_prior_chain", prior_check)
    runtime = {"profile_id": "accelerate_csc_v1", "runtime_identity": release["runtime_identity"],
               "runtime": {"executable_sha256": "e" * 64,
                           "libraries": {"install/bin/febio4": "e" * 64}}}
    monkeypatch.setattr(backend, "verify_profile", lambda *_args, **_kwargs: runtime)
    assert run.validate_release(release)["case_id"] == case
    for changed in ({"case_index": 4}, {"case_id": "n13_affine"},
                    {"output_directory": run.output_directory("n13_affine")},
                    {"deck_sha256": "0" * 64},
                    {"runtime_identity": {"path": "other", "sha256": "0" * 64}},
                    {"prior_receipts": release["prior_receipts"][:-1]}):
        with pytest.raises(ValueError):
            run.validate_release({**release, **changed})
    runtime["profile_id"] = "skyline"
    with pytest.raises(ValueError, match="backend"):
        run.validate_release(release)


@pytest.mark.parametrize("index,case_id", list(enumerate(run.REMAINING, start=3)))
def test_each_release_selects_only_its_frozen_row(monkeypatch, index, case_id):
    from scripts import mechanics_hbe_backend as backend
    from scripts import mechanics_nonpatient_sparse_deck as deck
    parent = json.loads((run.ROOT / run.PARENT).read_text())
    xml, _ = deck.build_deck(case_id)
    prefix = [{"case_id": row["case_id"], "path": row["path"], "sha256": row["sha256"]}
              for row in run.FIXED_PRIOR]
    prefix += [{"case_id": previous, "path": run.output_directory(previous) + "/receipt.json",
                "sha256": "a" * 64} for previous in run.REMAINING[:index - 3]]
    release = {"schema": "nonpatient-sparse-continuation-one-call-release-v1",
               "status": "root_released_one_native_call", "case_index": index, "case_id": case_id,
               "source_commit": "a" * 40, "source_bindings": {},
               "preparation": _binding(run.PREPARATION), "parent_declaration": _binding(run.PARENT),
               "backend_profile": parent["runtime"]["backend_profile"],
               "runtime_identity": parent["runtime"]["runtime_identity"],
               "prior_receipts": prefix, "deck_sha256": run.io.sha_bytes(xml.encode()),
               "output_directory": run.output_directory(case_id)}
    monkeypatch.setattr(run, "source_hashes", lambda *_args, **_kwargs: {})
    monkeypatch.setattr(run, "audit_loaded_modules", lambda **_kwargs: None)
    def preceding(bindings, requested_index, *_args, **_kwargs):
        assert requested_index == index and bindings == prefix
        return {"sha256": [row["sha256"] for row in bindings],
                "elapsed_seconds": 4.687578625278547 + (index - 3),
                "calls_consumed": index}
    monkeypatch.setattr(run, "validate_prior_chain", preceding)
    runtime = {"profile_id": "accelerate_csc_v1", "runtime_identity": release["runtime_identity"],
               "runtime": {"executable_sha256": "e" * 64,
                           "libraries": {"install/bin/febio4": "e" * 64}}}
    monkeypatch.setattr(backend, "verify_profile", lambda *_args, **_kwargs: runtime)
    result = run.validate_release(release)
    assert result["case_id"] == case_id and result["case_index"] == index
    assert result["deck"] == xml.encode() and result["output_directory"] == run.output_directory(case_id)


def test_preexisting_case_directory_prevents_second_attempt(tmp_path, monkeypatch):
    directory = tmp_path / run.output_directory("n9_nonuniform")
    directory.mkdir(parents=True)
    monkeypatch.setattr(run.io, "read_release", lambda *_: (b"{}", (1, 1)))
    monkeypatch.setattr(run, "validate_release", lambda *_args, **_kwargs:
                        {"output_directory": run.output_directory("n9_nonuniform"),
                         "case_id": "n9_nonuniform", "case_index": 3})
    with pytest.raises(FileExistsError):
        run.execute(tmp_path / "release.json", root=tmp_path)
    assert list(directory.iterdir()) == []


class FakeProcess:
    pid = 424242

    def __init__(self):
        self.waited = False

    def poll(self):
        return None

    def wait(self, timeout):
        self.waited = True
        return -9


@pytest.mark.parametrize("failure", ["rss", "output", "observer"])
def test_supervised_cap_or_observer_failure_kills_one_child(tmp_path, monkeypatch, failure):
    process = FakeProcess()
    killed = []
    monkeypatch.setattr(run.os, "killpg", lambda pid, sig: killed.append((pid, sig)))
    if failure == "output":
        monkeypatch.setattr(run.io, "active_bytes", lambda *_: run.CAPS["active_output_bytes"] + 1)
    def observer(*_args, **_kwargs):
        if failure == "observer":
            raise RuntimeError("unavailable")
        return ((run.CAPS["sampled_process_group_rss_bytes"] + 1 if failure == "rss" else 0),
                [{"pid": process.pid}])
    receipt = run.supervise("n9_nonuniform", "/private/febio4", tmp_path, {},
                            popen=lambda *_args, **_kwargs: process,
                            rss_observer=observer, sleep=lambda _: None)
    assert receipt["native_calls_attempted"] == 1 and process.waited and killed
    assert receipt["status"] == "failed_or_incomplete"
    assert receipt["command"][-2:] == ["-o", "n9_nonuniform.log"]


def test_supervised_wall_cap_kills_one_child(tmp_path, monkeypatch):
    process = FakeProcess()
    killed = []
    monkeypatch.setattr(run.os, "killpg", lambda pid, sig: killed.append((pid, sig)))
    ticks = iter((0., 601., 601.))
    monkeypatch.setattr(run.time, "monotonic", lambda: next(ticks))
    receipt = run.supervise("n13_affine", "/private/febio4", tmp_path, {},
                            popen=lambda *_args, **_kwargs: process,
                            rss_observer=lambda *_args, **_kwargs: (0, [{"pid": process.pid}]),
                            sleep=lambda _: None)
    assert receipt["native_calls_attempted"] == 1 and receipt["kill_reason"] == "wall_cap"
    assert process.waited and killed


@pytest.mark.parametrize("case_id,readout_type", [("n9_nonuniform", "nonuniform"),
                                                  ("n13_affine", "affine"),
                                                  ("n13_nonuniform", "nonuniform"),
                                                  ("n13_nonuniform_half_step", "nonuniform")])
def test_case_specific_saved_output_dispatch_and_backend(tmp_path, monkeypatch, case_id, readout_type):
    from scripts import mechanics_nonpatient_sparse_output as parser
    from scripts import mechanics_nonpatient_sparse_affine_readout as affine
    from scripts import mechanics_nonpatient_sparse_nonuniform_readout as nonuniform
    names = {case_id + suffix for suffix in (".feb", ".log", ".nodes.log", ".elements.log")}
    names.update(("console.txt", "receipt.json"))
    for name in names:
        (tmp_path / name).write_text("placeholder\n")
    (tmp_path / "console.txt").write_text("* Selecting linear solver accelerate *\n")
    monkeypatch.setattr(run, "audit_loaded_modules", lambda **_kwargs: None)
    monkeypatch.setattr(parser, "check_case_outputs", lambda selected, *_: {"case_id": selected, "passed": True})
    seen = []
    monkeypatch.setattr(affine, "check_affine_readout",
                        lambda *_args, case_id: seen.append(("affine", case_id)) or {"case_id": case_id, "passed": True})
    monkeypatch.setattr(nonuniform, "check_nonuniform_readout",
                        lambda selected, *_args: seen.append(("nonuniform", selected)) or {"case_id": selected, "passed": True})
    result = run.inspect_outputs(case_id, tmp_path, {"status": "native_exit_zero"})
    assert seen == [(readout_type, case_id)] and result["saved_output_checks"]["passed"]
    (tmp_path / "console.txt").write_text("* Selecting linear solver skyline *\n")
    with pytest.raises(ValueError, match="Accelerate"):
        run.inspect_outputs(case_id, tmp_path, {"status": "native_exit_zero"})


def test_half_step_declares_nine_states():
    case = design.EXPECTED_CASES[-1]
    assert case["id"] == "n13_nonuniform_half_step"
    assert [k * case["step_size"] for k in range(case["time_steps"] + 1)] == [
        0., .125, .25, .375, .5, .625, .75, .875, 1.]


def test_half_step_malformed_saved_record_cannot_pass(tmp_path):
    case_id = "n13_nonuniform_half_step"
    names = {case_id + suffix for suffix in (".feb", ".log", ".nodes.log", ".elements.log")}
    names.update(("console.txt", "receipt.json"))
    for name in names:
        (tmp_path / name).write_text("placeholder\n")
    (tmp_path / "console.txt").write_text("* Selecting linear solver accelerate *\n")
    with pytest.raises(ValueError):
        run.inspect_outputs(case_id, tmp_path, {"status": "native_exit_zero"})
