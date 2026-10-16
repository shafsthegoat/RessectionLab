"""Opt-in independent v3 supervisor review; source and generated process doubles.

No patient bytes, native ANTs calls, workers, signals, or production edits.
Run only before the separately authorized patient attempt; seals a new report.
"""
from __future__ import annotations

import ast
import builtins
from contextlib import ExitStack
from datetime import datetime, timezone
import hashlib
import importlib.util
import io
import json
from pathlib import Path
import signal
import subprocess
import sys
import tempfile
from types import SimpleNamespace
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
PINS = {
    "scripts/poc_ixi265_atropos_typed_mask_run.py": "b8d65e0522130df1a4e012811527ab17116a46436805c0152c0a3956a921e9ec",
    "tests/test_poc_ixi265_atropos_typed_mask_run.py": "527cff47ebaefabd17fdaf2e6ba13c1062acdf20fecfe9af4f1c1336f4d22657",
    "artifacts/poc-ixi265-atropos-typed-mask-runner-guard-followup-v3/preparation.json": "8a8335d6ab2f0b671c96eb80e6ae421e17cdaced294a5132dfeaf0c59ccfb6a1",
    "artifacts/poc-ixi265-atropos-typed-mask-runner-preparation-v3/preparation.json": "0e0a5b9069aca2fb0e4677c50a05593fdb5e40f79a62bd9a48dd0e6313d11797",
    "artifacts/poc-ixi265-atropos-typed-mask-runner-guard-followup-v3/initial-runner.py.txt": "dd7d71631c6724f519dd0af76f4f8338e260de9056f0c3f6c5fddae585085d85",
    "artifacts/poc-ixi265-atropos-typed-mask-runner-guard-followup-v3/initial-tests.py.txt": "2c60c8c2a28bfb82d6c5b22a6b8d7c658d5fc66ecbb96f2ce8072a84731eae3e",
    "artifacts/ixi265-typed-mask-independent-v3/supervisor-initial-findings.json": "41e74da8c8edd02418400ab87f73b29bcc155a265f383e47f1be1b15be3d01e9",
    "artifacts/poc-ixi265-atropos-typed-mask-runner-guard-followup-v3/prospective-run-spec.json": "166993c982b28770dfe61c34aae7aae6abee0bb392cb69c55e2b92be25e26194",
    "scripts/poc_ixi265_atropos_typed_mask.py": "211d715a05b2dd269358b7bc5585488a053859ba65c04b9f31f60b69e41ec954",
    "artifacts/ixi265-typed-mask-independent-v3/kernel-preflight-qa.json": "08fc33228534fd2adb392a9093b57790695560ab9a881ee96cebddf0a23636b5",
    "artifacts/ants-mask-dtype-independent-v1/independent-qa.json": "9c4b3bc648b9920ba7de4c5fd0c921b08382796e2ead3467caffc4213ee0c128",
    "artifacts/poc-ixi265-atropos-positive-domain-run-v2/outcome.json": "dce67f0acde21aa11fa99bdaa270108c8b53b5a091514182870e69e01fa78e21",
    "artifacts/poc-ixi265-atropos-run-v1/outcome.json": "55f4dd46f7cc692d4a66799cefca21bfa37430d3f5cf2ebe3eda675b6e4c989e",
}
CHECKS = []


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load(name, relative):
    spec = importlib.util.spec_from_file_location(name, ROOT/relative)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def mark(name):
    CHECKS.append(name)


def forbidden(*args, **kwargs):
    raise AssertionError("Forbidden payload, native execution, signal or process")


def refusal(callback):
    try: callback()
    except (ValueError, TypeError, KeyError): return
    raise AssertionError("Expected guard rejection")


def admitted(base):
    return {"patient_group": "IXI:265", "role": "TRAIN", "training_admitted": False,
            "T1_sha256": base.T1_SHA, "brain_mask_sha256": base.MASK_SHA, "host_sha256": base.HOST_SHA}


def source_paths(root):
    return {"T1": root/"outputs/poc-ixi265/originals/IXI265-T1.nii.gz",
            "brain_mask": root/"outputs/poc-ixi265/synthstrip/brain_mask.nii.gz",
            "host_bundle": root/"outputs/poc-ixi265/IXI265-Guys-0845.ressectionlab"}


def source_checks(runner, base, method):
    prep = runner.preparation_intact()
    assert prep["patient_fitting_released"] is False
    assert prep["kernel_parameters_source_admission_support_native_frames_changed"] is False
    assert runner.lineage() == runner.LINEAGE
    mark("all_frozen_source_control_history_and_exact_native_preflight_pins")
    old_method = runner.kernel().previous().method()
    for key in ("source_hashes", "resources", "parameters", "expected_unknown_indices_native", "expected_fit_domain_voxels"):
        assert method[key] == old_method[key]
    assert method["expected_fit_domain_voxels"] == 1451501
    assert len(method["expected_unknown_indices_native"]) == 8
    assert method["patient_fitting_released"] is False
    assert all(method[key] is False for key in ("anatomy_accepted", "host_accepted", "growth_use_accepted"))
    mark("frozen_v2_eight_unknown_support_parameters_and_unaccepted_semantics")

    initial = ast.parse((ROOT/"artifacts/poc-ixi265-atropos-typed-mask-runner-guard-followup-v3/initial-runner.py.txt").read_text())
    current = ast.parse(Path(runner.__file__).read_text())
    before = {n.name: n for n in initial.body if isinstance(n, ast.FunctionDef)}
    after = {n.name: n for n in current.body if isinstance(n, ast.FunctionDef)}
    for name in before:
        if name != "run": assert ast.dump(before[name]) == ast.dump(after[name]), name
    assert set(after)-set(before) == {"cleanup_worker"}
    worker = ast.unparse(after["worker"])
    assert "typed.fit_arrays(values, domain, image.affine, ants" in worker
    assert "old.positive_domain(values, brain)" in worker and "old.require_unknown_partition" in worker
    assert "unmodified_membership_probability_not_PVE" in worker
    assert "result.set_qform" in worker and "result.set_sform" in worker and "base.require_native_frame" in worker
    assert "np.array_equal(np.asarray(reopened.dataobj), data)" in worker
    mark("repair_changes_no_worker_source_support_kernel_parameters_or_output_math")
    mark("only_typed_kernel_with_raw_outputs_unknown_map_native_frame_and_reopen_checks")

    with patch.object(runner, "kernel", forbidden):
        refusal(lambda: runner.run({}, {}, "/unused", "/unused"))
        with tempfile.TemporaryDirectory() as temp:
            spec = Path(temp)/"unreleased.json"; spec.write_text('{"released_single_attempt":false}')
            refusal(lambda: runner.worker(spec))
    mark("unreleased_supervisor_and_worker_refuse_before_dependencies")
    bad_metadata = [{**admitted(base), **change} for change in (
        {"role": "SELECT"}, {"role": "EVAL"}, {"patient_group": "UPENN:001"},
        {"patient_group": "UTSW:001"}, {"training_admitted": True}, {"T1_sha256": "changed"},
        {"nested": {"roles": {"SELECT": True}}}, {"source": {"provenance": "generated"}})]
    lightweight = SimpleNamespace(previous=lambda: SimpleNamespace(reference=lambda: base))
    with patch.object(runner, "kernel", lambda: lightweight), patch.object(runner, "canonical_paths", forbidden):
        for meta in bad_metadata:
            refusal(lambda: runner.run({}, meta, "/unused", "/unused", released_single_attempt=True))
            with tempfile.TemporaryDirectory() as temp:
                spec = Path(temp)/"bad.json"; spec.write_text(json.dumps({"released_single_attempt": True, "metadata": meta}))
                refusal(lambda: runner.worker(spec))
            result = runner.source_continuity({}, meta, base)
            assert not result["payloads_hashed"] and not result["all_unchanged"]
    mark("eight_metadata_tamper_cases_refused_before_paths_in_supervisor_worker_and_postcheck")

    for fault in ("foreign", "extra", "leaf_symlink", "ancestor_symlink"):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp).resolve(); paths = source_paths(root)
            for path in paths.values(): path.parent.mkdir(parents=True, exist_ok=True)
            if fault == "foreign": paths["brain_mask"] = root/"protected.nii.gz"
            elif fault == "extra": paths["extra"] = root/"foreign.nii.gz"
            elif fault == "leaf_symlink": paths["brain_mask"].symlink_to(root/"foreign")
            else:
                paths["brain_mask"].parent.rmdir()
                paths["brain_mask"].parent.symlink_to(root/"foreign", target_is_directory=True)
            with patch.object(runner, "ROOT", root), patch.object(runner, "sha", forbidden):
                result = runner.source_continuity(paths, admitted(base), base)
                assert not result["all_unchanged"] and not result["payloads_hashed"]
        mark("all_paths_before_hash_"+fault)

    with tempfile.TemporaryDirectory() as temp:
        root = Path(temp).resolve(); paths = source_paths(root); seen = []
        expected = {"T1": base.T1_SHA, "brain_mask": base.MASK_SHA, "host_bundle": base.HOST_SHA}
        def hash_double(path):
            name = next(k for k, p in paths.items() if p == path); seen.append(name)
            if name == "brain_mask": raise FileNotFoundError("generated missing source")
            return expected[name]
        with patch.object(runner, "ROOT", root), patch.object(runner, "sha", hash_double):
            result = runner.source_continuity(paths, admitted(base), base)
        assert seen == list(paths) and result["files"]["T1"]["unchanged"] and result["files"]["host_bundle"]["unchanged"]
        assert "read_failure" in result["files"]["brain_mask"]
        record = {"status": "failed", "failure": "original failure"}
        runner.attach_continuity(record, result)
        assert record["failure"] == "original failure" and record["additional_source_continuity_failure"]
    mark("source_postcheck_attempts_all_three_files_and_preserves_original_failure")


def resource_checks(base):
    assert base.plan()["resources"] == {"threads": 2, "rss_limit_bytes": 3*1024**3,
        "minimum_available_bytes": 1024**3, "timeout_seconds": 600.0,
        "output_limit_bytes": 512*1024**2, "free_disk_floor_bytes": 50*1024**3,
        "sample_interval_seconds": .2, "max_voxels": 12000000, "attempts": 1}
    boundary = [3*1024**3, 1024**3, 600, 512*1024**2, 50*1024**3]
    assert base.resource_violation(*boundary) is None
    for index, direction, reason in [(0, 1, "sampled_process_family_RSS_limit"), (1, -1, "available_memory_floor"),
            (2, 1, "time_limit"), (3, 1, "output_and_temporary_bytes_limit"), (4, -1, "free_disk_floor")]:
        changed = boundary.copy(); changed[index] += direction
        assert base.resource_violation(*changed) == reason
        mark("resource_boundary_"+reason)
    children = [SimpleNamespace(memory_info=lambda: SimpleNamespace(rss=30), is_running=lambda: True),
                SimpleNamespace(memory_info=lambda: SimpleNamespace(rss=999), is_running=lambda: False)]
    class Process:
        def __init__(self, pid): self.pid = pid
        def memory_info(self): return SimpleNamespace(rss=20 if self.pid == 1001 else 10)
        def children(self, recursive): assert recursive; return children
        def is_running(self): return True
    assert base.sampled_rss(SimpleNamespace(Process=Process), 1001) == 60
    mark("sampled_rss_includes_supervisor_worker_and_live_recursive_children")


def simulate(runner, base, method, scenario):
    """Run actual full supervisor against absent source paths and fake OS process."""
    failure_scenarios = {"exit_race", "signal_permission", "cleanup_helper_error", "wait_timeout"}
    with tempfile.TemporaryDirectory() as temp, ExitStack() as stack:
        root = Path(temp).resolve(); runtime = root/"runtime"; runtime.mkdir()
        prep = root/"prep"; prep.mkdir(); (prep/"preparation.json").write_text("{}")
        output = root/"candidate-output"; paths = source_paths(root)
        old_markers = ("ixi265-n4-atropos-single-attempt.json", "ixi265-n4-atropos-positive-domain-v2-attempt.json")
        for name in old_markers: (runtime/name).write_text("generated immutable history")
        if scenario == "marker_exists": (runtime/runner.MARKER).write_text("prior generated attempt")
        if scenario == "output_exists": output.mkdir()
        light = SimpleNamespace(previous=lambda: SimpleNamespace(reference=lambda: base))
        for name, value in {"ROOT": root, "RUNTIME": runtime, "OUTPUT": output, "PREPARATION": prep,
                "kernel": lambda: light, "method": lambda: method,
                "preparation_intact": lambda: {"frozen_files": []}}.items():
            stack.enter_context(patch.object(runner, name, value))
        calls = []; runtime_calls = 0; popen_calls = []; signals = []
        def verify(_):
            nonlocal runtime_calls
            runtime_calls += 1
            if runtime_calls > 1: calls.append("runtime")
            return {"generated_runtime": "changed" if scenario == "runtime_false" and runtime_calls > 1 else "original"}
        def source_postcheck(*args):
            calls.append("source")
            return {"all_unchanged": scenario != "source_false", "generated_hash_double": True}
        def lineage_postcheck():
            calls.append("lineage")
            return {} if scenario == "lineage_false" else dict(runner.LINEAGE)
        stack.enter_context(patch.object(runner, "verify_runtime", verify))
        stack.enter_context(patch.object(runner, "lineage", lineage_postcheck))
        stack.enter_context(patch.object(runner, "source_continuity", source_postcheck))
        stack.enter_context(patch.object(base, "bound_inputs", lambda paths, metadata: calls.append("pre_source_hash_double")))
        stack.enter_context(patch.object(base, "sampled_rss", lambda ps, pid=None: 4*1024**3 if pid and scenario in failure_scenarios else 64*1024**2))
        # Keep real resource_violation and real atomic JSON writes throughout.
        stack.enter_context(patch.dict(sys.modules, {"psutil": SimpleNamespace(
            virtual_memory=lambda: SimpleNamespace(available=8*1024**3),
            disk_usage=lambda path: SimpleNamespace(free=100*1024**3), NoSuchProcess=ProcessLookupError)}))
        class Child:
            pid = 1001
            returncode = None if scenario in failure_scenarios else (-11 if scenario == "native_failure" else 0)
            waits = 0
            def poll(self): return self.returncode
            def wait(self, timeout):
                assert timeout == 5; self.waits += 1
                if scenario == "wait_timeout" and self.waits == 1: raise subprocess.TimeoutExpired("generated", timeout)
                self.returncode = -9 if scenario == "wait_timeout" else (0 if scenario == "exit_race" else -15)
                return self.returncode
        child = Child()
        def popen(args, **kwargs):
            popen_calls.append(args)
            assert args[:2] == [str(runtime/"bin/python"), "-I"]
            assert kwargs["start_new_session"] is True
            assert "PYTHONPATH" not in kwargs["env"]
            assert all(kwargs["env"][key] == "2" for key in ("ITK_GLOBAL_DEFAULT_NUMBER_OF_THREADS", "OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"))
            if scenario not in failure_scenarios|{"native_failure", "missing_candidate"}:
                (output/"candidate.json").write_text('{"generated":true}')
            if scenario == "history_false": (runtime/old_markers[1]).write_text("generated history drift")
            return child
        def kill(pid, sig):
            assert pid == child.pid; signals.append(sig)
            if scenario == "exit_race": child.returncode = 0; raise ProcessLookupError("generated exit race")
            if scenario == "signal_permission": raise PermissionError("generated signal permission failure")
            if sig == signal.SIGKILL: child.returncode = -9
        stack.enter_context(patch.object(runner.subprocess, "Popen", popen))
        stack.enter_context(patch.object(runner.os, "killpg", kill))
        if scenario == "cleanup_helper_error":
            def cleanup(_): raise RuntimeError("generated helper fault")
            stack.enter_context(patch.object(runner, "cleanup_worker", cleanup))
        if scenario in {"marker_exists", "output_exists"}:
            try: runner.run(paths, admitted(base), runtime, output, released_single_attempt=True)
            except (ValueError, FileExistsError): pass
            else: raise AssertionError("Reused attempt accepted")
            assert not popen_calls
            if scenario == "marker_exists": assert (runtime/runner.MARKER).read_text() == "prior generated attempt"
            mark("full_supervisor_"+scenario+"_refused_before_worker")
            return {"scenario": scenario, "worker_started": False, "refused": True}
        result = runner.run(paths, admitted(base), runtime, output, released_single_attempt=True)
        assert len(popen_calls) == 1 and calls == ["pre_source_hash_double", "source", "runtime", "lineage"]
        assert json.loads((output/"attempt.json").read_text()) == result
        assert all(result[key] is False for key in ("anatomy_accepted", "host_accepted", "growth_use_accepted"))
        assert not any(path.exists() for path in paths.values())
        assert result["status"] == ("fitted_candidate_unaccepted" if scenario == "success" else "failed")
        if scenario in failure_scenarios:
            assert result["failure"] == "sampled_process_family_RSS_limit"
            if scenario == "exit_race": assert result["cleanup"]["normal_exit_races"] == ["SIGTERM"]
            if scenario in {"signal_permission", "cleanup_helper_error"}: assert result["cleanup"]["errors"]
            if scenario == "wait_timeout": assert signals == [signal.SIGTERM, signal.SIGKILL] and child.waits == 2
        if scenario == "native_failure": assert result["exit_code"] == -11 and result["failure"] == "worker_failed_see_retained_log"
        if scenario == "missing_candidate": assert result["failure"] == "Missing candidate receipt"
        if scenario == "runtime_false": assert result["runtime_unchanged"] is False and "Runtime continuity" in result["continuity_guard_error"]
        if scenario == "lineage_false": assert result["lineage_unchanged"] is False and "Lineage continuity" in result["continuity_guard_error"]
        if scenario == "source_false": assert result["failure"] == "Source continuity postcheck failed"
        if scenario == "history_false": assert not all(result["historical_markers_unchanged"].values())
        mark("full_supervisor_"+scenario+"_durable_receipt_and_all_postchecks")
        return {"scenario": scenario, "status": result["status"], "failure": result.get("failure"),
            "all_postchecks_executed": True, "durable_final_receipt_matches_returned_record": True,
            "cleanup": result.get("cleanup"), "worker_invocations": 1, "actual_worker_started": False}


def main():
    destination = ROOT/"artifacts/ixi265-typed-mask-independent-v3/supervisor-qa.json"
    assert not destination.exists(), "Frozen report already exists"
    for relative, pin in PINS.items(): assert sha(ROOT/relative) == pin, relative
    guard = load("independent_v2_guard_helpers", "tests/first_ixi_positive_domain_qa.py")
    no_models = guard.NoModels(); sys.meta_path.insert(0, no_models)
    try:
        with patch.object(builtins, "open", guard.open_guard(builtins.open)), patch.object(io, "open", guard.open_guard(io.open)), \
                patch.object(subprocess, "Popen", forbidden), patch("os.killpg", forbidden):
            runner = load("independent_guarded_v3_supervisor", "scripts/poc_ixi265_atropos_typed_mask_run.py")
            assert not runner.OUTPUT.exists() and not (runner.RUNTIME/runner.MARKER).exists()
            base = runner.kernel().previous().reference(); method = runner.method()
            source_checks(runner, base, method); resource_checks(base)
            scenarios = [simulate(runner, base, method, name) for name in (
                "success", "native_failure", "missing_candidate", "exit_race", "signal_permission",
                "cleanup_helper_error", "wait_timeout", "runtime_false", "lineage_false", "source_false",
                "history_false", "marker_exists", "output_exists")]
            assert not runner.OUTPUT.exists() and not (runner.RUNTIME/runner.MARKER).exists()
    finally: sys.meta_path.remove(no_models)
    for relative, pin in PINS.items(): assert sha(ROOT/relative) == pin, relative
    report = {"schema": "ixi265-typed-supervisor-independent-review-v3",
        "status": "pass_prospective_guarded_supervisor_only_patient_run_unreleased",
        "created_utc": datetime.now(timezone.utc).isoformat(), "pins": PINS,
        "auditor_sha256": sha(__file__), "controls_passed": len(CHECKS), "controls": CHECKS,
        "whole_supervisor_generated_scenarios": scenarios,
        "scope": {"source_and_small_metadata_only": True, "patient_payload_reads": False,
            "native_ANTs_imports_or_calls": False, "actual_child_processes_or_signals": False,
            "real_attempt_marker_or_output_created": False, "old_failure_and_audit_bytes_unchanged": True},
        "fixed_initial_findings": ["Process-exit cleanup race retains original failure and completes postchecks/final receipt.",
            "False runtime or lineage continuity comparison forces failed status."],
        "physical_and_semantic_contract": {"fitting_support_voxels": 1451501, "unknown_abstained_voxels": 8,
            "original_source_intensity_and_brain_unchanged_by_design": True,
            "source_hashes_native_frame_raw_outputs_and_typed_kernel": "unchanged frozen source; prior exact-kernel native preflight remains the execution evidence",
            "posterior_probabilities_are_PVE": False, "anatomy_accepted": False, "host_accepted": False,
            "growth_use_accepted": False, "training_admitted": False},
        "resources": method["resources"],
        "limits": ["Generated process doubles prove supervisor control flow, not successful patient fitting or anatomical validity.",
            "No patient N4/Atropos run or release is performed by this audit.",
            "Two previous actual patient protocol attempts remain failures; zero successful candidates or anatomical acceptance before v3.",
            "Memory and elapsed-time guards are sampled; between-sample peaks are not hard reservations.",
            "Prior un-symbolicated real SIGSEGV causation remains inferred from the independently established mask representation defect."]}
    content = json.dumps(report, indent=2, allow_nan=False)+"\n"
    assert "/Users/" not in content
    with destination.open("x") as stream: stream.write(content)
    print(json.dumps({"artifact": str(destination.relative_to(ROOT)), "sha256": sha(destination),
        "controls_passed": len(CHECKS), "status": report["status"]}))


if __name__ == "__main__": main()
