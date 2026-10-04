"""Mocked comparison orchestration only; never construct or audit a history."""
from copy import deepcopy
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import pytest


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("native_batch_comparison_review",
    ROOT / "scripts/compare_independent_native_batch.py")
runner = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(runner)


def _mock_completed_report(monkeypatch):
    spec = runner.declaration()
    count = spec["expected_microsteps"]
    certificate = {"feasible": True, "failures": [], "fixture": "mock-certificate-only"}
    monkeypatch.setattr(runner, "reference_row", lambda _: {
        "independent_audit": {"certificate": certificate}})
    sources = {"mock_source": "mock-hash"}
    phases = [dict(phase=name, backend="batch" if name == "batch" else "scalar",
        error=None, hooks_restored=True, certificate=deepcopy(certificate),
        active_contact_scans=count, committed_prefixes=count,
        scalar_cell_queries=(6 if name == "batch" else 7) * count,
        batch_cell_queries=count if name == "batch" else 0,
        trace_digest="sha256:" + "a" * 64,
        audit_with_trace_seconds=1., trace_capture_seconds=.1)
        for name in spec["phase_order"]]
    report = dict(status="completed", complete_trace_parity=True, source_unchanged=True,
        sources=sources, declaration_hash=runner.digest(spec), native_strokes=1, phases=phases,
        binding=dict(source_hash=spec["expected_source_hash"], config_hash=spec["expected_config_hash"],
                     history_hash=spec["expected_history_hash"], microsteps=count))
    return report, spec, sources


def test_complete_mock_receipt_acceptance_is_possible(monkeypatch):
    report, spec, sources = _mock_completed_report(monkeypatch)
    assert runner.accepted_report(report, spec, sources)


@pytest.mark.parametrize("fault", ["batch_label_is_scalar", "scalar_used_batch", "batch_has_no_queries", "missing_backend"])
def test_phase_parity_cannot_claim_batch_execution_without_backend_evidence(fault, monkeypatch):
    report, spec, sources = _mock_completed_report(monkeypatch)
    if fault == "batch_label_is_scalar":
        report["phases"][1]["backend"] = "scalar"
    elif fault == "scalar_used_batch":
        report["phases"][0]["batch_cell_queries"] = 1
    elif fault == "batch_has_no_queries":
        report["phases"][1]["batch_cell_queries"] = 0
    else:
        del report["phases"][1]["backend"]
    assert not runner.accepted_report(report, spec, sources)


def test_rss_sampler_has_a_finite_subprocess_deadline(monkeypatch):
    def fake_ps(*args, **kwargs):
        assert 0 < kwargs.get("timeout", 0) <= 2, "Unbounded ps can stall the parent watchdog"
        return SimpleNamespace(returncode=0, stdout="123\n")

    monkeypatch.setattr(runner.subprocess, "run", fake_ps)
    assert runner.sampled_rss(1234) == 123 * 1024


def _mock_launch_inputs(monkeypatch):
    spec = runner.declaration()
    monkeypatch.setattr(runner.platform, "platform", lambda: "mock-platform-no-subprocess")
    monkeypatch.setattr(runner, "declaration", lambda: spec)
    monkeypatch.setattr(runner, "verify_declared_files", lambda _: None)
    monkeypatch.setattr(runner, "source_receipt", lambda _: {"mock_source": "mock-hash"})
    return spec


def test_supervision_exception_kills_and_reaps_worker_and_preserves_failure_receipt(tmp_path, monkeypatch):
    _mock_launch_inputs(monkeypatch)
    state = dict(killed=False, waited=False)

    class FakeWorker:
        pid = 123456

        def __init__(self, *args, **kwargs):
            pass

        def poll(self):
            return -9 if state["killed"] else None

        def wait(self, *args, **kwargs):
            state["waited"] = True
            return -9

    def failed_sampler(_, **kwargs):
        assert 0 < kwargs["timeout_seconds"] <= 1
        raise RuntimeError("deliberate mock sampler failure")

    def kill_group(pid, _):
        assert pid == FakeWorker.pid
        state["killed"] = True

    monkeypatch.setattr(runner.subprocess, "Popen", FakeWorker)
    monkeypatch.setattr(runner, "sampled_rss", failed_sampler)
    monkeypatch.setattr(runner.os, "killpg", kill_group)
    output = tmp_path / "fresh"
    assert runner.launch(output) == 1
    assert state == dict(killed=True, waited=True)
    receipt = json.loads((output / "launcher.json").read_text())
    assert receipt["status"] == "failed_or_incomplete"
    assert receipt["complete_history_parity_verified"] is False
    assert receipt["supervision_error"] == {
        "type": "RuntimeError", "message": "deliberate mock sampler failure"}


def test_disappearing_source_after_worker_exit_still_saves_failed_launcher(tmp_path, monkeypatch):
    _mock_launch_inputs(monkeypatch)
    calls = 0

    def source_receipt(_):
        nonlocal calls
        calls += 1
        if calls > 1:
            raise FileNotFoundError("mock source disappeared during supervision")
        return {"mock_source": "mock-hash"}

    class FinishedWorker:
        pid = 123456

        def __init__(self, *args, **kwargs):
            pass

        def poll(self):
            return 0

        def wait(self, *args, **kwargs):
            return 0

    monkeypatch.setattr(runner, "source_receipt", source_receipt)
    monkeypatch.setattr(runner.subprocess, "Popen", FinishedWorker)
    output = tmp_path / "fresh"
    assert runner.launch(output) == 1
    receipt = json.loads((output / "launcher.json").read_text())
    assert receipt["status"] == "failed_or_incomplete"
    assert receipt["source_unchanged"] is False
    assert receipt["source_error"]["type"] == "FileNotFoundError"


@pytest.mark.parametrize("fault", ["unequal_total", "boolean_count"])
def test_backend_counter_totals_must_be_equal_actual_integers(fault, monkeypatch):
    report, spec, sources = _mock_completed_report(monkeypatch)
    if fault == "unequal_total":
        report["phases"][1]["scalar_cell_queries"] += 1
    else:
        for row in report["phases"]:
            row["scalar_cell_queries"] = True
    assert not runner.accepted_report(report, spec, sources)


def test_worker_final_source_inventory_failure_keeps_original_failure_and_report(tmp_path, monkeypatch):
    _mock_launch_inputs(monkeypatch)
    for key in runner.THREAD_KEYS:
        monkeypatch.setenv(key, "1")
    calls = 0

    def source_receipt(_):
        nonlocal calls
        calls += 1
        if calls > 1:
            raise FileNotFoundError("mock source disappeared before worker finalization")
        return {"mock_source": "mock-hash"}

    def forbidden_runtime():
        raise RuntimeError("mock stops before any runtime or scene construction")

    monkeypatch.setattr(runner, "source_receipt", source_receipt)
    monkeypatch.setattr(runner, "reference_row", lambda _: {})
    monkeypatch.setattr(runner, "load_runtime", forbidden_runtime)
    monkeypatch.setattr(runner, "peak_rss_bytes", lambda: 1)
    assert runner.worker(tmp_path, runner.time.perf_counter() + 1) == 1
    report = json.loads((tmp_path / "report.json").read_text())
    assert report["status"] == "failed_or_incomplete"
    assert report["error"] == {"type": "RuntimeError",
        "message": "mock stops before any runtime or scene construction"}
    assert report["source_error"]["type"] == "FileNotFoundError"
    assert report["source_unchanged"] is False
    assert report["native_strokes"] == 0 and report["phases"] == []


def test_trace_hook_conflict_restores_all_originals_and_refuses_acceptance():
    names = ("_native_active_contacts", "_cell_collision", "_extend_independent_free_space")
    originals = {name: (lambda *args, **kwargs: None) for name in names}
    fake_evaluation = SimpleNamespace(**originals)
    with pytest.raises(RuntimeError, match="restoration conflict"):
        with runner.trace_prefixes(fake_evaluation):
            fake_evaluation._cell_collision = lambda *args, **kwargs: None
    assert runner._TRACING is False
    assert all(getattr(fake_evaluation, name) is original for name, original in originals.items())
