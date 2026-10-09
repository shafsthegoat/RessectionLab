"""Tiny generated controls; never opens actual predecessor outputs or FEBio."""
from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import tempfile
import time
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from scripts import febio_runtime
from scripts import mechanics_hbe_v5_remaining_one_shot as remaining

HERE = Path(__file__).parent
spec = importlib.util.spec_from_file_location("generated_probe50", HERE / "probe.py")
assert spec is not None and spec.loader is not None
probe = importlib.util.module_from_spec(spec)
spec.loader.exec_module(probe)


class Host:
    def __init__(self, states):
        self.states = iter(states)

    def host(self):
        available, mask = next(self.states)
        return {"available_percent": available, "kernel_pressure_mask": mask,
                "swap_used_bytes": 123}


class ProbeControls(unittest.TestCase):
    def test_series_observer_aborts_and_persists_below_45(self):
        with tempfile.TemporaryDirectory() as temporary:
            target = Path(temporary)
            series = probe.HostSeries(Host([(50, 1), (44, 1)]), target,
                                      time.monotonic())
            self.assertEqual(series.sample("before")["available_percent"], 50)
            with self.assertRaisesRegex(RuntimeError, "below required 45%"):
                series.observer(lambda pgid, *, timeout_seconds: (1024, [{"pid": 7}]),
                                7, timeout_seconds=.2)
            saved = json.loads((target / "host-time-series.json").read_text())
            self.assertEqual([x["available_percent"] for x in saved["samples"]],
                             [50, 44])
            self.assertEqual(saved["breach"]["available_percent"], 44)

    def test_series_aborts_on_non_normal_mask_even_above_45(self):
        with tempfile.TemporaryDirectory() as temporary:
            series = probe.HostSeries(Host([(50, 1), (49, 2)]), Path(temporary),
                                      time.monotonic())
            series.sample("before")
            with self.assertRaisesRegex(RuntimeError, "below required 45%/normal"):
                series.observer(lambda pgid, *, timeout_seconds: (0, []),
                                7, timeout_seconds=.2)
            self.assertEqual(series.breach["kernel_pressure_mask"], 2)

    def test_healthy_series_is_durable_every_tenth_observation(self):
        with tempfile.TemporaryDirectory() as temporary:
            target = Path(temporary)
            series = probe.HostSeries(Host([(50, 1)] * 11), target,
                                      time.monotonic())
            series.sample("before")
            for _ in range(9):
                series.observer(lambda pgid, *, timeout_seconds: (0, []),
                                7, timeout_seconds=.2)
            saved = json.loads((target / "host-time-series.json").read_text())
            self.assertEqual(len(saved["samples"]), 10)
            self.assertIsNone(saved["breach"])

    def test_initial_below_50_does_not_reserve_probe_target(self):
        with tempfile.TemporaryDirectory() as temporary:
            target = Path(temporary) / "attempt-01"
            diagnostic = SimpleNamespace(load_host_sampler=lambda: Host([(49, 1)]))
            compare = SimpleNamespace(selected_head=lambda commit: commit)
            with patch.object(probe, "TARGET", target), \
                    patch.object(probe, "load_dependencies", return_value=(compare, diagnostic, None)), \
                    patch.object(probe, "selected_sources", return_value={}), \
                    patch.object(probe, "self_sha", return_value="0" * 64):
                with self.assertRaisesRegex(RuntimeError, "below 50%/normal"):
                    probe.supervisor("a" * 40)
            self.assertFalse(target.exists())

    def test_post_reservation_host_drop_writes_failure_without_worker(self):
        with tempfile.TemporaryDirectory() as temporary:
            target = Path(temporary) / "attempt-01"
            diagnostic = SimpleNamespace(
                load_host_sampler=lambda: Host([(50, 1), (49, 1)]),
                THREAD_ENV={}, WALL_CAP=150, WORKER_WALL_CAP=147,
                RSS_CAP=3 * 1024**3, OUTPUT_CAP=4 * 1024**2)
            compare = SimpleNamespace(selected_head=lambda commit: commit)
            with patch.object(probe, "TARGET", target), \
                    patch.object(probe, "load_dependencies", return_value=(compare, diagnostic, None)), \
                    patch.object(probe, "selected_sources", return_value={}), \
                    patch.object(probe, "self_sha", return_value="0" * 64):
                self.assertEqual(probe.supervisor("a" * 40), 1)
            saved = json.loads((target / "receipt.json").read_text())
            self.assertEqual(saved["status"], "failed_or_incomplete")
            self.assertEqual(saved["host_breach"]["available_percent"], 49)
            self.assertEqual(saved["native_calls"], 0)
            self.assertFalse(saved["hbe_output_reserved"])
            self.assertTrue((target / "host-time-series.json").is_file())

    def test_worker_snapshot_guard_stops_non_normal_host(self):
        with tempfile.TemporaryDirectory() as temporary:
            target = Path(temporary) / "attempt-01"

            def original_snapshot(label, samples, *, started, host_sampler):
                samples.append({"direct_kernel_host": {"available_percent": 50,
                                                      "kernel_pressure_mask": 2}})

            diagnostic = SimpleNamespace(snapshot=original_snapshot)

            def old_worker(mode, source_commit, diagnostic_arg, hint):
                diagnostic_arg.snapshot("generated", [], started=time.monotonic(),
                                        host_sampler=None)
                return 0

            compare = SimpleNamespace(selected_head=lambda commit: commit,
                                      ARMS={}, run_worker_arm=old_worker)
            with patch.object(probe, "TARGET", target), \
                    patch.object(probe, "load_dependencies", return_value=(compare, diagnostic, None)), \
                    patch.object(probe, "self_sha", return_value="0" * 64):
                with self.assertRaisesRegex(RuntimeError, "crossed 45%/normal"):
                    probe.worker("a" * 40, "0" * 64)
            self.assertIs(diagnostic.snapshot, original_snapshot)

    def test_worker_first_snapshot_requires_real_50_start(self):
        def original_snapshot(label, samples, *, started, host_sampler):
            samples.append({"direct_kernel_host": {"available_percent": 49,
                                                  "kernel_pressure_mask": 1}})

        diagnostic = SimpleNamespace(snapshot=original_snapshot)

        def old_worker(mode, source_commit, diagnostic_arg, hint):
            diagnostic_arg.snapshot("stdlib_host_before_hbe_import", [],
                                    started=time.monotonic(), host_sampler=None)
            return 0

        compare = SimpleNamespace(selected_head=lambda commit: commit,
                                  ARMS={}, run_worker_arm=old_worker)
        with patch.object(probe, "load_dependencies", return_value=(compare, diagnostic, None)), \
                patch.object(probe, "self_sha", return_value="0" * 64):
            with self.assertRaisesRegex(RuntimeError, "crossed 50%/normal"):
                probe.worker("a" * 40, "0" * 64)
        self.assertIs(diagnostic.snapshot, original_snapshot)

    def _complete_fake_supervisor(self, *, tamper=None, drift=False,
                                  active_output=None):
        with tempfile.TemporaryDirectory() as temporary:
            target = Path(temporary) / "attempt-01"
            commit = "a" * 40
            closure = {"old_source_hashes": {"source": "b" * 64},
                       "run_id": "tension:N12:S60:reference",
                       "expected_adapted_deck_sha256": "c" * 64,
                       "adapter_receipt": {"generated": "adapter"}}
            diagnostic = SimpleNamespace(
                load_host_sampler=lambda: Host([(50, 1)] * 4),
                THREAD_ENV={}, WALL_CAP=150, WORKER_WALL_CAP=147,
                RSS_CAP=3 * 1024**3, OUTPUT_CAP=4 * 1024**2,
                OwnedWorker=lambda: SimpleNamespace(cleanup=lambda *args: {
                    "contained": True, "direct_child_reaped": True,
                    "fallback_used": False, "errors": [],
                    "remaining_members": []}))
            compare = SimpleNamespace(selected_head=lambda value: value)

            def fake_supervise(stage, command, directory, receipt, *,
                               rss_observer, **kwargs):
                self.assertEqual(stage, "readout")
                rss_observer(7, timeout_seconds=.2)
                receipt["readout_calls_attempted"] = 1
                receipt["readout_stage"] = {"status": "completed_within_caps"}
                worker = {"status": "read_only_validation_memory_diagnostic_complete",
                          "source_commit": commit, "native_calls": 0,
                          "release_written": False, "target_reserved": False,
                          "adapted_deck_sha256": "c" * 64,
                          "expected_adapted_deck_sha256": "c" * 64}
                identity = {"index": 8, "run_id": closure["run_id"],
                            "deck_sha256": "c" * 64,
                            "source_hashes": closure["old_source_hashes"],
                            "adapter_receipt": closure["adapter_receipt"],
                            "observed_head_preflight": commit}
                hints = {"eligible_files": 14, "hinted_file_opens": 14,
                         "eligible_bytes": 2801621755}
                if tamper == "identity":
                    identity["source_hashes"] = {}
                elif tamper == "missing_deck":
                    worker.pop("adapted_deck_sha256")
                    worker.pop("expected_adapted_deck_sha256")
                    identity.pop("deck_sha256")
                for name, value in (("worker-result.json", worker),
                                    ("validation-identity.json", identity),
                                    ("hint-audit.json", hints)):
                    (target / name).write_text(json.dumps(value))
                return {"status": "completed_within_caps"}

            source_hashes = ["0" * 64, "1" * 64] if drift else ["0" * 64] * 2
            with patch.object(probe, "TARGET", target), \
                    patch.object(probe, "load_dependencies", return_value=(compare, diagnostic, None)), \
                    patch.object(probe, "selected_sources", return_value=closure), \
                    patch.object(probe, "self_sha", side_effect=source_hashes), \
                    patch.object(remaining, "supervise_stage", side_effect=fake_supervise), \
                    patch.object(remaining, "active_bytes", return_value=(
                        active_output if active_output is not None else 1000)), \
                    patch.object(febio_runtime, "process_group_rss", return_value=(0, [])):
                exit_code = probe.supervisor(commit)
            saved = json.loads((target / "receipt.json").read_text())
            series = json.loads((target / "host-time-series.json").read_text())
            return exit_code, saved, series

    def test_fake_complete_supervisor_success_binds_identity_and_host_series(self):
        code, receipt, series = self._complete_fake_supervisor()
        self.assertEqual(code, 0)
        self.assertEqual(receipt["status"], "completed_read_only_50pct_probe")
        self.assertEqual(receipt["host_sample_count"], 3)
        self.assertEqual(len(series["samples"]), 3)
        self.assertEqual(receipt["probe_source_sha256"], "0" * 64)
        self.assertEqual(receipt["probe_source_sha256_postguard"], "0" * 64)
        self.assertEqual(receipt["native_calls"], 0)
        self.assertFalse(receipt["hbe_output_reserved"])

    def test_fake_supervisor_rejects_wrong_identity_and_missing_deck(self):
        for tamper in ("identity", "missing_deck"):
            with self.subTest(tamper=tamper):
                code, receipt, _ = self._complete_fake_supervisor(tamper=tamper)
                self.assertEqual(code, 1)
                self.assertEqual(receipt["status"], "failed_or_incomplete")

    def test_fake_supervisor_rejects_wrapper_drift_and_final_output_cap(self):
        code, receipt, _ = self._complete_fake_supervisor(drift=True)
        self.assertEqual(code, 1)
        self.assertEqual(receipt["status"], "failed_or_incomplete")
        code, receipt, _ = self._complete_fake_supervisor(active_output=4 * 1024**2)
        self.assertEqual(code, 1)
        self.assertEqual(receipt["status"], "failed_final_output_cap")


if __name__ == "__main__":
    unittest.main()
