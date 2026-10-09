"""Generated fake-process controls; never starts validation or a native child."""
from __future__ import annotations

from contextlib import redirect_stdout
import importlib.util
import io
from pathlib import Path
import subprocess
import tempfile
import unittest
from unittest.mock import patch

SOURCE = Path(__file__).with_name("diagnose.py")
SPEC = importlib.util.spec_from_file_location("hbe_preflight_memory_diagnostic", SOURCE)
assert SPEC is not None and SPEC.loader is not None
diagnostic = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(diagnostic)


class FakeProcess:
    pid = 246810

    def __init__(self, *, running: bool, direct_kill_error: Exception | None = None):
        self.running = running
        self.direct_kill_error = direct_kill_error
        self.kill_calls = 0
        self.wait_calls = 0

    def poll(self):
        return None if self.running else 0

    def kill(self):
        self.kill_calls += 1
        if self.direct_kill_error:
            raise self.direct_kill_error
        self.running = False

    def wait(self, timeout):
        self.wait_calls += 1
        if self.running:
            raise subprocess.TimeoutExpired("fake", timeout)
        return 0


class GeneratedCleanupTests(unittest.TestCase):
    def owner(self, process):
        owner = diagnostic.OwnedWorker()
        owner.process = process
        return owner

    def observer(self, *responses):
        sequence = iter(responses)

        def read(pid, *, timeout_seconds):
            assert pid == FakeProcess.pid and timeout_seconds > 0
            value = next(sequence)
            if isinstance(value, Exception):
                raise value
            return value

        return read

    def test_normal_completed_child_reaped_without_fallback(self):
        child = FakeProcess(running=False)
        with patch.object(diagnostic.os, "killpg") as signal_group:
            result = self.owner(child).cleanup(
                self.observer((0, []), (0, [])), diagnostic.time.monotonic() + 5)
        self.assertTrue(result["contained"])
        self.assertTrue(result["direct_child_reaped"])
        self.assertFalse(result["fallback_used"])
        self.assertEqual(result["errors"], [])
        self.assertEqual(child.wait_calls, 1)
        signal_group.assert_not_called()

    def test_killpg_permission_error_still_kills_and_reaps_owned_child(self):
        child = FakeProcess(running=True)
        observer = self.observer((1024, [{"pid": child.pid}]), (0, []))
        with patch.object(diagnostic.os, "killpg", side_effect=PermissionError), \
                patch.object(diagnostic.os, "kill") as signal_cached_pid:
            result = self.owner(child).cleanup(observer, diagnostic.time.monotonic() + 5)
        self.assertTrue(result["contained"])
        self.assertTrue(result["direct_child_reaped"])
        self.assertTrue(result["fallback_used"])
        self.assertIn("killpg:PermissionError", result["errors"])
        self.assertEqual(child.kill_calls, 1)
        self.assertEqual(child.wait_calls, 1)
        signal_cached_pid.assert_not_called()

    def test_observer_failure_uses_owned_child_fallback(self):
        child = FakeProcess(running=True)
        observer = self.observer(RuntimeError("unavailable"), (0, []))
        with patch.object(diagnostic.os, "killpg", side_effect=PermissionError):
            result = self.owner(child).cleanup(observer, diagnostic.time.monotonic() + 5)
        self.assertTrue(result["contained"])
        self.assertIn("initial_observer:RuntimeError", result["errors"])
        self.assertEqual(child.kill_calls, 1)
        self.assertTrue(result["direct_child_reaped"])

    def test_surviving_child_and_group_are_recorded_without_pid_signal(self):
        child = FakeProcess(running=True, direct_kill_error=PermissionError())
        members = [{"pid": child.pid}, {"pid": 13579}]
        observer = self.observer((2048, members), (2048, members))
        with patch.object(diagnostic.os, "killpg", side_effect=PermissionError), \
                patch.object(diagnostic.os, "kill") as signal_cached_pid:
            result = self.owner(child).cleanup(observer, diagnostic.time.monotonic() + 5)
        self.assertFalse(result["contained"])
        self.assertFalse(result["direct_child_reaped"])
        self.assertEqual(result["remaining_members"], members)
        self.assertIn("direct_kill:PermissionError", result["errors"])
        self.assertIn("reap:TimeoutExpired", result["errors"])
        signal_cached_pid.assert_not_called()

    def test_failed_stage_has_terminal_receipt_and_one_use_refusal(self):
        from scripts import mechanics_hbe_v5_remaining_one_shot as remaining

        selected = "e7ee678e27205ddbdcc26bc833861559942a5064"
        with tempfile.TemporaryDirectory() as temporary:
            target = Path(temporary) / "attempt-01"
            with patch.object(diagnostic, "OUT", target), \
                    patch.object(diagnostic, "require_selected_head", return_value=selected), \
                    patch.object(remaining, "supervise_stage", return_value={
                        "status": "failed_or_incomplete"}) as stage, \
                    redirect_stdout(io.StringIO()):
                self.assertEqual(diagnostic.supervisor(selected), 1)
                saved = diagnostic.json.loads((target / "receipt.json").read_text())
                self.assertEqual(saved["status"], "failed_or_incomplete")
                self.assertEqual(saved["native_calls"], 0)
                self.assertEqual(saved["hbe_readout_calls"], 0)
                self.assertFalse(saved["release_written"])
                self.assertFalse(saved["hbe_output_reserved"])
                self.assertEqual(stage.call_args.kwargs["wall_cap"], 147)
                self.assertEqual(stage.call_args.kwargs["rss_cap"], 3 * 1024**3)
                self.assertEqual(stage.call_args.kwargs["output_cap"], 4 * 1024**2)
                with self.assertRaisesRegex(RuntimeError, "already exists"):
                    diagnostic.supervisor(selected)
                self.assertEqual(diagnostic.json.loads((target / "receipt.json").read_text()),
                                 saved)

    def test_worker_exception_saves_terminal_result_without_validation(self):
        with tempfile.TemporaryDirectory() as temporary:
            target = Path(temporary) / "attempt-01"
            target.mkdir()
            with patch.object(diagnostic, "OUT", target), \
                    patch.object(diagnostic, "load_host_sampler",
                                 side_effect=RuntimeError("fake host sampler failure")):
                with self.assertRaisesRegex(RuntimeError, "fake host sampler failure"):
                    diagnostic.worker_terminal("0" * 40)
            saved = diagnostic.json.loads((target / "worker-result.json").read_text())
            self.assertEqual(saved["status"], "failed_or_incomplete")
            self.assertEqual(saved["error_type"], "RuntimeError")
            self.assertEqual(saved["native_calls"], 0)
            self.assertFalse(saved["release_written"])


if __name__ == "__main__":
    unittest.main()
