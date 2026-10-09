"""Tiny generated launch and source checks; never loads the model checkpoint."""
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import time
import unittest

from darwin_fast_sampler import FastDarwinSampler
from supervise_pair import (HERE, LAUNCH_GATE, abort_unreleased_gate,
                            bind_launch_identity, emergency_after_closeout_failure,
                            pre_release_inventory, require_accepted_baseline)


class FakeSlow:
    def __init__(self, snapshots):
        self.snapshots = iter(snapshots)
        self.last = None
        self.closed = False

    def status(self):
        self.last = next(self.snapshots, self.last)
        return self.last, 0.0, []

    def close(self):
        self.closed = True


class LaunchTests(unittest.TestCase):
    def launch(self, worker):
        read_fd, write_fd = os.pipe()
        child = subprocess.Popen([sys.executable, "-I", str(LAUNCH_GATE),
                                  str(read_fd), str(worker)],
                                 pass_fds=(read_fd,), start_new_session=True,
                                 stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL,
                                 stderr=subprocess.DEVNULL)
        os.close(read_fd)
        return child, write_fd

    def test_gate_eof_cannot_execute_generated_worker(self):
        with tempfile.TemporaryDirectory() as folder:
            marker = Path(folder) / "executed.txt"
            worker = Path(folder) / "generated_worker.py"
            worker.write_text("from pathlib import Path\nPath(%r).write_text('yes')\n" % str(marker))
            child, token_fd = self.launch(worker)
            try:
                os.close(token_fd)
                self.assertEqual(child.wait(timeout=3), 3)
                self.assertFalse(marker.exists())
            finally:
                if child.poll() is None:
                    child.kill()
                child.wait(timeout=1)

    def test_kernel_identity_binding_precedes_generated_worker_exec(self):
        with tempfile.TemporaryDirectory() as folder:
            marker = Path(folder) / "executed.txt"
            worker = Path(folder) / "generated_worker.py"
            worker.write_text("from pathlib import Path\nPath(%r).write_text('yes')\n" % str(marker))
            child, token_fd = self.launch(worker)
            fast = FastDarwinSampler()
            try:
                identity, pgid = bind_launch_identity(child, fast)
                self.assertIsNotNone(identity)
                self.assertEqual(pgid, child.pid)
                self.assertFalse(marker.exists())
                os.write(token_fd, b"G")
                os.close(token_fd)
                self.assertEqual(child.wait(timeout=3), 0)
                self.assertEqual(marker.read_text(), "yes")
            finally:
                if child.poll() is None:
                    child.kill()
                child.wait(timeout=1)

    def test_unreleased_binding_failure_closes_gate_and_persists_negative(self):
        with tempfile.TemporaryDirectory() as folder:
            marker = Path(folder) / "executed.txt"
            worker = Path(folder) / "generated_worker.py"
            worker.write_text("from pathlib import Path\nPath(%r).write_text('yes')\n" % str(marker))
            child, token_fd = self.launch(worker)
            try:
                result = abort_unreleased_gate(child, token_fd, FastDarwinSampler(),
                                               None, None, Path(folder),
                                               "fixture_binding_failed", "fixture-contract")
                self.assertFalse(result["model_released"])
                self.assertFalse(result["accepted_for_feasibility"])
                self.assertEqual(result["exit_code"], 3)
                self.assertFalse(result["residual_possible"])
                self.assertFalse(marker.exists())
                saved = json.loads((Path(folder) / "unreleased-launch-failure.json").read_text())
                self.assertFalse(saved["model_released"])
            finally:
                if child.poll() is None:
                    child.kill()
                child.wait(timeout=1)

    def test_synchronous_pre_release_inventory_uses_bound_pgid_not_stale_snapshot(self):
        stale_background = {"sample_monotonic": 2.0,
                            "project_process_inventory": [{"blocking_overlap": True, "pgid": 777}]}
        self.assertTrue(stale_background["project_process_inventory"])
        calls = []
        clean = {"sample_monotonic": 3.0, "project_process_inventory": []}
        def collect_now(supervisor_pid, child_pgid, allowlist):
            calls.append((supervisor_pid, child_pgid, allowlist))
            return clean
        self.assertIs(pre_release_inventory(111, 555, None, collect_now), clean)
        self.assertEqual(calls, [(111, 555, None)])
        with self.assertRaises(RuntimeError):
            pre_release_inventory(111, 555, None, lambda *_: stale_background)

    def test_ambiguous_release_emergency_stops_bound_generated_gate(self):
        with tempfile.TemporaryDirectory() as folder:
            worker = Path(folder) / "generated_worker.py"
            worker.write_text("import time\ntime.sleep(30)\n")
            child, token_fd = self.launch(worker)
            fast = FastDarwinSampler()
            identity, pgid = bind_launch_identity(child, fast)
            slow = FakeSlow([])
            try:
                os.write(token_fd, b"G")
                os.close(token_fd)
                result = emergency_after_closeout_failure(
                    child, fast, pgid, identity, slow, Path(folder),
                    RuntimeError("fixture release receipt fault"), "fixture-contract",
                    reason="model_release_attempt_failed_or_ambiguous")
                self.assertEqual(result["exit_code"], -9)
                self.assertFalse(result["accepted_for_feasibility"])
                self.assertTrue(result["manual_attention_required"])
                self.assertTrue(result["residual_possible"])
                self.assertTrue(result["orphan_status_unverified"])
                self.assertTrue(slow.closed)
                saved = json.loads((Path(folder) / "closeout-failure.json").read_text())
                self.assertEqual(saved["reason"], "model_release_attempt_failed_or_ambiguous")
            finally:
                if child.poll() is None:
                    child.kill()
                child.wait(timeout=1)

    def test_emergency_signals_known_detached_child(self):
        with tempfile.TemporaryDirectory() as folder:
            marker = Path(folder) / "descendant.pid"
            worker = Path(folder) / "generated_worker.py"
            worker.write_text(
                "import subprocess,sys,time\n"
                "from pathlib import Path\n"
                "child=subprocess.Popen([sys.executable,'-c','import time;time.sleep(30)'],"
                "start_new_session=True)\n"
                "Path(%r).write_text(str(child.pid))\n" % str(marker) +
                "time.sleep(30)\n")
            root, token_fd = self.launch(worker)
            fast = FastDarwinSampler()
            identity, pgid = bind_launch_identity(root, fast)
            descendant_pid = None
            descendant_identity = None
            slow = FakeSlow([])
            try:
                os.write(token_fd, b"G")
                os.close(token_fd)
                deadline = time.monotonic() + 3
                while not marker.exists() and time.monotonic() < deadline:
                    time.sleep(0.01)
                self.assertTrue(marker.exists())
                descendant_pid = int(marker.read_text())
                descendant_identity = fast.process_start_identity(descendant_pid)
                result = emergency_after_closeout_failure(
                    root, fast, pgid, identity, slow, Path(folder),
                    RuntimeError("fixture closeout failure"), "fixture-contract",
                    detached_seen={descendant_pid: descendant_identity})
                self.assertFalse(result["accepted_for_feasibility"])
                self.assertTrue(result["manual_attention_required"])
                self.assertTrue(any(action.get("pid") == descendant_pid and
                                    action.get("signaled") for action in result["cleanup_actions"]))
                self.assertIn("remaining_group_pids", result)
                self.assertIn("remaining_detached_pids", result)
            finally:
                if root.poll() is None:
                    root.kill()
                root.wait(timeout=1)
                if descendant_pid is not None and descendant_identity is not None:
                    fast.signal_if_same_process(descendant_pid, descendant_identity, 9)

    def test_emergency_records_unbound_same_group_survivor_without_signaling_it(self):
        with tempfile.TemporaryDirectory() as folder:
            marker = Path(folder) / "descendant.pid"
            worker = Path(folder) / "generated_worker.py"
            worker.write_text(
                "import subprocess,sys,time\n"
                "from pathlib import Path\n"
                "child=subprocess.Popen([sys.executable,'-c','import time;time.sleep(30)'],"
                "start_new_session=False)\n"
                "Path(%r).write_text(str(child.pid))\n" % str(marker))
            root, token_fd = self.launch(worker)
            fast = FastDarwinSampler()
            identity, pgid = bind_launch_identity(root, fast)
            descendant_pid = None
            descendant_identity = None
            try:
                os.write(token_fd, b"G")
                os.close(token_fd)
                self.assertEqual(root.wait(timeout=3), 0)
                descendant_pid = int(marker.read_text())
                descendant_identity = fast.process_start_identity(descendant_pid)
                result = emergency_after_closeout_failure(
                    root, fast, pgid, identity, FakeSlow([]), Path(folder),
                    RuntimeError("fixture closeout failure"), "fixture-contract")
                self.assertTrue(result["residual_possible"])
                self.assertIn(descendant_pid, result["remaining_group_pids"])
                self.assertFalse(any(action.get("pid") == descendant_pid and
                                     action.get("signaled") for action in result["cleanup_actions"]))
            finally:
                if root.poll() is None:
                    root.kill()
                root.wait(timeout=1)
                if descendant_pid is not None and descendant_identity is not None:
                    fast.signal_if_same_process(descendant_pid, descendant_identity, 9)

    def test_frozen_runner_sources_and_guard_limits_match_contract(self):
        contract = json.loads((HERE / "pair-contract.json").read_text())
        for name, expected in contract["runner_source_sha256"].items():
            self.assertEqual(hashlib.sha256((HERE / name).read_bytes()).hexdigest(), expected, name)
        self.assertEqual(contract["process_group_rss_cap_kib"], 3145728)
        self.assertEqual(contract["wall_time_cap_seconds"], 120)
        self.assertEqual(contract["host_abort_kernel_pressure_masks"], [2, 4])
        source = (HERE / "supervise_pair.py").read_text()
        self.assertNotIn("killpg(", source)
        self.assertIn("closeout_supervision(", source)

    def test_lazy_arm_needs_accepted_matched_baseline(self):
        with tempfile.TemporaryDirectory() as folder:
            from unittest import mock
            contract = {"model_file_sha256": {"fold_all/checkpoint_final.pth": "pinned"},
                        "input_npy_sha256": "input", "output_shape": [1, 3, 64, 64, 64]}
            with mock.patch("supervise_pair.HERE", Path(folder)):
                with self.assertRaises(FileNotFoundError):
                    require_accepted_baseline(contract, "contract")


if __name__ == "__main__":
    unittest.main()
