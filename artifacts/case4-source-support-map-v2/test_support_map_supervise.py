"""Generated/source-only release controls; never launches a child or reads arrays."""

import importlib.util
import json
import subprocess
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch


HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location("support_map_supervise", HERE / "support_map_supervise.py")
supervisor = importlib.util.module_from_spec(spec)
spec.loader.exec_module(supervisor)


class ReleaseControlTest(unittest.TestCase):
    def setUp(self):
        self.release = json.loads((HERE / "support-map-release.json").read_text())

    def test_exact_frozen_release_is_valid_without_opening_any_patient_file(self):
        supervisor.validate_release(self.release)

    def test_patient_or_executable_substitution_rejects(self):
        for key, replacement in (
            ("generator_path", "build/other-generator.py"),
            ("source_preparation_result_path", "outputs/other-patient.json"),
            ("source_preparation_result_sha256", "0" * 64),
            ("boundary_result_path", "build/other-diagnosis.json"),
            ("boundary_result_sha256", "0" * 64),
            ("ignored_helper_path", "src/resectionlab/scan_support_contract.py"),
            ("tracked_helper_path", "build/other-helper.py"),
            ("isolated_python_path", "/usr/bin/python3"),
            ("output_paths", ["/tmp/rewritten-map.nii.gz"]),
            ("tumor_inference_admitted", True),
            ("planning_admitted", True),
            ("resource_protocol", {**self.release["resource_protocol"],
                                   "maximum_process_group_rss_gib": 12}),
        ):
            with self.subTest(key=key):
                altered = dict(self.release)
                altered[key] = replacement
                with self.assertRaisesRegex(ValueError, "frozen Case4"):
                    supervisor.validate_release(altered)

    def test_ambient_python_packages_are_not_inherited(self):
        environment = supervisor.child_environment({
            "PATH": "/usr/bin", "PYTHONPATH": "/tmp/unpinned",
            "PYTHONHOME": "/tmp/other", "PYTHONINSPECT": "1",
            "ITK_GLOBAL_DEFAULT_NUMBER_OF_THREADS": "77",
        })
        self.assertEqual(environment["PATH"], "/usr/bin")
        self.assertNotIn("PYTHONPATH", environment)
        self.assertNotIn("PYTHONHOME", environment)
        self.assertNotIn("PYTHONINSPECT", environment)
        self.assertEqual(environment["PYTHONNOUSERSITE"], "1")
        self.assertEqual(environment["ITK_GLOBAL_DEFAULT_NUMBER_OF_THREADS"], "2")

    def test_permission_denied_on_identity_bound_stop_records_live_worker(self):
        class LiveProc:
            pid = 4242
            def poll(self):
                return None
            def wait(self, timeout):
                raise subprocess.TimeoutExpired("generated-worker", timeout)

        class DeniedSampler:
            def group(self, pgid, root_pid):
                return {"pids": [4242]}
            def process_start_identity(self, pid):
                return (100, 200)
            def signal_if_same_process(self, pid, identity, sig):
                raise PermissionError("generated identity-bound signal denial")

        with patch.object(supervisor.os, "getpgid", return_value=4242), \
             patch.object(supervisor.os, "killpg") as group_signal:
            cleanup = supervisor.safe_stop_owned_group(
                LiveProc(), DeniedSampler(), {}, {4242: (100, 200)}, (100, 200))
        group_signal.assert_not_called()
        self.assertFalse(cleanup["verification_complete"])
        self.assertTrue(cleanup["leader_still_alive"])
        self.assertEqual(cleanup["unresolved_owned_pids"], [4242])
        self.assertTrue(any(item["error_type"] == "PermissionError"
                            for item in cleanup["errors"]))
        record = {"stop_reason": "host_pressure", "stop_actions": [], "cleanup_errors": [],
                  "cleanup_verification_complete": None,
                  "unresolved_owned_group_pids": None,
                  "worker_liveness_after_cleanup": None}
        supervisor.record_cleanup(record, cleanup)
        self.assertEqual(record["stop_reason"], "host_pressure")
        self.assertEqual(supervisor.result_status(record, 0, map_exists=True,
                                                  result_exists=True, hold_exists=False), "failed")

    def test_permission_denied_after_completed_worker_leaves_unresolved_child(self):
        class CompletedProc:
            pid = 4242
            def poll(self):
                return 0
            def wait(self, timeout):
                return 0

        class ChildSampler:
            def group(self, pgid, root_pid):
                return {"pids": [4243]}
            def process_start_identity(self, pid):
                return (100, 200)
            def signal_if_same_process(self, pid, identity, sig):
                raise AssertionError("unknown child must not be signaled")

        with patch.object(supervisor.os, "getpgid", side_effect=PermissionError("generated denial")), \
             patch.object(supervisor.os, "killpg") as group_signal:
            cleanup = supervisor.safe_stop_owned_group(
                CompletedProc(), ChildSampler(), {}, {4242: (100, 200), 4243: (100, 201)},
                (100, 200))
        group_signal.assert_not_called()
        self.assertFalse(cleanup["verification_complete"])
        self.assertFalse(cleanup["leader_still_alive"])
        self.assertEqual(cleanup["unresolved_owned_pids"], [4243])
        self.assertTrue(any(item["error_type"] == "PermissionError"
                            for item in cleanup["errors"]))
        record = {"stop_reason": "owned_group_or_worker_not_verified_exited",
                  "cleanup_errors": cleanup["errors"],
                  "cleanup_verification_complete": cleanup["verification_complete"],
                  "unresolved_owned_group_pids": cleanup["unresolved_owned_pids"],
                  "worker_liveness_after_cleanup": cleanup["leader_still_alive"]}
        self.assertEqual(supervisor.result_status(record, 0, map_exists=True,
                                                  result_exists=True, hold_exists=False), "failed")

    def test_group_enumeration_denial_never_becomes_completed(self):
        class CompletedProc:
            pid = 4242
            def poll(self):
                return 0
            def wait(self, timeout):
                return 0

        class DeniedSampler:
            def group(self, pgid, root_pid):
                raise PermissionError("generated group enumeration denial")
            def signal_if_same_process(self, pid, identity, sig):
                raise AssertionError("no verified PID exists to signal")

        cleanup = supervisor.safe_stop_owned_group(CompletedProc(), DeniedSampler(), {},
                                                   {4242: (100, 200)}, (100, 200))
        self.assertFalse(cleanup["verification_complete"])
        self.assertIsNone(cleanup["unresolved_owned_pids"])
        self.assertTrue(any(item["error_type"] == "PermissionError"
                            for item in cleanup["errors"]))
        record = {"stop_reason": "final_group_enumeration_failed",
                  "cleanup_errors": cleanup["errors"],
                  "cleanup_verification_complete": cleanup["verification_complete"],
                  "unresolved_owned_group_pids": cleanup["unresolved_owned_pids"],
                  "worker_liveness_after_cleanup": cleanup["leader_still_alive"]}
        self.assertEqual(supervisor.result_status(record, 0, map_exists=True,
                                                  result_exists=True, hold_exists=False), "failed")

    def test_group_enumeration_denial_still_signals_known_live_leader(self):
        class LiveProc:
            pid = 4242
            def poll(self):
                return None
            def wait(self, timeout):
                raise subprocess.TimeoutExpired("generated-worker", timeout)

        class DeniedGroupSampler:
            def __init__(self):
                self.signals = []
            def group(self, pgid, root_pid):
                raise PermissionError("generated group enumeration denial")
            def signal_if_same_process(self, pid, identity, sig):
                self.signals.append((pid, identity, sig.name))
                return {"pid": pid, "identity_matched": True, "signaled": True}

        sampler = DeniedGroupSampler()
        with patch.object(supervisor.os, "getpgid", side_effect=AssertionError("leader PGID unnecessary")), \
             patch.object(supervisor.os, "killpg") as group_signal:
            cleanup = supervisor.safe_stop_owned_group(
                LiveProc(), sampler, {}, {4242: (100, 200)}, (100, 200))
        group_signal.assert_not_called()
        self.assertEqual([item[2] for item in sampler.signals], ["SIGTERM", "SIGKILL"])
        self.assertTrue(all(item[:2] == (4242, (100, 200)) for item in sampler.signals))
        self.assertFalse(cleanup["verification_complete"])
        self.assertIsNone(cleanup["unresolved_owned_pids"])
        self.assertTrue(cleanup["leader_still_alive"])
        self.assertTrue(any(item["error_type"] == "PermissionError"
                            for item in cleanup["errors"]))

    def test_completed_leader_never_signals_new_pid_in_recycled_group(self):
        class CompletedProc:
            pid = 4242
            def poll(self):
                return 0
            def wait(self, timeout):
                return 0

        class NewPidSampler:
            def group(self, pgid, root_pid):
                return {"pids": [4243]}
            def process_start_identity(self, pid):
                raise AssertionError("new PID identity must not establish ownership")
            def signal_if_same_process(self, pid, identity, sig):
                raise AssertionError("unobserved PID must not be signaled")

        with patch.object(supervisor.os, "getpgid", return_value=4242), \
             patch.object(supervisor.os, "killpg") as group_signal:
            cleanup = supervisor.safe_stop_owned_group(
                CompletedProc(), NewPidSampler(), {}, {4242: (100, 200)}, (100, 200))
        group_signal.assert_not_called()
        self.assertFalse(cleanup["verification_complete"])
        self.assertEqual(cleanup["unresolved_owned_pids"], [4243])
        self.assertTrue(any(item["error_type"] == "UnobservedAfterLeaderExit"
                            for item in cleanup["errors"]))

    def test_leader_exit_during_sample_does_not_promote_new_pid(self):
        class FlipProc:
            pid = 4242
            polls = 0
            def poll(self):
                self.polls += 1
                return None if self.polls == 1 else 0

        class ReusedGroupSampler:
            def process_start_identity(self, pid):
                return (100, 200) if pid == 4242 else (900, 901)

        previously_owned = {4242: (100, 200)}
        with self.assertRaisesRegex(RuntimeError, "exited during"):
            supervisor.retain_live_sample_identities(
                FlipProc(), ReusedGroupSampler(), {"pids": [4242, 4243]},
                (100, 200), previously_owned)
        self.assertEqual(previously_owned, {4242: (100, 200)})

    def test_exited_leader_rejects_sample_before_identity_reads(self):
        class ExitedProc:
            pid = 4242
            def poll(self):
                return 0

        class NoIdentitySampler:
            def process_start_identity(self, pid):
                raise AssertionError("no PID identity should be read after leader exit")

        previously_owned = {4242: (100, 200)}
        with self.assertRaisesRegex(RuntimeError, "exited or changed"):
            supervisor.retain_live_sample_identities(
                ExitedProc(), NoIdentitySampler(), {"pids": [4243]},
                (100, 200), previously_owned)
        self.assertEqual(previously_owned, {4242: (100, 200)})

    def test_provisional_negative_is_durable_no_clobber_before_cleanup(self):
        record = {"run_id": "generated", "release_sha256": "a" * 64,
                  "stop_reason": "generated_pressure_stop", "worker_pid": 4242,
                  "worker_start_identity": (100, 200), "samples": [{"generated": True}],
                  "receipt_write_errors": [], "provisional_negative_written": False}
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "provisional.json"
            with patch.object(supervisor, "NEGATIVE", path):
                supervisor.write_provisional_negative(record, "generated_cleanup")
                self.assertTrue(record["provisional_negative_written"])
                saved = json.loads(path.read_text())
                self.assertEqual(saved["stop_reason"], "generated_pressure_stop")
                self.assertEqual(saved["samples_recorded"], 1)
                self.assertEqual(saved["status"], "provisional_negative_cleanup_pending")
                first_bytes = path.read_bytes()
                supervisor.write_provisional_negative(record, "later_cleanup")
                self.assertEqual(path.read_bytes(), first_bytes)
                other = {**record, "provisional_negative_written": False,
                         "receipt_write_errors": []}
                supervisor.write_provisional_negative(other, "cannot_overwrite")
                self.assertEqual(path.read_bytes(), first_bytes)
                self.assertEqual(other["receipt_write_errors"][0]["error_type"],
                                 "FileExistsError")

    def test_provisional_write_denial_is_recorded_for_manual_attention(self):
        record = {"run_id": "generated", "release_sha256": "a" * 64,
                  "stop_reason": "generated_pressure_stop", "samples": [],
                  "receipt_write_errors": [], "provisional_negative_written": False}
        with patch.object(supervisor.os, "open", side_effect=PermissionError("generated denial")):
            supervisor.write_provisional_negative(record, "generated_cleanup")
        self.assertFalse(record["provisional_negative_written"])
        self.assertEqual(record["receipt_write_errors"][0]["error_type"], "PermissionError")

    def test_completed_status_needs_result_and_verified_cleanup(self):
        clean = {"stop_reason": None, "cleanup_errors": [],
                 "receipt_write_errors": [], "finalization_errors": [],
                 "cleanup_verification_complete": True,
                 "unresolved_owned_group_pids": [],
                 "worker_liveness_after_cleanup": False}
        self.assertEqual(supervisor.result_status(clean, 0, map_exists=True,
                                                  result_exists=True, hold_exists=False),
                         "completed_source_support_only_qc_pending")
        self.assertEqual(supervisor.result_status(clean, 0, map_exists=True,
                                                  result_exists=False, hold_exists=False),
                         "failed")
        uncertain = {**clean, "finalization_errors": [{"error_type": "PermissionError"}]}
        self.assertEqual(supervisor.result_status(uncertain, 0, map_exists=True,
                                                  result_exists=True, hold_exists=False),
                         "failed")


if __name__ == "__main__":
    unittest.main()
