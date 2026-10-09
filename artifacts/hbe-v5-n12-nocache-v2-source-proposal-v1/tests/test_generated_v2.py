"""Generated software controls only; no actual HBE/native/output replay."""
from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path
import shutil
import subprocess
import time
from types import SimpleNamespace
import tempfile
import unittest
from unittest.mock import patch

from scripts import mechanics_hbe_v5_n8_one_shot as io
from scripts import mechanics_hbe_v5_remaining_one_shot as actual_remaining

HERE = Path(__file__).parent
spec = importlib.util.spec_from_file_location("candidate_nocache_launcher", HERE / "hbe_v5_nocache_v2.py")
assert spec is not None and spec.loader is not None
launcher = importlib.util.module_from_spec(spec)
spec.loader.exec_module(launcher)
spec = importlib.util.spec_from_file_location("candidate_nocache_hint", HERE / "hbe_v5_nocache_hash_v1.py")
assert spec is not None and spec.loader is not None
real_hint = importlib.util.module_from_spec(spec)
spec.loader.exec_module(real_hint)


class TinyHint:
    @staticmethod
    def hinted_hash(path, *, maximum, original_hash, allowed_directories, audit):
        return real_hint.hinted_hash(
            path, maximum=maximum, original_hash=original_hash,
            allowed_directories=allowed_directories, audit=audit,
            minimum_bytes=1)


class FakeProcess:
    pid = 246810

    def __init__(self, *, running: bool, direct_kill_error=None):
        self.running = running
        self.direct_kill_error = direct_kill_error
        self.kill_calls = 0
        self.wait_calls = 0

    def poll(self):
        return None if self.running else 0

    def kill(self):
        self.kill_calls += 1
        if self.direct_kill_error is not None:
            raise self.direct_kill_error
        self.running = False

    def wait(self, timeout):
        self.wait_calls += 1
        if self.running:
            raise subprocess.TimeoutExpired("generated", timeout)
        return 0


class GeneratedLauncherTests(unittest.TestCase):
    def test_durable_native_receipt_must_equal_returned_receipt_and_remain_stable(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary).resolve()
            path = root / launcher.NATIVE_OUTPUT / "receipt.json"
            path.parent.mkdir(parents=True)
            returned = {"status": "passed_numerical_software_only",
                        "native_calls_attempted": 1}
            path.write_text(json.dumps(returned))
            record = {}
            saved = launcher._bind_native_receipt(root, record)
            launcher._require_matching_native_receipt(saved, returned)
            with self.assertRaisesRegex(ValueError, "returned receipt differs"):
                launcher._require_matching_native_receipt(
                    saved, dict(returned, native_calls_attempted=0))
            path.write_text(json.dumps(dict(returned, status="failed_or_incomplete")))
            with self.assertRaisesRegex(ValueError, "changed or disappeared"):
                launcher._rebind_native_receipt_stably(root, record)
            self.assertIn("native_receipt_sha256_first_binding", record)
            path.unlink()
            with self.assertRaisesRegex(ValueError, "changed or disappeared"):
                launcher._rebind_native_receipt_stably(root, record)

    def test_total_preparation_charge_includes_wrapper_time_and_sidecar(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary).resolve()
            old_output = root / launcher.NATIVE_OUTPUT
            old_output.mkdir(parents=True)
            record = {"old_validation_identity": {"previous": {"prep_seconds": 3}}}
            observed = {}

            def old_aggregate(previous, native, readout, prep, output, calls):
                observed.update(native=native, readout=readout, prep=prep,
                                output=output, calls=calls)
                return {"aggregate_prep_wall_seconds": previous["prep_seconds"] + prep}

            fake = SimpleNamespace(
                PREP_WALL=150,
                caps=lambda index: {"active_output_bytes": 8 * 1024**2},
                active_bytes=lambda directory, cap: 1234,
                _check_aggregate=old_aggregate)
            native = {"native_stage": {"elapsed_seconds": 1.5},
                      "readout_stage": {"elapsed_seconds": 0.5},
                      "native_calls_attempted": 1}
            started = time.monotonic() - 3.0
            launcher._charge_full_preparation(fake, root, record, started, native)
            self.assertGreaterEqual(observed["prep"], 1.0)
            self.assertEqual(observed["output"], 1234 + launcher.POLICY["sidecar_output_cap_bytes"])
            self.assertEqual(observed["calls"], 1)
            self.assertEqual(record["extension_resource_charge"]["charged_current_output_bytes"],
                             observed["output"])
            fake.PREP_WALL = 0.1
            with self.assertRaisesRegex(ValueError, "preparation wall cap"):
                launcher._charge_full_preparation(fake, root, record, started, native)

    def test_inclusive_pre_native_gate_rejects_before_reservation_and_supervision(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary).resolve()
            _, remaining, state, policy = self._fixture(root)
            remaining.PREP_WALL = 150
            remaining.caps = lambda index: {"active_output_bytes": 8 * 1024**2}
            remaining.active_bytes = lambda directory, cap: 0
            ledger_calls = []
            remaining._check_aggregate = lambda previous, native, readout, prep, output, calls: (
                ledger_calls.append((prep, output, calls)) or {})
            record = {"old_validation_identity": {"previous": {}}}
            started = [time.monotonic() - 149.5]

            def preflight(context, audit):
                launcher._check_pre_native_charge(
                    remaining, root, record, started[0], "reservation")

            def old_execute():
                remaining.validate_release({}, root=root)
                state["native_started"] = True

            with patch.dict(launcher.POLICY, policy):
                with self.assertRaisesRegex(ValueError, "before native reservation"):
                    launcher.scoped_execute(remaining, TinyHint, [root],
                                            old_execute, preflight)
            self.assertFalse(state["native_started"])
            self.assertEqual(ledger_calls, [])

            # The early check can pass, then elapsed combined preparation can
            # still forbid the old supervisor before a native child is made.
            started[0] = time.monotonic() - 1
            original_supervise = lambda stage, *args, **kwargs: state.update(
                native_started=True)
            remaining.supervise_stage = original_supervise

            def at_stage():
                started[0] = time.monotonic() - 149.5
                launcher._check_pre_native_charge(
                    remaining, root, record, started[0], "supervision")

            def execute_with_supervision():
                remaining.validate_release({}, root=root)
                remaining.supervise_stage("native", [], None, {})

            with patch.dict(launcher.POLICY, policy):
                with self.assertRaisesRegex(ValueError, "before native supervision"):
                    launcher.scoped_execute(
                        remaining, TinyHint, [root], execute_with_supervision,
                        preflight, native_stage_check=at_stage)
            self.assertFalse(state["native_started"])
            self.assertIs(remaining.supervise_stage, original_supervise)
            self.assertEqual(record["pre_native_resource_checks"][0]["phase"],
                             "reservation")
            self.assertEqual(ledger_calls[0][2], 1)
            self.assertEqual(ledger_calls[0][1], launcher.POLICY["sidecar_output_cap_bytes"])

    def test_final_sidecar_write_must_fit_prior_charged_reserve(self):
        start = time.monotonic() - 4
        record = {"extension_resource_charge": {
            "native_stage_seconds": 0, "readout_stage_seconds": 0,
            "launcher_inclusive_prep_seconds_with_finalization_reserve": 3,
            "per_row_prep_wall_seconds": 150}}
        with self.assertRaisesRegex(ValueError, "Final sidecar write exceeded"):
            launcher._actual_final_preparation(record, start)
        record["extension_resource_charge"]["launcher_inclusive_prep_seconds_with_finalization_reserve"] = 5
        self.assertLess(launcher._actual_final_preparation(record, start), 5)

    def test_owned_stage_reaps_normal_completed_child(self):
        process = FakeProcess(running=False)
        owner = launcher.OwnedStage()
        owner.process = process
        observer = lambda *args, **kwargs: (0, [])
        with patch.object(launcher.os, "killpg") as group_kill:
            cleanup = owner.cleanup(observer, time.monotonic() + 5)
        self.assertTrue(cleanup["contained"])
        self.assertTrue(cleanup["direct_child_reaped"])
        self.assertFalse(cleanup["fallback_used"])
        self.assertEqual(process.wait_calls, 1)
        group_kill.assert_not_called()

    def test_owned_stage_permission_error_still_kills_and_reaps_exact_child(self):
        process = FakeProcess(running=True)
        owner = launcher.OwnedStage()
        owner.process = process
        readings = iter([(1024, [{"pid": process.pid}]), (0, [])])
        observer = lambda *args, **kwargs: next(readings)
        with patch.object(launcher.os, "killpg", side_effect=PermissionError), \
                patch.object(launcher.os, "kill") as cached_pid_kill:
            cleanup = owner.cleanup(observer, time.monotonic() + 5)
        self.assertTrue(cleanup["contained"])
        self.assertTrue(cleanup["direct_child_reaped"])
        self.assertTrue(cleanup["fallback_used"])
        self.assertIn("killpg:PermissionError", cleanup["errors"])
        self.assertEqual(process.kill_calls, 1)
        self.assertEqual(process.wait_calls, 1)
        cached_pid_kill.assert_not_called()

    def test_owned_stage_survivor_is_unresolved_and_never_pid_signaled(self):
        process = FakeProcess(running=True, direct_kill_error=PermissionError())
        owner = launcher.OwnedStage()
        owner.process = process
        members = [{"pid": process.pid}, {"pid": 13579}]
        observer = lambda *args, **kwargs: (1024, members)
        with patch.object(launcher.os, "killpg", side_effect=PermissionError), \
                patch.object(launcher.os, "kill") as cached_pid_kill:
            cleanup = owner.cleanup(observer, time.monotonic() + 5)
        self.assertFalse(cleanup["contained"])
        self.assertFalse(cleanup["direct_child_reaped"])
        self.assertEqual(cleanup["remaining_members"], members)
        self.assertIn("direct_kill:PermissionError", cleanup["errors"])
        self.assertIn("reap:TimeoutExpired", cleanup["errors"])
        cached_pid_kill.assert_not_called()

    def test_exact_versioned_envelope_binds_committed_sources_and_old_release(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary).resolve()
            target = root / "launchers"
            target.mkdir()
            for relative in launcher.SOURCES:
                shutil.copy2(HERE / Path(relative).name, root / relative)
            subprocess.run(["/usr/bin/git", "init", "-q"], cwd=root, check=True)
            subprocess.run(["/usr/bin/git", "add", "launchers"], cwd=root, check=True)
            subprocess.run([
                "/usr/bin/git", "-c", "user.name=skamal23",
                "-c", "user.email=sayemkamal12@gmail.com", "commit", "-qm",
                "Generated source-binding fixture", "-m",
                "Co-authored-by: shafsthegoat <shafrir.p@gmail.com>"
            ], cwd=root, check=True)
            commit = subprocess.run(["/usr/bin/git", "rev-parse", "HEAD"],
                                    cwd=root, check=True, capture_output=True,
                                    text=True).stdout.strip()
            bindings = {relative: hashlib.sha256((root / relative).read_bytes()).hexdigest()
                        for relative in launcher.SOURCES}
            release_dir = root / "build"
            release_dir.mkdir()
            inner = {"ordinal": 8, "run_id": "tension:N12:S60:reference",
                     "status": "root_released_one_native_call", "source_commit": commit}
            inner_path = release_dir / "old-release.json"
            inner_path.write_text(json.dumps(inner))
            outer = {"schema": "hbe-v5-nocache-launch-envelope-v2",
                     "status": "root_released_one_native_call_with_declared_io_policy_v2",
                     "source_commit": commit, "extension_source_bindings": bindings,
                     "inner_release": {"path": "build/old-release.json",
                                       "sha256": hashlib.sha256(inner_path.read_bytes()).hexdigest()},
                     "sidecar_directory": launcher.SIDECAR,
                     "policy": launcher.POLICY}
            envelope_path = release_dir / "extension-envelope.json"
            envelope_path.write_text(json.dumps(outer))
            spec = importlib.util.spec_from_file_location(
                "generated_tracked_launcher", root / launcher.SOURCES[0])
            assert spec is not None and spec.loader is not None
            copied = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(copied)
            parsed = copied._read_exact_release(root, envelope_path)
            self.assertEqual(parsed[0], outer)
            self.assertEqual(parsed[3]["source_hashes"], bindings)
            self.assertEqual(parsed[4], inner_path)
            (root / launcher.SOURCES[1]).write_bytes(b"generated tamper")
            with self.assertRaisesRegex(ValueError, "hash differs"):
                copied._read_exact_release(root, envelope_path)

    def test_helper_import_compiles_exact_verified_source_bytes(self):
        source = HERE / "hbe_v5_nocache_hash_v1.py"
        expected = hashlib.sha256(source.read_bytes()).hexdigest()
        module = launcher._load_sibling("generated_bound_hint", source.name, expected)
        self.assertTrue(callable(module.hinted_hash))
        with self.assertRaisesRegex(RuntimeError, "changed before import"):
            launcher._load_sibling("generated_wrong_hint", source.name, "0" * 64)

    def _fixture(self, directory):
        files = [directory / f"tiny-{i}.bin" for i in range(3)]
        for i, file in enumerate(files):
            file.write_bytes(bytes([i + 1]) * (20 + i))
        state = {"native_started": False, "preflight_seen": False,
                 "after_preflight_hash_restored": False}
        remaining = SimpleNamespace(io=SimpleNamespace(file_hash=io.file_hash))

        def source_chain(*args, **kwargs):
            if args[1] == 2:
                return [remaining.io.file_hash(files[1], maximum=4096)]
            first = remaining.io.file_hash(files[0], maximum=4096)
            middle = remaining.validate_prior_chain([], 2)
            last = remaining.io.file_hash(files[2], maximum=4096)
            return [first, *middle, last]

        def source_validate(*args, **kwargs):
            digests = remaining.validate_prior_chain([], 1)
            return {"run_id": "tension:N12:S60:reference", "digests": digests}

        remaining.validate_prior_chain = source_chain
        remaining.validate_release = source_validate
        expected_bytes = sum(file.stat().st_size for file in files)
        policy = {"expected_preflight_hint_opens": 3,
                  "expected_preflight_hint_bytes": expected_bytes,
                  "expected_completed_hint_opens": 6,
                  "expected_completed_hint_bytes": 2 * expected_bytes}
        return files, remaining, state, policy

    def test_full_stub_execution_hints_preflight_and_post_native_recheck(self):
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            files, remaining, state, policy = self._fixture(directory)
            original_chain = remaining.validate_prior_chain
            original_validate = remaining.validate_release
            original_hash = remaining.io.file_hash

            def before_native(context, audit):
                state["preflight_seen"] = True
                self.assertIs(remaining.io.file_hash, original_hash)
                self.assertEqual(audit["hinted_file_opens"], 3)

            def old_execute():
                context = remaining.validate_release({}, root=directory)
                self.assertEqual(len(context["digests"]), 3)
                self.assertIs(remaining.validate_release, original_validate)
                state["after_preflight_hash_restored"] = remaining.io.file_hash is original_hash
                state["native_started"] = True  # No solver is actually called.
                remaining.validate_prior_chain([], 1)  # Frozen post-run recheck.
                self.assertIs(remaining.io.file_hash, original_hash)
                return {"status": "passed_numerical_software_only"}

            with patch.dict(launcher.POLICY, policy):
                result, audit = launcher.scoped_execute(
                    remaining, TinyHint, [directory], old_execute, before_native)
            self.assertEqual(result["status"], "passed_numerical_software_only")
            self.assertTrue(all(state.values()))
            self.assertEqual(audit["eligible_files"], 6)
            self.assertEqual(audit["hinted_file_opens"], 6)
            self.assertEqual(audit["eligible_bytes"], 2 * policy["expected_preflight_hint_bytes"])
            self.assertEqual(audit["hinted_file_paths"],
                             [str(x) for x in files] * 2)
            self.assertIs(remaining.validate_prior_chain, original_chain)
            self.assertIs(remaining.validate_release, original_validate)
            self.assertIs(remaining.io.file_hash, original_hash)

    def test_pre_native_refusal_restores_all_functions_without_native(self):
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            _, remaining, state, policy = self._fixture(directory)
            originals = (remaining.validate_prior_chain, remaining.validate_release,
                         remaining.io.file_hash)
            audit = {"eligible_bytes": 0, "eligible_files": 0,
                     "hinted_file_opens": 0, "hinted_file_paths": []}

            def refuse_native(context, current_audit):
                state["preflight_seen"] = True
                raise ValueError("generated host below launch floor")

            def old_execute():
                remaining.validate_release({}, root=directory)
                state["native_started"] = True

            with patch.dict(launcher.POLICY, policy):
                with self.assertRaisesRegex(ValueError, "below launch floor"):
                    launcher.scoped_execute(remaining, TinyHint, [directory],
                                            old_execute, refuse_native, audit=audit)
            self.assertTrue(state["preflight_seen"])
            self.assertFalse(state["native_started"])
            self.assertEqual(audit["hinted_file_opens"], 3)
            self.assertEqual((remaining.validate_prior_chain, remaining.validate_release,
                              remaining.io.file_hash), originals)

    def test_missing_nested_hint_coverage_refuses_before_native(self):
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            _, remaining, state, policy = self._fixture(directory)
            original_chain = remaining.validate_prior_chain

            def old_execute():
                remaining.validate_release({}, root=directory)
                state["native_started"] = True

            with patch.dict(launcher.POLICY, dict(policy,
                    expected_preflight_hint_opens=4)):
                with self.assertRaisesRegex(ValueError, "hint coverage"):
                    launcher.scoped_execute(remaining, TinyHint, [directory],
                                            old_execute, lambda *_: None)
            self.assertFalse(state["native_started"])
            self.assertIs(remaining.validate_prior_chain, original_chain)

    def test_native_stage_host_gate_runs_before_supervisor_and_restores(self):
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            _, remaining, state, policy = self._fixture(directory)
            started = []

            def source_supervise(stage, *args, **kwargs):
                started.append(stage)
                return {"status": "completed_within_caps"}

            remaining.supervise_stage = source_supervise

            def old_execute():
                remaining.validate_release({}, root=directory)
                remaining.supervise_stage("native", [], None, {})
                state["native_started"] = True

            with patch.dict(launcher.POLICY, policy):
                with self.assertRaisesRegex(ValueError, "generated launch guard"):
                    launcher.scoped_execute(
                        remaining, TinyHint, [directory], old_execute,
                        lambda *_: None,
                        native_stage_check=lambda: (_ for _ in ()).throw(
                            ValueError("generated launch guard")))
            self.assertEqual(started, [])
            self.assertFalse(state["native_started"])
            self.assertIs(remaining.supervise_stage, source_supervise)

    def test_owned_cleanup_exception_reaches_terminal_callback(self):
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            _, remaining, _, policy = self._fixture(directory)
            cleanup_records = []

            def source_supervise(stage, *args, **kwargs):
                return {"status": "completed_within_caps"}

            remaining.supervise_stage = source_supervise

            def old_execute():
                remaining.validate_release({}, root=directory)
                remaining.supervise_stage("native", [], None, {})

            with patch.dict(launcher.POLICY, policy), \
                    patch.object(launcher.OwnedStage, "cleanup", side_effect=RuntimeError):
                with self.assertRaisesRegex(RuntimeError, "cleanup incomplete"):
                    launcher.scoped_execute(
                        remaining, TinyHint, [directory], old_execute,
                        lambda *_: None, native_stage_check=lambda: None,
                        stage_cleanup_callback=lambda stage, cleanup:
                            cleanup_records.append((stage, cleanup)))
            self.assertEqual(cleanup_records[0][0], "native")
            self.assertIn("cleanup_exception:RuntimeError",
                          cleanup_records[0][1]["errors"])
            self.assertIs(remaining.supervise_stage, source_supervise)

    def test_extension_source_binding_rejects_changed_worktree_and_head(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            contents = {}
            for relative in launcher.SOURCES:
                path = root / relative
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_bytes(relative.encode())
                contents[relative] = path.read_bytes()
            bindings = {relative: hashlib.sha256(raw).hexdigest()
                        for relative, raw in contents.items()}
            commit = "a" * 40

            def fake_git(root_arg, *args):
                self.assertEqual(root_arg, root)
                if args == ("rev-parse", "HEAD"):
                    return (commit + "\n").encode()
                relative = args[-1].split(":", 1)[1]
                if args[1] == "-s":
                    return str(len(contents[relative])).encode()
                return contents[relative]

            with patch.object(launcher, "_git", side_effect=fake_git):
                result = launcher.verify_extension_sources(
                    root, commit, bindings, require_head=True)
                self.assertEqual(result["source_hashes"], bindings)
                (root / launcher.SOURCES[1]).write_bytes(b"changed")
                with self.assertRaisesRegex(ValueError, "hash differs"):
                    launcher.verify_extension_sources(root, commit, bindings,
                                                      require_head=True)
            with patch.object(launcher, "_git", return_value=("b" * 40).encode()):
                with self.assertRaisesRegex(ValueError, "Checkout differs"):
                    launcher.verify_extension_sources(root, commit, bindings,
                                                      require_head=True)

    def test_depleted_host_after_full_validation_refuses_native_reservation(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary).resolve()
            allowed = root / Path(actual_remaining.receipt_path(0)).parent
            allowed.mkdir(parents=True)
            files = [allowed / f"tiny-{i}.bin" for i in range(3)]
            for i, file in enumerate(files):
                file.write_bytes(bytes([i + 1]) * (20 + i))
            bytes_per_chain = sum(file.stat().st_size for file in files)
            started = []

            def source_chain(*args, **kwargs):
                if args[1] == 2:
                    return actual_remaining.io.file_hash(files[1], maximum=4096)
                actual_remaining.io.file_hash(files[0], maximum=4096)
                actual_remaining.validate_prior_chain([], 2)
                return actual_remaining.io.file_hash(files[2], maximum=4096)

            def source_validate(*args, **kwargs):
                actual_remaining.validate_prior_chain([], 1)
                return {"run_id": "tension:N12:S60:reference", "index": 8,
                        "deck": b"tiny-deck", "adapter_receipt": {},
                        "previous": {}, "source_hashes": {},
                        "observed_head_preflight": "a" * 40}

            def fake_execute(*args, **kwargs):
                actual_remaining.validate_release({}, root=root)
                started.append("native")
                return {"status": "passed_numerical_software_only"}

            class HostSampler:
                calls = 0

                def host(self):
                    self.calls += 1
                    return {"kernel_pressure_mask": 1,
                            "available_percent": 55 if self.calls == 1 else 44}

            envelope = {"source_commit": "a" * 40,
                        "extension_source_bindings": {
                            relative: "0" * 64 for relative in launcher.SOURCES}}
            inner_path = root / "inner.json"
            old_binding = (envelope, b"outer", (1, 2), {}, inner_path,
                           b"inner", (3, 4))
            with patch.dict(launcher.POLICY, {
                    "expected_preflight_hint_opens": 3,
                    "expected_preflight_hint_bytes": bytes_per_chain}), \
                    patch.object(launcher, "_read_exact_release", return_value=old_binding), \
                    patch.object(launcher, "_load_sibling", side_effect=[
                        TinyHint, SimpleNamespace(FastDarwinSampler=HostSampler)]), \
                    patch.object(actual_remaining, "validate_prior_chain", source_chain), \
                    patch.object(actual_remaining, "validate_release", source_validate), \
                    patch.object(actual_remaining, "execute", side_effect=fake_execute):
                outcome = launcher.execute_envelope(root / "outer.json", root=root)
            self.assertEqual(outcome["status"], "failed_or_incomplete")
            self.assertIn("unchanged 45%", outcome["failure"]["message"], outcome)
            self.assertEqual(outcome["preflight_hint_audit"]["hinted_file_opens"], 3)
            self.assertEqual(outcome["final_hint_audit"]["hinted_file_opens"], 3)
            self.assertEqual(started, [])
            self.assertFalse((root / launcher.NATIVE_OUTPUT).exists())
            self.assertTrue((root / launcher.SIDECAR / "receipt.json").is_file())

    def test_existing_policy_sidecar_refuses_second_use(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / launcher.SIDECAR).mkdir(parents=True)

            class HostSampler:
                def host(self):
                    raise AssertionError("Host must not be sampled after one-use refusal")

            envelope = {"source_commit": "a" * 40,
                        "extension_source_bindings": {
                            relative: "0" * 64 for relative in launcher.SOURCES}}
            old_binding = (envelope, b"outer", (1, 2), {}, root / "inner.json",
                           b"inner", (3, 4))
            with patch.object(launcher, "_read_exact_release", return_value=old_binding), \
                    patch.object(launcher, "_load_sibling", side_effect=[
                        TinyHint, SimpleNamespace(FastDarwinSampler=HostSampler)]):
                with self.assertRaisesRegex(ValueError, "one-use"):
                    launcher.execute_envelope(root / "outer.json", root=root)
            self.assertFalse((root / launcher.NATIVE_OUTPUT).exists())

    def test_generated_complete_old_runner_binds_policy_and_native_receipt(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary).resolve()
            allowed = root / Path(actual_remaining.receipt_path(0)).parent
            allowed.mkdir(parents=True)
            files = [allowed / f"tiny-{i}.bin" for i in range(3)]
            for i, file in enumerate(files):
                file.write_bytes(bytes([i + 1]) * (20 + i))
            bytes_per_chain = sum(file.stat().st_size for file in files)
            native_receipt = root / actual_remaining.receipt_path(8)
            stages = []

            def source_chain(*args, **kwargs):
                if args[1] == 2:
                    return actual_remaining.io.file_hash(files[1], maximum=4096)
                actual_remaining.io.file_hash(files[0], maximum=4096)
                actual_remaining.validate_prior_chain([], 2)
                return actual_remaining.io.file_hash(files[2], maximum=4096)

            def source_validate(*args, **kwargs):
                actual_remaining.validate_prior_chain([], 1)
                previous = {key: 0 for key in (
                    "native_seconds", "readout_seconds", "prep_seconds",
                    "output_bytes", "native_calls", "supplement_readout_seconds",
                    "supplement_prep_seconds", "supplement_output_bytes",
                    "supplement_replay_calls")}
                return {"run_id": "tension:N12:S60:reference", "index": 8,
                        "deck": b"tiny-deck", "adapter_receipt": {},
                        "previous": previous, "source_hashes": {},
                        "observed_head_preflight": "a" * 40}

            def source_supervise(stage, *args, **kwargs):
                stages.append(stage)
                return {"status": "completed_within_caps"}

            def fake_execute(*args, **kwargs):
                actual_remaining.validate_release({}, root=root)
                native_receipt.parent.mkdir(parents=True)
                saved = {
                    "status": "passed_numerical_software_only",
                    "native_calls_attempted": 1,
                    "readout_calls_attempted": 1,
                    "native_stage": {"elapsed_seconds": 0.0},
                    "readout_stage": {"elapsed_seconds": 0.0}}
                native_receipt.write_text(json.dumps(saved) + "\n")
                actual_remaining.supervise_stage("native", [], None, {})
                actual_remaining.supervise_stage("readout", [], None, {})
                actual_remaining.validate_prior_chain([], 1)
                return saved

            class HostSampler:
                calls = 0

                def host(self):
                    self.calls += 1
                    return {"kernel_pressure_mask": 1, "available_percent": 54,
                            "call": self.calls}

            envelope = {"source_commit": "a" * 40,
                        "extension_source_bindings": {
                            relative: "0" * 64 for relative in launcher.SOURCES}}
            inner_path = root / "inner.json"
            old_binding = (envelope, b"outer", (1, 2), {}, inner_path,
                           b"inner", (3, 4))
            def old_read(path):
                if path == root / "outer.json":
                    return b"outer", (1, 2)
                if path == inner_path:
                    return b"inner", (3, 4)
                if path == native_receipt:
                    return native_receipt.read_bytes(), (5, 6)
                raise AssertionError("Unexpected tiny release/receipt path")
            with patch.dict(launcher.POLICY, {
                    "expected_preflight_hint_opens": 3,
                    "expected_preflight_hint_bytes": bytes_per_chain,
                    "expected_completed_hint_opens": 6,
                    "expected_completed_hint_bytes": 2 * bytes_per_chain}), \
                    patch.object(launcher, "_read_exact_release", return_value=old_binding), \
                    patch.object(launcher, "_load_sibling", side_effect=[
                        TinyHint, SimpleNamespace(FastDarwinSampler=HostSampler)]), \
                    patch.object(launcher, "verify_extension_sources", return_value={}), \
                    patch.object(io, "read_release", side_effect=old_read), \
                    patch.object(actual_remaining, "validate_prior_chain", source_chain), \
                    patch.object(actual_remaining, "validate_release", source_validate), \
                    patch.object(actual_remaining, "supervise_stage", source_supervise), \
                    patch.object(actual_remaining, "execute", side_effect=fake_execute):
                result = launcher.execute_envelope(root / "outer.json", root=root)
            self.assertEqual(result["status"],
                             "passed_numerical_software_only_with_declared_io_policy_v2")
            self.assertEqual(result["final_hint_audit"]["hinted_file_opens"], 6)
            self.assertEqual(result["final_hint_audit"]["eligible_bytes"],
                             2 * bytes_per_chain)
            self.assertEqual(stages, ["native", "readout"])
            self.assertEqual(result["host_immediately_before_native_supervision"]["call"], 3)
            self.assertEqual(result["native_receipt_sha256"],
                             hashlib.sha256(native_receipt.read_bytes()).hexdigest())
            self.assertEqual(result["native_calls_attempted"], 1)
            self.assertEqual(result["hbe_readout_calls_attempted"], 1)
            self.assertGreater(result["extension_resource_charge"]["sidecar_reserved_output_bytes"], 0)
            self.assertEqual(len(result["preflight_hint_audit"]["hinted_file_paths"]), 3)
            self.assertEqual(len(result["final_hint_audit"]["hinted_file_paths"]), 6)
            self.assertTrue((root / launcher.SIDECAR / "receipt.json").is_file())


if __name__ == "__main__":
    unittest.main()
