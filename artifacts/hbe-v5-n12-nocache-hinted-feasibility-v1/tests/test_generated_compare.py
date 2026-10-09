"""Generated protocol controls; no full predecessor files are opened."""
from __future__ import annotations

from contextlib import redirect_stdout
import hashlib
import importlib.util
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from scripts import mechanics_hbe_v5_remaining_one_shot as remaining

HERE = Path(__file__).parent


def load(name):
    source = HERE / name
    spec = importlib.util.spec_from_file_location("test_" + source.stem, source)
    assert spec is not None and spec.loader is not None
    value = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(value)
    return value


comparison = load("compare.py")
hint = load("hash_hint.py")


class TinyDiagnostic:
    OUT = None

    def __init__(self, source_file):
        self.source_file = source_file
        self.observed_hash = None

    def worker_terminal(self, source_commit):
        self.observed_hash = remaining.validate_prior_chain([], 1)
        return 0

    def save_bounded(self, name, value):
        (self.OUT / name).write_text(json.dumps(value))


class GeneratedComparisonTests(unittest.TestCase):
    def test_validation_identity_captures_same_decision_inputs(self):
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)

            class IdentityDiagnostic(TinyDiagnostic):
                def worker_terminal(self, source_commit):
                    remaining.validate_release({}, root=directory)
                    return 0

            fake = IdentityDiagnostic(directory / "unused")
            record = {"run_id": "tension:N12:S60:reference", "index": 8,
                      "deck": b"tiny-deck", "previous": {"native_calls": 8},
                      "adapter_receipt": {"steps": 60},
                      "source_hashes": {"one": "abc"},
                      "observed_head_preflight": "0" * 40}
            with patch.dict(comparison.ARMS, {"baseline": directory}), \
                    patch.object(remaining, "validate_release", return_value=record) as check:
                self.assertEqual(comparison.run_worker_arm(
                    "baseline", "0" * 40, fake, hint), 0)
                check.assert_called_once()
            identity = json.loads((directory / "validation-identity.json").read_text())
            self.assertEqual(identity["deck_sha256"], hashlib.sha256(b"tiny-deck").hexdigest())
            self.assertEqual(identity["previous"], record["previous"])
            self.assertEqual(identity["adapter_receipt"], record["adapter_receipt"])

    def test_hinted_arm_changes_only_hash_file_handle_within_chain(self):
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            file = directory / "tiny.bin"
            content = b"tiny generated hash input"
            file.write_bytes(content)
            expected = hashlib.sha256(content).hexdigest()
            fake = TinyDiagnostic(file)
            original_hash = remaining.io.file_hash

            def source_chain(*args, **kwargs):
                return remaining.io.file_hash(file, maximum=4096)

            class TinyHint:
                @staticmethod
                def hinted_hash(path, *, maximum, original_hash,
                                allowed_directories, audit):
                    return hint.hinted_hash(
                        path, maximum=maximum, original_hash=original_hash,
                        allowed_directories=allowed_directories, audit=audit,
                        minimum_bytes=1)

            with patch.dict(comparison.ARMS, {"hinted": directory}), \
                    patch.object(remaining, "validate_prior_chain", source_chain):
                self.assertEqual(comparison.run_worker_arm(
                    "hinted", "0" * 40, fake, TinyHint,
                    allowed_directories=[directory]), 0)
                self.assertIs(remaining.validate_prior_chain, source_chain)
            self.assertIs(remaining.io.file_hash, original_hash)
            self.assertEqual(fake.observed_hash, expected)
            audit = json.loads((directory / "hint-audit.json").read_text())
            self.assertEqual(audit["eligible_files"], 1)
            self.assertEqual(audit["hinted_file_opens"], 1)
            self.assertEqual(audit["eligible_bytes"], len(content))

    def test_nested_predecessor_keeps_hint_before_inside_and_after(self):
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            files = [directory / f"generated-{index}.bin" for index in range(3)]
            for index, file in enumerate(files):
                file.write_bytes(bytes([index + 1]) * 41)
            expected = [hashlib.sha256(file.read_bytes()).hexdigest() for file in files]
            original_hash = remaining.io.file_hash

            def source_chain(*args, **kwargs):
                index = args[1]
                if index == 2:
                    return [remaining.io.file_hash(files[1], maximum=4096)]
                before = remaining.io.file_hash(files[0], maximum=4096)
                inside = remaining.validate_prior_chain([], 2)
                after = remaining.io.file_hash(files[2], maximum=4096)
                return [before, *inside, after]

            class TinyHint:
                @staticmethod
                def hinted_hash(path, *, maximum, original_hash,
                                allowed_directories, audit):
                    return hint.hinted_hash(
                        path, maximum=maximum, original_hash=original_hash,
                        allowed_directories=allowed_directories, audit=audit,
                        minimum_bytes=1)

            fake = TinyDiagnostic(files[0])
            with patch.dict(comparison.ARMS, {"hinted": directory}), \
                    patch.object(remaining, "validate_prior_chain", source_chain):
                self.assertEqual(comparison.run_worker_arm(
                    "hinted", "0" * 40, fake, TinyHint,
                    allowed_directories=[directory]), 0)
                self.assertIs(remaining.validate_prior_chain, source_chain)
            self.assertIs(remaining.io.file_hash, original_hash)
            self.assertEqual(fake.observed_hash, expected)
            audit = json.loads((directory / "hint-audit.json").read_text())
            self.assertEqual(audit["eligible_files"], 3)
            self.assertEqual(audit["hinted_file_opens"], 3)
            self.assertEqual(audit["hinted_file_paths"], [str(file) for file in files])

    def test_nested_exception_restores_original_hash_and_chain(self):
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            file = directory / "generated.bin"
            file.write_bytes(b"generated nested failure")
            original_hash = remaining.io.file_hash

            def source_chain(*args, **kwargs):
                if args[1] == 2:
                    raise RuntimeError("generated nested failure")
                remaining.io.file_hash(file, maximum=4096)
                remaining.validate_prior_chain([], 2)

            class TinyHint:
                @staticmethod
                def hinted_hash(path, *, maximum, original_hash,
                                allowed_directories, audit):
                    return hint.hinted_hash(
                        path, maximum=maximum, original_hash=original_hash,
                        allowed_directories=allowed_directories, audit=audit,
                        minimum_bytes=1)

            fake = TinyDiagnostic(file)
            with patch.dict(comparison.ARMS, {"hinted": directory}), \
                    patch.object(remaining, "validate_prior_chain", source_chain):
                with self.assertRaisesRegex(RuntimeError, "generated nested failure"):
                    comparison.run_worker_arm(
                        "hinted", "0" * 40, fake, TinyHint,
                        allowed_directories=[directory])
                self.assertIs(remaining.validate_prior_chain, source_chain)
            self.assertIs(remaining.io.file_hash, original_hash)
            audit = json.loads((directory / "hint-audit.json").read_text())
            self.assertEqual(audit["eligible_files"], 1)
            self.assertEqual(audit["hinted_file_opens"], 1)

    def test_baseline_arm_uses_original_hash_without_hint_audit(self):
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            file = directory / "tiny.bin"
            file.write_bytes(b"baseline")
            fake = TinyDiagnostic(file)
            original_hash = remaining.io.file_hash

            def source_chain(*args, **kwargs):
                return remaining.io.file_hash(file, maximum=4096)

            with patch.dict(comparison.ARMS, {"baseline": directory}), \
                    patch.object(remaining, "validate_prior_chain", source_chain):
                self.assertEqual(comparison.run_worker_arm(
                    "baseline", "0" * 40, fake, hint), 0)
            self.assertIs(remaining.io.file_hash, original_hash)
            self.assertEqual(fake.observed_hash, hashlib.sha256(b"baseline").hexdigest())
            self.assertFalse((directory / "hint-audit.json").exists())

    def test_existing_arm_directory_refuses_one_use_before_host_or_worker(self):
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary) / "hinted-01"
            directory.mkdir()
            with patch.dict(comparison.ARMS, {"hinted": directory}), \
                    patch.object(comparison, "selected_head", return_value="0" * 40), \
                    patch.object(comparison, "load_exact", return_value=object()):
                with self.assertRaisesRegex(RuntimeError, "already exists"):
                    comparison.supervisor("hinted", "0" * 40)

    def test_worker_refuses_wrapper_bytes_that_differ_from_supervisor(self):
        with patch.object(comparison, "comparison_source_sha", return_value="a" * 64), \
                patch.object(comparison, "selected_head") as selected:
            with self.assertRaisesRegex(RuntimeError, "wrapper changed"):
                comparison.worker("baseline", "0" * 40, "b" * 64)
            selected.assert_not_called()

    def test_supervisor_fails_when_wrapper_changes_before_terminal_receipt(self):
        actual_diagnostic = comparison.load_exact(
            comparison.DIAGNOSTIC, comparison.DIAGNOSTIC_SHA, "generated_postguard_diag")

        class Host:
            def host(self):
                return {"kernel_pressure_mask": 1, "available_percent": 56,
                        "swap_used_bytes": 0}

        class Owner:
            def cleanup(self, *args):
                return {"contained": True, "direct_child_reaped": True,
                        "fallback_used": False, "errors": []}

        with tempfile.TemporaryDirectory() as temporary:
            target = Path(temporary) / "baseline-01"
            deck_hash = hashlib.sha256(b"tiny-deck").hexdigest()

            def simulated_stage(*args, **kwargs):
                (target / "worker-result.json").write_text(json.dumps({
                    "status": "read_only_validation_memory_diagnostic_complete",
                    "native_calls": 0, "release_written": False,
                    "target_reserved": False, "adapted_deck_sha256": deck_hash}))
                (target / "validation-identity.json").write_text(json.dumps({
                    "index": 8, "deck_sha256": deck_hash,
                    "observed_head_preflight": "0" * 40}))
                self.assertEqual(args[1][-1], "a" * 64)
                return {"status": "completed_within_caps"}

            with patch.dict(comparison.ARMS, {"baseline": target}), \
                    patch.object(comparison, "selected_head", return_value="0" * 40), \
                    patch.object(comparison, "load_exact", side_effect=[
                        actual_diagnostic, object()]), \
                    patch.object(comparison, "comparison_source_sha", side_effect=[
                        "a" * 64, "b" * 64]), \
                    patch.object(actual_diagnostic, "load_host_sampler", return_value=Host()), \
                    patch.object(actual_diagnostic, "OwnedWorker", Owner), \
                    patch.object(remaining, "supervise_stage", side_effect=simulated_stage), \
                    redirect_stdout(io.StringIO()):
                self.assertEqual(comparison.supervisor("baseline", "0" * 40), 1)
            saved = json.loads((target / "receipt.json").read_text())
            self.assertEqual(saved["status"], "failed_or_incomplete")
            self.assertEqual(saved["comparison_source_sha256"], "a" * 64)
            self.assertEqual(saved["comparison_source_sha256_postguard"], "b" * 64)

    def test_failed_comparison_stage_saves_terminal_receipt(self):
        actual_diagnostic = comparison.load_exact(
            comparison.DIAGNOSTIC, comparison.DIAGNOSTIC_SHA, "generated_diag")

        class Host:
            def host(self):
                return {"kernel_pressure_mask": 1, "available_percent": 56,
                        "swap_used_bytes": 0}

        with tempfile.TemporaryDirectory() as temporary:
            target = Path(temporary) / "baseline-01"
            with patch.dict(comparison.ARMS, {"baseline": target}), \
                    patch.object(comparison, "selected_head", return_value="0" * 40), \
                    patch.object(comparison, "load_exact", side_effect=[
                        actual_diagnostic, object()]), \
                    patch.object(actual_diagnostic, "load_host_sampler", return_value=Host()), \
                    patch.object(remaining, "supervise_stage", return_value={
                        "status": "failed_or_incomplete"}), \
                    redirect_stdout(io.StringIO()):
                self.assertEqual(comparison.supervisor("baseline", "0" * 40), 1)
            saved = json.loads((target / "receipt.json").read_text())
            self.assertEqual(saved["status"], "failed_or_incomplete")
            self.assertEqual(saved["native_calls"], 0)
            self.assertFalse(saved["release_written"])


if __name__ == "__main__":
    unittest.main()
