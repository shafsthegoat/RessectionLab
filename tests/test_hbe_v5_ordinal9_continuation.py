"""Generated-only ordinal-9 launcher controls. No FEBio or predecessor replay."""
from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path
import shutil
import subprocess
import tempfile
import time
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from scripts import mechanics_hbe_v5_n8_one_shot as io
from scripts import mechanics_hbe_v5_remaining_one_shot as remaining


ROOT = Path(__file__).resolve().parents[1]
HERE = ROOT / "launchers"
spec = importlib.util.spec_from_file_location(
    "generated_ordinal9_continuation", HERE / "hbe_v5_ordinal9_continuation_v1.py")
assert spec is not None and spec.loader is not None
candidate = importlib.util.module_from_spec(spec)
spec.loader.exec_module(candidate)
spec = importlib.util.spec_from_file_location(
    "generated_ordinal9_hint", HERE / "hbe_v5_nocache_hash_v1.py")
assert spec is not None and spec.loader is not None
hint = importlib.util.module_from_spec(spec)
spec.loader.exec_module(hint)


class TinyHint:
    @staticmethod
    def hinted_hash(path, *, maximum, original_hash, allowed_directories, audit):
        return hint.hinted_hash(path, maximum=maximum, original_hash=original_hash,
                                allowed_directories=allowed_directories, audit=audit,
                                minimum_bytes=1)


def previous_ledger() -> dict:
    return {"sha256": ["0" * 64] * 8 + ["1" * 64],
            "native_seconds": 100., "readout_seconds": 50.,
            "prep_seconds": 20., "output_bytes": 1000,
            "native_calls": 9, "supplement_readout_seconds": 1.,
            "supplement_prep_seconds": 1., "supplement_output_bytes": 500,
            "supplement_replay_calls": 1,
            "combined_wall_seconds": 172., "combined_output_bytes": 1500}


class ContinuationTests(unittest.TestCase):
    def test_four_source_binding_and_exact_ordinal9_envelope(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary).resolve()
            for relative in candidate.SOURCES:
                target = root / relative
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(ROOT / relative, target)
            subprocess.run(["/usr/bin/git", "init", "-q"], cwd=root, check=True)
            subprocess.run(["/usr/bin/git", "add", "launchers", "scripts"],
                           cwd=root, check=True)
            subprocess.run(["/usr/bin/git", "-c", "user.name=skamal23",
                            "-c", "user.email=sayemkamal12@gmail.com", "commit", "-qm",
                            "Generated binding control", "-m",
                            "Co-authored-by: shafsthegoat <shafrir.p@gmail.com>"],
                           cwd=root, check=True)
            commit = subprocess.run(["/usr/bin/git", "rev-parse", "HEAD"],
                                    cwd=root, capture_output=True, text=True,
                                    check=True).stdout.strip()
            hashes = {relative: hashlib.sha256((root / relative).read_bytes()).hexdigest()
                      for relative in candidate.SOURCES}
            inner = {"ordinal": 9, "run_id": "tension:N16:S60:reference",
                     "status": "root_released_one_native_call", "source_commit": commit}
            release = root / "build" / "inner.json"
            release.parent.mkdir()
            release.write_text(json.dumps(inner))
            outer = {"schema": "hbe-v5-ordinal9-continuation-envelope-v1",
                     "status": "root_released_one_native_call_with_v2_ledger",
                     "source_commit": commit, "extension_source_bindings": hashes,
                     "inner_release": {"path": "build/inner.json",
                                       "sha256": hashlib.sha256(release.read_bytes()).hexdigest()},
                     "sidecar_directory": candidate.SIDECAR, "policy": candidate.POLICY}
            envelope = root / "build" / "outer.json"
            envelope.write_text(json.dumps(outer))
            local_spec = importlib.util.spec_from_file_location(
                "generated_bound_ordinal9", root / candidate.SOURCES[0])
            assert local_spec is not None and local_spec.loader is not None
            bound = importlib.util.module_from_spec(local_spec)
            local_spec.loader.exec_module(bound)
            self.assertEqual(bound._read_exact_release(root, envelope)[0], outer)
            for key, wrong in (("schema", "hbe-v5-nocache-launch-envelope-v2"),
                               ("status", "root_released_one_native_call")):
                with self.subTest(key=key):
                    changed = dict(outer, **{key: wrong})
                    envelope.write_text(json.dumps(changed))
                    with self.assertRaisesRegex(ValueError, "separately released"):
                        bound._read_exact_release(root, envelope)
            envelope.write_text(json.dumps(outer))
            (root / candidate.SOURCES[1]).write_bytes(b"tampered admission")
            with self.assertRaisesRegex(ValueError, "hash differs"):
                bound._read_exact_release(root, envelope)

    def test_saved_native_and_supplemental_ledger_are_bound_once(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary).resolve()
            native_dir = root / candidate.NATIVE_OUTPUT
            native_dir.mkdir(parents=True)
            before = previous_ledger()
            surcharge = {"prep_wall_seconds": 1.14, "output_bytes": 1024**2}
            record = {
                "charged_previous": before,
                "old_validation_identity": {"previous": before},
                "v2_ordinal8_exact_evidence": {"v2_sidecar_sha256": "2" * 64},
                "v2_continuation_ledger": {"authenticated_ordinal8_surcharge": surcharge},
                "native_receipt_sha256": "3" * 64,
            }
            native = {"ordinal": 9, "run_id": "tension:N16:S60:reference",
                      "status": "passed_numerical_software_only", "no_retry": True,
                      "native_calls_attempted": 1, "readout_calls_attempted": 1,
                      "prior_receipt_sha256": before["sha256"],
                      "native_stage": {"elapsed_seconds": .1},
                      "readout_stage": {"elapsed_seconds": .1}}
            with patch.object(remaining, "active_bytes", return_value=400):
                candidate._charge_full_preparation(remaining, root, record,
                                                   time.monotonic() - .5, native)
            after = record["supplemental_ledger_after_ordinal9"]
            self.assertEqual(after["charged_cumulative_ledger"]["sha256"],
                             before["sha256"] + ["3" * 64])
            self.assertEqual(after["ordinal8_surcharge"], surcharge)
            self.assertEqual(after["charged_cumulative_ledger"]["output_bytes"],
                             1000 + 400 + 1024**2)
            self.assertEqual(after["charged_cumulative_ledger"]["native_calls"], 10)
            self.assertEqual(before["sha256"][-1], "1" * 64)
            with patch.object(remaining, "active_bytes", return_value=400):
                with self.assertRaisesRegex(ValueError, "one-call lineage"):
                    candidate._charge_full_preparation(
                        remaining, root, record, time.monotonic() - .5,
                        dict(native, no_retry=False))

    def test_native_receipt_must_match_returned_bytes_and_remain_stable(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary).resolve()
            path = root / candidate.NATIVE_OUTPUT / "receipt.json"
            path.parent.mkdir(parents=True)
            returned = {"status": "passed_numerical_software_only",
                        "native_calls_attempted": 1}
            path.write_text(json.dumps(returned))
            record = {}
            saved = candidate._bind_native_receipt(root, record)
            candidate._require_matching_native_receipt(saved, returned)
            with self.assertRaisesRegex(ValueError, "returned receipt differs"):
                candidate._require_matching_native_receipt(saved, dict(returned, x=1))
            path.unlink()
            with self.assertRaisesRegex(ValueError, "changed or disappeared"):
                candidate._rebind_native_receipt_stably(root, record)

    def test_sidecar_inventory_requires_one_regular_bounded_receipt(self):
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            receipt = directory / "receipt.json"
            receipt.write_text("{}")
            self.assertEqual(candidate._check_sidecar_inventory(directory), 2)
            extra = directory / "extra.json"
            extra.write_text("{}")
            with self.assertRaisesRegex(ValueError, "inventory differs"):
                candidate._check_sidecar_inventory(directory)
            extra.unlink()
            receipt.unlink()
            receipt.symlink_to(directory / "elsewhere")
            with self.assertRaisesRegex(ValueError, "type or cap"):
                candidate._check_sidecar_inventory(directory)
            receipt.unlink()
            with patch.dict(candidate.POLICY, sidecar_output_cap_bytes=1):
                receipt.write_text("{}")
                with self.assertRaisesRegex(ValueError, "type or cap"):
                    candidate._check_sidecar_inventory(directory)

    def test_complete_generated_continuation_and_preflight_refusals(self):
        for scenario in ("pass", "host_low", "v2_changed", "missing_hint",
                         "source_changed", "release_changed"):
            with self.subTest(scenario=scenario), tempfile.TemporaryDirectory() as temporary:
                root = Path(temporary).resolve()
                allowed = root / Path(remaining.receipt_path(0)).parent
                allowed.mkdir(parents=True)
                files = [allowed / f"tiny-{i}.bin" for i in range(3)]
                for i, file in enumerate(files):
                    file.write_bytes(bytes([i + 1]) * (20 + i))
                original_hash = remaining.io.file_hash
                old_previous = previous_ledger()
                exact = {"v2_sidecar_sha256": "2" * 64}
                changed = {"v2_sidecar_sha256": "4" * 64}
                calls = {"v2": 0, "native": 0, "readout": 0}

                def source_chain(*args, **kwargs):
                    if args[1] == 2:
                        return remaining.io.file_hash(files[1], maximum=4096)
                    remaining.io.file_hash(files[0], maximum=4096)
                    remaining.validate_prior_chain([], 2)
                    return remaining.io.file_hash(files[2], maximum=4096)

                def source_validate(*args, **kwargs):
                    remaining.validate_prior_chain([], 1)
                    return {"run_id": remaining.ORDER[9], "index": 9,
                            "deck": b"tiny-deck", "adapter_receipt": {},
                            "previous": old_previous, "source_hashes": {},
                            "observed_head_preflight": "a" * 40}

                def verify_exact(*, root):
                    calls["v2"] += 1
                    return changed if scenario == "v2_changed" and calls["v2"] >= 2 else exact

                admission = SimpleNamespace(
                    verify_exact=verify_exact,
                    continuation_ledger=lambda previous, evidence, index: {
                        "charged_cumulative_ledger": dict(
                            previous,
                            prep_seconds=previous["prep_seconds"] + 1.14,
                            output_bytes=previous["output_bytes"] + 1024**2,
                            combined_wall_seconds=previous["combined_wall_seconds"] + 1.14,
                            combined_output_bytes=previous["combined_output_bytes"] + 1024**2,
                            v2_surcharge_applied={"source_ordinal8_v2_sidecar_sha256":
                                                  exact["v2_sidecar_sha256"]}),
                        "authenticated_ordinal8_surcharge": {
                            "prep_wall_seconds": 1.14, "output_bytes": 1024**2}})

                def old_supervise(stage, *args, **kwargs):
                    calls[stage] += 1
                    return {"status": "completed_within_caps"}

                def old_execute(*args, **kwargs):
                    remaining.validate_release({}, root=root)
                    path = root / candidate.NATIVE_OUTPUT / "receipt.json"
                    path.parent.mkdir(parents=True)
                    saved = {"ordinal": 9, "run_id": remaining.ORDER[9],
                             "status": "reserved", "no_retry": True,
                             "native_calls_attempted": 1, "readout_calls_attempted": 1,
                             "prior_receipt_sha256": old_previous["sha256"],
                             "native_stage": {"elapsed_seconds": 0.0},
                             "readout_stage": {"elapsed_seconds": 0.0}}
                    path.write_text(json.dumps(saved))
                    remaining.supervise_stage("native", [], None, {})
                    remaining.supervise_stage("readout", [], None, {})
                    saved["status"] = "passed_numerical_software_only"
                    path.write_text(json.dumps(saved))
                    remaining.validate_prior_chain([], 1)
                    return saved

                class HostSampler:
                    def host(self):
                        return {"kernel_pressure_mask": 1,
                                "available_percent": 53 if scenario == "host_low" else 54}

                outer = root / "outer.json"
                inner = root / "inner.json"
                envelope = {"source_commit": "a" * 40,
                            "extension_source_bindings": {
                                name: "0" * 64 for name in candidate.SOURCES}}
                sources = {"source_commit": "a" * 40, "observed_head": "a" * 40,
                           "source_hashes": envelope["extension_source_bindings"]}
                released = (envelope, b"outer", (1, 2), sources,
                            inner, b"inner", (3, 4))
                original_read = io.read_release
                reads = {"outer": 0}

                def small_read(path):
                    if path == outer:
                        reads["outer"] += 1
                        if scenario == "release_changed" and reads["outer"] >= 2:
                            return b"changed", (1, 2)
                        return b"outer", (1, 2)
                    if path == inner:
                        return b"inner", (3, 4)
                    return original_read(path)

                policy = {"expected_preflight_hint_opens": 3,
                          "expected_preflight_hint_bytes": sum(x.stat().st_size for x in files),
                          "expected_completed_hint_opens": 6,
                          "expected_completed_hint_bytes":
                              2 * sum(x.stat().st_size for x in files)}
                if scenario == "missing_hint":
                    policy["expected_preflight_hint_opens"] = 4
                source_checks = {"count": 0}

                def bound_sources(*args, **kwargs):
                    source_checks["count"] += 1
                    if scenario == "source_changed" and source_checks["count"] >= 2:
                        raise ValueError("generated source changed before native")
                    return sources

                with patch.dict(candidate.POLICY, policy), \
                        patch.object(candidate, "_read_exact_release", return_value=released), \
                        patch.object(candidate, "_load_bound_admission", return_value=admission), \
                        patch.object(candidate, "_load_sibling", side_effect=[
                            TinyHint, SimpleNamespace(FastDarwinSampler=HostSampler)]), \
                        patch.object(candidate, "verify_extension_sources",
                                     side_effect=bound_sources), \
                        patch.object(candidate, "_self_peak_rss_bytes", return_value=100), \
                        patch.object(io, "read_release", side_effect=small_read), \
                        patch.object(remaining, "validate_prior_chain", source_chain), \
                        patch.object(remaining, "validate_release", source_validate), \
                        patch.object(remaining, "supervise_stage", old_supervise), \
                        patch.object(remaining, "execute", side_effect=old_execute):
                    result = candidate.execute_envelope(outer, root=root)
                if scenario == "pass":
                    self.assertEqual(result["status"],
                                     "passed_numerical_software_only_with_v2_cumulative_ledger_v1",
                                     result)
                    self.assertEqual(result["final_hint_audit"]["hinted_file_opens"], 6)
                    self.assertEqual(result["supplemental_ledger_after_ordinal9"]["through_ordinal"], 9)
                    self.assertGreater(result["supplemental_ledger_after_ordinal9"]
                                       ["charged_cumulative_ledger"]["prep_seconds"],
                                       old_previous["prep_seconds"] + 1.14)
                    self.assertGreater(result["supplemental_ledger_after_ordinal9"]
                                       ["charged_cumulative_ledger"]["output_bytes"],
                                       old_previous["output_bytes"] + 2 * 1024**2)
                    self.assertEqual(calls["native"], 1)
                    self.assertEqual(calls["readout"], 1)
                    self.assertEqual(result["owned_stage_cleanup"]["native"]["contained"], True)
                    self.assertEqual(result["preflight_hint_audit"]["hinted_file_opens"], 3)
                else:
                    self.assertEqual(result["status"], "failed_or_incomplete", result)
                    self.assertEqual(calls["native"], 0)
                    if scenario in ("host_low", "v2_changed", "missing_hint"):
                        self.assertFalse((root / candidate.NATIVE_OUTPUT).exists())
                self.assertIs(remaining.io.file_hash, original_hash)


if __name__ == "__main__":
    unittest.main()
