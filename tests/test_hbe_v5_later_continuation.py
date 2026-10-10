"""Portable, generated-only HBE v5 later-continuation source controls.

Promote to tests/test_hbe_v5_later_continuation.py alongside the six reviewed
launcher sources. This test never opens HBE bulk data or starts a solver.
"""
from __future__ import annotations

import copy
import hashlib
import importlib.util
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
LAUNCHERS = ROOT / "launchers"


def load(name: str, basename: str):
    path = LAUNCHERS / basename
    if not path.is_file():
        raise AssertionError(f"Promoted continuation source missing: {basename}")
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


runtime = load("portable_later_runtime", "hbe_v5_later_continuation_runtime_v1.py")
core = load("portable_later_core", "hbe_v5_later_continuation_core_v1.py")
ledger = load("portable_later_ledger", "hbe_v5_later_ledger_math_v1.py")
ancestry = load("portable_later_ancestry", "hbe_v5_later_ancestry_v1.py")
row10 = load("portable_later_row10", "hbe_v5_ordinal10_continuation_v1.py")
row11 = load("portable_later_row11", "hbe_v5_ordinal11_continuation_v1.py")


def old_baseline(index: int) -> dict:
    return {"sha256": [f"{n + 1:064x}" for n in range(index)],
            "native_seconds": 100., "readout_seconds": 25.,
            "prep_seconds": 50., "output_bytes": 2_000_000,
            "native_calls": index, "combined_wall_seconds": 177.,
            "combined_output_bytes": 2_100_000,
            "supplement_readout_seconds": 1., "supplement_prep_seconds": 1.,
            "supplement_output_bytes": 100_000,
            "supplement_replay_calls": 1}


def charges(old: dict) -> list[dict]:
    return [{"ordinal": n,
             "native_receipt_sha256": old["sha256"][n],
             "sidecar_sha256": f"{n + 100:064x}",
             "native_prep_seconds": 3.,
             "launcher_inclusive_prep_seconds": 4. + n / 10,
             "reserved_sidecar_output_bytes": 1024**2}
            for n in range(8, old["native_calls"])]


class PortableLaterContinuation(unittest.TestCase):
    def test_nested_hint_and_owned_stage_seams_restore(self):
        remaining = SimpleNamespace()
        remaining.io = SimpleNamespace(file_hash=lambda path, *, maximum: path)

        def chain(nested=False):
            remaining.io.file_hash("abc", maximum=100)
            if not nested:
                remaining.validate_prior_chain(True)
            return {"chain": True}

        def validate():
            remaining.validate_prior_chain()
            return {"index": 10}

        def supervise(stage, *, popen):
            popen()
            return {"stage": stage}

        remaining.validate_prior_chain = chain
        remaining.validate_release = validate
        remaining.supervise_stage = supervise
        originals = (chain, validate, supervise, remaining.io.file_hash)
        seen = []

        class Hint:
            def hinted_hash(self, path, *, maximum, original_hash,
                            allowed_directories, audit):
                audit["eligible_files"] += 1
                audit["hinted_file_opens"] += 1
                audit["eligible_bytes"] += len(path)
                return original_hash(path, maximum=maximum)

        class Owner:
            def __call__(self):
                return object()

            def cleanup(self, observer, deadline):
                return {"contained": True, "direct_child_reaped": True,
                        "fallback_used": False, "remaining_members": [],
                        "errors": []}

        def execute():
            remaining.validate_release()
            remaining.supervise_stage("native")
            remaining.supervise_stage("readout")
            return {"status": "passed_numerical_software_only"}

        result, audit = core.scoped_old_execute(
            remaining, Hint(), [], execute,
            expected_hint_opens=2, expected_hint_bytes=6,
            preflight_check=lambda context, snapshot: seen.append(
                (context["index"], snapshot["eligible_files"])),
            native_stage_check=lambda: seen.append("pre_native"),
            stage_cleanup_callback=lambda stage, cleanup: seen.append(
                (stage, cleanup["direct_child_reaped"])),
            owner_factory=Owner, observer=lambda *_: (0, []))
        self.assertEqual(result["status"], "passed_numerical_software_only")
        self.assertEqual(audit["eligible_files"], 2)
        self.assertEqual(seen, [(10, 2), "pre_native", ("native", True),
                                ("readout", True)])
        self.assertEqual((remaining.validate_prior_chain,
                          remaining.validate_release,
                          remaining.supervise_stage,
                          remaining.io.file_hash), originals)

    def test_frozen_row_selection_and_exact_row11_geometry(self):
        runtime.validate_spec(row10.SPEC)
        self.assertEqual(row10.SPEC["run_id"], "tension:N24:S60:reference")
        self.assertEqual(runtime.policy(row10.SPEC)["expected_preflight_hint_opens"], 18)
        self.assertEqual(runtime.policy(row10.SPEC)["expected_preflight_hint_bytes"],
                         2_908_379_069)
        runtime.validate_spec(row11.SPEC)
        self.assertEqual(row11.SPEC["expected_preflight_hint_opens"], 20)
        self.assertEqual(row11.SPEC["expected_preflight_hint_bytes"],
                         2_908_379_069 + 118_531_657 + 74_276_117)
        self.assertEqual(ancestry.ROW10_DESCRIPTOR["row11_preflight_hint_bytes"],
                         row11.SPEC["expected_preflight_hint_bytes"])
        changed_row11 = copy.deepcopy(row11.SPEC)
        changed_row11["expected_preflight_hint_bytes"] += 1
        with self.assertRaisesRegex(ValueError, "exact observed row-10 geometry"):
            runtime.validate_spec(changed_row11)
        wrong = copy.deepcopy(row10.SPEC)
        wrong["run_id"] = "tension:N16:S60:reference"
        with self.assertRaisesRegex(ValueError, "frozen later tension"):
            runtime.validate_spec(wrong)

    def test_exact_once_predecessor_sidecar_arithmetic(self):
        old = old_baseline(10)
        verified = copy.deepcopy(old)
        charged = ledger.charge_old_baseline(old, verified, charges(old), 10)
        self.assertEqual(charged["authenticated_charge_order"], [8, 9])
        self.assertAlmostEqual(charged["incremental_prep_seconds"], 3.7)
        self.assertEqual(charged["incremental_output_bytes"], 2 * 1024**2)
        with self.assertRaisesRegex(ValueError, "Uncharged frozen old-chain"):
            ledger.charge_old_baseline(charged["charged_cumulative_ledger"],
                                       verified, charges(old), 10)
        omitted = charges(old)[:1]
        with self.assertRaisesRegex(ValueError, "Every declared"):
            ledger.charge_old_baseline(old, verified, omitted, 10)
        old11 = old_baseline(11)
        charged11 = ledger.charge_old_baseline(
            old11, copy.deepcopy(old11), charges(old11), 11)
        self.assertEqual(charged11["authenticated_charge_order"], [8, 9, 10])
        self.assertEqual(charged11["incremental_output_bytes"], 3 * 1024**2)
        with self.assertRaisesRegex(ValueError, "Every declared"):
            ledger.charge_old_baseline(old11, copy.deepcopy(old11),
                                       charges(old11)[:2], 11)

    def test_adapter_bytes_are_never_executed_during_selection(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            path = root / row10.SPEC["adapter_source"]
            path.parent.mkdir(parents=True)
            marker = root / "unexpected-execution"
            path.write_text("SPEC = " + repr(row10.SPEC) + "\n" +
                            f"open({str(marker)!r}, 'w').write('bad')\n")
            self.assertEqual(runtime._literal_adapter_spec(root,
                             row10.SPEC["adapter_source"]), row10.SPEC)
            self.assertFalse(marker.exists())

    def test_historical_git_blob_hashes_fail_closed(self):
        root = Path("/generated/root")
        paths = ("launchers/old.py", "scripts/old.py")
        blobs = {path: ("generated " + path).encode() for path in paths}
        pins = {path: hashlib.sha256(raw).hexdigest()
                for path, raw in blobs.items()}
        commit = "a" * 40

        def git(_, *args):
            raw = blobs[args[2].split(":", 1)[1]]
            return str(len(raw)).encode() if args[1] == "-s" else raw

        with mock.patch.object(runtime, "_git", side_effect=git):
            runtime._historical_source_blobs(root, commit, pins, paths)
            blobs[paths[0]] = b"changed"
            with self.assertRaisesRegex(ValueError, "blob differs"):
                runtime._historical_source_blobs(root, commit, pins, paths)
            blobs.pop(paths[0])
            with self.assertRaisesRegex(ValueError, "blob unavailable"):
                runtime._historical_source_blobs(root, commit, pins, paths)


if __name__ == "__main__":
    unittest.main()
