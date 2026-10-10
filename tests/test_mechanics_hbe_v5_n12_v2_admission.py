"""Portable compact-evidence controls; optional local saved-event check.

The portable tests read only tracked, small evidence copies. The historical
ignored output/release paths are used only by the explicitly skipped-when-
absent local check. No test invokes FEBio or hashes bulk predecessor output.
"""
from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from scripts import mechanics_hbe_v5_n12_v2_admission as admission

ROOT = Path(__file__).resolve().parents[1]
EVIDENCE = ROOT / "artifacts/hbe-v5-tension-n12-nocache-v2-result-v1"
COPIES = {
    admission.SIDECAR: "v2-sidecar-receipt.json",
    admission.NATIVE: "native-receipt.json",
    admission.INNER: "inner-release.json",
    admission.OUTER: "outer-envelope.json",
    admission.REVIEW: "RESULT_METADATA.json",
}


def exact_fixture_values():
    values = []
    for original, copy_name in COPIES.items():
        raw = (EVIDENCE / copy_name).read_bytes()
        if hashlib.sha256(raw).hexdigest() != admission.SHA[original]:
            raise ValueError("Tracked compact copy differs from original bound bytes")
        values.append(json.loads(raw))
    report = (EVIDENCE / "RESULT.md").read_bytes()
    if hashlib.sha256(report).hexdigest() != admission.REVIEW_REPORT_SHA:
        raise ValueError("Tracked independent report differs from bound bytes")
    return values


def historical_root(native: dict) -> Path:
    """Recover the immutable run path recorded by the saved native receipt."""
    recorded = native["release_path"]
    suffix = "/" + admission.INNER
    if not isinstance(recorded, str) or not recorded.endswith(suffix):
        raise ValueError("Saved native receipt does not name the bound release")
    return Path(recorded[:-len(suffix)])


def exact_fixture_result():
    values = exact_fixture_values()
    return admission._saved_fields(*values, historical_root(values[1]))


class AdmissionTests(unittest.TestCase):
    def test_tracked_compact_event_only_admits_numerical_ancestry(self):
        result = exact_fixture_result()
        self.assertEqual(result["ordinal"], 8)
        self.assertIsNone(result["physical_validation_pass"])
        self.assertFalse(result["patient_or_measured_response_accessed"])

    @unittest.skipUnless(
        all((ROOT / path).is_file() for path in
            (admission.SIDECAR, admission.NATIVE, admission.INNER, admission.OUTER)),
        "Local historical saved output/release paths are absent")
    def test_local_saved_event_path_and_historical_git_identity(self):
        """Integration check only; expected to skip in a clean checkout."""
        self.assertEqual(admission.verify_exact(root=ROOT), exact_fixture_result())

    def test_generated_semantic_mutations_refuse(self):
        mutations = {
            "sidecar_status": lambda s, n, i, o, r: s.__setitem__("status", "failed_or_incomplete"),
            "native_status": lambda s, n, i, o, r: n.__setitem__("status", "failed_or_incomplete"),
            "low_initial_host": lambda s, n, i, o, r: s["host_before_validation"].__setitem__("available_percent", 53),
            "missing_hint_open": lambda s, n, i, o, r: s["final_hint_audit"].__setitem__("hinted_file_opens", 27),
            "unexpected_hint_path": lambda s, n, i, o, r: s["preflight_hint_audit"]["hinted_file_paths"].__setitem__(0, str(ROOT / "unrelated.log")),
            "cleanup_fallback": lambda s, n, i, o, r: s["owned_stage_cleanup"]["native"].__setitem__("fallback_used", True),
            "extra_charge": lambda s, n, i, o, r: s["extension_resource_charge"].__setitem__("charged_current_output_bytes", s["extension_resource_charge"]["charged_current_output_bytes"] + 1),
            "wrapper_before_wall": lambda s, n, i, o, r: s.__setitem__("extension_before_old_execute_wall_seconds", 45),
            "wrapper_after_wall": lambda s, n, i, o, r: s.__setitem__("extension_after_old_execute_wall_seconds", 45),
            "wrapper_peak_rss": lambda s, n, i, o, r: s.__setitem__("extension_self_peak_rss_bytes_terminal", 3 * 1024**3 + 1),
            "review_readout_mismatch": lambda s, n, i, o, r: r.__setitem__("readout_sha256", "0" * 64),
            "review_inventory_mismatch": lambda s, n, i, o, r: r.__setitem__("output_bindings", {}),
            "wrong_outer_policy": lambda s, n, i, o, r: o["policy"].__setitem__("initial_available_percent_floor", 50),
            "unreviewed_patient_claim": lambda s, n, i, o, r: r.__setitem__("patient_data_accessed", True),
        }
        for name, mutate in mutations.items():
            with self.subTest(name=name):
                values = copy.deepcopy(exact_fixture_values())
                mutate(*values)
                with self.assertRaises(ValueError):
                    admission._saved_fields(*values, historical_root(values[1]))

    def test_changed_or_linked_small_receipt_refuses(self):
        raw = (EVIDENCE / COPIES[admission.SIDECAR]).read_bytes()
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            target = root / admission.SIDECAR
            target.parent.mkdir(parents=True)
            target.write_bytes(raw + b"\n")
            with self.assertRaisesRegex(ValueError, "bytes changed"):
                admission._small_bytes(root, admission.SIDECAR,
                                       admission.SHA[admission.SIDECAR])
            target.unlink()
            target.symlink_to(EVIDENCE / COPIES[admission.SIDECAR])
            with self.assertRaisesRegex(ValueError, "linked"):
                admission._small_bytes(root, admission.SIDECAR,
                                       admission.SHA[admission.SIDECAR])

    def test_future_rows_carry_one_surcharge_without_double_charge(self):
        exact = exact_fixture_result()
        delta = exact["surcharge"]
        for index in (9, 10, 11):
            with self.subTest(index=index):
                previous = {"sha256": ["0" * 64 for _ in range(index)],
                            "native_seconds": 100.0 + 50 * (index - 9),
                            "readout_seconds": 50.0 + 5 * (index - 9),
                            "prep_seconds": 20.0 + 3 * (index - 9),
                            "output_bytes": 1000 + 100 * (index - 9),
                            "native_calls": index,
                            "supplement_readout_seconds": 1.0,
                            "supplement_prep_seconds": 1.0,
                            "supplement_output_bytes": 500,
                            "supplement_replay_calls": 1,
                            "combined_wall_seconds": 172.0 + 58 * (index - 9),
                            "combined_output_bytes": 1500 + 100 * (index - 9)}
                previous["sha256"][8] = admission.SHA[admission.NATIVE]
                ledger = admission._arithmetic_surcharge(previous, exact, index)
                charged = ledger["charged_cumulative_ledger"]
                self.assertAlmostEqual(charged["prep_seconds"] - previous["prep_seconds"],
                                       delta["prep_wall_seconds"])
                self.assertEqual(charged["output_bytes"] - previous["output_bytes"],
                                 1024**2)
                self.assertEqual(charged["native_calls"], previous["native_calls"])
                with self.assertRaisesRegex(ValueError, "uncharged old predecessor"):
                    admission._arithmetic_surcharge(charged, exact, index)
                if index == 9:
                    self.assertEqual(admission.continuation_ledger(previous, exact, index),
                                     ledger)
                else:
                    with self.assertRaisesRegex(ValueError, "Later wrapper charges"):
                        admission.continuation_ledger(previous, exact, index)

    def test_future_assessment_combines_sidecar_and_old_chain_but_holds_execution(self):
        exact = exact_fixture_result()
        previous = {"sha256": ["0" * 64 for _ in range(9)],
                    "native_seconds": 100., "readout_seconds": 50.,
                    "prep_seconds": 20., "output_bytes": 1000,
                    "native_calls": 9, "supplement_readout_seconds": 1.,
                    "supplement_prep_seconds": 1., "supplement_output_bytes": 500,
                    "supplement_replay_calls": 1,
                    "combined_wall_seconds": 172., "combined_output_bytes": 1500}
        previous["sha256"][8] = admission.SHA[admission.NATIVE]
        bindings = [{"run_id": admission.remaining.ORDER[i],
                     "path": admission.remaining.receipt_path(i),
                     "sha256": previous["sha256"][i]} for i in range(9)]
        release = {"ordinal": 9, "run_id": admission.remaining.ORDER[9],
                   "prior_receipts": bindings}
        with patch.object(admission, "verify_exact", return_value=exact) as v2_gate, \
                patch.object(admission.remaining, "validate_release",
                             return_value={"index": 9, "previous": previous}) as old_gate:
            result = admission.assess_future_release(release, root=ROOT)
            self.assertEqual(result["status"],
                             "hold_execution_sidecar_integration_required")
            self.assertEqual(v2_gate.call_count, 2)
            old_gate.assert_called_once()
            release["prior_receipts"][8]["sha256"] = "1" * 64
            with self.assertRaisesRegex(ValueError, "exact v2 predecessor"):
                admission.assess_future_release(release, root=ROOT)
            old_gate.assert_called_once()


if __name__ == "__main__":
    unittest.main()
