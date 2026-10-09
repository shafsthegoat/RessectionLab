"""Generate tiny saved receipts to test second-arm release without weights."""

import hashlib
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest import mock

import supervise_pair


class MatchedBaselineGateTests(unittest.TestCase):
    def setUp(self):
        self.temporary = TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        (self.root / "baseline").mkdir()
        self.contract_path = self.root / "pair-contract.json"
        self.contract_path.write_text("{}\n")
        self.contract = {
            "input_npy_sha256": "input",
            "model_file_sha256": {"fold_all/checkpoint_final.pth": "checkpoint"},
            "saved_baseline_reference": {"logits_sha256": "historical"},
        }
        self.contract_sha = hashlib.sha256(self.contract_path.read_bytes()).hexdigest()
        (self.root / "baseline/logits.npy").write_bytes(b"generated fixture bytes")
        logits_sha = hashlib.sha256(b"generated fixture bytes").hexdigest()
        self.result = {
            "arm": "baseline", "stage_zero_recomputation_enabled": False,
            "dense_module_hooks_attached": False, "contract_sha256": self.contract_sha,
            "input_npy_sha256": "input", "checkpoint_sha256_after": "checkpoint",
            "logits_npy_sha256": logits_sha, "logits_raw_sha256": "raw",
        }
        self.supervision = {
            "arm": "baseline", "contract_sha256": self.contract_sha,
            "accepted_for_feasibility": True, "exit_code": 0,
            "manual_attention_required": False, "residual_possible": False,
            "watchdog_reason": None, "post_guard_reason": None,
        }
        self.comparison = {
            "mode": "baseline", "gate_pass": True,
            "contract_sha256": self.contract_sha,
            "historical_logits_npy_sha256": "historical",
            "baseline_logits_npy_sha256": logits_sha,
            "baseline_logits_raw_sha256": "raw",
        }
        self.write_receipts()

    def write_receipts(self):
        for path, data in (("baseline/result.json", self.result),
                           ("baseline/supervision.json", self.supervision),
                           ("baseline-comparison.json", self.comparison)):
            (self.root / path).write_text(json.dumps(data))

    def validate(self):
        with mock.patch.object(supervise_pair, "HERE", self.root), \
                mock.patch.object(supervise_pair, "CONTRACT", self.contract_path):
            supervise_pair.validate_matched_baseline(self.contract)

    def test_exact_accepted_baseline_passes(self):
        self.validate()

    def test_numerical_failure_and_output_mutation_hold_second_arm(self):
        self.comparison["gate_pass"] = False
        self.write_receipts()
        with self.assertRaises(ValueError):
            self.validate()
        self.comparison["gate_pass"] = True
        self.write_receipts()
        (self.root / "baseline/logits.npy").write_bytes(b"changed")
        with self.assertRaises(ValueError):
            self.validate()

    def test_guard_failure_or_missing_comparison_hold_second_arm(self):
        self.supervision["residual_possible"] = True
        self.write_receipts()
        with self.assertRaises(ValueError):
            self.validate()
        (self.root / "baseline-comparison.json").unlink()
        with self.assertRaises(FileNotFoundError):
            self.validate()


if __name__ == "__main__":
    unittest.main()
