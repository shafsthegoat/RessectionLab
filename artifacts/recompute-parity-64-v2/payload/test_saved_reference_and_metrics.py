"""Saved-only baseline binding and tiny numerical gates; never loads weights."""

import json
from pathlib import Path
import unittest

import numpy as np

from compare_recompute import numerical_metrics
from supervise_pair import validate_saved_baseline


HERE = Path(__file__).resolve().parent


class SavedReferenceAndMetricsTests(unittest.TestCase):
    def test_exact_accepted_baseline_is_bound_before_any_model_call(self):
        contract = json.loads((HERE / "pair-contract.json").read_text())
        validate_saved_baseline(contract)
        altered = json.loads(json.dumps(contract))
        altered["saved_baseline_reference"]["result_sha256"] = "0" * 64
        with self.assertRaisesRegex(ValueError, "baseline bytes changed"):
            validate_saved_baseline(altered)

    def test_logits_tolerance_and_exact_sequential_labels_are_separate(self):
        baseline = np.full((1, 3, 2, 2, 2), -1.0, dtype=np.float32)
        candidate = baseline.copy()
        candidate[0, 0, 0, 0, 0] = -0.9995
        metrics = numerical_metrics(candidate, baseline, 0.001, 0.001)
        self.assertTrue(metrics["all_logits_within_predeclared_tolerance"])
        self.assertEqual(metrics["decoded_label_disagreements"], 0)
        candidate[0, 0, 0, 0, 0] = 1e-5
        metrics = numerical_metrics(candidate, baseline, 0.001, 0.001)
        self.assertFalse(metrics["all_logits_within_predeclared_tolerance"])
        self.assertEqual(metrics["decoded_label_disagreements"], 1)

    def test_region_overwrite_order_detects_threshold_crossing(self):
        baseline = np.full((1, 3, 1, 1, 1), -0.5, dtype=np.float32)
        baseline[0, 0, 0, 0, 0] = 0.0001  # WT -> class 2
        baseline[0, 1, 0, 0, 0] = -0.0001
        candidate = baseline.copy()
        candidate[0, 1, 0, 0, 0] = 0.0001  # TC overwrites WT -> class 1
        metrics = numerical_metrics(candidate, baseline, 0.001, 0.001)
        self.assertTrue(metrics["all_logits_within_predeclared_tolerance"])
        self.assertEqual(metrics["decoded_label_disagreements"], 1)
        self.assertEqual(metrics["per_region_sign_disagreements"], [0, 1, 0])


if __name__ == "__main__":
    unittest.main()
