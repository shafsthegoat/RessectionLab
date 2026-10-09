"""Small generated comparator semantics checks; no checkpoint or model."""
import unittest

import numpy as np

from compare_pair import labels, pair_metrics


class ComparatorTests(unittest.TestCase):
    def test_sequential_region_overwrite_order(self):
        logits = np.zeros((1, 3, 1, 1, 4), dtype=np.float32)
        logits[0, :, 0, 0, :] = np.array([
            [1, 1, 1, -1],
            [-1, 1, 1, -1],
            [-1, -1, 1, 1],
        ], dtype=np.float32)
        self.assertEqual(labels(logits).ravel().tolist(), [2, 1, 3, 3])

    def test_tolerance_and_exact_label_disagreement_are_separate(self):
        left = np.full((1, 3, 1, 1, 2), -2.0, dtype=np.float32)
        right = left.copy()
        left[0, 0, 0, 0, 0] = np.float32(0.0001)
        right[0, 0, 0, 0, 0] = np.float32(-0.0001)
        report = pair_metrics(left, right, atol=0.001, rtol=0.001)
        self.assertTrue(report["all_logits_within_predeclared_tolerance"])
        self.assertEqual(report["per_region_sign_disagreements"], [1, 0, 0])
        self.assertEqual(report["decoded_label_disagreements"], 1)
        self.assertEqual(report["per_region_near_zero_counts_left"], [1, 0, 0])


if __name__ == "__main__":
    unittest.main()
