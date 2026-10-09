"""Tiny generated-only support boundary controls; no patient payloads."""

import unittest
import numpy as np

from resectionlab.scan_support_contract import diagnostic_display_state, encode_support_bits, unreviewed_input_domain


class SupportContractTest(unittest.TestCase):
    def test_scan_fov_and_estimated_mask_are_distinct(self):
        t1 = np.ones((3, 3, 3), bool)
        flair = t1.copy()
        flair[0, 0, 0] = False
        mask = np.zeros_like(t1)
        mask[1, 1, 1] = True
        bits = encode_support_bits(t1, flair, mask)
        self.assertEqual(bits[0, 0, 0], 1)  # T1 only, not observed by FLAIR.
        self.assertEqual(bits[2, 2, 2], 3)  # Both scans, anatomy unknown.
        self.assertEqual(bits[1, 1, 1], 7)  # Both scans and estimated mask.
        self.assertEqual(unreviewed_input_domain(bits).sum(), 1)

    def test_mask_beyond_original_fov_abstains_without_clipping(self):
        t1 = np.ones((3, 3, 3), bool)
        flair = t1.copy()
        flair[1, 1, 1] = False
        mask = np.zeros_like(t1)
        mask[1, 1, 1] = True
        with self.assertRaisesRegex(ValueError, "extends outside"):
            encode_support_bits(t1, flair, mask)

    def test_future_negative_output_does_not_call_unknown_absent(self):
        t1 = flair = np.ones((3, 3, 3), bool)
        mask = np.zeros((3, 3, 3), bool)
        mask[1, 1, 1] = True
        bits = encode_support_bits(t1, flair, mask)
        candidate = np.zeros((3, 3, 3), bool)
        candidate[1, 1, 1] = True
        candidate[0, 0, 0] = True  # Outside domain cannot be represented as truth.
        predicted = np.zeros_like(candidate)
        predicted[1, 1, 1] = True
        state, excluded_positive = diagnostic_display_state(candidate, bits, predicted)
        self.assertEqual(state[1, 1, 1], 1)
        self.assertEqual(state[0, 0, 0], -1)
        self.assertEqual(excluded_positive, 1)
        candidate[1, 1, 1] = False
        self.assertEqual(diagnostic_display_state(candidate, bits, predicted)[0][1, 1, 1], 0)
        predicted[1, 1, 1] = False
        self.assertEqual(diagnostic_display_state(candidate, bits, predicted)[0][1, 1, 1], -1)

    def test_invalid_mask_without_both_scans_and_prediction_coverage_reject(self):
        bits = np.zeros((3, 3, 3), np.uint8)
        bits[1, 1, 1] = 5  # Estimated mask but missing FLAIR FOV.
        with self.assertRaisesRegex(ValueError, "support bits"):
            unreviewed_input_domain(bits)
        bits[1, 1, 1] = 7
        invalid_prediction_coverage = np.ones((3, 3, 3), bool)
        with self.assertRaisesRegex(ValueError, "coverage extends"):
            diagnostic_display_state(np.zeros((3, 3, 3), bool), bits,
                                     invalid_prediction_coverage)

    def test_nonbinary_or_target_defined_inputs_reject(self):
        ones = np.ones((3, 3, 3), bool)
        bad = np.full((3, 3, 3), 2)
        with self.assertRaisesRegex(ValueError, "binary"):
            encode_support_bits(ones, ones, bad)
        with self.assertRaises(TypeError):
            encode_support_bits(ones, ones, ones, target_mask=ones)



if __name__ == "__main__":
    unittest.main()
