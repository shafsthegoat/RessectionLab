"""Tiny generated-only controls; never loads a model or patient input."""

import unittest

import torch
import torch.nn.functional as F

from inplace_instance_norm import normalize_private_conv_output_


class InplaceInstanceNormTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)

    def _compare(self, values, weight, bias, eps=1e-5):
        with torch.inference_mode():
            reference = F.instance_norm(values, None, None, weight, bias, True, 0.1, eps)
            candidate = values.clone()
            pointer = candidate.data_ptr()
            saved_weight = weight.clone()
            saved_bias = bias.clone()
            result = normalize_private_conv_output_(candidate, weight, bias, eps=eps)
            self.assertEqual(result.data_ptr(), pointer)
            self.assertTrue(torch.equal(weight, saved_weight))
            self.assertTrue(torch.equal(bias, saved_bias))
            self.assertTrue(torch.isfinite(result).all())
            return {
                "max_abs_error": torch.max(torch.abs(result - reference)).item(),
                "near_zero_reference": int((torch.abs(reference) < 0.001).sum().item()),
                "sign_mismatches": int(((result > 0) != (reference > 0)).sum().item()),
            }

    def test_generated_shapes_and_affine(self):
        cases = []
        for seed, shape in ((10, (1, 4, 9, 11, 13)), (11, (2, 3, 6, 7, 8)),
                            (12, (1, 8, 8, 8, 8))):
            torch.manual_seed(seed)
            values = torch.randn(shape, dtype=torch.float32)
            weight = torch.randn(shape[1], dtype=torch.float32)
            bias = torch.randn(shape[1], dtype=torch.float32)
            cases.append(self._compare(values, weight, bias))
        self.assertLessEqual(max(case["max_abs_error"] for case in cases), 1e-3)
        print({"generated_random_cases": cases})

    def test_constant_and_low_variance(self):
        values = torch.full((1, 2, 4, 5, 6), 3.0)
        values[0, 1, 1, 2, 3] += 1e-4
        report = self._compare(values, torch.tensor([1.2, -0.8]), torch.tensor([0.1, -0.2]))
        self.assertLessEqual(report["max_abs_error"], 1e-3)
        print({"generated_low_variance": report})

    def test_large_offset_is_a_numerical_counterexample(self):
        # This is a negative control, not an acceptance tolerance. Native
        # InstanceNorm and var_mean-based in-place arithmetic round differently
        # on low-variance maps with a large positive offset.
        torch.manual_seed(1)
        values = torch.randn((1, 4, 4, 5, 6)) * 0.1 + 1e6
        weight = torch.tensor([1.0, -0.5, 2.0, 0.1])
        bias = torch.zeros(4)
        report = self._compare(values, weight, bias)
        self.assertGreater(report["max_abs_error"], 1e-3)
        self.assertGreater(report["sign_mismatches"], 0)
        print({"generated_large_offset_counterexample": report})

    def test_no_implicit_ownership_proof(self):
        base = torch.randn((1, 2, 4, 5, 6))
        viewed = base[:, :, :, :, :]
        with torch.inference_mode():
            with self.assertRaisesRegex(ValueError, "view"):
                normalize_private_conv_output_(viewed, torch.ones(2), torch.zeros(2))
            # A detached alias has no visible _base. The operator cannot prove
            # exclusivity; graph-level attachment must reject this situation.
            detached_alias = base.detach()
            self.assertEqual(detached_alias.data_ptr(), base.data_ptr())
            self.assertIsNone(detached_alias._base)

    def test_fail_closed_contracts(self):
        x = torch.randn((1, 2, 4, 5, 6))
        w = torch.ones(2)
        b = torch.zeros(2)
        with self.assertRaisesRegex(ValueError, "inference_mode"):
            normalize_private_conv_output_(x.clone(), w, b)
        with torch.inference_mode():
            bad = ((x.double(), w, b), (x, w[:1], b),
                   (x[:, :, :, :, 0], w, b), (x, w, torch.full((2,), float("nan"))))
            for arguments in bad:
                with self.assertRaises(ValueError):
                    normalize_private_conv_output_(*arguments)


if __name__ == "__main__":
    unittest.main()
