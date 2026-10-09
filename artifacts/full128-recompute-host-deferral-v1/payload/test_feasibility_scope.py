"""Generated-only prospective full128 scope checks; no checkpoint/model read."""

import hashlib
import json
from pathlib import Path
import unittest


HERE = Path(__file__).resolve().parent
OLD = HERE.parent / "gliomoda-full128-lazy-feasibility-v2"


class ScopeTests(unittest.TestCase):
    def test_original_full128_input_and_unchanged_resource_rules(self):
        contract = json.loads((HERE / "pair-contract.json").read_text())
        original = json.loads((OLD / "pair-contract.json").read_text())
        for key in ("input_npy_sha256", "input_raw_sha256", "input_shape",
                    "output_shape", "model_file_sha256", "model_patch_size",
                    "output_depth_tile", "process_group_rss_cap_kib",
                    "wall_time_cap_seconds", "host_preflight_seconds",
                    "host_abort_kernel_pressure_masks", "host_abort_min_free_percent",
                    "host_abort_swap_growth_mib", "host_abort_free_drop_points",
                    "host_sample_target_interval_seconds", "host_sample_max_interval_seconds"):
            self.assertEqual(contract[key], original[key], key)
        self.assertEqual((HERE / "generated-input.npy").read_bytes(),
                         (OLD / "generated-input.npy").read_bytes())
        self.assertEqual(contract["arm_order"], ["recompute"])
        self.assertTrue(contract["single_full_patch_forward"])
        self.assertFalse(contract["native_arm_enabled"])
        self.assertIn("stop further full128", contract["stop_after_same_early_pressure_failure"])

    def test_only_reviewed_recompute_change_is_requested(self):
        worker = (HERE / "pair_worker.py").read_text()
        self.assertIn("attach_recomputed_stage_zero(", worker)
        self.assertIn("expected_encoder_stages=6, require_lazy_decoder=True", worker)
        self.assertNotIn("register_forward_hook", worker)
        self.assertNotIn("register_forward_pre_hook", worker)
        contract = json.loads((HERE / "pair-contract.json").read_text())
        self.assertEqual(contract["full128_parity_claim"],
                         "unavailable_no_accepted_native_full128_reference")
        for name, expected in contract["runner_source_sha256"].items():
            self.assertEqual(hashlib.sha256((HERE / name).read_bytes()).hexdigest(),
                             expected, name)


if __name__ == "__main__":
    unittest.main()
