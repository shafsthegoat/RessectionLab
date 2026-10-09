"""Generated v2 boundary controls: no predecessor, measured data or solver."""
from __future__ import annotations

import importlib.util
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from launchers import hbe_v5_nocache_v1 as v1
from scripts import mechanics_hbe_v5_remaining_one_shot as remaining

HERE = Path(__file__).parent
spec = importlib.util.spec_from_file_location("candidate_hbe_nocache_v2",
                                               HERE / "hbe_v5_nocache_v2.py")
assert spec is not None and spec.loader is not None
v2 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(v2)


class V2BoundaryTests(unittest.TestCase):
    def test_v2_is_separate_ordinal8_namespace_and_keeps_old_native_caps(self):
        self.assertEqual(v1.POLICY["initial_available_percent_floor"], 55)
        self.assertEqual(v2.POLICY["initial_available_percent_floor"], 54)
        self.assertEqual(v2.POLICY["pre_native_available_percent_floor"], 45)
        self.assertEqual(v1.POLICY["pre_native_available_percent_floor"], 45)
        self.assertEqual(v1.NATIVE_OUTPUT, v2.NATIVE_OUTPUT)
        self.assertNotEqual(v1.SIDECAR, v2.SIDECAR)
        self.assertEqual(v2.SOURCES[1:], v1.SOURCES[1:])
        self.assertEqual(v2.SOURCES[0], "launchers/hbe_v5_nocache_v2.py")
        self.assertEqual(v2.POLICY["sidecar_output_cap_bytes"],
                         v1.POLICY["sidecar_output_cap_bytes"])
        self.assertEqual(v2.POLICY["expected_completed_hint_opens"],
                         v1.POLICY["expected_completed_hint_opens"])

    def _attempt_with_generated_host(self, available_percent: int):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary).resolve()
            old_path = root / "old-release.json"
            outer_path = root / "v2-envelope.json"
            envelope = {"source_commit": "a" * 40,
                        "extension_source_bindings": {
                            relative: "0" * 64 for relative in v2.SOURCES}}
            old_binding = (envelope, b"outer", (1, 2), {}, old_path,
                           b"inner", (3, 4))

            class HostSampler:
                def host(self):
                    return {"available_percent": available_percent,
                            "kernel_pressure_mask": 1}

            def original_runner(*args, **kwargs):
                raise RuntimeError("generated original runner reached")

            with patch.object(v2, "_read_exact_release", return_value=old_binding), \
                    patch.object(v2, "_load_sibling", side_effect=[
                        SimpleNamespace(), SimpleNamespace(FastDarwinSampler=HostSampler)]), \
                    patch.object(remaining, "execute", side_effect=original_runner):
                result = v2.execute_envelope(outer_path, root=root)
            self.assertEqual(result["status"], "failed_or_incomplete")
            self.assertTrue((root / v2.SIDECAR / "receipt.json").is_file())
            self.assertFalse((root / v2.NATIVE_OUTPUT).exists())
            self.assertEqual(result["native_calls_attempted"], 0)
            return result

    def test_initial_53_refuses_before_original_runner(self):
        result = self._attempt_with_generated_host(53)
        self.assertIn("54%/normal", result["failure"]["message"])

    def test_initial_54_reaches_original_runner_without_native_reservation(self):
        result = self._attempt_with_generated_host(54)
        self.assertIn("generated original runner reached", result["failure"]["message"])

    def test_failed_v1_policy_attempt_blocks_v2_before_reservation(self):
        for prior_kind, failure in (("directory", "Prior v1 no-cache policy attempt"),
                                    ("dangling_symlink", "Symlinked extension path refused")):
            with self.subTest(prior_kind=prior_kind), tempfile.TemporaryDirectory() as temporary:
                root = Path(temporary).resolve()
                prior_sidecar = root / v2.PRIOR_POLICY_SIDECAR
                prior_sidecar.parent.mkdir(parents=True)
                if prior_kind == "directory":
                    prior_sidecar.mkdir()
                else:
                    prior_sidecar.symlink_to(root / "missing-v1-attempt")
                envelope = {"source_commit": "a" * 40,
                            "extension_source_bindings": {
                                relative: "0" * 64 for relative in v2.SOURCES}}
                old_binding = (envelope, b"outer", (1, 2), {}, root / "inner.json",
                               b"inner", (3, 4))
                with patch.object(v2, "_read_exact_release", return_value=old_binding), \
                        patch.object(v2, "_load_sibling", side_effect=[
                            SimpleNamespace(), SimpleNamespace(FastDarwinSampler=object)]), \
                        patch.object(remaining, "execute") as original_runner:
                    with self.assertRaisesRegex(ValueError, failure):
                        v2.execute_envelope(root / "outer.json", root=root)
                original_runner.assert_not_called()
                self.assertFalse((root / v2.SIDECAR).exists())
                self.assertFalse((root / v2.NATIVE_OUTPUT).exists())


if __name__ == "__main__":
    unittest.main()
