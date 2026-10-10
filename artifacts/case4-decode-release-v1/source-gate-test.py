"""Source-only decode gate controls; no patient, output or model array reads."""

import importlib.util
import json
import os
from pathlib import Path
import unittest
from unittest.mock import patch

HERE = Path(__file__).resolve().parent


class StopAtAudit(Exception):
    pass


class DecodeGateTests(unittest.TestCase):
    def test_worker_reaches_bound_independent_audit_before_payload(self):
        spec = importlib.util.spec_from_file_location("case4_decode_worker_control", HERE / "pair_worker.py")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        contract = json.loads((HERE / "pair-contract.json").read_text())
        observed = []

        def source_only_intercept(path, digest):
            observed.append(path)
            if path == module.AUDIT:
                self.assertEqual(digest, contract["independent_forward_audit"]["sha256"])
                raise StopAtAudit()

        with patch.object(module, "exact_file", side_effect=source_only_intercept), \
             patch.dict(os.environ, {"RESECTIONLAB_PAIR_ARM": "case4"}):
            with self.assertRaises(StopAtAudit):
                module.main()
        self.assertEqual(observed[-1], module.AUDIT)
        self.assertEqual(contract["independent_forward_audit"]["path"],
                         str(module.AUDIT.relative_to(module.ROOT)))
        self.assertFalse((HERE / "case4").exists())


if __name__ == "__main__":
    unittest.main()
