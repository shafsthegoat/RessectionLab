"""Exercise the real worker entry through early source binding, without weights."""

import ast
from contextlib import redirect_stdout
import importlib.util
import io
import os
from pathlib import Path
import symtable
import sys
import types
import unittest
from unittest import mock


HERE = Path(__file__).resolve().parent


class BeforeCheckpointSentinel(Exception):
    pass


def fake_imports():
    numpy = types.ModuleType("numpy")
    numpy.__version__ = "2.0.1"
    torch = types.ModuleType("torch")
    torch.__version__ = "2.14.1"
    torch.set_num_threads = lambda count: None
    torch.set_num_interop_threads = lambda count: None

    def stop_before_input_or_checkpoint(seed):
        assert seed == 0
        raise BeforeCheckpointSentinel()

    torch.manual_seed = stop_before_input_or_checkpoint
    modules = {"numpy": numpy, "torch": torch}
    names = ["nnunetv2", "nnunetv2.utilities",
             "nnunetv2.utilities.get_network_from_plans",
             "nnunetv2.utilities.label_handling",
             "nnunetv2.utilities.label_handling.label_handling",
             "nnunetv2.utilities.plans_handling",
             "nnunetv2.utilities.plans_handling.plans_handler"]
    for name in names:
        module = types.ModuleType(name)
        if name in ("nnunetv2", "nnunetv2.utilities",
                    "nnunetv2.utilities.label_handling", "nnunetv2.utilities.plans_handling"):
            module.__path__ = []
        modules[name] = module
    modules["nnunetv2.utilities.get_network_from_plans"].get_network_from_plans = object()
    modules["nnunetv2.utilities.label_handling.label_handling"].determine_num_input_channels = object()
    modules["nnunetv2.utilities.plans_handling.plans_handler"].PlansManager = object()
    return modules


class MainEntryTests(unittest.TestCase):
    def test_sampler_is_global_within_real_main_function(self):
        source = (HERE / "pair_worker.py").read_text()
        main = next(child for child in symtable.symtable(source, "pair_worker.py", "exec").get_children()
                    if child.get_name() == "main")
        sampler = main.lookup("FastDarwinSampler")
        self.assertTrue(sampler.is_global())
        self.assertFalse(sampler.is_local())
        self.assertTrue(any(isinstance(node, ast.FunctionDef) and node.name == "main"
                            for node in ast.parse(source).body))

    def test_actual_main_reaches_post_binding_sentinel_before_any_model_read(self):
        spec = importlib.util.spec_from_file_location("early_worker", HERE / "pair_worker.py")
        worker = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(worker)
        original_sha = worker.sha256_file
        hashed = []

        def bounded_hash(path):
            path = Path(path).resolve()
            if path == worker.INPUT or path == worker.INPUT_RECEIPT or worker.MODEL in path.parents:
                raise AssertionError("input/model bytes read before early-entry sentinel")
            hashed.append(path)
            return original_sha(path)

        for arm in ("recompute",):
            with self.subTest(arm=arm):
                hashed.clear()
                output = io.StringIO()
                with mock.patch.dict(sys.modules, fake_imports()), \
                        mock.patch.dict(os.environ, {"PYTORCH_ENABLE_MPS_FALLBACK": "0",
                                                     "RESECTIONLAB_PAIR_ARM": arm}), \
                        mock.patch.object(worker, "sha256_file", side_effect=bounded_hash), \
                        redirect_stdout(output):
                    with self.assertRaises(BeforeCheckpointSentinel):
                        worker.main()
                self.assertEqual(hashed, [HERE / "pair_worker.py", HERE / "darwin_fast_sampler.py",
                                          HERE / "pair-contract.json"])
                self.assertIn('"phase": "before_imports"', output.getvalue())


if __name__ == "__main__":
    unittest.main()
