"""Focused generated-only tests for the optional low-memory Conv3d adapter."""
import subprocess
import sys
from pathlib import Path

import pytest

from resectionlab import low_memory_conv3d as adapter

TARGET_MODULE_PATH = Path(adapter.__file__).resolve()


def test_import_succeeds_without_site_packages_or_torch():
    code = ("import importlib.util; "
            f"s=importlib.util.spec_from_file_location('low_memory_conv3d',{str(TARGET_MODULE_PATH)!r}); "
            "m=importlib.util.module_from_spec(s); s.loader.exec_module(m); "
            "assert callable(m.conv3d_tiled_depth) and callable(m.attach_tiled_conv3d)")
    result = subprocess.run([sys.executable, "-S", "-c", code],
                            capture_output=True, text=True, timeout=5)
    assert result.returncode == 0, result.stderr


@pytest.mark.parametrize("shape,weight_shape,stride,padding,dilation,groups,tile", [
    ((1, 2, 9, 11, 13), (4, 2, 3, 3, 3), (1, 1, 1), (1, 1, 1), (1, 1, 1), 1, 2),
    ((1, 4, 10, 12, 14), (8, 4, 3, 3, 3), (2, 2, 2), (1, 1, 1), (1, 1, 1), 1, 3),
    ((1, 4, 11, 12, 13), (6, 2, 3, 2, 3), (2, 1, 2), (2, 0, 1), (2, 1, 1), 2, 2),
])
def test_generated_operator_matches_native(shape, weight_shape, stride, padding, dilation, groups, tile):
    torch = pytest.importorskip("torch")
    with torch.random.fork_rng(devices=[]), torch.inference_mode():
        torch.manual_seed(20261009)
        x = torch.randn(shape, dtype=torch.float32)
        weight = torch.randn(weight_shape, dtype=torch.float32)
        bias = torch.randn(weight_shape[0], dtype=torch.float32)
        native = torch.nn.functional.conv3d(x, weight, bias, stride, padding, dilation, groups)
        tiled = adapter.conv3d_tiled_depth(x, weight, bias, stride, padding, dilation, groups, tile)
    assert native.shape == tiled.shape
    torch.testing.assert_close(tiled, native, rtol=1e-3, atol=1e-3)


def test_wrapper_preserves_all_aliases_and_full_feature_map_normalization():
    torch = pytest.importorskip("torch")
    class Tiny(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.conv = torch.nn.Conv3d(2, 4, 3, padding=1, bias=True)
            self.norm = torch.nn.InstanceNorm3d(4, affine=True)
            self.alias = self.conv
        def forward(self, x):
            return self.norm(self.conv(x))
    with torch.random.fork_rng(devices=[]), torch.inference_mode():
        torch.manual_seed(20261009)
        network = Tiny().eval()
        values = torch.randn((1, 2, 7, 9, 11), dtype=torch.float32)
        before = adapter.identity_snapshot(network)
        native = network(values)
        receipt = adapter.attach_tiled_conv3d(network, output_depth_tile=2)
        tiled = network(values)
        after = adapter.identity_snapshot(network)
    assert before == after
    assert receipt["wrapped_unique_conv3d_modules"] == 1
    assert receipt["parameter_state_module_identities_preserved"] is True
    assert isinstance(network.norm, torch.nn.InstanceNorm3d)
    torch.testing.assert_close(tiled, native, rtol=1e-3, atol=1e-3)
    with pytest.raises(ValueError, match="custom/previously wrapped"):
        adapter.attach_tiled_conv3d(network)


@pytest.mark.parametrize("padding", ["same", 1])
def test_unsupported_padding_modes_rejected(padding):
    torch = pytest.importorskip("torch")
    mode = "reflect" if padding == 1 else "zeros"
    module = torch.nn.Conv3d(2, 2, 3, padding=padding, padding_mode=mode).eval()
    with pytest.raises(ValueError, match="nonzero/implicit"):
        adapter.attach_tiled_conv3d(module)


def test_bias_free_single_tile_and_inference_guards():
    torch = pytest.importorskip("torch")
    with torch.random.fork_rng(devices=[]), torch.inference_mode():
        torch.manual_seed(17)
        values = torch.randn((1, 2, 5, 7, 9), dtype=torch.float32)
        weights = torch.randn((3, 2, 3, 3, 3), dtype=torch.float32)
        native = torch.nn.functional.conv3d(values, weights, None, padding=1)
        tiled = adapter.conv3d_tiled_depth(values, weights, None, padding=1,
                                           output_depth_tile=99)
    torch.testing.assert_close(tiled, native, rtol=1e-3, atol=1e-3)
    training = torch.nn.Conv3d(2, 2, 3, padding=1)
    with pytest.raises(ValueError, match="eval mode"):
        adapter.attach_tiled_conv3d(training)
    with pytest.raises(ValueError, match="CPU FP32"):
        adapter.attach_tiled_conv3d(training.double().eval())


def test_excessive_depth_padding_fails_before_slicing_or_partial_wrapping():
    torch = pytest.importorskip("torch")
    values = torch.zeros((1, 2, 3, 5, 5), dtype=torch.float32)
    weights = torch.zeros((2, 2, 3, 3, 3), dtype=torch.float32)
    with pytest.raises(ValueError, match="depth padding exceeds kernel support"):
        adapter.conv3d_tiled_depth(values, weights, padding=(3, 1, 1), output_depth_tile=1)
    network = torch.nn.Sequential(
        torch.nn.Conv3d(2, 2, 3, padding=1),
        torch.nn.Conv3d(2, 2, 3, padding=(3, 1, 1)),
    ).eval()
    before = adapter.identity_snapshot(network)
    with pytest.raises(ValueError, match="depth padding exceeds kernel support"):
        adapter.attach_tiled_conv3d(network)
    assert adapter.identity_snapshot(network) == before
    assert all("forward" not in module.__dict__ for module in network if isinstance(module, torch.nn.Conv3d))
