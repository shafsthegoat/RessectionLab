"""One tiny untrained six-stage CPU graph parity control; no checkpoint or patient data."""

import hashlib
import json
from pathlib import Path
import sys
import time

import torch


ROOT = Path(__file__).resolve().parents[3]
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "build/limited-input-guard-design/lazy-final-decoder-concat-v1"))
from dynamic_network_architectures.architectures.unet import PlainConvUNet
from resectionlab.low_memory_conv3d import attach_tiled_conv3d, identity_snapshot
from lazy_concat import attach_lazy_final_decoder_concat
from recompute_skip import attach_recomputed_stage_zero


def main():
    torch.set_num_threads(1)
    torch.manual_seed(781)
    shape = [1, 2, 32, 32, 64]
    features = [4, 4, 4, 4, 4, 4]
    network = PlainConvUNet(
        input_channels=2, n_stages=6, features_per_stage=features,
        conv_op=torch.nn.Conv3d, kernel_sizes=3, strides=[1, 2, 2, 2, 2, 2],
        n_conv_per_stage=[2] * 6, num_classes=3,
        n_conv_per_stage_decoder=[2] * 5, conv_bias=True,
        norm_op=torch.nn.InstanceNorm3d,
        norm_op_kwargs={"eps": 1e-5, "affine": True}, dropout_op=None,
        nonlin=torch.nn.LeakyReLU, nonlin_kwargs={"inplace": True},
        deep_supervision=False).eval()
    generated = torch.randn(shape, dtype=torch.float32)
    original = generated.clone()
    tiled = attach_tiled_conv3d(network, output_depth_tile=1)
    lazy = attach_lazy_final_decoder_concat(network, output_depth_tile=1, expected_stages=5)
    before = identity_snapshot(network)
    state_before = {name: tensor.clone() for name, tensor in network.state_dict().items()}
    with torch.inference_mode():
        start = time.perf_counter()
        baseline = network(generated)
        baseline_seconds = time.perf_counter() - start
    report = attach_recomputed_stage_zero(network, expected_encoder_stages=6,
                                          require_lazy_decoder=True)
    if identity_snapshot(network) != before:
        raise RuntimeError("attachment altered model aliases")
    with torch.inference_mode():
        start = time.perf_counter()
        candidate = network(generated)
        candidate_seconds = time.perf_counter() - start
    if list(baseline.shape) != [1, 3, 32, 32, 64] or candidate.shape != baseline.shape:
        raise RuntimeError("unexpected tiny graph output geometry")
    if not torch.isfinite(baseline).all() or not torch.isfinite(candidate).all():
        raise RuntimeError("nonfinite tiny graph output")
    if not torch.equal(baseline, candidate):
        raise RuntimeError("generated six-stage logits are not byte-identical")
    if not torch.equal(generated, original):
        raise RuntimeError("generated input changed")
    if identity_snapshot(network) != before:
        raise RuntimeError("forward altered model aliases")
    for name, tensor in network.state_dict().items():
        if not torch.equal(tensor, state_before[name]):
            raise RuntimeError("forward changed state tensor: " + name)
    raw = baseline.contiguous().numpy().tobytes()
    print(json.dumps({
        "scope": "generated_untrained_tiny_six_stage_graph_only",
        "input_shape": shape, "features_per_stage": features,
        "n_conv_per_stage": [2] * 6, "n_conv_per_stage_decoder": [2] * 5,
        "bottom_spatial_shape": [1, 1, 2], "cpu_fp32": True,
        "baseline_seconds": baseline_seconds, "recompute_seconds": candidate_seconds,
        "output_shape": list(candidate.shape),
        "output_raw_sha256": hashlib.sha256(raw).hexdigest(),
        "exact_logit_parity": True, "input_state_aliases_preserved": True,
        "tiled_conv_count": tiled["wrapped_unique_conv3d_modules"],
        "lazy_decoder_stages": lazy["final_decoder_stages"],
        "recompute_report": report,
    }, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
