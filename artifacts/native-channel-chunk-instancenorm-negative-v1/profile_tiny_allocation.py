"""One tiny generated CPU allocator profile; no host pressure or model input."""

import gc
import hashlib
import json
from pathlib import Path

import torch

from native_chunk_norm import native_norm_after_private_conv_


HERE = Path(__file__).resolve().parent


def peak_tracked_bytes(path):
    timeline = json.loads(path.read_text())
    if not isinstance(timeline, list) or len(timeline) != 2:
        raise ValueError("unexpected native profiler timeline")
    times, categories = timeline
    if not times or len(times) != len(categories):
        raise ValueError("missing native profiler allocation samples")
    return max(sum(row) for row in categories), len(times)


def one_profile(name, operation):
    path = HERE / (name + "-memory-timeline.json")
    with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU],
                                record_shapes=True, profile_memory=True,
                                with_stack=True) as profiler:
        with torch.inference_mode():
            output = operation()
            if not torch.isfinite(output).all():
                raise ValueError("nonfinite tiny generated output")
    profiler.export_memory_timeline(str(path), device="cpu")
    peak, count = peak_tracked_bytes(path)
    return output, {"peak_tracked_cpu_tensor_bytes": peak,
                    "timeline_points": count,
                    "timeline_sha256": hashlib.sha256(path.read_bytes()).hexdigest()}


def main():
    if torch.__version__ != "2.14.1":
        raise RuntimeError("unpinned Torch runtime")
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    torch.manual_seed(41)
    values = torch.randn((1, 8, 24, 24, 24), dtype=torch.float32)
    conv = torch.nn.Conv3d(8, 8, 3, padding=1, bias=False).eval()
    norm = torch.nn.InstanceNorm3d(8, eps=1e-5, affine=True,
                                   track_running_stats=False).eval()
    with torch.no_grad():
        norm.weight.copy_(torch.linspace(-1, 1, 8))
        norm.bias.copy_(torch.linspace(-0.03, 0.04, 8))
    baseline, base_metric = one_profile("native-full", lambda: norm(conv(values)))
    del baseline
    gc.collect()
    chunked, chunk_metric = one_profile(
        "native-channel-chunk-1",
        lambda: native_norm_after_private_conv_(conv, norm, values,
                                                 channel_chunk=1))
    with torch.inference_mode():
        reference = norm(conv(values))
    report = {
        "scope": "one tiny generated [1,8,24,24,24] CPU FP32 allocator profile; no model, patient, or host pressure probe",
        "torch_version": torch.__version__,
        "input_shape": list(values.shape),
        "input_bytes": values.numel() * values.element_size(),
        "output_shape": list(chunked.shape),
        "output_bytes": chunked.numel() * chunked.element_size(),
        "channel_chunk": 1,
        "max_abs_error": float((chunked - reference).abs().max()),
        "exact_equal": bool(torch.equal(chunked, reference)),
        "sign_disagreements": int(((chunked > 0) != (reference > 0)).sum()),
        "native_full": base_metric,
        "native_channel_chunk_1": chunk_metric,
        "profile_limit": "PyTorch CPU tensor allocator timeline, not total process RSS or native convolution workspace",
    }
    (HERE / "tiny-allocation-result.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"baseline_peak": base_metric["peak_tracked_cpu_tensor_bytes"],
                      "chunk_peak": chunk_metric["peak_tracked_cpu_tensor_bytes"],
                      "max_abs_error": report["max_abs_error"]}, sort_keys=True))


if __name__ == "__main__":
    if (HERE / "tiny-allocation-result.json").exists():
        raise SystemExit("tiny allocation experiment is one-shot")
    main()
