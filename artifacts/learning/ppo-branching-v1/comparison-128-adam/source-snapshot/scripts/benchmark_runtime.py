#!/usr/bin/env python3
"""Bounded synthetic CPU/MPS measurements; these are not patient-model results."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import math
import os
from pathlib import Path
import platform
import resource
import statistics
import time

# Set before scientific imports. Limit competing workers on a shared-memory Mac.
os.environ.setdefault("OMP_NUM_THREADS", "2")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "2")
os.environ.setdefault("MKL_NUM_THREADS", "2")

import numpy as np
import scipy
from scipy.ndimage import distance_transform_edt
import torch


SEED = 4102026


def process_peak_bytes() -> int:
    """ru_maxrss is cumulative process peak, not isolated workload allocation."""
    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return int(peak if platform.system() == "Darwin" else peak * 1024)


def timed(name, operation, *, dimensions, warmup, repeats, validate):
    for _ in range(warmup):
        validate(operation())
    peak_before = process_peak_bytes()
    samples = []
    for _ in range(repeats):
        start = time.perf_counter_ns()
        result = operation()
        samples.append((time.perf_counter_ns() - start) / 1e6)
        validate(result)
    return {
        "name": name,
        "dimensions": dimensions,
        "warmup_iterations": warmup,
        "measured_iterations": repeats,
        "elapsed_unit": "milliseconds",
        "median_ms": statistics.median(samples),
        "p95_ms": float(np.percentile(samples, 95)),
        "min_ms": min(samples),
        "max_ms": max(samples),
        "samples_ms": samples,
        "process_peak_rss_bytes_before_measured_iterations": peak_before,
        "process_peak_rss_bytes_after_measured_iterations": process_peak_bytes(),
    }


def edt_benchmark(size, repeats):
    coords = np.ogrid[:size, :size, :size]
    squared = sum((axis - size / 2) ** 2 for axis in coords)
    mask = squared > (size / 8) ** 2

    def validate(result):
        assert result.shape == mask.shape and result.dtype == np.float64
        assert result[size // 2, size // 2, size // 2] == 0
        assert np.isfinite(result[0, 0, 0]) and result[0, 0, 0] > 0

    return timed(
        "scipy_euclidean_distance_transform",
        lambda: distance_transform_edt(mask, sampling=(1.0, 1.0, 1.0)),
        dimensions={
            "shape_voxels": [size] * 3,
            "spacing_mm": [1.0] * 3,
            "input_dtype": "bool",
            "output_dtype": "float64",
            "input_bytes": mask.nbytes,
            "output_bytes": size ** 3 * 8,
            "timing_includes_output_allocation": True,
        },
        warmup=2,
        repeats=repeats,
        validate=validate,
    )


def geometry_benchmark(repeats):
    rng = np.random.default_rng(SEED)
    points = rng.uniform(-80, 80, (50_000, 3)).astype(np.float64)
    starts = rng.uniform(-60, 60, (8, 3)).astype(np.float64)
    ends = starts + rng.normal(0, 20, (8, 3))
    vectors = ends - starts
    lengths2 = (vectors * vectors).sum(axis=1)
    radius_mm = 2.0

    def capsule_contacts():
        offset = points[:, None, :] - starts[None, :, :]
        fraction = np.clip(
            np.einsum("ijk,jk->ij", offset, vectors) / lengths2, 0.0, 1.0
        )
        offset -= fraction[:, :, None] * vectors[None, :, :]
        distance2 = np.einsum("ijk,ijk->ij", offset, offset)
        return np.count_nonzero(distance2 <= radius_mm * radius_mm)

    expected = int(capsule_contacts())
    result = timed(
        "vectorized_point_to_capsule_contacts",
        capsule_contacts,
        dimensions={
            "point_count": len(points),
            "segment_count": len(starts),
            "point_segment_pairs": len(points) * len(starts),
            "radius_mm": radius_mm,
            "dtype": "float64",
            "timing_includes_temporary_allocation": True,
            "meaning": "synthetic point-obstacle proxy; not final volumetric geometry checker",
        },
        warmup=3,
        repeats=repeats,
        validate=lambda result: ensure(int(result) == expected),
    )
    result["contact_pair_count"] = expected
    return result


def ensure(condition):
    if not condition:
        raise AssertionError("Benchmark output failed validation")


def policy_update_benchmark(device_name, batch_size, repeats):
    torch.manual_seed(SEED)
    cpu_model = torch.nn.Sequential(
        torch.nn.Linear(128, 128),
        torch.nn.Tanh(),
        torch.nn.Linear(128, 64),
        torch.nn.Tanh(),
        torch.nn.Linear(64, 16),
    )
    observations = torch.randn(batch_size, 128)
    actions = torch.randint(0, 16, (batch_size,))
    advantages = torch.randn(batch_size)
    device = torch.device(device_name)
    model = cpu_model.to(device)
    observations, actions, advantages = (
        item.to(device) for item in (observations, actions, advantages)
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=3e-4)
    initial = [parameter.detach().cpu().clone() for parameter in model.parameters()]
    losses = []

    def synchronize():
        if device_name == "mps":
            torch.mps.synchronize()

    def update():
        synchronize()
        optimizer.zero_grad(set_to_none=True)
        log_probs = torch.log_softmax(model(observations), dim=-1)
        selected = log_probs.gather(1, actions[:, None]).squeeze(1)
        entropy = -(log_probs.exp() * log_probs).sum(dim=-1).mean()
        loss = -(selected * advantages).mean() - 0.01 * entropy
        loss.backward()
        optimizer.step()
        synchronize()
        return loss.detach()

    def validate(result):
        value = float(result.cpu())
        ensure(math.isfinite(value))
        losses.append(value)

    result = timed(
        "torch_policy_gradient_style_adam_update",
        update,
        dimensions={
            "device": device_name,
            "batch_size": batch_size,
            "observation_features": 128,
            "hidden_features": [128, 64],
            "action_count": 16,
            "parameter_count": sum(p.numel() for p in model.parameters()),
            "dtype": "float32",
            "optimizer": "Adam",
            "learning_rate": 3e-4,
            "entropy_coefficient": 0.01,
            "updates_per_timing_sample": 1,
            "timing_excludes_input_transfer_and_model_creation": True,
            "timing_includes_backward_optimizer_and_device_synchronization": True,
            "data_meaning": "fixed synthetic observations/actions/advantages; no environment or patient learning claim",
        },
        warmup=5,
        repeats=repeats,
        validate=validate,
    )
    change = max(
        float((parameter.detach().cpu() - before).abs().max())
        for parameter, before in zip(model.parameters(), initial)
    )
    ensure(change > 0)
    result["validation"] = {
        "finite_losses": True,
        "actual_gradient_updates": 5 + repeats,
        "max_absolute_parameter_change": change,
        "first_loss": losses[0],
        "last_loss": losses[-1],
        "interpretation": "numeric workload validation only; fixed advantages do not establish policy improvement",
    }
    if device_name == "mps":
        result["mps_allocated_bytes_after_workload"] = torch.mps.current_allocated_memory()
        result["mps_driver_allocated_bytes_after_workload"] = torch.mps.driver_allocated_memory()
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--repeats", type=int, default=15)
    args = parser.parse_args()
    if args.repeats < 3:
        parser.error("--repeats must be at least 3")
    torch.set_num_threads(2)
    torch.set_num_interop_threads(1)
    report = {
        "schema_version": 1,
        "captured_at": datetime.now(timezone.utc).isoformat(),
        "scope": "synthetic runtime microbenchmarks, not a trained patient model or a clinical claim",
        "seed": SEED,
        "runtime": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "scipy": scipy.__version__,
            "torch": torch.__version__,
            "os": platform.system(),
            "macos": platform.mac_ver()[0],
            "architecture": platform.machine(),
            "torch_intraop_threads": torch.get_num_threads(),
            "torch_interop_threads": torch.get_num_interop_threads(),
            "mps_built": torch.backends.mps.is_built(),
            "mps_available": torch.backends.mps.is_available(),
        },
        "measurement_notes": [
            "Inputs prepared before timing; validation excluded from elapsed times.",
            "Process RSS is cumulative peak and includes imports and earlier workloads.",
            "Warmups excluded; p95 is the linear-interpolated sample percentile.",
            "MPS timings synchronize before and after each update.",
            "Machine may run other project agents; no idle-machine or sustained throughput claim.",
        ],
        "workloads": [],
        "failures": [],
    }
    for size in (96, 160):
        report["workloads"].append(edt_benchmark(size, args.repeats))
    report["workloads"].append(geometry_benchmark(args.repeats))
    for batch_size in (128, 512):
        report["workloads"].append(policy_update_benchmark("cpu", batch_size, args.repeats))
        if report["runtime"]["mps_available"]:
            try:
                report["workloads"].append(policy_update_benchmark("mps", batch_size, args.repeats))
            except (RuntimeError, AssertionError) as error:
                report["failures"].append({
                    "device": "mps", "batch_size": batch_size, "error": str(error)
                })
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    for item in report["workloads"]:
        print(f'{item["name"]} {item["dimensions"]}: median={item["median_ms"]:.3f}ms p95={item["p95_ms"]:.3f}ms')
    if report["failures"]:
        print(json.dumps(report["failures"], indent=2))


if __name__ == "__main__":
    main()
