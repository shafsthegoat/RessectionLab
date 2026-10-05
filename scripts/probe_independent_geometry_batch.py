#!/usr/bin/env python3
"""Small, fixed numerical kernel control; cannot load cases or train policies."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import importlib.util
import json
from pathlib import Path
import platform
import resource
import signal
import statistics
import sys
import time
import tracemalloc

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
# Pin the implementation that the source receipt hashes. An editable install or
# another checkout on PYTHONPATH must not silently supply a different oracle.
sys.path.insert(0, str(ROOT / "src"))
from resectionlab import evaluation

PROTOTYPE = ROOT / "scripts/independent_geometry_batch_prototype.py"
SPEC = importlib.util.spec_from_file_location("independent_geometry_batch_prototype", PROTOTYPE)
batch = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(batch)
CONFIG = {
    "version": "independent-batch-kernel-control-v1",
    "seed": 604,
    "boxes_per_workload": 1024,
    "repetitions_per_phase": 2,
    "phase_order": ["scalar_before", "batch", "scalar_after"],
    "batch_size": 256,
    "memory_box_counts": [1024, 16384],
    "process_alarm_seconds": 45,
    "workloads": ["axial_anisotropic", "oblique_local_segment", "point_segment",
                  "large_translation", "tangency_scalar_fallback"],
    "relative_distance_tolerance": 2e-14,
    "absolute_distance_tolerance_mm2": 2e-14,
    "contact_tolerances_mm2": [0.0, 1e-10, 1e-9],
    "patient_data_loaded": False,
    "training_or_native_history_runs": 0,
}


def _write(path, value):
    with path.open("x") as stream:
        json.dump(value, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")


def _sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _array_sha(array):
    return hashlib.sha256(np.ascontiguousarray(array).tobytes()).hexdigest()


def _source_files():
    if Path(evaluation.__file__).resolve() != ROOT / "src/resectionlab/evaluation.py":
        raise RuntimeError("Loaded independent scalar source is not the receipted checkout")
    if Path(batch.__file__).resolve() != PROTOTYPE:
        raise RuntimeError("Loaded batch source is not the receipted prototype")
    paths = [PROTOTYPE, Path(__file__).resolve(), ROOT / "src/resectionlab/evaluation.py",
             ROOT / "tests/test_independent_geometry_batch.py"]
    return {str(path.relative_to(ROOT)): _sha(path) for path in paths}


def _fixtures():
    rng = np.random.default_rng(CONFIG["seed"])
    count = CONFIG["boxes_per_workload"]
    centers = np.argwhere(np.ones((16, 8, 8), dtype=bool)) * [.25, .5, 1.0]
    spacing = np.array([.25, .5, 1.0])
    lower, upper = centers - spacing / 2, centers + spacing / 2
    yield "axial_anisotropic", np.array([1.7, 1.4, -3.0]), np.array([1.7, 1.4, 9.0]), lower, upper, .45
    start, end = np.array([-1.2, 3.1, -.5]), np.array([5.3, .3, 8.2])
    yield "oblique_local_segment", start, end, lower, upper, 1.25
    yield "point_segment", start, start, lower, upper, 1.25
    random_lower = rng.uniform(-10, 10, size=(count, 3)) + 1e9
    random_upper = random_lower + rng.uniform(.01, 3, size=(count, 3))
    yield "large_translation", start + 1e9, end + 1e9, random_lower, random_upper, 1.25
    tangent_lower = np.zeros((count, 3)); tangent_lower[:, 0] = np.linspace(0, 10, count)
    tangent_lower[:, 1] = 1; tangent_lower[:, 2] = -1
    tangent_upper = tangent_lower + [.125, 1, 2]
    yield "tangency_scalar_fallback", np.array([-1., 0, 0]), np.array([12., 0, 0]), tangent_lower, tangent_upper, 1.0


def _scalar(start, end, lower, upper):
    return np.array([evaluation.segment_box_distance_sq(start, end, lo, hi) for lo, hi in zip(lower, upper)])


def _timed(call):
    samples, result = [], None
    for _ in range(CONFIG["repetitions_per_phase"]):
        begun = time.perf_counter()
        current = call()
        samples.append(time.perf_counter() - begun)
        if result is not None and not np.array_equal(current, result):
            raise AssertionError("Repeated kernel invocation changed its output")
        result = current
    return result, {"seconds": samples, "median_seconds": statistics.median(samples)}


def _measure_workload(fixture):
    name, start, end, lower, upper, radius = fixture
    phases, outputs = {}, {}
    for phase in CONFIG["phase_order"]:
        call = (lambda: batch.segment_box_distances_sq(start, end, lower, upper, batch_size=CONFIG["batch_size"])) if phase == "batch" else (lambda: _scalar(start, end, lower, upper))
        outputs[phase], phases[phase] = _timed(call)
    before, actual, after = (outputs[key] for key in CONFIG["phase_order"])
    if not np.array_equal(before, after):
        raise AssertionError("Scalar reversal changed its output")
    if not np.allclose(actual, before, rtol=CONFIG["relative_distance_tolerance"], atol=CONFIG["absolute_distance_tolerance_mm2"]):
        raise AssertionError("Batch distances disagree with scalar reference")
    contact_checks = []
    for tolerance in CONFIG["contact_tolerances_mm2"]:
        expected = np.flatnonzero(before <= radius * radius + tolerance)
        begun = time.perf_counter()
        indices = batch.segment_box_contact_indices(start, end, lower, upper, radius,
            tolerance_sq=tolerance, batch_size=CONFIG["batch_size"])
        elapsed = time.perf_counter() - begun
        first = batch.first_segment_box_contact(start, end, lower, upper, radius,
            tolerance_sq=tolerance, batch_size=CONFIG["batch_size"])
        if not np.array_equal(indices, expected) or first != (int(expected[0]) if len(expected) else None):
            raise AssertionError("Ordered contact or first-witness mismatch")
        contact_checks.append({"tolerance_mm2": tolerance, "ordered_indices_match": True,
            "contacts": len(indices), "first_witness": first, "indices_sha256": _array_sha(indices),
            "one_contact_call_seconds": elapsed, "includes_near_threshold_scalar_fallback": True})
    return {"workload": name, "boxes": len(lower), "phases": phases,
        "input_sha256": {key: _array_sha(value) for key, value in (("start", start), ("end", end), ("lower", lower), ("upper", upper))},
        "distance_sha256": {key: _array_sha(value) for key, value in outputs.items()},
        "bitwise_distance_matches": int(np.count_nonzero(actual == before)),
        "max_absolute_distance_difference_mm2": float(np.max(np.abs(actual - before))),
        "distance_tolerance_passed": True, "scalar_reversal_identical": True,
        "contact_checks": contact_checks,
        "interpretation": "Distance-kernel timings only; source transforms, occupancy, native history, and independent full-tool audit are excluded"}


def _memory_measurement(count):
    # Inputs exist before tracing. The required returned N-vector is reported
    # separately; NumPy/Python traced peak is not process resident memory.
    lower, upper = np.zeros((count, 3)), np.ones((count, 3))
    tracemalloc.start()
    try:
        result = batch.segment_box_distances_sq([-2, .2, 0], [2, .7, 3], lower, upper,
                                                batch_size=CONFIG["batch_size"])
        current, peak = tracemalloc.get_traced_memory()
        return {"boxes": count, "batch_size": CONFIG["batch_size"],
            "input_bytes_excluded": lower.nbytes + upper.nbytes, "required_output_bytes": result.nbytes,
            "traced_current_bytes": current, "traced_peak_bytes": peak,
            "peak_minus_required_output_bytes": peak - result.nbytes}
    finally:
        tracemalloc.stop()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    sources = _source_files()
    _write(output / "prospective.json", {"config": CONFIG, "sources": sources,
        "created_utc": datetime.now(timezone.utc).isoformat(), "argv": sys.argv,
        "python": sys.version, "python_executable": sys.executable, "numpy": np.__version__, "platform": platform.platform(),
        "scope": "Numerical unit control only; no patient or training data; production evaluator unchanged"})
    rows = []
    begun = time.perf_counter()
    def expired(signum, frame):
        raise TimeoutError("Declared 45-second kernel process alarm reached")
    previous_handler = signal.signal(signal.SIGALRM, expired)
    signal.setitimer(signal.ITIMER_REAL, CONFIG["process_alarm_seconds"])
    try:
        for fixture in _fixtures():
            row = _measure_workload(fixture)
            rows.append(row)
            _write(output / f"{len(rows):02d}-{row['workload']}.json", row)
        memory = [_memory_measurement(count) for count in CONFIG["memory_box_counts"]]
        if _source_files() != sources:
            raise AssertionError("Kernel/probe/reference/test source changed during measurement")
        rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        _write(output / "report.json", {"status": "completed", "config": CONFIG, "sources": sources,
            "rows": rows, "memory": memory, "elapsed_seconds": time.perf_counter() - begun,
            "peak_process_rss_bytes": int(rss if sys.platform == "darwin" else rss * 1024),
            "source_unchanged": True, "production_evaluator_wired": False,
            "limitations": ["One runtime and one process; other local work may be active",
                "No patient or full native audit speedup measured", "Only five fixed numerical workloads",
                "Contact-call samples are single observations, not paired timing estimates",
                "Memory trace excludes existing inputs and does not capture every resident allocation",
                "Near-threshold scalar fallback can eliminate the benefit for tangent or large-coordinate boxes"]})
    except BaseException as error:
        _write(output / "failure.json", {"status": "failed", "type": type(error).__name__,
            "message": str(error), "completed_workloads": len(rows), "elapsed_seconds": time.perf_counter() - begun})
        raise
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        signal.signal(signal.SIGALRM, previous_handler)


if __name__ == "__main__":
    main()
