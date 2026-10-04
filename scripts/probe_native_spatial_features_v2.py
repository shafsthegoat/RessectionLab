#!/usr/bin/env python3
"""Bounded descriptor algebra only: no simulator, patient, policy or optimizer."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import platform
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import numpy as np

from research.native_spatial_features_v2 import (DeclaredTangent, FrameConditioningError,
    MIN_TANGENT_SINE, ROUNDOFF_REJECTION_HALF_WIDTH, declared_access_frame,
    candidate_coordinates, cavity_residual_summary, descriptor_contract)


CONDITIONING_INPUT = Path("artifacts/native-spatial-feature-independent-v1/conditioning-counterexample.json")
ALIAS_INPUT = Path("artifacts/native-spatial-feature-probe-v1/probe/report.json")
OWNED_INPUTS = (Path("research/native_spatial_features.py"), Path("research/native_spatial_features_v2.py"),
    Path("scripts/probe_native_spatial_features_v2.py"), Path("tests/test_native_spatial_features_v2.py"),
    CONDITIONING_INPUT, ALIAS_INPUT)


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def source_receipt():
    return {"files": {str(name): sha(ROOT/name) for name in OWNED_INPUTS},
            "python": sys.version, "numpy": np.__version__, "platform": platform.platform()}


def attempt(normal, tangent, *, source_hash, frame_label, points, center=(0, 0, 0), declaration_id):
    reference = DeclaredTangent(tuple(tangent), source_hash, frame_label, declaration_id)
    try:
        frame = declared_access_frame(center, normal, reference=reference,
            expected_source_hash=source_hash, coordinate_frame=frame_label)
    except FrameConditioningError as error:
        return error.receipt, None
    return {**frame.to_dict(), "status": "accepted", "coordinates_mm": frame.project(points).tolist()}, frame


def run_probe():
    before = source_receipt(); start = time.perf_counter()
    original_evidence = json.loads((ROOT/CONDITIONING_INPUT).read_text())
    saved = original_evidence["threshold_rotation_counterexample"]
    q, normal, points = (np.asarray(saved[key]) for key in ("rotation", "normal", "points_mm"))
    source_hash = "sha256:" + before["files"][str(CONDITIONING_INPUT)]
    tangent = np.asarray((0., 1., 0.))
    initial, original_frame = attempt(normal, tangent, source_hash=source_hash,
        frame_label="RAS+", points=points, declaration_id="externally-fixed-y-reference-for-audited-example")
    rotated, transformed_frame = attempt(q @ normal, q @ tangent, source_hash=source_hash,
        frame_label="DECLARED_ROTATED_RAS+", points=points @ q.T,
        declaration_id="externally-fixed-y-reference-for-audited-example")
    assert original_frame is not None and transformed_frame is not None
    error = float(np.max(np.abs(original_frame.project(points)-transformed_frame.project(points @ q.T))))
    assert error < 1e-12
    rejected = []
    for n, t, p, label in ((normal, (1, 0, 0), points, "RAS+"),
                          (q @ normal, q @ np.asarray((1, 0, 0)), points @ q.T, "DECLARED_ROTATED_RAS+")):
        receipt, result = attempt(n, t, source_hash=source_hash, frame_label=label, points=p,
            declaration_id="externally-fixed-near-parallel-x-reference")
        assert result is None and receipt["status"] == "rejected"
        rejected.append(receipt)
    boundaries = []
    for sine in (.099, MIN_TANGENT_SINE-0.5*ROUNDOFF_REJECTION_HALF_WIDTH,
                 MIN_TANGENT_SINE, MIN_TANGENT_SINE+0.5*ROUNDOFF_REJECTION_HALF_WIDTH, .101):
        receipt, result = attempt((1, 0, 0), (np.sqrt(1-sine*sine), sine, 0),
            source_hash=source_hash, frame_label="RAS+", points=points,
            declaration_id="declared-engineering-boundary-probe")
        boundaries.append({"requested_sine": sine, "receipt": receipt})
        assert (result is not None) == (sine == .101)
    # Read frozen V1 geometry and reward observations; no new native transitions.
    v1 = json.loads((ROOT/ALIAS_INPUT).read_text())
    branches = v1["branches"]
    declared = declared_access_frame((3, 3, -.5), (0, 0, 1),
        reference=DeclaredTangent((1, 0, 0), v1["case_hash"], "RAS+", "explicit-synthetic-fixture-x-reference"),
        expected_source_hash=v1["case_hash"], coordinate_frame="RAS+")
    coords = candidate_coordinates(tuple(row["action_id"] for row in branches),
        [row["entry_mm"] for row in branches], [row["tip_mm"] for row in branches], declared)
    assert not np.array_equal(coords[0], coords[1])
    tissue = np.ones((7, 7, 8), bool)
    target = np.zeros_like(tissue); target[1:6, 1:6, 2:7] = True
    a, b = np.zeros_like(tissue), np.zeros_like(tissue)
    a[2:4, 2:4, 3] = True; b[1, 2:4, 3] = True; b[4, 2:4, 3] = True
    sa, sb = (cavity_residual_summary(tissue, target, mask, np.eye(4), declared) for mask in (a, b))
    assert np.array_equal(sa, sb) and not np.array_equal(a, b)
    after = source_receipt()
    if before != after:
        raise RuntimeError("Source or input evidence changed during the algebraic probe")
    return {"status": "completed_synthetic_declared_tangent_probe", "contract": descriptor_contract(),
        "source_receipt": before,
        "saved_v1_rotation_counterexample_error_mm": saved["max_abs_change_mm"],
        "fixed_explicit_reference": {"original": initial, "rotated": rotated, "max_abs_error_mm": error},
        "near_parallel_declared_reference_rejections": rejected, "engineering_boundary_results": boundaries,
        "archived_alias_geometry": {"source_report_sha256": before["files"][str(ALIAS_INPUT)],
            "frame_receipt": declared.to_dict(), "candidate_coordinates_mm": coords.tolist(),
            "distinguishes_pair": True,
            "historical_two_cut_returns_not_recomputed": [row["best_two_step_return"] for row in branches]},
        "unchanged_lossy_summary_counterexample": {"equal_summary": sa.tolist(),
            "cavity_a_indices": np.argwhere(a).tolist(), "cavity_b_indices": np.argwhere(b).tolist(),
            "native_reachability_verified": False},
        "native_transitions": 0, "simulators_constructed": 0, "policies_evaluated": 0,
        "optimizers_constructed": 0, "gradients": 0, "patients_loaded": 0,
        "elapsed_seconds": time.perf_counter()-start,
        "limits": ["No globally shared cross-patient tangent convention supplied",
            "Reference metadata equality is checked; external provenance authenticity is not established here",
            "Roundoff rejection band is an engineering margin, not a rigorous global error bound",
            "Coordinate changes must transport the explicit reference vector",
            "Contact history, tool, remaining budget, geometry/topology and candidate interactions remain incomplete",
            "No new learned-policy evidence or production profile"]}


def main():
    parser = argparse.ArgumentParser(); parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(); args.output.mkdir(parents=True, exist_ok=False)
    report = run_probe()
    (args.output/"report.json").write_text(json.dumps(report, indent=2, sort_keys=True, allow_nan=False)+"\n")
    print(json.dumps({"status": report["status"], "max_abs_error_mm": report["fixed_explicit_reference"]["max_abs_error_mm"],
                     "native_transitions": 0, "gradients": 0, "elapsed_seconds": report["elapsed_seconds"]}, indent=2))


if __name__ == "__main__":
    main()
