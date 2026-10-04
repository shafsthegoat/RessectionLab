"""Retain a numerical frame-boundary counterexample; no simulator or learner."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from research.native_spatial_features import VERSION, access_source_frame


def report():
    affine = np.eye(4)
    center = np.zeros(3)
    normal = np.array([1., 1e-8, 0.])
    points = np.array([[0., 1., 1.]])
    rotation, _ = np.linalg.qr(np.array([[.2, .5, .7], [.8, .1, .3], [.2, .4, .1]]))
    if np.linalg.det(rotation) < 0:
        rotation[:, 0] *= -1
    changed_affine = affine.copy(); changed_affine[:3, :3] = rotation
    original = access_source_frame(affine, center, normal)
    changed = access_source_frame(changed_affine, rotation @ center, rotation @ normal)
    before = original.project(points)
    after = changed.project(points @ rotation.T)
    positive = access_source_frame(affine, center, [1., 2e-8, 0.])
    negative = access_source_frame(affine, center, [1., -2e-8, 0.])
    return {
        "scope": "Numerical descriptor-conditioning counterexample only; no simulator, policy, patient or gradient",
        "descriptor_version": VERSION,
        "numpy_version": np.__version__,
        "descriptor_source_sha256": hashlib.sha256((ROOT / "research/native_spatial_features.py").read_bytes()).hexdigest(),
        "reproducer_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "threshold_rotation_counterexample": {
            "affine": affine.tolist(), "center_mm": center.tolist(), "normal": normal.tolist(),
            "points_mm": points.tolist(), "rotation": rotation.tolist(),
            "rotation_determinant": float(np.linalg.det(rotation)),
            "rotation_orthogonality_max_abs_error": float(np.max(np.abs(rotation.T @ rotation - np.eye(3)))),
            "original_frame": original.to_dict(), "changed_frame": changed.to_dict(),
            "original_coordinates_mm": before.tolist(), "changed_coordinates_mm": after.tolist(),
            "max_abs_change_mm": float(np.max(np.abs(before - after))),
            "interpretation": "At the declared tangent cutoff, floating-point rotation can change which source axis is selected; regular-fixture invariance does not establish unconditional numerical invariance.",
        },
        "near_parallel_normal_sensitivity": {
            "normal_a": [1., 2e-8, 0.], "normal_b": [1., -2e-8, 0.],
            "normal_max_abs_change": 4e-8,
            "point_mm": points.tolist(),
            "max_coordinate_change_mm": float(np.max(np.abs(positive.project(points) - negative.project(points)))),
            "interpretation": "These are different but nearby normals, not identical geometry; the first-axis tangent gauge is ill-conditioned near parallelism.",
        },
    }


if __name__ == "__main__":
    print(json.dumps(report(), indent=2, sort_keys=True, allow_nan=False))
