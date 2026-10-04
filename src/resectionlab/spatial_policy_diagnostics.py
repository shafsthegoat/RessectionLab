"""Observational coverage and native preview costs; no policy or geometry edits."""
from __future__ import annotations

from contextlib import contextmanager
import copy
from functools import wraps
import time

import numpy as np

from .spatial_policy import world_to_sample_grid


def segment_fraction_inside_grid(first, last):
    """Fraction of a straight normalized-coordinate segment inside [-1,1]^3.

    This is center-domain visibility, not swept-tool clearance or tissue safety.
    """
    first, last = np.asarray(first, float), np.asarray(last, float)
    lo, hi = 0., 1.
    for position, delta in zip(first, last - first):
        if delta == 0:
            if not -1 <= position <= 1:
                return 0.
            continue
        a, b = sorted(((-1 - position) / delta, (1 - position) / delta))
        lo, hi = max(lo, a), min(hi, b)
        if hi < lo:
            return 0.
    return float(max(0., hi - lo))


def spatial_coverage(observation, *, source_shape, source_affine, nominal_target, ray_samples=5):
    """Describe visible permitted inputs; the function accepts no private truth."""
    observation.assert_intact()
    shape = observation.image_channels.shape[1:]
    origin = np.linalg.solve(np.asarray(source_affine)[:3, :3],
        observation.affine_ras_mm[:3, 3] - np.asarray(source_affine)[:3, 3])
    target = observation.image_channels[2]
    covered = observation.coverage[2] & observation.channel_available[2]
    visible_mass = float(target[covered].sum())
    total_mass = None if nominal_target is None else float(np.asarray(nominal_target).sum())
    rows = []
    for index, identifier in enumerate(observation.action_ids):
        if index == 0 or not observation.action_mask[index]:
            continue
        geometry = observation.action_geometry[index]
        entry, tip = geometry[1:4], geometry[4:7]
        points = entry + np.linspace(0., 1., ray_samples)[:, None] * (tip - entry)
        grid = world_to_sample_grid(points, observation.affine_ras_mm, shape)
        inside = (np.abs(grid) <= 1).all(axis=1)
        rows.append({"action_id": identifier, "entry_inside_center_bounds": bool(inside[0]),
            "tip_inside_center_bounds": bool(inside[-1]), "ray_samples": ray_samples,
            "ray_samples_inside": int(inside.sum()), "sample_fraction_inside": float(inside.mean()),
            "straight_segment_fraction_inside": segment_fraction_inside_grid(grid[0], grid[-1])})
    return {"source_shape": list(source_shape), "crop_shape": list(shape),
        "crop_origin_source_voxels": origin.tolist(), "crop_last_center_source_voxels": (origin + np.asarray(shape) - 1).tolist(),
        "crop_affine_ras_mm": observation.affine_ras_mm.tolist(),
        "nominal_target_available": bool(observation.channel_available[2]),
        "nominal_target_positive_voxels_in_crop": int(np.count_nonzero(target[covered] > 0)),
        "nominal_target_positive_voxels_total": None if nominal_target is None else int(np.count_nonzero(nominal_target)),
        "nominal_target_mass_in_crop": visible_mass if observation.channel_available[2] else None,
        "nominal_target_mass_total": total_mass,
        "nominal_target_mass_fraction_visible": None if total_mass is None or total_mass == 0 else visible_mass / total_mass,
        "target_basis": "static permitted nominal annotation/estimate; not private reference or residual volume",
        "legal_nonstop_actions": len(rows),
        "sample_coverage_counts": {"none": sum(row["ray_samples_inside"] == 0 for row in rows),
            "partial": sum(0 < row["ray_samples_inside"] < ray_samples for row in rows),
            "full": sum(row["ray_samples_inside"] == ray_samples for row in rows)},
        "actions": rows,
        "interpretation": "image-center-domain coverage only; neither tool clearance nor clinical evidence adequacy"}


def nominal_depth_coverage(case):
    """An axial envelope upper bound, never a reachability/feasibility claim."""
    affine = np.asarray(case.affine_ras_mm)
    normal = np.asarray(case.access.normal_inward)
    access_voxel = np.linalg.solve(affine[:3, :3], case.access.center_mm - affine[:3, 3])
    directions = affine[:3, :3] / np.linalg.norm(affine[:3, :3], axis=0)
    axis = int(np.argmax(np.abs(directions.T @ normal)))
    sign = 1 if directions[:, axis] @ normal > 0 else -1
    first = int(np.floor(access_voxel[axis]) + 1) if sign > 0 else int(np.ceil(access_voxel[axis]) - 1)
    cells = np.asarray(case._candidate_voxels, dtype=int).reshape(-1, 3)
    depths = (cells @ affine[:3, :3].T + affine[:3, 3] - case.access.center_mm) @ normal
    retained = sorted(set(int((cell[axis] - first) * sign) for cell in cells))
    upper = None if len(depths) == 0 else float(depths.max() + max(tool.tip_radius_mm for tool in case.tools))
    target = case.nominal_target
    positions = np.empty((0, 3), int) if target is None else np.argwhere(np.asarray(target) > 0)
    target_depths = (positions @ affine[:3, :3].T + affine[:3, 3] - case.access.center_mm) @ normal
    return {"proposal_scope": case._candidate_scope, "retained_endpoint_voxels": len(cells),
        "nominal_source_axis": axis, "inward_sign": sign,
        "requested_depth_offsets_voxels": [0, 1, 2, 4, 8, 16, 24, 31]
            if case._candidate_scope == "fixed_access_grid_3columns_8depths_within_actor_crop" else None,
        "retained_depth_offsets_voxels": retained,
        "endpoint_depth_range_mm": None if not len(depths) else [float(depths.min()), float(depths.max())],
        "distal_tip_capsule_axial_upper_bound_mm": upper,
        "nominal_target_center_depth_range_mm": None if not len(target_depths) else [float(target_depths.min()), float(target_depths.max())],
        "nominal_target_centers_beyond_axial_upper_bound": None if upper is None or target is None else int((target_depths > upper).sum()),
        "nominal_target_centers_total": None if target is None else len(positions),
        "interpretation": "all declared endpoints before feasibility; axial center-distance upper bound ignores lateral distance, containment, blocked shaft travel and action horizon"}


class NativePreviewProfiler:
    """Process-local transparent wrapper; restores the exact original method."""
    def __init__(self, engine_type):
        self.engine_type = engine_type
        self.phase_name = "unclassified"
        self.records = {}

    def __enter__(self):
        self.original = self.engine_type.preview_stroke

        @wraps(self.original)
        def measured(instance, *args, **kwargs):
            record = self.records.setdefault(self.phase_name, {"started": 0, "returned": 0,
                "raised": 0, "seconds": 0., "feasible": 0, "rejected": 0, "returned_microsteps": 0,
                "rejection_reasons": {}})
            record["started"] += 1
            started = time.perf_counter()
            try:
                result = self.original(instance, *args, **kwargs)
                record["returned"] += 1
                record["feasible" if result.feasible else "rejected"] += 1
                record["returned_microsteps"] += len(result.microsteps)
                if not result.feasible:
                    reasons = record["rejection_reasons"]
                    reasons[result.reason] = reasons.get(result.reason, 0) + 1
                return result
            except BaseException:
                record["raised"] += 1
                raise
            finally:
                record["seconds"] += time.perf_counter() - started

        self.engine_type.preview_stroke = measured
        return self

    def __exit__(self, *args):
        self.engine_type.preview_stroke = self.original

    @contextmanager
    def phase(self, name):
        previous, self.phase_name = self.phase_name, name
        try:
            yield
        finally:
            self.phase_name = previous

    def snapshot(self):
        return {"scope": "process cumulative native preview calls, including constructor/fresh/next-inventory work",
                "phases": copy.deepcopy(self.records),
                "unmeasured": "internal preview subphases, bytes copied/hashed, engine initialization and integrity time require separate profiling"}
