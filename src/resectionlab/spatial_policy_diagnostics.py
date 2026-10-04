"""Observational coverage and native preview costs; no policy or geometry edits."""
from __future__ import annotations

from contextlib import contextmanager
from collections import Counter
import copy
from functools import wraps
import time

import numpy as np

from .spatial_policy import world_to_sample_grid
from .core import array_digest, semantic_digest


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


def spatial_coverage(observation, *, source_shape, source_affine, nominal_target, ray_samples=5,
                     native_affine=None):
    """Describe visible permitted inputs; the function accepts no private truth."""
    observation.assert_intact()
    shape = observation.image_channels.shape[1:]
    source_affine = np.asarray(source_affine)
    physical_affine = source_affine if native_affine is None else np.asarray(native_affine)
    origin = np.linalg.solve(physical_affine[:3, :3],
        observation.affine_ras_mm[:3, 3] - physical_affine[:3, 3])
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
        "original_source_affine_ras_mm": source_affine.tolist(),
        "native_physical_affine_ras_mm": physical_affine.tolist(),
        "frame_basis": "unchanged source voxel indices; native physical affine defines actor crop and tool rays",
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


def nominal_depth_coverage(case, *, native_affine=None):
    """An axial envelope upper bound, never a reachability/feasibility claim."""
    if getattr(case, "proposal_mode", "fixed_lattice") != "fixed_lattice":
        raise ValueError("Dynamic proposals require runtime_proposal_coverage with the actual emitted inventory")
    affine = np.asarray(case.affine_ras_mm)
    physical_affine = affine if native_affine is None else np.asarray(native_affine)
    normal = np.asarray(case.access.normal_inward)
    access_voxel = np.linalg.solve(affine[:3, :3], case.access.center_mm - affine[:3, 3])
    directions = affine[:3, :3] / np.linalg.norm(affine[:3, :3], axis=0)
    axis = int(np.argmax(np.abs(directions.T @ normal)))
    sign = 1 if directions[:, axis] @ normal > 0 else -1
    first = int(np.floor(access_voxel[axis]) + 1) if sign > 0 else int(np.ceil(access_voxel[axis]) - 1)
    cells = np.asarray(case._candidate_voxels, dtype=int).reshape(-1, 3)
    depths = (cells @ physical_affine[:3, :3].T + physical_affine[:3, 3] - case.access.center_mm) @ normal
    retained = sorted(set(int((cell[axis] - first) * sign) for cell in cells))
    upper = None if len(depths) == 0 else float(depths.max() + max(tool.tip_radius_mm for tool in case.tools))
    target = case.nominal_target
    positions = np.empty((0, 3), int) if target is None else np.argwhere(np.asarray(target) > 0)
    target_depths = (positions @ physical_affine[:3, :3].T + physical_affine[:3, 3] - case.access.center_mm) @ normal
    return {"proposal_scope": case._candidate_scope, "retained_endpoint_voxels": len(cells),
        "original_source_affine_ras_mm": affine.tolist(),
        "native_physical_affine_ras_mm": physical_affine.tolist(),
        "frame_basis": "original source affine determines proposal indices and axis; native physical affine determines depths",
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


def runtime_proposal_coverage(case, inventory, observation, *, native_affine=None, ray_samples=5):
    """Describe this state's emitted proposals without predicting later catalogs.

    Rejected preview attempts remain visible. Envelope membership is only a
    necessary geometric condition; it gives no removal or reachability credit.
    """
    observation.assert_intact()
    if hasattr(case, "assert_intact"):
        case.assert_intact()
    if "emitted" not in inventory:
        raise ValueError("Runtime diagnostics require explicit emitted proposal rows")
    emitted = inventory["emitted"]
    accepted_ids = [row["action_id"] for row in emitted if row["feasible"]]
    observed_ids = [identifier for identifier, allowed in zip(observation.action_ids[1:], observation.action_mask[1:]) if allowed]
    if accepted_ids != observed_ids or len({row["action_id"] for row in emitted}) != len(emitted):
        raise ValueError("Emitted inventory does not match distinct legal observed actions")
    if inventory.get("emitted_count", len(emitted)) != len(emitted):
        raise ValueError("Emitted proposal count differs from its runtime rows")
    source = np.asarray(case.affine_ras_mm)
    physical = source if native_affine is None else np.asarray(native_affine)
    if hasattr(case, "_native_affine_ras_mm") and not np.array_equal(physical, case._native_affine_ras_mm):
        raise ValueError("Runtime coverage frame differs from the frozen native grid")
    if hasattr(case, "source_hash") and (observation.source_id != case.source_hash
            or inventory.get("source_hash") != case.source_hash):
        raise ValueError("Runtime inventory and observation must bind the same permitted source")
    rejected_count = (len(inventory.get("ledger", emitted)) - len(accepted_ids)
        if getattr(case, "proposal_mode", "fixed_lattice") == "fixed_lattice"
        else len(emitted) - len(accepted_ids))
    for key, value in (("steps_taken", int(observation.state_features[0])),
                       ("max_steps", int(observation.state_features[1])),
                       ("remaining_steps", int(observation.state_features[1] - observation.state_features[0])),
                       ("accepted_count", len(accepted_ids)), ("rejected_count", rejected_count)):
        if key in inventory and inventory[key] != value:
            raise ValueError("Runtime inventory count or horizon differs from its observed/emitted catalog")
    if inventory.get("terminal") and emitted:
        raise ValueError("Terminal runtime inventory cannot retain active emitted proposals")
    provider = getattr(case, "_nominal_proposer", None)
    if provider is not None and (inventory.get("provider_model_hash") != provider.model_hash
            or inventory.get("nominal_target_hash") != array_digest(case.nominal_target)
            or inventory.get("nominal_provenance_hash") != (None if inventory.get("terminal")
                else semantic_digest(provider._provenance))):
        raise ValueError("Runtime proposal provider or permitted nominal evidence changed")
    normal, access = np.asarray(case.access.normal_inward), np.asarray(case.access.center_mm)
    tools = {tool.tool_id: tool for tool in case.tools}
    nominal = case.nominal_target
    target_cells = np.empty((0, 3), int) if nominal is None else np.argwhere(np.asarray(nominal) > 0)
    target_world = target_cells @ physical[:3, :3].T + physical[:3, 3]
    target_depths = (target_world - access) @ normal
    rows, envelopes = [], {"emitted": [], "accepted": []}
    shape = observation.image_channels.shape[1:]
    observed_indices = {identifier: index for index, identifier in enumerate(observation.action_ids)}
    for row in emitted:
        entry, tip = np.asarray(row["entry_mm"], float), np.asarray(row["tip_mm"], float)
        vector = tip - entry
        distance = float(np.linalg.norm(vector))
        if entry.shape != (3,) or tip.shape != (3,) or not np.isfinite([*entry, *tip]).all() or distance <= 0:
            raise ValueError("Emitted proposal requires a finite nonzero entry-to-tip segment")
        voxel = np.asarray(row["voxel"])
        if voxel.shape != (3,) or voxel.dtype.kind not in "iu" or np.any(voxel < 0):
            raise ValueError("Emitted proposal requires original source-cell indices")
        source_image = getattr(case, "structural_intensity", nominal)
        if source_image is not None and np.any(voxel >= np.asarray(source_image.shape)):
            raise ValueError("Emitted proposal source cell lies outside the native image")
        expected_tip = physical[:3, :3] @ voxel + physical[:3, 3]
        expected_entry = expected_tip - float((expected_tip - access) @ normal) * normal
        if not np.array_equal(tip, expected_tip) or not np.array_equal(entry, expected_entry):
            raise ValueError("Emitted geometry differs from its native source cell or projected entry")
        if row["tool_id"] not in tools:
            raise ValueError("Emitted proposal uses an undeclared tool")
        tool = tools[row["tool_id"]]
        if row["feasible"]:
            index = observed_indices[row["action_id"]]
            geometry = observation.action_geometry[index]
            dimensions = [tool.tip_radius_mm, tool.shaft_radius_mm, tool.working_length_mm,
                          tool.tip_length_mm, tool.max_access_angle_deg]
            if (observation.action_tool_ids[index] != row["tool_id"]
                    or not np.array_equal(np.asarray(entry, dtype=geometry.dtype), geometry[1:4])
                    or not np.array_equal(np.asarray(tip, dtype=geometry.dtype), geometry[4:7])
                    or not np.array_equal(np.asarray(dimensions, dtype=geometry.dtype), geometry[10:15])):
                raise ValueError("Accepted emitted geometry/tool differs from the actual observed encoding")
        if provider is not None:
            expected_id = "nominal-cavity-" + semantic_digest({"model": provider.model_hash,
                "cavity": inventory.get("cavity_state_hash"),
                "geometry": (tool.tool_id, tuple(entry), tuple(tip))}).split(":", 1)[1][:24]
            if row["action_id"] != expected_id:
                raise ValueError("Emitted proposal identifier differs from its declared provider/cavity geometry")
        points = entry + np.linspace(0., 1., ray_samples)[:, None] * vector
        grid = world_to_sample_grid(points, observation.affine_ras_mm, shape)
        inside = (np.abs(grid) <= 1).all(axis=1)
        active_start = entry - tool.tip_length_mm * vector / distance
        low, high = np.minimum(active_start, tip) - tool.tip_radius_mm, np.maximum(active_start, tip) + tool.tip_radius_mm
        distal = max(float((active_start - access) @ normal), float((tip - access) @ normal)) + tool.tip_radius_mm
        envelope = (low, high, distal, float((tip - access) @ normal))
        envelopes["emitted"].append(envelope)
        if row["feasible"]:
            envelopes["accepted"].append(envelope)
        rows.append({"action_id": row["action_id"], "tool_id": row["tool_id"],
            "family": row.get("family"), "feasible": bool(row["feasible"]),
            "reason": row.get("reason"), "entry_mm": entry.tolist(), "tip_mm": tip.tolist(),
            "tip_depth_mm": envelope[3], "entry_inside_center_bounds": bool(inside[0]),
            "tip_inside_center_bounds": bool(inside[-1]), "ray_samples": ray_samples,
            "ray_samples_inside": int(inside.sum()), "sample_fraction_inside": float(inside.mean()),
            "straight_segment_fraction_inside": segment_fraction_inside_grid(grid[0], grid[-1])})

    def envelope_report(items):
        if not items:
            return {"proposal_count": 0, "distal_active_tip_axial_upper_bound_mm": None,
                    "active_sweep_aabb_ras_mm": None, "target_centers_beyond_axial_bound": None,
                    "target_centers_outside_aabb": None, "endpoint_depth_range_mm": None}
        low = np.min([item[0] for item in items], axis=0)
        high = np.max([item[1] for item in items], axis=0)
        distal = max(item[2] for item in items)
        return {"proposal_count": len(items), "distal_active_tip_axial_upper_bound_mm": distal,
            "active_sweep_aabb_ras_mm": [low.tolist(), high.tolist()],
            "target_centers_beyond_axial_bound": None if nominal is None else int((target_depths > distal).sum()),
            "target_centers_outside_aabb": None if nominal is None else int(((target_world < low) | (target_world > high)).any(axis=1).sum()),
            "endpoint_depth_range_mm": [min(item[3] for item in items), max(item[3] for item in items)]}

    ledger = inventory.get("ledger", [])
    steps, maximum = (int(value) for value in observation.state_features[:2])
    return {"scope": "current_state_actual_emitted_proposals_only",
        "catalog_metadata": {key: value for key, value in inventory.items() if key not in {"emitted", "ledger"}},
        "catalog_count_basis": "Declared/omitted/duplicate/unavailable counts are family-tool-column slots; capped slots need not be unique geometries. Emitted actions are unique, actually previewed attempts. Legacy fixed rejected_count includes non-previewed outside-workspace slots.",
        "original_source_affine_ras_mm": source.tolist(), "native_physical_affine_ras_mm": physical.tolist(),
        "steps_taken": steps, "max_steps": maximum, "remaining_step_budget": max(0, maximum - steps),
        "episode_terminal": bool(inventory.get("terminal", steps >= maximum)),
        "proposal_dispositions": dict(Counter(row.get("proposal_reason", row.get("reason", "unspecified")) for row in ledger)),
        "preview_dispositions": dict(Counter(row.get("reason", "unspecified") for row in emitted)),
        "nominal_target_centers_total": None if nominal is None else len(target_cells),
        "nominal_target_center_depth_range_mm": None if not len(target_depths) else [float(target_depths.min()), float(target_depths.max())],
        "emitted_envelope": envelope_report(envelopes["emitted"]),
        "accepted_envelope": envelope_report(envelopes["accepted"]), "actions": rows,
        "ray_sample_coverage_counts": {"none": sum(row["ray_samples_inside"] == 0 for row in rows),
            "partial": sum(0 < row["ray_samples_inside"] < ray_samples for row in rows),
            "full": sum(row["ray_samples_inside"] == ray_samples for row in rows)},
        "target_basis": "static permitted nominal field; no private reference or residual/removal credit",
        "binding_scope": "Source/provider/nominal and source-cell geometry checked; accepted actions exactly match observation dtype projection and tools. Cavity/decision hashes are task-reported provenance, not reconstructed from cropped images; rejected feasibility remains native-preview authority.",
        "interpretation": "Current emitted/accepted active-tip sweep envelopes only. Bounds ignore cell containment, lateral gaps within the bounding box, prior-shaft clearance, interactions and horizon; future dynamic catalogs may extend them. No reachability or safety claim."}


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
