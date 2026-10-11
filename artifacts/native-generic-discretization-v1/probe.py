"""Actual generic tools: fixed physical sequence, only grid sampling changes.

Adapter to the existing bounded analytic native-discretization runner. This is
not a task, learner, new simulator, or proposal generator. No acquired inputs.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import sys
import time
from types import SimpleNamespace

for _name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[_name] = "1"

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
BASE_PATH = ROOT / "scripts/probe_native_discretization.py"
BASE_SHA256 = "6df8ade8e117667b77fef5285f99368b0d90ff8f04846caef77b213a53fa2eff"
if hashlib.sha256(BASE_PATH.read_bytes()).hexdigest() != BASE_SHA256:
    raise ValueError("Exact existing analytic benchmark adapter required")
_spec = importlib.util.spec_from_file_location("generic_resolution_base", BASE_PATH)
BASE = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(BASE)
_original_worker = BASE.worker
_original_launch = BASE.launch
DEADLINE = None
RUN_COUNTS = {"native_strokes_started": 0, "native_strokes_returned": 0, "native_seconds": 0.}


def declaration():
    return json.loads((HERE / "declaration.json").read_text())


def source_receipt():
    paths = [BASE_PATH, HERE / "probe.py", HERE / "declaration.json", HERE / "test_probe.py"]
    # Metadata-only import closure: actual core modules and old analytic helpers.
    paths += sorted((ROOT / "src/resectionlab").glob("*.py"))
    return {"files": {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest() for p in paths},
            "python": sys.version, "platform": BASE.platform.platform()}


def physical_actions(tool_id, spec):
    if tool_id not in spec["preparation_lattice_mm"]:
        raise ValueError("Declared actual generic tool required")
    spacing = spec["preparation_lattice_mm"][tool_id]
    actions = []
    for i, (x, y) in enumerate(spec["preparation_order_lattice_units"]):
        actions.append({"index": i + 1, "kind": "side_preparation", "tool_id": tool_id,
                        "entry_mm": [spacing*x, spacing*y, spec["entry_mm"][2]],
                        "tip_mm": [spacing*x, spacing*y, spec["preparation_tip_z_mm"]]})
    actions.append({"index": len(actions) + 1, "kind": "target_stroke", "tool_id": tool_id,
                    "entry_mm": list(spec["entry_mm"]), "tip_mm": [0., 0., spec["target_tip_z_mm"]]})
    if len(actions) != spec["actions_per_row"]:
        raise ValueError("Fixed complete action denominator changed")
    return actions


def continuum_rationale(tool, spec):
    """Conservative elementary enclosure, not a discrete feasibility assertion.

    Nine radius-r disks cover the square [-R,R]^2 if their Cartesian nearest
    center distance sqrt(2)*max(a/2,R-a) is below r. Target shaft tissue lies
    below the completed shallow cylinders and inside that square.
    """
    from math import sqrt
    a = spec["preparation_lattice_mm"][tool.tool_id]
    r, R, length = tool.tip_radius_mm, tool.shaft_radius_mm, tool.tip_length_mm
    prep_top = spec["preparation_tip_z_mm"] - length + R
    target_shaft_top = spec["target_tip_z_mm"] - length + R
    lateral_margin = r - sqrt(2)*max(a/2, R-a)
    aperture_margin = spec["access_radius_mm"] - (sqrt(2)*a + R)
    target_gap = spec["target_start_z_mm"] - (spec["preparation_tip_z_mm"] + r)
    target_penetration = spec["target_tip_z_mm"] + r - spec["target_start_z_mm"]
    if not (prep_top < 0 < target_shaft_top < spec["preparation_tip_z_mm"]
            and lateral_margin > 0 and aperture_margin > 0
            and target_gap > 0 and target_penetration > 0):
        raise ValueError("Predeclared positive-margin continuum construction failed")
    return {"preparation_shaft_highest_z_mm": prep_top,
            "target_shaft_highest_z_mm": target_shaft_top,
            "disk_cover_margin_mm": lateral_margin, "aperture_margin_mm": aperture_margin,
            "preparation_to_target_plane_gap_mm": target_gap,
            "target_stroke_cap_penetration_mm": target_penetration,
            "interpretation": "Continuous ideal cutting union covers the prescribed final shaft; whole-cell/native temporal acceptance remains an outcome."}


def grid_case(spacing, grid, spec):
    import numpy as np
    from resectionlab.geometry import AccessWindow, GENERIC_TOOLS
    from resectionlab.native_resection import NativeResectionConfig
    from resectionlab.native_spatial_task import DEFAULT_NATIVE_SPATIAL_REWARD
    declared = next(t for t in spec["tools"] if t["tool_id"] == grid["tool_id"])
    tool = next(t for t in GENERIC_TOOLS if t.tool_id == grid["tool_id"])
    if asdict(tool) != declared or asdict(DEFAULT_NATIVE_SPATIAL_REWARD) != spec["reward"]:
        raise ValueError("Actual generic tool or frozen reward changed")
    continuum_rationale(tool, spec)
    affine, tissue = BASE.sample_grid(spacing, grid, spec)
    indices = np.indices(tissue.shape, dtype=float)
    centers_z = sum(affine[2, i]*indices[i] for i in range(3)) + affine[2, 3]
    target = BASE.half_space_fractions(centers_z-spec["target_start_z_mm"], affine).astype(np.float32)
    target[~tissue] = 0.
    labels = (target > 0).astype(np.int16)
    source_hash = BASE.digest({"anatomy": spec["anatomy"], "shape": tissue.shape,
        "affine": affine.tolist(), "mask_sha256": hashlib.sha256(tissue.tobytes()).hexdigest(),
        "nominal_target_sha256": hashlib.sha256(target.tobytes()).hexdigest()})
    source = SimpleNamespace(mri=np.zeros(tissue.shape, np.uint8), affine=affine,
        frame="RAS+", semantic_hash=source_hash, reference_target=target)
    config = NativeResectionConfig(tissue, labels, affine,
        AccessWindow(spec["entry_mm"], (0, 0, 1), spec["access_radius_mm"], "fixed-analytic-access"),
        (tool,), source_hash, "analytic half-space; positive-intersection cells; no measured anatomy",
        case_id=f"{grid['name']}-{spacing:g}mm", max_tip_step_mm=spec["max_tip_step_mm"])
    return source, config


def plane_volume(indices, affine, start_z=0.):
    import numpy as np
    indices = np.asarray(indices, dtype=int).reshape(-1, 3)
    if not len(indices):
        return 0.
    centers_z = indices @ affine[2, :3] + affine[2, 3]
    return float(BASE.half_space_fractions(centers_z-start_z, affine).sum() * abs(np.linalg.det(affine[:3, :3])))


def _guard(spec):
    if ((DEADLINE is not None and time.perf_counter() >= DEADLINE)
            or BASE.peak_rss_bytes() > spec["caps"]["worker_peak_rss_bytes"]):
        raise TimeoutError("Declared numerical control time/RSS cap")


def numerical_row(spacing, grid, spec):
    import numpy as np
    from resectionlab.geometry import GeometryScene, capsule_voxel_indices
    from resectionlab.native_resection import NativeResectionEngine, contained_capsule_cells
    from resectionlab.native_spatial_task import NativeSpatialTask, DEFAULT_NATIVE_SPATIAL_REWARD
    started = time.perf_counter()
    source, config = grid_case(spacing, grid, spec)
    tool = config.tools[0]; actions = physical_actions(tool.tool_id, spec)
    scene = GeometryScene(np.zeros(config.tissue_mask.shape, bool), config.affine)
    static_contained = set(); static_contact = set()
    for action in actions:
        _guard(spec)
        entry, tip = np.asarray(action["entry_mm"]), np.asarray(action["tip_mm"])
        start = entry - np.array([0., 0., tool.tip_length_mm])
        touched = capsule_voxel_indices(scene, start, tip, tool.tip_radius_mm)
        touched = touched[config.tissue_mask[tuple(touched.T)]]
        contained = contained_capsule_cells(scene, touched, start, tip, tool.tip_radius_mm)
        static_contact.update(map(tuple, touched)); static_contained.update(map(tuple, contained))
    static = {"scope": "union of all prescribed ideal active sweeps; not executable clearance or a sequence certificate",
        "contained_physical_lower_mm3": plane_volume(sorted(static_contained), config.affine),
        "contact_physical_upper_mm3": plane_volume(sorted(static_contact), config.affine),
        "contained_cells": len(static_contained), "contact_cells": len(static_contact)}
    engine = NativeResectionEngine(config)
    scorer = SimpleNamespace(case=source, _config=config, reward_spec=DEFAULT_NATIVE_SPATIAL_REWARD, _planning=True)
    results = []; previous_tool = None; total_reward = 0.; all_removed = set(); all_contact = set()
    for action in actions:
        _guard(spec)
        before = engine.state_hash; stamp = time.perf_counter()
        RUN_COUNTS["native_strokes_started"] += 1
        if RUN_COUNTS["native_strokes_started"] > spec["caps"]["max_native_strokes"]:
            raise ValueError("Declared total primitive-call cap")
        try:
            result = engine.execute_stroke(tool.tool_id, action["tip_mm"], entry_mm=action["entry_mm"])
            RUN_COUNTS["native_strokes_returned"] += 1
        finally:
            RUN_COUNTS["native_seconds"] += time.perf_counter()-stamp
        row = {**action, "status": "accepted" if result.feasible else "rejected", "reason": result.reason,
            "source_state_hash": before, "result_state_hash": engine.state_hash,
            "failure_tip_mm": result.failure_tip_mm, "geometry_unknowns": result.geometry_unknowns,
            "native_seconds": time.perf_counter()-stamp,
            "removed_source_cell_mm3": result.removed_volume_mm3,
            "removed_physical_mm3": plane_volume(result.removed_indices_native, config.affine),
            "removed_physical_target_mm3": plane_volume(result.removed_indices_native, config.affine, spec["target_start_z_mm"])}
        if result.feasible:
            geometry = result.to_history_record()
            scored = NativeSpatialTask._score_record(scorer, geometry, previous_tool)
            row["frozen_reward"] = scored["reward"]; row["nominal_target_source_credit_mm3"] = scored["target_removed_mm3"]
            total_reward += scored["reward"]; previous_tool = tool.tool_id
            current = set(map(tuple, result.removed_indices_native))
            if current & all_removed: raise AssertionError("Duplicate native removal credit")
            all_removed.update(current); all_contact.update(map(tuple, result.contact_indices_native))
            if not np.all(engine.remaining_mask[engine.contact_mask & ~engine.removed_mask]):
                raise AssertionError("Partial contacts became free space")
        else:
            if engine.state_hash != before or result.removed_volume_mm3 != 0:
                raise AssertionError("Refused action changed the committed state")
        results.append(row)
        if not result.feasible:
            break
    completed = len(results) == len(actions) and all(r["status"] == "accepted" for r in results)
    for action in actions[len(results):]:
        results.append({**action, "status": "unexecuted_prior_refusal"})
    physical_removed = plane_volume(sorted(all_removed), config.affine)
    if not physical_removed <= static["contained_physical_lower_mm3"] + 1e-8:
        raise AssertionError("Executed removal exceeds planned static containment union")
    target_result = results[-1]
    record = {"row_id": config.case_id, "spacing_mm": spacing, "grid": grid,
        "shape": list(config.tissue_mask.shape), "affine": config.affine.tolist(),
        "source_hash": config.source_hash, "native_config_hash": config.fingerprint,
        "fixed_tool": asdict(tool), "physical_actions_sha256": BASE.digest(actions),
        "continuum_rationale": continuum_rationale(tool, spec), "static_not_executable": static,
        "actions": results, "planned_actions": len(actions), "native_calls": sum(r["status"] != "unexecuted_prior_refusal" for r in results),
        "native": {"feasible": completed, "complete_sequence": completed,
            "target_stroke_status": target_result["status"],
            "target_stroke_removed_physical_target_mm3": target_result.get("removed_physical_target_mm3", 0.),
            "accepted_actions": len(engine.history), "removed_cells": len(all_removed),
            "removed_source_cell_mm3": len(all_removed)*config.voxel_volume_mm3,
            "removed_physical_mm3": physical_removed,
            "removed_physical_target_mm3": plane_volume(sorted(all_removed), config.affine, spec["target_start_z_mm"]),
            "contacted_physical_upper_mm3": plane_volume(sorted(all_contact), config.affine),
            "removed_outer_voxelization_excess_mm3": len(all_removed)*config.voxel_volume_mm3-physical_removed,
            "partial_cells_kept_occupied": True, "frozen_reward_total": total_reward,
            "committed_history_hash": BASE.digest(engine.history)},
        "independent_audit": {"status": "pending" if engine.history else "not_applicable_rejected_no_committed_history"},
        "timing_seconds": {"numerical_row_total": time.perf_counter()-started}}
    return record, source, config, engine.history


def worker(output):
    """Reuse existing numerical/audit loop; select existing bounded batch checker."""
    global DEADLINE
    sys.path.insert(0, str(ROOT / "src"))
    from resectionlab import evaluation
    original_check = evaluation.independent_check_native_history
    spec = declaration(); DEADLINE = time.perf_counter() + spec["caps"]["cooperative_worker_seconds"]
    RUN_COUNTS.update(native_strokes_started=0, native_strokes_returned=0, native_seconds=0.)
    def checked(*args, **kwargs):
        return original_check(*args, **kwargs,
            distance_backend=spec["independent_check"]["distance_backend"],
            distance_batch_size=spec["independent_check"]["distance_batch_size"])
    evaluation.independent_check_native_history = checked
    try:
        code = _original_worker(output)
    finally:
        evaluation.independent_check_native_history = original_check; DEADLINE = None
    # Preserve missing/capped rows in an explicit denominator, never fill outcomes.
    report = json.loads((output / "report.json").read_text())
    report["version"] = spec["version"]
    seen = {r["row_id"] for r in report["rows"]}
    report["planned_rows"] = [f"{g['name']}-{h:g}mm" for h in spec["spacings_mm"] for g in spec["grids"]]
    report["missing_rows"] = [r for r in report["planned_rows"] if r not in seen]
    report["planned_action_slots"] = spec["expected_action_slots"]
    report["native_call_costs_including_incomplete_row"] = dict(RUN_COUNTS)
    report["numerical_threads"] = 1
    report["independent_check"] = spec["independent_check"]
    report["accepted_complete_sequences"] = sum(r["native"]["complete_sequence"] for r in report["rows"])
    report["verified_complete_sequences"] = sum(r["native"]["complete_sequence"]
        and r["independent_audit"].get("status") == "completed"
        and r["independent_audit"].get("certificate", {}).get("feasible") is True for r in report["rows"])
    BASE.write_json(output / "report.json", report)
    return code


# Existing launcher executes this adapter, whose worker still uses the original
# owned one-child lifecycle and caps. No source file is patched on disk.
BASE.__file__ = str(Path(__file__).resolve())
BASE.declaration = declaration
BASE.source_receipt = source_receipt
BASE.grid_case = grid_case
BASE.numerical_row = numerical_row


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args()
    return worker(args.output) if args.worker else _original_launch(args.output)


if __name__ == "__main__":
    raise SystemExit(main())
