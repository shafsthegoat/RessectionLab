#!/usr/bin/env python3
"""Fixed physical half-space/tool; actual revoxelization, never policy training.

Static cell containment/contact are geometric bounds, not executable removal.
Native execution retains partial cells and every causal shaft/frontier guard.
No production model is modified. A fresh output directory is mandatory.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict
from hashlib import sha256
from itertools import product
import json
from math import factorial, pi
from pathlib import Path
import platform
import resource
import subprocess
import sys
import time
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
DECLARATION = Path("manifests/experiments/native-discretization-control-v1.json")


def write_json(path: Path, value: object) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def digest(value: object) -> str:
    return "sha256:" + sha256(json.dumps(value, sort_keys=True, allow_nan=False).encode()).hexdigest()


def source_receipt() -> dict:
    paths = [DECLARATION, Path(__file__).relative_to(ROOT), Path("tests/test_native_discretization_control.py")]
    paths += [Path("src/resectionlab") / name for name in
              ("geometry.py", "native_resection.py", "evaluation.py", "worlds.py", "core.py")]
    return {"files": {str(path): sha256((ROOT / path).read_bytes()).hexdigest() for path in paths},
            "python": sys.version, "platform": platform.platform()}


def declaration() -> dict:
    return json.loads((ROOT / DECLARATION).read_text())


def half_space_fractions(centers_z, affine):
    """Exact box/half-space fractions via the weighted-uniform-sum CDF.

    A uniform point in an affine cell has z = center_z + sum(a_i U_i),
    U_i uniform[-.5,.5]. Inclusion-exclusion integrates the resulting simplex.
    Zero coefficients are omitted exactly, preserving axial and tilted grids.
    This helper does not call any production containment/contact routine.
    """
    import numpy as np
    centers = np.asarray(centers_z, dtype=float)
    widths = np.abs(np.asarray(affine, dtype=float)[2, :3])
    widths = widths[widths > 0]
    if not len(widths) or not np.isfinite(widths).all() or not np.isfinite(centers).all():
        raise ValueError("Finite nondegenerate physical cells are required")
    total = float(widths.sum())
    weights = widths / total
    t = (total / 2 - centers) / total
    cdf = np.zeros_like(t)
    for bits in product((0, 1), repeat=len(weights)):
        shift = sum(weight for bit, weight in zip(bits, weights) if bit)
        cdf += (-1) ** sum(bits) * np.maximum(t - shift, 0) ** len(weights)
    cdf /= factorial(len(weights)) * float(np.prod(weights))
    cdf = np.where(t <= 0, 0, np.where(t >= 1, 1, cdf))
    return np.clip(1 - cdf, 0, 1)


def ideal_volume(radius: float = 1.25) -> float:
    """z>=0 clips the fixed [-4,4] active capsule at its cylindrical section."""
    if not 0 <= radius < 4:
        raise ValueError("Reference formula requires proximal cap below tissue")
    return pi * radius**2 * 4 + 2 * pi * radius**3 / 3


def sample_grid(spacing: float, grid: dict, spec: dict):
    import numpy as np
    from scipy.spatial.transform import Rotation
    rotation = Rotation.from_euler("xyz", grid["rotation_xyz_degrees"], degrees=True).as_matrix()
    phase = np.asarray(grid["phase_source_voxels"], float)
    bounds = np.asarray(spec["anatomy"]["sampled_field_contains_box_mm"], float)
    corners = np.asarray(list(product(*zip(bounds[0], bounds[1]))))
    local = corners @ rotation / spacing - phase
    low = np.floor(local.min(axis=0)).astype(int) - 2
    high = np.ceil(local.max(axis=0)).astype(int) + 2
    shape = tuple(int(value) for value in high - low + 1)
    if int(np.prod(shape)) > spec["caps"]["max_grid_cells"]:
        raise ValueError("Declared grid-cell cap exceeded")
    affine = np.eye(4)
    affine[:3, :3] = spacing * rotation
    affine[:3, 3] = spacing * rotation @ (low + phase)
    grid_indices = np.indices(shape, dtype=float)
    centers_z = sum(affine[2, axis] * grid_indices[axis] for axis in range(3)) + affine[2, 3]
    # Outer voxelization: every positive-volume tissue intersection is occupied.
    # Whole-cell excess is explicitly quantified, never relabeled true anatomy.
    tissue = centers_z + np.abs(affine[2, :3]).sum() / 2 > 0
    return affine, tissue


def physical_volume(indices, affine) -> float:
    import numpy as np
    if not len(indices):
        return 0.0
    points = np.asarray(indices) @ affine[:3, :3].T + affine[:3, 3]
    return float(half_space_fractions(points[:, 2], affine).sum() * abs(np.linalg.det(affine[:3, :3])))


def grid_case(spacing: float, grid: dict, spec: dict):
    import numpy as np
    from resectionlab.geometry import AccessWindow
    from resectionlab.native_resection import NATIVE_GENERIC_TOOLS, NativeResectionConfig
    affine, tissue = sample_grid(spacing, grid, spec)
    tool = NATIVE_GENERIC_TOOLS[0]
    if any(asdict(tool)[key] != value for key, value in spec["tool"].items()):
        raise ValueError("The declared fixed instrument changed")
    source_hash = digest({"scene": spec["anatomy"], "shape": tissue.shape,
                          "affine": affine.tolist(), "mask_sha256": sha256(tissue.tobytes()).hexdigest()})
    source = SimpleNamespace(mri=np.zeros(tissue.shape, np.uint8), affine=affine, frame="RAS+", semantic_hash=source_hash)
    config = NativeResectionConfig(tissue, tissue.astype(np.int16), affine,
        AccessWindow(spec["entry_mm"], (0, 0, 1), spec["access_radius_mm"], "fixed-analytic-access"),
        (tool,), source_hash, "analytic z>=0 half-space; conservative positive-intersection source cells",
        case_id=f"{grid['name']}-{spacing:g}mm", max_tip_step_mm=spec["max_tip_step_mm"])
    return source, config


def numerical_row(spacing: float, grid: dict, spec: dict):
    import numpy as np
    from resectionlab.geometry import GeometryScene, capsule_voxel_indices
    from resectionlab.native_resection import NativeResectionEngine, contained_capsule_cells
    started = time.perf_counter()
    source, config = grid_case(spacing, grid, spec)
    preparation_seconds = time.perf_counter() - started
    tool = config.tools[0]
    entry, tip = np.asarray(spec["entry_mm"]), np.asarray(spec["tip_mm"])
    active_start = entry - np.array((0, 0, tool.tip_length_mm))
    scene = GeometryScene(np.zeros(config.tissue_mask.shape, bool), config.affine)
    static_started = time.perf_counter()
    contact = capsule_voxel_indices(scene, active_start, tip, tool.tip_radius_mm)
    contact = contact[config.tissue_mask[tuple(contact.T)]]
    contained = contained_capsule_cells(scene, contact, active_start, tip, tool.tip_radius_mm)
    lower, upper = physical_volume(contained, config.affine), physical_volume(contact, config.affine)
    ideal = ideal_volume(tool.tip_radius_mm)
    # These are bounds for the fixed continuous physical scene, not cut credit.
    if not lower <= ideal + 1e-8 or not ideal <= upper + 1e-8:
        raise AssertionError("Static physical-volume bracket failed")
    diameter = np.sqrt(3) * spacing
    minkowski_lower = ideal_volume(max(0.0, tool.tip_radius_mm - diameter - 1e-10))
    minkowski_upper = ideal_volume(tool.tip_radius_mm + diameter)
    if lower + 1e-8 < minkowski_lower or upper > minkowski_upper + 1e-8:
        raise AssertionError("Cell-diameter enclosure failed")
    static_seconds = time.perf_counter() - static_started
    engine = NativeResectionEngine(config)
    original_state = engine.state_hash
    native_started = time.perf_counter()
    result = engine.execute_stroke(tool.tool_id, tip)
    native_seconds = time.perf_counter() - native_started
    removed_physical = physical_volume(result.removed_indices_native, config.affine)
    contact_physical = physical_volume(result.contact_indices_native, config.affine)
    if removed_physical > lower + 1e-8:
        raise AssertionError("Native removal exceeds the geometric containment bound")
    if result.feasible:
        if not np.all(engine.remaining_mask[engine.contact_mask & ~engine.removed_mask]):
            raise AssertionError("Partial cells became free space")
        if not np.array_equal(result.contact_indices_native, contact):
            raise AssertionError("Whole-stroke contact union differs from the static cover")
    elif (engine.state_hash != original_state or engine.removed_mask.any() or engine.contact_mask.any()
          or not np.array_equal(engine.remaining_mask, config.tissue_mask)):
        raise AssertionError("Rejected stroke changed the native state")
    history = engine.history
    record = {
        "row_id": config.case_id, "spacing_mm": spacing, "grid": grid,
        "shape": list(config.tissue_mask.shape), "affine": config.affine.tolist(),
        "source_hash": config.source_hash, "native_config_hash": config.fingerprint,
        "fixed_entry_mm": entry.tolist(), "fixed_tip_mm": tip.tolist(), "fixed_tool": asdict(tool),
        "ideal_physical_active_sweep_mm3": ideal,
        "static_not_executable": {
            "contained_cell_count": len(contained), "contact_cell_count": len(contact),
            "contained_source_cell_mm3": len(contained) * config.voxel_volume_mm3,
            "contact_source_cell_upper_mm3": len(contact) * config.voxel_volume_mm3,
            "contained_physical_lower_mm3": lower, "contact_physical_upper_mm3": upper,
            "physical_bracket_width_mm3": upper - lower,
            "cell_diameter_theoretical_lower_mm3": minkowski_lower,
            "cell_diameter_theoretical_upper_mm3": minkowski_upper},
        "native": {"feasible": result.feasible, "reason": result.reason,
            "failure_tip_mm": result.failure_tip_mm, "completed_microsteps": len(result.microsteps),
            "removed_source_cell_mm3": result.removed_volume_mm3,
            "removed_physical_lower_mm3": removed_physical,
            "removed_outer_voxelization_excess_mm3": result.removed_volume_mm3 - removed_physical,
            "contacted_physical_upper_mm3": contact_physical,
            "partial_contact_physical_upper_mm3": contact_physical - removed_physical,
            "ideal_sweep_tissue_not_credited_removed_mm3": ideal - removed_physical,
            "partial_cells_kept_occupied": True, "rejected_transaction_unchanged": not result.feasible,
            "metrics": engine.metrics(), "geometry_unknowns": result.geometry_unknowns,
            "committed_history_hash": digest(history)},
        "independent_audit": {"status": "pending" if result.feasible else "not_applicable_rejected_no_committed_history"},
        "timing_seconds": {"grid_preparation": preparation_seconds, "static_bounds": static_seconds,
                           "native_stroke": native_seconds, "numerical_row_total": time.perf_counter() - started},
    }
    return record, source, config, history


def peak_rss_bytes() -> int:
    value = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return int(value if sys.platform == "darwin" else value * 1024)


def worker(output: Path) -> int:
    sys.path.insert(0, str(ROOT / "src"))
    import numpy as np
    import scipy
    from resectionlab.evaluation import independent_check_native_history
    spec, source_before = declaration(), source_receipt()
    started = time.perf_counter()
    deadline = started + spec["caps"]["cooperative_worker_seconds"]
    report = {"version": spec["version"], "status": "running", "declaration_hash": digest(spec),
        "source": source_before, "numpy": np.__version__, "scipy": scipy.__version__,
        "expected_rows": spec["expected_rows"], "rows": [], "human_patients": 0,
        "training_updates": 0, "anatomy_and_tool_fixed": True,
        "independent_audit_scope": "Existing separate native-history checker; no certificate for rejected, cancelled or unaudited rows."}
    pending = []

    def expired() -> bool:
        return time.perf_counter() >= deadline or peak_rss_bytes() > spec["caps"]["worker_peak_rss_bytes"]

    def save() -> None:
        report["worker_seconds"] = time.perf_counter() - started
        report["peak_worker_rss_bytes"] = peak_rss_bytes()
        report["completed_rows"] = len(report["rows"])
        write_json(output / "report.json", report)

    try:
        for spacing in spec["spacings_mm"]:
            for grid in spec["grids"]:
                if expired():
                    raise TimeoutError("Cooperative numerical-control time/RSS cap")
                row, case, config, history = numerical_row(spacing, grid, spec)
                report["rows"].append(row)
                # Keep only accepted analytic histories for later checking;
                # all arrays/source masks are re-created deterministically then.
                if history:
                    pending.append((row, history))
                del case, config
                save()
        for row, history in pending:
            if expired():
                row["independent_audit"] = {"status": "unexecuted_cap_reached"}
                continue
            audit_started = time.perf_counter()
            case, config = grid_case(row["spacing_mm"], row["grid"], spec)
            audit = independent_check_native_history(case, config.tools, history,
                tissue_mask=config.tissue_mask, access=config.access, cancelled=expired)
            cancelled = "independent_validation_cancelled" in audit.failures
            row["independent_audit"] = {"status": "cancelled_cap_reached" if cancelled else "completed",
                "seconds": time.perf_counter() - audit_started, "certificate": audit.to_dict()}
            save()
            if not audit.feasible and not cancelled:
                raise AssertionError(f"Independent native audit rejected {row['row_id']}: {audit.failures}")
        report["status"] = "completed" if all(row["independent_audit"]["status"] in
            {"completed", "not_applicable_rejected_no_committed_history"} for row in report["rows"]) else "completed_numerical_rows_audits_incomplete"
    except Exception as error:
        report["status"] = "cooperative_cap_reached" if isinstance(error, TimeoutError) else "failed"
        report["error"] = {"type": type(error).__name__, "message": str(error)}
    for row in report["rows"]:
        if row["independent_audit"]["status"] == "pending":
            row["independent_audit"] = {"status": "unexecuted_worker_incomplete"}
    report["source_unchanged"] = source_receipt() == source_before
    if not report["source_unchanged"]:
        report["status"] = "failed_source_changed"
    save()
    return 0 if report["status"] in {"completed", "completed_numerical_rows_audits_incomplete"} else 1


def launch(output: Path) -> int:
    if output.exists():
        raise FileExistsError("A fresh output directory is required")
    output.mkdir(parents=True)
    spec, before = declaration(), source_receipt()
    write_json(output / "prospective-source.json", {"source": before, "declaration": spec})
    command = [sys.executable, str(Path(__file__).resolve()), "--worker", "--output", str(output.resolve())]
    started = time.perf_counter()
    reason, observed_rss = None, 0
    with (output / "worker.log").open("w") as log:
        process = subprocess.Popen(command, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT)
        while process.poll() is None:
            if time.perf_counter() - started >= spec["caps"]["parent_wall_seconds"]:
                reason = "parent_wall_cap"
            try:
                rss = subprocess.check_output(["/bin/ps", "-o", "rss=", "-p", str(process.pid)], text=True)
                observed_rss = max(observed_rss, int(rss.strip()) * 1024)
            except (subprocess.CalledProcessError, ValueError):
                pass
            if observed_rss > spec["caps"]["worker_peak_rss_bytes"]:
                reason = "parent_rss_cap"
            if reason:
                process.kill()
                break
            time.sleep(0.1)
        code = process.wait()
    try:
        result = json.loads((output / "report.json").read_text())
    except (OSError, ValueError):
        result = None
    published = (isinstance(result, dict) and result.get("source") == before
        and result.get("source_unchanged") is True and result.get("declaration_hash") == digest(spec)
        and result.get("completed_rows") == spec["expected_rows"]
        and len(result.get("rows", [])) == spec["expected_rows"]
        and result.get("status") in {"completed", "completed_numerical_rows_audits_incomplete"})
    status = result["status"] if code == 0 and reason is None and published else "incomplete_or_failed"
    receipt = {"command": command, "worker_exit_code": code, "parent_stop_reason": reason,
        "parent_wall_seconds": time.perf_counter() - started, "peak_polled_worker_rss_bytes": observed_rss,
        "source_unchanged": source_receipt() == before,
        "worker_report_status": result.get("status") if isinstance(result, dict) else None,
        "numerical_rows_complete": bool(published),
        "independent_audits_complete": bool(published and result["status"] == "completed"), "status": status}
    if not receipt["source_unchanged"]:
        receipt["status"] = "failed_source_changed"
    write_json(output / "launcher.json", receipt)
    return 0 if receipt["status"] in {"completed", "completed_numerical_rows_audits_incomplete"} else 1


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args()
    return worker(args.output) if args.worker else launch(args.output)


if __name__ == "__main__":
    raise SystemExit(main())
