"""One bounded experimental arm against a pinned, retained 18-row baseline.

Same generated anatomy, tools, physical actions, reward and microstep. Only the
explicit idealized cumulative cut/evacuation model changes. No acquired inputs.
Canonical producer/checker must be installed before this runner is released.
"""
from __future__ import annotations

import argparse
from dataclasses import replace
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import resource
import sys
import time
from types import SimpleNamespace

for _name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[_name] = "1"

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
GENERIC_PATH = ROOT / "artifacts/native-generic-discretization-v1/probe.py"
GENERIC_SHA256 = "b4f07e29324a3dcc8fce2c861fc45b34fc1d2051176832f87e76ef3b3c235797"
if hashlib.sha256(GENERIC_PATH.read_bytes()).hexdigest() != GENERIC_SHA256:
    raise ValueError("Pinned generic-grid adapter required")
_loader = importlib.util.spec_from_file_location("retained_generic_grid_adapter", GENERIC_PATH)
GENERIC = importlib.util.module_from_spec(_loader)
_loader.loader.exec_module(GENERIC)
BASE = GENERIC.BASE
_original_worker = GENERIC._original_worker
_original_launch = GENERIC._original_launch
DEADLINE = None
LIVE_OUTPUT = None
RUN_COUNTS = {"native_strokes_started": 0, "native_strokes_returned": 0, "native_seconds": 0.}
COVERAGE_WORK = {"returned_previews_accounted": 0, "leaf_tests": 0, "new_leaf_count": 0,
                 "witness_rows": 0, "peak_tracked_cells": 0, "budget_refusals": 0}
physical_actions = GENERIC.physical_actions
plane_volume = GENERIC.plane_volume
continuum_rationale = GENERIC.continuum_rationale


def declaration():
    return json.loads((HERE / "declaration.json").read_text())


def cut_spec(spec):
    from resectionlab.native_cut_union import CutUnionSpec
    value = CutUnionSpec(**spec["experimental_cut_union"])
    if value.to_record()["mode"] != spec["experimental_mode"]:
        raise ValueError("Declared experimental mode changed")
    return value


def retained_baseline(spec):
    record = spec["retained_baseline"]
    path = ROOT / record["report_path"]
    raw = path.read_bytes()
    if hashlib.sha256(raw).hexdigest() != record["report_sha256"]:
        raise ValueError("Retained baseline report hash mismatch")
    prior = json.loads(raw)
    old_spec = json.loads((ROOT / record["declaration_path"]).read_text())
    for key in spec["fixed_baseline_fields"]:
        if spec[key] != old_spec[key]:
            raise ValueError("Matched physical baseline field changed: " + key)
    if (prior["status"] != "completed" or prior["completed_rows"] != 18
            or prior["missing_rows"] or prior["source_unchanged"] is not True):
        raise ValueError("Complete retained baseline required")
    if BASE.digest(old_spec) != prior["declaration_hash"]:
        raise ValueError("Retained baseline declaration identity mismatch")
    return prior


def source_receipt():
    # Same small import-closure receipt as the retained benchmark; root freezes
    # the final canonical producer and both independent-checker source files.
    paths = [GENERIC_PATH, GENERIC.BASE_PATH, HERE / "probe_union.py", HERE / "declaration.json", HERE / "test_probe_union.py"]
    paths += sorted((ROOT / "src/resectionlab").glob("*.py"))
    for name in ("native_cut_union.py", "native_resection.py", "evaluation.py", "independent_cut_union.py"):
        if not (ROOT / "src/resectionlab" / name).is_file():
            raise ValueError("Canonical experimental producer/checker missing: " + name)
    spec = declaration()
    retained_baseline(spec)
    paths += [ROOT / spec["retained_baseline"]["report_path"], ROOT / spec["retained_baseline"]["declaration_path"]]
    return {"files": {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest() for p in paths},
            "python": sys.version, "platform": BASE.platform.platform()}


def grid_case(spacing, grid, spec):
    source, config = GENERIC.grid_case(spacing, grid, spec)
    return source, replace(config, cut_union=cut_spec(spec))


def exact_target_cap(tool, spec):
    height = spec["target_tip_z_mm"] + tool.tip_radius_mm - spec["target_start_z_mm"]
    if not 0 < height < 2*tool.tip_radius_mm:
        raise ValueError("Declared analytic target reference does not apply")
    if not spec["preparation_tip_z_mm"] + tool.tip_radius_mm < spec["target_start_z_mm"]:
        raise ValueError("Preparations cannot overlap analytic target reference")
    return math.pi*height**2*(tool.tip_radius_mm-height/3)


def check_volume_bounds(physical_removed, target_removed, static, exact_target):
    # Cumulative union may contain cells absent from every individual capsule's
    # whole-cell lower set. The old single-capsule lower is only a reference.
    if not 0 <= physical_removed <= static["contact_physical_upper_mm3"] + 1e-8:
        raise AssertionError("Removal exceeds planned active-contact upper bound")
    if not 0 <= target_removed <= exact_target + 1e-8:
        raise AssertionError("Removal exceeds exact continuous target cap")


def _guard(spec):
    if ((DEADLINE is not None and time.perf_counter() >= DEADLINE)
            or BASE.peak_rss_bytes() > spec["caps"]["worker_peak_rss_bytes"]):
        raise TimeoutError("Declared numerical control time/RSS cap")


def _progress(row_id, action_index, in_progress):
    if LIVE_OUTPUT is not None:
        BASE.write_json(LIVE_OUTPUT/"progress.json", {
            "native_call_costs_including_incomplete_row": dict(RUN_COUNTS),
            "returned_preview_coverage_work_including_incomplete_row": dict(COVERAGE_WORK),
            "current_row": row_id, "current_action": action_index,
            "native_call_in_progress": in_progress,
            "scope": "Completed native runtime only; an interrupted active call is additionally bounded by parent wall time"})


def _attempt_record(result):
    # Scalar diagnostic only. Sparse committed proof stays in transient history
    # for the separate checker and is never emitted in report.json.
    value = getattr(result, "cut_union_attempt", None)
    if value is None:
        raise ValueError("Producer must account for every experimental preview")
    value = dict(value)
    names = ("leaf_tests", "new_leaf_count", "witness_rows", "peak_tracked_cells", "budget_refused")
    if any(name not in value for name in names):
        raise ValueError("Incomplete preview work sidecar")
    return {name: value[name] for name in names}


def numerical_row(spacing, grid, spec):
    import numpy as np
    from dataclasses import asdict
    from resectionlab.geometry import GeometryScene, capsule_voxel_indices
    from resectionlab.native_resection import NativeResectionEngine, contained_capsule_cells
    from resectionlab.native_spatial_task import NativeSpatialTask, DEFAULT_NATIVE_SPATIAL_REWARD
    started = time.perf_counter()
    source, config = grid_case(spacing, grid, spec)
    tool = config.tools[0]
    actions = physical_actions(tool.tool_id, spec)
    scene = GeometryScene(np.zeros(config.tissue_mask.shape, bool), config.affine)
    lower, upper = set(), set()
    for action in actions:
        _guard(spec)
        entry, tip = np.asarray(action["entry_mm"]), np.asarray(action["tip_mm"])
        start = entry - np.array([0., 0., tool.tip_length_mm])
        touched = capsule_voxel_indices(scene, start, tip, tool.tip_radius_mm)
        touched = touched[config.tissue_mask[tuple(touched.T)]]
        contained = contained_capsule_cells(scene, touched, start, tip, tool.tip_radius_mm)
        upper.update(map(tuple, touched)); lower.update(map(tuple, contained))
    static = {"scope": "planned active-sweep union; contained lower is reference only, not an experimental removal ceiling",
        "contained_physical_lower_mm3": plane_volume(sorted(lower), config.affine),
        "contact_physical_upper_mm3": plane_volume(sorted(upper), config.affine),
        "contained_cells": len(lower), "contact_cells": len(upper)}
    exact_target = exact_target_cap(tool, spec)
    engine = NativeResectionEngine(config)
    scorer = SimpleNamespace(case=source, _config=config, reward_spec=DEFAULT_NATIVE_SPATIAL_REWARD, _planning=True)
    results, removed, contacted = [], set(), set()
    previous_tool = None
    for action in actions:
        _guard(spec)
        before = engine.state_hash
        if RUN_COUNTS["native_strokes_started"] >= spec["caps"]["max_native_strokes"]:
            raise TimeoutError("Declared primitive-call ceiling")
        RUN_COUNTS["native_strokes_started"] += 1
        _progress(config.case_id, action["index"], True)
        stamp = time.perf_counter()
        try:
            result = engine.execute_stroke(tool.tool_id, action["tip_mm"], entry_mm=action["entry_mm"])
            RUN_COUNTS["native_strokes_returned"] += 1
        finally:
            elapsed = time.perf_counter()-stamp
            RUN_COUNTS["native_seconds"] += elapsed
            _progress(config.case_id, action["index"], False)
        attempt = _attempt_record(result)
        COVERAGE_WORK["returned_previews_accounted"] += 1
        for name in ("leaf_tests", "new_leaf_count", "witness_rows"):
            COVERAGE_WORK[name] += attempt[name]
        COVERAGE_WORK["peak_tracked_cells"] = max(COVERAGE_WORK["peak_tracked_cells"], attempt["peak_tracked_cells"])
        COVERAGE_WORK["budget_refusals"] += bool(attempt["budget_refused"])
        _progress(config.case_id, action["index"], False)
        row = {**action, "status": "accepted" if result.feasible else "rejected", "reason": result.reason,
            "source_state_hash": before, "result_state_hash": engine.state_hash,
            "failure_tip_mm": result.failure_tip_mm, "geometry_unknowns": result.geometry_unknowns,
            "native_seconds": elapsed, "cut_union_attempt": attempt,
            "removed_source_cell_mm3": result.removed_volume_mm3,
            "removed_physical_mm3": plane_volume(result.removed_indices_native, config.affine),
            "removed_physical_target_mm3": plane_volume(result.removed_indices_native, config.affine, spec["target_start_z_mm"])}
        if result.feasible:
            geometry = result.to_history_record()
            proof = geometry["cut_union"]
            row["cut_union_committed"] = {k: proof[k] for k in ("before_digest", "after_digest", "new_leaf_count", "leaf_tests", "coverage_only_progress")}
            scored = NativeSpatialTask._score_record(scorer, geometry, previous_tool)
            row["frozen_reward"] = scored["reward"]
            row["nominal_target_source_credit_mm3"] = scored["target_removed_mm3"]
            previous_tool = tool.tool_id
            current = set(map(tuple, result.removed_indices_native))
            if current & removed:
                raise AssertionError("Duplicate native removal credit")
            removed.update(current); contacted.update(map(tuple, result.contact_indices_native))
            if not np.all(engine.remaining_mask[engine.contact_mask & ~engine.removed_mask]):
                raise AssertionError("Partially certified cells became free space")
        elif engine.state_hash != before or result.removed_volume_mm3 != 0:
            raise AssertionError("Rejected preview changed committed state")
        results.append(row)
        if not result.feasible:
            break
    completed = len(results) == 10 and all(r["status"] == "accepted" for r in results)
    for action in actions[len(results):]:
        results.append({**action, "status": "unexecuted_prior_refusal"})
    physical = plane_volume(sorted(removed), config.affine)
    target = plane_volume(sorted(removed), config.affine, spec["target_start_z_mm"])
    check_volume_bounds(physical, target, static, exact_target)
    attempts = [r["cut_union_attempt"] for r in results if "cut_union_attempt" in r]
    stats = engine.metrics()
    coverage = {"attempted_leaf_tests": sum(r["leaf_tests"] for r in attempts),
        "attempted_new_leaf_count": sum(r["new_leaf_count"] for r in attempts),
        "attempted_witness_rows": sum(r["witness_rows"] for r in attempts),
        "peak_tracked_cells": max((r["peak_tracked_cells"] for r in attempts), default=0),
        "budget_refusals": sum(bool(r["budget_refused"]) for r in attempts),
        "coverage_only_accepted_actions": sum(bool(r.get("cut_union_committed", {}).get("coverage_only_progress")) for r in results),
        "committed_new_leaf_count": sum(r.get("cut_union_committed", {}).get("new_leaf_count", 0) for r in results),
        "partially_certified_occupied_cells": stats["partially_certified_occupied_cells"],
        "final_coverage_digest": stats["cut_union_coverage_digest"],
        "scope": "attempted counters include rejected work; committed counters exclude discarded previews"}
    return {"row_id": config.case_id, "spacing_mm": spacing, "grid": grid,
        "shape": list(config.tissue_mask.shape), "affine": config.affine.tolist(),
        "source_hash": config.source_hash, "native_config_hash": config.fingerprint,
        "fixed_tool": asdict(tool), "physical_actions_sha256": BASE.digest(actions),
        "continuum_rationale": continuum_rationale(tool, spec), "cut_union_spec": config.cut_union.to_record(),
        "static_not_executable": static, "exact_continuous_target_cap_mm3": exact_target,
        "actions": results, "planned_actions": 10, "native_calls": len(attempts), "coverage": coverage,
        "native": {"feasible": completed, "complete_sequence": completed,
            "target_stroke_status": results[-1]["status"],
            "target_stroke_removed_physical_target_mm3": results[-1].get("removed_physical_target_mm3", 0.),
            "accepted_actions": len(engine.history), "removed_cells": len(removed),
            "removed_source_cell_mm3": len(removed)*config.voxel_volume_mm3,
            "removed_physical_mm3": physical, "removed_physical_target_mm3": target,
            "contacted_physical_upper_mm3": plane_volume(sorted(contacted), config.affine),
            "removed_outer_voxelization_excess_mm3": len(removed)*config.voxel_volume_mm3-physical,
            "partial_cells_kept_occupied": True,
            "frozen_reward_total": sum(r.get("frozen_reward", 0.) for r in results),
            "committed_history_hash": BASE.digest(engine.history)},
        "independent_audit": {"status": "pending" if engine.history else "not_applicable_rejected_no_committed_history"},
        "timing_seconds": {"numerical_row_total": time.perf_counter()-started}}, source, config, engine.history


def comparison_rows(prior, current):
    by_id = {r["row_id"]: r for r in current}
    out = []
    for baseline in prior["rows"]:
        experimental = by_id.get(baseline["row_id"])
        row = {"row_id": baseline["row_id"], "baseline": baseline["native"],
               "experimental_status": "missing" if experimental is None else "recorded"}
        if experimental is not None:
            for key in ("spacing_mm", "grid", "shape", "affine", "source_hash", "fixed_tool", "physical_actions_sha256"):
                if baseline[key] != experimental[key]:
                    raise AssertionError("Matched row changed physical inputs: " + key)
            row["experimental"] = experimental["native"]
            row["experimental_audit_status"] = experimental["independent_audit"]["status"]
            row["experimental_coverage_budget_refusal"] = bool(experimental["coverage"]["budget_refusals"])
            row["physical_removed_difference_mm3"] = experimental["native"]["removed_physical_mm3"]-baseline["native"]["removed_physical_mm3"]
            row["physical_target_difference_mm3"] = experimental["native"]["removed_physical_target_mm3"]-baseline["native"]["removed_physical_target_mm3"]
            reference = experimental["exact_continuous_target_cap_mm3"]
            row["baseline_target_shortfall_mm3"] = reference-baseline["native"]["removed_physical_target_mm3"]
            row["experimental_target_shortfall_mm3"] = reference-experimental["native"]["removed_physical_target_mm3"]
            delta = row["physical_target_difference_mm3"]
            row["target_error_direction"] = "unchanged" if abs(delta) <= 1e-8 else "improved" if delta > 0 else "worse"
            row["both_sequences_complete"] = baseline["native"]["complete_sequence"] and experimental["native"]["complete_sequence"]
        out.append(row)
    return out


def useful_criterion(prior, report):
    new = {r["row_id"]: r for r in report["rows"]}
    complete_old = {r["row_id"] for r in prior["rows"] if r["native"]["complete_sequence"]}
    complete_new = {r["row_id"] for r in report["rows"] if r["native"]["complete_sequence"]
                    and not r["coverage"]["budget_refusals"]}
    gained = sorted(complete_new-complete_old)
    regressed = sorted(complete_old-complete_new)
    all_rows = set(new) == {r["row_id"] for r in prior["rows"]}
    all_audits = all(r["independent_audit"].get("status") == "completed"
        and r["independent_audit"].get("certificate", {}).get("feasible") is True
        for r in report["rows"] if r["native"]["accepted_actions"])
    target_bounded = all(0 <= r["native"]["removed_physical_target_mm3"] <= r["exact_continuous_target_cap_mm3"] + 1e-8
                         for r in report["rows"])
    accepted = (report.get("status") == "completed" and report.get("source_unchanged") is True
                and all_rows and all_audits and target_bounded and bool(gained) and not regressed)
    return {"meets_predeclared_useful_criterion": accepted, "additional_complete_rows": gained,
        "prior_complete_rows_now_incomplete_or_missing": regressed,
        "all_18_rows_present": all_rows, "all_nonempty_prefix_audits_accepted": all_audits,
        "all_physical_target_volumes_within_exact_cap": target_bounded,
        "scope": "Complementary-cut representation only; no initial outer-voxelization repair, convergence, patient, or clinical claim"}


def _cpu():
    usage = resource.getrusage(resource.RUSAGE_SELF)
    return {"user_seconds": usage.ru_utime, "system_seconds": usage.ru_stime}


def worker(output):
    global DEADLINE, LIVE_OUTPUT
    sys.path.insert(0, str(ROOT / "src"))
    from resectionlab import evaluation
    spec = declaration()
    prior = retained_baseline(spec)
    union = cut_spec(spec).to_record()
    old = evaluation.independent_check_native_history
    def checked(*args, **kwargs):
        return old(*args, **kwargs, distance_backend="batch", distance_batch_size=256, cut_union_spec=union)
    evaluation.independent_check_native_history = checked
    RUN_COUNTS.update(native_strokes_started=0, native_strokes_returned=0, native_seconds=0.)
    COVERAGE_WORK.update({name: 0 for name in COVERAGE_WORK})
    cpu_before = _cpu()
    DEADLINE = time.perf_counter()+spec["caps"]["cooperative_worker_seconds"]
    LIVE_OUTPUT = output
    try:
        code = _original_worker(output)
    finally:
        evaluation.independent_check_native_history = old
        DEADLINE = None
        LIVE_OUTPUT = None
    report = json.loads((output/"report.json").read_text())
    report["planned_rows"] = [f"{g['name']}-{h:g}mm" for h in spec["spacings_mm"] for g in spec["grids"]]
    present = {r["row_id"] for r in report["rows"]}
    report["missing_rows"] = [r for r in report["planned_rows"] if r not in present]
    report["planned_action_slots"] = 180
    report["native_call_costs_including_incomplete_row"] = dict(RUN_COUNTS)
    report["returned_preview_coverage_work_including_incomplete_row"] = dict(COVERAGE_WORK)
    report["numerical_threads"] = 1
    report["worker_cpu"] = {k: _cpu()[k]-cpu_before[k] for k in cpu_before}
    report["independent_check"] = {"cut_union_spec": union, "distance_backend": "batch", "distance_batch_size": 256,
        "scope": "every nonempty committed prefix, including coverage-only actions and prefixes with a later refusal"}
    report["accepted_complete_sequences"] = sum(r["native"]["complete_sequence"] for r in report["rows"])
    report["verified_complete_sequences"] = sum(r["native"]["complete_sequence"] and
        r["independent_audit"].get("status") == "completed" and
        r["independent_audit"].get("certificate", {}).get("feasible") is True for r in report["rows"])
    report["coverage_budget_refused_rows"] = [r["row_id"] for r in report["rows"] if r["coverage"]["budget_refusals"]]
    report["retained_baseline"] = {**spec["retained_baseline"], "cost_comparison": "retained nonpaired timings; not a matched runtime speedup measurement",
        "worker_seconds": prior["worker_seconds"], "peak_worker_rss_bytes": prior["peak_worker_rss_bytes"],
        "native_call_costs": prior["native_call_costs_including_incomplete_row"]}
    report["matched_comparison"] = comparison_rows(prior, report["rows"])
    report["useful_criterion"] = useful_criterion(prior, report)
    report["resource_comparison"] = {
        "scope": "Retained nonpaired costs; ratios are descriptive, not matched speedup estimates",
        "worker_runtime_ratio_to_retained": report.get("worker_seconds", 0)/prior["worker_seconds"],
        "worker_peak_rss_ratio_to_retained": report.get("peak_worker_rss_bytes", 0)/prior["peak_worker_rss_bytes"],
        "native_calls_current": RUN_COUNTS["native_strokes_started"],
        "native_calls_retained": prior["native_call_costs_including_incomplete_row"]["native_strokes_started"],
        "baseline_cpu_available": False}
    BASE.write_json(output/"report.json", report)
    return code


def launch(output):
    cpu_before = _cpu()
    code = _original_launch(output)
    path = output/"launcher.json"
    record = json.loads(path.read_text())
    record["parent_cpu"] = {k: _cpu()[k]-cpu_before[k] for k in cpu_before}
    record["parent_peak_rss_bytes"] = BASE.peak_rss_bytes()
    spec = declaration()
    planned = [f"{g['name']}-{h:g}mm" for h in spec["spacings_mm"] for g in spec["grids"]]
    try:
        report = json.loads((output/"report.json").read_text())
    except (OSError, ValueError):
        report = {"rows": []}
    seen = {r["row_id"] for r in report.get("rows", [])}
    record["planned_rows"] = planned
    record["missing_rows"] = [r for r in planned if r not in seen]
    record["planned_action_slots"] = 180
    try:
        record["last_worker_progress"] = json.loads((output/"progress.json").read_text())
    except (OSError, ValueError):
        record["last_worker_progress"] = None
    record["cost_scope"] = "Current experimental parent/worker costs; baseline timing retained from earlier run, not paired"
    BASE.write_json(path, record)
    return code


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
    return worker(args.output) if args.worker else launch(args.output)


if __name__ == "__main__":
    raise SystemExit(main())
