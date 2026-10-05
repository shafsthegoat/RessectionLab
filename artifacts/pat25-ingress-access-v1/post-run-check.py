#!/usr/bin/env python3
"""Prospective saved-only checker. Run only after root confirms termination.

Imports no project/solver/policy code. NumPy operates on saved small coordinate
arrays only. Bundle bytes may be hashed, never decoded. Prints a receipt; does
not modify producer outputs or silently overwrite a prior independent result.
"""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
from itertools import product
import json
import math
from pathlib import Path
import subprocess
import tarfile
import time

import numpy as np

SCHEMA = "pat25-ingress-saved-independent-check-v1"
VERSION = "pat25-screened-ingress-initial-comparison-v1"
REVIEWED_SOURCE_SHA256 = {
    "scripts/diagnose_pat25_ingress_access.py": "b3afc7510fcfcc3f53b0535c7128296f130b82431fb83d061058a8658cbe10e2",
    "src/resectionlab/native_ingress.py": "cd27eb57062d0afd72dba3b392f9ed03973a9fe5b353bb8fd0da74e6a1c6761f",
}
ATOL = 1e-12
FRAME_ATOL_MM = 1e-8
ROUNDING_BAND = 64 * np.finfo(np.float64).eps
FAMILIES = ("exposed_opening", "proximal_nominal", "distal_nominal")
EXITS = list(product(range(3), (-1, 1)))
SETTINGS = {"whole_worker_envelope_seconds": 180., "cooperative_seconds": 170.,
    "max_wall_seconds": 174., "max_rss_bytes": 6 * 1024**3, "cpu_threads": 1,
    "max_static_screens": 468, "max_initial_previews_per_inventory": 78,
    "max_total_previews": 156, "max_initial_inventories": 2, "automatic_retry": False}
LIMITS = [
    "No dense patient mask is decoded: blocked-cell membership, first-cell minimality, full-mask hashes, nominal crop integrals and clearance cannot be independently recomputed.",
    "Source/Git/archive/receipt bindings and small-coordinate arithmetic are checked, not protection against an authorized host rewriting the entire history.",
    "No-fallback/call counts and unchanged state are reconciled from the frozen code and saved trace/invariant receipts; this is not independent operating-system instruction tracing.",
    "Recorded sampled RSS can miss transient peaks. Saved supervision establishes its reported outcome, not an independent OS-enforced hard memory limit.",
    "The outer clock begins at its first stdlib timestamp; preceding interpreter startup and OS scheduling are explicitly not bounded by that receipt.",
    "Centerline/crop visibility and initial ingress do not certify whole-tool visibility, later stroke feasibility, skull access, clinical safety or efficacy.",
    "Target-center exclusions in saved envelope summaries cannot be reconstructed without the nominal array; their bounds/nulls and envelope arithmetic can be checked."]


def require(condition, label):
    if not condition:
        raise AssertionError(label)


def exact(actual, expected, label):
    require(actual == expected, label)


def close(actual, expected, label, atol=ATOL):
    first, second = np.asarray(actual, dtype=float), np.asarray(expected, dtype=float)
    require(first.shape == second.shape and np.isfinite(first).all()
            and np.isfinite(second).all() and np.all(np.abs(first - second) <= atol), label)


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()


def semantic(value):
    return "sha256:" + hashlib.sha256(canonical(value)).hexdigest()


def digest(path):
    value = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            value.update(chunk)
    return value.hexdigest()


def parse(raw):
    def invalid(value):
        raise ValueError("Nonfinite JSON: " + value)
    return json.loads(raw, parse_constant=invalid)


def load(path):
    return parse(Path(path).read_bytes())


def interval(first, last):
    lower, upper = 0., 1.
    for start, end in zip(first, last):
        delta = end - start
        if delta == 0:
            if not -1 <= start <= 1:
                return None
        else:
            enter, leave = sorted(((-1 - start) / delta, (1 - start) / delta))
            lower, upper = max(lower, enter), min(upper, leave)
            if upper < lower:
                return None
    return [float(lower), float(upper)]


def domain(points, affine, shape, *, cells=False):
    inverse = np.linalg.inv(np.asarray(affine, float))
    voxels = points @ inverse[:3, :3].T + inverse[:3, 3]
    grid = 2 * (voxels + .5) / shape - 1 if cells else 2 * voxels / (shape - 1) - 1
    grid = np.where(np.abs(np.abs(grid) - 1) <= ROUNDING_BAND, np.sign(grid), grid)
    return grid, (np.abs(grid) <= 1).all(axis=1)


def segment(saved, start, end, view):
    shape, affine = np.asarray(view["shape"]), np.asarray(view["affine_ras_mm"])
    points = np.asarray(start) + np.linspace(0., 1., 5)[:, None] * (np.asarray(end) - start)
    close(saved["sample_points_ras_mm"], points, "five saved physical sample points", FRAME_ATOL_MM)
    for cells, label in ((False, "center_domain"), (True, "fullcell_extent")):
        grid, inside = domain(points, affine, shape, cells=cells)
        exact(saved["sample_inside_" + label], inside.tolist(), label + " sample inclusion")
        span = interval(grid[0], grid[-1])
        if span is None:
            exact(saved[label + "_interval"], None, label + " absent interval")
        else:
            close(saved[label + "_interval"], span, label + " interval")
        close(saved["continuous_" + label + "_fraction"], 0. if span is None else span[1] - span[0], label + " fraction")
    coverage = np.asarray(saved["sample_channel_coverage_fraction"])
    require(coverage.shape == (5, 6) and np.isfinite(coverage).all()
            and np.all((coverage >= 0) & (coverage <= 1 + ATOL)), "fractional channel sample domain")
    _, inside = domain(points, affine, shape)
    require(np.all(coverage[~inside] == 0), "outside image cannot become known")
    for channel, (available, bounds) in enumerate(zip(view["channel_available"], view["coverage_fraction_range"])):
        if not available:
            require(np.all(coverage[:, channel] == 0), "unavailable channel samples must remain zero")
        # Constant saved coverage admits an independent value check without a dense mask.
        if bounds[0] == bounds[1]:
            close(coverage[inside, channel], np.full(inside.sum(), bounds[0]), "constant coverage interpolation")
    return inside


def proposal_ledger(ledger, common, *, native=False):
    reason_key = "proposal_reason" if native else "reason"
    offsets = common["adapter_options"]["proposal_config"]["offsets_source_voxels"]
    expected = [(column, offset, tool["tool_id"], family)
                for column, offset in enumerate(offsets) for tool in common["tools"] for family in FAMILIES]
    exact(len(ledger), 78, "complete 78-slot family/tool/column ledger")
    exact([(row["column_index"], row["offset_source_voxels"], row["tool_id"], row["family"])
           for row in ledger], expected, "complete stable ordered ledger")
    counts = Counter(row[reason_key] for row in ledger)
    exact(counts.get("CANDIDATE_CAP", 0), 0, "no capped dispositions")
    proposed = [row for row in ledger if row[reason_key] == "PROPOSED_UNCERTIFIED"]
    ids = [row["proposal_id"] for row in proposed]
    require(len(ids) == len(set(ids)), "unique proposed identifiers")
    for row in ledger:
        if row[reason_key] == "DUPLICATE_GEOMETRY":
            require(row["proposal_id"] in ids, "duplicate references an emitted proposal")
    return proposed, dict(counts)


def screening_check(screen, derived, common, native_affine):
    if screen["status"] == "not_executed":
        return {"status": "not_executed", "complete_exits": 0, "static_checks": 0}
    exact(screen["version"], "six-axis-any-entry-native-ingress-v1", "screen version")
    exact(screen["complete_stroke_certified"], False, "no complete-stroke authority")
    exact(screen["removal_authorized"], False, "no removal authority")
    rows = screen["exits"]
    exact([(r["axis"], r["outward_sign"]) for r in rows], EXITS, "all six exit statuses retained")
    unique_count, completed = 0, 0
    for row, source in zip(rows, derived):
        for key in ("axis", "outward_sign", "distance_mm"):
            exact(row[key], source[key], "screened exit metadata")
        if row["status"] != "screened":
            continue
        completed += 1
        exact(row["access"], source["access"], "screened access geometry")
        exact(row["cavity_state_hash"], "sha256:" + hashlib.sha256(
            (row["engine_model_hash"] + ":initial").encode()).hexdigest(), "screen initial state chain")
        batch = row["proposal_dispositions"]
        exact(batch["cavity_state_hash"], row["cavity_state_hash"], "batch initial state")
        exact(batch["model_hash"], row["provider_model_hash"], "batch provider")
        proposed, counts = proposal_ledger(batch["ledger"], common)
        exact(batch["counts"], counts, "screen dispositions")
        exact(batch["slot_count"], 78, "screen slot count")
        exact(batch["emitted_count"], len(batch["proposals"]), "screen emitted count")
        exact([r["proposal_id"] for r in proposed], [r["proposal_id"] for r in batch["proposals"]], "batch emission order")
        exact([r["proposal_id"] for r in row["poses"]], [r["proposal_id"] for r in batch["proposals"]], "every emitted pose screened")
        seen = {}
        for pose, ray in zip(row["poses"], batch["proposals"]):
            for key in ("proposal_id", "tool_id", "entry_mm", "tip_mm"):
                exact(pose[key], ray[key], "screen ray binding")
            identifier = "nominal-cavity-" + semantic({"model": batch["model_hash"],
                "cavity": batch["cavity_state_hash"], "geometry": [ray["tool_id"], ray["entry_mm"], ray["tip_mm"]]}).split(":", 1)[1][:24]
            exact(ray["proposal_id"], identifier, "source/cavity/geometry proposal identity")
            entry, tip = np.asarray(ray["entry_mm"], np.float64), np.asarray(ray["tip_mm"], np.float64)
            direction = (tip - entry) / np.linalg.norm(tip - entry)
            require(np.array_equal(direction, np.asarray(pose["axis_unit"])), "exact raw float64 direction")
            signature = (ray["tool_id"], entry.tobytes(), direction.tobytes())
            if signature in seen:
                prior = seen[signature]
                exact(pose["duplicate_of"], prior["proposal_id"], "reuse references exact first pose")
                for key in ("screen_index", "admissible", "geometry", "blocked_cell_count",
                            "first_blocked_cell", "first_blocked_cell_mm", "reasons"):
                    exact(pose[key], prior[key], "reused result equality")
            else:
                exact(pose["duplicate_of"], None, "unique pose cannot claim reuse")
                exact(pose["screen_index"], unique_count, "global unique static check count")
                unique_count += 1
                seen[signature] = pose
            blocked = pose["blocked_cell_count"]
            require(type(blocked) is int and blocked >= 0, "nonnegative blocked-cell count")
            if blocked == 0:
                exact([pose["first_blocked_cell"], pose["first_blocked_cell_mm"]], [None, None], "empty witness nulls")
            else:
                cell = np.asarray(pose["first_blocked_cell"])
                require(cell.shape == (3,) and cell.dtype.kind in "iu" and np.all(cell >= 0), "source-cell witness")
                close(pose["first_blocked_cell_mm"], native_affine[:3, :3] @ cell + native_affine[:3, 3], "witness physical frame", FRAME_ATOL_MM)
            reasons = ["HARD_GEOMETRY:" + failure["reason"] for failure in pose["geometry"]["failures"]]
            if blocked:
                reasons.append("SHAFT_BLOCKED_BY_REMAINING_NATIVE_TISSUE")
            exact(pose["reasons"], reasons, "static reasons from saved hard/shaft results")
            exact(pose["admissible"], not reasons, "ANY-pose admissibility")
        exact(row["eligible"], any(p["admissible"] for p in row["poses"]), "ANY entry/tool eligibility")
    require(0 <= screen["screen_count"] <= 468, "static budget")
    if screen["complete"]:
        exact(completed, 6, "complete selection requires every exit")
        exact(screen["screen_count"], unique_count, "exact static reuse accounting")
        eligible = [row for row in rows if row["eligible"]]
        winner = min(eligible, key=lambda r: (r["distance_mm"], r["axis"], r["outward_sign"])) if eligible else None
        exact(screen["selected_exit"], None if winner is None else [winner["axis"], winner["outward_sign"]], "distance/axis/sign selection")
        exact(screen["status"], "selected" if eligible else "no_ingress_admissible_access", "complete selection status")
    else:
        exact(screen["selected_exit"], None, "incomplete screening cannot select")
        require(unique_count <= screen["screen_count"], "partial count cannot shrink")
    return {"status": screen["status"], "complete_exits": completed, "static_checks": screen["screen_count"],
            "eligible_exits": [[r["axis"], r["outward_sign"]] for r in rows if r["eligible"]],
            "selected_exit": screen["selected_exit"]}


def inventory_check(record, common, screening_row):
    if record["status"] != "complete":
        return {"status": record["status"], "started_previews": record.get("started_preview_calls", 0),
                "completed_trace_rows": len(record.get("partial_preview_trace", []))}
    inventory, trace = record["initial_inventory"], record["trace"]
    proposed, counts = proposal_ledger(inventory["ledger"], common, native=True)
    exact(proposed, inventory["emitted"], "native emitted rows equal proposed ledger rows")
    exact(inventory["disposition_counts"], counts, "native disposition counts")
    emitted = inventory["emitted"]
    accepted = [row for row in emitted if row["feasible"]]
    exact([inventory["complete"], inventory["ledger_complete"], inventory["crop_clipping"]], [True, True, False], "complete uncropped inventory")
    for key in ("declared_slots",):
        exact(inventory[key], 78, key)
    for key in ("emitted_count", "evaluated_slots"):
        exact(inventory[key], len(emitted), key)
    exact(inventory["accepted_count"], len(accepted), "accepted count")
    exact(inventory["rejected_count"], len(emitted) - len(accepted), "rejected count")
    exact(inventory["omitted_count"], 0, "no omitted family slot")
    exact(inventory["duplicate_count"], counts.get("DUPLICATE_GEOMETRY", 0), "duplicate slot count")
    exact(inventory["unavailable_count"], 78 - len(emitted) - inventory["duplicate_count"], "unavailable slot count")
    exact(record["preview_calls"], len(trace), "all started native previews returned")
    exact(len(trace), len(emitted), "one full preview per unique emitted ray")
    require(len(trace) <= 78, "per-inventory preview budget")
    exact(record["reward"], common["objective"], "frozen reward")
    exact(record["horizon"], common["max_steps"], "frozen horizon")
    exact([inventory["steps_taken"], inventory["max_steps"], inventory["remaining_steps"], inventory["terminal"]], [0, common["max_steps"], common["max_steps"], False], "initial state/horizon")
    exact(record["invariants"], record["invariants_after"], "before/after task invariants")
    invariant = record["invariants"]
    exact(invariant["inventory_hash"], semantic(inventory), "inventory semantic hash")
    physical = [{key: row[key] for key in ("tool_id", "entry_mm", "tip_mm", "feasible", "reason")} for row in emitted]
    exact(invariant["physical_inventory_hash"], semantic(physical), "physical inventory hash")
    for key in ("source_hash", "decision_model_hash", "cavity_state_hash"):
        exact(invariant[key], inventory[key], "inventory/invariant " + key)
    exact(record["decision_model_hash"], inventory["decision_model_hash"], "decision model")
    if screening_row is not None:
        batch = screening_row["proposal_dispositions"]
        for key, target in (("provider_model_hash", "model_hash"), ("cavity_state_hash", "cavity_state_hash"),
                            ("nominal_target_hash", "nominal_target_hash"), ("nominal_provenance_hash", "nominal_provenance_hash")):
            exact(inventory[key], batch[target], "screen/native " + key)
        exact([r["action_id"] for r in emitted], [r["proposal_id"] for r in batch["proposals"]], "screen/native identical proposal IDs")
    for ray, returned in zip(emitted, trace):
        for key in ("tool_id", "entry_mm", "tip_mm", "feasible", "reason"):
            exact(ray[key], returned[key], "actual preview result binding")
        exact(returned["source_state_hash"], invariant["cavity_state_hash"], "preview initial state")
        require(returned["completed_preview_microsteps"] >= 0 and returned["preview_removed_cells"] >= 0, "uncommitted preview counts")
        if returned["reason"] == "SHAFT_BLOCKED_BY_REMAINING_NATIVE_TISSUE":
            require(returned["blocked_cell_count"] > 0 and returned["failure_step_index"] >= 0
                    and 0 <= returned["failure_insertion_fraction"] <= 1, "native shaft failure trace")
    return {"status": "complete", "started_previews": len(trace), "completed_trace_rows": len(trace),
            "accepted": len(accepted), "rejected": len(emitted) - len(accepted),
            "by_tool": {tool["tool_id"]: dict(Counter("accepted" if r["feasible"] else "rejected"
                for r in emitted if r["tool_id"] == tool["tool_id"])) for tool in common["tools"]}}


def coverage_check(record, common):
    if record["status"] != "complete":
        return None
    coverage, visibility = record["actor_coverage"], record["actor_visibility"]
    view, emitted = visibility["view"], record["initial_inventory"]["emitted"]
    affine, shape = np.asarray(view["affine_ras_mm"]), np.asarray(view["shape"])
    exact(view["shape"], common["adapter_options"]["crop_shape"], "unchanged actor shape")
    exact(coverage["crop_affine_ras_mm"], view["affine_ras_mm"], "coverage/view affine")
    for cells, key in ((False, "center_corners_ras_mm"), (True, "fullcell_corners_ras_mm")):
        corners = np.asarray(list(product(*[(-.5, n - .5) if cells else (0, n - 1) for n in shape])))
        close(view[key], corners @ affine[:3, :3].T + affine[:3, 3], "actor corners", FRAME_ATOL_MM)
    require(0 <= view["roundtrip_max_error_mm"] <= FRAME_ATOL_MM, "recorded roundtrip bound")
    native = np.asarray(coverage["native_physical_affine_ras_mm"])
    origin = np.linalg.solve(native[:3, :3], affine[:3, 3] - native[:3, 3])
    close(coverage["crop_origin_source_voxels"], origin, "crop origin", FRAME_ATOL_MM)
    close(coverage["crop_last_center_source_voxels"], origin + shape - 1, "crop last center", FRAME_ATOL_MM)
    total, visible = coverage["nominal_target_mass_total"], coverage["nominal_target_mass_in_crop"]
    if total is None or total == 0:
        exact(coverage["nominal_target_mass_fraction_visible"], None, "undefined target fraction remains null")
    else:
        require(0 <= visible <= total + ATOL, "visible target mass bounds")
        close(coverage["nominal_target_mass_fraction_visible"], visible / total, "target mass ratio")
    exact(visibility["cavity_source_cells"], 0, "initial source cavity")
    exact(visibility["cavity_visible_cells"], 0, "initial crop cavity")
    exact(visibility["cavity_channel_available"], True, "known initial empty cavity")
    exact([row["action_id"] for row in visibility["actions"]], [row["action_id"] for row in emitted], "all emitted rays visible in diagnostics")
    tools = {tool["tool_id"]: tool for tool in common["tools"]}
    samples = {}
    for action, row in zip(visibility["actions"], emitted):
        exact(action["accepted"], row["feasible"], "visibility acceptance")
        entry, tip = np.asarray(row["entry_mm"]), np.asarray(row["tip_mm"])
        axis = (tip - entry) / np.linalg.norm(tip - entry)
        tool = tools[row["tool_id"]]
        spans = {"entry_to_tip": (entry, tip),
            "approach_shaft_centerline": (entry - tool["working_length_mm"] * axis, entry - tool["tip_length_mm"] * axis),
            "deepest_shaft_centerline": (tip - tool["working_length_mm"] * axis, tip - tool["tip_length_mm"] * axis)}
        for name, (first, last) in spans.items():
            inside = segment(action["segments"][name], first, last, view)
            if name == "entry_to_tip":
                samples[row["action_id"]] = (inside, action["segments"][name]["continuous_center_domain_fraction"])
    accepted_ids = [row["action_id"] for row in emitted if row["feasible"]]
    exact([row["action_id"] for row in coverage["actions"]], accepted_ids, "accepted-only actor coverage")
    for row in coverage["actions"]:
        inside, fraction = samples[row["action_id"]]
        exact([row["ray_samples"], row["ray_samples_inside"], row["entry_inside_center_bounds"], row["tip_inside_center_bounds"]], [5, int(inside.sum()), bool(inside[0]), bool(inside[-1])], "five-point actor coverage")
        close(row["sample_fraction_inside"], inside.mean(), "sample fraction")
        close(row["straight_segment_fraction_inside"], fraction, "continuous actor fraction")
    counts = Counter("none" if not samples[key][0].any() else "full" if samples[key][0].all() else "partial" for key in accepted_ids)
    exact(coverage["sample_coverage_counts"], {key: counts[key] for key in ("none", "partial", "full")}, "accepted coverage denominator")
    exact(coverage["legal_nonstop_actions"], len(accepted_ids), "nonstop denominator")
    proposal = record["proposal_coverage"]
    exact(proposal["catalog_metadata"], {k: v for k, v in record["initial_inventory"].items() if k not in {"emitted", "ledger"}}, "proposal catalog join")
    for which, items in (("emitted", emitted), ("accepted", [r for r in emitted if r["feasible"]])):
        envelope = proposal[which + "_envelope"]
        exact(envelope["proposal_count"], len(items), "envelope count")
        if not items:
            require(all(value is None for key, value in envelope.items() if key != "proposal_count"), "empty envelope remains null")
        else:
            starts, tips, radii, depths = [], [], [], []
            normal, center = np.asarray(record["exit"]["access"]["normal_inward"]), np.asarray(record["exit"]["access"]["center_mm"])
            for row in items:
                entry, tip = np.asarray(row["entry_mm"]), np.asarray(row["tip_mm"])
                tool = tools[row["tool_id"]]
                starts.append(entry - tool["tip_length_mm"] * (tip - entry) / np.linalg.norm(tip - entry))
                tips.append(tip); radii.append(tool["tip_radius_mm"]); depths.append(float((tip - center) @ normal))
            low = np.min(np.minimum(starts, tips) - np.asarray(radii)[:, None], axis=0)
            high = np.max(np.maximum(starts, tips) + np.asarray(radii)[:, None], axis=0)
            close(envelope["active_sweep_aabb_ras_mm"], [low, high], "saved active-tip AABB", FRAME_ATOL_MM)
            upper = max(max(float((start - center) @ normal), depth) + radius for start, depth, radius in zip(starts, depths, radii))
            close(envelope["distal_active_tip_axial_upper_bound_mm"], upper, "saved active-tip axial bound", FRAME_ATOL_MM)
            close(envelope["endpoint_depth_range_mm"], [min(depths), max(depths)], "endpoint range", FRAME_ATOL_MM)
            for key in ("target_centers_beyond_axial_bound", "target_centers_outside_aabb"):
                count = envelope[key]
                require(count is None or type(count) is int and 0 <= count <= proposal["nominal_target_centers_total"], "reported target exclusion bounds")
    return {"target_fraction_visible": coverage["nominal_target_mass_fraction_visible"],
            "accepted_denominator": len(accepted_ids), "accepted_samples": coverage["sample_coverage_counts"],
            "cavity_cells": 0, "emitted_segments_checked": 3 * len(emitted),
            "mean_accepted_ray_fraction": None if not accepted_ids else float(np.mean([samples[key][1] for key in accepted_ids]))}


def historical_preparation(root, prior):
    record = prior["original_preparation"]
    path = root / record["path"]
    if path.is_file():
        raw = path.read_bytes()
    else:
        with tarfile.open(root / record["archive_fallback"]) as archive:
            member = archive.getmember("sub-PAT25/preparation.json")
            require(member.isfile(), "historical preparation is a regular member")
            raw = archive.extractfile(member).read()
    exact(hashlib.sha256(raw).hexdigest(), record["sha256"], "historical preparation bytes")
    value = parse(raw)
    exact(value["binding_hash"], record["binding_hash"], "historical binding identity")
    exact(semantic(value["binding"]), record["binding_hash"], "historical binding recomputation")
    return value


def check_run(args):
    require(args.root_confirmed_terminal, "Root termination confirmation is mandatory")
    root, run = args.repository.resolve(), args.run.resolve()
    outer_path, outer_log = run.with_name(run.name + ".outer.json"), run.with_name(run.name + ".outer.log")
    def output_hashes():
        paths = [p for p in sorted(run.iterdir()) if p.is_file()] + [outer_path, outer_log]
        return {str(p): digest(p) for p in paths}
    before = output_hashes()
    report, acceptance, supervisor = load(run / "receipt.json"), load(run / "acceptance.json"), load(run / "supervisor.json")
    outer = load(outer_path)
    manifest, release = load(args.manifest), load(args.release)
    exact(manifest["version"], VERSION, "prospective version")
    exact([manifest["subject"], manifest["role"]], ["sub-PAT25", "TRAIN"], "fixed patient role")
    exact(manifest["settings"], SETTINGS, "fixed resource/call settings")
    for name, expected in REVIEWED_SOURCE_SHA256.items():
        exact(manifest["source_sha256"][name], expected, "reviewed ingress source " + name)
    exact(report["settings"], SETTINGS, "executed settings")
    exact([report["version"], report["subject"], report["role"]], [VERSION, "sub-PAT25", "TRAIN"], "executed identity")
    exact(release["authorized"], True, "separate execution authorization")
    exact(release["declaration_sha256"], digest(args.manifest), "release declaration")
    exact(Path(release["output"]).resolve(), run, "released output directory")
    exact(release["phase"], "initial_inventory_diagnostic", "initial-only release")
    bound = report["execution_binding"]
    exact(bound["release_sha256"], digest(args.release), "worker release identity")
    exact(bound["declaration_sha256"], digest(args.manifest), "worker declaration identity")
    exact(bound["source_commit"], release["source_commit"], "worker committed source")
    exact(outer["manifest_sha256"], digest(args.manifest), "outer-to-worker manifest byte pin")
    exact(outer["release_sha256"], digest(args.release), "outer-to-worker release byte pin")
    exact(outer["acceptance_sha256"], digest(run / "acceptance.json"), "outer final acceptance identity")
    exact([outer["whole_attempt_seconds_limit"], outer["soft_stop_seconds"], outer["hard_cleanup_seconds"],
           outer["max_group_rss_bytes"], outer["automatic_retry"]], [180., 174., 178., 6 * 1024**3, False], "outer declared limits")
    archive_root = Path(release["archive_root"])
    closure = {**manifest["source_sha256"], **manifest["metadata_sha256"]}
    exact(digest(args.source_archive), args.source_archive_sha256, "source archive bytes")
    with tarfile.open(args.source_archive) as archive:
        exact(archive.pax_headers.get("comment"), release["source_commit"], "Git archive commit")
        for name, expected in closure.items():
            require(not Path(name).is_absolute() and ".." not in Path(name).parts, "safe source member")
            member = archive.getmember(name)
            require(member.isfile(), "regular committed member")
            raw = archive.extractfile(member).read()
            exact(hashlib.sha256(raw).hexdigest(), expected, "archived source/metadata " + name)
            committed = subprocess.run(["git", "-C", str(root), "show", release["source_commit"] + ":" + name],
                capture_output=True, check=True, timeout=15).stdout
            exact(committed, raw, "exact committed source/metadata " + name)
            exact(digest(archive_root / name), expected, "executed source/metadata " + name)
            exact(bound["inputs_before"][str(archive_root / name)], expected, "before-source binding " + name)
    for path, expected in bound["inputs_before"].items():
        exact(digest(path), expected, "unchanged bound input " + path)
    baseline = load(run / "execution-baseline.json")
    exact(baseline["inputs"], bound["inputs_before"], "parent/worker input closure")
    exact(baseline["source_commit"], release["source_commit"], "parent source commit")
    prior, common = manifest["legacy_input"], manifest["legacy_input"]["common_task"]
    saved = historical_preparation(archive_root, prior)
    exact([saved["subject"], saved["role"], saved["status"]], ["sub-PAT25", "TRAIN", "prepared"], "historical original")
    exact(common, saved["binding"]["common_task"], "historical tools/track/reward/horizon/adapter options")
    for key, value in prior["member"].items():
        exact(saved["binding"]["member"][key], value, "historical patient/source " + key)
    exact(prior["member"]["case_bundle_sha256"], bound["inputs_before"][str(archive_root / prior["member"]["case_bundle"])], "original encoded bundle")
    exact(prior["support_acknowledgment"], saved["binding"]["member"]["research_support_acknowledgment"], "unreviewed support unchanged")
    exact([prior["support_acknowledgment"]["cortical_access_permitted"],
           prior["support_acknowledgment"]["clinical_use_permitted"]], [False, False], "support is not promoted to approval")
    exact(prior["six_exits"], saved["binding"]["member"]["access_derivation"]["six_axis_exit_distances_mm"], "original six exits")
    cohort = load(archive_root / prior["cohort_path"])
    exact(digest(archive_root / prior["cohort_path"]), prior["cohort_sha256"], "prospective cohort identity")
    role = next(row for row in cohort["candidates"] if row["subject"] == "sub-PAT25")
    exact(role["development_role"], "population_training", "no role reassignment")
    derived = report.get("derived_exits", [])
    if derived:
        exact([(row["axis"], row["outward_sign"]) for row in derived], EXITS, "six derived exits")
        original_affine = np.asarray(saved["coverage"]["actor"]["original_source_affine_ras_mm"])
        derivation = saved["binding"]["member"]["access_derivation"]
        representative = np.asarray(derivation["representative_voxel"])
        for row, original in zip(derived, prior["six_exits"]):
            for key, value in original.items():
                exact(row[key], value, "unchanged original exit " + key)
            axis, sign = row["axis"], row["outward_sign"]
            boundary = np.asarray(row["boundary_voxel"])
            other = [dim for dim in range(3) if dim != axis]
            exact(boundary[other].tolist(), representative[other].tolist(), "unchanged transverse axis walk")
            require(sign * (boundary[axis] - representative[axis]) > 0, "outward boundary direction")
            close(row["distance_mm"], abs(boundary[axis] - representative[axis])
                  * np.linalg.norm(original_affine[:3, axis]), "source-axis physical distance", FRAME_ATOL_MM)
            close(row["access"]["center_mm"], original_affine[:3, :3] @ boundary + original_affine[:3, 3], "derived aperture center", FRAME_ATOL_MM)
            close(row["access"]["normal_inward"], -sign * original_affine[:3, axis] / np.linalg.norm(original_affine[:3, axis]), "derived inward axis", ATOL)
            exact(row["access"]["radius_mm"], 6., "original radius")
        require(sum(row["selected_original"] for row in derived) == 1, "one baseline exit")
        shortest = min(derived, key=lambda row: (row["distance_mm"], row["axis"], row["outward_sign"]))
        exact(shortest["selected_original"], True, "original deterministic shortest-exit baseline")
    native = np.asarray(saved["coverage"]["actor"]["native_physical_affine_ras_mm"])
    screen = screening_check(report["screening"], derived, common, native)
    comparisons = {}
    for name in ("original", "screened_selected"):
        row = report[name]
        screen_row = None
        if row["status"] == "complete":
            key = [row["exit"]["axis"], row["exit"]["outward_sign"]]
            if report["screening"].get("exits"):
                screen_row = next(r for r in report["screening"]["exits"] if [r["axis"], r["outward_sign"]] == key)
                if screen_row["status"] != "screened":
                    screen_row = None
            if name == "original":
                exact(row["initial_inventory"], saved["initial_inventory"], "original full inventory unchanged")
                exact(row["decision_model_hash"], saved["binding"]["decision_model_hash"], "original model unchanged")
            else:
                exact(key, report["screening"]["selected_exit"], "no full-preview fallback selection")
            exact(row["actor_coverage"]["original_source_affine_ras_mm"], saved["coverage"]["actor"]["original_source_affine_ras_mm"], "source frame unchanged")
            exact(row["actor_coverage"]["native_physical_affine_ras_mm"], native.tolist(), "native frame unchanged")
            for key in ("source_shape", "nominal_target_available", "nominal_target_positive_voxels_total", "nominal_target_mass_total"):
                exact(row["actor_coverage"][key], saved["coverage"]["actor"][key], "unchanged whole-source nominal metadata " + key)
        comparisons[name] = inventory_check(row, common, screen_row)
        comparisons[name]["coverage"] = coverage_check(row, common)
    total = sum(row["started_previews"] for row in comparisons.values())
    exact(report["total_preview_calls"], total, "total native preview count")
    require(total <= 156, "whole diagnostic native preview cap")
    for key in ("executed_transitions", "commits", "searches", "policy_forwards", "checkpoints", "optimizer_updates"):
        exact(report[key], 0 if report["observer_violation"] is None else None, "observer-backed " + key)
    if report["observer_violation"] is not None:
        exact(report["observer_health_verified"], False, "lost observer authority stays unverified")
    exact(report["automatic_retry"], False, "no retry")
    exact(acceptance["worker_receipt_sha256"], digest(run / "receipt.json"), "parent receipt binding")
    exact(acceptance["supervisor"], supervisor, "parent supervisor binding")
    exact(supervisor["declaration_sha256"], digest(args.manifest), "supervisor declaration")
    require(type(supervisor["returncode"]) is int, "actual worker return code")
    exact(supervisor["automatic_retry"], False, "supervisor no retry")
    exact(supervisor["inherits_outer_process_group"], True, "single outer-owned process group")
    exact(acceptance["automatic_retry"], False, "parent no retry")
    for key, value in SETTINGS.items():
        if key == "max_wall_seconds":
            require(0 < acceptance["supervisor_settings"][key] <= value, "remaining inner wall budget")
        else:
            exact(acceptance["supervisor_settings"][key], value, "unchanged inner resource setting")
    require(report["peak_rss_bytes"] >= 0 and supervisor["sampled_peak_rss_bytes"] >= 0, "recorded RSS")
    complete = outer["status"] == "complete"
    if complete:
        exact(acceptance["status"], "complete", "outer success requires inner acceptance")
        exact([outer["returncode"], outer["group_gone"], outer["reason"]], [0, True, None], "outer execution/cleanup outcome")
        require(0 <= outer["elapsed_seconds"] < 180 and 0 <= outer["sampled_peak_group_rss_bytes"] <= 6 * 1024**3,
                "recorded whole-attempt wall/group RSS bounds")
        exact([report["observer_health_verified"], report["observer_violation"]], [True, None], "healthy initial-only observer")
        exact([args.terminal_exit, supervisor["returncode"]], [0, 0], "successful authoritative parent/worker exit")
        exact([report["status"], supervisor["status"]], ["complete", "complete"], "successful worker/supervisor")
        exact([report["inputs_unchanged"], acceptance["inputs_unchanged"]], [True, True], "successful source preservation")
        require(report["elapsed_seconds"] < 170 and supervisor["seconds"] <= 180
                and report["peak_rss_bytes"] <= 6 * 1024**3
                and supervisor["sampled_peak_rss_bytes"] <= 6 * 1024**3, "recorded completion resource bounds")
        exact([supervisor["timed_out"], supervisor["termination_reason"]], [False, None], "no recorded limit termination")
        require(report["screening"]["complete"], "successful diagnostic requires complete screening")
        if report["screening"]["selected_exit"] is None:
            exact(report["screened_selected"]["status"], "not_executed_no_ingress_admissible_access", "no-initial-access null comparison")
        else:
            exact(report["screened_selected"]["status"], "complete", "selected native inventory completed")
    else:
        require(args.terminal_exit != 0, "failed acceptance matches authoritative parent failure")
    after = output_hashes()
    exact(after, before, "all raw outputs preserved during audit")
    return {"schema": SCHEMA, "status": "passed_saved_complete_evidence" if complete else "passed_saved_partial_evidence",
        "study_status": report["status"], "parent_status": outer["status"], "inner_acceptance_status": acceptance["status"],
        "authoritative_parent_exit": args.terminal_exit,
        "source_commit": release["source_commit"], "source_archive_sha256": args.source_archive_sha256,
        "declaration_sha256": digest(args.manifest), "release_sha256": digest(args.release),
        "checker_sha256": digest(__file__), "numerical_absolute_tolerance": ATOL,
        "physical_coordinate_absolute_tolerance_mm": FRAME_ATOL_MM,
        "screening": screen, "comparisons": comparisons, "total_native_previews": total,
        "resources": {"worker_seconds": report["elapsed_seconds"], "worker_peak_rss_bytes": report["peak_rss_bytes"],
            "supervisor": supervisor, "parent_seconds": acceptance["elapsed_seconds"], "outer": outer,
            "timing_scope": "nested worker/phase/supervisor times are not additive"},
        "bound_input_count": len(bound["inputs_before"]), "raw_output_sha256": before,
        "raw_outputs_unchanged": True, "limits": LIMITS,
        "scope": "Saved records and opaque hashes only; zero patient-array decode, native calls, training or refits"}


def self_test():
    exact(interval(np.array([-2., 0., 0.]), np.array([2., 0., 0.])), [.25, .75], "analytic interval")
    exact(interval(np.array([0., 2., 0.]), np.array([1., 2., 0.])), None, "analytic missed interval")
    pts = np.array([[0., 0., 0.], [3., 3., 3.], [3.01, 1., 1.]])
    _, inside = domain(pts, np.eye(4), np.array([4, 4, 4]))
    exact(inside.tolist(), [True, True, False], "analytic center bounds")
    _, inside = domain(pts, np.eye(4), np.array([4, 4, 4]), cells=True)
    exact(inside.tolist(), [True, True, True], "distinct full-cell bounds")
    try:
        exact(1, 2, "intentional mismatch")
    except AssertionError:
        pass
    else:
        raise AssertionError("checker must reject a mismatch")
    # Handwritten metadata only: no engine, proposer, geometry or patient call.
    common = {"tools": [{"tool_id": "one"}, {"tool_id": "two"}],
              "adapter_options": {"proposal_config": {"offsets_source_voxels": [[i, 0] for i in range(13)]}}}
    rows, derived = [], []
    for index, (axis, sign) in enumerate(EXITS):
        access = {"center_mm": [0., 0., 0.], "normal_inward": (-sign * np.eye(3)[axis]).tolist(), "radius_mm": 6.}
        base = {"axis": axis, "outward_sign": sign, "distance_mm": 1. if index == 5 else 3., "access": access}
        derived.append(dict(base))
        model, engine = "sha256:" + "a" * 64, "sha256:" + "b" * 64
        state = "sha256:" + hashlib.sha256((engine + ":initial").encode()).hexdigest()
        entry, tip = [0., 0., 0.], (-sign * np.eye(3)[axis]).tolist()
        identifier = "nominal-cavity-" + semantic({"model": model, "cavity": state, "geometry": ["one", entry, tip]}).split(":", 1)[1][:24]
        ray = {"proposal_id": identifier, "tool_id": "one", "entry_mm": entry, "tip_mm": tip}
        ledger = [{"column_index": column, "offset_source_voxels": [column, 0], "tool_id": tool,
                   "family": family, "reason": "NO_EXPOSED_REMAINING_TISSUE" if family == FAMILIES[0] else "NO_REMAINING_NOMINAL_TARGET", "proposal_id": None}
                  for column in range(13) for tool in ("one", "two") for family in FAMILIES]
        ledger[0].update(reason="PROPOSED_UNCERTIFIED", proposal_id=identifier)
        pose = {**ray, "axis_unit": tip, "duplicate_of": None, "screen_index": index,
                "admissible": True, "geometry": {"failures": []}, "blocked_cell_count": 0,
                "first_blocked_cell": None, "first_blocked_cell_mm": None, "reasons": []}
        rows.append({**base, "status": "screened", "eligible": True, "engine_model_hash": engine,
            "provider_model_hash": model, "cavity_state_hash": state, "poses": [pose],
            "proposal_dispositions": {"model_hash": model, "cavity_state_hash": state, "ledger": ledger,
                "proposals": [ray], "counts": dict(Counter(r["reason"] for r in ledger)), "slot_count": 78, "emitted_count": 1}})
    fixture = {"version": "six-axis-any-entry-native-ingress-v1", "complete_stroke_certified": False,
        "removal_authorized": False, "status": "selected", "complete": True,
        "selected_exit": [2, 1], "screen_count": 6, "exits": rows}
    exact(screening_check(fixture, derived, common, np.eye(4))["selected_exit"], [2, 1], "analytical selection metadata")
    corruptions = (lambda x: x.update(selected_exit=[0, -1]),
                   lambda x: x["exits"][0].update(eligible=False),
                   lambda x: x.update(screen_count=5),
                   lambda x: x["exits"][0]["proposal_dispositions"]["ledger"].pop())
    for corrupt in corruptions:
        changed = parse(canonical(fixture)); corrupt(changed)
        try:
            screening_check(changed, derived, common, np.eye(4))
        except AssertionError:
            pass
        else:
            raise AssertionError("constructed screening corruption was accepted")
    print(json.dumps({"status": "analytical_checker_self_test_only", "checks": 10, "actual_run_accessed": False}))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--self-test", action="store_true")
    parser.add_argument("--root-confirmed-terminal", action="store_true")
    parser.add_argument("--terminal-exit", type=int)
    for name in ("repository", "run", "manifest", "release", "source-archive"):
        parser.add_argument("--" + name, type=Path)
    parser.add_argument("--source-archive-sha256")
    args = parser.parse_args()
    if args.self_test:
        self_test(); return
    require(args.root_confirmed_terminal and args.terminal_exit is not None
            and all((args.repository, args.run, args.manifest, args.release, args.source_archive, args.source_archive_sha256)),
            "Explicit root terminal authority and exact run/source/release paths are required")
    started = time.monotonic()
    try:
        result = check_run(args)
    except Exception as error:
        result = {"schema": SCHEMA, "status": "failed_independent_check",
            "error": f"{type(error).__name__}: {error}", "checker_sha256": digest(__file__),
            "numerical_absolute_tolerance": ATOL, "physical_coordinate_absolute_tolerance_mm": FRAME_ATOL_MM,
            "limits": LIMITS}
    result["audit_wall_seconds"] = time.monotonic() - started
    print(json.dumps(result, sort_keys=True, indent=2, allow_nan=False))
    if result["status"] == "failed_independent_check":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
