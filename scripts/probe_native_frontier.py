#!/usr/bin/env python3
"""Isolated development-only native proposal experiment; no policy learning.

The default runtime is the completed procedural-to-UCSF study's immutable source
snapshot. This launcher does not import the working production implementation.
Both proposal inventories use the unchanged native cutting engine, same source,
same access disk, same named tools and same nominal reward. The expanded rule is
a new action-model identity. It is not a correction to the completed experiment.
"""
from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import asdict
from hashlib import sha256
import json
from pathlib import Path
import shutil
import sys
from time import perf_counter

import numpy as np


RULE = {
    "version": "residual-source-axis-columns-development-v1",
    "offsets_source_voxels": [[0, 0], [-2, 0], [2, 0], [0, -2], [0, 2],
        [-2, -2], [-2, 2], [2, -2], [2, 2], [-3, 0], [3, 0], [0, -3], [0, 3]],
    "primary_endpoint": "distal_remaining_annotated_cell_in_same_source_axis_column",
    "fallback_endpoint": "proximal_remaining_annotated_cell_if_distal_preview_rejected",
    "entry": "integer_transverse_source_column_projected_to_unchanged_access_plane",
    "entry_filter": "parallel_full_tool_radius_inside_declared_disk",
    "unsupported_access": "abstain_when_not_orthogonal_source_axis_or_not_integer_transverse_origin",
    "sort": "declared_offsets_then_unchanged_tool_order; ties_by_stable_ray_id",
    "state_inputs": "visible_remaining_native_tissue_and_supplied_radiological_labels_only",
    "certifier": "unchanged_complete_native_preview_including_prior_cavity_swept_shaft",
    "macromotion": "fixed_axis_insertion_and_full_retraction_before_next_action",
}
CAPS = {"max_actions": 3, "max_preview_checks": 128, "max_search_seconds": 45.}


def digest(value):
    return "sha256:" + sha256(json.dumps(value, sort_keys=True, allow_nan=False).encode()).hexdigest()


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def axis_rays(engine):
    """Finite residual target proposals; this helper grants no feasibility."""
    config = engine.config
    basis = np.asarray(config.affine)[:3, :3]
    spacing = np.linalg.norm(basis, axis=0)
    unit = basis / spacing[None, :]
    if not np.allclose(unit.T @ unit, np.eye(3), atol=1e-8):
        raise ValueError("EXPERIMENT_UNSUPPORTED_SHEARED_GRID")
    orientation = unit.T @ config.access.normal_inward
    axis = int(np.argmax(np.abs(orientation)))
    if not np.isclose(abs(orientation[axis]), 1., atol=1e-8):
        raise ValueError("EXPERIMENT_UNSUPPORTED_NONAXIAL_ACCESS")
    sign = 1 if orientation[axis] > 0 else -1
    transverse = [value for value in range(3) if value != axis]
    inverse = np.linalg.inv(config.affine)
    origin = inverse[:3, :3] @ config.access.center_mm + inverse[:3, 3]
    if not np.allclose(origin[transverse], np.rint(origin[transverse]), atol=1e-7):
        raise ValueError("EXPERIMENT_UNSUPPORTED_FRACTIONAL_TRANSVERSE_ORIGIN")
    rays, omitted = [], []
    for offset in RULE["offsets_source_voxels"]:
        entry_index = origin.copy()
        entry_index[transverse] = np.rint(origin[transverse]) + offset
        if np.any(entry_index[transverse] < 0) or np.any(entry_index[transverse] >= np.asarray(config.tissue_mask.shape)[transverse]):
            continue
        entry = basis @ entry_index + config.affine[:3, 3]
        radial = float(np.linalg.norm(entry - config.access.center_mm))
        selector = np.rint(entry_index).astype(int).tolist()
        selector[axis] = slice(None)
        labels = config.target_labels[tuple(selector)]
        remains = engine.remaining_mask[tuple(selector)]
        cells = np.flatnonzero((labels > 0) & remains)
        cells = cells[(cells - origin[axis]) * sign > 0]
        if not len(cells):
            omitted.append({"offset": offset, "reason": "NO_REMAINING_TARGET_IN_COLUMN"})
            continue
        cells = cells[np.argsort((cells - origin[axis]) * sign)]
        for tool in config.tools:
            if radial + max(tool.tip_radius_mm, tool.shaft_radius_mm) > config.access.radius_mm + 1e-9:
                omitted.append({"offset": offset, "tool_id": tool.tool_id, "reason": "FULL_TOOL_APERTURE_PREFILTER"})
                continue
            ends = []
            for index in dict.fromkeys((int(cells[-1]), int(cells[0]))):
                point = entry_index.copy()
                point[axis] = index
                ends.append((basis @ point + config.affine[:3, 3]).tolist())
            rays.append({"ray_id": f"axis:{offset[0]}:{offset[1]}:{tool.tool_id}",
                         "tool_id": tool.tool_id, "entry_mm": entry.tolist(), "endpoints_mm": ends})
    return rays, omitted


def fixed_rays(simulator):
    return [{"ray_id": f"fixed:{index}:{tool.tool_id}", "tool_id": tool.tool_id,
             "entry_mm": entry.tolist(), "endpoints_mm": [tip.tolist()]}
            for index, (entry, tip) in enumerate(zip(simulator.candidate_entries_mm, simulator.candidate_tips_mm))
            for tool in simulator.native_config.tools]


def measured_cost(engine, result, reward, partial_weight, current_tool):
    coords = tuple(result.removed_indices_native.T)
    target = float(np.count_nonzero(engine.config.target_labels[coords]) * engine.config.voxel_volume_mm3)
    normal = result.removed_volume_mm3 - target
    contacts = result.contact_indices_native
    remaining = engine.remaining_mask[tuple(contacts.T)]
    normal_contact = engine.config.target_labels[tuple(contacts.T)] == 0
    new_contact = ~engine.contact_mask[tuple(contacts.T)]
    removed = {tuple(cell) for cell in result.removed_indices_native}
    partial = sum(tuple(cell) not in removed for cell in contacts[remaining & normal_contact & new_contact]) * engine.config.voxel_volume_mm3
    motion = float(np.linalg.norm(np.asarray(result.tip_mm) - result.entry_mm)) * 2
    score = (reward.target_per_mm3 * target - reward.normal_per_mm3 * normal
             - partial_weight * reward.normal_per_mm3 * partial - reward.action_cost
             - reward.motion_per_mm * motion - reward.tool_change_cost * int(current_tool is not None and current_tool != result.tool_id))
    return {"target_mm3": target, "normal_mm3": normal, "partial_normal_contact_mm3": partial,
            "nominal_increment": float(score), "motion_mm": motion}


def greedy_inventory(engine, provider, reward, partial_weight, caps=CAPS):
    started = perf_counter()
    attempts, rounds, committed = [], [], []
    current_tool = None
    termination = "action_cap"
    for index in range(caps["max_actions"]):
        rays, omitted = provider(engine)
        best = None
        for ray in rays:
            for endpoint_index, tip in enumerate(ray["endpoints_mm"]):
                if len(attempts) >= caps["max_preview_checks"] or perf_counter() - started >= caps["max_search_seconds"]:
                    termination = "preview_cap" if len(attempts) >= caps["max_preview_checks"] else "wall_cap"
                    break
                checked = perf_counter()
                before = engine.state_hash
                result = engine.preview_stroke(ray["tool_id"], tip, entry_mm=ray["entry_mm"])
                assert engine.state_hash == before
                row = {"round": index, "ray_id": ray["ray_id"], "fallback": endpoint_index > 0,
                       "tip_mm": tip, "entry_mm": ray["entry_mm"], "tool_id": ray["tool_id"],
                       "feasible": result.feasible, "reason": result.reason,
                       "seconds": perf_counter() - checked, "failure_tip_mm": result.failure_tip_mm}
                if result.feasible:
                    row.update(measured_cost(engine, result, reward, partial_weight, current_tool))
                    key = (row["nominal_increment"], ray["ray_id"])
                    if best is None or key > best[0]:
                        best = (key, result, row)
                attempts.append(row)
                # Only rejected distal insertions nominate their proximal fallback.
                if result.feasible:
                    break
            if termination in {"preview_cap", "wall_cap"}:
                break
        rounds.append({"round": index, "omitted": omitted, "ray_count": len(rays)})
        if termination in {"preview_cap", "wall_cap"}:
            break
        if best is None or best[0][0] <= 0:
            termination = "no_positive_legal_proposal"
            break
        engine.commit_preview(best[1])
        committed.append(best[2])
        current_tool = best[1].tool_id
    return {"termination": termination, "search_seconds": perf_counter() - started,
            "preview_checks": len(attempts), "accepted_preview_count": sum(row["feasible"] for row in attempts),
            "rejection_counts": dict(Counter(row["reason"] for row in attempts if not row["feasible"])),
            "committed_actions": committed, "nominal_return": sum(row["nominal_increment"] for row in committed),
            "cumulative_partial_normal_contact_mm3": sum(row["partial_normal_contact_mm3"] for row in committed),
            "metrics": engine.metrics(), "rounds": rounds, "attempts": attempts}


def freeze_audit(case, engine, output, name):
    from resectionlab.evaluation import independent_check_native_history
    frozen = {"case_hash": case.semantic_hash, "engine_hash": engine.config.fingerprint, "history": engine.history}
    frozen["candidate_hash"] = digest(frozen)
    write_json(output / f"{name}-candidate.json", frozen)
    started = perf_counter()
    audit = independent_check_native_history(case, engine.config.tools, engine.history,
        tissue_mask=engine.config.tissue_mask, access=engine.config.access,
        hard_exclusion=engine.config.hard_exclusion, geometry_frame="RAS+")
    return {"candidate_hash": frozen["candidate_hash"], "seconds": perf_counter() - started, **audit.to_dict()}


def phantom(output, barrier):
    from resectionlab.core import CaseData, SourceRef
    from resectionlab.geometry import AccessWindow
    from resectionlab.native_resection import NativeResectionConfig, NativeResectionEngine, NATIVE_GENERIC_TOOLS
    from resectionlab.simulation import RewardSpec
    tissue = np.ones((17, 17, 20), dtype=bool)
    labels = np.zeros(tissue.shape, np.int16)
    labels[3:14, 3:14, 5:19] = 1
    hard = np.zeros(tissue.shape, bool)
    if barrier:
        hard[:, :, 12] = True
    case = CaseData("frontier-barrier" if barrier else "frontier-solid", tissue.astype(np.float32),
        {"target": labels > 0}, np.eye(4), (SourceRef("analytic", "synthetic:frontier", provenance="simulated"),), brain_mask=tissue)
    config = NativeResectionConfig(tissue, labels, np.eye(4), AccessWindow((8, 8, -.5), (0, 0, 1), 6.),
        NATIVE_GENERIC_TOOLS, case.semantic_hash, "explicit synthetic solid slab", hard_exclusion=hard)
    engine = NativeResectionEngine(config)
    row = greedy_inventory(engine, axis_rays, RewardSpec(), .05)
    row["independent_audit"] = freeze_audit(case, engine, output, case.case_id)
    assert row["independent_audit"]["feasible"]
    if barrier:
        assert not engine.removed_mask[:, :, 12:].any()
        assert row["rejection_counts"].get("HARD_GEOMETRY:FORBIDDEN_COLLISION", 0) > 0
    return row


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, default=Path("artifacts/learning/procedural-native-to-ucsf-v2"))
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    snapshot = json.loads((args.run / "worker-source.json").read_text())
    source = args.run / "frozen-source"
    for relative, expected in snapshot["numerical_runtime_sha256"].items():
        assert sha256((source / relative).read_bytes()).hexdigest() == expected, relative
    sys.path.insert(0, str((source / "src").resolve()))
    from resectionlab.imaging import load_case
    from resectionlab.native_simulation import make_native_patient_simulator
    from resectionlab.native_resection import NativeResectionEngine
    started = perf_counter()
    case = load_case(args.run / "source-case.ressectionlab")
    sim = make_native_patient_simulator(case, candidate_count=4, max_steps=3, max_actions=7)
    setup_seconds = perf_counter() - started
    declaration = {"rule": RULE, "caps_per_inventory": CAPS, "new_action_model": True,
        "posthoc_development_discovery": "early-probe.json generated hypotheses before this rule was frozen",
        "source_hash": case.semantic_hash, "original_adapter_hash": sim.decision_model_hash,
        "unchanged_engine_hash": sim.native_config.fingerprint, "tools": [asdict(tool) for tool in sim.native_config.tools],
        "access": {"center_mm": sim.native_config.access.center_mm.tolist(),
                   "normal_inward": sim.native_config.access.normal_inward.tolist(), "radius_mm": sim.native_config.access.radius_mm},
        "reward": asdict(sim.config.reward), "partial_contact_weight": sim.partial_contact_weight,
        "gradient_steps": 0, "world_partitions_opened": [], "final_worlds_used": False, "stress_worlds_used": False,
        "frozen_runtime_hash": snapshot["numerical_runtime_content_hash"],
        "script_sha256": sha256(Path(__file__).read_bytes()).hexdigest()}
    declaration["proposal_model_hash"] = digest(declaration)
    write_json(args.output / "declaration.json", declaration)
    shutil.copy2(__file__, args.output / "experiment-script.py")
    report = {"declaration_hash": digest(declaration), "setup_seconds": setup_seconds, "phantoms": {}}
    for barrier in (False, True):
        name = "barrier" if barrier else "solid"
        print("Checking constructed phantom " + name, flush=True)
        report["phantoms"][name] = phantom(args.output, barrier)
        write_json(args.output / "progress.json", report)
    for name in ("fixed", "expanded"):
        print("Checking development inventory " + name, flush=True)
        prepared = perf_counter()
        engine = NativeResectionEngine(sim.native_config)
        preparation = perf_counter() - prepared
        rows = fixed_rays(sim)
        provider = axis_rays if name == "expanded" else lambda _: (rows, [])
        result = greedy_inventory(engine, provider, sim.config.reward, sim.partial_contact_weight)
        result["engine_setup_seconds"] = preparation
        if name == "expanded":
            result["independent_audit"] = freeze_audit(case, engine, args.output, "patient-expanded")
        else:
            assert result["metrics"]["simulated_removed_target_mm3"] == 249.
            assert abs(result["nominal_return"] - 245.24) < 1e-8
        report[name] = result
        write_json(args.output / "progress.json", report)
        print(json.dumps({"inventory": name, "return": result["nominal_return"], "metrics": result["metrics"],
                          "previews": result["preview_checks"], "seconds": result["search_seconds"]}), flush=True)
    report["elapsed_seconds"] = perf_counter() - started
    report["clinical_deficit_probability"] = None
    write_json(args.output / "report.json", report)


if __name__ == "__main__":
    main()
