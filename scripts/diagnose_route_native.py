#!/usr/bin/env python3
"""Bounded route/native action screen; no policy training or cavity mutation.

Use a captured source directory to preserve the behavior of a UI/bridge version
while other development continues. Primary checks retain every route's exact
tool, aperture, entry and target. Alternate tool profiles are separate scenarios.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from dataclasses import asdict
import gc
import json
from pathlib import Path
import sys
import time


def screen_native_axis(case, output: Path, started: float, max_seconds: float) -> None:
    """Separate named native-grid scenario, keeping the successful factory's ray."""
    from dataclasses import replace
    from resectionlab.geometry import GENERIC_TOOLS, GeometryScene, ToolPose, check_motion
    from resectionlab.native_resection import NATIVE_GENERIC_TOOLS, NativeResectionEngine
    from resectionlab.native_simulation import make_native_patient_simulator
    from resectionlab.evaluation import independent_check_native_history
    from resectionlab.worlds import content_hash
    import numpy as np

    template = make_native_patient_simulator(case, candidate_count=1, max_steps=1, max_actions=3)
    source_config = template.native_config
    tip = tuple(float(v) for v in template.candidate_tips_mm[0])
    entry = tuple(float(v) for v in template.candidate_entries_mm[0])
    del template
    gc.collect()
    rows = []
    output.parent.mkdir(parents=True, exist_ok=True)
    for tool in (*GENERIC_TOOLS, *NATIVE_GENERIC_TOOLS):
        if time.perf_counter() - started >= max_seconds:
            break
        config = replace(source_config, tools=(tool,))
        engine = NativeResectionEngine(config)
        axis = (np.asarray(tip) - entry) / np.linalg.norm(np.asarray(tip) - entry)
        static = check_motion(tool, ToolPose(entry, axis), ToolPose(tip, axis),
                               GeometryScene(config.hard_exclusion, config.affine), config.access)
        preview = engine.preview_stroke(tool.tool_id, tip, entry_mm=entry)
        row = {"tool": asdict(tool), "entry_mm": entry, "target_mm": tip,
               "access": {"center_mm": config.access.center_mm.tolist(), "normal_inward": config.access.normal_inward.tolist(),
                          "radius_mm": config.access.radius_mm, "window_id": config.access.window_id},
               "decision_model_hash": config.fingerprint, "static_route_feasible": static.feasible,
               "native_stroke_feasible": preview.feasible, "native_reason": preview.reason,
               "contained_tissue_mm3": preview.removed_volume_mm3, "failure_tip_mm": preview.failure_tip_mm}
        if preview.feasible:
            engine.commit_preview(preview)
            history = engine.history
            frozen = {"source_hash": case.semantic_hash, "entry_mm": entry, "target_mm": tip,
                      "tool": asdict(tool), "decision_model_hash": config.fingerprint, "history": history}
            frozen["candidate_hash"] = content_hash(frozen)
            history_file = output.parent / f"aligned-{tool.tool_id}-candidate.json"
            history_file.write_text(json.dumps(frozen, indent=2) + "\n")
            checked = time.perf_counter()
            audit = independent_check_native_history(case, config.tools, history,
                tissue_mask=config.tissue_mask, access=config.access, hard_exclusion=config.hard_exclusion,
                geometry_frame="RAS+")
            row.update(independent_native_audit=audit.to_dict(), independent_seconds=time.perf_counter() - checked,
                       candidate_hash=frozen["candidate_hash"], candidate_file=str(history_file), metrics=engine.metrics())
        rows.append(row)
        print(json.dumps({"axis_tool": tool.tool_id, "native": preview.feasible,
                          "reason": preview.reason, "volume": preview.removed_volume_mm3,
                          "independent": row.get("independent_native_audit", {}).get("feasible")}), flush=True)
        del engine
        gc.collect()
    report = {"source_hash": case.semantic_hash, "case_id": case.case_id,
              "scenario": "explicit_native_grid_aligned_hypothetical_access_and_axial_union_centroid",
              "entry_mm": entry, "target_mm": tip,
              "source_target_label": int(source_config.target_labels[tuple(np.rint(np.linalg.inv(source_config.affine)[:3, :3] @ tip + np.linalg.inv(source_config.affine)[:3, 3]).astype(int))]),
              "tissue_support_provenance": source_config.tissue_support_provenance,
              "gradient_steps": 0, "model_assumptions_changed_during_rollout": False,
              "elapsed_seconds": time.perf_counter() - started, "tool_results": rows,
              "clinical_deficit_probability": None,
              "interpretation": "Separate explicit access/tool research scenario; original search routes remain rejected for native cutting."}
    output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--source", type=Path, default=Path(__file__).resolve().parents[1] / "src")
    parser.add_argument("--max-seconds", type=float, default=90.)
    parser.add_argument("--include-explicit-native-profiles", action="store_true")
    parser.add_argument("--allow-estimated-support", action="store_true",
                        help="Reproduce the desktop's explicitly reviewed nonzero-MRI support assumption")
    parser.add_argument("--native-axis-alternatives", action="store_true")
    args = parser.parse_args()
    sys.path.insert(0, str(args.source.resolve()))
    from resectionlab.imaging import load_case
    from resectionlab.planning import SearchConfig, generate_candidate_routes
    from resectionlab.native_resection import NATIVE_GENERIC_TOOLS
    from resectionlab.native_simulation import make_native_patient_simulator
    from resectionlab.route_native_diagnostics import initial_native_action_diagnostic, route_proposal_coverage
    import numpy as np

    started = time.perf_counter()
    case = load_case(args.case)
    if args.native_axis_alternatives:
        screen_native_axis(case, args.output, started, args.max_seconds)
        return
    source_tools = None
    if args.include_explicit_native_profiles:
        from resectionlab.geometry import GENERIC_TOOLS
        source_tools = (*GENERIC_TOOLS, *NATIVE_GENERIC_TOOLS)
    support = provenance = None
    if case.brain_mask is None:
        if not args.allow_estimated_support:
            raise ValueError("Explicit --allow-estimated-support is required for this case without a brain mask")
        if case.metadata.get("allow_nonzero_mri_access_support") is False or case.metadata.get("structural_coverage") == "full_head":
            raise ValueError("Full-head or explicitly unsupported MRI cannot define cortical access")
        support = np.isfinite(case.mri) & (case.mri != 0)
        provenance = {"source": case.semantic_hash, "method": "finite_nonzero_structural_MRI_support_v1", "evidence_type": "estimated"}
    search = generate_candidate_routes(case, tools=source_tools, config=SearchConfig(),
                                       support_mask=support, support_provenance=provenance)
    routes = search.candidates
    groups = defaultdict(list)
    for route in routes:
        groups[(route.window_id, route.tool_id)].append(route)
    rows, summaries = [], []
    timed_out = False
    for key, members in groups.items():
        if time.perf_counter() - started >= args.max_seconds:
            timed_out = True
            break
        first = members[0]
        group_started = time.perf_counter()
        sim = make_native_patient_simulator(case, access=first.window, tools=(first.tool,),
                                            candidate_count=4, max_steps=3, max_actions=7)
        diagnosis = initial_native_action_diagnostic(sim)
        group_rows = []
        for route in members:
            if time.perf_counter() - started >= args.max_seconds:
                timed_out = True
                break
            before = sim.engine.state_hash
            preview = sim.engine.preview_stroke(route.tool_id, route.target_mm, entry_mm=route.entry_mm)
            assert sim.engine.state_hash == before and sim.engine.revision == 0
            row = {
                "route_id": route.route_id, "window_id": route.window_id, "tool_id": route.tool_id,
                "tool": asdict(route.tool), "entry_mm": route.entry_mm, "target_mm": route.target_mm,
                "static_route_feasible": route.feasible,
                "static_route_category": route.category,
                "bridge_factory_non_stop_actions": diagnosis["non_stop_action_count"],
                "exact_native_stroke_feasible": preview.feasible,
                "exact_native_stroke_reason": preview.reason,
                "failure_tip_mm": preview.failure_tip_mm,
                "completed_microsteps_before_failure": len(preview.microsteps),
                "preview_contained_tissue_mm3": preview.removed_volume_mm3,
                "proposal_coverage": route_proposal_coverage(route, sim),
                "executed_removal_mm3": 0.,
            }
            rows.append(row)
            group_rows.append(row)
        summaries.append({"window_id": key[0], "tool_id": key[1], "access": {
            "center_mm": list(first.window.center_mm), "normal_inward": list(first.window.normal_inward),
            "radius_mm": first.window.radius_mm}, "initial_proposals": diagnosis,
            "exact_routes_screened": len(group_rows),
            "exact_native_routes_feasible": sum(row["exact_native_stroke_feasible"] for row in group_rows),
            "exact_native_rejection_counts": dict(Counter(row["exact_native_stroke_reason"] for row in group_rows if not row["exact_native_stroke_feasible"])),
            "elapsed_seconds": time.perf_counter() - group_started})
        print(json.dumps({"window": key[0], "tool": key[1], "initial_non_stop": diagnosis["non_stop_action_count"],
            "exact_native": summaries[-1]["exact_native_routes_feasible"], "screened": len(group_rows),
            "rejections": summaries[-1]["exact_native_rejection_counts"]}), flush=True)
        del sim
        gc.collect()
    report = {
        "case_id": case.case_id, "source_hash": case.semantic_hash, "planning_hash": case.planning_hash,
        "source_directory": str(args.source.resolve()), "search_config": asdict(SearchConfig()),
        "access_support_provenance": provenance,
        "route_count": len(routes), "screened_route_count": len(rows),
        "static_feasible_routes": sum(row["static_route_feasible"] for row in rows),
        "exact_native_feasible_routes": sum(row["exact_native_stroke_feasible"] for row in rows),
        "routes_missing_exact_ray_in_bridge_proposals": sum(not row["proposal_coverage"]["selected_route_exact_ray_proposed"] for row in rows),
        "elapsed_seconds": time.perf_counter() - started, "timed_out": timed_out,
        "max_seconds": args.max_seconds, "environment_steps": 0, "gradient_steps": 0,
        "tool_scenarios": "original_and_explicitly_added_native_profiles" if source_tools else "original_search_tools_unchanged",
        "clinical_deficit_probability": None, "groups": summaries, "routes": rows,
        "scope": "development diagnostic; frozen initial geometry only; previews do not remove tissue",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    print(json.dumps({key: report[key] for key in ("route_count", "screened_route_count", "static_feasible_routes",
        "exact_native_feasible_routes", "routes_missing_exact_ray_in_bridge_proposals", "elapsed_seconds", "timed_out")}))


if __name__ == "__main__":
    main()
