#!/usr/bin/env python3
"""Freeze and independently check named native-axis alternatives without RL."""
from __future__ import annotations

import argparse
from dataclasses import asdict
from hashlib import sha256
import json
from pathlib import Path
import shutil
import sys
from time import perf_counter


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case", type=Path, required=True)
    parser.add_argument("--source", type=Path, default=Path("src"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--original-report", type=Path)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    snapshot = args.output / "source-snapshot"
    shutil.copytree(args.source / "resectionlab", snapshot / "resectionlab",
                    ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
    shutil.copy2(__file__, args.output / Path(__file__).name)
    hashes = {str(path.relative_to(snapshot)): sha256(path.read_bytes()).hexdigest()
              for path in sorted(snapshot.rglob("*.py"))}
    sys.path.insert(0, str(snapshot.resolve()))
    import numpy as np
    from resectionlab.imaging import load_case
    from resectionlab.planning import generate_candidate_routes
    from resectionlab.native_routes import generate_native_axis_routes
    from resectionlab.native_resection import NativeResectionEngine, native_config_from_case
    from resectionlab.evaluation import independent_check_native_history
    from resectionlab.worlds import content_hash

    started = perf_counter()
    case = load_case(args.case)
    alternatives = generate_native_axis_routes(case)
    report = {"source_hash": case.semantic_hash, "source_module_sha256": hashes,
              "gradient_steps": 0, "clinical_deficit_probability": None,
              "search": alternatives.to_dict(), "independent_results": []}
    for route in alternatives.candidates:
        config = native_config_from_case(case, access=route.window, tools=(route.tool,))
        engine = NativeResectionEngine(config)
        frame = np.diag([-1., -1., 1.]) if case.frame == "LPS+" else np.eye(3)
        tip = frame @ route.target_mm
        entry = frame @ route.entry_mm
        result = engine.execute_stroke(route.tool_id, tip, entry_mm=entry)
        row = {"route_id": route.route_id, "tool": asdict(route.tool),
               "entry_mm": route.entry_mm, "target_mm": route.target_mm,
               "feasible": result.feasible, "reason": result.reason,
               "native_model_hash": config.fingerprint, "metrics": engine.metrics()}
        if result.feasible:
            frozen = {"source_hash": case.semantic_hash, "route": route.to_dict(),
                      "native_model_hash": config.fingerprint, "history": engine.history}
            frozen["candidate_hash"] = content_hash(frozen)
            (args.output / f"{route.tool_id}-candidate.json").write_text(json.dumps(frozen, indent=2) + "\n")
            audit = independent_check_native_history(case, config.tools, engine.history,
                tissue_mask=config.tissue_mask, access=config.access, hard_exclusion=config.hard_exclusion,
                geometry_frame="RAS+")
            row["independent_audit"] = audit.to_dict()
        report["independent_results"].append(row)
        print(json.dumps({"tool": route.tool_id, "feasible": result.feasible,
                          "metrics": row["metrics"], "independent": row.get("independent_audit", {}).get("feasible")}), flush=True)
    if args.original_report:
        old = json.loads(args.original_report.read_text())
        support = None if case.brain_mask is not None else np.isfinite(case.mri) & (case.mri != 0)
        provenance = None if support is None else {"source": case.semantic_hash,
            "method": "finite_nonzero_structural_MRI_support_v1", "evidence_type": "estimated"}
        original = generate_candidate_routes(case, support_mask=support, support_provenance=provenance)
        before = {(row["route_id"], row["tool_id"], tuple(row["entry_mm"]), tuple(row["target_mm"]))
                  for row in old["routes"]}
        after = {(row.route_id, row.tool_id, row.entry_mm, row.target_mm) for row in original.candidates}
        report["original_search_regression"] = {"before_count": len(before), "after_count": len(after),
                                                 "same_ids_tools_entries_targets": before == after}
    report["elapsed_seconds"] = perf_counter() - started
    (args.output / "report.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    print(json.dumps({"elapsed_seconds": report["elapsed_seconds"],
                      "original_search_regression": report.get("original_search_regression")}))


if __name__ == "__main__":
    main()
