#!/usr/bin/env python3
"""Replay the seven archived EDEN tasks against the current read-only backend."""

import argparse
from hashlib import sha256
import json
import os
from pathlib import Path
import resource
import subprocess
import sys
from tempfile import TemporaryDirectory
from time import perf_counter

from audit_eden_approach_shortlist import comparison_summary, read_receipts
from resectionlab import anatomy_approaches as planning, geometry, native_approach_comparison
from resectionlab.patient_anatomy import PatientAnatomySnapshot, PatientImagingCollection


def write_new(path, value):
    with TemporaryDirectory(dir=path.parent) as pending:
        receipt = Path(pending) / path.name
        receipt.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
        os.link(receipt, path)


def candidate_geometry(row):
    return tuple(row[key] for key in ("id", "sampleIndex", "entry", "target", "radiusMm", "direction"))


def source_code_hashes():
    paths = [Path(module.__file__) for module in (planning, geometry, native_approach_comparison)]
    return {path.name: sha256(path.read_bytes()).hexdigest() for path in paths}


def check_replay(before, after, before_paired, after_paired):
    for comparison in (after, after_paired, before_paired):
        if comparison["task"] != before["task"]:
            raise ValueError("The replay changed the original task or a paired arm")
        if comparison["scenarioSet"]["scenarios"] != before["scenarioSet"]["scenarios"]:
            raise ValueError("The declared EDEN perturbations changed")
    candidates = [candidate_geometry(row) for row in before["candidates"]]
    for arm in (after, *after_paired["arms"].values()):
        if [candidate_geometry(row) for row in arm["candidates"]] != candidates:
            raise ValueError("Candidate identities, coverage of the patch, or order changed")


def run_case(subject, archive_path, prepared, output):
    started = perf_counter()
    archived_data = archive_path.read_bytes()
    archived_receipt = json.loads(archived_data)
    archived = archived_receipt["persisted"]["comparison"]
    original = archived["proposalReport"]
    source = PatientImagingCollection(prepared / subject)
    before = comparison_summary(archived, source)
    before["timingsMs"] = {
        "suggest": archived_receipt["timingsMs"].get("suggest", archived_receipt.get("resumedSuggestion", {}).get("ms")),
        "compareInputs": archived_receipt["timingsMs"]["compareInputs"],
    }

    route_geometry = planning.NativeRouteGeometry(PatientAnatomySnapshot.capture(source))
    assessment_started = perf_counter()
    proposals, proposal_report = planning.suggest_approaches(source, original["task"], route_geometry)
    paired_started = perf_counter()
    paired = planning.compare_inputs(source, original["task"], route_geometry)
    assessed = perf_counter()
    check_replay(original, proposal_report, archived["inputComparison"], paired)
    current = {
        "snapshotId": source.identity(),
        "proposalReport": planning._bound_report(source, proposal_report),
        "inputComparison": planning._bound_report(source, paired),
        "approaches": proposals,
    }
    after = comparison_summary(current, source)
    after["timingsMs"] = {
        "suggest": (paired_started - assessment_started) * 1000,
        "compareInputs": (assessed - paired_started) * 1000,
    }
    rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    receipt = {
        "case": output.stem, "historicalReceipt": str(archive_path),
        "historicalReceiptSha256": sha256(archived_data).hexdigest(),
        "timingScope": {"before": "archived Electron suggestion/comparison elapsed",
                        "after": "current backend call elapsed, starting cold and then reusing the geometry cache"},
        "processPeakRssBytes": rss * (1 if sys.platform == "darwin" else 1024),
        "secondsToReport": perf_counter() - started,
        "backendIdentity": geometry.native_backend.strict_live_identity(),
        "sourceCodeSha256": source_code_hashes(),
        "before": before, "after": after,
        "candidateGeometriesAndScenarioInputsMatch": True,
        "rawCurrentAssessment": current,
    }
    write_new(output, receipt)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prepared", required=True, type=Path)
    parser.add_argument("--serial", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--resume", action="store_true", help="Reuse only receipts bound to the same sources and backend")
    parser.add_argument("--case-index", type=int, choices=range(7), help=argparse.SUPPRESS)
    args = parser.parse_args()
    cases = list(read_receipts(args.prepared, args.serial))
    if args.case_index is not None:
        subject, archive_path = cases[args.case_index]
        run_case(subject, archive_path, args.prepared, args.output)
        return
    args.output.mkdir(parents=True, exist_ok=args.resume)
    if (args.output / "summary.json").exists():
        parser.error("The reassessment already has a completed summary")
    started = perf_counter()
    env = {**os.environ, **dict.fromkeys(("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"), "1")}
    fields = ("assessmentVersion", "scenarioSetVersion", "scenarioCount", "enumerated", "measured", "nominalEligible",
              "finalists", "scenarioVetoed", "outsideScenarioBudget", "returned", "finalistGuardDispositions",
              "reasonCandidateCounts", "partialOrUnavailableLayerCounts", "arms", "timingsMs")
    results = []
    for index, (subject, archive_path) in enumerate(cases):
        target = "08" if "-08-" in archive_path.parent.name else "01"
        radius = "1" if "radius1" in archive_path.parent.name else "2"
        output = args.output / f"{subject}-{target}-radius{radius}.json"
        if args.resume and output.is_file():
            receipt = json.loads(output.read_bytes())
            if (receipt["case"] != output.stem or not receipt["candidateGeometriesAndScenarioInputsMatch"]
                    or receipt["historicalReceiptSha256"] != sha256(archive_path.read_bytes()).hexdigest()
                    or receipt["sourceCodeSha256"] != source_code_hashes()
                    or receipt["backendIdentity"] != geometry.native_backend.strict_live_identity()
                    or receipt["after"]["sourceSnapshotId"] != PatientImagingCollection(args.prepared / subject).identity()):
                raise ValueError(f"Cannot resume with an incompatible receipt: {output}")
        else:
            subprocess.run(
                [sys.executable, __file__, "--prepared", str(args.prepared), "--serial", str(args.serial),
                 "--output", str(output), "--case-index", str(index)], env=env, check=True,
            )
        receipt = json.loads(output.read_bytes())
        results.append({
            "case": receipt["case"], "report": output.name, "reportSha256": sha256(output.read_bytes()).hexdigest(),
            "sourceSnapshotId": receipt["after"]["sourceSnapshotId"],
            "processPeakRssBytes": receipt["processPeakRssBytes"],
            **{key: {field: receipt[key][field] for field in fields} for key in ("before", "after")},
        })
        print(f"Completed {output.stem} in {receipt['secondsToReport']:.2f} s", flush=True)
    write_new(args.output / "summary.json", {
        "schema": "eden-approach-reassessment-v1", "seconds": perf_counter() - started, "cases": results,
        "timingScope": receipt["timingScope"],
    })


if __name__ == "__main__":
    main()
