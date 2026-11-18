#!/usr/bin/env python3
"""Classify saved EDEN proposal ledgers and check independent tract witnesses."""

import argparse
from collections import Counter
from hashlib import sha256
import json
from pathlib import Path
from time import perf_counter

import numpy as np

from resectionlab.native_approach_comparison import distinct_alternatives, order_candidates
from resectionlab.patient_anatomy import PatientImagingCollection, world


NON_LAYER_REASONS = {
    "entry and target coincide": "invalid entry geometry",
    "inferior entries excluded by supratentorial scope": "superior-entry scope",
    "corridor exceeds declared maximum length": "length constraint",
    "whole corridor outside reference image field": "reference field coverage",
    "endpoint is outside or unassessed in the selected target label": "unresolved target placement",
}


def read_receipts(prepared, serial):
    for subject in ("CTRL_1277", "CTRL_0866", "CTRL_0933"):
        for target in ("01", "08"):
            if subject == "CTRL_1277":
                version = {"01": "v3", "08": "v2"}[target]
                directory = prepared / "verification" / f"{subject}-{target}-{version}"
            else:
                directory = serial / f"{subject}-{target}-serial"
            yield subject, directory / "report.json"
    yield "CTRL_1277", serial / "CTRL_1277-01-radius1-serial" / "report.json"


def comparison_summary(saved, source):
    report = saved["proposalReport"]
    if source.identity() != report["sourceSnapshotId"] or source.identity() != saved["snapshotId"]:
        raise ValueError("Archived routes and prepared anatomy bind different revisions")
    task, candidates = report["task"], report["candidates"]
    if len(candidates) != report["enumerated"] or len(report["scenarioSet"]["scenarios"]) == 0:
        raise ValueError("Incomplete candidate or scenario ledger")
    if any((row["target"], row["radiusMm"]) != (task["target"], task["radiusMm"]) for row in candidates):
        raise ValueError("Fixed-endpoint witnesses require every candidate to share the task target and radius")
    if saved["inputComparison"]["task"] != task:
        raise ValueError("Archived comparison arms used a different task")
    labels = {row.get("label", row["id"]): row["id"]
              for row in (*source.document["layers"], *source.document["tractography"])}
    finalists = [row["id"] for row in report["candidateFreeze"]["candidates"]]
    if set(finalists) != report["finalistOutcomes"].keys():
        raise ValueError("Archived shortlist does not match scenario assessments")
    audited, unknown_counts, contact_counts = [], Counter(), Counter()
    causes = Counter()
    for row in candidates:
        values = row.get("sourceReadings", {})
        unknown_counts.update(slot for slot, reading in values.items() if reading[0] != "full-field")
        contact_counts.update((slot, reading[2]) for slot, reading in values.items()
                              if reading[2] in {"detected", "unresolved"})
        nominal_reasons = []
        for reason in row["reasons"]:
            if reason.startswith("Declared hard guard failed"):
                continue
            layer = labels.get(reason.split(": ", 1)[0])
            if layer is None:
                kind = NON_LAYER_REASONS[reason]
            else:
                contact = values[layer][2]
                if "unassessed over" in reason:
                    kind = "unknown explicit exclusion"
                elif contact == "detected":
                    kind = "detected contact"
                elif contact == "unresolved":
                    kind = "unresolved voxel contact"
                elif "clearance below" in reason or "declared clearance" in reason:
                    kind = "clearance bound only"
                else:
                    raise ValueError(f"Unclassified reason {reason}")
            nominal_reasons.append({"kind": kind, "layer": layer, "reason": reason})
        causes.update({cause["kind"] for cause in nominal_reasons})
        outcome = report["finalistOutcomes"].get(row["id"])
        vetoes = [{k: scenario[k] for k in ("id", "conflictLayerIds", "constraintFailures")}
                  for scenario in outcome["scenarios"] if scenario["constraintFailures"]] if outcome else []
        audited.append({"sample": row["sampleIndex"], "id": row["id"], "entry": row["entry"],
                        "disposition": row["disposition"], "nominalReasons": nominal_reasons,
                        "unknownOrPartialSources": sorted(k for k, v in values.items() if v[0] != "full-field"),
                        "vetoes": vetoes})
    if sum(bool(row["nominalReasons"]) for row in audited) != len(candidates) - report["nominalEligible"]:
        raise ValueError("Nominal causes do not reconcile with the candidate count")
    if sum(bool(row["vetoes"]) for row in audited) != report["scenarioVetoCount"]:
        raise ValueError("Scenario vetoes do not reconcile with the candidate count")
    entries, patch = np.array([row["entry"] for row in candidates]), task["entryPatch"]
    displacements = entries - patch["center"]
    radii = np.linalg.norm(displacements, axis=1)
    return {
        "task": task, "sourceSnapshotId": source.identity(), "enumerated": len(candidates),
        "assessmentVersion": report["assessmentVersion"],
        "measured": report["evaluated"], "nominalEligible": report["nominalEligible"],
        "finalists": len(finalists), "scenarioVetoed": report["scenarioVetoCount"],
        "outsideScenarioBudget": report["nominalEligible"] - len(finalists), "returned": report["returned"],
        "scenarioSetVersion": report["scenarioSet"]["version"],
        "scenarioCount": len(report["scenarioSet"]["scenarios"]), "finalistIds": finalists,
        "selectedIds": report["selectedCandidateIds"],
        "finalistGuardDispositions": dict(Counter(row["guard"]["disposition"] for row in report["finalistOutcomes"].values())),
        "reasonCandidateCounts": {kind: causes[kind] for kind in (*NON_LAYER_REASONS.values(),
                                  "unknown explicit exclusion", "detected contact",
                                  "unresolved voxel contact", "clearance bound only")},
        "partialOrUnavailableLayerCounts": dict(unknown_counts),
        "contactCounts": [{"layer": layer, "status": status, "count": count}
                          for (layer, status), count in sorted(contact_counts.items())],
        "sourceManifestDeclaredMissing": source.document.get("missing", []),
        "entryPatch": {"minimumSuperiorOffsetMm": float(entries[:, 2].min() - task["target"][2]),
                       "samplesPerRadiusMm": dict(sorted(Counter(np.round(radii, 9)).items())),
                       "farthestFromPatchCenterMm": float(radii.max()),
                       "maximumOffPlaneDistanceMm": float(np.max(np.abs(displacements @ patch["normal"])))},
        "arms": {name: {"nominalEligible": sum(row["nominalEligible"] for row in arm["candidates"]),
                        "nominated": len(arm["nominated"]), "refused": arm["refusalCount"],
                        "outcomes": dict(Counter(row["status"] for row in arm["nominated"]))}
                 for name, arm in saved["inputComparison"]["arms"].items()},
        "candidates": audited,
    }


def point_distances(point, first, last):
    delta = last - first
    square = np.sum(delta * delta, axis=-1)
    along = np.sum((point - first) * delta, axis=-1)
    fraction = np.divide(along, square, out=np.zeros_like(along), where=square > 0).clip(0, 1)
    return np.linalg.norm(point - first - fraction[:, None] * delta, axis=1)


def segment_distances(start, end, first, last):
    """Two finite segment edges and the interior stationary point in RAS."""
    route, lines = end - start, last - first
    offset = first - start
    normal = np.cross(route, lines)
    norm2 = np.einsum("ij,ij->i", normal, normal)
    u = np.einsum("ij,ij->i", np.cross(offset, lines), normal)
    v = np.einsum("ij,ij->i", np.cross(offset, route), normal)
    interior = (norm2 > 0) & (u >= 0) & (u <= norm2) & (v >= 0) & (v <= norm2)
    projection = np.einsum("ij,ij->i", offset, normal)
    distance = np.full(len(first), np.inf)
    distance[interior] = abs(projection[interior]) / np.sqrt(norm2[interior])
    return np.minimum.reduce((distance, point_distances(start, first, last),
                              point_distances(end, first, last),
                              point_distances(first, start, end),
                              point_distances(last, start, end)))


def tract_segments(source, tract, lower, upper):
    """Prune disjoint boxes and exclude joins between supplied streamlines."""
    points = source.array(tract["points_ras_mm"])
    offsets = source.array(tract["offsets"])
    matrix = source.registration_matrix(tract["source_series"])
    cuts = offsets[1:-1] - 1
    for begin in range(0, len(points) - 1, 131072):
        end = min(begin + 131072, len(points) - 1)
        chunk = world(matrix, points[begin:end + 1])
        first, last = chunk[:-1], chunk[1:]
        keep = ((np.maximum(first, last) >= lower) & (np.minimum(first, last) <= upper)).all(axis=1)
        keep[cuts[(cuts >= begin) & (cuts < end)] - begin] = False
        if keep.any():
            yield first[keep], last[keep]


def contact_witnesses(source, tract, candidates, shift, radius):
    starts = np.array([row["entry"] for row in candidates]) - shift
    targets = np.array([row["target"] for row in candidates]) - shift
    points = np.concatenate([starts, targets])
    padding = radius + 1e-7
    lo, hi = points.min(axis=0) - padding, points.max(axis=0) + padding
    distances = np.full(len(starts), np.inf)
    target_distance = np.inf
    for first, last in tract_segments(source, tract, lo, hi):
        target_distance = min(target_distance, point_distances(targets[0], first, last).min())
        seg_lo, seg_hi = np.minimum(first, last), np.maximum(first, last)
        for index, (start, end) in enumerate(zip(starts, targets)):
            keep = ((seg_hi >= np.minimum(start, end) - padding)
                    & (seg_lo <= np.maximum(start, end) + padding)).all(axis=1)
            if keep.any():
                distance = segment_distances(start, end, first[keep], last[keep]).min()
                distances[index] = min(distances[index], distance)
    # Boxes were padded by the radius. Only distances at or below the radius
    # are proven exact here. A null entry certifies no detected contact.
    return ({row["sampleIndex"]: float(value) if value <= radius else None
             for row, value in zip(candidates, distances)},
            float(target_distance) if target_distance <= radius else None)


def tract_audit(report, source):
    """Recheck archived decisive labels and the CTRL_1277 shortlist omission."""
    radius = report["task"]["radiusMm"]
    tracts = {row["id"]: row for row in source.document["tractography"]}
    candidates = report["candidates"]
    eligible = [row for row in candidates if row["nominalEligible"]]
    witnesses = {}
    nominal_layers = {slot for row in candidates if not row["nominalEligible"]
                      for slot in row.get("detectedConflictLayerIds", []) if slot in tracts}
    for slot in sorted(nominal_layers):
        distances, target_distance = contact_witnesses(source, tracts[slot], candidates, np.zeros(3), radius)
        states = {row["sampleIndex"]: row["sourceReadings"][slot][2] == "detected" for row in candidates}
        if any(states[sample] != (distance is not None) for sample, distance in distances.items()):
            raise ValueError(f"Archived and independent {slot} nominal contact disagree")
        witnesses[slot] = {"contactDistancesMm": distances, "targetCenterDistanceMm": target_distance}
    audit = {"nominal": witnesses}
    if eligible and radius == 2:
        if not report["finalistOutcomes"]:
            raise ValueError("Expected archived shortlist to contain scored candidates")
        if not all(outcome["violatedScenarioIds"] == ["registration-09", "combined-09"]
                   for outcome in report["finalistOutcomes"].values()):
            raise ValueError("The archived common veto pair changed")
        scenario = next(s for s in report["scenarioSet"]["scenarios"] if s["id"] == "registration-09")
        slot = "tract-arcuates"
        shift = np.asarray(scenario["sourceTranslationsMm"][tracts[slot]["source_series"]])
        distances, target_distance = contact_witnesses(source, tracts[slot], eligible, shift, radius)
        if any(distances[row["sampleIndex"]] is None for row in eligible if row["id"] in report["finalistOutcomes"]):
            raise ValueError("Independent audit did not reproduce every archived finalist veto")
        ordered = order_candidates(eligible)
        audit["commonPerturbation"] = {
            "scenario": scenario, "layer": slot, "contactDistancesMm": distances,
            "targetCenterDistanceMm": target_distance,
            "unvetoedByThisTest": [row["sampleIndex"] for row in ordered if distances[row["sampleIndex"]] is None],
            "largestDistinctSetNominal": len(distinct_alternatives(ordered)),
        }
    return audit


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prepared", required=True, type=Path)
    parser.add_argument("--serial", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--tract-witnesses", action="store_true")
    args = parser.parse_args()
    if args.output.exists():
        parser.error("choose a new output path to preserve prior evidence")
    results = []
    for subject, path in read_receipts(args.prepared, args.serial):
        started = perf_counter()
        raw = path.read_bytes()
        receipt = json.loads(raw)
        source = PatientImagingCollection(args.prepared / subject)
        summary = comparison_summary(receipt["persisted"]["comparison"], source)
        summary.update(historicalSuggestMs=receipt["timingsMs"].get("suggest", receipt.get("resumedSuggestion", {}).get("ms")),
                       historicalCompareMs=receipt["timingsMs"]["compareInputs"])
        if args.tract_witnesses:
            proposal = receipt["persisted"]["comparison"]["proposalReport"]
            summary["tractWitnesses"] = tract_audit(proposal, source)
        summary.update(receipt=str(path), receiptSha256=sha256(raw).hexdigest(), auditSeconds=perf_counter() - started)
        results.append(summary)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x") as stream:
        json.dump({"schema": "eden-approach-shortlist-audit-v1", "cases": results}, stream, indent=2, allow_nan=False)
        stream.write("\n")


if __name__ == "__main__":
    main()
