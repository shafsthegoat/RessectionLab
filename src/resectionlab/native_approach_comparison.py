"""Source-geometry comparison and declared engineering sensitivity scenarios."""
from itertools import combinations
import math

import numpy as np

from .evaluation import WorldOutcome, upper_tail_cvar
from .patient_anatomy import digest

RICH_INPUTS = ["reference acquisition field", "selected target label", "supplied brain label",
               "supplied critical and named exclusion labels", "supplied vascular labels and streamline polylines",
               "source-linked grids, transforms and acquisition fields for those inputs", "declared entry patch and constraints"]
RESTRICTED_INPUTS = ["reference acquisition field", "selected reference-bound target label",
                     "reference-bound brain, head-surface and explicitly listed brainstem/cerebellar estimates",
                     "declared entry patch and available exclusions"]
METRICS = {
    "lengthMm": ("mm", "higher"),
    "targetReachMm3": ("mm³", "lower"),
    "nonTargetTraversalMm3": ("mm³", "higher"),
    "vascularClearanceMm": ("mm", "lower"),
    "tractClearanceMm": ("mm", "lower"),
    "otherCriticalClearanceMm": ("mm", "lower"),
}
_VASCULAR = {"vessels", "artery", "vein", "sinus"}
_TRACT = {"tract", "tractography", "motor", "language", "functional"}
_TAIL_FRACTION = .2
_DISCLOSURE = "Finite declared sensitivity design. Registration accuracy and anatomical detection sensitivity are unmeasured."
SOURCE_READING_FIELDS = ["coverage", "clearanceLowerBoundMm", "contact", "intersectedCellOrSegmentCount"]


def sensitivity_config(raw=None):
    if raw is not None and (not isinstance(raw, dict) or set(raw) - {"translationMm", "boundaryMm"}):
        raise ValueError("Unknown native route sensitivity setting")
    raw = raw or {}
    result = {}
    for name, default, maximum in (("translationMm", 1., 5.), ("boundaryMm", .5, 3.)):
        value = raw.get(name, default)
        if type(value) not in (int, float) or not math.isfinite(value) or not 0 <= value <= maximum:
            raise ValueError(f"{name} must be finite in [0, {maximum}] mm")
        result[name] = float(value)
    return result


def sample_positions(patch):
    """The task alone fixes an identical candidate budget for both arms."""
    center, normal = np.asarray(patch["center"]), np.asarray(patch["normal"])
    first = np.cross(normal, np.eye(3)[int(np.argmin(np.abs(normal)))])
    first /= np.linalg.norm(first)
    second = np.cross(normal, first)
    radius = patch["radiusMm"]
    if radius == 0:
        return [center]
    # Additional sampling within small disks cannot support comparable separation.
    rings, bearings = (8, 16) if radius >= 4 else (2, 8)
    return [center, *(center + radius * scale * (np.cos(angle) * first + np.sin(angle) * second)
                      for scale in np.linspace(1 / rings, 1, rings)
                      for angle in np.linspace(0, 2 * np.pi, bearings, endpoint=False))]


def _covered(row):
    return (row.get("available", False) and row.get("coverage", {}).get("wholeEnvelopeWithinField", False)
            and row.get("acquisitionCoverage", {}).get("wholeEnvelopeWithinField", True))


def source_readings(measurements):
    """Compact per-layer geometry, preserving partial and unavailable coverage."""
    result = {}
    for row in measurements["structures"]:
        if not row["available"]:
            result[row["layerId"]] = ["unavailable", None, None, None]
            continue
        contact = ("detected" if row["confirmedIntersection"] else
                   "unresolved" if row.get("intersectsOrUnresolved") else "none-detected")
        result[row["layerId"]] = ["full-field" if _covered(row) else "partial-field", row["clearanceLowerBoundMm"], contact,
                                   row.get("intersectedCellCount", row.get("intersectedSegmentCount"))]
    return result


def _unknown_source_layers(measurements):
    sources = (*measurements["structures"], *measurements["targets"], measurements["normalTraversal"])
    return sorted({row["layerId"] for row in sources if "layerId" in row and not _covered(row)})


def nominal_objectives(measurements):
    """Return values and evidence signatures used by the Pareto comparison."""
    targets = measurements["targets"]
    brain = measurements["normalTraversal"]
    values = dict.fromkeys(METRICS)
    evidence = {}
    values["lengthMm"] = measurements["lengthMm"]
    if len(targets) == 1 and _covered(targets[0]):
        values["targetReachMm3"] = targets[0]["coveredVoxelCenterVolumeMm3"]
        evidence["targetReachMm3"] = [targets[0]["layerId"]]
    if _covered(brain) and all(map(_covered, targets)):
        values["nonTargetTraversalMm3"] = brain["conservativeVolumeUpperBoundMm3"]
        evidence["nonTargetTraversalMm3"] = [brain["layerId"], *sorted(row["layerId"] for row in targets)]
    groups = {name: [] for name in ("vascularClearanceMm", "tractClearanceMm", "otherCriticalClearanceMm")}
    for row in measurements["structures"]:
        name = ("vascularClearanceMm" if row["kind"] in _VASCULAR else
                "tractClearanceMm" if row["kind"] in _TRACT else "otherCriticalClearanceMm")
        groups[name].append(row)
    for name, rows in groups.items():
        if rows and all(map(_covered, rows)):
            values[name] = min(row["clearanceLowerBoundMm"] for row in rows)
            evidence[name] = sorted(row["layerId"] for row in rows)
    return values, evidence


def order_candidates(candidates):
    """Prefer equal-evidence Pareto candidates, retaining dominated geometry."""
    groups = {}
    for row in candidates:
        signature = digest(row["evidence"])
        groups.setdefault(signature, []).append(row)
    for rows in groups.values():
        names = tuple(key for key in METRICS if rows[0]["objectives"][key] is not None)
        costs = np.array([[row["objectives"][key] * (1 if METRICS[key][1] == "higher" else -1)
                           for key in names] for row in rows])
        delta = costs[:, None, :] - costs[None, :, :]
        dominates = np.all(delta <= 1e-8, axis=2) & np.any(delta < -1e-8, axis=2)
        for row, count in zip(rows, dominates.sum(axis=0)):
            row["pareto"] = not bool(count)
            row["dominatorCount"] = int(count)
    def preference(row):
        values = row["objectives"]
        present = sum(v is not None for v in values.values())
        return (not row["pareto"], -present, values["lengthMm"], row["sampleIndex"])
    ordered = sorted(candidates, key=preference)
    if not ordered:
        return ordered
    front = [row for row in ordered if row["pareto"]]
    anchors = [front[0]]
    for name in ("targetReachMm3", "nonTargetTraversalMm3", "vascularClearanceMm", "tractClearanceMm", "otherCriticalClearanceMm"):
        comparable = [row for row in front if row["evidence"] == front[0]["evidence"] and row["objectives"][name] is not None]
        if comparable:
            sign = 1 if METRICS[name][1] == "higher" else -1
            anchors.append(min(comparable, key=lambda row: sign * row["objectives"][name]))
    return list({r["id"]: r for r in (*anchors, *ordered)}.values())


def distinct_alternatives(ordered):
    """Find the greatest supported cardinality (up to three), then break ties."""
    if not ordered:
        return []
    positions = np.array([row["entry"] for row in ordered])
    axes = np.array([row["direction"] for row in ordered])
    distinct = ((np.linalg.norm(positions[:, None, :] - positions[None, :, :], axis=2) >= 4 - 1e-8)
                & (axes @ axes.T <= np.cos(np.radians(3)) + 1e-12))
    best_pair = None
    for first, second in combinations(range(len(ordered)), 2):
        if distinct[first, second]:
            if best_pair is None:
                best_pair = [ordered[first], ordered[second]]
            valid = np.flatnonzero(distinct[first, second + 1:] & distinct[second, second + 1:])
            if len(valid):
                return [ordered[first], ordered[second], ordered[second + 1 + int(valid[0])]]
    return best_pair or [ordered[0]]


def freeze_scenarios(config, groups):
    """Freeze shared coherent translations and an outer boundary band."""
    translations, boundary = config["translationMm"], config["boundaryMm"]
    keys = sorted(groups)
    variants = [{"id": "nominal", "sourceTranslationsMm": {}, "boundaryMm": 0.}]
    if boundary:
        variants.append({"id": "boundary-only", "sourceTranslationsMm": {}, "boundaryMm": boundary})
    if translations and keys:
        axes = np.array([[1, 0, 0], [-1, 0, 0], [0, 1, 0], [0, -1, 0], [0, 0, 1], [0, 0, -1]], dtype=float)
        diagonals = np.array([[1, 1, 1], [-1, -1, -1], [1, -1, 1], [-1, 1, -1],
                              [-1, 1, 1], [1, -1, -1], [1, 1, -1], [-1, -1, 1]], dtype=float) / np.sqrt(3)
        directions = np.concatenate([axes, diagonals])
        for index in range(16):
            # For two independent groups, each cardinal direction is tested
            # separately. Additional fixtures shift the groups simultaneously.
            if index < 12:
                offsets = {keys[index % len(keys)]: directions[index // len(keys)]}
            else:
                offsets = {key: diagonals[(index - 12 + 3 * num) % len(diagonals)]
                           for num, key in enumerate(keys)}
            by_source = {sid: (translations * direction).tolist() for key, direction in offsets.items() for sid in groups[key]}
            variants.append({"id": f"registration-{index + 1:02d}", "sourceTranslationsMm": by_source, "boundaryMm": 0.})
            if boundary:
                variants.append({"id": f"combined-{index + 1:02d}", "sourceTranslationsMm": by_source, "boundaryMm": boundary})
    scenarios, seen = [], set()
    for row in variants:
        signature = digest((row["sourceTranslationsMm"], row["boundaryMm"]))
        if signature not in seen:
            scenarios.append(row)
            seen.add(signature)
    definition = {"version": "shared-native-registration-boundary-v1", "config": config,
                  "registrationGroups": groups, "scenarios": scenarios,
                  "assumptions": [
                      "The capsule and patient reference are fixed. Registered source fields, labels and streamlines under the same registration root translate together in patient RAS.",
                      "The configurable 1 mm default translation is near the label grid scale. Rotations and a calibrated registration distribution are unassessed.",
                      "Boundary sensitivity pads only critical voxel-cell contact by 0.5 mm by default, near half the common 1 mm model grid spacing. The capsule, tract polylines, target and brain traversal stay fixed.",
                      "Target reach and brain traversal remain unchanged when both layers are in the patient reference frame.",
                      _DISCLOSURE]}
    return {"id": digest(definition), **definition}


def candidate_freeze(candidates, scenario_set, binding):
    rows = [{"id": row["id"], "geometryHash": digest({key: row[key] for key in ("entry", "target", "radiusMm", "direction")})}
            for row in candidates]
    data = {"scenarioSetId": scenario_set["id"], "binding": binding, "candidates": rows}
    return {"id": digest(data), **data}


def assess_sensitivity(nominal, scenario_set, measure, *, check):
    """Reuse WorldOutcome and empirical CVaR for a fixed sensitivity design."""
    readings = []
    for scenario in scenario_set["scenarios"]:
        check()
        status = "assessed"
        try:
            measured = nominal if scenario["id"] == "nominal" else measure(scenario)
            values, _ = nominal_objectives(measured)
            failures = measured["constraintFailures"]
            if not measured["coverage"]["withinReferenceEnvelope"]:
                failures = [*failures, "whole corridor outside reference image field"]
            if failures:
                status = "constraint-violation"
            elif any(value is None for value in values.values()):
                status = "partial-coverage"
            conflicts = measured["detectedConflictLayerIds"]
            outcome = WorldOutcome(costs=values, events={"declaredConstraintViolation": bool(failures)},
                                   accessible_target_volume_mm3=values["targetReachMm3"], unknowns=tuple(measured["unknowns"]))
            observations = source_readings(measured)
            unknown_sources = _unknown_source_layers(measured)
        except ValueError as exc:
            status, conflicts, failures = "unassessed", [], []
            observations = {}
            unknown_sources = []
            outcome = WorldOutcome(costs=dict.fromkeys(METRICS), events={"declaredConstraintViolation": None}, unknowns=(str(exc),))
        readings.append({"id": scenario["id"], "status": status, "values": dict(outcome.costs),
                         "unknowns": list(outcome.unknowns), "conflictLayerIds": conflicts, "constraintFailures": failures,
                         "sourceReadings": observations, "unknownSourceLayerIds": unknown_sources,
                         "missingMetrics": [key for key, value in outcome.costs.items() if value is None]})
    metrics = {}
    for name, (units, direction) in METRICS.items():
        adverse = 1 if direction == "higher" else -1
        known = [(row["values"][name], row["id"]) for row in readings if row["values"][name] is not None]
        worst = max(known, key=lambda item: adverse * item[0]) if known else (None, None)
        perturbations = [value for value, sid in known if sid != "nominal"]
        complete = len(known) == len(readings)
        tail = (adverse * upper_tail_cvar([adverse * value for value in perturbations], 1 - _TAIL_FRACTION)
                if complete and len(perturbations) >= 5 and name != "lengthMm" else None)
        metrics[name] = {"units": units, "adverseDirection": direction, "nominal": readings[0]["values"][name],
                         "worst": worst[0], "worstScenarioId": worst[1], "tailMean": tail,
                         "tailFraction": _TAIL_FRACTION, "assessedCount": len(known), "unknownCount": len(readings) - len(known),
                         "complete": complete}
    return {"scenarioSetId": scenario_set["id"], "scenarioCount": len(readings), "metrics": metrics,
            "scenarios": readings, "sourceReadingFields": SOURCE_READING_FIELDS,
            "meaning": "Worst assessed includes nominal. Tail mean uses the adverse empirical fifth of nonnominal scenarios when complete and at least five are available. " + _DISCLOSURE}
