"""Bound readouts for reviewing static native approaches."""

import numpy as np

from .anatomy_planning import native_target_binding
from .patient_anatomy import digest, world

SCHEMA = "native-approach-consequence-evidence-v1"
REVIEW_LIMIT = 3
GEOMETRY = "derived_geometry"
UNREVIEWED = "unclassified_source"
UNKNOWN_MEASUREMENT = "Current source-bound route measurement is unavailable."
NO_EXECUTION = "This route has no recorded native procedure."
NO_MAPPING = "No admitted route-specific functional mapping or monitoring evidence."
NO_FLOW = "No admitted regional flow assessment for this route."
NO_LOAD = "No admitted tissue load assessment for this route."

OUTCOMES = {
    "targetRemovedMm3": ("Target removed", "mm3", NO_EXECUTION),
    "targetResidualMm3": ("Target residual after treatment", "mm3", NO_EXECUTION),
    "motorEncounterMm3": ("Motor mapping encounter", "mm3", NO_MAPPING),
    "languageEncounterMm3": ("Language mapping encounter", "mm3", NO_MAPPING),
    "bloodGeneratedMm3": ("Bleeding volume", "mm3", NO_EXECUTION),
    "supplyDeficitSeconds": ("Regional supply deficit", "s", NO_FLOW),
    "drainageDeficitSeconds": ("Regional drainage deficit", "s", NO_FLOW),
    "brainStressSquaredPa2Mm3S": ("Regional stress exposure", "Pa2*mm3*s", NO_LOAD),
    "brainStrainSquaredMm3S": ("Regional strain exposure", "mm3*s", NO_LOAD),
    "patientShiftErrorMm": ("Observed shift error", "mm", "No admitted linked shift observation."),
    "clinicalDeficitProbability": ("Clinical deficit probability", "1", "No validated clinical risk model for these alternatives."),
}


def _readout(label, unit, value=None, *, evidence=None, status="geometric_estimate", reason=None):
    return {"label": label, "unit": unit, "value": value, "evidenceClass": evidence,
            "status": status if value is not None else "unavailable",
            "reason": reason if value is not None else reason or UNKNOWN_MEASUREMENT}


def _source_class(row):
    provenance = row.get("provenance", {})
    kind = provenance.get("kind")
    if (row.get("kind") == "synthetic-target" or kind in {"simulated", "generated", "synthetic"}
            or provenance.get("cortical_access_evidence") == "generated-reference"):
        return "generated_evidence"
    if kind == "geometric_estimate":
        return GEOMETRY
    if kind in {"model_estimate", "image_derived_estimate"} or provenance.get("model") or provenance.get("inference"):
        return "model_estimate"
    if (kind in {"observed_source_annotation", "supplied_annotation", "source_annotation"}
            or provenance.get("source") or row.get("original")):
        return "supplied_annotation"
    return UNREVIEWED


def _coverage(reading, *, tract=False):
    label = reading.get("coverage", {})
    acquisition = label if tract else reading.get("acquisitionCoverage", {})
    label_complete = None if tract else label.get("wholeEnvelopeWithinField")
    acquired_complete = acquisition.get("wholeEnvelopeWithinField")
    status = ("unavailable" if not reading.get("available") else
              "full-field" if acquired_complete and (tract or label_complete) else "partial-field")
    return {"status": status, "labelFieldComplete": label_complete,
            "acquisitionFieldComplete": acquired_complete,
            "labelOutsideFaces": [] if tract else label.get("outsideFaces", []),
            "acquisitionOutsideFaces": acquisition.get("outsideFaces", [])}


def _geometry_readout(label, unit, value, coverage, reason=None):
    partial = coverage["status"] == "partial-field"
    if value is not None:
        reason = "Derived from available source geometry. The required envelope has incomplete field coverage." if partial else None
    return _readout(label, unit, value, evidence=GEOMETRY,
                    status="partial_coverage" if partial else "geometric_estimate", reason=reason)


def _source_readout(descriptor, reading, purpose):
    tract = "points_ras_mm" in descriptor
    coverage = _coverage(reading, tract=tract)
    contact = (None if not reading.get("available") or purpose != "protection" else
               "detected" if reading.get("confirmedIntersection") else
               "unresolved" if reading.get("intersectsOrUnresolved") else "none-detected")
    fields = {}
    if purpose == "protection":
        fields["clearanceLowerBoundMm"] = _geometry_readout(
            "Clearance lower bound", "mm", reading.get("clearanceLowerBoundMm"), coverage, reading.get("reason"))
    if tract:
        for key, label, unit in (
                ("intersectedSegmentCount", "Intersected supplied segments", "segments"),
                ("intersectedStreamlineCount", "Distinct supplied streamlines encountered", "streamlines")):
            count = reading.get(key)
            if key == "intersectedStreamlineCount" and count is None and contact == "none-detected":
                count = 0
            fields[key] = _geometry_readout(
                label, unit, count, coverage,
                reading.get("reason") or "This measurement does not contain a distinct streamline count.")
    identity = ({"pointsSha256": descriptor["points_ras_mm"]["sha256"], "offsetsSha256": descriptor["offsets"]["sha256"]}
                if tract else {"dataSha256": descriptor["data"]["sha256"], "labelValue": descriptor["label_value"],
                               "revision": descriptor.get("revision", 0)})
    return {
        "layerId": descriptor["id"], "label": descriptor.get("label", descriptor["id"]),
        "kind": "tractography" if tract else descriptor["kind"],
        "role": descriptor.get("role"), "purpose": purpose, "evidenceClass": _source_class(descriptor),
        "sourceSeriesId": descriptor["source_series"], "reviewStatus": descriptor.get("review_status"),
        "model": descriptor.get("provenance", {}).get("model"),
        "coverageMeaning": reading.get("coverageMeaning") or descriptor.get("coverage"),
        "sourceIdentity": identity,
        "coverage": coverage, "contact": contact, "readouts": fields,
        "reason": ("Used in the cortical access assessment." if purpose == "access" else reading.get("reason")),
    }


def _target_dimensions(target, selected):
    reason = target.get("reason") or ("Target measurement unavailable." if selected else "No labelled target selected.")
    coverage = _coverage(target)
    volume = _readout("Supported target label volume", "mm3", target.get("totalVolumeMm3"),
                      evidence=GEOMETRY, status="source_count", reason=reason if not target.get("available") else None)
    membership = target.get("targetPointInsideLabel")
    membership_reason = reason
    if coverage["status"] != "full-field" and membership is False:
        membership = None
        membership_reason = "Incomplete field coverage prevents interpreting negative endpoint membership."
    return {
        "targetLabelVolumeMm3": volume,
        "targetVoxelCenterCoverageMm3": _geometry_readout(
            "Target voxel centres reached", "mm3", target.get("coveredVoxelCenterVolumeMm3"), coverage, reason),
        "targetVoxelCenterCoverageFraction": _geometry_readout(
            "Fraction of supported label centres reached", "1", target.get("coveredVoxelCenterFraction"), coverage, reason),
        "targetPointInsideLabel": _geometry_readout(
            "Endpoint inside target label", "boolean", membership, coverage, membership_reason),
    }


def _landmark_residual(row, matrix):
    landmarks = row.get("landmarks")
    if row.get("status") != "reviewed" or type(landmarks) is not dict or not landmarks.get("reviewer"):
        return None, None
    # Only sampled correspondence residuals from the stored review are reported.
    try:
        moving, reference, reported = (np.asarray(landmarks[key], float)
                                       for key in ("moving", "reference", "errors_mm"))
        if (moving.ndim != 2 or moving.shape[1] != 3 or moving.shape != reference.shape or len(moving) < 3
                or not np.isfinite(moving).all() or not np.isfinite(reference).all()
                or np.linalg.matrix_rank(moving - moving.mean(axis=0)) < 2
                or np.linalg.matrix_rank(reference - reference.mean(axis=0)) < 2):
            return None, None
        errors = np.linalg.norm(world(matrix, moving) - reference, axis=1)
        maximum = float(errors.max())
        if (reported.shape != errors.shape or not np.isfinite(errors).all() or maximum > 2
                or not np.allclose(reported, errors, rtol=0, atol=1e-8)
                or not np.isclose(float(row.get("uncertainty_mm")), maximum, rtol=0, atol=1e-8)):
            return None, None
    except (KeyError, ValueError, TypeError):
        return None, None
    return maximum, len(errors)


def _registration(collection, series_id):
    row = collection.document["transforms"].get(series_id, {})
    try:
        matrix = collection.registration_matrix(series_id, require_review=True, require_geometry=True)
    except (KeyError, ValueError, OSError, StopIteration) as exc:
        residual, samples, accepted, reason = None, None, False, str(exc)
    else:
        residual, samples = _landmark_residual(row, matrix)
        accepted, reason = True, None
    return {"sourceSeriesId": series_id, "status": row.get("status", "unavailable"),
            "admitted": accepted, "reason": reason, "method": row.get("method"),
            "parentSeriesId": row.get("parent_series"), "transformDigest": digest(row) if row else None,
            "landmarkSampleCount": samples,
            "landmarkResidual": _readout("Maximum sampled landmark residual", "mm", residual, evidence=GEOMETRY,
                                         reason="No verified sampled landmark residual is recorded." if residual is None else None),
            "wholeFieldError": _readout("Whole-field registration error", "mm",
                                        reason="No measured whole-field registration error.")}


def _route_readouts(collection, approach, descriptors, registrations, target_bindings):
    current = approach["assessmentStatus"] == "current"
    measured = approach.get("measurements", {}) if current else {}
    target_id = approach.get("targetLayerId")
    target = next((row for row in measured.get("targets", []) if row["layerId"] == target_id), {})
    normal = measured.get("normalTraversal", {})
    access = measured.get("accessAssessment")
    structures = measured.get("structures", [])
    sources = [_source_readout(descriptors[row["layerId"]], row, "protection")
               for row in structures if row["layerId"] in descriptors]
    if target_id in descriptors and current:
        sources.append(_source_readout(descriptors[target_id], target, "target"))
    if normal.get("layerId") in descriptors:
        sources.append(_source_readout(descriptors[normal["layerId"]], normal, "traversal"))
    access_ids = {row["layerId"] for row in (access or {}).get("sourceLayers", {}).values()}
    for key in sorted(access_ids - {row["layerId"] for row in sources}):
        if key in descriptors:
            sources.append(_source_readout(descriptors[key], {}, "access"))

    route_registration_ids = {row["sourceSeriesId"] for row in sources}
    if current:
        route_registration_ids.add(collection.document["reference_series"])
    route_registrations = {}
    for series_id in sorted(route_registration_ids):
        while series_id and series_id not in route_registrations:
            if series_id not in registrations:
                registrations[series_id] = _registration(collection, series_id)
            row = registrations[series_id]
            route_registrations[series_id] = row
            series_id = row["parentSeriesId"]

    bundles = [row for row in sources if row["kind"] == "tractography"]
    detected_bundles = sorted({row["layerId"] for row in bundles if row["contact"] == "detected"})
    available_bundles = [row for row in bundles if row["coverage"]["status"] != "unavailable"]
    partial_bundles = any(row["coverage"]["status"] != "full-field" for row in bundles)
    target_source = next((row for row in sources if row["purpose"] == "target"), None)
    dimensions = {
        "lengthMm": _readout("Corridor length", "mm", measured.get("lengthMm"), evidence=GEOMETRY),
        **_target_dimensions(target, target_id),
        "nonTargetTraversalUpperBoundMm3": _geometry_readout(
            "Brain traversal volume upper bound", "mm3", normal.get("conservativeVolumeUpperBoundMm3"),
            _coverage(normal), normal.get("reason")),
        "tractBundlesEncountered": _readout(
            "Distinct source bundles encountered", "bundles", len(detected_bundles) if available_bundles else None,
            evidence=GEOMETRY, status="partial_coverage" if partial_bundles else "geometric_estimate",
            reason=("Some supplied bundles or source fields could not be assessed." if available_bundles and partial_bundles
                    else "No measurable supplied streamline bundle." if not available_bundles else None)),
        **{name: _readout(label, unit, reason=reason) for name, (label, unit, reason) in OUTCOMES.items()},
    }
    sensitivity = measured.get("sensitivity", {})
    missing = measured.get("declaredMissingAnatomy", [])
    reasons = ["Recorded treatment and route-specific functional, vascular and mechanical evidence are unavailable."]
    if not current:
        reasons.insert(0, approach.get("invalidationReason", UNKNOWN_MEASUREMENT))
    elif measured.get("constraintFailures"):
        reasons.insert(0, "One or more declared geometric constraints failed.")
    if current and (not access or access.get("status") != "supported"):
        reasons.insert(0, "Cortical access evidence is incomplete or a declared access constraint failed.")
    binding = approach.get("targetCandidateBinding")
    reference = measured.get("coverage", {}).get("reference", {})
    return {
        "approachId": approach["id"], "name": approach["name"], "parameterHash": approach.get("parameterHash"),
        "measurementHash": approach.get("measurementHash"), "sourceSnapshotId": approach.get("sourceSnapshotId"),
        "entry": approach["entry"], "targetPoint": approach["target"], "radiusMm": approach["radiusMm"],
        "targetLayerId": target_id,
        "targetCandidateBinding": binding or (target_bindings.get(target_id) if current else None),
        "assessmentStatus": approach["assessmentStatus"],
        "researchDisposition": {"status": "reassess", "reasons": reasons},
        "target": target_source, "dimensions": dimensions, "sources": sources,
        "registrations": list(route_registrations.values()),
        "referenceCoverage": {"wholeEnvelopeWithinField": reference.get("wholeEnvelopeWithinField"),
                              "outsideFaces": reference.get("outsideFaces", [])},
        "tractEncounterLayerIds": detected_bundles,
        "warnings": {"criticalContacts": [{"layerId": row["layerId"], "label": row["label"], "contact": row["contact"]}
                                           for row in sources if row["contact"] in {"detected", "unresolved"}],
                     "constraintFailures": measured.get("constraintFailures", []),
                     "access": access, "missingEvidence": missing},
        "uncertainty": {"scenarioSetId": sensitivity.get("scenarioSetId"), "scenarioCount": sensitivity.get("scenarioCount"),
                        "metrics": sensitivity.get("metrics", {}), "guard": measured.get("scenarioGuard"),
                        "limitations": measured.get("unknowns", []),
                        "meaning": "Declared stress scenarios probe geometric sensitivity. Anatomical detection sensitivity and clinical probabilities are unmeasured."},
    }


def consequence_evidence_view(collection, approaches, active_target, review_id=None):
    """Reuse current measurements and leave historical route readouts unavailable."""
    routes = approaches[:REVIEW_LIMIT]
    if isinstance(review_id, str) and all(row["id"] != review_id for row in routes):
        chosen = next((row for row in approaches if row["id"] == review_id), None)
        if chosen is not None:
            routes[-1] = chosen
    descriptors = {row["id"]: row for row in (*collection.document["layers"], *collection.document.get("tractography", []))}
    target_ids = {row.get("targetLayerId") for row in routes} | {active_target}
    target_bindings = {}
    for key in target_ids.intersection(descriptors):
        try:
            target_bindings[key] = native_target_binding(collection, descriptors[key])
        except (KeyError, ValueError, OSError, StopIteration):
            pass
    registrations = {}
    return {"schema": SCHEMA, "scope": "Static native geometry in patient RAS millimetres.",
            "sourceSnapshotId": collection.identity(), "targetLayerId": active_target,
            "targetCandidateBinding": target_bindings.get(active_target), "simulationExecutionAuthorized": False,
            "clinicalValidation": False,
            "routes": [_route_readouts(collection, route, descriptors, registrations, target_bindings) for route in routes]}
