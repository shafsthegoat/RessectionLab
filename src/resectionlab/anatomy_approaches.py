"""Saved native source-bound approach geometry, separate from simulator admission.

Source arrays stay on their original grids. Proposals reuse the static capsule
cell supercover and never call physical planning or authorize tool execution.
"""
from copy import deepcopy
import fcntl
import json
import os
import re
import tempfile
import uuid

import numpy as np
from scipy.spatial import cKDTree

from .patient_anatomy import PatientAnatomySnapshot, affine, digest, world
from .geometry import (GEOMETRY_VERSION, GeometryScene, AccessWindow,
                       capsule_voxel_indices, point_segment_distances)

VASCULAR_KINDS = {"vessels", "artery", "vein", "sinus"}
TRACT_KINDS = {"tract", "tractography", "motor", "language", "functional"}
SCHEMA = "patient-anatomy-approaches-v1"
FILENAME = "anatomy-approaches.json"
MAX_SAVED = 32
ASSESSMENT_VERSION = "native-source-capsule-assessment-v2"


def checked_route(start, end, radius_mm):
    try:
        start, end = np.asarray(start, float), np.asarray(end, float)
        radius_mm = float(radius_mm)
    except (TypeError, ValueError):
        raise ValueError("A finite physical route and positive radius <=50 mm are required") from None
    if (start.shape != (3,) or end.shape != (3,) or not np.isfinite(start).all()
            or not np.isfinite(end).all() or not np.isfinite(radius_mm)
            or not 0 < radius_mm <= 50):
        raise ValueError("A finite physical route and positive radius <=50 mm are required")
    length = float(np.linalg.norm(end - start))
    if not 1e-6 < length <= 1000:
        raise ValueError("Route length must be positive and <=1000 mm")
    return start, end, radius_mm, length


def envelope_coverage(a, shape, start, end, radius_mm):
    """Exact capsule containment in a convex affine voxel-cell image field.

    A sphere's extent along inverse-affine row i is r*||row_i||. Endpoint
    bounds plus that extent enclose the entire capsule, including its caps.
    """
    inverse = np.linalg.inv(affine(a))
    points = world(inverse, np.stack([start, end]))
    extent = radius_mm * np.linalg.norm(inverse[:3, :3], axis=1)
    lo, hi = points.min(axis=0) - extent, points.max(axis=0) + extent
    lower, upper = lo + .5, np.asarray(shape) - .5 - hi
    return {"wholeEnvelopeWithinField": bool(np.all(lower >= 0) and np.all(upper >= 0)),
            "centerlineWithinField": bool(np.all(points >= -.5) and np.all(points <= np.asarray(shape) - .5)),
            "envelopeVoxelBounds": [lo.tolist(), hi.tolist()],
            "outsideFaces": [f"{axis}{side}" for axis in range(3)
                             for side, values in (("-", lower), ("+", upper)) if values[axis] < 0],
            "meaning": "whole capsule in source image field; field coverage does not establish anatomical sensitivity"}


def _segment_bundle_distances(start, end, first, last):
    """Exact segment-to-segment distances, including parallel/degenerate paths."""
    route, segments, offset = end - start, last - first, start - first
    aa = float(route @ route)
    bb, cc = segments @ route, np.einsum("ij,ij->i", segments, segments)
    dd, ee = offset @ route, np.einsum("ij,ij->i", segments, offset)
    denominator = aa * cc - bb * bb
    # Match the maintained static geometry's near-parallel predicate exactly.
    stable = denominator > 1e-18 * np.maximum(aa * cc, 1)
    t = np.divide(bb * ee - cc * dd, denominator, out=np.zeros_like(cc), where=stable)
    u = np.divide(aa * ee - bb * dd, denominator, out=np.zeros_like(cc), where=stable)
    interior = stable & (t >= 0) & (t <= 1) & (u >= 0) & (u <= 1)
    distances = np.minimum(point_segment_distances(first, start, end), point_segment_distances(last, start, end))
    for endpoint in (start, end):
        fraction = np.clip(np.divide(np.einsum("ij,ij->i", endpoint - first, segments), cc,
                                      out=np.zeros_like(cc), where=cc > 0), 0, 1)
        distances = np.minimum(distances, np.linalg.norm(first + fraction[:, None] * segments - endpoint, axis=1))
    distances[interior] = np.minimum(distances[interior], np.linalg.norm(
        start + t[interior, None] * route - first[interior] - u[interior, None] * segments[interior], axis=1))
    return distances


class NativeRouteGeometry:
    """One immutable revision's analysis cache, with bounded geometry batches."""
    def __init__(self, snapshot, *, check=lambda: None):
        snapshot._check()
        self.snapshot, self.c, self.check = snapshot, snapshot.collection, check
        self.grids, self.masks, self.centers, self.tracts = {}, {}, {}, {}

    def grid(self, row):
        a = self.c.patient_affine(row)
        shape = tuple(row["data"]["shape"])
        key = (shape, a.tobytes())
        if key not in self.grids:
            self.grids[key] = GeometryScene(np.zeros(shape, bool), a)
        return key, self.grids[key]

    def mask(self, row):
        key = (row["data"]["sha256"], row["label_value"])
        if key not in self.masks:
            self.masks[key] = self.c.mask(row)
        return self.masks[key]

    def fully_inside_target(self, scene, indices, target):
        """Prove whole brain cells lie in supplied target cells before exclusion.

        A target-grid AABB encloses each transformed brain parallelepiped. Every
        target cell touching that box must be labelled, including boundary cells.
        This can retain extra brain cells; it cannot remove an unproved cell.
        """
        relative = np.linalg.inv(self.c.patient_affine(target)) @ scene.affine
        centers = world(relative, indices)
        extent = .5 * np.abs(relative[:3, :3]).sum(axis=1)
        lo = np.floor(centers - extent + .5).astype(int)
        hi = np.floor(centers + extent + .5).astype(int)
        mask = self.mask(target)
        inside = np.all((lo >= 0) & (hi < np.asarray(mask.shape)), axis=1)
        bounded = np.prod(hi - lo + 1, axis=1) <= 4096
        contained = np.zeros(len(indices), bool)
        for index in np.flatnonzero(inside & bounded):
            if index % 256 == 0:
                self.check()
            low, high = lo[index], hi[index] + 1
            contained[index] = bool(mask[low[0]:high[0], low[1]:high[1], low[2]:high[2]].all())
        return contained

    def measure(self, start, end, radius_mm, constraints=None):
        self.snapshot._check()
        start, end, radius_mm, length = checked_route(start, end, radius_mm)
        constraints = constraints or {}
        excluded = set(constraints.get("excludedLayerIds", []))
        c = self.c
        reference = c.series(c.document["reference_series"])
        ref_coverage = envelope_coverage(c.patient_affine(reference), reference["data"]["shape"], start, end, radius_mm)
        unknowns = list(c.document.get("missing", []))
        structures, targets, source_coverage, covers = [], [], [], {}

        def cover(row):
            key, scene = self.grid(row)
            if key not in covers:
                covers[key] = capsule_voxel_indices(scene, start, end, radius_mm, check=self.check)
            return scene, covers[key]

        for row in c.document["layers"]:
            self.check()
            forbidden = any(word in (row.get("label", "") + " " + row["id"]).lower()
                            for word in ("cerebell", "brainstem", "brain-stem", "brain stem", "spinal cord", "medulla", "pons"))
            critical = row["role"] == "critical" or row["kind"] in VASCULAR_KINDS | TRACT_KINDS or row["id"] in excluded or forbidden
            if not critical and row["role"] != "target":
                continue
            item = {"layerId": row["id"], "label": row.get("label", row["id"]), "kind": row["kind"],
                    "available": False, "confirmedIntersection": False,
                    "excluded": forbidden or row["id"] in excluded, "sourceSeries": row["source_series"],
                    "reviewStatus": row.get("review_status", "unreviewed"), "coverageMeaning": row.get("coverage", "unassessed"),
                    "sourceIdentity": {"dataSha256": row["data"]["sha256"], "labelValue": row["label_value"],
                                       "revision": row.get("revision", 0)}}
            try:
                a = c.patient_affine(row)
                field = envelope_coverage(a, row["data"]["shape"], start, end, radius_mm)
                item["coverage"] = field
                source_coverage.append({"layerId": row["id"], **field})
                if not field["wholeEnvelopeWithinField"]:
                    unknowns.append(row["id"] + ": corridor envelope outside source field")
                mask = self.mask(row)
                if not mask.any():
                    item["reason"] = "empty label; anatomy is unassessed"
                    unknowns.append(row["id"] + ": empty label")
                else:
                    scene, indices = cover(row)
                    hit = indices[mask[tuple(indices.T)]]
                    orthogonal = scene._orthogonal_spacing is not None
                    key = (row["id"], a.tobytes())
                    if key not in self.centers:
                        self.centers[key] = world(a, np.argwhere(mask))
                    centers = self.centers[key]
                    minimum, nearest, center_hits = float("inf"), None, 0
                    for offset in range(0, len(centers), 131072):
                        self.check()
                        points = centers[offset:offset + 131072]
                        distances = point_segment_distances(points, start, end)
                        index = int(distances.argmin())
                        if distances[index] < minimum:
                            minimum, nearest = float(distances[index]), points[index].tolist()
                        center_hits += int(np.count_nonzero(distances <= radius_mm))
                    lower = minimum - scene._cell_radius_mm - radius_mm
                    item.update(available=True, clearanceLowerBoundMm=float(lower),
                                intersectsOrUnresolved=bool(len(hit)), confirmedIntersection=bool(len(hit) if orthogonal else center_hits),
                                intersectionModel="exact capsule / affine voxel cell" if orthogonal else "conservative cell circumsphere; shear",
                                conservativeOverlapVolumeUpperBoundMm3=float(len(hit) * scene.voxel_volume_mm3),
                                intersectedCellCount=int(len(hit)), nearestPoint=nearest)
                    if not orthogonal and len(hit) and not center_hits:
                        unknowns.append(row["id"] + ": conservative sheared-cell contact unresolved")
                    if row["role"] == "target":
                        endpoint = np.floor(world(np.linalg.inv(a), end) + .5).astype(int)
                        endpoint_inside = bool(np.all(endpoint >= 0) and np.all(endpoint < mask.shape) and mask[tuple(endpoint)])
                        targets.append({**item, "targetPointInsideLabel": endpoint_inside,
                                        "totalVolumeMm3": float(len(centers) * scene.voxel_volume_mm3),
                                        "coveredVoxelCenterVolumeMm3": float(center_hits * scene.voxel_volume_mm3),
                                        "coveredVoxelCenterFraction": center_hits / len(centers),
                                        "coverageModel": "target voxel centres in entry-to-target capsule; geometric reach only, no removal"})
            except ValueError as exc:
                item["reason"] = str(exc)
                unknowns.append(row["id"] + ": " + str(exc))
            if critical:
                structures.append(item)
            if row["role"] == "target" and not item["available"]:
                targets.append(item)

        for tract in c.document.get("tractography", []):
            self.check()
            item = {"layerId": tract["id"], "label": tract.get("label", tract["id"]), "kind": "tractography",
                    "available": False, "confirmedIntersection": False, "excluded": tract["id"] in excluded,
                    "sourceSeries": tract["source_series"], "coverageMeaning": "supplied streamline paths; bundle extent and detection sensitivity unassessed",
                    "sourceIdentity": {"pointsSha256": tract["points_ras_mm"]["sha256"], "offsetsSha256": tract["offsets"]["sha256"]}}
            try:
                source = c.series(tract["source_series"])
                a = c.patient_affine(source)
                field = envelope_coverage(a, source["data"]["shape"], start, end, radius_mm)
                item["coverage"] = field
                source_coverage.append({"layerId": tract["id"], **field})
                if not field["wholeEnvelopeWithinField"]:
                    unknowns.append(tract["id"] + ": corridor envelope outside tract source field")
                if tract["id"] not in self.tracts:
                    points = world(c.registration_matrix(source["id"]), c.array(tract["points_ras_mm"]))
                    offsets = c.array(tract["offsets"])
                    valid = np.ones(max(0, len(points) - 1), bool)
                    ends = np.asarray(offsets[1:-1], int) - 1
                    valid[ends[(ends >= 0) & (ends < len(valid))]] = False
                    first, last = points[:-1][valid], points[1:][valid]
                    # Midpoint trees only prune. The segment predicate below
                    # still evaluates original transformed source coordinates.
                    tree = cKDTree((first + last) / 2) if len(first) else None
                    half = float(np.linalg.norm(last - first, axis=1).max() / 2) if len(first) else 0
                    self.tracts[tract["id"]] = (first, last, tree, half)
                first, last, tree, half = self.tracts[tract["id"]]
                if not len(first):
                    item["reason"] = "empty streamline bundle; unassessed"
                    unknowns.append(tract["id"] + ": empty streamline bundle")
                else:
                    # A nearest midpoint supplies an upper bound to the true
                    # route/segment minimum. Every segment that can improve it
                    # or intersect the capsule has a midpoint within this
                    # radius of a route sample (triangle inequality). Thus no
                    # positive label is lost through display-style sampling.
                    samples = np.linspace(start, end, max(2, int(np.ceil(length / 4)) + 1))
                    upper = float(tree.query(samples)[0].min())
                    step = length / (len(samples) - 1)
                    bound = max(upper, radius_mm) + half + step / 2 + 1e-8
                    selected = np.zeros(len(first), bool)
                    for sample in samples:
                        self.check()
                        selected[tree.query_ball_point(sample, bound)] = True
                    first, last = first[selected], last[selected]
                    minimum, nearest, hits = float("inf"), None, 0
                    for offset in range(0, len(first), 131072):
                        self.check()
                        aa, bb = first[offset:offset + 131072], last[offset:offset + 131072]
                        distances = _segment_bundle_distances(start, end, aa, bb)
                        index = int(distances.argmin())
                        if distances[index] < minimum:
                            minimum, nearest = float(distances[index]), ((aa[index] + bb[index]) / 2).tolist()
                        hits += int(np.count_nonzero(distances <= radius_mm))
                    item.update(available=True, clearanceLowerBoundMm=float(minimum - radius_mm),
                                intersectsOrUnresolved=bool(hits), confirmedIntersection=bool(hits),
                                intersectionModel="capsule versus supplied polyline segments; no inferred bundle thickness",
                                intersectedSegmentCount=hits, nearestPoint=nearest)
            except ValueError as exc:
                item["reason"] = str(exc)
                unknowns.append(tract["id"] + ": " + str(exc))
            structures.append(item)

        normal = {"available": False, "reason": "no supplied brain mask; normal tissue traversal unassessed"}
        brain = next((r for r in c.document["layers"] if r["kind"] == "brain"), None)
        if brain:
            try:
                if not self.mask(brain).any():
                    raise ValueError("empty brain mask; normal tissue traversal unassessed")
                scene, indices = cover(brain)
                selected = self.mask(brain)[tuple(indices.T)]
                for target in (r for r in c.document["layers"] if r["role"] == "target"):
                    selected &= ~self.fully_inside_target(scene, indices, target)
                normal = {"available": True, "layerId": brain["id"],
                          "conservativeVolumeUpperBoundMm3": float(selected.sum() * scene.voxel_volume_mm3),
                          "coverage": envelope_coverage(scene.affine, scene.shape, start, end, radius_mm),
                          "meaning": "whole-cell upper bound of non-target brain-mask traversal; subtract only cells proved wholly contained in supplied target labels; no tissue removal",
                          "targetExclusionModel": "complete transformed brain-cell target-grid box must be labelled; uncertain and partly labelled cells remain in the upper bound"}
                if not normal["coverage"]["wholeEnvelopeWithinField"]:
                    unknowns.append(brain["id"] + ": normal traversal outside supplied brain mask field")
            except ValueError as exc:
                normal["reason"] = str(exc)
        if not normal["available"]:
            unknowns.append("normal tissue traversal unassessed")
        if not any(s["available"] and s["kind"] in VASCULAR_KINDS for s in structures):
            unknowns.append("vascular anatomy unavailable")
        if not any(s["available"] and s["kind"] in TRACT_KINDS for s in structures):
            unknowns.append("named tract anatomy unavailable")
        if not targets:
            unknowns.append("no labelled target region; point reach only")
        if not ref_coverage["wholeEnvelopeWithinField"]:
            unknowns.append("corridor envelope outside reference image field")
        unknowns.extend(["unlabelled anatomy and anatomical detection sensitivity unassessed",
                         "registration accuracy unmeasured unless independently documented",
                         "skull opening, tissue forces and clinical outcomes unassessed"])
        conflicts = [s["layerId"] for s in structures if s.get("confirmedIntersection")]
        failures = []
        if constraints.get("maxLengthMm") is not None and length > constraints["maxLengthMm"]:
            failures.append("corridor exceeds declared maximum length")
        for item in structures:
            if item.get("excluded") and item.get("intersectsOrUnresolved"):
                failures.append(item["label"] + ": intersects or may intersect an excluded structure")
            if item.get("available") and item["clearanceLowerBoundMm"] < constraints.get("minClearanceMm", 0):
                failures.append(item["label"] + ": conservative clearance below declared minimum")
        axis = (end - start) / length
        return {"geometryVersion": GEOMETRY_VERSION, "assessmentVersion": ASSESSMENT_VERSION, "lengthMm": length, "radiusMm": radius_mm,
                "direction": axis.tolist(),
                "orientation": {"azimuthDeg": float(np.degrees(np.arctan2(axis[1], axis[0]))),
                                "elevationDeg": float(np.degrees(np.arcsin(np.clip(axis[2], -1, 1)))),
                                "frame": "patient RAS+; entry to target"},
                "structures": structures, "targets": targets, "normalTraversal": normal,
                "coverage": {"withinReferenceEnvelope": ref_coverage["wholeEnvelopeWithinField"],
                             "reference": ref_coverage, "sources": source_coverage,
                             "allAvailableSourceFieldsCoverEnvelope": bool(source_coverage) and all(s["wholeEnvelopeWithinField"] for s in source_coverage)},
                "status": "known-conflict" if conflicts or failures else "incomplete",
                "detectedConflictLayerIds": conflicts, "constraintFailures": failures,
                "labelledAnatomyResult": "detected-conflict" if conflicts else "no-detected-conflict",
                "unknowns": list(dict.fromkeys(unknowns)), "clinicalDeficitProbability": None,
                "simulationExecutionAuthorized": False,
                "assumptions": ["Static straight capsule in physical patient RAS millimetres; no brain shift or removal.",
                                "A numerical lower bound is not an exact clinical safety margin.",
                                "Field-of-view containment covers the complete capsule, including spherical ends.",
                                "Negative labels and streamline gaps never establish absence of critical anatomy."]}


def checked_approach(collection, value):
    if not isinstance(value, dict):
        raise ValueError("An approach object is required")
    start, end, radius, length = checked_route(value.get("entry", value.get("start")), value.get("target", value.get("end")), value.get("radiusMm", 1))
    key = value.get("id") or "approach-" + uuid.uuid4().hex[:16]
    if not isinstance(key, str) or not re.fullmatch(r"[A-Za-z0-9_-]{1,96}", key):
        raise ValueError("Invalid approach identity")
    name = value.get("name", "Manual approach")
    if not isinstance(name, str) or not name.strip() or len(name) > 120:
        raise ValueError("Approach name must contain 1–120 characters")
    constraints = value.get("constraints", {})
    if not isinstance(constraints, dict) or set(constraints) - {"maxLengthMm", "minClearanceMm", "excludedLayerIds", "supratentorialOnly"}:
        raise ValueError("Unknown approach constraint")
    constraints = deepcopy(constraints)
    for field, minimum, maximum in (("maxLengthMm", .1, 1000), ("minClearanceMm", 0, 50)):
        if field in constraints and (type(constraints[field]) not in (int, float) or not np.isfinite(constraints[field]) or not minimum <= constraints[field] <= maximum):
            raise ValueError("Invalid " + field)
    ids = {r["id"] for r in collection.document["layers"] + collection.document.get("tractography", [])}
    excluded = constraints.get("excludedLayerIds", [])
    if not isinstance(excluded, list) or len(excluded) > 128 or any(not isinstance(x, str) or x not in ids for x in excluded):
        raise ValueError("Unknown excluded layer")
    constraints["excludedLayerIds"] = sorted(set(excluded))
    # This product's automated scope is explicitly supratentorial. An unchecked
    # client cannot turn proposal search into an inferior whole-head search.
    if constraints.get("supratentorialOnly", True) is not True:
        raise ValueError("Only scoped supratentorial proposals are supported")
    constraints["supratentorialOnly"] = True
    patch = value.get("entryPatch") or {"center": start.tolist(), "normal": ((end - start) / length).tolist(), "radiusMm": 15.0}
    if not isinstance(patch, dict):
        raise ValueError("Invalid entry patch")
    center, normal = np.asarray(patch.get("center"), float), np.asarray(patch.get("normal"), float)
    patch_radius = patch.get("radiusMm", 15)
    if (center.shape != (3,) or normal.shape != (3,) or not np.isfinite(center).all()
            or not np.isfinite(normal).all() or np.linalg.norm(normal) < 1e-8
            or type(patch_radius) not in (int, float) or not np.isfinite(patch_radius) or not 0 <= patch_radius <= 40):
        raise ValueError("Entry patch needs physical center, normal and radius <=40 mm")
    normal = normal / np.linalg.norm(normal)
    patch = {"center": center.tolist(), "normal": normal.tolist(), "radiusMm": float(patch_radius),
             "meaning": "user-selected hypothetical supratentorial entry-centre disk; cortical/skull suitability unverified"}
    return {"id": key, "name": name.strip(), "entry": start.tolist(), "target": end.tolist(),
            "radiusMm": radius, "direction": ((end - start) / length).tolist(),
            "entryPatch": patch, "constraints": constraints}


def parameter_hash(approach):
    return digest({key: approach[key] for key in ("entry", "target", "radiusMm", "entryPatch", "constraints")})


def read_approaches(collection):
    path = collection.root / FILENAME
    if not path.exists():
        return []
    if path.stat().st_size > 2 * 1024**2:
        raise ValueError("Saved comparison exceeds metadata budget")
    data = json.loads(path.read_text())
    if not isinstance(data, dict) or data.get("schema") != SCHEMA or data.get("patientId") != collection.document["patient_id"] or not isinstance(data.get("approaches"), list) or len(data["approaches"]) > MAX_SAVED:
        raise ValueError("Saved comparison belongs to another patient or schema")
    result, identities = [], set()
    for saved in data["approaches"]:
        if not isinstance(saved, dict) or ("measurements" in saved and not isinstance(saved["measurements"], dict)):
            raise ValueError("Invalid saved approach measurements")
        # Stale exclusions may refer to retired source layers. Preserve the plan
        # for inspection, but never silently approve its cached measurement.
        candidate = deepcopy(saved)
        original_constraints = candidate.get("constraints", {})
        known = {r["id"] for r in collection.document["layers"] + collection.document.get("tractography", [])}
        retired = set(original_constraints.get("excludedLayerIds", [])) - known
        candidate["constraints"] = {**original_constraints, "excludedLayerIds": [x for x in original_constraints.get("excludedLayerIds", []) if x in known]}
        checked = checked_approach(collection, candidate)
        checked["constraints"] = original_constraints
        if checked["id"] in identities:
            raise ValueError("Duplicate saved approach")
        identities.add(checked["id"])
        checked.update({key: deepcopy(saved[key]) for key in ("sourceSnapshotId", "parameterHash", "measurementHash", "measurements", "origin", "tradeoff") if key in saved})
        valid = (not retired and saved.get("sourceSnapshotId") == collection.identity()
                 and saved.get("parameterHash") == parameter_hash(checked)
                 and saved.get("measurements", {}).get("geometryVersion") == GEOMETRY_VERSION
                 and saved.get("measurements", {}).get("assessmentVersion") == ASSESSMENT_VERSION
                 and saved.get("measurementHash") == digest(saved.get("measurements")))
        checked["assessmentStatus"] = "current" if valid else "stale" if saved.get("measurements") else "unassessed"
        if not valid:
            checked["invalidationReason"] = "Source anatomy, geometry engine or approach parameters changed; recompute before comparing."
        result.append(checked)
    return result


def save_approaches(collection, approaches, *, commit=lambda: None):
    if len(approaches) > MAX_SAVED:
        raise ValueError("At most 32 saved approaches are supported")
    data = {"schema": SCHEMA, "patientId": collection.document["patient_id"], "approaches": approaches}
    encoded = json.dumps(data, allow_nan=False, indent=2)
    if len(encoded) > 2 * 1024**2:
        raise ValueError("Saved comparison exceeds metadata budget")
    with (collection.root / ".write.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        current = json.loads((collection.root / "patient-anatomy.json").read_text())
        if current.get("snapshot_id") != collection.identity():
            raise ValueError("Patient changed; reload before saving approaches")
        commit()
        with tempfile.NamedTemporaryFile(dir=collection.root, mode="w", delete=False) as out:
            out.write(encoded)
            temporary = out.name
        os.replace(temporary, collection.root / FILENAME)


def assess_approach(collection, value, geometry=None):
    approach = checked_approach(collection, value)
    geometry = geometry or NativeRouteGeometry(PatientAnatomySnapshot.capture(collection))
    approach.update(sourceSnapshotId=collection.identity(), parameterHash=parameter_hash(approach),
                    assessmentStatus="current", measurements=geometry.measure(approach["entry"], approach["target"], approach["radiusMm"], approach["constraints"]))
    if approach["entry"][2] < approach["target"][2]:
        approach["measurements"]["constraintFailures"].append("entry inferior to target; outside the declared superior/lateral supratentorial proposal scope")
        approach["measurements"]["status"] = "known-conflict"
    patch = approach["entryPatch"]
    delta = np.asarray(approach["entry"]) - patch["center"]
    normal = np.asarray(patch["normal"])
    plane_distance = float(delta @ normal)
    radial_distance = float(np.linalg.norm(delta - plane_distance * normal))
    if abs(plane_distance) > 1e-5 or radial_distance > patch["radiusMm"] + 1e-5:
        approach["measurements"]["constraintFailures"].append("entry outside declared entry-centre disk; move the patch or choose an entry within its plane and radius")
        approach["measurements"]["status"] = "known-conflict"
    approach["measurementHash"] = digest(approach["measurements"])
    return approach


def suggest_approaches(collection, args, geometry=None):
    """Enumerate a small declared patch, never search the whole head surface."""
    base = checked_approach(collection, {**args, "entry": args.get("entry", args.get("start")),
                                       "target": args.get("target", args.get("end")),
                                       "entryPatch": args.get("entryPatch")})
    if "patchRadiusMm" in args:
        base["entryPatch"]["radiusMm"] = args["patchRadiusMm"]
        base = checked_approach(collection, base)
    geometry = geometry or NativeRouteGeometry(PatientAnatomySnapshot.capture(collection))
    patch = base["entryPatch"]
    # Reuse the static planner's access-window geometry in a user-bound patch.
    window = AccessWindow(tuple(patch["center"]), tuple(patch["normal"]), max(patch["radiusMm"], .01), "native-selected-entry-patch")
    normal = np.asarray(window.normal_inward)
    helper = np.eye(3)[int(np.argmin(np.abs(normal)))]
    first = np.cross(normal, helper); first /= np.linalg.norm(first)
    second = np.cross(normal, first)
    center = np.asarray(window.center_mm)
    positions = [center]
    for scale in (.55, 1.0):
        positions.extend(center + patch["radiusMm"] * scale * (np.cos(angle) * first + np.sin(angle) * second)
                         for angle in np.linspace(0, 2 * np.pi, 8, endpoint=False))
    accepted, reasons, evaluated = [], {}, 0
    for index, position in enumerate(positions):
        geometry.check()
        reason = None
        if position[2] < base["target"][2]:
            reason = "inferior entries excluded by supratentorial scope"
        elif any(np.linalg.norm(position - np.asarray(a["entry"])) < 1 for a in accepted):
            reason = "duplicate entry geometry"
        if reason:
            reasons[reason] = reasons.get(reason, 0) + 1
            continue
        candidate = assess_approach(collection, {**base, "id": "proposal-" + digest([collection.identity(), parameter_hash(base), position.tolist()])[:16],
                                                "entry": position.tolist(), "name": f"Patch alternative {index + 1}"}, geometry)
        evaluated += 1
        measured = candidate["measurements"]
        if not measured["coverage"]["withinReferenceEnvelope"]:
            reason = "whole corridor outside reference image field"
        elif measured["constraintFailures"]:
            reason = "declared clearance/length/exclusion constraints not met"
        if reason:
            reasons[reason] = reasons.get(reason, 0) + 1
        else:
            accepted.append(candidate)
    def clearance(candidate):
        values = [s["clearanceLowerBoundMm"] for s in candidate["measurements"]["structures"] if s["available"]]
        return min(values) if values else float("-inf")
    def target_coverage(candidate):
        return sum(t.get("coveredVoxelCenterVolumeMm3", 0) for t in candidate["measurements"]["targets"])
    ordered = [sorted(accepted, key=lambda a: (a["measurements"]["lengthMm"], a["id"])),
               sorted(accepted, key=lambda a: (-clearance(a), a["measurements"]["lengthMm"], a["id"])),
               sorted(accepted, key=lambda a: (-target_coverage(a), a["measurements"]["lengthMm"], a["id"]))]
    selected = []
    labels = ["Shortest measured corridor", "Clearance-focused alternative", "Target reach alternative"]
    for objective, (label, candidates) in enumerate(zip(labels, ordered)):
        for candidate in candidates:
            if all(np.linalg.norm(np.asarray(candidate["entry"]) - existing["entry"]) >= 4
                   and np.degrees(np.arccos(np.clip(np.dot(candidate["direction"], existing["direction"]), -1, 1))) >= 3
                   for existing in selected):
                reference = selected[0] if selected else candidate
                clearance_delta = clearance(candidate) - clearance(reference)
                coverage_delta = target_coverage(candidate) - target_coverage(reference)
                if objective == 1 and not clearance_delta > 1e-6:
                    label = "Distinct entry alternative"
                if objective == 2 and not coverage_delta > 1e-6:
                    label = "Distinct target-axis alternative"
                candidate.update(name=label, origin="bounded-native-entry-patch", tradeoff={
                    "selectionReason": label, "lengthMm": candidate["measurements"]["lengthMm"],
                    "minimumAvailableClearanceLowerBoundMm": clearance(candidate) if np.isfinite(clearance(candidate)) else None,
                    "targetVoxelCenterCoverageMm3": target_coverage(candidate),
                    "lengthDeltaFromFirstMm": candidate["measurements"]["lengthMm"] - reference["measurements"]["lengthMm"],
                    "minimumClearanceDeltaFromFirstMm": clearance_delta if np.isfinite(clearance_delta) else None,
                    "targetCoverageDeltaFromFirstMm3": coverage_delta,
                    "meaning": "lexicographic sampled alternatives; no universal safety score or clinical ranking"})
                selected.append(candidate)
                break
    report = {"enumerated": len(positions), "evaluated": evaluated, "accepted": len(selected), "eligible": len(accepted),
              "returned": len(selected), "rejected": sum(reasons.values()), "reasons": reasons,
              "selectionRule": "shorter length, then greater minimum available labelled clearance, then target-centre coverage; >=4 mm entry and >=3 degree axis separation",
              "scope": "selected entry patch only; entry must be at or superior to target; known cerebellar/brainstem masks excluded",
              "whyFewer": None if len(selected) == 3 else "Fewer than three distinct alternatives satisfy the sampled patch, source coverage and declared geometric constraints.",
              "clinicalRanking": False}
    return selected, report


def dispatch_plan(collection, args, *, geometry=None, commit=lambda: None):
    action = args.get("action", "list")
    saved = read_approaches(collection)
    report, selected_id = None, None
    if action == "suggest":
        proposals, report = suggest_approaches(collection, args, geometry)
        # Refresh the bounded proposal group while preserving manual/user edits.
        saved = [a for a in saved if a.get("origin") != "bounded-native-entry-patch"]
        saved.extend(proposals)
        selected_id = proposals[0]["id"] if proposals else None
    elif action in {"upsert", "assess"}:
        value = args.get("approach")
        if value is None and action == "assess":
            value = next((a for a in saved if a["id"] == args.get("approachId")), None)
            if value is None:
                raise ValueError("Unknown saved approach")
        if value is None:
            value = args
        approach = checked_approach(collection, value)
        if args.get("assess", True):
            approach = assess_approach(collection, approach, geometry)
        else:
            approach.update(sourceSnapshotId=collection.identity(), parameterHash=parameter_hash(approach), assessmentStatus="unassessed")
        approach["origin"] = "manual" if action == "upsert" else value.get("origin", "manual")
        if action == "assess" and value.get("tradeoff"):
            approach["tradeoff"] = value["tradeoff"]
        selected_id = approach["id"]
        position = next((index for index, item in enumerate(saved) if item["id"] == selected_id), None)
        if position is None:
            saved.append(approach)
        else:
            saved[position] = approach
    elif action == "delete":
        key = args.get("approachId")
        if not any(a["id"] == key for a in saved):
            raise ValueError("Unknown saved approach")
        saved = [a for a in saved if a["id"] != key]
    elif action != "list":
        raise ValueError("Unknown native approach action")
    if action != "list":
        save_approaches(collection, saved, commit=commit)
    return {"snapshotId": collection.identity(), "approaches": saved, "selectedId": selected_id,
            "proposalReport": report, "simulationExecutionAuthorized": False}
