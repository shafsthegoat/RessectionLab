"""Saved native source-bound approach geometry, separate from simulator admission.

Source arrays stay on their original grids. Proposals reuse the static capsule
cell supercover and never call physical planning or authorize tool execution.
"""
from copy import deepcopy
import fcntl
import hashlib
import json
import os
import re
import tempfile
import uuid

import numpy as np
from scipy.spatial import cKDTree

from .patient_anatomy import PatientAnatomySnapshot, PatientImagingCollection, affine, digest, world
from .geometry import (GEOMETRY_VERSION, GeometryScene, native_backend,
                       capsule_voxel_indices, point_segment_distances)
from .native_approach_comparison import (sensitivity_config, sample_positions, nominal_objectives,
                                        order_candidates, distinct_alternatives, freeze_scenarios,
                                        assess_sensitivity, candidate_freeze, source_readings,
                                        SOURCE_READING_FIELDS, RICH_INPUTS, RESTRICTED_INPUTS)

VASCULAR_KINDS = {"vessels", "artery", "vein", "sinus"}
TRACT_KINDS = {"tract", "tractography", "motor", "language", "functional"}
SCHEMA = "patient-anatomy-approaches-v1"
FILENAME = "anatomy-approaches.json"
MAX_SAVED = 32
MAX_APPROACH_METADATA_BYTES = 8 * 1024**2
ASSESSMENT_VERSION = "native-source-capsule-assessment-v3"


def _forbidden_compartment(row):
    identity = (row.get("label", "") + " " + row["id"]).lower()
    return any(word in identity for word in ("cerebell", "brainstem", "brain-stem", "brain stem", "spinal cord", "medulla", "pons"))


def _restricted_layer(row, reference_id, target_id):
    # Exact kinds and a bounded T1 compartment identification. Unknown kinds,
    # extra target masks, all vascular/tract layers and all non-reference series
    # are excluded from the restricted policy's input collection.
    kind = row["kind"]
    return row["source_series"] == reference_id and (
        row["id"] == target_id and row["role"] == "target" and kind not in VASCULAR_KINDS | TRACT_KINDS
        or row["role"] == "context" and (row["id"], kind) in {("brain", "brain"), ("head-surface", "context")}
        or row["role"] in {"context", "critical"} and (
            (row["id"] in {"head-7", "head-8", "head-16", "head-46", "head-47"} and kind == "brain-region")
            or kind in {"brainstem", "cerebellum", "cerebellar-white-matter", "cerebellar-cortex"}
        ) and _forbidden_compartment(row)
    )


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


def points_in_field(a, shape, points):
    voxel = world(np.linalg.inv(a), points)
    return np.all((voxel >= -.5) & (voxel <= np.asarray(shape) - .5), axis=1)


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
    def __init__(self, snapshot, *, check=lambda: None, cache=None):
        snapshot._check()
        self.snapshot, self.c, self.check = snapshot, snapshot.collection, check
        document = self.c.document
        self.reference = self.c.series(document["reference_series"])
        self.layers = tuple(document["layers"])
        self.tract_rows = tuple(document.get("tractography", []))
        self.missing = tuple(document.get("missing", []))
        if cache is None:
            self.grids, self.masks, self.populations, self.centers, self.tracts, self.affines = {}, {}, {}, {}, {}, {}
        else:
            self.grids, self.masks, self.populations = cache.grids, cache.masks, cache.populations
            self.centers, self.tracts, self.affines = cache.centers, {}, {}

    def restricted(self, target_id):
        document = self.c.document
        reference_id = self.reference["id"]
        layers = [row for row in self.layers if _restricted_layer(row, reference_id, target_id)]
        projected = PatientImagingCollection(self.c.root, {
            "schema": document["schema"], "patient_id": document["patient_id"], "reference_series": reference_id,
            "series": [self.reference], "layers": layers,
            "transforms": {reference_id: document["transforms"][reference_id]},
        })
        return NativeRouteGeometry(PatientAnatomySnapshot.capture(projected), check=self.check, cache=self)

    def registration_groups(self, target_id, excluded):
        relevant = [row for row in self.layers if (row["id"] in excluded or _forbidden_compartment(row)
                    or row["role"] == "critical" or row["kind"] in VASCULAR_KINDS | TRACT_KINDS | {"brain"}
                    or (row["role"] == "target" and (not target_id or row["id"] == target_id)))]
        transforms = self.c.document["transforms"]
        groups = {}
        for sid in sorted({row["source_series"] for row in (*relevant, *self.tract_rows)} - {self.reference["id"]}):
            root = sid
            seen = {root}
            parent = transforms.get(root, {}).get("parent_series")
            while parent and parent != self.reference["id"] and parent not in seen:
                root = parent
                seen.add(root)
                parent = transforms.get(root, {}).get("parent_series")
            groups.setdefault(root, []).append(sid)
        return groups

    def patient_affine(self, row):
        if row["id"] not in self.affines:
            self.affines[row["id"]] = self.c.patient_affine(row)
        return self.affines[row["id"]]

    def grid(self, row):
        a = self.patient_affine(row)
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

    def population(self, row):
        key = (row["data"]["sha256"], row["label_value"])
        if key not in self.populations:
            self.populations[key] = int(np.count_nonzero(self.mask(row)))
        return self.populations[key]

    def fully_inside_target(self, scene, indices, target, relative_shift=None):
        """Prove whole brain cells lie in supplied target cells before exclusion.

        A target-grid AABB encloses each transformed brain parallelepiped. Every
        target cell touching that box must be labelled, including boundary cells.
        This can retain extra brain cells; it cannot remove an unproved cell.
        """
        target_affine = self.patient_affine(target)
        inverse = np.linalg.inv(target_affine)
        relative = inverse @ scene.affine
        if relative_shift is not None:
            relative[:3, 3] += inverse[:3, :3] @ relative_shift
        mask = self.mask(target)
        if (np.array_equal(target_affine, scene.affine) and mask.shape == scene.shape
                and (relative_shift is None or not np.any(relative_shift))):
            return mask[tuple(indices.T)]
        centers = world(relative, indices)
        extent = .5 * np.abs(relative[:3, :3]).sum(axis=1)
        lo = np.floor(centers - extent + .5).astype(int)
        hi = np.floor(centers + extent + .5).astype(int)
        inside = np.all((lo >= 0) & (hi < np.asarray(mask.shape)), axis=1)
        bounded = np.prod(hi - lo + 1, axis=1) <= 4096
        contained = np.zeros(len(indices), bool)
        for index in np.flatnonzero(inside & bounded):
            if index % 256 == 0:
                self.check()
            low, high = lo[index], hi[index] + 1
            contained[index] = bool(mask[low[0]:high[0], low[1]:high[1], low[2]:high[2]].all())
        return contained

    def measure(self, start, end, radius_mm, constraints=None, *, target_id=None, scenario=None, memo=None):
        self.snapshot._check()
        start, end, radius_mm, length = checked_route(start, end, radius_mm)
        constraints = constraints or {}
        scenario = scenario or {}
        memo = {} if memo is None else memo
        shifts = {key: np.asarray(value, dtype=float) for key, value in scenario.get("sourceTranslationsMm", {}).items()}
        boundary_mm = scenario.get("boundaryMm", 0.0)
        zero = np.zeros(3)
        excluded = set(constraints.get("excludedLayerIds", []))
        c = self.c
        ref_coverage = envelope_coverage(self.patient_affine(self.reference), self.reference["data"]["shape"], start, end, radius_mm)
        unknowns = list(self.missing)
        structures, targets, source_coverage, covers = [], [], [], {}
        selected_targets = tuple(r for r in self.layers if r["role"] == "target" and (not target_id or r["id"] == target_id))
        selected_target_ids = {r["id"] for r in selected_targets}

        def cover(row, shift, radius):
            key, scene = self.grid(row)
            key = (key, tuple(shift), radius)
            if key not in covers:
                covers[key] = capsule_voxel_indices(scene, start - shift, end - shift, radius, check=self.check)
            return scene, covers[key]

        for row in self.layers:
            self.check()
            forbidden = _forbidden_compartment(row)
            critical = row["role"] == "critical" or row["kind"] in VASCULAR_KINDS | TRACT_KINDS or row["id"] in excluded or forbidden
            is_target = row["id"] in selected_target_ids
            if not critical and not is_target:
                continue
            shift = shifts.get(row["source_series"], zero)
            probe_radius = radius_mm + boundary_mm if critical and not is_target else radius_mm
            memo_key = (row["id"], tuple(shift), probe_radius)
            cached = memo.get(memo_key)
            if cached is not None:
                found, covered, unknown, reached = cached
                if critical:
                    structures.append(found)
                source_coverage.extend(covered)
                unknowns.extend(unknown)
                targets.extend(reached)
                continue
            first_coverage, first_unknown, first_target = len(source_coverage), len(unknowns), len(targets)
            item = {"layerId": row["id"], "label": row.get("label", row["id"]), "kind": row["kind"],
                    "available": False, "confirmedIntersection": False,
                    "excluded": forbidden or row["id"] in excluded, "sourceSeries": row["source_series"],
                    "reviewStatus": row.get("review_status", "unreviewed"), "coverageMeaning": row.get("coverage", "unassessed"),
                    "sourceIdentity": {"dataSha256": row["data"]["sha256"], "labelValue": row["label_value"],
                                       "revision": row.get("revision", 0)}}
            try:
                a = self.patient_affine(row)
                source = c.series(row["source_series"])
                lo, hi = start - shift, end - shift
                field = envelope_coverage(a, row["data"]["shape"], lo, hi, probe_radius)
                acquisition = envelope_coverage(self.patient_affine(source), source["data"]["shape"], lo, hi, probe_radius)
                item["coverage"] = field
                item["acquisitionCoverage"] = acquisition
                source_coverage.append({"layerId": row["id"], **field, "acquisitionCoverage": acquisition})
                if not field["wholeEnvelopeWithinField"] or not acquisition["wholeEnvelopeWithinField"]:
                    unknowns.append(row["id"] + ": corridor envelope outside source field")
                mask = self.mask(row)
                count = self.population(row)
                if not count:
                    item["reason"] = "empty label; anatomy is unassessed"
                    unknowns.append(row["id"] + ": empty label")
                else:
                    scene, indices = cover(row, shift, probe_radius)
                    hit = indices[mask[tuple(indices.T)]]
                    orthogonal = scene._orthogonal_spacing is not None
                    center_hits = int(np.count_nonzero(point_segment_distances(world(a, hit), lo, hi) <= probe_radius))
                    key = (row["data"]["sha256"], row["label_value"], a.tobytes())
                    if key not in self.centers:
                        centers = world(a, np.argwhere(mask))
                        self.centers[key] = cKDTree(centers) if count >= 4096 else centers
                    index = self.centers[key]
                    centers = index.data if isinstance(index, cKDTree) else index
                    minimum, nearest = float("inf"), None
                    if isinstance(index, cKDTree):
                        samples = np.linspace(lo, hi, max(2, int(np.ceil(length / 3)) + 1))
                        upper = float(index.query(samples)[0].min())
                        step = length / (len(samples) - 1)
                        # Project the nearest centre onto the segment. It is
                        # perpendicular to the axis or nearest an endpoint.
                        hits = index.query_ball_point(samples, float(np.hypot(upper, step / 2)) + 1e-8)
                        centers = centers[np.unique(np.concatenate(hits)).astype(np.intp)]
                    for offset in range(0, len(centers), 131072):
                        self.check()
                        points = centers[offset:offset + 131072]
                        distances = point_segment_distances(points, lo, hi)
                        closest = int(distances.argmin())
                        if distances[closest] < minimum:
                            minimum, nearest = float(distances[closest]), (points[closest] + shift).tolist()
                    lower = minimum - scene._cell_radius_mm - probe_radius
                    item.update(available=True, clearanceLowerBoundMm=float(lower),
                                intersectsOrUnresolved=bool(len(hit)), confirmedIntersection=bool(len(hit) if orthogonal else center_hits),
                                intersectionModel="exact capsule / affine voxel cell" if orthogonal else "conservative cell circumsphere; shear",
                                conservativeOverlapVolumeUpperBoundMm3=float(len(hit) * scene.voxel_volume_mm3),
                                intersectedCellCount=int(len(hit)), nearestPoint=nearest)
                    if not orthogonal and len(hit) and not center_hits:
                        unknowns.append(row["id"] + ": conservative sheared-cell contact unresolved")
                    if is_target:
                        endpoint = np.floor(world(np.linalg.inv(a), hi) + .5).astype(int)
                        endpoint_inside = bool(np.all(endpoint >= 0) and np.all(endpoint < mask.shape) and mask[tuple(endpoint)])
                        targets.append({**item, "targetPointInsideLabel": endpoint_inside,
                                        "totalVolumeMm3": float(count * scene.voxel_volume_mm3),
                                        "coveredVoxelCenterVolumeMm3": float(center_hits * scene.voxel_volume_mm3),
                                        "coveredVoxelCenterFraction": center_hits / count,
                                        "coverageModel": "target voxel centres in entry-to-target capsule; geometric reach only, no removal"})
            except ValueError as exc:
                item["reason"] = str(exc)
                unknowns.append(row["id"] + ": " + str(exc))
            if critical:
                structures.append(item)
            if is_target and not item["available"]:
                targets.append(item)
            memo[memo_key] = (item, source_coverage[first_coverage:], unknowns[first_unknown:], targets[first_target:])

        for tract in self.tract_rows:
            self.check()
            shift = shifts.get(tract["source_series"], zero)
            memo_key = (tract["id"], tuple(shift), radius_mm)
            cached = memo.get(memo_key)
            if cached is not None:
                found, covered, unknown, _ = cached
                structures.append(found)
                source_coverage.extend(covered)
                unknowns.extend(unknown)
                continue
            first_coverage, first_unknown = len(source_coverage), len(unknowns)
            item = {"layerId": tract["id"], "label": tract.get("label", tract["id"]), "kind": "tractography",
                    "available": False, "confirmedIntersection": False, "excluded": tract["id"] in excluded,
                    "sourceSeries": tract["source_series"], "coverageMeaning": "supplied streamline paths; bundle extent and detection sensitivity unassessed",
                    "sourceIdentity": {"pointsSha256": tract["points_ras_mm"]["sha256"], "offsetsSha256": tract["offsets"]["sha256"]}}
            try:
                source = c.series(tract["source_series"])
                a = self.patient_affine(source)
                lo, hi = start - shift, end - shift
                field = envelope_coverage(a, source["data"]["shape"], lo, hi, radius_mm)
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
                    samples = np.linspace(lo, hi, max(2, int(np.ceil(length / 4)) + 1))
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
                        distances = _segment_bundle_distances(lo, hi, aa, bb)
                        index = int(distances.argmin())
                        if distances[index] < minimum:
                            minimum, nearest = float(distances[index]), ((aa[index] + bb[index]) / 2 + shift).tolist()
                        hits += int(np.count_nonzero(distances <= radius_mm))
                    item.update(available=True, clearanceLowerBoundMm=float(minimum - radius_mm),
                                intersectsOrUnresolved=bool(hits), confirmedIntersection=bool(hits),
                                intersectionModel="capsule versus supplied polyline segments; no inferred bundle thickness",
                                intersectedSegmentCount=hits, nearestPoint=nearest,
                                nearestPointMeaning="midpoint of the closest supplied segment")
            except ValueError as exc:
                item["reason"] = str(exc)
                unknowns.append(tract["id"] + ": " + str(exc))
            structures.append(item)
            memo[memo_key] = (item, source_coverage[first_coverage:], unknowns[first_unknown:], ())

        brain = next((r for r in self.layers if r["kind"] == "brain"), None)
        normal_key = ("normal", tuple(shifts.get(brain["source_series"], zero)) if brain else (),
                      tuple(tuple(shifts.get(row["source_series"], zero)) for row in selected_targets))
        if normal_key not in memo:
            normal = {"available": False, "reason": "no supplied brain mask; normal tissue traversal unassessed"}
            if brain:
                normal["layerId"] = brain["id"]
                try:
                    if not self.population(brain):
                        raise ValueError("empty brain mask; normal tissue traversal unassessed")
                    shift = shifts.get(brain["source_series"], zero)
                    scene, indices = cover(brain, shift, radius_mm)
                    indices = indices[self.mask(brain)[tuple(indices.T)]]
                    selected = np.ones(len(indices), dtype=bool)
                    points = scene.voxel_to_world(indices)
                    source = c.series(brain["source_series"])
                    within_radius = point_segment_distances(points, start - shift, end - shift) <= radius_mm
                    known = within_radius & points_in_field(self.patient_affine(source), source["data"]["shape"], points)
                    nontarget = within_radius.copy()
                    classifiable = bool(selected_targets)
                    for target in selected_targets:
                        target_shift = shifts.get(target["source_series"], zero)
                        try:
                            mask = self.mask(target)
                            if not self.population(target):
                                raise ValueError("empty target label")
                            selected &= ~self.fully_inside_target(scene, indices, target, shift - target_shift)
                            source_target = c.series(target["source_series"])
                            relative_points = points + shift - target_shift
                            points_in_target_grid = world(np.linalg.inv(self.patient_affine(target)), relative_points)
                            cells = np.floor(points_in_target_grid + .5).astype(int)
                            in_grid = np.all((cells >= 0) & (cells < np.asarray(mask.shape)), axis=1)
                            known &= in_grid & points_in_field(self.patient_affine(source_target), source_target["data"]["shape"], relative_points)
                            nontarget[in_grid] &= ~mask[tuple(cells[in_grid].T)]
                        except ValueError:
                            # Preserve the whole-cell bound when uncertain target
                            # classification prevents exclusion or centre sampling.
                            classifiable = False
                    if not classifiable:
                        known[:] = False
                    centre_volume = float((known & nontarget).sum() * scene.voxel_volume_mm3) if classifiable and (known.any() or not within_radius.any()) else None
                    normal = {"available": True, "layerId": brain["id"],
                              "conservativeVolumeUpperBoundMm3": float(selected.sum() * scene.voxel_volume_mm3),
                              "supportedNonTargetVoxelCenterVolumeMm3": centre_volume,
                              "unclassifiedBrainVoxelCenterCount": int((within_radius & ~known).sum()),
                              "coverage": envelope_coverage(scene.affine, scene.shape, start - shift, end - shift, radius_mm),
                              "acquisitionCoverage": envelope_coverage(self.patient_affine(source), source["data"]["shape"], start - shift, end - shift, radius_mm),
                              "meaning": "Whole-cell upper bound in supplied brain mask; the centre-sampled non-target quantity requires known target classification. Neither quantity measures tissue removal.",
                              "targetExclusionModel": "Uncertain and partly labelled cells remain in the upper bound. Subtract only complete transformed brain cells proved inside a selected target."}
                except ValueError as exc:
                    normal["reason"] = str(exc)
            memo[normal_key] = normal
        normal = memo[normal_key]
        if normal["available"] and (not normal["coverage"]["wholeEnvelopeWithinField"]
                or not normal["acquisitionCoverage"]["wholeEnvelopeWithinField"]):
            unknowns.append(brain["id"] + ": normal traversal outside supplied brain mask or acquisition field")
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
            if item["layerId"] in excluded and (not item["available"] or
                    not item.get("coverage", {}).get("wholeEnvelopeWithinField") or
                    not item.get("acquisitionCoverage", {}).get("wholeEnvelopeWithinField", True)):
                failures.append(item["label"] + ": explicit exclusion is unassessed over part or all of the capsule")
            if item.get("available") and item["clearanceLowerBoundMm"] < constraints.get("minClearanceMm", 0):
                failures.append(item["label"] + ": conservative clearance below declared minimum")
        if target_id and not any(t.get("targetPointInsideLabel") for t in targets):
            failures.append("endpoint is outside or unassessed in the selected target label")
        axis = (end - start) / length
        return {"geometryVersion": GEOMETRY_VERSION, "assessmentVersion": ASSESSMENT_VERSION, "lengthMm": length, "radiusMm": radius_mm,
                "entry": start.tolist(), "target": end.tolist(), "direction": axis.tolist(),
                "capsule": {"start": start.tolist(), "end": end.tolist(), "radiusMm": radius_mm,
                            "lengthMm": length, "volumeMm3": float(np.pi * radius_mm**2 * (length + 4 * radius_mm / 3)),
                            "frame": "patient RAS+ mm", "endCaps": "spherical"},
                "targetLayerId": target_id, "simulatedRemovedTargetVolumeMm3": None,
                "orientation": {"azimuthDeg": float(np.degrees(np.arctan2(axis[1], axis[0]))),
                                "elevationDeg": float(np.degrees(np.arcsin(np.clip(axis[2], -1, 1)))),
                                "frame": "patient RAS+; entry to target"},
                "structures": structures, "targets": targets, "normalTraversal": normal,
                "coverage": {"withinReferenceEnvelope": ref_coverage["wholeEnvelopeWithinField"],
                             "reference": ref_coverage, "sources": source_coverage,
                             "allAvailableSourceFieldsCoverEnvelope": bool(source_coverage) and all(
                                 s["wholeEnvelopeWithinField"] and s.get("acquisitionCoverage", {}).get("wholeEnvelopeWithinField", True)
                                 for s in source_coverage)},
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
    normal_norm = np.linalg.norm(normal)
    if abs(normal_norm - 1.) > 1e-14:
        normal = normal / normal_norm
    patch = {"center": center.tolist(), "normal": normal.tolist(), "radiusMm": float(patch_radius),
             "meaning": "user-selected hypothetical supratentorial entry-centre disk; cortical/skull suitability unverified"}
    selected = value.get("targetLayerId")
    if selected is not None and (not isinstance(selected, str) or
            not any(row["id"] == selected and row["role"] == "target" for row in collection.document["layers"])):
        raise ValueError("Unknown selected target label")
    result = {"id": key, "name": name.strip(), "entry": start.tolist(), "target": end.tolist(),
            "radiusMm": radius, "direction": ((end - start) / length).tolist(),
            "entryPatch": patch, "constraints": constraints}
    if selected is not None:
        result["targetLayerId"] = selected
    if "sensitivityConfig" in value:
        result["sensitivityConfig"] = sensitivity_config(value["sensitivityConfig"])
    return result


def parameter_hash(approach):
    return digest({key: approach[key] for key in ("entry", "target", "radiusMm", "entryPatch", "constraints", "targetLayerId", "sensitivityConfig")
                   if key in approach})


def _read_plan(collection):
    path = collection.root / FILENAME
    if not path.exists():
        return [], {}, None, None
    if path.stat().st_size > MAX_APPROACH_METADATA_BYTES:
        raise ValueError("Saved comparison exceeds metadata budget")
    encoded = path.read_bytes()
    data = json.loads(encoded)
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
        target_id = candidate.get("targetLayerId")
        retired_target = target_id is not None and not any(row["id"] == target_id and row["role"] == "target" for row in collection.document["layers"])
        if retired_target:
            candidate.pop("targetLayerId")
        checked = checked_approach(collection, candidate)
        if retired_target:
            checked["targetLayerId"] = target_id
        checked["constraints"] = original_constraints
        if checked["id"] in identities:
            raise ValueError("Duplicate saved approach")
        identities.add(checked["id"])
        checked.update({key: deepcopy(saved[key]) for key in ("sourceSnapshotId", "parameterHash", "measurementHash", "measurements", "origin", "tradeoff") if key in saved})
        valid = (not retired and not retired_target and saved.get("sourceSnapshotId") == collection.identity()
                 and saved.get("parameterHash") == parameter_hash(checked)
                 and saved.get("measurements", {}).get("geometryVersion") == GEOMETRY_VERSION
                 and saved.get("measurements", {}).get("assessmentVersion") == ASSESSMENT_VERSION
                 and saved.get("measurementHash") == digest(saved.get("measurements")))
        checked["assessmentStatus"] = "current" if valid else "stale" if saved.get("measurements") else "unassessed"
        if not valid:
            checked["invalidationReason"] = "Source anatomy, geometry engine or approach parameters changed; recompute before comparing."
        result.append(checked)
    comparisons = data.get("comparisons", {})
    if not isinstance(comparisons, dict) or len(comparisons) > MAX_SAVED:
        raise ValueError("Invalid saved target comparisons")
    reports = {}
    for target, pair in comparisons.items():
        if not isinstance(pair, dict):
            raise ValueError("Invalid saved target comparison")
        reports[target] = {key: _read_report(collection, entry) for key, entry in pair.items()
                           if key in {"proposalReport", "inputComparison"}}
    active = data.get("activeTargetLayerId")
    if active is not None and not isinstance(active, str):
        raise ValueError("Invalid saved selected target")
    return result, reports, active, hashlib.sha256(encoded).hexdigest()


def read_approaches(collection):
    return _read_plan(collection)[0]


def _read_report(collection, entry):
    if entry is None:
        return None
    if not isinstance(entry, dict) or not isinstance(entry.get("report"), dict) or digest(entry["report"]) != entry.get("hash"):
        raise ValueError("Saved native comparison report integrity differs")
    report = entry["report"]
    if report.get("sourceSnapshotId") != collection.identity() or report.get("assessmentVersion") != ASSESSMENT_VERSION:
        report = {**report, "assessmentStatus": "stale", "invalidationReason": "Source anatomy or assessment engine changed; regenerate the comparison."}
    return report


def _bound_report(collection, report):
    return {**report, "sourceSnapshotId": collection.identity(), "assessmentVersion": ASSESSMENT_VERSION,
            "assessmentStatus": "current", "geometryKernels": {
                "capsuleVoxelSupercover": "compiled-cpp" if native_backend.cpp_enabled() else "reference-python",
                "sourceLabelAndStreamlineProximity": "numpy-scipy"}}


def save_approaches(collection, approaches, *, comparisons=None, target_id=None, expected_revision=..., commit=lambda: None):
    if len(approaches) > MAX_SAVED:
        raise ValueError("At most 32 saved approaches are supported")
    if comparisons and len(comparisons) > MAX_SAVED:
        raise ValueError("At most 32 saved target comparisons are supported")
    data = {"schema": SCHEMA, "patientId": collection.document["patient_id"], "approaches": approaches}
    if comparisons:
        data["comparisons"] = {target: {key: {"report": report, "hash": digest(report)} for key, report in pair.items()
                                         if report is not None} for target, pair in comparisons.items()}
    if target_id is not None:
        data["activeTargetLayerId"] = target_id
    encoded = json.dumps(data, allow_nan=False, separators=(",", ":")).encode("utf-8")
    if len(encoded) > MAX_APPROACH_METADATA_BYTES:
        raise ValueError("Saved comparison exceeds metadata budget")
    with (collection.root / ".write.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        path = collection.root / FILENAME
        revision = hashlib.sha256(path.read_bytes()).hexdigest() if path.exists() else None
        if expected_revision is not ... and expected_revision != revision:
            raise ValueError("Saved comparisons changed while evaluating; reopen and retry")
        current = json.loads((collection.root / "patient-anatomy.json").read_text())
        if current.get("snapshot_id") != collection.identity():
            raise ValueError("Patient changed; reload before saving approaches")
        commit()
        with tempfile.NamedTemporaryFile(dir=collection.root, mode="wb", delete=False) as out:
            out.write(encoded)
            temporary = out.name
        os.replace(temporary, path)


def _constraints_for_geometry(geometry, constraints):
    visible = {row["id"] for row in (*geometry.layers, *geometry.tract_rows)}
    return {**constraints, "excludedLayerIds": [key for key in constraints["excludedLayerIds"] if key in visible]}


def _measure_approach(geometry, approach, scenario=None, *, memo=None):
    measured = geometry.measure(approach["entry"], approach["target"], approach["radiusMm"], approach["constraints"],
                                target_id=approach.get("targetLayerId"), scenario=scenario, memo=memo)
    failures = measured["constraintFailures"]
    if approach["entry"][2] < approach["target"][2]:
        failures.append("entry inferior to target; outside the declared superior/lateral supratentorial proposal scope")
    patch = approach["entryPatch"]
    delta = np.asarray(approach["entry"]) - patch["center"]
    normal = np.asarray(patch["normal"])
    plane_distance = float(delta @ normal)
    radial_distance = float(np.linalg.norm(delta - plane_distance * normal))
    if abs(plane_distance) > 1e-5 or radial_distance > patch["radiusMm"] + 1e-5:
        failures.append("entry outside declared entry-centre disk; move the patch or choose an entry within its plane and radius")
    if failures:
        measured["status"] = "known-conflict"
    return measured


def _scenario_set(geometry, approach):
    config = sensitivity_config(approach.get("sensitivityConfig"))
    groups = geometry.registration_groups(approach.get("targetLayerId"), approach["constraints"]["excludedLayerIds"])
    return freeze_scenarios(config, groups)


def _record_measurements(collection, approach, measured):
    return {**approach, "sourceSnapshotId": collection.identity(), "parameterHash": parameter_hash(approach),
            "assessmentStatus": "current", "measurements": measured, "measurementHash": digest(measured)}


def assess_approach(collection, value, geometry=None):
    approach = checked_approach(collection, value)
    geometry = geometry or NativeRouteGeometry(PatientAnatomySnapshot.capture(collection))
    geometry.snapshot._check()
    memo = {}
    measured = _measure_approach(geometry, approach, memo=memo)
    scenarios = _scenario_set(geometry, approach)
    measured["sensitivity"] = assess_sensitivity(measured, scenarios, lambda scenario: _measure_approach(geometry, approach, scenario, memo=memo), check=geometry.check)
    measured["sensitivity"]["scenarioSet"] = scenarios
    measured["scenarioGuard"] = _scenario_guard(measured["sensitivity"])
    return _record_measurements(collection, approach, measured)


def _base_for_suggestions(collection, args):
    value = {**args, "entry": args.get("entry", args.get("start")),
             "target": args.get("target", args.get("end"))}
    targets = [row["id"] for row in collection.document["layers"] if row["role"] == "target"]
    if len(targets) == 1 and value.get("targetLayerId") is None:
        value["targetLayerId"] = targets[0]
    value["sensitivityConfig"] = sensitivity_config(args.get("sensitivityConfig", args.get("sensitivity")))
    base = checked_approach(collection, value)
    if "patchRadiusMm" in args:
        base["entryPatch"]["radiusMm"] = args["patchRadiusMm"]
        base = checked_approach(collection, base)
    return base


def _geometry_fields(measured):
    reasons = list(measured["constraintFailures"])
    if not measured["coverage"]["withinReferenceEnvelope"]:
        reasons.append("whole corridor outside reference image field")
    objectives, evidence = nominal_objectives(measured)
    return reasons, objectives, evidence


def _generate_nominal(geometry, base):
    """Freeze proposals using only this arm's own anatomy projection."""
    geometry.snapshot._check()
    base = {**base, "constraints": _constraints_for_geometry(geometry, base["constraints"])}
    rows, evaluated, eligible = [], {}, []
    target = np.asarray(base["target"])
    for index, entry in enumerate(sample_positions(base["entryPatch"])):
        geometry.check()
        axis = target - entry
        length = float(np.linalg.norm(axis))
        parameters = {"entry": entry.tolist(), "target": base["target"], "radiusMm": base["radiusMm"]}
        # ID and tie-breaking inputs contain no withheld image identities.
        key = "proposal-" + digest([parameters, base.get("targetLayerId")])[:16]
        row = {"id": key, "sampleIndex": index, **parameters,
               "direction": (axis / length).tolist() if length > 1e-6 else None,
               "disposition": "rejected", "reasons": [], "objectives": {"lengthMm": length},
               "evidence": {}, "pareto": None, "dominatorCount": None, "nominalEligible": False}
        rows.append(row)
        if length <= 1e-6:
            row["reasons"].append("entry and target coincide")
        elif length > min(base["constraints"].get("maxLengthMm", 1000), 1000):
            row["reasons"].append("corridor exceeds declared maximum length")
        elif entry[2] < target[2]:
            row["reasons"].append("inferior entries excluded by supratentorial scope")
        if row["reasons"]:
            continue
        route = {**base, **parameters, "id": key, "direction": row["direction"]}
        nominal = _measure_approach(geometry, route)
        evaluated[key] = (route, nominal)
        row["reasons"], row["objectives"], row["evidence"] = _geometry_fields(nominal)
        row["sourceReadings"] = source_readings(nominal)
        row["detectedConflictLayerIds"] = nominal["detectedConflictLayerIds"]
        if not row["reasons"]:
            row["disposition"] = "eligible"
            row["nominalEligible"] = True
            eligible.append(row)
    ordered = order_candidates(eligible)
    chosen = distinct_alternatives([row for row in ordered if row["pareto"]])
    if len(chosen) < 3:
        chosen = distinct_alternatives(ordered)
    for row in chosen:
        row["disposition"] = "selected"
    return rows, evaluated, ordered, chosen


_SELECTION_RULE = ("Guard declared constraints and reference field first. Prefer equal-evidence Pareto candidates and supported objective extremes, "
                   "then compare a frozen shortlist after declared scenario guards. Retain up to three entries separated by at least 4 mm and axes separated by at least 3 degrees. "
                   "Dominated routes may fill a distinct entry when the front alone cannot support three. "
                   "Break remaining ties by evidence coverage, shorter length and sampled order.")
_SCOPE = "Selected entry patch; inferior entries excluded; supplied cerebellar and brainstem labels guarded. Static source geometry only."


def _task_fields(collection, base):
    return {"patientId": collection.document["patient_id"], "targetLayerId": base.get("targetLayerId"),
            "entry": base["entry"], "entryPatch": base["entryPatch"], "target": base["target"], "radiusMm": base["radiusMm"],
            "constraints": base["constraints"], "sensitivityConfig": base["sensitivityConfig"]}


def _measure_frozen(geometry, route, nominal, scenarios):
    memo = {}
    return assess_sensitivity(nominal, scenarios, lambda scenario: _measure_approach(geometry, route, scenario, memo=memo), check=geometry.check)


def _scenario_guard(sensitivity):
    """Keep a known hard veto distinct from incomplete source support."""
    vetoes = [row["id"] for row in sensitivity["scenarios"] if row["constraintFailures"]]
    incomplete = [row["id"] for row in sensitivity["scenarios"] if row["missingMetrics"] or row["unknownSourceLayerIds"] or row["status"] == "unassessed"]
    disposition = "vetoed" if vetoes else "incomplete-coverage" if incomplete else "no-detected-hard-violation"
    return {"disposition": disposition, "vetoScenarioIds": vetoes, "incompleteScenarioIds": incomplete,
            "meaning": "Guard disposition in declared scenarios. Unknown anatomy and untested scenarios remain unassessed."}


def suggest_approaches(collection, args, geometry=None):
    """Enumerate the declared patch, then test frozen finalists on shared scenarios."""
    base = _base_for_suggestions(collection, args)
    geometry = geometry or NativeRouteGeometry(PatientAnatomySnapshot.capture(collection))
    rows, evaluated, ordered, nominally_selected = _generate_nominal(geometry, base)
    finalists = [*nominally_selected, *(row for row in ordered if row["disposition"] == "eligible")][:8]
    for row in ordered:
        row["nominalPareto"] = row["pareto"]
        row["disposition"] = "eligible"
    scenarios = _scenario_set(geometry, base)
    frozen = candidate_freeze(finalists, scenarios, {"sourceSnapshotId": collection.identity(), "task": _task_fields(collection, base)})
    results, assessed, viable = {}, {}, []
    for row in finalists:
        route, nominal = evaluated[row["id"]]
        sensitivity = _measure_frozen(geometry, route, nominal, scenarios)
        guard = _scenario_guard(sensitivity)
        row["scenarioGuard"] = guard
        results[row["id"]] = {"metrics": sensitivity["metrics"],
                               "violatedScenarioIds": guard["vetoScenarioIds"], "guard": guard,
                               "unassessedScenarioIds": [s["id"] for s in sensitivity["scenarios"] if s["status"] == "unassessed"],
                               "scenarios": [{key: s[key] for key in ("id", "status", "conflictLayerIds", "constraintFailures", "unknownSourceLayerIds", "missingMetrics")}
                                             for s in sensitivity["scenarios"]]}
        if guard["vetoScenarioIds"]:
            row["disposition"] = "scenario-vetoed"
            row["reasons"].append(f'Declared hard guard failed in {len(guard["vetoScenarioIds"])} sensitivity scenarios')
        else:
            row["disposition"] = "finalist"
            viable.append(row)
            assessed[row["id"]] = (route, nominal, sensitivity)
    survivors = order_candidates(viable)
    selected = distinct_alternatives([row for row in survivors if row["pareto"]])
    if len(selected) < 3:
        selected = distinct_alternatives(survivors)
    proposals = []
    for row in selected:
        row["disposition"] = "selected"
        route, nominal, sensitivity = assessed[row["id"]]
        values = row["objectives"]
        measured_clearances = [structure["clearanceLowerBoundMm"] for structure in nominal["structures"] if structure["available"]]
        label = "Distinct Pareto alternative" if row["pareto"] else "Distinct entry alternative"
        tradeoff = {"selectionReason": label, "lengthMm": values["lengthMm"], "targetVoxelCenterCoverageMm3": values["targetReachMm3"],
                    "minimumAvailableClearanceLowerBoundMm": min(measured_clearances, default=None), "objectives": values,
                    "meaning": "Separate static geometry objectives for supplied anatomy after declared stress guards."}
        proposals.append(_record_measurements(collection, {**route, "name": label, "origin": "bounded-native-entry-patch", "tradeoff": tradeoff},
                                             {**nominal, "sensitivity": sensitivity, "scenarioGuard": row["scenarioGuard"]}))
    reasons = {}
    for row in rows:
        for reason in row["reasons"]:
            reasons[reason] = reasons.get(reason, 0) + 1
    count = len(proposals)
    vetoed = len(finalists) - len(viable)
    if not ordered:
        shortfall = "No sampled corridor met the declared scope, source-field and geometry guards."
    elif count < 3:
        shortfall = (f"Fewer than three corridors in the {len(finalists)} frozen finalists meet the declared scenario guards and 4 mm entry / 3° axis separation. "
                     f"Scenario guards vetoed {vetoed}; {len(ordered) - len(finalists)} other nominally eligible routes were outside the sensitivity budget.")
    else:
        shortfall = None
    report = {"enumerated": len(rows), "evaluated": len(evaluated), "accepted": count, "eligible": len(ordered),
              "nominalEligible": len(ordered), "nominalSelectedCandidateIds": [row["id"] for row in nominally_selected],
              "scenarioVetoCount": vetoed, "scenarioGuardEligible": len(viable),
              "returned": count, "rejected": len(rows) - len(ordered) + vetoed, "reasons": reasons, "candidates": rows,
              "selectedCandidateIds": [row["id"] for row in selected], "scenarioSet": scenarios, "candidateFreeze": frozen,
              "task": frozen["binding"]["task"],
              "sourceReadingFields": SOURCE_READING_FIELDS,
              "finalistOutcomes": results, "selectionRule": _SELECTION_RULE, "scope": _SCOPE,
              "whyFewer": shortfall, "clinicalRanking": False}
    return proposals, report


def compare_inputs(collection, args, geometry=None):
    """Nominate with two input sets, then apply one source-reference evaluator."""
    base = _base_for_suggestions(collection, args)
    geometry = geometry or NativeRouteGeometry(PatientAnatomySnapshot.capture(collection))
    selected_target = base.get("targetLayerId")
    if not selected_target:
        raise ValueError("Select a labelled target for the paired input comparison")
    partial = geometry.restricted(selected_target)
    if not any(row["id"] == selected_target for row in partial.layers):
        raise ValueError("The paired input task requires a reference-bound selected target")
    # The restricted policy never sees reference outcomes or even a rich
    # shortlist. Both frozen nominations are complete before reference scoring.
    arms, nominated_routes, union = {}, {}, {}
    for name, arm_geometry, whitelist in (("restricted", partial, RESTRICTED_INPUTS), ("rich", geometry, RICH_INPUTS)):
        rows, evaluated, _, chosen = _generate_nominal(arm_geometry, base)
        nominated_routes[name] = [evaluated[row["id"]][0] for row in chosen]
        union.update({row["id"]: row for row in chosen})
        arms[name] = {"inputWhitelist": whitelist, "layerIds": [row["id"] for row in arm_geometry.layers],
                      "sourceSeriesIds": [row["id"] for row in arm_geometry.c.document["series"]],
                      "tractIds": [row["id"] for row in arm_geometry.tract_rows],
                      "selectedCandidateIds": [row["id"] for row in chosen], "candidates": rows}
    scenarios = _scenario_set(geometry, base)
    task = _task_fields(collection, base)
    frozen = candidate_freeze(list(union.values()), scenarios, {"sourceSnapshotId": collection.identity(), "task": task,
                             "nominations": {key: arm["selectedCandidateIds"] for key, arm in arms.items()}})
    outcomes = {}
    for approach in (route for routes in nominated_routes.values() for route in routes):
        if approach["id"] in outcomes:
            continue
        route = {**approach, "constraints": base["constraints"]}
        nominal = _measure_approach(geometry, route)
        reasons, objectives, _ = _geometry_fields(nominal)
        sensitivity = _measure_frozen(geometry, route, nominal, scenarios)
        guard = _scenario_guard(sensitivity)
        nominally_eligible = not reasons
        perturbed_vetoes = [sid for sid in guard["vetoScenarioIds"] if sid != "nominal"]
        if perturbed_vetoes:
            reasons.append(f'Declared hard guard failed in {len(perturbed_vetoes)} perturbation scenarios')
        outcomes[route["id"]] = {"candidateId": route["id"], "status": "refused" if guard["vetoScenarioIds"] else "eligible-static",
                                "reasons": reasons, "objectives": objectives,
                                "nominalEligible": nominally_eligible, "scenarioGuard": guard,
                                "detectedConflictLayerIds": nominal["detectedConflictLayerIds"],
                                "sensitivity": sensitivity}
    for arm in arms.values():
        arm["nominated"] = [outcomes[key] for key in arm["selectedCandidateIds"]]
        arm["refusalCount"] = sum(row["status"] == "refused" for row in arm["nominated"])
        arm["abstentionCount"] = 3 - len(arm["nominated"])
    return {"task": task, "candidateBudget": len(arms["rich"]["candidates"]), "scenarioSet": scenarios, "candidateFreeze": frozen,
            "arms": arms, "sourceReadingFields": SOURCE_READING_FIELDS,
            "guardProtection": "Both frozen nomination sets are assessed once against the rich source estimate with identical nominal and declared sensitivity guards. Each hard violation vetoes the nominated corridor. Incomplete source support is reported separately. Evaluator-only anatomy can veto a restricted nomination. No replacement or privileged retry. Existing simulator admission limits apply to both arms.",
            "referenceMeaning": "Richer source geometry is an imperfect development reference with explicit coverage and unmeasured sensitivity.",
            "selectionRule": _SELECTION_RULE, "nominationBudget": 3, "simulationExecutionAuthorized": False}


def dispatch_plan(collection, args, *, geometry=None, commit=lambda: None):
    action = args.get("action", "list")
    saved, comparisons, active, revision = _read_plan(collection)
    selected_id = None
    if action == "suggest":
        proposals, report = suggest_approaches(collection, args, geometry)
        active = report["task"]["targetLayerId"]
        comparisons.setdefault(active or "", {})["proposalReport"] = _bound_report(collection, report)
        # Refresh only this task's bounded proposal group. Edited candidates
        # may keep their original IDs until the matching proposal is regenerated.
        saved = [a for a in saved if a.get("origin") != "bounded-native-entry-patch" or a.get("targetLayerId") != active]
        generated_ids = {row["id"] for row in proposals}
        saved = [{**row, "id": "manual-" + uuid.uuid4().hex[:16]} if row["id"] in generated_ids else row for row in saved]
        saved.extend(proposals)
        selected_id = proposals[0]["id"] if proposals else None
    elif action == "compareInputs":
        input_comparison = compare_inputs(collection, args, geometry)
        active = input_comparison["task"]["targetLayerId"]
        comparisons.setdefault(active or "", {})["inputComparison"] = _bound_report(collection, input_comparison)
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
        active = approach.get("targetLayerId") if "targetLayerId" in value else active
        selected_id = approach["id"]
        position = next((index for index, item in enumerate(saved) if item["id"] == selected_id), None)
        if position is None:
            saved.append(approach)
        else:
            if approach.get("parameterHash") != saved[position].get("parameterHash"):
                _mark_historical(comparisons, saved[position], "A selected corridor was edited after this comparison was frozen.")
            saved[position] = approach
    elif action == "delete":
        key = args.get("approachId")
        deleted = next((a for a in saved if a["id"] == key), None)
        if not deleted:
            raise ValueError("Unknown saved approach")
        saved = [a for a in saved if a["id"] != key]
        _mark_historical(comparisons, deleted, "A selected corridor was removed after this comparison was frozen.")
    elif action == "list":
        active = args.get("targetLayerId", active)
    else:
        raise ValueError("Unknown native approach action")
    if active is not None and not isinstance(active, str):
        raise ValueError("Invalid selected target")
    if action != "list":
        save_approaches(collection, saved, comparisons=comparisons, target_id=active, expected_revision=revision, commit=commit)
    visible = [row for row in saved if row.get("targetLayerId") == active or
               row.get("targetLayerId") is None and row.get("origin") != "bounded-native-entry-patch"]
    if not active and "targetLayerId" not in args and not comparisons:
        visible = saved  # Unbound v1 workspaces may contain untouched manual plans.
    selected_comparisons = comparisons.get(active or "", {})
    return {"snapshotId": collection.identity(), "approaches": visible, "selectedId": selected_id,
            "activeTargetLayerId": active, "savedTargetLayerIds": sorted({key for key in comparisons if key}
                | {row["targetLayerId"] for row in saved if row.get("targetLayerId")}),
            "proposalReport": selected_comparisons.get("proposalReport"), "inputComparison": selected_comparisons.get("inputComparison"),
            "simulationExecutionAuthorized": False}


def _mark_historical(comparisons, row, reason):
    pair = comparisons.get(row.get("targetLayerId") or "", {})
    report = pair.get("proposalReport")
    if report and row["id"] in report["selectedCandidateIds"] and report["assessmentStatus"] == "current":
        pair["proposalReport"] = {**report, "assessmentStatus": "historical", "invalidationReason": reason}
