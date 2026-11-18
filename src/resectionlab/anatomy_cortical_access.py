"""Source evidence for the straight ipsilateral supratentorial cortical scope."""

import numpy as np

from .anatomy_evidence import inference_bounds, is_siam_context
from .geometry import capsule_voxel_indices, point_segment_distances
from .patient_anatomy import world


ACCESS_SCOPE = "ipsilateral-supratentorial-cortical"
HEMISPHERES = {side: f"{side}-cerebral-hemisphere" for side in ("left", "right")}
SURFACE = "cortical-entry-surface"
SUPRATENTORIAL = "supratentorial-compartment"
BARRIERS = ("falx-cerebri", "tentorium-cerebelli", "posterior-fossa")
REQUIRED = (*HEMISPHERES.values(), SURFACE, SUPRATENTORIAL, *BARRIERS[:2])
ACCESS_ROLES = {*REQUIRED, *BARRIERS}

_LEGACY_HEAD = {"head-7", "head-8", "head-16", "head-46", "head-47"}
_REFERENCE_PROTECTED_KINDS = {"brainstem", "cerebellum", "cerebellar-white-matter", "cerebellar-cortex"}
_PROTECTED_KINDS = _REFERENCE_PROTECTED_KINDS | {"medulla", "pons", "spinal-cord", "posterior-fossa"}


def protected_compartment(row, *, reference_inputs_only=False):
    """Retain positive compartment guards for typed layers and legacy EDEN IDs."""
    if is_siam_context(row):
        return False
    kinds = _REFERENCE_PROTECTED_KINDS if reference_inputs_only else _PROTECTED_KINDS
    return row["kind"] in kinds or (row["kind"] == "brain-region" and row["id"] in _LEGACY_HEAD)


def cortical_role(row):
    provenance = row.get("provenance")
    role = provenance.get("cortical_access_role") if isinstance(provenance, dict) else None
    return role if isinstance(role, str) and role in ACCESS_ROLES else None


def qualified_cortical_source(row):
    provenance = row.get("provenance", {})
    if (not cortical_role(row) or row.get("availability") == "empty_output"
            or row.get("evidence_status") == "empty-output"
            or provenance.get("evidence_use") == "context_only"):
        return False
    status = row.get("review_status")
    evidence = provenance.get("cortical_access_evidence")
    return (evidence == "generated-reference" and status == "generated-engineering"
            or evidence == "reviewed-source" and status == "reviewed"
            and provenance.get("kind") in {"observed_source_annotation", "supplied_annotation"})


class _Source:
    def __init__(self, geometry, row, shift):
        self.geometry = geometry
        self.row = row
        self.scene = geometry.grid(row)[1]
        self.mask = geometry.mask(row)
        self.shift = np.asarray(shift, float)
        self.acquisition = geometry.c.series(row["source_series"])
        self.acquisition_shape = np.asarray(self.acquisition["data"]["shape"])
        self.acquisition_from_label = np.linalg.inv(geometry.patient_affine(self.acquisition)) @ self.scene.affine
        self.bounds = inference_bounds(row)
        axial = row.get("coverage_fields", {}).get("supported_axial_indices")
        self.axial = np.ones(self.scene.shape[2], bool)
        if axial is not None:
            if not isinstance(axial, list) or not all(type(z) is int and 0 <= z < len(self.axial) for z in axial):
                raise ValueError("Invalid cortical access slice coverage")
            self.axial[:] = False
            self.axial[axial] = True
        if not geometry.population(row) or (axial is not None and not any(self.mask[:, :, z].any() for z in axial)):
            raise ValueError("Cortical access source has no supported foreground")

    def identity(self):
        return {"layerId": self.row["id"], "sourceSeries": self.acquisition["id"],
                "sourceSha256": self.acquisition["original"]["sha256"], "dataSha256": self.row["data"]["sha256"],
                "revision": self.row.get("revision", 0), "labelValue": self.row["label_value"],
                "evidence": self.row["provenance"]["cortical_access_evidence"]}

    def point(self, point):
        start = point - self.shift
        cell = np.floor(self.scene.world_to_voxel(start) + .5).astype(int)
        if (np.any(cell < 0) or np.any(cell >= self.scene.shape)
                or not self.covers(point, point, 0.) or not self.observed(cell[None])[0]):
            return None, cell
        return bool(self.mask[tuple(cell)]), cell

    def observed(self, cells):
        centers = world(self.acquisition_from_label, cells)
        known = self.axial[cells[:, 2]] & np.all((centers >= -.5) & (centers < self.acquisition_shape - .5), axis=1)
        if self.bounds is not None:
            low, high = self.bounds
            known &= np.all((cells >= low) & (cells <= high), axis=1)
        return known

    def covers(self, start, end, radius):
        start, end = start - self.shift, end - self.shift
        layer = self.geometry.layer_coverage(self.row, start, end, radius)
        acquisition = self.geometry.layer_coverage(self.acquisition, start, end, radius)
        return layer["wholeEnvelopeWithinField"] and acquisition["wholeEnvelopeWithinField"]

    def probe(self, start, end, radius, cache, *, entry_face=None, containment=False):
        """Return True/False only when source geometry establishes the result."""
        local_start, local_end = start - self.shift, end - self.shift
        key = (id(self.scene), tuple(local_start), tuple(local_end), radius)
        if key not in cache:
            cache[key] = capsule_voxel_indices(self.scene, local_start, local_end, radius, check=self.geometry.check)
        cells = cache[key]
        if entry_face is not None:
            axis, inside, direction = entry_face
            # The entry end-cap can straddle the exposed face. Require the whole
            # intersection on its inward side, without trimming the route axis.
            cells = cells[direction * (cells[:, axis] - inside) >= 0]
        known = self.observed(cells)
        foreground = self.mask[tuple(cells.T)]
        covered = bool(len(cells) and known.all() and self.covers(start, end, radius))
        if containment and covered and foreground.all():
            return True
        witnesses = cells[known & (~foreground if containment else foreground)]
        # Partial acquisition can establish positive contacts, but cannot
        # establish a violation from a zero that lies outside its support.
        if len(witnesses) and (covered or not containment):
            exact = self.scene._orthogonal_spacing is not None
            if exact or np.any(point_segment_distances(self.scene.voxel_to_world(witnesses), local_start, local_end) <= radius):
                return not containment
            return None
        return False if covered and not containment else None


def assess_cortical_access(geometry, start, end, radius, measured, scenario):
    result = {"scope": ACCESS_SCOPE, "status": "incomplete", "hemisphere": None,
              "sourceSnapshotId": geometry.snapshot.snapshot_id, "sourceLayers": {},
              "surfaceEntry": None, "postEntryEnvelopeSupported": None, "distalCapSupported": None,
              "missingEvidence": [], "violations": [], "simulationExecutionAuthorized": False}
    missing, violations = result["missingEvidence"], result["violations"]
    shifts = (scenario or {}).get("sourceTranslationsMm", {})
    boundary = (scenario or {}).get("boundaryMm", 0.)
    sources, cache = {}, {}
    for role in (*REQUIRED, BARRIERS[-1]):
        rows = geometry.cortical_sources.get(role, ())
        if len(rows) == 1:
            row = rows[0]
            try:
                source = _Source(geometry, row, shifts.get(row["source_series"], [0., 0., 0.]))
            except (ValueError, OSError):
                pass
            else:
                sources[role] = source
                result["sourceLayers"][role] = source.identity()
        if role in REQUIRED and role not in sources:
            missing.append(role)

    protected_ids = sorted(geometry.protected_compartment_ids.intersection(measured["detectedConflictLayerIds"]))
    violations.extend(f"Protected compartment contact: {layer_id}" for layer_id in protected_ids)
    for role in BARRIERS:
        if role not in sources:
            continue
        contact = sources[role].probe(start, end, radius + boundary, cache)
        if contact:
            violations.append(f"Capsule contact with source-labelled {role}")
        elif contact is None:
            missing.append(role + "-corridor-coverage")

    target_endpoint = (measured["targetLayerId"] is not None
                       and any(row.get("targetPointInsideLabel") for row in measured["targets"]))
    if not target_endpoint:
        missing.append("selected-target-label")
    target_sides = {side: sources[role].point(end)[0] for side, role in HEMISPHERES.items() if role in sources}
    side = next((side for side, inside in target_sides.items() if inside is True), None)
    if set(target_sides.values()) == {False} and len(target_sides) == len(HEMISPHERES):
        violations.append("Target lies outside both source-labelled cerebral hemispheres")
    if side is not None and list(target_sides.values()).count(True) == 1:
        result["hemisphere"] = side
        _check_hemisphere(result, sources, side, start, end, radius, cache)
    elif len(target_sides) == len(HEMISPHERES):
        missing.append("unambiguous-target-hemisphere")

    supra = sources.get(SUPRATENTORIAL)
    for field, a, region in (("postEntryEnvelopeSupported", start, "post-entry capsule"),
                             ("distalCapSupported", end, "distal cap")):
        contained = supra.probe(a, end, radius, cache, containment=True) if supra is not None else None
        if contained is False:
            violations.append(f"The {region} leaves the source-labelled supratentorial compartment")
        elif contained is None and supra is not None:
            missing.append("supratentorial-" + region.replace(" ", "-"))
        both = (result[field], contained)
        result[field] = False if False in both else True if all(both) else None

    if violations:
        result["status"] = "rejected"
    elif not missing and result["surfaceEntry"] and result["postEntryEnvelopeSupported"] and result["distalCapSupported"]:
        result["status"] = "supported"
    return result


def _check_hemisphere(result, sources, side, start, end, radius, cache):
    missing, violations = result["missingEvidence"], result["violations"]
    own = sources[HEMISPHERES[side]]
    other = sources.get(HEMISPHERES["left" if side == "right" else "right"])
    if other is not None:
        contact = other.probe(start, end, radius, cache)
        if contact:
            violations.append("Capsule enters the source-labelled contralateral cerebral hemisphere")
        elif contact is None:
            missing.append("contralateral-hemisphere-corridor-coverage")

    # Read immediately on either side of the nominated point, in patient RAS.
    # The step is a numerical tolerance for a voxel face, not a second patch.
    direction = (end - start) / np.linalg.norm(end - start)
    spacing = np.linalg.norm(own.scene.affine[:3, :3], axis=0)
    offset = direction * min(1e-5, float(spacing.min()) * 1e-5)
    before, before_cell = own.point(start - offset)
    after, after_cell = own.point(start + offset)
    if before is None or after is None:
        missing.append("hemisphere-entry-coverage")
        return
    if before or not after:
        violations.append("Entry must cross from outside to inside the source-labelled target hemisphere")
        return
    changed = np.flatnonzero(after_cell != before_cell)
    if len(changed) != 1:
        missing.append("unambiguous-cortical-entry-face")
        return
    axis = int(changed[0])
    sign = int(after_cell[axis] - before_cell[axis])
    surface = sources.get(SURFACE)
    if surface is not None:
        matches, cell = surface.point(start + offset)
        if matches is True:
            origin, target = own.scene.world_to_voxel(np.stack([start, end]) - own.shift)
            fraction = (after_cell[axis] - sign * .5 - origin[axis]) / (target[axis] - origin[axis])
            result["surfaceEntry"] = {"point": (start + fraction * (end - start)).tolist(),
                                      "surfaceLayerId": surface.row["id"], "surfaceCell": cell.tolist(),
                                      "hemisphereLayerId": own.row["id"], "faceAxis": axis,
                                      "hemisphereCell": after_cell.tolist(), "inwardSign": sign}
        elif matches is None:
            missing.append("cortical-entry-surface-coverage")
        else:
            violations.append("Entry face is outside the supplied cortical entry surface")

    for field, a, face, region in (("postEntryEnvelopeSupported", start, (axis, after_cell[axis], sign), "post-entry capsule"),
                                  ("distalCapSupported", end, None, "distal cap")):
        contained = own.probe(a, end, radius, cache, entry_face=face, containment=True)
        result[field] = contained
        if contained is False:
            violations.append(f"The {region} leaves the source-labelled target hemisphere")
        elif contained is None:
            missing.append("target-hemisphere-" + region.replace(" ", "-"))
