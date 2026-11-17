"""Snapshot-to-existing-planner adapter. Context is never a resection target."""

import numpy as np
from .core import CaseData, SourceRef, array_digest
from .critical_evidence import CriticalStructureEvidence
from .data_policy import ANATOMY_PRETRAINED_POLICY
from .patient_anatomy import world, digest, file_hash


def _target_focus(data, label_value, transform, check):
    """Find a labelled voxel centre nearest the physical label centroid."""
    def labelled_centers():
        slab = max(1, 131072 // np.prod(data.shape[1:]))
        for start in range(0, data.shape[0], slab):
            check()
            indices = np.argwhere(data[start:start + slab] == label_value)
            indices[:, 0] += start
            if len(indices):
                yield world(transform, indices)

    total, count = np.zeros(3), 0
    for points in labelled_centers():
        total += points.sum(axis=0)
        count += len(points)
    if not count:
        return None
    centroid = total / count
    nearest, distance = None, float("inf")
    for points in labelled_centers():
        distances = np.sum((points - centroid) ** 2, axis=1)
        index = int(distances.argmin())
        if distances[index] < distance:
            nearest, distance = points[index], distances[index]
    return nearest.tolist()


def native_target_options(collection, *, check=lambda: None):
    """Source-bound target choices independent of display or planner caches."""
    rows = [row for row in collection.document["layers"] if row["role"] == "target"]
    display = collection.document.get("display", {})
    hints = display.get("planning_targets", {})

    def point(value):
        if isinstance(value, list) and len(value) == 3 and all(
            type(coordinate) in (int, float) and np.isfinite(coordinate) for coordinate in value
        ):
            return value
        return None

    result = []
    for row in rows:
        check()
        layer_id = row["id"]
        option = {"layerId": layer_id, "label": row["label"], "target": None,
                  "provenance": row.get("provenance", {}).get("kind")}
        hint = hints.get(layer_id) if isinstance(hints, dict) else None
        hint = hint if isinstance(hint, dict) else display if len(rows) == 1 else {}
        try:
            transform = collection.patient_affine(row)
            data = collection.array(row["data"])
            focus = point(hint.get("cursor_ras_mm"))
            if focus is not None:
                index = np.floor(world(np.linalg.inv(transform), focus) + 0.5)
                if np.all((index >= 0) & (index < data.shape)) and data[tuple(index.astype(int))] == row["label_value"]:
                    option.update(target=focus, focusMeaning="Prepared point inside the supplied target label")
            if option["target"] is None:
                focus = _target_focus(data, row["label_value"], transform, check)
                if focus is None:
                    raise ValueError("Empty target label")
                option.update(target=focus, focusMeaning="Labelled voxel centre nearest the physical label centroid")
            entry = point(hint.get("entry_ras_mm"))
            if entry is not None and np.linalg.norm(np.asarray(entry) - option["target"]) > 1e-6:
                option["entry"] = entry
        except (ValueError, OSError) as error:
            option["reason"] = str(error)
        result.append(option)
    return result


def planning_case(snapshot, *, removed=None):
    snapshot._check()
    c = snapshot.collection
    grid = snapshot.simulation_grid(removed=removed)
    reference = c.series(c.document["reference_series"])
    image = c.array(reference["data"])
    ref = SourceRef(
        reference["id"],
        c.path(reference["original"]["path"]).as_uri(),
        reference["original"]["sha256"],
        c.document.get("license", "local-research-rights-unassessed"),
        "RAS+",
        "observed",
    )
    identity = {
        "dataset": c.document.get("dataset", "local-patient-collection"),
        "participant": c.document["patient_id"],
        "timepoint": reference["timepoint"],
    }
    evidence = {}
    metadata = {
        "patient_anatomy_snapshot": snapshot.snapshot_id,
        "anatomy_model_policy": ANATOMY_PRETRAINED_POLICY,
        "evidence_identity": identity,
        "simulation_derivative": grid["resampling"],
        "geometric_removal_preview": None if removed is None else array_digest(removed),
    }
    # The existing planner has one record for vascular obstacles. Union its selected
    # sources, retaining each original layer and transform in the derivative receipt.
    vascular = [
        row
        for row in c.document["layers"]
        if row["id"] in grid["critical"]
        and row["kind"] in {"vessels", "artery", "vein", "sinus"}
    ]
    for structure, rows in [
        ("vessels", vascular),
        (
            "motor",
            [
                r
                for r in c.document["layers"]
                if r["id"] in grid["critical"] and r["kind"] == "motor"
            ],
        ),
        (
            "language",
            [
                r
                for r in c.document["layers"]
                if r["id"] in grid["critical"] and r["kind"] == "language"
            ],
        ),
    ]:
        if not rows:
            continue
        mask = np.logical_or.reduce([grid["critical"][r["id"]] for r in rows])
        source = SourceRef(
            "snapshot-" + structure,
            (c.root / "revisions" / f"{snapshot.snapshot_id}.json").as_uri(),
            file_hash(c.root / "revisions" / f"{snapshot.snapshot_id}.json"),
            ref.license,
            "RAS+",
            "estimated",
        )
        models = sorted(
            {
                w["sha256"]
                for r in rows
                for w in r.get("provenance", {}).get("inference", {}).get("weights", [])
            }
        )
        item = CriticalStructureEvidence(
            evidence_id=structure,
            structure=structure,
            mask=mask,
            annotation_coverage=mask,
            affine_ras_mm=grid["affine"],
            case_id=c.document["patient_id"],
            reference_image_hash=array_digest(image),
            reference_source_id=ref.source_id,
            reference_source_sha256=ref.sha256,
            source=source,
            source_binding={
                **identity,
                "linkage_source": source.to_dict(),
                "coverage_source": source.to_dict(),
                "coverage_meaning": "source_documented_annotation_domain",
                "negative_meaning": "unlabelled anatomy is unknown",
            },
            derivation={
                "method": "patient_anatomy_snapshot",
                "source_image_sha256": ref.sha256,
                "annotation_sha256": source.sha256,
                "snapshot_id": snapshot.snapshot_id,
                "policy": ANATOMY_PRETRAINED_POLICY,
                "positive_mask_hash": array_digest(mask),
                "layers": rows,
                "resampling": grid["resampling"],
            },
            lineage={
                "kind": "model_assisted",
                "source": source.to_dict(),
                "model_sha256": models,
            },
        )
        evidence[structure] = item
    return CaseData(
        case_id=c.document["patient_id"],
        mri=image,
        compartments=grid["targets"],
        affine=grid["affine"],
        source_refs=(ref,),
        unknowns=tuple(grid["unknown"]),
        metadata=metadata,
        critical_evidence=evidence,
    )


def route_clearance(snapshot, start, end, radius_mm=1.0):
    """Native inspection independent of simulator admission, with whole-cell bounds."""
    from .anatomy_approaches import NativeRouteGeometry, VASCULAR_KINDS

    measured = NativeRouteGeometry(snapshot).measure(start, end, radius_mm)
    results = {name: {"available": False} for name in ("vessels", "motor", "language")}
    for item in measured["structures"]:
        name = "vessels" if item["kind"] in VASCULAR_KINDS else item["kind"]
        if not item["available"]:
            results.setdefault(name, {"available": False})
            continue
        previous = results.get(name, {})
        bound = item["clearanceLowerBoundMm"]
        if previous.get("available"):
            bound = min(bound, previous["clearance_lower_bound_mm"])
        results[name] = {"available": True, "clearance_lower_bound_mm": bound,
                         "intersects_or_unresolved": item["intersectsOrUnresolved"] or previous.get("intersects_or_unresolved", False)}
    return {
        "snapshot_id": snapshot.snapshot_id,
        "planning_hash": digest({"snapshot": snapshot.snapshot_id, "geometry": measured["geometryVersion"]}),
        "constraints": results,
        "within_source_image": measured["coverage"]["withinReferenceEnvelope"],
        "missing": measured["unknowns"],
        "measurements": measured,
        "clinical_clearance": False,
        "meaning": "conservative geometric capsule test; unlabelled anatomy remains unknown",
    }


def removal_preview(snapshot, point, radius_mm):
    """Reversible geometric removal of targets only, never mechanics or bleeding."""
    snapshot._check()
    c = snapshot.collection
    reference = c.series(c.document["reference_series"])
    a = c.patient_affine(reference)
    if (
        np.asarray(point).shape != (3,)
        or not np.isfinite([*point, radius_mm]).all()
        or not 0 < radius_mm <= 50
    ):
        raise ValueError("Invalid removal sphere")
    shape = tuple(reference["data"]["shape"])
    removed = np.zeros(shape, bool)
    inverse = np.linalg.inv(a)
    centre = world(inverse, point)
    extent = radius_mm * np.linalg.norm(inverse[:3, :3], axis=1)
    lo = np.maximum(0, np.floor(centre - extent).astype(int))
    hi = np.minimum(shape, np.ceil(centre + extent).astype(int) + 1)
    if np.all(hi > lo):
        indices = np.indices(hi - lo).reshape(3, -1).T + lo
        chosen = indices[np.linalg.norm(world(a, indices) - point, axis=1) <= radius_mm]
        removed[tuple(chosen.T)] = True
    return removed
