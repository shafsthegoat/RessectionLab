"""Snapshot-to-existing-planner adapter and native geometric target options."""

import numpy as np
from .core import CaseData, SourceRef, array_digest
from .critical_evidence import CriticalStructureEvidence
from .data_policy import ANATOMY_PRETRAINED_POLICY
from .anatomy_evidence import inference_bounds, is_dural_proxy, is_siam_context
from .patient_anatomy import world, digest, file_hash


LESION_CANDIDATE_KINDS = {
    "tumor", "lesion", "tumor-target", "lesion-target", "synthetic-target",
    "enhancing-lesion", "nonenhancing-lesion", "nonenhancing-abnormality",
    "edema", "edema-like-signal",
}


def is_native_target(row):
    if is_dural_proxy(row) or is_siam_context(row):
        return False
    return row["role"] == "target" or (
        row["role"] == "context" and row["kind"] in LESION_CANDIDATE_KINDS
        and row.get("provenance", {}).get("planning_target_candidate") is not False
    )


def native_target_binding(collection, row):
    """Bind a selection to evidence content, independently of display edits."""
    source = collection.series(row["source_series"])
    reference = collection.series(collection.document["reference_series"])
    transforms = collection.document["transforms"]
    chain, sid = {}, source["id"]
    while sid:
        if sid in chain or sid not in transforms:
            raise ValueError("Incomplete target registration chain")
        transform = transforms[sid]
        if transform["status"] not in {"reference", "reviewed", "source-provided"}:
            raise ValueError(f"Registration for {sid} needs review")
        anchor = collection.series(sid)
        chain[sid] = {"transform": transform, "sourceSha256": anchor["original"]["sha256"]}
        sid = transform.get("parent_series")
    collection.patient_affine(row)
    bounds = inference_bounds(row)
    layer = {key: row[key] for key in (
        "id", "label", "kind", "role", "revision", "review_status", "provenance", "coverage",
        "affine", "data", "label_value", "probability", "score_meaning", "ontology", "label_ontology",
    ) if key in row}
    if bounds is not None:
        layer["inference_bounds_voxel_inclusive"] = bounds
    return digest({
        "patient": collection.document["patient_id"], "layer": layer, "transforms": chain,
        "source": {key: source[key] for key in ("id", "timepoint", "original", "data", "affine")},
        "reference": {key: reference[key] for key in ("id", "timepoint", "original", "data", "affine")},
    })


def _inside_fields(points, fields):
    supported = np.ones(np.asarray(points).shape[:-1], dtype=bool)
    for inverse, low, high in fields:
        voxel = world(inverse, points)
        supported &= np.all((voxel >= np.asarray(low) - .5) & (voxel < np.asarray(high) + .5), axis=-1)
    return supported


def _target_focus(data, label_value, transform, fields, check):
    """Find a supported labelled voxel centre near the physical label centroid."""
    def labelled_centers():
        slab = max(1, 131072 // np.prod(data.shape[1:]))
        for start in range(0, data.shape[0], slab):
            check()
            indices = np.argwhere(data[start:start + slab] == label_value)
            indices[:, 0] += start
            if len(indices):
                yield world(transform, indices)

    total, count, labelled = np.zeros(3), 0, 0
    for points in labelled_centers():
        labelled += len(points)
        points = points[_inside_fields(points, fields)]
        total += points.sum(axis=0)
        count += len(points)
    if not labelled:
        raise ValueError("Empty target label")
    if not count:
        raise ValueError("No labelled voxel centre inside the acquisition, reference and inference fields")
    centroid = total / count
    nearest, distance = None, float("inf")
    for points in labelled_centers():
        points = points[_inside_fields(points, fields)]
        if not len(points):
            continue
        distances = np.sum((points - centroid) ** 2, axis=1)
        index = int(distances.argmin())
        if distances[index] < distance:
            nearest, distance = points[index], distances[index]
    return nearest.tolist()


def native_target_options(collection, *, layer_id=None, check=lambda: None):
    """Source-bound target choices independent of display or planner caches."""
    rows = [row for row in collection.document["layers"] if is_native_target(row)]
    if not rows:
        return []
    prepared_targets = [row["id"] for row in rows if row["role"] == "target"]
    display = collection.document.get("display", {})
    hints = display.get("planning_targets", {})
    reference = collection.series(collection.document["reference_series"])

    def point(value):
        if isinstance(value, list) and len(value) == 3 and all(
            type(coordinate) in (int, float) and np.isfinite(coordinate) for coordinate in value
        ):
            return value
        return None

    result = []
    for row in rows:
        if layer_id is not None and row["id"] != layer_id:
            continue
        check()
        key = row["id"]
        provenance = row.get("provenance", {})
        origin = ("Synthetic target" if row["kind"] == "synthetic-target" or provenance.get("kind") == "simulated" else
                  "Model estimate" if provenance.get("model") else
                  "Supplied annotation" if provenance.get("source") else "Anatomy layer")
        option = {"layerId": key, "label": row["label"], "target": None, "kind": row["kind"],
                  "origin": origin, "provenance": provenance.get("kind"), "model": provenance.get("model"),
                  "sourceSeries": row["source_series"], "layerRevision": row.get("revision", 0),
                  "labelValue": row["label_value"], "reviewStatus": row.get("review_status", "unreviewed"),
                  "coverageMeaning": row.get("coverage")}
        hint = hints.get(key) if isinstance(hints, dict) else None
        hint = hint if isinstance(hint, dict) else display if prepared_targets == [key] else {}
        try:
            binding = native_target_binding(collection, row)
            transform = collection.patient_affine(row)
            data = collection.array(row["data"])
            source = collection.series(row["source_series"])
            fields = [(np.linalg.inv(collection.patient_affine(series)), (0, 0, 0), np.asarray(series["data"]["shape"]) - 1)
                      for series in (source, reference)]
            if (bounds := inference_bounds(row)) is not None:
                fields.append((np.linalg.inv(transform), *bounds))
            focus = point(hint.get("cursor_ras_mm"))
            if focus is not None:
                index = np.floor(world(np.linalg.inv(transform), focus) + .5)
                if (np.all((index >= 0) & (index < data.shape))
                        and data[tuple(index.astype(int))] == row["label_value"] and _inside_fields(focus, fields)):
                    option.update(target=focus, focusMeaning="Prepared point inside the target label")
            if option["target"] is None:
                focus = _target_focus(data, row["label_value"], transform, fields, check)
                option.update(target=focus, focusMeaning="Labelled voxel centre nearest the physical label centroid")
            entry = point(hint.get("entry_ras_mm"))
            if entry is not None and np.linalg.norm(np.asarray(entry) - option["target"]) > 1e-6:
                option["entry"] = entry
            option["candidateBinding"] = binding
        except (ValueError, OSError, StopIteration) as error:
            option["reason"] = str(error) or "Target source series is unavailable"
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
