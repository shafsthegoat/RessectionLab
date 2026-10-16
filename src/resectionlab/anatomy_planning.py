"""Snapshot-to-existing-planner adapter. Context is never a resection target."""

from pathlib import Path
import numpy as np
from scipy.spatial import cKDTree
from .core import CaseData, SourceRef, array_digest
from .critical_evidence import CriticalStructureEvidence, resolve_critical_evidence
from .data_policy import ANATOMY_PRETRAINED_POLICY
from .patient_anatomy import world, affine, digest, file_hash


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
    """Conservative voxel-cell capsule query against the canonical critical resolver."""
    if (
        not np.isfinite([*start, *end, radius_mm]).all()
        or np.asarray(start).shape != (3,)
        or np.asarray(end).shape != (3,)
        or not 0 < radius_mm <= 50
    ):
        raise ValueError(
            "A finite physical route and positive radius <=50 mm are required"
        )
    case = planning_case(snapshot)
    constraints = resolve_critical_evidence(case)
    a = case.affine
    spacing = np.linalg.norm(a[:3, :3], axis=0)
    length = np.linalg.norm(np.asarray(end) - start)
    if length > 1000:
        raise ValueError("Route exceeds bounded patient workspace")
    count = max(2, int(np.ceil(length / (spacing.min() * 0.25))) + 1)
    points = np.linspace(start, end, count)
    step = length / (count - 1)
    results = {}
    for name, mask in constraints.masks.items():
        if mask is None:
            results[name] = {"available": False}
            continue
        indices = np.argwhere(mask)
        if not len(indices):
            results[name] = {"available": False}
            continue
        distance = cKDTree(world(a, indices)).query(points)[0]
        # Circumscribed cell radius plus half sample step; never treats a thin
        # obstacle as safe merely because a centreline sample missed its cell.
        cell_radius = 0.5 * sum(np.linalg.norm(a[:3, i]) for i in range(3))
        clearance = float(distance.min() - radius_mm - cell_radius - step * 0.5)
        results[name] = {
            "available": True,
            "clearance_lower_bound_mm": clearance,
            "intersects_or_unresolved": clearance <= 0,
        }
    voxel = world(np.linalg.inv(a), points)
    covered = bool(
        np.all(voxel >= -0.5) and np.all(voxel < np.array(case.mri.shape) - 0.5)
    )
    return {
        "snapshot_id": snapshot.snapshot_id,
        "planning_hash": case.planning_hash,
        "constraints": results,
        "within_source_image": covered,
        "missing": list(case.unknowns),
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
