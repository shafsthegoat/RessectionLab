"""Permission-gated, local-only EDEN2020 intake receipts; never planner admission.

The caller authenticates access and review receipts. This module checks their
bindings, not the identity or authority of the person who issued them. It has no
network, extraction, model inference, patient fusion, or training entry point.
"""
from __future__ import annotations

import hashlib
from pathlib import Path
import re
from collections.abc import Mapping

import nibabel as nib
import numpy as np

DOI = "10.5281/zenodo.5714812"
SCHEMA = "resectionlab.eden2020-local-intake.v1"
MODALITIES = ("T1_POST", "FLAIR", "TOF_MRA", "SWI", "DTI_32", "NODDI", "B0_REVERSE")
REQUIRED = MODALITIES[:5]


def _need(ok, message):
    if not ok:
        raise ValueError(message)


def _sha(value):
    _need(isinstance(value, str) and re.fullmatch(r"[0-9a-f]{64}", value), "raw SHA256 required")
    return value


def _file(root, record):
    _need(isinstance(record, Mapping), "source file declaration required")
    expected = _sha(record.get("sha256"))
    relative = record.get("path")
    _need(isinstance(relative, str) and relative and not Path(relative).is_absolute(), "relative local source path required")
    path = (root / relative).resolve()
    _need(path.is_relative_to(root), "source path escapes local intake root")
    _need(path.is_file(), "declared local source missing")
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(chunk)
    _need(h.hexdigest() == expected, "source file SHA256 mismatch")
    return path


def _image(root, record, *, diffusion=False, mask=False, reverse=False):
    path = _file(root, record)
    image = nib.load(path)
    _need(len(image.shape) in ((3, 4) if reverse else (4,) if diffusion else (3,)) and all(n > 0 for n in image.shape), "NIfTI dimensionality invalid")
    _need(np.prod(image.shape, dtype=np.int64) <= 256_000_000, "bounded intake voxel budget exceeded")
    _need(image.header.get_xyzt_units()[0] == "mm", "NIfTI spatial units must be mm")
    q, qc = image.get_qform(coded=True)
    s, sc = image.get_sform(coded=True)
    _need(int(qc) > 0 and int(sc) > 0 and np.isfinite(q).all() and np.isfinite(s).all()
          and np.allclose(q, s, atol=1e-4, rtol=0), "explicit agreeing qform/sform required")
    affine = np.asarray(image.affine, dtype=float)
    _need(np.isfinite(affine).all() and np.allclose(affine[3], [0, 0, 0, 1])
          and abs(np.linalg.det(affine[:3, :3])) > 1e-8, "finite invertible NIfTI affine required")
    data = np.asarray(image.dataobj)
    _need(np.isfinite(data).all(), "nonfinite NIfTI samples")
    if mask:
        _need(np.isin(data, (0, 1)).all() and np.any(data), "nonempty binary annotation required")
    return {"path": record["path"], "sha256": record["sha256"], "shape": list(image.shape),
            "affine_ras_mm": affine.tolist(), "axis_codes": list(nib.aff2axcodes(affine)),
            "spatial_units": "mm", "qform_code": int(qc), "sform_code": int(sc),
            "spacing_mm": np.linalg.norm(affine[:3, :3], axis=0).tolist()}


def _gradients(root, row, volumes):
    bval_path, bvec_path = _file(root, row["bval"]), _file(root, row["bvec"])
    bvals = np.asarray(np.loadtxt(bval_path), dtype=float).reshape(-1)
    bvecs = np.asarray(np.loadtxt(bvec_path), dtype=float)
    _need(bvals.shape == (volumes,) and bvecs.shape == (3, volumes), "gradient dimensions require N bvals and 3xN bvecs")
    _need(np.isfinite(bvals).all() and np.isfinite(bvecs).all() and np.all(bvals >= 0), "finite nonnegative gradients required")
    baseline = bvals <= 50
    norms = np.linalg.norm(bvecs, axis=0)
    _need(np.any(baseline) and np.any(~baseline), "diffusion needs baseline and weighted volumes")
    _need(np.all(norms[baseline] <= .01) and np.all(np.abs(norms[~baseline] - 1) <= .01), "gradient vector normalization invalid")
    _need(row.get("gradient_frame") in {"voxel", "scanner_RAS", "scanner_LPS"}, "explicit gradient coordinate frame required")
    return {"bval": dict(row["bval"]), "bvec": dict(row["bvec"]), "gradient_frame": row["gradient_frame"],
            "volume_count": volumes, "baseline_volumes": int(baseline.sum()),
            "weighted_volumes": int((~baseline).sum()), "directions_reoriented": False,
            "distortion_and_motion_correction_validated": False}


def inspect_local_case(root, declaration, *, permission):
    """Inspect declared local bytes only after exact permission/role validation.

    Input rows belong to one prospective subject; imaging is never aligned or
    resampled here. A passed receipt still requires downstream anatomical and
    planning admission. SELECT may be inspected but never authorizes fitting.
    """
    _need(isinstance(permission, Mapping) and permission.get("status") == "granted"
          and permission.get("doi") == DOI and permission.get("intended_use") == "research_planning"
          and permission.get("license_status") == "reviewed_permitted", "explicit dataset permission and license evidence required before local inspection")
    permission_hash = _sha(permission.get("evidence_sha256"))
    _need(isinstance(declaration, Mapping), "prospective patient declaration required")
    subject = declaration.get("subject")
    _need(isinstance(subject, str) and re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]{0,63}", subject), "explicit EDEN subject required")
    _need(subject in permission.get("subjects", []) and declaration.get("doi") == DOI, "permission must cover this DOI and subject")
    _need(declaration.get("role") in {"TRAIN", "SELECT"}
          and declaration.get("prospective_before_inspection") is True, "prospective TRAIN or SELECT declaration required; EVAL prohibited")
    split_hash = _sha(declaration.get("split_evidence_sha256"))
    series = declaration.get("series")
    _need(isinstance(series, Mapping) and "T1_POST" in series and set(series) <= set(MODALITIES), "known series and reference postcontrast T1 required")
    _need(all(isinstance(row, Mapping) and row.get("subject") == subject for row in series.values()), "same-subject series binding required")
    annotations = declaration.get("annotations", {})
    _need(isinstance(annotations, Mapping) and set(annotations) <= {"brain", "tumor", "vessels", "tracts"}, "known explicit annotation kinds required")
    _need(all(isinstance(row, Mapping) and row.get("subject") == subject for row in annotations.values()), "same-subject annotation binding required")
    # No resolve, exists, open, hashing, or nibabel access occurs above this line.
    root = Path(root).resolve(strict=True)
    inspected, blockers = {}, []
    for modality, row in series.items():
        diffusion = modality in {"DTI_32", "NODDI"}
        result = _image(root, row, diffusion=diffusion, reverse=modality == "B0_REVERSE")
        if diffusion:
            result["gradients"] = _gradients(root, row, result["shape"][3])
            if modality == "DTI_32" and result["gradients"]["weighted_volumes"] != 32:
                blockers.append("DTI_32: publisher 32-direction protocol not confirmed")
        result["metadata_review"] = row.get("metadata_review", "unreviewed")
        result["registration_review"] = row.get("registration_review", "unreviewed")
        result["coverage_review"] = row.get("coverage_review", "unreviewed")
        result["subject"] = subject
        inspected[modality] = result
    reference = inspected["T1_POST"]
    for modality, result in inspected.items():
        same_grid = result["shape"][:3] == reference["shape"][:3] and np.allclose(result["affine_ras_mm"], reference["affine_ras_mm"], atol=1e-4, rtol=0)
        result["same_grid_as_T1_POST"] = bool(same_grid)
        # Same grid is not evidence of anatomical alignment.
        if not same_grid:
            blockers.append(f"{modality}: separate grid; reviewed source-bound registration required")
        for name in ("metadata_review", "registration_review", "coverage_review"):
            if result[name] != "pass": blockers.append(f"{modality}: {name} incomplete")
    masks = {}
    for kind, row in annotations.items():
        mask = _image(root, row, mask=True)
        _need(mask["shape"] == reference["shape"] and np.allclose(mask["affine_ras_mm"], reference["affine_ras_mm"], atol=1e-4, rtol=0), "annotation must match reference physical grid")
        _need(row.get("provenance") in {"supplied_annotation", "model_estimate"}, "annotation provenance required")
        mask.update(provenance=row["provenance"], review_status=row.get("review_status", "unreviewed"))
        if mask["review_status"] != "pass": blockers.append(f"{kind}: annotation quality review incomplete")
        if row["provenance"] == "model_estimate" and row.get("model_license_status") != "reviewed_permitted":
            blockers.append(f"{kind}: model license review incomplete")
        masks[kind] = mask
    missing = [name for name in REQUIRED if name not in inspected]
    blockers += [f"{name}: modality missing" for name in missing]
    blockers += [f"{name}: annotation missing" for name in ("brain", "tumor") if name not in masks]
    if declaration.get("intended_use_review") != "pass": blockers.append("intended-use anatomical QC incomplete")
    return {"schema": SCHEMA, "doi": DOI, "subject": subject, "patient_group": "EDEN2020:" + subject,
            "role": declaration["role"], "permission_evidence_sha256": permission_hash,
            "split_evidence_sha256": split_hash, "series": inspected, "annotations": masks,
            "missing_modalities": missing, "intake_checks_passed": not blockers, "blockers": blockers,
            "planning_admitted": False, "training_admitted": False, "optimizer_updates_authorized": 0,
            "next_gate": "independent anatomy and downstream case/evidence admission",
            "unknown_anatomy": ["skull_CT", "complete_venous_topology", "functional_mapping"]
                + (["reviewed_tract_anatomy"] if "tracts" not in masks or masks["tracts"]["review_status"] != "pass" else []),
            "operative_action_labels": None, "clinical_outcomes": None,
            "license_status": permission["license_status"], "acquired_arrays_modified": False}
