#!/usr/bin/env python3
"""Reuse selected EDEN2020 volunteers as native anatomical planning studies.

No inference, registration, reconstruction, extraction or drive-wide scan runs.
Native sources and existing synthetic target geometry remain separate evidence.
"""
from __future__ import annotations

import argparse
from copy import deepcopy
import gc
import json
from pathlib import Path
import re

import nibabel as nib
from nibabel.spaces import vox2out_vox
import numpy as np
from scipy import ndimage as ndi

from resectionlab.anatomy_import import import_anatomy_derivative, import_nifti_series
from resectionlab.data_root import atomic_json, checked_root
from resectionlab.patient_anatomy import PatientImagingCollection, file_hash, world

URL = "https://zenodo.org/records/3994749"
LICENSE = "CC-BY-NC-ND-4.0; private noncommercial adaptation; no redistribution"
BRAIN_IDS = {2, 3, 7, 8, 10, 11, 12, 13, 16, 17, 18, 26, 41, 42, 46, 47,
             49, 50, 51, 52, 53, 54, 58, 77, 85, 99, 819, 820, 821, 822,
             843, 844, 865, 866, 869, 870}
COLORS = ["#61c4f2", "#f1b85b", "#b28af5", "#6ed4ac", "#ef879c", "#9ad15b", "#df9fe7"]
LIMITS = [
    "Healthy acquired anatomy with separate existing synthetic target; no disease displacement, infiltration or edema is modeled.",
    "Cached rigid maps are supplied estimates, not independently measured registration accuracy; source header ambiguities are retained.",
    "T1 head, brain, arterial and venous labels are unreviewed SuperSynth estimates; MRI head surface is not CT-measured skull.",
    "TOF vesselFM labels have incomplete scan coverage and unmeasured sensitivity; absent labels do not establish absent vessels.",
    "Supplied named tract reconstructions are geometrical baseline references, not functional mapping or complete white-matter anatomy.",
    "SWI is source contrast; no SWI-derived venous mask is available.",
    "Raw 4D diffusion, gradients and reverse-phase assets are retained without scalar-image conversion or new reconstruction.",
    "No clinical safety, neurological deficit probability, physiological simulation or physical execution admission is established.",
]


def header_geometry(image):
    q, qc = image.get_qform(coded=True)
    s, sc = image.get_sform(coded=True)
    return {"shape": list(image.shape), "affine_ras_mm": image.affine.tolist(),
            "voxel_spacing_mm": nib.affines.voxel_sizes(image.affine).tolist(),
            "qform_code": int(qc), "sform_code": int(sc),
            "qform": None if q is None else q.tolist(), "sform": None if s is None else s.tolist(),
            "units": list(image.header.get_xyzt_units()),
            "working_basis": "coded sform" if sc else "coded qform",
            "independent_frame_accuracy_measured": False}


def extraction_members(root, subject):
    base = root / "sources/eden2020-healthy-3994749"
    receipt = base / "provenance" / f"{subject}-extraction.json"
    members = {}
    extracted = (base / "extracted" / subject).resolve()
    for row in json.loads(receipt.read_text())["files"]:
        path = (extracted / row["path"]).resolve()
        if not path.is_relative_to(extracted) or not path.is_file():
            raise ValueError(f"Missing or escaping recorded source: {row['path']}")
        if path.stat().st_size != row["bytes"] or path.name in members:
            raise ValueError(f"Changed or ambiguous source member: {path}")
        members[path.name] = {**row, "absolute_path": str(path)}
    return members, receipt


def published_tract_alignment(points, trk, overlay, flair):
    """Cross-check TRK physical coordinates against the supplied voxel overlay."""
    image = nib.load(overlay)
    if image.shape != flair.shape or not np.allclose(image.affine, flair.affine, rtol=0, atol=1e-4):
        raise ValueError("Published tract overlay differs from the matching FLAIR grid")
    header = nib.streamlines.load(trk, lazy_load=True).header
    if not np.allclose(header["voxel_to_rasmm"], image.affine, rtol=0, atol=1e-4):
        raise ValueError("Published TRK and overlay coordinate headers differ")
    values = np.asanyarray(image.dataobj)
    if (not np.isfinite(values).all() or values.min() < 0
            or np.any((values >= 10000) & (values < 30000))):
        raise ValueError("Unresolved published fiber-overlay intensity encoding")
    foreground = values >= 30000
    inverse = np.linalg.inv(image.affine)
    voxels = np.rint(points @ inverse[:3, :3].T + inverse[:3, 3]).astype(np.int64)
    inside = np.all((voxels >= 0) & (voxels < image.shape), axis=1)
    agreement = float(foreground[tuple(voxels[inside].T)].mean()) if inside.any() else 0.0
    if not len(points) or not inside.all() or agreement < 0.99:
        raise ValueError("Published TRK points fail the independent overlay cross-check")
    return {"method": "nibabel TrackVis to RAS mm; matching FLAIR affine; supplied fiber-only overlay >=30000",
            "point_count": len(points), "all_points_inside_source": bool(inside.all()),
            "points_on_supplied_overlay_fraction": agreement, "overlay_sha256": file_hash(overlay),
            "source_frame_verified": True, "registration_accuracy_measured": False,
            "functional_identity_basis": "publisher filename; no independent functional validation"}


def superior_entry(envelope, native_affine, target):
    """A local superior outer head-surface point, for editable geometry only."""
    surface = envelope & ~ndi.binary_erosion(envelope)
    points = world(native_affine, np.argwhere(surface))
    target = np.asarray(target)
    local = (np.linalg.norm(points[:, :2] - target[:2], axis=1) <= 4) & (points[:, 2] > target[2] + 8)
    choices = points[local]
    if not len(choices):
        return None
    # Use the outer superior envelope in a narrow local column, never a sulcal
    # inner surface or an inferior shortest path. This is not a skull opening.
    choices = choices[choices[:, 2] >= choices[:, 2].max() - 1.5]
    score = np.linalg.norm(choices[:, :2] - target[:2], axis=1)
    return choices[int(np.argmin(score))].tolist()


def prepare(root, subject, *, variant=1):
    if not re.fullmatch(r"CTRL_[0-9]{4}", subject):
        raise ValueError("Choose an exact recorded EDEN2020 healthy volunteer ID, CTRL_NNNN")
    prepared = root / "prepared/eden2020-anatomy-v2"
    complete = prepared / "subjects" / subject / "complete.json"
    subject_row = json.loads(complete.read_text())
    if not subject_row["complete"] or subject_row["subject"] != subject:
        raise ValueError("Existing subject conversion is incomplete or mismatched")
    synthetic = next(r for r in subject_row["projects"] if r["synthetic"] and f"-SYNTHETIC-{variant:02d}-" in r["case_id"])
    output = root / "prepared/eden2020-planning-v1" / subject
    if (output / "patient-anatomy.json").exists():
        c = PatientImagingCollection(output)
        if c.document.get("preparation", {}).get("synthetic_source_case") != synthetic["case_id"]:
            raise ValueError("Existing study uses another target; preserve a distinct version")
        print(f"Reusing {output / 'patient-anatomy.json'}", flush=True)
        return json.loads((output / "source-map.json").read_text())
    members, extraction_receipt = extraction_members(root, subject)
    c = PatientImagingCollection.create(output, "EDEN2020:" + subject, timepoint="preoperative")
    c.document.update({"label": f"EDEN2020 {subject} · anatomical planning · synthetic target",
                       "source_url": URL, "source_license": LICENSE, "role": "DEVELOPMENT",
                       "clinical_validation": False, "training_admitted": False, "limitations": LIMITS,
                       "missing": ["CT-measured skull", "dura", "functional mapping", "complete vascular anatomy",
                                   "SWI-derived venous segmentation", "independently measured registration accuracy"]})
    source_map = {"schema": "eden2020-planning-source-map-v1", "subject": subject, "source_url": URL,
                  "collection": str(output / "patient-anatomy.json"), "extraction_receipt": str(extraction_receipt),
                  "existing_baseline": subject_row["projects"][0]["path"], "series": [], "tractography": [],
                  "synthetic_source": {"path": synthetic["path"], "sha256": synthetic["sha256"],
                                       "recipe": synthetic["recipe"]}, "limitations": LIMITS}
    source_specs = [("t1", "T1", "T1_3D_PROSET_Sag"), ("tof", "TOF", "s3DI_MC_HR"),
                    ("flair", "FLAIR", "3D_FLAIR_Tra"), ("swi", "SWI", "SWIp_axial")]
    fixed_shape, fixed_affine = vox2out_vox(
        nib.load(members[f"{subject}_T1_3D_PROSET_Sag.nii.gz"]["absolute_path"]),
        subject_row["projects"][0]["spacing_mm"],
    )
    for sid, modality, suffix in source_specs:
        member = members[f"{subject}_{suffix}.nii.gz"]
        path = Path(member["absolute_path"])
        print(f"Importing unchanged native {subject} {modality}: {nib.load(path).shape}", flush=True)
        row = import_nifti_series(c, path, series_id=sid, modality=modality)
        if row["original"]["sha256"] != member["sha256"]:
            raise ValueError(f"Source differs from extraction receipt: {path}")
        row.update({"label": f"{modality} · acquired healthy source", "source_url": URL,
                    "source_geometry": header_geometry(nib.load(path)), "healthy_acquisition": True})
        if sid != "t1":
            prior = "MRA" if sid == "tof" else modality
            receipt_path = prepared / "subjects" / subject / f"rigid-{prior}" / "registration.json"
            receipt = json.loads(receipt_path.read_text())
            if receipt["source_sha256"] != member["sha256"]:
                raise ValueError("Cached rigid map source identity differs")
            if (list(fixed_shape) != receipt["fixed_grid_shape"]
                    or not np.allclose(fixed_affine, receipt["fixed_affine"], rtol=0, atol=1e-6)):
                raise ValueError("Cached rigid map reference geometry differs from this native T1")
            c.set_registration(sid, receipt["moving_RAS_to_fixed_RAS_forward"], status="source-provided",
                               method="Imported cached rigid estimate: " + receipt["method"])
            c.document["transforms"][sid].update({"review_status": "unreviewed cached rigid estimate",
                                                 "clinical_alignment_validated": False,
                                                 "source_receipt": str(receipt_path), "source_receipt_sha256": file_hash(receipt_path),
                                                 "source_header_conflict_retained": receipt.get("source_header_conflict_retained", False),
                                                 "meaning": "source-provided pipeline estimate; source binding verified, independent accuracy unmeasured"})
        c.preview_pyramid(sid, levels=(2, 4) if sid in {"tof", "swi"} else (2,))
        source_map["series"].append({"id": sid, "modality": modality, "source": str(path),
                                     "sha256": member["sha256"], "native_shape": row["data"]["shape"],
                                     "package_original": row["original"]["path"], "package_array": row["data"]["path"]})
        gc.collect()

    head = root / subject_row["head_segmentation"]
    head_receipt = json.loads((head / "inference.json").read_text())
    head_path = head / "labels_1mm.nii.gz"
    if (head_receipt["source_sha256"] != c.series("t1")["original"]["sha256"]
            or file_hash(head_path) != head_receipt["output_sha256"]):
        raise ValueError("Existing native head-label identity differs")
    image = nib.load(head_path)
    labels = np.asarray(image.dataobj, np.uint16)
    common = {"kind": "model_estimate", "model": "SuperSynth", "source": c.original(head_path),
              "inference_receipt": str(head / "inference.json"), "source_resolution_mm": [1, 1, 1],
              "clinical_validation": False, "inference_parameters": head_receipt["parameters"]}
    def layer(data, layer_id, label, *, kind="context", role="context", color="#91b8c9", native_affine=None, source_series="t1", provenance=None):
        row = c.add_layer(data, image.affine if native_affine is None else native_affine,
                          layer_id=layer_id, source_series=source_series, kind=kind, role=role,
                          color=color, label=label, provenance=deepcopy(common if provenance is None else provenance))
        row["units_status"] = "mm"
        return row
    brain = np.isin(labels, list(BRAIN_IDS))
    layer(brain.astype(np.uint8), "brain", "Brain · T1 model estimate", kind="brain", color="#a3bed1")
    head_envelope = ndi.binary_fill_holes(labels > 0)
    layer(head_envelope.astype(np.uint8), "head-surface", "Head surface · T1 model estimate · no CT", color="#bead98")
    names = {}
    for line in (root / "cache/models/supersynth-source/FreeSurferColorLUT.txt").read_text().splitlines():
        fields = line.split()
        if len(fields) > 5 and fields[0].isdigit():
            names[int(fields[0])] = (fields[1], "#" + "".join(f"{int(v):02x}" for v in fields[2:5]))
    shared = c.put_array(labels)
    for code in sorted(int(v) for v in np.unique(labels) if v):
        name, color = names[code]
        sid = "t1-arteries" if code == 902 else "t1-veins" if code == 914 else f"head-{code}"
        row = {"id": sid, "patient_id": c.document["patient_id"], "source_series": "t1",
               "label": name.replace("-", " ") + " · T1 model estimate" + (" · no CT" if code in {915, 916} else ""),
               "kind": "artery" if code == 902 else "vein" if code == 914 else "brain-region",
               "role": "critical" if code in {16, 902, 914, 930} else "context", "color": color,
               "affine": image.affine.tolist(), "data": shared, "label_value": code, "revision": 0,
               "units_status": "mm", "review_status": "unreviewed model estimate",
               "coverage": "T1 model field; anatomical sensitivity unmeasured", "provenance": common}
        c.document["layers"].append(row)
    del labels

    work = prepared / "subjects" / subject
    inference = json.loads((work / "vesselfm-0p6mm/inference.json").read_text())
    vessel = next(r for r in inference["outputs"] if r["name"].endswith("_mask.nii.gz"))
    vessel_path = Path(vessel["path"])
    if file_hash(vessel_path) != vessel["sha256"]:
        raise ValueError("Existing vesselFM mask identity differs")
    vascular = nib.load(vessel_path)
    row = layer((np.asarray(vascular.dataobj) > 0).astype(np.uint8), "tof-arteries", "Arteries · TOF vesselFM estimate",
                kind="artery", role="critical", color="#f06269", native_affine=vascular.affine, source_series="tof",
                provenance={"kind": "model_estimate", "source": c.original(vessel_path), "modality": "TOF",
                            "inference_receipt": str(work / "vesselfm-0p6mm/inference.json"),
                            "clinical_validation": False, "complete_vessels": False})
    row["coverage"] = "Partial acquired TOF field; model sensitivity and unlabeled vessels unassessed"
    row["review_status"] = "unreviewed model and cached rigid registration"
    del vascular

    synthetic_labels = prepared / "labels" / (synthetic["case_id"] + ".nii.gz")
    target_image = nib.load(synthetic_labels)
    row = layer(np.isin(np.asarray(target_image.dataobj), [1, 2]).astype(np.uint8), "synthetic-target",
                "Synthetic target · existing core + rim", kind="tumor", role="target", color="#ee92d8", native_affine=target_image.affine,
                provenance={"kind": "simulated", "source": c.original(synthetic_labels), "source_case": synthetic["case_id"],
                            "source_package_sha256": synthetic["sha256"], "recipe": synthetic["recipe"],
                            "included_source_labels": [1, 2], "edema_label_3_excluded": True, "healthy_anatomy_unchanged": True,
                            "mass_effect": False, "infiltration": False, "clinical_validation": False})
    row["review_status"] = "synthetic engineering target; not acquired pathology"
    row["coverage"] = "Existing versioned synthetic geometry only"
    source_map["synthetic_source"]["target_volume_mm3"] = float(c.mask(row).sum() * abs(np.linalg.det(target_image.affine[:3, :3])))

    flair = nib.load(members[f"{subject}_3D_FLAIR_Tra.nii.gz"]["absolute_path"])
    trks = [members[name] for name in sorted(members) if name.endswith("_to_3DFLAIR.trk")]
    for index, member in enumerate(trks):
        path = Path(member["absolute_path"])
        name = path.name.removeprefix(subject + "_").removesuffix("_to_3DFLAIR.trk")
        print(f"Verifying supplied {subject} tract {name}", flush=True)
        tract = import_anatomy_derivative(c, path, source_series="flair", kind="tractography", label=name + " · supplied baseline")
        if tract["original"]["sha256"] != member["sha256"]:
            raise ValueError("Published TRK differs from recorded extraction")
        overlay = Path(members[f"{subject}_{name}_on_3DFLAIR.nii.gz"]["absolute_path"])
        proof = published_tract_alignment(c.array(tract["points_ras_mm"]), path, overlay, flair)
        if proof["overlay_sha256"] != members[overlay.name]["sha256"]:
            raise ValueError("Published overlay differs from recorded extraction")
        tract.update({"id": "tract-" + name.lower(), "color": COLORS[index % len(COLORS)], "role": "critical",
                      "coordinate_verification": proof, "review_status": "source frame verified; cached rigid alignment and functional completeness unreviewed",
                      "provenance": {"source_url": URL, "publisher_bundle_name": name, "source_overlay": str(overlay),
                                     "source_overlay_sha256": proof["overlay_sha256"], "functional_validation": False}})
        source_map["tractography"].append({"id": tract["id"], "label": name, "source": str(path),
                                           "sha256": member["sha256"], "streamline_count": tract["offsets"]["shape"][0] - 1, **proof})
        gc.collect()
    if len(trks) != subject_row["tract_count"]:
        raise ValueError("Published tract inventory differs from the completed manifest")

    assets = []
    for name, member in sorted(members.items()):
        if not any(key in name for key in ("raw_data_DTI_32", "raw_data_NODDI", "B0_reverse")):
            continue
        path = Path(member["absolute_path"])
        kind = "gradient_bvalues" if name.endswith(".bval") else "gradient_bvectors" if name.endswith(".bvec") else "reverse_phase_reference" if "B0_reverse" in name else "diffusion_4d"
        assets.append({"name": name, "kind": kind, "source_path": str(path), "sha256": member["sha256"],
                       "bytes": member["bytes"], "storage": "unchanged external source; drive required",
                       "source_identity_basis": "verified extraction receipt; size rechecked, not rehashed for this preparation",
                       "scalar_display": False, "reconstruction_run": False,
                       **({"source_geometry": header_geometry(nib.load(path))} if name.endswith(".nii.gz") else {})})
    c.document["source_assets"] = assets
    center = synthetic["recipe"]["center_ras_mm"]
    c.document["display"] = {"initial_layers": ["brain", "synthetic-target", "tof-arteries"], "cursor_ras_mm": center,
                              "entry_ras_mm": superior_entry(head_envelope, image.affine, center),
                              "entry_basis": "local superior outer estimated head surface; editable geometric starting point, no skull opening inferred",
                              "layer_opacity": {"brain": 0.18, "synthetic-target": 0.9, "tof-arteries": 0.85},
                              "source_series": "t1", "head_surface_available": True, "measured_skull_available": False}
    c.document["preparation"] = {"version": "eden2020-planning-v1", "subject": subject,
                                 "source_subject_manifest": str(complete), "synthetic_source_case": synthetic["case_id"],
                                 "synthetic_source_sha256": synthetic["sha256"], "registration_recomputed": False,
                                 "model_inference_run": False, "source_resolution_preserved": True}
    source_map["snapshot_id"] = c.save("Reuse EDEN sources, cached estimates, supplied named tracts and separate synthetic target")
    source_map["display"] = c.document["display"]
    source_map["source_assets"] = assets
    source_map["layer_count"] = len(c.document["layers"])
    atomic_json(output / "source-map.json", source_map)
    print(f"Saved {output / 'patient-anatomy.json'}", flush=True)
    return source_map


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path)
    parser.add_argument("--subjects", nargs="+", default=["CTRL_1277", "CTRL_0866"])
    parser.add_argument("--variant", type=int, default=1)
    args = parser.parse_args()
    if args.variant not in range(1, 9):
        parser.error("Choose an existing synthetic variant 1–8")
    root = checked_root(args.root, required_bytes=len(args.subjects) * 2 * 1024**3)
    output = root / "prepared/eden2020-planning-v1"
    output.mkdir(parents=True, exist_ok=True)
    maps = [prepare(root, subject, variant=args.variant) for subject in args.subjects]
    atomic_json(output / "collection.json", {"schema": "eden2020-planning-collection-v1", "primary": maps[0]["collection"],
                                              "studies": [{"subject": r["subject"], "path": r["collection"], "snapshot_id": r["snapshot_id"],
                                                           "tract_count": len(r["tractography"])} for r in maps],
                                              "source_url": URL, "limitations": LIMITS})


if __name__ == "__main__":
    main()
