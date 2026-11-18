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
import os
from pathlib import Path
import re
import shutil
import tempfile
from zipfile import ZipFile

import nibabel as nib
from nibabel.spaces import vox2out_vox
import numpy as np
from scipy import ndimage as ndi

from resectionlab.anatomy_import import import_anatomy_derivative, import_nifti_series
from resectionlab.data_root import atomic_json, checked_root, safe_path
from resectionlab.patient_anatomy import PatientImagingCollection, file_hash, world
from resectionlab.workspace_bundle import extension_layout

URL = "https://zenodo.org/records/3994749"
LICENSE = "CC-BY-NC-ND-4.0; private noncommercial adaptation; no redistribution"
SOURCE_COLLECTION = Path("prepared/eden2020-anatomy-v2")
LEGACY_STUDIES = Path("prepared/eden2020-planning-v1")
COMPARISON_SUBJECTS = ("CTRL_1277", "CTRL_0866", "CTRL_0933")
COMPARISON_TARGETS = {1: "synthetic-target", 8: "synthetic-target-08"}
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


def subject_sources(root, subject):
    if not re.fullmatch(r"CTRL_[0-9]{4}", subject):
        raise ValueError("Choose an exact recorded EDEN2020 healthy volunteer ID, CTRL_NNNN")
    complete = root / SOURCE_COLLECTION / "subjects" / subject / "complete.json"
    subject_row = json.loads(complete.read_text())
    if not subject_row["complete"] or subject_row["subject"] != subject:
        raise ValueError("Existing subject conversion is incomplete or mismatched")
    return complete, subject_row


def synthetic_source(subject_row, variant):
    subject = subject_row["subject"]
    expected = f"EDEN2020-{subject}-SYNTHETIC-{variant:02d}-SEGMENTED-v2"
    rows = [row for row in subject_row["projects"] if row["case_id"] == expected]
    if (len(rows) != 1 or rows[0]["synthetic"] is not True
            or rows[0]["person_id"] != f"EDEN2020:{subject}" or rows[0]["role"] != "DEVELOPMENT"):
        raise ValueError(f"Missing or mismatched prepared synthetic target {expected}")
    return rows[0]


def verified_target_label(root, synthetic):
    """Bind the existing label bytes to the archived, recorded synthetic case."""
    source_case = safe_path(root, synthetic["path"])
    label_path = safe_path(root, SOURCE_COLLECTION / "labels" / (synthetic["case_id"] + ".nii.gz"))
    if source_case.stat().st_size != synthetic["bytes"]:
        raise ValueError("Prepared synthetic case size differs from its receipt")
    with ZipFile(source_case) as archive:
        manifest = extension_layout(archive)
    originals = [row for row in manifest["originalFiles"]
                 if row["role"] == "annotation" and row["name"] == label_path.name]
    if (manifest["sessionHash"] != synthetic["session_hash"]
            or manifest["referenceCaseHash"] != synthetic["case_hash"]
            or len(originals) != 1 or file_hash(label_path) != originals[0]["sha256"]):
        raise ValueError("Prepared synthetic label identity differs from its archived case")
    return label_path, originals[0]["sha256"]


def add_synthetic_target(c, root, synthetic, layer_id, label, *, candidate=False):
    label_path, label_hash = verified_target_label(root, synthetic)
    image = nib.load(label_path)
    mask = np.isin(np.asarray(image.dataobj), [1, 2]).astype(np.uint8)
    if (list(mask.shape) != synthetic["shape"]
            or not np.allclose(nib.affines.voxel_sizes(image.affine), synthetic["spacing_mm"], rtol=0, atol=1e-5)
            or np.count_nonzero(mask) != synthetic["recipe"]["metrics"]["voxel_count"]):
        raise ValueError("Synthetic target grid or core/rim voxel inventory differs from its receipt")
    row = c.add_layer(
        mask, image.affine, layer_id=layer_id, source_series="t1",
        kind="tumor", role="context" if candidate else "target", color="#ee92d8", label=label,
        provenance={"kind": "simulated", "source": c.original(label_path),
                    "source_case": synthetic["case_id"], "source_package_sha256": synthetic["sha256"],
                    "recipe": synthetic["recipe"], "included_source_labels": [1, 2],
                    **({"planning_target_candidate": True} if candidate else {}),
                    "edema_label_3_excluded": True, "healthy_anatomy_unchanged": True,
                    "mass_effect": False, "infiltration": False, "clinical_validation": False},
    )
    if row["provenance"]["source"]["sha256"] != label_hash:
        raise ValueError("Synthetic label source changed during preparation")
    row.update(units_status="mm", review_status="synthetic engineering target; not acquired pathology",
               coverage="Existing versioned synthetic geometry only")
    return target_source_receipt(root, synthetic, row)


def target_source_receipt(root, synthetic, layer):
    recipe = synthetic["recipe"]
    return {"path": synthetic["path"], "sha256": synthetic["sha256"],
            "label_source": str(root / SOURCE_COLLECTION / "labels" / (synthetic["case_id"] + ".nii.gz")),
            "label_sha256": layer["provenance"]["source"]["sha256"], "layer_id": layer["id"],
            "case_id": synthetic["case_id"], "recipe": recipe,
            "target_volume_mm3": float(recipe["metrics"]["voxel_count"] * abs(np.linalg.det(np.asarray(layer["affine"])[:3, :3])))}


def add_synthetic_candidates(c, root, subject_row, variants, *, check=lambda: None):
    targets = []
    ontology = {1: "Synthetic core", 2: "Synthetic rim", 3: "Synthetic edema"}
    for variant in variants:
        check()
        synthetic = synthetic_source(subject_row, variant)
        layer_id = f"synthetic-{variant:02d}"
        target = add_synthetic_target(c, root, synthetic, layer_id,
                                      f"Synthetic target {variant:02d} · core + rim", candidate=True)
        label_path = Path(target["label_source"])
        if file_hash(label_path) != target["label_sha256"]:
            raise ValueError("Exact synthetic class source changed")
        image = nib.load(label_path)
        classes = np.asanyarray(image.dataobj)
        if not set(np.unique(classes)).issubset({0, *ontology}):
            raise ValueError("Unexpected EDEN synthetic class code")
        classes = classes.astype(np.uint8)
        for value, name in ontology.items():
            class_id = f"{layer_id}-class-{value}"
            c.add_layer(classes, image.affine, layer_id=class_id, source_series="t1",
                        kind="tumor", role="context", color="#ee92d8",
                        label=f"{name} {variant:02d}", label_value=value,
                        provenance={"kind": "simulated", "source_case": synthetic["case_id"],
                                    "source": c.original(label_path), "label_ontology": ontology,
                                    "healthy_anatomy_unchanged": True, "clinical_validation": False,
                                    "planning_target_candidate": False})
        targets.append(target)
    return targets


def prepare(root, subject, *, variant=1, output=None, candidate_variants=(), check=lambda: None):
    prepared = root / SOURCE_COLLECTION
    complete, subject_row = subject_sources(root, subject)
    synthetic = synthetic_source(subject_row, variant)
    output = output if output is not None else root / LEGACY_STUDIES / subject
    if (output / "patient-anatomy.json").exists():
        c = PatientImagingCollection(output)
        if c.document.get("preparation", {}).get("synthetic_source_case") != synthetic["case_id"]:
            raise ValueError("Existing study uses another target; preserve a distinct version")
        print(f"Reusing {output / 'patient-anatomy.json'}", flush=True)
        return json.loads((output / "source-map.json").read_text())
    members, extraction_receipt = extraction_members(root, subject)
    timepoint = None if candidate_variants else "preoperative"
    c = PatientImagingCollection.create(output, "EDEN2020:" + subject, timepoint=timepoint)
    c.document.update({"label": f"EDEN2020 {subject} · anatomical planning · synthetic target",
                       "source_url": URL, "source_license": LICENSE, "role": "DEVELOPMENT",
                       "clinical_validation": False, "training_admitted": False, "limitations": LIMITS,
                       "missing": ["CT-measured skull", "dura", "functional mapping", "complete vascular anatomy",
                                   "SWI-derived venous segmentation", "independently measured registration accuracy"]})
    source_map = {"schema": "eden2020-planning-source-map-v1", "subject": subject, "source_url": URL,
                  "collection": str(output / "patient-anatomy.json"), "extraction_receipt": str(extraction_receipt),
                  "existing_baseline": subject_row["projects"][0]["path"], "series": [], "tractography": [],
                  "limitations": LIMITS}
    source_specs = [("t1", "T1", "T1_3D_PROSET_Sag"), ("tof", "TOF", "s3DI_MC_HR"),
                    ("flair", "FLAIR", "3D_FLAIR_Tra"), ("swi", "SWI", "SWIp_axial")]
    fixed_shape, fixed_affine = vox2out_vox(
        nib.load(members[f"{subject}_T1_3D_PROSET_Sag.nii.gz"]["absolute_path"]),
        subject_row["projects"][0]["spacing_mm"],
    )
    for sid, modality, suffix in source_specs:
        check()
        member = members[f"{subject}_{suffix}.nii.gz"]
        path = Path(member["absolute_path"])
        print(f"Importing unchanged native {subject} {modality}: {nib.load(path).shape}", flush=True)
        row = import_nifti_series(c, path, series_id=sid, modality=modality, timepoint=timepoint)
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
            c.set_registration(sid, receipt["moving_RAS_to_fixed_RAS_forward"],
                               status="unreviewed" if candidate_variants else "source-provided",
                               method="Imported cached rigid estimate: " + receipt["method"])
            c.document["transforms"][sid].update({"review_status": "unreviewed cached rigid estimate",
                                                 "clinical_alignment_validated": False,
                                                 "source_receipt": str(receipt_path), "source_receipt_sha256": file_hash(receipt_path),
                                                 "source_header_conflict_retained": receipt.get("source_header_conflict_retained", False),
                                                 "meaning": "source-bound pipeline estimate; independent accuracy unmeasured"
                                                 if candidate_variants else "source-provided pipeline estimate; source binding verified, independent accuracy unmeasured"})
            if candidate_variants:
                transform = c.document["transforms"][sid]
                transform["quality"] = {
                    "status": "warning", "warnings": ["Cached estimate needs alignment review"],
                    "metrics": {"coverage": receipt.get("brain_scan_coverage_fraction")},
                    "meaning": "Prior algorithm and source identities verified; display alignment only",
                }
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
    if candidate_variants:
        common.update(preparation_cache_verified=True, source_sha256=head_receipt["source_sha256"],
                      output_sha256=head_receipt["output_sha256"], model_identity=head_receipt["identity"],
                      inference_receipt_sha256=file_hash(head / "inference.json"),
                      inference_receipt_original=c.original(head / "inference.json"),
                      source_array_sha256=c.series("t1")["data"]["sha256"])
    def layer(data, layer_id, label, *, kind="context", role="context", color="#91b8c9", native_affine=None, source_series="t1", provenance=None):
        row = c.add_layer(data, image.affine if native_affine is None else native_affine,
                          layer_id=layer_id, source_series=source_series, kind=kind, role=role,
                          color=color, label=label, provenance=deepcopy(common if provenance is None else provenance))
        row["units_status"] = "mm"
        return row
    brain = np.isin(labels, list(BRAIN_IDS))
    brain_row = layer(brain.astype(np.uint8), "brain",
                      "Brain envelope · T1 model estimate" if candidate_variants else "Brain · T1 model estimate",
                      kind="brain", color="#a3bed1")
    head_envelope = ndi.binary_fill_holes(labels > 0)
    envelope_row = layer(head_envelope.astype(np.uint8), "head-surface",
                         "Head surface · T1 model estimate · no CT", color="#bead98")
    lut = root / "cache/models/supersynth-source/FreeSurferColorLUT.txt"
    if candidate_variants:
        lut_sha256 = file_hash(lut)
        brain_row["provenance"].update(
            prepared_array_sha256=brain_row["data"]["sha256"], evidence_use="context_only",
            label_ontology={"name": "SuperSynth composite brain envelope",
                            "value": 1, "source_codes": sorted(BRAIN_IDS), "lut_sha256": lut_sha256},
        )
        envelope_row["provenance"].update(
            prepared_array_sha256=envelope_row["data"]["sha256"], evidence_use="context_only",
            label_ontology={"name": "SuperSynth head envelope from positive labels and closed internal holes",
                            "value": 1, "lut_sha256": lut_sha256},
        )
    names = {}
    for line in lut.read_text().splitlines():
        fields = line.split()
        if len(fields) > 5 and fields[0].isdigit():
            names[int(fields[0])] = (fields[1], "#" + "".join(f"{int(v):02x}" for v in fields[2:5]))
    shared = c.put_array(labels)
    typed_regions = {7: "cerebellar-white-matter", 8: "cerebellar-cortex", 16: "brainstem",
                     46: "cerebellar-white-matter", 47: "cerebellar-cortex",
                     911: "scalp", 915: "skull", 916: "skull"}
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
        if candidate_variants:
            row["kind"] = typed_regions.get(code, row["kind"])
            if code in {7, 8, 46, 47}:
                row["role"] = "critical"
            row["provenance"] = {
                **common, "prepared_array_sha256": shared["sha256"], "evidence_use": "context_only",
                "label_ontology": {"name": name, "value": code, "lut_sha256": lut_sha256},
            }
        c.document["layers"].append(row)
    del labels

    work = prepared / "subjects" / subject
    vascular_receipt = work / "vesselfm-0p6mm/inference.json"
    inference = json.loads(vascular_receipt.read_text())
    vessel = next(r for r in inference["outputs"] if r["name"].endswith("_mask.nii.gz"))
    vessel_path = Path(vessel["path"])
    if file_hash(vessel_path) != vessel["sha256"]:
        raise ValueError("Existing vesselFM mask identity differs")
    vascular = nib.load(vessel_path)
    vessel_layer_id = "tof-vessel-candidates" if candidate_variants else "tof-arteries"
    row = layer((np.asarray(vascular.dataobj) > 0).astype(np.uint8), vessel_layer_id,
                "Unclassified vessels · TOF vesselFM estimate" if candidate_variants else "Arteries · TOF vesselFM estimate",
                kind="vessels" if candidate_variants else "artery", role="critical",
                color="#f06269", native_affine=vascular.affine, source_series="tof",
                provenance={"kind": "model_estimate", "source": c.original(vessel_path), "modality": "TOF",
                            "inference_receipt": str(work / "vesselfm-0p6mm/inference.json"),
                            "clinical_validation": False, "complete_vessels": False})
    if candidate_variants:
        preprocessing_path = work / "vesselfm-0p6mm/input.json"
        preprocessing = json.loads(preprocessing_path.read_text())
        if (preprocessing["source_sha256"] != c.series("tof")["original"]["sha256"]
                or preprocessing["output_sha256"] != inference["input"]["sha256"]):
            raise ValueError("Existing VesselFM result does not bind the acquired TOF")
        row["provenance"].update(
            preparation_cache_verified=True, source_sha256=preprocessing["source_sha256"],
            output_sha256=vessel["sha256"], model=inference["model_id"], model_identity=inference["model"],
            weights=inference["weights"], inference_receipt_sha256=file_hash(vascular_receipt),
            preprocessing_receipt_sha256=file_hash(preprocessing_path),
            inference_receipt_original=c.original(vascular_receipt),
            preprocessing_receipt_original=c.original(preprocessing_path),
            prepared_array_sha256=row["data"]["sha256"],
            source_array_sha256=c.series("tof")["data"]["sha256"], evidence_use="positive_labels_only",
            label_ontology={"name": "vesselFM binary unclassified vessel candidates", "value": 1},
        )
    row["coverage"] = "Partial acquired TOF field; model sensitivity and unlabeled vessels unassessed"
    row["review_status"] = "unreviewed model and cached rigid registration"
    del vascular

    if candidate_variants:
        source_map["synthetic_candidates"] = add_synthetic_candidates(
            c, root, subject_row, candidate_variants, check=check)
        source_map["synthetic_source"] = source_map["synthetic_candidates"][0]
    else:
        source_map["synthetic_source"] = add_synthetic_target(
            c, root, synthetic, "synthetic-target", "Synthetic target · existing core + rim")

    flair = nib.load(members[f"{subject}_3D_FLAIR_Tra.nii.gz"]["absolute_path"])
    trks = [members[name] for name in sorted(members) if name.endswith("_to_3DFLAIR.trk")]
    for index, member in enumerate(trks):
        check()
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
    c.document["display"] = {"initial_layers": ["brain", source_map["synthetic_source"]["layer_id"], vessel_layer_id], "cursor_ras_mm": center,
                              "entry_ras_mm": superior_entry(head_envelope, image.affine, center),
                              "entry_basis": "local superior outer estimated head surface; editable geometric starting point, no skull opening inferred",
                              "layer_opacity": {"brain": 0.18, source_map["synthetic_source"]["layer_id"]: 0.9, vessel_layer_id: 0.85},
                              "source_series": "t1", "head_surface_available": True, "measured_skull_available": False}
    c.document["preparation"] = {"version": "eden2020-planning-v1", "subject": subject,
                                 "source_subject_manifest": str(complete), "synthetic_source_case": synthetic["case_id"],
                                 "synthetic_source_sha256": synthetic["sha256"], "registration_recomputed": False,
                                 "model_inference_run": False, "source_resolution_preserved": True}
    if candidate_variants:
        from resectionlab.anatomy_alignment import input_key
        for sid in ("tof", "flair", "swi"):
            transform = c.document["transforms"][sid]
            transform["input_key"] = input_key(c, c.series(sid), c.series("t1"))
            c.registration_matrix(sid)
    check()
    source_map["snapshot_id"] = c.save("Reuse EDEN sources, cached estimates, supplied named tracts and separate synthetic target")
    source_map["display"] = c.document["display"]
    source_map["source_assets"] = assets
    source_map["layer_count"] = len(c.document["layers"])
    atomic_json(output / "source-map.json", source_map)
    if output == root / LEGACY_STUDIES / subject:
        print(f"Saved {output / 'patient-anatomy.json'}", flush=True)
    return source_map


def package_members(document):
    """Collect portable content-addressed arrays and originals used by this snapshot."""
    members = {}
    pending = [document]
    while pending:
        item = pending.pop()
        if isinstance(item, list):
            pending.extend(item)
        elif isinstance(item, dict):
            name = item.get("path")
            if isinstance(name, str) and name.startswith(("arrays/", "originals/")):
                identity = item["sha256"]
                if identity != members.setdefault(name, identity):
                    raise ValueError("Conflicting package member identity")
            pending.extend(item.values())
    return members


def copy_immutable_study(source, destination):
    source_map = json.loads((source / "source-map.json").read_text())
    identity = source_map["snapshot_id"]
    if not re.fullmatch(r"[a-f0-9]{64}", identity):
        raise ValueError("Invalid frozen source snapshot identity")
    revision_name = f"revisions/{identity}.json"
    revision = safe_path(source, revision_name)
    document = json.loads(revision.read_text())
    collection = PatientImagingCollection(source, document)
    if document["snapshot_id"] != identity:
        raise ValueError("Existing source study does not match the frozen source map")
    output = PatientImagingCollection(destination, document)
    for name, expected_hash in package_members(document).items():
        member = output.path(name)
        member.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(collection.path(name), member)
        if file_hash(member) != expected_hash:
            raise ValueError(f"Copied anatomy member does not match its frozen identity: {name}")
    immutable_revision = output.path(revision_name)
    immutable_revision.parent.mkdir(exist_ok=True)
    shutil.copyfile(revision, immutable_revision)
    return output, source_map


def target_receipt(c, root, synthetic, layer_id):
    row = c.layer(layer_id)
    source = row["provenance"]["source"]
    _, label_hash = verified_target_label(root, synthetic)
    if (row["source_series"] != "t1" or row["role"] != "target" or row["kind"] != "tumor"
            or row["provenance"]["source_case"] != synthetic["case_id"]
            or row["provenance"]["source_package_sha256"] != synthetic["sha256"]
            or source["sha256"] != label_hash or file_hash(c.path(source["path"])) != label_hash):
        raise ValueError("Existing study target does not match the selected archived target")
    c.array(row["data"])  # Check the copied derived array identity without rebuilding it.
    return target_source_receipt(root, synthetic, row)


def prepare_comparison(root, subject, output):
    _, subject_row = subject_sources(root, subject)
    sources = {layer_id: synthetic_source(subject_row, variant)
               for variant, layer_id in COMPARISON_TARGETS.items()}
    if (output / "patient-anatomy.json").is_file():
        c = PatientImagingCollection(output)
        source_map = json.loads((output / "source-map.json").read_text())
        targets = [target_receipt(c, root, synthetic, layer_id)
                   for layer_id, synthetic in sources.items()]
        if (c.document["patient_id"] != f"EDEN2020:{subject}"
                or c.document["snapshot_id"] != source_map["snapshot_id"]
                or source_map.get("comparison_targets") != targets):
            raise ValueError("Existing comparison study differs from its preparation receipt")
        print(f"Reusing {output / 'patient-anatomy.json'}", flush=True)
        return source_map

    if output.exists() or output.is_symlink():
        raise ValueError(f"Preserve existing comparison output: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=output.parent, prefix=f".{subject}-") as temp:
        staged = Path(temp) / subject
        staged.mkdir()
        existing = root / LEGACY_STUDIES / subject
        if (existing / "source-map.json").is_file():
            c, source_map = copy_immutable_study(existing, staged)
            reused_snapshot = c.document["snapshot_id"]
        else:
            source_map = prepare(root, subject, output=staged)
            c = PatientImagingCollection(staged)
            reused_snapshot = None
        if c.document["patient_id"] != f"EDEN2020:{subject}":
            raise ValueError("Existing source study belongs to a different volunteer")
        primary_id = COMPARISON_TARGETS[1]
        primary = target_receipt(c, root, sources[primary_id], primary_id)
        c.layer(primary_id)["label"] = "Synthetic target 01 · existing core + rim"
        secondary_id = COMPARISON_TARGETS[8]
        secondary = add_synthetic_target(c, root, sources[secondary_id], secondary_id,
                                         "Synthetic target 08 · existing core + rim")
        targets = [primary, secondary]

        display = c.document["display"]
        head = c.layer("head-surface")
        head_mask = c.mask(head)
        head_affine = c.patient_affine(head)
        display["planning_targets"] = {
            item["layer_id"]: {
                "cursor_ras_mm": item["recipe"]["center_ras_mm"],
                "entry_ras_mm": superior_entry(head_mask, head_affine, item["recipe"]["center_ras_mm"]),
            }
            for item in targets
        }
        c.document["label"] = f"EDEN2020 {subject} · anatomical planning · synthetic targets 01 and 08"
        c.document["preparation"]["comparison_targets"] = targets
        c.document["preparation"]["reused_native_snapshot"] = reused_snapshot
        source_map.update(collection=str(output / "patient-anatomy.json"), display=display,
                          layer_count=len(c.document["layers"]), comparison_targets=targets,
                          reused_native_snapshot=reused_snapshot)
        source_map["snapshot_id"] = c.save("Add a separate existing EDEN synthetic target for anatomical comparison")
        atomic_json(staged / "source-map.json", source_map)
        os.rename(staged, output)
    print(f"Saved {output / 'patient-anatomy.json'}", flush=True)
    return source_map


def register_comparison_studies(maps, library):
    library.mkdir(parents=True, exist_ok=True)
    for item in maps:
        source = Path(item["collection"]).resolve(strict=True)
        link = library / f"EDEN2020-{item['subject']}-approach-comparison.anatomy.json"
        try:
            link.symlink_to(source)
        except FileExistsError:
            if not link.is_symlink() or link.resolve() != source:
                raise ValueError(f"Preserve existing prepared library entry: {link}") from None


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path)
    parser.add_argument("--subjects", nargs="+")
    parser.add_argument("--variant", type=int)
    parser.add_argument("--comparison-output", type=Path, help="Create isolated studies with existing variants 01 and 08")
    parser.add_argument("--app-prepared-root", type=Path, help="Register comparison studies in an isolated Cases profile")
    args = parser.parse_args()
    if args.variant is not None and args.variant not in range(1, 9):
        parser.error("Choose an existing synthetic variant 1–8")
    if args.comparison_output and args.variant is not None:
        parser.error("Comparison studies always include predeclared variants 01 and 08")
    if args.app_prepared_root and not args.comparison_output:
        parser.error("Use --app-prepared-root with --comparison-output")
    subjects = args.subjects or (COMPARISON_SUBJECTS if args.comparison_output else COMPARISON_SUBJECTS[:2])
    if len(set(subjects)) != len(subjects):
        parser.error("Choose each subject once")
    root = checked_root(args.root, required_bytes=0 if args.comparison_output else len(subjects) * 2 * 1024**3)
    collection_schema = "eden2020-planning-collection-v1"
    existing_studies = {}
    if args.comparison_output:
        output = args.comparison_output.resolve()
        for protected in (root / LEGACY_STUDIES, root / SOURCE_COLLECTION):
            protected = protected.resolve()
            if output.is_relative_to(protected) or protected.is_relative_to(output):
                parser.error("Choose a comparison output separate from the existing source collections")
        collection_file = output / "collection.json"
        if collection_file.exists():
            existing = json.loads(collection_file.read_text())
            rows = existing.get("studies")
            if (existing.get("schema") != collection_schema or not isinstance(rows, list) or not rows
                    or not all(isinstance(item, dict) and "subject" in item and "targets" in item for item in rows)):
                parser.error("The comparison output already contains a different collection")
            existing_studies = {item["subject"]: item for item in rows}
        for subject in subjects:
            subject_sources(root, subject)
        maps = [prepare_comparison(root, subject, output / subject) for subject in subjects]
        if args.app_prepared_root:
            register_comparison_studies(maps, args.app_prepared_root.resolve())
    else:
        output = root / LEGACY_STUDIES
        output.mkdir(parents=True, exist_ok=True)
        maps = [prepare(root, subject, variant=args.variant or 1) for subject in subjects]
    existing_studies.update({
        item["subject"]: {"subject": item["subject"], "path": item["collection"], "snapshot_id": item["snapshot_id"],
                          "tract_count": len(item["tractography"]),
                          **({"targets": item["comparison_targets"]} if args.comparison_output else {})}
        for item in maps
    })
    collection_file = output / "collection.json"
    document = {"schema": collection_schema,
                "primary": maps[0]["collection"], "studies": list(existing_studies.values()),
                "source_url": URL, "limitations": LIMITS}
    if not collection_file.is_file() or json.loads(collection_file.read_text()) != document:
        atomic_json(collection_file, document)


if __name__ == "__main__":
    main()
