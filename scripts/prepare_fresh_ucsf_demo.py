#!/usr/bin/env python3
"""Bounded, source-pinned UCSF development import and partial surgical scene.

Use the same command for every predeclared subject. Source mirrors are never
modified; all acquired bytes and receipts stay under ignored testcases/.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import re
import sys
import time
import urllib.parse
import urllib.request

import nibabel as nib
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from resectionlab.anatomy_import import import_anatomy_derivative, import_nifti_series
from resectionlab.anatomy_models import import_proposal, reconstruct
from resectionlab.imaging import load_case
from resectionlab.patient_anatomy import PatientImagingCollection, file_hash
from resectionlab.surgical_patient import prepare_surgical_environment

SOURCE_ROOT = ROOT / "testcases/downloaded-datasets/UCSF-fresh-development-20261026"
OUTPUT_ROOT = ROOT / "testcases/outputs/integrated-goal-20261026/fresh-patients"
MIRROR = "MedOtter/UCSF-PDGM"
SEQUENCES = {"T1c": "T1CE", "T1": "T1", "T2": "T2", "FLAIR": "FLAIR"}
LABEL_MEANINGS = {1: "context", 2: "context", 4: "tumor"}
SOURCE_LABEL_NAMES = {
    1: "Non-enhancing tumor core / necrosis · source label 1",
    2: "Peritumoral edema · source label 2",
    4: "Enhancing tumor · source label 4",
}


def write_json(path: Path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def declared(case):
    declaration = json.loads((SOURCE_ROOT / "development-declaration.json").read_text())
    if (declaration["role"] != "DEVELOPMENT" or case not in declaration["cases"]
            or declaration["mirror_repository"] != MIRROR):
        raise ValueError("Case lacks a frozen DEVELOPMENT declaration")
    return declaration


def declare_development(cases, rationale):
    if (not cases or len(set(cases)) != len(cases)
            or any(not re.fullmatch(r"UCSF-PDGM-[0-9]{4}", case) for case in cases)):
        raise ValueError("Declare distinct UCSF-PDGM four-digit subject IDs explicitly")
    path = SOURCE_ROOT / "development-declaration.json"
    if path.exists():
        previous = json.loads(path.read_text())
        if (previous.get("cases") != cases or previous.get("role") != "DEVELOPMENT"
                or previous.get("mirror_revision") != "e9372219cf1cd2fdd52260cd45f7514b4aa7638e"
                or (rationale and previous.get("selection_rule") != rationale)):
            raise ValueError("Existing development declaration differs; preserve its original pre-pixel assignment")
        return path
    if not rationale or not rationale.strip():
        raise ValueError("Record a pre-pixel selection rationale before declaring cases")
    declaration = {
        "schema": "resectionlab.fresh-source-development-declaration.v1",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "mirror_repository": MIRROR,
        "mirror_revision": "e9372219cf1cd2fdd52260cd45f7514b4aa7638e",
        "mirror_url": "https://huggingface.co/datasets/MedOtter/UCSF-PDGM",
        "cases": cases, "role": "DEVELOPMENT", "selection_rule": rationale,
        "purpose": "fresh-patient integration smoke test, no clinical or generalization claim",
        "image_fetch_cap_bytes": 60000000,
        "official_tcia_byte_equivalence": "unknown",
        "potential_BraTS_pretraining_overlap": "unknown",
    }
    write_json(path, declaration)
    return path


def source_listing(case, revision):
    url = f"https://huggingface.co/api/datasets/{MIRROR}/tree/{revision}/{case}?recursive=true&expand=true"
    with urllib.request.urlopen(url, timeout=30) as response:
        rows = json.load(response)
    expected = {f"{case}/{case}_{suffix}.nii.gz" for suffix in (*SEQUENCES, "tumor_segmentation")}
    if {row.get("path") for row in rows} != expected or len(rows) != 5:
        raise ValueError("Pinned mirror case does not contain exactly four MRI volumes and one mask")
    for row in rows:
        info = row.get("lfs") or {}
        if (row.get("type") != "file" or row.get("size") != info.get("size")
                or not isinstance(info.get("oid"), str) or len(info["oid"]) != 64):
            raise ValueError("Missing exact LFS size/SHA-256 for source image")
    return sorted(rows, key=lambda row: row["path"])


def reusable_files(sizes):
    """Check identical existing content by size and SHA before any new image GET."""
    candidates = {}
    for root, directories, files in os.walk(ROOT / "testcases", followlinks=False):
        directories[:] = [name for name in directories if not (Path(root) / name).is_symlink()]
        for name in files:
            path = Path(root) / name
            try:
                if path.stat().st_size in sizes:
                    candidates.setdefault(path.stat().st_size, []).append(path)
            except OSError:
                continue
    return candidates


def acquire(case, declaration):
    revision = declaration["mirror_revision"]
    rows = source_listing(case, revision)
    listed_size = sum(row["size"] for row in rows)
    if listed_size > declaration["image_fetch_cap_bytes"]:
        raise ValueError("Pinned image set exceeds declaration byte cap")
    candidates = reusable_files({row["size"] for row in rows})
    destination = SOURCE_ROOT / case
    destination.mkdir(parents=True, exist_ok=True)
    receipt = []
    downloaded = 0
    resident_bytes = sum(path.stat().st_size for path in SOURCE_ROOT.rglob("*.nii.gz")
                         if path.is_file() and not path.is_symlink())
    for row in rows:
        relative = row["path"]
        expected = row["lfs"]["oid"]
        target = destination / Path(relative).name
        url = f"https://huggingface.co/datasets/{MIRROR}/resolve/{revision}/{urllib.parse.quote(relative)}"
        if target.exists():
            if target.stat().st_size != row["size"] or file_hash(target) != expected:
                raise ValueError(f"Existing source differs from pinned LFS object: {target}")
            disposition = "existing_verified"
        else:
            match = next((p for p in candidates.get(row["size"], []) if file_hash(p) == expected), None)
            if match is not None:
                target.symlink_to(os.path.relpath(match, target.parent))
                disposition = "reused_identical"
            else:
                if resident_bytes + row["size"] > declaration["image_fetch_cap_bytes"]:
                    raise ValueError("Acquisition would exceed declared byte cap")
                temporary = target.with_name(target.name + ".partial")
                count = 0
                checksum = hashlib.sha256()
                with urllib.request.urlopen(url, timeout=60) as response, temporary.open("wb") as output:
                    while block := response.read(1024 * 1024):
                        count += len(block)
                        if count > row["size"]:
                            raise ValueError("Source transfer exceeded pinned byte length")
                        checksum.update(block)
                        output.write(block)
                if count != row["size"] or checksum.hexdigest() != expected:
                    raise ValueError("Source transfer differs from pinned LFS size/SHA-256")
                temporary.replace(target)
                downloaded += count
                resident_bytes += count
                disposition = "downloaded_verified"
        receipt.append({"path": str(target.relative_to(ROOT)), "remote_path": relative,
                        "url": url, "size_bytes": row["size"], "sha256": expected,
                        "checksum_authority": "pinned_mirror_git_lfs", "disposition": disposition,
                        "official_tcia_byte_equivalence": "unknown"})
    result = {"schema": "resectionlab.fresh-source-acquisition.v1", "patient_id": case,
              "role": "DEVELOPMENT", "mirror_revision": revision, "declared_total_bytes": listed_size,
              "downloaded_bytes_this_run": downloaded, "files": receipt,
              "acquired_at": datetime.now(timezone.utc).isoformat()}
    write_json(destination / "source-receipt.json", result)
    return result


def prepare(case, receipt, *, infer_brain):
    paths = {Path(row["remote_path"]).name.removeprefix(case + "_").removesuffix(".nii.gz"):
             ROOT / row["path"] for row in receipt["files"]}
    if set(paths) != {*SEQUENCES, "tumor_segmentation"}:
        raise ValueError("Source receipt lacks required MRI or segmentation")
    for suffix, path in paths.items():
        record = next(row for row in receipt["files"] if Path(row["remote_path"]).name == path.name)
        if file_hash(path) != record["sha256"]:
            raise ValueError(f"Source changed before native import: {suffix}")
    images = {suffix: nib.load(paths[suffix]) for suffix in SEQUENCES}
    reference = images["T1c"]
    for suffix, image in images.items():
        if image.shape != reference.shape or not np.allclose(image.affine, reference.affine, atol=1e-5):
            raise ValueError(f"Mirror structural geometry differs from T1CE: {suffix}")
        if image.header.get_xyzt_units()[0] != "mm":
            raise ValueError(f"MRI lacks declared millimeter units: {suffix}")
    segmentation = nib.load(paths["tumor_segmentation"])
    if segmentation.shape != reference.shape or not np.allclose(segmentation.affine, reference.affine, atol=1e-5):
        raise ValueError("Source segmentation geometry differs from T1CE")
    if segmentation.header.get_xyzt_units()[0] != "mm":
        raise ValueError("Source segmentation lacks declared millimeter units")
    labels = set(np.unique(np.asanyarray(segmentation.dataobj)).astype(int)) - {0}
    if not labels <= set(LABEL_MEANINGS) or 4 not in labels:
        raise ValueError(f"Unsupported or absent source enhancing label: {sorted(labels)}")
    native = OUTPUT_ROOT / case / "native"
    if (native / "patient-anatomy.json").exists():
        c = PatientImagingCollection(native)
        if c.document["patient_id"] != case:
            raise ValueError("Existing native collection belongs to another person")
    else:
        c = PatientImagingCollection.create(native, case, timepoint="preoperative")
        for suffix in ("T1c", "T1", "T2", "FLAIR"):
            row = import_nifti_series(c, paths[suffix], series_id=SEQUENCES[suffix], modality=SEQUENCES[suffix])
            row["label"] = suffix + " · mirror supplied"
            row["source_mirror_revision"] = receipt["mirror_revision"]
            if suffix != "T1c":
                c.set_registration(row["id"], np.eye(4), status="source-provided",
                    method="Pinned mirror declares co-registered BraTS grid; exact native affines verified; no independent landmark review")
        items = import_anatomy_derivative(c, paths["tumor_segmentation"], source_series="T1CE",
            kind="labels", label="Supplied tumor segmentation",
            role_map={value: {"kind": LABEL_MEANINGS[value], "label": SOURCE_LABEL_NAMES[value]}
                      for value in labels})
        for item in items:
            item["coverage"] = "Mirror supplied BraTS annotation; label extent and preprocessing unreviewed; outside field unassessed"
        c.document["missing"] = ["skull", "vessels", "angiography", "functional mapping", "tractography"]
        c.document["source_limitations"] = ["Skull-stripped structural mirror, not acquired CT or angiography",
            "Mirror-to-official TCIA byte equivalence and BraTS pretraining overlap unknown",
            "Source-provided coregistration has not received independent landmark review"]
        c.save("source-bound fresh development import")
    relabeled = False
    for item in c.document["layers"]:
        value = item.get("provenance", {}).get("source_label_value")
        if value in SOURCE_LABEL_NAMES and item["label"] != SOURCE_LABEL_NAMES[value]:
            item["label"] = SOURCE_LABEL_NAMES[value]
            relabeled = True
    if relabeled:
        c.save("clarify supplied source label meanings")
    target = next((row for row in c.document["layers"] if row["kind"] == "tumor" and row["source_series"] == "T1CE"), None)
    if target is None:
        raise ValueError("No source enhancing tumor target admitted")
    brain = next((row for row in c.document["layers"] if row["kind"] == "brain"), None)
    if infer_brain and brain is None:
        proposal = reconstruct(c, "T1CE", "synthstrip")
        brain_id = import_proposal(c, "T1CE", proposal)
        c.save("unreviewed SynthStrip brain estimate")
        brain = c.layer(brain_id)
    scene_report = None
    if brain is not None:
        scene_report = prepare_surgical_environment(c.root, OUTPUT_ROOT / case / "scene",
            layer_ids={"brain": brain["id"], "tumor": target["id"]})
        loaded = load_case(scene_report["casePath"])
        if not {"brain", "tumor"} <= set(loaded.compartments):
            raise ValueError("Saved scene lost brain or enhancing tumor material")
    reopened = PatientImagingCollection(c.root)
    if reopened.identity() != c.identity():
        raise ValueError("Native collection did not reopen identically")
    report = {"schema": "resectionlab.fresh-development-preparation.v1", "patient_id": case,
              "source_receipt": str((SOURCE_ROOT / case / "source-receipt.json").relative_to(ROOT)),
              "native_manifest": str((native / "patient-anatomy.json").relative_to(ROOT)),
              "native_snapshot_id": c.identity(), "native_reopen_identical": True,
              "source_modalities": [row["modality"] for row in c.document["series"]],
              "source_labels": {str(row["provenance"]["source_label_value"]): {"label": row["label"],
                                "kind": row["kind"], "role": row["role"],
                                "id": row["id"]} for row in c.document["layers"]
                                if "source_label_value" in row.get("provenance", {})},
              "brain_layer_id": brain["id"] if brain else None,
              "brain_review_status": brain.get("review_status") if brain else None,
              "brain_inference_receipt": str(Path(brain["provenance"]["inference"]["outputs"][0]["path"]).parent / "inference.json") if brain else None,
              "brain_model_output_sha256": brain["provenance"]["output_sha256"] if brain else None,
              "enhancing_target_outside_brain_cells": int(np.count_nonzero(c.mask(target) & ~c.mask(brain))) if brain else None,
              "scene_path": scene_report["casePath"] if scene_report else None,
              "scene_reopened": scene_report is not None,
              "scene_materials": list(loaded.compartments) if scene_report else [],
              "scene_semantic_hash": loaded.semantic_hash if scene_report else None,
              "missing_anatomy": c.document["missing"],
              "planning_limitations": c.document["source_limitations"],
              "clinical_validation": "none", "official_tcia_byte_equivalence": "unknown"}
    write_json(OUTPUT_ROOT / case / "preparation-receipt.json", report)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case", action="append", required=True)
    parser.add_argument("--declare-development", action="store_true", help="Write only the pre-pixel DEVELOPMENT declaration")
    parser.add_argument("--selection-rationale", help="Fixed case selection rule recorded before image access")
    parser.add_argument("--package-root", type=Path, default=ROOT / "testcases",
                        help="Ignored testcases root for source bytes and result packages")
    parser.add_argument("--infer-brain", action="store_true", help="Run bounded local SynthStrip and export partial scene")
    args = parser.parse_args()
    package_root = args.package_root.resolve()
    if not package_root.is_relative_to((ROOT / "testcases").resolve()):
        parser.error("Reproduction package root must remain under ignored testcases/")
    global SOURCE_ROOT, OUTPUT_ROOT
    SOURCE_ROOT = package_root / "downloaded-datasets/UCSF-fresh-development-20261026"
    OUTPUT_ROOT = package_root / "outputs/integrated-goal-20261026/fresh-patients"
    if args.declare_development:
        if args.infer_brain:
            parser.error("Declaration is metadata-only and cannot run inference")
        path = declare_development(args.case, args.selection_rationale)
        print(json.dumps({"status": "development_declared_metadata_only", "cases": args.case, "path": str(path)}))
        return
    if len(args.case) != 1:
        parser.error("Import exactly one predeclared case per run")
    started = time.monotonic()
    case = args.case[0]
    declaration = declared(case)
    receipt = acquire(case, declaration)
    report = prepare(case, receipt, infer_brain=args.infer_brain)
    print(json.dumps({"patient_id": case, "elapsed_seconds": time.monotonic() - started,
                      "downloaded_bytes": receipt["downloaded_bytes_this_run"],
                      "native_reopen_identical": report["native_reopen_identical"],
                      "scene_reopened": report["scene_reopened"], "receipt": str(OUTPUT_ROOT / case / "preparation-receipt.json")}))


if __name__ == "__main__":
    main()
