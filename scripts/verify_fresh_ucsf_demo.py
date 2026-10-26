#!/usr/bin/env python3
"""Read-only, exact source/native/scene audit for a fresh UCSF development case."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import nibabel as nib
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from resectionlab.imaging import load_case
from resectionlab.patient_anatomy import PatientImagingCollection, file_hash

SOURCE = ROOT / "testcases/downloaded-datasets/UCSF-fresh-development-20261026"
RESULTS = ROOT / "testcases/outputs/integrated-goal-20261026/fresh-patients"


def verify(case):
    declaration = json.loads((SOURCE / "development-declaration.json").read_text())
    assert case in declaration["cases"] and declaration["role"] == "DEVELOPMENT"
    source = json.loads((SOURCE / case / "source-receipt.json").read_text())
    prepared = json.loads((RESULTS / case / "preparation-receipt.json").read_text())
    c = PatientImagingCollection(RESULTS / case / "native")
    scene_report = json.loads((RESULTS / case / "scene/report.json").read_text())
    scene = load_case(scene_report["casePath"])
    checks = {}

    members = {}
    for row in source["files"]:
        path = ROOT / row["path"]
        assert path.stat().st_size == row["size_bytes"] and file_hash(path) == row["sha256"]
        members[path.name.removeprefix(case + "_").removesuffix(".nii.gz")] = path
    checks["five_pinned_lfs_source_files"] = len(members) == 5
    assert checks["five_pinned_lfs_source_files"]
    suffix_by_modality = {"T1CE": "T1c", "T1": "T1", "T2": "T2", "FLAIR": "FLAIR"}
    for modality, suffix in suffix_by_modality.items():
        row = c.series(modality)
        image = nib.load(members[suffix])
        assert row["units_status"] == "mm"
        assert row["original"]["sha256"] == file_hash(members[suffix])
        assert file_hash(c.path(row["original"]["path"])) == row["original"]["sha256"]
        assert tuple(row["data"]["shape"]) == image.shape
        assert np.array_equal(np.asarray(row["affine"]), image.affine)
        assert np.array_equal(c.array(row["data"]), image.get_fdata(dtype=np.float32))
        assert c.document["transforms"][modality]["status"] == ("reference" if modality == "T1CE" else "source-provided")
    checks["all_four_mri_arrays_and_affines_exact"] = True

    seg = nib.load(members["tumor_segmentation"])
    labels = np.asanyarray(seg.dataobj)
    source_layers = {row["provenance"]["source_label_value"]: row for row in c.document["layers"]
                     if "source_label_value" in row.get("provenance", {})}
    assert set(source_layers) == {int(x) for x in np.unique(labels)} - {0}
    for value, row in source_layers.items():
        assert row["units_status"] == "mm"
        assert row["provenance"]["source"]["sha256"] == file_hash(members["tumor_segmentation"])
        assert file_hash(c.path(row["provenance"]["source"]["path"])) == row["provenance"]["source"]["sha256"]
        assert np.array_equal(np.asarray(row["affine"]), seg.affine)
        assert np.array_equal(c.mask(row), labels == value)
    assert source_layers[4]["kind"] == "tumor" and source_layers[4]["role"] == "target"
    assert all(source_layers[value]["role"] == "context" for value in (1, 2))
    checks["supplied_labels_exact_and_explicit_roles"] = True

    brain = c.layer(prepared["brain_layer_id"])
    model_mask = brain["provenance"]
    assert model_mask["model"] == "synthstrip"
    assert file_hash(model_mask["path"]) == model_mask["output_sha256"]
    assert np.array_equal(c.mask(brain), np.asanyarray(nib.load(model_mask["path"]).dataobj) > 0)
    assert brain["review_status"] == "unreviewed"
    checks["brain_estimate_exact_unreviewed"] = True

    assert c.identity() == prepared["native_snapshot_id"] == scene_report["sourceCollectionHash"]
    assert scene.semantic_hash == prepared["scene_semantic_hash"] == scene_report["caseHash"]
    assert set(scene.compartments) == {"brain", "tumor"}
    assert np.array_equal(scene.compartments["tumor"], labels == 4)
    assert np.array_equal(scene.compartments["brain"], c.mask(brain))
    assert {"skull", "vessels"} <= set(c.document["missing"])
    assert {"bone", "vessels"} <= set(scene.metadata["missing_anatomy"])
    assert not np.any(scene.compartments["tumor"] & ~scene.compartments["brain"])
    checks["scene_reopened_with_exact_source_target_and_estimated_brain"] = True
    checks["skull_vessels_explicitly_missing"] = True
    return {"schema": "resectionlab.fresh-development-independent-verification.v1",
            "patient_id": case, "source_bytes": source["declared_total_bytes"],
            "native_snapshot_id": c.identity(), "scene_semantic_hash": scene.semantic_hash,
            "source_target_cells": int((labels == 4).sum()), "brain_estimate_cells": int(c.mask(brain).sum()),
            "checks": checks, "passed": all(checks.values()),
            "limits": ["Official TCIA source-byte equivalence and pretrained overlap unknown",
                       "Source-provided registration and SynthStrip estimate unreviewed",
                       "No source skull, vessels, angiography or functional anatomy"]}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case", required=True)
    parser.add_argument("--package-root", type=Path, default=ROOT / "testcases")
    args = parser.parse_args()
    package_root = args.package_root.resolve()
    if not package_root.is_relative_to((ROOT / "testcases").resolve()):
        parser.error("Reproduction package root must remain under ignored testcases/")
    global SOURCE, RESULTS
    SOURCE = package_root / "downloaded-datasets/UCSF-fresh-development-20261026"
    RESULTS = package_root / "outputs/integrated-goal-20261026/fresh-patients"
    result = verify(args.case)
    destination = RESULTS / args.case / "independent-verification.json"
    destination.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({"patient_id": args.case, "passed": result["passed"], "receipt": str(destination)}))


if __name__ == "__main__":
    main()
