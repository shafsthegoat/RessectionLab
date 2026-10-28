#!/usr/bin/env python3
"""Audit a bounded tumor estimate without changing the source native study or scene."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import nibabel as nib
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from resectionlab.patient_anatomy import PatientImagingCollection, file_hash

RESULTS = ROOT / "testcases/outputs/integrated-goal-20261026/fresh-patients"


def overlap(proposed, supplied, voxel_mm3):
    intersection = int(np.count_nonzero(proposed & supplied))
    count_a, count_b = int(proposed.sum()), int(supplied.sum())
    return {"model_cells": count_a, "source_annotation_cells": count_b,
            "intersection_cells": intersection,
            "model_volume_mm3": count_a * voxel_mm3,
            "source_annotation_volume_mm3": count_b * voxel_mm3,
            "intersection_volume_mm3": intersection * voxel_mm3,
            "dice_descriptive_only": 2 * intersection / (count_a + count_b) if count_a + count_b else None}


def compare(case, model):
    root = RESULTS / case
    prepared = json.loads((root / "preparation-receipt.json").read_text())
    c = PatientImagingCollection(root / "native")
    scene_report = json.loads((root / "scene/report.json").read_text())
    if c.identity() != prepared["native_snapshot_id"] or scene_report["sourceCollectionHash"] != c.identity():
        raise ValueError("Source native study or original scene changed after inference")
    receipts = [p for p in (root / "native/reconstructions").glob("*/receipt.json")
                if json.loads(p.read_text()).get("model") == model]
    if len(receipts) != 1:
        raise ValueError("Choose one exact model inference receipt")
    receipt = receipts[0]
    record = json.loads(receipt.read_text())
    if record["status"] != "proposal" or record["source"] != c.series("T1CE")["original"]["sha256"]:
        raise ValueError("Model receipt does not bind the expected source T1CE")
    brain = c.layer(prepared["brain_layer_id"])
    if record["brain_mask"]["layer_id"] != brain["id"]:
        raise ValueError("Model used a different brain mask")
    if file_hash(record["path"]) != record["output_sha256"] or file_hash(record["probability"]["path"]) != record["probability"]["sha256"]:
        raise ValueError("Model output changed after inference")
    mask_image = nib.load(record["path"])
    probability_image = nib.load(record["probability"]["path"])
    reference = c.series("T1CE")
    for image in (mask_image, probability_image):
        if image.shape != tuple(reference["data"]["shape"]) or not np.allclose(image.affine, reference["affine"], atol=1e-5):
            raise ValueError("Model output is not on the native source grid")
    mask = np.asanyarray(mask_image.dataobj)
    probability = probability_image.get_fdata(dtype=np.float32)
    if not np.isin(mask, (0, 1)).all() or not np.isfinite(probability).all() or not np.all((0 <= probability) & (probability <= 1)):
        raise ValueError("Model mask/score has unsupported values")
    proposed = mask > 0
    if not np.array_equal(proposed, probability >= .5):
        raise ValueError("Model mask differs from recorded 0.5 score threshold")
    source_layers = {row["provenance"]["source_label_value"]: row for row in c.document["layers"]
                     if "source_label_value" in row.get("provenance", {})}
    enhancing = c.mask(source_layers[4])
    all_supplied = np.logical_or.reduce([c.mask(row) for row in source_layers.values()])
    voxel_mm3 = float(abs(np.linalg.det(np.asarray(reference["affine"])[:3, :3])))
    result = {"schema": "resectionlab.fresh-glioma-model-comparison.v1", "patient_id": case,
              "inference_model": model, "model_receipt": str(receipt),
              "model_receipt_sha256": file_hash(receipt),
              "model_weight_sha256": record["files"]["model.onnx"],
              "source_t1ce_sha256": record["source"],
              "brain_layer_id": brain["id"],
              "native_snapshot_id_before_and_after": c.identity(),
              "original_scene_semantic_hash": prepared["scene_semantic_hash"],
              "model_foreground_cells": int(proposed.sum()),
              "model_outside_estimated_brain_cells": int(np.count_nonzero(proposed & ~c.mask(brain))),
              "against_supplied_enhancing_label_4": overlap(proposed, enhancing, voxel_mm3),
              "against_supplied_all_tumor_labels_1_2_4": overlap(proposed, all_supplied, voxel_mm3),
              "inference_elapsed_seconds": record["elapsed_seconds"],
              "inference_peak_sampled_rss_bytes": record["peak_sampled_rss_bytes"],
              "interpretation": "Descriptive overlap of an unreviewed model tumor proposal with mirror-supplied labels on the same source scan; model 'Tumor' and enhancing label 4 are not proven equivalent targets. Scores are not calibrated probabilities or clinical accuracy.",
              "source_scene_modified": False}
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case", required=True)
    parser.add_argument("--model", default="glioma-gbm")
    parser.add_argument("--package-root", type=Path, default=ROOT / "testcases")
    args = parser.parse_args()
    package_root = args.package_root.resolve()
    if not package_root.is_relative_to((ROOT / "testcases").resolve()):
        parser.error("Reproduction package root must remain under ignored testcases/")
    global RESULTS
    RESULTS = package_root / "outputs/integrated-goal-20261026/fresh-patients"
    result = compare(args.case, args.model)
    destination = RESULTS / args.case / "glioma-model-comparison.json"
    destination.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    print(json.dumps({"patient_id": args.case, "model_cells": result["model_foreground_cells"],
                      "source_scene_modified": result["source_scene_modified"], "receipt": str(destination)}))


if __name__ == "__main__":
    main()
