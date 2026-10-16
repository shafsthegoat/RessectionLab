#!/usr/bin/env python3
"""Reproducible native sub-06 packaging. Inputs remain unchanged; stages are cached.

Run with --help. Model runs are separately recorded by resectionlab.poc_models.
No reference segmentations are supplied for this patient: counts are NOT accuracy.
"""

import argparse
import json
from pathlib import Path
import nibabel as nib
import numpy as np
from scipy import ndimage as ndi
from resectionlab.patient_anatomy import (
    PatientImagingCollection,
    reconstruction_lock,
    file_hash,
)
from resectionlab.anatomy_geometry import mesh, vessel_graph


def build(args):
    output = Path(args.output)
    if (output / "patient-anatomy.json").exists():
        c = PatientImagingCollection(output)
    else:
        c = PatientImagingCollection.create(output, "ZENODO-17486320-meningioma-sub-06")
        scan = Path(args.source) / "scans"
        if not scan.exists():
            scan = Path(args.source)
        for name, filename, modality in [
            ("t1ce", "CE-T1W.nii.gz", "T1CE"),
            ("ct-bone", "CT-BONE.nii.gz", "CT"),
            ("ct-brain", "CT-BRAIN.nii.gz", "CT"),
        ]:
            c.add_series(
                scan / filename,
                series_id=name,
                modality=modality,
                patient_id=c.document["patient_id"],
            )
        # Both CT series remain independent until a source-bound transform is reviewed.
        for name, recordfile, color, kind, role in [
            ("vessels", args.vessels, "#ef5679", "vessels", "critical"),
            ("brain", args.brain, "#91c4b6", "brain", "context"),
        ]:
            record = json.loads(Path(recordfile).read_text())
            for item in record["outputs"]:
                if file_hash(item["path"]) != item["sha256"]:
                    raise ValueError("Model output changed")
            item = next(x for x in record["outputs"] if "mask" in Path(x["path"]).name)
            image = nib.load(item["path"])
            data = np.asarray(image.dataobj) > 0
            probability = None
            if record.get("probability"):
                p = record["probability"]
                if file_hash(p["path"]) != p["sha256"]:
                    raise ValueError("Probability output changed")
                probability = nib.load(p["path"]).get_fdata(dtype=np.float32)
            c.add_layer(
                data.astype(np.uint8),
                image.affine,
                layer_id=name,
                source_series="t1ce",
                kind=kind,
                role=role,
                color=color,
                probability=probability,
                provenance={
                    "inference": record,
                    "policy": "pretrained-estimates-allowed-v1",
                },
            )
        probability = nib.load(args.tumor_probability)
        p = probability.get_fdata(dtype=np.float32)
        c.add_layer(
            (p >= 0.5).astype(np.uint8),
            probability.affine,
            layer_id="tumor",
            source_series="t1ce",
            kind="tumor",
            role="target",
            color="#c56df1",
            probability=p,
            provenance={
                "model": "Raidionics meningioma proposal",
                "probability_source": c.original(args.tumor_probability),
                "policy": "pretrained-estimates-allowed-v1",
                "validation": "unreviewed; source annotations unavailable",
            },
        )
        ct = c.series("ct-bone")
        hu = c.array(ct["data"])
        # Continuous patient CT boundary proposal; retain all components, including non-skull bone.
        c.add_layer(
            (hu >= 300).astype(np.uint8),
            ct["affine"],
            layer_id="bone",
            source_series="ct-bone",
            kind="bone",
            role="context",
            color="#d9d1bb",
            probability=np.clip((hu - 200) / 200, 0, 1),
            provenance={
                "method": "native CT 300 HU boundary proposal",
                "anatomical_scope": "bone-density tissue; includes teeth and other bone, requires skull review",
            },
        )
        c.document["missing"] = [
            "angiographic validation",
            "diffusion tractography",
            "patient functional maps",
            "cranial nerves",
            "reviewed cortical and deep-structure parcellation",
            "reviewed CT-to-MRI registration",
        ]
        c.document["baseline"] = {
            "path": str(Path(args.source).resolve()),
            "preserved": True,
        }
        c.document["admission_policy"] = {
            "id": "pretrained-estimates-allowed-v1",
            "scope": "anatomy proposals; provenance and review required",
            "historical_records": "unchanged",
        }
        c.save("native reconstruction; no ROI clipping or tumor subtraction")
    metrics = {}
    for row in c.document["layers"]:
        geometry = mesh(c, row)
        mask = c.mask(row)
        _, components = ndi.label(mask, structure=np.ones((3, 3, 3)))
        metrics[row["id"]] = {
            "voxels": int(mask.sum()),
            "components_26": int(components),
            "vertices": len(geometry["positions"]),
            "faces": len(geometry["indices"]),
            "max_display_displacement_mm": float(geometry["max_displacement_mm"]),
        }
        if row["kind"] == "vessels":
            graph = vessel_graph(mask, row["affine"])
            np.savez_compressed(
                output / "vessel-geometry.npz",
                **{k: v for k, v in graph.items() if isinstance(v, np.ndarray)},
            )
            metrics[row["id"]]["skeleton_points"] = len(graph["points_ras_mm"])
    (output / "reconstruction-measurements.json").write_text(
        json.dumps(
            {"meaning": "geometry counts, not accuracy", "layers": metrics}, indent=2
        )
        + "\n"
    )
    print(
        json.dumps(
            {"path": str(output), "snapshot_id": c.identity(), "metrics": metrics},
            indent=2,
        )
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ["source", "output", "vessels", "brain", "tumor-probability"]:
        parser.add_argument("--" + name, required=True)
    with reconstruction_lock():
        build(parser.parse_args())
