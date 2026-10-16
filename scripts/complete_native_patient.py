"""Resume native previews and retain negative regression/overlap evidence."""

from pathlib import Path
import json
import nibabel as nib
import numpy as np
from scipy import ndimage as ndi
from skimage.filters import threshold_otsu
from resectionlab.patient_anatomy import (
    PatientImagingCollection,
    file_hash,
    reconstruction_lock,
)
from resectionlab.anatomy_geometry import mesh

c = PatientImagingCollection(
    "/Users/shaf/Downloads/RessectionLab-Native-Anatomy-sub-06"
)
with reconstruction_lock():
    old = c.identity()
    for source in c.document["series"]:
        if "preview_pyramid" not in source:
            c.preview_pyramid(source["id"])
    vessel = c.mask(c.layer("vessels"))
    brain = c.mask(c.layer("brain"))
    tumor = c.mask(c.layer("tumor"))
    c.document["overlap_review"] = {
        "vessels_outside_brain_voxels": int((vessel & ~brain).sum()),
        "vessel_tumor_overlap_voxels": int((vessel & tumor).sum()),
        "deleted_voxels": 0,
        "meaning": "disagreement review, not proof of prediction error",
    }
    if not any(x["id"] == "vascular-review" for x in c.document["layers"]):
        state = np.zeros(vessel.shape, np.uint8)
        state[vessel & ~brain] = 1
        state[vessel & tumor] = 2
        c.add_layer(
            state,
            c.layer("vessels")["affine"],
            layer_id="vascular-review",
            source_series="t1ce",
            kind="review-region",
            role="context",
            color="#ffad54",
            label="Vessels outside brain proposal",
            label_value=1,
            provenance={"purpose": "review excluded regions; vessel layer unchanged"},
        )
        c.add_layer(
            state,
            c.layer("vessels")["affine"],
            layer_id="vascular-tumor-overlap",
            source_series="t1ce",
            kind="review-region",
            role="context",
            color="#fff08e",
            label="Vessel / tumor overlap",
            label_value=2,
            provenance={"purpose": "review overlap; neither structure deleted"},
        )
    base = Path("/Users/shaf/Downloads/RessectionLab-Full-Anatomy-Patient")
    failed = base / "derived/raidionics-native-output/pred_Tumor.nii.gz"
    p = nib.load(failed).get_fdata(dtype=np.float32)
    regression = {
        "patient": "ZENODO-17486320-meningioma-sub-13",
        "input_sha256": file_hash(base / "scans/CE-T1W.nii.gz"),
        "native_probability_sha256": file_hash(failed),
        "maximum_score": float(p.max()),
        "voxels_at_publisher_threshold": int((p >= 0.5).sum()),
        "status": "failed tumor localization retained for regression; do not substitute enhancing regions",
        "source": str(failed),
    }
    c.document["regression_cases"] = [regression]
    c.document["license"] = "CC-BY-NC-4.0"
    c.document["dataset"] = "Zenodo 17486320"
    if not any(r["id"] == "scalp-envelope" for r in c.document["layers"]):
        source = c.series("t1ce")
        image = c.array(source["data"])
        threshold = float(threshold_otsu(image.ravel()[::32])) * 0.35
        envelope = ndi.binary_fill_holes(image > threshold)
        c.add_layer(
            envelope.astype(np.uint8),
            source["affine"],
            layer_id="scalp-envelope",
            source_series="t1ce",
            kind="head-envelope",
            role="context",
            color="#d1ad95",
            label="Head / scalp envelope proposal",
            provenance={
                "method": "native full-head intensity support with enclosed holes filled",
                "threshold": threshold,
                "scope": "head envelope; does not identify histological scalp layers",
            },
        )
    c.layer("bone")["score_meaning"] = (
        "continuous CT boundary field; not a model probability"
    )
    # Preserve the historical rigid estimate as a review candidate. Its sampling
    # direction is explicit and it never becomes fusion authority automatically.
    import SimpleITK as sitk

    path = base / "candidate-sub-06/derived/MRI-to-CT-sampling-transform.tfm"
    if path.exists() and "ct-bone" not in c.document["transforms"]:
        t = sitk.ReadTransform(str(path))
        origin = np.array(t.TransformPoint((0.0, 0.0, 0.0)))
        m = np.eye(4)
        m[:3, 3] = origin
        for i in range(3):
            m[:3, i] = np.array(t.TransformPoint(tuple(np.eye(3)[i]))) - origin
        lps = np.diag([-1.0, -1.0, 1.0, 1.0])
        c.set_registration(
            "ct-bone",
            lps @ np.linalg.inv(m) @ lps,
            status="unreviewed",
            method="archived SimpleITK CT-to-MRI estimate; landmark review pending",
        )
        c.document["transforms"]["ct-bone"]["source_transform"] = c.original(path)
    c.save(
        "native previews, overlap review and preserved negative regression",
        expected=old,
    )
    for name in ["tumor", "vessels", "brain"]:
        geometry = mesh(c, c.layer(name))
        print(name, len(geometry["positions"]), float(geometry["max_displacement_mm"]))
    (c.root / "regression-sub13.json").write_text(
        json.dumps(regression, indent=2) + "\n"
    )
    print(c.identity())
