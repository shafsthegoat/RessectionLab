"""Expose one measured low-b frame for inspection of a retained DWI acquisition."""

from math import prod
from pathlib import Path
from tempfile import TemporaryDirectory

import nibabel as nib
import numpy as np

from .imaging import inspect_nifti
from .patient_anatomy import MAX_ARRAY, digest, file_hash


def add_native_diffusion_baseline(collection, asset_id, *, check=lambda: None):
    """Keep the selected volume on its original lattice with a source-bound receipt."""
    asset = next((row for row in collection.source_records()
                  if row["id"] == asset_id and row.get("storage") == "source-asset"), None)
    if (not asset or asset["modality"] != "DWI" or len(asset["shape"]) != 4
            or not asset["original"]["name"].endswith((".nii", ".nii.gz"))):
        raise ValueError("Choose a retained 4D NIfTI diffusion acquisition")
    bval = asset.get("companions", {}).get("bval")
    if not bval:
        raise ValueError("Diffusion baseline needs the b-values retained with this acquisition")
    source = collection.path(asset["original"]["path"])
    bval_path = collection.path(bval["path"])
    if bval_path.stat().st_size > 1024**2:
        raise ValueError("Diffusion b-values exceed 1 MiB")
    if file_hash(source) != asset["original"]["sha256"] or file_hash(bval_path) != bval["sha256"]:
        raise ValueError("Retained diffusion acquisition or b-values changed")
    check()
    qc = inspect_nifti(source, dimensions=4)
    if (qc["source_units"] != "mm" or qc["shape"] != asset["shape"]
            or not np.array_equal(qc["affine_ras_mm"], asset["affine"])):
        raise ValueError("Diffusion inspection requires the archived, coded millimeter grid")
    if prod(qc["shape"][:3]) * np.dtype(np.float32).itemsize > MAX_ARRAY:
        raise ValueError("Diffusion baseline exceeds the native volume size limit")
    bvalues = np.loadtxt(bval_path, dtype=np.float64).reshape(-1)
    if (len(bvalues) != qc["shape"][3] or len(bvalues) < 2
            or not np.isfinite(bvalues).all() or np.any(bvalues < 0)):
        raise ValueError("Expected one finite, nonnegative b-value per diffusion volume")
    index = int(np.argmin(bvalues))
    if bvalues[index] > 50 or np.all(bvalues <= 50):
        raise ValueError("Expected a b-value of at most 50 s/mm² and a diffusion-weighted volume")
    receipt = {
        "schema": "native-diffusion-baseline-v1", "source_asset_id": asset_id,
        "source_sha256": asset["original"]["sha256"], "bvals_sha256": bval["sha256"],
        "volume_index": index, "b_value_s_per_mm2": float(bvalues[index]),
        "source_volume_count": len(bvalues),
        "low_b_volume_count": int(np.count_nonzero(bvalues <= 50)),
        "method": "first volume at the minimum b-value; native voxel samples without resampling",
        "evidence_use": "diffusion baseline inspection",
    }
    series_id = "diffusion-b0-" + digest(receipt)[:24]
    existing = next((row for row in collection.document["series"] if row["id"] == series_id), None)
    if existing:
        if existing.get("diffusion_baseline") != receipt:
            raise ValueError("Stored diffusion baseline provenance differs")
        derived = existing["original"]
        if file_hash(collection.path(derived["path"])) != derived["sha256"]:
            raise ValueError("Stored diffusion baseline identity changed")
        collection.require_native_geometry(existing)
        collection.array(existing["data"])
        return existing

    image = nib.load(source)
    signal = np.asarray(image.dataobj[..., index], dtype=np.float32)
    if not np.isfinite(signal).all() or np.any(signal < 0) or not np.any(signal > 0):
        raise ValueError("Diffusion baseline requires finite nonnegative source signal with positive samples")
    check()
    if file_hash(source) != receipt["source_sha256"] or file_hash(bval_path) != receipt["bvals_sha256"]:
        raise ValueError("Retained diffusion acquisition or b-values changed during extraction")

    volume = type(image)(signal, image.affine)
    volume.header.set_xyzt_units("mm")
    for form in ("qform", "sform"):
        matrix, code = getattr(image, f"get_{form}")(coded=True)
        getattr(volume, f"set_{form}")(matrix, code=int(code))
    with TemporaryDirectory(prefix=".diffusion-b0-", dir=collection.root) as scratch:
        path = Path(scratch) / "native-diffusion-baseline.nii.gz"
        nib.save(volume, path)
        check()
        row = collection.add_series(path, series_id=series_id, modality="DWI_B0", timepoint=asset["timepoint"])
    row.update(label=(asset.get("label") or asset_id) + " · diffusion baseline", diffusion_baseline=receipt,
               coverage=asset.get("coverage", {}), acquisition_identity=asset.get("acquisition_identity", {}),
               acquisition_datetime=asset.get("acquisition_datetime"))
    return row
