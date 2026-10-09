"""One read-only scan/FOV support computation from saved Case4 transforms."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import ants
import nibabel as nib
import numpy as np
import scipy.ndimage as ndi
import SimpleITK as sitk

from support_contract import encode_support_bits


ROOT = Path(__file__).resolve().parents[3]
HERE = Path(__file__).resolve().parent
PREP = ROOT / "outputs/scan-target/resect-case4-prep-v1/attempt-01"
MANIFEST = ROOT / "manifests/experiments/resect-case4-scan-diagnostic-v1.json"
PREP_RESULT_SHA = "92fdc93f1240c81ac0ecbd7910a8256a18693480bd4bed48b19e109fe308d72c"


def sha(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 << 20), b""):
            value.update(block)
    return value.hexdigest()


def exact(path: Path, expected: str) -> None:
    if not path.is_file() or sha(path) != expected:
        raise ValueError(f"source changed or unavailable: {path}")


def ants_affine_ras(image) -> np.ndarray:
    result = np.eye(4)
    lps_to_ras = np.diag([-1.0, -1.0, 1.0])
    result[:3, :3] = lps_to_ras @ np.asarray(image.direction, float) @ np.diag(image.spacing)
    result[:3, 3] = lps_to_ras @ np.asarray(image.origin, float)
    return result


def checked_dual_reader(path: Path):
    """Require exact voxel indexing and physical-frame agreement for one input."""
    nib_image = nib.load(str(path))
    ants_image = ants.image_read(str(path))
    nib_values = np.asarray(nib_image.dataobj)
    ants_values = ants_image.numpy()
    if (nib_image.shape != ants_values.shape
            or not np.isfinite(nib_values).all() or not np.isfinite(ants_values).all()
            or not np.allclose(nib_image.affine, ants_affine_ras(ants_image), atol=1e-4, rtol=0)
            or not np.array_equal(nib_values, ants_values)):
        raise ValueError(f"ANTs/nibabel voxel indices or RAS physical frames disagree: {path}")
    return nib_image, ants_image


def binary_values(values, label: str) -> np.ndarray:
    array = np.asarray(values)
    if not np.isfinite(array).all() or not np.isin(array, (0, 1)).all():
        raise ValueError(f"{label} has nonfinite or nonbinary voxel values")
    return array.astype(bool)


def independent_nearest_coverage(source_path: Path, atlas_image,
                                 transform_path: Path) -> np.ndarray:
    """SciPy voxel pull from the saved ITK affine; no transform fitting."""
    source = nib.load(str(source_path))
    source_data = binary_values(np.asarray(source.dataobj), "independent source coverage")
    transform = sitk.ReadTransform(str(transform_path))
    origin = np.asarray(transform.TransformPoint((0.0, 0.0, 0.0)))
    columns = [np.asarray(transform.TransformPoint(tuple(float(v) for v in e))) - origin
               for e in np.eye(3)]
    pull_lps = np.eye(4)
    pull_lps[:3, :3] = np.column_stack(columns)
    pull_lps[:3, 3] = origin
    ras_lps = np.diag([-1.0, -1.0, 1.0, 1.0])
    target_to_source_voxel = (np.linalg.inv(source.affine) @ ras_lps @ pull_lps
                              @ ras_lps @ atlas_image.affine)
    return ndi.affine_transform(
        source_data.astype(np.uint8), target_to_source_voxel[:3, :3],
        offset=target_to_source_voxel[:3, 3], output_shape=atlas_image.shape,
        order=0, mode="constant", cval=0, prefilter=False) > 0


def main() -> None:
    output = HERE / "support-bits-atlas.nii.gz"
    result_path = HERE / "support-map-result.json"
    hold_path = HERE / "discretization-hold.json"
    if output.exists() or result_path.exists() or hold_path.exists():
        raise FileExistsError("saved support map is one-shot; do not overwrite")
    receipt_path = PREP / "preparation-result.json"
    exact(receipt_path, PREP_RESULT_SHA)
    receipt = json.loads(receipt_path.read_text())
    manifest = json.loads(MANIFEST.read_text())
    if (manifest["case_id"] != "RESECT-Case4" or manifest["split_role"] != "DEVELOPMENT"
            or receipt["run_id"] != "resect-case4-scan-prep-v1-attempt-01"
            or receipt["model_inference_performed"] is not False
            or receipt["planning_performed"] is not False
            or receipt["anatomical_qc_status"] != "pending_after_generation"):
        raise ValueError("not the saved source-only unreviewed Case4 preparation")
    if sha(MANIFEST) != receipt["manifest_sha256"]:
        raise ValueError("saved source manifest disagrees with preparation")
    for key in ("preoperative_t1c", "preoperative_flair"):
        item = manifest[key]
        exact(ROOT / item["path"], item["sha256"])
    for row in receipt["generated_files"]:
        path = PREP / row["relative_path"]
        if not path.is_file() or path.stat().st_size != row["bytes"]:
            raise ValueError(f"saved preparation inventory changed: {path}")
        exact(path, row["sha256"])

    atlas_path = PREP / "atlas-squeezed-3d.nii.gz"
    transform_path = PREP / "t1c-to-atlas/t1c-image-to-atlas.mat"
    atlas_image, atlas = checked_dual_reader(atlas_path)
    atlas_mask_path = PREP / "prepared/qualified-mask-atlas.nii.gz"
    atlas_mask_nib, _ = checked_dual_reader(atlas_mask_path)
    t1_cov_path = PREP / "prepared/t1c-native-coverage.nii.gz"
    flair_cov_path = PREP / "prepared/flair-coverage-in-t1c.nii.gz"
    t1_cov_nib, t1_cov = checked_dual_reader(t1_cov_path)
    flair_cov_nib, flair_cov_t1 = checked_dual_reader(flair_cov_path)
    binary_values(np.asarray(t1_cov_nib.dataobj), "saved T1 coverage")
    binary_values(np.asarray(flair_cov_nib.dataobj), "saved FLAIR coverage")
    estimated_mask = binary_values(np.asarray(atlas_mask_nib.dataobj), "saved estimated mask")
    if (atlas_image.shape != atlas_mask_nib.shape
            or not np.allclose(atlas_image.affine, atlas_mask_nib.affine, atol=1e-4, rtol=0)
            or t1_cov_nib.shape != flair_cov_nib.shape
            or not np.allclose(t1_cov_nib.affine, flair_cov_nib.affine, atol=1e-4, rtol=0)):
        raise ValueError("saved atlas or native-coverage inputs have different physical grids")
    atlas_t1 = ants.apply_transforms(fixed=atlas, moving=t1_cov,
                                    transformlist=[str(transform_path)],
                                    interpolator="nearestNeighbor")
    atlas_flair = ants.apply_transforms(fixed=atlas, moving=flair_cov_t1,
                                       transformlist=[str(transform_path)],
                                       interpolator="nearestNeighbor")
    atlas_t1 = binary_values(atlas_t1.numpy(), "atlas T1 coverage")
    atlas_flair = binary_values(atlas_flair.numpy(), "atlas FLAIR coverage")
    independent_t1 = independent_nearest_coverage(t1_cov_path, atlas_image, transform_path)
    independent_flair = independent_nearest_coverage(flair_cov_path, atlas_image, transform_path)
    t1_mismatches = int(np.count_nonzero(independent_t1 != atlas_t1))
    flair_mismatches = int(np.count_nonzero(independent_flair != atlas_flair))
    if t1_mismatches or flair_mismatches:
        hold_path.write_text(json.dumps({
            "status": "hold_no_support_map_saved",
            "reason": "independent_SciPy_and_ANTs_saved_transform_nearest_voxel_mismatch",
            "t1_voxel_mismatches": t1_mismatches,
            "flair_voxel_mismatches": flair_mismatches,
            "preparation_result_sha256": PREP_RESULT_SHA,
            "transform_sha256": sha(transform_path)}, indent=2, sort_keys=True) + "\n")
        raise ValueError("independent saved-transform coverage reconstruction disagrees voxelwise")
    bits = encode_support_bits(atlas_t1, atlas_flair, estimated_mask)
    joint_fraction = float(np.mean(atlas_t1 & atlas_flair))
    if abs(joint_fraction - receipt["atlas_common_fov_fraction"]) > 1e-8:
        raise ValueError("independent saved-transform FOV differs from preparation")
    if bits.shape != atlas_image.shape:
        raise ValueError("support map differs from atlas grid")
    image = nib.Nifti1Image(bits, atlas_image.affine)
    image.header.set_xyzt_units("mm")
    image.set_data_dtype(np.uint8)
    nib.save(image, str(output))
    reopened = nib.load(str(output))
    if (reopened.shape != atlas_image.shape
            or not np.allclose(reopened.affine, atlas_image.affine, atol=1e-4, rtol=0)
            or not np.array_equal(np.asarray(reopened.dataobj), bits)):
        raise ValueError("saved support map failed exact grid/content reopening")
    counts = {str(code): int(np.count_nonzero(bits == code)) for code in range(8)}
    result = {
        "schema_version": "case4-support-map-v1",
        "source_preparation_result_sha256": PREP_RESULT_SHA,
        "source_manifest_sha256": sha(MANIFEST),
        "source_transform_sha256": sha(transform_path),
        "support_contract_source_sha256": sha(HERE / "support_contract.py"),
        "generator_source_sha256": sha(Path(__file__)),
        "support_bits_path": str(output.relative_to(ROOT)),
        "support_bits_sha256": sha(output),
        "support_bits_shape_xyz": list(bits.shape),
        "support_bits_affine_ras_mm": atlas_image.affine.tolist(),
        "ants_nibabel_voxel_and_frame_agreement": [
            "atlas_template", "atlas_estimated_mask",
            "t1_native_coverage", "flair_in_t1_coverage"],
        "independent_nearest_saved_transform_t1_voxel_mismatches": t1_mismatches,
        "independent_nearest_saved_transform_flair_voxel_mismatches": flair_mismatches,
        "fov_sampling_semantics": "saved-transform nearest-neighbor pull at atlas voxel centers; not continuous full-cell or tool-shaft coverage",
        "bit_semantics": {
            "1": "original T1 scan field of view within atlas grid",
            "2": "original FLAIR scan field of view within atlas grid",
            "4": "unreviewed estimated SynthStrip mask; not qualified anatomy",
        },
        "bit_value_counts": counts,
        "both_original_scan_fov_fraction_of_atlas_grid": joint_fraction,
        "both_original_scan_fov_voxels": int(np.count_nonzero(atlas_t1 & atlas_flair)),
        "estimated_mask_voxels": int(np.count_nonzero(estimated_mask)),
        "unreviewed_diagnostic_input_domain_voxels": counts["7"],
        "unknown_within_atlas_grid_voxels": int(bits.size - counts["7"]),
        "outside_atlas_grid_status": "unassessed_not_background",
        "unknown_inside_both_scans_outside_estimated_mask": counts["3"],
        "mask_anatomy_qualified": False,
        "diagnostic_input_domain_qualified": False,
        "tumor_target_estimated": False,
        "model_inference_performed": False,
        "registration_optimized": False,
        "route_or_policy_run": False,
        "planning_admitted": False,
        "clinical_use_admitted": False,
        "future_interface_note": "A future reviewed ScanEstimate uses separate binary mask/coverage. This unreviewed bitfield cannot be promoted to coverage with qc_status pass; unknown outside any independently qualified ROI remains unknown, not negative tumor.",
    }
    result_path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"result": str(result_path), "sha256": sha(result_path),
                      "counts": counts}, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
