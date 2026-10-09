"""One read-only Case4 coverage-edge diagnostic; never writes a support map."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import ants
import nibabel as nib
import numpy as np
import scipy.ndimage as ndi
import SimpleITK as sitk


ROOT = Path(__file__).resolve().parents[3]
HERE = Path(__file__).resolve().parent
PREP = ROOT / "outputs/scan-target/resect-case4-prep-v1/attempt-01"
PREP_RESULT = PREP / "preparation-result.json"
PREP_RESULT_SHA = "92fdc93f1240c81ac0ecbd7910a8256a18693480bd4bed48b19e109fe308d72c"
HOLD = ROOT / "build/scan-target-estimator-research/case4-support-contract-v1/discretization-hold.json"
HOLD_SHA = "3c64662024efccbfc990922460294d2c9cb6cfc830ad562d9d9f9174268856f1"
TRANSFORM = PREP / "t1c-to-atlas/t1c-image-to-atlas.mat"
TRANSFORM_SHA = "90e85485698c6d8fc2dea108a980e3140079d97671dcfc5111114ba25a9f95fb"
OUT = HERE / "grid-constant-comparison.json"


def sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def exact(path: Path, expected: str, size: int | None = None) -> None:
    if (not path.is_file() or (size is not None and path.stat().st_size != size)
            or sha(path) != expected):
        raise ValueError(f"saved source changed or unavailable: {path}")


def binary(values, label: str) -> np.ndarray:
    array = np.asarray(values)
    if array.ndim != 3 or not np.isfinite(array).all() or not np.isin(array, (0, 1)).all():
        raise ValueError(f"{label} must remain finite 3-D binary")
    return array.astype(np.uint8)


def target_to_source_voxel(source_affine: np.ndarray, target_affine: np.ndarray,
                           transform_path: Path) -> np.ndarray:
    """Independent ITK LPS physical point action expressed in NIfTI RAS voxels."""
    transform = sitk.ReadTransform(str(transform_path))
    origin = np.asarray(transform.TransformPoint((0., 0., 0.)))
    columns = [np.asarray(transform.TransformPoint(tuple(float(v) for v in axis))) - origin
               for axis in np.eye(3)]
    pull_lps = np.eye(4)
    pull_lps[:3, :3] = np.column_stack(columns)
    pull_lps[:3, 3] = origin
    ras_lps = np.diag([-1., -1., 1., 1.])
    return np.linalg.inv(source_affine) @ ras_lps @ pull_lps @ ras_lps @ target_affine


def independent_nearest(source: np.ndarray, target_shape: tuple[int, int, int],
                        voxel_pull: np.ndarray, *, mode: str) -> np.ndarray:
    return ndi.affine_transform(source, voxel_pull[:3, :3], offset=voxel_pull[:3, 3],
                                output_shape=target_shape, order=0, mode=mode,
                                cval=0, prefilter=False)


def compare_one(label: str, source_path: Path, atlas_path: Path) -> dict:
    source_nib, atlas_nib = nib.load(str(source_path)), nib.load(str(atlas_path))
    source = binary(np.asarray(source_nib.dataobj), label + " saved coverage")
    if source_nib.ndim != 3 or atlas_nib.ndim != 3:
        raise ValueError("saved source/atlas must be 3-D")
    fixed, moving = ants.image_read(str(atlas_path)), ants.image_read(str(source_path))
    expected = binary(ants.apply_transforms(
        fixed=fixed, moving=moving, transformlist=[str(TRANSFORM)],
        interpolator="nearestNeighbor", defaultvalue=0).numpy(), label + " ANTs coverage")
    pull = target_to_source_voxel(source_nib.affine, atlas_nib.affine, TRANSFORM)
    old = independent_nearest(source, atlas_nib.shape, pull, mode="constant")
    grid = independent_nearest(source, atlas_nib.shape, pull, mode="grid-constant")
    return {
        "source_sha256": sha(source_path),
        "source_shape_xyz": list(source_nib.shape),
        "atlas_shape_xyz": list(atlas_nib.shape),
        "source_affine_ras_mm": source_nib.affine.tolist(),
        "atlas_affine_ras_mm": atlas_nib.affine.tolist(),
        "ants_coverage_voxels": int(np.count_nonzero(expected)),
        "scipy_constant_vs_ants_mismatches": int(np.count_nonzero(old != expected)),
        "scipy_grid_constant_vs_ants_mismatches": int(np.count_nonzero(grid != expected)),
        "constant_only_boundary_corrections": int(np.count_nonzero((old != expected) & (grid == expected))),
        "new_grid_disagreements": int(np.count_nonzero(grid != expected)),
    }


def main() -> None:
    if OUT.exists():
        raise FileExistsError("diagnostic replay is one-shot; preserve original result")
    exact(PREP_RESULT, PREP_RESULT_SHA)
    exact(HOLD, HOLD_SHA)
    exact(TRANSFORM, TRANSFORM_SHA)
    prep = json.loads(PREP_RESULT.read_text())
    hold = json.loads(HOLD.read_text())
    if (prep["run_id"] != "resect-case4-scan-prep-v1-attempt-01"
            or prep["model_inference_performed"] is not False
            or prep["planning_performed"] is not False
            or hold["status"] != "hold_no_support_map_saved"
            or hold["preparation_result_sha256"] != PREP_RESULT_SHA
            or hold["transform_sha256"] != TRANSFORM_SHA):
        raise ValueError("not the preserved source-only Case4 mismatch")
    for item in prep["generated_files"]:
        exact(PREP / item["relative_path"], item["sha256"], item["bytes"])
    atlas = PREP / "atlas-squeezed-3d.nii.gz"
    t1 = compare_one("t1", PREP / "prepared/t1c-native-coverage.nii.gz", atlas)
    flair = compare_one("flair", PREP / "prepared/flair-coverage-in-t1c.nii.gz", atlas)
    if (t1["scipy_constant_vs_ants_mismatches"] != hold["t1_voxel_mismatches"]
            or flair["scipy_constant_vs_ants_mismatches"] != hold["flair_voxel_mismatches"]):
        status = "old_mismatch_not_reproduced"
    elif (t1["scipy_grid_constant_vs_ants_mismatches"] != 0
          or flair["scipy_grid_constant_vs_ants_mismatches"] != 0):
        status = "grid_boundary_fix_incomplete"
    else:
        status = "grid_boundary_fix_explains_saved_hold"
    result = {
        "schema_version": "case4-nearest-neighbor-boundary-diagnosis-v1",
        "status": status,
        "case_id": "RESECT-Case4", "split_role": "DEVELOPMENT",
        "original_hold_sha256": HOLD_SHA,
        "source_preparation_result_sha256": PREP_RESULT_SHA,
        "saved_transform_sha256": TRANSFORM_SHA,
        "diagnostic_source_sha256": sha(Path(__file__)),
        "t1": t1, "flair": flair,
        "source_only_saved_transform_replay": True,
        "new_registration_performed": False,
        "model_inference_performed": False,
        "support_map_written": False,
        "anatomical_acceptance": False,
        "planning_admitted": False,
        "interpretation": "Nearest-neighbor boundary implementation comparison only; no patient anatomy or tumor validation",
    }
    OUT.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"result": str(OUT), "result_sha256": sha(OUT),
                      "status": status}, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
