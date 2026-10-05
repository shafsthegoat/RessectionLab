"""Physical-frame rigid DWI/T1 alignment with explicit unresolved correction QC.

Registration changes inter-image pose. It does not correct eddy currents,
susceptibility distortions, motion within a DWI acquisition, or its gradients.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import json
from pathlib import Path
import shutil
import time
from typing import Any, Callable

import nibabel as nib
import numpy as np

from .diffusion import image_axis_basis
from .imaging import ImagingError, file_sha256, inspect_nifti


RAS_TO_LPS = np.diag([-1., -1., 1., 1.])


class PreprocessingError(ImagingError):
    """Named preparation or registration failure suitable for a case QC report."""


@dataclass(frozen=True)
class RegistrationConfig:
    iterations_per_level: int = 120
    sampling_fraction: float = 0.25
    random_seed: int = 41
    threads: int = 2
    maximum_center_displacement_mm: float = 40
    maximum_rotation_degrees: float = 25

    def validate(self) -> None:
        if (not 1 <= self.iterations_per_level <= 500 or not 0 < self.sampling_fraction <= 1
                or not 1 <= self.threads <= 4 or self.random_seed < 0
                or not 0 < self.maximum_center_displacement_mm <= 100
                or not 0 < self.maximum_rotation_degrees <= 45):
            raise PreprocessingError("INVALID_REGISTRATION_BUDGET", "Use a bounded positive registration configuration.")


def local_tool_inventory() -> dict[str, Any]:
    """Inspect availability, without installing or invoking preprocessing tools."""
    import importlib.metadata
    paths = {name: shutil.which(name) for name in
             ("topup", "eddy", "eddy_cpu", "bet", "flirt", "epi_reg", "antsRegistration", "mrconvert")}
    versions = {}
    for package in ("SimpleITK", "dipy"):
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            versions[package] = None
    return {"executables_on_path": paths, "python_packages": versions}


def to_sitk(array: np.ndarray, affine_ras_mm: np.ndarray):
    """Convert XYZ NumPy data and RAS geometry to SimpleITK's ZYX/LPS contract."""
    import SimpleITK as sitk

    array = np.asarray(array)
    if array.ndim != 3 or not np.isfinite(array).all():
        raise PreprocessingError("INVALID_REGISTRATION_IMAGE", "Expected a finite 3D scalar array.")
    basis_ras = image_axis_basis(affine_ras_mm)
    spacing = np.linalg.norm(affine_ras_mm[:3, :3], axis=0)
    image = sitk.GetImageFromArray(np.transpose(array, (2, 1, 0)).astype(np.float32))
    image.SetSpacing(tuple(spacing))
    image.SetDirection(tuple((RAS_TO_LPS[:3, :3] @ basis_ras).ravel()))
    image.SetOrigin(tuple(RAS_TO_LPS[:3, :3] @ affine_ras_mm[:3, 3]))
    return image


def sitk_transform_to_ras(transform) -> np.ndarray:
    """Extract a linear transform including its rotation center and LPS signs."""
    origin = np.asarray(transform.TransformPoint((0., 0., 0.)))
    linear = np.column_stack([
        np.asarray(transform.TransformPoint(tuple(axis))) - origin for axis in np.eye(3)
    ])
    matrix = np.eye(4)
    matrix[:3, :3] = linear
    matrix[:3, 3] = origin
    points = np.array([[3.1, -7.5, 4.2], [-10., 11., 17.]])
    if not np.allclose(np.array([transform.TransformPoint(tuple(p)) for p in points]),
                       points @ linear.T + origin, atol=1e-7):
        raise PreprocessingError("NONLINEAR_REGISTRATION_UNSUPPORTED", "This export only supports global linear transforms.")
    return RAS_TO_LPS @ matrix @ RAS_TO_LPS


def validate_rigid(matrix: np.ndarray) -> None:
    matrix = np.asarray(matrix)
    if (matrix.shape != (4, 4) or not np.isfinite(matrix).all()
            or not np.allclose(matrix[3], [0, 0, 0, 1])
            or not np.allclose(matrix[:3, :3].T @ matrix[:3, :3], np.eye(3), atol=1e-5)
            or not np.isclose(np.linalg.det(matrix[:3, :3]), 1, atol=1e-5)):
        raise PreprocessingError("NONRIGID_OR_REFLECTED_TRANSFORM", "Registration must preserve distances and handedness.")


def histogram_nmi(fixed: np.ndarray, moving: np.ndarray, mask: np.ndarray, *, bins: int = 32) -> float:
    """Independent fixed-bin normalized mutual information on declared paired voxels."""
    values_a, values_b = fixed[mask], moving[mask]
    if len(values_a) < 100 or not np.isfinite(values_a).all() or not np.isfinite(values_b).all():
        raise PreprocessingError("INSUFFICIENT_REGISTRATION_OVERLAP", "Need at least 100 finite paired voxels for QC.")
    if ((values_a < 0).any() or (values_a > 1).any()
            or (values_b < 0).any() or (values_b > 1).any()):
        raise PreprocessingError("QC_INTENSITY_RANGE_INVALID", "Independent histogram QC requires normalized values in [0,1].")
    histogram, _, _ = np.histogram2d(values_a, values_b, bins=bins, range=((0, 1), (0, 1)))
    probability = histogram / histogram.sum()

    def entropy(p):
        positive = p[p > 0]
        return float(-np.sum(positive * np.log(positive)))

    joint = entropy(probability)
    if joint <= 0:
        raise PreprocessingError("CONSTANT_REGISTRATION_IMAGE", "Registration QC needs nonconstant signals.")
    return (entropy(probability.sum(axis=0)) + entropy(probability.sum(axis=1))) / joint


def _normalize(array: np.ndarray) -> tuple[np.ndarray, list[float]]:
    positive = array[array > 0]
    if positive.size < 100:
        raise PreprocessingError("CONSTANT_REGISTRATION_IMAGE", "Registration needs substantial positive image support.")
    low, high = np.percentile(positive, [1, 99])
    if not high > low:
        raise PreprocessingError("CONSTANT_REGISTRATION_IMAGE", "Registration image has no usable intensity variation.")
    return np.clip((array - low) / (high - low), 0, 1).astype(np.float32), [float(low), float(high)]


def register_t1_to_b0(
    b0: np.ndarray, b0_affine_ras_mm: np.ndarray, t1: np.ndarray, t1_affine_ras_mm: np.ndarray,
    b0_mask: np.ndarray, *, config: RegistrationConfig = RegistrationConfig(),
    cancelled: Callable[[], bool] | None = None,
) -> dict[str, Any]:
    """Register T1 sampling into b0; return the physical DWI-RAS→T1-RAS map.

    The estimate starts at identity in physical space, preserving source header
    pose. A supplied b0 mask limits the metric; its review is a separate question.
    """
    import SimpleITK as sitk

    config.validate()
    b0_mask = np.asarray(b0_mask)
    if b0_mask.shape != b0.shape or b0_mask.dtype != bool or b0_mask.sum() < 100:
        raise PreprocessingError("INVALID_REGISTRATION_MASK", "Use a boolean b0-grid mask containing at least 100 voxels.")
    fixed_raw = to_sitk(b0, b0_affine_ras_mm)
    moving_raw = to_sitk(t1, t1_affine_ras_mm)
    fixed_norm, fixed_window = _normalize(b0)
    moving_norm, moving_window = _normalize(t1)
    fixed = to_sitk(fixed_norm, b0_affine_ras_mm)
    moving = to_sitk(moving_norm, t1_affine_ras_mm)
    mask_image = sitk.Cast(to_sitk(b0_mask.astype(np.uint8), b0_affine_ras_mm), sitk.sitkUInt8)
    center_voxel = np.argwhere(b0_mask).mean(axis=0)
    center_ras = nib.affines.apply_affine(b0_affine_ras_mm, center_voxel)
    initial = sitk.Euler3DTransform()
    initial.SetCenter(tuple(RAS_TO_LPS[:3, :3] @ center_ras))
    registration = sitk.ImageRegistrationMethod()
    registration.SetNumberOfThreads(config.threads)
    registration.SetMetricAsMattesMutualInformation(numberOfHistogramBins=32)
    registration.SetMetricSamplingStrategy(registration.RANDOM)
    registration.SetMetricSamplingPercentage(config.sampling_fraction, config.random_seed)
    registration.SetMetricFixedMask(mask_image)
    registration.SetInterpolator(sitk.sitkLinear)
    registration.SetOptimizerAsRegularStepGradientDescent(
        learningRate=1., minStep=0.001, numberOfIterations=config.iterations_per_level,
        relaxationFactor=0.5, gradientMagnitudeTolerance=1e-5,
    )
    registration.SetOptimizerScalesFromPhysicalShift()
    registration.SetShrinkFactorsPerLevel([4, 2, 1])
    registration.SetSmoothingSigmasPerLevel([2., 1., 0.])
    registration.SmoothingSigmasAreSpecifiedInPhysicalUnitsOn()
    registration.SetInitialTransform(initial, inPlace=False)
    history = []

    def iteration():
        history.append({"level": int(registration.GetCurrentLevel()),
                        "iteration": int(registration.GetOptimizerIteration()),
                        "sampled_negative_mattes_mi": float(registration.GetMetricValue())})
        if cancelled is not None and cancelled():
            registration.StopRegistration()

    registration.AddCommand(sitk.sitkIterationEvent, iteration)
    if cancelled is not None and cancelled():
        raise PreprocessingError("REGISTRATION_CANCELLED", "Registration was cancelled before optimization.")
    started = time.perf_counter()
    try:
        fitted = registration.Execute(fixed, moving)
    except RuntimeError as error:
        raise PreprocessingError("REGISTRATION_OPTIMIZER_FAILED", str(error)) from error
    elapsed = time.perf_counter() - started
    if cancelled is not None and cancelled():
        raise PreprocessingError("REGISTRATION_CANCELLED", "Registration was cancelled during optimization.")
    matrix = sitk_transform_to_ras(fitted)
    validate_rigid(matrix)
    inverse = np.linalg.inv(matrix)
    before = np.transpose(sitk.GetArrayFromImage(sitk.Resample(moving, fixed, initial, sitk.sitkLinear, 0)), (2, 1, 0))
    after = np.transpose(sitk.GetArrayFromImage(sitk.Resample(moving, fixed, fitted, sitk.sitkLinear, 0)), (2, 1, 0))
    moved_raw = np.transpose(sitk.GetArrayFromImage(sitk.Resample(moving_raw, fixed_raw, fitted, sitk.sitkLinear, 0)), (2, 1, 0))
    # Compare identical paired voxels; do not improve a QC score by discarding
    # difficult pre- or post-registration overlap independently.
    common_mask = b0_mask & (before > 0) & (after > 0)
    nmi_before = histogram_nmi(fixed_norm, before, common_mask)
    nmi_after = histogram_nmi(fixed_norm, after, common_mask)
    center_shift = float(np.linalg.norm(nib.affines.apply_affine(matrix, center_ras) - center_ras))
    rotation = float(np.degrees(np.arccos(np.clip((np.trace(matrix[:3, :3]) - 1) / 2, -1, 1))))
    overlap = float(common_mask.sum() / b0_mask.sum())
    flags = []
    if nmi_after < nmi_before:
        flags.append("INDEPENDENT_NMI_DECREASED")
    if overlap < 0.9:
        flags.append("LIMITED_COMMON_IMAGE_SUPPORT")
    if center_shift > config.maximum_center_displacement_mm or rotation > config.maximum_rotation_degrees:
        flags.append("LARGE_RIGID_TRANSFORM_REQUIRES_REVIEW")
    report = {
        "method": "SimpleITK Euler3D rigid; Mattes mutual information; physical identity initialization",
        "simpleitk_version": sitk.Version_VersionString(), "configuration": asdict(config),
        "fixed_image": "mean_b0", "moving_image": "T1",
        "transform_direction": "DWI_RAS_mm_to_T1_RAS_mm",
        "DWI_RAS_mm_to_T1_RAS_mm": matrix.tolist(),
        "T1_RAS_mm_to_DWI_RAS_mm": inverse.tolist(),
        "fixed_affine_ras_mm": b0_affine_ras_mm.tolist(),
        "moving_affine_ras_mm": t1_affine_ras_mm.tolist(),
        "elapsed_registration_seconds": elapsed,
        "optimization_iterations": len(history), "optimizer_history": history,
        "optimizer_stop_reason": registration.GetOptimizerStopConditionDescription(),
        "independent_qc": {"implementation": "NumPy joint histogram; fixed 32 bins over normalized [0,1]",
                           "metric": "(H(fixed)+H(moving))/H(joint)",
                           "same_paired_voxels_before_and_after": True,
                           "paired_voxels": int(common_mask.sum()), "mask_common_support_fraction": overlap,
                           "nmi_before": nmi_before, "nmi_after": nmi_after,
                           "center_displacement_mm": center_shift, "rotation_degrees": rotation,
                           "flags": flags},
        "normalization_windows": {"fixed_mean_b0": fixed_window, "moving_T1": moving_window},
        "registration_reviewed": False, "mask_reviewed": False,
        "preprocessing_ready": False, "clinical_deficit_probability": None,
        "functional_coverage": {"motor": "unknown", "language": "unknown"},
        "unresolved": ["susceptibility_correction_not_established", "volume_motion_and_eddy_correction_not_established",
                       "rotated_gradient_provenance_not_established", "independent_anatomical_landmark_review_pending",
                       "mask_and_lesion_region_review_pending"],
    }
    return {"report": report, "t1_on_b0": moved_raw, "before_normalized": before,
            "after_normalized": after, "b0_normalized": fixed_norm,
            "qc_common_mask": common_mask, "sitk_fixed_to_moving_transform": fitted}


def save_registration_qc(result: dict, output: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fixed = result["b0_normalized"]
    before, after = result["before_normalized"], result["after_normalized"]
    indices = [int(fixed.shape[2] * proportion) for proportion in (0.35, 0.5, 0.65)]
    figure, axes = plt.subplots(3, 3, figsize=(10, 10), facecolor="#111827")
    for row, z in enumerate(indices):
        for column, image in enumerate((fixed, before, after)):
            ax = axes[row, column]
            ax.imshow(fixed[:, :, z].T, origin="lower", cmap="gray", vmin=0, vmax=1)
            if column:
                ax.contour(image[:, :, z].T, levels=[0.2, 0.5, 0.75], colors=["#4ade80", "#22d3ee", "#fb7185"], linewidths=0.5)
            ax.set_title(("Mean b0", "T1 contours before", "T1 contours after")[column] + f" | k={z}", color="white", fontsize=10)
            ax.set_xlabel("acquired DWI image i index", color="#cbd5e1", fontsize=8)
            ax.tick_params(colors="#94a3b8", labelsize=7)
    figure.suptitle("Rigid DWI/T1 alignment diagnostic\nUncorrected EPI; anatomical review pending", color="white", fontsize=14)
    figure.tight_layout()
    figure.savefig(output, dpi=140, facecolor=figure.get_facecolor())
    plt.close(figure)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mean-b0", type=Path, required=True)
    parser.add_argument("--t1", type=Path, required=True)
    parser.add_argument("--b0-mask", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--source-state", choices=["raw_uncorrected", "correction_outputs_pending_review"], required=True)
    parser.add_argument("--save-qc", action="store_true")
    args = parser.parse_args()
    import SimpleITK as sitk

    qc = {key: inspect_nifti(path) for key, path in
          (("mean_b0", args.mean_b0), ("T1", args.t1), ("b0_mask", args.b0_mask))}
    if (qc["b0_mask"]["shape"] != qc["mean_b0"]["shape"]
            or not np.allclose(qc["b0_mask"]["affine_ras_mm"], qc["mean_b0"]["affine_ras_mm"], atol=1e-5)):
        raise PreprocessingError("REGISTRATION_MASK_FRAME_MISMATCH", "The mask must match the mean-b0 grid exactly.")
    b0 = nib.load(args.mean_b0).get_fdata(dtype=np.float32)
    t1 = nib.load(args.t1).get_fdata(dtype=np.float32)
    mask = nib.load(args.b0_mask).get_fdata(dtype=np.float32)
    if not np.isin(mask, [0, 1]).all():
        raise PreprocessingError("REGISTRATION_MASK_NONBINARY", "A registration mask must be explicitly binary.")
    affine = np.asarray(qc["mean_b0"]["affine_ras_mm"])
    result = register_t1_to_b0(b0, affine, t1, np.asarray(qc["T1"]["affine_ras_mm"]), mask.astype(bool))
    args.output.mkdir(parents=True, exist_ok=True)
    image = nib.Nifti1Image(result["t1_on_b0"], affine)
    image.header.set_xyzt_units("mm")
    image.set_sform(affine, code=1)
    image.set_qform(affine, code=1)
    nib.save(image, args.output / "T1_on_b0.nii.gz")
    sitk.WriteTransform(result["sitk_fixed_to_moving_transform"], str(args.output / "DWI_LPS_to_T1_LPS.tfm"))
    if args.save_qc:
        save_registration_qc(result, args.output / "registration_qc.png")
    report = {"schema_version": 1, "created_at": datetime.now(timezone.utc).isoformat(),
              "implementation_sha256": file_sha256(__file__), "source_state": args.source_state,
              "source_hashes": {key: file_sha256(path) for key, path in
                                (("mean_b0", args.mean_b0), ("T1", args.t1), ("b0_mask", args.b0_mask))},
              "header_qc": qc, "local_tools": local_tool_inventory(), **result["report"]}
    # Hash only completed artifacts. A redirected log in this directory may
    # still be written after this point and is not a reproducible data object.
    artifact_names = ["T1_on_b0.nii.gz", "DWI_LPS_to_T1_LPS.tfm"]
    if args.save_qc:
        artifact_names.append("registration_qc.png")
    report["artifact_hashes"] = {name: file_sha256(args.output / name) for name in artifact_names}
    (args.output / "registration_report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({key: value for key, value in report.items() if key != "optimizer_history"}, indent=2))


if __name__ == "__main__":
    main()
