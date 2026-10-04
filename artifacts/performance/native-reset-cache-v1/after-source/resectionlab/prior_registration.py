"""Bounded linear atlas registration with explicit frames and review-required output.

SimpleITK estimates a fixed-patient LPS -> moving-template LPS pull transform.
The public output is its inverse in RAS millimeters for anatomy.register_prior.
Global image similarity is a fitting diagnostic, not independent landmark or
functional localization validation. No registration run accepts its own overlay.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
import json
from pathlib import Path
import time
from typing import Any, Callable

import nibabel as nib
import numpy as np
from scipy.ndimage import distance_transform_edt

from .core import semantic_digest
from .imaging import file_sha256, inspect_nifti


FRAME_FLIP = np.diag([-1., -1., 1., 1.])
REGISTRATION_VERSION = "simpleitk-linear-prior-v3"


def image_from_ras(data: np.ndarray, affine_ras_mm: np.ndarray):
    """Create a SimpleITK LPS image without changing physical voxel locations."""
    import SimpleITK as sitk

    data, affine = np.asarray(data), np.asarray(affine_ras_mm, dtype=float)
    if (data.ndim != 3 or not np.isfinite(data).all() or affine.shape != (4, 4)
            or not np.isfinite(affine).all() or not np.allclose(affine[3], [0, 0, 0, 1])):
        raise ValueError("A finite 3D image and homogeneous RAS-mm affine are required")
    lps = FRAME_FLIP @ affine
    spacing = np.linalg.norm(lps[:3, :3], axis=0)
    if np.any(spacing <= 0):
        raise ValueError("Voxel spacing must be positive")
    direction = lps[:3, :3] / spacing
    if not np.allclose(direction.T @ direction, np.eye(3), atol=1e-5):
        raise ValueError("Sheared grids require an explicit prior resampling operation")
    image = sitk.GetImageFromArray(data.transpose(2, 1, 0).copy())
    image.SetSpacing(tuple(spacing))
    image.SetOrigin(tuple(lps[:3, 3]))
    image.SetDirection(tuple(direction.ravel()))
    return image


def linear_transform_matrix(transform) -> np.ndarray:
    """Recover the physical homogeneous matrix, including nonzero transform centers."""
    origin = np.asarray(transform.TransformPoint((0., 0., 0.)))
    matrix = np.eye(4)
    matrix[:3, 3] = origin
    for axis in range(3):
        basis = np.eye(3)[axis]
        matrix[:3, axis] = np.asarray(transform.TransformPoint(tuple(basis))) - origin
    for point in ((2., -3., 7.), (11., 5., -2.)):
        if not np.allclose(matrix[:3, :3] @ point + origin, transform.TransformPoint(point), atol=1e-6):
            raise ValueError("Only global linear transforms can be exported as a 4x4 matrix")
    if not np.isfinite(matrix).all() or np.linalg.det(matrix[:3, :3]) <= 0:
        raise ValueError("Invalid or reflected registration transform")
    return matrix


def template_to_patient_ras(transform) -> np.ndarray:
    return FRAME_FLIP @ np.linalg.inv(linear_transform_matrix(transform)) @ FRAME_FLIP


def verify_template_grid(template_path: str | Path, priors: dict) -> dict[str, Any]:
    """Check a template against equal-resolution prior grids; this is not anatomy QC."""
    qc = inspect_nifti(template_path)
    affine = np.asarray(qc["affine_ras_mm"])
    matched, mismatched = [], []
    for name, prior in priors.items():
        if np.allclose(np.linalg.norm(prior.affine_ras_mm[:3, :3], axis=0), qc["spacing_mm"]):
            matches = tuple(qc["shape"]) == prior.data.shape and np.allclose(affine, prior.affine_ras_mm)
            (matched if matches else mismatched).append(name)
    if mismatched or not matched:
        raise ValueError(f"Template grid does not match expected prior grids: {mismatched}")
    return {"template_file": Path(template_path).name, "sha256": file_sha256(template_path),
            "matched_maps": matched, "shape": qc["shape"], "affine_ras_mm": affine.tolist(),
            "scope": "grid_and_declared_template_provenance_only"}


@dataclass(frozen=True)
class RegistrationSettings:
    rigid_start_angles_deg: tuple[float, ...] = (0., -6., 6.)
    iterations_per_level: int = 90
    sampling_fraction: float = .18
    max_seconds_per_candidate: float = 45.
    seed: int = 41004
    threads: int = 2
    lesion_exclusion_buffer_mm: float = 5.
    affine_refinement: bool = True
    optimizer: str = "regular_step"

    def __post_init__(self):
        if (not self.rigid_start_angles_deg or not np.isfinite(self.rigid_start_angles_deg).all()
                or not isinstance(self.iterations_per_level, int) or self.iterations_per_level < 1
                or not 0 < self.sampling_fraction <= 1
                or not np.isfinite(self.max_seconds_per_candidate) or self.max_seconds_per_candidate <= 0
                or not isinstance(self.threads, int) or self.threads < 1
                or not np.isfinite(self.lesion_exclusion_buffer_mm) or self.lesion_exclusion_buffer_mm < 0
                or self.optimizer not in {"regular_step", "line_search"}):
            raise ValueError("Invalid bounded registration settings")


def _normalise(data):
    positive = data[data > 0]
    if len(positive) < 100:
        raise ValueError("Insufficient nonzero anatomical support")
    high = np.percentile(positive, 99.5)
    return np.clip(data / max(high, 1e-6), 0, 1).astype(np.float32)


def _center(mask_image):
    import SimpleITK as sitk
    stats = sitk.LabelShapeStatisticsImageFilter()
    stats.Execute(sitk.Cast(mask_image > 0, sitk.sitkUInt8))
    if not stats.HasLabel(1):
        raise ValueError("Registration mask is empty")
    return np.asarray(stats.GetCentroid(1))


def _diagnostics(fixed, moving, fixed_brain, moving_brain, evaluation_mask, transform):
    import SimpleITK as sitk
    warped = sitk.Resample(moving, fixed, transform, sitk.sitkLinear, 0., sitk.sitkFloat32)
    warped_mask = sitk.Resample(moving_brain, fixed, transform, sitk.sitkNearestNeighbor, 0, sitk.sitkUInt8)
    a, b = sitk.GetArrayFromImage(fixed), sitk.GetArrayFromImage(warped)
    fa, mb = sitk.GetArrayFromImage(fixed_brain).astype(bool), sitk.GetArrayFromImage(warped_mask).astype(bool)
    selected = sitk.GetArrayFromImage(evaluation_mask).astype(bool) & mb
    corr = float(np.corrcoef(a[selected], b[selected])[0, 1]) if selected.sum() > 10 else None
    if corr is not None and not np.isfinite(corr):
        corr = None
    dice = 2 * np.count_nonzero(fa & mb) / max(1, np.count_nonzero(fa) + np.count_nonzero(mb))
    return {"brain_support_dice": float(dice), "lesion_excluded_intensity_correlation": corr,
            "comparison_voxels": int(selected.sum()), "independent_landmark_error_mm": None,
            "diagnostic_scope": "global_fit_only_no_independent_anatomical_or_functional_ground_truth"}


def register_template_to_patient(patient_path: str | Path, template_path: str | Path, *,
                                 lesion_path: str | Path | None = None, case_hash: str,
                                 template_identity: str = "unspecified_template_requires_source_verification",
                                 settings: RegistrationSettings | None = None,
                                 progress: Callable[[dict], None] | None = None,
                                 cancelled: Callable[[], bool] | None = None) -> dict[str, Any]:
    """Compare bounded rigid starts and optional affine refinement on local imaging.

    Inputs must be skull stripped, with positive intensities indicating brain
    support. The supplied lesion is excluded from similarity fitting after a
    declared physical buffer. This cannot correct unmodeled mass effect.
    """
    import SimpleITK as sitk
    settings = settings or RegistrationSettings()
    if not case_hash:
        raise ValueError("An immutable case hash is required")
    if cancelled and cancelled():
        raise InterruptedError("Registration cancelled before loading")
    started = time.perf_counter()
    patient_qc, template_qc = inspect_nifti(patient_path), inspect_nifti(template_path)
    p = nib.load(str(patient_path)).get_fdata(dtype=np.float32)
    t = nib.load(str(template_path)).get_fdata(dtype=np.float32)
    patient_affine = np.asarray(patient_qc["affine_ras_mm"])
    lesion = np.zeros(p.shape, dtype=bool)
    if lesion_path is not None:
        lqc = inspect_nifti(lesion_path)
        if lqc["shape"] != list(p.shape) or not np.allclose(lqc["affine_ras_mm"], patient_affine):
            raise ValueError("Lesion exclusion mask is not in the patient structural grid")
        labels = nib.load(str(lesion_path)).get_fdata(dtype=np.float32)
        if not np.isfinite(labels).all() or np.any(labels < 0):
            raise ValueError("Lesion labels must be finite and nonnegative")
        lesion = labels > 0
    exclusion = (distance_transform_edt(~lesion, sampling=patient_qc["spacing_mm"])
                 <= settings.lesion_exclusion_buffer_mm) if lesion.any() else lesion
    fixed_full = image_from_ras(_normalise(p), patient_affine)
    moving = image_from_ras(_normalise(t), np.asarray(template_qc["affine_ras_mm"]))
    fixed_brain_full = image_from_ras((p > 0).astype(np.uint8), patient_affine)
    fit_mask_full = image_from_ras(((p > 0) & ~exclusion).astype(np.uint8), patient_affine)
    moving_brain = image_from_ras((t > 0).astype(np.uint8), np.asarray(template_qc["affine_ras_mm"]))
    shrink = [max(1, int(round(2 / s))) for s in fixed_full.GetSpacing()]
    fixed = sitk.Shrink(fixed_full, shrink)
    fixed_brain = sitk.Resample(fixed_brain_full, fixed, sitk.Transform(), sitk.sitkNearestNeighbor, 0, sitk.sitkUInt8)
    fit_mask = sitk.Resample(fit_mask_full, fixed, sitk.Transform(), sitk.sitkNearestNeighbor, 0, sitk.sitkUInt8)
    fixed_center, moving_center = _center(fixed_brain), _center(moving_brain)
    results, transforms = [], {}

    def execute(name, initial):
        if cancelled and cancelled():
            raise InterruptedError("Registration cancelled")
        method = sitk.ImageRegistrationMethod()
        method.SetNumberOfThreads(settings.threads)
        method.SetMetricAsMattesMutualInformation(32)
        method.SetMetricSamplingStrategy(method.RANDOM)
        method.SetMetricSamplingPercentage(settings.sampling_fraction, settings.seed)
        method.SetMetricFixedMask(fit_mask)
        method.SetMetricMovingMask(moving_brain)
        method.SetInterpolator(sitk.sitkLinear)
        if settings.optimizer == "line_search":
            method.SetOptimizerAsGradientDescentLineSearch(learningRate=.5,
                    numberOfIterations=settings.iterations_per_level, convergenceMinimumValue=1e-6,
                    convergenceWindowSize=12, lineSearchMaximumIterations=10,
                    maximumStepSizeInPhysicalUnits=.5)
        else:
            method.SetOptimizerAsRegularStepGradientDescent(learningRate=.2 if name.startswith("affine") else 1., minStep=.001,
                            numberOfIterations=settings.iterations_per_level, relaxationFactor=.5)
        method.SetOptimizerScalesFromPhysicalShift()
        method.SetShrinkFactorsPerLevel([2, 1] if name.startswith("affine") else [4, 2, 1])
        method.SetSmoothingSigmasPerLevel([1., 0.] if name.startswith("affine") else [2., 1., 0.])
        method.SmoothingSigmasAreSpecifiedInPhysicalUnitsOn()
        method.SetInitialTransform(initial, inPlace=False)
        begin = time.perf_counter()
        trace, budget_stopped = [], False

        def update():
            nonlocal budget_stopped
            elapsed = time.perf_counter() - begin
            trace.append({"level": int(method.GetCurrentLevel()), "iteration": int(method.GetOptimizerIteration()),
                          "metric": float(method.GetMetricValue()), "elapsed_seconds": elapsed})
            if elapsed >= settings.max_seconds_per_candidate or (cancelled and cancelled()):
                budget_stopped = True
                method.StopRegistration()
        method.AddCommand(sitk.sitkIterationEvent, update)
        try:
            transform = method.Execute(fixed, moving)
            if cancelled and cancelled():
                raise InterruptedError("Registration cancelled")
            matrix = template_to_patient_ras(transform)
            singular = np.linalg.svd(matrix[:3, :3], compute_uv=False)
            result = {"candidate_id": name, "status": "candidate_requires_alignment_review",
                      "qc_state": "alignment_review_required", "mni_ras_to_patient_ras_mm": matrix.tolist(),
                      "patient_lps_to_template_lps_mm": linear_transform_matrix(transform).tolist(),
                      "elapsed_seconds": time.perf_counter() - begin, "optimizer_metric": float(method.GetMetricValue()),
                      "optimizer_stop": method.GetOptimizerStopConditionDescription(), "budget_stopped": budget_stopped,
                      "gradient_iterations": len(trace), "trace": trace,
                      "linear_scale_singular_values": singular.tolist(),
                      "plausibility_flags": [] if np.all((singular > .7) & (singular < 1.4)) else ["large_global_scale_requires_review"],
                      **_diagnostics(fixed, moving, fixed_brain, moving_brain, fit_mask, transform)}
            transforms[name] = transform
        except (RuntimeError, ValueError) as exc:
            result = {"candidate_id": name, "status": "registration_failed", "error": str(exc),
                      "elapsed_seconds": time.perf_counter() - begin, "trace": trace}
        results.append(result)
        if progress:
            progress({k: v for k, v in result.items() if k != "trace"})
        return result

    for angle in settings.rigid_start_angles_deg:
        initial = sitk.Euler3DTransform()
        initial.SetCenter(tuple(fixed_center))
        initial.SetRotation(0., 0., float(np.deg2rad(angle)))
        initial.SetTranslation(tuple(moving_center - fixed_center))
        execute(f"rigid_start_{angle:g}deg", initial)
    usable = [r for r in results if r["status"] != "registration_failed" and not r["plausibility_flags"]]
    if settings.affine_refinement and usable:
        best = min(usable, key=lambda r: r["optimizer_metric"])
        matrix = linear_transform_matrix(transforms[best["candidate_id"]])
        initial = sitk.AffineTransform(3)
        initial.SetCenter(tuple(fixed_center))
        initial.SetMatrix(tuple(matrix[:3, :3].ravel()))
        initial.SetTranslation(tuple(matrix[:3, 3] - fixed_center + matrix[:3, :3] @ fixed_center))
        execute("affine_from_best_rigid", initial)
    result = {"schema": "resectionlab.prior_registration/1", "version": REGISTRATION_VERSION,
              "implementation_sha256": file_sha256(__file__),
              "case_hash": case_hash, "patient_sha256": file_sha256(patient_path),
              "template_sha256": file_sha256(template_path),
              "lesion_sha256": None if lesion_path is None else file_sha256(lesion_path),
              "template_identity": template_identity,
              "patient_affine_ras_mm": patient_affine.tolist(), "patient_shape": list(p.shape),
              "settings": asdict(settings), "simpleitk_version": sitk.Version_VersionString(),
              "fit_mask_excluded_voxels": int(np.count_nonzero((p > 0) & exclusion)),
              "elapsed_seconds": time.perf_counter() - started, "candidates": results,
              "qc_state": "alignment_review_required", "patient_specific_function": False,
              "clinical_deficit_probability": None,
              "limitations": ["no_independent_landmark_ground_truth", "population_prior_only",
                              "linear_transform_does_not_model_lesion_mass_effect", "no_expert_alignment_review",
                              "template_commercial_redistribution_rights_unresolved"]}
    result["input_configuration_hash"] = semantic_digest({k: result[k] for k in
            ("version", "case_hash", "patient_sha256", "template_sha256", "template_identity", "lesion_sha256", "settings")})
    return result


def render_registration_qc(patient_path: str | Path, template_path: str | Path,
                           report: dict[str, Any], destination: str | Path, *,
                           lesion_path: str | Path | None = None) -> Path:
    """Render patient MRI with candidate template contours; preserve failed rows in JSON."""
    import matplotlib.pyplot as plt
    from scipy.ndimage import affine_transform
    from .app.reslice import sample_slice, slice_geometry

    if file_sha256(patient_path) != report["patient_sha256"] or file_sha256(template_path) != report["template_sha256"]:
        raise ValueError("Registration QC sources differ from saved run hashes")
    image, template = nib.load(str(patient_path)), nib.load(str(template_path))
    patient_affine = np.asarray(inspect_nifti(patient_path)["affine_ras_mm"])
    template_affine = np.asarray(inspect_nifti(template_path)["affine_ras_mm"])
    patient, moving = image.get_fdata(), template.get_fdata()
    lesion = np.zeros(patient.shape, bool)
    if lesion_path:
        if file_sha256(lesion_path) != report["lesion_sha256"]:
            raise ValueError("Registration lesion source changed")
        lesion = nib.load(str(lesion_path)).get_fdata() > 0
    center_index = np.argwhere(lesion if lesion.any() else patient > 0).mean(axis=0)
    center = (patient_affine @ [*center_index, 1])[:3]
    candidates = [c for c in report["candidates"] if c["status"] != "registration_failed"]
    if not candidates:
        raise ValueError("No registration candidate is available for visual QC")
    figure, axes = plt.subplots(len(candidates), 3, figsize=(12, 3.5 * len(candidates)),
                                squeeze=False, facecolor="#07101e")
    for row, candidate in enumerate(candidates):
        matrix = np.linalg.inv(template_affine) @ np.linalg.inv(candidate["mni_ras_to_patient_ras_mm"]) @ patient_affine
        warped = affine_transform(moving, matrix[:3, :3], matrix[:3, 3], output_shape=patient.shape,
                                  order=1, prefilter=False)
        for column, plane in enumerate(("axial", "coronal", "sagittal")):
            geometry = slice_geometry(patient.shape, patient_affine, center, plane, 384)
            original = sample_slice(patient, patient_affine, geometry)
            aligned = sample_slice(warped, patient_affine, geometry)
            target = sample_slice(lesion.astype(float), patient_affine, geometry, order=0)
            axis = axes[row, column]
            axis.imshow(original, cmap="gray", vmin=0, vmax=np.percentile(patient[patient > 0], 99))
            levels = np.percentile(moving[moving > 0], [20, 55, 80])
            axis.contour(aligned, levels=levels, colors=["#36d6e7"] * 3, linewidths=.4, alpha=.75)
            if target.any():
                axis.contour(target, levels=[.5], colors=["#ff527a"], linewidths=.9)
            correlation = candidate["lesion_excluded_intensity_correlation"]
            correlation_text = "unavailable" if correlation is None else f"{correlation:.3f}"
            axis.set_title(f"{candidate['candidate_id']} / {plane}\n"
                           f"Brain-support Dice {candidate['brain_support_dice']:.3f}; intensity correlation {correlation_text}",
                           color="white", fontsize=9)
            axis.axis("off")
    figure.suptitle("Candidate atlas alignment at tumor centroid\n"
                     "Gray: patient T1 · Cyan: template contours · Pink: source tumor annotation\n"
                     "Population prior · Expert alignment review pending", color="white", fontsize=11)
    figure.tight_layout(rect=(0, 0, 1, .90 if len(candidates) < 3 else .94))
    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(destination, dpi=145, facecolor=figure.get_facecolor())
    plt.close(figure)
    return destination


def main():
    """Run a bounded local registration comparison and save review artifacts."""
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--patient", required=True, type=Path)
    parser.add_argument("--template", required=True, type=Path)
    parser.add_argument("--template-identity", required=True)
    parser.add_argument("--lesion", type=Path)
    parser.add_argument("--case-hash", required=True)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--optimizer", choices=("regular_step", "line_search"), default="regular_step")
    args = parser.parse_args()
    report = register_template_to_patient(args.patient, args.template, lesion_path=args.lesion,
        template_identity=args.template_identity, case_hash=args.case_hash,
        settings=RegistrationSettings(optimizer=args.optimizer))
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "registration_comparison.json").write_text(json.dumps(report, indent=2) + "\n")
    render_registration_qc(args.patient, args.template, report, args.output / "alignment_qc.png", lesion_path=args.lesion)
    print(json.dumps({"report": str(args.output / "registration_comparison.json"),
                      "qc_state": report["qc_state"], "elapsed_seconds": report["elapsed_seconds"]}))


if __name__ == "__main__":
    main()
