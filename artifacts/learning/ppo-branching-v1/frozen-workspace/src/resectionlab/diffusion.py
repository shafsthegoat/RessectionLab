"""Auditable diffusion reconstruction in the acquired image frame.

Raw acquisitions may produce explicitly diagnostic derivatives. They cannot
silently become tract-aware planning evidence. No output identifies a named
functional tract or estimates a postoperative outcome.
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass, field
from datetime import datetime, timezone
import json
from pathlib import Path
import time
from typing import Any

import nibabel as nib
import numpy as np

from .imaging import ImagingError, file_sha256, inspect_nifti


class DiffusionError(ImagingError):
    """Named diffusion input, preprocessing, or reconstruction failure."""


def image_axis_basis(affine: np.ndarray) -> np.ndarray:
    """Unit image-axis columns in RAS; preserve handedness, reject shear."""
    affine = np.asarray(affine, dtype=float)
    if (affine.shape != (4, 4) or not np.isfinite(affine).all()
            or not np.allclose(affine[3], [0, 0, 0, 1])):
        raise DiffusionError("INVALID_AFFINE", "Expected a finite homogeneous voxel-to-RAS affine.")
    spacing = np.linalg.norm(affine[:3, :3], axis=0)
    if (spacing <= 0).any():
        raise DiffusionError("INVALID_AFFINE", "Image spacing must be positive.")
    basis = affine[:3, :3] / spacing
    if not np.allclose(basis.T @ basis, np.eye(3), atol=1e-4):
        raise DiffusionError("GRADIENT_FRAME_SHEARED", "Resample a sheared grid with reviewed gradient rotation.")
    return basis


def gradients_in_image_axes(bvecs: np.ndarray, affine: np.ndarray, convention: str) -> np.ndarray:
    """Convert explicitly declared vectors to physical unit image-axis components.

    BIDS/FSL flips the first component for right-handed NIfTI axes. World vectors
    rotate without voxel scaling. Per-volume motion rotation is a separate gate.
    """
    vectors = np.array(bvecs, dtype=float, copy=True)
    if vectors.ndim != 2 or vectors.shape[1] != 3 or not np.isfinite(vectors).all():
        raise DiffusionError("INVALID_GRADIENT_VALUES", "Expected finite N×3 vectors.")
    basis = image_axis_basis(affine)
    if convention == "BIDS_FSL":
        if np.linalg.det(basis) > 0:
            vectors[:, 0] *= -1
    elif convention == "world_RAS":
        vectors = vectors @ basis
    elif convention == "world_LPS":
        vectors = (vectors * [-1, -1, 1]) @ basis
    elif convention != "image_axes":
        raise DiffusionError("GRADIENT_FRAME_UNRESOLVED", "Declare BIDS_FSL, image_axes, world_RAS, or world_LPS.")
    return vectors


def validate_gradients(bvals: np.ndarray, bvecs: np.ndarray, volumes: int) -> None:
    if bvals.shape != (volumes,) or bvecs.shape != (volumes, 3):
        raise DiffusionError("GRADIENT_COUNT_MISMATCH", "One b-value and vector are required for every DWI volume.")
    if not np.isfinite(bvals).all() or not np.isfinite(bvecs).all() or (bvals < 0).any():
        raise DiffusionError("INVALID_GRADIENT_VALUES", "Gradients must be finite and b-values nonnegative.")
    weighted = bvals > 50
    if not (~weighted).any() or weighted.sum() < 6:
        raise DiffusionError("INSUFFICIENT_DIRECTIONS", "Need b0 and at least six weighted directions.")
    norm = np.linalg.norm(bvecs, axis=1)
    if np.any(abs(norm[weighted] - 1) > 0.01):
        raise DiffusionError("GRADIENT_NORM_INVALID", "Weighted vectors must have unit norm within 0.01.")
    if np.any((norm[~weighted] > 0.01) & (abs(norm[~weighted] - 1) > 0.01)):
        raise DiffusionError("B0_GRADIENT_NORM_INVALID", "A b0 vector must be zero or unit length.")
    g = bvecs[weighted]
    design = np.column_stack((g[:, 0] ** 2, g[:, 1] ** 2, g[:, 2] ** 2,
                              2*g[:, 0]*g[:, 1], 2*g[:, 0]*g[:, 2], 2*g[:, 1]*g[:, 2]))
    if np.linalg.matrix_rank(design) < 6:
        raise DiffusionError("DEGENERATE_GRADIENT_DIRECTIONS", "Directions do not identify a diffusion tensor.")


@dataclass
class DiffusionData:
    signal: np.ndarray
    affine_ras_mm: np.ndarray
    bvals: np.ndarray
    bvecs_image: np.ndarray
    gradient_convention: str
    source_hashes: dict[str, str] = field(default_factory=dict)
    header_qc: dict[str, Any] = field(default_factory=dict)

    def validate(self) -> None:
        if self.signal.ndim != 4 or not np.isfinite(self.signal).all() or (self.signal < 0).any():
            raise DiffusionError("INVALID_DIFFUSION_SIGNAL", "DWI must be a finite, nonnegative 4D array.")
        image_axis_basis(self.affine_ras_mm)
        validate_gradients(self.bvals, self.bvecs_image, self.signal.shape[-1])


def load_diffusion(dwi_path: str | Path, bvals_path: str | Path | None,
                   bvecs_path: str | Path | None, *, gradient_convention: str) -> DiffusionData:
    if bvals_path is None or bvecs_path is None:
        raise DiffusionError("MISSING_GRADIENTS", "Raw directional diffusion needs b-values and b-vectors.")
    for path in (dwi_path, bvals_path, bvecs_path):
        if not Path(path).is_file():
            raise DiffusionError("MISSING_DIFFUSION_INPUT", f"Missing input {Path(path).name}.")
    qc = inspect_nifti(dwi_path, dimensions=4)
    bvals = np.loadtxt(bvals_path, dtype=float).reshape(-1)
    bvecs = np.loadtxt(bvecs_path, dtype=float)
    if bvecs.shape == (3, qc["shape"][-1]):
        bvecs = bvecs.T
    validate_gradients(bvals, bvecs, qc["shape"][-1])
    affine = np.asarray(qc["affine_ras_mm"])
    data = DiffusionData(
        nib.load(dwi_path).get_fdata(dtype=np.float32), affine, bvals,
        gradients_in_image_axes(bvecs, affine, gradient_convention), gradient_convention,
        {"dwi": file_sha256(dwi_path), "bvals": file_sha256(bvals_path), "bvecs": file_sha256(bvecs_path)}, qc,
    )
    data.validate()
    return data


def planning_gates(data: DiffusionData, evidence: dict[str, Any] | None) -> list[str]:
    """Check preprocessing provenance, not merely a caller-supplied ready boolean."""
    evidence = evidence or {}
    issues = []
    for key, code in (
        ("motion_corrected", "DWI_MOTION_UNCORRECTED"),
        ("distortion_corrected", "DWI_DISTORTION_UNCORRECTED"),
        ("gradient_rotation_reviewed", "GRADIENT_ROTATION_UNREVIEWED"),
        ("brain_mask_reviewed", "DIFFUSION_MASK_UNREVIEWED"),
        ("registration_reviewed", "DIFFUSION_REGISTRATION_UNREVIEWED"),
    ):
        if evidence.get(key) is not True:
            issues.append(code)
    if not evidence.get("correction_provenance"):
        issues.append("PREPROCESSING_PROVENANCE_MISSING")
    if (not data.source_hashes or evidence.get("corrected_dwi_sha256") != data.source_hashes.get("dwi")
            or evidence.get("rotated_bvec_sha256") != data.source_hashes.get("bvecs")):
        issues.append("PREPROCESSING_EVIDENCE_STALE")
    transform = np.asarray(evidence.get("dwi_RAS_mm_to_T1_RAS_mm", []), dtype=float)
    if (transform.shape != (4, 4) or not np.isfinite(transform).all()
            or not np.allclose(transform[3], [0, 0, 0, 1])
            or not np.allclose(transform[:3, :3].T @ transform[:3, :3], np.eye(3), atol=1e-4)
            or not np.isclose(np.linalg.det(transform[:3, :3]), 1.0, atol=1e-4)
            or not evidence.get("structural_sha256")):
        issues.append("DIFFUSION_REGISTRATION_TRANSFORM_UNRESOLVED")
    return issues


def _mask(data: DiffusionData, mask: np.ndarray) -> np.ndarray:
    mask = np.asarray(mask)
    if mask.shape != data.signal.shape[:3] or mask.dtype != bool or not mask.any():
        raise DiffusionError("INVALID_DIFFUSION_MASK", "Supply a nonempty boolean mask on the DWI grid.")
    if np.any(data.signal[..., data.bvals <= 50].mean(axis=-1)[mask] <= 0):
        raise DiffusionError("NONPOSITIVE_BASELINE", "Every fitted voxel requires positive mean b0 signal.")
    return mask


def fit_tensor(data: DiffusionData, mask: np.ndarray, *, diagnostic_only: bool = False,
               evidence: dict[str, Any] | None = None, max_b: float = 1200,
               chunk_voxels: int = 2048) -> dict[str, Any]:
    """Fit bounded WLS low-shell tensors; unknown coverage remains NaN plus mask."""
    from dipy.core.gradients import gradient_table
    from dipy.reconst.dti import TensorModel

    data.validate()
    mask = _mask(data, mask)
    gates = planning_gates(data, evidence)
    if gates and not diagnostic_only:
        raise DiffusionError(gates[0], "Planning reconstruction blocked: " + ", ".join(gates))
    if not np.isfinite(max_b) or not 50 < max_b <= 1500 or chunk_voxels < 1:
        raise DiffusionError("INVALID_TENSOR_CONFIGURATION", "Low-shell tensor fit needs 50<max_b<=1500 and positive chunks.")
    selected = data.bvals <= max_b
    validate_gradients(data.bvals[selected], data.bvecs_image[selected], int(selected.sum()))
    gtab = gradient_table(data.bvals[selected], bvecs=data.bvecs_image[selected], b0_threshold=50)
    model = TensorModel(gtab, fit_method="WLS")
    shape = mask.shape
    fa = np.full(shape, np.nan, dtype=np.float32)
    md = np.full(shape, np.nan, dtype=np.float32)
    principal = np.full((*shape, 3), np.nan, dtype=np.float32)
    residual = np.full(shape, np.nan, dtype=np.float32)
    basis = image_axis_basis(data.affine_ras_mm)
    indices = np.flatnonzero(mask)
    flat_signal = data.signal.reshape(-1, data.signal.shape[-1])
    start = time.perf_counter()
    for offset in range(0, len(indices), chunk_voxels):
        index = indices[offset:offset + chunk_voxels]
        signal = flat_signal[index][:, selected]
        baseline = signal[:, data.bvals[selected] <= 50].mean(axis=1)
        fit = model.fit(signal)
        fa.flat[index] = fit.fa
        md.flat[index] = fit.md
        principal.reshape(-1, 3)[index] = fit.evecs[..., :, 0] @ basis.T
        prediction = fit.predict(gtab, S0=baseline)
        residual.flat[index] = np.sqrt(np.mean((prediction - signal) ** 2, axis=1)) / baseline
    if not np.isfinite(fa[mask]).all() or not np.isfinite(principal[mask]).all():
        raise DiffusionError("TENSOR_FIT_NONFINITE", "Tensor fitting produced invalid fitted voxels.")
    report = {
        "model": "DIPY WLS diffusion tensor",
        "mode": "diagnostic_only" if diagnostic_only else "preprocessing_reviewed_reconstruction",
        "elapsed_seconds": time.perf_counter() - start,
        "selected_volume_indices": np.flatnonzero(selected).tolist(),
        "selected_bvalues_s_per_mm2": sorted(set(data.bvals[selected].tolist())),
        "excluded_volume_count": int((~selected).sum()), "chunk_voxels": chunk_voxels,
        "fitted_voxels": int(mask.sum()), "native_grid_voxels": int(mask.size),
        "fa_median_in_mask": float(np.median(fa[mask])),
        "fa_p95_in_mask": float(np.percentile(fa[mask], 95)),
        "mean_diffusivity_median_mm2_per_s": float(np.median(md[mask])),
        "normalized_signal_rmse_median": float(np.median(residual[mask])),
        "normalized_signal_rmse_definition": "RMSE against measured volumes / mean b0 signal; diagnostic fit residual",
        "planning_gate_failures": gates,
        "usable_for_tract_aware_planning": False,
        "motor_coverage": "unknown", "language_coverage": "unknown",
        "clinical_deficit_probability": None,
        "output_frame": "acquired_DWI_RAS_mm",
        "principal_vector_components": "unit RAS, antipodally symmetric",
        "source_affine_ras_mm": data.affine_ras_mm.tolist(),
        "source_hashes": data.source_hashes,
        "gradient_convention": data.gradient_convention,
        "limitations": ["single_tensor_cannot_resolve_crossing_fibers", "mask_requires_review",
                        "no_functional_bundle_or_endpoint_validation", "not_registered_to_structural_image"],
    }
    return {"fa": fa, "md": md, "principal_ras": principal, "normalized_rmse": residual,
            "fit_coverage": mask.copy(), "report": report}


def probabilistic_crop_audit(data: DiffusionData, tensor: dict[str, Any], *,
                              crop_start: tuple[int, int, int], crop_shape: tuple[int, int, int] = (24, 24, 24),
                              shell_b: float = 2800, random_seed: int = 41,
                              seed_count: int = 64, realizations: int = 3) -> dict[str, Any]:
    """Bounded CSA probabilistic tracks for inspection; never a functional prior.

    Repeated tracking samples only a fixed fitted ODF. It does not bootstrap the
    acquisition, quantify total anatomical uncertainty, or establish coverage.
    """
    from dipy.core.gradients import gradient_table
    from dipy.data import small_sphere
    from dipy.direction import ProbabilisticDirectionGetter
    from dipy.reconst.shm import CsaOdfModel
    from dipy.tracking.local_tracking import LocalTracking
    from dipy.tracking.stopping_criterion import BinaryStoppingCriterion

    data.validate()
    if (tensor.get("report", {}).get("source_hashes") != data.source_hashes
            or tensor["fa"].shape != data.signal.shape[:3]
            or tensor["fit_coverage"].shape != data.signal.shape[:3]):
        raise DiffusionError("STALE_TENSOR_DERIVATIVE", "The tracking mask must belong to this DWI acquisition and grid.")
    start_index = np.asarray(crop_start, dtype=int)
    shape = np.asarray(crop_shape, dtype=int)
    if (start_index.shape != (3,) or shape.shape != (3,) or (start_index < 0).any()
            or (shape < 3).any() or (shape > 32).any()
            or (start_index + shape > np.array(data.signal.shape[:3])).any()
            or not 1 <= seed_count <= 256 or not 2 <= realizations <= 8):
        raise DiffusionError("INVALID_TRACKING_AUDIT_BUDGET", "Use an in-bounds 3–32 voxel crop, 1–256 seeds, and 2–8 runs.")
    selected = (data.bvals <= 50) | (abs(data.bvals - shell_b) <= 50)
    weighted_count = int(np.count_nonzero(selected & (data.bvals > 50)))
    if weighted_count < 28 or shell_b < 1500:
        raise DiffusionError("CSA_SHELL_UNSUPPORTED", "The diagnostic order-6 CSA audit needs ≥28 directions on a b≥1500 shell.")
    validate_gradients(data.bvals[selected], data.bvecs_image[selected], int(selected.sum()))
    slices = tuple(slice(int(a), int(a + b)) for a, b in zip(start_index, shape))
    signal = data.signal[slices][..., selected]
    mask = tensor["fit_coverage"][slices] & (tensor["fa"][slices] >= 0.2)
    locations = np.argwhere(mask)
    if len(locations) < 1:
        raise DiffusionError("NO_TRACKING_SEEDS", "No FA≥0.2 fitted voxel exists in the requested crop.")
    gtab = gradient_table(data.bvals[selected], bvecs=data.bvecs_image[selected])
    started = time.perf_counter()
    model = CsaOdfModel(gtab, sh_order_max=6)
    if np.linalg.matrix_rank(model.B) < 28:
        raise DiffusionError("CSA_GRADIENT_DESIGN_DEGENERATE", "The selected shell does not identify all 28 order-6 SH coefficients.")
    odf = model.fit(signal, mask=mask).odf(small_sphere)
    if not np.isfinite(odf).all():
        raise DiffusionError("CSA_FIT_NONFINITE", "CSA reconstruction produced nonfinite ODF values.")
    negative_fraction = float(np.mean(odf[mask] < 0))
    pmf = np.maximum(odf, 0)
    pmf[~mask] = 0
    direction_getter = ProbabilisticDirectionGetter.from_pmf(
        pmf, max_angle=30, sphere=small_sphere, pmf_threshold=0.1,
    )
    spacing = np.linalg.norm(data.affine_ras_mm[:3, :3], axis=0)
    axis_mm_affine = np.diag([*spacing, 1.0])
    rng = np.random.default_rng(random_seed)
    chosen = locations[rng.choice(len(locations), min(seed_count, len(locations)), replace=False)]
    seeds_axis_mm = chosen * spacing
    stopping = BinaryStoppingCriterion(mask)
    streamlines = []
    run_ids = []
    run_summaries = []
    occupied_runs = np.zeros(tuple(shape), dtype=np.uint8)
    for realization in range(realizations):
        generator = LocalTracking(direction_getter, stopping, seeds_axis_mm, axis_mm_affine, 0.75,
                                  max_cross=1, maxlen=160, return_all=True,
                                  random_seed=random_seed + realization)
        visited = np.zeros(tuple(shape), dtype=bool)
        lengths = []
        discarded = 0
        for points_axis_mm in generator:
            voxels = np.asarray(points_axis_mm) / spacing
            points = nib.affines.apply_affine(data.affine_ras_mm, voxels + start_index)
            length = float(np.linalg.norm(np.diff(points, axis=0), axis=1).sum())
            if length < 5:
                discarded += 1
                continue
            streamlines.append(points.astype(np.float32))
            run_ids.append(realization)
            lengths.append(length)
            index = np.rint(voxels).astype(int)
            inside = ((index >= 0) & (index < shape)).all(axis=1)
            visited[tuple(index[inside].T)] = True
        occupied_runs += visited
        run_summaries.append({"seed": random_seed + realization, "retained_streamlines": len(lengths),
                              "discarded_under_5mm": discarded,
                              "length_median_mm": float(np.median(lengths)) if lengths else None,
                              "visited_voxels": int(visited.sum())})
    if not streamlines:
        raise DiffusionError("NO_DIAGNOSTIC_STREAMLINES", "The fixed diagnostic protocol returned no path ≥5 mm.")
    offsets = np.r_[0, np.cumsum([len(line) for line in streamlines])].astype(np.int64)
    report = {
        "model": "DIPY CSA order 6; nonnegative ODF samples used for probabilistic tracking",
        "spherical_harmonic_basis": "DIPY CsaOdfModel legacy descoteaux07 default; pending upstream deprecation",
        "csa_smoothing": 0.006,
        "mode": "diagnostic_only", "elapsed_seconds": time.perf_counter() - started,
        "shell_b_s_per_mm2": shell_b, "weighted_direction_count": weighted_count,
        "crop_start_voxels": start_index.tolist(), "crop_shape_voxels": shape.tolist(),
        "seed_count_per_realization": len(chosen), "realizations": run_summaries,
        "step_size_mm": 0.75, "maximum_turn_degrees": 30, "max_steps_per_half": 160,
        "stopping_rule": "unreviewed low-shell FA>=0.2 fitted-mask boundary",
        "negative_odf_fraction_before_clipping": negative_fraction,
        "streamline_count": len(streamlines), "point_frame": "acquired_DWI_RAS_mm",
        "functional_labels": [], "endpoint_review": "not_performed",
        "motor_coverage": "unknown", "language_coverage": "unknown",
        "clinical_deficit_probability": None, "usable_for_tract_aware_planning": False,
        "visitation_meaning": "fraction of fixed-ODF tracking runs visiting a crop voxel; not an anatomical or clinical probability",
        "limitations": ["raw_motion_and_distortion_uncorrected_unless_separately_documented",
                        "crop_and_FA_stopping_truncate_paths", "crossing_tracts_unvalidated",
                        "single_ODF_fit_does_not_measure_reconstruction_uncertainty",
                        "unvisited_voxels_do_not_establish_absent_fibers", "no_CST_or_language_bundle_identification"],
    }
    return {"points_ras_mm": np.concatenate(streamlines), "offsets": offsets,
            "realization_ids": np.asarray(run_ids, dtype=np.int16),
            "run_visitation_fraction": occupied_runs.astype(np.float32) / realizations, "report": report}


def _save_volume(path: Path, data: np.ndarray, affine: np.ndarray) -> None:
    image = nib.Nifti1Image(data, affine)
    image.header.set_xyzt_units("mm")
    image.set_sform(affine, code=1)
    image.set_qform(affine, code=1)
    nib.save(image, path)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dwi", type=Path, required=True)
    parser.add_argument("--bvals", type=Path, required=True)
    parser.add_argument("--bvecs", type=Path, required=True)
    parser.add_argument("--gradient-convention", choices=["BIDS_FSL", "image_axes", "world_RAS", "world_LPS"], required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--diagnostic-only", action="store_true", help="Allow explicitly labeled raw-signal diagnostics; never planning export")
    parser.add_argument("--probabilistic-crop", action="store_true")
    args = parser.parse_args()
    from dipy import __version__ as dipy_version
    from dipy.segment.mask import median_otsu

    data = load_diffusion(args.dwi, args.bvals, args.bvecs, gradient_convention=args.gradient_convention)
    baseline = data.signal[..., data.bvals <= 50].mean(axis=-1)
    _, mask = median_otsu(baseline, median_radius=2, numpass=2, dilate=1)
    mask &= baseline > 0
    tensor = fit_tensor(data, mask, diagnostic_only=args.diagnostic_only)
    args.output.mkdir(parents=True, exist_ok=True)
    for name in ("fa", "md", "principal_ras", "normalized_rmse", "fit_coverage"):
        array = tensor[name].astype(np.uint8) if name == "fit_coverage" else tensor[name]
        _save_volume(args.output / f"{name}.nii.gz", array, data.affine_ras_mm)
    report = {"schema_version": 1, "created_at": datetime.now(timezone.utc).isoformat(),
              "implementation_sha256": file_sha256(__file__),
              "dipy_version": dipy_version, "tensor": tensor["report"],
              "mask": {"method": "median_otsu on mean b0", "median_radius": 2, "passes": 2,
                       "dilate": 1, "reviewed": False}, "failed_variants": []}
    if args.probabilistic_crop:
        crop_start = tuple((np.array(mask.shape) - 24) // 2)
        try:
            tracking = probabilistic_crop_audit(data, tensor, crop_start=crop_start)
            np.savez_compressed(args.output / "diagnostic_probabilistic_tracks.npz",
                                **{k: v for k, v in tracking.items() if k != "report"})
            report["probabilistic_crop"] = tracking["report"]
        except DiffusionError as error:
            report["failed_variants"].append({"stage": "probabilistic_crop", "code": error.code, "reason": str(error)})
    report["artifact_hashes"] = {path.name: file_sha256(path) for path in args.output.iterdir() if path.is_file() and path.suffix != ".json"}
    (args.output / "reconstruction_report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
