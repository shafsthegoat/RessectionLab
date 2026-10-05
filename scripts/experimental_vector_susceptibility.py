#!/usr/bin/env python3
"""Independent synthetic-only physical-vector susceptibility feasibility study.

Implements the density transport equations documented in
docs/vector-susceptibility-phantom.md; contains/imports no GPL solver code.
The optimizer uses distorted native-grid images, affines and physical PE vectors.
Analytic truth is generated and evaluated only by the parent process.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import sys

import nibabel as nib
import numpy as np

import experimental_pyhysco_phantom as simulation

ROOT = Path(__file__).resolve().parents[1]
CONFIG = {
    "schema_version": 1, "experiment": "physical_vector_susceptibility_v1",
    "shape": [96, 96, 60], "control_shape": [12, 12, 8], "precision": "float64",
    "spacing_mm": 2.5, "relative_prescription_degrees": 0.8,
    "maximum_displacement_mm": 12., "smoothness_weight": 300 / 256**2,
    "jacobian_weight": 0.001, "minimum_final_jacobian": 0.2,
    "maximum_iterations": 40, "maximum_function_evaluations": 60,
    "threads": 2, "timeout_seconds_per_case": 55, "rss_limit_bytes": 4 * 1024**3,
    "evaluation_support_fraction": .05, "max_image_error_ratio": .75,
    "max_field_rmse_mm": 1., "max_coordinate_reorientation_difference_mm": .02,
    "patient_execution": False, "motion_eddy_outlier_correction": False,
}


def validate_pair(images, affines, pe_world):
    if len(images) != 2 or len(affines) != 2 or len(pe_world) != 2:
        raise ValueError("Exactly two images, affines and PE directions are required")
    for image, affine, direction in zip(images, affines, pe_world):
        if image.ndim != 3 or min(image.shape) < 3 or not np.isfinite(image).all():
            raise ValueError("Finite 3D images with at least three samples per axis required")
        if affine.shape != (4, 4) or not np.isfinite(affine).all() or not np.allclose(affine[3], [0, 0, 0, 1]):
            raise ValueError("Invalid affine")
        basis = simulation.unit_basis(affine)
        if not np.allclose(basis.T @ basis, np.eye(3), atol=1e-5):
            raise ValueError("Orthogonal physical image axes required")
        if direction.shape != (3,) or not np.isfinite(direction).all() or not np.isclose(np.linalg.norm(direction), 1):
            raise ValueError("Physical PE direction must be a finite unit vector")
    if np.dot(pe_world[0], pe_world[1]) > -.5:
        raise ValueError("This bounded experiment requires broadly opposite PE directions")


def exact_identity_certificate(images, affines, pe_world):
    """Prove a global zero-objective solution for an exactly identical pair.

    This certifies the stated nonnegative mathematical objective, not uniquely
    correct anatomy. Equal arrays on different grids and almost equal arrays
    are explicitly ineligible. No failed numerical result enters this path.
    """
    validate_pair(images, affines, pe_world)
    if not np.array_equal(images[0], images[1]) or not np.array_equal(affines[0], affines[1]):
        return None
    if np.ptp(images[0]) == 0:
        return None
    shape = images[0].shape
    displacement = np.zeros(shape, dtype=float)
    gradient = np.stack(np.gradient(displacement), axis=-1)
    jacobians = [1 + gradient @ direction for direction in pe_world]
    corrected = [image.astype(float) * jac for image, jac in zip(images, jacobians)]
    residual = float(np.sum((corrected[0] - corrected[1])**2))
    smoothness = float(np.sum(gradient**2))
    barrier = float(sum(np.sum(jac - 1 - np.log(jac)) for jac in jacobians))
    certificate = {"method": "analytic_exact_identity", "exact_array_equality": True,
        "exact_affine_equality": True, "data_objective": residual,
        "smoothness_objective": smoothness, "barrier_objective": barrier,
        "total_objective": residual + smoothness + barrier,
        "minimum_jacobian": float(min(j.min() for j in jacobians)),
        "maximum_displacement_mm": float(np.max(np.abs(displacement))),
        "optimizer_called": False, "anatomical_uniqueness_claim": False}
    if certificate["total_objective"] != 0 or certificate["minimum_jacobian"] != 1:
        raise RuntimeError("Independent exact-identity certificate failed")
    return certificate


class PhysicalPair:
    """Native-grid interpolation and physical Jacobians for a scalar mm field."""

    def __init__(self, images, affines, pe_world, *, maximum_displacement_mm=12., dtype=None):
        import torch
        validate_pair(images, affines, pe_world)
        self.torch = torch
        self.dtype = dtype or torch.float32
        self.shape = images[0].shape
        self.affine = affines[0]
        self.spacing = tuple(np.linalg.norm(self.affine[:3, :3], axis=0))
        self.scale = max(float(max(np.max(np.abs(image)) for image in images)), 1e-12)
        self.images = [torch.tensor(image / self.scale, dtype=self.dtype)[None, None] for image in images]
        reference_voxels = np.moveaxis(np.indices(self.shape, dtype=float), 0, -1)
        reference_basis = simulation.unit_basis(self.affine)
        self.pe_reference = [torch.tensor(reference_basis.T @ direction, dtype=self.dtype) for direction in pe_world]
        self.sampling = []
        mask = np.ones(self.shape, dtype=bool)
        for image, affine, direction in zip(images, affines, pe_world):
            inverse = np.linalg.inv(affine)
            # The reference image samples its own exact integer grid. A
            # world->voxel round trip introduces tiny signed errors at the
            # nonsmooth knots of a trilinear interpolant. Other images use
            # their full relative affine and never have their headers erased.
            mapping = np.linalg.solve(affine, self.affine)
            native_voxels = reference_voxels if np.array_equal(affine, self.affine) else (
                reference_voxels @ mapping[:3, :3].T + mapping[:3, 3])
            vector_voxels_per_mm = inverse[:3, :3] @ direction
            margin = maximum_displacement_mm / np.linalg.norm(affine[:3, :3], axis=0) + 1
            mask &= np.all((native_voxels >= margin) & (native_voxels <= np.array(image.shape) - 1 - margin), axis=-1)
            self.sampling.append((torch.tensor(native_voxels, dtype=self.dtype),
                                  torch.tensor(vector_voxels_per_mm, dtype=self.dtype), image.shape))
        if mask.sum() < 100:
            raise ValueError("Insufficient fixed geometric overlap after displacement margin")
        self.mask = torch.tensor(mask)

    def correct(self, field):
        import torch.nn.functional as functional
        gradients = self.torch.stack(self.torch.gradient(field, spacing=self.spacing), dim=-1)
        corrected, jacobians = [], []
        for image, direction, (base, vector, shape) in zip(self.images, self.pe_reference, self.sampling):
            native = base + field[..., None] * vector
            factors = self.torch.tensor(shape, dtype=self.dtype) - 1
            grid = (2 * native / factors - 1)[..., [2, 1, 0]][None]
            sampled = functional.grid_sample(image, grid, mode="bilinear", padding_mode="zeros",
                                              align_corners=True)[0, 0]
            jacobian = 1 + (gradients * direction).sum(dim=-1)
            corrected.append(sampled * jacobian)
            jacobians.append(jacobian)
        return corrected, jacobians, gradients

    def objective(self, field, smoothness_weight, jacobian_weight):
        corrected, jacobians, gradients = self.correct(field)
        data = .5 * ((corrected[0] - corrected[1])[self.mask]**2).mean()
        smoothness = .5 * (gradients**2).sum(dim=-1).mean()
        # Smooth finite extension below epsilon allows line search to recover
        # from bad trial steps; a positive final Jacobian remains mandatory.
        barrier = self.torch.zeros((), dtype=field.dtype)
        for jac in jacobians:
            safe = jac.clamp_min(.001)
            barrier = barrier + (safe - 1 - safe.log() + ((.001 - jac).clamp_min(0) / .001)**2).mean()
        objective = data + smoothness_weight * smoothness + jacobian_weight * barrier
        return objective, {"data": data, "smoothness": smoothness, "barrier": barrier}


def optimize_pair(images, affines, pe_world, config):
    import torch
    import torch.nn.functional as functional
    certificate = exact_identity_certificate(images, affines, pe_world)
    if certificate:
        return np.zeros(images[0].shape, np.float32), images, {
            "status": "certified_identity", "certificate": certificate,
            "minimum_jacobian": 1., "iterations": 0, "function_evaluations": 0}
    dtype = getattr(torch, config["precision"])
    pair = PhysicalPair(images, affines, pe_world, maximum_displacement_mm=config["maximum_displacement_mm"], dtype=dtype)
    controls = torch.zeros((1, 1, *config["control_shape"]), dtype=dtype, requires_grad=True)

    def field():
        bounded = config["maximum_displacement_mm"] * torch.tanh(controls)
        return functional.interpolate(bounded, size=pair.shape, mode="trilinear", align_corners=True)[0, 0]

    optimizer = torch.optim.LBFGS([controls], lr=1., max_iter=config["maximum_iterations"],
        max_eval=config["maximum_function_evaluations"], history_size=10,
        tolerance_grad=1e-8, tolerance_change=1e-10, line_search_fn="strong_wolfe")
    history = []

    def closure():
        optimizer.zero_grad()
        loss, terms = pair.objective(field(), config["smoothness_weight"], config["jacobian_weight"])
        if not torch.isfinite(loss):
            raise RuntimeError("Nonfinite trial objective")
        loss.backward()
        history.append({"objective": float(loss.detach()), **{k: float(v.detach()) for k, v in terms.items()}})
        return loss

    optimizer.step(closure)
    with torch.no_grad():
        estimate = field()
        final, terms = pair.objective(estimate, config["smoothness_weight"], config["jacobian_weight"])
        corrected, jacobians, _ = pair.correct(estimate)
        minimum = min(float(j.min()) for j in jacobians)
        maximum = float(estimate.abs().max())
        state = optimizer.state[controls]
        metrics = {"status": "bounded_optimization_completed", "objective_initial": history[0],
            "objective_final": {"objective": float(final), **{k: float(v) for k, v in terms.items()}},
            "minimum_jacobian": minimum, "maximum_displacement_mm": maximum,
            "iterations": state["n_iter"], "function_evaluations": state["func_evals"],
            "fixed_geometric_support_voxels": int(pair.mask.sum()), "history": history,
            "pe_in_reference_basis": [v.tolist() for v in pair.pe_reference],
            "geometry_candidate_pass": minimum > config["minimum_final_jacobian"] and maximum < config["maximum_displacement_mm"]}
        return estimate.numpy(), [image.numpy() * pair.scale for image in corrected], metrics


def worker(pair_manifest):
    import torch
    torch.set_num_threads(2)
    torch.set_num_interop_threads(1)
    metadata = json.loads(pair_manifest.read_text())
    if metadata.get("synthetic_only") is not True:
        raise ValueError("This experiment only accepts generated synthetic manifests")
    directory = pair_manifest.parent
    inputs = [nib.load(directory / name) for name in metadata["images"]]
    arrays = [image.get_fdata(dtype=np.float32) for image in inputs]
    affines = [image.affine for image in inputs]
    # PE belongs to each serialized native image's voxel axes, as in BIDS.
    # Derive physical vectors from those authoritative stored affines rather
    # than pairing them with pre-serialization, differently rounded vectors.
    directions = [simulation.unit_basis(a) @ np.asarray(p) for a, p in zip(affines, metadata["pe_voxel"])]
    field, corrected, metrics = optimize_pair(arrays, affines, directions, metadata["config"])
    simulation.save_image(directory / "estimated-field-mm.nii.gz", field, affines[0])
    for index, array in enumerate(corrected):
        simulation.save_image(directory / f"corrected-{index}.nii.gz", array, affines[0])
    metrics["native_input_affines"] = [a.tolist() for a in affines]
    metrics["pe_world_from_stored_affines"] = [p.tolist() for p in directions]
    metrics["output_affine"] = affines[0].tolist()
    metrics["threads"] = torch.get_num_threads()
    (directory / "optimizer-result.json").write_text(json.dumps(metrics, indent=2) + "\n")


def run(output):
    output.mkdir(parents=True, exist_ok=False)
    config = dict(CONFIG)
    config["script_sha256"] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    config["simulator_sha256"] = hashlib.sha256(Path(simulation.__file__).read_bytes()).hexdigest()
    (output / "frozen-config.json").write_text(json.dumps(config, indent=2, sort_keys=True) + "\n")
    shape = config["shape"]
    affine = simulation.make_affine(shape)
    basis = simulation.unit_basis(affine)
    truth = simulation.analytic_density(simulation.grid_world(shape, affine), basis)
    support = truth > config["evaluation_support_fraction"] * truth.max()
    results = {"config": config, "cases": {}, "patient_processing": False,
               "production_gate_changes": False, "truth_in_optimizer": False}
    cases = [("zero", 0., 0., False), ("same_grid", 4., 0., False),
             ("native_rotated", 4., .8, False), ("negative_field", -4., .8, False),
             ("coordinate_reorientation", 4., .8, True)]
    for name, amplitude, angle, reorient in cases:
        directory = output / name
        directory.mkdir()
        native_affines = [affine.copy(), simulation.make_affine(shape, angle)]
        directions = [simulation.unit_basis(a)[:, 1] * sign for a, sign in zip(native_affines, [1, -1])]
        arrays, qc = [], []
        for a, p in zip(native_affines, directions):
            image, checks = simulation.distorted_image(shape, a, basis, p, amplitude)
            arrays.append(image)
            qc.append(checks)
        expected_field, _ = simulation.analytic_field(simulation.grid_world(shape, affine), basis, amplitude)
        if reorient:
            transform = np.eye(4)
            transform[:3, :3] = simulation.rotation(0, 23) @ simulation.rotation(2, -31)
            transform[:3, 3] = [32., -15., 8.]
            native_affines = [transform @ a for a in native_affines]
            directions = [transform[:3, :3] @ p for p in directions]
        for index, (array, a) in enumerate(zip(arrays, native_affines)):
            simulation.save_image(directory / f"image-{index}.nii.gz", array, a)
        # This manifest deliberately excludes amplitude, truth arrays, field
        # formula, evaluation mask and error thresholds from optimizer inputs.
        optimizer_config = {k: config[k] for k in ["control_shape", "precision", "maximum_displacement_mm", "smoothness_weight",
            "jacobian_weight", "maximum_iterations", "maximum_function_evaluations", "minimum_final_jacobian"]}
        manifest = {"synthetic_only": True, "images": ["image-0.nii.gz", "image-1.nii.gz"],
                    "pe_voxel": [[0., 1., 0.], [0., -1., 0.]],
                    "pe_world_before_header_serialization": [p.tolist() for p in directions], "config": optimizer_config}
        (directory / "pair.json").write_text(json.dumps(manifest, indent=2) + "\n")
        env = os.environ.copy()
        env.update({"OMP_NUM_THREADS": "2", "OPENBLAS_NUM_THREADS": "2", "MKL_NUM_THREADS": "2"})
        command = [sys.executable, str(Path(__file__).resolve()), "--worker", str(directory / "pair.json")]
        resources = simulation.bounded_run(command, directory / "worker.log", env,
                                           config["timeout_seconds_per_case"], config["rss_limit_bytes"])
        record = {"resources": resources, "forward_simulation_qc": qc}
        if resources["returncode"]:
            record["status"] = "worker_failed_or_limited"
        else:
            optimization = json.loads((directory / "optimizer-result.json").read_text())
            record["optimization"] = optimization
            estimate = nib.load(directory / "estimated-field-mm.nii.gz").get_fdata()
            corrected = [nib.load(directory / f"corrected-{i}.nii.gz").get_fdata() for i in (0, 1)]
            # Evaluation-only regridding supplies the raw PA error baseline.
            baseline_pa = simulation.regrid(arrays[1], native_affines[1], shape, native_affines[0])
            before = np.mean([simulation.relative_rmse(a, truth, support) for a in [arrays[0], baseline_pa]])
            after = np.mean([simulation.relative_rmse(a, truth, support) for a in corrected])
            field_error = float(np.sqrt(np.mean((estimate[support] - expected_field[support])**2)))
            record.update({"status": "evaluated", "image_relative_rmse_before": float(before),
                "image_relative_rmse_after": float(after), "field_rmse_mm": field_error,
                "zero_field_baseline_rmse_mm": float(np.sqrt(np.mean(expected_field[support]**2))),
                "image_error_ratio": float(after / before) if before > 1e-7 else None,
                "output_affine_exactly_preserved": bool(np.array_equal(nib.load(directory / "estimated-field-mm.nii.gz").affine,
                                                                          nib.load(directory / "image-0.nii.gz").affine)),
                "numerical_checks_pass": bool(after < 1e-6 and field_error == 0) if amplitude == 0 else
                    bool(after / before < config["max_image_error_ratio"] and field_error < config["max_field_rmse_mm"]
                         and optimization["geometry_candidate_pass"]),
                "clinical_acceptance": False})
            truth_dir = directory / "evaluation-only"
            truth_dir.mkdir()
            simulation.save_image(truth_dir / "true-field-mm.nii.gz", expected_field, native_affines[0])
            simulation.save_image(truth_dir / "true-image.nii.gz", truth, native_affines[0])
        results["cases"][name] = record
        (output / "results.json").write_text(json.dumps(results, indent=2) + "\n")
    if all(results["cases"][name]["status"] == "evaluated" for name in ["native_rotated", "coordinate_reorientation"]):
        original = nib.load(output / "native_rotated/estimated-field-mm.nii.gz").get_fdata()
        reoriented = nib.load(output / "coordinate_reorientation/estimated-field-mm.nii.gz").get_fdata()
        diff = float(np.max(np.abs(original - reoriented)))
        results["coordinate_invariance"] = {"max_field_difference_mm": diff,
            "passed": diff < config["max_coordinate_reorientation_difference_mm"]}
    baseline_path = ROOT / "artifacts/pyhysco-phantom-v1/results.json"
    if baseline_path.exists():
        baseline = json.loads(baseline_path.read_text())["cases"]["known_distortion"]
        results["previous_pyhysco_same_grid_baseline"] = {"artifact_sha256": hashlib.sha256(baseline_path.read_bytes()).hexdigest(),
            "image_relative_rmse_after": baseline["image_relative_rmse_after_mean"], "field_rmse_mm": baseline["field_rmse_mm"],
            "wall_seconds": baseline["resources"]["wall_seconds"], "comparison_is_single_phantom_not_equivalence": True}
    results["status"] = "bounded_synthetic_feasibility_completed"
    (output / "results.json").write_text(json.dumps(results, indent=2) + "\n")
    print(json.dumps({"status": results["status"], "cases": {k: v.get("numerical_checks_pass") for k, v in results["cases"].items()},
                      "coordinate_invariance": results.get("coordinate_invariance")}, indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / "outputs/vector_susceptibility/v1")
    parser.add_argument("--worker", type=Path, help=argparse.SUPPRESS)
    arguments = parser.parse_args()
    if arguments.worker:
        worker(arguments.worker.resolve())
    else:
        run(arguments.output.resolve())


if __name__ == "__main__":
    main()
