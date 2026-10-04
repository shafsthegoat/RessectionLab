#!/usr/bin/env python3
"""Bounded, synthetic-only audit of the optional external PyHySCO 0.0.4 tool.

No patient-input argument is provided. GPL code is installed separately, never
imported into the application or this evaluator, and is not redistributed here.
The solver receives only generated distorted images and frozen hyperparameters;
truth, physical simulation, output adaptation and evaluation stay outside it.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
import zipfile

import nibabel as nib
import numpy as np
from scipy.ndimage import affine_transform

ROOT = Path(__file__).resolve().parents[1]
WHEEL_SHA256 = "589e6509602b98b5f275d6e772c8ead8c8641ed1f4f20e2e24efaefb022bd884"
CONFIG = {
    "schema_version": 1, "experiment": "analytic_pyhysco_geometry_v1",
    "shape": [96, 96, 60], "spacing_mm": 2.5, "relative_rotation_deg": 0.8,
    "amplitude_mm": 4.0, "max_iter": 25, "alpha": 300, "beta": 0.0001,
    "precision": "single", "correction": "jac", "threads": 2,
    "timeout_seconds_per_case": 90, "rss_limit_bytes": 8 * 1024**3,
    "evaluation_support_fraction_of_truth_peak": 0.05,
    "predeclared_checks": {
        "zero_field_rmse_mm_max": 0.05,
        "zero_image_relative_rmse_max": 0.002,
        "distorted_image_rmse_after_over_before_max": 0.75,
        "field_rmse_mm_max": 1.0,
        "minimum_estimated_jacobian": 0.0,
    },
}


class GeometryError(ValueError):
    """The solver cannot express this pair's physical geometry."""


def rotation(axis: int, degrees: float) -> np.ndarray:
    angle = np.deg2rad(degrees)
    result = np.eye(3)
    first, second = ((1, 2), (2, 0), (0, 1))[axis]
    result[first, first] = result[second, second] = np.cos(angle)
    result[first, second] = -np.sin(angle)
    result[second, first] = np.sin(angle)
    return result


def make_affine(shape=CONFIG["shape"], relative_degrees=0.0) -> np.ndarray:
    """Oblique, left-handed 2.5 mm grid, rotated about its physical center."""
    basis = rotation(2, 17) @ rotation(1, -10) @ rotation(0, 8) @ np.diag([-1., 1., 1.])
    basis = basis @ rotation(2, relative_degrees)
    affine = np.eye(4)
    affine[:3, :3] = basis * CONFIG["spacing_mm"]
    affine[:3, 3] = np.array([12., -18., 32.]) - affine[:3, :3] @ ((np.asarray(shape) - 1) / 2)
    return affine


def grid_world(shape, affine) -> np.ndarray:
    ijk = np.moveaxis(np.indices(shape, dtype=float), 0, -1)
    return ijk @ affine[:3, :3].T + affine[:3, 3]


def unit_basis(affine) -> np.ndarray:
    matrix = np.asarray(affine)[:3, :3]
    return matrix / np.linalg.norm(matrix, axis=0)


def transport_direction(vector, from_affine, to_affine) -> np.ndarray:
    """Transport a physical direction, using unit axes rather than voxel sizes."""
    vector = np.asarray(vector, dtype=float)
    if vector.shape != (3,) or not np.isclose(np.linalg.norm(vector), 1):
        raise GeometryError("Phase-encoding direction must be a unit vector")
    return np.linalg.solve(unit_basis(to_affine), unit_basis(from_affine) @ vector)


def require_supported_geometry(shape1, affine1, pe1, shape2, affine2, pe2, axis=1):
    """Fail before invoking an axis-only solver; regridding does not rotate PE."""
    if tuple(shape1) != tuple(shape2) or not np.allclose(affine1, affine2, atol=1e-5, rtol=0):
        raise GeometryError("Different physical grids: second-image affine must be respected")
    basis = unit_basis(affine1)
    if not np.allclose(basis.T @ basis, np.eye(3), atol=1e-5, rtol=0):
        raise GeometryError("Sheared grids are unsupported")
    expected = np.eye(3)[axis]
    if not np.allclose(pe1, expected, atol=1e-6, rtol=0) or not np.allclose(pe2, -expected, atol=1e-6, rtol=0):
        raise GeometryError("Unsupported PE vectors: model requires exactly opposite single-axis directions")


def regrid(data, source_affine, target_shape, target_affine):
    mapping = np.linalg.solve(source_affine, target_affine)
    return affine_transform(data, mapping[:3, :3], mapping[:3, 3], output_shape=tuple(target_shape),
                            order=3, mode="constant", cval=0., prefilter=True)


def analytic_density(world, reference_basis) -> np.ndarray:
    """Compact smooth anatomical-like object, not a simulated clinical brain."""
    q = (world - np.array([12., -18., 32.])) @ reference_basis
    envelope = np.maximum(1 - np.sum((q / [100., 100., 70.])**2, axis=-1), 0)**3
    density = np.ones(q.shape[:-1]) * 0.3
    for center, width, height in [([-24., -12., 2.], [18., 26., 22.], 0.9),
                                  ([26., 18., 12.], [25., 15., 18.], 0.7),
                                  ([8., -34., -20.], [14., 12., 14.], 0.5)]:
        density += height * np.exp(-0.5 * np.sum(((q - center) / width)**2, axis=-1))
    return envelope * density


def analytic_field(world, reference_basis, amplitude):
    q = (world - np.array([12., -18., 32.])) @ reference_basis
    f = np.sin(q[..., 1] / 35) + 0.35 * np.cos(q[..., 0] / 30)
    envelope = np.exp(-np.sum(q*q, axis=-1) / (2 * 75**2))
    df = np.stack([-0.35 * np.sin(q[..., 0] / 30) / 30,
                   np.cos(q[..., 1] / 35) / 35, np.zeros_like(f)], axis=-1)
    grad_q = amplitude * envelope[..., None] * (df - f[..., None] * q / 75**2)
    return amplitude * f * envelope, grad_q @ reference_basis.T


def distorted_image(shape, affine, reference_basis, pe_world, amplitude):
    """Solve y=x+d(x)p, then apply exact density/Jacobian forward modulation."""
    observed_world = grid_world(shape, affine)
    displacement = np.zeros(tuple(shape))
    for _ in range(12):
        source_world = observed_world - displacement[..., None] * pe_world
        field, gradient = analytic_field(source_world, reference_basis, amplitude)
        jacobian = 1 + gradient @ pe_world
        displacement -= (displacement - field) / jacobian
    source_world = observed_world - displacement[..., None] * pe_world
    field, gradient = analytic_field(source_world, reference_basis, amplitude)
    jacobian = 1 + gradient @ pe_world
    if np.min(jacobian) <= 0:
        raise ValueError("Analytic forward map folds")
    inverse_residual = float(np.max(np.abs(displacement - field)))
    return (analytic_density(source_world, reference_basis) / jacobian).astype(np.float32), {
        "min_forward_jacobian": float(jacobian.min()), "max_inverse_residual_mm": inverse_residual,
    }


def save_image(path, data, affine):
    image = nib.Nifti1Image(np.asarray(data, dtype=np.float32), affine)
    image.header.set_xyzt_units("mm")
    image.set_qform(affine, code=1)
    image.set_sform(affine, code=1)
    nib.save(image, path)


def adapt_outputs(prefix: Path, reference: Path, destination: Path, intensity_scale: float,
                  intensity_floor: float = 0.):
    """Restore image and staggered field sampling outside the GPL executable.

    The field has one more sample along PE. Averaging neighbors produces the
    cell-centered displacement; its native nodes start half a voxel before
    the first image center, not at that center. These are displacements in mm,
    not a field in Hz, and no readout-time assumption is needed for the phantom.
    """
    ref = nib.load(reference)
    destination.mkdir(parents=True, exist_ok=True)
    images = []
    raw_identity = []
    for number in (1, 2):
        raw = nib.load(str(prefix) + f"-im{number}Corrected.nii.gz")
        if raw.shape != ref.shape:
            raise GeometryError("Corrected image shape changed")
        array = raw.get_fdata() / intensity_scale
        if not np.isfinite(array).all():
            raise ValueError("Nonfinite corrected output")
        raw_identity.append(bool(np.array_equal(raw.affine, np.eye(4))))
        images.append(array)
    raw_field = nib.load(str(prefix) + "-EstFieldMap.nii.gz")
    expected_shape = list(ref.shape)
    expected_shape[1] += 1
    if raw_field.shape != tuple(expected_shape):
        raise GeometryError("Unexpected staggered field shape")
    nodes = raw_field.get_fdata()
    if not np.isfinite(nodes).all():
        raise ValueError("Nonfinite displacement output")
    node_affine = ref.affine.copy()
    node_affine[:3, 3] -= 0.5 * ref.affine[:3, 1]
    centered = (nodes[:, :-1] + nodes[:, 1:]) / 2
    save_image(destination / "displacement-nodes-mm.nii.gz", nodes, node_affine)
    save_image(destination / "displacement-centers-mm.nii.gz", centered, ref.affine)
    derivative = np.diff(nodes, axis=1) / np.linalg.norm(ref.affine[:3, 1])
    # PyHySCO normalizes (I-floor)*scale before density modulation. Undoing
    # this requires floor*Jacobian, not merely adding the scalar floor.
    for index, sign in enumerate((1, -1)):
        images[index] += intensity_floor * (1 + sign * derivative)
        save_image(destination / f"corrected-{index + 1}.nii.gz", images[index], ref.affine)
    return images, centered, derivative, {"raw_corrected_affines_are_identity": raw_identity,
        "raw_field_affine_is_identity": bool(np.array_equal(raw_field.affine, np.eye(4))),
        "image_affine": ref.affine.tolist(), "node_affine": node_affine.tolist(),
        "field_semantics": "signed displacement in millimeters along reference voxel +j",
        "image_intensity_rescale": float(1 / intensity_scale),
        "input_intensity_floor_restored_with_jacobian": float(intensity_floor)}


def verify_runtime(runtime: Path, wheel: Path):
    if hashlib.sha256(wheel.read_bytes()).hexdigest() != WHEEL_SHA256:
        raise ValueError("Unreviewed wheel checksum")
    checked = 0
    with zipfile.ZipFile(wheel) as archive:
        for name in archive.namelist():
            if name.endswith(".py"):
                if (runtime / name).read_bytes() != archive.read(name):
                    raise ValueError(f"Installed optional source changed: {name}")
                checked += 1
    return checked


def child_env(runtime):
    env = os.environ.copy()
    env.update({"PYTHONPATH": str(runtime.resolve()), "OMP_NUM_THREADS": "2",
                "OPENBLAS_NUM_THREADS": "2", "MKL_NUM_THREADS": "2", "MPLBACKEND": "Agg",
                "CUDA_VISIBLE_DEVICES": ""})
    return env


def bounded_run(command, log_path, env, timeout, rss_limit):
    """Monitor the sole Python child; no shell or unmonitored worker pool."""
    started = time.monotonic()
    peak_rss = 0
    stopped = None
    with log_path.open("w") as log:
        process = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT, env=env,
                                   start_new_session=True)
        while process.poll() is None:
            reading = subprocess.run(["ps", "-o", "rss=", "-p", str(process.pid)],
                                     capture_output=True, text=True, check=False)
            if reading.returncode == 0 and reading.stdout.strip():
                peak_rss = max(peak_rss, int(reading.stdout.strip()) * 1024)
            if peak_rss > rss_limit or time.monotonic() - started > timeout:
                stopped = "rss_limit" if peak_rss > rss_limit else "timeout"
                os.killpg(process.pid, signal.SIGKILL)
                break
            time.sleep(0.1)
        process.wait()
    return {"returncode": process.returncode, "wall_seconds": time.monotonic() - started,
            "sampled_peak_rss_bytes": peak_rss, "stopped": stopped,
            "rss_measurement": "ps RSS every ~0.1 seconds; may miss transient peaks"}


def relative_rmse(image, truth, support):
    return float(np.linalg.norm((image - truth)[support]) / np.linalg.norm(truth[support]))


def geometry_probe(runtime, output, first_path, second_path, altered_path):
    """Behavioral check: changing only second affine leaves loader data identical."""
    code = (
        "import json,sys,torch; from EPI_MRI.utils import load_data; "
        "a=load_data(sys.argv[1],im2=sys.argv[2],phase_encoding_direction=2); "
        "b=load_data(sys.argv[1],im2=sys.argv[3],phase_encoding_direction=2); "
        "print(json.dumps({'second_affine_ignored':all(torch.equal(x,y) if torch.is_tensor(x) "
        "else x==y for x,y in zip(a,b))}))"
    )
    completed = subprocess.run([sys.executable, "-c", code, str(first_path), str(second_path), str(altered_path)],
                               env=child_env(runtime), capture_output=True, text=True, timeout=30)
    (output / "loader-probe.log").write_text(completed.stdout + completed.stderr)
    if completed.returncode:
        raise RuntimeError("External geometry probe failed; see loader-probe.log")
    return json.loads(completed.stdout.strip().splitlines()[-1])


def run_experiment(runtime, wheel, output):
    output.mkdir(parents=True, exist_ok=False)
    config = dict(CONFIG)
    config.update({"wheel_sha256": WHEEL_SHA256,
                   "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest()})
    # Freeze before constructing truth or running the solver. Never tune on results.
    frozen = json.dumps(config, indent=2, sort_keys=True) + "\n"
    (output / "frozen-config.json").write_text(frozen)
    checked = verify_runtime(runtime, wheel)
    shape = CONFIG["shape"]
    affine = make_affine(shape)
    affine2 = make_affine(shape, CONFIG["relative_rotation_deg"])
    basis = unit_basis(affine)
    world = grid_world(shape, affine)
    truth = analytic_density(world, basis)
    support = truth > CONFIG["evaluation_support_fraction_of_truth_peak"] * truth.max()
    truth_dir = output / "evaluation-only"
    truth_dir.mkdir()
    save_image(truth_dir / "undistorted.nii.gz", truth, affine)
    truth_field, _ = analytic_field(world, basis, CONFIG["amplitude_mm"])
    save_image(truth_dir / "true-displacement-mm.nii.gz", truth_field, affine)
    report = {"config": config, "config_sha256": hashlib.sha256(frozen.encode()).hexdigest(),
              "runtime_python_files_verified": checked, "cases": {}, "patient_processing": False,
              "truth_used_by_optimizer": False, "complete_motion_eddy_correction": False,
              "status": "running"}
    for name, amplitude, rotated in [("zero", 0., False),
                                      ("known_distortion", CONFIG["amplitude_mm"], False),
                                      ("rotated_negative_control", CONFIG["amplitude_mm"], True)]:
        case_dir = output / name
        case_dir.mkdir()
        ap, ap_sim = distorted_image(shape, affine, basis, basis[:, 1], amplitude)
        native_affine = affine2 if rotated else affine
        pa, pa_sim = distorted_image(shape, native_affine, basis, -unit_basis(native_affine)[:, 1], amplitude)
        save_image(case_dir / "ap.nii.gz", ap, affine)
        save_image(case_dir / "pa-native.nii.gz", pa, native_affine)
        pa_regridded = regrid(pa, native_affine, shape, affine) if rotated else pa
        save_image(case_dir / "pa-on-ap-grid.nii.gz", pa_regridded, affine)
        pe2 = transport_direction([0., -1., 0.], native_affine, affine)
        gates = {}
        for stage, aff, pe in [("native", native_affine, [0., -1., 0.]),
                               ("after_regrid", affine, pe2)]:
            try:
                require_supported_geometry(shape, affine, [0., 1., 0.], shape, aff, pe)
                gates[stage] = "accepted"
            except GeometryError as error:
                gates[stage] = str(error)
        if not rotated:
            require_supported_geometry(shape, affine, [0., 1., 0.], shape, affine, pe2)
        # Only a named analytic negative control may bypass this experimental
        # model gate; there is deliberately no API for patient images.
        prefix = case_dir / "raw"
        command = [sys.executable, str(runtime / "scripts/pyhysco.py"), str(case_dir / "ap.nii.gz"),
                   str(case_dir / "pa-on-ap-grid.nii.gz"), "2", "--precision", "single",
                   "--max_iter", str(CONFIG["max_iter"]), "--alpha", str(CONFIG["alpha"]),
                   "--beta", str(CONFIG["beta"]), "--correction", "jac", "--output_dir", str(prefix)]
        result = {"geometry_gates": gates, "pa_pe_in_ap_basis": pe2.tolist(),
                  "simulation": [ap_sim, pa_sim], "unsafe_model_negative_control": rotated,
                  "eligible_as_correction_evidence": False, "command": command}
        result["resources"] = bounded_run(command, case_dir / "solver.log", child_env(runtime),
                                          CONFIG["timeout_seconds_per_case"], CONFIG["rss_limit_bytes"])
        if result["resources"]["returncode"] != 0:
            result["status"] = "solver_failed"
            report["cases"][name] = result
            continue
        # Cubic PA regridding can produce tiny negative boundary overshoots.
        # Preserve them and undo the tool's normalization exactly outside it.
        minimum = float(min(ap.min(), pa_regridded.min()))
        scale = 256 / (float(max(ap.max(), pa_regridded.max())) - minimum)
        images, estimated, derivative, headers = adapt_outputs(prefix, case_dir / "ap.nii.gz",
                                                               case_dir / "adapted", scale, minimum)
        solver_log = (case_dir / "solver.log").read_text()
        warnings = [line for line in solver_log.splitlines()
                    if "line search failed" in line or "tensor(nan)" in line]
        before = np.mean([relative_rmse(ap, truth, support), relative_rmse(pa_regridded, truth, support)])
        after = np.mean([relative_rmse(image, truth, support) for image in images])
        field_truth = truth_field if amplitude else np.zeros_like(truth_field)
        field_rmse = float(np.sqrt(np.mean((estimated[support] - field_truth[support])**2)))
        result.update({"status": "evaluated", "headers": headers, "solver_warnings": warnings,
            "image_relative_rmse_before_mean": float(before), "image_relative_rmse_after_mean": float(after),
            "image_error_ratio": float(after / before) if before > 1e-7 else None,
            "field_rmse_mm": field_rmse, "zero_field_baseline_rmse_mm": float(np.sqrt(np.mean(field_truth[support]**2))),
            "opposite_sign_field_rmse_mm": float(np.sqrt(np.mean((estimated[support] + field_truth[support])**2))),
            "minimum_estimated_jacobian": float(min((1 + derivative).min(), (1 - derivative).min())),
            "physical_landmark_roundtrip_error_mm": float(np.max(np.abs(
                nib.load(case_dir / "adapted/corrected-1.nii.gz").affine - nib.load(case_dir / "ap.nii.gz").affine))),
            "maximum_unmodelled_pa_displacement_mm": float(np.max(np.abs(field_truth)) * np.linalg.norm(pe2 - [0., -1., 0.])),
        })
        bounds = CONFIG["predeclared_checks"]
        if amplitude == 0:
            passed = after < bounds["zero_image_relative_rmse_max"] and field_rmse < bounds["zero_field_rmse_mm_max"]
        else:
            passed = after / before < bounds["distorted_image_rmse_after_over_before_max"] and field_rmse < bounds["field_rmse_mm_max"]
        result["numerical_checks_pass"] = bool(passed and result["minimum_estimated_jacobian"] > 0)
        result["eligible_as_correction_evidence"] = bool(result["numerical_checks_pass"] and not rotated and not warnings)
        report["cases"][name] = result
        if rotated:
            altered = case_dir / "pa-same-array-wrong-affine.nii.gz"
            save_image(altered, pa, affine)
            report["loader_probe"] = geometry_probe(runtime, output, case_dir / "ap.nii.gz", case_dir / "pa-native.nii.gz", altered)
        (output / "results.json").write_text(json.dumps(report, indent=2) + "\n")
    report["status"] = "phantom_only_completed"
    report["patient_geometry_supported"] = False
    report["patient_correction_authorized_by_result"] = False
    (output / "results.json").write_text(json.dumps(report, indent=2) + "\n")
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runtime", type=Path, default=ROOT / "data/optional-runtimes/pyhysco-0.0.4")
    parser.add_argument("--wheel", type=Path, required=True, help="Reviewed 0.0.4 wheel; SHA256 is enforced")
    parser.add_argument("--output", type=Path, default=ROOT / "outputs/pyhysco_phantom/v1")
    args = parser.parse_args()
    report = run_experiment(args.runtime.resolve(), args.wheel.resolve(), args.output.resolve())
    print(json.dumps({"output": str(args.output), "status": report["status"],
                      "cases": {name: value.get("numerical_checks_pass") for name, value in report["cases"].items()},
                      "patient_geometry_supported": False}, indent=2))


if __name__ == "__main__":
    main()
