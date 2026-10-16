#!/usr/bin/env python3
"""One hash-allowlisted IXI265 extraction/rigid-registration display candidate.

No downloads, cohort mutation, vessel inference, or anatomical acceptance.
The source vessel reference is confined to DisplaySource attachments.
"""
from __future__ import annotations

import argparse
from dataclasses import replace
from datetime import datetime, timezone
import gzip
import hashlib
import json
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import time

import nibabel as nib
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
OLD = ROOT.with_name("RessectionLab")
CASE = "IXI265-Guys-0845"
SELECTED = OLD / "build/ixi265-paired-header-qc-v1/selected"
ALLOWLIST = {
    "T1": ("acquired/IXI265-T1.nii.gz", "317faf612b8e8a4e319fe3ec67404561b928ea9090fc23d1da20ce0900257950"),
    "MRA": ("acquired/IXI265-MRA.nii.gz", "a87b381b882242b7d61a1930a8332d44200fdb3c56c68fc09c24a057eea86dc4"),
    "reference": ("derived_reference/IXI265-derived_vessel_label.nii.gz", "9da13c332156c71a5200dbf88368bb87478914fe73034a42ca4eabf1dd5cdbfe"),
}
COHORT_SHA = "b27f28c21089e61f6c4899f172f608cd0f81391588777443939196b1700cb127"


def sha(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def checked_source(path, expected):
    """Hash before decoding; reject extra/truncated payload and invalid voxel data."""
    path = Path(path)
    if sha(path) != expected:
        raise ValueError("Allowlisted source hash mismatch")
    image = nib.load(path)
    expanded = 0
    with gzip.open(path, "rb") as stream:
        while block := stream.read(1024 * 1024):
            expanded += len(block)
            if expanded > 256 * 1024**2:
                raise ValueError("Selected image exceeds decoding budget")
    required = int(image.dataobj.offset + np.prod(image.shape) * image.get_data_dtype().itemsize)
    if expanded != required or len(image.shape) != 3:
        raise ValueError("Unexpected full NIfTI payload length or dimensionality")
    array = image.get_fdata(dtype=np.float32)
    if not np.isfinite(array).all():
        raise ValueError("Nonfinite selected source voxels")
    return image, array, expanded


def save_array(array, reference, path, *, binary=False):
    header = reference.header.copy()
    header.set_data_dtype(np.uint8 if binary else np.float32)
    image = nib.Nifti1Image(array.astype(np.uint8 if binary else np.float32), reference.affine, header)
    image.set_qform(reference.get_qform(), int(reference.header["qform_code"]))
    image.set_sform(reference.get_sform(), int(reference.header["sform_code"]))
    nib.save(image, path)


def interior_coverage(shape):
    """Conservative geometric sampling domain; never an annotation completeness mask."""
    mask = np.zeros(shape, np.uint8)
    mask[1:-1, 1:-1, 1:-1] = 1
    return mask


def nonempty_binary(values):
    # Avoid NumPy 2.0.1's large-array float quicksort: np.unique on this
    # 26-million-voxel binary reference segfaults in the installed arm64 build.
    return bool(np.all((values == 0) | (values == 1)) and np.any(values == 1))


def paths(output):
    return {name: output / "originals" / Path(relative).name for name, (relative, _) in ALLOWLIST.items()}


def preflight(output):
    (output / "originals").mkdir(parents=True, exist_ok=True)
    cohort = ROOT / "manifests/experiments/ixi-component-person-cohort-v1.json"
    if sha(cohort) != COHORT_SHA:
        raise ValueError("Frozen cohort changed")
    records = {}
    for name, (relative, digest) in ALLOWLIST.items():
        source = SELECTED / relative
        image, values, expanded = checked_source(source, digest)
        if name == "reference" and not nonempty_binary(values):
            raise ValueError("Reference is not a nonempty binary mask")
        target = paths(output)[name]
        if target.exists() and sha(target) != digest:
            raise ValueError("Archived selected original changed")
        if not target.exists():
            shutil.copyfile(source, target)
        records[name] = {"path": str(target), "original_path": str(source), "sha256": digest,
                         "bytes": target.stat().st_size, "expanded_bytes": expanded,
                         "gzip_crc_and_full_length_verified": True, "all_voxels_finite": True,
                         "shape": list(image.shape), "affine_ras_mm": image.affine.tolist(),
                         "qform": image.get_qform().tolist(), "sform": image.get_sform().tolist(),
                         "axis_codes": list(nib.aff2axcodes(image.affine)),
                         "range": [float(values.min()), float(values.max())]}
        del values, image
    write_json(output / "source-receipt.json", {
        "schema": "ixi265-source-intake-v1", "case_id": CASE, "sources": records,
        "cohort_sha256": COHORT_SHA, "frozen_role": "TRAIN", "training_admitted": False,
        "eligibility": "publisher-described healthy participants; individual pathology review not performed",
        "source_url": "https://brain-development.org/ixi-dataset/", "raw_license": "CC-BY-SA-3.0",
        "annotation_url": "https://zenodo.org/records/17393202", "annotation_license": "CC-BY-4.0",
        "annotation_lineage": "Source model-assisted (SynthStrip/Frangi) and manually refined vessel reference; not fresh inference",
        "acquisition_datetime": None, "same_acquisition_time_verified": False,
        "annotation_completeness": "unknown; unlabelled voxels are not certified vessel-free",
        "policy_use": "source reference display only; excluded from primary compartments/evidence"})


def extract(output):
    from resectionlab.brain_extraction import ASSETS, mps_runner_source
    from resectionlab.poc_models import MODEL_CONFIG
    destination = output / "synthstrip"
    destination.mkdir(exist_ok=True)
    source = paths(output)["T1"]
    if sha(source) != ALLOWLIST["T1"][1]:
        raise ValueError("T1 changed before extraction")
    assets = ROOT / "data/models/synthstrip-v1"
    for name in ("mri_synthstrip", "synthstrip.1.pt", "LICENSE_FreeSurfer.txt"):
        if sha(assets / name) != ASSETS[name][1]:
            raise ValueError("Existing SynthStrip asset changed")
    # Reuse the exact 6 GiB device adapter budget already used by poc_models.
    # Source, float32 arithmetic, architecture and native-grid restoration are unchanged.
    runner = destination / "synthstrip_mps_existing_6g.py"
    runner.write_text(mps_runner_source((assets/"mri_synthstrip").read_text(), 6*1024**3))
    command = [sys.executable, str(runner), "-i", str(source), "-m", str(destination / "brain_mask.nii.gz"),
               "--model", str(assets / "synthstrip.1.pt"), "--threads", "2", "--gpu"]
    started = time.monotonic()
    subprocess.run(command, check=True)
    reference = nib.load(source)
    mask_path = destination / "brain_mask.nii.gz"
    mask = nib.load(mask_path)
    values = mask.get_fdata(dtype=np.float32)
    if (mask.shape != reference.shape or not np.allclose(mask.affine, reference.affine, rtol=0, atol=1e-5)
            or not nonempty_binary(values)):
        raise ValueError("Invalid SynthStrip native geometry/output")
    identity = {"input_sha256": sha(source), "weights_sha256": ASSETS["synthstrip.1.pt"][1],
                "runner_sha256": sha(runner), "device": "mps", "threads": 2,
                "modified_device_adapter": "existing poc_models 6GiB MPS adapter", "metal_allocator_cap_bytes": 6*1024**3}
    write_json(destination / "inference.json", {
        "schema": "resectionlab-poc-inference-v1", "status": "succeeded", "case": CASE,
        "model_id": "synthstrip-main", "model": MODEL_CONFIG["synthstrip-main"],
        "weights": [{"path": str((assets / "synthstrip.1.pt").resolve()), "sha256": ASSETS["synthstrip.1.pt"][1],
                     "bytes": (assets / "synthstrip.1.pt").stat().st_size}],
        "input": {"path": str(source), "sha256": sha(source), "modality": "T1", "shape": list(reference.shape),
                  "affine_ras_mm": reference.affine.tolist()}, "device": "mps", "identity": identity,
        "cache_key": hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()[:20],
        "elapsed_seconds": time.monotonic() - started, "command": command,
        "outputs": [{"path": str(mask_path), "sha256": sha(mask_path), "name": mask_path.name,
                     "foreground_voxels": int(values.sum()), "shape": list(mask.shape), "affine_ras_mm": mask.affine.tolist()}],
        "review_status": "estimated_unreviewed", "clinical_validation": "none", "cortical_access_permitted": False})


def normalized(array, support=None):
    values = array[(array > 0) if support is None else support]
    low, high = np.percentile(values, [1, 99])
    if not high > low:
        raise ValueError("Insufficient image contrast")
    return np.clip((array-low)/(high-low), 0, 1).astype(np.float32), [float(low), float(high)]


def coarse_image(image, spacing=1.5):
    import SimpleITK as sitk
    size = np.ceil((np.array(image.GetSize())-1)*np.array(image.GetSpacing())/spacing).astype(int)+1
    return sitk.Resample(image, [int(n) for n in size], sitk.Transform(3, sitk.sitkIdentity),
                         sitk.sitkLinear, image.GetOrigin(), [spacing]*3, image.GetDirection(), 0., sitk.sitkFloat32)


def resample(array, source_affine, target_array, target_affine, transform, *, label=False):
    import SimpleITK as sitk
    from resectionlab.preprocessing import to_sitk
    reference = to_sitk(np.zeros(target_array.shape, np.float32), target_affine)
    result = sitk.Resample(to_sitk(array, source_affine), reference, transform,
                           sitk.sitkNearestNeighbor if label else sitk.sitkLinear, 0., sitk.sitkFloat32)
    return sitk.GetArrayFromImage(result).transpose(2, 1, 0)


def rigid_candidate(fixed_array, fixed_affine, moving_array, moving_affine, brain, *, timeout=180., iterations=80):
    """Return fixed-RAS→moving-RAS sampling transform, without anatomical acceptance."""
    import SimpleITK as sitk
    from resectionlab.preprocessing import RAS_TO_LPS, to_sitk, sitk_transform_to_ras, validate_rigid, histogram_nmi
    sitk.ProcessObject.SetGlobalDefaultNumberOfThreads(2)
    fixed_norm, fixed_window = normalized(fixed_array, brain)
    moving_norm, moving_window = normalized(moving_array)
    full_fixed = to_sitk(fixed_norm, fixed_affine)
    full_moving = to_sitk(moving_norm, moving_affine)
    fixed, moving = coarse_image(full_fixed), coarse_image(full_moving)
    identity = sitk.Transform(3, sitk.sitkIdentity)
    coverage_image = to_sitk(interior_coverage(moving_array.shape), moving_affine)
    coverage = sitk.Resample(coverage_image, fixed, identity, sitk.sitkNearestNeighbor, 0., sitk.sitkUInt8)
    brain_image = sitk.Resample(to_sitk(brain, fixed_affine), fixed, identity, sitk.sitkNearestNeighbor, 0., sitk.sitkUInt8)
    mask = sitk.BinaryErode(coverage & brain_image, [2]*3)
    mask_values = sitk.GetArrayFromImage(mask)
    if mask_values.sum() < 1000:
        raise ValueError("Insufficient partial-field overlap for rigid registration")
    center_index = np.argwhere(mask_values).mean(axis=0)[::-1]
    center_lps = fixed.TransformContinuousIndexToPhysicalPoint(tuple(center_index))
    initial = sitk.Euler3DTransform()
    initial.SetCenter(center_lps)
    method = sitk.ImageRegistrationMethod()
    method.SetNumberOfThreads(2)
    method.SetMetricAsMattesMutualInformation(32)
    method.SetMetricSamplingStrategy(method.RANDOM)
    method.SetMetricSamplingPercentage(.20, 41)
    method.SetMetricFixedMask(mask)
    method.SetInterpolator(sitk.sitkLinear)
    method.SetOptimizerAsRegularStepGradientDescent(learningRate=1., minStep=.002,
        numberOfIterations=iterations, relaxationFactor=.5, gradientMagnitudeTolerance=1e-5)
    method.SetOptimizerScalesFromPhysicalShift()
    method.SetShrinkFactorsPerLevel([4, 2, 1])
    method.SetSmoothingSigmasPerLevel([2., 1., 0.])
    method.SmoothingSigmasAreSpecifiedInPhysicalUnitsOn()
    method.SetInitialTransform(initial, inPlace=False)
    started = time.monotonic()
    history = []
    timed_out = False
    def observe():
        nonlocal timed_out
        history.append({"level": int(method.GetCurrentLevel()), "iteration": int(method.GetOptimizerIteration()),
                        "sampled_negative_mattes_mi": float(method.GetMetricValue())})
        if time.monotonic()-started >= timeout:
            timed_out = True
            method.StopRegistration()
    method.AddCommand(sitk.sitkIterationEvent, observe)
    fitted = method.Execute(fixed, moving)
    elapsed = time.monotonic()-started
    if timed_out or elapsed > timeout:
        raise TimeoutError("Registration exceeded its 180-second limit")
    matrix = sitk_transform_to_ras(fitted)
    validate_rigid(matrix)
    before_image = sitk.Resample(moving, fixed, initial, sitk.sitkLinear, 0.)
    after_image = sitk.Resample(moving, fixed, fitted, sitk.sitkLinear, 0.)
    after_coverage = sitk.Resample(coverage_image, fixed, fitted, sitk.sitkNearestNeighbor, 0., sitk.sitkUInt8)
    common = mask_values.astype(bool) & sitk.GetArrayFromImage(after_coverage).astype(bool)
    fixed_values = sitk.GetArrayFromImage(fixed)
    before = sitk.GetArrayFromImage(before_image)
    after = sitk.GetArrayFromImage(after_image)
    nmi_before, nmi_after = [histogram_nmi(fixed_values, values, common) for values in (before, after)]
    center_ras = RAS_TO_LPS[:3, :3] @ center_lps
    shift = float(np.linalg.norm(nib.affines.apply_affine(matrix, center_ras)-center_ras))
    rotation = float(np.rad2deg(np.arccos(np.clip((np.trace(matrix[:3, :3])-1)/2, -1, 1))))
    flags = []
    if nmi_after < nmi_before: flags.append("matched_NMI_decreased")
    if shift > 20 or rotation > 15: flags.append("large_pose_change")
    if common.sum()/mask_values.sum() < .9: flags.append("overlap_loss")
    return fitted, {
        "method": "SimpleITK Euler3D rigid, identity physical pose, Mattes MI32", "simpleitk_version": sitk.Version_VersionString(),
        "fixed": "T1", "moving": "MRA", "T1_RAS_mm_to_MRA_RAS_mm": matrix.tolist(),
        "MRA_RAS_mm_to_T1_RAS_mm": np.linalg.inv(matrix).tolist(),
        "sampling_direction": "T1 output physical point -> MRA input physical point",
        "configuration": {"threads": 2, "seed": 41, "sampling_fraction": .2, "iterations_per_level": iterations,
                          "coarse_spacing_mm": 1.5, "shrink_levels": [4, 2, 1], "timeout_seconds": timeout},
        "elapsed_registration_seconds": elapsed, "optimizer_history": history,
        "optimizer_stop_reason": method.GetOptimizerStopConditionDescription(),
        "normalization_windows": {"T1": fixed_window, "MRA": moving_window},
        "matched_objective": {"metric": "fixed-bin histogram NMI32, same paired voxels", "before": nmi_before,
                              "after": nmi_after, "voxels": int(common.sum()), "overlap_retained_fraction": float(common.sum()/mask_values.sum())},
        "center_displacement_mm": shift, "rotation_degrees": rotation, "flags": flags,
        "registration_reviewed": False, "anatomical_registration_accepted": False,
        "classification": "software_valid_candidate_anatomical_review_pending",
        "metric_improvement_is_anatomical_acceptance": False}


def register(output):
    import SimpleITK as sitk
    from scipy import ndimage
    source = paths(output)
    t1, mra, annotation = [nib.load(source[key]) for key in ("T1", "MRA", "reference")]
    t1_data, mra_data, label = [image.get_fdata(dtype=np.float32) for image in (t1, mra, annotation)]
    brain = nib.load(output / "synthstrip/brain_mask.nii.gz").get_fdata(dtype=np.float32) > 0
    fitted, report = rigid_candidate(t1_data, t1.affine, mra_data, mra.affine, brain)
    destination = output / "registration"
    destination.mkdir(exist_ok=True)
    sitk.WriteTransform(fitted, str(destination / "T1_to_MRA_sampling.tfm"))
    sitk.WriteTransform(fitted.GetInverse(), str(destination / "MRA_to_T1_points.tfm"))
    identity = sitk.Transform(3, sitk.sitkIdentity)
    before = resample(mra_data, mra.affine, t1_data, t1.affine, identity)
    after = resample(mra_data, mra.affine, t1_data, t1.affine, fitted)
    native_coverage = interior_coverage(mra.shape)
    before_coverage = resample(native_coverage, mra.affine, t1_data, t1.affine, identity, label=True) > 0
    after_coverage = resample(native_coverage, mra.affine, t1_data, t1.affine, fitted, label=True) > 0
    registered_label = resample(label, annotation.affine, t1_data, t1.affine, fitted, label=True) > 0
    excluded_border_label_voxels = int((registered_label & ~after_coverage).sum())
    registered_label &= after_coverage
    outputs = [(before, t1, "MRA_on_T1_before.nii.gz", False), (after, t1, "MRA_on_T1_candidate.nii.gz", False),
        (native_coverage, mra, "MRA_native_sampling_coverage.nii.gz", True),
        (before_coverage, t1, "MRA_coverage_on_T1_before.nii.gz", True),
        (after_coverage, t1, "MRA_coverage_on_T1_candidate.nii.gz", True),
        (registered_label, t1, "source_vessels_on_T1_candidate.nii.gz", True)]
    for values, reference, name, binary in outputs:
        save_array(values, reference, destination / name, binary=binary)
    corners = np.array(np.meshgrid(*[(0, n-1) for n in t1.shape], indexing="ij")).reshape(3, -1).T
    points = nib.affines.apply_affine(t1.affine, corners)
    matrix = np.asarray(report["T1_RAS_mm_to_MRA_RAS_mm"])
    roundtrip = nib.affines.apply_affine(np.linalg.inv(matrix), nib.affines.apply_affine(matrix, points))
    boundary = brain & ~ndimage.binary_erosion(brain)
    report["software_grid_qc"] = {"maximum_point_roundtrip_error_mm": float(np.linalg.norm(roundtrip-points, axis=1).max()),
        "rotation_determinant": float(np.linalg.det(matrix[:3, :3])), "label_outside_saved_coverage_voxels": int((registered_label & ~after_coverage).sum()),
        "source_reference_voxels_excluded_by_conservative_border_coverage": excluded_border_label_voxels,
        "brain_fraction_with_MRA_sampling_before": float((brain & before_coverage).sum()/brain.sum()),
        "brain_fraction_with_MRA_sampling_after": float((brain & after_coverage).sum()/brain.sum()),
        "registered_reference_outside_brain_voxels": int((registered_label & ~brain).sum()),
        "registered_reference_total_voxels": int(registered_label.sum()),
        "reference_near_brain_boundary_voxels": int((registered_label & ndimage.binary_dilation(boundary, iterations=2)).sum()),
        "coverage_meaning": "Conservative acquired MRA FOV; excludes one native border voxel. Does not establish annotation completeness."}
    report["source_hashes"] = {key: sha(path) for key, path in source.items()}
    report["brain_mask_sha256"] = sha(output / "synthstrip/brain_mask.nii.gz")
    report["outputs"] = [{"path": str(destination/name), "sha256": sha(destination/name)} for _, _, name, _ in outputs]
    write_json(destination / "registration.json", report)
    render_qc(output, t1, t1_data, before, after, brain, before_coverage, after_coverage, registered_label)


def render_qc(output, reference, t1, before, after, brain, before_coverage, after_coverage, vessels):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    # Reorientation only for figures; exported images retain their native grid.
    def canonical(values):
        return nib.as_closest_canonical(nib.Nifti1Image(values.astype(np.float32), reference.affine)).get_fdata(dtype=np.float32)
    t1, before, after, brain, before_coverage, after_coverage, vessels = map(canonical, (t1, before, after, brain, before_coverage, after_coverage, vessels))
    t1, _ = normalized(t1, brain > 0)
    before, _ = normalized(before)
    after, _ = normalized(after)
    center = np.argwhere((brain > 0) & (after_coverage > 0)).mean(axis=0).astype(int)
    figure, axes = plt.subplots(3, 4, figsize=(15, 11), facecolor="white")
    for row, (dimension, plane) in enumerate([(2, "Axial"), (1, "Coronal"), (0, "Sagittal")]):
        index = int(center[dimension])
        sliced = lambda values: np.take(values, index, axis=dimension).T
        for column, moving in enumerate((before, after)):
            axes[row, column].imshow(sliced(t1), cmap="gray", origin="lower", vmin=0, vmax=1)
            axes[row, column].imshow(sliced(moving), cmap="magma", origin="lower", vmin=0, vmax=1, alpha=.42)
            axes[row, column].contour(sliced(before_coverage if column == 0 else after_coverage), levels=[.5], colors=["cyan"], linewidths=.6)
            axes[row, column].set_title(f"{plane}: {'header identity' if column == 0 else 'rigid candidate'}")
        grid = np.indices(sliced(t1).shape)
        checker = ((grid[0]//12 + grid[1]//12) % 2).astype(bool)
        axes[row, 2].imshow(np.where(checker, sliced(t1), sliced(after)), cmap="gray", origin="lower", vmin=0, vmax=1)
        axes[row, 2].set_title("Candidate T1/MRA checkerboard")
        axes[row, 3].imshow(sliced(t1), cmap="gray", origin="lower", vmin=0, vmax=1)
        axes[row, 3].contour(sliced(brain), levels=[.5], colors=["deepskyblue"], linewidths=.7)
        if sliced(vessels).any():
            axes[row, 3].contour(sliced(vessels), levels=[.5], colors=["red"], linewidths=.6)
        axes[row, 3].set_title("Estimated brain / registered source vessels")
        for axis in axes[row]: axis.axis("off")
    figure.suptitle("IXI265: software registration candidate, anatomical acceptance pending. Cyan = partial MRA FOV.")
    figure.tight_layout()
    figure.savefig(output / "registration/registration-qc.png", dpi=140)
    plt.close(figure)


def package(output):
    from resectionlab.core import SourceRef
    from resectionlab.imaging import load_nifti_case
    from resectionlab.patient_workspace import _brain_record
    from resectionlab.workspace_bundle import OriginalFile, load_workspace, save_workspace, view_state
    from resectionlab.workspace_imaging import DisplaySource, source_registration
    source = paths(output)
    receipt_path = output / "source-receipt.json"
    registration_path = output / "registration/registration.json"
    inference_path = output / "synthstrip/inference.json"
    brain_path = output / "synthstrip/brain_mask.nii.gz"
    registered_label = output / "registration/source_vessels_on_T1_candidate.nii.gz"
    registration = json.loads(registration_path.read_text())
    common = {"patient_id": CASE, "patient_name": CASE, "source_name": "IXI", "source_url": "https://brain-development.org/ixi-dataset/",
        "source_license": "CC-BY-SA-3.0", "source_eligibility": "publisher-described healthy; individual pathology review not performed",
        "eligibility_source_receipt": {"url": "https://brain-development.org/ixi-dataset/",
            "source_html_sha256": "f80f30853eeeeec6caf90ae4410c712fe447161808c238dea8111396020efe8c",
            "individual_scan_pathology_review": False},
        "anatomy_workbench_only": True, "acquisition_datetime": None, "cohort_role": "TRAIN", "training_admitted": False,
        "patient_group": "IXI:265", "split": {"role": "TRAIN", "patient_group": "IXI:265", "training_admitted": False},
        "cohort_sha256": COHORT_SHA, "registration_candidate": registration,
        "workspace_registration": {"kind": "estimated-unreviewed", "overlayPermitted": False},
        "segmentation_cache_policy": "saved masks only; never reinfer on open", "annotation_policy_use": "excluded",
        "missing_anatomy": ["skull", "tissue_classes", "ventricles", "fresh_vessel_inference"]}
    brain_layer = _brain_record(brain_path, inference_path, {"T1": ALLOWLIST["T1"][1]})
    primary = load_nifti_case(source["T1"], case_id=CASE,
        source_url=common["source_url"], license="CC-BY-SA-3.0", metadata={**common, "selected_modality": "T1", "anatomy_layers": {"brain": brain_layer}})
    brain_image = nib.load(brain_path)
    brain_values = brain_image.get_fdata(dtype=np.float32)
    if (brain_image.shape != primary.mri.shape or not np.allclose(brain_image.affine, primary.affine, rtol=0, atol=1e-5)
            or not nonempty_binary(brain_values)):
        raise ValueError("Brain output geometry or values changed before packaging")
    primary = replace(primary, brain_mask=brain_values > 0, source_refs=(*primary.source_refs,
        SourceRef("supplied_brain_mask", brain_path.resolve().as_uri(), sha256=sha(brain_path),
                  license=brain_layer["model"]["license"], native_frame="RAS+", provenance="estimated")))
    layer = {"name": "source_vessels", "label": "Source vessel reference (registration estimated)", "kind": "vessels",
        "provenance": "source-provided", "source_role": "display_only_public_reference", "policy_use": "excluded",
        "registration_status": "estimated_unreviewed", "coverage": "Partial MRA FOV; incomplete reference; unlabelled anatomy unknown",
        "model": None, "source_lineage": "Published SynthStrip/Frangi-assisted, manually refined reference; no new vessel inference",
        "source_license": "CC-BY-4.0", "color": "#ff597c"}
    sources = []
    for image, mask, modality in [(source["MRA"], source["reference"], "MRA"), (source["T1"], registered_label, "T1")]:
        case = load_nifti_case(image, case_id=CASE,
            source_url=common["source_url"], license="CC-BY-SA-3.0",
            metadata={**common, "selected_modality": modality, "anatomy_layers": {"source_vessels": layer}, "annotation_use": "display_only"})
        label_image = nib.load(mask)
        values = label_image.get_fdata(dtype=np.float32)
        if (label_image.shape != case.mri.shape or not np.allclose(label_image.affine, case.affine, rtol=0, atol=1e-4)
                or not nonempty_binary(values)):
            raise ValueError("Source display reference geometry or values changed")
        labels = {"source_vessels": values > 0}
        case = replace(case, compartments=labels, source_compartments=labels, source_refs=(*case.source_refs,
            SourceRef("supplied_target_annotation", mask.resolve().as_uri(), sha256=sha(mask),
                      license="CC-BY-4.0", native_frame="RAS+", provenance="observed")))
        sources.append(DisplaySource(case, modality, "source-provided"))
    if primary.compartments or primary.critical_evidence or primary.prior_proposals:
        raise ValueError("Source reference leaked into primary policy-visible anatomy")
    state = view_state(primary, sources)
    state["states"]["primary"]["viewer"] = {"layerOpacity": {"brain": .16}, "layerColor": {"brain": "#65bfff"}}
    originals = []
    archived = [(path, "annotation" if key == "reference" else "imaging") for key, path in source.items()]
    archived += [(brain_path, "model-result"), (inference_path, "metadata"), (receipt_path, "metadata"), (registration_path, "metadata"),
                 (registered_label, "annotation")]
    archived += [(path, "metadata") for path in sorted((output/"registration").glob("*.tfm"))]
    archived += [(path, "model-result") for path in sorted((output/"registration").glob("*coverage*.nii.gz"))]
    for path in [ROOT/"artifacts/ixi-t1-mra-vessel-byte-intake-v1/source-rights-and-head-proofs.json",
                 OLD/"build/ixi-vessel-source-preparation-v1/ixi-page.html",
                 OLD/"build/ixi-vessel-source-preparation-v1/zenodo-record.json"]:
        archived.append((path, "metadata"))
    for path, role in archived:
        originals.append(OriginalFile.from_path(path, role=role))
    bundle = output / (CASE + ".ressectionlab")
    identity = save_workspace(primary, bundle, sources=sources, imaging=state, episode=None, selection=None, artifacts={}, originals=originals)
    reopened, _, restored = load_workspace(bundle)
    if reopened.semantic_hash != primary.semantic_hash or restored["sessionHash"] != identity or restored["imagingState"] != state:
        raise RuntimeError("Canonical bundle changed on reopen")
    if reopened.compartments or reopened.critical_evidence or reopened.prior_proposals:
        raise ValueError("Reopened primary leaked reference labels")
    if [item.sha256 for item in originals] != [item.sha256 for item in restored["originals"]]:
        raise RuntimeError("Original archive changed")
    write_json(output / "bundle-receipt.json", {"case_id": CASE, "bundle": str(bundle), "sha256": sha(bundle),
        "case_hash": primary.semantic_hash, "session_hash": identity, "reopened_identical": True, "original_files": len(originals),
        "primary_target_compartments": list(reopened.compartments), "primary_estimated_layers": ["brain"],
        "source_reference_policy_excluded_by_display_attachment": True, "display_series": [source.manifest(primary) for source in sources],
        "fusion_permitted": [source_registration(primary, source.case)[1]["overlayPermitted"] for source in sources],
        "registration_status": "estimated_unreviewed", "anatomical_registration_accepted": False})


def bounded_stage(output, stage, timeout):
    import psutil
    command = [sys.executable, str(Path(__file__).resolve()), "--output", str(output), "--stage", stage]
    environment = dict(os.environ, PYTHONPATH=str(ROOT/"src"), PYTHONFAULTHANDLER="1", OMP_NUM_THREADS="2", OPENBLAS_NUM_THREADS="2",
        MKL_NUM_THREADS="2", ITK_GLOBAL_DEFAULT_NUMBER_OF_THREADS="2", PYTORCH_ENABLE_MPS_FALLBACK="1", MPLBACKEND="Agg")
    started = time.monotonic()
    peak = 0
    failure = None
    rss_limit = int((.6 if stage == "preflight" else 4) * 1024**3)
    minimum_available = psutil.virtual_memory().available
    attempt = 1
    while (output / (stage + f"-attempt{attempt}.log")).exists():
        attempt += 1
    stem = stage + f"-attempt{attempt}"
    with (output / (stem + ".log")).open("w") as log:
        process = subprocess.Popen(command, env=environment, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
        while process.poll() is None:
            try:
                parent = psutil.Process(process.pid)
                family = [parent, *parent.children(recursive=True)]
                rss = sum(p.memory_info().rss for p in family if p.is_running()) + psutil.Process().memory_info().rss
                peak = max(peak, rss)
            except psutil.NoSuchProcess:
                pass
            minimum_available = min(minimum_available, psutil.virtual_memory().available)
            if time.monotonic()-started > timeout or peak > rss_limit or minimum_available < .75*1024**3:
                failure = "wall_budget_exceeded" if time.monotonic()-started > timeout else "RSS_budget_exceeded" if peak > rss_limit else "system_available_memory_below_0.75GiB"
                os.killpg(process.pid, signal.SIGTERM)
                try: process.wait(timeout=3)
                except subprocess.TimeoutExpired: os.killpg(process.pid, signal.SIGKILL)
                break
            time.sleep(.2)
        process.wait()
    result = {"stage": stage, "command": command, "elapsed_seconds": time.monotonic()-started,
              "exit_code": process.returncode, "peak_total_family_plus_supervisor_rss_bytes_sampled": peak,
              "rss_limit_bytes": rss_limit,
              "minimum_system_available_memory_bytes_sampled": minimum_available,
              "failure": failure, "sampling_interval_seconds": .2, "memory_is_sampled_not_exact_peak": True}
    result["log"] = str(output / (stem + ".log"))
    write_json(output / (stem + "-runtime.json"), result)
    if failure or process.returncode:
        raise RuntimeError(f"{stage} failed; inspect {output / (stem+'.log')}")
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT/"outputs/poc-ixi265")
    parser.add_argument("--stage", choices=["preflight", "extraction", "registration", "package"])
    args = parser.parse_args()
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    if args.stage:
        {"preflight": preflight, "extraction": extract, "registration": register, "package": package}[args.stage](output)
        return
    if (output/"bundle-receipt.json").exists():
        raise ValueError("Completed candidate exists; inspect/reopen it rather than reinfer")
    runtimes = []
    for stage, budget in [("preflight", 60), ("extraction", 120), ("registration", 180), ("package", 120)]:
        runtimes.append(bounded_stage(output, stage, budget))
    write_json(output/"run-receipt.json", {"schema": "poc-ixi265-registration-run-v1", "created_at": datetime.now(timezone.utc).isoformat(),
        "script_sha256": sha(Path(__file__)), "case_id": CASE, "stages": runtimes, "downloads": False,
        "fresh_models_run": ["SynthStrip"], "source_vessel_reference_is_inference": False,
        "anatomical_registration_accepted": False, "status": "candidate_saved"})
    print(json.dumps({"status": "candidate_saved", "bundle_receipt": str(output/"bundle-receipt.json")}))


if __name__ == "__main__":
    main()
