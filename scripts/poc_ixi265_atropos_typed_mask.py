"""Prepared v3 N4 mask representation repair; no patient runner or fit release.

This worker kernel changes only the N4 mask's API representation to float32.
Original positive-domain support, intensity values, native frame, parameters,
UNKNOWN cells and raw posterior semantics remain the frozen v2 protocol.
An injected backend supports source-only transport controls. Native use requires
a separately released, source-bound supervisor; this module opens no images.
"""
from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
V2_PATH = ROOT/"scripts/poc_ixi265_atropos_positive_domain.py"
V2_SHA = "0652eca1ba81304e053f24c9a9fc2f8c89174bfe7447accfd21105263592d81d"
GENERATED_OUTCOME_SHA = "f7a0844d9569ddd8add755c167d316a0c8a7fd840aef76a9ce35ed8674a53102"
GENERATED_OBSERVATIONS_SHA = "173a3c355b28fa38a9c89a0707f2a8ecb7716e2294dbe2e4ffa189f5fcc52332"


def sha(path):
    with Path(path).open("rb") as stream: return hashlib.file_digest(stream, "sha256").hexdigest()


def previous():
    if sha(V2_PATH) != V2_SHA: raise ValueError("Frozen v2 source changed")
    spec = importlib.util.spec_from_file_location("frozen_positive_domain_v2", V2_PATH)
    module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
    return module


def method():
    return {**previous().method(), "schema": "resectionlab-ixi265-atropos-typed-mask-candidate-v3",
        "status": "kernel_prepared_patient_runner_not_released",
        "protocol": "v2_positive_domain_with_native_N4_float_mask_contract",
        "single_attempt_marker": "ixi265-n4-atropos-typed-mask-v3-attempt.json",
        "v2_source_sha256": V2_SHA, "N4_mask_numpy_dtype": "float32", "N4_mask_native_pixeltype": "float",
        "mask_value_support_or_frame_change": False, "N4_or_Atropos_parameter_change": False,
        "generated_diagnostic_outcome_sha256": GENERATED_OUTCOME_SHA,
        "generated_diagnostic_observations_sha256": GENERATED_OBSERVATIONS_SHA,
        "new_kernel_native_preflight_required_before_any_patient_release": True,
        "patient_fitting_released": False,
        "diagnostic_limit": "generated float success supports API repair; does not prove prior crash cause or anatomical accuracy"}


def require_float_mask(mask, native_mask, affine, base):
    if native_mask.pixeltype != "float":
        raise ValueError("N4 boundary requires native mask.pixeltype == 'float'")
    actual = native_mask.numpy()
    if actual.dtype != np.float32 or not np.array_equal(actual, mask.astype(np.float32)):
        raise ValueError("N4 boundary float mask must preserve exact binary values")
    frame = base.require_native_frame(mask.shape, affine, native_mask.shape,
        native_mask.origin, native_mask.spacing, native_mask.direction)
    return {"native_pixeltype": "float", "numpy_dtype": "float32", "binary_values_exact": True,
        "support_voxels": int(mask.sum()), "support_unchanged": True, "frame": frame}


def fit_arrays(values, domain, affine, ants, *, stage_callback=None):
    """Worker-only kernel; caller must admit and bind source before decoding it."""
    base = previous().reference()
    if (values.ndim != 3 or values.size > base.MAX_VOXELS or domain.shape != values.shape
            or domain.dtype != np.bool_ or not domain.any() or not np.isfinite(values).all()
            or np.any(values[domain] <= 0)):
        raise ValueError("Bounded finite source and exact positive fitting support required")
    geometry = base.ras_to_lps_geometry(affine)
    image = ants.from_numpy(np.array(values, dtype=np.float32, copy=True), **geometry)
    native_mask = ants.from_numpy(np.array(domain, dtype=np.float32, copy=True), **geometry)
    frames = {}
    def check(name, candidate):
        frames[name] = base.require_native_frame(values.shape, affine, candidate.shape,
            candidate.origin, candidate.spacing, candidate.direction)
    check("input", image)
    if image.pixeltype != "float" or not np.array_equal(image.numpy(), values.astype(np.float32)):
        raise ValueError("Input image representation changed")
    boundary = require_float_mask(domain, native_mask, affine, base)
    frames["mask"] = boundary["frame"]
    if stage_callback: stage_callback("N4_start", {"mask_boundary": boundary, "parameters": base.N4_PARAMETERS})
    corrected = ants.n4_bias_field_correction(image, mask=native_mask, **base.N4_PARAMETERS)
    check("N4", corrected)
    corrected_values = corrected.numpy()
    if not np.isfinite(corrected_values).all() or np.any(corrected_values[domain] <= 0):
        raise ValueError("Invalid N4 corrected intensities")
    if stage_callback: stage_callback("N4_complete", {"frame": frames["N4"]})
    # Pinned Atropos wrapper performs its documented unsigned-int mask clone.
    # Recheck the same float object at the boundary before handing it onward.
    require_float_mask(domain, native_mask, affine, base)
    if stage_callback: stage_callback("Atropos_start", {"parameters": base.ATROPOS_PARAMETERS})
    result = ants.atropos(a=corrected, x=native_mask, **base.ATROPOS_PARAMETERS)
    check("classes", result["segmentation"])
    labels = result["segmentation"].numpy()
    posteriors = []
    for index, probability in enumerate(result["probabilityimages"], 1):
        check("posterior_"+str(index), probability)
        posteriors.append(probability.numpy())
    audit = base.validate_classes(corrected_values, domain, labels, posteriors)
    if stage_callback: stage_callback("Atropos_complete", {"classes": audit})
    return corrected_values, labels.astype(np.uint8), posteriors, {
        "frames": frames, "classes": audit, "N4_mask_boundary": boundary,
        "mask_representation_only_change": True, "parameters_changed": False}


if __name__ == "__main__":
    print(json.dumps(method(), indent=2))
