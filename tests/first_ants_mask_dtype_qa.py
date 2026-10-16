"""Independent recount of sealed generated-only N4 mask-type diagnostics.

Reads only exact generated outputs, frozen metadata, and pinned source text.
No ANTs/model import, native execution, patient image, or bundle decoding.
"""
from __future__ import annotations
import ast
from datetime import datetime, timezone
import hashlib
import itertools
import json
from pathlib import Path
import re
import resource
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
ART = ROOT/"artifacts/poc-ants-mask-dtype-generated-v1"
BASE = ROOT/"outputs/poc-ants-mask-dtype-generated-v1"
PINS = {
    "artifacts/poc-ants-mask-dtype-generated-v1/outcome.json": "f7a0844d9569ddd8add755c167d316a0c8a7fd840aef76a9ce35ed8674a53102",
    "artifacts/poc-ants-mask-dtype-generated-v1/observations.json": "173a3c355b28fa38a9c89a0707f2a8ecb7716e2294dbe2e4ffa189f5fcc52332",
    "scripts/poc_ants_mask_dtype_diagnostic.py": "1feb74ffca22a14d226bc9b52b0b5049a607225e95bb8dee6d11df5151aa003c",
    "artifacts/poc-ants-mask-dtype-generated-v1/source-basis.json": "5e14e9cfed6b8670041b365206bd78991c8944fdce7f2704e53a1771ec525891",
    "artifacts/poc-ixi265-atropos-preparation-v1/installed-runtime-audit.json": "8dc074db35e17be3b10f05e68e19ff61cdd44a6d1e0e744bbfc25e1da9073e79",
    "artifacts/poc-ixi265-atropos-positive-domain-run-v2/outcome.json": "dce67f0acde21aa11fa99bdaa270108c8b53b5a091514182870e69e01fa78e21",
    "artifacts/ixi265-positive-domain-independent-v2/result-qa.json": "9c90b165a5da5657394ee1d26f0aa35193bc072e627b3b198d5fea1018beae72",
    "artifacts/ixi265-positive-domain-independent-v2/prospective-qa.json": "894af5ba11bb92ed6f1cc6f41ea93d5d6e6b4d49053f9cd04987dc875744a3bc",
}


def sha(path):
    with Path(path).open("rb") as stream: return hashlib.file_digest(stream, "sha256").hexdigest()


def stamp(path):
    p = Path(path); return {"sha256": sha(p), "bytes": p.stat().st_size, "mtime_ns": p.stat().st_mtime_ns}


def record(array):
    return {"shape": list(array.shape), "dtype": str(array.dtype),
            "C_order_bytes_sha256": hashlib.sha256(np.ascontiguousarray(array).tobytes()).hexdigest()}


def independently_generate():
    i, j, k = np.ogrid[:48, :48, :48]
    radius2 = (i-23.5)**2 + (j-23.5)**2 + (k-23.5)**2
    truth = np.where(radius2 <= 81, 3, np.where(radius2 <= 225, 2, np.where(radius2 <= 400, 1, 0))).astype(np.uint8)
    bias = np.exp(.12*((i-23.5)/24) + .08*((j-23.5)/24)**2 + .04*np.sin((k-23.5)/24))
    values = (np.array([0., 40., 85., 140.])[truth]*bias).astype(np.float32)
    return values, truth > 0, truth


def frame_check(frame):
    source, candidate = (np.asarray(frame[k]) for k in ("source_affine_ras_mm", "candidate_affine_ras_mm"))
    assert np.array_equal(source, np.eye(4)) and np.array_equal(candidate, np.eye(4))
    corners = np.array(list(itertools.product((-.5, 47.5), repeat=3)))
    errors = np.linalg.norm(corners@source[:3, :3].T+source[:3, 3] - (corners@candidate[:3, :3].T+candidate[:3, 3]), axis=1)
    assert errors.tolist() == frame["eight_cell_corner_errors_mm"] == [0.]*8
    assert frame["array_permutation_or_resampling"] is False


def source_contract(basis):
    cache = ROOT/"outputs/poc-first-qa/ants-mask-dtype-upstream-text"
    verified = []
    for entry in basis["primary_sources"]:
        path = cache/entry["url"].rsplit("/", 1)[-1]
        assert sha(path) == entry["sha256"] and path.stat().st_size == entry["bytes"]
        verified.append({"url": entry["url"], "sha256": entry["sha256"], "bytes": entry["bytes"]})
    cpp = (cache/"N4BiasFieldCorrection.cxx").read_text()
    io = (cache/"ReadWriteData.h").read_text()
    assert "using RealType = float;" in cpp and "using MaskImageType = itk::Image<RealType, ImageDimension>;" in cpp
    assert "ReadImage<MaskImageType>(maskImage, inputMaskFile.c_str())" in cpp
    assert "typename RImageType::Pointer Rimage = *(static_cast<typename RImageType::Pointer *>(ptr));" in io
    assert "typedef TImageType RImageType;" in io
    build = cache/"configure_ANTs.sh"
    assert sha(build) == "835b51ebd3d6aa445d865d18204165146ac0e3009fd00b4fd0089747ff118786"
    assert basis["upstream_ANTs_commit"] in build.read_text()
    installed = ROOT/".tools/poc-antspyx-0.6.3-py312-v1/runtime/lib/python3.12/site-packages"
    for entry in basis["installed_API_sources"]:
        assert sha(installed/entry["path"]) == entry["sha256"]
    bias_source = (installed/"ants/ops/bias_correction.py").read_text()
    function = next(n for n in ast.parse(bias_source).body if isinstance(n, ast.FunctionDef) and n.name == "n4_bias_field_correction")
    function_source = ast.unparse(function)
    assert "image.clone('float')" in function_source
    assert "'x': mask" in function_source
    assert "mask.clone(" not in function_source
    atropos_source = (installed/"ants/segmentation/atropos.py").read_text()
    assert "mymask = x.clone('unsigned int')" in atropos_source
    return {"primary_texts_independently_read_and_rehashed": verified,
        "pinned_build_script_sha256": sha(build), "ANTs_build_commit": basis["upstream_ANTs_commit"],
        "installed_API_text_hashes_verified": basis["installed_API_sources"],
        "native_contract": "N4 instantiates both image and mask as float. Pointer input is statically interpreted as the requested image type before an identity-type cast filter; no safe source-pixel-type adaptation is performed.",
        "python_boundary": "Pinned N4 wrapper casts intensity to float but forwards supplied mask unchanged. Atropos separately clones its mask to unsigned int.",
        "scope": "Verified pinned source contract, not a universal proof of every binary crash's cause."}


def main():
    destination = ROOT/"artifacts/ants-mask-dtype-independent-v1/independent-qa.json"
    assert not destination.exists(), "Existing evidence is immutable"
    before = {name: stamp(ROOT/name) for name in PINS}
    assert {name: row["sha256"] for name, row in before.items()} == PINS
    outcome = json.loads((ART/"outcome.json").read_text())
    observations = json.loads((ART/"observations.json").read_text())
    declaration = json.loads((ART/"declaration.json").read_text())
    basis = json.loads((ART/"source-basis.json").read_text())
    assert sha(ART/"declaration.json") == outcome["declaration_sha256"] == observations["declaration_sha256"]
    assert observations["outcome_sha256"] == PINS["artifacts/poc-ants-mask-dtype-generated-v1/outcome.json"]
    assert declaration["runner_sha256"] == PINS["scripts/poc_ants_mask_dtype_diagnostic.py"]
    assert declaration["arms"] == ["uint8", "float32"] and declaration["released_generated_two_arm_diagnostic"] is True
    frozen_runtime = json.loads((ROOT/"artifacts/poc-ixi265-atropos-preparation-v1/installed-runtime-audit.json").read_text())
    old_runtime, actual_runtime = dict(frozen_runtime), dict(declaration["runtime"])
    old_runtime["prefix"] = str(Path(old_runtime["prefix"]).resolve())
    actual_runtime["prefix"] = str(Path(actual_runtime["prefix"]).resolve())
    assert actual_runtime == old_runtime
    assert basis["native_binaries"] == frozen_runtime["installed_native_binary_hashes"]
    outputs_before = {}
    for entry in outcome["manifest"]:
        path = BASE/entry["path"]
        assert path.is_relative_to(BASE) and ".." not in Path(entry["path"]).parts
        outputs_before[entry["path"]] = stamp(path)
        assert outputs_before[entry["path"]]["sha256"] == entry["sha256"] and outputs_before[entry["path"]]["bytes"] == entry["bytes"]
        if (ART/entry["path"]).is_file(): assert sha(ART/entry["path"]) == entry["sha256"]
    source = source_contract(basis)
    values, mask, truth = independently_generate()
    generated = declaration["generated_input"]
    assert generated["values"] == record(values) and generated["binary_mask"] == record(mask) and generated["generated_class_truth"] == record(truth)
    assert int(mask.sum()) == generated["mask_voxels"] == 33552
    assert [int(np.sum(truth == n)) for n in (1, 2, 3)] == generated["class_counts"] == [19224, 11216, 3112]
    true_range = [float(values[mask].min()), float(values[mask].max())]
    assert true_range == generated["masked_intensity_range"]
    corrected, recount = {}, {}
    for arm, attempt in zip(("uint8", "float32"), outcome["arms"]):
        assert attempt == json.loads((ART/arm/"attempt.json").read_text())
        assert attempt["arm"] == arm and attempt["attempts"] == 1 and attempt["exit_code"] == 0
        limits = attempt["resources"]
        assert attempt["elapsed_seconds"] < limits["seconds_per_arm"]
        assert attempt["peak_supervisor_and_process_family_rss_bytes_sampled"] < limits["rss_limit_bytes"]
        assert attempt["minimum_available_memory_bytes_sampled"] > limits["available_memory_floor_bytes"]
        assert attempt["minimum_free_disk_bytes_sampled"] > limits["disk_floor_bytes"]
        assert attempt["combined_output_and_temporary_bytes"] < limits["output_temporary_limit_bytes"]
        assert limits["RSS_scope"] == "supervising_Python_plus_worker_and_recursive_descendants"
        stages = json.loads((ART/arm/"stages.json").read_text())
        result = json.loads((ART/arm/"result.json").read_text())
        assert stages["generated_input"] == generated and stages["worker_pid"] == attempt["worker_pid"]
        assert stages["N4_started"] and stages["N4_completed"]
        assert stages["Atropos_started"] == stages["Atropos_completed"] == (arm == "float32")
        boundary = result["mask_boundary"]
        assert boundary == stages["mask_boundary"]
        assert boundary["representation"] == record(mask.astype(arm))
        assert boundary["native_pixeltype"] == {"uint8": "unsigned char", "float32": "float"}[arm]
        assert boundary["numpy_dtype"] == arm and boundary["values_exact"] and boundary["support_exact"]
        assert boundary["binary_values"] == [0, 1]
        for f in (boundary["frame"], result["native_frame"], stages["N4_frame"]): frame_check(f)
        array = np.load(BASE/arm/"n4_raw.npy", allow_pickle=False)
        assert record(array) == result["N4_raw"] == observations["comparison"][arm]["array"]
        assert array.dtype == np.float32 and array.shape == (48, 48, 48)
        assert np.isfinite(array).all() and (array[mask] > 0).all()
        assert np.array_equal(array[~mask], values[~mask])
        assert result["input_unchanged"] and not result["patient_inputs_read"] and not result["anatomy_accepted"]
        assert result["N4_parameters"] == declaration["parameters"]["N4"]
        log = (ART/arm/"worker.log").read_text()
        match = re.search(r"Original intensity range:\s*\[([^,]+), ([^\]]+)\]", log)
        assert match is not None
        logged_range = [float(v) for v in match.groups()]
        warning = "Your input image contains nonpositive values" in log
        assert warning == observations["comparison"][arm]["log_nonpositive_warning"] == (arm == "uint8")
        if arm == "float32": assert np.allclose(logged_range, true_range, rtol=1e-5, atol=1e-4)
        else: assert logged_range == [0., 91.9219] and not np.allclose(logged_range, true_range)
        delta = float(np.max(abs(array-values)))
        assert delta == observations["comparison"][arm]["maximum_absolute_change_from_generated_input"]
        cvs = [float(np.std(array[truth == n], dtype=np.float64)/np.mean(array[truth == n], dtype=np.float64)) for n in (1, 2, 3)]
        assert np.allclose(cvs, observations["comparison"][arm]["generated_class_CV"], rtol=2e-6, atol=1e-9)
        corrected[arm] = array
        recount[arm] = {"native_mask_type": boundary["native_pixeltype"], "exact_binary_support_representation": boundary["representation"],
            "N4_raw": record(array), "logged_original_masked_range": logged_range, "nonpositive_warning": warning,
            "independent_generated_masked_range": true_range, "maximum_change_from_input": delta,
            "independent_float64_generated_class_CV": cvs, "N4_completed": True, "Atropos_completed": arm == "float32",
            "exit_code": 0, "resources": {k: attempt[k] for k in ("elapsed_seconds", "peak_supervisor_and_process_family_rss_bytes_sampled", "minimum_available_memory_bytes_sampled", "combined_output_and_temporary_bytes")}}
    between = float(np.max(abs(corrected["uint8"]-corrected["float32"])))
    assert between == observations["maximum_absolute_between_arm_difference"] == 36.21226501464844
    labels = np.load(BASE/"float32/classes_raw.npy", allow_pickle=False)
    result = json.loads((ART/"float32/result.json").read_text())
    assert record(labels) == result["raw_class_representation"]
    assert np.array_equal(labels, truth) and np.isfinite(labels).all()
    assert result["generated_truth_agreement"] == observations["float_generated_truth_agreement"] == 1.
    assert result["Atropos_parameters"] == declaration["parameters"]["Atropos"]
    for f in result["Atropos_frames"]: frame_check(f)
    total = np.zeros((48, 48, 48), np.float64)
    posteriors = []
    for code in (1, 2, 3):
        p = np.load(BASE/f"float32/posterior_{code}_raw.npy", allow_pickle=False)
        assert p.shape == mask.shape and p.dtype == np.float32 and np.isfinite(p).all()
        assert p.min() >= 0 and p.max() <= 1 and not p[~mask].any()
        total += p
        posteriors.append(record(p))
    error = float(np.max(abs(total[mask]-1)))
    assert error == result["raw_class_validation"]["posterior_sum_max_abs_error"] == observations["float_raw_class_validation"]["posterior_sum_max_abs_error"]
    assert error <= 2e-5 and not total[~mask].any()
    means = []
    for row in result["raw_class_validation"]["classes"]:
        assert row["voxels"] == int((labels == row["class"]).sum()) and not row["anatomical_acceptance"]
        mean = float(np.mean(corrected["float32"][labels == row["class"]], dtype=np.float64)); means.append(mean)
        assert np.isclose(mean, row["corrected_intensity_mean"], rtol=1e-6)
    assert means == sorted(means)
    assert outputs_before == {name: stamp(BASE/name) for name in outputs_before}
    assert before == {name: stamp(ROOT/name) for name in PINS}
    assert not any(k.split(".")[0] in {"ants", "torch", "nibabel", "totalsegmentator", "monai"} for k in sys.modules)
    report = {"schema": "ants-mask-dtype-independent-generated-audit-v1", "status": "pass_generated_diagnostic_recount",
        "created_utc": datetime.now(timezone.utc).isoformat(), "pinned_metadata_and_prior_v2_failure_unchanged": before,
        "generated_output_manifest_unchanged": outputs_before, "independent_source_contract": source,
        "independently_recreated_input": {"values": record(values), "mask": record(mask), "truth": record(truth),
            "masked_voxels": int(mask.sum()), "class_counts": [19224, 11216, 3112], "masked_range": true_range},
        "arms": recount, "maximum_paired_N4_difference": between,
        "float_arm_class_truth_exact_on_all_33552_generated_foreground_cells": True,
        "float_raw_posteriors": posteriors, "float_posterior_sum_max_abs_error": error,
        "float_class_corrected_intensity_means_float64": means,
        "conclusions": ["Exact pinned native N4 and wrapper source establish the float mask representation contract.",
            "The uint8 arm exited successfully yet reported a masked source intensity range inconsistent with the exact declared support, including an impossible zero minimum. This is observed mask misinterpretation at the native interface.",
            "The float32 arm reported the correct masked range, completed N4 and Atropos, and reproduced all generated classes with normalized raw membership probabilities.",
            "The paired outputs differ materially despite identical intensity, support values, frame and N4 settings; native mask representation is the declared controlled change.",
            "This makes the historical uint8 adapter a concrete defect and a plausible cause of v2 failure. It does not prove the exact cause of the unsymbolicated patient SIGSEGV."],
        "limitations": ["Generated concentric class agreement does not estimate real-patient tissue accuracy or anatomical validity.",
            "NPY files contain array shape/dtype without physical transforms; frame verification here checks sealed native-boundary/output metadata, independently recomputing all eight cell corners.",
            "There were two separately declared generated arms, one attempt each, not patient retries. The real-protocol history stays two failures with zero accepted tissue results.",
            "Raw class membership probabilities are not physical partial-volume fractions, and CSF-like intensity is not a ventricular segmentation.",
            "No new native execution, ANTs import, patient image read, host promotion or patient fitting release occurs in this audit."],
        "no_native_execution_or_patient_reads": True, "auditor_sha256": sha(__file__),
        "own_process_peak_rss_bytes": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss}
    content = json.dumps(report, indent=2, allow_nan=False)+"\n"; assert "/Users/" not in content
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("x") as out: out.write(content)
    print(json.dumps({"status": report["status"], "artifact": str(destination.relative_to(ROOT)), "sha256": sha(destination)}))


if __name__ == "__main__": main()
