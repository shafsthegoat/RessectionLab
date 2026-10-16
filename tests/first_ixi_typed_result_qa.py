"""Opt-in independent saved-result audit for the single IXI265 typed-mask v3 attempt.

Without --released-pins, runs tiny generated validation controls only. Actual
source/output decoding requires the parent's final attempt/declaration/candidate
hashes after its worker exits. Never imports ANTs or executes a model/solver.
The immutable prospective review, v1 failure, and quality audit are preserved.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import gc
import hashlib
import itertools
import json
from pathlib import Path
import resource
import sys
import time

import numpy as np
import psutil

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT/"outputs/poc-ixi265-atropos-typed-mask-v3"
DESTINATION = ROOT/"artifacts/ixi265-typed-mask-independent-v3/result-qa.json"
RUNNER = "scripts/poc_ixi265_atropos_typed_mask_run.py"
RUNNER_SHA = "b8d65e0522130df1a4e012811527ab17116a46436805c0152c0a3956a921e9ec"
KERNEL_SHA = "211d715a05b2dd269358b7bc5585488a053859ba65c04b9f31f60b69e41ec954"
PREPARATION = "artifacts/poc-ixi265-atropos-typed-mask-runner-guard-followup-v3"
OUTCOME = "artifacts/poc-ixi265-atropos-typed-mask-run-v3/outcome.json"
FROZEN = {
    RUNNER: RUNNER_SHA,
    "scripts/poc_ixi265_atropos_typed_mask.py": KERNEL_SHA,
    PREPARATION+"/preparation.json": "8a8335d6ab2f0b671c96eb80e6ae421e17cdaced294a5132dfeaf0c59ccfb6a1",
    "artifacts/ixi265-typed-mask-independent-v3/supervisor-qa.json": "1013cc0cde560fdce0e3c16338edd100421623cba2629c8bf3eaeeee8a8ed0c4",
    "artifacts/ixi265-typed-mask-independent-v3/kernel-preflight-qa.json": "08fc33228534fd2adb392a9093b57790695560ab9a881ee96cebddf0a23636b5",
    "artifacts/ixi265-typed-mask-independent-v3/supervisor-initial-findings.json": "41e74da8c8edd02418400ab87f73b29bcc155a265f383e47f1be1b15be3d01e9",
    "artifacts/ants-mask-dtype-independent-v1/independent-qa.json": "9c4b3bc648b9920ba7de4c5fd0c921b08382796e2ead3467caffc4213ee0c128",
    "artifacts/ixi265-quality-independent-v1/independent-qa.json": "7a51ca88a898b11eb7fc132180d5f77da4a0bffaf49ceb6e5b8eb229d6d162a6",
    "artifacts/ixi265-positive-domain-independent-v2/result-qa.json": "9c90b165a5da5657394ee1d26f0aa35193bc072e627b3b198d5fea1018beae72",
    "artifacts/poc-ixi265-atropos-run-v1/outcome.json": "55f4dd46f7cc692d4a66799cefca21bfa37430d3f5cf2ebe3eda675b6e4c989e",
    "artifacts/poc-ixi265-atropos-positive-domain-run-v2/outcome.json": "dce67f0acde21aa11fa99bdaa270108c8b53b5a091514182870e69e01fa78e21",
    "artifacts/poc-ixi265-atropos-preparation-v1/installed-runtime-audit.json": "8dc074db35e17be3b10f05e68e19ff61cdd44a6d1e0e744bbfc25e1da9073e79",
}
SOURCE = {
    "T1": ("outputs/poc-ixi265/originals/IXI265-T1.nii.gz", "317faf612b8e8a4e319fe3ec67404561b928ea9090fc23d1da20ce0900257950"),
    "brain_mask": ("outputs/poc-ixi265/synthstrip/brain_mask.nii.gz", "98710d7d1fa578dd510bf40b953e6d34926d8ff0c766cba16c005ffb6dbff192"),
    "host_bundle": ("outputs/poc-ixi265/IXI265-Guys-0845.ressectionlab", "c735daaf6d60f5f0d1567760a2accc18d4bd4838a096d8e0b5f490ee88abaff1"),
}
UNKNOWN = ((60,125,75), (67,135,49), (69,131,40), (89,120,83), (107,123,65), (127,219,68), (128,115,125), (185,127,119))
NAMES = {"n4_corrected_estimated", "class_labels_raw", "fit_domain_native", "excluded_unknown_native",
         "class_1_posterior", "class_2_posterior", "class_3_posterior", "csf_like_intensity_estimated",
         "gray_matter_like_intensity_estimated", "white_matter_like_intensity_estimated"}
MEMORY = []


def sha(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def stamp(path):
    p = Path(path)
    return {"sha256": sha(p), "bytes": p.stat().st_size, "mtime_ns": p.stat().st_mtime_ns}


def memory(phase):
    row = {"phase": phase, "rss_bytes": psutil.Process().memory_info().rss,
           "available_bytes": psutil.virtual_memory().available}
    MEMORY.append(row)
    assert row["rss_bytes"] < 1024**3 and row["available_bytes"] >= 1024**3, "Independent audit memory guard"


def voxel_corners(shape, affine):
    indices = np.array(list(itertools.product(*[(-.5, n-.5) for n in shape])))
    return indices @ affine[:3, :3].T + affine[:3, 3]


def check_frame(image, source):
    assert image.shape == source.shape
    affine_error = float(np.max(abs(image.affine-source.affine)))
    corner_error = np.linalg.norm(voxel_corners(image.shape, image.affine)-voxel_corners(source.shape, source.affine), axis=1)
    assert affine_error <= 1e-4 and corner_error.max() <= 1e-4, "Saved native frame exceeds declared tolerance"
    checks = {"selected_affine_max_abs_error": affine_error, "tolerance_mm": 1e-4,
        "bitwise_affine_equality_claimed": False, "native_shape": list(image.shape), "dtype": str(image.get_data_dtype())}
    for form in ("qform", "sform"):
        actual, acode = getattr(image, "get_"+form)(coded=True)
        original, ocode = getattr(source, "get_"+form)(coded=True)
        assert acode == ocode
        checks[form+"_code"] = int(acode)
        if acode:
            error = np.linalg.norm(voxel_corners(image.shape, actual)-voxel_corners(source.shape, original), axis=1)
            assert error.max() < 1e-4
            checks[form+"_eight_source_cell_corner_errors_mm"] = error.tolist()
    checks["eight_selected_cell_corner_errors_mm"] = np.linalg.norm(voxel_corners(image.shape, image.affine)-voxel_corners(source.shape, source.affine), axis=1).tolist()
    return checks


def check_partition(brain, domain, unknown, labels):
    assert brain.dtype == domain.dtype == unknown.dtype == np.bool_
    assert brain.shape == domain.shape == unknown.shape == labels.shape
    assert np.array_equal(brain.astype(np.uint8), domain.astype(np.uint8)+unknown.astype(np.uint8))
    assert np.array_equal(labels > 0, domain)
    assert np.isfinite(labels).all() and np.array_equal(labels, np.rint(labels))
    assert set(np.unique(labels)) <= {0, 1, 2, 3}
    assert all(np.any(labels == code) for code in (1, 2, 3))


def check_posterior(posterior, domain):
    assert posterior.shape == domain.shape and np.isfinite(posterior).all()
    assert float(posterior.min()) >= 0 and float(posterior.max()) <= 1
    assert np.count_nonzero(posterior[~domain]) == 0


def generated_controls():
    brain = np.ones((3, 2, 2), bool)
    unknown = np.zeros_like(brain); unknown[0, 0, 0] = True
    domain = brain & ~unknown
    labels = np.broadcast_to(np.arange(1, 4)[:, None, None], domain.shape).copy(); labels[unknown] = 0
    check_partition(brain, domain, unknown, labels)
    pp = [np.where(domain, np.where(labels == k, .8, .1), 0) for k in (1, 2, 3)]
    for p in pp: check_posterior(p, domain)
    assert np.max(abs(sum(pp)[domain]-1)) < 1e-14
    faults = 0
    for kind in ("unknown_class", "outside_posterior", "nan", "negative", "larger_than_one", "extra_abstention"):
        bad_labels, bad = labels.copy(), pp[0].copy()
        if kind == "unknown_class": bad_labels[unknown] = 1
        elif kind == "outside_posterior": bad[unknown] = .1
        elif kind == "nan": bad[1, 0, 0] = np.nan
        elif kind == "negative": bad[1, 0, 0] = -.00001
        elif kind == "larger_than_one": bad[1, 0, 0] = 1.00001
        else: bad_labels[1, 0, 0] = 0
        try:
            check_partition(brain, domain, unknown, bad_labels); check_posterior(bad, domain)
        except AssertionError: faults += 1
    assert faults == 6
    return {"valid_generated_partition_and_probabilities": True, "six_corruption_controls_refused": faults,
            "real_images_read": False, "production_validation_helpers_called": False}



def extended_generated_controls():
    from types import SimpleNamespace
    result = generated_controls()
    def image(affine, code=1):
        return SimpleNamespace(shape=(17, 11, 9), affine=affine, get_data_dtype=lambda: np.dtype("float32"),
            get_qform=lambda coded: (affine.copy(), code), get_sform=lambda coded: (affine.copy(), code))
    source = image(np.diag([.9375, .9375, 1.2, 1.0]))
    valid = source.affine.copy(); valid[0, 3] += 1e-6
    check_frame(image(valid), source)
    refused = 0
    for fault in ("translation", "accumulated_corner", "form_code"):
        bad = source.affine.copy()
        if fault == "translation": bad[1, 3] += .001
        if fault == "accumulated_corner": bad[0, 0] += .00004
        try: check_frame(image(bad, 2 if fault == "form_code" else 1), source)
        except AssertionError: refused += 1
    assert refused == 3
    result.update(sub_tolerance_nonbitwise_frame_accepted=True,
        three_translation_corner_or_form_code_corruptions_refused=refused)
    return result


def metadata_audit(pins):
    assert pins["worker_exited"] is True and pins["payload_audit_released"] is True
    assert pins["status"] in ("failed", "fitted_candidate_unaccepted")
    fixed = {name: stamp(ROOT/name) for name in FROZEN}
    assert {name: row["sha256"] for name, row in fixed.items()} == FROZEN
    outcome_record = stamp(ROOT/OUTCOME)
    assert outcome_record["sha256"] == pins["outcome_sha256"]
    outcome = json.loads((ROOT/OUTCOME).read_text())
    sealed = (ROOT/OUTCOME).parent
    inventory_path = sealed/"output-inventory.json"
    assert sha(inventory_path) == pins["inventory_sha256"] == outcome["output_inventory_sha256"]
    inventory = json.loads(inventory_path.read_text())
    assert Path(inventory["output_root"]) == BASE
    assert sum(row["bytes"] for row in inventory["all_files"]) == inventory["total_retained_bytes"]
    assert {row["path"] for row in inventory["all_files"]} == {
        str(path.relative_to(BASE)) for path in BASE.rglob("*") if path.is_file()}
    for row in inventory["all_files"]:
        path = BASE/row["path"]
        assert path.is_relative_to(BASE) and path.resolve() == path
        assert path.stat().st_size == row["bytes"] and sha(path) == row["sha256"]
    documents, records = {}, {}
    required = {"attempt.json", "declaration.json", "worker-input.json", "worker.log"}
    if pins["status"] == "fitted_candidate_unaccepted": required |= {"candidate.json", "stages.json"}
    assert required <= set(pins["files"]) <= required | {"stages.json"}
    for name, pin in pins["files"].items():
        path = BASE/name
        assert path.resolve() == path
        records[name] = stamp(path); assert records[name]["sha256"] == pin
        if name.endswith(".json"): documents[name] = json.loads(path.read_text())
    attempt, declaration = documents["attempt.json"], documents["declaration.json"]
    documents["sealed-inventory"] = inventory
    for field, name in (("attempt_sha256", "attempt.json"), ("declaration_sha256", "declaration.json")):
        assert outcome[field] == records[name]["sha256"]
    if pins["status"] == "fitted_candidate_unaccepted":
        assert outcome["candidate_sha256"] == records["candidate.json"]["sha256"]
        assert outcome["stages_sha256"] == records["stages.json"]["sha256"]
        assert outcome["support"] == documents["candidate.json"]["support"]
        assert outcome["numerical_checks"] == documents["candidate.json"]["numerical_audit"]
    assert outcome["status"] == attempt["status"] and outcome["exit_code"] == attempt["exit_code"]
    assert all(outcome[key] == attempt[key] for key in ("runtime_unchanged", "lineage_unchanged", "historical_markers_unchanged"))
    assert outcome["source_continuity"] == attempt["source_continuity_after_attempt"]
    assert all(outcome["resources"][key] == attempt[key] for key in outcome["resources"])
    stage_names = [row["name"] for row in documents.get("stages.json", {}).get("stages", [])]
    assert outcome["actual_attempt_denominator"] == {"total_patient_protocol_attempts_started": 3,
        "retained_failed_attempts": 2, "completed_fitting_candidates": int(pins["status"] == "fitted_candidate_unaccepted"),
        "N4_started": 1+int("N4_start" in stage_names), "N4_completed": int("N4_complete" in stage_names),
        "Atropos_started": int("Atropos_start" in stage_names), "Atropos_completed": int("Atropos_complete" in stage_names),
        "anatomy_accepted": 0, "host_accepted": 0, "growth_use_accepted": 0}
    for relative, pin in outcome["retained_failures"].items(): assert FROZEN[relative] == pin
    release = sealed/"root-release-declaration.json"
    assert sha(release) == outcome["root_release_declaration_sha256"]
    released = json.loads(release.read_text())
    assert released["released_attempts"] == 1 and released["prior_real_attempts_started"] == 2
    assert released["no_automatic_retry_tuning_crop_source_or_support_changes"] is True
    assert sha(sealed/"released-run-spec.json") == outcome["released_spec_sha256"] == released["released_spec_sha256"]
    released_spec = json.loads((sealed/"released-run-spec.json").read_text())
    prospective = json.loads((ROOT/PREPARATION/"prospective-run-spec.json").read_text())
    assert released_spec.pop("released_single_attempt") is True
    assert prospective.pop("released_single_attempt") is False and released_spec == prospective
    method = json.loads((ROOT/PREPARATION/"prospective-method.json").read_text())
    assert attempt["status"] == pins["status"]
    assert attempt["runner_sha256"] == declaration["runner_sha256"] == RUNNER_SHA
    assert attempt["kernel_sha256"] == declaration["kernel_sha256"] == KERNEL_SHA
    assert declaration["preparation_sha256"] == FROZEN[PREPARATION+"/preparation.json"]
    for key in ("patient_group", "role", "training_admitted", "parameters", "resources", "source_hashes", "expected_unknown_indices_native"):
        assert declaration[key] == method[key]
    assert declaration["attempts"] == 1 and declaration["automatic_retry_or_tuning"] is False
    assert declaration["released_single_attempt"] is True and declaration["patient_fitting_released"] is True
    for key in ("anatomy_accepted", "host_accepted", "growth_use_accepted"):
        assert declaration[key] is attempt[key] is False
    for key in ("brain_mask_replaced", "intensity_offset_or_clipping", "source_crop"):
        assert declaration[key] is False
    assert attempt["resources"] == method["resources"]
    expected_metadata = {"patient_group": "IXI:265", "role": "TRAIN", "training_admitted": False,
        "T1_sha256": SOURCE["T1"][1], "brain_mask_sha256": SOURCE["brain_mask"][1], "host_sha256": SOURCE["host_bundle"][1]}
    assert declaration["metadata"] == documents["worker-input.json"]["metadata"] == expected_metadata
    for key, (relative, pin) in SOURCE.items():
        path = ROOT/relative; assert path.resolve() == path
        assert declaration["sources"][key] == documents["worker-input.json"]["paths"][key] == str(path)
        assert declaration["source_hashes"][key] == pin
    spec = documents["worker-input.json"]
    assert spec["released_single_attempt"] is True and Path(spec["output"]) == BASE
    runtime = ROOT/".tools/poc-antspyx-0.6.3-py312-v1/runtime"
    assert Path(spec["runtime"]).resolve() == Path(declaration["runtime"]["prefix"]).resolve() == runtime.resolve()
    marker = runtime/"ixi265-n4-atropos-typed-mask-v3-attempt.json"
    assert json.loads(marker.read_text()) == declaration
    markers = {"v3": stamp(marker)}
    assert set(declaration["retained_historical_markers"]) == {
        "ixi265-n4-atropos-single-attempt.json", "ixi265-n4-atropos-positive-domain-v2-attempt.json"}
    for name, pin in declaration["retained_historical_markers"].items():
        markers[name] = stamp(runtime/name); assert markers[name]["sha256"] == pin
    assert all(attempt["historical_markers_unchanged"].values())
    assert attempt["runtime_unchanged"] is True and attempt["lineage_unchanged"] is True
    assert attempt["source_continuity_after_attempt"]["all_unchanged"] is True
    assert attempt["source_continuity_after_attempt"]["payloads_hashed"] is True
    for key, (_, pin) in SOURCE.items():
        assert attempt["source_continuity_after_attempt"]["files"][key] == {
            "sha256": pin, "expected_sha256": pin, "unchanged": True}
    assert attempt["declaration_sha256"] == records["declaration.json"]["sha256"]
    for relative, pin in declaration["lineage"].items():
        path = ROOT/relative
        assert path.is_relative_to(ROOT) and path.resolve() == path and sha(path) == pin
    if pins["status"] == "fitted_candidate_unaccepted":
        assert [stage["name"] for stage in documents["stages.json"]["stages"]] == [
            "N4_start", "N4_complete", "Atropos_start", "Atropos_complete"]
        assert documents["stages.json"]["worker_pid"] == attempt["worker_pid"]
        assert documents["stages.json"]["support"] == documents["candidate.json"]["support"]
        for row in attempt["manifest"]:
            path = BASE/row["path"]
            assert path.is_relative_to(BASE) and path.resolve() == path
            assert sha(path) == row["sha256"] and path.stat().st_size == row["bytes"]
    retained_runtime = json.loads((ROOT/"artifacts/poc-ixi265-atropos-preparation-v1/installed-runtime-audit.json").read_text())
    for key in ("packages", "runtime_plan_sha256", "python", "verified_official_API_sources", "installed_native_binary_hashes"):
        assert declaration["runtime"][key] == retained_runtime[key]
    return documents, records, fixed, markers, outcome, outcome_record


def successful_candidate(documents):
    import nibabel as nib
    candidate, attempt = documents["candidate.json"], documents["attempt.json"]
    declaration = documents["declaration.json"]
    assert candidate["status"] == "fitted_candidate_unaccepted" and candidate["runner_sha256"] == RUNNER_SHA
    assert candidate["source_bytes_unchanged"] is True
    assert not any(candidate[k] for k in ("anatomy_accepted", "host_accepted", "growth_use_accepted"))
    assert candidate["parameters"] == declaration["parameters"] and candidate["resources"] == attempt["resources"]
    retained = json.loads((ROOT/"artifacts/poc-ixi265-atropos-preparation-v1/installed-runtime-audit.json").read_text())
    runtime_keys = ("packages", "runtime_plan_sha256", "verified_official_API_sources", "installed_native_binary_hashes")
    assert all(candidate["runtime"][key] == retained[key] for key in runtime_keys)
    assert candidate["package"]["version"] == "0.6.3"
    assert candidate["package"]["wheel_sha256"] == "e378329089f622c72eacc47e4e2c18503d0f3fbf2cc39ccb9df65880c0a7846f"
    assert attempt["exit_code"] == 0
    limits = attempt["resources"]
    assert attempt["elapsed_seconds"] <= limits["timeout_seconds"]
    assert attempt["peak_supervisor_and_process_family_rss_bytes_sampled"] <= limits["rss_limit_bytes"]
    assert attempt["minimum_available_memory_bytes_sampled"] >= limits["minimum_available_bytes"]
    assert attempt["output_and_temporary_bytes"] <= limits["output_limit_bytes"]
    assert attempt["minimum_free_disk_bytes_sampled"] >= limits["free_disk_floor_bytes"]
    outputs = {row["name"]: row for row in candidate["outputs"]}
    assert len(candidate["outputs"]) == len(outputs) == 10 and set(outputs) == NAMES
    output_before = {}
    for name, row in outputs.items():
        output_before[name] = stamp(BASE/(name+".nii.gz"))
        assert output_before[name]["sha256"] == row["sha256"] and output_before[name]["bytes"] == row["bytes"]
        assert row["native_shape"] == [256, 256, 150]
    source = nib.load(ROOT/SOURCE["T1"][0])
    brain_image = nib.load(ROOT/SOURCE["brain_mask"][0])
    assert source.shape == (256, 256, 150)
    assert source.get_data_dtype() == np.dtype("int16")
    source_frames = {"brain_mask": check_frame(brain_image, source)}
    intensity = np.asarray(source.dataobj, dtype=np.float32)
    raw_brain = np.asarray(brain_image.dataobj)
    assert set(np.unique(raw_brain)) == {0, 1}
    brain = raw_brain.astype(bool); del raw_brain
    assert int(brain.sum()) == 1451509
    assert np.isfinite(intensity).all() and intensity.min() == 0
    assert set(map(tuple, np.argwhere(brain & (intensity == 0)))) == set(UNKNOWN)
    expected_domain = brain & (intensity > 0)
    expected_unknown = brain & ~expected_domain
    frames = {}
    def array(name):
        image = nib.load(BASE/(name+".nii.gz"))
        frames[name] = check_frame(image, source)
        result = np.asarray(image.dataobj)
        assert result.shape == source.shape
        memory("loaded_"+name)
        return result
    raw_domain, raw_unknown = array("fit_domain_native"), array("excluded_unknown_native")
    assert set(np.unique(raw_domain)) == set(np.unique(raw_unknown)) == {0, 1}
    domain, unknown = raw_domain.astype(bool), raw_unknown.astype(bool)
    del raw_domain, raw_unknown
    assert np.array_equal(domain, expected_domain) and np.array_equal(unknown, expected_unknown)
    assert domain.sum() == 1451501 and unknown.sum() == 8
    del expected_domain, expected_unknown, intensity
    labels = array("class_labels_raw")
    check_partition(brain, domain, unknown, labels)
    support = candidate["support"]
    assert support["original_brain_voxels"] == int(brain.sum()) and support["fit_domain_voxels"] == int(domain.sum())
    assert support["excluded_unknown_voxels"] == 8 and support["unknown_indices_native"] == [list(p) for p in UNKNOWN]
    assert support["excluded_semantics"] == "UNKNOWN_ABSTAINED" and support["source_arrays_modified"] is False
    total = np.zeros(source.shape, np.float64)
    posterior_stats = []
    raw_posteriors = []
    for code in (1, 2, 3):
        name = f"class_{code}_posterior"
        posterior = array(name)
        check_posterior(posterior, domain)
        assert posterior.dtype == np.float32
        assert outputs[name]["semantics"] == "unmodified_membership_probability_not_PVE"
        raw_paths = [row["path"] for row in documents["sealed-inventory"]["all_files"]
            if row["path"].startswith("temporary/") and row["path"].endswith(f"prob{code:02d}.nii.gz")]
        assert len(raw_paths) == 1
        raw_image = nib.load(BASE/raw_paths[0])
        assert raw_image.shape == source.shape
        physical_error = np.linalg.norm(voxel_corners(raw_image.shape, raw_image.affine)-voxel_corners(source.shape, source.affine), axis=1)
        assert float(np.max(abs(raw_image.affine-source.affine))) <= 1e-4 and physical_error.max() <= 1e-4
        raw = np.asarray(raw_image.dataobj)
        assert raw.dtype == posterior.dtype and np.array_equal(raw, posterior)
        raw_posteriors.append({"class": code, "saved_array_exactly_equals_retained_native_raw_array": True,
            "native_temporary_selected_frame_eight_corner_errors_mm": physical_error.tolist()})
        del raw, raw_image
        total += posterior
        posterior_stats.append({"class": code, "dtype": str(posterior.dtype), "minimum": float(posterior.min()), "maximum": float(posterior.max()),
            "nonzero_outside_fitting_support": int(np.count_nonzero(posterior[~domain]))})
        del posterior; gc.collect()
    sum_error = float(np.max(abs(total[domain]-1)))
    assert sum_error <= 2e-5
    assert np.count_nonzero(total[~domain]) == 0
    assert sum_error == candidate["numerical_audit"]["classes"]["posterior_sum_max_abs_error"]
    del total
    corrected = array("n4_corrected_estimated")
    assert np.isfinite(corrected).all() and np.all(corrected[domain] > 0)
    classes = []
    names = ("csf_like_intensity_estimated", "gray_matter_like_intensity_estimated", "white_matter_like_intensity_estimated")
    for code, name in enumerate(names, 1):
        selected = labels == code
        selected_values = corrected[selected]
        row = {"class": code, "voxels": int(selected.sum()), "float64_mean_corrected_intensity": float(np.mean(selected_values, dtype=np.float64)),
               "declared_assumption": name, "anatomical_acceptance": False}
        recorded = candidate["numerical_audit"]["classes"]["classes"][code-1]
        assert recorded["class"] == code and recorded["voxels"] == row["voxels"] and recorded["declared_assumption"] == name
        assert recorded["anatomical_acceptance"] is False
        assert np.isclose(row["float64_mean_corrected_intensity"], recorded["corrected_intensity_mean"], rtol=2e-6, atol=1e-5)
        hard = array(name)
        assert set(np.unique(hard)) <= {0, 1} and np.array_equal(hard, selected)
        classes.append(row)
        del hard, selected, selected_values
    assert classes[0]["float64_mean_corrected_intensity"] < classes[1]["float64_mean_corrected_intensity"] < classes[2]["float64_mean_corrected_intensity"]
    assert sum(c["voxels"] for c in classes) == 1451501
    assert candidate["numerical_audit"]["classes"]["probabilities_modified"] is False
    assert output_before == {name: stamp(BASE/(name+".nii.gz")) for name in outputs}
    memory("candidate_complete")
    boundary = candidate["numerical_audit"]["N4_mask_boundary"]
    assert boundary["native_pixeltype"] == "float" and boundary["numpy_dtype"] == "float32"
    assert boundary["binary_values_exact"] is True and boundary["support_unchanged"] is True and boundary["support_voxels"] == 1451501
    assert candidate["numerical_audit"]["mask_representation_only_change"] is True
    assert candidate["numerical_audit"]["parameters_changed"] is False
    return {"status": "pass_saved_numerical_candidate_remains_unaccepted",
        "recorded_N4_float_mask_boundary": boundary,
 "outputs_unchanged": output_before,
        "native_frame_checks": frames, "source_frame_checks": source_frames,
        "source_shape": list(source.shape), "source_selected_affine_ras_mm": source.affine.tolist(),
        "support": {"original_brain": 1451509, "fitting_domain": 1451501, "unknown_abstained": 8, "unknown_indices_native": [list(p) for p in UNKNOWN]},
        "source_arrays_and_original_brain_preserved": True, "class_zero_exactly_outside_fitting_domain": True,
        "raw_posterior_checks": posterior_stats, "posterior_sum_max_abs_error_independent": sum_error,
        "raw_native_temporary_posterior_comparisons": raw_posteriors,
        "corrected_finite_positive_on_fitting_domain": True, "ordered_classes": classes, "hard_masks_equal_raw_class_ids": True,
        "installed_package_API_and_binary_metadata_match_frozen_setup": True,
        "runtime_keys_compared": list(runtime_keys), "runtime_plan_sha256": retained["runtime_plan_sha256"],
        "actual_attempt_resource_receipt": {k: attempt[k] for k in ("elapsed_seconds", "peak_supervisor_and_process_family_rss_bytes_sampled", "minimum_available_memory_bytes_sampled", "output_and_temporary_bytes", "minimum_free_disk_bytes_sampled", "exit_code")},
        "no_anatomical_host_growth_training_or_PVE_acceptance": True}


def failed_attempt(documents):
    attempt = documents["attempt.json"]
    assert attempt["status"] == "failed"
    stages = [s["name"] for s in documents.get("stages.json", {}).get("stages", [])]
    assert stages == ["N4_start", "N4_complete", "Atropos_start", "Atropos_complete"][:len(stages)]
    return {"status": "pass_failed_attempt_metadata_and_source_continuity_only",
        "failure": attempt["failure"], "exit_code": attempt["exit_code"], "stages_observed": stages,
        "N4_entered": "N4_start" in stages, "N4_completed": "N4_complete" in stages,
        "Atropos_entered": "Atropos_start" in stages, "Atropos_completed": "Atropos_complete" in stages,
        "candidate_receipt_exists": (BASE/"candidate.json").is_file(),
        "native_output_array_decodes": False, "partial_files_or_stage_completions_are_not_candidate_acceptance": True,
        "candidate_accepted": False, "failure_metadata_preserved": True}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--released-pins", type=Path)
    args = parser.parse_args()
    controls = extended_generated_controls()
    if args.released_pins is None:
        print(json.dumps({"status": "prepared_generated_controls_pass", "controls": controls,
            "patient_or_result_payload_reads": False})); return
    started = time.monotonic()
    assert not DESTINATION.exists(), "Saved result audit is immutable"
    pins = json.loads(args.released_pins.read_text())
    assert pins["auditor_sha256"] == sha(__file__)
    documents, records, fixed, markers, outcome, outcome_record = metadata_audit(pins)
    source_before = {name: {"path": relative, **stamp(ROOT/relative)} for name, (relative, _) in SOURCE.items()}
    assert {name: row["sha256"] for name, row in source_before.items()} == {name: pin for name, (_, pin) in SOURCE.items()}
    memory("before_source_decode")
    result = successful_candidate(documents) if pins["status"] == "fitted_candidate_unaccepted" else failed_attempt(documents)
    assert source_before == {name: {"path": relative, **stamp(ROOT/relative)} for name, (relative, _) in SOURCE.items()}
    assert fixed == {name: stamp(ROOT/name) for name in FROZEN}
    assert records == {name: stamp(BASE/name) for name in records}
    assert outcome_record == stamp(ROOT/OUTCOME)
    assert sha((ROOT/OUTCOME).parent/"output-inventory.json") == pins["inventory_sha256"]
    runtime = ROOT/".tools/poc-antspyx-0.6.3-py312-v1/runtime"
    assert markers["v3"] == stamp(runtime/"ixi265-n4-atropos-typed-mask-v3-attempt.json")
    for name, row in markers.items():
        if name != "v3": assert row == stamp(runtime/name)
    assert not any(key.split(".")[0] in {"ants", "torch", "totalsegmentator", "monai"} for key in sys.modules)
    if pins["status"] == "failed": assert "nibabel" not in sys.modules
    memory("finished")
    report = {"schema": "ixi265-typed-mask-independent-result-audit-v3", "status": result["status"],
        "created_utc": datetime.now(timezone.utc).isoformat(), "result": result,
        "sources_unchanged": source_before, "host_bundle_hashed_only": True,
        "frozen_prior_evidence_unchanged": fixed, "new_attempt_metadata_unchanged": records,
        "sealed_outcome": {"path": OUTCOME, **outcome_record}, "single_attempt_markers": markers,
        "sealed_output_inventory": {"sha256": pins["inventory_sha256"],
            "total_retained_bytes": documents["sealed-inventory"]["total_retained_bytes"],
            "all_files_independently_hashed": True,
            "final_attempt_receipt_included_in_retained_bytes": True},
        "generated_controls": controls,
        "historical_v1": {"started": 1, "N4_completions": 0, "Atropos_completions": 0, "accepted_tissues": 0},
        "historical_v2": {"started": 1, "N4_entries": 1, "N4_completions": 0, "Atropos_completions": 0, "accepted_tissues": 0},
        "v3": {"started": 1, "completed_numerical_candidates": int(pins["status"] == "fitted_candidate_unaccepted"), "accepted_tissues": 0, "automatic_retries": 0},
        "combined_real_protocol_denominator": {"attempts_started": 3, "successful_candidates": int(pins["status"] == "fitted_candidate_unaccepted"), "accepted_tissues": 0, "automatic_retries": 0},
        "source_fix_clipping_normalization_or_mask_replacement": False,
        "model_or_solver_import_or_execution": False, "elapsed_seconds": time.monotonic()-started,
        "own_process_peak_rss_bytes": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss, "memory_samples": MEMORY,
        "auditor_sha256": sha(__file__), "release_pins_sha256": sha(args.released_pins),
        "limitations": ["Saved numerical consistency does not establish anatomical correctness or biophysical validity.",
            "Intensity-ordered classes are unreviewed assumptions; raw memberships are not physical partial-volume fractions.",
            "Eight unknown cells remain abstained; CSF-like intensity does not identify ventricles.",
            "No host upgrade, tissue-conditioned growth, training admission or clinical accuracy is accepted.",
            "Frozen package/API/binary metadata is compared without ANTs import or a new binary hash sweep.",
            "The production memory cap is sampled; between-sample peaks are not established.",
            "Native frame checks use the declared 1e-4 mm tolerance and eight voxel-cell corners; bitwise-affine equality is not claimed."]}
    content = json.dumps(report, indent=2, allow_nan=False)+"\n"
    assert "/Users/" not in content
    DESTINATION.parent.mkdir(parents=True, exist_ok=True)
    with DESTINATION.open("x") as stream: stream.write(content)
    print(json.dumps({"status": report["status"], "artifact": str(DESTINATION.relative_to(ROOT)),
        "sha256": sha(DESTINATION), "peak_rss_bytes": report["own_process_peak_rss_bytes"]}))


if __name__ == "__main__": main()


