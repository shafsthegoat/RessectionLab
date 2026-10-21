"""Opt-in independent saved-result audit for the single IXI265 v2 attempt.

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
import struct
import sys
import time
import uuid

import numpy as np
import psutil

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT/"outputs/poc-ixi265-atropos-positive-domain-v2"
DESTINATION = ROOT/"artifacts/ixi265-positive-domain-independent-v2/result-qa.json"
RUNNER = "scripts/poc_ixi265_atropos_positive_domain.py"
RUNNER_SHA = "0652eca1ba81304e053f24c9a9fc2f8c89174bfe7447accfd21105263592d81d"
FROZEN = {
    RUNNER: RUNNER_SHA,
    "scripts/poc_ixi265_atropos.py": "4a5c5b4b2f85153212ac27e6356d5f1dca5c8b42d77257fb55a2fee53253624b",
    "artifacts/ixi265-positive-domain-independent-v2/prospective-qa.json": "894af5ba11bb92ed6f1cc6f41ea93d5d6e6b4d49053f9cd04987dc875744a3bc",
    "artifacts/ixi265-quality-independent-v1/independent-qa.json": "7a51ca88a898b11eb7fc132180d5f77da4a0bffaf49ceb6e5b8eb229d6d162a6",
    "artifacts/poc-ixi265-atropos-run-v1/outcome.json": "55f4dd46f7cc692d4a66799cefca21bfa37430d3f5cf2ebe3eda675b6e4c989e",
    "artifacts/poc-ixi265-atropos-positive-domain-preparation-v2/preparation.json": "2c1d39e4a318598385dfe331de08e1f6de3e2dba5850a8b0acd8978045376c3d",
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
FAILURE_OUTCOME = "artifacts/poc-ixi265-atropos-positive-domain-run-v2/outcome.json"
FAILURE_OUTCOME_SHA = "dce67f0acde21aa11fa99bdaa270108c8b53b5a091514182870e69e01fa78e21"


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
    assert np.array_equal(image.affine, source.affine), "Saved selected native affine changed"
    checks = {"selected_affine_exact": True, "native_shape": list(image.shape), "dtype": str(image.get_data_dtype())}
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


def metadata_audit(pins):
    assert pins["worker_exited"] is True and pins["payload_audit_released"] is True
    assert pins["status"] in ("failed", "fitted_candidate_unaccepted")
    fixed = {name: stamp(ROOT/name) for name in FROZEN}
    assert {name: row["sha256"] for name, row in fixed.items()} == FROZEN
    documents, records = {}, {}
    required = {"attempt.json", "declaration.json", "worker-input.json", "worker.log"}
    if pins["status"] == "fitted_candidate_unaccepted": required.add("candidate.json")
    assert set(pins["files"]) == required
    for name in sorted(required):
        records[name] = stamp(BASE/name)
        assert records[name]["sha256"] == pins["files"][name]
        if name.endswith(".json"): documents[name] = json.loads((BASE/name).read_text())
    attempt, declaration = documents["attempt.json"], documents["declaration.json"]
    prep = json.loads((ROOT/"artifacts/poc-ixi265-atropos-positive-domain-preparation-v2/preparation.json").read_text())
    assert attempt["status"] == pins["status"]
    assert attempt["runner_sha256"] == declaration["runner_sha256"] == RUNNER_SHA
    assert declaration["preparation_sha256"] == FROZEN["artifacts/poc-ixi265-atropos-positive-domain-preparation-v2/preparation.json"]
    for doc in (attempt, declaration):
        for key in ("patient_group", "role", "training_admitted", "parameters", "resources", "source_hashes", "expected_unknown_indices_native"):
            assert doc[key] == prep["method"][key]
        assert all(doc[k] is False for k in ("anatomy_accepted", "host_accepted", "growth_use_accepted", "brain_mask_replaced", "intensity_offset_or_clipping", "source_crop"))
    assert declaration["v1_attempt_retained"] is True and declaration["automatic_retry_or_tuning"] is False
    assert declaration["real_attempt_number_for_this_protocol"] == 1
    runtime = ROOT/".tools/poc-antspyx-0.6.3-py312-v1/runtime"
    assert Path(declaration["runtime"]).resolve() == runtime.resolve()
    assert Path(declaration["output"]) == BASE
    marker = runtime/"ixi265-n4-atropos-positive-domain-v2-attempt.json"
    assert json.loads(marker.read_text()) == declaration
    old_marker = runtime/"ixi265-n4-atropos-single-attempt.json"
    assert sha(old_marker) == "000c813b68f7d96d78b22de068db50e4890cc45d5e07877a3be57256d82ae870"
    for key, (path, expected) in SOURCE.items():
        assert declaration["sources"][key] == {"path": str(ROOT/path), "sha256": expected}
        assert documents["worker-input.json"]["paths"][key] == str(ROOT/path)
    assert documents["worker-input.json"]["released_single_attempt"] is True
    return documents, records, fixed, {"v1_marker_sha256": sha(old_marker), "v2_marker_sha256": sha(marker)}


def successful_candidate(documents):
    import nibabel as nib
    candidate, attempt = documents["candidate.json"], documents["attempt.json"]
    assert candidate["status"] == "fitted_candidate_unaccepted" and candidate["runner_sha256"] == RUNNER_SHA
    assert candidate["source_bytes_unchanged"] is True
    assert not any(candidate[k] for k in ("anatomy_accepted", "host_accepted", "growth_use_accepted"))
    assert candidate["parameters"] == attempt["parameters"] and candidate["resources"] == attempt["resources"]
    retained = json.loads((ROOT/"artifacts/poc-ixi265-atropos-preparation-v1/installed-runtime-audit.json").read_text())
    runtime_keys = ("packages", "runtime_plan_sha256", "verified_official_API_sources", "installed_native_binary_hashes")
    assert all(candidate["runtime"][key] == retained[key] for key in runtime_keys)
    assert candidate["package"]["version"] == "0.6.3"
    assert candidate["package"]["wheel_sha256"] == "e378329089f622c72eacc47e4e2c18503d0f3fbf2cc39ccb9df65880c0a7846f"
    assert attempt["exit_code"] == 0
    limits = attempt["resources"]
    assert attempt["elapsed_seconds"] <= limits["timeout_seconds"]
    assert attempt["peak_process_family_rss_bytes_sampled"] <= limits["rss_limit_bytes"]
    assert attempt["minimum_available_memory_bytes_sampled"] >= limits["minimum_available_bytes"]
    assert attempt["output_and_temporary_bytes"] <= limits["output_limit_bytes"]
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
    for code in (1, 2, 3):
        name = f"class_{code}_posterior"
        posterior = array(name)
        check_posterior(posterior, domain)
        assert outputs[name]["semantics"] == "unmodified_fitted_membership_probability_not_PVE"
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
    return {"status": "pass_saved_numerical_candidate_remains_unaccepted", "outputs_unchanged": output_before,
        "native_frame_checks": frames, "source_frame_checks": source_frames,
        "source_shape": list(source.shape), "source_selected_affine_ras_mm": source.affine.tolist(),
        "support": {"original_brain": 1451509, "fitting_domain": 1451501, "unknown_abstained": 8, "unknown_indices_native": [list(p) for p in UNKNOWN]},
        "source_arrays_and_original_brain_preserved": True, "class_zero_exactly_outside_fitting_domain": True,
        "raw_posterior_checks": posterior_stats, "posterior_sum_max_abs_error_independent": sum_error,
        "corrected_finite_positive_on_fitting_domain": True, "ordered_classes": classes, "hard_masks_equal_raw_class_ids": True,
        "installed_package_API_and_binary_metadata_match_frozen_setup": True,
        "runtime_keys_compared": list(runtime_keys), "runtime_plan_sha256": retained["runtime_plan_sha256"],
        "actual_attempt_resource_receipt": {k: attempt[k] for k in ("elapsed_seconds", "peak_process_family_rss_bytes_sampled", "minimum_available_memory_bytes_sampled", "output_and_temporary_bytes", "exit_code")},
        "no_anatomical_host_growth_training_or_PVE_acceptance": True}


def failed_attempt(documents):
    """Metadata, byte hashes, and a native binary load-command header only."""
    assert sha(ROOT/FAILURE_OUTCOME) == FAILURE_OUTCOME_SHA
    outcome = json.loads((ROOT/FAILURE_OUTCOME).read_text())
    attempt = documents["attempt.json"]
    assert outcome["status"] == "failed_native_SIGSEGV_after_entering_N4"
    assert attempt["exit_code"] == outcome["worker_exit_code"] == -11
    assert outcome["attempts_started"] == 1 and outcome["automatic_retries"] == 0
    assert outcome["N4_entered"] is True and outcome["N4_completions"] == 0
    assert outcome["Atropos_entered"] is False and outcome["Atropos_completions"] == 0
    assert outcome["accepted_tissue_results"] == 0
    assert outcome["candidate_file_exists"] is False and not (BASE/"candidate.json").exists()
    assert not any(BASE.rglob("*.nii*"))
    assert outcome["no_supervisor_resource_limit_fired"] is True
    resources = attempt["resources"]
    assert attempt["elapsed_seconds"] == outcome["elapsed_seconds"] < resources["timeout_seconds"]
    assert attempt["peak_process_family_rss_bytes_sampled"] == outcome["peak_supervisor_plus_worker_family_rss_bytes_sampled"] < resources["rss_limit_bytes"]
    assert attempt["minimum_available_memory_bytes_sampled"] == outcome["minimum_available_memory_bytes_sampled"] > resources["minimum_available_bytes"]
    assert attempt["output_and_temporary_bytes"] == outcome["output_and_temporary_bytes_at_supervisor_exit"] < resources["output_limit_bytes"]
    output_names = {str(p.relative_to(BASE)) for p in BASE.rglob("*") if p.is_file()}
    assert output_names == {row["path"] for row in outcome["output_manifest"]}
    for row in outcome["output_manifest"]:
        assert sha(BASE/row["path"]) == row["sha256"] and (BASE/row["path"]).stat().st_size == row["bytes"]
    assert outcome["exact_N4_call"]["parameters"] == attempt["parameters"]["N4"]
    log = (BASE/"worker.log").read_text()
    assert "Running N4 for 3-dimensional images." in log
    assert "Atropos" not in log
    # Independently parse the sealed macOS crash report; do not copy host paths.
    crash = outcome["native_crash_report"]
    path = Path(crash["path"])
    assert sha(path) == crash["sha256"]
    decoder = json.JSONDecoder()
    raw = path.read_text()
    _, offset = decoder.raw_decode(raw)
    body, _ = decoder.raw_decode(raw[offset:].lstrip())
    assert body["pid"] == crash["worker_pid"] == 21276
    assert body["parentPid"] == crash["supervisor_pid"] == 21273
    assert body["exception"] == crash["exception"] and body["termination"] == crash["termination"]
    assert body["exception"]["signal"] == "SIGSEGV" and body["exception"]["type"] == "EXC_BAD_ACCESS"
    assert body["procLaunch"] == crash["launch"] and body["captureTime"] == crash["capture"]
    faulting = body["threads"][body["faultingThread"]]["frames"]
    assert faulting[0]["symbol"] == "_platform_memmove"
    native_frames = [frame for frame in faulting if body["usedImages"][frame["imageIndex"]].get("uuid") == crash["ANTs_binary_uuid"]]
    assert len(native_frames) == 6
    installed = ROOT/".tools/poc-antspyx-0.6.3-py312-v1/runtime/lib/python3.12/site-packages/ants/lib.cpython-312-darwin.so"
    with installed.open("rb") as source:
        header = source.read(32)
        words = struct.unpack("<8I", header)
        assert words[0] == 0xfeedfacf
        commands = source.read(words[5])
    position = 0; binary_uuid = None
    for _ in range(words[4]):
        command, size = struct.unpack_from("<II", commands, position)
        assert size >= 8
        if command == 0x1b:
            binary_uuid = str(uuid.UUID(bytes=commands[position+8:position+24]))
        position += size
    assert position == len(commands)
    assert binary_uuid == crash["ANTs_binary_uuid"] == "e52ccef8-46ab-3b61-9cea-c3218f5a31a8"
    retained = json.loads((ROOT/"artifacts/poc-ixi265-atropos-preparation-v1/installed-runtime-audit.json").read_text())
    assert outcome["runtime_binary_hashes"] == retained["installed_native_binary_hashes"]
    assert next(row["sha256"] for row in outcome["runtime_binary_hashes"] if row["path"] == "ants/lib.cpython-312-darwin.so") == crash["ANTs_binary_sha256"]
    assert all(row["unchanged"] and row["sha256"] == row["after_sha256"] == SOURCE[row["role"]][1] for row in outcome["source_continuity"])
    assert outcome["all_three_sources_unchanged"] and outcome["v1_marker_unchanged"]
    assert not outcome["anatomy_host_growth_acceptance"] and outcome["no_source_or_catalog_promotion"]
    assert sha(ROOT/FAILURE_OUTCOME) == FAILURE_OUTCOME_SHA
    return {"status": "failed_attempt_retained_no_candidate_acceptance", "sealed_outcome_sha256": FAILURE_OUTCOME_SHA,
        "failure": attempt["failure"], "support_and_native_frame_guards_passed": True,
        "guard_evidence": "Inference from pinned worker control flow: support construction and inherited input/native-frame guards precede the observed native N4 entry banner.",
        "N4_entered": True, "N4_completions": 0, "Atropos_entered": False, "Atropos_completions": 0,
        "phase_evidence": "Retained log ends at N4 entry, then the native worker exits SIGSEGV. Sealed outcome records no completed N4 and no Atropos entry; no saved candidate, classes or posteriors exist.",
        "worker_exit_code": -11, "candidate_exists": False, "source_arrays_decoded": False,
        "posteriors_claimed": False, "crash_report_sha256": crash["sha256"], "worker_pid": 21276, "supervisor_pid": 21273,
        "exception": {"type": "EXC_BAD_ACCESS", "signal": "SIGSEGV"},
        "faulting_top_symbol": "_platform_memmove", "matching_ANTs_native_frames": len(native_frames),
        "installed_MachO_UUID_independently_read": binary_uuid,
        "installed_binary_hash_retained_metadata": crash["ANTs_binary_sha256"],
        "installed_binary_header_bytes_read": 32+len(commands), "native_library_import_or_execution": False,
        "new_native_binary_hash_sweep": False, "no_supervisor_resource_limit_fired": True,
        "actual_attempt_resource_receipt": {k: attempt[k] for k in ("elapsed_seconds", "peak_process_family_rss_bytes_sampled", "minimum_available_memory_bytes_sampled", "output_and_temporary_bytes", "exit_code")},
        "prospective_45_checks_remain_valid_only_for_generated_contracts": True,
        "native_compatibility_not_proven_by_generated_contract_checks": True,
        "failure_cause": "Native memory-access failure established. Exact adapter/runtime cause remains unresolved by this crash evidence alone; any separate generated mask-dtype probe must be evaluated independently."}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--released-pins", type=Path)
    args = parser.parse_args()
    controls = generated_controls()
    if args.released_pins is None:
        print(json.dumps({"status": "prepared_generated_controls_pass", "controls": controls, "real_source_decode": False})); return
    started = time.monotonic()
    assert not DESTINATION.exists(), "Saved audit is immutable"
    pins = json.loads(args.released_pins.read_text())
    documents, records, frozen, markers = metadata_audit(pins)
    source_before = {name: {"path": path, **stamp(ROOT/path)} for name, (path, _) in SOURCE.items()}
    assert {name: row["sha256"] for name, row in source_before.items()} == {name: pin for name, (_, pin) in SOURCE.items()}
    memory("before_source_decode")
    if pins["status"] == "fitted_candidate_unaccepted":
        result = successful_candidate(documents)
    else:
        result = failed_attempt(documents)
    assert source_before == {name: {"path": path, **stamp(ROOT/path)} for name, (path, _) in SOURCE.items()}
    assert frozen == {name: stamp(ROOT/name) for name in FROZEN}
    assert records == {name: stamp(BASE/name) for name in records}
    assert not any(k.split(".")[0] in {"ants", "torch", "totalsegmentator", "monai"} for k in sys.modules)
    if pins["status"] == "failed": assert "nibabel" not in sys.modules
    memory("finished")
    report = {"schema": "ixi265-positive-domain-independent-result-audit-v2", "status": result["status"],
        "created_utc": datetime.now(timezone.utc).isoformat(), "result": result,
        "sources_unchanged": source_before, "host_bundle_hashed_only": True,
        "frozen_prior_evidence_unchanged": frozen, "new_attempt_metadata_unchanged": records,
        "single_attempt_markers": markers, "generated_controls": controls,
        "historical_v1": {"started": 1, "N4_completions": 0, "Atropos_completions": 0, "accepted_tissues": 0},
        "v2": {"started": 1, "completed_numerical_candidates": int(pins["status"] == "fitted_candidate_unaccepted"), "accepted_tissues": 0, "automatic_retries": 0},
        "combined_real_protocol_denominator": {"attempts_started": 2, "successful_candidates": int(pins["status"] == "fitted_candidate_unaccepted"), "accepted_tissues": 0, "automatic_retries": 0},
        "source_fix_clipping_normalization_or_mask_replacement": False,
        "model_or_solver_import_or_execution": False, "elapsed_seconds": time.monotonic()-started,
        "own_process_peak_rss_bytes": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss, "memory_samples": MEMORY,
        "auditor_sha256": sha(__file__), "release_pins_sha256": sha(args.released_pins),
        "limitations": ["This verifies saved numerical consistency, not anatomical correctness or biophysical validity.",
            "Intensity-ordered hard-class labels are unreviewed assumptions; posterior memberships are not physical partial-volume fractions.",
            "Unknown cells remain abstained and CSF-like intensity does not identify ventricles.",
            "No tissue-conditioned growth, host upgrade, training admission or clinical accuracy is accepted.",
            "The frozen API/native-binary and runtime metadata are compared without reimporting ANTs; no numerical fitter is rerun.",
            "The production RSS cap is sampled; between-sample peaks are not established."]}
    content = json.dumps(report, indent=2, allow_nan=False)+"\n"
    assert "/Users/" not in content
    DESTINATION.parent.mkdir(parents=True, exist_ok=True)
    with DESTINATION.open("x") as out: out.write(content)
    print(json.dumps({"status": report["status"], "artifact": str(DESTINATION.relative_to(ROOT)), "sha256": sha(DESTINATION), "peak_rss_bytes": report["own_process_peak_rss_bytes"]}))


if __name__ == "__main__":
    main()
