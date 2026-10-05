"""Audit frozen V1 baseline receipts only; never reopen original patient files."""
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import subprocess

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
RUN = ROOT / "artifacts/mechanics/resect-case4-baseline-alignment-v1"
OUTPUT = Path(__file__).with_name("saved-result-review.json")
FREEZE_SHA256 = "4395f951a35591c184f6f332f25422457598ade0cf8c69131e24a34d67f5a09e"


def content(path):
    path = Path(path).resolve()
    if path.is_relative_to(ROOT / "data"):
        raise RuntimeError("Original patient data are outside this audit's access scope")
    return path.read_bytes()


def sha(path):
    return hashlib.sha256(content(path)).hexdigest()


def load(path):
    return json.loads(content(path))


def transform(points, matrix):
    homogeneous = np.column_stack((points, np.ones(len(points))))
    return (np.asarray(matrix) @ homogeneous.T).T[:, :3]


def proper_fit(source, destination):
    """Independent reverse-covariance formulation of the fixed proper fit."""
    centered_source = source - source.mean(axis=0)
    centered_destination = destination - destination.mean(axis=0)
    for cloud in (centered_source, centered_destination):
        singular = np.linalg.svd(cloud, compute_uv=False)
        assert singular[1] / singular[0] > 1e-6
    left, _, right_t = np.linalg.svd(centered_destination.T @ centered_source)
    determinant_correction = np.eye(3)
    determinant_correction[-1, -1] = np.linalg.det(left @ right_t)
    rotation = left @ determinant_correction @ right_t
    result = np.eye(4)
    result[:3, :3] = rotation
    result[:3, 3] = destination.mean(axis=0) - rotation @ source.mean(axis=0)
    return result


def assert_close(actual, expected, *, tolerance=1e-10):
    np.testing.assert_allclose(actual, expected, rtol=0, atol=tolerance)
    return float(np.max(np.abs(np.asarray(actual) - np.asarray(expected))))


def main():
    assert sha(RUN / "failed-attempt-freeze.json") == FREEZE_SHA256
    freeze = load(RUN / "failed-attempt-freeze.json")
    for name, identity in freeze["files"].items():
        path = (RUN / name).resolve()
        assert path.is_relative_to(RUN)
        assert sha(path) == identity
    before_artifacts = {name: sha(RUN / name) for name in freeze["files"]}
    declaration = load(RUN / "declaration.json")
    release = load(RUN / "root-release.json")
    result = load(RUN / "alignment-receipt.json")
    resource = load(RUN / "resource-receipt.json")
    header = load(ROOT / declaration["baseline_header_receipt"])
    snapshot = load(ROOT / declaration["helper_snapshot_manifest"])
    assert result["source_hashes_before"] == result["source_hashes_after"]
    pinned = {str((RUN / name).relative_to(ROOT)): release[key] for name, key in (
        ("align.py", "script_sha256"), ("declaration.json", "declaration_sha256"),
        ("coordinate-justification.json", "coordinate_justification_sha256"))}
    pinned[declaration["baseline_header_receipt"]] = declaration["baseline_header_receipt_sha256"]
    pinned[declaration["helper_snapshot_manifest"]] = declaration["helper_snapshot_manifest_sha256"]
    pinned.update({item["path"]: item["sha256"] for item in snapshot["files"]})
    pinned.update({item["local_path"]: item["sha256"] for item in declaration["files"]})
    assert pinned == result["source_hashes_before"]
    patient_identities = {item["local_path"]: item["sha256"] for item in declaration["files"]}
    assert len(patient_identities) == 4
    assert not any("during" in path for path in patient_identities)
    for name, identity in pinned.items():
        if name not in patient_identities:
            assert sha(ROOT / name) == identity
    for name in ("align.py", "declaration.json", "coordinate-justification.json"):
        committed = subprocess.check_output(["git", "show", f"3247411:{(RUN / name).relative_to(ROOT)}"], cwd=ROOT)
        assert hashlib.sha256(committed).hexdigest() == sha(RUN / name)

    source = np.array(result["FLAIR_reference_RAS_mm"])
    destination = np.array(result["beforeUS_reference_RAS_mm"])
    assert source.shape == destination.shape == (15, 3)
    assert np.isfinite(source).all() and np.isfinite(destination).all()
    assert result["N"] == 15 and result["row_ids"] == list(range(1, 16))
    assert result["pair_role"] == "mri_to_before_us"
    assert_close(result["source_world_to_RAS"], np.eye(4), tolerance=0)
    assert_close(result["destination_world_to_RAS"], np.eye(4), tolerance=0)
    matrix = proper_fit(source, destination)
    transform_delta = assert_close(matrix, result["FLAIR_RAS_to_beforeUS_RAS_mm"])
    rotation = matrix[:3, :3]
    assert_close(rotation.T @ rotation, np.eye(3))
    assert abs(np.linalg.det(rotation) - 1) < 1e-10
    raw = np.linalg.norm(source - destination, axis=1)
    fitted = np.linalg.norm(transform(source, matrix) - destination, axis=1)
    loo = []
    for row in range(len(source)):
        keep = [i for i in range(len(source)) if i != row]
        fold = proper_fit(source[keep], destination[keep])
        loo.append(np.linalg.norm(transform(source[row:row + 1], fold)[0] - destination[row]))
    loo = np.array(loo)
    residual_deltas = {}
    summaries = {}
    for key, name, values in (("raw", "raw_distances_mm", raw), ("fit", "fit_residuals_mm", fitted),
                              ("leave_one_out", "leave_one_out_residuals_mm", loo)):
        residual_deltas[key] = assert_close(values, result[name])
        summary = dict(rms_mm=float(np.sqrt(np.mean(values ** 2))), mean_mm=float(values.mean()),
                       median_mm=float(np.median(values)), maximum_mm=float(values.max()))
        for metric, value in summary.items():
            assert_close(value, result["summaries"][key][metric])
        summaries[key] = summary
    assert summaries["fit"]["rms_mm"] <= summaries["raw"]["rms_mm"] + 1e-8
    assert result["LOO_RMS_worse_than_raw"] == (summaries["leave_one_out"]["rms_mm"] > summaries["raw"]["rms_mm"])

    headers = {Path(item["source_path"]).name: item["header"] for item in header["files"]}
    expected_planes = {}
    bound_deltas = {}
    for label, points, name in (("FLAIR_reference", source, "Case4-FLAIR.nii.gz"),
                                ("beforeUS_reference", destination, "Case4-US-before.nii.gz"),
                                ("MRI_reference_in_T1", source, "Case4-T1.nii.gz")):
        original = np.array(headers[name]["selected_affine"])
        shape = np.array(headers[name]["shape"])
        voxel = transform(points, np.linalg.inv(original))
        inside = ((voxel >= -.5) & (voxel <= shape - .5)).all(axis=1)
        assert result["bounds"][label]["inside_count"] == int(inside.sum()) == 15
        assert result["bounds"][label]["outside_row_ids"] == []
        bound_deltas[label] = assert_close(voxel, result["bounds"][label]["voxel_coordinates"])
        if label != "MRI_reference_in_T1":
            expected_planes[name] = np.clip(np.rint(np.median(voxel, axis=0)).astype(int), 0, shape - 1)
    assert len(result["plots"]) == 6
    expected_names = {f"{group}-axis{axis}.png" for group in ("MRI-native", "US-native") for axis in range(3)}
    assert {item["path"] for item in result["plots"]} == expected_names
    for item in result["plots"]:
        assert item["index"] == int(expected_planes[item["base_image"]][item["axis"]])
        assert sha(RUN / item["path"]) == item["sha256"] == freeze["files"][item["path"]]

    assert resource["status"] == "failed" and resource["failure"] == "RSS_MONITOR_FAILED"
    assert resource["exit_code"] == 0
    assert resource["result_sha256"] == sha(RUN / "alignment-receipt.json")
    assert resource["log_sha256"] == sha(RUN / "run.log")
    assert resource["script_sha256"] == sha(RUN / "align.py")
    assert resource["budget"] == declaration["budget"]
    assert resource["elapsed_seconds"] < declaration["budget"]["wall_seconds"]
    assert resource["maximum_sampled_parent_child_RSS_bytes"] < declaration["budget"]["RSS_bytes"]
    for key in ("outliers_removed", "alternative_fits_selected", "clinical_accuracy_validated",
                "brain_domain_available", "anatomical_alignment_accepted", "duringUS_or_before_during_tag_opened",
                "B_or_V_motion_accessed", "resampled_volumes_saved"):
        assert result[key] is False
    assert result["native_affines_preserved"] is True
    chronology = load(RUN / "preparation-history/chronology.json")
    for entry in chronology["files"]:
        original = subprocess.check_output(["git", "show", f"3247411:{(RUN / entry['name']).relative_to(ROOT)}"], cwd=ROOT)
        assert hashlib.sha256(original).hexdigest() == entry["committed_sha256"] == sha(ROOT / entry["committed_copy"])
        assert sha(RUN / entry["name"]) == entry["during_execution_sha256"]
    assert before_artifacts == {name: sha(RUN / name) for name in freeze["files"]}
    review = {
        "status": "saved_numerical_artifacts_consistent_supervised_attempt_failed",
        "created_at_utc": datetime.now(timezone.utc).isoformat(), "N_baseline_pairs": 15,
        "freeze_sha256": FREEZE_SHA256, "audit_script_sha256": sha(Path(__file__)),
        "fit_matrix_max_absolute_difference": transform_delta,
        "per_row_residual_max_absolute_differences_mm": residual_deltas,
        "source_grid_coordinate_max_absolute_differences_voxels": bound_deltas,
        "recomputed_summaries": summaries, "fixed_views_verified": 6,
        "frozen_artifact_count_verified": len(freeze["files"]),
        "committed_numerical_source_control_bytes_exact": True,
        "source_header_and_helper_files_rehashed": True,
        "input_hash_validation": "Saved before/after identities equal pinned declaration; no original patient files reopened",
        "original_affine_check": "Each saved voxel coordinate recomputed using original pinned header sform; original NIfTI bytes not reopened",
        "resource_status": "failed", "resource_failure": "RSS_MONITOR_FAILED",
        "worker_exit_code": 0, "elapsed_seconds": resource["elapsed_seconds"],
        "maximum_sampled_RSS_bytes": resource["maximum_sampled_parent_child_RSS_bytes"],
        "resource_failure_cause": "Unknown: V1 retained no raw ps result; no resource-pass claim",
        "preparation_chronology": "Committed16-test owner metadata copies verified against Git; pre-execution18-test refresh preserved separately",
        "scope_evidence": "Previously reviewed exact allowlist/source plus saved flags; no independent OS syscall trace",
        "original_patient_files_reopened": False, "closed_B_V_or_during_values_accessed": False,
        "image_arrays_recomputed": False, "image_pixels_independently_interpreted": False,
        "new_fit_or_tuning": False, "anatomical_or_clinical_acceptance": False,
        "limitations": "These are baseline calibration/LOO diagnostics, not sealed displacement V or mechanics accuracy. Entire supervised attempt remains failed.",
    }
    with OUTPUT.open("x") as stream:
        json.dump(review, stream, indent=2, allow_nan=False)
        stream.write("\n")
    print(json.dumps({"receipt": str(OUTPUT.relative_to(ROOT)), "sha256": sha(OUTPUT),
                      "status": review["status"], "RMS_mm": {k: v["rms_mm"] for k, v in summaries.items()}}))


if __name__ == "__main__":
    main()
