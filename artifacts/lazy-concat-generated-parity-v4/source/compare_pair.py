"""Saved-only generated 64-cubed matched lazy-concat comparison."""
import hashlib
import json
from pathlib import Path

import numpy as np


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
CONTRACT = HERE / "pair-contract.json"


def sha256_file(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def accepted_arm(arm, contract, contract_sha):
    folder = HERE / arm
    supervision = json.loads((folder / "supervision.json").read_text())
    result = json.loads((folder / "result.json").read_text())
    path = folder / "logits.npy"
    if (supervision.get("arm") != arm or
            supervision.get("accepted_for_feasibility") is not True or
            supervision.get("exit_code") != 0 or
            supervision.get("manual_attention_required") is True or
            supervision.get("residual_possible") is True or
            supervision.get("watchdog_reason") is not None or
            supervision.get("post_guard_reason") is not None or
            supervision.get("contract_sha256") != contract_sha or
            result.get("arm") != arm or
            result.get("contract_sha256") != contract_sha or
            result.get("checkpoint_sha256_after") !=
            contract["model_file_sha256"]["fold_all/checkpoint_final.pth"] or
            result.get("input_npy_sha256") != contract["input_npy_sha256"] or
            result.get("output_shape") != contract["output_shape"] or
            result.get("input_shape") != contract["input_shape"] or
            result.get("output_nonfinite") != 0 or
            result.get("safe_globals_restored") is not True or
            result.get("full_feature_map_instancenorm_unchanged") is not True or
            result.get("dropout_configured") is not False or
            result.get("single_generated_volume_no_sliding_window_tta_autocast") is not True or
            result.get("alias_groups") != 88 or
            result.get("alias_compared_pairs") != 184 or
            result.get("wrapper_report", {}).get("wrapped_unique_conv3d_modules") != 27 or
            result.get("wrapper_report", {}).get(
                "parameter_state_module_identities_preserved") is not True or
            result.get("tiled_adapter_sha256") != contract["tracked_adapter_sha256"] or
            result.get("lazy_concat_source_sha256") !=
            contract["runner_source_sha256"]["lazy_concat.py"] or
            result.get("observed_module_counts") != contract["phase_probe_expected_counts"] or
            result.get("observer_samples") != contract["expected_observer_samples"] or
            result.get("phase_rss_source") != "direct_libproc_self_pid_no_subprocess" or
            (arm == "baseline" and result.get("lazy_attachment_report") is not None) or
            (arm == "lazy" and (
                result.get("lazy_attachment_report", {}).get(
                    "parameter_state_module_identities_preserved") is not True or
                result.get("lazy_attachment_report", {}).get(
                    "final_decoder_stages") != 5 or
                result.get("lazy_attachment_report", {}).get(
                    "first_conv_depth_tile") != 1)) or
            result.get("logits_npy_sha256") != sha256_file(path)):
        raise ValueError("arm lacks accepted exact-source output: " + arm)
    array = np.load(path, allow_pickle=False)
    if (array.dtype != np.float32 or list(array.shape) != contract["output_shape"] or
            not array.flags.c_contiguous or not np.isfinite(array).all() or
            result.get("logits_raw_sha256") != hashlib.sha256(
                array.tobytes(order="C")).hexdigest()):
        raise ValueError("arm output bytes/shape/finiteness changed: " + arm)
    return array, result, supervision


def labels(logits):
    # Pinned nnU-Net region order is WT, TC, ET with sequential [2, 1, 3]
    # overwrite. Sigmoid > 0.5 has the same sign threshold as raw logits > 0.
    decoded = np.zeros(logits.shape[2:], dtype=np.uint8)
    for region, label in enumerate((2, 1, 3)):
        decoded[logits[0, region] > 0] = label
    return decoded


def pair_metrics(left, right, atol, rtol):
    difference = np.abs(left - right)
    tolerance = atol + rtol * np.abs(right)
    left_labels, right_labels = labels(left), labels(right)
    return {
        "all_logits_within_predeclared_tolerance": bool(np.all(difference <= tolerance)),
        "exact_raw_logit_equality": bool(np.array_equal(left, right)),
        "max_absolute_logit_error": float(difference.max()),
        "max_tolerance_excess": float(np.maximum(difference - tolerance, 0).max()),
        "decoded_label_disagreements": int(np.count_nonzero(left_labels != right_labels)),
        "per_region_sign_disagreements": [int(np.count_nonzero(
            (left[0, i] > 0) != (right[0, i] > 0))) for i in range(3)],
        "per_region_near_zero_counts_left": [int(np.count_nonzero(
            np.abs(left[0, i]) <= atol)) for i in range(3)],
        "per_region_near_zero_counts_right": [int(np.count_nonzero(
            np.abs(right[0, i]) <= atol)) for i in range(3)],
        "per_region_min_abs_margin_left": [float(np.abs(left[0, i]).min()) for i in range(3)],
        "per_region_min_abs_margin_right": [float(np.abs(right[0, i]).min()) for i in range(3)],
    }


def main():
    contract = json.loads(CONTRACT.read_text())
    if (contract["gate"] != "CPU_TILED_LAZY_CONCAT_PARITY_64_V4" or
            contract["arm_order"] != ["baseline", "lazy"] or
            contract["region_order"] != ["WT", "TC", "ET"] or
            contract["regions_class_order"] != [2, 1, 3] or
            sha256_file(Path(__file__).resolve()) !=
            contract["runner_source_sha256"]["compare_pair.py"]):
        raise ValueError("comparison contract or source changed")
    for name, expected in contract["runner_source_sha256"].items():
        if sha256_file(HERE / name) != expected:
            raise ValueError("comparison runner source changed: " + name)
    contract_sha = sha256_file(CONTRACT)
    baseline, base_result, base_supervision = accepted_arm("baseline", contract, contract_sha)
    lazy, lazy_result, lazy_supervision = accepted_arm("lazy", contract, contract_sha)
    reference = contract["saved_tile1_reference"]
    reference_path = ROOT / reference["logits_path"]
    if (sha256_file(reference_path) != reference["logits_npy_sha256"] or
            sha256_file(ROOT / reference["result_path"]) != reference["result_sha256"] or
            sha256_file(ROOT / reference["supervision_path"]) != reference["supervision_sha256"]):
        raise ValueError("historical saved tile1 reference changed")
    historical = np.load(reference_path, allow_pickle=False)
    if (historical.dtype != np.float32 or list(historical.shape) != contract["output_shape"] or
            not historical.flags.c_contiguous or not np.isfinite(historical).all()):
        raise ValueError("historical output format changed")
    tolerance = {"atol": contract["parity_atol"], "rtol": contract["parity_rtol"]}
    matched = pair_metrics(lazy, baseline, **tolerance)
    historical_match = pair_metrics(baseline, historical, **tolerance)
    accepted = (matched["all_logits_within_predeclared_tolerance"] and
                matched["decoded_label_disagreements"] == 0 and
                historical_match["all_logits_within_predeclared_tolerance"] and
                historical_match["decoded_label_disagreements"] == 0)
    report = {
        "scope": "Saved generated-only 64-cubed instrumented comparison; no 128-cubed or patient inference claim",
        "gate_pass": bool(accepted), "contract_sha256": contract_sha,
        "numerical_rule": tolerance,
        "matched_lazy_vs_baseline": matched,
        "baseline_vs_saved_tile1_reference": historical_match,
        "sampled_peak_resident_bytes": {
            "baseline": base_supervision["sampled_peak_process_group_resident_bytes"],
            "lazy": lazy_supervision["sampled_peak_process_group_resident_bytes"]},
        "forward_seconds": {"baseline": base_result["forward_seconds"],
                            "lazy": lazy_result["forward_seconds"]},
        "peak_difference_bytes_baseline_minus_lazy": (
            base_supervision["sampled_peak_process_group_resident_bytes"] -
            lazy_supervision["sampled_peak_process_group_resident_bytes"]),
        "single_run_host_variability_limit": True,
    }
    with (HERE / "comparison.json").open("x") as stream:
        json.dump(report, stream, indent=2, sort_keys=True)
        stream.write("\n")
    print(json.dumps({"gate_pass": report["gate_pass"],
                      "peak_difference_bytes_baseline_minus_lazy":
                      report["peak_difference_bytes_baseline_minus_lazy"]}, sort_keys=True))
    if not report["gate_pass"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
