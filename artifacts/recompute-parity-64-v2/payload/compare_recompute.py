"""Saved-only generated 64-cube historical anchor and matched-arm parity."""

import hashlib
import json
from pathlib import Path
import sys

import numpy as np

from supervise_pair import validate_saved_baseline, validate_matched_baseline


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
CONTRACT = HERE / "pair-contract.json"


def sha256_file(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _array(path, expected_sha, shape, raw_sha):
    if sha256_file(path) != expected_sha:
        raise ValueError("saved logit file changed")
    values = np.load(path, allow_pickle=False)
    if (values.dtype != np.float32 or list(values.shape) != shape or
            not values.flags.c_contiguous or not np.isfinite(values).all() or
            hashlib.sha256(values.tobytes(order="C")).hexdigest() != raw_sha):
        raise ValueError("saved logit content is invalid")
    return values


def labels(logits):
    decoded = np.zeros(logits.shape[2:], dtype=np.uint8)
    for region, label in enumerate((2, 1, 3)):
        decoded[logits[0, region] > 0] = label
    return decoded


def numerical_metrics(candidate, baseline, atol, rtol):
    difference = np.abs(candidate - baseline)
    tolerance = atol + rtol * np.abs(baseline)
    return {
        "all_logits_within_predeclared_tolerance": bool(np.all(difference <= tolerance)),
        "exact_raw_logit_equality": bool(np.array_equal(candidate, baseline)),
        "max_absolute_logit_error": float(difference.max()),
        "max_tolerance_excess": float(np.maximum(difference - tolerance, 0).max()),
        "decoded_label_disagreements": int(np.count_nonzero(labels(candidate) != labels(baseline))),
        "per_region_sign_disagreements": [int(np.count_nonzero(
            (candidate[0, i] > 0) != (baseline[0, i] > 0))) for i in range(3)],
        "per_region_near_zero_counts_candidate": [int(np.count_nonzero(
            np.abs(candidate[0, i]) <= atol)) for i in range(3)],
        "per_region_near_zero_counts_baseline": [int(np.count_nonzero(
            np.abs(baseline[0, i]) <= atol)) for i in range(3)],
        "per_region_min_abs_margin_candidate": [float(np.abs(candidate[0, i]).min()) for i in range(3)],
        "per_region_min_abs_margin_baseline": [float(np.abs(baseline[0, i]).min()) for i in range(3)],
    }


def arm_array(contract, arm, expected_recompute):
    folder = HERE / arm
    supervision = json.loads((folder / "supervision.json").read_text())
    result = json.loads((folder / "result.json").read_text())
    contract_sha = sha256_file(CONTRACT)
    report = result.get("recompute_attachment_report")
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
            result.get("input_shape") != contract["input_shape"] or
            result.get("output_shape") != contract["output_shape"] or
            result.get("output_nonfinite") != 0 or
            result.get("safe_globals_restored") is not True or
            result.get("dense_module_hooks_attached") is not False or
            result.get("stage_zero_recomputation_enabled") is not expected_recompute or
            result.get("full_feature_map_instancenorm_unchanged") is not True or
            result.get("dropout_configured") is not False or
            result.get("single_generated_volume_no_sliding_window_tta_autocast") is not True or
            result.get("alias_groups") != 88 or result.get("alias_compared_pairs") != 184 or
            result.get("wrapper_report", {}).get("wrapped_unique_conv3d_modules") != 27 or
            result.get("wrapper_report", {}).get(
                "parameter_state_module_identities_preserved") is not True or
            result.get("lazy_attachment_report", {}).get(
                "parameter_state_module_identities_preserved") is not True or
            result.get("lazy_attachment_report", {}).get("final_decoder_stages") != 5 or
            result.get("recompute_source_sha256") !=
            contract["runner_source_sha256"]["recompute_skip.py"] or
            result.get("tiled_adapter_sha256") != contract["tracked_adapter_sha256"] or
            result.get("lazy_concat_source_sha256") !=
            contract["runner_source_sha256"]["lazy_concat.py"] or
            result.get("reference_baseline_contract_sha256") !=
            contract["saved_baseline_reference"]["contract_sha256"]):
        raise ValueError(arm + " arm lacks accepted exact-source result")
    if expected_recompute:
        if (not isinstance(report, dict) or
                report.get("state_parameter_module_identities_preserved") is not True or
                report.get("initial_stage_zero_released_after_stage_one") is not True or
                report.get("stage_zero_recomputed_once_on_decoder_access") is not True):
            raise ValueError("recompute attachment report missing")
    elif report is not None:
        raise ValueError("baseline unexpectedly attached recomputation")
    values = _array(folder / "logits.npy", result["logits_npy_sha256"],
                    contract["output_shape"], result["logits_raw_sha256"])
    return values, result, supervision


def main(mode):
    contract = json.loads(CONTRACT.read_text())
    if (contract["gate"] != "CPU_RECOMPUTE_STAGE0_PARITY_64_V2" or
            contract["arm_order"] != ["baseline", "recompute"] or
            contract["region_order"] != ["WT", "TC", "ET"] or
            contract["regions_class_order"] != [2, 1, 3] or
            sha256_file(Path(__file__).resolve()) !=
            contract["runner_source_sha256"]["compare_recompute.py"]):
        raise ValueError("comparison gate, region order or source changed")
    for name, expected in contract["runner_source_sha256"].items():
        if sha256_file(HERE / name) != expected:
            raise ValueError("comparison source changed: " + name)
    validate_saved_baseline(contract)
    reference = contract["saved_baseline_reference"]
    historical = _array(ROOT / reference["logits_path"], reference["logits_sha256"],
                        contract["output_shape"], reference["logits_raw_sha256"])
    matched_base, baseline_result, baseline_supervision = arm_array(contract, "baseline", False)
    if mode == "baseline":
        comparator = historical
        candidate = matched_base
        path = HERE / "baseline-comparison.json"
    elif mode == "pair":
        validate_matched_baseline(contract)
        comparator = matched_base
        candidate, result, supervision = arm_array(contract, "recompute", True)
        path = HERE / "comparison.json"
    else:
        raise ValueError("require baseline or pair comparison mode")
    metrics = numerical_metrics(candidate, comparator,
                                contract["parity_atol"], contract["parity_rtol"])
    accepted = (metrics["all_logits_within_predeclared_tolerance"] and
                metrics["decoded_label_disagreements"] == 0)
    report = {
        "scope": ("Saved generated 64-cube matched hook-free baseline versus historical hooked reference"
                  if mode == "baseline" else
                  "Saved generated 64-cube hook-free recomputation versus matched hook-free baseline"),
        "mode": mode, "gate_pass": bool(accepted),
        "contract_sha256": sha256_file(CONTRACT),
        "baseline_logits_npy_sha256": baseline_result["logits_npy_sha256"],
        "baseline_logits_raw_sha256": baseline_result["logits_raw_sha256"],
        "numerical_rule": {"atol": contract["parity_atol"], "rtol": contract["parity_rtol"],
                           "exact_decoded_labels": True},
        "metrics": metrics,
    }
    if mode == "baseline":
        report["historical_logits_npy_sha256"] = reference["logits_sha256"]
        report["historical_hook_observed_memory_not_comparable"] = True
    else:
        report["recompute_logits_npy_sha256"] = result["logits_npy_sha256"]
        report["sampled_peak_process_group_resident_bytes"] = {
            "baseline": baseline_supervision["sampled_peak_process_group_resident_bytes"],
            "recompute": supervision["sampled_peak_process_group_resident_bytes"]}
        report["forward_seconds_single_runs"] = {
            "baseline": baseline_result["forward_seconds"],
            "recompute": result["forward_seconds"]}
        report["memory_and_time_interpretation"] = (
            "Matched instrumentation and inputs, but sequential single runs on a shared host; "
            "exploratory sampled RSS and timing only, not a robust speed or memory estimate")
    with path.open("x") as stream:
        json.dump(report, stream, indent=2, sort_keys=True)
        stream.write("\n")
    print(json.dumps({"gate_pass": report["gate_pass"],
                      "max_absolute_logit_error": metrics["max_absolute_logit_error"],
                      "decoded_label_disagreements": metrics["decoded_label_disagreements"]},
                     sort_keys=True))
    if not accepted:
        raise SystemExit(1)


if __name__ == "__main__":
    if len(sys.argv) != 2 or sys.argv[1] not in ("baseline", "pair"):
        raise SystemExit("require baseline or pair saved-only mode")
    main(sys.argv[1])
