"""Compare two saved generated-input FP32 logit arrays without model execution."""

import hashlib
import json
from pathlib import Path

import numpy as np


HERE = Path(__file__).resolve().parent
CONTRACT = HERE / "gate-a-contract.json"
OUTPUT = HERE / "gate-a-comparison.json"


def digest(path):
    sha = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            sha.update(block)
    return sha.hexdigest()


def load_checked(device, contract):
    folder = HERE / device
    receipt = json.loads((folder / "result.json").read_text())
    supervision = json.loads((folder / "supervision.json").read_text())
    path = folder / "logits.npy"
    if receipt["device"] != device or receipt["input_npy_sha256"] != contract["input_npy_sha256"]:
        raise ValueError("device/input receipt mismatch")
    if (supervision["exit_code"] != 0 or supervision["watchdog_reason"] is not None
            or supervision["accepted_for_pair"] is not True):
        raise ValueError("device run did not complete under guard")
    if digest(path) != receipt["logits_npy_sha256"]:
        raise ValueError("logit file differs from receipt")
    logits = np.load(path, allow_pickle=False)
    if logits.shape != (1, 3, 32, 64, 64) or logits.dtype != np.float32 or not logits.flags.c_contiguous:
        raise ValueError("unexpected logit shape/dtype/layout")
    if hashlib.sha256(logits.tobytes(order="C")).hexdigest() != receipt["logits_raw_sha256"]:
        raise ValueError("raw logit bytes differ from receipt")
    if not np.isfinite(logits).all():
        raise ValueError("nonfinite logits")
    return logits, receipt


def decode_regions(logits, order):
    segmentation = np.zeros((logits.shape[0], *logits.shape[2:]), dtype=np.uint8)
    for index, label in enumerate(order):
        segmentation[logits[:, index] > 0.0] = label
    return segmentation


def main():
    if OUTPUT.exists():
        raise FileExistsError("comparison receipt already exists")
    contract = json.loads(CONTRACT.read_text())
    if contract["gate"] != "A" or contract["full_128_patch_gate_b_enabled"]:
        raise ValueError("only Gate A comparison is permitted")
    cpu, cpu_receipt = load_checked("cpu", contract)
    mps, mps_receipt = load_checked("mps", contract)
    absolute = np.abs(cpu.astype(np.float64) - mps.astype(np.float64))
    allclose = bool(np.allclose(cpu, mps, rtol=contract["parity_rtol"],
                               atol=contract["parity_atol"], equal_nan=False))
    tolerance = contract["parity_atol"] + contract["parity_rtol"] * np.abs(mps.astype(np.float64))
    out_of_tolerance = int(np.count_nonzero(absolute > tolerance))
    order = [2, 1, 3]  # pinned dataset.json regions_class_order; sequential overwrite
    if order != json.loads((HERE.parents[2] / "data/models/gliomoda-v1.0.2/t1c-t2f-pinned-v1/dataset.json").read_text())["regions_class_order"]:
        raise ValueError("region decode order changed")
    decoded_disagreement = int(np.count_nonzero(decode_regions(cpu, order) != decode_regions(mps, order)))
    decoded_total = int(np.prod(cpu.shape[2:]) * cpu.shape[0])
    threshold_margin = []
    near_zero_band = contract["parity_atol"]
    for index, region_name in enumerate(("whole tumor", "tumor core", "enhancing tumor")):
        cpu_region = cpu[:, index]
        mps_region = mps[:, index]
        crossing = (cpu_region > 0.0) != (mps_region > 0.0)
        crossings = int(np.count_nonzero(crossing))
        threshold_margin.append({
            "region": region_name,
            "near_zero_band_abs_logit": near_zero_band,
            "cpu_near_zero_cells": int(np.count_nonzero(np.abs(cpu_region) <= near_zero_band)),
            "mps_near_zero_cells": int(np.count_nonzero(np.abs(mps_region) <= near_zero_band)),
            "threshold_crossing_cells": crossings,
            "minimum_cpu_abs_logit": float(np.min(np.abs(cpu_region))),
            "minimum_mps_abs_logit": float(np.min(np.abs(mps_region))),
            "maximum_abs_logit_difference_on_crossings": float(np.max(np.abs(cpu_region[crossing] - mps_region[crossing]))) if crossings else None,
            "maximum_distance_to_zero_on_crossings": float(np.max(np.maximum(np.abs(cpu_region[crossing]), np.abs(mps_region[crossing])))) if crossings else None,
        })
    passed = allclose and (decoded_disagreement == 0 if contract["require_exact_decoded_region_labels"] else True)
    report = {
        "scope": "Generated-only 32x64x64 CPU/MPS logit parity; no patient or 128^3 inference",
        "gate_a_pass": passed,
        "raw_logit_allclose": allclose,
        "rtol": contract["parity_rtol"], "atol": contract["parity_atol"],
        "max_absolute_error": float(np.max(absolute)),
        "rmse": float(np.sqrt(np.mean(np.square(absolute)))),
        "out_of_tolerance_elements": out_of_tolerance,
        "total_logit_elements": int(cpu.size),
        "decoded_disagreement_cells": decoded_disagreement,
        "decoded_total_cells": decoded_total,
        "decoded_disagreement_fraction": decoded_disagreement / decoded_total,
        "per_region_threshold_margins": threshold_margin,
        "region_order": ["whole tumor", "tumor core", "enhancing tumor"],
        "regions_class_order": order,
        "cpu_raw_logit_sha256": cpu_receipt["logits_raw_sha256"],
        "mps_raw_logit_sha256": mps_receipt["logits_raw_sha256"],
        "gate_b_128_patch_enabled": False,
        "interpretation": "Numerical backend parity on generated input only; not segmentation accuracy, patient transfer, or clinical evidence",
    }
    OUTPUT.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report, sort_keys=True))


if __name__ == "__main__":
    main()
