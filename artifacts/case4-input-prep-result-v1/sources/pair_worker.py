"""One gated Case4 DEVELOPMENT model-input preparation, no trained forward.

This file is an unreleased candidate. The supervising contract is absent until
root freezes exact source and input bindings for a single reviewed attempt.
"""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import sys
import time


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
CONTRACT = HERE / "pair-contract.json"
OUT = HERE / "case4"
START = time.monotonic()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def phase(name: str) -> None:
    print(json.dumps({"phase": name, "elapsed_seconds": time.monotonic() - START},
                     sort_keys=True), flush=True)


def exact_source(path: Path, expected: str) -> None:
    if (not path.is_file() or path.is_symlink() or
            sha256_file(path) != expected):
        raise ValueError("input preparation executing source changed: " + str(path))


def main() -> None:
    phase("before_source_checks")
    contract = json.loads(CONTRACT.read_text())
    if (contract.get("gate") != "CASE4_DEVELOPMENT_INPUT_PREP_V1" or
            contract.get("arm_order") != ["case4"] or
            os.environ.get("RESECTIONLAB_PAIR_ARM") != "case4" or
            contract.get("release_status") != "root_released_once" or
            contract.get("patient_input_preparation_admitted") is not True or
            contract.get("model_training_or_patient_inference") is not False or
            contract.get("planning_admitted") is not False or
            contract.get("clinical_evidence") is not False or
            contract.get("split_role") != "DEVELOPMENT"):
        raise ValueError("Case4 input preparation has no exact one-shot release")
    exact_source(Path(__file__).resolve(),
                 contract["runner_source_sha256"]["pair_worker.py"])
    source_paths = {
        "package_init": ROOT / "src/resectionlab/__init__.py",
        "diagnostic_runner": ROOT / "src/resectionlab/scan_diagnostic_runner.py",
        "preprocess_bridge": ROOT / "src/resectionlab/scan_preprocess_bridge.py",
        "scan_adapter": ROOT / "src/resectionlab/scan_target_adapter.py",
        "support_contract": ROOT / "src/resectionlab/scan_support_contract.py",
    }
    for name, path in source_paths.items():
        exact_source(path, contract["preparation_source_sha256"][name])
    exact_source(ROOT / ".tools/scan-target-runtime/requirements-lock.txt",
                 contract["runtime_lock_sha256"])
    if not OUT.is_dir() or (OUT / "input").exists() or (OUT / "result.json").exists():
        raise FileExistsError("Case4 input preparation output is absent or already used")
    if os.environ.get("PYTHONNOUSERSITE") != "1":
        raise EnvironmentError("isolated user site must remain disabled")
    sys.path.insert(0, str(ROOT / "src"))
    from resectionlab.scan_diagnostic_runner import write_case4_model_input

    phase("before_case4_input_preparation")
    receipt = write_case4_model_input(ROOT, OUT / "input")
    phase("after_case4_input_preparation")
    saved = OUT / "input/input-receipt.json"
    tensor = OUT / "input/input.npy"
    if (receipt.get("schema") != "case4_unreviewed_diagnostic_input_v1" or
            receipt.get("source_origin") != "verified_case4_source_receipts" or
            receipt.get("model_forward_performed") is not False or
            receipt.get("planning_eligible") is not False or
            sha256_file(tensor) != receipt.get("input_npy_sha256")):
        raise ValueError("Case4 input receipt or tensor changed after preparation")
    result = {
        "schema": "case4-development-input-preparation-v1",
        "scope": "one_unreviewed_scan_only_model_input_no_forward",
        "arm": "case4",
        "contract_sha256": sha256_file(CONTRACT),
        "input_receipt_sha256": sha256_file(saved),
        "input_npy_sha256": receipt["input_npy_sha256"],
        "input_raw_sha256": receipt["input_raw_sha256"],
        "source_sha256": receipt["source_sha256"],
        "geometry_sha256": receipt["geometry_sha256"],
        "evidence_sha256": receipt["evidence_sha256"],
        "input_shape": receipt["input_shape"],
        "model_inference_performed": False,
        "planning_admitted": False,
        "evaluation_admitted": False,
        "clinical_evidence": False,
        "anatomical_qc": "unreviewed_inferior_mask_omission",
        "training_overlap_status": "unknown",
    }
    with (OUT / "result.json").open("x") as stream:
        json.dump(result, stream, indent=2, sort_keys=True)
        stream.write("\n")
    phase("after_input_result")


if __name__ == "__main__":
    main()
