"""One guarded decode of the already saved Case4 DEVELOPMENT network output."""

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
MODEL = HERE.parent / "model_backend"
INPUT = MODEL / "case4-input"
AUDIT = ROOT / "build/scan-target-estimator-independent/case4-forward-v1-actual-audit.md"
START = time.monotonic()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def exact_file(path: Path, expected: str) -> None:
    if not path.is_file() or path.is_symlink() or sha256_file(path) != expected:
        raise ValueError("exact decode input/source changed: " + str(path))


def phase(name: str) -> None:
    print(json.dumps({"phase": name, "elapsed_seconds": time.monotonic() - START},
                     sort_keys=True), flush=True)


def main() -> None:
    phase("before_source_checks")
    contract = json.loads(CONTRACT.read_text())
    if (contract.get("gate") != "CASE4_DEVELOPMENT_DISPLAY_DECODE_V1" or
            contract.get("arm_order") != ["case4"] or
            os.environ.get("RESECTIONLAB_PAIR_ARM") != "case4" or
            contract.get("release_status") != "root_released_once" or
            contract.get("split_role") != "DEVELOPMENT" or
            contract.get("model_inference_performed_in_this_step") is not False or
            contract.get("planning_admitted") is not False or
            contract.get("evaluation_admitted") is not False or
            contract.get("clinical_evidence") is not False):
        raise ValueError("Case4 display decode has no exact one-shot release")
    exact_file(Path(__file__).resolve(), contract["runner_source_sha256"]["pair_worker.py"])
    source_paths = {
        "package_init": ROOT / "src/resectionlab/__init__.py",
        "diagnostic_runner": ROOT / "src/resectionlab/scan_diagnostic_runner.py",
        "preprocess_bridge": ROOT / "src/resectionlab/scan_preprocess_bridge.py",
        "scan_adapter": ROOT / "src/resectionlab/scan_target_adapter.py",
        "support_contract": ROOT / "src/resectionlab/scan_support_contract.py",
    }
    for name, path in source_paths.items():
        exact_file(path, contract["decode_source_sha256"][name])
    exact_file(ROOT / ".tools/scan-target-runtime/requirements-lock.txt",
               contract["runtime_lock_sha256"])
    audit = contract["independent_forward_audit"]
    if audit.get("path") != str(AUDIT.relative_to(ROOT)):
        raise ValueError("Case4 independent audit path changed")
    exact_file(AUDIT, audit["sha256"])
    expected_forward = contract["accepted_forward_sha256"]
    for name, relative in (
        ("contract", "pair-contract.json"),
        ("result", "case4/result.json"),
        ("supervision", "case4/supervision.json"),
        ("logits", "case4/logits.npy"),
        ("input_receipt", "case4-input/input-receipt.json"),
        ("input_npy", "case4-input/input.npy"),
    ):
        exact_file(MODEL / relative, expected_forward[name])
    if not OUT.is_dir() or (OUT / "display").exists() or (OUT / "result.json").exists():
        raise FileExistsError("Case4 display output is absent or already used")
    if os.environ.get("PYTHONNOUSERSITE") != "1":
        raise EnvironmentError("isolated user site must remain disabled")
    sys.path.insert(0, str(ROOT / "src"))
    from resectionlab.scan_diagnostic_runner import write_case4_display_from_saved

    phase("before_case4_display_decode")
    descriptor = write_case4_display_from_saved(ROOT, INPUT, MODEL, OUT / "display")
    phase("after_case4_display_decode")
    if (descriptor.get("schema") != "case4_unreviewed_diagnostic_display_layer_v1" or
            descriptor.get("model_output_verified") is not True or
            descriptor.get("planning_eligible") is not False or
            descriptor.get("evaluation_eligible") is not False or
            descriptor.get("clinical_evidence") is not False or
            descriptor.get("training_overlap_status") != "unknown" or
            descriptor.get("anatomical_qc") != "unreviewed_inferior_mask_omission"):
        raise ValueError("Case4 display artifact changed diagnostic-only scope")
    display = OUT / "display"
    for name in ("state", "coverage"):
        exact_file(display / descriptor["outputs"][name]["path"],
                   descriptor["outputs"][name]["sha256"])
    result = {
        "schema": "case4-development-display-decode-v1",
        "scope": "one_unreviewed_case4_diagnostic_layer_no_planning",
        "arm": "case4",
        "contract_sha256": sha256_file(CONTRACT),
        "forward_release_sha256": expected_forward["contract"],
        "forward_result_sha256": expected_forward["result"],
        "display_layer_sha256": sha256_file(display / "display-layer.json"),
        "state_sha256": descriptor["outputs"]["state"]["sha256"],
        "coverage_sha256": descriptor["outputs"]["coverage"]["sha256"],
        "predicted_voxels": descriptor["predicted_voxels"],
        "unknown_voxels": descriptor["unknown_voxels"],
        "model_inference_performed_in_this_step": False,
        "planning_admitted": False,
        "evaluation_admitted": False,
        "clinical_evidence": False,
        "anatomical_qc": "unreviewed_inferior_mask_omission",
        "training_overlap_status": "unknown",
    }
    with (OUT / "result.json").open("x") as stream:
        json.dump(result, stream, indent=2, sort_keys=True)
        stream.write("\n")
    phase("after_display_result")


if __name__ == "__main__":
    main()
