"""Validate the completion evidence and exact native masks of a CT worker."""

from hashlib import file_digest
import json
import math
from pathlib import Path

import nibabel as nib
import numpy as np

from .poc_worker_supervision import MEMORY_LIMIT_BYTES

CT_MASKS = ("brain_mask.nii.gz", "skull_mask.nii.gz")
PIPE_FAILURE = "worker_output_not_closed"
PENDING_VALIDATION = "worker_output_pending_validation"


def _sha(path):
    with path.open("rb") as stream:
        return file_digest(stream, "sha256").hexdigest()


def _read_evidence(path):
    if path.is_symlink() or not path.is_file() or path.stat().st_size > 1024**2:
        raise ValueError(f"Missing or invalid worker evidence: {path.name}")
    payload = path.read_bytes()
    if len(payload) > 1024**2:
        raise ValueError(f"Worker evidence exceeds its size budget: {path.name}")
    record = json.loads(payload)
    if not isinstance(record, dict):
        raise ValueError(f"Invalid worker evidence: {path.name}")
    return record, payload


def _positive_number(value):
    return type(value) in (int, float) and math.isfinite(value) and value > 0


def _worker_is_complete(worker, elapsed):
    return (_positive_number(elapsed) and _positive_number(worker.get("compute_seconds"))
            and worker["compute_seconds"] <= elapsed
            and isinstance(worker.get("runtime_versions"), dict))


def ct_masks(output, reference, *, require_nonempty):
    """Require both worker artifacts with values and geometry on the input grid."""
    results = []
    for name in CT_MASKS:
        path = output / name
        if path.is_symlink() or not path.is_file():
            raise ValueError(f"Missing completed worker artifact {name}")
        checksum = _sha(path)
        mask = nib.load(path)
        if (mask.shape != reference.shape or mask.header.get_xyzt_units()[0] != reference.header.get_xyzt_units()[0]
                or not np.allclose(mask.affine, reference.affine, atol=1e-5, rtol=0)):
            raise ValueError(f"Native geometry mismatch: {name}")
        for form in ("qform", "sform"):
            actual, actual_code = getattr(mask, "get_" + form)(coded=True)
            expected, expected_code = getattr(reference, "get_" + form)(coded=True)
            if actual_code != expected_code or (actual_code and
                    not np.allclose(actual, expected, atol=1e-5, rtol=0)):
                raise ValueError(f"Native {form} mismatch: {name}")
        data = np.asanyarray(mask.dataobj)
        foreground = int(np.count_nonzero(data))
        if (require_nonempty and foreground == 0) or not np.isin(data, (0, 1)).all():
            raise ValueError(f"Invalid or empty CT worker mask: {name}")
        if _sha(path) != checksum:
            raise ValueError(f"Worker artifact changed during validation: {name}")
        results.append({"path": str(path), "name": name, "sha256": checksum, "shape": list(mask.shape),
                        "affine_ras_mm": mask.affine.tolist(), "foreground_voxels": foreground,
                        "provenance": "model_inference_unreviewed"})
    return results


def ct_worker_completion(output, elapsed):
    worker, _ = _read_evidence(output / "worker.json")
    if not _worker_is_complete(worker, elapsed):
        raise ValueError("CT worker completion evidence is invalid")
    return worker


def recover_ct_worker(receipt, *, cache_root, image, source_hash, modality, model,
                      expected_identity=None, expected_weights=None, weight_root=None):
    """Read and validate a zero-exit attempt without modifying the original receipt."""
    from .anatomy_model_identity import retained_ct_runner

    root = Path(cache_root).resolve()
    receipt = Path(receipt)
    if receipt.name != "inference.json" or receipt.is_symlink():
        raise ValueError("Expected an existing CT worker attempt receipt")
    receipt = receipt.resolve()
    output = receipt.parent
    if not output.is_relative_to(root) or len(output.relative_to(root).parts) != 3:
        raise ValueError("CT worker receipt is outside the guarded inference cache")
    try:
        record, receipt_bytes = _read_evidence(receipt)
        request, request_bytes = _read_evidence(output / "request.json")
        worker, worker_bytes = _read_evidence(output / "worker.json")
        source, identity, weights = record["input"], record["identity"], record["weights"]
        case, identifier, attempt = output.relative_to(root).parts
        key = record.get("cache_key")
        reference = nib.load(image)
        if (record.get("schema") != "resectionlab-poc-inference-v1"
                or (record.get("status"), record.get("failure")) not in {
                    ("failed", PIPE_FAILURE), ("pending_validation", PENDING_VALIDATION)}
                or type(record.get("exit_code")) is not int or record["exit_code"] != 0
                or modality not in {"CT", "CTA"} or identifier != "totalsegmentator-skull"
                or record["model_id"] != identifier or request["model_id"] != identifier
                or record["case"] != case or request["case"] != case
                or not isinstance(key, str) or not (attempt == key or attempt.startswith(key + "-retry-"))
                or record["device"] not in {"mps", "cpu"} or request["device"] != record["device"]
                or record["model"] != model or request["parameters"] != model["parameters"]
                or request["modality"] != modality or source["modality"] != modality
                or request["image"] != source["path"] or not Path(source["path"]).is_absolute()
                or request["output"] != str(output) or source["sha256"] != source_hash
                or source["shape"] != list(reference.shape)
                or not np.allclose(source["affine_ras_mm"], reference.affine, atol=1e-5, rtol=0)
                or len(weights) != 1 or request["weights"] != weights
                or record["worker"] != worker or not _worker_is_complete(worker, record["elapsed_seconds"])
                or not _positive_number(record["peak_process_family_rss_bytes_sampled"])
                or record["peak_process_family_rss_bytes_sampled"] > MEMORY_LIMIT_BYTES
                or not retained_ct_runner(record["launcher_sha256"], identity["runner_sha256"])):
            raise ValueError("CT completion, model configuration or acquired source binding differs")
        if identity != {"image_sha256": source_hash, "weights": [weights[0]["sha256"]],
                        "model_id": identifier, "parameters": model["parameters"],
                        "device": record["device"], "runner_sha256": identity["runner_sha256"]}:
            raise ValueError("CT inference identity does not match the recorded inputs")
        if expected_identity is not None and identity != expected_identity:
            raise ValueError("CT inference identity differs from the current request")
        if expected_weights is not None:
            if [(w["sha256"], w["bytes"]) for w in weights] != [(w["sha256"], w["bytes"]) for w in expected_weights]:
                raise ValueError("Retained checkpoint differs from the installed weights")
        else:
            weight = Path(weights[0]["path"]).resolve()
            if (weight_root is None or not weight.is_relative_to(Path(weight_root).resolve())
                    or weight.stat().st_size != weights[0]["bytes"] or _sha(weight) != weights[0]["sha256"]):
                raise ValueError("Retained checkpoint differs from its verified model cache")
        # Clean completion checkpoints can legitimately contain empty masks.
        # Historical failures need the stronger positive-output proof.
        outputs = ct_masks(output, reference, require_nonempty=record["failure"] == PIPE_FAILURE)
    except (KeyError, TypeError, IndexError, OSError, nib.filebasedimages.ImageFileError) as error:
        raise ValueError("CT worker evidence or native mask could not be read") from error
    if (receipt.read_bytes() != receipt_bytes or (output / "worker.json").read_bytes() != worker_bytes
            or (output / "request.json").read_bytes() != request_bytes or _sha(Path(image)) != source_hash):
        raise ValueError("Source or worker evidence changed during validation")
    return {
        **record, "status": "succeeded", "failure": None, "outputs": outputs,
        "recovery": {
            "status": ("validated_after_log_pipe_cleanup_failure" if record["failure"] == PIPE_FAILURE
                       else "validated_from_completed_worker_checkpoint"),
            "original_receipt": str(receipt.relative_to(root)),
            "original_receipt_sha256": _sha(receipt),
            "worker_completion_sha256": _sha(output / "worker.json"), "new_model_runs": 0,
        },
    }
