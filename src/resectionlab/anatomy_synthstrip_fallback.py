"""Offer one explicit CPU SynthStrip attempt after a source-bound MPS OOM."""

import hashlib
import json
import os
from pathlib import Path
import re

import numpy as np

from .anatomy_model_registry import model_environment
from .patient_anatomy import digest, file_hash

CPU_MODEL = "synthstrip-cpu"
CPU_TIMEOUT_SECONDS = 600
CPU_RETRY_STATUS = "retry_available"
_MEMORY_ERROR = re.compile(rb"^RuntimeError: MPS backend out of memory \(.*max allowed: 6\.00 GiB\)", re.MULTILINE)
_ERROR_PREFIX = "RuntimeError: worker_exit_1; inspect "
_MAX_EVIDENCE_BYTES = 5 * 1024**2


def _cache_roots():
    from .poc_models import ROOT

    default = ROOT / "outputs/poc-models"
    local_cache = ROOT / "testcases/reconstruction-runs"
    roots = {}
    resolved = default.resolve()
    if resolved == default or (default.is_symlink() and resolved == local_cache):
        roots = {default: resolved, resolved: resolved}
    configured = os.environ.get("RESSECTIONLAB_POC_OUTPUT_ROOT")
    if not configured:
        return roots
    candidate = Path(configured).resolve()
    if candidate.is_relative_to((ROOT / "testcases").resolve()):
        roots[candidate] = candidate
    else:
        from .data_root import checked_root
        try:
            if candidate.is_relative_to(checked_root()):
                roots[candidate] = candidate
        except (OSError, ValueError):
            pass
    return roots


def _proof(c, source, mps, environment):
    from .poc_models import MODEL_CONFIG

    reason = mps.get("reason", "")
    if not isinstance(reason, str) or not reason.startswith(_ERROR_PREFIX):
        return None
    case = "anatomy-" + digest(c.document["patient_id"])[:16]
    path = Path(reason.removeprefix(_ERROR_PREFIX))
    if not path.is_absolute() or path.name != "inference.log" or ".." in path.parts:
        return None
    try:
        log = path.resolve(strict=True)
        # Only the known root alias may differ from the physical path. The
        # case, model, attempt and evidence files must still resolve literally.
        roots = _cache_roots()
        root = next((root for root in path.parents if root in roots), None)
        if root is None or log != roots[root] / path.relative_to(root):
            return None
        case_part, model_part, attempt_part, _ = path.relative_to(root).parts
        if case_part != case or model_part != "synthstrip-main":
            return None
        receipt = log.with_name("inference.json")
        if receipt.is_symlink() or receipt.stat().st_size > _MAX_EVIDENCE_BYTES:
            return None
        stored = receipt.read_bytes()
        record = json.loads(stored)
        identity = record["identity"]
        cache_key = hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()[:20]
        expected_weights = sorted(value for role, value in environment["files"].items() if role.startswith("weights/"))
        modality = "T1c" if source["modality"] == "T1CE" else source["modality"]
        config = MODEL_CONFIG["synthstrip-main"]
        if (not expected_weights or sorted(identity["weights"]) != expected_weights
                or identity["parameters"] != config["parameters"]
                or record["model"]["version"] != config["version"]
                or record["input"]["shape"] != source["data"]["shape"]
                or not np.allclose(record["input"]["affine_ras_mm"], source["affine"], atol=1e-5)
                or record["status"] != "failed" or record["failure"] != "worker_exit_1"
                or record["device"] != "mps" or identity["device"] != "mps"
                or record["case"] != case or record["model_id"] != "synthstrip-main"
                or identity["model_id"] != "synthstrip-main" or record["input"]["modality"] != modality
                or identity["image_sha256"] != source["original"]["sha256"]
                or record["input"]["sha256"] != source["original"]["sha256"]
                or record["cache_key"] != cache_key
                or not (attempt_part == cache_key or attempt_part.startswith(cache_key + "-retry-"))
                or log.stat().st_size > _MAX_EVIDENCE_BYTES):
            return None
        log_bytes = log.read_bytes()
        if not _MEMORY_ERROR.search(log_bytes):
            return None
    except (OSError, TypeError, AttributeError, KeyError, ValueError):
        return None
    return {
        "device": "mps", "failure": "mps_memory_budget_exceeded", "inputKey": mps["inputKey"],
        "source_sha256": source["original"]["sha256"], "receipt_sha256": hashlib.sha256(stored).hexdigest(),
        "log_sha256": hashlib.sha256(log_bytes).hexdigest(),
    }


def _retained_proof(c, source, mps):
    for row in c.document["layers"]:
        provenance = row.get("provenance", {})
        proof = provenance.get("fallback_from")
        if (row["kind"] == "brain" and row["source_series"] == source["id"]
                and provenance.get("model") == CPU_MODEL and provenance.get("device") == "cpu"
                and isinstance(proof, dict) and proof.get("inputKey") == mps["inputKey"]
                and proof.get("source_sha256") == source["original"]["sha256"]):
            return proof
    return None


def offer_cpu_retry(c, mps, receipts, stage_builder):
    receipt = receipts.get(mps["id"], {})
    if (mps.get("model") != "synthstrip" or mps["status"] not in {"ready", "failed"}
            or receipt.get("status") != "failed"):
        return None
    source = c.series(mps["series_id"])
    try:
        environment = model_environment("synthstrip", include_paths=True)
    except (OSError, ValueError):
        return None
    proof = _retained_proof(c, source, receipt) or _proof(c, source, receipt, environment)
    if proof is None:
        return None

    fallback = stage_builder(c, {
        "id": "brain-cpu:" + source["id"], "label": "Brain envelope · SynthStrip CPU retry",
        "capability": "brain", "series_id": source["id"], "device": "cpu",
        "model": CPU_MODEL, "options": {"device": "cpu", "fallback_from": proof, "timeout": CPU_TIMEOUT_SECONDS},
        "environment": environment, "status": "ready", "depends_on": [],
    }, receipts)
    if fallback["status"] == "ready" and fallback["id"] not in receipts:
        fallback.update(status=CPU_RETRY_STATUS,
                        reason="One CPU attempt is available after the recorded 6 GiB MPS memory limit failure.")
    elif fallback["status"] != "cached":
        fallback["retryable"] = False
        if fallback["status"] == "ready":
            fallback.update(status="failed", reason="The CPU retry has already been used for this study and source.")
    mps.update(status="failed", retryable=False, device="mps", inputKey=receipt["inputKey"],
               reason="MPS reached its 6 GiB allocation limit. The original attempt is retained.")
    return fallback


def reconstruct_cpu(c, stage, *, cancelled):
    from .poc_models import run

    source = c.series(stage["series_id"])
    path = c.path(source["original"]["path"])
    if file_hash(path) != stage["options"]["fallback_from"]["source_sha256"]:
        raise ValueError("SynthStrip CPU input changed after the recorded MPS failure")
    modality = "T1c" if source["modality"] == "T1CE" else source["modality"]
    result = run(path, "anatomy-" + digest(c.document["patient_id"])[:16], "synthstrip-main",
                 modality, device="cpu", timeout=CPU_TIMEOUT_SECONDS, cancelled=cancelled)
    if result["device"] != "cpu" or result["input"]["sha256"] != source["original"]["sha256"]:
        raise ValueError("SynthStrip CPU cache identity differs from the requested source and device")
    output = next((item for item in result["outputs"] if item["name"] == "brain_mask.nii.gz"), None)
    if output is None:
        raise ValueError("SynthStrip CPU did not produce a brain mask")
    model = result["model"]
    return {
        "model": CPU_MODEL, "kind": "brain", "source_series": source["id"], "source": source["original"]["sha256"],
        "source_modality": source["modality"], "version": model["version"], "device": "cpu",
        "parameters": {**model["parameters"], "timeout_seconds": CPU_TIMEOUT_SECONDS, "rss_limit_bytes": 8 * 1024**3},
        "path": output["path"], "output_sha256": output["sha256"], "inference": result,
        "availability": "empty_output" if output["foreground_voxels"] == 0 else "available",
        "fallback_from": stage["options"]["fallback_from"], "method": "image_derived_estimate",
        "label_ontology": {"value": 1, "meaning": "SynthStrip CPU extracted brain envelope"},
        "clinical_validation": False, "evidence_use": "context_only",
    }
