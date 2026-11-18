"""Validate already prepared native predictions before substituting for a model job."""

import json
from pathlib import Path

import nibabel as nib
import numpy as np

from .anatomy_model_registry import TUMOR_ADAPTERS, _identities, source_volume, synthseg_assets, tumor_assets
from .head_segmentation import ASSETS, MODEL_SHA, PARAMETERS
from .poc_models import MODEL_CONFIG

SYNTHSTRIP_SHA = "37417f802196186441aae3e7f385d94f8a98c64a88acaeaa2723af995c653e33"
VESSELFM_SHA = "6da4fbd06dd9c3b597e5d50cc628deb3026919ff126d01afbd13ac8fe22ee10c"


def _check_file(path, expected):
    path = Path(path)
    return _identities([path])[str(path)] == expected


def _read_receipt(path):
    path = Path(path)
    if path.stat().st_size > 1024**2:
        raise ValueError("Inference receipt is too large")
    return json.loads(path.read_text())


def _legacy_output(c, provenance, row, layer, expected_model):
    """Legacy receipts qualify only with their exact raw and output files."""
    if layer["kind"] == "brain":
        record = provenance.get("inference")
        if not record and provenance.get("inference_receipt"):
            record = _read_receipt(provenance["inference_receipt"])
        if not record or record.get("model_id") != "synthstrip-main":
            return None
        if (record.get("status") != "succeeded" or record["input"]["sha256"] != row["original"]["sha256"]
                or record["model"]["version"] != MODEL_CONFIG["synthstrip-main"]["version"]
                or [w["sha256"] for w in record["weights"]] != [SYNTHSTRIP_SHA]):
            return None
        output = next((item for item in record["outputs"] if item["name"] == "brain_mask.nii.gz"), None)
        retained_mask = provenance.get("model_mask_original")
        if output and retained_mask and retained_mask["sha256"] == output["sha256"]:
            return {**output, "path": str(c.path(retained_mask["path"]))}
        return output
    if expected_model in TUMOR_ADAPTERS:
        record = provenance.get("native_rerun", provenance)
        if (record.get("model") != expected_model or record.get("source") != row["original"]["sha256"]
                or record.get("status") != "proposal" or record.get("foreground_voxels", 0) <= 0
                or TUMOR_ADAPTERS[expected_model][1] != row["modality"]):
            return None
        folder, packages = tumor_assets(expected_model)
        files = ("model.onnx", "pipeline.json", "pre_processing.ini", "model_config.txt")
        if (not all(_check_file(folder / name, record["files"][name]) for name in files)
                or not _check_file(packages / "raidionicsseg/fit.py", record["runner_sha256"])):
            return None
        if expected_model in {"glioma-gbm", "glioma-lgg"}:
            dependency = record["brain_mask"]
            brain = c.layer(dependency["layer_id"])
            if (brain["source_series"] != row["id"] or brain["kind"] != "brain"
                    or brain["data"] != dependency["data"] or brain.get("revision", 0) != dependency["revision"]
                    or not np.allclose(brain["affine"], dependency["affine"])):
                return None
        return {"path": record["path"], "sha256": record["output_sha256"]}
    if provenance.get("model") != "synthseg" or provenance.get("source") != row["original"]["sha256"]:
        return None
    _, script, weights = synthseg_assets(provenance["version"])
    lut = script.parents[2] / "data/labels_classes_priors"
    suffix = "" if provenance["version"] == 1 else "_2.0"
    class_ids = np.load(lut / f"synthseg_segmentation_labels{suffix}.npy", allow_pickle=False)
    if provenance["version"] == 2:
        class_ids = np.concatenate((class_ids, np.load(lut / "synthseg_parcellation_labels.npy", allow_pickle=False)))
    if (layer["label_value"] not in class_ids or layer["role"] != "context"
            or not _check_file(script, provenance["runner_sha256"])
            or not all(_check_file(p, provenance["weights"][p.name]) for p in weights)):
        return None
    return {"path": provenance["path"], "sha256": provenance["output_sha256"]}


def _source_identity(c, row, *, compare_native_array=False):
    original = row["original"]
    path = c.path(original["path"])
    if not _check_file(path, original["sha256"]):
        return False
    stored = c.array(row["data"])
    if not compare_native_array:
        return True
    im = nib.load(path)
    if im.shape != stored.shape or not np.allclose(im.affine, row["affine"]):
        return False
    return np.array_equal(im.get_fdata(dtype=np.float32), stored)


def _migration_output(c, row, layer, stored):
    p = layer["provenance"]
    ontology = p["label_ontology"]
    receipt_ref = p["inference_receipt_original"]
    output = p["source"]
    if (p.get("clinical_validation") is not False or p["source_sha256"] != row["original"]["sha256"]
            or p["source_array_sha256"] != row["data"]["sha256"] or p["prepared_array_sha256"] != stored["sha256"]
            or ontology["value"] != layer["label_value"] or not ontology["name"]
            or receipt_ref["sha256"] != p["inference_receipt_sha256"] or output["sha256"] != p["output_sha256"]
            or not _check_file(c.path(receipt_ref["path"]), receipt_ref["sha256"])
            or not _check_file(c.path(output["path"]), output["sha256"])):
        return False
    receipt = _read_receipt(c.path(receipt_ref["path"]))
    im = nib.load(c.path(output["path"]))
    if (receipt["status"] != "succeeded" or list(im.shape) != stored["shape"]
            or not np.allclose(im.affine, layer["affine"], atol=1e-5)):
        return False
    identity = p["model_identity"]
    if p["model"] == "SuperSynth":
        return (p.get("evidence_use") == "context_only"
                and ontology["lut_sha256"] == ASSETS["FreeSurferColorLUT.txt"]
                and receipt["identity"] == identity and identity["model_sha256"] == MODEL_SHA
                and identity["source_sha256"] == row["original"]["sha256"]
                and identity["parameters"] == PARAMETERS and receipt["output_sha256"] == output["sha256"])
    if p["model"] == "vesselfm-base":
        preprocessing = p["preprocessing_receipt_original"]
        if not _check_file(c.path(preprocessing["path"]), preprocessing["sha256"]):
            return False
        linkage = _read_receipt(c.path(preprocessing["path"]))
        return (row["modality"] in {"CTA", "TOF", "MRA"} and source_volume(row)
                and p["evidence_use"] == "positive_labels_only" and layer["kind"] == "vessels"
                and identity["version"] == MODEL_CONFIG["vesselfm-base"]["version"]
                and receipt["model_id"] == "vesselfm-base" and receipt["model"] == identity
                and [w["sha256"] for w in receipt["weights"]] == [VESSELFM_SHA]
                and linkage["source_sha256"] == row["original"]["sha256"]
                and linkage["output_sha256"] == receipt["input"]["sha256"]
                and output["sha256"] in {item["sha256"] for item in receipt["outputs"]})
    return False


def reusable_model_layer(c, row, kinds, *, model, native=False):
    """Find one evidenced prepared output. Never change its review or role."""
    for layer in c.document["layers"]:
        if (layer["source_series"] != row["id"] or layer["kind"] not in kinds
                or layer.get("preparation_input") or layer.get("availability") == "empty_output"):
            continue
        provenance = layer.get("provenance", {})
        migrated = provenance.get("preparation_cache_verified") is True
        prepared_model = str(provenance.get("model", "")).lower()
        if migrated and prepared_model != model:
            continue
        if not migrated and model not in {"synthstrip", "synthseg", *TUMOR_ADAPTERS}:
            continue
        if native and (row["data"]["shape"] != layer["data"]["shape"]
                       or not np.allclose(layer["affine"], row["affine"], atol=1e-5)):
            continue
        stored = layer.get("original_data", layer["data"])
        try:
            if migrated:
                if not _migration_output(c, row, layer, stored) or not _source_identity(c, row):
                    continue
                c.array(stored)
            else:
                output = _legacy_output(c, provenance, row, layer, model)
                if output is None or not _check_file(output["path"], output["sha256"]):
                    continue
                im = nib.load(output["path"])
                data = c.array(stored)
                if (im.shape != data.shape or not np.allclose(im.affine, layer["affine"], atol=1e-5)
                        or not np.array_equal(np.asarray(im.dataobj), data)
                        or not _source_identity(c, row, compare_native_array=True)):
                    continue
            if np.any(c.array(layer["data"]) == layer["label_value"]):
                return layer
        except (KeyError, StopIteration, TypeError, OSError, ValueError, nib.filebasedimages.ImageFileError):
            continue
    return None
