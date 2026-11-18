"""Coverage and identity of native-grid, brain-cropped Raidionics proposals."""

import hashlib

import numpy as np
from scipy import ndimage as ndi


def proposal_crop(c, source, record, shape, affine):
    binding = record.get("brain_mask")
    if not binding:
        return None
    if (list(shape) != source["data"]["shape"] or binding["data"]["shape"] != source["data"]["shape"]
            or not np.allclose(affine, source["affine"], atol=1e-5)
            or not np.allclose(binding["affine"], source["affine"], atol=1e-5)):
        raise ValueError("Brain-cropped model coverage needs the exact native source grid")

    data = c.array(binding["data"])
    current = next((row for row in c.document["layers"] if row["id"] == binding["layer_id"]), None)
    if current and (current["source_series"] != source["id"] or current["kind"] != "brain"):
        raise ValueError("Brain crop binding does not belong to the model source")
    # Older receipts lack the selected value. Match it against the exact
    # recorded binary mask digest, including after a later brain correction.
    values = [1] if not current else [current.get("original_label_value"), current["label_value"], 1]
    if "brain_input_label_value" in record:
        values = [record["brain_input_label_value"]]
    for value in dict.fromkeys(values):
        if value is None:
            continue
        mask = np.ascontiguousarray(data == value, dtype=np.uint8)
        if hashlib.sha256(mask).hexdigest() == binding["selected_mask_sha256"]:
            return _largest_component_bounds(mask)
    raise ValueError("Cannot reproduce the brain mask used for tumor model coverage")


def _largest_component_bounds(mask):
    # RaidionicsSeg 1.5.x keeps the bounding box of the largest scipy default
    # 6-connected component. Ties keep label scan order.
    labels, count = ndi.label(mask)
    if not count:
        raise ValueError("Cannot determine coverage from an empty brain mask")
    sizes = np.bincount(labels.ravel())
    sizes[0] = 0
    selected = ndi.find_objects(labels, max_label=count)[int(sizes.argmax()) - 1]
    return [[axis.start for axis in selected], [axis.stop - 1 for axis in selected]]


def compatible_proposal(c, series_id, record, values, probability, affine):
    """Find an original proposal with identical model evidence and scores."""
    for layer in c.document["layers"]:
        if layer["kind"] != "tumor" or layer["source_series"] != series_id or "probability" not in layer:
            continue
        p = layer.get("provenance", {})
        p = p.get("native_rerun", p)
        if (p.get("model") != record["model"] or p.get("source") != record["source"]
                or p.get("files") != record["files"] or p.get("runner_sha256") != record["runner_sha256"]
                or p.get("runtime_source_sha256") != record.get("runtime_source_sha256")
                or p.get("brain_mask", {}).get("selected_mask_sha256") != record["brain_mask"]["selected_mask_sha256"]
                or not np.allclose(layer["affine"], affine, atol=1e-5)):
            continue
        original = c.array(layer.get("original_data", layer["data"]))
        label_value = layer.get("original_label_value", layer["label_value"])
        score = c.array(layer["probability"])
        if np.array_equal(original == label_value, values > 0) and np.array_equal(score, probability):
            return layer
    return None


def bind_proposal_crop(c, layer, bounds):
    """Add model scope without altering geometry or overriding manual edits."""
    fields = layer.setdefault("coverage_fields", {})
    box = tuple(slice(lo, hi + 1) for lo, hi in zip(*bounds))
    mask = c.mask(layer)
    has_positive_outside = int(mask.sum()) != int(mask[box].sum())
    fields["model_inference_bounds_voxel_inclusive"] = bounds
    if has_positive_outside:
        if fields.get("inference_bounds_voxel_inclusive") == bounds:
            fields.pop("inference_bounds_voxel_inclusive")
        layer["coverage"] = (
            "Manual correction extends beyond the recorded model crop. Model scores outside this crop are unassessed."
        )
    else:
        fields["inference_bounds_voxel_inclusive"] = bounds
        layer["coverage"] = (
            "Brain-masked model inside the recorded native crop. Outside the crop is uncovered. Zero scores leave anatomical absence unassessed."
        )


def retained_crop_jobs(c):
    """Propose a metadata-only upgrade for retained, older model outputs."""
    from .patient_anatomy import file_hash

    for layer in c.document["layers"]:
        provenance = layer.get("provenance", {})
        record = provenance.get("native_rerun", provenance)
        fields = layer.get("coverage_fields", {})
        if (layer["kind"] != "tumor" or record.get("model") not in {"glioma-gbm", "glioma-lgg"}
                or not record.get("brain_mask") or fields.get("model_inference_bounds_voxel_inclusive") is not None):
            continue
        yield {
            "id": "tumor-crop-coverage:" + layer["id"],
            "label": "Record tumor model coverage",
            "capability": "tumor",
            "series_id": layer["source_series"],
            "model": "restore-tumor-crop-coverage",
            "options": {"proposal_layer_id": layer["id"]},
            "environment": {"adapter_sha256": file_hash(__file__), "model_files": record.get("files"),
                            "brain": record["brain_mask"], "scores": layer.get("probability")},
            "status": "ready" if "probability" in layer else "unavailable",
            "reason": ("Recover the native model crop from retained evidence" if "probability" in layer
                       else "Original model scores unavailable. Its model crop could not be verified."),
            "depends_on": [],
        }


def restore_retained_crop(c, layer_id):
    layer = c.layer(layer_id)
    provenance = layer.get("provenance", {})
    record = provenance.get("native_rerun", provenance)
    source = c.series(layer["source_series"])
    if record.get("source") != source["original"]["sha256"]:
        raise ValueError("Retained tumor proposal source identity differs")
    scores = c.array(layer["probability"])
    original = c.array(layer.get("original_data", layer["data"]))
    original_value = layer.get("original_label_value", layer["label_value"])
    threshold = record.get("parameters", {}).get("threshold", .5)
    if (original.shape != scores.shape or not np.isfinite(scores).all()
            or scores.min() < 0 or scores.max() > 1
            or not np.array_equal(original == original_value, scores >= threshold)):
        raise ValueError("Retained model scores do not reproduce the original tumor proposal")
    bounds = proposal_crop(c, source, record, original.shape, layer["affine"])
    box = tuple(slice(lo, hi + 1) for lo, hi in zip(*bounds))
    if np.count_nonzero(scores) != np.count_nonzero(scores[box]):
        raise ValueError("Retained model scores extend outside the reconstructed native crop")
    bind_proposal_crop(c, layer, bounds)
    return {"id": layer_id, "sha256": layer.get("original_data", layer["data"])["sha256"]}
