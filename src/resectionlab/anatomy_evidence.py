"""Eligibility for native geometric vascular constraints."""

VASCULAR_KINDS = {"vessel", "vessels", "artery", "vein", "sinus"}


def is_siam_context(layer):
    """Each SIAM v2 tissue class is admitted for image inspection only."""
    return layer.get("provenance", {}).get("model") == "siam-v2"


def vascular_evidence_supported(collection, layer):
    """Allow source-associated annotations and supported positive model labels."""
    if layer["kind"] not in VASCULAR_KINDS:
        return True
    provenance = layer.get("provenance", {})
    if (layer.get("availability") == "empty_output" or layer.get("evidence_status") == "empty-output"
            or provenance.get("evidence_use") == "context_only" or provenance.get("kind") == "derived_annotation_union"):
        return False
    inferred = provenance.get("kind") in {"model_estimate", "image_derived_estimate", "geometric_estimate"}
    inferred = inferred or bool(provenance.get("model") or provenance.get("inference"))
    # These legacy EDEN descriptors preceded the preparation evidence policy.
    if layer["id"] in {"t1-arteries", "t1-veins"} and layer["source_series"].lower() == "t1":
        return False
    if not inferred:
        return True
    # This allowlist concerns positive geometry. VesselFM does not establish
    # arterial/venous class, completeness, or clear space beyond its labels.
    from .anatomy_model_registry import ANGIOGRAPHY, source_volume

    source = collection.series(layer["source_series"])
    model = str(provenance.get("model", "")).lower()
    return (model in {"vesselfm", "vesselfm-base"} and layer["kind"] in {"vessel", "vessels"}
            and provenance.get("kind") != "geometric_estimate"
            and source["modality"] in ANGIOGRAPHY and source_volume(source))


def inference_bounds(layer):
    """Validate the inclusive output-voxel bounds of a restored inference crop."""
    fields = layer.get("coverage_fields", {})
    if not isinstance(fields, dict):
        raise ValueError("Invalid layer coverage fields")
    bounds = fields.get("inference_bounds_voxel_inclusive")
    if bounds is None:
        return None
    shape = layer["data"]["shape"]
    if (not isinstance(bounds, (list, tuple)) or len(bounds) != 2 or len(shape) != 3
            or any(not isinstance(row, (list, tuple)) or len(row) != 3 for row in bounds)
            or any(type(lo) is not int or type(hi) is not int or not 0 <= lo <= hi < size
                   for lo, hi, size in zip(*bounds, shape))):
        raise ValueError("Invalid inference coverage bounds")
    return tuple(bounds[0]), tuple(bounds[1])


def is_dural_proxy(layer):
    return layer["kind"] == "dura" and layer.get("provenance", {}).get("model") == "dural-envelope"
