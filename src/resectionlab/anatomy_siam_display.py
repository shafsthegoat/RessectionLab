"""Source-supported display of unedited native SIAM predictions."""

import json

import nibabel as nib
import numpy as np
from scipy import ndimage as ndi
from skimage.morphology import convex_hull_image

from .data_root import atomic_json
from .patient_anatomy import digest

VERSION = "siam-mri-head-support-v2"
MARGIN_MM = 4.0
AIR_CLEARANCE_MM = 2.0


def _head_support(source, matrix, check):
    """Enclose the dominant connected MRI signal on each acquired axial section."""
    spacing = nib.affines.voxel_sizes(matrix)
    stride = np.maximum.reduce([np.ones(3), np.round(1.6 / spacing), np.ceil(np.array(source.shape) / 192)]).astype(int)
    sampled = np.asarray(source[tuple(slice(None, None, step) for step in stride)])
    low, high = np.percentile(sampled, [1, 99])
    if high <= low:
        return None, "MRI has insufficient intensity variation"
    threshold = float(low + 0.09 * (high - low))
    check()
    signal = ndi.binary_opening(sampled > threshold)
    components, count = ndi.label(signal, structure=np.ones((3, 3, 3), bool))
    sizes = np.bincount(components.ravel())
    if not count or max(sizes[1:]) < max(32, sampled.size * 0.005):
        return None, "No continuous MRI tissue signal sufficient to bound the head"
    head = components == np.argmax(sizes[1:]) + 1
    superior = int(np.flatnonzero(nib.orientations.io_orientation(matrix)[:, 0] == 2)[0])
    axial = np.moveaxis(head, superior, 0)
    # Convex sections retain dark bone, CSF and disconnected anatomy between
    # measured tissues. The margin also accommodates opening and coarse sampling.
    for i, plane in enumerate(axial):
        if i % 12 == 0:
            check()
        if plane.any():
            plane[:] = convex_hull_image(plane)
    sample_spacing = spacing * stride
    margin = float(MARGIN_MM + 0.5 * np.linalg.norm(sample_spacing))
    distance = ndi.distance_transform_edt(~head, sampling=sample_spacing)
    # Retain every model class inside the convex sections plus coarse-grid
    # quantization allowance, even when the native MRI signal there is dark.
    interior_margin = float(0.5 * (np.linalg.norm(sample_spacing) + min(spacing)))
    return (distance <= margin, distance <= interior_margin, stride), {
        "interiorMarginMm": interior_margin, "nativeAirClearanceMm": AIR_CLEARANCE_MM,
        "threshold": threshold, "percentileWindow": [float(low), float(high)],
        "sampleSpacingMm": sample_spacing.tolist(), "marginMm": margin,
    }


def _native_margin(source, start, stop, spacing, threshold):
    """Protect measured tissue in the outer margin; only veto observed background."""
    # Pad the slab for native opening and physical-distance tests. Keep the
    # acquisition borders, where a complete source neighborhood is unavailable.
    padding = 2 + np.ceil(AIR_CLEARANCE_MM / spacing).astype(int)
    first = max(0, start - padding[0])
    last = min(len(source), stop + padding[0])
    signal = source[first:last] > threshold
    coherent = ndi.binary_opening(signal)
    if coherent.any():
        distance = ndi.distance_transform_edt(~coherent, sampling=spacing)
        signal |= distance <= AIR_CLEARANCE_MM
    protected = signal[start - first:stop - first]
    for axis, (size, pad) in enumerate(zip(source.shape, padding)):
        indices = np.arange(start, stop) if axis == 0 else np.arange(size)
        edge = (indices < pad) | (indices >= size - pad)
        shape = [1, 1, 1]
        shape[axis] = -1
        protected |= edge.reshape(shape)
    return protected


def _build(c, row, series, check):
    raw = c.array(row["data"])
    source = c.array(series["data"])
    matrix = np.asarray(row["affine"])
    spacing = nib.affines.voxel_sizes(matrix)
    head, details = _head_support(source, matrix, check)
    if head is None:
        return {"status": "unavailable", "reason": details, "data": row["data"]}
    support, interior, stride = head
    indices = [np.minimum((np.arange(n) + step // 2) // step, limit - 1)
               for n, step, limit in zip(raw.shape, stride, support.shape)]
    output = np.empty_like(raw, subok=False)
    mask = np.empty(raw.shape, np.uint8)
    before = np.zeros(17, np.int64)
    after = before.copy()
    for i in range(0, len(raw), 8):
        check()
        sl = slice(i, i + 8)
        index = np.ix_(indices[0][sl], indices[1], indices[2])
        native_support = support[index]
        core = interior[index]
        if np.any(native_support & ~core):
            native_support &= core | _native_margin(source, i, min(i + 8, len(raw)), spacing, details["threshold"])
        mask[sl] = native_support
        output[sl] = np.where(native_support, raw[sl], 0)
        before += np.bincount(raw[sl].ravel(), minlength=17)
        after += np.bincount(output[sl].ravel(), minlength=17)
    return {"status": "supported", "data": c.put_array(output), "mask": c.put_array(mask), "before": before.tolist(),
            "after": after.tolist(), **details}


def _native_prediction(row, *, original):
    provenance = row.get("provenance", {})
    if provenance.get("model") != "siam-v2":
        return None
    if row.get("edits"):
        if not original or not row.get("original_data"):
            return None
        row = {**row, "data": row["original_data"], "label_value": row["original_label_value"]}
    if provenance.get("label_ontology", {}).get("value") != row.get("label_value"):
        return None
    return row


def _identity(c, row, series):
    p, raw = row["provenance"], row["data"]
    if (p.get("source_array_sha256") != series["data"]["sha256"]
            or p.get("source") != series["original"]["sha256"]
            or raw["shape"] != series["data"]["shape"]
            or not np.allclose(row["affine"], series["affine"], atol=1e-5, rtol=0)):
        raise ValueError("SIAM display requires its original MRI identity and lattice")
    # Reopened packages may have new files at the same relative paths. The
    # collection verifies checksums, caching successful checks by file mtime.
    c.array(raw)
    c.array(series["data"])
    return {"version": VERSION, "schema": 2, "rawDataSha256": raw["sha256"],
            "sourceSha256": series["original"]["sha256"], "sourceArraySha256": series["data"]["sha256"],
            "affine": row["affine"], "shape": raw["shape"]}


def _cached_support(c, row, series, identity, check):
    key = digest(identity)
    path = c.root / "display-cache" / f"{VERSION}-{key}.json"
    if not hasattr(c, "_siam_display_cache"):
        c._siam_display_cache = {}
    entry = c._siam_display_cache.get(key)
    if entry is None and path.is_file():
        cached = json.loads(path.read_text())
        if cached["identity"] != identity or cached["checksum"] != digest(cached["result"]):
            raise ValueError("SIAM display support cache integrity failure")
        entry = cached["result"]
    if entry is not None:
        try:
            c.array(entry["data"])
            if entry["status"] == "supported":
                c.array(entry["mask"])
        except FileNotFoundError:
            entry = None
    if entry is None:
        entry = _build(c, row, series, check)
        path.parent.mkdir(exist_ok=True)
        atomic_json(path, {"identity": identity, "result": entry, "checksum": digest(entry)})
    c._siam_display_cache[key] = entry
    return entry


def _component_removal_history(row):
    edits = row.get("edits")
    if not isinstance(edits, list) or not edits or row.get("provenance", {}).get("model") != "siam-v2":
        return False
    for edit in edits:
        history = edit.get("operations") if isinstance(edit, dict) else None
        if not isinstance(history, list):
            return False
        operations = [item.get("operation") if isinstance(item, dict) else None for item in history]
        if "remove-components" not in operations or any(op not in ("remove-components", "undo", "redo") for op in operations):
            return False
    return True


def _supported_correction(c, row, base, entry, check):
    # An accepted component deletion should only subtract from the former display.
    # Mixed edit histories can contain deliberate additions outside the MRI envelope.
    corrected = c.array(row["data"])
    check()
    key = ("component-removal", row["data"]["sha256"], row["label_value"],
           base["data"]["sha256"], base["label_value"], entry["data"]["sha256"])
    cache = c._siam_display_cache
    if key not in cache:
        mask = corrected == row["label_value"]
        # Even a malformed imported history must not hide manually introduced voxels.
        if np.any(mask & (c.array(base["data"]) != base["label_value"])):
            return None
        total = int(np.count_nonzero(mask))
        mask &= c.array(entry["data"]) == base["label_value"]
        cache[key] = (c.put_array(mask.astype(np.uint8)), total, int(np.count_nonzero(mask)))
    data, total, retained = cache[key]
    return {**row, "data": data, "label_value": 1}, total, retained


def supported_tissue_layer(collection, row, *, check=lambda: None, raw_qc=False):
    """Get a display-only row and receipt. Preserve supplied labels and corrections."""
    removal = not raw_qc and _component_removal_history(row)
    base = _native_prediction(row, original=raw_qc or removal)
    if base is None:
        return row, None
    c = collection
    series = c.series(base["source_series"])
    identity = _identity(c, base, series)
    report = {k: v for k, v in identity.items() if k not in {"shape", "affine", "schema"}}
    if raw_qc:
        count = int(np.count_nonzero(c.array(base["data"]) == base["label_value"]))
        return base, {**report, "status": "unfiltered", "sampledDataSha256": identity["rawDataSha256"],
                      "sourceVoxels": count, "retainedVoxels": count, "hiddenVoxels": 0}
    entry = _cached_support(c, base, series, identity, check)
    report.update({k: v for k, v in entry.items() if k not in {"data", "mask", "before", "after"}})
    if removal:
        if entry["status"] != "supported":
            return row, None
        result = _supported_correction(c, row, base, entry, check)
        if result is None:
            return row, None
        display, total, retained = result
        report.update(maskSha256=entry["mask"]["sha256"], sampledDataSha256=display["data"]["sha256"],
                      correctedDataSha256=row["data"]["sha256"], sourceVoxels=total,
                      retainedVoxels=retained, hiddenVoxels=total - retained)
        return display, report
    report["sampledDataSha256"] = entry["data"]["sha256"]
    if entry["status"] == "supported":
        report["maskSha256"] = entry["mask"]["sha256"]
        value = base["label_value"]
        original, retained = entry["before"][value], entry["after"][value]
        report.update(sourceVoxels=original, retainedVoxels=retained, hiddenVoxels=original - retained)
    else:
        count = int(np.count_nonzero(c.array(base["data"]) == base["label_value"]))
        report.update(sourceVoxels=count, retainedVoxels=count, hiddenVoxels=0)
    return {**base, "data": entry["data"]}, report


def source_head_support(collection, series, *, check=lambda: None):
    """Return the native exterior mask shared by SIAM overlays and 3D MRI rendering."""
    c = collection
    for row in c.document["layers"]:
        if row["source_series"] == series["id"]:
            base = _native_prediction(row, original=True)
            if base is not None:
                identity = _identity(c, base, series)
                entry = _cached_support(c, base, series, identity, check)
                return c.array(entry["mask"]) if entry["status"] == "supported" else None
    return None
