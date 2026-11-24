"""Read-only review of uncertain connections on an acquired TOF lattice."""

from base64 import b64encode

import numpy as np
from scipy import ndimage as ndi

from .anatomy_display import display_affine
from .anatomy_geometry import vascular_cleanup_source
from .anatomy_model_registry import source_volume
from .patient_anatomy import affine, digest, file_hash, world
from .vascular_source_branches import build_source_branches, evaluate_source_gaps


_MAX_FINDINGS = 128
_EVIDENCE_ORDER = {
    "uncertain-image-supported": 0,
    "weak-source-image-support": 1,
    "source-image-unassessed": 2,
}


def _review_window(signal):
    values = signal.ravel(order="K")[::max(1, signal.size // 96_000)]
    samples = values[np.isfinite(values) & (values != 0)]
    low, high = map(float, np.percentile(samples, [1, 99.5])) if samples.size else (0., 1.)
    return (low, high) if high > low else (min(0., low), max(1., high))


def _gap_sections(signal, mask, source_affine, candidate, window, *, check):
    """Sample two longitudinal planes and a transverse plane on the scan window."""
    points = np.asarray(candidate["points_ras_mm"])
    clear = ~np.asarray(candidate["center_predicted_vessel"])
    center = points[clear].mean(axis=0)
    tangent = (points[-1] - points[0]) / candidate["geometry"]["distance_mm"]
    seed = np.eye(3)[np.argmin(np.abs(tangent))]
    across = np.cross(tangent, seed)
    across /= np.linalg.norm(across)
    other = np.cross(tangent, across)
    inverse = np.linalg.inv(source_affine)
    step = float(np.clip(np.min(np.linalg.norm(source_affine[:3, :3], axis=0)) / 2, .15, .35))
    offsets = np.arange(-int(np.ceil(4 / step)), int(np.ceil(4 / step)) + 1) * step
    x, y = np.meshgrid(offsets, offsets)
    left, right = (points[clear][[0, -1]] - center) @ tangent
    dashes = (np.rint(x / step).astype(int) % 2) == 0
    path = (x >= left - step / 2) & (x <= right + step / 2) & (np.abs(y) < step / 2) & dashes
    sections = []
    for title, horizontal, vertical, marking in (
        ("Along gap · first plane", tangent, across, path),
        ("Along gap · second plane", tangent, other, path),
        ("Across gap midpoint", across, other, np.hypot(x, y) < step),
    ):
        check()
        coordinates = world(inverse, center + x[..., None] * horizontal + y[..., None] * vertical)
        indices = np.moveaxis(coordinates, -1, 0)
        gray = ndi.map_coordinates(signal, indices, order=1, output=np.float32, cval=np.nan, prefilter=False)
        estimate = ndi.map_coordinates(mask, indices, order=0, prefilter=False)
        outline = estimate & ~ndi.binary_erosion(estimate, border_value=1)
        covered = np.isfinite(gray)
        overlay = np.where(covered, np.where(marking & ~estimate, 2, outline), 0).astype(np.uint8)
        sections.append((title, gray, overlay))

    low, high = window
    scale = max(high - low, np.finfo(np.float32).eps * max(abs(high), 1.0))
    return {"window": [low, high], "sections": [
        {"title": title, "shape": list(gray.shape), "spacingMm": [step, step],
         "image": b64encode(np.clip(np.nan_to_num((gray - low) / scale, nan=0.) * 255, 0, 255)
                            .astype(np.uint8).tobytes()).decode("ascii"),
         "overlay": b64encode(overlay.tobytes()).decode("ascii")}
        for title, gray, overlay in sections
    ]}


def tof_gap_review(collection, layer_id, series_id, *, finding_id=None, review_id=None,
                   check=lambda: None, progress=lambda *_: None):
    """Build and review endpoint hypotheses from verified scan and label members."""
    row = collection.layer(layer_id)
    series = collection.series(series_id)
    unavailable = {"status": "unavailable", "sourceLayer": layer_id, "sourceSeries": series_id}
    if (series_id != row["source_series"] or series["modality"] != "TOF" or not source_volume(series)
            or row["kind"] not in {"vessel", "vessels", "artery", "vein", "sinus"}
            or row.get("provenance", {}).get("model") == "siam-v2"):
        return {**unavailable, "reason": "Select a vessel layer on its acquired TOF scan."}
    source = vascular_cleanup_source(collection, row) or row
    members = (row, source)
    if any(layer["data"]["shape"] != series["data"]["shape"]
           or layer["affine"] != series["affine"] for layer in members):
        return {**unavailable, "reason": "The estimate must share the native TOF lattice."}
    try:
        collection.require_native_geometry(series)
        for layer in members:
            collection.require_native_geometry(layer, allow_legacy_annotation_grid=True)
    except ValueError:
        return {**unavailable, "reason": "The source lattice requires geometry review."}
    frame = affine(series["affine"])
    axes = frame[:3, :3] / np.linalg.norm(frame[:3, :3], axis=0)
    if not np.allclose(axes.T @ axes, np.eye(3), atol=1e-5):
        return {**unavailable, "reason": "Gap inspection requires an orthogonal source grid."}

    check()
    original = collection.path(series["original"]["path"])
    original_key = (str(original), original.stat().st_mtime_ns, series["original"]["sha256"])
    if original_key not in collection._verified:
        if file_hash(original) != series["original"]["sha256"]:
            raise ValueError("Original native TOF checksum differs")
        collection._verified.add(original_key)
    identity = {"schema": "native-tof-gap-inspection-v1", "snapshotId": collection.identity(),
                "sourceLayer": layer_id, "sourceSeries": series_id, "estimateLayer": source["id"],
                "sourceSha256": series["original"]["sha256"], "signalSha256": series["data"]["sha256"],
                "sourceEstimateSha256": source["data"]["sha256"], "selectedEstimateSha256": row["data"]["sha256"],
                "sourceShape": series["data"]["shape"], "sourceAffineRasMm": series["affine"]}
    expected_id = digest(identity)
    if finding_id is not None and (type(finding_id) is not str or review_id != expected_id):
        raise ValueError("TOF gap review changed. Refresh the findings.")
    signal = collection.array(series["data"])
    mask = collection.mask(source)
    selected = collection.array(row["data"]) if source is not row else None
    check()
    cache = getattr(collection, "_tof_gap_review_cache", None)
    if cache is None or cache[0] != expected_id:
        progress(.1, "Tracing the native TOF vessel estimate")
        graph = build_source_branches(mask, frame, source_scan_sha256=identity["sourceSha256"], check=check)
        progress(.72, "Sampling acquired signal between compatible endpoints")
        gaps = evaluate_source_gaps(graph, mask, signal, check=check)
        present = None
        if selected is not None:
            labels = selected.ravel()
            present = {branch["id"] for branch in graph["branches"]
                       if (labels[branch["source_cell_ids"]] == row["label_value"]).all()}
        hypotheses = [candidate for candidate in gaps["candidates"] if candidate["state"] in _EVIDENCE_ORDER]
        candidates = [candidate for candidate in hypotheses
                      if present is None or all(branch in present for branch in candidate["branch_ids"])]
        candidates.sort(key=lambda item: (_EVIDENCE_ORDER[item["state"]], item["geometry"]["distance_mm"], item["id"]))
        coverage = {**gaps["coverage"], "candidatesOutsideSelectedLayer": len(hypotheses) - len(candidates),
                    "findingsTruncated": len(candidates) > _MAX_FINDINGS}
        candidates = {candidate["id"]: candidate for candidate in candidates[:_MAX_FINDINGS]}
        cache = collection._tof_gap_review_cache = (expected_id, coverage, candidates, _review_window(signal))
    _, coverage, candidates, window = cache

    findings = [{"id": item["id"], "evidence": item["state"], "distanceMm": item["geometry"]["distance_mm"],
                 "endpointCells": item["source_cell_ids"], "branchIds": item["branch_ids"],
                 "sameComponent": item["within_source_component"], "stations": item["gap_station_count"],
                 "supportedStations": item["supported_gap_station_count"]} for item in candidates.values()]
    partial = (coverage["source_branch_graph_status"] not in {"complete", "empty-prediction"}
               or coverage["crowded_terminals_abstained"] or coverage["pair_budget_exhausted"]
               or coverage["findingsTruncated"])
    result = {**identity, "reviewId": expected_id, "status": "partial" if partial else "complete",
              "coverage": coverage, "findings": findings, "predictedVoxelsAdded": 0}
    if finding_id is not None and finding_id not in candidates:
        raise ValueError("TOF gap review changed. Refresh the findings.")
    candidate = candidates.get(finding_id) if finding_id is not None else next(iter(candidates.values()), None)
    if candidate is not None:
        points = np.asarray(candidate["points_ras_mm"])
        clear = ~np.asarray(candidate["center_predicted_vessel"])
        center = points[clear].mean(axis=0)
        try:
            displayed, _ = display_affine(collection, series, preview=True)
        except ValueError:
            displayed = frame
        displayed_center = world(displayed @ np.linalg.inv(frame), center)
        result["preview"] = {"findingId": candidate["id"], "centerRasMm": displayed_center.tolist(),
                             **_gap_sections(signal, mask, frame, candidate, window, check=check)}
    return result
