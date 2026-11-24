"""Conservative, reversible review of native angiographic vessel predictions."""

from base64 import b64encode
from dataclasses import asdict, dataclass

import numpy as np
from scipy import ndimage as ndi, sparse
from scipy.spatial import cKDTree

from .anatomy_siam_display import _head_support
from .patient_anatomy import affine, digest, file_hash, world


RETAINED, UNCERTAIN, HIDDEN = 1, 2, 3
CONNECTIVITY = np.ones((3, 3, 3), dtype=bool)
EXTERIOR_DISPLAY = "native-tof-exterior-display-v1"
_AIR_CLEARANCE_MM = 2.0
_MAX_LABEL_CELLS = 128_000_000
_MAX_NEIGHBORHOOD_CELLS = 4_000_000
_MAX_EXTERIOR_FOREGROUND = 4_000_000
ATTACHED_REVIEW = "native-tof-attached-uncertainty-v1"
_ATTACHED_DARK_PERCENTILE = 10.0
_ATTACHED_CORE_RADIUS_MM = 1.0
_ATTACHED_MINIMUM_VOLUME_MM3 = 2.5
_ATTACHED_CONTEXT_MM = 2.0
_ATTACHED_ANCHOR_CONTRAST_IQR = 0.15
_MAX_ATTACHED_WEAK_CELLS = 500_000
_MAX_ATTACHED_FINDINGS = 24


@dataclass(frozen=True)
class ComponentPolicy:
    version: str = "native-tof-detached-signal-v1"
    compact_volume_mm3: float = 0.32
    compact_axis_ratio: float = 2.0
    small_volume_mm3: float = 0.85
    isolation_mm: float = 1.5
    outer_ring_mm: float = 2.0
    inner_ring_mm: float = 0.55
    maximum_peak_contrast: float = 0.75
    score_median_ceiling: float = 0.8
    baseline_signal_percentile: float = 10.0
    very_low_signal_percentile: float = 5.0


def _bounds(points, spacing, shape, radius):
    pad = np.ceil(radius / spacing).astype(int)
    start = np.maximum(points.min(axis=0) - pad, 0)
    stop = np.minimum(points.max(axis=0) + pad + 1, shape)
    return start, tuple(slice(a, b) for a, b in zip(start, stop))


def _source_contrast(points, signal, mask, spacing, policy):
    start, slices = _bounds(points, spacing, signal.shape, policy.outer_ring_mm)
    local_signal = signal[slices]
    local = np.ones(local_signal.shape, dtype=bool)
    local[tuple((points - start).T)] = False
    distance = ndi.distance_transform_edt(local, sampling=spacing)
    ring = (
        (distance >= policy.inner_ring_mm)
        & (distance <= policy.outer_ring_mm)
        & ~mask[slices]
        & np.isfinite(local_signal)
    )
    nearby = local_signal[ring]
    foreground = signal[tuple(points.T)]
    if len(nearby) < 20 or not np.isfinite(foreground).all():
        return None
    lo, middle, hi = np.percentile(nearby, [25, 50, 75])
    peak = float(foreground.max())
    scale = max(float(hi - lo), np.finfo(np.float32).eps * max(abs(float(middle)), 1.0))
    return {
        "foreground_peak": peak,
        "foreground_median": float(np.median(foreground)),
        "ring_median": float(middle),
        "ring_iqr": float(hi - lo),
        "peak_contrast_iqr": (peak - float(middle)) / scale,
    }


def exterior_air_components(mask, signal, native_affine, *, check=lambda: None):
    """Propose whole-component presentation exclusions in sampled exterior air.

    The head envelope nominates a region for review. Both the component and
    its native-resolution neighborhood must be dark. An attached candidate,
    a candidate at an acquisition face, or nearby measured signal stays
    visible. The original foreground and source field are never written.
    """
    mask = np.asarray(mask)
    signal = np.asarray(signal)
    if (mask.dtype != np.bool_ or mask.ndim != 3 or mask.shape != signal.shape
            or signal.dtype.kind not in "fui"):
        raise ValueError("Expected a binary prediction and matched native TOF signal")
    frame = affine(native_affine)
    spacing = np.linalg.norm(frame[:3, :3], axis=0)
    axes = frame[:3, :3] / spacing
    if not np.allclose(axes.T @ axes, np.eye(3), atol=1e-5):
        return [], {"status": "unavailable", "reason": "Native TOF lattice is sheared"}
    foreground = int(np.count_nonzero(mask))
    if not foreground:
        return [], {"status": "unchanged", "reason": "The source prediction is empty"}
    if foreground > _MAX_EXTERIOR_FOREGROUND:
        return [], {"status": "unavailable", "reason": "Native TOF foreground exceeds review workspace"}
    points = np.column_stack(np.nonzero(mask))
    start, slices = _bounds(points, spacing, mask.shape, 0)
    if np.prod([axis.stop - axis.start for axis in slices]) > _MAX_LABEL_CELLS:
        return [], {"status": "unavailable", "reason": "Native TOF prediction exceeds labeling workspace"}
    check()
    head, evidence = _head_support(signal, frame, check)
    if head is None or not np.isfinite(evidence["percentileWindow"]).all():
        return [], {"status": "unavailable", "reason": "No reliable native TOF head envelope"}
    support, _, stride = head
    sample = np.minimum((points + stride // 2) // stride, np.array(support.shape) - 1)
    outside = ~support[tuple(sample.T)]
    # Tie the low component ceiling to the observed signal floor. Use the
    # more permissive head threshold solely to protect neighboring signal.
    low, high = evidence["percentileWindow"]
    ceiling = low + 0.02 * (high - low)
    foreground_signal = signal[tuple(points.T)]
    distance_to_face = np.minimum(points + .5, np.array(mask.shape) - .5 - points) * spacing
    eligible = outside & np.isfinite(foreground_signal) & (foreground_signal <= ceiling)
    eligible &= (distance_to_face > _AIR_CLEARANCE_MM).all(axis=1)
    report = {
        "status": "unchanged",
        "sourceVoxels": foreground,
        "foregroundSignalCeiling": float(ceiling),
        **evidence,
    }
    if not eligible.any():
        return [], report

    check()
    labels, count = ndi.label(mask[slices], structure=CONNECTIVITY)
    memberships = labels[tuple((points - start).T)]
    del labels
    reject = np.zeros(count + 1, dtype=bool)
    reject[memberships[~eligible]] = True
    candidates = np.flatnonzero(~reject[memberships])
    candidates = candidates[np.argsort(memberships[candidates], kind="stable")]
    breaks = np.flatnonzero(np.diff(memberships[candidates])) + 1
    excluded = []
    for group in np.split(candidates, breaks):
        if not len(group):
            break
        check()
        cells = points[group]
        origin, bounds = _bounds(cells, spacing, mask.shape, _AIR_CLEARANCE_MM)
        neighborhood = signal[bounds]
        if neighborhood.size > _MAX_NEIGHBORHOOD_CELLS:
            continue
        nearby_signal = ~np.isfinite(neighborhood) | (neighborhood > evidence["threshold"])
        if np.any(nearby_signal):
            distance = ndi.distance_transform_edt(~nearby_signal, sampling=spacing)
            if np.any(distance[tuple((cells - origin).T)] <= _AIR_CLEARANCE_MM):
                continue
        excluded.append(np.ravel_multi_index(cells.T, mask.shape))
    return excluded, report


def supported_tof_layer(collection, row, *, check=lambda: None):
    """Get an exterior-air display derivative bound to the original TOF layer."""
    provenance = row.get("provenance", {})
    if (row["kind"] != "vessels" or provenance.get("model") != "vesselfm"
            or row.get("edits") or not row.get("probability")):
        return row, None
    from .anatomy_model_registry import source_volume

    series = collection.series(row["source_series"])
    if series["modality"] != "TOF" or not source_volume(series):
        return row, None
    if (row["affine"] != series["affine"] or row["data"]["shape"] != series["data"]["shape"]
            or provenance.get("source") != series["original"]["sha256"]):
        return row, {"scope": "exterior-air", "reason": "Source identity or native lattice could not be verified"}
    try:
        collection.require_native_geometry(series)
        collection.require_native_geometry(row)
    except ValueError:
        return row, {"scope": "exterior-air", "reason": "The source lattice requires geometry review"}
    identity = {
        "version": EXTERIOR_DISPLAY,
        "sourceSha256": series["original"]["sha256"],
        "sourceSignalSha256": series["data"]["sha256"],
        "rawDataSha256": row["data"]["sha256"],
        "sourceProbabilitySha256": row["probability"]["sha256"],
        "affine": series["affine"],
        "label": row["label_value"],
    }
    key = digest(identity)
    if not hasattr(collection, "_tof_display_cache"):
        collection._tof_display_cache = {}
    cache = collection._tof_display_cache
    data, receipt = cache.get(key, (None, None))
    if data is not None:
        try:
            collection.array(data)
        except FileNotFoundError:
            receipt = None
    if receipt is None:
        mask = np.ascontiguousarray(collection.mask(row))
        groups, evidence = exterior_air_components(mask, collection.array(series["data"]), series["affine"], check=check)
        hidden = sum(len(group) for group in groups)
        if 0 < hidden < evidence["sourceVoxels"]:
            for group in groups:
                mask.ravel()[group] = False
            check()
            data = collection.put_array(mask.astype(np.uint8))
            reason = (
                f"Exterior-air view hides {len(groups):,} detached source-dark findings. "
                "Use Full source estimate to inspect them."
            )
        else:
            data = None
            hidden = 0
            reason = evidence.get("reason", "No detached finding passed the exterior-air screen")
        receipt = {
            **{k: v for k, v in identity.items() if k not in {"affine", "label"}},
            "scope": "exterior-air",
            "hiddenVoxels": hidden,
            "excludedFragments": len(groups) if hidden else 0,
            "sampledDataSha256": data["sha256"] if data else row["data"]["sha256"],
            "evidence": evidence,
            "reason": reason,
        }
        cache[key] = data, receipt
    return ({**row, "data": data, "label_value": 1} if data else row), receipt


def _weak_components(points, shape, check):
    """Keep sparse weak-signal components off the full native-sized label grid."""
    flat = np.ravel_multi_index(points.T, shape)
    strides = np.cumprod((1, *shape[:0:-1]))[::-1]
    starts, stops = [], []
    for step in np.argwhere(CONNECTIVITY) - 1:
        if tuple(step) <= (0, 0, 0):
            continue
        check()
        valid = np.ones(len(points), dtype=bool)
        for axis in np.flatnonzero(step):
            limit = shape[axis] - 1 if step[axis] > 0 else 0
            valid &= points[:, axis] != limit
        source = np.flatnonzero(valid)
        target = flat[source] + step @ strides
        index = np.searchsorted(flat, target).clip(max=len(flat) - 1)
        adjacent = flat[index] == target
        starts.append(source[adjacent].astype(np.int32))
        stops.append(index[adjacent].astype(np.int32))
    row, col = np.concatenate(starts), np.concatenate(stops)
    graph = sparse.coo_matrix((np.ones(len(row), dtype=bool), (row, col)), shape=(len(flat), len(flat)))
    return sparse.csgraph.connected_components(graph, directed=False)[1]


def attached_vessel_uncertainty(mask, signal, scores, native_affine, *, check=lambda: None):
    """Locate thick, weak-signal regions attached to a brighter TOF estimate.

    Returned C-order source cells are an inspection footprint. A low-signal
    lumen is possible, so these findings carry no exclusion or correction.
    """
    mask = np.asarray(mask)
    signal = np.asarray(signal)
    scores = np.asarray(scores)
    if (mask.dtype != np.bool_ or mask.ndim != 3 or mask.shape != signal.shape
            or signal.dtype.kind not in "fui" or scores.shape != mask.shape
            or scores.dtype.kind != "f"):
        raise ValueError("Expected a native binary prediction, TOF signal and model scores")
    frame = affine(native_affine)
    spacing = np.linalg.norm(frame[:3, :3], axis=0)
    axes = frame[:3, :3] / spacing
    limits = {
        "nativeSignalPercentile": _ATTACHED_DARK_PERCENTILE,
        "minimumCoreRadiusMm": _ATTACHED_CORE_RADIUS_MM,
        "minimumFootprintMm3": _ATTACHED_MINIMUM_VOLUME_MM3,
        "contextRadiusMm": _ATTACHED_CONTEXT_MM,
        "minimumNeighborContrastForegroundIqr": _ATTACHED_ANCHOR_CONTRAST_IQR,
        "maximumWeakForegroundVoxels": _MAX_ATTACHED_WEAK_CELLS,
        "maximumFindings": _MAX_ATTACHED_FINDINGS,
    }
    result = {
        "algorithm": ATTACHED_REVIEW,
        "status": "unavailable",
        "sourceForegroundVoxels": int(np.count_nonzero(mask)),
        "limits": limits,
        "findings": [],
    }
    if not np.allclose(axes.T @ axes, np.eye(3), atol=1e-5):
        return {}, {**result, "reason": "Native TOF lattice is sheared"}
    if spacing.max() > _ATTACHED_CORE_RADIUS_MM / 1.5:
        return {}, {**result, "reason": "Native lattice undersamples the physical screen"}
    foreground = result["sourceForegroundVoxels"]
    if not 100 <= foreground <= _MAX_EXTERIOR_FOREGROUND:
        return {}, {**result, "reason": "Source foreground is too sparse or exceeds review workspace"}

    check()
    points = np.column_stack(np.nonzero(mask))
    image_values = signal[tuple(points.T)]
    if not np.isfinite(image_values).all():
        return {}, {**result, "reason": "Native foreground signal has missing measurements"}
    dark, lo, hi = np.percentile(image_values, [_ATTACHED_DARK_PERCENTILE, 25, 75])
    scale = float(hi - lo)
    if scale <= np.finfo(np.float32).eps * max(abs(float(lo)), abs(float(hi)), 1.0):
        return {}, {**result, "reason": "Native foreground signal has insufficient contrast"}
    weak_cells = image_values <= dark
    if np.count_nonzero(weak_cells) > _MAX_ATTACHED_WEAK_CELLS:
        return {}, {**result, "reason": "Weak native foreground exceeds review workspace"}
    weak = points[weak_cells]
    del points, image_values
    labels = _weak_components(weak, mask.shape, check)
    ordered = np.argsort(labels, kind="stable")
    sizes = np.bincount(labels)
    offsets = np.r_[0, sizes.cumsum()]
    voxel_volume = float(abs(np.linalg.det(frame[:3, :3])))
    component_ids = np.flatnonzero(sizes * voxel_volume >= _ATTACHED_MINIMUM_VOLUME_MM3)
    component_ids = component_ids[np.argsort(-sizes[component_ids], kind="stable")]
    spacing = np.asarray(spacing)
    lattice_shape = np.array(mask.shape)
    footprints = {}
    review_limited = 0
    for label in component_ids:
        check()
        cells = weak[ordered[offsets[label]:offsets[label + 1]]]
        lower, upper = cells.min(axis=0), cells.max(axis=0)
        local_shape = upper - lower + 1
        if np.any((local_shape + 1) * spacing < 2 * _ATTACHED_CORE_RADIUS_MM):
            continue
        if np.prod(local_shape + 2) > _MAX_NEIGHBORHOOD_CELLS:
            review_limited += 1
            continue
        group = np.zeros(local_shape, dtype=bool)
        group[tuple((cells - lower).T)] = True
        distance = ndi.distance_transform_edt(np.pad(group, 1), sampling=spacing)[1:-1, 1:-1, 1:-1]
        core = distance >= _ATTACHED_CORE_RADIUS_MM
        if not core.any():
            continue
        clearance = np.minimum(lower + .5, lattice_shape - upper - .5) * spacing
        if np.any(clearance <= _ATTACHED_CONTEXT_MM):
            continue
        start, context = _bounds(cells, spacing, mask.shape, _ATTACHED_CONTEXT_MM)
        source = signal[context]
        if source.size > _MAX_NEIGHBORHOOD_CELLS or not np.isfinite(source).all():
            review_limited += 1
            continue
        local_mask = mask[context]
        weak_region = np.zeros(source.shape, bool)
        weak_region[tuple((cells - start).T)] = True
        # A physical opening localizes the thick portion and leaves
        # peripheral low-signal filaments outside the inspection footprint.
        opened = group & (ndi.distance_transform_edt(~core, sampling=spacing) < _ATTACHED_CORE_RADIUS_MM)
        stations = np.argwhere(opened) + lower
        volume = len(stations) * voxel_volume
        if volume < _ATTACHED_MINIMUM_VOLUME_MM3:
            continue
        station_in_context = np.ones(source.shape, dtype=bool)
        station_in_context[tuple((stations - start).T)] = False
        ring = ndi.distance_transform_edt(station_in_context, sampling=spacing) <= _ATTACHED_CONTEXT_MM
        touching = ndi.binary_dilation(weak_region) & ring & ~weak_region & local_mask & (source > dark)
        if not touching.any():
            continue
        location = tuple(stations.T)
        measured = signal[location]
        candidate90 = float(np.percentile(measured, 90))
        attachment75 = float(np.percentile(source[touching], 75))
        contrast = (attachment75 - candidate90) / scale
        if contrast < _ATTACHED_ANCHOR_CONTRAST_IQR:
            continue
        values = scores[location]
        if not np.isfinite(values).all() or np.any((values < 0) | (values > 1)):
            review_limited += 1
            continue
        check()
        background = source[~local_mask & ring]
        center = np.array(np.unravel_index(distance.argmax(), distance.shape)) + lower
        footprint = np.ravel_multi_index(location, mask.shape)
        finding_id = f"attached-{int(footprint[0])}"
        footprints[finding_id] = footprint
        result["findings"].append({
            "id": finding_id,
            "centerIjk": center.tolist(),
            "centerRasMm": world(frame, center).tolist(),
            "boundsIjk": [stations.min(axis=0).tolist(), (stations.max(axis=0) + 1).tolist()],
            "voxels": len(stations),
            "volumeMm3": volume,
            "peakCoreRadiusMm": float(distance.max()),
            "foregroundContactVoxels": int(np.count_nonzero(touching)),
            "signalP90": candidate90,
            "backgroundSignalMedian": float(np.median(background)) if background.size else None,
            "attachmentSignalP75": attachment75,
            "attachmentContrastForegroundIqr": contrast,
            "uncalibratedModelScoreMedian": float(np.median(values)),
        })
        if len(footprints) == _MAX_ATTACHED_FINDINGS:
            break
    result.update(
        status="partial" if review_limited or len(footprints) == _MAX_ATTACHED_FINDINGS else "complete",
        foregroundDarkSignalCeiling=float(dark),
        foregroundSignalIqr=scale,
        groupsSkippedForMissingEvidenceOrWorkspace=review_limited,
    )
    result["findings"].sort(key=lambda finding: -finding["voxels"])
    return footprints, result


def review_attached_tof_layer(collection, row, *, check=lambda: None):
    """Bind read-only attachment findings to the native acquisition and scores."""
    from .anatomy_model_registry import source_volume

    provenance = row.get("provenance", {})
    series = collection.series(row["source_series"])
    if (row["kind"] != "vessels" or provenance.get("model") != "vesselfm"
            or row.get("edits") or not row.get("probability")
            or series["modality"] != "TOF" or not source_volume(series)):
        return {}, {"status": "unavailable", "reason": "Select an unedited native TOF vesselFM estimate"}
    if (row["affine"] != series["affine"] or row["data"]["shape"] != series["data"]["shape"]
            or row["probability"]["shape"] != series["data"]["shape"]
            or provenance.get("source") != series["original"]["sha256"]):
        return {}, {"status": "unavailable", "reason": "Source identity or native lattice could not be verified"}
    try:
        collection.require_native_geometry(series)
        collection.require_native_geometry(row)
    except ValueError:
        return {}, {"status": "unavailable", "reason": "The source lattice requires geometry review"}
    check()
    original = collection.path(series["original"]["path"])
    original_key = (str(original), original.stat().st_mtime_ns, series["original"]["sha256"])
    if original_key not in collection._verified:
        if file_hash(original) != series["original"]["sha256"]:
            raise ValueError("Original native TOF checksum differs")
        collection._verified.add(original_key)
    identity = {
        "algorithm": ATTACHED_REVIEW,
        "sourceSeries": series["id"],
        "sourceLayer": row["id"],
        "sourceSha256": series["original"]["sha256"],
        "sourceSignalSha256": series["data"]["sha256"],
        "sourceDataSha256": row["data"]["sha256"],
        "sourceScoreSha256": row["probability"]["sha256"],
        "sourceShape": series["data"]["shape"],
        "sourceAffineRasMm": series["affine"],
        "sourceLabel": row["label_value"],
    }
    # Resolve and verify all array members even for a previously seen identity.
    source_array = collection.array(series["data"])
    predicate_array = collection.array(row["data"])
    scores = collection.array(row["probability"])
    check()
    review_id = digest(identity)
    cached = getattr(collection, "_attached_tof_review_cache", None)
    if cached is not None and cached[0] == review_id:
        return cached[1], cached[2]
    footprints, analysis = attached_vessel_uncertainty(
        predicate_array == row["label_value"], source_array, scores, series["affine"], check=check,
    )
    receipt = {**analysis, **identity, "reviewId": review_id}
    collection._attached_tof_review_cache = (review_id, footprints, receipt)
    return footprints, receipt


def attached_tof_sections(collection, row, cells, finding, *, check=lambda: None):
    """Return small, paired source and review overlays through the native core."""
    series = collection.series(row["source_series"])
    signal = collection.array(series["data"])
    labels = collection.array(row["data"])
    spacing = np.linalg.norm(affine(series["affine"])[:3, :3], axis=0)
    center = np.asarray(finding["centerIjk"], dtype=int)
    lower, upper = np.asarray(finding["boundsIjk"])
    half_extent = np.clip(np.maximum(center - lower, upper - 1 - center) + np.ceil(2 / spacing), 1, 48).astype(int)
    lo = np.maximum(0, center - half_extent)
    hi = np.minimum(np.shape(signal), center + half_extent + 1)

    planes, measurements = [], []
    for axis, index in enumerate(center):
        check()
        section = [slice(a, b) for a, b in zip(lo, hi)]
        section[axis] = int(index)
        section = tuple(section)
        gray = signal[section].T
        estimate = labels[section].T == row["label_value"]
        outline = estimate & ~ndi.binary_erosion(estimate, border_value=1)
        u, v = [dimension for dimension in range(3) if dimension != axis]
        ijk = np.repeat(center[:, None], gray.size, axis=1)
        uu, vv = np.meshgrid(np.arange(lo[u], hi[u]), np.arange(lo[v], hi[v]))
        ijk[u], ijk[v] = uu.ravel(), vv.ravel()
        inside = np.isin(np.ravel_multi_index(ijk, signal.shape), cells).reshape(gray.shape)
        overlay = np.where(inside, 2, outline).astype(np.uint8)
        measurements.append(gray[np.isfinite(gray)])
        planes.append((axis, index, [spacing[v], spacing[u]], gray, overlay))
    available = np.concatenate(measurements)
    low, high = map(float, np.percentile(available, [1, 99]))
    scale = max(high - low, np.finfo(np.float32).eps * max(abs(high), 1.0))
    sections = []
    for axis, index, pixel_spacing, gray, overlay in planes:
        normalized = np.nan_to_num((gray.astype(float) - low) / scale, nan=0, posinf=0, neginf=0)
        pixels = np.clip(normalized * 255, 0, 255).astype(np.uint8)
        sections.append({
            "axis": axis, "index": int(index), "shape": list(gray.shape), "spacingMm": pixel_spacing,
            "image": b64encode(pixels.tobytes()).decode("ascii"),
            "overlay": b64encode(overlay.tobytes()).decode("ascii"),
        })
    return {"window": [low, high], "sections": sections}


def classify_components(mask, signal, scores, native_affine, *, policy=ComponentPolicy()):
    """Classify whole 26-connected components without changing the prediction.

    This is a review proposal. In particular, an isolated bright fragment may
    be a distal vessel or extracranial anatomy and stays available. The rule
    abstains near the image extent and on low-information scans.
    """
    mask = np.asarray(mask)
    signal = np.asarray(signal)
    if mask.dtype != np.bool_ or mask.ndim != 3 or mask.shape != signal.shape:
        raise ValueError("Expected a binary prediction and matched native TOF signal")
    if scores is not None and np.shape(scores) != mask.shape:
        raise ValueError("Model scores must match the native TOF grid")
    frame = affine(native_affine)
    spacing = np.linalg.norm(frame[:3, :3], axis=0)
    axes = frame[:3, :3] / spacing
    if not np.allclose(axes.T @ axes, np.eye(3), atol=1e-5):
        raise ValueError("Component measurements require an orthogonal native TOF grid")
    voxel_volume = float(abs(np.linalg.det(frame[:3, :3])))
    points = np.column_stack(np.nonzero(mask))
    partition = np.zeros(mask.shape, dtype=np.uint8)
    if not len(points):
        raise ValueError("Empty vessel prediction")

    start, slices = _bounds(points, spacing, mask.shape, 0)
    labels, count = ndi.label(mask[slices], structure=CONNECTIVITY)
    memberships = labels[tuple((points - start).T)]
    del labels
    permutation = np.argsort(memberships, kind="stable")
    sizes = np.bincount(memberships, minlength=count + 1)
    groups = np.split(permutation, np.cumsum(sizes[1:-1]))
    network = sizes[memberships] * voxel_volume >= policy.small_volume_mm3
    vessels = signal[tuple(points[network].T)]
    vessels = vessels[np.isfinite(vessels)]
    have_signal_reference = len(vessels) >= 100 and np.ptp(vessels) > 0
    low, very_low = np.percentile(vessels, [
        policy.baseline_signal_percentile, policy.very_low_signal_percentile
    ]) if have_signal_reference else (0, 0)
    coordinates = world(frame, points)
    tree = cKDTree(coordinates[network]) if network.any() else None
    head, head_evidence = _head_support(signal, frame, lambda: None)
    outside_head = None
    if head is not None:
        support, _, stride = head
        lattice = np.minimum((points + stride // 2) // stride, np.array(support.shape) - 1)
        outside_head = ~support[tuple(lattice.T)]
    records = []
    for group in groups:
        cells = points[group]
        extent = (np.ptp(cells, axis=0) + 1) * spacing
        volume = len(cells) * voxel_volume
        source_id = int(np.ravel_multi_index(cells[0], mask.shape))
        component_id = f"component-{source_id}"
        state = RETAINED if volume >= policy.small_volume_mm3 else UNCERTAIN
        reason = "retained prediction" if state == RETAINED else "detached component needs review"
        edge = bool(np.any(
            np.minimum((cells + 0.5) * spacing, (np.array(mask.shape) - 0.5 - cells) * spacing)
            < policy.outer_ring_mm
        ))
        ratio = float(extent.max() / extent.min())
        evidence = None
        score = None
        separation = None
        wholly_outside = outside_head is not None and bool(outside_head[group].all())
        if wholly_outside:
            signal_values = signal[tuple(cells.T)]
            if (not edge and np.isfinite(signal_values).all()
                    and float(signal_values.max()) <= head_evidence["threshold"]):
                state = HIDDEN
                reason = "separate component wholly in dark space beyond the native head envelope"
            else:
                state = UNCERTAIN
                reason = "extracranial or boundary finding needs review"
        if (state == UNCERTAIN and not wholly_outside and volume <= policy.compact_volume_mm3
                and ratio <= policy.compact_axis_ratio):
            evidence = _source_contrast(cells, signal, mask, spacing, policy)
            separation = float(tree.query(coordinates[group])[0].min()) if tree is not None else None
            if scores is not None:
                values = scores[tuple(cells.T)]
                if np.isfinite(values).all() and np.all((values >= 0) & (values <= 1)):
                    score = float(np.median(values))
            if (
                not edge
                and have_signal_reference
                and separation is not None
                and separation >= policy.isolation_mm
                and evidence is not None
                and evidence["foreground_peak"] <= low
                and evidence["peak_contrast_iqr"] <= policy.maximum_peak_contrast
                and (
                    (score is not None and score <= policy.score_median_ceiling)
                    or evidence["foreground_peak"] <= very_low
                )
            ):
                state = HIDDEN
                reason = "isolated compact finding with weak native TOF support"
        partition[tuple(cells.T)] = state
        records.append({
            "id": component_id,
            "source_anchor_cell": source_id,
            "state": state,
            "proposed_state": state,
            "reason": reason,
            "voxels": len(cells),
            "volume_mm3": volume,
            "bounds_ijk": [cells.min(axis=0).tolist(), (cells.max(axis=0) + 1).tolist()],
            "centroid_ras_mm": coordinates[group].mean(axis=0).tolist(),
            "axis_extent_mm": extent.tolist(),
            "near_source_extent": edge,
            "outside_native_head_envelope": wholly_outside,
            "separation_from_larger_candidate_mm": separation,
            "model_score_median": score,
            "source_contrast": evidence,
        })
    counts = {
        state: sum(row["voxels"] for row in records if row["state"] == state)
        for state in (RETAINED, UNCERTAIN, HIDDEN)
    }
    return partition, {
        "policy": asdict(policy),
        "reference_signal_available": bool(have_signal_reference),
        "reference_low_signal": float(low) if have_signal_reference else None,
        "reference_very_low_signal": float(very_low) if have_signal_reference else None,
        "native_head_envelope": head_evidence,
        "component_count": count,
        "foreground_voxels": len(points),
        "retained_voxels": int(counts[RETAINED]),
        "uncertain_voxels": int(counts[UNCERTAIN]),
        "hidden_voxels": int(counts[HIDDEN]),
        "hidden_components": sum(row["state"] == HIDDEN for row in records),
        "new_positive_voxels": 0,
        "geometry": "native TOF, whole-component visibility only",
        "source_prediction_modified": False,
        "scores_calibrated": False,
        "mechanics_admitted": False,
        "components": records,
    }


def review_component_visibility(partition, report, *, restore=(), exclude=()):
    """Move reviewed whole components between uncertain and hidden visibility."""
    if set(restore) & set(exclude):
        raise ValueError("A source component has conflicting review decisions")
    result = np.array(partition, copy=True)
    analysis = {**report, "components": [row.copy() for row in report["components"]]}
    components = {row["id"]: row for row in analysis["components"]}
    actions = [(name, HIDDEN, UNCERTAIN) for name in sorted(set(restore))]
    actions.extend((name, UNCERTAIN, HIDDEN) for name in sorted(set(exclude)))
    for name, previous, current in actions:
        row = components.get(name)
        if row is None or row["state"] != previous:
            raise ValueError(f"Unknown source component or incompatible visibility state: {name}")
        start, stop = np.asarray(row["bounds_ijk"], dtype=int)
        anchor = np.asarray(np.unravel_index(row["source_anchor_cell"], result.shape))
        if (np.any(start < 0) or np.any(stop > result.shape)
                or np.any(anchor < start) or np.any(anchor >= stop)):
            raise ValueError(f"Invalid source component bounds: {name}")
        region = result[tuple(slice(a, b) for a, b in zip(start, stop))]
        labels, _ = ndi.label(region == previous, structure=CONNECTIVITY)
        selected = labels[tuple(anchor - start)]
        if not selected or np.count_nonzero(labels == selected) != row["voxels"]:
            raise ValueError(f"Source component differs from the review receipt: {name}")
        region[labels == selected] = current
        row.update(state=current, reason="explicit component review")
        direction = 1 if current == HIDDEN else -1
        analysis["hidden_voxels"] += direction * row["voxels"]
        analysis["uncertain_voxels"] -= direction * row["voxels"]
        analysis["hidden_components"] += direction
    return result, analysis


def separate_reviewed_components(mask, components):
    """Return two disjoint, exhaustive groups of whole native components."""
    foreground = np.asarray(mask)
    if foreground.ndim != 3 or foreground.dtype != np.bool_ or not components:
        raise ValueError("A native binary foreground and reviewed components are required")
    cells = np.flatnonzero(foreground)
    labels, count = ndi.label(foreground, structure=CONNECTIVITY)
    groups = labels.ravel()[cells]
    sizes = np.bincount(groups, minlength=count + 1)
    anchors = np.full(count + 1, foreground.size, dtype=np.int64)
    np.minimum.at(anchors, groups, cells)
    selected = np.zeros(count + 1, dtype=bool)
    for anchor, size in components.items():
        if type(anchor) is not int or type(size) is not int or not 0 <= anchor < foreground.size:
            raise ValueError("Invalid reviewed component anchor or voxel count")
        label = labels.ravel()[anchor]
        if label == 0 or anchors[label] != anchor or sizes[label] != size:
            raise ValueError(f"Whole native component differs from the reviewed selection: {anchor}")
        selected[label] = True
    compartment = selected[labels]
    main = foreground & ~compartment
    if not np.any(main):
        raise ValueError("The separation leaves no main network")
    return main, compartment
