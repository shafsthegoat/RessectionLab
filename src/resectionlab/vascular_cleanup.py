"""Conservative, reversible review of native angiographic vessel predictions."""

from dataclasses import asdict, dataclass

import numpy as np
from scipy import ndimage as ndi
from scipy.spatial import cKDTree

from .anatomy_siam_display import _head_support
from .patient_anatomy import affine, world


RETAINED, UNCERTAIN, HIDDEN = 1, 2, 3
CONNECTIVITY = np.ones((3, 3, 3), dtype=bool)


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


def restore_hidden_components(partition, report, component_ids):
    """Restore entire reviewed components in a fresh visibility partition."""
    result = np.array(partition, copy=True)
    analysis = {**report, "components": [row.copy() for row in report["components"]]}
    components = {row["id"]: row for row in analysis["components"]}
    for name in sorted(set(component_ids)):
        row = components.get(name)
        if row is None or row["state"] != HIDDEN:
            raise ValueError(f"Unknown or visible source component: {name}")
        start, stop = np.asarray(row["bounds_ijk"], dtype=int)
        anchor = np.asarray(np.unravel_index(row["source_anchor_cell"], result.shape))
        if (np.any(start < 0) or np.any(stop > result.shape)
                or np.any(anchor < start) or np.any(anchor >= stop)):
            raise ValueError(f"Invalid source component bounds: {name}")
        region = result[tuple(slice(a, b) for a, b in zip(start, stop))]
        labels, _ = ndi.label(region == HIDDEN, structure=CONNECTIVITY)
        selected = labels[tuple(anchor - start)]
        if not selected or np.count_nonzero(labels == selected) != row["voxels"]:
            raise ValueError(f"Hidden source component differs from the review receipt: {name}")
        region[labels == selected] = UNCERTAIN
        row.update(state=UNCERTAIN, reason="explicit component review")
        analysis["hidden_voxels"] -= row["voxels"]
        analysis["uncertain_voxels"] += row["voxels"]
        analysis["hidden_components"] -= 1
    return result, analysis
