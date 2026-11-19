"""Acquired grayscale for route-local display, sampled independently of labels."""

from itertools import product

import numpy as np
from scipy import ndimage as ndi

from .anatomy_model_registry import STRUCTURAL_MRI, ct_intensities, source_volume


MAX_TEXTURE_EDGE = 256
MAX_TEXTURE_VOXELS = 2_000_000


def _texture_grid(grid, frame, native):
    """Fit the native sampling bandwidth along each route axis within the display budget."""
    intervals = np.asarray(frame["shape"]) - 1
    # A native axis can contribute up to half a cycle per voxel. Project all
    # three onto a local step, including oblique or sheared acquisition axes.
    native_steps = np.linalg.solve(native[:3, :3], grid[:3, :3])
    # The renderer interpolates this grid a second time. Extra samples preserve
    # native interpolation detail when the route lattice falls between voxel centres.
    desired = 1.5 * intervals * np.abs(native_steps).sum(axis=0)

    def intervals_at(coarsening):
        return np.clip(np.ceil(desired / coarsening), 1, MAX_TEXTURE_EDGE - 1).astype(int)

    counts = intervals_at(1.)
    if np.prod(counts + 1) > MAX_TEXTURE_VOXELS:
        # Keep relative pitch while fitting integer dimensions and both endpoints.
        lower, upper = 1., float(desired.max())
        for _ in range(20):
            mid = np.sqrt(lower) * np.sqrt(upper)
            trial = intervals_at(mid)
            if np.prod(trial + 1) <= MAX_TEXTURE_VOXELS:
                upper = mid
            else:
                lower = mid
        counts = intervals_at(upper)
    scale = intervals / counts
    return counts + 1, scale * frame["spacingMm"], grid @ np.diag([*scale, 1.])


def _source_overview(data, native, frame, low, high):
    """Show the full acquired cross-section containing both saved route endpoints."""
    ray = np.asarray(frame["basisRas"][2])
    target = np.asarray(frame["targetRasMm"])
    axis = np.eye(3)[np.argmin(np.abs(ray))]
    normal = axis - ray * (axis @ ray)
    normal /= np.linalg.norm(normal)
    up = np.array([0., 0., 1.]) if abs(normal[2]) < .9 else np.array([0., 1., 0.])
    right = np.cross(up, normal)
    right /= np.linalg.norm(right)
    down = np.cross(right, normal)
    axes = np.stack([right, down])
    corners = np.array(list(product(*[(0, n - 1) for n in data.shape])))
    xy = (corners @ native[:3, :3].T + native[:3, 3] - target) @ axes.T
    lo, hi = xy.min(axis=0), xy.max(axis=0)
    pitch = max(float((hi - lo).max()) / 159, 0.25)
    width, height = np.maximum(2, 1 + np.ceil((hi - lo) / pitch).astype(int))
    y, x = np.indices((height, width))
    ras = target[:, None, None] + right[:, None, None] * (lo[0] + x*pitch) + down[:, None, None] * (lo[1] + y*pitch)
    vox = np.linalg.solve(native[:3, :3], (ras - native[:3, 3, None, None]).reshape(3, -1))
    signal = ndi.map_coordinates(data, vox.reshape(3, height, width), order=1, prefilter=False,
                                 mode="constant", cval=np.nan, output=np.float32)
    grey = np.zeros((height, width), dtype=np.uint8)
    covered = np.isfinite(signal)
    grey[covered] = 1 + np.rint(254 * np.clip((signal[covered] - low) / (high - low), 0, 1)).astype(np.uint8)
    return {"signal": grey, "shape": [int(height), int(width)], "spacingMm": pitch,
            "rightRas": right.tolist(), "upRas": (-down).tolist(),
            "targetPx": (-lo / pitch).tolist(), "entryPx": ((axes @ ray * frame["entryDepthMm"] - lo) / pitch).tolist()}


def operative_intensity(collection, target_source, grid, frame, *, check):
    target = collection.series(target_source)
    sources = sorted(collection.document["series"], key=lambda s: (
        s["modality"] not in STRUCTURAL_MRI, s["id"] != target_source,
        s["id"] != collection.document["reference_series"], s["id"],
    ))
    for source in sources:
        check()
        modality = source["modality"]
        ct = modality in {"CT", "CTA"}
        if (source["timepoint"] != target["timepoint"] or (not ct and modality not in STRUCTURAL_MRI)
                or not source_volume(source)):
            continue
        try:
            native = collection.patient_affine(source)
            if ct and not ct_intensities(collection, source):
                continue
            data = collection.array(source["data"])
        except InterruptedError:
            raise
        except (ValueError, OSError):
            continue
        shape, spacing, texture_grid = _texture_grid(grid, frame, native)
        transform = np.linalg.solve(native, texture_grid)
        values = np.empty(shape, np.float32)
        for x in range(0, shape[0], 8):
            check()
            # Linear interpolation needs native sample centres on both sides.
            # Beyond those centres, distinct sentinel 0 leaves the display neutral.
            ndi.affine_transform(data, transform[:3, :3], offset=transform[:3, 3] + x * transform[:3, 0],
                                 output=values[x:x+8], order=1, mode="constant", cval=np.nan, prefilter=False)
        covered = np.isfinite(values)
        tissue = values[covered & ((values > -800) if ct else (values != 0))]
        if len(tissue) < 8:
            continue
        low, high = np.percentile(tissue, (1, 99))
        if not high > low:
            continue
        signal = np.zeros(shape, dtype=np.uint8)
        scaled = np.clip((values[covered].astype(np.float64) - low) / (high - low), 0, 1)
        signal[covered] = 1 + np.rint(254 * scaled).astype(np.uint8)
        return {"shape": shape.tolist(), "origin": frame["origin"], "spacingMm": spacing.tolist(),
                "signal": signal, **({"overview": _source_overview(data, native, frame, low, high)}
                                   if "entryDepthMm" in frame else {}),
                "unknownValue": 0, "sourceSeries": source["id"], "modality": modality,
                "sourceSha256": source["original"]["sha256"], "dataSha256": source["data"]["sha256"],
                "window": [float(low), float(high)], "windowUnits": "HU" if ct else "source intensity",
                "windowMethod": "Local 1st–99th percentiles excluding CT air or MRI zero background",
                "coveredVoxels": int(covered.sum()), "sourceShape": source["data"]["shape"],
                "sourceVoxelToPatientRasMm": native.tolist()}
    return None
