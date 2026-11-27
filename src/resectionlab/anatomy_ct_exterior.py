"""Identify an isolated posterior sheet for native CT volume presentation."""

import numpy as np
from scipy import ndimage as ndi

from .anatomy_model_registry import ct_intensities


SCHEMA = "native-ct-exterior-sheet-v1"
FOREGROUND_HU = -500
POSTERIOR_CLEARANCE_MM = 8.
_NEIGHBORS = np.ones((3, 3, 3), dtype=bool)
_ANCHORS = {"brain", "artery", "vessel", "vessels", "vein", "sinus"}


def _cardinal_axes(affine):
    linear = np.asarray(affine, dtype=float)[:3, :3]
    native_spacing = np.linalg.norm(linear, axis=0)
    if np.any(native_spacing <= 0):
        return None
    directions = linear / native_spacing
    axes = np.abs(directions).argmax(axis=1)
    main = directions[np.arange(3), axes]
    if len(set(axes)) != 3 or np.any(np.abs(main) < .995):
        return None
    return axes, main < 0, native_spacing[axes]


def _ras_order(array, axes, reversed_axes):
    return array.transpose(axes)[tuple(slice(None, None, -1 if reverse else 1) for reverse in reversed_axes)]


def _source_labels(collection, scan):
    """Gather physical guards independent of the currently visible layers."""
    for row in collection.document["layers"]:
        if (row["source_series"] != scan["id"] or row["data"]["shape"] != scan["data"]["shape"]
                or row["affine"] != scan["affine"]):
            continue
        # An exact acquired-lattice annotation still protects its occupied
        # cells when only index-based source inspection is available.
        yield row


def _posterior_limit(collection, rows, axes, reversed_axes, spacing):
    groups = ([row for row in rows if row["kind"] == "brain"],
              [row for row in rows if row["kind"] in _ANCHORS - {"brain"}])
    for group in groups:
        centers = []
        for row in group:
            labels = _ras_order(collection.array(row["data"]), axes, reversed_axes)
            counts = np.count_nonzero(labels[::3, :, ::3] == row["label_value"], axis=(0, 2))
            if counts.sum() >= 16:
                centers.append(int(np.searchsorted(np.cumsum(counts), counts.sum() / 2)))
        if centers:
            return min(len(counts) // 2, max(centers) - int(np.ceil(POSTERIOR_CLEARANCE_MM / spacing[1])))
    return 0


def ct_exterior_sheet(collection, scan, *, check):
    """Return a compact display mask after source and anatomical vetoes.

    The candidate must span the lateral acquisition boundaries, reach the
    posterior and superior faces, have low peak attenuation and be separate
    from the remaining foreground. Nearby native labels always veto it.
    """
    if not ct_intensities(collection, scan):
        return None
    try:
        collection.require_native_geometry(scan)
    except ValueError:
        return None
    orientation = _cardinal_axes(scan["affine"])
    if orientation is None:
        return None
    axes, reversed_axes, spacing = orientation
    rows = list(_source_labels(collection, scan))
    if not any(row["kind"] in _ANCHORS for row in rows):
        return None
    check()
    stop = _posterior_limit(collection, rows, axes, reversed_axes, spacing)
    if stop < 4:
        return None
    raw = _ras_order(collection.array(scan["data"]), axes, reversed_axes)
    # This display-only qualification should not delay large native imports.
    posterior = raw[:, :stop, :]
    if raw.shape[0] * spacing[0] < 85. or posterior.size > 20_000_000:
        return None
    check()
    components, number = ndi.label(posterior >= FOREGROUND_HU, structure=_NEIGHBORS)
    if number == 0:
        return None
    counts = np.bincount(components.ravel())
    adjoining = np.unique(components[:, -1, :])
    adjoining = adjoining[adjoining != 0]
    if not len(adjoining):
        return None
    patient = adjoining[np.argmax(counts[adjoining])]
    candidates = []
    for index, bounds in enumerate(ndi.find_objects(components), 1):
        if (bounds is None or counts[index] * spacing.prod() < 600.
                or counts[index] > .2 * counts[patient]
                or bounds[0].start != 0 or bounds[0].stop != raw.shape[0]
                or bounds[1].start != 0 or bounds[1].stop > stop - 2
                or bounds[2].stop != raw.shape[2]):
            continue
        extent = np.array([axis.stop - axis.start for axis in bounds]) * spacing
        if extent[1] > 60. or extent[2] < 16.:
            continue
        core = components[bounds] == index
        signal = posterior[bounds][core]
        if signal.max() > 600. or np.count_nonzero(signal >= 110.) < .2 * counts[index]:
            continue
        # Require a broad signal sheet over the transverse acquisition, beyond
        # the two boundary touches. A fine cable or point chain cannot qualify.
        profile = np.count_nonzero(core, axis=(1, 2)) * spacing[1] * spacing[2]
        thin, typical = np.percentile(profile, (10, 50))
        if thin < 4. or typical < 10.:
            continue
        candidates.append((index, bounds))
    if len(candidates) != 1:
        return None

    index, bounds = candidates[0]
    padding = np.ceil(2. / spacing).astype(int)
    lower = np.maximum(0, [axis.start for axis in bounds] - padding)
    upper = np.minimum(posterior.shape, [axis.stop for axis in bounds] + padding)
    cropped = tuple(slice(a, b) for a, b in zip(lower, upper, strict=True))
    labels = components[cropped]
    core = labels == index
    footprint = ndi.binary_dilation(core, structure=_NEIGHBORS, iterations=1)
    if np.any(footprint & (labels > 0) & ~core):
        return None
    guard = ndi.distance_transform_edt(~core, sampling=spacing) <= 2.
    if np.any(guard & (labels == patient)):
        return None
    by_array = {}
    for row in rows:
        _, values = by_array.setdefault(row["data"]["path"], (row["data"], set()))
        values.add(row["label_value"])
    for reference, values in by_array.values():
        check()
        labels = _ras_order(collection.array(reference), axes, reversed_axes)[cropped]
        if np.any(np.isin(labels[guard], list(values))):
            return None
    # Convert the derivative back to the original XYZ native voxel order.
    start = np.where(reversed_axes, np.asarray(raw.shape) - upper, lower)
    native_order = np.argsort(axes)
    mask = np.ascontiguousarray(_ras_order(footprint, [0, 1, 2], reversed_axes).transpose(native_order), dtype=np.uint8)
    return {"schema": SCHEMA, "offset": start[native_order].tolist(), "mask": mask}
