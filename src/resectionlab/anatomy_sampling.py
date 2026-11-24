"""Categorical sampling on physical display fields, bounded by native acquisition and inference support."""

from itertools import product

import numpy as np

from .anatomy_evidence import inference_bounds
from .patient_anatomy import world


def native_field_bounds(native, native_shape, field, field_shape):
    """Native samples that bracket a display field without changing its spacing."""
    corners = list(product(*[(-.5, size - .5) for size in field_shape]))
    points = np.round(world(np.linalg.solve(native, field), corners) + .5, 7)
    shape = np.asarray(native_shape)
    start = np.floor(points.min(axis=0)).astype(int)
    stop = np.ceil(points.max(axis=0)).astype(int)
    partial = bool(np.any(start < 0) or np.any(stop > shape))
    return np.clip(start, 0, shape), np.clip(stop, 0, shape), partial


def _coordinate(matrix, axis, coords):
    return (matrix[axis, 0] * coords[0] + matrix[axis, 1] * coords[1]
            + matrix[axis, 2] * coords[2] + matrix[axis, 3])


def _sample_values(data, native, source_affine, source_shape, grid, shape, bounds, slices, check):
    """Pull a categorical grid once, preserving source-cell and acquisition support."""
    layer_from_field = np.linalg.solve(native, grid)
    source_from_layer = np.linalg.solve(source_affine, native)
    source_from_field = np.linalg.solve(source_affine, grid)
    low, high = map(np.asarray, bounds)
    # Omit redundant acquisition tests only when whole label cells are
    # contained. Cropped or tilted labels can protrude past acquired voxels.
    corners = world(source_from_layer, list(product(*zip(low - .5, high + .5))))
    contained = ((np.array_equal(native, source_affine) and np.all(high < source_shape))
                 or (np.all(corners > -.5) and np.all(corners < source_shape - .5)))
    axial = None
    if slices is not None:
        axial = np.zeros(data.shape[2], dtype=bool)
        axial[list(slices)] = True

    values = np.zeros(shape, dtype=data.dtype)
    covered = np.zeros(shape, dtype=bool)
    y = np.arange(shape[1])[None, :, None]
    z = np.arange(shape[2])[None, None, :]
    # Bound intermediate coordinates independently of trajectory and study size.
    slab_size = max(1, 262144 // (shape[1] * shape[2]))
    for start in range(0, shape[0], slab_size):
        check()
        stop = min(start + slab_size, shape[0])
        coords = (np.arange(start, stop)[:, None, None], y, z)
        cells = [np.floor(_coordinate(layer_from_field, axis, coords) + .5).astype(np.int32) for axis in range(3)]
        valid = np.ones((stop - start, *shape[1:]), dtype=bool)
        for axis, cell in enumerate(cells):
            valid &= (cell >= low[axis]) & (cell <= high[axis])
        if not valid.any():
            continue
        if not contained:
            for axis, size in enumerate(source_shape):
                point = _coordinate(source_from_field, axis, coords)
                center = _coordinate(source_from_layer, axis, cells)
                valid &= (point >= -.5) & (point < size - .5) & (center >= -.5) & (center < size - .5)
        for axis, cell in enumerate(cells):
            np.clip(cell, 0, data.shape[axis] - 1, out=cell)
        if axial is not None:
            valid &= axial[cells[2]]
        covered[start:stop] = valid
        values[start:stop] = data[tuple(cells)]
    return values, covered


def sample_layer(collection, row, grid, shape, check, cache, *, frames=None):
    source = collection.series(row["source_series"])
    native, source_affine = frames if frames is not None else (collection.patient_affine(row), collection.patient_affine(source))
    data = collection.array(row["data"], check=check)
    bounds = inference_bounds(row) or ((0, 0, 0), tuple(size - 1 for size in data.shape))
    slices = row.get("coverage_fields", {}).get("supported_axial_indices")
    if slices is not None:
        if (not isinstance(slices, list)
                or any(type(index) is not int or not 0 <= index < data.shape[2] for index in slices)):
            raise ValueError("Invalid supported axial slice indices")
        slices = tuple(slices)
    source_shape = tuple(source["data"]["shape"])
    key = (row["data"]["sha256"], native.tobytes(), source_affine.tobytes(), source_shape, bounds, slices)
    if cache.get("key") != key:
        # A segmentation can represent many separately attributed tissues in
        # one categorical array. Keep just the current native grid in memory.
        cache.clear()
        values, covered = _sample_values(data, native, source_affine, np.asarray(source_shape),
                                         grid, shape, bounds, slices, check)
        cache.update(key=key, values=values, covered=covered)
    return (cache["values"] == row["label_value"]) & cache["covered"], cache["covered"]
