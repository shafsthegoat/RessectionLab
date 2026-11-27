"""Display normals sampled from a native-grid vascular score estimate."""

import numpy as np
from scipy.ndimage import gaussian_filter, label, map_coordinates
from scipy.spatial import cKDTree

from .anatomy_geometry import _suppress_vascular_scores


# This optional pass shares the bridge with workspace loading and native replay.
# Budget the local grid, point work and KD neighborhoods independently of the
# operative mesh and the original patient arrays already resident in memory.
_WORK_BYTES = 96 * 1024**2
_FILTER_SAMPLES = 280_000_000
_NEIGHBOR_BATCH = 100_000
_NEIGHBOR_TOTAL = 12_000_000
_COMPONENT_RADIUS_MM = .70


def _unit(vectors):
    return vectors / np.maximum(np.linalg.norm(vectors, axis=1, keepdims=True), 1e-20)


def _smoothstep(low, high, values):
    t = np.clip((values - low) / (high - low), 0., 1.)
    return t * t * (3. - 2. * t)


def _isolated_vertices(points, mask, owners, linear, spacing, check):
    components, count = label(mask)
    check()
    isolated = np.ones(len(points), dtype=bool)
    if count <= 1:
        return isolated
    centers = np.argwhere(mask)
    labels = components[tuple(centers.T)]
    point_labels = components[tuple(owners.T)]
    tree = cKDTree(centers @ linear.T)
    positions = points @ linear.T
    footprint = np.minimum(2 * np.ceil(_COMPONENT_RADIUS_MM / spacing) + 1, mask.shape)
    candidates = min(len(centers), int(np.prod(footprint)))
    if candidates > _NEIGHBOR_BATCH:
        return None
    batch = min(4096, _NEIGHBOR_BATCH // max(1, candidates))
    visits = 0
    for first in range(0, len(points), batch):
        check()
        block = slice(first, first + batch)
        neighbors = tree.query_ball_point(positions[block], r=_COMPONENT_RADIUS_MM)
        visits += sum(map(len, neighbors))
        if visits > _NEIGHBOR_TOTAL:
            return None
        isolated[block] = [np.all(labels[cells] == owner)
                           for cells, owner in zip(neighbors, point_labels[block])]
    return isolated


def score_lighting_normals(scores, mask, positions, source_normals, source_cells, affine, *, check=lambda: None):
    """Return native gradient covectors and confidence for the exact score wall.

    Position, ownership, triangulation and the physical collision normals stay
    outside this lighting field. Misshapen local support keeps the mesh normal.
    """
    linear = np.asarray(affine, dtype=float)[:3, :3]
    spacing = np.linalg.norm(linear, axis=0)
    if np.any(spacing <= 0):
        return None
    directions = linear / spacing
    if not np.allclose(directions.T @ directions, np.eye(3), rtol=0., atol=1e-3):
        return None

    sigma = .30 / spacing
    margin = np.ceil(4 * sigma).astype(int) + 2
    low = np.maximum(0, np.floor(positions.min(axis=0)).astype(int) - margin)
    high = np.minimum(mask.shape, np.ceil(positions.max(axis=0)).astype(int) + margin + 1)
    roi = tuple(slice(a, b) for a, b in zip(low, high))
    grid_samples = int(np.prod(high - low + 2))
    grid_bytes = 16 * grid_samples
    point_bytes = 192 * len(positions)
    kernel_widths = np.sum(2 * np.ceil(4 * sigma) + 1)
    if grid_bytes + point_bytes > _WORK_BYTES or 4 * grid_samples * kernel_widths > _FILTER_SAMPLES:
        return None
    local_mask = mask[roi]
    tree_bytes = 96 * np.count_nonzero(local_mask)
    if grid_bytes + point_bytes + tree_bytes > _WORK_BYTES:
        return None
    check()
    owners = np.column_stack(np.unravel_index(source_cells, mask.shape)) - low
    samples = positions.astype(float) - low
    isolated = _isolated_vertices(samples, local_mask, owners, linear, spacing, check)
    if isolated is None:
        return None
    data = np.pad(_suppress_vascular_scores(scores[roi].copy(order="C"), local_mask), 1)
    coordinates = samples.T + 1.
    field = gaussian_filter(data, sigma, mode="constant")
    value = map_coordinates(field, coordinates, order=1)
    derivative = np.empty_like(samples)
    for axis in range(3):
        check()
        order = tuple(int(i == axis) for i in range(3))
        gradient = gaussian_filter(data, sigma, order=order, mode="constant")
        derivative[:, axis] = -map_coordinates(gradient, coordinates, order=1)

    inverse = np.linalg.inv(linear)
    normal = _unit(source_normals @ inverse)
    world_gradient = derivative @ inverse
    gradient_length = np.linalg.norm(world_gradient, axis=1)
    direction = _unit(world_gradient)
    alignment = np.einsum("ij,ij->i", direction, normal)
    if np.median(alignment) < 0:
        derivative *= -1.
    drift = abs(value - .5) / np.maximum(gradient_length, 1e-20)
    confidence = (_smoothstep(.18, .55, gradient_length)
                  * (1. - _smoothstep(.055, .14, drift)) * isolated)
    if not np.any(confidence > 0):
        return None
    return np.column_stack((_unit(derivative), confidence)).astype(np.float32)
