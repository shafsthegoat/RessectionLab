"""Shared vessel measurements for the vascular rendering and reconstruction gates.

Every function measures supplied arrays. None of them infers a missing
reference: overlap and topology agreement need an independent annotation.
"""

from __future__ import annotations

import numpy as np
from scipy import ndimage as ndi
from skimage.morphology import skeletonize

FULL = np.ones((3, 3, 3), bool)


def _binary(mask):
    mask = np.asarray(mask)
    if mask.ndim != 3 or not all(mask.shape):
        raise ValueError('A non-empty three-dimensional mask is required')
    return mask.astype(bool)


def _matched(first, second):
    first, second = _binary(first), _binary(second)
    if first.shape != second.shape:
        raise ValueError('Compared masks must share one grid')
    return first, second


def dice(predicted, reference):
    predicted, reference = _matched(predicted, reference)
    total = int(predicted.sum()) + int(reference.sum())
    return 2 * int((predicted & reference).sum()) / total if total else None


def centerline(mask):
    """Lee skeleton that keeps one voxel in any component the thinning erases.

    Lee thinning removes some small symmetric components entirely. Each such
    component keeps its deepest voxel so that it still counts in centerline
    measurements.
    """
    mask = _binary(mask)
    line = skeletonize(mask).astype(bool)
    labels, count = ndi.label(mask, FULL)
    if count:
        missing = np.setdiff1d(np.arange(1, count + 1), np.unique(labels[line]))
        if len(missing):
            depth = ndi.distance_transform_edt(mask)
            for position in ndi.maximum_position(depth, labels, missing):
                line[tuple(int(value) for value in position)] = True
    return line


def cl_dice(predicted, reference):
    """Centerline Dice: harmonic mean of topology precision and sensitivity."""
    predicted, reference = _matched(predicted, reference)
    if not predicted.any() or not reference.any():
        return None
    predicted_line, reference_line = centerline(predicted), centerline(reference)
    precision = (predicted_line & reference).sum() / max(int(predicted_line.sum()), 1)
    sensitivity = (reference_line & predicted).sum() / max(int(reference_line.sum()), 1)
    total = precision + sensitivity
    return float(2 * precision * sensitivity / total) if total else 0.0


def betti(mask):
    """Betti numbers of the union of closed voxel cubes (26-connected foreground)."""
    mask = np.pad(_binary(mask), 1)
    cubes = int(mask.sum())
    if not cubes:
        return {'b0': 0, 'b1': 0, 'b2': 0}

    def joined(axes):
        cells = mask
        for axis in axes:
            lower = [slice(None)] * 3
            upper = [slice(None)] * 3
            lower[axis], upper[axis] = slice(None, -1), slice(1, None)
            cells = cells[tuple(lower)] | cells[tuple(upper)]
        return int(cells.sum())

    faces = sum(joined([axis]) for axis in range(3))
    edges = sum(joined(pair) for pair in ([0, 1], [0, 2], [1, 2]))
    vertices = joined([0, 1, 2])
    euler = vertices - edges + faces - cubes
    b0 = int(ndi.label(mask, FULL)[1])
    # Enclosed cavities are six-connected background regions other than the exterior.
    b2 = int(ndi.label(~mask)[1]) - 1
    return {'b0': b0, 'b1': b0 + b2 - euler, 'b2': b2}


def betti_error(predicted, reference):
    predicted, reference = _matched(predicted, reference)
    found, wanted = betti(predicted), betti(reference)
    return {
        'predicted': found, 'reference': wanted,
        'b0_error': abs(found['b0'] - wanted['b0']), 'b1_error': abs(found['b1'] - wanted['b1']),
    }


def floating_components(mask, roots=None, spacing=(1.0, 1.0, 1.0)):
    """Count 26-connected components that do not reach a root.

    Without roots the largest component is treated as the tree. The count and
    the detached volume are reported, never removed.
    """
    mask = _binary(mask)
    labels, count = ndi.label(mask, FULL)
    if not count:
        return {'components': 0, 'floating': 0, 'floating_voxels': 0, 'floating_mm3': 0.0, 'floating_fraction': 0.0}
    sizes = np.bincount(labels.ravel())[1:]
    if roots is None:
        rooted = {int(sizes.argmax()) + 1}
    else:
        roots = _binary(roots)
        if roots.shape != mask.shape:
            raise ValueError('Roots and mask must share one grid')
        rooted = set(np.unique(labels[roots & mask]).tolist()) - {0}
        if not rooted:
            raise ValueError('No root voxel lies inside the mask')
    loose = [index + 1 for index in range(count) if index + 1 not in rooted]
    voxels = int(sum(sizes[index - 1] for index in loose))
    return {
        'components': int(count), 'floating': len(loose), 'floating_voxels': voxels,
        'floating_mm3': float(voxels * np.prod(spacing)), 'floating_fraction': voxels / int(sizes.sum()),
    }


def surface_quality(points, faces):
    """Dihedral turn across shared edges, and edge manifoldness, in physical space."""
    points, faces = np.asarray(points, float), np.asarray(faces, np.int64)
    if not len(faces):
        return {'faces': 0, 'normal_turn_degrees_p50': None, 'normal_turn_degrees_p95': None,
                'open_edges': 0, 'non_manifold_edges': 0}
    a, b, c = points[faces].transpose(1, 0, 2)
    normals = np.cross(b - a, c - a)
    normals /= np.maximum(np.linalg.norm(normals, axis=1), 1e-20)[:, None]
    edges = np.sort(faces[:, [0, 1, 1, 2, 2, 0]].reshape(-1, 2), axis=1)
    _, mapped, counts = np.unique(edges, axis=0, return_inverse=True, return_counts=True)
    owner = np.argsort(mapped.ravel(), kind='stable') // 3
    start = (np.cumsum(counts) - counts)[counts == 2]
    turn = np.degrees(np.arccos(np.clip(np.einsum('ij,ij->i', normals[owner[start]], normals[owner[start + 1]]), -1, 1)))
    return {
        'faces': int(len(faces)),
        'normal_turn_degrees_p50': float(np.percentile(turn, 50)) if len(turn) else None,
        'normal_turn_degrees_p95': float(np.percentile(turn, 95)) if len(turn) else None,
        'open_edges': int(np.count_nonzero(counts == 1)),
        'non_manifold_edges': int(np.count_nonzero(counts > 2)),
    }


def source_deviation(points, normals, volume, voxel_to_world, *, reach_mm=None, step_mm=None, min_drop=None):
    """Distance in millimetres from each wall point to the acquired lumen edge.

    Angiographic lumen is brighter than its surroundings. The acquired signal
    is sampled along the outward physical normal, and the edge is the steepest
    fall in that profile. A point with no clear fall inside the reach counts as
    a full-reach miss, so a wall that leaves the scan cannot score well.
    The contrast floor comes from the whole volume, not from the surface.
    """
    points, normals = np.asarray(points, float), np.asarray(normals, float)
    frame = np.asarray(voxel_to_world, float)
    if points.shape != normals.shape or points.ndim != 2 or points.shape[1] != 3 or frame.shape != (4, 4):
        raise ValueError('Matching surface points, normals and a 4x4 voxel-to-world matrix are required')
    spacing = np.linalg.norm(frame[:3, :3], axis=0)
    voxel_mm = float(spacing.min())
    reach = 2 * voxel_mm if reach_mm is None else float(reach_mm)
    step = voxel_mm / 10 if step_mm is None else float(step_mm)
    empty = {'points': int(len(points)), 'resolved_fraction': None, 'mm_p50': None, 'mm_p95': None,
             'voxels_p50': None, 'voxels_p95': None, 'voxel_mm': voxel_mm, 'reach_mm': reach}
    if not len(points):
        return empty
    volume = np.asarray(volume)
    if min_drop is None:
        sample = volume[::4, ::4, ::4]
        min_drop = 0.05 * float(np.percentile(sample, 99.9) - np.percentile(sample, 50))
    inverse = np.linalg.inv(frame)
    unit = normals / np.maximum(np.linalg.norm(normals, axis=1), 1e-20)[:, None]
    voxel = points @ inverse[:3, :3].T + inverse[:3, 3]
    # Physical steps along the world normal map to index steps of varying length.
    direction = unit @ inverse[:3, :3].T
    span = np.abs(direction).max() * reach
    shape = np.asarray(volume.shape)
    lower = np.clip(np.floor(voxel.min(axis=0) - span - 4).astype(int), 0, shape - 1)
    upper = np.clip(np.ceil(voxel.max(axis=0) + span + 5).astype(int), lower + 1, shape)
    # Cubic sampling needs a prefiltered copy, so bound it to the surface.
    local = np.asarray(volume[tuple(slice(a, b) for a, b in zip(lower, upper))], np.float32)
    offsets = np.arange(-reach, reach + step / 2, step)
    samples = (voxel - lower)[:, None, :] + offsets[None, :, None] * direction[:, None, :]
    profile = ndi.map_coordinates(local, samples.reshape(-1, 3).T, order=3, mode='nearest').reshape(len(points), -1)
    fall = -np.gradient(profile, step, axis=1)
    steepest = fall.argmax(axis=1)
    rows = np.arange(len(points))
    # Signal lost across the edge: brightest sample inside it minus dimmest outside it.
    inside = np.maximum.accumulate(profile, axis=1)[rows, steepest]
    outside = np.minimum.accumulate(profile[:, ::-1], axis=1)[:, ::-1][rows, steepest]
    resolved = (steepest > 0) & (steepest < len(offsets) - 1) & (fall[rows, steepest] > 0) & (inside - outside >= min_drop)
    distance = np.where(resolved, np.abs(offsets[steepest]), reach)
    p50, p95 = np.percentile(distance, [50, 95])
    return {**empty, 'resolved_fraction': float(resolved.mean()), 'mm_p50': float(p50), 'mm_p95': float(p95),
            'voxels_p50': float(p50 / voxel_mm), 'voxels_p95': float(p95 / voxel_mm), 'min_drop': float(min_drop)}


def image_sharpness(image, *, edges=4000, reach=16, seed=0):
    """Edge rise width and edge energy of a rendered frame.

    Rise width is the median 10 to 90 percent distance, in pixels, across the
    strongest luminance edges. Compare frames only at a matched display scale.
    """
    image = np.asarray(image, np.float32)
    if image.ndim == 3:
        image = image[..., :3] @ np.array([0.2126, 0.7152, 0.0722], np.float32)
    if image.ndim != 2 or min(image.shape) < 4 * reach:
        raise ValueError('A two-dimensional frame larger than the edge window is required')
    if image.max() > 1.5:
        image = image / 255
    gy, gx = ndi.sobel(image, 0) / 8, ndi.sobel(image, 1) / 8
    strength = np.hypot(gx, gy)
    inner = np.zeros_like(strength, bool)
    inner[reach:-reach, reach:-reach] = True
    peak = (strength == ndi.maximum_filter(strength, 3)) & inner & (strength > 0.02)
    rows, cols = np.nonzero(peak)
    if not len(rows):
        return {'edges': 0, 'rise_px_p50': None, 'rise_px_p90': None, 'edge_energy': float((strength ** 2).mean())}
    order = np.argsort(strength[rows, cols])[::-1][: max(edges * 4, edges)]
    if len(order) > edges:
        order = np.random.default_rng(seed).choice(order, edges, replace=False)
    rows, cols = rows[order], cols[order]
    scale = strength[rows, cols]
    offsets = np.arange(-reach, reach + 0.25, 0.5)
    r = rows[:, None] + offsets[None] * (gy[rows, cols] / scale)[:, None]
    c = cols[:, None] + offsets[None] * (gx[rows, cols] / scale)[:, None]
    profile = ndi.map_coordinates(image, [r.ravel(), c.ravel()], order=1, mode='nearest').reshape(len(rows), -1)
    middle = len(offsets) // 2
    widths = []
    for line in profile:
        # Walk outward from the edge while the profile still changes.
        low = middle
        while low > 0 and line[low] - line[low - 1] > 1e-3:
            low -= 1
        high = middle
        while high < len(line) - 1 and line[high + 1] - line[high] > 1e-3:
            high += 1
        span = line[high] - line[low]
        # A walk that reaches the window end has no measurable rise.
        if span < 0.04 or low == 0 or high == len(line) - 1:
            continue
        segment = line[low: high + 1]
        ten = np.interp(line[low] + 0.1 * span, segment, offsets[low: high + 1])
        ninety = np.interp(line[low] + 0.9 * span, segment, offsets[low: high + 1])
        widths.append(ninety - ten)
    return {
        'edges': len(widths),
        'rise_px_p50': float(np.percentile(widths, 50)) if widths else None,
        'rise_px_p90': float(np.percentile(widths, 90)) if widths else None,
        'edge_energy': float((strength ** 2).mean()),
    }
