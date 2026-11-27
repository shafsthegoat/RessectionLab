"""An additional physical-space pass for supplied, indexed CTA arteries."""

import numpy as np
from scipy import sparse
from scipy.spatial import cKDTree

from .surface_fairing import (
    _vascular_displacement_limits,
    _vascular_normal_balance,
    intersection_vertices,
)

POLICY = 'source-bound-publisher-cta-fairing-v2'


def _face_geometry(points, triangles):
    corners = points[triangles]
    normals = np.cross(corners[:, 1] - corners[:, 0], corners[:, 2] - corners[:, 0])
    area = np.linalg.norm(normals, axis=1)
    return corners.mean(axis=1), normals / np.maximum(area, 1e-12)[:, None], area


def _guided_vertices(points, triangles, fixed, *, cancelled):
    # Adjacent faces guide a small re-positioning of the existing vertices.
    # Keep irregular boundaries and isolated small components at their prior position.
    edge_uses = np.sort(triangles[:, [0, 1, 1, 2, 2, 0]].reshape(-1, 2), axis=1)
    edges, inverse, uses = np.unique(edge_uses, axis=0, return_inverse=True, return_counts=True)
    starts = (np.cumsum(uses) - uses)[uses == 2]
    left, right = (np.argsort(inverse)[np.column_stack((starts, starts + 1))] // 3).T
    _, normals, area = _face_geometry(points, triangles)
    alignment = np.einsum('ij,ij->i', normals[left], normals[right])
    scale = np.sqrt(np.minimum(area[left], area[right]) / np.maximum(np.maximum(area[left], area[right]), 1e-12))
    weights = (0.1 + 0.9 * ((1 + np.clip(alignment, -1, 1)) * 0.5) ** 2) * scale
    face_graph = sparse.csr_matrix(
        (np.r_[weights, weights], (np.r_[left, right], np.r_[right, left])), shape=(len(triangles),) * 2,
    )
    face_degree = np.asarray(face_graph.sum(axis=1)).ravel()
    guide = normals.copy()
    for _ in range(3):
        guide = (guide + face_graph @ guide) / (1 + face_degree)[:, None]
        guide /= np.maximum(np.linalg.norm(guide, axis=1), 1e-12)[:, None]

    a, b = edges.T
    graph = sparse.csr_matrix((np.ones(len(edges) * 2), (np.r_[a, b], np.r_[b, a])), shape=(len(points),) * 2)
    vertex_degree = np.asarray(graph.sum(axis=1)).ravel()
    operator = sparse.diags(1 / np.maximum(1, vertex_degree)) @ graph
    _, components = sparse.csgraph.connected_components(graph, directed=False)
    fixed |= np.bincount(components)[components] < 30
    fixed[edges[uses != 2].ravel()] = True

    incident = triangles.ravel()
    mass = np.maximum(1e-12, np.bincount(incident, weights=np.repeat(area, 3), minlength=len(points)))
    working = points.copy()
    for _ in range(4):
        if cancelled():
            raise InterruptedError('Publisher vessel refinement cancelled')
        centers = working[triangles].mean(axis=1)
        heights = np.einsum('nki,ni->nk', centers[:, None, :] - working[triangles], guide)
        weighted = (guide[:, None, :] * (heights * area[:, None])[:, :, None]).reshape(-1, 3)
        step = np.column_stack([np.bincount(incident, weights=col, minlength=len(points)) for col in weighted.T])
        working += 0.8 * step / mass[:, None]
        working[fixed] = points[fixed]
    result = _vascular_normal_balance(points, working, triangles, operator)
    result[fixed] = points[fixed]
    return result


def refine_publisher_vessel(source, prior, triangles, affine, mask, origin, shape, venous_cells, *, cancelled):
    linear = affine[:3, :3]
    inverse = np.linalg.inv(linear)
    spacing = np.linalg.norm(linear, axis=0)
    if not np.allclose((linear / spacing).T @ (linear / spacing), np.eye(3), atol=1e-5):
        return prior
    if cancelled():
        raise InterruptedError('Publisher vessel refinement cancelled')
    altitude = 1 / np.linalg.norm(inverse, axis=1).max()
    limit = _vascular_displacement_limits(source, mask, origin, spacing, altitude, True)
    raw = source.astype(float) @ linear.T
    baseline = prior.astype(float) @ linear.T
    fixed = ((source <= 0) | (source >= np.asarray(shape) - 1)).any(axis=1)
    if len(venous_cells):
        # Preserve existing vessel-to-vessel separations where the images and
        # the same-grid venous annotation give little room for a new position.
        distance = cKDTree(venous_cells * spacing).query(source * spacing, distance_upper_bound=2.0)[0]
        fixed |= distance <= 2.0
    next_points = _guided_vertices(baseline, triangles, fixed, cancelled=cancelled)
    move = next_points - baseline
    maximum = min(0.16, 0.24 * altitude)
    move *= np.minimum(1., maximum / np.maximum(np.linalg.norm(move, axis=1), 1e-12))[:, None]
    candidate = (prior + move @ inverse.T).astype(np.float32)
    _, source_normals, source_area = _face_geometry(raw, triangles)
    # Check the actual stored float32 points against the original physical
    # envelope and both previously validated display geometries.
    for _ in range(5):
        physical = candidate.astype(float) @ linear.T
        outside = np.linalg.norm(physical - raw, axis=1) > limit
        outside &= np.any(candidate != prior, axis=1)
        physical[outside] = baseline[outside]
        candidate[outside] = prior[outside]
        _, normals, area = _face_geometry(physical, triangles)
        changed = np.any(candidate != prior, axis=1)[triangles].any(axis=1)
        bad = changed & ((np.einsum('ij,ij->i', source_normals, normals) <= 0) | (area < 0.1 * source_area))
        rollback = np.unique(np.r_[triangles[bad].ravel(),
                                   intersection_vertices(raw, physical, triangles, cancelled),
                                   intersection_vertices(baseline, physical, triangles, cancelled)])
        rollback = rollback[np.any(candidate[rollback] != prior[rollback], axis=1)]
        if not len(rollback):
            return candidate
        candidate[rollback] = prior[rollback]
    return prior
