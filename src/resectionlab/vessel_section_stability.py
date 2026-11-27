"""Hold the source outline and enclosed voids of sampled open vessel sections."""

from dataclasses import dataclass

import numpy as np
from scipy import ndimage as ndi
from scipy.spatial import cKDTree
from skimage.morphology import skeletonize


@dataclass(frozen=True)
class Section:
    edges: np.ndarray
    keys: np.ndarray
    roles: np.ndarray
    hole_count: int


def _source_stations(mask, linear):
    """Sample interior source centerline planes across the available calibers."""
    spacing = np.linalg.norm(linear, axis=0)
    padded = np.pad(mask, 1)
    skeleton = skeletonize(padded)
    centers = np.column_stack(np.where(skeleton))
    if len(centers) < 7:
        return []
    physical = centers * spacing
    neighbors = cKDTree(physical).query(physical, k=7)[1]
    radii = ndi.distance_transform_edt(padded, sampling=spacing)[skeleton]
    eligible = ((centers > 2) & (centers < np.asarray(mask.shape) - 1)).all(axis=1)
    local = physical[neighbors[eligible]] - physical[eligible, None]
    values, vectors = np.linalg.eigh(local.transpose(0, 2, 1) @ local)
    coherent = values[:, -1] > 8 * values[:, -2]
    order = np.argsort(radii[eligible][coherent], kind='stable')
    selected = order[np.linspace(0, len(order) - 1, min(len(order), 192), dtype=int)]
    starts = (centers[eligible][coherent][selected] - 1) @ linear.T
    normals = vectors[coherent][selected, :, -1] @ (linear / spacing).T
    return list(zip(starts, normals))


def _inside(polygon, point):
    x, y = (polygon - point).T
    next_x, next_y = np.roll(x, -1), np.roll(y, -1)
    # Boundary contact does not establish a strictly enclosed region.
    dx, dy = next_x - x, next_y - y
    on_line = np.abs(x * dy - y * dx) <= 1e-8 * np.hypot(dx, dy)
    if np.any(on_line & (x * next_x + y * next_y <= 1e-16)):
        return False
    crossing = (y > 0) != (next_y > 0)
    intercept = x[crossing] - y[crossing] * (next_x[crossing] - x[crossing]) / (next_y[crossing] - y[crossing])
    return np.count_nonzero(intercept > 0) % 2 == 1


def _encloses(outer, inner):
    if np.any(inner.min(axis=0) <= outer.min(axis=0)) or np.any(inner.max(axis=0) >= outer.max(axis=0)):
        return False
    if not all(_inside(outer, point) for point in inner):
        return False
    # A chord of the inner loop can still cross a concave outer outline.
    nxt_outer, nxt_inner = np.roll(outer, -1, axis=0), np.roll(inner, -1, axis=0)
    offset = inner[None, :, :] - outer[:, None, :]
    first, second = (nxt_outer - outer)[:, None, :], (nxt_inner - inner)[None, :, :]

    def cross(a, b):
        return a[..., 0] * b[..., 1] - a[..., 1] * b[..., 0]

    opposite_outer = cross(first, offset) * cross(first, offset + second) < 0
    opposite_inner = cross(second, offset) * cross(second, offset - first) < 0
    return not np.any(opposite_outer & opposite_inner)


def _section_roles(rings, points):
    """Label the center-containing exterior and its immediate interior holes."""
    roles = np.zeros(len(points), np.uint8)
    polygons = [points[ring] for ring in rings]
    areas = [abs(x @ np.roll(y, -1) - y @ np.roll(x, -1)) for x, y in (xy.T for xy in polygons)]
    containing = [i for i, xy in enumerate(polygons) if _inside(xy, (0, 0))]
    if len(containing) % 2 == 0:
        return roles, 0
    outer = min(containing, key=areas.__getitem__)
    roles[rings[outer]] = 1
    interior = sorted((i for i, xy in enumerate(polygons) if areas[i] < areas[outer]
                       and _encloses(polygons[outer], xy)), key=areas.__getitem__, reverse=True)
    holes = []
    for i in interior:
        if not any(_encloses(polygons[hole], polygons[i]) for hole in holes):
            roles[rings[i]] = 2
            holes.append(i)
    return roles, len(holes)


def _closed_rings(connections, count):
    adjacency = [[] for _ in range(count)]
    for a, b in connections:
        adjacency[a].append(b)
        adjacency[b].append(a)
    visited = set()
    rings = []
    for start, links in enumerate(adjacency):
        if start in visited or len(links) != 2:
            continue
        ring = [start]
        visited.add(start)
        previous, current = start, links[0]
        while current not in visited and len(adjacency[current]) == 2:
            ring.append(current)
            visited.add(current)
            a, b = adjacency[current]
            previous, current = current, b if a == previous else a
        if current == start and len(ring) >= 3:
            rings.append(ring)
    return rings


def _cut(points, faces, center, normal):
    relative = points - center
    distance = np.einsum('ij,j->i', relative, normal)
    distance[np.abs(distance) < 1e-9] = 1e-9
    positive = distance > 0
    signs = positive[faces]
    cut = faces[signs.any(axis=1) & ~signs.all(axis=1)]
    edges = cut[:, [[0, 1], [1, 2], [2, 0]]].reshape(-1, 2)
    edges = np.sort(edges[positive[edges[:, 0]] != positive[edges[:, 1]]], axis=1)
    keys, remap = np.unique(edges[:, 0].astype(np.int64) * len(points) + edges[:, 1], return_inverse=True)
    edges = np.column_stack((keys // len(points), keys % len(points)))
    rings = _closed_rings(remap.reshape(-1, 2), len(edges))
    first, second = edges.T
    fraction = distance[first] / (distance[first] - distance[second])
    intersections = relative[first] + fraction[:, None] * (relative[second] - relative[first])
    axis = np.cross(normal, np.eye(3)[np.argmin(np.abs(normal))])
    axis /= np.linalg.norm(axis)
    plane_points = intersections @ np.stack((axis, np.cross(normal, axis)), axis=1)
    roles, hole_count = _section_roles(rings, plane_points)
    return Section(edges, keys, roles, hole_count), positive


def _changed_section(before, after):
    if before.hole_count != after.hole_count:
        return True
    if not (before.roles == 1).any() and not (after.roles == 1).any():
        return False
    _, old, new = np.intersect1d(before.keys, after.keys, assume_unique=True, return_indices=True)
    if not (before.roles == 1).any() or not (after.roles == 1).any():
        return True
    # Common crossed mesh edges retain their roles through ordinary vertex motion.
    # A changed role is a newly joined/separated branch or a change in void membership.
    return bool(np.any(before.roles[old] != after.roles[new]) or not np.any(before.roles[old] == 1))


def _restore_vertices(before, after, reference_side, candidate_side, faces, full):
    region = np.unique(np.concatenate((before.edges[before.roles > 0], after.edges[after.roles > 0])).ravel())
    affected = faces[np.isin(faces, region).any(axis=1)]
    vertices = np.unique(affected)
    flips = vertices[reference_side[vertices] != candidate_side[vertices]]
    return vertices if full or not len(flips) else flips


def preserve_open_sections(source, candidate, faces, mask, affine, *, cancelled):
    """Restore local source vertices when fairing changes a sampled section's topology.

    All checks use the actual float32 mesh in physical coordinates. Unsampled
    planes and open chains carry no section-topology guarantee.
    """
    diagnostics = {'sectionStations': 0, 'sectionCorrections': 0, 'sectionRestoredVertices': 0}
    if not np.any(source != candidate):
        return candidate, diagnostics
    linear = np.asarray(affine)[:3, :3]
    raw = source.astype(float) @ linear.T
    stations = []
    for center, normal in _source_stations(mask, linear):
        if cancelled():
            raise InterruptedError('Open contour section check cancelled')
        before, side = _cut(raw, faces, center, normal)
        if (before.roles == 1).any():
            stations.append((center, normal, before, side))
    diagnostics['sectionStations'] = len(stations)
    corrections = set()
    restored = np.zeros(len(source), dtype=bool)
    display = candidate.copy()
    for attempt in range(3):
        physical = display.astype(float) @ linear.T
        pins = []
        for index, (center, normal, before, side) in enumerate(stations):
            if cancelled():
                raise InterruptedError('Open contour section check cancelled')
            after, candidate_side = _cut(physical, faces, center, normal)
            if _changed_section(before, after):
                corrections.add(index)
                pins.append(_restore_vertices(before, after, side, candidate_side, faces, full=attempt > 0))
        if not pins:
            diagnostics['sectionCorrections'] = len(corrections)
            diagnostics['sectionRestoredVertices'] = int(restored.sum())
            return display, diagnostics
        if attempt < 2:
            keep = np.unique(np.concatenate(pins))
            restored[keep] |= np.any(display[keep] != source[keep], axis=1)
            display[keep] = source[keep]
    diagnostics['sectionCorrections'] = len(corrections)
    diagnostics['sectionRestoredVertices'] = int(np.count_nonzero(np.any(candidate != source, axis=1)))
    diagnostics['fallbackReason'] = 'Source contour retained because a sampled section changed topology.'
    return source, diagnostics
