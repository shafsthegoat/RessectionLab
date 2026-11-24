"""Source-cell branch review and separate gap proposals on the native TOF grid."""

from dataclasses import asdict, dataclass
from hashlib import sha256
from itertools import permutations, product
import math
import re

import numpy as np
from scipy import ndimage as ndi
from scipy.spatial import cKDTree

from .patient_anatomy import affine, world
from .vascular_homology import HomologyLimit, representative_subgraph


_FACE = ndi.generate_binary_structure(3, 1)
_FULL = np.ones((3, 3, 3), bool)
_FORWARD = tuple(
    step for step in product((-1, 0, 1), repeat=3) if step > (0, 0, 0)
)
_MAX_LABEL_CELLS = 128_000_000
# A connected native network can span most of the scan even when its
# foreground is sparse. Keep the component workspace bounded at the label
# grid scale, while the separate EDT cap keeps floating-point work local.
_MAX_COMPONENT_CELLS = 128_000_000
_MAX_SKELETON_NODES = 200_000
_MAX_REPRESENTATIVE_NODES = 100_000
_MAX_REPRESENTATIVE_EDGES = 1_000_000
_MAX_REPRESENTATIVE_CHORDS = 4096
_RADIUS_HALO_MM = 6.0
_RADIUS_TILE_CELLS = 48
_MAX_RADIUS_WORK_CELLS = 4_000_000
_H0_H1_MATCH = "source-h0-h1-matched"
_H0_H1_CAVITIES_OMITTED = "source-h0-h1-matched-cavities-unrepresented"
_H0_H1_MATCHED_STATES = {_H0_H1_MATCH, _H0_H1_CAVITIES_OMITTED}


def _native_grid(mask, native_affine, signal):
    mask = np.asarray(mask)
    if mask.ndim != 3 or not all(mask.shape) or mask.dtype != np.bool_:
        raise ValueError("Expected a three-dimensional binary source prediction")
    if signal is not None:
        signal = np.asarray(signal)
        if signal.shape != mask.shape or signal.dtype.kind not in "fui":
            raise ValueError("Source signal must match the native prediction grid")
    frame = affine(native_affine)
    spacing = np.linalg.norm(frame[:3, :3], axis=0)
    axes = frame[:3, :3] / spacing
    if not np.allclose(axes.T @ axes, np.eye(3), atol=1e-5):
        raise ValueError("Branch evidence requires an orthogonal native TOF grid")
    return mask, frame, spacing, signal


def _mask_digest(mask):
    digest = sha256(np.asarray(mask.shape, dtype="<u8").tobytes())
    for start in range(0, mask.shape[0], 16):
        digest.update(np.ascontiguousarray(mask[start : start + 16]).view(np.uint8))
    return digest.hexdigest()


def _component_crop(mask):
    bounds = []
    for axis in range(3):
        positive = np.flatnonzero(mask.any(axis=tuple(i for i in range(3) if i != axis)))
        if not len(positive):
            return None
        bounds.append(slice(int(positive[0]), int(positive[-1]) + 1))
    return tuple(bounds)


def _sparse_edges(skeleton, points):
    flat = np.ravel_multi_index(points.T, skeleton.shape)
    rows = []
    for step in _FORWARD:
        neighbors = points + step
        starts = np.flatnonzero(skeleton[tuple(neighbors.T)])
        ends = np.searchsorted(flat, np.ravel_multi_index(neighbors[starts].T, skeleton.shape))
        rows.append(np.column_stack((starts, ends)))
    return np.concatenate(rows)


def _face_support(mask, points, edges):
    start, end = points[edges[:, 0]], points[edges[:, 1]]
    delta = end - start
    distance = np.abs(delta).sum(axis=1)
    supported = distance == 1
    for axes in permutations(range(3)):
        active = np.flatnonzero(~supported & (delta[:, axes[0]] != 0) & (delta[:, axes[1]] != 0))
        first = start[active].copy()
        first[:, axes[0]] += delta[active, axes[0]]
        second = first.copy()
        second[:, axes[1]] += delta[active, axes[1]]
        supported[active] = mask[tuple(first.T)] & mask[tuple(second.T)]
    return supported


def _plane_sites(plane):
    """Counts of closed squares, both edge orientations, and their vertices."""
    first = np.pad(plane, ((1, 1), (0, 0)))
    edges0 = first[:-1] | first[1:]
    second = np.pad(plane, ((0, 0), (1, 1)))
    edges1 = second[:, :-1] | second[:, 1:]
    padded = np.pad(edges0, ((0, 0), (1, 1)))
    vertices = padded[:, :-1] | padded[:, 1:]
    return (
        np.count_nonzero(plane),
        np.count_nonzero(edges0) + np.count_nonzero(edges1),
        np.count_nonzero(vertices),
    )


def _voxel_topology(work, components):
    """Count the cubical Euler characteristic one plane at a time."""
    previous = np.zeros(work.shape[1:], bool)
    euler = 0
    for plane in work:
        faces, edges, vertices = _plane_sites(previous | plane)
        euler += faces - edges + vertices
        cubes, faces, edges = _plane_sites(plane)
        euler += -cubes + faces - edges
        previous = plane
    faces, edges, vertices = _plane_sites(previous)
    euler += faces - edges + vertices
    cavities = ndi.label(~work, structure=_FACE)[1] - 1
    return {"components26": int(components), "euler26": int(euler),
            "cavities6": int(cavities), "cycle_rank26": int(components + cavities - euler)}


def _annotation_radii(mask, indices, spacing, work, local):
    shape = np.array(mask.shape)
    if work.size <= _MAX_RADIUS_WORK_CELLS:
        # Separate 26-components have a background cell between them. Keeping
        # just this component therefore preserves its nearest background.
        radius = ndi.distance_transform_edt(work, sampling=spacing)[tuple(local.T)]
    else:
        radius = _tiled_annotation_radii(mask, indices, spacing, shape)
    boundary = np.minimum((indices + 0.5) * spacing, (shape - 0.5 - indices) * spacing).min(axis=1)
    return np.minimum(radius, _RADIUS_HALO_MM), radius >= boundary, radius >= _RADIUS_HALO_MM


def _tiled_annotation_radii(mask, indices, spacing, shape):
    radius = np.full(len(indices), np.nan)
    # Only centerline-containing tiles need an EDT. A finite halo bounds each
    # transform.
    halo = np.minimum(np.ceil(_RADIUS_HALO_MM / spacing), shape + 1).astype(int)
    tiles = indices // _RADIUS_TILE_CELLS
    _, tiles = np.unique(tiles, axis=0, return_inverse=True)
    permutation = np.argsort(tiles, kind="stable")
    breaks = np.flatnonzero(np.diff(tiles[permutation])) + 1
    for group in np.split(permutation, breaks):
        points = indices[group]
        start = np.maximum(points.min(axis=0) - halo, 0)
        stop = np.minimum(points.max(axis=0) + halo + 1, shape)
        if math.prod(stop - start + 2) > _MAX_RADIUS_WORK_CELLS:
            continue
        crop = tuple(slice(lo, hi) for lo, hi in zip(start, stop))
        local = np.pad(mask[crop], 1)
        distance = ndi.distance_transform_edt(local, sampling=spacing)
        radius[group] = distance[tuple((points - start + 1).T)]
    return radius


def _branch_paths(node_count, edges):
    neighbors = [[] for _ in range(node_count)]
    for edge, (a, b) in enumerate(edges.tolist()):
        neighbors[a].append((b, edge))
        neighbors[b].append((a, edge))
    for group in neighbors:
        group.sort()
    degrees = np.array([len(group) for group in neighbors])
    used = np.zeros(len(edges), bool)

    def path(start, successor, edge):
        nodes, visited_edges = [start], []
        while not used[edge]:
            used[edge] = True
            nodes.append(successor)
            visited_edges.append(edge)
            if degrees[successor] != 2:
                break
            other, next_edge = neighbors[successor][0]
            if used[next_edge]:
                other, next_edge = neighbors[successor][1]
            successor, edge = other, next_edge
        if nodes[-1] < nodes[0] or (nodes[-1] == nodes[0] and nodes[-2] < nodes[1]):
            nodes.reverse()
            visited_edges.reverse()
        return nodes, visited_edges

    branches = []
    for start in np.flatnonzero(degrees != 2):
        if not degrees[start]:
            branches.append(([int(start)], []))
        for successor, edge in neighbors[start]:
            if not used[edge]:
                branches.append(path(int(start), successor, edge))
    # Every remaining edge belongs to a pure cycle, whose minimum C-order
    # source cell anchors and orients it deterministically.
    for start, group in enumerate(neighbors):
        for successor, edge in group:
            if not used[edge]:
                branches.append(path(start, successor, edge))
    return branches, degrees


def _edge_blocks(node_count, edges):
    """Iterative depth-first partition into biconnected edge blocks."""
    neighbors = [[] for _ in range(node_count)]
    for edge, (a, b) in enumerate(edges.tolist()):
        neighbors[a].append((b, edge))
        neighbors[b].append((a, edge))
    visited = [-1] * node_count
    low = [-1] * node_count
    pending_edges = []
    time = 0
    for root, group in enumerate(neighbors):
        if visited[root] >= 0 or not group:
            continue
        visited[root] = low[root] = time
        time += 1
        stack = [(root, -1, iter(group))]
        while stack:
            node, incoming, cursor = stack[-1]
            next_edge = next(cursor, None)
            if next_edge is None:
                stack.pop()
                if stack:
                    parent = stack[-1][0]
                    low[parent] = min(low[parent], low[node])
                    if low[node] >= visited[parent]:
                        block = [pending_edges.pop()]
                        while block[-1] != incoming:
                            block.append(pending_edges.pop())
                        yield block
                continue
            other, edge = next_edge
            if visited[other] < 0:
                pending_edges.append(edge)
                visited[other] = low[other] = time
                time += 1
                stack.append((other, edge, iter(neighbors[other])))
            elif edge != incoming and visited[other] < visited[node]:
                pending_edges.append(edge)
                low[node] = min(low[node], visited[other])


def _bounded_representative(indices, edges, components):
    chords = len(edges) - len(indices) + components
    if (
        len(indices) <= _MAX_REPRESENTATIVE_NODES
        and len(edges) <= _MAX_REPRESENTATIVE_EDGES
        and chords <= _MAX_REPRESENTATIVE_CHORDS
    ):
        result = representative_subgraph(indices, edges)
        return result["edges"], result["components"], result["betti1"], "component"
    # Every simple cycle and every clique lies in one biconnected block.
    # Reducing the blocks independently keeps the same H0 and H1 inclusion
    # guarantee, and bounds each invocation of the existing nerve routine.
    keep = np.ones(len(edges), bool)
    cycles = 0
    for block in _edge_blocks(len(indices), edges):
        if len(block) == 1:
            continue
        nodes, flat_edges = np.unique(edges[block], return_inverse=True)
        if len(nodes) > _MAX_REPRESENTATIVE_NODES or len(block) > _MAX_REPRESENTATIVE_EDGES:
            raise HomologyLimit("Vascular H1 biconnected block edge or node cap exceeded")
        if len(block) - len(nodes) + 1 > _MAX_REPRESENTATIVE_CHORDS:
            raise HomologyLimit("Vascular H1 biconnected block chord cap exceeded")
        reduced = representative_subgraph(indices[nodes], flat_edges.reshape(-1, 2))
        keep[np.asarray(block)[reduced["removed_edge_indices"]]] = False
        cycles += reduced["betti1"]
    return edges[keep], components, cycles, "biconnected-blocks"


def _id(kind, scan, cells):
    digest = sha256(bytes.fromhex(scan))
    digest.update(kind.encode("ascii"))
    digest.update(np.asarray(cells, dtype="<u8").tobytes())
    return f"{kind}-{digest.hexdigest()[:24]}"


def _finite_values(values):
    return [float(value) if np.isfinite(value) else None for value in values]


def _component_branches(
    work, offset, mask, frame, spacing, signal, scan, component, remaining_nodes,
):
    from skimage.morphology import skeletonize

    skeleton = skeletonize(work, method="lee")
    node_count = int(np.count_nonzero(skeleton))
    if not node_count:
        component["status"] = "skeleton-empty"
        return []
    if node_count > remaining_nodes:
        component["status"] = "skeleton-node-budget-abstained"
        return []
    local = np.argwhere(skeleton)
    indices = local + offset - 1
    source_cells = np.ravel_multi_index(indices.T, mask.shape)
    raw_edges = _sparse_edges(skeleton, local)
    raw_degree = np.bincount(raw_edges.ravel(), minlength=node_count)
    source_topology = _voxel_topology(work, 1)
    skeleton_topology = source_topology if node_count == component["source_voxels"] else (
        _voxel_topology(skeleton, ndi.label(skeleton, structure=_FULL)[1])
    )
    topology_status = "representative-cap-abstained"
    edges = raw_edges
    try:
        edges, components, cycles, scope = _bounded_representative(
            indices, raw_edges, skeleton_topology["components26"]
        )
    except HomologyLimit as exc:
        component["coverage_reason"] = str(exc)
    else:
        component["representative_scope"] = scope
        component["representative_components"] = int(components)
        component["representative_cycle_rank"] = int(cycles)
        matches_source = (
            source_topology == skeleton_topology
            and components == source_topology["components26"]
            and cycles == source_topology["cycle_rank26"]
        )
        topology_status = "source-h0-h1-unverified"
        if matches_source:
            topology_status = _H0_H1_CAVITIES_OMITTED if source_topology["cavities6"] else _H0_H1_MATCH
    face_support = _face_support(work, local, edges)
    length = np.linalg.norm((indices[edges[:, 0]] - indices[edges[:, 1]]) * spacing, axis=1)
    points = world(frame, indices)
    radii, radius_extent, radius_halo = _annotation_radii(mask, indices, spacing, work, local)
    sampled_signal = signal[tuple(indices.T)] if signal is not None else None
    paths, degrees = _branch_paths(node_count, edges)
    branches = []
    for nodes, path_edges in paths:
        cells = source_cells[nodes]
        branch = {
            "id": _id("branch", scan, cells),
            "source_component_anchor_cell": component["source_component_anchor_cell"],
            "source_cell_ids": cells.tolist(),
            "points_ras_mm": points[nodes].tolist(),
            "annotation_radii_mm": _finite_values(radii[nodes]),
            "radius_limited_by_image_extent": radius_extent[nodes].tolist(),
            "radius_limited_by_workspace": radius_halo[nodes].tolist(),
            "length_mm": float(length[path_edges].sum()),
            "closed": bool(len(nodes) > 1 and nodes[0] == nodes[-1]),
            "endpoint_degrees": degrees[[nodes[0], nodes[-1]]].tolist(),
            "endpoint_raw_degrees": raw_degree[[nodes[0], nodes[-1]]].tolist(),
            "edge_local_face_supported": face_support[path_edges].tolist(),
            "topology_status": topology_status,
        }
        if sampled_signal is not None:
            branch["source_signal"] = _finite_values(sampled_signal[nodes])
        branches.append(branch)
    component.update({
        "status": topology_status,
        "source_topology": source_topology,
        "skeleton_topology": skeleton_topology,
        "skeleton_nodes": node_count,
        "raw_edges": len(raw_edges),
        "branch_edges": len(edges),
        "branches": len(branches),
        "radius_workspace_abstained_nodes": int(np.count_nonzero(~np.isfinite(radii))),
        "radius_workspace_limited_nodes": int(np.count_nonzero(radius_halo)),
    })
    return branches


def build_source_branches(mask, native_affine, *, source_scan_sha256, signal=None):
    """Derive JSON-ready source-cell paths, preserving the original prediction.

    The scan digest namespaces branch identity. Build from the unfiltered
    source prediction so component visibility overrides leave IDs unchanged.
    The source-to-background distances are annotation measurements.
    """
    mask, frame, spacing, signal = _native_grid(mask, native_affine, signal)
    if not isinstance(source_scan_sha256, str) or not re.fullmatch(r"[a-fA-F0-9]{64}", source_scan_sha256):
        raise ValueError("A SHA-256 identity for the native scan is required")
    scan = source_scan_sha256.lower()
    foreground = int(np.count_nonzero(mask))
    result = {
        "algorithm": "native-tof-source-branches-v1",
        "source_scan_sha256": scan,
        "source_prediction_sha256": _mask_digest(mask),
        "source_shape": list(mask.shape),
        "source_affine_ras_mm": frame.tolist(),
        "source_cell_index_order": "C",
        "radius_meaning": "Distance to an annotation background cell center, with limits marked.",
        "components": [],
        "branches": [],
    }
    crop = _component_crop(mask) if foreground else None
    if crop is None:
        result["coverage"] = {"status": "empty-prediction", "source_foreground_voxels": 0}
        return result
    if math.prod(axis.stop - axis.start for axis in crop) > _MAX_LABEL_CELLS:
        result["coverage"] = {
            "status": "label-workspace-abstained", "source_foreground_voxels": foreground,
            "unrepresented_foreground_voxels": foreground,
        }
        return result
    origin = np.array([axis.start for axis in crop])
    labels, count = ndi.label(mask[crop], structure=_FULL)
    # Label counts in slabs avoid a dense int64 copy of the cropped grid.
    sizes = np.zeros(count + 1, np.int64)
    for start in range(0, len(labels), 16):
        slab = labels[start : start + 16]
        sizes += np.bincount(slab[slab != 0], minlength=count + 1)
    bounds = ndi.find_objects(labels)
    remaining_nodes = _MAX_SKELETON_NODES
    unrepresented_foreground = 0
    for label in sorted(range(1, count + 1), key=lambda value: (-sizes[value], value)):
        slices = bounds[label - 1]
        offset = origin + np.array([axis.start for axis in slices])
        region = labels[slices]
        # The label order is C-order, but the bounding corner need not be
        # occupied. Find the first actual source cell without scanning a
        # potentially large region into a coordinate list.
        first_plane = region[0] == label
        first = int(first_plane.argmax())
        local_anchor = (0, *np.unravel_index(first, first_plane.shape))
        source_anchor = int(np.ravel_multi_index(offset + local_anchor, mask.shape))
        component = {
            "source_component_anchor_cell": source_anchor,
            "source_voxels": int(sizes[label]),
        }
        if math.prod(axis.stop - axis.start + 2 for axis in slices) > _MAX_COMPONENT_CELLS:
            component["status"] = "component-workspace-abstained"
            branches = []
        else:
            work = np.pad(region == label, 1)
            branches = _component_branches(
                work, offset, mask, frame, spacing, signal, scan, component, remaining_nodes,
            )
        remaining_nodes -= component.get("skeleton_nodes", 0)
        if not branches:
            unrepresented_foreground += component["source_voxels"]
        result["components"].append(component)
        result["branches"].extend(branches)
    result["components"].sort(key=lambda row: row["source_component_anchor_cell"])
    result["branches"].sort(key=lambda row: row["source_cell_ids"])
    uncertified = sum(row["status"] not in _H0_H1_MATCHED_STATES for row in result["components"])
    cavities_omitted = sum(row.get("source_topology", {}).get("cavities6", 0) > 0 for row in result["components"])
    radius_unknown = sum(row.get("radius_workspace_abstained_nodes", 0) for row in result["components"])
    result["coverage"] = {
        "status": "partial" if uncertified or cavities_omitted or radius_unknown else "complete",
        "source_foreground_voxels": foreground,
        "source_components26": count,
        "components_with_branches": sum("branches" in row for row in result["components"]),
        "components_without_source_h0_h1_match": uncertified,
        "components_with_unrepresented_cavities6": cavities_omitted,
        "unrepresented_foreground_voxels": unrepresented_foreground,
        "skeleton_nodes": _MAX_SKELETON_NODES - remaining_nodes,
        "radius_workspace_abstained_nodes": radius_unknown,
    }
    return result


@dataclass(frozen=True)
class GapPolicy:
    version: str = "native-tof-gap-review-v1"
    minimum_distance_mm: float = 0.45
    maximum_distance_mm: float = 3.0
    tangent_length_mm: float = 1.4
    minimum_tangent_alignment: float = 0.75
    maximum_radius_ratio: float = 2.25
    station_spacing_mm: float = 0.35
    ring_margin_mm: float = 0.65
    minimum_ring_samples: int = 8
    minimum_local_contrast_iqr: float = 1.0
    minimum_endpoint_contrast_fraction: float = 0.30
    minimum_supported_fraction: float = 0.8
    maximum_neighbors: int = 64
    maximum_pairs: int = 20_000

    def __post_init__(self):
        measures = (
            self.minimum_distance_mm, self.maximum_distance_mm, self.tangent_length_mm,
            self.maximum_radius_ratio, self.station_spacing_mm, self.ring_margin_mm,
            self.minimum_local_contrast_iqr,
        )
        fractions = (
            self.minimum_tangent_alignment, self.minimum_endpoint_contrast_fraction,
            self.minimum_supported_fraction,
        )
        if (
            not all(math.isfinite(value) and value > 0 for value in measures)
            or not all(0 < value <= 1 for value in fractions)
            or self.maximum_distance_mm < self.minimum_distance_mm
            or self.maximum_radius_ratio < 1
            or self.station_spacing_mm * 256 < self.maximum_distance_mm
            or type(self.minimum_ring_samples) is not int
            or not 1 <= self.minimum_ring_samples <= 16
            or type(self.maximum_neighbors) is not int
            or not 1 <= self.maximum_neighbors <= 256
            or type(self.maximum_pairs) is not int
            or not 1 <= self.maximum_pairs <= 100_000
        ):
            raise ValueError("Gap policy exceeds the supported sampling or pair bounds")


def _terminal(branch, side, policy):
    reverse = slice(None) if side == 0 else slice(None, None, -1)
    if (
        branch["topology_status"] not in _H0_H1_MATCHED_STATES
        or branch["endpoint_raw_degrees"][side] != 1
        or branch["length_mm"] < policy.tangent_length_mm
    ):
        return None
    points = np.asarray(branch["points_ras_mm"])[reverse]
    distance = np.r_[0, np.cumsum(np.linalg.norm(np.diff(points, axis=0), axis=1))]
    count = np.searchsorted(distance, policy.tangent_length_mm, side="left")
    count = min(count, len(points) - 1)
    radii = np.asarray(branch["annotation_radii_mm"][reverse][:count + 1], dtype=float)
    if (
        not np.isfinite(radii).all()
        or any(branch["radius_limited_by_image_extent"][reverse][:count + 1])
        or any(branch["radius_limited_by_workspace"][reverse][:count + 1])
    ):
        return None
    coordinate = np.array([np.interp(policy.tangent_length_mm, distance, a) for a in points.T])
    tangent = points[0] - coordinate
    norm = np.linalg.norm(tangent)
    if norm <= np.finfo(float).eps:
        return None
    return {
        "branch_id": branch["id"],
        "cell": branch["source_cell_ids"][0 if side == 0 else -1],
        "point": points[0],
        "outward_tangent": tangent / norm,
        "median_radius_mm": float(np.median(radii)),
        "same_component_key": branch["source_component_anchor_cell"],
        "topology_status": branch["topology_status"],
    }


def _endpoint_pairs(tree, points, crowded, maximum_distance):
    for left, point in enumerate(points):
        if crowded[left]:
            continue
        for right in sorted(tree.query_ball_point(point, maximum_distance)):
            if right > left and not crowded[right]:
                yield left, right


def _sample_gap(start, end, line, mask, signal, ras_to_ijk, spacing, policy):
    fractions = np.linspace(0, 1, max(3, math.ceil(line["distance_mm"] / policy.station_spacing_mm) + 1))
    centers = start["point"] + fractions[:, None] * (end["point"] - start["point"])
    direction = (centers[-1] - centers[0]) / line["distance_mm"]
    cross = np.eye(3)[np.argmin(np.abs(direction))]
    first = np.cross(direction, cross)
    first /= np.linalg.norm(first)
    second = np.cross(direction, first)
    angles = np.arange(16) * (np.pi / 8)
    unit_ring = np.cos(angles[:, None]) * first + np.sin(angles[:, None]) * second
    radii = np.interp(fractions, [0, 1], [start["median_radius_mm"], end["median_radius_mm"]])
    offsets = np.maximum(radii + policy.ring_margin_mm, 1.25 * spacing.max())
    ring = centers[:, None, :] + offsets[:, None, None] * unit_ring
    center_ijk = world(ras_to_ijk, centers).T
    ring_ijk = np.moveaxis(world(ras_to_ijk, ring), -1, 0)
    center_signal = ndi.map_coordinates(
        signal, center_ijk, output=np.float64, order=1, mode="constant", cval=np.nan,
    )
    ring_signal = ndi.map_coordinates(
        signal, ring_ijk, output=np.float64, order=1, mode="constant", cval=np.nan,
    )
    center_foreground = ndi.map_coordinates(mask, center_ijk, order=0, cval=1)
    ring_foreground = ndi.map_coordinates(mask, ring_ijk, output=np.float32, order=1, cval=1)
    ring_available = np.isfinite(ring_signal) & (ring_foreground < 0.01)
    medians = np.full(len(centers), np.nan)
    iqrs = np.full(len(centers), np.nan)
    for i, available in enumerate(ring_available):
        if available.sum() >= policy.minimum_ring_samples:
            lo, medians[i], hi = np.percentile(ring_signal[i, available], [25, 50, 75])
            iqrs[i] = hi - lo
    contrast = center_signal - medians
    # Rounded affine inversion at the scan border can affect the interpolator.
    # Endpoints remain source-owned through their exact discrete cell IDs.
    center_foreground[[0, -1]] = True
    clear = np.flatnonzero(~center_foreground)
    unknown = ~np.isfinite(contrast)
    endpoint_reference = float(min(contrast[0], contrast[-1])) if not unknown[[0, -1]].any() else None
    contrast_iqr = contrast / np.maximum(iqrs, np.finfo(np.float32).eps * np.maximum(abs(medians), 1))
    support = np.zeros(len(centers), bool)
    if endpoint_reference is not None and endpoint_reference > 0:
        support = (
            (contrast_iqr >= policy.minimum_local_contrast_iqr)
            & (contrast >= endpoint_reference * policy.minimum_endpoint_contrast_fraction)
        )
    if not len(clear):
        state = "no-sampled-gap"
    elif center_foreground[clear[0] : clear[-1] + 1].any():
        state = "crosses-source-foreground"
    elif endpoint_reference is None or unknown[clear].any():
        state = "source-image-unassessed"
    elif (
        support[clear].mean() >= policy.minimum_supported_fraction
        and support[clear[[0, -1]]].all()
    ):
        state = "uncertain-image-supported"
    else:
        state = "weak-source-image-support"
    return {
        "state": state,
        "points_ras_mm": centers.tolist(),
        "center_signal": _finite_values(center_signal),
        "ring_median_signal": _finite_values(medians),
        "ring_signal_iqr": _finite_values(iqrs),
        "ring_samples": ring_available.sum(axis=1).tolist(),
        "ring_radius_mm": offsets.tolist(),
        "center_predicted_vessel": center_foreground.tolist(),
        "center_minus_ring_iqr": _finite_values(contrast_iqr),
        "station_image_supported": support.tolist(),
        "minimum_endpoint_contrast": endpoint_reference,
        "gap_station_count": len(clear),
        "supported_gap_station_count": int(support[clear].sum()),
    }


def evaluate_source_gaps(graph, mask, signal, *, policy=GapPolicy()):
    """Test TOF support along endpoint-compatible straight paths for review."""
    mask, frame, spacing, signal = _native_grid(mask, graph["source_affine_ras_mm"], signal)
    if signal is None:
        raise ValueError("Gap evaluation requires native TOF signal")
    if graph["source_shape"] != list(mask.shape) or graph["source_prediction_sha256"] != _mask_digest(mask):
        raise ValueError("Gap evaluation must use the branch graph's source prediction")
    endpoints = []
    endpoint_count = 0
    for branch in graph["branches"]:
        for side, degree in enumerate(branch["endpoint_degrees"]):
            if degree == 1:
                endpoint_count += 1
                terminal = _terminal(branch, side, policy)
                if terminal is not None:
                    endpoints.append(terminal)
    endpoints.sort(key=lambda terminal: terminal["cell"])
    points = np.asarray([terminal["point"] for terminal in endpoints]).reshape(-1, 3)
    tree = cKDTree(points)
    counts = tree.query_ball_point(points, policy.maximum_distance_mm, return_length=True)
    crowded = counts - 1 > policy.maximum_neighbors
    ras_to_ijk = np.linalg.inv(frame)
    tested_pairs = 0
    exhausted = False
    candidates = []
    for first, second in _endpoint_pairs(tree, points, crowded, policy.maximum_distance_mm):
        if tested_pairs >= policy.maximum_pairs:
            exhausted = True
            break
        tested_pairs += 1
        left, right = endpoints[first], endpoints[second]
        displacement = right["point"] - left["point"]
        distance = float(np.linalg.norm(displacement))
        if distance < policy.minimum_distance_mm:
            continue
        direction = displacement / distance
        alignment = [float(left["outward_tangent"] @ direction), float(right["outward_tangent"] @ -direction)]
        radii = [left["median_radius_mm"], right["median_radius_mm"]]
        radius_ratio = max(radii) / min(radii)
        if min(alignment) < policy.minimum_tangent_alignment or radius_ratio > policy.maximum_radius_ratio:
            continue
        cells = [left["cell"], right["cell"]]
        geometry = {
            "distance_mm": distance,
            "outward_tangent_alignment": alignment,
            "endpoint_annotation_radii_mm": radii,
            "annotation_radius_ratio": radius_ratio,
        }
        evidence = _sample_gap(left, right, geometry, mask, signal, ras_to_ijk, spacing, policy)
        candidates.append({
            "id": _id("gap", graph["source_scan_sha256"], cells),
            "source_cell_ids": cells,
            "branch_ids": [left["branch_id"], right["branch_id"]],
            "branch_topology_statuses": [left["topology_status"], right["topology_status"]],
            "within_source_component": left["same_component_key"] == right["same_component_key"],
            "geometry": geometry,
            **evidence,
        })
    return {
        "policy": asdict(policy),
        "source_scan_sha256": graph["source_scan_sha256"],
        "source_prediction_sha256": graph["source_prediction_sha256"],
        "predicted_voxels_added": 0,
        "gap_role": "Uncertain source-image review proposals.",
        "coverage": {
            "source_branch_graph_status": graph["coverage"]["status"],
            "graph_terminals": endpoint_count,
            "eligible_terminals": len(endpoints),
            "crowded_terminals_abstained": int(np.count_nonzero(crowded)),
            "pair_budget_exhausted": exhausted,
            "nearby_pairs_tested": tested_pairs,
            "geometry_compatible_pairs": len(candidates),
            "image_supported_proposals": sum(
                row["state"] == "uncertain-image-supported" for row in candidates
            ),
            "image_unassessed_proposals": sum(row["state"] == "source-image-unassessed" for row in candidates),
        },
        "candidates": candidates,
    }
