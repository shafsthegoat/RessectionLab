"""Compiled, cached display meshes and source-derived vascular geometry."""

from __future__ import annotations
import os
import tempfile
import numpy as np
from scipy import ndimage as ndi
from skimage.morphology import skeletonize
from .patient_anatomy import digest, affine, world
from .surface_fairing import fair_surface, POLICY


def _save_cache(path, values):
    """Interrupted writers cannot leave a partially readable geometry cache."""
    fd, temporary = tempfile.mkstemp(dir=path.parent, suffix=".npz")
    try:
        with os.fdopen(fd, "wb") as stream:
            np.savez_compressed(stream, **values)
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def _poly(values, probability):
    import vtk
    from vtk.util.numpy_support import numpy_to_vtk
    image = vtk.vtkImageData()
    image.SetDimensions(*values.shape)
    image.GetPointData().SetScalars(numpy_to_vtk(np.asarray(values).ravel(order="F"), deep=True))
    if probability:
        contour = vtk.vtkFlyingEdges3D()
        contour.SetValue(0, 0.5)
    else:
        contour = vtk.vtkSurfaceNets3D()
        contour.SetValue(0, 1)
        contour.SetOutputMeshTypeToQuads()
        contour.SmoothingOff()
    contour.SetInputData(image)
    contour.Update()
    return contour.GetOutput()


def _arrays(poly):
    from vtk.util.numpy_support import vtk_to_numpy

    if not poly.GetNumberOfPoints():
        return np.empty((0, 3), np.float32), np.empty((0, 3), np.uint32)
    points = vtk_to_numpy(poly.GetPoints().GetData()).copy()
    cells = poly.GetPolys()
    connectivity = vtk_to_numpy(cells.GetConnectivityArray())
    offsets = vtk_to_numpy(cells.GetOffsetsArray())
    widths = np.diff(offsets)
    if np.all(widths == 3):
        faces = connectivity.reshape(-1, 3).astype(np.uint32)
    elif np.all(widths == 4):
        quads = connectivity.reshape(-1, 4)
        faces = np.concatenate([quads[:, [0, 1, 2]], quads[:, [0, 2, 3]]]).astype(
            np.uint32
        )
    else:
        raise ValueError("Unexpected VTK surface cell topology")
    return points, faces


def normals(points, faces):
    result = np.zeros_like(points, dtype=np.float64)
    f = points[faces]
    n = np.cross(f[:, 1] - f[:, 0], f[:, 2] - f[:, 0])
    for i in range(3):
        np.add.at(result, faces[:, i], n)
    result /= np.maximum(np.linalg.norm(result, axis=1, keepdims=True), 1e-20)
    return result.astype(np.float32)


def mesh(collection, row, *, cancelled=lambda: False):
    """Separate source/display mesh; same connectivity, bounded local movement."""
    key = digest(
        {
            "algorithm": POLICY["version"],
            "data": row["data"],
            "probability": row.get("probability"),
            "affine": row["affine"],
            "label": row["label_value"],
            "revision": row["revision"],
            "kind": row["kind"],
        }
    )
    cache = collection.root / "meshes" / f"{key}.npz"
    cache.parent.mkdir(exist_ok=True)
    if cache.exists():
        with np.load(cache, allow_pickle=False) as z:
            return {k: z[k] for k in z.files}
    mask = collection.mask(row)
    where = np.nonzero(mask)
    if not len(where[0]):
        return {
            "source": np.empty((0, 3), np.float32),
            "positions": np.empty((0, 3), np.float32),
            "normals": np.empty((0, 3), np.float32),
            "indices": np.empty((0, 3), np.uint32),
            "max_displacement_mm": np.array(0.0),
        }
    lo = np.maximum(0, np.array([v.min() for v in where]) - 2)
    hi = np.minimum(mask.shape, np.array([v.max() for v in where]) + 3)
    slices = tuple(slice(a, b) for a, b in zip(lo, hi))
    cropped = mask[slices]
    probability = "probability" in row and not row.get("edits")
    values = (
        collection.array(row["probability"])[slices]
        if probability
        else cropped.astype(np.uint8)
    )
    if probability and (not np.isfinite(values).all() or not np.array_equal(values >= .5, cropped)):
        probability = False
        values = cropped.astype(np.uint8)
    values = np.pad(values, 1)
    if cancelled():
        raise InterruptedError("Mesh preparation cancelled")
    raw, faces = _arrays(_poly(values, probability))
    triangles = raw[faces]
    areas = np.linalg.norm(
        np.cross(triangles[:, 1] - triangles[:, 0], triangles[:, 2] - triangles[:, 0]),
        axis=1,
    )
    degenerate = int(np.count_nonzero(areas <= 1e-12))
    # Exact threshold ties can emit zero-area VTK triangles. They represent no
    # boundary area; omit only those faces from both versions, retaining vertices.
    faces = faces[areas > 1e-12]
    a = affine(row["affine"])
    # Quantize in final source coordinates before all geometry safety checks.
    offset = lo - 1
    raw = (raw + offset).astype(np.float32)
    smoothed, diagnostics = fair_surface(raw, faces, a, row["kind"],
        mask=np.pad(cropped, 1), mask_origin=offset, cancelled=cancelled)
    rejected = diagnostics["restoredComponents"] > 0
    result = {
        "source": raw,
        "positions": smoothed,
        "normals": normals(smoothed, faces),
        "indices": faces,
        "smoothing_rejected": np.array(rejected),
        "restored_components": np.array(diagnostics["restoredComponents"]),
        "backed_off_vertices": np.array(diagnostics["backedOffVertices"]),
        "policy_version": np.array(POLICY["version"]),
        "fallback_reason": np.array(diagnostics.get("fallbackReason", "")),
        "source_degenerate_faces_removed": np.array(degenerate),
        "max_displacement_mm": np.array(
            np.linalg.norm((smoothed - raw) @ a[:3, :3].T, axis=1).max(initial=0)
        ),
    }
    if cancelled():
        raise InterruptedError("Mesh preparation cancelled before cache admission")
    _save_cache(cache, result)
    return result


def vessel_graph(mask, native_affine):
    """Source-owned Lee skeleton with explicit topology/coverage diagnostics.

    This is annotation geometry, not a measured vascular network. Diagonal
    skeleton adjacency is distinguished from a local six-face foreground path.
    Neither category estimates lumen, conductance, flow or missing branches.
    """
    a = affine(native_affine)
    mask = np.asarray(mask)
    if mask.ndim != 3 or not all(mask.shape) or mask.dtype.kind not in 'bui':
        raise ValueError('Vessel graph requires an explicit binary source mask')
    if mask.size > 128_000_000:
        raise ValueError('Vessel graph source exceeds declared grid cap')
    if mask.dtype.kind != 'b' and not np.isin(mask, [0, 1]).all():
        raise ValueError('Vessel graph requires an explicit binary source mask')
    spacing = np.linalg.norm(a[:3, :3], axis=0)
    if not np.allclose(
        (a[:3, :3] / spacing).T @ (a[:3, :3] / spacing), np.eye(3), atol=1e-5
    ):
        raise ValueError("Radius extraction requires an orthogonal source grid")
    if np.count_nonzero(mask) > 1_000_000:
        raise ValueError('Vessel graph foreground exceeds declared one-million-cell cap')
    source_ids = np.flatnonzero(mask)
    source_ijk = np.column_stack(np.unravel_index(source_ids, mask.shape))
    lower = source_ijk.min(axis=0) if len(source_ids) else np.zeros(3, int)
    upper = source_ijk.max(axis=0)+1 if len(source_ids) else np.ones(3, int)
    if int(np.prod(upper-lower+2)) > 32_000_000:
        raise ValueError('Vessel graph cropped workspace exceeds declared cap')
    crop = tuple(slice(int(lo), int(hi)) for lo, hi in zip(lower, upper))
    # Padding represents unassessed exterior for thinning/EDT only, never a
    # negative anatomical finding. Boundary-limited distances are flagged.
    work = np.pad(mask[crop].astype(bool), 1)
    skeleton = skeletonize(work, method='lee')
    local_indices = np.argwhere(skeleton)
    if len(local_indices) > 100_000:
        raise ValueError('Vessel graph skeleton exceeds declared node cap')
    indices = local_indices+lower-1
    lookup = {tuple(p): i for i, p in enumerate(indices)}
    edges = []
    for i, p in enumerate(indices):
        for delta in np.ndindex(3, 3, 3):
            q = tuple(p + np.array(delta) - 1)
            j = lookup.get(q)
            if j is not None and j > i:
                edges.append((i, j))
                if len(edges) > 1_000_000:
                    raise ValueError('Vessel graph skeleton exceeds declared edge cap')
    edges = np.array(edges, dtype=np.uint32).reshape(-1, 2)
    source6, count6 = ndi.label(work, structure=ndi.generate_binary_structure(3, 1))
    source26, count26 = ndi.label(work, structure=np.ones((3, 3, 3)))
    sampled6 = source6[tuple(local_indices.T)]
    sampled26 = source26[tuple(local_indices.T)]
    represented = set(map(int, sampled26))
    represented6 = set(map(int, sampled6))
    lost = [i for i in range(1, count26+1) if i not in represented]
    lost6 = [i for i in range(1, count6+1) if i not in represented6]
    sizes = np.bincount(source26.ravel())
    sizes6 = np.bincount(source6.ravel())
    local_supported = []
    for left, right in edges:
        start, end = local_indices[[left, right]]
        # At most eight source voxels; no path outside the edge's local box.
        lo, hi = np.minimum(start, end), np.maximum(start, end)
        visited = {tuple(start)}; pending = [tuple(start)]
        while pending:
            cell = np.array(pending.pop())
            for axis in range(3):
                for sign in (-1, 1):
                    other = cell.copy(); other[axis] += sign; key = tuple(other)
                    if (np.all(other >= lo) and np.all(other <= hi) and work[key] and key not in visited):
                        visited.add(key); pending.append(key)
        local_supported.append(tuple(end) in visited)
    radius = ndi.distance_transform_edt(work, sampling=spacing)[tuple(local_indices.T)]
    edge_face = np.abs(indices[edges[:, 0]]-indices[edges[:, 1]]).sum(axis=1) == 1
    boundary_distance = np.minimum((indices+.5)*spacing, (np.array(mask.shape)-.5-indices)*spacing).min(axis=1)
    _, skeleton_components = ndi.label(skeleton, structure=np.ones((3, 3, 3)))
    from skimage.measure import euler_number
    source_euler = int(euler_number(work, connectivity=3))
    skeleton_euler = int(euler_number(skeleton, connectivity=3))
    # Euler of a voxel complex differs from V-E of its all-26-neighbor graph:
    # corner/edge/face cliques can add spurious graph cycles. Expose both.
    from scipy.sparse import coo_matrix
    from scipy.sparse.csgraph import connected_components
    adjacency = coo_matrix((np.ones(len(edges)),(edges[:,0],edges[:,1])),shape=(len(indices),len(indices))).tocsr()
    graph_components = int(connected_components(adjacency,directed=False,return_labels=False))
    graph_cycles = len(edges)-len(indices)+graph_components
    _, background_components = ndi.label(~skeleton,structure=ndi.generate_binary_structure(3,1))
    cavities = int(background_components)-1
    voxel_cycles = int(skeleton_components)+cavities-skeleton_euler
    _, source_background_components = ndi.label(~work,structure=ndi.generate_binary_structure(3,1))
    source_cavities = int(source_background_components)-1
    source_cycles = int(count26)+source_cavities-source_euler
    from .vascular_homology import representative_subgraph, HomologyLimit
    try:
        representative = representative_subgraph(indices,edges)
        representative_fields = {'h1_'+key:value for key,value in representative.items()}
        representative_fields['h1_status'] = np.array('certified-H0-H1-over-GF2')
        representative_fields['h1_matches_source_betti'] = (representative['components'] == count26
            and representative['betti1'] == source_cycles and source_cavities == 0)
    except HomologyLimit as exc:
        # Raw source diagnostics remain useful, but a partial basis must never
        # masquerade as a representative graph or certify a subset of cycles.
        representative_fields = {'h1_status':np.array('cap-abstained'), 'h1_cap_reason':np.array(str(exc)),
            'h1_matches_source_betti':False}
    return {
        'points_ras_mm': world(a, indices).astype(np.float64),
        'source_cell_ids': np.ravel_multi_index(indices.T, mask.shape).astype(np.int64),
        'source_shape': np.array(mask.shape, np.int64), 'source_affine_ras_mm': a.copy(),
        'edges': edges, 'edge_face_adjacent': edge_face,
        'edge_local_face_supported': np.asarray(local_supported, bool),
        'edge_same_source6_component': sampled6[edges[:,0]] == sampled6[edges[:,1]],
        'source_component6': sampled6, 'source_component26': sampled26,
        'radius_mm': radius.astype(np.float64),
        'radius_limited_by_image_extent': radius >= boundary_distance,
        'components': int(count26), 'source_components6': int(count6),
        'skeleton_components26': int(skeleton_components),
        'source_euler26': source_euler, 'skeleton_euler26': skeleton_euler,
        'unrepresented_source_components26': len(lost),
        'unrepresented_source_components6': count6-len(represented6),
        'unrepresented_source26_voxels': int(sizes[lost].sum()) if lost else 0,
        'unrepresented_source6_voxels': int(sizes6[lost6].sum()) if lost6 else 0,
        'source_foreground_voxels': len(source_ids),
        'source_boundary_foreground_voxels': int(np.any((source_ijk == 0) | (source_ijk == np.array(mask.shape)-1), axis=1).sum()),
        'voxel_skeleton_betti_preserved26': not lost and count26 == skeleton_components and source_euler == skeleton_euler and source_cavities == cavities,
        'edge_graph_components': graph_components, 'edge_graph_cycle_rank': graph_cycles,
        'edge_graph_euler': len(indices)-len(edges), 'voxel_skeleton_cycle_rank26': voxel_cycles,
        'voxel_skeleton_cavities6_background': cavities,
        'source_cavities6_background': source_cavities, 'source_cycle_rank26': source_cycles,
        'edge_graph_matches_voxel_betti': graph_components == skeleton_components and graph_cycles == voxel_cycles and cavities == 0,
        **representative_fields,
        'meaning': 'Derived source-voxel skeleton and 26-neighbor adjacency. Voxel and edge-graph topology are separate checks. Annotation-to-background-center distance is not measured lumen radius. No inferred flow.',
    }


def inspect_vessel_geometry(collection, layer_id, point=None):
    """Revision-keyed derived graph with source identity and unknown connectivity."""
    from scipy.spatial import cKDTree
    import skimage

    row = collection.layer(layer_id)
    if row["kind"] not in {"vessels", "artery", "vein", "sinus"}:
        raise ValueError("Select a vascular layer")
    a = collection.patient_affine(row)
    key = digest(
        {
            "algorithm": "source-owned-skeleton-26-v4",
            "skimage_version": skimage.__version__,
            "data": row["data"],
            "label": row["label_value"],
            "affine": a.tolist(),
        }
    )
    path = collection.root / "graphs" / f"{key}.npz"
    path.parent.mkdir(exist_ok=True)
    if path.exists():
        with np.load(path, allow_pickle=False) as archive:
            graph = {name: archive[name] for name in archive.files}
    else:
        graph = vessel_graph(collection.mask(row), a)
        graph = {k: v for k, v in graph.items() if isinstance(v, (np.ndarray, int))}
        _save_cache(path, graph)
    points = graph["points_ras_mm"]
    edges = graph["edges"]
    degree = np.bincount(edges.ravel(), minlength=len(points))
    result = {
        "snapshot_id": collection.identity(),
        "graph_id": key,
        "layer_id": layer_id,
        "nodes": len(points),
        "edges": len(edges),
        "components": int(graph["components"]),
        "endpoints": int((degree == 1).sum()),
        "junction_voxels": int((degree > 2).sum()),
        "path": str(path),
        "source_components6": int(graph['source_components6']),
        "skeleton_components26": int(graph['skeleton_components26']),
        "voxel_skeleton_betti_preserved26": bool(graph['voxel_skeleton_betti_preserved26']),
        "edge_graph_cycle_rank": int(graph['edge_graph_cycle_rank']),
        "voxel_skeleton_cycle_rank26": int(graph['voxel_skeleton_cycle_rank26']),
        "edge_graph_matches_voxel_betti": bool(graph['edge_graph_matches_voxel_betti']),
        "unrepresented_source_components26": int(graph['unrepresented_source_components26']),
        "unrepresented_source_components6": int(graph['unrepresented_source_components6']),
        "unrepresented_source26_voxels": int(graph['unrepresented_source26_voxels']),
        "unrepresented_source6_voxels": int(graph['unrepresented_source6_voxels']),
        "diagonal_edges_without_local_face_support": int((~graph['edge_local_face_supported']).sum()),
        "edges_across_source6_components": int((~graph['edge_same_source6_component']).sum()),
        "source_boundary_foreground_voxels": int(graph['source_boundary_foreground_voxels']),
        "h1_status": str(graph['h1_status']),
        "h1_representative_edges": len(graph['h1_edges']) if 'h1_edges' in graph else None,
        "h1_cycle_rank": int(graph['h1_betti1']) if 'h1_betti1' in graph else None,
        "h1_matches_source_betti": bool(graph['h1_matches_source_betti']),
        "boundary_limited_distance_nodes": int(graph['radius_limited_by_image_extent'].sum()),
        "radius_is_lumen_measurement": False,
        "flow_model_admitted": False,
        "meaning": "Source-owned derived skeleton; annotation distance is not lumen radius. Diagonal-only contacts, lost components and unassessed branches remain unknown; no flow inference.",
    }
    if point is not None and len(points):
        if np.asarray(point).shape != (3,) or not np.isfinite(point).all():
            raise ValueError("Invalid physical query")
        distance, index = cKDTree(points).query(point)
        result["nearest"] = {
            "point_ras_mm": points[index].tolist(),
            "distance_mm": float(distance),
            "radius_mm": float(graph["radius_mm"][index]),
            "degree": int(degree[index]),
            "source_cell_id": int(graph['source_cell_ids'][index]),
            "radius_limited_by_image_extent": bool(graph['radius_limited_by_image_extent'][index]),
        }
    return result
