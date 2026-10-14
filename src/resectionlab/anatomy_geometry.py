"""Compiled, cached display meshes and measured vascular geometry."""

from __future__ import annotations
import os
import tempfile
import numpy as np
from scipy import ndimage as ndi
from skimage.morphology import skeletonize
from .patient_anatomy import digest, affine, world


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


def _poly(values, probability, smooth):
    import vtk
    from vtk.util.numpy_support import numpy_to_vtk

    image = vtk.vtkImageData()
    image.SetDimensions(*values.shape)
    image.GetPointData().SetScalars(
        numpy_to_vtk(np.asarray(values).ravel(order="F"), deep=True)
    )
    if probability:
        contour = vtk.vtkFlyingEdges3D()
        contour.SetValue(0, 0.5)
    else:
        contour = vtk.vtkSurfaceNets3D()
        contour.SetValue(0, 1)
        contour.SetOutputMeshTypeToQuads()
        contour.SetSmoothing(bool(smooth))
        if smooth:
            smoother = contour.GetSmoother()
            smoother.SetNumberOfIterations(20)
            smoother.SetRelaxationFactor(0.25)
            smoother.SetConstraintDistance(0.5)
    contour.SetInputData(image)
    contour.Update()
    output = contour.GetOutput()
    if probability and smooth and output.GetNumberOfPoints():
        smoothing = vtk.vtkWindowedSincPolyDataFilter()
        smoothing.SetInputData(output)
        smoothing.SetNumberOfIterations(15)
        smoothing.SetPassBand(0.1)
        smoothing.BoundarySmoothingOff()
        smoothing.FeatureEdgeSmoothingOff()
        smoothing.NonManifoldSmoothingOff()
        smoothing.NormalizeCoordinatesOn()
        smoothing.Update()
        output = smoothing.GetOutput()
    return output


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
            "algorithm": "native-vtk-v5",
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
    values = np.pad(values, 1)
    if cancelled():
        raise InterruptedError("Mesh preparation cancelled")
    raw, faces = _arrays(_poly(values, probability, False))
    smoothed, sf = _arrays(_poly(values, probability, True))
    if raw.shape != smoothed.shape or not np.array_equal(faces, sf):
        smoothed = raw.copy()
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
    spacing = np.linalg.norm(a[:3, :3], axis=0)
    delta = (smoothed - raw) @ a[:3, :3].T
    limit = np.full(
        len(raw), 0.5 / np.max(np.linalg.norm(np.linalg.inv(a)[:3, :3], axis=1))
    )
    if row["kind"] in {"vessels", "artery", "vein", "sinus"}:
        radius = ndi.distance_transform_edt(np.pad(cropped, 1), sampling=spacing)
        index = np.clip(np.rint(raw).astype(int), 0, np.array(radius.shape) - 1)
        limit = np.minimum(
            limit, 0.25 * np.maximum(radius[tuple(index.T)], min(spacing) * 0.5)
        )
    length = np.linalg.norm(delta, axis=1)
    factor = np.minimum(1, limit / np.maximum(length, 1e-12))
    smoothed = raw + (smoothed - raw) * factor[:, None]
    # Pin faces which invert or collapse, rather than disabling an entire vessel tree.
    for _ in range(4):
        f0, f1 = raw[faces], smoothed[faces]
        n0 = np.cross(f0[:, 1] - f0[:, 0], f0[:, 2] - f0[:, 0])
        n1 = np.cross(f1[:, 1] - f1[:, 0], f1[:, 2] - f1[:, 0])
        bad = (np.einsum("ij,ij->i", n0, n1) <= 0) | (
            np.linalg.norm(n1, axis=1) < 0.1 * np.linalg.norm(n0, axis=1)
        )
        if not bad.any():
            break
        ids = np.unique(faces[bad])
        smoothed[ids] = raw[ids]
    # Pinning one face can affect its neighbours. Reject the candidate globally
    # if the bounded repair still leaves an inverted or collapsed triangle.
    f0, f1 = raw[faces], smoothed[faces]
    n0 = np.cross(f0[:, 1] - f0[:, 0], f0[:, 2] - f0[:, 0])
    n1 = np.cross(f1[:, 1] - f1[:, 0], f1[:, 2] - f1[:, 0])
    source_valid = np.linalg.norm(n0, axis=1) > 1e-12
    rejected = bool(
        np.any(
            source_valid
            & (
                (np.einsum("ij,ij->i", n0, n1) <= 0)
                | (np.linalg.norm(n1, axis=1) < 0.1 * np.linalg.norm(n0, axis=1))
            )
        )
    )
    if rejected:
        smoothed = raw.copy()
    offset = lo - 1
    raw = (raw + offset).astype(np.float32)
    smoothed = (smoothed + offset).astype(np.float32)
    result = {
        "source": raw,
        "positions": smoothed,
        "normals": normals(smoothed, faces),
        "indices": faces,
        "smoothing_rejected": np.array(rejected),
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
    """Measured skeleton graph, without connecting gaps or assigning vessel identity."""
    a = affine(native_affine)
    spacing = np.linalg.norm(a[:3, :3], axis=0)
    if not np.allclose(
        (a[:3, :3] / spacing).T @ (a[:3, :3] / spacing), np.eye(3), atol=1e-5
    ):
        raise ValueError("Radius extraction requires an orthogonal source grid")
    skeleton = skeletonize(mask)
    indices = np.argwhere(skeleton)
    lookup = {tuple(p): i for i, p in enumerate(indices)}
    edges = []
    for i, p in enumerate(indices):
        for delta in np.ndindex(3, 3, 3):
            q = tuple(p + np.array(delta) - 1)
            j = lookup.get(q)
            if j is not None and j > i:
                edges.append((i, j))
    radius = ndi.distance_transform_edt(mask, sampling=spacing)
    _, count = ndi.label(mask, structure=np.ones((3, 3, 3)))
    return {
        "points_ras_mm": world(a, indices).astype(np.float32),
        "edges": np.array(edges, dtype=np.uint32).reshape(-1, 2),
        "radius_mm": radius[tuple(indices.T)].astype(np.float32),
        "components": int(count),
        "meaning": "voxel skeleton and inscribed radius estimates; no inferred missing branches",
    }


def inspect_vessel_geometry(collection, layer_id, point=None):
    """Revision-keyed measured graph; never fabricate a connection across a gap."""
    from scipy.spatial import cKDTree

    row = collection.layer(layer_id)
    if row["kind"] not in {"vessels", "artery", "vein", "sinus"}:
        raise ValueError("Select a vascular layer")
    a = collection.patient_affine(row)
    key = digest(
        {
            "algorithm": "skeleton-26-v1",
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
        "meaning": "measured voxel skeleton, inscribed radii; no gap bridging or artery/vein assignment",
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
        }
    return result
