"""Source-constrained display positions for a coarse native brain envelope."""

import numpy as np
from scipy.ndimage import map_coordinates

from .surface_fairing import _intersection_vertices


BRAIN_ENVELOPE_POLICY = "native-brain-envelope-v1"
MAX_TRIANGLES = 1_000_000
MAX_ADDITIONAL_MM = .4


def _low_pass(points, faces):
    from vtkmodules.util.numpy_support import numpy_to_vtk, numpy_to_vtkIdTypeArray, vtk_to_numpy
    from vtkmodules.vtkCommonCore import vtkPoints
    from vtkmodules.vtkCommonDataModel import vtkCellArray, vtkPolyData
    from vtkmodules.vtkFiltersCore import vtkWindowedSincPolyDataFilter

    vertices = vtkPoints()
    vertices.SetData(numpy_to_vtk(np.ascontiguousarray(points), deep=True))
    cells = vtkCellArray()
    offsets = np.arange(0, 3 * len(faces) + 1, 3, dtype=np.int64)
    cells.SetData(numpy_to_vtkIdTypeArray(offsets, deep=True),
                  numpy_to_vtkIdTypeArray(faces.astype(np.int64).ravel(), deep=True))
    surface = vtkPolyData()
    surface.SetPoints(vertices)
    surface.SetPolys(cells)
    smoothing = vtkWindowedSincPolyDataFilter()
    smoothing.SetInputData(surface)
    smoothing.SetNumberOfIterations(32)
    smoothing.SetPassBand(.025)
    smoothing.NormalizeCoordinatesOn()
    smoothing.BoundarySmoothingOff()
    smoothing.FeatureEdgeSmoothingOff()
    smoothing.NonManifoldSmoothingOff()
    smoothing.Update()
    result = smoothing.GetOutput()
    if (result.GetNumberOfPoints() != len(points)
            or not np.array_equal(vtk_to_numpy(result.GetPolys().GetConnectivityArray()), faces.ravel())):
        return None
    return vtk_to_numpy(result.GetPoints().GetData()).astype(np.float64)


def _face_normals(points, faces):
    a, b, c = points[faces].transpose(1, 0, 2)
    return np.cross(b - a, c - a)


def _turn_decreases(faces, before, after, vertex_count):
    keys = np.empty(3 * len(faces), dtype=np.uint64)
    for corner in range(3):
        a, b = faces[:, corner], faces[:, (corner + 1) % 3]
        keys[corner * len(faces):(corner + 1) * len(faces)] = (
            np.minimum(a, b).astype(np.uint64) * vertex_count + np.maximum(a, b)
        )
    order = np.argsort(keys, kind="stable")
    sorted_keys = keys[order]
    starts = np.r_[0, np.flatnonzero(np.diff(sorted_keys)) + 1, len(order)]
    pairs = starts[:-1][np.diff(starts) == 2]
    if not len(pairs):
        return False
    pairs = pairs[::max(1, len(pairs) // 16384)]
    a, b = order[pairs] % len(faces), order[pairs + 1] % len(faces)

    def mean_turn_squared(normals):
        first, second = normals[a], normals[b]
        scale = np.linalg.norm(first, axis=1) * np.linalg.norm(second, axis=1)
        cosine = np.einsum("ij,ij->i", first, second) / np.maximum(scale, 1e-20)
        return float(np.mean(np.arccos(np.clip(cosine, -1, 1)) ** 2))

    original = mean_turn_squared(before)
    return original > 0 and mean_turn_squared(after) < .95 * original


def _contrast(signal, points, directions, distances):
    result = np.zeros(len(points))
    for distance in distances:
        offset = distance * directions
        before = map_coordinates(signal, (points - offset).T, output=np.float32, order=1, mode="nearest")
        after = map_coordinates(signal, (points + offset).T, output=np.float32, order=1, mode="nearest")
        result += np.abs(before - after)
    return result / len(distances)


def _source_guard(current, proposal, direction, mask, signal, cancelled):
    """Shorten the extra step at a categorical or measured contrast boundary."""
    retained = current.copy()
    steps = proposal - current
    active = np.flatnonzero(np.any(steps != 0, axis=1))
    if not len(active):
        return retained
    baseline = current[active]
    signal_strength = _contrast(signal, baseline, direction[active], (.5, .85))
    finite = np.isfinite(signal_strength)
    if not finite.any():
        return retained
    floor = np.percentile(signal_strength[finite], 10)
    minimum = np.where(signal_strength > floor, .92 * signal_strength, 0)
    residual = np.abs(map_coordinates(mask, baseline.T, output=np.float32, order=1) - .5) + .06
    active, minimum, residual = active[finite], minimum[finite], residual[finite]
    for scale in (1., .5, .25):
        if cancelled():
            raise InterruptedError("Brain envelope preparation cancelled")
        candidate = current[active] + steps[active] * scale
        observed = _contrast(signal, candidate, direction[active], (.5, .85))
        membership = map_coordinates(mask, candidate.T, output=np.float32, order=1)
        keep = (observed >= minimum) & (np.abs(membership - .5) <= residual)
        retained[active[keep]] = candidate[keep]
        active, minimum, residual = active[~keep], minimum[~keep], residual[~keep]
        if not len(active):
            break
    return retained


def refine_brain_envelope(raw, current, faces, linear, mask, signal, *, cancelled):
    """Return an optional same-index derivative of an unedited brain estimate."""
    if not len(faces) or len(faces) > MAX_TRIANGLES:
        return None
    edge_distance = np.minimum(raw, np.array(mask.shape) - 1 - raw).min(axis=1)
    interior = edge_distance > 2
    if not interior.any():
        return None
    if cancelled():
        raise InterruptedError("Brain envelope preparation cancelled")
    inverse = np.linalg.inv(linear)
    physical = current.astype(np.float64) @ linear.T
    target = _low_pass(physical, faces)
    if target is None or not np.isfinite(target).all():
        return None
    weight = np.clip((edge_distance - 2) / 4, 0, 1)
    weight = weight * weight * (3 - 2 * weight)
    extra = (target - physical) * weight[:, None]
    extra *= np.minimum(1, .3995 / np.maximum(np.linalg.norm(extra, axis=1), 1e-20))[:, None]
    proposal = (current + extra @ inverse.T).astype(np.float32)
    proposal[~interior] = current[~interior]
    source_limit = np.maximum(.88, np.linalg.norm((current - raw) @ linear.T, axis=1))
    for _ in range(5):
        over = np.linalg.norm((proposal - raw) @ linear.T, axis=1) > source_limit + 1e-6
        if not over.any():
            break
        proposal[over] = .5 * (current[over] + proposal[over])
    over = np.linalg.norm((proposal - raw) @ linear.T, axis=1) > source_limit + 1e-6
    proposal[over] = current[over]

    original_normals = _face_normals(physical, faces)
    vertex_normals = np.zeros_like(physical)
    for corner in faces.T:
        np.add.at(vertex_normals, corner, original_normals)
    vertex_normals /= np.maximum(np.linalg.norm(vertex_normals, axis=1, keepdims=True), 1e-20)
    direction = vertex_normals @ inverse.T
    proposal = _source_guard(current, proposal, direction, mask, signal, cancelled)
    result = proposal @ linear.T
    min_area = .01 * np.einsum("ij,ij->i", original_normals, original_normals)
    backed_off = np.zeros(len(current), bool)
    dirty = None
    for _ in range(5):
        actual = _face_normals(result, faces)
        invalid = ((np.einsum("ij,ij->i", original_normals, actual) <= 0)
                   | (np.einsum("ij,ij->i", actual, actual) < min_area))
        intersecting = _intersection_vertices(physical, result, faces, cancelled, dirty)
        rollback = np.union1d(faces[invalid].ravel(), intersecting)
        if not len(rollback):
            break
        proposal[rollback] = current[rollback]
        result[rollback] = physical[rollback]
        backed_off[rollback] = True
        dirty = np.zeros(len(current), bool)
        dirty[rollback] = True
    else:
        return None
    if not np.isfinite(proposal).all() or not np.any(proposal != current):
        return None
    if np.linalg.norm(result - physical, axis=1).max() > MAX_ADDITIONAL_MM:
        return None

    # Verify on a third physical offset. A negative check leaves the proven
    # current surface available on the original labels.
    sample = np.flatnonzero(interior)[::17]
    before = _contrast(signal, current[sample], direction[sample], (.7,))
    after = _contrast(signal, proposal[sample], direction[sample], (.7,))
    if (not np.isfinite(before).all() or not np.isfinite(after).all()
            or np.median(after) < .95 * np.median(before)):
        return None
    checked = np.arange(0, len(faces), max(1, len(faces) // 16384))
    a = original_normals[checked]
    b = actual[checked]
    area_ratio = np.linalg.norm(b, axis=1).sum() / max(np.linalg.norm(a, axis=1).sum(), 1e-20)
    if not .975 <= area_ratio <= 1.025:
        return None
    if not _turn_decreases(faces, original_normals, actual, len(current)):
        return None
    return proposal, int(np.count_nonzero(backed_off))
