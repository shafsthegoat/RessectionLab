"""Transactional clearance of the source-bound score wall between rod poses."""

from concurrent import futures
from itertools import combinations
import threading

import numpy as np

from . import native_backend
from . import surgical_vessel_surface as surfaces


PROPERTY = 'vesselWallMotionGuard'
POLICY = 'continuous-source-triangles-v1'
CURRENT_POLICY = 'continuous-source-triangles-v2'
POLICIES = (POLICY, CURRENT_POLICY)
_MIN_AREA_RATIO = .2
_LOCAL_SEARCH_DEPTH = 8
_PARALLEL_MIN_FACES = 16_384
_HINGE_FILTER_MIN_FACES = 16_384
_STOCK_NATIVE_CLEAR = native_backend.wall_sweep_clear
_STOCK_NATIVE_CLEAR_CODE = _STOCK_NATIVE_CLEAR.__code__
_native_slot = threading.Lock()
_native_worker = None


def topology(frames):
    return tuple((wall.source.label, edited.identity if edited is not None else None)
                 for wall, _, _, _, _, edited in frames)


def _shared_edge_quads(faces):
    edges = np.sort(np.concatenate((faces[:, :2], faces[:, 1:], faces[:, ::2])), axis=1)
    opposite = np.concatenate((faces[:, 2], faces[:, 0], faces[:, 1]))
    order = np.lexsort(edges.T[::-1])
    edges, opposite = edges[order], opposite[order]
    starts = np.r_[0, np.flatnonzero(np.any(np.diff(edges, axis=0), axis=1)) + 1]
    counts = np.diff(np.r_[starts, len(edges)])
    paired = starts[counts == 2]
    regular = np.column_stack((edges[paired], opposite[paired], opposite[paired + 1]))
    irregular = [(*edges[lo], a, b) for lo, n in zip(starts[counts > 2], counts[counts > 2])
                 for a, b in combinations(opposite[lo:lo + n], 2)]
    return np.vstack((regular, np.asarray(irregular, dtype=int).reshape(-1, 4)))


def _face_normals(vertices):
    edge = vertices[:, 1] - vertices[:, 0]
    return np.cross(edge[:, None, :], vertices[:, 2:] - vertices[:, :1])


def _facet_areas_clear(reference, before, after):
    first = _face_normals(before)[:, 0]
    middle = _face_normals((before + after) / 2.)[:, 0]
    last = _face_normals(after)[:, 0]
    area = np.linalg.norm(first, axis=1)
    axis = first / np.maximum(area[:, None], 1e-30)
    half = np.einsum('ij,ij->i', middle, axis)
    end = np.einsum('ij,ij->i', last, axis)
    a = 2. * (end + area - 2. * half)
    b = end - area - a
    t = np.clip(-b / (2. * np.maximum(a, 1e-30)), 0., 1.)
    lowest = np.minimum(np.minimum(area, end), np.where(a > 0., area + t * (b + t * a), area))
    return bool(np.all(lowest >= _MIN_AREA_RATIO * np.maximum(area, reference)))


def _hinges_clear(before, after, depth=0):
    """Certify that edge neighbors cannot overlap across their common edge."""
    midpoint = (before + after) / 2.
    middle = _face_normals(midpoint)
    unit = middle / np.maximum(np.linalg.norm(middle, axis=2)[:, :, None], 1e-30)
    # Same-side coplanar faces have parallel normals when both use the same
    # shared-edge winding. Opposite projections exclude that event.
    axis = unit[:, 0] - unit[:, 1]
    at_mid = np.einsum('ijk,ik->ij', middle, axis)
    if np.any(at_mid[:, 0] <= 0.) or np.any(at_mid[:, 1] >= 0.):
        return False
    at_start = np.einsum('ijk,ik->ij', _face_normals(before), axis)
    at_end = np.einsum('ijk,ik->ij', _face_normals(after), axis)
    # Normal projections are quadratic. Their Bernstein coefficients bound
    # them for the entire interval, including intermediate coplanar poses.
    control = 2. * at_mid - .5 * (at_start + at_end)
    lower = np.minimum(np.minimum(at_start[:, 0], at_end[:, 0]), control[:, 0])
    upper = np.maximum(np.maximum(at_start[:, 1], at_end[:, 1]), control[:, 1])
    uncertain = (lower <= 0.) | (upper >= 0.)
    return not uncertain.any() or (depth < _LOCAL_SEARCH_DEPTH
                                   and _hinges_clear(before[uncertain], midpoint[uncertain], depth + 1)
                                   and _hinges_clear(midpoint[uncertain], after[uncertain], depth + 1))


def _local_face_guard(rest, faces, owners, *, hinge_filter=False):
    reference = np.linalg.norm(_face_normals(rest)[:, 0], axis=1)
    points = np.zeros((int(faces.max()) + 1, 3))
    point_owners = np.zeros(len(points), dtype=int)
    points[faces] = rest
    point_owners[faces] = owners
    adjacent = _shared_edge_quads(faces)
    adjacent_owners = point_owners[adjacent]
    shared_points = np.array_equal(points[faces], rest) and np.array_equal(point_owners[faces], owners)

    # Adjacent triangles reuse most owner comparisons. Compare each pair's
    # increments once, then gather the affected facets and hinges in order.
    pivot_owners = np.r_[np.repeat(owners[:, 0], 2), np.repeat(adjacent_owners[:, 0], 3)]
    other_owners = np.r_[owners[:, 1:].ravel(), adjacent_owners[:, 1:].ravel()]
    pairs, inverse = np.unique(np.sort(np.column_stack((pivot_owners, other_owners)), axis=1),
                              axis=0, return_inverse=True)
    first_owner, second_owner = pairs.T
    face_pairs = inverse[:2 * len(faces)].reshape(-1, 2)
    hinge_pairs = inverse[2 * len(faces):].reshape(-1, 3)

    def clear(prior, candidate, static):
        if static:
            facets = np.arange(len(faces))
            hinges = np.arange(len(adjacent))
        else:
            increment = candidate - prior
            if not np.any(increment):
                return True
            moving = np.any(increment[first_owner] != increment[second_owner], axis=1)
            facets = np.flatnonzero(moving[face_pairs].any(axis=1))
            hinges = np.flatnonzero(moving[hinge_pairs].any(axis=1))
            if not (len(facets) or len(hinges)):
                return True

        before = points + prior[point_owners]
        after = before if candidate is prior else points + candidate[point_owners]
        if len(facets):
            if shared_points:
                corners = faces[facets]
                start, end = before[corners], after[corners]
            else:
                start = rest[facets] + prior[owners[facets]]
                end = rest[facets] + candidate[owners[facets]]
            if not _facet_areas_clear(reference[facets], start, end):
                return False
        if len(hinges):
            corners = adjacent[hinges]
            separated = hinge_filter and not static and native_backend.wall_hinges_separated(before, after, corners)
            if not separated and not _hinges_clear(before[corners], after[corners]):
                return False
        return True

    return clear


def _serial_native_clear(check_local, source, prior, candidate, static):
    return (check_local is None or check_local(prior, candidate, static)) and (
        native_backend.wall_sweep_clear(source, prior, candidate, static))


def _clear_in_parallel(check_local, source, prior, candidate):
    global _native_worker
    # A busy worker never queues another engine behind it. Only the pure C++
    # check runs off-thread, using the owned read-only inputs to this call.
    if not _native_slot.acquire(blocking=False):
        return _serial_native_clear(check_local, source, prior, candidate, False)
    try:
        try:
            kernel = native_backend.verified_wall_sweep_kernel()
        except native_backend.NativeIntegrityError:
            return _serial_native_clear(check_local, source, prior, candidate, False)
        try:
            if _native_worker is None:
                _native_worker = futures.ThreadPoolExecutor(max_workers=1)
            prior.flags.writeable = False
            candidate.flags.writeable = False
            job = _native_worker.submit(kernel, source, prior, candidate, False)
        except RuntimeError:
            if _native_worker is not None:
                _native_worker.shutdown(cancel_futures=True)
                _native_worker = None
            return _serial_native_clear(check_local, source, prior, candidate, False)

        try:
            if not check_local(prior, candidate, False):
                return False
            # Keep the solver-thread integrity check in the original logical
            # order, after local approval and before using the native result.
            if kernel is not native_backend.verified_wall_sweep_kernel():
                raise native_backend.NativeIntegrityError('Native wall sweep entrypoint changed')
            return job.result()
        finally:
            # Even a local rejection or exception drains the native job before
            # the engine can advance, publish or observe cancellation.
            if not job.cancel():
                job.exception()
    finally:
        _native_slot.release()


def _build(frames, *, local_guard=False, shared_vertex=False):
    triangles, vertices, owners = [], [], []
    vertex_start = owner_start = 0
    for frame in frames:
        edited = frame[5]
        if edited is not None and len(edited.cuboids):
            raise ValueError('VESSEL_WALL_SWEEP_UNSUPPORTED: cut has no closed score-wall cap')
        rest, faces, bindings, offsets = surfaces.sweep_mesh(frame)
        triangles.append(rest)
        # Drop unused indices so multiple source layers have disjoint adjacency.
        unique, inverse = np.unique(faces, return_inverse=True)
        vertices.append(inverse.reshape(faces.shape) + vertex_start)
        owners.append(bindings + owner_start)
        vertex_start += len(unique)
        owner_start += len(offsets)
    geometry = (np.ascontiguousarray(np.concatenate(triangles), dtype=np.float64),
                np.ascontiguousarray(np.concatenate(vertices), dtype=np.int64),
                np.ascontiguousarray(np.concatenate(owners), dtype=np.int64), owner_start)
    hinge_filter = shared_vertex and len(geometry[0]) >= _HINGE_FILTER_MIN_FACES
    check_local = (_local_face_guard(*geometry[:3], hinge_filter=hinge_filter)
                   if local_guard and len(geometry[0]) else None)
    if native_backend.wall_sweep_available(shared_vertex=shared_vertex):
        source = native_backend.make_wall_sweep(*geometry, shared_vertex=shared_vertex)
        parallel = shared_vertex and check_local is not None and len(geometry[0]) >= _PARALLEL_MIN_FACES

        def clear(prior, candidate, static):
            if (parallel and not static and native_backend.wall_sweep_clear is _STOCK_NATIVE_CLEAR
                    and _STOCK_NATIVE_CLEAR.__code__ is _STOCK_NATIVE_CLEAR_CODE):
                return _clear_in_parallel(check_local, source, prior, candidate)
            return _serial_native_clear(check_local, source, prior, candidate, static)

        return clear
    from .surgical_wall_sweep_reference import WallSweepReference

    source = WallSweepReference(*geometry, shared_vertex=shared_vertex)

    def clear(prior, candidate, static):
        return (check_local is None or check_local(prior, candidate, static)) and (
            source.clear(prior, candidate, static=static))

    return clear


def is_clear(engine, prior, candidate, *, static=False):
    if not prior or not candidate:
        raise ValueError('VESSEL_WALL_SWEEP_UNSUPPORTED: native score-wall geometry is required')
    key = topology(candidate)
    if not static and topology(prior) != key:
        raise ValueError('VESSEL_WALL_SWEEP_UNSUPPORTED: source topology changed inside a rod step')
    transition_check = getattr(engine, '_transition_check', None)
    if transition_check is not None:
        transition_check()
    cached = getattr(engine, '_vessel_wall_sweep', None)
    if cached is None or cached[0] != key:
        from .surgical_vessel_rods import PROPERTY as RODS_PROPERTY, mechanics

        profile = engine.scene.properties.get(RODS_PROPERTY, {})
        shared_vertex = profile.get(PROPERTY) == CURRENT_POLICY
        cached = key, _build(candidate, local_guard=shared_vertex or mechanics(engine) is not None,
                             shared_vertex=shared_vertex)
        engine._vessel_wall_sweep = cached
    end = np.concatenate([surfaces.sweep_offsets(frame) for frame in candidate])
    start = np.concatenate([surfaces.sweep_offsets(frame) for frame in prior]) if not static else end
    clear = cached[1](start, end, static)
    if transition_check is not None:
        transition_check()
    return clear


def require_clear(engine, prior, candidate, *, static=False):
    if not is_clear(engine, prior, candidate, static=static):
        raise ValueError('VESSEL_WALL_SWEEP_UNSAFE: cannot certify native vessel wall clearance')
