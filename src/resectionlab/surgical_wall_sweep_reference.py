"""Conservative reference check for continuously transported native wall faces."""

from itertools import combinations
import operator

import numpy as np
from scipy.spatial import cKDTree


_CLEARANCE = 1e-9
_MAX_DEPTH = 12
_MAX_VISITS = 4_000_000
_MAX_COUNT = 500_000
_MAX_COORDINATE = 4096.
_MIN_AXIS_SQUARED = 1e-30
_BATCH_SIZE = 256
_AXES = np.eye(3)
# The displaced positions can be twice the input bound. Reserve roundoff for
# evaluating swept box centers and the tree's distance comparisons.
_TREE_MARGIN = _CLEARANCE + 8 * np.spacing(2 * _MAX_COORDINATE)


def _coordinates(value, shape):
    try:
        array = np.asarray(value)
        if array.shape != shape or array.dtype.kind not in 'fiu':
            return None
        array = np.asarray(array, dtype=np.float64)
    except (TypeError, ValueError, OverflowError):
        return None
    if not (np.isfinite(array).all() and np.all(np.abs(array) <= _MAX_COORDINATE)):
        return None
    return array


def _indices(value, shape, upper):
    array = np.asarray(value)
    if (array.shape != shape or array.dtype.kind not in 'iu'
            or np.any(array < 0) or np.any(array >= upper)):
        raise ValueError('Invalid wall sweep indices')
    return np.array(array, dtype=np.int64, copy=True)


def _axes(a, b):
    ae = np.roll(a, -1, axis=1) - a
    be = np.roll(b, -1, axis=1) - b
    an = np.cross(ae[:, 0], ae[:, 1])
    bn = np.cross(be[:, 0], be[:, 1])
    cross_edges = np.cross(ae[:, :, None], be[:, None, :]).reshape(len(a), 9, 3)
    axes = np.concatenate((np.broadcast_to(_AXES, (len(a), 3, 3)), an[:, None], bn[:, None],
                           cross_edges, np.cross(an[:, None], ae), np.cross(bn[:, None], be)), axis=1)
    squared = np.einsum('nki,nki->nk', axes, axes)
    return axes, _CLEARANCE * np.sqrt(squared), squared >= _MIN_AXIS_SQUARED


def _separating_sides(a, b, axes, margin, usable):
    origin = a[:, :1]
    pa = np.einsum('nvi,nki->nvk', a - origin, axes)
    pb = np.einsum('nvi,nki->nvk', b - origin, axes)
    return ((pa.max(axis=1) + margin < pb.min(axis=1)) & usable,
            (pb.max(axis=1) + margin < pa.min(axis=1)) & usable)


def _instantaneous_clear(a, b):
    left, right = _separating_sides(a, b, *_axes(a, b))
    return (left | right).any(axis=1)


def _interval_clear(a0, a1, b0, b1, budget):
    """Return clearance and the unused interval budget for the next pair batch."""
    pending = [(a0, a1, b0, b1, 0)]
    while pending:
        a0, a1, b0, b1, depth = pending.pop()
        budget -= len(a0)
        if budget <= 0:
            return False, budget
        am = .5 * a0 + .5 * a1
        bm = .5 * b0 + .5 * b1
        axes, margin, usable = _axes(am, bm)
        left0, right0 = _separating_sides(a0, b0, axes, margin, usable)
        left1, right1 = _separating_sides(a1, b1, axes, margin, usable)
        # One fixed axis and sign must certify both endpoints. Each pairwise
        # projected vertex difference then stays positive along the interval.
        possible = ~((left0 & left1) | (right0 & right1)).any(axis=1)
        if not possible.any():
            continue
        if depth == _MAX_DEPTH:
            return False, budget
        am, bm = am[possible], bm[possible]
        left, right = _separating_sides(am, bm, axes[possible], margin[possible], usable[possible])
        if not (left | right).any(axis=1).all():
            return False, budget
        pending.append((am, a1[possible], bm, b1[possible], depth + 1))
        pending.append((a0[possible], am, b0[possible], bm, depth + 1))
    return True, budget


def _signed_vertex_rays(a, b, corner_a, corner_b):
    rows = np.arange(len(a))
    offsets = np.arange(1, 3)
    left = a[rows[:, None], (corner_a[:, None] + offsets) % 3] - a[rows, corner_a, None]
    right = b[rows, corner_b, None] - b[rows[:, None], (corner_b[:, None] + offsets) % 3]
    return np.concatenate((left, right), axis=1)


def _vertex_axes(rays):
    lengths = np.linalg.norm(rays, axis=2, keepdims=True)
    unit = rays / np.maximum(lengths, _CLEARANCE)
    # Vertices, edge midpoints and facet normals of the unit-ray convex hull
    # cover the support feature nearest the origin.
    edges = [unit[:, a] + unit[:, b] for a, b in combinations(range(4), 2)]
    facets = [np.cross(unit[:, b] - unit[:, a], unit[:, c] - unit[:, a])
              for a, b, c in combinations(range(4), 3)]
    axes = np.stack((unit.sum(axis=1), *unit.transpose(1, 0, 2), *edges, *facets,
                     *(-normal for normal in facets)), axis=1)
    squared = np.einsum('nki,nki->nk', axes, axes)
    usable = (squared >= _MIN_AXIS_SQUARED) & (lengths > _CLEARANCE).all(axis=1)
    return axes, _CLEARANCE * np.sqrt(squared), usable


def _positive_rays(rays, axes, margin, usable):
    dots = np.einsum('nri,nki->nrk', rays, axes)
    return (dots > margin[:, None]).all(axis=1) & usable


def _vertex_interval_clear(before, after, budget):
    pending = [(before, after, 0)]
    while pending:
        before, after, depth = pending.pop()
        budget -= len(before)
        if budget <= 0:
            return False, budget
        middle = .5 * before + .5 * after
        axes = _vertex_axes(middle)
        middle_clear = _positive_rays(middle, *axes)
        if not middle_clear.any(axis=1).all():
            return False, budget
        # With a fixed plane through the common vertex, the signed ray
        # projections interpolate linearly between the checked endpoints.
        uncertain = ~(middle_clear & _positive_rays(before, *axes)
                       & _positive_rays(after, *axes)).any(axis=1)
        if not uncertain.any():
            continue
        if depth == _MAX_DEPTH:
            return False, budget
        pending.append((middle[uncertain], after[uncertain], depth + 1))
        pending.append((before[uncertain], middle[uncertain], depth + 1))
    return True, budget


class WallSweepReference:
    """Check packed owner offsets against the same rest-face geometry each time.

    Rest positions are face corners of shape ``(F, 3, 3)``. Face vertex indices
    and corner owner indices are each ``(F, 3)``. Owner offsets are ``(N, 3)``.
    """

    def __init__(self, rest, faces, owners, owner_count, *, shared_vertex=False):
        if isinstance(owner_count, (bool, np.bool_)):
            raise ValueError('Invalid wall sweep owner count')
        try:
            count = operator.index(owner_count)
        except TypeError as error:
            raise ValueError('Invalid wall sweep owner count') from error
        geometry = np.asarray(rest)
        if not 0 <= count <= _MAX_COUNT or geometry.ndim != 3 or len(geometry) > _MAX_COUNT:
            raise ValueError('Wall sweep geometry exceeds supported limits')
        size = len(geometry)
        points = _coordinates(geometry, (size, 3, 3))
        if points is None:
            raise ValueError('Invalid wall sweep rest positions')
        self._rest = points.copy()
        self._faces = _indices(faces, (size, 3), 3 * size)
        self._owners = _indices(owners, (size, 3), count)
        self._owner_count = count
        self._shared_vertex = shared_vertex
        if shared_vertex:
            order = np.argsort(self._faces.ravel(), kind='stable')
            repeated = np.diff(self._faces.ravel()[order]) == 0
            previous, following = order[:-1][repeated], order[1:][repeated]
            packed = self._rest.reshape(-1, 3)
            bindings = self._owners.ravel()
            if (np.any(previous // 3 == following // 3)
                    or not np.array_equal(bindings[previous], bindings[following])
                    or not np.array_equal(packed[previous], packed[following])):
                raise ValueError('Wall sweep requires a single position and owner per shared vertex')

    def clear(self, prior, candidate, *, static=False):
        """Return whether the evaluated geometry is certified clear."""
        shape = (self._owner_count, 3)
        prior = _coordinates(prior, shape)
        candidate = _coordinates(candidate, shape)
        if prior is None or candidate is None:
            return False
        if not len(self._rest):
            return True
        if static:
            dirty = np.ones(len(self._rest), dtype=bool)
        else:
            increment = candidate - prior
            moved = np.any(increment != 0., axis=1)
            dirty = moved[self._owners].any(axis=1)
            if not dirty.any():
                return True

        before = self._rest + prior[self._owners]
        after = self._rest + candidate[self._owners]
        low = np.minimum(before.min(axis=1), after.min(axis=1))
        high = np.maximum(before.max(axis=1), after.max(axis=1))
        centers = .5 * low + .5 * high
        radii = .5 * (high - low).max(axis=1)
        # Each box is contained in a Chebyshev ball of this radius, including
        # the small clearance and arithmetic allowance in the neighbor query.
        reach = radii + radii.max() + _TREE_MARGIN
        tree = cKDTree(centers)
        active = np.flatnonzero(dirty)
        neighbor_budget = pair_budget = interval_budget = _MAX_VISITS
        for start in range(0, len(active), _BATCH_SIZE):
            batch = active[start:start + _BATCH_SIZE]
            counts = tree.query_ball_point(centers[batch], reach[batch], p=np.inf, return_length=True)
            neighbor_budget -= int(counts.sum())
            if neighbor_budget <= 0:
                return False
            neighbors = tree.query_ball_point(centers[batch], reach[batch], p=np.inf, return_sorted=False)
            first = np.repeat(batch, counts)
            second = np.concatenate(neighbors).astype(np.intp, copy=False)
            # Count each dirty pair once, including a lower-index stationary face.
            relevant = (second > first) | ~dirty[second]
            first, second = first[relevant], second[relevant]
            # Match the native pair orientation for translation-invariant projections.
            first, second = np.minimum(first, second), np.maximum(first, second)
            relevant = ((low[first] <= high[second] + _CLEARANCE)
                        & (low[second] <= high[first] + _CLEARANCE)).all(axis=1)
            first, second = first[relevant], second[relevant]
            shared = self._faces[first, :, None] == self._faces[second, None, :]
            adjacency = shared.sum(axis=(1, 2))
            eligible = adjacency <= int(self._shared_vertex)
            if not static:
                left, right = increment[self._owners[first]], increment[self._owners[second]]
                common = (left == left[:, :1]).all(axis=(1, 2)) & (right == left[:, :1]).all(axis=(1, 2))
                eligible &= ~common
            pair_budget -= int(eligible.sum())
            if pair_budget <= 0:
                return False
            pinned = eligible & (adjacency == 1)
            first_pinned, second_pinned = first[pinned], second[pinned]
            corners = np.nonzero(shared[pinned])
            for lo in range(0, len(first_pinned), _BATCH_SIZE):
                part = slice(lo, lo + _BATCH_SIZE)
                a, b = first_pinned[part], second_pinned[part]
                corner_a, corner_b = corners[1][part], corners[2][part]
                final = _signed_vertex_rays(after[a], after[b], corner_a, corner_b)
                if static:
                    if not _positive_rays(final, *_vertex_axes(final)).any(axis=1).all():
                        return False
                    continue
                initial = _signed_vertex_rays(before[a], before[b], corner_a, corner_b)
                passed, interval_budget = _vertex_interval_clear(initial, final, interval_budget)
                if not passed:
                    return False
            detached = eligible & (adjacency == 0)
            first, second = first[detached], second[detached]
            for at in range(0, len(first), _BATCH_SIZE):
                a = first[at:at + _BATCH_SIZE]
                b = second[at:at + _BATCH_SIZE]
                if static:
                    if not _instantaneous_clear(after[a], after[b]).all():
                        return False
                    continue
                if not _instantaneous_clear(before[a], before[b]).all():
                    return False
                passed, interval_budget = _interval_clear(before[a], after[a], before[b], after[b], interval_budget)
                if not passed:
                    return False
        return True
