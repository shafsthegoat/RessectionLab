"""Centerline tubes: one vessel geometry for drawing, contact, rods and tags.

A vessel mask becomes a graph of centerline samples with a cross-section at
every sample: a radius plus angular harmonics in a rotation-minimising frame
(and, at a tip, an end-cap shape). Consecutive samples of an edge bound a round
cone (the convex hull of two balls) whose wall is displaced by the interpolated
section offsets,

    F_s(p) = d_cone(p) - (1 - t) h_a(w) - t h_b(w),

with ``t`` the axis parameter of ``p`` and ``w`` its unit direction from the
axis. The wall is the zero level of

    F(p) = max(min over segments F_s(p), box(p), cut(p))

a hard union of the segments (junction nodes add an inscribed ball; the
smooth-minimum junction term in ``_junction_term`` is kept for models with a
positive ``junction_blend`` but the current fit sets it to zero), floored by the
acquired box and by the model's ordered edit record: removed source cells cut
out as voxel boxes, extended one voxel through faces whose neighbour was not
vessel (``with_removed``, ``limits``). An edit never refits the sections.
Meshing, tags and owners use ``F`` (``raw``). Collision uses a lower bound on
the distance to the wall: each ``F_s`` divided by a bound ``L_s`` on its
gradient outside the wall, refined to a safe radius (see ``TubeModel.bounds``
and ``TubeModel._segment_field``).

Centerlines come from ``vascular_source_branches.build_source_branches`` (Lee
skeleton with certified H0/H1 representative edges). Samples are smoothed on
the graph (Taubin), recentred, and given sub-voxel radii from rays in the
normal plane to the ``LEVEL`` crossing of the trilinear soft signal (a model
probability, or the mask itself). Radii and sections are then fitted by
regularised least squares to the source half-level wall points each segment
owns. Everything is in the physical frame of the source affine, RAS+
millimetres; mesh positions are also kept in source IJK.
"""

from dataclasses import dataclass, replace
from functools import cached_property
from itertools import chain, product
import time

import numpy as np
from scipy import ndimage as ndi, sparse
from scipy.sparse.csgraph import connected_components
from scipy.sparse.linalg import spsolve
from scipy.spatial import cKDTree

from .patient_anatomy import affine as checked_affine
from .vascular_source_branches import _mask_digest, build_source_branches

MODEL = 'centerline-tubes-v2'
LEVEL = .5
RAYS = 16
RAY_STEPS = 48
BISECTIONS = 10
SMOOTHING_ROUNDS = 8
RECENTRE_ROUNDS = 2
BLEND_FRACTION = 0.
BLEND_REACH = 1.5
SURFACE_FIT_ROUNDS = 2
FIT_ACCEPT = 1.
FIT_MIN_POINTS = 12
SECTION_ORDER = 6
SECTION_ROUNDS = 3
SECTION_REACH = 1.5
SECTION_SMOOTH = 1.
SECTION_RIDGE = .05
SECTION_LIMIT = .6
MESH_SUBDIVISION = 2
BLOCK = 96
CLASS_NAMES = {1: 'artery', 2: 'vein', 3: 'unclassified'}


def _round_cone(p, a, b, ra, rb):
    """Exact signed distance to the convex hull of two balls (I. Quilez)."""
    ba = b - a
    l2 = np.sum(ba * ba, axis=-1)
    rr = ra - rb
    a2 = l2 - rr * rr
    pa = p - a
    y = np.sum(pa * ba, axis=-1)
    z = y - l2
    x2 = np.sum((pa * l2[..., None] - ba * y[..., None]) ** 2, axis=-1)
    y2, z2 = y * y * l2, z * z * l2
    k = np.sign(rr) * rr * rr * x2
    safe = np.where(l2 > 0, l2, 1.)
    il2 = 1. / safe
    cone = np.where(np.sign(z) * a2 * z2 > k, np.sqrt(x2 + z2) * il2 - rb,
                    np.where(np.sign(y) * a2 * y2 < k, np.sqrt(x2 + y2) * il2 - ra,
                             (np.sqrt(np.maximum(x2 * a2 * il2, 0.)) + y * rr) * il2 - ra))
    # One ball contains the other (or the cone has no length): their union.
    balls = np.minimum(np.linalg.norm(pa, axis=-1) - ra, np.linalg.norm(p - b, axis=-1) - rb)
    return np.where(a2 > 1e-12 * np.maximum(l2, 1e-12), cone, balls)


def _smin(a, b, k):
    h = np.maximum(k - np.abs(a - b), 0.) / np.maximum(k, 1e-12)
    return np.minimum(a, b) - h * h * k * .25


def _section_basis(x, y, axial, order):
    """Columns Re z^n, Im z^n (n = 1..order, z = x + iy) then axial^2 and axial^4."""
    out = np.empty((len(x), 2 * order + 2))
    re, im = x, y
    for n in range(order):
        out[:, 2 * n], out[:, 2 * n + 1] = re, im
        re, im = re * x - im * y, re * y + im * x
    out[:, -2] = axial * axial
    out[:, -1] = out[:, -2] ** 2
    return out


def _section_swell(c):
    """Largest offset of section coefficients (n, columns) over all directions."""
    return np.hypot(c[:, 0:-2:2], c[:, 1:-2:2]).sum(axis=1) + np.abs(c[:, -2:]).sum(axis=1)


def _lipschitz(bounds, side, cap, rows=slice(None)):
    along, taper = bounds['along'][rows], bounds['taper'][rows]
    value = np.maximum(1 + along ** 2 + side ** 2 + 2 * taper * along, 1 + cap ** 2 + 2 * taper * cap)
    return np.where(np.isfinite(value), np.sqrt(value), 1e3)


def _section_derivatives(c):
    """Bounds on |grad h| and on its change per radian of direction, per row of coefficients."""
    order = (c.shape[1] - 2) // 2
    n = np.arange(1, order + 1)
    size = np.hypot(c[:, 0:-2:2], c[:, 1:-2:2])
    first = (n * size).sum(axis=1) + 2 * np.abs(c[:, -2]) + 4 * np.abs(c[:, -1])
    return first, (n * n * size).sum(axis=1) + 2 * np.abs(c[:, -2]) + 12 * np.abs(c[:, -1]) + first


def _section_eval(c, u, v, outward, w):
    """Offset h and its gradient in the direction w (directions (k, rows, 3)), per row of coefficients."""
    order = (c.shape[1] - 2) // 2
    x, y = np.einsum('krj,rj->kr', w, u), np.einsum('krj,rj->kr', w, v)
    axial = np.maximum(np.einsum('krj,rj->kr', w, outward), 0.)
    h = c[:, -2] * axial ** 2 + c[:, -1] * axial ** 4
    du = dv = 0.
    re, im = np.ones_like(x), np.zeros_like(x)
    for k in range(order):
        alpha, beta = c[:, 2 * k], c[:, 2 * k + 1]
        du = du + (k + 1) * (alpha * re + beta * im)
        dv = dv + (k + 1) * (beta * re - alpha * im)
        re, im = re * x - im * y, re * y + im * x
        h = h + alpha * re + beta * im
    dx = 2 * c[:, -2] * axial + 4 * c[:, -1] * axial ** 3
    return h, du[..., None] * u + dv[..., None] * v + dx[..., None] * outward


_DIRECTIONS = {}


def _gradient_bounds(ca, cb, ua, va, ub, vb, a, b, ra, rb, circle=256, sphere=4096, cover=.045):
    """Per segment (rows), bounds A, B on the offset gradient beside the axis and C at the end caps
    (and the angular slopes behind B and C, before division by the axis distance).

    Beside the axis the direction w is normal to it: A bounds |h_b(w) - h_a(w)| / length and B the
    gradient of h along the circle of w over the axis distance, which outside the wall exceeds the
    interpolated wall radius (the round cone contains the ball of the interpolated radius about each
    axis point), so B is the larger end's ratio. At a cap w ranges over the outer hemisphere and C is
    the tangential gradient over ``radius + h``. Directions are sampled; every sample stands for the
    arc or cap within half its spacing, over which values move by at most that angle times the
    derivative bounds.
    """
    length = np.linalg.norm(b - a, axis=1)
    axis = (b - a) / length[:, None]
    first_a, second_a = _section_derivatives(ca)
    first_b, second_b = _section_derivatives(cb)
    e1, e2 = _basis(axis)
    angle = np.arange(circle) * 2 * np.pi / circle
    w = np.cos(angle)[:, None, None] * e1 + np.sin(angle)[:, None, None] * e2
    along = np.cross(axis, w)
    arc = np.pi / circle
    ha, ga = _section_eval(ca, ua, va, -axis, w)
    hb, gb = _section_eval(cb, ub, vb, axis, w)
    change = (np.abs(hb - ha) + arc * (first_a + first_b)).max(axis=0) / length
    slope_a = np.abs(np.sum(ga * along, axis=-1)) + arc * second_a
    slope_b = np.abs(np.sum(gb * along, axis=-1)) + arc * second_b
    ratio = lambda slope, h, r, first: np.where(r + h - arc * first > 0, slope / np.maximum(r + h - arc * first, 1e-12), np.inf)
    side = np.maximum(ratio(slope_a, ha, ra, first_a), ratio(slope_b, hb, rb, first_b)).max(axis=0)
    side_slope = np.maximum(slope_a, slope_b).max(axis=0)
    if sphere not in _DIRECTIONS:
        full = _fibonacci(sphere)
        _DIRECTIONS[sphere] = full[full[:, 2] >= -cover]
    cap = _DIRECTIONS[sphere]
    caps, cap_slopes = [], []
    for c, u, v, sign, r, first, second in ((ca, ua, va, -1, ra, first_a, second_a), (cb, ub, vb, 1, rb, first_b, second_b)):
        result, steepest = np.empty(len(c)), np.empty(len(c))
        for start in range(0, len(c), 64):
            rows = slice(start, start + 64)
            w = cap[:, None, :1] * e1[rows] + cap[:, None, 1:2] * e2[rows] + cap[:, None, 2:] * (sign * axis[rows])
            h, g = _section_eval(c[rows], u[rows], v[rows], sign * axis[rows], w)
            g = g - np.sum(g * w, axis=-1)[..., None] * w
            wall = r[rows] + h - cover * first[rows]
            slope = np.linalg.norm(g, axis=-1) + cover * second[rows]
            result[rows] = np.where(wall > 0, slope / np.maximum(wall, 1e-12), np.inf).max(axis=0)
            steepest[rows] = slope.max(axis=0)
        caps.append(result)
        cap_slopes.append(steepest)
    return change, side, np.maximum(*caps), side_slope, np.maximum(*cap_slopes)


def _edge_frames(points, edges):
    """Rotation-minimising normal frames (u, v) per edge sample, concatenated in edge order (double reflection)."""
    us, vs = [], []
    for samples in edges:
        p = points[samples]
        if len(p) < 2 or not np.any(np.ptp(p, axis=0) > 0):
            u, v = _basis(np.array([[0., 0., 1.]]))
            us.append(np.repeat(u, len(p), 0))
            vs.append(np.repeat(v, len(p), 0))
            continue
        t = np.gradient(p, axis=0)
        norm = np.linalg.norm(t, axis=1)
        t[norm < 1e-12] = p[-1] - p[0]
        t /= np.linalg.norm(t, axis=1)[:, None]
        u = np.empty_like(p)
        u[0] = _basis(t[:1])[0][0]
        for i in range(len(p) - 1):
            r, tt = u[i], t[i]
            v1 = p[i + 1] - p[i]
            c1 = v1 @ v1
            if c1 > 1e-18:
                r, tt = r - 2 / c1 * (v1 @ r) * v1, tt - 2 / c1 * (v1 @ tt) * v1
            v2 = t[i + 1] - tt
            c2 = v2 @ v2
            if c2 > 1e-18:
                r = r - 2 / c2 * (v2 @ r) * v2
            r = r - (r @ t[i + 1]) * t[i + 1]
            length = np.linalg.norm(r)
            u[i + 1] = r / length if length > 1e-9 else _basis(t[i + 1:i + 2])[0][0]
        us.append(u)
        vs.append(np.cross(t, u))
    if not us:
        return np.zeros((0, 3)), np.zeros((0, 3))
    return np.concatenate(us), np.concatenate(vs)


@dataclass(frozen=True)
class TubeModel:
    affine: np.ndarray
    shape: tuple
    points: np.ndarray
    edges: tuple
    edge_radii: tuple
    edge_class: tuple
    edge_label: tuple
    edge_cells: tuple
    junctions: np.ndarray
    junction_edges: tuple
    junction_radius: np.ndarray
    junction_blend: np.ndarray
    junction_reach: np.ndarray
    receipt: dict
    edge_sections: tuple = None
    removed: tuple = ()
    removed_open: tuple = ()

    @cached_property
    def spacing(self):
        return np.linalg.norm(self.affine[:3, :3], axis=0)

    @cached_property
    def sections(self):
        """Section coefficients per edge sample, concatenated in edge order."""
        if self.edge_sections is None or not len(self.edge_sections):
            return np.zeros((sum(len(e) for e in self.edges), 2 * SECTION_ORDER + 2))
        return np.concatenate([np.asarray(c, float).reshape(len(e), -1) for c, e in zip(self.edge_sections, self.edges)])

    @cached_property
    def frames(self):
        return _edge_frames(self.points, self.edges)

    @cached_property
    def segments(self):
        """Section cones (a, b, ra, rb, owner, edge samples ga/gb, frames, coefficients, gradient bound).

        Owners past the edge count are node balls (no edge samples, ga = gb = -1).
        """
        a, b, ra, rb, owner, ga = [], [], [], [], [], []
        offset = 0
        for e, (samples, radii) in enumerate(zip(self.edges, self.edge_radii)):
            index = offset + np.arange(len(samples))
            offset += len(samples)
            if len(samples) == 1:
                samples, radii, index = np.r_[samples, samples], np.r_[radii, radii], np.r_[index, index]
            a.append(self.points[samples[:-1]])
            b.append(self.points[samples[1:]])
            ra.append(radii[:-1])
            rb.append(radii[1:])
            owner.append(np.full(len(samples) - 1, e))
            ga.append(np.column_stack((index[:-1], index[1:])))
        centres = self.points[self.junctions].reshape(-1, 3)
        a.append(centres)
        b.append(centres)
        ra.append(self.junction_radius)
        rb.append(self.junction_radius)
        owner.append(len(self.edges) + np.arange(len(self.junctions)))
        ga.append(np.full((len(self.junctions), 2), -1))
        a, b = np.concatenate(a), np.concatenate(b)
        ra, rb = np.concatenate(ra).astype(float), np.concatenate(rb).astype(float)
        ga = np.concatenate(ga).astype(np.int64)
        sections, (u, v) = self.sections, self.frames
        columns = sections.shape[1]
        real = ga[:, 0] >= 0
        pick = lambda values, i, width: np.where(real[:, None], values[np.maximum(ga[:, i], 0)], 0.) if len(values) else np.zeros((len(ga), width))
        ca, cb = pick(sections, 0, columns), pick(sections, 1, columns)
        ua, va, ub, vb = pick(u, 0, 3), pick(v, 0, 3), pick(u, 1, 3), pick(v, 1, 3)
        length = np.linalg.norm(b - a, axis=1)
        swell = np.maximum(_section_swell(ca), _section_swell(cb))
        shaped = (swell > 0) & (length > 0)
        centre = (a + b) / 2
        bound = length / 2 + np.maximum(ra, rb) + swell
        return {'a': a, 'b': b, 'ra': ra, 'rb': rb, 'owner': np.concatenate(owner).astype(np.int64), 'ga': ga,
                'ua': ua, 'va': va, 'ub': ub, 'vb': vb, 'ca': ca, 'cb': cb, 'shaped': shaped, 'swell': swell,
                'centre': centre, 'bound': bound, 'tree': cKDTree(centre) if len(centre) else None}

    @cached_property
    def bounds(self):
        """Per segment, bounds on the gradient of ``d_cone - offset`` outside the wall.

        Beside the axis grad(offset) = A (axis direction) + B (around the axis); the cone gradient lies
        in the plane of the axis and the point, at most the taper angle from normal to the axis, so
        |grad|^2 <= 1 + A^2 + B^2 + 2 sin(taper) A. At an end cap the offset gradient is tangential
        to the end ball, which the cone gradient leaves by at most the taper angle:
        |grad|^2 <= 1 + C^2 + 2 sin(taper) C. B and C are angular slopes over the axis distance, which
        outside the wall exceeds the wall radius; ``slope`` and ``capSlope`` are those slopes alone.
        See ``_gradient_bounds``.
        """
        seg = self.segments
        count = len(seg['a'])
        out = {key: np.zeros(count) for key in ('along', 'side', 'cap', 'slope', 'capSlope', 'taper')}
        rows = np.flatnonzero(seg['shaped'])
        if len(rows):
            a, b, ra, rb = seg['a'][rows], seg['b'][rows], seg['ra'][rows], seg['rb'][rows]
            found = _gradient_bounds(seg['ca'][rows], seg['cb'][rows], seg['ua'][rows], seg['va'][rows],
                                     seg['ub'][rows], seg['vb'][rows], a, b, ra, rb)
            for key, value in zip(('along', 'side', 'cap', 'slope', 'capSlope'), found):
                out[key][rows] = value
            out['taper'][rows] = np.minimum(np.abs(ra - rb) / np.linalg.norm(b - a, axis=1), 1.)
        out['lip'] = _lipschitz(out, out['side'], out['cap'])
        return out

    @property
    def lipschitz(self):
        return self.bounds['lip']

    def _terms(self, p, s):
        """Axis parameter, frame coordinates of the axis direction at both ends and the end-cap coordinates."""
        seg = self.segments
        a, ab = seg['a'][s], seg['b'][s] - seg['a'][s]
        l2 = np.einsum('ij,ij->i', ab, ab)
        t = np.clip(np.einsum('ij,ij->i', p - a, ab) / np.maximum(l2, 1e-30), 0., 1.)
        w = p - a - t[:, None] * ab
        w /= np.maximum(np.linalg.norm(w, axis=1), 1e-12)[:, None]
        along = np.einsum('ij,ij->i', w, ab) / np.sqrt(np.maximum(l2, 1e-30))
        dot = lambda x: np.einsum('ij,ij->i', w, seg[x][s])
        return t, (dot('ua'), dot('va'), np.maximum(-along, 0.)), (dot('ub'), dot('vb'), np.maximum(along, 0.))

    def _offset(self, p, s):
        seg = self.segments
        order = (seg['ca'].shape[1] - 2) // 2
        t, end_a, end_b = self._terms(p, s)
        ha = np.einsum('ij,ij->i', _section_basis(*end_a, order), seg['ca'][s])
        hb = np.einsum('ij,ij->i', _section_basis(*end_b, order), seg['cb'][s])
        return (1 - t) * ha + t * hb

    def _segment_field(self, p, s, *, index=None, size=None, cap=np.inf, raw=False, safe=True):
        """Field of segments ``s`` at points ``p`` (paired rows).

        ``F = d_cone - offset`` is divided by the segment's gradient bound ``L`` (``raw`` skips this), so it
        stays 1-Lipschitz outside the wall. With ``safe``, a positive value is instead the largest radius
        ``r`` with ``r L(r) <= F``, where ``L(r)`` bounds the gradient over the ball of radius ``r`` (axis
        distance at least the point's less ``r``): a lower bound on the distance to the segment's wall,
        between ``F / L`` and ``F``. With ``index`` (target per row, ``size`` targets), a row is evaluated
        exactly only when its lower bound can beat the best upper bound of its target (or ``cap``);
        otherwise its upper bound, which leaves every per-target minimum unchanged, is returned.
        """
        seg = self.segments
        cone = _round_cone(p, seg['a'][s], seg['b'][s], seg['ra'][s], seg['rb'][s])
        d = cone
        shaped = seg['shaped'][s]
        if index is not None and shaped.any():
            swell, lip = seg['swell'][s], 1. if raw else self.lipschitz[s]
            upper = np.where(cone + swell > 0, cone + swell, (cone + swell) / lip)
            lower = np.where(cone - swell < 0, cone - swell, (cone - swell) / lip)
            best = np.full(size, cap, float)
            np.minimum.at(best, index, upper)
            d = np.where(shaped, upper, cone)
            shaped &= lower < best[index]
        shaped = np.flatnonzero(shaped)
        if len(shaped):
            k = s[shaped]
            value = cone[shaped] - self._offset(p[shaped], k)
            d[shaped] = value if raw else value / self.lipschitz[k]
            if safe and not raw:
                out = np.flatnonzero(value > 0)
                d[shaped[out]] = self._safe_radius(p[shaped[out]], k[out], value[out])
        return d

    def _safe_radius(self, p, s, value, steps=12):
        bounds, seg = self.bounds, self.segments
        a, ab = seg['a'][s], seg['b'][s] - seg['a'][s]
        t = np.clip(np.einsum('ij,ij->i', p - a, ab) / np.maximum(np.einsum('ij,ij->i', ab, ab), 1e-30), 0., 1.)
        rho = np.linalg.norm(p - a - t[:, None] * ab, axis=1)
        low, high = value / bounds['lip'][s], value.copy()
        for _ in range(steps):
            r = (low + high) / 2
            near = np.maximum(rho - r, 1e-12)
            side = np.minimum(bounds['side'][s], bounds['slope'][s] / near)
            cap = np.minimum(bounds['cap'][s], bounds['capSlope'][s] / near)
            fits = r * _lipschitz(bounds, side, cap, s) <= value
            low, high = np.where(fits, r, low), np.where(fits, high, r)
        return low

    @cached_property
    def junction_members(self):
        """Edges (sorted) then the node ball, padded with -1."""
        width = max((len(edges) for edges in self.junction_edges), default=0) + 1
        members = np.full((len(self.junctions), width), -1, np.int64)
        for j, edges in enumerate(self.junction_edges):
            members[j, :len(edges)] = edges
            members[j, len(edges)] = len(self.edges) + j
        return members

    @cached_property
    def junction_tree(self):
        return cKDTree(self.points[self.junctions].reshape(-1, 3))

    @cached_property
    def junction_segments(self):
        """Per junction, the member segments that can reach its blend region."""
        seg = self.segments
        result = []
        for j, members in enumerate(self.junction_members):
            centre = self.points[self.junctions[j]]
            near = seg['tree'].query_ball_point(centre, 2 * self.junction_reach[j] + seg['bound'].max())
            near = np.asarray(sorted(near), np.int64)
            result.append(near[np.isin(seg['owner'][near], members[members >= 0])])
        return tuple(result)

    def _junction_term(self, j, points, distances):
        """Smooth union of member distances (n, members) at points of junction j.

        The fade of ``k`` adds at most ``blend / (4 reach)`` to the gradient; dividing by that
        bound keeps the term 1-Lipschitz without moving its zero level.
        """
        centre = self.points[self.junctions[j]]
        reach = self.junction_reach[j]
        fade = np.clip((2 * reach - np.linalg.norm(points - centre, axis=1)) / reach, 0., 1.)
        k = self.junction_blend[j] * fade
        value = distances[:, 0]
        for column in range(1, distances.shape[1]):
            value = _smin(value, distances[:, column], k)
        return value / (1 + self.junction_blend[j] / (4 * max(reach, 1e-12)))

    def _member_distances(self, j, points, raw=False):
        seg = self.segments
        members = self.junction_members[j]
        members = members[members >= 0]
        near = self.junction_segments[j]
        owner = seg['owner'][near]
        result = np.empty((len(points), len(members)))
        step = max(1, 200_000 // max(len(near), 1))
        for start in range(0, len(points), step):
            p = points[start:start + step]
            d = self._segment_field(np.repeat(p, len(near), 0), np.tile(near, len(p)), raw=raw, safe=False).reshape(len(p), len(near))
            for column, m in enumerate(members):
                result[start:start + step, column] = np.min(np.where(owner == m, d, np.inf), axis=1)
        return result

    def with_removed(self, cells, open_faces=None):
        """The same fitted model with removed source cells appended to its ordered edit record.

        ``open_faces`` (6 bits per cell: -i, +i, -j, +j, -k, +k) marks faces whose neighbour was not a
        vessel cell; the cut extends outward through them, so wall that bulges past the voxels goes too.
        """
        record, flags = list(self.removed), list(self.removed_open or (0,) * len(self.removed))
        seen = set(record)
        for i, cell in enumerate(cells):
            if int(cell) not in seen:
                seen.add(int(cell))
                record.append(int(cell))
                flags.append(0 if open_faces is None else int(open_faces[i]))
        return replace(self, removed=tuple(record), removed_open=tuple(flags))

    @cached_property
    def _removed_cells(self):
        ijk = np.column_stack(np.unravel_index(np.asarray(self.removed, np.int64), self.shape)).astype(float)
        flags = np.asarray(self.removed_open or (0,) * len(self.removed), np.int64)
        # Wall bulges less than a voxel past the cells; a voxel more covers it without reaching a neighbour.
        bits = (flags[:, None] >> np.arange(6)) & 1
        low = (.5 + bits[:, 0::2]) * self.spacing
        high = (.5 + bits[:, 1::2]) * self.spacing
        centre = ijk + (high - low) / 2 / self.spacing
        reach = float(self.segments['bound'].max(initial=0.)) + 4 * float(self.spacing.max())
        return centre, (high + low) / 2, cKDTree(ijk @ self.affine[:3, :3].T + self.affine[:3, 3]), reach

    def limits(self, points):
        """Field floor from the acquired box and the removed cells (each a voxel box, cut out flat)."""
        value = self.acquired(points)
        if not self.removed:
            return value
        centre, half, tree, reach = self._removed_cells
        distance, near = tree.query(points, k=min(16, len(centre)), distance_upper_bound=reach)
        distance, near = distance.reshape(len(points), -1), near.reshape(len(points), -1)
        close = np.flatnonzero(np.isfinite(distance[:, 0]))
        if not len(close):
            return value
        inverse = np.linalg.inv(self.affine)
        local = points[close] @ inverse[:3, :3].T + inverse[:3, 3]
        found = near[close]
        valid = found < len(centre)
        pick = np.minimum(found, len(centre) - 1)
        q = np.abs(local[:, None] - centre[pick]) * self.spacing - half[pick]
        box = np.linalg.norm(np.maximum(q, 0.), axis=-1) + np.minimum(q.max(axis=-1), 0.)
        value[close] = np.maximum(value[close], -np.min(np.where(valid, box, np.inf), axis=1))
        return value

    def acquired(self, points):
        """Signed distance to the acquired box, half a voxel past the outer sample centres."""
        inverse = np.linalg.inv(self.affine)
        ijk = points @ inverse[:3, :3].T + inverse[:3, 3]
        half = np.asarray(self.shape) / 2
        q = (np.abs(ijk - (half - .5)) - half) * self.spacing
        return np.linalg.norm(np.maximum(q, 0.), axis=-1) + np.minimum(q.max(axis=-1), 0.)

    def nearest_segments(self, points, *, reach, edges_only=False, raw=False):
        """Smallest segment field (no junction blends) and its segment per point; -1 beyond ``reach``."""
        points = np.asarray(points, float).reshape(-1, 3)
        seg = self.segments
        value = np.full(len(points), float(reach))
        which = np.full(len(points), -1, np.int64)
        if not len(points) or seg['tree'] is None:
            return value, which
        for start in range(0, len(points), 20000):
            chunk = points[start:start + 20000]
            lists = seg['tree'].query_ball_point(chunk, reach + seg['bound'].max())
            counts = np.fromiter(map(len, lists), np.int64, len(chunk))
            pi = np.repeat(np.arange(len(chunk)), counts)
            si = np.fromiter(chain.from_iterable(lists), np.int64, int(counts.sum()))
            keep = np.linalg.norm(chunk[pi] - seg['centre'][si], axis=1) <= reach + seg['bound'][si]
            if edges_only:
                keep &= seg['ga'][si, 0] >= 0
            pi, si = pi[keep], si[keep]
            if not len(pi):
                continue
            d = self._segment_field(chunk[pi], si, index=pi, size=len(chunk), cap=reach, raw=raw)
            order = np.lexsort((d, pi))
            pi, si, d = pi[order], si[order], d[order]
            first = np.r_[True, pi[1:] != pi[:-1]]
            hit = d[first] < reach
            value[start + pi[first][hit]] = d[first][hit]
            which[start + pi[first][hit]] = si[first][hit]
        return value, which

    def evaluate(self, points, *, reach=None, raw=False):
        """Field value and owning edge at points; values are capped at ``reach`` (mm).

        ``raw`` skips the gradient normalisation (same zero level; no longer a distance bound).
        """
        points = np.asarray(points, float).reshape(-1, 3)
        reach = float(4 * self.spacing.max() if reach is None else reach)
        n = len(points)
        value = np.full(n, reach)
        owner = np.full(n, -1, np.int64)
        seg = self.segments
        if not n or seg['tree'] is None:
            return value, owner
        for start in range(0, n, 20000):
            rows = slice(start, start + 20000)
            chunk = points[rows]
            local_value, which = self.nearest_segments(chunk, reach=reach, raw=raw)
            local_owner = np.where(which >= 0, seg['owner'][which], -1)
            if len(self.junctions):
                pairs = self.junction_tree.query_ball_point(chunk, 2 * self.junction_reach.max())
                jcounts = np.fromiter(map(len, pairs), np.int64, len(chunk))
                pj = np.repeat(np.arange(len(chunk)), jcounts)
                jj = np.fromiter(chain.from_iterable(pairs), np.int64, int(jcounts.sum()))
                for j in np.unique(jj[self.junction_blend[jj] > 0]):
                    rows_j = pj[jj == j]
                    rows_j = rows_j[local_value[rows_j] < reach]
                    if not len(rows_j):
                        continue
                    term = self._junction_term(j, chunk[rows_j], self._member_distances(j, chunk[rows_j], raw=raw))
                    local_value[rows_j] = np.minimum(local_value[rows_j], term)
            value[rows] = np.minimum(np.maximum(local_value, self.limits(chunk)), reach)
            owner[rows] = local_owner
        ball = owner >= len(self.edges)
        owner[ball] = [self.junction_edges[j][0] if len(self.junction_edges[j]) else -1
                       for j in owner[ball] - len(self.edges)]
        return value, owner

    def query(self, points, *, reach=None):
        """Distance bound (mm), outward unit normal, closest wall point and owning edge.

        The distance is the field value: exact inside a single section cone, elsewhere a lower bound
        outside the wall. The closest point is projected onto the zero level by Newton steps on the raw
        field. Beyond ``reach`` the distance is ``inf``, normal and closest point are NaN and the owner -1.
        """
        points = np.asarray(points, float).reshape(-1, 3)
        reach = float(4 * self.spacing.max() if reach is None else reach)
        values, normal, owner = self._probe(points, reach, raw=False)
        beyond = values >= reach
        distance = np.where(beyond, np.inf, values)
        closest = points - values[:, None] * normal
        near = np.flatnonzero(~beyond)
        for _ in range(4):
            if not len(near):
                break
            raw, gradient, _ = self._probe(closest[near], reach, raw=True, unit=False)
            step = raw / np.maximum(np.einsum('ij,ij->i', gradient, gradient), 1e-24)
            closest[near] -= step[:, None] * gradient
        closest[beyond] = np.nan
        normal[beyond] = np.nan
        owner[beyond] = -1
        return distance, normal, closest, owner

    def _probe(self, points, reach, *, raw, unit=True):
        eps = 1e-3 * float(self.spacing.min())
        offsets = np.vstack((np.zeros(3), np.eye(3) * eps, -np.eye(3) * eps))
        values, owner = self.evaluate((points[:, None] + offsets).reshape(-1, 3), reach=reach, raw=raw)
        values = values.reshape(len(points), 7)
        gradient = (values[:, 1:4] - values[:, 4:7]) / (2 * eps)
        if unit:
            gradient = gradient / np.maximum(np.linalg.norm(gradient, axis=1), 1e-12)[:, None]
        return values[:, 0], gradient, owner.reshape(len(points), 7)[:, 0]

    @property
    def wall_tolerance(self):
        """How far (mm) the meshed wall may lie outside the zero level: one mesh grid step.

        Marching-cubes triangles cut the concave creases where section cones meet; ``audit(..., model=)``
        measures the largest excursion (``wall_outside_max_mm``), which must stay below this.
        """
        return float(self.spacing.min()) / MESH_SUBDIVISION

    def capsule_clearance(self, a, b, radius, *, step=None, tolerance=None):
        """Smallest wall clearance (mm) of a tool capsule, its axis point and the wall normal there.

        The clearance is a lower bound: the smallest distance bound over axis samples, less half the
        sampling step (the true distance is 1-Lipschitz) and less ``tolerance`` (default
        ``wall_tolerance``; a caller holding an audited mesh may pass its ``wall_outside_max_mm``), so a
        positive clearance means the capsule meets neither the implicit nor the meshed wall.
        """
        a, b = np.asarray(a, float), np.asarray(b, float)
        step = float(self.spacing.min() / 8 if step is None else step)
        count = max(2, int(np.ceil(np.linalg.norm(b - a) / step)) + 1)
        axis = a + np.linspace(0., 1., count)[:, None] * (b - a)
        distance, normal, _, owner = self.query(axis, reach=radius + 4 * self.spacing.max())
        i = int(np.argmin(distance))
        gap = np.linalg.norm(b - a) / (count - 1) / 2
        tolerance = self.wall_tolerance if tolerance is None else float(tolerance)
        return float(distance[i] - radius - gap - tolerance), axis[i], normal[i], int(owner[i])

    def to_dict(self):
        return {
            'model': MODEL, 'affine': self.affine.tolist(), 'shape': list(self.shape),
            'points': self.points.tolist(), 'edges': [e.tolist() for e in self.edges],
            'edgeRadii': [r.tolist() for r in self.edge_radii], 'edgeClass': list(self.edge_class),
            'edgeLabel': [int(x) for x in self.edge_label], 'edgeCells': [list(map(int, c)) for c in self.edge_cells],
            'junctions': self.junctions.tolist(), 'junctionEdges': [list(map(int, e)) for e in self.junction_edges],
            'junctionRadius': self.junction_radius.tolist(), 'junctionBlend': self.junction_blend.tolist(),
            'junctionReach': self.junction_reach.tolist(),
            'edgeSections': [c.tolist() for c in np.split(self.sections, np.cumsum([len(e) for e in self.edges])[:-1])],
            'receipt': self.receipt,
            **({'removedCells': list(self.removed), 'removedOpenFaces': list(self.removed_open)} if self.removed else {}),
        }

    @classmethod
    def from_dict(cls, value):
        if not isinstance(value, dict) or value.get('model') != MODEL:
            raise ValueError('VESSEL_TUBES_REQUIRE_' + MODEL.upper().replace('-', '_'))
        return cls(
            checked_affine(value['affine']), tuple(value['shape']), np.asarray(value['points'], float).reshape(-1, 3),
            tuple(np.asarray(e, np.int64) for e in value['edges']), tuple(np.asarray(r, float) for r in value['edgeRadii']),
            tuple(value['edgeClass']), tuple(value['edgeLabel']), tuple(tuple(c) for c in value['edgeCells']),
            np.asarray(value['junctions'], np.int64), tuple(np.asarray(e, np.int64) for e in value['junctionEdges']),
            np.asarray(value['junctionRadius'], float), np.asarray(value['junctionBlend'], float),
            np.asarray(value['junctionReach'], float), value['receipt'],
            tuple(np.asarray(c, float).reshape(len(e), -1) for c, e in zip(value['edgeSections'], value['edges'])),
            tuple(int(c) for c in value.get('removedCells', ())), tuple(int(c) for c in value.get('removedOpenFaces', ())),
        )


# ---------------------------------------------------------------- extraction


def _frame(mask, affine):
    mask = np.asarray(mask)
    if mask.ndim != 3 or mask.dtype != np.bool_:
        raise ValueError('Vessel tubes require a three-dimensional binary mask')
    frame = checked_affine(affine)
    spacing = np.linalg.norm(frame[:3, :3], axis=0)
    axes = frame[:3, :3] / spacing
    if not np.allclose(axes.T @ axes, np.eye(3), atol=1e-5):
        raise ValueError('Vessel tubes require an orthogonal source grid')
    return mask, frame, spacing


class _Signal:
    """Trilinear soft signal on a crop, sampled at RAS points."""

    def __init__(self, soft, frame, lo, hi):
        self.values = np.asarray(soft[tuple(slice(a, b) for a, b in zip(lo, hi))], np.float32)
        self.inverse = np.linalg.inv(frame)
        self.lo = np.asarray(lo)

    def __call__(self, points):
        ijk = points.reshape(-1, 3) @ self.inverse[:3, :3].T + self.inverse[:3, 3] - self.lo
        return ndi.map_coordinates(self.values, ijk.T, order=1, mode='constant', cval=0.).reshape(points.shape[:-1])


def _basis(tangent):
    helper = np.where((np.abs(tangent[:, :1]) < .9), [[1., 0., 0.]], [[0., 1., 0.]])
    u = np.cross(tangent, helper)
    u /= np.linalg.norm(u, axis=1)[:, None]
    return u, np.cross(tangent, u)


def _crossings(signal, origins, directions, reach):
    """Distance along each ray to the first fall below LEVEL; nan without one."""
    n = origins.shape[0]
    shape = directions.shape[:-1]
    result = np.full(shape, np.nan)
    steps = np.linspace(0., 1., RAY_STEPS)
    for start in range(0, n, max(1, 400_000 // (np.prod(shape[1:], dtype=int) * RAY_STEPS))):
        rows = slice(start, start + max(1, 400_000 // (np.prod(shape[1:], dtype=int) * RAY_STEPS)))
        o, d, r = origins[rows], directions[rows], reach[rows]
        extra = (1,) * (d.ndim - 2)
        s = steps * r.reshape(-1, *extra, 1)
        values = signal(o.reshape(-1, *extra, 1, 3) + s[..., None] * d[..., None, :])
        s = np.broadcast_to(s, values.shape)
        below = values < LEVEL
        found = below.any(axis=-1) & ~below[..., 0]
        k = np.argmax(below, axis=-1)
        index = np.clip(k, 1, RAY_STEPS - 1)
        low = np.take_along_axis(s, index[..., None] - 1, axis=-1)[..., 0]
        high = np.take_along_axis(s, index[..., None], axis=-1)[..., 0]
        anchor = np.broadcast_to(o.reshape(-1, *extra, 3), d.shape)
        for _ in range(BISECTIONS):
            middle = (low + high) / 2
            inside = signal(anchor + middle[..., None] * d) >= LEVEL
            low, high = np.where(inside, middle, low), np.where(inside, high, middle)
        result[rows] = np.where(found, (low + high) / 2, np.nan)
    return result


def _sections(signal, points, tangents, reach):
    u, v = _basis(tangents)
    angles = np.arange(RAYS) * 2 * np.pi / RAYS
    directions = np.cos(angles)[None, :, None] * u[:, None] + np.sin(angles)[None, :, None] * v[:, None]
    return _crossings(signal, points, directions, reach), directions


def _fibonacci(count):
    k = np.arange(count) + .5
    z = 1 - 2 * k / count
    angle = np.pi * (1 + 5 ** .5) * k
    r = np.sqrt(1 - z * z)
    return np.column_stack((r * np.cos(angle), r * np.sin(angle), z))


def _tangents(points, edges, count):
    tangent = np.zeros((count, 3))
    for samples in edges:
        if len(samples) < 2:
            continue
        p = points[samples]
        t = np.gradient(p, axis=0)
        if samples[0] == samples[-1] and len(samples) > 2:
            t[0] = t[-1] = p[1] - p[-2]
        tangent[samples] += t / np.maximum(np.linalg.norm(t, axis=1), 1e-12)[:, None]
    tangent[np.linalg.norm(tangent, axis=1) < 1e-9] = (0., 0., 1.)
    return tangent / np.linalg.norm(tangent, axis=1)[:, None]


def _smoothing_operator(edges, weights, count):
    pairs = np.array([(a, b) for samples in edges for a, b in zip(samples[:-1], samples[1:]) if a != b], np.int64)
    if not len(pairs):
        return None, np.zeros(count)
    i, j = pairs.T
    w = sparse.coo_matrix((np.r_[weights[j], weights[i]], (np.r_[i, j], np.r_[j, i])), shape=(count, count)).tocsr()
    return w, np.asarray(w.sum(axis=1)).ravel()


def _taubin(points, operator, degree, pinned, rounds):
    if operator is None:
        return points
    free = ((degree > 0) & ~pinned)[:, None]
    safe = np.maximum(degree, 1e-30)[:, None]
    for _ in range(rounds):
        for factor in (.5, -.53):
            points = points + factor * free * (operator @ points / safe - points)
    return points


def _translation(offset):
    shift = np.eye(4)
    shift[:3, 3] = offset
    return shift


def _surface_fit(points, edges, edge_radii, wall, voxel, floor):
    """Radii as the median axis distance of the source wall points each tube is nearest to."""
    a, b, ra, rb, key = [], [], [], [], []
    offsets = np.cumsum([0] + [len(s) for s in edges])
    for e, (samples, radii) in enumerate(zip(edges, edge_radii)):
        if len(samples) < 2:
            continue
        a.append(points[samples[:-1]])
        b.append(points[samples[1:]])
        ra.append(radii[:-1])
        rb.append(radii[1:])
        key.append(offsets[e] + np.arange(len(samples) - 1))
    if not a or wall is None or not len(wall):
        return edge_radii
    a, b, ra, rb, key = map(np.concatenate, (a, b, ra, rb, key))
    tree = cKDTree((a + b) / 2)
    k = min(8, len(a))
    rows, distances = [], []
    for start in range(0, len(wall), 100_000):
        p = wall[start:start + 100_000]
        _, near = tree.query(p, k=k)
        near = near.reshape(len(p), k)
        ab = b[near] - a[near]
        t = np.clip(np.einsum('nki,nki->nk', p[:, None] - a[near], ab) / np.maximum(np.einsum('nki,nki->nk', ab, ab), 1e-30), 0, 1)
        axis = np.linalg.norm(p[:, None] - (a[near] + t[..., None] * ab), axis=2)
        residual = np.abs(axis - ((1 - t) * ra[near] + t * rb[near]))
        best = residual.argmin(axis=1)
        n = np.arange(len(p))
        accept = residual[n, best] < FIT_ACCEPT * voxel
        sample = key[near[n, best]] + (t[n, best] >= .5)
        rows.append(sample[accept])
        distances.append(axis[n, best][accept])
    rows, distances = np.concatenate(rows), np.concatenate(distances)
    order = np.lexsort((distances, rows))
    rows, distances = rows[order], distances[order]
    starts = np.flatnonzero(np.r_[True, rows[1:] != rows[:-1]])
    counts = np.diff(np.r_[starts, len(rows)])
    fitted = np.full(offsets[-1], np.nan)
    enough = counts >= FIT_MIN_POINTS
    first, size = starts[enough], counts[enough]
    fitted[rows[first]] = (distances[first + (size - 1) // 2] + distances[first + size // 2]) / 2
    result = []
    for e, radii in enumerate(edge_radii):
        values = fitted[offsets[e]:offsets[e + 1]]
        values = np.where(np.isfinite(values), values, radii)
        if len(values) >= 3:
            padded = np.r_[values[0], values, values[-1]]
            values = (padded[:-2] + padded[1:-1] + padded[2:]) / 3
        result.append(np.maximum(values, floor))
    return result


def _near_ends(points, edges, edge_junctions, junction_samples, junction_clear, tips, annotation):
    """Samples whose normal sections cross another branch (junctions) or an end cap (tips)."""
    near = np.zeros(len(points), bool)
    tip_set = set(tips.tolist())
    for e, samples in enumerate(edges):
        ends = [(points[junction_samples[j]], junction_clear[j]) for j in edge_junctions[e]]
        ends += [(points[t], np.nan_to_num(annotation[t], nan=0.)) for t in {int(samples[0]), int(samples[-1])} & tip_set]
        for centre, clear in ends:
            near[samples[np.linalg.norm(points[samples] - centre, axis=1) < clear]] = True
    return near


def _merge(paths):
    """Join open paths through cells where exactly two path ends meet."""
    paths = [list(p) for p in paths]
    degree, ends = {}, {}
    for i, path in enumerate(paths):
        if len(path) > 1:
            for cell in (path[0], path[-1]):
                degree[cell] = degree.get(cell, 0) + 1
            if path[0] != path[-1]:
                ends.setdefault(path[0], set()).add(i)
                ends.setdefault(path[-1], set()).add(i)
    for cell in [c for c, d in degree.items() if d == 2]:
        ids = ends.get(cell, set())
        if len(ids) != 2:
            continue
        i, j = sorted(ids)
        first = paths[i] if paths[i][-1] == cell else paths[i][::-1]
        second = paths[j] if paths[j][0] == cell else paths[j][::-1]
        for k in (i, j):
            for end in (paths[k][0], paths[k][-1]):
                ends[end].discard(k)
        paths[i], paths[j] = first + second[1:], None
        if paths[i][0] != paths[i][-1]:
            ends[paths[i][0]].add(i)
            ends[paths[i][-1]].add(i)
    return [p for p in paths if p is not None]


def extract(mask, affine, *, probability=None, classes=None, named=None, scheme=None, source_sha256=None,
            check=lambda: None):
    """Centerline tube model of a binary vessel mask on its source grid.

    ``probability`` (same grid, 0 to 1) is the soft signal for radii; without it
    the mask itself is used, so the reference boundary is the mask's half-level.
    ``classes`` (1 artery, 2 vein, 3 unclassified) or ``named`` labels with a
    ``vascular_rooting.LabelScheme`` give each edge its class and named label.
    """
    started = time.perf_counter()
    mask, frame, spacing = _frame(mask, affine)
    shape = mask.shape
    if probability is not None and np.shape(probability) != shape:
        raise ValueError('Vessel tube probability must match the mask grid')
    if not mask.any():
        raise ValueError('Vessel tubes require a non-empty mask')
    graph = build_source_branches(mask, frame, source_scan_sha256=source_sha256 or _mask_digest(mask), check=check)
    branches = graph['branches']
    radius_of = {}
    for branch in branches:
        for cell, radius in zip(branch['source_cell_ids'], branch['annotation_radii_mm']):
            radius_of[cell] = np.nan if radius is None else radius
    voxel = float(spacing.min())
    floor = .5 * voxel

    # Prune spurs whose tip lies inside the inscribed ball at their junction.
    paths = [b['source_cell_ids'] for b in branches]
    ends = {}
    for path in paths:
        if len(path) > 1 and path[0] != path[-1]:
            for cell in (path[0], path[-1]):
                ends[cell] = ends.get(cell, 0) + 1
    ijk_of = lambda cells: np.column_stack(np.unravel_index(np.asarray(cells, np.int64), shape))
    kept, pruned = [], 0
    for path in paths:
        if len(path) > 1 and path[0] != path[-1]:
            degrees = ends[path[0]], ends[path[-1]]
            if 1 in degrees and max(degrees) >= 3:
                root = path[0] if degrees[0] >= 3 else path[-1]
                tip = path[-1] if root == path[0] else path[0]
                length = np.linalg.norm((ijk_of([tip])[0] - ijk_of([root])[0]) * spacing)
                if length <= radius_of[root] + .5 * voxel:
                    pruned += 1
                    continue
        kept.append(path)
    paths = _merge(kept)
    # Components the thinning removed entirely become one ball each.
    labels = ndi.label(mask, structure=np.ones((3, 3, 3), bool))[0]
    represented = set(labels.ravel()[[p[0] for p in paths]].tolist())
    missing = [c['source_component_anchor_cell'] for c in graph['components'] if not c.get('branches')]
    for anchor in missing:
        component = labels.ravel()[anchor]
        if component in represented:
            continue
        found = np.flatnonzero(labels.ravel() == component)
        ijk = np.column_stack(np.unravel_index(found, shape))
        paths.append([int(found[np.argmin(np.linalg.norm((ijk - ijk.mean(axis=0)) * spacing, axis=1))])])
        radius_of[paths[-1][0]] = np.nan
    check()

    cells = np.unique(np.fromiter(chain.from_iterable(paths), np.int64))
    index = {int(c): i for i, c in enumerate(cells)}
    ijk = ijk_of(cells)
    points = ijk @ frame[:3, :3].T + frame[:3, 3]
    annotation = np.array([radius_of[int(c)] for c in cells], float)
    edges = [np.array([index[c] for c in path], np.int64) for path in paths]
    count = len(points)
    end_degree = np.zeros(count, np.int64)
    for samples in edges:
        if len(samples) > 1:
            np.add.at(end_degree, samples[[0, -1]], 1)
    junction_samples = np.flatnonzero(end_degree >= 3)
    tips = np.flatnonzero(end_degree == 1)

    lo = np.maximum(ijk.min(axis=0) - np.ceil((np.nanmax(annotation, initial=voxel) * 3 + 4 * spacing.max()) / spacing), 0).astype(int)
    hi = np.minimum(ijk.max(axis=0) + np.ceil((np.nanmax(annotation, initial=voxel) * 3 + 4 * spacing.max()) / spacing) + 1, shape).astype(int)
    soft = mask if probability is None else probability
    signal = _Signal(soft, frame, lo, hi)
    wall, wall_faces = source_surface(signal.values, frame @ _translation(lo), level=LEVEL)
    reach = np.clip(np.nan_to_num(annotation, nan=voxel) * 2.5 + 2 * spacing.max(), 2 * spacing.max(), 20.)

    # Samples within the inscribed ball of a junction see the other branches.
    junction_clear = np.nan_to_num(annotation[junction_samples], nan=voxel)
    edge_junctions = [[] for _ in edges]
    for e, samples in enumerate(edges):
        for end in {int(samples[0]), int(samples[-1])}:
            where = np.searchsorted(junction_samples, end)
            if where < len(junction_samples) and junction_samples[where] == end:
                edge_junctions[e].append(where)
    weights = np.nan_to_num(annotation, nan=voxel) ** 2
    operator, degree = _smoothing_operator(edges, weights, count)
    pinned = np.zeros(count, bool)
    pinned[tips] = True
    points = _taubin(points, operator, degree, pinned, SMOOTHING_ROUNDS)
    for _ in range(RECENTRE_ROUNDS):
        check()
        near_junction = _near_ends(points, edges, edge_junctions, junction_samples, junction_clear, tips, annotation)
        tangents = _tangents(points, edges, count)
        radii, directions = _sections(signal, points, tangents, reach)
        opposite = np.roll(radii, -RAYS // 2, axis=1)[:, :RAYS // 2]
        half = radii[:, :RAYS // 2]
        pair = np.isfinite(half) & np.isfinite(opposite)
        u = directions[:, :RAYS // 2]
        a = np.einsum('np,npi,npj->nij', pair.astype(float), u, u)
        rhs = np.einsum('np,npi->ni', np.where(pair, (half - opposite) / 2, 0.), u)
        movable = (pair.sum(axis=1) >= 3) & (~near_junction | (end_degree == 1)) & (end_degree <= 1)
        shift = np.zeros_like(points)
        shift[movable] = np.linalg.solve(a[movable] + 1e-9 * np.eye(3), rhs[movable][..., None])[..., 0]
        length = np.linalg.norm(shift, axis=1)
        shift *= (np.minimum(length, .5 * voxel) / np.maximum(length, 1e-12))[:, None]
        points = _taubin(points + shift, operator, degree, pinned, SMOOTHING_ROUNDS // 2)

    check()
    near_junction = _near_ends(points, edges, edge_junctions, junction_samples, junction_clear, tips, annotation)
    tangents = _tangents(points, edges, count)
    rays, _ = _sections(signal, points, tangents, reach)
    measured = np.full(count, np.nan)
    enough = np.isfinite(rays).sum(axis=1) >= RAYS // 2
    measured[enough] = np.nanmedian(rays[enough], axis=1)
    unmasked = measured.copy()
    measured[near_junction] = np.nan
    # Near junctions and tips a section crosses other walls; the inscribed ball does not.
    if near_junction.any():
        rows = np.flatnonzero(near_junction)
        inside = signal(points[rows]) >= LEVEL
        measured[rows[inside]] = point_mesh_distance(points[rows[inside]], wall, wall_faces)[0]
    fallback = np.maximum(np.nan_to_num(annotation, nan=voxel) - .5 * spacing.max(), floor)

    edge_radii, short = [], 0
    for samples in edges:
        values = measured[samples].copy()
        good = np.flatnonzero(np.isfinite(values))
        if len(good):
            nearest = good[np.abs(np.arange(len(values))[:, None] - good[None]).argmin(axis=1)]
            values = values[nearest]
            if len(values) >= 3:
                padded = np.r_[values[0], values, values[-1]]
                values = (padded[:-2] + padded[1:-1] + padded[2:]) / 3
        else:
            # Every sample sees another branch: the narrowest section is the least affected.
            inner = unmasked[samples[1:-1]] if len(samples) > 2 else unmasked[samples]
            values = np.full(len(samples), np.nanmin(inner) if np.isfinite(inner).any() else np.nan)
            values = np.where(np.isfinite(values), values, fallback[samples])
            short += 1
        edge_radii.append(np.maximum(values, floor))

    # Tips: extend along the axis to the measured end of the lumen.
    extension = []
    for e, samples in enumerate(edges):
        for side in (0, -1):
            tip = samples[side]
            if end_degree[tip] != 1 or len(samples) < 2:
                continue
            inner = samples[min(3, len(samples) - 1)] if side == 0 else samples[max(-4, -len(samples))]
            direction = points[tip] - points[inner]
            direction /= max(np.linalg.norm(direction), 1e-12)
            extent = _crossings(signal, points[tip][None], direction[None, None], reach[[tip]] * 2)[0, 0]
            r = edge_radii[e][side]
            if not np.isfinite(extent):
                continue
            if extent - r > .1 * voxel:
                extension.append((e, side, points[tip] + (extent - r) * direction, r))
            elif extent - r < -.1 * voxel:
                # Pull a protruding end ball back, never past its neighbour.
                previous = points[samples[1] if side == 0 else samples[-2]]
                room = max(float((points[tip] - previous) @ direction), 0.)
                points[tip] -= min(r - extent, room) * direction
    for e, side, point, r in extension:
        new = len(points)
        points = np.vstack((points, point))
        if side == 0:
            edges[e] = np.r_[new, edges[e]]
            edge_radii[e] = np.r_[r, edge_radii[e]]
        else:
            edges[e] = np.r_[edges[e], new]
            edge_radii[e] = np.r_[edge_radii[e], r]

    for _ in range(SURFACE_FIT_ROUNDS):
        check()
        edge_radii = _surface_fit(points, edges, edge_radii, wall, voxel, floor)

    # Isolated samples (one-voxel skeletons) are balls.
    for e, samples in enumerate(edges):
        if len(samples) == 1:
            tangent = np.array([[0., 0., 1.]])
            r, _ = _sections(signal, points[samples], tangent, reach[samples])
            value = np.nanmedian(r) if np.isfinite(r).any() else fallback[samples[0]]
            edge_radii[e] = np.array([max(float(value), floor)])

    incident_of = [set() for _ in junction_samples]
    for e, found in enumerate(edge_junctions):
        for j in found:
            incident_of[j].add(e)
    # Node balls: the largest ball about the node inside the half level.
    sphere = _fibonacci(64)
    balls = _crossings(signal, points[junction_samples], np.broadcast_to(sphere, (len(junction_samples), 64, 3)),
                       reach[junction_samples]) if len(junction_samples) else np.zeros((0, 64))
    balls = np.where(np.isfinite(balls), balls, np.inf).min(axis=1) if len(balls) else np.zeros(0)
    junction_edges, junction_radius, junction_blend, junction_reach = [], [], [], []
    for j, sample in enumerate(junction_samples):
        incident = sorted(incident_of[j])
        end_radii = [edge_radii[e][0 if edges[e][0] == sample else -1] for e in incident]
        ball = float(balls[j]) if np.isfinite(balls[j]) else float(fallback[sample])
        ball = max(ball, floor)
        junction_edges.append(np.asarray(incident, np.int64))
        junction_radius.append(ball)
        junction_blend.append(BLEND_FRACTION * min(end_radii))
        junction_reach.append(BLEND_REACH * max(max(end_radii), ball))

    edge_class, edge_label = [], []
    for path in paths:
        values = np.asarray(path, np.int64)
        label = 0
        if named is not None:
            found = np.asarray(named).ravel()[values]
            found = found[found > 0]
            label = int(np.bincount(found).argmax()) if len(found) else 0
        if classes is not None:
            found = np.asarray(classes).ravel()[values]
            found = found[found > 0]
            kind = CLASS_NAMES.get(int(np.bincount(found).argmax()) if len(found) else 3, 'unclassified')
        elif label and scheme is not None:
            kind = CLASS_NAMES[scheme.label_class(label)]
        else:
            kind = 'unclassified'
        edge_class.append(kind)
        edge_label.append(label)

    links = np.unique(np.sort(np.array([(a, b) for e in edges for a, b in zip(e[:-1], e[1:]) if a != b],
                                       np.int64).reshape(-1, 2), axis=1), axis=0)
    graph_components = connected_components(sparse.coo_matrix(
        (np.ones(len(links)), tuple(links.T)), shape=(len(points),) * 2), directed=False)[0]
    receipt = {
        'model': MODEL, 'level': LEVEL,
        'graphBetti': [int(graph_components), int(len(links) - len(points) + graph_components)], 'softSignal': 'probability' if probability is not None else 'mask',
        'skeleton': graph['algorithm'], 'skeletonCoverage': graph['coverage'],
        'prunedSpurs': pruned, 'ballsForEmptySkeletons': len(missing), 'edges': len(edges), 'junctions': len(junction_samples),
        'samples': len(points), 'tipsExtended': len(extension), 'edgesWithoutClearSection': short,
        'blendFraction': BLEND_FRACTION, 'blendReach': BLEND_REACH,
        'radiusFloorMm': floor, 'seconds': round(time.perf_counter() - started, 3),
    }
    model = TubeModel(frame, tuple(shape), points, tuple(edges), tuple(edge_radii), tuple(edge_class),
                      tuple(edge_label), tuple(tuple(map(int, p)) for p in paths), junction_samples.astype(np.int64),
                      tuple(junction_edges), np.asarray(junction_radius, float), np.asarray(junction_blend, float),
                      np.asarray(junction_reach, float), receipt)
    model = _fit_sections(model, wall, voxel, floor, check)
    # Tubes of separate 26-connected source components keep a gap the mesh grid resolves.
    labels = labels.ravel()
    owner_component = np.r_[[labels[p[0]] for p in paths], labels[cells[junction_samples]]]
    return _separate(model, owner_component, gap=float(spacing.min()) / MESH_SUBDIVISION, floor=floor)


def _limit_sections(coefficients, radii):
    """Scale sections so the largest offset stays within ``SECTION_LIMIT`` of the radius."""
    swell = _section_swell(coefficients)
    return coefficients * np.minimum(1., SECTION_LIMIT * radii / np.maximum(swell, 1e-30))[:, None]


def _fit_sections(model, wall, voxel, floor, check):
    """Radii and harmonic sections by regularised least squares to the source wall points each segment owns."""
    if wall is None or not len(wall):
        return model
    edges = model.edges
    sizes = np.array([len(e) for e in edges])
    offsets = np.r_[0, np.cumsum(sizes)]
    total, order = int(offsets[-1]), SECTION_ORDER
    columns = 2 * order + 2
    width = columns + 1
    degree = np.zeros(len(model.points), np.int64)
    for samples in edges:
        if len(samples) > 1:
            np.add.at(degree, samples[[0, -1]], 1)
    tip = np.zeros(total, bool)
    for e, samples in enumerate(edges):
        if len(samples) > 1:
            tip[offsets[e]] |= degree[samples[0]] == 1
            tip[offsets[e + 1] - 1] |= degree[samples[-1]] == 1
    # Ridge (higher orders cost more; end-cap terms only at tips) and smoothness along each edge.
    ridge = np.r_[SECTION_RIDGE, SECTION_RIDGE * np.repeat(np.arange(1, order + 1), 2), SECTION_RIDGE, SECTION_RIDGE]
    ridge = np.tile(ridge, (total, 1))
    ridge[~tip, -2:] = 1e3
    pairs = np.concatenate([offsets[e] + np.c_[np.arange(n - 1), np.arange(1, n)] for e, n in enumerate(sizes) if n > 1]
                           or [np.zeros((0, 2), np.int64)])
    rows = np.arange(len(pairs) * width)
    first = (pairs[:, :1] * width + np.arange(width)).ravel()
    second = (pairs[:, 1:] * width + np.arange(width)).ravel()
    smooth = sparse.csr_matrix((np.r_[np.full(len(rows), SECTION_SMOOTH), np.full(len(rows), -SECTION_SMOOTH)],
                                (np.r_[rows, rows], np.r_[first, second])), shape=(len(rows), total * width))
    regular = (smooth.T @ smooth + sparse.diags(ridge.ravel() ** 2)).tocsr()
    split = lambda values: tuple(np.split(values, offsets[1:-1]))
    radii = np.concatenate(model.edge_radii).astype(float)
    sections = np.zeros((total, columns))
    used = 0
    for _ in range(SECTION_ROUNDS):
        check()
        current = replace(model, edge_radii=split(radii), edge_sections=split(sections))
        seg = current.segments
        _, s = current.nearest_segments(wall, reach=SECTION_REACH * voxel, edges_only=True, raw=True)
        keep = s >= 0
        keep[keep] = seg['ga'][s[keep], 0] != seg['ga'][s[keep], 1]
        p, s = wall[keep], s[keep]
        used = len(p)
        if not used:
            break
        d = _round_cone(p, seg['a'][s], seg['b'][s], seg['ra'][s], seg['rb'][s])
        t, end_a, end_b = current._terms(p, s)
        ones = np.ones((len(p), 1))
        values = np.c_[np.c_[ones, _section_basis(*end_a, order)] * (1 - t)[:, None],
                       np.c_[ones, _section_basis(*end_b, order)] * t[:, None]]
        cols = np.c_[seg['ga'][s, :1] * width + np.arange(width), seg['ga'][s, 1:] * width + np.arange(width)]
        a = sparse.csr_matrix((values.ravel(), (np.repeat(np.arange(len(p)), 2 * width), cols.ravel())),
                              shape=(len(p), total * width))
        solution = spsolve((a.T @ a + regular).tocsc(), a.T @ d).reshape(total, width)
        radii = np.maximum(radii + solution[:, 0], floor)
        sections = _limit_sections(solution[:, 1:], radii)
    receipt = {**model.receipt, 'sectionOrder': order, 'sectionRounds': SECTION_ROUNDS, 'sectionWallPoints': used}
    return replace(model, edge_radii=split(radii), edge_sections=split(sections), receipt=receipt)


def _closest_between(a0, a1, b0, b1):
    """Parameters of the closest points between segment pairs (Ericson 5.1.9)."""
    d1, d2, r = a1 - a0, b1 - b0, a0 - b0
    dot = lambda x, y: np.einsum('ij,ij->i', x, y)
    a, e, f, c, b = dot(d1, d1), dot(d2, d2), dot(d2, r), dot(d1, r), dot(d1, d2)
    denom = a * e - b * b
    s = np.where(denom > 1e-12, np.clip((b * f - c * e) / np.where(denom > 1e-12, denom, 1.), 0, 1), 0.)
    t = (b * s + f) / np.where(e > 1e-12, e, 1.)
    s = np.where(t < 0, np.clip(-c / np.where(a > 1e-12, a, 1.), 0, 1), np.where(t > 1, np.clip((b - c) / np.where(a > 1e-12, a, 1.), 0, 1), s))
    t = np.clip(t, 0, 1)
    s = np.where(a > 1e-12, s, 0.)
    t = np.where(e > 1e-12, t, 0.)
    return s, t


def _separate(model, owner_component, *, gap, floor, rounds=3):
    """Shrink radii where tubes of different source components come closer than ``gap``."""
    edge_radii = [r.copy() for r in model.edge_radii]
    balls = model.junction_radius.copy()
    cut = np.cumsum([len(e) for e in model.edges])[:-1]
    sections = lambda: tuple(np.split(_limit_sections(model.sections, np.concatenate(edge_radii)), cut))
    shrunk = 0
    for _ in range(rounds):
        current = replace(model, edge_radii=tuple(edge_radii), junction_radius=balls, edge_sections=sections())
        seg = current.segments
        if seg['tree'] is None:
            break
        component = owner_component[seg['owner']]
        pairs = seg['tree'].query_pairs(2 * seg['bound'].max() + gap, output_type='ndarray')
        pairs = pairs[component[pairs[:, 0]] != component[pairs[:, 1]]]
        if not len(pairs):
            break
        i, j = pairs.T
        si, tj = _closest_between(seg['a'][i], seg['b'][i], seg['a'][j], seg['b'][j])
        pi = seg['a'][i] + si[:, None] * (seg['b'][i] - seg['a'][i])
        pj = seg['a'][j] + tj[:, None] * (seg['b'][j] - seg['a'][j])
        ri = (1 - si) * seg['ra'][i] + si * seg['rb'][i] + seg['swell'][i]
        rj = (1 - tj) * seg['ra'][j] + tj * seg['rb'][j] + seg['swell'][j]
        overlap = gap - (np.linalg.norm(pi - pj, axis=1) - ri - rj)
        hit = overlap > 0
        if not hit.any():
            break
        need = {}
        for k, amount in zip(np.r_[i[hit], j[hit]], np.r_[overlap[hit], overlap[hit]] / 2):
            need[k] = max(need.get(k, 0.), amount)
        offsets = np.cumsum([0] + [max(len(e) - 1, 1) for e in model.edges])
        for k, amount in need.items():
            owner = seg['owner'][k]
            if owner >= len(model.edges):
                balls[owner - len(model.edges)] = max(balls[owner - len(model.edges)] - amount, floor)
            else:
                local = k - offsets[owner]
                rows = [local, min(local + 1, len(edge_radii[owner]) - 1)]
                edge_radii[owner][rows] = np.maximum(edge_radii[owner][rows] - amount, floor)
            shrunk += 1
    receipt = {**model.receipt, 'separatedSegments': shrunk}
    return replace(model, edge_radii=tuple(edge_radii), junction_radius=balls, edge_sections=sections(), receipt=receipt)


# --------------------------------------------------------------- grid field


class _Grid:
    """Points ``origin + g * step`` in source IJK, g integer, clamped at ``fill``."""

    def __init__(self, model, step, fill, *, origin=None, shape=None):
        self.model = model
        self.frame = model.affine
        self.step = np.asarray(step, float)
        self.fill = float(fill)
        seg = model.segments
        inverse = np.linalg.inv(self.frame)
        a = seg['a'] @ inverse[:3, :3].T + inverse[:3, 3]
        b = seg['b'] @ inverse[:3, :3].T + inverse[:3, 3]
        pad = (np.maximum(seg['ra'], seg['rb']) + seg['swell'] + self.fill)[:, None] / model.spacing
        low, high = np.minimum(a, b) - pad, np.maximum(a, b) + pad
        if origin is None:
            origin = low.min(axis=0) - 2 * self.step
            shape = np.ceil((high.max(axis=0) + 2 * self.step - origin) / self.step).astype(int) + 1
        self.origin, self.shape = np.asarray(origin, float), np.asarray(shape, int)
        self.seg_lo = np.maximum(np.ceil((low - self.origin) / self.step), 0).astype(np.int64)
        self.seg_hi = np.minimum(np.floor((high - self.origin) / self.step), self.shape - 1).astype(np.int64)
        centres = model.points[model.junctions].reshape(-1, 3) @ inverse[:3, :3].T + inverse[:3, 3]
        jpad = (2 * model.junction_reach)[:, None] / model.spacing
        self.j_lo = np.maximum(np.ceil((centres - jpad - self.origin) / self.step), 0).astype(np.int64)
        self.j_hi = np.minimum(np.floor((centres + jpad - self.origin) / self.step), self.shape - 1).astype(np.int64)

    def ras(self, g):
        ijk = self.origin + g * self.step
        return ijk @ self.frame[:3, :3].T + self.frame[:3, 3]

    def blocks(self, size, halo=0):
        """Block corners whose inclusive point range [c - halo, c + size + halo] meets a segment box."""
        found = set()
        for lo, hi in zip(self.seg_lo, self.seg_hi):
            ranges = [range(max(0, -(-(l - size - halo) // size)), min((h + halo) // size, (n - 1) // size) + 1)
                      for l, h, n in zip(lo, hi, self.shape)]
            found.update(product(*ranges))
        return sorted(found)

    def values(self, lo, hi):
        """Field on points lo..hi inclusive (grid indices; may extend past the grid)."""
        model, seg = self.model, self.model.segments
        size = hi - lo + 1
        out = np.full(size, self.fill)
        hit = np.flatnonzero(np.all((self.seg_lo <= hi) & (self.seg_hi >= lo), axis=1))
        if len(hit):
            extent_all = np.minimum(self.seg_hi[hit], hi) - np.maximum(self.seg_lo[hit], lo) + 1
            batch = np.cumsum(np.prod(extent_all, axis=1)) // 400_000
            for group in np.split(hit, np.flatnonzero(np.diff(batch)) + 1):
                a = np.maximum(self.seg_lo[group], lo)
                extent = np.minimum(self.seg_hi[group], hi) - a + 1
                counts = np.prod(extent, axis=1)
                which = np.repeat(np.arange(len(group)), counts)
                local = np.arange(counts.sum()) - np.repeat(np.cumsum(counts) - counts, counts)
                plane = extent[which, 1] * extent[which, 2]
                g = a[which] + np.column_stack((local // plane, (local % plane) // extent[which, 2], local % extent[which, 2]))
                flat = np.ravel_multi_index((g - lo).T, size)
                d = model._segment_field(self.ras(g), group[which], index=flat, size=out.size, cap=out.reshape(-1), raw=True)
                np.minimum.at(out.reshape(-1), flat, d)
        near = np.flatnonzero(np.all((self.j_lo <= hi) & (self.j_hi >= lo), axis=1) & (model.junction_blend > 0))
        if len(near):
            base = out.copy()
            for j in near:
                a, b = np.maximum(self.j_lo[j], lo), np.minimum(self.j_hi[j], hi)
                box = tuple(slice(x - l, y - l + 1) for x, y, l in zip(a, b, lo))
                inside = np.argwhere(base[box] < self.fill) + a
                if not len(inside):
                    continue
                for start in range(0, len(inside), 50_000):
                    g = inside[start:start + 50_000]
                    p = self.ras(g)
                    term = model._junction_term(j, p, model._member_distances(j, p, raw=True))
                    flat = np.ravel_multi_index((g - lo).T, size)
                    out.reshape(-1)[flat] = np.minimum(out.reshape(-1)[flat], term)
        g = np.stack(np.meshgrid(*[np.arange(a, b + 1) for a, b in zip(lo, hi)], indexing='ij'), -1).reshape(-1, 3)
        out = np.maximum(out, model.limits(self.ras(g)).reshape(out.shape))
        return np.minimum(out, self.fill)


# ------------------------------------------------------------------ meshing


@dataclass(frozen=True)
class TubeMesh:
    positions_ijk: np.ndarray
    positions: np.ndarray
    faces: np.ndarray
    normals: np.ndarray
    owners: np.ndarray
    receipt: dict


def mesh(model, *, subdivision=MESH_SUBDIVISION, block=BLOCK, check=lambda: None):
    """Watertight wall of the implicit tubes; grid step is smallest spacing / ``subdivision``."""
    from skimage.measure import marching_cubes

    started = time.perf_counter()
    h = float(model.spacing.min()) / subdivision
    step = h / model.spacing
    fill = 2 * h + float(model.junction_blend.max(initial=0.)) / 4
    grid = _Grid(model, step, fill)
    keys, positions, gradients, faces = [], [], [], []
    count = 0
    interior = 0
    nudge = 1e-3 * h
    for corner in grid.blocks(block, halo=1):
        check()
        lo = np.asarray(corner) * block
        hi = np.minimum(lo + block, grid.shape - 1)
        values = grid.values(lo - 1, hi + 1)
        values = np.where(np.abs(values) < nudge, np.where(values < 0, -nudge, nudge), values)
        inner = values[1:-1, 1:-1, 1:-1]
        if inner.min() >= 0 or inner.max() <= 0 or min(inner.shape) < 2:
            continue
        verts, tris, _, _ = marching_cubes(inner, 0., method='lewiner', allow_degenerate=True)
        verts = verts.astype(np.float64)
        gradient = np.stack(np.gradient(values), axis=-1)
        local_grad = np.stack([ndi.map_coordinates(gradient[..., i], (verts + 1).T, order=1) for i in range(3)], axis=1)
        integral = np.abs(verts - np.rint(verts)) < 1e-4
        on_edge = integral.sum(axis=1) == 2
        axis = np.argmin(integral, axis=1)
        base = np.where(integral, np.rint(verts), np.floor(verts)).astype(np.int64) + lo
        key = np.ravel_multi_index(base.T, grid.shape + 1, mode='clip') * 3 + axis
        key = np.where(on_edge, key, -(interior + 1 + np.arange(len(verts))))
        interior += len(verts)
        keys.append(key)
        positions.append(verts + lo)
        gradients.append(local_grad)
        faces.append(tris + count)
        count += len(verts)
    if not faces:
        empty = np.zeros((0, 3))
        return TubeMesh(empty, empty.copy(), np.zeros((0, 3), np.int64), empty.copy(), np.zeros(0, np.int64), {})
    keys = np.concatenate(keys)
    unique, inverse = np.unique(keys, return_inverse=True)
    grid_points = np.zeros((len(unique), 3))
    np.add.at(grid_points, inverse, np.concatenate(positions))
    grid_points /= np.bincount(inverse, minlength=len(unique))[:, None]
    gradient = np.zeros((len(unique), 3))
    np.add.at(gradient, inverse, np.concatenate(gradients))
    faces = inverse[np.concatenate(faces)].astype(np.int64)
    ijk = grid.origin + grid_points * step
    ras = ijk @ model.affine[:3, :3].T + model.affine[:3, 3]
    axes = model.affine[:3, :3] / model.spacing
    normals = (gradient / (step * model.spacing)) @ axes.T
    normals /= np.maximum(np.linalg.norm(normals, axis=1), 1e-12)[:, None]
    if _signed_volume(ras, faces) < 0:
        faces = faces[:, ::-1].copy()
    # A smooth union can leave a sub-grid pocket beside a crotch; a wall
    # component that encloses no centerline sample is such a pocket.
    count, labels = connected_components(sparse.coo_matrix(
        (np.ones(2 * len(faces)), (faces[:, :2].ravel(), faces[:, 1:].ravel())), shape=(len(ras),) * 2), directed=False)
    low = np.full((count, 3), np.inf)
    high = np.full((count, 3), -np.inf)
    np.minimum.at(low, labels, ras)
    np.maximum.at(high, labels, ras)
    samples = model.points
    keep = np.array([np.any(np.all((samples >= lo) & (samples <= hi), axis=1)) for lo, hi in zip(low, high)])
    islands = int(np.count_nonzero(~keep))
    if islands:
        used = keep[labels]
        faces = faces[used[faces[:, 0]]]
        index = np.cumsum(used) - 1
        ras, ijk, normals, faces = ras[used], ijk[used], normals[used], index[faces]
    _, owners = model.evaluate(ras, reach=2 * h, raw=True)
    receipt = {'model': MODEL, 'gridStepMm': h, 'subdivision': subdivision, 'block': block, 'droppedPockets': islands,
               'vertices': len(ras), 'faces': len(faces), 'seconds': round(time.perf_counter() - started, 3)}
    return TubeMesh(ijk, ras, faces, normals, owners, receipt)


def _signed_volume(points, faces):
    a, b, c = points[faces].transpose(1, 0, 2)
    return float(np.einsum('ij,ij->i', a, np.cross(b, c)).sum() / 6)


# ------------------------------------------------------------------- audits


def _segment_hits(a, b, triangles, eps=1e-9):
    """Strict segment-triangle crossings (Moller-Trumbore)."""
    v0, v1, v2 = triangles.transpose(1, 0, 2)
    direction = b - a
    e1, e2 = v1 - v0, v2 - v0
    pvec = np.cross(direction, e2)
    det = np.einsum('ij,ij->i', e1, pvec)
    scale = np.linalg.norm(e1, axis=1) * np.linalg.norm(e2, axis=1) * np.linalg.norm(direction, axis=1)
    valid = np.abs(det) > 1e-12 * np.maximum(scale, 1e-30)
    inv = 1. / np.where(valid, det, 1.)
    tvec = a - v0
    u = np.einsum('ij,ij->i', tvec, pvec) * inv
    qvec = np.cross(tvec, e1)
    v = np.einsum('ij,ij->i', direction, qvec) * inv
    t = np.einsum('ij,ij->i', e2, qvec) * inv
    return valid & (u > eps) & (v > eps) & (u + v < 1 - eps) & (t > eps) & (t < 1 - eps)


def audit(points, faces, *, self_intersections=True, model=None, check=lambda: None):
    """Closedness, manifoldness, orientation, self-intersection and topology of a triangle mesh.

    With the ``model`` the mesh came from, also the largest field value (mm, before the gradient
    normalisation, so an upper bound on the distance outside the zero level up to the taper) over the
    vertices, edge midpoints and face centroids, to compare with ``model.wall_tolerance``.
    """
    points, faces = np.asarray(points, float), np.asarray(faces, np.int64)
    result = {'vertices': len(points), 'faces': len(faces)}
    directed = faces[:, [0, 1, 1, 2, 2, 0]].reshape(-1, 2)
    undirected = np.sort(directed, axis=1)
    keys, inverse, counts = np.unique(undirected[:, 0] * len(points) + undirected[:, 1],
                                      return_inverse=True, return_counts=True)
    edges = np.column_stack(np.divmod(keys, len(points)))
    inverse = inverse.ravel()
    forward = (directed[:, 0] < directed[:, 1]).astype(np.int64)
    orientation = np.bincount(inverse, weights=2 * forward - 1, minlength=len(edges))
    result['open_edges'] = int(np.count_nonzero(counts == 1))
    result['non_manifold_edges'] = int(np.count_nonzero(counts > 2))
    result['inconsistent_edges'] = int(np.count_nonzero((counts == 2) & (orientation != 0)))
    a, b, c = points[faces].transpose(1, 0, 2)
    area = np.linalg.norm(np.cross(b - a, c - a), axis=1) / 2
    result['degenerate_faces'] = int(np.count_nonzero(area <= 1e-14))
    result['min_face_area_mm2'] = float(area.min(initial=np.inf))
    # A vertex is manifold when its incident faces form one fan: join the
    # corners of a vertex across every edge two faces share, then count fans.
    # Directed row 3 f + k is the edge from corner k of face f to corner k + 1.
    order = np.argsort(inverse, kind='stable')
    paired = np.flatnonzero(counts == 2)
    start = (np.cumsum(counts) - counts)[paired]
    first, second = order[start], order[start + 1]
    del order, start, paired
    corner = lambda row, vertex: (3 * (row // 3) + np.where(directed[row, 0] == vertex, row % 3, (row % 3 + 1) % 3)).astype(np.int32)
    links = np.concatenate([np.column_stack((corner(first, undirected[first, end]), corner(second, undirected[first, end])))
                            for end in (0, 1)])
    del first, second
    graph = sparse.csr_matrix((np.ones(len(links), np.int8), (links[:, 0], links[:, 1])), shape=(faces.size,) * 2)
    del links
    _, fan = connected_components(graph, directed=False)
    del graph
    fans = np.unique(faces.ravel().astype(np.int64) * faces.size + fan) // faces.size
    del fan
    per_vertex = np.bincount(fans, minlength=len(points))
    bad_vertices = int(np.count_nonzero(per_vertex > 1) + np.count_nonzero(per_vertex == 0))
    result['non_manifold_vertices'] = int(bad_vertices)
    del directed, undirected, inverse, fans
    result['volume_mm3'] = _signed_volume(points, faces)
    components, labels = connected_components(
        sparse.coo_matrix((np.ones(len(edges)), (edges[:, 0], edges[:, 1])), shape=(len(points),) * 2), directed=False)
    used = np.bincount(labels[faces[:, 0]], minlength=components) > 0
    euler = (np.bincount(labels, minlength=components) - np.bincount(labels[edges[:, 0]], minlength=components)
             + np.bincount(labels[faces[:, 0]], minlength=components))
    result['components'] = int(used.sum())
    result['b1'] = int(np.sum((2 - euler[used]) // 2))
    result['euler_by_component_min'] = int(euler[used].min(initial=2))
    if self_intersections:
        result['self_intersections'] = _self_intersections(points, faces, edges, check)
    if model is not None:
        reach = 2 * model.wall_tolerance
        outside = -reach
        for cells in (points[:, None], edges, faces):
            for start in range(0, len(cells), 200_000):
                check()
                probe = cells[start:start + 200_000] if cells.dtype.kind == 'f' else points[cells[start:start + 200_000]]
                outside = max(outside, model.evaluate(probe.mean(axis=1), reach=reach, raw=True)[0].max(initial=-reach))
        result['wall_outside_max_mm'] = float(outside)
        result['wall_tolerance_mm'] = model.wall_tolerance
    return result


def _self_intersections(points, faces, edges, check):
    triangles = points[faces]
    centres = triangles.mean(axis=1)
    radius = np.linalg.norm(triangles - centres[:, None], axis=2).max(initial=0.)
    tree = cKDTree(centres)
    hits = 0
    for start in range(0, len(edges), 5_000):
        check()
        part = edges[start:start + 5_000]
        a, b = points[part[:, 0]], points[part[:, 1]]
        lists = tree.query_ball_point((a + b) / 2, np.linalg.norm(b - a, axis=1) / 2 + radius + 1e-9)
        counts = np.fromiter(map(len, lists), np.int64, len(part))
        row = np.repeat(np.arange(len(part)), counts)
        tri = np.fromiter(chain.from_iterable(lists), np.int64, int(counts.sum()))
        shared = np.any((faces[tri] == part[row, :1]) | (faces[tri] == part[row, 1:]), axis=1)
        row, tri = row[~shared], tri[~shared]
        hits += int(np.count_nonzero(_segment_hits(a[row], b[row], triangles[tri])
                                     | _coplanar_overlaps(a[row], b[row], triangles[tri])))
    return hits


def _coplanar_overlaps(a, b, triangles, eps=1e-9):
    """Segments lying in a triangle's plane that cross its edges or lie inside it."""
    v0, v1, v2 = triangles.transpose(1, 0, 2)
    normal = np.cross(v1 - v0, v2 - v0)
    size = np.linalg.norm(normal, axis=1)
    unit = normal / np.maximum(size, 1e-30)[:, None]
    tolerance = eps * np.maximum(np.sqrt(size), 1e-30)
    flat = ((np.abs(np.einsum('ij,ij->i', a - v0, unit)) < tolerance)
            & (np.abs(np.einsum('ij,ij->i', b - v0, unit)) < tolerance) & (size > 1e-30))
    rows = np.flatnonzero(flat)
    result = np.zeros(len(a), bool)
    if not len(rows):
        return result
    keep = np.array([[1, 2], [0, 2], [0, 1]])[np.argmax(np.abs(normal[rows]), axis=1)]
    take = lambda x: np.take_along_axis(x[rows], keep, axis=1)
    p, q, t = take(a), take(b), [take(v) for v in (v0, v1, v2)]
    cross = lambda o, u, w: (u[:, 0] - o[:, 0]) * (w[:, 1] - o[:, 1]) - (u[:, 1] - o[:, 1]) * (w[:, 0] - o[:, 0])
    hit = np.zeros(len(rows), bool)
    for i in range(3):
        c, d = t[i], t[(i + 1) % 3]
        hit |= ((cross(p, q, c) * cross(p, q, d) < 0) & (cross(c, d, p) * cross(c, d, q) < 0))
    middle = (p + q) / 2
    sides = np.stack([cross(t[i], t[(i + 1) % 3], middle) for i in range(3)], axis=1)
    hit |= np.all(sides > 0, axis=1) | np.all(sides < 0, axis=1)
    result[rows] = hit
    return result


def _closest_on_triangles(p, tri):
    """Closest points on triangles (Ericson, Real-Time Collision Detection 5.1.5)."""
    a, b, c = tri[:, 0], tri[:, 1], tri[:, 2]
    ab, ac, ap = b - a, c - a, p - a
    dot = lambda x, y: np.einsum('ij,ij->i', x, y)
    d1, d2 = dot(ab, ap), dot(ac, ap)
    bp = p - b
    d3, d4 = dot(ab, bp), dot(ac, bp)
    cp = p - c
    d5, d6 = dot(ab, cp), dot(ac, cp)
    vc = d1 * d4 - d3 * d2
    vb = d5 * d2 - d1 * d6
    va = d3 * d6 - d5 * d4
    denom = np.where(np.abs(va + vb + vc) > 1e-30, va + vb + vc, 1e-30)
    v, w = vb / denom, vc / denom
    result = a + v[:, None] * ab + w[:, None] * ac
    safe = lambda x: np.where(np.abs(x) > 1e-30, x, 1e-30)
    t_bc = (d4 - d3) / safe((d4 - d3) + (d5 - d6))
    rules = [
        ((va <= 0) & (d4 - d3 >= 0) & (d5 - d6 >= 0), b + t_bc[:, None] * (c - b)),
        ((vb <= 0) & (d2 >= 0) & (d6 <= 0), a + (d2 / safe(d2 - d6))[:, None] * ac),
        ((vc <= 0) & (d1 >= 0) & (d3 <= 0), a + (d1 / safe(d1 - d3))[:, None] * ab),
        ((d6 >= 0) & (d5 <= d6), c), ((d3 >= 0) & (d4 <= d3), b), ((d1 <= 0) & (d2 <= 0), a),
    ]
    for condition, value in rules:
        result = np.where(condition[:, None], value, result)
    return result


def point_mesh_distance(points, mesh_points, faces, *, k=16):
    """Distance and closest point from points to a triangle mesh (k nearest triangle centres)."""
    points = np.asarray(points, float)
    triangles = np.asarray(mesh_points, float)[np.asarray(faces)]
    tree = cKDTree(triangles.mean(axis=1))
    k = min(k, len(triangles))
    distance = np.full(len(points), np.inf)
    closest = np.zeros_like(points)
    for start in range(0, len(points), 50_000):
        rows = slice(start, start + 50_000)
        _, near = tree.query(points[rows], k=k)
        near = near.reshape(len(points[rows]), -1)
        p = np.repeat(points[rows], near.shape[1], axis=0)
        q = _closest_on_triangles(p, triangles[near.ravel()]).reshape(*near.shape, 3)
        d = np.linalg.norm(q - points[rows][:, None], axis=2)
        best = d.argmin(axis=1)
        distance[rows] = d[np.arange(len(best)), best]
        closest[rows] = q[np.arange(len(best)), best]
    return distance, closest


def source_surface(soft, affine, *, level=LEVEL):
    """Half-level wall of the soft source signal (positions RAS mm, faces)."""
    from skimage.measure import marching_cubes

    soft = np.asarray(soft)
    occupied = np.argwhere(soft >= level)
    lo = np.maximum(occupied.min(axis=0) - 2, 0)
    hi = np.minimum(occupied.max(axis=0) + 3, soft.shape)
    crop = np.pad(np.asarray(soft[tuple(slice(a, b) for a, b in zip(lo, hi))], np.float32), 1)
    verts, faces, _, _ = marching_cubes(crop, level, method='lewiner')
    ijk = verts.astype(float) + lo - 1
    frame = checked_affine(affine)
    return ijk @ frame[:3, :3].T + frame[:3, 3], faces.astype(np.int64)


def deviation(tube_mesh, soft, affine, *, level=LEVEL, source=None, model=None, source_classes=None):
    """Two-sided wall deviation in mm, smallest-spacing voxels and index-space voxels.

    ``walls`` leaves out points within one voxel of the acquisition edge, where
    the source half level is the cut face of the scan rather than a vessel wall.
    Per-class summaries use ``source_classes`` (1 artery, 2 vein, 3 unclassified on the source grid)
    for source wall points and the owning edge's class for tube points (the source class where no
    edge owns one); without it both sides use the edge class. A point no tube owns within the search
    distance keeps its full distance and is counted under ``unowned``.
    """
    frame = checked_affine(affine)
    spacing = np.linalg.norm(frame[:3, :3], axis=0)
    axes = frame[:3, :3] / spacing
    inverse = np.linalg.inv(frame)
    shape = np.asarray(np.shape(soft))
    source_points, source_faces = source if source is not None else source_surface(soft, frame, level=level)

    def side(points, target_points, target_faces):
        distance, closest = point_mesh_distance(points, target_points, target_faces)
        ijk = points @ inverse[:3, :3].T + inverse[:3, 3]
        return (distance / spacing.min(), np.linalg.norm(((closest - points) @ axes) / spacing, axis=1),
                distance, np.all((ijk >= 1) & (ijk <= shape - 2), axis=1))

    def source_kind(points):
        # The commonest positive class among the eight samples around each wall point.
        ijk = points @ inverse[:3, :3].T + inverse[:3, 3]
        base = np.floor(ijk).astype(np.int64)
        votes = np.zeros((len(points), 4), np.int64)
        for corner in product((0, 1), repeat=3):
            cell = np.clip(base + corner, 0, shape - 1)
            votes[np.arange(len(points)), np.asarray(source_classes)[tuple(cell.T)].astype(np.int64).clip(0, 3)] += 1
        votes[:, 0] = 0
        return np.where(votes.max(axis=1) > 0, votes.argmax(axis=1), 0)

    def summary(voxels, index, distance):
        if not len(voxels):
            return {'points': 0}
        return {'points': len(voxels), 'mm_mean': float(distance.mean()), 'mm_p95': float(np.percentile(distance, 95)),
                'voxels_mean': float(voxels.mean()), 'voxels_p95': float(np.percentile(voxels, 95)),
                'voxels_max': float(voxels.max()), 'index_voxels_mean': float(index.mean()),
                'index_voxels_p95': float(np.percentile(index, 95)),
                'fraction_within_half_voxel': float(np.mean(voxels <= .5))}

    names = np.array(['none', *CLASS_NAMES.values()])
    result = {'level': level, 'voxel_mm': float(spacing.min()),
              'classes_from': 'source' if source_classes is not None else 'tube edges' if model is not None else None}
    sides = {'tube_to_source': (tube_mesh.positions, source_points, source_faces, tube_mesh.owners, False),
             'source_to_tube': (source_points, tube_mesh.positions, tube_mesh.faces, None, True)}
    for name, (points, target_points, target_faces, owners, on_source) in sides.items():
        voxels, index, distance, wall = side(points, target_points, target_faces)
        entry = {'all': summary(voxels, index, distance), 'walls': summary(voxels[wall], index[wall], distance[wall])}
        if model is not None or source_classes is not None:
            if model is not None and owners is None:
                owners = model.evaluate(points, reach=4 * float(spacing.max()), raw=True)[1]
            kinds = (np.array([*model.edge_class, 'none'])[owners] if model is not None
                     else np.full(len(points), 'none', dtype=object))
            if source_classes is not None:
                fallback = names[source_kind(points)]
                kinds = fallback if on_source else np.where(kinds == 'none', fallback, kinds)
            if owners is not None:
                entry['unowned'] = int(np.count_nonzero(wall & (owners < 0)))
            for kind in sorted(set(kinds[wall].tolist())):
                keep = wall & (kinds == kind)
                entry[kind] = summary(voxels[keep], index[keep], distance[keep])
        result[name] = entry
    return result


# ------------------------------------------------------------ tags and rods


def tag_volume(model, *, width=None, check=lambda: None):
    """Soft vessel tag on the source grid: 1 inside, 0 outside, linear across ``width`` mm.

    Returns ``(origin_ijk, values)`` with values float32 over the occupied crop.
    """
    width = float(model.spacing.min() if width is None else width)
    grid = _Grid(model, np.ones(3), width)
    lo = np.maximum(grid.origin, 0)
    lo = np.ceil(lo).astype(int)
    hi = np.minimum(np.floor(grid.origin + (grid.shape - 1)).astype(int), np.asarray(model.shape) - 1)
    if np.any(hi < lo):
        return lo, np.zeros((0, 0, 0), np.float32)
    grid = _Grid(model, np.ones(3), width, origin=lo.astype(float), shape=hi - lo + 1)
    out = np.empty(tuple(hi - lo + 1), np.float32)
    for corner in product(*[range(0, n, 64) for n in out.shape]):
        check()
        a = np.asarray(corner)
        b = np.minimum(a + 63, np.asarray(out.shape) - 1)
        values = grid.values(a, b)
        out[tuple(slice(x, y + 1) for x, y in zip(a, b))] = np.clip(.5 - values / width, 0., 1.)
    return lo, out


def rod_sites(model, cells):
    """Rod site per source cell: closest centerline point inside the cell and its radius, ``[x, y, z, r]``."""
    seg = model.segments
    real = np.flatnonzero(seg['owner'] < len(model.edges))
    tree = cKDTree(seg['centre'][real])
    cells = np.asarray(list(cells), np.int64)
    ijk = np.column_stack(np.unravel_index(cells, model.shape)).astype(float)
    centre = ijk @ model.affine[:3, :3].T + model.affine[:3, 3]
    k = min(8, len(real))
    _, near = tree.query(centre, k=k)
    near = real[near.reshape(len(cells), -1)]
    a, b = seg['a'][near], seg['b'][near]
    ab = b - a
    t = np.clip(np.einsum('nki,nki->nk', centre[:, None] - a, ab) / np.maximum(np.einsum('nki,nki->nk', ab, ab), 1e-30), 0, 1)
    foot = a + t[..., None] * ab
    best = np.linalg.norm(foot - centre[:, None], axis=2).argmin(axis=1)
    rows = np.arange(len(cells))
    point = foot[rows, best]
    radius = (1 - t[rows, best]) * seg['ra'][near[rows, best]] + t[rows, best] * seg['rb'][near[rows, best]]
    inverse = np.linalg.inv(model.affine)
    local = np.clip(point @ inverse[:3, :3].T + inverse[:3, 3], ijk - .5, ijk + .5)
    point = local @ model.affine[:3, :3].T + model.affine[:3, 3]
    radius = np.clip(radius, .03, 8.)
    return {int(c): [*map(float, p), float(r)] for c, p, r in zip(cells, point, radius)}
