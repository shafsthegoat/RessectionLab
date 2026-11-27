"""Smooth display tubes for thin vessel branches.

A vessel one to three voxels across cannot be drawn smoothly from its voxels:
a mask follows the lattice in steps and changes width from slice to slice, so
its surface beads. This module redraws only those thin parts, for display:

1. the shown voxels (with the nearby part of the tree they extend) are thinned
   to a centerline graph;
2. every centerline voxel gets a radius from the vessel volume it owns divided
   by its length of centerline, averaged along the graph, so calibre changes
   gradually;
3. centerline positions are relaxed along the graph, each kept within half a
   voxel of the voxel it came from and free ends held: lattice steps and their
   slow beat patterns straighten out, while a real bend cannot be cut by more
   than half a voxel;
4. where the radius is below the thin limit the vessel is drawn as a round
   tube: a soft field that is one half at the tube's radius, with a smallest
   radius the voxel grid can still show as an unbroken tube.

A tube this thin sits at the sampling limit of the grid. Read back with
trilinear interpolation, the field's half level wanders by about nine percent
of the width as the tube crosses the lattice, and its ``WALL_LEVEL`` by about
four; both drawings therefore take the wall at ``WALL_LEVEL``, about a fifth of
a voxel outside the radius. Thinner tubes than the smallest radius bead.

Short twigs of the centerline and the few voxels left standing between tubes
are artefacts of a lumpy mask and are drawn neither as tubes nor as voxels.
Thick vessels keep their voxels and scores, and a tube runs a short way into
the thick vessel it leaves so the two meet without a seam. Their surface is
contoured together with the tubes: from their own score field where the layer
has one, otherwise from their softened mask (``softened_mask``), which
replaces the stepped label surface of such a layer. Nothing here changes a source layer; the result is a
display field and a flag for the voxels it replaces. Distances are millimetres
on the layer's own grid, whose axes are taken as orthogonal.
"""

from dataclasses import asdict, dataclass

import numpy as np
from scipy import sparse
from scipy.spatial import cKDTree
from scipy.special import erfc

VERSION = 'thin-vessel-display-tubes-v1'
# Row key of the packed tube field of a display row (an array record of its study).
DISPLAY_TUBES = 'display_tubes'
FIELD_LEVELS = 127
# Field level drawn as the tube wall: inside the band over which the path-traced
# vessel material rises around a tag (gate .12 to .42, lumen floor .3 to .42),
# and the level the surface contour is taken at.
WALL_LEVEL = .35
# Version of the surface made from a tube field (softened mask that keeps every
# voxel, rescale, one contour).
SURFACE = 'thin-tubes-kept-voxels-no-specks-surface-v1'
# Samples within reach that may set a voxel's value, and voxels per nearest-sample query.
_NEAREST = 12
_QUERY_BLOCK = 1 << 20
_OFFSETS = np.array([(x, y, z) for x in (-1, 0, 1) for y in (-1, 0, 1) for z in (-1, 0, 1)
                     if (x, y, z) > (0, 0, 0)], np.int64)


@dataclass(frozen=True)
class TubeDisplay:
    """Frozen on the development CTA; applied unchanged to every other case."""

    version: str = VERSION
    # A centerline voxel is thin when its averaged radius is below this many mean voxel spacings.
    thin_radius_voxels: float = 1.5
    # Smallest drawn radius, in half voxel diagonals. At 0.9 the trilinear field stays above one half all
    # along a centerline in any direction (simulated on 0.3 to 0.75 mm grids), so the tube never breaks,
    # and its drawn radius ripples by about four percent; thinner tubes bead.
    min_radius_diagonals: float = .9
    # Width of the soft wall, in mean voxel spacings (the width a smoothed mask edge has).
    wall_sigma_voxels: float = .6
    radius_rounds: int = 8
    # Centerline relaxation: rounds of moving each voxel toward its neighbours' mean, and how far
    # (in voxels, per axis) it may end from the centre of the voxel it came from.
    position_rounds: int = 80
    position_step: float = .5
    position_tolerance_voxels: float = .5
    # A tube continues this many centerline steps into the thick vessel it leaves.
    overlap_steps: int = 4
    # Parent tree voxels within this distance join the centerline, so a branch ends inside its parent.
    context_mm: float = 2.5
    samples_per_edge: int = 3
    # A thin side twig of the centerline shorter than this many smallest radii is a thinning
    # artefact of a lumpy mask, not a branch: it is not drawn, and its voxels are not drawn either.
    spur_radii: float = 3.
    # Voxels left between tubes in face-connected groups smaller than this are leftovers of the
    # branches around them, not vessels of their own: they are not drawn as voxels.
    speck_voxels: int = 8

    def __post_init__(self):
        if (self.thin_radius_voxels <= 0 or self.min_radius_diagonals < .9 or self.wall_sigma_voxels <= 0
                or self.radius_rounds < 0 or self.position_rounds < 0 or self.overlap_steps < 0
                or not 0 < self.position_step <= 1 or not 0 <= self.position_tolerance_voxels <= 1
                or self.context_mm < 0 or self.samples_per_edge < 1 or self.spur_radii < 0
                or self.speck_voxels < 1):
            raise ValueError('Invalid tube display rule')


def pack(field, thin):
    """One byte per voxel: the tube field in seven bits and the replaced-voxel flag in the lowest."""
    return ((np.rint(np.clip(field, 0, 1) * FIELD_LEVELS).astype(np.uint8)) << 1) | thin.astype(np.uint8)


def unpack(packed):
    """``(field in [0, 1] as float32, replaced-voxel flag)`` of a packed tube array."""
    packed = np.asarray(packed)
    return (packed >> 1).astype(np.float32) / FIELD_LEVELS, (packed & 1).astype(bool)


def _crop(mask, margin):
    axes = [np.flatnonzero(mask.any(axis=tuple(i for i in range(3) if i != axis))) for axis in range(3)]
    lo = [max(0, int(axis[0]) - margin[i]) for i, axis in enumerate(axes)]
    hi = [min(mask.shape[i], int(axis[-1]) + 1 + margin[i]) for i, axis in enumerate(axes)]
    return tuple(slice(a, b) for a, b in zip(lo, hi))


def _graph(points, shape):
    """Edges between 26-adjacent centerline voxels, each once."""
    strides = np.array([shape[1] * shape[2], shape[2], 1], np.int64)
    linear = points @ strides
    order = np.argsort(linear)
    ordered = linear[order]
    edges = []
    for offset in _OFFSETS:
        moved = points + offset
        inside = np.all((moved >= 0) & (moved < shape), axis=1)
        target = moved[inside] @ strides
        at = np.searchsorted(ordered, target)
        at[at == len(ordered)] = 0
        found = ordered[at] == target
        edges.append(np.stack([np.flatnonzero(inside)[found], order[at[found]]], axis=1))
    return np.concatenate(edges) if edges else np.empty((0, 2), np.int64)


def _averaging(edges, count):
    """Row-normalised neighbour average; a voxel without neighbours keeps its own value."""
    both = np.concatenate([edges, edges[:, ::-1]])
    adjacency = sparse.csr_matrix((np.ones(len(both)), (both[:, 0], both[:, 1])), shape=(count, count))
    degree = np.asarray(adjacency.sum(axis=1)).ravel()
    lonely = degree == 0
    scale = sparse.diags(1 / np.where(lonely, 1, degree))
    return scale @ adjacency + sparse.diags(lonely.astype(float)), degree


def thin_tubes(own, spacing, *, context=None, rule=TubeDisplay(), check=lambda: None):
    """Tube field for the thin parts of ``own``: ``(packed array or None, receipt)``.

    ``context`` holds voxels of the tree that ``own`` extends; they shape the
    centerline near a junction and are never redrawn. The packed array (see
    ``unpack``) holds the field, one half on the tube wall, and flags the
    ``own`` voxels the tubes replace.
    """
    from skimage.morphology import skeletonize

    own = np.asarray(own, bool)
    spacing = np.asarray(spacing, float)
    if own.ndim != 3 or spacing.shape != (3,) or not np.all(spacing > 0):
        raise ValueError('Tube display needs a 3D mask and three positive spacings')
    if context is not None and np.asarray(context).shape != own.shape:
        raise ValueError('The tree and the shown voxels are on different grids')
    mean = float(np.prod(spacing) ** (1 / 3))
    half_diagonal = float(np.linalg.norm(spacing) / 2)
    thin_mm = rule.thin_radius_voxels * mean
    minimum = rule.min_radius_diagonals * half_diagonal
    sigma = rule.wall_sigma_voxels * mean
    receipt = {'version': rule.version, 'rule': asdict(rule), 'thinRadiusMm': thin_mm, 'minimumDrawnRadiusMm': minimum,
               'wallSigmaMm': sigma, 'centerlineVoxels': 0, 'tubeCenterlineVoxels': 0, 'tubeLengthMm': 0.,
               'replacedVoxels': 0, 'ownVoxels': int(own.sum())}
    if not own.any():
        return None, receipt
    reach = max(thin_mm, minimum) + 3 * sigma
    margin = [int(np.ceil((reach + rule.context_mm) / step)) + 2 for step in spacing]
    box = _crop(own, margin)
    origin = np.array([part.start for part in box])
    mine = own[box]
    work = mine.copy()
    if context is not None and rule.context_mm > 0:
        near = np.asarray(context[box], bool) & ~mine
        if near.any():
            # Only the part of the tree beside these voxels: it anchors the junctions.
            cells = np.argwhere(near)
            distance, _ = cKDTree(np.argwhere(mine) * spacing).query(cells * spacing, distance_upper_bound=rule.context_mm)
            keep = cells[np.isfinite(distance)]
            work[tuple(keep.T)] = True
    check()
    skeleton = skeletonize(work)
    check()
    points = np.argwhere(skeleton)
    count = len(points)
    receipt['centerlineVoxels'] = count
    if count < 2:
        return None, receipt
    edges = _graph(points, np.array(work.shape))
    average, degree = _averaging(edges, count)
    position = points * spacing
    own_skeleton = mine[tuple(points.T)]
    if not own_skeleton.any():
        return None, receipt
    # Radius: owned vessel volume over owned centerline length, both averaged along
    # the graph. Only this layer's voxels and centerline count: the tree beside it
    # shapes the centerline but must neither thicken nor thin these branches.
    cells = np.argwhere(mine)
    own_index = np.flatnonzero(own_skeleton)
    owner = own_index[cKDTree(position[own_index]).query(cells * spacing)[1]]
    volume = np.bincount(owner, minlength=count) * float(np.prod(spacing))
    inner = edges[own_skeleton[edges[:, 0]] & own_skeleton[edges[:, 1]]]
    radius_average, _ = _averaging(inner, count)
    length = np.zeros(count)
    if len(inner):
        span = np.linalg.norm(position[inner[:, 0]] - position[inner[:, 1]], axis=1)
        np.add.at(length, inner[:, 0], span / 2)
        np.add.at(length, inner[:, 1], span / 2)
    length = np.maximum(length, mean / 2)  # A lone voxel is a short stub, not a point.
    for _ in range(rule.radius_rounds):
        volume, length = (volume + radius_average @ volume) / 2, (length + radius_average @ length) / 2
    radius = np.sqrt(volume / (np.pi * length))
    check()
    thin = (radius < thin_mm) & own_skeleton
    if not thin.any():
        return None, receipt
    # Short thin twigs: from each free end along the chain to the first fork.
    spur = np.zeros(count, bool)
    if rule.spur_radii > 0 and len(edges):
        both = np.concatenate([edges, edges[:, ::-1]])
        order = np.argsort(both[:, 0], kind='stable')
        neighbours, starts = both[order, 1], np.searchsorted(both[order, 0], np.arange(count + 1))
        longest = rule.spur_radii * minimum
        for tip in np.flatnonzero((degree == 1) & thin):
            chain, travelled, previous, current = [tip], 0., -1, tip
            while True:
                onward = [n for n in neighbours[starts[current]:starts[current + 1]] if n != previous]
                if len(onward) != 1:
                    break  # A fork (or nothing): the twig ends here.
                step = float(np.linalg.norm(position[onward[0]] - position[current]))
                if travelled + step >= longest or not thin[onward[0]]:
                    chain = None
                    break
                travelled += step
                previous, current = current, onward[0]
                if degree[current] > 2:
                    break
                chain.append(current)
            if chain is not None and degree[current] > 2:
                spur[chain] = True
    check()
    replaced = np.zeros(work.shape, bool)
    replaced[tuple(cells.T)] = thin[owner]
    specks = 0
    if rule.speck_voxels > 1 and replaced.any():
        from scipy import ndimage as ndi

        groups, found = ndi.label(mine & ~replaced)
        if found:
            small = np.flatnonzero(np.bincount(groups.ravel(), minlength=found + 1)[1:] < rule.speck_voxels) + 1
            leftover = np.isin(groups, small)
            specks = int(leftover.sum())
            replaced |= leftover
            del leftover
        del groups
    receipt['leftoverVoxels'] = specks
    check()
    if not (thin & ~spur).any():
        # Only twigs were thin: they are left out of the drawing and nothing replaces them.
        packed = np.zeros(own.shape, np.uint8)
        packed[box] = pack(np.zeros(work.shape, np.float32), replaced)
        receipt.update(prunedTwigVoxels=int(spur.sum()), replacedVoxels=int(replaced.sum()), origin=origin.tolist())
        return packed, receipt
    # Relaxed centerline: free ends and lone voxels stay where they are, and no
    # voxel leaves the tolerance box around its own centre.
    held = degree <= 1
    limit = rule.position_tolerance_voxels * spacing
    smooth = position.copy()
    for _ in range(rule.position_rounds):
        moved = smooth + rule.position_step * (average @ smooth - smooth)
        moved = np.clip(moved, position - limit, position + limit)
        moved[held] = position[held]
        smooth = moved
    # Drawn part of the graph: the thin voxels and a few steps into what they leave.
    drawn = thin & ~spur
    both = np.concatenate([edges, edges[:, ::-1]])
    adjacency = sparse.csr_matrix((np.ones(len(both), bool), (both[:, 0], both[:, 1])), shape=(count, count))
    for _ in range(rule.overlap_steps):
        drawn |= ((adjacency @ drawn.astype(np.int32)) > 0) & ~spur
    # Inside a thick vessel the connector is no wider than the thin limit.
    draw_radius = np.maximum(np.where(thin, radius, np.minimum(radius, thin_mm)), minimum)
    kept = drawn[edges[:, 0]] & drawn[edges[:, 1]]
    segments = edges[kept]
    steps = np.arange(1, rule.samples_per_edge + 1) / (rule.samples_per_edge + 1)
    centres = [smooth[drawn]]
    radii = [draw_radius[drawn]]
    for step in steps:
        centres.append(smooth[segments[:, 0]] * (1 - step) + smooth[segments[:, 1]] * step)
        radii.append(draw_radius[segments[:, 0]] * (1 - step) + draw_radius[segments[:, 1]] * step)
    centres, radii = np.concatenate(centres), np.concatenate(radii)
    check()
    # Voxels the tubes can reach: boxes around the drawn centerline voxels.
    grow = [int(np.ceil((float(radii.max()) + 3 * sigma) / step)) + 1 for step in spacing]
    seeds = np.zeros(work.shape, bool)
    index = np.clip(np.rint(smooth[drawn] / spacing).astype(int), 0, np.array(work.shape) - 1)
    seeds[tuple(index.T)] = True
    from scipy import ndimage as ndi

    for axis, steps_ in enumerate(grow):
        shape = [1, 1, 1]
        shape[axis] = 2 * steps_ + 1
        seeds = ndi.binary_dilation(seeds, structure=np.ones(shape, bool))
    candidates = np.argwhere(seeds)
    del seeds
    # The field is the largest tube value over every sample in reach, so where a
    # small tube leaves a larger one the larger wall is not dented by nearer samples
    # of the smaller. Out-of-reach neighbours point one past the end, at radius zero.
    tree = cKDTree(centres)
    reach_mm = float(radii.max()) + 3 * sigma
    padded = np.append(radii, 0.)
    levels = np.zeros(work.shape, np.uint8)
    for start in range(0, len(candidates), _QUERY_BLOCK):
        block = candidates[start:start + _QUERY_BLOCK]
        distance, nearest = tree.query(block * spacing, k=min(_NEAREST, len(centres)), distance_upper_bound=reach_mm, workers=-1)
        if distance.ndim == 1:
            distance, nearest = distance[:, None], nearest[:, None]
        value = (.5 * erfc((distance - padded[nearest]) / (np.sqrt(2) * sigma))).max(axis=1)
        levels[tuple(block.T)] = np.rint(np.clip(value, 0, 1) * FIELD_LEVELS).astype(np.uint8)
        del distance, nearest, value
        check()
    packed = np.zeros(own.shape, np.uint8)
    packed[box] = (levels << 1) | replaced.astype(np.uint8)
    del levels
    if len(segments):
        thin_segments = (thin & ~spur)[segments[:, 0]] & (thin & ~spur)[segments[:, 1]]
        tube_length = float(np.linalg.norm(smooth[segments[thin_segments, 0]] - smooth[segments[thin_segments, 1]], axis=1).sum())
    else:
        tube_length = 0.
    shown = radii[:int(drawn.sum())][thin[drawn]]
    receipt.update(tubeCenterlineVoxels=int((thin & ~spur).sum()), prunedTwigVoxels=int(spur.sum()), tubeLengthMm=tube_length, replacedVoxels=int(replaced.sum()),
                   drawnRadiusMm={'min': float(shown.min()), 'median': float(np.median(shown)), 'max': float(shown.max())},
                   atMinimumRadius=int((radius[thin] < minimum).sum()), origin=origin.tolist(),
                   maxCenterlineShiftMm=float(np.linalg.norm(smooth - position, axis=1)[drawn].max()))
    return packed, receipt


def surface_field(field):
    """The tube field rescaled so that a contour at one half is the wall at ``WALL_LEVEL``."""
    return np.minimum(1, np.asarray(field, np.float32) * np.float32(.5 / WALL_LEVEL))


def without_lone_samples(values, keep, *, margin=1 / 64):
    """A contour field without single lattice samples that alone reach one half.

    Where a tube's skirt just touches the level at one sample and at none of
    its six neighbours, the contour is a speck a fraction of a voxel across
    that belongs to no vessel. Such a sample is set just below the level;
    samples in ``keep`` (the layer's own remaining voxels) are never changed."""
    above = values >= .5
    beside = np.zeros(values.shape, bool)
    for axis in range(3):
        for shift in (1, -1):
            moved = np.roll(above, shift, axis=axis)
            edge = [slice(None)] * 3
            edge[axis] = 0 if shift == 1 else -1
            moved[tuple(edge)] = False
            beside |= moved
    lone = above & ~beside & ~np.asarray(keep, bool)
    if lone.any():
        values = values.copy()
        values[lone] = .5 - margin
    return values


def softened_mask(mask, *, sigma_voxels=.6, margin=1 / 64):
    """A mask as a contour field: smoothed, but still above one half at every voxel of the mask and below
    it at every other, so the half-level surface keeps the mask's face-connected pieces together."""
    from scipy import ndimage as ndi

    mask = np.asarray(mask, bool)
    soft = ndi.gaussian_filter(mask.astype(np.float32), sigma_voxels, mode='constant', truncate=2 / sigma_voxels)
    return np.where(mask, np.maximum(soft, np.float32(.5 + margin)), np.minimum(soft, np.float32(.5 - margin)))


def thick_mask(own, packed):
    """The shown voxels the tubes do not replace."""
    return np.asarray(own, bool) & ~unpack(packed)[1]


def display_tags(own, packed, *, sigma_voxels=.6):
    """Soft tags for the path-traced volume, 0 to 255: the thick voxels smoothed as a mask is, joined to the tube field."""
    from scipy import ndimage as ndi

    packed = np.asarray(packed)
    # The tube field in tag units, in integers: no full-grid float array is made.
    tags = ((packed >> 1).astype(np.uint16) * 255 + FIELD_LEVELS // 2) // FIELD_LEVELS
    tags = tags.astype(np.uint8)
    thick = np.asarray(own, bool) & ((packed & 1) == 0)
    if thick.any():
        box = _crop(thick, [3, 3, 3])
        soft = ndi.gaussian_filter(thick[box].astype(np.float32), sigma_voxels, mode='constant', truncate=2 / sigma_voxels)
        tags[box] = np.maximum(tags[box], np.rint(soft * 255).astype(np.uint8))
    return tags
