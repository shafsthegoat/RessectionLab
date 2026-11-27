"""Source-resolution anatomy surfaces for the operative Rehearse view.

The engine of a host rehearsal (``surgical_host_rehearsal``) works on a 1 mm
grid resampled nearest-neighbour from a 1.35 mm prepared study. Meshing those
cells draws terraces. This module builds, once per operative case, smooth
surfaces from the finer sources the prepared study was made from (the native
T1, the 1 mm categorical head model, the vessel model probability and the
published tracts), in RAS millimetres, and binds every vertex to the engine so
a renderer can move flap patches by a body transform and hide vertices whose
cell was removed without meshing voxels.

Surfaces are isosurfaces of Gaussian-smoothed partial-volume fields, sampled on
a grid aligned with the working grid, then relaxed by Laplacian rounds with
each vertex held within ``constraintMm`` of the source label boundary (a
constrained smoothing: label terraces lie within half a voxel of the boundary
they digitise, so the relaxed surface passes through them without stepping),
and finished by a few Taubin rounds. The skin follows the acquired T1 where it
is given (Taubin rounds only, held near the head-model boundary).
Every surface records its source file, checksum, resolution, smoothing and
whether it is acquired, estimated, assumed or generated, and its measured
distance from the source label boundary.

The asset is one ``.npz`` holding a JSON header and the arrays, keyed by the
operative case hash and the host case hash. It is a display companion of the
engine, never an input to it: removal, contact and replay stay with the engine.
"""
from __future__ import annotations

import hashlib
import json
import math
import os
from pathlib import Path
import tempfile
import time
import zipfile

import numpy as np
from scipy import ndimage, sparse
from scipy.sparse.csgraph import connected_components
from scipy.spatial import cKDTree

SCHEMA = 'resectionlab.operative-source-surfaces.v1'
ASSET_NAME = 'operative-source-surfaces.npz'
HEADER_KEY = 'header.json'
SCOPE = ('Display surfaces of a rehearsal derived from model estimates and acquired images. The skull is a T1 '
         'model estimate, not CT; the dura is assumed; vessels and tracts are incomplete estimates. Not a patient '
         'plan and not an engine input.')

# SuperSynth / FreeSurfer codes of the EDEN2020 v2 head model.
BRAIN_IDS = frozenset({2, 3, 7, 8, 10, 11, 12, 13, 16, 17, 18, 26, 41, 42, 46, 47, 49, 50, 51, 52, 53, 54, 58, 77,
                       85, 99, 819, 820, 821, 822, 843, 844, 865, 866, 869, 870})
CSF_IDS = frozenset({4, 5, 14, 15, 24, 43, 44})
WHITE_IDS = frozenset({2, 41, 77})
CORTEX_IDS = frozenset({3, 42})
BONE_IDS = frozenset({915, 916})
ARTERY_ID, VEIN_ID = 902, 914
OUTSIDE_IDS = frozenset({0, 917})
# Deep structures closed inside the white surface (as a FreeSurfer white surface encloses them).
INNER_IDS = WHITE_IDS | (BRAIN_IDS - CORTEX_IDS - {7, 8, 46, 47, 16}) | {4, 5, 14, 15, 43, 44}

# name, operative roles it bounds, engine body it may belong to
SURFACES = (
    ('skin', ('skin',), 'scalp'),
    ('skullOuter', ('cortical_bone', 'cancellous_bone'), 'bone'),
    ('skullInner', ('cortical_bone', 'cancellous_bone'), 'bone'),
    ('dura', ('dura',), 'dura'),
    ('pial', ('cortex', 'cerebellum', 'brainstem'), None),
    ('white', ('white', 'deep'), None),
    ('tumourRim', ('tumour_rim',), None),
    ('tumourCore', ('tumour_core',), None),
)
BODY_CODES = {'scalp': 1, 'bone': 2, 'dura': 3}
# Operative roles whose cells make up each engine body's layer.
BODY_LAYER_ROLES = {'scalp': ('skin', 'subcutaneous'), 'bone': ('cortical_bone', 'cancellous_bone'),
                    'dura': ('dura',)}
# Per surface: grid step (mm), Gaussian sigma (mm), Taubin rounds, constraint (mm).
# Label-derived surfaces use plain (shrinking) Laplacian rounds held by the constraint: Taubin
# rounds keep wavelengths over about a dozen edges, which is where label terraces live.
DEFAULTS = {
    'skin': dict(step=.8, sigma=.8, rounds=12, constraint=1.5, taubin=True),
    'skullOuter': dict(step=.8, sigma=1., rounds=80, constraint=.6, taubin=False),
    'skullInner': dict(step=.8, sigma=1., rounds=80, constraint=.6, taubin=False),
    'pial': dict(step=.8, sigma=.55, rounds=40, constraint=.45, taubin=False),
    'white': dict(step=.8, sigma=.55, rounds=40, constraint=.45, taubin=False),
    'tumourRim': dict(step=.8, sigma=1., rounds=40, constraint=.7, taubin=False),
    'tumourCore': dict(step=.8, sigma=1., rounds=40, constraint=.7, taubin=False),
}
MARGIN_MM = 10.
FINISH_ROUNDS = 4
# The assumed dural sheet: a continuous membrane lining the inner table over the craniotomy and a
# margin around it. Positions are its outer face, DURA_OFFSET_MM inside the inner table; the
# membrane is DURA_THICKNESS_MM thick (assumed: no dura is imaged or labelled).
DURA_OFFSET_MM = .1
DURA_THICKNESS_MM = .6
DURA_REACH_MM = 12.
# The skin covers the whole head the T1 shows (the overview); other surfaces stay in the working box.
SKIN_HEAD_MARGIN_MM = 4.
VESSEL_CLASS = {0: 'unknown', 1: 'artery', 2: 'vein'}
# Cortical surface vessels below the model's mask threshold: a connected extension of the mask.
VESSEL_LOW_THRESHOLD = .3
VESSEL_SURFACE_BAND_MM = (-1., 3.)
TRACT_STREAMLINES = 160
TRACT_STEP_MM = 2.


def sha256_file(path):
    digest = hashlib.sha256()
    with open(path, 'rb') as handle:
        for block in iter(lambda: handle.read(1 << 20), b''):
            digest.update(block)
    return digest.hexdigest()


def _load_nifti(path, dtype=None):
    import nibabel as nib
    image = nib.load(str(path))
    data = np.asarray(image.dataobj)
    if dtype is not None:
        data = data.astype(dtype, copy=False)
    affine = np.asarray(image.affine, float)
    return data, affine, [float(v) for v in np.linalg.norm(affine[:3, :3], axis=0)]


def _case_affine(case):
    affine = np.asarray(case.affine, float)
    if case.frame == 'LPS+':
        affine = np.diag([-1., -1., 1., 1.]) @ affine
    return affine


# ----------------------------------------------------------------------------- operative case

def operative_context(case):
    """Declarations, working grid, layer labels and body masks of an operative case."""
    from .core import thaw_json
    from .surgical_host_rehearsal import PROPERTY
    metadata = thaw_json(case.metadata)
    properties = metadata.get('surgical_properties') or {}
    declaration = properties.get(PROPERTY)
    exposure = properties.get('operativeExposureV1')
    if not declaration or not exposure:
        raise ValueError('SOURCE_SURFACES_NOT_HOST_REHEARSAL: the case declares no host rehearsal and exposure')
    affine = _case_affine(case)
    shape = tuple(int(v) for v in case.mri.shape)
    role_of = {row['name']: row['role'] for row in declaration['layers']}
    names = [row['name'] for row in declaration['layers'] if row['name'] in case.compartments]
    labels = np.zeros(shape, np.uint16)
    for index, name in enumerate(names, 1):
        labels[np.asarray(case.compartments[name], bool)] = index
    roles = {}
    for index, name in enumerate(names, 1):
        roles.setdefault(role_of[name], []).append(index)
    bodies = {}
    for name, body in exposure['bodies'].items():
        mask = np.zeros(shape, bool)
        mask.flat[np.asarray(body['cells'], np.int64)] = True
        hinge = np.zeros(shape, bool)
        if body.get('hingeEdges'):
            hinge.flat[np.asarray([edge[0] for edge in body['hingeEdges']], np.int64)] = True
        bodies[name] = {'mask': mask, 'hinge': hinge}
    return {'case': case, 'declaration': declaration, 'exposure': exposure, 'affine': affine, 'shape': shape,
            'labels': labels, 'names': names, 'roleOf': role_of, 'roleIds': roles, 'bodies': bodies,
            'metadata': metadata}


def _role_mask(context, roles):
    ids = [i for role in roles for i in context['roleIds'].get(role, [])]
    return np.isin(context['labels'], ids) if ids else np.zeros(context['shape'], bool)


def layer_names(context, roles):
    return [name for name in context['names'] if context['roleOf'][name] in roles]


# ----------------------------------------------------------------------------- sampling grid

def sampling_grid(affine, shape, *, step, margin=MARGIN_MM):
    """A grid aligned with the working grid, ``step`` mm apart, covering the box plus ``margin`` mm."""
    spacing = float(np.linalg.norm(affine[:3, 0]))
    cells = margin / spacing
    count = tuple(int(math.floor((n - 1 + 2 * cells) * spacing / step)) + 1 for n in shape)
    local = np.diag([step / spacing] * 3 + [1.])
    local[:3, 3] = -cells
    return count, affine @ local


def covering_grid(affine, shape, *, step, mask, mask_affine, margin=SKIN_HEAD_MARGIN_MM, limit=None):
    """A grid aligned with the working grid, ``step`` mm apart, covering ``mask`` (on ``mask_affine``) plus
    ``margin`` mm and at least the working box plus ``MARGIN_MM``; ``limit`` (affine, shape) bounds it to
    another image's field of view."""
    spacing = float(np.linalg.norm(affine[:3, 0]))
    inverse = np.linalg.inv(affine)

    def extent(cells, source_affine):
        world = np.asarray(cells, float) @ source_affine[:3, :3].T + source_affine[:3, 3]
        ijk = world @ inverse[:3, :3].T + inverse[:3, 3]
        return ijk.min(axis=0), ijk.max(axis=0)

    corners = lambda lo, hi: np.array(np.meshgrid(*[[a, b] for a, b in zip(lo, hi)], indexing='ij')).reshape(3, -1).T
    found = np.argwhere(mask)
    lo, hi = extent(corners(found.min(axis=0), found.max(axis=0)), mask_affine)
    lo, hi = lo - margin / spacing, hi + margin / spacing
    if limit is not None:
        limit_affine, limit_shape = limit
        fov_lo, fov_hi = extent(corners([0, 0, 0], np.array(limit_shape) - 1), limit_affine)
        lo, hi = np.maximum(lo, fov_lo), np.minimum(hi, fov_hi)
    box = MARGIN_MM / spacing
    lo = np.minimum(lo, -box)
    hi = np.maximum(hi, np.array(shape) - 1 + box)
    count = tuple(int(math.floor((b - a) * spacing / step)) + 1 for a, b in zip(lo, hi))
    local = np.diag([step / spacing] * 3 + [1.])
    local[:3, 3] = lo
    return count, affine @ local


def _crop_bounds(source_affine, source_shape, grid_shape, grid_affine, pad):
    corners = np.array(np.meshgrid(*[[0, n - 1] for n in grid_shape], indexing='ij')).reshape(3, -1).T
    world = corners @ grid_affine[:3, :3].T + grid_affine[:3, 3]
    inverse = np.linalg.inv(source_affine)
    ijk = world @ inverse[:3, :3].T + inverse[:3, 3]
    lo = np.clip(np.floor(ijk.min(axis=0) - pad), 0, np.array(source_shape)).astype(int)
    hi = np.clip(np.ceil(ijk.max(axis=0) + pad) + 1, 0, np.array(source_shape)).astype(int)
    return tuple(slice(a, b) for a, b in zip(lo, hi))


def _translate(affine, offset):
    moved = np.array(affine, float)
    moved[:3, 3] = affine[:3, :3] @ np.asarray(offset, float) + affine[:3, 3]
    return moved


def resample(volume, source_affine, grid_shape, grid_affine, *, order=1, cval=0.):
    """Sample ``volume`` (on ``source_affine``) at every node of a grid, in slabs."""
    transform = np.linalg.inv(source_affine) @ grid_affine
    out = np.empty(grid_shape, np.float32)
    ii, jj = np.meshgrid(np.arange(grid_shape[0]), np.arange(grid_shape[1]), indexing='ij')
    base = transform[:3, 0:1] * ii.ravel() + transform[:3, 1:2] * jj.ravel() + transform[:3, 3:4]
    for k in range(grid_shape[2]):
        coordinates = base + transform[:3, 2:3] * k
        out[:, :, k] = ndimage.map_coordinates(volume, coordinates, order=order, mode='constant',
                                               cval=cval, prefilter=False).reshape(grid_shape[:2])
    return out


def sample_points(volume, affine, points, *, order=1, cval=0.):
    inverse = np.linalg.inv(affine)
    ijk = np.asarray(points, float) @ inverse[:3, :3].T + inverse[:3, 3]
    return ndimage.map_coordinates(volume, ijk.T, order=order, mode='nearest' if cval is None else 'constant',
                                   cval=0. if cval is None else cval, prefilter=False)


def label_distance(mask, spacing):
    """Signed distance (mm) to the boundary of a binary label at voxel centres; positive outside.

    The boundary is taken between voxel centres, so a centre next to the
    boundary is half a voxel from it.
    """
    half = .5 * float(min(spacing))
    outside = ndimage.distance_transform_edt(~mask, sampling=spacing)
    inside = ndimage.distance_transform_edt(mask, sampling=spacing)
    return np.where(mask, half - inside, outside - half).astype(np.float32)


# ----------------------------------------------------------------------------- mesh operations

def _unique_edges(faces):
    edges = np.sort(np.concatenate([faces[:, [0, 1]], faces[:, [1, 2]], faces[:, [2, 0]]]), axis=1)
    return np.unique(edges, axis=0, return_counts=True)


def vertex_normals(points, faces):
    a, b, c = (points[faces[:, i]] for i in range(3))
    face = np.cross(b - a, c - a)
    normals = np.zeros_like(points)
    for i in range(3):
        np.add.at(normals, faces[:, i], face)
    length = np.linalg.norm(normals, axis=1)
    normals /= np.maximum(length, 1e-12)[:, None]
    return normals


def _umbrella(faces, count):
    edges, _ = _unique_edges(faces)
    rows = np.concatenate([edges[:, 0], edges[:, 1]])
    cols = np.concatenate([edges[:, 1], edges[:, 0]])
    adjacency = sparse.csr_matrix((np.ones(len(rows)), (rows, cols)), shape=(count, count))
    degree = np.asarray(adjacency.sum(axis=1)).ravel()
    return sparse.diags(1. / np.maximum(degree, 1)) @ adjacency


def boundary_vertices(faces, count):
    edges, uses = _unique_edges(faces)
    pinned = np.zeros(count, bool)
    pinned[edges[uses == 1].ravel()] = True
    return pinned


def constrained_taubin(points, faces, *, rounds, constraint=None, limit=None, lam=.5, mu=-.53):
    """Taubin smoothing (``mu=None``: plain Laplacian); with ``constraint(points) -> signed mm`` vertices
    are held within ``limit`` of its zero level."""
    points = np.array(points, float)
    if rounds <= 0 or not len(faces):
        return points
    operator = _umbrella(faces, len(points))
    pinned = boundary_vertices(faces, len(points))
    fixed = points[pinned].copy()
    for _ in range(rounds):
        for factor in ((lam,) if mu is None else (lam, mu)):
            points += factor * (operator @ points - points)
            points[pinned] = fixed
        if constraint is not None:
            distance = constraint(points)
            # A bounded step per round keeps a vertex far from the boundary from overshooting.
            excess = np.clip(distance - np.clip(distance, -limit, limit), -.5, .5)
            moving = np.abs(excess) > 0
            if moving.any():
                normals = vertex_normals(points, faces)
                points[moving] -= excess[moving, None] * normals[moving]
    return points


def drop_small_components(points, faces, *, minimum_area_mm2):
    """Remove mesh pieces smaller than ``minimum_area_mm2``; returns points, faces, pieces dropped."""
    if not len(faces):
        return points, faces, 0
    count, labels = connected_components(sparse.coo_matrix(
        (np.ones(2 * len(faces)), (faces[:, :2].ravel(), faces[:, 1:].ravel())), shape=(len(points),) * 2),
        directed=False)
    a, b, c = (points[faces[:, i]] for i in range(3))
    area = .5 * np.linalg.norm(np.cross(b - a, c - a), axis=1)
    piece = labels[faces[:, 0]]
    totals = np.bincount(piece, weights=area, minlength=count)
    keep_piece = totals >= minimum_area_mm2
    kept = keep_piece[piece]
    faces = faces[kept]
    used = np.zeros(len(points), bool)
    used[faces.ravel()] = True
    index = np.cumsum(used) - 1
    return points[used], index[faces], int(np.count_nonzero(~keep_piece & (totals > 0)))


def isosurface(field, grid_affine, level, *, outward_decreasing=True):
    """Marching-cubes surface of ``field`` at ``level``; faces wound so normals point outward."""
    from skimage.measure import marching_cubes
    if not (field.min() < level < field.max()):
        return np.zeros((0, 3)), np.zeros((0, 3), np.int64)
    verts, faces, _, _ = marching_cubes(field, level, allow_degenerate=False)
    points = verts @ grid_affine[:3, :3].T + grid_affine[:3, 3]
    faces = faces.astype(np.int64)
    # Winding: the outward face normal points toward lower field values.
    a, b, c = (verts[faces[:, i]] for i in range(3))
    normal = np.cross(b - a, c - a)
    centre = (a + b + c) / 3
    probe = np.linspace(0, len(faces) - 1, min(len(faces), 4000)).astype(int)
    unit = normal[probe] / np.maximum(np.linalg.norm(normal[probe], axis=1), 1e-12)[:, None]
    ahead = ndimage.map_coordinates(field, (centre[probe] + .4 * unit).T, order=1, mode='nearest')
    behind = ndimage.map_coordinates(field, (centre[probe] - .4 * unit).T, order=1, mode='nearest')
    decreasing = np.mean(ahead < behind) > .5
    if decreasing != outward_decreasing:
        faces = faces[:, ::-1].copy()
    return points, faces


def mesh_quality(points, faces, normals):
    if not len(faces):
        return {'meanEdgeMm': None, 'normalStepMeanDeg': None, 'normalStepP99Deg': None}
    edges, _ = _unique_edges(faces)
    length = np.linalg.norm(points[edges[:, 0]] - points[edges[:, 1]], axis=1)
    cosine = np.clip(np.einsum('ij,ij->i', normals[edges[:, 0]], normals[edges[:, 1]]), -1, 1)
    angle = np.degrees(np.arccos(cosine))
    a, b, c = (points[faces[:, i]] for i in range(3))
    area = .5 * np.linalg.norm(np.cross(b - a, c - a), axis=1)
    return {'meanEdgeMm': round(float(length.mean()), 4), 'normalStepMeanDeg': round(float(angle.mean()), 3),
            'normalStepP99Deg': round(float(np.percentile(angle, 99)), 3),
            'areaMm2': round(float(area.sum()), 1)}


def label_deviation(points, mask, affine, *, samples=40000, seed=20261127):
    """Distance (mm) from surface vertices to the source label's own boundary.

    The reference is the marching-cubes surface of the unsmoothed binary label
    at level 0.5 (the label boundary midway between voxel centres).
    """
    from skimage.measure import marching_cubes
    from .vessel_tubes import point_mesh_distance
    if not len(points) or not mask.any():
        return {'meanAbsMm': None, 'p95Mm': None, 'maxMm': None, 'samples': 0}
    rng = np.random.default_rng(seed)
    chosen = points if len(points) <= samples else points[rng.choice(len(points), samples, replace=False)]
    inverse = np.linalg.inv(affine)
    ijk = chosen @ inverse[:3, :3].T + inverse[:3, 3]
    lo = np.maximum(np.floor(ijk.min(axis=0)) - 4, 0).astype(int)
    hi = np.minimum(np.ceil(ijk.max(axis=0)) + 5, mask.shape).astype(int)
    crop = np.pad(mask[tuple(slice(a, b) for a, b in zip(lo, hi))].astype(np.float32), 1)
    verts, faces, _, _ = marching_cubes(crop, .5)
    reference = (verts - 1 + lo) @ affine[:3, :3].T + affine[:3, 3]
    distance, _ = point_mesh_distance(chosen, reference, faces)
    return {'meanAbsMm': round(float(distance.mean()), 4), 'p95Mm': round(float(np.percentile(distance, 95)), 4),
            'maxMm': round(float(distance.max()), 4), 'samples': int(len(chosen)),
            'reference': 'marching cubes of the unsmoothed source label at 0.5'}


# ----------------------------------------------------------------------------- engine binding

def bind_cells(points, context, roles, *, reach_mm=3.):
    """Containing working cell of each vertex (-1 outside) and the nearest cell of ``roles`` within reach."""
    affine, shape = context['affine'], context['shape']
    inverse = np.linalg.inv(affine)
    ijk = np.rint(points @ inverse[:3, :3].T + inverse[:3, 3]).astype(np.int64)
    inside = np.all((ijk >= 0) & (ijk < np.array(shape)), axis=1)
    cells = np.full(len(points), -1, np.int64)
    cells[inside] = np.ravel_multi_index(ijk[inside].T, shape)
    owner = np.full(len(points), -1, np.int64)
    mask = _role_mask(context, roles)
    if mask.any() and inside.any():
        spacing = float(np.linalg.norm(affine[:3, 0]))
        distance, nearest = ndimage.distance_transform_edt(~mask, sampling=spacing, return_indices=True)
        found = ijk[inside]
        near = distance[tuple(found.T)] <= reach_mm
        target = np.stack([nearest[a][tuple(found.T)] for a in range(3)], axis=1)
        values = np.full(len(found), -1, np.int64)
        values[near] = np.ravel_multi_index(target[near].T, shape)
        owner[inside] = values
    return cells.astype(np.int32), owner.astype(np.int32)


def body_footprint(context, body, *, smoothing_cells=1.5):
    """Smoothed signed in-plane distance (mm, positive inside) to a body's column footprint."""
    mask = context['bodies'][body]['mask']
    footprint = mask.any(axis=2)
    spacing = float(np.linalg.norm(context['affine'][:3, 0]))
    inside = ndimage.distance_transform_edt(footprint, sampling=spacing)
    outside = ndimage.distance_transform_edt(~footprint, sampling=spacing)
    signed = np.where(footprint, inside - .5 * spacing, -(outside - .5 * spacing))
    return ndimage.gaussian_filter(signed.astype(np.float64), smoothing_cells, mode='nearest')


def bind_body(points, context, body, *, reach_mm=3., lateral_cells=4):
    """Body flag and signed in-plane distance to the body's smooth outline for every vertex.

    A vertex belongs to the body when its column lies inside the smoothed
    footprint and it lies within ``reach_mm`` of the body's layer cells under
    or beside that footprint. The distance is given for every vertex so a
    renderer can split triangles along its zero level.
    """
    affine, shape = context['affine'], context['shape']
    inverse = np.linalg.inv(affine)
    ijk = points @ inverse[:3, :3].T + inverse[:3, 3]
    signed = body_footprint(context, body)
    edge = ndimage.map_coordinates(signed, ijk[:, :2].T, order=1, mode='nearest')
    mask = context['bodies'][body]['mask']
    layer = _role_mask(context, BODY_LAYER_ROLES[body]) | mask
    footprint = ndimage.binary_dilation(mask.any(axis=2), iterations=lateral_cells)
    k = np.nonzero(mask.any(axis=(0, 1)))[0]
    band = np.zeros(shape[2], bool)
    band[max(0, k.min() - 3):k.max() + 4] = True
    extended = layer & footprint[:, :, None] & band[None, None, :]
    spacing = float(np.linalg.norm(affine[:3, 0]))
    distance = ndimage.distance_transform_edt(~extended, sampling=spacing)
    cell = np.rint(ijk).astype(np.int64)
    inside = np.all((cell >= 0) & (cell < np.array(shape)), axis=1)
    near = np.zeros(len(points), bool)
    near[inside] = distance[tuple(cell[inside].T)] <= reach_mm
    flag = np.where(near & (edge > 0), BODY_CODES[body], 0).astype(np.uint8)
    return flag, edge.astype(np.float32), near


def body_outline(context, body, points, near, *, outermost=True, step_mm=.5):
    """Smooth closed outline of a body's footprint, lifted onto a surface (points near the body)."""
    from skimage.measure import find_contours
    signed = body_footprint(context, body)
    contours = find_contours(signed, 0.)
    if not contours or not near.any():
        return np.zeros((0, 3), np.float32), np.zeros(0, np.uint8)
    contour = max(contours, key=len)
    affine = context['affine']
    inverse = np.linalg.inv(affine)
    local = points[near] @ inverse[:3, :3].T + inverse[:3, 3]
    tree = cKDTree(local[:, :2])
    heights = []
    for radius in (1., 2., 4.):
        found = tree.query_ball_point(contour, r=radius)
        pick = np.max if outermost else np.min
        heights = [pick(local[f, 2]) if f else np.nan for f in found]
        if np.isfinite(heights).mean() > .9:
            break
    heights = np.asarray(heights, float)
    valid = np.isfinite(heights)
    if not valid.any():
        return np.zeros((0, 3), np.float32), np.zeros(0, np.uint8)
    index = np.arange(len(heights))
    heights = np.interp(index, index[valid], heights[valid], period=len(heights))
    path = np.column_stack([contour, heights])
    closed = np.allclose(contour[0], contour[-1])
    if closed:
        path = path[:-1]
    # Smooth along the curve and resample by arc length.
    mode = 'wrap' if closed else 'nearest'
    path = ndimage.gaussian_filter1d(path, 2., axis=0, mode=mode)
    world = path @ affine[:3, :3].T + affine[:3, 3]
    ring = np.vstack([world, world[:1]]) if closed else world
    arc = np.r_[0., np.cumsum(np.linalg.norm(np.diff(ring, axis=0), axis=1))]
    samples = np.arange(0., arc[-1], step_mm)
    resampled = np.column_stack([np.interp(samples, arc, ring[:, a]) for a in range(3)])
    hinge = context['bodies'][body]['hinge'].any(axis=2)
    flags = np.zeros(len(resampled), np.uint8)
    if hinge.any():
        ijk = resampled @ inverse[:3, :3].T + inverse[:3, 3]
        hinge_distance = ndimage.distance_transform_edt(~hinge)
        flags = (ndimage.map_coordinates(hinge_distance, ijk[:, :2].T, order=1, mode='nearest') <= 1.5
                 ).astype(np.uint8)
    return resampled.astype(np.float32), flags


# ----------------------------------------------------------------------------- source fields

def _field_from_mask(mask, affine, spacing, sigma_mm, grid_shape, grid_affine):
    """Smoothed occupancy of ``mask`` sampled on the grid (the source is cropped to the grid first)."""
    crop = _crop_bounds(affine, mask.shape, grid_shape, grid_affine, pad=4 + 4 * sigma_mm / min(spacing))
    sub = mask[crop].astype(np.float32)
    sub = ndimage.gaussian_filter(sub, [sigma_mm / s for s in spacing], mode='nearest')
    local = _translate(affine, [c.start for c in crop])
    return resample(sub, local, grid_shape, grid_affine, order=1), crop


def _constraint(mask, affine, spacing, crop):
    sub = mask[crop]
    distance = label_distance(sub, spacing)
    local = _translate(affine, [c.start for c in crop])
    return lambda points: sample_points(distance, local, points, cval=None)


def _surface(mask, affine, spacing, grid_shape, grid_affine, *, sigma, rounds, constraint,
             minimum_area_mm2=40., field=None, level=.5, taubin=False):
    if field is None:
        field, crop = _field_from_mask(mask, affine, spacing, sigma, grid_shape, grid_affine)
    else:
        crop = _crop_bounds(affine, mask.shape, grid_shape, grid_affine, pad=6)
    points, faces = isosurface(field, grid_affine, level)
    points, faces, dropped = drop_small_components(points, faces, minimum_area_mm2=minimum_area_mm2)
    if rounds and len(faces):
        points = constrained_taubin(points, faces, rounds=rounds,
                                    constraint=_constraint(mask, affine, spacing, crop) if constraint else None,
                                    limit=constraint, mu=-.53 if taubin else None)
        if not taubin:
            # A few unconstrained Taubin rounds remove the sub-millimetre dimples the constraint leaves.
            points = constrained_taubin(points, faces, rounds=FINISH_ROUNDS)
    normals = vertex_normals(points, faces) if len(faces) else np.zeros_like(points)
    return points, faces, normals, dropped


def head_masks(labels):
    """Source masks from the categorical head model, by surface."""
    occupied = ~np.isin(labels, list(OUTSIDE_IDS))
    head = ndimage.binary_fill_holes(occupied)
    brain = np.isin(labels, list(BRAIN_IDS))
    csf = np.isin(labels, list(CSF_IDS))
    vessel = np.isin(labels, [ARTERY_ID, VEIN_ID])
    intracranial = brain | csf
    # Vessels against the intracranial space (sinuses, cortical vessels) are inside the inner table.
    intracranial |= vessel & ndimage.binary_dilation(intracranial, iterations=2)
    intracranial = ndimage.binary_fill_holes(intracranial)
    bone = np.isin(labels, list(BONE_IDS))
    cranium = ndimage.binary_fill_holes(bone | intracranial)
    white = ndimage.binary_fill_holes(np.isin(labels, list(INNER_IDS)))
    return {'skin': head, 'skullOuter': cranium, 'skullInner': intracranial, 'pial': brain, 'white': white,
            'bone': bone}


def _skin_field_from_t1(t1, t1_affine, t1_spacing, head, head_affine, sigma, grid_shape, grid_affine):
    """Normalised T1 intensity within 3 mm of the head-model boundary; 1 deeper inside and 0 further out.

    Returns the field and the level: half-way between the median intensity 1 to
    3 mm inside the model boundary (scalp) and 1 to 3 mm outside it (air).
    """
    crop = _crop_bounds(t1_affine, t1.shape, grid_shape, grid_affine, pad=6)
    sub = ndimage.gaussian_filter(t1[crop].astype(np.float32), [sigma / s for s in t1_spacing], mode='nearest')
    local = _translate(t1_affine, [c.start for c in crop])
    positive = sub[sub > 0]
    scale = float(np.percentile(positive, 99.)) if positive.size else 1.
    image = resample(sub / max(scale, 1e-6), local, grid_shape, grid_affine, order=1)
    distance = label_distance(head, [float(v) for v in np.linalg.norm(head_affine[:3, :3], axis=0)])
    head_distance = resample(distance, head_affine, grid_shape, grid_affine, order=1, cval=50.)
    # Half-way between the scalp just inside the model boundary and the air just outside it.
    inner = (head_distance >= -3.) & (head_distance <= -1.)
    outer = (head_distance >= 1.) & (head_distance <= 3.)
    level = .5 * (float(np.median(image[inner])) + float(np.median(image[outer]))) if inner.any() and outer.any() else .1
    field = np.where(head_distance < -3., 1., np.where(head_distance > 3., 0., image)).astype(np.float32)
    return field, level


# ----------------------------------------------------------------------------- vessels and tracts

def _cortical_extension(mask, probability, crop, labels, label_spacing, intracranial_margin, sample):
    """Voxels the model scores at or above ``VESSEL_LOW_THRESHOLD`` (below its mask threshold) that lie on the
    cortical surface (``VESSEL_SURFACE_BAND_MM`` from the head-model brain boundary, positive outside) and are
    connected (26-neighbourhood) to the selected mask, within ``crop``. Hysteresis on the model's own score:
    nothing is added that the probability does not support or that does not join an existing vessel."""
    brain = np.isin(labels, list(BRAIN_IDS))
    signed = label_distance(brain, label_spacing)
    sub_mask = mask[crop]
    candidate = (probability[crop] >= VESSEL_LOW_THRESHOLD) & ~sub_mask
    cells = np.argwhere(candidate)
    full = cells + np.array([c.start for c in crop])
    lo, hi = VESSEL_SURFACE_BAND_MM
    depth = sample(full, signed, np.float32(1e3))
    keep = (depth >= lo) & (depth <= hi) & sample(full, intracranial_margin, False)
    candidate[:] = False
    candidate[tuple(cells[keep].T)] = True
    components, count = ndimage.label(sub_mask | candidate, np.ones((3, 3, 3)))
    joined = np.zeros(count + 1, bool)
    joined[np.unique(components[sub_mask])] = True
    joined[0] = False
    added_sub = candidate & joined[components]
    added = np.zeros(mask.shape, bool)
    added[crop] = added_sub
    return {'mask': added, 'voxels': int(added_sub.sum()), 'candidateVoxels': int(keep.sum()),
            'threshold': VESSEL_LOW_THRESHOLD, 'surfaceBandMm': list(VESSEL_SURFACE_BAND_MM),
            'method': ('hysteresis on the vessel model score: voxels scoring at least '
                       f'{VESSEL_LOW_THRESHOLD} (the mask is the 0.5 level) within {lo} to {hi} mm of the head-model '
                       'brain boundary, kept only where 26-connected to the selected mask')}


def vessel_assets(context, paths, *, check=lambda: None):
    """Centreline tubes of the vessel model in its own frame, exported in host RAS."""
    from . import vessel_tubes as tubes
    mask, vessel_affine, spacing = _load_nifti(paths['vesselMask'])
    mask = mask > 0
    probability = None
    if paths.get('vesselProbability'):
        probability, probability_affine, _ = _load_nifti(paths['vesselProbability'], np.float32)
        if probability.shape != mask.shape or not np.allclose(probability_affine, vessel_affine, atol=1e-4):
            raise ValueError('SOURCE_SURFACES_VESSEL_GRID_MISMATCH')
    forward = np.asarray(paths.get('vesselToHost', np.eye(4)), float)
    host_affine = forward @ vessel_affine
    raw_voxels = int(mask.sum())
    raw_components = int(ndimage.label(mask, np.ones((3, 3, 3)))[1])
    selection = 'raw vessel model mask'
    classes = None
    # Crop to the working box plus margin, in the vessel model's own grid.
    grid_shape, grid_affine = sampling_grid(context['affine'], context['shape'], step=1., margin=MARGIN_MM)
    crop = _crop_bounds(host_affine, mask.shape, grid_shape, grid_affine, pad=2)
    added = np.zeros(mask.shape, bool)
    extension = None
    if paths.get('headLabels'):
        labels, label_affine, label_spacing = _load_nifti(paths['headLabels'], np.int32)
        intracranial = np.isin(labels, list(BRAIN_IDS | CSF_IDS))
        margin = ndimage.distance_transform_edt(~intracranial, sampling=label_spacing) <= 2.
        inverse = np.linalg.inv(label_affine)

        def label_index(cells):
            world = cells @ host_affine[:3, :3].T + host_affine[:3, 3]
            index = np.rint(world @ inverse[:3, :3].T + inverse[:3, 3]).astype(int)
            return index, np.all((index >= 0) & (index < labels.shape), axis=1)

        def sample(volume, cells, fill):
            index, valid = label_index(cells)
            out = np.full(len(cells), fill, volume.dtype)
            out[valid] = volume[tuple(index[valid].T)]
            return out

        cells = np.argwhere(mask)
        mask = np.zeros_like(mask)
        mask[tuple(cells[sample(margin, cells, False)].T)] = True
        selection = 'vessel model mask within 2 mm of the head-model intracranial tissue and CSF'
        if probability is not None:
            extension = _cortical_extension(mask, probability, crop, labels, label_spacing, margin,
                                            lambda cells, volume, fill: sample(volume, cells, fill))
            added = extension.pop('mask')
            mask |= added
        # A vessel cell takes the head model's artery or vein class within 1.5 mm, the nearer one if both.
        cells = np.argwhere(mask)
        code = np.full(len(cells), 3, np.uint8)
        reach = {}
        for value, kind in ((1, ARTERY_ID), (2, VEIN_ID)):
            distance = ndimage.distance_transform_edt(labels != kind, sampling=label_spacing)
            reach[value] = sample(distance, cells, np.inf)
        code[(reach[1] <= 1.5) & (reach[1] <= reach[2])] = 1
        code[(reach[2] <= 1.5) & (reach[2] < reach[1])] = 2
        classes = np.zeros(mask.shape, np.uint8)
        classes[tuple(cells.T)] = code
    sub = mask[crop]
    if not sub.any():
        return None, {}
    local = _translate(vessel_affine, [c.start for c in crop])
    model = tubes.extract(sub, local, probability=None if probability is None else probability[crop],
                          classes=None if classes is None else classes[crop], check=check)
    wall = tubes.mesh(model, subdivision=1, check=check)
    rotation = forward[:3, :3]
    to_host = lambda p: np.asarray(p, float) @ rotation.T + forward[:3, 3]
    points = to_host(model.points)
    offsets, polyline, radii, kinds, support = [0], [], [], [], []
    added_sub = added[crop]
    local_inverse = np.linalg.inv(local)
    for samples, edge_radius, kind in zip(model.edges, model.edge_radii, model.edge_class):
        polyline.append(points[samples])
        radii.append(np.asarray(edge_radius, float))
        offsets.append(offsets[-1] + len(samples))
        kinds.append({'artery': 1, 'vein': 2}.get(kind, 0))
        # 1: most of the centreline runs through voxels below the mask threshold (the cortical extension).
        ijk = np.rint(np.asarray(model.points)[samples] @ local_inverse[:3, :3].T + local_inverse[:3, 3]).astype(int)
        valid = np.all((ijk >= 0) & (ijk < added_sub.shape), axis=1)
        fraction = float(added_sub[tuple(ijk[valid].T)].mean()) if valid.any() else 0.
        support.append(1 if fraction > .5 else 0)
    polyline = np.concatenate(polyline)
    radii = np.concatenate(radii)
    inverse = np.linalg.inv(context['affine'])
    lengths, inside_length = [], 0.
    box = np.array(context['shape']) - 1
    segment_inside = []
    for s in range(len(offsets) - 1):
        run = polyline[offsets[s]:offsets[s + 1]]
        if len(run) < 2:
            lengths.append(0.)
            mid = run.mean(axis=0, keepdims=True)
        else:
            steps = np.linalg.norm(np.diff(run, axis=0), axis=1)
            lengths.append(float(steps.sum()))
            mid = (run[1:] + run[:-1]) / 2
            ijk = mid @ inverse[:3, :3].T + inverse[:3, 3]
            inside_length += float(steps[np.all((ijk >= -.5) & (ijk <= box + .5), axis=1)].sum())
        ijk = mid @ inverse[:3, :3].T + inverse[:3, 3]
        segment_inside.append(bool(np.all((ijk >= -.5) & (ijk <= box + .5), axis=1).any()))
    wall_points = to_host(wall.positions)
    wall_normals = wall.normals @ rotation.T
    wall_cells, _ = bind_cells(wall_points, context, ('angiography_vessel', 'artery', 'vein'), reach_mm=0.)
    edge_count = len(model.edges)
    owners = np.where(wall.owners < edge_count, wall.owners, edge_count).astype(np.uint32)
    arrays = {
        'vessels.centrelinePoints': polyline.astype(np.float32),
        'vessels.centrelineRadii': radii.astype(np.float32),
        'vessels.centrelineOffsets': np.asarray(offsets, np.uint32),
        'vessels.segmentKind': np.asarray(kinds, np.uint8),
        'vessels.segmentInsideBox': np.asarray(segment_inside, np.uint8),
        'vessels.segmentSupport': np.asarray(support, np.uint8),
        'vessels.wallPositions': wall_points.astype(np.float32),
        'vessels.wallNormals': wall_normals.astype(np.float32),
        'vessels.wallTriangles': wall.faces.astype(np.uint32),
        'vessels.wallSegment': owners,
        'vessels.wallCells': wall_cells,
    }
    kind_length = {}
    for kind, length in zip(kinds, lengths):
        kind_length[VESSEL_CLASS[kind]] = kind_length.get(VESSEL_CLASS[kind], 0.) + length
    header = {
        'model': model.receipt.get('model'), 'source': str(paths['vesselMask']),
        'sha256': sha256_file(paths['vesselMask']),
        'probability': None if not paths.get('vesselProbability') else {
            'source': str(paths['vesselProbability']), 'sha256': sha256_file(paths['vesselProbability']),
            'meaning': 'uncalibrated model score used as the soft wall signal'},
        'resolutionMm': spacing, 'origin': 'estimated',
        'frame': 'tubes fitted in the vessel model grid, then mapped rigidly into host RAS',
        'vesselToHost': forward.tolist(), 'selection': selection,
        'classes': ('per segment, the majority over its centreline cells of the head-model artery or vein label within '
                    '1.5 mm (the nearer if both), else unknown'),
        'coverage': {
            'rawVoxels': raw_voxels, 'rawComponents': raw_components, 'selectedVoxelsInBox': int(sub.sum()),
            'segments': len(kinds), 'graphComponents': int(model.receipt['graphBetti'][0]),
            'centrelineMm': round(float(sum(lengths)), 1), 'centrelineInsideBoxMm': round(inside_length, 1),
            'centrelineMmByKind': {k: round(v, 1) for k, v in sorted(kind_length.items())},
            'radiusMm': {'median': round(float(np.median(radii)), 3), 'p95': round(float(np.percentile(radii, 95)), 3)},
            'addedCentrelineMm': round(float(sum(l for l, s in zip(lengths, support) if s)), 1),
            'addedSegments': int(sum(support)),
        },
        'corticalExtension': extension,
        'segmentSupport': ('per segment: 0 the model mask (score at least 0.5), 1 the connected cortical extension '
                           '(most of its centreline in voxels scoring below 0.5)'),
        'wall': {'vertices': int(len(wall_points)), 'triangles': int(len(wall.faces)),
                 'gridStepMm': wall.receipt.get('gridStepMm'),
                 'segmentIndex': 'owning centreline segment; the segment count means a junction ball'},
        'complete': False,
        'limitations': ['vessel model output is an uncalibrated estimate; absent vessels are not proven absent',
                        'MRA coverage is partial; rigid registration is unreviewed'],
    }
    return header, arrays


def tract_assets(context, paths, *, limit=TRACT_STREAMLINES, step_mm=TRACT_STEP_MM, seed=20261127):
    import nibabel as nib
    rows, arrays = [], {}
    names = {context['roleOf'][name]: name for name in context['names']}
    for entry in paths.get('tracts') or []:
        bundle = entry['bundle']
        loaded = nib.streamlines.load(str(entry['path']))
        streamlines = list(loaded.streamlines)
        forward = np.asarray(entry.get('toHost', np.eye(4)), float)
        rng = np.random.default_rng(seed)
        order = rng.permutation(len(streamlines))[:limit] if len(streamlines) > limit else np.arange(len(streamlines))
        points, offsets = [], [0]
        for index in sorted(order):
            line = np.asarray(streamlines[index], float) @ forward[:3, :3].T + forward[:3, 3]
            if len(line) < 2:
                continue
            arc = np.r_[0., np.cumsum(np.linalg.norm(np.diff(line, axis=0), axis=1))]
            samples = np.linspace(0., arc[-1], max(2, int(arc[-1] // step_mm) + 1))
            points.append(np.column_stack([np.interp(samples, arc, line[:, a]) for a in range(3)]))
            offsets.append(offsets[-1] + len(points[-1]))
        if not points:
            continue
        key = 'tracts.' + bundle
        arrays[key + '.points'] = np.concatenate(points).astype(np.float32)
        arrays[key + '.offsets'] = np.asarray(offsets, np.uint32)
        rows.append({'bundle': bundle, 'layer': names.get(bundle), 'source': str(entry['path']),
                     'sha256': sha256_file(entry['path']), 'streamlines': len(streamlines),
                     'exported': len(offsets) - 1, 'stepMm': step_mm, 'origin': 'estimated',
                     'method': 'published reconstruction, rigidly registered (unreviewed); decimated for display',
                     'arrays': {'points': key + '.points', 'offsets': key + '.offsets'}})
    return rows, arrays


# ----------------------------------------------------------------------------- build

def build_assets(operative_case, host_paths, *, check=lambda: None, log=lambda message: None):
    """Source-resolution surfaces of an operative case. Returns ``{'header': ..., 'arrays': ...}``.

    ``host_paths`` names the sources: ``headLabels`` (categorical head model,
    required), ``t1`` (acquired T1 for the skin; optional), ``tumourLabels`` with
    ``tumourCodes`` ({'tumour_core': 1, 'tumour_rim': 2}), ``vesselMask`` and
    ``vesselProbability`` with ``vesselToHost`` (4x4 RAS map into the host),
    ``tracts`` ([{'bundle', 'path', 'toHost'}]) and ``hostStudy`` (to verify the
    host case hash). All sources must share the host RAS world or carry their map.
    """
    started = time.perf_counter()
    context = operative_context(operative_case)
    declaration = context['declaration']
    affine, shape = context['affine'], context['shape']
    host_verified = False
    if host_paths.get('hostStudy'):
        with zipfile.ZipFile(host_paths['hostStudy']) as archive:
            manifest = json.loads(archive.read('manifest.json'))
        if manifest.get('case_semantic_hash') != declaration['hostCaseHash']:
            raise ValueError('SOURCE_SURFACES_HOST_MISMATCH: the host study is not the rehearsal host')
        host_verified = True
    labels, label_affine, label_spacing = _load_nifti(host_paths['headLabels'], np.int32)
    label_sha = sha256_file(host_paths['headLabels'])
    masks = head_masks(labels)
    frame_check = _frame_agreement(context, labels, label_affine)
    if frame_check['brainAgreement'] < .7:
        raise ValueError('SOURCE_SURFACES_FRAME_MISMATCH: the head model does not overlie the working grid')
    arrays, surfaces = {}, []

    def add(name, points, faces, normals, provenance, *, mask_for_deviation, deviation_affine, extra=None):
        roles = dict((n, r) for n, r, _ in SURFACES)[name]
        body = dict((n, b) for n, _, b in SURFACES)[name]
        # Bind the positions as stored (float32), so a reader rounding them finds the same cells.
        points = np.asarray(points, np.float32).astype(np.float64)
        cells, owners = bind_cells(points, context, roles)
        key = 'surface.' + name
        arrays[key + '.positions'] = points.astype(np.float32)
        arrays[key + '.normals'] = normals.astype(np.float32)
        arrays[key + '.triangles'] = faces.astype(np.uint32)
        arrays[key + '.cells'] = cells
        arrays[key + '.ownerCells'] = owners
        row = {'name': name, 'layers': layer_names(context, roles), 'roles': list(roles), 'body': body,
               'vertices': int(len(points)), 'triangles': int(len(faces)), 'provenance': provenance,
               'quality': mesh_quality(points, faces, normals),
               'labelDeviation': label_deviation(points, mask_for_deviation, deviation_affine),
               'arrays': {'positions': key + '.positions', 'normals': key + '.normals',
                          'triangles': key + '.triangles', 'cells': key + '.cells', 'ownerCells': key + '.ownerCells'}}
        if body is not None:
            flag, edge, near = bind_body(points, context, body)
            arrays[key + '.body'] = flag
            arrays[key + '.bodyEdgeMm'] = edge
            row['arrays'].update(body=key + '.body', bodyEdgeMm=key + '.bodyEdgeMm')
            row['bodyVertices'] = int(np.count_nonzero(flag))
            row['_near'] = near
        row.update(extra or {})
        surfaces.append(row)
        log(f'{name}: {len(points)} vertices, {len(faces)} triangles')
        check()
        return row

    head_source = {'source': str(host_paths['headLabels']), 'sha256': label_sha, 'resolutionMm': label_spacing}
    meshes = {}
    for name in ('skin', 'skullOuter', 'skullInner', 'pial', 'white'):
        settings = dict(DEFAULTS[name])
        grid_shape, grid_affine = sampling_grid(affine, shape, step=settings['step'])
        mask = masks[name]
        if name == 'skin' and host_paths.get('t1'):
            t1, t1_affine, t1_spacing = _load_nifti(host_paths['t1'], np.float32)
            # The whole head the T1 shows, so an overview is a complete head; vertices outside the
            # working grid are bound to no engine cell.
            grid_shape, grid_affine = covering_grid(affine, shape, step=settings['step'], mask=mask,
                                                    mask_affine=label_affine, limit=(t1_affine, t1.shape))
            field, level = _skin_field_from_t1(t1, t1_affine, t1_spacing, mask, label_affine, settings['sigma'],
                                               grid_shape, grid_affine)
            points, faces, normals, dropped = _surface(mask, label_affine, label_spacing, grid_shape, grid_affine,
                                                       sigma=settings['sigma'], rounds=settings['rounds'],
                                                       constraint=settings['constraint'], field=field, level=level,
                                                       minimum_area_mm2=200., taubin=True)
            provenance = {'source': str(host_paths['t1']), 'sha256': sha256_file(host_paths['t1']),
                          'resolutionMm': t1_spacing, 'smoothingSigmaMm': settings['sigma'], 'origin': 'acquired',
                          'method': ('isointensity of the acquired T1 half-way between scalp and air intensities, '
                                     'inside a 3 mm shell of the head-model skin boundary; Taubin relaxed and held '
                                     f'within {settings["constraint"]} mm of the head-model boundary'),
                          'extent': ('the whole head in the T1 field of view (head model plus '
                                     f'{SKIN_HEAD_MARGIN_MM} mm); only vertices inside the working grid are bound to '
                                     'engine cells'),
                          'level': round(level, 4), 'gridStepMm': settings['step'], 'smoothingRounds': settings['rounds'],
                          'constraintMm': settings['constraint'], 'droppedPieces': dropped, 'guide': head_source}
        else:
            minimum = 200. if name in ('skin', 'skullOuter', 'skullInner') else 40.
            closing = None
            if name in ('skullOuter', 'skullInner'):
                closed = ndimage.binary_closing(mask, structure=ndimage.generate_binary_structure(3, 1), iterations=2)
                closed = ndimage.binary_fill_holes(closed)
                closing = int(np.count_nonzero(closed & ~mask))
                mask = closed
            points, faces, normals, dropped = _surface(mask, label_affine, label_spacing, grid_shape, grid_affine,
                                                       sigma=settings['sigma'], rounds=settings['rounds'],
                                                       constraint=settings['constraint'], minimum_area_mm2=minimum)
            what = {'skin': 'outer boundary of all head-model tissue (holes filled)',
                    'skullOuter': 'outer table: boundary of bone and the intracranial space together',
                    'skullInner': 'inner table: boundary of the intracranial space (brain, CSF, adjoining vessels)',
                    'pial': 'outer boundary of the brain parenchyma classes (cortical grey matter outer boundary)',
                    'white': 'boundary of cerebral white matter with the enclosed deep nuclei and ventricles'}[name]
            provenance = {**head_source, 'smoothingSigmaMm': settings['sigma'], 'origin': 'estimated',
                          'method': (what + '; Gaussian partial-volume field at level 0.5, Laplacian relaxed and held '
                                     f'within {settings["constraint"]} mm of the label boundary'),
                          'gridStepMm': settings['step'], 'smoothingRounds': settings['rounds'],
                          'constraintMm': settings['constraint'], 'droppedPieces': dropped}
            if name.startswith('skull'):
                provenance['estimate'] = 'skull from a T1 model estimate (SuperSynth), not CT-measured bone'
                provenance['gapsClosed'] = {'voxels': closing, 'operation': 'binary closing, 2 face steps, holes filled'}
        meshes[name] = (points, faces, normals)
        add(name, points, faces, normals, provenance, mask_for_deviation=masks[name], deviation_affine=label_affine)
        del points, faces, normals

    # The assumed dural sheet: the inner table around the craniotomy, offset inward.
    inner_points, inner_faces, inner_normals = meshes['skullInner']
    dura_points, dura_faces, dura_receipt = _dural_sheet(context, inner_points, inner_faces, inner_normals)
    dura_normals = vertex_normals(dura_points, dura_faces) if len(dura_faces) else np.zeros_like(dura_points)
    add('dura', dura_points, dura_faces, dura_normals,
        {**head_source, 'origin': 'assumed', 'smoothingSigmaMm': DEFAULTS['skullInner']['sigma'],
         'method': (f'one continuous sheet lining the near inner table within {DURA_REACH_MM} mm of the bone and '
                    f'dural flap footprints, interior holes closed, outer face {DURA_OFFSET_MM} mm inside the '
                    'inner table; no dura is imaged or labelled'),
         'thicknessMm': DURA_THICKNESS_MM, 'thickness': 'assumed membrane thickness (0.5 to 0.8 mm in adults)',
         'sheet': dura_receipt,
         'flap': 'split by the dura body footprint outline (bodyEdgeMm zero level), hinge from the engine body'},
        mask_for_deviation=masks['skullInner'], deviation_affine=label_affine)

    if host_paths.get('tumourLabels'):
        tumour, tumour_affine, tumour_spacing = _load_nifti(host_paths['tumourLabels'], np.int32)
        codes = host_paths.get('tumourCodes') or {'tumour_core': 1, 'tumour_rim': 2}
        tumour_sha = sha256_file(host_paths['tumourLabels'])
        for name, members in (('tumourRim', (codes['tumour_core'], codes['tumour_rim'])),
                              ('tumourCore', (codes['tumour_core'],))):
            mask = np.isin(tumour, members)
            if not mask.any():
                continue
            settings = DEFAULTS[name]
            grid_shape, grid_affine = sampling_grid(affine, shape, step=settings['step'])
            points, faces, normals, dropped = _surface(mask, tumour_affine, tumour_spacing, grid_shape, grid_affine,
                                                       sigma=settings['sigma'], rounds=settings['rounds'],
                                                       constraint=settings['constraint'], minimum_area_mm2=5.)
            add(name, points, faces, normals,
                {'source': str(host_paths['tumourLabels']), 'sha256': tumour_sha, 'resolutionMm': tumour_spacing,
                 'smoothingSigmaMm': settings['sigma'], 'origin': 'generated',
                 'method': ('generated lesion mask (' + ('core and rim' if name == 'tumourRim' else 'core') +
                            '), Gaussian field at level 0.5, Laplacian relaxed and held within '
                            f'{settings["constraint"]} mm of the mask boundary'),
                 'gridStepMm': settings['step'], 'smoothingRounds': settings['rounds'],
                 'constraintMm': settings['constraint'], 'droppedPieces': dropped},
                mask_for_deviation=mask, deviation_affine=tumour_affine)

    # Smooth cut outlines of each body, on the surfaces the cut passes through.
    outlines = {}
    by_name = {row['name']: row for row in surfaces}
    for body, outer, inner in (('scalp', 'skin', 'skullOuter'), ('bone', 'skullOuter', 'skullInner'),
                               ('dura', 'dura', None)):
        entry = {}
        for side, surface in (('outer', outer), ('inner', inner)):
            if surface is None or surface not in by_name:
                continue
            row = by_name[surface]
            positions = arrays['surface.' + surface + '.positions'].astype(float)
            if surface == 'skullOuter' and body == 'scalp':
                # The deep face of the scalp flap rests on the outer table under its footprint.
                near = bind_body(positions, context, 'scalp', reach_mm=12.)[2]
            else:
                near = row['_near']
            points, hinge = body_outline(context, body, positions, near, outermost=True)
            key = f'outline.{body}.{side}'
            arrays[key + '.points'] = points
            arrays[key + '.hinge'] = hinge
            entry[side] = {'surface': surface, 'points': key + '.points', 'hinge': key + '.hinge',
                           'lengthMm': round(float(np.linalg.norm(np.diff(points, axis=0), axis=1).sum()), 1)
                           if len(points) > 1 else 0.}
        outlines[body] = {**entry, 'closed': True,
                          'method': ('zero level of the Gaussian-smoothed (1.5 cells) signed in-plane distance to the '
                                     'body footprint columns, lifted onto the surface and smoothed along the curve'),
                          'hingeFlag': 'points within 1.5 cells of a hinge column (the uncut base) are flagged 1'}
    for row in surfaces:
        row.pop('_near', None)

    vessels = None
    if host_paths.get('vesselMask'):
        log('vessels: extracting centreline tubes')
        vessels, vessel_arrays = vessel_assets(context, host_paths, check=check)
        arrays.update(vessel_arrays)
    tracts, tract_arrays = tract_assets(context, host_paths)
    arrays.update(tract_arrays)

    header = {
        'schema': SCHEMA, 'scope': SCOPE,
        'operativeCaseId': operative_case.case_id, 'operativeCaseHash': operative_case.semantic_hash,
        'hostCaseId': declaration['hostCaseId'], 'hostCaseHash': declaration['hostCaseHash'],
        'hostVerified': host_verified,
        'frame': 'RAS+ millimetres (the host world); positions, normals and polylines',
        'workingGrid': {'shape': list(shape), 'affine': affine.tolist(),
                        'spacingMm': float(np.linalg.norm(affine[:3, 0]))},
        'marginMm': MARGIN_MM, 'frameCheck': frame_check,
        'binding': {
            'cells': 'flat C-order index of the working cell containing the vertex, or -1 outside the grid',
            'ownerCells': 'nearest working cell (within 3 mm) owned by one of the surface layers, or -1',
            'body': 'engine body the vertex moves with: 0 none, 1 scalp, 2 bone, 3 dura',
            'bodyEdgeMm': ('signed in-plane distance to the smooth outline of the surface\'s body, positive inside; '
                           'its zero level is the cut'),
            'normals': 'unit, outward from the enclosed tissue',
        },
        'surfaces': surfaces, 'outlines': outlines, 'vessels': vessels,
        'tracts': tracts if tracts else None,
        'tractsSkipped': None if tracts else 'no published streamlines were supplied for this subject',
        'build': {'seconds': None, 'module': 'resectionlab.surgical_source_surfaces'},
    }
    header['build']['seconds'] = round(time.perf_counter() - started, 1)
    return {'header': header, 'arrays': arrays}


def _frame_agreement(context, labels, label_affine):
    """Fraction of working brain and bone cells whose head-model label (nearest) is brain or bone."""
    affine, shape = context['affine'], context['shape']
    result = {}
    for key, roles, ids in (('brainAgreement', ('cortex', 'white', 'deep'), BRAIN_IDS),
                            ('boneAgreement', ('cortical_bone', 'cancellous_bone'), BONE_IDS)):
        mask = _role_mask(context, roles)
        cells = np.argwhere(mask)
        if not len(cells):
            result[key] = None
            continue
        cells = cells[np.random.default_rng(1).choice(len(cells), min(20000, len(cells)), replace=False)]
        world = cells @ affine[:3, :3].T + affine[:3, 3]
        inverse = np.linalg.inv(label_affine)
        index = np.rint(world @ inverse[:3, :3].T + inverse[:3, 3]).astype(int)
        valid = np.all((index >= 0) & (index < labels.shape), axis=1)
        found = np.zeros(len(cells), bool)
        found[valid] = np.isin(labels[tuple(index[valid].T)], list(ids))
        result[key] = round(float(found.mean()), 4)
    result['brainAgreement'] = result['brainAgreement'] or 0.
    return result


def boundary_loops(faces):
    """Closed boundary loops of a triangle mesh, as vertex lists in boundary-edge order."""
    edges, uses = _unique_edges(faces)
    single = {tuple(edge) for edge in edges[uses == 1]}
    following = {}
    for tri in faces:
        for a, b in ((tri[0], tri[1]), (tri[1], tri[2]), (tri[2], tri[0])):
            if (min(a, b), max(a, b)) in single:
                following[int(a)] = int(b)
    loops, seen = [], set()
    for start in following:
        if start in seen:
            continue
        loop, vertex = [], start
        while vertex is not None and vertex not in seen:
            seen.add(vertex)
            loop.append(vertex)
            vertex = following.get(vertex)
        if vertex == start and len(loop) >= 3:
            loops.append(loop)
    return loops


def fill_interior_holes(points, faces):
    """Close every boundary loop but the longest with a fan about its centroid; returns points, faces, holes closed."""
    loops = boundary_loops(faces)
    if len(loops) <= 1:
        return points, faces, 0
    perimeter = [float(np.linalg.norm(np.diff(points[loop + loop[:1]], axis=0), axis=1).sum()) for loop in loops]
    outer = int(np.argmax(perimeter))
    new_points, new_faces = [points], [faces]
    count = len(points)
    for index, loop in enumerate(loops):
        if index == outer:
            continue
        new_points.append(points[loop].mean(axis=0, keepdims=True))
        # A boundary edge a→b of a kept triangle; the hole is closed by b→a→centre.
        ring = np.asarray(loop + loop[:1])
        new_faces.append(np.column_stack([ring[1:], ring[:-1], np.full(len(loop), count)]))
        count += 1
    return np.vstack(new_points), np.vstack(new_faces).astype(np.int64), len(loops) - 1


def _dural_sheet(context, points, faces, normals):
    """The assumed dura: one continuous sheet lining the near inner table over the craniotomy and a margin.

    Inner-table faces whose columns lie within ``DURA_REACH_MM`` of the bone and dural flap footprints,
    that face outward (the near table) and lie at the depth of the flaps, joined into the largest piece
    with every interior hole closed, relaxed a few rounds and moved ``DURA_OFFSET_MM`` inward. Its
    flap is split by the dura body's footprint outline (``bodyEdgeMm``), the same outline the engine's
    dura body was cut along.
    """
    empty = np.zeros((0, 3)), np.zeros((0, 3), np.int64), {}
    if not len(faces):
        return empty
    affine, shape = context['affine'], context['shape']
    region = context['bodies']['dura']['mask'] | context['bodies']['bone']['mask']
    if not region.any():
        return empty
    columns = region.any(axis=2)
    spacing = float(np.linalg.norm(affine[:3, 0]))
    lateral = ndimage.distance_transform_edt(~columns, sampling=spacing)
    inverse = np.linalg.inv(affine)
    ijk = points @ inverse[:3, :3].T + inverse[:3, 3]
    within = ndimage.map_coordinates(lateral, ijk[:, :2].T, order=1, mode='nearest') <= DURA_REACH_MM
    # The near table faces out of the head (the grid's third axis points outward at the entry).
    outward = np.array(context["exposure"].get("exteriorNormalRas") or affine[:3, 2], float)
    outward /= np.linalg.norm(outward)
    facing = normals @ outward > .2
    k = np.nonzero(region.any(axis=(0, 1)))[0]
    depth = (ijk[:, 2] >= k.min() - 20. / spacing) & (ijk[:, 2] <= k.max() + 10. / spacing)
    keep = within & facing & depth
    chosen = faces[keep[faces].all(axis=1)]
    if not len(chosen):
        return empty
    used = np.zeros(len(points), bool)
    used[chosen.ravel()] = True
    index = np.cumsum(used) - 1
    sheet, sheet_faces = points[used], index[chosen]
    # The largest piece only (a stray patch of the opposite wall of a sulcus-like fold is not dura).
    count, labels = connected_components(sparse.coo_matrix(
        (np.ones(2 * len(sheet_faces)), (sheet_faces[:, :2].ravel(), sheet_faces[:, 1:].ravel())),
        shape=(len(sheet),) * 2), directed=False)
    piece = labels[sheet_faces[:, 0]]
    largest = np.bincount(piece, minlength=count).argmax()
    sheet_faces = sheet_faces[piece == largest]
    used = np.zeros(len(sheet), bool)
    used[sheet_faces.ravel()] = True
    index = np.cumsum(used) - 1
    sheet, sheet_faces = sheet[used], index[sheet_faces]
    sheet, sheet_faces, closed = fill_interior_holes(sheet, sheet_faces)
    # Relax the closures and the selection's jagged rim into the sheet (the rim stays pinned).
    sheet = constrained_taubin(sheet, sheet_faces, rounds=6)
    sheet_normals = vertex_normals(sheet, sheet_faces)
    sheet = sheet - DURA_OFFSET_MM * sheet_normals
    receipt = {'holesClosed': int(closed), 'boundaryLoops': len(boundary_loops(sheet_faces)),
               'reachMm': DURA_REACH_MM, 'offsetMm': DURA_OFFSET_MM, 'thicknessMm': DURA_THICKNESS_MM}
    return sheet, sheet_faces, receipt


# ----------------------------------------------------------------------------- asset file

def _array_digest(array):
    contiguous = np.ascontiguousarray(array)
    return hashlib.sha256(memoryview(contiguous.reshape(-1)).cast('B')).hexdigest()


def write_asset(asset, path):
    """Write the header and arrays as one compressed ``.npz``; the header indexes every array."""
    path = Path(path)
    header = json.loads(json.dumps(asset['header'], allow_nan=False))
    arrays = {key: np.ascontiguousarray(value) for key, value in asset['arrays'].items()}
    header['arrays'] = {key: {'dtype': value.dtype.str, 'shape': list(value.shape), 'sha256': _array_digest(value)}
                        for key, value in sorted(arrays.items())}
    payload = {HEADER_KEY: np.frombuffer(json.dumps(header, sort_keys=True, separators=(',', ':')).encode(), np.uint8),
               **arrays}
    path.parent.mkdir(parents=True, exist_ok=True)
    handle, temporary = tempfile.mkstemp(prefix='.source-surfaces-', suffix='.npz', dir=path.parent)
    os.close(handle)
    try:
        np.savez_compressed(temporary, **payload)
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)
    return header


def read_asset(path, *, case_hash=None, host_case_hash=None, verify=True):
    """Read and check an asset; refuses another case, another host, a wrong schema or a changed array."""
    with np.load(str(path), allow_pickle=False) as data:
        if HEADER_KEY not in data.files:
            raise ValueError('SOURCE_SURFACES_INVALID: no header')
        header = json.loads(bytes(data[HEADER_KEY]).decode())
        if header.get('schema') != SCHEMA:
            raise ValueError('SOURCE_SURFACES_SCHEMA: expected ' + SCHEMA)
        if case_hash is not None and header.get('operativeCaseHash') != case_hash:
            raise ValueError('SOURCE_SURFACES_CASE_MISMATCH: the asset belongs to another operative case')
        if host_case_hash is not None and header.get('hostCaseHash') != host_case_hash:
            raise ValueError('SOURCE_SURFACES_HOST_MISMATCH: the asset belongs to another host study')
        index = header.get('arrays') or {}
        if set(index) != set(data.files) - {HEADER_KEY}:
            raise ValueError('SOURCE_SURFACES_INVALID: array index differs from the file')
        arrays = {}
        for key, row in index.items():
            value = data[key]
            if value.dtype.str != row['dtype'] or list(value.shape) != row['shape']:
                raise ValueError('SOURCE_SURFACES_INVALID: array ' + key + ' changed shape or type')
            if verify and _array_digest(value) != row['sha256']:
                raise ValueError('SOURCE_SURFACES_INVALID: array ' + key + ' changed')
            arrays[key] = value
    return {'header': header, 'arrays': arrays}


# ----------------------------------------------------------------------------- EDEN2020 discovery

def eden_host_paths(operative_case, host_study, data_root):
    """Source paths of an EDEN2020 v2 host from the operative case's carried host record."""
    from .core import thaw_json
    metadata = thaw_json(operative_case.metadata)
    data = metadata.get('data_case') or {}
    root = Path(data_root)
    subject = data.get('source_subject')
    if not subject or not data.get('head_model_receipt'):
        raise ValueError('SOURCE_SURFACES_HOST_SOURCES_UNKNOWN: the case carries no EDEN2020 host record')
    receipt = root / data['head_model_receipt']
    paths = {'hostStudy': str(host_study), 'headLabels': str(receipt.parent / 'labels_1mm.nii.gz')}
    for ref in operative_case.source_refs:
        if ref.source_id == 'structural' and ref.uri.startswith('file://'):
            from urllib.parse import unquote, urlparse
            paths['t1'] = unquote(urlparse(ref.uri).path)
    registrations = data.get('registrations') or {}
    vessel_receipt = root / 'prepared/eden2020-anatomy-v2/subjects' / subject / 'vesselfm-0p6mm/inference.json'
    if vessel_receipt.exists() and 'MRA' in registrations:
        record = json.loads(vessel_receipt.read_text())
        paths['vesselMask'] = next(row['path'] for row in record['outputs'] if row['name'].endswith('_mask.nii.gz'))
        if (record.get('probability') or {}).get('path'):
            paths['vesselProbability'] = record['probability']['path']
        paths['vesselToHost'] = registrations['MRA']['moving_RAS_to_fixed_RAS_forward']
    host_id = declaration_host_id(metadata)
    tumour = root / 'prepared/eden2020-anatomy-v2/labels' / (host_id + '.nii.gz') if host_id else None
    if tumour is not None and tumour.exists():
        paths['tumourLabels'] = str(tumour)
        paths['tumourCodes'] = {'tumour_core': 1, 'tumour_rim': 2}
    if 'FLAIR' in registrations:
        forward = registrations['FLAIR']['moving_RAS_to_fixed_RAS_forward']
        paths['tracts'] = [{'bundle': Path(row['tract_only_source']).name.split(subject + '_', 1)[1].split('_to_')[0],
                            'path': row['tract_only_source'], 'toHost': forward}
                           for row in data.get('published_tracts') or [] if row.get('tract_only_source')]
    missing = [key for key, value in paths.items()
               if isinstance(value, str) and key not in ('hostStudy',) and not Path(value).exists()]
    if missing:
        raise ValueError('SOURCE_SURFACES_SOURCE_MISSING: ' + ', '.join(sorted(missing)))
    return paths


def declaration_host_id(metadata):
    from .surgical_host_rehearsal import PROPERTY
    return ((metadata.get('surgical_properties') or {}).get(PROPERTY) or {}).get('hostCaseId')


# ----------------------------------------------------------------------------- desktop bridge

def bridge_result(case, *, encode, case_path=None, asset_path=None, check=lambda: None):
    """The ``operativeSourceSurfaces`` bridge result for an open case: the header and every array encoded.

    The asset is ``asset_path``, or ``ASSET_NAME`` beside ``case_path``. A
    missing, foreign or malformed asset gives ``status: 'unavailable'`` with a
    code and reason; nothing is built or written here.
    """
    from .core import thaw_json
    from .surgical_host_rehearsal import PROPERTY

    def unavailable(code, reason):
        return {'caseHash': case.semantic_hash, 'status': 'unavailable', 'code': code, 'reason': reason}

    declared = ((thaw_json(case.metadata).get('surgical_properties') or {}).get(PROPERTY) or {})
    if not declared.get('hostCaseHash'):
        return unavailable('NOT_HOST_REHEARSAL', 'Source surfaces exist only for host rehearsal cases')
    if asset_path is not None:
        path = Path(asset_path)
        if not path.name.lower().endswith('.npz'):
            return unavailable('UNSUPPORTED_FILE_TYPE', 'The source-surface asset is an .npz file')
    elif case_path is not None:
        path = Path(case_path).parent / ASSET_NAME
    else:
        return unavailable('ASSET_PATH_REQUIRED', 'Name the case file or the source-surface asset')
    if path.is_symlink() or not path.is_file():
        return unavailable('ASSET_NOT_FOUND', f'No source-surface asset at {path.name} for this case')
    if path.stat().st_size > 1024 ** 3:
        return unavailable('ASSET_TOO_LARGE', 'The source-surface asset exceeds 1 GiB')
    try:
        asset = read_asset(path, case_hash=case.semantic_hash, host_case_hash=declared['hostCaseHash'])
    except (ValueError, OSError, zipfile.BadZipFile) as error:
        code = str(error).split(':', 1)[0] if str(error).startswith('SOURCE_SURFACES_') else 'SOURCE_SURFACES_INVALID'
        return unavailable(code, str(error))
    grid = asset['header'].get('workingGrid') or {}
    if (list(grid.get('shape') or []) != list(case.mri.shape)
            or not np.allclose(np.asarray(grid.get('affine'), float), _case_affine(case), atol=1e-6)):
        return unavailable('SOURCE_SURFACES_GRID_MISMATCH', 'The asset was built on another working grid')
    check()
    arrays = {}
    for key, value in sorted(asset['arrays'].items()):
        check()
        arrays[key] = encode(value, value.dtype.name)
    return {'caseHash': case.semantic_hash, 'status': 'available', 'assetName': path.name,
            'header': asset['header'], 'arrays': arrays}
