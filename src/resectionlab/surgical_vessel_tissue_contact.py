"""Local, frictionless exchange between native vessel walls and soft lattice cells."""

from dataclasses import dataclass

import numpy as np
from scipy import sparse

from . import surgical_vessel_surface as surfaces
from . import surgical_wall_sweep as wall_sweep
from .surgical_glioma_materials import SOFT, material_values


@dataclass(frozen=True)
class Patch:
    translation: sparse.csr_matrix
    radial: tuple
    cells: np.ndarray
    indices: np.ndarray
    normal: np.ndarray
    tangent: np.ndarray
    offset: np.ndarray
    rest_gap: np.ndarray
    stiffness: np.ndarray
    depth_limit: np.ndarray


def _patch(engine, frame):
    triangles, faces, _, _ = surfaces.sweep_mesh(frame)
    binding = frame[0].vertex_motion.binding
    movable = np.asarray(binding.translation.sum(axis=1)).ravel()
    active = np.flatnonzero(np.any(movable[faces] > 1e-10, axis=1))
    if not len(active):
        return None
    triangles = triangles[active]
    points = triangles.mean(axis=1)
    crosses = np.cross(triangles[:, 1] - triangles[:, 0], triangles[:, 2] - triangles[:, 0])
    double_area = np.linalg.norm(crosses, axis=1)
    inverse = engine.scene.inverse
    ijk = points @ inverse[:3, :3].T + inverse[:3, 3]
    centers = np.floor(ijk + .5).astype(int)
    shape = np.asarray(engine.scene.shape)
    labels = engine.scene.labels.ravel()
    soft = np.array([label for label, region in engine.by_label.items() if region['material'] in SOFT])
    best = np.full(len(points), np.inf)
    best_cell = np.full(len(points), -1, dtype=int)
    best_axis = np.zeros(len(points), dtype=int)
    best_sign = np.zeros(len(points))
    best_gap = np.zeros(len(points))
    axes = inverse[:3, :3]
    widths = 1. / np.linalg.norm(axes, axis=1)
    normals = axes * widths[:, None]
    for axis in range(3):
        alignment = np.abs(crosses @ normals[axis]) / np.maximum(double_area, 1e-30)
        for sign in (-1, 1):
            for step in (0, 1):
                cell = centers.copy()
                cell[:, axis] -= step * sign
                neighbor = cell.copy()
                neighbor[:, axis] += sign
                gap = sign * (ijk[:, axis] - cell[:, axis]) - .5
                score = np.abs(gap) * widths[axis] / np.maximum(alignment, 1e-30)
                # Keep only exposed interior source faces near the initial
                # wall. Small rest overlaps come from mismatched sampling of
                # the soft mask and vascular score surface.
                eligible = ((score < best) & (alignment >= .25) & (gap >= -.25) & (gap <= .75)
                            & ((cell >= 0) & (cell < shape) & (neighbor >= 0) & (neighbor < shape)).all(axis=1))
                rows = np.flatnonzero(eligible)
                selected = np.ravel_multi_index(cell[rows].T, tuple(shape))
                adjacent = np.ravel_multi_index(neighbor[rows].T, tuple(shape))
                valid = np.isin(labels[selected], soft) & ~np.isin(labels[adjacent], soft)
                rows = rows[valid]
                best[rows], best_cell[rows] = score[rows], selected[valid]
                best_axis[rows], best_sign[rows], best_gap[rows] = axis, sign, gap[rows]
    used = np.flatnonzero(best_cell >= 0)
    if not len(used):
        return None
    cells, indices = np.unique(best_cell[used], return_inverse=True)
    axes = best_axis[used]
    normal = normals[axes] * best_sign[used, None]
    face_width = widths[axes]
    # Rest overlap is assigned zero contact compression, avoiding an artificial
    # initial load between independently discretized source estimates.
    cell_ijk = np.column_stack(np.unravel_index(cells[indices], tuple(shape)))
    offset = ijk[used] - cell_ijk
    rest_gap = np.maximum(best_gap[used], 0.) * face_width
    tangent = np.arange(3) != axes[:, None]
    projected_area = np.abs(np.einsum('ij,ij->i', crosses[used], normal)) / 2.
    # Approximate normal tangent modulus as 3G. Convert Pa to N/mm², then
    # multiply by projected contact area and divide by normal cell width.
    modulus = 3e-6 * material_values(engine, cells, 'equilibriumShearPa')[indices]
    stiffness = modulus * projected_area / face_width
    faces = faces[active[used]]
    q = sparse.csr_matrix((np.full(faces.size, 1 / 3),
                           (np.arange(len(faces)).repeat(3), faces.ravel())),
                          shape=(len(faces), binding.translation.shape[0]))
    return Patch((q @ binding.translation).tocsr(),
                 tuple((q @ radial).tocsr() for radial in binding.radial),
                 cells, indices, normal, tangent, offset, rest_gap, stiffness, .5 * face_width)


def patches(engine, frames, bonds):
    key = wall_sweep.topology(frames), bonds
    cached = getattr(engine, '_vessel_tissue_contacts', None)
    if cached is None or cached[0] != key:
        cached = key, tuple(patch for frame in frames if (patch := _patch(engine, frame)) is not None)
        engine._vessel_tissue_contacts = cached
    return cached[1]


@dataclass(frozen=True)
class Contact:
    patch: Patch
    tissue_shifts: np.ndarray
    retained: np.ndarray

    def loads(self, displacement, shrink, inverse):
        shifts = self.patch.translation @ displacement - np.column_stack([radial @ shrink for radial in self.patch.radial])
        relative = shifts - self.tissue_shifts[self.patch.indices]
        gap = np.einsum('ij,ij->i', relative, self.patch.normal)
        depth = np.maximum(0., -gap - self.patch.rest_gap)
        active = np.flatnonzero((depth > 0.) & self.retained[self.patch.indices])
        loads = np.zeros_like(relative)
        if not len(active):
            return loads
        new_tangent = self.patch.offset[active] + relative[active] @ inverse.T
        edge = np.max(np.abs(new_tangent) * self.patch.tangent[active], axis=1)
        # Taper the load at the edge of a sampled voxel face.
        weight = np.clip((.5 - edge) / .1, 0., 1.)
        force = weight * self.patch.stiffness[active] * np.minimum(depth[active], self.patch.depth_limit[active])
        loads[active] = force[:, None] * self.patch.normal[active]
        return loads


def contacts(engine, frames, bonds, excluded):
    deformation = engine.state['deformation']
    return tuple(Contact(patch, np.array([deformation.get(str(c), (0., 0., 0.)) for c in patch.cells]),
                         ~np.isin(patch.cells, excluded))
                 for patch in patches(engine, frames, bonds))
