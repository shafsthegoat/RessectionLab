"""Continuous piecewise-linear transport on the indexed native wall."""

from dataclasses import dataclass
from hashlib import sha256

import numpy as np
from scipy import sparse
from scipy.sparse import coo_matrix, csr_matrix


PROPERTY = 'vesselWallTransport'
POLICY = 'source-vertex-v1'


def geometry_binding(positions, triangles, source_cells):
    """Match the little-endian array descriptors supplied to the desktop."""
    return ':'.join(sha256(np.asarray(data, dtype=dtype, order='C').tobytes()).hexdigest()
                    for data, dtype in ((positions, '<f4'), (triangles, '<u4'), (source_cells, '<u4')))


def _owner_interpolation(points, faces, owners, owner_count, minimum_spacing):
    edges = np.unique(np.sort(np.concatenate((faces[:, :2], faces[:, 1:], faces[:, ::2])), axis=1), axis=0)
    length = np.linalg.norm(points[edges[:, 0]] - points[edges[:, 1]], axis=1)
    edge_weights = np.exp(-.5 * (length / .34)**2)
    i, j = edges.T
    count = len(points)
    graph = coo_matrix((np.tile(edge_weights, 2), (np.r_[i, j], np.r_[j, i])), shape=(count, count)).tocsr()
    row_mass = np.asarray(graph.sum(axis=1)).ravel()
    diffusion = sparse.diags(np.where(row_mass == 0., 1., .4)) + sparse.diags(.6 / np.maximum(row_mass, 1e-30)) @ graph
    initial = coo_matrix((np.ones(count), (np.arange(count), owners)), shape=(count, owner_count)).tocsr()
    return (diffusion @ (diffusion @ initial)).tocoo(), edges[length < .015 * minimum_spacing]


def _pool_subresolution_edges(matrices, short_edges, protected, vertex_count):
    if not len(short_edges):
        return matrices
    # Pool the response, not the rest positions. This keeps extremely short
    # mesh edges from acquiring large relative motion under one rod increment.
    a, b = short_edges.T
    graph = coo_matrix((np.ones(len(a)), (a, b)), shape=(vertex_count, vertex_count)).tocsr()
    _, component = sparse.csgraph.connected_components(graph, directed=False)
    if protected.any():
        fixed = np.isin(component, component[protected])
        component[fixed] = vertex_count + np.flatnonzero(fixed)
    _, component = np.unique(component, return_inverse=True)
    counts = np.bincount(component)
    accumulator = coo_matrix((np.ones(vertex_count), (component, np.arange(vertex_count))),
                             shape=(len(counts), vertex_count)).tocsr()
    average = sparse.diags(1. / counts) @ accumulator
    return tuple((accumulator.T @ (average @ matrix)).tocsr() for matrix in matrices)


@dataclass(frozen=True)
class VertexBinding:
    identity: str
    translation: csr_matrix
    radial: tuple[csr_matrix, csr_matrix, csr_matrix]

    def offsets(self, delta, compression):
        shrink = np.column_stack([weights @ compression for weights in self.radial])
        return self.translation @ delta - shrink


@dataclass(frozen=True)
class VertexMotion:
    binding: VertexBinding
    offsets: np.ndarray


def bind(wall, rods, points, faces, source_owners, live_bonds, identity, minimum_spacing, protected):
    # Stencils cross only shared edges of this intact or newly cut surface.
    # The original native owner map still tags every contact and risk cell.
    interpolation, short_edges = _owner_interpolation(points, faces, source_owners,
                                                      len(wall.positions), minimum_spacing)
    slots = np.full(len(wall.positions), -1, dtype=int)
    slots[wall.influenced] = np.arange(len(wall.influenced))
    used = slots[interpolation.col] >= 0
    vertices, selected = interpolation.row[used], slots[interpolation.col[used]]
    weight = interpolation.data[used] * wall.weights[selected]
    links = wall.links[selected]
    ends = np.asarray([bond[:2] for bond in rods.bonds], dtype=int).reshape(-1, 2)[links]
    start, end = rods.rest[ends].transpose(1, 0, 2)
    axis = end - start
    local = points[vertices] - start
    length2 = np.einsum('ij,ij->i', axis, axis)
    fraction = np.clip(np.einsum('ij,ij->i', local, axis) / np.maximum(length2, 1e-20), 0., 1.)
    live = np.isin(links, live_bonds)
    fraction = np.where(live, fraction, np.rint(wall.fractions[selected]))
    radial = local - fraction[:, None] * axis
    blend = weight[:, None] * np.column_stack((1. - fraction, fraction))
    row, column = np.repeat(vertices, 2), ends.ravel()
    shape = len(points), len(rods.cells)

    def matrix(values):
        return coo_matrix((values.ravel(), (row, column)), shape=shape).tocsr()

    weights = (matrix(blend), *(matrix(blend * radial[:, axis, None]) for axis in range(3)))
    translation, *radial_weights = _pool_subresolution_edges(weights, short_edges, protected, len(points))
    return VertexBinding(identity, translation, tuple(radial_weights))
