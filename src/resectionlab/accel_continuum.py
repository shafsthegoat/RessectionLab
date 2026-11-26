"""Opt-in continuum and sparse-solve fast paths (see ``acceleration``).

``continuum`` (exact): memoized element operators, vectorized assembly and a
vectorized exposure integral. Every arithmetic step matches the reference
expression element by element, so matrices and ledgers are bit-identical.

``sparse-ordering`` (reordered): nested-dissection ordering before SuperLU's
symmetric-mode factorization of the same symmetric positive-definite systems.
The equations are unchanged; floating-point summation order is not.
"""
from __future__ import annotations

from collections import OrderedDict
from functools import partial
import threading

import numpy as np
from scipy import sparse
from scipy.sparse import csgraph
from scipy.sparse.linalg import splu

_OPERATOR_CACHE_LIMIT = 32
_operator_cache = OrderedDict()
_operator_lock = threading.Lock()


def _readonly(array):
    array = np.array(array, copy=True)
    array.setflags(write=False)
    return array


def element_operators(affine, reference):
    """Memoize ``reference(affine)`` by the exact linear-part bytes.

    The operators depend only on the 3x3 linear part. Cached arrays are
    read-only so an accidental in-place edit fails instead of leaking.
    """
    linear = np.ascontiguousarray(np.asarray(affine, float)[:3, :3])
    key = (getattr(reference, '__code__', reference), linear.tobytes())
    with _operator_lock:
        hit = _operator_cache.get(key)
        if hit is not None:
            _operator_cache.move_to_end(key)
            return hit
    value = tuple(_readonly(part) for part in reference(affine))
    with _operator_lock:
        _operator_cache[key] = value
        while len(_operator_cache) > _OPERATOR_CACHE_LIMIT:
            _operator_cache.popitem(last=False)
    return value


def _cell_labels(scene, mesh):
    return scene.labels.reshape(-1)[np.asarray(mesh.cells, dtype=np.int64)].astype(np.int64)


def _branch_coefficients(materials, labels, branch):
    """Per-element (mu N/mm^2 at Pa scale, Poisson) for one Maxwell branch."""
    unique, inverse = np.unique(labels, return_inverse=True)
    pa = np.empty(len(unique)); nu = np.empty(len(unique))
    for row, label in enumerate(unique):
        m = materials[str(int(label))]
        pa[row] = [m['shearPa'], *m['relaxationShearPa']][branch]
        nu[row] = m['poissonRatio']
    return pa[inverse], nu[inverse]


def assemble(scene, mesh, materials, element_operators):
    """Vectorized ``surgical_continuum.assemble``; identical COO input order."""
    shear, lame, gradients = element_operators(scene.affine)
    branch_count = len(next(iter(materials.values()))['relaxationShearPa'])
    dofs = np.asarray(mesh.element_dofs, dtype=np.int64).reshape((-1, 24))
    rows = np.repeat(dofs, 24, axis=1).ravel()
    cols = np.tile(dofs, (1, 24)).ravel()
    labels = _cell_labels(scene, mesh)
    shape = (len(mesh.nodes) * 3,) * 2
    matrices = []
    for branch in range(1 + branch_count):
        pa, nu = _branch_coefficients(materials, labels, branch)
        mu = pa * 1e-6  # Pa -> N/mm².
        lam = 2 * mu * nu / (1 - 2 * nu)
        values = (mu[:, None, None] * shear + lam[:, None, None] * lame).ravel()
        if not len(values):
            values = []
        matrices.append(sparse.coo_matrix((values, (rows, cols)), shape=shape).tocsr())
    return tuple(matrices), gradients


def exposure_increment(scene, mesh, gradients, displacement, memories, materials, h, gradient,
                       *, include_peaks=False):
    """Vectorized ``surgical_continuum.exposure_increment`` (same expressions).

    ``gradient`` is the module's own contraction (``_gradient``), passed in so
    the declared tensor-contraction revision is honoured unchanged.
    """
    gradients = gradients[:8]
    nodes = mesh.element_nodes
    local = displacement.reshape((-1, 3))[nodes]
    first = gradient(scene, local, gradients)
    strain = (first + first.transpose(0, 1, 3, 2)) / 2.
    labels = _cell_labels(scene, mesh)
    stress = np.zeros_like(strain)
    eye = np.eye(3)
    for branch, field in enumerate([displacement, *[displacement - z for z in memories]]):
        # Branch 0's field is the displacement itself: reuse its contraction.
        g = first if branch == 0 else gradient(scene, field.reshape((-1, 3))[nodes], gradients)
        eps = (g + g.transpose(0, 1, 3, 2)) / 2.
        mu, nu = _branch_coefficients(materials, labels, branch)
        lam = 2 * mu * nu / (1 - 2 * nu)
        stress += 2 * mu[:, None, None, None] * eps + lam[:, None, None, None] * np.trace(eps, axis1=2, axis2=3)[:, :, None, None] * eye
    weight = mesh.volume_mm3 * h / 8.
    result = {}
    for label in sorted(set(labels.tolist())):
        select = labels == label
        strain_sq = strain[select] ** 2
        stress_sq = stress[select] ** 2
        row = {'strainSquaredVolumeTimeMm3Seconds': float(np.sum(strain_sq) * weight),
               'stressSquaredVolumeTimePa2Mm3Seconds': float(np.sum(stress_sq) * weight)}
        if include_peaks:
            row.update(peakStrainNorm=float(np.sqrt(strain_sq.sum(axis=(2, 3)).max())),
                       peakStressNormPa=float(np.sqrt(stress_sq.sum(axis=(2, 3)).max())))
        result[str(label)] = row
    return result


# --- sparse-ordering -------------------------------------------------------

_LEAF = 48


def _block_size(matrix):
    """3 when the free system is whole 3-dof nodes with a block pattern."""
    n = matrix.shape[0]
    if n % 3:
        return 1
    coo = matrix.tocoo()
    blocks = sparse.coo_matrix((np.ones(len(coo.row)), (coo.row // 3, coo.col // 3)),
                               shape=(n // 3, n // 3)).tocsr()
    return 3 if blocks.nnz * 9 >= coo.nnz else 1


def nested_dissection(matrix, *, leaf=_LEAF):
    """Level-set nested dissection of a symmetric pattern; returns a permutation.

    Recursively splits each connected component at the middle BFS level from a
    pseudo-peripheral vertex, ordering both halves before the separator.
    """
    n = matrix.shape[0]
    if n == 0:
        return np.empty(0, np.int64)
    block = _block_size(matrix)
    m = n // block
    coo = matrix.tocoo()
    graph = sparse.coo_matrix((np.ones(len(coo.row)), (coo.row // block, coo.col // block)),
                              shape=(m, m)).tocsr()
    graph = ((graph + graph.T) > 0).astype(np.int8).tocsr()
    graph.setdiag(0)
    graph.eliminate_zeros()
    order = []
    # Explicit stack of (nodes) and (separator) markers keeps recursion shallow.
    stack = [('split', np.arange(m))]
    while stack:
        kind, nodes = stack.pop()
        if kind == 'emit':
            order.append(nodes)
            continue
        if len(nodes) <= leaf:
            order.append(nodes)
            continue
        sub = graph[nodes][:, nodes]
        count, labels = csgraph.connected_components(sub, directed=False)
        if count > 1:
            for component in reversed(range(count)):
                stack.append(('split', nodes[labels == component]))
            continue
        start = 0
        for _ in range(2):
            distance = csgraph.shortest_path(sub, unweighted=True, indices=start, directed=False)
            start = int(np.argmax(distance))
        distance = csgraph.shortest_path(sub, unweighted=True, indices=start, directed=False).astype(np.int64)
        depth = int(distance.max())
        if depth < 2:
            order.append(nodes)
            continue
        middle = depth // 2
        # Pushed in reverse: left, right, then the separator last.
        stack.append(('emit', nodes[distance == middle]))
        stack.append(('split', nodes[distance > middle]))
        stack.append(('split', nodes[distance < middle]))
    node_order = np.concatenate(order) if order else np.empty(0, np.int64)
    if block == 1:
        return node_order.astype(np.int64)
    return (block * node_order[:, None] + np.arange(block)).ravel().astype(np.int64)


class OrderedFactor:
    """SuperLU on a nested-dissection-permuted SPD matrix, same ``solve`` API."""

    def __init__(self, matrix):
        matrix = sparse.csc_matrix(matrix)
        self.shape = matrix.shape
        self.perm = nested_dissection(matrix)
        permuted = matrix[self.perm][:, self.perm].tocsc()
        self._lu = splu(permuted, permc_spec='NATURAL', diag_pivot_thresh=0.,
                        options={'SymmetricMode': True})
        # One probe solve: an unsuitable (non-SPD) system falls back to the
        # caller's reference factorization rather than returning poor answers.
        probe = np.linspace(1., 2., matrix.shape[0])
        solution = self.solve(probe)
        scale = max(float(np.max(np.abs(matrix.data), initial=0.)), 1e-300)
        residual = float(np.max(np.abs(matrix @ solution - probe), initial=0.))
        size = float(np.max(np.abs(solution), initial=0.))
        if not np.isfinite(solution).all() or residual > 1e-8 * max(1., scale * size):
            raise ArithmeticError('ordered factorization failed its probe solve')

    def solve(self, rhs):
        rhs = np.asarray(rhs)
        result = np.empty(rhs.shape, np.result_type(rhs.dtype, np.float64))
        result[self.perm] = self._lu.solve(np.ascontiguousarray(rhs[self.perm]))
        return result

    def __call__(self, rhs):
        return self.solve(rhs)


def _stock(reference):
    """Only the stock SciPy factorizers are replaced; instrumented ones are kept."""
    from scipy.sparse import linalg
    if reference is linalg.factorized:
        return True
    return isinstance(reference, partial) and reference.func is linalg.splu


def ordered_factor(matrix, reference):
    """Return an ``OrderedFactor``; defer to ``reference(matrix)`` otherwise.

    ``reference`` must be SciPy's ``factorized`` or a ``functools.partial`` of
    ``splu``. A replaced or wrapped factorizer (a test probe, a custom solver)
    is called unchanged, as is the reference for systems the probe rejects.
    """
    if not _stock(reference):
        return reference(matrix)
    try:
        return OrderedFactor(matrix)
    except ArithmeticError:
        return reference(matrix)
    except RuntimeError:
        # Singular under natural ordering: let the reference raise its own error.
        return reference(matrix)
