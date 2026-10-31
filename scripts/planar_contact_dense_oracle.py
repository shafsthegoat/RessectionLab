"""Small, independent dense Q1/normal-contact verification oracle.

This module deliberately imports no resectionlab assembly or solver code. Hex8
corners use itertools.product((0, 1), repeat=3) order. Units are mm, N and Pa.
It is a bounded numerical check, not a patient model or a production backend.
"""
from itertools import product

import numpy as np
from scipy.linalg import lstsq
from scipy.optimize import LinearConstraint, minimize, nnls


_CORNERS = np.array(list(product((0., 1.), repeat=3)))


def assemble_dense(nodes_mm, element_nodes, shear_pa, poisson_ratio):
    """Full 2x2x2 Gauss integration, derived from physical node coordinates."""
    nodes = np.asarray(nodes_mm, dtype=float)
    elements = np.asarray(element_nodes, dtype=int)
    mu = np.broadcast_to(np.asarray(shear_pa, float), (len(elements),)) * 1e-6
    nu = np.broadcast_to(np.asarray(poisson_ratio, float), (len(elements),))
    matrix = np.zeros((nodes.size, nodes.size))
    for e, ids in enumerate(elements):
        xyz = nodes[ids]
        local = np.zeros((24, 24))
        lame = 2 * mu[e] * nu[e] / (1 - 2 * nu[e])
        material = np.diag([2 * mu[e]] * 3 + [mu[e]] * 3)
        material[:3, :3] += lame
        for q in product((.5 - .5 / np.sqrt(3), .5 + .5 / np.sqrt(3)), repeat=3):
            factors = np.where(_CORNERS == 1, q, 1 - np.asarray(q))
            derivative = np.empty((8, 3))
            for a in range(3):
                derivative[:, a] = (2 * _CORNERS[:, a] - 1) * np.prod(
                    factors[:, [b for b in range(3) if b != a]], axis=1)
            jacobian = xyz.T @ derivative
            gradient = derivative @ np.linalg.inv(jacobian)
            b = np.zeros((6, 24))
            for j, (dx, dy, dz) in enumerate(gradient):
                k = 3 * j
                b[0, k], b[1, k + 1], b[2, k + 2] = dx, dy, dz
                b[3, k:k + 2] = dy, dx
                b[4, k + 1:k + 3] = dz, dy
                b[5, [k, k + 2]] = dz, dx
            local += b.T @ material @ b * abs(np.linalg.det(jacobian)) / 8
        dofs = (3 * ids[:, None] + np.arange(3)).ravel()
        matrix[np.ix_(dofs, dofs)] += local
    return matrix


def rectangular_surface_rows(nodes_mm, element_nodes, faces, origin_mm, axis_u,
                             axis_v, inward_normal, size_mm, order=2, include_guards=True):
    """Independent rectangular Gauss sampling of axis-aligned faces in a frame.

    Faces are (element row, reference axis, outward sign) triples. Each face's
    sides must be parallel to the supplied orthonormal tangents. Returns C,
    reference targets, points and area weights; C u is normal displacement. No area factor belongs
    in a pointwise displacement constraint or its point-force multiplier. Each
    face's Gauss rows precede four zero-weight corner guards by default. Corner
    gaps bound the bilinear face gap over the entire clipped rectangle, so Gauss
    order variation preserves the same feasible set, not a pressure refinement.
    """
    xyz = np.asarray(nodes_mm, float)
    elements = np.asarray(element_nodes, int)
    origin = np.asarray(origin_mm, float)
    frame = np.array([axis_u, axis_v, inward_normal], float)
    if not np.allclose(frame @ frame.T, np.eye(3), atol=1e-12, rtol=0):
        raise ValueError('oracle rectangle frame must be orthonormal')
    rows, points, weights, targets = [], [], [], []
    half = np.asarray(size_mm, float) / 2
    quadrature, gauss_weights = np.polynomial.legendre.leggauss(order)
    for element, axis, sign in faces:
        local_ids = np.flatnonzero(_CORNERS[:, axis] == (sign + 1) / 2)
        ids = elements[element]
        face_nodes = ids[local_ids]
        coordinates = (xyz[face_nodes] - origin) @ frame.T
        if np.ptp(coordinates[:, 2]) > 1e-9:
            raise ValueError('oracle requires faces parallel to rectangle plane')
        lower = np.maximum(coordinates[:, :2].min(axis=0), -half)
        upper = np.minimum(coordinates[:, :2].max(axis=0), half)
        if np.any(upper <= lower):
            continue
        original_lower, original_upper = coordinates[:, :2].min(0), coordinates[:, :2].max(0)
        if any(not np.all(np.isclose(coordinates[:, a], original_lower[a], atol=1e-9)
                          | np.isclose(coordinates[:, a], original_upper[a], atol=1e-9))
               for a in range(2)):
            raise ValueError('oracle rectangle requires frame-aligned faces')
        samples = [((lower + upper) / 2 + np.array([quadrature[i], quadrature[j]])
                    * (upper - lower) / 2,
                    np.prod(upper - lower) / 4 * gauss_weights[i] * gauss_weights[j])
                   for i, j in product(range(order), repeat=2)]
        if include_guards:
            samples.extend((np.array(p), 0.) for p in product(*zip(lower, upper)))
        for p, weight in samples:
            fraction = (p - original_lower) / (original_upper - original_lower)
            shape = np.prod(np.where(np.isclose(coordinates[:, :2], original_upper),
                                     fraction, 1 - fraction), axis=1)
            row = np.zeros(xyz.size)
            for node, value in zip(face_nodes, shape):
                row[3 * node:3 * node + 3] = value * frame[2]
            rows.append(row)
            point = shape @ xyz[face_nodes]
            points.append(point)
            targets.append(float((origin - point) @ frame[2]))
            weights.append(weight)
    return {'contact_rows': np.asarray(rows).reshape((-1, xyz.size)),
            'points_mm': np.asarray(points).reshape((-1, 3)),
            'targets_at_zero_mm': np.array(targets), 'weights_mm2': np.array(weights)}


def _polish(k, f, c, b, guess):
    """Solve active equality KKT systems; recover redundant duals with NNLS."""
    u = guess.copy()
    active = np.flatnonzero(c @ u - b < 1e-8)
    for iteration in range(100):
        a = c[active]
        system = np.block([[k, -a.T], [a, np.zeros((len(a), len(a)))]])
        result = lstsq(system, np.r_[f, b[active]], cond=1e-13, lapack_driver='gelsy')[0]
        u = result[:len(f)]
        gap = c @ u - b
        if len(gap) and gap.min() < -2e-11:
            worst = int(np.argmin(gap))
            if worst in active:
                raise RuntimeError('oracle inconsistent active constraints')
            active = np.r_[active, worst]
            continue
        multipliers = np.zeros(len(b))
        if len(active):
            multipliers[active], residual = nnls(a.T, k @ u - f, maxiter=10000)
        else:
            residual = np.linalg.norm(k @ u - f)
        if residual < 2e-12:
            return u, multipliers, iteration + 1
        # An equality with a negative force is not binding. Redundant systems
        # first get a full nonnegative force recovery above, before pruning.
        signed = result[len(f):]
        if not len(signed) or signed.min() >= -1e-13:
            raise RuntimeError('oracle could not certify active KKT system')
        active = np.delete(active, int(np.argmin(signed)))
    raise RuntimeError('oracle active-set polish did not converge')


def solve(nodes_mm, element_nodes, shear_pa, poisson_ratio, branch_shear_pa,
          relaxation_seconds, free_dofs_bool, contact_rows, contact_targets_mm,
          dt_s, previous_u_mm=None, previous_z_mm=None):
    """Primal constrained energy solve followed by a dense KKT certification.

    Constraints are C u - b >= 0 and K u - f - C.T lambda = 0. Fixed DOFs are
    zero. Maxwell memories are branch-by-DOF displacement arrays. Material
    shear and Poisson ratio can be scalar or element-length arrays; branch
    shear is branch-length (uniform material) or branch-by-element-length.
    """
    ndof = np.asarray(nodes_mm).size
    free = np.asarray(free_dofs_bool, bool).reshape(ndof)
    rows = contact_rows.toarray() if hasattr(contact_rows, 'toarray') else contact_rows
    c_full = np.asarray(rows, float).reshape((-1, ndof))
    b = np.asarray(contact_targets_mm, float).reshape(len(c_full))
    times = np.asarray(relaxation_seconds, float).reshape(-1)
    branch_mu = np.asarray(branch_shear_pa, float)
    if branch_mu.shape[:1] != (len(times),) or dt_s <= 0 or np.any(times <= 0):
        raise ValueError('invalid oracle branch spectrum or step')
    z0 = (np.zeros((len(times), ndof)) if previous_z_mm is None else
          np.asarray(previous_z_mm, float).reshape((len(times), ndof)))
    k_full = assemble_dense(nodes_mm, element_nodes, shear_pa, poisson_ratio)
    f_full = np.zeros(ndof)
    ratios = times / (times + dt_s)
    for mu, ratio, z in zip(branch_mu, ratios, z0):
        branch = assemble_dense(nodes_mm, element_nodes, mu, poisson_ratio)
        k_full += ratio * branch
        f_full += ratio * (branch @ z)
    k, f, c = k_full[np.ix_(free, free)], f_full[free], c_full[:, free]
    np.linalg.cholesky(k)  # reject unanchored or indefinite verification cases
    initial = (np.linalg.solve(k, f) if previous_u_mm is None else
               np.asarray(previous_u_mm, float).reshape(ndof)[free])
    scale = max(float(np.max(np.diag(k))), 1e-30)
    constraints = [] if not len(b) else [LinearConstraint(c, b, np.inf)]
    result = minimize(lambda u: (.5 * u @ k @ u - f @ u) / scale, initial,
                      jac=lambda u: (k @ u - f) / scale,
                      constraints=constraints, method='SLSQP',
                      options={'ftol': 1e-14, 'maxiter': 2000})
    reduced, multipliers, iterations = _polish(k, f, c, b, result.x)
    u = np.zeros(ndof)
    u[free] = reduced
    z = np.array([(tau * old + dt_s * u) / (tau + dt_s)
                  for tau, old in zip(times, z0)]).reshape((len(times), ndof))
    force = c_full.T @ multipliers
    gap = c_full @ u - b
    diagnostics = {
        'max_gap_violation_mm': float(max(0., -np.min(gap, initial=0.))),
        'max_dual_violation_n': float(max(0., -np.min(multipliers, initial=0.))),
        'max_stationarity_residual_n': float(np.max(np.abs((k_full @ u - f_full - force)[free]), initial=0.)),
        'max_complementarity_nmm': float(np.max(np.abs(multipliers * gap), initial=0.)),
        'primal_optimizer_success': bool(result.success), 'kkt_polish_iterations': iterations,
    }
    if (diagnostics['max_gap_violation_mm'] > 1e-9 or
            diagnostics['max_stationarity_residual_n'] > 1e-10 or
            diagnostics['max_dual_violation_n'] > 1e-10 or
            diagnostics['max_complementarity_nmm'] > 1e-10):
        raise RuntimeError(f'oracle certification failed: {diagnostics}')
    return {'u_mm': u, 'z_mm': z, 'multipliers_n': multipliers,
            'contact_nodal_force_n': force, 'gaps_mm': gap,
            'diagnostics': diagnostics}
