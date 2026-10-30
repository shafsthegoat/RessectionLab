"""Opt-in offline finite rectangular blade on a source-owned small-strain Q1 body.

This is a fixed reference-footprint, normal-only Signorini problem. It never
creates a SurgicalEngine hold or provides rigid support through contact. No
action integration, friction, edge migration or finite-strain contact is implied.
"""
from __future__ import annotations

from dataclasses import dataclass
from itertools import product
import time

import numpy as np
from scipy import sparse
from scipy.optimize import minimize
from scipy.sparse.linalg import splu

from .core import semantic_digest, immutable_array
from . import surgical_continuum as continuum

SCHEMA = 'resectionlab.planar-blade-contact.v1'
GAP_MM = 1e-9
FORCE_N = 1e-10
BALANCE_NMM = 1e-10
MAX_CELLS, MAX_DOFS, MAX_POINTS, MAX_STEPS = 512, 2187, 512, 120
MAX_STEP_SECONDS = 15.


def _owned(value, dtype=float):
    return immutable_array(value, dtype)


def _freeze_sparse(matrix):
    for name in ('data', 'indices', 'indptr'):
        setattr(matrix, name, immutable_array(getattr(matrix, name)))
    return matrix


@dataclass(frozen=True)
class PlanarBlade:
    origin_mm: np.ndarray
    axis_u: np.ndarray
    axis_v: np.ndarray
    inward_normal: np.ndarray
    size_mm: np.ndarray

    @classmethod
    def create(cls, origin_mm, axis_u, axis_v, inward_normal, size_mm):
        origin = np.asarray(origin_mm, float)
        frame = np.asarray([axis_u, axis_v, inward_normal], float)
        size = np.asarray(size_mm, float)
        if (origin.shape != (3,) or frame.shape != (3, 3) or size.shape != (2,)
                or not all(np.isfinite(v).all() for v in (origin, frame, size))
                or np.any(size <= 0) or np.max(np.abs(frame @ frame.T - np.eye(3))) > 1e-10):
            raise ValueError('PLANAR_CONTACT_ORTHONORMAL_BLADE_REQUIRED')
        return cls(_owned(origin), *(_owned(v) for v in frame), _owned(size))


@dataclass(frozen=True)
class ContactState:
    model_id: str
    displacement_mm: np.ndarray
    memory_mm: np.ndarray
    offset_mm: float
    steps: int


def _faces(scene, mesh, faces, *, exposed):
    if not isinstance(faces, (list, tuple)) or not faces or len(faces) > 6 * MAX_CELLS:
        raise ValueError('PLANAR_CONTACT_EXPLICIT_FACE_SET_REQUIRED')
    rows = {c: i for i, c in enumerate(mesh.cells)}
    result = []
    seen = set()
    for face in faces:
        if (not isinstance(face, (tuple, list)) or len(face) != 3
                or any(isinstance(v, (bool, np.bool_)) or not isinstance(v, (int, np.integer)) for v in face)):
            raise ValueError('PLANAR_CONTACT_SOURCE_FACE_REQUIRED')
        cell, axis, sign = map(int, face)
        if cell not in rows or axis not in (0, 1, 2) or sign not in (-1, 1) or (cell, axis, sign) in seen:
            raise ValueError('PLANAR_CONTACT_SOURCE_FACE_REQUIRED')
        seen.add((cell, axis, sign))
        ijk = np.array(np.unravel_index(cell, scene.shape))
        neighbor = ijk.copy(); neighbor[axis] += sign
        if np.any(neighbor < 0) or np.any(neighbor >= scene.shape):
            raise ValueError('PLANAR_CONTACT_UNKNOWN_OUT_OF_IMAGE_FACE')
        other = int(np.ravel_multi_index(neighbor, scene.shape))
        if other in rows or exposed and int(scene.labels.flat[other]) != 0:
            raise ValueError('PLANAR_CONTACT_FACE_NOT_EXPOSED')
        result.append((cell, rows[cell], axis, sign))
    return result


def _surface(scene, mesh, faces, blade, order):
    """Exact reference rectangle clipping; Gauss rows plus Q1 corner guards.

    On each aligned clipped rectangle the normal gap is bilinear, so its four
    vertex inequalities bound the entire patch. Guard multipliers are integrated
    forces, not pressures. Their zero integration weight is never used to discard
    their force. Redundant Gauss multipliers need not be unique.
    """
    if type(order) is not int or not 1 <= order <= 4:
        raise ValueError('PLANAR_CONTACT_GAUSS_ORDER')
    a = scene.affine[:3, :3]; inverse = np.linalg.inv(a)
    if np.max(np.abs((a.T @ a) - np.diag(np.diag(a.T @ a)))) > 1e-10:
        raise ValueError('PLANAR_CONTACT_ORTHOGONAL_SOURCE_GRID_REQUIRED')
    xi, wi = np.polynomial.legendre.leggauss(order)
    locations, weights, shape_rows, source_ids, guards = [], [], [], [], []
    for cell, row, axis, sign in faces:
        local_corners = np.flatnonzero(continuum.CORNERS[:, axis] == (sign + 1) // 2)
        corners = mesh.nodes[mesh.element_nodes[row, local_corners]]
        outward = sign * inverse[axis] / np.linalg.norm(inverse[axis])
        if np.linalg.norm(outward + blade.inward_normal) > 1e-10:
            raise ValueError('PLANAR_CONTACT_FACE_ORIENTATION')
        if np.max(np.abs((corners - blade.origin_mm) @ blade.inward_normal)) > GAP_MM:
            raise ValueError('PLANAR_CONTACT_COPLANAR_INITIAL_SURFACE_REQUIRED')
        tangents = [a[:, k] / np.linalg.norm(a[:, k]) for k in range(3) if k != axis]
        if not all(any(abs(float(t @ b)) > 1 - 1e-10 for t in tangents) for b in (blade.axis_u, blade.axis_v)):
            raise ValueError('PLANAR_CONTACT_ALIGNED_RECTANGULAR_FACES_REQUIRED')
        uv = np.column_stack(((corners - blade.origin_mm) @ blade.axis_u, (corners - blade.origin_mm) @ blade.axis_v))
        low = np.maximum(uv.min(axis=0), -blade.size_mm / 2)
        high = np.minimum(uv.max(axis=0), blade.size_mm / 2)
        if np.any(high - low <= 1e-12):
            continue
        samples = [((low + high) / 2 + np.array([x, y]) * (high - low) / 2,
                    wx * wy * float(np.prod(high - low)) / 4, False)
                   for (x, wx), (y, wy) in product(zip(xi, wi), repeat=2)]
        samples += [(np.array([x, y]), 0., True) for x, y in product(*zip(low, high))]
        ijk = np.array(np.unravel_index(cell, scene.shape))
        for point, weight, guard in samples:
            world = blade.origin_mm + point[0] * blade.axis_u + point[1] * blade.axis_v
            ref = inverse @ (world - scene.affine[:3, 3]) - ijk + .5
            shape = np.prod(np.where(continuum.CORNERS, ref, 1 - ref), axis=1)
            shape_rows.append((row, shape)); locations.append(world); weights.append(weight)
            source_ids.append(cell); guards.append(guard)
    if not locations or len(locations) > MAX_POINTS:
        raise ValueError('PLANAR_CONTACT_EMPTY_FOOTPRINT_OR_POINT_LIMIT')
    cols, data = [], []
    for row, shape in shape_rows:
        cols.extend((3 * mesh.element_nodes[row, :, None] + np.arange(3)).ravel())
        data.extend((shape[:, None] * blade.inward_normal).ravel())
    c = sparse.coo_matrix((data, (np.repeat(np.arange(len(locations)), 24), cols)),
                          shape=(len(locations), 3 * len(mesh.nodes))).tocsr()
    c.eliminate_zeros()
    points = np.asarray(locations)
    b0 = (blade.origin_mm - points) @ blade.inward_normal
    return c, _owned(b0), _owned(weights), _owned(points), _owned(source_ids, np.int64), _owned(guards, bool), tuple(shape_rows)


def build(scene, cells, *, contact_faces, support_faces, blade, quadrature_order=2):
    """Explicit opt-in builder. It installs no scene property or live engine mode."""
    scene.require_cache_source()
    if not isinstance(blade, PlanarBlade):
        raise ValueError('PLANAR_CONTACT_EXPLICIT_BLADE_REQUIRED')
    # Revalidate even directly constructed dataclasses; detach all input arrays.
    blade = PlanarBlade.create(blade.origin_mm, blade.axis_u, blade.axis_v, blade.inward_normal, blade.size_mm)
    p = continuum.validate(scene)
    if p is None or len(p['relaxationSeconds']) > 1:
        raise ValueError('PLANAR_CONTACT_ELASTIC_OR_ONE_MAXWELL_REQUIRED')
    cells = tuple(cells)
    if not 1 <= len(cells) <= MAX_CELLS:
        raise ValueError('PLANAR_CONTACT_SOURCE_CELL_LIMIT')
    mesh = continuum.voxel_mesh(scene, cells, max_dofs=min(MAX_DOFS, p['maxDofs']))
    contacts = _faces(scene, mesh, contact_faces, exposed=True)
    supports = _faces(scene, mesh, support_faces, exposed=False)
    fixed = set()
    for _, row, axis, sign in supports:
        local = continuum.CORNERS[:, axis] == (sign + 1) // 2
        fixed.update(map(int, mesh.element_nodes[row, local]))
    # Contact is deliberately absent here, even when the blade initially touches.
    continuum.check_support(mesh, fixed, set())
    free = np.ones(3 * len(mesh.nodes), bool)
    for node in fixed:
        free[3 * node:3 * node + 3] = False
    if not np.any(free):
        raise ValueError('PLANAR_CONTACT_FREE_TISSUE_REQUIRED')
    matrices, gradients = continuum.assemble(scene, mesh, p['materials'])
    surface = _surface(scene, mesh, contacts, blade, quadrature_order)
    identity = semantic_digest({'schema': SCHEMA, 'sceneId': scene.identity, 'cells': list(mesh.cells),
        'contactFaces': [list(f) for f in contact_faces], 'supportFaces': [list(f) for f in support_faces],
        'blade': {k: v.tolist() for k, v in vars(blade).items()}, 'gaussOrder': quadrature_order,
        'materials': p['materials'], 'relaxationSeconds': p['relaxationSeconds']})
    return PlanarContact(scene, mesh, matrices, gradients, _owned(free, bool), blade, surface,
                         _owned(p['relaxationSeconds']), identity)


class PlanarContact:
    def __init__(self, scene, mesh, matrices, gradients, free, blade, surface, times, identity):
        self.scene, self.mesh, self.matrices, self.gradients = scene, mesh, matrices, gradients
        self.free, self.blade, self.times, self.identity = free, blade, times, identity
        (self.contact, self.targets_at_zero_mm, self.weights_mm2, self.points_mm,
         self.source_ids, self.guard_rows, self._shape_rows) = surface
        self._prepared = {}
        for name in ('nodes', 'element_nodes', 'element_dofs'):
            setattr(mesh, name, immutable_array(getattr(mesh, name)))
        self.gradients = immutable_array(gradients)
        for matrix in (*matrices, self.contact):
            _freeze_sparse(matrix)
        # Array snapshots prevent mutation through setflags(); object/buffer
        # checks also reject replaced public operators or geometry under an old ID.
        self._binding = self._binding_objects()
        self._array_metadata = tuple((a, a.shape, a.dtype.str, a.strides)
                                     for a in self._binding if isinstance(a, np.ndarray))
        self._model_id = self.identity
        self._matrix_shapes = tuple(matrix.shape for matrix in (*self.matrices, self.contact))

    def _binding_objects(self):
        return (self.scene, self.mesh, self.blade, self.free, self.times,
            self.mesh.nodes, self.mesh.element_nodes, self.mesh.element_dofs,
            self.mesh.cells, self.mesh.node_owners, self.mesh.components, self.gradients,
            self.blade.origin_mm, self.blade.axis_u, self.blade.axis_v,
            self.blade.inward_normal, self.blade.size_mm,
            self.targets_at_zero_mm, self.weights_mm2, self.points_mm, self.source_ids,
            self.guard_rows, *self.matrices, self.contact,
            *(v for matrix in (*self.matrices, self.contact) for v in (matrix.data, matrix.indices, matrix.indptr)))

    def _require_binding(self):
        current = self._binding_objects()
        if (self.identity != self._model_id or tuple(matrix.shape for matrix in (*self.matrices, self.contact)) != self._matrix_shapes
                or len(current) != len(self._binding) or any(a is not b for a, b in zip(current, self._binding))
                or any(a.shape != shape or a.dtype.str != dtype or a.strides != strides
                       for a, shape, dtype, strides in self._array_metadata)):
            raise ValueError('PLANAR_CONTACT_MODEL_CHANGED_REBUILD_REQUIRED')

    def initial_state(self):
        n = len(self.free)
        return ContactState(self.identity, _owned(np.zeros(n)), _owned(np.zeros((len(self.times), n))), 0., 0)

    def _prepare(self, h, check):
        if h in self._prepared:
            return self._prepared[h]
        if len(self._prepared) >= 16:
            raise ValueError('PLANAR_CONTACT_TIME_STEP_CACHE_LIMIT')
        ratio = self.times / (self.times + h)
        effective = self.matrices[0].copy()
        for r, k in zip(ratio, self.matrices[1:]):
            effective = effective + r * k
        a = effective[self.free][:, self.free].tocsc()
        check()
        factor = splu(a, permc_spec='MMD_AT_PLUS_A', options={'SymmetricMode': True})
        c = self.contact[:, self.free]
        response = factor.solve(c.T.toarray())
        residual = float(np.max(np.abs(a @ response - c.T.toarray()), initial=0.))
        compliance = np.asarray(c @ response)
        asymmetry = float(np.max(np.abs(compliance - compliance.T), initial=0.))
        scale = float(np.max(np.abs(compliance), initial=0.))
        tolerance = 64 * np.finfo(float).eps * len(compliance) * max(scale, 1e-300)
        compliance = (compliance + compliance.T) / 2
        if (not np.isfinite(response).all() or residual > FORCE_N or asymmetry > tolerance
                or np.linalg.eigvalsh(compliance).min(initial=0.) < -tolerance):
            raise ValueError('PLANAR_CONTACT_BODY_OR_COMPLIANCE_RESIDUAL')
        check()
        result = (ratio, a, factor, response, compliance)
        self._prepared[h] = result
        return result

    def step(self, state, *, offset_mm, dt_s, check=lambda: None):
        """Return a detached candidate state and certified receipt; never mutate history.

        Positive offset moves the plane inward. g=C*u-b >=0 means separation;
        lambda>=0 is compressive integrated force. All vector forces are parallel
        to the fixed inward normal, so contact has exactly zero tangential traction.
        """
        started = time.perf_counter()
        def guard():
            check()
            if time.perf_counter() - started > MAX_STEP_SECONDS:
                raise ValueError('PLANAR_CONTACT_STEP_TIME_LIMIT')
        guard(); self._require_binding(); self.scene.require_cache_source()
        if (not isinstance(state, ContactState) or state.model_id != self.identity
                or type(state.steps) is not int or not 0 <= state.steps < MAX_STEPS
                or type(offset_mm) not in (int, float) or not np.isfinite(offset_mm)
                or type(dt_s) not in (int, float) or not np.isfinite(dt_s) or dt_s <= 0
                or type(state.offset_mm) not in (int, float) or not np.isfinite(state.offset_mm)):
            raise ValueError('PLANAR_CONTACT_STATE_OR_SCHEDULE')
        u0 = np.asarray(state.displacement_mm); z0 = np.asarray(state.memory_mm); n = len(self.free)
        if (u0.shape != (n,) or z0.shape != (len(self.times), n)
                or not np.isfinite(u0).all() or not np.isfinite(z0).all()
                or np.any(u0[~self.free] != 0) or np.any(z0[:, ~self.free] != 0)):
            raise ValueError('PLANAR_CONTACT_HISTORY_OR_SUPPORT')
        old_b = self.targets_at_zero_mm + state.offset_mm
        old_gap = self.contact @ u0 - old_b
        if np.min(old_gap, initial=0.) < -GAP_MM:
            raise ValueError('PLANAR_CONTACT_INITIAL_PENETRATION')
        ratio, a, factor, response, compliance = self._prepare(float(dt_s), guard)
        rhs = np.zeros(n)
        for r, k, z in zip(ratio, self.matrices[1:], z0):
            rhs += r * (k @ z)
        base = factor.solve(rhs[self.free]); c = self.contact[:, self.free]
        if np.max(np.abs(a @ base - rhs[self.free]), initial=0.) > FORCE_N:
            raise ValueError('PLANAR_CONTACT_BODY_RESIDUAL')
        b = self.targets_at_zero_mm + offset_mm
        deficit = b - c @ base
        m = len(b)
        def objective(x):
            guard()
            return float(.5 * x @ (compliance @ x) - deficit @ x), compliance @ x - deficit
        if np.max(deficit, initial=0.) <= 0:
            force = np.zeros(m); iterations = 0; optimizer_status = 0; function_calls = 0
        else:
            solved = minimize(objective, np.zeros(m), jac=True, method='L-BFGS-B', bounds=[(0., None)] * m,
                options={'gtol': GAP_MM / 10, 'ftol': 0., 'maxiter': 1000, 'maxfun': 5000, 'maxls': 50})
            force = solved.x; iterations = int(solved.nit); optimizer_status = int(solved.status); function_calls = int(solved.nfev)
            active = np.flatnonzero(force > 0.)
            if len(active):
                values = np.linalg.lstsq(compliance[np.ix_(active, active)], deficit[active],
                                        rcond=64 * np.finfo(float).eps * len(active))[0]
                polished = np.zeros(m); polished[active] = values
                gap = compliance @ polished - deficit
                if (np.min(polished, initial=0.) >= 0 and np.min(gap, initial=0.) >= -GAP_MM
                        and np.max(np.abs(polished * gap), initial=0.) <= BALANCE_NMM):
                    force = polished
        u = np.zeros(n); u[self.free] = base + response @ force
        gap = self.contact @ u - b
        z = (self.times[:, None] * z0 + dt_s * u) / (self.times[:, None] + dt_s)
        residual = self.matrices[0] @ u - self.contact.T @ force
        for k, v in zip(self.matrices[1:], z):
            residual += k @ (u - v)
        equilibrium = float(np.max(np.abs(residual[self.free]), initial=0.))
        complementarity = float(np.max(np.abs(force * gap), initial=0.))
        if (not all(np.isfinite(v).all() for v in (u, z, force, gap))
                or np.min(force, initial=0.) < -FORCE_N or np.min(gap, initial=0.) < -GAP_MM
                or equilibrium > FORCE_N or complementarity > BALANCE_NMM):
            raise ValueError('PLANAR_CONTACT_UNCONVERGED_KKT')
        element_u = u.reshape(-1, 3)[self.mesh.element_nodes]
        gradient = np.einsum('eai,gaj->egij', element_u, self.gradients)
        maximum_gradient = float(np.linalg.norm(gradient, axis=(2, 3)).max(initial=0.))
        if maximum_gradient > min(.2, self.scene.properties[continuum.PROPERTY]['maxElementGradientNorm']):
            raise ValueError('PLANAR_CONTACT_SMALL_GRADIENT_LIMIT')
        # This is a finite REFERENCE footprint. Measure lateral migration rather
        # than silently claiming a current-configuration edge contact algorithm.
        displacement_at_points = np.asarray([shape @ u.reshape(-1, 3)[self.mesh.element_nodes[row]]
                                            for row, shape in self._shape_rows])
        current = self.points_mm + displacement_at_points - self.blade.origin_mm
        uv = np.column_stack((current @ self.blade.axis_u, current @ self.blade.axis_v))
        migration = np.maximum(np.abs(uv) - self.blade.size_mm / 2, 0.).max(axis=1)
        def energy(x, history):
            return float(.5 * x @ (self.matrices[0] @ x) + sum(
                .5 * (x - v) @ (k @ (x - v)) for k, v in zip(self.matrices[1:], history)))
        before_energy, stored = energy(u0, z0), energy(u, z)
        delta = u - u0
        numerical = float(.5 * delta @ (self.matrices[0] @ delta)); viscous = 0.
        for tau, k, v, old in zip(self.times, self.matrices[1:], z, z0):
            dz = v - old; elastic_delta = (u - v) - (u0 - old)
            viscous += float(tau / dt_s * dz @ (k @ dz))
            numerical += float(.5 * elastic_delta @ (k @ elastic_delta))
        work = float(force @ (b - old_b)); activation = float(force @ old_gap)
        balance = work - (stored - before_energy) - viscous - numerical - activation
        if (not np.isfinite([before_energy, stored, viscous, numerical, work, activation, balance]).all()
                or abs(balance) > BALANCE_NMM or min(viscous, numerical, activation) < -BALANCE_NMM):
            raise ValueError('PLANAR_CONTACT_ENERGY_BALANCE')
        nodal_force = np.asarray(self.contact.T @ force)
        vector_force = force[:, None] * self.blade.inward_normal
        tangential = vector_force - (vector_force @ self.blade.inward_normal)[:, None] * self.blade.inward_normal
        receipt = {'schema': SCHEMA, 'modelId': self.identity,
            'reactionOnBladeN': (-vector_force.sum(axis=0)).tolist(), 'normalReactionN': float(force.sum()),
            'contactAreaMm2': float(self.weights_mm2[gap <= GAP_MM].sum()),
            'footprintAreaMm2': float(self.weights_mm2.sum()),
            'minimumGapMm': float(gap.min()), 'maxPenetrationMm': float(max(0., -gap.min())),
            'maxTensileForceN': float(max(0., -force.min())),
            'maxTangentialForceN': float(np.linalg.norm(tangential, axis=1).max(initial=0.)),
            'maxComplementarityNmm': complementarity, 'maxEquilibriumResidualN': equilibrium,
            'actuatorWorkNmm': work, 'storedEnergyNmm': stored, 'previousStoredEnergyNmm': before_energy,
            'viscousDissipationNmm': viscous, 'numericalDissipationNmm': numerical,
            'contactActivationDissipationNmm': activation, 'energyBalanceResidualNmm': balance,
            'maxElementGradientNorm': maximum_gradient, 'contactPoints': m,
            'gaussPoints': int((~self.guard_rows).sum()), 'cornerGuardRows': int(self.guard_rows.sum()),
            'optimizerIterations': iterations, 'optimizerStatus': optimizer_status,
            'optimizerFunctionCalls': function_calls,
            'maxReferenceFootprintMigrationMm': float(migration.max(initial=0.)),
            'maxLoadedFootprintMigrationMm': float(migration[force > FORCE_N].max(initial=0.)),
            'footprintConvention': 'fixed-reference-rectangle-small-strain',
            'currentConfigurationEdgeContactValidated': False,
            'wallSeconds': time.perf_counter() - started,
            'contactSourceIds': sorted(set(map(int, self.source_ids[gap <= GAP_MM]))),
            'clinicalValidation': False, 'patientPredictionValidation': False, 'physiologicalRL': 'hold'}
        guard()
        self._require_binding(); self.scene.require_cache_source()
        return {'state': ContactState(self.identity, _owned(u), _owned(z), float(offset_mm), state.steps + 1),
                'receipt': receipt, 'normal_forces_n': _owned(force), 'gaps_mm': _owned(gap),
                'contact_nodal_force_n': _owned(nodal_force)}
