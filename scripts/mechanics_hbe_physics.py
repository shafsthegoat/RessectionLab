"""Independent hex8 readout for the declared HBE baseline; never solves mechanics.

Inputs are physical coordinates and raw reactions, not a solver's error summary.
Eight-point quadrature implements the declared three-field, zero-augmentation
energy. Sampled positive Jacobians do not prove positivity everywhere in a hex.
"""
from __future__ import annotations

import hashlib
import itertools
import json
import math

import numpy as np

SIGNS = np.array([[-1,-1,-1],[1,-1,-1],[1,1,-1],[-1,1,-1],
                  [-1,-1,1],[1,-1,1],[1,1,1],[-1,1,1]], dtype=float)
GAUSS = np.array(list(itertools.product((-1/math.sqrt(3), 1/math.sqrt(3)), repeat=3)))
SITES = np.vstack((GAUSS, SIGNS, np.zeros((1, 3))))
K_OVER_MU = 149/3


def positive_finite(value):
    value = float(value)
    if not math.isfinite(value) or value <= 0:
        raise ValueError("Positive finite physical scale required")
    return value


def physical_product(*values):
    return positive_finite(math.prod(values))


def finite_array(value, shape=None):
    result = np.array(value, dtype=np.float64, copy=True)
    if not np.isfinite(result).all() or (shape is not None and result.shape != shape):
        raise ValueError("Complete finite array with declared shape required")
    return result


def shape_functions(local):
    p = finite_array(local, (3,))
    return np.prod(1 + SIGNS*p, axis=1)/8


def shape_derivatives(local):
    p = finite_array(local, (3,))
    return np.column_stack([SIGNS[:, a]*np.prod(1+SIGNS[:, [b for b in range(3) if b != a]]*
                           p[[b for b in range(3) if b != a]], axis=1)/8 for a in range(3)])


DERIVATIVES = np.array([shape_derivatives(p) for p in SITES])


class HexMesh:
    """Own a private snapshot of canonical one-based FEBio mesh data."""

    def __setattr__(self, name, value):
        if getattr(self, '_sealed', False):
            raise AttributeError("Mesh snapshot is immutable")
        object.__setattr__(self, name, value)

    def __init__(self, rest_nodes_m, elements_hex8, top_node_ids, bottom_node_ids, *, radius_m, height_m):
        nodes = finite_array(rest_nodes_m)
        elements_raw = np.asarray(elements_hex8)
        if nodes.ndim != 2 or nodes.shape[1] != 3 or len(nodes) < 8:
            raise ValueError("Expected N by 3 rest nodes")
        if elements_raw.ndim != 2 or elements_raw.shape[1] != 8 or not len(elements_raw):
            raise ValueError("Expected M by 8 canonical hex8 connectivity")
        if not np.issubdtype(elements_raw.dtype, np.integer):
            raise ValueError("Integer one-based node IDs required")
        cells = np.array(elements_raw, dtype=np.int64, copy=True)-1
        if cells.min() < 0 or cells.max() >= len(nodes) or any(len(set(row)) != 8 for row in cells):
            raise ValueError("Invalid or repeated hex node IDs")
        if len({tuple(row) for row in cells}) != len(cells):
            raise ValueError("Duplicate hex elements")
        if not np.array_equal(np.unique(cells), np.arange(len(nodes))):
            raise ValueError("Mesh contains unreferenced nodes")
        self.radius_m, self.height_m = float(radius_m), float(height_m)
        if not all(math.isfinite(v) and v > 0 for v in (self.radius_m, self.height_m)):
            raise ValueError("Positive physical dimensions required")
        top, bottom = self._ids(top_node_ids, len(nodes)), self._ids(bottom_node_ids, len(nodes))
        if set(top) & set(bottom):
            raise ValueError("Top and bottom overlap")
        tolerance = 1e-10*max(self.radius_m, self.height_m)
        if np.max(np.abs(nodes[top, 2]-self.height_m)) > tolerance or np.max(np.abs(nodes[bottom, 2])) > tolerance:
            raise ValueError("Boundary IDs do not match declared end planes")
        if set(top) != set(np.flatnonzero(np.abs(nodes[:, 2]-self.height_m) <= tolerance)) or set(bottom) != set(np.flatnonzero(np.abs(nodes[:, 2]) <= tolerance)):
            raise ValueError("Missing end-face nodes")
        rest_maps = np.einsum('mni,snj->msij', nodes[cells], DERIVATIVES)
        determinants = np.linalg.det(rest_maps)
        if np.any(determinants <= 0) or not np.isfinite(determinants).all():
            raise ValueError("Nonpositive rest Jacobian")
        scaled = determinants/np.prod(np.linalg.norm(rest_maps, axis=-2), axis=-1)
        if np.min(scaled) <= .1 or not np.isfinite(scaled).all():
            raise ValueError("Rest scaled Jacobian fails declared gate")
        # Bytes-backed buffers cannot be made writeable; external input edits do not leak in.
        self._nodes = np.frombuffer(nodes.tobytes(), dtype=np.float64).reshape(nodes.shape)
        self._cells = np.frombuffer(cells.tobytes(), dtype=np.int64).reshape(cells.shape)
        self._top, self._bottom = tuple(top), tuple(bottom)
        self._rest_inverse = np.linalg.inv(rest_maps)
        self._weights = determinants[:, :8]
        self._volumes = self._weights.sum(axis=1)
        self._bbox_min, self._bbox_max = nodes[cells].min(axis=1), nodes[cells].max(axis=1)
        payload = dict(nodes=nodes.tolist(), cells=cells.tolist(), top=list(top), bottom=list(bottom),
                       radius=self.radius_m, height=self.height_m)
        self.fingerprint = hashlib.sha256(json.dumps(payload, sort_keys=True, separators=(',', ':')).encode()).hexdigest()
        self._sealed = True

    @staticmethod
    def _ids(values, count):
        a = np.asarray(values)
        if a.ndim != 1 or not len(a) or not np.issubdtype(a.dtype, np.integer):
            raise ValueError("Nonempty integer boundary IDs required")
        if len(set(map(int, a))) != len(a) or a.min() < 1 or a.max() > count:
            raise ValueError("Duplicate or invalid boundary IDs")
        return tuple(int(i)-1 for i in a)

    @classmethod
    def from_manifest(cls, mesh):
        if mesh.get('indexing', 'one_based') != 'one_based':
            raise ValueError("Expected one-based persisted mesh")
        nodes = mesh['rest_nodes_m']
        if mesh['node_ids'] != list(range(1, len(nodes)+1)):
            raise ValueError("Unordered/noncontiguous node IDs")
        if mesh['element_ids'] != list(range(1, len(mesh['elements_hex8'])+1)):
            raise ValueError("Unordered/noncontiguous element IDs")
        return cls(nodes, mesh['elements_hex8'], mesh['boundaries']['top']['node_ids'],
                   mesh['boundaries']['bottom']['node_ids'], radius_m=mesh['geometry']['radius_m'],
                   height_m=mesh['geometry']['height_m'])

    @property
    def rest_nodes_m(self):
        return self._nodes.copy()

    @property
    def element_count(self):
        return len(self._cells)

    @property
    def rest_volume_m3(self):
        return float(self._volumes.sum())

    def deformation(self, current_nodes_m, mu_Pa):
        current = finite_array(current_nodes_m, self._nodes.shape)
        mu = float(mu_Pa)
        if not math.isfinite(mu) or mu <= 0:
            raise ValueError("Positive finite stiffness required")
        moved_maps = np.einsum('mni,snj->msij', current[self._cells], DERIVATIVES)
        gradients = moved_maps @ self._rest_inverse
        jacobians = np.linalg.det(gradients)
        if np.any(jacobians <= 0) or not np.isfinite(jacobians).all():
            raise ValueError("Nonpositive or nonfinite deformed Jacobian")
        J = jacobians[:, :8]
        Jbar = np.sum(J*self._weights, axis=1)/self._volumes
        Ibar = np.sum(gradients[:, :8]**2, axis=(-1, -2))*J**(-2/3)
        dev = mu/2*(Ibar-3)
        # FEBio three-field pressure uses the reference-volume-weighted cell J.
        volumetric = K_OVER_MU*mu/4*(Jbar**2-1-2*np.log(Jbar))
        energy = np.sum(dev*self._weights, axis=1)+self._volumes*volumetric
        if not np.isfinite(energy).all():
            raise ValueError("Nonfinite reconstructed energy")
        return {'minimum_sampled_J': float(jacobians.min()), 'cell_volume_ratios': Jbar,
                'energy_J': float(energy.sum()), 'cell_energy_J': energy}

    def probe_map(self, points_m):
        """Rest-coordinate inverse map; ties use first canonical element ID."""
        points = finite_array(points_m)
        if points.ndim != 2 or points.shape[1] != 3:
            raise ValueError("Expected physical probe coordinates")
        tolerance = 1e-10*max(self.radius_m, self.height_m)
        result = []
        for point in points:
            candidates = np.flatnonzero(np.all(point >= self._bbox_min-tolerance, axis=1) &
                                       np.all(point <= self._bbox_max+tolerance, axis=1))
            accepted = None
            for index in candidates:
                X = self._nodes[self._cells[index]]
                local = np.zeros(3)
                for _ in range(24):
                    residual = shape_functions(local) @ X-point
                    if np.linalg.norm(residual) <= tolerance:
                        break
                    try:
                        local -= np.linalg.solve(X.T @ shape_derivatives(local), residual)
                    except np.linalg.LinAlgError:
                        break
                    if not np.isfinite(local).all() or np.max(np.abs(local)) > 8:
                        break
                if np.isfinite(local).all() and np.max(np.abs(local)) <= 1+1e-8 and np.linalg.norm(shape_functions(local) @ X-point) <= tolerance:
                    accepted = (int(index), tuple(shape_functions(local)))
                    break
            if accepted is None:
                raise ValueError("Probe outside mesh or inverse map failed; no zero substitution")
            result.append(accepted)
        return tuple(result)

    def interpolate_displacement(self, current_nodes_m, probe_map):
        displacement = finite_array(current_nodes_m, self._nodes.shape)-self._nodes
        return np.array([np.asarray(weights) @ displacement[self._cells[index]] for index, weights in probe_map])

    def read_frame(self, current_nodes_m, raw_reactions_N, *, mu_Pa, branch, fraction):
        mu_Pa = positive_finite(mu_Pa)
        current = finite_array(current_nodes_m, self._nodes.shape)
        reaction = finite_array(raw_reactions_N, self._nodes.shape)
        if not math.isfinite(fraction) or not 0 <= fraction <= 1:
            raise ValueError("Invalid load fraction")
        top = np.array(self._top)
        prescribed = self._nodes.copy()
        if branch in ('compression', 'tension'):
            amount = (-1 if branch == 'compression' else 1)*.15*self.height_m*fraction
            prescribed[top, 2] += amount
        elif branch in ('torsion_neg', 'torsion_pos'):
            amount = (-1 if branch == 'torsion_neg' else 1)*.15*self.height_m/self.radius_m*fraction
            c, s = math.cos(amount), math.sin(amount)
            prescribed[top, :2] = self._nodes[top, :2] @ np.array([[c, s], [-s, c]])
        else:
            raise ValueError("Unknown frozen branch")
        boundary = np.array(sorted(set(self._top) | set(self._bottom)))
        moments = np.cross(current, reaction)
        net_force, net_moment = reaction.sum(axis=0), moments.sum(axis=0)
        F0, T0 = physical_product(mu_Pa,self.radius_m,self.radius_m), physical_product(mu_Pa,self.radius_m,self.radius_m,self.radius_m)
        force_tolerance = positive_finite(1e-8*F0+1e-5*np.linalg.norm(reaction, axis=1).sum())
        moment_tolerance = positive_finite(1e-8*T0+1e-5*np.linalg.norm(moments, axis=1).sum())
        boundary_error = float(np.max(np.linalg.norm(current[boundary]-prescribed[boundary], axis=1)))
        free = np.array(sorted(set(range(len(current)))-set(boundary)), dtype=int)
        free_reaction = float(np.max(np.linalg.norm(reaction[free], axis=1))) if len(free) else 0.
        result = self.deformation(current, mu_Pa)
        result.update(fraction=float(fraction), input_coordinate=float(amount),
                      applied_force_N=float(-reaction[top, 2].sum()),
                      applied_torque_Nm=float(-moments[top, 2].sum()),
                      net_force_N=net_force.tolist(), net_moment_Nm=net_moment.tolist(),
                      prescribed_error_m=boundary_error, free_node_reaction_max_N=free_reaction)
        result['checks'] = {'force_balance': bool(np.linalg.norm(net_force) <= force_tolerance),
                            'moment_balance': bool(np.linalg.norm(net_moment) <= moment_tolerance),
                            'prescribed_motion': boundary_error <= 1e-8*self.radius_m,
                            'free_node_reactions': free_reaction <= 1e-8*F0}
        result['passed'] = all(result['checks'].values())
        return result


def fixed_probes(radius_m, height_m):
    points = []
    for z in (.25, .5, .75):
        points.append([0, 0, z*height_m])
        for r in (.25, .5, .75):
            points.extend([[radius_m*r*math.cos(j*math.pi/4), radius_m*r*math.sin(j*math.pi/4), z*height_m] for j in range(8)])
    return np.array(points)


def energy_work_check(coordinate, applied_response, energy_J, *, mu_Pa, radius_m, height_m):
    mu_Pa, radius_m, height_m = map(positive_finite, (mu_Pa, radius_m, height_m))
    x, y, energy = map(finite_array, (coordinate, applied_response, energy_J))
    if x.ndim != 1 or len(x) < 2 or y.shape != x.shape or energy.shape != x.shape:
        raise ValueError("Complete work/energy sequence required")
    dx = np.diff(x)
    if not (np.all(dx > 0) or np.all(dx < 0)) or x[0] != 0:
        raise ValueError("Sequence must start at rest and progress monotonically")
    work = np.concatenate(([0.], np.cumsum(dx*(y[1:]+y[:-1])/2)))
    delta_energy = energy-energy[0]
    work, delta_energy = map(finite_array, (work, delta_energy))
    E0 = physical_product(mu_Pa,radius_m,radius_m,height_m)
    limit = positive_finite(1e-6*E0+.02*max(float(np.max(np.abs(work))), float(np.max(np.abs(delta_energy)))))
    maximum_error = float(np.max(np.abs(work-delta_energy)))
    return {'passed': maximum_error <= limit, 'maximum_error_J': maximum_error,
            'limit_J': limit, 'work_J': work.tolist(), 'energy_change_J': delta_energy.tolist()}


def compare_refinement(coarse, medium, fine, *, response_scale, radius_m, kind='mesh'):
    """Arrays already sampled at exactly the same physical load/probe points."""
    response_scale, radius_m = map(positive_finite, (response_scale, radius_m))
    if kind not in ('mesh', 'step'):
        raise ValueError("Unknown refinement kind")
    outputs = []
    for value in (coarse, medium, fine):
        response, probes = finite_array(value['response']), finite_array(value['probe_displacements_m'])
        if response.ndim != 1 or probes.shape != (len(response), 75, 3):
            raise ValueError("Common 75-probe/load-state arrays required")
        outputs.append((response, probes))
    if any(a.shape != outputs[0][0].shape for a, _ in outputs):
        raise ValueError("Load grids differ")
    (r0,p0),(r1,p1),(r2,p2) = outputs
    coarse_change, fine_change = float(np.max(np.abs(r1-r0))), float(np.max(np.abs(r2-r1)))
    coarse_motion = float(np.max(np.linalg.norm(p1-p0, axis=-1)))
    fine_motion = float(np.max(np.linalg.norm(p2-p1, axis=-1)))
    relative, floor_factor, motion_factor = (.02,1e-5,.002) if kind == 'mesh' else (.002,1e-6,.0002)
    floor = physical_product(floor_factor,response_scale)
    response_limit = positive_finite(floor+relative*float(np.max(np.abs(r2))))
    motion_limit = physical_product(motion_factor,radius_m)
    # Step comparison uses base/base/refined; monotone trend only has three mesh levels.
    trend = kind == 'step' or fine_change < coarse_change or max(fine_change,coarse_change) <= floor
    motion_trend = kind == 'step' or fine_motion < coarse_motion or max(fine_motion,coarse_motion) <= motion_limit
    checks = {'response': fine_change <= response_limit, 'motion': fine_motion <= motion_limit,
              'response_trend': trend, 'motion_trend': motion_trend}
    return {'passed': all(checks.values()), 'checks': checks, 'response_change': fine_change,
            'response_limit': response_limit, 'motion_change_m': fine_motion,
            'motion_limit_m': motion_limit, 'coarse_response_change': coarse_change,
            'coarse_motion_change_m': coarse_motion}


def compare_scale(mesh, base_current, doubled_current, base_reactions, doubled_reactions, *, mu_Pa):
    mu_Pa = positive_finite(mu_Pa)
    x1,x2,r1,r2 = map(finite_array, (base_current,doubled_current,base_reactions,doubled_reactions))
    if any(a.shape != x1.shape for a in (x2,r1,r2)) or x1.ndim < 2 or x1.shape[-1] != 3:
        raise ValueError("Scale pair primitive shapes differ")
    if x1.shape[-2:] != mesh._nodes.shape:
        raise ValueError("Scale pair mesh shape differs")
    radius_m = mesh.radius_m
    F0 = physical_product(2,mu_Pa,radius_m,radius_m)
    T0 = physical_product(2,mu_Pa,radius_m,radius_m,radius_m)
    motion_limit = positive_finite(1e-6*radius_m+1e-5*float(np.max(np.abs(x1-mesh._nodes))))
    reaction_limit = finite_array(1e-7*F0+1e-4*np.abs(2*r1))
    top = np.array(mesh._top)
    torque1 = -np.cross(x1[..., top, :], r1[..., top, :])[..., 2].sum(axis=-1)
    torque2 = -np.cross(x2[..., top, :], r2[..., top, :])[..., 2].sum(axis=-1)
    torque_limit = finite_array(1e-7*T0+1e-4*np.abs(2*torque1))
    return {'passed': bool(np.max(np.abs(x2-x1)) <= motion_limit and np.all(np.abs(r2-2*r1) <= reaction_limit) and np.all(np.abs(torque2-2*torque1) <= torque_limit)),
            'motion_max_error_m': float(np.max(np.abs(x2-x1))),
            'reaction_max_error_N': float(np.max(np.abs(r2-2*r1))),
            'torque_max_error_Nm': float(np.max(np.abs(torque2-2*torque1)))}
