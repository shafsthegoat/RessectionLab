"""Independent lower-half axial reconstruction; never solves or reads HBE curves.

The upper displacement is (ux, uy, d-uz), not a reflected displacement vector.
Raw nodal reactions are vectors and are summed where the two halves meet.
Reconstructed fields are never represented as a native full-cylinder solve.
"""
from __future__ import annotations

from contextlib import ExitStack
from copy import deepcopy
import hashlib
import itertools
import json
import math

import numpy as np

from scripts.mechanics_hbe_access import local_path, verify_binding
from scripts.mechanics_hbe_outputs import iter_data_records, check_solver_records
from scripts.mechanics_hbe_physics import HexMesh, SIGNS, finite_array, fixed_probes, energy_work_check
from scripts.mechanics_hbe_readout import ORIGINAL_PROTOCOL_SHA256, read_run


DECLARATION_SHA256 = '0df5587ab7a70eb1ac092919ff067b4d85ce6c7457af9726875beb58af19ad1e'
RADIUS_M = .004
FULL_HEIGHT_M = .00489159
MU_PA = 1000.
TIMES = np.linspace(0, 1, 61)
PRIMITIVE_KEYS = {'mesh', 'deck', 'loading', 'nodes', 'elements', 'solver'}


def _fingerprint(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()).hexdigest()


def _ids(value, count, length, name):
    raw = np.asarray(value)
    if (raw.shape != (length,) or not np.issubdtype(raw.dtype, np.integer)
            or np.any(raw < 1) or np.any(raw > count)
            or len(np.unique(raw)) != length):
        raise ValueError(f'Invalid one-based {name} mapping')
    return np.array(raw - 1, dtype=np.int64)


class HalfHeightReconstruction:
    """Own and validate exact node/cell correspondences before reading states."""

    def __init__(self, full_manifest, half_manifest, mapping):
        if (mapping.get('full_mesh_fingerprint') != _fingerprint(full_manifest)
                or mapping.get('half_mesh_fingerprint') != _fingerprint(half_manifest)):
            raise ValueError('Mapping does not bind the complete original and half mesh manifests')
        self.full = HexMesh.from_manifest(full_manifest)
        self.half = HexMesh.from_manifest(half_manifest)
        self.X = self.full.rest_nodes_m
        self.Xh = self.half.rest_nodes_m
        self.height = self.full.height_m
        self.radius = self.full.radius_m
        self.tolerance = 1e-10 * max(self.height, self.radius)
        if (self.half.height_m != self.height / 2
                or self.half.radius_m != self.radius
                or mapping.get('schema') != 'hbe-halfheight-mapping-v1'):
            raise ValueError('A declared lower half of the original geometry is required')
        if (np.any(self.Xh[:, 2] < -self.tolerance)
                or np.any(self.Xh[:, 2] > self.height / 2 + self.tolerance)):
            raise ValueError('Half mesh extends outside the retained lower domain')
        cells = np.asarray(full_manifest['elements_hex8'], dtype=np.int64) - 1
        half_cells = np.asarray(half_manifest['elements_hex8'], dtype=np.int64) - 1
        self.bottom = np.array(HexMesh._ids(half_manifest['boundaries']['bottom']['node_ids'], len(self.Xh)))
        self.midplane = np.array(HexMesh._ids(half_manifest['boundaries']['top']['node_ids'], len(self.Xh)))
        records = mapping.get('reflections')
        if not isinstance(records, list) or len(records) != 2:
            raise ValueError('Exactly lower and upper reconstruction records required')
        self.node_maps = []
        self.cell_maps = []
        coordinate_errors = []
        for index, (record, signs, offset) in enumerate(zip(
                records, ((1, 1, 1), (1, 1, -1)), ((0., 0., 0.), (0., 0., self.height)))):
            if record.get('signs') != list(signs) or record.get('rest_offset_m') != list(offset):
                raise ValueError('Reconstruction must use the fixed horizontal midplane')
            node_map = _ids(record.get('node_full_ids'), len(self.X), len(self.Xh), 'node')
            cell_map = _ids(record.get('element_full_ids'), len(cells), len(half_cells), 'element')
            local = np.asarray(record.get('element_half_to_full_local'))
            if (local.shape != half_cells.shape or not np.issubdtype(local.dtype, np.integer)
                    or not np.all(np.sort(local, axis=1) == np.arange(8))):
                raise ValueError('Every reflected hex requires its exact local permutation')
            target = SIGNS[local]
            transforms = np.einsum('ni,mnj->mij', SIGNS, target) / 8
            if (not np.array_equal(np.einsum('ni,mij->mnj', SIGNS, transforms), target)
                    or not np.all(transforms @ np.swapaxes(transforms, 1, 2) == np.eye(3))
                    or not np.all(np.linalg.det(transforms) == np.prod(signs))):
                raise ValueError('Local mapping is not the required oriented cube symmetry')
            mapped_connectivity = np.take_along_axis(cells[cell_map], local, axis=1)
            if not np.array_equal(mapped_connectivity, node_map[half_cells]):
                raise ValueError('Reflected connectivity does not identify the original full cells')
            reflected = self.Xh * np.asarray(signs) + np.asarray(offset)
            error = float(np.max(np.linalg.norm(reflected - self.X[node_map], axis=1)))
            if error > self.tolerance:
                raise ValueError('Original rest coordinates fail the frozen reflection tolerance')
            if index == 0:
                if not np.array_equal(self.Xh, self.X[node_map]):
                    raise ValueError('Retained original coordinates were modified')
                if (mapping.get('half_to_full_node_ids') != (node_map + 1).tolist()
                        or mapping.get('half_to_full_element_ids') != (cell_map + 1).tolist()):
                    raise ValueError('Retained-half IDs disagree with the identity map')
            self.node_maps.append(node_map)
            self.cell_maps.append(cell_map)
            coordinate_errors.append(error)
        counts = np.bincount(np.concatenate(self.node_maps), minlength=len(self.X))
        mid = np.abs(self.X[:, 2] - self.height / 2) <= self.tolerance
        if not np.array_equal(counts, np.where(mid, 2, 1)):
            raise ValueError('Node coverage/overlap differs from the single shared midplane')
        if not np.array_equal(np.sort(np.concatenate(self.cell_maps)), np.arange(len(cells))):
            raise ValueError('Each full cell must be reconstructed exactly once')
        self.maximum_coordinate_mismatch_m = max(coordinate_errors)

    def lift(self, current_nodes_m, raw_reactions_N, elements, full_displacement_m):
        """Return full current nodes, summed raw reactions and reflected logs."""
        current = finite_array(current_nodes_m, self.Xh.shape)
        raw = finite_array(raw_reactions_N, self.Xh.shape)
        logged = finite_array(elements, (self.half.element_count, 8))
        d = float(full_displacement_m)
        if not math.isfinite(d):
            raise ValueError('Finite original full displacement required')
        u = current - self.Xh
        lifted_u = np.empty_like(self.X)
        lifted_raw = np.zeros_like(self.X)
        lifted_elements = np.empty((self.full.element_count, 8))
        assigned = np.zeros(len(self.X), dtype=bool)
        duplicate_error = 0.
        for index, (node_map, cell_map) in enumerate(zip(self.node_maps, self.cell_maps)):
            moved_u = u.copy()
            moved_raw = raw.copy()
            moved_elements = logged.copy()
            if index:
                moved_u[:, 2] = d - u[:, 2]
                moved_raw[:, 2] *= -1
                # Cauchy components are xx,yy,zz,xy,yz,xz; Q sigma Q^T.
                moved_elements[:, 4:6] *= -1
            shared = assigned[node_map]
            if shared.any():
                error = float(np.max(np.linalg.norm(lifted_u[node_map[shared]] - moved_u[shared], axis=1)))
                duplicate_error = max(duplicate_error, error)
                if error > self.tolerance:
                    raise ValueError('Shared midplane displacement contributions disagree')
            lifted_u[node_map[~shared]] = moved_u[~shared]
            np.add.at(lifted_raw, node_map, moved_raw)
            assigned[node_map] = True
            lifted_elements[cell_map] = moved_elements
        return {
            'current_nodes_m': self.X + lifted_u,
            'raw_reactions_N': lifted_raw,
            'logged_elements': lifted_elements,
            'maximum_shared_displacement_mismatch_m': duplicate_error,
        }

    def half_frame(self, current, raw, *, full_displacement_m):
        """Evaluate genuine half-domain BCs; midplane x/y remain free."""
        current = finite_array(current, self.Xh.shape)
        raw = finite_array(raw, self.Xh.shape)
        d = float(full_displacement_m)
        if not math.isfinite(d):
            raise ValueError('Finite original full displacement required')
        bottom_error = np.linalg.norm(current[self.bottom] - self.Xh[self.bottom], axis=1).max()
        mid_error = np.abs(current[self.midplane, 2] - self.Xh[self.midplane, 2] - d / 2).max()
        free_raw = raw.copy()
        free_raw[self.bottom] = 0
        free_raw[self.midplane, 2] = 0
        F0 = MU_PA * self.radius ** 2
        T0 = F0 * self.radius
        moments = np.cross(current, raw)
        ratios = {
            'force_balance': float(np.linalg.norm(raw.sum(axis=0)) / (1e-8 * F0 + 1e-5 * np.linalg.norm(raw, axis=1).sum())),
            'moment_balance': float(np.linalg.norm(moments.sum(axis=0)) / (1e-8 * T0 + 1e-5 * np.linalg.norm(moments, axis=1).sum())),
            'prescribed_motion': float(max(bottom_error, mid_error) / (1e-8 * self.radius)),
            'free_dof_reaction': float(np.linalg.norm(free_raw, axis=1).max() / (1e-8 * F0)),
        }
        return {
            **self.half.deformation(current, MU_PA),
            'force_from_bottom_N': float(raw[self.bottom, 2].sum()),
            'force_from_midplane_N': float(-raw[self.midplane, 2].sum()),
            'ratios': ratios,
        }


def _maxima(target, values):
    for key, value in values.items():
        value = float(value)
        if not math.isfinite(value):
            raise ValueError('Nonfinite numerical comparison')
        target[key] = max(target.get(key, 0.), value)


def _criteria(ratios, minimum_J):
    result = {name: {'actual': float(value), 'limit': 1., 'units': 'ratio_to_declared_numerical_tolerance'}
              for name, value in ratios.items()}
    result['minimum_sampled_J'] = {'actual': float(minimum_J), 'limit': 0., 'units': 'dimensionless_strictly_positive'}
    return result


def _frame_ratios(mesh, state, raw):
    R = mesh.radius_m
    F0 = MU_PA * R ** 2
    T0 = F0 * R
    moments = np.cross(state['current_nodes_m'], raw)
    return {
        'force_balance': np.linalg.norm(state['net_force_N']) / (1e-8 * F0 + 1e-5 * np.linalg.norm(raw, axis=1).sum()),
        'moment_balance': np.linalg.norm(state['net_moment_Nm']) / (1e-8 * T0 + 1e-5 * np.linalg.norm(moments, axis=1).sum()),
        'prescribed_motion': state['prescribed_error_m'] / (1e-8 * R),
        'free_node_reaction': state['free_node_reaction_max_N'] / (1e-8 * F0),
    }


def _records(stack, root, binding, count, fields, name):
    stream = stack.enter_context(local_path(root, binding['path']).open(encoding='utf-8', errors='strict'))
    return iter_data_records(stream, expected_times=TIMES, item_count=count, field_count=fields, record_name=name)


def _load_contract(root, half_bindings, full_bindings, reconstruction_binding,
                   declaration_binding, expected_branch, expected_mesh_N):
    if (DECLARATION_SHA256 is None or declaration_binding.get('sha256') != DECLARATION_SHA256
            or expected_branch not in ('compression', 'tension')
            or type(expected_mesh_N) is not int or expected_mesh_N not in (8, 12)):
        raise ValueError('Exact separately declared four-case half-height study required')
    declaration = verify_binding(root, declaration_binding, maximum_bytes=1024**2, read_json=True)
    if declaration.get('schema') != 'hbe-halfheight-equivalence-v1':
        raise ValueError('Wrong half-height declaration schema')
    run_id = f'{expected_branch}:N{expected_mesh_N}:S60:reference'
    original = declaration['full_references'][run_id]
    if original['primitive_bindings'] != full_bindings:
        raise ValueError('Full comparison inputs differ from the prospectively selected native run')
    if set(half_bindings) != PRIMITIVE_KEYS or set(full_bindings) != PRIMITIVE_KEYS:
        raise ValueError('Exactly six bound half and full inputs required')
    all_bindings = (list(half_bindings.values()) + list(full_bindings.values())
                    + [reconstruction_binding, declaration_binding, original['readout'], original['execution'],
                       declaration['original_protocol']])
    for binding in all_bindings:
        verify_binding(root, binding)
    wrapper = verify_binding(root, reconstruction_binding, maximum_bytes=16 * 1024**2, read_json=True)
    if (wrapper.get('schema') != 'hbe-halfheight-reconstruction-v1'
            or wrapper.get('full_mesh') != full_bindings['mesh']
            or wrapper.get('half_mesh') != half_bindings['mesh']
            or wrapper.get('declaration') != declaration_binding):
        raise ValueError('Reconstruction provenance differs from the actual mesh/declaration inputs')
    verify_binding(root, wrapper['mesh_source'], maximum_bytes=1024**2)
    all_bindings.append(wrapper['mesh_source'])
    full = verify_binding(root, full_bindings['mesh'], maximum_bytes=16 * 1024**2, read_json=True)
    half = verify_binding(root, half_bindings['mesh'], maximum_bytes=16 * 1024**2, read_json=True)
    loading = verify_binding(root, half_bindings['loading'], maximum_bytes=1024**2, read_json=True)
    expected = {
        'branch': expected_branch, 'steps': 60, 'mu_Pa': MU_PA,
        'protocol_sha256': ORIGINAL_PROTOCOL_SHA256,
        'halfheight_equivalence_declaration_sha256': DECLARATION_SHA256,
        'mesh_sha256': half_bindings['mesh']['sha256'], 'deck_sha256': half_bindings['deck']['sha256'],
        'full_mesh_sha256': full_bindings['mesh']['sha256'],
        'reconstruction_sha256': reconstruction_binding['sha256'],
        'full_height_m': FULL_HEIGHT_M, 'retained_height_m': FULL_HEIGHT_M / 2,
        'prescribed_dofs': {'bottom': 'xyz', 'midplane': 'z'},
        'node_fields': 'x;y;z;ux;uy;uz;Rx;Ry;Rz',
        'element_fields': 'sx;sy;sz;sxy;syz;sxz;J;sed',
        'raw_reaction_convention': 'body_on_constraint', 'load_coordinate_units': 'm',
        'min_residual_N2': (1e-10 * MU_PA * RADIUS_M ** 2) ** 2,
    }
    if any(loading.get(key) != value for key, value in expected.items()):
        raise ValueError('Half loading identity, fixture or physical convention differs')
    if not math.isclose(loading.get('K_Pa', 0), 149 * MU_PA / 3, rel_tol=1e-14):
        raise ValueError('Declared bulk/shear ratio changed')
    if full.get('mesh_N') != expected_mesh_N or half.get('mesh_N') != expected_mesh_N:
        raise ValueError('Mesh level changed')
    if half.get('full_height_m') != FULL_HEIGHT_M or half.get('full_geometry') != full['geometry']:
        raise ValueError('Original full geometry must be preserved explicitly')
    if half.get('boundary_roles') != {'top': 'artificial_midplane', 'bottom': 'physical_bonded_plate', 'side': 'traction_free_outer'}:
        raise ValueError('Physical plate and artificial midplane must remain distinguished')
    reconstruction = HalfHeightReconstruction(full, half, wrapper['mapping'])
    if reconstruction.radius != RADIUS_M or reconstruction.height != FULL_HEIGHT_M:
        raise ValueError('Original specimen dimensions changed')
    counts = declaration['extraction']['expected_counts'][str(expected_mesh_N)]
    actual_counts = {'full_elements': reconstruction.full.element_count, 'full_nodes': len(reconstruction.X),
                     'half_elements': reconstruction.half.element_count, 'half_nodes': len(reconstruction.Xh),
                     'midplane_nodes': len(reconstruction.midplane)}
    if actual_counts != counts:
        raise ValueError('Exact original/half node, cell and midplane counts required')
    coordinate = TIMES * (-1 if expected_branch == 'compression' else 1) * .15 * FULL_HEIGHT_M
    for name, values, tolerance in (('times', TIMES, 2e-15), ('load_coordinate', coordinate, 1e-18)):
        actual = finite_array(loading.get(name), values.shape)
        if not np.allclose(actual, values, rtol=2e-14 if name == 'load_coordinate' else 0, atol=tolerance):
            raise ValueError('Original full physical loading grid changed')
    return reconstruction, coordinate, all_bindings, declaration


def read_halfheight_equivalence(root, *, half_bindings, full_bindings,
                                reconstruction_binding, declaration_binding,
                                expected_branch, expected_mesh_N):
    """Read saved native states and return numerical/equivalence evidence only.

    Complete original full run checks precede a streaming paired comparison.
    All original bindings are rehashed before return. No native call, inference,
    calibration or measured-data reader exists on this path.
    """
    # Own the prospective selection even if a caller later mutates its mappings.
    half_bindings, full_bindings, reconstruction_binding, declaration_binding = deepcopy(
        (half_bindings, full_bindings, reconstruction_binding, declaration_binding))
    model, coordinate, bindings, declaration = _load_contract(
        root, half_bindings, full_bindings, reconstruction_binding,
        declaration_binding, expected_branch, expected_mesh_N,
    )
    full_receipt, _ = read_run(
        root, full_bindings, protocol_sha256=ORIGINAL_PROTOCOL_SHA256,
        expected_branch=expected_branch, expected_mesh_N=expected_mesh_N,
        expected_steps=60, expected_mu_Pa=MU_PA,
    )
    with local_path(root, half_bindings['solver']['path']).open(encoding='utf-8', errors='strict') as stream:
        half_solver = check_solver_records(stream, expected_times=TIMES,
                                          residual_floor_N2=(1e-10 * MU_PA * RADIUS_M ** 2) ** 2)
    probe_map = model.full.probe_map(fixed_probes(RADIUS_M, FULL_HEIGHT_M))
    half_ratios = {}
    lifted_ratios = {}
    equivalence_ratios = {}
    maxima = {'node_motion_m': 0., 'probe_motion_m': 0., 'full_motion_m': 0.,
              'logged_stress_Pa': 0., 'logged_J': 0., 'logged_sed_Pa': 0.,
              'shared_displacement_mismatch_m': 0.}
    half_energy, half_bottom_force, half_midplane_force = [], [], []
    lifted_energy, lifted_force, lifted_torque, lifted_probes = [], [], [], []
    half_min_J = lifted_min_J = math.inf
    F0 = MU_PA * RADIUS_M ** 2
    E0 = F0 * FULL_HEIGHT_M
    with ExitStack() as stack:
        streams = (
            _records(stack, root, half_bindings['nodes'], len(model.Xh), 9, 'mechanics_nodes_si'),
            _records(stack, root, half_bindings['elements'], model.half.element_count, 8, 'mechanics_elements_si'),
            _records(stack, root, full_bindings['nodes'], len(model.X), 9, 'mechanics_nodes_si'),
            _records(stack, root, full_bindings['elements'], model.full.element_count, 8, 'mechanics_elements_si'),
        )
        for hn, he, fn, fe in itertools.zip_longest(*streams):
            records = (hn, he, fn, fe)
            if any(row is None for row in records) or len({row['step'] for row in records}) != 1:
                raise ValueError('Four complete synchronized primitive streams required')
            index = hn['step']
            if np.any(he['values'][:, 6] <= 0) or np.any(fe['values'][:, 6] <= 0):
                raise ValueError('Nonpositive logged element Jacobian')
            n = hn['values']
            current, raw = n[:, :3], n[:, 6:9]
            half = model.half_frame(current, raw, full_displacement_m=coordinate[index])
            lifted = model.lift(current, raw, he['values'], coordinate[index])
            state = model.full.read_frame(lifted['current_nodes_m'], lifted['raw_reactions_N'],
                                          mu_Pa=MU_PA, branch=expected_branch, fraction=TIMES[index])
            state['current_nodes_m'] = lifted['current_nodes_m']
            _maxima(half_ratios, half['ratios'])
            _maxima(lifted_ratios, _frame_ratios(model.full, state, lifted['raw_reactions_N']))
            primitive_error = float(np.linalg.norm(n[:, 3:6] - (current - model.Xh), axis=1).max()) / (1e-8 * RADIUS_M)
            if index == 0:
                primitive_error = max(primitive_error,
                                      float(np.linalg.norm(current - model.Xh, axis=1).max()) / (1e-8 * RADIUS_M),
                                      float(np.linalg.norm(raw, axis=1).max()) / (1e-8 * F0))
            _maxima(half_ratios, {'primitive_consistency': primitive_error})
            full_current, full_raw = fn['values'][:, :3], fn['values'][:, 6:9]
            probes = model.full.interpolate_displacement(lifted['current_nodes_m'], probe_map)
            full_probes = np.asarray(full_receipt['probe_displacements_m'][index])
            full_force = full_receipt['applied_force_N'][index]
            full_energy = full_receipt['energy_J'][index]
            force_limit = 1e-7 * F0 + 1e-4 * abs(full_force)
            energy_limit = 1e-7 * E0 + 1e-4 * abs(full_energy)
            _maxima(equivalence_ratios, {
                'raw_reaction': np.max(np.abs(lifted['raw_reactions_N'] - full_raw) / (1e-7 * F0 + 1e-4 * np.abs(full_raw))),
                'lifted_axial_force': abs(state['applied_force_N'] - full_force) / force_limit,
                'bottom_axial_force_scale_one': abs(half['force_from_bottom_N'] - full_force) / force_limit,
                'midplane_axial_force_scale_one': abs(half['force_from_midplane_N'] - full_force) / force_limit,
                'lifted_energy': abs(state['energy_J'] - full_energy) / energy_limit,
                'half_energy_scale_two': abs(2 * half['energy_J'] - full_energy) / energy_limit,
            })
            stress_difference = lifted['logged_elements'] - fe['values']
            _maxima(maxima, {
                'node_motion_m': np.linalg.norm(lifted['current_nodes_m'] - full_current, axis=1).max(),
                'probe_motion_m': np.linalg.norm(probes - full_probes, axis=1).max(),
                'full_motion_m': np.linalg.norm(full_current - model.X, axis=1).max(),
                'logged_stress_Pa': np.abs(stress_difference[:, :6]).max(),
                'logged_J': np.abs(stress_difference[:, 6]).max(),
                'logged_sed_Pa': np.abs(stress_difference[:, 7]).max(),
                'shared_displacement_mismatch_m': lifted['maximum_shared_displacement_mismatch_m'],
            })
            half_min_J = min(half_min_J, half['minimum_sampled_J'])
            lifted_min_J = min(lifted_min_J, state['minimum_sampled_J'])
            half_energy.append(half['energy_J'])
            half_bottom_force.append(half['force_from_bottom_N'])
            half_midplane_force.append(half['force_from_midplane_N'])
            lifted_energy.append(state['energy_J'])
            lifted_force.append(state['applied_force_N'])
            lifted_torque.append(state['applied_torque_Nm'])
            lifted_probes.append(probes.tolist())
    half_work = energy_work_check(coordinate / 2, half_midplane_force, half_energy,
                                  mu_Pa=MU_PA, radius_m=RADIUS_M, height_m=FULL_HEIGHT_M / 2)
    lifted_work = energy_work_check(coordinate, lifted_force, lifted_energy,
                                    mu_Pa=MU_PA, radius_m=RADIUS_M, height_m=FULL_HEIGHT_M)
    _maxima(half_ratios, {'work_energy': half_work['maximum_error_J'] / half_work['limit_J'],
                         'solver_residual': max(row['actual_N2'] / row['limit_N2'] for row in half_solver['states'])})
    _maxima(lifted_ratios, {'work_energy': lifted_work['maximum_error_J'] / lifted_work['limit_J']})
    motion_limit = 1e-6 * RADIUS_M + 1e-5 * maxima['full_motion_m']
    _maxima(equivalence_ratios, {
        'node_motion': maxima['node_motion_m'] / motion_limit,
        'probe_motion': maxima['probe_motion_m'] / motion_limit,
    })
    # The declaration retains this cross-run work difference as diagnostic;
    # the original work/energy tolerance gates each domain independently.
    maxima['twice_half_vs_full_work_max_difference_J'] = float(np.max(np.abs(
        2 * np.asarray(half_work['work_J']) - np.asarray(full_receipt['energy_work']['work_J']))))
    for binding in bindings:
        verify_binding(root, binding)
    half_passed = all(value <= 1 for value in half_ratios.values()) and half_min_J > 0
    lifted_passed = all(value <= 1 for value in lifted_ratios.values()) and lifted_min_J > 0
    equivalent = all(value <= 1 for value in equivalence_ratios.values())
    return {
        'schema': 'hbe-halfheight-equivalence-readout-v1',
        'branch': expected_branch, 'mesh_N': expected_mesh_N, 'steps': 60, 'mu_Pa': MU_PA, 'frame_count': len(lifted_force),
        'protocol_sha256': ORIGINAL_PROTOCOL_SHA256,
        'declaration': declaration_binding, 'reconstruction': reconstruction_binding,
        'half_primitive_bindings': half_bindings, 'full_primitive_bindings': full_bindings,
        'full_native': full_receipt,
        'half_native': {
            'passed': half_passed, 'criteria': _criteria(half_ratios, half_min_J), 'solver': half_solver,
            'energy_work': half_work, 'energy_J': half_energy,
            'force_from_bottom_N': half_bottom_force, 'force_from_midplane_N': half_midplane_force,
            'load_coordinate_m': (coordinate / 2).tolist(),
        },
        'reconstructed_full': {
            'provenance': 'reflection_of_saved_half_native_fields_not_a_full_native_solve',
            'passed': lifted_passed, 'criteria': _criteria(lifted_ratios, lifted_min_J),
            'energy_work': lifted_work, 'energy_J': lifted_energy,
            'applied_force_N': lifted_force, 'applied_torque_Nm': lifted_torque,
            'probe_displacements_m': lifted_probes, 'load_coordinate_m': coordinate.tolist(),
            'native_full_solver_claim': False,
        },
        'equivalence': {
            'passed': equivalent,
            'criteria': {key: {'actual': value, 'limit': 1., 'units': 'ratio_to_declared_equivalence_tolerance'}
                         for key, value in equivalence_ratios.items()},
            'motion_limit_m': motion_limit, 'diagnostics': maxima,
            'logged_stress_J_sed_differences_are_diagnostic_only': True,
            'maximum_rest_coordinate_mismatch_m': model.maximum_coordinate_mismatch_m,
        },
        'passed': full_receipt['passed'] and half_passed and lifted_passed and equivalent,
        'measured_data_access': False, 'logged_sed_used_for_energy_gate': False,
        'sampled_J_positivity_is_not_everywhere_proof': True,
    }
