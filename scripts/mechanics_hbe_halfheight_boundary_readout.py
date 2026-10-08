"""Independent fixed-N24 boundary diagnostic; no solver or measured-curve access."""
from __future__ import annotations

from contextlib import ExitStack
from copy import deepcopy
import itertools
import math

import numpy as np

from scripts import mechanics_hbe_halfheight_readout as inherited
from scripts import mechanics_hbe_halfheight_boundary as core
from scripts import mechanics_hbe_physics as physics

access = core.access
R, H, MU = inherited.RADIUS_M, inherited.FULL_HEIGHT_M, inherited.MU_PA
TIMES = inherited.TIMES


def read_boundary_run(root, *, half_bindings, reconstruction_binding,
                     declaration_binding, expected_variant):
    """Exhaust all 61 native states, independently evaluate both domains."""
    half_bindings, reconstruction_binding, declaration_binding = deepcopy(
        (half_bindings, reconstruction_binding, declaration_binding))
    if set(half_bindings) != inherited.PRIMITIVE_KEYS:
        raise ValueError('Exactly six native half primitive bindings required')
    for binding in half_bindings.values():
        access.verify_binding(root, binding)
    study, wrapper, half_manifest, full_manifest = core.verify_prepared_case(
        root, declaration_binding, half_bindings, reconstruction_binding, expected_variant)
    full_binding = wrapper['full_mesh']
    counts = study['variants'][expected_variant]
    if ((len(full_manifest['rest_nodes_m']), len(full_manifest['elements_hex8'])) !=
            (counts['full_nodes'], counts['full_elements'])
            or (len(half_manifest['rest_nodes_m']), len(half_manifest['elements_hex8'])) !=
            (counts['half_nodes'], counts['half_elements'])
            or max(counts['half_nodes'], counts['half_elements']) > study['native_half_parser_item_limit']):
        raise ValueError('Declared native/full geometry item bounds exceeded')
    model = inherited.HalfHeightReconstruction(full_manifest, half_manifest, wrapper['mapping'])
    coordinate = -TIMES * .15 * H
    weights = regional_weights(half_manifest, study)
    endpoint = None
    with access.local_path(root, half_bindings['solver']['path']).open(encoding='utf-8', errors='strict') as stream:
        solver = inherited.check_solver_records(stream, expected_times=TIMES,
                                                residual_floor_N2=(1e-10*MU*R**2)**2)
    probe_map = model.full.probe_map(inherited.fixed_probes(R, H))
    half_ratios, full_ratios, reconstruction_ratios = {}, {}, {}
    half_energy, bottom_force, midplane_force = [], [], []
    full_energy, full_force, full_torque, full_probes = [], [], [], []
    minimum_half_J = minimum_full_J = math.inf
    shared_error = 0.
    F0, E0 = MU*R**2, MU*R**2*H
    with ExitStack() as stack:
        streams = (
            inherited._records(stack, root, half_bindings['nodes'], len(model.Xh), 9, 'mechanics_nodes_si'),
            inherited._records(stack, root, half_bindings['elements'], model.half.element_count, 8,
                               'mechanics_elements_si'))
        for hn, he in itertools.zip_longest(*streams):
            if hn is None or he is None or hn['step'] != he['step']:
                raise ValueError('Two complete synchronized native primitive streams required')
            index = hn['step']
            if index != len(full_force) or np.any(he['values'][:, 6] <= 0):
                raise ValueError('Out-of-order state or nonpositive logged element Jacobian')
            n = hn['values']
            current, raw = n[:, :3], n[:, 6:9]
            half = model.half_frame(current, raw, full_displacement_m=coordinate[index])
            lifted = model.lift(current, raw, he['values'], coordinate[index])
            full = model.full.read_frame(lifted['current_nodes_m'], lifted['raw_reactions_N'],
                                         mu_Pa=MU, branch='compression', fraction=TIMES[index])
            full['current_nodes_m'] = lifted['current_nodes_m']
            inherited._maxima(half_ratios, half['ratios'])
            inherited._maxima(full_ratios, inherited._frame_ratios(model.full, full, lifted['raw_reactions_N']))
            consistency = float(np.linalg.norm(n[:, 3:6]-(current-model.Xh), axis=1).max())/(1e-8*R)
            if index == 0:
                consistency = max(consistency,
                    float(np.linalg.norm(current-model.Xh, axis=1).max())/(1e-8*R),
                    float(np.linalg.norm(raw, axis=1).max())/(1e-8*F0))
            inherited._maxima(half_ratios, {'primitive_consistency': consistency})
            # These are internal half/full reconstruction identities, with the
            # accepted equivalence tolerances; no external native-full run exists.
            force_limit = 1e-7*F0 + 1e-4*abs(full['applied_force_N'])
            energy_limit = 1e-7*E0 + 1e-4*abs(full['energy_J'])
            inherited._maxima(reconstruction_ratios, {
                'bottom_force_scale_one': abs(half['force_from_bottom_N']-full['applied_force_N'])/force_limit,
                'midplane_force_scale_one': abs(half['force_from_midplane_N']-full['applied_force_N'])/force_limit,
                'half_energy_scale_two': abs(2*half['energy_J']-full['energy_J'])/energy_limit})
            shared_error = max(shared_error, lifted['maximum_shared_displacement_mismatch_m'])
            minimum_half_J = min(minimum_half_J, half['minimum_sampled_J'])
            minimum_full_J = min(minimum_full_J, full['minimum_sampled_J'])
            half_energy.append(half['energy_J'])
            bottom_force.append(half['force_from_bottom_N'])
            midplane_force.append(half['force_from_midplane_N'])
            full_energy.append(full['energy_J'])
            full_force.append(full['applied_force_N'])
            full_torque.append(full['applied_torque_Nm'])
            full_probes.append(model.full.interpolate_displacement(lifted['current_nodes_m'], probe_map).tolist())
            if index == 60:
                endpoint = endpoint_regions(weights, raw, he['values'], half['cell_energy_J'])
    if len(full_force) != 61:
        raise ValueError('All initial and converged native states are required')
    half_work = inherited.energy_work_check(coordinate/2, midplane_force, half_energy,
                                            mu_Pa=MU, radius_m=R, height_m=H/2)
    full_work = inherited.energy_work_check(coordinate, full_force, full_energy,
                                            mu_Pa=MU, radius_m=R, height_m=H)
    inherited._maxima(half_ratios, {
        'work_energy': half_work['maximum_error_J']/half_work['limit_J'],
        'solver_residual': max(row['actual_N2']/row['limit_N2'] for row in solver['states'])})
    inherited._maxima(full_ratios, {'work_energy': full_work['maximum_error_J']/full_work['limit_J']})
    half_pass = all(v <= 1 for v in half_ratios.values()) and minimum_half_J > 0
    full_pass = all(v <= 1 for v in full_ratios.values()) and minimum_full_J > 0
    reconstruction_pass = all(v <= 1 for v in reconstruction_ratios.values())
    for binding in (list(half_bindings.values()) + [declaration_binding, reconstruction_binding, full_binding,
                    study['original_protocol'], wrapper['mesh_source'], wrapper['extraction_source']]):
        access.verify_binding(root, binding)
    return {
        'schema': 'hbe-halfheight-boundary-run-readout-v1', 'branch': 'compression',
        'variant': expected_variant, 'endpoint_regions': endpoint, 'mesh_N': 24, 'steps': 60, 'mu_Pa': MU, 'frame_count': 61,
        'protocol_sha256': study['original_protocol']['sha256'], 'declaration': declaration_binding,
        'reconstruction': reconstruction_binding, 'primitive_bindings': half_bindings,
        'passed': half_pass and full_pass and reconstruction_pass,
        'half_native': {'passed': half_pass, 'criteria': inherited._criteria(half_ratios, minimum_half_J),
                        'solver': solver, 'energy_work': half_work, 'energy_J': half_energy,
                        'force_from_bottom_N': bottom_force, 'force_from_midplane_N': midplane_force,
                        'load_coordinate_m': (coordinate/2).tolist()},
        'reconstructed_full': {'passed': full_pass, 'criteria': inherited._criteria(full_ratios, minimum_full_J),
                               'provenance': 'reflection_of_native_half_fields_not_a_full_native_solve',
                               'native_full_solver_claim': False},
        'reconstruction_consistency': {'passed': reconstruction_pass,
            'criteria': {key: {'actual': value, 'limit': 1., 'units': 'ratio_to_accepted_reconstruction_tolerance'}
                         for key, value in reconstruction_ratios.items()},
            'maximum_shared_displacement_mismatch_m': shared_error},
        'full_native_equivalence_evaluated': False,
        # Flat full-domain quantities use the original refinement function;
        # their provenance remains explicitly reconstructed above.
        'applied_force_N': full_force, 'applied_torque_Nm': full_torque,
        'probe_displacements_m': full_probes, 'energy_J': full_energy,
        'energy_work': full_work, 'load_coordinate_m': coordinate.tolist(),
        'measured_data_accessed': False, 'physical_validation_pass': None,
        'logged_sed_used_for_energy_gate': False, 'sampled_J_positivity_is_not_everywhere_proof': True}


def regional_weights(mesh, study):
    """Fixed physical bins integrated with reference quadrature, not cell rings.

All variants retain the same cross-section. Radial classification therefore uses
identical quadrature locations; axial bin boundaries are retained source planes.
Bin volumes must agree across variants before any regional comparison is made.
"""
    core.require_study(study)
    X, C = np.asarray(mesh['rest_nodes_m']), np.asarray(mesh['elements_hex8'])-1
    rb = np.asarray(study['diagnostics']['volume_radial_bins_r_over_R'])
    zb = np.asarray(study['diagnostics']['volume_axial_bins_z_over_H'])
    shapes = np.array([physics.shape_functions(p) for p in physics.GAUSS])
    points = np.einsum('gn,mnc->mgc', shapes, X[C])
    det, _ = core.BASE.rest_jacobians(X, C+1)
    weights = det[:, :8]
    radial = np.searchsorted(rb, np.linalg.norm(points[:, :, :2], axis=2)/R, side='right')-1
    axial = np.searchsorted(zb, points[:, :, 2]/H, side='right')-1
    if np.any(radial < 0) or np.any(radial >= len(rb)-1) or np.any(axial < 0) or np.any(axial >= len(zb)-1):
        raise ValueError('Reference volume quadrature outside fixed physical bins')
    volume = np.stack([np.sum(weights*((radial == i) & (axial == j)), axis=1)
                      for j in range(len(zb)-1) for i in range(len(rb)-1)], axis=1)
    if not np.allclose(volume.sum(axis=1), weights.sum(axis=1), rtol=1e-13, atol=1e-24):
        raise ValueError('Fixed volume bins do not partition reference quadrature')
    faces = np.asarray(mesh['boundaries']['bottom']['faces_quad4'])-1
    Q = X[faces]
    signs = np.array([[-1,-1], [1,-1], [1,1], [-1,1]])
    area = np.zeros((len(X), len(rb)-1))
    for xi, eta in itertools.product((-1/math.sqrt(3), 1/math.sqrt(3)), repeat=2):
        shape = (1+signs[:, 0]*xi)*(1+signs[:, 1]*eta)/4
        dx = signs[:, 0]*(1+signs[:, 1]*eta)/4
        dy = signs[:, 1]*(1+signs[:, 0]*xi)/4
        jac = np.linalg.norm(np.cross(np.einsum('fni,n->fi', Q, dx), np.einsum('fni,n->fi', Q, dy)), axis=1)
        where = np.einsum('fni,n->fi', Q, shape)
        bins = np.searchsorted(rb, np.linalg.norm(where[:, :2], axis=1)/R, side='right')-1
        if np.any(jac <= 0) or np.any(bins < 0) or np.any(bins >= len(rb)-1):
            raise ValueError('Bottom-face reference quadrature invalid')
        for i in range(len(rb)-1):
            np.add.at(area[:, i], faces.ravel(), (jac[:, None]*shape[None, :]*(bins == i)[:, None]).ravel())
    if np.any(volume.sum(axis=0) <= 0) or np.any(area.sum(axis=0) <= 0):
        raise ValueError('Empty declared physical region')
    return {'volume': volume, 'area': area, 'cell_volume': weights.sum(axis=1),
            'volume_bins': [[float(rb[i]), float(rb[i+1]), float(zb[j]), float(zb[j+1])]
                            for j in range(len(zb)-1) for i in range(len(rb)-1)],
            'area_bins': [[float(rb[i]), float(rb[i+1])] for i in range(len(rb)-1)]}


def endpoint_regions(weights, raw, elements, cell_energy):
    volume, area = weights['volume'], weights['area']
    raw = physics.finite_array(raw, (len(area), 3))
    logged = physics.finite_array(elements, (len(volume), 8))
    energy = physics.finite_array(cell_energy, (len(volume),))
    s = logged[:, :6]
    vm = np.sqrt(((s[:, 0]-s[:, 1])**2+(s[:, 1]-s[:, 2])**2+(s[:, 2]-s[:, 0])**2)/2
                 +3*np.sum(s[:, 3:]**2, axis=1))
    tributary = area.sum(axis=1)
    traction = np.divide(raw[:, 2], tributary, out=np.zeros(len(raw)), where=tributary > 0)
    v, a = volume.sum(axis=0), area.sum(axis=0)
    return {'frame': 60, 'volume_bins_rR_zH': weights['volume_bins'], 'area_bins_rR': weights['area_bins'],
            'reference_volume_m3': v.tolist(), 'reference_area_m2': a.tolist(),
            'cell_average_stress_vm_volume_mean_Pa': ((vm@volume)/v).tolist(),
            'cell_average_stress_vm_volume_rms_Pa': np.sqrt((vm**2@volume)/v).tolist(),
            'logged_cell_J_volume_mean': ((logged[:, 6]@volume)/v).tolist(),
            'independent_energy_J': ((energy/weights['cell_volume'])@volume).tolist(),
            'signed_nodal_traction_proxy_area_mean_Pa': ((traction@area)/a).tolist(),
            'signed_nodal_traction_proxy_integral_N': (traction@area).tolist(),
            'regional_energy_approximation': 'element_total_energy_apportioned_by_bin_volume;piecewise_constant_element_energy_density;not_Gauss_energy_inside_radial_bins',
            'pointwise_stress_or_traction_claim': False, 'acceptance_gate': False}


def baseline_endpoint(root, study):
    """Additional regional diagnostic after authenticated complete P1 replay."""
    source = study['baseline_P1']['primitive_bindings']
    for record in source.values():
        access.verify_binding(root, record)
    mesh = access.verify_binding(root, source['mesh'], maximum_bytes=16*1024**2, read_json=True)
    if (len(mesh['rest_nodes_m']), len(mesh['elements_hex8'])) != (12439, 10368):
        raise ValueError('Exact native P1 counts required')
    weights, endpoint = regional_weights(mesh, study), None
    model = physics.HexMesh.from_manifest(mesh)
    with ExitStack() as stack:
        records = (inherited._records(stack, root, source['nodes'], 12439, 9, 'mechanics_nodes_si'),
                   inherited._records(stack, root, source['elements'], 10368, 8, 'mechanics_elements_si'))
        count = 0
        for nodes, elements in itertools.zip_longest(*records):
            if nodes is None or elements is None or nodes['step'] != count or elements['step'] != count:
                raise ValueError('Incomplete baseline regional source')
            if count == 60:
                current, raw = nodes['values'][:, :3], nodes['values'][:, 6:9]
                endpoint = endpoint_regions(weights, raw, elements['values'], model.deformation(current, MU)['cell_energy_J'])
            count += 1
    if count != 61 or endpoint is None:
        raise ValueError('Missing baseline endpoint')
    for record in source.values():
        access.verify_binding(root, record)
    return endpoint


def local_order(F1, F2, F4, *, floor_N=1.6e-7):
    """Scalar algebra only; not a continuum error estimator or acceptance gate."""
    values = tuple(float(v) for v in (F1, F2, F4, floor_N))
    if not all(math.isfinite(v) for v in values) or floor_N <= 0:
        raise ValueError('Finite endpoint forces and positive original floor required')
    d12, d24 = F2-F1, F4-F2
    result = {'delta12_N': d12, 'delta24_N': d24, 'order': None, 'local_limit_N': None,
              'local_remaining_indicator_N': None, 'continuum_error_bound': False}
    eligible = d12*d24 > 0 and abs(d12) > floor_N and abs(d24) > floor_N and abs(d24) < abs(d12)
    if eligible:
        p = math.log2(abs(d12))-math.log2(abs(d24))
        correction = d24/(abs(d12)/abs(d24)-1)
        if all(math.isfinite(v) for v in (p, correction, F4+correction)):
            result.update(order=p, local_limit_N=F4+correction, local_remaining_indicator_N=correction)
    result['eligible'] = result['order'] is not None
    return result


def signed_subtraction(minuend, subtrahend):
    """Explicit contrast direction, also independently testable scalar algebra."""
    a, b = physics.finite_array(minuend), physics.finite_array(subtrahend)
    if a.shape != b.shape:
        raise ValueError('Matched contrast shapes required')
    return a-b


def comparison_report(runs, study):
    core.require_study(study)
    if set(runs) != {'P1', *core.ORDERED_VARIANTS}:
        raise ValueError('Exactly one authenticated P1 and three new variants required')
    force, probe = {}, {}
    for variant, row in runs.items():
        if ((row.get('branch'), row.get('mesh_N'), row.get('steps'), row.get('mu_Pa'), row.get('frame_count')) !=
                ('compression', 24, 60, MU, 61) or row.get('passed') is not True
                or row.get('protocol_sha256') != study['original_protocol']['sha256']):
            raise ValueError('Individually passing complete fixed-protocol receipts required')
        if variant == 'P1':
            prior = study['baseline_P1']
            if (row.get('schema') != 'hbe-halfheight-spatial-run-readout-v1'
                    or row.get('primitive_bindings') != prior['primitive_bindings']
                    or row.get('execution_binding') != prior['execution']):
                raise ValueError('Baseline receipt lost its native origin')
        elif (row.get('schema') != 'hbe-halfheight-boundary-run-readout-v1' or row.get('variant') != variant
                or row.get('declaration') != {'path': core.DECLARATION_PATH, 'sha256': core.DECLARATION_SHA256}
                or set(row.get('execution_binding', {})) != {'path', 'sha256'}):
            raise ValueError('Variant receipt lost source/execution identity')
        if (row.get('half_native', {}).get('passed') is not True
                or row.get('reconstructed_full', {}).get('passed') is not True
                or row.get('reconstruction_consistency', {}).get('passed') is not True
                or row.get('full_native_equivalence_evaluated') is not False):
            raise ValueError('Complete native-half and reconstructed-full checks required')
        force[variant] = physics.finite_array(row['applied_force_N'], (61,))
        probe[variant] = physics.finite_array(row['probe_displacements_m'], (61, 75, 3))
        regions = row['endpoint_regions']
        reference = runs['P1']['endpoint_regions']
        for key in ('volume_bins_rR_zH', 'area_bins_rR'):
            if regions[key] != reference[key]:
                raise ValueError('Physical regional bins changed')
        for key in ('reference_volume_m3', 'reference_area_m2'):
            if not np.allclose(physics.finite_array(regions[key]), physics.finite_array(reference[key]),
                               rtol=study['diagnostics']['weight_conservation_relative_tolerance'], atol=1e-24):
                raise ValueError('Regional quadrature measures changed')
    comparisons = {}
    for first, second in (('P1', 'P2'), ('P1', 'I2'), ('I2', 'P2'), ('P2', 'P4')):
        delta = signed_subtraction(force[second], force[first])
        motion = np.linalg.norm(probe[second]-probe[first], axis=2).max(axis=1)
        limit = 1.6e-7+.02*float(np.abs(force[second]).max())
        comparisons[f'{first}:{second}'] = {'signed_force_difference_N': delta.tolist(),
            'difference_direction': f'{second}_minus_{first}',
            'endpoint_signed_force_difference_N': float(delta[-1]), 'maximum_absolute_force_difference_N': float(np.abs(delta).max()),
            'force_limit_N': limit, 'force_limit_reference': second, 'original_force_change_check': bool(np.abs(delta).max() <= limit),
            'maximum_probe_difference_m_by_state': motion.tolist(), 'maximum_probe_difference_m': float(motion.max()),
            'motion_limit_m': 8e-6, 'original_motion_change_check': bool(motion.max() <= 8e-6)}
    first, last = comparisons['P1:P2'], comparisons['P2:P4']
    trend = {}
    for key, floor in (('maximum_absolute_force_difference_N', 1.6e-7), ('maximum_probe_difference_m', 8e-6)):
        a, b = first[key], last[key]
        trend[key] = {'first': a, 'last': b, 'floor': floor,
                      'original_decrease_check': bool(b < a or (a <= floor and b <= floor))}
    return {'schema': 'hbe-halfheight-boundary-comparison-v1', 'study_sha256': core.DECLARATION_SHA256,
            'passed_individual_numerical_checks': True, 'primary': comparisons['I2:P2'], 'comparisons': comparisons,
            'secondary_endpoint': local_order(*(float(force[v][-1]) for v in ('P1', 'P2', 'P4'))),
            'secondary_all_state_original_trend': trend,
            'diagnostic_complete': True, 'spatial_convergence_accepted': False, 'calibration_released': False,
            'measured_data_accessed': False, 'physical_validation_pass': None,
            'preserved_failures': study['preserved_failures'], 'finest_level_step_convergence_evaluated': False}
