"""Exact finer-half numerical readout and fixed spatial comparison; no solving.

Only genuine native half logs are read. Full fields are explicitly reconstructed;
there is no claim of a nonexistent full-native fine-case equivalence comparison.
"""
from __future__ import annotations

from contextlib import ExitStack
from copy import deepcopy
import itertools
import math

import numpy as np

from scripts import mechanics_hbe_halfheight_readout as inherited
from scripts import mechanics_hbe_halfheight_spatial as core
from scripts import mechanics_hbe_readout as original_readout

access = core.access
R, H, MU = inherited.RADIUS_M, inherited.FULL_HEIGHT_M, inherited.MU_PA
TIMES = inherited.TIMES


def read_spatial_run(root, *, half_bindings, reconstruction_binding,
                     declaration_binding, expected_branch, expected_mesh_N):
    """Exhaust all 61 native states, independently evaluate both domains."""
    half_bindings, reconstruction_binding, declaration_binding = deepcopy(
        (half_bindings, reconstruction_binding, declaration_binding))
    if set(half_bindings) != inherited.PRIMITIVE_KEYS:
        raise ValueError('Exactly six native half primitive bindings required')
    for binding in half_bindings.values():
        access.verify_binding(root, binding)
    study, wrapper, half_manifest = core.verify_prepared_case(
        root, declaration_binding, half_bindings, reconstruction_binding,
        expected_branch, expected_mesh_N)
    full_binding = study['levels'][str(expected_mesh_N)]['full_mesh']
    full_manifest = access.verify_binding(root, full_binding, maximum_bytes=16*1024**2, read_json=True)
    counts = study['levels'][str(expected_mesh_N)]['expected_counts']
    if (max(counts['full_nodes'], counts['full_elements']) > study['full_geometry_item_limit']
            or max(counts['half_nodes'], counts['half_elements']) > study['native_half_parser_item_limit']):
        raise ValueError('Declared native/full geometry item bounds exceeded')
    model = inherited.HalfHeightReconstruction(full_manifest, half_manifest, wrapper['mapping'])
    coordinate = TIMES * (-1 if expected_branch == 'compression' else 1) * .15 * H
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
                                         mu_Pa=MU, branch=expected_branch, fraction=TIMES[index])
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
        'schema': 'hbe-halfheight-spatial-run-readout-v1', 'branch': expected_branch,
        'mesh_N': expected_mesh_N, 'steps': 60, 'mu_Pa': MU, 'frame_count': 61,
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


def replay_prior_runs(root, study):
    """Future supervised readout only: exactly replay five completed old runs."""
    core.require_study(study)
    result = {}
    for key, source in study['prior_runs'].items():
        branch, mesh_id, _, _ = key.split(':')
        N = int(mesh_id[1:])
        accepted = access.verify_binding(root, source['readout'], maximum_bytes=16*1024**2, read_json=True)
        if N == 16:
            actual, _ = original_readout.read_resolution_run(root, source['primitive_bindings'],
                declaration_binding=study['original_resolution_declaration'], expected_branch=branch, expected_mesh_N=N)
        else:
            actual, _ = original_readout.read_run(root, source['primitive_bindings'],
                protocol_sha256=study['original_protocol']['sha256'], expected_branch=branch,
                expected_mesh_N=N, expected_steps=60, expected_mu_Pa=MU)
        actual['execution_binding'] = source['execution']
        if actual['passed'] is not True or access.canonical_json(actual) != access.canonical_json(accepted):
            raise ValueError('Replayed prior receipt differs from its accepted complete result')
        result[key] = actual
    return result


def comparison_report(runs, study):
    core.require_study(study)
    wanted = set(study['prior_runs']) | set(core.ORDERED_RUNS)
    if set(runs) != wanted:
        raise ValueError('Exactly five reused and three new spatial receipts required')
    for key, row in runs.items():
        branch, mesh_id, _, _ = key.split(':')
        if (row.get('passed') is not True or row.get('protocol_sha256') != study['original_protocol']['sha256']
                or (row.get('branch'), row.get('mesh_N'), row.get('steps'), row.get('mu_Pa'), row.get('frame_count'))
                   != (branch, int(mesh_id[1:]), 60, MU, 61)):
            raise ValueError('Complete individually passing spatial receipt required')
        if key in study['prior_runs']:
            source = study['prior_runs'][key]
            expected_schema = 'hbe-resolution-run-readout-v1' if mesh_id == 'N16' else 'hbe-run-readout-v1'
            if (row.get('schema') != expected_schema or row.get('primitive_bindings') != source['primitive_bindings']
                    or row.get('execution_binding') != source['execution']):
                raise ValueError('Prior comparison receipt lost its accepted native origin')
        elif (row.get('schema') != 'hbe-halfheight-spatial-run-readout-v1'
                or row.get('declaration') != {'path': core.DECLARATION_PATH, 'sha256': core.DECLARATION_SHA256}
                or row.get('half_native', {}).get('passed') is not True
                or row.get('reconstructed_full', {}).get('passed') is not True
                or row.get('reconstruction_consistency', {}).get('passed') is not True
                or row.get('full_native_equivalence_evaluated') is not False
                or row.get('reconstructed_full', {}).get('native_full_solver_claim') is not False
                or set(row.get('primitive_bindings', {})) != inherited.PRIMITIVE_KEYS
                or set(row.get('execution_binding', {})) != {'path', 'sha256'}):
            raise ValueError('New comparison receipt lost its native-half/reconstructed-full origin')
    groups = {}
    for triplet in study['co_primary_triplets']:
        for branch in study['branches']:
            rows = [runs[f'{branch}:N{N}:S60:reference'] for N in triplet]
            groups[f'{branch}:N{triplet[0]}-N{triplet[1]}-N{triplet[2]}'] = original_readout._refinement_group(*rows, kind='mesh')
    passed = all((v['actual'] < v['limit'] if v.get('comparison') == 'lt' else v['actual'] <= v['limit'])
                 for group in groups.values() for v in group.values())
    return {'schema': 'hbe-halfheight-spatial-comparison-v1', 'study_sha256': core.DECLARATION_SHA256,
            'co_primary_triplets': study['co_primary_triplets'], 'groups': groups, 'passed': passed,
            'preserved_failures': study['preserved_failures'], 'measured_data_accessed': False,
            'physical_validation_pass': None, 'calibration_released': False,
            'finest_level_step_convergence_evaluated': False}
