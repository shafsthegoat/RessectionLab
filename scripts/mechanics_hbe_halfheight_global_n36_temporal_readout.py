"""Exact N36 S120 readout and common-state temporal diagnostic; no solving.

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
from scripts import mechanics_hbe_halfheight_global_n36_temporal as core
from scripts import mechanics_hbe_readout as original_readout
from scripts import mechanics_hbe_outputs as outputs
from scripts import mechanics_hbe_halfheight_global_n36_readout as spatial_reader

access = core.access
R, H, MU = inherited.RADIUS_M, inherited.FULL_HEIGHT_M, inherited.MU_PA
TIMES = core.TIMES


def primitive_limit(kind):
    return {'nodes': 768*1024**2, 'elements': 512*1024**2,
            'solver': 16*1024**2, 'mesh': 32*1024**2,
            'deck': 32*1024**2, 'loading': 1024**2}[kind]


def verify_native_bindings(root, bindings):
    if set(bindings) != inherited.PRIMITIVE_KEYS:
        raise ValueError('Exactly six native N36 primitive bindings required')
    # In particular, the node log is authenticated with 768 MiB both before
    # and after parse. The legacy access/parser defaults remain unchanged.
    for name, binding in bindings.items():
        access.verify_binding(root, binding, maximum_bytes=primitive_limit(name))


def iter_global_records(lines, *, study, kind, item_count, field_count, record_name,
                        expected_times, expected_run_id):
    core.require_study(study)
    times = tuple(float(t) for t in expected_times)
    if (expected_run_id != core.RUN_ID or kind not in ('nodes', 'elements')
            or times != tuple(float(t) for t in TIMES)):
        raise ValueError('Exact N36 compression S120 parser contract required')
    spec = study['parser'][kind]
    if ((item_count, field_count, record_name) != (spec['items'], spec['fields'], spec['name'])
            or type(item_count) is not int or type(field_count) is not int):
        raise ValueError('Exact N36 native item/field counts required')
    return outputs._iter_data_records(lines, expected_times=times,
        item_count=item_count, field_count=field_count, record_name=record_name,
        maximum_bytes=spec['maximum_bytes'], maximum_items=spec['items'])


def _records(stack, root, binding, study, kind):
    stream = stack.enter_context(access.local_path(root, binding['path']).open(encoding='utf-8', errors='strict'))
    spec = study['parser'][kind]
    return iter_global_records(stream, study=study, kind=kind, item_count=spec['items'],
        field_count=spec['fields'], record_name=spec['name'], expected_times=TIMES, expected_run_id=core.RUN_ID)


def read_temporal_run(root, *, half_bindings, reconstruction_binding, declaration_binding):
    """All 121 native states; unchanged independent half/full physics checks."""
    half_bindings, reconstruction_binding, declaration_binding = deepcopy(
        (half_bindings, reconstruction_binding, declaration_binding))
    verify_native_bindings(root, half_bindings)
    study, wrapper, half_manifest, full_manifest = core.verify_prepared_case(
        root, declaration_binding, half_bindings, reconstruction_binding)
    full_binding = wrapper['full_mesh']
    expected_branch, expected_mesh_N = 'compression', 36
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
            _records(stack, root, half_bindings['nodes'], study, 'nodes'),
            _records(stack, root, half_bindings['elements'], study, 'elements'))
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
    if len(full_force) != 121:
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
    verify_native_bindings(root, half_bindings)
    for binding in [declaration_binding, reconstruction_binding, full_binding,
                    study['baseline']['original_protocol'], wrapper['mesh_source'], wrapper['extraction_source']]:
        access.verify_binding(root, binding)
    return {
        'schema': 'hbe-halfheight-global-n36-temporal-run-readout-v1', 'branch': expected_branch,
        'mesh_N': expected_mesh_N, 'steps': 120, 'mu_Pa': MU, 'frame_count': 121,
        'protocol_sha256': study['baseline']['original_protocol']['sha256'], 'declaration': declaration_binding,
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


def signed_margin_report(delta, margin):
    """Directional scalar sensitivity, not a force tolerance or error bound."""
    negative, positive = margin['negative_loss_delta_N'], margin['positive_loss_delta_N']
    if (margin.get('status') != 'bounded_scalar_decision_margin'
            or not all(math.isfinite(v) for v in (delta, negative, positive))
            or not negative < 0 < positive):
        raise ValueError('Finite directional baseline endpoint margins required')
    minimum = min(abs(negative), positive)
    return {'signed_S120_minus_S60_endpoint_N': delta,
            'negative_loss_delta_N': negative, 'positive_loss_delta_N': positive,
            'absolute_shift_over_minimum_margin': abs(delta)/minimum,
            'inside_fixed_other_state_scalar_interval': negative < delta < positive,
            'scope': 'endpoint only; other old states fixed in baseline margin calculation',
            'force_error_bound': False, 'used_as_acceptance_gate': False}


def state_classification(row):
    return {key: row.get(key) for key in ('status', 'increasing_order_drift',
        'force_limit_instability', 'remaining_within_allowance')} | {
        'triplet_eligibility': {key: value['status'] for key, value in row['orders'].items()}}


def metric_pass(row):
    return row['actual'] < row['limit'] if row.get('comparison') == 'lt' else row['actual'] <= row['limit']


def comparison_report(baseline_rows, fine, study, baseline_comparison):
    """Runner-authenticated saved S60 receipts and one actual S120 receipt.

Reusing compact independently verified readouts requires no old raw replay.
Substitution of S120 common values is explicitly a mixed-step sensitivity.
"""
    core.require_study(study)
    if set(baseline_rows) != {8, 12, 16, 24, 32, 36}:
        raise ValueError('Exactly six authenticated baseline mesh-level receipts required')
    base = baseline_rows[36]
    for N, row in baseline_rows.items():
        if ((row.get('branch'), row.get('mesh_N'), row.get('steps'), row.get('mu_Pa'), row.get('frame_count'))
                != ('compression', N, 60, MU, 61) or row.get('passed') is not True
                or row.get('protocol_sha256') != study['baseline']['original_protocol']['sha256']):
            raise ValueError('Complete passing S60 baseline receipts required')
        F = np.asarray(row['applied_force_N'], float)
        P = np.asarray(row['probe_displacements_m'], float)
        coordinate = np.asarray(row['load_coordinate_m' if N >= 24 else 'load_coordinate'], float)
        if (F.shape != (61,) or P.shape != (61, 75, 3) or coordinate.shape != (61,)
                or not np.isfinite(F).all() or not np.isfinite(P).all()
                or not np.array_equal(coordinate, -inherited.TIMES*.15*H if N >= 24 else inherited.TIMES*(-.15*H))):
            raise ValueError('Exact finite original S60 coordinates/probes required')
    if ((fine.get('branch'), fine.get('mesh_N'), fine.get('steps'), fine.get('mu_Pa'), fine.get('frame_count'))
            != ('compression', 36, 120, MU, 121) or fine.get('passed') is not True
            or fine.get('schema') != 'hbe-halfheight-global-n36-temporal-run-readout-v1'
            or fine.get('declaration') != {'path': core.DECLARATION_PATH, 'sha256': core.DECLARATION_SHA256}
            or fine.get('reconstruction') != study['baseline']['reconstruction']
            or fine.get('protocol_sha256') != study['baseline']['original_protocol']['sha256']
            or fine.get('full_native_equivalence_evaluated') is not False
            or any(fine.get(key, {}).get('passed') is not True
                   for key in ('half_native', 'reconstructed_full', 'reconstruction_consistency'))
            or fine['reconstructed_full'].get('native_full_solver_claim') is not False):
        raise ValueError('Complete source-bound individually passing native S120 readout required')
    force, probes = np.asarray(fine['applied_force_N'], float), np.asarray(fine['probe_displacements_m'], float)
    coordinate = np.asarray(fine['load_coordinate_m'], float)
    if (force.shape != (121,) or probes.shape != (121, 75, 3) or coordinate.shape != (121,)
            or not np.isfinite(force).all() or not np.isfinite(probes).all()
            or not np.array_equal(coordinate, -TIMES*.15*H)
            or not np.array_equal(coordinate[::2], np.asarray(base['load_coordinate_m']))):
        raise ValueError('121 finite native states and exact 61-state coordinate matching required')
    temporal = original_readout._refinement_group(base, base, fine, kind='step')
    temporal_pass = all(metric_pass(row) for row in temporal.values())
    delta = force[::2]-np.asarray(base['applied_force_N'])
    motion = np.linalg.norm(probes[::2]-np.asarray(base['probe_displacements_m']), axis=-1).max(axis=1)
    spatial_allowance = spatial_reader.FORCE_FLOOR_N+.02*float(np.max(np.abs(force[::2])))
    mixed = [spatial_reader.state_diagnostic({N: float(force[2*i] if N == 36 else
                row['applied_force_N'][i]) for N, row in baseline_rows.items()},
                index=i, force_limit_N=spatial_allowance) for i in range(61)]
    if (baseline_comparison.get('schema') != 'hbe-halfheight-global-n36-comparison-v1'
            or baseline_comparison.get('candidate_for_temporal_review') is not True
            or len(baseline_comparison.get('states', [])) != 61):
        raise ValueError('Exact accepted conditional S60 comparison required')
    changes = [{'state': i, 'before': state_classification(before), 'after': state_classification(after)}
               for i, (before, after) in enumerate(zip(baseline_comparison['states'], mixed))
               if state_classification(before) != state_classification(after)]
    adjacent = original_readout._refinement_group(baseline_rows[24], baseline_rows[32], fine, kind='mesh')
    old_adjacent = baseline_comparison['original_style_N24_N32_N36_metrics']
    adjacent_changes = [key for key, value in adjacent.items() if metric_pass(value) != metric_pass(old_adjacent[key])]
    unresolved = [r['state'] for r in mixed[1:] if r['status'] != 'resolved_conditional_model']
    instability = [r['state'] for r in mixed[1:]
                   if r['increasing_order_drift'] is True or r['force_limit_instability'] is True]
    exceed = [r['state'] for r in mixed[1:] if r['remaining_within_allowance'] is False]
    stable = not changes and not adjacent_changes
    candidate = temporal_pass and stable and not unresolved and not instability and not exceed
    maximum = int(np.argmax(np.abs(delta)))
    return {'schema': 'hbe-halfheight-global-n36-temporal-comparison-v1',
        'study_sha256': core.DECLARATION_SHA256,
        'baseline_comparison': study['baseline']['comparison'], 'fine_run_id': core.RUN_ID,
        'original_temporal_criteria': temporal, 'original_temporal_checks_passed': temporal_pass,
        'common_state_count': 61, 'signed_force_shift_N': delta.tolist(),
        'maximum_absolute_force_shift_N': float(abs(delta[maximum])), 'maximum_force_shift_state': maximum,
        'common_state_maximum_probe_vector_shifts_m': motion.tolist(),
        'endpoint_sensitivity': signed_margin_report(float(delta[-1]), baseline_comparison['endpoint_actual_decision_sensitivity']),
        'mixed_step_sensitivity': {'scope': 'only N36 replaced by S120 even indices; all older levels remain S60',
            'homogeneous_spatial_convergence_study': False, 'force_allowance_N': spatial_allowance,
            'states': mixed, 'endpoint': mixed[-1], 'classification_changes': changes,
            'adjacent_metrics': adjacent, 'adjacent_classification_changes': adjacent_changes,
            'classification_unchanged': stable, 'unresolved_nonrest_states': unresolved,
            'instability_states': instability, 'remaining_exceedance_states': exceed},
        'status': ('original_temporal_criteria_failed' if not temporal_pass else
                   'temporal_tolerance_pass_but_spatial_classification_fragile' if not candidate else
                   'finest_compression_temporal_candidate'),
        'compression_temporal_candidate': candidate, 'calibration_released': False,
        'spatial_convergence_accepted': False, 'physical_validation_pass': None,
        'measured_data_accessed': False, 'automatic_next_run_permitted': False,
        'preserved_failures': study['preserved_failures'],
        'interpretation': 'Static load-increment check only; mixed-step sensitivity is not continuum or physical validation.'}
