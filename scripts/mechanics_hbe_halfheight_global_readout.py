"""Exact N32 native readout and prospective unequal-spacing diagnostics; no solving.

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
from scripts import mechanics_hbe_halfheight_global as core
from scripts import mechanics_hbe_readout as original_readout
from scripts import mechanics_hbe_outputs as outputs

access = core.access
R, H, MU = inherited.RADIUS_M, inherited.FULL_HEIGHT_M, inherited.MU_PA
TIMES = inherited.TIMES


def primitive_limit(kind):
    return {'nodes': 384*1024**2, 'elements': 256*1024**2,
            'solver': 16*1024**2, 'mesh': 32*1024**2,
            'deck': 32*1024**2, 'loading': 1024**2}[kind]


def verify_native_bindings(root, bindings):
    if set(bindings) != inherited.PRIMITIVE_KEYS:
        raise ValueError('Exactly six native N32 primitive bindings required')
    # In particular, the node log is authenticated with 384 MiB both before
    # and after parse. The legacy access/parser defaults remain unchanged.
    for name, binding in bindings.items():
        access.verify_binding(root, binding, maximum_bytes=primitive_limit(name))


def iter_global_records(lines, *, study, kind, item_count, field_count, record_name,
                        expected_times, expected_run_id):
    core.require_study(study)
    times = tuple(float(t) for t in expected_times)
    if (expected_run_id != core.RUN_ID or kind not in ('nodes', 'elements')
            or times != tuple(float(t) for t in TIMES)):
        raise ValueError('Exact N32 compression S60 parser contract required')
    spec = study['parser'][kind]
    if ((item_count, field_count, record_name) != (spec['items'], spec['fields'], spec['name'])
            or type(item_count) is not int or type(field_count) is not int):
        raise ValueError('Exact N32 native item/field counts required')
    return outputs._iter_data_records(lines, expected_times=times,
        item_count=item_count, field_count=field_count, record_name=record_name,
        maximum_bytes=spec['maximum_bytes'], maximum_items=spec['items'])


def _records(stack, root, binding, study, kind):
    stream = stack.enter_context(access.local_path(root, binding['path']).open(encoding='utf-8', errors='strict'))
    spec = study['parser'][kind]
    return iter_global_records(stream, study=study, kind=kind, item_count=spec['items'],
        field_count=spec['fields'], record_name=spec['name'], expected_times=TIMES, expected_run_id=core.RUN_ID)


def read_global_run(root, *, half_bindings, reconstruction_binding, declaration_binding):
    """All 61 native states; unchanged independent half/full physics checks."""
    half_bindings, reconstruction_binding, declaration_binding = deepcopy(
        (half_bindings, reconstruction_binding, declaration_binding))
    verify_native_bindings(root, half_bindings)
    study, wrapper, half_manifest, full_manifest = core.verify_prepared_case(
        root, declaration_binding, half_bindings, reconstruction_binding)
    full_binding = wrapper['full_mesh']
    expected_branch, expected_mesh_N = 'compression', 32
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
    verify_native_bindings(root, half_bindings)
    for binding in [declaration_binding, reconstruction_binding, full_binding,
                    study['original_protocol'], wrapper['mesh_source'], wrapper['extraction_source']]:
        access.verify_binding(root, binding)
    return {
        'schema': 'hbe-halfheight-global-run-readout-v1', 'branch': expected_branch,
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


TRIPLETS = ((8, 12, 16), (12, 16, 24), (16, 24, 32))
FORCE_FLOOR_N = 1.6e-7


def order_ratio(p, a, b):
    """Stable positive-order quotient; its continuous value at zero is a/b."""
    return a/b if p == 0 else math.expm1(p*a)/(-math.expm1(-p*b))


def unequal_order(levels, forces):
    """Signed scalar discretization diagnostic; never an error-bound claim.

    F(N)=F_limit+C/N**p is a conditional model. Null estimates are retained,
    including differences below the original absolute floor.
    """
    if tuple(levels) not in TRIPLETS or len(forces) != 3:
        raise ValueError('One of the three prospectively declared triplets required')
    row = {'levels': list(levels), 'status': 'invalid', 'order': None,
        'signed_remaining_indicator_N': None, 'absolute_remaining_indicator_N': None,
        'force_limit_N': None, 'continuum_error_bound': False}
    if not all(math.isfinite(v) for v in forces):
        return row
    first, second, third = (float(v) for v in forces)
    d12, d23 = second-first, third-second
    if not all(math.isfinite(v) for v in (d12, d23)):
        return row
    row.update(signed_first_change_N=d12, signed_second_change_N=d23)
    low12, low23 = abs(d12) <= FORCE_FLOOR_N, abs(d23) <= FORCE_FLOOR_N
    if low12 or low23:
        row['status'] = ('below_original_difference_floor' if low12 and low23
                         else 'insufficient_increment_resolution')
        return row
    if math.copysign(1., d12) != math.copysign(1., d23):
        row['status'] = 'opposite_signs'
        return row
    a, b = math.log(levels[1]/levels[0]), math.log(levels[2]/levels[1])
    ratio = d12/d23
    if not math.isfinite(ratio):
        return row
    row.update(increment_ratio=ratio, positive_order_ratio_lower_limit=a/b)
    if ratio <= a/b or ratio > order_ratio(16., a, b):
        row['status'] = 'no_admissible_positive_root'
        return row
    lower, upper = 0., 16.
    for _ in range(80):
        midpoint = (lower+upper)/2
        value = order_ratio(midpoint, a, b)
        if not math.isfinite(value):
            return row
        if value < ratio:
            lower = midpoint
        else:
            upper = midpoint
    p = (lower+upper)/2
    if not math.isfinite(p) or p < 1e-6:
        row['status'] = 'no_admissible_positive_root'
        return row
    correction = d23/math.expm1(p*b)
    limit = third+correction
    if not all(math.isfinite(v) for v in (correction, limit)):
        return row
    row.update(status='eligible', order=p, signed_remaining_indicator_N=correction,
               absolute_remaining_indicator_N=abs(correction), force_limit_N=limit)
    return row


def state_diagnostic(forces, *, index, force_limit_N):
    """Compare three signed triplets without concealing unresolved states."""
    if set(forces) != {8, 12, 16, 24, 32} or not 0 <= index <= 60:
        raise ValueError('Exact five-level common-state scalar inputs required')
    if not math.isfinite(force_limit_N) or force_limit_N < FORCE_FLOOR_N:
        raise ValueError('Original finite series-wide force allowance required')
    row = {'state': index, 'fraction': index/60, 'status': 'rest/not_estimated' if index == 0 else 'unresolved',
        'orders': {}, 'order_changes': None, 'increasing_order_drift': None,
        'force_limit_instability': None, 'latest_limit_disagreement_N': None,
        'two_latest_limit_envelope_N': None, 'remaining_within_allowance': None,
        'continuum_error_bound': False}
    if index == 0:
        return row
    orders = [unequal_order(ns, [forces[n] for n in ns]) for ns in TRIPLETS]
    row['orders'] = {'-'.join(f'N{n}' for n in ns): v for ns, v in zip(TRIPLETS, orders)}
    old, recent, current = orders
    if all(v['status'] == 'eligible' for v in orders):
        changes = [abs(recent['order']-old['order']), abs(current['order']-recent['order'])]
        row.update(order_changes=changes, increasing_order_drift=changes[1] > changes[0]+1e-6)
    if recent['status'] == current['status'] == 'eligible':
        disagreement = abs(current['force_limit_N']-recent['force_limit_N'])
        envelope = max(abs(current['force_limit_N']-forces[32]), abs(recent['force_limit_N']-forces[32]))
        if math.isfinite(disagreement) and math.isfinite(envelope):
            row.update(status='resolved_conditional_model', latest_limit_disagreement_N=disagreement,
                force_limit_instability=disagreement > force_limit_N,
                two_latest_limit_envelope_N=envelope, remaining_within_allowance=envelope <= force_limit_N)
    return row


def replay_prior_runs(root, study):
    """Future supervised solve phase only: exactly three old full-domain runs."""
    core.require_study(study)
    result = {}
    for key, source in study['prior_runs'].items():
        N = int(key.split(':')[1][1:])
        accepted = access.verify_binding(root, source['readout'], maximum_bytes=16*1024**2, read_json=True)
        if N == 16:
            actual, _ = original_readout.read_resolution_run(root, source['primitive_bindings'],
                declaration_binding=study['original_resolution_declaration'], expected_branch='compression', expected_mesh_N=N)
        else:
            actual, _ = original_readout.read_run(root, source['primitive_bindings'],
                protocol_sha256=study['original_protocol']['sha256'], expected_branch='compression',
                expected_mesh_N=N, expected_steps=60, expected_mu_Pa=MU)
        actual['execution_binding'] = source['execution']
        if actual['passed'] is not True or access.canonical_json(actual) != access.canonical_json(accepted):
            raise ValueError('Replayed prior receipt differs from accepted complete raw readout')
        result[key] = actual
    return result


def comparison_report(runs, study):
    """Exact four authenticated prior receipts and the one new native receipt.

    The runner authenticates source and native execution origin before calling
    this numerical comparison. A favorable screen is only a temporal-review
    candidate. No path here grants spatial acceptance or calibration access.
    """
    core.require_study(study)
    p1 = study['baseline_P1']
    wanted = set(study['prior_runs']) | {p1['run_id'], core.RUN_ID}
    if set(runs) != wanted:
        raise ValueError('Exactly four old and one new complete receipts required')
    forces, probes = {}, {}
    for key, row in runs.items():
        N = int(key.split(':')[1][1:])
        if (row.get('passed') is not True or row.get('protocol_sha256') != study['original_protocol']['sha256']
                or (row.get('branch'), row.get('mesh_N'), row.get('steps'), row.get('mu_Pa'), row.get('frame_count'))
                   != ('compression', N, 60, MU, 61)):
            raise ValueError('Complete individually passing numerical receipt required')
        if key in study['prior_runs'] or key == p1['run_id']:
            source = p1 if key == p1['run_id'] else study['prior_runs'][key]
            schema = ('hbe-halfheight-spatial-run-readout-v1' if N == 24 else
                      'hbe-resolution-run-readout-v1' if N == 16 else 'hbe-run-readout-v1')
            if (row.get('schema') != schema or row.get('primitive_bindings') != source['primitive_bindings']
                    or row.get('execution_binding') != source['execution']):
                raise ValueError('Prior receipt lost its authenticated native origin')
            if N == 24 and (row.get('declaration') != study['original_spatial_declaration']
                            or row.get('reconstruction') != p1['reconstruction']):
                raise ValueError('Uniform P1 N24 reconstruction origin differs')
        elif (row.get('schema') != 'hbe-halfheight-global-run-readout-v1'
                or row.get('declaration') != {'path': core.DECLARATION_PATH, 'sha256': core.DECLARATION_SHA256}
                or set(row.get('primitive_bindings', {})) != inherited.PRIMITIVE_KEYS
                or set(row.get('execution_binding', {})) != {'path', 'sha256'}):
            raise ValueError('New N32 receipt lost its native origin')
        if N in (24, 32) and (row.get('half_native', {}).get('passed') is not True
                or row.get('reconstructed_full', {}).get('passed') is not True
                or row.get('reconstruction_consistency', {}).get('passed') is not True
                or row.get('full_native_equivalence_evaluated') is not False
                or row.get('reconstructed_full', {}).get('native_full_solver_claim') is not False):
            raise ValueError('Native half and reconstructed full identities must pass separately')
        force, probe = np.asarray(row['applied_force_N'], float), np.asarray(row['probe_displacements_m'], float)
        coordinate = np.asarray(row['load_coordinate_m' if N >= 24 else 'load_coordinate'], float)
        expected_coordinate = -TIMES*.15*H if N >= 24 else TIMES*(-.15*H)
        if (force.shape != (61,) or probe.shape != (61, 75, 3) or coordinate.shape != (61,)
                or not np.isfinite(force).all() or not np.isfinite(probe).all()
                or not np.array_equal(coordinate, expected_coordinate)):
            raise ValueError('Complete finite common physical states and all 75 probes required')
        forces[N], probes[N] = force, probe
    allowance = FORCE_FLOOR_N+.02*float(np.max(np.abs(forces[32])))
    original_style = original_readout._refinement_group(
        *[runs[f'compression:N{N}:S60:reference'] for N in (16, 24, 32)], kind='mesh')
    change_pass = all(original_style[k]['actual'] <= original_style[k]['limit'] for k in ('reaction', 'motion'))
    states = [state_diagnostic({N: float(F[i]) for N, F in forces.items()}, index=i, force_limit_N=allowance)
              for i in range(61)]
    resolved = [v for v in states[1:] if v['status'] == 'resolved_conditional_model']
    unresolved = [v['state'] for v in states[1:] if v['status'] != 'resolved_conditional_model']
    exceed = [v['state'] for v in resolved if not v['remaining_within_allowance']]
    flags = [v['state'] for v in states[1:]
             if v['increasing_order_drift'] is True or v['force_limit_instability'] is True]
    endpoint_eligible = states[-1]['orders']['N16-N24-N32']['status'] == 'eligible'
    candidate = change_pass and endpoint_eligible and not unresolved and not exceed and not flags
    status = ('remaining_spatial_error_not_within_allowance' if exceed else
              'unresolved_nonrest_spatial_error' if unresolved else
              'conditional_order_or_limit_instability' if flags else
              'original_adjacent_change_checks_failed' if not change_pass else
              'candidate_for_finest_level_temporal_review')
    maximum = max(resolved, key=lambda v: v['two_latest_limit_envelope_N']) if resolved else None
    eligibility = {}
    for v in states[1:]:
        for triplet, estimate in v['orders'].items():
            counts = eligibility.setdefault(triplet, {})
            counts[estimate['status']] = counts.get(estimate['status'], 0)+1
    return {'schema': 'hbe-halfheight-global-comparison-v1', 'study_sha256': core.DECLARATION_SHA256,
        'status': status, 'candidate_for_temporal_review': candidate,
        'original_style_N16_N24_N32_metrics': original_style,
        'original_adjacent_change_checks_passed': change_pass,
        'original_raw_trend_retained_as_separate_metric': True,
        'force_allowance_N': allowance, 'states': states, 'endpoint': states[-1],
        'eligibility_counts': eligibility, 'unresolved_nonrest_states': unresolved,
        'remaining_exceedance_states': exceed, 'instability_states': flags,
        'maximum_eligible_envelope_N': None if maximum is None else maximum['two_latest_limit_envelope_N'],
        'maximum_eligible_envelope_state': None if maximum is None else maximum['state'],
        'interpretation': 'conditional two-limit model-sensitivity envelope; not a rigorous error bound or continuum certification',
        'forecast_not_observation': study['numerical_design']['forecast_not_observation'],
        'preserved_failures': study['preserved_failures'], 'spatial_convergence_accepted': False,
        'finest_level_step_convergence_evaluated': False, 'calibration_released': False,
        'measured_data_accessed': False, 'physical_validation_pass': None}
