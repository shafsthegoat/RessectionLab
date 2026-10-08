"""Small shared temporal primitives extracted from the accepted N36 path.

No authorization, geometry selection, mesher or native solver is provided here.
Exact-case adapters authenticate inputs before and after these calculations.
"""
from contextlib import ExitStack
import itertools
import math
from pathlib import Path
import subprocess
import time
import numpy as np
from scripts import mechanics_hbe_halfheight_readout as inherited
from scripts import mechanics_hbe_outputs as outputs
from scripts import mechanics_hbe_access as access
from scripts import mechanics_hbe_halfheight_global_n36_experiment as receipts
from scripts.mechanics_hbe_halfheight_global_n36_temporal import verify_schedule_only_change

R, H, MU = inherited.RADIUS_M, inherited.FULL_HEIGHT_M, inherited.MU_PA
TIMES = np.linspace(0., 1., 121)


def remaining_seconds(deadline, cap, *, now=None):
    now = time.monotonic() if now is None else now
    if not all(math.isfinite(x) for x in (deadline, cap, now)) or cap <= 0:
        raise ValueError('Finite positive nested time allowance required')
    value = min(cap, deadline-now)
    if value <= 0:
        raise TimeoutError('Inclusive aggregate allowance exhausted')
    return value


def pure_child(command, directory, deadline, cap):
    """Child remains in outer supervised group; bounded wait, kill and reap."""
    directory = Path(directory)
    started, process = time.monotonic(), None
    record = {'schema': 'hbe-pure-child-execution-v1', 'command': command,
              'cap_seconds': cap, 'included_in_aggregate': True,
              'status': 'starting', 'solver_calls': 0, 'gmsh_generation_calls': 0}
    try:
        with (directory/'console.txt').open('x') as stream:
            process = subprocess.Popen(command, cwd=directory, stdout=stream, stderr=subprocess.STDOUT)
            record['pid'] = process.pid
            record['exit_code'] = process.wait(timeout=remaining_seconds(deadline, cap))
        if record['exit_code'] != 0:
            raise RuntimeError('Pure preparation child failed')
        remaining_seconds(deadline, cap)
        record['status'] = 'completed'
    except BaseException as error:
        record.update(status='failed_or_incomplete', error={'type':type(error).__name__, 'message':str(error)})
        raise
    finally:
        if process is not None:
            if process.poll() is None:
                process.kill()
            process.wait()
        record['elapsed_seconds'] = time.monotonic()-started
        receipts.durable_json(directory, directory/'child-execution.json', record)
    remaining_seconds(deadline, cap)
    return record


def records(stack, root, binding, count, fields, name):
    stream = stack.enter_context(access.local_path(root, binding['path']).open(encoding='utf-8',errors='strict'))
    return outputs.iter_data_records(stream, expected_times=TIMES, item_count=count,
        field_count=fields, record_name=name, maximum_bytes=256*1024**2)


def read_frames(root, half_bindings, model, *, branch):
    """Accepted 121-state half/full equations, one native frame at a time."""
    if branch not in ('compression','tension'):
        raise ValueError('Axial half-height branch required')
    coordinate = TIMES * (-1 if branch == 'compression' else 1) * .15 * H
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
            records(stack, root, half_bindings['nodes'], len(model.Xh), 9, 'mechanics_nodes_si'),
            records(stack, root, half_bindings['elements'], model.half.element_count, 8, 'mechanics_elements_si'))
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
                                         mu_Pa=MU, branch=branch, fraction=TIMES[index])
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
    return {
        'branch': branch, 'steps': 120, 'mu_Pa': MU, 'frame_count': 121,
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

