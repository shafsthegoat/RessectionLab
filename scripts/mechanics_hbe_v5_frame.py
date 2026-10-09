"""Pure, generated-fixture HBE v5 axial frame evaluator; never runs FEBio.

This deliberately does not produce an executable or physically validated run
receipt. A future reader must independently bind native files, topology and
every solver state before it can use these same frame calculations.
"""
from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass

import numpy as np

from scripts import mechanics_hbe_branch_calibration_v5 as v5
from scripts.mechanics_hbe_halfheight_readout import HalfHeightReconstruction
from scripts.mechanics_hbe_physics import HexMesh, finite_array, fixed_probes

NODE_FIELDS = 'x;y;z;ux;uy;uz;Rx;Ry;Rz'
ELEMENT_FIELDS = 'sx;sy;sz;sxy;syz;sxz;J;sed'


def _bytes(value):
    if isinstance(value, str):
        value = value.encode('utf-8')
    if not isinstance(value, bytes):
        raise ValueError('Exact source bytes required')
    return value


def _sha(value):
    return hashlib.sha256(_bytes(value)).hexdigest()


def verified_schedule(study_bytes, prior_bytes, run_id, source_deck_bytes,
                      adapted_deck_bytes, preparation_receipt):
    """Authenticate an actual adapted deck against the pinned v5 declaration.

    The source bytes are caller-supplied. Their identity must also be checked
    against the separate source-binding manifest before any future native use.
    """
    study_raw, prior_raw = _bytes(study_bytes), _bytes(prior_bytes)
    if _sha(study_raw) != v5.DECLARATION_SHA256:
        raise ValueError('Pinned v5 declaration SHA256 differs')
    study, prior = json.loads(study_raw), json.loads(prior_raw)
    if (_sha(prior_raw) != study['previous_v4']['sha256']
            or study['phase_gates'] != dict.fromkeys(study['phase_gates'], False)):
        raise ValueError('v4 ancestry or closed phase gates differ')
    row = v5.run_spec(study, prior, run_id)
    source, adapted = _bytes(source_deck_bytes), _bytes(adapted_deck_bytes)
    _, expected_adapted, receipt = v5.adapt_deck(study, prior, run_id, source)
    if adapted != expected_adapted.encode('utf-8') or preparation_receipt != receipt:
        raise ValueError('Actual adapted deck or its schedule receipt differs')
    if receipt['source_deck_sha256'] != _sha(source) or receipt['adapted_deck_sha256'] != _sha(adapted):
        raise ValueError('Deck bytes differ from preparation hashes')
    times = tuple(receipt['times'])
    full = tuple(receipt['full_load_coordinates_m'])
    native = tuple(receipt['native_boundary_coordinates_m'])
    factor = .5 if row['native_domain'] == 'lower_half_reconstructed' else 1.
    if (len(times) != row['frame_count'] or len(full) != len(times)
            or len(native) != len(times) or receipt['native_boundary_factor'] != factor
            or native != tuple(factor * value for value in full)
            or full[-1].hex() != row['endpoint_float64_hex']):
        raise ValueError('Exact extended-endpoint input schedule required')
    return {'run_id': run_id, 'branch': row['branch'], 'steps': row['steps'],
            'native_domain': row['native_domain'], 'mu_Pa': row['reference_mu_Pa'],
            'times': times, 'full_coordinates_m': full, 'native_coordinates_m': native,
            'v5_declaration_sha256': _sha(study_raw), 'v4_declaration_sha256': _sha(prior_raw),
            'source_deck_sha256': _sha(source), 'adapted_deck_sha256': _sha(adapted),
            'source_binding_checked': False, 'native_execution_released': False}


def _record(record, *, step, time, name, count, fields):
    if (record.get('step') != step or record.get('name') != name
            or record.get('declared_time') != time
            or record.get('time') != float(format(time, '.9g'))):
        raise ValueError('Mismatched parsed primitive ID/time/step')
    return finite_array(record.get('values'), (count, fields))


def _ratios(mesh, current, raw, *, coordinate_m, half):
    X = mesh.rest_nodes_m
    top, bottom = np.asarray(mesh._top), np.asarray(mesh._bottom)
    prescribed = X.copy()
    prescribed[top, 2] += coordinate_m
    if not half:
        top_error = np.linalg.norm(current[top] - prescribed[top], axis=1).max()
    else:
        top_error = np.abs(current[top, 2] - prescribed[top, 2]).max()
    bottom_error = np.linalg.norm(current[bottom] - X[bottom], axis=1).max()
    free_raw = raw.copy()
    free_raw[bottom] = 0
    free_raw[top] = 0 if not half else free_raw[top] * [1, 1, 0]
    moments = np.cross(current, raw)
    F0 = 1000. * mesh.radius_m**2
    T0 = F0 * mesh.radius_m
    ratios = {
        'force_balance': float(np.linalg.norm(raw.sum(axis=0)) /
                               (1e-8 * F0 + 1e-5 * np.linalg.norm(raw, axis=1).sum())),
        'moment_balance': float(np.linalg.norm(moments.sum(axis=0)) /
                                (1e-8 * T0 + 1e-5 * np.linalg.norm(moments, axis=1).sum())),
        'prescribed_motion': float(max(bottom_error, top_error) / (1e-8 * mesh.radius_m)),
        'free_dof_reaction': float(np.linalg.norm(free_raw, axis=1).max() / (1e-8 * F0)),
    }
    return ratios


@dataclass(frozen=True)
class PreparedFrame:
    """One validated local geometry snapshot; no cross-run cache."""

    native_mesh: HexMesh
    full_mesh: HexMesh
    reconstruction_model: HalfHeightReconstruction | None
    probe_map: tuple
    mapping_sha256: str | None
    contract_key: tuple


def _contract_key(contract):
    """Bind a prepared snapshot to one exact source/deck/schedule contract."""
    return (contract['run_id'], contract['branch'], contract['steps'],
            contract['native_domain'], contract['mu_Pa'],
            contract['v5_declaration_sha256'], contract['v4_declaration_sha256'],
            contract['source_deck_sha256'], contract['adapted_deck_sha256'],
            tuple(contract['times']), tuple(contract['full_coordinates_m']),
            tuple(contract['native_coordinates_m']))


def _freeze_geometry_arrays(mesh):
    for value in vars(mesh).values():
        if isinstance(value, np.ndarray):
            value.setflags(write=False)


def prepare_generated_frame(contract, mesh_manifest, *, reconstruction=None):
    """Validate fixed mesh/mapping once; never cache across independent runs."""
    key = _contract_key(contract)
    native_mesh = HexMesh.from_manifest(mesh_manifest)
    full_mesh, model, mapping_sha = native_mesh, None, None
    if reconstruction is not None:
        full_manifest, mapping = reconstruction
        model = HalfHeightReconstruction(full_manifest, mesh_manifest, mapping)
        if model.half.fingerprint != native_mesh.fingerprint:
            raise ValueError('Half mesh reconstruction differs')
        full_mesh = model.full
        mapping_sha = hashlib.sha256(json.dumps(
            mapping, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()).hexdigest()
    probes = full_mesh.probe_map(fixed_probes(native_mesh.radius_m, full_mesh.height_m))
    _freeze_geometry_arrays(native_mesh)
    if model is not None:
        _freeze_geometry_arrays(model.half)
        _freeze_geometry_arrays(model.full)
        for value in vars(model).values():
            if isinstance(value, np.ndarray):
                value.setflags(write=False)
        for name in ('node_maps', 'cell_maps'):
            values = tuple(getattr(model, name))
            for value in values:
                value.setflags(write=False)
            setattr(model, name, values)
    return PreparedFrame(native_mesh, full_mesh, model, probes, mapping_sha, key)


def evaluate_generated_frame(contract, mesh_manifest, node_record, element_record,
                             *, reconstruction=None):
    """Evaluate one parsed generated frame, rebuilding its geometry each call."""
    prepared = prepare_generated_frame(contract, mesh_manifest, reconstruction=reconstruction)
    return evaluate_prepared_frame(contract, prepared, node_record, element_record)


def evaluate_prepared_frame(contract, prepared, node_record, element_record):
    """Evaluate one frame with validated stream-local geometry and unchanged gates.

    The record pair must come from `_iter_data_records`, whose one-based ID
    checks are not repeated here. The caller owns this per-stream context.
    """
    if not isinstance(prepared, PreparedFrame):
        raise ValueError('Validated stream-local geometry required')
    if _contract_key(contract) != prepared.contract_key:
        raise ValueError('Prepared geometry belongs to a different run contract')
    if contract.get('native_execution_released') is not False or contract.get('source_binding_checked') is not False:
        raise ValueError('Only closed, synthetic-fixture contracts are accepted')
    step = node_record.get('step')
    if type(step) is not int or not 0 <= step <= contract['steps']:
        raise ValueError('Frame outside declared v5 schedule')
    time = contract['times'][step]
    native_mesh = prepared.native_mesh
    X = native_mesh.rest_nodes_m
    values = _record(node_record, step=step, time=time,
                     name='mechanics_nodes_si', count=len(X), fields=9)
    logged = _record(element_record, step=step, time=time,
                     name='mechanics_elements_si', count=native_mesh.element_count, fields=8)
    if np.any(logged[:, 6] <= 0):
        raise ValueError('Nonpositive logged element J')
    current, raw = values[:, :3], values[:, 6:9]
    R, mu = native_mesh.radius_m, contract['mu_Pa']
    if mu != 1000. or R != .004:
        raise ValueError('Pinned HBE reference material or radius differs')
    motion_error = np.linalg.norm(values[:, 3:6] - (current - X), axis=1).max()
    primitive_ratio = float(motion_error / (1e-8 * R))
    if step == 0:
        primitive_ratio = max(primitive_ratio,
                              float(np.linalg.norm(current - X, axis=1).max() / (1e-8 * R)),
                              float(np.linalg.norm(raw, axis=1).max() / (1e-8 * mu * R**2)))
    full_coordinate = contract['full_coordinates_m'][step]
    native_coordinate = contract['native_coordinates_m'][step]
    half = contract['native_domain'] == 'lower_half_reconstructed'
    if half != (prepared.reconstruction_model is not None):
        raise ValueError('Representation-specific reconstruction required')
    native_ratios = _ratios(native_mesh, current, raw,
                            coordinate_m=native_coordinate, half=half)
    native_ratios['primitive_consistency'] = primitive_ratio
    native_state = native_mesh.deformation(current, mu)
    top, bottom = np.asarray(native_mesh._top), np.asarray(native_mesh._bottom)
    native_force = float(-raw[top, 2].sum())
    bottom_force = float(raw[bottom, 2].sum())
    full_mesh, full_current, full_raw = prepared.full_mesh, current, raw
    reflected_logs = None
    if half:
        lifted = prepared.reconstruction_model.lift(current, raw, logged, full_coordinate)
        full_current, full_raw = lifted['current_nodes_m'], lifted['raw_reactions_N']
        reflected_logs = lifted['logged_elements']
    if (full_mesh.height_m != .00489159
            or native_mesh.height_m != (.00489159 / 2 if half else .00489159)):
        raise ValueError('Pinned HBE full/half height differs')
    full_ratios = _ratios(full_mesh, full_current, full_raw,
                          coordinate_m=full_coordinate, half=False)
    full_state = full_mesh.deformation(full_current, mu)
    full_force = float(-full_raw[np.asarray(full_mesh._top), 2].sum())
    energy_limit = 1e-7 * mu * R**2 * full_mesh.height_m + 1e-4 * abs(full_state['energy_J'])
    force_limit = 1e-7 * mu * R**2 + 1e-4 * abs(full_force)
    scale_ratios = ({'force_bottom_scale_one': abs(bottom_force - full_force) / force_limit,
                     'force_midplane_scale_one': abs(native_force - full_force) / force_limit,
                     'energy_scale_two': abs(2 * native_state['energy_J'] - full_state['energy_J']) / energy_limit}
                    if half else {})
    # This is an analytical fixture sign control, not a universal constitutive theorem.
    sign_ok = step == 0 or full_force * full_coordinate > 0
    full_ratios['force_sign_fixture'] = 0. if sign_ok else 2.
    criteria = {**{f'native_{key}': value for key, value in native_ratios.items()},
                **{f'full_{key}': value for key, value in full_ratios.items()},
                **scale_ratios}
    probes = full_mesh.interpolate_displacement(full_current, prepared.probe_map)
    if probes.shape != (75, 3):
        raise ValueError('Original 75 physical probes required')
    minimum_J = min(native_state['minimum_sampled_J'], full_state['minimum_sampled_J'])
    return {
        'schema': 'hbe-v5-generated-single-frame-v1', 'run_id': contract['run_id'],
        'frame': step, 'pseudo_time': time, 'input_coordinate_full_m': full_coordinate,
        'input_coordinate_native_m': native_coordinate,
        'representation': 'reconstructed_full' if half else 'full_native_fixture',
        'applied_force_N': full_force, 'raw_bottom_support_force_N': bottom_force,
        'raw_top_or_midplane_force_N': native_force,
        'raw_bottom_support_vector_N': raw[bottom].sum(axis=0).tolist(),
        'raw_top_or_midplane_vector_N': raw[top].sum(axis=0).tolist(),
        'probe_displacements_m': probes.tolist(), 'energy_J': full_state['energy_J'],
        'native_energy_J': native_state['energy_J'], 'minimum_sampled_J': minimum_J,
        'minimum_logged_J': float(logged[:, 6].min()),
        'logged_fields': ELEMENT_FIELDS, 'logged_stress_units': 'Pa',
        'logged_stress_component_min_Pa': logged[:, :6].min(axis=0).tolist(),
        'logged_stress_component_max_Pa': logged[:, :6].max(axis=0).tolist(),
        'logged_sed_range_Pa': [float(logged[:, 7].min()), float(logged[:, 7].max())],
        'reflected_stress_component_min_Pa': (reflected_logs[:, :6].min(axis=0).tolist()
                                              if reflected_logs is not None else None),
        'criteria_ratios': criteria, 'fixture_passed': all(value <= 1 for value in criteria.values()),
        'provenance': {'v5_declaration_sha256': contract['v5_declaration_sha256'],
                       'v4_declaration_sha256': contract['v4_declaration_sha256'],
                       'source_deck_sha256': contract['source_deck_sha256'],
                       'adapted_deck_sha256': contract['adapted_deck_sha256'],
                       'mesh_fingerprint': native_mesh.fingerprint,
                       'full_mesh_fingerprint': full_mesh.fingerprint,
                       'reconstruction_mapping_sha256': prepared.mapping_sha256,
                       'native_output_observed': False, 'generated_fixture_only': True,
                       'measured_response_accessed': False, 'patient_data_accessed': False,
                       'physical_validation_pass': None, 'calibration_released': False},
        'logged_sed_used_for_energy_gate': False,
        'sampled_J_positivity_is_not_everywhere_proof': True,
    }
