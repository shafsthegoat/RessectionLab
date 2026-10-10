"""Source-only HBE v5 twelve-row *generated-fixture* diagnostic.

This module compares compact arrays without admitting native execution, opening
measured responses, fitting tissue, or qualifying specimen/patient mechanics.
Even a passing diagnostic always has numerical_qualification_passed=False.
"""
from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np

from scripts import mechanics_hbe_branch_calibration_v4 as v4
from scripts import mechanics_hbe_branch_calibration_v5 as v5
from scripts import mechanics_hbe_halfheight_global_n36_readout as compression_history
from scripts import mechanics_hbe_halfheight_global_n36_temporal_readout as compression_step
from scripts import mechanics_hbe_readout as original_readout
from scripts import mechanics_hbe_v5_source_bindings as sources
from scripts.mechanics_hbe_access import verify_binding


COMPRESSION_LEVELS = (8, 12, 16, 24, 32, 36)
COMPRESSION_TRIPLETS = ((8, 12, 16), (12, 16, 24), (16, 24, 32), (24, 32, 36))
TENSION_LEVELS = (8, 12, 16, 24)
TENSION_TRIPLETS = ((12, 16, 24), (8, 16, 24))
ADJACENT = {
    'compression': ((8, 12), (12, 16), (16, 24), (24, 32), (32, 36)),
    'tension': ((8, 12), (12, 16), (16, 24)),
}


def _run(branch: str, n: int, steps: int = 60) -> str:
    return f'{branch}:N{n}:S{steps}:reference'


def _metric_pass(metric: dict) -> bool:
    actual, limit = metric['actual'], metric['limit']
    return actual < limit if metric.get('comparison') == 'lt' else actual <= limit


def _metrics_pass(metrics: dict) -> bool:
    return all(_metric_pass(item) for item in metrics.values())


def _view(study: dict, prior: dict, run_id: str, readout: dict,
          expected_source_sha256: str) -> tuple[dict, bool]:
    """Check exact v5 fixture identity and return the old numerical helper view."""
    spec = v5.run_spec(study, prior, run_id)
    schedule = v5.schedule(study, prior, run_id)
    if not isinstance(readout, dict):
        raise ValueError('Complete generated stream readout required')
    expected_representation = ('reconstructed_full' if spec['native_domain'] == 'lower_half_reconstructed'
                               else 'full_native_fixture')
    if (readout.get('schema') != 'hbe-v5-complete-stream-v1'
            or readout.get('run_id') != run_id
            or readout.get('steps') != spec['steps']
            or readout.get('frame_count') != spec['frame_count']
            or readout.get('representation') != expected_representation):
        raise ValueError('Generated v5 row identity or representation differs')
    provenance = readout.get('provenance', {})
    if (provenance.get('v5_declaration_sha256') != v5.DECLARATION_SHA256
            or provenance.get('source_deck_sha256') != expected_source_sha256
            or provenance.get('generated_fixture_only') is not True
            or provenance.get('native_output_observed') is not False
            or provenance.get('measured_response_accessed') is not False
            or provenance.get('patient_data_accessed') is not False
            or provenance.get('physical_validation_pass') is not None):
        raise ValueError('Only source-bound, generated, unmeasured fixture rows are admitted')
    return _numerical_view(study, prior, run_id, readout)


def _numerical_view(study: dict, prior: dict, run_id: str, readout: dict) -> tuple[dict, bool]:
    """Shared exact numerical contract; provenance admission belongs to callers."""
    spec = v5.run_spec(study, prior, run_id)
    schedule = v5.schedule(study, prior, run_id)
    coordinates = readout.get('load_coordinate_full_m')
    if (not isinstance(coordinates, list)
            or coordinates != schedule['full_coordinates_m']):
        raise ValueError('Exact v5 full-coordinate vector required')
    force = np.asarray(readout.get('applied_force_N'), dtype=float)
    probes = np.asarray(readout.get('probe_displacements_m'), dtype=float)
    if (force.shape != (spec['frame_count'],)
            or probes.shape != (spec['frame_count'], 75, 3)
            or not np.isfinite(force).all() or not np.isfinite(probes).all()):
        raise ValueError('Complete finite signed force and 75 three-component probes required')
    ratios = readout.get('criteria_max_ratio')
    if (not isinstance(ratios, dict) or not ratios
            or any(type(value) not in (int, float) or not math.isfinite(value) or value < 0
                   for value in ratios.values())):
        raise ValueError('Finite nonnegative complete-stream criteria required')
    solver = readout.get('solver', {})
    full_work = readout.get('full_energy_work', {})
    native_work = readout.get('native_energy_work')
    if (type(readout.get('numerical_passed')) is not bool
            or type(solver.get('passed')) is not bool
            or type(full_work.get('passed')) is not bool
            or (spec['native_domain'] == 'lower_half_reconstructed'
                and (not isinstance(native_work, dict) or type(native_work.get('passed')) is not bool))
            or (spec['native_domain'] == 'full_native' and native_work is not None)):
        raise ValueError('Declared generated numerical checks are incomplete')
    minimums = (readout.get('minimum_sampled_J'), readout.get('minimum_logged_J'))
    if any(type(value) not in (int, float) or not math.isfinite(value) for value in minimums):
        raise ValueError('Finite sampled and logged J minima required')
    calculated_pass = (solver['passed'] and full_work['passed']
                       and (native_work is None or native_work['passed'])
                       and all(value <= 1 for value in ratios.values())
                       and min(minimums) > 0)
    if readout['numerical_passed'] != calculated_pass:
        raise ValueError('Generated readout numerical flag contradicts its criteria')
    view = {'run_id': run_id, 'branch': spec['branch'], 'mesh_N': spec['N'],
            'steps': spec['steps'], 'mu_Pa': spec['reference_mu_Pa'],
            'frame_count': spec['frame_count'], 'load_coordinate_m': coordinates,
            'applied_force_N': force.tolist(), 'probe_displacements_m': probes.tolist()}
    return view, calculated_pass


def _signed_pair(coarse: dict, fine: dict) -> dict:
    """Retain every common-state signed difference; this alone is diagnostic."""
    if coarse['branch'] != fine['branch'] or coarse['steps'] != 60:
        raise ValueError('Same-branch S60 coarse row required')
    kind = 'step' if coarse['mesh_N'] == fine['mesh_N'] else 'mesh'
    if kind == 'step' and fine['steps'] != 120:
        raise ValueError('Declared S120 fine row required')
    if kind == 'mesh' and (fine['steps'] != 60 or fine['mesh_N'] <= coarse['mesh_N']):
        raise ValueError('Increasing S60 mesh pair required')
    stride = fine['steps'] // 60
    x0, x1 = coarse['load_coordinate_m'], fine['load_coordinate_m'][::stride]
    if x0 != x1 or len(x1) != 61:
        raise ValueError('Exact 61 common-state v5 coordinates required')
    f0 = np.asarray(coarse['applied_force_N'])
    f1 = np.asarray(fine['applied_force_N'])[::stride]
    p0 = np.asarray(coarse['probe_displacements_m'])
    p1 = np.asarray(fine['probe_displacements_m'])[::stride]
    delta_f, delta_p = f1 - f0, p1 - p0
    floor, relative, probe_limit = v4.PAIR_LIMITS[kind]
    force_limit = floor + relative * float(np.max(np.abs(f1)))
    maximum_force = float(np.max(np.abs(delta_f)))
    maximum_probe = float(np.max(np.linalg.norm(delta_p, axis=-1)))
    return {'coarse_run': coarse['run_id'], 'fine_run': fine['run_id'],
            'kind': kind, 'common_state_count': 61, 'probe_count': 75,
            'signed_force_difference_N': delta_f.tolist(),
            'signed_probe_component_difference_m': delta_p.tolist(),
            'maximum_force_change_N': maximum_force, 'force_limit_N': force_limit,
            'maximum_probe_norm_change_m': maximum_probe, 'probe_norm_limit_m': probe_limit,
            'pairwise_limit_passed': maximum_force <= force_limit and maximum_probe <= probe_limit,
            'diagnostic_only': True}


def _groups(views: dict, branch: str, triplets: tuple[tuple[int, int, int], ...],
            replacement: dict | None = None) -> dict:
    result = {}
    for levels in triplets:
        members = [replacement if replacement is not None and n == replacement['mesh_N']
                   else views[_run(branch, n)] for n in levels]
        metrics = original_readout._refinement_group(*members, kind='mesh')
        result['-'.join(f'N{n}' for n in levels)] = {
            'metrics': metrics, 'passed': _metrics_pass(metrics)}
    return result


def _compression_states(views: dict, replacement: dict | None = None) -> dict:
    forces = {n: np.asarray((replacement if replacement is not None and n == 36
                             else views[_run('compression', n)])['applied_force_N'])
              for n in COMPRESSION_LEVELS}
    if replacement is not None:
        forces[36] = forces[36][::2]
    allowance = v4.PAIR_LIMITS['mesh'][0] + v4.PAIR_LIMITS['mesh'][1] * float(np.max(np.abs(forces[36])))
    states = [compression_history.state_diagnostic(
        {n: float(force[i]) for n, force in forces.items()}, index=i, force_limit_N=allowance)
        for i in range(61)]
    unresolved = [row['state'] for row in states[1:] if row['status'] != 'resolved_conditional_model']
    exceed = [row['state'] for row in states[1:] if row['remaining_within_allowance'] is False]
    unstable = [row['state'] for row in states[1:] if row['increasing_order_drift'] is True
                or row['force_limit_instability'] is True]
    endpoint_eligible = states[-1]['orders']['N24-N32-N36']['status'] == 'eligible'
    return {'states': states, 'force_allowance_N': allowance,
            'unresolved_nonrest_states': unresolved, 'remaining_exceedance_states': exceed,
            'instability_states': unstable, 'endpoint_latest_triplet_eligible': endpoint_eligible,
            'passed_conditional_screen': endpoint_eligible and not unresolved and not exceed and not unstable,
            'continuum_error_bound': False}


def _classification_changes(before: dict, after: dict) -> list[str]:
    return [f'{group}/{name}' for group in before
            for name in before[group]['metrics']
            if _metric_pass(before[group]['metrics'][name]) !=
            _metric_pass(after[group]['metrics'][name])]


def _historical_n32(root: Path, prior: dict) -> dict:
    n36 = verify_binding(root, prior['evidence']['n36_spatial_declaration'],
                         maximum_bytes=1024**2, read_json=True)
    binding = n36['preserved_failures']['N32_comparison']
    comparison = verify_binding(root, binding, maximum_bytes=8 * 1024**2, read_json=True)
    if (comparison.get('schema') != 'hbe-halfheight-global-comparison-v1'
            or comparison.get('status') != 'remaining_spatial_error_not_within_allowance'
            or comparison.get('candidate_for_temporal_review') is not False
            or comparison.get('spatial_convergence_accepted') is not False
            or comparison.get('remaining_exceedance_states') != list(range(49, 61))
            or not comparison.get('maximum_eligible_envelope_N', 0) > comparison.get('force_allowance_N', math.inf)):
        raise ValueError('Hash-bound prior N32 negative result differs')
    return {'binding': binding, 'status': comparison['status'],
            'force_allowance_N': comparison['force_allowance_N'],
            'maximum_eligible_envelope_N': comparison['maximum_eligible_envelope_N'],
            'remaining_exceedance_states': comparison['remaining_exceedance_states'],
            'historical_only_not_v5_veto': True}


def compare_generated_rows(root: Path, rows: dict) -> dict:
    """Compare all twelve generated v5 fixture streams without any release switch.

    The returned screen can help test a future comparator, but can never admit
    native evidence or claim numerical or physical specimen qualification.
    """
    root = Path(root)
    study, prior = v5.validate_preparation(root)
    sources.validate_binding_manifest(root, inspect_sources=False)
    binding = json.loads(sources._read_bound(root, {
        'path': sources.BINDING_PATH, 'sha256': sources.BINDING_SHA256}, maximum=64 * 1024))
    ordered = [row['run_id'] for row in study['ordered_reference_runs']]
    if not isinstance(rows, dict) or set(rows) != set(ordered):
        raise ValueError('Exactly twelve declared v5 generated rows required')
    views, individual = {}, {}
    for run_id in ordered:
        key = binding['run_source_keys'][run_id]
        views[run_id], individual[run_id] = _view(
            study, prior, run_id, rows[run_id], binding['source_decks'][key]['sha256'])
    historical = _historical_n32(root, prior)
    result = _compare_views(prior, ordered, views, individual, historical)
    result['individual_generated_stream_pass'] = result.pop('individual_stream_pass')
    result['generated_diagnostic_screen_passed'] = result.pop('diagnostic_screen_passed')
    return {**result, 'schema': 'hbe-v5-twelve-row-generated-diagnostic-v1',
            'numerical_qualification_passed': False, 'native_execution_admitted': False,
            'adapted_deck_authenticated': False, 'source_deck_bytes_authenticated': False,
            'physical_validation_pass': None, 'calibration_released': False,
            'measured_response_accessed': False, 'patient_data_accessed': False}


def _compare_views(prior: dict, ordered: list, views: dict, individual: dict, historical: dict) -> dict:
    """Admission-neutral frozen numerical arithmetic, limits and classifications."""
    coverage = {run_id: v4.strict_coverage(
        prior, views[run_id]['branch'], v4.frozen_coordinates(prior, views[run_id]['branch']),
        views[run_id]['load_coordinate_m'],
        member_sha256=prior['coordinates'][views[run_id]['branch']]['member']['sha256'])
        for run_id in ordered}
    adjacent = {branch: {f'N{a}-N{b}': _signed_pair(views[_run(branch, a)], views[_run(branch, b)])
                         for a, b in pairs} for branch, pairs in ADJACENT.items()}
    adjacent['compression']['N36-S60-S120'] = _signed_pair(
        views[_run('compression', 36)], views[_run('compression', 36, 120)])
    adjacent['tension']['N24-S60-S120'] = _signed_pair(
        views[_run('tension', 24)], views[_run('tension', 24, 120)])
    frozen_pairs = {}
    for pair in prior['pairwise_checks']:
        result = v4.compare_pairwise_arrays(prior, pair['id'],
                                            views[pair['coarse_run']], views[pair['fine_run']])
        key = ('N' + str(views[pair['coarse_run']]['mesh_N']) + '-N' +
               str(views[pair['fine_run']]['mesh_N']) if pair['kind'] == 'mesh' else
               'N' + str(views[pair['coarse_run']]['mesh_N']) + '-S60-S120')
        adjacent_result = adjacent[views[pair['coarse_run']]['branch']][key]
        if (result['signed_force_difference_N'] != adjacent_result['signed_force_difference_N']
                or result['signed_probe_component_difference_m'] !=
                adjacent_result['signed_probe_component_difference_m']):
            raise ValueError('Frozen v4 pair and v5 signed common-state differences disagree')
        frozen_pairs[pair['id']] = result
    compression_groups = _groups(views, 'compression', COMPRESSION_TRIPLETS)
    tension_groups = _groups(views, 'tension', TENSION_TRIPLETS)
    latest_metrics = compression_groups['N24-N32-N36']['metrics']
    compression_required_spatial = {
        'reaction': latest_metrics['reaction'], 'motion': latest_metrics['motion'],
        'passed': all(_metric_pass(latest_metrics[key]) for key in ('reaction', 'motion')),
        'raw_trend_diagnostic_only': {key: latest_metrics[key]
                                      for key in ('reaction_trend', 'motion_trend')},
    }
    ungated_adjacent = {
        'compression': [key for key, pair in adjacent['compression'].items()
                        if key not in ('N32-N36', 'N36-S60-S120')
                        and not pair['pairwise_limit_passed']],
        'tension': [key for key, pair in adjacent['tension'].items()
                    if key not in ('N16-N24', 'N24-S60-S120')
                    and not pair['pairwise_limit_passed']],
    }
    compression_states = _compression_states(views)
    compression_fine = views[_run('compression', 36, 120)]
    tension_fine = views[_run('tension', 24, 120)]
    compression_mixed_groups = _groups(views, 'compression', COMPRESSION_TRIPLETS,
                                        replacement=compression_fine)
    tension_mixed_groups = _groups(views, 'tension', TENSION_TRIPLETS,
                                    replacement=tension_fine)
    compression_mixed_states = _compression_states(views, replacement=compression_fine)
    changed_states = [index for index, (old, new) in enumerate(zip(
        compression_states['states'], compression_mixed_states['states']))
        if compression_step.state_classification(old) != compression_step.state_classification(new)]
    changed_comp_groups = _classification_changes(compression_groups, compression_mixed_groups)
    changed_tension_groups = _classification_changes(tension_groups, tension_mixed_groups)
    temporal_comp = original_readout._refinement_group(
        views[_run('compression', 36)], views[_run('compression', 36)], compression_fine,
        kind='step')
    temporal_tension = original_readout._refinement_group(
        views[_run('tension', 24)], views[_run('tension', 24)], tension_fine,
        kind='step')
    compression_mixed_pass = (compression_mixed_states['passed_conditional_screen']
                              and not changed_states and not changed_comp_groups)
    tension_mixed_pass = not changed_tension_groups
    screen = (all(individual.values()) and all(pair['passed'] for pair in frozen_pairs.values())
              and compression_required_spatial['passed']
              and all(group['passed'] for group in tension_groups.values())
              and compression_states['passed_conditional_screen']
              and _metrics_pass(temporal_comp) and _metrics_pass(temporal_tension)
              and compression_mixed_pass and tension_mixed_pass)
    return {'ordered_run_ids': ordered, 'individual_stream_pass': individual,
            'coordinate_only_coverage': coverage, 'historical_N32_negative': historical,
            'all_adjacent_signed_differences': adjacent,
            'ungated_adjacent_limit_exceedances': ungated_adjacent,
            'four_frozen_v4_pair_diagnostics': frozen_pairs,
            'compression_spatial_groups': compression_groups,
            'compression_required_spatial': compression_required_spatial,
            'tension_spatial_groups': tension_groups,
            'compression_conditional': compression_states,
            'compression_temporal': {'metrics': temporal_comp,
                                     'passed': _metrics_pass(temporal_comp),
                                     'mixed_spatial_groups': compression_mixed_groups,
                                     'mixed_conditional': compression_mixed_states,
                                     'state_classification_changes': changed_states,
                                     'group_classification_changes': changed_comp_groups,
                                     'mixed_passed': compression_mixed_pass},
            'tension_temporal': {'metrics': temporal_tension,
                                 'passed': _metrics_pass(temporal_tension),
                                 'mixed_spatial_groups': tension_mixed_groups,
                                 'group_classification_changes': changed_tension_groups,
                                 'mixed_passed': tension_mixed_pass},
            'diagnostic_screen_passed': bool(screen),
            'continuum_error_bound': False}
