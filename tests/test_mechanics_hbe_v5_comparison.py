"""Generated array diagnostics only: no native solve or measured HBE responses."""
from copy import deepcopy
import json
import math
from pathlib import Path
import subprocess
import zipfile

import pytest

from scripts import mechanics_hbe_branch_calibration_v5 as v5
from scripts import mechanics_hbe_v5_comparison as comparison
from scripts import mechanics_hbe_v5_source_bindings as sources


ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(autouse=True)
def no_native_or_measured_access(monkeypatch):
    def forbidden(*args, **kwargs):
        pytest.fail('Generated comparator must not run FEBio or open measured archives')
    monkeypatch.setattr(subprocess, 'Popen', forbidden)
    monkeypatch.setattr(zipfile, 'ZipFile', forbidden)
    original_open = Path.open

    def restricted_open(path, *args, **kwargs):
        resolved = path.resolve()
        if resolved.is_relative_to(ROOT / 'data/mechanics') or resolved.is_relative_to(ROOT / 'outputs/mechanics'):
            forbidden()
        return original_open(path, *args, **kwargs)
    monkeypatch.setattr(Path, 'open', restricted_open)


def generated_rows():
    study, prior = v5.validate_preparation(ROOT)
    binding = json.loads((ROOT / sources.BINDING_PATH).read_text())
    rows = {}
    for spec in study['ordered_reference_runs']:
        run_id, branch, n, steps = (spec['run_id'], spec['branch'], spec['N'], spec['steps'])
        coordinates = v5.schedule(study, prior, run_id)['full_coordinates_m']
        endpoint = coordinates[-1]
        sign = -1 if branch == 'compression' else 1
        force = [sign * .03 * (i / steps) + .1 * (i / steps) / n**2
                 for i in range(steps + 1)]
        probes = [[[0., 0., .1 * endpoint * (i / steps) + 1e-5 * (i / steps) / n**2]
                   for _ in range(75)] for i in range(steps + 1)]
        key = binding['run_source_keys'][run_id]
        half = spec['native_domain'] == 'lower_half_reconstructed'
        rows[run_id] = {
            'schema': 'hbe-v5-complete-stream-v1', 'run_id': run_id,
            'steps': steps, 'frame_count': steps + 1,
            'representation': 'reconstructed_full' if half else 'full_native_fixture',
            'numerical_passed': True, 'solver': {'passed': True},
            'full_energy_work': {'passed': True},
            'native_energy_work': {'passed': True} if half else None,
            'criteria_max_ratio': {'fixture_only': .1},
            'minimum_sampled_J': .9, 'minimum_logged_J': .9,
            'load_coordinate_full_m': coordinates,
            'applied_force_N': force, 'probe_displacements_m': probes,
            'provenance': {
                'v5_declaration_sha256': v5.DECLARATION_SHA256,
                'source_deck_sha256': binding['source_decks'][key]['sha256'],
                'generated_fixture_only': True, 'native_output_observed': False,
                'measured_response_accessed': False, 'patient_data_accessed': False,
                'physical_validation_pass': None},
        }
    return rows


def test_all_twelve_generated_rows_pass_diagnostic_but_never_qualify():
    result = comparison.compare_generated_rows(ROOT, generated_rows())
    assert result['generated_diagnostic_screen_passed'] is True
    assert len(result['ordered_run_ids']) == 12
    assert len(result['four_frozen_v4_pair_diagnostics']) == 4
    assert len(result['compression_spatial_groups']) == 4
    assert len(result['tension_spatial_groups']) == 2
    assert result['compression_conditional']['passed_conditional_screen'] is True
    assert set(result['compression_required_spatial']['raw_trend_diagnostic_only']) == {
        'reaction_trend', 'motion_trend'}
    assert result['compression_temporal']['mixed_passed'] is True
    assert result['tension_temporal']['mixed_passed'] is True
    assert all(row['covered_row_count'] == 30 for row in result['coordinate_only_coverage'].values())
    assert result['historical_N32_negative']['status'] == 'remaining_spatial_error_not_within_allowance'
    assert result['historical_N32_negative']['historical_only_not_v5_veto'] is True
    assert result['numerical_qualification_passed'] is False
    assert result['native_execution_admitted'] is False
    assert result['adapted_deck_authenticated'] is False
    assert result['source_deck_bytes_authenticated'] is False
    assert result['physical_validation_pass'] is None
    assert result['calibration_released'] is False


def test_odd_s120_spike_cannot_enlarge_common_state_denominator():
    rows = generated_rows()
    run_id = comparison._run('tension', 24, 120)
    baseline = rows[run_id]['applied_force_N'][120]
    rows[run_id]['applied_force_N'][120] = baseline + .0002
    rows[run_id]['applied_force_N'][119] = 1e6
    result = comparison.compare_generated_rows(ROOT, rows)
    pair = result['four_frozen_v4_pair_diagnostics']['tension_temporal']
    assert pair['signed_force_difference_N'][-1] == pytest.approx(.0002)
    assert pair['force_limit_N'] < .0001
    assert pair['passed'] is False
    assert result['generated_diagnostic_screen_passed'] is False
    assert result['numerical_qualification_passed'] is False


def test_exact_endpoint_rejects_one_ulp_inward():
    rows = generated_rows()
    run_id = comparison._run('tension', 24, 120)
    coordinates = rows[run_id]['load_coordinate_full_m']
    coordinates[-1] = math.nextafter(coordinates[-1], 0.)
    with pytest.raises(ValueError, match='Exact v5 full-coordinate'):
        comparison.compare_generated_rows(ROOT, rows)


@pytest.mark.parametrize('mutation', ('missing', 'wrong_representation', 'native_claim', 'wrong_source'))
def test_identity_and_provenance_mutations_fail_closed(mutation):
    rows = generated_rows()
    run_id = comparison._run('compression', 8)
    if mutation == 'missing':
        rows.pop(run_id)
    elif mutation == 'wrong_representation':
        rows[run_id]['representation'] = 'reconstructed_full'
    elif mutation == 'native_claim':
        rows[run_id]['provenance']['native_output_observed'] = True
    else:
        rows[run_id]['provenance']['source_deck_sha256'] = '0' * 64
    with pytest.raises(ValueError):
        comparison.compare_generated_rows(ROOT, rows)


def test_signed_probe_damage_fails_frozen_pair_and_group():
    rows = generated_rows()
    run_id = comparison._run('tension', 24)
    for vector in rows[run_id]['probe_displacements_m'][-1]:
        vector[0] += 1e-5
    result = comparison.compare_generated_rows(ROOT, rows)
    pair = result['four_frozen_v4_pair_diagnostics']['tension_spatial']
    assert pair['signed_probe_component_difference_m'][-1][0][0] == pytest.approx(1e-5)
    assert pair['passed'] is False
    assert result['generated_diagnostic_screen_passed'] is False


def test_compression_order_or_envelope_failure_is_retained():
    rows = generated_rows()
    run_id = comparison._run('compression', 36)
    rows[run_id]['applied_force_N'][-1] += .001
    result = comparison.compare_generated_rows(ROOT, rows)
    assert result['compression_conditional']['passed_conditional_screen'] is False
    assert result['compression_conditional']['unresolved_nonrest_states'] or \
        result['compression_conditional']['remaining_exceedance_states'] or \
        result['compression_conditional']['instability_states']
    assert result['generated_diagnostic_screen_passed'] is False


def test_tension_mixed_step_classification_change_is_visible():
    rows = generated_rows()
    run_id = comparison._run('tension', 24, 120)
    rows[run_id]['applied_force_N'][-1] += .002
    result = comparison.compare_generated_rows(ROOT, rows)
    assert result['tension_temporal']['group_classification_changes']
    assert result['tension_temporal']['mixed_passed'] is False
    assert result['generated_diagnostic_screen_passed'] is False


def test_original_mesh_trend_is_strict_above_floor():
    rows = generated_rows()
    base, increment = .03125, 1 / 8192
    for n, factor in ((8, 2), (12, 1.5), (16, 1), (24, 0)):
        run_id = comparison._run('tension', n)
        rows[run_id]['applied_force_N'] = [(base + factor * increment) * (i / 60)
                                            for i in range(61)]
    rows[comparison._run('tension', 24, 120)]['applied_force_N'] = [base * (i / 120)
                                                                    for i in range(121)]
    result = comparison.compare_generated_rows(ROOT, rows)
    metric = result['tension_spatial_groups']['N8-N16-N24']['metrics']['reaction_trend']
    assert metric['comparison'] == 'lt'
    assert metric['actual'] == metric['limit'] == increment
    assert result['tension_spatial_groups']['N8-N16-N24']['passed'] is False
    assert result['generated_diagnostic_screen_passed'] is False


def test_old_compression_coarse_triplet_failure_remains_nongating():
    rows = generated_rows()
    rows[comparison._run('compression', 8)]['applied_force_N'] = deepcopy(
        rows[comparison._run('compression', 12)]['applied_force_N'])
    result = comparison.compare_generated_rows(ROOT, rows)
    assert result['compression_spatial_groups']['N8-N12-N16']['passed'] is False
    assert result['compression_conditional']['passed_conditional_screen'] is True
    assert result['compression_required_spatial']['passed'] is True
    assert result['generated_diagnostic_screen_passed'] is True
    assert result['numerical_qualification_passed'] is False


def test_ungated_tension_n8_sign_reversal_is_reported_as_limitation():
    rows = generated_rows()
    run_id = comparison._run('tension', 8)
    rows[run_id]['applied_force_N'] = [-value for value in rows[run_id]['applied_force_N']]
    result = comparison.compare_generated_rows(ROOT, rows)
    assert 'N8-N12' in result['ungated_adjacent_limit_exceedances']['tension']
    assert result['tension_spatial_groups']['N8-N16-N24']['passed'] is True
    assert result['generated_diagnostic_screen_passed'] is True
    assert result['numerical_qualification_passed'] is False


def test_unbound_adapted_hash_cannot_claim_authentication():
    rows = generated_rows()
    rows[comparison._run('compression', 8)]['provenance']['adapted_deck_sha256'] = '0' * 64
    result = comparison.compare_generated_rows(ROOT, rows)
    assert result['generated_diagnostic_screen_passed'] is True
    assert result['adapted_deck_authenticated'] is False
    assert result['native_execution_admitted'] is False
