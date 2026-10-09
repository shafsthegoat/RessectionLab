"""Synthetic arrays and pinned coordinate metadata only; no measured/native IO."""
from copy import deepcopy
import hashlib
import io
import json
import math
from pathlib import Path
import subprocess
import zipfile
import zlib

import pytest

from scripts import mechanics_hbe_branch_calibration_v4 as core

ROOT = Path(__file__).resolve().parents[1]
STUDY = json.loads((ROOT / core.DECLARATION_PATH).read_text())


@pytest.fixture(autouse=True)
def forbid_measured_native_and_torque_io(monkeypatch):
    def forbidden(*args, **kwargs):
        pytest.fail('No measured archive, native process, or mesher may run in preparation tests')
    monkeypatch.setattr(subprocess, 'Popen', forbidden)
    monkeypatch.setattr(zipfile, 'ZipFile', forbidden)
    original = io.open
    safe_metadata = {ROOT / binding['path'] for binding in STUDY['evidence'].values()}

    def guarded_open(path, *args, **kwargs):
        if isinstance(path, (str, Path)):
            resolved = Path(path).resolve()
            if (resolved.is_relative_to(ROOT / 'data/mechanics')
                    or (resolved.is_relative_to(ROOT / 'outputs/mechanics') and resolved not in safe_metadata)):
                forbidden()
        return original(path, *args, **kwargs)
    monkeypatch.setattr(io, 'open', guarded_open)


@pytest.fixture
def study():
    return deepcopy(STUDY)


def row(study, run_id):
    x = core.planned_schedule(study, run_id)['full_coordinates_m']
    return {'run_id': run_id, 'load_coordinate_m': x, 'applied_force_N': [0.0] * len(x),
            'probe_displacements_m': [[[0.0, 0.0, 0.0] for _ in range(75)] for _ in x]}


def test_complete_preparation_metadata_authenticates_without_any_measured_or_native_io():
    checked = core.validate_preparation(ROOT)
    assert checked == STUDY
    assert checked['status'] == 'diagnostic_preparation_only_not_execution_ready'
    assert checked['future_qualification_dependency_not_release']['minimum_reference_call_count'] == 12
    assert len(checked['execution_blockers']) == 6
    assert checked['qualification_limits']['automatic_fit_release'] is False
    assert checked['planning_budget_not_release']['actual_release_caps_frozen'] is False
    assert checked['planning_budget_not_release']['remaining_proposed_overhead_seconds'] == 180


def test_immutable_declaration_hash_refuses_changed_tolerance_before_other_io(tmp_path):
    target = tmp_path / core.DECLARATION_PATH
    target.parent.mkdir(parents=True)
    changed = deepcopy(STUDY)
    changed['loading']['coverage_absolute_tolerance_m'] = 1e-5
    target.write_text(json.dumps(changed))
    with pytest.raises(ValueError, match='hash changed'):
        core.validate_preparation(tmp_path)


@pytest.mark.parametrize('branch', core.AXIAL)
def test_complete_coordinate_vectors_and_exact_decimal_endpoints(study, branch):
    coordinates = core.frozen_coordinates(study, branch)
    assert len(coordinates) == 30
    assert core.coordinate_hash(coordinates) == core.COORDINATE_HASHES[branch]
    assert coordinates[-1].hex() == study['coordinates'][branch]['endpoint_float64_hex']
    assert abs(coordinates[-1]) > .15 * study['scientific_invariants']['geometry']['height_m']


@pytest.mark.parametrize('branch', core.AXIAL)
@pytest.mark.parametrize('mutation', ['missing', 'duplicate', 'reversed', 'sign', 'mm', 'nonfinite', 'boolean', 'hash'])
def test_coordinate_vector_scope_errors_are_terminal(study, branch, mutation):
    item = study['coordinates'][branch]
    if mutation == 'missing': item['coordinates_m'].pop()
    elif mutation == 'duplicate': item['coordinates_m'][1] = item['coordinates_m'][0]
    elif mutation == 'reversed': item['coordinates_m'].reverse()
    elif mutation == 'sign': item['coordinates_m'] = [-x for x in item['coordinates_m']]
    elif mutation == 'mm': item['coordinate_unit'] = 'mm'
    elif mutation == 'nonfinite': item['coordinates_m'][3] = float('nan')
    elif mutation == 'boolean': item['coordinates_m'][3] = True
    elif mutation == 'hash': item['coordinate_sha256'] = '0' * 64
    with pytest.raises(ValueError):
        core.frozen_coordinates(study, branch)


@pytest.mark.parametrize('branch', core.AXIAL)
def test_csv_parser_does_not_convert_response_strings(study, branch):
    item = study['coordinates'][branch]
    tokens = [repr(x) for x in item['coordinates_m']]
    tokens[-1] = item['endpoint_decimal_literal_m']
    raw = ('displacement,force\n' + ''.join(x + ',MUST_NOT_BE_CONVERTED\n' for x in tokens)).encode()
    parsed = core._parse_coordinate_csv(raw, branch)
    assert parsed['coordinates_m'] == item['coordinates_m']
    assert parsed['response_values_used'] is False
    assert 'response' not in parsed
    # Public verifier rejects synthetic bytes against the real frozen member.
    with pytest.raises(ValueError, match='member identity'):
        core.verify_coordinate_payload(study, branch, item['member']['path'], raw)
    # Isolated synthetic hash-bound fixture exercises the success path without an archive read.
    item['member'].update(bytes=len(raw), sha256=hashlib.sha256(raw).hexdigest(), crc32=f'{zlib.crc32(raw):08x}')
    assert core.verify_coordinate_payload(study, branch, item['member']['path'], raw) == parsed
    with pytest.raises(ValueError, match='member identity'):
        core.verify_coordinate_payload(study, branch, 'HBE_01/HBE_01_03/torsion_pos_c3.csv', raw)


@pytest.mark.parametrize('mutation', ['row', 'header', 'extra_column', 'duplicate', 'nan', 'literal'])
def test_malformed_coordinate_csv_refused(mutation):
    lines = ['displacement,force'] + [f'{index * 1e-5},unused' for index in range(1, 31)]
    if mutation == 'row': lines.pop()
    elif mutation == 'header': lines[0] = 'force,displacement'
    elif mutation == 'extra_column': lines[4] += ',unexpected'
    elif mutation == 'duplicate': lines[5] = lines[4]
    elif mutation == 'nan': lines[4] = 'nan,unused'
    elif mutation == 'literal': lines[4] = 'nonsense,unused'
    with pytest.raises(ValueError):
        core._parse_coordinate_csv(('\n'.join(lines) + '\n').encode(), 'tension')


@pytest.mark.parametrize('run_id', [run['run_id'] for run in STUDY['ordered_reference_runs']])
def test_every_entire_extended_schedule_is_from_rest_and_endpoint_exact(study, run_id):
    plan = core.planned_schedule(study, run_id)
    actual = plan['full_coordinates_m']
    assert core.check_native_coordinate_array(study, run_id, actual) == actual
    assert actual[0] == 0.0
    assert actual[-1] == float(plan['endpoint_decimal_literal_m'])
    assert all(full == 2 * half for full, half in zip(actual, plan['half_coordinates_m']))
    assert plan['solve_entire_extended_branch'] is True
    for endpoint in (math.nextafter(actual[-1], 0.0), math.nextafter(actual[-1], math.copysign(math.inf, actual[-1]))):
        shifted = list(actual); shifted[-1] = endpoint
        with pytest.raises(ValueError, match='Native endpoint differs'):
            core.check_native_coordinate_array(study, run_id, shifted)
    missing = actual[:-1]
    with pytest.raises(ValueError, match='Every native state'):
        core.check_native_coordinate_array(study, run_id, missing)
    old = [value * (.15 * .00489159 / abs(actual[-1])) for value in actual]
    with pytest.raises(ValueError, match='Native endpoint differs'):
        core.check_native_coordinate_array(study, run_id, old)


@pytest.mark.parametrize('branch', core.AXIAL)
def test_all_original_rows_exact_new_endpoint_pass_old_and_inward_endpoints_fail(study, branch):
    item = study['coordinates'][branch]
    x = item['coordinates_m']
    covered = core.strict_coverage(study, branch, x, [0.0, x[-1]], member_sha256=item['member']['sha256'])
    assert covered['covered_row_count'] == 30
    assert covered['outside_rows'] == []
    assert covered['coverage_tolerance_m'] == 0.0
    old = math.copysign(.15 * .00489159, x[-1])
    for end in (old, math.nextafter(x[-1], 0.0)):
        with pytest.raises(ValueError, match=r'outside solved interval: \[30\]'):
            core.strict_coverage(study, branch, x, [0.0, end], member_sha256=item['member']['sha256'])
    outward_query = list(x)
    outward_query[-1] = math.nextafter(x[-1], math.copysign(math.inf, x[-1]))
    with pytest.raises(ValueError, match='all coordinate rows'):
        core.strict_coverage(study, branch, outward_query, [0.0, x[-1]], member_sha256=item['member']['sha256'])
    with pytest.raises(ValueError, match='all coordinate rows'):
        core.strict_coverage(study, branch, x, [0.0, x[-1]], member_sha256='0' * 64)


@pytest.mark.parametrize('branch', core.AXIAL)
def test_independent_support_primitive_exact_endpoints_and_one_ulp_outward(branch):
    endpoint = float(core.ENDPOINTS[branch])
    low, high = sorted([0.0, endpoint])
    assert core.strict_closed_interval([low, high], [0.0, endpoint]) == (low, high)
    for query in [math.nextafter(low, -math.inf), math.nextafter(high, math.inf)]:
        with pytest.raises(ValueError, match='outside solved interval'):
            core.strict_closed_interval([query], [0.0, endpoint])


@pytest.mark.parametrize('pair_id', [pair['id'] for pair in STUDY['pairwise_checks']])
def test_pairwise_signed_arrays_and_original_limits_at_all_61_states_75_probes(study, pair_id):
    pair = next(p for p in study['pairwise_checks'] if p['id'] == pair_id)
    coarse = row(study, pair['coarse_run']); fine = row(study, pair['fine_run'])
    coarse['applied_force_N'] = [1e-4] * len(coarse['applied_force_N'])
    fine['applied_force_N'] = [1e-4 + 1e-8] * len(fine['applied_force_N'])
    fine['probe_displacements_m'][-1][-1] = [-1e-8, 2e-8, -2e-8]
    report = core.compare_pairwise_arrays(study, pair_id, coarse, fine)
    floor, relative, motion = core.PAIR_LIMITS[pair['kind']]
    assert report['force_limit_N'] == floor + relative * (1e-4 + 1e-8)
    assert report['probe_norm_limit_m'] == motion
    assert report['signed_force_difference_N'][-1] == pytest.approx(1e-8)
    assert report['signed_probe_component_difference_m'][-1][-1] == [-1e-8, 2e-8, -2e-8]
    assert report['maximum_probe_norm_change_m'] == pytest.approx(3e-8)
    assert len(report['signed_probe_component_difference_m']) == 61
    assert all(len(state) == 75 for state in report['signed_probe_component_difference_m'])
    assert report['passed'] is True and report['diagnostic_only'] is True
    assert report['native_primitives_authenticated'] is False
    assert report['calibration_released'] is False and report['continuum_accuracy_claim'] is False


@pytest.mark.parametrize('kind,pair_id', [('mesh', 'compression_spatial'), ('step', 'tension_temporal')])
@pytest.mark.parametrize('criterion', ['force', 'probe_norm'])
def test_one_ulp_over_existing_pairwise_limit_fails_without_tolerance_inflation(study, kind, pair_id, criterion):
    pair = next(p for p in study['pairwise_checks'] if p['id'] == pair_id)
    coarse = row(study, pair['coarse_run']); fine = row(study, pair['fine_run'])
    floor, _, motion = core.PAIR_LIMITS[kind]
    # Fine response remains zero: the allowed difference is exactly the frozen floor.
    if criterion == 'force': coarse['applied_force_N'][-1] = floor
    else: coarse['probe_displacements_m'][-1][-1][0] = motion
    assert core.compare_pairwise_arrays(study, pair_id, coarse, fine)['passed'] is True
    if criterion == 'force': coarse['applied_force_N'][-1] = math.nextafter(floor, math.inf)
    else: coarse['probe_displacements_m'][-1][-1][0] = math.nextafter(motion, math.inf)
    assert core.compare_pairwise_arrays(study, pair_id, coarse, fine)['passed'] is False


def test_temporal_force_denominator_uses_only_common_states(study):
    pair = next(p for p in study['pairwise_checks'] if p['id'] == 'compression_temporal')
    coarse = row(study, pair['coarse_run']); fine = row(study, pair['fine_run'])
    fine['applied_force_N'][1] = 0.25  # An unmatched S120 state is excluded.
    report = core.compare_pairwise_arrays(study, pair['id'], coarse, fine)
    assert report['fine_state_indices'] == list(range(0, 121, 2))
    assert report['force_limit_N'] == 1.6e-8
    assert report['maximum_force_change_N'] == 0.0
    coarse['applied_force_N'][-1] = 1e-4
    assert core.compare_pairwise_arrays(study, pair['id'], coarse, fine)['passed'] is False


@pytest.mark.parametrize('change', ['run', 'missing_force', 'missing_probe', 'extra_probe', 'nonfinite', 'coordinate'])
def test_pair_scope_errors_fail_closed_before_diagnostics(study, change):
    pair = study['pairwise_checks'][0]
    coarse = row(study, pair['coarse_run']); fine = row(study, pair['fine_run'])
    if change == 'run': fine['run_id'] = coarse['run_id']
    elif change == 'missing_force': fine['applied_force_N'].pop()
    elif change == 'missing_probe': fine['probe_displacements_m'][20].pop()
    elif change == 'extra_probe': fine['probe_displacements_m'][20].append([0.0] * 3)
    elif change == 'nonfinite': fine['probe_displacements_m'][20][0][1] = float('inf')
    elif change == 'coordinate': fine['load_coordinate_m'][12] += 1e-7
    with pytest.raises(ValueError):
        core.compare_pairwise_arrays(study, pair['id'], coarse, fine)


@pytest.mark.parametrize('gate', [core.require_execution_ready, core.require_fit_eligible, core.require_torque_access])
@pytest.mark.parametrize('claimed_status', ['all_six_passed', 'failed', 'authorized', 'qualified', 'frozen'])
def test_no_claimed_status_or_forged_release_opens_execution_fit_or_holdout(gate, claimed_status):
    with pytest.raises(ValueError):
        gate({'authorized': True, 'status': claimed_status, 'passed': True, 'native_calls': 6})


def test_six_candidate_does_not_silently_stand_for_twelve_or_stronger_gates(study):
    dependency = study['future_qualification_dependency_not_release']
    assert dependency['additional_to_six_candidate'] == [
        'compression:N8:S60:reference', 'compression:N12:S60:reference',
        'compression:N16:S60:reference', 'compression:N24:S60:reference',
        'tension:N8:S60:reference', 'tension:N12:S60:reference']
    assert dependency['not_implemented_or_authorized'] is True
    assert 'two_latest_limit_remaining_error_envelope' in dependency['original_gates_required']
    assert 'mixed_step_envelope_and_classification_stability' in dependency['original_gates_required']
    for run_id in dependency['additional_to_six_candidate']:
        with pytest.raises(ValueError, match='Exact declared reference run'):
            core.run_spec(study, run_id)
