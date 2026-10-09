"""Coordinate-coverage preparation and pure diagnostic guards, with execution closed.

This module has no archive opener, fit, native launcher, mesher, or torque reader.
Array diagnostics do not authenticate native primitives or qualify a calibration.
The old v3 primitive checker is deliberately not called: it assumes +/-15%.
"""
from copy import deepcopy
import csv
from decimal import Decimal, InvalidOperation
import hashlib
import io
import json
import math
import zlib

from scripts import mechanics_hbe_access as access

DECLARATION_PATH = 'manifests/experiments/hbe-01-03-branch-calibration-v4.json'
DECLARATION_SHA256 = '85ed9ca8cb1a9048e885f27424678e97f20cb6dd2e2cbb5cfd6dccfe6dafb52b'
AXIAL = ('compression', 'tension')
ENDPOINTS = {'compression': '-0.00073726', 'tension': '0.0007360099999999'}
COORDINATE_HASHES = {
    'compression': '881665105d646fbc47a116b0e6bcad7d4ad79f26d43b386c8f297e3b4624fa2e',
    'tension': 'a1690a5cff8fb75fc619984074ba53dbb044f3ea6894f72f698cb36b1edc0731',
}
RUNS = (('compression', 32, 60), ('compression', 36, 60), ('compression', 36, 120),
        ('tension', 16, 60), ('tension', 24, 60), ('tension', 24, 120))
PAIR_LIMITS = {'mesh': (1.6e-7, .02, 8e-6), 'step': (1.6e-8, .002, 8e-7)}


def _finite(value):
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise ValueError('Finite numeric values required; booleans are not coordinates')
    return float(value)


def coordinate_hash(values):
    values = [_finite(value) for value in values]
    return hashlib.sha256(json.dumps(values, separators=(',', ':'), allow_nan=False).encode()).hexdigest()


def _branch(branch):
    if branch not in AXIAL:
        raise ValueError('Only the exact two axial coordinate branches are permitted')
    return -1 if branch == 'compression' else 1


def _coordinate_vector(branch, values, count):
    sign = _branch(branch)
    if not isinstance(values, (list, tuple)) or len(values) != count:
        raise ValueError('Complete original coordinate row count required')
    values = [_finite(value) for value in values]
    if any(sign * value <= 0 for value in values):
        raise ValueError('Strictly signed axial coordinates in metres required')
    if any(sign * (right - left) <= 0 for left, right in zip(values, values[1:])):
        raise ValueError('Coordinates must be unique and strictly monotone in loading order')
    return values


def frozen_coordinates(study, branch):
    """Recheck complete 30-value vectors against fixed diagnostic hashes."""
    _branch(branch)
    item = study['coordinates'][branch]
    values = _coordinate_vector(branch, item['coordinates_m'], 30)
    if (item['coordinate_unit'] != 'm' or item['coordinate_column'] != 0
            or item['header'] != ['displacement', 'force'] or type(item['row_count']) is not int
            or item['row_count'] != 30 or item['sign'] != ('negative' if branch == 'compression' else 'positive')
            or item['coordinate_sha256'] != COORDINATE_HASHES[branch]
            or coordinate_hash(values) != COORDINATE_HASHES[branch]
            or item['endpoint_decimal_literal_m'] != ENDPOINTS[branch]
            or item['endpoint_float64_m'] != float(ENDPOINTS[branch])
            or item['endpoint_float64_hex'] != float(ENDPOINTS[branch]).hex()
            or values[-1] != item['endpoint_float64_m']
            or (min(values), max(values)) != (item['minimum_m'], item['maximum_m'])
            or item['rest_coordinate_m'] != 0.0):
        raise ValueError('Frozen axial coordinate identity, units, hash or endpoint differs')
    return values


def _parse_coordinate_csv(payload, branch):
    """Pure bytes parser; response strings are transported but never converted or returned.

    This low-level routine does not authorize reading a file. The public payload
    verifier below additionally binds full member bytes and the coordinate vector.
    """
    _branch(branch)
    if not isinstance(payload, bytes) or len(payload) > 4096:
        raise ValueError('Bounded axial CSV bytes required')
    rows = list(csv.reader(io.StringIO(payload.decode('utf-8', errors='strict')), strict=True))
    if len(rows) != 31 or rows[0] != ['displacement', 'force'] or any(len(row) != 2 for row in rows):
        raise ValueError('Exact axial header, 30 rows and two columns required')
    tokens = [row[0] for row in rows[1:]]
    try:
        if any(not token or token != token.strip() or not Decimal(token).is_finite() for token in tokens):
            raise ValueError('Finite unpadded coordinate literals required')
        values = _coordinate_vector(branch, [float(token) for token in tokens], 30)
    except (InvalidOperation, OverflowError) as error:
        raise ValueError('Malformed axial coordinate literal') from error
    return {'coordinates_m': values, 'coordinate_sha256': coordinate_hash(values),
            'endpoint_decimal_literal_m': tokens[-1], 'response_values_used': False}


def verify_coordinate_payload(study, branch, member_path, payload):
    """Verify already-supplied allowlisted bytes, without opening any member itself."""
    values = frozen_coordinates(study, branch)
    expected = study['coordinates'][branch]['member']
    if (member_path != f'HBE_01/HBE_01_03/{branch}_c3.csv' or member_path != expected['path']
            or not isinstance(payload, bytes) or len(payload) != expected['bytes']
            or hashlib.sha256(payload).hexdigest() != expected['sha256']
            or f'{zlib.crc32(payload):08x}' != expected['crc32']):
        raise ValueError('Exact allowlisted axial member identity, size and hashes required')
    parsed = _parse_coordinate_csv(payload, branch)
    if (parsed['coordinates_m'] != values
            or parsed['coordinate_sha256'] != COORDINATE_HASHES[branch]
            or parsed['endpoint_decimal_literal_m'] != ENDPOINTS[branch]):
        raise ValueError('CSV coordinate vector or literal drifted from the frozen declaration')
    return parsed


def run_spec(study, run_id):
    matches = [run for run in study['ordered_reference_runs'] if run['run_id'] == run_id]
    if len(matches) != 1:
        raise ValueError('Exact declared reference run required')
    run = matches[0]
    if (run['branch'], run['N'], run['steps']) not in RUNS:
        raise ValueError('Undeclared reference mesh or schedule')
    if (type(run['N']) is not int or type(run['steps']) is not int
            or run['frame_count'] != run['steps'] + 1 or run['mu_Pa'] != 1000.0
            or run['endpoint_decimal_literal_m'] != ENDPOINTS[run['branch']]
            or run['start_from_rest'] is not True or run['solve_entire_extended_branch'] is not True
            or run_id != f'{run["branch"]}:N{run["N"]}:S{run["steps"]}:reference'):
        raise ValueError('Full extended reference run identity differs')
    return deepcopy(run)


def planned_schedule(study, run_id):
    """Coordinates for an entire fresh branch; this does not construct a native deck."""
    run = run_spec(study, run_id)
    frozen_coordinates(study, run['branch'])
    endpoint = float(run['endpoint_decimal_literal_m'])
    times = [index / run['steps'] for index in range(run['steps'] + 1)]
    return {'pseudotimes': times, 'full_coordinates_m': [endpoint * value for value in times],
            'half_coordinates_m': [endpoint * value / 2 for value in times],
            'endpoint_decimal_literal_m': run['endpoint_decimal_literal_m'],
            'endpoint_float64_hex': endpoint.hex(), 'solve_entire_extended_branch': True}


def check_native_coordinate_array(study, run_id, coordinates):
    """Check externally reconstructed coordinates, never fill them from this schedule.

    The native primitive reconstruction needed to supply this input is deliberately
    unavailable in v4 preparation. Interior grid consistency uses the prior tiny
    tolerance; both endpoints and subsequent measured-row coverage use no tolerance.
    """
    run = run_spec(study, run_id)
    expected = planned_schedule(study, run_id)['full_coordinates_m']
    if not isinstance(coordinates, (list, tuple)) or len(coordinates) != len(expected):
        raise ValueError('Every native state in the complete new domain is required')
    actual = [_finite(value) for value in coordinates]
    sign = _branch(run['branch'])
    if actual[0] != 0.0 or actual[-1] != expected[-1]:
        raise ValueError('Native endpoint differs; no inward rounding, margin or tolerance')
    if any(sign * (b - a) <= 0 for a, b in zip(actual, actual[1:])):
        raise ValueError('Native loading coordinates must be strictly monotone from rest')
    if any(not math.isclose(a, b, rel_tol=2e-14, abs_tol=1e-18) for a, b in zip(actual, expected)):
        raise ValueError('Every native coordinate must match the declared equal-step schedule')
    return actual


def strict_closed_interval(queries, solved_coordinates):
    """Independent support primitive: exact IEEE-754 ordering, with no tolerance."""
    if not isinstance(solved_coordinates, (list, tuple)) or len(solved_coordinates) < 2:
        raise ValueError('Independently reconstructed solved interval required')
    solved = [_finite(value) for value in solved_coordinates]
    values = [_finite(value) for value in queries]
    low, high = min(solved), max(solved)
    outside = [index + 1 for index, value in enumerate(values) if value < low or value > high]
    if outside:
        raise ValueError(f'Original axial rows outside solved interval: {outside}; no extrapolation')
    return low, high


def strict_coverage(study, branch, coordinates, solved_coordinates, *, member_sha256):
    """All original rows, strict closed interval, no tolerance or response values."""
    expected = frozen_coordinates(study, branch)
    values = _coordinate_vector(branch, coordinates, 30)
    if (values != expected or coordinate_hash(values) != COORDINATE_HASHES[branch]
            or member_sha256 != study['coordinates'][branch]['member']['sha256']):
        raise ValueError('Original member and all coordinate rows must remain hash-identical')
    if not isinstance(solved_coordinates, (list, tuple)) or len(solved_coordinates) < 2:
        raise ValueError('Independently reconstructed solved interval required')
    solved = [_finite(value) for value in solved_coordinates]
    sign = _branch(branch)
    if solved[0] != 0.0 or any(sign * (b - a) <= 0 for a, b in zip(solved, solved[1:])):
        raise ValueError('Solved interval must follow the branch strictly from rest')
    low, high = strict_closed_interval(values, solved)
    return {'branch': branch, 'row_count': 30, 'covered_row_count': 30, 'outside_rows': [],
            'solved_min_m': low, 'solved_max_m': high, 'coordinate_sha256': coordinate_hash(values),
            'member_sha256': member_sha256, 'coverage_tolerance_m': 0.0,
            'coverage_only_not_numerical_qualification': True}


def _arrays(study, run_id, row):
    spec = run_spec(study, run_id)
    if row.get('run_id') != run_id:
        raise ValueError('Pair input run identity differs')
    coordinates = check_native_coordinate_array(study, run_id, row['load_coordinate_m'])
    force = [_finite(value) for value in row['applied_force_N']]
    probes = row['probe_displacements_m']
    if len(force) != spec['frame_count'] or len(probes) != spec['frame_count']:
        raise ValueError('All native force and probe states required')
    checked = []
    for state in probes:
        if not isinstance(state, (list, tuple)) or len(state) != 75:
            raise ValueError('All 75 original physical probes required at every state')
        values = []
        for vector in state:
            if not isinstance(vector, (list, tuple)) or len(vector) != 3:
                raise ValueError('Three signed probe displacement components required')
            values.append([_finite(value) for value in vector])
        checked.append(values)
    return spec, coordinates, force, checked


def compare_pairwise_arrays(study, pair_id, coarse, fine):
    """Pure 61-state/75-probe diagnostic; no three-level trend or fit admission."""
    pairs = [pair for pair in study['pairwise_checks'] if pair['id'] == pair_id]
    if len(pairs) != 1:
        raise ValueError('Exact declared branch pair required')
    pair = pairs[0]
    low, x0, force0, probes0 = _arrays(study, pair['coarse_run'], coarse)
    high, x1, force1, probes1 = _arrays(study, pair['fine_run'], fine)
    if low['branch'] != high['branch'] or low['steps'] != 60:
        raise ValueError('Same-branch S60 baseline required')
    kind = pair['kind']
    if kind == 'step' and (low['N'] != high['N'] or high['steps'] != 120):
        raise ValueError('Temporal pair must preserve mesh and compare S60/S120')
    if kind == 'mesh' and (low['N'] >= high['N'] or high['steps'] != 60):
        raise ValueError('Spatial pair must preserve S60 and compare increasing meshes')
    if kind not in PAIR_LIMITS:
        raise ValueError('Only frozen mesh or step pair diagnostics exist')
    stride = high['steps'] // low['steps']
    indices = list(range(0, high['steps'] + 1, stride))
    if len(indices) != 61 or any(not math.isclose(x0[i], x1[j], rel_tol=2e-14, abs_tol=1e-18)
                                 for i, j in enumerate(indices)):
        raise ValueError('All 61 common physical loading coordinates required')
    force_delta = [force1[j] - force0[i] for i, j in enumerate(indices)]
    probe_delta = [[[probes1[j][p][axis] - probes0[i][p][axis] for axis in range(3)]
                    for p in range(75)] for i, j in enumerate(indices)]
    force_change = max(abs(value) for value in force_delta)
    probe_change = max(math.hypot(*vector) for state in probe_delta for vector in state)
    floor, relative, probe_limit = PAIR_LIMITS[kind]
    # The comparator only observes the 61 shared loading states. An unmatched
    # S120 midpoint must not enlarge the allowed difference at those states.
    force_limit = floor + relative * max(abs(force1[j]) for j in indices)
    return {'pair_id': pair_id, 'kind': kind, 'common_state_count': 61, 'probe_count': 75,
            'coarse_run': pair['coarse_run'], 'fine_run': pair['fine_run'],
            'fine_state_indices': indices, 'signed_force_difference_N': force_delta,
            'signed_probe_component_difference_m': probe_delta,
            'maximum_force_change_N': force_change, 'force_limit_N': force_limit,
            'maximum_probe_norm_change_m': probe_change, 'probe_norm_limit_m': probe_limit,
            'passed': force_change <= force_limit and probe_change <= probe_limit,
            'native_primitives_authenticated': False, 'calibration_released': False,
            'continuum_accuracy_claim': False, 'diagnostic_only': True}


def validate_preparation(root):
    """Authenticate metadata, invariants and coordinate history; no measured archive IO."""
    study = access.verify_binding(root, {'path': DECLARATION_PATH, 'sha256': DECLARATION_SHA256},
                                  maximum_bytes=1024**2, read_json=True)
    previous = access.verify_binding(root, study['previous_study'], maximum_bytes=1024**2, read_json=True)
    for key, value in study['preserved_v3_fields'].items():
        if access.canonical_json(value) != access.canonical_json(previous[key]):
            raise ValueError('Inherited physical, role, torsion or scale contract differs')
    protocol = access.verify_binding(root, previous['protocol'], maximum_bytes=1024**2, read_json=True)
    for key, value in study['scientific_invariants'].items():
        if access.canonical_json(value) != access.canonical_json(protocol[key]):
            raise ValueError('Original geometry, material, weighting or metrics changed')
    evidence = {}
    for key, binding in study['evidence'].items():
        evidence[key] = access.verify_binding(root, binding, maximum_bytes=8*1024**2,
                                             read_json=key not in ('proposal', 'v3_access_ledger'))
    diagnosis = evidence['coordinate_diagnosis']
    state = evidence['v3_state']
    if (state.get('status') != 'failed_or_incomplete' or state.get('calibration_responses_accessed') is not True
            or state.get('held_out_access_attempted') is not False or state.get('held_out_responses_accessed') is not False
            or type(state.get('native_calls')) is not int or state['native_calls'] != 0
            or type(state.get('mesher_calls')) is not int or state['mesher_calls'] != 0
            or 'fit' in state or 'freeze' in state
            or state.get('error') != {'type': 'ValueError', 'message': 'Measured coordinate outside solved range; no extrapolation'}):
        raise ValueError('Preserve the v3 terminal no-fit failure and held-out boundary')
    for branch in AXIAL:
        coordinates = frozen_coordinates(study, branch)
        prior = diagnosis['branches'][branch]
        if (coordinates != prior['measured_coordinates_m']
                or study['coordinates'][branch]['member'] != prior['member']
                or prior['response_values_used'] is not False):
            raise ValueError('Coordinate-only diagnostic binding differs')
    if [(r['branch'], r['N'], r['steps']) for r in study['ordered_reference_runs']] != list(RUNS):
        raise ValueError('Exact prospective six-call order required')
    for run in study['ordered_reference_runs']:
        planned_schedule(study, run['run_id'])
    if (study['status'] != 'diagnostic_preparation_only_not_execution_ready'
            or study['qualification_limits']['automatic_fit_release'] is not False
            or study['phase_contract']['reference_diagnostic']['released'] is not False
            or study['phase_contract']['later_calibration']['available_in_this_version'] is not False):
        raise ValueError('Diagnostic-only preparation cannot authorize execution or fit')
    review = evidence['independent_protocol_review']
    if (review['execution_release_approved'] is not False
            or review['six_calls_sufficient_for_v3_equivalent_branch_qualification'] is not False
            or review['stronger_reference_call_count_to_reproduce_prior_branch_contracts'] != 12):
        raise ValueError('Independent no-release and twelve-reference dependency must remain explicit')
    return study


def require_execution_ready(*args, **kwargs):
    """Explicit fail-closed boundary; a JSON release cannot bypass missing source."""
    raise ValueError('v4 preparation has no native primitive adapter or supervised execution path; no execution release')


def require_fit_eligible(*args, **kwargs):
    """Even perfect pairwise arrays cannot substitute for omitted spatial safeguards."""
    raise ValueError('v4 six-call diagnostic cannot release fitting; separately preregister same-endpoint multi-level qualification')


def require_torque_access(*args, **kwargs):
    raise ValueError('v4 preparation cannot access torque; no qualified fit, paired confirmations or durable prediction freeze')
