"""Pure algebra, refusal and authenticated saved-geometry controls only.

No native execution, study preparation, response-curve parsing, fabricated logs,
generated specimen examples or calibration data are permitted by these tests.
"""
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import sys
import xml.etree.ElementTree as ET

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts import mechanics_hbe_halfheight_boundary as core
from scripts import mechanics_hbe_halfheight_boundary_readout as reader
from scripts import mechanics_hbe_halfheight_boundary_experiment as runner


@pytest.fixture(autouse=True)
def no_execution_or_response_parsing(monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError('Native execution/response parsing forbidden in pure controls')
    monkeypatch.setattr(runner.subprocess, 'Popen', forbidden)
    monkeypatch.setattr(runner.old, 'solve', forbidden)
    monkeypatch.setattr(runner.runtime, 'supervise', forbidden)
    monkeypatch.setattr(reader.inherited, '_records', forbidden)


@pytest.fixture(scope='module')
def study():
    return core.declaration(ROOT, {'path': core.DECLARATION_PATH, 'sha256': core.DECLARATION_SHA256})


@pytest.fixture(scope='module')
def geometry(study):
    # In-memory transformation of the actual hash-pinned saved mesh. No study
    # preparation entry point or files are produced.
    return {name: core.load_variant(ROOT, study, name) for name in core.ORDERED_VARIANTS}


def test_frozen_scope_and_unchanged_fourteen_helpers(study):
    assert len(study['inherited_source_sha256']) == 14
    for key, record in study['baseline_P1']['source_bindings'].items():
        source = ROOT/'scripts'/Path(record['path']).name
        assert hashlib.sha256(source.read_bytes()).hexdigest() == study['inherited_source_sha256'][key]
    assert study['ordered_variants'] == ['P2', 'I2', 'P4']
    assert study['diagnostics']['signed_difference'] == 'second_minus_first;primary=P2_minus_I2'
    assert study['diagnostics']['comparison_pairs_first_second'][2] == ['I2', 'P2']
    assert study['gates']['mesh_motion_limit_m'] == 8e-6
    assert study['gates']['mesh_reaction_absolute_floor_N'] == 1.6e-7
    assert study['gates']['mesh_reaction_relative_to_max_abs_fine'] == .02
    assert study['measured_data_access'] is study['calibration'] is study['patient_data_access'] is False
    assert study['budgets'] == {'aggregate_specimen_seconds': 1200, 'each_active_run_output_bytes': 512*1024**2,
        'each_solver_seconds': 420, 'generated_output_bytes': 2*1024**3, 'gmsh_generation_calls': 0,
        'maximum_specimen_solver_calls': 3, 'pure_preparation_seconds': 60, 'runtime_threads': 1,
        'sampled_process_family_rss_bytes': 3*1024**3}


@pytest.mark.parametrize('variant', ['P2', 'I2', 'P4'])
def test_saved_geometry_subdivision_invariants(study, geometry, variant):
    source = core.access.verify_binding(ROOT, study['full_mesh'], read_json=True)
    result = geometry[variant]
    full, half, proof = result['full_mesh'], result['mesh'], result['subdivision_verification']
    X, C = np.array(source['rest_nodes_m']), np.array(source['elements_hex8'])
    Y, D = np.array(full['rest_nodes_m']), np.array(full['elements_hex8'])
    replaced = np.array(full['subdivision']['replaced_source_cell_ids'])-1
    untouched = np.ones(len(C), dtype=bool); untouched[replaced] = False
    assert np.array_equal(Y[:len(X)], X)
    assert np.array_equal(D[:len(C)][untouched], C[untouched])
    assert full['boundaries']['bottom'] == source['boundaries']['bottom']
    assert full['boundaries']['top'] == source['boundaries']['top']
    for row in full['subdivision']['generated_node_ancestry']:
        A, B = X[np.array(row['source_edge_node_ids'])-1]
        point = Y[row['node_id']-1]
        assert np.array_equal(A[:2], B[:2]) and np.array_equal(point[:2], A[:2])
        assert point[2] == A[2]+(B[2]-A[2])*row['fraction_numerator']/row['fraction_denominator']
        assert full['gmsh_node_ids'][row['node_id']-1] is None
    count = study['variants'][variant]
    assert (len(Y), len(D), len(half['rest_nodes_m']), len(half['elements_hex8'])) == (
        count['full_nodes'], count['full_elements'], count['half_nodes'], count['half_elements'])
    assert proof['maximum_parent_volume_error_ratio'] <= 1
    assert proof['shared_face_conformity'] is True
    # Independent volume sum, with original quadrature and explicit parent IDs.
    det, _ = core.BASE.rest_jacobians(Y, D)
    original, _ = core.BASE.rest_jacobians(X, C)
    parents = np.array(full['subdivision']['cell_parent_source_ids'])
    volume = np.bincount(parents, weights=det[:, :8].sum(axis=1), minlength=len(C)+1)[1:]
    np.testing.assert_allclose(volume, original[:, :8].sum(axis=1), rtol=1e-11, atol=1e-24)
    model = reader.inherited.HalfHeightReconstruction(full, half, result['mapping'])
    assert model.full.element_count == count['full_elements']
    assert model.maximum_coordinate_mismatch_m <= study['geometry']['coordinate_tolerance_m']


def test_nested_plate_nodes_and_matched_control_size(geometry):
    p2, p4, i2 = (geometry[v]['full_mesh'] for v in ('P2', 'P4', 'I2'))
    assert set(map(tuple, p2['rest_nodes_m'])) <= set(map(tuple, p4['rest_nodes_m']))
    assert len(p2['rest_nodes_m']) == len(i2['rest_nodes_m'])
    assert len(p2['elements_hex8']) == len(i2['elements_hex8'])
    assert p2['rest_nodes_m'] != i2['rest_nodes_m']


def test_fixed_physical_region_measures_are_conserved(study, geometry):
    half = core.access.verify_binding(ROOT, study['half_mesh'], read_json=True)
    reference = reader.regional_weights(half, study)
    for result in geometry.values():
        weights = reader.regional_weights(result['mesh'], study)
        assert weights['volume_bins'] == reference['volume_bins']
        assert weights['area_bins'] == reference['area_bins']
        for key in ('volume', 'area'):
            np.testing.assert_allclose(weights[key].sum(axis=0), reference[key].sum(axis=0), rtol=1e-11, atol=1e-24)
        np.testing.assert_allclose(weights['volume'].sum(axis=1), weights['cell_volume'], rtol=1e-13, atol=1e-24)


@pytest.mark.parametrize('variant', ['P2', 'I2', 'P4'])
def test_deck_keeps_original_physics_and_half_cut(study, geometry, variant):
    protocol = core.access.verify_binding(ROOT, study['original_protocol'], read_json=True)
    result = geometry[variant]
    baseline_xml, _ = core.BASE.specimen_deck(result['mesh'], 'compression', 60, 1000., protocol)
    xml, loading = core.halfheight_deck(result, protocol, study, variant)
    before, after = ET.fromstring(baseline_xml), ET.fromstring(xml)
    for name in ('Control', 'Material', 'Mesh', 'MeshDomains', 'LoadData', 'Output'):
        assert ET.tostring(before.find(name)) == ET.tostring(after.find(name))
    top = [bc for bc in after.find('Boundary') if bc.attrib['node_set'].startswith('top_node_')]
    assert len(top) == 1777 and all(bc.findtext('dof') == 'z' and bc.findtext('value') == '0.5' for bc in top)
    bottom = [bc for bc in after.find('Boundary') if bc.attrib['node_set'] == 'bottom']
    assert {bc.findtext('dof') for bc in bottom} == set('xyz')
    assert loading['mu_Pa'] == 1000. and loading['steps'] == 60
    transformed = core.backend.transform_deck(xml.encode())
    core.backend.verify_deck(xml.encode(), transformed.encode())


def test_wrong_specimen_or_variant_refused_before_transformation(study):
    with pytest.raises(ValueError, match='Only P2'):
        core.load_variant(ROOT, study, 'N32')
    wrong = deepcopy(study); wrong['variants']['P4']['full_nodes'] += 1
    with pytest.raises(ValueError, match='frozen boundary'):
        core.require_study(wrong)
    with pytest.raises(ValueError, match='Undeclared'):
        core.variant_of('compression:N24:S60:reference')


def test_scalar_primary_contrast_direction():
    # Dimensionless scalar subtraction control, not a force-response fixture.
    assert float(reader.signed_subtraction(7., 3.)) == 4.
    assert float(reader.signed_subtraction(3., 7.)) == -4.
    with pytest.raises(ValueError, match='Matched contrast'):
        reader.signed_subtraction([1.], [1., 2.])


def test_scalar_geometric_sequence_order_and_null_cases():
    # Pure geometric-series arithmetic; these are not specimen predictions.
    result = reader.local_order(1., .5, .25)
    assert result['order'] == 1. and result['local_limit_N'] == 0.
    assert result['local_remaining_indicator_N'] == -.25
    for values in ((1., .5, .75), (1., .75, .25), (1., 1., 1.), (0., 1e-8, 1.5e-8)):
        row = reader.local_order(*values)
        assert row['order'] is row['local_limit_N'] is row['local_remaining_indicator_N'] is None
        assert row['continuum_error_bound'] is False
    with pytest.raises(ValueError, match='Finite endpoint'):
        reader.local_order(float('nan'), 0., 0.)


def test_existing_parser_limits_are_unchanged():
    from scripts.mechanics_hbe_outputs import iter_data_records, iter_resolution_records
    with pytest.raises(ValueError, match='Invalid bounded output'):
        next(iter_data_records([], expected_times=reader.TIMES, item_count=20001, field_count=9, record_name='x'))
    with pytest.raises(ValueError, match='Invalid bounded output'):
        next(iter_resolution_records([], expected_times=reader.TIMES, item_count=25001, field_count=9,
             record_name='x', declaration_sha256=core.DECLARATION_SHA256))


def test_prior_execution_authentication_without_response_parsing(study):
    seen = []
    def bound(record, *, json_value=True, maximum_bytes=256*1024**2):
        seen.append(record['path'])
        assert not (json_value and record == study['baseline_P1']['readout'])
        return core.access.verify_binding(ROOT, record, maximum_bytes=maximum_bytes, read_json=json_value)
    runner.prior_metadata(ROOT, study, {'runtime_identity': study['runtime_identity']}, bound)
    assert study['baseline_P1']['execution']['path'] in seen
    assert study['baseline_P1']['readout']['path'] in seen
    execution = core.access.verify_binding(ROOT, study['baseline_P1']['execution'], read_json=True)['execution']
    runner.require_completed_native(execution, 420)
    invalid = deepcopy(execution); invalid['elapsed_seconds'] = 420
    with pytest.raises(ValueError, match='Complete bounded'):
        runner.require_completed_native(invalid, 420)
    invalid = deepcopy(execution); invalid['cleanup']['reaped'] = False
    with pytest.raises(ValueError, match='Complete bounded'):
        runner.require_completed_native(invalid, 420)
    compile(runner.P1_REPLAY_CODE, '<pinned-P1-replay-code>', 'exec')


def test_no_phase_release_is_inferred():
    with pytest.raises(ValueError, match='Separate preparation'):
        runner.preflight(ROOT, {}, {}, None)


@pytest.mark.parametrize('elapsed,passes', [(59.5, True), (60., False), (61., False)])
def test_inclusive_preparation_clock_is_reused(tmp_path, monkeypatch, elapsed, passes):
    monkeypatch.setattr(runner.time, 'monotonic', lambda: 100+elapsed)
    result = runner.finalize_result(tmp_path, tmp_path, tmp_path,
        {'status': 'prepared_not_solved'}, 100., 60., 10000)
    assert (result['status'] == 'prepared_not_solved') is passes
