"""Analytical comparison contracts only; no RESECT observations or FEM solves."""
from dataclasses import replace
import hashlib
import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest

from resectionlab import mechanics_landmarks as lm
from scripts import mechanics_patient_comparison as c


POINTS = np.array([
    [0, 0, 0], [2, 0, 0], [0, 3, 0], [0, 0, 4],
    [2, 3, 0], [2, 0, 4], [0, 3, 4], [2, 3, 4],
    [1, 0, 0], [0, 1, 0], [0, 0, 1], [1, 1, 1],
], float)


def prepared(observed=None):
    observed = POINTS+[.25, -.5, 1] if observed is None else observed
    rows = [' '.join(map(str, (*point, *target))) for point, target in zip(POINTS, observed, strict=True)]
    payload = ('MNI Tag Point File\nVolumes = 2;\nPoints =\n'+'\n'.join(rows)+'\n;\n').encode()
    qc = lm.LandmarkFrameQC('1'*64, '2'*64, np.eye(4), np.eye(4), '3'*64, 'analytical common frame')
    binding = lm.LandmarkPairBinding('constructed_only', lm.DISPLACEMENT_ROLE,
        hashlib.sha256(payload).hexdigest(), '1'*64, '2'*64, qc)
    partition = lm.partition_displacement_sources(lm.parse_tag_sources(payload, binding))
    return payload, binding, partition, lm.build_forward_landmarks(payload, binding, partition)


def forward_points(source, observed):
    return lm.ForwardLandmarks('constructed_only', '1'*64, '2'*64, tuple(range(1, 7)),
                               source, observed, np.eye(4), np.eye(4))


def validation(payload, binding, partition, forward, field):
    frozen = lm.freeze_landmark_model(forward, source_binding=binding,
                                      model_sha256='4'*64, prediction_sha256=field.sha256)
    return lm.reveal_validation_landmarks(payload, binding, partition, frozen)


def save(root, name, value):
    data = c._json(value)
    (root/name).write_bytes(data)
    return {'path': name, 'sha256': c._sha(data)}


def test_proper_rigid_recovers_known_rotation_translation_without_reflection():
    rotation = np.array([[0, -1, 0], [1, 0, 0], [0, 0, 1.]])
    observed = POINTS @ rotation.T+[4, -7, 2]
    _, _, _, forward = prepared(observed)
    field = c.fit_baselines(forward, protocol_sha256='9'*64)
    rigid = field.manifest()['rigid']
    np.testing.assert_allclose(rigid['rotation'], rotation, atol=2e-15)
    np.testing.assert_allclose(rigid['translation_mm'], [4, -7, 2], atol=2e-15)
    sampled = c.sample_baselines(field, POINTS)
    np.testing.assert_allclose(sampled['proper_rigid']['predicted_ras_mm'], observed, atol=5e-15)
    assert np.linalg.det(rigid['rotation']) == pytest.approx(1)


def test_reflected_observations_never_produce_a_reflection_fit():
    source = np.vstack([np.diag([1., 2, 4]), -np.diag([1., 2, 4])])
    observed = source*[-1, 1, 1]
    field = c.fit_baselines(forward_points(source, observed), protocol_sha256='9'*64)
    rigid = field.manifest()['rigid']
    assert rigid['status'] == 'available'
    assert np.linalg.det(rigid['rotation']) == pytest.approx(1)
    predicted = c.sample_baselines(field, source)['proper_rigid']['predicted_ras_mm']
    assert np.linalg.norm(np.array(predicted)-observed) > 1


def test_exact_idw_coincidence_equal_distance_and_stable_near_coincidence():
    source = np.vstack([np.eye(3), -np.eye(3)])
    displacement = np.array([[1, 2, 3], [2, 3, 4], [3, 4, 5], [4, 5, 6], [5, 6, 7], [6, 7, 8.]])
    field = c.fit_baselines(forward_points(source, source+displacement), protocol_sha256='9'*64)
    sampled = c.sample_baselines(field, np.vstack([source, np.zeros(3)]))
    np.testing.assert_array_equal(sampled['inverse_distance_squared']['predicted_ras_mm'][:6], source+displacement)
    np.testing.assert_allclose(sampled['inverse_distance_squared']['predicted_ras_mm'][6], displacement.mean(axis=0))
    source[0] = 0
    field = c.fit_baselines(forward_points(source, source+displacement), protocol_sha256='9'*64)
    tiny = c.sample_baselines(field, [[1e-200, 0, 0]])['inverse_distance_squared']['predicted_ras_mm']
    np.testing.assert_allclose(tiny, [displacement[0]], atol=0)


def test_duplicate_or_planar_B_rejected_and_collapsed_observations_retained_as_rigid_failure():
    _, _, _, forward = prepared()
    source = forward.source_ras_mm.copy()
    source[1] = source[0]
    with pytest.raises(ValueError, match='Duplicate'):
        c.fit_baselines(replace(forward, source_ras_mm=source), protocol_sha256='9'*64)
    source = np.column_stack([np.arange(6), np.arange(6)**2, np.zeros(6)])
    with pytest.raises(ValueError, match='rank-three'):
        c.fit_baselines(replace(forward, source_ras_mm=source), protocol_sha256='9'*64)
    collapsed = replace(forward, observed_ras_mm=np.zeros((6, 3)))
    field = c.fit_baselines(collapsed, protocol_sha256='9'*64)
    result = c.sample_baselines(field, [[1, 1, 1]])
    assert result['proper_rigid']['predicted_ras_mm'] == [None]
    assert result['proper_rigid']['status'] == ['nonunique_rigid_fit']
    assert result['no_shift']['predicted_ras_mm'] == [[1, 1, 1]]


def test_metrics_match_closed_form_and_preserve_all_and_common_coverage():
    payload, binding, partition, forward = prepared()
    field = c.fit_baselines(forward, protocol_sha256='9'*64)
    held = validation(payload, binding, partition, forward, field)
    queries = c.FieldQueries(held.row_ids, tuple(map(tuple, held.source_ras_mm)))
    samples = c.FieldSamples(queries.sha256,
        tuple([(.25, -.5, 1)]*(len(held.row_ids)-2)+[None, None]),
        tuple(['supported']*(len(held.row_ids)-2)+['outside_reference_domain', 'unsupported_reference_geometry']))
    result = c.compare_landmarks(field, held, external_samples=samples)
    methods = result['methods']
    expected = np.sqrt(.25**2+.5**2+1)
    assert methods['no_shift']['all_supported']['rms_mm'] == pytest.approx(expected)
    assert methods['no_shift']['all_supported']['median_mm'] == pytest.approx(expected)
    assert methods['no_shift']['all_supported']['maximum_mm'] == pytest.approx(expected)
    assert methods['no_shift']['all_supported']['count'] == 6
    assert methods['no_shift']['common_supported']['count'] == 4
    assert methods['finite_element_field']['excluded_counts'] == {'outside_reference_domain': 1, 'unsupported_reference_geometry': 1}
    assert methods['finite_element_field']['per_landmark'][-1]['error_mm'] is None
    differences = result['paired_differences']['no_shift_minus_inverse_distance_squared']
    assert differences['count'] == 6
    assert differences['mean_error_difference_mm'] == pytest.approx(expected)
    assert result['confidence_interval'] is None and result['clinical_injury_probability'] is None
    assert result['physical_action_response_validated'] is False
    assert result['B_residuals']['inverse_distance_squared']['summary']['maximum_mm'] == 0


def test_null_support_is_never_a_zero_error_or_silent_denominator_change():
    payload, binding, partition, forward = prepared()
    field = c.fit_baselines(forward, protocol_sha256='9'*64)
    held = validation(payload, binding, partition, forward, field)
    query = c.FieldQueries(held.row_ids, tuple(map(tuple, held.source_ras_mm)))
    none = c.FieldSamples(query.sha256, (None,)*6, ('outside_reference_domain',)*6)
    report = c.compare_landmarks(field, held, external_samples=none)
    assert report['methods']['finite_element_field']['all_supported'] == {'count': 0, 'rms_mm': None, 'median_mm': None, 'maximum_mm': None}
    assert report['methods']['no_shift']['all_supported']['count'] == 6
    assert report['common_supported_ids'] == []
    with pytest.raises(ValueError, match='null'):
        c.FieldSamples(query.sha256, ((0, 0, 0),), ('outside_reference_domain',))
    with pytest.raises(ValueError, match='bound'):
        c.compare_landmarks(field, held, external_samples=replace(none, query_sha256='8'*64))


def test_B_only_fit_does_not_accept_validation_or_change_with_hidden_outcomes():
    payload, binding, partition, forward = prepared()
    field = c.fit_baselines(forward, protocol_sha256='9'*64)
    observed = POINTS+[.25, -.5, 1]
    observed[np.array(partition.validation_ids)-1] += 100
    other_forward = prepared(observed)[3]
    assert c.fit_baselines(other_forward, protocol_sha256='9'*64).payload == field.payload
    held = validation(payload, binding, partition, forward, field)
    with pytest.raises(TypeError, match='six observed B'):
        c.fit_baselines(held, protocol_sha256='9'*64)
    rules = field.manifest()['rules']
    assert rules['preoperative_scan_only_model'] is False
    assert rules['material_properties_are_deployment_observations'] is False


def frozen_fixture(root, *, external=None):
    payload, binding, partition, forward = prepared()
    protocol = save(root, 'protocol.json', {'test_only': 'analytical contract'})
    field = c.fit_baselines(forward, protocol_sha256=protocol['sha256'])
    freeze = c.freeze_comparison(root, forward=forward, source_binding=binding,
        protocol_binding=protocol, field=field, output_directory='study', external_field=external)
    return payload, binding, partition, forward, field, freeze


def test_complete_field_is_durable_before_any_V_numeric_access_and_no_second_attempt(tmp_path, monkeypatch):
    payload, binding, partition, forward, field, freeze = frozen_fixture(tmp_path)
    original = lm._coordinate_tokens
    count = 0
    def instrumented(tokens):
        nonlocal count
        assert (tmp_path/'study/freeze.json').exists()
        assert (tmp_path/'study/validation-attempt.json').exists()
        count += 1
        return original(tokens)
    monkeypatch.setattr(lm, '_coordinate_tokens', instrumented)
    saved, report = c.evaluate_frozen_comparison(tmp_path, freeze_binding=freeze,
        payload=payload, source_binding=binding, partition=partition)
    assert count > 0 and (tmp_path/saved['path']).exists()
    assert report['methods']['inverse_distance_squared']['all_supported']['rms_mm'] < 1e-14
    with pytest.raises(FileExistsError):
        c.evaluate_frozen_comparison(tmp_path, freeze_binding=freeze,
            payload=payload, source_binding=binding, partition=partition)


@pytest.mark.parametrize('changed', ['protocol.json', 'study/model.json', 'study/prediction-field.json'])
def test_frozen_input_drift_stops_before_reveal(tmp_path, monkeypatch, changed):
    payload, binding, partition, _, _, freeze = frozen_fixture(tmp_path)
    (tmp_path/changed).write_text('{}')
    def forbidden(*args, **kwargs):
        raise AssertionError('V must remain sealed on identity failure')
    monkeypatch.setattr(lm, 'reveal_validation_landmarks', forbidden)
    with pytest.raises(ValueError, match='bytes changed'):
        c.evaluate_frozen_comparison(tmp_path, freeze_binding=freeze,
            payload=payload, source_binding=binding, partition=partition)


def test_failed_reveal_is_durably_recorded_and_cannot_restart(tmp_path):
    payload, binding, partition, _, _, freeze = frozen_fixture(tmp_path)
    with pytest.raises(ValueError, match='declared source'):
        c.evaluate_frozen_comparison(tmp_path, freeze_binding=freeze,
            payload=payload+b' ', source_binding=binding, partition=partition)
    assert (tmp_path/'study/validation-attempt.json').exists()
    with pytest.raises(FileExistsError):
        c.evaluate_frozen_comparison(tmp_path, freeze_binding=freeze,
            payload=payload, source_binding=binding, partition=partition)


def test_external_sampler_receives_sources_only_and_null_failure_cannot_disappear(tmp_path):
    payload, binding, partition, forward = prepared()
    protocol = save(tmp_path, 'protocol.json', {'test_only': 'not numerical proof'})
    stub = tmp_path/'analytical_sampler.py'
    stub.write_text('''from scripts.mechanics_patient_comparison import FieldSamples
def sample(root, field, queries):
    assert not hasattr(queries, "observed_ras_mm")
    return FieldSamples(queries.sha256, (None,)*len(queries.row_ids),
                        ("outside_reference_domain",)*len(queries.row_ids))
''')
    spec = importlib.util.spec_from_file_location('analytical_sampler', stub)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    artifact = save(tmp_path, 'analytical-artifact.json', {'fixture': 'not an actual FEM field'})
    external = {'schema': 'frozen-tet10-displacement-field-v1', 'frame': 'RAS+', 'units': 'mm',
        'interpolator_entrypoint': 'sample',
        'mesh_length_units': 'm', 'nodal_displacement_units': 'm',
        'tet10_indexing': 'zero_based', 'tet10_order': 'vertices0123_edges01_12_20_03_13_23',
        'direction': 'reference_to_displaced', 'forward_hash': forward.forward_hash,
        'protocol_sha256': protocol['sha256'], 'artifacts': {
            'reference_mesh': artifact, 'geometry_frame': artifact,
            'nodal_displacements': artifact, 'numerical_evidence': artifact,
            'interpolator_source': {'path': stub.name, 'sha256': c._sha(stub.read_bytes())}}}
    field = c.fit_baselines(forward, protocol_sha256=protocol['sha256'])
    freeze = c.freeze_comparison(tmp_path, forward=forward, source_binding=binding,
        protocol_binding=protocol, field=field, output_directory='study', external_field=external)
    _, report = c.evaluate_frozen_comparison(tmp_path, freeze_binding=freeze,
        payload=payload, source_binding=binding, partition=partition, external_sampler=module.sample)
    assert report['methods']['finite_element_field']['all_supported']['count'] == 0
    assert report['external_mechanics_validated_by_this_helper'] is False


def test_external_mesh_failure_statuses_remain_distinct_null_results():
    samples = c.FieldSamples('9'*64, (None, None, None),
        ('ambiguous_reference_location', 'nonconforming_reference_mesh', 'uncertain_reference_boundary'))
    assert samples.status == ('ambiguous_reference_location', 'nonconforming_reference_mesh', 'uncertain_reference_boundary')
    assert samples.displacement_ras_mm == (None, None, None)
    with pytest.raises(ValueError, match='null'):
        c.FieldSamples('9'*64, ((0, 0, 0),), ('ambiguous_reference_location',))
