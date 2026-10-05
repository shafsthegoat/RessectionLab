"""Independent analytical controls; no original patient records or solver calls."""
from dataclasses import replace
import hashlib
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from resectionlab import mechanics_landmarks as lm
from scripts import mechanics_patient_comparison as c


SOURCE = np.array([
    [-3, -2, -1], [4, -2, -1], [-3, 5, -1], [-3, -2, 6],
    [4, 5, -1], [4, -2, 6], [-3, 5, 6], [4, 5, 6],
    [0, 0, 0], [1, 2, 3], [-1, 3, 2], [2, -1, 4],
], dtype=float)


def records(destinations=None):
    target = SOURCE + [0.5, -1, 2] if destinations is None else destinations
    body = '\n'.join(' '.join(str(x) for x in (*a, *b))
                     for a, b in zip(SOURCE, target, strict=True))
    payload = ('MNI Tag Point File\nVolumes = 2;\nPoints =\n' + body + '\n;\n').encode()
    qc = lm.LandmarkFrameQC('a'*64, 'b'*64, np.eye(4), np.eye(4), 'c'*64,
                            'constructed coordinate control only')
    binding = lm.LandmarkPairBinding('analytic_review', lm.DISPLACEMENT_ROLE,
        hashlib.sha256(payload).hexdigest(), 'a'*64, 'b'*64, qc)
    partition = lm.partition_displacement_sources(lm.parse_tag_sources(payload, binding))
    forward = lm.build_forward_landmarks(payload, binding, partition)
    return payload, binding, partition, forward


def write_json(root, name, value):
    payload = (json.dumps(value, sort_keys=True, indent=2, allow_nan=False)+'\n').encode()
    (root/name).write_bytes(payload)
    return {'path': name, 'sha256': hashlib.sha256(payload).hexdigest()}


def freeze(root, *, destinations=None, external=False, external_extra=None):
    root.mkdir(exist_ok=True)
    payload, binding, partition, forward = records(destinations)
    protocol = write_json(root, 'protocol.json', {'scope': 'analytic review, no mechanics inference'})
    module = None
    field_manifest = None
    if external:
        sampler = root/'fixed_sampler.py'
        sampler.write_text('''from scripts.mechanics_patient_comparison import FieldSamples
def sample(root, field, queries):
    assert set(vars(queries)) == {"row_ids", "source_ras_mm"}
    return FieldSamples(queries.sha256, (None,)*len(queries.row_ids),
                        ("outside_reference_domain",)*len(queries.row_ids))
def alternate(root, field, queries):
    return FieldSamples(queries.sha256, ((0., 0., 0.),)*len(queries.row_ids),
                        ("supported",)*len(queries.row_ids))
''')
        spec = importlib.util.spec_from_file_location('review_fixed_sampler', sampler)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        artifact = write_json(root, 'field-fixture.json', {'analytic_stub': True})
        field_manifest = {
            'schema': 'frozen-tet10-displacement-field-v1', 'frame': 'RAS+', 'units': 'mm',
            'mesh_length_units': 'm', 'nodal_displacement_units': 'm',
            'tet10_indexing': 'zero_based',
            'tet10_order': 'vertices0123_edges01_12_20_03_13_23',
            'direction': 'reference_to_displaced', 'forward_hash': forward.forward_hash,
            'protocol_sha256': protocol['sha256'], 'interpolator_entrypoint': 'sample',
            'artifacts': {key: artifact for key in (
                'reference_mesh', 'geometry_frame', 'nodal_displacements', 'numerical_evidence')},
        }
        field_manifest['artifacts']['interpolator_source'] = {
            'path': sampler.name, 'sha256': hashlib.sha256(sampler.read_bytes()).hexdigest()}
        field_manifest.update(external_extra or {})
    field = c.fit_baselines(forward, protocol_sha256=protocol['sha256'])
    frozen = c.freeze_comparison(root, forward=forward, source_binding=binding,
        protocol_binding=protocol, field=field, output_directory='comparison',
        external_field=field_manifest)
    return payload, binding, partition, forward, frozen, module


def evaluate(root, prepared, *, sampler=None):
    payload, binding, partition, _, frozen, _ = prepared
    return c.evaluate_frozen_comparison(root, freeze_binding=frozen,
        payload=payload, source_binding=binding, partition=partition,
        external_sampler=sampler)


def forbid_reveal(monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError('Invalid frozen identity reached validation reveal')
    monkeypatch.setattr(lm, 'reveal_validation_landmarks', forbidden)


def test_same_source_file_alternative_entrypoint_is_rejected_before_validation(tmp_path, monkeypatch):
    prepared = freeze(tmp_path, external=True)
    forbid_reveal(monkeypatch)
    with pytest.raises(ValueError, match='(?i)entrypoint|interpolator|sampler'):
        evaluate(tmp_path, prepared, sampler=prepared[-1].alternate)
    assert not (tmp_path/'comparison/validation-attempt.json').exists()


def test_actual_read_length_is_bounded_when_stat_is_stale(tmp_path, monkeypatch):
    data = b'x'*128
    path = tmp_path/'growing-artifact.bin'
    path.write_bytes(data)
    original_stat = Path.stat
    def stale_stat(self, *args, **kwargs):
        if self == path:
            return SimpleNamespace(st_size=1)
        return original_stat(self, *args, **kwargs)
    monkeypatch.setattr(Path, 'stat', stale_stat)
    with pytest.raises(ValueError, match='(?i)bound|size|length'):
        c._read(tmp_path, {'path': path.name, 'sha256': hashlib.sha256(data).hexdigest()},
                maximum_bytes=16)


@pytest.mark.parametrize('extra', [
    {'during_image_path': 'unopened-later-image.nii.gz'},
    {'interpolation_exponent': 9},
])
def test_external_field_refuses_undeclared_later_input_or_interpolation_setting(tmp_path, extra):
    with pytest.raises(ValueError, match='(?i)field|schema|key|unexpected'):
        freeze(tmp_path, external=True, external_extra=extra)
    assert not (tmp_path/'comparison/freeze.json').exists()


def test_validation_destinations_change_only_audit_freeze_not_forward_or_prediction(tmp_path):
    initial = freeze(tmp_path/'first')
    changed = SOURCE + [0.5, -1, 2]
    changed[np.array(initial[2].validation_ids)-1] += [19, -23, 31]
    modified = freeze(tmp_path/'second', destinations=changed)
    first = json.loads((tmp_path/'first/comparison/freeze.json').read_bytes())
    second = json.loads((tmp_path/'second/comparison/freeze.json').read_bytes())
    assert initial[3].forward_hash == modified[3].forward_hash
    assert first['model'] == second['model']
    assert first['prediction_field'] == second['prediction_field']
    assert first['landmark_freeze']['source_audit_hash'] != second['landmark_freeze']['source_audit_hash']
    assert first['freeze_hash'] != second['freeze_hash']


def test_malformed_withheld_destination_is_unread_until_durable_attempt(tmp_path):
    _, _, partition, _ = records()
    targets = (SOURCE+[0.5, -1, 2]).astype(object)
    targets[partition.validation_ids[0]-1, 1] = 'SEALED_NOT_A_NUMBER'
    prepared = freeze(tmp_path, destinations=targets)
    assert (tmp_path/'comparison/freeze.json').exists()
    with pytest.raises(ValueError):
        evaluate(tmp_path, prepared)
    assert (tmp_path/'comparison/validation-attempt.json').exists()
    assert not (tmp_path/'comparison/evaluation.json').exists()
    with pytest.raises(FileExistsError):
        evaluate(tmp_path, prepared)


@pytest.mark.parametrize('tamper', ['source_identity', 'external_artifact'])
def test_pre_reveal_source_or_external_artifact_drift_refuses(tmp_path, monkeypatch, tamper):
    prepared = freeze(tmp_path, external=True)
    if tamper == 'source_identity':
        identity = c.running_sources()
        monkeypatch.setattr(c, 'running_sources', lambda: {**identity, 'comparison': 'f'*64})
    else:
        (tmp_path/'field-fixture.json').write_text('{"changed":true}')
    forbid_reveal(monkeypatch)
    with pytest.raises(ValueError):
        evaluate(tmp_path, prepared, sampler=prepared[-1].sample)
    assert not (tmp_path/'comparison/validation-attempt.json').exists()


def test_post_reveal_prediction_drift_preserves_attempt_without_publishing(tmp_path, monkeypatch):
    prepared = freeze(tmp_path)
    original = c.compare_landmarks
    def drift_after_scoring(*args, **kwargs):
        report = original(*args, **kwargs)
        (tmp_path/'comparison/prediction-field.json').write_bytes(b'{}')
        return report
    monkeypatch.setattr(c, 'compare_landmarks', drift_after_scoring)
    with pytest.raises(ValueError, match='bytes changed'):
        evaluate(tmp_path, prepared)
    assert (tmp_path/'comparison/validation-attempt.json').exists()
    assert not (tmp_path/'comparison/evaluation.json').exists()


def test_pinned_source_only_sampler_receives_queries_after_freeze_and_reports_nulls(tmp_path):
    prepared = freeze(tmp_path, external=True)
    _, report = evaluate(tmp_path, prepared, sampler=prepared[-1].sample)
    total = len(prepared[2].validation_ids)
    assert report['methods']['finite_element_field']['all_supported']['count'] == 0
    assert report['methods']['no_shift']['all_supported']['count'] == total
    assert report['methods']['no_shift']['common_supported']['count'] == 0
    assert report['paired_differences']['no_shift_minus_proper_rigid']['count'] == total
    assert report['paired_differences']['no_shift_minus_finite_element_field']['count'] == 0
    assert report['external_mechanics_validated_by_this_helper'] is False


def test_metrics_use_each_methods_full_support_and_pairwise_not_global_intersection():
    anchors = np.vstack([np.diag([2., 3., 5.]), -np.diag([2., 3., 5.])])
    rotation = np.array([[0., -1, 0], [1, 0, 0], [0, 0, 1]])
    translation = np.array([2., -3., 1.])
    conditioned = anchors@rotation.T+translation
    forward = lm.ForwardLandmarks('analytic', 'a'*64, 'b'*64, tuple(range(1, 7)),
        anchors, conditioned, np.eye(4), np.eye(4))
    field = c.fit_baselines(forward, protocol_sha256='c'*64)
    query = np.array([[0., 0, 0], [1, 1, 1], [8, 2, 3], [-4, -1, 2], [0, 8, 1], [2, -3, 9]])
    destination = query@rotation.T+translation+np.arange(6)[:, None]*np.array([.2, -.1, .3])
    held = lm.ValidationLandmarks(tuple(range(7, 13)), query, destination, 'd'*64, 'e'*64)
    supported = [0, 2, 5]
    external = c.FieldSamples(c.FieldQueries(held.row_ids, tuple(map(tuple, query))).sha256,
        tuple((0., 0., 0.) if i in supported else None for i in range(6)),
        tuple('supported' if i in supported else 'outside_reference_domain' for i in range(6)))
    result = c.compare_landmarks(field, held, external_samples=external)
    manual = {'no_shift': query, 'proper_rigid': query@rotation.T+translation}
    weights = 1/np.sum((query[:, None, :]-anchors[None, :, :])**2, axis=2)
    manual['inverse_distance_squared'] = query+(weights/weights.sum(axis=1)[:, None])@(conditioned-anchors)
    errors = {name: np.linalg.norm(predicted-destination, axis=1) for name, predicted in manual.items()}
    for name, error in errors.items():
        actual = result['methods'][name]
        assert actual['all_supported']['count'] == 6
        assert actual['all_supported']['rms_mm'] == pytest.approx(np.sqrt(np.mean(error**2)))
        assert actual['all_supported']['median_mm'] == pytest.approx(np.median(error))
        assert actual['all_supported']['maximum_mm'] == pytest.approx(max(error))
        assert actual['common_supported']['rms_mm'] == pytest.approx(np.sqrt(np.mean(error[supported]**2)))
    baseline_pair = result['paired_differences']['proper_rigid_minus_inverse_distance_squared']
    assert baseline_pair['count'] == 6
    assert baseline_pair['mean_error_difference_mm'] == pytest.approx(np.mean(errors['proper_rigid']-errors['inverse_distance_squared']))
    external_pair = result['paired_differences']['proper_rigid_minus_finite_element_field']
    assert [row['row_id'] for row in external_pair['per_landmark']] == [7, 9, 12]
    assert external_pair['mean_error_difference_mm'] == pytest.approx(np.mean((errors['proper_rigid']-errors['no_shift'])[supported]))


def test_baseline_canonical_bytes_cannot_replace_fit_at_freeze(tmp_path):
    _, binding, _, forward = records()
    protocol = write_json(tmp_path, 'protocol.json', {'scope': 'analytic'})
    value = c.fit_baselines(forward, protocol_sha256=protocol['sha256']).manifest()
    value['rigid']['translation_mm'][0] += 5
    with pytest.raises(ValueError, match='six-B'):
        c.freeze_comparison(tmp_path, forward=forward, source_binding=binding,
            protocol_binding=protocol, field=c.BaselineField(c._json(value)),
            output_directory='comparison')
    assert not (tmp_path/'comparison/freeze.json').exists()


def test_exact_query_ids_order_and_positions_are_bound():
    points = tuple(map(tuple, SOURCE[:6]))
    query = c.FieldQueries(tuple(range(1, 7)), points)
    assert replace(query, row_ids=tuple(range(6, 0, -1))).sha256 != query.sha256
    assert replace(query, source_ras_mm=tuple(reversed(points))).sha256 != query.sha256
    with pytest.raises(ValueError):
        c.FieldQueries((1,)*6, points)
    with pytest.raises(ValueError):
        c.FieldSamples(query.sha256, ((float('nan'), 0, 0),), ('supported',))


@pytest.mark.parametrize('reason', ['ambiguous_reference_location', 'nonconforming_reference_mesh'])
def test_reference_geometry_failures_remain_distinct_null_coverage(reason):
    query = c.FieldQueries((1,), ((0., 0., 0.),))
    samples = c.FieldSamples(query.sha256, (None,), (reason,))
    assert samples.status == (reason,)
    assert samples.displacement_ras_mm == (None,)
    with pytest.raises(ValueError, match='null'):
        c.FieldSamples(query.sha256, ((0., 0., 0.),), (reason,))


def test_artifact_symlink_escape_is_rejected(tmp_path):
    root = tmp_path/'root'
    root.mkdir()
    outside = tmp_path/'outside.json'
    outside.write_bytes(b'{}')
    (root/'alias.json').symlink_to(outside)
    with pytest.raises(ValueError, match='escapes'):
        c._read(root, {'path': 'alias.json', 'sha256': hashlib.sha256(b'{}').hexdigest()})
