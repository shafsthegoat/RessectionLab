"""Frozen conditional-displacement baselines and independent landmark metrics.

The fit receives only six permitted intraoperative B observations. Complete
evaluable fields are persisted before the existing landmark helper reveals V
sources and destinations together. No registration, solver or material fit is
implemented here. A caller must separately obtain the declared data release.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import inspect
import itertools
import json
import math
import os
from pathlib import Path
import re

import numpy as np

from resectionlab import mechanics_landmarks as lm


METHODS = ('no_shift', 'proper_rigid', 'inverse_distance_squared')
FIELD_STATUSES = (
    'supported', 'outside_reference_domain', 'unsupported_reference_geometry',
    'ambiguous_reference_location', 'nonconforming_reference_mesh',
    'uncertain_reference_boundary',
)
MAX_POINTS = 4096
RIGID_RANK_RATIO = 1e-12


def _json(value):
    return (json.dumps(value, sort_keys=True, indent=2, allow_nan=False)+'\n').encode()


def _sha(data):
    return hashlib.sha256(data).hexdigest()


def _hash(value):
    if not isinstance(value, str) or re.fullmatch(r'(?:sha256:)?[0-9a-f]{64}', value) is None:
        raise ValueError('An exact SHA256 identity is required')
    return value.removeprefix('sha256:')


def _points(value, *, count=None):
    raw = np.asarray(value)
    if (raw.dtype.kind not in 'iuf' or raw.ndim != 2 or raw.shape[1:] != (3,)
            or not 1 <= len(raw) <= MAX_POINTS or (count is not None and len(raw) != count)
            or not np.isfinite(raw).all() or np.any(np.abs(raw) > 1e6)):
        raise ValueError('Finite bounded RAS millimetre points are required')
    return np.array(raw, dtype=np.float64, copy=True)


def _ids(value, count):
    ids = tuple(value)
    if (len(ids) != count or len(set(ids)) != count
            or any(type(i) is not int or not 1 <= i <= MAX_POINTS for i in ids)):
        raise ValueError('Unique one-based landmark row IDs are required')
    return ids


def running_sources():
    """Actual helper identities; do not substitute caller-supplied source labels."""
    return {
        'comparison': _sha(Path(__file__).read_bytes()),
        'landmarks': _sha(Path(lm.__file__).read_bytes()),
    }


def _proper_rigid(source, observed):
    source_center = source.mean(axis=0)
    target_center = observed.mean(axis=0)
    covariance = (source-source_center).T @ (observed-target_center)
    left, singular, right_t = np.linalg.svd(covariance)
    orientation = float(np.linalg.det(right_t.T @ left.T))
    # Two independent directions suffice for a unique proper rotation. A
    # reflected fit with tied two smallest singular values is also nonunique.
    ambiguous = (singular[0] == 0 or singular[1] <= RIGID_RANK_RATIO*singular[0]
                 or (orientation < 0 and singular[1]-singular[2]
                     <= RIGID_RANK_RATIO*singular[0]))
    if ambiguous:
        return {'status': 'nonunique_rigid_fit', 'rotation': None, 'translation_mm': None,
                'singular_values_mm2': singular.tolist()}
    correction = np.diag([1., 1., 1. if orientation >= 0 else -1.])
    rotation = right_t.T @ correction @ left.T
    translation = target_center-rotation @ source_center
    if not np.allclose(rotation.T @ rotation, np.eye(3), rtol=0, atol=1e-12) or not math.isclose(float(np.linalg.det(rotation)), 1., abs_tol=1e-12):
        raise ValueError('Proper rotation calculation failed its numerical invariant')
    return {'status': 'available', 'rotation': rotation.tolist(),
            'translation_mm': translation.tolist(), 'singular_values_mm2': singular.tolist()}


@dataclass(frozen=True)
class BaselineField:
    """Immutable complete baseline coefficients; contains no V coordinates."""
    payload: bytes

    def __post_init__(self):
        if type(self.payload) is not bytes or len(self.payload) > 1024**2:
            raise ValueError('Bounded immutable baseline field bytes are required')
        value = json.loads(self.payload)
        if value.get('schema') != 'conditional-baseline-field-v1' or _json(value) != self.payload:
            raise ValueError('Canonical baseline field is required')

    @property
    def sha256(self):
        return _sha(self.payload)

    def manifest(self):
        return json.loads(self.payload)


def fit_baselines(forward: lm.ForwardLandmarks, *, protocol_sha256) -> BaselineField:
    if type(forward) is not lm.ForwardLandmarks:
        raise TypeError('Only the six observed B landmarks may enter baseline fitting')
    forward.assert_intact()
    source = _points(forward.source_ras_mm, count=6)
    observed = _points(forward.observed_ras_mm, count=6)
    if len(np.unique(source, axis=0)) != 6:
        raise ValueError('Duplicate B sources cannot define a unique exact interpolant')
    singular = np.linalg.svd(source-source.mean(axis=0), compute_uv=False)
    if singular[0] == 0 or singular[-1]/singular[0] <= 1e-6:
        raise ValueError('B sources fail the frozen rank-three eligibility rule')
    value = {
        'schema': 'conditional-baseline-field-v1',
        'protocol_sha256': _hash(protocol_sha256),
        'forward_hash': forward.forward_hash,
        'frame': 'RAS+', 'units': 'mm',
        'observations': forward.to_manifest(),
        'rigid': _proper_rigid(source, observed),
        'rules': {
            'methods': list(METHODS), 'idw_power': 2,
            'idw_neighbors': 'all_six', 'idw_coincidence': 'exact_coordinate_equality',
            'idw_equal_distance': 'equal_weights', 'idw_duplicate_B': 'reject',
            'rigid_rank_ratio': RIGID_RANK_RATIO,
            'observation_status': 'six_observed_intraoperative_internal_motions',
            'preoperative_scan_only_model': False,
            'material_properties_are_deployment_observations': False,
        },
    }
    return BaselineField(_json(value))


def sample_baselines(field: BaselineField, query_ras_mm):
    """Evaluate a previously fixed field. Queries are evaluator-only, never fit inputs."""
    if type(field) is not BaselineField:
        raise TypeError('A complete fixed baseline field is required')
    value = field.manifest()
    query = _points(query_ras_mm)
    source = _points(value['observations']['source_ras_mm'], count=6)
    observed = _points(value['observations']['observed_ras_mm'], count=6)
    displacement = observed-source
    idw = []
    for point in query:
        coincident = np.flatnonzero(np.all(source == point, axis=1))
        if len(coincident):
            if len(coincident) != 1:
                raise ValueError('Duplicate B coordinates cannot be resolved by a tie break')
            motion = displacement[coincident[0]]
        else:
            # math.dist handles very small distances without squaring underflow.
            # Relative weights stay <=1, avoiding the singular 1/r**2 overflow.
            distance = np.array([math.dist(point, anchor) for anchor in source])
            weights = (distance.min()/distance)**2
            motion = (weights/weights.sum()) @ displacement
        idw.append((point+motion).tolist())
    rigid = value['rigid']
    rigid_points = None
    if rigid['status'] == 'available':
        rigid_points = (query @ np.asarray(rigid['rotation']).T
                        + np.asarray(rigid['translation_mm'])).tolist()
    return {
        'no_shift': {'predicted_ras_mm': query.tolist(), 'status': ['supported']*len(query)},
        'proper_rigid': {
            'predicted_ras_mm': rigid_points if rigid_points is not None else [None]*len(query),
            'status': ['supported' if rigid_points is not None else 'nonunique_rigid_fit']*len(query),
        },
        'inverse_distance_squared': {'predicted_ras_mm': idw, 'status': ['supported']*len(query)},
    }


@dataclass(frozen=True)
class FieldQueries:
    """Only V source positions are handed to the frozen external-field sampler."""
    row_ids: tuple[int, ...]
    source_ras_mm: tuple[tuple[float, float, float], ...]

    def __post_init__(self):
        points = _points(self.source_ras_mm)
        object.__setattr__(self, 'row_ids', _ids(self.row_ids, len(points)))
        object.__setattr__(self, 'source_ras_mm', tuple(map(tuple, points.tolist())))

    @property
    def sha256(self):
        return _sha(_json({'row_ids': self.row_ids, 'source_ras_mm': self.source_ras_mm,
                          'frame': 'RAS+', 'units': 'mm'}))


@dataclass(frozen=True)
class FieldSamples:
    """External interpolation result, not a claim of verified mechanics."""
    query_sha256: str
    displacement_ras_mm: tuple[tuple[float, float, float] | None, ...]
    status: tuple[str, ...]

    def __post_init__(self):
        object.__setattr__(self, 'query_sha256', _hash(self.query_sha256))
        statuses = tuple(self.status)
        values = tuple(self.displacement_ras_mm)
        if not 1 <= len(values) <= MAX_POINTS or len(values) != len(statuses):
            raise ValueError('Complete external field coverage is required')
        frozen = []
        for status, value in zip(statuses, values, strict=True):
            if status not in FIELD_STATUSES or (status == 'supported') != (value is not None):
                raise ValueError('Unsupported field samples must remain null with an explicit reason')
            frozen.append(None if value is None else tuple(_points([value], count=1)[0]))
        object.__setattr__(self, 'displacement_ras_mm', tuple(frozen))
        object.__setattr__(self, 'status', statuses)


def _summary(errors):
    if not errors:
        return {'count': 0, 'rms_mm': None, 'median_mm': None, 'maximum_mm': None}
    values = np.asarray(errors, float)
    return {'count': len(values), 'rms_mm': float(np.sqrt(np.mean(values**2))),
            'median_mm': float(np.median(values)), 'maximum_mm': float(values.max())}


def compare_landmarks(field, validation: lm.ValidationLandmarks, *, external_samples=None):
    """Independent point errors on all available and identical common supports.

    This pure calculation does not authorize or establish disclosure chronology;
    evaluate_frozen_comparison supplies that boundary for the file workflow.
    """
    if type(validation) is not lm.ValidationLandmarks:
        raise TypeError('Evaluator-only validation landmarks are required')
    source = _points(validation.source_ras_mm)
    observed = _points(validation.observed_ras_mm, count=len(source))
    ids = _ids(validation.row_ids, len(source))
    if len(np.unique(source, axis=0)) != len(source):
        raise ValueError('Duplicate V coordinates cannot count as additional independent coverage')
    predictions = sample_baselines(field, source)
    queries = FieldQueries(ids, tuple(map(tuple, source)))
    if external_samples is not None:
        if type(external_samples) is not FieldSamples or external_samples.query_sha256 != queries.sha256 or len(external_samples.status) != len(source):
            raise ValueError('External prediction is not bound to these evaluator queries')
        predictions['finite_element_field'] = {
            'predicted_ras_mm': [None if motion is None else (point+motion).tolist()
                                 for point, motion in zip(source, external_samples.displacement_ras_mm, strict=True)],
            'status': list(external_samples.status),
        }
    by_method = {}
    errors = {}
    for method, prediction in predictions.items():
        method_errors = []
        rows = []
        excluded = {}
        for row_id, expected, predicted, status in zip(ids, observed, prediction['predicted_ras_mm'], prediction['status'], strict=True):
            error = None if predicted is None else math.dist(expected, predicted)
            if error is None:
                excluded[status] = excluded.get(status, 0)+1
            else:
                method_errors.append(error)
            rows.append({'row_id': row_id, 'status': status, 'error_mm': error,
                         'predicted_ras_mm': predicted})
        errors[method] = [row['error_mm'] for row in rows]
        by_method[method] = {'all_supported': _summary(method_errors), 'total_landmarks': len(ids),
                             'excluded_counts': excluded, 'per_landmark': rows}
    common = [i for i in range(len(ids)) if all(values[i] is not None for values in errors.values())]
    for method in by_method:
        by_method[method]['common_supported'] = _summary([errors[method][i] for i in common])
    paired = {}
    for first, second in itertools.combinations(predictions, 2):
        eligible = [i for i in range(len(ids)) if errors[first][i] is not None and errors[second][i] is not None]
        differences = [errors[first][i]-errors[second][i] for i in eligible]
        paired[f'{first}_minus_{second}'] = {
            'count': len(eligible), 'definition': 'Euclidean error first minus second; negative favors first',
            'mean_error_difference_mm': float(np.mean(differences)) if differences else None,
            'per_landmark': [{'row_id': ids[i], 'error_difference_mm': difference}
                             for i, difference in zip(eligible, differences, strict=True)],
        }
    observations = field.manifest()['observations']
    boundary_predictions = sample_baselines(field, observations['source_ras_mm'])
    boundary_residuals = {}
    for method, prediction in boundary_predictions.items():
        residuals = [None if point is None else math.dist(point, observed_point)
                     for point, observed_point in zip(prediction['predicted_ras_mm'],
                                                     observations['observed_ras_mm'], strict=True)]
        boundary_residuals[method] = {
            'meaning': 'residual on the six conditioning observations, not validation',
            'summary': _summary([error for error in residuals if error is not None]),
            'per_landmark': [{'row_id': row_id, 'error_mm': error}
                             for row_id, error in zip(observations['boundary_ids'], residuals, strict=True)],
        }
    return {'schema': 'conditional-displacement-comparison-v1', 'frame': 'RAS+', 'units': 'mm',
            'methods': by_method, 'common_supported_ids': [ids[i] for i in common],
            'paired_differences': paired, 'validation_landmarks': len(ids),
            'B_residuals': boundary_residuals,
            'confidence_interval': None, 'clinical_injury_probability': None,
            'physical_action_response_validated': False,
            'baseline_support_meaning': 'global mathematical evaluation, not verified retained tissue',
            'external_mechanics_validated_by_this_helper': False}


def _path(root, relative):
    if not isinstance(relative, str) or Path(relative).is_absolute():
        raise ValueError('Repository-relative artifact path required')
    path = (Path(root).resolve()/relative).resolve()
    if not path.is_relative_to(Path(root).resolve()):
        raise ValueError('Artifact escapes the declared root')
    return path


def _read(root, binding, *, json_data=False, maximum_bytes=64*1024**2):
    if set(binding) != {'path', 'sha256'}:
        raise ValueError('Exact artifact path/hash binding required')
    path = _path(root, binding['path'])
    # Bound the read itself; a prior stat may be stale if a writer is active.
    with path.open('rb') as stream:
        payload = stream.read(maximum_bytes+1)
    if len(payload) > maximum_bytes:
        raise ValueError('Comparison artifact exceeds bound')
    if _sha(payload) != _hash(binding['sha256']):
        raise ValueError('Frozen artifact bytes changed')
    return json.loads(payload) if json_data else payload


def _save(root, relative, payload):
    path = _path(root, relative)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('xb') as stream:
        stream.write(payload)
        stream.flush()
        os.fsync(stream.fileno())
    descriptor = os.open(path.parent, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)
    return {'path': relative, 'sha256': _sha(payload)}


def _check_external(root, external, *, forward_hash, protocol_sha256):
    if external is None:
        return
    allowed = {
        'schema', 'frame', 'units', 'mesh_length_units', 'nodal_displacement_units',
        'tet10_indexing', 'tet10_order', 'direction', 'forward_hash', 'protocol_sha256',
        'artifacts', 'interpolator_entrypoint',
    }
    if set(external) != allowed:
        raise ValueError('External field contains missing or undeclared inputs/settings')
    if (external.get('schema') != 'frozen-tet10-displacement-field-v1'
            or external.get('frame') != 'RAS+' or external.get('units') != 'mm'
            or external.get('mesh_length_units') != 'm'
            or external.get('nodal_displacement_units') != 'm'
            or external.get('tet10_indexing') != 'zero_based'
            or external.get('tet10_order') != 'vertices0123_edges01_12_20_03_13_23'
            or external.get('direction') != 'reference_to_displaced'
            or external.get('forward_hash') != forward_hash
            or _hash(external.get('protocol_sha256')) != _hash(protocol_sha256)):
        raise ValueError('External field identity/frame/direction differs')
    entrypoint = external.get('interpolator_entrypoint')
    if not isinstance(entrypoint, str) or re.fullmatch(r'[A-Za-z_][A-Za-z0-9_]*', entrypoint) is None:
        raise ValueError('An exact top-level interpolator entrypoint is required')
    artifacts = external.get('artifacts', {})
    if set(artifacts) != {'reference_mesh', 'geometry_frame', 'nodal_displacements', 'numerical_evidence', 'interpolator_source'}:
        raise ValueError('Complete external field artifacts are required')
    for binding in artifacts.values():
        _read(root, binding)


def freeze_comparison(root, *, forward, source_binding, protocol_binding,
                      field, output_directory, external_field=None):
    """Persist complete fields before any V source/destination access.

    A separate root release still authorizes data use and fixes this directory.
    The external field is an upstream result, not a solver-verification shortcut.
    """
    protocol = _read(root, protocol_binding)
    expected = fit_baselines(forward, protocol_sha256=_sha(protocol))
    if type(field) is not BaselineField or field.payload != expected.payload:
        raise ValueError('Baseline field differs from the fixed six-B calculation')
    external = None if external_field is None else json.loads(_json(external_field))
    _check_external(root, external, forward_hash=forward.forward_hash,
                    protocol_sha256=protocol_binding['sha256'])
    sources = running_sources()
    model = {'schema': 'conditional-comparison-model-v1', 'protocol': protocol_binding,
             'running_sources': sources, 'forward_hash': forward.forward_hash,
             'baseline_field_sha256': field.sha256,
             'scope': 'conditional_displacement_from_six_intraoperative_observations'}
    prefix = str(Path(output_directory))
    model_binding = _save(root, prefix+'/model.json', _json(model))
    prediction_binding = _save(root, prefix+'/prediction-field.json', _json({
        'baseline': field.manifest(), 'external_field': external}))
    frozen = lm.freeze_landmark_model(
        forward, source_binding=source_binding,
        model_sha256=model_binding['sha256'], prediction_sha256=prediction_binding['sha256'],
    )
    record = {'schema': 'conditional-comparison-freeze-v1', 'model': model_binding,
              'prediction_field': prediction_binding, 'protocol': protocol_binding,
              'running_sources': sources, 'directory': output_directory,
              'landmark_freeze': {name: getattr(frozen, name) for name in
                                  ('forward_hash', 'partition_hash', 'model_sha256',
                                   'prediction_sha256', 'source_audit_hash')},
              'freeze_hash': frozen.freeze_hash}
    # Recheck all inputs before committing the final freeze marker.
    _read(root, protocol_binding)
    _check_external(root, external, forward_hash=forward.forward_hash,
                    protocol_sha256=protocol_binding['sha256'])
    if running_sources() != sources:
        raise ValueError('Executing comparison sources changed during freeze')
    return _save(root, prefix+'/freeze.json', _json(record))


def evaluate_frozen_comparison(root, *, freeze_binding, payload, source_binding,
                               partition, external_sampler=None):
    """Reveal V once, then sample only fixed fields with no fitting hook.

    Any failed reveal leaves a durable attempt marker. This is reproducibility
    on a cooperative host; authorization and independent FEM verification are
    obligations of the outer experiment runner, not inferred from these hashes.
    """
    record = _read(root, freeze_binding, json_data=True)
    if record.get('schema') != 'conditional-comparison-freeze-v1' or record.get('running_sources') != running_sources():
        raise ValueError('Comparison source/freeze identity differs')
    model = _read(root, record['model'], json_data=True)
    predictions = _read(root, record['prediction_field'], json_data=True)
    _read(root, record['protocol'])
    frozen = lm.LandmarkModelFreeze(**record['landmark_freeze'])
    if (frozen.freeze_hash != record['freeze_hash']
            or _hash(frozen.model_sha256) != record['model']['sha256']
            or _hash(frozen.prediction_sha256) != record['prediction_field']['sha256']
            or model['protocol'] != record['protocol']
            or model['running_sources'] != record['running_sources']):
        raise ValueError('Saved model/field binding differs from landmark freeze')
    field = BaselineField(_json(predictions['baseline']))
    if field.sha256 != model['baseline_field_sha256']:
        raise ValueError('Complete baseline field changed')
    external = predictions['external_field']
    _check_external(root, external, forward_hash=frozen.forward_hash,
                    protocol_sha256=record['protocol']['sha256'])
    if (external is None) != (external_sampler is None):
        raise ValueError('External field and its frozen sampler must both be supplied')
    if external is not None:
        sampler_path = Path(inspect.getsourcefile(external_sampler) or '').resolve()
        binding = external['artifacts']['interpolator_source']
        if sampler_path != _path(root, binding['path']):
            raise ValueError('Executed interpolator differs from frozen source')
        if (getattr(external_sampler, '__name__', None) != external['interpolator_entrypoint']
                or getattr(external_sampler, '__qualname__', None) != external['interpolator_entrypoint']):
            raise ValueError('Executed interpolator differs from frozen entrypoint')
    prefix = str(Path(record['directory']))
    _save(root, prefix+'/validation-attempt.json', _json({'freeze': freeze_binding}))
    validation = lm.reveal_validation_landmarks(payload, source_binding, partition, frozen)
    samples = None
    if external is not None:
        query = FieldQueries(validation.row_ids, tuple(map(tuple, validation.source_ras_mm)))
        samples = external_sampler(root, json.loads(_json(external)), query)
        if type(samples) is not FieldSamples:
            raise TypeError('The frozen external field must return typed, complete sample coverage')
        _check_external(root, external, forward_hash=frozen.forward_hash,
                        protocol_sha256=record['protocol']['sha256'])
    report = compare_landmarks(field, validation, external_samples=samples)
    for binding in (freeze_binding, record['model'], record['prediction_field'], record['protocol']):
        _read(root, binding)
    if running_sources() != record['running_sources']:
        raise ValueError('Executing sources changed during evaluation')
    report['freeze'] = freeze_binding
    report['landmark_freeze_hash'] = frozen.freeze_hash
    return _save(root, prefix+'/evaluation.json', _json(report)), report
