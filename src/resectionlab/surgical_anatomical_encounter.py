"""Source-linked, whole-cell anatomical near-encounters; no injury model.

This optional endpoint uses a supplied categorical context layer on its exact
source grid. A declared research margin expands the complete nominal tools.
The unique retained source cells intersected anywhere in a sequence are counted
once, using the exact box-distance kernel's closed-boundary tolerance of 1e-9mm.
Historical removal, new removal and this geometric union stay separate.
"""
from __future__ import annotations

import copy
from dataclasses import dataclass, field

import numpy as np
from scipy.spatial.transform import Rotation

from .core import array_digest, immutable_array, semantic_digest
from .patient_anatomy import PatientImagingCollection
from .surgical_motion import envelopes
from .surgical_tools import TOOLS
from .surgical_task_admission import admission, validate_admission

SCHEMA = 'resectionlab.source-context-encounter.v1'
METHOD = 'unique-retained-source-cells-in-complete-nominal-axial-tool-envelope-v1'
SCOPE = ('Whole-cell geometric near-encounter within a declared research margin; '
         'not tissue removal, mechanical injury, function or clinical deficit. '
         'Only the supplied source image grid is assessed; outside anatomy remains unknown.')


def _require(value, message):
    if not value: raise ValueError('ANATOMICAL_ENCOUNTER: '+message)


def _eligible(engine):
    value = admission(engine)
    if value is None: return []
    adapter = engine.scene.source['adapter']
    # Additional context must share an already byte-verified categorical source
    # and frame with a selected supplied annotation. No names infer anatomy.
    verified = [r for r in adapter['sourceLayers'].values()
                if r['provenance'].get('kind') == 'observed_source_annotation']
    native_layers = {r['id']: r for r in adapter['sourceManifest']['layers']}
    series = {r['id']: r for r in adapter['sourceManifest']['series']}
    result = []
    for row in adapter['sourceManifest']['layers']:
        sibling = next((r for r in verified if row.get('data') == r['sourceArray']
            and row.get('affine') == r['sourceAffine']
            and row.get('source_series') == native_layers[r['layerId']]['source_series']
            and row.get('provenance', {}).get('source') == r['provenance'].get('source')), None)
        if (sibling is not None and row.get('role') == 'context' and row.get('kind') == 'context'
                and type(row.get('label_value')) is int and row['label_value'] > 0
                and row.get('provenance', {}).get('kind') == 'observed_source_annotation'
                # add_layer's native_affine uses its source-series RAS frame;
                # older packages omit a redundant per-layer units field. This
                # inheritance is allowed only for the verified shared source
                # frame above. Explicit unknown/other layer units still refuse.
                and row.get('units_status', series.get(row.get('source_series'), {}).get('units_status')) == 'mm'
                and tuple(row['data']['shape']) == engine.scene.shape
                and np.array_equal(sibling['registeredAffine'], engine.scene.affine)):
            result.append((row, sibling))
    return result


def context_options(engine):
    return [{'layerId': r['id'], 'labelValue': r['label_value'], 'reviewStatus': r['review_status'],
        'description': r.get('label') or 'Supplied categorical context label '+str(r['label_value']),
        'coverage': r.get('coverage', 'unknown'), 'scope': SCOPE} for r, _ in _eligible(engine)]


@dataclass(frozen=True)
class AnatomicalObservation:
    mask: np.ndarray
    affine: np.ndarray
    review_status: str
    _array_layout: tuple = field(init=False, repr=False)

    def __post_init__(self):
        mask = np.asarray(self.mask); affine = np.asarray(self.affine, float)
        _require(mask.dtype == np.bool_ and mask.ndim == 3 and mask.size > 0 and np.any(mask),
                 'a nonempty explicit boolean context mask is required')
        _require(affine.shape == (4, 4) and np.isfinite(affine).all() and np.array_equal(affine[3], [0, 0, 0, 1])
            and abs(np.linalg.det(affine[:3, :3])) > 1e-9, 'a finite RAS millimeter grid is required')
        basis = affine[:3, :3]
        _require(np.allclose(basis.T@basis, np.diag(np.sum(basis*basis, axis=0)), rtol=0., atol=1e-10),
                 'this exact source-cell endpoint requires an orthogonal grid')
        _require(self.review_status in ('reviewed', 'unreviewed'), 'canonical public review status is required')
        object.__setattr__(self, 'mask', immutable_array(mask, bool))
        object.__setattr__(self, 'affine', immutable_array(affine, np.float64))
        object.__setattr__(self, '_array_layout', self._layout())

    def _layout(self):
        return tuple((a.shape, a.dtype.str, a.strides, a.__array_interface__['data'][0]) for a in (self.mask, self.affine))

    def assert_intact(self):
        _require(self._layout() == self._array_layout, 'immutable public context array layout changed')

    def manifest(self):
        self.assert_intact()
        return {'schema': SCHEMA, 'method': METHOD, 'maskHash': array_digest(self.mask),
            'affineHash': array_digest(self.affine), 'shape': list(self.mask.shape),
            'role': 'supplied-categorical-context', 'reviewStatus': self.review_status,
            'sampling': 'exact-identity-source-cell-membership', 'coverage': 'source-grid-only; outside-unassessed',
            'scope': SCOPE, 'evaluationReferencesExposed': False}


def project_context(engine, layer_id):
    """Native qualification only; source identity stays outside public inputs."""
    value = admission(engine); _require(value is not None, 'source baseline qualification is required')
    validate_admission(engine, value)
    matches = [(row, sibling) for row, sibling in _eligible(engine) if row['id'] == layer_id]
    _require(len(matches) == 1, 'selected context is unavailable or not bound to verified source bytes/frame')
    row, sibling = matches[0]
    _require(row['review_status'] == 'reviewed' or value['task']['allowUnreviewedEvidence'],
             'unreviewed context requires the qualified research opt-in')
    native = PatientImagingCollection(value['verification']['nativeCollectionPath'])
    _require(native.identity() == value['sourceEvidence']['sourceCollectionHash'], 'native collection changed')
    array = native.array(row['data'])
    registered = native.registration_matrix(row['source_series'])@np.asarray(row['affine'])
    _require(np.array_equal(registered, engine.scene.affine), 'exact source frame differs')
    observation = AnatomicalObservation(array == row['label_value'], registered,
        'reviewed' if row['review_status'] == 'reviewed' else 'unreviewed')
    binding = {'schema': SCHEMA, 'sourceCollectionHash': native.identity(),
        'nativeManifestSha256': value['verification']['nativeManifestSha256'],
        'layerId': layer_id, 'labelValue': row['label_value'], 'revision': row.get('revision', 0),
        'sourceLabel': row.get('label'),
        'sourceArray': copy.deepcopy(row['data']), 'original': copy.deepcopy(row['provenance']['source']),
        'registeredAffine': registered.tolist(), 'registration': copy.deepcopy(native.document['transforms']),
        'reviewStatus': row['review_status'], 'coverage': row.get('coverage', 'unknown'),
        'provenance': copy.deepcopy(row['provenance']), 'verifiedSiblingLayerId': sibling['layerId'],
        'publicObservation': observation.manifest()}
    binding['bindingSeal'] = semantic_digest(binding)
    return observation, binding


def normalize_limit(raw):
    _require(type(raw) is dict and set(raw) == {'marginMm', 'maxNearMm3'}, 'exact margin and near-volume controls required')
    for key, high in (('marginMm', 5.), ('maxNearMm3', 1000.)):
        _require(type(raw[key]) in (int, float) and np.isfinite(raw[key]) and 0 <= raw[key] <= high,
                 key+' exceeds the finite research bounds')
    return {key: float(raw[key]) for key in ('marginMm', 'maxNearMm3')}


def axial_envelope(start, end, tool, parameters):
    """Exact union of nominal fixed-axis capsules; no microstep inflation.

    Distinguish this geometric endpoint from the engine's conservative collision
    inflation. Axial collinear translation makes endpoint-capsule union exact
    and independent of how a command is subdivided. Placement is not a sweep.
    """
    _require(tool in ('cusa', 'frazier_suction'), 'only qualified CUSA/Frazier axial envelopes are supported')
    axis = Rotation.from_quat(start['quaternion']).apply([0., 0., 1.])
    other = Rotation.from_quat(end['quaternion']).apply([0., 0., 1.])
    _require(np.allclose(axis, other, atol=1e-10, rtol=0.) and
        np.linalg.norm(np.cross(np.asarray(end['position'])-start['position'], axis)) <= 1e-8,
        'only unchanged-axis axial translation is supported')
    first = envelopes(TOOLS[tool], parameters, start['position'], start['quaternion'])
    last = envelopes(TOOLS[tool], parameters, end['position'], end['quaternion'])
    return [(min((a, b, c, d), key=lambda p: float(p@axis)),
             max((a, b, c, d), key=lambda p: float(p@axis)), radius, part)
            for (a, b, radius, part), (c, d, _, _) in zip(first, last)]


class EncounterTracker:
    """Detached sequence evidence, never engine state or a physical injury law."""
    def __init__(self, observation, engine, limit):
        _require(type(observation) is AnatomicalObservation, 'typed public anatomical observation required')
        observation.assert_intact()
        _require(observation.mask.shape == engine.scene.shape and np.array_equal(observation.affine, engine.scene.affine),
                 'public context and native model frames differ')
        self.observation = observation; self.limit = normalize_limit(limit)
        self.baseline_removed = frozenset(engine.state['removed'])
        self.seen = {'shaft': set(), 'working': set()}

    def add_cover(self, engine, cover):
        self.observation.assert_intact()
        mask = self.observation.mask.ravel(); removed = set(engine.state['removed'])
        for start, end, radius, part in cover:
            cells = engine.cells(start, end, radius+self.limit['marginMm'])
            self.seen[part].update(int(cell) for cell in cells if mask[cell] and int(cell) not in removed)

    def observe(self, engine, action, previous_slots):
        for hand, slot in engine.state['slots'].items():
            params = engine.parameters(slot)
            cover = (axial_envelope(previous_slots[hand], slot, slot['instrument'], params)
                     if hand == action['slot'] and action['operation'] in ('move', 'apply', 'withdraw')
                     else envelopes(TOOLS[slot['instrument']], params, slot['position'], slot['quaternion']))
            self.add_cover(engine, cover)

    def receipt(self, engine):
        mask = self.observation.mask.ravel(); union = self.seen['shaft'] | self.seen['working']
        historical = sum(bool(mask[cell]) for cell in self.baseline_removed)
        new = set(engine.state['removed'])-self.baseline_removed
        volume = len(union)*engine.scene.voxel_mm3
        return {'schema': SCHEMA, 'method': METHOD, **self.limit,
            'sourceContextMm3': int(mask.sum())*engine.scene.voxel_mm3,
            'baselineRemovedContextMm3': historical*engine.scene.voxel_mm3,
            'baselineRetainedContextMm3': (int(mask.sum())-historical)*engine.scene.voxel_mm3,
            'newRemovedContextMm3': sum(bool(mask[cell]) for cell in new)*engine.scene.voxel_mm3,
            'nearSourceCells': sorted(union), 'nearPositionsMm': engine.scene.points(sorted(union)[:12]).tolist(), 'nearMm3': volume,
            'shaftNearMm3': len(self.seen['shaft'])*engine.scene.voxel_mm3,
            'workingNearMm3': len(self.seen['working'])*engine.scene.voxel_mm3,
            'withinLimit': volume <= self.limit['maxNearMm3'], 'scope': SCOPE}
