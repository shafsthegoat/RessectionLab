"""Declarative admission for source-assisted post-exposure geometry tasks.

An admission qualifies a precise software task and native baseline. It does not
certify annotation completeness, measured mechanics, or clinical suitability.
Physical snapshots stay unchanged; this record belongs to the workspace.
"""
from __future__ import annotations

import copy
from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from .core import array_digest, freeze_json, semantic_digest, thaw_json
from .patient_anatomy import PatientImagingCollection, digest, file_hash
from .surgical_tools import catalog

SCHEMA = 'resectionlab.surgical-task-admission.v1'
TASK = 'post-exposure-inactive-access-v1'
INPUT_CONTRACT = 'annotation-assisted-source-geometry-v1'
OPERATIONS = ['exchange', 'place', 'move', 'withdraw']
UNKNOWN_ROLES = ('skull', 'vessels', 'functional', 'tractography', 'shift')
_DECLARATIONS_ROOT = Path(__file__).resolve().parents[2]/'testcases/downloaded-datasets'
_SOURCE_CHECKS = OrderedDict()


def _require(condition, message):
    if not condition: raise ValueError('TASK_ADMISSION: ' + message)


def _hash(value):
    return type(value) is str and len(value) == 64 and all(c in '0123456789abcdef' for c in value)


def source_evidence(scene, *, declarations_root=None):
    """Project actual source metadata; absent anatomy never becomes zero risk."""
    scene.require_cache_source()
    _require(any(r.get('target') and r['material'] == 'tumor' for r in scene.regions),
             'a source-declared tumor target is required')
    adapter = scene.source.get('adapter')
    _require(type(adapter) is dict and adapter.get('type') == 'native-patient-research-derivative',
             'an acquired native derivative with source provenance is required')
    manifest = adapter.get('sourceManifest')
    patient = adapter.get('sourcePatient')
    _require(type(manifest) is dict and type(patient) is str and manifest.get('patient_id') == patient,
             'patient and native manifest identity differ')
    snapshot = digest({k: v for k, v in manifest.items() if k != 'snapshot_id'})
    _require(manifest.get('snapshot_id') == snapshot == adapter.get('sourceCollectionHash'),
             'native semantic snapshot differs')
    _require(_hash(adapter.get('sourceManifestSha256')), 'original native-manifest byte hash is missing')
    _require(np.array_equal(scene.affine, adapter.get('targetAffine')), 'derivative coordinate frame differs')
    from .surgical_staged_planning import require_staged_development
    role = require_staged_development(SimpleNamespace(metadata={'surgical_adapter': adapter}),
        _DECLARATIONS_ROOT if declarations_root is None else declarations_root)
    series = manifest.get('series', [])
    _require(type(series) is list and bool(series), 'acquired imaging series are missing')
    _require(manifest.get('study_timepoint') == 'preoperative' and all(row.get('timepoint') == 'preoperative' for row in series),
             'this task requires a coherent preoperative source stage; later observations need a compatible task')
    images = []
    for row in series:
        _require(row.get('patient_id') == patient and row.get('units_status') == 'mm'
                 and _hash(row.get('original', {}).get('sha256')),
                 'image patient, spatial units or original-file identity is unknown')
        images.append({key: copy.deepcopy(row.get(key)) for key in
            ('id', 'modality', 'timepoint', 'acquisition_datetime', 'acquisition_resolution', 'affine', 'grid_spacing_mm')}
            | {'originalSha256': row['original']['sha256'], 'shape': row['data']['shape']})
    native_layers = {row['id']: row for row in manifest.get('layers', [])}
    annotations = scene.source.get('annotationHashes', {})
    layers = adapter.get('sourceLayers', {})
    _require(type(layers) is dict and bool(layers), 'source anatomy bindings are missing')
    evidence = {}
    for name, layer in layers.items():
        original = native_layers.get(layer.get('layerId'))
        _require(original is not None and original.get('patient_id') == patient,
                 'selected anatomy is not bound to the native patient')
        _require(layer.get('sampledMaskHash') == annotations.get(name)
                 and layer.get('sourceArray') == original.get('data')
                 and layer.get('sourceAffine') == original.get('affine')
                 and layer.get('reviewStatus') == original.get('review_status')
                 and layer.get('revision') == original.get('revision', 0)
                 and layer.get('provenance') == original.get('provenance', {}),
                 'anatomy data, review status or provenance differs from its native binding')
        evidence[name] = {'layerId': layer['layerId'], 'maskHash': layer['sampledMaskHash'],
            'sourceArraySha256': layer['sourceArray']['sha256'],
            'kind': 'source-annotation' if layer['provenance'].get('kind') == 'observed_source_annotation' else 'model-estimate',
            'reviewStatus': layer['reviewStatus'], 'revision': layer['revision'],
            'provenanceHash': semantic_digest(layer['provenance']),
            'coverage': original.get('coverage', 'unknown'), 'sourceSeries': original.get('source_series'),
            'sourceLabelValue': original.get('label_value'), 'nativeRole': original.get('role'),
            'sampling': layer['sampling'], 'registeredAffine': layer['registeredAffine'],
            'uncertainty': copy.deepcopy(layer.get('uncertainty'))}
    _require(all(row['name'] in evidence for row in scene.regions), 'material mapping has unbound anatomy')
    target_names = [row['name'] for row in scene.regions if row.get('target') and row['material'] == 'tumor']
    _require(bool(target_names) and all(evidence[name]['nativeRole'] == 'target' for name in target_names),
             'a source-declared tumor target is required')
    _require('brain' in evidence and any(r['name'] == 'brain' and r['material'] == 'soft_tissue' for r in scene.regions),
             'surrounding brain evidence is required')
    available = sorted({'target', *evidence, *({'skull'} if 'bone' in evidence else set())})
    return {'patientId': patient, 'studyTimepoint': manifest.get('study_timepoint'),
        'sourceCollectionHash': snapshot, 'sourceManifestSha256': adapter['sourceManifestSha256'],
        **role, 'images': images, 'anatomy': evidence,
        'registration': copy.deepcopy(manifest.get('transforms', [])),
        'nativeMissing': copy.deepcopy(manifest.get('missing', [])),
        'sourceLimitations': copy.deepcopy(manifest.get('source_limitations', [])),
        'availableEvidence': available, 'unknownEvidence': sorted(set(UNKNOWN_ROLES)-set(available)),
        'frame': 'RAS+', 'units': 'mm', 'shape': list(scene.shape), 'affine': scene.affine.tolist(),
        'sourceLabelsHash': array_digest(scene.labels), 'mapping': copy.deepcopy(scene.regions)}


def _sphere(value):
    _require(type(value) is dict and set(value) == {'centerMm', 'radiusMm'}, 'a bounded RAS sphere is required')
    point, radius = value['centerMm'], value['radiusMm']
    _require(type(point) is list and len(point) == 3 and all(type(x) in (int, float) and np.isfinite(x) for x in point),
             'sphere center must be finite RAS millimeters')
    _require(type(radius) in (int, float) and np.isfinite(radius) and .5 <= radius <= 15., 'sphere radius is outside task bounds')
    return {'centerMm': list(map(float, point)), 'radiusMm': float(radius)}


def task_spec(*, entry, axis, target, required_evidence=('brain', 'target'), allow_unreviewed=False):
    """A declared planning region/direction, never inferred completed removal."""
    direction = np.asarray(axis, float)
    _require(direction.shape == (3,) and np.isfinite(direction).all() and np.linalg.norm(direction) > 0.,
             'a finite nonzero proposal direction is required')
    _require(type(allow_unreviewed) is bool, 'unreviewed-evidence choice must be explicit')
    _require(type(required_evidence) in (list, tuple) and all(type(x) is str and x for x in required_evidence),
             'required evidence must be named explicitly')
    return {'id': TASK, 'entry': _sphere(entry), 'target': _sphere(target),
        'axisRasUnit': (direction/np.linalg.norm(direction)).tolist(),
        'requiredEvidence': sorted(set(required_evidence)), 'allowUnreviewedEvidence': allow_unreviewed,
        'allowedOperations': OPERATIONS[:], 'allowedTools': ['cusa'], 'toolRadiusMm': 1.5,
        'inputContract': INPUT_CONTRACT,
        'plannerInputs': ['source-material-labels', 'source-target-annotation', 'declared-protection-masks',
                          'current-native-physical-state', 'declared-tool-geometry'],
        'evaluationOnlyInputs': [], 'clinicalOutcomes': 'unsupported'}


def _validate_task(value):
    _require(type(value) is dict, 'task definition must be an object')
    expected = task_spec(entry=value.get('entry'), target=value.get('target'), axis=value.get('axisRasUnit'),
        required_evidence=value.get('requiredEvidence'), allow_unreviewed=value.get('allowUnreviewedEvidence'))
    # Normalization can round an already normalized axis by one ulp. Preserve
    # its exact declaration after independently checking unit length.
    _require(np.isclose(np.linalg.norm(value['axisRasUnit']), 1., rtol=0., atol=1e-12), 'proposal direction must be unit length')
    expected['axisRasUnit'] = value['axisRasUnit']
    _require(value == expected, 'task scope, permitted actions or observation contract differs')
    return expected


def _models(engine):
    return {'physicsHash': semantic_digest(engine.scene.properties), 'instrumentCatalogHash': semantic_digest(catalog()),
            'toolMechanicsRevision': 2, 'physicalClockRevision': 1}


def _source_members(scene):
    adapter = scene.source['adapter']; manifest = adapter['sourceManifest']
    originals = [row['original'] for row in manifest['series']]
    arrays = [row['data'] for row in manifest['series']]
    for row in adapter['sourceLayers'].values():
        arrays.append(row['sourceArray'])
        if row['provenance'].get('kind') == 'observed_source_annotation':
            original = row['provenance'].get('source', {})
            _require(_hash(original.get('sha256')) and type(original.get('path')) is str,
                     'supplied annotation original-file identity is missing')
            originals.append(original)
    return originals, arrays


def _verify_source_bytes(scene, verification):
    """Cold-check source bytes; warm reuse also checks every member's metadata.

    This process-private bounded cache never trusts a caller's self-seal as a
    byte-verification receipt. Replacement, writes or metadata changes require a
    full recheck. Paths stay within the declared native collection.
    """
    try:
        return _verify_source_members(scene, verification)
    except OSError as error:
        raise ValueError('TASK_ADMISSION: native source evidence is unavailable: '+str(error.filename)) from error


def _verify_source_members(scene, verification):
    collection = PatientImagingCollection(verification['nativeCollectionPath'])
    manifest_path = collection.root/'patient-anatomy.json'
    adapter = scene.source['adapter']
    _require(collection.identity() == adapter['sourceCollectionHash'], 'external native snapshot differs')
    originals, arrays = _source_members(scene)
    paths = [manifest_path] + [collection.path(row['path']) for row in originals+arrays]
    def fingerprint():
        result = []
        for path in paths:
            stat = path.stat()
            result.append((str(path.resolve()), stat.st_dev, stat.st_ino, stat.st_size, stat.st_mtime_ns, stat.st_ctime_ns))
        return tuple(result)
    before = fingerprint()
    key = (scene.identity, semantic_digest(verification))
    if _SOURCE_CHECKS.get(key) == before:
        _SOURCE_CHECKS.move_to_end(key)
        return
    _require(file_hash(manifest_path) == adapter['sourceManifestSha256'], 'original native manifest bytes differ')
    for row in originals:
        _require(file_hash(collection.path(row['path'])) == row['sha256'], 'original acquired image or annotation bytes differ')
    for row in arrays: collection.array(row)
    # Recompute the declared derivative from the verified native masks. Hashes
    # copied into adapter metadata alone cannot authorize exchanged labels.
    from .surgical_patient import _resample
    native_layers = {row['id']: row for row in collection.document['layers']}
    ownership = np.zeros(scene.shape, np.uint16)
    for region in scene.regions:
        name = region['name']; layer = adapter['sourceLayers'][name]
        original = native_layers[layer['layerId']]
        registered = collection.registration_matrix(original['source_series']) @ np.asarray(original['affine'])
        _require(np.allclose(registered, layer['registeredAffine'], rtol=0., atol=1e-12), 'registered anatomy frame differs')
        sampled = _resample(collection.array(original['data']), registered, scene.affine, scene.shape,
            label=original['label_value'], conservative=name in ('bone', 'vessels'))
        _require(array_digest(sampled) == layer['sampledMaskHash'], 'derived anatomy does not match verified native source')
        ownership[sampled] = region['id']
    _require(np.array_equal(ownership, scene.labels), 'material ownership does not match the verified source masks')
    _require(before == fingerprint(), 'native source changed during byte verification')
    _SOURCE_CHECKS[key] = before
    while len(_SOURCE_CHECKS) > 8: _SOURCE_CHECKS.popitem(last=False)


@dataclass(frozen=True)
class _BoundAdmission:
    record: object


def admission(engine):
    bound = getattr(engine, '_planning_task_admission', None)
    return thaw_json(bound.record) if type(bound) is _BoundAdmission else None


def validate_admission(engine, value, *, require_baseline=True):
    _require(type(value) is dict and set(value) == {'schema', 'sourceSceneId', 'sourceEvidence', 'task', 'models',
        'baseline', 'verification', 'admissionSeal'}, 'complete admission record is required')
    body = {key: row for key, row in value.items() if key != 'admissionSeal'}
    _require(value['schema'] == SCHEMA and semantic_digest(body) == value['admissionSeal'], 'admission seal differs')
    _require(value['sourceSceneId'] == engine.scene.identity, 'source scene differs')
    _require(value['sourceEvidence'] == source_evidence(engine.scene), 'source evidence or DEVELOPMENT declaration changed')
    _validate_task(value['task'])
    _require(value['models'] == _models(engine), 'physics, clock or instrument-model declaration differs')
    verification = value['verification']
    _require(type(verification) is dict and set(verification) == {'nativeCollectionPath', 'nativeManifestSha256', 'verifiedOriginalSha256', 'verifiedArraySha256', 'scope'}
             and type(verification['nativeCollectionPath']) is str and Path(verification['nativeCollectionPath']).is_absolute()
             and verification['nativeManifestSha256'] == value['sourceEvidence']['sourceManifestSha256']
             and verification['scope'] == 'source-byte-and-layout-consistency; no anatomical or clinical validation',
             'native verification declaration differs')
    originals, arrays = _source_members(engine.scene)
    _require(verification['verifiedOriginalSha256'] == sorted({row['sha256'] for row in originals})
             and verification['verifiedArraySha256'] == sorted({row['sha256'] for row in arrays}), 'verified source members differ')
    _verify_source_bytes(engine.scene, verification)
    if require_baseline:
        expected = {'stateId': engine.state_id, 'snapshotHash': semantic_digest(engine.snapshot()),
                    'actionsHash': semantic_digest(engine.actions), 'position': len(engine.actions)}
        _require(value['baseline'] == expected, 'return to the admitted native planning checkpoint')
        _require(engine.state.get('toolMechanicsRevision') == 2 and engine.state.get('physicalClockRevision') == 1,
                 'revised tool geometry and shared physical clock are required')
        from .surgical_cusa_v3 import validate_parameters
        for slot in engine.state['slots'].values():
            if slot['instrument'] == 'cusa': validate_parameters(engine.parameters(slot))
        _require(not any(s['active'] or s['inserted'] or s['grip'] for s in engine.state['slots'].values())
                 and not engine.state['holds'], 'both tools must be inactive, released and withdrawn')
        _require(not any(np.linalg.norm(v) > 0. for v in engine.state.get('deformation', {}).values()),
                 'this static geometry task requires an undeformed native baseline')
        _require(bool(engine.state['removed']), 'an executed post-exposure state is required')
        target_ids = [r['id'] for r in engine.scene.regions if r.get('target') and r['material'] == 'tumor']
        remaining = np.flatnonzero(np.isin(engine.scene.labels.ravel(), target_ids))
        remaining = np.setdiff1d(remaining, engine.state['removed'])
        selected = np.linalg.norm(engine.scene.points(remaining)-value['task']['target']['centerMm'], axis=1) <= value['task']['target']['radiusMm']
        _require(bool(np.any(selected)), 'declared target region has no retained source tumor')
        _require(not engine.touched(value['task']['entry']['centerMm'], value['task']['entry']['centerMm'], .5),
                 'declared entry center is not clear in the completed native state')
        removed = set(engine.state['removed'])
        entry = value['task']['entry']['centerMm']
        _require(any(cell in removed for cell in engine.cells(entry, entry, .5)),
                 'declared entry is not linked to locally removed source tissue')
        if not value['task']['allowUnreviewedEvidence']:
            _require(all(row['reviewStatus'] == 'reviewed' for row in value['sourceEvidence']['anatomy'].values())
                     and all(row.get('status') in ('reference', 'reviewed') for row in value['sourceEvidence']['registration'].values()),
                     'source anatomy or registration is unreviewed; explicit research acceptance is required')
    return value


def bind_admission(engine, value):
    validate_admission(engine, value, require_baseline=False)
    engine._planning_task_admission = _BoundAdmission(freeze_json(value))
    return engine


def prepare_admission(engine, *, native_collection, task):
    """Verify actual native bytes once, then seal a source-bound baseline task."""
    evidence = source_evidence(engine.scene)
    from .surgical_engine import SurgicalEngine
    SurgicalEngine.restore(engine.scene, engine.snapshot(), use_cache=False)
    collection = PatientImagingCollection(native_collection)
    native_path = collection.root/'patient-anatomy.json'
    _require(collection.identity() == evidence['sourceCollectionHash'] and file_hash(native_path) == evidence['sourceManifestSha256'],
             'original native manifest differs from the source adapter')
    originals, arrays = _source_members(engine.scene)
    value = {'schema': SCHEMA, 'sourceSceneId': engine.scene.identity, 'sourceEvidence': evidence,
        'task': _validate_task(task), 'models': _models(engine),
        'baseline': {'stateId': engine.state_id, 'snapshotHash': semantic_digest(engine.snapshot()),
                     'actionsHash': semantic_digest(engine.actions), 'position': len(engine.actions)},
        'verification': {'nativeCollectionPath': str(collection.root.resolve()), 'nativeManifestSha256': file_hash(native_path),
            'verifiedOriginalSha256': sorted({row['sha256'] for row in originals}),
            'verifiedArraySha256': sorted({row['sha256'] for row in arrays}),
            'scope': 'source-byte-and-layout-consistency; no anatomical or clinical validation'}}
    value['admissionSeal'] = semantic_digest(value)
    validate_admission(engine, value)
    return value


def check_program(engine, actions):
    value = admission(engine)
    if value is None: return
    task = value['task']
    tools = {hand: slot['instrument'] for hand, slot in engine.state['slots'].items()}
    for action in actions:
        hand, operation = action.get('slot'), action.get('operation')
        _require(hand in tools and operation in task['allowedOperations'], 'program exceeds the admitted task actions')
        if operation == 'exchange': tools[hand] = action.get('instrument')
        else: _require('instrument' not in action, 'instrument exchange must be explicit')
        _require(tools[hand] in task['allowedTools'] and
                 action.get('parameters', {}).get('radiusMm', task['toolRadiusMm']) == task['toolRadiusMm'],
                 'program exceeds the admitted instrument geometry')
