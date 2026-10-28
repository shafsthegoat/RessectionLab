"""Memory-bounded research derivative of explicitly selected patient annotations.

The source collection is read-only. Unreviewed registration requires an explicit
opt-in; its status and evidence are copied verbatim, never promoted to reviewed.
"""
from __future__ import annotations

import copy
import hashlib
import itertools
import json
from pathlib import Path
import numpy as np
from scipy.ndimage import affine_transform
from .core import CaseData, SourceRef, array_digest
from .imaging import save_case
from .patient_anatomy import PatientImagingCollection

ORDER = ('brain', 'tumor', 'bone', 'vessels')
MATERIAL = {'brain': 'soft_tissue', 'tumor': 'tumor', 'bone': 'bone', 'vessels': 'vessel'}
COLORS = {'brain': '#c799b0', 'tumor': '#db6b95', 'bone': '#e5d6b8', 'vessels': '#e7645c'}
ANATOMY_KINDS = {'brain': 'brain', 'bone': 'skull', 'tumor': 'tumor', 'vessels': 'vessels'}
COVERAGE = {
    'brain': 'Brain extraction estimate; boundary and tumor-distorted regions unreviewed.',
    'bone': 'CT bone-density estimate includes teeth and other bone; skull boundary and registration unreviewed.',
    'tumor': 'Tumor proposal without supplied ground-truth annotation; extent and boundaries unreviewed.',
    'vessels': 'Experimental structural-MRI vessel estimate; vascular completeness and small branches unvalidated.',
}


def _anatomy_metadata(name, row):
    provenance = row.get('provenance', {})
    inference = provenance.get('inference', {})
    model = inference.get('model', {})
    model_name = model.get('name') if isinstance(model, dict) else model
    if not model_name:
        declared = provenance.get('model')
        model_name = declared.get('name') if isinstance(declared, dict) else declared
    if not model_name:
        model_name = provenance.get('method') or 'Unreviewed source segmentation estimate'
    source_annotation = provenance.get('kind') == 'observed_source_annotation'
    coverage = COVERAGE[name]
    if name == 'vessels' and provenance.get('source_modality') in {'MRA', 'TOF', 'TOF-MRA', 'CTA'}:
        coverage = 'Angiography-derived vessel annotation or estimate; vascular completeness and small branches unvalidated.'
    return {'kind': ANATOMY_KINDS[name], 'color': COLORS[name],
            'provenance': 'observed' if source_annotation else 'estimated',
            'review_status': row.get('review_status', 'unreviewed'), 'model': {'name': model_name},
            'coverage': (row.get('coverage') or 'Source annotation; completeness unassessed.') if source_annotation
                        else coverage + ' ' + row.get('coverage', ''),
            'source_provenance': copy.deepcopy(provenance)}

DEFAULT_PHYSICS = {'schema': 'resectionlab.tissue-physics.v1',
    'provenance': {'kind': 'assumed-research', 'description': 'Explicit research wound-aperture assumptions; radius is NOT inferred vessel diameter. No calibrated patient mechanics.'},
    'bone': {'model': 'rigid-cell-removal-v1'},
    'vessel': {'model': 'local-poiseuille-v1', 'radiusMm': .15, 'pressurePa': 1000,
               'viscosityPaS': .0035, 'pathLengthMm': 5, 'maxFlowMm3PerSecond': 50, 'coagulationExposure': 2}}


def _sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def _resample(array, source_affine, target_affine, shape, *, label=None, conservative=False):
    """Chunked linear intensities or categorical sampling at <=0.5 mm spacing."""
    matrix = np.linalg.inv(source_affine) @ target_affine
    result = np.zeros(shape, np.float32 if label is None else bool)
    # 2 samples per mm per axis; finer requested grids use a single center.
    subdivisions = np.maximum(1, np.ceil(np.linalg.norm(target_affine[:3, :3], axis=0) / .5).astype(int)) if conservative else np.ones(3, int)
    offsets = list(itertools.product(*[(np.arange(n) + .5) / n - .5 for n in subdivisions]))
    for z in range(0, shape[2], 8):
        block_shape = (shape[0], shape[1], min(8, shape[2] - z))
        target = result[:, :, z:z + block_shape[2]]
        for offset in offsets:
            origin = matrix[:3, 3] + matrix[:3, :3] @ (np.array([0, 0, z]) + offset)
            sampled = affine_transform(array, matrix[:3, :3], origin, output_shape=block_shape,
                                       output=np.float32 if label is None else array.dtype,
                                       order=1 if label is None else 0, mode='constant', cval=0, prefilter=False)
            if label is None:
                target[:] = sampled
            else:
                target |= sampled == label
    return result


def prepare_surgical_patient(collection_path, output_dir, *, layer_ids,
                             resolution_mm=1., allow_unreviewed_registration=False,
                             tissue_physics=None, require_complete=True):
    """Save patient.ressectionlab and report.json; return the auditable report.

    layer_ids maps exactly brain/tumor/bone/vessels to existing collection IDs.
    MRI uses linear interpolation; brain/tumor use nearest-neighbor centers.
    Bone/vessels use any-hit samples spaced at most 0.5 mm within each output
    cell. This deliberately expands coarse-cell occupancy; it neither fills
    holes nor guarantees capture of structures below the sampling spacing.
    All overlapping sampled masks remain annotations; ownership priority is
    brain < tumor < bone < vessels. No attachments are fabricated.
    """
    if require_complete and (set(layer_ids) != set(ORDER) or len(set(layer_ids.values())) != 4):
        raise ValueError('Select exactly four distinct brain/tumor/bone/vessels layers')
    if not layer_ids or not set(layer_ids) <= set(ORDER) or len(set(layer_ids.values())) != len(layer_ids):
        raise ValueError('Select distinct existing anatomy layers for the declared roles')
    selected_order = tuple(name for name in ORDER if name in layer_ids)
    if not np.isfinite(resolution_mm) or not .5 <= resolution_mm <= 2:
        raise ValueError('Research derivative resolution must be between 0.5 and 2 mm')
    root = Path(collection_path)
    if root.is_file(): root = root.parent
    collection = PatientImagingCollection(root)
    manifest_path = root / 'patient-anatomy.json'
    manifest_hash = _sha(manifest_path)
    manifest = copy.deepcopy(collection.document)
    reference = collection.series(manifest['reference_series'])
    reference_affine = collection.patient_affine(reference)
    spacing = np.linalg.norm(reference_affine[:3, :3], axis=0)
    # Preserve original voxel axes, including handedness and obliquity.
    target_affine = reference_affine.copy()
    target_affine[:3, :3] = reference_affine[:3, :3] / spacing * resolution_mm
    ratio = resolution_mm / spacing
    target_affine[:3, 3] = (reference_affine @ np.r_[((ratio - 1) / 2), 1])[:3]
    shape = tuple(np.ceil(np.asarray(reference['data']['shape']) / ratio).astype(int))
    if np.prod(shape) > 80_000_000:
        raise ValueError('Requested derivative exceeds 80 million cells')
    rows = {name: collection.layer(layer_ids[name]) for name in selected_order}
    for name, row in rows.items():
        if row.get('units_status', 'mm') != 'mm':
            raise ValueError('Selected anatomy layer requires declared millimeter units: ' + name)
    warnings = ['Research derivative of source annotations and segmentation estimates; provenance and review status are retained, not independently validated.',
                'Bone/vessel any-hit coarse occupancy can enlarge boundaries and still miss structures smaller than 0.5 mm.',
                'Wound aperture and flow parameters are explicit assumptions, not measured vessel diameter or patient physiology.']
    affines = {}
    source_refs = []
    used_series = {reference['id'], *(r['source_series'] for r in rows.values())}
    for sid in sorted(used_series):
        series = collection.series(sid)
        if series.get('units_status') != 'mm':
            raise ValueError('Every source series must declare millimeter units')
        transform = manifest['transforms'].get(sid)
        if not transform:
            raise ValueError('Missing registration for ' + sid)
        # Validate source hashes, transform digest and parent graph regardless of status.
        matrix = collection.registration_matrix(sid)
        chain, current = [], sid
        while current:
            record = manifest['transforms'][current]
            chain.append((current, record['status']))
            current = record.get('parent_series')
        unreviewed = [name for name, status in chain if status not in ('reference', 'reviewed', 'source-provided')]
        if unreviewed:
            if not allow_unreviewed_registration:
                raise ValueError('Unreviewed registration requires explicit research opt-in: ' + ', '.join(unreviewed))
            warnings.append('Research opt-in uses unreviewed registration: ' + ', '.join(unreviewed) + '; source review status unchanged.')
        warnings.extend(transform.get('quality', {}).get('warnings', []))
        affines[sid] = matrix
        original = series['original']
        original_path = collection.path(original['path'])
        if _sha(original_path) != original['sha256']:
            raise ValueError('Source original checksum differs')
        source_refs.append(SourceRef(sid, original_path.as_uri(), original['sha256'], native_frame='RAS+'))
    source_refs.append(SourceRef('source-manifest', manifest_path.resolve().as_uri(), manifest_hash, native_frame='RAS+'))
    mri = _resample(collection.array(reference['data']), reference_affine, target_affine, shape)
    masks, layer_report = {}, {}
    for name in selected_order:
        row = rows[name]
        raw = collection.array(row['data'])
        registered = affines[row['source_series']] @ np.asarray(row['affine'])
        sampled = _resample(raw, registered, target_affine, shape, label=row['label_value'], conservative=name in ('bone', 'vessels'))
        if not sampled.any():
            raise ValueError('Selected layer has no occupied cells in the derivative: ' + name)
        masks[name] = sampled
        original_count = sum(int(np.count_nonzero(raw[:, :, z] == row['label_value'])) for z in range(raw.shape[2]))
        layer_report[name] = {'layerId': row['id'], 'sourceOccupiedCells': original_count,
            'sampledOccupiedCells': int(sampled.sum()), 'sampledMaskHash': array_digest(sampled),
            'sourceArray': copy.deepcopy(row['data']), 'sourceAffine': row['affine'],
            'revision': row.get('revision', 0), 'originalData': copy.deepcopy(row.get('original_data')),
            'edits': copy.deepcopy(row.get('edits', [])),
            'registeredAffine': registered.tolist(), 'provenance': copy.deepcopy(row.get('provenance', {})),
            'reviewStatus': row.get('review_status'),
            'uncertainty': copy.deepcopy(row.get('uncertainty')), 'scoreMeaning': row.get('score_meaning'),
            'probability': copy.deepcopy(row.get('probability')), 'sampling': 'any-hit <=0.5mm samples' if name in ('bone', 'vessels') else 'nearest-neighbor center'}
    overlaps = {a + ':' + b: int(np.count_nonzero(masks[a] & masks[b])) for a, b in itertools.combinations(selected_order, 2)}
    ownership = np.zeros(shape, np.uint8)
    for i, name in enumerate(selected_order, 1): ownership[masks[name]] = i
    ownership_counts = {name: int(np.count_nonzero(ownership == i)) for i, name in enumerate(selected_order, 1)}
    if not all(ownership_counts.values()):
        raise ValueError('Ownership precedence eliminated an entire requested material')
    physics = copy.deepcopy(DEFAULT_PHYSICS if tissue_physics is None else tissue_physics)
    adapter = {'type': 'native-patient-research-derivative', 'warnings': list(dict.fromkeys(warnings)),
        'representations': {'acquiredImages': 'immutable source original checksums and source reference array',
            'anatomicalEstimates': 'selected revisioned segmentation masks; review status and uncertainty retained',
            'displaySurfaces': 'excluded; display smoothing never changes these simulation masks'},
        'sourcePatient': manifest['patient_id'], 'sourceCollectionHash': collection.identity(),
        'sourceManifestSha256': manifest_hash, 'sourceManifest': manifest,
        'resolutionMm': resolution_mm, 'targetAffine': target_affine.tolist(),
        'intensityResampling': 'linear interpolation of original reference image; outside-source samples zero; no modeled intensity changes',
        'sourceLayers': layer_report, 'overlapCells': overlaps, 'ownershipPrecedence': list(selected_order),
        'ownershipCells': ownership_counts, 'allowUnreviewedRegistration': bool(allow_unreviewed_registration)}
    physics_key = 'tissuePhysicsV2' if physics.get('schema') == 'resectionlab.tissue-physics.v2' else 'tissuePhysics'
    case = CaseData(case_id=manifest['patient_id'] + '-surgical-research', mri=mri,
        compartments=masks, affine=target_affine, frame='RAS+', source_refs=tuple(source_refs),
        metadata={'surgical_adapter': adapter, 'surgical_properties': {physics_key: physics},
            'anatomy_layers': {name: _anatomy_metadata(name, rows[name]) for name in selected_order},
            'missing_anatomy': [name for name in ORDER if name not in rows],
            'description': 'Acquired reference image with explicitly selected source annotations and research estimates; missing anatomy remains unknown.'})
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    path = save_case(case, output / 'patient.ressectionlab')
    if _sha(manifest_path) != manifest_hash:
        raise ValueError('Source manifest changed during export')
    report = {'schema': 'resectionlab.surgical-patient-export.v1', 'casePath': str(path.resolve()),
              'caseHash': case.semantic_hash, 'shape': list(map(int, shape)), 'voxelMm3': abs(float(np.linalg.det(target_affine[:3, :3]))),
              'ownershipVolumesMm3': {k: v * abs(float(np.linalg.det(target_affine[:3, :3]))) for k, v in ownership_counts.items()},
              'tissuePhysics': physics, **adapter}
    (output / 'report.json').write_text(json.dumps(report, indent=2, allow_nan=False) + '\n')
    return report


def prepare_surgical_environment(collection_path, output_dir, *, layer_ids,
                                 resolution_mm=1., allow_unreviewed_registration=False):
    """One interoperable export path for desktop preparation, CLI, and RL.

    Missing roles are explicit. Source manifests are never rewritten. The saved
    scene archive pins the complete physics profile, mapping, and source identity.
    """
    from .imaging import load_case
    from .surgical_scene import from_case
    from .surgical_engine import SurgicalEngine
    from .surgical_environment import SCHEMA, standardize
    from .surgical_physics_v2 import default_profile
    report = prepare_surgical_patient(collection_path, output_dir, layer_ids=layer_ids,
        resolution_mm=resolution_mm, allow_unreviewed_registration=allow_unreviewed_registration,
        tissue_physics=default_profile(), require_complete=False)
    case = load_case(report['casePath'])
    scene = standardize(from_case(case))
    from .surgical_rl_protection import GuardConfig
    engine = SurgicalEngine(scene, guard_config=GuardConfig(require_critical_coverage=False))
    record = {'environmentSchema': SCHEMA, 'physicsProfile': copy.deepcopy(scene.properties['tissuePhysicsV2']),
              'mapping': scene.regions, 'branches': {'main': engine.snapshot()}, 'branchId': 'main',
              'position': 0, 'parents': {'main': None}, 'timeline': {'main': []}}
    save_case(case, report['casePath'], artifacts={'surgical_toolbox': record})
    environment = {**scene.descriptor()['environment'], 'caseHash': case.semantic_hash,
                   'sceneId': scene.identity, 'regions': scene.regions,
                   'physicsProfile': record['physicsProfile'],
                   'provenanceReport': 'report.json'}
    destination = Path(output_dir) / 'environment.json'
    destination.write_text(json.dumps(environment, indent=2, allow_nan=False) + '\n')
    return {**report, 'environmentPath': str(destination.resolve()), 'environment': environment}
