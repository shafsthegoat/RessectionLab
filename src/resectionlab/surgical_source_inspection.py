"""Exact-grid source exports with their provenance marker.

Arbitrary source annotations remain independent masks, including overlaps.
Owner decision 2026-11-27: the engine, planners and RL environment operate on
any 3D reconstruction, so the ``cranialPlanningCandidateV1`` marker is a
provenance label, never a refusal. A scene carrying it is an ordinary operable
scene; an explicit protection declaration becomes real protection. The compiled
resident simulator still treats the marker as its retired inspection contract,
so such scenes run on the Python reference path (see ``legacy_marker``).
"""
from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path

import numpy as np

from .core import CaseData, SourceRef, array_digest
from .patient_anatomy import PatientImagingCollection

PROPERTY = 'cranialPlanningCandidateV1'
SCHEMA = 'resectionlab.cranial-planning-candidate.v1'


def legacy_marker(scene):
    """True when a scene carries the provenance marker; it grants and refuses nothing."""
    return PROPERTY in scene.properties


def inspection_case(collection_path, *, source_group, target_layer_ids,
                    expected_originals, expected_reference_grid, expected_target_cells,
                    expected_target_ids_sha256, blockers,
                    critical_evidence=None, evidence_identity=None, check=lambda: None):
    """Build a source-labelled CaseData from an existing verified native collection.

    No resampling, registration estimation, segmentation, material adoption or
    native transition is performed. All collection layers are retained; the
    caller must supply exact originals and the complete original target count.
    Other source series are referenced, not decoded into duplicate MRI fields.
    Paths are trusted builder configuration, not case/request authority.
    """
    check()
    collection = PatientImagingCollection(collection_path)
    manifest = copy.deepcopy(collection.document)
    manifest_path = Path(collection.root)/'patient-anatomy.json'
    manifest_bytes = manifest_path.read_bytes()
    manifest_sha = hashlib.sha256(manifest_bytes).hexdigest()
    if json.loads(manifest_bytes) != manifest:
        raise ValueError('SOURCE_INSPECTION_MANIFEST_CHANGED')
    if manifest['patient_id'] != source_group:
        raise ValueError('SOURCE_INSPECTION_PATIENT_DIFFERS')
    series, rows = manifest['series'], manifest['layers']
    targets = set(target_layer_ids)
    if (not targets or not targets <= {r['id'] for r in rows} or len(rows) > 32
            or not isinstance(expected_originals, dict) or not expected_originals
            or type(expected_target_cells) is not int or expected_target_cells < 1):
        raise ValueError('SOURCE_INSPECTION_EXPLICIT_COMPLETE_INPUTS_REQUIRED')
    if (not isinstance(blockers, list) or not blockers or any(type(r) is not dict
            or type(r.get('code')) is not str or not r['code']
            or type(r.get('message')) is not str or not r['message'] for r in blockers)):
        raise ValueError('SOURCE_INSPECTION_EXPLICIT_BLOCKERS_REQUIRED')
    reference = collection.series(manifest['reference_series'])
    shape = tuple(reference['data']['shape'])
    # This is the existing desktop CaseData accounting, not an increased cap.
    estimate = int(np.prod(shape))*(4+2*len(rows))+128
    estimate += sum(e.mask.nbytes+e.annotation_coverage.nbytes+e.affine_ras_mm.nbytes
                    for e in (critical_evidence or {}).values())
    if estimate > 1024**3:
        raise ValueError('SOURCE_INSPECTION_DESKTOP_CASE_SIZE_LIMIT')
    if reference.get('units_status') != 'mm':
        raise ValueError('SOURCE_INSPECTION_REFERENCE_MM_REQUIRED')
    affine = collection.patient_affine(reference)
    if (not isinstance(expected_reference_grid, dict)
            or expected_reference_grid.get('seriesId') != reference['id']
            or expected_reference_grid.get('originalSha256') != reference['original']['sha256']
            or expected_reference_grid.get('shape') != list(shape)
            or not np.array_equal(expected_reference_grid.get('affineRasMm'), affine)):
        raise ValueError('SOURCE_INSPECTION_REFERENCE_GRID_DIFFERS')
    originals = {r['original']['name']: r['original']['sha256'] for r in series}
    if len(originals) != len(series) or originals != expected_originals:
        raise ValueError('SOURCE_INSPECTION_ALL_ORIGINALS_REQUIRED')
    refs = []
    for row in series:
        check()
        original = row['original']
        path = collection.path(original['path'])
        with path.open('rb') as stream:
            sha = hashlib.file_digest(stream, 'sha256').hexdigest()
        if row['patient_id'] != source_group or sha != original['sha256']:
            raise ValueError('SOURCE_INSPECTION_ORIGINAL_CHANGED')
        refs.append(SourceRef('image:'+row['id'], path.as_uri(), sha, native_frame='RAS+', provenance='observed'))
    masks, layers, declarations, target_union = {}, {}, {}, np.zeros(shape, bool)
    for row in rows:
        check()
        if (row['patient_id'] != source_group or tuple(row['data']['shape']) != shape
                or not np.array_equal(collection.patient_affine(row), affine)):
            raise ValueError('SOURCE_INSPECTION_EXACT_NATIVE_GRID_REQUIRED')
        raw = collection.array(row['data'])
        mask = np.asarray(raw == row['label_value'], dtype=bool)
        name = row['id']
        provenance = ('observed' if row.get('provenance', {}).get('kind') == 'observed_source_annotation'
                      else 'estimated')
        masks[name] = mask
        if name in targets:
            target_union |= mask
        # Do not turn inspection exclusions into millions of executable source
        # IDs/KD points. The full masks and exact original declarations remain.
        layers[name] = {'kind': 'tumor' if name in targets else 'unknown',
            'color': row.get('color', '#92a7b8'), 'review_status': row.get('review_status', 'unreviewed'),
            'rehearsal_target': name in targets, 'provenance': provenance,
            'source_kind': row['kind'], 'source_role': row['role'],
            'source_provenance': copy.deepcopy(row.get('provenance', {})),
            'inspection_protection_declared': row['role'] == 'critical',
            'coverage': row.get('coverage', 'Source annotation; completeness unknown.')}
        declarations[name] = {'sourceLayer': copy.deepcopy(row), 'maskHash': array_digest(mask),
                              'cells': int(mask.sum()), 'physicalProtectionIndexConstructed': False}
        refs.append(SourceRef('annotation:'+name, collection.path(row['data']['path']).as_uri(),
                              row['data']['sha256'], native_frame='RAS+', provenance=provenance))
    target_ids = np.flatnonzero(target_union).astype('<u4')
    if (len(target_ids) != expected_target_cells
            or hashlib.sha256(target_ids.tobytes()).hexdigest() != expected_target_ids_sha256):
        raise ValueError('SOURCE_INSPECTION_FULL_ORIGINAL_TARGET_DIFFERS')
    refs.append(SourceRef('source-manifest', manifest_path.as_uri(), manifest_sha, native_frame='RAS+'))
    marker = {'schema': SCHEMA, 'status': 'inspection-only', 'sourcePatientGroup': source_group,
        'sourceCollectionHash': collection.identity(), 'fullOriginalTargetCells': expected_target_cells,
        'fullOriginalTargetIdsSHA256': expected_target_ids_sha256,
        'targetIdConvention': 'C-order flat uint32 little-endian bytes on the unchanged original reference grid',
        'targetLayerIds': sorted(targets), 'blockers': copy.deepcopy(blockers),
        'physicalProtectionIndexConstructed': False, 'patientTaskAdmission': False,
        'officialTrainClaim': False, 'annotationAuthority': False,
        'meaning': 'Complete source masks/declarations retained with their provenance. The scene is operable; protection applies only where a mapping declares it, and material/entry/work/exit remain unqualified.'}
    mri = collection.array(reference['data'])
    check()
    case = CaseData(case_id=source_group+'-source-inspection', mri=mri, affine=affine,
        compartments=masks, source_compartments=masks, source_refs=tuple(refs), frame='RAS+',
        critical_evidence=critical_evidence or {},
        unknowns=('Unlabelled and outside-source anatomy remain unknown; source zeros are not certified air.',
                  'Brain/vessel/skin/dura/falx, calvarium/support and tool space remain unqualified unless separately evidenced.'),
        metadata={'anatomy_workbench_only': True, 'anatomy_layers': layers,
            **({'evidence_identity': copy.deepcopy(evidence_identity)} if evidence_identity is not None else {}),
            'surgical_properties': {PROPERTY: marker},
            'source_inspection_annotations': declarations,
            'data_case': {'family_id': source_group, 'role': 'DEVELOPMENT_CANDIDATE', 'split': 'DEVELOPMENT_CANDIDATE'},
            'surgical_adapter': {'type': 'native-patient-source-inspection', 'sourceRole': 'DEVELOPMENT_CANDIDATE',
                'sourcePatient': source_group, 'sourceCollectionHash': collection.identity(),
                'sourceManifestSha256': manifest_sha, 'sourceManifest': manifest,
                'sourceLayers': copy.deepcopy(declarations), 'ownershipPrecedence': [r['id'] for r in rows],
                'targetAffine': affine.tolist(), 'intensityResampling': 'none; exact native reference',
                'physicalProtectionIndexConstructed': False}})
    if hashlib.sha256(manifest_path.read_bytes()).hexdigest() != manifest_sha:
        raise ValueError('SOURCE_INSPECTION_MANIFEST_CHANGED')
    check()
    return case


def export_inspection_case(collection_path, destination, **kwargs):
    """Save the existing .ressectionlab format; no new project/dashboard format."""
    from .imaging import save_case
    case = inspection_case(collection_path, **kwargs)
    return save_case(case, destination)
