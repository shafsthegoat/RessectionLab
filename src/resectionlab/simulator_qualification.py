"""Evidence coverage and independent source-cell accounting, never admission.

Metadata declarations are not measured anatomy. Missing evidence stays unknown.
This module does not train, fetch data, infer anatomy, or change engine state.
"""
from __future__ import annotations

from collections import Counter
import hashlib
import json
from pathlib import Path

import numpy as np


SCHEMA = 'resectionlab.simulator-qualification.v1'
MR_MODALITIES = frozenset({'T1', 'T1c', 'T1CE', 'T2', 'FLAIR', 'MRI',
    'MRA', 'TOF-MRA', 'TOF', 'MRV', 'DTI', 'DWI', 'diffusion', 'SWI', 'DWI b0', 'DTI FA diagnostic'})


def file_sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def grid_check(affine, units):
    """Check declared coordinates only, without accepting registration."""
    try:
        a = np.asarray(affine, dtype=float)
        valid = (a.shape == (4, 4) and np.isfinite(a).all()
            and np.allclose(a[3], [0, 0, 0, 1])
            and abs(float(np.linalg.det(a[:3, :3]))) > 1e-9)
    except (ValueError, TypeError):
        valid = False
    return {'validAffine': bool(valid), 'declaredUnits': units,
        'millimeterGridDeclared': bool(valid and units == 'mm'),
        'spacingMm': np.linalg.norm(a[:3, :3], axis=0).tolist() if valid and units == 'mm' else None,
        'voxelMm3': abs(float(np.linalg.det(a[:3, :3]))) if valid and units == 'mm' else None,
        'registrationAccepted': False}


def native_coverage(manifest):
    """Keep every version separate; never union anatomy across subjects/versions."""
    patient = manifest['patient_id']
    series = manifest.get('series', [])
    layers = manifest.get('layers', [])
    mismatches = [r.get('id') for r in [*series, *layers] if r.get('patient_id') != patient]
    modalities = sorted({r.get('modality', 'UNKNOWN') for r in series})
    kinds = Counter(r.get('kind', 'unknown') for r in layers)
    angiography = sorted(set(modalities) & {'MRA', 'TOF-MRA', 'TOF', 'CTA', 'MRV', 'CTV'})
    grids = [grid_check(r.get('affine'), r.get('units_status')) for r in [*series, *layers]]
    diffusion = manifest.get('diffusion', {})
    vessel_labels = any(kinds[k] for k in ('vessels', 'vessel', 'artery', 'vein', 'sinus'))
    transforms = manifest.get('transforms', {})
    return {'patientId': patient, 'snapshotId': manifest.get('snapshot_id'),
        'revision': manifest.get('revision'), 'timepoint': manifest.get('study_timepoint'),
        'declaredSource': manifest.get('source', 'unknown'), 'declaredLicense': manifest.get('license', 'unknown'),
        'payloadVerification': 'not performed; metadata declarations only',
        'crossPatientRows': mismatches, 'modalities': modalities,
        'gridDeclarationsValid': bool(grids) and all(g['millimeterGridDeclared'] for g in grids),
        'series': [{'id': r.get('id'), 'modality': r.get('modality'),
            'originalSha256': r.get('original', {}).get('sha256'),
            'grid': grid_check(r.get('affine'), r.get('units_status')),
            'registration': transforms.get(r.get('id'), {'status': 'missing'})} for r in series],
        'layers': [{'id': r.get('id'), 'kind': r.get('kind'), 'sourceSeries': r.get('source_series'),
            'reviewStatus': r.get('review_status', 'unknown'), 'coverage': r.get('coverage', 'unknown'),
            'grid': grid_check(r.get('affine'), r.get('units_status')),
            'provenanceKind': r.get('provenance', {}).get('kind') if isinstance(r.get('provenance'), dict) else r.get('provenance'),
            'declaredDataSha256': r.get('data', {}).get('sha256')} for r in layers],
        'coverage': {'MRI': bool(set(modalities) & MR_MODALITIES),
            'CT': 'CT' in modalities, 'tumorLabels': bool(kinds['tumor']),
            'brainLabels': bool(kinds['brain']), 'skullLabels': bool(kinds['skull'] or kinds['bone']),
            'duraLabels': bool(kinds['dura']), 'vascularImaging': angiography,
            'vesselLabels': vessel_labels,
            'arteryVeinSinusLabels': {k: bool(kinds[k]) for k in ('artery', 'vein', 'sinus')},
            'diffusion': {'series': sorted(m for m in modalities if m in {'DTI', 'DWI', 'diffusion', 'DWI b0', 'DTI FA diagnostic'}),
                'declaredSourceVolumeCount': len(diffusion.get('source_members', [])),
                'tractographyQualified': False,
                'note': 'Diagnostic tensor scalars and raw diffusion are not motor/language pathway or mapping evidence'},
            'motorMapping': 'unknown', 'languageMapping': 'unknown',
            'intraoperativeUpdates': 'not established by this metadata adapter',
            'measuredMechanicalBoundaries': 'unknown', 'measuredVascularBoundaries': 'unknown'},
        'declaredMissing': manifest.get('missing', []),
        'sourceLimitations': manifest.get('source_limitations', []),
        'gates': {'patientSpecificVascularEvaluation': 'unsupported: registered typed graph and measured/declared boundaries not qualified',
            'motorOutcome': 'unsupported', 'languageOutcome': 'unsupported',
            'causalToolMechanics': 'unvalidated', 'largeScaleRL': 'hold'},
        'usableForMetadataComparison': not mismatches and bool(grids) and all(g['millimeterGridDeclared'] for g in grids)}


def validate_removed_ids(scene, ids, *, previous=()):
    """Independently check each accepted boundary without retaining its state."""
    if any(type(c) is not int or not 0 <= c < scene.labels.size for c in ids):
        raise ValueError('INVALID_REMOVED_SOURCE_ID')
    current = set(ids)
    if len(current) != len(ids):
        raise ValueError('DUPLICATE_REMOVED_SOURCE_ID')
    labels = scene.labels.ravel()
    regions = {r['id']: r for r in scene.regions}
    counts = Counter(int(labels[c]) for c in ids)
    if 0 in counts or not set(counts) <= set(regions):
        raise ValueError('REMOVED_SOURCE_WITHOUT_MATERIAL')
    if not set(previous) <= current:
        raise ValueError('SOURCE_REMOVAL_REGENERATED')
    return current


def removed_accounting(scene, state):
    """Count unique ORIGINAL labels, independently of engine.metrics()."""
    ids = state['removed']
    validate_removed_ids(scene, ids)
    labels = scene.labels.ravel()
    regions = {r['id']: r for r in scene.regions}
    counts = Counter(int(labels[c]) for c in ids)
    volume = abs(float(np.linalg.det(scene.affine[:3, :3])))
    target = sum(n * volume for label, n in counts.items() if regions[label].get('target'))
    total_target = sum(int(np.count_nonzero(scene.labels == r['id'])) * volume
        for r in scene.regions if r.get('target'))
    return {'sourceCellCount': len(ids), 'sourceIdsSha256': hashlib.sha256(
        np.asarray(sorted(ids), dtype='<i8').tobytes()).hexdigest(),
        'byLabelMm3': {regions[label]['name']: n * volume for label, n in sorted(counts.items())},
        'targetRemovedMm3': target, 'nonTargetRemovedMm3': len(ids) * volume - target,
        'initialTargetMm3': total_target, 'residualTargetMm3': total_target - target}


def removal_phases(scene, states, *, exposure_end, work_start, work_end):
    boundaries = [0, exposure_end, work_start, work_end, max(states)]
    if boundaries != sorted(boundaries) or set(boundaries) - set(states):
        raise ValueError('INVALID_PHASE_BOUNDARIES')
    results = {p: removed_accounting(scene, states[p]) for p in sorted(set(boundaries))}
    previous = set()
    for p in sorted(states):
        current = set(states[p]['removed'])
        if not previous <= current:
            raise ValueError('SOURCE_REMOVAL_REGENERATED')
        previous = current
    phases = []
    for name, a, b in zip(('authored_exposure', 'post_exposure_setup', 'finite_work', 'cleanup'), boundaries, boundaries[1:]):
        phases.append({'phase': name, 'start': a, 'end': b,
            **{key: results[b][key] - results[a][key] for key in ('targetRemovedMm3', 'nonTargetRemovedMm3')}})
    return {'phases': phases, 'fullEpisode': results[max(states)],
        'accessCostIncluded': True, 'authoredExposureIsValidatedAccess': False,
        'phaseBoundarySourceIdentityMonotone': True}


def role_reservations(root):
    """Read frozen metadata only. Never decode held-out imaging or landmarks."""
    records = []
    for name in ('manifests/experiments/remind-component-cohort-v1.json',
                 'manifests/resect-component-cohort-v1.json'):
        path = Path(root) / name
        document = json.loads(path.read_text())
        records.append({'path': name, 'sha256': file_sha256(path),
            'members': [{'patientGroup': r['patient_group'], 'role': r['role']} for r in document['members']],
            'payloadAccess': False, 'rolesChanged': False})
    return records
