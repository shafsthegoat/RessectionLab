"""Read-only original-image joins to trusted, predecode source-role receipts.

Case metadata is a claim. It cannot grant TRAIN status to an evaluation patient.
The application/CLI supplies the trusted catalog root, never a path in the case.
Owner decision 2026-11-27: a reserved or unproven role is recorded, not
refused. Planning, search and learning may use the source; their outputs carry
this role so a later evaluation can exclude it. Reserved pixels stay undecoded.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import re

import nibabel as nib
import numpy as np

from .core import array_digest


def _sha(path):
    with path.open('rb') as stream: return hashlib.file_digest(stream, 'sha256').hexdigest()


def patient_role_evidence(case, role_root=None):
    if (case.metadata.get('generated') is True and case.metadata.get('is_synthetic') is True
            and case.source_refs and all(ref.provenance == 'simulated' and ref.uri.startswith('synthetic://')
                                        for ref in case.source_refs)):
        return {'status': 'generated_control', 'patientEvidence': False}
    if role_root is None:
        return {'status': 'unavailable', 'reason': 'SOURCE_ROLE_EVIDENCE_REQUIRED'}
    root = Path(role_root).resolve()
    family = case.metadata.get('data_case', {}).get('family_id')
    match = re.fullmatch(r'TopCoW:(\d{3})', family or '')
    if not match:
        return {'status': 'unavailable', 'reason': 'UNSUPPORTED_SOURCE_ROLE_REGISTRY'}
    path = root/'catalog'/'topbrain'/f'{match[1]}-selective-extraction.json'
    if not path.is_file():
        return {'status': 'unavailable', 'reason': 'SOURCE_ROLE_RECEIPT_UNAVAILABLE'}
    receipt = json.loads(path.read_text()); declared = receipt.get('family', {})
    if declared.get('family_id') != family:
        raise ValueError('SOURCE_ROLE: trusted receipt names a different family')
    if declared.get('role') != 'TRAIN' or declared.get('role_frozen_before_new_image_decode') is not True:
        # Recorded, not refused. No reserved image is opened to verify pixels,
        # so this evidence never claims a pixel join.
        role = declared.get('role')
        return {'status': 'reserved_or_unproven_source', 'patientEvidence': True, 'family': family,
            'role': role if type(role) is str and role else 'UNKNOWN',
            'roleFrozenBeforeDecode': declared.get('role_frozen_before_new_image_decode') is True,
            'receiptSha256': _sha(path), 'pixelsVerified': False,
            'evaluationUse': 'Exclude from any held-out evaluation of work planned, tuned or trained on this source.'}
    rows = [row for row in receipt.get('files', []) if row.get('source_member') == declared.get('images', {}).get('ct')]
    if len(rows) != 1: raise ValueError('SOURCE_ROLE: exact original CTA receipt required')
    row = rows[0]
    if row['sha256'] not in {ref.sha256 for ref in case.source_refs if ref.provenance == 'observed'}:
        raise ValueError('SOURCE_ROLE: case original does not join the registered patient')
    source = (root/row['path']).resolve()
    if not source.is_relative_to(root) or _sha(source) != row['sha256']:
        raise ValueError('SOURCE_ROLE: trusted original is unavailable or changed')
    image = nib.load(source)
    # Compare actual pixels/frame before trusting metadata. No reserved image is
    # opened: the trusted receipt's TRAIN role was checked before this read.
    affine = case.affine if case.frame == 'RAS+' else np.diag([-1., -1., 1., 1.])@case.affine
    if (case.mri.shape != image.shape or not np.array_equal(affine, image.affine)
            or not np.array_equal(case.mri, np.asarray(image.dataobj, dtype=np.float32))):
        raise ValueError('SOURCE_ROLE: native image or physical frame differs from the registered original')
    return {'status': 'verified_train_source', 'patientEvidence': True, 'family': family,
        'role': 'TRAIN', 'receiptSha256': _sha(path), 'originalSha256': row['sha256'],
        'imageArrayHash': array_digest(case.mri), 'affineHash': array_digest(affine)}


def source_role_label(scene, classification=None):
    """Train/evaluation role recorded with planning and learning outputs.

    A scene carries no trusted role receipt, so an acquired role is the case's
    own declaration (a claim, marked unverified) or UNKNOWN. This is metadata
    for later evaluation exclusion; no caller may use it to refuse a source.
    """
    if classification is None:
        from .surgical_planning_admission import source_classification
        classification = source_classification(scene)
    if classification == 'generated-engineering':
        return {'role': 'GENERATED_CONTROL', 'basis': 'generated-source', 'verified': True}
    adapter = scene.source.get('adapter') if isinstance(getattr(scene, 'source', None), dict) else None
    declared = adapter.get('sourceRole') if isinstance(adapter, dict) else None
    if type(declared) is str and declared:
        return {'role': declared, 'basis': 'case-declared-claim', 'verified': False}
    return {'role': 'UNKNOWN', 'basis': 'undeclared', 'verified': False}
