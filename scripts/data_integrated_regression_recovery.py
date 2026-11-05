"""Read-only authority validator for a proved pre-load one-pass interruption.

This module never constructs a bridge or resets a decision. The separately
reviewed external executor owns the single continuation claim and original caps.
"""
from __future__ import annotations
import hashlib
import json
from pathlib import Path


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def inventory(root, directory):
    """Every regular file matters; only OS AppleDouble sidecars are excluded."""
    rows = []
    for path in sorted(directory.rglob('*')):
        if path.is_symlink():
            raise ValueError('Symlink in interrupted evidence')
        if path.is_file() and not path.name.startswith('._'):
            rows.append({'path': str(path.relative_to(root)), 'bytes': path.stat().st_size,
                         'sha256': sha256(path)})
    return rows


def validate_authority(authority, expected):
    if authority != expected or authority.get('physical_execution_approved') is not True:
        raise ValueError('Exact independent recovery authority required')


def validate_interruption(root, directory, audit, claim, *, original_driver_sha256,
                          original_decision_sha256, source_sha256, mri_sha256, mri_bytes):
    """Prove no create/preview/step from immutable pre-load transfer evidence.

    This proof is valid only for the exact independently reviewed original
    executor/backend: loadCase emits MRI, then all compartment transfers before
    returning, with no intermediate pruning. The executor never calls create
    until loadCase returns. The recovery reviewer must approve this control flow.
    """
    root, directory = Path(root), Path(directory)
    relative = str(directory.relative_to(root))
    expected_claim = root / 'integrated-v1/regressions/decision-use' / (original_decision_sha256 + '.json')
    if Path(claim) != expected_claim or not claim.is_file():
        raise ValueError('Exact consumed original decision claim required')
    if audit.get('schema') != 'interrupted-one-pass-execution-read-only-audit-v1':
        raise ValueError('Unsupported interruption audit')
    if (audit.get('executor_sha256') != original_driver_sha256
            or audit.get('source_sha256') != source_sha256
            or audit.get('physical_commits') != 0 or type(audit.get('physical_commits')) is not int
            or audit.get('public_previews') != 0 or type(audit.get('public_previews')) is not int
            or audit.get('original_claim_remains_consumed') is not True
            or audit.get('source_mri_transfer_sha256') != mri_sha256
            or audit.get('consumed_claim') != {'path': str(claim.relative_to(root)), 'sha256': sha256(claim)}):
        raise ValueError('Interruption/source/original claim proof differs')
    rows = inventory(root, directory)
    if rows != audit.get('evidence_files'):
        raise ValueError('Interrupted evidence inventory changed')
    names = [row['path'][len(relative) + 1:] for row in rows]
    bins = [row for row in rows if row['path'].endswith('.bin')]
    transfer_names = [name for name in names if name.startswith('transfers/bridge-')
                      and name.endswith('/' + mri_sha256 + '.bin') and len(Path(name).parts) == 3]
    if (len(rows) != 3 or len(transfer_names) != 1
            or set(names) != {'reproduction.json', 'runs/.bridge-integrity-key', *transfer_names}
            or len(bins) != 1 or bins[0]['sha256'] != mri_sha256 or bins[0]['bytes'] != mri_bytes):
        raise ValueError('Only complete MRI transfer and pre-load bookkeeping allowed')
    reproduction = json.loads((directory / 'reproduction.json').read_text())
    original_claim = json.loads(claim.read_text())
    if (original_claim != {'output': relative, **reproduction}
            or reproduction['driver_sha256'] != original_driver_sha256
            or reproduction['decision_sha256'] != original_decision_sha256
            or reproduction['source_sha256'] != source_sha256):
        raise ValueError('Original claim/reproduction differs')
    return {'original_output': relative, 'original_claim_sha256': sha256(claim),
            'zero_commits': True, 'zero_public_previews': True}
