"""Versioned, byte-preserving NIfTI header records; no voxel or frame repair."""
from __future__ import annotations

import base64
import hashlib
import math

from .structural_evidence import _hash

SCHEMA = 'resectionlab.nifti1-header-record/2'


def _encoded_float(value):
    """Keep uninterpreted nonfinite storage JSON-safe; raw bytes remain exact."""
    value = float(value)
    return value if math.isfinite(value) else 'nan' if math.isnan(value) else '+inf' if value > 0 else '-inf'


def nifti1_header_record_coded_v2(raw: bytes, source_file_sha256: str) -> dict:
    """Record inactive quaternion encoding without inventing an active qform.

    qform_code zero makes the quaternion inactive, including its stored qfac.
    Its bytes and values are retained without asking nibabel to construct that
    transform. Active qforms retain nibabel's normal validation. This recorder
    does not establish valid geometry, identity grids or anatomical admission.
    """
    import nibabel as nib

    if not isinstance(raw, bytes) or len(raw) != 348:
        raise ValueError('Header recorder v2 requires exactly 348 original bytes')
    header = nib.Nifti1Header(binaryblock=raw, check=False)
    if int(header['sizeof_hdr']) != 348 or bytes(header['magic']) != b'n+1\0':
        raise ValueError('Header recorder v2 requires single-file NIfTI-1')
    qcode, scode = int(header['qform_code']), int(header['sform_code'])
    if qcode < 0 or scode < 0:
        raise ValueError('Negative coded transform is unsupported')
    qform = header.get_qform() if qcode else None
    summary = {
        'shape': list(header.get_data_shape()), 'spatial_units': header.get_xyzt_units()[0],
        'xyzt_units_code': int(header['xyzt_units']),
        'pixdim': [_encoded_float(x) for x in header['pixdim']],
        'qform_code': qcode, 'sform_code': scode,
        'qform_status': 'active_decoded' if qcode else 'inactive_not_decoded',
        'qform_numeric': None if qform is None else qform.tolist(),
        'quaternion_encoding': {k: _encoded_float(header[k]) for k in
                                ('quatern_b', 'quatern_c', 'quatern_d', 'qoffset_x', 'qoffset_y', 'qoffset_z')},
        'sform_numeric': [[_encoded_float(v) for v in row] for row in header.get_sform()],
    }
    return {'schema': SCHEMA, 'source_file_sha256': _hash(source_file_sha256, 'header source file'),
            'header_base64': base64.b64encode(raw).decode('ascii'),
            'header_sha256': hashlib.sha256(raw).hexdigest(), 'raw_grid': summary}
