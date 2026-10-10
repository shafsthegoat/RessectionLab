"""Import one reviewed saved diagnostic for an existing auxiliary image.

This decodes saved state/coverage only; it never invokes a model, updates the
planning case, or turns the separate unreviewed output into an annotation.
"""
from __future__ import annotations
import gzip
import hashlib
import io
import json
import math
from pathlib import Path
import numpy as np

# Exact saved decoder descriptor accepted by the independent numeric/frame audit.
# This authenticates that saved unreviewed output, not anatomical/clinical quality.
CASE4_DESCRIPTOR_SHA256 = 'd5d2ec1e8d2f99ec1842d4f4d1509986158075fea6cbb56d62718f1a2f8a18b4'
MAX_VOXELS = 64 * 1024**2
MAX_JSON_BYTES = 64 * 1024
HEADER_ALLOWANCE = 1024**2


def _bytes(path: Path, limit: int) -> bytes:
    if path.is_symlink() or not path.is_file():
        raise ValueError('Diagnostic file must be a regular local file')
    with path.open('rb') as stream:
        data = stream.read(limit + 1)
    if len(data) > limit:
        raise ValueError('Diagnostic file exceeds the bounded display limit')
    return data


def _nifti(path, expected, dtype, shape, affine):
    import nibabel as nib
    compressed = _bytes(path, math.prod(shape) + HEADER_ALLOWANCE)
    if hashlib.sha256(compressed).hexdigest() != expected:
        raise ValueError('Saved diagnostic file checksum changed')
    with gzip.GzipFile(fileobj=io.BytesIO(compressed)) as stream:
        raw = stream.read(math.prod(shape) + HEADER_ALLOWANCE + 1)
    if len(raw) > math.prod(shape) + HEADER_ALLOWANCE:
        raise ValueError('Diagnostic expansion exceeds its declared grid')
    header = nib.Nifti1Header.from_fileobj(io.BytesIO(raw), check=True)
    offset = float(header['vox_offset'])
    slope, intercept = float(header['scl_slope']), float(header['scl_inter'])
    if (header.get_data_shape() != tuple(shape) or header.get_data_dtype() != np.dtype(dtype)
            or not math.isfinite(offset) or offset < 352 or offset > HEADER_ALLOWANCE
            or offset != int(offset) or len(raw) != int(offset) + math.prod(shape)
            or (math.isfinite(slope) and slope != 1) or (math.isfinite(intercept) and intercept != 0)
            or header.get_xyzt_units()[0] != 'mm' or int(header['sform_code']) <= 0):
        raise ValueError('Diagnostic NIfTI shape, dtype, units or scaling changed')
    frame = np.asarray(header.get_sform(), dtype=float)
    if frame.shape != (4, 4) or not np.isfinite(frame).all() or not np.allclose(frame, affine, atol=1e-5, rtol=0):
        raise ValueError('Diagnostic authoritative sform differs from selected image')
    image = nib.Nifti1Image.from_bytes(raw)
    array = np.ascontiguousarray(np.asanyarray(image.dataobj), dtype=dtype)
    return array, {'shape': list(shape), 'affineRAS': frame.tolist(), 'sformCode': int(header['sform_code'])}


def load_saved_diagnostic(path: Path, expected_sha256: str, source, check=lambda: None):
    raw = _bytes(Path(path), MAX_JSON_BYTES)
    if hashlib.sha256(raw).hexdigest() != expected_sha256:
        raise ValueError('This is not the reviewed saved diagnostic descriptor')
    d = json.loads(raw)
    check()
    shape = d['shape_xyz']
    refs = [r for r in source.source_refs if r.source_id == 'structural' and r.provenance == 'observed']
    if (len(refs) != 1 or refs[0].sha256 != d['source_sha256']['t1c']
            or d['schema'] != 'case4_unreviewed_diagnostic_display_layer_v1'
            or d['scope'] != 'display_only_atlas_native_grid'
            or d['model_output_verified'] is not True
            or d['anatomical_qc'] != 'unreviewed_inferior_mask_omission'
            or d['training_overlap_status'] != 'unknown'
            or d['same_grid_only'] is not True
            or any(d[key] is not False for key in ('planning_eligible','evaluation_eligible','clinical_evidence'))
            or not isinstance(shape, list) or len(shape) != 3
            or any(type(n) is not int or n <= 0 or n > 4096 for n in shape)
            or math.prod(shape) > MAX_VOXELS or tuple(shape) != source.mri.shape):
        raise ValueError('Diagnostic is not bound to the selected auxiliary prepared T1c')
    affine = np.asarray(d['affine_ras_mm'], dtype=float)
    expected_affine = (np.diag([-1.,-1.,1.,1.]) @ source.affine if source.frame == 'LPS+' else source.affine)
    if source.frame not in {'RAS+','LPS+'} or affine.shape != (4,4) or not np.isfinite(affine).all() or not np.allclose(affine, expected_affine, atol=1e-5, rtol=0):
        raise ValueError('Diagnostic and selected auxiliary source frames disagree')
    arrays, grids = {}, {}
    for name, dtype in (('state','int8'),('coverage','uint8')):
        output = d['outputs'][name]
        if output['path'] != name+'.nii.gz' or output['dtype'] != dtype:
            raise ValueError('Only the exact colocated saved state and coverage are accepted')
        arrays[name], grids[name] = _nifti(Path(path).parent/output['path'], output['sha256'], dtype, shape, affine)
        check()
    state, coverage = arrays['state'], arrays['coverage']
    if (not np.isin(coverage, (0,1)).all() or not np.isin(state, (-1,0,1)).all()
            or not np.array_equal(state == -1, coverage == 0)
            or type(d['predicted_voxels']) is not int or int(coverage.sum()) != d['predicted_voxels']
            or type(d['unknown_voxels']) is not int or int((state == -1).sum()) != d['unknown_voxels']):
        raise ValueError('Diagnostic unknown/covered states or counts disagree')
    return d, arrays, grids


def import_diagnostic_layer(session, args, request, progress):
    from .desktop_bridge import BridgeError, _keys, _path
    _keys(args, {'caseHash','seriesId','descriptorPath'})
    if CASE4_DESCRIPTOR_SHA256 is None:
        raise BridgeError('DIAGNOSTIC_NOT_PUBLISHED','No reviewed saved diagnostic has been published for display.')
    entry = session._get_case(args.get('caseHash'))
    series_id = args.get('seriesId')
    source = entry.display_sources.get(series_id) if isinstance(series_id, str) else None
    if source is None:
        raise BridgeError('DIAGNOSTIC_SOURCE_REQUIRED','Select the prepared T1c in the imaging workspace first.')
    progress(.1,'Checking a saved diagnostic against the selected image')
    descriptor, arrays, grids = load_saved_diagnostic(
        _path(args.get('descriptorPath'), kind='json'), CASE4_DESCRIPTOR_SHA256, source.case, request.check)
    request.check()
    if session._get_case(args.get('caseHash')) is not entry or entry.display_sources.get(series_id) is not source:
        raise BridgeError('STALE_CASE','The selected diagnostic image changed during import.')
    result = {'caseHash': entry.case.semantic_hash, 'seriesId': series_id,
              'descriptor': descriptor, 'sourceSha256': descriptor['source_sha256']['t1c'],
              'stateGrid': grids['state'], 'coverageGrid': grids['coverage'],
              'loadedStateSha256': descriptor['outputs']['state']['sha256'],
              'loadedCoverageSha256': descriptor['outputs']['coverage']['sha256'],
              'state': session.transfers.array(arrays['state'], 'int8'),
              'coverage': session.transfers.array(arrays['coverage'], 'uint8')}
    # Keep the latest two snapshots alive across the runtime's post-request prune.
    # Cancellation before commit preserves the previous result; uncommitted new
    # transfers are discarded by the runtime's existing finally cleanup.
    request.begin_commit()
    entry.diagnostic_transfer_paths = (result['state']['path'], result['coverage']['path'])
    return result
