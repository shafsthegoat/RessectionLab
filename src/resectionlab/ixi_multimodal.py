"""Small healthy-subject IXI assembly helpers; no synthetic lesion or anatomy.

Tensor scalars use the published gradient table without claiming a resolved
physical gradient frame. FA/MD are invariant to a common orthogonal frame change;
directional tracks and functional planning are deliberately not produced.
"""
from pathlib import Path
import nibabel as nib
import numpy as np

from .diffusion import validate_gradients
from .patient_anatomy import file_hash

SUBJECT = 'IXI265-Guys-0845'


def diffusion_volumes(directory):
    root = Path(directory)
    paths = [root / f'{SUBJECT}-DTI-{i:02d}.nii.gz' for i in range(16)]
    images = [nib.load(p) for p in paths]
    reference = images[0]
    if reference.ndim != 3 or reference.header.get_xyzt_units()[0] != 'mm':
        raise ValueError('Diffusion requires native 3D millimeter source volumes')
    for image in images:
        if image.shape != reference.shape or not np.allclose(image.affine, reference.affine, rtol=0, atol=1e-5):
            raise ValueError('Diffusion volumes have different physical grids')
    signal = np.stack([im.get_fdata(dtype=np.float32) for im in images], axis=-1)
    if not np.isfinite(signal).all() or (signal < 0).any():
        raise ValueError('Invalid diffusion intensities')
    bvals = np.loadtxt(root / 'bvals.txt').reshape(-1)
    bvecs = np.loadtxt(root / 'bvecs.txt').T
    validate_gradients(bvals, bvecs, len(paths))
    return reference, signal, bvals, bvecs, [
        {'name': p.name, 'sha256': file_hash(p), 'shape': list(im.shape),
         'affine_ras_mm': im.affine.tolist(), 'qform_code': int(im.header['qform_code']),
         'sform_code': int(im.header['sform_code'])} for p, im in zip(paths, images, strict=True)]


def tensor_scalars(signal, bvals, bvecs, mask, chunk_voxels=4096):
    """Bounded log-linear OLS tensor diagnostic with explicit invalid-fit coverage.

    The first published b0 vector is unit length, permitted by validate_gradients.
    Signals are floored only for the log fit; original measurements stay intact.
    """
    signal = np.asarray(signal)
    mask = np.asarray(mask)
    validate_gradients(np.asarray(bvals), np.asarray(bvecs), signal.shape[-1])
    if signal.ndim != 4 or mask.shape != signal.shape[:3] or mask.dtype != bool or not mask.any():
        raise ValueError('Nonempty native-grid boolean diffusion mask required')
    if not np.isfinite(signal).all() or (signal < 0).any() or chunk_voxels < 1:
        raise ValueError('Finite nonnegative diffusion and positive chunk size required')
    g = np.asarray(bvecs, float).copy()
    weighted = np.asarray(bvals) > 50
    g[weighted] /= np.linalg.norm(g[weighted], axis=1)[:, None]
    gx, gy, gz = g.T
    design = np.column_stack((np.ones(len(g)), -np.asarray(bvals)[:, None] *
                             np.column_stack((gx*gx, gy*gy, gz*gz, 2*gx*gy, 2*gx*gz, 2*gy*gz))))
    if np.linalg.matrix_rank(design) != 7:
        raise ValueError('Tensor with intercept requires full-rank gradients')
    inverse = np.linalg.pinv(design)
    fa, md, residual = [np.zeros(mask.shape, np.float32) for _ in range(3)]
    valid = np.zeros(mask.shape, bool)
    ids = np.flatnonzero(mask)
    flat = signal.reshape(-1, signal.shape[-1])
    for offset in range(0, len(ids), chunk_voxels):
        cells = ids[offset:offset+chunk_voxels]
        measured = flat[cells].astype(float)
        baseline = measured[:, ~weighted].mean(1)
        if (baseline <= 0).any():
            raise ValueError('Nonpositive diffusion baseline within fitted mask')
        beta = np.log(np.maximum(measured, 1.)) @ inverse.T
        d = beta[:, 1:]
        tensors = np.empty((len(cells), 3, 3))
        tensors[:, 0, 0], tensors[:, 1, 1], tensors[:, 2, 2] = d[:, 0], d[:, 1], d[:, 2]
        tensors[:, 0, 1] = tensors[:, 1, 0] = d[:, 3]
        tensors[:, 0, 2] = tensors[:, 2, 0] = d[:, 4]
        tensors[:, 1, 2] = tensors[:, 2, 1] = d[:, 5]
        eigen = np.linalg.eigvalsh(tensors)
        mean = eigen.mean(1)
        ok = (eigen > 0).all(1) & np.isfinite(eigen).all(1)
        # Negative eigenvalues are excluded, never silently clipped to anatomy.
        fraction = np.sqrt(1.5 * np.sum((eigen - mean[:, None])**2, axis=1) /
                           np.maximum(np.sum(eigen**2, axis=1), 1e-30))
        fa.flat[cells[ok]] = fraction[ok]
        md.flat[cells[ok]] = mean[ok]
        residual.flat[cells] = np.sqrt(np.mean((np.exp(beta @ design.T) - measured)**2, axis=1)) / baseline
        valid.flat[cells[ok]] = True
    return {'fa': fa, 'md': md, 'normalized_rmse': residual, 'fit_coverage': valid,
            'report': {'method': 'NumPy log-linear ordinary least squares diffusion tensor',
                       'requested_voxels': len(ids), 'positive_definite_fit_voxels': int(valid.sum()),
                       'excluded_nonpositive_tensor_voxels': int(mask.sum()-valid.sum()),
                       'normalized_signal_rmse_median': float(np.median(residual[mask])),
                       'fa_median': float(np.median(fa[valid])) if valid.any() else None,
                       'gradient_normalization': 'weighted vectors normalized after <=0.01 source-norm tolerance check',
                       'gradient_physical_frame': 'unresolved; only rotation-invariant scalars exported',
                       'motion_corrected': False, 'distortion_corrected': False,
                       'usable_for_tract_aware_planning': False,
                       'tractography': 'not generated; source gradient frame and preprocessing unreviewed',
                       'functional_coverage': 'unknown'}}


def require_healthy(case):
    """Fail closed before simulator export if a lesion/target slipped in."""
    from .core import thaw_json
    metadata = thaw_json(case.metadata)
    if 'tumor' in case.compartments or any(
        row.get('kind') == 'tumor' or row.get('role') == 'target'
        for row in metadata.get('anatomy_layers', {}).values()
    ):
        raise ValueError('Healthy IXI case must not contain a tumor or target')
