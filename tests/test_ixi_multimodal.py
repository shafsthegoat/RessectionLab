"""Numerical controls for healthy-source assembly, independent of downloaded data."""
import importlib.util
from pathlib import Path

import nibabel as nib
import numpy as np
import pytest

from resectionlab.core import CaseData, SourceRef
from resectionlab.ixi_multimodal import diffusion_volumes, tensor_scalars, require_healthy, SUBJECT


def gradients():
    rng = np.random.default_rng(45)
    g = rng.normal(size=(16, 3))
    g /= np.linalg.norm(g, axis=1)[:, None]
    return np.r_[0., np.full(15, 1000.)], g


def acquisition(eigen=(.0003, .0005, .0015)):
    b, g = gradients()
    signal = 900 * np.exp(-b * np.sum(g*g*np.asarray(eigen), axis=1))
    return np.broadcast_to(signal, (3, 4, 5, 16)).copy(), b, g


def test_tensor_recovers_known_anisotropy_and_is_orthogonal_frame_invariant():
    signal, b, g = acquisition()
    mask = np.ones(signal.shape[:3], bool)
    mask[0] = False
    actual = tensor_scalars(signal, b, g, mask, chunk_voxels=7)
    eigen = np.array([.0003, .0005, .0015])
    expected = np.sqrt(1.5 * np.sum((eigen-eigen.mean())**2) / np.sum(eigen**2))
    np.testing.assert_allclose(actual['fa'][mask], expected, rtol=1e-5)
    np.testing.assert_allclose(actual['md'][mask], eigen.mean(), rtol=1e-5)
    assert not actual['fit_coverage'][~mask].any()
    assert actual['normalized_rmse'][mask].max() < 1e-10
    rotation, _ = np.linalg.qr(np.random.default_rng(7).normal(size=(3, 3)))
    reflected = tensor_scalars(signal, b, (g @ rotation) * [-1, 1, 1], mask)
    np.testing.assert_allclose(actual['fa'], reflected['fa'], atol=1e-6)
    assert actual['report']['usable_for_tract_aware_planning'] is False


def test_negative_tensor_eigenvalues_are_unknown_instead_of_clipped():
    signal, b, g = acquisition(eigen=(-.0001, .0005, .0015))
    fit = tensor_scalars(signal, b, g, np.ones(signal.shape[:3], bool))
    assert not fit['fit_coverage'].any()
    assert fit['report']['excluded_nonpositive_tensor_voxels'] == 60


def test_diffusion_rejects_mismatched_physical_grid(tmp_path):
    signal, b, g = acquisition()
    for i in range(16):
        affine = np.eye(4)
        if i == 9:
            affine[0, 3] = 4
        image = nib.Nifti1Image(signal[..., i], affine)
        image.header.set_xyzt_units('mm')
        nib.save(image, tmp_path/f'{SUBJECT}-DTI-{i:02d}.nii.gz')
    np.savetxt(tmp_path/'bvals.txt', b)
    np.savetxt(tmp_path/'bvecs.txt', g.T)
    with pytest.raises(ValueError, match='different physical grids'):
        diffusion_volumes(tmp_path)


def test_healthy_case_never_accepts_tumor_or_removal_target():
    source = SourceRef('structural', 'generated://healthy-control', provenance='simulated')
    case = CaseData('healthy', mri=np.ones((3, 3, 3)), compartments={'brain': np.ones((3, 3, 3), bool)}, affine=np.eye(4), source_refs=(source,))
    require_healthy(case)
    lesion = CaseData('bad', mri=case.mri, compartments={'tumor': np.ones(case.mri.shape, bool)}, affine=np.eye(4), source_refs=(source,))
    with pytest.raises(ValueError, match='must not contain'):
        require_healthy(lesion)


def test_acquisition_selects_only_exact_subject_and_direction_members():
    spec = importlib.util.spec_from_file_location('acquire_ixi', Path(__file__).parents[1]/'scripts/acquire_ixi265_diffusion.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    assert module.selected('./IXI265-Guys-0845-DTI-00.nii.gz')
    assert module.selected('./IXI265-Guys-0845-DTI-15.nii.gz')
    assert not module.selected('./IXI266-Guys-0845-DTI-00.nii.gz')
    assert not module.selected('./IXI265-HH-0845-DTI-00.nii.gz')
    assert not module.selected('./IXI265-Guys-0845-DTI-16.nii.gz')


def test_simulator_retains_angiographic_annotation_origin_without_promoting_review():
    from resectionlab.surgical_patient import _anatomy_metadata
    row = {'provenance': {'source_modality': 'MRA', 'method': 'Published partial annotation'},
           'review_status': 'unreviewed', 'coverage': 'Unlabeled cells unknown'}
    metadata = _anatomy_metadata('vessels', row)
    assert 'Angiography-derived' in metadata['coverage']
    assert 'structural-MRI' not in metadata['coverage']
    assert metadata['review_status'] == 'unreviewed'
    assert metadata['source_provenance'] == row['provenance']
