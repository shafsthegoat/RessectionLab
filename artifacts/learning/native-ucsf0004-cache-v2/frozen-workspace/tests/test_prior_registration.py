from pathlib import Path

import nibabel as nib
import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from resectionlab.prior_registration import (FRAME_FLIP, RegistrationSettings, image_from_ras,
    linear_transform_matrix, register_template_to_patient, template_to_patient_ras)

sitk = pytest.importorskip("SimpleITK")


def test_array_bridge_preserves_oblique_anisotropic_physical_landmarks():
    data = np.arange(7 * 8 * 9, dtype=np.float32).reshape(7, 8, 9)
    affine = np.eye(4)
    affine[:3, :3] = Rotation.from_euler("z", 21, degrees=True).as_matrix() @ np.diag([1.3, 2.1, 3.2])
    affine[:3, 3] = [81, -124, -50]
    image = image_from_ras(data, affine)
    for index in [(0, 0, 0), (2, 4, 7), (6, 7, 8)]:
        ras = affine @ [*index, 1]
        np.testing.assert_allclose(image.TransformIndexToPhysicalPoint(index), (FRAME_FLIP @ ras)[:3])
        assert image[index] == data[index]


def test_transform_export_inverts_fixed_to_moving_and_respects_centers():
    transform = sitk.Euler3DTransform()
    transform.SetCenter((16., 33., -20.))
    transform.SetRotation(.1, -.05, .2)
    transform.SetTranslation((7., -9., 11.))
    matrix = linear_transform_matrix(transform)
    forward_ras = template_to_patient_ras(transform)
    for fixed_ras in [[10, 20, -8, 1], [-30, 40, 35, 1]]:
        moving_lps = transform.TransformPoint(tuple((FRAME_FLIP @ fixed_ras)[:3]))
        moving_ras = FRAME_FLIP @ [*moving_lps, 1]
        np.testing.assert_allclose(forward_ras @ moving_ras, fixed_ras, atol=1e-10)
        np.testing.assert_allclose(matrix @ (FRAME_FLIP @ fixed_ras), [*moving_lps, 1], atol=1e-10)


def test_reflection_shear_and_preload_cancellation_reject():
    reflection = sitk.AffineTransform(3)
    reflection.SetMatrix(tuple(np.diag([-1., 1., 1.]).ravel()))
    with pytest.raises(ValueError, match="reflected"):
        linear_transform_matrix(reflection)
    shear = np.eye(4)
    shear[0, 1] = .2
    with pytest.raises(ValueError, match="Sheared"):
        image_from_ras(np.zeros((3, 3, 3)), shear)
    with pytest.raises(InterruptedError):
        register_template_to_patient("missing.nii.gz", "missing.nii.gz", case_hash="test", cancelled=lambda: True)


@pytest.mark.parametrize("budget", [0, -1, float("inf"), float("nan")])
def test_unbounded_or_invalid_registration_budget_rejects(budget):
    with pytest.raises(ValueError, match="bounded"):
        RegistrationSettings(max_seconds_per_candidate=budget)


def _save(path: Path, data, affine):
    image = nib.Nifti1Image(data.astype(np.float32), affine)
    image.header.set_xyzt_units("mm")
    image.set_qform(affine, 1)
    image.set_sform(affine, 1)
    nib.save(image, path)


def test_actual_registration_recovers_known_physical_translation_but_keeps_review_gate(tmp_path):
    grid = np.indices((32, 36, 30))
    a = ((grid[0] - 15) / 11)**2 + ((grid[1] - 17) / 14)**2 + ((grid[2] - 14) / 10)**2
    data = np.where(a < 1, (1 - a) + .5 * np.exp(-((grid[0]-10)**2 + (grid[1]-21)**2 + (grid[2]-11)**2)/20), 0)
    template_affine = np.diag([1.5, 1.5, 1.5, 1.])
    patient_affine = template_affine.copy()
    patient_affine[:3, 3] = [7, -4, 3]
    patient, template = tmp_path / "patient.nii.gz", tmp_path / "template.nii.gz"
    _save(patient, data, patient_affine)
    _save(template, data, template_affine)
    report = register_template_to_patient(patient, template, case_hash="fixture",
        settings=RegistrationSettings(rigid_start_angles_deg=(0.,), iterations_per_level=20,
                                      sampling_fraction=1., affine_refinement=False, threads=1,
                                      max_seconds_per_candidate=10))
    candidate = report["candidates"][0]
    assert candidate["status"] == "candidate_requires_alignment_review"
    estimated = np.asarray(candidate["mni_ras_to_patient_ras_mm"])
    template_point = np.array([22.5, 25.5, 21, 1])
    expected = template_point + [7, -4, 3, 0]
    np.testing.assert_allclose(estimated @ template_point, expected, atol=1.5)
    assert candidate["brain_support_dice"] > .90
    assert candidate["independent_landmark_error_mm"] is None
    assert report["qc_state"] == "alignment_review_required"
    assert report["clinical_deficit_probability"] is None
