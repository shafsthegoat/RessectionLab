import numpy as np
import pytest
from scipy.ndimage import map_coordinates

from resectionlab.preprocessing import (
    PreprocessingError, RAS_TO_LPS, RegistrationConfig, histogram_nmi,
    register_t1_to_b0, sitk_transform_to_ras, to_sitk, validate_rigid,
)


def test_xyz_ras_image_round_trip_to_itk_lps_preserves_physical_landmark():
    sitk = pytest.importorskip("SimpleITK")
    angle = np.deg2rad(17)
    affine = np.array([[-2*np.cos(angle), -3*np.sin(angle), 0, 35],
                       [-2*np.sin(angle), 3*np.cos(angle), 0, -22],
                       [0, 0, 4, 13], [0, 0, 0, 1]], dtype=float)
    array = np.arange(5*6*7, dtype=float).reshape(5, 6, 7)
    image = to_sitk(array, affine)
    np.testing.assert_array_equal(sitk.GetArrayFromImage(image).transpose(2, 1, 0), array)
    index = np.array([2, 3, 4, 1])
    expected_lps = (RAS_TO_LPS @ affine @ index)[:3]
    np.testing.assert_allclose(image.TransformIndexToPhysicalPoint((2, 3, 4)), expected_lps)


def test_transform_export_includes_center_and_has_correct_point_direction():
    sitk = pytest.importorskip("SimpleITK")
    transform = sitk.Euler3DTransform()
    transform.SetCenter((12., -8., 5.))
    transform.SetRotation(0.04, -0.02, 0.09)
    transform.SetTranslation((2., 3., -4.))
    matrix = sitk_transform_to_ras(transform)
    validate_rigid(matrix)
    points = np.array([[4., 8., 12.], [-10., 3., -5.]])
    for point in points:
        lps = RAS_TO_LPS[:3, :3] @ point
        expected = RAS_TO_LPS[:3, :3] @ transform.TransformPoint(tuple(lps))
        actual = matrix[:3, :3] @ point + matrix[:3, 3]
        np.testing.assert_allclose(actual, expected, atol=1e-12)
    np.testing.assert_allclose(np.linalg.inv(matrix) @ matrix, np.eye(4), atol=1e-12)


@pytest.mark.parametrize("factor", [-1., 1.1])
def test_reflection_and_scale_are_rejected(factor):
    matrix = np.eye(4)
    matrix[0, 0] = factor
    with pytest.raises(PreprocessingError, match="NONRIGID_OR_REFLECTED_TRANSFORM"):
        validate_rigid(matrix)


def test_independent_mi_prefers_same_pairs_and_rejects_constant_signal():
    rng = np.random.default_rng(5)
    data = rng.random((12, 12, 12))
    mask = np.ones(data.shape, bool)
    assert histogram_nmi(data, data, mask) > histogram_nmi(data, rng.random(data.shape), mask)
    with pytest.raises(PreprocessingError, match="CONSTANT_REGISTRATION_IMAGE"):
        histogram_nmi(np.ones_like(data), np.ones_like(data), mask)
    with pytest.raises(PreprocessingError, match="QC_INTENSITY_RANGE_INVALID"):
        histogram_nmi(data - 3, data, mask)


def phantom():
    grid = np.indices((48, 48, 48), dtype=float)
    t1 = np.zeros((48, 48, 48))
    for center, radii, intensity in [((22, 23, 25), (9, 11, 12), 0.4),
                                    ((16, 18, 27), (4, 5, 6), 0.8),
                                    ((29, 28, 18), (6, 4, 5), 0.6),
                                    ((24, 14, 16), (4, 3, 4), 0.9)]:
        distance = sum(((grid[i] - center[i]) / radii[i]) ** 2 for i in range(3))
        t1 += intensity * np.exp(-distance / 2)
    affine = np.diag([2., 2., 2., 1.])
    affine[:3, 3] = -48
    translation = np.array([4., -3., 2.])
    b0 = map_coordinates(t1, grid + translation[:, None, None, None] / 2, order=1)
    return b0.astype(np.float32), t1.astype(np.float32), affine, translation


def test_registration_recovers_injected_physical_translation_and_improves_qc():
    pytest.importorskip("SimpleITK")
    b0, t1, affine, expected = phantom()
    result = register_t1_to_b0(b0, affine, t1, affine, b0 > 0.05,
                              config=RegistrationConfig(iterations_per_level=100, threads=1))
    matrix = np.asarray(result["report"]["DWI_RAS_mm_to_T1_RAS_mm"])
    np.testing.assert_allclose(matrix[:3, 3], expected, atol=0.6)
    qc = result["report"]["independent_qc"]
    assert qc["nmi_after"] > qc["nmi_before"]
    assert result["report"]["preprocessing_ready"] is False
    assert result["report"]["registration_reviewed"] is False
    assert result["report"]["clinical_deficit_probability"] is None


def test_cancel_and_invalid_mask_stop_without_exporting_a_transform():
    pytest.importorskip("SimpleITK")
    b0, t1, affine, _ = phantom()
    with pytest.raises(PreprocessingError, match="REGISTRATION_CANCELLED"):
        register_t1_to_b0(b0, affine, t1, affine, b0 > 0.05, cancelled=lambda: True)
    with pytest.raises(PreprocessingError, match="INVALID_REGISTRATION_MASK"):
        register_t1_to_b0(b0, affine, t1, affine, np.zeros_like(b0, dtype=bool))


def test_excessive_compute_budget_is_rejected():
    with pytest.raises(PreprocessingError, match="INVALID_REGISTRATION_BUDGET"):
        RegistrationConfig(threads=10).validate()
