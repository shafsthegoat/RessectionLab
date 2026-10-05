import numpy as np
import pytest

from resectionlab.app.annotation import edit_sphere


def test_physical_brush_is_anisotropic_in_voxel_indices():
    original = np.zeros((11, 11, 11), dtype=bool)
    affine = np.diag([1., 2., 3., 1.])
    result = edit_sphere(original, affine, [5, 10, 15], 2., add=True)
    assert result[3, 5, 5] and result[5, 4, 5]
    assert not result[5, 5, 4]  # three millimeters away in z
    assert not original.any()


def test_brush_uses_transformed_physical_center():
    affine = np.array([[0, -2, 0, 20], [1, 0, 0, -4], [0, 0, 3, 8], [0, 0, 0, 1]], dtype=float)
    center = affine[:3, :3] @ [5, 5, 5] + affine[:3, 3]
    result = edit_sphere(np.ones((11, 11, 11), bool), affine, center, .4, add=False)
    assert result.sum() == 1330 and not result[5, 5, 5]


def test_outside_brush_does_not_wrap_negative_indices():
    original = np.zeros((11, 11, 11), bool)
    assert not edit_sphere(original, np.eye(4), [-100, 0, 0], 1., add=True).any()
    with pytest.raises(ValueError):
        edit_sphere(original, np.eye(4), [0, 0, 0], -1, add=True)
