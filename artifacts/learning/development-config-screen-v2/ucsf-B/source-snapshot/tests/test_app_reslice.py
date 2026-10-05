"""Analytic landmarks check the MRI viewer independently of its renderer."""

import numpy as np
import pytest

from resectionlab.app.reslice import (
    PLANE_AXES, ras_affine, sample_slice, slice_geometry, window_to_rgb, world_bounds,
)


def test_anisotropic_translated_ramp_retains_physical_location():
    shape = (17, 21, 13)
    affine = np.diag([2., 3., 4., 1.])
    affine[:3, 3] = [-21., 13., 7.]
    x, y, z = np.indices(shape)
    volume = 2 * x + 3 * y + 4 * z
    cursor = affine[:3, :3] @ np.array([8., 10., 6.]) + affine[:3, 3]
    for plane in PLANE_AXES:
        geometry = slice_geometry(shape, affine, cursor, plane, max_pixels=61)
        values = sample_slice(volume.astype(float), affine, geometry)
        middle = geometry.world_at(.5, .5)
        assert np.allclose(middle, cursor)
        assert values[geometry.height // 2, geometry.width // 2] == pytest.approx(70, abs=2)


def test_oblique_physical_ramp_is_sampled_in_world_plane():
    angle = np.deg2rad(21)
    affine = np.array([[np.cos(angle), -np.sin(angle), 0, -20],
                       [np.sin(angle), np.cos(angle), 0, 4], [0, 0, 2., -13], [0, 0, 0, 1.]])
    shape = (31, 31, 15)
    indices = np.indices(shape).reshape(3, -1)
    world = affine[:3, :3] @ indices + affine[:3, 3, None]
    volume = (world[0] + 2 * world[1] + 3 * world[2]).reshape(shape)
    cursor = affine[:3, :3] @ np.array([15, 15, 7]) + affine[:3, 3]
    geometry = slice_geometry(shape, affine, cursor, "axial", max_pixels=101)
    values = sample_slice(volume, affine, geometry)
    assert values[50, 50] == pytest.approx(cursor @ [1, 2, 3], abs=1e-6)


def test_lps_affine_flips_only_patient_x_and_y():
    lps = np.diag([2., 3., 4., 1.])
    lps[:3, 3] = [10, 20, 30]
    assert np.array_equal(ras_affine(lps, "LPS+"), np.diag([-1, -1, 1, 1]) @ lps)
    with pytest.raises(ValueError, match="Unsupported coordinate frame"):
        ras_affine(lps, "scanner_unknown")


def test_mask_reslice_preserves_discrete_labels():
    data = np.zeros((11, 11, 11), dtype=np.uint8)
    data[5, 5, 5] = 4
    geometry = slice_geometry(data.shape, np.eye(4), [5, 5, 5], "coronal", max_pixels=11)
    image = sample_slice(data, np.eye(4), geometry, order=0)
    assert set(np.unique(image)) == {0, 4}
    assert image[5, 5] == 4


def test_windowing_is_bounded_and_handles_constant_image():
    assert np.array_equal(window_to_rgb(np.array([[-10., 50., 1000.]]), 0, 100)[0, :, 0], [0, 127, 255])
    assert window_to_rgb(np.zeros((2, 2)), 0, 0).shape == (2, 2, 3)


def test_world_bounds_use_extreme_voxel_centers():
    affine = np.diag([-2., 3., 4., 1.])
    assert np.array_equal(world_bounds((3, 5, 2), affine), [[-4, 0, 0], [0, 12, 4]])
