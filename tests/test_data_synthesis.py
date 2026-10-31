import numpy as np
import pytest

from resectionlab.data_synthesis import appearance, geometry, mild_deformation


def host():
    grid = np.indices((64, 64, 64)).astype(float)
    brain = np.sum((grid - 31.5) ** 2, axis=0) < 29**2
    image = (100 + grid[0] * 2 + grid[1]) * brain
    angle = .3
    affine = np.array([[np.cos(angle), -np.sin(angle), 0, -30],
                       [np.sin(angle), np.cos(angle), 0, -30], [0, 0, 1, -30], [0, 0, 0, 1]])
    return image.astype(np.float32), brain, affine


def test_reproducible_physical_compartments_and_distinct_generated_observation():
    image, brain, affine = host()
    masks, recipe = geometry(brain, affine, seed=17)
    other, repeated = geometry(brain, affine, seed=17)
    assert recipe == repeated
    for name, mask in masks.items():
        assert np.array_equal(mask, other[name]) and mask.any()
        assert not np.any(mask & ~brain)
    generated, record = appearance(image, brain, masks, seed=17)
    assert not np.array_equal(generated, image)
    assert np.isfinite(generated).all() and record['modified_voxels'] > 0
    assert not record['clinical_appearance_validated']
    with pytest.raises(ValueError, match='supported acquired'):
        appearance(image, brain, masks, seed=17, modality='T1CE')


def test_one_invertible_physical_deformation_moves_all_compartments_with_fixed_boundary():
    image, brain, affine = host()
    masks, recipe = geometry(brain, affine, seed=17)
    moved, support, labels, field, report = mild_deformation(image, brain, affine, masks,
        center_ras_mm=recipe['center_ras_mm'], amplitude_mm=1.)
    assert .2 < report['jacobian_min'] < 1 < report['jacobian_max'] < 5
    assert report['inverse_residual_max_mm'] < .5
    assert report['fixed_boundary_max_mm'] == 0
    assert np.isfinite(moved).all() and field.shape == (3, *brain.shape)
    assert any(not np.array_equal(labels[name], mask) for name, mask in masks.items())
    assert all(not np.any(mask & ~support) for mask in labels.values())
