"""Generated boundary/transform controls only; never opens Case4 arrays."""

import tempfile
import unittest
from pathlib import Path

import ants
import nibabel as nib
import numpy as np
import SimpleITK as sitk

from replay import independent_nearest, target_to_source_voxel


class NearestBoundaryTest(unittest.TestCase):
    def test_oblique_frame_and_nonidentity_itk_translation(self):
        with tempfile.TemporaryDirectory() as directory:
            directory = Path(directory)
            source_path = directory / "source.nii.gz"
            fixed_path = directory / "target.nii.gz"
            transform_path = directory / "translation.mat"
            angle = np.deg2rad(19.)
            rotation = np.array([[np.cos(angle), -np.sin(angle), 0.],
                                 [np.sin(angle), np.cos(angle), 0.],
                                 [0., 0., 1.]])
            source_affine = np.eye(4)
            source_affine[:3, :3] = rotation @ np.diag([-.8, -1.1, 1.4])
            source_affine[:3, 3] = (12., -4., 3.)
            translation_lps = np.array([1.2, -.7, .3])
            translation = sitk.AffineTransform(3)
            translation.SetTranslation(tuple(translation_lps))
            sitk.WriteTransform(translation, str(transform_path))
            target_affine = source_affine.copy()
            target_affine[:3, 3] += (source_affine[:3, :3] @ np.array([-.25, 0., 0.])
                                      - np.diag([-1., -1., 1.]) @ translation_lps)
            ones = np.ones((5, 5, 5), np.uint8)
            nib.save(nib.Nifti1Image(ones, source_affine), str(source_path))
            nib.save(nib.Nifti1Image(np.zeros_like(ones), target_affine), str(fixed_path))
            pull = target_to_source_voxel(source_affine, target_affine, transform_path)
            np.testing.assert_allclose(pull[:3, :3], np.eye(3), atol=1e-6)
            np.testing.assert_allclose(pull[:3, 3], [-.25, 0., 0.], atol=1e-6)
            from_ants = ants.apply_transforms(
                fixed=ants.image_read(str(fixed_path)),
                moving=ants.image_read(str(source_path)),
                transformlist=[str(transform_path)],
                interpolator="nearestNeighbor", defaultvalue=0).numpy()
            old = independent_nearest(ones, ones.shape, pull, mode="constant")
            grid = independent_nearest(ones, ones.shape, pull, mode="grid-constant")
            np.testing.assert_array_equal(grid, from_ants)
            self.assertGreater(int(np.count_nonzero(old != from_ants)), 0)

    def test_exact_half_voxel_tie_and_beyond_boundary(self):
        with tempfile.TemporaryDirectory() as directory:
            directory = Path(directory)
            transform_path = directory / "identity.mat"
            sitk.WriteTransform(sitk.AffineTransform(3), str(transform_path))
            affine = np.diag([-1., -1., 1., 1.])
            source = np.ones((4, 4, 4), np.uint8)
            source_path = directory / "source.nii.gz"
            target_path = directory / "target.nii.gz"
            nib.save(nib.Nifti1Image(source, affine), str(source_path))
            moving = ants.image_read(str(source_path))
            for shift in (-.75, -.5, -.49, -.25, .25, .49, .5, .75):
                with self.subTest(shift=shift):
                    target_affine = affine.copy()
                    target_affine[:3, 3] += affine[:3, :3] @ np.array([shift, 0., 0.])
                    nib.save(nib.Nifti1Image(np.zeros_like(source), target_affine),
                             str(target_path))
                    pull = target_to_source_voxel(affine, target_affine, transform_path)
                    expected = ants.apply_transforms(
                        fixed=ants.image_read(str(target_path)), moving=moving,
                        transformlist=[str(transform_path)],
                        interpolator="nearestNeighbor", defaultvalue=0).numpy()
                    grid = independent_nearest(source, source.shape, pull,
                                               mode="grid-constant")
                    np.testing.assert_array_equal(grid, expected)


if __name__ == "__main__":
    unittest.main()
