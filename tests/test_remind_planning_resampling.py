"""Focused generated world-coordinate, domain and native-sampling controls."""
import itertools
from pathlib import Path
import types
import unittest

MODULE = Path(__file__).resolve().parents[1] / "src/resectionlab/remind_planning_qc.py"
qc = types.ModuleType("remind_planning_qc_under_test"); qc.__file__ = str(MODULE)
exec(compile(MODULE.read_bytes(), str(MODULE), "exec"), qc.__dict__)
try:
    import numpy as np
except ImportError:
    np = None


def geometry(shape, affine):
    return {"shape_xyz": list(shape), "affine_xyz_to_ras_mm": affine.tolist()}


@unittest.skipIf(np is None, "optional numpy unavailable")
class ResamplingControls(unittest.TestCase):
    def test_public_grid_selection_native_samples_and_explicit_roundoff(self):
        mr = np.eye(4); mr[0, 1] = 1e-7; mr[:3, 3] = [8, -13, 2]
        public = mr.copy(); public[:3, :3] *= .5; public[:3, 3] = (mr @ [3, 4, 5, 1])[:3]
        start, shape, native, derived, record = qc.mr_sampling_crop(geometry((20, 21, 22), mr), geometry((16, 18, 20), public))
        self.assertTrue(record["all_output_cell_corners_inside_public_source_domain"])
        self.assertFalse(record["uses_private_labels"])
        self.assertFalse(record["uses_label_positive_extent"])
        self.assertGreater(record["maximum_world_corner_difference_mm"], 0)
        self.assertLess(record["maximum_world_corner_difference_mm"], .001)
        np.testing.assert_equal(native[:3, :3], mr[:3, :3])
        original = np.arange(20*21*22, dtype=np.float32).reshape(20, 21, 22)
        crop = original[tuple(slice(int(o), int(o)+n) for o, n in zip(start, shape))].copy()
        for ijk in ((0, 0, 0), tuple(n-1 for n in shape)):
            self.assertEqual(crop[ijk], original[tuple(np.asarray(ijk)+start)])
            np.testing.assert_allclose(native @ [*ijk, 1], mr @ [*(np.asarray(ijk)+start), 1], atol=1e-12)
        _, domain, _ = qc.resample_binary_nn(np.ones((16, 18, 20), np.uint8), public, derived, shape)
        self.assertTrue(np.all(domain))

    def test_oblique_anisotropic_nn_against_scalar_world_coordinate_oracle(self):
        source = (np.arange(5*6*7).reshape(5, 6, 7)%3 == 0).astype(np.uint8)
        source_affine = np.array([[0, -2, 0, 4], [1, 0, 0, -3], [0, 0, 3, 2], [0, 0, 0, 1]], dtype=float)
        target_affine = np.array([[1, .1, 0, -4], [0, 1, .2, -5], [0, 0, 2, 0], [0, 0, 0, 1]], dtype=float)
        shape = (9, 8, 12); values, domain, report = qc.resample_binary_nn(source, source_affine, target_affine, shape)
        for ijk in itertools.product(*[range(n) for n in shape]):
            world = target_affine @ [*ijk, 1]
            point = np.linalg.solve(source_affine, world)[:3]
            inside = all(-.5 <= p < n-.5 for p, n in zip(point, source.shape))
            self.assertEqual(bool(domain[ijk]), inside)
            expected = source[tuple(int(np.floor(p+.5)) for p in point)] if inside else 0
            self.assertEqual(values[ijk], expected)
        outside = 0
        for ijk in zip(*np.nonzero(source)):
            point = np.linalg.solve(target_affine, source_affine @ [*ijk, 1])[:3]
            outside += not all(-.5 <= p < n-.5 for p, n in zip(point, shape))
        self.assertEqual(report["source_positive_centres_outside_target_grid"], outside)
        self.assertAlmostEqual(report["source_positive_volume_mm3"], int(source.sum())*6)

    def test_domain_is_unknown_outside_and_half_ties_are_explicit(self):
        source = np.array([0, 1, 0], np.uint8).reshape(3, 1, 1)
        affine = np.eye(4); affine[0, 3] = -.5
        values, domain, _ = qc.resample_binary_nn(source, np.eye(4), affine, (4, 1, 1))
        np.testing.assert_equal(values.ravel(), [0, 1, 0, 0])
        np.testing.assert_equal(domain.ravel(), [1, 1, 1, 0])
        self.assertEqual(values[0, 0, 0], values[3, 0, 0])
        self.assertNotEqual(domain[0, 0, 0], domain[3, 0, 0])

    def test_cropped_source_positives_and_sampling_loss_are_reported(self):
        source = np.zeros((8, 2, 2), np.uint8); source[1, 0, 0] = 1; source[7, 0, 0] = 1
        target = np.eye(4); target[0, 0] = 2
        values, _, report = qc.resample_binary_nn(source, np.eye(4), target, (3, 2, 2))
        self.assertEqual(report["source_positive_voxels"], 2)
        self.assertEqual(report["source_positive_centres_outside_target_grid"], 1)
        self.assertEqual(int(values.sum()), 0)  # Inside subvoxel label at x=1 is missed by NN.
        self.assertTrue(report["resampling_can_omit_subvoxel_labels"])
        self.assertEqual(int(source.sum()), 2)

    def test_roundoff_or_no_contained_coverage_refuses(self):
        mr = np.eye(4); mr[0, 1] = .01
        with self.assertRaisesRegex(ValueError, "roundoff"):
            qc.mr_sampling_crop(geometry((20, 20, 20), mr), geometry((20, 20, 20), np.eye(4)))
        public = np.eye(4); public[:3, 3] = 100
        with self.assertRaisesRegex(ValueError, "contained"):
            qc.mr_sampling_crop(geometry((20, 20, 20), np.eye(4)), geometry((5, 5, 5), public))


if __name__ == "__main__": unittest.main()
