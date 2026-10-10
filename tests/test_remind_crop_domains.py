"""Tiny synthetic geometry/label controls; never load patient files."""
import itertools
from pathlib import Path
import unittest
import numpy as np

MODULE = Path(__file__).resolve().parents[1] / "src/resectionlab/remind_planning_qc.py"
namespace = {"__file__": str(MODULE)}
exec(compile(MODULE.read_bytes(), str(MODULE), "exec"), namespace)
crop = namespace["mr_sampling_domain_union_crop"]

def geometry(shape, origin=(0, 0, 0), angle=0):
    a = np.eye(4); c, s = np.cos(angle), np.sin(angle)
    a[:3, :3] = [[c, -s, 0], [s, c, 0], [0, 0, 1]]; a[:3, 3] = origin
    return {'shape_xyz': list(shape), 'affine_xyz_to_ras_mm': a.tolist()}

def inside_all_centres(g, affine, shape):
    coords = np.asarray(list(itertools.product(*[range(n) for n in g['shape_xyz']])))
    mapped = (np.c_[coords, np.ones(len(coords))] @ (np.linalg.inv(affine) @ g['affine_xyz_to_ras_mm']).T)[:, :3]
    return bool(np.all(mapped >= -.5) and np.all(mapped < np.asarray(shape)-.5))

class GeneratedControls(unittest.TestCase):
    def test_target_outside_support_domain_is_preserved_with_unknown_mask(self):
        mr = geometry((32, 32, 32)); support = geometry((6, 6, 6), (8, 8, 8)); target = geometry((4, 4, 4), (6, 9, 9))
        start, shape, native, derived, info = crop(mr, support, target)
        self.assertTrue(inside_all_centres(target, derived, shape))
        self.assertTrue(inside_all_centres(support, derived, shape))
        a, domain, counts = namespace['resample_binary_nn'](np.ones((6, 6, 6), np.uint8), support['affine_xyz_to_ras_mm'], derived, shape)
        t, td, tc = namespace['resample_binary_nn'](np.ones((4, 4, 4), np.uint8), target['affine_xyz_to_ras_mm'], derived, shape)
        self.assertGreater(np.count_nonzero((t != 0) & (domain == 0)), 0)
        self.assertTrue(np.all(a[domain == 0] == 0))
        self.assertEqual(tc['source_positive_centres_outside_target_grid'], 0)
        self.assertEqual(counts['source_positive_centres_outside_target_grid'], 0)
        self.assertEqual(info['image_interpolation'], 'none')

    def test_oblique_support_avoids_uniform_inset_loss(self):
        mr = geometry((50, 50, 50)); support = geometry((20, 24, 20), (20, 10, 10), .17); target = geometry((4, 4, 4), (18, 15, 11))
        start, shape, native, derived, info = crop(mr, support, target)
        self.assertTrue(inside_all_centres(support, derived, shape))
        self.assertTrue(inside_all_centres(target, derived, shape))
        self.assertEqual(info['uniform_inset_MR_cells_per_face'], 0)
        self.assertTrue(np.array_equal(native[:3, 3], np.asarray(start, float)))

    def test_outside_MRI_is_refused_without_extrapolation(self):
        with self.assertRaisesRegex(ValueError, 'outside_acquired_MRI:whole_tumor'):
            crop(geometry((20, 20, 20)), geometry((6, 6, 6), (5, 5, 5)), geometry((4, 4, 4), (18, 4, 4)))

    def test_halo_clips_at_acquired_boundary_without_padding(self):
        mr = geometry((20, 20, 20)); support = geometry((5, 5, 5)); target = geometry((3, 3, 3), (1, 1, 1))
        start, shape, native, derived, info = crop(mr, support, target)
        self.assertEqual(start.tolist(), [0, 0, 0]); self.assertTrue(info['clipped_to_MR_coverage'])
        self.assertTrue(inside_all_centres(support, derived, shape))

    def test_nonnegligible_MRI_shear_refused(self):
        mr = geometry((32, 32, 32)); mr['affine_xyz_to_ras_mm'][0][1] = .02
        with self.assertRaisesRegex(ValueError, 'roundoff_exceeds_bound'):
            crop(mr, geometry((8, 8, 8), (8, 8, 8)), geometry((3, 3, 3), (8, 8, 8)))

    def test_native_voxel_cap_refused_before_array_allocation(self):
        with self.assertRaisesRegex(ValueError, 'public_MR_crop_extent'):
            crop(geometry((500, 500, 500)), geometry((400, 400, 400), (20, 20, 20)), geometry((10, 10, 10), (20, 20, 20)))

    def test_target_partition_distinguishes_label_zero_and_unknown_domain(self):
        target = np.array([1, 1, 1, 0], np.uint8)
        support = np.array([1, 0, 0, 0], np.uint8)
        domain = np.array([1, 1, 0, 0], np.uint8)
        r = namespace['public_target_support_domain_relation'](target, support, domain)
        self.assertEqual(r['full_placed_target_voxels'], 3)
        self.assertEqual([r[k] for k in ['target_in_known_support_positive', 'target_in_known_support_zero', 'target_in_unknown_support_domain']], [1, 1, 1])
        self.assertFalse(r['known_label_zero_is_physical_empty'])
        self.assertFalse(r['unknown_label_zero_is_observed_empty'])
        with self.assertRaisesRegex(ValueError, 'support_positive_outside_source_domain'):
            namespace['public_target_support_domain_relation'](target, np.ones(4, np.uint8), domain)


if __name__ == '__main__': unittest.main()
