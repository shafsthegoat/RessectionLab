"""Generated geometry/intensity controls only; no patient, checkpoint or network."""

import hashlib
from importlib.metadata import PackageNotFoundError, version
from importlib.util import find_spec
import json
from pathlib import Path
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
MODEL_META = ROOT / "data/models/gliomoda-v1.0.2/t1c-t2f-pinned-v1"
if not (MODEL_META / "plans.json").is_file() or not (MODEL_META / "dataset.json").is_file():
    raise unittest.SkipTest("optional pinned GlioMODA JSON metadata is unavailable")
try:
    NNUNET_VERSION = version("nnunetv2")
except PackageNotFoundError:
    raise unittest.SkipTest("optional nnU-Net 2.5.2 runtime is unavailable")
if NNUNET_VERSION != "2.5.2":
    raise unittest.SkipTest("optional nnU-Net runtime is not pinned 2.5.2")
for dependency in ("numpy", "nibabel", "torch", "ants", "SimpleITK", "acvl_utils",
                   "batchgenerators", "skimage"):
    if find_spec(dependency) is None:
        raise unittest.SkipTest(f"optional scan diagnostic runtime lacks {dependency}")

import nibabel as nib  # noqa: E402
import numpy as np  # noqa: E402
from nnunetv2.inference.export_prediction import (  # noqa: E402
    convert_predicted_logits_to_segmentation_with_correct_shape,
)
from nnunetv2.utilities.plans_handling.plans_handler import PlansManager  # noqa: E402

sys.path.insert(0, str(ROOT / "src"))

from resectionlab.scan_support_contract import diagnostic_display_state  # noqa: E402
from resectionlab.scan_target_adapter import (  # noqa: E402
    CoverageFailure, FrameMismatch, ModelContractFailure, SourceMismatch,
)
from resectionlab.scan_preprocess_bridge import prepare_scan_only_patch  # noqa: E402

AFFINE = np.diag([-1.0, -1.0, 1.0, 1.0])


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def save(path, array, affine=AFFINE):
    image = nib.Nifti1Image(array, affine)
    image.header.set_xyzt_units("mm")
    image.set_sform(affine, code=2)
    image.set_qform(None, code=0)
    nib.save(image, str(path))
    return path


class GeneratedBridgeTests(unittest.TestCase):
    def setUp(self):
        self.folder = tempfile.TemporaryDirectory(prefix="generated-scan-preprocess-")
        self.addCleanup(self.folder.cleanup)
        self.work = Path(self.folder.name)
        x, y, z = np.indices((14, 12, 10))
        self.domain = np.zeros((14, 12, 10), bool)
        self.domain[3:10, 2:9, 2:8] = True
        t1 = np.zeros(self.domain.shape, np.float32)
        flair = np.zeros_like(t1)
        t1[self.domain] = (1 + x + 2*y + 3*z)[self.domain]
        flair[self.domain] = (20 + 3*x + y + z)[self.domain]
        bits = np.full(self.domain.shape, 3, np.uint8)
        bits[self.domain] = 7
        self.t1 = save(self.work / "t1c.nii.gz", t1)
        self.flair = save(self.work / "flair.nii.gz", flair)
        self.support = save(self.work / "support.nii.gz", bits)
        self.t1_raw = t1
        self.flair_raw = flair
        self.arguments = {
            "t1c_path": self.t1, "t1c_sha256": sha(self.t1),
            "flair_path": self.flair, "flair_sha256": sha(self.flair),
            "support_map_path": self.support, "support_map_sha256": sha(self.support),
            "plans_path": MODEL_META / "plans.json", "plans_sha256": sha(MODEL_META / "plans.json"),
            "dataset_path": MODEL_META / "dataset.json", "dataset_sha256": sha(MODEL_META / "dataset.json"),
        }

    def test_real_default_preprocessor_and_analytic_masked_zscore(self):
        result = prepare_scan_only_patch(**self.arguments)
        self.assertEqual(result.tensor_czyx.shape, (2, 128, 128, 128))
        self.assertEqual(result.tensor_czyx.dtype, np.float32)
        self.assertEqual(result.geometry.crop_bbox_zyx, ((2, 8), (2, 9), (3, 10)))
        self.assertEqual(result.geometry.shape_after_cropping_zyx, (6, 7, 7))
        self.assertEqual(result.geometry.preprocessed_shape_zyx, (6, 7, 7))
        self.assertEqual(int(result.prospective_domain_xyz().sum()), int(self.domain.sum()))
        self.assertTrue(np.array_equal(result.candidate_domain_xyz(), self.domain))
        voxel_xyz = (4, 3, 3)
        patch_zyx = tuple(voxel_xyz[::-1][axis] - result.geometry.crop_bbox_zyx[axis][0]
                          - result.geometry.patch_start_zyx[axis] for axis in range(3))
        for channel, raw in enumerate((self.t1_raw, self.flair_raw)):
            expected = (raw[voxel_xyz] - raw[self.domain].mean()) / raw[self.domain].std()
            self.assertAlmostEqual(float(result.tensor_czyx[(channel, *patch_zyx)]),
                                   float(expected), places=5)
        metadata = result.geometry.display_only_metadata()
        self.assertEqual(metadata["display_kind"], "candidate_segmentation_unreviewed")
        self.assertEqual(metadata["prediction_coverage_status"],
                         "unavailable_until_verified_output_receipt")
        self.assertFalse(metadata["planning_eligible"])
        self.assertEqual(metadata["affine_ras_mm"], AFFINE.tolist())
        json.dumps(metadata)

    def test_inverse_crop_pad_and_output_coverage_preserve_unknown(self):
        result = prepare_scan_only_patch(**self.arguments)
        voxel_xyz = (4, 3, 3)
        patch_zyx = tuple(voxel_xyz[::-1][axis] - result.geometry.crop_bbox_zyx[axis][0]
                          - result.geometry.patch_start_zyx[axis] for axis in range(3))
        limited = np.zeros((128, 128, 128), np.uint8)
        limited[patch_zyx] = 1
        mapped = result.geometry.map_patch_coverage_to_atlas(limited, result.configuration)
        self.assertEqual(int(mapped.sum()), 1)
        self.assertTrue(mapped[voxel_xyz])
        candidate = np.zeros(self.domain.shape, np.uint8)
        candidate[voxel_xyz] = 1
        state, excluded = diagnostic_display_state(candidate, result.support_bits_xyz,
                                                    mapped & result.candidate_domain_xyz())
        self.assertEqual(excluded, 0)
        self.assertEqual(int(state[voxel_xyz]), 1)
        self.assertEqual(int(state[5, 4, 4]), -1)  # bit 7, but no model output here
        self.assertEqual(int(state[0, 0, 0]), -1)  # both scan FOVs, outside estimated mask
        with self.assertRaises(CoverageFailure):
            result.geometry.map_patch_coverage_to_atlas(np.full((128, 128, 128), 2),
                                                        result.configuration)

    def test_logits_decode_matches_upstream_nnunet_export_and_partial_coverage(self):
        result = prepare_scan_only_patch(**self.arguments)
        logits = np.full((3, 128, 128, 128), -8.0, np.float32)
        voxel_xyz = (4, 3, 3)
        patch_zyx = tuple(voxel_xyz[::-1][axis] - result.geometry.crop_bbox_zyx[axis][0]
                          - result.geometry.patch_start_zyx[axis] for axis in range(3))
        logits[(slice(None), *patch_zyx)] = 8.0
        full = result.geometry._insert_patch(logits)
        g = result.geometry
        properties = {
            "spacing": list(g.original_spacing_zyx_mm),
            "shape_after_cropping_and_before_resampling": list(g.shape_after_cropping_zyx),
            "shape_before_cropping": list(g.shape_before_cropping_zyx),
            "bbox_used_for_cropping": [list(pair) for pair in g.crop_bbox_zyx],
        }
        manager = PlansManager(json.loads(Path(self.arguments["plans_path"]).read_text()))
        native = convert_predicted_logits_to_segmentation_with_correct_shape(
            full, manager, result.configuration, result.label_manager, properties,
            num_threads_torch=1).transpose(2, 1, 0)
        actual = g.map_patch_logits_to_atlas(logits, result.configuration, result.label_manager)
        np.testing.assert_array_equal(actual, native)
        supplied = np.zeros((128, 128, 128), np.uint8)
        supplied[patch_zyx] = 1
        display = result.decode_supplied_logits_for_display(logits, supplied)
        self.assertEqual(int(display.state_xyz[voxel_xyz]), 1)
        self.assertEqual(int(display.state_xyz[5, 4, 4]), -1)
        self.assertEqual(int(display.prediction_coverage_xyz.sum()), 1)
        self.assertFalse(display.metadata["planning_eligible"])
        self.assertEqual(display.metadata["prediction_coverage_status"],
                         "supplied_output_unverified")
        self.assertEqual(display.metadata["decoding_semantics"],
                         "pinned_nnunet_probability_resample_then_regions")

    def test_reversed_source_and_changed_plan_fail_closed(self):
        swapped = dict(self.arguments, t1c_path=self.flair, flair_path=self.t1)
        with self.assertRaises(SourceMismatch):
            prepare_scan_only_patch(**swapped)
        plans = json.loads(Path(self.arguments["plans_path"]).read_text())
        plans["configurations"]["3d_fullres"]["normalization_schemes"][0] = "NoNormalization"
        altered = self.work / "altered-plans.json"
        altered.write_text(json.dumps(plans))
        changed = dict(self.arguments, plans_path=altered, plans_sha256=sha(altered))
        with self.assertRaises(ModelContractFailure):
            prepare_scan_only_patch(**changed)
        plans = json.loads(Path(self.arguments["plans_path"]).read_text())
        plans["configurations"]["3d_fullres"]["resampling_fn_probabilities_kwargs"]["order"] = 0
        altered.write_text(json.dumps(plans))
        changed = dict(self.arguments, plans_path=altered, plans_sha256=sha(altered))
        with self.assertRaises(ModelContractFailure):
            prepare_scan_only_patch(**changed)

    def test_wrong_support_frame_and_unmasked_intensity_refuse(self):
        shifted = AFFINE.copy()
        shifted[0, 3] = 7
        bad_support = save(self.work / "shifted-support.nii.gz",
                           np.asarray(nib.load(str(self.support)).dataobj), shifted)
        changed = dict(self.arguments, support_map_path=bad_support,
                       support_map_sha256=sha(bad_support))
        with self.assertRaises(FrameMismatch):
            prepare_scan_only_patch(**changed)
        t1 = self.t1_raw.copy()
        t1[0, 0, 0] = 99  # scan data would escape the unreviewed bit-7 domain
        bad_t1 = save(self.work / "unmasked-t1c.nii.gz", t1)
        changed = dict(self.arguments, t1c_path=bad_t1, t1c_sha256=sha(bad_t1))
        with self.assertRaises(CoverageFailure):
            prepare_scan_only_patch(**changed)

    def test_two_mm_scan_spacing_uses_pinned_inverse_resampling(self):
        spaced = np.diag([-2.0, -2.0, 2.0, 1.0])
        t1 = save(self.work / "t1c-2mm.nii.gz", self.t1_raw, spaced)
        flair = save(self.work / "flair-2mm.nii.gz", self.flair_raw, spaced)
        support = save(self.work / "support-2mm.nii.gz",
                       np.asarray(nib.load(str(self.support)).dataobj), spaced)
        args = dict(self.arguments, t1c_path=t1, t1c_sha256=sha(t1),
                    flair_path=flair, flair_sha256=sha(flair),
                    support_map_path=support, support_map_sha256=sha(support))
        result = prepare_scan_only_patch(**args)
        self.assertNotEqual(result.geometry.preprocessed_shape_zyx,
                            result.geometry.shape_after_cropping_zyx)
        self.assertEqual(result.geometry.original_spacing_zyx_mm, (2.0, 2.0, 2.0))
        self.assertEqual(result.geometry.target_spacing_zyx_mm, (1.0, 1.0, 1.0))
        self.assertEqual(int(result.prospective_domain_xyz().sum()), int(self.domain.sum()))
        logits = np.full((3, 128, 128, 128), -4.0, np.float32)
        g = result.geometry
        full = g._insert_patch(logits)
        manager = PlansManager(json.loads(Path(args["plans_path"]).read_text()))
        properties = {
            "spacing": list(g.original_spacing_zyx_mm),
            "shape_after_cropping_and_before_resampling": list(g.shape_after_cropping_zyx),
            "shape_before_cropping": list(g.shape_before_cropping_zyx),
            "bbox_used_for_cropping": [list(pair) for pair in g.crop_bbox_zyx],
        }
        native = convert_predicted_logits_to_segmentation_with_correct_shape(
            full, manager, result.configuration, result.label_manager, properties,
            num_threads_torch=1).transpose(2, 1, 0)
        mapped = g.map_patch_logits_to_atlas(logits, result.configuration, result.label_manager)
        np.testing.assert_array_equal(mapped, native)

    def test_anisotropic_partial_patch_never_turns_missing_logits_into_negatives(self):
        x, y, z = np.indices((150, 12, 10))
        domain = np.zeros((150, 12, 10), bool)
        domain[2:148, 2:10, 2:8] = True
        t1 = np.zeros(domain.shape, np.float32)
        flair = np.zeros_like(t1)
        t1[domain] = (1 + x + 2*y + 3*z)[domain]
        flair[domain] = (20 + 3*x + y + z)[domain]
        bits = np.full(domain.shape, 3, np.uint8)
        bits[domain] = 7
        anisotropic = np.diag([-2.0, -1.0, 1.0, 1.0])
        t1_path = save(self.work / "wide-t1c.nii.gz", t1, anisotropic)
        flair_path = save(self.work / "wide-flair.nii.gz", flair, anisotropic)
        support_path = save(self.work / "wide-support.nii.gz", bits, anisotropic)
        args = dict(self.arguments, t1c_path=t1_path, t1c_sha256=sha(t1_path),
                    flair_path=flair_path, flair_sha256=sha(flair_path),
                    support_map_path=support_path, support_map_sha256=sha(support_path))
        result = prepare_scan_only_patch(**args)
        g = result.geometry
        self.assertGreater(g.preprocessed_shape_zyx[2], 128)
        self.assertNotEqual(g.preprocessed_shape_zyx, g.shape_after_cropping_zyx)
        supplied = np.ones((128, 128, 128), np.uint8)
        logits = np.full((3, 128, 128, 128), -8.0, np.float32)
        display = result.decode_supplied_logits_for_display(logits, supplied)
        covered = display.prediction_coverage_xyz
        omitted = domain & ~covered
        self.assertTrue(covered.any())
        self.assertTrue(omitted.any())
        self.assertTrue(np.all(display.state_xyz[covered] == 0))
        self.assertTrue(np.all(display.state_xyz[omitted] == -1))
        self.assertTrue(np.all(display.state_xyz[~domain] == -1))
        # Sentinel logits on the omitted preprocessed voxels would create
        # positive native labels. The source-only adapter never displays them.
        full = g._insert_patch(logits)
        actual_patch = g._insert_patch(supplied[None])[0].astype(bool)
        full[:, ~actual_patch] = 20.0
        manager = PlansManager(json.loads(Path(args["plans_path"]).read_text()))
        properties = {
            "spacing": list(g.original_spacing_zyx_mm),
            "shape_after_cropping_and_before_resampling": list(g.shape_after_cropping_zyx),
            "shape_before_cropping": list(g.shape_before_cropping_zyx),
            "bbox_used_for_cropping": [list(pair) for pair in g.crop_bbox_zyx],
        }
        sentinel_labels = convert_predicted_logits_to_segmentation_with_correct_shape(
            full, manager, result.configuration, result.label_manager, properties,
            num_threads_torch=1).transpose(2, 1, 0)
        self.assertGreater(int(np.count_nonzero((sentinel_labels > 0) & omitted)), 0)


if __name__ == "__main__":
    unittest.main()
