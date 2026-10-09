"""Generated geometry controls only: no patient scans, labels or model call."""

import json
import importlib.util
import sys
import tempfile
import unittest
from pathlib import Path

_OPTIONAL_RUNTIME = ("ants", "nibabel", "SimpleITK", "nnunetv2", "torch")
_missing = [name for name in _OPTIONAL_RUNTIME if importlib.util.find_spec(name) is None]
if _missing:
    raise unittest.SkipTest("Scan adapter requires the optional pinned runtime: " + ", ".join(_missing))

import ants
import nibabel as nib
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from resectionlab.scan_target_adapter import (  # noqa: E402
    AtlasShapeFailure, CoverageFailure, FrameMismatch, ImageTransform,
    MaskNotQualified, MissingModality, QualifiedMask, RegistrationFailure, Scan,
    SourceMismatch,
    ModelContractFailure, assert_mask_coverage, assert_same_frame,
    check_model_metadata_only, manual_predictor_from_preverified_network,
    limited_mask_from_existing_synthstrip_record,
    map_record_bound_t1_support_to_flair, map_t1_support_to_native_flair,
    _prepare_pair_with_fixed_affines,
    prepare_registered_diagnostic_pair, register_rigid_scan_only,
    verify_scan_registration_receipt,
    read_ordered_nnunet_channels,
    read_volume, resample_image, sha256_file, validate_scan_inputs,
    whole_tumor_from_labels,
)


AFFINE = np.diag([-1.0, -1.0, 1.0, 1.0])  # nib RAS; ANTs physical LPS is +XYZ


def save_nifti(path, data, affine=AFFINE):
    image = nib.Nifti1Image(np.asarray(data), affine)
    image.header.set_xyzt_units("mm")
    nib.save(image, str(path))
    if np.asarray(data).ndim != 3:
        return None
    return read_volume(path, sha256_file(path))


class GeneratedAdapterTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix="scan-adapter-generated-")
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)

    def test_atlas_singleton_axis_squeeze_preserves_physical_affine(self):
        atlas = self.root / "atlas.nii.gz"
        affine = np.array([[-.7, 0, 0, 4], [0, -1.1, 0, 20],
                           [0, 0, 2.3, -7], [0, 0, 0, 1]], dtype=float)
        save_nifti(atlas, np.ones((9, 10, 11, 1), dtype=np.float32), affine)
        original_file_affine = nib.load(str(atlas)).affine
        result = read_volume(atlas, sha256_file(atlas), atlas=True, working_dir=self.root / "private")
        self.assertEqual(result.shape_xyz, (9, 10, 11))
        np.testing.assert_array_equal(result.affine_ras_mm, original_file_affine)
        bad = self.root / "atlas-nonsingleton.nii.gz"
        save_nifti(bad, np.ones((9, 10, 11, 2), dtype=np.float32), affine)
        with self.assertRaises(AtlasShapeFailure):
            read_volume(bad, sha256_file(bad), atlas=True, working_dir=self.root / "bad")

    def test_nnunet_channel_and_zyx_spacing_are_physically_checked(self):
        affine = np.array([[-.7, 0, 0, 4], [0, -1.1, 0, 20],
                           [0, 0, 2.3, -7], [0, 0, 0, 1]], dtype=float)
        first = np.zeros((9, 10, 11), dtype=np.float32)
        second = np.zeros_like(first)
        first[1, 2, 3] = 19
        second[1, 2, 3] = 37
        t1c = save_nifti(self.root / "t1c.nii.gz", first, affine)
        flair = save_nifti(self.root / "flair.nii.gz", second, affine)
        data, props = read_ordered_nnunet_channels(t1c, flair)
        self.assertEqual(data.shape, (2, 11, 10, 9))
        self.assertEqual(float(data[0, 3, 2, 1]), 19)
        self.assertEqual(float(data[1, 3, 2, 1]), 37)
        np.testing.assert_allclose(props["spacing"], [2.3, 1.1, .7], atol=1e-5)
        shifted = affine.copy()
        shifted[0, 3] += 5
        wrong = save_nifti(self.root / "same-shape-wrong-origin.nii.gz", second, shifted)
        with self.assertRaises(FrameMismatch):
            read_ordered_nnunet_channels(t1c, wrong)

    def test_two_image_transforms_and_native_label_inverse(self):
        flair_array = np.zeros((16, 16, 16), dtype=np.float32)
        flair_array[2, 5, 6] = 1
        t1c_array = np.zeros_like(flair_array)
        t1c_array[4, 5, 6] = 1
        atlas_array = np.zeros_like(flair_array)
        atlas_array[4, 8, 6] = 1
        flair = save_nifti(self.root / "flair.nii.gz", flair_array)
        t1c = save_nifti(self.root / "t1c.nii.gz", t1c_array)
        atlas = save_nifti(self.root / "atlas.nii.gz", atlas_array)
        f_to_t = self.root / "flair-image-to-t1c.mat"
        t_to_a = self.root / "t1c-image-to-atlas.mat"
        ants.write_transform(ants.create_ants_transform(transform_type="Euler3DTransform",
                                                       translation=(-2, 0, 0)), str(f_to_t))
        ants.write_transform(ants.create_ants_transform(transform_type="Euler3DTransform",
                                                       translation=(0, -3, 0)), str(t_to_a))
        step_one = ImageTransform(f_to_t, sha256_file(f_to_t), "flair", "t1c")
        step_two = ImageTransform(t_to_a, sha256_file(t_to_a), "t1c", "atlas")
        with self.assertRaises(RegistrationFailure):
            resample_image(flair, t1c, step_one, source_frame="t1c", target_frame="flair")
        flair_in_t1c = resample_image(flair, t1c, step_one,
                                     source_frame="flair", target_frame="t1c")
        self.assertEqual(tuple(np.unravel_index(flair_in_t1c.numpy().argmax(), flair_array.shape)), (4, 5, 6))
        flair_in_t1c_path = self.root / "flair-in-t1c.nii.gz"
        ants.image_write(flair_in_t1c, str(flair_in_t1c_path))
        flair_t = read_volume(flair_in_t1c_path, sha256_file(flair_in_t1c_path))
        assert_same_frame(t1c, flair_t)
        flair_in_atlas = resample_image(flair_t, atlas, step_two,
                                       source_frame="t1c", target_frame="atlas")
        self.assertEqual(tuple(np.unravel_index(flair_in_atlas.numpy().argmax(), atlas_array.shape)), (4, 8, 6))
        self.assertEqual(tuple(np.unravel_index(resample_image(t1c, atlas, step_two,
                                                               source_frame="t1c", target_frame="atlas").numpy().argmax(),
                                                atlas_array.shape)), (4, 8, 6))
        # The affine's point action goes fixed -> moving, opposite its image action.
        np.testing.assert_allclose(ants.read_transform(str(f_to_t)).apply_to_point((4, 5, 6)), (2, 5, 6))
        np.testing.assert_allclose(ants.read_transform(str(t_to_a)).apply_to_point((4, 8, 6)), (4, 5, 6))
        label_atlas = self.root / "generated-estimate-label.nii.gz"
        labels = np.zeros_like(atlas_array, dtype=np.uint8)
        labels[4, 8, 6] = 1
        model_output = save_nifti(label_atlas, labels)
        wt_in_t1c = resample_image(model_output, t1c, step_two,
                                  source_frame="atlas", target_frame="t1c", inverse=True, label=True)
        self.assertEqual(tuple(np.argwhere(wt_in_t1c.numpy() == 1)[0]), (4, 5, 6))
        wt_in_t1c_path = self.root / "generated-estimate-wt-t1c.nii.gz"
        ants.image_write(wt_in_t1c, str(wt_in_t1c_path))
        native = resample_image(read_volume(wt_in_t1c_path, sha256_file(wt_in_t1c_path)),
                                flair, step_one, source_frame="t1c", target_frame="flair",
                                inverse=True, label=True)
        self.assertEqual(tuple(np.argwhere(native.numpy() == 1)[0]), (2, 5, 6))
        self.assertEqual(np.count_nonzero(native.numpy()), 1)

    def test_missing_modality_mask_lineage_and_coverage_abstain(self):
        t1c = Scan("t1c", self.root / "t1c.nii.gz", "1" * 64, "preoperative_source")
        flair = Scan("t2f", self.root / "flair.nii.gz", "2" * 64, "preoperative_source")
        mask = QualifiedMask(self.root / "mask.nii.gz", "3" * 64, "2" * 64,
                             "flair", "qualified_scan_derived")
        validate_scan_inputs(t1c, flair, mask)
        with self.assertRaises(MissingModality):
            validate_scan_inputs(t1c, None, mask)
        with self.assertRaises(MaskNotQualified):
            validate_scan_inputs(t1c, flair, QualifiedMask(mask.path, mask.expected_sha256,
                                                           "4" * 64, "flair", mask.qc_status))
        limited = QualifiedMask(mask.path, mask.expected_sha256, flair.expected_sha256,
                                "flair", "documented_limited_scan_derived",
                                "inferior field omission; diagnostic ROI only")
        with self.assertRaises(MaskNotQualified):
            validate_scan_inputs(t1c, flair, limited)
        validate_scan_inputs(t1c, flair, limited, diagnostic_only=True)
        support = np.zeros((4, 5, 6), dtype=bool)
        support[2, 3, 4] = True
        valid = np.ones_like(support)
        assert_mask_coverage(support, valid, valid)
        valid[2, 3, 4] = False
        with self.assertRaises(CoverageFailure):
            assert_mask_coverage(support, valid)

    def test_end_to_end_generated_scan_pair_preparation_is_unreviewed(self):
        flair_data = np.zeros((16, 16, 16), dtype=np.float32)
        flair_data[2, 5, 6] = 37
        t1c_data = np.zeros_like(flair_data)
        t1c_data[4, 5, 6] = 19
        mask_data = np.zeros_like(flair_data, dtype=np.uint8)
        mask_data[2, 5, 6] = 1
        t1c_path, flair_path, mask_path, atlas_path = (
            self.root / name for name in ("t1c.nii.gz", "flair.nii.gz", "mask.nii.gz", "atlas.nii.gz"))
        save_nifti(t1c_path, t1c_data)
        save_nifti(flair_path, flair_data)
        save_nifti(mask_path, mask_data)
        save_nifti(atlas_path, np.zeros_like(flair_data))
        t1c = Scan("t1c", t1c_path, sha256_file(t1c_path), "preoperative_source")
        flair = Scan("t2f", flair_path, sha256_file(flair_path), "preoperative_source")
        mask = QualifiedMask(mask_path, sha256_file(mask_path), flair.expected_sha256,
                             "flair", "qualified_scan_derived")
        f_to_t, t_to_a = self.root / "f-to-t.mat", self.root / "t-to-a.mat"
        ants.write_transform(ants.create_ants_transform(transform_type="Euler3DTransform",
                                                       translation=(-2, 0, 0)), str(f_to_t))
        ants.write_transform(ants.create_ants_transform(transform_type="Euler3DTransform",
                                                       translation=(0, -3, 0)), str(t_to_a))
        first = ImageTransform(f_to_t, sha256_file(f_to_t), "flair", "t1c")
        second = ImageTransform(t_to_a, sha256_file(t_to_a), "t1c", "atlas")
        prepared = _prepare_pair_with_fixed_affines(t1c, flair, mask, atlas_path,
                                                    sha256_file(atlas_path), first, second,
                                                    self.root / "prepared",
                                                    require_registration_receipts=False)
        self.assertEqual(prepared.qc_status, "geometry_only_requires_local_anatomy_review")
        self.assertTrue(0 < prepared.atlas_common_fov_fraction <= 1)
        ordered, _ = read_ordered_nnunet_channels(prepared.t1c_atlas, prepared.flair_atlas)
        self.assertEqual(float(ordered[0, 6, 8, 4]), 19)
        self.assertEqual(float(ordered[1, 6, 8, 4]), 37)
        self.assertEqual(np.count_nonzero(ordered[0]), 1)
        self.assertEqual(np.count_nonzero(ordered[1]), 1)
        self.assertEqual(np.count_nonzero(np.asarray(nib.load(str(prepared.atlas_mask.path)).dataobj)), 1)
        t1_support = self.root / "t1-known-support.nii.gz"
        support = np.zeros_like(mask_data)
        support[4, 5, 6] = 1
        save_nifti(t1_support, support)
        native_support = map_t1_support_to_native_flair(prepared, t1_support,
                                                        sha256_file(t1_support),
                                                        source_t1c_sha256=t1c.expected_sha256,
                                                        output_path=self.root / "prepared/native-flair-support.nii.gz")
        assert_same_frame(native_support, prepared.flair_native)
        mapped_data = np.asarray(nib.load(str(native_support.path)).dataobj)
        self.assertEqual(tuple(np.argwhere(mapped_data == 1)[0]), (2, 5, 6))
        with self.assertRaises(SourceMismatch):
            map_t1_support_to_native_flair(prepared, mask_path, sha256_file(mask_path),
                                           source_t1c_sha256=flair.expected_sha256,
                                           output_path=self.root / "bad-support.nii.gz")
        limited = QualifiedMask(mask_path, sha256_file(mask_path), flair.expected_sha256,
                                "flair", "documented_limited_scan_derived",
                                "generated test mask contains one voxel only")
        diagnostic = _prepare_pair_with_fixed_affines(t1c, flair, limited, atlas_path,
                                                      sha256_file(atlas_path), first, second,
                                                      self.root / "diagnostic", diagnostic_only=True,
                                                      require_registration_receipts=False)
        self.assertEqual(diagnostic.qc_status, "diagnostic_forward_only_mask_limitations_retained")

    def test_scan_only_rigid_registration_improves_generated_alignment(self):
        x, y, z = np.meshgrid(*[np.arange(32)] * 3, indexing="ij")
        moving_data = (np.exp(-((x - 11)**2 + (y - 15)**2 + (z - 17)**2) / 24)
                       + .7 * np.exp(-((x - 20)**2 + (y - 10)**2 + (z - 9)**2) / 12))
        fixed_data = (np.exp(-((x - 13)**2 + (y - 16)**2 + (z - 17)**2) / 24)
                      + .7 * np.exp(-((x - 22)**2 + (y - 11)**2 + (z - 9)**2) / 12))
        moving = save_nifti(self.root / "generated-moving.nii.gz", moving_data.astype(np.float32))
        fixed = save_nifti(self.root / "generated-fixed.nii.gz", fixed_data.astype(np.float32))
        transform = register_rigid_scan_only(fixed, moving, moving_frame="flair",
                                             fixed_frame="t1c", working_dir=self.root / "rigid")
        aligned = resample_image(moving, fixed, transform,
                                 source_frame="flair", target_frame="t1c").numpy()
        before = float(np.mean((moving_data - fixed_data)**2))
        after = float(np.mean((aligned - fixed_data)**2))
        self.assertLess(after, before / 10)
        self.assertEqual(sha256_file(transform.path), transform.sha256)
        receipt = verify_scan_registration_receipt(transform, moving=moving, fixed=fixed)
        self.assertEqual(receipt["provenance"], "estimated_from_scans_only")

    def test_existing_scan_mask_record_binds_exact_inputs_and_diagnostic_runner(self):
        x, y, z = np.meshgrid(*[np.arange(32)] * 3, indexing="ij")
        def phantom(dx, dy):
            return (np.exp(-((x - 11 - dx)**2 + (y - 15 - dy)**2 + (z - 17)**2) / 24)
                    + .7 * np.exp(-((x - 20 - dx)**2 + (y - 10 - dy)**2 + (z - 9)**2) / 12)).astype(np.float32)
        t1c_path, flair_path, atlas_path = (
            self.root / name for name in ("diag-t1c.nii.gz", "diag-flair.nii.gz", "diag-atlas.nii.gz"))
        save_nifti(t1c_path, phantom(0, 0))
        save_nifti(flair_path, phantom(-2, 0))
        save_nifti(atlas_path, phantom(0, 2))
        mask_path = self.root / "main_mask.nii.gz"
        save_nifti(mask_path, (((x - 11)**2 + (y - 15)**2 + (z - 17)**2) <= 9).astype(np.uint8))
        t1c = Scan("t1c", t1c_path, sha256_file(t1c_path), "preoperative_source")
        flair = Scan("t2f", flair_path, sha256_file(flair_path), "preoperative_source")
        record = {"model": {"name": "SynthStrip"}, "input_sha256": t1c.expected_sha256,
                  "artifact_hashes": {"main_mask.nii.gz": sha256_file(mask_path)},
                  "provenance": "estimated", "exit_code": 0, "failure": None,
                  "brain_reviewed": False, "cortical_access_permitted": False}
        record_path = self.root / "synthstrip-record.json"
        record_path.write_text(json.dumps(record))
        record_sha = sha256_file(record_path)
        limited = limited_mask_from_existing_synthstrip_record(
            t1c, mask_path, record_path, expected_record_sha256=record_sha,
            limitations="inferior field omitted; ROI diagnostic only")
        self.assertEqual(limited.frame, "t1c")
        record["input_sha256"] = flair.expected_sha256
        record_path.write_text(json.dumps(record))
        with self.assertRaises(SourceMismatch):
            limited_mask_from_existing_synthstrip_record(
                t1c, mask_path, record_path, expected_record_sha256=record_sha,
                limitations="known omission")
        record["input_sha256"] = t1c.expected_sha256
        record_path.write_text(json.dumps(record))
        prepared = prepare_registered_diagnostic_pair(
            t1c, flair, synthstrip_mask_path=mask_path,
            synthstrip_record_path=record_path,
            synthstrip_record_sha256=record_sha,
            mask_limitations="inferior field omitted; ROI diagnostic only",
            atlas_path=atlas_path, atlas_sha256=sha256_file(atlas_path),
            private_dir=self.root / "receipt-run")
        self.assertEqual(prepared.qc_status, "diagnostic_forward_only_mask_limitations_retained")
        self.assertTrue(prepared.flair_to_t1c.generation_receipt_path.is_file())
        self.assertTrue(prepared.t1c_to_atlas.generation_receipt_path.is_file())
        support_flair = map_record_bound_t1_support_to_flair(
            prepared, t1c, synthstrip_mask_path=mask_path,
            synthstrip_record_path=record_path,
            synthstrip_record_sha256=record_sha,
            limitations="inferior field omitted; ROI diagnostic only",
            output_path=self.root / "receipt-run/t1-support-in-flair.nii.gz")
        assert_same_frame(support_flair, prepared.flair_native)
        self.assertGreater(np.count_nonzero(np.asarray(nib.load(str(support_flair.path)).dataobj)), 0)

    def test_exported_label_semantics_are_not_logit_semantics(self):
        labels = np.array([[[0, 1, 2, 3]]], dtype=np.uint8)
        np.testing.assert_array_equal(whole_tumor_from_labels(labels), [[[0, 1, 1, 1]]])
        with self.assertRaises(FrameMismatch):
            whole_tumor_from_labels(labels.astype(np.float32))

    def test_metadata_contract_rejects_swapped_channels_before_model_load(self):
        folder = self.root / "dummy-model"
        (folder / "fold_all").mkdir(parents=True)
        source = Path(__file__).resolve().parents[1] / "data/models/gliomoda-v1.0.2/t1c-t2f-pinned-v1"
        dataset = json.loads((source / "dataset.json").read_text())
        plans = json.loads((source / "plans.json").read_text())
        (folder / "dataset.json").write_text(json.dumps(dataset))
        (folder / "plans.json").write_text(json.dumps(plans))
        (folder / "fold_all/checkpoint_final.pth").write_bytes(b"not a model")
        files = ["dataset.json", "plans.json", "fold_all/checkpoint_final.pth"]

        def write_receipt():
            receipt = {"files": {name: {"sha256": sha256_file(folder / name)} for name in files}}
            path = self.root / "dummy-receipt.json"
            path.write_text(json.dumps(receipt))
            return path

        receipt_path = write_receipt()
        accepted = check_model_metadata_only(folder, receipt_path)
        self.assertEqual(accepted["dataset"]["channel_names"], {"0": "t1c", "1": "t2f"})
        dataset["channel_names"] = {"0": "t2f", "1": "t1c"}
        (folder / "dataset.json").write_text(json.dumps(dataset))
        receipt_path = write_receipt()  # even matching new hashes must not mask the semantic violation
        with self.assertRaises(ModelContractFailure):
            check_model_metadata_only(folder, receipt_path)

        # Constructor-only control with a tiny generated network; this cannot
        # certify the pinned checkpoint's architecture or patient inference.
        import torch
        with self.assertRaises(ModelContractFailure):
            manual_predictor_from_preverified_network(
                torch.nn.Conv3d(2, 3, 1), accepted, checkpoint_sha256="0" * 64,
                strict_state_verified=True, device=torch.device("cpu"))
        predictor = manual_predictor_from_preverified_network(
            torch.nn.Conv3d(2, 3, 1), accepted,
            checkpoint_sha256=accepted["checkpoint_sha256"],
            strict_state_verified=True, device=torch.device("cpu"))
        self.assertEqual(predictor.label_manager.num_segmentation_heads, 3)
        self.assertEqual(len(predictor.list_of_parameters), 1)


if __name__ == "__main__":
    unittest.main()
