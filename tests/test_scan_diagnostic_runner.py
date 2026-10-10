"""Generated handoff controls plus read-only exact Case4 metadata checks."""

import hashlib
from importlib.metadata import PackageNotFoundError, version
from importlib.util import find_spec
from pathlib import Path
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
MODEL = ROOT / "data/models/gliomoda-v1.0.2/t1c-t2f-pinned-v1"
if not (MODEL / "plans.json").is_file() or not (MODEL / "dataset.json").is_file():
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

sys.path.insert(0, str(ROOT / "src"))

from resectionlab.scan_preprocess_bridge import prepare_scan_only_patch  # noqa: E402
from resectionlab.scan_diagnostic_runner import (  # noqa: E402
    DiagnosticHandoffError, RECEIPT_PINS, _require_prepared_input, decode_model_output,
    model_input_array, read_accepted_forward, sha256_file, verify_case4_metadata,
    write_unverified_display_control, write_unverified_model_input_control,
    write_case4_display_from_saved,
)

AFFINE = np.diag([-1.0, -1.0, 1.0, 1.0])


def save(path, array):
    image = nib.Nifti1Image(array, AFFINE)
    image.header.set_xyzt_units("mm")
    image.set_sform(AFFINE, code=2)
    image.set_qform(None, code=0)
    nib.save(image, str(path))
    return path


class DiagnosticHandoffControls(unittest.TestCase):
    def test_exact_saved_case4_metadata_chain_without_patient_pixels(self):
        bindings = verify_case4_metadata(ROOT, check_input_bytes=False)
        self.assertEqual(bindings.t1c_sha256,
                         "608516aa3c55e8450e777d892dfa5fb5db5824046f85085a25b884e0fcf9d953")
        self.assertEqual(bindings.flair_sha256,
                         "046a8b7aa9620bdf0aeb2b0e40632b4c10ff58cff1ed83a458bd7b05cd86f993")
        self.assertEqual(bindings.support_sha256,
                         "4932902a641aac5d884fbfec543b8da9cdefa9437407e488c1ebef6b9ca700f2")
        with tempfile.TemporaryDirectory(prefix="generated-receipt-tamper-") as tmp:
            fake_root = Path(tmp)
            for relative, _digest in RECEIPT_PINS.values():
                copied = fake_root / relative
                copied.parent.mkdir(parents=True, exist_ok=True)
                copied.write_bytes((ROOT / relative).read_bytes())
            support = fake_root / RECEIPT_PINS["support"][0]
            support.write_bytes(support.read_bytes() + b"\n")
            with self.assertRaises(DiagnosticHandoffError):
                verify_case4_metadata(fake_root, check_input_bytes=False)

    def test_generated_preprocess_handoff_display_and_release_refusal(self):
        with tempfile.TemporaryDirectory(prefix="generated-case4-diagnostic-") as tmp:
            work = Path(tmp)
            x, y, z = np.indices((14, 12, 10))
            domain = np.zeros((14, 12, 10), bool)
            domain[3:10, 2:9, 2:8] = True
            t1 = np.zeros(domain.shape, np.float32)
            flair = np.zeros_like(t1)
            t1[domain] = (1 + x + 2 * y + 3 * z)[domain]
            flair[domain] = (20 + 3 * x + y + z)[domain]
            bits = np.full(domain.shape, 3, np.uint8)
            bits[domain] = 7
            t1_path = save(work / "t1.nii.gz", t1)
            flair_path = save(work / "flair.nii.gz", flair)
            support_path = save(work / "support.nii.gz", bits)
            args = {
                "t1c_path": t1_path, "t1c_sha256": sha256_file(t1_path),
                "flair_path": flair_path, "flair_sha256": sha256_file(flair_path),
                "support_map_path": support_path,
                "support_map_sha256": sha256_file(support_path),
                "plans_path": MODEL / "plans.json",
                "plans_sha256": sha256_file(MODEL / "plans.json"),
                "dataset_path": MODEL / "dataset.json",
                "dataset_sha256": sha256_file(MODEL / "dataset.json"),
            }
            prepared = prepare_scan_only_patch(**args)
            input_receipt = write_unverified_model_input_control(prepared, work / "handoff")
            loaded = np.load(work / "handoff/input.npy", allow_pickle=False)
            np.testing.assert_array_equal(loaded, model_input_array(prepared))
            self.assertEqual(input_receipt["input_raw_sha256"],
                             hashlib.sha256(loaded.tobytes(order="C")).hexdigest())
            with self.assertRaises(FileExistsError):
                write_unverified_model_input_control(prepared, work / "handoff")
            logits = np.full((1, 3, 128, 128, 128), -8.0, np.float32)
            point_xyz = (4, 3, 3)
            point_zyx = tuple(point_xyz[::-1][axis] -
                              prepared.geometry.crop_bbox_zyx[axis][0] -
                              prepared.geometry.patch_start_zyx[axis] for axis in range(3))
            logits[(0, slice(None), *point_zyx)] = 8.0
            with self.assertRaises(DiagnosticHandoffError):
                decode_model_output(prepared, logits,
                                    input_receipt=input_receipt,
                                    accepted_forward={"status": "accepted_case4_development_single128"})
            with self.assertRaises(DiagnosticHandoffError):
                read_accepted_forward(work / "nonexistent-model-release", input_receipt,
                                      "0e29f882310fe8cb076d6cadb982067ef53c6a32231f40ae17d9c173aa4307b3")
            with self.assertRaises(DiagnosticHandoffError):
                write_case4_display_from_saved(ROOT, work / "handoff",
                                               work / "nonexistent-model-release",
                                               work / "patient-display")
            display = prepared.decode_supplied_logits_for_display(
                logits[0], np.ones((128, 128, 128), np.uint8))
            artifact = write_unverified_display_control(prepared, display, work / "display")
            self.assertFalse(artifact["planning_eligible"])
            self.assertFalse(artifact["evaluation_eligible"])
            self.assertFalse(artifact["model_output_verified"])
            self.assertEqual(artifact["schema"], "unverified_scan_diagnostic_display_control_v1")
            self.assertEqual(artifact["output_origin"],
                             "unverified_supplied_arrays_may_include_patient_data")
            state_img = nib.load(str(work / "display/state.nii.gz"))
            state = np.asarray(state_img.dataobj)
            coverage = np.asarray(nib.load(str(work / "display/coverage.nii.gz")).dataobj)
            self.assertEqual(state_img.header.get_sform(coded=True)[1], 2)
            np.testing.assert_array_equal(state_img.affine, AFFINE)
            self.assertEqual(int(state[point_xyz]), 1)
            self.assertEqual(int(state[0, 0, 0]), -1)
            self.assertTrue(np.array_equal(state != -1, coverage != 0))

    def test_full_geometry_and_origin_binding_rejects_substitution(self):
        from dataclasses import replace
        from types import SimpleNamespace

        # A geometrically different prepared input with identical tensor bytes
        # must not borrow a saved input receipt. Use a real dataclass below so
        # the fingerprint exercises exactly the production serializer.
        from resectionlab.scan_preprocess_bridge import PatchGeometry
        geometry = PatchGeometry(
            (1, 1, 1), tuple(tuple(float(x) for x in row) for row in np.eye(4)),
            (1, 1, 1), ((0, 1), (0, 1), (0, 1)), (1, 1, 1), (1, 1, 1),
            (1., 1., 1.), (1., 1., 1.), (0, 0, 0),
            "p", "d", "t", "f", "s")
        prepared = SimpleNamespace(tensor_czyx=np.zeros((2, 128, 128, 128), np.float32),
                                   geometry=geometry)
        with tempfile.TemporaryDirectory(prefix="generated-geometry-binding-") as tmp:
            receipt = write_unverified_model_input_control(prepared, Path(tmp) / "handoff")
            tampered = SimpleNamespace(tensor_czyx=prepared.tensor_czyx,
                                       geometry=replace(geometry, crop_bbox_zyx=(
                                           (0, 1), (0, 1), (1, 2))))
            with self.assertRaises(DiagnosticHandoffError):
                _require_prepared_input(tampered, receipt, patient=False)
            forged = dict(receipt, schema="case4_unreviewed_diagnostic_input_v1",
                          source_origin="verified_case4_source_receipts")
            with self.assertRaises(DiagnosticHandoffError):
                _require_prepared_input(prepared, forged, patient=True)


if __name__ == "__main__":
    unittest.main()
