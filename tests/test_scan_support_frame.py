"""Generated NIfTI frame controls only; no patient data or model calls."""

from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import importlib.util


_OPTIONAL_RUNTIME = ("ants", "nibabel", "SimpleITK", "nnunetv2", "torch")
_missing = [name for name in _OPTIONAL_RUNTIME if importlib.util.find_spec(name) is None]
if _missing:
    raise unittest.SkipTest("Scan frame guard requires optional runtime: " + ", ".join(_missing))

import nibabel as nib
import numpy as np


sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from resectionlab import scan_target_adapter as adapter  # noqa: E402


class SupportFrameTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.folder = Path(self.temp.name)
        self.affine = np.diag([-1.0, -1.0, 1.0, 1.0])
        self.affine[1, 3] = 4.0

    def volume(self, name, *, sform_code=2, qform_code=0,
               sform_affine=None, qform_affine=None):
        path = self.folder / f"{name}.nii.gz"
        image = nib.Nifti1Image(np.ones((5, 5, 5), np.uint8), self.affine)
        image.header.set_xyzt_units("mm")
        image.set_sform(self.affine if sform_affine is None else sform_affine, code=sform_code)
        image.set_qform(self.affine if qform_affine is None else qform_affine, code=qform_code)
        nib.save(image, str(path))
        return path, adapter.sha256_file(path)

    def test_coded_sform_accepts_unset_qform_on_same_grid(self):
        reference, reference_hash = self.volume("reference", sform_code=1, qform_code=2)
        support, support_hash = self.volume("support", sform_code=2, qform_code=0)
        checked = adapter.require_support_map_frame(support, support_hash,
                                                    reference, reference_hash)
        self.assertEqual(checked.shape_xyz, (5, 5, 5))
        np.testing.assert_allclose(checked.affine_ras_mm, self.affine)
        self.assertEqual(checked.sha256, support_hash)

    def test_qform_only_support_is_rejected(self):
        reference, reference_hash = self.volume("reference", sform_code=1, qform_code=2)
        support, support_hash = self.volume("support", sform_code=0, qform_code=2)
        with self.assertRaisesRegex(adapter.FrameMismatch, "coded sform"):
            adapter.require_support_map_frame(support, support_hash,
                                              reference, reference_hash)

    def test_shifted_sform_is_rejected_even_when_shape_matches(self):
        reference, reference_hash = self.volume("reference", sform_code=1, qform_code=2)
        shifted = self.affine.copy()
        shifted[1, 3] += 1.0
        support, support_hash = self.volume("support", sform_affine=shifted)
        with self.assertRaisesRegex(adapter.FrameMismatch, "different physical grids"):
            adapter.require_support_map_frame(support, support_hash,
                                              reference, reference_hash)

    def test_conflicting_coded_qform_is_rejected(self):
        reference, reference_hash = self.volume("reference", sform_code=1, qform_code=2)
        shifted = self.affine.copy()
        shifted[0, 3] += 1.0
        support, support_hash = self.volume("support", qform_code=2,
                                            qform_affine=shifted)
        with self.assertRaisesRegex(adapter.FrameMismatch, "conflicting coded"):
            adapter.require_support_map_frame(support, support_hash,
                                              reference, reference_hash)

    def test_substituted_hash_is_rejected_before_read(self):
        reference, reference_hash = self.volume("reference", sform_code=1, qform_code=2)
        support, _ = self.volume("support")
        with self.assertRaisesRegex(adapter.SourceMismatch, "SHA-256 mismatch"):
            adapter.require_support_map_frame(support, "0" * 64,
                                              reference, reference_hash)

    def test_changed_final_hash_is_rejected(self):
        reference, reference_hash = self.volume("reference", sform_code=1, qform_code=2)
        support, support_hash = self.volume("support")
        changed = adapter.PreparedVolume(support, (5, 5, 5), self.affine, "f" * 64)
        with patch.object(adapter, "read_volume", return_value=changed):
            with self.assertRaisesRegex(adapter.SourceMismatch, "changed during"):
                adapter.require_support_map_frame(support, support_hash,
                                                  reference, reference_hash)


if __name__ == "__main__":
    unittest.main()
