"""Safety checks for channel identity, common image geometry, and class mapping."""
import importlib.util
import json
from pathlib import Path

import nibabel as nib
import numpy as np
import pytest

SPEC = importlib.util.spec_from_file_location("poc_brats", Path(__file__).parents[1] / "scripts/poc_brats_inference.py")
brats = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(brats)


def image(affine=None):
    result = nib.Nifti1Image(np.ones((8, 8, 8), dtype=np.float32), np.eye(4) if affine is None else affine)
    result.set_qform(result.affine, 1)
    result.set_sform(result.affine, 1)
    return result


def test_geometry_rejects_equal_shape_with_different_physical_location():
    shifted = np.eye(4)
    shifted[0, 3] = 10
    with pytest.raises(ValueError, match="affine mismatch"):
        brats.verify_geometry([image(), image(shifted)])
    with pytest.raises(ValueError, match="1 mm"):
        brats.verify_geometry([image(np.diag([2, 2, 2, 1]))])


def test_upstream_class_priority_and_nested_binary_masks():
    channels = np.array([[False, True, False, True], [False, True, True, False],
                         [False, False, False, True]])
    labels = brats.labels_from_channels(channels)
    assert labels.tolist() == [0, 1, 2, 4]
    assert np.isin(labels, [1, 4]).tolist() == [False, True, False, True]


def test_inputs_follow_model_order_and_never_load_source_annotation(tmp_path):
    files = []
    for channel in reversed(brats.CHANNELS):
        path = tmp_path / f"{channel}.nii.gz"
        nib.save(image(), path)
        files.append({"modality": channel, "role": "structural_mri", "path": path.name,
                      "sha256": brats.sha256(path)})
    files.append({"modality": "tumor_segmentation", "role": "supplied_tumor_annotation",
                  "path": "does-not-exist.nii.gz", "sha256": "not-read"})
    manifest = tmp_path / "source.json"
    manifest.write_text(json.dumps({"case_id": "test-case", "spatial_frame": {
        "cross_modality_geometry_equal": True}, "files": files}))
    _, inputs = brats.prepare_inputs(manifest, tmp_path)
    assert [item["modality"] for item in inputs] == ["T1c", "T1", "T2", "FLAIR"]
    files[0]["sha256"] = "incorrect"
    manifest.write_text(json.dumps({"case_id": "test-case", "spatial_frame": {
        "cross_modality_geometry_equal": True}, "files": files}))
    with pytest.raises(ValueError, match="checksum mismatch"):
        brats.prepare_inputs(manifest, tmp_path)
