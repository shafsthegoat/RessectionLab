"""Independent auditor rejects source/frame/derivation defects on analytic grids."""
import importlib.util
from copy import deepcopy
from pathlib import Path

import nibabel as nib
import numpy as np
import pytest


SCRIPT = Path(__file__).resolve().parents[1] / "scripts/audit_brain_extraction.py"
SPEC = importlib.util.spec_from_file_location("audit_brain_extraction", SCRIPT)
audit = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(audit)


def save(path, data, affine=None):
    volume = nib.Nifti1Image(np.asarray(data, dtype=np.float32), np.eye(4) if affine is None else affine)
    volume.header.set_xyzt_units("mm")
    nib.save(volume, path)
    return volume


def test_independent_source_hash_check_detects_modified_input(tmp_path):
    path = tmp_path / "source"
    path.write_bytes(b"observed source")
    original = audit.digest(path)
    audit.verify_file(path, original)
    path.write_bytes(b"modified source")
    with pytest.raises(ValueError, match="Pinned file hash mismatch"):
        audit.verify_file(path, original)


@pytest.mark.parametrize("failure", ["left_right", "translation", "nonfinite", "fractional", "shape"])
def test_independent_native_artifact_validation(tmp_path, failure):
    shape = (8, 9, 10)
    reference = save(tmp_path / "t1.nii", np.indices(shape)[0])
    values = np.ones(shape)
    affine = np.eye(4)
    if failure == "left_right":
        affine[0, 0], affine[0, 3] = -1, shape[0] - 1
    elif failure == "translation":
        affine[1, 3] = 1
    elif failure == "nonfinite":
        values[2, 3, 4] = np.nan
    elif failure == "fractional":
        values[2, 3, 4] = 0.5
    else:
        values = values[:-1]
    path = tmp_path / "mask.nii"
    save(path, values, affine)
    with pytest.raises(ValueError):
        audit.read_native(path, reference, binary=True)


def test_independent_orientation_route_reindexes_source_by_physical_location(tmp_path):
    shape = (8, 9, 10)
    reference = save(tmp_path / "t1.nii", np.indices(shape)[0])
    original = np.zeros(shape)
    original[1, 3, 4] = 0.75
    affine = np.eye(4)
    affine[0, 0], affine[0, 3] = -1, shape[0] - 1
    path = tmp_path / "source.nii"
    save(path, original, affine)
    result = audit.source_annotation(path, reference, threshold=0.5)
    assert result[6, 3, 4] and result.sum() == 1
    affine[0, 3] += 0.1
    save(path, original, affine)
    with pytest.raises(ValueError, match="registration beyond an axis permutation"):
        audit.source_annotation(path, reference, threshold=0.5)


def test_independent_bookkeeping_reports_even_one_excluded_annotation_voxel():
    mask = np.zeros((12, 12, 12), dtype=bool)
    mask[2:10, 2:10, 2:10] = True
    distance = np.where(mask, -2.0, 8.0)
    tumor = mask.copy()
    tumor[1, 4, 4] = True
    report = audit.measure(mask, distance, tumor, np.diag([2., 3., 4., 1.]), border_mm=1)
    assert report["source_annotation_outside_voxels"] == 1
    assert report["flags"] == ["SOURCE_ANNOTATION_OUTSIDE_ESTIMATED_ENVELOPE"]
    assert report["volume_ml"] == pytest.approx(mask.sum() * 24 / 1000)
    assert report["reconstructed_mask_mismatch_voxels"] == 0
    assert report["outside_annotation_world_ras_mm_bbox"] == [[2., 12., 16.], [2., 12., 16.]]


def test_mask_cannot_disagree_with_declared_distance_derivation():
    mask = np.zeros((12, 12, 12), dtype=bool)
    mask[2:10, 2:10, 2:10] = True
    distance = np.where(mask, -2.0, 8.0)
    distance[2, 3, 3] = 5
    with pytest.raises(ValueError, match="differs from declared"):
        audit.measure(mask, distance, mask.copy(), np.eye(4), border_mm=1)
    with pytest.raises(ValueError, match="no positive mask support"):
        audit.measure(np.ones_like(mask), np.ones_like(distance) * 10,
                      mask.copy(), np.eye(4), border_mm=1)


def test_qform_sform_comparison_uses_accumulated_physical_corner_error(tmp_path):
    shape = (8, 9, 160)
    reference = save(tmp_path / "reference.nii", np.ones(shape))
    path = tmp_path / "mask.nii"
    volume = save(path, np.ones(shape))
    qform = np.eye(4)
    qform[2, 2] = 1.001
    volume.set_qform(qform, code=1)
    volume.set_sform(np.eye(4), code=1)
    nib.save(volume, path)
    # Every matrix entry is within the old 0.01 tolerance, but its last voxel is
    # displaced 0.159 mm between the two transforms.
    with pytest.raises(ValueError, match="qform/sform physical corner conflict"):
        audit.read_native(path, reference, binary=True)


@pytest.fixture
def frozen_record():
    config = {"device": "mps", "cpu_threads": 2, "border_mm": 1, "timeout_seconds": 300,
              "maximum_rss_bytes": 6442450944, "annotation_threshold": 0.5}
    model = {"model": "fixed synthetic audit fixture"}
    frozen = {"subject": "sub-PAT05", "implementation_sha256": "fixture",
              "declared_at": "2026-10-04T01:00:00+00:00", "changes_allowed_during_case_run": False,
              "brain_reviewed": False, "cortical_access_permitted": False, "variant_order": ["main"],
              "frozen_configuration": config, "models": {"main": model}, "cohort_role": "development"}
    report = {"implementation_sha256": "fixture", "created_at": "2026-10-04T01:01:00+00:00",
              "source_annotation": {"geometry_and_threshold": {"threshold": 0.5}},
              "variants": {"main": {"inference": {"configuration": deepcopy(config), "model": deepcopy(model)}}}}
    return frozen, report


@pytest.mark.parametrize("defect", ["subject", "implementation", "timestamp", "timezone", "review", "threshold", "model", "setting"])
def test_frozen_experiment_check_rejects_drift(frozen_record, defect):
    frozen, report = frozen_record
    audit.check_frozen_configuration(frozen, report, subject="sub-PAT05")
    if defect == "subject":
        frozen["subject"] = "sub-PAT28"
    elif defect == "implementation":
        report["implementation_sha256"] = "changed"
    elif defect == "timestamp":
        frozen["declared_at"] = "2026-10-04T01:02:00+00:00"
    elif defect == "timezone":
        frozen["declared_at"] = "2026-10-04T01:00:00"
    elif defect == "review":
        frozen["brain_reviewed"] = True
    elif defect == "threshold":
        report["source_annotation"]["geometry_and_threshold"]["threshold"] = 0.4
    elif defect == "model":
        report["variants"]["main"]["inference"]["model"] = {"model": "different"}
    else:
        report["variants"]["main"]["inference"]["configuration"]["border_mm"] = 2
    with pytest.raises(ValueError):
        audit.check_frozen_configuration(frozen, report, subject="sub-PAT05")
