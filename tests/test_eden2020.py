"""Generated local NIfTI only; no EDEN files, network, models, or training."""
import hashlib

import nibabel as nib
import numpy as np
import pytest

from resectionlab.eden2020 import DOI, inspect_local_case


def file_record(path):
    return {"path": path.name, "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}


def image_record(root, name, shape=(4, 4, 4), affine=None, units="mm", data=None):
    affine = np.eye(4) if affine is None else affine
    image = nib.Nifti1Image(np.ones(shape, np.float32) if data is None else data, affine)
    image.header.set_xyzt_units(units)
    image.set_qform(affine, code=1)
    image.set_sform(affine, code=1)
    path = root / (name + ".nii.gz")
    nib.save(image, path)
    return file_record(path)


def fixture(root):
    permission = {"status": "granted", "doi": DOI, "intended_use": "research_planning",
                  "license_status": "reviewed_permitted", "evidence_sha256": "1" * 64, "subjects": ["generated-01"]}
    declaration = {"doi": DOI, "subject": "generated-01", "role": "TRAIN", "prospective_before_inspection": True,
                   "split_evidence_sha256": "2" * 64, "intended_use_review": "pass", "series": {}, "annotations": {}}
    for modality in ("T1_POST", "FLAIR", "TOF_MRA", "SWI", "DTI_32"):
        row = image_record(root, modality, shape=(4, 4, 4, 33) if modality == "DTI_32" else (4, 4, 4))
        row.update(subject="generated-01", metadata_review="pass", registration_review="pass", coverage_review="pass")
        if modality == "DTI_32":
            bval, bvec = root / "dti.bval", root / "dti.bvec"
            np.savetxt(bval, [[0] + [1000] * 32])
            vectors = np.zeros((3, 33)); vectors[0, 1:] = 1
            np.savetxt(bvec, vectors)
            row.update(bval=file_record(bval), bvec=file_record(bvec), gradient_frame="voxel")
        declaration["series"][modality] = row
    for kind in ("brain", "tumor"):
        declaration["annotations"][kind] = {**image_record(root, kind), "subject": "generated-01",
            "provenance": "supplied_annotation", "review_status": "pass"}
    return declaration, permission


def test_generated_intake_never_grants_planning_or_training(tmp_path):
    declaration, permission = fixture(tmp_path)
    result = inspect_local_case(tmp_path, declaration, permission=permission)
    assert result["intake_checks_passed"] and not result["planning_admitted"] and not result["training_admitted"]
    assert result["optimizer_updates_authorized"] == 0 and result["operative_action_labels"] is None
    assert result["series"]["DTI_32"]["gradients"]["weighted_volumes"] == 32
    declaration["role"] = "SELECT"
    assert inspect_local_case(tmp_path, declaration, permission=permission)["role"] == "SELECT"


@pytest.mark.parametrize("change", ["permission", "license", "subject", "eval", "prospective", "mixed_subject"])
def test_permission_and_role_rejection_before_any_local_inspection(tmp_path, monkeypatch, change):
    declaration, permission = fixture(tmp_path)
    if change == "permission": permission["status"] = "pending"
    if change == "license": permission["license_status"] = "unknown"
    if change == "subject": permission["subjects"] = []
    if change == "eval": declaration["role"] = "EVAL"
    if change == "prospective": declaration["prospective_before_inspection"] = False
    if change == "mixed_subject": declaration["series"]["SWI"]["subject"] = "someone-else"
    monkeypatch.setattr("resectionlab.eden2020._file", lambda *a: pytest.fail("source inspected"))
    with pytest.raises(ValueError): inspect_local_case(tmp_path / "absent", declaration, permission=permission)


@pytest.mark.parametrize("change", ["hash", "escape", "units", "qform", "nonfinite", "singular"])
def test_source_identity_and_geometry_fail_closed(tmp_path, change):
    declaration, permission = fixture(tmp_path)
    row = declaration["series"]["T1_POST"]
    if change == "hash": row["sha256"] = "0" * 64
    elif change == "escape": row["path"] = "../outside.nii.gz"
    elif change == "units": row.update(image_record(tmp_path, "bad_units", units="unknown"))
    elif change == "nonfinite": row.update(image_record(tmp_path, "bad_samples", data=np.full((4, 4, 4), np.nan)))
    elif change == "singular":
        # Missing coded geometry is rejected, including a disabled qform.
        image = nib.load(tmp_path / row["path"]); image.set_qform(None, code=0)
        nib.save(image, tmp_path / row["path"]); row.update(file_record(tmp_path / row["path"]))
    elif change == "qform":
        image = nib.load(tmp_path / row["path"]); affine = image.affine.copy(); affine[0, 3] = 10
        image.set_qform(affine, code=1); nib.save(image, tmp_path / row["path"])
        row.update(file_record(tmp_path / row["path"]))
    with pytest.raises(ValueError): inspect_local_case(tmp_path, declaration, permission=permission)


@pytest.mark.parametrize("change", ["dimensions", "nonfinite", "normalization", "baseline", "frame"])
def test_gradient_contract(tmp_path, change):
    declaration, permission = fixture(tmp_path)
    row = declaration["series"]["DTI_32"]
    if change == "frame": row.pop("gradient_frame")
    elif change == "baseline":
        np.savetxt(tmp_path / "dti.bval", [[1000] * 33])
        row["bval"] = file_record(tmp_path / "dti.bval")
    else:
        data = {"dimensions": np.ones((2, 33)), "nonfinite": np.full((3, 33), np.nan),
                "normalization": np.full((3, 33), 2.)}[change]
        np.savetxt(tmp_path / "dti.bvec", data)
        row["bvec"] = file_record(tmp_path / "dti.bvec")
    with pytest.raises(ValueError): inspect_local_case(tmp_path, declaration, permission=permission)


def test_missingness_and_reviews_block_qualification_without_filling_anatomy(tmp_path):
    declaration, permission = fixture(tmp_path)
    del declaration["series"]["TOF_MRA"]
    del declaration["annotations"]["tumor"]
    declaration["series"]["SWI"]["registration_review"] = "unreviewed"
    result = inspect_local_case(tmp_path, declaration, permission=permission)
    assert not result["intake_checks_passed"]
    assert "TOF_MRA" in result["missing_modalities"] and "tumor" not in result["annotations"]
    assert "complete_venous_topology" in result["unknown_anatomy"]
    assert any("registration_review" in reason for reason in result["blockers"])


def test_matching_dimensions_do_not_fuse_separate_affines_and_model_license_is_required(tmp_path):
    declaration, permission = fixture(tmp_path)
    affine = np.eye(4); affine[0, 3] = 5
    declaration["series"]["SWI"].update(image_record(tmp_path, "shifted", affine=affine))
    declaration["annotations"]["tumor"]["provenance"] = "model_estimate"
    result = inspect_local_case(tmp_path, declaration, permission=permission)
    assert not result["intake_checks_passed"]
    assert not result["series"]["SWI"]["same_grid_as_T1_POST"]
    assert any("model license" in reason for reason in result["blockers"])
    assert result["acquired_arrays_modified"] is False
