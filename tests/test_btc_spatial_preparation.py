"""Four-case preparation keeps frozen source identities and development roles.

Only tiny synthetic image arrays are opened here. Real acquisition receipts are
read as metadata to check that production constants bind the completed batch.
"""
from copy import deepcopy
from datetime import datetime, timezone
from hashlib import md5, sha256
import importlib.util
import json
from pathlib import Path

import nibabel as nib
import numpy as np
import pytest

from resectionlab.imaging import load_case, read_case_artifacts


REPOSITORY = Path(__file__).resolve().parents[1]
SCRIPTS = REPOSITORY / "scripts"
spec = importlib.util.spec_from_file_location("spatial_preparation", SCRIPTS / "prepare_btc_case.py")
preparation = importlib.util.module_from_spec(spec)
spec.loader.exec_module(preparation)
SUBJECTS = ("sub-PAT22", "sub-PAT25", "sub-PAT26", "sub-PAT27")


@pytest.fixture
def acquisition(monkeypatch):
    monkeypatch.syspath_prepend(str(SCRIPTS))
    import acquire_btc_case
    return acquire_btc_case


@pytest.fixture(params=SUBJECTS)
def synthetic_source(request, tmp_path, monkeypatch, acquisition):
    subject = request.param
    root = tmp_path / "synthetic-sources"
    t1, annotation, _, _, names = preparation.source_layout(subject)
    for name in names:
        path = root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        if name.endswith(".json"):
            path.write_text("{}")
        elif name in {"README", "CHANGES"}:
            path.write_text("Tiny synthetic fixture; not patient data.")
    (root / "participants.tsv").write_text(
        "participant_id\t age\ttumor type & grade\ttumor size (cub cm)\ttumor location\thandedness\n"
        f"{subject}\t44\tSynthetic glioma\t11.49\tFrontal\t1\n"
    )
    shape = (24, 25, 26)
    flip = np.diag([-1., 1., 1., 1.])
    flip[0, 3] = shape[0] - 1
    fractional = np.zeros(shape, dtype=np.float32)
    fractional[2:4, 3:5, 4:6] = 0.8
    fractional[4:6, 3:5, 4:6] = 0.3
    for name, values, affine in ((t1, np.indices(shape)[0] + 1, np.eye(4)),
                                 (annotation, fractional, flip)):
        image = nib.Nifti1Image(np.asarray(values, dtype=np.float32), affine)
        image.header.set_xyzt_units("mm")
        image.set_qform(affine, 1)
        image.set_sform(affine, 1)
        nib.save(image, root / name)

    # Explicitly replace the frozen declaration with a synthetic-only source
    # fixture; source-binding validation remains enabled against that fixture.
    declaration = json.loads(acquisition.SPATIAL_MANIFEST.read_text())
    for entry in declaration["shared_release_files"]:
        payload = (root / entry["path"]).read_bytes()
        entry.update(bytes=len(payload), sha256=sha256(payload).hexdigest())
    candidate = next(x for x in declaration["candidates"] if x["subject"] == subject)
    for entry in candidate["files"]:
        payload = (root / entry["path"]).read_bytes()
        entry["expected_bytes"] = len(payload)
        if "expected_annex_md5" in entry:
            entry["expected_annex_md5"] = md5(payload, usedforsecurity=False).hexdigest()
        else:
            entry["sha256"] = sha256(payload).hexdigest()
    declaration_path = tmp_path / "synthetic-declaration.json"
    declaration_path.write_text(json.dumps(declaration))
    monkeypatch.setattr(acquisition, "SPATIAL_MANIFEST", declaration_path)
    monkeypatch.setattr(acquisition, "SPATIAL_SHA256", sha256(declaration_path.read_bytes()).hexdigest())
    manifest = acquisition.spatial_manifest(subject)
    for entry in manifest["files"]:
        entry["sha256"] = sha256((root / entry["path"]).read_bytes()).hexdigest()
    manifest_path = tmp_path / "synthetic-acquisition.json"
    manifest_path.write_text(json.dumps(manifest))
    monkeypatch.setitem(preparation.SPATIAL_SOURCE_MANIFEST_SHA256, subject, sha256(manifest_path.read_bytes()).hexdigest())
    return root, manifest_path, manifest


def test_production_manifest_pins_match_completed_acquisition_metadata(acquisition, tmp_path):
    assert tuple(preparation.SPATIAL_SOURCE_MANIFEST_SHA256) == SUBJECTS
    receipt_root = REPOSITORY / "artifacts/btc-spatial-acquisition-v1"
    batch = json.loads((receipt_root / "batch-record.json").read_text())
    attempts = {x["subject"]: x for x in batch["attempts"]}
    for subject, expected in preparation.SPATIAL_SOURCE_MANIFEST_SHA256.items():
        path = receipt_root / f"{subject}-source-manifest.json"
        assert sha256(path.read_bytes()).hexdigest() == expected == attempts[subject]["source_manifest_sha256"]
        assert len(acquisition.validate_manifest(json.loads(path.read_text()), tmp_path)) == 7


def test_preparation_preserves_role_source_annotation_and_closed_gates(synthetic_source, tmp_path, monkeypatch):
    root, manifest_path, manifest = synthetic_source
    original = {x["path"]: sha256((root / x["path"]).read_bytes()).hexdigest() for x in manifest["files"]}
    monkeypatch.setattr(preparation, "audit_diffusion", lambda *a, **k: pytest.fail("Unrequested DWI processing"))
    output = tmp_path / "prepared.ressectionlab"
    report = preparation.prepare(manifest_path, root, output)
    case = load_case(output)
    artifacts = read_case_artifacts(output)
    role = manifest["development_role"]
    assert report["benchmark_role"] == case.metadata["split"]["development_role"] == role
    assert case.metadata["split"]["policy_training_allowed"] is (manifest["subject"] in {"sub-PAT22", "sub-PAT25"})
    assert case.metadata["split"]["patient_group"] == manifest["patient_group"]
    assert case.metadata["split"]["role"] == "development"
    assert case.metadata["split"]["outer_role_locked"] is True
    assert case.metadata["split"]["external_holdout_eligible"] is False
    assert artifacts["preparation"]["split"] == report["split"]
    assert report["selection_cohort_sha256"] == case.metadata["selection_cohort_sha256"] == manifest["selection_cohort_sha256"]
    assert report["acquisition_manifest_sha256"] == sha256(manifest_path.read_bytes()).hexdigest()
    assert report["target_input_status"] == "annotation_assisted_research_input"
    assert report["target_policy_input_allowed"] is case.metadata["target_policy_input_allowed"] is False
    assert "hidden_environment_reference_only" in case.metadata["target_usage"]
    assert report["benchmark_track"] == case.metadata["benchmark_track"] == "annotation_assisted_threshold_scenario"
    assert case.metadata["annotation_review"] == "pending"
    assert report["source_files_unchanged"] and report["source_hashes_checked"] == 7 and report["reopened_identical"]
    assert case.brain_mask is None and case.context is None and artifacts["plans"] == []
    assert case.metadata["allow_nonzero_mri_access_support"] is False
    assert case.metadata["structural_coverage"] == "full_head"
    assert report["clinical_deficit_probability"] is None
    assert report["automatic_cortical_access_status"] == "blocked_without_reviewed_cerebral_mask"
    assert report["diffusion_input_audit"]["issues"] == ["MISSING_DIRECTIONAL_DIFFUSION"]
    assert all(x["value"] is None for x in report["withheld_context"].values())
    mask = case.compartments["fractional_source_target_threshold_scenario"]
    assert mask.sum() == 8 and mask[20, 3, 4] and not mask[2, 3, 4]
    assert np.array_equal(mask, case.source_compartments["fractional_source_target_threshold_scenario"])
    assert {name: sha256((root / name).read_bytes()).hexdigest() for name in original} == original


def test_spatial_replay_cutoff_keeps_unknown_timed_context_unavailable(synthetic_source, tmp_path):
    root, path, _ = synthetic_source
    output = tmp_path / "replay.ressectionlab"
    preparation.prepare(path, root, output, planning_as_of=datetime(2026, 10, 4, tzinfo=timezone.utc))
    case = load_case(output)
    assert case.context.planner_values() == {}
    assert case.context.planning_view()["source_diagnosis"]["value"] is None


def test_unapproved_threshold_fails_before_image_open(synthetic_source, tmp_path, monkeypatch):
    root, path, _ = synthetic_source
    monkeypatch.setattr(preparation, "load_fractional_annotation_case", lambda *a, **k: pytest.fail("Image opened"))
    output = tmp_path / "rejected.ressectionlab"
    with pytest.raises(ValueError, match="declared annotation threshold"):
        preparation.prepare(path, root, output, annotation_threshold=0.25)
    assert not output.exists()


def test_changed_receipt_fails_before_image_open(synthetic_source, tmp_path, monkeypatch):
    root, path, _ = synthetic_source
    path.write_bytes(path.read_bytes() + b" ")
    monkeypatch.setattr(preparation, "load_fractional_annotation_case", lambda *a, **k: pytest.fail("Image opened"))
    with pytest.raises(ValueError, match="frozen completed receipt"):
        preparation.prepare(path, root, tmp_path / "rejected.ressectionlab")


@pytest.mark.parametrize("kind", ["role", "group", "source", "cohort", "context", "anatomy"])
def test_forged_manifest_fails_even_with_updated_local_receipt(tmp_path, monkeypatch, acquisition, kind):
    manifest = acquisition.spatial_manifest("sub-PAT26")
    for entry in manifest["files"]:
        entry["sha256"] = entry["sha256"] or "a" * 64
    if kind == "role":
        manifest["development_role"] = "population_training"
    elif kind == "group":
        manifest["patient_group"] = "BTC:sub-PAT22"
    elif kind == "source":
        manifest["files"][-1]["source_url"] += "changed"
    elif kind == "cohort":
        manifest["selection_cohort_sha256"] = "0" * 64
    elif kind == "context":
        manifest["clinical_context"]["eligible_as_preoperative_policy_input"] = True
    else:
        manifest["anatomical_gates"]["cortical_access_permitted"] = True
    path = tmp_path / "forged.json"
    path.write_text(json.dumps(manifest))
    monkeypatch.setitem(preparation.SPATIAL_SOURCE_MANIFEST_SHA256, "sub-PAT26", sha256(path.read_bytes()).hexdigest())
    monkeypatch.setattr(preparation, "load_fractional_annotation_case", lambda *a, **k: pytest.fail("Image opened"))
    with pytest.raises(ValueError, match="Declared BTC source contract"):
        preparation.prepare(path, tmp_path / "absent", tmp_path / "rejected.ressectionlab")


def test_changed_source_bytes_fail_before_image_open(synthetic_source, tmp_path, monkeypatch):
    root, path, manifest = synthetic_source
    image_path = root / next(x["path"] for x in manifest["files"] if x["path"].endswith(".nii.gz"))
    payload = bytearray(image_path.read_bytes())
    payload[-1] ^= 1
    image_path.write_bytes(payload)
    monkeypatch.setattr(preparation, "load_fractional_annotation_case", lambda *a, **k: pytest.fail("Image opened"))
    with pytest.raises(ValueError, match="Source checksum/size mismatch"):
        preparation.prepare(path, root, tmp_path / "rejected.ressectionlab")


@pytest.mark.parametrize("subject", ["sub-PAT29", "sub-PAT31", "sub-PAT99", "../sub-PAT22"])
def test_sealed_or_arbitrary_subject_cannot_reach_image_loader(subject, tmp_path, monkeypatch, acquisition):
    manifest = deepcopy(acquisition.spatial_manifest("sub-PAT22"))
    manifest["subject"] = subject
    path = tmp_path / "forged.json"
    path.write_text(json.dumps(manifest))
    monkeypatch.setattr(preparation, "load_fractional_annotation_case", lambda *a, **k: pytest.fail("Image opened"))
    with pytest.raises(ValueError, match="reviewed development subject"):
        preparation.prepare(path, tmp_path / "absent", tmp_path / "rejected.ressectionlab")


def test_historical_subject_layouts_remain_unchanged():
    for subject, count in {"sub-PAT05": 7, "sub-PAT16": 7, "sub-PAT20": 7, "sub-PAT28": 15}.items():
        assert len(preparation.source_layout(subject)[-1]) == count
        assert preparation._development_split({"subject": subject}) == {
            "role": "development_demo", "patient_group": f"BTC:{subject}" if subject in {"sub-PAT16", "sub-PAT20"} else f"BTC-{subject}",
            "visit": "preop", "external_holdout_eligible": False,
        }
