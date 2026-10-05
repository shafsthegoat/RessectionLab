"""Import and persistence never promote estimated structure into working anatomy."""
from dataclasses import replace
from datetime import datetime, timezone
import json
from pathlib import Path
import subprocess
import sys
from zipfile import ZipFile

import nibabel as nib
import numpy as np
import pytest

from resectionlab.brain_extraction import ASSETS
from resectionlab.imaging import (
    ImagingError, file_sha256, import_brain_extraction_evidence, load_case,
    load_nifti_case, save_case,
)
from resectionlab.structural_evidence import BrainEnvelopeReview


def write(path, values, affine=None):
    image = nib.Nifti1Image(np.asarray(values, np.float32), np.eye(4) if affine is None else affine)
    image.header.set_xyzt_units("mm")
    nib.save(image, path)
    return path


@pytest.fixture
def inputs(tmp_path):
    shape = (8, 9, 10)
    source = write(tmp_path / "t1.nii.gz", np.indices(shape)[0])
    target = np.zeros(shape)
    target[2:4, 3:5, 4:6] = 1
    target_path = write(tmp_path / "target.nii.gz", target)
    mask = np.ones(shape)
    mask[2, 3, 4] = 0
    mask_path = write(tmp_path / "main_mask.nii.gz", mask)
    report_path = tmp_path / "brain_extraction_report.json"
    report = {
        "source_t1_sha256": file_sha256(source),
        "variants": {"main": {"inference": {
            "input_sha256": file_sha256(source), "failure": None, "exit_code": 0,
            "brain_reviewed": False, "cortical_access_permitted": False,
            "model": {"name": "SynthStrip", "variant": "main",
                      "files": {"synthstrip.1.pt": {"sha256": ASSETS["synthstrip.1.pt"][1]}}},
            "artifact_hashes": {"main_mask.nii.gz": file_sha256(mask_path)},
            "configuration": {"border_mm": 1, "device": "synthetic_test_fixture"},
        }, "qc": {"flags": []}}},
    }
    report_path.write_text(json.dumps(report))
    case = load_nifti_case(source, target_path, metadata={"structural_coverage": "full_head",
                                                        "allow_nonzero_mri_access_support": False})
    return case, {"source_image_path": source, "mask_path": mask_path,
                  "report_path": report_path, "variant": "main"}, report


def test_extraction_import_is_display_only_and_source_bound(inputs):
    case, kwargs, _ = inputs
    updated = import_brain_extraction_evidence(case, **kwargs)
    assert updated.brain_mask is None
    assert updated.planning_hash == case.planning_hash
    assert updated.semantic_hash != case.semantic_hash
    assert updated.revision == case.revision + 1
    assert not case.structural_evidence
    evidence = next(iter(updated.structural_evidence.values()))
    assert evidence.review_status == "review_required"
    assert evidence.cortical_access_permitted is False
    assert evidence.metadata["current_target_annotation_outside_voxels"] == 1
    assert "CURRENT_TARGET_ANNOTATION_OUTSIDE_ESTIMATED_ENVELOPE" in evidence.metadata["qc_flags"]
    assert updated.metadata["allow_nonzero_mri_access_support"] is False
    assert import_brain_extraction_evidence(updated, **kwargs) is updated
    with pytest.raises(ValueError):
        evidence.mask.setflags(write=True)


def test_current_case_image_or_frame_change_rejects_stale_extraction(inputs):
    case, kwargs, _ = inputs
    changed_mri = case.mri.copy()
    changed_mri[1, 1, 1] += 1
    with pytest.raises(ImagingError, match="EXTRACTION_CASE_IMAGE_CHANGED"):
        import_brain_extraction_evidence(case.revised(mri=changed_mri), **kwargs)
    changed_affine = case.affine.copy()
    changed_affine[0, 3] = 1
    with pytest.raises(ImagingError, match="EXTRACTION_CASE_IMAGE_CHANGED"):
        import_brain_extraction_evidence(case.revised(affine=changed_affine), **kwargs)


@pytest.mark.parametrize("defect", ["source", "model", "mask_bytes", "review", "failed", "wrong_grid"])
def test_unverified_report_or_mask_cannot_be_attached(inputs, defect):
    case, kwargs, report = inputs
    inference = report["variants"]["main"]["inference"]
    if defect == "source":
        report["source_t1_sha256"] = "0" * 64
    elif defect == "model":
        inference["model"]["files"]["synthstrip.1.pt"]["sha256"] = "0" * 64
    elif defect == "mask_bytes":
        inference["artifact_hashes"]["main_mask.nii.gz"] = "0" * 64
    elif defect == "review":
        inference["brain_reviewed"] = True
    elif defect == "failed":
        inference["failure"] = "EXTRACTION_PROCESS_FAILED"
    else:
        affine = np.eye(4)
        affine[0, 0] = -1
        write(kwargs["mask_path"], np.ones(case.mri.shape), affine)
        inference["artifact_hashes"]["main_mask.nii.gz"] = file_sha256(kwargs["mask_path"])
    kwargs["report_path"].write_text(json.dumps(report))
    with pytest.raises(ImagingError):
        import_brain_extraction_evidence(case, **kwargs)
    assert case.brain_mask is None and not case.structural_evidence


def test_bundle_preserves_unreviewed_evidence_and_hash_bound_review(inputs, tmp_path):
    case, kwargs, _ = inputs
    proposed = import_brain_extraction_evidence(case, **kwargs)
    path = save_case(proposed, tmp_path / "evidence.rslcase")
    reopened = load_case(path)
    assert reopened.semantic_hash == proposed.semantic_hash
    assert reopened.planning_hash == case.planning_hash
    evidence = next(iter(reopened.structural_evidence.values()))
    assert evidence.review_status == "review_required" and reopened.brain_mask is None
    review = BrainEnvelopeReview(evidence.evidence_hash, "synthetic-test-reviewer",
                                  datetime(2026, 10, 4, tzinfo=timezone.utc), "accepted",
                                  "Synthetic persistence test only; no patient review occurred")
    reviewed_evidence = replace(evidence, review=review)
    reviewed = reopened.revised(structural_evidence={evidence.evidence_id: reviewed_evidence})
    restored = load_case(save_case(reviewed, tmp_path / "reviewed-fixture.rslcase"))
    assert restored.structural_evidence[evidence.evidence_id].review_status == "accepted"
    assert restored.structural_evidence[evidence.evidence_id].cortical_access_permitted is False
    assert restored.brain_mask is None
    assert restored.planning_hash == case.planning_hash


def test_bundle_rejects_forged_review_status(inputs, tmp_path):
    case, kwargs, _ = inputs
    path = save_case(import_brain_extraction_evidence(case, **kwargs), tmp_path / "forged.rslcase")
    with ZipFile(path) as archive:
        manifest = json.loads(archive.read("manifest.json"))
        arrays = archive.read("arrays.npz")
    next(iter(manifest["structural_evidence"].values()))["manifest"]["review_status"] = "accepted"
    with ZipFile(path, "w") as archive:
        archive.writestr("manifest.json", json.dumps(manifest))
        archive.writestr("arrays.npz", arrays)
    with pytest.raises(ImagingError, match="INVALID_BUNDLE_DATA"):
        load_case(path)


def test_attach_cli_saves_separate_reopenable_proposals(inputs, tmp_path):
    case, kwargs, _ = inputs
    original = save_case(case, tmp_path / "original.rslcase", artifacts={"settings": {"opacity": 0.4}})
    source_hash = file_sha256(kwargs["source_image_path"])
    script = Path(__file__).resolve().parents[1] / "scripts/attach_brain_evidence.py"
    output = tmp_path / "exports" / "with_evidence.rslcase"
    result = subprocess.run([sys.executable, str(script), "--case", str(original),
                             "--source-image", str(kwargs["source_image_path"]),
                             "--report", str(kwargs["report_path"]), "--variant", "main",
                             "--output", str(output)], text=True, capture_output=True, check=True)
    assert json.loads(result.stdout)["working_anatomy_unchanged"]
    assert load_case(output).brain_mask is None
    assert load_case(original).structural_evidence == {}
    assert file_sha256(kwargs["source_image_path"]) == source_hash


@pytest.mark.parametrize("alias", ["case", "source", "report", "mask"])
def test_attach_cli_rejects_overwriting_known_source_artifacts(inputs, tmp_path, alias):
    case, kwargs, _ = inputs
    original = save_case(case, tmp_path / "original.rslcase")
    protected = {"case": original, "source": kwargs["source_image_path"],
                 "report": kwargs["report_path"], "mask": kwargs["mask_path"]}
    before = {name: path.read_bytes() for name, path in protected.items()}
    script = Path(__file__).resolve().parents[1] / "scripts/attach_brain_evidence.py"
    result = subprocess.run([sys.executable, str(script), "--case", str(original),
                             "--source-image", str(kwargs["source_image_path"]),
                             "--report", str(kwargs["report_path"]), "--output", str(protected[alias])],
                            text=True, capture_output=True)
    assert result.returncode == 2
    assert all(path.read_bytes() == before[name] for name, path in protected.items())
