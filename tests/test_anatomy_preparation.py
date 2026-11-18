import json

import nibabel as nib
import numpy as np
import pytest

from resectionlab import anatomy_preparation as preparation
from resectionlab.anatomy_package import open_package, save_package
from resectionlab.patient_anatomy import PatientImagingCollection


def _study(tmp_path):
    image = tmp_path / "fixture.nii.gz"
    nib.save(nib.Nifti1Image(np.ones((5, 5, 5), np.uint8), np.eye(4)), image)
    study = PatientImagingCollection.create(tmp_path / "study", "fixture")
    study.add_series(image, series_id="s", modality="T1")
    study.save()
    return study


def _job(job_id, *, environment=1, depends_on=()):
    return dict(id=job_id, capability=job_id, model=job_id, environment=environment,
                series_id="s", options={}, status="ready", reason="", depends_on=list(depends_on))


def _layer(candidate, key):
    return candidate.add_layer(np.ones((5, 5, 5), np.uint8), np.eye(4), layer_id=key,
                               source_series="s", kind="brain", role="context", color="#ffffff")


def _output(row):
    return "cached", "fixture mask", [{"id": row["id"], "sha256": row["data"]["sha256"]}]


def test_prepare_resumes_after_cancellation_and_retries_failed_stage(tmp_path, monkeypatch):
    study = _study(tmp_path)
    versions = {"brain": 1, "optional": 1}

    def jobs(_study, _options):
        return [_job(key, environment=version) for key, version in versions.items()]

    calls = []

    def run(candidate, stage, _cancelled, _progress):
        key = stage["id"]
        calls.append(key)
        if key == "optional":
            raise RuntimeError("fixture worker failed")
        return _output(_layer(candidate, "brain"))

    monkeypatch.setattr(preparation, "_jobs", jobs)
    monkeypatch.setattr(preparation, "_run_stage", run)
    done = []
    with pytest.raises(InterruptedError):
        preparation.prepare_anatomy(study, on_stage=done.append, cancelled=lambda: bool(done))
    assert calls == ["brain"] and study.layer("brain")["review_status"] == "unreviewed"
    assert done[0]["status"] == "complete"
    result = preparation.prepare_anatomy(study)
    assert [s["status"] for s in result["outcomes"]] == ["cached", "failed"]
    assert calls == ["brain", "optional"]
    archive = tmp_path / "study.rsworkspace"
    save_package(study, archive)
    copy = PatientImagingCollection(open_package(archive, tmp_path / "copies").parent)
    warm = preparation.prepare_anatomy(copy)
    assert calls == ["brain", "optional"] and warm["document"]["snapshot_id"] == study.identity()
    prior = warm["plan"]["id"]
    versions["optional"] = 2
    with pytest.raises(ValueError, match="inputs changed"):
        preparation.prepare_anatomy(copy, plan_id=prior)
    preparation.prepare_anatomy(copy)
    assert calls == ["brain", "optional", "optional"]
    preparation.prepare_anatomy(copy, retry_failed=True)
    assert calls == ["brain", "optional", "optional", "optional"]


def test_interrupted_stage_remains_resumable_without_admitting_partial_layer(tmp_path, monkeypatch):
    study = _study(tmp_path)
    monkeypatch.setattr(preparation, "_jobs", lambda *_: [_job("brain"), _job("next")])
    calls = []

    def run(candidate, stage, _cancelled, _progress):
        key = stage["id"]
        calls.append(key)
        row = _layer(candidate, key)
        if calls == ["brain", "next"]:
            raise InterruptedError("fixture worker interrupted")
        return _output(row)

    monkeypatch.setattr(preparation, "_run_stage", run)
    with pytest.raises(InterruptedError, match="completed stages"):
        preparation.prepare_anatomy(study)
    reopened = PatientImagingCollection(study.root)
    assert [row["id"] for row in reopened.document["layers"]] == ["brain"]
    plan = preparation.preparation_plan(reopened)
    assert [s["status"] for s in plan["stages"]] == ["cached", "interrupted"]
    resumed = preparation.prepare_anatomy(reopened, plan_id=plan["id"])
    assert calls == ["brain", "next", "next"]
    assert [s["status"] for s in resumed["outcomes"]] == ["cached", "complete"]
    assert [s["status"] for s in resumed["plan"]["stages"]] == ["cached", "cached"]


def test_optional_adapter_failure_preserves_independent_stages_and_blocks_dependents(tmp_path, monkeypatch):
    study = _study(tmp_path)
    monkeypatch.setattr(preparation, "_jobs", lambda *_: [
        _job("parent"), _job("dependent", depends_on=["parent"]), _job("independent"),
    ])
    calls = []

    def run(candidate, stage, _cancelled, _progress):
        key = stage["id"]
        calls.append(key)
        row = _layer(candidate, key)
        if calls == ["parent"]:
            raise nib.filebasedimages.ImageFileError("fixture model output unreadable")
        return _output(row)

    monkeypatch.setattr(preparation, "_run_stage", run)
    first = preparation.prepare_anatomy(study)
    assert [s["status"] for s in first["outcomes"]] == ["failed", "blocked", "complete"]
    assert [s["status"] for s in first["plan"]["stages"]] == ["failed", "blocked", "cached"]
    assert "ImageFileError" in first["outcomes"][0]["reason"]
    assert [layer["id"] for layer in study.document["layers"]] == ["independent"]
    retried = preparation.prepare_anatomy(study, plan_id=first["plan"]["id"], retry_failed=True)
    assert calls == ["parent", "independent", "parent", "dependent"]
    assert [s["status"] for s in retried["outcomes"]] == ["complete", "complete", "cached"]


@pytest.mark.parametrize("change", [
    {"inputKey": "bad-checksum"},
    {"status": "complete"},
    {"reason": None},
    {"elapsedSeconds": float("nan")},
    {"outputs": {"id": "brain", "sha256": "a" * 64}},
    {"outputs": [{"id": "brain", "sha256": "a" * 64}] * 2},
])
def test_malformed_preparation_checkpoint_is_rejected_before_planning(tmp_path, change):
    study = _study(tmp_path)
    receipt = {"status": "cached", "reason": "completed", "inputKey": "b" * 64,
               "outputs": [{"id": "brain", "sha256": "a" * 64}], "elapsedSeconds": 1.0, **change}
    study.path(preparation.RECEIPTS).write_text(json.dumps({
        "schema": preparation.SCHEMA, "patient": "fixture", "stages": {"brain": receipt},
    }))
    with pytest.raises(ValueError, match="preparation checkpoint"):
        preparation.preparation_plan(study)


def test_empty_stage_requires_explicit_retry(tmp_path, monkeypatch):
    study = _study(tmp_path)
    monkeypatch.setattr(preparation, "_jobs", lambda *_: [_job("brain")])
    calls = []

    def run(candidate, stage, _cancelled, _progress):
        calls.append(stage["id"])
        return _output(_layer(candidate, "brain")) if len(calls) > 1 else ("empty_output", "No supported label", [])

    monkeypatch.setattr(preparation, "_run_stage", run)
    assert preparation.prepare_anatomy(study)["outcomes"][0]["status"] == "empty_output"
    resumed = PatientImagingCollection(study.root)
    assert preparation.prepare_anatomy(resumed)["outcomes"][0]["status"] == "empty_output"
    assert calls == ["brain"]
    assert preparation.prepare_anatomy(resumed, retry_failed=True)["outcomes"][0]["status"] == "complete"
    assert calls == ["brain", "brain"]
