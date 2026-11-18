import nibabel as nib
import numpy as np
import pytest

from resectionlab import anatomy_preparation as preparation
from resectionlab.anatomy_package import open_package, save_package
from resectionlab.patient_anatomy import PatientImagingCollection


def test_prepare_resumes_after_cancellation_and_retries_failed_stage(tmp_path, monkeypatch):
    image = tmp_path / "fixture.nii.gz"
    nib.save(nib.Nifti1Image(np.ones((5, 5, 5), np.uint8), np.eye(4)), image)
    study = PatientImagingCollection.create(tmp_path / "study", "fixture")
    study.add_series(image, series_id="s", modality="T1")
    study.save()
    versions = {"brain": 1, "optional": 1}

    def jobs(_study, _options):
        return [dict(id=key, capability=key, model=key, environment=version,
                     series_id="s", options={}, status="ready", reason="")
                for key, version in versions.items()]

    calls = []

    def run(candidate, stage, _cancelled, _progress):
        key = stage["id"]
        calls.append(key)
        if key == "optional":
            raise RuntimeError("fixture worker failed")
        row = candidate.add_layer(np.ones((5, 5, 5), np.uint8), np.eye(4), layer_id="brain",
                                  source_series="s", kind="brain", role="context", color="#ffffff")
        return "cached", "fixture mask", [{"id": row["id"], "sha256": row["data"]["sha256"]}]

    monkeypatch.setattr(preparation, "_jobs", jobs)
    monkeypatch.setattr(preparation, "_run_stage", run)
    done = []
    with pytest.raises(InterruptedError):
        preparation.prepare_anatomy(study, on_stage=done.append, cancelled=lambda: bool(done))
    assert calls == ["brain"] and study.layer("brain")["review_status"] == "unreviewed"
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
